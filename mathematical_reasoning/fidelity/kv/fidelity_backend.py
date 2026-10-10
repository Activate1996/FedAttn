#!/usr/bin/env python3
"""Measure the global KV states available to each participant at every prefill layer.

The backend uses the FedAttn mask builders and records model activations during
prefill. Random masks are generated from the configured seeds for each sample.
Use run_fig4_fidelity.py to run the configured measurements.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import random
import sys
from pathlib import Path

from deviation_io import load_docs, resolve_sample_plan, point_metadata, planned_points, read_completed

from kv_metrics import METRIC_ID
from kv_observer import KVObserver


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def chosen_indices(total, count):
    count = min(total, count)
    if count < 1:
        raise ValueError("Need at least one sample.")
    return [i * total // count for i in range(count)]


def sample_seed(seed, sample_id):
    return int(digest([seed, sample_id])[:8], 16)


def load_original(repo):
    root = Path(repo).resolve()
    fed = root / "FedAttn" if (root / "FedAttn").is_dir() else root
    sys.path.insert(0, str(fed))
    sys.path.insert(0, str(fed / "evaluate_gsm8k"))
    spec = importlib.util.spec_from_file_location("_fedattn_original_evaluate", fed / "evaluate_gsm8k" / "main_evaluate_gsm8k.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, fed


class Captured(Exception):
    pass


def capture_masks(original, wrapper, tokenizer, model, device, prompt, prompt_lengths,
                  chunks, hp, split_way, policy, seed):
    """Capture masks from the FedAttn builders before token generation begins."""
    captured = {}
    old_generate = wrapper.generate
    replacements = {}

    def stop(input_ids, **kw):
        captured.update(kw)
        captured["input_ids"] = input_ids
        raise Captured()

    real_build = original.build_prefill_and_decode_mask

    def record_build(*args, **kwargs):
        captured["chunks"] = [[int(x) for x in ch] for ch in kwargs["tok_chunks"][0]]
        return real_build(*args, **kwargs)

    def wrap_decode(fn):
        def call(*args, **kwargs):
            result = fn(*args, **kwargs)
            captured["all_decode_masks"] = result
            return result
        return call

    old_random = random.getstate()
    try:
        random.seed(seed)
        wrapper.generate = stop
        original.build_prefill_and_decode_mask = record_build
        for key in ("build_decode_masks_main", "build_decode_masks_sample",
                    "build_decode_masks_customized", "build_decode_masks_unipolicy"):
            replacements[key] = getattr(original, key)
            setattr(original, key, wrap_decode(replacements[key]))
        comp = [hp.get("ratio_comp", 1.0)] * (hp["num_clients"] - 1)
        comp += [hp.get("ratio_last_comp", hp.get("ratio_comp", 1.0))]
        comm = [hp.get("ratio_comm", 1.0)] * (hp["num_clients"] - 1)
        comm += [hp.get("ratio_last_comm", hp.get("ratio_comm", 1.0))]
        original.generate_responses(
            model=model, dtype=model.dtype, wrapper=wrapper, device=device,
            tokenizer=tokenizer, tokenizer_padding_side=tokenizer.padding_side,
            input_txt=[prompt], prompt_token_nums=prompt_lengths,
            loaded_chunks_batch=None if chunks is None else [chunks], hyperparams=hp,
            split_way=split_way, comm_policy=policy, ratios_comp=comp, ratios_comm=comm)
    except Captured:
        pass
    finally:
        wrapper.generate = old_generate
        original.build_prefill_and_decode_mask = real_build
        for key, fn in replacements.items():
            setattr(original, key, fn)
        random.setstate(old_random)
    if "all_decode_masks" not in captured:
        raise RuntimeError("Original builder was not intercepted; inspect source compatibility.")
    captured["decode_layer_masks"] = captured["all_decode_masks"][-1]
    # These tables specify noncausal input-token availability at each layer.
    # Reading them neither runs decoding nor changes prefill attention.
    captured["kv_availability_masks"] = captured["all_decode_masks"]
    return captured


def validate_chunks(captured):
    length = int(captured["padding_mask"][0].sum())
    flattened = [x for group in captured["chunks"] for x in group]
    if sorted(flattened) != list(range(length)):
        raise ValueError("Token chunks must partition every valid input token exactly once.")
    if not captured["chunks"][-1] or max(captured["chunks"][-1]) != length - 1:
        raise ValueError("Publisher (last client) must own the final prompt token.")


def mask_hash(captured):
    h = hashlib.sha256()
    cache = {}
    for mask in captured["prefill_layer_masks"]:
        key = id(mask)
        if key not in cache:
            cache[key] = hashlib.sha256((mask == 0).detach().cpu().numpy().tobytes()).digest()
        h.update(cache[key])
    return h.hexdigest()


def restore_layer(torch, captured, layer, hp, policy):
    if policy != "sample_uni_comm" or any(hp.get(k, 1.0) != 1.0 for k in ("ratio_comp", "ratio_last_comp")):
        raise ValueError("restore_remote_layer requires sample_uni_comm with full local computation.")
    if layer < 0 or layer >= len(captured["prefill_layer_masks"]) or layer % hp["num_local_forward"]:
        raise ValueError("restore_remote_layer must be an existing synchronization layer.")
    ids = captured["input_ids"]
    valid = captured["padding_mask"].bool()
    size = ids.shape[1]
    full = captured["prefill_layer_masks"][layer].clone()
    keys = torch.arange(size, device=ids.device)
    visible = (keys[None, :] <= keys[:, None]) & valid[0][None, :]
    full.fill_(torch.finfo(full.dtype).min)
    full[0, 0].masked_fill_(visible, 0)
    # Preserve padding-query behaviour from the captured mask for this single sample.
    full[0, 0, ~valid[0], :] = captured["prefill_layer_masks"][layer][0, 0, ~valid[0], :]
    captured["prefill_layer_masks"][layer] = full
    for client_masks in captured["all_decode_masks"]:
        restored = client_masks[layer].clone()
        restored.fill_(torch.finfo(restored.dtype).min)
        restored.masked_fill_(valid[:, None, None, :], 0)
        client_masks[layer] = restored
    captured["decode_layer_masks"] = captured["all_decode_masks"][-1]


def align_prefill_masks(torch, model, captured):
    """Match additive masks to model precision, retaining shared layer masks."""
    converted = {}
    for mask in captured["prefill_layer_masks"]:
        if id(mask) not in converted:
            target_dtype = torch.bool if mask.dtype == torch.bool else model.dtype
            converted[id(mask)] = mask.to(
                device=captured["input_ids"].device, dtype=target_dtype)
    captured["prefill_layer_masks"] = [
        converted[id(mask)] for mask in captured["prefill_layer_masks"]]


def run_prefill(torch, model, wrapper, captured, mechanism_layers, query_count, reference=None):
    """Run masked prefill and observe each layer's keys after RoPE and its values."""
    if mechanism_layers:
        raise ValueError("This toolkit measures KV availability at every layer")
    align_prefill_masks(torch, model, captured)
    try:
        wrapper.batch_size, wrapper.input_len = captured["input_ids"].shape
        wrapper.current_token_counts = captured["padding_mask"].sum(1)
        wrapper.is_prefill, wrapper.layer_ptr = True, 0
        wrapper.set_masks(prefill_masks=captured["prefill_layer_masks"], decode_masks=None)
        with torch.inference_mode(), KVObserver(torch, model, captured, reference) as observer:
            model.model(input_ids=captured["input_ids"], attention_mask=captured["padding_mask"],
                        position_ids=wrapper.create_position_ids(captured["padding_mask"], captured["padding_side"]),
                        use_cache=False, output_hidden_states=False, return_dict=True)
            result = observer.result()
    finally:
        wrapper.reset()
    return result


def metrics(torch, current, reference, captured):
    return {key: current[key] for key in (
        "kv_layers", "layer_count", "layer_participant_mean_relative_kv_frobenius", "participant_layer_means")}


def create_runtime(manifest):
    """Load one model for reuse across KV fidelity segmentation settings."""
    import torch
    import transformers
    from server_resources import resolve_checkpoint_path
    original, fed = load_original(manifest["source_repo"])
    device = manifest.get("device", "cuda:0")
    dtype_name = manifest.get("dtype", "float32")
    if dtype_name not in ("float32", "bfloat16", "float16"):
        raise ValueError("dtype must be float32, bfloat16 or float16")
    checkpoint_path = resolve_checkpoint_path(manifest["model_path"])
    print(f"Resolved {manifest.get('model', '')}: {checkpoint_path}", flush=True)
    tokenizer = transformers.AutoTokenizer.from_pretrained(checkpoint_path, local_files_only=True)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        checkpoint_path, torch_dtype=getattr(torch, dtype_name),
        attn_implementation="sdpa", local_files_only=True).to(device).eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    wrapper = original.LayerMaskedTransformerWrapper(model, device)
    if not all(isinstance(layer.self_attn, wrapper.attention_class) for layer in model.model.layers):
        wrapper.attention_class.forward = wrapper._orig_attn_fwd
        raise RuntimeError("Wrapper patches a different attention class; use the original transformers version")
    if manifest.get("num_layers", wrapper.n_layers) != wrapper.n_layers:
        wrapper.attention_class.forward = wrapper._orig_attn_fwd
        raise ValueError("Model depth does not match the panel model label")
    checkpoint_files = []
    for pattern in ("*.safetensors", "*.bin", "config.json", "tokenizer*", "vocab*", "merges*"):
        for path in sorted(Path(checkpoint_path).glob(pattern)):
            if path.is_file():
                stat = path.stat()
                checkpoint_files.append([path.name, stat.st_size, stat.st_mtime_ns])
    return dict(torch=torch, transformers=transformers, original=original, fed=fed,
                model=model, tokenizer=tokenizer, wrapper=wrapper, device=device,
                model_path=str(Path(checkpoint_path).resolve()), model_reference=manifest["model_path"],
                source_repo=str(Path(manifest["source_repo"]).resolve()), dtype_name=dtype_name,
                checkpoint_files=checkpoint_files)


def close_runtime(runtime):
    import gc
    wrapper = runtime.get("wrapper")
    if wrapper is not None:
        wrapper.reset()
        wrapper.attention_class.forward = wrapper._orig_attn_fwd
    torch = runtime.get("torch")
    runtime.clear()
    del wrapper
    gc.collect()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_manifest(manifest_path, generate=False, resume=False, runtime=None):
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if generate:
        raise ValueError("This runner measures prefill representations only")
    if manifest.get("metric_id") != METRIC_ID:
        raise ValueError("Use the new Fig4 KV-fidelity manifest and a fresh output root")
    docs = load_docs(manifest["data_jsonl"])
    sample_plan = resolve_sample_plan(docs, manifest)
    indices = sample_plan["sample_indices"]
    names = [c["name"] for c in manifest["configs"]]
    if len(set(names)) != len(names) or "CenAttn" in names:
        raise ValueError("Unique config names required; CenAttn is reserved")
    common = dict(manifest["hyperparams"])
    for key in ("num_shots", "num_clients", "num_local_forward", "max_new_tokens", "do_sample"):
        if key not in common:
            raise ValueError(f"Missing historical hyperparameter: {key}")
    configs = [{"name": "CenAttn", "comm_policy": "main", "num_local_forward": 1}] + manifest["configs"]
    points = planned_points(manifest, configs)
    own_runtime = runtime is None
    if own_runtime:
        runtime = create_runtime(manifest)
    try:
        if str(Path(manifest["source_repo"]).resolve()) != runtime["source_repo"]:
            raise ValueError("Cannot reuse a runtime with different source_repo")
        if runtime.get("model_reference", runtime["model_path"]) != manifest["model_path"]:
            raise ValueError("Cannot reuse a runtime with different model reference")
        if manifest.get("dtype", "float32") != runtime["dtype_name"] or manifest.get("device", "cuda:0") != runtime["device"]:
            raise ValueError("Cannot reuse a runtime with different dtype/device")
        torch, transformers = runtime["torch"], runtime["transformers"]
        original, fed = runtime["original"], runtime["fed"]
        tokenizer, model, wrapper = runtime["tokenizer"], runtime["model"], runtime["wrapper"]
        device = runtime["device"]
        layers = manifest.get("mechanism_layers", [])
        if any(type(layer) is not int or layer < 0 or layer >= wrapper.n_layers for layer in layers):
            raise ValueError("mechanism_layers contains an invalid zero-based layer")
        query_count = manifest.get("mechanism_query_count", 1)
        if type(query_count) is not int or query_count < 1:
            raise ValueError("mechanism_query_count must be positive")
        fewshot, lengths = original.load_prompts(list(range(common["num_shots"])), tokenizer, True,
            prompt_prefix=manifest.get("prompt_prefix", "gsm8k_prompt_"),
            prompt_suffix=manifest.get("prompt_suffix", ".txt"))
        source_hashes = {str(path.relative_to(fed)): file_hash(path) for path in
                        [fed / "utils.py", fed / "Wrapper.py", fed / "evaluate_gsm8k" / "main_evaluate_gsm8k.py"]}
        chunk_paths = set()
        if manifest.get("token_chunks_json"):
            chunk_paths.add(manifest["token_chunks_json"])
        for config in configs:
            if config.get("token_chunks_json"):
                chunk_paths.add(config["token_chunks_json"])
        chunk_hashes = {str(Path(path).resolve()): file_hash(path) for path in sorted(chunk_paths)}
        provenance = {
            "schema_version": 4, "metric_id": METRIC_ID,
            **sample_plan, "manifest_sha256": file_hash(manifest_path),
            "data_sha256": file_hash(manifest["data_jsonl"]), "source_hashes": source_hashes,
            "probe_script_sha256": file_hash(__file__),
            "measurement_source_hashes": {name: file_hash(Path(__file__).with_name(name))
                                          for name in ("kv_metrics.py", "kv_observer.py")},
            "fewshot_prompt_sha256": digest(fewshot),
            "model_path": runtime["model_path"], "checkpoint_files": runtime["checkpoint_files"],
            "torch_version": torch.__version__, "transformers_version": transformers.__version__,
            "dtype": str(model.dtype), "device": str(device), "planned_points": points,
            "generated_utility": False,
            "token_chunks_file_hashes": chunk_hashes,
            "figure": manifest.get("figure"), "model": manifest.get("model", Path(manifest["model_path"]).name),
            "split_way": manifest["split_way"], "x_axis": points[0].get("x_axis"),
            "reference_implementation": "N=common num_clients, H=1 full causal attention; separate CenAttn diagnostic",
            "limitations": "Prefill fidelity excludes generated trajectories. Sparse masks, if requested, are newly seeded."
        }
        logical_manifest = {k: v for k, v in manifest.items() if k not in
                            ("output_jsonl", "_notes", "data_jsonl", "sample_plan_json")}
        identity = {k: v for k, v in provenance.items() if k not in ("manifest_sha256", "data_sha256")}
        provenance["run_id"] = digest({"settings": logical_manifest, "identity": identity})
        output = Path(manifest["output_jsonl"])
        output.parent.mkdir(parents=True, exist_ok=True)
        complete = read_completed(output, provenance, resume=resume)
        sidecar = Path(str(output) + ".manifest.json")
        if not sidecar.exists():
            sidecar.write_text(json.dumps({"manifest": manifest, "provenance": provenance}, ensure_ascii=False, indent=2), encoding="utf-8")
        if output.exists() and output.stat().st_size:
            with output.open("rb") as handle:
                handle.seek(-1, 2)
                needs_newline = handle.read(1) != b"\n"
            if needs_newline:
                with output.open("a", encoding="utf-8") as handle:
                    handle.write("\n")
        chunks_cache = {}
        expected = len(indices) * len(configs)
        new_rows = 0
        with output.open("a", encoding="utf-8") as stream:
            for sample_index in indices:
                if all((sample_index, c["name"]) in complete for c in configs):
                    continue
                doc = docs[sample_index]
                prompt = original.doc_to_text(doc, fewshot)
                reference = None
                reference_ids = None
                sample_id = doc.get("id", doc.get("sample_id", sample_index))
                seed = sample_seed(manifest.get("seed", 0), sample_id)
                for config in configs:
                    key = (sample_index, config["name"])
                    already = key in complete
                    if already and config["name"] != "CenAttn":
                        continue
                    hp = dict(common)
                    hp.update({k: v for k, v in config.items() if k not in
                               ("name", "comm_policy", "token_chunks_json", "restore_remote_layer")})
                    if hp["num_shots"] != common["num_shots"]:
                        raise ValueError("All settings must use the same pooled prompt")
                    for name in ("num_clients", "num_local_forward"):
                        if type(hp[name]) is not int or hp[name] < 1:
                            raise ValueError(f"{name} must be a positive integer")
                    chunk_path = config.get("token_chunks_json", manifest.get("token_chunks_json"))
                    chunks = None
                    if chunk_path:
                        if chunk_path not in chunks_cache:
                            chunks_cache[chunk_path] = json.loads(Path(chunk_path).read_text(encoding="utf-8"))
                        chunks = chunks_cache[chunk_path][sample_index]
                        if len(chunks) != hp["num_clients"]:
                            raise ValueError("token_chunks_json client count differs from config")
                    policy = config.get("comm_policy", "main")
                    captured = capture_masks(original, wrapper, tokenizer, model, device, prompt, lengths,
                                             chunks, hp, manifest["split_way"], policy, seed)
                    validate_chunks(captured)
                    if "restore_remote_layer" in config:
                        restore_layer(torch, captured, config["restore_remote_layer"], hp, policy)
                    ids = captured["input_ids"][0].cpu().tolist()
                    if reference_ids is not None and ids != reference_ids:
                        raise ValueError("Cannot compare fidelity: input token IDs changed across settings")
                    current = run_prefill(torch, model, wrapper, captured, layers, query_count, reference=reference)
                    if reference is None:
                        reference, reference_ids = current, ids
                    if already:
                        if complete[key]["input_ids_sha256"] != digest(ids):
                            raise ValueError("Resumed central input tokens differ from saved tokens")
                        del captured, current
                        continue
                    record = {
                        "schema_version": 4, "metric_id": METRIC_ID, "run_id": provenance["run_id"],
                        **point_metadata(manifest, config, hp),
                        "sample_set_sha256": sample_plan["sample_set_sha256"],
                        "n_expected_samples": len(indices),
                        "sample_id": sample_id, "sample_index": sample_index,
                        "question": doc["question"], "answer": doc["answer"],
                        "question_sha256": digest(doc["question"]), "answer_sha256": digest(doc["answer"]),
                        "config": config["name"], "hyperparams": hp, "comm_policy": policy,
                        "mask_seed": seed, "input_ids_sha256": digest(ids), "token_chunks_sha256": digest(captured["chunks"]),
                        "publisher_is_suffix": sorted(captured["chunks"][-1]) == list(range(
                            min(captured["chunks"][-1]), int(captured["padding_mask"][0].sum()))),
                        "prefill_masks_sha256": mask_hash(captured), "prompt_tokens": len(ids),
                        "kv_availability_sha256": current["kv_availability_sha256"],
                        "kv_stage": "post_RoPE_K_and_projected_V_before_layer_attention",
                        "restore_remote_layer": config.get("restore_remote_layer"),
                        "fidelity": metrics(torch, current, reference, captured), "mechanism": current["mechanism"],
                        "utility_source": "not_generated"}
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                    stream.flush()
                    complete[key] = record
                    new_rows += 1
                    if new_rows == 1 or new_rows % 50 == 0 or len(complete) == expected:
                        print(f"{manifest.get('model', '')} {manifest['split_way']} sample={sample_index} "
                              f"config={config['name']} mean_layer_participant_kv_deviation="
                              f"{record['fidelity']['layer_participant_mean_relative_kv_frobenius']:.6g} "
                              f"saved={len(complete)}/{expected}", flush=True)
                    del captured, current
        result = {"output_jsonl": str(output), "n_expected_records": expected,
                  "n_saved_records": len(complete), "new_records": new_rows,
                  "complete": len(complete) == expected, "sample_indices": indices,
                  "sample_set_sha256": sample_plan["sample_set_sha256"]}
        Path(str(output) + ".status.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result
    finally:
        if own_runtime:
            close_runtime(runtime)
        else:
            runtime["wrapper"].reset()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest")
    parser.add_argument("--resume", action="store_true", help="Skip saved sample/config records after checking identical run identity")
    args = parser.parse_args()
    if not args.manifest:
        parser.error("--manifest is required")
    print(json.dumps(run_manifest(args.manifest, generate=False, resume=args.resume), indent=2))


if __name__ == "__main__":
    main()
