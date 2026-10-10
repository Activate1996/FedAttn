#!/usr/bin/env python3
"""Measure FedAttn prefill representations on matched prompts.

Use the FedAttn inference environment with torch, transformers, and jsonlines.
Masks are obtained from the inference mask builders, with random seeds recorded
for each sample. Publisher responses can be generated using the measured masks.
Use --write-example example_probe.json to create an example measurement manifest.
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


def example():
    return {
        "schema_version": 2,
        "figure": 2,
        "model": "Qwen2.5-3B",
        "num_layers": 36,
        "source_repo": "/path/to/FedAttn-main",
        "model_path": "/path/to/Qwen2.5-3B",
        "data_jsonl": "/path/to/original_gsm8k_test.jsonl",
        "output_jsonl": "/path/to/probe_results.jsonl",
        "device": "cuda:0",
        "dtype": "float32",
        "seed": 0,
        "sample_count": 64,
        "sample_selection": "stride",
        "sample_start": 0,
        "sample_step": 20,
        "subset_seed": 20260917,
        "split_way": "smart",
        "hyperparams": {"num_shots": 4, "num_clients": 4, "num_local_forward": 9,
                        "max_new_tokens": 256, "do_sample": False},
        "mechanism_layers": [],
        "mechanism_query_count": 1,
        "generate_clients": "publisher",
        "configs": [{"name": f"H{h}", "comm_policy": "main", "num_local_forward": h}
                    for h in range(2, 37)],
        "_notes": [
            "Set the paths and hyperparameters for the target experiment.",
            "This Fig. 2 example sweeps H; representation measurements do not require --generate.",
            "The inference loader omits torch_dtype; specify and verify the dtype used for measurement.",
            "Sampling uses range(sample_start, dataset_size, sample_step)[:sample_count].",
            "Use sample_indices to select dataset row indices, or share sample_plan_json across panels.",
            "Optional token_chunks_json stores token partitions indexed by dataset row.",
            "Different num_clients may be specified per config; pooled prompt must remain identical.",
            "mechanism_layers are zero-based; choose valid layers for your model.",
            "--generate defaults to the task publisher, under the very masks measured here.",
            "Set generate_clients to all when matching a figure averaged over every client; this costs more decoding.",
            "For sampled decoding, specify do_sample and the complete set of sampling parameters.",
            "Pair sparse-mask fidelity with accuracy only when the same masks and samples were used.",
            "restore_remote_layer requires sample_uni_comm, all computation ratios=1, and a synchronization layer.",
            "An addback case is an illustration; use a prespecified subset for population claims."
        ]
    }


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
    """Capture the FedAttn masks immediately before the first generation call."""
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
    # Preserve padding-query behaviour from the captured mask; batch size is one.
    full[0, 0, ~valid[0], :] = captured["prefill_layer_masks"][layer][0, 0, ~valid[0], :]
    captured["prefill_layer_masks"][layer] = full
    for client_masks in captured["all_decode_masks"]:
        restored = client_masks[layer].clone()
        restored.fill_(torch.finfo(restored.dtype).min)
        restored.masked_fill_(valid[:, None, None, :], 0)
        client_masks[layer] = restored
    captured["decode_layer_masks"] = captured["all_decode_masks"][-1]


class MechanismRecorder:
    """Observe SDPA queries, keys, and values after rotary encoding.

    Compare all causally available keys with the applied mask using the same
    queries, keys, and values. This measures attention renormalisation and
    omitted-value effects, not semantic relevance or cross-run state deviation.
    """
    def __init__(self, torch, model, captured, layers, query_count):
        self.torch, self.model, self.cap = torch, model, captured
        self.layers, self.query_count = set(layers), query_count
        self.records, self.handles, self.current = [], [], None

    def __enter__(self):
        if not self.layers:
            return self
        self.F = self.torch.nn.functional
        self.original = self.F.scaled_dot_product_attention
        for idx, layer in enumerate(self.model.model.layers):
            if idx in self.layers:
                self.handles.append(layer.self_attn.register_forward_pre_hook(self._pre(idx)))
                self.handles.append(layer.self_attn.register_forward_hook(self._post))
        self.F.scaled_dot_product_attention = self._sdpa
        return self

    def _pre(self, idx):
        def hook(module, args):
            self.current = (idx, module)
        return hook

    def _post(self, module, args, output):
        self.current = None

    def _sdpa(self, q, k, v, *args, **kwargs):
        if self.current is not None:
            mask = kwargs.get("attn_mask", args[0] if len(args) > 0 else None)
            dropout = kwargs.get("dropout_p", args[1] if len(args) > 1 else 0.0)
            causal = kwargs.get("is_causal", args[2] if len(args) > 2 else False)
            if dropout:
                raise ValueError("Mechanism measurement requires model.eval() and zero attention dropout.")
            self._measure(q, k, v, mask, causal, kwargs.get("scale"))
        return self.original(q, k, v, *args, **kwargs)

    def _measure(self, q, k, v, mask, causal, scale):
        t = self.torch
        layer_idx, module = self.current
        if q.shape[0] != 1 or q.shape[-2] != k.shape[-2]:
            raise ValueError("Recorder supports batch-one, cache-free prefill only.")
        length = int(self.cap["padding_mask"][0].sum())
        size = q.shape[-2]
        offset = size - length if self.cap["padding_side"] == "left" else 0
        publisher = sorted(self.cap["chunks"][-1])
        rows = t.tensor([offset + x for x in publisher[-self.query_count:]], device=q.device)
        if k.shape[1] != q.shape[1]:
            if q.shape[1] % k.shape[1]:
                raise ValueError("Invalid GQA head dimensions.")
            groups = q.shape[1] // k.shape[1]
            k, v = k.repeat_interleave(groups, 1), v.repeat_interleave(groups, 1)
        logits = t.matmul(q[0].index_select(1, rows).float(), k[0].float().transpose(-1, -2))
        logits *= scale if scale is not None else q.shape[-1] ** -0.5
        keys = t.arange(size, device=q.device)
        causal_valid = (keys[None, :] <= rows[:, None]) & self.cap["padding_mask"][0].bool()[None, :]
        full_allowed = causal_valid[None].expand(logits.shape[0], -1, -1)
        actual_allowed = full_allowed.clone()
        additive = t.zeros_like(logits)
        if mask is not None:
            # SDPA masks have shape [B, 1, Q, K] or [B, H, Q, K].
            expanded = t.broadcast_to(mask, (1, logits.shape[0], size, size))[0].index_select(1, rows)
            if expanded.dtype == t.bool:
                actual_allowed &= expanded
            else:
                actual_allowed &= expanded > -1e4
                additive = t.where(actual_allowed, expanded.float(), t.zeros_like(logits))
                if t.any(additive != 0):
                    raise ValueError("Expected zero/min additive masks; relative-bias attention needs an adapted observer.")
        elif not causal:
            # Require causal restrictions from the FedAttn mask builders.
            raise RuntimeError("No causal mask reached SDPA; inspect wrapper compatibility.")
        if t.any(actual_allowed.sum(-1) == 0):
            raise ValueError("Selected query has no valid key.")
        p_full = logits.masked_fill(~full_allowed, -float("inf")).softmax(-1)
        p_keep = logits.masked_fill(~actual_allowed, -float("inf")).softmax(-1)
        own = t.zeros(size, dtype=t.bool, device=q.device)
        own[t.tensor(publisher, device=q.device) + offset] = True
        remote = ~own & self.cap["padding_mask"][0].bool()
        dropped = full_allowed & ~actual_allowed
        mass = (p_full * dropped).sum(-1)
        values = v[0].float()
        u_full, u_keep = p_full @ values, p_keep @ values
        dropped_sum = (p_full * dropped) @ values
        # Identity: full - kept = dropped_mass * (dropped_mean - kept_mean).
        rhs = dropped_sum - mass[..., None] * u_keep
        delta = u_full - u_keep
        flat_delta = delta.transpose(0, 1).reshape(len(rows), -1)
        projected = t.nn.functional.linear(flat_delta, module.o_proj.weight.float(), bias=None)
        measurable = mass > 1e-8
        contrast = dropped_sum / mass.clamp_min(1e-8)[..., None] - u_keep
        contrast_norm = contrast.norm(dim=-1)
        self.records.append({
            "layer": layer_idx, "query_positions": rows.cpu().tolist(),
            "query_count": len(rows),
            "actual_remote_attention_mass": (p_keep * remote).sum(-1).mean().item(),
            "full_remote_attention_mass_same_state": (p_full * remote).sum(-1).mean().item(),
            "dropped_attention_mass_same_state": mass.mean().item(),
            "dropped_value_contrast_norm_mean": contrast_norm[measurable].mean().item() if measurable.any() else None,
            "dropped_value_contrast_head_query_count": int(measurable.sum()),
            "actual_local_attention_mass": (p_keep * own).sum(-1).mean().item(),
            "full_local_attention_mass_same_state": (p_full * own).sum(-1).mean().item(),
            "head_output_delta_rms": delta.square().mean().sqrt().item(),
            "projected_attention_delta_frobenius": projected.norm().item(),
            "decomposition_max_abs_residual": (delta - rhs).abs().max().item(),
            "full_causal_key_count_mean": full_allowed.sum(-1).float().mean().item(),
            "actual_key_count_mean": actual_allowed.sum(-1).float().mean().item(),
            "note": "Fixed-state all-causal vs actual-mask diagnostic; attention mass is not task relevance."
        })

    def __exit__(self, *exc):
        if self.layers:
            self.F.scaled_dot_product_attention = self.original
        for handle in self.handles:
            handle.remove()


def run_prefill(torch, model, wrapper, captured, mechanism_layers, query_count):
    saved = {}

    def final_block_hook(module, inputs, output):
        value = output[0] if isinstance(output, (tuple, list)) else output
        saved["pre_final_norm"] = value[0].detach().float().cpu()

    hook = model.model.layers[-1].register_forward_hook(final_block_hook)
    wrapper.batch_size, wrapper.input_len = captured["input_ids"].shape
    wrapper.current_token_counts = captured["padding_mask"].sum(1)
    wrapper.is_prefill, wrapper.layer_ptr = True, 0
    wrapper.set_masks(prefill_masks=captured["prefill_layer_masks"], decode_masks=None)
    try:
        with torch.inference_mode(), MechanismRecorder(torch, model, captured, mechanism_layers, query_count) as recorder:
            output = model.model(
                input_ids=captured["input_ids"], attention_mask=captured["padding_mask"],
                position_ids=wrapper.create_position_ids(captured["padding_mask"], captured["padding_side"]),
                use_cache=False, output_hidden_states=False, return_dict=True)
            last = captured["input_ids"].shape[1] - 1 if captured["padding_side"] == "left" else int(captured["padding_mask"].sum()) - 1
            saved["last_logits"] = model.lm_head(output.last_hidden_state[:, last, :])[0].float().cpu()
            saved["mechanism"] = recorder.records
        if mechanism_layers and len(saved["mechanism"]) != len(set(mechanism_layers)):
            raise RuntimeError("SDPA observer did not record every requested layer; use the original SDPA environment.")
    finally:
        hook.remove()
        wrapper.reset()
    return saved


def metrics(torch, current, reference, captured):
    mask = captured["padding_mask"][0].bool().cpu()
    offset = len(mask) - int(mask.sum()) if captured["padding_side"] == "left" else 0
    publisher = torch.tensor(captured["chunks"][-1], dtype=torch.long) + offset
    last = publisher.max().reshape(1)
    h, ref = current["pre_final_norm"], reference["pre_final_norm"]
    result = {}
    for name, index in [("all_tokens", mask), ("publisher_tokens", publisher), ("last_prompt_token", last)]:
        difference = (h[index] - ref[index]).norm().item()
        denominator = ref[index].norm().item()
        result[name] = {"absolute_frobenius": difference, "reference_frobenius": denominator,
                        "per_entry_rms": difference / math.sqrt(h[index].numel()),
                        "relative_frobenius": difference / max(denominator, 1e-12),
                        "n_tokens": int(h[index].shape[0])}
    lp, lq = reference["last_logits"].log_softmax(-1), current["last_logits"].log_softmax(-1)
    p, q = lp.exp(), lq.exp()
    lm = torch.logaddexp(lp, lq) - math.log(2)
    result["first_token_distribution"] = {
        "kl_cen_to_setting": max(0.0, (p * (lp - lq)).sum().item()),
        "js_divergence": max(0.0, (0.5 * (p * (lp - lm)).sum() + 0.5 * (q * (lq - lm)).sum()).item()),
        "argmax_agreement": bool(reference["last_logits"].argmax() == current["last_logits"].argmax())}
    return result


def generate_client(torch, wrapper, tokenizer, captured, hp, seed, client):
    # Reuse captured masks, including any single-layer restoration.
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        output = wrapper.generate(
            captured["input_ids"], padding_side=captured["padding_side"],
            padding_mask=captured["padding_mask"],
            prefill_layer_masks=captured["prefill_layer_masks"],
            decode_layer_masks=captured["all_decode_masks"][client],
            max_new_tokens=hp["max_new_tokens"], do_sample=hp["do_sample"],
            temperature=hp.get("temperature", 0.2), top_k=hp.get("top_k", 20), top_p=hp.get("top_p", 0.9))
        return tokenizer.decode(output[0, captured["input_ids"].shape[1]:], skip_special_tokens=True)
    finally:
        wrapper.reset()


def create_runtime(manifest):
    """Load one model and reuse it across its Fig. 2/3 panels."""
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
        generate_clients = manifest.get("generate_clients", "publisher")
        if generate_clients not in ("publisher", "all"):
            raise ValueError("generate_clients must be publisher or all")
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
            **sample_plan, "manifest_sha256": file_hash(manifest_path),
            "data_sha256": file_hash(manifest["data_jsonl"]), "source_hashes": source_hashes,
            "probe_script_sha256": file_hash(__file__), "fewshot_prompt_sha256": digest(fewshot),
            "model_path": runtime["model_path"], "checkpoint_files": runtime["checkpoint_files"],
            "torch_version": torch.__version__, "transformers_version": transformers.__version__,
            "dtype": str(model.dtype), "device": str(device), "planned_points": points,
            "generated_utility": generate, "generate_clients": generate_clients,
            "token_chunks_file_hashes": chunk_hashes,
            "figure": manifest.get("figure"), "model": manifest.get("model", Path(manifest["model_path"]).name),
            "split_way": manifest["split_way"], "x_axis": points[0].get("x_axis"),
            "reference_implementation": "N=common num_clients, H=1 full causal attention; central plot x=1",
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
                    current = run_prefill(torch, model, wrapper, captured, layers, query_count)
                    if reference is None:
                        reference, reference_ids = current, ids
                    if already:
                        if complete[key]["input_ids_sha256"] != digest(ids):
                            raise ValueError("Resumed central input tokens differ from saved tokens")
                        del captured, current
                        continue
                    record = {
                        "schema_version": 2, "run_id": provenance["run_id"],
                        **point_metadata(manifest, config, hp),
                        "sample_set_sha256": sample_plan["sample_set_sha256"],
                        "n_expected_samples": len(indices),
                        "sample_id": sample_id, "sample_index": sample_index,
                        "question": doc["question"], "answer": doc["answer"],
                        "question_sha256": digest(doc["question"]), "answer_sha256": digest(doc["answer"]),
                        "config": config["name"], "hyperparams": hp, "comm_policy": policy,
                        "mask_seed": seed, "input_ids_sha256": digest(ids), "token_chunks_sha256": digest(captured["chunks"]),
                        "prefill_masks_sha256": mask_hash(captured), "prompt_tokens": len(ids),
                        "restore_remote_layer": config.get("restore_remote_layer"),
                        "fidelity": metrics(torch, current, reference, captured), "mechanism": current["mechanism"],
                        "utility_source": "fresh_same_mask" if generate else "not_generated"}
                    if generate:
                        clients = range(hp["num_clients"]) if generate_clients == "all" else [hp["num_clients"] - 1]
                        completions, accs = {}, {}
                        for client in clients:
                            completion = generate_client(torch, wrapper, tokenizer, captured, hp, seed, client)
                            completions[str(client)] = completion
                            accs[str(client)] = original.is_correct(completion, doc["answer"])
                        publisher = str(hp["num_clients"] - 1)
                        record.update(completion=completions[publisher], acc=accs[publisher],
                                      completions_by_client=completions, acc_by_client=accs,
                                      accuracy_mean=sum(accs.values()) / len(accs), generated_clients=generate_clients)
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                    stream.flush()
                    complete[key] = record
                    new_rows += 1
                    print(f"{manifest.get('model', '')} {manifest['split_way']} sample={sample_index} "
                          f"config={config['name']} rel_F={record['fidelity']['all_tokens']['relative_frobenius']:.6g} "
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
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Skip saved sample/config records after checking identical run identity")
    parser.add_argument("--write-example")
    args = parser.parse_args()
    if args.write_example:
        Path(args.write_example).write_text(json.dumps(example(), indent=2) + "\n")
        return
    if not args.manifest:
        parser.error("--manifest is required")
    print(json.dumps(run_manifest(args.manifest, generate=args.generate, resume=args.resume), indent=2))


if __name__ == "__main__":
    main()
