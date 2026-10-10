#!/usr/bin/env python3
"""Measure block-output error over all valid tokens against a shared CenAttn prefill.

Measurements use prefill activations without generating answers. For H=4,
synchronization uses zero-based layers 0, 4, 8, ... (one-based blocks 1, 5, 9, ...).
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import shutil
import sys
import zipfile

from deviation_io import load_docs, make_sample_plan
from server_resources import resolve_data_jsonl, resolve_model_reference
import trajectory_backend as backend

LAYERS = {"Qwen2.5-0.5B": 24, "Qwen2.5-1.5B": 28,
          "Qwen2.5-3B": 36, "Qwen2.5-7B": 28}
SPLITS = ("even", "even_question_last", "smart", "smart_question_last")


def absolute_path(value, base):
    p = Path(value).expanduser()
    return (p if p.is_absolute() else base / p).resolve()


def immutable_json(path, obj):
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != obj:
            raise ValueError(f"Different configuration at {path}; use a new output_root")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(obj, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


@contextmanager
def output_lock(root):
    """Require a distinct output directory for each concurrent run."""
    root.mkdir(parents=True, exist_ok=True)
    path = root / "trajectory_running.lock"
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise RuntimeError(f"{path} exists. Use separate output roots for parallel jobs. "
                           "If an earlier job was killed, remove this lock only after confirming it stopped.") from exc
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(str(os.getpid()))
        yield
    finally:
        path.unlink(missing_ok=True)


def prepare(settings_path, output_override=None):
    path = Path(settings_path).resolve()
    cfg = json.loads(path.read_text(encoding="utf-8"))
    base = path.parent
    root = absolute_path(output_override or cfg["output_root"], base)
    source = absolute_path(cfg["source_repo"], base)
    fed = source / "FedAttn" if (source / "FedAttn").is_dir() else source
    if not (fed / "evaluate_gsm8k" / "main_evaluate_gsm8k.py").is_file():
        raise FileNotFoundError(f"Original main_evaluate_gsm8k.py not found under {source}")
    for key in ("H", "num_clients", "num_shots"):
        if type(cfg.get(key)) is not int or cfg[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    splits = cfg.get("splits", list(SPLITS))
    if not splits or len(set(splits)) != len(splits) or set(splits) - set(SPLITS):
        raise ValueError("Invalid or duplicate split names")
    models = cfg.get("models", {})
    if not models or set(models) - set(LAYERS):
        raise ValueError("models must contain supported canonical Qwen model names")
    dtypes = {m: cfg.get("dtype_by_model", {}).get(m, "float32") for m in models}
    if set(dtypes.values()) - {"float32", "bfloat16", "float16"}:
        raise ValueError("dtype_by_model supports float32, bfloat16, float16")
    data_path = resolve_data_jsonl(cfg, base, root)
    docs = load_docs(data_path)
    sample_plan = make_sample_plan(docs, cfg)
    # Reuse an existing sample plan to preserve the selected sample identities.
    if cfg.get("sample_plan_json"):
        old = json.loads(absolute_path(cfg["sample_plan_json"], base).read_text(encoding="utf-8"))
        if old != sample_plan:
            raise ValueError("Sampling/data differ from the supplied previous sample_plan.json")
    resolved = {m: resolve_model_reference(ref, base) for m, ref in models.items()}
    settings = {"source_repo": str(source), "models": resolved,
                "dtype_by_model": dtypes, "splits": splits,
                "H": cfg["H"], "num_clients": cfg["num_clients"], "num_shots": cfg["num_shots"],
                "device": cfg.get("device", "cuda:0"), "seed": cfg.get("seed", 0),
                "prompt_prefix": cfg.get("prompt_prefix", "gsm8k_prompt_"),
                "prompt_suffix": cfg.get("prompt_suffix", ".txt")}
    plan = {"schema_version": 1, "sample_plan": sample_plan, "settings": settings,
            "models": {m: {"num_layers": LAYERS[m], "dtype": dtypes[m],
                           "records_file": f"trajectory_{m}.jsonl"} for m in models},
            "measurement": "complete decoder block output before the following normalization; all valid input tokens",
            "normalization": "per-sample and per-block CenAttn Frobenius norm; then arithmetic mean across samples"}
    immutable_json(root / "sample_plan.json", sample_plan)
    immutable_json(root / "trajectory_plan.json", plan)
    print(f"Samples ({len(sample_plan['sample_indices'])}): {sample_plan['sample_indices']}", flush=True)
    print(f"H={cfg['H']}, N={cfg['num_clients']}, shots={cfg['num_shots']}; "
          "global block labels follow actual masks, not a presumed decrease in error.", flush=True)
    return root, plan, docs, fed


def forward_blocks(runtime, captured, reference=None):
    """Store reference block states in CPU memory and retain only FedAttn error scalars."""
    torch, model, wrapper = runtime["torch"], runtime["model"], runtime["wrapper"]
    backend.validate_chunks(captured)
    backend.align_prefill_masks(torch, model, captured)
    valid = captured["padding_mask"][0].bool().cpu()
    offset = len(valid) - int(valid.sum()) if captured["padding_side"] == "left" else 0
    pub = torch.tensor(captured["chunks"][-1], dtype=torch.long) + offset
    n_layers = len(model.model.layers)
    states = [None] * n_layers
    fields = ("delta_traj_all", "delta_traj_pub", "absolute_traj_all", "absolute_traj_pub",
              "reference_norm_traj_all", "reference_norm_traj_pub")
    measured = {name: [None] * n_layers for name in fields}
    visited = set()

    def hook_at(index):
        def hook(module, inputs, output):
            value = output[0] if isinstance(output, (tuple, list)) else output
            h = value[0].detach().float().cpu()
            if index in visited:
                raise RuntimeError(f"Decoder block {index + 1} was visited twice")
            visited.add(index)
            if reference is None:
                states[index] = h
                return
            ref = reference[index]
            if h.shape != ref.shape:
                raise ValueError("FedAttn/CenAttn block shapes differ")
            for scope, token_index in (("all", valid), ("pub", pub)):
                difference = (h[token_index] - ref[token_index]).norm().item()
                denominator = ref[token_index].norm().item()
                relative = difference / max(denominator, 1e-12)
                if not all(math.isfinite(v) for v in (difference, denominator, relative)):
                    raise ValueError(f"Non-finite deviation at block {index + 1}; no row saved")
                measured[f"absolute_traj_{scope}"][index] = difference
                measured[f"reference_norm_traj_{scope}"][index] = denominator
                measured[f"delta_traj_{scope}"][index] = relative
        return hook

    handles = []
    wrapper.batch_size, wrapper.input_len = captured["input_ids"].shape
    if wrapper.batch_size != 1:
        raise ValueError("Trajectory replay uses one sample per forward")
    wrapper.current_token_counts = captured["padding_mask"].sum(1)
    wrapper.is_prefill, wrapper.layer_ptr = True, 0
    wrapper.set_masks(prefill_masks=captured["prefill_layer_masks"], decode_masks=None)
    try:
        for i, layer in enumerate(model.model.layers):
            handles.append(layer.register_forward_hook(hook_at(i)))
        with torch.inference_mode():
            model.model(input_ids=captured["input_ids"], attention_mask=captured["padding_mask"],
                        position_ids=wrapper.create_position_ids(captured["padding_mask"], captured["padding_side"]),
                        use_cache=False, output_hidden_states=False, return_dict=True)
        if len(visited) != n_layers:
            raise RuntimeError("Not every decoder block was captured")
    finally:
        for handle in handles:
            handle.remove()
        wrapper.reset()
    if reference is None:
        return states
    measured.update(delta_traj=measured["delta_traj_all"],
                    absolute_delta_traj=measured["absolute_traj_all"],
                    n_all_tokens=int(valid.sum()), n_publisher_tokens=int(pub.numel()))
    return measured


def verify_schedule(torch, captured, full_allowed, H):
    cache, global_blocks = {}, []
    for i, mask in enumerate(captured["prefill_layer_masks"]):
        if id(mask) not in cache:
            cache[id(mask)] = torch.equal(mask == 0, full_allowed)
        actual_global = cache[id(mask)]
        if actual_global:
            global_blocks.append(i + 1)
        if actual_global != (i % H == 0):
            raise ValueError(f"Actual mask at block {i + 1} disagrees with original H={H} schedule")
    return global_blocks


def load_completed(path, run_id, sample_plan, splits, resume):
    rows = {}
    if not path.exists():
        return rows
    if not resume:
        raise FileExistsError(f"{path} exists; pass --resume for the same run")
    expected = {(i, s) for i in sample_plan["sample_indices"] for s in splits}
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Interrupted or corrupt JSONL at {path}:{lineno}; inspect before resuming") from exc
        key = (row["sample_index"], row["split_way"])
        if key not in expected or row.get("run_id") != run_id:
            raise ValueError("Saved trajectory does not match source/runtime/sample/config provenance")
        if key in rows and rows[key] != row:
            raise ValueError("Conflicting duplicate trajectory record")
        rows[key] = row
    return rows


def run_model(model_name, plan, docs, fed, root, resume):
    settings, samples = plan["settings"], plan["sample_plan"]
    model_meta = plan["models"][model_name]
    manifest = {"source_repo": settings["source_repo"], "model": model_name,
                "model_path": settings["models"][model_name], "num_layers": model_meta["num_layers"],
                "device": settings["device"], "dtype": model_meta["dtype"]}
    print(f"Loading {model_name} once ({manifest['dtype']})", flush=True)
    runtime = backend.create_runtime(manifest)
    try:
        original, tokenizer = runtime["original"], runtime["tokenizer"]
        fewshot, lengths = original.load_prompts(list(range(settings["num_shots"])), tokenizer, True,
                                               settings["prompt_prefix"], settings["prompt_suffix"])
        chunks_by_split, chunk_hashes = {}, {}
        for split in settings["splits"]:
            p = fed / "evaluate_gsm8k" / "data_process_token_chunks" / (
                f"data_process_{model_name}_shot{settings['num_shots']}_c{settings['num_clients']}_{split}.json")
            chunks_by_split[split] = json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None
            chunk_hashes[split] = backend.file_hash(p) if p.is_file() else None
        source_hashes = {str(p.relative_to(fed)): backend.file_hash(p) for p in
                         (fed / "utils.py", fed / "Wrapper.py", fed / "evaluate_gsm8k" / "main_evaluate_gsm8k.py")}
        program_hashes = {name: backend.file_hash(Path(__file__).with_name(name)) for name in
                          (Path(__file__).name, "trajectory_backend.py", "deviation_io.py", "server_resources.py")}
        provenance = {"manifest": manifest, "sample_plan": samples,
                      "H": settings["H"], "num_clients": settings["num_clients"], "num_shots": settings["num_shots"],
                      "splits": settings["splits"], "seed": settings["seed"],
                      "source_hashes": source_hashes, "program_hashes": program_hashes,
                      "fewshot_prompt_sha256": backend.digest(fewshot), "chunk_hashes": chunk_hashes,
                      "checkpoint_path": runtime["model_path"], "checkpoint_files": runtime["checkpoint_files"],
                      "torch_version": runtime["torch"].__version__,
                      "transformers_version": runtime["transformers"].__version__}
        run_id = backend.digest(provenance)
        output = root / model_meta["records_file"]
        sidecar = Path(str(output) + ".manifest.json")
        if output.exists() and not sidecar.exists():
            raise ValueError("Cannot resume an existing trajectory without its provenance sidecar")
        immutable_json(sidecar, {"run_id": run_id, "provenance": provenance})
        completed = load_completed(output, run_id, samples, settings["splits"], resume)
        hp = {"num_shots": settings["num_shots"], "num_clients": settings["num_clients"],
              "num_local_forward": settings["H"], "max_new_tokens": 256, "do_sample": False}
        total = len(samples["sample_indices"]) * len(settings["splits"])
        # Add a separator if the last complete JSON record has no trailing newline.
        if output.exists() and output.stat().st_size:
            with output.open("rb") as handle:
                handle.seek(-1, 2)
                needs_newline = handle.read(1) != b"\n"
            if needs_newline:
                with output.open("a", encoding="utf-8") as handle:
                    handle.write("\n")
        with output.open("a", encoding="utf-8") as stream:
            for sample_index in samples["sample_indices"]:
                pending = [s for s in settings["splits"] if (sample_index, s) not in completed]
                if not pending:
                    continue
                doc = docs[sample_index]
                prompt = original.doc_to_text(doc, fewshot)
                seed = backend.sample_seed(settings["seed"], doc.get("id", doc.get("sample_id", sample_index)))

                def capture(split, H):
                    chunk_rows = chunks_by_split[split]
                    chunks = None if chunk_rows is None else chunk_rows[sample_index]
                    if chunks is not None and len(chunks) != settings["num_clients"]:
                        raise ValueError("Historical token chunk participant count differs")
                    local_hp = dict(hp, num_local_forward=H)
                    result = backend.capture_masks(original, runtime["wrapper"], tokenizer, runtime["model"],
                                                   runtime["device"], prompt, lengths, chunks, local_hp,
                                                   split, "main", seed)
                    backend.validate_chunks(result)
                    return result

                central_cap = capture(pending[0], 1)
                reference_ids = central_cap["input_ids"].detach().cpu().tolist()
                full_allowed = central_cap["prefill_layer_masks"][0] == 0
                if not all(runtime["torch"].equal(m == 0, full_allowed)
                           for m in central_cap["prefill_layer_masks"]):
                    raise ValueError("H=1 reference is not globally visible at every layer")
                reference = forward_blocks(runtime, central_cap)
                del central_cap
                for split in pending:
                    captured = capture(split, settings["H"])
                    if captured["input_ids"].detach().cpu().tolist() != reference_ids:
                        raise ValueError("Pooled prompt token IDs changed across segmentations")
                    global_blocks = verify_schedule(runtime["torch"], captured, full_allowed, settings["H"])
                    measured = forward_blocks(runtime, captured, reference)
                    row = {"schema_version": 1, "run_id": run_id, "model": model_name, "split_way": split,
                           "sample_index": sample_index, "question_sha256": backend.digest(doc["question"]),
                           "answer_sha256": backend.digest(doc["answer"]),
                           "sample_set_sha256": samples["sample_set_sha256"],
                           "dtype": model_meta["dtype"], "num_layers": model_meta["num_layers"],
                           "H": settings["H"], "num_clients": settings["num_clients"], "num_shots": settings["num_shots"],
                           "block_indices": list(range(1, model_meta["num_layers"] + 1)),
                           "global_blocks": global_blocks,
                           "input_ids_sha256": backend.digest(reference_ids[0]),
                           "token_chunks_sha256": backend.digest(captured["chunks"]),
                           "publisher_token_indices": captured["chunks"][-1],
                           "prefill_masks_sha256": backend.mask_hash(captured), **measured}
                    stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                    completed[(sample_index, split)] = row
                    print(f"{model_name} {split} sample={sample_index} "
                          f"last_delta_all={row['delta_traj'][-1]:.6g} saved={len(completed)}/{total}", flush=True)
                    del captured
                del reference, full_allowed
        print(f"Finished {model_name}: {len(completed)}/{total} sample/split trajectories", flush=True)
    finally:
        backend.close_runtime(runtime)


def export_bundle(root):
    from trajectory_io import export_trajectories
    result = export_trajectories(root)
    bundle = root / "block_trajectory_plot_bundle"
    bundle.mkdir(exist_ok=True)
    for key in ("summary_path", "sample_csv_path", "points_csv_path"):
        p = Path(result[key])
        shutil.copy2(p, bundle / p.name)
    shutil.copy2(root / "sample_plan.json", bundle / "sample_plan.json")
    for name in ("trajectory_io.py",):
        shutil.copy2(Path(__file__).with_name(name), bundle / name)
    archive = root / "block_trajectory_plot_bundle.zip"
    temporary = archive.with_suffix(".zip.tmp")
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as handle:
        for p in sorted(bundle.iterdir()):
            if p.is_file():
                handle.write(p, f"block_trajectory_plot_bundle/{p.name}")
    temporary.replace(archive)
    print(f"Copy to plotting host: {archive}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", default=str(Path(__file__).with_name("settings_trajectory_server212.json")))
    parser.add_argument("--models", nargs="+", choices=list(LAYERS))
    parser.add_argument("--output-root", help="Separate output root for each concurrent GPU job")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--export-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    cfg_path = Path(args.settings).resolve()
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    root = absolute_path(args.output_root or cfg["output_root"], cfg_path.parent)
    with output_lock(root):
        root, plan, docs, fed = prepare(args.settings, args.output_root)
        selected = args.models or list(plan["models"])
        if set(selected) - set(plan["models"]):
            parser.error("A selected model is missing from settings.models")
        if args.prepare_only or args.export_only:
            export_bundle(root)
            return
        try:
            for model_name in selected:
                run_model(model_name, plan, docs, fed, root, args.resume)
        finally:
            # Preserve the measurement traceback if exporting partial results also fails.
            active_error = sys.exc_info()[0] is not None
            try:
                export_bundle(root)
            except Exception as exc:
                if not active_error:
                    raise
                print(f"Partial trajectory export failed: {exc}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
