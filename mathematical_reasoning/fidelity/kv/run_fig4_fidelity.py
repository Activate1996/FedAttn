#!/usr/bin/env python3
"""Measure available global KV states for every participant and prefill layer."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import socket
import zipfile

from deviation_io import load_docs, make_sample_plan

METRIC_ID = "prefill_available_kv_mean_over_participants_blocks_v1"
from server_resources import resolve_data_jsonl, resolve_model_reference

MODELS = {"Qwen2.5-0.5B": 24, "Qwen2.5-1.5B": 28,
          "Qwen2.5-3B": 36, "Qwen2.5-7B": 28}
SPLITS = ["even", "even_question_last", "smart", "smart_question_last"]
SCHEMES = ["uni_fr", "uni_bk", "inc_gp", "dec_gp"]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def absolute(value, base):
    p = Path(value).expanduser()
    return (p if p.is_absolute() else base / p).resolve()


def immutable_json(path, value):
    path = Path(path)
    if path.exists():
        if read_json(path) != value:
            raise ValueError(f"Plan changed at {path}. Use a NEW --output-root.")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")


@contextmanager
def output_lock(root):
    root.mkdir(parents=True, exist_ok=True)
    path = root / "fig4_running.lock"
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise RuntimeError(f"{path} exists. Two jobs must use different output roots. "
                           "Remove a stale lock only after checking that its job has stopped.") from exc
    try:
        with os.fdopen(fd, "w") as f:
            json.dump({"pid": os.getpid(), "host": socket.gethostname()}, f)
        yield
    finally:
        path.unlink(missing_ok=True)


def ratios(values, label):
    if (not isinstance(values, list) or not values or
            any(type(v) not in (int, float) or not 0 < v <= 1 for v in values) or
            len(set(values)) != len(values)):
        raise ValueError(f"{label} must contain unique numeric ratios in (0,1]")
    return sorted(values)


def build_configs(settings, model):
    """The baseline is fixed-H FedAttn; a retention ratio of 1 does not imply CenAttn."""
    h = settings["H_by_model"][model]
    if type(h) is not int or not 1 <= h <= MODELS[model]:
        raise ValueError(f"Invalid H for {model}")
    configs = [{"name": "FedAttn", "comm_policy": "main"}]
    points = []
    for axis, field, policy in [("A", "ratio_comp", "sample_uni_comp"),
                                ("B", "ratio_comm", "sample_uni_comm")]:
        for value in ratios(settings[field + "_values"], field):
            name = "FedAttn" if value == 1 else f"{axis}_{value:g}"
            if value != 1:
                configs.append({"name": name, "comm_policy": policy, field: value})
            points.append({"axis": axis, "x_axis": field, "x_value": value, "config": name})
    values = settings.get("publisher_intervals_by_model", {}).get(model, list(range(1, h + 1)))
    if (not values or len(set(values)) != len(values) or
            any(type(v) is not int or not 1 <= v <= h for v in values)):
        raise ValueError(f"publisher_intervals_by_model[{model}] must be unique integers in [1,H]")
    for value in sorted(values):
        name = "FedAttn" if value == h else f"C_Hlast{value}"
        if value != h:
            configs.append({"name": name, "comm_policy": "uni_extra", "last_num_local_forward": value})
        points.append({"axis": "C", "x_axis": "last_num_local_forward", "x_value": value, "config": name})
    schemes = settings.get("schemes", SCHEMES)
    if not schemes or len(set(schemes)) != len(schemes) or set(schemes) - set(SCHEMES):
        raise ValueError(f"schemes must be unique values from {SCHEMES}")
    for scheme in schemes:
        name = "D_" + scheme
        configs.append({"name": name, "comm_policy": scheme})
        points.append({"axis": "D", "x_axis": "scheme", "x_value": scheme, "config": name})
    return configs, points


def prepare(settings_path, root, selected_models=None, selected_splits=None):
    settings_path = Path(settings_path).resolve()
    cfg = read_json(settings_path)
    base = settings_path.parent
    source = absolute(cfg["source_repo"], base)
    fed = source / "FedAttn" if (source / "FedAttn").is_dir() else source
    for relative in ("evaluate_gsm8k/main_evaluate_gsm8k.py", "Wrapper.py", "utils.py"):
        if not (fed / relative).is_file():
            raise FileNotFoundError(f"Original source missing: {fed / relative}")
    models = selected_models or list(cfg["models"])
    splits = selected_splits or cfg.get("splits", SPLITS)
    if not models or len(set(models)) != len(models) or set(models) - set(MODELS) or set(models) - set(cfg["models"]):
        raise ValueError("models must be unique canonical names present in settings")
    if not splits or len(set(splits)) != len(splits) or set(splits) - set(SPLITS):
        raise ValueError("Unknown or duplicate split names")
    if type(cfg.get("seed", 0)) is not int:
        raise ValueError("seed must be an integer")
    data = resolve_data_jsonl(cfg, base, root)
    sample = make_sample_plan(load_docs(data), cfg)
    if cfg.get("sample_plan_json"):
        previous = read_json(absolute(cfg["sample_plan_json"], base))
        if previous != sample:
            raise ValueError("Previous sample_plan_json differs in question selection or dataset order")
    sample_path = root / "sample_plan.json"
    panels = []
    manifests = []
    for model in models:
        dtype = cfg.get("dtype_by_model", {}).get(model, "bfloat16" if model == "Qwen2.5-7B" else "float32")
        if dtype not in ("float32", "float16", "bfloat16"):
            raise ValueError(f"Unsupported dtype: {dtype}")
        if model == "Qwen2.5-7B" and dtype == "float32":
            raise ValueError("7B must use 16-bit precision: bfloat16 (default) or float16")
        configs, points = build_configs(cfg, model)
        for split in splits:
            stem = f"fig4_{model}_{split}"
            manifest_path = root / "manifests" / (stem + ".json")
            hp = {"num_shots": 4, "num_clients": 4, "num_local_forward": cfg["H_by_model"][model],
                  "max_new_tokens": 256, "do_sample": False,
                  "ratio_comp": 1.0, "ratio_comm": 1.0}
            manifest = {"schema_version": 4, "metric_id": METRIC_ID, "figure": 4, "model": model, "num_layers": MODELS[model],
                        "source_repo": str(source), "model_path": resolve_model_reference(cfg["models"][model], base),
                        "data_jsonl": data, "sample_plan_json": str(sample_path),
                        "output_jsonl": str(root / (stem + ".jsonl")),
                        "device": cfg.get("device", "cuda:0"), "dtype": dtype, "seed": cfg.get("seed", 0),
                        "split_way": split, "hyperparams": hp, "configs": configs,
                        "prompt_prefix": cfg.get("prompt_prefix", "gsm8k_prompt_"),
                        "prompt_suffix": cfg.get("prompt_suffix", ".txt"),
                        "mechanism_layers": [], "reuse_verified_baseline": False,
                        "fidelity_metric": METRIC_ID}
            chunks = fed / "evaluate_gsm8k" / "data_process_token_chunks" / f"data_process_{model}_shot4_c4_{split}.json"
            if chunks.is_file():
                manifest["token_chunks_json"] = str(chunks.resolve())
            panel = {"model": model, "split_way": split, "dtype": dtype,
                     "H": hp["num_local_forward"], "records_file": stem + ".jsonl",
                     "manifest_file": "manifests/" + stem + ".json", "points": points,
                     "n_configs_including_reference": len(configs) + 1}
            panels.append(panel)
            manifests.append((manifest_path, manifest))
    plan = {"schema_version": 4, "metric_id": METRIC_ID, "models": models, "splits": splits,
            "sample_plan": sample, "panels": panels,
            "measurement": "Actual current-layer prefill K after RoPE and projected V; each participant available global token rows, zero-filled at unavailable rows; full Cen publisher KV reference",
            "normalization": "Joint K/V relative Frobenius over ALL valid global input tokens per participant per layer. Equal mean over participants, layers, then samples. Missing rows remain in the reference norm. Same metric for all four axes.",
            "metric_by_axis": {axis: "fidelity.layer_participant_mean_relative_kv_frobenius" for axis in "ABCD"},
            "notes": ["Attention masks follow the FedAttn causal and sparse attention implementation.",
                      "Ratio=1 and H_last=H are fixed-H FedAttn endpoints, not CenAttn.",
                      "Configuration grids include explicit full-retention and equal-interval endpoints.",
                      "All axes measure every participant and layer. Every distinct configuration runs its own prefill; only availability masks are zero-filled in the metric.",
                      "C may change participant KV availability even if causal prefill hidden states are unchanged. This is a KV-view metric, not post-block hidden-state error.",
                      "Sparse masks are constructed with the configured seed; this measurement does not generate answers."]}
    # Validate the complete plan before writing panel manifests.
    immutable_json(root / "fig4_plan.json", plan)
    immutable_json(sample_path, sample)
    for path, manifest in manifests:
        immutable_json(path, manifest)
    total = len(sample["sample_indices"]) * sum(p["n_configs_including_reference"] for p in panels)
    print(f"Shared samples ({len(sample['sample_indices'])}): {sample['sample_indices']}", flush=True)
    print(f"Prepared {len(panels)} model/split panels; {total} sample/config records (including CenAttn).", flush=True)
    for model in models:
        first = next(p for p in panels if p["model"] == model)
        print(f"  {model}: H={first['H']}, N=4, shots=4, dtype={first['dtype']}", flush=True)
    return plan


def export_bundle(root):
    from export_fig4_fidelity import export_results
    summary = export_results(root)
    bundle = root / "fig4_fidelity_plot_bundle.zip"
    members = [summary, root / "fig4_plan.json", root / "sample_plan.json"]
    members += sorted(root.glob("fig4_fidelity_*.csv"))
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in members:
            archive.write(path, path.name)
    print(f"Fig.4 points: {read_json(summary)['point_counts']} (partial points remain gaps)", flush=True)
    print(f"Summary: {summary}\nCopy to plotting host: {bundle}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", default=str(Path(__file__).with_name("settings_fig4_server212.json")))
    parser.add_argument("--models", nargs="+", choices=list(MODELS))
    parser.add_argument("--splits", nargs="+", choices=SPLITS)
    parser.add_argument("--output-root", help="Use a separate root for each parallel job / changed plan")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--export-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    settings_path = Path(args.settings).resolve()
    cfg = read_json(settings_path)
    root = absolute(args.output_root or cfg["output_root"], settings_path.parent)
    with output_lock(root):
        if args.export_only:
            if args.prepare_only:
                parser.error("--prepare-only and --export-only cannot be combined")
            export_bundle(root)
            return
        plan = prepare(settings_path, root, args.models, args.splits)
        if args.prepare_only:
            return
        import fidelity_backend as backend
        try:
            for model in plan["models"]:
                panels = [p for p in plan["panels"] if p["model"] == model]
                first = read_json(root / panels[0]["manifest_file"])
                runtime = backend.create_runtime(first)
                try:
                    for panel in panels:
                        print(f"Running Fig.4 / {model} / {panel['split_way']}", flush=True)
                        backend.run_manifest(root / panel["manifest_file"], generate=False,
                                             resume=args.resume, runtime=runtime)
                finally:
                    backend.close_runtime(runtime)
        finally:
            # Partial summaries retain missing points and remain marked as incomplete.
            export_bundle(root)


if __name__ == "__main__":
    main()
