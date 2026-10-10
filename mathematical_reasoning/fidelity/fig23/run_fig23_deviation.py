#!/usr/bin/env python3
"""Measure Fig. 2/3 representation deviation using a shared sample plan.

Run with an explicit settings file:
  python run_fig23_deviation.py --settings settings.json --prepare-only
  python run_fig23_deviation.py --settings settings.json --resume

The --models, --splits, and --figures arguments select panels for execution.
The complete plan is retained so exports also represent unrun panels. This
runner measures prefill representations without generating answers. Relative
paths are resolved against the settings file.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import shutil
import zipfile

from deviation_io import load_docs, make_sample_plan, planned_points, resolve_sample_plan
from server_resources import resolve_data_jsonl, resolve_model_reference


SPLITS = ("even", "smart", "even_question_last", "smart_question_last")
MODEL_LAYERS = {"Qwen2.5-0.5B": 24, "Qwen2.5-1.5B": 28, "Qwen2.5-3B": 36, "Qwen2.5-7B": 28}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def resolve_path(value, base):
    path = Path(value).expanduser()
    return str((path if path.is_absolute() else base / path).resolve())


def write_identical_or_new(path, value):
    """Write a new plan or validate that an existing plan is identical."""
    path = Path(path)
    if path.exists():
        if read_json(path) != value:
            raise ValueError(f"Existing file conflicts with this plan: {path}. Use a new output_root for changed settings.")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents overwriting a file created by another process.
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def validated_grid(values, upper, label):
    if not isinstance(values, list) or not values:
        raise ValueError(f"{label} must be a nonempty list of the actual original plot values")
    if any(type(v) is not int or not 1 <= v <= upper for v in values):
        raise ValueError(f"{label} values must be integers in [1, {upper}]")
    if len(set(values)) != len(values):
        raise ValueError(f"{label} contains duplicate values")
    # Include x=1 once through the dense CenAttn reference.
    return sorted(set(values) - {1})


def discover_historical_metadata(results_root):
    """Read main-policy configuration metadata from saved result filenames.

    Use avg_acc records to define each panel's configuration grid. Fall back to
    per-client result filenames only when that panel has no avg_acc records,
    preserving the full grid whenever aggregate results are available.
    """
    from analyze_existing_outputs import parse_meta

    root = Path(results_root)
    if not root.is_dir():
        raise NotADirectoryError(f"results_root does not exist: {root}")
    records = []
    for path in sorted(root.rglob("results_*_client*.jsonl")):
        if path.parent.name != "main":
            continue
        meta = parse_meta(path)
        records.append(("raw", str(path), meta))
    for path in sorted(root.rglob("avg_acc_results_*.json")):
        if path.parent.name != "avg_acc" or path.parent.parent.name != "main":
            continue
        # Convert the aggregate filename to a per-client filename for metadata parsing.
        fake = path.parent.parent / (path.name[len("avg_acc_"):-len(".json")] + "_client0.jsonl")
        meta = parse_meta(fake)
        records.append(("avg_acc", str(path), meta))
    if not records:
        raise ValueError(f"No parseable original main-policy results or avg_acc files found under {root}")
    return records


def historical_grid(records, figure, model, split, layers):
    candidates = {"raw": [], "avg_acc": []}
    fixed_h = 9 if model == "Qwen2.5-3B" else 4
    for kind, path, meta in records:
        if (meta["model"] != model or meta["split_way"] != split or
                meta["policy"] != "main" or meta["do_sample"] or meta["max_new_tokens"] != 256):
            continue
        if figure == 2:
            matches = meta["num_shots"] == 4 and meta["num_clients"] == 4
            value = meta["num_local_forward"]
        else:
            matches = meta["num_shots"] == 8 and meta["num_local_forward"] == fixed_h
            value = meta["num_clients"]
            # The participant sweep includes at most eight participants.
            matches = matches and value <= 8
        if matches:
            candidates[kind].append((value, path))
    found = candidates["avg_acc"] or candidates["raw"]
    if not found:
        raise ValueError(
            f"No original plotted configurations found for Fig.{figure}, {model}, {split}. "
            "Check results_root, or set results_root=null and explicitly supply the original x grids. "
            "Panels are never silently omitted.")
    values = validated_grid(sorted({v for v, _ in found}), layers if figure == 2 else 8,
                            f"historical Fig.{figure}/{model}/{split}")
    return values, {"kind": "historical_avg_acc" if candidates["avg_acc"] else "historical_raw",
                    "files": sorted({p for _, p in found})}


def load_plot_grid(path):
    """Read and validate a saved panel configuration grid."""
    grid = read_json(path)
    panels = {}
    for panel in grid.get("panels", []):
        key = (panel["figure"], panel["model"], panel["split_way"])
        if key in panels:
            raise ValueError(f"Duplicate panel in plot_grid_json: {key}")
        axis = "num_local_forwards" if key[0] == 2 else "num_clients"
        if key[0] not in (2, 3) or panel.get("x_axis") != axis:
            raise ValueError(f"Invalid figure/x_axis in plot_grid_json: {key}")
        panels[key] = panel
    if not panels:
        raise ValueError("plot_grid_json contains no panels")
    return panels


def prepare(settings_path):
    settings_path = Path(settings_path).resolve()
    settings = read_json(settings_path)
    base = settings_path.parent
    for field in ("source_repo", "output_root", "models"):
        if field not in settings:
            raise ValueError(f"Missing settings field: {field}")
    source_repo = resolve_path(settings["source_repo"], base)
    source = Path(source_repo)
    fed = source / "FedAttn" if (source / "FedAttn").is_dir() else source
    if not (fed / "evaluate_gsm8k" / "main_evaluate_gsm8k.py").is_file():
        raise FileNotFoundError(f"source_repo must contain the original FedAttn/evaluate_gsm8k code: {source_repo}")
    models = settings["models"]
    if not isinstance(models, dict) or not models:
        raise ValueError("models must map original canonical model names to local model paths")
    for model, path in models.items():
        if model not in MODEL_LAYERS:
            raise ValueError(f"Unsupported model key {model!r}; expected one of {list(MODEL_LAYERS)}")
        if not isinstance(path, str) or not path:
            raise ValueError(f"Invalid model path or Hugging Face repo ID for {model}")
    figures = settings.get("figures", [2, 3])
    splits = settings.get("splits", list(SPLITS))
    if not figures or any(type(f) is not int or f not in (2, 3) for f in figures) or len(set(figures)) != len(figures):
        raise ValueError("figures must contain unique values selected from [2, 3]")
    if not splits or any(s not in SPLITS for s in splits) or len(set(splits)) != len(splits):
        raise ValueError(f"splits must contain unique values selected from {list(SPLITS)}")
    if settings.get("dtype", "float32") not in ("float32", "bfloat16", "float16"):
        raise ValueError("dtype must be float32, bfloat16 or float16, matching the historical run")
    output_root = Path(resolve_path(settings["output_root"], base))
    data_jsonl = resolve_data_jsonl(settings, base, output_root)
    docs = load_docs(data_jsonl)
    requested_sample_plan = make_sample_plan(docs, settings)
    sample_path = output_root / "sample_plan.json"
    if sample_path.exists():
        # Validate dataset content, ordering, and the requested sample selection.
        # Every model and input split uses the same sample indices.
        old_sample_plan = resolve_sample_plan(docs, {"sample_plan_json": str(sample_path)})
        if old_sample_plan != requested_sample_plan:
            raise ValueError(f"Requested sampling differs from {sample_path}; use another output_root")
    records = None
    plot_grid = None
    if settings.get("plot_grid_json"):
        plot_grid = load_plot_grid(resolve_path(settings["plot_grid_json"], base))
    elif settings.get("results_root"):
        records = discover_historical_metadata(resolve_path(settings["results_root"], base))
    elif (2 in figures and not isinstance(settings.get("fig2_h_values"), dict)) or (3 in figures and not isinstance(settings.get("fig3_n_values"), list)):
        raise ValueError("Without results_root, explicitly set fig2_h_values per model and fig3_n_values. No full-layer grid is guessed.")

    panels = []
    manifests = []
    for model, model_path in models.items():
        layers = MODEL_LAYERS[model]
        for figure in figures:
            for split in splits:
                if plot_grid is not None:
                    key = (figure, model, split)
                    if key not in plot_grid:
                        raise ValueError(f"Plot host grid is missing requested panel {key}; export all desired models/splits")
                    values = validated_grid(plot_grid[key]["x_values"], layers if figure == 2 else 8, str(key))
                    origin = {"kind": "plot_host_grid", "x_values": [1] + values}
                elif records is not None:
                    values, origin = historical_grid(records, figure, model, split, layers)
                else:
                    values = (settings["fig2_h_values"].get(model) if figure == 2 else settings["fig3_n_values"])
                    values = validated_grid(values, layers if figure == 2 else 8, f"Fig.{figure}/{model}")
                    origin = {"kind": "explicit_settings", "note": "Explicitly configured measurement grid."}
                hp = {"num_shots": 4 if figure == 2 else 8, "num_clients": 4,
                      "num_local_forward": 9 if model == "Qwen2.5-3B" else 4,
                      "max_new_tokens": 256, "do_sample": False}
                configs = []
                for value in values:
                    cfg = {"name": f"{'H' if figure == 2 else 'N'}{value}", "comm_policy": "main",
                           "num_local_forward" if figure == 2 else "num_clients": value}
                    n = 4 if figure == 2 else value
                    historical_chunks = fed / "evaluate_gsm8k" / "data_process_token_chunks" / (
                        f"data_process_{model}_shot{hp['num_shots']}_c{n}_{split}.json")
                    if historical_chunks.is_file():
                        cfg["token_chunks_json"] = str(historical_chunks.resolve())
                    configs.append(cfg)
                stem = f"fig{figure}_{model}_{split}"
                manifest_path = output_root / "manifests" / f"{stem}.json"
                output_path = output_root / f"{stem}_probe.jsonl"
                manifest = {"schema_version": 2, "source_repo": source_repo,
                            "model_path": resolve_model_reference(model_path, base), "model": model,
                            "num_layers": layers, "figure": figure, "split_way": split,
                            "data_jsonl": data_jsonl, "sample_plan_json": str(sample_path),
                            "output_jsonl": str(output_path), "device": settings.get("device", "cuda:0"),
                            "dtype": settings.get("dtype", "float32"), "seed": settings.get("seed", 0),
                            "hyperparams": hp, "configs": configs, "mechanism_layers": [],
                            "grid_source": origin}
                complete_configs = [{"name": "CenAttn", "comm_policy": "main", "num_local_forward": 1}] + configs
                panel = {"figure": figure, "model": model, "split_way": split,
                         "x_axis": "num_local_forwards" if figure == 2 else "num_clients",
                         "planned_points": planned_points(manifest, complete_configs),
                         "output_jsonl": str(output_path), "manifest_path": str(manifest_path)}
                panels.append(panel)
                manifests.append((manifest_path, manifest))

    # Check all panels for conflicts before writing new panel files.
    for path, manifest in manifests:
        if path.exists() and read_json(path) != manifest:
            raise ValueError(f"Existing manifest conflicts: {path}. Use a new output_root for a changed run.")
    batch_path = output_root / "batch_plan.json"
    if batch_path.exists():
        old_batch = read_json(batch_path)
        if old_batch.get("sample_plan") != requested_sample_plan:
            raise ValueError(f"Existing batch has a different sample plan: {batch_path}")
        merged = {(p["figure"], p["model"], p["split_way"]): p for p in old_batch["panels"]}
        for panel in panels:
            key = (panel["figure"], panel["model"], panel["split_way"])
            if key in merged and merged[key] != panel:
                raise ValueError(f"Existing batch panel conflicts: {key}")
            merged[key] = panel
        all_panels = list(merged.values())
    else:
        all_panels = panels
    if not sample_path.exists():
        write_identical_or_new(sample_path, requested_sample_plan)
    for path, manifest in manifests:
        write_identical_or_new(path, manifest)
    batch = {"schema_version": 2, "sample_plan": requested_sample_plan,
             "sample_plan_json": str(sample_path),
             "panels": sorted(all_panels, key=lambda p: (p["figure"], p["model"], p["split_way"]))}
    # Extend the batch plan with new panels while retaining existing entries.
    if not batch_path.exists() or read_json(batch_path) != batch:
        tmp = batch_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(batch, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        tmp.replace(batch_path)
    print(f"Shared samples ({len(requested_sample_plan['sample_indices'])}, zero-based): {requested_sample_plan['sample_indices']}")
    print(f"Prepared {len(panels)} panels; {sum(len(p['planned_points']) for p in panels)} points; batch plan: {batch_path}")
    return settings, panels, batch_path, output_root


def export_summary(output_root, batch_path):
    subprocess.run([sys.executable, str(Path(__file__).with_name("export_representation_deviation.py")),
                    "--results-root", str(output_root), "--plan", str(batch_path),
                    "--output-dir", str(output_root / "curves")], check=True)
    # Bundle the exported summaries and sample plan for local plotting.
    bundle = output_root / "plot_host_bundle"
    bundle.mkdir(exist_ok=True)
    for source in (output_root / "curves").glob("representation_deviation_*"):
        shutil.copy2(source, bundle / source.name)
    shutil.copy2(output_root / "sample_plan.json", bundle / "sample_plan.json")
    for name in ("export_representation_deviation.py",):
        source = Path(__file__).with_name(name)
        if source.exists():
            shutil.copy2(source, bundle / name)
    (bundle / "README.txt").write_text(
        "Portable plotting data for the Fig. 2 and Fig. 3 representation-deviation experiments.\n"
        "representation_deviation_summary.json contains panel arrays, per-point statistics, and sample counts.\n"
        "representation_deviation_points.csv contains one row per configuration;\n"
        "representation_deviation_per_sample.csv contains the individual measurements.\n"
        "get_representation_curve(..., figure=2 or 3, model=..., split_way=..., x_values=x)\n"
        "in export_representation_deviation.py reads means in the requested x order.\n"
        "Missing or incomplete points return NaN. sample_plan.json records the shared dataset row indices.\n",
        encoding="utf-8")
    archive = output_root / "plot_host_bundle.zip"
    temporary = output_root / "plot_host_bundle.zip.tmp"
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as handle:
        for source in sorted(bundle.iterdir()):
            if source.is_file():
                handle.write(source, "plot_host_bundle/" + source.name)
    temporary.replace(archive)
    print(f"Copy to plotting host: {archive}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", default=str(Path(__file__).with_name("settings_server212.json")))
    parser.add_argument("--prepare-only", action="store_true", help="Write and validate every panel/sample plan without loading a model")
    parser.add_argument("--models", nargs="+", help="Canonical names to execute; full plan remains saved")
    parser.add_argument("--splits", nargs="+", choices=SPLITS)
    parser.add_argument("--figures", nargs="+", type=int, choices=(2, 3))
    parser.add_argument("--resume", action="store_true", help="Skip already saved identical sample/config records")
    args = parser.parse_args()
    settings, panels, batch_path, output_root = prepare(args.settings)
    if args.models and set(args.models) - set(settings["models"]):
        parser.error(f"Unknown selected models: {sorted(set(args.models) - set(settings['models']))}")
    selected = [p for p in panels if (not args.models or p["model"] in args.models) and
                (not args.splits or p["split_way"] in args.splits) and
                (not args.figures or p["figure"] in args.figures)]
    if not selected:
        parser.error("Execution filters selected no panels")
    if args.prepare_only:
        export_summary(output_root, batch_path)
        return
    from probe_prefill import close_runtime, create_runtime, run_manifest
    try:
        for model in settings["models"]:
            group = [p for p in selected if p["model"] == model]
            if not group:
                continue
            print(f"Loading {model} once for {len(group)} panels", flush=True)
            runtime = create_runtime(read_json(group[0]["manifest_path"]))
            try:
                for panel in group:
                    print(f"Running Fig.{panel['figure']} / {model} / {panel['split_way']}", flush=True)
                    run_manifest(panel["manifest_path"], generate=False, resume=args.resume, runtime=runtime)
            finally:
                close_runtime(runtime)
    finally:
        # Retain saved sample records after interruption and leave incomplete means null.
        export_summary(output_root, batch_path)


if __name__ == "__main__":
    main()
