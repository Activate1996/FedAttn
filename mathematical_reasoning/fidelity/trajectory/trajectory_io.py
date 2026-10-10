"""Validate and aggregate block trajectories using Python's standard library.

Every segmentation setting uses the same sample plan. Incomplete panels have
null means so comparisons use the same set of samples.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
from pathlib import Path
import statistics

SPLIT_LABELS = {
    "even": "TokAg", "even_question_last": "TokEx",
    "smart": "SemAg", "smart_question_last": "SemEx",
}
ARRAY_FIELDS = (
    "delta_traj_pub", "delta_traj_all", "absolute_traj_pub",
    "absolute_traj_all", "reference_norm_traj_pub", "reference_norm_traj_all",
)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _integer(value, name, minimum=0):
    _require(type(value) is int and value >= minimum,
             f"{name} must be an integer >= {minimum}; got {value!r}")
    return value


def _atomic_text(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def _csv_text(fieldnames, rows):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def _validate_row(row, model, metadata, settings, sample_plan, expected_ids, source):
    label = f"{source}: model={model}"
    for key, expected in (
        ("model", model), ("sample_set_sha256", sample_plan["sample_set_sha256"]),
        ("dtype", metadata["dtype"]), ("num_layers", metadata["num_layers"]),
        ("H", settings["H"]), ("num_clients", settings["num_clients"]),
        ("num_shots", settings["num_shots"]),
    ):
        _require(row.get(key) == expected,
                 f"{label}: {key} mismatch: {row.get(key)!r} != {expected!r}")
    _require(isinstance(row.get("run_id"), str) and row["run_id"],
             f"{label}: run_id must be a nonempty string")
    _require(row.get("split_way") in settings["splits"], f"{label}: unexpected split_way")
    _integer(row.get("sample_index"), f"{label}: sample_index")
    _require(row["sample_index"] in expected_ids, f"{label}: sample_index is outside the sample plan")
    questions = {entry["sample_index"]: entry for entry in sample_plan.get("sample_questions", [])}
    if row["sample_index"] in questions:
        for key in ("question_sha256", "answer_sha256"):
            _require(row.get(key) == questions[row["sample_index"]].get(key),
                     f"{label}: {key} differs from the planned sample")
    n_layers = _integer(row.get("num_layers"), f"{label}: num_layers", 1)
    indices = list(range(1, n_layers + 1))
    _require(row.get("block_indices") == indices, f"{label}: block_indices must equal 1..num_layers")
    global_blocks = row.get("global_blocks")
    _require(isinstance(global_blocks, list) and all(type(i) is int for i in global_blocks),
             f"{label}: global_blocks must be an integer list")
    _require(global_blocks == sorted(set(global_blocks)) and set(global_blocks) <= set(indices),
             f"{label}: global_blocks must be unique, increasing block indices")
    n_pub = _integer(row.get("n_publisher_tokens"), f"{label}: n_publisher_tokens", 1)
    n_all = _integer(row.get("n_all_tokens"), f"{label}: n_all_tokens", 1)
    _require(n_pub <= n_all, f"{label}: publisher token count exceeds all-token count")
    for field in ARRAY_FIELDS:
        array = row.get(field)
        _require(isinstance(array, list) and len(array) == n_layers,
                 f"{label}: {field} must have num_layers entries")
        _require(all(type(x) in (int, float) and math.isfinite(x) and x >= 0 for x in array),
                 f"{label}: {field} must contain finite, nonnegative numbers")
    for alias, field in (("delta_traj", "delta_traj_all"),
                         ("absolute_delta_traj", "absolute_traj_all")):
        if alias in row:
            _require(row[alias] == row[field], f"{label}: {alias} differs from {field}")
    return row


def _moments(values):
    """Population standard deviation describes the measured sample set, not uncertainty in its mean."""
    return statistics.fmean(values), statistics.pstdev(values)


def _aggregate_metric(rows, field, length, complete, global_blocks):
    if not complete:
        return {"mean": [None] * length, "std": [None] * length,
                "paired_increments": None}
    columns = [[row[field][i] for row in rows] for i in range(length)]
    moments = [_moments(column) for column in columns]
    increments = [[row[field][i] - row[field][i - 1] for row in rows]
                  for i in range(1, length)]
    increment_moments = [_moments(column) for column in increments]
    diagnostic = {
        "to_block_indices": list(range(2, length + 1)),
        "mean": [x[0] for x in increment_moments],
        "std": [x[1] for x in increment_moments],
        "fraction_negative": [sum(x < 0 for x in column) / len(column)
                              for column in increments],
    }
    for kind, want_global in (("global", True), ("local", False)):
        values = [value for i, column in enumerate(increments, start=2)
                  if (i in global_blocks) == want_global for value in column]
        diagnostic[kind] = {
            "n_sample_block_pairs": len(values),
            "mean_increment": statistics.fmean(values) if values else None,
            "fraction_negative": sum(x < 0 for x in values) / len(values) if values else None,
        }
    return {"mean": [x[0] for x in moments], "std": [x[1] for x in moments],
            "paired_increments": diagnostic}


def export_trajectories(output_root, plan_path=None):
    """Export trajectory_summary.json and two long-format CSV files in output_root.

    Returns summary, summary_path, sample_csv_path, and points_csv_path. Missing
    record files are valid incomplete runs. Malformed or conflicting data raises
    ValueError before any output is replaced. No inference or plotting is run.
    """
    root = Path(output_root).expanduser().resolve()
    plan_file = Path(plan_path).expanduser().resolve() if plan_path else root / "trajectory_plan.json"
    plan = json.loads(plan_file.read_text(encoding="utf-8"))
    _require(plan.get("schema_version") == 1, "Unsupported plan schema_version")
    settings = plan["settings"]
    sample_plan = plan["sample_plan"]
    expected = sample_plan["sample_indices"]
    _require(isinstance(expected, list) and expected, "sample_indices must be a nonempty list")
    for value in expected:
        _integer(value, "sample_indices entry")
    _require(len(expected) == len(set(expected)), "sample_indices must be unique")
    _require(sample_plan["n_expected_samples"] == len(expected), "n_expected_samples mismatch")
    _require(isinstance(sample_plan.get("sample_set_sha256"), str)
             and sample_plan["sample_set_sha256"], "Missing sample_set_sha256")
    _integer(settings.get("H"), "H", 1)
    _integer(settings.get("num_clients"), "num_clients", 1)
    _integer(settings.get("num_shots"), "num_shots")
    splits = settings["splits"]
    _require(isinstance(splits, list) and splits and len(splits) == len(set(splits)),
             "settings.splits must be a nonempty, unique list")
    _require(set(splits) <= set(SPLIT_LABELS), "Unknown split in settings.splits")
    summary = {
        "schema_version": 1, "sample_plan": sample_plan, "settings": settings,
        "metric_definition": {
            "relative": "Frobenius norm of hidden-state difference / max(centralized reference Frobenius norm, 1e-12)",
            "absolute": "Frobenius norm of hidden-state difference",
            "all": "All valid non-padding tokens, aligned to centralized token order",
            "pub": "Publisher token slice, aligned to centralized token order",
            "std": "Population standard deviation across planned samples; descriptive, not a confidence interval",
            "aggregation": "Per-sample norms first, then arithmetic mean across the complete fixed sample set",
            "increments": "Paired per-sample difference from the previous block; block 1 has no preceding measured block",
        }, "panels": {},
    }
    sample_csv_rows, point_csv_rows = [], []
    sample_header = ["model", "split_way", "split_label", "sample_index", "run_id",
                     "sample_set_sha256", "dtype", "H", "num_clients", "num_shots",
                     "n_publisher_tokens", "n_all_tokens", "block_index", "is_global"] + list(ARRAY_FIELDS)
    point_header = ["model", "split_way", "split_label", "dtype", "H", "num_clients",
                    "num_shots", "sample_set_sha256", "status", "n", "n_expected",
                    "block_index", "is_global"]
    for scope in ("all", "pub"):
        for metric in ("relative", "absolute"):
            point_header += [f"{scope}_{metric}_mean", f"{scope}_{metric}_std"]
    _require(isinstance(plan.get("models"), dict) and plan["models"], "Plan must contain models")
    for model, metadata in plan["models"].items():
        n_layers = _integer(metadata.get("num_layers"), f"{model}: num_layers", 1)
        dtype = metadata.get("dtype")
        _require(isinstance(dtype, str) and dtype, f"{model}: missing dtype")
        if model in settings.get("dtype_by_model", {}):
            _require(dtype == settings["dtype_by_model"][model], f"{model}: plan dtype mismatch")
        records_path = Path(metadata["records_file"])
        if not records_path.is_absolute():
            records_path = plan_file.parent / records_path
        records = {}
        actual_global_blocks = None
        model_run_id = None
        if records_path.exists():
            with records_path.open(encoding="utf-8") as stream:
                for line_number, line in enumerate(stream, start=1):
                    if not line.strip():
                        continue
                    source = f"{records_path}:{line_number}"
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"{source}: invalid JSON: {exc}") from exc
                    _require(isinstance(row, dict), f"{source}: record must be an object")
                    row = _validate_row(row, model, metadata, settings, sample_plan, set(expected), source)
                    if model_run_id is None:
                        model_run_id = row["run_id"]
                    _require(row["run_id"] == model_run_id,
                             f"{source}: records for one model have different run_id provenance")
                    if actual_global_blocks is None:
                        actual_global_blocks = row["global_blocks"]
                    _require(row["global_blocks"] == actual_global_blocks,
                             f"{source}: inconsistent global block schedule")
                    key = (row["split_way"], row["sample_index"])
                    _require(key not in records or records[key] == row,
                             f"{source}: conflicting duplicate model/split/sample record, including run provenance")
                    records[key] = row
        # Leave the observed schedule unset for missing runs rather than inferring it from H.
        global_blocks = actual_global_blocks if actual_global_blocks is not None else []
        summary["panels"][model] = {}
        for split in splits:
            rows = [records[(split, index)] for index in expected if (split, index) in records]
            missing = [index for index in expected if (split, index) not in records]
            complete = not missing
            panel = {
                "model": model, "split_way": split, "split_label": SPLIT_LABELS[split],
                "block_indices": list(range(1, n_layers + 1)), "global_blocks": global_blocks,
                "num_layers": n_layers, "dtype": dtype, "H": settings["H"],
                "run_id": model_run_id,
                "num_clients": settings["num_clients"], "num_shots": settings["num_shots"],
                "sample_set_sha256": sample_plan["sample_set_sha256"],
                "sample_indices": [row["sample_index"] for row in rows],
                "missing_sample_indices": missing, "n": len(rows), "n_expected": len(expected),
                "status": "complete" if complete else "incomplete", "metrics": {},
            }
            for scope in ("all", "pub"):
                panel["metrics"][scope] = {}
                for metric, prefix in (("relative", "delta"), ("absolute", "absolute")):
                    panel["metrics"][scope][metric] = _aggregate_metric(
                        rows, f"{prefix}_traj_{scope}", n_layers, complete, global_blocks)
            summary["panels"][model][split] = panel
            for row in rows:
                for i, block in enumerate(row["block_indices"]):
                    sample_point = {key: row[key] for key in sample_header
                                    if key in row and key not in ARRAY_FIELDS}
                    sample_point.update(split_label=SPLIT_LABELS[split], block_index=block,
                                        is_global=int(block in global_blocks))
                    sample_point.update({field: row[field][i] for field in ARRAY_FIELDS})
                    sample_csv_rows.append(sample_point)
            for i, block in enumerate(panel["block_indices"]):
                point = {key: panel[key] for key in point_header if key in panel}
                point.update(block_index=block, is_global=int(block in global_blocks))
                for scope in ("all", "pub"):
                    for metric in ("relative", "absolute"):
                        for stat in ("mean", "std"):
                            point[f"{scope}_{metric}_{stat}"] = panel["metrics"][scope][metric][stat][i]
                point_csv_rows.append(point)
    summary_path = root / "trajectory_summary.json"
    sample_csv_path = root / "trajectory_per_sample.csv"
    points_csv_path = root / "trajectory_points.csv"
    _atomic_text(summary_path, json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    _atomic_text(sample_csv_path, _csv_text(sample_header, sample_csv_rows))
    _atomic_text(points_csv_path, _csv_text(point_header, point_csv_rows))
    return {"summary": summary, "summary_path": str(summary_path),
            "sample_csv_path": str(sample_csv_path), "points_csv_path": str(points_csv_path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--plan", type=Path, help="Defaults to OUTPUT_ROOT/trajectory_plan.json")
    args = parser.parse_args()
    result = export_trajectories(args.output_root, args.plan)
    for name in ("summary_path", "sample_csv_path", "points_csv_path"):
        print(result[name])


if __name__ == "__main__":
    main()
