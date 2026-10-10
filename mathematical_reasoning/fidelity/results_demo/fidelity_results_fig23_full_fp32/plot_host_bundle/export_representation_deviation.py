#!/usr/bin/env python3
"""Export Fig. 2/3 representation-deviation measurements without accuracy data.

Only the Python standard library is required. Example:
  python export_representation_deviation.py --results-root probe_results --output-dir curves
  python export_representation_deviation.py --inputs 'probe_results/*_probe.jsonl' --output-dir curves

Every input requires its accompanying .jsonl.manifest.json file. Planned
points with zero or incomplete sample coverage have null means; the plotting
data reader returns NaN at those positions. The sample standard deviation and SEM
are descriptive summaries across the chosen questions, not design-based CIs
for a systematically selected subset.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import statistics
from pathlib import Path, PurePosixPath, PureWindowsPath

METRICS = ("absolute_frobenius", "relative_frobenius", "per_entry_rms")
META_FIELDS = ("figure", "model", "split_way", "x_axis")


def _read_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def _side_value(sidecar, key, default=None):
    """Read runtime metadata from provenance, top-level fields, or the manifest."""
    for source in (sidecar.get("provenance", {}), sidecar, sidecar.get("manifest", {})):
        if key in source:
            return source[key]
    return default


def _require(sidecar, key, context):
    result = _side_value(sidecar, key)
    if result is None:
        raise ValueError(f"{context}: missing sidecar field {key!r}; use the v2 probe")
    return result


def _int(value, label):
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    return value


def _finite(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite number")
    if value < 0:
        raise ValueError(f"{label} cannot be negative")
    return float(value)


def _question_map(sidecar):
    mapping = _side_value(sidecar, "sample_question_sha256", {})
    if isinstance(mapping, dict):
        result = {int(k): str(v) for k, v in mapping.items()}
    else:
        result = {}
    for row in _side_value(sidecar, "sample_questions", []):
        result[int(row["sample_index"])] = row["question_sha256"]
    return result


def _execution_signature(sidecar, point):
    manifest = sidecar.get("manifest", {})
    hp = dict(manifest.get("hyperparams", {}))
    config = next((c for c in manifest.get("configs", []) if c.get("name") == point["config"]), {})
    if point["config"] == "CenAttn":
        config = {"comm_policy": "main", "num_local_forward": 1}
    if hp:
        hp.update({key: value for key, value in config.items()
                   if key not in ("name", "comm_policy", "token_chunks_json", "restore_remote_layer",
                                  "point_key", "x_value", "figure", "model", "split_way", "x_axis")})
    return {"hyperparams": hp, "comm_policy": config.get("comm_policy", "main"),
            "dtype": _side_value(sidecar, "dtype"),
            "source_hashes": _side_value(sidecar, "source_hashes"),
            "probe_script_sha256": _side_value(sidecar, "probe_script_sha256"),
            "checkpoint_files": _side_value(sidecar, "checkpoint_files"),
            "torch_version": _side_value(sidecar, "torch_version"),
            "transformers_version": _side_value(sidecar, "transformers_version"),
            "fewshot_prompt_sha256": _side_value(sidecar, "fewshot_prompt_sha256"),
            "mask_seed": manifest.get("seed", 0),
            "token_chunks_json": config.get("token_chunks_json", manifest.get("token_chunks_json"))}


def _point_identity(point):
    return (int(point["figure"]), point["model"], point["split_way"], point["x_axis"], point["x_value"])


def _statistics(values, complete):
    if not complete:
        return {"mean": None, "std": None, "sem": None}
    n = len(values)
    sd = statistics.stdev(values) if n >= 2 else None
    return {"mean": statistics.fmean(values), "std": sd, "sem": sd / math.sqrt(n) if sd is not None else None}


def discover_inputs(inputs=None, results_root=None, allow_empty=False):
    paths = []
    for item in inputs or []:
        found = sorted(glob.glob(str(item), recursive=True))
        if not found:
            raise FileNotFoundError(f"No input matched {item!r}")
        paths.extend(Path(p).resolve() for p in found)
    if results_root is not None:
        root = Path(results_root)
        if not root.is_dir():
            raise NotADirectoryError(root)
        paths.extend(p.resolve() for p in root.rglob("*_probe.jsonl"))
        # A prepared run may be interrupted before its first data record.
        paths.extend(Path(str(p)[:-len(".manifest.json")]).resolve()
                     for p in root.rglob("*_probe.jsonl.manifest.json"))
    unique = sorted(set(paths))
    if not unique and not allow_empty:
        raise ValueError("No probe inputs found. Use --inputs for nonstandard filenames.")
    return unique


def _resolve_plan_path(value, plan_path, recorded_root):
    path = type(recorded_root)(value) if recorded_root is not None else Path(value)
    if not path.is_absolute():
        return plan_path.parent.joinpath(*path.parts).resolve()
    if recorded_root is not None:
        try:
            relative = path.relative_to(recorded_root)
        except ValueError:
            pass
        else:
            return plan_path.parent.joinpath(*relative.parts).resolve()
    return Path(value).resolve()


def export_results(paths, output_dir, plan_file=None):
    """Validate schema-v2 runs, write summary JSON and two CSVs, and return the summary.

    The optional plan_file specifies batch_plan.json. When present, panels
    without measurement records are also exported with missing values. Paths
    within a relocated run are resolved relative to the current plan directory.
    """
    virtual_sides = {}
    paths = {Path(path).resolve() for path in paths}
    if plan_file is not None:
        plan_path = Path(plan_file).resolve()
        batch = _read_json(plan_path)
        sample_plan = batch["sample_plan"]
        sample_plan_path = batch.get("sample_plan_json")
        recorded_sample_plan = ((PureWindowsPath(sample_plan_path)
                                 if PureWindowsPath(sample_plan_path).drive else PurePosixPath(sample_plan_path))
                                if sample_plan_path else None)
        recorded_root = (recorded_sample_plan.parent if recorded_sample_plan is not None
                         and recorded_sample_plan.is_absolute() else None)
        for panel in batch["panels"]:
            path = _resolve_plan_path(panel["output_jsonl"], plan_path, recorded_root)
            metadata = {key: panel[key] for key in META_FIELDS}
            manifest = {}
            manifest_path = panel.get("manifest_path")
            if manifest_path:
                manifest_path = _resolve_plan_path(manifest_path, plan_path, recorded_root)
                if manifest_path.is_file():
                    manifest = _read_json(manifest_path)
            virtual_sides[path] = {"manifest": manifest, "provenance": {
                **sample_plan, **metadata, "planned_points": panel["planned_points"],
                "run_id": f"not_started:{path}"}}
            paths.add(path)
    if not paths:
        raise ValueError("No runs or planned panels to export")
    plans = {}
    records = {}
    question_hashes = {}
    common_indices = None
    common_sample_hash = None
    runs = []
    seen_run_ids = {}
    for path in sorted(paths):
        path = Path(path).resolve()
        side_path = Path(str(path) + ".manifest.json")
        if side_path.is_file():
            side = _read_json(side_path)
            if path in virtual_sides:
                expected = virtual_sides[path]
                for field in (*META_FIELDS, "sample_indices", "sample_set_sha256", "planned_points"):
                    if _side_value(side, field) != _side_value(expected, field):
                        raise ValueError(f"{path}: actual run {field} differs from batch plan")
        elif path in virtual_sides and not path.exists():
            side = virtual_sides[path]
        else:
            raise FileNotFoundError(f"Missing probe sidecar: {side_path}")
        run_id = str(_require(side, "run_id", path))
        sample_hash = str(_require(side, "sample_set_sha256", path))
        indices = list(_require(side, "sample_indices", path))
        if not indices or len(set(indices)) != len(indices):
            raise ValueError(f"{path}: expected sample_indices must be nonempty and unique")
        for idx in indices:
            if _int(idx, "sample_index") < 0:
                raise ValueError(f"{path}: negative sample_index")
        indices = sorted(indices)
        if common_indices is None:
            common_indices, common_sample_hash = indices, sample_hash
        elif common_indices != indices or common_sample_hash != sample_hash:
            raise ValueError(f"{path}: mixed sample sets across panels; export only runs using the same sample plan")
        metadata = {key: _require(side, key, path) for key in META_FIELDS}
        metadata["figure"] = _int(metadata["figure"], "figure")
        expected_axis = {2: "num_local_forwards", 3: "num_clients"}.get(metadata["figure"])
        if metadata["x_axis"] != expected_axis:
            raise ValueError(f"{path}: figure and x_axis disagree")
        for idx, sha in _question_map(side).items():
            if idx not in indices:
                raise ValueError(f"{path}: sidecar question hash for an unplanned sample")
            if idx in question_hashes and question_hashes[idx] != sha:
                raise ValueError(f"{path}: changed question text at sample index {idx}")
            question_hashes[idx] = sha
        planned = _require(side, "planned_points", path)
        if not isinstance(planned, list) or not planned:
            raise ValueError(f"{path}: planned_points must be a nonempty list")
        local_plans = {}
        local_identities = set()
        for entry in planned:
            point = dict(metadata, **entry)
            for key in META_FIELDS:
                if point[key] != metadata[key]:
                    raise ValueError(f"{path}: planned point changes panel metadata {key}")
            point_key = str(point["point_key"])
            point["x_value"] = _int(point["x_value"], "x_value")
            if point["x_value"] < 1:
                raise ValueError(f"{path}: x_value must be positive")
            if "config" not in point:
                point["config"] = point.get("name")
            if not isinstance(point["config"], str):
                raise ValueError(f"{path}: planned point requires config name")
            if point_key in local_plans or _point_identity(point) in local_identities:
                raise ValueError(f"{path}: duplicate planned point")
            local_identities.add(_point_identity(point))
            local_plans[point_key] = point
            identity = _point_identity(point)
            point["_execution_signature"] = _execution_signature(side, point)
            if identity in plans:
                previous = plans[identity]
                if previous["point_key"] != point_key or previous["config"] != point["config"]:
                    raise ValueError(f"{path}: conflicting duplicate point at {identity}")
                previous_signature = previous["_execution_signature"]
                for field, value in point["_execution_signature"].items():
                    if value is not None and previous_signature.get(field) is not None and value != previous_signature[field]:
                        raise ValueError(f"{path}: incompatible {field} across duplicate point {identity}")
                # Combine repeated runs only when their numerical settings agree
                # for the same curve point.
                for field in ("hyperparams", "comm_policy", "dtype", "config_sha256"):
                    if field in point and field in previous and point[field] != previous[field]:
                        raise ValueError(f"{path}: conflicting {field} at {identity}")
            else:
                plans[identity] = point
        run_descriptor = {"metadata": metadata, "sample_set_sha256": sample_hash,
                          "planned_points": planned, "sample_indices": indices}
        if run_id in seen_run_ids and seen_run_ids[run_id] != run_descriptor:
            raise ValueError(f"{path}: reused run_id with different settings")
        seen_run_ids[run_id] = run_descriptor
        local_rows = {}
        if path.exists():
            with path.open(encoding="utf-8") as stream:
                for lineno, line in enumerate(stream, 1):
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"{path}:{lineno}: malformed JSON; do not aggregate a partial line") from exc
                    prefix = f"{path}:{lineno}"
                    if row.get("run_id") != run_id:
                        raise ValueError(f"{prefix}: run_id differs from sidecar")
                    for key, expected in metadata.items():
                        if row.get(key) != expected:
                            raise ValueError(f"{prefix}: {key} differs from sidecar")
                    if row.get("sample_set_sha256") != sample_hash or row.get("n_expected_samples") != len(indices):
                        raise ValueError(f"{prefix}: sample plan differs from sidecar")
                    point_key = row.get("point_key")
                    if point_key not in local_plans:
                        raise ValueError(f"{prefix}: unplanned point {point_key!r}")
                    point = local_plans[point_key]
                    if row.get("config") != point["config"] or row.get("x_value") != point["x_value"]:
                        raise ValueError(f"{prefix}: point config or x_value differs from plan")
                    idx = _int(row.get("sample_index"), f"{prefix}: sample_index")
                    if idx not in indices:
                        raise ValueError(f"{prefix}: unplanned sample index {idx}")
                    sha = row.get("question_sha256")
                    if not isinstance(sha, str) or not sha:
                        raise ValueError(f"{prefix}: question_sha256 is required")
                    if idx in question_hashes and question_hashes[idx] != sha:
                        raise ValueError(f"{prefix}: changed question text at sample index {idx}")
                    question_hashes[idx] = sha
                    measurements = row.get("fidelity", {}).get("all_tokens", {})
                    clean = {m: _finite(measurements.get(m), f"{prefix}: {m}") for m in METRICS}
                    for name in ("n_tokens", "reference_frobenius"):
                        if name in measurements:
                            clean[name] = _finite(measurements[name], f"{prefix}: {name}")
                    expected_hp = point["_execution_signature"]["hyperparams"]
                    if expected_hp and row.get("hyperparams") != expected_hp:
                        raise ValueError(f"{prefix}: hyperparams differ from planned configuration")
                    if row.get("comm_policy") != point["_execution_signature"]["comm_policy"]:
                        raise ValueError(f"{prefix}: comm_policy differs from planned configuration")
                    identity = _point_identity(point)
                    key = (identity, idx)
                    # Compare masks, precision, and inputs even when the scalar deviations
                    # happen to be equal.
                    signature_fields = ("question_sha256", "answer_sha256", "hyperparams", "comm_policy", "input_ids_sha256",
                                        "token_chunks_sha256", "prefill_masks_sha256", "dtype", "config_sha256")
                    signature = {name: row.get(name) for name in signature_fields}
                    signature["all_tokens"] = clean
                    if key in records:
                        if records[key]["signature"] != signature:
                            raise ValueError(f"{prefix}: conflicting duplicate measurement for {identity}, sample {idx}")
                        records[key]["source_files"].add(str(path))
                    else:
                        records[key] = {"signature": signature, "metrics": clean, "source_files": {str(path)},
                                        "question_sha256": sha, "run_id": run_id}
                    local_rows[(point_key, idx)] = True
        run_points = []
        for key, point in local_plans.items():
            done = sorted(idx for p, idx in local_rows if p == key)
            missing = sorted(set(indices) - set(done))
            run_points.append({"point_key": key, "x_value": point["x_value"], "config": point["config"],
                               "status": "complete" if not missing else "incomplete" if done else "missing",
                               "n": len(done), "n_expected": len(indices), "missing_sample_indices": missing})
        runs.append({"run_id": run_id, "input_file": str(path), "started": side_path.is_file(), "points": run_points})
    point_summaries, per_sample = [], []
    panels = {}
    for identity, point in sorted(plans.items()):
        measured = {idx: records[(identity, idx)] for idx in common_indices if (identity, idx) in records}
        missing = sorted(set(common_indices) - set(measured))
        complete = not missing
        summary = {key: point[key] for key in (*META_FIELDS, "x_value", "config", "point_key")}
        summary.update(status="complete" if complete else "incomplete" if measured else "missing",
                       n=len(measured), n_expected=len(common_indices), missing_sample_indices=missing)
        for metric in METRICS:
            summary[metric] = _statistics([v["metrics"][metric] for v in measured.values()], complete)
        point_summaries.append(summary)
        panel = panels.setdefault(str(point["figure"]), {}).setdefault(point["model"], {}).setdefault(point["split_way"],
                    {"x_axis": point["x_axis"], "x_values": [], "point_keys": [], "status": [], "n": [], "n_expected": [],
                     **{m: {"mean": [], "std": [], "sem": []} for m in METRICS}})
        for src, dst in (("x_value", "x_values"), ("point_key", "point_keys"), ("status", "status"), ("n", "n"), ("n_expected", "n_expected")):
            panel[dst].append(summary[src])
        for metric in METRICS:
            for stat in ("mean", "std", "sem"):
                panel[metric][stat].append(summary[metric][stat])
        for idx, item in measured.items():
            per_sample.append({**{k: point[k] for k in (*META_FIELDS, "x_value", "config", "point_key")},
                               "sample_index": idx, "question_sha256": item["question_sha256"],
                               "sample_set_sha256": common_sample_hash, **item["metrics"],
                               "source_files": "|".join(sorted(item["source_files"]))})
    output = {"schema_version": 2, "measurement": "final Transformer block, before final RMSNorm, all valid input tokens",
              "sample_set_sha256": common_sample_hash, "sample_indices": common_indices,
              "n_expected_samples": len(common_indices), "statistics_note":
              "Arithmetic mean across the selected questions. std uses ddof=1; sem=std/sqrt(n). These are descriptive, not design-based confidence intervals. Incomplete points have null statistics.",
              "panels": panels, "points": point_summaries, "runs": runs}
    dest = Path(output_dir)
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "representation_deviation_summary.json").write_text(json.dumps(output, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    raw_fields = [*META_FIELDS, "x_value", "config", "point_key", "sample_index", "question_sha256", "sample_set_sha256",
                  *METRICS, "n_tokens", "reference_frobenius", "source_files"]
    _write_csv(dest / "representation_deviation_per_sample.csv", per_sample, raw_fields)
    summary_fields = [*META_FIELDS, "x_value", "config", "point_key", "status", "n", "n_expected", "missing_sample_indices",
                      *(f"{metric}_{stat}" for metric in METRICS for stat in ("mean", "std", "sem"))]
    flat_summary = []
    for point in point_summaries:
        flat = {k: v for k, v in point.items() if k not in METRICS}
        flat["missing_sample_indices"] = json.dumps(flat["missing_sample_indices"])
        for metric in METRICS:
            flat.update({f"{metric}_{stat}": val for stat, val in point[metric].items()})
        flat_summary.append(flat)
    _write_csv(dest / "representation_deviation_points.csv", flat_summary, summary_fields)
    return output


def _write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def get_representation_curve(summary_file, model, split_way, figure, x_values, metric="relative_frobenius"):
    """Return mean deviations in the requested x order; missing points are NaN.

    model is a model name such as 'Qwen2.5-3B'; split_way is 'even', 'smart',
    'even_question_last', or 'smart_question_last'; figure is 2 or 3.
    The metric uses all_tokens representations and requires no accuracy data.
    """
    if metric not in METRICS:
        raise ValueError(f"metric must be one of {METRICS}")
    data = _read_json(summary_file) if not isinstance(summary_file, dict) else summary_file
    panel = data.get("panels", {}).get(str(figure), {}).get(model, {}).get(split_way)
    if panel is None:
        return [math.nan for _ in x_values]
    mapping = dict(zip(panel["x_values"], panel[metric]["mean"]))
    return [float(mapping[x]) if x in mapping and mapping[x] is not None else math.nan for x in x_values]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="*", help="Explicit probe files or quoted globs; no directory auto-discovery")
    parser.add_argument("--results-root", help="Recursively find only *_probe.jsonl and their sidecars")
    parser.add_argument("--plan", help="Optional batch_plan.json: include wholly unstarted panels as missing")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    if not args.inputs and not args.results_root and not args.plan:
        parser.error("specify --inputs, --results-root, or --plan")
    try:
        result = export_results(discover_inputs(args.inputs, args.results_root, allow_empty=bool(args.plan)), args.output_dir, args.plan)
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(2, f"error: {exc}\n")
    counts = {status: sum(p["status"] == status for p in result["points"]) for status in ("complete", "incomplete", "missing")}
    print(f"Saved {len(result['points'])} planned points: {counts}; selected samples={result['n_expected_samples']}")
    print(Path(args.output_dir) / "representation_deviation_summary.json")


if __name__ == "__main__":
    main()
