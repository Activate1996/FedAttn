#!/usr/bin/env python3
"""Validate prefill KV records and export means over the fixed sample set.

Only Python's standard library is required. Incomplete points have null means
so comparisons use the same set of questions.
"""
from __future__ import annotations

import argparse
import csv
from contextlib import contextmanager
import io
import json
import math
import os
from pathlib import Path
import statistics

SPLIT_LABELS = {"even": "TokAg", "even_question_last": "TokEx",
                "smart": "SemAg", "smart_question_last": "SemEx"}
AXES = {"A": "ratio_comp", "B": "ratio_comm",
        "C": "last_num_local_forward", "D": "scheme"}
METRIC_ID = "prefill_available_kv_mean_over_participants_blocks_v1"
PRIMARY_FIELD = "fidelity.layer_participant_mean_relative_kv_frobenius"
METRIC_FIELDS = {axis: PRIMARY_FIELD for axis in AXES}
MODEL_LAYERS = {"Qwen2.5-0.5B": 24, "Qwen2.5-1.5B": 28, "Qwen2.5-3B": 36, "Qwen2.5-7B": 28}
LAYER_PARTICIPANT_COLUMNS = ["metric_id", "model", "split_way", "dtype", "H", "config", "sample_index", "block_index", "participant_index", "is_publisher", "n_tokens", "n_kv_heads", "head_dim", "k_reference_frobenius", "v_reference_frobenius", "n_available_tokens", "n_missing_tokens", "coverage", "relative_frobenius", "absolute_frobenius", "reference_frobenius", "per_entry_rms", "k_relative_frobenius", "v_relative_frobenius", "available_error_squared", "missing_reference_squared", "total_reference_squared", "run_id", "sample_set_sha256"]
# Norms are evaluated in FP32 even when model inference uses BF16.
NORM_REL_TOL = 2e-5
NORM_ABS_TOL = 1e-7


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _atomic_text(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def _csv_text(columns, rows):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


@contextmanager
def _atomic_csv(path, columns):
    """Stream layer-participant tables without retaining all rows in memory."""
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
            writer.writeheader()
            yield writer
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _finite_nonnegative(value, label):
    require(type(value) in (int, float) and math.isfinite(value) and value >= 0,
            f"{label} must be a finite nonnegative number")
    return value


def _validate_finite(value, label):
    if isinstance(value, dict):
        for key, child in value.items():
            _validate_finite(child, f"{label}.{key}")
    elif isinstance(value, list):
        for i, child in enumerate(value):
            _validate_finite(child, f"{label}[{i}]")
    elif isinstance(value, float):
        require(math.isfinite(value), f"{label} contains a nonfinite value")


def _close(actual, expected, label):
    require(math.isclose(actual, expected, rel_tol=NORM_REL_TOL, abs_tol=NORM_ABS_TOL),
            f"{label}: {actual!r} disagrees with expected {expected!r}")


def _validate_identity(value, label, *, allow_sample_plan_version=False):
    # Expanding the sample plan can set the provenance schema version to 2.
    # This exception applies only to provenance; manifests and records require
    # KV schema 4 and must pass the KV and numerical validation below.
    # Preserve saved provenance and run_id to maintain resume compatibility.
    valid_schema = value.get("schema_version") == 4 or (
        allow_sample_plan_version and value.get("schema_version") == 2
    )
    require(valid_schema and value.get("metric_id") == METRIC_ID,
            f"{label}: expected KV-availability schema 4 and metric_id={METRIC_ID}; old hidden-state records cannot be reused, rerun in a new output directory")


def _validate_kv_layer(layer, n_clients, config, label):
    reference = layer.get("reference_kv")
    require(isinstance(reference, dict), f"{label}: missing reference_kv")
    for field in ("n_tokens", "n_kv_heads", "head_dim"):
        require(type(reference.get(field)) is int and reference[field] > 0,
                f"{label}: invalid reference_kv.{field}")
    for field in ("reference_frobenius", "k_reference_frobenius", "v_reference_frobenius"):
        _finite_nonnegative(reference.get(field), f"{label}: reference_kv.{field}")
    reference_norm = reference["reference_frobenius"]
    k_norm, v_norm = reference["k_reference_frobenius"], reference["v_reference_frobenius"]
    _close(reference_norm, math.hypot(k_norm, v_norm), f"{label}: joint K/V reference norm")
    participants = layer.get("participant_kv")
    require(isinstance(participants, list) and len(participants) == n_clients and
            all(isinstance(participant, dict) for participant in participants),
            f"{label}: each layer must contain all participant_kv entries")
    require([p.get("participant_index") for p in participants] == list(range(n_clients)),
            f"{label}: participants must have consecutive zero-based indices")
    n_tokens = reference["n_tokens"]
    n_entries = 2 * n_tokens * reference["n_kv_heads"] * reference["head_dim"]
    for i, participant in enumerate(participants):
        tag = f"{label}: participant {i}"
        require(type(participant.get("is_publisher")) is bool and participant["is_publisher"] == (i == n_clients - 1),
                f"{tag}: only the last participant must be marked publisher")
        require(type(participant.get("n_tokens")) is int and participant["n_tokens"] == n_tokens,
                f"{tag}: every participant must use the same complete global token axis")
        available = participant.get("n_available_tokens")
        require(type(available) is int and 0 <= available <= n_tokens,
                f"{tag}: invalid number of available tokens")
        for field in ("coverage", "relative_frobenius", "absolute_frobenius", "reference_frobenius",
                      "per_entry_rms", "k_relative_frobenius", "v_relative_frobenius",
                      "available_error_squared", "missing_reference_squared", "total_reference_squared"):
            _finite_nonnegative(participant.get(field), f"{tag}: {field}")
        require(participant["coverage"] <= 1, f"{tag}: coverage exceeds one")
        _close(participant["coverage"], available / n_tokens, f"{tag}: availability coverage")
        _close(participant["reference_frobenius"], reference_norm, f"{tag}: complete Cen reference norm")
        total = participant["total_reference_squared"]
        missing = participant["missing_reference_squared"]
        available_error = participant["available_error_squared"]
        _close(total, reference_norm ** 2, f"{tag}: total reference squared norm")
        require(missing <= total or math.isclose(missing, total, rel_tol=NORM_REL_TOL, abs_tol=NORM_ABS_TOL),
                f"{tag}: missing reference energy exceeds total reference energy")
        difference = participant["absolute_frobenius"]
        _close(difference, math.sqrt(available_error + missing), f"{tag}: error decomposition")
        _close(participant["relative_frobenius"], difference / max(reference_norm, 1e-12),
               f"{tag}: relative KV deviation")
        _close(participant["per_entry_rms"], difference / math.sqrt(n_entries),
               f"{tag}: RMS over the complete K and V tensors")
        k_error = participant["k_relative_frobenius"] * max(k_norm, 1e-12)
        v_error = participant["v_relative_frobenius"] * max(v_norm, 1e-12)
        _close(difference, math.hypot(k_error, v_error), f"{tag}: separate K/V deviations")
        if available == 0:
            _close(available_error, 0, f"{tag}: no available token can contribute value error")
            _close(missing, total, f"{tag}: an empty view must omit the entire reference")
        if available == n_tokens:
            _close(missing, 0, f"{tag}: a complete view cannot omit reference tokens")
        if config == "CenAttn":
            require(available == n_tokens and participant["coverage"] == 1 and
                    all(participant[field] == 0 for field in
                        ("absolute_frobenius", "relative_frobenius", "per_entry_rms",
                         "k_relative_frobenius", "v_relative_frobenius", "available_error_squared", "missing_reference_squared")),
                    f"{tag}: CenAttn must have full KV coverage and exactly zero self-deviation")
    value = _finite_nonnegative(layer.get("participant_mean_relative_kv_frobenius"), f"{label}: participant KV mean")
    _close(value, statistics.fmean(p["relative_frobenius"] for p in participants),
           f"{label}: equal-weight participant KV mean")


def _validate_layers(fidelity, n_clients, n_layers, config, label):
    layers = fidelity.get("kv_layers")
    require(type(fidelity.get("layer_count")) is int and fidelity["layer_count"] == n_layers and
            isinstance(layers, list) and len(layers) == n_layers,
            f"{label}: missing/incompatible all-layer KV fidelity; rerun the KV-availability toolkit in a new output directory")
    require(all(isinstance(layer, dict) for layer in layers) and
            [layer.get("block_index") for layer in layers] == list(range(1, n_layers + 1)),
            f"{label}: expected each decoder block exactly once, indexed from 1")
    shape = None
    for layer in layers:
        layer_label = f"{label}: block {layer['block_index']}"
        _validate_kv_layer(layer, n_clients, config, layer_label)
        current_shape = tuple(layer["reference_kv"][field] for field in ("n_tokens", "n_kv_heads", "head_dim"))
        require(shape is None or shape == current_shape,
                f"{layer_label}: global KV shape changes between blocks")
        shape = current_shape
    participant_means = fidelity.get("participant_layer_means")
    require(isinstance(participant_means, list) and len(participant_means) == n_clients and
            all(isinstance(participant, dict) for participant in participant_means) and
            [p.get("participant_index") for p in participant_means] == list(range(n_clients)),
            f"{label}: invalid participant_layer_means")
    for i, participant in enumerate(participant_means):
        require(type(participant.get("is_publisher")) is bool and participant["is_publisher"] == (i == n_clients - 1),
                f"{label}: invalid publisher identity in participant_layer_means")
        value = _finite_nonnegative(participant.get("relative_kv_frobenius"), f"{label}: participant layer mean")
        _close(value, statistics.fmean(layer["participant_kv"][i]["relative_frobenius"] for layer in layers),
               f"{label}: equal-weight participant layer mean")
    value = _finite_nonnegative(fidelity.get("layer_participant_mean_relative_kv_frobenius"), f"{label}: layer-participant KV mean")
    _close(value, statistics.fmean(layer["participant_mean_relative_kv_frobenius"] for layer in layers),
           f"{label}: equal-weight mean over all participants and all blocks")


def _metric_value(row, metric_field):
    value = row
    for key in metric_field.split("."):
        value = value[key]
    return value


def _aggregate(rows, expected, metric_field=PRIMARY_FIELD, value_fn=None):
    present = {row["sample_index"] for row in rows}
    missing = [i for i in expected if i not in present]
    values = [(value_fn(row) if value_fn is not None else _metric_value(row, metric_field)) for row in rows]
    complete = not missing
    return {
        "n": len(values), "n_expected": len(expected),
        "status": "complete" if complete else ("incomplete" if values else "missing"),
        "mean": statistics.fmean(values) if complete else None,
        "std": statistics.pstdev(values) if complete else None,
        "sem": statistics.stdev(values) / math.sqrt(len(values))
               if complete and len(values) > 1 else None,
        "missing_sample_indices": missing,
    }


def _read_panel(root, panel, sample_plan):
    expected = sample_plan["sample_indices"]
    expected_ids = set(expected)
    # The FedAttn baseline is measured even when the requested grid omits
    # retention=1 for A/B or interval=H for C.
    expected_configs = {point["config"] for point in panel["points"]} | {"CenAttn", "FedAttn"}
    records_path = Path(panel["records_file"])
    if not records_path.is_absolute():
        records_path = root / records_path
    provenance_path = Path(str(records_path) + ".manifest.json")
    rows = {}
    run_id = None
    n_clients = None
    n_layers = MODEL_LAYERS[panel["model"]]
    manifest_file = panel.get("manifest_file")
    if manifest_file:
        planned_manifest_path = root / manifest_file
        if planned_manifest_path.exists():
            planned_manifest = json.loads(planned_manifest_path.read_text(encoding="utf-8"))
            _validate_identity(planned_manifest, str(planned_manifest_path))
            n_clients = planned_manifest["hyperparams"]["num_clients"]
            require(type(n_clients) is int and n_clients >= 2, f"{planned_manifest_path}: invalid participant count")
            require(type(planned_manifest.get("num_layers")) is int and planned_manifest["num_layers"] == n_layers,
                    f"{planned_manifest_path}: num_layers differs from the model depth {n_layers}")
    if provenance_path.exists():
        sidecar = json.loads(provenance_path.read_text(encoding="utf-8"))
        manifest, provenance = sidecar["manifest"], sidecar["provenance"]
        _validate_identity(manifest, f"{provenance_path}: manifest")
        _validate_identity(provenance, f"{provenance_path}: provenance", allow_sample_plan_version=True)
        recorded_n_clients = manifest.get("hyperparams", {}).get("num_clients")
        require(type(recorded_n_clients) is int and recorded_n_clients >= 2,
                f"{provenance_path}: invalid hyperparams.num_clients")
        require(n_clients is None or n_clients == recorded_n_clients,
                f"{provenance_path}: participant count differs from the planned manifest")
        n_clients = recorded_n_clients
        require(type(manifest.get("num_layers")) is int and manifest["num_layers"] == n_layers,
                f"{provenance_path}: missing/incorrect num_layers for {panel['model']}; rerun the all-layer measurement")
        for key in ("model", "split_way", "dtype"):
            require(manifest.get(key) == panel[key], f"{provenance_path}: {key} mismatch")
        for key in ("sample_indices", "sample_set_sha256", "n_expected_samples"):
            require(provenance.get(key) == sample_plan[key],
                    f"{provenance_path}: {key} differs from the planned sample set")
        require({p["config"] for p in provenance["planned_points"]} == expected_configs,
                f"{provenance_path}: planned config names differ")
        require({c["name"] for c in manifest["configs"]} | {"CenAttn"} == expected_configs,
                f"{provenance_path}: manifest config names differ")
        run_id = provenance.get("run_id")
        require(isinstance(run_id, str) and run_id, f"{provenance_path}: missing run_id")
    if records_path.exists() and records_path.stat().st_size:
        require(provenance_path.exists(), f"{records_path}: missing provenance sidecar")
        questions = {q["sample_index"]: q for q in sample_plan.get("sample_questions", [])}
        with records_path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                label = f"{records_path}:{line_number}"
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{label}: invalid JSON; inspect an interrupted final write") from exc
                require(isinstance(row, dict), f"{label}: expected an object")
                _validate_identity(row, label)
                index, config = row.get("sample_index"), row.get("config")
                require(type(index) is int and index in expected_ids,
                        f"{label}: unexpected sample_index {index!r}")
                require(config in expected_configs, f"{label}: unexpected config {config!r}")
                for key, value in (("run_id", run_id),
                                   ("sample_set_sha256", sample_plan["sample_set_sha256"]),
                                   ("n_expected_samples", len(expected))):
                    require(row.get(key) == value, f"{label}: {key} mismatch")
                for key in ("model", "split_way"):
                    if key in row:
                        require(row[key] == panel[key], f"{label}: {key} mismatch")
                if index in questions:
                    for key in ("question_sha256", "answer_sha256"):
                        require(row.get(key) == questions[index].get(key),
                                f"{label}: {key} mismatch")
                require(isinstance(row.get("input_ids_sha256"), str) and row["input_ids_sha256"],
                        f"{label}: missing input token identity")
                fidelity = row.get("fidelity")
                require(isinstance(fidelity, dict), f"{label}: missing KV fidelity")
                _validate_finite(fidelity, f"{label}: fidelity")
                _validate_layers(fidelity, n_clients, n_layers, config, label)
                key = (index, config)
                require(key not in rows or rows[key] == row,
                        f"{label}: conflicting duplicate sample/config record")
                rows[key] = row
    for (index, config), row in rows.items():
        reference = rows.get((index, "CenAttn"))
        require(reference is not None, f"{records_path}: sample {index} has no saved CenAttn reference")
        require(row["input_ids_sha256"] == reference["input_ids_sha256"],
                f"{records_path}: input token IDs change for sample {index}, {config}")
        for layer, reference_layer in zip(row["fidelity"]["kv_layers"], reference["fidelity"]["kv_layers"]):
            saved_ref, cen_ref = layer["reference_kv"], reference_layer["reference_kv"]
            tag = f"{records_path}: block {layer['block_index']}, sample {index}, {config}"
            for field in ("n_tokens", "n_kv_heads", "head_dim"):
                require(saved_ref[field] == cen_ref[field], f"{tag}: centralized KV {field} changes")
            for field in ("reference_frobenius", "k_reference_frobenius", "v_reference_frobenius"):
                _close(saved_ref[field], cen_ref[field], f"{tag}: centralized KV {field}")
    return rows, run_id, n_clients, n_layers


def export_results(output_root):
    """Export the mean over all layers and participants, with diagnostic CSVs."""
    root = Path(output_root).expanduser().resolve()
    with _atomic_csv(root / "fig4_fidelity_layer_participants.csv", LAYER_PARTICIPANT_COLUMNS) as writer:
        return _export_results(root, writer)


def _export_results(output_root, layer_writer):
    root = Path(output_root).expanduser().resolve()
    plan = json.loads((root / "fig4_plan.json").read_text(encoding="utf-8"))
    _validate_identity(plan, "Fig.4 plan")
    sample = plan["sample_plan"]
    expected = sample["sample_indices"]
    require(isinstance(expected, list) and expected and
            all(type(i) is int and i >= 0 for i in expected) and len(expected) == len(set(expected)),
            "sample_indices must be nonempty, unique, nonnegative integers")
    require(sample["n_expected_samples"] == len(expected), "Sample count mismatch")
    require(isinstance(sample.get("sample_set_sha256"), str) and sample["sample_set_sha256"],
            "Missing sample_set_sha256")
    require(isinstance(plan.get("models"), list) and plan["models"] and
            len(plan["models"]) == len(set(plan["models"])) and set(plan["models"]) <= set(MODEL_LAYERS), "Invalid model list")
    require(isinstance(plan.get("splits"), list) and plan["splits"] and
            len(plan["splits"]) == len(set(plan["splits"])) and set(plan["splits"]) <= set(SPLIT_LABELS),
            "Invalid split list")
    expected_panels = {(m, s) for m in plan["models"] for s in plan["splits"]}
    seen_panels = set()
    summary = {
        "schema_version": 4, "metric_id": METRIC_ID, "figure": 4, "sample_plan": sample,
        "models": plan["models"], "splits": plan["splits"],
        "metric_definition": {
            "field": PRIMARY_FIELD,
            "formula": "mean_s mean_l mean_i sqrt(||K_available[s,l,i]-K_Cen[s,l]||_F^2 + ||V_available[s,l,i]-V_Cen[s,l]||_F^2) / max(sqrt(||K_Cen[s,l]||_F^2 + ||V_Cen[s,l]||_F^2), 1e-12)",
            "by_axis": {axis: {"field": PRIMARY_FIELD} for axis in AXES},
            "representation": "Post-RoPE K and projected V at the attention input of each decoder block, using the native KV heads",
            "tokens": "The same complete global valid prefill-token axis for every participant; padding and generated answers excluded",
            "reference": "CenAttn publisher complete global K and V at the corresponding block",
            "unavailable_tokens": "Unavailable KV vectors are zero-filled for this diagnostic only; model forward and causal attention masks remain unchanged",
            "aggregation": "Joint K/V norm ratio per sample/block/participant, then equal means across participants, all decoder blocks, and the complete planned question set",
            "weights": "Equal weights per participant, per block, and per question; no pooling of Frobenius norms across these dimensions",
            "decomposition": "absolute_frobenius^2 = available_error_squared + missing_reference_squared; reference_frobenius^2 = total_reference_squared",
            "std": "Population standard deviation across the measured questions",
            "sem": "Sample standard deviation (ddof=1) / sqrt(n); null for n=1; not a confidence interval",
            "incomplete": "Mean/std/SEM are null until every planned question is present",
        },
        "notes": [
            "Relative KV-availability deviation is not bounded by one.",
            "Retention=1 is the FedAttn setting at the fixed H, not necessarily CenAttn; its deviation is not forced to zero.",
            "Axis C measures KV availability before causal query masking; publisher sharing can change this diagnostic even when causal prefill hidden-state outputs are unchanged.",
        ],
        "panels": [],
        "point_counts": {"complete": 0, "incomplete": 0, "missing": 0},
    }
    point_rows, sample_rows, participant_rows = [], [], []
    for panel in plan["panels"]:
        tag = (panel["model"], panel["split_way"])
        require(tag in expected_panels and tag not in seen_panels, f"Unexpected/duplicate panel: {tag}")
        seen_panels.add(tag)
        require(panel.get("dtype") in ("float32", "float16", "bfloat16"), f"{tag}: invalid dtype")
        require(type(panel.get("H")) is int and panel["H"] > 0, f"{tag}: invalid H")
        coordinates = set()
        for point in panel["points"]:
            axis = point.get("axis")
            require(axis in AXES and point.get("x_axis") == AXES[axis], f"{tag}: invalid axis")
            value = point.get("x_value")
            require((axis == "D" and isinstance(value, str) and value) or
                    (axis != "D" and type(value) in (int, float) and math.isfinite(value)),
                    f"{tag}: invalid x_value")
            coordinate = (axis, value)
            require(coordinate not in coordinates, f"{tag}: duplicate axis coordinate {coordinate}")
            coordinates.add(coordinate)
            require(isinstance(point.get("config"), str) and point["config"] and point["config"] != "CenAttn",
                    f"{tag}: invalid config name")
        require({p["axis"] for p in panel["points"]} == set(AXES), f"{tag}: missing planned axis")
        records, run_id, n_clients, n_layers = _read_panel(root, panel, sample)
        out = {key: value for key, value in panel.items() if key not in ("points", "records_file", "manifest_path")}
        out.update(split_label=SPLIT_LABELS[panel["split_way"]], run_id=run_id,
                   sample_set_sha256=sample["sample_set_sha256"], n_participants=n_clients, n_layers=n_layers, points=[])
        out["baseline"] = _aggregate([records[(i, "CenAttn")] for i in expected
                                      if (i, "CenAttn") in records], expected)
        for point in panel["points"]:
            rows = [records[(i, point["config"])] for i in expected if (i, point["config"]) in records]
            metric_field = METRIC_FIELDS[point["axis"]]
            result = {**point, "metric_field": metric_field, **_aggregate(rows, expected, metric_field)}
            result["participant_layer_means"] = [
                {"participant_index": i, "is_publisher": i == n_clients - 1,
                 **_aggregate(rows, expected, value_fn=lambda row, i=i: row["fidelity"]["participant_layer_means"][i]["relative_kv_frobenius"])}
                for i in range(n_clients or 0)
            ]
            result["layer_means"] = [
                {"block_index": i + 1,
                 **_aggregate(rows, expected, value_fn=lambda row, i=i: row["fidelity"]["kv_layers"][i]["participant_mean_relative_kv_frobenius"])}
                for i in range(n_layers)
            ]
            out["points"].append(result)
            summary["point_counts"][result["status"]] += 1
            point_rows.append({"metric_id": METRIC_ID, "model": panel["model"], "split_way": panel["split_way"],
                               "dtype": panel["dtype"], "H": panel["H"],
                               "run_id": run_id, "sample_set_sha256": sample["sample_set_sha256"],
                               **{k: result[k] for k in ("axis", "x_axis", "x_value", "config", "metric_field", "n", "n_expected", "status", "mean", "std", "sem")}})
        for (index, config), row in sorted(records.items()):
            fidelity = row["fidelity"]
            base = {"metric_id": METRIC_ID, "model": panel["model"], "split_way": panel["split_way"],
                    "dtype": panel["dtype"], "H": panel["H"], "config": config,
                    "sample_index": index, "run_id": run_id,
                    "sample_set_sha256": sample["sample_set_sha256"]}
            shape = fidelity["kv_layers"][0]["reference_kv"]
            sample_rows.append({**base, "metric_field": PRIMARY_FIELD,
                                "n_layers": n_layers, "n_participants": n_clients,
                                **{field: shape[field] for field in ("n_tokens", "n_kv_heads", "head_dim")},
                                "layer_participant_mean_relative_kv_frobenius": fidelity["layer_participant_mean_relative_kv_frobenius"]})
            for participant in fidelity["participant_layer_means"]:
                participant_rows.append({**base, "n_layers": n_layers, **participant})
            for layer in fidelity["kv_layers"]:
                for participant in layer["participant_kv"]:
                    layer_writer.writerow({**base, "block_index": layer["block_index"],
                                           "n_kv_heads": layer["reference_kv"]["n_kv_heads"],
                                           "head_dim": layer["reference_kv"]["head_dim"],
                                           "k_reference_frobenius": layer["reference_kv"]["k_reference_frobenius"],
                                           "v_reference_frobenius": layer["reference_kv"]["v_reference_frobenius"],
                                           "n_missing_tokens": participant["n_tokens"] - participant["n_available_tokens"], **participant})
        summary["panels"].append(out)
    require(seen_panels == expected_panels, "Plan panels do not cover every model/split")
    summary_path = root / "fig4_fidelity_summary.json"
    _atomic_text(summary_path, json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    _atomic_text(root / "fig4_fidelity_points.csv", _csv_text(
        ["metric_id", "model", "split_way", "dtype", "H", "axis", "x_axis", "x_value", "config", "metric_field", "n", "n_expected", "status", "mean", "std", "sem", "run_id", "sample_set_sha256"], point_rows))
    _atomic_text(root / "fig4_fidelity_samples.csv", _csv_text(
        ["metric_id", "model", "split_way", "dtype", "H", "config", "sample_index", "metric_field", "n_layers", "n_participants", "n_tokens", "n_kv_heads", "head_dim", "layer_participant_mean_relative_kv_frobenius", "run_id", "sample_set_sha256"], sample_rows))
    _atomic_text(root / "fig4_fidelity_participants.csv", _csv_text(
        ["metric_id", "model", "split_way", "dtype", "H", "config", "sample_index", "n_layers", "participant_index", "is_publisher", "relative_kv_frobenius", "run_id", "sample_set_sha256"], participant_rows))
    return summary_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, help="Directory containing fig4_plan.json and records")
    args = parser.parse_args()
    path = export_results(args.output_root)
    result = json.loads(path.read_text(encoding="utf-8"))
    print(f"Fig.4 points: {result['point_counts']}")
    print(path)


if __name__ == "__main__":
    main()
