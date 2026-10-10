"""Validate completed measurements and prepare portable plotting summaries."""
import json
import math
import shutil
from pathlib import Path


KV_METRIC = "prefill_available_kv_mean_over_participants_blocks_v1"
KV_FIELD = "fidelity.layer_participant_mean_relative_kv_frobenius"
FIG23_MEASUREMENT = "final Transformer block, before final RMSNorm, all valid input tokens"


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _finite(values, label):
    _require(all(type(v) in (int, float) and math.isfinite(v) and v >= 0 for v in values),
             f"{label}: invalid measurement mean")


def _complete(point, count, label):
    _require(point.get("status") == "complete" and point.get("n") == count
             and point.get("n_expected") == count and not point.get("missing_sample_indices"),
             f"{label}: incomplete sample coverage; summary was not copied")


def prepare_plot_data(experiment, result_roots, output_root):
    """Validate selected completed jobs and copy their summaries without changing bytes."""
    _require(experiment in ("fig23", "kv", "trajectory"), "Unknown experiment")
    _require(isinstance(result_roots, dict) and result_roots, "Provide selected result roots")
    output_root = Path(output_root).expanduser().resolve()
    staged, seen_models, reference_sample, reference_settings = [], set(), None, None
    for key, root in result_roots.items():
        root = Path(root).expanduser().resolve()
        if experiment == "fig23":
            _require(key in ("fp32", "7b_bf16"), "Fig23 job must be fp32 or 7b_bf16")
            source = root / "curves" / "representation_deviation_summary.json"
            destination = output_root / "fig23" / key / source.name
        elif experiment == "kv":
            _require(key in ("fp32", "7b_bf16"), "KV job must be fp32 or 7b_bf16")
            source = root / "fig4_fidelity_summary.json"
            destination = output_root / "kv" / key / source.name
        else:
            _require(type(key) is int and key > 0, "Trajectory job must be a positive integer H")
            source = root / "trajectory_summary.json"
            destination = output_root / "trajectory" / f"trajectory_summary_h{key}.json"
        data = json.loads(source.read_text(encoding="utf-8-sig"))
        label = str(source)
        sample = data if experiment == "fig23" else data["sample_plan"]
        indices = sample["sample_indices"]
        _require(isinstance(indices, list) and indices and all(type(i) is int and i >= 0 for i in indices)
                 and len(indices) == len(set(indices)), f"{label}: invalid sample indices")
        count = len(indices)
        _require(sample.get("n_expected_samples") == count and bool(sample.get("sample_set_sha256")),
                 f"{label}: invalid sample count or fingerprint")
        if reference_sample is not None:
            for field in ("sample_indices", "n_expected_samples", "sample_set_sha256",
                          "sample_questions", "dataset_questions_sha256"):
                _require(sample.get(field) == reference_sample.get(field), f"{label}: mismatched {field}")
        reference_sample = sample

        if experiment == "fig23":
            _require(data.get("schema_version") == 2 and data.get("measurement") == FIG23_MEASUREMENT,
                     f"{label}: expected original final-block schema 2")
            points = data["points"]
            models = {point["model"] for point in points}
            _require(points and models == {m for panels in data["panels"].values() for m in panels},
                     f"{label}: empty or inconsistent model panels")
            for point in points:
                _complete(point, count, label)
                _finite([point[m]["mean"] for m in ("relative_frobenius", "absolute_frobenius", "per_entry_rms")], label)
            for panels in data["panels"].values():
                for splits in panels.values():
                    for panel in splits.values():
                        size = len(panel["x_values"])
                        _require(size and panel["status"] == ["complete"] * size
                                 and panel["n"] == panel["n_expected"] == [count] * size,
                                 f"{label}: incomplete plotting panel")
                        _finite(panel["relative_frobenius"]["mean"], label)
        elif experiment == "kv":
            _require(data.get("schema_version") == 4 and data.get("figure") == 4
                     and data.get("metric_id") == KV_METRIC
                     and data["metric_definition"]["field"] == KV_FIELD, f"{label}: expected KV schema 4")
            models = {panel["model"] for panel in data["panels"]}
            _require(models and models == set(data["models"]), f"{label}: inconsistent model panels")
            identities = [(p["model"], p["split_way"]) for p in data["panels"]]
            _require(len(identities) == len(set(identities))
                     and set(identities) == {(m, s) for m in models for s in data["splits"]},
                     f"{label}: missing or duplicate model/split panels")
            for panel in data["panels"]:
                _require(panel["sample_set_sha256"] == sample["sample_set_sha256"] and panel["points"],
                         f"{label}: invalid panel samples or empty points")
                for point in [panel["baseline"], *panel["points"]]:
                    _complete(point, count, label)
                    _finite([point["mean"]], label)
                    for item in point.get("participant_layer_means", []) + point.get("layer_means", []):
                        _complete(item, count, label)
                        _finite([item["mean"]], label)
        else:
            settings = data["settings"]
            _require(data.get("schema_version") == 1 and settings["H"] == key,
                     f"{label}: expected trajectory schema 1 with H={key}")
            if reference_settings is not None:
                for field in ("num_clients", "num_shots", "seed", "prompt_prefix", "prompt_suffix"):
                    _require(settings.get(field) == reference_settings.get(field), f"{label}: mismatched {field}")
            reference_settings = settings
            _require(bool(data["panels"]), f"{label}: empty trajectory panels")
            for model, splits in data["panels"].items():
                _require(set(splits) == set(settings["splits"]), f"{label}: missing segmentation panels")
                for split, panel in splits.items():
                    _complete(panel, count, label)
                    _require(panel["sample_indices"] == indices and panel["sample_set_sha256"] == sample["sample_set_sha256"]
                             and panel["H"] == key and panel["model"] == model and panel["split_way"] == split,
                             f"{label}: mismatched trajectory panel")
                    _require(panel["block_indices"] == list(range(1, panel["num_layers"] + 1)),
                             f"{label}: invalid decoder block indices")
                    means = panel["metrics"]["all"]["relative"]["mean"]
                    _require(len(means) == panel["num_layers"], f"{label}: missing block means")
                    _finite(means, label)
        if experiment != "trajectory":
            _require(not (seen_models & models), f"{label}: model appears in more than one selected shard")
            seen_models.update(models)
        staged.append((source, destination))
    for source, destination in staged:
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source != destination:
            shutil.copy2(source, destination)
    return [destination for _, destination in staged]
