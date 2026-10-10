#!/usr/bin/env python3
"""Resolve local model checkpoints and GSM8K data for fidelity measurements.

Hugging Face model identifiers and dataset identifiers are distinct from local
filesystem paths. Resource resolution uses local files and caches without
implicitly downloading checkpoints or datasets.
"""
from __future__ import annotations

from contextlib import contextmanager
import importlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile


_HUB_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*$")


def resolve_model_reference(value, base):
    """Return an absolute local path or preserve a namespace/model Hub identifier."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Model reference must be a nonempty local path or Hugging Face repo_id")
    value = value.strip()
    base = Path(base).expanduser().resolve()
    path = Path(value).expanduser()
    if path.is_absolute() or value.startswith((".", "~")) or (base / path).exists():
        return str((path if path.is_absolute() else base / path).resolve())
    if _HUB_ID.fullmatch(value):
        return value
    return str((base / path).resolve())


@contextmanager
def offline_huggingface():
    """Disable Hugging Face Hub and datasets network access, including prior imports.

    Resource resolution is sequential in this runner. Do not use this context
    concurrently with another thread changing Hugging Face offline settings.
    """
    env_keys = ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE")
    previous_env = {key: os.environ.get(key) for key in env_keys}
    for key in env_keys:
        os.environ[key] = "1"
    restored = []
    try:
        # Import Hub constants before download helpers so offline mode applies
        # whether or not the packages have already been imported.
        for module_name, key in (("huggingface_hub.constants", "HF_HUB_OFFLINE"),
                                 ("datasets.config", "HF_DATASETS_OFFLINE")):
            try:
                module = importlib.import_module(module_name)
            except ImportError:
                continue
            if hasattr(module, key):
                restored.append((module, key, getattr(module, key)))
                setattr(module, key, True)
        yield
    finally:
        for module, key, value in reversed(restored):
            setattr(module, key, value)
        for key, value in previous_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _local_snapshot(path):
    """Accept a model directory, a snapshot, or a Hub repository cache directory."""
    path = Path(path).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"Local checkpoint directory does not exist: {path}")
    if (path / "config.json").is_file():
        return path
    snapshots = path / "snapshots"
    if snapshots.is_dir():
        ref = path / "refs" / "main"
        if ref.is_file():
            candidate = snapshots / ref.read_text(encoding="utf-8").strip()
            if (candidate / "config.json").is_file():
                return candidate.resolve()
        choices = [p for p in snapshots.iterdir() if p.is_dir() and (p / "config.json").is_file()]
        if len(choices) == 1:
            return choices[0].resolve()
        if len(choices) > 1:
            raise ValueError(f"Multiple cached checkpoint snapshots under {path}; set an exact snapshot path")
    raise FileNotFoundError(f"Checkpoint config.json not found in {path}; set the actual model/snapshot directory")


def resolve_checkpoint_path(value):
    """Resolve a reference to its local checkpoint directory without networking."""
    value = str(value)
    path = Path(value).expanduser()
    if path.is_absolute() or path.exists() or not _HUB_ID.fullmatch(value):
        return str(_local_snapshot(path))
    try:
        with offline_huggingface():
            from huggingface_hub import snapshot_download
            snapshot = snapshot_download(repo_id=value, local_files_only=True)
        return str(_local_snapshot(snapshot))
    except (ImportError, OSError, ValueError) as exc:
        raise RuntimeError(
            f"Checkpoint {value!r} is unavailable in the local Hugging Face cache. "
            "No download was attempted. Set this model entry in settings_server212.json "
            "to its existing absolute checkpoint/snapshot directory, or run in the original "
            "environment with its original HF_HOME/HF_HUB_CACHE. "
            f"Underlying error: {exc}"
        ) from exc


def _qa_rows(dataset):
    rows = []
    for index, item in enumerate(dataset):
        if not isinstance(item, dict):
            raise ValueError(f"Dataset row {index} is not an object")
        for key in ("question", "answer"):
            if not isinstance(item.get(key), str) or not item[key].strip():
                raise ValueError(f"Dataset row {index} has no nonempty {key} string")
        rows.append({"question": item["question"], "answer": item["answer"]})
    if not rows:
        raise ValueError("The selected dataset is empty")
    return rows


def _immutable_jsonl(path, rows):
    """Publish a complete file atomically without replacing different existing data."""
    path = Path(path)
    content = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise ValueError(f"Dataset content/order differs from {path}; use a new output_root")
        return str(path.resolve())
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n",
                                         prefix=".qa-", suffix=".tmp", dir=path.parent,
                                         delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        # Create the destination atomically only if no other writer has created it.
        # Readers cannot observe a partially written JSONL file.
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_text(encoding="utf-8") != content:
                raise ValueError(f"A concurrent writer supplied different data for {path}")
        return str(path.resolve())
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def resolve_data_jsonl(settings, base, output_root):
    """Resolve one ordered question-answer file shared by all models, splits, and points.

    Explicit data_jsonl has precedence. Otherwise dataset_source='hf_cache'
    (default) reads the gsm8k/main test split from the local cache; source
    'load_from_disk' reads dataset_path and selects test from a DatasetDict.
    Neither mode substitutes a partial results file for the dataset.
    """
    base = Path(base).expanduser().resolve()
    if settings.get("data_jsonl"):
        path = Path(settings["data_jsonl"]).expanduser()
        path = (path if path.is_absolute() else base / path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"data_jsonl does not exist: {path}")
        return str(path)
    source = settings.get("dataset_source", "hf_cache")
    if source == "jsonl":
        raise ValueError("dataset_source='jsonl' requires data_jsonl in settings")
    if source not in ("hf_cache", "load_from_disk"):
        raise ValueError("dataset_source must be 'hf_cache', 'jsonl', or 'load_from_disk'")
    try:
        with offline_huggingface():
            from datasets import load_dataset, load_from_disk
            if source == "hf_cache":
                dataset = load_dataset("gsm8k", "main", split="test", download_mode="reuse_dataset_if_exists")
            else:
                if not settings.get("dataset_path"):
                    raise ValueError("dataset_source='load_from_disk' requires dataset_path")
                path = Path(settings["dataset_path"]).expanduser()
                path = (path if path.is_absolute() else base / path).resolve()
                dataset = load_from_disk(str(path))
                # DatasetDict is a dict subclass; a plain Dataset is not.
                if isinstance(dataset, dict):
                    if "test" not in dataset:
                        raise ValueError(f"Saved DatasetDict has no test split: {path}")
                    dataset = dataset["test"]
            rows = _qa_rows(dataset)
    except Exception as exc:
        raise RuntimeError(
            "Cannot obtain the original ordered GSM8K test data locally. No download "
            "was attempted. Use the original environment/cache, or set data_jsonl in settings "
            "with a FULL, original-order question/answer JSONL (not a sampled file); "
            "alternatively set dataset_source='load_from_disk' and dataset_path. "
            f"Underlying error: {exc}"
        ) from exc
    if source == "hf_cache" and len(rows) != 1319:
        raise ValueError(f"Expected 1319 original GSM8K test rows, found {len(rows)}; refusing a changed/partial split")
    return _immutable_jsonl(Path(output_root) / "shared_gsm8k_test.jsonl", rows)
