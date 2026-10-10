"""Participant views of current-layer prefill KV aligned to the full token axis.

Unavailable rows are zeroed only for measurement; model attention is unchanged.
This module is independent of PyTorch. Subtractions and reductions use float32.
"""
import math
import numpy as np

METRIC_ID = "prefill_available_kv_mean_over_participants_blocks_v1"


def measure_kv_layer(k, v, reference_k, reference_v, availability, block_index):
    arrays = [np.asarray(x) for x in (k, v, reference_k, reference_v)]
    shape = arrays[0].shape
    if len(shape) != 3 or any(d <= 0 for d in shape) or any(x.shape != shape for x in arrays):
        raise ValueError("K, V and their Cen references must share [KV heads, valid tokens, head dim]")
    if any(x.dtype != np.float32 or not np.isfinite(x).all() for x in arrays):
        raise ValueError("KV inputs must be finite float32 arrays")
    if type(block_index) is not int or block_index < 1:
        raise ValueError("block_index must be a positive integer")
    available = np.asarray(availability)
    if available.dtype != np.bool_ or available.ndim != 2 or available.shape[0] < 1 or available.shape[1] != shape[1]:
        raise ValueError("availability must be a boolean [participants, valid tokens] array")
    k, v, reference_k, reference_v = arrays
    # Each token's energy sums native KV heads and head dimensions. No GQA
    # repetition or participant token weighting enters the normalization.
    k_error = np.square(k - reference_k).sum(axis=(0, 2), dtype=np.float32)
    v_error = np.square(v - reference_v).sum(axis=(0, 2), dtype=np.float32)
    k_reference = np.square(reference_k).sum(axis=(0, 2), dtype=np.float32)
    v_reference = np.square(reference_v).sum(axis=(0, 2), dtype=np.float32)
    if not all(np.isfinite(x).all() for x in (k_error, v_error, k_reference, v_reference)):
        raise ValueError("Nonfinite KV squared norm in FP32")
    k_reference_squared = float(k_reference.sum(dtype=np.float32))
    v_reference_squared = float(v_reference.sum(dtype=np.float32))
    reference_squared = k_reference_squared + v_reference_squared
    denominator = math.sqrt(reference_squared)
    n_heads, n_tokens, head_dim = shape
    participants = []
    for n, mask in enumerate(available):
        k_present = float(k_error[mask].sum(dtype=np.float32))
        v_present = float(v_error[mask].sum(dtype=np.float32))
        k_missing = float(k_reference[~mask].sum(dtype=np.float32))
        v_missing = float(v_reference[~mask].sum(dtype=np.float32))
        present_error = k_present + v_present
        missing_error = k_missing + v_missing
        absolute = math.sqrt(present_error + missing_error)
        n_available = int(mask.sum())
        participants.append({
            "participant_index": n, "is_publisher": n == len(available) - 1,
            "n_tokens": n_tokens, "n_available_tokens": n_available,
            "coverage": n_available / n_tokens,
            "absolute_frobenius": absolute, "reference_frobenius": denominator,
            "relative_frobenius": absolute / max(denominator, 1e-12),
            "per_entry_rms": absolute / math.sqrt(2 * n_heads * n_tokens * head_dim),
            "k_relative_frobenius": math.sqrt(k_present + k_missing) / max(math.sqrt(k_reference_squared), 1e-12),
            "v_relative_frobenius": math.sqrt(v_present + v_missing) / max(math.sqrt(v_reference_squared), 1e-12),
            "available_error_squared": present_error,
            "missing_reference_squared": missing_error,
            "total_reference_squared": reference_squared,
        })
    return {
        "block_index": block_index,
        "reference_kv": {"n_tokens": n_tokens, "n_kv_heads": n_heads, "head_dim": head_dim,
                         "reference_frobenius": denominator,
                         "k_reference_frobenius": math.sqrt(k_reference_squared),
                         "v_reference_frobenius": math.sqrt(v_reference_squared)},
        "participant_kv": participants,
        "participant_mean_relative_kv_frobenius": math.fsum(p["relative_frobenius"] for p in participants) / len(participants),
    }
