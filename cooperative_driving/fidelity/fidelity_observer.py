
import functools
import hashlib
import math
import random

import numpy as np

HIDDEN_METRIC_ID = "prefill_mean_block_global_hidden_relative_frobenius_v1"
KV_METRIC_ID = "prefill_available_global_kv_mean_block_participant_v1"


def _numpy(value):
    if hasattr(value, "detach"):
        return value.detach().float().cpu().numpy()
    return np.asarray(value)


def build_availability(config, token_pos, token_chunks, sample_key, asker_cav_id, n_layers):
    """Construct KV availability from the Wrapper ownership and sampling rules.

    The result has available[layer, participant, full_sequence_token]. Publisher
    text is included in token_chunks, but only token_pos[group] sensor positions
    can be sampled by A and B. Missing current-layer KV is not filled from earlier layers.
    """
    setting = config["setting"]
    global_layers, local_layers = config["global_layers"], config["local_layers"]
    seq_len = int(token_pos["seq_len"])
    names = ["ego", "car1"]
    if len(token_chunks) != 2 or set(token_chunks) != set(names):
        raise ValueError("This V2V fidelity toolkit requires two groups: ego and car1")
    if asker_cav_id not in ("ego", "1"):
        raise ValueError("Use dataset asker_cav_id='ego' or '1', as the original Wrapper expects")
    publisher = "ego" if asker_cav_id == "ego" else "car1"
    collaborator = "car1" if publisher == "ego" else "ego"
    flattened = [int(i) for name in names for i in token_chunks[name]]
    if sorted(flattened) != list(range(seq_len)):
        raise ValueError("Vehicle token groups must partition all expanded input positions exactly")
    global_set, local_set = set(global_layers), set(local_layers)
    if global_set & local_set or global_set | local_set != set(range(n_layers)):
        raise ValueError("Global and local layer sets must partition all zero-based layers")
    keep = float(setting.get("input_keep_ratio", 1.0))
    if not 0 <= keep <= 1:
        raise ValueError("input_keep_ratio must be between zero and one")
    retained = np.ones(seq_len, dtype=bool)
    for group in names:
        sensors = list(token_pos[group])
        if not set(sensors).issubset(token_chunks[group]):
            raise ValueError(f"{group} sensor positions are outside their token group")
        if keep < 1:
            rng = random.Random(f"{sample_key}|{setting.get('seed', 0)}|A|{group}")
            order = rng.sample(sensors, len(sensors))
            retained[order[int(len(sensors) * keep):]] = False
    available = np.zeros((n_layers, len(names), seq_len), dtype=bool)
    for layer_idx in range(n_layers):
        for participant, owner in enumerate(names):
            available[layer_idx, participant, token_chunks[owner]] = True
        if layer_idx in global_set:
            collaborator_allowed = setting.get("collaborator_global_layers")
            publisher_allowed = setting.get("publisher_global_layers")
            for source in names:
                shared = set(token_chunks[source])
                ratio = float(setting.get("ratios", {}).get(source, 1.0))
                if not 0 <= ratio <= 1:
                    raise ValueError("Communication retention must be between zero and one")
                source_off = ((source == collaborator and collaborator_allowed is not None
                               and layer_idx not in collaborator_allowed)
                              or (source == publisher and publisher_allowed is not None
                                  and layer_idx not in publisher_allowed))
                if source_off:
                    shared.clear()
                elif ratio < 1:
                    sensors = list(token_pos[source])
                    rng = random.Random(
                        f"{sample_key}|{setting.get('seed', 0)}|B|{layer_idx}|{source}")
                    order = rng.sample(sensors, len(sensors))
                    shared.difference_update(order[int(len(sensors) * ratio):])
                shared = sorted(shared)
                for participant, owner in enumerate(names):
                    if owner != source:
                        available[layer_idx, participant, shared] = True
        available[layer_idx, :, ~retained] = False
    digest = hashlib.sha256(available.tobytes(order="C")).hexdigest()
    return {
        "available": available, "retained": retained,
        "participant_names": names, "publisher": publisher,
        "token_chunks": {name: list(token_chunks[name]) for name in names},
        "seq_len": seq_len, "n_layers": n_layers,
        "kv_availability_sha256": digest,
        "n_retained_input_tokens": int(retained.sum()),
    }


def validate_availability_against_masks(bundle, prefill_masks, batch_index=0):
    """Validate attention masks for every retained query without modifying them.

    Future keys count as received KV but remain inaccessible to earlier queries.
    Dummy query rows removed by group A are excluded only from this mask check.
    """
    n_layers, n_participants, seq_len = bundle["available"].shape
    if len(prefill_masks) != n_layers:
        raise ValueError("Mask count differs from the model layer count")
    keys = np.arange(seq_len)
    digest = hashlib.sha256()
    for layer_idx, mask in enumerate(prefill_masks):
        raw = _numpy(mask)
        if raw.ndim != 4 or raw.shape[1:] != (1, seq_len, seq_len):
            raise ValueError("Expected prefill mask [config_batch,1,expanded_tokens,expanded_tokens]")
        raw = raw[batch_index, 0]
        if not np.all((raw == 0) | (raw <= -1e4)):
            raise ValueError("Mask must contain zero or blocked entries, not attention biases")
        visible = raw == 0
        digest.update(visible.tobytes(order="C"))
        for participant in range(n_participants):
            name = bundle["participant_names"][participant]
            queries = np.asarray(bundle["token_chunks"][name], dtype=int)
            queries = queries[bundle["retained"][queries]]
            expected = (bundle["available"][layer_idx, participant][None, :]
                        & (keys[None, :] <= queries[:, None]))
            actual = visible[queries, :]
            if not np.array_equal(actual, expected):
                row, col = np.argwhere(actual != expected)[0]
                raise ValueError(
                    f"KV protocol differs from actual mask at block {layer_idx + 1}, "
                    f"participant={name}, query={queries[row]}, key={col}; "
                    "check Wrapper version, sample ID, seed and vehicle role")
    bundle["prefill_mask_sha256"] = digest.hexdigest()
    return digest.hexdigest()


def measure_hidden_layer(hidden, reference, block_index):
    hidden, reference = np.asarray(hidden), np.asarray(reference)
    if hidden.ndim != 2 or hidden.shape != reference.shape:
        raise ValueError("Hidden states must share [all_valid_tokens, hidden_dimension]")
    hidden, reference = hidden.astype(np.float32, copy=False), reference.astype(np.float32, copy=False)
    if not np.isfinite(hidden).all() or not np.isfinite(reference).all():
        raise ValueError("Nonfinite hidden state")
    error_squared = float(np.square(hidden - reference).sum(dtype=np.float64))
    reference_squared = float(np.square(reference).sum(dtype=np.float64))
    if not math.isfinite(error_squared + reference_squared):
        raise ValueError("Nonfinite hidden-state squared norm")
    absolute, denominator = math.sqrt(error_squared), math.sqrt(reference_squared)
    return {"block_index": block_index, "n_tokens": hidden.shape[0],
            "absolute_frobenius": absolute, "reference_frobenius": denominator,
            "relative_frobenius": absolute / max(denominator, 1e-12)}


def measure_kv_layer(k, v, reference_k, reference_v, available, block_index,
                     participant_names, publisher):
    arrays = [np.asarray(x, dtype=np.float32) for x in (k, v, reference_k, reference_v)]
    if arrays[0].ndim != 3 or any(x.shape != arrays[0].shape for x in arrays):
        raise ValueError("K and V must share [native_KV_heads, all_valid_tokens, head_dimension]")
    if any(not np.isfinite(x).all() for x in arrays):
        raise ValueError("Nonfinite KV state")
    if available.shape != (len(participant_names), arrays[0].shape[1]):
        raise ValueError("KV availability has inconsistent participant/token dimensions")
    k, v, reference_k, reference_v = arrays
    error = (np.square(k - reference_k).sum(axis=(0, 2), dtype=np.float64)
             + np.square(v - reference_v).sum(axis=(0, 2), dtype=np.float64))
    ref = (np.square(reference_k).sum(axis=(0, 2), dtype=np.float64)
           + np.square(reference_v).sum(axis=(0, 2), dtype=np.float64))
    reference_squared = float(ref.sum())
    if not np.isfinite(error).all() or not np.isfinite(ref).all():
        raise ValueError("Nonfinite KV squared norm")
    denominator = math.sqrt(reference_squared)
    rows = []
    for index, name in enumerate(participant_names):
        mask = available[index]
        present_error, missing_error = float(error[mask].sum()), float(ref[~mask].sum())
        absolute = math.sqrt(present_error + missing_error)
        rows.append({
            "participant_index": index, "participant": name, "is_publisher": name == publisher,
            "n_tokens": len(mask), "n_available_tokens": int(mask.sum()),
            "coverage": float(mask.mean()), "relative_frobenius": absolute / max(denominator, 1e-12),
            "absolute_frobenius": absolute, "reference_frobenius": denominator,
            "available_error_squared": present_error, "missing_reference_squared": missing_error,
            "total_reference_squared": reference_squared,
        })
    return {"block_index": block_index, "participant_kv": rows,
            "reference_kv": {"n_tokens": arrays[0].shape[1], "n_kv_heads": arrays[0].shape[0],
                             "head_dim": arrays[0].shape[2], "reference_frobenius": denominator},
            "participant_mean_relative_kv_frobenius": math.fsum(r["relative_frobenius"] for r in rows) / len(rows)}


def _is_kv_pair(value):
    return (isinstance(value, (tuple, list)) and len(value) >= 2
            and all(hasattr(x, "shape") and len(x.shape) == 4 for x in value[:2]))


def _cache_pair(cache, layer_idx):
    if cache is None:
        return None
    if _is_kv_pair(cache):
        return cache[0], cache[1]
    if hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
        return cache.key_cache[layer_idx], cache.value_cache[layer_idx]
    if hasattr(cache, "layers"):
        layer = cache.layers[layer_idx]
        if hasattr(layer, "keys") and hasattr(layer, "values"):
            return layer.keys, layer.values
    try:
        layer = cache[layer_idx]
        if _is_kv_pair(layer):
            return layer[0], layer[1]
    except (IndexError, KeyError, TypeError, AttributeError):
        pass
    return None


class PrefillObserver:
    """Capture a single-sample prefill through the attention cache and hooks.

    Install after V2VFedAttnWrapper. Multimodal generation uses max_new_tokens=1
    and use_cache=True for one prefill forward pass; the selected token is
    discarded without a decode forward pass. The attention implementation,
    dtype, masks, and RoPE calculations remain unchanged.
    """
    def __init__(self, model, token_chunks, asker_cav_id, availability, reference=None):
        self.model, self.availability = model, availability
        publisher = "ego" if asker_cav_id == "ego" else "car1"
        if token_chunks != availability["token_chunks"] or publisher != availability["publisher"]:
            raise ValueError("Observer ownership differs from the validated availability metadata")
        self.layers = model.model.layers
        self.n_layers = len(self.layers)
        self.reference = reference
        self.hidden = [None] * self.n_layers
        self.kv = [None] * self.n_layers
        self._hidden_states = [None] * self.n_layers if reference is None else None
        self._kv_states = [None] * self.n_layers if reference is None else None
        self._forwards, self._handles = [], []
        if model.training:
            raise ValueError("Call model.eval() before measuring deterministic fidelity")
        if self.n_layers != availability["n_layers"]:
            raise ValueError("Availability and model layer counts differ")
        if reference is None and not availability["available"].all():
            raise ValueError("Cen reference requires complete KV at every participant and layer")
        if reference is not None:
            if (len(reference.get("hidden_states", [])) != self.n_layers
                    or len(reference.get("kv_states", [])) != self.n_layers):
                raise ValueError("Reference must contain every complete block output and KV pair")

    def __enter__(self):
        try:
            for index, layer in enumerate(self.layers):
                attn = layer.self_attn
                original = attn.forward
                self._forwards.append((attn, original))
                attn.forward = self._attention_forward(index, original)
                self._handles.append(layer.register_forward_hook(self._block_hook(index)))
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def _attention_forward(self, index, original):
        @functools.wraps(original)
        def wrapped(*args, **kwargs):
            hidden = args[0] if args else kwargs.get("hidden_states")
            if hidden is None or tuple(hidden.shape[:2]) != (1, self.availability["seq_len"]):
                raise ValueError("Observer requires one complete multimodal prefill, without decode or padding")
            if self.kv[index] is not None:
                raise RuntimeError("Repeated attention forward: request exactly one prefill")
            result = original(*args, **kwargs)
            pair = None
            if isinstance(result, (tuple, list)):
                for candidate in reversed(result[1:]):
                    pair = _cache_pair(candidate, index)
                    if pair is not None:
                        break
            if pair is None:
                cache = kwargs.get("past_key_value", kwargs.get("past_key_values"))
                pair = _cache_pair(cache, index)
            if pair is None:
                raise RuntimeError(
                    "Cannot read actual attention KV cache. Set use_cache=True; "
                    "this observer supports legacy Llama KV tuples and Hugging Face Cache objects")
            key, value = (self._native_kv(x, index) for x in pair)
            if self.reference is None:
                self._kv_states[index] = (key.copy(), value.copy())
                reference_key, reference_value = key, value
            else:
                reference_key, reference_value = self.reference["kv_states"][index]
            self.kv[index] = measure_kv_layer(
                key, value, reference_key, reference_value,
                self.availability["available"][index], index + 1,
                self.availability["participant_names"], self.availability["publisher"])
            return result
        return wrapped

    def _native_kv(self, tensor, index):
        if len(tensor.shape) != 4 or tensor.shape[0] != 1 or tensor.shape[2] != self.availability["seq_len"]:
            raise ValueError("Expected actual prefill KV cache [1,native_heads,expanded_tokens,head_dim]")
        attn = self.layers[index].self_attn
        config = getattr(attn, "config", self.model.config)
        native_heads = getattr(attn, "num_key_value_heads", getattr(config, "num_key_value_heads", None))
        if native_heads is None:
            native_heads = getattr(attn, "num_heads", getattr(config, "num_attention_heads", None))
        if native_heads is None or tensor.shape[1] != native_heads:
            raise ValueError("Attention cache must contain native KV heads before GQA repetition")
        return _numpy(tensor[0]).copy()

    def _block_hook(self, index):
        def hook(module, args, result):
            hidden = result[0] if isinstance(result, (tuple, list)) else result
            if tuple(hidden.shape[:2]) != (1, self.availability["seq_len"]):
                raise ValueError("Complete block output does not match expanded input token count")
            if self.hidden[index] is not None:
                raise RuntimeError("Repeated block forward: request exactly one prefill")
            hidden = _numpy(hidden[0]).copy()
            if self.reference is None:
                self._hidden_states[index] = hidden
                reference = hidden
            else:
                reference = self.reference["hidden_states"][index]
            self.hidden[index] = measure_hidden_layer(hidden, reference, index + 1)
        return hook

    def result(self):
        if any(x is None for x in self.hidden) or any(x is None for x in self.kv):
            raise RuntimeError("Not every complete block and KV cache was captured")
        if self.reference is None:
            self.reference = {"hidden_states": self._hidden_states, "kv_states": self._kv_states}
        names = self.availability["participant_names"]
        hidden_mean = math.fsum(x["relative_frobenius"] for x in self.hidden) / self.n_layers
        kv_mean = math.fsum(x["participant_mean_relative_kv_frobenius"] for x in self.kv) / self.n_layers
        participant_means = [
            {"participant_index": n, "participant": name,
             "is_publisher": name == self.availability["publisher"],
             "relative_kv_frobenius": math.fsum(
                 layer["participant_kv"][n]["relative_frobenius"] for layer in self.kv) / self.n_layers}
            for n, name in enumerate(names)]
        return {
            "hidden_metric_id": HIDDEN_METRIC_ID, "kv_metric_id": KV_METRIC_ID,
            "hidden_layers": self.hidden, "kv_layers": self.kv,
            "layer_count": self.n_layers,
            "n_tokens": self.availability["seq_len"],
            "n_retained_input_tokens": self.availability["n_retained_input_tokens"],
            "hidden_mean": hidden_mean, "kv_mean": kv_mean,
            "participant_means": participant_means,
            "layer_mean_relative_hidden_frobenius": hidden_mean,
            "layer_participant_mean_relative_kv_frobenius": kv_mean,
            "participant_layer_means": participant_means,
            "kv_availability_sha256": self.availability["kv_availability_sha256"],
            "prefill_mask_sha256": self.availability.get("prefill_mask_sha256"),
        }

    def __exit__(self, *exc):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        for attn, original in reversed(self._forwards):
            attn.forward = original
        self._forwards.clear()
