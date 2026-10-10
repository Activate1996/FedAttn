"""Observe Qwen SDPA inputs without changing attention or generating tokens."""
import hashlib
import math
import numpy as np

from kv_metrics import measure_kv_layer


def availability_masks(torch, captured, n_layers):
    """Read noncausal input-KV ownership and sharing sets from the mask builders.

    The tables also define decode masks, but reading them does not run decoding.
    Causal prefill masks do not indicate whether future keys are shared.
    Padding is excluded from the global token axis.
    """
    valid = captured["padding_mask"][0].bool().cpu().numpy()
    masks = captured["kv_availability_masks"]
    if len(masks) != len(captured["chunks"]) or any(len(row) != n_layers for row in masks):
        raise ValueError("Original KV availability table has wrong participant/layer dimensions")
    memo = {}
    layers = []
    for layer in range(n_layers):
        values = []
        for participant in range(len(masks)):
            mask = masks[participant][layer]
            if tuple(mask.shape) != (1, 1, 1, len(valid)):
                raise ValueError("Expected original [1,1,1,input_length] KV availability mask")
            if id(mask) not in memo:
                if mask.dtype == torch.bool:
                    allowed = mask[0, 0, 0].detach().cpu().numpy().astype(bool, copy=False)
                else:
                    raw = mask[0, 0, 0].detach().float().cpu().numpy()
                    if not np.all((raw == 0) | (raw <= -1e4)):
                        raise ValueError("Expected zero/blocked availability masks, not attention biases")
                    allowed = raw == 0
                memo[id(mask)] = allowed[valid].copy()
            values.append(memo[id(mask)])
        layers.append(np.stack(values))
    return layers


def availability_hash(layers):
    h = hashlib.sha256()
    for value in layers:
        h.update(str(value.shape).encode("ascii"))
        h.update(value.tobytes(order="C"))
    return h.hexdigest()


class KVObserver:
    def __init__(self, torch, model, captured, reference=None):
        self.torch, self.model, self.captured = torch, model, captured
        self.reference = reference
        self.n_layers = len(model.model.layers)
        self.available = availability_masks(torch, captured, self.n_layers)
        self.valid = captured["padding_mask"][0].bool().cpu().numpy()
        self.layers = [None] * self.n_layers
        self.states = [None] * self.n_layers if reference is None else None
        self.handles, self.current = [], None
        self.patched = False
        if reference is not None and len(reference.get("kv_states", [])) != self.n_layers:
            raise ValueError("Cen reference must contain actual KV for every layer")
        if reference is None and not all(value.all() for value in self.available):
            raise ValueError("Cen publisher reference must have complete global KV at every layer")

    def __enter__(self):
        self.F = self.torch.nn.functional
        self.original = self.F.scaled_dot_product_attention
        try:
            for index, layer in enumerate(self.model.model.layers):
                self.handles.append(layer.self_attn.register_forward_pre_hook(self._pre(index)))
                self.handles.append(layer.self_attn.register_forward_hook(self._post))
            self.F.scaled_dot_product_attention = self._sdpa
            self.patched = True
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def _pre(self, index):
        def hook(module, args):
            self.current = (index, module)
        return hook

    def _post(self, module, args, result):
        self.current = None

    def _sdpa(self, q, k, v, *args, **kwargs):
        if self.current is not None:
            dropout = kwargs.get("dropout_p", args[1] if len(args) > 1 else 0.0)
            if dropout:
                raise ValueError("KV fidelity requires eval mode and zero attention dropout")
            self._measure(q, k, v)
        return self.original(q, k, v, *args, **kwargs)

    def _native_kv(self, tensor, module):
        if len(tensor.shape) != 4 or tensor.shape[0] != 1 or tensor.shape[-2] != len(self.valid):
            raise ValueError("KV observer requires cache-free, batch-one prefill on all input tokens")
        config = getattr(module, "config", self.model.config)
        native_heads = getattr(module, "num_key_value_heads", getattr(config, "num_key_value_heads", None))
        if type(native_heads) is not int or native_heads < 1:
            raise ValueError("Cannot determine Qwen's native KV head count")
        heads = tensor.shape[1]
        if heads == native_heads:
            value = tensor[0]
        else:
            attention_heads = getattr(module, "num_heads", getattr(config, "num_attention_heads", None))
            if heads != attention_heads or heads % native_heads:
                raise ValueError("Unexpected SDPA KV/GQA head layout")
            # Qwen repeat_kv expands each native head into adjacent copies.
            # Remove this repetition to measure the native KV matrices.
            value = tensor[0].reshape(native_heads, heads // native_heads, tensor.shape[-2], tensor.shape[-1])[:, 0]
        return value.detach().float().cpu().numpy()[:, self.valid, :].copy()

    def _measure(self, q, k, v):
        index, module = self.current
        if self.layers[index] is not None:
            raise RuntimeError(f"SDPA executed more than once in layer {index + 1}")
        if q.shape[0] != 1 or q.shape[-2] != len(self.valid):
            raise ValueError("Query shape is not a full batch-one prefill")
        key, value = self._native_kv(k, module), self._native_kv(v, module)
        if self.reference is None:
            self.states[index] = (key, value)
            ref_key, ref_value = key, value
        else:
            ref_key, ref_value = self.reference["kv_states"][index]
        self.layers[index] = measure_kv_layer(key, value, ref_key, ref_value,
                                              self.available[index], index + 1)

    def result(self):
        if any(layer is None for layer in self.layers):
            raise RuntimeError("Not every Qwen layer reached SDPA; use the original SDPA environment")
        result = {
            "kv_layers": self.layers, "layer_count": self.n_layers,
            "layer_participant_mean_relative_kv_frobenius": math.fsum(
                layer["participant_mean_relative_kv_frobenius"] for layer in self.layers) / self.n_layers,
            "participant_layer_means": [
                {"participant_index": n, "is_publisher": n == len(self.available[0]) - 1,
                 "relative_kv_frobenius": math.fsum(
                     layer["participant_kv"][n]["relative_frobenius"] for layer in self.layers) / self.n_layers}
                for n in range(len(self.available[0]))],
            "kv_availability_sha256": availability_hash(self.available),
            "mechanism": [],
        }
        if self.states is not None:
            result["kv_states"] = self.states
        return result

    def __exit__(self, *exc):
        if self.patched:
            self.F.scaled_dot_product_attention = self.original
            self.patched = False
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.current = None
