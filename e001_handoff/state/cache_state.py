"""Complete Qwen3.5 hybrid-cache capture, deep clone, and restoration."""

from __future__ import annotations

import hashlib
from typing import Any, Iterable

import torch
from transformers import DynamicCache


STATE_FORMAT_VERSION = "latentport-e001-qwen3.5-cache-v1"


def _clone_tensor(tensor: torch.Tensor | None, device: str | torch.device | None) -> torch.Tensor | None:
    if tensor is None:
        return None
    result = tensor.detach().clone(memory_format=torch.contiguous_format)
    return result.to(device) if device is not None else result


def capture_cache(
    cache: DynamicCache,
    config,
    *,
    device: str | torch.device | None = None,
) -> dict[str, Any]:
    """Capture every persistent tensor and every continuation-relevant flag."""

    if len(cache.layers) != config.num_hidden_layers:
        raise ValueError(f"Cache has {len(cache.layers)} layers; expected {config.num_hidden_layers}")
    layers: list[dict[str, Any]] = []
    for layer_index, (layer_type, layer) in enumerate(zip(config.layer_types, cache.layers, strict=True)):
        if layer_type == "full_attention":
            layers.append(
                {
                    "layer_index": layer_index,
                    "layer_type": layer_type,
                    "cache_class": type(layer).__name__,
                    "keys": _clone_tensor(layer.keys, device),
                    "values": _clone_tensor(layer.values, device),
                    "is_initialized": bool(layer.is_initialized),
                }
            )
        elif layer_type == "linear_attention":
            layers.append(
                {
                    "layer_index": layer_index,
                    "layer_type": layer_type,
                    "cache_class": type(layer).__name__,
                    "number_of_states": int(layer.number_of_states),
                    "conv_states": {
                        int(index): _clone_tensor(value, device) for index, value in layer.conv_states.items()
                    },
                    "recurrent_states": {
                        int(index): _clone_tensor(value, device)
                        for index, value in layer.recurrent_states.items()
                    },
                    "is_conv_states_initialized": {
                        int(index): bool(value) for index, value in layer.is_conv_states_initialized.items()
                    },
                    "is_recurrent_states_initialized": {
                        int(index): bool(value) for index, value in layer.is_recurrent_states_initialized.items()
                    },
                    "has_previous_state": {
                        int(index): bool(value) for index, value in layer.has_previous_state.items()
                    },
                    "conv_kernel_size": {
                        int(index): None if value is None else int(value)
                        for index, value in layer.conv_kernel_size.items()
                    },
                    "record_past": bool(layer.record_past),
                }
            )
        else:
            raise ValueError(f"Unsupported canonical layer type {layer_type!r} at layer {layer_index}")
    return {
        "format_version": STATE_FORMAT_VERSION,
        "num_hidden_layers": int(config.num_hidden_layers),
        "layer_types": list(config.layer_types),
        "sequence_length": int(cache.get_seq_length()),
        "layers": layers,
    }


def restore_cache(
    captured: dict[str, Any],
    config,
    *,
    device: str | torch.device,
) -> DynamicCache:
    """Restore captured state into a new cache with no shared tensor storage."""

    if captured["format_version"] != STATE_FORMAT_VERSION:
        raise ValueError(f"Unknown cache-state format: {captured['format_version']}")
    if captured["layer_types"] != list(config.layer_types):
        raise ValueError("Captured layer-type pattern does not match target config")
    cache = DynamicCache(config=config)
    for record, layer in zip(captured["layers"], cache.layers, strict=True):
        if record["layer_type"] == "full_attention":
            keys = _clone_tensor(record["keys"], device)
            values = _clone_tensor(record["values"], device)
            if bool(record["is_initialized"]) != (keys is not None and values is not None):
                raise ValueError(f"Inconsistent attention initialization at layer {record['layer_index']}")
            if record["is_initialized"]:
                layer.keys = keys
                layer.values = values
                layer.dtype = keys.dtype
                layer.device = keys.device
                layer.is_initialized = True
        else:
            if int(record["number_of_states"]) != layer.number_of_states:
                raise ValueError(f"State-count mismatch at layer {record['layer_index']}")
            for state_index in range(layer.number_of_states):
                conv = _clone_tensor(record["conv_states"][state_index], device)
                recurrent = _clone_tensor(record["recurrent_states"][state_index], device)
                if record["is_conv_states_initialized"][state_index]:
                    if conv is None:
                        raise ValueError(f"Missing initialized conv state at layer {record['layer_index']}")
                    layer.lazy_initialization(
                        conv_states=conv,
                        state_idx=state_index,
                        conv_kernel_size=record["conv_kernel_size"][state_index],
                    )
                    layer.conv_states[state_index].copy_(conv)
                if record["is_recurrent_states_initialized"][state_index]:
                    if recurrent is None:
                        raise ValueError(f"Missing initialized recurrent state at layer {record['layer_index']}")
                    layer.lazy_initialization(recurrent_states=recurrent, state_idx=state_index)
                    layer.recurrent_states[state_index].copy_(recurrent)
                layer.has_previous_state[state_index] = record["has_previous_state"][state_index]
                layer.conv_kernel_size[state_index] = record["conv_kernel_size"][state_index]
            layer.record_past = bool(record["record_past"])
            tensors = [
                value
                for value in (*layer.conv_states.values(), *layer.recurrent_states.values())
                if value is not None
            ]
            if tensors:
                layer.device = tensors[0].device
                layer.dtype = tensors[0].dtype
    if cache.get_seq_length() != captured["sequence_length"]:
        raise ValueError(
            f"Restored cache length {cache.get_seq_length()} != captured {captured['sequence_length']}"
        )
    return cache


def install_components(
    captured: dict[str, Any],
    config,
    *,
    device: str | torch.device,
    include_kv: bool,
    include_gdn_recurrent: bool,
    include_gdn_convolution: bool,
) -> DynamicCache:
    """Install an exact subset of a structurally corresponding cache state.

    Omitted components remain genuinely fresh/uninitialized. This is important:
    zero-filled initialized recurrent buffers would make ``has_previous_state``
    select a different runtime branch from a canonical empty target cache.
    """

    if captured["layer_types"] != list(config.layer_types):
        raise ValueError("Component installation requires identical layer correspondence")
    cache = DynamicCache(config=config)
    for record, layer in zip(captured["layers"], cache.layers, strict=True):
        if record["layer_type"] == "full_attention":
            if include_kv and record["is_initialized"]:
                keys = _clone_tensor(record["keys"], device)
                values = _clone_tensor(record["values"], device)
                layer.keys = keys
                layer.values = values
                layer.dtype = keys.dtype
                layer.device = keys.device
                layer.is_initialized = True
            continue

        state_index = 0
        if include_gdn_convolution:
            conv = _clone_tensor(record["conv_states"][state_index], device)
            if conv is None:
                raise ValueError(f"Missing GDN convolution state at layer {record['layer_index']}")
            layer.lazy_initialization(
                conv_states=conv,
                state_idx=state_index,
                conv_kernel_size=record["conv_kernel_size"][state_index],
            )
            layer.conv_states[state_index].copy_(conv)
        if include_gdn_recurrent:
            recurrent = _clone_tensor(record["recurrent_states"][state_index], device)
            if recurrent is None:
                raise ValueError(f"Missing GDN recurrent state at layer {record['layer_index']}")
            layer.lazy_initialization(recurrent_states=recurrent, state_idx=state_index)
            layer.recurrent_states[state_index].copy_(recurrent)
        if include_gdn_convolution or include_gdn_recurrent:
            # GDN continuation is only meaningful when both components are present.
            if include_gdn_convolution != include_gdn_recurrent:
                raise ValueError("Canonical component installation cannot activate partial GDN state")
            layer.has_previous_state[state_index] = True
            layer.conv_kernel_size[state_index] = record["conv_kernel_size"][state_index]
            layer.record_past = False
            layer.device = recurrent.device
            layer.dtype = conv.dtype
    if include_kv and cache.get_seq_length() != captured["sequence_length"]:
        raise ValueError("Installed attention history has the wrong logical length")
    if not include_kv and cache.get_seq_length() != 0:
        raise ValueError("KV-omitted cache unexpectedly contains attention history")
    return cache


def compose_component_state(
    *,
    kv_state: dict[str, Any],
    gdn_state: dict[str, Any],
) -> dict[str, Any]:
    """Compose K/V from one state and complete GDN memory from another."""

    if kv_state["layer_types"] != gdn_state["layer_types"]:
        raise ValueError("Cannot compose states with different layer patterns")
    if kv_state["sequence_length"] != gdn_state["sequence_length"]:
        raise ValueError("Cannot compose states representing different prefix lengths")
    layers = []
    for kv_layer, gdn_layer in zip(kv_state["layers"], gdn_state["layers"], strict=True):
        layers.append(kv_layer if kv_layer["layer_type"] == "full_attention" else gdn_layer)
    return {
        "format_version": STATE_FORMAT_VERSION,
        "num_hidden_layers": kv_state["num_hidden_layers"],
        "layer_types": list(kv_state["layer_types"]),
        "sequence_length": kv_state["sequence_length"],
        "layers": layers,
    }


def deep_clone_cache(cache: DynamicCache, config, *, device: str | torch.device | None = None) -> DynamicCache:
    captured = capture_cache(cache, config)
    if device is None:
        device = next(tensor.device for tensor in iter_state_tensors(captured))
    return restore_cache(captured, config, device=device)


def iter_state_tensors(captured: dict[str, Any]) -> Iterable[torch.Tensor]:
    for layer in captured["layers"]:
        if layer["layer_type"] == "full_attention":
            for role in ("keys", "values"):
                if layer[role] is not None:
                    yield layer[role]
        else:
            for role in ("conv_states", "recurrent_states"):
                for tensor in layer[role].values():
                    if tensor is not None:
                        yield tensor


def state_nbytes(captured: dict[str, Any]) -> int:
    return sum(tensor.numel() * tensor.element_size() for tensor in iter_state_tensors(captured))


def component_nbytes(
    captured: dict[str, Any],
    *,
    include_kv: bool,
    include_gdn: bool,
) -> int:
    total = 0
    for layer in captured["layers"]:
        if layer["layer_type"] == "full_attention" and include_kv:
            for role in ("keys", "values"):
                tensor = layer[role]
                total += tensor.numel() * tensor.element_size()
        elif layer["layer_type"] == "linear_attention" and include_gdn:
            for role in ("conv_states", "recurrent_states"):
                tensor = layer[role][0]
                total += tensor.numel() * tensor.element_size()
    return total


def state_checksum(captured: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    digest.update(captured["format_version"].encode("utf-8"))
    digest.update(str(captured["sequence_length"]).encode("ascii"))
    for layer in captured["layers"]:
        digest.update(f"{layer['layer_index']}:{layer['layer_type']}".encode("ascii"))
        tensors: list[tuple[str, int, torch.Tensor]] = []
        if layer["layer_type"] == "full_attention":
            for role in ("keys", "values"):
                if layer[role] is not None:
                    tensors.append((role, 0, layer[role]))
            digest.update(str(layer["is_initialized"]).encode("ascii"))
        else:
            for role in ("conv_states", "recurrent_states"):
                for state_index, tensor in sorted(layer[role].items()):
                    if tensor is not None:
                        tensors.append((role, state_index, tensor))
            for role in (
                "is_conv_states_initialized",
                "is_recurrent_states_initialized",
                "has_previous_state",
                "conv_kernel_size",
            ):
                digest.update(repr(sorted(layer[role].items())).encode("ascii"))
            digest.update(str(layer["record_past"]).encode("ascii"))
        for role, state_index, tensor in tensors:
            cpu = tensor.detach().contiguous().cpu()
            digest.update(f"{role}:{state_index}:{tuple(cpu.shape)}:{cpu.dtype}".encode("ascii"))
            digest.update(cpu.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def caches_share_storage(left: DynamicCache, right: DynamicCache, config) -> bool:
    def live_tensors(cache: DynamicCache) -> Iterable[torch.Tensor]:
        for layer_type, layer in zip(config.layer_types, cache.layers, strict=True):
            if layer_type == "full_attention":
                if layer.keys is not None:
                    yield layer.keys
                if layer.values is not None:
                    yield layer.values
            else:
                for tensor in (*layer.conv_states.values(), *layer.recurrent_states.values()):
                    if tensor is not None:
                        yield tensor

    for left_tensor, right_tensor in zip(live_tensors(left), live_tensors(right), strict=True):
        if left_tensor.data_ptr() == right_tensor.data_ptr():
            return True
    return False
