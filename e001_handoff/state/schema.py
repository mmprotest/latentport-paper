"""Generate the exact persistent-state schema from a populated runtime cache."""

from __future__ import annotations

from typing import Any

import torch


def _tensor_record(
    *,
    field_name: str,
    layer_index: int,
    layer_type: str,
    tensor_role: str,
    tensor: torch.Tensor,
    semantic_interpretation: str,
    restore_method: str,
) -> dict[str, Any]:
    return {
        "field_name": field_name,
        "layer_index": layer_index,
        "layer_type": layer_type,
        "tensor_role": tensor_role,
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype).removeprefix("torch."),
        "device": str(tensor.device),
        "byte_size": tensor.numel() * tensor.element_size(),
        "semantic_interpretation": semantic_interpretation,
        "restore_method": restore_method,
    }


def _metadata_record(
    *,
    field_name: str,
    layer_index: int,
    layer_type: str,
    value: Any,
    semantic_interpretation: str,
    restore_method: str,
) -> dict[str, Any]:
    return {
        "field_name": field_name,
        "layer_index": layer_index,
        "layer_type": layer_type,
        "tensor_role": "structural_metadata",
        "shape": [],
        "dtype": type(value).__name__,
        "device": "host",
        "byte_size": 0,
        "value": value,
        "semantic_interpretation": semantic_interpretation,
        "restore_method": restore_method,
    }


def schema_from_cache(cache, config, *, model_repository: str, model_revision: str) -> dict[str, Any]:
    fields: list[dict[str, Any]] = []
    for layer_index, (layer_type, layer) in enumerate(zip(config.layer_types, cache.layers, strict=True)):
        if layer_type == "full_attention":
            fields.extend(
                [
                    _tensor_record(
                        field_name="keys",
                        layer_index=layer_index,
                        layer_type=layer_type,
                        tensor_role="full_attention_rotary_embedded_key_cache",
                        tensor=layer.keys,
                        semantic_interpretation=(
                            "RoPE-embedded attention keys with axes batch, KV head, historical position, head dimension"
                        ),
                        restore_method="Deep-clone into target DynamicLayer.keys and set is_initialized/dtype/device",
                    ),
                    _tensor_record(
                        field_name="values",
                        layer_index=layer_index,
                        layer_type=layer_type,
                        tensor_role="full_attention_value_cache",
                        tensor=layer.values,
                        semantic_interpretation=(
                            "Attention values with axes batch, KV head, historical position, head dimension"
                        ),
                        restore_method="Deep-clone into target DynamicLayer.values and set is_initialized/dtype/device",
                    ),
                    _metadata_record(
                        field_name="is_initialized",
                        layer_index=layer_index,
                        layer_type=layer_type,
                        value=bool(layer.is_initialized),
                        semantic_interpretation="Whether K/V storage is populated",
                        restore_method="Assign after K/V tensor installation",
                    ),
                ]
            )
        elif layer_type == "linear_attention":
            fields.extend(
                [
                    _tensor_record(
                        field_name="conv_states[0]",
                        layer_index=layer_index,
                        layer_type=layer_type,
                        tensor_role="gated_deltanet_qkv_short_memory",
                        tensor=layer.conv_states[0],
                        semantic_interpretation=(
                            "Last four unactivated packed Q, K, V projection vectors; channel order is Q[2048], K[2048], V[4096]"
                        ),
                        restore_method=(
                            "LinearAttentionLayer.lazy_initialization(conv_states=..., conv_kernel_size=4), then copy_"
                        ),
                    ),
                    _tensor_record(
                        field_name="recurrent_states[0]",
                        layer_index=layer_index,
                        layer_type=layer_type,
                        tensor_role="gated_deltanet_recurrent_matrix",
                        tensor=layer.recurrent_states[0],
                        semantic_interpretation=(
                            "Persistent delta-rule memory with axes batch, value head, normalized key dimension, value dimension"
                        ),
                        restore_method=(
                            "LinearAttentionLayer.lazy_initialization(recurrent_states=...), then copy_"
                        ),
                    ),
                ]
            )
            metadata = [
                ("number_of_states", int(layer.number_of_states), "Number of independent recurrent/conv states"),
                ("is_conv_states_initialized[0]", bool(layer.is_conv_states_initialized[0]), "Conv storage initialized flag"),
                (
                    "is_recurrent_states_initialized[0]",
                    bool(layer.is_recurrent_states_initialized[0]),
                    "Recurrent storage initialized flag",
                ),
                ("has_previous_state[0]", bool(layer.has_previous_state[0]), "Selects cached decode rather than fresh prefill"),
                ("conv_kernel_size[0]", int(layer.conv_kernel_size[0]), "Required short-memory width"),
                ("record_past", bool(layer.record_past), "Whether rollback history rather than minimal state is retained"),
            ]
            for name, value, meaning in metadata:
                fields.append(
                    _metadata_record(
                        field_name=name,
                        layer_index=layer_index,
                        layer_type=layer_type,
                        value=value,
                        semantic_interpretation=meaning,
                        restore_method="Assign exact captured value after target layer initialization",
                    )
                )
        else:
            raise ValueError(f"Unexpected layer type {layer_type!r}")

    fields.append(
        _metadata_record(
            field_name="cache.sequence_length",
            layer_index=-1,
            layer_type="global",
            value=int(cache.get_seq_length()),
            semantic_interpretation=(
                "Logical historical length inferred by DynamicCache from the first full-attention K tensor"
            ),
            restore_method="Construct K/V tensors at exact historical length; verify get_seq_length()",
        )
    )
    tensor_bytes = sum(record["byte_size"] for record in fields)
    return {
        "schema_version": "latentport-e001-state-schema-v1",
        "model_repository": model_repository,
        "model_revision": model_revision,
        "runtime_cache_class": type(cache).__name__,
        "language_layers": int(config.num_hidden_layers),
        "layer_type_pattern": list(config.layer_types),
        "gated_deltanet_layers": sum(value == "linear_attention" for value in config.layer_types),
        "full_attention_layers": sum(value == "full_attention" for value in config.layer_types),
        "total_tensor_bytes_at_probe_length": tensor_bytes,
        "fields": fields,
    }

