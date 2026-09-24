"""Capture and compare only state components valid for convergence analysis."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


@torch.inference_mode()
def capture_convergence_state(cache, config, *, historical_prefix_length: int) -> dict[str, Any]:
    layers = []
    for layer_index, (layer_type, layer) in enumerate(zip(config.layer_types, cache.layers, strict=True)):
        if layer_type == "linear_attention":
            layers.append(
                {
                    "layer_index": layer_index,
                    "layer_type": layer_type,
                    "recurrent": layer.recurrent_states[0].detach().cpu().clone(),
                    "convolution": layer.conv_states[0].detach().cpu().clone(),
                }
            )
        else:
            if layer.get_seq_length() < historical_prefix_length:
                raise ValueError("Attention cache is shorter than the historical prefix")
            layers.append(
                {
                    "layer_index": layer_index,
                    "layer_type": layer_type,
                    "new_keys": layer.keys[:, :, historical_prefix_length:, :].detach().cpu().clone(),
                    "new_values": layer.values[:, :, historical_prefix_length:, :].detach().cpu().clone(),
                }
            )
    return {"layers": layers, "new_tokens": int(cache.get_seq_length() - historical_prefix_length)}


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(F.cosine_similarity(left.float().flatten(), right.float().flatten(), dim=0).item())


def _normalized_error(reference: torch.Tensor, candidate: torch.Tensor) -> float:
    denominator = torch.linalg.vector_norm(reference.float().flatten()).clamp_min(1e-12)
    return float(
        (torch.linalg.vector_norm((candidate.float() - reference.float()).flatten()) / denominator).item()
    )


def compare_convergence_states(reference: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    if reference["new_tokens"] != candidate["new_tokens"]:
        raise ValueError("Convergence states were captured after different token counts")
    gdn_layers = []
    attention_layers = []
    for native, handoff in zip(reference["layers"], candidate["layers"], strict=True):
        if native["layer_index"] != handoff["layer_index"] or native["layer_type"] != handoff["layer_type"]:
            raise ValueError("Convergence layer correspondence failure")
        if native["layer_type"] == "linear_attention":
            per_head_error = []
            per_head_cosine = []
            for head in range(native["recurrent"].shape[1]):
                per_head_error.append(
                    _normalized_error(native["recurrent"][0, head], handoff["recurrent"][0, head])
                )
                per_head_cosine.append(_cosine(native["recurrent"][0, head], handoff["recurrent"][0, head]))
            gdn_layers.append(
                {
                    "layer_index": native["layer_index"],
                    "recurrent_normalized_frobenius": _normalized_error(
                        native["recurrent"], handoff["recurrent"]
                    ),
                    "recurrent_cosine": _cosine(native["recurrent"], handoff["recurrent"]),
                    "recurrent_per_head_normalized_frobenius": per_head_error,
                    "recurrent_per_head_cosine": per_head_cosine,
                    "convolution_normalized_l2": _normalized_error(
                        native["convolution"], handoff["convolution"]
                    ),
                    "convolution_cosine": _cosine(native["convolution"], handoff["convolution"]),
                }
            )
        else:
            attention_layers.append(
                {
                    "layer_index": native["layer_index"],
                    "compared_historical_entries": 0,
                    "compared_new_entries": native["new_keys"].shape[-2],
                    "new_key_normalized_l2": _normalized_error(
                        native["new_keys"], handoff["new_keys"]
                    ),
                    "new_key_cosine": _cosine(native["new_keys"], handoff["new_keys"]),
                    "new_value_normalized_l2": _normalized_error(
                        native["new_values"], handoff["new_values"]
                    ),
                    "new_value_cosine": _cosine(native["new_values"], handoff["new_values"]),
                }
            )
    return {
        "new_tokens": reference["new_tokens"],
        "gdn_layers": gdn_layers,
        "attention_layers": attention_layers,
        "mean_gdn_recurrent_normalized_frobenius": sum(
            row["recurrent_normalized_frobenius"] for row in gdn_layers
        )
        / len(gdn_layers),
        "mean_gdn_recurrent_cosine": sum(row["recurrent_cosine"] for row in gdn_layers)
        / len(gdn_layers),
        "mean_convolution_normalized_l2": sum(row["convolution_normalized_l2"] for row in gdn_layers)
        / len(gdn_layers),
        "mean_new_key_normalized_l2": sum(row["new_key_normalized_l2"] for row in attention_layers)
        / len(attention_layers),
        "mean_new_value_normalized_l2": sum(row["new_value_normalized_l2"] for row in attention_layers)
        / len(attention_layers),
    }
