"""Independent K/R/C composition for the complete E002 factorial."""

from __future__ import annotations

from typing import Any

from experiments.latentport.e001_handoff.state.cache_state import STATE_FORMAT_VERSION
from experiments.latentport.e002_coupler.runtime.constants import FACTORIAL_CONDITIONS


def condition_components(condition: str) -> dict[str, str]:
    if condition not in FACTORIAL_CONDITIONS:
        raise ValueError(f"unknown factorial condition: {condition}")
    return {"K": condition[0], "R": condition[1], "C": condition[2]}


def condition_complexity(condition: str) -> int:
    condition_components(condition)
    return condition.count("T")


def compose_condition(
    source_state: dict[str, Any], translated_state: dict[str, Any], condition: str
) -> dict[str, Any]:
    components = condition_components(condition)
    if source_state["layer_types"] != translated_state["layer_types"]:
        raise ValueError("source/translated layer patterns differ")
    if source_state["sequence_length"] != translated_state["sequence_length"]:
        raise ValueError("source/translated logical prefix lengths differ")
    layers = []
    for direct, translated in zip(source_state["layers"], translated_state["layers"], strict=True):
        if direct["layer_index"] != translated["layer_index"] or direct["layer_type"] != translated["layer_type"]:
            raise ValueError("source/translated layer correspondence failed")
        if direct["layer_type"] == "full_attention":
            layers.append(direct if components["K"] == "D" else translated)
            continue
        recurrent_source = direct if components["R"] == "D" else translated
        convolution_source = direct if components["C"] == "D" else translated
        layers.append(
            {
                "layer_index": direct["layer_index"],
                "layer_type": direct["layer_type"],
                "cache_class": translated["cache_class"],
                "number_of_states": 1,
                "conv_states": {0: convolution_source["conv_states"][0]},
                "recurrent_states": {0: recurrent_source["recurrent_states"][0]},
                "is_conv_states_initialized": {0: True},
                "is_recurrent_states_initialized": {0: True},
                "has_previous_state": {0: True},
                "conv_kernel_size": {0: 4},
                "record_past": False,
            }
        )
    return {
        "format_version": STATE_FORMAT_VERSION,
        "num_hidden_layers": 32,
        "layer_types": list(source_state["layer_types"]),
        "sequence_length": int(source_state["sequence_length"]),
        "layers": layers,
        "e002_condition": condition,
    }

