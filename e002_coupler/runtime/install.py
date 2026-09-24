"""Canonical inference and differentiable training cache installation."""

from __future__ import annotations

from types import MethodType
from typing import Any

from transformers import DynamicCache

from experiments.latentport.e001_handoff.state.cache_state import restore_cache


def install_for_inference(state: dict[str, Any], config, *, device: str = "cuda:0") -> DynamicCache:
    return restore_cache(state, config, device=device)


def _assign_recurrent(self, recurrent_states, state_idx: int = 0, **kwargs):
    self.recurrent_states[state_idx] = recurrent_states
    self.is_recurrent_states_initialized[state_idx] = True
    return recurrent_states


def install_differentiable(state: dict[str, Any], config) -> DynamicCache:
    """Install state without detach/copy barriers on correction gradients.

    Linear-cache updates are switched from static-address in-place copies to
    graph-preserving assignment. This path is used only for nine-token FIT and
    VALIDATION calls; canonical inference continues to use the sealed E001
    restore implementation.
    """

    if state["layer_types"] != list(config.layer_types):
        raise ValueError("differentiable state layer pattern mismatch")
    cache = DynamicCache(config=config)
    for record, layer in zip(state["layers"], cache.layers, strict=True):
        if record["layer_type"] == "full_attention":
            layer.keys = record["keys"]
            layer.values = record["values"]
            layer.dtype = record["keys"].dtype
            layer.device = record["keys"].device
            layer.is_initialized = True
        else:
            layer.conv_states[0] = record["conv_states"][0]
            layer.recurrent_states[0] = record["recurrent_states"][0]
            layer.is_conv_states_initialized[0] = True
            layer.is_recurrent_states_initialized[0] = True
            layer.has_previous_state[0] = True
            layer.conv_kernel_size[0] = 4
            # Assignment avoids the convolution state's static-address copy.
            layer.record_past = True
            layer.device = record["recurrent_states"][0].device
            layer.dtype = record["conv_states"][0].dtype
            layer.update_recurrent_state = MethodType(_assign_recurrent, layer)
    if cache.get_seq_length() != state["sequence_length"]:
        raise RuntimeError("differentiable cache logical length mismatch")
    return cache

