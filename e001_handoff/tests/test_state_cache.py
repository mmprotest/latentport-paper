from __future__ import annotations

import torch
from transformers import DynamicCache

from experiments.latentport.e001_handoff.runtime.constants import SOURCE
from experiments.latentport.e001_handoff.runtime.modeling import load_text_config
from experiments.latentport.e001_handoff.state.cache_state import (
    caches_share_storage,
    capture_cache,
    restore_cache,
    state_checksum,
    state_nbytes,
)


def _synthetic_cache():
    config = load_text_config(SOURCE)
    cache = DynamicCache(config=config)
    for index, (layer_type, layer) in enumerate(zip(config.layer_types, cache.layers, strict=True)):
        if layer_type == "full_attention":
            keys = torch.full((1, 4, 7, 256), index + 0.25, dtype=torch.bfloat16)
            values = torch.full((1, 4, 7, 256), index + 0.5, dtype=torch.bfloat16)
            layer.update(keys, values)
        else:
            conv = torch.full((1, 8192, 4), index + 0.25, dtype=torch.bfloat16)
            recurrent = torch.full((1, 32, 128, 128), index + 0.5, dtype=torch.float32)
            layer.update_conv_state(conv, state_idx=0, conv_kernel_size=4)
            layer.update_recurrent_state(recurrent, state_idx=0)
    return config, cache


def test_deep_clone_semantics_and_checksum():
    config, original = _synthetic_cache()
    captured = capture_cache(original, config)
    restored = restore_cache(captured, config, device="cpu")
    assert not caches_share_storage(original, restored, config)
    assert state_checksum(captured) == state_checksum(capture_cache(restored, config))
    assert state_nbytes(captured) == 52_133_888
    original.layers[0].conv_states[0].zero_()
    assert torch.count_nonzero(restored.layers[0].conv_states[0]) > 0


def test_cache_position_is_inferred_from_attention_history():
    config, original = _synthetic_cache()
    restored = restore_cache(capture_cache(original, config), config, device="cpu")
    assert original.get_seq_length() == 7
    assert restored.get_seq_length() == 7
    assert all(
        layer.has_previous_state[0]
        for layer_type, layer in zip(config.layer_types, restored.layers, strict=True)
        if layer_type == "linear_attention"
    )
