"""E002 continuation scoring, training logits, and 256-token state repair."""

from __future__ import annotations

from typing import Any

import torch

from experiments.latentport.e001_handoff.runtime.scoring import (
    compare_full_distributions,
    position_ids,
    score_from_cache_segmented,
)
from experiments.latentport.e001_handoff.state.convergence import capture_convergence_state


def score_primary(
    model,
    cache,
    *,
    prefix_length: int,
    bridge_token_id: int,
    continuation_token_ids: list[int],
    retain_logits: bool = True,
) -> dict[str, Any]:
    return score_from_cache_segmented(
        model,
        cache,
        prefix_length=prefix_length,
        bridge_token_id=bridge_token_id,
        continuation_token_ids=continuation_token_ids,
        retain_logits=retain_logits,
        capture_convergence=False,
    )


def compare_primary(candidate: dict[str, Any], native: dict[str, Any]) -> dict[str, Any]:
    return compare_full_distributions(candidate, native)


def training_logits(
    model,
    cache,
    *,
    prefix_length: int,
    bridge_token_id: int,
    continuation_token_ids: list[int],
) -> torch.Tensor:
    """Return nine logits: bridge output plus eight teacher-forced outputs."""

    if len(continuation_token_ids) < 9:
        raise ValueError("training requires at least nine continuation targets")
    device = next(model.parameters()).device
    input_ids = torch.tensor(
        [[bridge_token_id, *continuation_token_ids[:8]]], dtype=torch.long, device=device
    )
    output = model(
        input_ids=input_ids,
        position_ids=position_ids(prefix_length, 9, device),
        past_key_values=cache,
        use_cache=True,
        logits_to_keep=9,
    )
    if output.logits.shape[1] != 9:
        raise RuntimeError("training continuation accounting failed")
    return output.logits


@torch.inference_mode()
def capture_repair_trajectory(
    model,
    cache,
    *,
    prefix_length: int,
    bridge_token_id: int,
    continuation_token_ids: list[int],
) -> dict[int, dict[str, Any]]:
    if len(continuation_token_ids) < 255:
        raise ValueError("state repair requires 255 continuation input tokens")
    device = next(model.parameters()).device
    trajectory: dict[int, dict[str, Any]] = {}
    processed = 0
    for checkpoint in (1, 4, 16, 64, 256):
        if processed == 0:
            inputs = [bridge_token_id]
        else:
            start = processed - 1
            stop = checkpoint - 1
            inputs = continuation_token_ids[start:stop]
        token_tensor = torch.tensor([inputs], dtype=torch.long, device=device)
        model(
            input_ids=token_tensor,
            position_ids=position_ids(prefix_length + processed, len(inputs), device),
            past_key_values=cache,
            use_cache=True,
            logits_to_keep=1,
        )
        processed = checkpoint
        trajectory[checkpoint] = capture_convergence_state(
            cache, model.config, historical_prefix_length=prefix_length
        )
    return trajectory

