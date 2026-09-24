"""Exact E001 de-rotation/re-rotation for cached Qwen3.5 text keys."""

from __future__ import annotations

import torch


ROTARY_DIM = 64
HEAD_DIM = 256
ROPE_THETA = 10_000_000.0


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    first, second = x.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def rope_coefficients(positions: torch.Tensor, *, device=None) -> tuple[torch.Tensor, torch.Tensor]:
    """Reproduce runtime coefficients for pure text, including BF16 rounding."""

    if device is None:
        device = positions.device
    inv_freq = 1.0 / (
        ROPE_THETA ** (torch.arange(0, ROTARY_DIM, 2, device=device, dtype=torch.float32) / ROTARY_DIM)
    )
    frequencies = positions.to(device=device, dtype=torch.float32).unsqueeze(-1) * inv_freq
    embedding = torch.cat((frequencies, frequencies), dim=-1)
    # The runtime casts cos/sin to the BF16 key dtype before multiplication.
    return embedding.cos().to(torch.bfloat16).float(), embedding.sin().to(torch.bfloat16).float()


def derotate_cached_key(keys: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    if keys.shape[-1] != HEAD_DIM:
        raise ValueError(f"Expected head dimension {HEAD_DIM}, got {keys.shape[-1]}")
    cos, sin = rope_coefficients(positions, device=keys.device)
    rotary, passthrough = keys.float().split((ROTARY_DIM, HEAD_DIM - ROTARY_DIM), dim=-1)
    # BF16-rounded coefficients do not satisfy cos^2+sin^2 == 1 exactly.
    # Divide by the pair norm to invert the actual 2x2 runtime transform.
    denominator = (cos.square() + sin.square()).clamp_min(1e-12)
    unrotated = (rotary * cos - rotate_half(rotary) * sin) / denominator
    return torch.cat((unrotated, passthrough), dim=-1)


def rerotate_key(keys: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    if keys.shape[-1] != HEAD_DIM:
        raise ValueError(f"Expected head dimension {HEAD_DIM}, got {keys.shape[-1]}")
    cos, sin = rope_coefficients(positions, device=keys.device)
    rotary, passthrough = keys.float().split((ROTARY_DIM, HEAD_DIM - ROTARY_DIM), dim=-1)
    rotated = rotary * cos + rotate_half(rotary) * sin
    return torch.cat((rotated, passthrough), dim=-1)

