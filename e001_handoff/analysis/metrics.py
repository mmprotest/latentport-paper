"""Numerically stable fidelity metrics shared by E001 phases."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


def compare_logits(reference: torch.Tensor, candidate: torch.Tensor, *, top_k: int = 5) -> dict[str, Any]:
    reference = reference.detach().float().flatten()
    candidate = candidate.detach().float().flatten()
    if reference.shape != candidate.shape:
        raise ValueError(f"Logit shapes differ: {reference.shape} vs {candidate.shape}")
    finite = bool(torch.isfinite(reference).all() and torch.isfinite(candidate).all())
    if not finite:
        return {
            "finite": False,
            "cosine": math.nan,
            "max_abs_diff": math.inf,
            "top1_agreement": False,
            "topk_overlap": math.nan,
        }
    reference_top = torch.topk(reference, k=top_k).indices
    candidate_top = torch.topk(candidate, k=top_k).indices
    overlap = torch.isin(reference_top, candidate_top).sum().item() / top_k
    return {
        "finite": True,
        "cosine": float(F.cosine_similarity(reference, candidate, dim=0).item()),
        "max_abs_diff": float((reference - candidate).abs().max().item()),
        "top1_agreement": bool(reference_top[0].item() == candidate_top[0].item()),
        "topk_overlap": float(overlap),
    }


def token_nll(logits: torch.Tensor, target_token_id: int) -> float:
    logits = logits.detach().float().flatten()
    return float(-F.log_softmax(logits, dim=-1)[target_token_id].item())


def normalized_frobenius(reference: torch.Tensor, candidate: torch.Tensor, eps: float = 1e-12) -> float:
    numerator = torch.linalg.vector_norm((candidate.float() - reference.float()).flatten())
    denominator = torch.linalg.vector_norm(reference.float().flatten()).clamp_min(eps)
    return float((numerator / denominator).item())


def flattened_cosine(reference: torch.Tensor, candidate: torch.Tensor) -> float:
    return float(F.cosine_similarity(reference.float().flatten(), candidate.float().flatten(), dim=0).item())


def delta_nll(condition_nll: float, native_nll: float) -> float:
    return float(condition_nll - native_nll)


def tqr(source_nll: float, full_nll: float, native_nll: float) -> float | None:
    denominator = source_nll - native_nll
    if denominator <= 0:
        return None
    return float((source_nll - full_nll) / denominator)


def paired_bootstrap_mean_ci(
    improvements: np.ndarray,
    *,
    resamples: int = 10_000,
    seed: int = 2026083103,
    level: float = 0.95,
) -> tuple[float, float]:
    values = np.asarray(improvements, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("Bootstrap input must be a non-empty finite one-dimensional array")
    generator = np.random.default_rng(seed)
    means = np.empty(resamples, dtype=np.float64)
    # Chunk draws to bound peak memory for large document counts.
    chunk = 1_000
    cursor = 0
    while cursor < resamples:
        take = min(chunk, resamples - cursor)
        indices = generator.integers(0, values.size, size=(take, values.size))
        means[cursor : cursor + take] = values[indices].mean(axis=1)
        cursor += take
    alpha = (1.0 - level) / 2.0
    low, high = np.quantile(means, [alpha, 1.0 - alpha])
    return float(low), float(high)

