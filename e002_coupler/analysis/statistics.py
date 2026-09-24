"""Document-level repeated-measures contrasts and paired bootstrap inference."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from experiments.latentport.e002_coupler.runtime.constants import FACTORIAL_CONDITIONS


def bootstrap_mean_ci(
    values: np.ndarray,
    *,
    resamples: int = 10_000,
    seed: int = 2026090103,
    level: float = 0.95,
) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("bootstrap values must be finite, non-empty, and one-dimensional")
    generator = np.random.default_rng(seed)
    means = np.empty(resamples, dtype=np.float64)
    cursor = 0
    while cursor < resamples:
        take = min(1000, resamples - cursor)
        indices = generator.integers(0, values.size, size=(take, values.size))
        means[cursor : cursor + take] = values[indices].mean(axis=1)
        cursor += take
    alpha = (1.0 - level) / 2.0
    low, high = np.quantile(means, [alpha, 1.0 - alpha])
    return float(low), float(high)


def bootstrap_statistic_ci(
    rows: np.ndarray,
    statistic: Callable[[np.ndarray], float],
    *,
    resamples: int = 10_000,
    seed: int = 2026090103,
    level: float = 0.95,
) -> tuple[float, float]:
    rows = np.asarray(rows, dtype=np.float64)
    if rows.ndim < 1 or rows.shape[0] == 0 or not np.isfinite(rows).all():
        raise ValueError("bootstrap rows must be finite and non-empty")
    generator = np.random.default_rng(seed)
    samples = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        selection = generator.integers(0, rows.shape[0], size=rows.shape[0])
        samples[index] = statistic(rows[selection])
    alpha = (1.0 - level) / 2.0
    low, high = np.quantile(samples, [alpha, 1.0 - alpha])
    return float(low), float(high)


def factorial_document_contrasts(outcomes: dict[str, float]) -> dict[str, float]:
    if set(outcomes) != set(FACTORIAL_CONDITIONS):
        raise ValueError("factorial outcomes must contain exactly eight conditions")
    codes = {condition: np.asarray([1.0 if value == "T" else -1.0 for value in condition]) for condition in FACTORIAL_CONDITIONS}
    y = {condition: float(outcomes[condition]) for condition in FACTORIAL_CONDITIONS}
    result = {}
    names = ("KV", "recurrent", "convolution")
    for axis, name in enumerate(names):
        result[name] = sum(y[c] * codes[c][axis] for c in FACTORIAL_CONDITIONS) / 4.0
    for left, right, name in (
        (0, 1, "KV_x_recurrent"),
        (0, 2, "KV_x_convolution"),
        (1, 2, "recurrent_x_convolution"),
    ):
        # Difference-in-differences, averaged over the remaining factor.
        result[name] = sum(
            y[c] * codes[c][left] * codes[c][right] for c in FACTORIAL_CONDITIONS
        ) / 2.0
    result["KV_x_recurrent_x_convolution"] = sum(
        y[c] * float(np.prod(codes[c])) for c in FACTORIAL_CONDITIONS
    )
    return result


def summarize_factorial(
    document_outcomes: list[dict[str, float]], *, seed: int = 2026090103
) -> dict[str, Any]:
    contrasts = [factorial_document_contrasts(row) for row in document_outcomes]
    effects = {}
    for name in contrasts[0]:
        values = np.asarray([row[name] for row in contrasts], dtype=np.float64)
        effects[name] = {
            "estimate": float(values.mean()),
            "median_document_effect": float(np.median(values)),
            "bootstrap_ci": list(bootstrap_mean_ci(values, seed=seed)),
            "document_effects": values.tolist(),
        }
    return {
        "method": "orthogonal within-document repeated-measures contrasts with document bootstrap",
        "documents": len(document_outcomes),
        "effects": effects,
    }

