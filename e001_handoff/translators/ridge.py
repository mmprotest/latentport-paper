"""Deterministic batched ridge fitting for E001 linear state maps."""

from __future__ import annotations

from dataclasses import dataclass

import torch


NORMALIZATION_NAMES = ("center_only", "per_feature_zscore")


@dataclass
class BatchedRidgeResult:
    weight: torch.Tensor
    source_mean: torch.Tensor
    target_mean: torch.Tensor
    source_scale: torch.Tensor
    target_scale: torch.Tensor
    lambda_relative: torch.Tensor
    normalization_code: torch.Tensor
    validation_error: torch.Tensor
    condition_number: torch.Tensor


def _per_sample_normalized_l2(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    difference = torch.linalg.vector_norm(prediction - target, dim=-1)
    denominator = torch.linalg.vector_norm(target, dim=-1).clamp_min(1e-12)
    return (difference / denominator).mean(dim=1)


def fit_batched_ridge(
    fit_source: torch.Tensor,
    fit_target: torch.Tensor,
    validation_source: torch.Tensor,
    validation_target: torch.Tensor,
    *,
    lambda_grid: list[float],
    normalization_names: list[str],
) -> BatchedRidgeResult:
    """Fit independent maps for shape ``[groups, samples, features]``."""

    tensors = (fit_source, fit_target, validation_source, validation_target)
    if any(tensor.ndim != 3 for tensor in tensors):
        raise ValueError("Batched ridge tensors must have [group, sample, feature] shape")
    groups, _, source_dim = fit_source.shape
    target_dim = fit_target.shape[-1]
    if validation_source.shape[0] != groups or validation_target.shape[0] != groups:
        raise ValueError("FIT/VALIDATION group counts differ")
    source_mean = fit_source.mean(dim=1)
    target_mean = fit_target.mean(dim=1)
    source_centered = fit_source - source_mean[:, None, :]
    target_centered = fit_target - target_mean[:, None, :]
    validation_source_centered = validation_source - source_mean[:, None, :]

    best_error = torch.full((groups,), torch.inf, device=fit_source.device)
    best_condition = torch.full((groups,), torch.inf, device=fit_source.device)
    best_lambda = torch.zeros((groups,), device=fit_source.device)
    best_normalization = torch.zeros((groups,), dtype=torch.int64, device=fit_source.device)
    best_weight = torch.zeros((groups, source_dim, target_dim), device=fit_source.device)
    best_source_scale = torch.ones((groups, source_dim), device=fit_source.device)
    best_target_scale = torch.ones((groups, target_dim), device=fit_source.device)
    identity = torch.eye(source_dim, device=fit_source.device).expand(groups, -1, -1)

    for normalization_code, normalization_name in enumerate(normalization_names):
        if normalization_name not in NORMALIZATION_NAMES:
            raise ValueError(f"Unknown preregistered normalization {normalization_name}")
        if normalization_name == "center_only":
            source_scale = torch.ones_like(source_mean)
            target_scale = torch.ones_like(target_mean)
        else:
            source_scale = source_centered.square().mean(dim=1).sqrt().clamp_min(1e-6)
            target_scale = target_centered.square().mean(dim=1).sqrt().clamp_min(1e-6)
        x = source_centered / source_scale[:, None, :]
        y = target_centered / target_scale[:, None, :]
        gram = torch.bmm(x.transpose(1, 2), x)
        cross = torch.bmm(x.transpose(1, 2), y)
        ridge_scale = gram.diagonal(dim1=-2, dim2=-1).sum(dim=-1) / source_dim
        validation_x = validation_source_centered / source_scale[:, None, :]
        for lambda_relative in lambda_grid:
            regularized = gram + (
                float(lambda_relative) * ridge_scale.clamp_min(1e-12)
            )[:, None, None] * identity
            weight = torch.linalg.solve(regularized, cross)
            prediction = (
                torch.bmm(validation_x, weight) * target_scale[:, None, :] + target_mean[:, None, :]
            )
            error = _per_sample_normalized_l2(prediction, validation_target)
            condition = torch.linalg.cond(regularized)
            better_error = error < (best_error - 1e-12)
            tied_error = torch.isclose(error, best_error, rtol=0.0, atol=1e-12)
            better_condition = condition < (best_condition - 1e-9)
            tied_condition = torch.isclose(condition, best_condition, rtol=0.0, atol=1e-9)
            smaller_lambda = float(lambda_relative) < best_lambda
            update = better_error | (
                tied_error & (better_condition | (tied_condition & smaller_lambda))
            )
            if update.any():
                best_error = torch.where(update, error, best_error)
                best_condition = torch.where(update, condition, best_condition)
                best_lambda = torch.where(
                    update, torch.full_like(best_lambda, float(lambda_relative)), best_lambda
                )
                best_normalization = torch.where(
                    update,
                    torch.full_like(best_normalization, normalization_code),
                    best_normalization,
                )
                best_weight[update] = weight[update]
                best_source_scale[update] = source_scale[update]
                best_target_scale[update] = target_scale[update]

    return BatchedRidgeResult(
        weight=best_weight,
        source_mean=source_mean,
        target_mean=target_mean,
        source_scale=best_source_scale,
        target_scale=best_target_scale,
        lambda_relative=best_lambda,
        normalization_code=best_normalization,
        validation_error=best_error,
        condition_number=best_condition,
    )


def apply_batched_ridge(source: torch.Tensor, result: BatchedRidgeResult) -> torch.Tensor:
    normalized = (source - result.source_mean[:, None, :]) / result.source_scale[:, None, :]
    return (
        torch.bmm(normalized, result.weight) * result.target_scale[:, None, :]
        + result.target_mean[:, None, :]
    )

