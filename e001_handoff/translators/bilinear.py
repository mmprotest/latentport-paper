"""Canonical bilinear ridge translator for Gated DeltaNet recurrent state."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class BilinearLayerResult:
    key_map: torch.Tensor
    value_map: torch.Tensor
    source_mean: torch.Tensor
    target_mean: torch.Tensor
    lambda_relative: torch.Tensor
    validation_error: torch.Tensor
    direct_validation_error: torch.Tensor
    mean_validation_error: torch.Tensor


def _validation_error(
    source_centered: torch.Tensor,
    target_centered: torch.Tensor,
    target_raw_norm: torch.Tensor,
    key_map: torch.Tensor,
    value_map: torch.Tensor,
) -> torch.Tensor:
    prediction = torch.matmul(key_map[:, None], source_centered)
    prediction = torch.matmul(prediction, value_map[:, None].transpose(-1, -2))
    difference = torch.linalg.matrix_norm(prediction - target_centered, ord="fro", dim=(-2, -1))
    return (difference / target_raw_norm).mean(dim=1)


def _ridge_solve(gram: torch.Tensor, cross: torch.Tensor, lambda_relative: float) -> torch.Tensor:
    dimension = gram.shape[-1]
    ridge_scale = gram.diagonal(dim1=-2, dim2=-1).sum(dim=-1) / dimension
    identity = torch.eye(dimension, dtype=gram.dtype, device=gram.device).expand(gram.shape[0], -1, -1)
    regularized = gram + (
        float(lambda_relative) * ridge_scale.clamp_min(1e-12)
    )[:, None, None] * identity
    return torch.linalg.solve(regularized, cross)


def fit_bilinear_layer(
    fit_source: torch.Tensor,
    fit_target: torch.Tensor,
    validation_source: torch.Tensor,
    validation_target: torch.Tensor,
    *,
    lambda_grid: list[float],
    alternations: int = 3,
) -> BilinearLayerResult:
    """Fit all heads for one layer.

    Inputs use ``[head, checkpoint, key_dimension, value_dimension]``. The
    canonical geometry has 32 heads and 128x128 matrices, but the solver is
    dimension-generic so small deterministic unit tests can audit it.
    """

    if any(tensor.ndim != 4 for tensor in (fit_source, fit_target, validation_source, validation_target)):
        raise ValueError("Bilinear tensors must use [head, sample, key, value] shape")
    heads, _, source_key_dim, source_value_dim = fit_source.shape
    target_key_dim, target_value_dim = fit_target.shape[-2:]
    if (source_key_dim, source_value_dim) != (target_key_dim, target_value_dim):
        raise ValueError("E001 bilinear identity initialization requires matching recurrent geometry")
    source_mean = fit_source.mean(dim=1)
    target_mean = fit_target.mean(dim=1)
    fit_source = fit_source - source_mean[:, None]
    fit_target = fit_target - target_mean[:, None]
    target_raw_norm = torch.linalg.matrix_norm(validation_target, ord="fro", dim=(-2, -1)).clamp_min(1e-12)
    validation_source_centered = validation_source - source_mean[:, None]
    validation_target_centered = validation_target - target_mean[:, None]
    direct_error = (
        torch.linalg.matrix_norm(validation_source - validation_target, ord="fro", dim=(-2, -1))
        / target_raw_norm
    ).mean(dim=1)
    mean_error = (
        torch.linalg.matrix_norm(-validation_target_centered, ord="fro", dim=(-2, -1))
        / target_raw_norm
    ).mean(dim=1)

    best_error = torch.full((heads,), torch.inf, device=fit_source.device)
    best_lambda = torch.zeros((heads,), device=fit_source.device)
    best_key_map = torch.zeros((heads, target_key_dim, source_key_dim), device=fit_source.device)
    best_value_map = torch.zeros((heads, target_value_dim, source_value_dim), device=fit_source.device)
    identity_key = torch.eye(source_key_dim, device=fit_source.device).expand(heads, -1, -1)
    identity_value = torch.eye(source_value_dim, device=fit_source.device).expand(heads, -1, -1)

    for lambda_relative in lambda_grid:
        key_map = identity_key.clone()
        value_map = identity_value.clone()
        for _ in range(alternations):
            # With B fixed, solve Y ~= A Z where Z = X B^T. Sum over
            # checkpoint and value-column axes to obtain the key-space normal equations.
            z = torch.matmul(fit_source, value_map[:, None].transpose(-1, -2))
            key_gram = torch.einsum("hniv,hnjv->hij", z, z)
            key_cross = torch.einsum("hniv,hnjv->hij", z, fit_target)
            key_map_transpose = _ridge_solve(key_gram, key_cross, float(lambda_relative))
            key_map = key_map_transpose.transpose(-1, -2).contiguous()
            del z, key_gram, key_cross, key_map_transpose

            # With A fixed, solve Y ~= W B^T where W = A X. Rows across
            # checkpoint and target-key axes are independent ridge samples.
            w = torch.matmul(key_map[:, None], fit_source)
            w_rows = w.reshape(heads, -1, source_value_dim)
            y_rows = fit_target.reshape(heads, -1, target_value_dim)
            value_gram = torch.bmm(w_rows.transpose(1, 2), w_rows)
            value_cross = torch.bmm(w_rows.transpose(1, 2), y_rows)
            value_map_transpose = _ridge_solve(value_gram, value_cross, float(lambda_relative))
            value_map = value_map_transpose.transpose(-1, -2).contiguous()
            del w, w_rows, value_gram, value_cross, value_map_transpose

        error = _validation_error(
            validation_source_centered,
            validation_target_centered,
            target_raw_norm,
            key_map,
            value_map,
        )
        if not torch.isfinite(error).all():
            raise FloatingPointError(f"Non-finite recurrent validation error at lambda {lambda_relative}")
        better = error < (best_error - 1e-12)
        tied = torch.isclose(error, best_error, rtol=0.0, atol=1e-12)
        update = better | (tied & (float(lambda_relative) < best_lambda))
        if update.any():
            best_error = torch.where(update, error, best_error)
            best_lambda = torch.where(
                update, torch.full_like(best_lambda, float(lambda_relative)), best_lambda
            )
            best_key_map[update] = key_map[update]
            best_value_map[update] = value_map[update]
        del key_map, value_map, error

    return BilinearLayerResult(
        key_map=best_key_map,
        value_map=best_value_map,
        source_mean=source_mean,
        target_mean=target_mean,
        lambda_relative=best_lambda,
        validation_error=best_error,
        direct_validation_error=direct_error,
        mean_validation_error=mean_error,
    )


def apply_bilinear(source: torch.Tensor, result: BilinearLayerResult) -> torch.Tensor:
    centered = source - result.source_mean
    return result.target_mean + torch.matmul(
        torch.matmul(result.key_map, centered), result.value_map.transpose(-1, -2)
    )

