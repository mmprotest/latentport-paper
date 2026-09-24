from __future__ import annotations

import torch

from experiments.latentport.e001_handoff.translators.ridge import (
    apply_batched_ridge,
    fit_batched_ridge,
)
from experiments.latentport.e001_handoff.translators.rope import (
    derotate_cached_key,
    rerotate_key,
)


def test_rope_round_trip_uses_runtime_bfloat16_coefficients():
    generator = torch.Generator().manual_seed(7)
    keys = torch.randn(13, 256, generator=generator)
    positions = torch.tensor([0, 1, 2, 3, 127, 128, 255, 511, 1023, 2048, 4095, 8191, 16383])
    rotated = rerotate_key(keys, positions)
    restored = derotate_cached_key(rotated, positions)
    assert torch.allclose(keys, restored, atol=2e-5, rtol=2e-5)


def test_ridge_reproducibly_recovers_linear_map():
    generator = torch.Generator().manual_seed(11)
    x = torch.randn(3, 200, 6, generator=generator)
    true_weight = torch.randn(3, 6, 5, generator=generator)
    bias = torch.randn(3, 5, generator=generator)
    y = torch.bmm(x, true_weight) + bias[:, None, :]
    xv = torch.randn(3, 50, 6, generator=generator)
    yv = torch.bmm(xv, true_weight) + bias[:, None, :]
    kwargs = {
        "lambda_grid": [1e-8, 1e-4],
        "normalization_names": ["center_only", "per_feature_zscore"],
    }
    first = fit_batched_ridge(x, y, xv, yv, **kwargs)
    second = fit_batched_ridge(x, y, xv, yv, **kwargs)
    prediction = apply_batched_ridge(xv, first)
    assert torch.allclose(prediction, yv, atol=2e-4, rtol=2e-4)
    assert torch.equal(first.weight, second.weight)
    assert torch.equal(first.lambda_relative, second.lambda_relative)
