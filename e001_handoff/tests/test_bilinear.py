from __future__ import annotations

import torch

from experiments.latentport.e001_handoff.translators.bilinear import (
    apply_bilinear,
    fit_bilinear_layer,
)


def test_bilinear_shape_and_reproducibility():
    generator = torch.Generator().manual_seed(23)
    heads, samples, dimension = 2, 80, 5
    x = torch.randn(heads, samples, dimension, dimension, generator=generator)
    a = torch.eye(dimension).expand(heads, -1, -1).clone()
    b = torch.eye(dimension).expand(heads, -1, -1).clone()
    a += 0.05 * torch.randn(a.shape, generator=generator)
    b += 0.05 * torch.randn(b.shape, generator=generator)
    mu_x = torch.randn(heads, dimension, dimension, generator=generator)
    mu_y = torch.randn(heads, dimension, dimension, generator=generator)
    y = mu_y[:, None] + torch.matmul(
        torch.matmul(a[:, None], x - mu_x[:, None]), b[:, None].transpose(-1, -2)
    )
    xv = torch.randn(heads, 20, dimension, dimension, generator=generator)
    yv = mu_y[:, None] + torch.matmul(
        torch.matmul(a[:, None], xv - mu_x[:, None]), b[:, None].transpose(-1, -2)
    )
    kwargs = {"lambda_grid": [1e-8, 1e-4], "alternations": 3}
    first = fit_bilinear_layer(x, y, xv, yv, **kwargs)
    second = fit_bilinear_layer(x, y, xv, yv, **kwargs)
    assert first.key_map.shape == (heads, dimension, dimension)
    assert first.value_map.shape == (heads, dimension, dimension)
    assert torch.equal(first.key_map, second.key_map)
    assert torch.equal(first.value_map, second.value_map)
    predictions = torch.stack([apply_bilinear(xv[head], type(first)(
        key_map=first.key_map[head],
        value_map=first.value_map[head],
        source_mean=first.source_mean[head],
        target_mean=first.target_mean[head],
        lambda_relative=first.lambda_relative[head],
        validation_error=first.validation_error[head],
        direct_validation_error=first.direct_validation_error[head],
        mean_validation_error=first.mean_validation_error[head],
    )) for head in range(heads)])
    assert torch.allclose(predictions, yv, atol=2e-3, rtol=2e-3)
