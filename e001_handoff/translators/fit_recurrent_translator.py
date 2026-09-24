"""Fit the frozen per-layer/head GDN bilinear recurrent-state translator."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import save_file

from experiments.latentport.e001_handoff.runtime.constants import ATTEMPT_ROOT, E001_ROOT, load_preregistration
from experiments.latentport.e001_handoff.runtime.modeling import seed_everything
from experiments.latentport.e001_handoff.translators.bilinear import fit_bilinear_layer


TRANSLATOR_ROOT = E001_ROOT / "translators" / "frozen"


def _array(split: str, role: str):
    return np.load(
        ATTEMPT_ROOT / split / "paired_states" / f"{role}_recurrent.npy",
        mmap_mode="r",
    )


def _load_layer(array, layer_position: int) -> torch.Tensor:
    cpu = np.array(array[:, layer_position], dtype=np.float32, copy=True, order="C")
    gpu = torch.from_numpy(cpu).to("cuda:0")
    del cpu
    result = gpu.permute(1, 0, 2, 3).contiguous()
    del gpu
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    prereg = load_preregistration()
    seed_everything(prereg["seeds"]["global"])
    fit_source_map = _array("fit", "source")
    fit_target_map = _array("fit", "target")
    validation_source_map = _array("validation", "source")
    validation_target_map = _array("validation", "target")
    expected_fit_shape = (1024, 24, 32, 128, 128)
    expected_validation_shape = (256, 24, 32, 128, 128)
    if fit_source_map.shape != expected_fit_shape or fit_target_map.shape != expected_fit_shape:
        raise RuntimeError("Unexpected FIT recurrent array shape")
    if (
        validation_source_map.shape != expected_validation_shape
        or validation_target_map.shape != expected_validation_shape
    ):
        raise RuntimeError("Unexpected VALIDATION recurrent array shape")

    layer_results = []
    diagnostics: list[dict[str, Any]] = []
    language_layers = [index for index in range(32) if index % 4 != 3]
    for layer_position, language_layer in enumerate(language_layers):
        fit_source = _load_layer(fit_source_map, layer_position)
        fit_target = _load_layer(fit_target_map, layer_position)
        validation_source = _load_layer(validation_source_map, layer_position)
        validation_target = _load_layer(validation_target_map, layer_position)
        result = fit_bilinear_layer(
            fit_source,
            fit_target,
            validation_source,
            validation_target,
            lambda_grid=prereg["gdn_recurrent_translator"]["ridge_relative_grid"],
            alternations=3,
        )
        layer_results.append(result)
        for head in range(32):
            diagnostics.append(
                {
                    "language_layer": language_layer,
                    "value_head": head,
                    "lambda_relative": float(result.lambda_relative[head].item()),
                    "validation_normalized_frobenius": float(result.validation_error[head].item()),
                    "direct_copy_validation_normalized_frobenius": float(
                        result.direct_validation_error[head].item()
                    ),
                    "mean_state_validation_normalized_frobenius": float(
                        result.mean_validation_error[head].item()
                    ),
                }
            )
        print(
            f"GDN recurrent layer {language_layer}: "
            f"bilinear={result.validation_error.mean().item():.6f} "
            f"direct={result.direct_validation_error.mean().item():.6f} "
            f"mean={result.mean_validation_error.mean().item():.6f}",
            flush=True,
        )
        del fit_source, fit_target, validation_source, validation_target
        torch.cuda.empty_cache()

    fields = (
        "key_map",
        "value_map",
        "source_mean",
        "target_mean",
        "lambda_relative",
        "validation_error",
        "direct_validation_error",
        "mean_validation_error",
    )
    tensors = {
        field: torch.stack([getattr(result, field).detach().cpu() for result in layer_results], dim=0)
        for field in fields
    }
    TRANSLATOR_ROOT.mkdir(parents=True, exist_ok=True)
    output = TRANSLATOR_ROOT / "gdn_recurrent_translator.safetensors"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite translator: {output}")
    save_file(tensors, output, metadata={"format": "latentport-e001-gdn-bilinear-v1"})
    metadata = {
        "component": "gdn_recurrent_state",
        "formula": "S9_hat = mu9 + A (S4 - mu4) B^T",
        "granularity": "per corresponding GDN language layer and value head",
        "fit_samples_per_map": 1024,
        "validation_samples_per_map": 256,
        "fit_algorithm": "identity initialization followed by exactly three alternating ridge updates of A then B per candidate",
        "normalization": "center_only",
        "lambda_grid": prereg["gdn_recurrent_translator"]["ridge_relative_grid"],
        "nonlinear_components": None,
        "maps": diagnostics,
        "tensor_file": str(output.relative_to(E001_ROOT)).replace("\\", "/"),
        "tensor_file_sha256": _sha256(output),
        "parameter_count_weights": tensors["key_map"].numel() + tensors["value_map"].numel(),
        "fit_mean_values": tensors["source_mean"].numel() + tensors["target_mean"].numel(),
        "stored_numeric_values": sum(tensor.numel() for tensor in tensors.values()),
        "tensor_bytes": output.stat().st_size,
        "mean_validation_normalized_frobenius": float(tensors["validation_error"].mean().item()),
        "mean_direct_copy_validation_normalized_frobenius": float(
            tensors["direct_validation_error"].mean().item()
        ),
        "mean_mean_state_validation_normalized_frobenius": float(
            tensors["mean_validation_error"].mean().item()
        ),
    }
    metadata_path = TRANSLATOR_ROOT / "gdn_recurrent_translator.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "component": metadata["component"],
                "parameter_count_weights": metadata["parameter_count_weights"],
                "tensor_bytes": metadata["tensor_bytes"],
                "tensor_file_sha256": metadata["tensor_file_sha256"],
                "mean_validation_normalized_frobenius": metadata[
                    "mean_validation_normalized_frobenius"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
