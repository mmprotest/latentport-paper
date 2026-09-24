"""Fit and validate canonical E001 KV and GDN-convolution ridge maps."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import save_file

from experiments.latentport.e001_handoff.runtime.constants import ATTEMPT_ROOT, E001_ROOT, load_preregistration
from experiments.latentport.e001_handoff.runtime.modeling import seed_everything
from experiments.latentport.e001_handoff.translators.ridge import fit_batched_ridge
from experiments.latentport.e001_handoff.translators.rope import derotate_cached_key


TRANSLATOR_ROOT = E001_ROOT / "translators" / "frozen"


def _bf16_array_to_float(array: np.ndarray, device: str = "cuda:0") -> torch.Tensor:
    copied = np.array(array, dtype=np.uint16, copy=True, order="C")
    return torch.from_numpy(copied).view(torch.bfloat16).to(device=device).float()


def _np(split: str, role: str, component: str):
    path = ATTEMPT_ROOT / split / "paired_states" / f"{role}_{component}.npy"
    return np.load(path, mmap_mode="r")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _pack_ridge(prefix: str, results: list, destination: dict[str, torch.Tensor]) -> None:
    fields = (
        "weight",
        "source_mean",
        "target_mean",
        "source_scale",
        "target_scale",
        "lambda_relative",
        "normalization_code",
        "validation_error",
        "condition_number",
    )
    for field in fields:
        destination[f"{prefix}.{field}"] = torch.stack(
            [getattr(result, field).detach().cpu() for result in results], dim=0
        )


def fit_kv(prereg: dict) -> dict[str, Any]:
    fit_positions = _np("fit", "source", "kv_positions").astype(np.int64)
    validation_positions = _np("validation", "source", "kv_positions").astype(np.int64)
    if not np.array_equal(fit_positions, _np("fit", "target", "kv_positions")):
        raise RuntimeError("FIT source/target KV position mismatch")
    if not np.array_equal(validation_positions, _np("validation", "target", "kv_positions")):
        raise RuntimeError("VALIDATION source/target KV position mismatch")

    tensors: dict[str, torch.Tensor] = {}
    metadata: dict[str, Any] = {
        "component": "full_attention_kv",
        "fit_samples_per_map": int(fit_positions.size),
        "validation_samples_per_map": int(validation_positions.size),
        "layer_head_role_maps": [],
        "cached_k_policy": "invert exact BF16-rounded source RoPE, fit in de-rotated space, apply exact target BF16-rounded RoPE",
    }
    for role_name, file_component in (
        ("K", "kv_keys_bf16_u16"),
        ("V", "kv_values_bf16_u16"),
    ):
        fit_source = _bf16_array_to_float(_np("fit", "source", file_component))
        fit_target = _bf16_array_to_float(_np("fit", "target", file_component))
        validation_source = _bf16_array_to_float(_np("validation", "source", file_component))
        validation_target = _bf16_array_to_float(_np("validation", "target", file_component))
        # [document, sample, layer, head, dim] -> per-layer [head, sample, dim]
        role_results = []
        for layer_position in range(8):
            x = fit_source[:, :, layer_position].permute(2, 0, 1, 3).reshape(4, -1, 256)
            y = fit_target[:, :, layer_position].permute(2, 0, 1, 3).reshape(4, -1, 256)
            xv = validation_source[:, :, layer_position].permute(2, 0, 1, 3).reshape(4, -1, 256)
            yv = validation_target[:, :, layer_position].permute(2, 0, 1, 3).reshape(4, -1, 256)
            if role_name == "K":
                fit_pos = torch.from_numpy(np.array(fit_positions, copy=True)).to("cuda:0").flatten()
                val_pos = torch.from_numpy(np.array(validation_positions, copy=True)).to("cuda:0").flatten()
                x = torch.stack([derotate_cached_key(head, fit_pos) for head in x], dim=0)
                y = torch.stack([derotate_cached_key(head, fit_pos) for head in y], dim=0)
                xv = torch.stack([derotate_cached_key(head, val_pos) for head in xv], dim=0)
                yv = torch.stack([derotate_cached_key(head, val_pos) for head in yv], dim=0)
            result = fit_batched_ridge(
                x,
                y,
                xv,
                yv,
                lambda_grid=prereg["kv_translator"]["ridge_relative_grid"],
                normalization_names=prereg["kv_translator"]["normalization_candidates"],
            )
            role_results.append(result)
            for head in range(4):
                metadata["layer_head_role_maps"].append(
                    {
                        "language_layer": 3 + 4 * layer_position,
                        "kv_head": head,
                        "role": role_name,
                        "lambda_relative": float(result.lambda_relative[head].item()),
                        "normalization": prereg["kv_translator"]["normalization_candidates"][
                            int(result.normalization_code[head].item())
                        ],
                        "validation_normalized_l2": float(result.validation_error[head].item()),
                        "condition_number": float(result.condition_number[head].item()),
                    }
                )
            print(
                f"KV {role_name} layer {3 + 4 * layer_position}: "
                f"validation={result.validation_error.mean().item():.6f}",
                flush=True,
            )
        _pack_ridge(f"{role_name.lower()}", role_results, tensors)
        del fit_source, fit_target, validation_source, validation_target
        torch.cuda.empty_cache()

    TRANSLATOR_ROOT.mkdir(parents=True, exist_ok=True)
    output = TRANSLATOR_ROOT / "kv_translator.safetensors"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite translator: {output}")
    save_file(tensors, output, metadata={"format": "latentport-e001-kv-ridge-v1"})
    metadata.update(
        {
            "tensor_file": str(output.relative_to(E001_ROOT)).replace("\\", "/"),
            "tensor_file_sha256": _sha256(output),
            "parameter_count_weights": sum(
                tensor.numel() for key, tensor in tensors.items() if key.endswith(".weight")
            ),
            "stored_numeric_values": sum(tensor.numel() for tensor in tensors.values()),
            "tensor_bytes": output.stat().st_size,
        }
    )
    metadata_path = TRANSLATOR_ROOT / "kv_translator.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metadata


def _conv_groups(tensor: torch.Tensor) -> torch.Tensor:
    # Runtime packing is Q[16x128], K[16x128], V[32x128]. Each of the four
    # convolution lags is an independent ridge sample; no Q/K/V mixing occurs.
    q, k, v = tensor.split((2048, 2048, 4096), dim=1)
    groups = [
        q.reshape(q.shape[0], 16, 128, 4),
        k.reshape(k.shape[0], 16, 128, 4),
        v.reshape(v.shape[0], 32, 128, 4),
    ]
    return torch.cat(groups, dim=1).permute(1, 0, 3, 2).reshape(64, -1, 128)


def fit_convolution(prereg: dict) -> dict[str, Any]:
    fit_source_map = _np("fit", "source", "convolution_bf16_u16")
    fit_target_map = _np("fit", "target", "convolution_bf16_u16")
    validation_source_map = _np("validation", "source", "convolution_bf16_u16")
    validation_target_map = _np("validation", "target", "convolution_bf16_u16")
    tensors: dict[str, torch.Tensor] = {}
    results = []
    maps = []
    group_roles = [*("Q" for _ in range(16)), *("K" for _ in range(16)), *("V" for _ in range(32))]
    group_heads = [*range(16), *range(16), *range(32)]
    language_layers = [index for index in range(32) if index % 4 != 3]
    for layer_position, language_layer in enumerate(language_layers):
        x = _conv_groups(_bf16_array_to_float(fit_source_map[:, layer_position]))
        y = _conv_groups(_bf16_array_to_float(fit_target_map[:, layer_position]))
        xv = _conv_groups(_bf16_array_to_float(validation_source_map[:, layer_position]))
        yv = _conv_groups(_bf16_array_to_float(validation_target_map[:, layer_position]))
        result = fit_batched_ridge(
            x,
            y,
            xv,
            yv,
            lambda_grid=prereg["gdn_convolution_translator"]["ridge_relative_grid"],
            normalization_names=prereg["gdn_convolution_translator"]["normalization_candidates"],
        )
        results.append(result)
        for group in range(64):
            maps.append(
                {
                    "language_layer": language_layer,
                    "packed_component": group_roles[group],
                    "component_head": group_heads[group],
                    "lambda_relative": float(result.lambda_relative[group].item()),
                    "normalization": prereg["gdn_convolution_translator"]["normalization_candidates"][
                        int(result.normalization_code[group].item())
                    ],
                    "validation_normalized_l2": float(result.validation_error[group].item()),
                    "condition_number": float(result.condition_number[group].item()),
                }
            )
        print(
            f"GDN convolution layer {language_layer}: validation={result.validation_error.mean().item():.6f}",
            flush=True,
        )
        del x, y, xv, yv
    _pack_ridge("conv", results, tensors)
    TRANSLATOR_ROOT.mkdir(parents=True, exist_ok=True)
    output = TRANSLATOR_ROOT / "gdn_convolution_translator.safetensors"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite translator: {output}")
    save_file(tensors, output, metadata={"format": "latentport-e001-gdn-conv-ridge-v1"})
    metadata = {
        "component": "gdn_convolution_state",
        "granularity": "per language layer, packed Q/K/V component, and corresponding 128-channel head; four lags are independent samples",
        "qkv_mixing": False,
        "fit_samples_per_map": 1024 * 4,
        "validation_samples_per_map": 256 * 4,
        "maps": maps,
        "tensor_file": str(output.relative_to(E001_ROOT)).replace("\\", "/"),
        "tensor_file_sha256": _sha256(output),
        "parameter_count_weights": sum(
            tensor.numel() for key, tensor in tensors.items() if key.endswith(".weight")
        ),
        "stored_numeric_values": sum(tensor.numel() for tensor in tensors.values()),
        "tensor_bytes": output.stat().st_size,
    }
    metadata_path = TRANSLATOR_ROOT / "gdn_convolution_translator.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--component", choices=("kv", "convolution"), required=True)
    args = parser.parse_args()
    prereg = load_preregistration()
    seed_everything(prereg["seeds"]["global"])
    result = fit_kv(prereg) if args.component == "kv" else fit_convolution(prereg)
    print(
        json.dumps(
            {
                "component": result["component"],
                "parameter_count_weights": result["parameter_count_weights"],
                "tensor_bytes": result["tensor_bytes"],
                "tensor_file_sha256": result["tensor_file_sha256"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
