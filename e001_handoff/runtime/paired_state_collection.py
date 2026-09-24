"""Collect exact paired Qwen3.5 persistent states on FIT/VALIDATION."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import DynamicCache

from experiments.latentport.e001_handoff.runtime.constants import (
    ATTEMPT_ROOT,
    E001_ROOT,
    MODEL_SPECS,
    load_preregistration,
)
from experiments.latentport.e001_handoff.runtime.modeling import (
    load_language_model,
    load_text_config,
    seed_everything,
)
from experiments.latentport.e001_handoff.runtime.scoring import position_ids


def read_manifest(split: str) -> list[dict[str, Any]]:
    path = E001_ROOT / "data" / split / "manifest.jsonl"
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def kv_positions(row: dict[str, Any], *, count: int, length: int, seed: int) -> np.ndarray:
    required = {0, length - 1, *range(127, length, 128)}
    local_seed = seed ^ int(row["selection_sha256"][:16], 16)
    generator = np.random.default_rng(local_seed)
    available = np.array(sorted(set(range(length)) - required), dtype=np.int64)
    fill = generator.choice(available, size=count - len(required), replace=False)
    return np.array(sorted(required | set(int(value) for value in fill)), dtype=np.int32)


def _tensor_u16(tensor: torch.Tensor) -> np.ndarray:
    if tensor.dtype != torch.bfloat16:
        raise TypeError(f"Expected BF16 tensor, got {tensor.dtype}")
    return tensor.detach().contiguous().cpu().view(torch.uint16).numpy()


def _tensor_f32(tensor: torch.Tensor) -> np.ndarray:
    if tensor.dtype != torch.float32:
        raise TypeError(f"Expected FP32 tensor, got {tensor.dtype}")
    return tensor.detach().contiguous().cpu().numpy()


def _sha256_arrays(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        digest.update(str(array.shape).encode("ascii"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(memoryview(np.ascontiguousarray(array)).cast("B"))
    return digest.hexdigest()


def _create_array(path: Path, *, dtype, shape: tuple[int, ...]):
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite paired-state evidence: {path}")
    return np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)


def collect(role: str, split: str) -> dict[str, Any]:
    prereg = load_preregistration()
    seed_everything(prereg["seeds"]["global"])
    if split not in ("fit", "validation"):
        raise ValueError("Paired state collection is restricted to FIT/VALIDATION")
    rows = read_manifest(split)
    expected_documents = prereg["splits"][split]["documents"]
    if len(rows) != expected_documents:
        raise RuntimeError(f"{split} manifest has {len(rows)} rows, expected {expected_documents}")
    spec = MODEL_SPECS[role]
    config = load_text_config(spec)
    layer_types = list(config.layer_types)
    gdn_layers = [index for index, value in enumerate(layer_types) if value == "linear_attention"]
    attention_layers = [index for index, value in enumerate(layer_types) if value == "full_attention"]
    prefix_length = prereg["splits"][split]["prefix_tokens"]
    stride = prereg["splits"][split]["checkpoint_stride"]
    checkpoints_per_document = prereg["splits"][split]["checkpoints_per_document"]
    checkpoint_count = len(rows) * checkpoints_per_document
    kv_samples = prereg["kv_translator"]["position_samples_per_fit_document"]

    output_dir = ATTEMPT_ROOT / split / "paired_states"
    output_dir.mkdir(parents=True, exist_ok=True)
    completion_path = output_dir / f"{spec.role}_collection.json"
    if completion_path.exists():
        raise FileExistsError(f"Refusing to overwrite completed evidence: {completion_path}")
    paths = {
        "recurrent": output_dir / f"{spec.role}_recurrent.npy",
        "convolution": output_dir / f"{spec.role}_convolution_bf16_u16.npy",
        "kv_keys": output_dir / f"{spec.role}_kv_keys_bf16_u16.npy",
        "kv_values": output_dir / f"{spec.role}_kv_values_bf16_u16.npy",
        "kv_positions": output_dir / f"{spec.role}_kv_positions.npy",
    }
    recurrent = _create_array(
        paths["recurrent"],
        dtype=np.float32,
        shape=(checkpoint_count, len(gdn_layers), 32, 128, 128),
    )
    convolution = _create_array(
        paths["convolution"],
        dtype=np.uint16,
        shape=(checkpoint_count, len(gdn_layers), 8192, 4),
    )
    keys = _create_array(
        paths["kv_keys"],
        dtype=np.uint16,
        shape=(len(rows), kv_samples, len(attention_layers), 4, 256),
    )
    values = _create_array(
        paths["kv_values"],
        dtype=np.uint16,
        shape=(len(rows), kv_samples, len(attention_layers), 4, 256),
    )
    positions = _create_array(
        paths["kv_positions"],
        dtype=np.int32,
        shape=(len(rows), kv_samples),
    )

    model = load_language_model(spec)
    device = next(model.parameters()).device
    checkpoint_records: list[dict[str, Any]] = []
    document_records: list[dict[str, Any]] = []
    started = time.perf_counter()
    for document_index, row in enumerate(rows):
        token_ids = row["token_ids"][:prefix_length]
        if len(token_ids) != prefix_length:
            raise RuntimeError(f"{split} document {document_index} is short")
        inputs = torch.tensor(token_ids, dtype=torch.long, device=device).unsqueeze(0)
        document_start = time.perf_counter()
        for checkpoint_index, end in enumerate(range(stride, prefix_length + 1, stride)):
            # Each checkpoint is a fresh one-shot native prefill. The pinned
            # runtime is not numerically chunk-invariant, so incrementally
            # extending one cache would train on an implementation artifact
            # not present in canonical one-shot prefix processing.
            cache = DynamicCache(config=config)
            with torch.inference_mode():
                model(
                    input_ids=inputs[:, :end],
                    position_ids=position_ids(0, end, device),
                    past_key_values=cache,
                    use_cache=True,
                    logits_to_keep=1,
                )
            if cache.get_seq_length() != end:
                raise RuntimeError(f"Cache length {cache.get_seq_length()} != checkpoint {end}")
            recurrent_cpu = np.stack(
                [_tensor_f32(cache.layers[layer].recurrent_states[0][0]) for layer in gdn_layers], axis=0
            )
            convolution_cpu = np.stack(
                [_tensor_u16(cache.layers[layer].conv_states[0][0]) for layer in gdn_layers], axis=0
            )
            flat_index = document_index * checkpoints_per_document + checkpoint_index
            recurrent[flat_index] = recurrent_cpu
            convolution[flat_index] = convolution_cpu
            checkpoint_records.append(
                {
                    "flat_checkpoint_index": flat_index,
                    "document_index": document_index,
                    "document_id": row["document_id"],
                    "prefix_tokens": end,
                    "gdn_state_sha256": _sha256_arrays(recurrent_cpu, convolution_cpu),
                }
            )
        sampled_positions = kv_positions(
            row,
            count=kv_samples,
            length=prefix_length,
            seed=prereg["seeds"]["kv_position_sampling"],
        )
        positions[document_index] = sampled_positions
        position_tensor = torch.tensor(sampled_positions, dtype=torch.long, device=device)
        key_tensor = torch.stack(
            [cache.layers[layer].keys[0, :, position_tensor, :] for layer in attention_layers], dim=0
        ).permute(2, 0, 1, 3)
        value_tensor = torch.stack(
            [cache.layers[layer].values[0, :, position_tensor, :] for layer in attention_layers], dim=0
        ).permute(2, 0, 1, 3)
        key_cpu = _tensor_u16(key_tensor)
        value_cpu = _tensor_u16(value_tensor)
        keys[document_index] = key_cpu
        values[document_index] = value_cpu
        document_records.append(
            {
                "document_index": document_index,
                "document_id": row["document_id"],
                "selection_sha256": row["selection_sha256"],
                "token_ids_sha256": hashlib.sha256(
                    ",".join(str(value) for value in token_ids).encode("ascii")
                ).hexdigest(),
                "kv_sample_positions": sampled_positions.tolist(),
                "sampled_kv_sha256": _sha256_arrays(key_cpu, value_cpu),
                "collection_wall_ms": (time.perf_counter() - document_start) * 1000.0,
            }
        )
        recurrent.flush()
        convolution.flush()
        keys.flush()
        values.flush()
        positions.flush()
        print(
            f"{spec.role} {split} {document_index + 1}/{len(rows)} "
            f"elapsed={(time.perf_counter() - started):.1f}s",
            flush=True,
        )

    del model
    torch.cuda.empty_cache()
    array_inventory = {}
    for name, path in paths.items():
        array_inventory[name] = {
            "path": str(path.relative_to(ATTEMPT_ROOT)).replace("\\", "/"),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    result = {
        "collection_version": "latentport-e001-paired-state-v1",
        "role": spec.role,
        "model_repository": spec.repository,
        "model_revision": spec.revision,
        "split": split.upper(),
        "documents": len(rows),
        "prefix_tokens": prefix_length,
        "checkpoint_stride": stride,
        "checkpoint_capture_mode": prereg["splits"][split]["checkpoint_capture_mode"],
        "checkpoints": checkpoint_count,
        "gdn_layers": gdn_layers,
        "attention_layers": attention_layers,
        "recurrent_dtype": "float32",
        "convolution_storage": "exact bfloat16 bit patterns stored as uint16",
        "kv_storage": "exact bfloat16 bit patterns stored as uint16",
        "array_inventory": array_inventory,
        "checkpoint_records": checkpoint_records,
        "document_records": document_records,
        "wall_seconds": time.perf_counter() - started,
    }
    completion_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key not in ("checkpoint_records", "document_records")}, indent=2))
    return result


def sha256_file(path: Path, chunk_size: int = 32 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", choices=("source", "target"), required=True)
    parser.add_argument("--split", choices=("fit", "validation"), required=True)
    args = parser.parse_args()
    collect(args.role, args.split)


if __name__ == "__main__":
    main()
