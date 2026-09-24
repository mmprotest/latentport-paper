"""Immutable correction serialization and hashing."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

from experiments.latentport.e002_coupler.correction.model import IdentityAnchoredCoupler


def sha256_file(path: Path, chunk_bytes: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def save_correction(
    model: IdentityAnchoredCoupler,
    tensor_path: Path,
    metadata_path: Path,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    if tensor_path.exists() or metadata_path.exists():
        raise FileExistsError("refusing to overwrite a correction artifact")
    tensor_path.parent.mkdir(parents=True, exist_ok=True)
    tensors = {name: tensor.detach().contiguous().cpu() for name, tensor in model.state_dict().items()}
    save_file(tensors, tensor_path)
    record = dict(metadata)
    record.update(
        {
            "format": "latentport-e002-identity-coupler-v1",
            "rank": model.rank,
            "basis_seed": model.basis_seed,
            "parameter_count": model.parameter_count,
            "tensor_file": tensor_path.name,
            "tensor_bytes": tensor_path.stat().st_size,
            "tensor_sha256": sha256_file(tensor_path),
        }
    )
    digest = hashlib.sha256()
    digest.update(_canonical(record))
    with tensor_path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    record["correction_hash"] = digest.hexdigest()
    with metadata_path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return record


def verify_correction(tensor_path: Path, metadata_path: Path) -> dict[str, Any]:
    record = json.loads(metadata_path.read_text(encoding="utf-8"))
    if sha256_file(tensor_path) != record["tensor_sha256"]:
        raise RuntimeError("correction tensor hash mismatch")
    expected = record.pop("correction_hash")
    digest = hashlib.sha256()
    digest.update(_canonical(record))
    with tensor_path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    if digest.hexdigest() != expected:
        raise RuntimeError("combined correction hash mismatch")
    record["correction_hash"] = expected
    return record


def load_correction(tensor_path: Path, metadata_path: Path, *, device: str = "cpu") -> IdentityAnchoredCoupler:
    record = verify_correction(tensor_path, metadata_path)
    model = IdentityAnchoredCoupler(record["rank"], basis_seed=record["basis_seed"])
    model.load_state_dict(load_file(tensor_path, device=device), strict=True)
    model.to(device)
    return model

