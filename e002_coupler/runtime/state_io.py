"""Write-once state and logits persistence helpers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from experiments.latentport.e001_handoff.state.cache_state import state_checksum, state_nbytes


def save_state_once(path: Path, state: dict[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    checksum = state_checksum(state)
    with path.open("xb") as handle:
        torch.save(state, handle)
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "state_tensor_bytes": state_nbytes(state),
        "state_checksum": checksum,
    }


def load_state(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return torch.load(handle, map_location="cpu", weights_only=False)


def save_logits_once(path: Path, logits: torch.Tensor) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    bf16 = logits.detach().to(torch.bfloat16).contiguous().cpu()
    array = bf16.view(torch.uint16).numpy()
    content_hash = hashlib.sha256(memoryview(np.ascontiguousarray(array)).cast("B")).hexdigest()
    with path.open("xb") as handle:
        np.save(handle, array, allow_pickle=False)
    return {
        "path": str(path),
        "shape": list(array.shape),
        "storage": "bfloat16 bit patterns as uint16 NPY",
        "bytes": path.stat().st_size,
        "content_sha256": content_hash,
    }


def load_logits(path: Path, *, device: str = "cpu") -> torch.Tensor:
    array = np.load(path, mmap_mode="r")
    if array.dtype != np.uint16:
        raise TypeError("stored logits are not uint16 bfloat16 bit patterns")
    tensor = torch.from_numpy(np.array(array, copy=True)).view(torch.bfloat16)
    return tensor.to(device)


def write_json_once(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")

