"""E002 integrity gate over the sealed E001 inputs actually consumed."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import transformers
from datasets import __version__ as datasets_version
from huggingface_hub import __version__ as hub_version

from experiments.latentport.e001_handoff.runtime.constants import SOURCE, TARGET
from experiments.latentport.e001_handoff.runtime.modeling import load_tokenizer
from experiments.latentport.e002_coupler.runtime.constants import ATTEMPT_ROOT, E001_ROOT, E002_ROOT, ensure_attempt_layout


def sha256_file(path: Path, chunk_bytes: int = 32 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def _iter_reference_records(value: Any) -> Iterable[tuple[str, dict[str, Any]]]:
    if isinstance(value, dict):
        if set(value) >= {"path", "sha256"}:
            yield value["path"], value
        else:
            for child in value.values():
                yield from _iter_reference_records(child)


def _write_once(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _hash_model_files(model_revisions: dict[str, Any]) -> list[dict[str, Any]]:
    results = []
    for role, spec in (("source", SOURCE), ("target", TARGET)):
        record = model_revisions[role]
        if record["repository"] != spec.repository or record["revision"] != spec.revision:
            raise RuntimeError(f"{role} frozen model identity mismatch")
        tokenizer_path = spec.snapshot / "tokenizer.json"
        tokenizer_hash = sha256_file(tokenizer_path)
        if tokenizer_hash != record["tokenizer_json_sha256"]:
            raise RuntimeError(f"{role} tokenizer hash mismatch")
        results.append({"role": role, "name": "tokenizer.json", "sha256": tokenizer_hash})
        for weight in record["weight_files"]:
            path = spec.snapshot / weight["name"]
            print(f"hashing {role} {weight['name']}", flush=True)
            actual = sha256_file(path)
            if actual != weight["sha256"] or path.stat().st_size != weight["bytes"]:
                raise RuntimeError(f"{role} model shard mismatch: {weight['name']}")
            results.append(
                {
                    "role": role,
                    "name": weight["name"],
                    "bytes": path.stat().st_size,
                    "sha256": actual,
                }
            )
    return results


def _read_only_state_smoke() -> dict[str, Any]:
    results: dict[str, Any] = {}
    expectations = {
        "fit_recurrent": ((1024, 24, 32, 128, 128), np.dtype("float32")),
        "fit_convolution": ((1024, 24, 8192, 4), np.dtype("uint16")),
        "validation_recurrent": ((256, 24, 32, 128, 128), np.dtype("float32")),
        "validation_convolution": ((256, 24, 8192, 4), np.dtype("uint16")),
    }
    refs = json.loads((E002_ROOT / "reuse" / "e001_refs.json").read_text(encoding="utf-8"))
    for name, (shape, dtype) in expectations.items():
        split, component = name.split("_", 1)
        ref = refs["source_training_states"][f"{split}_{component}"]
        path = E001_ROOT / ref["path"]
        stat_before = path.stat()
        array = np.load(path, mmap_mode="r")
        if array.shape != shape or array.dtype != dtype or array.flags.writeable:
            raise RuntimeError(f"read-only state load contract failed for {name}")
        indices = (0, shape[0] - 1)
        digest = hashlib.sha256()
        for index in indices:
            sample = np.asarray(array[index]).reshape(-1)
            edge = np.concatenate((sample[:64], sample[-64:]))
            digest.update(memoryview(np.ascontiguousarray(edge)).cast("B"))
        del array
        stat_after = path.stat()
        if (stat_before.st_size, stat_before.st_mtime_ns) != (stat_after.st_size, stat_after.st_mtime_ns):
            raise RuntimeError(f"state file changed during read-only load: {name}")
        results[name] = {
            "shape": list(shape),
            "dtype": str(dtype),
            "writeable": False,
            "edge_sample_sha256": digest.hexdigest(),
            "file_unchanged": True,
        }
    return results


def run_integrity_gate() -> dict[str, Any]:
    ensure_attempt_layout()
    refs = json.loads((E002_ROOT / "reuse" / "e001_refs.json").read_text(encoding="utf-8"))
    verified_refs = []
    for relative, record in _iter_reference_records(refs):
        path = E001_ROOT / relative
        print(f"verifying E001 reference {relative}", flush=True)
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = sha256_file(path)
        if actual != record["sha256"]:
            raise RuntimeError(f"E001 reference hash mismatch: {relative}")
        verified_refs.append({"path": relative, "bytes": path.stat().st_size, "sha256": actual})

    result_record = json.loads((E001_ROOT / refs["result"]["path"]).read_text(encoding="utf-8"))
    if result_record["canonical_verdict"] != "RECURRENT_STATE_TRANSLATABLE":
        raise RuntimeError("E001 canonical verdict mismatch")
    model_revisions = json.loads(
        (E001_ROOT / refs["model_revisions"]["path"]).read_text(encoding="utf-8")
    )
    model_files = _hash_model_files(model_revisions)
    tokenizer = load_tokenizer()
    if len(tokenizer) != 248_077:
        raise RuntimeError("tokenizer vocabulary mismatch")
    runtime = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "transformers": transformers.__version__,
        "datasets": datasets_version,
        "huggingface_hub": hub_version,
        "gpu": torch.cuda.get_device_name(0),
        "bf16_supported": bool(torch.cuda.is_bf16_supported()),
    }
    expected_runtime = {
        "python": "3.11.9",
        "torch": "2.13.0+cu130",
        "torch_cuda": "13.0",
        "transformers": "5.16.1",
        "datasets": "5.0.1",
        "huggingface_hub": "1.29.0",
        "gpu": "NVIDIA GeForce RTX 5090",
        "bf16_supported": True,
    }
    if runtime != expected_runtime:
        raise RuntimeError(f"E001 runtime compatibility mismatch: {runtime}")
    schemas = {}
    for role in ("source", "target"):
        schema = json.loads(
            (E001_ROOT / refs[f"{role}_schema"]["path"]).read_text(encoding="utf-8")
        )
        if schema["language_layers"] != 32 or len(schema["fields"]) != 217:
            raise RuntimeError(f"{role} schema completeness mismatch")
        schemas[role] = {
            "sha256": refs[f"{role}_schema"]["sha256"],
            "fields": len(schema["fields"]),
            "layer_pattern": schema["layer_type_pattern"],
        }
    output = {
        "experiment_id": "LATENTPORT_E002_COUPLER",
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "e001_canonical_verdict": result_record["canonical_verdict"],
        "e001_locked_records_parsed": False,
        "verified_references": verified_refs,
        "model_files": model_files,
        "runtime": runtime,
        "schemas": schemas,
        "read_only_state_load": _read_only_state_smoke(),
        "combined_translator_hash": refs["combined_translator_hash"],
    }
    _write_once(ATTEMPT_ROOT / "implementation" / "e001_integrity.json", output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    result = run_integrity_gate()
    print(json.dumps({"status": result["status"], "verified": len(result["verified_references"])}))


if __name__ == "__main__":
    main()

