"""Raw-evidence and final-artifact integrity validators used before and after LOCKED."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from experiments.latentport.e002_coupler.runtime.constants import (
    PRIMARY_LOCKED_CONDITIONS,
    STATE_REPAIR_CHECKPOINTS,
)
from experiments.latentport.e002_coupler.runtime.lock_guard import sha256_file


def validate_raw_record(record: dict[str, Any]) -> None:
    required = {
        "document_id",
        "corpus",
        "split",
        "prefix_token_ids",
        "bridge_token_id",
        "continuation_token_ids",
        "conditions",
        "derived",
        "state_checksums",
        "correction_hash",
        "base_state",
        "correction_magnitude",
        "state_repair",
        "timing",
    }
    missing = required - set(record)
    if missing:
        raise ValueError(f"raw evidence fields missing: {sorted(missing)}")
    if len(record["prefix_token_ids"]) != 4096 or len(record["continuation_token_ids"]) != 64:
        raise ValueError("raw evidence token horizon is incomplete")
    if set(record["conditions"]) != set(PRIMARY_LOCKED_CONDITIONS):
        raise ValueError("raw evidence does not contain exactly the canonical conditions")
    for condition, value in record["conditions"].items():
        if len(value["per_token_nll"]) != 64 or len(value["per_token_entropy"]) != 64:
            raise ValueError(f"per-token evidence is incomplete for {condition}")
        if value["logits"]["shape"][0] != 64:
            raise ValueError(f"raw logits horizon is incomplete for {condition}")
    if set(record["state_repair"]) != {"BASE_STATE", "JOINT_CORRECTED"}:
        raise ValueError("state repair conditions are incomplete")
    expected = {str(value) for value in STATE_REPAIR_CHECKPOINTS}
    for trajectory in record["state_repair"].values():
        if set(trajectory) != expected:
            raise ValueError("state repair checkpoints are incomplete")


def verify_raw_evidence(
    path: Path, *, documents: int = 64, artifact_root: Path | None = None
) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if len(rows) != documents:
        raise ValueError(f"raw evidence contains {len(rows)} documents, expected {documents}")
    ids = []
    for record in rows:
        validate_raw_record(record)
        ids.append(record["document_id"])
        if artifact_root is not None:
            for condition, value in record["conditions"].items():
                logits = artifact_root / value["logits"]["path"]
                if not logits.is_file() or logits.stat().st_size != value["logits"]["bytes"]:
                    raise ValueError(f"raw logit file is missing or resized for {condition}")
    if len(ids) != len(set(ids)):
        raise ValueError("raw evidence document IDs are not unique")
    return {
        "documents": len(rows),
        "sha256": sha256_file(path),
        "unique_document_ids": True,
        "logit_files_present_and_sized": artifact_root is not None,
    }


def verify_artifact_manifest(path: Path, root: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    records = manifest.get("files", [])
    if not records:
        raise ValueError("final artifact manifest contains no files")
    seen = set()
    for record in records:
        relative = record["path"]
        if relative in seen:
            raise ValueError(f"duplicate final-manifest path: {relative}")
        seen.add(relative)
        candidate = root / relative
        if not candidate.is_file() or candidate.stat().st_size != record["bytes"]:
            raise ValueError(f"final-manifest file missing or resized: {relative}")
        if sha256_file(candidate) != record["sha256"]:
            raise ValueError(f"final-manifest hash mismatch: {relative}")
    return {"files": len(records), "verified": True}
