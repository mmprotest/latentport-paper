"""One-shot access control and integrity checks for E002 LOCKED evidence."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from experiments.latentport.e002_coupler.runtime.constants import ATTEMPT_ROOT, E002_ROOT


FROZEN_MANIFEST = E002_ROOT / "FROZEN_MANIFEST.json"
LOCKED_START = ATTEMPT_ROOT / "locked" / "LOCKED_RUN_STARTED.json"


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def verify_frozen_manifest() -> dict[str, Any]:
    if not FROZEN_MANIFEST.is_file():
        raise PermissionError("E002 LOCKED access denied before FROZEN_MANIFEST.json exists")
    manifest = json.loads(FROZEN_MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("freeze_state") != "FROZEN_BEFORE_LOCKED":
        raise PermissionError("E002 frozen manifest does not authorize LOCKED execution")
    for record in manifest["files"]:
        path = E002_ROOT / record["path"]
        if not path.is_file() or path.stat().st_size != record["bytes"]:
            raise RuntimeError(f"Frozen file missing or resized: {record['path']}")
        if sha256_file(path) != record["sha256"]:
            raise RuntimeError(f"Frozen manifest mismatch: {record['path']}")
    return manifest


def load_locked_rows(*, authorization: bool = False) -> list[dict[str, Any]]:
    if not authorization:
        raise PermissionError("Explicit verified E002 LOCKED authorization is required")
    verify_frozen_manifest()
    path = E002_ROOT / "data" / "locked" / "manifest.jsonl"
    with path.open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if len(rows) != 64:
        raise RuntimeError(f"E002 LOCKED manifest count is {len(rows)}, expected 64")
    return rows


def begin_locked_run() -> dict[str, Any]:
    manifest = verify_frozen_manifest()
    LOCKED_START.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "frozen_manifest_sha256": sha256_file(FROZEN_MANIFEST),
        "correction_hash": manifest["correction_hash"],
        "base_state": manifest["base_state"],
        "locked_execution_ordinal": 1,
    }
    try:
        with LOCKED_START.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n")
    except FileExistsError as error:
        raise PermissionError("Canonical E002 LOCKED execution has already started; rerun forbidden") from error
    return record

