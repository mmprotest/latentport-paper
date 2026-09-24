"""Create the write-once final E002 artifact inventory after all reports exist."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from experiments.latentport.e002_coupler.runtime.constants import ATTEMPT_ROOT, E002_ROOT
from experiments.latentport.e002_coupler.runtime.lock_guard import sha256_file


OUTPUT = ATTEMPT_ROOT / "FINAL_ARTIFACT_MANIFEST.json"


def _paths() -> list[Path]:
    paths = []
    for relative in (
        "PROTOCOL.md",
        "PREREGISTRATION.json",
        "ADAPTIVE_RESEARCH_LEDGER.md",
        "FROZEN_MANIFEST.json",
        "reuse/e001_refs.json",
    ):
        paths.append(E002_ROOT / relative)
    paths.extend(path for path in (E002_ROOT / "data").rglob("*") if path.is_file())
    paths.extend(path for path in ATTEMPT_ROOT.rglob("*") if path.is_file())
    unique = {
        path.resolve(): path
        for path in paths
        if path.is_file()
        and path.resolve() != OUTPUT.resolve()
        and "__pycache__" not in path.parts
        and path.name != ".gitkeep"
        and path.name != "final_integrity_audit.json"
    }
    return sorted(unique.values(), key=lambda path: path.relative_to(E002_ROOT).as_posix())


def build_manifest() -> dict:
    records = []
    paths = _paths()
    for index, path in enumerate(paths):
        records.append(
            {
                "path": path.relative_to(E002_ROOT).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
        if (index + 1) % 100 == 0:
            print(f"final manifest hashed {index + 1}/{len(paths)} files", flush=True)
    result = json.loads((ATTEMPT_ROOT / "verdict" / "RESULT.json").read_text(encoding="utf-8"))
    return {
        "experiment_id": "LATENTPORT_E002_COUPLER",
        "attempt": "attempt_001",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "canonical_verdict": result["canonical_verdict"],
        "correction_hash": result["correction_hash"],
        "frozen_manifest_hash": result["frozen_manifest_hash"],
        "files": records,
        "total_files": len(records),
        "total_bytes": sum(record["bytes"] for record in records),
    }


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError("refusing to overwrite the final E002 artifact manifest")
    manifest = build_manifest()
    with OUTPUT.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"files": manifest["total_files"], "bytes": manifest["total_bytes"]}, indent=2))


if __name__ == "__main__":
    main()

