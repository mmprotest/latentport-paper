"""Build a complete final artifact inventory without rehashing known giant evidence arrays."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from experiments.latentport.e001_handoff.runtime.constants import ATTEMPT_ROOT, E001_ROOT
from experiments.latentport.e001_handoff.runtime.lock_guard import sha256_file


OUTPUT = ATTEMPT_ROOT / "FINAL_ARTIFACT_MANIFEST.json"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def known_hashes() -> dict[str, str]:
    known: dict[str, str] = {}
    for split in ("fit", "validation"):
        for role in ("source", "target"):
            path = ATTEMPT_ROOT / split / "paired_states" / f"{role}_collection.json"
            collection = json.loads(path.read_text(encoding="utf-8"))
            for record in collection["array_inventory"].values():
                known[record["path"]] = record["sha256"]
    for pass_name in ("source_pass.jsonl", "translation_pass.jsonl"):
        for record in read_jsonl(ATTEMPT_ROOT / "locked" / pass_name):
            known[record["state_file"]["path"]] = record["state_file"]["serialized_sha256"]
    for record in read_jsonl(ATTEMPT_ROOT / "locked" / "raw_evidence.jsonl"):
        known[record["native_state_file"]["path"]] = record["native_state_file"]["serialized_sha256"]
    return known


def build_manifest() -> dict[str, Any]:
    known = known_hashes()
    files = []
    for path in sorted(ATTEMPT_ROOT.rglob("*")):
        if not path.is_file() or path == OUTPUT or "__pycache__" in path.parts:
            continue
        relative = path.relative_to(ATTEMPT_ROOT).as_posix()
        digest = known.get(relative)
        hash_source = "recorded_at_creation" if digest is not None else "verified_final_scan"
        if digest is None:
            digest = sha256_file(path)
        files.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": digest,
                "hash_source": hash_source,
            }
        )
    root_files = []
    for name in (
        "PROTOCOL.md", "PREREGISTRATION.json", "FROZEN_MANIFEST.json",
        "ADAPTIVE_RESEARCH_LEDGER.md",
    ):
        path = E001_ROOT / name
        root_files.append({"path": name, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    result = json.loads((ATTEMPT_ROOT / "verdict" / "RESULT.json").read_text(encoding="utf-8"))
    return {
        "experiment_id": "LATENTPORT_E001_HANDOFF",
        "attempt": "attempt_001",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "canonical_verdict": result["canonical_verdict"],
        "e001_status": result["e001_status"],
        "artifact_files": len(files),
        "artifact_bytes": sum(record["bytes"] for record in files),
        "files": files,
        "experiment_root_files": root_files,
        "large_file_hash_policy": "Paired arrays and serialized LOCKED states use SHA-256 values computed and recorded when each immutable file was created; all other files were hashed in the final scan.",
    }


def main() -> None:
    manifest = build_manifest()
    with OUTPUT.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"files": manifest["artifact_files"], "bytes": manifest["artifact_bytes"]}, indent=2))


if __name__ == "__main__":
    main()
