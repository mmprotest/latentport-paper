"""Create the immutable pre-LOCKED E001 execution manifest exactly once."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from experiments.latentport.e001_handoff.runtime.constants import E001_ROOT
from experiments.latentport.e001_handoff.runtime.lock_guard import FROZEN_MANIFEST, sha256_file


STATIC_FILES = (
    "PROTOCOL.md",
    "PREREGISTRATION.json",
    "runtime/requirements.lock",
    "runtime/environment.lock.txt",
    "data/MANIFEST_SUMMARY.json",
    "data/fit/manifest.jsonl",
    "data/validation/manifest.jsonl",
    "data/locked/manifest.jsonl",
    "data/long/manifest.jsonl",
    "artifacts/attempt_001/implementation/model_revisions.json",
    "artifacts/attempt_001/implementation/runtime_environment.json",
    "artifacts/attempt_001/implementation/architecture_correspondence.json",
    "artifacts/attempt_001/implementation/state_schema_source.json",
    "artifacts/attempt_001/implementation/state_schema_target.json",
    "artifacts/attempt_001/implementation/checkpoint_chunk_equivalence.json",
    "artifacts/attempt_001/implementation/learned_translation_install_smoke.json",
    "artifacts/attempt_001/state_validation/same_model_restore_4b.json",
    "artifacts/attempt_001/state_validation/same_model_restore_9b.json",
    "artifacts/attempt_001/diagnostics/direct_zero_training_smoke.json",
    "artifacts/attempt_001/fit/paired_states/source_collection.json",
    "artifacts/attempt_001/fit/paired_states/target_collection.json",
    "artifacts/attempt_001/validation/paired_states/source_collection.json",
    "artifacts/attempt_001/validation/paired_states/target_collection.json",
    "artifacts/attempt_001/implementation/prelocked_test_results.json",
)


def _source_files() -> Iterable[Path]:
    for directory in ("runtime", "state", "translators", "analysis", "tests"):
        for path in sorted((E001_ROOT / directory).rglob("*.py")):
            if "__pycache__" not in path.parts:
                yield path


def frozen_paths(*, require_all: bool = True) -> list[Path]:
    paths = [E001_ROOT / relative for relative in STATIC_FILES]
    paths.extend(_source_files())
    paths.extend(sorted((E001_ROOT / "translators" / "frozen").glob("*")))
    unique = {path.resolve(): path for path in paths if path.is_file()}
    expected = {str((E001_ROOT / relative).resolve()) for relative in STATIC_FILES}
    missing = sorted(expected - {str(path) for path in unique})
    if missing and require_all:
        raise FileNotFoundError(f"Cannot freeze; required evidence is absent: {missing}")
    return sorted(unique.values(), key=lambda path: path.relative_to(E001_ROOT).as_posix())


def build_manifest() -> dict:
    prereg = json.loads((E001_ROOT / "PREREGISTRATION.json").read_text(encoding="utf-8"))
    if prereg["registration_state"] != "FROZEN_BEFORE_LOCKED":
        raise RuntimeError("PREREGISTRATION must be marked FROZEN_BEFORE_LOCKED before manifest creation")
    if prereg["locked_evaluation_started"]:
        raise RuntimeError("Cannot create the pre-LOCKED manifest after LOCKED evaluation starts")
    records = []
    for path in frozen_paths(require_all=True):
        records.append(
            {
                "path": path.relative_to(E001_ROOT).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    translator_records = [
        record for record in records if record["path"].startswith("translators/frozen/")
    ]
    translator_digest = hashlib.sha256()
    for record in translator_records:
        translator_digest.update(f"{record['path']}:{record['sha256']}\n".encode("utf-8"))
    return {
        "experiment_id": "LATENTPORT_E001_HANDOFF",
        "attempt": "attempt_001",
        "freeze_state": "FROZEN_BEFORE_LOCKED",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_model": prereg["models"]["source"],
        "target_model": prereg["models"]["target"],
        "translator_hash": translator_digest.hexdigest(),
        "translator_files": translator_records,
        "locked_manifest_sha256": next(
            record["sha256"] for record in records if record["path"] == "data/locked/manifest.jsonl"
        ),
        "files": records,
    }


def write_manifest() -> dict:
    if FROZEN_MANIFEST.exists():
        raise FileExistsError(f"Refusing to overwrite frozen evidence: {FROZEN_MANIFEST}")
    manifest = build_manifest()
    with FROZEN_MANIFEST.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return manifest


def main() -> None:
    print(json.dumps(write_manifest(), indent=2))


if __name__ == "__main__":
    main()
