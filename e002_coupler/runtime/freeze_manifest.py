"""Create the immutable pre-LOCKED E002 execution manifest exactly once."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from experiments.latentport.e001_handoff.runtime.constants import SOURCE, TARGET
from experiments.latentport.e002_coupler.correction.serialization import verify_correction
from experiments.latentport.e002_coupler.runtime.constants import ATTEMPT_ROOT, E002_ROOT
from experiments.latentport.e002_coupler.runtime.lock_guard import FROZEN_MANIFEST, sha256_file


STATIC_FILES = (
    "PROTOCOL.md",
    "PREREGISTRATION.json",
    "ADAPTIVE_RESEARCH_LEDGER.md",
    "reuse/e001_refs.json",
    "data/validation_factorial/manifest.jsonl",
    "data/validation_factorial/selection.json",
    "data/fit_correction/reference.json",
    "data/validation_correction/reference.json",
    "data/locked/manifest.jsonl",
    "data/locked/selection.json",
    "data/long/manifest.jsonl",
    "artifacts/attempt_001/implementation/e001_integrity.json",
    "artifacts/attempt_001/implementation/same_model_restore_smoke.json",
    "artifacts/attempt_001/implementation/locked_path_smoke.json",
    "artifacts/attempt_001/implementation/prelocked_test_results.json",
    "artifacts/attempt_001/implementation/prelocked_tests_final.xml",
    "artifacts/attempt_001/implementation/prelocked_tests_final_2.xml",
    "artifacts/attempt_001/factorial/base_state_selection.json",
    "artifacts/attempt_001/factorial/factorial_validation_statistics.json",
    "artifacts/attempt_001/factorial/raw_evidence.jsonl",
    "artifacts/attempt_001/fit/correction_data_inventory.json",
    "artifacts/attempt_001/fit/correction_selection.json",
    "artifacts/attempt_001/fit/correction.safetensors",
    "artifacts/attempt_001/fit/correction.json",
)


def _source_files() -> Iterable[Path]:
    for directory in ("runtime", "factorial", "correction", "analysis", "tests"):
        for path in sorted((E002_ROOT / directory).rglob("*.py")):
            if "__pycache__" not in path.parts:
                yield path


def frozen_paths() -> list[Path]:
    paths = [E002_ROOT / relative for relative in STATIC_FILES]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Cannot freeze E002; required evidence is absent: {missing}")
    paths.extend(_source_files())
    paths.extend(
        path
        for path in (ATTEMPT_ROOT / "fit" / "candidates").rglob("*")
        if path.is_file() and path.name in ("correction.json", "correction.safetensors")
    )
    unique = {path.resolve(): path for path in paths}
    return sorted(unique.values(), key=lambda path: path.relative_to(E002_ROOT).as_posix())


def build_manifest() -> dict:
    prereg = json.loads((E002_ROOT / "PREREGISTRATION.json").read_text(encoding="utf-8"))
    e001_refs = json.loads((E002_ROOT / "reuse" / "e001_refs.json").read_text(encoding="utf-8"))
    correction = verify_correction(
        ATTEMPT_ROOT / "fit" / "correction.safetensors",
        ATTEMPT_ROOT / "fit" / "correction.json",
    )
    base = json.loads(
        (ATTEMPT_ROOT / "factorial" / "base_state_selection.json").read_text(encoding="utf-8")
    )
    records = [
        {
            "path": path.relative_to(E002_ROOT).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in frozen_paths()
    ]
    return {
        "experiment_id": "LATENTPORT_E002_COUPLER",
        "attempt": "attempt_001",
        "freeze_state": "FROZEN_BEFORE_LOCKED",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_model": {"repository": SOURCE.repository, "revision": SOURCE.revision},
        "target_model": {"repository": TARGET.repository, "revision": TARGET.revision},
        "e001_reference_manifest_hash": e001_refs["frozen_manifest"]["sha256"],
        "base_state": base["selected_state"],
        "base_state_selection_hash": sha256_file(
            ATTEMPT_ROOT / "factorial" / "base_state_selection.json"
        ),
        "correction_hash": correction["correction_hash"],
        "correction_rank": correction["rank"],
        "identity_lambda": correction["identity_lambda"],
        "locked_manifest_sha256": sha256_file(E002_ROOT / "data" / "locked" / "manifest.jsonl"),
        "long_manifest_sha256": sha256_file(E002_ROOT / "data" / "long" / "manifest.jsonl"),
        "files": records,
    }


def write_manifest() -> dict:
    if FROZEN_MANIFEST.exists():
        raise FileExistsError(f"Refusing to overwrite frozen E002 evidence: {FROZEN_MANIFEST}")
    manifest = build_manifest()
    with FROZEN_MANIFEST.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return manifest


def main() -> None:
    print(json.dumps(write_manifest(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
