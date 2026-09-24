"""Final E002 integrity audit over sealed prior, raw evidence, figures, and manifests."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from experiments.latentport.e002_coupler.analysis.audit import (
    verify_artifact_manifest,
    verify_raw_evidence,
)
from experiments.latentport.e002_coupler.correction.serialization import verify_correction
from experiments.latentport.e002_coupler.runtime.constants import ATTEMPT_ROOT, E001_ROOT, E002_ROOT
from experiments.latentport.e002_coupler.runtime.integrity import _iter_reference_records, sha256_file
from experiments.latentport.e002_coupler.runtime.lock_guard import verify_frozen_manifest
from experiments.latentport.e002_coupler.runtime.state_io import write_json_once


OUTPUT = ATTEMPT_ROOT / "implementation" / "final_integrity_audit.json"


def run() -> dict:
    refs = json.loads((E002_ROOT / "reuse" / "e001_refs.json").read_text(encoding="utf-8"))
    checked_refs = 0
    for relative, record in _iter_reference_records(refs):
        path = E001_ROOT / relative
        if not path.is_file() or sha256_file(path) != record["sha256"]:
            raise RuntimeError(f"sealed E001 reference changed: {relative}")
        checked_refs += 1
    frozen = verify_frozen_manifest()
    correction = verify_correction(
        ATTEMPT_ROOT / "fit" / "correction.safetensors",
        ATTEMPT_ROOT / "fit" / "correction.json",
    )
    raw = verify_raw_evidence(
        ATTEMPT_ROOT / "locked" / "raw_evidence.jsonl", artifact_root=ATTEMPT_ROOT
    )
    manifest = verify_artifact_manifest(
        ATTEMPT_ROOT / "FINAL_ARTIFACT_MANIFEST.json", E002_ROOT
    )
    result = json.loads((ATTEMPT_ROOT / "verdict" / "RESULT.json").read_text(encoding="utf-8"))
    if result["correction_hash"] != correction["correction_hash"]:
        raise RuntimeError("RESULT correction hash mismatch")
    if result["frozen_manifest_hash"] != sha256_file(E002_ROOT / "FROZEN_MANIFEST.json"):
        raise RuntimeError("RESULT frozen-manifest hash mismatch")
    required_figures = {
        "factorial_condition_nll.png",
        "factorial_interactions.png",
        "base_vs_corrected_delta_nll.png",
        "native_context_recovery.png",
        "source_vs_corrected.png",
        "corrected_vs_shuffled.png",
        "correction_magnitude_by_layer.png",
        "state_repair_to_256_tokens.png",
        "timing_breakdown.png",
        "verdict_summary.png",
    }
    if result["canonical_verdict"] in ("NEAR_NATIVE_HANDOFF", "LONG_CONTEXT_HANDOFF"):
        required_figures.add("headline_result.png")
    missing_figures = [name for name in required_figures if not (ATTEMPT_ROOT / "figures" / name).is_file()]
    if missing_figures:
        raise RuntimeError(f"required E002 figures are missing: {missing_figures}")
    if not (ATTEMPT_ROOT / "REPORT.md").is_file() or not (ATTEMPT_ROOT / "EXECUTIVE_SUMMARY.md").is_file():
        raise RuntimeError("final narrative artifacts are missing")
    long_exists = (ATTEMPT_ROOT / "long_context" / "long_result.json").is_file()
    if long_exists != bool(result["long_test_run"]):
        raise RuntimeError("LONG execution/result accounting mismatch")
    audit = {
        "experiment_id": "LATENTPORT_E002_COUPLER",
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "e001_reference_files_reverified": checked_refs,
        "e001_artifacts_unchanged": True,
        "frozen_e002_files_verified": len(frozen["files"]),
        "correction_hash": correction["correction_hash"],
        "raw_evidence": raw,
        "final_artifact_manifest": manifest,
        "required_figures": sorted(required_figures),
        "report_present": True,
        "executive_summary_present": True,
        "canonical_verdict": result["canonical_verdict"],
        "long_execution_consistent": True,
    }
    return audit


def main() -> None:
    audit = run()
    write_json_once(OUTPUT, audit)
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
