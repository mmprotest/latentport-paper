from __future__ import annotations

import json
from pathlib import Path

from experiments.latentport.e001_handoff.analysis.build_final_manifest import build_manifest
from experiments.latentport.e001_handoff.analysis.integrity import validate_raw_evidence_file
from experiments.latentport.e001_handoff.runtime.constants import ATTEMPT_ROOT, E001_ROOT
from experiments.latentport.e001_handoff.runtime.lock_guard import sha256_file, verify_frozen_manifest


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_actual_raw_evidence_is_complete():
    result = validate_raw_evidence_file(ATTEMPT_ROOT / "locked" / "raw_evidence.jsonl")
    assert result == {"pass": True, "documents": 64, "errors": []}


def test_final_result_required_fields_and_frozen_hashes():
    result = load(ATTEMPT_ROOT / "verdict" / "RESULT.json")
    required = {
        "e001_status", "canonical_verdict", "source_model", "source_revision", "target_model",
        "target_revision", "fit_documents", "validation_documents", "locked_documents",
        "same_model_restore_4b_pass", "same_model_restore_9b_pass", "native_9b_nll",
        "source_4b_nll", "empty_9b_nll", "kv_only_nll", "kv_gdn_direct_nll",
        "full_translated_nll", "full_shuffled_nll", "kv_only_delta_nll",
        "full_translated_delta_nll", "full_vs_kv_improvement_fraction",
        "full_vs_kv_bootstrap_ci", "tqr", "full_vs_shuffled_delta",
        "full_vs_shuffled_bootstrap_ci", "mean_js_to_native", "top1_agreement_to_native",
        "long_test_run", "long_delta_nll", "long_tqr", "long_pass", "native_9b_prefill_ms",
        "source_4b_prefill_ms", "translation_ms", "state_install_ms", "bridge_ms",
        "translator_parameter_count", "translator_bytes", "source_state_bytes",
        "target_state_bytes", "translator_hash", "frozen_manifest_hash", "next_hypothesis",
    }
    assert required <= set(result)
    assert result["canonical_verdict"] == "RECURRENT_STATE_TRANSLATABLE"
    assert result["raw_evidence_sha256"] == sha256_file(
        ATTEMPT_ROOT / "locked" / "raw_evidence.jsonl"
    )


def test_required_figure_inventory():
    expected = {
        "delta_nll_conditions.png", "kv_vs_full_translation.png",
        "target_quality_retention.png", "native_vs_handoff_position_curve.png",
        "shuffled_state_control.png", "gdn_translation_error_by_layer.png",
        "state_convergence_over_tokens.png", "kv_translation_error_by_layer.png",
        "timing_breakdown.png", "verdict_summary.png",
    }
    found = {path.name for path in (ATTEMPT_ROOT / "figures").glob("*.png")}
    assert found == expected
    assert all((ATTEMPT_ROOT / "figures" / name).stat().st_size > 20_000 for name in expected)


def test_report_has_all_required_central_sections():
    report = (ATTEMPT_ROOT / "REPORT.md").read_text(encoding="utf-8")
    for heading in (
        "# Executive Finding", "# Can the 9B Model Pick Up Where the 4B Model Left Off?",
        "# Does Recurrent Memory Add Anything Beyond KV Cache?",
        "# Is the Transferred State About the Actual Context?",
        "# Does the 9B Model Repair Translation Error After Handoff?",
        "# Is It Potentially Faster Than Re-Prefilling?",
    ):
        assert heading in report
    assert "RECURRENT_STATE_TRANSLATABLE" in report
    assert "E001 stops here" in report


def test_conditional_runs_obey_frozen_triggers():
    result = load(ATTEMPT_ROOT / "verdict" / "RESULT.json")
    assert result["oracle_diagnostics_unlocked"] is False
    assert result["oracle_diagnostics_run"] is False
    assert result["long_test_eligible"] is False
    assert result["long_test_run"] is False


def test_frozen_manifest_still_verifies_after_reporting():
    assert verify_frozen_manifest()["translator_hash"] == load(
        ATTEMPT_ROOT / "verdict" / "RESULT.json"
    )["translator_hash"]


def test_final_manifest_contract_or_content():
    path = ATTEMPT_ROOT / "FINAL_ARTIFACT_MANIFEST.json"
    manifest = load(path) if path.exists() else build_manifest()
    listed = {record["path"] for record in manifest["files"]}
    assert "REPORT.md" in listed
    assert "EXECUTIVE_SUMMARY.md" in listed
    assert "verdict/RESULT.json" in listed
    assert "locked/raw_evidence.jsonl" in listed
    assert manifest["canonical_verdict"] == "RECURRENT_STATE_TRANSLATABLE"
