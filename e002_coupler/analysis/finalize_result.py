"""Freeze machine-readable final E002 result before narrative report generation."""

from __future__ import annotations

import json
from typing import Any

from experiments.latentport.e001_handoff.runtime.constants import SOURCE, TARGET
from experiments.latentport.e002_coupler.analysis.verdict import final_verdict, next_hypothesis
from experiments.latentport.e002_coupler.correction.model import parameter_inventory
from experiments.latentport.e002_coupler.runtime.constants import ATTEMPT_ROOT, E002_ROOT
from experiments.latentport.e002_coupler.runtime.lock_guard import sha256_file, verify_frozen_manifest
from experiments.latentport.e002_coupler.runtime.state_io import write_json_once


RESULT = ATTEMPT_ROOT / "verdict" / "RESULT.json"


def _json(path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_result() -> dict[str, Any]:
    if not (ATTEMPT_ROOT / "diagnostics" / "postverdict_ablations.json").is_file():
        raise RuntimeError("required post-verdict ablations are incomplete")
    metrics = _json(ATTEMPT_ROOT / "locked" / "locked_metrics.json")
    frozen_verdict = _json(ATTEMPT_ROOT / "verdict" / "final_verdict.json")
    verdict_4k = frozen_verdict["canonical_4k_verdict"]
    long_path = ATTEMPT_ROOT / "long_context" / "long_result.json"
    if verdict_4k == "NEAR_NATIVE_HANDOFF" and not long_path.is_file():
        raise RuntimeError("NEAR_NATIVE_HANDOFF requires the conditional frozen LONG run")
    if verdict_4k != "NEAR_NATIVE_HANDOFF" and long_path.is_file():
        raise RuntimeError("LONG evidence exists although the 4K verdict did not unlock it")
    long = _json(long_path) if long_path.is_file() else None
    verdict = final_verdict(verdict_4k, long_pass=long["long_pass"] if long else None)
    if verdict != frozen_verdict["canonical_verdict"]:
        raise RuntimeError("final verdict freeze no longer matches canonical inputs")
    correction = _json(ATTEMPT_ROOT / "fit" / "correction.json")
    selection = _json(ATTEMPT_ROOT / "fit" / "correction_selection.json")
    factorial = _json(ATTEMPT_ROOT / "factorial" / "base_state_selection.json")
    refs = _json(E002_ROOT / "reuse" / "e001_refs.json")
    frozen = verify_frozen_manifest()
    conditions = metrics["conditions"]
    repair = metrics["state_repair"]
    timing = metrics["timing"]
    result = {
        "experiment_id": "LATENTPORT_E002_COUPLER",
        "result_stage": "FINAL_VERDICT_FROZEN_BEFORE_NARRATIVE",
        "e002_status": "COMPLETE",
        "canonical_4k_verdict": verdict_4k,
        "canonical_verdict": verdict,
        "source_model": SOURCE.repository,
        "source_revision": SOURCE.revision,
        "target_model": TARGET.repository,
        "target_revision": TARGET.revision,
        "e001_reference_manifest_hash": refs["frozen_manifest"]["sha256"],
        "e001_kv_translator_hash": refs["translators"]["kv_tensors"]["sha256"],
        "e001_gdn_translator_hash": refs["translators"]["gdn_tensors"]["sha256"],
        "e001_conv_translator_hash": refs["translators"]["conv_tensors"]["sha256"],
        "factorial_documents": 32,
        "correction_fit_documents": 128,
        "correction_validation_documents": 32,
        "locked_documents": 64,
        "long_documents": 16 if long else 0,
        "factorial_validation_metrics": factorial["condition_metrics"],
        "factorial_locked_metrics": {name: conditions[name] for name in factorial["condition_metrics"]},
        "factorial_interaction_metrics": {
            "validation": factorial["interaction_metrics"],
            "locked": metrics["factorial_interactions"],
            "replication": metrics["coupling_replication"],
        },
        "base_state": metrics["base_state"],
        "correction_rank": correction["rank"],
        "correction_parameter_count": correction["parameter_count"],
        "correction_bytes": correction["tensor_bytes"],
        "correction_parameter_inventory": parameter_inventory(correction["rank"]),
        "correction_parameter_ratio_to_target": correction["parameter_count"] / TARGET.expected_parameter_count,
        "correction_bytes_ratio_to_target_bfloat16": correction["tensor_bytes"] / (TARGET.expected_parameter_count * 2),
        "identity_lambda": correction["identity_lambda"],
        "native_9b_nll": conditions["NATIVE_9B"]["nll"],
        "source_4b_nll": conditions["SOURCE_4B"]["nll"],
        "empty_9b_nll": conditions["EMPTY_9B"]["nll"],
        "base_state_nll": conditions["BASE_STATE"]["nll"],
        "joint_corrected_nll": conditions["JOINT_CORRECTED"]["nll"],
        "joint_shuffled_nll": conditions["JOINT_SHUFFLED"]["nll"],
        "base_delta_nll": metrics["base_delta_nll"],
        "corrected_delta_nll": metrics["corrected_delta_nll"],
        "remaining_gap_reduction": metrics["remaining_gap_reduction"],
        "remaining_gap_bootstrap_ci": metrics["remaining_gap_bootstrap_ci"],
        "native_context_recovery": metrics["native_context_recovery"],
        "tqr": metrics["tqr"],
        "corrected_vs_base_delta": -metrics["comparisons"]["base_minus_corrected"]["mean_difference"],
        "base_vs_corrected_improvement": metrics["comparisons"]["base_minus_corrected"],
        "corrected_vs_source_delta": metrics["comparisons"]["corrected_minus_source"]["mean_difference"],
        "corrected_vs_source_bootstrap_ci": metrics["comparisons"]["corrected_minus_source"]["bootstrap_ci"],
        "corrected_vs_shuffled_delta": metrics["comparisons"]["corrected_minus_shuffled"]["mean_difference"],
        "corrected_vs_shuffled_bootstrap_ci": metrics["comparisons"]["corrected_minus_shuffled"]["bootstrap_ci"],
        "top1_agreement_native": metrics["top1_agreement_native"],
        "js_divergence_native": metrics["js_divergence_native"],
        "correction_magnitude_summary": metrics["correction_magnitude"],
        "state_repair_token_1": repair["1"],
        "state_repair_token_4": repair["4"],
        "state_repair_token_16": repair["16"],
        "state_repair_token_64": repair["64"],
        "state_repair_token_256": repair["256"],
        "native_9b_prefill_ms": timing["native_9b_prefill_ms"],
        "source_4b_prefill_ms": timing["source_4b_prefill_ms"],
        "base_construction_ms": timing["base_construction_ms"],
        "joint_correction_ms": timing["joint_correction_ms"],
        "state_install_ms": timing["state_install_ms"],
        "bridge_ms": timing["bridge_ms"],
        "handoff_margin_ms": timing["handoff_margin_ms"],
        "long_test_run": long is not None,
        "long_delta_nll": long["long_delta_nll"] if long else None,
        "long_ncr": long["long_ncr"] if long else None,
        "long_source_advantage": long["long_source_advantage"] if long else None,
        "long_pass": long["long_pass"] if long else None,
        "correction_hash": correction["correction_hash"],
        "frozen_manifest_hash": sha256_file(E002_ROOT / "FROZEN_MANIFEST.json"),
        "locked_raw_evidence_hash": metrics["raw_evidence_sha256"],
        "selected_candidate": selection["selected"]["selected_candidate"],
        "next_hypothesis": frozen_verdict["next_hypothesis"],
    }
    return result


def main() -> None:
    result = build_result()
    write_json_once(RESULT, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
