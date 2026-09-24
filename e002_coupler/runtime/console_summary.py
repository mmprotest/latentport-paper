"""Print the protocol-mandated final E002 console summary."""

from __future__ import annotations

import json

from experiments.latentport.e002_coupler.runtime.constants import ATTEMPT_ROOT, FACTORIAL_CONDITIONS


def main() -> None:
    result = json.loads((ATTEMPT_ROOT / "verdict" / "RESULT.json").read_text(encoding="utf-8"))
    factorial = result["factorial_validation_metrics"]
    improvement = result["base_vs_corrected_improvement"]
    magnitude = result["correction_magnitude_summary"]["all_layers"]
    repair = lambda token: result[f"state_repair_token_{token}"]["JOINT_CORRECTED"]
    lines = [
        "LATENTPORT E002 COMPLETE",
        "",
        f"STATUS: {result['e002_status']}",
        f"CANONICAL VERDICT: {result['canonical_verdict']}",
        "",
        f"SOURCE MODEL: {result['source_model']}@{result['source_revision']}",
        f"TARGET MODEL: {result['target_model']}@{result['target_revision']}",
        "",
        f"E001 REFERENCE VERIFIED: {result['e001_reference_manifest_hash']}",
        f"BASE STATE: {result['base_state']}",
        "",
        "FACTORIAL VALIDATION",
        *[f"{name}: NLL {factorial[name]['mean_nll']:.6f}; DeltaNLL {factorial[name]['mean_delta_nll']:.6f}" for name in FACTORIAL_CONDITIONS],
        "",
        "LOCKED NLL",
        f"NATIVE_9B: {result['native_9b_nll']:.6f}",
        f"SOURCE_4B: {result['source_4b_nll']:.6f}",
        f"EMPTY_9B: {result['empty_9b_nll']:.6f}",
        f"BASE_STATE: {result['base_state_nll']:.6f}",
        f"JOINT_CORRECTED: {result['joint_corrected_nll']:.6f}",
        f"JOINT_SHUFFLED: {result['joint_shuffled_nll']:.6f}",
        "",
        "PRIMARY CORRECTION",
        f"BASE DELTA NLL: {result['base_delta_nll']:.6f}",
        f"CORRECTED DELTA NLL: {result['corrected_delta_nll']:.6f}",
        f"REMAINING GAP REDUCTION: {result['remaining_gap_reduction']:.6f}",
        f"95% CI: [{improvement['bootstrap_ci'][0]:.6f}, {improvement['bootstrap_ci'][1]:.6f}] base minus corrected NLL",
        "",
        "NATIVE CONTEXT RECOVERY:",
        f"NCR: {result['native_context_recovery']:.6f}",
        "",
        f"TQR: {result['tqr'] if result['tqr'] is not None else 'undefined'}",
        "",
        "SOURCE SWITCH",
        f"CORRECTED VS SOURCE DELTA: {result['corrected_vs_source_delta']:.6f}",
        f"95% CI: [{result['corrected_vs_source_bootstrap_ci'][0]:.6f}, {result['corrected_vs_source_bootstrap_ci'][1]:.6f}]",
        "",
        "CONTENT SPECIFICITY",
        f"CORRECTED VS SHUFFLED: {result['corrected_vs_shuffled_delta']:.6f}",
        f"95% CI: [{result['corrected_vs_shuffled_bootstrap_ci'][0]:.6f}, {result['corrected_vs_shuffled_bootstrap_ci'][1]:.6f}]",
        "",
        "CORRECTION",
        f"RANK: {result['correction_rank']}",
        f"PARAMETERS: {result['correction_parameter_count']}",
        f"BYTES: {result['correction_bytes']}",
        f"MEDIAN RELATIVE MAGNITUDE: {magnitude['median']:.6f}",
        f"MAX RELATIVE MAGNITUDE: {magnitude['maximum']:.6f}",
        "",
        "STATE REPAIR",
        *[
            f"TOKEN {token}: recurrent error {repair(token)['recurrent_normalized_frobenius']:.6f}; cosine {repair(token)['recurrent_cosine']:.6f}"
            for token in (1, 4, 16, 64, 256)
        ],
        "",
        "TIMING",
        f"NATIVE 9B PREFILL: {result['native_9b_prefill_ms']:.3f} ms",
        f"SOURCE 4B PREFILL: {result['source_4b_prefill_ms']:.3f} ms",
        f"BASE CONSTRUCTION: {result['base_construction_ms']:.3f} ms",
        f"JOINT CORRECTION: {result['joint_correction_ms']:.3f} ms",
        f"STATE INSTALL: {result['state_install_ms']:.3f} ms",
        f"BRIDGE: {result['bridge_ms']:.3f} ms",
        f"HANDOFF MARGIN: {result['handoff_margin_ms']:.3f} ms",
        "",
        f"LONG TEST RUN: {result['long_test_run']}",
        f"LONG TEST PASS: {result['long_pass']}",
        "",
        f"NEXT HYPOTHESIS: {result['next_hypothesis']}",
    ]
    print("\n".join(lines))


if __name__ == "__main__":
    main()
