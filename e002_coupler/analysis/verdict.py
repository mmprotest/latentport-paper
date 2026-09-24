"""Pure preregistered E002 verdict ladder and exact H3 mapping."""

from __future__ import annotations

from typing import Any


NEXT_HYPOTHESES = {
    "LONG_CONTEXT_HANDOFF": (
        "H3: The corrected Qwen3.5 state interface remains valid across additional model sizes, "
        "enabling a multi-size state compatibility graph rather than a single 4B→9B edge."
    ),
    "NEAR_NATIVE_HANDOFF": (
        "H3: The near-native handoff can be fused into a GPU-local state-transfer kernel whose "
        "end-to-end latency is lower than native 9B prefill while preserving E002 fidelity."
    ),
    "FULL_STATE_HANDOFF": (
        "H3: The remaining fidelity gap is dominated by a small subset of layers/components "
        "identified by E002 post-verdict ablations, and correcting only those components can "
        "preserve handoff quality at lower translation cost."
    ),
    "JOINT_CORRECTION_WORKS": (
        "H3: The residual target mismatch is concentrated in specific depth regions, and a sparse "
        "layer-selective correction can close the remaining gap more efficiently than global "
        "low-rank correction."
    ),
    "IDENTITY_STATE_CONFIRMED": (
        "H3: Direct recurrent-state compatibility is the main transferable mechanism, while "
        "attention/convolution mismatch limits target fidelity; isolate the smallest non-recurrent "
        "correction required for successful handoff."
    ),
    "COUPLING_HYPOTHESIS_REJECTED": (
        "H3: The remaining fidelity gap is not primarily a low-rank component-coupling problem; "
        "test whether a short target-side adaptation prefix can repair imported source state more "
        "effectively than additional state translation."
    ),
}


def direct_heavy(condition: str) -> bool:
    return len(condition) == 3 and condition.count("D") >= 2


def determine_4k_verdict(metrics: dict[str, Any]) -> str:
    improvement_ci = metrics["comparisons"]["base_minus_corrected"]["bootstrap_ci"]
    source_ci = metrics["comparisons"]["corrected_minus_source"]["bootstrap_ci"]
    shuffled_ci = metrics["comparisons"]["corrected_minus_shuffled"]["bootstrap_ci"]
    identity_ci = metrics["comparisons"]["ttt_minus_base"]["bootstrap_ci"]
    joint_works = improvement_ci[0] > 0 and metrics["remaining_gap_reduction"] >= 0.25
    full = (
        joint_works
        and metrics["corrected_delta_nll"] <= 0.10
        and metrics["native_context_recovery"] >= 0.90
        and source_ci[1] < 0
        and shuffled_ci[1] < 0
    )
    near = (
        full
        and metrics["corrected_delta_nll"] <= 0.05
        and metrics["native_context_recovery"] >= 0.95
        and metrics["top1_agreement_native"] >= 0.90
    )
    if near:
        return "NEAR_NATIVE_HANDOFF"
    if full:
        return "FULL_STATE_HANDOFF"
    if joint_works:
        return "JOINT_CORRECTION_WORKS"
    if direct_heavy(metrics["base_state"]) and identity_ci[0] > 0:
        return "IDENTITY_STATE_CONFIRMED"
    if metrics["coupling_replication"]["weak_or_inconsistent"]:
        return "COUPLING_HYPOTHESIS_REJECTED"
    raise RuntimeError(
        "E002 verdict ladder is unassigned: correction failed, identity criterion failed, and "
        "LOCKED interactions were not weak/inconsistent"
    )


def final_verdict(verdict_4k: str, *, long_pass: bool | None) -> str:
    if verdict_4k != "NEAR_NATIVE_HANDOFF":
        if long_pass is not None:
            raise ValueError("LONG result supplied when 4K did not unlock LONG")
        return verdict_4k
    return "LONG_CONTEXT_HANDOFF" if long_pass else "NEAR_NATIVE_HANDOFF"


def next_hypothesis(verdict: str) -> str:
    try:
        return NEXT_HYPOTHESES[verdict]
    except KeyError as error:
        raise ValueError(f"no scientific H3 is defined for verdict {verdict}") from error

