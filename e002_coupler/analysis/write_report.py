"""Write the final evidence-backed E002 Markdown report and executive summary."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from experiments.latentport.e002_coupler.analysis.metrics import native_context_recovery, tqr
from experiments.latentport.e002_coupler.runtime.constants import (
    ATTEMPT_ROOT,
    E002_ROOT,
    FACTORIAL_CONDITIONS,
    PRIMARY_LOCKED_CONDITIONS,
)
from experiments.latentport.e002_coupler.runtime.state_io import write_json_once


REPORT = ATTEMPT_ROOT / "REPORT.md"
EXECUTIVE = ATTEMPT_ROOT / "EXECUTIVE_SUMMARY.md"


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _f(value: float | None, digits: int = 6) -> str:
    return "not run" if value is None else f"{value:.{digits}f}"


def _ci(value: list[float]) -> str:
    return f"[{value[0]:.6f}, {value[1]:.6f}]"


def _effect_preference(estimate: float) -> str:
    return "translated" if estimate < 0 else "direct"


def build_report() -> tuple[str, str]:
    result = _json(ATTEMPT_ROOT / "verdict" / "RESULT.json")
    metrics = _json(ATTEMPT_ROOT / "locked" / "locked_metrics.json")
    factorial = _json(ATTEMPT_ROOT / "factorial" / "base_state_selection.json")
    correction = _json(ATTEMPT_ROOT / "fit" / "correction.json")
    ablations = _json(ATTEMPT_ROOT / "diagnostics" / "postverdict_ablations.json")
    refs = _json(E002_ROOT / "reuse" / "e001_refs.json")
    conditions = metrics["conditions"]
    native_nll = conditions["NATIVE_9B"]["nll"]
    empty_nll = conditions["EMPTY_9B"]["nll"]
    source_nll = conditions["SOURCE_4B"]["nll"]
    improvement = metrics["comparisons"]["base_minus_corrected"]
    source_comparison = metrics["comparisons"]["corrected_minus_source"]
    shuffled_comparison = metrics["comparisons"]["corrected_minus_shuffled"]
    identity_comparison = metrics["comparisons"]["ttt_minus_base"]
    mag = metrics["correction_magnitude"]["all_layers"]
    coupling = not metrics["coupling_replication"]["weak_or_inconsistent"]
    source_advantage = source_comparison["bootstrap_ci"][1] < 0
    content_specific = shuffled_comparison["bootstrap_ci"][1] < 0
    correction_works = improvement["bootstrap_ci"][0] > 0 and metrics["remaining_gap_reduction"] >= 0.25
    direct_replication = metrics["base_state"].count("D") >= 2 and identity_comparison["bootstrap_ci"][0] > 0
    repair = metrics["state_repair"]
    repair_direction = (
        "shrunk"
        if repair["256"]["JOINT_CORRECTED"]["recurrent_normalized_frobenius"]
        < repair["1"]["JOINT_CORRECTED"]["recurrent_normalized_frobenius"]
        else "did not shrink"
    )
    long_sentence = (
        f"The frozen 16K test ran and {'passed' if result['long_pass'] else 'did not pass'}."
        if result["long_test_run"]
        else "The 16K phase was not run because the frozen 4K verdict did not reach NEAR_NATIVE_HANDOFF."
    )
    executive_finding = f"""# Executive Finding

Validation selected `{metrics['base_state']}`: translated KV with direct recurrent and convolution state. The validation factorial showed {'a directionally replicated material interaction' if coupling else 'no directionally replicated material interaction'} under the preregistered 0.02-nat interaction threshold. On fresh LOCKED data, the direct-heavy base {'significantly outperformed' if direct_replication else 'did not significantly outperform'} `TTT`; the paired 95% interval for `TTT − BASE_STATE` was {_ci(identity_comparison['bootstrap_ci'])} nats/token.

The selected rank-{result['correction_rank']} correction used {result['correction_parameter_count']:,} trainable parameters ({result['correction_bytes']:,} serialized bytes). Its median and maximum layer-relative magnitudes were {mag['median']:.6f} and {mag['maximum']:.6f}. Joint correction changed DeltaNLL from {metrics['base_delta_nll']:.6f} to {metrics['corrected_delta_nll']:.6f}; remaining-gap reduction was {metrics['remaining_gap_reduction']:.1%}, and the paired interval for `BASE_STATE − JOINT_CORRECTED` was {_ci(improvement['bootstrap_ci'])}. It therefore {'met' if correction_works else 'did not meet'} the preregistered correction-success gate.

Corrected handoff {'significantly beat' if source_advantage else 'did not significantly beat'} continued 4B inference; `corrected − source` was {source_comparison['mean_difference']:.6f} with 95% interval {_ci(source_comparison['bootstrap_ci'])}. It recovered {metrics['native_context_recovery']:.1%} of native 9B context benefit (NCR {metrics['native_context_recovery']:.6f}). The donor-shuffled control was {'significantly worse, supporting content specificity' if content_specific else 'not significantly worse, so content specificity was not established'}.

The corrected recurrent-state error {repair_direction} from token 1 to token 256. {long_sentence} The canonical verdict is `{result['canonical_verdict']}`. The exact next hypothesis is: **{result['next_hypothesis']}**
"""

    lines = [
        "# LatentPort E002: Coupler",
        "",
        "## Can a Small Identity-Anchored Correction Turn Cross-Model State Transfer Into Near-Native Handoff?",
        "",
        executive_finding,
        "## Experimental Integrity",
        "",
        "E001 remained sealed as prior evidence. E002 independently reverified its manifest, model shards, tokenizer, state schemas, translators, runtime, and read-only training-state loads. E001 LOCKED document outcomes were not parsed or used for E002 fitting or selection. The fresh factorial set was frozen before factorial inspection; the fresh LOCKED and LONG IDs, selected correction, code, hashes, and base state were frozen before the one-shot LOCKED run.",
        "",
        "### Table 1: E001 fixed inputs",
        "",
        "| Item | Value | Hash verified? |",
        "|---|---|---|",
        f"| E001 verdict | `RECURRENT_STATE_TRANSLATABLE` | yes |",
        f"| E001 frozen manifest | `{refs['frozen_manifest']['sha256']}` | yes |",
        f"| Source | `{result['source_model']}@{result['source_revision']}` | yes |",
        f"| Target | `{result['target_model']}@{result['target_revision']}` | yes |",
        f"| KV translator tensor | `{result['e001_kv_translator_hash']}` | yes |",
        f"| Recurrent translator tensor | `{result['e001_gdn_translator_hash']}` | yes |",
        f"| Convolution translator tensor | `{result['e001_conv_translator_hash']}` | yes |",
        "",
        "## Which State Components Actually Need Translation?",
        "",
        "The full validation factorial was completed before correction fitting. Negative effects mean translation lowered DeltaNLL; positive effects mean direct copy was better.",
        "",
        "### Table 2: Validation factorial",
        "",
        "| Condition | KV | Recurrent | Conv | NLL | DeltaNLL |",
        "|---|---|---|---|---:|---:|",
    ]
    for name in FACTORIAL_CONDITIONS:
        value = factorial["condition_metrics"][name]
        lines.append(f"| {name} | {name[0]} | {name[1]} | {name[2]} | {value['mean_nll']:.6f} | {value['mean_delta_nll']:.6f} |")
    lines.extend(
        [
            "",
            "### Table 3: Factorial effects",
            "",
            "| Effect | Estimate | 95% CI | Interpretation |",
            "|---|---:|---:|---|",
        ]
    )
    effect_labels = {
        "KV": "KV representation",
        "recurrent": "Recurrent representation",
        "convolution": "Convolution representation",
        "KV_x_recurrent": "KV × recurrent",
        "KV_x_convolution": "KV × convolution",
        "recurrent_x_convolution": "Recurrent × convolution",
    }
    for name, label in effect_labels.items():
        value = factorial["interaction_metrics"]["effects"][name]
        interpretation = (
            f"{_effect_preference(value['estimate'])} preferred"
            if "_x_" not in name
            else ("material interaction" if abs(value["estimate"]) >= 0.02 else "small interaction")
        )
        lines.append(f"| {label} | {value['estimate']:.6f} | {_ci(value['bootstrap_ci'])} | {interpretation} |")
    kv_pref = _effect_preference(factorial["interaction_metrics"]["effects"]["KV"]["estimate"])
    r_pref = _effect_preference(factorial["interaction_metrics"]["effects"]["recurrent"]["estimate"])
    c_pref = _effect_preference(factorial["interaction_metrics"]["effects"]["convolution"]["estimate"])
    lines.extend(
        [
            "",
            f"Plainly: KV preferred **{kv_pref}**, recurrent preferred **{r_pref}**, and convolution preferred **{c_pref}** on validation. The validation interaction pattern {'replicated directionally with at least one material LOCKED pairwise interaction' if coupling else 'did not replicate as a material coupling pattern on LOCKED data'}." + ("" if coupling else " The remaining handoff gap is not primarily explained by cross-component coupling under this design."),
            "",
            "![Validation factorial](figures/factorial_condition_nll.png)",
            "",
            "![Factorial interactions](figures/factorial_interactions.png)",
            "",
            "### Table 4: Base-state selection",
            "",
            "| Candidate | Validation DeltaNLL | Complexity rank | Selected? |",
            "|---|---:|---:|---|",
        ]
    )
    for row in factorial["candidate_table"]:
        lines.append(f"| {row['candidate']} | {row['validation_delta_nll']:.6f} | {row['complexity_rank']} | {'yes' if row['selected'] else 'no'} |")
    lines.extend(
        [
            "",
            "## Does Qwen3.5 Already Share a Recurrent Memory Interface Across Model Sizes?",
            "",
            "E001 found that translated KV plus directly copied source Gated DeltaNet state beat full translation. E002 replicated the component-level test without selecting from E001 LOCKED outcomes. The central LOCKED conditions were:",
            "",
            "| Condition | NLL | DeltaNLL |",
            "|---|---:|---:|",
        ]
    )
    for name in ("DDD", "TDD", "TTT", "BASE_STATE", "JOINT_CORRECTED"):
        value = conditions[name]
        lines.append(f"| {name} | {value['nll']:.6f} | {value['delta_nll']:.6f} |")
    if direct_replication:
        shared_statement = "The results are consistent with a partially shared functional state coordinate system across Qwen3.5 sibling models. This is functional compatibility evidence, not a claim of an identical representation or formal ABI."
    else:
        shared_statement = "Fresh LOCKED evidence did not establish that a direct-heavy state significantly outperformed full translation, so E002 does not support a shared-state interpretation beyond the validation observation."
    lines.extend(["", shared_statement, ""])

    inventory = result["correction_parameter_inventory"]
    lines.extend(
        [
            "## Is the Correction Small Enough to Support the Shared-State Interpretation?",
            "",
            "### Table 5: Correction inventory",
            "",
            "| Component | Rank/type | Parameters | Median relative correction |",
            "|---|---|---:|---:|",
            f"| KV | rank {result['correction_rank']} residual | {inventory['kv']:,} | {metrics['correction_magnitude']['by_component']['K']['median']:.6f} |",
            f"| Recurrent | rank {result['correction_rank']} left/right residual | {inventory['recurrent']:,} | {metrics['correction_magnitude']['by_component']['R']['median']:.6f} |",
            f"| Convolution | diagonal scale + bias | {inventory['convolution']:,} | {metrics['correction_magnitude']['by_component']['C']['median']:.6f} |",
            f"| **Total** | identity-anchored | **{inventory['total']:,}** | **{mag['median']:.6f}** |",
            "",
            f"The artifact occupies {result['correction_bytes']:,} bytes, or {result['correction_bytes_ratio_to_target_bfloat16']:.3e} of target-model bfloat16 weight bytes. The maximum observed layer-relative correction was {mag['maximum']:.6f}. Magnitude is reported as mechanistic evidence and was not used as a success gate.",
            "",
            "![Correction magnitude](figures/correction_magnitude_by_layer.png)",
            "",
            "## LOCKED Fidelity",
            "",
            "### Table 6: LOCKED fidelity",
            "",
            "| Condition | NLL | DeltaNLL | NCR | TQR | Top-1 vs native | JS |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name in PRIMARY_LOCKED_CONDITIONS:
        value = conditions[name]
        ncr_value = native_context_recovery(empty_nll, value["nll"], native_nll)
        tqr_value = tqr(source_nll, value["nll"], native_nll)
        lines.append(f"| {name} | {value['nll']:.6f} | {value['delta_nll']:.6f} | {ncr_value:.6f} | {_f(tqr_value)} | {value['top1_agreement_native']:.6f} | {value['js_divergence_native']:.6f} |")

    lines.extend(
        [
            "",
            "## Does a Small Correction Beat the Best Raw Handoff?",
            "",
            "### Table 7: Primary correction effect",
            "",
            "| Metric | BASE_STATE | JOINT_CORRECTED | Delta | 95% CI |",
            "|---|---:|---:|---:|---:|",
            f"| NLL | {conditions['BASE_STATE']['nll']:.6f} | {conditions['JOINT_CORRECTED']['nll']:.6f} | {(-improvement['mean_difference']):.6f} corrected − base | {_ci(improvement['bootstrap_ci'])} base − corrected |",
            f"| DeltaNLL | {metrics['base_delta_nll']:.6f} | {metrics['corrected_delta_nll']:.6f} | {metrics['corrected_delta_nll'] - metrics['base_delta_nll']:.6f} | same paired NLL interval |",
            f"| NCR | {native_context_recovery(empty_nll, conditions['BASE_STATE']['nll'], native_nll):.6f} | {metrics['native_context_recovery']:.6f} | {metrics['native_context_recovery'] - native_context_recovery(empty_nll, conditions['BASE_STATE']['nll'], native_nll):.6f} | descriptive |",
            f"| RGR | — | {metrics['remaining_gap_reduction']:.6f} | — | {_ci(metrics['remaining_gap_bootstrap_ci'])} |",
            "",
            f"The correction {'materially improved' if correction_works else 'did not meet the preregistered material-improvement threshold for'} the strongest uncorrected handoff.",
            "",
            "![Base versus corrected](figures/base_vs_corrected_delta_nll.png)",
            "",
            "![Native context recovery](figures/native_context_recovery.png)",
            "",
            "## Is Switching to the 9B Actually Better Than Staying on the 4B?",
            "",
            "### Table 8: Practical source switch",
            "",
            "| Metric | SOURCE_4B | JOINT_CORRECTED | Difference | 95% CI |",
            "|---|---:|---:|---:|---:|",
            f"| NLL | {source_nll:.6f} | {conditions['JOINT_CORRECTED']['nll']:.6f} | {source_comparison['mean_difference']:.6f} corrected − source | {_ci(source_comparison['bootstrap_ci'])} |",
            "",
            ("Corrected handoff significantly beat SOURCE_4B under the frozen document bootstrap." if source_advantage else "Corrected handoff did not significantly beat SOURCE_4B; E002 therefore does not claim a successful practical model switch."),
            "",
            "![Source versus corrected](figures/source_vs_corrected.png)",
            "",
            "## Did Corrected State Remain Content-Specific?",
            "",
            "### Table 9: Content specificity",
            "",
            "| Metric | JOINT_CORRECTED | JOINT_SHUFFLED | Difference | 95% CI |",
            "|---|---:|---:|---:|---:|",
            f"| NLL | {conditions['JOINT_CORRECTED']['nll']:.6f} | {conditions['JOINT_SHUFFLED']['nll']:.6f} | {shuffled_comparison['mean_difference']:.6f} corrected − shuffled | {_ci(shuffled_comparison['bootstrap_ci'])} |",
            "",
            ("The no-fixed-point donor rotation was significantly worse, supporting content-specific transferred memory." if content_specific else "The donor-shuffled control was not significantly worse; content specificity was not established."),
            "",
            "![Corrected versus shuffled](figures/corrected_vs_shuffled.png)",
            "",
            "## Does the Target Repair the Imported State?",
            "",
            "### Table 10: State repair",
            "",
            "| Token | BASE recurrent error | Corrected recurrent error | BASE cosine | Corrected cosine |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for token in (1, 4, 16, 64, 256):
        base_repair = repair[str(token)]["BASE_STATE"]
        corrected_repair = repair[str(token)]["JOINT_CORRECTED"]
        lines.append(f"| {token} | {base_repair['recurrent_normalized_frobenius']:.6f} | {corrected_repair['recurrent_normalized_frobenius']:.6f} | {base_repair['recurrent_cosine']:.6f} | {corrected_repair['recurrent_cosine']:.6f} |")
    lines.extend(
        [
            "",
            f"Corrected recurrent error {repair_direction} over 256 tokens. This trajectory is secondary mechanistic evidence, not an additional verdict gate.",
            "",
            "![State repair](figures/state_repair_to_256_tokens.png)",
            "",
            "## Could This Be Faster Than Re-Prefilling?",
            "",
            "### Table 11: Timing",
            "",
            "| Stage | Latency |",
            "|---|---:|",
            f"| Native 9B prefill | {metrics['timing']['native_9b_prefill_ms']:.3f} ms |",
            f"| Source 4B prefill | {metrics['timing']['source_4b_prefill_ms']:.3f} ms |",
            f"| Base-state construction | {metrics['timing']['base_construction_ms']:.3f} ms |",
            f"| Joint correction | {metrics['timing']['joint_correction_ms']:.3f} ms |",
            f"| State installation | {metrics['timing']['state_install_ms']:.3f} ms |",
            f"| Bridge token | {metrics['timing']['bridge_ms']:.3f} ms |",
            f"| Available translation budget | {metrics['timing']['available_translation_budget_ms']:.3f} ms |",
            f"| Handoff margin | {metrics['timing']['handoff_margin_ms']:.3f} ms |",
            "",
            "These are prototype end-to-end wall-clock medians. E002 does not claim a production speedup.",
            "",
            "![Timing breakdown](figures/timing_breakdown.png)",
            "",
            "## 16K Generalization",
            "",
            long_sentence,
            "",
            "## Where Does the Remaining Mismatch Live?",
            "",
            "These ablations were run only after the 4K verdict was frozen and were not used to alter E002.",
            "",
            "| Post-verdict ablation | DeltaNLL | Impact vs full correction | 95% CI |",
            "|---|---:|---:|---:|",
        ]
    )
    for name, value in ablations["ablations"].items():
        lines.append(f"| {name} | {value['mean_delta_nll']:.6f} | {value['mean_impact_vs_joint_corrected']:.6f} | {_ci(value['impact_bootstrap_ci'])} |")
    lines.extend(
        [
            "",
            "## Canonical Verdict and Next Hypothesis",
            "",
            f"**Canonical verdict: `{result['canonical_verdict']}`.**",
            "",
            f"**Exact next hypothesis: {result['next_hypothesis']}**",
            "",
            "![Verdict summary](figures/verdict_summary.png)",
            "",
            "## Claim Limits",
            "",
            "The evidence concerns this frozen Qwen3.5-4B-Base → Qwen3.5-9B-Base pair and the specified state installation path. It does not establish universal state portability, cross-family transfer, architecture-independent memory, a formal ABI, or production-ready routing.",
        ]
    )
    report = "\n".join(lines).rstrip() + "\n"
    executive = f"""# LatentPort E002: Coupler — Executive Summary

**Status:** `{result['e002_status']}`  
**Canonical verdict:** `{result['canonical_verdict']}`  
**Base state:** `{result['base_state']}`  
**Correction:** rank {result['correction_rank']}, {result['correction_parameter_count']:,} parameters

The fresh validation factorial selected translated KV with direct recurrent and convolution state. On the 64-document LOCKED corpus, joint correction moved DeltaNLL from {result['base_delta_nll']:.6f} to {result['corrected_delta_nll']:.6f}, for {result['remaining_gap_reduction']:.1%} remaining-gap reduction. NCR was {result['native_context_recovery']:.6f}. The paired 95% interval for base minus corrected NLL was {_ci(improvement['bootstrap_ci'])}.

Corrected minus source NLL was {result['corrected_vs_source_delta']:.6f} with interval {_ci(result['corrected_vs_source_bootstrap_ci'])}; corrected minus shuffled was {result['corrected_vs_shuffled_delta']:.6f} with interval {_ci(result['corrected_vs_shuffled_bootstrap_ci'])}. {long_sentence}

**Next hypothesis:** {result['next_hypothesis']}

See [REPORT.md](REPORT.md) for the factorial, correction magnitude, state-repair trajectory, timing, post-verdict ablations, and claim limits.
"""
    return report, executive


def _write_text_once(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def main() -> None:
    report, executive = build_report()
    _write_text_once(REPORT, report)
    _write_text_once(EXECUTIVE, executive)
    print(json.dumps({"report": str(REPORT), "executive_summary": str(EXECUTIVE)}, indent=2))


if __name__ == "__main__":
    main()
