"""Frozen document-level statistics and canonical E001 verdict ladder."""

from __future__ import annotations

from typing import Any

import numpy as np

from experiments.latentport.e001_handoff.analysis.metrics import paired_bootstrap_mean_ci


CONDITIONS = (
    "NATIVE_9B",
    "SOURCE_4B",
    "EMPTY_9B",
    "KV_ONLY",
    "KV_GDN_DIRECT",
    "FULL_TRANSLATED",
    "FULL_SHUFFLED",
)


NEXT_HYPOTHESES = {
    "STRONG_FULL_STATE_HANDOFF": "H2: A single intermediate model-state representation can replace pairwise source-to-target translators across multiple Qwen3.5 model sizes without materially reducing handoff fidelity.",
    "FULL_STATE_HANDOFF": "H2: Full-state handoff can be made practically faster than native target prefill by fusing KV and Gated DeltaNet translation on GPU while preserving the E001 fidelity result.",
    "RECURRENT_STATE_TRANSLATABLE": "H2: Cross-component coupling between translated KV and translated Gated DeltaNet state is the main remaining source of target-fidelity loss, and a jointly fitted low-rank correction can close the gap without a deep nonlinear translator.",
    "KV_ONLY_TRANSFER": "H2: Qwen3.5 Gated DeltaNet recurrent state is not aligned by simple bilinear translation, but a small oracle decomposition can determine whether failure is caused by state-space mismatch or by nonlinear coupling with convolution and attention memory.",
    "NO_TRANSLATABLE_STATE": "H2: Cross-model handoff failure arises because persistent memory is model-specific despite matching tensor geometry; test whether semantically matched target state can be recovered from a short target-side bridge prefix instead of direct state translation.",
}


def _ci(values: np.ndarray, seed: int) -> tuple[float, float]:
    return paired_bootstrap_mean_ci(values, resamples=10_000, seed=seed, level=0.95)


def compute_locked_statistics(rows: list[dict[str, Any]], *, bootstrap_seed: int) -> dict[str, Any]:
    if len(rows) != 64:
        raise ValueError(f"Canonical LOCKED statistics require 64 documents, got {len(rows)}")
    nll = {
        condition: np.array([row["conditions"][condition]["nll"] for row in rows], dtype=np.float64)
        for condition in CONDITIONS
    }
    if not all(np.isfinite(values).all() for values in nll.values()):
        stable = False
    else:
        stable = True
    native = nll["NATIVE_9B"]
    delta = {condition: nll[condition] - native for condition in CONDITIONS}
    full_vs_kv = nll["KV_ONLY"] - nll["FULL_TRANSLATED"]
    full_vs_shuffled = nll["FULL_SHUFFLED"] - nll["FULL_TRANSLATED"]
    full_vs_empty = nll["EMPTY_9B"] - nll["FULL_TRANSLATED"]
    kv_vs_empty = nll["EMPTY_9B"] - nll["KV_ONLY"]
    positive_denominator = delta["KV_ONLY"] > 0
    improvement_fractions = np.full(len(rows), np.nan)
    improvement_fractions[positive_denominator] = (
        delta["KV_ONLY"][positive_denominator] - delta["FULL_TRANSLATED"][positive_denominator]
    ) / delta["KV_ONLY"][positive_denominator]
    tqr_denominator = nll["SOURCE_4B"] - native
    valid_tqr = tqr_denominator > 0
    per_document_tqr = np.full(len(rows), np.nan)
    per_document_tqr[valid_tqr] = (
        nll["SOURCE_4B"][valid_tqr] - nll["FULL_TRANSLATED"][valid_tqr]
    ) / tqr_denominator[valid_tqr]
    aggregate_tqr_denominator = nll["SOURCE_4B"].mean() - native.mean()
    aggregate_nll_tqr = (
        (nll["SOURCE_4B"].mean() - nll["FULL_TRANSLATED"].mean()) / aggregate_tqr_denominator
        if aggregate_tqr_denominator > 0
        else None
    )
    condition_summary = {}
    for condition in CONDITIONS:
        condition_rows = [row["conditions"][condition] for row in rows]
        condition_summary[condition] = {
            "mean_nll": float(nll[condition].mean()),
            "median_nll": float(np.median(nll[condition])),
            "mean_delta_nll": float(delta[condition].mean()),
            "median_delta_nll": float(np.median(delta[condition])),
            "mean_top1_agreement_to_native": (
                float(np.mean([value.get("top1_agreement_to_native", 1.0) for value in condition_rows]))
                if condition != "SOURCE_4B"
                else None
            ),
            "mean_top5_overlap_to_native": (
                float(np.mean([value.get("mean_top5_overlap_to_native", 1.0) for value in condition_rows]))
                if condition != "SOURCE_4B"
                else None
            ),
            "mean_js_to_native": (
                float(np.mean([value.get("mean_js_to_native", 0.0) for value in condition_rows]))
                if condition not in ("NATIVE_9B", "SOURCE_4B")
                else (0.0 if condition == "NATIVE_9B" else None)
            ),
        }
    finite_improvement_fraction = improvement_fractions[positive_denominator]
    stats = {
        "stable_finite_behavior": stable,
        "documents": len(rows),
        "condition_summary": condition_summary,
        "full_vs_kv_mean_nll_improvement": float(full_vs_kv.mean()),
        "full_vs_kv_bootstrap_ci": list(_ci(full_vs_kv, bootstrap_seed)) if stable else [None, None],
        "full_vs_kv_mean_improvement_fraction": (
            float(finite_improvement_fraction.mean()) if finite_improvement_fraction.size else None
        ),
        "full_vs_kv_median_improvement_fraction": (
            float(np.median(finite_improvement_fraction)) if finite_improvement_fraction.size else None
        ),
        "full_vs_kv_improvement_fraction_documents": int(positive_denominator.sum()),
        "full_vs_shuffled_mean_nll_improvement": float(full_vs_shuffled.mean()),
        "full_vs_shuffled_bootstrap_ci": list(_ci(full_vs_shuffled, bootstrap_seed))
        if stable
        else [None, None],
        "full_vs_empty_mean_nll_improvement": float(full_vs_empty.mean()),
        "full_vs_empty_bootstrap_ci": list(_ci(full_vs_empty, bootstrap_seed)) if stable else [None, None],
        "kv_vs_empty_mean_nll_improvement": float(kv_vs_empty.mean()),
        "kv_vs_empty_bootstrap_ci": list(_ci(kv_vs_empty, bootstrap_seed)) if stable else [None, None],
        "mean_document_tqr": float(np.nanmean(per_document_tqr)) if valid_tqr.any() else None,
        "median_document_tqr": float(np.nanmedian(per_document_tqr)) if valid_tqr.any() else None,
        "tqr_exclusion_count": int((~valid_tqr).sum()),
        "aggregate_nll_tqr": None if aggregate_nll_tqr is None else float(aggregate_nll_tqr),
        "per_document": [
            {
                "document_id": row["document_id"],
                "nll": {condition: float(nll[condition][index]) for condition in CONDITIONS},
                "delta_nll": {condition: float(delta[condition][index]) for condition in CONDITIONS},
                "improvement_fraction": (
                    None if not positive_denominator[index] else float(improvement_fractions[index])
                ),
                "tqr": None if not valid_tqr[index] else float(per_document_tqr[index]),
            }
            for index, row in enumerate(rows)
        ],
    }
    return stats


def choose_verdict(stats: dict[str, Any], *, long_run: bool = False, long_pass: bool = False) -> str:
    if not stats["stable_finite_behavior"]:
        return "NO_TRANSLATABLE_STATE"
    full_beats_empty = stats["full_vs_empty_bootstrap_ci"][0] > 0
    kv_beats_empty = stats["kv_vs_empty_bootstrap_ci"][0] > 0
    full_beats_kv = stats["full_vs_kv_bootstrap_ci"][0] > 0
    recurrent = (
        full_beats_kv
        and stats["full_vs_kv_mean_improvement_fraction"] is not None
        and stats["full_vs_kv_median_improvement_fraction"] is not None
        and stats["full_vs_kv_mean_improvement_fraction"] >= 0.25
        and stats["full_vs_kv_median_improvement_fraction"] >= 0.25
    )
    if not full_beats_empty:
        return "NO_TRANSLATABLE_STATE"
    if not recurrent:
        # Highest attained rung: useful KV transfer without satisfying the
        # complete recurrent contribution gate. If even KV is not useful,
        # no named transfer rung has been attained.
        return "KV_ONLY_TRANSFER" if kv_beats_empty else "NO_TRANSLATABLE_STATE"
    full_state = (
        stats["condition_summary"]["FULL_TRANSLATED"]["mean_delta_nll"] <= 0.20
        and stats["aggregate_nll_tqr"] is not None
        and stats["aggregate_nll_tqr"] >= 0.75
        and stats["full_vs_shuffled_bootstrap_ci"][0] > 0
    )
    if not full_state:
        return "RECURRENT_STATE_TRANSLATABLE"
    strong_4k = (
        stats["condition_summary"]["FULL_TRANSLATED"]["mean_delta_nll"] <= 0.10
        and stats["aggregate_nll_tqr"] >= 0.90
    )
    if strong_4k and long_run and long_pass:
        return "STRONG_FULL_STATE_HANDOFF"
    return "FULL_STATE_HANDOFF"
