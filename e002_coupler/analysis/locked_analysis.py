"""Canonical document-level E002 LOCKED aggregation and verdict freeze."""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

import numpy as np

from experiments.latentport.e002_coupler.analysis.metrics import (
    native_context_recovery,
    remaining_gap_reduction,
    tqr,
)
from experiments.latentport.e002_coupler.analysis.statistics import (
    bootstrap_mean_ci,
    bootstrap_statistic_ci,
    summarize_factorial,
)
from experiments.latentport.e002_coupler.analysis.verdict import determine_4k_verdict
from experiments.latentport.e002_coupler.runtime.constants import (
    ATTEMPT_ROOT,
    FACTORIAL_CONDITIONS,
    PRIMARY_LOCKED_CONDITIONS,
)
from experiments.latentport.e002_coupler.runtime.lock_guard import sha256_file, verify_frozen_manifest
from experiments.latentport.e002_coupler.runtime.state_io import write_json_once


RAW = ATTEMPT_ROOT / "locked" / "raw_evidence.jsonl"
METRICS = ATTEMPT_ROOT / "locked" / "locked_metrics.json"
VERDICT_4K = ATTEMPT_ROOT / "verdict" / "4k_verdict.json"


def _read_raw() -> list[dict[str, Any]]:
    verify_frozen_manifest()
    execution = json.loads(
        (ATTEMPT_ROOT / "locked" / "locked_execution.json").read_text(encoding="utf-8")
    )
    if execution["status"] != "LOCKED_COMPLETE" or execution["raw_evidence_sha256"] != sha256_file(RAW):
        raise RuntimeError("LOCKED evidence is incomplete or hash-mismatched")
    with RAW.open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if len(rows) != 64:
        raise RuntimeError("LOCKED raw evidence must contain all 64 documents")
    return rows


def _paired(values: np.ndarray) -> dict[str, Any]:
    return {
        "mean_difference": float(np.mean(values)),
        "median_difference": float(np.median(values)),
        "bootstrap_ci": list(bootstrap_mean_ci(values)),
        "documents": int(values.size),
        "bootstrap_resamples": 10_000,
        "unit": "document",
    }


def _rgr_statistic(rows: np.ndarray) -> float:
    base_delta = float(np.mean(rows[:, 0] - rows[:, 2]))
    corrected_delta = float(np.mean(rows[:, 1] - rows[:, 2]))
    if abs(base_delta) < 1e-12:
        raise ValueError("RGR bootstrap denominator is numerically zero")
    return (base_delta - corrected_delta) / base_delta


def _condition_summary(rows: list[dict[str, Any]], condition: str, native_mean: float) -> dict[str, Any]:
    values = [row["conditions"][condition] for row in rows]
    nll = float(np.mean([value["nll"] for value in values]))
    return {
        "nll": nll,
        "delta_nll": nll - native_mean,
        "mean_entropy": float(np.mean([np.mean(value["per_token_entropy"]) for value in values])),
        "top1_agreement_native": float(
            np.mean([value.get("top1_agreement_to_native", 1.0) for value in values])
        ),
        "top5_overlap_native": float(
            np.mean([value.get("mean_top5_overlap_to_native", 1.0) for value in values])
        ),
        "js_divergence_native": float(
            np.mean([value.get("mean_js_to_native", 0.0) for value in values])
        ),
        "document_nll": [float(value["nll"]) for value in values],
    }


def _repair_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for checkpoint in (1, 4, 16, 64, 256):
        result[str(checkpoint)] = {}
        for condition in ("BASE_STATE", "JOINT_CORRECTED"):
            records = [row["state_repair"][condition][str(checkpoint)] for row in rows]
            result[str(checkpoint)][condition] = {
                "recurrent_normalized_frobenius": float(
                    np.mean([record["mean_gdn_recurrent_normalized_frobenius"] for record in records])
                ),
                "recurrent_cosine": float(
                    np.mean([record["mean_gdn_recurrent_cosine"] for record in records])
                ),
                "convolution_normalized_l2": float(
                    np.mean([record["mean_convolution_normalized_l2"] for record in records])
                ),
                "new_key_normalized_l2": float(
                    np.mean([record["mean_new_key_normalized_l2"] for record in records])
                ),
                "new_value_normalized_l2": float(
                    np.mean([record["mean_new_value_normalized_l2"] for record in records])
                ),
            }
    return result


def _magnitude_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    layer_rows = [
        record
        for row in rows
        for record in row["correction_magnitude"]["layer_relative_magnitudes"]
    ]
    by_component = {}
    for component in ("K", "R", "C"):
        values = np.asarray(
            [record["relative_norm"] for record in layer_rows if record["component"] == component],
            dtype=np.float64,
        )
        by_component[component] = {
            "median": float(np.median(values)),
            "mean": float(np.mean(values)),
            "p90": float(np.quantile(values, 0.9)),
            "maximum": float(np.max(values)),
        }
    values = np.asarray([record["relative_norm"] for record in layer_rows], dtype=np.float64)
    return {
        "by_component": by_component,
        "all_layers": {
            "median": float(np.median(values)),
            "mean": float(np.mean(values)),
            "p90": float(np.quantile(values, 0.9)),
            "maximum": float(np.max(values)),
        },
        "layer_records": layer_rows,
    }


def _timing(rows: list[dict[str, Any]]) -> dict[str, float]:
    median = lambda values: float(statistics.median(float(value) for value in values))
    native = median(row["timing"]["native_9b_prefill"]["wall_ms"] for row in rows)
    source = median(row["timing"]["source_4b_prefill"]["wall_ms"] for row in rows)
    base = median(row["timing"]["base_construction_ms"] for row in rows)
    correction = median(row["timing"]["joint_correction_ms"] for row in rows)
    install = median(row["timing"]["state_install_ms"]["JOINT_CORRECTED"] for row in rows)
    bridge = median(row["timing"]["bridge_ms"]["JOINT_CORRECTED"] for row in rows)
    available = native - source
    return {
        "aggregation": "document median wall-clock latency",
        "native_9b_prefill_ms": native,
        "source_4b_prefill_ms": source,
        "base_construction_ms": base,
        "joint_correction_ms": correction,
        "state_install_ms": install,
        "bridge_ms": bridge,
        "prototype_handoff_overhead_ms": base + correction + install + bridge,
        "available_translation_budget_ms": available,
        "handoff_margin_ms": available - (base + correction + install + bridge),
    }


def aggregate() -> dict[str, Any]:
    rows = _read_raw()
    native_mean = float(np.mean([row["conditions"]["NATIVE_9B"]["nll"] for row in rows]))
    summaries = {
        condition: _condition_summary(rows, condition, native_mean)
        for condition in PRIMARY_LOCKED_CONDITIONS
    }
    nll = {
        condition: np.asarray(summary["document_nll"], dtype=np.float64)
        for condition, summary in summaries.items()
    }
    base_delta = summaries["BASE_STATE"]["delta_nll"]
    corrected_delta = summaries["JOINT_CORRECTED"]["delta_nll"]
    comparisons = {
        "base_minus_corrected": _paired(nll["BASE_STATE"] - nll["JOINT_CORRECTED"]),
        "corrected_minus_source": _paired(nll["JOINT_CORRECTED"] - nll["SOURCE_4B"]),
        "corrected_minus_shuffled": _paired(nll["JOINT_CORRECTED"] - nll["JOINT_SHUFFLED"]),
        "ttt_minus_base": _paired(nll["TTT"] - nll["BASE_STATE"]),
    }
    rgr = remaining_gap_reduction(base_delta, corrected_delta)
    if rgr is None:
        raise RuntimeError("canonical RGR is undefined because the LOCKED base gap is non-positive")
    rgr_rows = np.column_stack((nll["BASE_STATE"], nll["JOINT_CORRECTED"], nll["NATIVE_9B"]))
    rgr_ci = list(bootstrap_statistic_ci(rgr_rows, _rgr_statistic))
    factorial_rows = [
        {
            condition: row["conditions"][condition]["nll"]
            - row["conditions"]["NATIVE_9B"]["nll"]
            for condition in FACTORIAL_CONDITIONS
        }
        for row in rows
    ]
    factorial_effects = summarize_factorial(factorial_rows)
    validation = json.loads(
        (ATTEMPT_ROOT / "factorial" / "base_state_selection.json").read_text(encoding="utf-8")
    )["interaction_metrics"]
    validation_effects = validation["effects"]
    pairwise = ("KV_x_recurrent", "KV_x_convolution", "recurrent_x_convolution")
    directional = {
        name: bool(
            np.sign(factorial_effects["effects"][name]["estimate"])
            == np.sign(validation_effects[name]["estimate"])
        )
        for name in pairwise
    }
    material_locked = {
        name: abs(factorial_effects["effects"][name]["estimate"]) >= 0.02 for name in pairwise
    }
    coupling = {
        "directional_replication": directional,
        "material_locked": material_locked,
        "weak_or_inconsistent": not any(
            directional[name] and material_locked[name] for name in pairwise
        ),
    }
    frozen = verify_frozen_manifest()
    metrics = {
        "documents": 64,
        "unit": "document",
        "bootstrap_resamples": 10_000,
        "base_state": frozen["base_state"],
        "conditions": summaries,
        "factorial_interactions": factorial_effects,
        "coupling_replication": coupling,
        "base_delta_nll": base_delta,
        "corrected_delta_nll": corrected_delta,
        "remaining_gap_reduction": rgr,
        "remaining_gap_bootstrap_ci": rgr_ci,
        "native_context_recovery": native_context_recovery(
            summaries["EMPTY_9B"]["nll"],
            summaries["JOINT_CORRECTED"]["nll"],
            summaries["NATIVE_9B"]["nll"],
        ),
        "tqr": tqr(
            summaries["SOURCE_4B"]["nll"],
            summaries["JOINT_CORRECTED"]["nll"],
            summaries["NATIVE_9B"]["nll"],
        ),
        "top1_agreement_native": summaries["JOINT_CORRECTED"]["top1_agreement_native"],
        "js_divergence_native": summaries["JOINT_CORRECTED"]["js_divergence_native"],
        "comparisons": comparisons,
        "correction_magnitude": _magnitude_summary(rows),
        "state_repair": _repair_summary(rows),
        "timing": _timing(rows),
        "raw_evidence_sha256": sha256_file(RAW),
    }
    return metrics


def main() -> None:
    metrics = aggregate()
    verdict = determine_4k_verdict(metrics)
    write_json_once(METRICS, metrics)
    record = {
        "freeze_state": "CANONICAL_4K_VERDICT_FROZEN_BEFORE_ABLATIONS",
        "canonical_4k_verdict": verdict,
        "locked_metrics_sha256": sha256_file(METRICS),
        "post_verdict_ablations_started": False,
        "long_unlocked": verdict == "NEAR_NATIVE_HANDOFF",
    }
    write_json_once(VERDICT_4K, record)
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
