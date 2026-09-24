"""Deterministic extraction of the sealed E001/E002 public observations."""
from __future__ import annotations

import sys
import numpy as np

from common import (
    ROOT, E1, E2, A1, A2, R1, R2, M1, M2, AB, FA, CELLS, C1, C2, SOURCES,
    array, bootstrap, canonical, read, read_lines, require, sha256, validate_roster,
    write_csv, write_json, print_sources,
)

RAW1 = f"{A1}/locked/raw_evidence.jsonl"
RAW2 = f"{A2}/locked/raw_evidence.jsonl"
S1 = f"{A1}/locked/derived_statistics.json"
S2 = f"{A2}/locked/locked_metrics.json"
SELECTION = f"{A2}/fit/correction_selection.json"


def public_value(value):
    """Remove machine-local paths from generated summaries. Sealed inputs stay unchanged."""
    if isinstance(value, str):
        normalized = value.replace("\\", "/")
        drive = len(normalized) > 2 and normalized[1] == ":" and normalized[2] == "/"
        home = normalized.startswith("/Users/") or normalized.startswith("/home/")
        marker = "site-packages/"
        if marker in normalized and (drive or home or normalized.startswith("/")):
            return normalized.split(marker, 1)[1]
        if drive or home:
            return normalized.rsplit("/", 1)[-1]
        return value
    if isinstance(value, list):
        return [public_value(item) for item in value]
    if isinstance(value, dict):
        return {key: public_value(item) for key, item in value.items()}
    return value


def record(value, source, pointer, method="sealed_aggregate", ci=None, ci_source=None,
           observations=None):
    return {"estimate": float(value), "ci95": ci, "source_artifact": source,
            "json_path": pointer, "method": method, "ci_source": ci_source,
            "observation_artifact": observations}


def ratio(numerator, denominator, positive_only=False):
    if denominator == 0 or (positive_only and denominator <= 0):
        return None
    return float(numerator / denominator)


def load_raw(path, roster, expected_hash, conditions):
    require(sha256(ROOT / path) == expected_hash, f"Raw evidence hash mismatch: {path}")
    rows = read_lines(path)
    require(len(rows) == len(roster), f"Raw document count changed: {path}")
    for index, (row, manifest) in enumerate(zip(rows, roster)):
        require(row["document_id"] == manifest["document_id"] and row["document_index"] == index,
                f"Document ordering mismatch: {path}, row {index}")
        require(row["split"] == "LOCKED" and set(row["conditions"]) == set(conditions),
                f"Split or condition schema changed: {path}, row {index}")
        array([row["conditions"][condition]["nll"] for condition in conditions])
    return rows


def build():
    SOURCES.clear()
    r1, r2 = canonical()
    p1, p2 = read(f"{E1}/PREREGISTRATION.json"), read(f"{E2}/PREREGISTRATION.json")
    s1, s2 = read(S1), read(S2)
    roster1 = validate_roster(read_lines(M1), 64, "LOCKED")
    roster2 = validate_roster(read_lines(M2), 64, "LOCKED")
    require([r["selection_sha256"] for r in roster2] ==
            sorted(r["selection_sha256"] for r in roster2), "E002 hash order changed")
    all_c2 = {**C2, **{cell: f"{cell.lower()}_nll" for cell in CELLS}}
    raw1 = load_raw(RAW1, roster1, r1["raw_evidence_sha256"], C1)
    raw2 = load_raw(RAW2, roster2, r2["locked_raw_evidence_hash"], all_c2)

    datasets = []
    for raw, mapping, source, condition in (
        (raw1, C1, RAW1, "full_translated_nll"),
        (raw2, all_c2, RAW2, "joint_corrected_nll"),
    ):
        rows = []
        for index, document in enumerate(raw):
            row = {"document_id": document["document_id"], "split": "LOCKED", "document_index": index}
            row.update({column: float(document["conditions"][name]["nll"]) for name, column in mapping.items()})
            row["ncr"] = ratio(row["empty_9b_nll"] - row[condition],
                               row["empty_9b_nll"] - row["native_9b_nll"])
            row["tqr"] = ratio(row["source_4b_nll"] - row[condition],
                               row["source_4b_nll"] - row["native_9b_nll"], positive_only=True)
            if source == RAW1:
                row["improvement_fraction"] = ratio(
                    row["kv_only_nll"] - row[condition],
                    row["kv_only_nll"] - row["native_9b_nll"], positive_only=True)
            row["wrong_donor_document_id"] = document["shuffle"]["donor_document_id"]
            row["observation_status"] = "raw_observations"
            row["source_artifact"] = source
            rows.append(row)
        datasets.append(rows)
    e1, e2 = datasets

    ablations = read_lines(AB)
    require(len(ablations) == 64, "E002 ablation cohort changed")
    ablation_rows = []
    for index, (manifest, ablation) in enumerate(zip(roster2, ablations)):
        require(ablation["document_index"] == index and ablation["document_id"] == manifest["document_id"],
                "E002 ablation alignment failed")
        require(len(ablation["outcomes"]) == 6, "E002 ablation schema changed")
        for name, outcome in sorted(ablation["outcomes"].items()):
            ablation_rows.append({
                "document_id": manifest["document_id"], "split": "LOCKED_POSTVERDICT",
                "document_index": index, "condition": name, "nll": float(outcome["nll"]),
                "delta_nll_to_native": float(outcome["delta_nll_to_native"]),
                "impact_vs_joint_corrected": float(outcome["impact_vs_joint_corrected"]),
                "source_artifact": AB,
            })

    factorial = read_lines(FA)
    validation_roster = validate_roster(
        read_lines(f"{E2}/data/validation_factorial/manifest.jsonl"), 32, "VALIDATION_FACTORIAL")
    require(len(factorial) == 32 and [r["document_id"] for r in factorial] ==
            [r["document_id"] for r in validation_roster], "E002 validation alignment changed")
    validation = []
    for index, raw in enumerate(factorial):
        row = {"document_id": raw["document_id"], "split": "VALIDATION_FACTORIAL",
               "document_index": index, "native_9b_nll": float(raw["native_9b"]["nll"])}
        row.update({f"{cell.lower()}_nll": float(raw["conditions"][cell]["nll"]) for cell in CELLS})
        row["source_artifact"] = FA
        validation.append(row)

    def conditions(result, mapping, rows, source, raw_source, seed):
        metrics = {}
        for name, column in mapping.items():
            values = array([row[column] for row in rows])
            pointer = (f"factorial_locked_metrics.{name}.nll" if name in CELLS else column)
            expected = result["factorial_locked_metrics"][name]["nll"] if name in CELLS else result[column]
            require(abs(values.mean() - expected) < 1e-12, f"{name}: sealed mean mismatch")
            metrics[name] = record(values.mean(), source, pointer, "mean_of_raw_document_nll",
                                   bootstrap(values, seed),
                                   "new document-mean bootstrap; not a sealed mean CI", raw_source)
        return metrics

    metrics1 = conditions(r1, C1, e1, R1, RAW1, p1["seeds"]["bootstrap"])
    metrics2 = conditions(r2, all_c2, e2, R2, RAW2, p2["seeds"]["bootstrap"])
    comparisons1 = {
        "kv_vs_empty": record(s1["kv_vs_empty_mean_nll_improvement"], S1, "kv_vs_empty_mean_nll_improvement",
                             "paired_document_difference", s1["kv_vs_empty_bootstrap_ci"],
                             "kv_vs_empty_bootstrap_ci", RAW1),
        "full_vs_kv": record(r1["full_vs_kv_delta"], R1, "full_vs_kv_delta",
                            "paired_document_difference", r1["full_vs_kv_bootstrap_ci"],
                            "full_vs_kv_bootstrap_ci", RAW1),
        "full_vs_wrong_donor": record(r1["full_vs_shuffled_delta"], R1, "full_vs_shuffled_delta",
                                     "paired_document_difference", r1["full_vs_shuffled_bootstrap_ci"],
                                     "full_vs_shuffled_bootstrap_ci", RAW1),
        "mean_document_improvement_fraction": record(r1["full_vs_kv_improvement_fraction"], R1,
                                                     "full_vs_kv_improvement_fraction",
                                                     "mean_of_valid_document_ratios", observations=RAW1),
        "median_document_improvement_fraction": record(r1["full_vs_kv_median_improvement_fraction"], R1,
                                                       "full_vs_kv_median_improvement_fraction",
                                                       "median_of_valid_document_ratios", observations=RAW1),
        "ncr_kv": record((r1["empty_9b_nll"] - r1["kv_only_nll"]) /
                         (r1["empty_9b_nll"] - r1["native_9b_nll"]), R1,
                         "(empty_9b_nll - kv_only_nll) / (empty_9b_nll - native_9b_nll)",
                         "new_ratio_of_document_means", observations=RAW1),
        "ncr_full": record((r1["empty_9b_nll"] - r1["full_translated_nll"]) /
                           (r1["empty_9b_nll"] - r1["native_9b_nll"]), R1,
                           "(empty_9b_nll - full_translated_nll) / (empty_9b_nll - native_9b_nll)",
                           "new_ratio_of_document_means", observations=RAW1),
        "tqr": record(r1["tqr"], R1, "tqr", "ratio_of_document_means", observations=RAW1),
    }
    comparisons2 = {
        "base_minus_corrected": record(r2["base_vs_corrected_improvement"]["mean_difference"], R2,
                                      "base_vs_corrected_improvement.mean_difference",
                                      "paired_document_difference",
                                      r2["base_vs_corrected_improvement"]["bootstrap_ci"],
                                      "base_vs_corrected_improvement.bootstrap_ci", RAW2),
        "corrected_minus_source": record(r2["corrected_vs_source_delta"], R2,
                                        "corrected_vs_source_delta", "paired_document_difference",
                                        r2["corrected_vs_source_bootstrap_ci"], "corrected_vs_source_bootstrap_ci", RAW2),
        "corrected_minus_wrong_donor": record(r2["corrected_vs_shuffled_delta"], R2,
                                             "corrected_vs_shuffled_delta", "paired_document_difference",
                                             r2["corrected_vs_shuffled_bootstrap_ci"],
                                             "corrected_vs_shuffled_bootstrap_ci", RAW2),
        "corrected_excess_nll": record(r2["corrected_delta_nll"], R2, "corrected_delta_nll",
                                      "difference_of_document_means", observations=RAW2),
        "remaining_gap_reduction": record(r2["remaining_gap_reduction"], R2, "remaining_gap_reduction",
                                         "ratio_of_document_means", r2["remaining_gap_bootstrap_ci"],
                                         "remaining_gap_bootstrap_ci", RAW2),
        "ncr": record(r2["native_context_recovery"], R2, "native_context_recovery",
                      "ratio_of_document_means", observations=RAW2),
        "tqr": record(r2["tqr"], R2, "tqr", "ratio_of_document_means", observations=RAW2),
        "ttt_minus_base": record(s2["comparisons"]["ttt_minus_base"]["mean_difference"], S2,
                                 "comparisons.ttt_minus_base.mean_difference", "paired_document_difference",
                                 s2["comparisons"]["ttt_minus_base"]["bootstrap_ci"],
                                 "comparisons.ttt_minus_base.bootstrap_ci", RAW2),
    }
    require(p1["splits"]["locked"]["prefix_tokens"] == p2["splits"]["locked"]["prefix_tokens"],
            "Primary prefix lengths differ")
    require(p1["evaluation"]["teacher_forced_tokens"] == p2["evaluation"]["teacher_forced_tokens"],
            "Primary target counts differ")
    headline = {
        "schema_version": 1, "route": "Qwen3.5 4B → 9B", "evaluation": "teacher-forced",
        "independent_unit": "document", "prefix_tokens": p1["splits"]["locked"]["prefix_tokens"],
        "scored_tokens_per_document": p1["evaluation"]["teacher_forced_tokens"],
        "full_headline_reproduction_available": True,
        "E001": {"result_artifact": R1, "documents": len(e1), "corpus": "PG19",
                 "bootstrap_seed": p1["seeds"]["bootstrap"], "conditions": metrics1,
                 "comparisons": comparisons1, "verdict": r1["canonical_verdict"],
                 "long_test_run": r1["long_test_run"]},
        "E002": {"result_artifact": R2, "documents": len(e2), "corpus": "FineWeb-Edu",
                 "bootstrap_seed": p2["seeds"]["bootstrap"], "conditions": metrics2,
                 "comparisons": comparisons2, "verdict": r2["canonical_verdict"],
                 "long_test_run": r2["long_test_run"]},
    }
    selection = read(SELECTION)
    secondary = {
        "e001_diagnostics": {"source_artifact": f"{A1}/diagnostics/postverdict_analysis.json",
                            "data": read(f"{A1}/diagnostics/postverdict_analysis.json")},
        "e001_fidelity": {"source_artifact": S1, "data": s1["condition_summary"]},
        "e002_fidelity": {"source_artifact": S2, "data": {
            name: {k: v for k, v in metrics.items() if k != "document_nll"}
            for name, metrics in s2["conditions"].items()}},
        "architecture": {"source_artifact": f"{A1}/implementation/architecture_correspondence.json",
                         "data": public_value(read(f"{A1}/implementation/architecture_correspondence.json"))},
        "e002_selection": {"source_artifact": SELECTION, "candidate_grid": selection["candidate_grid"],
                          "selected_candidate": selection["selected"]["selected_candidate"]},
        "e002_correction": {"source_artifact": R2,
                           "inventory": r2["correction_parameter_inventory"],
                           "magnitude": {k: v for k, v in r2["correction_magnitude_summary"].items() if k != "layer_records"},
                           "bytes": r2["correction_bytes"], "rank": r2["correction_rank"]},
        "e002_timing": {"source_artifact": S2, "data": s2["timing"]},
        "e002_ablations": {"source_artifact": f"{A2}/diagnostics/postverdict_ablations.json",
                          "data": read(f"{A2}/diagnostics/postverdict_ablations.json")["ablations"]},
        "e002_factorial": {"source_artifact": R2,
                          "validation": r2["factorial_validation_metrics"],
                          "locked": r2["factorial_locked_metrics"],
                          "effects": r2["factorial_interaction_metrics"]},
        "e002_repair": {"source_artifact": R2,
                       "data": {str(k): r2[f"state_repair_token_{k}"] for k in (1, 4, 16, 64, 256)}},
    }
    headline["source_sha256"] = dict(sorted(SOURCES.items()))
    outputs = {
        "derived/e001_per_document.csv": (list(e1[0]), e1),
        "derived/e002_per_document.csv": (list(e2[0]), e2),
        "derived/e002_factorial_validation.csv": (list(validation[0]), validation),
        "derived/e002_ablation_per_document.csv": (list(ablation_rows[0]), ablation_rows),
    }
    return headline, secondary, outputs


def main():
    headline, secondary, outputs = build()
    for path, (fields, rows) in outputs.items():
        write_csv(path, fields, rows)
        print(f"WROTE {path}: {len(rows)} rows")
    write_json("derived/headline_results.json", headline)
    write_json("derived/secondary_results.json", secondary)
    print_sources()


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, TypeError, OSError) as error:
        sys.exit(f"EXTRACTION FAILED: {error}")
