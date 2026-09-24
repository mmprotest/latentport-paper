"""Verify available observations; fail closed when complete headline checks are blocked."""
from __future__ import annotations

import argparse
import ast
import sys
from typing import Any, Callable

import numpy as np

from common import (
    ROOT, E1, E2, A1, A2, R1, R2, M1, M2, AB, CELLS, C1, C2, SOURCES,
    array, bootstrap, ratio_bootstrap, contrasts, canonical, csv_rows, read,
    read_lines, require, sha256,
)

FAILURES = []
BLOCKED = []
COUNT = 0


def check(name, actual, expected, atol=1e-12):
    global COUNT
    COUNT += 1
    try:
        if isinstance(expected, (str, bool)) or expected is None:
            valid = actual == expected
        else:
            valid = bool(np.allclose(actual, expected, rtol=0, atol=atol, equal_nan=False))
        if not valid:
            raise ValueError(f"actual={actual!r}, sealed={expected!r}")
        print(f"{name:.<78} PASS")
    except (ValueError, TypeError) as error:
        FAILURES.append(f"{name}: {error}")
        print(f"{name:.<78} FAIL")


def blocked(name, needed):
    BLOCKED.append(f"{name}: requires {needed}")
    print(f"{name:.<78} BLOCKED")


def col(rows, name):
    return array([float(row[name]) for row in rows])


def sealed_function(relative, names, extra=None):
    """Cross-check exact pure functions without importing sealed inference modules."""
    tree = ast.parse((ROOT / relative).read_text(encoding="utf-8"))
    definitions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    require({node.name for node in definitions} == set(names), f"Missing pure functions in {relative}")
    scope = {"np": np, "Any": Any, "Callable": Callable, "FACTORIAL_CONDITIONS": CELLS}
    scope.update(extra or {})
    exec(compile(ast.Module(body=definitions, type_ignores=[]), relative, "exec"), scope)
    return scope


def verify_hashes():
    counts = {"matched": 0, "absent": 0, "large_skipped": 0}
    for root in (E1, E2):
        for relative in ("FROZEN_MANIFEST.json", "artifacts/attempt_001/FINAL_ARTIFACT_MANIFEST.json"):
            manifest = read(f"{root}/{relative}")
            base = ROOT / root
            if root == E1 and relative.startswith("artifacts/"):
                base = base / "artifacts/attempt_001"
            for entry in manifest["files"]:
                path = (base / entry["path"]).resolve()
                require(path.is_relative_to(ROOT / root), "Manifest path escapes evidence package")
                if not path.exists():
                    counts["absent"] += 1
                elif path.stat().st_size > 10_000_000:
                    counts["large_skipped"] += 1
                else:
                    if sha256(path) != entry["sha256"]:
                        FAILURES.append(f"Sealed hash mismatch: {path.relative_to(ROOT).as_posix()}")
                    else:
                        counts["matched"] += 1
    check("Included lightweight artifacts match original manifest hashes",
          not any("hash mismatch" in f for f in FAILURES), True)
    print(f"Manifest entries: {counts}. Missing full-run artifacts are not certified.")


def verify():
    r1, r2 = canonical()
    derived = read("derived/headline_results.json")
    for path, digest in derived["source_sha256"].items():
        check(f"Source hash: {path}", sha256(ROOT / path), digest)
    verify_hashes()

    # Verify every serialized derived field too: the plots must not accept stale
    # or manually edited numbers that evade the scientific recomputations.
    from extract_results import build
    expected, secondary, csv_outputs = build()
    check("Headline JSON equals deterministic sealed-source extraction",
          read("derived/headline_results.json") == expected, True)
    check("Secondary JSON equals deterministic sealed-source extraction",
          read("derived/secondary_results.json") == secondary, True)
    for path, (fields, expected_rows) in csv_outputs.items():
        text_rows = [{k: "" if row[k] is None else str(row[k]) for k in fields} for row in expected_rows]
        check(f"Derived CSV schema and contents: {path}", csv_rows(path) == text_rows, True)
    e1, e2 = csv_rows("derived/e001_per_document.csv"), csv_rows("derived/e002_per_document.csv")
    v2 = csv_rows("derived/e002_factorial_validation.csv")
    ab = csv_rows("derived/e002_ablation_per_document.csv")
    for label, rows, roster in (("E001", e1, read_lines(M1)), ("E002", e2, read_lines(M2))):
        check(f"{label} derived document count", len(rows), 64)
        check(f"{label} document IDs follow the frozen roster",
              [r["document_id"] for r in rows] == [r["document_id"] for r in roster], True)
        check(f"{label} independent document IDs", len({r["document_id"] for r in rows}), 64)
        check(f"{label} split is LOCKED", all(r["split"] == "LOCKED" for r in rows), True)
    check("E002 validation document count", len(v2), 32)
    check("E002 validation/test documents disjoint",
          not ({r["document_id"] for r in v2} & {r["document_id"] for r in e2}), True)

    # Result-stage cross-checks do not replace observation-level recomputation.
    canonical1 = read(f"{A1}/verdict/CANONICAL_4K_RESULT.json")
    table1 = read(f"{A1}/diagnostics/required_tables.json")
    by_condition = {r["condition"]: r for r in table1["locked_fidelity"]}
    for condition, key in C1.items():
        check(f"E001 {condition} aggregate-table agreement", by_condition[condition]["mean_nll"], r1[key])
        check(f"E001 {condition} frozen-result agreement", canonical1[key], r1[key])
    for key in ("full_vs_kv_bootstrap_ci", "full_vs_shuffled_bootstrap_ci"):
        check(f"E001 {key} frozen-result agreement", canonical1[key], r1[key])
    check("E001 full vs KV arithmetic", r1["kv_only_nll"] - r1["full_translated_nll"], r1["full_vs_kv_delta"])
    check("E001 wrong-donor arithmetic", r1["full_shuffled_nll"] - r1["full_translated_nll"], r1["full_vs_shuffled_delta"])
    check("E001 full excess NLL arithmetic", r1["full_translated_nll"] - r1["native_9b_nll"], r1["full_translated_delta_nll"])
    check("E001 TQR from sealed means", (r1["source_4b_nll"] - r1["full_translated_nll"]) /
          (r1["source_4b_nll"] - r1["native_9b_nll"]), r1["tqr"])
    for name, condition in (("ncr_kv", "kv_only_nll"), ("ncr_full", "full_translated_nll")):
        check(f"E001 {name}: derived ratio of sealed means",
              (r1["empty_9b_nll"] - r1[condition]) / (r1["empty_9b_nll"] - r1["native_9b_nll"]),
              derived["E001"]["comparisons"][name]["estimate"])
    check("E001 stronger full-state fidelity gate remains failed", r1["full_translated_delta_nll"] > 0.20, True)
    check("E001 stronger TQR gate remains failed", r1["tqr"] < 0.75, True)
    check("E001 canonical verdict", r1["canonical_verdict"], "RECURRENT_STATE_TRANSLATABLE")

    seeds = [read(f"{root}/PREREGISTRATION.json")["seeds"]["bootstrap"] for root in (E1, E2)]
    f1 = sealed_function(f"{E1}/analysis/metrics.py", ["paired_bootstrap_mean_ci"])
    f2 = sealed_function(f"{E2}/analysis/statistics.py",
                         ["bootstrap_mean_ci", "bootstrap_statistic_ci", "factorial_document_contrasts"])
    if "kv_only_nll" in e1[0]:
        for condition, key in C1.items():
            check(f"E001 {condition} observed mean", col(e1, key).mean(), r1[key])
        for left, right, key in (("kv_only_nll", "full_translated_nll", "full_vs_kv"),
                                  ("full_shuffled_nll", "full_translated_nll", "full_vs_shuffled")):
            values = col(e1, left) - col(e1, right)
            check(f"E001 {key} mean", values.mean(), r1[f"{key}_delta"])
            check(f"E001 {key} bootstrap CI", bootstrap(values, seeds[0]), r1[f"{key}_bootstrap_ci"])
        check("E001 all 64 documents improve over KV", int((col(e1, "kv_only_nll") > col(e1, "full_translated_nll")).sum()), 64)
        fractions = [float(r["improvement_fraction"]) for r in e1 if r["improvement_fraction"] != ""]
        check("E001 mean document improvement fraction", np.mean(fractions), r1["full_vs_kv_improvement_fraction"])
        check("E001 median document improvement fraction", np.median(fractions), r1["full_vs_kv_median_improvement_fraction"])

        improvement = col(e1, "kv_only_nll") - col(e1, "full_translated_nll")
        denominator = col(e1, "kv_only_nll") - col(e1, "native_9b_nll")
        valid = denominator > 0
        require(valid.any(), "No valid E001 improvement denominator")
        computed_fractions = improvement[valid] / denominator[valid]
        check("E001 independently recomputed document fraction mean", computed_fractions.mean(),
              r1["full_vs_kv_improvement_fraction"])
        check("E001 independently recomputed document fraction median", np.median(computed_fractions),
              r1["full_vs_kv_median_improvement_fraction"])
        source_gap = col(e1, "source_4b_nll") - col(e1, "native_9b_nll")
        valid = source_gap > 0
        check("E001 document TQR exclusion count", int((~valid).sum()), r1["tqr_exclusion_count"])
        check("E001 mean document TQR", np.mean(
            (col(e1, "source_4b_nll")[valid] - col(e1, "full_translated_nll")[valid]) / source_gap[valid]),
            r1["mean_document_tqr"])
        compact_path = f"{A1}/locked/derived_statistics.json"
        if (ROOT / compact_path).exists():
            stats = read(compact_path)
            for left, right, prefix in (
                ("empty_9b_nll", "kv_only_nll", "kv_vs_empty"),
                ("empty_9b_nll", "full_translated_nll", "full_vs_empty"),
            ):
                values = col(e1, left) - col(e1, right)
                check(f"E001 {prefix} observed mean", values.mean(), stats[f"{prefix}_mean_nll_improvement"])
                check(f"E001 {prefix} observed CI", bootstrap(values, seeds[0]), stats[f"{prefix}_bootstrap_ci"])

    else:
        blocked("E001 observed means, KV/empty and wrong-donor CIs, 64/64, document ratios",
                f"{A1}/locked/raw_evidence.jsonl or sealed derived_statistics.json")

    # Native and corrected scores are recovered from six independent ablation identities.
    raw_ab = read_lines(AB)
    for index, row in enumerate(raw_ab):
        for outcome in row["outcomes"].values():
            require(abs(outcome["nll"] - outcome["delta_nll_to_native"] - float(e2[index]["native_9b_nll"])) < 1e-12,
                    "Native ablation identity does not match CSV")
            require(abs(outcome["nll"] - outcome["impact_vs_joint_corrected"] - float(e2[index]["joint_corrected_nll"])) < 1e-12,
                    "Corrected ablation identity does not match CSV")
    check("E002 six ablation identities agree with every derived row", True, True)
    for key in ("native_9b_nll", "joint_corrected_nll", "base_state_nll"):
        check(f"E002 {key} document mean", col(e2, key).mean(), r2[key])
    native, corrected, base = (col(e2, k) for k in ("native_9b_nll", "joint_corrected_nll", "base_state_nll"))
    improvement = base - corrected
    check("E002 corrected vs base mean", improvement.mean(), r2["base_vs_corrected_improvement"]["mean_difference"])
    check("E002 corrected vs base median", np.median(improvement), r2["base_vs_corrected_improvement"]["median_difference"])
    check("E002 corrected vs base bootstrap CI", bootstrap(improvement, seeds[1]),
          r2["base_vs_corrected_improvement"]["bootstrap_ci"])
    check("E001 bootstrap implementation cross-check on available vector",
          bootstrap(improvement, seeds[0]), f1["paired_bootstrap_mean_ci"](improvement, seed=seeds[0]))
    check("E002 bootstrap implementation cross-check",
          bootstrap(improvement, seeds[1]), f2["bootstrap_mean_ci"](improvement))
    check("E002 corrected excess NLL", np.mean(corrected - native), r2["corrected_delta_nll"])
    check("E002 remaining gap reduction", (base.mean() - corrected.mean()) / (base.mean() - native.mean()),
          r2["remaining_gap_reduction"])
    ratio_rows = np.column_stack([base, corrected, native])
    check("E002 remaining gap bootstrap CI", ratio_bootstrap(ratio_rows, seeds[1]), r2["remaining_gap_bootstrap_ci"])
    for key, actual in (
        ("corrected_vs_source_delta", corrected.mean() - r2["source_4b_nll"]),
        ("corrected_vs_shuffled_delta", corrected.mean() - r2["joint_shuffled_nll"]),
        ("native_context_recovery", (r2["empty_9b_nll"] - corrected.mean()) / (r2["empty_9b_nll"] - native.mean())),
        ("tqr", (r2["source_4b_nll"] - corrected.mean()) / (r2["source_4b_nll"] - native.mean())),
    ):
        check(f"E002 {key} using sealed baseline means", actual, r2[key])
    if "source_4b_nll" in e2[0]:
        for key in ("source_4b_nll", "empty_9b_nll", "joint_shuffled_nll"):
            check(f"E002 {key} observed mean", col(e2, key).mean(), r2[key])
        check("E002 corrected vs source bootstrap CI", bootstrap(corrected - col(e2, "source_4b_nll"), seeds[1]),
              r2["corrected_vs_source_bootstrap_ci"])
        check("E002 corrected vs wrong donor bootstrap CI", bootstrap(corrected - col(e2, "joint_shuffled_nll"), seeds[1]),
              r2["corrected_vs_shuffled_bootstrap_ci"])
    else:
        blocked("E002 observed source/empty/wrong-donor means and paired control CIs",
                f"{A2}/locked/raw_evidence.jsonl or sealed locked_metrics.json")

    for split, rows in (("locked", e2), ("validation", v2)):
        matrix = np.column_stack([col(rows, f"{c.lower()}_nll") for c in CELLS])
        values = contrasts(matrix - col(rows, "native_9b_nll")[:, None])
        for index, cell in enumerate(CELLS):
            sealed = r2[f"factorial_{split}_metrics"][cell]
            key = "nll" if split == "locked" else "mean_nll"
            check(f"E002 {split} {cell} mean", matrix[:, index].mean(), sealed[key])
        for name, vector in values.items():
            sealed = r2["factorial_interaction_metrics"][split]["effects"][name]
            check(f"E002 {split} {name} contrast", vector.mean(), sealed["estimate"])
            check(f"E002 {split} {name} document effects", vector, sealed["document_effects"])
            check(f"E002 {split} {name} bootstrap CI", bootstrap(vector, seeds[1]), sealed["bootstrap_ci"])
        reference = f2["factorial_document_contrasts"](dict(zip(CELLS, matrix[0])))
        for name in reference:
            check(f"E002 {split} {name} implementation cross-check", contrasts(matrix)[name][0], reference[name])

    ab_summary = read(f"{A2}/diagnostics/postverdict_ablations.json")
    for name, sealed in ab_summary["ablations"].items():
        rows = [r for r in ab if r["condition"] == name]
        check(f"E002 {name} document count", len(rows), 64)
        impact = col(rows, "impact_vs_joint_corrected")
        check(f"E002 {name} impact mean", impact.mean(), sealed["mean_impact_vs_joint_corrected"])
        check(f"E002 {name} impact CI", bootstrap(impact, seeds[1]), sealed["impact_bootstrap_ci"])
    check("E002 corrected remains above native", corrected.mean() > native.mean(), True)
    check("E002 near-native excess NLL gate remains failed", r2["corrected_delta_nll"] > 0.05, True)
    check("E002 near-native NCR gate remains failed", r2["native_context_recovery"] < 0.95, True)
    check("E002 near-native agreement gate remains failed", r2["top1_agreement_native"] < 0.90, True)
    check("E002 canonical verdict", r2["canonical_verdict"], "FULL_STATE_HANDOFF")
    for label, result in (("E001", r1), ("E002", r2)):
        check(f"{label} conditional 16K branch not run", result["long_test_run"], False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--available-only", action="store_true",
                        help="Check the explicitly incomplete public subset; never certify complete reproduction.")
    args = parser.parse_args()
    try:
        verify()
    except (OSError, ValueError, KeyError, TypeError) as error:
        FAILURES.append(str(error))
    for error in FAILURES:
        print(f"FAIL: {error}")
    if FAILURES:
        print(f"NUMERICAL/ARTIFACT VERIFICATION FAILED ({len(FAILURES)} failures)")
        return 1
    if BLOCKED:
        for item in BLOCKED:
            print(f"BLOCKED: {item}")
        print(f"{COUNT} AVAILABLE CHECKS PASSED; COMPLETE HEADLINE VERIFICATION BLOCKED")
        return 0 if args.available_only else 2
    print(f"ALL CHECKS PASSED ({COUNT} checks)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
