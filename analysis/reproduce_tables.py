"""Write compact, source-addressable summaries from derived data only."""
from __future__ import annotations

from common import ROOT, csv_rows, parse, write_csv


RECOMPUTED_CONTRASTS = {
    "full_vs_kv", "full_vs_wrong_donor", "kv_vs_empty",
    "base_minus_corrected", "corrected_minus_source",
    "corrected_minus_wrong_donor", "remaining_gap_reduction",
}


def ci_status(name, metric):
    if not metric["ci95"]:
        return "not_reported"
    source = metric.get("ci_source") or ""
    if source.startswith("new document-mean bootstrap"):
        return "recomputed_document_mean_bootstrap"
    if name in RECOMPUTED_CONTRASTS and metric.get("observation_artifact"):
        return "recomputed_from_included_observations"
    if metric.get("observation_artifact"):
        return "sealed_interval_observations_included"
    return "sealed_interval_observations_missing"


def main():
    h = parse((ROOT / "derived/headline_results.json").read_text(encoding="utf-8"))
    for experiment in ("E001", "E002"):
        summary = h[experiment]
        rows = []
        for group in ("conditions", "comparisons"):
            for name, metric in summary[group].items():
                dimensionless = ("ncr" in name or "tqr" in name or "fraction" in name
                                 or name == "remaining_gap_reduction")
                ci = metric["ci95"]
                rows.append({
                    "metric": f"{name}_mean_nll" if group == "conditions" else name,
                    "point_estimate": metric["estimate"],
                    "ci95_low": ci[0] if ci else None, "ci95_high": ci[1] if ci else None,
                    "units": "ratio" if dimensionless else "nats/token",
                    "sample_count": summary["documents"], "independent_unit": "document",
                    "split": "LOCKED", "source_artifact": metric["source_artifact"],
                    "json_path": metric["json_path"], "estimate_method": metric["method"],
                    "ci_source": metric["ci_source"],
                    "ci_status": ci_status(name, metric),
                })
        write_csv(f"tables/{experiment.lower()}_summary.csv", list(rows[0]), rows)
        print(f"WROTE tables/{experiment.lower()}_summary.csv: {len(rows)} metrics")
    secondary = parse((ROOT / "derived/secondary_results.json").read_text(encoding="utf-8"))
    rows = []
    f = secondary["e002_factorial"]
    for split in ("validation", "locked"):
        for name, metric in f["effects"][split]["effects"].items():
            rows.append({
                "metric": name, "split": split.upper(), "point_estimate": metric["estimate"],
                "ci95_low": metric["bootstrap_ci"][0], "ci95_high": metric["bootstrap_ci"][1],
                "units": "nats/token", "sample_count": f["effects"][split]["documents"],
                "independent_unit": "document", "source_artifact": f["source_artifact"],
                "json_path": f"factorial_interaction_metrics.{split}.effects.{name}",
            })
    write_csv("tables/e002_factorial_effects.csv", list(rows[0]), rows)
    print("WROTE tables/e002_factorial_effects.csv")
    selection = secondary["e002_selection"]
    chosen = selection["selected_candidate"]
    candidates = []
    for item in selection["candidate_grid"]:
        candidates.append({
            "candidate": item["candidate"],
            "rank": item["rank"],
            "identity_lambda": item["identity_lambda"],
            "parameter_count": item["parameter_count"],
            "best_epoch": item["best_epoch"],
            "best_validation_behavior_kl": item["best_validation_behavior_kl"],
            "selected": item["candidate"] == chosen,
            "source_artifact": selection["source_artifact"],
        })
    write_csv("tables/e002_correction_candidates.csv", list(candidates[0]), candidates)
    print(f"WROTE tables/e002_correction_candidates.csv: {len(candidates)} candidates")


if __name__ == "__main__":
    main()
