"""Write compact, source-addressable summaries from derived data only."""
from __future__ import annotations

from common import ROOT, csv_rows, parse, write_csv


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
                    "ci_status": ("recomputed_from_included_observations" if experiment == "E002"
                                  and name in ("base_minus_corrected", "remaining_gap_reduction")
                                  else "sealed_interval_observations_missing") if ci else "not_reported",
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


if __name__ == "__main__":
    main()
