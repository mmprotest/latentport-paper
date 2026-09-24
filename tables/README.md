# Reproducible tables

Run `python analysis/reproduce_tables.py` after extraction.

- `e001_summary.csv`: seven condition means and the E001 comparisons.
- `e002_summary.csv`: primary and factorial condition means and the E002 comparisons.
- `e002_factorial_effects.csv`: all seven contrasts on validation and LOCKED data.
- `e002_correction_candidates.csv`: the eight correction candidates in `correction_selection.json`, including which candidate was selected.

Every headline row includes the point estimate, available 95% CI, units,
sample count, independent unit, split, source artifact, JSON path, estimate
method, and CI status. Paired-effect intervals are not repurposed as
uncertainty on individual condition means.

`recomputed_from_included_observations` means the verifier recomputes that
paired interval from the included document rows.
`recomputed_document_mean_bootstrap` is a new document-mean bootstrap written
by the extractor. It is not a sealed interval for the condition mean.
`sealed_interval_observations_included` means the document rows are present
and the published interval is copied from the sealed metrics; that particular
interval is not one of the contrasts the verifier recomputes.
`not_reported` means the sealed result did not supply an interval. A blank
CI is not zero uncertainty.

Full precision is preserved. Rounding in the README follows the paper.
