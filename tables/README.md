# Reproducible tables

Run `python analysis/reproduce_tables.py` after extraction.

- `e001_summary.csv`: seven sealed condition means and eight comparisons/ratios.
- `e002_summary.csv`: primary/factorial condition means and seven comparisons/ratios.
- `e002_factorial_effects.csv`: all seven contrasts on validation and LOCKED data.

Every headline row includes the point estimate, available 95% CI, units,
sample count, independent unit, split, source artifact, JSON path, estimate
method, and CI status. Mean-NLL intervals are not supplied by the sealed
headline result and are left blank. Paired-effect intervals are not
repurposed as uncertainty on individual condition means.

E001's document outcomes and E002's baseline/control outcomes are missing.
Their reported CIs remain explicitly labeled as unverified from observations.
The E002 correction and remaining-gap CIs do reproduce. Full precision is
preserved; rounding in README is for readability.
