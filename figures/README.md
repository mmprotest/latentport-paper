# Reproducible public figures

Run `python analysis/reproduce_figures.py` after extraction. All numerical
inputs come from `derived/`. Each figure is a vector PDF and a 300-dpi PNG,
using matplotlib's default colors and bundled DejaVu Sans. Original figures
inside the evidence folders and the manuscript PDF remain unchanged.

| Output stem | Paper counterpart | Input and interpretation |
|---|---|---|
| `main_handoff_comparison` | Figure 2 | Headline excess NLL. E001 uses empty, KV-only, full translated GDN package, and native 9B. E002 uses continued 4B, TDD base, corrected 9B, and native 9B. Paired-difference CIs are reproduced from included observations |
| `e001_paired_documents` | Figure 3 | One point per PG19 document, KV-only NLL against full translated NLL. All 64 points fall below equality. The paper plots excess NLL; equality is the same comparison |
| `e002_factorial_comparison` | Figure 4, Tables 9–10 | Validation and LOCKED factorial observations; document mean excess NLL and recomputed 95% document intervals |
| `e002_complete_comparison` | Figure 5 | E002 condition means, including empty, continued 4B, and wrong-donor, recomputed from included LOCKED observations |
| `state_convergence` | Figure 6 | Sealed mean recurrent-state error checkpoints; no checkpoint CIs are available |

These reproduce supported quantitative content, not the manuscript's exact
typesetting. Paper Figure 1 is a conceptual schematic retained in the PDF.
No points are digitized or invented.

The hero asks how much benefit persists beyond KV-only transfer. It uses
horizontal bars of excess NLL starting at zero, separate panels for the two
corpora, and explicit paired-difference CI text. No per-condition error bar
is inferred from an effect CI. Panel scales differ and are labeled.
The E001 third bar is the fully translated package (excess NLL 0.2070), not
the direct-GDN condition.

The factorial plot shows eight conditions on 32 validation and 64 LOCKED
documents in separate panels with a shared zero-based scale. D/T order is
KV/recurrent/convolution; D means direct copy. Shape and direct labels
identify conditions without relying on color. All figures state
teacher-forced scope or describe teacher-forced checkpoints.

For exact file and JSON paths see [ARTIFACT_MAP.md](../docs/ARTIFACT_MAP.md).
