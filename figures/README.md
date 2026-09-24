# Reproducible public figures

Run `python analysis/reproduce_figures.py` after extraction. All numerical
inputs come from `derived/`. Each figure is a vector PDF and a 300-dpi PNG,
using matplotlib's default colors and bundled DejaVu Sans. Original figures
inside the evidence folders and the manuscript PDF remain unchanged.

| Output stem | Paper counterpart | Input and interpretation |
|---|---|---|
| `main_handoff_comparison` | Figure 2 | Headline means minus the same split's native mean; reported E001 paired gain CI and recomputed E002 correction CI in captions |
| `e002_factorial_comparison` | Figure 4, Tables 9–10 | Validation and LOCKED factorial observations; document mean excess NLL and newly recomputed 95% document CIs |
| `e002_complete_comparison` | Figure 5 | Sealed E002 means, including the empty/source/wrong-donor aggregate-only baselines |
| `state_convergence` | Figure 6 | Sealed mean recurrent-state error checkpoints; no CIs available |

These reproduce supported quantitative content, not the manuscript's exact
typesetting. Paper Figure 1 is a conceptual schematic retained in the PDF.
**Paper Figure 3 cannot be reproduced**: E001's per-document NLLs are absent.
No points are digitized or invented. The script creates the paired scatter
only if exact original raw evidence or the sealed per-document summary is later restored.

The hero asks how much benefit persists beyond KV-only transfer. It uses
horizontal bars of excess NLL starting at zero, separate panels for the two
corpora, and explicit paired-difference CI text. No per-condition error bar
is inferred from an effect CI. Panel scales differ and are labeled.
The caption also states the incomplete artifact coverage.

The factorial plot shows eight conditions on 32 validation and 64 LOCKED
documents in separate panels with a shared zero-based scale. D/T order is
KV/recurrent/convolution; D means direct copy. Shape and direct labels
identify conditions without relying on color. All figures state
teacher-forced scope or describe teacher-forced checkpoints.

For exact file and JSON paths see [ARTIFACT_MAP.md](../docs/ARTIFACT_MAP.md).
