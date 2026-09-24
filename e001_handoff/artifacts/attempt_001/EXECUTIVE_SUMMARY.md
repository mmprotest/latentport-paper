# LatentPort E001: Handoff — Executive Summary

## Verdict

`RECURRENT_STATE_TRANSLATABLE`

Qwen3.5-9B did pick up useful, context-specific information from a translated Qwen3.5-4B live inference state without rereading the 4,096-token prefix. The experiment demonstrated a substantial contribution from translated Gated DeltaNet memory beyond translated attention KV, but it did not meet the stronger full-state handoff fidelity gate.

## Core result

| Condition | NLL | DeltaNLL vs native 9B |
|---|---:|---:|
| Native 9B | 2.1609 | 0.0000 |
| Source 4B | 2.3074 | 0.1465 |
| Empty 9B | 3.2457 | 1.0847 |
| KV only | 3.1153 | 0.9544 |
| KV + direct GDN | 2.2913 | 0.1304 |
| Full translated | 2.3679 | 0.2070 |
| Full shuffled | 3.3162 | 1.1553 |

Full translation improved NLL over KV-only by 0.7473 nats/token, with paired-bootstrap 95% CI [0.6921, 0.8047]. Mean and median improvement fractions were 79.7% and 80.2%, well above the frozen 25% recurrent-state threshold.

The shuffled-state control failed as expected: correct state beat shuffled state by 0.9482 nats/token, 95% CI [0.8779, 1.0221]. The transfer was therefore about the actual context rather than a generic late-state effect.

## Why this is not full-state handoff

Full-translated DeltaNLL was 0.2070, narrowly above the 0.20 threshold. More importantly, aggregate TQR was −0.413, far below 0.75, because full handoff NLL was worse than the native 4B baseline on this split. `KV_GDN_DIRECT` also outperformed the learned full translator, indicating that naturally aligned GDN state was useful and that the learned component maps introduced behavioral error.

Target inference partially repaired state mismatch: recurrent normalized error declined from 0.469 after one new token to 0.405 after 64, and convolution error declined from 0.437 to 0.214. It did not converge to native state.

Both same-model restore gates passed exactly enough: 16/16 contexts per model, 100% top-1 agreement, minimum next-logit cosine ≥0.99999988, and zero maximum logit difference.

The conditional 16K test did not run because the 4K result did not reach `FULL_STATE_HANDOFF`. Oracle diagnostics did not unlock because full translation passed the primary full-versus-KV gate.

## Timing

Native 9B 4K prefill averaged 885.2 ms; source 4B prefill averaged 662.6 ms, leaving a 222.6 ms translation budget. Extraction, prototype translation, and installation totaled about 152.4 ms before equivalent bridge work, but this is a Python research prototype—not a production speed claim.

## Exact next hypothesis

> **H2: Cross-component coupling between translated KV and translated Gated DeltaNet state is the main remaining source of target-fidelity loss, and a jointly fitted low-rank correction can close the gap without a deep nonlinear translator.**

See [`REPORT.md`](REPORT.md) and [`verdict/RESULT.json`](verdict/RESULT.json). E001 stops here; E002 was not started.
