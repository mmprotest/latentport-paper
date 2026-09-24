# Data dictionary

NLL is the mean negative natural-log probability assigned to observed next
tokens, in **nats/token**; lower is better. The document is the independent
unit. Conditions are paired repeated measures, not independent groups.
Each primary document has 64 scored targets after a 4,096-token prefix and
one real bridge token. E002 secondary state tracking extends to 256 inputs.

## Common CSV fields

| Column | Meaning |
|---|---|
| `document_id` | Frozen corpus document identifier; E001 PG19 URL or E002 corpus UUID |
| `split` | `LOCKED`: untouched primary test; `VALIDATION_FACTORIAL`: E002 component selection; `LOCKED_POSTVERDICT`: subsequent diagnostics on the same test documents |
| `document_index` | Zero-based position in the frozen source ordering; never an independent observation itself |
| `source_artifact` | Repository-relative source file(s); semicolon separates multiple files |
| `observation_status` | `raw_observations` when the row comes from LOCKED raw evidence. Empty ratio fields mean a nonpositive denominator, not a missing document |

## E001: `e001_per_document.csv`

Contains **64 LOCKED documents** extracted from
`e001_handoff/artifacts/attempt_001/locked/raw_evidence.jsonl`.
Missing ratios use an empty CSV field when their denominator is nonpositive.

| NLL column | Original condition | Meaning |
|---|---|---|
| `native_9b_nll` | NATIVE_9B | Native target reads its prefix |
| `source_4b_nll` | SOURCE_4B | Continued source baseline |
| `empty_9b_nll` | EMPTY_9B | Target starts without the historical prefix state |
| `kv_only_nll` | KV_ONLY | Translated attention KV; no transferred GDN state package |
| `kv_gdn_direct_nll` | KV_GDN_DIRECT | Translated KV, directly copied recurrent/convolution state |
| `full_translated_nll` | FULL_TRANSLATED | Learned KV, recurrent, and convolution maps |
| `full_shuffled_nll` | FULL_SHUFFLED | Another document's complete translated package |

## E002: `e002_per_document.csv`

Contains 64 rows from `e002_coupler/artifacts/attempt_001/locked/raw_evidence.jsonl`.
The ablation identities are a cross-check, not the only source of native and
corrected scores. The first three identity fields above accompany:

| Column | Meaning |
|---|---|
| `native_9b_nll` | Native NLL from the LOCKED raw record; identical to the ablation identity `nll - delta_nll_to_native` |
| `joint_corrected_nll` | Corrected NLL from the LOCKED raw record; identical to `nll - impact_vs_joint_corrected` |
| `source_4b_nll`, `empty_9b_nll`, `joint_shuffled_nll` | Continued 4B, empty target, and wrong-donor NLL from the same raw records |
| `ddd_nll` … `ttt_nll` | Eight factorial condition NLLs |
| `base_state_nll` | Selected TDD value, identical to `tdd_nll` |
| `ncr`, `tqr` | Per-document ratios. Empty when the denominator is nonpositive |
| `observation_status`, `source_artifact` | `raw_observations` and the LOCKED raw path |

Factorial letters are in **KV / recurrent / convolution** order.
**D** means direct copy; **T** means the frozen learned translation. Both retain
a state component; D does not mean deletion. `BASE_STATE` is TDD.
`JOINT_CORRECTED` applies the selected correction to TDD.
`JOINT_SHUFFLED` installs another document's corrected package.

Continued-4B, empty-target, wrong-donor, per-document NCR, and per-document TQR
are included from the LOCKED raw evidence. Aggregate means and ratios are
also retained in `headline_results.json`.

## Supplemental CSVs

- `e002_factorial_validation.csv`: 32 independent validation documents,
  native NLL and eight factorial NLLs; directly extracted from validation
  raw evidence. This is not the correction-training validation split.
- `e002_ablation_per_document.csv`: 384 rows = 64 documents × six
  post-verdict ablations. Additional columns: `condition`, `nll`,
  `delta_nll_to_native`, `impact_vs_joint_corrected`. The latter is
  ablated minus corrected NLL; positive means removal worsens continuation.
  The independent sample count remains 64, not 384.

## Derived metrics

Let N, E, S, K, F, B, and C denote native 9B, empty 9B, source 4B,
KV-only, full translated, uncorrected base, and corrected NLL respectively.

| Metric | Definition | Interpretation |
|---|---|---|
| Excess NLL / DeltaNLL | condition − N | Remaining loss above native target |
| E001 improvement fraction | (K − F) / (K − N), documentwise for K − N > 0 | Reduction of KV-only excess; reported mean 0.796533 and median 0.802222 |
| NCR | (mean E − mean condition) / (mean E − mean N) | Native context benefit recovered, not accuracy; E002 headline 0.917755 |
| TQR | (mean S − mean condition) / (mean S − mean N), denominator > 0 | Target advantage over source retained; E001 −0.413227, E002 0.405755 |
| E002 remaining-gap reduction | (mean B − mean C) / (mean B − mean N) | Fraction of the base's target gap closed; 0.275382 |

E001 NCR in this companion is newly computed aggregate arithmetic, not a
sealed E001 headline field. Do not substitute it for the mean document
improvement fraction. Ratios need not be between 0 and 1.

Headline JSON records point estimates, CI endpoints where present, source
files, JSON paths, methods, and source hashes.
Summary-table CI blanks mean **not reported**, not zero uncertainty.
`sealed_interval_observations_missing` is reserved for a reported interval
whose document rows are absent. The current headline checkout does not use
that status for the primary contrasts.
