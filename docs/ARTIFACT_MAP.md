# Artifact map

All paths below are relative to this repository. Existing packages are named
`e001_handoff/` and `e002_coupler/`; no replacement `E001/` or `E002/`
directories were created.

## Canonical attempts and provenance

| Experiment | Canonical result | Selection / seal |
|---|---|---|
| E001 | `e001_handoff/artifacts/attempt_001/verdict/RESULT.json` | `e001_handoff/artifacts/CURRENT_ATTEMPT.json:attempt` and `e001_handoff/FROZEN_MANIFEST.json:attempt` both specify `attempt_001` |
| E002 | `e002_coupler/artifacts/attempt_001/verdict/RESULT.json` | `e002_coupler/FROZEN_MANIFEST.json:attempt` and `artifacts/attempt_001/verdict/final_verdict.json` |

Both protocols and preregistrations are at the experiment root. Both reports
are `artifacts/attempt_001/REPORT.md`. E001 additionally preserves
`verdict/CANONICAL_4K_RESULT.json`; E002 preserves
`verdict/4k_verdict.json` and `verdict/final_verdict.json`.
The extractor rejects ambiguous attempt directories instead of selecting
the latest filename.

Each root has `FROZEN_MANIFEST.json`; each attempt has
`FINAL_ARTIFACT_MANIFEST.json`. E001's final manifest paths are relative
to its attempt; E002's are relative to its experiment root. These manifests
describe larger historical runs, including state dumps that are not in this
checkout. They are not inventories of every file required for a GPU rerun.
The LOCKED observation files used for the headline statistics are present;
see [MISSING_ARTIFACTS.md](MISSING_ARTIFACTS.md).

[ARTIFACT_INVENTORY.csv](ARTIFACT_INVENTORY.csv) lists the 360 original
non-environment files inspected, including the original LICENSE and PDF,
their sizes and SHA-256 hashes, and Git inclusion status.
No scientific artifact was rewritten.

## Exact headline paths

In the following table, **E1 RESULT** means
`e001_handoff/artifacts/attempt_001/verdict/RESULT.json`; **E2 RESULT** means
`e002_coupler/artifacts/attempt_001/verdict/RESULT.json`.

| Claim / output | Source and JSON path / computation | Check available |
|---|---|---|
| E001 condition means; hero left panel | E1 RESULT: `empty_9b_nll`, `kv_only_nll`, `full_translated_nll`, `native_9b_nll` | Recomputed from `locked/raw_evidence.jsonl`; agrees with `diagnostics/required_tables.json:locked_fidelity[*].mean_nll` |
| E001 continued source / direct GDN / donor means | E1 RESULT: `source_4b_nll`, `kv_gdn_direct_nll`, `full_shuffled_nll` | Aggregate agreement |
| E001 KV improvement over empty | E1 RESULT: `empty_9b_nll - kv_only_nll` | Arithmetic |
| E001 full vs KV gain and 95% CI | E1 RESULT: `full_vs_kv_delta`, `full_vs_kv_bootstrap_ci` | Recomputed from LOCKED document rows; 64/64 documents improve |
| E001 source specificity | E1 RESULT: `full_vs_shuffled_delta`, `full_vs_shuffled_bootstrap_ci` | Recomputed from LOCKED document rows |
| E001 79.7% document-gap reduction | E1 RESULT: `full_vs_kv_improvement_fraction`; median `full_vs_kv_median_improvement_fraction` | Recomputed mean and median of valid document ratios; not a ratio of means |
| E001 NCR in companion | E1 RESULT: `(empty_9b_nll - condition_nll)/(empty_9b_nll - native_9b_nll)` | New arithmetic from sealed means |
| E001 TQR and failed gate | E1 RESULT: `tqr`, `full_translated_delta_nll`; gate in `e001_handoff/analysis/verdict.py:choose_verdict` | Arithmetic; threshold failure visible |
| E002 base mean | E2 RESULT: `factorial_locked_metrics.TDD.document_nll[*]` → mean → `base_state_nll` | Recomputed |
| E002 corrected/native means | Ablation identities below → mean → E2 RESULT: `joint_corrected_nll`, `native_9b_nll` | Recovered and recomputed |
| E002 continued source, empty, donor means | E2 RESULT: `source_4b_nll`, `empty_9b_nll`, `joint_shuffled_nll` | Recomputed from `locked/raw_evidence.jsonl` |
| E002 correction improvement and 95% CI | Paired base minus corrected → E2 RESULT: `base_vs_corrected_improvement.mean_difference/bootstrap_ci` | Recomputed |
| E002 corrected minus continued source | E2 RESULT: `corrected_vs_source_delta`, `corrected_vs_source_bootstrap_ci` | Recomputed paired mean and bootstrap CI |
| E002 corrected minus donor | E2 RESULT: `corrected_vs_shuffled_delta`, `corrected_vs_shuffled_bootstrap_ci` | Recomputed paired mean and bootstrap CI |
| E002 remaining-gap reduction and CI | Paired base/corrected/native observations → E2 RESULT: `remaining_gap_reduction`, `remaining_gap_bootstrap_ci` | Recomputed ratio of means and paired bootstrap |
| E002 NCR / TQR | E2 RESULT: `native_context_recovery`, `tqr`; formulas in DATA_DICTIONARY | Recomputed from included document means, including the previously aggregate-only baselines |
| E002 failed stronger gates | E2 RESULT: `corrected_delta_nll`, `native_context_recovery`, `top1_agreement_native`; `e002_coupler/analysis/verdict.py:determine_4k_verdict` | Threshold failures visible |

Every generated summary-table row carries its exact source artifact and
JSON path; `derived/headline_results.json:source_sha256` records extraction
input hashes. Figures read the derived files, not handwritten numbers.

## Observation files and computations

| Source | Grain / extraction |
|---|---|
| `e001_handoff/data/locked/manifest.jsonl` | 64 roster records; `document_id`, `split`, `split_index`. No outcomes. |
| `e002_coupler/data/locked/manifest.jsonl` | 64 roster records in frozen selection-hash order |
| E2 RESULT: `factorial_locked_metrics.<DDD…TTT>.document_nll` | Eight aligned vectors of 64 observed NLLs; TDD is the base |
| `e002_coupler/artifacts/attempt_001/diagnostics/postverdict_ablation_raw.jsonl` | 64 documents × six `outcomes`; native = `outcomes.<name>.nll - delta_nll_to_native`; corrected = `nll - impact_vs_joint_corrected` |
| `e002_coupler/runtime/postverdict_ablations.py:main` | Exact writer establishing those two algebraic identities |
| `e002_coupler/runtime/locked_run.py:main` and `analysis/locked_analysis.py:_condition_summary` | Manifest-order execution and document-vector order |
| `e002_coupler/artifacts/attempt_001/factorial/raw_evidence.jsonl` | 32 validation documents; native `native_9b.nll`; cells `conditions.<cell>.nll` |
| `e002_coupler/artifacts/attempt_001/factorial/documents/document_000.json` … `document_031.json` | Individual validation results corresponding to the JSONL records |

Factorial contrasts are recomputed from the eight within-document outcomes,
using the exact D/T coding and bootstrap in
`e002_coupler/analysis/statistics.py`. Their reference values are E2 RESULT:
`factorial_interaction_metrics.<validation|locked>.effects.<name>`.
Post-verdict ablation effect references are
`diagnostics/postverdict_ablations.json:ablations.<name>`.

## Paper figure and table coverage

The existing `LatentPort.pdf` has 14 pages and six figures. No LaTeX source
or separate paper plot source is included. Experiment plotting code is
`e001_handoff/analysis/postverdict_artifacts.py` and
`e002_coupler/analysis/generate_figures.py`; it depends on absent original
run artifacts and the old package layout.

| Paper item | Public reproduction / gap |
|---|---|
| Figure 1 | Conceptual schematic remains in the original PDF; no new scientific values |
| Figure 2 | `figures/main_handoff_comparison.*`: primary excess-NLL means; paired CIs reproduced from included observations |
| Figure 3 | `figures/e001_paired_documents.*`: paired PG19 NLLs from E001 LOCKED raw evidence. The paper plots excess NLL; the equality comparison is the same. Original scatter pixels are not treated as observations |
| Figure 4 | `figures/e002_factorial_comparison.*`: full validation/LOCKED factorial; means and document CIs |
| Figure 5 | `figures/e002_complete_comparison.*`: E002 means, including empty, source, and wrong-donor, recomputed from included observations |
| Figure 6 | `figures/state_convergence.*`: E001 `diagnostics/postverdict_analysis.json:state_convergence`; E2 RESULT `state_repair_token_<1|4|16|64|256>` |
| Tables 1–2 | Architecture/metric definitions are preserved in the PDF, schema artifacts, and DATA_DICTIONARY; not newly typeset |
| Tables 3–4, 7–8 | Primary NLL/effect content in `tables/e001_summary.csv` and `e002_summary.csv`; these are compact summaries, not full typeset replicas of every fidelity column |
| Table 5 | `tables/e002_correction_candidates.csv` from `fit/correction_selection.json`. This is the candidate grid, not a typeset copy of the PDF table |
| Table 6 | Correction inventory and magnitude summary remain in E2 RESULT |
| Tables 9–10 | Cell vectors/means and all effects preserved; `tables/e002_factorial_effects.csv` exports estimates and CIs |
| Table 11 | Ablation observations in `derived/e002_ablation_per_document.csv`; all six effects/CIs verified against the sealed ablation summary |
| Table 12 | Sealed E2 RESULT timing fields retained; no end-to-end production interpretation |

“Reproduced figure” means reproducible quantitative content in a new layout,
not a byte-for-byte copy of the PDF artwork. State-convergence curves use
sealed checkpoint means. They do not add uncertainty intervals the sealed
diagnostics do not contain.

## Audit files

There is **no `INDEPENDENT_AUDIT.json`** in either package.
E001 includes `implementation/REPOSITORY_AUDIT.md`,
`final_test_results.json`, and restoration artifacts. E002 includes
`implementation/e001_integrity.json` and `final_integrity_audit.json`.
These record original checks. They do not prove that every full-run tensor
used by those checks is in this checkout. The manuscript's local revision
feasibility audit is also absent. Headline LOCKED observations are present
and are what `analysis/verify_results.py` recomputes.

[PUBLIC_RELEASE_AUDIT.md](PUBLIC_RELEASE_AUDIT.md) is a companion-layer release
check, not a newly invented independent scientific audit.
