# Claims and evidence

All claims refer to Qwen3.5-4B-Base → Qwen3.5-9B-Base at a 4K prefix.
“Reported” means preserved in a sealed artifact. “Recomputed” means obtained
from included independent document observations or the documented exact
ablation identities. Aggregate agreement is not observation-level replication.

## 1. KV-only transfer gives limited benefit

**Claim:** KV-only recovers a small part of native target context benefit.
**Evidence / experiment:** E001 empty NLL 3.245654 and KV-only 3.115271;
gain 0.130383 nats/token.
**Artifact / metric:** E001 `verdict/RESULT.json`, `empty_9b_nll`,
`kv_only_nll`, `native_9b_nll`; NCR from means is about 0.1202.
**Important limitation:** Aggregate arithmetic is verified. E001's underlying
64 observations and the exact KV-versus-empty CI artifact are missing.

## 2. Adding persistent state materially improves continuation

**Claim:** Adding the translated GDN state package beyond fixed translated
KV improves teacher-forced continuation.
**Evidence / experiment:** E001 full NLL 2.367948; gain over KV 0.747323,
reported paired 95% CI [0.692050, 0.804669].
**Artifact / metric:** E001 `RESULT.json:full_vs_kv_delta` and
`full_vs_kv_bootstrap_ci`.
**Important limitation:** The intervention bundles recurrent matrices,
convolution history, and initialization semantics; it does not isolate
recurrent matrices alone. The paper reports all 64 documents improve, but
this checkout cannot independently check that claim. The full handoff gate
fails. Direct GDN reuse also beats the tested learned GDN maps.

## 3. Wrong-donor control supports source specificity

**Claim:** The useful transferred package carries information about the
corresponding document.
**Evidence / experiment:** E001 wrong-donor minus correct full-state NLL
0.948233, reported CI [0.877881, 1.022084]. E002 corrected minus wrong-donor
NLL −1.188365, reported CI [−1.295645, −1.085469].
**Artifact / metric:** E001 `full_vs_shuffled_*`; E002
`corrected_vs_shuffled_*` in their respective `RESULT.json`.
**Important limitation:** A deterministic donor permutation has no fixed
points: each target receives another document's complete transferred package.
It tests package specificity, not which individual component stores which
information. The exact donor observations/CIs cannot be replayed here.

## 4. E002 correction improves the held-out handoff

**Claim:** A pair-specific correction improves the selected uncorrected TDD base.
**Evidence / experiment:** E002 base 2.018422, corrected 1.989399; improvement
0.029022, 95% CI [0.022828, 0.035413]; remaining-gap reduction 0.275382,
CI [0.224143, 0.338594].
**Artifact / metric:** E002 `factorial_locked_metrics.TDD.document_nll`,
`diagnostics/postverdict_ablation_raw.jsonl`, and
`RESULT.json:base_vs_corrected_improvement`, `remaining_gap_*`.
**Important limitation:** These means and CIs recompute. Corrected/native
document scores are recovered from exact ablation identities. They are not
re-run inference. No material component-interaction mechanism was established
under the registered threshold.

## 5. Corrected E002 handoff remains worse than native 9B

**Claim:** Corrected continuation does not reach native-target fidelity.
**Evidence / experiment:** Corrected minus native NLL is 0.076367. NCR is
0.917755 and top-1 agreement 0.864258; all three near-native thresholds fail
(0.05, 0.95, and 0.90 respectively).
**Artifact / metric:** E002 `RESULT.json:corrected_delta_nll`,
`native_context_recovery`, `top1_agreement_native`.
**Important limitation:** Native and corrected means recompute; NCR uses a
sealed empty-target mean. `FULL_STATE_HANDOFF` is E002's intermediate verdict,
not an equivalence assertion. No equivalence test was passed.

## 6. No free-generation result

**Claim boundary:** The experiments evaluate teacher-forced continuation.
**Evidence / experiment:** Both preregistrations specify 64 observed targets
per document; the paper expressly excludes free-generation equivalence.
**Artifact / metric:** Both `PREREGISTRATION.json:evaluation` sections;
paper scope and limitations.
**Important limitation:** No on-policy generation, conversation, or downstream
task outcome is established.

## 7. No end-to-end production speed result

**Claim boundary:** This repository does not demonstrate a production speedup.
**Evidence / experiment:** Timings are prototype stage measurements; E001's
report explicitly omits evidence serialization from scientific translation timing.
**Artifact / metric:** E001 `REPORT.md` timing section and
`diagnostics/postverdict_analysis.json:timing`; E002 `RESULT.json` timing fields.
**Important limitation:** Stage timings are not a deployed pipeline comparison.
No throughput, cost, energy, or production model-switching claim follows.

## 8. One principal model pair

**Claim boundary:** One-model-pair existence proof; cross-scale transfer within
a closely related hybrid family.
**Evidence / experiment:** Both results specify the same source and target revisions.
**Artifact / metric:** Both `RESULT.json:source_model/source_revision/target_model/target_revision`.
**Important limitation:** One direction, two same-family siblings, exact
persistent-state geometry; no larger-model or other-pair success is included.

## 9. Pair-specific supervised correction

**Claim:** The small learned correction is additional pair-specific training.
**Evidence / experiment:** E002 selects rank 4, 434,176 parameters, identity
penalty 0.01; training uses target-side state/behavior supervision on training
data, with held-out correction validation and fresh LOCKED evaluation.
**Artifact / metric:** E002 `PREREGISTRATION.json:training/correction` and
`RESULT.json:correction_parameter_count/correction_rank/identity_lambda`.
**Important limitation:** “Small correction” excludes the existing KV translator
and does not mean a training-free universal adapter. Candidate-selection
records and selected correction weights are absent from this checkout.

## 10. No general cross-family claim

**Claim boundary:** Nothing here establishes universal state compatibility.
**Evidence / experiment:** No cross-family experimental condition exists in
the two included protocols.
**Artifact / metric:** E001 architecture correspondence and both protocols.
**Important limitation:** Matching shapes permit testing, not a guarantee that
another model can interpret the state. No generalized interface or arbitrary
architecture support is demonstrated.
