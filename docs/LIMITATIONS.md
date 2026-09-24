# Limitations

- **One directed route.** Evidence covers Qwen3.5-4B-Base → Qwen3.5-9B-Base,
  two same-family siblings with matching persistent-state geometry. There is
  no cross-family or arbitrary-architecture compatibility result.
- **Teacher forcing.** The primary endpoint scores 64 observed targets after
  a 4,096-token prefix. No on-policy free-running generation, downstream
  task, or long conversation evaluation is included.
- **A bundled intervention.** E001 adds recurrent matrices, convolution
  history, and previous-state initialization semantics together. Its contrast
  does not identify the causal contribution of recurrent matrices alone.
- **Direct reuse matters.** Translated KV plus directly copied GDN state
  outperforms the tested fully learned GDN mapping. E002's factorial compares
  direct and translated components; it does not test their absence.
- **Pair-specific target supervision.** E002's correction is fitted for this
  source/target pair using target state and behavioral information. Its
  434,176 parameters are additional to the existing KV mapping.
- **Remaining target gap.** E001 fails its full-handoff fidelity and TQR
  gates. E002 passes its intermediate full-handoff gate but fails all three
  near-native criteria. Corrected NLL remains 0.076367 nats/token above native
  9B; no native equivalence is claimed.
- **Different corpora.** E001 uses held-out PG19 documents and E002 fresh
  FineWeb-Edu documents. Cross-experiment aggregate differences are not a
  controlled estimate of the correction effect.
- **No 16K result.** The conditional longer-context branch is not run in
  either experiment. E002's 256-token state tracking is a secondary diagnostic,
  not a 16K or free-generation evaluation.
- **No production speed or cost result.** Recorded stage timings come from
  a scientific prototype. They do not demonstrate end-to-end latency,
  throughput, cost, energy savings, or production readiness.
- **Artifact coverage.** Headline LOCKED observations, derived statistics,
  correction-selection records, and the factorial evidence used for the
  published component analysis are included and verified. A from-scratch
  GPU rerun is not self-contained: model checkpoints, the enclosing runtime
  package, and gitignored tensors including the selected correction weights
  are outside this checkout. No independent audit JSON and no manuscript
  feasibility audit are included. Do not treat those gaps as missing
  headline numbers.

These boundaries apply even when an original experiment's internal verdict
uses the label `FULL_STATE_HANDOFF`.
