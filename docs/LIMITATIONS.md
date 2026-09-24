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
- **Artifact coverage.** The checkout omits original LOCKED observations,
  some training/selection artifacts, and the manuscript-referenced local
  feasibility audit. Independent audit files are not present. Some E002
  outcomes can be recovered exactly; E001 sampling uncertainty and E002
  baseline/control intervals cannot yet be independently reproduced here.

These boundaries apply even when an original experiment's internal verdict
uses the label `FULL_STATE_HANDOFF`.
