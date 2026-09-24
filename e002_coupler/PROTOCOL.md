# LatentPort E002: Coupler

## Frozen question

Can a zero-initialized, identity-anchored correction with rank at most four
improve Qwen3.5-4B-Base to Qwen3.5-9B-Base recurrent-state handoff beyond the
strongest direct/translated component mixture, without replaying the target
prefix or changing either model?

## Sealed prior

E001 is immutable prior evidence with canonical verdict
`RECURRENT_STATE_TRANSLATABLE`. Its aggregate NLLs are cited only as motivation.
No E001 LOCKED state, label, document outcome, or per-document metric is an E002
training or selection input.

## Frozen hypothesis

> **H2: Cross-component coupling between translated KV and translated Gated
> DeltaNet state is the main remaining source of target-fidelity loss, and a
> jointly fitted low-rank correction can close the gap without a deep nonlinear
> translator.**

## Design

The persistent state is decomposed into `K` (full-attention KV), `R` (Gated
DeltaNet recurrent matrices), and `C` (Gated DeltaNet convolution memory). Each
component is either direct source state (`D`) or the corresponding frozen E001
translation (`T`). All eight cells (`DDD`, `DDT`, `DTD`, `DTT`, `TDD`, `TDT`,
`TTD`, `TTT`) are evaluated on a fresh 32-document, 4096-token validation set.
The cell with the lowest mean document-level DeltaNLL is selected; cells within
0.01 nats/token of the minimum are tie-broken by fewer translated components
and then lexical condition order.

The selected cell is the immutable identity anchor. The correction is:

```text
corrected_state = BASE_STATE + residual(BASE_STATE)
```

For recurrent matrices the residual is a shared-across-heads rank-`r` left and
right correction. For KV it is a rank-`r` feature correction shared across
sequence positions and heads. To satisfy both exact zero initialization and
non-zero initial gradients, one factor of each low-rank product is a frozen,
seeded orthonormal buffer and the other is a zero-initialized trainable factor.
Convolution state uses zero-initialized per-layer/channel scale residual and
bias, shared across the four convolution positions. No nonlinear mapper is
present.

## Behavioral fit

Correction candidates use E001 FIT (128 documents, 1024-token prefixes) and
E001 VALIDATION (32 documents) only. Full source KV is recaptured because E001
stored sampled KV positions; the sealed E001 last-checkpoint recurrent and
convolution arrays are loaded read-only, equality-checked against recapture,
and installed as the canonical training inputs. Native 9B logits are freshly
precomputed.

The target-model loss is full-vocabulary `KL(P_native_9B || P_handoff)` over
nine positions: the bridge output plus eight teacher-forced continuation
outputs. Native and handoff use the same one-call nine-input continuation
schedule (`bridge + first eight continuation tokens`) during correction
training. This schedule avoids mutating a differentiable cache between forward
calls and is implementation-equivalence tested. Target and source weights are
frozen. No target historical prefix is processed in a handoff branch.

## Locked evaluation

After the correction, hyperparameters, hashes, fresh document IDs, and all
implementation code are frozen, exactly one 64-document 4K LOCKED run evaluates
the canonical conditions and a no-fixed-point corrected-state rotation. The
primary horizon is 64 teacher-forced targets. State repair uses the same
documents and advances native, base, and corrected states to 1, 4, 16, 64, and
256 post-handoff tokens. The document is the statistical unit; paired bootstrap
intervals use 10,000 resamples.

LONG (16 documents at 16K) is run without changes only when the frozen 4K
verdict reaches `NEAR_NATIVE_HANDOFF`. Post-verdict component and depth
ablations run only after the canonical verdict is written.

## Falsification and stopping

Ranks above four, more than two million trainable parameters, MLPs,
Transformers, model changes, target-prefix replay, LOCKED tuning, document
dropping, and post-LOCKED base-state changes are forbidden. Negative results
are retained. E002 stops after its exact decision-tree H3 is recorded.

