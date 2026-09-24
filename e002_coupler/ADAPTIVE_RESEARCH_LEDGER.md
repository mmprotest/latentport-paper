# LatentPort E002 Adaptive Research Ledger

## 2026-09-01 — E001 intake

- Accepted E001 only as sealed prior evidence.
- Confirmed canonical verdict `RECURRENT_STATE_TRANSLATABLE` and the exact model
  revisions recorded in E001.
- Did not parse or use E001 LOCKED per-document evidence.
- Verified the E001 frozen manifest, runtime versions, model shards, tokenizer,
  translators, state schemas, and source FIT/VALIDATION arrays before creating
  E002 scientific artifacts.

## 2026-09-01 — zero-initialized low-rank parameterization

The literal product of two trainable factors initialized to zero has zero
gradient for both factors. To preserve the protocol's exact identity start
without silently creating a dead correction, E002 freezes a deterministic
orthonormal input factor and trains only a zero-initialized output factor for
each rank-constrained product. The resulting residual is exactly zero at
initialization, has rank at most `r`, has non-zero first-order gradients, and
is not a replacement mapping. This resolution was frozen before any E002 model
outcome was observed.

## Locked decisions

- Rank grid: `{2, 4}`.
- Identity lambda grid: `{0, 1e-4, 1e-3, 1e-2}`.
- Learning-rate grid: `{1e-3}`.
- Full-vocabulary KL; no adaptive top-k fallback.
- Maximum three epochs, deterministic order, validation-only selection.
- Pairwise interaction materiality threshold: `0.02` nats/token.

Further entries must describe execution facts only. They may not change E002
after LOCKED begins.

## 2026-09-01 — pre-LOCKED execution facts

- The fresh 32-document factorial manifest hash is
  `7c94d12f0e528d7c8206eb6780ca52d79ce16bd465e58bf344e95406975ce6f2`.
- Validation selected `TDD` under the frozen 0.01-nat lower-complexity tie
  rule. `TDT` had the lowest raw mean DeltaNLL, but its 0.001151-nat advantage
  over `TDD` was inside the tie tolerance.
- The preregistered grid selected rank 4 and identity lambda `1e-2` at epoch
  3, with validation behavioral KL `0.16191834025084972`.
- The selected correction has 434,176 trainable parameters, 1,901,432
  serialized bytes, and correction hash
  `e93a15a21b6eba37ec794d94c45780059a3653f9466c26db1b20960419260db3`.
- The fresh 64-document LOCKED manifest hash is
  `ac0b4db6065f5960e0d2bd7cf321f716fb16b9ae8026b12528c45f619398e524`.
- The conditional 16-document LONG manifest hash is
  `7bbe94629ac8d2363c145e4d278e6c8a085486107ae9506ecaabc4a1b334bc2b`.
- Split selection used no model outcomes; LOCKED and LONG have zero overlap
  with each other and with all frozen exclusion sets.
