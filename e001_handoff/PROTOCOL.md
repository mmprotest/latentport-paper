# LatentPort E001: Handoff

## Question

Can the complete live inference state of `Qwen/Qwen3.5-4B-Base` be translated into a state from which `Qwen/Qwen3.5-9B-Base` continues held-out text without the 9B model processing the skipped prefix?

## Frozen hypothesis

> **H1: A low-capacity translation of Qwen3.5-4B's complete hybrid inference state into Qwen3.5-9B's state space will allow Qwen3.5-9B to continue held-out contexts without re-prefilling them, with substantially higher target-model fidelity than translating the attention KV cache alone.**

The scientific claim concerns target-model behavior, not tensor reconstruction alone.

## Scientific sequence

Observation → Hypothesis → Implementation → Benchmark → Inspection → Redesign → Next Hypothesis

1. Verify the exact installed model implementations and cache layouts.
2. Pass complete same-model state capture/restore for both models.
3. Run a separate direct-copy smoke test.
4. Freeze public corpora and deterministic split manifests.
5. Collect paired source/target states on identical token IDs.
6. Fit only preregistered ridge and bilinear/ridge translators.
7. Select regularization/normalization on VALIDATION only.
8. Freeze translators, hashes, runtime, and manifests.
9. Run the cross-domain 4K LOCKED evaluation exactly once.
10. Freeze the canonical verdict before diagnostics.
11. Unlock only the two preregistered oracle diagnostics if FULL_TRANSLATED fails its primary gate.
12. Run 16K LONG only if the 4K result reaches FULL_STATE_HANDOFF.
13. Preserve all raw and negative results, generate the report, and choose exactly one next hypothesis.

## Fixed models and runtime constraints

- Source: `Qwen/Qwen3.5-4B-Base`
- Target: `Qwen/Qwen3.5-9B-Base`
- Official revisions only; exact commit hashes are frozen before execution.
- BF16 canonical weights and computation except FP32 metric accumulation and translator fitting.
- No fine-tuning, LoRA, weight edits, quantization, MTP, speculative decoding, or instruct checkpoints.
- Models may be resident sequentially. Serialization timing is separate from fidelity.
- Canonical attention implementation is explicitly frozen after runtime inspection.

## Architecture gate

The expected language stack is 32 layers in `linear, linear, linear, full` repetition: 24 Gated DeltaNet layers and 8 full-attention layers. Every persistent recurrent, convolution, KV, and bookkeeping tensor is inspected from runtime code and live cache objects. Any corresponding persistent-memory geometry mismatch stops E001 with `INCONCLUSIVE_ARCHITECTURE_MISMATCH`; no reshaping, cropping, padding, tiling, or invented alignment is permitted.

## Phase A implementation gate

Each model is tested independently on 16 deterministic contexts spanning 256, 512, 1024, and 2048 tokens. Complete live state is deeply cloned into a fresh equivalent cache. Uninterrupted and restored branches process the same bridge token and the same 64-token teacher-forced continuation.

Required for each model:

- top-1 agreement: 100%
- minimum next-token full-logit cosine: at least 0.999999
- no NaN or Inf
- maximum absolute logit difference within the tolerance frozen from pre-science implementation smoke testing

Cross-model work cannot begin until both models pass.

## Direct-copy smoke

Eight separate 1024-token contexts are used for `DIRECT_GDN`, `DIRECT_KV`, and `DIRECT_FULL`. Results are diagnostic only and cannot change H1 or the canonical learned-translation protocol.

## Canonical data

- FIT: 128 web documents, 1024-token prefixes, states captured every 128 tokens.
- VALIDATION: 32 disjoint web documents, 1024-token prefixes.
- LOCKED: 64 disjoint long-form documents, 4096-token prefixes, different corpus/domain.
- LONG: 16 fresh long-form documents, 16384-token prefixes, conditional only.
- Each example reserves one bridge token and 64 continuation tokens beyond the prefix.
- Documents are selected only by deterministic SHA-256 ordering and length eligibility.
- Token IDs are produced once by the frozen tokenizer and must be identical for source and target.

## Translators

### Full-attention KV

Independent ridge maps are fit per layer, KV head, and K/V role. Position sampling, normalization candidates, and scale-relative ridge grid are frozen in `PREREGISTRATION.json`. Runtime inspection determines whether cached K is RoPE-rotated; when mathematically invertible it is de-rotated, translated, and re-rotated at the same positions.

### Gated DeltaNet recurrent state

For each corresponding layer and head:

`S9_hat = mu9 + A (S4 - mu4) B^T`

`A` and `B` are fit with the preregistered ridge-regularized alternating/separable least-squares procedure. The map remains bilinear. No nonlinear rescue is allowed.

### Gated DeltaNet convolution state

Independent linear ridge maps respect the exact runtime packing of Q, K, and V components. Components are not mixed unless runtime semantics show they are a single packed channel axis.

Structural positions, lengths, offsets, and layer metadata are constructed deterministically; they are never learned.

## Bridge and scoring protocol

The source processes exactly the prefix. Its state is captured and translated. A fresh target runtime receives the translated state and does not process any prefix token. It processes exactly one real bridge token, then is teacher-forced on the following 64 held-out tokens. Token accounting is recorded per condition.

Canonical LOCKED conditions are exactly:

`NATIVE_9B`, `SOURCE_4B`, `EMPTY_9B`, `KV_ONLY`, `KV_GDN_DIRECT`, `FULL_TRANSLATED`, `FULL_SHUFFLED`.

`FULL_SHUFFLED` uses a deterministic no-fixed-point rotation within equal prefix length. Recipient bridge/future tokens and structural metadata remain unchanged.

## Endpoints and inference

At every continuation position record NLL, full-logit-derived agreement metrics, top-1, top-5, entropy, Jensen-Shannon divergence, and numerically stable KL to native 9B. The primary endpoint is document-level excess NLL:

`DeltaNLL(c) = NLL(c) - NLL(NATIVE_9B)`.

Target-quality retention is:

`TQR = (NLL_SOURCE_4B - NLL_FULL_TRANSLATED) / (NLL_SOURCE_4B - NLL_NATIVE_9B)`.

Documents with a non-positive TQR denominator are excluded only from document-level TQR and counted. Aggregate NLL is always reported.

The primary comparison is KV_ONLY versus FULL_TRANSLATED using 10,000 paired document bootstrap resamples at the frozen seed. Tokens are not treated as independent samples. Position bins are 1, 2–4, 5–16, and 17–64.

## Verdict ladder

Exactly one verdict is assigned using the frozen rules in `PREREGISTRATION.json`: `NO_TRANSLATABLE_STATE`, `KV_ONLY_TRANSFER`, `RECURRENT_STATE_TRANSLATABLE`, `FULL_STATE_HANDOFF`, or `STRONG_FULL_STATE_HANDOFF`. `INCONCLUSIVE_IMPLEMENTATION`, `INCONCLUSIVE_ARCHITECTURE_MISMATCH`, and `INCONCLUSIVE_TOKENIZATION_MISMATCH` are reserved for invalid execution, never poor translator performance.

## Locked-analysis boundary

No LOCKED document, token, state, or metric may be read by fitting, selection, normalization, or regularization code. Translator and manifest hashes are verified immediately before LOCKED. No model pair, corpus, layer correspondence, teacher-forced horizon, bridge logic, threshold, or translator class may change after lock. Canonical RESULT is written before narrative interpretation.

## Conditional branches

If FULL_TRANSLATED fails its primary gate, only `ORACLE_KV_TRANSLATED_GDN` and `TRANSLATED_KV_ORACLE_GDN` unlock, after verdict freeze. They cannot change the verdict. LONG runs unchanged only after at least FULL_STATE_HANDOFF.

## Claim limits

All conclusions follow the exact allowed wording and next-hypothesis decision tree frozen in `PREREGISTRATION.json`. E001 ends after one next hypothesis is selected; E002 is not started.
