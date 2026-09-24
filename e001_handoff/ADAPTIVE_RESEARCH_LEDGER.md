# LatentPort E001 Adaptive Research Ledger

This append-only ledger records observations and implementation decisions. It cannot modify the preregistered canonical experiment after LOCKED begins. Post-verdict inspection may motivate exactly one next hypothesis but cannot alter E001.

## 2026-08-31 — Observation

- No prior `experiments/latentport/e001_handoff` directory or scientific evidence existed; `attempt_001` is therefore the first attempt.
- The repository already had extensive unrelated modified files. E001 is isolated under a new path and those changes are preserved.
- The required RTX 5090 is present with 32,607 MiB total VRAM; driver 591.86 advertises CUDA 13.1.
- The repository root `.venv` is unusable because its Microsoft Store base interpreter cannot be launched in the current session. It is not modified. A dedicated E001 environment will be created from the working system CPython 3.11.9.
- Official Hugging Face repositories for both requested Base checkpoints exist.
- Official model cards report 32 language layers with 8 repetitions of 3 Gated DeltaNet layers followed by 1 full-attention layer. They report matching GDN head geometry (32 value heads, 16 QK heads, 128 head dimension) and matching full-attention KV geometry (4 KV heads, 256 head dimension) despite hidden sizes 2560 versus 4096.
- Transformers 5.15 introduced a native linear-attention/cache refactor and explicitly made external kernels opt-in. The canonical runtime will use a pinned released Transformers build and explicit attention/kernel settings, then prove correctness through Phase A rather than assuming it.

## 2026-08-31 — Implementation

- Created an isolated CPython 3.11.9 environment under E001 without modifying the repository root environment.
- Pinned PyTorch 2.13.0+cu130, Transformers 5.16.1, Datasets 5.0.1, Hugging Face Hub 1.29.0, and all analysis dependencies. The complete resolved package set is frozen in `runtime/environment.lock.txt`.
- CUDA initialization, RTX 5090 compute capability 12.0, BF16 support, and a BF16 CUDA matrix multiplication all passed.
- Frozen the source repository at commit `1001bb4d826a52d1f399e183466143f4da7b741b` and the target repository at commit `68c46c4b3498877f3ef123c856ecfde50c39f404` before weight download. Official LFS SHA-256 values are recorded in `model_revisions.json`.

## 2026-08-31 — Hypothesis

H1 is frozen verbatim in `PREREGISTRATION.json` before model download, cache inspection, direct handoff, translator fitting, or evaluation.

## Ledger rules

1. Append; do not rewrite earlier observations after they influence a decision.
2. Label entries as Observation, Hypothesis, Implementation, Benchmark, Inspection, Redesign, or Next Hypothesis.
3. Record all failures and invalid attempts.
4. Never use this ledger to authorize a condition, translator, threshold, or data access not frozen in the preregistration.

## 2026-08-31 — Observation

- Exact runtime inspection confirmed the expected 32-layer `linear, linear, linear, full` pattern in both checkpoints: 24 Gated DeltaNet layers and 8 full-attention layers.
- Both models expose identical persistent-state geometry: GDN recurrent state `[1, 32, 128, 128]` in FP32, packed Q/K/V convolution state `[1, 8192, 4]` in BF16, and attention K/V `[1, 4, L, 256]` in BF16.
- The official checkpoints can be loaded as text-only `Qwen3_5ForCausalLM` instances by mapping `model.language_model.*` to `model.*`; both loads had zero missing, unexpected, or mismatched language tensors.

## 2026-08-31 — Implementation

- Canonical inference uses Transformers eager full attention and the native PyTorch Gated DeltaNet fallback with optional external kernels disabled.
- The implementation-only source restore smoke covered 256- and 512-token prefixes followed by one bridge token and 64 teacher-forced continuation positions. All compared logits were bit-identical; observed maximum absolute difference was `0.0`.
- Following the preregistered one-time freeze rule and before any cross-model result was examined, the Phase A maximum absolute logit-difference tolerance was frozen at `0.0`.

## 2026-08-31 — Benchmark

- Phase A passed for both official models on all 16 contexts (four each at 256, 512, 1024, and 2048 tokens). Both models had 100% top-1 agreement, zero maximum absolute logit difference, and no NaN/Inf across the 64-token teacher-forced comparisons.
- The non-canonical direct-copy smoke was run unchanged on eight 1024-token implementation-only streams. Mean DeltaNLL versus native 9B was `+7.5191` for `DIRECT_GDN`, `+5.2067` for `DIRECT_KV`, and `+0.1979` for `DIRECT_FULL`. These results do not select or alter any canonical translator.

## 2026-08-31 — Implementation

- Before canonical corpus content was downloaded or inspected, FIT/VALIDATION was frozen to rows 0–19,999 of immutable FineWeb-Edu sample shard `sample/100BT/000_00000.parquet` at commit `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`.
- LOCKED/LONG was frozen to the entire immutable PG19 test parquet at commit `c021754c8e01c5b1cc83a1f549c1f97fbbb756b8`.
- Both file LFS SHA-256 values and the deterministic hash-order assignment rule are frozen in `PREREGISTRATION.json`. Selection depends only on corpus identity, physical row, document identity, and exact tokenizer length—not any model score or state result.

## 2026-08-31 — Observation

- Before paired-state arrays were created, an implementation control compared one fresh 1024-token source prefill with eight successive 128-token calls over the same tokens. The resulting cache checksums differed and the identical bridge token produced a maximum logit difference of `0.125` (cosine `0.9999611`, same top-1 and top-5).
- Therefore, incremental checkpoint collection would introduce a runtime chunking artifact relative to canonical one-shot native prefill.

## 2026-08-31 — Redesign

- FIT and VALIDATION still use the frozen eight checkpoint lengths `128, 256, …, 1024`, but each checkpoint is now captured by independently prefilling that exact prefix into a fresh cache in one call. This decision was made before paired source/target state evidence or translator results existed.
- The failed chunk-equivalence evidence remains in `implementation/checkpoint_chunk_equivalence.json`; it is not hidden or reclassified as scientific translator failure.

## 2026-08-31 — Implementation

- FIT and VALIDATION paired-state collection completed with 1,024 and 256 independently prefetched checkpoints, respectively. Exact source/target tokenizer identity and token-ID equality passed before collection.
- The frozen low-capacity inventory contains independent de-rotated ridge K/V maps, packed-component/head ridge convolution maps without Q/K/V mixing, and per-layer/head bilinear recurrent maps with exactly three alternating ridge updates. No nonlinear component was introduced.
- Before LOCKED access, the identical target continuation execution schedule was frozen as checkpoints `1, 4, 16, 64` for every condition. This controls the runtime chunking effect discovered above while preserving the specified bridge token and 64-token horizon.
- The verdict ladder is interpreted as the highest fully satisfied named rung; `NO_TRANSLATABLE_STATE` is the conservative residual if no positive-transfer rung is fully satisfied. This resolves otherwise unnamed logical combinations without changing a threshold or observed result.

## 2026-08-31 — Benchmark

- The single LOCKED run completed all 64 PG19 documents and seven frozen conditions. `FULL_TRANSLATED` reduced mean excess NLL from `0.9544` for `KV_ONLY` to `0.2070`; the paired improvement was `0.7473` nats/token with 95% bootstrap CI `[0.6921, 0.8047]`.
- Both mean and median improvement fractions exceeded the 0.25 recurrent-state threshold. `FULL_TRANSLATED` also beat `FULL_SHUFFLED` by `0.9482` nats/token with 95% CI `[0.8779, 1.0221]`.
- Aggregate TQR was `-0.4132`, and full translated DeltaNLL was `0.2070`; therefore the frozen canonical verdict is `RECURRENT_STATE_TRANSLATABLE`, not `FULL_STATE_HANDOFF`.
- The oracle trigger did not fire because full translation passed the registered FULL-vs-KV primary gate. The LONG trigger did not fire because the 4K verdict did not reach `FULL_STATE_HANDOFF`.

## 2026-08-31 — Inspection

- The first post-verdict figure-generation call selected Matplotlib's Tk backend and failed because the isolated runtime lacks a usable Tcl/Tk installation. Machine results, derived statistics, diagnostic JSON, and per-document CSV had already been written; no scientific computation failed.

## 2026-08-31 — Redesign

- Figure generation alone was switched to Matplotlib's deterministic noninteractive Agg backend. Existing scientific artifacts are opened read-only and are not overwritten during the figure-only retry.
