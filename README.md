# LatentPort: Beyond KV Cache - Cross-Model Transfer of Recurrent Memory in Hybrid Language Models

**Simon P. Villani**

[Paper PDF](LatentPort.pdf) · [Citation](CITATION.cff) · [BibTeX](#citation)

We study cross-model transfer of persistent recurrent state in hybrid language models. In the tested Qwen3.5 4B → 9B Base-model pair, attention KV translation recovers little of the available continuation improvement; adding the persistent recurrent/convolutional state package produces a much larger improvement. A small pair-specific, target-supervised correction improves a held-out handoff further. This is a **one-model-pair existence proof** of cross-scale transfer within a closely related hybrid family.

**Reproducibility status: partial.** The sealed summaries and substantial E002 observations are included, but both original LOCKED result folders are missing. E001 per-document headline checks and E002 baseline/control CIs cannot yet be reproduced. The default verifier and CI fail explicitly on these gaps. See [reproducibility](docs/REPRODUCIBILITY.md) and the [release audit](docs/PUBLIC_RELEASE_AUDIT.md). Permanent arXiv metadata is [pending](TODO_PUBLICATION_METADATA.md).

## Main result

![Teacher-forced handoff comparison for Qwen3.5 4B to 9B](figures/main_handoff_comparison.png)

Lower excess NLL is better; zero is the native 9B reference. Each experiment scores 64 teacher-forced targets per document after a 4,096-token prefix. The panels use different corpora and scales. Intervals describe paired differences, not individual bars. [Vector PDF](figures/main_handoff_comparison.pdf) · [figure provenance](figures/README.md).

## What E001 shows

On 64 held-out PG19 documents, the sealed result reports:

- Mean NLL: empty 9B **3.2457**, KV-only **3.1153**, full translated state **2.3679**, native 9B **2.1609** nats/token.
- KV-only gains **0.1304** over empty; full state gains a further **0.7473**, reported 95% CI **[0.6921, 0.8047]**.
- Wrong-donor full state has NLL **3.3162**; correct state gains **0.9482**, reported CI **[0.8779, 1.0221]**.
- The stronger handoff gate fails: excess NLL **0.2070** and TQR **−0.4132**. Translated KV with **direct** recurrent/convolution reuse performs better than the tested fully learned maps (NLL **2.2913**).

These means and arithmetic agree across included summaries. Their per-document CIs and the paper's “64/64 improve” result remain unverified in this checkout. [Sealed result](e001_handoff/artifacts/attempt_001/verdict/RESULT.json).

## What E002 shows

On 64 fresh held-out FineWeb-Edu documents:

- Validation selects **TDD**: translated KV, directly copied recurrent and convolution state.
- Base NLL **2.0184** becomes corrected NLL **1.9894**. The **0.0290** improvement and 95% CI **[0.0228, 0.0354]** recompute from included evidence.
- Corrected handoff is **0.0521** better than continued 4B, reported CI **[0.0185, 0.0843]**; the source mean is **2.0415**.
- Native 9B is still better: NLL **1.9130**, leaving **0.0764** excess NLL. NCR is **0.9178** and TQR **0.4058**.
- The correction has **434,176** trainable parameters. The stronger near-native gate fails; the conditional 16K branch does not run.

Native and corrected document scores are recovered by exact identities from preserved post-verdict ablations, cross-checked across all six ablations. Source/control means and intervals remain aggregate-only. NCR is recovered context benefit, not accuracy or native equivalence. [Sealed result](e002_coupler/artifacts/attempt_001/verdict/RESULT.json).

## Why the result matters

Hybrid models retain persistent memory beyond attention KV. KV-only transfer is an incomplete abstraction for this tested pair. E001 intervenes on the **GDN package together**: recurrent matrices, convolution history, and initialization semantics. It does not isolate the recurrent matrices from the other changes. Direct reuse beating the learned GDN maps also limits what can be concluded about the value of those maps.

## What this does NOT show

**This repository does not demonstrate a production speedup.**

**The experiments evaluate teacher-forced continuation. Free-running generation remains outside the scope of this paper.**

There is one directed, geometry-matched, same-family route; no cross-family, arbitrary-model, native-equivalent, free-generation, or deployable model-switching result is claimed. E002's experiment label `FULL_STATE_HANDOFF` is a registered threshold result, not a production or equivalence claim. [Limitations](docs/LIMITATIONS.md) · [claims and evidence](docs/CLAIMS_AND_EVIDENCE.md).

## Repository contents

| Path | Purpose |
|---|---|
| [e001_handoff/](e001_handoff/) | Existing E001 evidence package, preserved unchanged |
| [e002_coupler/](e002_coupler/) | Existing E002 evidence package, preserved unchanged |
| [analysis/](analysis/) | Lightweight extraction, verification, tables, and matplotlib figures |
| [derived/](derived/) | Supported observations, explicit coverage status, and source hashes |
| [figures/](figures/) / [tables/](tables/) | Deterministic public quantitative outputs |
| [docs/ARTIFACT_MAP.md](docs/ARTIFACT_MAP.md) | Claim-to-file and JSON-path map |

Large local tensors and environments are excluded from ordinary Git staging; they are not needed for artifact statistics. No model inference or downloads are required by the companion scripts.

## Reproduce the reported results

Use Python 3.11 in a fresh virtual environment:

```bash
python -m pip install -r requirements.txt
python analysis/extract_results.py
python analysis/reproduce_figures.py
python analysis/reproduce_tables.py
```

These regenerate the supported public outputs. They cannot regenerate the missing E001 scatter observations or every paper table; [coverage is listed explicitly](docs/ARTIFACT_MAP.md).

## Verify the artifacts

```bash
python analysis/verify_results.py
```

**Currently exits 2:** required observations are missing. To check only the explicitly incomplete available subset:

```bash
python analysis/verify_results.py --available-only
python analysis/validate_public_metadata.py
```

A subset pass is not a complete reproduction certificate. Corrupted inputs or numerical mismatches fail in both modes.

## Citation

The title above is copied exactly from the included PDF. Use [CITATION.cff](CITATION.cff). This local BibTeX is **provisional** until a permanent arXiv ID is available:

```bibtex
@misc{villani2026latentport,
  author = {Villani, Simon P.},
  title = {{LatentPort}: Beyond {KV} Cache - Cross-Model Transfer of Recurrent Memory in Hybrid Language Models},
  year = {2026},
  note = {Preprint; permanent publication identifier pending},
  url = {https://github.com/mmprotest/latentport-paper}
}
```

## License / patent note

The pre-existing Apache-2.0 LICENSE is unchanged. Its intended scope needs owner review in light of the patent-pending work; see [LICENSE_NOTICE.md](LICENSE_NOTICE.md).
