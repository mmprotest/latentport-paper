# Public release completion

Date: 2026-09-24. This record covers repository hygiene for the live preprint.
It does not change the paper's results.

arXiv: [2609.25053](https://arxiv.org/abs/2609.25053).
Hugging Face: [papers/2609.25053](https://huggingface.co/papers/2609.25053).
Repository: [github.com/mmprotest/latentport-paper](https://github.com/mmprotest/latentport-paper).

The public-release fixes were committed and pushed to `main`. Verified history:

- `90cf1d62f6363c909383898f448d6da0248979d5` — Changes
- `ab1507d943e5163b4f20c7e1dbd523f1a5da1c9e` — Finalize public release metadata
- `5654368f23923ed0ae8d2ebef1569df93c29a20f` — Record green CI for release metadata

## Changes made

Public documentation and citation now point at the live preprint and describe
the checkout that is actually here.

- `README.md` — public first screen, key results, direct-reuse finding, two meanings of reproducibility, limitations, citation.
- `CITATION.cff` — arXiv identifier and URL. No DOI was added.
- `docs/PUBLICATION.md` — completed metadata record. `TODO_PUBLICATION_METADATA.md` was removed.
- `docs/REPRODUCIBILITY.md`, `docs/MISSING_ARTIFACTS.md`, `docs/ARTIFACT_MAP.md`, `docs/PUBLIC_RELEASE_AUDIT.md`, `docs/DATA_DICTIONARY.md`, `docs/CLAIMS_AND_EVIDENCE.md`, `docs/LIMITATIONS.md`, `derived/README.md`, `tables/README.md`, `figures/README.md` — stale "observations missing" statements replaced with the current coverage. The earlier audit is kept and labeled as earlier.
- `LICENSE_NOTICE.md` — descriptive wording only. `LICENSE` was not edited.
- `analysis/extract_results.py` — generated summaries rewrite machine-local absolute paths. Sealed inputs are unchanged.
- `analysis/reproduce_figures.py` — captions no longer say the LOCKED observations are absent. The paired E001 scatter is produced when those rows exist.
- `analysis/reproduce_tables.py` — CI status matches the verifier, and the correction candidate grid is exported.
- `derived/`, `tables/`, and the affected files in `figures/` — regenerated from the frozen evidence.
- `.gitattributes` — `e001_handoff/` and `e002_coupler/` are marked non-normalized so Git does not rewrite sealed newlines.

**Resolved.** Fifteen E001 files matched the frozen manifest only with their original CRLF bytes. An earlier Git blob had stored LF, which would fail hash checks on checkout. The sealed bytes were committed, and `.gitattributes` marks `e001_handoff/` and `e002_coupler/` as non-normalized so Git does not rewrite them. The manifests were not edited. Frozen hashes verify in CI. The file list below is the set that had that newline-only difference.

The fifteen files are:

- `e001_handoff/artifacts/attempt_001/diagnostics/direct_zero_training_smoke.json`
- `e001_handoff/artifacts/attempt_001/implementation/architecture_correspondence.json`
- `e001_handoff/artifacts/attempt_001/implementation/checkpoint_chunk_equivalence.json`
- `e001_handoff/artifacts/attempt_001/implementation/learned_translation_install_smoke.json`
- `e001_handoff/artifacts/attempt_001/implementation/state_schema_source.json`
- `e001_handoff/artifacts/attempt_001/implementation/state_schema_target.json`
- `e001_handoff/artifacts/attempt_001/locked/locked_execution.json`
- `e001_handoff/artifacts/attempt_001/locked/per_document_metrics.csv`
- `e001_handoff/artifacts/attempt_001/state_validation/same_model_restore_4b.json`
- `e001_handoff/artifacts/attempt_001/state_validation/same_model_restore_9b.json`
- `e001_handoff/artifacts/attempt_001/state_validation/same_model_restore_smoke.json`
- `e001_handoff/data/MANIFEST_SUMMARY.json`
- `e001_handoff/translators/frozen/gdn_convolution_translator.json`
- `e001_handoff/translators/frozen/gdn_recurrent_translator.json`
- `e001_handoff/translators/frozen/kv_translator.json`

No other hash mismatch was only a newline change. Large gitignored tensors and absent full-run state dumps were left alone.

## Scientific integrity

- Frozen scientific values were not altered.
- Frozen manifests were not edited.
- Expected hashes were not weakened or replaced.
- No missing evidence was invented. Headline files that the old docs called missing are present and match the recorded SHA-256 values.
- Generated `derived/secondary_results.json` no longer copies `C:\Users\Simon\...` paths. The sealed architecture file still has them.
- A second extraction and table run was byte-identical to the first.

## Reproducibility

From the included frozen evidence, the repository regenerates:

- E001 condition means, the 0.7473 nats/token KV contrast and its paired CI, the wrong-donor contrast, document improvement fractions, and the 64/64 count
- E002 corrected, native, continued-4B, empty, and wrong-donor means, the correction contrast, the continued-4B contrast, and the factorial main effects and interactions on both the 32-document validation split and the 64-document LOCKED split
- Public figures for the E001 paired documents, the E002 factorial, the E002 corrected handoff, and the headline comparison
- The correction candidate grid in `tables/e002_correction_candidates.csv`

That is not a from-scratch GPU rerun. The rerun still needs the recorded Qwen3.5 checkpoints, the enclosing `experiments.latentport` package, and gitignored tensors, including `correction.safetensors`.

## Verification

Commands used the repository `.venv` (Python 3.11, NumPy 2.2.6, matplotlib 3.10.8, PyYAML 6.0.3).

`python analysis/extract_results.py` exited 0 and wrote the four derived CSVs. Source SHA-256 lines included the LOCKED raw files:

- E001 raw evidence `1639038217b9bb376de33be57ab624e083a093a45dc5a5f8749b565247c9e546`
- E002 raw evidence `a189c454a4f52a8ceb03ec09e0e109efa139ce01225a472c59a70148acd0c629`

`python analysis/verify_results.py` exited 0:

```text
Included lightweight artifacts match original manifest hashes................. PASS
Manifest entries: {'matched': 301, 'absent': 2315, 'large_skipped': 69}. Missing full-run artifacts are not certified.
ALL CHECKS PASSED (202 checks)
```

The absent manifest entries are historical full-run files, not the headline observations. Files larger than 10 MB are skipped by that hash pass.

`python analysis/reproduce_figures.py` exited 0 and wrote:

- `figures/main_handoff_comparison.pdf` and `.png`
- `figures/e002_factorial_comparison.pdf` and `.png`
- `figures/e002_complete_comparison.pdf` and `.png`
- `figures/state_convergence.pdf` and `.png`
- `figures/e001_paired_documents.pdf` and `.png`

`python analysis/reproduce_tables.py` exited 0:

```text
WROTE tables/e001_summary.csv: 15 metrics
WROTE tables/e002_summary.csv: 22 metrics
WROTE tables/e002_factorial_effects.csv
WROTE tables/e002_correction_candidates.csv: 8 candidates
```

`python analysis/validate_public_metadata.py` exited 0:

```text
CITATION.cff YAML and workflow structure: PASS
```

There is no configured formatter or public pytest suite. Importing `experiments` fails with `ModuleNotFoundError: No module named 'experiments'`. The sealed experiment tests are outside `.github/workflows/verify.yml`. The workflow's local equivalents above are the checks that were run.

Before these documentation edits, the same verifier already passed on the working tree (202 checks). The failure mode for CI was the LF blobs, not a numerical mismatch.

## Remaining blockers

- **Release commit.** Resolved on `main` by commit `90cf1d6` (`Changes`). The `Verify paper artifacts` workflow for that push completed successfully. Do not edit the manifests to match a different line ending.
- **License scope.** `LICENSE` remains Apache-2.0. `LICENSE_NOTICE.md` still says the intended scope, given patent-pending work, needs an owner decision. This pass did not choose a license or interpret patent rights.
- **From-scratch rerun.** Model checkpoints, the original package namespace, and `correction.safetensors` are not in this checkout. Large `*.pt`, `*.npy`, and `*.safetensors` files stay gitignored.
- **Not in the repository, and not invented.** `INDEPENDENT_AUDIT.json`, LaTeX sources, and the manuscript feasibility audit.
- **Local PDF.** `LatentPort.pdf` is the included submission copy (`arXiv:submit/8044432`). Cite https://arxiv.org/abs/2609.25053. The submission stamp may show `cs.AI`. The live record's primary category is `cs.CL`.

## Final metadata cleanup

The live arXiv record is the citation source. Primary category is `cs.CL`. Secondary subject is `cs.AI`.

- `README.md` BibTeX now uses `primaryClass = {cs.CL}`. The submission-PDF sentence that treated `cs.AI` as the citation category was replaced.
- `CITATION.cff` lists `cs.CL` then `cs.AI` in `keywords`. Both arXiv identifiers describe `cs.CL` as primary and `cs.AI` as secondary. Citation File Format has no separate primary-class field.
- `docs/PUBLICATION.md` records the same categories. The included PDF was not edited.
- Sealed evidence, manifests, hashes, and CI workflow files were not edited.

GitHub repository settings, confirmed with `gh repo view` after `gh repo edit`:

- Description: `Cross-model transfer of persistent recurrent inference state without prefix replay`
- Homepage: `https://arxiv.org/abs/2609.25053`
- Topics: `ai-research`, `inference`, `kv-cache`, `language-models`, `llm`, `machine-learning`, `qwen`, `recurrent-state`
- Default branch: `main`

Verification after the metadata edits, using `.venv` (Python 3.11):

- `python analysis/extract_results.py` exited 0
- `python analysis/verify_results.py` exited 0: `ALL CHECKS PASSED (202 checks)`; manifest matches 301, absent historical entries 2315, large files skipped 69
- `python analysis/reproduce_figures.py` exited 0
- `python analysis/reproduce_tables.py` exited 0
- `python analysis/validate_public_metadata.py` exited 0: `CITATION.cff YAML and workflow structure: PASS`

Those reruns did not change `derived/`, `tables/`, or `figures/`.

The latest `Verify paper artifacts` workflow on `main` completed successfully. The latest checked run before this documentation commit was `35963248994` on `5654368f23923ed0ae8d2ebef1569df93c29a20f` (`Record green CI for release metadata`).

License/patent scope remains a human-owner decision and was not changed.
