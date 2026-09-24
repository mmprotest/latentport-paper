# Reproducibility

## A. Reproduce published statistics from included artifacts

**Artifact-level reproduction: partial.** E002's base/corrected/native means,
correction improvement and CI, remaining-gap reduction and CI, factorial
contrasts and CIs, and post-verdict ablation effects can be recomputed.
E001's primary per-document observations and E002's source, empty-target, and
wrong-donor observations are absent. Aggregate arithmetic is checkable; the
missing observations' sampling uncertainty is not.

From the repository root, create an isolated Python 3.11 environment:

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell:
# .\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python analysis/extract_results.py
python analysis/verify_results.py
python analysis/reproduce_figures.py
python analysis/reproduce_tables.py
```

The extractor also accepts the small, hash-matching sealed per-document summaries
listed in [MISSING_ARTIFACTS.md](MISSING_ARTIFACTS.md). Neither those summaries
nor the original raw files are currently present.

The default verification command currently exits **2**, explaining the required
checks it cannot perform. Run the figure and table commands separately after
reading that output. Exit **1** indicates a schema, hash, or numerical error.

For the explicitly incomplete available-data check:

```bash
python analysis/verify_results.py --available-only
python analysis/validate_public_metadata.py
```

This mode returns zero only if all available checks pass. It still prints
`COMPLETE HEADLINE VERIFICATION BLOCKED`; it does not certify the absent
observations. CI uses the strict default, so this checkout is intentionally
not green until the missing evidence is restored and verified.

No model downloads, network access after dependency installation, GPU,
PyTorch, llama.cpp build, pandas, or SciPy are needed. NumPy implements the
statistics; matplotlib renders the figures; PyYAML validates public metadata.
Scripts resolve paths relative to their own location and write only
`derived/`, `figures/`, and `tables/`.

### Independent unit and exact bootstrap

Both primary tests use 64 independent documents and 64 scored teacher-forced
targets per document. Conditions are repeated measures on the same document.
E001 uses PG19; E002 uses fresh FineWeb-Edu documents. Do not pool their means
or interpret the difference between experiments as a treatment effect.

Each paired contrast is formed **within document**, then averaged across
documents. Bootstrap resampling draws 64 whole document indices with
replacement, retaining condition pairing, for 10,000 replicates.
The generator is NumPy `default_rng`; the percentile interval uses
`np.quantile`'s default linear interpolation at
`alpha = (1 - 0.95) / 2` and `1 - alpha`.

| Experiment | Seed | Sealed implementation |
|---|---:|---|
| E001 | 2026083103 | `e001_handoff/analysis/metrics.py::paired_bootstrap_mean_ci` |
| E002 | 2026090103 | `e002_coupler/analysis/statistics.py::bootstrap_mean_ci` |

The public implementation is independent and cross-checked against the
unchanged pure functions extracted from these files with Python's AST. This
avoids importing their inference dependencies. E001's implementation can be
cross-checked on an available vector; its original primary CIs cannot be
recomputed without its observations.

E002 remaining-gap reduction resamples the paired base/corrected/native rows
and computes a **ratio of sample means in each replicate**, using the same
seed and 10,000 replicates. It is not a bootstrap of per-document ratios.

The paper's E001 79.7% statistic is a **mean of document ratios**, with
nonpositive KV-minus-native denominators excluded. It is neither NCR nor
the ratio of the split's mean excess NLLs. E001's aggregate TQR and mean
document TQR differ; the sealed aggregate TQR is the headline quantity.

### Recovery of E002 observations

The unchanged post-verdict ablation source records, for each document and
each of six ablations:

```text
native_nll    = ablation.nll - ablation.delta_nll_to_native
corrected_nll = ablation.nll - ablation.impact_vs_joint_corrected
```

These identities follow the included ablation writer. All six ablations must
recover exactly identical values for each document or extraction fails.
The recovered 64 native/corrected pairs reproduce their sealed means exactly.
The TDD/base vector comes from
`RESULT.json:factorial_locked_metrics.TDD.document_nll`.

Alignment is checked against the frozen LOCKED document IDs and indices.
The original LOCKED runner iterates the manifest in frozen selection-hash
order, and the aggregator appends document NLL in that same order. These
recovered values are not fresh model evaluations. The method cannot recover
continued-4B, empty-target, or wrong-donor observations.

### Determinism and scope

CSV files use UTF-8, LF newlines, explicit column order, and round-trip float
strings. JSON is sorted, rejects non-finite values, and contains no generation
timestamp. Figures use Agg, DejaVu Sans, fixed dimensions, and no PDF creation
timestamp. Repeated runs in the same environment should be byte-identical.
Different platforms or matplotlib dependencies may render different figure
bytes while leaving scientific values unchanged.

See [ARTIFACT_MAP.md](ARTIFACT_MAP.md) for paper figures and tables that remain
unavailable. The scripts do not digitize the PDF or synthesize observations.

## B. Re-run model experiments

**Full model-level reproduction: partial code is included; an end-to-end
runnable public experiment is not currently included.**

The original E001/E002 runtime, state, translator, correction, and analysis
modules are present as evidence. They import a larger
`experiments.latentport` package namespace, depend on frozen local paths and
artifacts, and preserve one-shot execution guards. This companion does not
patch those sealed files or recommend invoking the original runners in place.

| Item | Recorded original configuration |
|---|---|
| Source | `Qwen/Qwen3.5-4B-Base` |
| Source revision | `1001bb4d826a52d1f399e183466143f4da7b741b` |
| Target | `Qwen/Qwen3.5-9B-Base` |
| Target revision | `68c46c4b3498877f3ef123c856ecfde50c39f404` |
| Backend | PyTorch 2.13.0+cu130; Transformers 5.16.1 |
| Original Python | 3.11.9 |
| Original hardware | NVIDIA GeForce RTX 5090, 32 GB class; models loaded sequentially |
| Model weights | BF16, no quantization |
| Persistent memory | Recurrent matrices FP32; KV and convolution BF16 |
| 4K state payload | 186,122,240 bytes per model |
| Original measured peak allocated VRAM | Source 10.75 GiB; target 19.63 GiB, as reported in E001 |
| Runtime dependency record | `e001_handoff/runtime/requirements.lock` |
| Detailed environment | `e001_handoff/artifacts/attempt_001/implementation/runtime_environment.json` |
| Model file hash records | `e001_handoff/artifacts/attempt_001/implementation/model_revisions.json` and E002 `implementation/e001_integrity.json` |

These are recorded run versions, not dependencies required by this companion.
There is no llama.cpp-based public rerun path in these experiments. Individual
source-code hashes are recorded in the frozen manifests; no separate public
runtime commit should be inferred.

Missing material includes paired training-state arrays, selected E002
correction weights and selection records, original LOCKED records, some
corpus/token artifacts, and the enclosing runtime package setup. Local E001
translator weights and E002 factorial state dumps are preserved but excluded
from Git staging. Their presence does not make the rerun self-contained.
The full-run manifests record roughly 169 GB of E001 artifacts and 104 GB of
E002 artifacts; these are historical inventory sizes, not a verified minimum
resource estimate. No elapsed-time or end-to-end resource estimate for a
public rerun has been validated.
