# Reproducibility

Two different claims are easy to mix together. This repository supports the first. It does not, by itself, support the second.

## A. Reproduce published statistics from included artifacts

**Headline reproduction from frozen evidence: available.**

The repository includes the frozen E001/E002 evidence required to regenerate
the published headline aggregate statistics, tables, and figures.

This is distinct from re-running the original GPU experiment from scratch,
which requires the specified model checkpoints and runtime environment.

Included and hash-checked:

| Evidence | Path |
|---|---|
| E001 LOCKED raw evidence | `e001_handoff/artifacts/attempt_001/locked/raw_evidence.jsonl` |
| E001 LOCKED derived statistics | `e001_handoff/artifacts/attempt_001/locked/derived_statistics.json` |
| E002 LOCKED raw evidence | `e002_coupler/artifacts/attempt_001/locked/raw_evidence.jsonl` |
| E002 LOCKED metrics | `e002_coupler/artifacts/attempt_001/locked/locked_metrics.json` |
| E002 correction selection | `e002_coupler/artifacts/attempt_001/fit/correction_selection.json` |
| E002 validation factorial | `e002_coupler/artifacts/attempt_001/factorial/raw_evidence.jsonl` |
| E002 LOCKED factorial vectors | `e002_coupler/artifacts/attempt_001/verdict/RESULT.json` and the LOCKED raw file |

`python analysis/verify_results.py` recomputes the headline means, paired
bootstrap intervals, the E001 64/64 improvement count, E002 factorial
contrasts, and the post-verdict ablation effects from those files. It also
checks SHA-256 values in the frozen manifests for included files of at most
10 MB. Larger tensors are skipped by that hash pass and are not required for
the headline statistics.

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
python analysis/validate_public_metadata.py
```

Exit **0** means the checks passed. Exit **1** indicates a schema, hash, or
numerical mismatch. Exit **2** would mean a required observation file is
absent. On the current checkout the observation files above are present, so
a passing run is exit 0 rather than a partial certificate.

No model downloads, network access after dependency installation, GPU,
PyTorch, llama.cpp, pandas, or SciPy are needed for this path. NumPy
implements the statistics; matplotlib renders the figures; PyYAML validates
public metadata. Scripts resolve paths relative to their own location and
write only `derived/`, `figures/`, and `tables/`.

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
unchanged pure functions extracted from these files with Python's AST.
This avoids importing their inference dependencies.

E002 remaining-gap reduction resamples the paired base/corrected/native rows
and computes a **ratio of sample means in each replicate**, using the same
seed and 10,000 replicates. It is not a bootstrap of per-document ratios.

The paper's E001 79.7% statistic is a **mean of document ratios**, with
nonpositive KV-minus-native denominators excluded. It is neither NCR nor
the ratio of the split's mean excess NLLs. E001's aggregate TQR and mean
document TQR differ; the sealed aggregate TQR is the headline quantity.

### What the derived tables are

`analysis/extract_results.py` reads the LOCKED raw JSONL files, checks their
recorded SHA-256 values, and writes one row per document. E002 native and
corrected scores are also cross-checked against the six post-verdict ablation
identities. Those recovered values are not fresh model evaluations.

Condition-mean intervals newly computed by the extractor are document-mean
bootstraps. They are not sealed uncertainty intervals for the condition means.
The paper's reported intervals are the paired contrasts, and those are
recomputed by the verifier.

### Determinism and scope

CSV files use UTF-8, LF newlines, explicit column order, and round-trip float
strings. JSON is sorted, rejects non-finite values, and contains no generation
timestamp. Figures use Agg, DejaVu Sans, fixed dimensions, and no PDF creation
timestamp. Repeated runs in the same environment should be byte-identical for
`derived/` and `tables/`. Different platforms or matplotlib builds may render
different figure bytes while leaving scientific values unchanged.

See [ARTIFACT_MAP.md](ARTIFACT_MAP.md) for the paper figure each public plot
corresponds to. The scripts do not digitize the PDF or synthesize observations.
Generated summaries replace machine-local absolute paths with package-relative
references. The sealed evidence files keep their original paths.

## B. Re-run model experiments

**Full model-level reproduction: the original runtime is present as evidence; an end-to-end public rerun is not self-contained.**

The original E001/E002 runtime, state, translator, correction, and analysis
modules are present. They import a larger `experiments.latentport` package
namespace, depend on frozen local paths and artifacts, and preserve one-shot
execution guards. This companion does not patch those sealed files or
recommend invoking the original runners in place.

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

These are recorded run versions, not dependencies required by the headline scripts.

A from-scratch rerun still needs the model checkpoints, the enclosing runtime
package, paired training-state arrays, corpus token artifacts, and the
selected E002 correction weights (`correction.safetensors`, recorded in the
selection file and not included in this checkout). Local translator tensors
and E002 factorial state dumps may exist on a development machine; `*.pt`,
`*.npy`, and `*.safetensors` are gitignored. Their presence does not make
the rerun self-contained.

The full-run manifests record roughly 169 GB of E001 artifacts and 104 GB of
E002 artifacts. Those are historical inventory sizes, not a verified minimum
resource estimate. No elapsed-time or end-to-end resource estimate for a
public rerun has been validated. Recorded stage timings are not a production
latency result.

Sealed experiment unit tests import `experiments.latentport` and are outside
the lightweight GitHub Actions job in `.github/workflows/verify.yml`.
