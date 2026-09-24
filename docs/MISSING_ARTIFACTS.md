# Artifact coverage

Earlier release notes described a partial checkout in which the LOCKED
observation files below were absent. They are present in this repository.
Their sizes and SHA-256 values match the sealed expectations. Do not
regenerate them to make a hash match, and do not edit the frozen manifests.

## Headline evidence now included

| Repository path | Bytes | SHA-256 |
|---|---:|---|
| `e001_handoff/artifacts/attempt_001/locked/derived_statistics.json` | 53,849 | `725f5eddacd78097852a8d712e4e93f62ce854fb2f1b76d05d657fb380ebfb52` |
| `e001_handoff/artifacts/attempt_001/locked/raw_evidence.jsonl` | 16,577,489 | `1639038217b9bb376de33be57ab624e083a093a45dc5a5f8749b565247c9e546` |
| `e001_handoff/artifacts/attempt_001/locked/per_document_metrics.csv` | 12,132 | `6538fa8cd8b57e89e442cc778277069ea20c30a798d031c712630d62e6596f94` |
| `e002_coupler/artifacts/attempt_001/locked/locked_metrics.json` | 550,197 | `8a2b2aff4719534395c4d75aff1dc6a3703bf0f154ede361a02c301c048ad40a` |
| `e002_coupler/artifacts/attempt_001/locked/raw_evidence.jsonl` | 41,118,241 | `a189c454a4f52a8ceb03ec09e0e109efa139ce01225a472c59a70148acd0c629` |
| `e002_coupler/artifacts/attempt_001/fit/correction_selection.json` | 274,534 | `76e658211aaf933922dbe1b1e714eef7877db996831e18d61a3c6d9674f5f62c` |

E002 validation factorial raw evidence and the per-document factorial JSON
records are also included. Together with the LOCKED factorial vectors, they
support the published component analysis. `analysis/reproduce_tables.py`
exports the correction candidate grid from the selection file.

The verifier's manifest pass reports many additional historical paths as
absent. Those are full-run state dumps, training arrays, and similar large
artifacts. Missing full-run files are not a failure of the headline checks.
Files larger than 10 MB are not hash-certified by that pass even when present.

## Still required for a from-scratch model rerun

- The recorded Qwen3.5 4B and 9B checkpoints and the original runtime package
- Paired training-state arrays and corpus token artifacts listed in the frozen manifests
- Selected E002 correction weights, `correction.safetensors`, named by the selection file
- Gitignored local tensors (`*.pt`, `*.npy`, `*.safetensors`), including translator weights and factorial state dumps when they exist only on a development machine

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md). No elapsed-time or production-latency claim is attached to those missing materials.

## Not included, and not invented

- No `INDEPENDENT_AUDIT.json` was supplied
- The manuscript's local revision feasibility audit is not included
- LaTeX sources and the original paper plotting scripts' full runtime inputs are not included
- The public figures are quantitative reconstructions from the frozen observations, not byte copies of the PDF artwork
