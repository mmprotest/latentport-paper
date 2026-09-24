# Missing evidence needed for complete reproduction

The existing evidence packages are partial exports of larger sealed runs.
Do not regenerate or edit files to make them match. Retrieve only the exact
already-sealed E001/E002 artifacts from an author-approved source.

## Minimal compact observation files

These two small files contain the original per-document outcomes needed for
the primary statistical checks. The extractor supports them as alternatives
to the raw JSONL files and validates their recorded SHA-256 hashes.

| Exact repository path | Bytes | Expected SHA-256 |
|---|---:|---|
| `e001_handoff/artifacts/attempt_001/locked/derived_statistics.json` | 53,849 | `725f5eddacd78097852a8d712e4e93f62ce854fb2f1b76d05d657fb380ebfb52` |
| `e002_coupler/artifacts/attempt_001/locked/locked_metrics.json` | 550,197 | `8a2b2aff4719534395c4d75aff1dc6a3703bf0f154ede361a02c301c048ad40a` |

E001's `per_document[*].nll` provides all seven conditions and document IDs.
E002's `conditions.<condition>.document_nll` provides the missing baseline
vectors in the frozen manifest order. These are genuine sealed observations,
not reconstructions from aggregate means.

## Original primary raw files

| Exact repository path | Bytes | Expected SHA-256 |
|---|---:|---|
| `e001_handoff/artifacts/attempt_001/locked/raw_evidence.jsonl` | 16,577,489 | `1639038217b9bb376de33be57ab624e083a093a45dc5a5f8749b565247c9e546` |
| `e002_coupler/artifacts/attempt_001/locked/raw_evidence.jsonl` | 41,118,241 | `a189c454a4f52a8ceb03ec09e0e109efa139ce01225a472c59a70148acd0c629` |

The raw records support more detailed audits, including token-level metrics,
donor assignments, execution paths, and original scoring evidence. Compact
summaries suffice for headline statistics but do not replace those audits.

The E001 original `locked/per_document_metrics.csv` is 12,132 bytes with
SHA-256 `6538fa8cd8b57e89e442cc778277069ea20c30a798d031c712630d62e6596f94`.
It is a useful additional cross-check; the current extractor uses the structured
summary or raw JSONL rather than guessing that CSV's schema.

## Other gaps

- E002 `artifacts/attempt_001/fit/correction_selection.json` is 274,534 bytes,
  SHA-256 `76e658211aaf933922dbe1b1e714eef7877db996831e18d61a3c6d9674f5f62c`.
  It is needed to reproduce the paper's candidate-selection Table 5 from
  experiment artifacts.
- No independent audit JSON was supplied; do not manufacture one.
- The paper's local revision feasibility audit is not included.
- LaTeX sources and original paper-specific plotting code are not included.
- Full inference reproduction additionally needs the runtime/package setup,
  paired training/token artifacts, and selected E002 correction weights.
  See the original manifests for their inventory.

After the exact observation files arrive, rerun extraction, strict
verification, figures, and tables. Update all coverage notices only after the
checks pass. The current documents and figure captions deliberately describe
the existing partial checkout; a future restoration needs a corresponding
documentation review.
