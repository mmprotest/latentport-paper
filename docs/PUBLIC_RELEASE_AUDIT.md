# Public release audit

**Status: not ready for the requested complete public reproduction claim.**

The companion layer is implemented. Scientific evidence files, their names,
and the existing LICENSE and paper PDF are preserved. Missing observations,
selection evidence, and licensing/publication decisions remain explicit.
This is a repository preparation audit, not an independent scientific audit.

## Initial repository inspection

- Tracked files at entry: README.md, .gitignore, LICENSE.
- Untracked material already present: LatentPort.pdf, e001_handoff/, e002_coupler/.
- E001: 105 non-environment files; E002: 253.
- Both packages contain protocols, preregistrations, frozen manifests, final
  manifests, reports, results, implementation records, and original analysis code.
- E001 has no included per-document LOCKED outcomes. E002 includes 32 factorial
  validation records and 64 post-verdict ablation records; its sealed RESULT
  additionally embeds eight LOCKED factorial vectors.
- Both original LOCKED result directories are absent.
- No INDEPENDENT_AUDIT.json, LaTeX source, existing citation file, or original
  GitHub Actions workflow was found.
- The exact PDF title and author are used. Only a temporary arXiv submission
  stamp was found for this paper.

## Size, duplicates, and Git inclusion

[ARTIFACT_INVENTORY.csv](ARTIFACT_INVENTORY.csv) gives every original
non-environment file's size and hash.
[LOCAL_FILES_OVER_10MB.csv](LOCAL_FILES_OVER_10MB.csv) lists **all 94 local
files over 10,000,000 bytes**, including ignored dependency environments.
Filter `over_50_mb=True` for **81 files over 50,000,000 bytes**.

Of these, **67 scientific tensors exceed 10 MB; 66 exceed 50 MB**:

- E001 frozen translator tensors: KV 17,042,160 bytes; convolution
  103,840,600 bytes; recurrent 201,339,648 bytes.
- E002 factorial state dumps: 32 source and 32 translated `.pt` files,
  each 186,144,035 bytes, totaling 11,913,218,240 bytes.

The remaining 27/15 large files are ignored dependencies. No pretrained model
weight dump was found outside the ignored environments; the scientific
binaries are translator weights and captured states. Their intended local
use is preserved; they are not needed for lightweight statistics and should
not be indiscriminately committed. All are now ignored, not deleted.
The ordinary Git candidate set contains no file over 10 MB.

Hash comparison found one nontrivial duplicate pair:
`e002_coupler/artifacts/attempt_001/factorial/base_state_selection.json` and
`factorial_validation_statistics.json`, each 13,096 bytes. The other duplicate
group comprises 22 one-byte placeholder files. All are retained.

## Privacy and unpublished material

The scan covered candidate source/metadata text, the PDF's extracted text,
and the original evidence inventory without loading tensor pickle contents.
No API-key, Hugging Face-token, GitHub-token, password-assignment, or private-key
material was identified by the checks used. This is a bounded scan result,
not a guarantee about every possible secret encoding.

Historical absolute paths remain in sealed metadata and runtime code.
They include E001 runtime constants, data-manifest builders/summaries,
architecture/runtime implementation records, E002's data-manifest builder,
and all 32 E002 factorial source and 32 translation metadata records.
They contain local usernames/locations and are left unchanged for provenance.
New public narrative documentation does not reproduce those personal paths.

Two email-like matches occur in corpus URL metadata:
`e001_handoff/data/fit/manifest.jsonl` line 69 and
`e001_handoff/data/validation/manifest.jsonl` line 3.
Their values are not repeated here. They are upstream corpus URLs, not
identified model-service credentials; retain provenance and review their
public inclusion if the owner considers them sensitive.

The sole later-experiment identifier occurs in
`e002_coupler/PREREGISTRATION.json:forbidden`, prohibiting its execution.
No later experiment directory or later research result was imported.
Original prospective hypotheses and E002 post-verdict ablations remain
sealed historical E001/E002 material. No broader compiler or production
runtime was added or accessed.

## Numerical and publication blockers

- [MISSING_ARTIFACTS.md](MISSING_ARTIFACTS.md) lists exact compact and raw
  observation paths and expected hashes.
- E001 primary means and CIs agree across sealed summaries, but its
  per-document means, bootstrap CIs, improvement fractions, and “64/64”
  claim cannot yet be independently recomputed.
- E002 native/corrected/base means, correction effect and CI, remaining-gap
  reduction and CI, factorial contrasts, and ablation effects recompute.
  Continued-source/empty/wrong-donor observations remain absent.
- NCR and TQR arithmetic matches the sealed means; this does not reproduce
  missing baseline observations or their uncertainty.
- Paper Figure 3 and candidate-selection Table 5 cannot be rebuilt from the
  included scientific observations/selection records. Other generated figures
  reproduce supported quantitative content in a new layout.
- The default verifier exits 2 for missing required observations; CI uses this
  strict mode. The available-only mode labels its limited scope.
- LICENSE already contained Apache-2.0. It has not been replaced, narrowed,
  or supplemented with a new patent grant. The owner must review the intended
  licensing scope before releasing currently untracked material.
- Permanent arXiv metadata is unresolved; see TODO_PUBLICATION_METADATA.md.

## Final paper-claim checklist

Here “aggregate agreement” is explicitly weaker than an observation-level pass.

- [x] E001 reported means match included sealed aggregates.
- [x] E001 reported CIs match included sealed result versions.
- [x] E001 wrong-donor point and reported CI match sealed artifacts.
- [ ] E001 observed means/CIs and wrong-donor effect independently reproduced:
  blocked by missing document observations.
- [x] E002 base, corrected, and native means recomputed.
- [x] E002 corrected-vs-base effect and CI recomputed.
- [x] E002 corrected-vs-continued-4B point agrees with sealed means.
- [ ] E002 continued-4B and wrong-donor paired CIs independently reproduced:
  blocked by missing baseline/control observations.
- [x] NCR arithmetic recomputes correctly from the available/sealed means.
- [x] TQR arithmetic recomputes correctly from the available/sealed means.
- [x] Failed stronger gates remain visible.
- [x] No free-generation claim.
- [x] No production-speed claim.
- [x] No cross-family claim.
- [x] No native-equivalence claim.
- [x] No later experiment added.

No commit, push, GitHub setting change, or publication was performed.
Suggested commit groups: public docs/citation/notices; extraction/verification
and derived data; figures/tables; CI. Include existing public evidence deliberately
and review the untracked set; do not use an indiscriminate add-all commit.
