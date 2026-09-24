# Public release audit

## Current repository status (2026-09-24)

Headline reproduction from the included frozen observations is available.
`python analysis/verify_results.py` recomputes the E001 and E002 aggregate
results, including the E001 64/64 count and the E002 baseline and control
intervals. The LOCKED raw evidence, derived statistics, LOCKED metrics,
correction-selection file, and factorial evidence listed in
[MISSING_ARTIFACTS.md](MISSING_ARTIFACTS.md) are present and match their
expected hashes.

Publication metadata points at arXiv `2609.25053`:
https://arxiv.org/abs/2609.25053. See [PUBLICATION.md](PUBLICATION.md).

Two limits remain:

- Re-running the original GPU experiment is not self-contained. It still
  needs the model checkpoints, the enclosing runtime package, and gitignored
  tensors, including `correction.safetensors`.
- [LICENSE](../LICENSE) is Apache-2.0 and has not been amended.
  [LICENSE_NOTICE.md](../LICENSE_NOTICE.md) still leaves the intended scope
  of that license, given patent-pending work, as an owner decision.

**Resolved: newline-sensitive hashes.** Fifteen E001 evidence files had
differed from their frozen hashes only because Git had stored LF where the
sealed bytes used CRLF. Those files were restored to their sealed byte
representation and committed with normalization safeguards. `.gitattributes`
marks `e001_handoff/` and `e002_coupler/` as non-normalized so a later
checkout does not rewrite them. The manifests were not edited, and the
expected hashes were not weakened. Frozen hashes now verify successfully
in CI. See [PUBLIC_RELEASE_COMPLETION.md](PUBLIC_RELEASE_COMPLETION.md).

The sections below are the earlier inspection. They are retained as
provenance. Where they say observations are missing or that publication
metadata is pending, the current status above replaces them.

## Earlier release audit status

**Earlier status: not ready for a complete public reproduction claim,
because the LOCKED observation files had not yet been restored.**

The companion layer was implemented. Scientific evidence files, their names,
and the existing LICENSE and paper PDF were preserved. This is a repository
preparation audit, not an independent scientific audit.

## Initial repository inspection

This list describes the checkout at the earlier audit, before the LOCKED
observation files were restored. It is not the current file inventory.

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

## Earlier numerical blockers, and what changed

At the earlier audit, [MISSING_ARTIFACTS.md](MISSING_ARTIFACTS.md) listed
exact observation paths that were absent. Those headline files are now
present and hash-matched. The current consequences are:

- E001 per-document means, bootstrap CIs, improvement fractions, and the
  64/64 result recompute from LOCKED raw evidence.
- E002 continued-source, empty-target, and wrong-donor means and paired
  control intervals recompute from LOCKED raw evidence.
- Paper Figure 3 is regenerated as `figures/e001_paired_documents.*`.
  The candidate grid is exported as `tables/e002_correction_candidates.csv`.
- The default verifier is the strict check. A current passing run exits 0.
- LICENSE remains Apache-2.0. It has not been replaced, narrowed, or
  supplemented with a new patent grant. The owner still needs to review
  the intended licensing scope. See the current status above.
- arXiv metadata is recorded in [PUBLICATION.md](PUBLICATION.md).

## Final paper-claim checklist

Current checklist. Observation-level checks use the included LOCKED raw files.

- [x] E001 reported means match included sealed aggregates.
- [x] E001 reported CIs match included sealed result versions.
- [x] E001 wrong-donor point and reported CI match sealed artifacts.
- [x] E001 observed means/CIs and wrong-donor effect independently reproduced.
- [x] E002 base, corrected, and native means recomputed.
- [x] E002 corrected-vs-base effect and CI recomputed.
- [x] E002 corrected-vs-continued-4B point and paired CI recomputed.
- [x] E002 continued-4B and wrong-donor paired CIs independently reproduced.
- [x] NCR arithmetic recomputes correctly from the included observations.
- [x] TQR arithmetic recomputes correctly from the included observations.
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
