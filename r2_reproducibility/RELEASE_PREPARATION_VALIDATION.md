# Actual source-recovery checkpoint — 2026-10-11

Status: bounded local source recovery and release preparation verified. Not a published R2 release, a full from-zero public replay, author scientific approval or completion of every reviewer request.

## Fresh evidence in this checkpoint

| Check | Actual result |
| --- | --- |
| MicrobiomeHD q/effect input recovery | Both files acquired using the integrated helper from commit-pinned public URLs; 79,577 and 52,044 bytes; exact historical SHA-256 matches; remote_verified |
| ENA historical snapshot recovery | Both frozen reports recovered by explicitly requested local import with exact historical byte counts/hashes; original provenance unchanged; separate fresh receipts; local_verified, not remote acquisition |
| Current ENA API | PRJNA447983 returned 140 x 9 identical record values after diagnostic sorting but different row order and SHA; PRJEB27928 returned HTTP 500/timeout. No changed payload adopted |
| Historical-full generator using freshly acquired MicrobiomeHD inputs | 12 tables exactly reproduced: ten files byte-identical, two gzip matrices identical after decompression; fixed historical cMD metadata/matrix supplied locally; targets used only after generation |
| Identity enrichment and audit using recovered ENA inputs | Four tables byte-identical to frozen historical targets |
| Existing closure identity-prepare consumer using new enrichment | Three tables byte-identical; no model fitting. Bundled aggregate checkpoint separately confirmed identical to the historical source used by this consumer |
| Regression tests | 138 passed: inherited 119 plus 19 source-recovery tests; all seven suites freshly run in this checkpoint |

Exact generated-result hashes, source modes and test counts are in `release_preparation_validation.json`. Original provenance files and the aggregate checkpoint are pinned in `m7_recovery/m7_source_inventory.json`. The aggregate summary is not a newly generated audit or evidence of unobservable participant independence.

## Safety and review

The helper refuses unsafe hosts/redirects/paths, existing or overlapping destinations, oversized or changed source bytes, and invalid provenance. It requires a fresh external root and never sorts source payloads or changes the historical hash contract. Keep the documented `m7_recovery/` layout: the whole parent package is then protected, including sibling paths. Historical source receipts remain separate from current execution receipts.

An independent task review found a nested-package output-boundary defect. A retained failing behavior test reproduced it; the fix and its regression passed, and scoped re-review found no remaining Important/Critical findings. The final whole-candidate review is retained separately in the local handoff, not represented as author scientific sign-off.

Existing Python 3.12.10 was used; no installation or new environment was created. Numerical package versions are captured in the JSON. No full RF, nested-suite, external-model refit or 2,000-bootstrap replay was performed. The recovered-input historical stage only repeated its bounded historical calculations. No scientific rule or reported result changed. No source tree, previous delivery, public repository, archive or journal submission was modified.

## Remaining boundary

The clean code ZIP excludes individual source records, matrices, predictions, models, environments and local configs. MicrobiomeHD is acquired directly, not redistributed under project MIT. Exact ENA snapshots are in a separate LOCAL REVIEW ONLY archive pending a specific archival/redistribution decision. Historical public release v1.0.1 and DOI 10.5281/zenodo.20415480 do not contain these local R2 additions. Public release, durable access to dated snapshots and author review remain separate actions.
