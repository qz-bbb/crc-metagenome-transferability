# v2.1.0 Four-definition prediction code

This release adds `prediction_extension/` and `report_extension/` while retaining
the v2.0.0 reproducibility workflows and historical result files unchanged.

## Included

- Config-relative inputs and fresh-output protection; pinned public source
  recovery instructions rather than private working-directory dependencies.
- Training-only filtering, CLR, association, ARP/DCP/ICP construction under four
  direction definitions; one fold/one model per command with fixed parameters.
- Complete nine-panel, two-classifier grids; sealed blind external prediction
  and separate outcome attachment; all six external cohorts retained.
- Tie-aware AUROC and average precision, paired stratified bootstrap,
  equal-study summaries, membership and directional audits.
- Figure 8 renderer with five aggregate reference tables; portable TSV
  counterparts of the new Supplementary Table S4 prediction sheets.
- Exact environment/configuration records, tests and result-to-script maps.

## Verification scope

The existing 138 baseline tests passed. Added tests cover portable interfaces,
input preservation, fixed grids, missing/duplicate records and numerical
behavior. Actual-source replay rebuilt full-training and one outer-fold
selection with identical state hashes. All 540 saved-prediction AUROC/AP values
(nine internal studies and six external cohorts, 18 models, 2 metrics) matched.
Stored 2,000-replicate arrays regenerated 72 macro rows and 1,394 paired contrasts
without numerical drift at 1e-12 tolerance. Figure 8 rendered from the same
summary tables. The seventeen S4 data-sheet exports contain 53,976 rows.

These are bounded checks. This packaging release did **not** refit the complete
model grid, redraw every bootstrap, reinstall packages from the network or test
another operating system. The captured scikit-learn 1.8 environment emits a
deprecation warning for the historical `penalty` argument; fixed parameters are
preserved instead of silently adopting a newer estimator contract.

## Scientific and data boundary

The four-definition extension was planned after earlier external results were
known. It is post-outcome sensitivity using already examined cohorts, not
untouched confirmatory validation or proof of ICP superiority/equivalence.
Fixed-model conditional intervals omit model-development and future-population
uncertainty. All unfavorable and secondary-cohort results remain included.

The new extension contains no raw reads, abundance matrices, individual sample
tables/predictions, fitted-model binaries, caches, credentials or private local
configuration. Those outputs can be generated locally through the documented
workflow. Aggregate reference tables do not contain individual identifiers.
The historical third-party ENA snapshots retain their source attribution and
retrieval sidecars; software's MIT license does not relicense those data.
`PUBLICATION_MANIFEST.tsv` remains the historical v2.0.0 manifest;
`PREDICTION_PUBLICATION_MANIFEST.tsv` covers the added extension and new metadata.

Public availability is separate from author scientific approval or journal
submission. This release does not claim either action occurred.
