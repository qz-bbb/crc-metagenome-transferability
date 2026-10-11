# Actual M7 extension validation — 2026-10-11

Historical-stage record: this file describes the preceding generator repair and its 119-test run. The subsequent source-access preparation, additional recovery tests and fresh recovered-input runs are documented in `RELEASE_PREPARATION_VALIDATION.md`. The later checkpoint verified the MicrobiomeHD remote routes and included the broad aggregate identity checkpoint; the dated ENA source records and public-release boundary remain separate. Do not interpret this historical stage's limits as a statement that those later checks were unperformed.

Status: named historical-intermediate generator gaps repaired and locally verified, with remaining source-access/release limitations. This does not certify all reviewer comments or manuscript submission readiness.

## Actual fresh executions

- Historical full generation from upstream metadata/matrix, exact algorithm snapshots and MicrobiomeHD source tables: final source-guarded run completed in 47.6 s. Twelve generated tables compared exactly with historical targets: ten uncompressed files byte-identical; both 1,395 x 301 gzip matrices identical after decompression and in parsed ordered values. No reference target was configured for the guarded generating run; comparison was a separate later operation. Five named consumer dependencies are therefore generated, not copied.
- Historical CLR consumer: three output TSVs byte-identical; taxonomy-context consumer: four output TSVs byte-identical. These executions used the earlier packaged generator output, before added preflight hardening. The final hardened generator produced exactly the same scientific files. RF-consumed DCP/ICP member selections matched at 29/18; the historical full RF model stage was not rerun.
- Accession enrichment: 373 rows, 195 unique original UIDs. Training audit: 1,257 rows, 1,214 accession-resolved (594 CRC, 620 control), 43 alias-only records retained as excluded, all from Wirbel. Four TSVs byte-identical. The integrated closure consumer used the new enrichment manifest and reproduced its three output TSVs byte for byte. These counts do not prove universal participant independence.
- MP4 sparsity: all eight output TSVs byte-identical. The four-definition consumer used newly generated sparsity inputs and reproduced all nine TSVs byte for byte. These descriptive outputs reuse locked panels/evidence and recorded external replication q values; they are not new tests or new validation cohorts.
- Final fresh test run: 119/119 passed (23 boundary/integration, 41 existing scientific, 9 definition, 16 historical-generator, 14 sparsity, 16 identity tests). Coverage includes exact-value/axis tamper rejection, output safety, failed-comparison nonzero exit, scoped ambiguous identity matching, duplicated records, profile order, absent-feature semantics, zero/finite effects, and integrity checks under optimized Python.

## Environment and provenance

Existing Python 3.12.10; numpy 2.4.5, pandas 3.0.3, scipy 1.17.1, statsmodels 0.14.6, scikit-learn 1.8.0, pyreadr 0.5.6 and joblib 1.5.3. Exact environment summary and output hashes are in m7_validation_summary.json. This extension performed no installation or new isolated-venv validation. The original baseline's separately recorded isolated environment/source-access evidence is retained in VALIDATION.md and validation_summary.json, explicitly as prior work.

Historical source AST helpers/selected statement intervals were recovered from exact retained versions, with cleaned source hashes in algorithm_provenance.json. Actual original serialization boundaries are retained. All old runs, source data and submitted files remained unchanged. No historical package/version, Git commit, model chronology or human review signature was invented. Public release/submission was not performed.

## Limits

No claim is made of ICP superiority, new clinical/causal evidence, universal identity resolution, full model/2,000-bootstrap rerun, complete fresh network source acquisition, or cross-platform reproduction. Two dated ENA snapshots with original provenance, two MicrobiomeHD tables and the broader identity audit summary remain externally configured historical inputs. They are not included in the code-only archive. M7's selected intermediate-generation problem is addressed; complete public end-to-end access still requires a separately reviewed clean release/source-distribution decision.
