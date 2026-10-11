# Prior baseline validation (retained evidence, not this M7 run)

This report and validation_summary.json describe the preceding finalization candidate. Its 68-test count, environment creation, download checks and metric replay must not be attributed to the current M7 extension. See M7_VALIDATION.md and m7_validation_summary.json for the extension's fresh 119-test and generator/consumer evidence. Its original package receipt was retained outside this candidate; the new delivery has its own receipt.

Status: DONE_WITH_CONCERNS. Local reproducibility implementation and bounded verification completed. Full public end-to-end acquisition of every derived historical checkpoint and independent author review are not certified.

- Independent Python 3.12.10 venv with local distribution payloads; all scientific module imports resolved within that venv, with system site-packages disabled. Network pip install failed due to SOCKS proxy support and was not counted as successful.
- 18 boundary tests, 41 scientific helper/statistical tests and 9 controller-module tests passed (68 total). Tests cover config/output boundaries, unsafe archive members, hash drift, exact matrix canonicalization, threshold serialization, selection/identity guards, tied metrics and finite/zero-effect behavior.
- cMD source recovery: fresh pinned metadata and representative FengQ official RDA downloads SHA-match. All twelve locally held source files verified; reconstructed metadata identical and 944 × 1,395 matrix values/axes exactly identical.
- MP4 source recovery: existing full metadata/profile archive checksums and all 34 extracted input hashes passed. Live official record inspected. Full profile archive not redownloaded.
- MP4 audit replay: 1,257 training records, nine studies, 632 CRC/625 controls; external CZ profile/metadata ID join and finite nonnegative abundance QC passed. No model fit invoked.
- Rule/identity extension: explicit `rule-init` and `identity-prepare` executed. Identity selection returned 1,214 records (594 CRC/620 controls), matching the audited 43 exclusions.
- Frozen replay: all eleven rule eligibility/selected memberships and target sizes reproduced; 110 rule-ablation and 72 identity-sensitivity AUROC/AP point estimates reproduced at absolute tolerance 1e-12. Twenty paired bootstrap repeats were deterministic. Historical 2,000-repeat CIs were not replayed.
- Historical CLR direction branch fully recomputed; all three output TSVs have identical SHA-256 to the prior candidate replay.
- Controller definition module included unchanged; its nine tests passed here. Full four-definition endpoint replay and result review are controller-owned and are not attributed to this task.
- Package manifest/ZIP contents and archive hash are recorded separately in `package_receipt.json`; no data matrices, individual records, models, environments, cache or local config are exported.

Not rerun: full 11-fold/500-tree models; full identity refit/prediction/evaluation; original nested/matched-size/RF full computation; historical bootstrap intervals; independent author review; public release. The preceding candidate's numerical report remains prior-run evidence, not a new execution claim.
