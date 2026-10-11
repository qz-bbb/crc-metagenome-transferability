# Completed multicohort endpoint replay

This is post-hoc computational reproduction of frozen published-in-draft endpoints,
not a new outcome-blind evaluation. Do not rerun historical amendment or admission
scripts or manufacture new chronology. Preserve the original admission/clinical
audit as a separate evidence record.

Primary: LiuNN_2022 (78 CRC/85 controls), YangY_2021 (100/100), IIGM_TU (18/39).
Sensitivity: YangJ_2020 (98/95, household-control clustering not identifiable),
NHSII (14/448, female-only/history definition; 435 adenoma excluded).
Previously observed/post-hoc directional evidence: IIGM_CZ (59/46; 19 adenoma excluded).

All 12 training-locked pipelines evaluated without refitting. Preserve all 72
classification rows, including weak results. Primary directional endpoint uses
mean CLR differences and fixed training coefficient signs, detection in included
CRC/control specimens, numerical-zero tolerance 1e-12, whole-panel denominator;
undetected and zero-direction taxa are discordant. BH applies to all 800 coordinates.
Raw-abundance median direction is secondary and never replaces the primary endpoint.
Use original fixed seeds and 2,000 stratified participant bootstrap draws.

Initial source audit checks aliases and exact profile fingerprints. It cannot
prove absence of hidden participant duplication; later accession-linkage audit is
not performed by this numerical replay.

