# Source rights and access audit — 2026-10-11

**Publication addendum (v2.0.0):** The audit below preserves the pre-release observations and decisions as historical evidence. The author subsequently authorized publication of the cleaned R2 software and exact ENA snapshots on 2026-10-11. Current ENA access is the separately attributed `../source_snapshots/ena_20261010/` directory in the tagged repository and its linked archive; the code-only ZIP still excludes those source records. MicrobiomeHD tables remain excluded. Current execution instructions are in `RELEASE_PREPARATION.md`. This addendum changes access status, not source ownership, scientific conclusions or the scope of prior validation.

This is a factual source-policy and availability record, not a new licence grant or a substitute for the author's institutional/legal review. Scientific reproducibility and redistribution permission are separate checks.

| Source | Observed primary evidence | Candidate handling |
| --- | --- | --- |
| ENA public project reports PRJNA447983 / PRJEB27928 | [ENA/INSDC policy](https://ena-browser-docs.readthedocs.io/en/latest/about/policies.html) describes public access/use and redistribution; [EMBL-EBI terms](https://www.ebi.ac.uk/about/terms-of-use/) retain original-owner rights and resource-specific conditions | Preserve attribution and dated provenance. Do not label the records as project-owned MIT data. Public report snapshots are excluded from the code ZIP; a separate local-only review archive is prepared for an explicit later distribution decision. |
| MicrobiomeHD supplementary q/effect tables | [Author repository](https://github.com/cduvallet/microbiomeHD), commit a6325fbd0af4280f330ca174bd34611abd7a66f0; repository API reported no recognised licence and no root licence file. [Zenodo 840333](https://zenodo.org/records/840333) API specifies CC-BY-NC-4.0 for that dataset record, which does not by itself establish that the two separate GitHub files have identical terms. | Use direct commit-pinned recovery links and cite Duvallet et al., DOI [10.1038/s41467-017-01973-8](https://doi.org/10.1038/s41467-017-01973-8). Do not embed/relicense the third-party tables under project MIT. |
| Project software | Existing [v1.0.1 release](https://github.com/qz-bbb/crc-metagenome-transferability/releases/tag/v1.0.1) and [repository licence](https://github.com/qz-bbb/crc-metagenome-transferability/blob/v1.0.1/LICENSE) specify MIT | Preserve the existing software licence/attribution. The licence does not override upstream data rights. This local R2 candidate is not yet a release. |
| Broad identity audit summary | Retained 2026-10-10 aggregate output, inspected for individual records and private paths | Included only as an unchanged aggregate historical checkpoint with SHA. It is not a fresh audit or proof of participant independence. |

## Actual remote-access checks

The two MicrobiomeHD files were downloaded from the commit-pinned author paths. Both were byte-identical to the historical inputs: 79,577 and 52,044 bytes respectively. Those source-specific routes replace the previously untested acquisition instructions.

The ENA reports did not both reproduce historical bytes remotely. PRJNA447983 first timed out; a second request returned 140 rows and nine columns, all field values identical after sorting by run accession, but in a different row order. It was 19,542 bytes, SHA-256 63400bee1480170936f6de4e75ac8fa62a421006aafa210316a544cf75f2dfb4, whereas the frozen report is 843c5ae888f3d550d3c36230118d17fc0d88d0b1f094f18c8e4f807dece0db0f. The changed payload was not adopted. PRJEB27928 returned HTTP 500, then timed out. Neither a current-order response nor a failed request is an exact dated-source recovery.

The recovery helper therefore retains strict byte-hash checks. Its explicit local-import mode accepts only the frozen snapshots and labels them local_verified. Original retrieval sidecars remain unchanged and separate from new execution receipts. No historical timestamp is reconstructed as if newly observed.

## Public release status

Read-only GitHub API inspection found v1.0.1 (2026-05-27) as the newest listed release. Zenodo record 20415480 identifies that same version with DOI 10.5281/zenodo.20415480. These are historical-public identifiers, not identifiers for the local R2 code. Do not overwrite an availability statement to imply that this R2 candidate is already archived there. A separately approved release must be inspected after publication before assigning its version/DOI to the manuscript.
