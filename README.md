# CRC metagenome transferability

## v2.1.0 Four-definition prediction extension

The latest addition is a portable, separately scoped sensitivity analysis:

- [Ordered prediction workflow](prediction_extension/README.md): source recovery,
  train-only selection, one model per invocation, sealed predictions, all-cohort
  evaluation, membership and direction audits.
- [Figure 8 and supplementary-table route](report_extension/README.md): render
  the five published aggregate reference tables immediately, or use rebuilt
  outputs from the prediction workflow. Individual prediction tables and
  abundance matrices are not bundled with this extension.
- [Release scope and verification](RELEASE_NOTES_v2.1.0.md).
- `PREDICTION_PUBLICATION_MANIFEST.tsv` identifies all added extension payloads.

Quick figure replay (Python with the recorded plotting dependencies):

```text
python -B report_extension/render_figure8.py --internal-summary report_extension/reference_outputs/internal --external-summary report_extension/reference_outputs/external --output ../figure8_replay
```

This is a **post-outcome sensitivity extension** using already examined external
cohorts. It does not establish ICP superiority, equivalence or clinical utility.
The archival update does not change the historical 1,395-sample/301-feature
analysis. Legacy payloads and v2.0.0 workflows below remain unchanged.

## Historical v2.0.0 release documentation

# CRC metagenome transferability: R2 reproducibility release

Software and reproducibility materials for **Transferability-aware prioritization of colorectal cancer gut metagenome signals across public cohorts**, by Zhun Qiu, Northeast Forestry University.

## Version v2.0.0

- Start with [r2_reproducibility/README.md](r2_reproducibility/README.md) for the ordered workflow, explicit commands, environment, and limits of the performed checks.
- [RELEASE_NOTES_v2.0.0.md](RELEASE_NOTES_v2.0.0.md) distinguishes this release from the historical v1.0.1 baseline and describes the checks actually performed.
- [source_snapshots/ena_20261010](source_snapshots/ena_20261010) preserves two exact public ENA project-report snapshots, with their original retrieval sidecars and byte hashes. These are small metadata reports, not sequencing reads or patient-level clinical datasets.
- The code-only directory excludes abundance matrices, individual predictions, models, private inputs, credentials, environments and caches. Other required processed sources are recovered through checksum-locked official or author-deposited routes; MicrobiomeHD tables are not redistributed here.
- The pre-existing `code/`, `results/`, `figures/`, `supplementary_tables/`, and root `manifest.json` are the **historical baseline**, retained unchanged. They are not the R2 workflow or the R2 file manifest.

The R2 release manifest is [PUBLICATION_MANIFEST.tsv](PUBLICATION_MANIFEST.tsv). The GitHub release and its Zenodo-linked archive preserve this version; use the actual DOI linked from the verified release record rather than treating the historical DOI as the R2 identifier. Version v1.0.1 remains separately archived at [10.5281/zenodo.20415480](https://doi.org/10.5281/zenodo.20415480).

## Source rights

Project software is MIT licensed. The public ENA metadata retains its original source attribution and applicable [ENA/INSDC policies](https://ena-browser-docs.readthedocs.io/en/latest/about/policies.html) and [EMBL-EBI terms](https://www.ebi.ac.uk/about/terms-of-use/); it is **not relicensed as project-owned MIT data**. See the snapshot README and [source rights audit](r2_reproducibility/SOURCE_RIGHTS_AND_ACCESS.md).

## Scope of verification

The release records bounded computational checks, not a complete fresh-environment replay of every model, every bootstrap interval, and every legacy figure. The broader identity audit summary remains a frozen aggregate checkpoint. Publication does not establish universal participant independence, ICP superiority, clinical validation, causal evidence, or independent human scientific review. Historical validation reports retain the status and limitations true at their dated checkpoints; current release instructions supersede only their publication/access status.
