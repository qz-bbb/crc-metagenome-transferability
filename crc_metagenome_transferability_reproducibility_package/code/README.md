# Robustness and transferability of colorectal cancer gut metagenome signals across public cohorts

Target journal: BMC Microbiology

Author: Zhun Qiu

This repository-style package supports a reproducible computational reanalysis of sample-level public processed CRC/control gut metagenome matrices. It uses curatedMetagenomicData/ExperimentHub-derived processed relative-abundance tables and associated metadata already resolved in the local project workspace.

## Data source

The analysis uses public, de-identified processed microbiome abundance tables and metadata. It does not contain private identifiable human data and does not include raw sequencing reads.

## Rebuild outline

1. Rebuild or verify the curated CRC/control processed matrix and metadata in the data root.
2. Run `scripts/run_analysis.py` to produce primary association, ordination, permutation, LOSO, and transportability outputs.
3. Run the post-analysis scripts for random-effects heterogeneity, ecological/oral-associated context, robustness sensitivity, and publication tables.
4. Run `scripts/render_journal_figures.py` to regenerate PDF, PNG, and TIFF figures.
5. Run `scripts/build_bmc_submission_package.py` to assemble the BMC submission package.

## Random seeds

Primary scripts use fixed random seeds where stochastic procedures are used. The LOSO elastic-net stress test uses `random_state = 42`; bootstrap and permutation procedures are recorded in their respective outputs.

## Expected outputs

The expected outputs include manuscript DOCX/PDF files, BMC-style figure files, machine-readable supplementary tables, result CSV files, and this reproducibility package.

## Limitations

The package supports processed-data association, robustness, heterogeneity, and transferability analyses. It does not support causal, clinical diagnostic, stage-specific, mechanistic, functional, strain-level, oral-source, or raw-read reprocessing claims.
