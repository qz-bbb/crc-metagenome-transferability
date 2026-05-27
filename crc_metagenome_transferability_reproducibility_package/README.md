# Transferability-aware prioritization of colorectal cancer gut metagenome signals across public cohorts

## Overview
This repository contains analysis scripts, generated result tables, machine-readable supplementary tables, and figure files supporting the manuscript "Transferability-aware prioritization of colorectal cancer gut metagenome signals across public cohorts".

## Author
Zhun Qiu  
Northeast Forestry University, Harbin, Heilongjiang, China  
qz@nefu.edu.cn

## Target journal
BMC Microbiology

## Study description
This project is a reproducible computational reanalysis of sample-level public processed colorectal cancer gut metagenome matrices. The workflow evaluates study-adjusted association, covariate robustness, random-effects heterogeneity, cross-cohort direction stability, leave-one-study-out separability, pairwise cross-study transfer, and sensitivity to feature filtering and pseudocount choice.

## Data sources
The analysis uses public processed metagenomic resources derived from curatedMetagenomicData/ExperimentHub. No new human participants were recruited, no intervention was performed, and no identifiable private information was accessed.

## Repository structure
- code/: analysis and figure-generation scripts
- results/: generated result CSV files
- supplementary_tables/: machine-readable supplementary tables
- figures/: manuscript and supplementary figure files in PDF/PNG format
- manifest.json: file manifest for this reproducibility package

## Path configuration note
The analysis scripts were originally executed in a local Windows workspace. To reproduce the analysis in another environment, users should update the DATA_DIR, RUN_DIR, and MICROBIOMEHD_DIR variables, or provide equivalent paths to the curatedMetagenomicData-derived processed matrix, metadata, and optional benchmark resources. The generated result CSV files and machine-readable supplementary tables are included in this repository to support interpretation and verification of the manuscript results.

Some scripts may still contain local Windows path assumptions such as D:\, C:\, or UNC-style path separators from the original execution environment. These paths should be replaced with project-local or environment-specific paths before rerunning the workflow outside the original workspace.

## Main outputs
Key generated outputs include:
- full association results
- covariate sensitivity results
- random-effects heterogeneity results
- leave-one-study-out performance results
- pairwise transfer AUROC matrix
- filter sensitivity results
- pseudocount sensitivity results
- leave-one-cohort-out robustness results
- compact leakage-aware LOSO panel summary
- full random-panel LOSO iteration-level table for repository archival
- variance partitioning results
- candidate panel benchmarking results
- ecological guild score results
- supplementary tables S1-S5
- supplementary tables S6-S10
- manuscript figures and supplementary figures

## Software and reproducibility
Please see code/session_info.txt, environment.yml, requirements.txt, or equivalent files if present. Random seeds used in the analysis are documented in the scripts and result outputs where applicable.

## Interpretation limits
The results support bounded processed-data association, robustness, heterogeneity, and transferability claims. They do not establish causal, clinical diagnostic, stage-specific, mechanistic, functional, strain-level, oral-source, or raw-read reprocessing claims.

## Citation
If this manuscript is published, please cite the final published article. A Zenodo DOI will be added after archiving.

## Second-pass leakage-aware revision

This package now includes leakage-aware LOSO panel benchmarking, Aitchison variance partitioning with 999 permutations, panel comparison statistics, taxonomy-defined ecological guild score analyses, and a claim decision table. Fixed global panel benchmarks are retained only as retrospective processed-data transferability stress tests.
