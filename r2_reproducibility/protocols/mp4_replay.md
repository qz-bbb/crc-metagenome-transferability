# Uniform MetaPhlAn4 computational replay

This document defines a fresh execution of the already completed R2 analysis.
It is not a new preregistration and does not recreate original lock timestamps.

Use author-deposited uniform MetaPhlAn4 percentage profiles from Zenodo
10.5281/zenodo.15069069. Original CRC/control participant metadata defines the
training match. Exclude diagnostic conflicts. Expected training: 1,257 participants,
9 studies, 632 CRC and 625 controls. Derive filtering (prevalence >= 5%, detected
in >= 6 training studies), pseudocount and CLR coordinates from training only.
Rebuild ARP29, DCP29, ICP (capped at 18), exact-ICP-sized ARP/DCP and seeded
prevalence-matched random comparator, using the preserved statistical functions.
Fit fixed Elastic Net and RF parameters. No external tuning or feature selection.

Lock before prediction. Primary CZ contrast is 59 CRC versus 46 source-labelled
controls, including eight hyperplastic-polyp participants that cannot be separately
identified. Do not claim 59 versus 38 or independently validate the historical
301-feature model. The 19 adenoma samples have only an exploratory separate contrast.

Intervals use 2,000 stratified participant bootstraps, standard tie-aware AUROC/AP;
conditional on fitted models and sampled cohorts. This replay is automated and is
not independent human author approval.

