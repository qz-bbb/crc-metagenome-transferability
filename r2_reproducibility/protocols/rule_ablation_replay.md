# Bounded rule ablation and sparsity explanation

Date: 2026-10-10. This is a NEW EXPLANATORY, post-outcome analysis. The historical
LOSO and all six external-cohort outcomes were already known before design. It is
not preregistered and must not replace or retroactively redefine their endpoints.

## Question and fixed rules

Which eligibility criterion removes associated taxa, and does sparse relative
abundance explain disagreement between raw-median and CLR direction evidence?

Historical branch: the original 1,395 CRC/control participants, 11 study labels,
and original unfiltered 944-taxon input. For each outer study, reconstruct the
saved training-only preprocessing (prevalence >=0.05, detected in >=6 training
studies, pseudocount half the minimum positive training value, CLR in the selected
training feature universe). Recalculate training-only association and ICP
statistics and verify against the saved nested-LOSO intermediate, before reuse.
The outer test cohort never influences feature selection, panel size, scaling,
or fitting. Inner leave-study association uses fixed outer-training CLR space,
as in the original ICP definition; it is a robustness rule, not a tuned model.

Base association eligibility A: study-adjusted HC3/BH q<0.10. Direction D:
raw-relative-abundance median direction agrees with the pooled CLR coefficient
in >=60% of eligible training studies; exact zero is discordant. Heterogeneity H:
random-effects q<0.10 and I²<75%. Robustness R: inner leave-study association sign
retained >=90% and q<0.10 retained >=80%. Compare A, A+D, A+H, A+R, A+D+H+R.
Rank all eligible pools by ascending association q then full taxonomy string.
Target size = min(18, number of full-ICP candidates) in that outer training fold.
Never pad a failing pool or tune size based on performance. Report empty pools
explicitly; zero target size prevents that fold's matched-size comparison.
These are eligibility ablations, not newly optimized algorithms. A is explicitly
q-filtered; check that its selected membership equals the legacy matched-size ARP.

Use identical fixed Elastic Net (saga, C=0.5, l1_ratio=0.5, balanced, seed42,
max_iter2000) and Random Forest (500 trees, sqrt, min_leaf2, no depth limit,
bootstrap, balanced_subsample, seed20260817). Standardization fitted on training.
One model at a time; BLAS/OpenMP=1, RF n_jobs=1. No tuning or model selection.
All warnings are retained. Per-fold checkpoints. No new downloads or publishing.

Report AUROC and average precision; 2,000 within-test-cohort stratified participant
bootstraps, using sklearn tie-aware metrics and the same draws for every paired
panel comparison. CI is conditional on trained models and observed cohorts; not
training uncertainty, future-study prediction intervals, or cluster correction.
Report all folds/models, including unfavorable outcomes and candidate shortages.
Across-fold summaries are descriptive; no unplanned favorable-subgroup tests.

## Exclusion and sparsity interpretation

Save independent failure flags and an explicitly order-dependent waterfall
(A -> D -> H -> R), because a taxon can fail more than one rule. For each taxon
and cohort report CRC/control n, detection fractions, arithmetic mean among
positive observations (NA when no positives), raw medians and CLR mean effects.
Zero-median differences remain in the direction denominator as discordant.
Positive-only abundance is descriptive, subject to detection conditioning, not
a causal biological effect or a new confirmatory test. No new FDR fishing.

Common-MetaPhlAn4 branch uses the previously locked 800-feature coordinates,
pseudocount and training panels only for explanation. Recompute descriptive
prevalence/positive abundance/raw median for each of six previously evaluated
cohorts and attach existing full-universe CLR tests without replacing them.
Preserve explicit/unreported profile-row distinctions and original zero policy.
Classic CRC taxa are a predefined display subset: Fusobacterium nucleatum,
Parvimonas micra, Gemella morbillorum, Dialister pneumosintes, Peptostreptococcus
stomatis, Solobacterium moorei, Bacteroides fragilis. Display available exact
taxonomy names; do not manufacture equivalence or replace missing taxa silently.

## Verification and deliverables

Tests precede real compute: rule boundaries, no padding, deterministic ranking,
zero handling, tied metrics, label-independent selection interface. Check raw
input hashes, feature order, training/test disjointness, independent recomputed
association statistics, membership equivalence, and output prediction grids.
Freeze source/protocol hashes in a start receipt. Produce scripts, TSVs, environment,
per-fold receipts, summary report and output SHA-256 manifest. No manuscript,
DOCX, XLSX, public release or stronger claims are part of this bounded module.

