# Fixed 2 by 2 effect-definition decomposition

Created before this batch's new calculations, after the earlier external results were known. This is a post-outcome explanatory analysis, not a preregistration, new model, or independent validation.

## Locked inputs and question

Use the common-MetaPhlAn4 training lock: 1,257 samples, nine original study labels and 800 features; source deposited percent-abundance profiles, fixed feature order and pseudocount. Historical MP3/301 results and all fitted models/predictions remain untouched. The question is whether direction selection changes with abundance scale (raw percent vs CLR) and summary statistic (mean vs median).

## Rules

1. Compute CRC minus control effects under raw mean, raw median, CLR mean and CLR median within every eligible training study. All four use the same nine-study denominator, not just detected or nonzero studies.
2. The direction reference remains the sign of the frozen study-adjusted CLR association coefficient. This isolates rule substitution conditional on the original screen; it is not a complete alternative scale-specific association workflow.
3. Every exact/numerical-zero effect is discordant (absolute tolerance 1e-12 for all four definitions); never drop zero studies from the denominator. A taxon undetected in both groups is discordant even if CLR centering yields a nonzero numerical contrast. Report detected/non-detected and zero counts separately.
4. Keep frozen association q<0.10 and direction agreement cutoffs 0.60, 0.70 and 0.80. For diagnostic ICP eligibility, replace only D; retain original random-effects q<0.10, I2<75, inner sign/q-retention criteria from the existing rule implementation. No parameter search.
5. Report complete 800-feature per-study effects, candidate sets, exact-zero/tolerance counts, threshold denominators and four-definition intersections. Diagnostic capped sets use original q/taxon ordering and original caps, and are not fitted as new models.
6. Explain classic CRC taxa using a fixed previously reported list, not newly selected examples. Validate raw-median and mean-CLR effects against the prior sparsity tables, recording any numerical-guard differences explicitly.
7. Apply the same four definitions descriptively to all six previously evaluated external cohorts and all original locked panels. No external screening, training, recalibration or omission of unfavorable cohorts. Existing external predictions and p/q results remain frozen.
8. No new p-values for a favorable definition, no equivalence claim, no general conclusion that all consistency rules fail. Membership differences isolate operational definitions but not microbial mechanism or a causal explanation of classifier performance.

## Checks

Synthetic fixtures for zero medians with positive means, negative signs, class/axis validation, CLR row-sum invariance, no-detection safeguards, threshold denominators and read-only input hashing. Recompute existing endpoints independently and compare before narrative integration. Retain environment, input/output hashes and numeric results. Figure is data-driven and source-linked.

