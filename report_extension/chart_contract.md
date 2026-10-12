# Figure 8 chart contract

Analytical question: How do four direction definitions change equal-size ICP prediction, and do those changes establish an advantage over ARP?

Supported takeaway: The raw-median ICP has lower macro AUROC than the alternative definitions, with uncertainty depending on learner and scope. Improvements relative to raw median do not establish ICP superiority over ARP; intervals crossing zero do not establish equivalence.

Surface: standalone journal figure, static Matplotlib renderer; PDF, editable-text SVG and 300 dpi PNG. Figure footprint: 180 mm wide, 205 mm tall. Minimum text: 8 pt at publication width. No branding.

Family: uncertainty and benchmark; four faceted horizontal dot-and-interval plots. A/B show nine panel groups, two learner marks each, with 95% fixed-model conditional bootstrap intervals. C shows four EN focal contrast rows (two definitions, two scopes), with nominal 97.5% conditional intervals. D shows eight definition/learner rows, each with internal and external paired ICP-minus-ARP 95% intervals. AUROC axes are identical across A/B; contrasts have explicit zero lines and labeled, locally focused delta axes.

Data sufficiency: exactly 36 AUROC macro rows for A/B; 4 focal contrast rows for C; 16 ICP-minus-ARP rows for D. Source export retains all 72 macro rows (AUROC and AP), full contrasted rows, source paths and line numbers. All six external cohorts remain individual rows in a separate source table, with sample counts, case/control counts, prevalence, panel size, role and conditional CI. No six-cohort macro is produced. Missing/duplicate matches block figure generation.

Palette: hard two-root cap, blue #0072B2 and orange #D55E00 plus charcoal/grey/white. A/B use filled circles for EN and open triangles for RF. C/D use filled circles for internal and open squares for external; D row labels identify learner. Intervals use solid internal and dashed external strokes. Both color and non-color distinctions are retained in grayscale.

Evidence limits: post-outcome sensitivity extension; equal-study macro estimates, not pooled sample-level estimates; intervals condition on fixed fitted models and omit development/refit and new-population uncertainty. No new results, CIs, model fits or inferential tests are computed. Duplicate feature panels are identified in caption/notes. The external raw-mean focal lower 97.5% bound is negative and retained without truncation or positivity labeling.

QA: inspect full-width PDF raster, 120 mm preview and grayscale; confirm page size, font minimum, no text outside page, exact numeric round-trip, interval sign/order, common 18-taxon groups, source row counts, cohort identity and source SHA-256. A caption-adjacent review PDF is provided separately. Record self-review, tests and remaining limitations in REPORT.md.
