# Figure 8 and supplementary prediction tables

This module renders existing estimates. It never selects panels, fits models,
resamples participants or chooses favorable cohorts. Numerical generation is in
`../prediction_extension/`. Historical Figures 1–7 remain in the v2.0.0 workflow.

## Figure 8

From the repository root, supply the two summary output directories:

```text
python -B report_extension/render_figure8.py --internal-summary ../work/internal_summary --external-summary ../work/external_summary --output ../reports/figure8
```

Required internal tables: `macro_metrics.tsv`, `paired_contrasts.tsv`.
Required external tables: `primary3_macro_metrics.tsv`,
`paired_contrasts_all_scopes.tsv`, `all6_cohort_metrics.tsv`.
These are produced by the corresponding summary commands in the prediction
module. Outputs include PDF/SVG/PNG, all plotted rows, extended source tables,
a caption-adjacent review PDF, size/font checks and grayscale preview.
Use Python 3.12 with matplotlib, Pillow and PyMuPDF (see the environment file).
No new scientific quantities are generated. PDF timestamps may change; compare
source rows, plotted geometry and pixels rather than ZIP/PDF wrapper hashes.

## Supplementary Table S4

`export_tables.py` produces portable lossless TSV counterparts of the 17 new
prediction data sheets plus notes and source manifest. It does not recreate the
48 historical XLSX sheets or proprietary presentation formatting. Those are
separately retained in the manuscript supplement; the numerical new results are
fully reconstructable without the spreadsheet authoring software.

```text
python -B report_extension/export_tables.py --manifest ../work/s4_sources.json --output ../reports/s4_prediction
```

The manifest is a JSON object with `groups`, each containing `name` (the S4 sheet
name) and `files` (paths relative to the manifest). Example:

```json
{"groups":[{"name":"Prediction_B_membership","files":[{"path":"fold0/panel_membership.tsv","held_out_study":"StudyName"}]},{"name":"Prediction_C_macro","files":["external_summary/primary3_macro_metrics.tsv"]}]}
```

For the complete supplement include internal macro, study metrics, contrasts,
fold summary, panel overlaps/counts, eligibility changes, raw-median losses,
nine memberships and predictions; external macro, cohort metrics, contrasts,
full-training membership, training overlaps, all-cohort direction summary and
six labeled predictions. `S4_sheet_routes.tsv` maps all names to generator
commands. File lists are explicit and preserve order. The optional
`held_out_study` field inserts a fold identity only when missing from that source.

Outputs must be new or empty and disjoint from sources and this repository.
Nothing in either script uploads output. Per-participant prediction sheets are
local results, not part of this public code release. Do not publish them without
an additional rights/privacy review. The external comparisons remain
post-outcome sensitivity, not untouched confirmation; intervals crossing zero
establish neither superiority nor equivalence.

Tests: `python -B -m unittest discover -s report_extension/tests -v`.
