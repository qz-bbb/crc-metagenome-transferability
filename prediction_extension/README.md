# Four-definition CRC predictive extension

This module makes the retained nine-study and six-cohort prediction analyses portable. It reuses their numerical algorithms and fixed rules. Computational replay does not supply new biological validation, prospective confirmation, author scientific review, or proof of ICP superiority/equivalence. See [PROTOCOL.md](PROTOCOL.md), [INPUT_OUTPUT.md](INPUT_OUTPUT.md), and [RESULT_SCRIPT_MAP.tsv](RESULT_SCRIPT_MAP.tsv).

Use the captured Python 3.12.10 numerical environment in requirements.txt. No packages are installed by the workflow. Validation used the existing matching numerical installation; a fresh network installation and another operating system were not tested. Keep the whole repository together: source verification reuses ../r2_reproducibility/input_contract.tsv. No original working folders are required.

Run from the repository root:

```text
python -B -m unittest discover -s prediction_extension/tests -v
python -B -m prediction_extension --config /work/local/training.json audit
```

For an unrelated working directory, use the absolute launcher; config paths always resolve from the config file:

```text
python -B /path/to/repository/prediction_extension/run_prediction.py --config /work/local/training.json audit
```

These paths are examples, not mandatory platform paths. The tested launcher also supports spaces. All commands require a fresh output_root outside code and all configured input trees. They reject equal, ancestor and nested source overlaps, existing output directories, duplicate axes, invalid labels and incomplete model grids. Failed actions can leave an unsealed partial output; preserve it for diagnosis and choose another fresh destination. The workflow never deletes, overwrites, implicitly resumes or publishes.

## Recover the omitted inputs

Use the existing verified recovery helpers; do not download original sequencing reads. See [R2 source access](../r2_reproducibility/SOURCE_ACCESS.md). The repository provides exact source URLs/checksums; no local historical run is a required public dependency.

```text
python -B /path/to/repository/r2_reproducibility/source_access.py download --root /work/inputs --family cmd
python -B /path/to/repository/r2_reproducibility/source_access.py reconstruct-cmd --root /work/inputs
python -B /path/to/repository/r2_reproducibility/source_access.py download --root /work/inputs --family mp4
python -B /path/to/repository/r2_reproducibility/source_access.py extract-mp4 --root /work/inputs
```

Write /work/local/core.json with these keys:

```json
{
  "source_root": "../inputs/mp4_source",
  "baseline_metadata": "../inputs/cmd/reconstructed/curated_crc_case_control_sample_metadata.tsv",
  "output_root": "../inputs/core_replay"
}
```

Generate the nine-study training manifest with the established audit; no baseline model fitting is needed for training-manifest recovery:

```text
python -B /path/to/repository/r2_reproducibility/portable.py --config /work/local/core.json mp4-audit
```

The resulting training_manifest.tsv is under core_replay/mp4_initial/01_inputs. It preserves the original cMD-derived selection/order. Copy config.training.example.json outside the repository to /work/local/training.json, then edit source/output paths. source_root means the folder containing extracted/, not extracted/ itself.

Direction-context replay also retains the historical external Welch CLR-mean BH values, with the entire 800-feature family. Recover those tables through the established explicit MP4/multicohort stages:

```text
python -B /path/to/repository/r2_reproducibility/portable.py --config /work/local/core.json mp4-fit
python -B /path/to/repository/r2_reproducibility/portable.py --config /work/local/core.json multi-audit
python -B /path/to/repository/r2_reproducibility/portable.py --config /work/local/core.json multi-predict --cohort LiuNN_2022
python -B /path/to/repository/r2_reproducibility/portable.py --config /work/local/core.json multi-evaluate --cohort LiuNN_2022
python -B /path/to/repository/r2_reproducibility/portable.py --config /work/local/core.json multi-replicate --cohort LiuNN_2022
```

These baseline stages include model fitting; they are explicit optional upstream recovery actions, never started by this module. Repeat only the required bounded cohort stages for the other five names. The historical direction file is core_replay/mp4_multicohort/02_analysis/results/<cohort>/species_replication_all800.tsv. If already present, reuse verified inputs. Acquisition/full baseline fitting were not repeated for this extension. The narrower prediction/metric routes do not require historical direction tables.

## Explicit bounded workflow

Each invocation performs one action. Copy a config example into /work/local and give every action a distinct output_root.

1. audit loads only training inputs, checks the expected 1,257 samples (632 CRC,625 control) across nine studies and source labels, and exports metadata/QC/study inventory.
2. select --fold 0 selects on the eight training studies for the first held-out study in sorted order. Use indices 0..8 explicitly; there is no automatic nine-fold driver. select with no --fold performs full-training selection. Selection writes state.joblib, feature_evidence.tsv, direction_details.tsv, panel_membership.tsv and state_summary.json. The fitted-state fingerprint is unchanged from the source algorithm.
3. fit-model fits exactly one selected panel and learner. For a fold state, it predicts that held-out study without accepting its labels for fitting; evaluation labels are attached after prediction. For a full-training state, it emits one frozen model.joblib. Use config.fit.example.json, pointing state at the appropriate selection.
4. assemble requires all 18 distinct fit-model output directories and one identical selection-state hash. For a fold state, it writes the complete prediction grid plus metadata. For a full-training state, it writes bundle.joblib containing the 18 fixed models, state and training identity/vector guards.
5. external-predict --cohort <name> reads one profile matrix, transforms using the full-training state, and writes blind predictions for every source sample, including adenomas. It does not parse clinical metadata. Supply prior completed blind_dirs as prediction proceeds, so exact IDs/retained-coordinate vectors are checked across cohorts.
6. external-evaluate --cohort <name> requires all six sealed blind cohort directories from the same model lock before attaching outcomes. It retains that verified model SHA256 and a canonical fingerprint of the complete six-cohort blind receipt set, even if optional model_bundle is omitted. Adenomas are excluded. It evaluates exactly one cohort using fixed 2,000 paired class-stratified draws.
7. evaluate --fold <index> evaluates one complete internal held-out prediction grid. Its study_names must be the sorted nine names in the audit inventory. Predictions may be one fold or a complete saved table; matching metadata are supplied explicitly.
8. summary-internal requires nine completed evaluation_dirs; summary-external requires all six evaluations to share nonempty verified model-lock and complete-blind-set identities, with each receipt cohort matching its metric scope. Older external evaluation receipts lacking these identities are rejected. They aggregate fixed arrays without fitting/resampling, preserve all panels/learners and negative contrasts, and write macro/study/contrast tables plus all labelled probabilities.
9. membership uses state_dirs containing either the full-training selection or all nine fold selections. It outputs candidate counts, selected/eligible overlap, changed candidates, raw-median exclusion categories and training overlap. The nine-fold route also exports fold partition summaries.
10. directions --cohort <name> requires the full lock, all six blind directories and one historical_directions all800 table. It computes all four external directional summaries on fixed training members and the whole feature universe, checks shared historical coordinates, and transfers only historical CLR-mean q values. It never changes selection.
11. point-replay reads saved labelled predictions and metadata, validates the full 18-model sample grid and recomputes AUROC/AP without fitting or bootstrap draws.

Examples (set a separate fresh output_root for each):

```text
python -B -m prediction_extension --config /work/local/training.json audit
python -B -m prediction_extension --config /work/local/fold0.select.json select --fold 0
python -B -m prediction_extension --config /work/local/full.select.json select
python -B -m prediction_extension --config /work/local/ARP.elastic_net.fit.json fit-model --panel ARP --classifier elastic_net
python -B -m prediction_extension --config /work/local/assemble.json assemble
python -B -m prediction_extension --config /work/local/LiuNN.predict.json external-predict --cohort LiuNN_2022
python -B -m prediction_extension --config /work/local/LiuNN.evaluate.json external-evaluate --cohort LiuNN_2022
python -B -m prediction_extension --config /work/local/fold0.evaluate.json evaluate --fold 0
python -B -m prediction_extension --config /work/local/internal.summary.json summary-internal
python -B -m prediction_extension --config /work/local/external.summary.json summary-external
python -B -m prediction_extension --config /work/local/membership.json membership
python -B -m prediction_extension --config /work/local/LiuNN.directions.json directions --cohort LiuNN_2022
python -B -m prediction_extension --config /work/local/point.json point-replay
```

The complete fit grid is ARP plus DCP/ICP crossed with raw_mean,raw_median,clr_mean,clr_median; every panel is crossed with elastic_net and random_forest. config.assemble.example.json lists all 18 directories explicitly. Change state/output paths for outer-fold runs; do not reuse the full-training lock across folds. No missing panel is padded or silently omitted.

For summary configs, provide output_root, evaluation_dirs (nine internal or six external directories), and study_names for internal summaries. For a fold membership config, list the nine select output directories under state_dirs. See INPUT_OUTPUT.md for the other keys. Optional baseline_predictions on external-predict produces the original six-model raw-median replay check; recover that older grid using the documented multi-predict route. Absence does not assert a historical probability replay pass.

## Scientific and publication boundary

Primary external cohorts are LiuNN_2022,YangY_2021,IIGM_TU; secondary cohorts are YangJ_2020,NHSII,IIGM_CZ. All external comparisons remain post-outcome sensitivity. IIGM_CZ retains 59 CRC and 46 controls, including eligible polyp controls, and excludes 19 adenomas. NHSII excludes 435 adenomas. Forty-three training identities remain accession-unresolved aliases. Exact sample/vector checks do not establish universal participant independence.

Only code, tests, protocol, examples and aggregate input/result inventories belong in this module. Do not add local configs, individual sample tables/probabilities, abundance matrices, fitted models, caches or environments to a public export. Runtime receipts include local provenance paths and must be reviewed separately. The reporting/export companion is [report_extension](../report_extension/README.md). It owns Figure 8 and the S4 prediction-sheet TSV export. Author declarations, manuscript integration and external release metadata are separate.
