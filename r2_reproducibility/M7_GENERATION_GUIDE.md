# M7 historical intermediate generation

Scope: reconstruct the named missing dependencies, not select new scientific rules. All commands below are explicit single-stage invocations. They do not publish, download raw reads, or automatically run a full model suite. Keep data, local configs and fresh outputs outside this code directory. Relative paths resolve from the config file; after copying an example, update its input, output and algorithm paths accordingly. Do not replace a pinned source contract to make a changed dataset pass.

## Execution and dependency order

| Stage | Required upstream evidence | Generated dependency | Existing consumer |
| --- | --- | --- | --- |
| historical-matrices | Reconstructed 1,395-sample cMD metadata and 944-species source matrix | Filtered percent abundance and CLR, each 1,395 x 301 | Historical CLR/context/RF stages |
| historical-panels | Same sources and shipped algorithm snapshots | Matrices, membership, old CLR sets, partial RF-consumed fields | Partial preparation only; not the full interpretation table |
| historical-full | Same sources plus the two exact MicrobiomeHD q/effect tables | Five complete historical dependencies and supporting tables | portable.py historical-clr / historical-context / historical-rf |
| identity-enrich | Historical cMD metadata, two dated ENA TSVs and their provenance | Enriched identity manifest and generation receipt | identity-audit or closures.py identity-prepare |
| identity-audit | Generated enrichment and receipt; MP4 training inventory | Full, accession-resolved and excluded training lists | Audit only; does not refit identity models |
| sparsity | Locked MP4 features/panels/evidence, training/cohort inventory, same-MP4 profiles and existing species-replication tables | Eight descriptive sparsity tables | Four-definition decomposition via prior_sparsity |

1. Recover the official processed cMD/MP4 sources using the existing source_access.py routes. Prior source-access verification is in SOURCE_ACCESS.md; the present extension did not redownload them.
2. Configure intermediates/historical/config.example.json outside the package. For the complete historical consumer chain run only historical-full; it includes matrix generation. Matrices/panels are optional bounded diagnostic alternatives, each requiring its own fresh destination.

```text
python -B prepare_intermediates.py --config ../local/historical.json historical-full
```

Set the core config's legacy_abundance, legacy_clr, legacy_membership, legacy_interpretation and old_clr_sets to the files named by output_mapping.json. These are the only five additional keys; supporting table keys are not portable.py config keys.

```text
python -B portable.py --config ../local/core.json historical-clr
python -B portable.py --config ../local/core.json historical-context
```

The full generator runs the original association, LOCO, covariate, heterogeneity and bounded historical logistic-weight calculations needed for interpretation. It does not run the historical full Random Forest, nested model suite or bootstrap intervals. Source functions and original TSV parse/serialize boundaries are preserved; this is not rounding a new result to match an old result. generation_contract.json checks source versions before computation and before PASS. Only the baseline matrix may use its exact canonical-value hash instead of its compressed-file hash. Generated legacy matrices also pass exact canonical-value consumer checks; the other three target inputs retain exact byte hashes.

3. Generate MP4 training/model and multicohort artifacts using the existing explicit core stages when they are not already available. Those stages are documented in README.md; they were not fully refitted in this extension. Do not start all fits merely to test a generator.
4. Configure the two identity examples with separate outputs, then run:

```text
python -B prepare_intermediates.py --config ../local/identity.enrich.json identity-enrich
python -B prepare_intermediates.py --config ../local/identity.audit.json identity-audit
```

For the established closure consumer, set optional identity_enriched_manifest to the new manifest, while retaining the other baseline/multicohort inputs. Its original manifest SHA is still enforced. Run closures.py identity-prepare to verify the actual consumer. The broader identity_audit_summary.json remains a separately sourced historical audit checkpoint; narrow enrichment is not a substitute for it.

5. Configure intermediates/sparsity/config.example.json, then run:

```text
python -B prepare_intermediates.py --config ../local/sparsity.json sparsity
python -B definition_decomposition/run_definitions.py --config ../local/definitions.json
```

Set prior_sparsity in the definition config to the new sparsity output. The sparsity stage reuses locked training evidence and pre-existing external species-replication q values, checks the raw/CLR effects against those endpoints, and does not fit new models or compute new inferential tests. Source non-detection is zero-filled only within the same MetaPhlAn4 feature representation; unresolved MP3/MP4 species identity is never converted to zero.

## Outputs and verification

Each generator refuses unsafe overlaps or completed output overwrites and records hashes/receipts. Use separate fresh directories for reruns. Source files must remain immutable throughout execution: identity and sparsity record their hash baseline after some initial parsing, so these receipts are not a transaction-safe snapshot of concurrently edited files. The reviewed historical sources remained unchanged and their exact outputs were independently compared. Optional historical comparison targets are post-generation test oracles, not generation inputs; failed or missing comparisons return nonzero. Identity matching is exact normalized, project-scoped and ambiguity-rejecting; multiple distinct public runs are retained. No comparison target enters identity or sparsity generation.

M7_VALIDATION.md and m7_validation_summary.json report actual execution. File manifest.tsv covers the clean code/documents except itself. Runtime receipt paths are local by design and must not be included in a public export without review. Initial historical/sparsity RED test outputs were observed but not locally retained; later fix RED/GREEN logs and real-run comparisons are retained internally. No reconstructed historical test log is presented as contemporaneous.

## Source-access boundary and v2.0.0 update

The dated ENA inputs described below are now preserved separately at `../source_snapshots/ena_20261010/` in the full tagged repository archive. They remain excluded from the code-only ZIP. Follow the current `RELEASE_PREPARATION.md` import instructions; original sidecars and exact hashes are unchanged. The following paragraph records the earlier source-recovery checkpoint, not the later release status. Publication does not supply author scientific sign-off or make every model/legacy artifact a fully replayed workflow.

The slim archive deliberately contains no individual sample matrices/identifiers, predictions, models, environments or source-data records. In the subsequent source-recovery checkpoint, both required MicrobiomeHD files were freshly recovered from a commit-pinned author repository and matched the historical bytes. `m7_recovery/m7_source_inventory.json` and `RELEASE_PREPARATION.md` supply the recovery route. Accession enrichment still requires the exact dated ENA snapshots and original provenance sidecars. A current API response differed in row order and another failed; neither was adopted as an exact historical input. Explicit local import verifies the historical snapshots, preserves the shipped original sidecars and emits a separate fresh receipt. The broader identity audit summary is now included as an unchanged aggregate checkpoint under `m7_recovery/provenance/`, explicitly not regenerated by the helper. The package alone is still not an all-input public from-zero archive: the dated ENA source records are not embedded. A public R2 release and author scientific sign-off remain separate steps.
