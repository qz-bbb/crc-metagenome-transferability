# Historical intermediate recovery

The generator reconstructs the five historical input dependencies consumed by the portable historical CLR, taxonomy-context, and RF stages. It uses independently sourced cMD abundance/metadata, the two original MicrobiomeHD source tables, and the six bounded historical algorithm snapshots in `source_algorithms/`.

Use the existing numerical environment recorded by the main reproducibility package. No installation is performed by this tool.

```text
python -B -m unittest test_historical_intermediates -v
python -B historical_intermediates.py --config config.example.json matrices
python -B historical_intermediates.py --config config.example.json panels
python -B historical_intermediates.py --config config.example.json full
```

Each invocation requires a fresh output root; configure a different directory for another stage. `matrices` needs only baseline matrix and metadata. `panels` additionally needs the source snapshots and regenerates guild membership, old CLR sets and the two RF-consumed interpretation fields. Its explicitly named `rf_consumed_fields_NOT_FULL_INTERPRETATION.tsv` is not a full interpretation table. `full` also needs MicrobiomeHD q-values/effects and produces the complete 29-row, 21-column interpretation.

Outputs include `output_mapping.json`, `generation_receipt.json` with source/output SHA-256 and versions, and optional `comparisons.json`. An optional `targets` mapping supplies independent post-generation checks. Targets are never available in the computation context and are not used as their own upstream. Missing comparison targets are explicitly reported as failures while preserving the generated artifacts and receipt.

For the portable consumer config, use only `legacy_clr`, `legacy_abundance`, `legacy_membership`, `legacy_interpretation` and `old_clr_sets` entries from the output mapping, resolved relative to its generated output directory. The mapping also contains supporting tables, which are not extra portable configuration keys.

Historical semantics are retained: metadata/sample order and original species order; prevalence >= 5% and presence in >= 3 studies; percent abundance without renormalization; minimum retained positive value / 2 pseudocount; original OLS-HC3, BH, direction denominators, LOCO, random-effects, and model parameters. Original TSV write/read boundaries are replayed in memory with the default pandas parser. This is necessary for exact historical bytes and does not round to a target or alter tolerances.

The source snapshot `main` in `run_analysis.py` contains only selected statement intervals and is not an executable entrypoint. The generator never calls historical source `main()`. All source helper function ASTs are unchanged; private global paths and unrelated analyses are excluded.

For a slim code export, include this module's code, tests, example config, `generation_contract.json`, the six `source_algorithms/*.py` files, and `algorithm_provenance.json`. This integrated distribution also includes a sanitized `source_algorithms/algorithm_source_manifest.json` describing extraction checks without private paths; the unsanitized internal original must not be exported. Exclude local configs, generated directories, consumer outputs and internal reports. `export_algorithms.py` is an internal maintenance utility, not a required run dependency.
