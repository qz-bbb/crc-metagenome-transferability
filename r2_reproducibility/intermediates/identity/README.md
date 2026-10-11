# Historical accession identity reconstruction

This offline generator reconstructs the dated accession enrichment used by the CRC identity sensitivity. It does not fit models, download data, infer hidden participants, or generate the broader clinical/identity audit summary.

The source transformation comes from `cohort_source_audit.py`, dated 2026-10-10, SHA-256 `de6c1de2b17bdd465a0c61a65d40264cc65e67c70591397379b925dd2164774c`. It joins exact uppercase/alphanumeric tokens only within the original Thomas and Wirbel projects. More than one matching original UID is an error; several distinct public runs for one UID are preserved. `identity_source_contract.json` lists source versions, official URLs and required hashes without individual source records.

Required local inputs are the reconstructed historical 1,395-row cMD metadata and both dated ENA project TSVs with their original provenance sidecars. The audit stage additionally needs the existing 1,257-row MP4 training inventory and the generated enrichment receipt. The exact supplied `identity_core.py` snapshot is the candidate portable version, SHA-256 `247f59b959bc4a491254bee4e411f8137604c4ae1add4fc2c46317d0c3f2c692`; its numerical functions are unchanged.

Paths in a config resolve relative to that config. Use separate fresh output directories for the two stages:

```text
python -B identity_sources.py --config config.enrich.example.json enrich
python -B identity_sources.py --config config.audit.example.json audit
```

Enrichment writes `05_review/original_enriched_identity_manifest.tsv` (373 x 7; 195 unique UIDs) and `enrichment_receipt.json`. Audit writes `results/training_identity_audit.tsv` (1,257 x 65), `results/training_manifest.tsv` (1,214 x 65), `results/excluded_training.tsv` (43 x 65), and `audit_receipt.json`. Historical TSVs explicitly use UTF-8 without BOM and CRLF so byte comparisons are portable. Original training order is retained; selected/excluded outputs preserve its respective subsequences.

Both stages refuse existing output and output/input overlap. The generating stages have no comparison-target input or fallback. The audit requires the supplied generated manifest to match its enrichment receipt. Keep generated tables and receipts local; source data and individual metadata are excluded from the slim code export.

The 43 excluded records remain unresolved Wirbel aliases. Accession linkage and reproducible tables do not prove universal participant independence. Full `identity_audit_summary.json` remains a separately sourced broader-audit dependency; it is not fabricated from this enrichment recipe. An integration hook may pass the generated enriched manifest directly to the existing identity preparation consumer.

Verification uses `python -B -m unittest -v test_identity_sources`. The separate local `compare_identity.py` accepts explicitly named comparison references; it is not part of generation. Export the generator, unchanged core, source contract, example configs, tests and this README; exclude local configs, generated data and local receipts/logs.
