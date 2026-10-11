# Exact ENA project-report snapshots — retained 2026-10-10

These are unchanged public ENA project-report snapshots retained on 2026-10-10 for PRJNA447983 and PRJEB27928, accompanied by their original retrieval provenance and SHA-256 manifest. Their inclusion in the R2 release was explicitly authorized on 2026-10-11. They remain separate from the MIT software directory and from privately held source data. No raw sequencing reads are included.

Purpose: permit explicit exact-byte recovery of the historical identity-enrichment dependency when the live API changes ordering or is unavailable. This package is not itself independent biological evidence and does not regenerate the broad historical identity audit.

The current PRJNA447983 endpoint returned the same 140-by-9 records in different row order. PRJEB27928 returned errors/timeouts during the bounded check. Do not silently substitute current API bytes for these pinned historical inputs.

The original `.provenance.json` files record their actual historical retrieval, not the current preparation date. The recovery helper creates a separate fresh receipt and labels this workflow `local_verified`.

Sources: [PRJNA447983](https://www.ebi.ac.uk/ena/browser/view/PRJNA447983) and [PRJEB27928](https://www.ebi.ac.uk/ena/browser/view/PRJEB27928). Exact requested fields, source API URLs and original retrieval metadata are retained in the sidecars and in `r2_reproducibility/m7_recovery/m7_source_inventory.json`.

Source attribution and conditions: [ENA/INSDC policy](https://ena-browser-docs.readthedocs.io/en/latest/about/policies.html) and [EMBL-EBI terms](https://www.ebi.ac.uk/about/terms-of-use/). Source-owner and resource-specific conditions still apply. The project MIT software licence does not relicense these third-party records; no legal certification is implied.

From the sibling `r2_reproducibility/` directory of a full repository/archive extraction, run:

```text
python -B m7_recovery/recover_m7_sources.py --root ../../recovered_inputs/ena_frozen --family ena --local-dir ../source_snapshots/ena_20261010
```

Use a fresh destination outside the repository. Do not sort records or edit expected hashes to force a match.
