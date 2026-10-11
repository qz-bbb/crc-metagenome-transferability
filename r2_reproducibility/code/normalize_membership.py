#!/usr/bin/env python
"""Create an unambiguous long-form panel-membership audit table.

The first full run predates the ``||`` separator in the wide membership file;
taxonomy strings themselves contain single pipe characters, so this script
parses both the legacy and current representation without changing analysis
results.
"""
from __future__ import annotations
import re
from pathlib import Path
import pandas as pd

OUT = Path('UNCONFIGURED_OUTPUT')
SRC = OUT / "nested_loso_panel_membership.tsv"
DST = OUT / "nested_loso_panel_membership_long.tsv"

def split_taxa(value: str) -> list[str]:
    if not isinstance(value, str) or not value:
        return []
    if "||" in value:
        return [x.rstrip("|") for x in value.split("||") if x]
    # Legacy wide output: every taxon starts with the same kingdom prefix.
    return [x.rstrip("|") for x in re.split(r"(?=k__[^|]+\|)", value) if x]

def main() -> None:
    df = pd.read_csv(SRC, sep="\t")
    rows = []
    for _, r in df.iterrows():
        taxa = split_taxa(r.get("taxa", ""))
        for rank, taxon in enumerate(taxa, start=1):
            rows.append({
                "held_out_study": r["held_out_study"],
                "panel": r["panel"],
                "panel_size_reported": int(r["panel_size"]),
                "panel_rank": rank,
                "taxon": taxon,
                "selected_features": int(r["selected_features"]),
            })
    long = pd.DataFrame(rows)
    long.to_csv(DST, sep="\t", index=False)
    expected = df["panel_size"].astype(int).sum()
    observed = len(long)
    if expected != observed:
        raise SystemExit(f"membership row mismatch: expected {expected}, observed {observed}")
    print(f"PASS membership rows={observed} panels={len(df)}")

if __name__ == "__main__":
    main()

