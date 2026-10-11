"""Harmonize direction-denominator rules for the primary and CLR scales.

The primary analysis uses median relative-abundance differences; the CLR
sensitivity uses mean CLR differences. Both scales use the same eligible-study
universe and count an exact zero difference as discordant in this audit.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd


RUN = Path('UNCONFIGURED_OUTPUT')
OUT = RUN / "02_analysis" / "clr_direction_audit"
DATA = Path('UNCONFIGURED_INPUT')
META = DATA / "curated_crc_case_control_sample_metadata.tsv"
ABUND = Path('UNCONFIGURED_ABUNDANCE')
CLR = Path('UNCONFIGURED_CLR')
OLD_SET = Path('UNCONFIGURED_OLD_SET')


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def bh(values: np.ndarray) -> np.ndarray:
    out = np.full(values.shape, np.nan, dtype=float)
    ok = np.isfinite(values)
    if not ok.any():
        return out
    idx = np.where(ok)[0]
    order = idx[np.argsort(values[idx])]
    adj = values[order] * len(order) / np.arange(1, len(order) + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    out[order] = np.clip(adj, 0, 1)
    return out


def association_q(clr: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    import statsmodels.api as sm

    y = (meta["study_condition"].astype(str) == "CRC").astype(float)
    design = pd.concat([pd.DataFrame({"CRC_status": y}, index=meta.index),
                        pd.get_dummies(meta["study_name"].astype(str), prefix="study", drop_first=True, dtype=float)], axis=1)
    design = sm.add_constant(design, has_constant="add").astype(float)
    rows = []
    for taxon in clr.columns:
        fit = sm.OLS(clr[taxon].astype(float), design).fit(cov_type="HC3")
        rows.append({"taxon": taxon, "coef_crc": float(fit.params["CRC_status"]), "p_value": float(fit.pvalues["CRC_status"])})
    out = pd.DataFrame(rows)
    out["q_value_bh"] = bh(out["p_value"].to_numpy(float))
    return out


def direction_stats(values: pd.DataFrame, meta: pd.DataFrame, scale: str) -> pd.DataFrame:
    rows = []
    for taxon in values.columns:
        eligible = same = zero = 0
        for study in sorted(meta["study_name"].astype(str).unique()):
            idx = meta["study_name"].astype(str) == study
            sub = meta.loc[idx]
            if sub["study_condition"].astype(str).nunique() < 2:
                continue
            crc_idx = sub.index[sub["study_condition"].astype(str) == "CRC"]
            ctrl_idx = sub.index[sub["study_condition"].astype(str) != "CRC"]
            if scale == "relative_abundance_median":
                diff = float(values.loc[crc_idx, taxon].median() - values.loc[ctrl_idx, taxon].median())
            else:
                diff = float(values.loc[crc_idx, taxon].mean() - values.loc[ctrl_idx, taxon].mean())
            eligible += 1
            if diff == 0.0:
                zero += 1
            rows.append({"taxon": taxon, "study_name": study, "scale": scale, "difference": diff, "is_zero_difference": diff == 0.0})
        # taxon-level summary is written below from long rows
    long = pd.DataFrame(rows)
    return long


def summarise(long: pd.DataFrame, assoc: pd.DataFrame) -> pd.DataFrame:
    sign = assoc.set_index("taxon")["coef_crc"]
    records = []
    for (taxon, scale), g in long.groupby(["taxon", "scale"], sort=True):
        global_sign = np.sign(float(sign.loc[taxon]))
        same = int(((np.sign(g["difference"].to_numpy(float)) == global_sign) & (g["difference"].to_numpy(float) != 0)).sum())
        eligible = len(g)
        zero = int(g["is_zero_difference"].sum())
        records.append({"taxon": taxon, "scale": scale, "eligible_studies": eligible, "zero_difference_studies": zero,
                        "same_direction_studies": same, "discordant_studies": eligible - same,
                        "direction_fraction_same": same / eligible if eligible else np.nan,
                        "global_coefficient_sign": "CRC_higher" if global_sign > 0 else "control_higher" if global_sign < 0 else "zero"})
    return pd.DataFrame(records)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(META, sep="\t").set_index("sample_uid", drop=False)
    abundance = pd.read_csv(ABUND, sep="\t", index_col=0).loc[meta.index]
    clr = pd.read_csv(CLR, sep="\t", index_col=0).loc[meta.index]
    assoc = association_q(clr, meta)
    long = pd.concat([direction_stats(abundance, meta, "relative_abundance_median"), direction_stats(clr, meta, "clr_mean")], ignore_index=True)
    summary = summarise(long, assoc)
    merged = summary.merge(assoc, on="taxon", how="left")
    for threshold in (0.60, 0.70, 0.80):
        merged[f"selected_{threshold:.2f}"] = (merged["q_value_bh"] < 0.10) & (merged["direction_fraction_same"] >= threshold)
    merged.to_csv(OUT / "clr_direction_denominator_taxon_summary.tsv", sep="\t", index=False)
    long.to_csv(OUT / "clr_direction_denominator_by_study.tsv", sep="\t", index=False)
    old = pd.read_csv(OLD_SET, sep="\t")
    rows = []
    for scale in ("relative_abundance_median", "clr_mean"):
        g = merged[merged.scale == scale]
        for threshold in (0.60, 0.70, 0.80):
            selected = set(g.loc[g[f"selected_{threshold:.2f}"], "taxon"])
            rows.append({"scale": scale, "threshold": threshold, "candidate_count": len(selected), "zero_difference_total": int(g["zero_difference_studies"].sum()),
                         "eligible_studies_min": int(g["eligible_studies"].min()), "eligible_studies_max": int(g["eligible_studies"].max()),
                         "rule": "same eligible study universe; exact zero difference is discordant"})
    # Retrospective old CLR sensitivity sets are included only to document the denominator change.
    for _, row in old.iterrows():
        if str(row["panel_type"]).startswith("CLR-DCP"):
            rows.append({"scale": "old_clr_script_excluding_zero", "threshold": float(row["direction_threshold"]), "candidate_count": int(row["panel_size"]),
                         "zero_difference_total": np.nan, "eligible_studies_min": np.nan, "eligible_studies_max": np.nan,
                         "rule": "historical script excluded exact-zero studies from denominator"})
    comp = pd.DataFrame(rows)
    comp.to_csv(OUT / "clr_direction_denominator_comparison.tsv", sep="\t", index=False)
    receipt = {"status": "PASS", "n_samples": len(meta), "n_studies": int(meta.study_name.nunique()), "n_features": clr.shape[1],
               "inputs_sha256": {"metadata": sha256(META), "abundance": sha256(ABUND), "clr": sha256(CLR)},
               "direction_rule": "same eligible study universe; exact zero difference is discordant",
               "scales": {"relative_abundance_median": "within-study median CRC-control difference", "clr_mean": "within-study mean CLR CRC-control difference"},
               "outputs": [str(OUT / x) for x in ["clr_direction_denominator_taxon_summary.tsv", "clr_direction_denominator_by_study.tsv", "clr_direction_denominator_comparison.tsv"]]}
    (OUT / "clr_direction_audit_receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    (OUT / "clr_direction_audit_claim_boundary.md").write_text("""# CLR direction-denominator audit\n\nBoth effect scales use the same eligible study universe. An exact zero CRC-minus-control difference is counted as discordant rather than removed from the denominator. The primary relative-abundance median scale is retained for traceability; the CLR mean scale is a sensitivity definition. Candidate counts and membership differences are reported without treating one scale as intrinsically correct. Historical 119/100/67 CLR panels are retained only as a denominator-change audit and are not used as nested held-out validation.\n""", encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()

