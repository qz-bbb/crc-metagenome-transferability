"""Split the mixed B. fragilis/Bilophila/Enterobacteriaceae context set.

This is a taxonomy-defined context audit, not a functional or mechanistic
analysis. Each category is scored separately and compared with the original
composite on direction, random-effects heterogeneity, and score-only LOSO.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import chi2, norm
from sklearn.metrics import average_precision_score, roc_auc_score


RUN = Path('UNCONFIGURED_OUTPUT')
OUT = RUN / "02_analysis" / "context_set_audit"
DATA = Path('UNCONFIGURED_INPUT')
META = DATA / "curated_crc_case_control_sample_metadata.tsv"
CLR = Path('UNCONFIGURED_CLR')
MEMBERSHIP = Path('UNCONFIGURED_MEMBERSHIP')


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def dl_meta(diffs: list[float], ses: list[float]) -> dict[str, float]:
    d = np.asarray(diffs, dtype=float)
    v = np.maximum(np.asarray(ses, dtype=float) ** 2, 1e-12)
    k = len(d)
    w = 1.0 / v
    fixed = float(np.sum(w * d) / np.sum(w))
    q = float(np.sum(w * (d - fixed) ** 2))
    c = float(np.sum(w) - np.sum(w * w) / np.sum(w))
    tau2 = max(0.0, (q - (k - 1)) / c) if c > 0 else 0.0
    wr = 1.0 / (v + tau2)
    est = float(np.sum(wr * d) / np.sum(wr))
    se = float(np.sqrt(1.0 / np.sum(wr)))
    p = float(2 * norm.sf(abs(est / se))) if se > 0 else np.nan
    i2 = float(max(0.0, (q - (k - 1)) / q) * 100.0) if q > 0 else 0.0
    return {"k_studies": k, "random_effect": est, "random_se": se, "random_ci_low": est - 1.96 * se,
            "random_ci_high": est + 1.96 * se, "p_value": p, "tau2": tau2, "i2_percent": i2, "q_statistic": q}


def category_for(row: pd.Series) -> str:
    taxon = str(row.taxon)
    species = str(row.species)
    genus = str(row.genus)
    if species == "Bacteroides fragilis":
        return "B_fragilis"
    if genus == "Bilophila":
        return "Bilophila"
    if "f__Enterobacteriaceae" in taxon:
        return "Enterobacteriaceae"
    return "outside_split"


def score_association(scores: pd.Series, meta: pd.DataFrame) -> dict[str, float]:
    y = (meta.study_condition.astype(str) == "CRC").astype(float)
    design = pd.concat([pd.DataFrame({"CRC_status": y}, index=meta.index), pd.get_dummies(meta.study_name.astype(str), prefix="study", drop_first=True, dtype=float)], axis=1)
    fit = sm.OLS(scores.astype(float), sm.add_constant(design, has_constant="add").astype(float)).fit(cov_type="HC3")
    return {"coef_crc": float(fit.params["CRC_status"]), "p_value": float(fit.pvalues["CRC_status"]), "direction": "CRC_higher" if fit.params["CRC_status"] > 0 else "control_higher"}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(META, sep="\t").set_index("sample_uid", drop=False)
    clr = pd.read_csv(CLR, sep="\t", index_col=0).loc[meta.index]
    m = pd.read_csv(MEMBERSHIP)
    m = m[m.guild == "inflammation_Bacteroides_Enterobacteriaceae_panel"].copy()
    m["category"] = m.apply(category_for, axis=1)
    m = m[m.category != "outside_split"]
    categories = {c: sorted(m.loc[m.category == c, "taxon"].astype(str)) for c in ["B_fragilis", "Bilophila", "Enterobacteriaceae"]}
    categories["composite"] = sorted(set().union(*categories.values()))
    # Per-taxon direction and heterogeneity within the selected composite members.
    taxon_rows = []
    for taxon in categories["composite"]:
        diffs, ses = [], []
        for study in sorted(meta.study_name.astype(str).unique()):
            sub = meta[meta.study_name.astype(str) == study]
            crc = clr.loc[sub.index[sub.study_condition.astype(str) == "CRC"], taxon].to_numpy(float)
            ctrl = clr.loc[sub.index[sub.study_condition.astype(str) != "CRC"], taxon].to_numpy(float)
            if len(crc) == 0 or len(ctrl) == 0:
                continue
            diffs.append(float(crc.mean() - ctrl.mean()))
            ses.append(float(np.sqrt(np.var(crc, ddof=1) / len(crc) + np.var(ctrl, ddof=1) / len(ctrl))))
        assoc = score_association(clr[taxon], meta)
        taxon_rows.append({"taxon": taxon, **{"category": next(c for c, ts in categories.items() if taxon in ts and c != "composite")}, **assoc, **dl_meta(diffs, ses), "n_taxa_in_composite": len(categories["composite"])})
    taxon_df = pd.DataFrame(taxon_rows)
    # Full-data category summaries and strict score-only LOSO.
    summary_rows, loso_rows = [], []
    studies = sorted(meta.study_name.astype(str).unique())
    for category, taxa in categories.items():
        score = clr[taxa].mean(axis=1)
        assoc = score_association(score, meta)
        diffs, ses = [], []
        for study in studies:
            sub = meta[meta.study_name.astype(str) == study]
            crc = score.loc[sub.index[sub.study_condition.astype(str) == "CRC"]].to_numpy(float)
            ctrl = score.loc[sub.index[sub.study_condition.astype(str) != "CRC"]].to_numpy(float)
            diffs.append(float(crc.mean() - ctrl.mean()))
            ses.append(float(np.sqrt(np.var(crc, ddof=1) / len(crc) + np.var(ctrl, ddof=1) / len(ctrl))))
        summary_rows.append({"category": category, "n_taxa": len(taxa), "taxa": ";".join(taxa), **assoc, **dl_meta(diffs, ses), "orientation_rule": "full-data coefficient sign for context annotation; LOSO uses training-fold sign"})
        for fold_index, held_out in enumerate(studies, start=1):
            train = meta.study_name.astype(str) != held_out
            test = ~train
            train_assoc = score_association(score.loc[train], meta.loc[train])
            oriented = score if train_assoc["coef_crc"] > 0 else -score
            y = (meta.loc[test, "study_condition"].astype(str) == "CRC").astype(int).to_numpy()
            p = oriented.loc[test].to_numpy(float)
            loso_rows.append({"category": category, "held_out_study": held_out, "fold_index": fold_index, "n_taxa": len(taxa), "training_orientation": train_assoc["direction"], "auroc": roc_auc_score(y, p), "average_precision": average_precision_score(y, p), "n_test": len(y), "n_crc_test": int(y.sum()), "n_control_test": int((1-y).sum()), "interpretation": "score-only LOSO stress test; not clinical validation"})
    summary = pd.DataFrame(summary_rows)
    loso = pd.DataFrame(loso_rows)
    summary["mean_loso_auroc"] = loso.groupby("category").auroc.mean().reindex(summary.category).to_numpy()
    summary["median_loso_auroc"] = loso.groupby("category").auroc.median().reindex(summary.category).to_numpy()
    summary["mean_loso_average_precision"] = loso.groupby("category").average_precision.mean().reindex(summary.category).to_numpy()
    summary["median_loso_average_precision"] = loso.groupby("category").average_precision.median().reindex(summary.category).to_numpy()
    m.to_csv(OUT / "context_set_split_membership.tsv", sep="\t", index=False)
    taxon_df.to_csv(OUT / "context_set_split_taxon_evidence.tsv", sep="\t", index=False)
    summary.to_csv(OUT / "context_set_split_summary.tsv", sep="\t", index=False)
    loso.to_csv(OUT / "context_set_split_loso.tsv", sep="\t", index=False)
    receipt = {"status": "PASS", "n_samples": len(meta), "n_studies": len(studies), "categories": {k: len(v) for k, v in categories.items()}, "inputs_sha256": {"metadata": sha256(META), "clr": sha256(CLR), "membership": sha256(MEMBERSHIP)}, "rule": "B. fragilis, Bilophila, and Enterobacteriaceae are analyzed separately and compared with the original composite; taxonomy-defined context only", "outputs": [str(OUT / x) for x in ["context_set_split_membership.tsv", "context_set_split_taxon_evidence.tsv", "context_set_split_summary.tsv", "context_set_split_loso.tsv"]]}
    (OUT / "context_set_split_receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    (OUT / "context_set_split_claim_boundary.md").write_text("""# Mixed taxonomy context-set split\n\nThe former Bacteroides fragilis/Bilophila/Enterobacteriaceae composite is split into B. fragilis, Bilophila, and Enterobacteriaceae categories and compared with the composite. The outputs quantify direction, DerSimonian-Laird heterogeneity, and score-only LOSO. These are taxonomy-defined context annotations; they do not establish shared function, inflammation, metabolite activity, oral source, strain behavior, or clinical utility.\n""", encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()

