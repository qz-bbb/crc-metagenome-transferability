from __future__ import annotations

import json
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests


warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.linear_model._logistic")


RUN_DIR = Path(r"D:\1\projects\gut_microbiome_colorectal_cancer\runs\run_20260525_182306")
W5 = RUN_DIR / "agents" / "05-analysis" / "workspace"
DATA_DIR = Path(r"D:\1\Knowledge Base\true_raw\data\BIOC_curatedMetagenomicData_CRC")
META_PATH = DATA_DIR / "curated_crc_case_control_sample_metadata.tsv"
MATRIX_PATH = DATA_DIR / "curated_crc_case_control_species_relative_abundance.tsv.gz"
FEATURE_PATH = DATA_DIR / "curated_crc_species_feature_index.tsv"
CLR_PATH = W5 / "processed_data" / "B" / "filtered_species_clr.tsv.gz"
ASSOC_PATH = W5 / "results" / "C" / "cohort_adjusted_crc_control_association.tsv"
OUT_K = W5 / "results" / "K_ecology_oral_signature"
OUT_L = W5 / "results" / "L_pairwise_transfer"
SCRIPT_PATH = W5 / "scripts" / "build_ecology_transfer_analysis.py"


ORAL_ASSOCIATED_GENERA = {
    "Actinomyces",
    "Atopobium",
    "Campylobacter",
    "Dialister",
    "Fusobacterium",
    "Gemella",
    "Granulicatella",
    "Leptotrichia",
    "Parvimonas",
    "Peptostreptococcus",
    "Porphyromonas",
    "Prevotella",
    "Selenomonas",
    "Solobacterium",
    "Streptococcus",
    "Treponema",
    "Veillonella",
}


def ensure_dirs() -> None:
    OUT_K.mkdir(parents=True, exist_ok=True)
    OUT_L.mkdir(parents=True, exist_ok=True)


def write_tsv(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=False)
    return path


def genus_from_taxon(taxon: str) -> str:
    for part in str(taxon).replace(";", "|").split("|"):
        if part.startswith("g__"):
            return part.replace("g__", "")
    return ""


def species_from_taxon(taxon: str) -> str:
    for part in str(taxon).replace(";", "|").split("|"):
        if part.startswith("s__"):
            return part.replace("s__", "").replace("_", " ")
    return ""


def bh(values: list[float]) -> list[float]:
    arr = np.asarray(values, dtype=float)
    out = np.full(arr.shape, np.nan, dtype=float)
    mask = np.isfinite(arr)
    if mask.any():
        out[mask] = multipletests(arr[mask], method="fdr_bh")[1]
    return out.tolist()


def hedges_g(case: np.ndarray, control: np.ndarray) -> tuple[float, float, float, int, int]:
    case = case[np.isfinite(case)]
    control = control[np.isfinite(control)]
    n1 = len(case)
    n0 = len(control)
    if n1 < 3 or n0 < 3:
        return np.nan, np.nan, np.nan, n1, n0
    s1 = float(np.var(case, ddof=1))
    s0 = float(np.var(control, ddof=1))
    pooled_var = ((n1 - 1) * s1 + (n0 - 1) * s0) / (n1 + n0 - 2)
    if pooled_var <= 0:
        return np.nan, np.nan, np.nan, n1, n0
    cohen_d = (float(np.mean(case)) - float(np.mean(control))) / math.sqrt(pooled_var)
    correction = 1 - (3 / (4 * (n1 + n0) - 9))
    g = cohen_d * correction
    se = math.sqrt((n1 + n0) / (n1 * n0) + (g * g) / (2 * (n1 + n0 - 2)))
    p = 2 * stats.norm.sf(abs(g / se)) if se > 0 else np.nan
    return float(g), float(se), float(p), n1, n0


def random_effects(effects: pd.DataFrame) -> dict[str, float]:
    valid = effects[np.isfinite(effects["effect"]) & np.isfinite(effects["se"]) & (effects["se"] > 0)].copy()
    k = int(len(valid))
    if k < 2:
        return {
            "k_cohorts": k,
            "random_effect": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "se_random": np.nan,
            "z": np.nan,
            "pvalue": np.nan,
            "tau2": np.nan,
            "i2_percent": np.nan,
        }
    yi = valid["effect"].to_numpy(float)
    vi = np.square(valid["se"].to_numpy(float))
    wi = 1 / vi
    fixed = float(np.sum(wi * yi) / np.sum(wi))
    q = float(np.sum(wi * np.square(yi - fixed)))
    c = float(np.sum(wi) - (np.sum(np.square(wi)) / np.sum(wi)))
    tau2 = max(0.0, (q - (k - 1)) / c) if c > 0 else 0.0
    w_star = 1 / (vi + tau2)
    pooled = float(np.sum(w_star * yi) / np.sum(w_star))
    se = math.sqrt(float(1 / np.sum(w_star)))
    z = pooled / se if se > 0 else np.nan
    p = 2 * stats.norm.sf(abs(z)) if np.isfinite(z) else np.nan
    i2 = max(0.0, (q - (k - 1)) / q) * 100 if q > 0 else 0.0
    return {
        "k_cohorts": k,
        "random_effect": pooled,
        "ci_low": pooled - 1.96 * se,
        "ci_high": pooled + 1.96 * se,
        "se_random": se,
        "z": float(z),
        "pvalue": float(p),
        "tau2": float(tau2),
        "i2_percent": float(i2),
    }


def make_model() -> object:
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            l1_ratio=0.5,
            C=0.5,
            class_weight="balanced",
            max_iter=4000,
            random_state=42,
        ),
    )


def safe_auc(y_true: np.ndarray, score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return np.nan
    return float(roc_auc_score(y_true, score))


def safe_ap(y_true: np.ndarray, score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return np.nan
    return float(average_precision_score(y_true, score))


def main() -> None:
    ensure_dirs()
    metadata = pd.read_csv(META_PATH, sep="\t")
    matrix = pd.read_csv(MATRIX_PATH, sep="\t", index_col=0).T
    matrix = matrix.loc[metadata["sample_uid"]]
    feature_index = pd.read_csv(FEATURE_PATH, sep="\t")
    clr = pd.read_csv(CLR_PATH, sep="\t", index_col=0).loc[metadata["sample_uid"]]
    assoc = pd.read_csv(ASSOC_PATH, sep="\t")

    y = (metadata["study_condition"].to_numpy() == "CRC").astype(int)
    study = metadata["study_name"].to_numpy()

    normalized = matrix.div(matrix.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    positive = normalized.where(normalized > 0, np.nan)
    shannon = -(positive * np.log(positive)).sum(axis=1).fillna(0)
    species_richness = (matrix > 0).sum(axis=1)

    taxa_meta = pd.DataFrame({
        "taxon_long": matrix.columns,
        "genus": [genus_from_taxon(x) for x in matrix.columns],
        "species_short": [species_from_taxon(x) for x in matrix.columns],
    })
    taxa_meta["oral_associated_panel"] = taxa_meta["genus"].isin(ORAL_ASSOCIATED_GENERA)
    oral_taxa = taxa_meta.loc[taxa_meta["oral_associated_panel"], "taxon_long"].tolist()
    oral_abundance_sum = matrix[oral_taxa].sum(axis=1) if oral_taxa else pd.Series(0.0, index=matrix.index)
    oral_log_abundance = np.log10(oral_abundance_sum + 1e-5)
    oral_species_richness = (matrix[oral_taxa] > 0).sum(axis=1) if oral_taxa else pd.Series(0, index=matrix.index)

    sample_scores = pd.DataFrame({
        "sample_uid": metadata["sample_uid"],
        "study_name": metadata["study_name"],
        "condition": metadata["study_condition"],
        "condition_crc": y,
        "shannon_all_species": shannon.to_numpy(float),
        "species_richness_all": species_richness.to_numpy(int),
        "oral_associated_abundance_percent": oral_abundance_sum.to_numpy(float),
        "log10_oral_associated_abundance_percent": oral_log_abundance.to_numpy(float),
        "oral_associated_species_richness": oral_species_richness.to_numpy(int),
    })
    write_tsv(sample_scores, OUT_K / "sample_ecology_scores.tsv")

    panel = taxa_meta[taxa_meta["oral_associated_panel"]].copy()
    panel["overall_prevalence"] = (matrix[panel["taxon_long"]] > 0).mean(axis=0).to_numpy(float) if not panel.empty else []
    panel["mean_abundance_percent"] = matrix[panel["taxon_long"]].mean(axis=0).to_numpy(float) if not panel.empty else []
    write_tsv(panel, OUT_K / "oral_associated_taxa_panel.tsv")

    metric_specs = [
        ("shannon_all_species", "Shannon diversity, all species"),
        ("species_richness_all", "Observed species richness"),
        ("log10_oral_associated_abundance_percent", "log10 oral-associated abundance score"),
        ("oral_associated_species_richness", "Oral-associated species richness"),
    ]
    cohort_rows: list[dict[str, object]] = []
    for metric, label in metric_specs:
        for cohort in sorted(sample_scores["study_name"].unique()):
            subset = sample_scores[sample_scores["study_name"] == cohort]
            case = subset.loc[subset["condition"] == "CRC", metric].to_numpy(float)
            control = subset.loc[subset["condition"] == "control", metric].to_numpy(float)
            effect, se, p, n_crc, n_control = hedges_g(case, control)
            cohort_rows.append({
                "metric": metric,
                "metric_label": label,
                "cohort": cohort,
                "effect": effect,
                "se": se,
                "pvalue": p,
                "n_crc": n_crc,
                "n_control": n_control,
                "mean_crc": float(np.nanmean(case)) if len(case) else np.nan,
                "mean_control": float(np.nanmean(control)) if len(control) else np.nan,
            })
    cohort_effects = pd.DataFrame(cohort_rows)
    write_tsv(cohort_effects, OUT_K / "per_cohort_metric_effects.tsv")

    meta_rows: list[dict[str, object]] = []
    for metric, label in metric_specs:
        re_out = random_effects(cohort_effects[cohort_effects["metric"] == metric])
        re_out.update({"metric": metric, "metric_label": label})
        meta_rows.append(re_out)
    metric_meta = pd.DataFrame(meta_rows)
    metric_meta["qvalue_bh"] = bh(metric_meta["pvalue"].tolist())
    metric_meta = metric_meta[[
        "metric",
        "metric_label",
        "k_cohorts",
        "random_effect",
        "ci_low",
        "ci_high",
        "se_random",
        "z",
        "pvalue",
        "qvalue_bh",
        "tau2",
        "i2_percent",
    ]]
    write_tsv(metric_meta, OUT_K / "random_effects_metric_meta.tsv")

    oral_genera = set(panel["genus"]) if not panel.empty else set()
    assoc_oral = assoc[assoc["genus"].isin(oral_genera)].copy()
    assoc_oral = assoc_oral.sort_values("qvalue_bh").head(40)
    write_tsv(assoc_oral, OUT_K / "top_oral_associated_taxa_associations.tsv")

    x = clr.to_numpy(float)
    cohorts = sorted(pd.Series(study).unique())
    perf_rows: list[dict[str, object]] = []
    for train_cohort in cohorts:
        train_idx = np.where(study == train_cohort)[0]
        y_train = y[train_idx]
        if len(np.unique(y_train)) < 2:
            continue
        for test_cohort in cohorts:
            test_idx = np.where(study == test_cohort)[0]
            y_test = y[test_idx]
            if len(np.unique(y_test)) < 2:
                status = "SKIPPED_ONE_CLASS_TEST"
                score = np.full(len(test_idx), np.nan)
            elif train_cohort == test_cohort:
                min_class = int(np.min(np.bincount(y_train)))
                n_splits = max(3, min(5, min_class))
                cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
                try:
                    score = cross_val_predict(make_model(), x[train_idx], y_train, cv=cv, method="predict_proba")[:, 1]
                    status = "WITHIN_COHORT_CV"
                except Exception:
                    score = np.full(len(test_idx), np.nan)
                    status = "FAILED"
            else:
                try:
                    model = make_model()
                    model.fit(x[train_idx], y_train)
                    score = model.predict_proba(x[test_idx])[:, 1]
                    status = "TRAIN_ONE_TEST_OTHER"
                except Exception:
                    score = np.full(len(test_idx), np.nan)
                    status = "FAILED"
            auroc = safe_auc(y_test, score) if np.isfinite(score).any() else np.nan
            ap = safe_ap(y_test, score) if np.isfinite(score).any() else np.nan
            perf_rows.append({
                "train_cohort": train_cohort,
                "test_cohort": test_cohort,
                "status": status,
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                "train_crc": int(y_train.sum()),
                "train_control": int(len(y_train) - y_train.sum()),
                "test_crc": int(y_test.sum()),
                "test_control": int(len(y_test) - y_test.sum()),
                "auroc": auroc,
                "average_precision": ap,
            })
    perf = pd.DataFrame(perf_rows)
    write_tsv(perf, OUT_L / "pairwise_transfer_performance.tsv")
    matrix_auc = perf.pivot(index="train_cohort", columns="test_cohort", values="auroc").reset_index()
    write_tsv(matrix_auc, OUT_L / "pairwise_transfer_auroc_matrix.tsv")

    offdiag = perf[(perf["train_cohort"] != perf["test_cohort"]) & np.isfinite(perf["auroc"])]
    diag = perf[(perf["train_cohort"] == perf["test_cohort"]) & np.isfinite(perf["auroc"])]
    summary_rows = [
        {"metric": "oral_associated_taxa_panel_n", "value": len(panel)},
        {"metric": "oral_associated_genera_present_n", "value": len(oral_genera)},
        {"metric": "metric_meta_q_lt_0_10", "value": int((metric_meta["qvalue_bh"] < 0.10).sum())},
        {"metric": "oral_abundance_random_effect", "value": float(metric_meta.loc[metric_meta["metric"] == "log10_oral_associated_abundance_percent", "random_effect"].iloc[0])},
        {"metric": "oral_abundance_qvalue", "value": float(metric_meta.loc[metric_meta["metric"] == "log10_oral_associated_abundance_percent", "qvalue_bh"].iloc[0])},
        {"metric": "oral_abundance_i2_percent", "value": float(metric_meta.loc[metric_meta["metric"] == "log10_oral_associated_abundance_percent", "i2_percent"].iloc[0])},
        {"metric": "pairwise_within_cohort_median_auroc", "value": float(diag["auroc"].median())},
        {"metric": "pairwise_off_diagonal_median_auroc", "value": float(offdiag["auroc"].median())},
        {"metric": "pairwise_off_diagonal_min_auroc", "value": float(offdiag["auroc"].min())},
        {"metric": "pairwise_off_diagonal_max_auroc", "value": float(offdiag["auroc"].max())},
        {"metric": "pairwise_off_diagonal_pairs", "value": int(len(offdiag))},
        {"metric": "pairwise_off_diagonal_above_0_5", "value": int((offdiag["auroc"] > 0.5).sum())},
    ]
    summary_df = pd.DataFrame(summary_rows)
    write_tsv(summary_df, OUT_K / "ecology_transfer_summary.tsv")
    write_tsv(summary_df, OUT_L / "ecology_transfer_summary.tsv")

    report = [
        "# Ecology, Oral-Associated Score, and Pairwise Transfer Report",
        "",
        f"- Oral-associated genus panel present in matrix: {len(oral_genera)} genera / {len(panel)} species features.",
        "- Oral-associated scores are literature-prior, taxonomy-name-based summaries; they are not an independently inferred oral-source model.",
        "- Ecological metric effects use within-cohort Hedges g and DerSimonian-Laird random-effects meta-analysis.",
        f"- Oral abundance score random effect: {summary_rows[3]['value']:.3f}; q={summary_rows[4]['value']:.3g}; I2={summary_rows[5]['value']:.1f}%.",
        f"- Pairwise train-study x test-study transfer median off-diagonal AUROC: {summary_rows[7]['value']:.3f}; min={summary_rows[8]['value']:.3f}; max={summary_rows[9]['value']:.3f}.",
        "- Pairwise transfer is a cohort-dependence stress test and is not diagnostic validation.",
    ]
    (OUT_K / "ecology_transfer_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    (OUT_L / "ecology_transfer_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    print(json.dumps({
        "oral_taxa_features": len(panel),
        "oral_genera": len(oral_genera),
        "metric_meta_q_lt_0_10": int((metric_meta["qvalue_bh"] < 0.10).sum()),
        "pairwise_off_diagonal_median_auroc": float(offdiag["auroc"].median()),
        "pairwise_off_diagonal_pairs": int(len(offdiag)),
        "outputs": [str(OUT_K), str(OUT_L)],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
