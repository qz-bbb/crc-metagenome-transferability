from __future__ import annotations

import json
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests


RUN_DIR = Path(r"D:\1\projects\gut_microbiome_colorectal_cancer\runs\run_20260525_182306")
WORKSPACE = RUN_DIR / "agents" / "05-analysis" / "workspace"
RESULTS = WORKSPACE / "results"
PROCESSED = WORKSPACE / "processed_data"
DATA_DIR = Path(r"D:\1\Knowledge Base\true_raw\data\BIOC_curatedMetagenomicData_CRC")
OUT = RESULTS / "N_transferability_prioritization"
OUT.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 20260527
RANDOM_PANEL_REPEATS = 500


def read_tsv(path: Path, **kwargs) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", **kwargs)


def write_tsv(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=False)
    return path


def bh(values: pd.Series | list[float]) -> np.ndarray:
    values = pd.Series(values, dtype=float)
    out = np.full(len(values), np.nan)
    mask = values.notna()
    if mask.any():
        out[mask.to_numpy()] = multipletests(values[mask], method="fdr_bh")[1]
    return out


def parse_taxon(taxon: str) -> dict[str, str]:
    parts = {}
    for token in str(taxon).split("|"):
        if "__" in token:
            rank, value = token.split("__", 1)
            parts[rank] = value
    species = parts.get("s", taxon).replace("_", " ")
    genus = parts.get("g", "")
    family = parts.get("f", "")
    return {"taxon_long": taxon, "genus": genus, "family": family, "species_short": species}


def design_matrix(metadata: pd.DataFrame, terms: list[str]) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    blocks = []
    term_cols: dict[str, list[str]] = {}
    index = metadata.index
    const = pd.DataFrame({"const": np.ones(len(metadata))}, index=index)
    blocks.append(const)
    term_cols["const"] = ["const"]
    for term in terms:
        if term == "study":
            block = pd.get_dummies(metadata["study_name"].astype(str), prefix="study", drop_first=True, dtype=float)
        elif term == "condition":
            block = pd.DataFrame({"condition_crc": (metadata["study_condition"].astype(str) == "CRC").astype(float)}, index=index)
        elif term == "age":
            age = pd.to_numeric(metadata["age"], errors="coerce")
            block = pd.DataFrame({"age_z": (age - age.mean()) / age.std(ddof=0)}, index=index)
        elif term == "BMI":
            bmi = pd.to_numeric(metadata["BMI"], errors="coerce")
            block = pd.DataFrame({"BMI_z": (bmi - bmi.mean()) / bmi.std(ddof=0)}, index=index)
        elif term == "gender":
            block = pd.get_dummies(metadata["gender"].astype(str), prefix="gender", drop_first=True, dtype=float)
        else:
            raise ValueError(f"Unknown term: {term}")
        blocks.append(block)
        term_cols[term] = list(block.columns)
    design = pd.concat(blocks, axis=1)
    design = design.loc[:, design.notna().all(axis=0)]
    return design, term_cols


def rss_for_design(x: np.ndarray, design: pd.DataFrame) -> tuple[float, int]:
    matrix = design.to_numpy(dtype=float)
    coef, *_ = np.linalg.lstsq(matrix, x, rcond=None)
    residual = x - matrix @ coef
    rank = int(np.linalg.matrix_rank(matrix))
    rss = float(np.sum(residual**2))
    return rss, rank


def variance_partitioning(clr: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    x_all = clr.to_numpy(dtype=float)
    x_all = x_all - x_all.mean(axis=0, keepdims=True)
    total_ss_all = float(np.sum(x_all**2))
    rows = []

    all_design, _ = design_matrix(meta, ["study"])
    rss_null, rank_null = rss_for_design(x_all, pd.DataFrame({"const": np.ones(len(meta))}, index=meta.index))
    rss_study, rank_study = rss_for_design(x_all, all_design)
    df_term = rank_study - rank_null
    df_resid = len(meta) - rank_study
    f_value = ((rss_null - rss_study) / df_term) / (rss_study / df_resid)
    rows.append(
        {
            "term": "study_label",
            "model_scope": "all_samples",
            "n_samples": len(meta),
            "df_term": df_term,
            "df_residual": df_resid,
            "partial_r2_percent": 100 * (rss_null - rss_study) / total_ss_all,
            "pseudo_f": f_value,
            "conditioning": "marginal study-label model",
        }
    )

    reduced, _ = design_matrix(meta, ["study"])
    full, _ = design_matrix(meta, ["study", "condition"])
    rss_reduced, rank_reduced = rss_for_design(x_all, reduced)
    rss_full, rank_full = rss_for_design(x_all, full)
    df_term = rank_full - rank_reduced
    df_resid = len(meta) - rank_full
    f_value = ((rss_reduced - rss_full) / df_term) / (rss_full / df_resid)
    rows.append(
        {
            "term": "CRC_status",
            "model_scope": "all_samples",
            "n_samples": len(meta),
            "df_term": df_term,
            "df_residual": df_resid,
            "partial_r2_percent": 100 * (rss_reduced - rss_full) / total_ss_all,
            "pseudo_f": f_value,
            "conditioning": "after study label",
        }
    )

    cov_meta = meta.loc[meta[["age", "BMI", "gender"]].notna().all(axis=1)].copy()
    cov_clr = clr.loc[cov_meta.index]
    x_cov = cov_clr.to_numpy(dtype=float)
    x_cov = x_cov - x_cov.mean(axis=0, keepdims=True)
    total_ss_cov = float(np.sum(x_cov**2))
    cov_terms = ["age", "BMI", "gender"]
    base_terms = ["study", "condition"]
    for term in cov_terms:
        reduced_terms = [t for t in base_terms + cov_terms if t != term]
        reduced, _ = design_matrix(cov_meta, reduced_terms)
        full, _ = design_matrix(cov_meta, base_terms + cov_terms)
        rss_reduced, rank_reduced = rss_for_design(x_cov, reduced)
        rss_full, rank_full = rss_for_design(x_cov, full)
        df_term = rank_full - rank_reduced
        df_resid = len(cov_meta) - rank_full
        f_value = ((rss_reduced - rss_full) / df_term) / (rss_full / df_resid)
        rows.append(
            {
                "term": term,
                "model_scope": "complete_case_covariates",
                "n_samples": len(cov_meta),
                "df_term": df_term,
                "df_residual": df_resid,
                "partial_r2_percent": 100 * (rss_reduced - rss_full) / total_ss_cov,
                "pseudo_f": f_value,
                "conditioning": "after study label, CRC status, and other available covariates",
            }
        )
    table = pd.DataFrame(rows)
    table["method"] = "distance-equivalent multivariate linear model on retained CLR matrix"
    return table


def fit_probabilities(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray) -> np.ndarray:
    scaler = StandardScaler()
    x_train_z = scaler.fit_transform(x_train)
    x_test_z = scaler.transform(x_test)
    model = LogisticRegression(
        penalty="elasticnet",
        solver="saga",
        l1_ratio=0.5,
        C=0.5,
        class_weight="balanced",
        max_iter=2000,
        random_state=42,
        n_jobs=1,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        warnings.filterwarnings("ignore", category=FutureWarning)
        model.fit(x_train_z, y_train)
    return model.predict_proba(x_test_z)[:, 1]


def safe_auc(y_true: np.ndarray, score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return np.nan
    return float(roc_auc_score(y_true, score))


def safe_ap(y_true: np.ndarray, score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return np.nan
    return float(average_precision_score(y_true, score))


def eval_loso(x: pd.DataFrame, meta: pd.DataFrame, features: list[str], panel_id: str, repeat_id: int | None = None) -> pd.DataFrame:
    y = (meta["study_condition"].astype(str) == "CRC").astype(int).to_numpy()
    study = meta["study_name"].astype(str)
    x_feat = x[features].to_numpy(dtype=float)
    rows = []
    for held_out in sorted(study.unique()):
        test_mask = (study == held_out).to_numpy()
        train_mask = ~test_mask
        y_train = y[train_mask]
        y_test = y[test_mask]
        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            rows.append({"panel_id": panel_id, "repeat_id": repeat_id, "held_out_study": held_out, "auroc": np.nan, "average_precision": np.nan, "status": "SKIPPED_SINGLE_CLASS"})
            continue
        prob = fit_probabilities(x_feat[train_mask], y_train, x_feat[test_mask])
        rows.append(
            {
                "panel_id": panel_id,
                "repeat_id": repeat_id,
                "held_out_study": held_out,
                "n_train": int(train_mask.sum()),
                "n_test": int(test_mask.sum()),
                "n_test_crc": int(y_test.sum()),
                "n_test_control": int((1 - y_test).sum()),
                "auroc": safe_auc(y_test, prob),
                "average_precision": safe_ap(y_test, prob),
                "status": "COMPUTED",
            }
        )
    return pd.DataFrame(rows)


def eval_pairwise(x: pd.DataFrame, meta: pd.DataFrame, features: list[str], panel_id: str, repeat_id: int | None = None) -> pd.DataFrame:
    y = (meta["study_condition"].astype(str) == "CRC").astype(int).to_numpy()
    study = meta["study_name"].astype(str)
    x_feat = x[features].to_numpy(dtype=float)
    rows = []
    for train_study in sorted(study.unique()):
        train_mask = (study == train_study).to_numpy()
        y_train = y[train_mask]
        if len(np.unique(y_train)) < 2:
            continue
        scaler = StandardScaler()
        x_train_z = scaler.fit_transform(x_feat[train_mask])
        model = LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            l1_ratio=0.5,
            C=0.5,
            class_weight="balanced",
            max_iter=2000,
            random_state=42,
            n_jobs=1,
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            warnings.filterwarnings("ignore", category=FutureWarning)
            model.fit(x_train_z, y_train)
        for test_study in sorted(study.unique()):
            test_mask = (study == test_study).to_numpy()
            y_test = y[test_mask]
            if len(np.unique(y_test)) < 2:
                status = "SKIPPED_SINGLE_CLASS"
                auc = np.nan
                ap = np.nan
            else:
                prob = model.predict_proba(scaler.transform(x_feat[test_mask]))[:, 1]
                auc = safe_auc(y_test, prob)
                ap = safe_ap(y_test, prob)
                status = "COMPUTED"
            rows.append(
                {
                    "panel_id": panel_id,
                    "repeat_id": repeat_id,
                    "train_study": train_study,
                    "test_study": test_study,
                    "is_diagonal": train_study == test_study,
                    "n_train": int(train_mask.sum()),
                    "n_test": int(test_mask.sum()),
                    "auroc": auc,
                    "average_precision": ap,
                    "status": status,
                }
            )
    return pd.DataFrame(rows)


def iqr(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return np.nan
    return float(values.quantile(0.75) - values.quantile(0.25))


def panel_metrics(loso: pd.DataFrame, pairwise: pd.DataFrame) -> dict[str, float]:
    loso_auc = pd.to_numeric(loso["auroc"], errors="coerce").dropna()
    off = pairwise.loc[~pairwise["is_diagonal"].astype(bool), "auroc"].dropna()
    diag = pairwise.loc[pairwise["is_diagonal"].astype(bool), "auroc"].dropna()
    diag_median = float(diag.median()) if not diag.empty else np.nan
    off_median = float(off.median()) if not off.empty else np.nan
    return {
        "loso_median_auroc": float(loso_auc.median()) if not loso_auc.empty else np.nan,
        "loso_min_auroc": float(loso_auc.min()) if not loso_auc.empty else np.nan,
        "loso_iqr_auroc": iqr(loso_auc),
        "pairwise_offdiag_median_auroc": off_median,
        "pairwise_offdiag_min_auroc": float(off.min()) if not off.empty else np.nan,
        "pairwise_offdiag_iqr_auroc": iqr(off),
        "pairwise_diag_median_auroc": diag_median,
        "transferability_loss": diag_median - off_median if np.isfinite(diag_median) and np.isfinite(off_median) else np.nan,
    }


def build_candidate_panels(clr: pd.DataFrame) -> tuple[dict[str, list[str]], pd.DataFrame]:
    assoc = read_tsv(RESULTS / "C" / "cohort_adjusted_crc_control_association.tsv")
    stable = read_tsv(RESULTS / "M_robustness_interpretation" / "stable_candidate_interpretation.tsv")
    all_features = list(clr.columns)
    top29 = assoc.sort_values(["qvalue_bh", "pvalue"]).head(29)["taxon_long"].tolist()
    stable29 = stable["taxon_long"].tolist()
    high18 = stable.loc[stable["interpretation_group"] == "high_consistency_candidate", "taxon_long"].tolist()
    panels = {
        "all_301_retained_species": all_features,
        "top29_by_study_adjusted_bh_q": top29,
        "stable29_transferability_aware": stable29,
        "high18_consistency_candidates": high18,
    }
    rows = []
    descriptions = {
        "all_301_retained_species": "All retained species after prevalence and cohort-presence filtering.",
        "top29_by_study_adjusted_bh_q": "Top 29 species ranked by pooled study-adjusted BH q value.",
        "stable29_transferability_aware": "Twenty-nine FDR < 0.10 candidates with at least 60% same-direction cohort support.",
        "high18_consistency_candidates": "Stable candidates with high-consistency support across LOCO and heterogeneity layers.",
    }
    for panel_id, features in panels.items():
        for rank, taxon in enumerate(features, start=1):
            meta = parse_taxon(taxon)
            rows.append({"panel_id": panel_id, "panel_description": descriptions[panel_id], "rank_within_panel": rank, **meta})
    return panels, pd.DataFrame(rows)


def panel_benchmark(clr: pd.DataFrame, meta: pd.DataFrame) -> dict[str, pd.DataFrame]:
    panels, membership = build_candidate_panels(clr)
    rng = np.random.default_rng(RANDOM_SEED)
    all_features = list(clr.columns)
    loso_tables = []
    pairwise_tables = []
    summary_rows = []
    random_summary_rows = []
    for panel_id, features in panels.items():
        loso = eval_loso(clr, meta, features, panel_id)
        pairwise = eval_pairwise(clr, meta, features, panel_id)
        loso_tables.append(loso)
        pairwise_tables.append(pairwise)
        summary_rows.append(
            {
                "panel_id": panel_id,
                "panel_type": "fixed",
                "n_features": len(features),
                "random_repeats": 0,
                **panel_metrics(loso, pairwise),
            }
        )
    for repeat in range(1, RANDOM_PANEL_REPEATS + 1):
        features = rng.choice(all_features, size=29, replace=False).tolist()
        panel_id = "random29_species_panels"
        loso = eval_loso(clr, meta, features, panel_id, repeat)
        pairwise = eval_pairwise(clr, meta, features, panel_id, repeat)
        metrics = panel_metrics(loso, pairwise)
        random_summary_rows.append({"panel_id": panel_id, "panel_type": "random_repeat", "repeat_id": repeat, "n_features": 29, **metrics})
        loso_tables.append(loso)
        pairwise_tables.append(pairwise)
    random_summary = pd.DataFrame(random_summary_rows)
    aggregate = {"panel_id": "random29_species_panels", "panel_type": "random_aggregate", "n_features": 29, "random_repeats": RANDOM_PANEL_REPEATS}
    for col in [
        "loso_median_auroc",
        "loso_min_auroc",
        "loso_iqr_auroc",
        "pairwise_offdiag_median_auroc",
        "pairwise_offdiag_min_auroc",
        "pairwise_offdiag_iqr_auroc",
        "pairwise_diag_median_auroc",
        "transferability_loss",
    ]:
        aggregate[col] = float(random_summary[col].median())
        aggregate[f"{col}_random_iqr"] = iqr(random_summary[col])
    summary_rows.append(aggregate)
    return {
        "membership": membership,
        "loso": pd.concat(loso_tables, ignore_index=True),
        "pairwise": pd.concat(pairwise_tables, ignore_index=True),
        "summary": pd.DataFrame(summary_rows),
        "random_summary": random_summary,
    }


def taxa_matching(clr: pd.DataFrame, predicate, guild_id: str, source_category: str, expected_direction: str, notes: str) -> list[dict[str, str]]:
    rows = []
    for taxon in clr.columns:
        parsed = parse_taxon(taxon)
        if predicate(parsed, taxon):
            rows.append(
                {
                    "guild_id": guild_id,
                    **parsed,
                    "source_category": source_category,
                    "expected_crc_direction": expected_direction,
                    "notes": notes,
                }
            )
    return rows


def build_guild_membership(clr: pd.DataFrame) -> pd.DataFrame:
    butyrate_species = {
        "Faecalibacterium prausnitzii",
        "Roseburia faecis",
        "Roseburia hominis",
        "Roseburia intestinalis",
        "Roseburia inulinivorans",
        "Anaerostipes hadrus",
        "Agathobaculum butyriciproducens",
        "Butyricicoccus pullicaecorum",
        "Butyricimonas virosa",
        "Butyricimonas synergistica",
        "Eubacterium eligens",
        "Eubacterium hallii",
        "Eubacterium rectale",
        "Fusicatenibacter saccharivorans",
        "Intestinimonas butyriciproducens",
        "Gemmiger formicilis",
        "Ruminococcus bromii",
        "Ruminococcus lactaris",
    }
    rows = taxa_matching(
        clr,
        lambda parsed, _taxon: parsed["species_short"] in butyrate_species or parsed["genus"] in {"Roseburia"},
        "butyrate_scfa_commensal",
        "literature-prior butyrate/SCFA-associated commensal taxonomy panel",
        "control_higher_or_crc_depleted",
        "Taxonomy-name panel for ecological coherence; not a functional metagenomic profile.",
    )
    oral = read_tsv(RESULTS / "K_ecology_oral_signature" / "oral_associated_taxa_panel.tsv")
    for _, row in oral.iterrows():
        rows.append(
            {
                "guild_id": "oral_pathobiont_associated",
                "taxon_long": row["taxon_long"],
                "genus": row["genus"],
                "family": parse_taxon(row["taxon_long"])["family"],
                "species_short": row["species_short"],
                "source_category": "literature-prior oral/pathobiont-associated taxonomy panel",
                "expected_crc_direction": "crc_higher",
                "notes": "Taxonomy-name panel for stool metagenome context; not an oral-source attribution model.",
            }
        )
    rows.extend(
        taxa_matching(
            clr,
            lambda parsed, taxon: parsed["genus"] == "Bacteroides" or parsed["family"] == "Enterobacteriaceae",
            "bacteroides_enterobacteriaceae_context",
            "Bacteroides plus Enterobacteriaceae taxonomy context panel",
            "crc_higher_or_context_dependent",
            "Taxonomic context panel only; not an inflammation mechanism or functional profile.",
        )
    )
    guild = pd.DataFrame(rows).drop_duplicates(["guild_id", "taxon_long"]).sort_values(["guild_id", "genus", "species_short"])
    return guild


def ols_score_association(score: pd.Series, meta: pd.DataFrame) -> tuple[float, float, float, float, int]:
    df = pd.DataFrame({"score": score, "condition_crc": (meta["study_condition"].astype(str) == "CRC").astype(float), "study": meta["study_name"].astype(str)}, index=meta.index)
    design = pd.concat([df[["condition_crc"]], pd.get_dummies(df["study"], prefix="study", drop_first=True, dtype=float)], axis=1)
    design = sm.add_constant(design, has_constant="add")
    model = sm.OLS(df["score"].astype(float), design.astype(float)).fit(cov_type="HC3")
    coef = float(model.params["condition_crc"])
    se = float(model.bse["condition_crc"])
    p = float(model.pvalues["condition_crc"])
    ci_low = coef - 1.96 * se
    ci_high = coef + 1.96 * se
    return coef, ci_low, ci_high, p, len(df)


def per_cohort_score_effects(score: pd.Series, meta: pd.DataFrame, guild_id: str) -> pd.DataFrame:
    rows = []
    for study, idx in meta.groupby("study_name").groups.items():
        sub_meta = meta.loc[idx]
        y = score.loc[idx]
        condition = (sub_meta["study_condition"].astype(str) == "CRC").astype(float)
        if condition.nunique() < 2:
            continue
        design = sm.add_constant(pd.DataFrame({"condition_crc": condition}, index=sub_meta.index), has_constant="add")
        model = sm.OLS(y.astype(float), design.astype(float)).fit(cov_type="HC3")
        rows.append(
            {
                "guild_id": guild_id,
                "study_name": study,
                "n": len(sub_meta),
                "n_crc": int(condition.sum()),
                "n_control": int((1 - condition).sum()),
                "coef_crc": float(model.params["condition_crc"]),
                "se_hc3": float(model.bse["condition_crc"]),
                "pvalue": float(model.pvalues["condition_crc"]),
            }
        )
    return pd.DataFrame(rows)


def dersimonian_laird(effect: np.ndarray, se: np.ndarray) -> dict[str, float]:
    mask = np.isfinite(effect) & np.isfinite(se) & (se > 0)
    y = effect[mask]
    s = se[mask]
    k = len(y)
    if k < 2:
        return {k: np.nan for k in ["random_effect", "random_se", "ci_low", "ci_high", "z_value", "pvalue", "tau2_dl", "i2_percent", "q_statistic"]}
    w = 1 / (s**2)
    fixed = np.sum(w * y) / np.sum(w)
    q = np.sum(w * (y - fixed) ** 2)
    df = k - 1
    c = np.sum(w) - (np.sum(w**2) / np.sum(w))
    tau2 = max(0.0, (q - df) / c) if c > 0 else 0.0
    wr = 1 / (s**2 + tau2)
    random = np.sum(wr * y) / np.sum(wr)
    random_se = math.sqrt(1 / np.sum(wr))
    z = random / random_se if random_se > 0 else np.nan
    p = 2 * stats.norm.sf(abs(z)) if np.isfinite(z) else np.nan
    i2 = max(0.0, (q - df) / q) * 100 if q > 0 else 0.0
    return {
        "k_cohorts": k,
        "random_effect": random,
        "random_se": random_se,
        "random_ci_low": random - 1.96 * random_se,
        "random_ci_high": random + 1.96 * random_se,
        "z_value": z,
        "pvalue": p,
        "tau2_dl": tau2,
        "i2_percent": i2,
        "q_statistic": q,
    }


def guild_score_analysis(clr: pd.DataFrame, meta: pd.DataFrame) -> dict[str, pd.DataFrame]:
    membership = build_guild_membership(clr)
    membership["retained_for_score"] = membership["taxon_long"].isin(clr.columns)
    score_df = pd.DataFrame(index=clr.index)
    association_rows = []
    per_cohort_tables = []
    meta_rows = []
    loco_rows = []
    loso_rows = []
    pairwise_rows = []
    for guild_id, group in membership.groupby("guild_id"):
        features = [f for f in group["taxon_long"].tolist() if f in clr.columns]
        score = clr[features].mean(axis=1)
        score_df[guild_id] = score
        coef, ci_low, ci_high, p, n = ols_score_association(score, meta)
        association_rows.append(
            {
                "guild_id": guild_id,
                "n_samples": n,
                "n_taxa": len(features),
                "coef_crc_study_adjusted": coef,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "pvalue": p,
                "direction": "CRC_higher" if coef > 0 else "control_higher",
            }
        )
        per = per_cohort_score_effects(score, meta, guild_id)
        per_cohort_tables.append(per)
        dl = dersimonian_laird(per["coef_crc"].to_numpy(dtype=float), per["se_hc3"].to_numpy(dtype=float))
        dl.update({"guild_id": guild_id, "n_taxa": len(features)})
        meta_rows.append(dl)
        sub_loco = []
        for excluded in sorted(meta["study_name"].astype(str).unique()):
            keep = meta["study_name"].astype(str) != excluded
            c, lo, hi, pv, nn = ols_score_association(score.loc[keep], meta.loc[keep])
            sub_loco.append({"guild_id": guild_id, "excluded_study": excluded, "coef_crc": c, "pvalue": pv, "n_samples": nn, "direction": "CRC_higher" if c > 0 else "control_higher"})
        loco_rows.extend(sub_loco)
        score_matrix = pd.DataFrame({guild_id: score}, index=clr.index)
        loso_rows.append(eval_loso(score_matrix, meta, [guild_id], f"guild_score_{guild_id}"))
        pairwise_rows.append(eval_pairwise(score_matrix, meta, [guild_id], f"guild_score_{guild_id}"))
    assoc = pd.DataFrame(association_rows)
    assoc["qvalue_bh"] = bh(assoc["pvalue"])
    meta_table = pd.DataFrame(meta_rows)
    meta_table["qvalue_bh_random_effect"] = bh(meta_table["pvalue"])
    loco = pd.DataFrame(loco_rows)
    loco["qvalue_bh_within_exclusion"] = loco.groupby("excluded_study")["pvalue"].transform(lambda x: bh(x))
    loco_summary = []
    for guild_id, group in loco.groupby("guild_id"):
        baseline_dir = assoc.loc[assoc["guild_id"] == guild_id, "direction"].iloc[0]
        loco_summary.append(
            {
                "guild_id": guild_id,
                "n_exclusions": len(group),
                "direction_preservation_fraction": float((group["direction"] == baseline_dir).mean()),
                "fdr_lt_0_10_fraction": float((group["qvalue_bh_within_exclusion"] < 0.10).mean()),
                "loco_robust": bool(((group["direction"] == baseline_dir).mean() >= 0.90) and ((group["qvalue_bh_within_exclusion"] < 0.10).mean() >= 0.80)),
            }
        )
    loso = pd.concat(loso_rows, ignore_index=True)
    pairwise = pd.concat(pairwise_rows, ignore_index=True)
    transfer_rows = []
    for panel_id in loso["panel_id"].unique():
        transfer_rows.append({"panel_id": panel_id, **panel_metrics(loso[loso["panel_id"] == panel_id], pairwise[pairwise["panel_id"] == panel_id])})
    scores_out = score_df.reset_index().rename(columns={"index": "sample_uid"})
    return {
        "membership": membership,
        "sample_scores": scores_out,
        "association": assoc,
        "per_cohort": pd.concat(per_cohort_tables, ignore_index=True),
        "random_effects": meta_table,
        "loco": loco,
        "loco_summary": pd.DataFrame(loco_summary),
        "loso": loso,
        "pairwise": pairwise,
        "transfer_summary": pd.DataFrame(transfer_rows),
    }


def update_module_status(paths: dict[str, Path]) -> None:
    status_path = WORKSPACE / "module_status.tsv"
    if status_path.exists():
        status = pd.read_csv(status_path, sep="\t")
    else:
        status = pd.DataFrame(columns=["module_id", "status", "primary_output", "notes"])
    new_rows = pd.DataFrame(
        [
            {
                "module_id": "N",
                "status": "COMPLETED",
                "primary_output": str(paths["variance"]),
                "notes": "Variance partitioning of retained CLR matrix by study label, CRC status, and available covariates.",
            },
            {
                "module_id": "O",
                "status": "COMPLETED",
                "primary_output": str(paths["panel_summary"]),
                "notes": f"Panel benchmarking completed for all species, q-ranked top 29, stable 29, high-consistency 18, and {RANDOM_PANEL_REPEATS} random 29-species panels.",
            },
            {
                "module_id": "P",
                "status": "COMPLETED",
                "primary_output": str(paths["guild_association"]),
                "notes": "Ecological guild score association, heterogeneity, LOCO, and transferability summaries completed.",
            },
        ]
    )
    status = status[~status["module_id"].isin(["N", "O", "P"])]
    pd.concat([status, new_rows], ignore_index=True).to_csv(status_path, sep="\t", index=False)


def main() -> None:
    meta = read_tsv(DATA_DIR / "curated_crc_case_control_sample_metadata.tsv").set_index("sample_uid", drop=False)
    clr = read_tsv(PROCESSED / "B" / "filtered_species_clr.tsv.gz", index_col=0)
    common = clr.index.intersection(meta.index)
    clr = clr.loc[common]
    meta = meta.loc[common]

    variance = variance_partitioning(clr, meta)
    variance_path = write_tsv(variance, OUT / "variance_partitioning.tsv")

    panel = panel_benchmark(clr, meta)
    panel_membership_path = write_tsv(panel["membership"], OUT / "candidate_panel_membership.tsv")
    panel_loso_path = write_tsv(panel["loso"], OUT / "panel_benchmark_loso_folds.tsv")
    panel_pairwise_path = write_tsv(panel["pairwise"], OUT / "panel_benchmark_pairwise.tsv")
    panel_summary_path = write_tsv(panel["summary"], OUT / "panel_benchmark_summary.tsv")
    random_summary_path = write_tsv(panel["random_summary"], OUT / "random29_repeat_summary.tsv")

    guild = guild_score_analysis(clr, meta)
    guild_membership_path = write_tsv(guild["membership"], OUT / "guild_panel_membership.tsv")
    guild_scores_path = write_tsv(guild["sample_scores"], OUT / "guild_sample_scores.tsv")
    guild_association_path = write_tsv(guild["association"], OUT / "guild_score_association.tsv")
    guild_per_cohort_path = write_tsv(guild["per_cohort"], OUT / "guild_score_per_cohort_effects.tsv")
    guild_random_effects_path = write_tsv(guild["random_effects"], OUT / "guild_score_random_effects.tsv")
    guild_loco_path = write_tsv(guild["loco"], OUT / "guild_score_loco.tsv")
    guild_loco_summary_path = write_tsv(guild["loco_summary"], OUT / "guild_score_loco_summary.tsv")
    guild_loso_path = write_tsv(guild["loso"], OUT / "guild_score_loso.tsv")
    guild_pairwise_path = write_tsv(guild["pairwise"], OUT / "guild_score_pairwise.tsv")
    guild_transfer_path = write_tsv(guild["transfer_summary"], OUT / "guild_score_transfer_summary.tsv")

    paths = {
        "variance": variance_path,
        "panel_membership": panel_membership_path,
        "panel_loso": panel_loso_path,
        "panel_pairwise": panel_pairwise_path,
        "panel_summary": panel_summary_path,
        "random_summary": random_summary_path,
        "guild_membership": guild_membership_path,
        "guild_scores": guild_scores_path,
        "guild_association": guild_association_path,
        "guild_per_cohort": guild_per_cohort_path,
        "guild_random_effects": guild_random_effects_path,
        "guild_loco": guild_loco_path,
        "guild_loco_summary": guild_loco_summary_path,
        "guild_loso": guild_loso_path,
        "guild_pairwise": guild_pairwise_path,
        "guild_transfer": guild_transfer_path,
    }
    update_module_status(paths)
    summary = {
        "random_seed": RANDOM_SEED,
        "random_panel_repeats": RANDOM_PANEL_REPEATS,
        "variance_terms": variance[["term", "partial_r2_percent"]].to_dict("records"),
        "panel_summary": panel["summary"].to_dict("records"),
        "guild_association": guild["association"].to_dict("records"),
        "outputs": {k: str(v) for k, v in paths.items()},
    }
    (OUT / "transferability_prioritization_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report = [
        "# Transferability-aware candidate prioritization analysis",
        "",
        f"- Random seed: {RANDOM_SEED}",
        f"- Random 29-species panel repeats: {RANDOM_PANEL_REPEATS}",
        f"- Variance partitioning terms: {', '.join(variance['term'])}",
        f"- Fixed candidate panels: all 301 retained species, top 29 by study-adjusted BH q, stable 29, high-consistency 18.",
        f"- Guild panels: {', '.join(guild['association']['guild_id'])}",
        "",
        "These modules are processed-matrix robustness and transferability analyses. They do not add causal, diagnostic, stage-specific, functional, strain-level, oral-source, or raw-read claims.",
    ]
    (OUT / "transferability_prioritization_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"status": "COMPLETED", "outputs": {k: str(v) for k, v in paths.items()}}, indent=2))


if __name__ == "__main__":
    main()
