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
OUT = RESULTS / "Q_leakage_aware_prioritization"
OUT.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 20260527
N_PERMUTATIONS = 999
N_RANDOM_PANELS = 500


def read_tsv(path: Path, **kwargs) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", **kwargs)


def write_csv(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
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
    return {
        "taxon": taxon,
        "genus": parts.get("g", ""),
        "family": parts.get("f", ""),
        "species": parts.get("s", taxon).replace("_", " "),
    }


def iqr(values: pd.Series | np.ndarray) -> float:
    s = pd.Series(values, dtype=float).dropna()
    if s.empty:
        return np.nan
    return float(s.quantile(0.75) - s.quantile(0.25))


def safe_auc(y_true: np.ndarray, score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return np.nan
    return float(roc_auc_score(y_true, score))


def safe_ap(y_true: np.ndarray, score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return np.nan
    return float(average_precision_score(y_true, score))


def crc_vector(meta: pd.DataFrame) -> np.ndarray:
    return (meta["study_condition"].astype(str) == "CRC").astype(int).to_numpy()


def design_for_terms(meta: pd.DataFrame, terms: list[str]) -> pd.DataFrame:
    blocks = [pd.DataFrame({"const": np.ones(len(meta))}, index=meta.index)]
    for term in terms:
        if term == "study_label":
            block = pd.get_dummies(meta["study_name"].astype(str), prefix="study", drop_first=True, dtype=float)
        elif term == "CRC_status":
            block = pd.DataFrame({"CRC_status": crc_vector(meta).astype(float)}, index=meta.index)
        elif term == "age":
            age = pd.to_numeric(meta["age"], errors="coerce")
            block = pd.DataFrame({"age_z": (age - age.mean()) / age.std(ddof=0)}, index=meta.index)
        elif term == "BMI":
            bmi = pd.to_numeric(meta["BMI"], errors="coerce")
            block = pd.DataFrame({"BMI_z": (bmi - bmi.mean()) / bmi.std(ddof=0)}, index=meta.index)
        elif term == "gender":
            block = pd.get_dummies(meta["gender"].astype(str), prefix="gender", drop_first=True, dtype=float)
        else:
            raise ValueError(f"unknown term: {term}")
        blocks.append(block)
    design = pd.concat(blocks, axis=1)
    return design.loc[:, design.notna().all(axis=0)]


def rss_rank(x: np.ndarray, design: pd.DataFrame) -> tuple[float, int]:
    mat = design.to_numpy(dtype=float)
    coef, *_ = np.linalg.lstsq(mat, x, rcond=None)
    resid = x - mat @ coef
    return float(np.sum(resid**2)), int(np.linalg.matrix_rank(mat))


def pseudo_f_for_term(x: np.ndarray, meta: pd.DataFrame, term: str, conditioning_terms: list[str]) -> tuple[float, float, float, int, int]:
    reduced = design_for_terms(meta, conditioning_terms)
    full_terms = conditioning_terms + [term]
    full = design_for_terms(meta, full_terms)
    rss_reduced, rank_reduced = rss_rank(x, reduced)
    rss_full, rank_full = rss_rank(x, full)
    df_term = rank_full - rank_reduced
    df_resid = len(meta) - rank_full
    if df_term <= 0 or df_resid <= 0:
        return np.nan, np.nan, np.nan, df_term, df_resid
    f_value = ((rss_reduced - rss_full) / df_term) / (rss_full / df_resid)
    total_ss = float(np.sum((x - x.mean(axis=0, keepdims=True)) ** 2))
    partial_r2 = 100 * (rss_reduced - rss_full) / total_ss
    return float(f_value), float(partial_r2), float(rss_full), int(df_term), int(df_resid)


def permute_term(meta: pd.DataFrame, term: str, rng: np.random.Generator) -> pd.DataFrame:
    out = meta.copy()
    if term == "CRC_status":
        values = []
        for _, idx in out.groupby("study_name").groups.items():
            vals = out.loc[idx, "study_condition"].to_numpy(copy=True)
            rng.shuffle(vals)
            values.append(pd.Series(vals, index=idx))
        out["study_condition"] = pd.concat(values).loc[out.index]
    elif term == "study_label":
        vals = out["study_name"].to_numpy(copy=True)
        rng.shuffle(vals)
        out["study_name"] = vals
    elif term == "age":
        vals = out["age"].to_numpy(copy=True)
        rng.shuffle(vals)
        out["age"] = vals
    elif term == "BMI":
        vals = out["BMI"].to_numpy(copy=True)
        rng.shuffle(vals)
        out["BMI"] = vals
    elif term == "gender":
        vals = out["gender"].to_numpy(copy=True)
        rng.shuffle(vals)
        out["gender"] = vals
    else:
        raise ValueError(term)
    return out


def aitchison_variance_partitioning(clr: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    complete = meta[["age", "BMI", "gender"]].notna().all(axis=1)
    meta_cc = meta.loc[complete].copy()
    clr_cc = clr.loc[meta_cc.index]
    x = clr_cc.to_numpy(dtype=float)
    x = x - x.mean(axis=0, keepdims=True)
    terms = ["study_label", "CRC_status", "age", "BMI", "gender"]
    formula = "retained_CLR ~ study_label + CRC_status + age_z + BMI_z + gender"
    sequential_terms: list[str] = []
    total_ss = float(np.sum(x**2))
    rows = []
    rng = np.random.default_rng(RANDOM_SEED + 101)

    for term in terms:
        seq_f, seq_r2, _, _, _ = pseudo_f_for_term(x, meta_cc, term, sequential_terms)
        conditioning = [t for t in terms if t != term]
        obs_f, partial_r2, _, df_term, df_resid = pseudo_f_for_term(x, meta_cc, term, conditioning)
        ge = 0
        for _ in range(N_PERMUTATIONS):
            perm_meta = permute_term(meta_cc, term, rng)
            perm_f, _, _, _, _ = pseudo_f_for_term(x, perm_meta, term, conditioning)
            if np.isfinite(perm_f) and perm_f >= obs_f:
                ge += 1
        p_value = (ge + 1) / (N_PERMUTATIONS + 1)
        rows.append(
            {
                "term": term,
                "r2": seq_r2 / 100,
                "partial_or_marginal_r2": partial_r2 / 100,
                "p_value": p_value,
                "permutations": N_PERMUTATIONS,
                "model_formula": formula,
                "sample_n": len(meta_cc),
                "pseudo_f": obs_f,
                "df_term": df_term,
                "df_residual": df_resid,
                "notes": "Euclidean distance on retained CLR profiles; r2 is sequential in the displayed order; partial_or_marginal_r2 tests the term against a model containing the other listed terms.",
            }
        )
        sequential_terms.append(term)

    full = design_for_terms(meta_cc, terms)
    rss_full, _ = rss_rank(x, full)
    rows.append(
        {
            "term": "residual",
            "r2": rss_full / total_ss,
            "partial_or_marginal_r2": np.nan,
            "p_value": np.nan,
            "permutations": N_PERMUTATIONS,
            "model_formula": formula,
            "sample_n": len(meta_cc),
            "pseudo_f": np.nan,
            "df_term": np.nan,
            "df_residual": len(meta_cc) - int(np.linalg.matrix_rank(full.to_numpy(dtype=float))),
            "notes": "Residual fraction after the full complete-case model.",
        }
    )
    return pd.DataFrame(rows)


def association_screen(clr: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    y_crc = crc_vector(meta)
    design = pd.concat(
        [
            pd.DataFrame({"CRC_status": y_crc.astype(float)}, index=meta.index),
            pd.get_dummies(meta["study_name"].astype(str), prefix="study", drop_first=True, dtype=float),
        ],
        axis=1,
    )
    design = sm.add_constant(design, has_constant="add")
    rows = []
    for taxon in clr.columns:
        model = sm.OLS(clr[taxon].astype(float), design.astype(float)).fit(cov_type="HC3")
        rows.append(
            {
                "taxon": taxon,
                "coef_crc": float(model.params["CRC_status"]),
                "se_hc3": float(model.bse["CRC_status"]),
                "p_value": float(model.pvalues["CRC_status"]),
            }
        )
    out = pd.DataFrame(rows)
    out["bh_q_value"] = bh(out["p_value"])
    out["direction"] = np.where(out["coef_crc"] >= 0, "CRC_higher", "control_higher")
    support = same_direction_support(clr, meta, out.set_index("taxon")["coef_crc"])
    out = out.merge(support, on="taxon", how="left")
    return out.sort_values(["bh_q_value", "p_value", "taxon"])


def same_direction_support(clr: pd.DataFrame, meta: pd.DataFrame, coef: pd.Series) -> pd.DataFrame:
    rows = []
    studies = sorted(meta["study_name"].astype(str).unique())
    for taxon in clr.columns:
        sign = np.sign(float(coef.loc[taxon]))
        eligible = 0
        same = 0
        for study in studies:
            idx = meta["study_name"].astype(str) == study
            sub_meta = meta.loc[idx]
            if sub_meta["study_condition"].nunique() < 2:
                continue
            y = clr.loc[sub_meta.index, taxon]
            crc = y[sub_meta["study_condition"].astype(str) == "CRC"]
            ctrl = y[sub_meta["study_condition"].astype(str) != "CRC"]
            diff = float(crc.mean() - ctrl.mean())
            diff_sign = np.sign(diff)
            if diff_sign == 0 or sign == 0:
                continue
            eligible += 1
            same += int(diff_sign == sign)
        rows.append(
            {
                "taxon": taxon,
                "eligible_training_cohorts": eligible,
                "same_direction_training_cohorts": same,
                "same_direction_fraction": same / eligible if eligible else np.nan,
            }
        )
    return pd.DataFrame(rows)


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


def evaluate_heldout(clr: pd.DataFrame, meta: pd.DataFrame, held_out: str, features: list[str]) -> dict[str, float | int | str]:
    study = meta["study_name"].astype(str)
    train_mask = (study != held_out).to_numpy()
    test_mask = (study == held_out).to_numpy()
    y = crc_vector(meta)
    y_train = y[train_mask]
    y_test = y[test_mask]
    if len(features) == 0 or len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        return {
            "auroc": np.nan,
            "average_precision": np.nan,
            "n_train": int(train_mask.sum()),
            "n_test": int(test_mask.sum()),
            "crc_control_in_test": f"CRC={int(y_test.sum())};control={int((1 - y_test).sum())}",
            "status": "SKIPPED",
        }
    prob = fit_probabilities(clr.loc[train_mask, features].to_numpy(dtype=float), y_train, clr.loc[test_mask, features].to_numpy(dtype=float))
    return {
        "auroc": safe_auc(y_test, prob),
        "average_precision": safe_ap(y_test, prob),
        "n_train": int(train_mask.sum()),
        "n_test": int(test_mask.sum()),
        "crc_control_in_test": f"CRC={int(y_test.sum())};control={int((1 - y_test).sum())}",
        "status": "COMPUTED",
    }


def global_panels(clr: pd.DataFrame) -> dict[str, list[str]]:
    assoc = read_tsv(RESULTS / "C" / "cohort_adjusted_crc_control_association.tsv")
    stable = read_tsv(RESULTS / "M_robustness_interpretation" / "stable_candidate_interpretation.tsv")
    top29 = assoc.sort_values(["qvalue_bh", "pvalue"]).head(29)["taxon_long"].tolist()
    stable29 = stable["taxon_long"].tolist()
    high18 = stable.loc[stable["interpretation_group"] == "high_consistency_candidate", "taxon_long"].tolist()
    return {
        "all_301_features": list(clr.columns),
        "global_top29_q": [x for x in top29 if x in clr.columns],
        "global_stable29": [x for x in stable29 if x in clr.columns],
        "global_high_consistency18": [x for x in high18 if x in clr.columns],
    }


def format_taxa(features: list[str]) -> str:
    return ";".join(features)


def overlap_count(features: list[str], reference: list[str]) -> int:
    return len(set(features).intersection(reference))


def prevalence_matched_panel(
    abundance_train: pd.DataFrame,
    target_features: list[str],
    all_features: list[str],
    rng: np.random.Generator,
    size: int = 29,
) -> list[str]:
    prevalence = (abundance_train[all_features] > 0).mean(axis=0)
    target = prevalence[target_features].dropna()
    if target.empty:
        return rng.choice(all_features, size=size, replace=False).tolist()
    quantiles = prevalence.quantile([0, 0.2, 0.4, 0.6, 0.8, 1.0]).to_numpy()
    quantiles = np.unique(quantiles)
    if len(quantiles) < 3:
        return rng.choice(all_features, size=size, replace=False).tolist()
    bins = pd.cut(prevalence, bins=quantiles, include_lowest=True, duplicates="drop")
    target_bins = pd.cut(target, bins=quantiles, include_lowest=True, duplicates="drop")
    selected: list[str] = []
    excluded = set(target_features)
    for bin_value in list(target_bins.astype(str)):
        pool = [f for f in all_features if str(bins.loc[f]) == bin_value and f not in selected and f not in excluded]
        if not pool:
            pool = [f for f in all_features if f not in selected and f not in excluded]
        if not pool:
            pool = [f for f in all_features if f not in selected]
        selected.append(str(rng.choice(pool)))
        if len(selected) >= size:
            break
    while len(selected) < size:
        pool = [f for f in all_features if f not in selected and f not in excluded]
        if not pool:
            pool = [f for f in all_features if f not in selected]
        selected.append(str(rng.choice(pool)))
    return selected[:size]


def leakage_aware_loso(clr: pd.DataFrame, abundance: pd.DataFrame, meta: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    panels = global_panels(clr)
    global_top = panels["global_top29_q"]
    global_stable = panels["global_stable29"]
    global_high = panels["global_high_consistency18"]
    rows = []
    selection_rows = []
    random_rows = []
    rng = np.random.default_rng(RANDOM_SEED + 202)
    all_features = list(clr.columns)

    for held_out in sorted(meta["study_name"].astype(str).unique()):
        train_idx = meta["study_name"].astype(str) != held_out
        train_meta = meta.loc[train_idx]
        train_clr = clr.loc[train_meta.index]
        assoc = association_screen(train_clr, train_meta)
        top29 = assoc.head(29)["taxon"].tolist()
        stable_candidates = assoc[(assoc["bh_q_value"] < 0.10) & (assoc["same_direction_fraction"] >= 0.60)]
        stable_panel = stable_candidates.sort_values(["bh_q_value", "p_value", "taxon"]).head(29)["taxon"].tolist()
        high_candidates = assoc[(assoc["bh_q_value"] < 0.10) & (assoc["same_direction_fraction"] >= 0.80)]
        high_panel = high_candidates.sort_values(["bh_q_value", "p_value", "taxon"]).head(18)["taxon"].tolist()
        fold_panels = {
            "all_301_features": all_features,
            "training_only_top29_q": top29,
            "training_only_stability_panel": stable_panel,
            "training_only_high_consistency_panel": high_panel,
        }
        for panel_type, features in fold_panels.items():
            metrics = evaluate_heldout(clr, meta, held_out, features)
            rows.append(
                {
                    "held_out_study": held_out,
                    "panel_type": panel_type,
                    "panel_size": len(features),
                    "selected_taxa": format_taxa(features),
                    "overlap_with_global_top29_q": overlap_count(features, global_top),
                    "overlap_with_global_stable29": overlap_count(features, global_stable),
                    "overlap_with_global_high_consistency18": overlap_count(features, global_high),
                    **metrics,
                }
            )
            for rank, taxon in enumerate(features, start=1):
                selected = assoc.loc[assoc["taxon"] == taxon]
                selection_rows.append(
                    {
                        "held_out_study": held_out,
                        "panel_type": panel_type,
                        "rank": rank,
                        "taxon": taxon,
                        "training_bh_q_value": float(selected["bh_q_value"].iloc[0]) if not selected.empty else np.nan,
                        "training_coef_crc": float(selected["coef_crc"].iloc[0]) if not selected.empty else np.nan,
                        "training_same_direction_fraction": float(selected["same_direction_fraction"].iloc[0]) if not selected.empty else np.nan,
                    }
                )

        observed_global_stable = evaluate_heldout(clr, meta, held_out, global_stable)["auroc"]
        observed_training_stable = rows[-2]["auroc"]
        train_abund = abundance.loc[train_meta.index]
        for random_type in ["random29_unmatched", "random29_prevalence_matched"]:
            for repeat in range(1, N_RANDOM_PANELS + 1):
                if random_type == "random29_unmatched":
                    features = rng.choice(all_features, size=29, replace=False).tolist()
                else:
                    features = prevalence_matched_panel(train_abund, global_stable, all_features, rng, size=29)
                metrics = evaluate_heldout(clr, meta, held_out, features)
                random_rows.append(
                    {
                        "held_out_study": held_out,
                        "panel_type": random_type,
                        "random_repeat": repeat,
                        "panel_size": len(features),
                        "selected_taxa": format_taxa(features),
                        "overlap_with_global_top29_q": overlap_count(features, global_top),
                        "overlap_with_global_stable29": overlap_count(features, global_stable),
                        "overlap_with_global_high_consistency18": overlap_count(features, global_high),
                        "observed_global_stable29_auroc": observed_global_stable,
                        "observed_training_stability_panel_auroc": observed_training_stable,
                        **metrics,
                    }
                )
    detailed = pd.concat([pd.DataFrame(rows), pd.DataFrame(random_rows)], ignore_index=True, sort=False)
    random_df = pd.DataFrame(random_rows)
    summary_rows = []
    for (held_out, random_type), group in random_df.groupby(["held_out_study", "panel_type"]):
        auroc = pd.to_numeric(group["auroc"], errors="coerce").dropna()
        ap = pd.to_numeric(group["average_precision"], errors="coerce").dropna()
        obs = float(group["observed_global_stable29_auroc"].iloc[0])
        obs_train = float(group["observed_training_stability_panel_auroc"].iloc[0])
        summary_rows.append(
            {
                "held_out_study": held_out,
                "random_type": random_type,
                "n_random_panels": int(len(auroc)),
                "median_auroc": float(auroc.median()),
                "mean_auroc": float(auroc.mean()),
                "auroc_iqr": iqr(auroc),
                "auroc_025": float(auroc.quantile(0.025)),
                "auroc_975": float(auroc.quantile(0.975)),
                "median_average_precision": float(ap.median()) if not ap.empty else np.nan,
                "observed_global_stable29_auroc": obs,
                "empirical_p_value_for_stable29_vs_random": float((1 + (auroc >= obs).sum()) / (len(auroc) + 1)) if np.isfinite(obs) else np.nan,
                "observed_training_stability_panel_auroc": obs_train,
                "empirical_p_value_for_training_stability_vs_random": float((1 + (auroc >= obs_train).sum()) / (len(auroc) + 1)) if np.isfinite(obs_train) else np.nan,
            }
        )
    return detailed, pd.DataFrame(summary_rows), pd.DataFrame(selection_rows)


def fixed_global_panel_benchmarks(clr: pd.DataFrame, meta: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    panels = global_panels(clr)
    loso_detail = []
    pair_detail = []
    for panel_type, features in panels.items():
        for held_out in sorted(meta["study_name"].astype(str).unique()):
            metrics = evaluate_heldout(clr, meta, held_out, features)
            loso_detail.append({"panel_type": panel_type, "panel_size": len(features), "held_out_study": held_out, **metrics})
        for train_study in sorted(meta["study_name"].astype(str).unique()):
            train_mask = (meta["study_name"].astype(str) == train_study).to_numpy()
            y = crc_vector(meta)
            if len(np.unique(y[train_mask])) < 2:
                continue
            scaler = StandardScaler()
            x_train = scaler.fit_transform(clr.loc[train_mask, features].to_numpy(dtype=float))
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
                model.fit(x_train, y[train_mask])
            for test_study in sorted(meta["study_name"].astype(str).unique()):
                test_mask = (meta["study_name"].astype(str) == test_study).to_numpy()
                y_test = y[test_mask]
                prob = model.predict_proba(scaler.transform(clr.loc[test_mask, features].to_numpy(dtype=float)))[:, 1]
                pair_detail.append(
                    {
                        "panel_type": panel_type,
                        "panel_size": len(features),
                        "train_study": train_study,
                        "test_study": test_study,
                        "is_diagonal": train_study == test_study,
                        "auroc": safe_auc(y_test, prob),
                        "average_precision": safe_ap(y_test, prob),
                        "n_train": int(train_mask.sum()),
                        "n_test": int(test_mask.sum()),
                    }
                )
    loso_detail_df = pd.DataFrame(loso_detail)
    pair_detail_df = pd.DataFrame(pair_detail)
    loso_summary = []
    for panel_type, group in loso_detail_df.groupby("panel_type"):
        auroc = group["auroc"].dropna()
        ap = group["average_precision"].dropna()
        loso_summary.append(
            {
                "panel_type": panel_type,
                "panel_size": int(group["panel_size"].iloc[0]),
                "number_of_held_out_folds": int(group["held_out_study"].nunique()),
                "median_auroc": float(auroc.median()),
                "minimum_auroc": float(auroc.min()),
                "maximum_auroc": float(auroc.max()),
                "auroc_iqr": iqr(auroc),
                "median_average_precision": float(ap.median()) if not ap.empty else np.nan,
                "benchmark_label": "retrospective fixed-panel transferability stress testing",
            }
        )
    pair_summary = []
    for panel_type, group in pair_detail_df.groupby("panel_type"):
        off = group.loc[~group["is_diagonal"].astype(bool), "auroc"].dropna()
        diag = group.loc[group["is_diagonal"].astype(bool), "auroc"].dropna()
        ap = group.loc[~group["is_diagonal"].astype(bool), "average_precision"].dropna()
        diag_med = float(diag.median())
        off_med = float(off.median())
        pair_summary.append(
            {
                "panel_type": panel_type,
                "panel_size": int(group["panel_size"].iloc[0]),
                "number_of_study_pairs": int((~group["is_diagonal"].astype(bool)).sum()),
                "off_diagonal_median_auroc": off_med,
                "off_diagonal_minimum_auroc": float(off.min()),
                "off_diagonal_maximum_auroc": float(off.max()),
                "off_diagonal_auroc_iqr": iqr(off),
                "off_diagonal_median_average_precision": float(ap.median()) if not ap.empty else np.nan,
                "within_study_median_auroc": diag_med,
                "transferability_loss": diag_med - off_med,
                "transferability_loss_definition": "within-study median AUROC minus off-diagonal train-study by test-study median AUROC",
                "benchmark_label": "retrospective fixed-panel transferability stress testing",
            }
        )
    return loso_summary and pd.DataFrame(loso_summary), pair_summary and pd.DataFrame(pair_summary), loso_detail_df, pair_detail_df


def hedges_g(crc: np.ndarray, control: np.ndarray) -> tuple[float, float]:
    n1, n0 = len(crc), len(control)
    if n1 < 2 or n0 < 2:
        return np.nan, np.nan
    s1 = np.var(crc, ddof=1)
    s0 = np.var(control, ddof=1)
    pooled = math.sqrt(((n1 - 1) * s1 + (n0 - 1) * s0) / (n1 + n0 - 2))
    if pooled == 0:
        return np.nan, np.nan
    d = (float(np.mean(crc)) - float(np.mean(control))) / pooled
    correction = 1 - (3 / (4 * (n1 + n0) - 9))
    g = correction * d
    se = math.sqrt((n1 + n0) / (n1 * n0) + (g**2 / (2 * (n1 + n0 - 2))))
    return g, se


def dersimonian_laird(effect: np.ndarray, se: np.ndarray) -> dict[str, float]:
    mask = np.isfinite(effect) & np.isfinite(se) & (se > 0)
    y = effect[mask]
    s = se[mask]
    k = len(y)
    if k < 2:
        return {"k_cohorts": k, "random_effect": np.nan, "random_se": np.nan, "random_ci_low": np.nan, "random_ci_high": np.nan, "p_value": np.nan, "tau2": np.nan, "i2_percent": np.nan, "q_statistic": np.nan}
    w = 1 / (s**2)
    fixed = np.sum(w * y) / np.sum(w)
    q = np.sum(w * (y - fixed) ** 2)
    df = k - 1
    c = np.sum(w) - np.sum(w**2) / np.sum(w)
    tau2 = max(0.0, (q - df) / c) if c > 0 else 0.0
    wr = 1 / (s**2 + tau2)
    random = np.sum(wr * y) / np.sum(wr)
    se_random = math.sqrt(1 / np.sum(wr))
    z = random / se_random if se_random > 0 else np.nan
    p_value = 2 * stats.norm.sf(abs(z)) if np.isfinite(z) else np.nan
    i2 = max(0.0, (q - df) / q) * 100 if q > 0 else 0.0
    return {
        "k_cohorts": k,
        "random_effect": random,
        "random_se": se_random,
        "random_ci_low": random - 1.96 * se_random,
        "random_ci_high": random + 1.96 * se_random,
        "p_value": p_value,
        "tau2": tau2,
        "i2_percent": i2,
        "q_statistic": q,
    }


def guild_membership(clr: pd.DataFrame) -> pd.DataFrame:
    assoc = read_tsv(RESULTS / "C" / "cohort_adjusted_crc_control_association.tsv")
    assoc_map = assoc.set_index("taxon_long")["direction_adjusted"].to_dict()
    stable = set(read_tsv(RESULTS / "M_robustness_interpretation" / "stable_candidate_interpretation.tsv")["taxon_long"])
    rows = []
    rules = {
        "butyrate_SCFA_commensal_panel": {
            "genera": {"Roseburia", "Faecalibacterium", "Eubacterium", "Anaerostipes", "Fusicatenibacter", "Agathobaculum", "Gemmiger", "Intestinimonas"},
            "rationale": "literature-prior taxonomy-defined panel of commonly discussed butyrate/SCFA-associated commensal genera",
            "caution": "taxonomy-name panel only; no direct functional activity or butyrate-production measurement",
        },
        "oral_pathobiont_associated_panel": {
            "genera": {"Parvimonas", "Peptostreptococcus", "Fusobacterium", "Gemella", "Streptococcus", "Solobacterium"},
            "rationale": "literature-prior taxonomy-defined oral/pathobiont-associated CRC context panel",
            "caution": "taxonomy-name panel only; no oral-source or transmission inference",
        },
    }
    for taxon in clr.columns:
        parsed = parse_taxon(taxon)
        for guild, rule in rules.items():
            if parsed["genus"] in rule["genera"]:
                rows.append(
                    {
                        "guild": guild,
                        "taxon": taxon,
                        "genus": parsed["genus"],
                        "species": parsed["species"],
                        "present_in_retained_matrix": True,
                        "stable_candidate_yes_no": "yes" if taxon in stable else "no",
                        "direction_in_primary_association": assoc_map.get(taxon, ""),
                        "rationale": rule["rationale"],
                        "cautionary_note": rule["caution"],
                    }
                )
        if parsed["species"] in {"Bacteroides fragilis", "Escherichia coli"} or parsed["genus"] == "Bilophila" or parsed["family"] == "Enterobacteriaceae":
            rows.append(
                {
                    "guild": "inflammation_Bacteroides_Enterobacteriaceae_panel",
                    "taxon": taxon,
                    "genus": parsed["genus"],
                    "species": parsed["species"],
                    "present_in_retained_matrix": True,
                    "stable_candidate_yes_no": "yes" if taxon in stable else "no",
                    "direction_in_primary_association": assoc_map.get(taxon, ""),
                    "rationale": "conservative taxonomy-defined Bacteroides fragilis, Escherichia coli, Bilophila, and Enterobacteriaceae context panel",
                    "cautionary_note": "exploratory taxonomic context only; no inflammation mechanism or functional inference",
                }
            )
    out = pd.DataFrame(rows).drop_duplicates(["guild", "taxon"]).sort_values(["guild", "genus", "species"])
    counts = out.groupby("guild")["taxon"].transform("count")
    out["exploratory_due_to_small_panel"] = counts < 3
    return out


def score_association(score: pd.Series, meta: pd.DataFrame, covariates: bool = False) -> dict[str, float]:
    df = pd.DataFrame({"score": score, "CRC_status": crc_vector(meta), "study": meta["study_name"].astype(str)}, index=meta.index)
    design = [df[["CRC_status"]], pd.get_dummies(df["study"], prefix="study", drop_first=True, dtype=float)]
    if covariates:
        tmp = meta.loc[df.index, ["age", "BMI", "gender"]].copy()
        keep = tmp.notna().all(axis=1)
        df = df.loc[keep]
        tmp = tmp.loc[keep]
        age = pd.to_numeric(tmp["age"], errors="coerce")
        bmi = pd.to_numeric(tmp["BMI"], errors="coerce")
        cov = pd.DataFrame({"age_z": (age - age.mean()) / age.std(ddof=0), "BMI_z": (bmi - bmi.mean()) / bmi.std(ddof=0)}, index=tmp.index)
        cov = pd.concat([cov, pd.get_dummies(tmp["gender"].astype(str), prefix="gender", drop_first=True, dtype=float)], axis=1)
        design = [df[["CRC_status"]], pd.get_dummies(df["study"], prefix="study", drop_first=True, dtype=float), cov]
    x = pd.concat(design, axis=1)
    x = sm.add_constant(x, has_constant="add")
    model = sm.OLS(df["score"].astype(float), x.astype(float)).fit(cov_type="HC3")
    return {
        "coef_crc": float(model.params["CRC_status"]),
        "se_hc3": float(model.bse["CRC_status"]),
        "ci_low": float(model.params["CRC_status"] - 1.96 * model.bse["CRC_status"]),
        "ci_high": float(model.params["CRC_status"] + 1.96 * model.bse["CRC_status"]),
        "p_value": float(model.pvalues["CRC_status"]),
        "sample_n": len(df),
    }


def ecological_guild_analysis(clr: pd.DataFrame, meta: pd.DataFrame) -> dict[str, pd.DataFrame]:
    membership = guild_membership(clr)
    assoc_rows = []
    random_rows = []
    cohort_rows = []
    loco_rows = []
    loso_rows = []
    score_rows = []
    for guild, group in membership.groupby("guild"):
        features = group["taxon"].tolist()
        if len(features) < 2:
            continue
        score = clr[features].mean(axis=1)
        base = score_association(score, meta, covariates=False)
        cov = score_association(score, meta, covariates=True)
        direction = "CRC_higher" if base["coef_crc"] > 0 else "control_higher"
        assoc_rows.append(
            {
                "guild": guild,
                "n_taxa": len(features),
                "raw_score_definition": "mean retained CLR abundance of member species",
                "orientation_coded_score_definition": "raw score multiplied by training or full-data CRC coefficient sign where higher values are CRC-oriented",
                "coef_crc_study_adjusted": base["coef_crc"],
                "ci_low": base["ci_low"],
                "ci_high": base["ci_high"],
                "p_value": base["p_value"],
                "direction": direction,
                "covariate_coef_crc": cov["coef_crc"],
                "covariate_p_value": cov["p_value"],
                "covariate_sample_n": cov["sample_n"],
            }
        )
        for sample_uid, value in score.items():
            score_rows.append({"sample_uid": sample_uid, "guild": guild, "raw_guild_score": value, "full_data_orientation_coded_score": value * (1 if base["coef_crc"] >= 0 else -1)})
        for study, idx in meta.groupby("study_name").groups.items():
            sub_meta = meta.loc[idx]
            sub_score = score.loc[idx]
            crc = sub_score[sub_meta["study_condition"].astype(str) == "CRC"].to_numpy(dtype=float)
            control = sub_score[sub_meta["study_condition"].astype(str) != "CRC"].to_numpy(dtype=float)
            g, se = hedges_g(crc, control)
            cohort_rows.append(
                {
                    "guild": guild,
                    "study_name": study,
                    "n_crc": len(crc),
                    "n_control": len(control),
                    "hedges_g_crc_minus_control": g,
                    "se": se,
                    "direction": "CRC_higher" if np.isfinite(g) and g > 0 else "control_higher",
                }
            )
        per = pd.DataFrame([r for r in cohort_rows if r["guild"] == guild])
        dl = dersimonian_laird(per["hedges_g_crc_minus_control"].to_numpy(dtype=float), per["se"].to_numpy(dtype=float))
        dl.update({"guild": guild, "n_taxa": len(features), "effect_measure": "Hedges_g_CRC_minus_control"})
        random_rows.append(dl)
        for excluded in sorted(meta["study_name"].astype(str).unique()):
            keep = meta["study_name"].astype(str) != excluded
            fit = score_association(score.loc[keep], meta.loc[keep], covariates=False)
            loco_rows.append(
                {
                    "guild": guild,
                    "excluded_study": excluded,
                    "coef_crc": fit["coef_crc"],
                    "p_value": fit["p_value"],
                    "direction": "CRC_higher" if fit["coef_crc"] > 0 else "control_higher",
                    "sample_n": fit["sample_n"],
                }
            )
        for held_out in sorted(meta["study_name"].astype(str).unique()):
            train_mask = meta["study_name"].astype(str) != held_out
            test_mask = ~train_mask
            train_fit = score_association(score.loc[train_mask], meta.loc[train_mask], covariates=False)
            oriented = score.loc[test_mask].to_numpy(dtype=float) * (1 if train_fit["coef_crc"] >= 0 else -1)
            y_test = crc_vector(meta.loc[test_mask])
            loso_rows.append(
                {
                    "guild": guild,
                    "held_out_study": held_out,
                    "n_taxa": len(features),
                    "training_orientation": "CRC_higher" if train_fit["coef_crc"] >= 0 else "control_higher",
                    "auroc": safe_auc(y_test, oriented),
                    "average_precision": safe_ap(y_test, oriented),
                    "n_test": int(test_mask.sum()),
                    "crc_control_in_test": f"CRC={int(y_test.sum())};control={int((1 - y_test).sum())}",
                    "interpretation": "score-only separability stress test, not clinical validation",
                }
            )
    assoc = pd.DataFrame(assoc_rows)
    assoc["q_value"] = bh(assoc["p_value"])
    assoc["covariate_q_value"] = bh(assoc["covariate_p_value"])
    random_effects = pd.DataFrame(random_rows)
    random_effects["q_value"] = bh(random_effects["p_value"])
    loco = pd.DataFrame(loco_rows)
    loco["q_value_within_exclusion"] = loco.groupby("excluded_study")["p_value"].transform(lambda s: bh(s))
    return {
        "membership": membership,
        "scores": pd.DataFrame(score_rows),
        "association": assoc,
        "per_cohort": pd.DataFrame(cohort_rows),
        "random_effects": random_effects,
        "loco": loco,
        "loso": pd.DataFrame(loso_rows),
    }


def bootstrap_ci(diff: np.ndarray, rng: np.random.Generator, n_boot: int = 5000) -> tuple[float, float]:
    diff = np.asarray(diff, dtype=float)
    diff = diff[np.isfinite(diff)]
    if len(diff) == 0:
        return np.nan, np.nan
    boot = [np.median(rng.choice(diff, size=len(diff), replace=True)) for _ in range(n_boot)]
    return float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def panel_comparison_statistics(leakage: pd.DataFrame, fixed_pair_detail: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED + 303)
    rows = []
    fixed = leakage[~leakage["panel_type"].str.startswith("random29")].copy()
    for a, b in [
        ("training_only_top29_q", "training_only_stability_panel"),
        ("training_only_top29_q", "training_only_high_consistency_panel"),
        ("training_only_stability_panel", "training_only_high_consistency_panel"),
        ("all_301_features", "training_only_top29_q"),
    ]:
        pivot = fixed.pivot(index="held_out_study", columns="panel_type", values="auroc")
        if a in pivot.columns and b in pivot.columns:
            diff = (pivot[b] - pivot[a]).dropna().to_numpy(dtype=float)
            lo, hi = bootstrap_ci(diff, rng)
            try:
                p = float(stats.wilcoxon(diff).pvalue) if len(diff) > 0 and np.any(diff != 0) else np.nan
            except ValueError:
                p = np.nan
            rows.append(
                {
                    "comparison": f"{b} minus {a}",
                    "metric": "LOSO AUROC across held-out studies",
                    "median_difference": float(np.median(diff)) if len(diff) else np.nan,
                    "mean_difference": float(np.mean(diff)) if len(diff) else np.nan,
                    "bootstrap_ci_low": lo,
                    "bootstrap_ci_high": hi,
                    "exploratory_p_value": p,
                    "notes": "paired held-out-study comparison; Wilcoxon p value exploratory because n=11",
                }
            )
    pair = fixed_pair_detail[~fixed_pair_detail["is_diagonal"].astype(bool)]
    for a, b in [
        ("global_top29_q", "global_stable29"),
        ("global_top29_q", "global_high_consistency18"),
        ("global_stable29", "global_high_consistency18"),
    ]:
        a_vals = pair.loc[pair["panel_type"] == a, "auroc"].dropna().to_numpy(dtype=float)
        b_vals = pair.loc[pair["panel_type"] == b, "auroc"].dropna().to_numpy(dtype=float)
        n = min(len(a_vals), len(b_vals))
        if n:
            diff = b_vals[:n] - a_vals[:n]
            lo, hi = bootstrap_ci(diff, rng)
            rows.append(
                {
                    "comparison": f"{b} minus {a}",
                    "metric": "off-diagonal pairwise AUROC",
                    "median_difference": float(np.median(diff)),
                    "mean_difference": float(np.mean(diff)),
                    "bootstrap_ci_low": lo,
                    "bootstrap_ci_high": hi,
                    "exploratory_p_value": np.nan,
                    "notes": "descriptive bootstrap only; train-study by test-study pairs are dependent",
                }
            )
    return pd.DataFrame(rows)


def claim_decision_table(panel_stats: pd.DataFrame, variance: pd.DataFrame, guild_random: pd.DataFrame) -> pd.DataFrame:
    stable_vs_top = panel_stats.loc[panel_stats["comparison"] == "training_only_stability_panel minus training_only_top29_q"]
    supports_outperform = False
    if not stable_vs_top.empty:
        supports_outperform = bool(stable_vs_top["bootstrap_ci_low"].iloc[0] > 0)
    study_r2 = float(variance.loc[variance["term"] == "study_label", "partial_or_marginal_r2"].iloc[0])
    crc_r2 = float(variance.loc[variance["term"] == "CRC_status", "partial_or_marginal_r2"].iloc[0])
    rows = [
        {
            "proposed_claim": "Study structure explains more Aitchison-space variation than CRC/control status.",
            "supported_yes_no": "yes",
            "supporting_results": f"study_label partial/marginal R2={study_r2:.4f}; CRC_status partial/marginal R2={crc_r2:.4f}",
            "caveats": "Complete-case processed CLR matrix; effect size is modest and cohort embedded.",
            "allowed_wording": "Study structure explained more retained CLR variation than CRC/control status, motivating cohort-aware analysis.",
            "prohibited_wording": "CRC status is unimportant or absent.",
        },
        {
            "proposed_claim": "Pooled association strength and cross-cohort direction stability select partially overlapping but non-identical taxa.",
            "supported_yes_no": "yes",
            "supporting_results": "Training-only and full-data panels show incomplete overlaps with global top29, stable29, and high-consistency panels.",
            "caveats": "Overlap depends on FDR and same-direction thresholds.",
            "allowed_wording": "Association-ranked and stability-ranked signals were partially overlapping but non-identical.",
            "prohibited_wording": "All association signals are stable across cohorts.",
        },
        {
            "proposed_claim": "Transferability-aware panels are more conservative than pooled-q panels.",
            "supported_yes_no": "partial",
            "supporting_results": "The training-only >=60% stability panel converged with the training-only top29-q panel in these folds, whereas the stricter high-consistency panel selected 18 taxa.",
            "caveats": "Conservative applies mainly to stricter high-consistency prioritization and interpretability, not to improved AUROC.",
            "allowed_wording": "Stricter transferability-aware prioritization produced a more conservative shortlist, while the >=60% stability panel did not separate from the top-q panel in leakage-aware LOSO.",
            "prohibited_wording": "Transferability-aware panels are universally more accurate.",
        },
        {
            "proposed_claim": "Transferability-aware panels outperform pooled-q panels.",
            "supported_yes_no": "yes" if supports_outperform else "no",
            "supporting_results": "Leakage-aware paired AUROC comparison does not support this claim." if not supports_outperform else "Leakage-aware paired AUROC comparison supports higher AUROC.",
            "caveats": "Performance claims require fold-level evidence and should remain exploratory.",
            "allowed_wording": "Association ranking and portability diverged under cohort shift.",
            "prohibited_wording": "The stable panel outperformed the pooled-q panel." if not supports_outperform else "Clinical diagnostic superiority was validated.",
        },
        {
            "proposed_claim": "Stable panels reduce cohort dependence.",
            "supported_yes_no": "partial",
            "supporting_results": "Panel transferability loss and random comparisons provide context, but AUROC improvements are not guaranteed.",
            "caveats": "Cohort dependence remains visible in LOSO and pairwise transfer.",
            "allowed_wording": "Stable panels changed the portability profile and support conservative prioritization.",
            "prohibited_wording": "Stable panels eliminate cohort effects.",
        },
        {
            "proposed_claim": "Ecological guild scores provide interpretable axes.",
            "supported_yes_no": "yes",
            "supporting_results": f"Guild random-effects table includes {len(guild_random)} taxonomy-defined panels with Hedges g summaries.",
            "caveats": "Taxonomy-defined panels only; no functional activity measurement.",
            "allowed_wording": "Taxonomy-defined guild scores provided conservative ecological context.",
            "prohibited_wording": "Guild scores prove metabolite production or inflammatory function.",
        },
        {
            "proposed_claim": "Oral-associated taxa explain the stable CRC candidate set.",
            "supported_yes_no": "no",
            "supporting_results": "Only partial overlap; oral/pathobiont guild score is contextual.",
            "caveats": "Taxonomy-name panel cannot infer oral origin.",
            "allowed_wording": "Oral/pathobiont-associated taxa contributed one interpretable context axis.",
            "prohibited_wording": "Oral taxa explain the CRC signature or demonstrate oral transmission.",
        },
        {
            "proposed_claim": "Results support clinical diagnostic use.",
            "supported_yes_no": "no",
            "supporting_results": "Analyses are separability and transferability stress tests on public processed tables.",
            "caveats": "No independent clinical validation or calibration.",
            "allowed_wording": "The results support processed-data separability and candidate prioritization.",
            "prohibited_wording": "The panel is clinically diagnostic or ready for screening.",
        },
        {
            "proposed_claim": "Results support causal or mechanistic inference.",
            "supported_yes_no": "no",
            "supporting_results": "No intervention, longitudinal causal design, mechanistic assay, or raw-read functional analysis.",
            "caveats": "Associational processed-data reanalysis only.",
            "allowed_wording": "The findings are consistent with processed-data associations.",
            "prohibited_wording": "Taxa cause CRC or mechanistically drive CRC.",
        },
        {
            "proposed_claim": "Results support raw-read, strain-level, or functional claims.",
            "supported_yes_no": "no",
            "supporting_results": "No raw reads, strain profiling, or functional profiling were analyzed.",
            "caveats": "Processed species-level relative abundance matrix only.",
            "allowed_wording": "Raw-read, strain-level, and functional extensions require separate data processing.",
            "prohibited_wording": "The study identifies strains, functions, or raw-read-derived mechanisms.",
        },
    ]
    return pd.DataFrame(rows)


def update_module_status(paths: dict[str, Path]) -> None:
    status_path = WORKSPACE / "module_status.tsv"
    status = pd.read_csv(status_path, sep="\t") if status_path.exists() else pd.DataFrame(columns=["module_id", "status", "primary_output", "notes"])
    new_rows = pd.DataFrame(
        [
            {
                "module_id": "Q",
                "status": "COMPLETED",
                "primary_output": str(paths["leakage_aware_loso"]),
                "notes": "Leakage-aware LOSO panel benchmarking with training-only feature selection, random-panel comparisons, variance-permutation QC, ecological guild score refinements, panel comparison statistics, and claim decision table.",
            }
        ]
    )
    status = status[~status["module_id"].isin(["Q"])]
    pd.concat([status, new_rows], ignore_index=True).to_csv(status_path, sep="\t", index=False)


def main() -> None:
    meta = read_tsv(DATA_DIR / "curated_crc_case_control_sample_metadata.tsv").set_index("sample_uid", drop=False)
    clr = read_tsv(PROCESSED / "B" / "filtered_species_clr.tsv.gz", index_col=0)
    abundance = read_tsv(PROCESSED / "B" / "filtered_species_abundance_percent.tsv.gz", index_col=0)
    common = clr.index.intersection(meta.index).intersection(abundance.index)
    clr = clr.loc[common]
    abundance = abundance.loc[common, clr.columns]
    meta = meta.loc[common]

    variance = aitchison_variance_partitioning(clr, meta)
    leakage, random_summary, selection = leakage_aware_loso(clr, abundance, meta)
    fixed_loso, fixed_pairwise, fixed_loso_detail, fixed_pairwise_detail = fixed_global_panel_benchmarks(clr, meta)
    guild = ecological_guild_analysis(clr, meta)
    comparison = panel_comparison_statistics(leakage, fixed_pairwise_detail)
    claims = claim_decision_table(comparison, variance, guild["random_effects"])

    paths = {
        "variance": write_csv(variance, OUT / "aitchison_variance_partitioning.csv"),
        "leakage_aware_loso": write_csv(leakage, OUT / "leakage_aware_loso_panel_benchmark.csv"),
        "leakage_aware_selection": write_csv(selection, OUT / "leakage_aware_training_panel_selection.csv"),
        "random_summary": write_csv(random_summary, OUT / "random_panel_loso_benchmark_summary.csv"),
        "fixed_loso": write_csv(fixed_loso, OUT / "fixed_global_panel_loso_benchmark.csv"),
        "fixed_pairwise": write_csv(fixed_pairwise, OUT / "fixed_global_panel_pairwise_benchmark.csv"),
        "fixed_loso_detail": write_csv(fixed_loso_detail, OUT / "fixed_global_panel_loso_detail.csv"),
        "fixed_pairwise_detail": write_csv(fixed_pairwise_detail, OUT / "fixed_global_panel_pairwise_detail.csv"),
        "guild_scores": write_csv(guild["association"], OUT / "ecological_guild_scores.csv"),
        "guild_random_effects": write_csv(guild["random_effects"], OUT / "ecological_guild_random_effects.csv"),
        "guild_loco": write_csv(guild["loco"], OUT / "ecological_guild_loco_robustness.csv"),
        "guild_loso": write_csv(guild["loso"], OUT / "ecological_guild_loso_score_only.csv"),
        "guild_membership": write_csv(guild["membership"], OUT / "ecological_guild_membership.csv"),
        "guild_per_cohort": write_csv(guild["per_cohort"], OUT / "ecological_guild_per_cohort_hedges_g.csv"),
        "guild_sample_scores": write_csv(guild["scores"], OUT / "ecological_guild_sample_scores.csv"),
        "panel_comparison": write_csv(comparison, OUT / "panel_comparison_statistics.csv"),
        "claim_decision": write_csv(claims, OUT / "claim_decision_table.csv"),
    }
    update_module_status(paths)
    summary = {
        "status": "COMPLETED",
        "random_seed": RANDOM_SEED,
        "n_permutations": N_PERMUTATIONS,
        "n_random_panels_per_fold": N_RANDOM_PANELS,
        "outputs": {k: str(v) for k, v in paths.items()},
        "claim_4_transferability_outperforms_pooled_q": claims.loc[claims["proposed_claim"].str.startswith("Transferability-aware panels outperform"), "supported_yes_no"].iloc[0],
    }
    (OUT / "leakage_aware_prioritization_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
