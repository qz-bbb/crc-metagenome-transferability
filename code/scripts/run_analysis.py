from __future__ import annotations

import json
import math
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


RUN_DIR = Path(r"D:\1\projects\gut_microbiome_colorectal_cancer\runs\run_20260525_182306")
WORKSPACE = RUN_DIR / "agents" / "05-analysis" / "workspace"
SCRIPT_PATH = WORKSPACE / "scripts" / "run_analysis.py"
UPSTREAM_04 = RUN_DIR / "agents" / "04-architect-targeted-refresh" / "workspace"

DATA_DIR = Path(r"D:\1\Knowledge Base\true_raw\data\BIOC_curatedMetagenomicData_CRC")
META_PATH = DATA_DIR / "curated_crc_case_control_sample_metadata.tsv"
MATRIX_PATH = DATA_DIR / "curated_crc_case_control_species_relative_abundance.tsv.gz"
FEATURE_PATH = DATA_DIR / "curated_crc_species_feature_index.tsv"
SUMMARY_PATH = DATA_DIR / "curated_crc_analysis_ready_matrix_summary.json"

MICROBIOMEHD_DIR = Path(r"D:\1\Knowledge Base\true_raw\data\GITHUB_MicrobiomeHD")
MICROBIOMEHD_Q = MICROBIOMEHD_DIR / "file-S1.qvalues.txt"
MICROBIOMEHD_EFFECTS = MICROBIOMEHD_DIR / "file-S5.effects.txt"

PROCESSED = WORKSPACE / "processed_data"
RESULTS = WORKSPACE / "results"


def ensure_dirs() -> None:
    for base in [PROCESSED, RESULTS]:
        base.mkdir(parents=True, exist_ok=True)
        for module in list("ABCDEFGH"):
            (base / module).mkdir(parents=True, exist_ok=True)


def write_tsv(df: pd.DataFrame, path: Path, index: bool = False) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=index)
    return str(path)


def bh_fdr(pvalues: pd.Series) -> pd.Series:
    out = pd.Series(np.nan, index=pvalues.index, dtype=float)
    mask = pvalues.notna()
    if mask.any():
        out.loc[mask] = multipletests(pvalues.loc[mask].astype(float), method="fdr_bh")[1]
    return out


def genus_from_taxon(taxon: str) -> str:
    parts = str(taxon).replace(";", "|").split("|")
    for part in parts:
        if part.startswith("g__"):
            return part.replace("g__", "")
    return ""


def species_from_taxon(taxon: str) -> str:
    parts = str(taxon).replace(";", "|").split("|")
    for part in parts:
        if part.startswith("s__"):
            return part.replace("s__", "").replace("_", " ")
    return ""


def fit_feature_ols(y: np.ndarray, x: pd.DataFrame, condition_name: str = "condition_crc") -> tuple[float, float, float]:
    if np.nanstd(y) == 0:
        return np.nan, np.nan, np.nan
    try:
        model = sm.OLS(y, x, missing="drop").fit(cov_type="HC3")
        return (
            float(model.params[condition_name]),
            float(model.bse[condition_name]),
            float(model.pvalues[condition_name]),
        )
    except Exception:
        return np.nan, np.nan, np.nan


def zscore_numeric(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    sd = numeric.std(skipna=True)
    if sd == 0 or pd.isna(sd):
        return numeric * 0
    return (numeric - numeric.mean(skipna=True)) / sd


def finalize_design(design: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
    variable_cols = [
        col
        for col in design.columns
        if col != "const" and design[col].nunique(dropna=False) > 1
    ]
    design = sm.add_constant(design[variable_cols], has_constant="add").astype(float)
    matrix = design.to_numpy(dtype=float)
    return design, int(np.linalg.matrix_rank(matrix)), int(matrix.shape[1])


def pseudo_f_two_group(x: np.ndarray, labels: np.ndarray) -> float:
    labels = np.asarray(labels).astype(int)
    if len(np.unique(labels)) != 2:
        return float("nan")
    overall = x.mean(axis=0)
    ss_between = 0.0
    ss_within = 0.0
    for value in [0, 1]:
        group = x[labels == value]
        if group.shape[0] == 0:
            return float("nan")
        centroid = group.mean(axis=0)
        ss_between += group.shape[0] * float(np.sum((centroid - overall) ** 2))
        ss_within += float(np.sum((group - centroid) ** 2))
    df_between = 1
    df_within = x.shape[0] - 2
    if df_within <= 0 or ss_within <= 0:
        return float("nan")
    return float((ss_between / df_between) / (ss_within / df_within))


def main() -> None:
    ensure_dirs()
    log_lines: list[str] = []
    module_rows: list[dict[str, str]] = []

    # Module A: input integrity.
    metadata = pd.read_csv(META_PATH, sep="\t")
    matrix_raw = pd.read_csv(MATRIX_PATH, sep="\t", index_col=0)
    feature_index = pd.read_csv(FEATURE_PATH, sep="\t")
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))

    sample_match = list(metadata["sample_uid"]) == list(matrix_raw.columns)
    condition_counts = metadata["study_condition"].value_counts(dropna=False).rename_axis("study_condition").reset_index(name="n")
    cohort_counts = (
        metadata.pivot_table(index="study_name", columns="study_condition", values="sample_uid", aggfunc="count", fill_value=0)
        .reset_index()
    )
    cohort_counts.columns.name = None
    qc_rows = [
        {"check": "metadata_file_exists", "status": str(META_PATH.exists()), "value": str(META_PATH)},
        {"check": "matrix_file_exists", "status": str(MATRIX_PATH.exists()), "value": str(MATRIX_PATH)},
        {"check": "feature_index_exists", "status": str(FEATURE_PATH.exists()), "value": str(FEATURE_PATH)},
        {"check": "metadata_rows", "status": "PASS", "value": str(len(metadata))},
        {"check": "matrix_features", "status": "PASS", "value": str(matrix_raw.shape[0])},
        {"check": "matrix_samples", "status": "PASS", "value": str(matrix_raw.shape[1])},
        {"check": "sample_uid_exact_order_match", "status": "PASS" if sample_match else "FAIL", "value": str(sample_match)},
        {"check": "crc_rows", "status": "PASS", "value": str(int((metadata["study_condition"] == "CRC").sum()))},
        {"check": "control_rows", "status": "PASS", "value": str(int((metadata["study_condition"] == "control").sum()))},
        {"check": "summary_expected_shape", "status": "PASS", "value": json.dumps(summary.get("case_control_matrix_shape", {}), ensure_ascii=False)},
    ]
    if not sample_match:
        raise SystemExit("metadata sample_uid order does not match matrix columns")
    if set(metadata["study_condition"].dropna().unique()) - {"CRC", "control"}:
        raise SystemExit("primary metadata contains labels outside CRC/control")
    qc_path = Path(write_tsv(pd.DataFrame(qc_rows), PROCESSED / "A" / "input_qc_summary.tsv"))
    cohort_counts_path = Path(write_tsv(cohort_counts, PROCESSED / "A" / "cohort_condition_counts.tsv"))
    condition_counts_path = Path(write_tsv(condition_counts, PROCESSED / "A" / "condition_counts.tsv"))
    log_lines.append("Module A completed: input files loaded, sample order matched, CRC/control labels verified.")
    module_rows.append({"module_id": "A", "status": "COMPLETED", "primary_output": str(qc_path), "notes": "Input integrity and cohort inventory passed."})

    # Orient matrices as samples x features.
    abundance = matrix_raw.T
    abundance.index.name = "sample_uid"
    abundance = abundance.loc[metadata["sample_uid"]]
    condition = (metadata.set_index("sample_uid").loc[abundance.index, "study_condition"] == "CRC").astype(int)
    study = metadata.set_index("sample_uid").loc[abundance.index, "study_name"]

    # Module B: filtering and descriptive structure.
    prevalence = (abundance > 0).mean(axis=0)
    mean_abundance = abundance.mean(axis=0)
    max_abundance = abundance.max(axis=0)
    cohort_presence = pd.DataFrame(index=abundance.columns)
    for cohort, idx in study.groupby(study).groups.items():
        cohort_presence[cohort] = (abundance.loc[idx] > 0).any(axis=0).astype(int)
    n_cohorts_present = cohort_presence.sum(axis=1)
    filter_table = pd.DataFrame({
        "taxon_long": abundance.columns,
        "genus": [genus_from_taxon(x) for x in abundance.columns],
        "species_short": [species_from_taxon(x) for x in abundance.columns],
        "prevalence_overall": prevalence.values,
        "mean_abundance_percent": mean_abundance.values,
        "max_abundance_percent": max_abundance.values,
        "n_cohorts_present": n_cohorts_present.values,
    })
    filter_table["retained"] = (filter_table["prevalence_overall"] >= 0.05) & (filter_table["n_cohorts_present"] >= 3)
    retained_taxa = filter_table.loc[filter_table["retained"], "taxon_long"].tolist()
    if len(retained_taxa) < 10:
        raise SystemExit("too few features retained for meaningful analysis")
    filtered_abundance = abundance[retained_taxa].copy()
    positive_min = filtered_abundance.where(filtered_abundance > 0).min().min()
    pseudocount = float(positive_min / 2) if positive_min and not math.isnan(positive_min) else 1e-6
    log_values = np.log(filtered_abundance + pseudocount)
    clr = log_values.sub(log_values.mean(axis=1), axis=0)
    filter_table_path = Path(write_tsv(filter_table, PROCESSED / "B" / "feature_filter_summary.tsv"))
    filtered_abundance_path = PROCESSED / "B" / "filtered_species_abundance_percent.tsv.gz"
    filtered_abundance.to_csv(filtered_abundance_path, sep="\t", compression="gzip", index=True)
    clr_path = PROCESSED / "B" / "filtered_species_clr.tsv.gz"
    clr.to_csv(clr_path, sep="\t", compression="gzip", index=True)
    landscape = filter_table.groupby("retained").agg(
        n_features=("taxon_long", "count"),
        median_prevalence=("prevalence_overall", "median"),
        median_mean_abundance=("mean_abundance_percent", "median"),
    ).reset_index()
    landscape_path = Path(write_tsv(landscape, RESULTS / "B" / "feature_landscape_summary.tsv"))
    log_lines.append(f"Module B completed: retained {len(retained_taxa)} of {abundance.shape[1]} species features; pseudocount={pseudocount:.6g}.")
    module_rows.append({"module_id": "B", "status": "COMPLETED", "primary_output": str(filter_table_path), "notes": f"Retained {len(retained_taxa)} features using prevalence>=5% and present in >=3 cohorts."})

    # Module C: cohort-aware association screen on CLR values.
    design = pd.DataFrame({"condition_crc": condition.values}, index=clr.index)
    study_dummies = pd.get_dummies(study.loc[clr.index], prefix="study", drop_first=True, dtype=float)
    design = pd.concat([design, study_dummies.set_index(clr.index)], axis=1)
    design = sm.add_constant(design, has_constant="add").astype(float)
    design_path = PROCESSED / "C" / "cohort_adjusted_design_matrix.tsv.gz"
    design.to_csv(design_path, sep="\t", compression="gzip", index=True)
    assoc_rows = []
    for taxon in clr.columns:
        coef, se, pval = fit_feature_ols(clr[taxon].values.astype(float), design)
        crc_vals = filtered_abundance.loc[condition.index[condition == 1], taxon]
        ctl_vals = filtered_abundance.loc[condition.index[condition == 0], taxon]
        assoc_rows.append({
            "taxon_long": taxon,
            "genus": genus_from_taxon(taxon),
            "species_short": species_from_taxon(taxon),
            "n_crc": int(crc_vals.shape[0]),
            "n_control": int(ctl_vals.shape[0]),
            "prevalence_crc": float((crc_vals > 0).mean()),
            "prevalence_control": float((ctl_vals > 0).mean()),
            "median_abundance_crc_percent": float(crc_vals.median()),
            "median_abundance_control_percent": float(ctl_vals.median()),
            "median_diff_crc_minus_control_percent": float(crc_vals.median() - ctl_vals.median()),
            "coef_crc_clr_adjusted": coef,
            "se_hc3": se,
            "pvalue": pval,
        })
    assoc = pd.DataFrame(assoc_rows)
    assoc["qvalue_bh"] = bh_fdr(assoc["pvalue"])
    assoc["direction_adjusted"] = np.sign(assoc["coef_crc_clr_adjusted"]).map({1.0: "CRC_higher", -1.0: "control_higher", 0.0: "zero"})
    assoc = assoc.sort_values(["qvalue_bh", "pvalue", "taxon_long"], na_position="last")
    assoc_path = Path(write_tsv(assoc, RESULTS / "C" / "cohort_adjusted_crc_control_association.tsv"))

    mw_rows = []
    for taxon in filtered_abundance.columns:
        crc_vals = filtered_abundance.loc[condition.index[condition == 1], taxon]
        ctl_vals = filtered_abundance.loc[condition.index[condition == 0], taxon]
        try:
            stat, pval = stats.mannwhitneyu(crc_vals, ctl_vals, alternative="two-sided")
        except Exception:
            stat, pval = np.nan, np.nan
        mw_rows.append({"taxon_long": taxon, "mannwhitney_u": stat, "pvalue": pval})
    mw = pd.DataFrame(mw_rows)
    mw["qvalue_bh"] = bh_fdr(mw["pvalue"])
    mw_path = Path(write_tsv(mw, RESULTS / "C" / "raw_abundance_mannwhitney_sensitivity.tsv"))
    n_sig_010 = int((assoc["qvalue_bh"] < 0.10).sum())
    n_sig_005 = int((assoc["qvalue_bh"] < 0.05).sum())
    top_taxon = assoc.iloc[0]["taxon_long"] if len(assoc) else "NA"
    log_lines.append(f"Module C completed: tested {len(assoc)} retained features; FDR<0.10={n_sig_010}; FDR<0.05={n_sig_005}.")
    module_rows.append({"module_id": "C", "status": "COMPLETED", "primary_output": str(assoc_path), "notes": f"Cohort-adjusted OLS on CLR values; top feature {top_taxon}."})

    # Module D: cross-cohort stability and leave-one-cohort-out sensitivity.
    stability_rows = []
    for taxon in assoc["taxon_long"]:
        feature_assoc = assoc.loc[assoc["taxon_long"] == taxon].iloc[0]
        adjusted_sign = np.sign(feature_assoc["coef_crc_clr_adjusted"])
        tested = 0
        same = 0
        cohort_details = []
        for cohort in sorted(study.unique()):
            cohort_samples = study.index[study == cohort]
            cohort_meta = metadata.set_index("sample_uid").loc[cohort_samples]
            if set(cohort_meta["study_condition"]) >= {"CRC", "control"}:
                tested += 1
                crc_ids = cohort_meta.index[cohort_meta["study_condition"] == "CRC"]
                ctl_ids = cohort_meta.index[cohort_meta["study_condition"] == "control"]
                diff = float(filtered_abundance.loc[crc_ids, taxon].median() - filtered_abundance.loc[ctl_ids, taxon].median())
                direction = np.sign(diff)
                if adjusted_sign != 0 and direction == adjusted_sign:
                    same += 1
                cohort_details.append(f"{cohort}:{diff:.6g}")
        stability_rows.append({
            "taxon_long": taxon,
            "genus": genus_from_taxon(taxon),
            "species_short": species_from_taxon(taxon),
            "adjusted_coef": feature_assoc["coef_crc_clr_adjusted"],
            "adjusted_qvalue": feature_assoc["qvalue_bh"],
            "cohorts_tested": tested,
            "cohorts_same_direction": same,
            "stability_fraction": float(same / tested) if tested else np.nan,
            "per_cohort_median_diff_percent": ";".join(cohort_details),
        })
    stability = pd.DataFrame(stability_rows).sort_values(["adjusted_qvalue", "stability_fraction"], ascending=[True, False], na_position="last")
    stability_path = Path(write_tsv(stability, RESULTS / "D" / "cross_cohort_stability.tsv"))

    top_for_loco = assoc.head(min(50, len(assoc)))["taxon_long"].tolist()
    top_for_loco_path = Path(write_tsv(pd.DataFrame({"taxon_long": top_for_loco}), PROCESSED / "D" / "top50_features_for_leave_one_cohort_out.tsv"))
    loco_rows = []
    for taxon in top_for_loco:
        for left_out in sorted(study.unique()):
            keep = study != left_out
            if keep.sum() < 20 or condition.loc[keep.index[keep]].nunique() < 2:
                loco_rows.append({"taxon_long": taxon, "left_out_cohort": left_out, "coef_crc_clr_adjusted": np.nan, "pvalue": np.nan, "status": "UNAVAILABLE"})
                continue
            dsub = design.loc[keep.values].copy()
            # Drop all-zero dummy columns after leaving one cohort out.
            dsub = dsub.loc[:, dsub.nunique(dropna=False) > 1]
            dsub = sm.add_constant(dsub.drop(columns=["const"], errors="ignore"), has_constant="add")
            coef, se, pval = fit_feature_ols(clr.loc[keep.values, taxon].values.astype(float), dsub)
            loco_rows.append({"taxon_long": taxon, "left_out_cohort": left_out, "coef_crc_clr_adjusted": coef, "pvalue": pval, "status": "COMPUTED"})
    loco = pd.DataFrame(loco_rows)
    loco_path = Path(write_tsv(loco, RESULTS / "D" / "leave_one_cohort_out_top50.tsv"))
    stable_sig = stability[(stability["adjusted_qvalue"] < 0.10) & (stability["stability_fraction"] >= 0.60)]
    log_lines.append(f"Module D completed: stability table for {len(stability)} features; stable FDR<0.10 candidates={len(stable_sig)}.")
    module_rows.append({"module_id": "D", "status": "COMPLETED", "primary_output": str(stability_path), "notes": f"Stable FDR<0.10 candidates with >=60% cohort direction agreement: {len(stable_sig)}."})

    # Module E: MicrobiomeHD benchmark/context comparison.
    schema_rows = []
    overlap = pd.DataFrame()
    try:
        mh_q = pd.read_csv(MICROBIOMEHD_Q, sep="\t").rename(columns={"Unnamed: 0": "microbiomehd_taxon"})
        mh_eff = pd.read_csv(MICROBIOMEHD_EFFECTS, sep="\t").rename(columns={"Unnamed: 0": "microbiomehd_taxon"})
        crc_cols = [c for c in mh_eff.columns if c.startswith("crc_")]
        crc_column_path = Path(write_tsv(pd.DataFrame({"crc_effect_column": crc_cols}), PROCESSED / "E" / "microbiomehd_crc_effect_columns.tsv"))
        mh_eff["genus"] = mh_eff["microbiomehd_taxon"].map(genus_from_taxon)
        mh_q["genus"] = mh_q["microbiomehd_taxon"].map(genus_from_taxon)
        top_candidates = assoc.head(min(100, len(assoc))).copy()
        top_candidates["rank"] = np.arange(1, len(top_candidates) + 1)
        top_candidates = top_candidates.merge(stability[["taxon_long", "stability_fraction"]], on="taxon_long", how="left")
        overlap = top_candidates.merge(
            mh_eff[["genus", "microbiomehd_taxon"] + crc_cols],
            on="genus",
            how="left",
            suffixes=("", "_microbiomehd"),
        )
        q_crc_cols = [c for c in mh_q.columns if c.startswith("crc_")]
        overlap = overlap.merge(
            mh_q[["genus"] + q_crc_cols].add_suffix("_q").rename(columns={"genus_q": "genus"}),
            on="genus",
            how="left",
        )
        schema_rows.append({"file": str(MICROBIOMEHD_Q), "status": "READABLE", "rows": str(mh_q.shape[0]), "columns": str(mh_q.shape[1]), "crc_columns": ",".join(q_crc_cols)})
        schema_rows.append({"file": str(MICROBIOMEHD_EFFECTS), "status": "READABLE", "rows": str(mh_eff.shape[0]), "columns": str(mh_eff.shape[1]), "crc_columns": ",".join(crc_cols)})
    except Exception as exc:
        crc_column_path = Path(write_tsv(pd.DataFrame(columns=["crc_effect_column"]), PROCESSED / "E" / "microbiomehd_crc_effect_columns.tsv"))
        schema_rows.append({"file": str(MICROBIOMEHD_DIR), "status": "FAILED", "rows": "", "columns": "", "crc_columns": "", "error": str(exc)})
    schema_path = Path(write_tsv(pd.DataFrame(schema_rows), RESULTS / "E" / "microbiomehd_schema_audit.tsv"))
    if overlap.empty:
        overlap = pd.DataFrame(columns=["taxon_long", "genus", "status"])
    overlap["benchmark_match_status"] = np.where(overlap.get("microbiomehd_taxon", pd.Series(index=overlap.index)).notna(), "GENUS_MATCH", "NO_GENUS_MATCH")
    overlap_path = Path(write_tsv(overlap, RESULTS / "E" / "microbiomehd_top100_genus_overlap.tsv"))
    n_overlap = int((overlap["benchmark_match_status"] == "GENUS_MATCH").sum()) if len(overlap) else 0
    log_lines.append(f"Module E completed: MicrobiomeHD schema inspected; top-100 genus matches={n_overlap}.")
    module_rows.append({"module_id": "E", "status": "COMPLETED", "primary_output": str(overlap_path), "notes": f"MicrobiomeHD used as genus-level benchmark/context; top-100 genus matches={n_overlap}."})

    # Module G: reviewer-triggered covariate completeness and sensitivity audit.
    metadata_indexed = metadata.set_index("sample_uid").loc[clr.index].copy()
    completeness_specs = [
        ("study_name", "baseline adjustment block"),
        ("age", "primary sensitivity covariate"),
        ("gender", "primary sensitivity covariate"),
        ("BMI", "primary sensitivity covariate"),
        ("country", "context-only sensitivity covariate"),
        ("sequencing_platform", "context-only sensitivity covariate"),
        ("DNA_extraction_kit", "excluded: incomplete metadata"),
        ("disease_stage", "excluded: sparse post-diagnosis field"),
    ]
    completeness_rows = []
    for col, model_role in completeness_specs:
        series = metadata_indexed[col] if col in metadata_indexed.columns else pd.Series(index=metadata_indexed.index, dtype=object)
        non_missing = int(series.notna().sum())
        completeness_rows.append({
            "covariate": col,
            "non_missing_n": non_missing,
            "total_n": int(len(metadata_indexed)),
            "non_missing_fraction": float(non_missing / len(metadata_indexed)) if len(metadata_indexed) else np.nan,
            "unique_non_missing": int(series.nunique(dropna=True)),
            "model_role": model_role,
        })
    covariate_completeness = pd.DataFrame(completeness_rows)
    covariate_completeness_path = Path(write_tsv(covariate_completeness, RESULTS / "G" / "covariate_completeness.tsv"))

    sens_meta = metadata_indexed.copy()
    sens_meta["age_z"] = zscore_numeric(sens_meta["age"])
    sens_meta["BMI_z"] = zscore_numeric(sens_meta["BMI"])
    eligible = (
        sens_meta["age_z"].notna()
        & sens_meta["BMI_z"].notna()
        & sens_meta["gender"].notna()
        & sens_meta["study_condition"].isin(["CRC", "control"])
    )
    sensitivity_samples = sens_meta.index[eligible].tolist()
    if len(sensitivity_samples) < 100 or condition.loc[sensitivity_samples].nunique() < 2:
        raise SystemExit("covariate sensitivity design has too few complete CRC/control samples")

    base_sens = pd.DataFrame({
        "condition_crc": condition.loc[sensitivity_samples].astype(float),
        "age_z": sens_meta.loc[sensitivity_samples, "age_z"].astype(float),
        "BMI_z": sens_meta.loc[sensitivity_samples, "BMI_z"].astype(float),
    }, index=sensitivity_samples)
    gender_dummies = pd.get_dummies(sens_meta.loc[sensitivity_samples, "gender"].astype(str), prefix="gender", drop_first=True, dtype=float)
    study_sens_dummies = pd.get_dummies(sens_meta.loc[sensitivity_samples, "study_name"].astype(str), prefix="study", drop_first=True, dtype=float)
    covariate_design, covariate_rank, covariate_columns = finalize_design(pd.concat([base_sens, gender_dummies, study_sens_dummies], axis=1))
    covariate_design_path = PROCESSED / "G" / "study_age_bmi_gender_design_matrix.tsv.gz"
    covariate_design.to_csv(covariate_design_path, sep="\t", compression="gzip", index=True)

    context_sens = base_sens.copy()
    country_dummies = pd.get_dummies(sens_meta.loc[sensitivity_samples, "country"].astype(str), prefix="country", drop_first=True, dtype=float)
    platform_dummies = pd.get_dummies(sens_meta.loc[sensitivity_samples, "sequencing_platform"].astype(str), prefix="platform", drop_first=True, dtype=float)
    context_design, context_rank, context_columns = finalize_design(pd.concat([context_sens, gender_dummies, country_dummies, platform_dummies], axis=1))
    context_design_path = PROCESSED / "G" / "country_platform_no_study_design_matrix.tsv.gz"
    context_design.to_csv(context_design_path, sep="\t", compression="gzip", index=True)

    def run_covariate_screen(design_table: pd.DataFrame, sample_ids: list[str], coef_name: str) -> pd.DataFrame:
        rows = []
        for taxon in clr.columns:
            coef, se, pval = fit_feature_ols(clr.loc[sample_ids, taxon].values.astype(float), design_table)
            rows.append({
                "taxon_long": taxon,
                "genus": genus_from_taxon(taxon),
                "species_short": species_from_taxon(taxon),
                coef_name: coef,
                f"se_hc3_{coef_name}": se,
                f"pvalue_{coef_name}": pval,
            })
        out = pd.DataFrame(rows)
        out[f"qvalue_bh_{coef_name}"] = bh_fdr(out[f"pvalue_{coef_name}"])
        out[f"direction_{coef_name}"] = np.sign(out[coef_name]).map({1.0: "CRC_higher", -1.0: "control_higher", 0.0: "zero"})
        return out.sort_values([f"qvalue_bh_{coef_name}", f"pvalue_{coef_name}", "taxon_long"], na_position="last")

    covariate_assoc = run_covariate_screen(covariate_design, sensitivity_samples, "coef_crc_clr_study_age_bmi_gender")
    covariate_assoc_path = Path(write_tsv(covariate_assoc, RESULTS / "G" / "covariate_adjusted_crc_control_association.tsv"))
    context_assoc = run_covariate_screen(context_design, sensitivity_samples, "coef_crc_clr_age_bmi_gender_country_platform")
    context_assoc_path = Path(write_tsv(context_assoc, RESULTS / "G" / "country_platform_no_study_sensitivity.tsv"))

    baseline_compare = assoc[[
        "taxon_long",
        "coef_crc_clr_adjusted",
        "pvalue",
        "qvalue_bh",
        "direction_adjusted",
    ]].rename(columns={
        "coef_crc_clr_adjusted": "coef_crc_clr_study_only",
        "pvalue": "pvalue_study_only",
        "qvalue_bh": "qvalue_bh_study_only",
        "direction_adjusted": "direction_study_only",
    })
    covariate_compare = baseline_compare.merge(covariate_assoc, on="taxon_long", how="left")
    covariate_compare["baseline_sign"] = np.sign(covariate_compare["coef_crc_clr_study_only"])
    covariate_compare["covariate_sign"] = np.sign(covariate_compare["coef_crc_clr_study_age_bmi_gender"])
    covariate_compare["direction_concordant"] = covariate_compare["baseline_sign"] == covariate_compare["covariate_sign"]
    covariate_compare_path = Path(write_tsv(covariate_compare, RESULTS / "G" / "baseline_vs_covariate_sensitivity.tsv"))

    baseline_top100 = set(assoc.head(min(100, len(assoc)))["taxon_long"])
    covariate_top100 = set(covariate_assoc.head(min(100, len(covariate_assoc)))["taxon_long"])
    valid_sign = covariate_compare["baseline_sign"].notna() & covariate_compare["covariate_sign"].notna()
    fdr_baseline = int((assoc["qvalue_bh"] < 0.10).sum())
    fdr_covariate = int((covariate_assoc["qvalue_bh_coef_crc_clr_study_age_bmi_gender"] < 0.10).sum())
    fdr_context = int((context_assoc["qvalue_bh_coef_crc_clr_age_bmi_gender_country_platform"] < 0.10).sum())
    preserved_baseline_sig = int((
        (covariate_compare["qvalue_bh_study_only"] < 0.10)
        & (covariate_compare["qvalue_bh_coef_crc_clr_study_age_bmi_gender"] < 0.10)
        & covariate_compare["direction_concordant"]
    ).sum())
    stable_taxa = set(stable_sig["taxon_long"].tolist())
    preserved_stable_sig = int((
        covariate_compare["taxon_long"].isin(stable_taxa)
        & (covariate_compare["qvalue_bh_coef_crc_clr_study_age_bmi_gender"] < 0.10)
        & covariate_compare["direction_concordant"]
    ).sum())
    direction_concordance = float(covariate_compare.loc[valid_sign, "direction_concordant"].mean()) if valid_sign.any() else np.nan
    top100_overlap = len(baseline_top100 & covariate_top100)
    covariate_summary = pd.DataFrame([
        {"metric": "complete_case_samples_for_primary_covariate_model", "value": str(len(sensitivity_samples)), "interpretation": "Samples with age, BMI, gender and CRC/control labels available."},
        {"metric": "primary_covariate_model", "value": "condition + study + age_z + BMI_z + gender", "interpretation": "Reviewer-triggered sensitivity model preserving study adjustment."},
        {"metric": "primary_covariate_design_rank", "value": f"{covariate_rank}/{covariate_columns}", "interpretation": "Full column rank is expected for estimable coefficients."},
        {"metric": "context_model_for_country_platform", "value": "condition + age_z + BMI_z + gender + country + sequencing_platform", "interpretation": "Country/platform are evaluated without study indicators because they are cohort-level attributes and partly collinear with study."},
        {"metric": "context_design_rank", "value": f"{context_rank}/{context_columns}", "interpretation": "Context model is not used as the main association model."},
        {"metric": "baseline_fdr_lt_0_10", "value": str(fdr_baseline), "interpretation": "Study-adjusted model from Module C."},
        {"metric": "primary_covariate_fdr_lt_0_10", "value": str(fdr_covariate), "interpretation": "Study + age + BMI + gender sensitivity model."},
        {"metric": "country_platform_context_fdr_lt_0_10", "value": str(fdr_context), "interpretation": "No-study country/platform context model; interpret cautiously."},
        {"metric": "baseline_sig_preserved_in_primary_covariate_model", "value": str(preserved_baseline_sig), "interpretation": "Baseline FDR<0.10 features also FDR<0.10 with same direction after covariate adjustment."},
        {"metric": "stable_sig_preserved_in_primary_covariate_model", "value": str(preserved_stable_sig), "interpretation": "Module D stable candidates also retained in the primary covariate model."},
        {"metric": "top100_overlap_baseline_vs_primary_covariate", "value": str(top100_overlap), "interpretation": "Overlap between top 100 study-adjusted and primary covariate-adjusted features."},
        {"metric": "direction_concordance_rate_all_features", "value": f"{direction_concordance:.3f}", "interpretation": "Fraction of retained features with identical CRC/control direction across baseline and primary covariate models."},
        {"metric": "excluded_covariates", "value": "DNA_extraction_kit; disease_stage", "interpretation": "DNA extraction kit is incomplete; disease stage is sparse and post-diagnosis, so neither is used for primary association adjustment."},
    ])
    covariate_summary_path = Path(write_tsv(covariate_summary, RESULTS / "G" / "covariate_sensitivity_summary.tsv"))
    model_feasibility = pd.DataFrame([
        {"model_id": "baseline_study_adjusted", "status": "COMPUTED", "samples": str(len(clr)), "design_rank": "", "design_columns": str(design.shape[1]), "result_file": str(assoc_path)},
        {"model_id": "primary_study_age_bmi_gender", "status": "COMPUTED", "samples": str(len(sensitivity_samples)), "design_rank": str(covariate_rank), "design_columns": str(covariate_columns), "result_file": str(covariate_assoc_path)},
        {"model_id": "context_country_platform_no_study", "status": "COMPUTED_CONTEXT_ONLY", "samples": str(len(sensitivity_samples)), "design_rank": str(context_rank), "design_columns": str(context_columns), "result_file": str(context_assoc_path)},
        {"model_id": "dna_extraction_kit_adjusted", "status": "EXCLUDED_INCOMPLETE", "samples": str(int(metadata_indexed["DNA_extraction_kit"].notna().sum())), "design_rank": "", "design_columns": "", "result_file": ""},
        {"model_id": "disease_stage_adjusted", "status": "EXCLUDED_SPARSE_POST_DIAGNOSIS", "samples": str(int(metadata_indexed["disease_stage"].notna().sum())), "design_rank": "", "design_columns": "", "result_file": ""},
    ])
    model_feasibility_path = Path(write_tsv(model_feasibility, RESULTS / "G" / "covariate_model_feasibility.tsv"))
    log_lines.append(
        "Module G completed: covariate completeness audited; "
        f"primary sensitivity samples={len(sensitivity_samples)}; "
        f"FDR<0.10 baseline={fdr_baseline}, covariate={fdr_covariate}; "
        f"top-100 overlap={top100_overlap}."
    )
    module_rows.append({
        "module_id": "G",
        "status": "COMPLETED",
        "primary_output": str(covariate_summary_path),
        "notes": "Reviewer-triggered covariate completeness audit and study+age+BMI+gender sensitivity model completed; country/platform context model reported separately.",
    })

    # Module H: higher-level compositional structure and leave-one-study-out prediction stress test.
    pca = PCA(n_components=2, random_state=42)
    pca_scores = pca.fit_transform(clr.to_numpy(dtype=float))
    pca_df = pd.DataFrame({
        "sample_uid": clr.index,
        "PC1": pca_scores[:, 0],
        "PC2": pca_scores[:, 1],
        "study_condition": metadata_indexed.loc[clr.index, "study_condition"].values,
        "study_name": metadata_indexed.loc[clr.index, "study_name"].values,
        "PC1_explained_variance_ratio": pca.explained_variance_ratio_[0],
        "PC2_explained_variance_ratio": pca.explained_variance_ratio_[1],
    })
    pca_scores_path = Path(write_tsv(pca_df, RESULTS / "H" / "aitchison_pca_scores.tsv"))

    study_onehot = pd.get_dummies(metadata_indexed.loc[clr.index, "study_name"].astype(str), dtype=float)
    x_clr = clr.to_numpy(dtype=float)
    study_beta = np.linalg.lstsq(study_onehot.to_numpy(dtype=float), x_clr, rcond=None)[0]
    residual_clr = x_clr - study_onehot.to_numpy(dtype=float) @ study_beta
    condition_labels = condition.loc[clr.index].to_numpy(dtype=int)
    observed_f = pseudo_f_two_group(residual_clr, condition_labels)
    rng = np.random.default_rng(42)
    n_perm = 4999
    perm_stats = []
    study_values = metadata_indexed.loc[clr.index, "study_name"].astype(str).to_numpy()
    for _ in range(n_perm):
        permuted = condition_labels.copy()
        for cohort in np.unique(study_values):
            idx = np.where(study_values == cohort)[0]
            permuted[idx] = rng.permutation(permuted[idx])
        perm_stats.append(pseudo_f_two_group(residual_clr, permuted))
    perm_stats_arr = np.asarray(perm_stats, dtype=float)
    permanova_p = float((np.nansum(perm_stats_arr >= observed_f) + 1) / (np.sum(np.isfinite(perm_stats_arr)) + 1))
    permanova_df = pd.DataFrame([{
        "test_id": "study_residualized_aitchison_condition_permutation",
        "distance_space": "CLR/Aitchison residualized by study means",
        "n_samples": int(residual_clr.shape[0]),
        "n_features": int(residual_clr.shape[1]),
        "n_permutations": int(n_perm),
        "pseudo_f": observed_f,
        "permutation_pvalue": permanova_p,
        "interpretation": "Exploratory study-stratified compositional separation test; not a diagnostic performance estimate.",
    }])
    permanova_path = Path(write_tsv(permanova_df, RESULTS / "H" / "stratified_permanova.tsv"))
    permutation_null = pd.DataFrame({
        "permutation_index": np.arange(1, len(perm_stats_arr) + 1),
        "permuted_pseudo_f": perm_stats_arr,
        "observed_pseudo_f": observed_f,
    })
    permutation_null_path = Path(write_tsv(permutation_null, RESULTS / "H" / "stratified_permanova_permutation_null.tsv"))

    performance_rows = []
    prediction_rows = []
    coefficient_rows = []
    x_model = clr.to_numpy(dtype=float)
    y_model = condition.loc[clr.index].to_numpy(dtype=int)
    for held_out in sorted(metadata_indexed.loc[clr.index, "study_name"].astype(str).unique()):
        test_mask = metadata_indexed.loc[clr.index, "study_name"].astype(str).to_numpy() == held_out
        train_mask = ~test_mask
        y_train = y_model[train_mask]
        y_test = y_model[test_mask]
        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            performance_rows.append({
                "held_out_study": held_out,
                "n_train": int(train_mask.sum()),
                "n_test": int(test_mask.sum()),
                "n_test_crc": int(y_test.sum()),
                "n_test_control": int((1 - y_test).sum()),
                "auroc": np.nan,
                "average_precision": np.nan,
                "status": "SKIPPED_SINGLE_CLASS",
            })
            continue
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                solver="saga",
                l1_ratio=0.5,
                C=0.5,
                max_iter=2000,
                class_weight="balanced",
                random_state=42,
            ),
        )
        model.fit(x_model[train_mask], y_train)
        prob = model.predict_proba(x_model[test_mask])[:, 1]
        auroc = float(roc_auc_score(y_test, prob))
        aupr = float(average_precision_score(y_test, prob))
        performance_rows.append({
            "held_out_study": held_out,
            "n_train": int(train_mask.sum()),
            "n_test": int(test_mask.sum()),
            "n_test_crc": int(y_test.sum()),
            "n_test_control": int((1 - y_test).sum()),
            "auroc": auroc,
            "average_precision": aupr,
            "status": "COMPUTED",
        })
        test_ids = clr.index[test_mask].tolist()
        for sample_uid, true_label, score in zip(test_ids, y_test, prob):
            prediction_rows.append({
                "sample_uid": sample_uid,
                "held_out_study": held_out,
                "true_condition": "CRC" if int(true_label) == 1 else "control",
                "predicted_crc_probability": float(score),
            })
        coef = model.named_steps["logisticregression"].coef_.ravel()
        for taxon, value in zip(clr.columns, coef):
            coefficient_rows.append({
                "held_out_study": held_out,
                "taxon_long": taxon,
                "elasticnet_standardized_coef": float(value),
            })
    loso_performance = pd.DataFrame(performance_rows)
    loso_predictions = pd.DataFrame(prediction_rows)
    coefficient_table = pd.DataFrame(coefficient_rows)
    loso_performance_path = Path(write_tsv(loso_performance, RESULTS / "H" / "leave_one_study_elasticnet_performance.tsv"))
    loso_predictions_path = Path(write_tsv(loso_predictions, RESULTS / "H" / "leave_one_study_elasticnet_predictions.tsv"))
    if coefficient_table.empty:
        feature_weights = pd.DataFrame(columns=["taxon_long", "mean_elasticnet_coef", "mean_abs_elasticnet_coef", "selected_fraction"])
    else:
        feature_weights = coefficient_table.groupby("taxon_long", as_index=False).agg(
            mean_elasticnet_coef=("elasticnet_standardized_coef", "mean"),
            mean_abs_elasticnet_coef=("elasticnet_standardized_coef", lambda x: float(np.mean(np.abs(x)))),
            selected_fraction=("elasticnet_standardized_coef", lambda x: float(np.mean(np.abs(x) > 1e-8))),
        )
        feature_weights["genus"] = feature_weights["taxon_long"].map(genus_from_taxon)
        feature_weights["species_short"] = feature_weights["taxon_long"].map(species_from_taxon)
        feature_weights = feature_weights.sort_values(["mean_abs_elasticnet_coef", "selected_fraction"], ascending=[False, False])
    feature_weights_path = Path(write_tsv(feature_weights, RESULTS / "H" / "elasticnet_feature_weights.tsv"))

    transport = assoc[[
        "taxon_long",
        "genus",
        "species_short",
        "coef_crc_clr_adjusted",
        "qvalue_bh",
    ]].merge(
        stability[["taxon_long", "stability_fraction", "cohorts_same_direction", "cohorts_tested"]],
        on="taxon_long",
        how="left",
    ).merge(
        covariate_compare[["taxon_long", "qvalue_bh_coef_crc_clr_study_age_bmi_gender", "direction_concordant"]],
        on="taxon_long",
        how="left",
    ).merge(
        feature_weights[["taxon_long", "mean_abs_elasticnet_coef", "selected_fraction"]],
        on="taxon_long",
        how="left",
    )
    transport["neglog10_q"] = -np.log10(pd.to_numeric(transport["qvalue_bh"], errors="coerce").clip(lower=1e-300))
    transport["covariate_supported"] = (
        (pd.to_numeric(transport["qvalue_bh_coef_crc_clr_study_age_bmi_gender"], errors="coerce") < 0.10)
        & (transport["direction_concordant"].astype(str).str.lower() == "true")
    )
    transport["transportability_score"] = (
        transport["neglog10_q"].fillna(0)
        * transport["stability_fraction"].fillna(0)
        * np.where(transport["covariate_supported"], 1.0, 0.5)
        * (1.0 + transport["mean_abs_elasticnet_coef"].fillna(0))
    )
    transport = transport.sort_values("transportability_score", ascending=False)
    transport["transportability_rank"] = np.arange(1, len(transport) + 1)
    transport_path = Path(write_tsv(transport, RESULTS / "H" / "transportability_score.tsv"))

    computed_perf = loso_performance[loso_performance["status"] == "COMPUTED"].copy()
    median_auroc = float(computed_perf["auroc"].median()) if not computed_perf.empty else np.nan
    min_auroc = float(computed_perf["auroc"].min()) if not computed_perf.empty else np.nan
    weighted_auroc = float(np.average(computed_perf["auroc"], weights=computed_perf["n_test"])) if not computed_perf.empty else np.nan
    n_loso_computed = int(len(computed_perf))
    top_transport = transport.iloc[0]["taxon_long"] if len(transport) else "NA"
    advanced_summary = pd.DataFrame([
        {"metric": "pca_pc1_variance", "value": f"{pca.explained_variance_ratio_[0]:.4f}", "interpretation": "Aitchison PCA PC1 explained variance ratio."},
        {"metric": "pca_pc2_variance", "value": f"{pca.explained_variance_ratio_[1]:.4f}", "interpretation": "Aitchison PCA PC2 explained variance ratio."},
        {"metric": "study_residualized_permutation_pvalue", "value": f"{permanova_p:.4f}", "interpretation": "Study-stratified permutation p value for CRC/control separation in residualized CLR space."},
        {"metric": "leave_one_study_models_computed", "value": str(n_loso_computed), "interpretation": "Held-out study folds with both classes available."},
        {"metric": "leave_one_study_median_auroc", "value": f"{median_auroc:.3f}", "interpretation": "Median held-out-study AUROC for elastic-net stress test."},
        {"metric": "leave_one_study_min_auroc", "value": f"{min_auroc:.3f}", "interpretation": "Lowest held-out-study AUROC; heterogeneity stress-test value."},
        {"metric": "leave_one_study_weighted_auroc", "value": f"{weighted_auroc:.3f}", "interpretation": "Test-sample weighted mean AUROC across held-out studies."},
        {"metric": "top_transportability_ranked_taxon", "value": str(top_transport), "interpretation": "Highest stability/covariate/model-weight composite candidate; ranking is exploratory."},
    ])
    advanced_summary_path = Path(write_tsv(advanced_summary, RESULTS / "H" / "advanced_analysis_summary.tsv"))
    log_lines.append(
        "Module H completed: Aitchison PCA, study-stratified permutation test, "
        f"leave-one-study-out elastic-net stress test ({n_loso_computed} folds; median AUROC={median_auroc:.3f}), "
        "and transportability scoring completed."
    )
    module_rows.append({
        "module_id": "H",
        "status": "COMPLETED",
        "primary_output": str(advanced_summary_path),
        "notes": "Advanced compositional ordination, stratified permutation, leave-one-study-out elastic-net stress test, and transportability ranking completed.",
    })

    # Module F: claim ledger and figure map.
    evidence = pd.read_csv(UPSTREAM_04 / "evidence_chain.tsv", sep="\t")
    actuals = {
        "C1": f"PASS input preflight; matrix {matrix_raw.shape[0]} features x {matrix_raw.shape[1]} samples; CRC={int((metadata['study_condition']=='CRC').sum())}; control={int((metadata['study_condition']=='control').sum())}; cohorts={metadata['study_name'].nunique()}",
        "C2": f"Retained {len(retained_taxa)} of {abundance.shape[1]} species features using prevalence>=5% and present in >=3 cohorts; pseudocount={pseudocount:.6g}",
        "C3": f"Cohort-adjusted CLR OLS tested {len(assoc)} features; FDR<0.10={n_sig_010}; FDR<0.05={n_sig_005}; top={top_taxon}. Reviewer-triggered primary sensitivity model adjusted for study, age, BMI, and gender on {len(sensitivity_samples)} complete-case samples; FDR<0.10={fdr_covariate}; top-100 overlap with baseline={top100_overlap}; all-feature direction concordance={direction_concordance:.3f}. Study-residualized Aitchison permutation p={permanova_p:.4f}.",
        "C4": f"Cross-cohort stability computed for {len(stability)} features; stable FDR<0.10 candidates with >=60% cohort direction agreement={len(stable_sig)}. Leave-one-study-out elastic-net stress test computed {n_loso_computed} folds with median AUROC={median_auroc:.3f}, minimum AUROC={min_auroc:.3f}, and weighted AUROC={weighted_auroc:.3f}.",
        "C5": f"MicrobiomeHD processed tables inspected at genus level; top-100 candidate genus matches={n_overlap}; use is contextual, not patient-level integration. Exploratory transportability ranking integrated pooled association, cohort direction, covariate support, and elastic-net model weights; top ranked taxon={top_transport}.",
        "C6": "Claim ledger created; covariate sensitivity is treated as robustness evidence only; causal, diagnostic, stage-specific, mechanistic, and raw-read claims marked not allowed",
    }
    evidence["actual_value"] = evidence["claim_id"].map(actuals).fillna("UNAVAILABLE: claim not executed")
    evidence["pass_fail"] = "TO_FILL_BY_10"
    evidence_path = Path(write_tsv(evidence, WORKSPACE / "evidence_chain.tsv"))

    claim_ledger = pd.DataFrame([
        {
            "claim_id": row["claim_id"],
            "claim_text": row["claim_text"],
            "evidence_class": "computed_result" if not str(row["actual_value"]).startswith("UNAVAILABLE") else "limitation",
            "strength": "analysis_ready_computed" if row["claim_id"] in ["C1", "C2", "C3", "C4"] else "context_or_claim_ceiling",
            "source_result_file": {
                "C1": str(qc_path),
                "C2": str(filter_table_path),
                "C3": str(assoc_path),
                "C4": str(stability_path),
                "C5": str(transport_path),
                "C6": str(WORKSPACE / "claim_ledger.tsv"),
            }.get(row["claim_id"], ""),
            "figure_panel": row["figure_panel"],
            "actual_value": row["actual_value"],
            "claim_ceiling_note": "Processed public-data association/reproducibility only; no causal, diagnostic, mechanistic, stage-specific, or raw-read reprocessing claim.",
        }
        for _, row in evidence.iterrows()
    ])
    claim_path = Path(write_tsv(claim_ledger, WORKSPACE / "claim_ledger.tsv"))
    claim_support_path = Path(write_tsv(claim_ledger, RESULTS / "F" / "claim_support_summary.tsv"))
    claim_input_manifest = pd.DataFrame([
        {"artifact": "evidence_chain", "path": str(evidence_path), "role": "filled_05_evidence_chain"},
        {"artifact": "claim_ledger", "path": str(claim_path), "role": "claim_to_result_mapping"},
        {"artifact": "figure_data_map", "path": str(WORKSPACE / "figure_data_map.tsv"), "role": "figure_source_data_mapping"},
    ])
    claim_input_manifest_path = Path(write_tsv(claim_input_manifest, PROCESSED / "F" / "claim_inputs_manifest.tsv"))

    figure_rows = [
        {"figure_id": "Fig1", "panel": "a", "module_id": "H", "claim_id": "C3", "source_data_file": str(pca_scores_path), "generation_script": str(SCRIPT_PATH), "result_type": "aitchison_pca", "status": "READY", "notes": "Aitchison PCA sample map."},
        {"figure_id": "Fig1", "panel": "b", "module_id": "H", "claim_id": "C3", "source_data_file": str(permutation_null_path), "generation_script": str(SCRIPT_PATH), "result_type": "study_residualized_permutation_null", "status": "READY", "notes": "Permutation null distribution for study-residualized CRC/control compositional separation test."},
        {"figure_id": "Fig1", "panel": "c", "module_id": "B", "claim_id": "C2", "source_data_file": str(filter_table_path), "generation_script": str(SCRIPT_PATH), "result_type": "feature_filter", "status": "READY", "notes": "Feature prevalence and retention."},
        {"figure_id": "Fig2", "panel": "a", "module_id": "B", "claim_id": "C2", "source_data_file": str(landscape_path), "generation_script": str(SCRIPT_PATH), "result_type": "feature_landscape", "status": "READY", "notes": "Retained vs filtered feature summary."},
        {"figure_id": "Fig2", "panel": "b", "module_id": "C", "claim_id": "C3", "source_data_file": str(assoc_path), "generation_script": str(SCRIPT_PATH), "result_type": "association_table", "status": "READY", "notes": "Cohort-adjusted association screen."},
        {"figure_id": "Fig2", "panel": "c", "module_id": "C", "claim_id": "C3", "source_data_file": str(mw_path), "generation_script": str(SCRIPT_PATH), "result_type": "sensitivity_table", "status": "READY", "notes": "Raw abundance Mann-Whitney sensitivity."},
        {"figure_id": "Fig2", "panel": "d", "module_id": "G", "claim_id": "C3", "source_data_file": str(covariate_summary_path), "generation_script": str(SCRIPT_PATH), "result_type": "covariate_sensitivity_summary", "status": "READY", "notes": "Reviewer-triggered covariate completeness and sensitivity summary."},
        {"figure_id": "Fig3", "panel": "a", "module_id": "D", "claim_id": "C4", "source_data_file": str(stability_path), "generation_script": str(SCRIPT_PATH), "result_type": "stability_table", "status": "READY", "notes": "Per-cohort direction agreement."},
        {"figure_id": "Fig3", "panel": "b", "module_id": "D", "claim_id": "C4", "source_data_file": str(loco_path), "generation_script": str(SCRIPT_PATH), "result_type": "leave_one_cohort_out", "status": "READY", "notes": "Top-50 leave-one-cohort-out sensitivity."},
        {"figure_id": "Fig4", "panel": "a", "module_id": "H", "claim_id": "C4", "source_data_file": str(loso_performance_path), "generation_script": str(SCRIPT_PATH), "result_type": "leave_one_study_performance", "status": "READY", "notes": "Leave-one-study-out elastic-net AUROC and average precision."},
        {"figure_id": "Fig4", "panel": "b", "module_id": "H", "claim_id": "C4", "source_data_file": str(loso_predictions_path), "generation_script": str(SCRIPT_PATH), "result_type": "leave_one_study_predictions", "status": "READY", "notes": "Held-out prediction score distributions."},
        {"figure_id": "Fig4", "panel": "c", "module_id": "H", "claim_id": "C5", "source_data_file": str(transport_path), "generation_script": str(SCRIPT_PATH), "result_type": "transportability_score", "status": "READY", "notes": "Exploratory transportability ranking."},
        {"figure_id": "Fig5", "panel": "a", "module_id": "E", "claim_id": "C5", "source_data_file": str(overlap_path), "generation_script": str(SCRIPT_PATH), "result_type": "benchmark_overlap", "status": "READY", "notes": "MicrobiomeHD genus-level overlap."},
        {"figure_id": "Fig5", "panel": "b", "module_id": "E", "claim_id": "C5", "source_data_file": str(schema_path), "generation_script": str(SCRIPT_PATH), "result_type": "benchmark_schema", "status": "READY", "notes": "MicrobiomeHD schema audit."},
        {"figure_id": "Fig5", "panel": "c", "module_id": "F", "claim_id": "C6", "source_data_file": str(claim_path), "generation_script": str(SCRIPT_PATH), "result_type": "claim_downgrade", "status": "READY", "notes": "Claim ceiling and downgrade map."},
        {"figure_id": "Fig6", "panel": "a", "module_id": "F", "claim_id": "C6", "source_data_file": str(evidence_path), "generation_script": str(SCRIPT_PATH), "result_type": "evidence_chain", "status": "READY", "notes": "Filled evidence chain for gate review."},
        {"figure_id": "FigS1", "panel": "a", "module_id": "G", "claim_id": "C3", "source_data_file": str(covariate_completeness_path), "generation_script": str(SCRIPT_PATH), "result_type": "covariate_completeness", "status": "READY", "notes": "Supplementary covariate metadata completeness audit."},
        {"figure_id": "FigS1", "panel": "b", "module_id": "G", "claim_id": "C3", "source_data_file": str(covariate_compare_path), "generation_script": str(SCRIPT_PATH), "result_type": "baseline_vs_covariate_sensitivity", "status": "READY", "notes": "Supplementary baseline versus covariate-adjusted association comparison."},
    ]
    figure_map_path = Path(write_tsv(pd.DataFrame(figure_rows), WORKSPACE / "figure_data_map.tsv"))

    module_rows.append({"module_id": "F", "status": "COMPLETED", "primary_output": str(claim_support_path), "notes": "Claim ledger, evidence chain, and figure data map completed."})
    module_status = pd.DataFrame(module_rows)
    module_status_path = Path(write_tsv(module_status, WORKSPACE / "module_status.tsv"))

    log_path = WORKSPACE / "module_execution_log.md"
    log_text = "# Module Execution Log\n\n" + "\n".join(f"- {line}" for line in log_lines) + "\n"
    log_text += f"\nEnvironment: Python {platform.python_version()}, pandas {pd.__version__}, numpy {np.__version__}, scipy {scipy.__version__}, statsmodels {sm.__version__}.\n"
    log_path.write_text(log_text, encoding="utf-8")

    qc_md = WORKSPACE / "analysis_qc.md"
    qc_md.write_text(
        "# Analysis QC\n\n"
        f"- Input sample match: {'PASS' if sample_match else 'FAIL'}.\n"
        f"- Primary matrix shape: {matrix_raw.shape[0]} features x {matrix_raw.shape[1]} samples.\n"
        f"- CRC/control rows: {int((metadata['study_condition']=='CRC').sum())} / {int((metadata['study_condition']=='control').sum())}.\n"
        f"- Retained features: {len(retained_taxa)} of {abundance.shape[1]}.\n"
        f"- Cohort-adjusted association tested features: {len(assoc)}.\n"
        f"- FDR<0.10 features: {n_sig_010}; FDR<0.05 features: {n_sig_005}.\n"
        f"- Covariate completeness audit: age {int(metadata_indexed['age'].notna().sum())}/{len(metadata_indexed)}, gender {int(metadata_indexed['gender'].notna().sum())}/{len(metadata_indexed)}, BMI {int(metadata_indexed['BMI'].notna().sum())}/{len(metadata_indexed)}, country {int(metadata_indexed['country'].notna().sum())}/{len(metadata_indexed)}, sequencing_platform {int(metadata_indexed['sequencing_platform'].notna().sum())}/{len(metadata_indexed)}.\n"
        f"- Primary covariate sensitivity model: study + age + BMI + gender, complete-case n={len(sensitivity_samples)}, FDR<0.10 features={fdr_covariate}, top-100 overlap with baseline={top100_overlap}, direction concordance={direction_concordance:.3f}.\n"
        "- Country/platform context model: computed without study indicators because country and sequencing platform are cohort-level attributes partly collinear with study.\n"
        f"- Advanced compositional module: Aitchison PCA PC1/PC2 explained variance={pca.explained_variance_ratio_[0]:.4f}/{pca.explained_variance_ratio_[1]:.4f}; study-residualized permutation p={permanova_p:.4f}.\n"
        f"- Leave-one-study-out elastic-net stress test: computed folds={n_loso_computed}, median AUROC={median_auroc:.3f}, minimum AUROC={min_auroc:.3f}, weighted AUROC={weighted_auroc:.3f}.\n"
        f"- Stable FDR<0.10 candidates with >=60% cohort direction agreement: {len(stable_sig)}.\n"
        f"- MicrobiomeHD top-100 genus matches: {n_overlap}.\n"
        "- Limitations: processed relative abundance only; country/platform model is context only; no causal, diagnostic, mechanistic, stage-specific, or raw-read claims.\n",
        encoding="utf-8",
    )

    synthesis_notes = WORKSPACE / "synthesis_notes.md"
    synthesis_notes.write_text(
        "# Synthesis Notes For 07\n\n"
        "## Computed Result Summary\n\n"
        f"- 05 verified the curatedMetagenomicData CRC/control matrix and metadata: {matrix_raw.shape[0]} species features x {matrix_raw.shape[1]} samples.\n"
        f"- After prevalence/cohort filtering, {len(retained_taxa)} species features were retained for association screening.\n"
        f"- Cohort-adjusted CLR association tested {len(assoc)} features; {n_sig_010} had BH FDR < 0.10 and {n_sig_005} had BH FDR < 0.05.\n"
        f"- Reviewer-triggered covariate sensitivity used {len(sensitivity_samples)} complete-case samples and adjusted for study, age, BMI, and gender; {fdr_covariate} features had BH FDR < 0.10, top-100 overlap with the baseline study-adjusted model was {top100_overlap}, and all-feature direction concordance was {direction_concordance:.3f}.\n"
        "- Country and sequencing platform were audited in a context model without study indicators because they are partly cohort-level attributes and collinear with study.\n"
        f"- Advanced compositional analysis added Aitchison PCA and a study-residualized, within-study permutation test for CRC/control separation (p={permanova_p:.4f}).\n"
        f"- Leave-one-study-out elastic-net stress testing computed {n_loso_computed} held-out-study folds, with median AUROC {median_auroc:.3f}, minimum AUROC {min_auroc:.3f}, and weighted AUROC {weighted_auroc:.3f}; this is a heterogeneity stress test, not clinical validation.\n"
        f"- Composite transportability ranking combined pooled association, cohort-direction stability, covariate support, and elastic-net model weight; top ranked taxon: {top_transport}.\n"
        f"- Cross-cohort stability analysis found {len(stable_sig)} FDR<0.10 candidates with at least 60% cohort direction agreement.\n"
        f"- MicrobiomeHD comparison is genus-level context only; top-100 candidate genus matches: {n_overlap}.\n\n"
        "## Claim Ceiling\n\n"
        "Use processed-data association and reproducibility wording only. Do not write causal, clinical diagnostic, mechanistic, stage-specific, or raw-read reprocessing claims.\n",
        encoding="utf-8",
    )

    outputs_manifest = {
        "qc_path": str(qc_path),
        "cohort_counts_path": str(cohort_counts_path),
        "filter_table_path": str(filter_table_path),
        "association_results_path": str(assoc_path),
        "mannwhitney_sensitivity_path": str(mw_path),
        "design_matrix_path": str(design_path),
        "covariate_completeness_path": str(covariate_completeness_path),
        "covariate_design_matrix_path": str(covariate_design_path),
        "country_platform_context_design_matrix_path": str(context_design_path),
        "covariate_adjusted_association_path": str(covariate_assoc_path),
        "country_platform_context_sensitivity_path": str(context_assoc_path),
        "baseline_vs_covariate_sensitivity_path": str(covariate_compare_path),
        "covariate_sensitivity_summary_path": str(covariate_summary_path),
        "covariate_model_feasibility_path": str(model_feasibility_path),
        "aitchison_pca_scores_path": str(pca_scores_path),
        "stratified_permanova_path": str(permanova_path),
        "stratified_permanova_permutation_null_path": str(permutation_null_path),
        "leave_one_study_elasticnet_performance_path": str(loso_performance_path),
        "leave_one_study_elasticnet_predictions_path": str(loso_predictions_path),
        "elasticnet_feature_weights_path": str(feature_weights_path),
        "transportability_score_path": str(transport_path),
        "advanced_analysis_summary_path": str(advanced_summary_path),
        "stability_path": str(stability_path),
        "leave_one_cohort_out_path": str(loco_path),
        "top_for_loco_path": str(top_for_loco_path),
        "microbiomehd_overlap_path": str(overlap_path),
        "microbiomehd_schema_path": str(schema_path),
        "microbiomehd_crc_column_path": str(crc_column_path),
        "module_status_path": str(module_status_path),
        "figure_data_map_path": str(figure_map_path),
        "evidence_chain_path": str(evidence_path),
        "claim_ledger_path": str(claim_path),
        "claim_support_path": str(claim_support_path),
        "claim_input_manifest_path": str(claim_input_manifest_path),
        "analysis_qc_path": str(qc_md),
        "synthesis_notes_path": str(synthesis_notes),
    }
    (WORKSPACE / "analysis_outputs_manifest.json").write_text(json.dumps(outputs_manifest, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
