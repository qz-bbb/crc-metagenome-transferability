from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests


RUN_DIR = Path(r"D:\1\projects\gut_microbiome_colorectal_cancer\runs\run_20260525_182306")
W5 = RUN_DIR / "agents" / "05-analysis" / "workspace"
DATA_DIR = Path(r"D:\1\Knowledge Base\true_raw\data\BIOC_curatedMetagenomicData_CRC")
OUT = W5 / "results" / "M_robustness_interpretation"
SCRIPT_PATH = W5 / "scripts" / "build_robustness_interpretation_analysis.py"

META_PATH = DATA_DIR / "curated_crc_case_control_sample_metadata.tsv"
MATRIX_PATH = DATA_DIR / "curated_crc_case_control_species_relative_abundance.tsv.gz"
BASELINE_ASSOC_PATH = W5 / "results" / "C" / "cohort_adjusted_crc_control_association.tsv"
BASELINE_STABILITY_PATH = W5 / "results" / "D" / "cross_cohort_stability.tsv"
COVARIATE_PATH = W5 / "results" / "G" / "baseline_vs_covariate_sensitivity.tsv"
TRANSPORT_PATH = W5 / "results" / "H" / "transportability_score.tsv"
META_PATH_J = W5 / "results" / "J_meta_heterogeneity" / "random_effects_meta_analysis.tsv"
ORAL_PANEL_PATH = W5 / "results" / "K_ecology_oral_signature" / "oral_associated_taxa_panel.tsv"
MICROBIOMEHD_PATH = W5 / "results" / "E" / "microbiomehd_top100_genus_overlap.tsv"


def read_tsv(path: Path, **kwargs) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"missing input: {path}")
    return pd.read_csv(path, sep="\t", **kwargs)


def write_tsv(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=False)
    return path


def bh_fdr(pvalues: pd.Series) -> pd.Series:
    out = pd.Series(np.nan, index=pvalues.index, dtype=float)
    mask = pvalues.notna()
    if mask.any():
        out.loc[mask] = multipletests(pvalues.loc[mask].astype(float), method="fdr_bh")[1]
    return out


def genus_from_taxon(taxon: str) -> str:
    for part in str(taxon).replace(";", "|").split("|"):
        if part.startswith("g__"):
            return part.replace("g__", "")
    return ""


def species_from_taxon(taxon: str) -> str:
    for part in str(taxon).replace(";", "|").split("|"):
        if part.startswith("s__"):
            return part.replace("s__", "").replace("_", " ")
    return str(taxon)


def fit_feature_ols(y: np.ndarray, design: pd.DataFrame) -> tuple[float, float, float]:
    if np.nanstd(y) == 0:
        return np.nan, np.nan, np.nan
    try:
        fit = sm.OLS(y, design, missing="drop").fit(cov_type="HC3")
        return (
            float(fit.params["condition_crc"]),
            float(fit.bse["condition_crc"]),
            float(fit.pvalues["condition_crc"]),
        )
    except Exception:
        return np.nan, np.nan, np.nan


def make_design(metadata: pd.DataFrame, sample_index: pd.Index) -> pd.DataFrame:
    meta = metadata.set_index("sample_uid").loc[sample_index]
    condition = (meta["study_condition"] == "CRC").astype(int)
    study_dummies = pd.get_dummies(meta["study_name"], prefix="study", drop_first=True, dtype=float)
    design = pd.concat([condition.rename("condition_crc"), study_dummies], axis=1)
    return sm.add_constant(design, has_constant="add").astype(float)


def clr_transform(abundance: pd.DataFrame, pseudocount: float) -> pd.DataFrame:
    log_values = np.log(abundance + pseudocount)
    return log_values.sub(log_values.mean(axis=1), axis=0)


def select_taxa(abundance: pd.DataFrame, metadata: pd.DataFrame, prevalence_threshold: float, min_cohorts_present: int) -> list[str]:
    meta = metadata.set_index("sample_uid").loc[abundance.index]
    prevalence = (abundance > 0).mean(axis=0)
    cohort_presence = []
    for taxon in abundance.columns:
        present = 0
        for _, group in meta.groupby("study_name"):
            ids = group.index
            present += int((abundance.loc[ids, taxon] > 0).any())
        cohort_presence.append(present)
    cohort_presence = pd.Series(cohort_presence, index=abundance.columns)
    return abundance.columns[(prevalence >= prevalence_threshold) & (cohort_presence >= min_cohorts_present)].tolist()


def association_screen(clr: pd.DataFrame, abundance: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    design = make_design(metadata, clr.index)
    meta = metadata.set_index("sample_uid").loc[clr.index]
    condition = (meta["study_condition"] == "CRC")
    rows = []
    for taxon in clr.columns:
        coef, se, pvalue = fit_feature_ols(clr[taxon].to_numpy(float), design)
        crc_vals = abundance.loc[condition.index[condition], taxon]
        ctl_vals = abundance.loc[condition.index[~condition], taxon]
        rows.append({
            "taxon_long": taxon,
            "genus": genus_from_taxon(taxon),
            "species_short": species_from_taxon(taxon),
            "coef_crc_clr_adjusted": coef,
            "se_hc3": se,
            "pvalue": pvalue,
            "prevalence_crc": float((crc_vals > 0).mean()),
            "prevalence_control": float((ctl_vals > 0).mean()),
        })
    assoc = pd.DataFrame(rows)
    assoc["qvalue_bh"] = bh_fdr(assoc["pvalue"])
    assoc["direction_sign"] = np.sign(pd.to_numeric(assoc["coef_crc_clr_adjusted"], errors="coerce"))
    return assoc.sort_values(["qvalue_bh", "pvalue", "taxon_long"], na_position="last")


def stability_fraction(abundance: pd.DataFrame, metadata: pd.DataFrame, assoc: pd.DataFrame) -> pd.Series:
    meta = metadata.set_index("sample_uid").loc[abundance.index]
    out = {}
    for _, row in assoc.iterrows():
        taxon = row["taxon_long"]
        sign = np.sign(float(row["coef_crc_clr_adjusted"])) if np.isfinite(row["coef_crc_clr_adjusted"]) else 0
        tested = 0
        same = 0
        for _, group in meta.groupby("study_name"):
            crc_ids = group.index[group["study_condition"] == "CRC"]
            control_ids = group.index[group["study_condition"] == "control"]
            if len(crc_ids) == 0 or len(control_ids) == 0:
                continue
            diff = abundance.loc[crc_ids, taxon].median() - abundance.loc[control_ids, taxon].median()
            diff_sign = np.sign(float(diff))
            tested += 1
            if sign != 0 and diff_sign == sign:
                same += 1
        out[taxon] = same / tested if tested else np.nan
    return pd.Series(out)


def summarize_against_baseline(assoc: pd.DataFrame, baseline_assoc: pd.DataFrame, abundance: pd.DataFrame, metadata: pd.DataFrame) -> dict[str, object]:
    base = baseline_assoc.set_index("taxon_long")
    current = assoc.set_index("taxon_long")
    common = current.index.intersection(base.index)
    top100_base = set(baseline_assoc.head(100)["taxon_long"])
    top100_current = set(assoc.head(min(100, len(assoc)))["taxon_long"])
    signs_same = (
        np.sign(current.loc[common, "coef_crc_clr_adjusted"].astype(float))
        == np.sign(base.loc[common, "coef_crc_clr_adjusted"].astype(float))
    )
    stability = stability_fraction(abundance[current.index], metadata, assoc.reset_index())
    return {
        "retained_features": int(len(assoc)),
        "fdr_lt_0_10": int((pd.to_numeric(assoc["qvalue_bh"], errors="coerce") < 0.10).sum()),
        "fdr_lt_0_05": int((pd.to_numeric(assoc["qvalue_bh"], errors="coerce") < 0.05).sum()),
        "top100_overlap_with_baseline": int(len(top100_base.intersection(top100_current))),
        "common_features_with_baseline": int(len(common)),
        "direction_concordance_with_baseline": float(np.mean(signs_same)) if len(signs_same) else np.nan,
        "stable_candidates_fdr10_stability60": int(((pd.to_numeric(assoc.set_index("taxon_long")["qvalue_bh"], errors="coerce") < 0.10) & (stability >= 0.60)).sum()),
    }


def run_filter_sensitivity(abundance: pd.DataFrame, metadata: pd.DataFrame, baseline_assoc: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for prevalence_threshold in [0.03, 0.05, 0.10]:
        for min_cohorts in [2, 3, 4]:
            taxa = select_taxa(abundance, metadata, prevalence_threshold, min_cohorts)
            subset = abundance[taxa]
            min_positive = subset.where(subset > 0).min().min()
            pseudocount = float(min_positive / 2)
            assoc = association_screen(clr_transform(subset, pseudocount), subset, metadata)
            summary = summarize_against_baseline(assoc, baseline_assoc, subset, metadata)
            rows.append({
                "prevalence_threshold": prevalence_threshold,
                "min_cohorts_present": min_cohorts,
                "pseudocount": pseudocount,
                **summary,
            })
    return pd.DataFrame(rows)


def run_pseudocount_sensitivity(abundance: pd.DataFrame, metadata: pd.DataFrame, baseline_assoc: pd.DataFrame) -> pd.DataFrame:
    taxa = baseline_assoc["taxon_long"].tolist()
    subset = abundance[taxa]
    min_positive = subset.where(subset > 0).min().min()
    rows = []
    for multiplier in [0.25, 0.50, 1.00]:
        pseudocount = float(min_positive * multiplier)
        assoc = association_screen(clr_transform(subset, pseudocount), subset, metadata)
        summary = summarize_against_baseline(assoc, baseline_assoc, subset, metadata)
        rows.append({
            "pseudocount_rule": f"minimum_positive_x_{multiplier:g}",
            "pseudocount": pseudocount,
            **summary,
        })
    return pd.DataFrame(rows)


def run_stability_threshold_sensitivity(stability: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for fdr_cutoff in [0.05, 0.10]:
        for threshold in [0.50, 0.60, 0.70, 0.80]:
            hits = stability[(pd.to_numeric(stability["adjusted_qvalue"], errors="coerce") < fdr_cutoff) & (pd.to_numeric(stability["stability_fraction"], errors="coerce") >= threshold)]
            rows.append({
                "fdr_cutoff": fdr_cutoff,
                "same_direction_threshold": threshold,
                "candidate_count": int(len(hits)),
                "crc_higher": int((pd.to_numeric(hits["adjusted_coef"], errors="coerce") > 0).sum()),
                "control_higher": int((pd.to_numeric(hits["adjusted_coef"], errors="coerce") < 0).sum()),
            })
    return pd.DataFrame(rows)


def run_leave_one_cohort_association(abundance: pd.DataFrame, metadata: pd.DataFrame, baseline_assoc: pd.DataFrame, stable_taxa: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    cohorts = sorted(metadata["study_name"].astype(str).unique())
    taxa = baseline_assoc["taxon_long"].tolist()
    base_sign = baseline_assoc.set_index("taxon_long")["coef_crc_clr_adjusted"].astype(float).apply(np.sign)
    for left_out in cohorts:
        keep_ids = metadata.loc[metadata["study_name"].astype(str) != left_out, "sample_uid"].tolist()
        subset_abundance = abundance.loc[keep_ids, taxa]
        min_positive = subset_abundance.where(subset_abundance > 0).min().min()
        assoc = association_screen(clr_transform(subset_abundance, float(min_positive / 2)), subset_abundance, metadata[metadata["sample_uid"].isin(keep_ids)])
        assoc = assoc.set_index("taxon_long")
        for taxon in stable_taxa:
            row = assoc.loc[taxon]
            rows.append({
                "taxon_long": taxon,
                "genus": genus_from_taxon(taxon),
                "species_short": species_from_taxon(taxon),
                "left_out_cohort": left_out,
                "coef_crc_clr_adjusted": float(row["coef_crc_clr_adjusted"]),
                "pvalue": float(row["pvalue"]),
                "qvalue_bh": float(row["qvalue_bh"]),
                "same_direction_as_baseline": bool(np.sign(float(row["coef_crc_clr_adjusted"])) == base_sign.loc[taxon]),
                "fdr_lt_0_10": bool(float(row["qvalue_bh"]) < 0.10),
            })
    loco = pd.DataFrame(rows)
    summary_rows = []
    for taxon, group in loco.groupby("taxon_long", sort=False):
        summary_rows.append({
            "taxon_long": taxon,
            "genus": genus_from_taxon(taxon),
            "species_short": species_from_taxon(taxon),
            "leave_one_cohort_models": int(len(group)),
            "same_direction_models": int(group["same_direction_as_baseline"].sum()),
            "same_direction_fraction": float(group["same_direction_as_baseline"].mean()),
            "fdr_lt_0_10_models": int(group["fdr_lt_0_10"].sum()),
            "fdr_lt_0_10_fraction": float(group["fdr_lt_0_10"].mean()),
            "min_qvalue": float(pd.to_numeric(group["qvalue_bh"], errors="coerce").min()),
            "max_qvalue": float(pd.to_numeric(group["qvalue_bh"], errors="coerce").max()),
        })
    summary = pd.DataFrame(summary_rows)
    summary["leave_one_cohort_robust"] = (summary["same_direction_fraction"] >= 0.90) & (summary["fdr_lt_0_10_fraction"] >= 0.80)
    return loco, summary.sort_values(["leave_one_cohort_robust", "same_direction_fraction", "fdr_lt_0_10_fraction"], ascending=[False, False, False])


def build_candidate_interpretation(
    stability: pd.DataFrame,
    covariate: pd.DataFrame,
    transport: pd.DataFrame,
    meta: pd.DataFrame,
    oral_panel: pd.DataFrame,
    microhd: pd.DataFrame,
    loco_summary: pd.DataFrame,
) -> pd.DataFrame:
    stable = stability[(pd.to_numeric(stability["adjusted_qvalue"], errors="coerce") < 0.10) & (pd.to_numeric(stability["stability_fraction"], errors="coerce") >= 0.60)].copy()
    oral_taxa = set(oral_panel["taxon_long"].astype(str))
    microhd_status = microhd.set_index("taxon_long")["benchmark_match_status"].to_dict()
    merged = (
        stable.merge(
            covariate[["taxon_long", "qvalue_bh_coef_crc_clr_study_age_bmi_gender", "direction_concordant"]],
            on="taxon_long",
            how="left",
        )
        .merge(
            transport[["taxon_long", "transportability_rank", "selected_fraction", "mean_abs_elasticnet_coef"]],
            on="taxon_long",
            how="left",
        )
        .merge(
            meta[["taxon_long", "random_effect", "random_ci_low", "random_ci_high", "qvalue_bh_random_effect", "i2_percent", "meta_stable_candidate"]],
            on="taxon_long",
            how="left",
        )
        .merge(
            loco_summary[["taxon_long", "same_direction_fraction", "fdr_lt_0_10_fraction", "leave_one_cohort_robust"]],
            on="taxon_long",
            how="left",
        )
    )
    merged["direction"] = np.where(pd.to_numeric(merged["adjusted_coef"], errors="coerce") > 0, "CRC_higher", "control_higher")
    merged["oral_associated_taxonomy_panel"] = merged["taxon_long"].isin(oral_taxa)
    merged["microbiomehd_genus_match"] = merged["taxon_long"].map(microhd_status).fillna("NOT_IN_TOP100_CONTEXT")
    merged["covariate_supported"] = pd.to_numeric(merged["qvalue_bh_coef_crc_clr_study_age_bmi_gender"], errors="coerce") < 0.10
    merged["meta_supported"] = pd.to_numeric(merged["qvalue_bh_random_effect"], errors="coerce") < 0.10
    merged["elasticnet_supported"] = pd.to_numeric(merged["selected_fraction"], errors="coerce") >= 0.50
    merged["high_heterogeneity"] = pd.to_numeric(merged["i2_percent"], errors="coerce") >= 75
    layer_cols = ["covariate_supported", "meta_supported", "elasticnet_supported", "leave_one_cohort_robust", "oral_associated_taxonomy_panel"]
    merged["support_layer_count"] = merged[layer_cols].fillna(False).astype(bool).sum(axis=1)
    merged["interpretation_group"] = np.select(
        [
            merged["leave_one_cohort_robust"].fillna(False) & merged["meta_supported"].fillna(False) & ~merged["high_heterogeneity"].fillna(False),
            merged["high_heterogeneity"].fillna(False),
            merged["oral_associated_taxonomy_panel"].fillna(False) & (merged["direction"] == "CRC_higher"),
        ],
        ["high_consistency_candidate", "heterogeneous_candidate", "oral_associated_context_candidate"],
        default="bounded_candidate",
    )
    columns = [
        "taxon_long",
        "genus",
        "species_short",
        "direction",
        "adjusted_coef",
        "adjusted_qvalue",
        "stability_fraction",
        "cohorts_same_direction",
        "qvalue_bh_coef_crc_clr_study_age_bmi_gender",
        "random_effect",
        "qvalue_bh_random_effect",
        "i2_percent",
        "same_direction_fraction",
        "fdr_lt_0_10_fraction",
        "leave_one_cohort_robust",
        "selected_fraction",
        "transportability_rank",
        "oral_associated_taxonomy_panel",
        "microbiomehd_genus_match",
        "support_layer_count",
        "interpretation_group",
    ]
    return merged[columns].sort_values(["support_layer_count", "stability_fraction", "adjusted_qvalue"], ascending=[False, False, True])


def update_interfaces(paths: dict[str, Path], summary: pd.DataFrame) -> None:
    module_status_path = W5 / "module_status.tsv"
    status = read_tsv(module_status_path)
    status = status[status["module_id"] != "M"]
    status = pd.concat([
        status,
        pd.DataFrame([{
            "module_id": "M",
            "status": "COMPLETED",
            "primary_output": str(paths["summary"]),
            "notes": "Robustness and stable-candidate interpretation completed; thresholds, pseudocounts, leave-one-cohort-out association, and evidence-layer summaries were computed.",
        }]),
    ], ignore_index=True)
    write_tsv(status, module_status_path)

    fig_map_path = W5 / "figure_data_map.tsv"
    fig_map = read_tsv(fig_map_path)
    fig_map = fig_map[fig_map["figure_id"] != "Fig6"]
    fig6_rows = [
        {"figure_id": "Fig6", "panel": "a", "module_id": "M", "claim_id": "C10", "source_data_file": str(paths["filter"]), "generation_script": str(SCRIPT_PATH), "result_type": "filter_threshold_sensitivity", "status": "READY", "notes": "Filter prevalence/cohort-presence sensitivity for retained features, FDR counts, top-rank overlap, and stable candidates."},
        {"figure_id": "Fig6", "panel": "b", "module_id": "M", "claim_id": "C10", "source_data_file": str(paths["pseudocount"]), "generation_script": str(SCRIPT_PATH), "result_type": "pseudocount_sensitivity", "status": "READY", "notes": "Pseudocount sensitivity for direction concordance, FDR counts, and top-rank overlap."},
        {"figure_id": "Fig6", "panel": "c", "module_id": "M", "claim_id": "C11", "source_data_file": str(paths["loco_summary"]), "generation_script": str(SCRIPT_PATH), "result_type": "leave_one_cohort_association_sensitivity", "status": "READY", "notes": "Leave-one-cohort-out association robustness for stable candidates."},
        {"figure_id": "Fig6", "panel": "d", "module_id": "M", "claim_id": "C11", "source_data_file": str(paths["candidate"]), "generation_script": str(SCRIPT_PATH), "result_type": "stable_candidate_interpretation_map", "status": "READY", "notes": "Evidence-layer interpretation map for the 29 stable candidates."},
    ]
    fig_map = pd.concat([fig_map, pd.DataFrame(fig6_rows)], ignore_index=True)
    write_tsv(fig_map, fig_map_path)

    manifest_path = W5 / "analysis_outputs_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig")) if manifest_path.exists() else {}
    for key, value in paths.items():
        manifest[f"robustness_{key}_path"] = str(value)
    for _, row in summary.iterrows():
        manifest[f"robustness_metric_{row['metric']}"] = row["value"]
    manifest["robustness_script_path"] = str(SCRIPT_PATH)
    manifest_path.write_text(json.dumps(manifest, indent=4, ensure_ascii=False), encoding="utf-8")

    log_path = W5 / "module_execution_log.md"
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write("\n- Module M completed: robustness sensitivity and stable-candidate interpretation tables generated from existing processed matrix.\n")

    synthesis_path = W5 / "synthesis_notes.md"
    with synthesis_path.open("a", encoding="utf-8") as handle:
        handle.write("\n## Module M robustness and interpretation addendum\n\n")
        for _, row in summary.iterrows():
            handle.write(f"- {row['metric']}: {row['value']} ({row['interpretation']})\n")

    handoff_path = W5 / "handoff.json"
    handoff = json.loads(handoff_path.read_text(encoding="utf-8-sig"))
    outputs = set(handoff.get("outputs_written", []))
    for value in paths.values():
        outputs.add(str(value))
    outputs.update([str(module_status_path), str(fig_map_path), str(manifest_path), str(log_path), str(synthesis_path), str(handoff_path)])
    handoff["outputs_written"] = sorted(outputs)
    handoff.setdefault("evidence_level_summary", {})["robustness_interpretation"] = "Module M added threshold, pseudocount, leave-one-cohort-out association, and stable-candidate evidence-layer robustness checks."
    handoff.setdefault("limitations", []).append("Module M sensitivity analyses remain processed-matrix robustness checks; they do not add raw-read, functional, strain-level, causal, diagnostic, or stage-specific evidence.")
    handoff["created_at"] = datetime.now().astimezone().isoformat()
    handoff["module_status_summary"] = status["status"].value_counts().to_dict()
    handoff_path.write_text(json.dumps(handoff, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    metadata = read_tsv(META_PATH)
    matrix = read_tsv(MATRIX_PATH, index_col=0).T
    matrix = matrix.loc[metadata["sample_uid"]]
    baseline_assoc = read_tsv(BASELINE_ASSOC_PATH)
    stability = read_tsv(BASELINE_STABILITY_PATH)
    covariate = read_tsv(COVARIATE_PATH)
    transport = read_tsv(TRANSPORT_PATH)
    meta = read_tsv(META_PATH_J)
    oral_panel = read_tsv(ORAL_PANEL_PATH)
    microhd = read_tsv(MICROBIOMEHD_PATH)

    stable_taxa = stability.loc[
        (pd.to_numeric(stability["adjusted_qvalue"], errors="coerce") < 0.10)
        & (pd.to_numeric(stability["stability_fraction"], errors="coerce") >= 0.60),
        "taxon_long",
    ].tolist()

    filter_sensitivity = run_filter_sensitivity(matrix, metadata, baseline_assoc)
    pseudocount_sensitivity = run_pseudocount_sensitivity(matrix, metadata, baseline_assoc)
    stability_thresholds = run_stability_threshold_sensitivity(stability)
    loco, loco_summary = run_leave_one_cohort_association(matrix, metadata, baseline_assoc, stable_taxa)
    candidate = build_candidate_interpretation(stability, covariate, transport, meta, oral_panel, microhd, loco_summary)

    paths = {
        "filter": write_tsv(filter_sensitivity, OUT / "filter_threshold_sensitivity.tsv"),
        "pseudocount": write_tsv(pseudocount_sensitivity, OUT / "pseudocount_sensitivity.tsv"),
        "stability_thresholds": write_tsv(stability_thresholds, OUT / "stability_threshold_sensitivity.tsv"),
        "loco": write_tsv(loco, OUT / "leave_one_cohort_association_sensitivity.tsv"),
        "loco_summary": write_tsv(loco_summary, OUT / "leave_one_cohort_candidate_summary.tsv"),
        "candidate": write_tsv(candidate, OUT / "stable_candidate_interpretation.tsv"),
    }

    summary_rows = [
        {"metric": "filter_configs_tested", "value": len(filter_sensitivity), "interpretation": "Prevalence and cohort-presence sensitivity configurations."},
        {"metric": "filter_fdr10_range", "value": f"{int(filter_sensitivity['fdr_lt_0_10'].min())}-{int(filter_sensitivity['fdr_lt_0_10'].max())}", "interpretation": "Range of FDR<0.10 feature counts across filter sensitivity configurations."},
        {"metric": "filter_top100_overlap_range", "value": f"{int(filter_sensitivity['top100_overlap_with_baseline'].min())}-{int(filter_sensitivity['top100_overlap_with_baseline'].max())}", "interpretation": "Range of top-100 overlap with the baseline filter."},
        {"metric": "pseudocount_direction_concordance_min", "value": f"{float(pseudocount_sensitivity['direction_concordance_with_baseline'].min()):.3f}", "interpretation": "Minimum direction concordance versus baseline across pseudocount settings."},
        {"metric": "pseudocount_top100_overlap_min", "value": int(pseudocount_sensitivity['top100_overlap_with_baseline'].min()), "interpretation": "Minimum top-100 overlap versus baseline across pseudocount settings."},
        {"metric": "stable_candidates_fdr10_stability60", "value": len(stable_taxa), "interpretation": "Primary stable-candidate count under FDR<0.10 and >=60% same-direction support."},
        {"metric": "stable_candidates_fdr10_stability70", "value": int(stability_thresholds[(stability_thresholds["fdr_cutoff"] == 0.10) & (stability_thresholds["same_direction_threshold"] == 0.70)]["candidate_count"].iloc[0]), "interpretation": "Stable-candidate count under stricter >=70% same-direction support."},
        {"metric": "leave_one_cohort_robust_candidates", "value": int(candidate["leave_one_cohort_robust"].fillna(False).sum()), "interpretation": "Stable candidates preserving direction in >=90% and FDR<0.10 in >=80% leave-one-cohort-out association screens."},
        {"metric": "stable_candidate_oral_panel_count", "value": int(candidate["oral_associated_taxonomy_panel"].fillna(False).sum()), "interpretation": "Stable candidates that are members of the literature-prior oral-associated taxonomy panel."},
        {"metric": "stable_candidate_microbiomehd_genus_match_count", "value": int((candidate["microbiomehd_genus_match"] == "GENUS_MATCH").sum()), "interpretation": "Stable candidates with genus-level MicrobiomeHD context among the top-100 benchmark table."},
        {"metric": "high_consistency_candidate_count", "value": int((candidate["interpretation_group"] == "high_consistency_candidate").sum()), "interpretation": "Stable candidates with coherent leave-one-cohort and random-effects support without high I2."},
    ]
    summary = pd.DataFrame(summary_rows)
    paths["summary"] = write_tsv(summary, OUT / "robustness_interpretation_summary.tsv")

    report = ["# Robustness And Stable-Candidate Interpretation Report", ""]
    for _, row in summary.iterrows():
        report.append(f"- {row['metric']}: {row['value']} - {row['interpretation']}")
    report.append("")
    report.append("These analyses are processed-matrix robustness checks. They do not authorize raw-read, functional, strain-level, causal, diagnostic, oral-source, or stage-specific claims.")
    (OUT / "robustness_interpretation_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    paths["report"] = OUT / "robustness_interpretation_report.md"

    update_interfaces(paths, summary)
    print(json.dumps({"status": "pass", "outputs": {k: str(v) for k, v in paths.items()}}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
