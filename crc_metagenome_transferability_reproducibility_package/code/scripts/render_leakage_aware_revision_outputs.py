from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.api.types import is_object_dtype, is_string_dtype


RUN_DIR = Path(r"D:\1\projects\gut_microbiome_colorectal_cancer\runs\run_20260525_182306")
ANALYSIS_DIR = RUN_DIR / "agents" / "05-analysis" / "workspace" / "results" / "Q_leakage_aware_prioritization"
FIG_WORK = RUN_DIR / "agents" / "06-figures" / "workspace" / "figures" / "Q_leakage_aware_revision"
BMC = RUN_DIR / "BMC_submission"
BMC_FIG = BMC / "figures"
BMC_RESULTS = BMC / "results"
BMC_SUPP = BMC / "supplementary_tables"
REPRO = BMC / "crc_metagenome_transferability_reproducibility_package"

FIG_WORK.mkdir(parents=True, exist_ok=True)
BMC_FIG.mkdir(parents=True, exist_ok=True)
BMC_RESULTS.mkdir(parents=True, exist_ok=True)
BMC_SUPP.mkdir(parents=True, exist_ok=True)


PANEL_LABELS = {
    "all_301_features": "All 301",
    "training_only_top29_q": "Training top29 q",
    "training_only_stability_panel": "Training stability",
    "training_only_high_consistency_panel": "Training high18",
    "global_top29_q": "Global top29 q",
    "global_stable29": "Global stable29",
    "global_high_consistency18": "Global high18",
}


GUILD_LABELS = {
    "butyrate_SCFA_associated_commensal_panel": "Butyrate/SCFA-\nassociated commensal",
    "butyrate_SCFA_commensal_panel": "Butyrate/SCFA-\nassociated commensal",
    "oral_pathobiont_associated_panel": "Oral/pathobiont\nassociated",
    "Bacteroides_Enterobacteriaceae_context_panel": "Bacteroides/\nEnterobacteriaceae context",
    "inflammation_Bacteroides_Enterobacteriaceae_panel": "Bacteroides/\nEnterobacteriaceae context",
}


def partial_r2_column(df: pd.DataFrame) -> str:
    if "term_deletion_partial_r2" in df.columns:
        return "term_deletion_partial_r2"
    return "partial_or_marginal_r2"


def save_fig(fig: plt.Figure, base: str) -> dict[str, str]:
    paths = {
        "work_pdf": FIG_WORK / f"{base}.pdf",
        "work_png": FIG_WORK / f"{base}.png",
        "bmc_pdf": BMC_FIG / f"{base}.pdf",
        "bmc_png": BMC_FIG / f"{base}.png",
    }
    fig.savefig(paths["work_pdf"], bbox_inches="tight")
    fig.savefig(paths["work_png"], dpi=300, bbox_inches="tight")
    shutil.copy2(paths["work_pdf"], paths["bmc_pdf"])
    shutil.copy2(paths["work_png"], paths["bmc_png"])
    return {k: str(v) for k, v in paths.items()}


def figure_variance() -> dict[str, str]:
    df = pd.read_csv(ANALYSIS_DIR / "aitchison_variance_partitioning.csv")
    terms = df[df["term"] != "residual"].copy()
    residual = df[df["term"] == "residual"].copy()
    labels = terms["term"].replace({"study_label": "Study label", "CRC_status": "CRC status"}).tolist() + ["Residual"]
    values = (terms[partial_r2_column(terms)].to_numpy(dtype=float) * 100).tolist() + (residual["r2"].to_numpy(dtype=float) * 100).tolist()
    colors = ["#4C78A8", "#E45756", "#72B7B2", "#72B7B2", "#72B7B2", "#D0D0D0"]
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    y = np.arange(len(labels))
    ax.barh(y, values, color=colors[: len(values)], edgecolor="white")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Term-deletion partial or residual variation (%)")
    ax.set_title("Aitchison-space variance partitioning")
    for yi, val, term in zip(y, values, labels):
        if term == "Residual":
            txt = f"{val:.1f}%"
        else:
            p = terms.iloc[yi]["p_value"]
            p_label = "p≤0.001" if p <= 0.001 else f"p={p:.3f}"
            txt = f"{val:.2f}%\n{p_label}"
        ax.text(val + 0.8, yi, txt, va="center", fontsize=9)
    ax.set_xlim(0, max(values) * 1.18)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    paths = save_fig(fig, "Figure_variance_partitioning")
    plt.close(fig)
    return paths


def figure_1_overview() -> dict[str, str]:
    pca = pd.read_csv(RUN_DIR / "agents" / "05-analysis" / "workspace" / "results" / "H" / "aitchison_pca_scores.tsv", sep="\t")
    feature = pd.read_csv(RUN_DIR / "agents" / "05-analysis" / "workspace" / "processed_data" / "B" / "feature_filter_summary.tsv", sep="\t")
    variance = pd.read_csv(ANALYSIS_DIR / "aitchison_variance_partitioning.csv")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))

    ax = axes[0]
    colors = {"CRC": "#D55E00", "control": "#0072B2"}
    markers = {"CRC": "o", "control": "x"}
    for status, group in pca.groupby("study_condition"):
        ax.scatter(group["PC1"], group["PC2"], s=16, alpha=0.62, color=colors.get(status, "#666666"), marker=markers.get(status, "o"), label=status)
    pc1 = float(pca["PC1_explained_variance_ratio"].iloc[0]) * 100
    pc2 = float(pca["PC2_explained_variance_ratio"].iloc[0]) * 100
    ax.set_xlabel(f"PC1 ({pc1:.1f}% var.)")
    ax.set_ylabel(f"PC2 ({pc2:.1f}% var.)")
    ax.set_title("A. Aitchison PCA")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    terms = variance[variance["term"].isin(["study_label", "CRC_status", "age", "BMI", "gender"])].copy()
    labels = terms["term"].replace({"study_label": "Study label", "CRC_status": "CRC status"}).tolist()
    values = terms[partial_r2_column(terms)].to_numpy(dtype=float) * 100
    y = np.arange(len(labels))
    ax.barh(y, values, color=["#4C78A8", "#E45756", "#72B7B2", "#72B7B2", "#72B7B2"], edgecolor="white")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Term-deletion partial variation (%)")
    ax.set_title("B. Variance partitioning")
    for yi, val in zip(y, values):
        ax.text(val + 0.15, yi, f"{val:.2f}%", va="center", fontsize=8)
    residual = variance.loc[variance["term"] == "residual", "r2"].iloc[0] * 100
    ax.text(0.98, 0.05, f"Residual: {residual:.1f}%", transform=ax.transAxes, ha="right", va="bottom", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[2]
    retained = feature["retained"].astype(bool)
    ax.scatter(feature.loc[~retained, "prevalence_overall"], feature.loc[~retained, "mean_abundance_percent"], s=10, alpha=0.25, color="#BDBDBD", label="Filtered")
    ax.scatter(feature.loc[retained, "prevalence_overall"], feature.loc[retained, "mean_abundance_percent"], s=12, alpha=0.65, color="#4C9A9A", label="Retained")
    ax.set_yscale("log")
    ax.set_xlabel("Overall prevalence")
    ax.set_ylabel("Mean relative abundance (%)")
    ax.set_title("C. Feature prevalence-abundance")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    paths = save_fig(fig, "Figure_1")
    plt.close(fig)
    return paths


def figure_leakage_panel() -> dict[str, str]:
    leakage = pd.read_csv(ANALYSIS_DIR / "leakage_aware_loso_panel_benchmark.csv")
    random_summary = pd.read_csv(ANALYSIS_DIR / "random_panel_loso_benchmark_summary.csv")
    comp = pd.read_csv(ANALYSIS_DIR / "panel_comparison_statistics.csv")
    fixed = leakage[~leakage["panel_type"].str.startswith("random29")].copy()
    fixed["panel_label"] = fixed["panel_type"].map(PANEL_LABELS)
    order = ["all_301_features", "training_only_top29_q", "training_only_stability_panel", "training_only_high_consistency_panel"]
    fig, axes = plt.subplots(2, 2, figsize=(12.2, 8.2))

    ax = axes[0, 0]
    data = [fixed.loc[fixed["panel_type"] == p, "auroc"].dropna() for p in order]
    ax.boxplot(data, labels=[PANEL_LABELS[p] for p in order], showfliers=False, patch_artist=True, boxprops={"facecolor": "#D8E3EF"})
    for i, vals in enumerate(data, start=1):
        ax.scatter(np.full(len(vals), i) + np.linspace(-0.08, 0.08, len(vals)), vals, s=24, color="#2F3A45", alpha=0.8)
    ax.axhline(0.5, color="#999999", linestyle="--", linewidth=1)
    ax.set_ylabel("Held-out AUROC")
    ax.set_title("A. Leakage-aware LOSO panels")
    ax.tick_params(axis="x", rotation=18)

    ax = axes[0, 1]
    overlap_order = ["training_only_top29_q", "training_only_stability_panel", "training_only_high_consistency_panel"]
    med = fixed.groupby("panel_type")["overlap_with_global_top29_q"].median().reindex(overlap_order)
    stable = fixed.groupby("panel_type")["overlap_with_global_stable29"].median().reindex(overlap_order)
    high = fixed.groupby("panel_type")["overlap_with_global_high_consistency18"].median().reindex(overlap_order)
    x = np.arange(len(overlap_order))
    width = 0.25
    ax.bar(x - width, med, width, label="Global top29 q", color="#4C78A8")
    ax.bar(x, stable, width, label="Global stable29", color="#F58518")
    ax.bar(x + width, high, width, label="Global high18", color="#54A24B")
    ax.set_xticks(x)
    ax.set_xticklabels([PANEL_LABELS[p] for p in overlap_order], rotation=18, ha="right")
    ax.set_ylabel("Median overlap count")
    ax.set_title("B. Median overlap with full-data reference panels")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1, 0]
    for random_type, color in [("random29_unmatched", "#9E9E9E"), ("random29_prevalence_matched", "#6BAED6")]:
        sub = random_summary[random_summary["random_type"] == random_type]
        ax.scatter(sub["median_auroc"], sub["observed_training_stability_panel_auroc"], label=random_type.replace("random29_", ""), color=color, alpha=0.85)
    ax.plot([0.4, 0.9], [0.4, 0.9], color="#999999", linestyle="--", linewidth=1)
    ax.set_xlim(0.48, 0.88)
    ax.set_ylim(0.48, 0.88)
    ax.set_xlabel("Random-panel median AUROC in fold")
    ax.set_ylabel("Observed training-stability AUROC")
    ax.set_title("C. Random-panel fold context")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1, 1]
    loso = comp[comp["metric"] == "LOSO AUROC across held-out studies"].copy()
    y = np.arange(len(loso))
    ax.errorbar(loso["median_difference"], y, xerr=[loso["median_difference"] - loso["bootstrap_ci_low"], loso["bootstrap_ci_high"] - loso["median_difference"]], fmt="o", color="#333333", ecolor="#666666", capsize=3)
    ax.axvline(0, color="#999999", linestyle="--", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels([c.replace("training_only_", "").replace("_", " ") for c in loso["comparison"]], fontsize=8)
    ax.set_xlabel("Median paired AUROC difference")
    ax.set_title("D. Paired LOSO differences")
    fig.tight_layout()
    paths = save_fig(fig, "Figure_7")
    shutil.copy2(BMC_FIG / "Figure_7.pdf", BMC_FIG / "Figure_leakage_aware_panel_benchmarking.pdf")
    shutil.copy2(BMC_FIG / "Figure_7.png", BMC_FIG / "Figure_leakage_aware_panel_benchmarking.png")
    plt.close(fig)
    return paths


def figure_guilds() -> dict[str, str]:
    random = pd.read_csv(ANALYSIS_DIR / "ecological_guild_random_effects.csv")
    per = pd.read_csv(ANALYSIS_DIR / "ecological_guild_per_cohort_hedges_g.csv")
    loso = pd.read_csv(ANALYSIS_DIR / "ecological_guild_loso_score_only.csv")
    member = pd.read_csv(ANALYSIS_DIR / "ecological_guild_membership.csv")
    random["label"] = random["guild"].map(GUILD_LABELS)
    fig, axes = plt.subplots(2, 2, figsize=(12.2, 8.2))

    ax = axes[0, 0]
    y = np.arange(len(random))
    ax.errorbar(random["random_effect"], y, xerr=[random["random_effect"] - random["random_ci_low"], random["random_ci_high"] - random["random_effect"]], fmt="o", color="#333333", ecolor="#666666", capsize=3)
    ax.axvline(0, color="#999999", linestyle="--", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(random["label"])
    for yi, row in random.iterrows():
        ax.text(row["random_ci_high"] + 0.04, yi, f"q={row['q_value']:.2g}; I²={row['i2_percent']:.1f}%", va="center", fontsize=8)
    ax.set_xlabel("Random-effects Hedges g (CRC minus control)")
    ax.set_title("A. Guild-level random-effects summaries")

    ax = axes[0, 1]
    guilds = list(random["guild"])
    for i, guild in enumerate(guilds):
        vals = per.loc[per["guild"] == guild, "hedges_g_crc_minus_control"].dropna()
        ax.scatter(np.full(len(vals), i), vals, color="#4C78A8", alpha=0.75)
        ax.plot([i - 0.22, i + 0.22], [vals.median(), vals.median()], color="#222222", linewidth=2)
    ax.axhline(0, color="#999999", linestyle="--", linewidth=1)
    ax.set_xticks(np.arange(len(guilds)))
    ax.set_xticklabels([GUILD_LABELS[g] for g in guilds], rotation=18, ha="right")
    ax.set_ylabel("Cohort Hedges g")
    ax.set_title("B. Cohort-level effect spread")

    ax = axes[1, 0]
    overlap = member.groupby("guild").agg(total=("taxon", "count"), stable=("stable_candidate_yes_no", lambda s: (s == "yes").sum())).reindex(guilds)
    x = np.arange(len(guilds))
    ax.bar(x, overlap["total"], color="#D8D8D8", label="Panel taxa")
    ax.bar(x, overlap["stable"], color="#F58518", label="Stable candidates")
    ax.set_xticks(x)
    ax.set_xticklabels([GUILD_LABELS[g] for g in guilds], rotation=18, ha="right")
    ax.set_ylabel("Taxon count")
    ax.set_title("C. Guild membership and stable-candidate overlap")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1, 1]
    data = [loso.loc[loso["guild"] == g, "auroc"].dropna() for g in guilds]
    ax.boxplot(data, labels=[GUILD_LABELS[g] for g in guilds], showfliers=False, patch_artist=True, boxprops={"facecolor": "#DDEAD7"})
    ax.axhline(0.5, color="#999999", linestyle="--", linewidth=1)
    ax.set_ylabel("Score-only LOSO AUROC")
    ax.set_title("D. Score-only separability stress test")
    ax.tick_params(axis="x", rotation=18)
    fig.tight_layout()
    paths = save_fig(fig, "Figure_8")
    shutil.copy2(BMC_FIG / "Figure_8.pdf", BMC_FIG / "Figure_ecological_guild_scores.pdf")
    shutil.copy2(BMC_FIG / "Figure_8.png", BMC_FIG / "Figure_ecological_guild_scores.png")
    plt.close(fig)
    return paths


def copy_results() -> list[str]:
    copied = []
    mapping = {
        "aitchison_variance_partitioning.csv": "aitchison_variance_partitioning.csv",
        "leakage_aware_loso_panel_benchmark.csv": "leakage_aware_loso_panel_benchmark.csv",
        "random_panel_loso_benchmark_summary.csv": "random_panel_loso_benchmark_summary.csv",
        "fixed_global_panel_loso_benchmark.csv": "fixed_global_panel_loso_benchmark.csv",
        "fixed_global_panel_pairwise_benchmark.csv": "fixed_global_panel_pairwise_benchmark.csv",
        "fixed_global_panel_loso_detail.csv": "fixed_global_panel_loso_detail.csv",
        "fixed_global_panel_pairwise_detail.csv": "fixed_global_panel_pairwise_detail.csv",
        "ecological_guild_scores.csv": "ecological_guild_scores.csv",
        "ecological_guild_random_effects.csv": "ecological_guild_random_effects.csv",
        "ecological_guild_loco_robustness.csv": "ecological_guild_loco_robustness.csv",
        "ecological_guild_loso_score_only.csv": "ecological_guild_loso_score_only.csv",
        "ecological_guild_membership.csv": "ecological_guild_membership.csv",
        "ecological_guild_per_cohort_hedges_g.csv": "ecological_guild_per_cohort_hedges_g.csv",
        "panel_comparison_statistics.csv": "panel_comparison_statistics.csv",
        "claim_decision_table.csv": "claim_decision_table.csv",
    }
    guild_value_mapping = {
        "butyrate_SCFA_commensal_panel": "butyrate_SCFA_associated_commensal_panel",
        "inflammation_Bacteroides_Enterobacteriaceae_panel": "Bacteroides_Enterobacteriaceae_context_panel",
    }
    for src_name, dst_name in mapping.items():
        src = ANALYSIS_DIR / src_name
        if src.exists():
            dst = BMC_RESULTS / dst_name
            if src_name == "aitchison_variance_partitioning.csv":
                df = pd.read_csv(src)
                if "partial_or_marginal_r2" in df.columns:
                    df = df.rename(columns={"partial_or_marginal_r2": "term_deletion_partial_r2"})
                if "notes" in df.columns:
                    df["notes"] = df["notes"].astype(str).str.replace(
                        "partial_or_marginal_r2 tests the term against a model containing the other listed terms.",
                        "term_deletion_partial_r2 tests the term against a model containing the other listed terms.",
                        regex=False,
                    )
                df.to_csv(dst, index=False)
            elif src_name == "leakage_aware_loso_panel_benchmark.csv":
                df = pd.read_csv(src)
                df = df[~df["panel_type"].str.startswith("random29")].copy()
                df.to_csv(dst, index=False)
            elif src_name.startswith("ecological_guild_"):
                df = pd.read_csv(src)
                for col in df.columns:
                    df[col] = df[col].replace(guild_value_mapping)
                    if is_string_dtype(df[col]) or is_object_dtype(df[col]):
                        df[col] = df[col].str.replace(
                            "no inflammation mechanism or functional inference",
                            "no mechanistic or functional inference",
                            regex=False,
                        )
                df.to_csv(dst, index=False)
            else:
                shutil.copy2(src, dst)
            copied.append(str(dst))
    return copied


def write_excel() -> list[str]:
    outputs = []
    loso_panels = pd.read_csv(ANALYSIS_DIR / "leakage_aware_loso_panel_benchmark.csv")
    loso_panels_xlsx = loso_panels[~loso_panels["panel_type"].str.startswith("random29")].copy()
    long_taxa = loso_panels_xlsx["selected_taxa"].astype(str).str.len() > 30000
    loso_panels_xlsx.loc[long_taxa, "selected_taxa"] = "ALL_RETAINED_FEATURES; see CSV file or Training_selection sheet for machine-readable taxon-level records"
    variance = pd.read_csv(ANALYSIS_DIR / "aitchison_variance_partitioning.csv")
    if "partial_or_marginal_r2" in variance.columns:
        variance = variance.rename(columns={"partial_or_marginal_r2": "term_deletion_partial_r2"})
    if "notes" in variance.columns:
        variance["notes"] = variance["notes"].astype(str).str.replace(
            "partial_or_marginal_r2 tests the term against a model containing the other listed terms.",
            "term_deletion_partial_r2 tests the term against a model containing the other listed terms.",
            regex=False,
        )

    loso_summary = pd.read_csv(ANALYSIS_DIR / "fixed_global_panel_loso_benchmark.csv")
    pairwise_summary = pd.read_csv(ANALYSIS_DIR / "fixed_global_panel_pairwise_benchmark.csv")
    fixed_combined = loso_summary.merge(pairwise_summary, on=["panel_type", "panel_size", "benchmark_label"], how="left")
    fixed_combined = fixed_combined.rename(
        columns={
            "median_auroc": "LOSO_median_AUROC",
            "minimum_auroc": "LOSO_minimum_AUROC",
            "maximum_auroc": "LOSO_maximum_AUROC",
            "auroc_iqr": "LOSO_AUROC_IQR",
            "median_average_precision": "LOSO_median_average_precision",
            "within_study_median_auroc": "pairwise_within_study_median_AUROC",
            "off_diagonal_median_auroc": "pairwise_off_diagonal_median_AUROC",
            "off_diagonal_auroc_iqr": "pairwise_off_diagonal_AUROC_IQR",
            "off_diagonal_minimum_auroc": "pairwise_off_diagonal_minimum_AUROC",
            "off_diagonal_maximum_auroc": "pairwise_off_diagonal_maximum_AUROC",
            "number_of_study_pairs": "number_of_off_diagonal_pairs",
        }
    )
    fixed_combined = fixed_combined[
        [
            "panel_type",
            "panel_size",
            "LOSO_median_AUROC",
            "LOSO_minimum_AUROC",
            "LOSO_maximum_AUROC",
            "LOSO_AUROC_IQR",
            "LOSO_median_average_precision",
            "pairwise_within_study_median_AUROC",
            "pairwise_off_diagonal_median_AUROC",
            "pairwise_off_diagonal_AUROC_IQR",
            "pairwise_off_diagonal_minimum_AUROC",
            "pairwise_off_diagonal_maximum_AUROC",
            "transferability_loss",
            "number_of_off_diagonal_pairs",
            "benchmark_label",
        ]
    ]

    random_summary = pd.read_csv(ANALYSIS_DIR / "random_panel_loso_benchmark_summary.csv")
    empirical_summary = random_summary.groupby("random_type").agg(
        n_held_out_studies=("held_out_study", "nunique"),
        median_random_median_AUROC=("median_auroc", "median"),
        median_training_stability_AUROC=("observed_training_stability_panel_auroc", "median"),
        median_empirical_p_training_stability_vs_random=("empirical_p_value_for_training_stability_vs_random", "median"),
        median_global_stable29_AUROC=("observed_global_stable29_auroc", "median"),
        median_empirical_p_global_stable29_vs_random=("empirical_p_value_for_stable29_vs_random", "median"),
    ).reset_index()

    def relabel_guilds(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        mapping = {
            "butyrate_SCFA_commensal_panel": "butyrate_SCFA_associated_commensal_panel",
            "inflammation_Bacteroides_Enterobacteriaceae_panel": "Bacteroides_Enterobacteriaceae_context_panel",
        }
        for col in out.columns:
            out[col] = out[col].replace(mapping)
            if is_string_dtype(out[col]) or is_object_dtype(out[col]):
                out[col] = out[col].str.replace(
                    "no inflammation mechanism or functional inference",
                    "no mechanistic or functional inference",
                    regex=False,
                )
        return out

    tables = {
        "Supplementary_Table_S6_aitchison_variance_partitioning.xlsx": {
            "Variance_partitioning": variance
        },
        "Supplementary_Table_S7_leakage_aware_LOSO_panel_summary.xlsx": {
            "LOSO_panels": loso_panels_xlsx,
            "Random_summary": random_summary,
            "Empirical_summary": empirical_summary,
            "Training_selection": pd.read_csv(ANALYSIS_DIR / "leakage_aware_training_panel_selection.csv"),
        },
        "Supplementary_Table_S8_fixed_global_panel_benchmarking.xlsx": {
            "Combined_summary": fixed_combined,
            "LOSO_summary": loso_summary,
            "Pairwise_summary": pairwise_summary,
            "LOSO_detail": pd.read_csv(ANALYSIS_DIR / "fixed_global_panel_loso_detail.csv"),
            "Pairwise_detail": pd.read_csv(ANALYSIS_DIR / "fixed_global_panel_pairwise_detail.csv"),
        },
        "Supplementary_Table_S9_ecological_guild_membership_and_score_results.xlsx": {
            "Membership": relabel_guilds(pd.read_csv(ANALYSIS_DIR / "ecological_guild_membership.csv")),
            "Score_association": relabel_guilds(pd.read_csv(ANALYSIS_DIR / "ecological_guild_scores.csv")),
            "Random_effects": relabel_guilds(pd.read_csv(ANALYSIS_DIR / "ecological_guild_random_effects.csv")),
            "Per_cohort_Hedges_g": relabel_guilds(pd.read_csv(ANALYSIS_DIR / "ecological_guild_per_cohort_hedges_g.csv")),
            "LOCO": relabel_guilds(pd.read_csv(ANALYSIS_DIR / "ecological_guild_loco_robustness.csv")),
            "LOSO_score_only": relabel_guilds(pd.read_csv(ANALYSIS_DIR / "ecological_guild_loso_score_only.csv")),
        },
        "Supplementary_Table_S10_panel_comparison_statistics.xlsx": {
            "Panel_comparisons": pd.read_csv(ANALYSIS_DIR / "panel_comparison_statistics.csv")
        },
        "Supplementary_Table_S11_claim_decision_table.xlsx": {
            "Claim_decisions": pd.read_csv(ANALYSIS_DIR / "claim_decision_table.csv")
        },
    }
    for old_name in [
        "Supplementary_Table_S7_leakage_aware_loso_panel_benchmarking.xlsx",
        "Supplementary_Table_S7_leakage_aware_loso_panel_benchmarking.csv",
    ]:
        for root in [BMC_SUPP, REPRO / "supplementary_tables"]:
            old_path = root / old_name
            if old_path.exists():
                old_path.unlink()
    for filename, sheets in tables.items():
        out = BMC_SUPP / filename
        with pd.ExcelWriter(out, engine="openpyxl") as writer:
            for sheet, df in sheets.items():
                df.to_excel(writer, sheet_name=sheet[:31], index=False)
        outputs.append(str(out))
        first = next(iter(sheets.values()))
        csv_out = BMC_SUPP / filename.replace(".xlsx", ".csv")
        first.to_csv(csv_out, index=False)
        outputs.append(str(csv_out))
    return outputs


def remove_submission_tiffs() -> list[str]:
    removed = []
    for pattern in ("*.tif", "*.tiff"):
        for path in BMC_FIG.glob(pattern):
            removed.append(str(path))
            path.unlink()
    return removed


def update_repro_package() -> None:
    if not REPRO.exists():
        return
    for sub in ["results", "supplementary_tables", "figures", "code/scripts"]:
        (REPRO / sub).mkdir(parents=True, exist_ok=True)
    for path in BMC_RESULTS.glob("*.csv"):
        if path.name in {
            "aitchison_variance_partitioning.csv",
            "leakage_aware_loso_panel_benchmark.csv",
            "random_panel_loso_benchmark_summary.csv",
            "fixed_global_panel_loso_benchmark.csv",
            "fixed_global_panel_pairwise_benchmark.csv",
            "ecological_guild_scores.csv",
            "ecological_guild_random_effects.csv",
            "ecological_guild_loco_robustness.csv",
            "ecological_guild_loso_score_only.csv",
            "ecological_guild_membership.csv",
            "panel_comparison_statistics.csv",
            "claim_decision_table.csv",
        }:
            shutil.copy2(path, REPRO / "results" / path.name)
    full_random = ANALYSIS_DIR / "leakage_aware_loso_panel_benchmark.csv"
    if full_random.exists():
        shutil.copy2(full_random, REPRO / "results" / "full_random_panel_loso_iterations.csv")
    for path in BMC_SUPP.glob("Supplementary_Table_S*_*.xlsx"):
        if any(tag in path.name for tag in ["S6_", "S7_", "S8_", "S9_", "S10_", "S11_"]):
            shutil.copy2(path, REPRO / "supplementary_tables" / path.name)
    for path in [BMC_FIG / "Figure_7.pdf", BMC_FIG / "Figure_7.png", BMC_FIG / "Figure_8.pdf", BMC_FIG / "Figure_8.png", BMC_FIG / "Figure_variance_partitioning.pdf", BMC_FIG / "Figure_variance_partitioning.png"]:
        if path.exists():
            shutil.copy2(path, REPRO / "figures" / path.name)
    script = RUN_DIR / "agents" / "05-analysis" / "workspace" / "scripts" / "build_leakage_aware_prioritization_analysis.py"
    if script.exists():
        shutil.copy2(script, REPRO / "code" / "scripts" / script.name)


def main() -> None:
    figures = {}
    figures["figure_1_overview"] = figure_1_overview()
    figures["variance"] = figure_variance()
    figures["leakage_panel"] = figure_leakage_panel()
    figures["guilds"] = figure_guilds()
    copied_results = copy_results()
    supp = write_excel()
    removed_tiffs = remove_submission_tiffs()
    update_repro_package()
    report = {"figures": figures, "results_copied": copied_results, "supplementary_tables": supp, "removed_tiffs": removed_tiffs}
    out_report = BMC / "reports" / "leakage_aware_revision_outputs_report.json"
    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
