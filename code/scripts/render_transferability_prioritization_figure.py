from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


RUN_DIR = Path(r"D:\1\projects\gut_microbiome_colorectal_cancer\runs\run_20260525_182306")
ANALYSIS = RUN_DIR / "agents" / "05-analysis" / "workspace" / "results" / "N_transferability_prioritization"
FIG_DIR = RUN_DIR / "agents" / "06-figures" / "workspace" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


COLORS = {
    "study": "#4C78A8",
    "crc": "#E45756",
    "cov": "#72B7B2",
    "top": "#D55E00",
    "stable": "#0072B2",
    "high": "#009E73",
    "all": "#666666",
    "random": "#B8B8B8",
}


def read(name: str) -> pd.DataFrame:
    return pd.read_csv(ANALYSIS / name, sep="\t")


def save_all(fig, stem: str) -> dict[str, str]:
    out = {}
    for ext in [".pdf", ".png", ".tiff"]:
        path = FIG_DIR / f"{stem}{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        out[ext] = str(path)
    return out


def fmt_panel(label: str) -> str:
    mapping = {
        "all_301_retained_species": "All 301",
        "top29_by_study_adjusted_bh_q": "Top 29 q",
        "stable29_transferability_aware": "Stable 29",
        "high18_consistency_candidates": "High 18",
        "random29_species_panels": "Random 29",
    }
    return mapping.get(label, label)


def main() -> None:
    variance = read("variance_partitioning.tsv")
    panel = read("panel_benchmark_summary.tsv")
    random = read("random29_repeat_summary.tsv")
    guild_assoc = read("guild_score_association.tsv")
    guild_meta = read("guild_score_random_effects.tsv")
    guild = guild_assoc.merge(guild_meta[["guild_id", "random_effect", "random_ci_low", "random_ci_high", "qvalue_bh_random_effect", "i2_percent"]], on="guild_id", how="left")

    fig, axes = plt.subplots(2, 2, figsize=(14.2, 9.2), constrained_layout=True)
    ax = axes[0, 0]
    order = ["study_label", "CRC_status", "age", "gender", "BMI"]
    v = variance.set_index("term").loc[order].reset_index()
    colors = [COLORS["study"], COLORS["crc"], COLORS["cov"], COLORS["cov"], COLORS["cov"]]
    ax.barh(np.arange(len(v)), v["partial_r2_percent"], color=colors, edgecolor="white")
    ax.set_yticks(np.arange(len(v)))
    ax.set_yticklabels(["Study label", "CRC status", "Age", "Sex/gender", "BMI"])
    ax.invert_yaxis()
    ax.set_xlabel("Partial variation explained in retained CLR matrix (%)")
    ax.set_title("A. Variance partitioning")
    for i, val in enumerate(v["partial_r2_percent"]):
        ax.text(val + 0.08, i, f"{val:.2f}%", va="center", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[0, 1]
    fixed = panel[panel["panel_type"].isin(["fixed", "random_aggregate"])].copy()
    fixed["label"] = fixed["panel_id"].map(fmt_panel)
    x = np.arange(len(fixed))
    ax.plot(x, fixed["loso_median_auroc"], marker="o", color="#222222", label="LOSO median AUROC")
    ax.plot(x, fixed["pairwise_offdiag_median_auroc"], marker="s", color="#0072B2", label="Pairwise off-diagonal median AUROC")
    ax.fill_between(
        [x.min() - 0.3, x.max() + 0.3],
        [0.5, 0.5],
        [0.5, 0.5],
        color="none",
    )
    ax.axhline(0.5, color="#888888", lw=1, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels(fixed["label"], rotation=25, ha="right")
    ax.set_ylim(0.40, 0.90)
    ax.set_ylabel("AUROC")
    ax.set_title("B. Candidate panel cross-cohort separability")
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1, 0]
    random_values = [
        random["loso_median_auroc"].dropna(),
        random["pairwise_offdiag_median_auroc"].dropna(),
        random["transferability_loss"].dropna(),
    ]
    bp = ax.boxplot(random_values, positions=[0, 1, 2], widths=0.55, patch_artist=True, showfliers=False)
    for patch in bp["boxes"]:
        patch.set_facecolor(COLORS["random"])
        patch.set_edgecolor("#666666")
    for median in bp["medians"]:
        median.set_color("#111111")
    metric_cols = ["loso_median_auroc", "pairwise_offdiag_median_auroc", "transferability_loss"]
    offsets = {"all_301_retained_species": -0.15, "top29_by_study_adjusted_bh_q": -0.05, "stable29_transferability_aware": 0.05, "high18_consistency_candidates": 0.15}
    point_colors = {
        "all_301_retained_species": COLORS["all"],
        "top29_by_study_adjusted_bh_q": COLORS["top"],
        "stable29_transferability_aware": COLORS["stable"],
        "high18_consistency_candidates": COLORS["high"],
    }
    for panel_id, off in offsets.items():
        row = panel.loc[panel["panel_id"] == panel_id].iloc[0]
        for pos, col in enumerate(metric_cols):
            ax.scatter(pos + off, row[col], s=36, color=point_colors[panel_id], edgecolor="white", linewidth=0.5, zorder=3)
    legend_handles = [
        Line2D([0], [0], marker="s", color="none", markerfacecolor=COLORS["random"], markeredgecolor="#666666", label="Random 29 distribution"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["all"], markeredgecolor="white", label="All 301"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["top"], markeredgecolor="white", label="Top 29 q"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["stable"], markeredgecolor="white", label="Stable 29"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["high"], markeredgecolor="white", label="High 18"),
    ]
    ax.legend(handles=legend_handles, frameon=False, fontsize=8, loc="upper right")
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["LOSO median", "Pairwise median", "Transferability loss"], rotation=15, ha="right")
    ax.set_ylabel("AUROC or AUROC difference")
    ax.set_title("C. Fixed panels against 500 random 29-species panels")
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1, 1]
    guild_labels = {
        "butyrate_scfa_commensal": "Butyrate/SCFA\ncommensal",
        "oral_pathobiont_associated": "Oral/pathobiont\nassociated",
        "bacteroides_enterobacteriaceae_context": "Bacteroides/\nEnterobacteriaceae",
    }
    guild = guild.sort_values("random_effect")
    y = np.arange(len(guild))
    x = guild["random_effect"]
    xerr = np.vstack([x - guild["random_ci_low"], guild["random_ci_high"] - x])
    ax.errorbar(x, y, xerr=xerr, fmt="o", color="#222222", ecolor="#555555", capsize=3)
    ax.axvline(0, color="#888888", lw=1, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels([guild_labels.get(g, g) for g in guild["guild_id"]])
    ax.set_xlabel("Random-effects CRC coefficient for guild score")
    ax.set_title("D. Ecological guild score effects")
    for i, row in guild.iterrows():
        ax.text(row["random_ci_high"] + 0.03, list(guild.index).index(i), f"q={row['qvalue_bh_random_effect']:.2g}; I²={row['i2_percent']:.1f}%", va="center", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)

    outputs = save_all(fig, "Fig7_transferability_prioritization")
    plt.close(fig)
    report = FIG_DIR.parent / "reports" / "Fig7_transferability_prioritization_qc.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(pd.Series(outputs).to_json(indent=2), encoding="utf-8")
    print(outputs)


if __name__ == "__main__":
    main()
