"""Render existing B/C estimates only; never refit, resample or pool cohorts."""
from pathlib import Path
import csv
import hashlib
import json
import textwrap
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from PIL import Image, ImageOps
import fitz

import argparse
from paths import output_directory
BASE = B = C = None
CAPTION = Path(__file__).with_name("Figure_8_caption.md")
STEM = "Figure_8_definition_prediction"
BLUE, ORANGE, INK = "#0072B2", "#D55E00", "#303030"
PANELS = ["ARP", "DCP_raw_median", "DCP_raw_mean", "DCP_clr_median", "DCP_clr_mean",
          "ICP_raw_median", "ICP_raw_mean", "ICP_clr_median", "ICP_clr_mean"]
LABELS = ["ARP", "DCP raw median", "DCP raw mean", "DCP CLR median", "DCP CLR mean",
          "ICP raw median", "ICP raw mean", "ICP CLR median", "ICP CLR mean"]
LEARNERS = ["elastic_net", "random_forest"]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    return [dict(row, source_path=str(path), source_line=str(i + 2)) for i, row in enumerate(rows)]


def export(name, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (BASE / name).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    with (BASE / name).open(encoding="utf-8", newline="") as stream:
        replay = list(csv.DictReader(stream, delimiter="\t"))
    for original, returned in zip(rows, replay):
        assert all(returned[k] == v for k, v in original.items()), name
    assert len(rows) == len(replay)


def one(rows, **filters):
    selected = [row for row in rows if all(row.get(k) == v for k, v in filters.items())]
    assert len(selected) == 1, (filters, len(selected))
    return selected[0]


def interval(ax, row, y, color, marker, opened=False, level="95", dashed=False, value="estimate"):
    estimate = float(row[value])
    low, high = float(row[f"ci{level}_low"]), float(row[f"ci{level}_high"])
    assert low <= estimate <= high
    ax.plot([low, high], [y, y], color=color, linewidth=1.0,
            linestyle="--" if dashed else "-", zorder=2)
    ax.plot([low, low], [y - .045, y + .045], color=color, linewidth=.8)
    ax.plot([high, high], [y - .045, y + .045], color=color, linewidth=.8)
    ax.plot(estimate, y, marker=marker, markersize=4.4, markeredgewidth=.9,
            color=color, markerfacecolor="white" if opened else color, zorder=3)


def style(ax, n, labels, limits, ticks, xlabel, zero=False):
    ax.set_ylim(n - .45, -.55)
    ax.set_yticks(range(n), labels)
    ax.set_xlim(*limits)
    ax.set_xticks(ticks)
    ax.tick_params(axis="y", length=0, pad=5)
    ax.tick_params(axis="x", length=3, pad=3)
    ax.set_xlabel(xlabel, fontsize=8.3, labelpad=6)
    ax.grid(axis="x", color="#E3E3E3", linewidth=.5, zorder=0)
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)
    if zero:
        ax.axvline(0, color=INK, linewidth=.8, zorder=1)


def main():
    global BASE, B, C
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--internal-summary',required=True,type=Path)
    parser.add_argument('--external-summary',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    B=args.internal_summary.resolve(); C=args.external_summary.resolve()
    paths = [B / "macro_metrics.tsv", B / "paired_contrasts.tsv", C / "primary3_macro_metrics.tsv",
             C / "paired_contrasts_all_scopes.tsv", C / "all6_cohort_metrics.tsv"]
    if not all(p.is_file() for p in paths):
        raise FileNotFoundError("All five summary source tables are required")
    BASE=output_directory(args.output,[B,C])
    snapshots = [{"path": str(p), "sha256": sha(p)} for p in paths]
    internal, bc, external, cc, cohorts = [read(p) for p in paths]
    macro = []
    for rows, context, scope in [(internal, "nested_internal", "MACRO"),
                                 (external, "external_primary3", "PRIMARY3")]:
        macro.extend([dict(r, context=context, scope=scope, panel_size="18",
                           aggregation="equal_study_macro", ci_level="95",
                           ci_method="fixed_model_conditional_bootstrap",
                           inference_status="post_outcome_sensitivity") for r in rows])
    assert len(macro) == 72
    assert {r["panel"] for r in macro} == set(PANELS)
    assert all(r["n_studies"] == ("9" if r["scope"] == "MACRO" else "3") for r in macro)
    for scope in ["MACRO", "PRIMARY3"]:
        for learner in LEARNERS:
            for metric in ["AUROC", "AP"]:
                for first, second in [("ARP", "DCP_clr_mean"), ("ICP_raw_mean", "ICP_clr_mean")]:
                    r1 = one(macro, scope=scope, panel=first, classifier=learner, metric=metric)
                    r2 = one(macro, scope=scope, panel=second, classifier=learner, metric=metric)
                    assert all(r1[k] == r2[k] for k in ["mean", "median", "minimum", "ci95_low", "ci95_high"])
    export("Figure_8_macro_source.tsv", macro)
    contrast_sources = []
    for rows, context, scope, n in [(bc, "nested_internal", "MACRO", "9"),
                                     (cc, "external_primary3", "PRIMARY3", "3")]:
        for r in rows:
            if r["scope"] == scope:
                contrast_sources.append(dict(r, source_panel=r["panel"],
                    panel=r["panel"] or r["contrast"].partition("_minus_")[0],
                    context=context, n_studies=n, panel_size="18",
                    aggregation="equal_study_macro", ci_method="paired_fixed_model_conditional_bootstrap"))
    export("Figure_8_contrast_source.tsv", contrast_sources)
    assert len(cohorts) == 108 and len({r["study_name"] for r in cohorts}) == 6
    assert all(r["k"] == "18" for r in cohorts)
    cohort_source = [dict(r, context="external_individual_cohort", aggregation="individual_cohort",
                         n_studies="1", ci_method="fixed_model_conditional_bootstrap",
                         inference_status="post_outcome_sensitivity") for r in cohorts]
    export("Figure_8_all6_external_source.tsv", cohort_source)
    cohort_long = []
    for r in cohort_source:
        for metric, key in [("AUROC", "auroc"), ("AP", "ap")]:
            cohort_long.append(dict(r, metric=metric, estimate=r[key],
                ci95_low=r[key + "_ci95_low"], ci95_high=r[key + "_ci95_high"], ci_level="95"))
    export("Figure_8_all6_external_metric_source.tsv", cohort_long)
    plotted = []
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.3, "text.color": INK,
        "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
        "axes.linewidth": .6, "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig = plt.figure(figsize=(180 / 25.4, 205 / 25.4))
    fig.text(.025, .984, "Prediction across direction definitions", fontsize=11,
             fontweight="bold", va="top")
    fig.text(.025, .958, "18 taxa per panel · equal-study macro · post-outcome sensitivity", fontsize=8.3, va="top")
    for x, scope, letter, title, count in [(.26, "MACRO", "A", "Nested internal", "9 studies"),
                                         (.66, "PRIMARY3", "B", "External primary cohorts", "3 cohorts")]:
        ax = fig.add_axes([x, .583, .31, .300])
        for i, panel in enumerate(PANELS):
            for learner, offset, color, mark, opened in [(LEARNERS[0], -.14, BLUE, "o", False),
                                                       (LEARNERS[1], .14, ORANGE, "^", True)]:
                row = one(macro, scope=scope, panel=panel, classifier=learner, metric="AUROC")
                interval(ax, row, i + offset, color, mark, opened, value="mean")
                plotted.append(dict(row, figure_panel=letter, plotted_ci_level="95"))
        style(ax, 9, LABELS if letter == "A" else [""] * 9, (.62, .91), [.65, .75, .85], "Macro AUROC")
        fig.text(x, .917, f"{letter}  {title}", fontsize=9, fontweight="bold")
        fig.text(x, .896, f"{count} · 95% conditional intervals", fontsize=8)
    learner_handles = [Line2D([], [], marker="o", color=BLUE, lw=1, label="EN"),
                       Line2D([], [], marker="^", markerfacecolor="white", color=ORANGE, lw=1, label="RF")]
    fig.legend(handles=learner_handles, loc="upper right", bbox_to_anchor=(.98, .956),
               fontsize=8, frameon=False, ncol=2, handlelength=1.5, columnspacing=1.0)

    axc = fig.add_axes([.26, .185, .31, .285])
    focal = [("ICP_raw_mean_minus_raw_median", "Raw mean"),
             ("ICP_clr_median_minus_raw_median", "CLR median")]
    clabels = []
    for i, (contrast, label) in enumerate(focal):
        for j, (scope, scope_label, color, mark, opened) in enumerate([
                ("MACRO", "internal", BLUE, "o", False), ("PRIMARY3", "external", ORANGE, "s", True)]):
            row = one(contrast_sources, scope=scope, contrast=contrast, classifier="elastic_net", metric="AUROC")
            interval(axc, row, i * 2 + j, color, mark, opened, level="975", dashed=opened)
            plotted.append(dict(row, figure_panel="C", plotted_ci_level="97.5"))
            clabels.append(f"{label}\n{scope_label}")
    style(axc, 4, clabels, (-.03, .19), [0, .05, .10, .15], "Δ macro AUROC", zero=True)
    fig.text(.025, .505, "C  EN: ICP change from raw median", fontsize=9, fontweight="bold")
    fig.text(.025, .483, "Two focal contrasts · nominal 97.5% intervals", fontsize=8)

    axd = fig.add_axes([.75, .185, .22, .285])
    dlabels = []
    for i, panel in enumerate(["ICP_raw_median", "ICP_raw_mean", "ICP_clr_median", "ICP_clr_mean"]):
        for j, learner in enumerate(LEARNERS):
            y = i * 2 + j
            dlabels.append(f"{panel.removeprefix('ICP_').replace('clr', 'CLR').replace('_', ' ')} / {'EN' if j == 0 else 'RF'}")
            for scope, offset, color, mark, opened in [("MACRO", -.13, BLUE, "o", False),
                                                     ("PRIMARY3", .13, ORANGE, "s", True)]:
                row = one(contrast_sources, scope=scope, contrast=panel + "_minus_ARP", classifier=learner, metric="AUROC")
                interval(axd, row, y + offset, color, mark, opened, dashed=opened)
                plotted.append(dict(row, figure_panel="D", plotted_ci_level="95"))
    style(axd, 8, dlabels, (-.21, .05), [-.20, -.10, 0], "Δ macro AUROC", zero=True)
    fig.text(.63, .505, "D  ICP − ARP", fontsize=9, fontweight="bold")
    fig.text(.63, .483, "Paired 95% conditional intervals", fontsize=8)
    scope_handles = [Line2D([], [], marker="o", color=BLUE, lw=1, label="Internal"),
                     Line2D([], [], marker="s", markerfacecolor="white", color=ORANGE, ls="--", lw=1, label="External")]
    fig.legend(handles=scope_handles, loc="lower left", bbox_to_anchor=(.02, .125),
               fontsize=8, frameon=False, ncol=2, handlelength=1.7)
    notes = ["C external raw-mean lower bound: −0.00002164 (includes zero).",
             "ARP = DCP CLR mean; ICP raw mean = ICP CLR mean: duplicate panels, not replication.",
             "Conditional uncertainty omits refitting and new-population uncertainty.",
             "Neither ICP superiority over ARP nor equivalence is established."]
    for y, note in zip([.110, .088, .066, .044], notes):
        fig.text(.025, y, note, fontsize=8)
    for ext in ["pdf", "svg", "png"]:
        fig.savefig(BASE / f"{STEM}.{ext}", dpi=300)
    plt.close(fig)
    assert len(plotted) == 56
    export("Figure_8_plotted_source_rows.tsv", plotted)
    near_zero = one(contrast_sources, scope="PRIMARY3", contrast="ICP_raw_mean_minus_raw_median",
                    classifier="elastic_net", metric="AUROC")
    assert float(near_zero["ci975_low"]) < 0 < float(near_zero["ci975_high"])
    qa = BASE / "qa"
    qa.mkdir(exist_ok=True)
    with fitz.open(BASE / f"{STEM}.pdf") as doc:
        page = doc[0]
        spans = [s for block in page.get_text("dict")["blocks"] if "lines" in block
                 for line in block["lines"] for s in line["spans"] if s["text"].strip()]
        minimum = min(s["size"] for s in spans)
        outside = [s["text"] for s in spans if s["bbox"][0] < 0 or s["bbox"][1] < 0
                   or s["bbox"][2] > page.rect.width or s["bbox"][3] > page.rect.height]
        assert minimum >= 7.999 and not outside, (minimum, outside)
        page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False).save(qa / "Figure_8_publication_width.png")
        dimensions = [page.rect.width * 25.4 / 72, page.rect.height * 25.4 / 72]
    im = Image.open(BASE / f"{STEM}.png")
    im.resize((720, round(im.height * 720 / im.width))).save(qa / "Figure_8_120mm_preview.png")
    ImageOps.grayscale(im).resize((1080, round(im.height * 1080 / im.width))).save(qa / "Figure_8_grayscale.png")

    # Caption-adjacent review page; the standalone journal figure remains 180 mm wide.
    review = fitz.open()
    page = review.new_page(width=210 / 25.4 * 72, height=297 / 25.4 * 72)
    with fitz.open(BASE / f"{STEM}.pdf") as src:
        page.show_pdf_page(fitz.Rect(15/25.4*72, 8/25.4*72, 195/25.4*72, 213/25.4*72), src, 0)
    caption = CAPTION.read_text(encoding="utf-8").strip()
    overflow = page.insert_textbox(fitz.Rect(15/25.4*72, 216/25.4*72, 195/25.4*72, 291/25.4*72),
                                  caption.replace("−", "-"), fontsize=8, fontname="helv", lineheight=1.15)
    assert overflow >= 0, overflow
    review.save(BASE / "Figure_8_review_with_caption.pdf")
    review.close()
    assert snapshots == [{"path": str(p), "sha256": sha(p)} for p in paths], "Source mutation"
    receipt = {"status": "PASS", "operation": "existing summary rendering; no new analysis",
        "sources": snapshots, "figure_size_mm": dimensions, "minimum_font_pt": minimum,
        "text_outside_page": outside, "macro_source_rows": len(macro),
        "contrast_source_rows": len(contrast_sources), "plotted_rows": len(plotted),
        "all6_individual_cohort_model_rows": len(cohorts),
        "external_cohorts": list(dict.fromkeys(r["study_name"] for r in cohorts)),
        "source_export_roundtrip": "exact string equality", "duplicate_macro_metrics_identical": True,
        "all6_long_metric_rows": len(cohort_long), "negative_lower_bound": near_zero["ci975_low"],
        "palette": [BLUE, ORANGE, INK], "sources_unchanged": True,
        "python": sys.version, "executable": sys.executable,
        "pdf_sha256": sha(BASE / f"{STEM}.pdf")}
    (qa / "figure8_validation.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
