from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd


RUN_DIR = Path(r"D:\1\projects\gut_microbiome_colorectal_cancer\runs\run_20260525_182306")
ANALYSIS = RUN_DIR / "agents" / "05-analysis" / "workspace"
FIGURES = RUN_DIR / "agents" / "06-figures" / "workspace" / "figures"
BMC = RUN_DIR / "BMC_submission"
N_RESULTS = ANALYSIS / "results" / "N_transferability_prioritization"
SUPP = BMC / "supplementary_tables"
RESULTS = BMC / "results"
BMC_FIGS = BMC / "figures"
CODE = BMC / "code" / "scripts"
REPRO = BMC / "crc_metagenome_transferability_reproducibility_package"


def read(name: str) -> pd.DataFrame:
    return pd.read_csv(N_RESULTS / name, sep="\t")


def write_xlsx(path: Path, sheets: dict[str, pd.DataFrame]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for sheet, df in sheets.items():
            df.to_excel(writer, sheet_name=sheet[:31], index=False)
            ws = writer.sheets[sheet[:31]]
            ws.freeze_panes = "A2"
            for column_cells in ws.columns:
                values = [str(cell.value) if cell.value is not None else "" for cell in column_cells[:200]]
                ws.column_dimensions[column_cells[0].column_letter].width = min(max([len(v) for v in values] + [10]) + 2, 70)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def main() -> None:
    SUPP.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    BMC_FIGS.mkdir(parents=True, exist_ok=True)
    CODE.mkdir(parents=True, exist_ok=True)

    variance = read("variance_partitioning.tsv")
    panel_summary = read("panel_benchmark_summary.tsv")
    random_summary = read("random29_repeat_summary.tsv")
    panel_loso = read("panel_benchmark_loso_folds.tsv")
    panel_pairwise = read("panel_benchmark_pairwise.tsv")
    candidate_membership = read("candidate_panel_membership.tsv")
    guild_membership = read("guild_panel_membership.tsv")
    guild_assoc = read("guild_score_association.tsv")
    guild_meta = read("guild_score_random_effects.tsv")
    guild_loco = read("guild_score_loco_summary.tsv")
    guild_transfer = read("guild_score_transfer_summary.tsv")
    guild_per_cohort = read("guild_score_per_cohort_effects.tsv")

    write_xlsx(SUPP / "Supplementary_Table_S6_variance_partitioning.xlsx", {"S6_variance": variance})
    write_csv(variance, SUPP / "Supplementary_Table_S6_variance_partitioning.csv")

    write_xlsx(
        SUPP / "Supplementary_Table_S7_panel_benchmarking.xlsx",
        {
            "Summary": panel_summary,
            "Random29_repeats": random_summary,
            "LOSO_folds": panel_loso,
            "Pairwise_transfer": panel_pairwise,
        },
    )
    write_csv(panel_summary, SUPP / "Supplementary_Table_S7_panel_benchmarking_summary.csv")

    write_xlsx(SUPP / "Supplementary_Table_S8_candidate_panel_membership.xlsx", {"S8_panel_membership": candidate_membership})
    write_csv(candidate_membership, SUPP / "Supplementary_Table_S8_candidate_panel_membership.csv")

    write_xlsx(
        SUPP / "Supplementary_Table_S9_ecological_guild_score_results.xlsx",
        {
            "Association": guild_assoc,
            "Random_effects": guild_meta,
            "LOCO_summary": guild_loco,
            "Transfer_summary": guild_transfer,
            "Per_cohort_effects": guild_per_cohort,
        },
    )
    write_csv(guild_assoc, SUPP / "Supplementary_Table_S9_ecological_guild_score_association.csv")

    write_xlsx(SUPP / "Supplementary_Table_S10_ecological_guild_panel_membership.xlsx", {"S10_guild_membership": guild_membership})
    write_csv(guild_membership, SUPP / "Supplementary_Table_S10_ecological_guild_panel_membership.csv")

    result_map = {
        "variance_partitioning_results.csv": variance,
        "panel_benchmark_summary.csv": panel_summary,
        "panel_benchmark_loso_folds.csv": panel_loso,
        "panel_benchmark_pairwise.csv": panel_pairwise,
        "random29_repeat_summary.csv": random_summary,
        "candidate_panel_membership.csv": candidate_membership,
        "guild_score_association.csv": guild_assoc,
        "guild_score_random_effects.csv": guild_meta,
        "guild_score_loco_summary.csv": guild_loco,
        "guild_score_transfer_summary.csv": guild_transfer,
        "guild_panel_membership.csv": guild_membership,
    }
    for name, df in result_map.items():
        write_csv(df, RESULTS / name)

    for ext in [".pdf", ".png", ".tiff"]:
        src = FIGURES / f"Fig7_transferability_prioritization{ext}"
        if src.exists():
            shutil.copy2(src, BMC_FIGS / f"Figure_7{ext}")

    scripts = [
        ANALYSIS / "scripts" / "build_transferability_prioritization_analysis.py",
        RUN_DIR / "agents" / "06-figures" / "workspace" / "figure_scripts" / "render_transferability_prioritization_figure.py",
        Path(__file__),
    ]
    for script in scripts:
        if script.exists():
            shutil.copy2(script, CODE / script.name)

    if REPRO.exists():
        for sub in ["code", "results", "supplementary_tables", "figures"]:
            (REPRO / sub).mkdir(parents=True, exist_ok=True)
        for script in scripts:
            if script.exists():
                shutil.copy2(script, REPRO / "code" / "scripts" / script.name)
        for name, df in result_map.items():
            write_csv(df, REPRO / "results" / name)
        for path in SUPP.glob("Supplementary_Table_S[6-9]*"):
            shutil.copy2(path, REPRO / "supplementary_tables" / path.name)
        for path in SUPP.glob("Supplementary_Table_S10*"):
            shutil.copy2(path, REPRO / "supplementary_tables" / path.name)
        for ext in [".pdf", ".png"]:
            src = BMC_FIGS / f"Figure_7{ext}"
            if src.exists():
                shutil.copy2(src, REPRO / "figures" / f"Figure_7{ext}")
        manifest_path = REPRO / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest.setdefault("notes", []).append("Transferability-aware candidate prioritization outputs were added in the revision package.")
            outputs = manifest.setdefault("main_outputs", [])
            for name in ["variance_partitioning_results.csv", "panel_benchmark_summary.csv", "guild_score_association.csv", "Figure_7.pdf", "Figure_7.png"]:
                if name not in outputs:
                    outputs.append(name)
            manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    report = {
        "supplementary_tables_added": [
            "Supplementary_Table_S6_variance_partitioning",
            "Supplementary_Table_S7_panel_benchmarking",
            "Supplementary_Table_S8_candidate_panel_membership",
            "Supplementary_Table_S9_ecological_guild_score_results",
            "Supplementary_Table_S10_ecological_guild_panel_membership",
        ],
        "figure_added": str(BMC_FIGS / "Figure_7.pdf"),
        "results_added": sorted(result_map),
    }
    out = BMC / "reports" / "transferability_revision_outputs_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
