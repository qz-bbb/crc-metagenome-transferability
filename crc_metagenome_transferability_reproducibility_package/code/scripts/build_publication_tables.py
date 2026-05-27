from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


RUN_DIR = Path(r"D:\1\projects\gut_microbiome_colorectal_cancer\runs\run_20260525_182306")
W5 = RUN_DIR / "agents" / "05-analysis" / "workspace"
DATA_DIR = Path(r"D:\1\Knowledge Base\true_raw\data\BIOC_curatedMetagenomicData_CRC")
OUT = W5 / "results" / "I_publication_tables"

METADATA = DATA_DIR / "curated_crc_case_control_sample_metadata.tsv"
ASSOC = W5 / "results" / "C" / "cohort_adjusted_crc_control_association.tsv"
STABILITY = W5 / "results" / "D" / "cross_cohort_stability.tsv"
COVARIATE = W5 / "results" / "G" / "baseline_vs_covariate_sensitivity.tsv"
TRANSPORT = W5 / "results" / "H" / "transportability_score.tsv"
LOSO_PERFORMANCE = W5 / "results" / "H" / "leave_one_study_elasticnet_performance.tsv"
LOSO_PREDICTIONS = W5 / "results" / "H" / "leave_one_study_elasticnet_predictions.tsv"


def read_tsv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"missing input table: {path}")
    return pd.read_csv(path, sep="\t")


def write_tsv(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=False)
    return path


def fmt_num(value: object, digits: int = 3) -> str:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return "NA"
    return f"{float(numeric):.{digits}f}"


def fmt_p(value: object) -> str:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return "NA"
    if numeric < 0.001:
        return f"{numeric:.2e}"
    return f"{numeric:.3f}"


def fmt_percent(value: object, digits: int = 1) -> str:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return "NA"
    return f"{100 * float(numeric):.{digits}f}%"


def median_iqr(series: pd.Series) -> str:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return "NA"
    q1 = values.quantile(0.25)
    med = values.quantile(0.50)
    q3 = values.quantile(0.75)
    return f"{med:.1f} ({q1:.1f}-{q3:.1f})"


def category_counts(series: pd.Series) -> str:
    cleaned = series.dropna().astype(str)
    cleaned = cleaned[cleaned.str.len() > 0]
    if cleaned.empty:
        return "NA"
    counts = cleaned.value_counts().sort_index()
    return "; ".join(f"{idx}={int(val)}" for idx, val in counts.items())


def unique_values(series: pd.Series) -> str:
    cleaned = sorted(v for v in series.dropna().astype(str).unique() if v)
    return "; ".join(cleaned) if cleaned else "NA"


def bootstrap_auc_ci(y_true: np.ndarray, scores: np.ndarray, n_boot: int = 2000, seed: int = 42) -> tuple[float, float, int]:
    rng = np.random.default_rng(seed)
    aucs: list[float] = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        sampled_y = y_true[idx]
        if len(np.unique(sampled_y)) < 2:
            continue
        aucs.append(float(roc_auc_score(sampled_y, scores[idx])))
    if not aucs:
        return (np.nan, np.nan, 0)
    return (float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5)), len(aucs))


def build_table1(metadata: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for cohort, group in metadata.groupby("study_name", sort=True):
        crc_n = int((group["study_condition"] == "CRC").sum())
        control_n = int((group["study_condition"] == "control").sum())
        rows.append(
            {
                "Cohort": cohort,
                "CRC": crc_n,
                "Control": control_n,
                "Total": crc_n + control_n,
                "Age, median (IQR)": median_iqr(group["age"]),
                "Age non-missing": int(group["age"].notna().sum()),
                "BMI, median (IQR)": median_iqr(group["BMI"]),
                "BMI non-missing": int(group["BMI"].notna().sum()),
                "Sex/gender metadata": category_counts(group["gender"]),
                "Country": unique_values(group["country"]),
                "Sequencing platform": unique_values(group["sequencing_platform"]),
                "Disease-stage non-missing": int(group["disease_stage"].notna().sum()),
            }
        )
    return pd.DataFrame(rows)


def build_table2() -> pd.DataFrame:
    assoc = read_tsv(ASSOC)
    stability = read_tsv(STABILITY)
    covariate = read_tsv(COVARIATE)
    transport = read_tsv(TRANSPORT)
    merged = (
        assoc.merge(
            stability[["taxon_long", "cohorts_same_direction", "cohorts_tested", "stability_fraction"]],
            on="taxon_long",
            how="left",
        )
        .merge(
            covariate[
                [
                    "taxon_long",
                    "qvalue_bh_coef_crc_clr_study_age_bmi_gender",
                    "direction_concordant",
                ]
            ],
            on="taxon_long",
            how="left",
        )
        .merge(
            transport[
                [
                    "taxon_long",
                    "mean_abs_elasticnet_coef",
                    "selected_fraction",
                    "transportability_rank",
                    "transportability_score",
                ]
            ],
            on="taxon_long",
            how="left",
        )
    )
    stable = merged[
        (pd.to_numeric(merged["qvalue_bh"], errors="coerce") < 0.10)
        & (pd.to_numeric(merged["stability_fraction"], errors="coerce") >= 0.60)
    ].copy()
    stable["ci_low"] = stable["coef_crc_clr_adjusted"] - 1.96 * stable["se_hc3"]
    stable["ci_high"] = stable["coef_crc_clr_adjusted"] + 1.96 * stable["se_hc3"]
    stable = stable.sort_values(["transportability_rank", "qvalue_bh", "species_short"], na_position="last")
    rows: list[dict[str, object]] = []
    for _, row in stable.iterrows():
        rows.append(
            {
                "Taxon": row["species_short"],
                "Direction": row["direction_adjusted"],
                "CLR coefficient (95% CI)": f"{fmt_num(row['coef_crc_clr_adjusted'])} ({fmt_num(row['ci_low'])}, {fmt_num(row['ci_high'])})",
                "P value": fmt_p(row["pvalue"]),
                "BH q value": fmt_p(row["qvalue_bh"]),
                "CRC prevalence": fmt_percent(row["prevalence_crc"]),
                "Control prevalence": fmt_percent(row["prevalence_control"]),
                "Cohort same-direction": f"{int(row['cohorts_same_direction'])}/{int(row['cohorts_tested'])}",
                "Covariate q value": fmt_p(row["qvalue_bh_coef_crc_clr_study_age_bmi_gender"]),
                "Elastic-net selected folds": fmt_percent(row["selected_fraction"], digits=0),
                "Transportability rank": int(row["transportability_rank"]),
            }
        )
    return pd.DataFrame(rows)


def build_table3() -> pd.DataFrame:
    performance = read_tsv(LOSO_PERFORMANCE)
    predictions = read_tsv(LOSO_PREDICTIONS)
    rows: list[dict[str, object]] = []
    for _, row in performance.sort_values("held_out_study").iterrows():
        cohort = row["held_out_study"]
        pred = predictions[predictions["held_out_study"] == cohort].copy()
        if pred.empty or row["status"] != "COMPUTED":
            ci_low, ci_high, valid_boot = np.nan, np.nan, 0
        else:
            y_true = (pred["true_condition"].astype(str) == "CRC").astype(int).to_numpy()
            scores = pd.to_numeric(pred["predicted_crc_probability"], errors="coerce").to_numpy(dtype=float)
            ci_low, ci_high, valid_boot = bootstrap_auc_ci(y_true, scores, seed=42 + len(rows))
        rows.append(
            {
                "Held-out cohort": cohort,
                "Training n": int(row["n_train"]),
                "Test n": int(row["n_test"]),
                "CRC/control in test": f"{int(row['n_test_crc'])}/{int(row['n_test_control'])}",
                "AUROC (95% bootstrap CI)": f"{fmt_num(row['auroc'])} ({fmt_num(ci_low)}, {fmt_num(ci_high)})",
                "Average precision": fmt_num(row["average_precision"]),
                "Bootstrap replicates used": int(valid_boot),
                "Status": row["status"],
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    metadata = read_tsv(METADATA)
    table1 = build_table1(metadata)
    table2 = build_table2()
    table3 = build_table3()

    paths = {
        "Table 1": write_tsv(table1, OUT / "Table1_cohort_sample_characteristics.tsv"),
        "Table 2": write_tsv(table2, OUT / "Table2_stable_candidate_taxa.tsv"),
        "Table 3": write_tsv(table3, OUT / "Table3_leave_one_study_out_performance.tsv"),
    }
    manifest = pd.DataFrame(
        [
            {
                "table": key,
                "path": str(path),
                "rows": int(pd.read_csv(path, sep="\t").shape[0]),
                "status": "READY",
            }
            for key, path in paths.items()
        ]
    )
    write_tsv(manifest, OUT / "publication_table_manifest.tsv")
    report = (
        "# Publication Tables Report\n\n"
        "- Table 1 summarizes cohort-level sample and metadata characteristics from the verified case/control metadata.\n"
        "- Table 2 lists all FDR<0.10 candidate taxa meeting >=60% cohort-direction stability, with HC3 coefficient intervals from the study-adjusted CLR model.\n"
        "- Table 3 reports leave-one-study-out elastic-net stress-test performance with deterministic 2,000-resample bootstrap AUROC intervals computed from held-out predictions.\n"
        "- These tables are descriptive/reporting artifacts and do not change the manuscript claim boundary.\n"
    )
    (OUT / "publication_tables_report.md").write_text(report, encoding="utf-8", newline="\n")
    print(manifest.to_json(orient="records", indent=2))


if __name__ == "__main__":
    main()
