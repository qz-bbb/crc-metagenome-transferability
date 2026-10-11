from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

try:
    import resource  # type: ignore
except ImportError:  # Windows
    resource = None


RUN_DIR = Path('UNCONFIGURED_OUTPUT')
OUT_DIR = RUN_DIR / "02_analysis" / "rf_icp_repair"
OLD_RUN = Path('UNCONFIGURED_LEGACY')
OLD_W5 = OLD_RUN / "agents" / "05-analysis" / "workspace"
DATA_DIR = Path('UNCONFIGURED_INPUT')
META_PATH = DATA_DIR / "curated_crc_case_control_sample_metadata.tsv"
CLR_PATH = OLD_W5 / "processed_data" / "B" / "filtered_species_clr.tsv.gz"
INTERPRETATION_PATH = OLD_W5 / "results" / "M_robustness_interpretation" / "stable_candidate_interpretation.tsv"

MODEL_SEED = 20260817
BOOTSTRAP_REPLICATES = 2000
THRESHOLDS = (0.60, 0.70, 0.80)


def rss_mb() -> float | None:
    if resource is not None:
        if platform.system() == "Windows":
            return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / (1024 * 1024)
        return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0
    try:
        import psutil
        return float(psutil.Process(os.getpid()).memory_info().rss) / (1024 * 1024)
    except Exception:
        if platform.system() == "Windows":
            try:
                import ctypes
                from ctypes import wintypes
                class Counters(ctypes.Structure):
                    _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD), ("PeakWorkingSetSize", ctypes.c_size_t),
                                ("WorkingSetSize", ctypes.c_size_t), ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t), ("PagefileUsage", ctypes.c_size_t),
                                ("PeakPagefileUsage", ctypes.c_size_t)]
                c = Counters(); c.cb = ctypes.sizeof(Counters)
                ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(c), c.cb)
                value = float(c.PeakWorkingSetSize) / (1024 * 1024)
                return value if value > 0 else None
            except Exception:
                pass
        return None


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def bh(values: pd.Series) -> np.ndarray:
    values = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    out = np.full(values.shape, np.nan)
    valid = np.isfinite(values)
    if valid.any():
        order = np.argsort(values[valid])
        ranked = values[valid][order]
        adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
        adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
        tmp = np.empty_like(adjusted)
        tmp[order] = np.clip(adjusted, 0, 1)
        out[valid] = tmp
    return out


def association_screen(clr: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    y = (meta["study_condition"].astype(str) == "CRC").astype(float)
    design = pd.concat(
        [pd.DataFrame({"CRC_status": y}, index=meta.index),
         pd.get_dummies(meta["study_name"].astype(str), prefix="study", drop_first=True, dtype=float)],
        axis=1,
    )
    design = sm.add_constant(design, has_constant="add").astype(float)
    rows = []
    for taxon in clr.columns:
        fit = sm.OLS(clr[taxon].astype(float), design).fit(cov_type="HC3")
        rows.append({"taxon": taxon, "coef_crc": float(fit.params["CRC_status"]),
                     "se_hc3": float(fit.bse["CRC_status"]), "p_value": float(fit.pvalues["CRC_status"])})
    out = pd.DataFrame(rows)
    out["q_value_bh"] = bh(out["p_value"])
    out["direction"] = np.where(out["coef_crc"] >= 0, "CRC_higher", "control_higher")
    support = direction_support(clr, meta, out.set_index("taxon")["coef_crc"])
    return out.merge(support, on="taxon", how="left").sort_values(["q_value_bh", "p_value", "taxon"])


def direction_support(clr: pd.DataFrame, meta: pd.DataFrame, coef: pd.Series) -> pd.DataFrame:
    rows = []
    for taxon in clr.columns:
        global_sign = np.sign(float(coef.loc[taxon]))
        eligible = 0
        same = 0
        diffs: list[float] = []
        for study in sorted(meta["study_name"].astype(str).unique()):
            idx = meta["study_name"].astype(str) == study
            sub = meta.loc[idx]
            if sub["study_condition"].astype(str).nunique() < 2:
                continue
            crc = clr.loc[sub.index[sub["study_condition"].astype(str) == "CRC"], taxon]
            ctrl = clr.loc[sub.index[sub["study_condition"].astype(str) != "CRC"], taxon]
            diff = float(crc.mean() - ctrl.mean())
            diffs.append(diff)
            dsign = np.sign(diff)
            if dsign == 0 or global_sign == 0:
                continue
            eligible += 1
            same += int(dsign == global_sign)
        rows.append({"taxon": taxon, "eligible_studies": eligible,
                     "same_direction_studies": same,
                     "dcp": same / eligible if eligible else np.nan,
                     "mean_study_clr_crc_minus_control": float(np.mean(diffs)) if diffs else np.nan})
    return pd.DataFrame(rows)


def select_panel(assoc: pd.DataFrame, threshold: float, cap: int | None = None) -> list[str]:
    keep = assoc[(assoc["q_value_bh"] < 0.10) & (assoc["dcp"] >= threshold)]
    selected = keep.sort_values(["q_value_bh", "p_value", "taxon"])["taxon"].tolist()
    return selected[:cap] if cap is not None else selected


def fit_elastic_net(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray) -> np.ndarray:
    scaler = StandardScaler()
    model = LogisticRegression(penalty="elasticnet", solver="saga", l1_ratio=0.5, C=0.5,
                               class_weight="balanced", max_iter=2000, random_state=42, n_jobs=1)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        model.fit(scaler.fit_transform(x_train), y_train)
    return model.predict_proba(scaler.transform(x_test))[:, 1]


def fit_rf(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray) -> np.ndarray:
    model = RandomForestClassifier(n_estimators=500, max_features="sqrt", max_depth=None,
                                   min_samples_leaf=2, bootstrap=True,
                                   class_weight="balanced_subsample", random_state=MODEL_SEED, n_jobs=1)
    model.fit(x_train, y_train)
    return model.predict_proba(x_test)[:, 1]


def score(y: np.ndarray, prob: np.ndarray) -> dict[str, float]:
    pred = (prob >= 0.5).astype(int)
    return {"auroc": float(roc_auc_score(y, prob)),
            "average_precision": float(average_precision_score(y, prob)),
            "balanced_accuracy_at_0.5": float(balanced_accuracy_score(y, pred))}


def bootstrap_metric_ci(y: np.ndarray, prob: np.ndarray, seed: int) -> dict[str, float]:
    """Stratified percentile bootstrap using sklearn metrics per resample."""
    rng = np.random.default_rng(seed)
    c0 = np.flatnonzero(y == 0)
    c1 = np.flatnonzero(y == 1)
    auc, ap = [], []
    for _ in range(BOOTSTRAP_REPLICATES):
        idx = np.concatenate([rng.choice(c0, len(c0), replace=True), rng.choice(c1, len(c1), replace=True)])
        yy, pp = y[idx], prob[idx]
        auc.append(roc_auc_score(yy, pp))
        ap.append(average_precision_score(yy, pp))
    return {"auroc_ci_low": float(np.quantile(auc, 0.025)), "auroc_ci_high": float(np.quantile(auc, 0.975)),
            "average_precision_ci_low": float(np.quantile(ap, 0.025)), "average_precision_ci_high": float(np.quantile(ap, 0.975))}


def derived_seed(*parts: str) -> int:
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).digest()
    return MODEL_SEED + int.from_bytes(digest[:4], "little")


def add_bootstrap_and_pairwise(metrics_df: pd.DataFrame, pred_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ci_rows = []
    for key, group in metrics_df.iterrows():
        p = pred_df[(pred_df.fold_index == group.fold_index) & (pred_df.held_out_study == group.held_out_study) &
                    (pred_df.panel_type == group.panel_type) & (pred_df.model == group.model)]
        y = (p.true_condition == "CRC").astype(int).to_numpy()
        prob = p.predicted_crc_probability.to_numpy(float)
        ci_rows.append(bootstrap_metric_ci(y, prob, derived_seed(str(group.fold_index), group.panel_type, group.model)))
    ci_df = pd.concat([metrics_df.reset_index(drop=True), pd.DataFrame(ci_rows)], axis=1)
    paired = ci_df.pivot_table(index=["fold_index", "held_out_study", "panel_type"], columns="model", values=["auroc", "average_precision"])
    paired.columns = [f"{metric}_{model}" for metric, model in paired.columns]
    paired = paired.reset_index()
    panel_meta = ci_df[["fold_index", "held_out_study", "panel_type", "direction_threshold", "panel_size"]].drop_duplicates()
    paired = paired.merge(panel_meta, on=["fold_index", "held_out_study", "panel_type"], how="left")
    paired["delta_auroc_rf_minus_elastic_net"] = paired["auroc_random_forest_sensitivity"] - paired["auroc_elastic_net_baseline"]
    paired["delta_average_precision_rf_minus_elastic_net"] = paired["average_precision_random_forest_sensitivity"] - paired["average_precision_elastic_net_baseline"]
    summary_rows = []
    for panel, group in paired.groupby("panel_type", sort=True):
        rng = np.random.default_rng(derived_seed("paired", panel))
        au = group["delta_auroc_rf_minus_elastic_net"].to_numpy(float)
        ap = group["delta_average_precision_rf_minus_elastic_net"].to_numpy(float)
        au_boot = np.array([au[rng.integers(0, len(au), len(au))].mean() for _ in range(BOOTSTRAP_REPLICATES)])
        ap_boot = np.array([ap[rng.integers(0, len(ap), len(ap))].mean() for _ in range(BOOTSTRAP_REPLICATES)])
        summary_rows.append({"panel_type": panel, "n_folds": len(group), "delta_auroc_mean": au.mean(), "delta_auroc_median": np.median(au),
                             "delta_auroc_min": au.min(), "delta_auroc_max": au.max(), "delta_auroc_bootstrap_ci_low": np.quantile(au_boot, .025), "delta_auroc_bootstrap_ci_high": np.quantile(au_boot, .975),
                             "delta_average_precision_mean": ap.mean(), "delta_average_precision_median": np.median(ap), "delta_average_precision_min": ap.min(), "delta_average_precision_max": ap.max(),
                             "delta_average_precision_bootstrap_ci_low": np.quantile(ap_boot, .025), "delta_average_precision_bootstrap_ci_high": np.quantile(ap_boot, .975), "bootstrap_replicates": BOOTSTRAP_REPLICATES})
    return ci_df, paired, pd.DataFrame(summary_rows)


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    meta = pd.read_csv(META_PATH, sep="\t").set_index("sample_uid", drop=False)
    clr = pd.read_csv(CLR_PATH, sep="\t", index_col=0)
    if clr.shape[1] != 301:
        raise RuntimeError(f"Expected 301 retained CLR features, found {clr.shape[1]}")
    if list(meta.index) != list(clr.index):
        meta = meta.loc[clr.index]
    if not meta.index.equals(clr.index):
        raise RuntimeError("Metadata and CLR sample order mismatch")
    if not bool(meta["study_condition"].astype(str).isin(["CRC", "control"]).all()):
        raise RuntimeError("Unexpected study_condition labels")
    return meta, clr


def original_panels(clr: pd.DataFrame) -> dict[str, set[str]]:
    """Verified prior global DCP/ICP sets, used only for overlap annotations."""
    interpretation = pd.read_csv(INTERPRETATION_PATH, sep="\t")
    dcp = set(interpretation["taxon_long"].astype(str))
    icp = set(interpretation.loc[interpretation["interpretation_group"] == "high_consistency_candidate", "taxon_long"].astype(str))
    cols = set(clr.columns)
    return {"original_DCP": dcp & cols, "original_ICP": icp & cols}


def write_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def smoke() -> dict:
    started = time.time()
    meta, clr = load_inputs()
    held_out = sorted(meta["study_name"].astype(str).unique())[0]
    train = meta["study_name"].astype(str) != held_out
    test = ~train
    assoc = association_screen(clr.loc[train], meta.loc[train])
    features = select_panel(assoc, 0.60, cap=29)
    if not features:
        raise RuntimeError("Smoke selection returned no features")
    y_train = (meta.loc[train, "study_condition"].astype(str) == "CRC").astype(int).to_numpy()
    y_test = (meta.loc[test, "study_condition"].astype(str) == "CRC").astype(int).to_numpy()
    prob = fit_rf(clr.loc[train, features].to_numpy(float), y_train, clr.loc[test, features].to_numpy(float))
    metric = score(y_test, prob)
    dcp = assoc.loc[assoc["taxon"].isin(features), ["taxon", "dcp", "mean_study_clr_crc_minus_control"]].copy()
    dcp.to_csv(OUT_DIR / "smoke_clr_dcp.tsv", sep="\t", index=False)
    receipt = {"status": "PASS", "mode": "smoke", "held_out_study": held_out,
               "panel": "CLR-DCP60", "panel_size": len(features),
               "n_total": int(len(meta)), "n_train": int(train.sum()), "n_test": int(test.sum()),
               "n_crc_test": int(y_test.sum()), "n_control_test": int((1 - y_test).sum()),
               "clr_shape": list(clr.shape), "clr_dcp_definition": "per-taxon fraction of eligible training studies whose mean CLR CRC-control difference agrees with training-wide CRC coefficient",
               "rf_parameters": {"n_estimators": 500, "max_features": "sqrt", "max_depth": None, "min_samples_leaf": 2, "bootstrap": True, "class_weight": "balanced_subsample", "random_state": MODEL_SEED, "n_jobs": 1},
               "metrics": metric, "elapsed_seconds": time.time() - started, "peak_rss_mb": rss_mb(),
               "memory_check": {"status": "PASS_NO_OOM", "clr_dense_bytes_estimate": int(clr.to_numpy(dtype=float).nbytes), "peak_rss_note": "Windows peak RSS may be unavailable; completion without allocation failure is recorded."},
               "input_sha256": {"metadata": sha256(META_PATH), "clr": sha256(CLR_PATH)},
               "python": sys.version}
    write_json(receipt, OUT_DIR / "smoke_receipt.json")
    pd.DataFrame([{**receipt, **metric}]).drop(columns=["metrics", "rf_parameters", "input_sha256", "python"]).to_csv(OUT_DIR / "smoke_metrics.tsv", sep="\t", index=False)
    return receipt


def run_full() -> dict:
    started = time.time()
    meta, clr = load_inputs()
    studies = sorted(meta["study_name"].astype(str).unique())
    all_results: list[dict] = []
    panel_rows: list[dict] = []
    prediction_rows: list[dict] = []
    errors: list[dict] = []
    # Full-data CLR-DCP sensitivity sets are descriptive fixed-global panels;
    # they are evaluated separately from the training-only panel definitions.
    full_assoc = association_screen(clr, meta)
    global_panels = {
        "CLR-DCP60": select_panel(full_assoc, 0.60),
        "CLR-DCP70": select_panel(full_assoc, 0.70),
        "CLR-DCP80": select_panel(full_assoc, 0.80),
    }
    prior_global = original_panels(clr)
    # Retrospective fixed ICP: the locked Integrated-consistency18 set from
    # the prior verified analysis, evaluated in the same 11-study LOSO arm.
    global_panels["global_locked_ICP"] = sorted(prior_global["original_ICP"])
    dcp_global_rows = []
    for name, features in global_panels.items():
        selected = set(features)
        dcp_n = len(selected & prior_global["original_DCP"])
        icp_n = len(selected & prior_global["original_ICP"])
        threshold = float(name.replace("CLR-DCP", "")) / 100 if name.startswith("CLR-DCP") else np.nan
        rule = ("full-data OLS-HC3 BH FDR <0.10 and per-study mean CLR CRC-control DCP threshold"
                if name.startswith("CLR-DCP") else "prior verified Integrated-consistency18 taxa")
        dcp_global_rows.append({"panel_type": name, "direction_threshold": threshold,
                                "panel_size": len(features), "selected_taxa": ";".join(features),
                                "original_DCP_size": len(prior_global["original_DCP"]), "original_ICP_size": len(prior_global["original_ICP"]),
                                "intersection_original_DCP": dcp_n, "jaccard_original_DCP": dcp_n / len(selected | prior_global["original_DCP"]),
                                "intersection_original_ICP": icp_n, "jaccard_original_ICP": icp_n / len(selected | prior_global["original_ICP"]),
                                "selection_rule": rule})
    pd.DataFrame(dcp_global_rows).to_csv(OUT_DIR / "clr_dcp_global_sets.tsv", sep="\t", index=False)
    full_assoc.assign(original_DCP=full_assoc["taxon"].isin(prior_global["original_DCP"]),
                      original_ICP=full_assoc["taxon"].isin(prior_global["original_ICP"])).to_csv(OUT_DIR / "clr_dcp_global_taxon_statistics.tsv", sep="\t", index=False)
    for fold_index, held_out in enumerate(studies, start=1):
        train = meta["study_name"].astype(str) != held_out
        test = ~train
        assoc = association_screen(clr.loc[train], meta.loc[train])
        panels = {
            "all_301_features": list(clr.columns),
            "training_only_top29_q": assoc.head(29)["taxon"].tolist(),
            "training_only_DCP60": select_panel(assoc, 0.60, cap=29),
            "training_only_DCP80": select_panel(assoc, 0.80, cap=18),
        }
        prior = original_panels(clr)
        for panel_type, features in panels.items():
            threshold = {"training_only_DCP60": 0.60, "training_only_DCP80": 0.80}.get(panel_type, np.nan)
            regime = "training_only_selection"
            selected_set = set(features)
            dcp_intersection = len(selected_set & prior["original_DCP"])
            icp_intersection = len(selected_set & prior["original_ICP"])
            panel_rows.append({"fold_index": fold_index, "held_out_study": held_out, "panel_type": panel_type,
                               "direction_threshold": threshold, "panel_size": len(features),
                               "original_DCP_size": len(prior["original_DCP"]), "original_ICP_size": len(prior["original_ICP"]),
                               "intersection_original_DCP": dcp_intersection, "jaccard_original_DCP": dcp_intersection / len(selected_set | prior["original_DCP"]) if selected_set | prior["original_DCP"] else np.nan,
                               "intersection_original_ICP": icp_intersection, "jaccard_original_ICP": icp_intersection / len(selected_set | prior["original_ICP"]) if selected_set | prior["original_ICP"] else np.nan,
                               "selected_taxa": ";".join(features),
                               "selection_rule": "all 301 globally locked features" if panel_type == "all_301_features" else ("training-only OLS-HC3 BH FDR ranking; top 29" if panel_type == "training_only_top29_q" else f"training-only OLS-HC3 BH FDR <0.10 and mean-CLR DCP >= {threshold:.2f}; cap {len(features)}"),
                               "evaluation_regime": regime,
                               "training_sample_n": int(train.sum()), "training_study_n": int(meta.loc[train, "study_name"].nunique())})
            if not features:
                errors.append({"fold_index": fold_index, "held_out_study": held_out, "panel_type": panel_type, "error": "no selected features"})
                continue
            y_train = (meta.loc[train, "study_condition"].astype(str) == "CRC").astype(int).to_numpy()
            y_test = (meta.loc[test, "study_condition"].astype(str) == "CRC").astype(int).to_numpy()
            x_train = clr.loc[train, features].to_numpy(float)
            x_test = clr.loc[test, features].to_numpy(float)
            for model_name, fitter in (("elastic_net_baseline", fit_elastic_net), ("random_forest_sensitivity", fit_rf)):
                try:
                    prob = fitter(x_train, y_train, x_test)
                    metrics = score(y_test, prob)
                    row = {"fold_index": fold_index, "held_out_study": held_out, "panel_type": panel_type,
                           "evaluation_regime": regime,
                           "direction_threshold": threshold, "panel_size": len(features), "model": model_name,
                           "n_train": int(train.sum()), "n_test": int(test.sum()), "n_crc_test": int(y_test.sum()),
                           "n_control_test": int((1 - y_test).sum()), **metrics, "status": "COMPUTED"}
                    all_results.append(row)
                    for uid, truth, p in zip(meta.loc[test, "sample_uid"], y_test, prob):
                        prediction_rows.append({**{k: row[k] for k in ["fold_index", "held_out_study", "panel_type", "direction_threshold", "panel_size", "model"]},
                                                "evaluation_regime": regime,
                                                "sample_uid": uid, "true_condition": "CRC" if truth else "control", "predicted_crc_probability": float(p)})
                except Exception as exc:
                    errors.append({"fold_index": fold_index, "held_out_study": held_out, "panel_type": panel_type, "model": model_name, "error": repr(exc)})
        # Retrospective global CLR-DCP sensitivity panels; these use all-study
        # selection and are explicitly separated from training-only rows.
        for panel_type, features in global_panels.items():
            threshold = (float(panel_type.replace("CLR-DCP", "")) / 100.0
                         if panel_type.startswith("CLR-DCP") else np.nan)
            selected_set = set(features)
            dcp_intersection = len(selected_set & prior["original_DCP"])
            icp_intersection = len(selected_set & prior["original_ICP"])
            panel_rows.append({"fold_index": fold_index, "held_out_study": held_out, "panel_type": panel_type,
                               "direction_threshold": threshold, "panel_size": len(features),
                               "original_DCP_size": len(prior["original_DCP"]), "original_ICP_size": len(prior["original_ICP"]),
                               "intersection_original_DCP": dcp_intersection, "jaccard_original_DCP": dcp_intersection / len(selected_set | prior["original_DCP"]) if selected_set | prior["original_DCP"] else np.nan,
                               "intersection_original_ICP": icp_intersection, "jaccard_original_ICP": icp_intersection / len(selected_set | prior["original_ICP"]) if selected_set | prior["original_ICP"] else np.nan,
                               "selected_taxa": ";".join(features), "selection_rule": (f"full-data OLS-HC3 BH FDR <0.10 and mean-CLR DCP >= {threshold:.2f}; retrospective fixed global panel"
                                   if panel_type.startswith("CLR-DCP") else "prior verified Integrated-consistency18 taxa; retrospective fixed global panel"),
                               "evaluation_regime": "retrospective_global_selection", "training_sample_n": int(train.sum()), "training_study_n": int(meta.loc[train, "study_name"].nunique())})
            if not features:
                continue
            y_train = (meta.loc[train, "study_condition"].astype(str) == "CRC").astype(int).to_numpy()
            y_test = (meta.loc[test, "study_condition"].astype(str) == "CRC").astype(int).to_numpy()
            x_train = clr.loc[train, features].to_numpy(float)
            x_test = clr.loc[test, features].to_numpy(float)
            for model_name, fitter in (("elastic_net_baseline", fit_elastic_net), ("random_forest_sensitivity", fit_rf)):
                try:
                    prob = fitter(x_train, y_train, x_test)
                    metrics = score(y_test, prob)
                    row = {"fold_index": fold_index, "held_out_study": held_out, "panel_type": panel_type,
                           "evaluation_regime": "retrospective_global_selection", "direction_threshold": threshold, "panel_size": len(features), "model": model_name,
                           "n_train": int(train.sum()), "n_test": int(test.sum()), "n_crc_test": int(y_test.sum()),
                           "n_control_test": int((1 - y_test).sum()), **metrics, "status": "COMPUTED"}
                    all_results.append(row)
                    for uid, truth, p in zip(meta.loc[test, "sample_uid"], y_test, prob):
                        prediction_rows.append({**{k: row[k] for k in ["fold_index", "held_out_study", "panel_type", "direction_threshold", "panel_size", "model", "evaluation_regime"]},
                                                "sample_uid": uid, "true_condition": "CRC" if truth else "control", "predicted_crc_probability": float(p)})
                except Exception as exc:
                    errors.append({"fold_index": fold_index, "held_out_study": held_out, "panel_type": panel_type, "model": model_name, "error": repr(exc)})
    metrics_df = pd.DataFrame(all_results)
    panels_df = pd.DataFrame(panel_rows)
    pred_df = pd.DataFrame(prediction_rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(OUT_DIR / "rf_clr_loso_metrics.tsv", sep="\t", index=False)
    panels_df.to_csv(OUT_DIR / "rf_clr_training_panel_selection.tsv", sep="\t", index=False)
    pred_df.to_csv(OUT_DIR / "rf_clr_loso_predictions.tsv.gz", sep="\t", index=False, compression="gzip")
    metrics_ci, paired_df, paired_summary = add_bootstrap_and_pairwise(metrics_df, pred_df)
    metrics_ci.to_csv(OUT_DIR / "rf_clr_loso_metrics_with_bootstrap_ci.tsv", sep="\t", index=False)
    paired_df.to_csv(OUT_DIR / "rf_minus_elastic_net_fold_differences.tsv", sep="\t", index=False)
    paired_summary.to_csv(OUT_DIR / "rf_minus_elastic_net_summary.tsv", sep="\t", index=False)
    error_columns = ["fold_index", "held_out_study", "panel_type", "model", "error"]
    pd.DataFrame(errors, columns=error_columns).to_csv(OUT_DIR / "rf_clr_errors.tsv", sep="\t", index=False)
    summary = (metrics_df.groupby(["model", "panel_type", "direction_threshold"], dropna=False)
               .agg(n_folds=("held_out_study", "nunique"), mean_auroc=("auroc", "mean"), median_auroc=("auroc", "median"),
                    mean_average_precision=("average_precision", "mean"), median_average_precision=("average_precision", "median"),
                    mean_balanced_accuracy=("balanced_accuracy_at_0.5", "mean"), mean_panel_size=("panel_size", "mean"))
               .reset_index())
    summary.to_csv(OUT_DIR / "rf_clr_summary_metrics.tsv", sep="\t", index=False)
    expected_rows = len(studies) * 8 * 2
    receipt = {"status": "PASS" if len(errors) == 0 and len(metrics_df) == expected_rows else "PARTIAL",
               "mode": "full", "studies": studies, "n_studies": len(studies), "n_panel_types": 8, "n_models": 2,
               "expected_metric_rows": expected_rows, "observed_metric_rows": int(len(metrics_df)),
               "n_errors": len(errors), "thresholds": list(THRESHOLDS), "global_locked_feature_count": int(clr.shape[1]),
               "preprocessing_boundary": "301-feature CLR matrix and pseudocount globally locked from prior verified run; association/panel selection and model fit restricted to training studies per LOSO fold",
               "panel_names": ["all_301_features", "training_only_top29_q", "training_only_DCP60", "training_only_DCP80", "CLR-DCP60", "CLR-DCP70", "CLR-DCP80", "global_locked_ICP"],
               "training_only_ICP": {"status": "NOT_IMPLEMENTED_IN_BATCH_A", "reason": "Full nested ICP requires the separate Batch B rule audit; no training-fold ICP membership is inferred here."},
               "overlap_reference_panels": {k: len(v) for k, v in original_panels(clr).items()},
               "rf_parameters": {"n_estimators": 500, "max_features": "sqrt", "max_depth": None, "min_samples_leaf": 2, "bootstrap": True, "class_weight": "balanced_subsample", "random_state": MODEL_SEED, "n_jobs": 1},
               "bootstrap": {"replicates": BOOTSTRAP_REPLICATES, "seed_base": MODEL_SEED, "method": "stratified held-out sample bootstrap; paired RF-minus-ElasticNet fold resampling for summary CIs"},
               "elapsed_seconds": time.time() - started, "peak_rss_mb": rss_mb(),
               "input_sha256": {"metadata": sha256(META_PATH), "clr": sha256(CLR_PATH)},
               "outputs": [str(p) for p in [OUT_DIR / "rf_clr_loso_metrics.tsv", OUT_DIR / "rf_clr_loso_metrics_with_bootstrap_ci.tsv", OUT_DIR / "rf_minus_elastic_net_fold_differences.tsv", OUT_DIR / "rf_minus_elastic_net_summary.tsv", OUT_DIR / "rf_clr_training_panel_selection.tsv", OUT_DIR / "rf_clr_loso_predictions.tsv.gz", OUT_DIR / "rf_clr_summary_metrics.tsv", OUT_DIR / "rf_clr_errors.tsv"]]}
    write_json(receipt, OUT_DIR / "rf_clr_run_receipt.json")
    return receipt


def write_claim_boundary(receipt: dict) -> None:
    text = f"""# CLR, ICP, and Random Forest repair claim boundary

## Scope and evidence

This module reuses the verified 301-feature CLR matrix from the prior run. The feature universe, pseudocount, and CLR coordinates are globally locked before LOSO. For each held-out study, association testing and direction-based panel selection are refit on the remaining training studies only. Direction concordance proportion (DCP) is the fraction of eligible training studies in which the mean CLR CRC-minus-control difference has the same sign as the training-wide study-adjusted CRC coefficient. Panels are named CLR-DCP60, CLR-DCP70, and CLR-DCP80 and use BH FDR < 0.10 with the corresponding DCP threshold; no tuning or post-hoc threshold optimization was performed. Overlap/Jaccard columns compare each fold-specific panel with the prior global DCP (Stable29) and ICP (Integrated-consistency18) sets and are descriptive only.

The retrospective `global_locked_ICP` panel is the prior verified Integrated-consistency18 set (n=18), evaluated in the same 11-study LOSO arm. It is a fixed transfer sensitivity panel; no held-out study contributes to its membership. A fully nested training-only ICP was not implemented in Batch A and must not be inferred from these results. The fixed Random Forest sensitivity model uses `n_estimators=500`, `max_features='sqrt'`, `max_depth=None`, `min_samples_leaf=2`, `bootstrap=True`, `class_weight='balanced_subsample'`, `random_state={MODEL_SEED}`, and `n_jobs=1`. Elastic Net metrics are included only as the already-specified baseline vehicle for model-class comparison.

## What the results support

- Fold-level AUROC, average precision, and balanced accuracy are descriptive held-out-study performance measurements conditional on the observed 11 study labels.
- Differences between the fixed Random Forest and Elastic Net quantify model-class sensitivity under the same globally locked CLR inputs and training-only panel rules.
- Panel size and membership changes across DCP thresholds, plus the fixed ICP transfer arm, quantify operational sensitivity of the prioritization rule.

## What the results do not support

- No clinical diagnostic utility, prospective validation, calibration, causal mechanism, or independent external validation claim.
- No claim that Random Forest is superior; this is a single fixed, non-tuned sensitivity model, not a model-selection benchmark.
- No fully end-to-end leakage-free claim: global feature filtering, pseudocount, and CLR coordinates were locked using the complete processed matrix before LOSO.
- DCP is an operational direction-agreement score, not a probability, meta-analytic confidence, or formal validation statistic.
- The fixed ICP arm is retrospective and globally selected; it is not a leakage-free nested ICP estimate.

## Run status

- Status: `{receipt.get('status')}`; observed metric rows: {receipt.get('observed_metric_rows', 'smoke-only')}.
- Outputs are limited to this new run directory; the prior run was read-only.
"""
    (OUT_DIR / "rf_clr_claim_boundary.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "full"], required=True)
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    receipt = smoke() if args.mode == "smoke" else run_full()
    write_claim_boundary(receipt)
    print(json.dumps(receipt, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()

