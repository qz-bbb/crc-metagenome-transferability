"""Paired, stratified fixed-prediction uncertainty for the nine-study Batch B.

No inputs are read at import time. Metric order is always (AUROC, AP).
Intervals condition on the fitted models and these cohorts, and do not measure
training uncertainty, population uncertainty, or equivalence.
"""
import hashlib
from numbers import Integral

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


DEFINITIONS = ("raw_mean", "raw_median", "clr_mean", "clr_median")
PANELS = ("ARP",) + tuple(f"{family}_{definition}" for family in ("DCP", "ICP")
                          for definition in DEFINITIONS)
CLASSIFIERS = ("elastic_net", "random_forest")
DEFAULT_MODELS = tuple((panel, clf) for panel in PANELS for clf in CLASSIFIERS)
METRICS = ("AUROC", "AP")
PRIMARY = {"ICP_raw_mean_minus_raw_median", "ICP_clr_median_minus_raw_median"}


def validate_predictions(pred, metadata, expected_models=None):
    """Validate a complete sample/model grid and join labels by sample_uid.

    ``expected_models`` optionally contains (panel, classifier) pairs, or the
    equivalent ``panel::classifier`` strings, for bounded synthetic checks.
    Metadata may contain a single held-out fold or several complete folds.
    Returned rows are sorted by study, model_key, sample_uid.
    """
    pred_required = {"sample_uid", "held_out_study", "panel", "classifier", "k", "y",
                     "probability_crc"}
    meta_required = {"sample_uid", "study_name", "study_condition"}
    for frame, required, name in ((pred, pred_required, "predictions"),
                                  (metadata, meta_required, "metadata")):
        if not isinstance(frame, pd.DataFrame) or not required.issubset(frame.columns):
            raise ValueError(f"{name} lacks required columns: {sorted(required)}")
        if frame.empty or frame[list(required)].isna().any().any():
            raise ValueError(f"{name} is empty or contains missing required values")
    p = pred[list(sorted(pred_required))].copy()
    m = metadata[list(sorted(meta_required))].copy()
    for column in ("sample_uid", "held_out_study", "panel", "classifier"):
        if not p[column].map(lambda value: isinstance(value, str) and bool(value.strip())).all():
            raise ValueError(f"prediction {column} must contain nonempty strings")
    for column in ("sample_uid", "study_name", "study_condition"):
        if not m[column].map(lambda value: isinstance(value, str) and bool(value.strip())).all():
            raise ValueError(f"metadata {column} must contain nonempty strings")
    if m.sample_uid.duplicated().any():
        raise ValueError("duplicate metadata sample_uid")
    if not m.study_condition.isin(["CRC", "control"]).all():
        raise ValueError("metadata study_condition must be CRC/control")
    if set(p.sample_uid) != set(m.sample_uid):
        raise ValueError("missing or extra prediction sample_uid")
    if p.duplicated(["sample_uid", "panel", "classifier"]).any():
        raise ValueError("duplicate sample/model prediction rows")
    models = DEFAULT_MODELS if expected_models is None else tuple(
        tuple(pair.split("::")) if isinstance(pair, str) else tuple(pair)
        for pair in expected_models)
    if not models or any(len(pair) != 2 for pair in models) or len(set(models)) != len(models):
        raise ValueError("expected_models must contain unique model pairs")
    model_set = set(models)
    if set(zip(p.panel, p.classifier)) != model_set:
        raise ValueError("missing or extra model configurations")
    if not p.groupby("sample_uid", sort=False).size().eq(len(models)).all():
        raise ValueError("incomplete sample/model grid")
    try:
        numeric = p[["y", "k", "probability_crc"]].to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("labels, k and probabilities must be numeric") from exc
    if not np.isfinite(numeric).all():
        raise ValueError("nonfinite labels, k or probabilities")
    if not np.isin(numeric[:, 0], [0, 1]).all():
        raise ValueError("prediction y must be binary")
    if not ((numeric[:, 1] > 0) & (numeric[:, 1] == np.floor(numeric[:, 1]))).all():
        raise ValueError("k must be a positive integer; unavailable folds cannot be evaluated")
    if not ((numeric[:, 2] >= 0) & (numeric[:, 2] <= 1)).all():
        raise ValueError("probabilities must be in [0,1]")
    p["y"] = numeric[:, 0].astype(int)
    p["k"] = numeric[:, 1].astype(int)
    p["probability_crc"] = numeric[:, 2]
    aligned = p.merge(m, on="sample_uid", how="left", validate="many_to_one")
    if not aligned.held_out_study.eq(aligned.study_name).all():
        raise ValueError("held-out study disagrees with metadata")
    if not aligned.y.eq(aligned.study_condition.map({"CRC": 1, "control": 0})).all():
        raise ValueError("prediction labels disagree with metadata")
    if not aligned.groupby("study_name").k.nunique().eq(1).all():
        raise ValueError("k differs within a held-out study")
    aligned["model_key"] = aligned.panel + "::" + aligned.classifier
    return aligned.sort_values(["study_name", "model_key", "sample_uid"]).reset_index(drop=True)


def evaluate_study(pred, metadata, study_index, n_boot=2000, seed=20261011):
    """Evaluate one full 18-model fold with identical stratified indices.

    The controller must supply the index in the sorted nine-study name list.
    The SHA256 provenance is of the actual B x n index array as little-endian
    int64 bytes (C order), following sorted sample_uid alignment.
    """
    for value, name, minimum in ((n_boot, "n_boot", 1), (study_index, "study_index", 0),
                                 (seed, "seed", 0)):
        if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    aligned = validate_predictions(pred, metadata)
    studies = aligned.study_name.unique()
    if len(studies) != 1:
        raise ValueError("evaluate_study requires exactly one held-out study")
    keys = sorted(aligned.model_key.unique())
    groups = [aligned.loc[aligned.model_key.eq(key)].sort_values("sample_uid") for key in keys]
    y = groups[0].y.to_numpy(dtype=int)
    class_indices = [np.flatnonzero(y == label) for label in (0, 1)]
    if any(len(indices) == 0 for indices in class_indices):
        raise ValueError("both CRC and control samples are required")
    rng = np.random.default_rng(np.random.SeedSequence([seed, 1, study_index]))
    indices = np.concatenate([rng.choice(pool, size=(n_boot, len(pool)), replace=True)
                              for pool in class_indices], axis=1)
    index_hash = hashlib.sha256(np.asarray(indices, dtype="<i8").tobytes(order="C")).hexdigest()
    point = np.empty((len(keys), 2), dtype=float)
    bootstrap = np.empty((n_boot, len(keys), 2), dtype=float)
    unique_scores = []
    rows = []
    for j, (key, group) in enumerate(zip(keys, groups)):
        probabilities = group.probability_crc.to_numpy(dtype=float)
        # Only exact equality permits reuse. All model labels stay in the result.
        duplicate = next((previous for previous, scores in unique_scores
                          if np.array_equal(scores, probabilities)), None)
        if duplicate is not None:
            point[j] = point[duplicate]
            bootstrap[:, j] = bootstrap[:, duplicate]
        else:
            unique_scores.append((j, probabilities))
            point[j] = [roc_auc_score(y, probabilities), average_precision_score(y, probabilities)]
            for b, idx in enumerate(indices):
                bootstrap[b, j] = [roc_auc_score(y[idx], probabilities[idx]),
                                    average_precision_score(y[idx], probabilities[idx])]
        ci = np.quantile(bootstrap[:, j], [0.025, 0.975], axis=0)
        panel, classifier = key.split("::")
        rows.append({"study_name": studies[0], "model_key": key, "panel": panel,
                     "classifier": classifier, "n": len(y), "n_crc": int(y.sum()),
                     "n_control": int((y == 0).sum()), "prevalence": float(y.mean()),
                     "k": int(group.k.iloc[0]), "auroc": point[j, 0],
                     "auroc_ci95_low": ci[0, 0], "auroc_ci95_high": ci[1, 0],
                     "ap": point[j, 1], "ap_ci95_low": ci[0, 1], "ap_ci95_high": ci[1, 1]})
    return {"metrics": pd.DataFrame(rows), "model_keys": keys, "point": point,
            "bootstrap": bootstrap, "index_hash": index_hash}


def _contrast_specs(keys):
    """Construct 41 prespecified linear contrasts per metric."""
    specs = []

    def add(name, family, classifier, panel, coefficients):
        weights = np.zeros(len(keys))
        for key, weight in coefficients.items():
            weights[keys.index(key)] = weight
        specs.append((name, family, classifier, panel, weights))

    for family in ("DCP", "ICP"):
        for classifier in CLASSIFIERS:
            for definition in ("raw_mean", "clr_mean", "clr_median"):
                add(f"{family}_{definition}_minus_raw_median", family, classifier, "",
                    {f"{family}_{definition}::{classifier}": 1,
                     f"{family}_raw_median::{classifier}": -1})
            for definition in DEFINITIONS:
                add(f"{family}_{definition}_minus_ARP", family, classifier, "",
                    {f"{family}_{definition}::{classifier}": 1, f"ARP::{classifier}": -1})
            add(f"{family}_interaction", family, classifier, "",
                {f"{family}_clr_mean::{classifier}": 1,
                 f"{family}_clr_median::{classifier}": -1,
                 f"{family}_raw_mean::{classifier}": -1,
                 f"{family}_raw_median::{classifier}": 1})
    for panel in PANELS:
        add(f"{panel}_RF_minus_EN", "ARP" if panel == "ARP" else panel.split("_")[0],
            "RF_minus_EN", panel,
            {f"{panel}::random_forest": 1, f"{panel}::elastic_net": -1})
    return specs


def summarize_studies(study_results, expected_studies):
    """Return equal-study macro metrics and paired linear contrasts.

    Exactly nine named folds and the complete, identically ordered model grid
    are required. Replicates are averaged across studies before quantiles.
    Only the two EN ICP AUROC contrasts get nominal 97.5% intervals.
    """
    expected = list(expected_studies)
    if len(expected) != 9 or len(set(expected)) != 9:
        raise ValueError("expected_studies must explicitly list nine unique studies")
    if set(study_results) != set(expected):
        raise ValueError("missing or extra held-out study results")
    keys = list(study_results[expected[0]]["model_keys"])
    if len(keys) != 18 or set(keys) != {f"{p}::{c}" for p, c in DEFAULT_MODELS}:
        raise ValueError("summary requires the full 18-model grid")
    points, bootstraps = [], []
    boot_shape = None
    for study in expected:
        result = study_results[study]
        if list(result["model_keys"]) != keys:
            raise ValueError("divergent model key order across studies")
        point = np.asarray(result["point"], dtype=float)
        boots = np.asarray(result["bootstrap"], dtype=float)
        if point.shape != (18, 2) or boots.ndim != 3 or boots.shape[1:] != (18, 2) or boots.shape[0] < 1:
            raise ValueError("invalid point or bootstrap shape")
        if boot_shape is not None and boots.shape != boot_shape:
            raise ValueError("divergent bootstrap shapes across studies")
        boot_shape = boots.shape
        if not np.isfinite(point).all() or not np.isfinite(boots).all():
            raise ValueError("nonfinite study scores")
        if not ((point >= 0) & (point <= 1)).all() or not ((boots >= 0) & (boots <= 1)).all():
            raise ValueError("study scores outside [0,1]")
        metrics = result["metrics"]
        required = {"model_key", "study_name", "auroc", "ap"}
        if not required.issubset(metrics.columns) or metrics.model_key.duplicated().any():
            raise ValueError("invalid study metric table")
        if set(metrics.model_key) != set(keys) or not metrics.study_name.eq(study).all():
            raise ValueError("study metric labels disagree with result")
        metric_points = metrics.set_index("model_key").loc[keys, ["auroc", "ap"]].to_numpy(dtype=float)
        # CSV checkpoints may differ by a few floating point rounding bits.
        # Score arrays remain authoritative; substantive disagreement fails.
        if not np.allclose(point, metric_points, rtol=0, atol=1e-12):
            raise ValueError("point array disagrees with study metric table")
        points.append(point)
        bootstraps.append(boots)
    point_stack = np.stack(points)
    macro_point = point_stack.mean(axis=0)
    macro_boot = np.stack(bootstraps).mean(axis=0)
    ci = np.quantile(macro_boot, [0.025, 0.975], axis=0)
    macro_rows = []
    for j, key in enumerate(keys):
        panel, classifier = key.split("::")
        for metric_index, metric in enumerate(METRICS):
            fold_values = point_stack[:, j, metric_index]
            macro_rows.append({"model_key": key, "panel": panel, "classifier": classifier,
                               "metric": metric, "n_studies": 9,
                               "mean": macro_point[j, metric_index],
                               "median": np.median(fold_values), "minimum": np.min(fold_values),
                               "ci95_low": ci[0, j, metric_index], "ci95_high": ci[1, j, metric_index]})
    contrast_rows = []
    specs = _contrast_specs(keys)
    scopes = [(study, point, boots) for study, point, boots in zip(expected, points, bootstraps)]
    scopes.append(("MACRO", macro_point, macro_boot))
    for scope, point, boots in scopes:
        for name, family, classifier, panel, weights in specs:
            estimate = weights @ point
            samples = np.einsum("bjm,j->bm", boots, weights)
            ci95 = np.quantile(samples, [0.025, 0.975], axis=0)
            for metric_index, metric in enumerate(METRICS):
                primary = (name in PRIMARY and classifier == "elastic_net" and metric == "AUROC")
                ci975 = (np.quantile(samples[:, metric_index], [0.0125, 0.9875])
                         if primary else [np.nan, np.nan])
                contrast_rows.append({"scope": scope, "contrast": name, "family": family,
                                      "classifier": classifier, "panel": panel, "metric": metric,
                                      "estimate": estimate[metric_index],
                                      "ci95_low": ci95[0, metric_index], "ci95_high": ci95[1, metric_index],
                                      "ci975_low": ci975[0], "ci975_high": ci975[1], "primary": primary})
    return pd.DataFrame(macro_rows), pd.DataFrame(contrast_rows)
