"""Fixed-model, paired external sensitivity uncertainty for Batch C.

Metric order is (AUROC, AP). All intervals condition on fitted models/current
cohorts; none supplies training uncertainty or confirmatory inference.
The audited Batch B module supplies the prediction validator and contrasts.
"""
import hashlib
from numbers import Integral

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from .evaluation import validate_predictions, DEFAULT_MODELS, _contrast_specs, PRIMARY
METRICS = ("AUROC", "AP")
INFERENCE_STATUS = "post_outcome_sensitivity"


def evaluate_cohort(pred, metadata, cohort_index, n_boot=2000, seed=20261011):
    """Evaluate a full 18-model cohort with shared class-stratified indices.

    ``cohort_index`` is the controller's index in sorted six cohort names.
    Controls are sampled as one entire array, followed by the entire case
    array. Index SHA256 covers little-endian int64 bytes in C order, with
    samples aligned in sorted sample_uid order, exactly as in Batch B.
    """
    for value, name, minimum in ((n_boot, "n_boot", 1), (cohort_index, "cohort_index", 0),
                                 (seed, "seed", 0)):
        if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    aligned = validate_predictions(pred, metadata)
    studies = aligned.study_name.unique()
    if len(studies) != 1:
        raise ValueError("evaluate_cohort requires exactly one external cohort")
    keys = sorted(aligned.model_key.unique())
    groups = [aligned.loc[aligned.model_key.eq(key)].sort_values("sample_uid") for key in keys]
    y = groups[0].y.to_numpy(dtype=int)
    pools = [np.flatnonzero(y == label) for label in (0, 1)]
    if any(len(pool) == 0 for pool in pools):
        raise ValueError("both CRC and control samples are required")
    rng = np.random.default_rng(np.random.SeedSequence([seed, 2, cohort_index]))
    indices = np.concatenate([rng.choice(pool, size=(n_boot, len(pool)), replace=True)
                              for pool in pools], axis=1)
    index_hash = hashlib.sha256(np.asarray(indices, dtype="<i8").tobytes(order="C")).hexdigest()
    point = np.empty((len(keys), 2), dtype=float)
    bootstrap = np.empty((n_boot, len(keys), 2), dtype=float)
    unique_scores, rows = [], []
    for j, (key, group) in enumerate(zip(keys, groups)):
        probabilities = group.probability_crc.to_numpy(dtype=float)
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


def _cohort_names(names, count, label):
    try:
        values = list(names)
    except TypeError as exc:
        raise ValueError(f"{label} must contain {count} unique cohort names") from exc
    if (len(values) != count or
            any(not isinstance(name, str) or not name.strip() for name in values) or
            len(set(values)) != count):
        raise ValueError(f"{label} must contain {count} unique cohort names")
    return values


def summarize_external(cohort_results, expected_cohorts, primary_cohorts):
    """Return PRIMARY3 equal-cohort macro metrics and paired contrasts.

    Require all six full grids. Only the three prespecified primary cohorts
    contribute to macro scores. Their bootstrap arrays are averaged before
    quantiles. Individual cohorts plus PRIMARY3 produce 574 contrast rows.
    ``protocol_focal`` flags two EN ICP AUROC contrasts for nominal 97.5%
    descriptive intervals; it does not declare confirmatory endpoints.
    """
    expected = _cohort_names(expected_cohorts, 6, "expected_cohorts")
    primary = _cohort_names(primary_cohorts, 3, "primary_cohorts")
    if not set(primary).issubset(expected):
        raise ValueError("primary_cohorts must be contained in expected_cohorts")
    if not isinstance(cohort_results, dict) or set(cohort_results) != set(expected):
        raise ValueError("missing or extra external cohort results")
    first = cohort_results[expected[0]]
    if not isinstance(first, dict) or "model_keys" not in first:
        raise ValueError("cohort result lacks model_keys")
    keys = list(first["model_keys"])
    if len(keys) != 18 or set(keys) != {f"{p}::{c}" for p, c in DEFAULT_MODELS}:
        raise ValueError("summary requires the full 18-model grid")
    validated = {}
    boot_shape = None
    for study in expected:
        result = cohort_results[study]
        if not isinstance(result, dict) or not {"model_keys", "point", "bootstrap", "metrics"}.issubset(result):
            raise ValueError("incomplete cohort result")
        if list(result["model_keys"]) != keys:
            raise ValueError("divergent model key order across cohorts")
        try:
            point = np.asarray(result["point"], dtype=float)
            boots = np.asarray(result["bootstrap"], dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("cohort scores must be numeric arrays") from exc
        if point.shape != (18, 2) or boots.ndim != 3 or boots.shape[1:] != (18, 2) or boots.shape[0] < 1:
            raise ValueError("invalid point or bootstrap shape")
        if boot_shape is not None and boots.shape != boot_shape:
            raise ValueError("divergent bootstrap shapes across cohorts")
        boot_shape = boots.shape
        if not np.isfinite(point).all() or not np.isfinite(boots).all():
            raise ValueError("nonfinite cohort scores")
        # sklearn AP can exceed one by one ULP at perfect ranking. Permit
        # machine-roundoff only; retain original standard-metric values.
        # This does not relax the strict [0,1] input probability validator.
        endpoint_tolerance = 8 * np.finfo(float).eps
        if (not ((point >= -endpoint_tolerance) & (point <= 1 + endpoint_tolerance)).all()
                or not ((boots >= -endpoint_tolerance) & (boots <= 1 + endpoint_tolerance)).all()):
            raise ValueError("cohort scores outside [0,1]")
        metrics = result["metrics"]
        required = {"model_key", "study_name", "auroc", "ap"}
        if not isinstance(metrics, pd.DataFrame) or not required.issubset(metrics.columns):
            raise ValueError("invalid cohort metric table")
        if metrics.model_key.duplicated().any() or set(metrics.model_key) != set(keys) or not metrics.study_name.eq(study).all():
            raise ValueError("cohort metric labels disagree with result")
        ordered = metrics.set_index("model_key").loc[keys]
        for column, expected_labels in (("panel", [key.split("::")[0] for key in keys]),
                                         ("classifier", [key.split("::")[1] for key in keys])):
            if column in ordered and list(ordered[column]) != expected_labels:
                raise ValueError("cohort metric model labels disagree with result")
        try:
            metric_points = ordered[["auroc", "ap"]].to_numpy(dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("metric table scores must be numeric") from exc
        if not np.allclose(point, metric_points, rtol=0, atol=1e-12):
            raise ValueError("point array disagrees with cohort metric table")
        validated[study] = (point, boots)
    point_stack = np.stack([validated[study][0] for study in primary])
    macro_point = point_stack.mean(axis=0)
    macro_boot = np.stack([validated[study][1] for study in primary]).mean(axis=0)
    ci = np.quantile(macro_boot, [0.025, 0.975], axis=0)
    macro_rows = []
    for j, key in enumerate(keys):
        panel, classifier = key.split("::")
        for metric_index, metric in enumerate(METRICS):
            values = point_stack[:, j, metric_index]
            macro_rows.append({"model_key": key, "panel": panel, "classifier": classifier,
                               "metric": metric, "n_studies": 3, "mean": macro_point[j, metric_index],
                               "median": np.median(values), "minimum": np.min(values),
                               "ci95_low": ci[0, j, metric_index], "ci95_high": ci[1, j, metric_index],
                               "inference_status": INFERENCE_STATUS})
    contrast_rows = []
    specs = _contrast_specs(keys)
    scopes = [(study, *validated[study]) for study in expected]
    scopes.append(("PRIMARY3", macro_point, macro_boot))
    for scope, point, boots in scopes:
        for name, family, classifier, panel, weights in specs:
            estimate = weights @ point
            samples = np.einsum("bjm,j->bm", boots, weights)
            ci95 = np.quantile(samples, [0.025, 0.975], axis=0)
            for metric_index, metric in enumerate(METRICS):
                focal = name in PRIMARY and classifier == "elastic_net" and metric == "AUROC"
                ci975 = (np.quantile(samples[:, metric_index], [0.0125, 0.9875])
                         if focal else [np.nan, np.nan])
                contrast_rows.append({"scope": scope, "contrast": name, "family": family,
                                      "classifier": classifier, "panel": panel, "metric": metric,
                                      "estimate": estimate[metric_index], "ci95_low": ci95[0, metric_index],
                                      "ci95_high": ci95[1, metric_index], "ci975_low": ci975[0],
                                      "ci975_high": ci975[1], "protocol_focal": focal,
                                      "inference_status": INFERENCE_STATUS})
    return pd.DataFrame(macro_rows), pd.DataFrame(contrast_rows)
