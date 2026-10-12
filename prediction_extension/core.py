"""Validated numerical primitives for the four-definition training analysis."""
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

DEFINITIONS = ('raw_mean', 'raw_median', 'clr_mean', 'clr_median')
ZERO_TOLERANCE = 1e-12


def require_numeric(raw):
    arr = np.asarray(raw, dtype=float)
    if arr.ndim != 2 or min(arr.shape) == 0 or not np.isfinite(arr).all() or (arr < 0).any():
        raise ValueError('Nonempty finite nonnegative sample by feature matrix required')
    return arr


def require_labels(y, n):
    labels = np.asarray(y)
    if labels.ndim != 1 or len(labels) != n or set(labels.tolist()) != {0, 1}:
        raise ValueError('Aligned binary labels with both classes required')
    return labels.astype(int)


def clr(raw, pseudocount):
    arr = require_numeric(raw)
    if not np.isfinite(pseudocount) or pseudocount <= 0:
        raise ValueError('Positive finite pseudocount required')
    logs = np.log(arr + pseudocount)
    transformed = logs - logs.mean(axis=1, keepdims=True)
    if not np.isfinite(transformed).all():
        raise ValueError('Nonfinite CLR result')
    return transformed


def effect_definitions(raw, y, pseudocount):
    arr = require_numeric(raw)
    labels = require_labels(y, len(arr))
    transformed = clr(arr, pseudocount)
    effects = {}
    for scale, values in [('raw', arr), ('clr', transformed)]:
        for name, reducer in [('mean', np.mean), ('median', np.median)]:
            effect = reducer(values[labels == 1], axis=0) - reducer(values[labels == 0], axis=0)
            if not np.isfinite(effect).all():
                raise ValueError('Nonfinite group effect')
            effects[f'{scale}_{name}'] = effect
    return effects, (arr > 0).any(axis=0)


def concordance(effect, reference_sign, detected, zero_tolerance=ZERO_TOLERANCE):
    effect, reference = np.asarray(effect, float), np.asarray(reference_sign, float)
    flags = np.asarray(detected, bool)
    if (effect.ndim != 1 or effect.shape != reference.shape or effect.shape != flags.shape
            or not np.isfinite(effect).all() or not np.isfinite(reference).all()
            or not np.isfinite(zero_tolerance) or zero_tolerance < 0):
        raise ValueError('Finite aligned effects, reference signs, detection flags and tolerance required')
    return (np.abs(effect) > zero_tolerance) & flags & (reference != 0) & (np.sign(effect) == np.sign(reference))


def agreement(concordant):
    matrix = np.asarray(concordant, bool)
    if matrix.ndim != 2 or min(matrix.shape) == 0:
        raise ValueError('Nonempty study by feature concordance matrix required')
    return matrix.mean(axis=0)


def common_panel_size(eligible_sets, cap=18):
    groups = list(eligible_sets.values()) if isinstance(eligible_sets, dict) else list(eligible_sets)
    if len(groups) != 8 or isinstance(cap, bool) or not isinstance(cap, (int, np.integer)) or cap < 0:
        raise ValueError('Exactly eight candidate groups and a nonnegative integer cap required')
    return int(min(cap, *(len(group) for group in groups)))


def paired_metrics(y, prediction_map, indices):
    labels = require_labels(y, len(y))
    if len(prediction_map) < 2:
        raise ValueError('At least two prediction vectors required for paired contrasts')
    predictions = {}
    for name, vector in prediction_map.items():
        values = np.asarray(vector, dtype=float)
        if values.ndim != 1 or len(values) != len(labels) or not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
            raise ValueError(f'Invalid probabilities for {name}')
        predictions[name] = values
    resamples = np.asarray(indices)
    if (resamples.ndim != 2 or min(resamples.shape) == 0
            or not np.issubdtype(resamples.dtype, np.integer)
            or (resamples < 0).any() or (resamples >= len(labels)).any()):
        raise ValueError('Nonempty two-dimensional integer resampling indices required')
    rows = []
    metrics = [('AUROC', roc_auc_score), ('AP', average_precision_score)]
    for replicate, ix in enumerate(resamples):
        sampled_labels = require_labels(labels[ix], len(ix))
        scores = {name: [float(metric(sampled_labels, values[ix])) for _, metric in metrics]
                  for name, values in predictions.items()}
        for left, right in combinations(predictions, 2):
            for position, (name, _) in enumerate(metrics):
                lv, rv = scores[left][position], scores[right][position]
                rows.append({'replicate': replicate, 'left': left, 'right': right,
                             'contrast': f'{left} - {right}', 'metric': name,
                             'left_value': lv, 'right_value': rv, 'difference': lv - rv})
    return pd.DataFrame(rows)
