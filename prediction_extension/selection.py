"""Train-only filtering, statistical evidence, panel selection and identities."""
import hashlib
import importlib.metadata
import json

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests

from .core import DEFINITIONS, clr, concordance, common_panel_size, effect_definitions, require_labels, require_numeric

DEFAULT_CONFIG = dict(prevalence_threshold=.05, min_studies=6, direction_threshold=.60,
                      q_threshold=.10, sign_retention=.90, q_retention=.80,
                      i2_threshold=75., cap=18, zero_tolerance=1e-12)


def validated_config(config):
    config = {} if config is None else dict(config)
    if set(config) - set(DEFAULT_CONFIG):
        raise ValueError('Unknown training configuration key')
    result = {**DEFAULT_CONFIG, **config}
    if any(not np.isfinite(value) for value in result.values()):
        raise ValueError('Finite configuration values required')
    for name in ['prevalence_threshold', 'direction_threshold', 'q_threshold', 'sign_retention', 'q_retention']:
        if not 0 <= result[name] <= 1:
            raise ValueError(f'{name} must lie in [0,1]')
    for name in ['min_studies', 'cap']:
        if isinstance(result[name], bool) or not isinstance(result[name], (int, np.integer)) or result[name] < (1 if name == 'min_studies' else 0):
            raise ValueError(f'Invalid integer {name}')
        result[name] = int(result[name])
    if not 0 <= result['i2_threshold'] <= 100 or result['zero_tolerance'] < 0:
        raise ValueError('Invalid heterogeneity threshold or zero tolerance')
    return result


def validate_training(raw, y, studies, sample_ids):
    if not isinstance(raw, pd.DataFrame) or raw.columns.has_duplicates or raw.index.has_duplicates:
        raise ValueError('DataFrame with unique sample IDs and feature IDs required')
    ids = list(sample_ids)
    if raw.index.tolist() != ids or len(set(ids)) != len(ids):
        raise ValueError('Unique sample IDs must equal training row index in order')
    if not all(isinstance(taxon, str) and taxon for taxon in raw.columns):
        raise ValueError('Nonempty full taxon string IDs required')
    arr = require_numeric(raw)
    labels = require_labels(y, len(arr))
    study = np.asarray(studies)
    if study.ndim != 1 or len(study) != len(arr) or not all(isinstance(s, str) and s for s in study):
        raise ValueError('Aligned nonempty study strings required')
    for name in pd.unique(study):
        require_labels(labels[study == name], int((study == name).sum()))
    return arr, labels, study, ids


def split_outer_fold(raw, y, studies, sample_ids, heldout):
    _, labels, study, ids = validate_training(raw, y, studies, sample_ids)
    test = study == heldout
    if not test.any() or test.all():
        raise ValueError('Heldout study must define nonempty disjoint training and test partitions')
    result = {}
    for name, mask in [('train', ~test), ('test', test)]:
        result[name] = {'raw': raw.loc[mask].copy(), 'y': labels[mask].copy(),
                        'studies': study[mask].copy(), 'sample_ids': [sid for sid, keep in zip(ids, mask) if keep]}
    if set(result['train']['sample_ids']) & set(result['test']['sample_ids']):
        raise ValueError('Overlapping train/test sample identifiers')
    return result


def bh(p):
    values = np.asarray(p, dtype=float)
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError('Complete finite valid p-value family required for BH')
    return multipletests(values, method='fdr_bh')[1]


def association(x, y, studies):
    values = np.asarray(x, float)
    labels = require_labels(y, len(values))
    study = np.asarray(studies)
    if values.ndim != 2 or min(values.shape) == 0 or not np.isfinite(values).all() or study.ndim != 1 or len(study) != len(labels):
        raise ValueError('Finite aligned CLR, labels and studies required')
    dummies = pd.get_dummies(pd.Series(study, dtype='category'), drop_first=True, dtype=float)
    design = np.column_stack([np.ones(len(labels)), labels.astype(float), dummies.to_numpy()])
    if np.linalg.matrix_rank(design) != design.shape[1] or len(labels) <= design.shape[1]:
        raise ValueError('Association design is rank deficient or has no residual degrees of freedom')
    leverage = np.einsum('ij,ji->i', design, np.linalg.pinv(design))
    if (leverage >= 1 - 1e-12).any():
        raise ValueError('HC3 association is undefined for unit-leverage observations')
    coefficients, pvalues = [], []
    for column in values.T:
        fit = sm.OLS(column, design).fit(cov_type='HC3')
        coefficients.append(float(fit.params[1]))
        pvalues.append(float(fit.pvalues[1]))
    if not np.isfinite(coefficients).all() or not np.isfinite(pvalues).all():
        raise ValueError('Nonfinite association result; complete BH family required')
    return pd.DataFrame({'coef': coefficients, 'p': pvalues, 'q': bh(pvalues)})


def study_effects(x, y, studies):
    values = np.asarray(x, float)
    labels = require_labels(y, len(values))
    study = np.asarray(studies)
    if values.ndim != 2 or min(values.shape) == 0 or not np.isfinite(values).all() or study.ndim != 1 or len(study) != len(labels):
        raise ValueError('Finite aligned CLR, labels and studies required')
    effects, variances = [], []
    for name in pd.unique(study):
        case, control = values[(study == name) & (labels == 1)], values[(study == name) & (labels == 0)]
        if not len(case) or not len(control):
            raise ValueError('Each study must contain CRC and control samples')
        effects.append(case.mean(axis=0) - control.mean(axis=0))
        # Historical estimator assigns zero within-group variance to singleton groups.
        cv = np.var(case, axis=0, ddof=1) if len(case) > 1 else np.zeros(values.shape[1])
        kv = np.var(control, axis=0, ddof=1) if len(control) > 1 else np.zeros(values.shape[1])
        variances.append(np.maximum(cv / len(case) + kv / len(control), 1e-12))
    return np.asarray(effects), np.asarray(variances)


def random_effects(effects, variances):
    effect, variance = np.asarray(effects, float), np.asarray(variances, float)
    if effect.ndim != 2 or min(effect.shape) == 0 or effect.shape != variance.shape or not np.isfinite(effect).all() or not np.isfinite(variance).all() or (variance < 0).any():
        raise ValueError('Aligned finite effects and nonnegative variances required')
    k, width = effect.shape
    p, i2 = np.empty(width), np.empty(width)
    for j in range(width):
        e, v = effect[:, j], np.maximum(variance[:, j], 1e-12)
        w = 1 / v
        mu = np.sum(w * e) / np.sum(w)
        q = np.sum(w * (e - mu) ** 2)
        degrees = max(k - 1, 1)
        c = np.sum(w) - np.sum(w * w) / np.sum(w)
        tau2 = max(0., (q - degrees) / c) if c > 0 else 0.
        wr = 1 / (v + tau2)
        pooled, se = np.sum(wr * e) / np.sum(wr), np.sqrt(1 / np.sum(wr))
        p[j] = 2 * norm.sf(abs(pooled / se))
        i2[j] = max(0., (q - degrees) / q) * 100 if q > 0 else 0.
    if not np.isfinite(p).all() or not np.isfinite(i2).all():
        raise ValueError('Nonfinite random-effects result')
    return p, i2


def fingerprint(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()


def verify_training_state_integrity(state):
    """Reject mutation of fields represented by the existing derived-state hash."""
    try:
        recomputed = fingerprint({'training_inputs': state['hashes']['training_inputs'],
                                  'features': state['features'], 'pseudocount': state['pseudocount'],
                                  'memberships': state['memberships'], 'config': state['config'],
                                  'package_versions': state['package_versions']})
        stored = state['hashes']['training_state']
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError('Invalid derived training state identity payload') from exc
    if recomputed != stored:
        raise ValueError('Derived training state identity mismatch')
    return recomputed


def package_versions():
    return {name: importlib.metadata.version(name) for name in ['numpy', 'pandas', 'scipy', 'statsmodels', 'scikit-learn']}


def coefficient_sign_concordance(coefficients, reference):
    values, signs = np.asarray(coefficients, float), np.asarray(reference, float)
    if values.ndim != 1 or values.shape != signs.shape or not np.isfinite(values).all() or not np.isfinite(signs).all():
        raise ValueError('Finite aligned association coefficients and reference signs required')
    return (np.sign(values) == np.sign(signs)) & (signs != 0)


def training_input_fingerprint(raw, labels, studies, ids, config, versions):
    # All-zero training columns carry no information and may originate exclusively in heldout data.
    nonzero_columns = sorted(raw.columns[(raw.to_numpy(dtype=float) > 0).any(axis=0)])
    return fingerprint({'sample_ids': ids, 'raw_columns': nonzero_columns,
                        'raw': raw.loc[:, nonzero_columns].to_numpy(dtype=float).tolist(),
                        'labels': np.asarray(labels).tolist(), 'studies': np.asarray(studies).tolist(),
                        'config': config, 'package_versions': versions})


def build_training_state(raw_train, y_train, study_train, sample_ids, config):
    _, labels, studies, ids = validate_training(raw_train, y_train, study_train, sample_ids)
    settings = validated_config(config)
    raw = raw_train.loc[:, sorted(raw_train.columns)]
    values = raw.to_numpy(dtype=float)
    present = values > 0
    prevalence = present.mean(axis=0)
    detected_studies = np.sum([present[studies == name].any(axis=0) for name in pd.unique(studies)], axis=0)
    retained = (prevalence >= settings['prevalence_threshold']) & (detected_studies >= settings['min_studies']) & present.any(axis=0)
    features = raw.columns[retained].tolist()
    if not features:
        raise ValueError('No positive training features pass prevalence and study support')
    values = values[:, retained]
    pseudocount = float(values[values > 0].min() / 2)
    transformed = clr(values, pseudocount)
    evidence = association(transformed, labels, studies)
    evidence.index = pd.Index(features, name='taxon')
    evidence['assoc_coef'], evidence['assoc_p'], evidence['assoc_q'] = evidence.coef, evidence.p, evidence.q
    evidence['prevalence'], evidence['detected_studies'] = prevalence[retained], detected_studies[retained]
    reference = np.sign(evidence.coef.to_numpy())
    effects, variances = study_effects(transformed, labels, studies)
    evidence['re_p'], evidence['i2'] = random_effects(effects, variances)
    evidence['re_q'] = bh(evidence.re_p.to_numpy())
    inner_sign, inner_q = [], []
    direction_matrices = {definition: [] for definition in DEFINITIONS}
    details = []
    for name in pd.unique(studies):
        keep = studies != name
        inner = association(transformed[keep], labels[keep], studies[keep])
        inner_sign.append(coefficient_sign_concordance(inner.coef.to_numpy(), reference))
        inner_q.append(inner.q.to_numpy() < settings['q_threshold'])
        within, detected = effect_definitions(values[~keep], labels[~keep], pseudocount)
        for definition in DEFINITIONS:
            effect = within[definition]
            same = concordance(effect, reference, detected, settings['zero_tolerance'])
            reverse = concordance(-effect, reference, detected, settings['zero_tolerance'])
            direction_matrices[definition].append(same)
            for j, taxon in enumerate(features):
                details.append({'study': name, 'taxon': taxon, 'definition': definition,
                                'effect': float(effect[j]), 'reference_sign': int(reference[j]),
                                'exact_zero': bool(effect[j] == 0),
                                'tolerance_zero': bool(abs(effect[j]) <= settings['zero_tolerance']),
                                'detected': bool(detected[j]), 'same_direction': bool(same[j]),
                                'reverse_direction': bool(reverse[j])})
    evidence['inner_sign_agreement'] = np.mean(inner_sign, axis=0)
    evidence['inner_q_fraction'] = np.mean(inner_q, axis=0)
    evidence['eligible_study_count'] = len(pd.unique(studies))
    eligible = {}
    for definition, matrix in direction_matrices.items():
        evidence[f'direction_{definition}'] = np.mean(matrix, axis=0)
        evidence[f'same_count_{definition}'] = np.sum(matrix, axis=0)
        dcp = (evidence.q < settings['q_threshold']) & (evidence[f'direction_{definition}'] >= settings['direction_threshold'])
        icp = dcp & (evidence.inner_sign_agreement >= settings['sign_retention']) & (evidence.inner_q_fraction >= settings['q_retention']) & (evidence.re_q < settings['q_threshold']) & (evidence.i2 < settings['i2_threshold'])
        for family, mask in [('DCP', dcp), ('ICP', icp)]:
            eligible[f'{family}_{definition}'] = set(evidence.index[mask])
            evidence[f'eligible_{family}_{definition}'] = mask
    ranked = evidence.reset_index().sort_values(['q', 'taxon']).taxon.tolist()
    evidence['direction_agreement'] = evidence['direction_raw_median']
    k = common_panel_size(eligible, settings['cap'])
    memberships = {'ARP': ranked[:k]}
    memberships.update({label: [taxon for taxon in ranked if taxon in candidates][:k] for label, candidates in eligible.items()})
    versions = package_versions()
    input_hash = training_input_fingerprint(raw_train, labels, studies, ids, settings, versions)
    state_hash = fingerprint({'training_inputs': input_hash, 'features': features,
                              'pseudocount': pseudocount, 'memberships': memberships, 'config': settings,
                              'package_versions': versions})
    return {'features': features, 'pseudocount': pseudocount, 'evidence': evidence,
            'memberships': memberships, 'eligible_counts': {label: len(group) for label, group in eligible.items()},
            'k': k, 'hashes': {'training_inputs': input_hash, 'training_state': state_hash},
            'direction_details': pd.DataFrame(details), 'config': settings,
            'package_versions': versions, 'sample_ids': ids, 'training_studies': studies.tolist()}
