"""Frozen-state CLR predictors and deterministic fixed-parameter fitting."""
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .core import DEFINITIONS, clr, require_labels, require_numeric
from .selection import fingerprint, package_versions, training_input_fingerprint, validate_training, verify_training_state_integrity


class PanelUnavailableError(ValueError):
    """The common-size grid contains a legitimately empty panel."""


class ModelConvergenceError(RuntimeError):
    """The locked estimator did not converge; no tuning is performed."""


def model_parameters(classifier):
    if classifier == 'elastic_net':
        return dict(penalty='elasticnet', solver='saga', l1_ratio=.5, C=.5,
                    class_weight='balanced', max_iter=2000, random_state=42)
    if classifier == 'random_forest':
        return dict(n_estimators=500, max_features='sqrt', max_depth=None,
                    min_samples_leaf=2, bootstrap=True, class_weight='balanced_subsample',
                    random_state=20260817, n_jobs=1)
    raise ValueError('Classifier must be elastic_net or random_forest')


def make_model(classifier):
    params = model_parameters(classifier)
    model = LogisticRegression(**params) if classifier == 'elastic_net' else RandomForestClassifier(**params)
    return Pipeline([('scaler', StandardScaler()), ('classifier', model)])


def panel_label(family, definition):
    if family == 'ARP' and definition is None:
        return 'ARP'
    if family in ('DCP', 'ICP') and definition in DEFINITIONS:
        return f'{family}_{definition}'
    raise ValueError('Invalid panel family/definition combination')


def selected_features(train_state, family, definition):
    label = panel_label(family, definition)
    if label not in train_state['memberships']:
        raise ValueError('Panel absent from frozen training state')
    selected = train_state['memberships'][label]
    if not selected:
        raise PanelUnavailableError(f'{label} unavailable: empty common-size panel')
    if len(set(selected)) != len(selected) or not set(selected).issubset(train_state['features']):
        raise ValueError('Invalid selected feature IDs')
    return selected


def model_fingerprint(train_state, classifier, family, definition):
    verify_training_state_integrity(train_state)
    selected = selected_features(train_state, family, definition)
    params = model_parameters(classifier)
    return fingerprint({'training_state': train_state['hashes']['training_state'],
                        'classifier': classifier, 'family': family, 'definition': definition,
                        'parameters': params, 'selected_order': selected,
                        'scaler': StandardScaler().get_params(), 'package_versions': package_versions()})


def panel_predictors(train_state, raw_train, raw_test, family, definition):
    selected = selected_features(train_state, family, definition)
    features = train_state['features']
    if len(set(features)) != len(features) or not features:
        raise ValueError('Invalid retained feature universe')
    transformed = []
    for raw in [raw_train, raw_test]:
        if not isinstance(raw, pd.DataFrame) or raw.columns.has_duplicates or raw.index.has_duplicates:
            raise ValueError('Unique sample and feature IDs required for predictors')
        require_numeric(raw)
        missing = [taxon for taxon in features if taxon not in raw.columns]
        if missing:
            raise ValueError(f'Missing retained feature IDs: {missing}')
        full_clr = clr(raw.loc[:, features], train_state['pseudocount'])
        transformed.append(full_clr[:, [features.index(taxon) for taxon in selected]])
    return tuple(transformed)


def fit_panel(train_state, raw_train, y_train, raw_test, classifier, family, definition):
    verify_training_state_integrity(train_state)
    selected_features(train_state, family, definition)
    _, labels, studies, ids = validate_training(raw_train, y_train, train_state['training_studies'], train_state['sample_ids'])
    versions = package_versions()
    if versions != train_state['package_versions']:
        raise ValueError('Runtime package versions differ from frozen training state')
    input_hash = training_input_fingerprint(raw_train, labels, studies, ids, train_state['config'], versions)
    if input_hash != train_state['hashes']['training_inputs']:
        raise ValueError('Training input differs from frozen state identity')
    if set(raw_train.index) & set(raw_test.index):
        raise ValueError('Train/test sample identifiers overlap')
    predictors, test_predictors = panel_predictors(train_state, raw_train, raw_test, family, definition)
    model = make_model(classifier)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error', ConvergenceWarning)
            model.fit(predictors, labels)
    except ConvergenceWarning as exc:
        raise ModelConvergenceError(f'{classifier} did not converge with locked parameters: {exc}') from exc
    probabilities = model.predict_proba(test_predictors)[:, 1]
    if probabilities.ndim != 1 or not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError('Estimator returned invalid CRC probabilities')
    return probabilities
