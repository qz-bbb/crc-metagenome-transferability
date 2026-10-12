"""Training-only fitting, blind transformation and separately attached outcomes."""
import warnings
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from .core import clr, require_numeric
from .selection import (verify_training_state_integrity, validate_training, package_versions,
                         training_input_fingerprint)
from .models import make_model, model_fingerprint, ModelConvergenceError
from .evaluation import DEFAULT_MODELS, validate_predictions

BASELINE_PANELS = {'ARP':'ARP_nICP','DCP_raw_median':'DCP_nICP','ICP_raw_median':'ICP'}
BASELINE_CLASSIFIERS = {'elastic_net':'EN','random_forest':'RF'}


def fit_locked(state, raw, y):
    """No external arrays/labels accepted. Return 18 fitted fixed pipelines."""
    verify_training_state_integrity(state)
    _, labels, studies, ids = validate_training(raw, y, state['training_studies'], state['sample_ids'])
    versions = package_versions()
    if versions != state['package_versions'] or training_input_fingerprint(
            raw, labels, studies, ids, state['config'], versions) != state['hashes']['training_inputs']:
        raise ValueError('Training/runtime drift from frozen state')
    if state['k'] <= 0:
        raise ValueError('Empty common panel; no padding')
    full = clr(raw.loc[:,state['features']], state['pseudocount'])
    models, log = {}, []
    for panel, classifier in DEFAULT_MODELS:
        members = state['memberships'][panel]
        if len(members) != state['k'] or len(set(members)) != len(members):
            raise ValueError('Inconsistent common panel size')
        columns = [state['features'].index(t) for t in members]
        model = make_model(classifier)
        try:
            with warnings.catch_warnings(record=True) as messages:
                warnings.simplefilter('always')
                warnings.simplefilter('error', ConvergenceWarning)
                model.fit(full[:,columns], labels)
        except ConvergenceWarning as exc:
            raise ModelConvergenceError(f'{panel}/{classifier}: {exc}') from exc
        family, _, definition = panel.partition('_')
        key = f'{panel}::{classifier}'
        models[key] = {'pipeline':model,'indices':columns,'features':members,
            'fingerprint':model_fingerprint(state,classifier,family,definition or None)}
        log.append({'model_key':key,'n_train':len(raw),'k':len(members),'status':'PASS',
            'warnings':' | '.join(sorted({f'{m.category.__name__}: {m.message}' for m in messages}))})
        print(f'LOCKED FIT {key}', flush=True)
    return models, pd.DataFrame(log)


def transform_blind(raw, state, same_source_verified=False):
    if not isinstance(raw,pd.DataFrame) or raw.empty or raw.columns.has_duplicates or raw.index.has_duplicates:
        raise ValueError('Unique nonempty external samples/taxa required')
    require_numeric(raw)
    features = state['features']
    if not features or len(set(features)) != len(features) or not np.isfinite(state['pseudocount']) or state['pseudocount'] <= 0:
        raise ValueError('Invalid frozen preprocessing')
    missing = [t for t in features if t not in raw.columns]
    if missing and not same_source_verified:
        raise ValueError('Missing taxon rows require verified identical MP4 source coordinates')
    return clr(raw.reindex(columns=features,fill_value=0.),state['pseudocount']), missing


def attach_outcomes(blind, meta, cohort, expected_counts, expected_models=None):
    models = DEFAULT_MODELS if expected_models is None else expected_models
    required = ['Sample Source ID','Study condition']
    if not set(required).issubset(meta.columns) or meta[required].isna().any().any() or meta['Sample Source ID'].duplicated().any():
        raise ValueError('Invalid external outcome metadata')
    if set(blind.sample_id) != set(meta['Sample Source ID']):
        raise ValueError('Missing/extra metadata or prediction samples')
    observed = meta['Study condition'].value_counts().to_dict()
    expected = {k:int(v) for k,v in expected_counts.items() if v}
    if observed != expected or not set(observed).issubset({'CRC','Control','Adenoma'}):
        raise ValueError(f'Clinical labels/counts changed: {observed} versus {expected}')
    if blind.duplicated(['sample_id','panel','classifier']).any() or len(blind)!=len(meta)*len(models):
        raise ValueError('Incomplete/duplicate blind model grid')
    if set(zip(blind.panel,blind.classifier)) != set(models):
        raise ValueError('Missing/extra models')
    # Validate all source samples, including subsequently excluded adenomas.
    if not blind.groupby('sample_id').size().eq(len(models)).all():
        raise ValueError('Incomplete per-sample blind grid')
    probs = blind.probability_crc.to_numpy(dtype=float)
    if not np.isfinite(probs).all() or ((probs<0)|(probs>1)).any():
        raise ValueError('Invalid blind probabilities')
    joined = blind.merge(meta[required],left_on='sample_id',right_on='Sample Source ID',validate='many_to_one')
    p = joined.loc[joined['Study condition'].ne('Adenoma')].copy()
    p['sample_uid'] = cohort + '__' + p.sample_id
    p['held_out_study'] = cohort
    p['y'] = p['Study condition'].map({'CRC':1,'Control':0}).astype(int)
    m = meta.loc[meta['Study condition'].ne('Adenoma'),required].copy()
    m['sample_uid'] = cohort + '__' + m['Sample Source ID']
    m['study_name'] = cohort
    m['study_condition'] = m['Study condition'].replace({'Control':'control'})
    validate_predictions(p,m,expected_models=models)
    return p,m


def compare_baseline(new, old):
    rows=[]
    for panel, previous in BASELINE_PANELS.items():
        for classifier, old_classifier in BASELINE_CLASSIFIERS.items():
            a = new.loc[new.panel.eq(panel)&new.classifier.eq(classifier)].set_index('sample_id')
            b = old.loc[old.panel.eq(previous)&old.classifier.eq(old_classifier)].set_index('sample_id')
            if a.empty or a.index.has_duplicates or b.index.has_duplicates or set(a.index)!=set(b.index):
                raise ValueError('Historical replay missing/duplicate sample identities')
            av=a.probability_crc.to_numpy(float)
            bv=b.loc[a.index].probability_crc.to_numpy(float)
            ok=bool(np.allclose(av,bv,atol=1e-10,rtol=1e-8))
            rows.append({'panel':panel,'classifier':classifier,'n':len(a),
                         'max_abs_difference':float(np.max(np.abs(av-bv))),'pass':ok})
    checks=pd.DataFrame(rows)
    if not checks['pass'].all():
        raise ValueError(f'Historical raw-median probability drift: {checks.to_dict("records")}')
    return checks
