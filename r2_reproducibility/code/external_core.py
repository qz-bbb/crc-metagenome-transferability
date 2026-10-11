"""Small, tested boundaries for a train-locked external cohort evaluation."""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score


def match_training(old, public):
    if old.sample_uid.duplicated().any() or old.sample_id.duplicated().any():
        raise ValueError('Original sample identifiers are not unique')
    candidates=public[public['Sample Source ID'].isin(old.sample_id)].copy()
    if candidates['Sample Source ID'].duplicated().any():
        raise ValueError('Ambiguous public sample ID match')
    joined=old.merge(candidates,left_on='sample_id',right_on='Sample Source ID',how='left',validate='one_to_one')
    joined['exclusion_reason']=''
    missing=joined.source_stem.isna()
    conflict=(joined.study_condition.str.lower()!=joined['Study condition'].str.lower()) & ~missing
    joined.loc[missing,'exclusion_reason']='profile_unavailable'
    joined.loc[conflict,'exclusion_reason']='label_conflict'
    # Retain first original-order specimen per study-qualified subject, if needed.
    eligible=joined.exclusion_reason.eq('')
    subject=joined.subject_id.fillna(joined.sample_id).astype(str)
    key=joined.study_name.astype(str)+'::'+subject
    dup=key[eligible].duplicated()
    joined.loc[dup[dup].index,'exclusion_reason']='duplicate_subject'
    return joined[joined.exclusion_reason.eq('')].copy(), joined


def require_numeric(x):
    arr=np.asarray(x,dtype=float)
    if not np.isfinite(arr).all() or (arr<0).any():
        raise ValueError('Nonfinite or negative abundance; never silently impute')
    return arr


def assert_finite_statistics(frame):
    numeric=frame.select_dtypes(include=[np.number])
    if not np.isfinite(numeric.to_numpy()).all():
        raise ValueError('Nonfinite association result; BH family must not silently shrink')


def attach_labels(pred,meta,expected_counts,expected_models):
    if meta['Sample Source ID'].duplicated().any(): raise ValueError('Duplicate clinical ID')
    if meta['Study condition'].value_counts().to_dict()!=expected_counts:
        raise ValueError('Clinical group counts changed')
    if pred.duplicated(['sample_id','panel','classifier']).any(): raise ValueError('Duplicate model/sample prediction')
    keys=pred.panel+'::'+pred.classifier
    if set(keys)!=set(expected_models): raise ValueError('Incomplete model grid')
    ids=set(meta['Sample Source ID'])
    for k in expected_models:
        if set(pred.loc[keys.eq(k),'sample_id'])!=ids: raise ValueError('Incomplete sample grid')
    if not np.isfinite(pred.probability_crc).all() or not pred.probability_crc.between(0,1).all():
        raise ValueError('Invalid probabilities')
    return pred.merge(meta[['Sample Source ID','Study condition']],left_on='sample_id',right_on='Sample Source ID',validate='many_to_one')


def clr(x,pseudocount):
    a=require_numeric(x)
    if not np.isfinite(pseudocount) or pseudocount<=0:
        raise ValueError('Invalid pseudocount')
    logs=np.log(a+pseudocount)
    return logs-logs.mean(axis=1,keepdims=True)


def fit_preprocess(x,studies,prevalence=.05,min_studies=6):
    arr=require_numeric(x)
    studies=np.asarray(studies)
    if len(studies)!=len(x) or x.columns.duplicated().any():
        raise ValueError('Invalid training dimensions')
    present=arr>0
    prev=present.mean(axis=0)
    ns=np.sum([present[studies==s].any(axis=0) for s in np.unique(studies)],axis=0)
    selected=(prev>=prevalence)&(ns>=min_studies)
    if not selected.any(): raise ValueError('No training features pass')
    train=arr[:,selected]
    p=float(train[train>0].min()/2)
    state={'features':list(x.columns[selected]),'pseudocount':p,
           'prevalence_threshold':prevalence,'minimum_studies':min_studies,
           'prevalence':prev[selected].tolist(),'detected_studies':ns[selected].astype(int).tolist()}
    return state,clr(train,p)


def transform_external(x,state,same_pipeline=False):
    require_numeric(x)
    if x.columns.duplicated().any(): raise ValueError('Duplicate feature names')
    missing=[f for f in state['features'] if f not in x.columns]
    if missing and not same_pipeline:
        raise ValueError('Cross-version missing features are not biological zeros')
    aligned=x.reindex(columns=state['features'],fill_value=0.)
    coverage={'n_features':len(state['features']), 'n_exact_rows':len(state['features'])-len(missing),
              'not_reported_features':missing, 'policy':'same-source-pipeline non-detection only'}
    return clr(aligned,state['pseudocount']),coverage


def matched_random(taxa,prev,targets,excluded,seed):
    rng=np.random.default_rng(seed); prev=np.asarray(prev); choices=[]
    available={i for i,t in enumerate(taxa) if t not in excluded}
    for target in targets:
        if not available: raise ValueError('Insufficient size-matched random candidates')
        index=taxa.index(target); ix=np.array(sorted(available))
        d=np.abs(prev[ix]-prev[index]); pick=int(rng.choice(ix[d==d.min()]))
        choices.append(taxa[pick]); available.remove(pick)
    return choices


def bootstrap_metrics(y,predictions,n_boot=2000,seed=20261010):
    y=np.asarray(y)
    if set(np.unique(y))!={0,1}: raise ValueError('Binary labels required')
    for p in predictions.values():
        if len(p)!=len(y) or not np.isfinite(p).all(): raise ValueError('Invalid predictions')
    point={k:(float(roc_auc_score(y,p)),float(average_precision_score(y,p))) for k,p in predictions.items()}
    boot={k:np.empty((n_boot,2)) for k in predictions}
    rng=np.random.default_rng(seed); groups=[np.flatnonzero(y==i) for i in (0,1)]
    for b in range(n_boot):
        ix=np.concatenate([rng.choice(g,len(g),replace=True) for g in groups])
        yy=y[ix]
        for k,p in predictions.items():
            pp=np.asarray(p)[ix]
            boot[k][b]=roc_auc_score(yy,pp),average_precision_score(yy,pp)
    return {'point':point,'bootstrap':boot}

