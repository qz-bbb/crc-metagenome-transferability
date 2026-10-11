"""Pure fixed-rule helpers. No filesystem or test-cohort labels in selection."""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

def eligibility(stats):
    a=stats.copy()
    a['A']=a.q.lt(.1)
    a['D']=a.direction_agreement.ge(.6)
    a['H']=a.re_q.lt(.1)&a.i2.lt(75)
    a['R']=a.inner_sign_agreement.ge(.9)&a.inner_q_fraction.ge(.8)
    a['AD']=a.A&a.D; a['AH']=a.A&a.H; a['AR']=a.A&a.R
    a['ICP']=a.A&a.D&a.H&a.R
    a['all_failed_rules']=[';'.join(k for k in ['A','D','H','R'] if not bool(r[k])) for _,r in a.iterrows()]
    a['first_failed_waterfall_rule']=[next((k for k in ['A','D','H','R'] if not bool(r[k])),'retained') for _,r in a.iterrows()]
    a['association_rank']=a.q.rank(method='min')
    return a

def panels(stats):
    a=eligibility(stats).sort_values(['q','taxon'],na_position='last')
    n=min(18,int(a.ICP.sum()))
    return {k:a.loc[a[k],'taxon'].head(n).tolist() for k in ['A','AD','AH','AR','ICP']},n

def abundance_summary(raw,y,taxa):
    raw=np.asarray(raw,float); y=np.asarray(y); out={'taxon':list(taxa)}
    for label,c in [('crc',1),('control',0)]:
        x=raw[y==c]; positive=(x>0).sum(axis=0)
        out[f'n_{label}']=np.repeat(len(x),len(taxa))
        out[f'detection_fraction_{label}']=(x>0).mean(axis=0)
        out[f'positive_mean_{label}']=np.divide(x.sum(axis=0),positive,out=np.full(len(taxa),np.nan),where=positive>0)
        out[f'median_{label}']=np.median(x,axis=0)
    a=pd.DataFrame(out); a['raw_median_difference']=a.median_crc-a.median_control
    a['zero_raw_median_difference']=a.raw_median_difference.eq(0)
    a['both_medians_zero']=a.median_crc.eq(0)&a.median_control.eq(0)
    return a

def bootstrap_metrics(y,predictions,n_boot=2000,seed=20261010):
    y=np.asarray(y); classes=[np.flatnonzero(y==c) for c in [0,1]]
    if min(map(len,classes))==0: raise ValueError('Both classes required')
    point={k:np.array([roc_auc_score(y,p),average_precision_score(y,p)]) for k,p in predictions.items()}
    result={k:np.empty((n_boot,2)) for k in predictions}; rng=np.random.default_rng(seed)
    for b in range(n_boot):
        ix=np.concatenate([rng.choice(c,len(c),replace=True) for c in classes])
        for k,p in predictions.items():
            result[k][b]=roc_auc_score(y[ix],np.asarray(p)[ix]),average_precision_score(y[ix],np.asarray(p)[ix])
    return point,result
