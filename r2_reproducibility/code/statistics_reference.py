#!/usr/bin/env python
"""Train-only preprocessing and nested LOSO sensitivity analysis.

This script is intentionally standalone and conservative.  It uses only the
public cMD relative-abundance matrix and metadata; no external cohort is read.
The outer held-out study is never used for filtering, CLR, panel selection, or
model fitting.  The resulting analysis is a sensitivity analysis and does not
replace the globally locked primary analysis.
"""
from __future__ import annotations
import argparse, gzip, hashlib, json, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import chi2, norm
from statsmodels.stats.multitest import multipletests
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler

BASE = Path('UNCONFIGURED_INPUT')
MATRIX = BASE / "curated_crc_case_control_species_relative_abundance.tsv.gz"
META = BASE / "curated_crc_case_control_sample_metadata.tsv"
OUT = Path('UNCONFIGURED_OUTPUT')
SEED = 20261009
RF_SEED = 20260817
BOOT = 2000

def sha256(p: Path) -> str:
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()

def read_matrix(path: Path) -> pd.DataFrame:
    with gzip.open(path, 'rt') if str(path).endswith('.gz') else path.open('r') as f:
        x=pd.read_csv(f, sep='\t')
    if x.columns[0] != 'taxon_long': x=x.rename(columns={x.columns[0]:'taxon_long'})
    x=x.set_index('taxon_long')
    return x.apply(pd.to_numeric, errors='coerce').fillna(0.0)

def bh(q):
    out=np.full(len(q), np.nan)
    ok=np.isfinite(q)
    if ok.any(): out[ok]=multipletests(np.asarray(q)[ok], method='fdr_bh')[1]
    return out

def clr_transform(a, p):
    z=np.log(np.asarray(a, float)+p)
    return z-z.mean(axis=1, keepdims=True)

def association(x, y, studies):
    """OLS CLR feature ~ CRC + study indicators, HC3 p values."""
    d=pd.get_dummies(pd.Series(studies, dtype='category'), drop_first=True, dtype=float)
    z=np.column_stack([np.ones(len(y)), np.asarray(y,float), d.to_numpy()])
    co=[]; pv=[]
    for j in range(x.shape[1]):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                r=sm.OLS(x[:,j], z).fit(cov_type='HC3')
            co.append(float(r.params[1])); pv.append(float(r.pvalues[1]))
        except Exception:
            co.append(np.nan); pv.append(np.nan)
    q=bh(np.asarray(pv))
    return pd.DataFrame({'taxon':np.arange(x.shape[1]),'coef':co,'p':pv,'q':q})

def study_effects(x, y, studies):
    """Return study-level CRC-control mean differences and sampling variances."""
    rows=[]; variances=[]
    for s in pd.unique(studies):
        ix=np.asarray(studies)==s
        cv=x[ix & (y==1)]; kv=x[ix & (y==0)]
        if len(cv)==0 or len(kv)==0:
            continue
        rows.append(cv.mean(axis=0)-kv.mean(axis=0))
        vc=np.var(cv,axis=0,ddof=1) if len(cv)>1 else np.zeros(x.shape[1])
        vk=np.var(kv,axis=0,ddof=1) if len(kv)>1 else np.zeros(x.shape[1])
        variances.append(np.maximum(vc/len(cv)+vk/len(kv),1e-12))
    return np.asarray(rows), np.asarray(variances)

def random_effects(effects, variances):
    """DerSimonian-Laird pooled-effect p values and I2 per feature."""
    k,p=effects.shape; pv=np.full(p,np.nan); i2=np.full(p,np.nan)
    for j in range(p):
        e=effects[:,j]; v=np.maximum(variances[:,j],1e-12); w=1/v; mu=np.sum(w*e)/np.sum(w)
        Q=np.sum(w*(e-mu)**2); df=max(k-1,1)
        c=np.sum(w)-np.sum(w*w)/np.sum(w)
        tau2=max(0.0,(Q-df)/c) if c>0 else 0.0
        wr=1.0/(v+tau2); pooled=np.sum(wr*e)/np.sum(wr); se=np.sqrt(1.0/np.sum(wr))
        pv[j]=2*norm.sf(abs(pooled/se)) if se>0 else np.nan
        i2[j]=max(0.0,(Q-df)/Q)*100 if Q>0 else 0.0
    return pv,i2

def panel_selection(x, y, studies, taxa, raw=None, min_studies=6):
    # Association is evaluated on CLR coordinates, while the locked primary
    # DCP direction criterion uses within-study medians of untransformed
    # relative abundance.  Keeping these scales explicit prevents the nested
    # ICP from silently becoming a CLR-direction variant of the DCP.
    a=association(x,y,studies); effects, variances=study_effects(x,y,studies)
    if raw is None:
        raw=x
    raw_effects=[]
    for s in pd.unique(studies):
        ix=np.asarray(studies)==s
        cv=raw[ix & (y==1)]; kv=raw[ix & (y==0)]
        if len(cv)==0 or len(kv)==0:
            continue
        raw_effects.append(np.median(cv,axis=0)-np.median(kv,axis=0))
    raw_effects=np.asarray(raw_effects)
    global_sign=np.sign(a.coef.to_numpy()); global_sign[global_sign==0]=np.nan
    suniq=list(pd.unique(studies)); agreement=[]
    for j,sgn in enumerate(global_sign):
        dirs=np.sign(raw_effects[:,j]); eligible=np.isfinite(dirs)
        denom=len(dirs)  # zero differences remain discordant
        agreement.append(np.sum(dirs==sgn)/denom if denom else np.nan)
    agreement=np.asarray(agreement); a['direction_agreement']=agreement
    a['taxon']=taxa
    dcp=a[(a.q<0.10)&(a.direction_agreement>=0.60)].sort_values(['q','taxon'])
    # ICP: leave-one-study-out inner association sign/q fractions + DL heterogeneity
    signs=[[] for _ in range(x.shape[1])]; qpass=[[] for _ in range(x.shape[1])]
    for held in suniq:
        keep=np.asarray(studies)!=held
        if len(np.unique(y[keep]))<2 or len(pd.unique(np.asarray(studies)[keep]))<2: continue
        # BH is applied across all features within each inner training fold,
        # rather than to a one-feature model.
        aa=association(x[keep],y[keep],np.asarray(studies)[keep])
        for j in range(x.shape[1]):
            signs[j].append(np.sign(aa.coef.iloc[j])); qpass[j].append(bool(aa.q.iloc[j]<0.10))
    signs=[np.nanmean(np.asarray(v)==global_sign[j]) if v else np.nan for j,v in enumerate(signs)]
    qpass=[np.nanmean(v) if v else np.nan for v in qpass]
    re_p,i2=random_effects(effects,variances); a['inner_sign_agreement']=signs; a['inner_q_fraction']=qpass; a['re_q']=bh(re_p); a['re_p']=re_p; a['i2']=i2
    icp=a[(a.q<0.10)&(a.direction_agreement>=0.60)&(a.inner_sign_agreement>=0.90)&(a.inner_q_fraction>=0.80)&(a.re_q<0.10)&(a.i2<75)].sort_values(['q','taxon'])
    arp=a.sort_values(['q','taxon'])
    return {'ARP29':list(arp.taxon.iloc[:29]),'ARP18':list(arp.taxon.iloc[:18]),
            'DCP29':list(dcp.taxon.iloc[:29]),'DCP18':list(dcp.taxon.iloc[:18]),
            'ICP18':list(icp.taxon.iloc[:18]), 'all':list(taxa), '_stats':a}

def metrics(y,p,seed):
    r={'auroc':np.nan,'auprc':np.nan,'auroc_lo':np.nan,'auroc_hi':np.nan,'auprc_lo':np.nan,'auprc_hi':np.nan,'bootstrap_attempted':0,'bootstrap_valid':0}
    if len(np.unique(y))<2:return r
    r['auroc']=roc_auc_score(y,p); r['auprc']=average_precision_score(y,p)
    rng=np.random.default_rng(seed); vals=[]
    y=np.asarray(y); p=np.asarray(p); classes=np.unique(y)
    class_ix={c:np.flatnonzero(y==c) for c in classes}
    for _ in range(BOOT):
        r['bootstrap_attempted'] += 1
        # Stratified resampling preserves the observed case/control support;
        # ties in the prediction scores are handled by sklearn metrics.
        ix=np.concatenate([rng.choice(class_ix[c], size=len(class_ix[c]), replace=True) for c in classes])
        rng.shuffle(ix); yy=y[ix]; pp=p[ix]
        vals.append((roc_auc_score(yy,pp),average_precision_score(yy,pp)))
        r['bootstrap_valid'] += 1
    if vals:
        v=np.asarray(vals); r['auroc_lo'],r['auroc_hi']=np.quantile(v[:,0],[.025,.975]); r['auprc_lo'],r['auprc_hi']=np.quantile(v[:,1],[.025,.975])
    return r

def tie_unit_test():
    y=np.array([0,0,1,1]); p=np.array([.5,.5,.5,.5]); m=metrics(y,p,1)
    assert abs(m['auroc']-.5)<1e-12 and abs(m['auprc']-.5)<1e-12

def fit_predict(xtr,ytr,xte,kind):
    scaler=StandardScaler(); a=scaler.fit_transform(xtr); b=scaler.transform(xte)
    if kind=='RF':
        m=RandomForestClassifier(n_estimators=500,max_features='sqrt',max_depth=None,min_samples_leaf=2,bootstrap=True,class_weight='balanced_subsample',random_state=RF_SEED,n_jobs=1)
    else:
        m=LogisticRegression(penalty='elasticnet',solver='saga',l1_ratio=.5,C=.5,class_weight='balanced',max_iter=2000,random_state=42,n_jobs=1)
    with warnings.catch_warnings(): warnings.simplefilter('ignore'); m.fit(a,ytr)
    return m.predict_proba(b)[:,1]

def main():
    global BOOT
    ap=argparse.ArgumentParser(); ap.add_argument('--smoke',action='store_true'); ap.add_argument('--bootstrap',type=int,default=2000); args=ap.parse_args()
    BOOT=args.bootstrap
    OUT.mkdir(parents=True,exist_ok=True); started=time.time(); tie_unit_test()
    mat=read_matrix(MATRIX); meta=pd.read_csv(META,sep='\t').set_index('sample_uid')
    common=[s for s in mat.columns if s in meta.index]; mat=mat[common]; meta=meta.loc[common]
    keep=meta['study_condition'].isin(['CRC','control']); mat=mat.loc[:,keep]; meta=meta.loc[keep]
    y=(meta.study_condition=='CRC').astype(int).to_numpy(); studies=meta.study_name.to_numpy(); taxa=mat.index.to_numpy(); arr=mat.to_numpy().T
    outs=[]; panels=[]; preds=[]; errors=[]; folds=list(pd.unique(studies));
    if args.smoke: folds=folds[:1]
    for fold_i,held in enumerate(folds):
        try:
            tr=studies!=held; te=~tr; xtr0=arr[tr]; xte0=arr[te]; ytr=y[tr]; yte=y[te]; str_=studies[tr]
            prev=(xtr0>0).mean(axis=0); ns=np.array([len(set(str_[xtr0[:,j]>0])) for j in range(arr.shape[1])])
            sel=(prev>=.05)&(ns>=6); selected=taxa[sel]; xtr0=xtr0[:,sel]; xte0=xte0[:,sel]
            positive=xtr0[xtr0>0]; p=float(np.min(positive)/2) if len(positive) else 1e-12
            xtr=clr_transform(xtr0,p); xte=clr_transform(xte0,p)
            ps=panel_selection(xtr,ytr,str_,selected,raw=xtr0); stats=ps.pop('_stats');
            # Prevalence-matched random18: match the nested ICP18 prevalence
            # profile, exclude every evaluated candidate panel, and resolve
            # equal-distance ties by a seeded draw. This is a fixed-size
            # negative control, not a tuned competitor.
            rng=np.random.default_rng(SEED+fold_i); candidates=set(range(len(selected))); chosen=[]
            target_idx=[int(np.where(selected==t)[0][0]) for t in ps['ICP18']]
            excluded=set()
            for pn in ('ARP18','DCP18','ICP18'):
                excluded.update(int(np.where(selected==t)[0][0]) for t in ps[pn])
            candidates-=excluded
            for target in target_idx:
                if not candidates: break
                cc=np.array(sorted(candidates)); dist=np.abs(prev[sel][cc]-prev[sel][target])
                best=cc[np.flatnonzero(dist==dist.min())]
                pick=int(rng.choice(best)); chosen.append(pick); candidates.remove(pick)
            random_taxa=[selected[i] for i in chosen]
            panel_ids={'nested_all_filtered':ps['all'],'ARP29':ps['ARP29'],'DCP29':ps['DCP29'],'ARP18':ps['ARP18'],'DCP18':ps['DCP18'],'ICP':ps['ICP18'],'prevalence_matched_random_nICP':random_taxa}
            for panel,panel_taxa in panel_ids.items():
                idx=[int(np.where(selected==t)[0][0]) for t in panel_taxa if t in set(selected)]
                if not idx:
                    # Preserve an explicit empty-panel row so a fold with no
                    # train-only ICP candidates is visible, not silently lost.
                    for model in ('ElasticNet','RF'):
                        outs.append({'held_out_study':held,'model':model,'panel':panel,'panel_size':0,'n_train':int(tr.sum()),'n_test':int(te.sum()),'nested_preprocessing':True,'status':'empty_panel'})
                    panels.append({'held_out_study':held,'panel':panel,'panel_size':0,'taxa':'','selected_features':len(selected)})
                    continue
                for model in ('ElasticNet','RF'):
                    pr=fit_predict(xtr[:,idx],ytr,xte[:,idx],'RF' if model=='RF' else 'EN'); mm=metrics(yte,pr,SEED+fold_i)
                    row={'held_out_study':held,'model':model,'panel':panel,'panel_size':len(idx),'n_train':int(tr.sum()),'n_test':int(te.sum()),'n_train_studies':len(set(str_)),'prevalence_min':.05,'study_min':6,'pseudocount':p,'nested_preprocessing':True,**mm}; outs.append(row)
                    for sample,yy,pp in zip(meta.index[te],yte,pr): preds.append({'held_out_study':held,'model':model,'panel':panel,'sample_uid':sample,'y':int(yy),'pred':float(pp)})
                panels.append({'held_out_study':held,'panel':panel,'panel_size':len(idx),'taxa':'||'.join(panel_taxa),'selected_features':len(selected)})
            stats.to_csv(OUT/f'fold_{fold_i:02d}_association.tsv',sep='\t',index=False)
        except Exception as e: errors.append({'held_out_study':held,'error':repr(e)})
    pd.DataFrame(outs).to_csv(OUT/'nested_loso_metrics.tsv',sep='\t',index=False); pd.DataFrame(preds).to_csv(OUT/'nested_loso_predictions.tsv.gz',sep='\t',index=False,compression='gzip'); pd.DataFrame(panels).to_csv(OUT/'nested_loso_panel_membership.tsv',sep='\t',index=False); pd.DataFrame(errors).to_csv(OUT/'nested_loso_errors.tsv',sep='\t',index=False)
    # Compact panel-level audit makes the actual (up-to-18) train-only ICP
    # candidate count explicit and prevents a capped panel name from implying
    # that every fold contained exactly 18 eligible taxa.
    ptab=pd.DataFrame(panels)
    summary=[]
    for pn,g in ptab.groupby('panel',dropna=False):
        sizes=pd.to_numeric(g.panel_size,errors='coerce')
        summary.append({'panel':pn,'folds':int(len(g)),'min_size':int(sizes.min()),'max_size':int(sizes.max()),'mean_size':float(sizes.mean()),'nonempty_folds':int((sizes>0).sum())})
    pd.DataFrame(summary).to_csv(OUT/'nested_loso_panel_summary.tsv',sep='\t',index=False)
    panel_names=['nested_all_filtered','ARP29','DCP29','ARP18','DCP18','ICP','prevalence_matched_random_nICP']
    expected_metric_rows=len(folds)*len(panel_names)*2
    observed_metric_rows=len(outs)
    expected_membership_rows=len(folds)*len(panel_names)
    observed_membership_rows=len(panels)
    metric_keys={(r.get('held_out_study'),r.get('model'),r.get('panel')) for r in outs}
    expected_keys={(held,model,panel) for held in folds for model in ('ElasticNet','RF') for panel in panel_names}
    missing_keys=sorted(expected_keys-metric_keys)
    receipt={'status':'PASS' if (not errors and observed_metric_rows==expected_metric_rows and observed_membership_rows==expected_membership_rows and not missing_keys) else 'PARTIAL','smoke':args.smoke,'folds':len(folds),'samples':int(len(y)),'studies':len(set(studies)),'features_raw':int(len(taxa)),'matrix_sha256':sha256(MATRIX),'metadata_sha256':sha256(META),'bootstrap_replicates':BOOT,'preprocessing':'train-only prevalence >=0.05 and >=6 studies; pseudocount=min train positive/2; CLR applied with train-selected features and train-derived pseudocount; OLS HC3/BH; DCP directions use within-study median relative abundance with exact zero discordant; ICP uses inner leave-one-study association, DerSimonian-Laird q and I2','panel_names':panel_names,'expected_metric_rows':expected_metric_rows,'observed_metric_rows':observed_metric_rows,'expected_membership_rows':expected_membership_rows,'observed_membership_rows':observed_membership_rows,'missing_metric_keys':missing_keys,'bootstrap_valid_min':int(min([r.get('bootstrap_valid',0) for r in outs] or [0])),'bootstrap_valid_max':int(max([r.get('bootstrap_valid',0) for r in outs] or [0])),'models':{'RF':{'n_estimators':500,'max_features':'sqrt','min_samples_leaf':2,'class_weight':'balanced_subsample','random_state':RF_SEED},'ElasticNet':{'solver':'saga','penalty':'elasticnet','l1_ratio':.5,'C':.5,'random_state':42},},'elapsed_seconds':time.time()-started,'errors':errors}
    (OUT/'nested_loso_receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    (OUT/'nested_loso_claim_boundary.md').write_text('# Claim boundary\n\nThis is a train-only preprocessing LOSO sensitivity analysis. The DCP layer uses the locked within-study median relative-abundance direction rule; ICP is recomputed inside each outer training fold with inner leave-one-study association robustness and random-effects heterogeneity. The observed ICP size is capped at 18 but may be smaller in a fold; the prevalence-matched negative control is matched to the observed ICP size and prevalence profile. It is not independent external validation and does not replace the globally locked primary analysis.\n',encoding='utf-8')

if __name__=='__main__': main()

