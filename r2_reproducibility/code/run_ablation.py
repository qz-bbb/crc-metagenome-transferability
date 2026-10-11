"""Checkpointed post-outcome explanatory ablations; does not alter older runs."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[k]='1'
import argparse, hashlib, importlib.util, importlib.metadata, json, platform, sys, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from ablation_core import eligibility, panels, abundance_summary, bootstrap_metrics
from exact_tsv import read_exact

# Populated exclusively by closures.py before any computation.
HERE=Path(__file__).resolve().parent
NESTED=MATRIX=META=None

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def save(p,x):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf8')
def table(p,x):
    p.parent.mkdir(parents=True,exist_ok=True);x.to_csv(p,sep='\t',index=False,float_format='%.17g')
def reference():
    s=importlib.util.spec_from_file_location('nested_reference',Path(__file__).parent/'statistics_reference.py')
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def load():
    m=reference(); a=m.read_matrix(MATRIX); meta=pd.read_csv(META,sep='\t').set_index('sample_uid')
    cols=[c for c in a.columns if c in meta.index];meta=meta.loc[cols];a=a[cols]
    k=meta.study_condition.isin(['CRC','control']);return m,a.loc[:,k],meta.loc[k]

def init():
    if (HERE/'start_receipt.json').exists():raise ValueError('Already frozen; use fold/finalize')
    old=json.loads((NESTED/'nested_loso_receipt.json').read_text())
    # closures.begin verifies the exact canonical source matrix when a faithful
    # reconstruction has a different gzip wrapper/decimal representation.
    assert sha(META)==old['metadata_sha256']
    paths=[MATRIX,META,Path(__file__).parent/'statistics_reference.py',NESTED/'nested_loso_receipt.json',NESTED/'nested_loso_panel_membership_long.tsv']
    paths += sorted(NESTED.glob('fold_*_association.tsv'))
    paths += [Path(__file__).parent/p for p in ['run_ablation.py','ablation_core.py','test_ablation_core.py']]
    paths=[p for p in paths if p.exists()]
    save(HERE/'start_receipt.json',{'analysis_status':'post-outcome explanatory','started_utc':pd.Timestamp.now(tz='UTC').isoformat(),
         'hashes':{str(p):sha(p) for p in paths},'python':sys.version,'platform':platform.platform(),
         'packages':{p:importlib.metadata.version(p) for p in ['numpy','pandas','scipy','statsmodels','scikit-learn','threadpoolctl']},
         'models_threads':1,'bootstrap':2000,'prior_outcomes_already_known':True})
    print('PROTOCOL AND SOURCES FROZEN',flush=True)

def check():
    r=json.loads((HERE/'start_receipt.json').read_text(encoding='utf8'))
    for p,h in r['hashes'].items():
        if sha(p)!=h:raise ValueError('Frozen input changed: '+p)

def fit(xtr,ytr,xte,kind):
    sc=StandardScaler(); a=sc.fit_transform(xtr);b=sc.transform(xte)
    if kind=='RF':m=RandomForestClassifier(n_estimators=500,max_features='sqrt',max_depth=None,min_samples_leaf=2,bootstrap=True,class_weight='balanced_subsample',random_state=20260817,n_jobs=1)
    else:m=LogisticRegression(penalty='elasticnet',solver='saga',l1_ratio=.5,C=.5,class_weight='balanced',max_iter=2000,random_state=42,n_jobs=1)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always');m.fit(a,ytr)
    return m.predict_proba(b)[:,1],[str(z.message) for z in w]

def fold(i):
    check(); dest=HERE/f'folds/{i:02d}'
    if (dest/'receipt.json').exists():raise ValueError('Completed fold exists; do not overwrite')
    t=time.time();m,mat,meta=load();studies=meta.study_name.to_numpy();held=list(pd.unique(studies))[i]
    tr=studies!=held;te=~tr; y=meta.study_condition.eq('CRC').astype(int).to_numpy()
    raw=mat.to_numpy().T; taxes=mat.index.to_numpy(); train=raw[tr];st=studies[tr]
    assert set(meta.index[tr]).isdisjoint(set(meta.index[te]));assert held not in set(st)
    preval=(train>0).mean(axis=0);ns=np.array([len(set(st[train[:,j]>0])) for j in range(raw.shape[1])]);f=(preval>=.05)&(ns>=6)
    taxa=taxes[f];train=train[:,f];test=raw[te][:,f];pseudo=float(train[train>0].min()/2)
    xtr=m.clr_transform(train,pseudo);xte=m.clr_transform(test,pseudo)
    with threadpool_limits(1):fresh=m.panel_selection(xtr,y[tr],st,taxa,raw=train)['_stats']
    saved=read_exact(NESTED/f'fold_{i:02d}_association.tsv')
    assert saved.taxon.tolist()==fresh.taxon.tolist()==list(taxa)
    differences={}
    for c in saved.columns[1:]:
        np.testing.assert_allclose(fresh[c],saved[c],atol=1e-9,rtol=1e-8,equal_nan=True)
        differences[c]=float(np.nanmax(abs(fresh[c].to_numpy()-saved[c].to_numpy())))
    ps,n=panels(fresh); flags=eligibility(fresh); flags['held_out_study']=held
    # Save full eligibility as well as size-truncated modeled membership.
    for k,v in ps.items():flags[k+'_selected']=flags.taxon.isin(v)
    if n:assert ps['A']==fresh.sort_values(['q','taxon']).head(n).taxon.tolist()
    flags['target_panel_size']=n; table(dest/'eligibility.tsv',flags)
    waterfall=pd.DataFrame([{'held_out_study':held,'stage':k,'n_eligible':int(v)} for k,v in [
      ('all_filtered',len(flags)),('A',flags.A.sum()),('A_D',flags.AD.sum()),('A_D_H',(flags.A&flags.D&flags.H).sum()),('A_D_H_R',flags.ICP.sum())]])
    table(dest/'waterfall.tsv',waterfall)
    spare=[]
    for s in pd.unique(st):
        ix=st==s;a=abundance_summary(train[ix],y[tr][ix],taxa);a['study_name']=s;a['outer_held_out_study']=held
        a['mean_clr_difference']=xtr[ix][y[tr][ix]==1].mean(axis=0)-xtr[ix][y[tr][ix]==0].mean(axis=0)
        a['training_reference_sign']=np.sign(fresh.coef.to_numpy());a['raw_direction_concordant']=np.sign(a.raw_median_difference)==a.training_reference_sign
        a['clr_direction_concordant']=np.sign(a.mean_clr_difference)==a.training_reference_sign
        spare.append(a)
    table(dest/'training_study_sparsity.tsv',pd.concat(spare,ignore_index=True))
    metrics=[];paired=[];predrows=[];warnings_out=[]
    if n:
      for kind in ['EN','RF']:
        pp={}
        for panel,members in ps.items():
            pos=[list(taxa).index(t) for t in members]
            with threadpool_limits(1):prob,w=fit(xtr[:,pos],y[tr],xte[:,pos],kind)
            warnings_out.extend([{'panel':panel,'model':kind,'warning':v} for v in w]);pp[panel]=prob
            for sid,yy,pr in zip(meta.index[te],y[te],prob):predrows.append({'held_out_study':held,'model':kind,'panel':panel,'sample_uid':sid,'y':int(yy),'pred':float(pr)})
        point,boot=bootstrap_metrics(y[te],pp,2000,20261010+i)
        for panel,arr in point.items():
            ci=np.quantile(boot[panel],[.025,.975],axis=0)
            metrics.append({'held_out_study':held,'model':kind,'panel':panel,'size':n,'eligible_candidates':int(flags[panel].sum()),'n_train':int(tr.sum()),'n_test':int(te.sum()),'auroc':arr[0],'ap':arr[1],'auroc_low':ci[0,0],'auroc_high':ci[1,0],'ap_low':ci[0,1],'ap_high':ci[1,1]})
            if panel!='A':
                delta=arr-point['A'];ci=np.quantile(boot[panel]-boot['A'],[.025,.975],axis=0)
                paired.append({'held_out_study':held,'model':kind,'panel_minus_A':panel,'size':n,'delta_auroc':delta[0],'delta_ap':delta[1],'auroc_low':ci[0,0],'auroc_high':ci[1,0],'ap_low':ci[0,1],'ap_high':ci[1,1]})
    table(dest/'metrics.tsv',pd.DataFrame(metrics));table(dest/'paired.tsv',pd.DataFrame(paired));table(dest/'predictions.tsv.gz',pd.DataFrame(predrows))
    # Empty warnings remain represented by a JSON list, not a malformed empty TSV.
    save(dest/'receipt.json',{'status':'PASS','held_out_study':held,'fold_index':i,'target_size':n,'n_features':len(taxa),'pseudocount':pseudo,
          'start_receipt_sha256':sha(HERE/'start_receipt.json'),'fresh_vs_cached_max_abs_differences':differences,'test_labels_used_for_selection':False,
          'train_test_ids_disjoint':True,'inner_preprocessing_scope':'fixed outer-training coordinates','n_bootstrap':2000,'warnings':warnings_out,
          'files':{p.name:sha(p) for p in dest.iterdir() if p.is_file()},'elapsed_seconds':time.time()-t})
    print(f'fold {i:02d} {held}: PASS; target {n}; {time.time()-t:.1f}s',flush=True)

def finalize():
    check(); dirs=[HERE/f'folds/{i:02d}' for i in range(11)]
    if not all((p/'receipt.json').exists() for p in dirs):raise ValueError('Incomplete 11 folds')
    for p in dirs:
        r=json.loads((p/'receipt.json').read_text());assert r['status']=='PASS'
        for f,h in r['files'].items():assert sha(p/f)==h
    for stem in ['metrics','paired','eligibility','waterfall','training_study_sparsity']:
        d=pd.concat([read_exact(p/(stem+'.tsv')) for p in dirs],ignore_index=True);table(HERE/(stem+'.tsv'),d)
    d=read_exact(HERE/'metrics.tsv');assert len(d)==110
    summary=d.groupby(['model','panel']).agg(folds=('auroc','size'),auroc_median=('auroc','median'),auroc_mean=('auroc','mean'),auroc_min=('auroc','min'),ap_median=('ap','median'),ap_mean=('ap','mean')).reset_index()
    table(HERE/'metrics_summary.tsv',summary)
    p=read_exact(HERE/'paired.tsv');table(HERE/'paired_summary.tsv',p.groupby(['model','panel_minus_A']).agg(folds=('delta_auroc','size'),median_delta_auroc=('delta_auroc','median'),mean_delta_auroc=('delta_auroc','mean'),median_delta_ap=('delta_ap','median'),auroc_wins=('delta_auroc',lambda z:int((z>0).sum()))).reset_index())
    print(summary.to_string(index=False),flush=True)
    save(HERE/'analysis_receipt.json',{'status':'PASS','folds':11,'metrics':110,'paired':len(p),'new_explanatory_post_outcome':True,
       'missing_folds':[],'heldout_used_in_selection':False,'finalized_utc':pd.Timestamp.now(tz='UTC').isoformat()})

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('action',choices=['init','fold','finalize']);a.add_argument('--fold',type=int);args=a.parse_args()
    {'init':init,'fold':lambda:fold(args.fold),'finalize':finalize}[args.action]()

