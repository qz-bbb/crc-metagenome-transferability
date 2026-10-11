"""Post-outcome identity-resolution sensitivity; fit/predict/evaluate separated."""
from pathlib import Path
import argparse, importlib.metadata, json, platform, sys, warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from identity_core import resolve_training, assert_disjoint_ids, paired_delta

# All paths populated exclusively by closures.py.
CODE=Path(__file__).resolve().parent
HERE=CODE
RUN=RUNS=OLD=MULTI=None
ENRICHED_MANIFEST=None
import run_external as old
from external_core import fit_preprocess, transform_external, matched_random, bootstrap_metrics, attach_labels, assert_finite_statistics
from run_multicohort import COHORTS
OUT=HERE/'results'; LOCK=HERE/'model_lock'; SEED=20261010

def save(name,value): old.save_json(HERE/name,value)
def table(name,value): old.table(HERE/name,value)

def check_lock():
    lock=json.loads((LOCK/'lock.json').read_text())
    for p,h in lock['files'].items():
        if old.sha(Path(p))!=h: raise ValueError('Sensitivity lock changed: '+p)
    return lock

def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    training=pd.read_csv(old.INP/'training_manifest.tsv',sep='\t',keep_default_na=False)
    enriched_path=ENRICHED_MANIFEST if ENRICHED_MANIFEST is not None else MULTI/'05_review/original_enriched_identity_manifest.tsv'
    enriched=pd.read_csv(enriched_path,sep='\t',keep_default_na=False)
    audit=resolve_training(training,enriched)
    if len(audit)!=1257 or int((~audit.accession_resolved).sum())!=43:
        raise ValueError('Identity count differs from audited expectation')
    table('results/training_identity_audit.tsv',audit)
    selected=audit[audit.accession_resolved].copy()
    table('results/training_manifest.tsv',selected)
    table('results/excluded_training.tsv',audit[~audit.accession_resolved])
    env={p:importlib.metadata.version(p) for p in ['numpy','pandas','scipy','statsmodels','scikit-learn','joblib']}
    inputs=[old.INP/'training_manifest.tsv',enriched_path,
      MULTI/'05_review/identity_audit_summary.json',Path(__file__),CODE/'identity_core.py',CODE/'test_identity.py',
      CODE/'run_external.py',CODE/'external_core.py',old.REFERENCE]
    inputs.extend(old.EXTRACTED/'metaphlan4_profiles'/f'{s}.tsv' for s in selected.source_stem.unique())
    save('input_manifest.json',{'files':{str(p):old.sha(p) for p in inputs},'python':sys.version,'os':platform.platform(),
      'packages':env,'selected_n':len(selected),'counts':selected.study_condition.value_counts().to_dict(),
      'excluded_n':43,'purpose':'post-outcome identity sensitivity; not proof of universal participant-level independence'})
    print('IDENTITY_AUDIT',len(selected),selected.study_condition.value_counts().to_dict(),flush=True)

def fit():
    if (LOCK/'lock.json').exists(): raise ValueError('Existing sensitivity lock is immutable')
    for p,h in json.loads((HERE/'input_manifest.json').read_text())['files'].items():
        if old.sha(Path(p))!=h: raise ValueError('Frozen input changed: '+p)
    train=pd.read_csv(OUT/'training_manifest.tsv',sep='\t',keep_default_na=False)
    blocks=[]
    for stem,g in train.groupby('source_stem',sort=True):
        x=old.species_profile(stem,g.sample_id).loc[g.sample_id]
        x.index=g.sample_uid.to_numpy(); blocks.append(x)
    universe=sorted(set().union(*(set(b.columns) for b in blocks)))
    raw=pd.concat([b.reindex(columns=universe,fill_value=0.) for b in blocks]).loc[train.sample_uid]
    state,x=fit_preprocess(raw,train.study_name.to_numpy())
    taxa=state['features']; retained=raw[taxa].to_numpy(); y=train.study_condition.eq('CRC').astype(int).to_numpy()
    ref=old.reference_module(); original=ref.association
    def guarded(*args,**kwargs):
        result=original(*args,**kwargs); assert_finite_statistics(result); return result
    ref.association=guarded
    candidate=ref.panel_selection(x,y,train.study_name.to_numpy(),np.array(taxa),raw=retained)
    stats=candidate.pop('_stats'); table('results/feature_evidence.tsv',stats)
    n=len(candidate['ICP18'])
    if not n or len(candidate['DCP29'])<n: raise ValueError('Cannot construct exact-size panels')
    panels={k:list(candidate[v]) for k,v in [('ARP29','ARP29'),('DCP29','DCP29'),('ICP','ICP18')]}
    panels['ARP_nICP']=panels['ARP29'][:n]; panels['DCP_nICP']=panels['DCP29'][:n]
    panels['random_nICP']=matched_random(taxa,state['prevalence'],panels['ICP'],set(sum(panels.values(),[])),SEED)
    save('results/preprocessing.json',state); save('results/panels.json',panels)
    prev=json.loads((old.LOCK/'panels.json').read_text()) if (old.LOCK/'panels.json').exists() else joblib.load(old.LOCK/'models.joblib')['panels']
    membership=[]
    for p,features in panels.items():
        a=set(prev[p]); b=set(features)
        membership.append({'panel':p,'baseline_n':len(a),'sensitivity_n':len(b),'intersection':len(a&b),
          'jaccard':len(a&b)/len(a|b),'lost':';'.join(sorted(a-b)),'gained':';'.join(sorted(b-a))})
    table('results/panel_membership_comparison.tsv',pd.DataFrame(membership))
    models={}; warning_rows=[]
    for panel,features in panels.items():
        ix=[taxa.index(t) for t in features]
        for kind in ['EN','RF']:
            clf=LogisticRegression(penalty='elasticnet',solver='saga',C=.5,l1_ratio=.5,class_weight='balanced',max_iter=2000,random_state=42) if kind=='EN' else RandomForestClassifier(n_estimators=500,max_features='sqrt',max_depth=None,min_samples_leaf=2,bootstrap=True,class_weight='balanced_subsample',random_state=20260817,n_jobs=1)
            model=Pipeline([('scale',StandardScaler()),('model',clf)])
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always'); model.fit(x[:,ix],y)
            if any(issubclass(w.category,ConvergenceWarning) for w in caught): raise ValueError('Model failed to converge')
            warning_rows.extend({'panel':panel,'classifier':kind,'warning':str(w.message)} for w in caught)
            models[panel+'::'+kind]={'pipeline':model,'indices':ix}
            print('FIT',panel,kind,len(ix),flush=True)
    LOCK.mkdir(exist_ok=True)
    joblib.dump({'state':state,'panels':panels,'models':models,'training_ids':train.sample_uid.tolist()},LOCK/'models.joblib')
    save('results/model_warnings.json',warning_rows)
    files=[LOCK/'models.joblib',OUT/'training_manifest.tsv',OUT/'feature_evidence.tsv',OUT/'panels.json',OUT/'preprocessing.json',HERE/'input_manifest.json']
    save('model_lock/lock.json',{'files':{str(p):old.sha(p) for p in files},'training_n':len(train),'feature_n':len(taxa),
      'panel_sizes':{k:len(v) for k,v in panels.items()},'external_data_used_in_fit':False,'seed':SEED})

def predict():
    check_lock(); bundle=joblib.load(LOCK/'models.joblib')
    train=pd.read_csv(OUT/'training_manifest.tsv',sep='\t',keep_default_na=False)
    records=[]; manifest=[]
    for cohort,(stem,_,role) in COHORTS.items():
        raw=old.species_profile(stem)
        assert_disjoint_ids(train.sample_id,raw.index)
        x,coverage=transform_external(raw,bundle['state'],same_pipeline=True)
        save(f'results/{cohort}_coverage.json',coverage)
        for key,obj in bundle['models'].items():
            panel,kind=key.split('::'); p=obj['pipeline'].predict_proba(x[:,obj['indices']])[:,1]
            records.extend({'cohort':cohort,'sample_id':sid,'panel':panel,'classifier':kind,'probability_crc':float(prob)} for sid,prob in zip(raw.index,p))
        path=old.EXTRACTED/'metaphlan4_profiles'/f'{stem}.tsv'
        manifest.append({'cohort':cohort,'role':role,'n_profile':len(raw),'path':str(path),'sha256':old.sha(path)})
        print('PREDICT',cohort,len(raw),flush=True)
    table('results/predictions.tsv',pd.DataFrame(records))
    save('results/prediction_receipt.json',{'source_profiles':manifest,'predictions_sha256':old.sha(OUT/'predictions.tsv'),
       'model_sha256':old.sha(LOCK/'models.joblib'),'labels_used_for_prediction':False})

def evaluate(cohort):
    check_lock(); bundle=joblib.load(LOCK/'models.joblib')
    receipt=json.loads((OUT/'prediction_receipt.json').read_text())
    if old.sha(OUT/'predictions.tsv')!=receipt['predictions_sha256']: raise ValueError('Prediction file changed')
    stem,counts,role=COHORTS[cohort]
    pred=pd.read_csv(OUT/'predictions.tsv',sep='\t'); pred=pred[pred.cohort.eq(cohort)]
    meta=pd.read_csv(old.EXTRACTED/'metadata'/f'{stem}.tsv',sep='\t',keep_default_na=False)
    labeled=attach_labels(pred,meta,counts,bundle['models'])
    selected=meta[meta['Study condition'].isin(['CRC','Control'])]
    ids=selected['Sample Source ID'].tolist(); y=selected['Study condition'].eq('CRC').astype(int).to_numpy()
    baseline=pd.read_csv(MULTI/'02_analysis/results'/cohort/'predictions_blind.tsv',sep='\t')
    preds={}
    for key in bundle['models']:
        p,k=key.split('::')
        a=pred[pred.panel.eq(p)&pred.classifier.eq(k)].set_index('sample_id').probability_crc
        b=baseline[baseline.panel.eq(p)&baseline.classifier.eq(k)].set_index('sample_id').probability_crc
        if set(a.index)!=set(meta['Sample Source ID']) or set(b.index)!=set(a.index): raise ValueError('External sample universe changed')
        preds['new::'+key]=a.loc[ids].to_numpy(); preds['old::'+key]=b.loc[ids].to_numpy()
    metrics=bootstrap_metrics(y,preds,2000,SEED)
    rows=[]
    for key in bundle['models']:
        panel,kind=key.split('::'); new=metrics['point']['new::'+key]; base=metrics['point']['old::'+key]
        ci=np.quantile(metrics['bootstrap']['new::'+key],[.025,.975],axis=0)
        dci=paired_delta(metrics['bootstrap']['new::'+key],metrics['bootstrap']['old::'+key])
        rows.append({'cohort':cohort,'role':role,'panel':panel,'classifier':kind,'n':len(y),'crc':int(y.sum()),'control':int((y==0).sum()),
          'auroc':new[0],'auroc_lower':ci[0,0],'auroc_upper':ci[1,0],'ap':new[1],'ap_lower':ci[0,1],'ap_upper':ci[1,1],
          'baseline_auroc':base[0],'baseline_ap':base[1],'auroc_delta':new[0]-base[0],'auroc_delta_lower':dci[0][0],
          'auroc_delta_upper':dci[1][0],'ap_delta':new[1]-base[1],'ap_delta_lower':dci[0][1],'ap_delta_upper':dci[1][1]})
    table(f'results/{cohort}_metrics.tsv',pd.DataFrame(rows))
    save(f'results/{cohort}_evaluation_receipt.json',{'bootstrap':2000,'seed':SEED,'paired_resampling':True,
      'clinical_metadata_sha256':old.sha(old.EXTRACTED/'metadata'/f'{stem}.tsv'),'n':len(y),
      'baseline_prediction_sha256':old.sha(MULTI/'02_analysis/results'/cohort/'predictions_blind.tsv')})
    print('EVALUATE',cohort,len(y),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('stage',choices=['prepare','fit','predict','evaluate']); parser.add_argument('--cohort',choices=list(COHORTS)); args=parser.parse_args()
    if args.stage=='evaluate':
        if not args.cohort: parser.error('--cohort required')
        evaluate(args.cohort)
    else: globals()[args.stage]()

