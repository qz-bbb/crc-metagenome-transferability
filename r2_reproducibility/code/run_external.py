"""Separate audit, train-lock, external prediction, and evaluation commands.

Reuses the prior R2 panel-selection implementation by a frozen hash. No external
matrix or label is read in fit(). Run from this local branch; never writes old runs.
"""
from pathlib import Path
import argparse, hashlib, importlib.util, importlib.metadata, json, platform, shutil, sys, time, warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from external_core import match_training, fit_preprocess, transform_external, matched_random, bootstrap_metrics, require_numeric, attach_labels, assert_finite_statistics

ROOT = Path('UNCONFIGURED_OUTPUT')
PROJECT=ROOT.parent
SOURCE = Path('UNCONFIGURED_INPUT')
EXTRACTED=SOURCE/'extracted'
BASEMETA = Path('UNCONFIGURED_METADATA')
REFERENCE = Path(__file__).with_name('statistics_reference.py')
EXT='This_study_cohort3__ONCOBIOME_IIGM_CZ'
INP=ROOT/'01_inputs'; OUT=ROOT/'02_analysis/results'; LOCK=ROOT/'02_analysis/model_lock'
SEED=20261010

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()

def save_json(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')

def stamp():
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()

def table(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    x.to_csv(p,sep='\t',index=False,float_format='%.17g')

def species_profile(stem,columns=None):
    p=EXTRACTED/'metaphlan4_profiles'/f'{stem}.tsv'
    df=pd.read_csv(p,sep='\t',index_col=0)
    if df.index.duplicated().any() or df.columns.duplicated().any():
        raise ValueError(f'Duplicate profile axis: {stem}')
    df=df.loc[df.index.str.rsplit('|',n=1).str[-1].str.startswith('s__')]
    if columns is not None: df=df.loc[:,list(columns)]
    require_numeric(df)
    totals=df.sum(axis=0)
    if not totals.between(99.9,100.1).all():
        raise ValueError(f'Species totals are not approximately percentages: {stem}, {totals.min()}, {totals.max()}')
    return df.T

def audit():
    INP.mkdir(parents=True,exist_ok=True)
    base=pd.read_csv(BASEMETA,sep='\t',keep_default_na=False)
    pieces=[]; paths=[BASEMETA,SOURCE/'zenodo_15069069.json',SOURCE/'piccinno_metadata_full.tar.gz',
                       SOURCE/'metaphlan4_profiles_full.tar.gz',REFERENCE,ROOT/'00_receipts/PROTOCOL.md']
    for p in sorted((EXTRACTED/'metadata').glob('*.tsv')):
        m=pd.read_csv(p,sep='\t',keep_default_na=False); m['source_stem']=p.stem
        pieces.append(m); paths.append(p)
    public=pd.concat(pieces,ignore_index=True)
    training, exclusions=match_training(base,public)
    table(INP/'training_manifest.tsv',training)
    table(INP/'training_selection_audit.tsv',exclusions)
    ext=public[public.source_stem.eq(EXT)].copy()
    if len(ext)!=124 or ext['Sample Source ID'].duplicated().any(): raise ValueError('External metadata mismatch')
    if set(ext['Sample Source ID']) & set(base.sample_id): raise ValueError('Train/external identifier overlap')
    table(INP/'external_metadata.tsv',ext)
    checks=[]
    for stem in sorted(set(training.source_stem)|{EXT}):
        profile=species_profile(stem)
        metadata=public[public.source_stem.eq(stem)]
        if set(profile.index)!=set(metadata['Sample Source ID']): raise ValueError(f'Metadata mismatch {stem}')
        paths.append(EXTRACTED/'metaphlan4_profiles'/f'{stem}.tsv')
        checks.append({'source_stem':stem,'n_samples':len(profile),'n_species_rows':profile.shape[1],
                       'percent_total_min':profile.sum(axis=1).min(),'percent_total_max':profile.sum(axis=1).max(),
                       'metadata_profile_id_match':True})
    table(INP/'profile_qc.tsv',pd.DataFrame(checks))
    source=json.loads((SOURCE/'zenodo_15069069.json').read_text(encoding='utf-8'))
    for name,local in [('metadata.tar.gz','piccinno_metadata_full.tar.gz'),('metaphlan4_profiles.tar.gz','metaphlan4_profiles_full.tar.gz')]:
        expected=next(f['checksum'] for f in source['files'] if f['key']==name).split(':')[1]
        actual=hashlib.md5((SOURCE/local).read_bytes()).hexdigest()
        if actual!=expected: raise ValueError('Archive checksum mismatch')
    table(INP/'source_manifest.tsv',pd.DataFrame([{'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)} for p in paths]))
    table(INP/'training_counts.tsv',training.groupby(['study_name','study_condition']).size().rename('n').reset_index())
    env={'python':sys.version,'executable':sys.executable,'platform':platform.platform(),
         'packages':{p:importlib.metadata.version(p) for p in ['numpy','pandas','scipy','statsmodels','scikit-learn','joblib']},
         'seed':SEED,'ai_use':'Codex substantive analysis coding; exact backend version not verifiably recorded; author independent rerun pending'}
    save_json(ROOT/'00_receipts/environment.json',env)
    summary={'created_utc':stamp(),'training_n':len(training),'training_studies':training.study_name.nunique(),
             'training_labels':training.study_condition.value_counts().to_dict(),
             'exclusion_counts':exclusions.exclusion_reason.value_counts().to_dict(),
             'external_n':len(ext),'external_labels':ext['Study condition'].value_counts().to_dict(),
             'source':'Zenodo 15069069; MetaPhlAn4 Jan21 --statq 0.1',
             'profile_qc':'PASS; finite nonnegative percentages and exact metadata-ID joins'}
    save_json(INP/'audit_summary.json',summary)
    print(json.dumps(summary,indent=2),flush=True)

def verify_sources():
    manifest=pd.read_csv(INP/'source_manifest.tsv',sep='\t')
    for row in manifest.itertuples():
        if sha(row.path)!=row.sha256: raise ValueError(f'Source changed: {row.path}')

def reference_module():
    spec=importlib.util.spec_from_file_location('r2_reference',REFERENCE)
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module

def fit():
    start=time.time(); verify_sources()
    if (LOCK/'lock.json').exists(): raise ValueError('Existing lock is immutable; do not refit after test evaluation')
    if shutil.disk_usage(ROOT).free < 600*1024**2: raise ValueError('Insufficient free disk for bounded fit')
    training=pd.read_csv(INP/'training_manifest.tsv',sep='\t',keep_default_na=False)
    blocks=[]; zeros=[]
    for stem,rows in training.groupby('source_stem',sort=True):
        p=species_profile(stem,rows.sample_id)
        p.index=rows.set_index('sample_id').loc[p.index,'sample_uid']
        blocks.append(p)
    # Same-database nonreported clades are zeros, not missing specimen profiles.
    universe=sorted(set().union(*(set(b.columns) for b in blocks)))
    for b in blocks:
        zeros.append({'study_name':training.set_index('sample_uid').loc[b.index[0],'study_name'],
                      'samples':len(b),'nonreported_union_taxa':len(set(universe)-set(b.columns))})
    raw=pd.concat([b.reindex(columns=universe,fill_value=0.) for b in blocks]).loc[training.sample_uid]
    state,x=fit_preprocess(raw,training.study_name.to_numpy())
    taxa=state['features']; retained=raw.loc[:,taxa].to_numpy(); y=training.study_condition.eq('CRC').astype(int).to_numpy()
    print(f'TRAIN ONLY: n={len(y)}, species_universe={len(universe)}, retained={len(taxa)}, pseudo={state["pseudocount"]}',flush=True)
    ref=reference_module()
    # Fail closed instead of allowing the historical helper to omit failed tests
    # from BH. The historical file remains unchanged.
    original_association=ref.association
    def strict_association(*args,**kwargs):
        result=original_association(*args,**kwargs)
        assert_finite_statistics(result)
        return result
    ref.association=strict_association
    panels=ref.panel_selection(x,y,training.study_name.to_numpy(),np.asarray(taxa),raw=retained)
    stats=panels.pop('_stats'); n=len(panels['ICP18'])
    assert_finite_statistics(stats)
    selected={'ARP29':panels['ARP29'],'DCP29':panels['DCP29'],'ICP':panels['ICP18']}
    if n:
        selected['ARP_nICP']=panels['ARP29'][:n]
        selected['DCP_nICP']=panels['DCP29'][:n]
        if len(selected['DCP_nICP'])!=n: raise ValueError('DCP cannot supply size-matched ICP comparator')
        excluded=set().union(*(set(v) for v in selected.values()))
        selected['random_nICP']=matched_random(taxa,state['prevalence'],selected['ICP'],excluded,SEED)
    LOCK.mkdir(parents=True,exist_ok=True)
    table(LOCK/'feature_evidence.tsv',stats)
    table(LOCK/'training_nonreported_rows.tsv',pd.DataFrame(zeros))
    table(LOCK/'feature_manifest.tsv',pd.DataFrame({'feature_order':range(len(taxa)),'taxon':taxa,
                                                  'prevalence':state['prevalence'],'detected_studies':state['detected_studies']}))
    table(LOCK/'panels.tsv',pd.DataFrame([{'panel':p,'rank':i+1,'taxon':t} for p,ts in selected.items() for i,t in enumerate(ts)]))
    save_json(LOCK/'preprocessing.json',state)
    bundle={'state':state,'panels':selected,'models':{},'training_ids':training.sample_uid.tolist()}
    fitlog=[]
    for panel,ts in selected.items():
        if not ts:
            fitlog.append({'panel':panel,'classifier':'none','n_features':0,'status':'unavailable; no threshold relaxation'})
            continue
        ix=[taxa.index(t) for t in ts]
        for kind in ['EN','RF']:
            if kind=='EN':
                model=LogisticRegression(penalty='elasticnet',solver='saga',C=.5,l1_ratio=.5,
                                        class_weight='balanced',max_iter=2000,random_state=42)
            else:
                model=RandomForestClassifier(n_estimators=500,max_features='sqrt',max_depth=None,min_samples_leaf=2,
                                              bootstrap=True,class_weight='balanced_subsample',random_state=20260817,n_jobs=1)
            pipe=make_pipeline(StandardScaler(),model)
            with warnings.catch_warnings(record=True) as ws:
                warnings.simplefilter('always'); pipe.fit(x[:,ix],y)
            convergence=[str(w.message) for w in ws if issubclass(w.category,ConvergenceWarning)]
            if convergence: raise RuntimeError(f'Convergence failed for {panel}/{kind}: {convergence}')
            bundle['models'][f'{panel}::{kind}']={'pipeline':pipe,'indices':ix}
            fitlog.append({'panel':panel,'classifier':kind,'n_features':len(ts),'status':'fitted',
                           'warnings':' | '.join(str(w.message) for w in ws),
                           'n_iterations':int(model.n_iter_[0]) if kind=='EN' else 500})
            print(f'FROZEN TRAIN MODEL {panel} {kind}: {len(ts)} features',flush=True)
    table(LOCK/'fit_log.tsv',pd.DataFrame(fitlog))
    joblib.dump(bundle,LOCK/'models.joblib',compress=3)
    core_files=[Path(__file__),Path(__file__).with_name('external_core.py'),REFERENCE,INP/'training_manifest.tsv',
                INP/'external_metadata.tsv',INP/'source_manifest.tsv',ROOT/'00_receipts/PROTOCOL.md']
    artifacts=[p for p in sorted(LOCK.glob('*')) if p.is_file()]
    lock={'locked_utc':stamp(),'training_n':len(y),'training_studies':training.study_name.nunique(),
          'training_crc':int(y.sum()),'training_control':int((1-y).sum()),'feature_count':len(taxa),
          'panel_sizes':{k:len(v) for k,v in selected.items()},'external_data_used_for_fitting':False,
          'algorithm':'same R2 training panel rule; new harmonized feature universe',
          'input_code_hashes':{str(p):sha(p) for p in core_files},
          'locked_artifact_hashes':{str(p.relative_to(ROOT)):sha(p) for p in artifacts},
          'elapsed_seconds':time.time()-start}
    save_json(LOCK/'lock.json',lock)
    print(json.dumps({k:v for k,v in lock.items() if 'hashes' not in k},indent=2),flush=True)

def check_lock():
    lock=json.loads((LOCK/'lock.json').read_text(encoding='utf-8'))
    for p,h in lock['input_code_hashes'].items():
        if sha(p)!=h: raise ValueError(f'Frozen input/code changed: {p}')
    for p,h in lock['locked_artifact_hashes'].items():
        if sha(ROOT/p)!=h: raise ValueError(f'Frozen model artifact changed: {p}')
    return lock

def predict():
    check_lock(); verify_sources(); OUT.mkdir(parents=True,exist_ok=True)
    bundle=joblib.load(LOCK/'models.joblib')
    raw=species_profile(EXT)
    x,coverage=transform_external(raw,bundle['state'],same_pipeline=True)
    rows=[]
    for key,entry in bundle['models'].items():
        panel,kind=key.split('::'); p=entry['pipeline'].predict_proba(x[:,entry['indices']])[:,1]
        for i,sid in enumerate(raw.index):
            rows.append({'sample_id':sid,'panel':panel,'classifier':kind,'probability_crc':p[i]})
    table(OUT/'external_predictions_blind.tsv',pd.DataFrame(rows))
    coverage['panel_coverage']={p:{'n_features':len(ts),'n_exact_rows':sum(t in raw.columns for t in ts),
                                  'nonreported':[t for t in ts if t not in raw.columns]} for p,ts in bundle['panels'].items()}
    save_json(OUT/'external_feature_coverage.json',coverage)
    save_json(OUT/'prediction_receipt.json',{'predicted_utc':stamp(),'lock_sha256':sha(LOCK/'lock.json'),
              'prediction_sha256':sha(OUT/'external_predictions_blind.tsv'),
              'n_samples':len(raw),'n_models':len(bundle['models']),
              'labels_read_by_predict_command':False,'models_retrained':False})
    print(json.dumps(coverage,indent=2),flush=True)

def evaluate():
    check_lock(); verify_sources()
    receipt=json.loads((OUT/'prediction_receipt.json').read_text(encoding='utf-8'))
    if receipt['lock_sha256']!=sha(LOCK/'lock.json') or receipt['prediction_sha256']!=sha(OUT/'external_predictions_blind.tsv'):
        raise ValueError('Prediction/lock receipt mismatch')
    pred=pd.read_csv(OUT/'external_predictions_blind.tsv',sep='\t')
    meta=pd.read_csv(INP/'external_metadata.tsv',sep='\t',keep_default_na=False)
    bundle=joblib.load(LOCK/'models.joblib')
    pred=attach_labels(pred,meta,{'CRC':59,'Control':46,'Adenoma':19},set(bundle['models']))
    table(OUT/'external_predictions_labeled.tsv',pred)
    metrics=[]; differences=[]
    for contrast,labels in [('CRC_vs_study_controls',['CRC','Control']),('CRC_vs_all_nonCRC_exploratory',['CRC','Control','Adenoma'])]:
        subset=pred[pred['Study condition'].isin(labels)].copy()
        subset['key']=subset.panel+'::'+subset.classifier
        wide=subset.pivot(index='sample_id',columns='key',values='probability_crc').sort_index()
        clinical=meta.set_index('Sample Source ID').loc[wide.index,'Study condition']
        y=clinical.eq('CRC').astype(int).to_numpy()
        if wide.isna().any().any(): raise ValueError('Incomplete prediction grid')
        bs=bootstrap_metrics(y,{k:wide[k].to_numpy() for k in wide},2000,SEED)
        for k,point in bs['point'].items():
            panel,kind=k.split('::'); ci=np.quantile(bs['bootstrap'][k],[.025,.975],axis=0)
            metrics.append({'contrast':contrast,'panel':panel,'classifier':kind,'n':len(y),'n_crc':int(y.sum()),
                            'n_noncrc':int((1-y).sum()),'auroc':point[0],'auroc_lower':ci[0,0],'auroc_upper':ci[1,0],
                            'auprc_ap':point[1],'auprc_lower':ci[0,1],'auprc_upper':ci[1,1],
                            'ap_null_prevalence':y.mean(),'bootstrap_replicates':2000})
        pairs=[]
        for kind in ['EN','RF']:
            for a,b in [('DCP29','ARP29'),('ICP','ARP29'),('ICP','ARP_nICP'),('DCP_nICP','ARP_nICP'),('ICP','random_nICP')]:
                pairs.append((f'{a}::{kind}',f'{b}::{kind}'))
        for panel in sorted(subset.panel.unique()): pairs.append((f'{panel}::RF',f'{panel}::EN'))
        for a,b in pairs:
            if a not in bs['point'] or b not in bs['point']: continue
            delta=np.asarray(bs['point'][a])-np.asarray(bs['point'][b]); ci=np.quantile(bs['bootstrap'][a]-bs['bootstrap'][b],[.025,.975],axis=0)
            differences.append({'contrast':contrast,'comparison':f'{a} minus {b}','delta_auroc':delta[0],
                                'delta_auroc_lower':ci[0,0],'delta_auroc_upper':ci[1,0],
                                'delta_auprc_ap':delta[1],'delta_auprc_lower':ci[0,1],'delta_auprc_upper':ci[1,1]})
        print(f'EVALUATED {contrast}: {len(y)} participants, 2000 paired bootstrap replicates',flush=True)
    table(OUT/'external_metrics.tsv',pd.DataFrame(metrics)); table(OUT/'paired_differences.tsv',pd.DataFrame(differences))
    save_json(OUT/'evaluation_receipt.json',{'evaluated_utc':stamp(),'lock_sha256':sha(LOCK/'lock.json'),
              'predictions_sha256':sha(OUT/'external_predictions_blind.tsv'),'metrics_sha256':sha(OUT/'external_metrics.tsv'),
              'labeled_predictions_sha256':sha(OUT/'external_predictions_labeled.tsv'),
              'paired_differences_sha256':sha(OUT/'paired_differences.tsv'),
              'metrics_rows':len(metrics),'paired_difference_rows':len(differences),
              'external_tuning':False,'conditional_ci_limitation':'Fitted models and one cohort held fixed; no training or between-cohort uncertainty'})
    print(pd.DataFrame(metrics).query("contrast=='CRC_vs_study_controls'")[['panel','classifier','auroc','auprc_ap']].to_string(index=False),flush=True)

def verify():
    check_lock(); verify_sources()
    bundle=joblib.load(LOCK/'models.joblib'); ext=species_profile(EXT)
    x,_=transform_external(ext,bundle['state'],same_pipeline=True)
    pred=pd.read_csv(OUT/'external_predictions_blind.tsv',sep='\t')
    receipt=json.loads((OUT/'evaluation_receipt.json').read_text(encoding='utf-8'))
    checks={'predictions_sha256':'external_predictions_blind.tsv','metrics_sha256':'external_metrics.tsv',
            'labeled_predictions_sha256':'external_predictions_labeled.tsv','paired_differences_sha256':'paired_differences.tsv'}
    if receipt['lock_sha256']!=sha(LOCK/'lock.json'): raise ValueError('Evaluation used a different lock')
    for key,name in checks.items():
        if receipt[key]!=sha(OUT/name): raise ValueError(f'Evaluation output changed: {name}')
    em=pd.read_csv(INP/'external_metadata.tsv',sep='\t')
    joined=attach_labels(pred,em,{'CRC':59,'Control':46,'Adenoma':19},set(bundle['models']))
    maxdiff=0.
    for key,entry in bundle['models'].items():
        panel,kind=key.split('::')
        saved=pred[(pred.panel==panel)&(pred.classifier==kind)].set_index('sample_id').loc[ext.index,'probability_crc'].to_numpy()
        replay=entry['pipeline'].predict_proba(x[:,entry['indices']])[:,1]
        diff=float(np.abs(saved-replay).max()); maxdiff=max(maxdiff,diff)
        if diff>1e-14: raise ValueError('Replay probability mismatch')
    from sklearn.metrics import roc_auc_score, average_precision_score
    labeled=pd.read_csv(OUT/'external_predictions_labeled.tsv',sep='\t'); metrics=pd.read_csv(OUT/'external_metrics.tsv',sep='\t')
    pd.testing.assert_frame_equal(joined,labeled,check_dtype=False)
    expected={(c,*k.split('::')) for c in ['CRC_vs_study_controls','CRC_vs_all_nonCRC_exploratory'] for k in bundle['models']}
    actual=set(map(tuple,metrics[['contrast','panel','classifier']].to_numpy()))
    if actual!=expected or len(metrics)!=len(expected): raise ValueError('Incomplete metric grid')
    delta=pd.read_csv(OUT/'paired_differences.tsv',sep='\t')
    expected_pairs=set()
    for c in ['CRC_vs_study_controls','CRC_vs_all_nonCRC_exploratory']:
        for kind in ['EN','RF']:
            for a,b in [('DCP29','ARP29'),('ICP','ARP29'),('ICP','ARP_nICP'),('DCP_nICP','ARP_nICP'),('ICP','random_nICP')]:
                if f'{a}::{kind}' in bundle['models'] and f'{b}::{kind}' in bundle['models']:
                    expected_pairs.add((c,f'{a}::{kind} minus {b}::{kind}'))
        for panel in bundle['panels']:
            if f'{panel}::RF' in bundle['models'] and f'{panel}::EN' in bundle['models']:
                expected_pairs.add((c,f'{panel}::RF minus {panel}::EN'))
    if set(map(tuple,delta[['contrast','comparison']].to_numpy()))!=expected_pairs or len(delta)!=len(expected_pairs):
        raise ValueError('Incomplete paired-comparison grid')
    indexed=metrics.set_index(['contrast','panel','classifier'])
    for row in delta.itertuples():
        a,b=row.comparison.split(' minus ')
        x=indexed.loc[(row.contrast,*a.split('::'))]; z=indexed.loc[(row.contrast,*b.split('::'))]
        if abs(row.delta_auroc-(x.auroc-z.auroc))>1e-14 or abs(row.delta_auprc_ap-(x.auprc_ap-z.auprc_ap))>1e-14:
            raise ValueError('Paired point estimate mismatch')
    for row in metrics.itertuples():
        d=labeled[(labeled.panel==row.panel)&(labeled.classifier==row.classifier)]
        if row.contrast=='CRC_vs_study_controls': d=d[d['Study condition'].isin(['CRC','Control'])]
        y=d['Study condition'].eq('CRC').astype(int)
        if abs(roc_auc_score(y,d.probability_crc)-row.auroc)>1e-14: raise ValueError('AUROC mismatch')
        if abs(average_precision_score(y,d.probability_crc)-row.auprc_ap)>1e-14: raise ValueError('AP mismatch')
    tr=pd.read_csv(INP/'training_manifest.tsv',sep='\t')
    if set(tr.sample_id)&set(em['Sample Source ID']): raise ValueError('Overlap')
    totalbytes=sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file())
    if totalbytes>250*1024**2: raise ValueError('Run exceeds resource budget')
    rec={'verified_utc':stamp(),'status':'PASS','replayed_models':len(bundle['models']),
         'max_probability_replay_difference':maxdiff,'recomputed_metric_rows':len(metrics),
         'input_and_model_hashes_unchanged':True,'train_external_exact_id_overlap':0,
         'run_bytes':totalbytes,'author_independent_rerun':'not yet confirmed'}
    save_json(ROOT/'00_receipts/verification.json',rec); print(json.dumps(rec,indent=2),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('command',choices=['audit','fit','predict','evaluate','verify'])
    args=ap.parse_args(); globals()[args.command]()

