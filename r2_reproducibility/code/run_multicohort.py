"""Reuse a frozen model; audit/predict/evaluate/replicate are distinct stages.
No fitting or panel-selection code is called by this expansion.
"""
from pathlib import Path
import argparse, hashlib, json, sys, tarfile, time, re
import numpy as np
import pandas as pd
import joblib
from scipy.stats import spearmanr
from multicohort_core import check_ids, direction_fraction, bootstrap_effects, bootstrap_directions, welch_effects, ZERO_TOL

ROOT = Path('UNCONFIGURED_OUTPUT')
OLD = Path('UNCONFIGURED_INITIAL_BRANCH')
sys.path.insert(0,str(OLD/'02_analysis'))
import run_external as old
from external_core import transform_external, bootstrap_metrics, attach_labels
INP=ROOT/'01_inputs'; OUT=ROOT/'02_analysis/results'; REC=ROOT/'00_receipts'
COHORTS={
 'LiuNN_2022':('Public_study__LiuNN_2022',{'CRC':78,'Control':85},'primary'),
 'YangY_2021':('Public_study__YangY_2021',{'CRC':100,'Control':100},'primary'),
 'IIGM_TU':('This_study_cohort6__IIGM_TU',{'CRC':18,'Control':39},'primary'),
 'YangJ_2020':('Public_study__YangJ_2020',{'CRC':98,'Control':95},'sensitivity_household_controls'),
 'NHSII':('This_study_cohort5__NSHII',{'CRC':14,'Control':448,'Adenoma':435},'sensitivity_history_female_small_crc'),
 'IIGM_CZ':(old.EXT,{'CRC':59,'Control':46,'Adenoma':19},'previously_evaluated_posthoc_replication')}
SEED=20261011
def table(p,d): old.table(p,d)
def save(p,d): old.save_json(p,d)
def sha(p): return old.sha(p)

def read_meta(stem):
    return pd.read_csv(old.EXTRACTED/'metadata'/f'{stem}.tsv',sep='\t',keep_default_na=False)

def audit():
    old.check_lock(); old.verify_sources(); INP.mkdir(parents=True,exist_ok=True)
    if (REC/'expansion_lock.json').exists(): raise ValueError('Expansion already locked')
    originals=pd.read_csv(old.BASEMETA,sep='\t',keep_default_na=False)
    bundle=joblib.load(old.LOCK/'models.joblib'); state=bundle['state']
    training=pd.read_csv(old.INP/'training_manifest.tsv',sep='\t',keep_default_na=False)
    input_paths=[REC/'PROTOCOL.md',Path(__file__),Path(__file__).with_name('multicohort_core.py'),
                 Path(__file__).with_name('test_multicohort.py'),old.LOCK/'lock.json']
    for arc,folder in [('piccinno_metadata_full.tar.gz','metadata'),('metaphlan4_profiles_full.tar.gz','metaphlan4_profiles')]:
        with tarfile.open(old.SOURCE/arc,'r:gz') as tar:
            members={Path(t.name).name:t for t in tar if t.isfile()}
            for stem,_,_ in COHORTS.values():
                p=old.EXTRACTED/folder/f'{stem}.tsv'
                if tar.extractfile(members[p.name]).read()!=p.read_bytes(): raise ValueError('Source archive mismatch')
                input_paths.append(p)
    training_ids=set(originals.sample_id.astype(str))|set(originals.sample_uid.astype(str))
    fingerprints={}
    for stem,rows in training.groupby('source_stem',sort=True):
        raw=old.species_profile(stem,rows.sample_id).reindex(columns=state['features'],fill_value=0.)
        for sid,row in raw.iterrows(): fingerprints[hashlib.sha256(np.asarray(row,dtype='<f8').tobytes()).hexdigest()]=f'training::{sid}'
    seen=set(); summaries=[]; coverage_rows=[]; sample_rows=[]
    for cohort,(stem,counts,role) in COHORTS.items():
        m=read_meta(stem); raw=old.species_profile(stem)
        check_ids(m,raw.index,training_ids)
        if m['Study condition'].value_counts().to_dict()!=counts: raise ValueError(f'Unexpected counts: {cohort}')
        if set(m['Sample Source ID'])&seen: raise ValueError('Cross-external ID duplicate')
        seen.update(m['Sample Source ID'])
        aligned=raw.reindex(columns=state['features'],fill_value=0.)
        for sid,row in aligned.iterrows():
            fp=hashlib.sha256(np.asarray(row,dtype='<f8').tobytes()).hexdigest()
            if fp in fingerprints: raise ValueError(f'Exact profile duplicate {cohort}/{sid}: {fingerprints[fp]}')
            fingerprints[fp]=f'{cohort}::{sid}'
        _,coverage=transform_external(raw,state,same_pipeline=True)
        save(INP/f'{cohort}_coverage.json',coverage)
        for panel,features in bundle['panels'].items():
            coverage_rows.append({'cohort':cohort,'panel':panel,'n_features':len(features),
              'explicit_profile_rows':sum(t in raw.columns for t in features),
              'unreported_taxa':' ; '.join(t for t in features if t not in raw.columns)})
        summaries.append({'cohort':cohort,'source_stem':stem,'role':role,'crc':counts['CRC'],'control':counts['Control'],
          'adenoma_excluded':counts.get('Adenoma',0),'total_profile_n':len(m),'species_rows':raw.shape[1],
          'explicit_800_rows':coverage['n_exact_rows'],'id_overlap_original_1395':0,'identical_800_vector_overlap':0,
          'age_available':int(pd.to_numeric(m['Age'],errors='coerce').notna().sum()),
          'sex_available':int(m.Sex.isin(['Female','Male']).sum()),
          'identity_limit':'public identifiers/profile fingerprints only; accession linkage requires a separate source audit'})
        for _,row in m.iterrows(): sample_rows.append({'cohort':cohort,**row.to_dict()})
    table(INP/'cohort_inventory.tsv',pd.DataFrame(summaries))
    table(INP/'panel_coverage.tsv',pd.DataFrame(coverage_rows))
    table(INP/'external_sample_manifest.tsv',pd.DataFrame(sample_rows))
    input_paths += [INP/'cohort_inventory.tsv',INP/'external_sample_manifest.tsv',INP/'panel_coverage.tsv']
    lock={'created_utc':old.stamp(),'prior_model_lock_sha256':sha(old.LOCK/'lock.json'),
          'previously_observed_classification_cohort':'IIGM_CZ',
          'cohorts':list(COHORTS),'retraining':False,'bootstrap_replicates':2000,'seed':SEED,
          'hashes':{str(p):sha(p) for p in input_paths}}
    save(REC/'expansion_lock.json',lock)
    print(pd.DataFrame(summaries).to_string(index=False),flush=True)

def check():
    old.check_lock(); old.verify_sources()
    lock=json.loads((REC/'expansion_lock.json').read_text(encoding='utf-8'))
    for path,h in lock['hashes'].items():
        if sha(path)!=h: raise ValueError(f'Frozen expansion changed: {path}')
    return lock

def predict(cohort):
    check(); stem,_,_=COHORTS[cohort]
    bundle=joblib.load(old.LOCK/'models.joblib'); raw=old.species_profile(stem)
    x,_=transform_external(raw,bundle['state'],same_pipeline=True); rows=[]
    for key,entry in bundle['models'].items():
        panel,kind=key.split('::'); probs=entry['pipeline'].predict_proba(x[:,entry['indices']])[:,1]
        for sid,p in zip(raw.index,probs): rows.append({'sample_id':sid,'panel':panel,'classifier':kind,'probability_crc':p})
    dest=OUT/cohort; table(dest/'predictions_blind.tsv',pd.DataFrame(rows))
    save(dest/'prediction_receipt.json',{'created_utc':old.stamp(),'expansion_lock_sha256':sha(REC/'expansion_lock.json'),
      'predictions_sha256':sha(dest/'predictions_blind.tsv'),'n_samples':len(raw),'models':len(bundle['models']),
      'metadata_read_by_prediction_stage':False,'models_refitted':False})
    print(f'{cohort}: {len(raw)} samples, {len(bundle["models"])} unchanged models',flush=True)

def evaluate(cohort):
    check(); start=time.time(); stem,counts,role=COHORTS[cohort]; dest=OUT/cohort
    receipt=json.loads((dest/'prediction_receipt.json').read_text())
    if receipt['expansion_lock_sha256']!=sha(REC/'expansion_lock.json') or receipt['predictions_sha256']!=sha(dest/'predictions_blind.tsv'):
        raise ValueError('Prediction receipt mismatch')
    meta=read_meta(stem); bundle=joblib.load(old.LOCK/'models.joblib')
    pred=attach_labels(pd.read_csv(dest/'predictions_blind.tsv',sep='\t'),meta,counts,set(bundle['models']))
    table(dest/'predictions_labeled.tsv',pred)
    subset=pred[pred['Study condition'].isin(['CRC','Control'])].copy()
    subset['key']=subset.panel+'::'+subset.classifier
    wide=subset.pivot(index='sample_id',columns='key',values='probability_crc').sort_index()
    y=meta.set_index('Sample Source ID').loc[wide.index,'Study condition'].eq('CRC').astype(int).to_numpy()
    if wide.isna().any().any(): raise ValueError('Incomplete prediction grid')
    boot=bootstrap_metrics(y,{k:wide[k].to_numpy() for k in wide},2000,SEED+list(COHORTS).index(cohort))
    rows=[]; paired=[]
    for key,point in boot['point'].items():
        ci=np.quantile(boot['bootstrap'][key],[.025,.975],axis=0); panel,kind=key.split('::')
        rows.append({'cohort':cohort,'role':role,'panel':panel,'classifier':kind,'n':len(y),'crc':int(y.sum()),
          'control':int((1-y).sum()),'auroc':point[0],'auroc_lower':ci[0,0],'auroc_upper':ci[1,0],
          'ap':point[1],'ap_lower':ci[0,1],'ap_upper':ci[1,1],'ap_baseline':float(y.mean())})
    for kind in ['EN','RF']:
        for a,b in [('ICP','ARP_nICP'),('DCP_nICP','ARP_nICP'),('DCP29','ARP29')]:
            ka,kb=f'{a}::{kind}',f'{b}::{kind}'
            delta=np.asarray(boot['point'][ka])-np.asarray(boot['point'][kb])
            ci=np.quantile(boot['bootstrap'][ka]-boot['bootstrap'][kb],[.025,.975],axis=0)
            paired.append({'cohort':cohort,'classifier':kind,'comparison':f'{a} minus {b}','delta_auroc':delta[0],
              'delta_auroc_lower':ci[0,0],'delta_auroc_upper':ci[1,0],'delta_ap':delta[1],'delta_ap_lower':ci[0,1],'delta_ap_upper':ci[1,1]})
    table(dest/'classification_metrics.tsv',pd.DataFrame(rows)); table(dest/'classification_paired.tsv',pd.DataFrame(paired))
    save(dest/'classification_receipt.json',{'completed_utc':old.stamp(),'seconds':time.time()-start,'n_boot':2000,
      'hashes':{p.name:sha(p) for p in [dest/'classification_metrics.tsv',dest/'classification_paired.tsv',dest/'predictions_labeled.tsv']}})
    print(pd.DataFrame(rows)[['cohort','panel','classifier','auroc','ap']].to_string(index=False),flush=True)

def replicate(cohort):
    check(); stem,_,role=COHORTS[cohort]; dest=OUT/cohort
    meta=read_meta(stem); meta=meta[meta['Study condition'].isin(['CRC','Control'])].sort_values('Sample Source ID')
    raw=old.species_profile(stem).loc[meta['Sample Source ID']]
    bundle=joblib.load(old.LOCK/'models.joblib'); features=bundle['state']['features']
    x,_=transform_external(raw,bundle['state'],same_pipeline=True)
    evidence=pd.read_csv(old.LOCK/'feature_evidence.tsv',sep='\t').set_index('taxon').loc[features]
    reference=evidence.coef.to_numpy(); y=meta['Study condition'].eq('CRC').astype(int).to_numpy()
    effect,p,q=welch_effects(x,y)
    relative=raw.reindex(columns=features,fill_value=0.).to_numpy()
    raw_effect=np.median(relative[y==1],axis=0)-np.median(relative[y==0],axis=0)
    coordinate_concordant=(np.sign(effect)==np.sign(reference))&(np.abs(effect)>ZERO_TOL)&(np.abs(reference)>ZERO_TOL)
    detected=(relative>0).any(axis=0)
    concordant=coordinate_concordant & detected
    detail=pd.DataFrame({'cohort':cohort,'taxon':features,'training_coef':reference,'training_q':evidence.q.to_numpy(),
      'external_mean_clr_difference':effect,'external_welch_p':p,'external_q_full_800':q,'concordant_clr':concordant,
      'coordinate_concordant_clr':coordinate_concordant,'externally_detected':detected,
      'external_median_relative_difference':raw_effect,'explicit_profile_row':[t in raw.columns for t in features]})
    table(dest/'species_replication_all800.tsv',detail)
    union=sorted(set().union(*map(set,bundle['panels'].values())))
    union_indices=[features.index(t) for t in union]; ref=reference[union_indices]
    boot_agree=bootstrap_directions(x[:,union_indices],y,relative[:,union_indices],ref,2000,SEED+100+list(COHORTS).index(cohort))
    rows=[]; fractions={}
    for panel,taxa in bundle['panels'].items():
        ix=[features.index(t) for t in taxa]; uix=[union.index(t) for t in taxa]
        fractions[panel]=boot_agree[:,uix].mean(axis=1); ci=np.quantile(fractions[panel],[.025,.975])
        rows.append({'cohort':cohort,'role':role,'panel':panel,'n_features':len(ix),
          'concordant_n':int(concordant[ix].sum()),'direction_fraction':float(concordant[ix].mean()),
          'detected_n':int(detected[ix].sum()),'coordinate_direction_fraction':direction_fraction(effect[ix],reference[ix]),
          'direction_lower':ci[0],'direction_upper':ci[1],
          'concordant_q10_n':int((concordant[ix]&(q[ix]<.1)).sum()),
          'q10_fraction':float((concordant[ix]&(q[ix]<.1)).mean()),
          'spearman_effect_r':float(spearmanr(reference[ix],effect[ix]).statistic),
          'raw_median_direction_fraction':direction_fraction(raw_effect[ix],reference[ix]),
          'raw_zero_difference_n':int((raw_effect[ix]==0).sum())})
    table(dest/'panel_replication.tsv',pd.DataFrame(rows))
    bootstrap_frame=pd.DataFrame(fractions); bootstrap_frame.insert(0,'replicate',range(2000))
    table(dest/'direction_bootstrap.tsv',bootstrap_frame)
    indexed=pd.DataFrame(rows).set_index('panel'); pairs=[]
    for a,b in [('ICP','ARP_nICP'),('DCP_nICP','ARP_nICP'),('DCP29','ARP29'),('ICP','random_nICP')]:
        ci=np.quantile(fractions[a]-fractions[b],[.025,.975])
        pairs.append({'cohort':cohort,'comparison':f'{a} minus {b}',
          'delta_direction_fraction':indexed.loc[a,'direction_fraction']-indexed.loc[b,'direction_fraction'],
          'lower':ci[0],'upper':ci[1]})
    table(dest/'replication_paired.tsv',pd.DataFrame(pairs))
    save(dest/'replication_receipt.json',{'completed_utc':old.stamp(),'n_boot':2000,'label_contrast':'CRC vs source Control',
      'expansion_lock_sha256':sha(REC/'expansion_lock.json'),
      'hashes':{n:sha(dest/n) for n in ['species_replication_all800.tsv','panel_replication.tsv','direction_bootstrap.tsv','replication_paired.tsv']}})
    print(pd.DataFrame(rows)[['cohort','panel','concordant_n','n_features','direction_fraction','q10_fraction']].to_string(index=False),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('command',choices=['audit','predict','evaluate','replicate'])
    ap.add_argument('--cohort',choices=list(COHORTS)); args=ap.parse_args()
    if args.command=='audit': audit()
    elif args.cohort is None: ap.error('--cohort required')
    else: globals()[args.command](args.cohort)

