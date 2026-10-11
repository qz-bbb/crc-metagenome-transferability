"""Descriptive MP4 explanation only; never fits or optimizes external models."""
import argparse, hashlib, json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from ablation_core import eligibility, abundance_summary
KEYS={'model_lock','training_manifest','inventory','extracted','replication_root','output'}
CLASSIC=['Fusobacterium_nucleatum','Parvimonas_micra','Gemella_morbillorum','Dialister_pneumosintes',
         'Peptostreptococcus_stomatis','Solobacterium_moorei','Bacteroides_fragilis']

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def save(path,value):
    Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')

def table(path,frame):
    frame.to_csv(path,sep='\t',index=False,float_format='%.17g')

def require_equal(observed,expected,message):
    if observed!=expected:raise ValueError(message)

def validate_output(output,inputs):
    out=Path(output).resolve()
    for source in inputs:
        source=Path(source).resolve()
        if out==source or out in source.parents or source in out.parents:
            raise ValueError('Output/input overlap is not permitted')
    if out.exists():raise ValueError('Output exists; preserve earlier results')
    return out

def align_profile(a,ids,features):
    ids=list(ids);features=list(features)
    if a.index.duplicated().any() or a.columns.duplicated().any() or len(set(ids))!=len(ids) or len(set(features))!=len(features):raise ValueError('Duplicate profile/requested axis')
    if set(ids)-set(a.columns):raise ValueError('Missing sample IDs')
    a=a.loc[a.index.str.rsplit('|',n=1).str[-1].str.startswith('s__'),ids].T
    if not np.isfinite(a.to_numpy()).all():raise ValueError('Abundances must be finite')
    if (a.to_numpy()<0).any():raise ValueError('Abundances must be non-negative')
    if not a.sum(axis=1).between(99.9,100.1).all():raise ValueError('Expected species-level percent totals near 100')
    present=[f in a.columns for f in features]
    # Identical source/MP4 pipeline: the previous lock explicitly permitted
    # absent profile rows to denote source non-detection, never MP3 crosswalking.
    return a.reindex(columns=features,fill_value=0.).to_numpy(),present

def raw_profile(extracted,stem,ids,features):
    if Path(stem).name!=stem:raise ValueError('Invalid source stem')
    p=extracted/'metaphlan4_profiles'/f'{stem}.tsv'
    with p.open(encoding='utf-8') as f:header=f.readline().rstrip('\r\n').split('\t')[1:]
    if len(set(header))!=len(header):raise ValueError('Duplicate profile sample header')
    raw,present=align_profile(pd.read_csv(p,sep='\t',index_col=0),ids,features)
    return raw,present,p

def describe(raw,y,features,coef,pseudo):
    raw=np.asarray(raw,float);y=np.asarray(y);coef=np.asarray(coef,float)
    if raw.ndim!=2 or raw.shape!=(len(y),len(features)) or coef.shape!=(len(features),):raise ValueError('Input shape mismatch')
    if set(y.tolist())!={0,1}:raise ValueError('Both classes 0 and 1 required')
    if not np.isfinite(pseudo) or pseudo<=0:raise ValueError('Positive finite pseudocount required')
    if not np.isfinite(raw).all() or (raw<0).any() or not np.isfinite(coef).all():raise ValueError('Finite non-negative abundance and finite reference required')
    a=abundance_summary(raw,y,features);z=np.log(raw+pseudo);z-=z.mean(axis=1,keepdims=True)
    a['mean_clr_difference']=z[y==1].mean(axis=0)-z[y==0].mean(axis=0)
    a['training_reference_sign']=np.sign(coef)
    a['externally_detected']=(raw>0).any(axis=0)
    a['raw_concordant']=(np.sign(a.raw_median_difference)==np.sign(coef))&a.raw_median_difference.ne(0)
    a['clr_concordant']=(np.sign(a.mean_clr_difference)==np.sign(coef))&(a.mean_clr_difference.abs()>1e-12)&a.externally_detected
    a['raw_CLR_direction_disagreement']=(np.sign(a.raw_median_difference)!=np.sign(a.mean_clr_difference))
    return a

def main(config_path):
    config_path=Path(config_path).resolve();cfg=json.loads(config_path.read_text(encoding='utf-8'))
    if set(cfg)!=KEYS:raise ValueError(f'Config keys must be exactly {sorted(KEYS)}')
    cfg={k:(config_path.parent/Path(v)).resolve() for k,v in cfg.items()}
    dest=validate_output(cfg['output'],[v for k,v in cfg.items() if k!='output']+[Path(__file__).parent])
    LOCK=cfg['model_lock'];EXTRACT=cfg['extracted']
    start=time.time();state=json.loads((LOCK/'preprocessing.json').read_text());features=state['features'];pseudo=state['pseudocount']
    e=pd.read_csv(LOCK/'feature_evidence.tsv',sep='\t').set_index('taxon').loc[features].reset_index()
    flags=eligibility(e);flags['species']=flags.taxon.str.rsplit('|s__',n=1).str[-1]
    panels=pd.read_csv(LOCK/'panels.tsv',sep='\t');training=pd.read_csv(cfg['training_manifest'],sep='\t')
    inventory=pd.read_csv(cfg['inventory'],sep='\t')
    if training.duplicated(['source_stem','sample_id']).any():raise ValueError('Duplicate training sample')
    if not training.study_condition.isin(['CRC','control']).all():raise ValueError('Unrecognized training label (historical training uses lowercase control)')
    if inventory.cohort.duplicated().any():raise ValueError('Duplicate external cohort')
    if set(training.source_stem)&set(inventory.source_stem):raise ValueError('Training/external source overlap')
    if not panels.taxon.isin(features).all():raise ValueError('Panel feature outside locked universe')
    inputs=[config_path,Path(__file__),Path(__file__).with_name('ablation_core.py'),LOCK/'preprocessing.json',LOCK/'feature_evidence.tsv',LOCK/'panels.tsv',cfg['training_manifest'],cfg['inventory']]
    for stem in set(training.source_stem)|set(inventory.source_stem):
        inputs.extend([EXTRACT/'metaphlan4_profiles'/f'{stem}.tsv',EXTRACT/'metadata'/f'{stem}.tsv'])
    inputs += [cfg['replication_root']/str(c)/'species_replication_all800.tsv' for c in inventory.cohort]
    hashes={str(p):sha(p) for p in inputs}
    dest.mkdir(parents=True)
    save(dest/'start_receipt.json',{'created_utc':pd.Timestamp.now(tz='UTC').isoformat(),'prior_external_outcomes_known':True,
       'inputs':hashes,'status':'DESCRIPTIVE_POST_OUTCOME_NOT_NEW_VALIDATION'})
    for panel,g in panels.groupby('panel'):flags[panel+'_selected']=flags.taxon.isin(g.taxon)
    studyrows=[];rawpieces=[];ypieces=[]
    for stem,g in training.groupby('source_stem',sort=True):
        raw,present,p=raw_profile(EXTRACT,stem,g.sample_id,features);y=g.study_condition.eq('CRC').astype(int).to_numpy()
        a=describe(raw,y,features,e.coef.to_numpy(),pseudo);a['cohort']=stem;a['branch']='training';a['explicit_profile_row']=present
        studyrows.append(a);rawpieces.append(raw);ypieces.append(y)
    trainingstats=pd.concat(studyrows,ignore_index=True)
    agreement=trainingstats.groupby('taxon',sort=False).agg(eligible_studies=('raw_concordant','size'),raw_agreement=('raw_concordant','mean'),raw_zero_studies=('zero_raw_median_difference','sum'),both_medians_zero_studies=('both_medians_zero','sum'),clr_agreement=('clr_concordant','mean')).reset_index()
    flags=flags.merge(agreement,on='taxon',validate='one_to_one')
    np.testing.assert_allclose(flags.raw_agreement,flags.direction_agreement,atol=1e-12,rtol=0)
    pooled=abundance_summary(np.concatenate(rawpieces),np.concatenate(ypieces),features)
    flags=flags.merge(pooled,on='taxon',validate='one_to_one')
    table(dest/'training_taxon_exclusion_and_sparsity.tsv',flags)
    table(dest/'training_per_study_sparsity.tsv',trainingstats)
    extrows=[];panelrows=[]
    for inv in inventory.itertuples():
        meta=pd.read_csv(EXTRACT/'metadata'/f'{inv.source_stem}.tsv',sep='\t');meta=meta[meta['Study condition'].isin(['CRC','Control'])].sort_values('Sample Source ID')
        raw,present,p=raw_profile(EXTRACT,inv.source_stem,meta['Sample Source ID'],features);y=meta['Study condition'].eq('CRC').astype(int).to_numpy()
        a=describe(raw,y,features,e.coef.to_numpy(),pseudo);a['cohort']=inv.cohort;a['branch']='external';a['role']=inv.role;a['explicit_profile_row']=present
        old=pd.read_csv(cfg['replication_root']/inv.cohort/'species_replication_all800.tsv',sep='\t').set_index('taxon').loc[features]
        np.testing.assert_allclose(a.mean_clr_difference,old.external_mean_clr_difference,rtol=1e-10,atol=1e-10)
        np.testing.assert_allclose(a.raw_median_difference,old.external_median_relative_difference,rtol=1e-10,atol=1e-10)
        require_equal(a.clr_concordant.to_list(),old.concordant_clr.to_list(),'External CLR concordance mismatch')
        a['existing_external_q_full800']=old.external_q_full_800.to_numpy()
        a=a.merge(flags[['taxon','species','all_failed_rules','first_failed_waterfall_rule','q','direction_agreement','i2','re_q','inner_sign_agreement','inner_q_fraction']],on='taxon',validate='one_to_one')
        extrows.append(a)
        for panel,g in panels.groupby('panel',sort=False):
            s=a[a.taxon.isin(g.taxon)]
            panelrows.append({'cohort':inv.cohort,'role':inv.role,'panel':panel,'n_taxa':len(s),
              'raw_zero_differences':int(s.zero_raw_median_difference.sum()),'both_medians_zero':int(s.both_medians_zero.sum()),
              'raw_concordant':int(s.raw_concordant.sum()),'clr_concordant':int(s.clr_concordant.sum()),
              'median_detection_crc':float(s.detection_fraction_crc.median()),'median_detection_control':float(s.detection_fraction_control.median()),
              'zero_raw_but_clr_concordant':int((s.zero_raw_median_difference&s.clr_concordant).sum()),
              'zero_raw_but_clr_concordant_q10':int((s.zero_raw_median_difference&s.clr_concordant&s.existing_external_q_full800.lt(.1)).sum())})
    ext=pd.concat(extrows,ignore_index=True);table(dest/'external_taxon_sparsity.tsv',ext);table(dest/'panel_sparsity_summary.tsv',pd.DataFrame(panelrows))
    table(dest/'classic_crc_training.tsv',flags[flags.species.isin(CLASSIC)])
    table(dest/'classic_crc_external.tsv',ext[ext.species.isin(CLASSIC)])
    table(dest/'classic_crc_availability.tsv',pd.DataFrame([{'requested_species':s,'exact_locked_feature_present':bool(flags.species.eq(s).any()),'matching_policy':'exact source species only; no SGB synonym assumed'} for s in CLASSIC]))
    table(dest/'global_training_waterfall.tsv',pd.DataFrame([{'stage':k,'n_eligible':int(v)} for k,v in [('all_filtered',len(flags)),('A',flags.A.sum()),('A_D',flags.AD.sum()),('A_D_H',(flags.A&flags.D&flags.H).sum()),('A_D_H_R',flags.ICP.sum())]]))
    for p,h in json.loads((dest/'start_receipt.json').read_text())['inputs'].items():
        require_equal(sha(p),h,'Source changed during generation: '+p)
    save(dest/'receipt.json',{'status':'PASS','prior_external_outcomes_known':True,'n_training':len(training),'training_studies':len(studyrows),
       'n_external_taxon_rows':len(ext),'external_cohorts':len(extrows),'features':len(features),'old_raw_CLR_results_replayed':True,
       'raw_training_agreement_replayed':True,'no_fitting_or_new_tests':True,'elapsed_seconds':time.time()-start,
       'files':{p.name:sha(p) for p in dest.glob('*.tsv')}})
    print('MP4 sparsity explanatory tables PASS',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,required=True)
    main(parser.parse_args().config)
