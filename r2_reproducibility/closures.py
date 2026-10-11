"""Explicitly configured closure replay, one checkpoint per invocation."""
import argparse
import csv
import importlib
import json
from pathlib import Path
import sys

PACKAGE = Path(__file__).resolve().parent
sys.path.insert(0, str(PACKAGE))
import portable
sys.path.insert(0, str(PACKAGE / 'code'))
KEYS = portable.PATH_KEYS | {'nested_reference_root', 'identity_baseline_root',
                            'identity_multicohort_root', 'frozen_rule_root', 'frozen_identity_root',
                            'identity_enriched_manifest'}
MARKER = '.closure-replay.json'


def load_config(path):
    path = Path(path).resolve(); value = json.loads(path.read_text(encoding='utf-8-sig'))
    if set(value) - KEYS: raise ValueError('Unknown configuration keys: '+str(sorted(set(value)-KEYS)))
    return {key: (path.parent / item).resolve() for key, item in value.items() if item}


def check_output(config):
    output = config['output_root'].resolve()
    sources = [PACKAGE]
    for key, path in config.items():
        if key != 'output_root': sources.append(path if key.endswith('_root') else path.parent)
    for source in sources:
        if output == source or source in output.parents or output in source.parents:
            raise ValueError('Output overlaps package or configured input tree')
    if output.exists() and not (output / MARKER).is_file(): raise ValueError('Existing unmarked output refused')
    return output


def begin(config, required):
    missing = portable.missing_inputs(config, required)
    if missing: raise FileNotFoundError('Missing required inputs: '+str(missing))
    output = check_output(config)
    portable.verify_contract(config, required)
    overrides = {}
    if 'identity_enriched_manifest' in config:
        overrides['identity_multicohort_root/05_review/original_enriched_identity_manifest.tsv'] = config['identity_enriched_manifest']
    portable.verify_contract(config, required, PACKAGE / 'closure_input_contract.tsv', overrides=overrides)
    bound = {k: str(v) for k,v in config.items()}
    marker = output / MARKER
    if marker.exists() and json.loads(marker.read_text())['configuration'] != bound:
        raise ValueError('Replay configuration changed; choose a new output')
    if not marker.exists(): portable.write_json(marker, {'configuration':bound, 'post_outcome_replay':True})
    return output


def configure_rule(config, output):
    run = importlib.import_module('run_ablation')
    run.HERE = output / 'rule_ablation'
    run.NESTED = config['nested_reference_root']
    run.MATRIX = config['baseline_matrix']; run.META = config['baseline_metadata']
    return run


def configure_identity(config, output):
    old = portable.configure_external(config, output)
    old.ROOT = config['identity_baseline_root']; old.INP = old.ROOT / '01_inputs'
    old.OUT = old.ROOT / '02_analysis/results'; old.LOCK = old.ROOT / '02_analysis/model_lock'
    run = importlib.import_module('run_identity')
    run.old = old; run.OLD = old.ROOT; run.MULTI = config['identity_multicohort_root']
    run.ENRICHED_MANIFEST = config.get('identity_enriched_manifest')
    run.HERE = output / 'identity_sensitivity'; run.OUT = run.HERE / 'results'; run.LOCK = run.HERE / 'model_lock'
    return run


def replay_frozen(config, output):
    """Pure panel reconstruction and point metrics; no model fitting."""
    import numpy as np
    import pandas as pd
    from sklearn.metrics import roc_auc_score, average_precision_score
    from ablation_core import panels, eligibility, bootstrap_metrics
    from identity_core import resolve_training, assert_disjoint_ids
    from exact_tsv import read_exact
    rule = config['frozen_rule_root']; nested = config['nested_reference_root']
    checks = []
    for index in range(11):
        stats = read_exact(nested / f'fold_{index:02d}_association.tsv')
        selected, size = panels(stats)
        saved = read_exact(rule / f'folds/{index:02d}/eligibility.tsv')
        if stats.taxon.tolist() != saved.taxon.tolist(): raise ValueError('Frozen taxon order mismatch')
        flags = eligibility(stats)
        for key in ['A','D','H','R','AD','AH','AR','ICP']:
            if not np.array_equal(flags[key], saved[key]): raise ValueError('Eligibility mismatch '+key)
        for key, members in selected.items():
            if set(members) != set(saved.loc[saved[key+'_selected'], 'taxon']): raise ValueError('Selected membership mismatch')
        if size != int(saved.target_panel_size.iloc[0]): raise ValueError('Target size changed')
        checks.append({'kind':'rule_selection','fold':index,'size':size,'status':'PASS'})
    # Recompute all rule point metrics from frozen prediction probabilities.
    metric_count = 0
    for index in range(11):
        pred = read_exact(rule / f'folds/{index:02d}/predictions.tsv.gz')
        metric = read_exact(rule / f'folds/{index:02d}/metrics.tsv')
        for (model,panel), group in pred.groupby(['model','panel'],sort=False):
            expected = metric[metric.model.eq(model)&metric.panel.eq(panel)].iloc[0]
            observed = [roc_auc_score(group.y,group.pred),average_precision_score(group.y,group.pred)]
            np.testing.assert_allclose(observed,[expected.auroc,expected.ap],rtol=0,atol=1e-12)
            metric_count += 1
    # Bounded bootstrap behavior check; historical 2000-replicate intervals are not replayed.
    group = pred[(pred.model=='EN')&(pred.panel=='A')]
    first = bootstrap_metrics(group.y, {'A':group.pred.to_numpy()}, 20, 20261010)
    second = bootstrap_metrics(group.y, {'A':group.pred.to_numpy()}, 20, 20261010)
    np.testing.assert_array_equal(first[1]['A'],second[1]['A'])
    training = pd.read_csv(config['identity_baseline_root']/'01_inputs/training_manifest.tsv',sep='\t',keep_default_na=False)
    enriched_path = config.get('identity_enriched_manifest', config['identity_multicohort_root']/'05_review/original_enriched_identity_manifest.tsv')
    enriched = pd.read_csv(enriched_path,sep='\t',keep_default_na=False)
    audit = resolve_training(training,enriched)
    frozen = pd.read_csv(config['frozen_identity_root']/'results/training_identity_audit.tsv',sep='\t',keep_default_na=False)
    pd.testing.assert_frame_equal(audit.reset_index(drop=True),frozen.reset_index(drop=True),check_dtype=False)
    identity_count = 0
    ipred = read_exact(config['frozen_identity_root']/'results/predictions.tsv')
    for cohort, prediction in ipred.groupby('cohort',sort=False):
        from run_multicohort import COHORTS
        stem,_,_ = COHORTS[cohort]
        meta = pd.read_csv(config['source_root']/'extracted/metadata'/f'{stem}.tsv',sep='\t',keep_default_na=False)
        assert_disjoint_ids(audit.loc[audit.accession_resolved,'sample_id'],meta['Sample Source ID'])
        meta = meta[meta['Study condition'].isin(['CRC','Control'])].set_index('Sample Source ID')
        metrics = read_exact(config['frozen_identity_root']/'results'/f'{cohort}_metrics.tsv')
        for (panel,kind), group in prediction.groupby(['panel','classifier'],sort=False):
            probabilities=group.set_index('sample_id').probability_crc.loc[meta.index]
            y=meta['Study condition'].eq('CRC').astype(int)
            expected=metrics[metrics.panel.eq(panel)&metrics.classifier.eq(kind)].iloc[0]
            np.testing.assert_allclose([roc_auc_score(y,probabilities),average_precision_score(y,probabilities)],
                                       [expected.auroc,expected.ap],rtol=0,atol=1e-12)
            identity_count+=1
    portable.write_json(output/'frozen_replay.json', {'status':'PASS','rule_selection_checks':checks,
        'rule_point_metrics':metric_count,'identity_point_metrics':identity_count,
        'identity_n':len(audit),'identity_selected_n':int(audit.accession_resolved.sum()),
        'smoke_bootstrap':20,'historical_intervals_replayed':False,'model_fitting_repeated':False,
        'scope':'frozen source/result verification and pure recomputation; not new biological validation'})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('stage',choices=['rule-init','rule-fold','rule-finalize','identity-prepare','identity-fit','identity-predict','identity-evaluate','frozen-replay'])
    parser.add_argument('--fold',type=int); parser.add_argument('--cohort')
    args=parser.parse_args(); config=load_config(args.config)
    if args.stage=='rule-fold' and (args.fold is None or not 0<=args.fold<11): parser.error('--fold 0..10 required')
    if args.stage=='identity-evaluate' and not args.cohort: parser.error('--cohort required')
    required=['baseline_metadata','baseline_matrix','nested_reference_root'] if args.stage.startswith('rule') else ['baseline_metadata','source_root','identity_baseline_root','identity_multicohort_root']
    if args.stage=='frozen-replay': required=list(KEYS-{'output_root','legacy_clr','legacy_abundance','legacy_interpretation','legacy_membership','old_clr_sets','identity_enriched_manifest'})
    output=begin(config,required)
    if args.stage.startswith('rule-'):
        run=configure_rule(config,output)
        if args.stage=='rule-fold': run.fold(args.fold)
        else: getattr(run,{'rule-init':'init','rule-finalize':'finalize'}[args.stage])()
    elif args.stage.startswith('identity-'):
        run=configure_identity(config,output); action=args.stage.split('-')[1]
        if action=='prepare':
            if (run.HERE/'input_manifest.json').exists(): raise ValueError('Prepared identity destination exists')
            run.prepare()
        elif action=='evaluate': run.evaluate(args.cohort)
        else: getattr(run,action)()
    else: replay_frozen(config,output)
    portable.write_json(output/'stage_receipts'/(args.stage+'.json'),{'status':'PASS','stage':args.stage,
        'runner_sha256':portable.sha(__file__),'post_outcome_replay':True,'author_review_supplied':False})


if __name__=='__main__': main()
