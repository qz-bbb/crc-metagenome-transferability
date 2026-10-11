"""Recover historical cMD intermediates from independently sourced inputs.

Stages are explicit: matrices; panels (four historical dependencies and an
honestly labeled two-column RF input); full (the complete interpretation).
Source main() routines are never run. Comparison targets are read after output
generation only, and are never made available to the computation context.
"""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
import warnings

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '2'
sys.dont_write_bytecode = True

import numpy as np
import pandas as pd
from historical_core import checkpoint, compare_frames, fragment, fresh_output, read_matrix, safe_functions, sha, transform


def load_config(path):
    path = Path(path).resolve()
    value = json.loads(path.read_text(encoding='utf-8-sig'))
    allowed = {'baseline_metadata', 'baseline_matrix', 'source_scripts', 'rf_source',
               'microbiomehd_q', 'microbiomehd_effects', 'output_root', 'targets', 'consumer_code'}
    if set(value) - allowed:
        raise ValueError(f'Unknown configuration keys: {set(value) - allowed}')
    for key in set(value) - {'targets'}:
        value[key] = (path.parent / value[key]).resolve()
    value['targets'] = {key: (path.parent / target).resolve() for key, target in value.get('targets', {}).items()}
    return value


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def verify_sources(config, stage):
    """Bind every scientific source and algorithm BEFORE executing any code."""
    contract_path = Path(__file__).with_name('generation_contract.json')
    contract = json.loads(contract_path.read_text(encoding='utf-8'))
    keys = ['baseline_metadata', 'baseline_matrix']
    if stage == 'full': keys += ['microbiomehd_q', 'microbiomehd_effects']
    actual = {}
    for key in keys:
        path = config[key]; expected = contract['inputs'][key]; observed = sha(path)
        accepted = observed == expected['sha256']
        if not accepted and key == 'baseline_matrix':
            sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
            from portable import canonical_matrix_sha
            accepted = canonical_matrix_sha(path) == expected['canonical_sha256']
        if not accepted: raise ValueError('Frozen source mismatch: '+key)
        actual[str(path)] = observed
    if stage != 'matrices':
        for name, expected in contract['algorithms'].items():
            path = config['rf_source'] if name == 'run_rf_clr_sensitivity.py' else config['source_scripts']/name
            observed = sha(path)
            if observed != expected: raise ValueError('Frozen source mismatch: '+name)
            actual[str(path)] = observed
    for path in [contract_path, Path(__file__), Path(__file__).with_name('historical_core.py')]:
        actual[str(path)] = sha(path)
    return actual


def verify_unchanged(hashes):
    for path, expected in hashes.items():
        if sha(path) != expected: raise ValueError('Source changed during generation: '+path)


def reconstruct_panels(config, raw, metadata, abundance, clr, full=False):
    scripts = config['source_scripts']
    original = scripts / 'run_analysis.py'
    env = safe_functions(original, ['bh_fdr', 'genus_from_taxon', 'species_from_taxon',
                                   'fit_feature_ols', 'zscore_numeric', 'finalize_design'])
    condition = (metadata.set_index('sample_uid').loc[clr.index, 'study_condition'] == 'CRC').astype(int)
    study = metadata.set_index('sample_uid').loc[clr.index, 'study_name']
    # Historical output helpers are replaced with inert helpers. They return
    # paths because the original main interval wraps them with Path().
    env.update({'metadata': metadata, 'abundance': raw.T, 'filtered_abundance': abundance,
                'clr': clr, 'condition': condition, 'study': study,
                'PROCESSED': Path('inert_processed'), 'RESULTS': Path('inert_results'),
                'write_tsv': lambda frame, path, **kwargs: str(path)})
    fragment(original, 'design = pd.DataFrame({', 'assoc = assoc.sort_values(', env)
    fragment(original, 'stability_rows =', 'stability = pd.DataFrame(stability_rows)', env)
    assoc, stability = env['assoc'], env['stability']
    stable = stability[(stability.adjusted_qvalue < .10) & (stability.stability_fraction >= .60)]
    robustness = scripts / 'build_robustness_interpretation_analysis.py'
    rob = safe_functions(robustness, ['bh_fdr', 'genus_from_taxon', 'species_from_taxon',
                                    'fit_feature_ols', 'make_design', 'clr_transform', 'association_screen',
                                    'run_leave_one_cohort_association', 'build_candidate_interpretation'])
    print('Bounded LOCO: 11 screens of 301 features', flush=True)
    loco, loco_summary = rob['run_leave_one_cohort_association'](raw.T, metadata, checkpoint(assoc), stable.taxon_long.tolist())
    meta_source = scripts / 'build_meta_heterogeneity_analysis.py'
    meta_functions = safe_functions(meta_source, ['bh_fdr', 'normal_pvalue', 'chi2_sf', 'species_from_taxon',
                                                   'genus_from_taxon', 'per_cohort_effects', 'random_effects_meta', 'build_meta_table'])
    per_cohort = meta_functions['per_cohort_effects'](checkpoint(clr, index=True), metadata.set_index('sample_uid'))
    random = meta_functions['build_meta_table'](per_cohort)
    random['meta_stable_candidate'] = ((random.qvalue_bh_random_effect < .10)
                                      & (random.same_random_direction_fraction >= .60) & (random.i2_percent < 75))
    groups = stable[['taxon_long']].merge(loco_summary[['taxon_long', 'leave_one_cohort_robust']], on='taxon_long')
    groups = groups.merge(checkpoint(random)[['taxon_long', 'qvalue_bh_random_effect', 'i2_percent']], on='taxon_long')
    high = groups.leave_one_cohort_robust & (groups.qvalue_bh_random_effect < .10) & (groups.i2_percent < 75)
    rf_input = groups[['taxon_long']].copy()
    # Only the literal equality high_consistency_candidate is consumed by RF;
    # remaining groups must not pretend to be fully generated interpretation.
    rf_input['interpretation_group'] = np.where(high, 'high_consistency_candidate', 'not_high_consistency_RF_ONLY')
    leak_source = scripts / 'build_leakage_aware_prioritization_analysis.py'
    guild = safe_functions(leak_source, ['parse_taxon', 'guild_membership'])
    def supplied_table(path, **kwargs):
        if Path(path).name == 'cohort_adjusted_crc_control_association.tsv': return checkpoint(assoc)
        if Path(path).name == 'stable_candidate_interpretation.tsv': return rf_input
        raise ValueError(f'Unexpected legacy table read: {path}')
    guild.update({'RESULTS': Path('generated_in_memory'), 'read_tsv': supplied_table})
    membership = guild['guild_membership'](clr)
    rf = safe_functions(config['rf_source'], ['bh', 'association_screen', 'direction_support', 'select_panel'])
    clr_assoc = rf['association_screen'](checkpoint(clr, index=True), metadata.set_index('sample_uid', drop=False))
    prior = {'original_DCP': set(stable.taxon_long), 'original_ICP': set(rf_input.loc[high, 'taxon_long'])}
    sets = []
    for threshold in (.60, .70, .80):
        features = rf['select_panel'](clr_assoc, threshold)
        selected = set(features)
        row = {'panel_type': f'CLR-DCP{int(threshold * 100)}', 'direction_threshold': threshold,
               'panel_size': len(features), 'selected_taxa': ';'.join(features)}
        for name in ('DCP', 'ICP'):
            reference = prior['original_' + name]
            n = len(selected & reference)
            row.update({f'original_{name}_size': len(reference), f'intersection_original_{name}': n,
                        f'jaccard_original_{name}': n / len(selected | reference)})
        # Preserve the original target's field order (a later source-only
        # selection_rule annotation is deliberately excluded).
        sets.append(row)
    sets = pd.DataFrame(sets)[['panel_type', 'direction_threshold', 'panel_size', 'selected_taxa',
                               'original_DCP_size', 'original_ICP_size', 'intersection_original_DCP',
                               'jaccard_original_DCP', 'intersection_original_ICP', 'jaccard_original_ICP']]
    tables = {'association': assoc, 'stability': stability, 'loco': loco, 'loco_summary': loco_summary,
              'random_effects': random, 'per_cohort_effects': per_cohort,
              'legacy_membership': membership, 'old_clr_sets': sets, 'rf_consumed_fields': rf_input}
    models = None
    if full:
        print('G: covariate OLS; E: independent MicrobiomeHD genus context', flush=True)
        fragment(original, 'metadata_indexed =', 'covariate_compare_path =', env)
        env.update({'MICROBIOMEHD_Q': config['microbiomehd_q'], 'MICROBIOMEHD_EFFECTS': config['microbiomehd_effects'],
                    'MICROBIOMEHD_DIR': config['microbiomehd_q'].parent})
        fragment(original, 'schema_rows =', 'overlap_path =', env)
        if env['overlap'].empty or env['schema_rows'][0]['status'] != 'READABLE':
            raise ValueError('MicrobiomeHD context reconstruction failed')
        print('H: 11 historical logistic fits; no RF fits', flush=True)
        fragment(original, 'performance_rows =', 'transport_path =', env)
        ecology = safe_functions(scripts / 'build_ecology_transfer_analysis.py', [], ['ORAL_ASSOCIATED_GENERA'])
        oral = pd.DataFrame({'taxon_long': [taxon for taxon in raw.index
                            if env['genus_from_taxon'](taxon) in ecology['ORAL_ASSOCIATED_GENERA']]})
        interpretation = rob['build_candidate_interpretation'](checkpoint(stability), checkpoint(env['covariate_compare']),
                                      checkpoint(env['transport']), checkpoint(random), oral,
                                      checkpoint(env['overlap']), loco_summary)
        tables.update({'covariate': env['covariate_compare'], 'microbiomehd_overlap': env['overlap'],
                       'transport': env['transport'], 'feature_weights': env['feature_weights'],
                       'legacy_interpretation': interpretation})
        model = env['model'].named_steps['logisticregression']
        models = {'parameters': model.get_params(), 'fitted_l1_ratio': getattr(model, 'l1_ratio_', None),
                  'source_has_explicit_penalty': False, 'original_model_label': 'elasticnet',
                  'n_fits': len(env['performance_rows'])}
    return tables, models


def generate(config, stage):
    started = time.perf_counter()
    required = ['baseline_matrix', 'baseline_metadata']
    if stage != 'matrices': required += ['source_scripts', 'rf_source']
    if stage == 'full': required += ['microbiomehd_q', 'microbiomehd_effects']
    if any(key not in config or not config[key].exists() for key in required):
        raise FileNotFoundError(f'Missing required source inputs for {stage}: {required}')
    source_start_hashes = verify_sources(config, stage)
    for key in ('baseline_matrix', 'baseline_metadata', 'microbiomehd_q', 'microbiomehd_effects'):
        if key in config and config[key] in config['targets'].values():
            raise ValueError('A target cannot be its own upstream input')
    output = fresh_output(config['output_root'], [config[key] for key in required] + list(config['targets'].values()))
    write_json(output/'source_start_receipt.json', {'stage':stage, 'inputs':source_start_hashes})
    metadata = pd.read_csv(config['baseline_metadata'], sep='\t')
    raw = read_matrix(config['baseline_matrix'])
    abundance, clr, pseudocount = transform(raw, metadata)
    names = {'legacy_abundance': 'filtered_species_abundance_percent.tsv.gz',
             'legacy_clr': 'filtered_species_clr.tsv.gz', 'legacy_membership': 'ecological_guild_membership.csv',
             'old_clr_sets': 'clr_dcp_global_sets.tsv', 'legacy_interpretation': 'stable_candidate_interpretation.tsv',
             'rf_consumed_fields': 'rf_consumed_fields_NOT_FULL_INTERPRETATION.tsv'}
    mapping = {}
    for key, table in [('legacy_abundance', abundance), ('legacy_clr', clr)]:
        table.to_csv(output / names[key], sep='\t', index=True, compression={'method': 'gzip', 'mtime': 0})
        mapping[key] = names[key]
    print(f'Matrices generated: {clr.shape}; pseudocount={pseudocount}', flush=True)
    models, recorded_warnings = None, []
    if stage != 'matrices':
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            tables, models = reconstruct_panels(config, raw, metadata, abundance, clr, stage == 'full')
        recorded_warnings = sorted(set(str(item.message) for item in caught))
        for key, table in tables.items():
            filename = names.get(key, key + '.tsv')
            table.to_csv(output / filename, sep=',' if filename.endswith('.csv') else '\t', index=False)
            mapping[key] = filename
    comparisons = {}
    # Targets are first parsed here, after all generation completes.
    for key, target in config['targets'].items():
        if not target.is_file():
            comparisons[key] = {'pass': False, 'status': 'comparison_target_missing'}
            continue
        if key not in mapping:
            comparisons[key] = {'pass': False, 'status': 'not_generated'}
            continue
        index = 0 if key in ('legacy_clr', 'legacy_abundance') else None
        delimiter = ',' if target.suffix == '.csv' else '\t'
        generated = pd.read_csv(output / mapping[key], sep=delimiter, index_col=index)
        reference = pd.read_csv(target, sep=delimiter, index_col=index)
        result = compare_frames(generated, reference, 1e-12 if index is not None else 1e-10)
        result.update({'target_sha256': sha(target), 'generated_sha256': sha(output / mapping[key])})
        comparisons[key] = result
    write_json(output / 'output_mapping.json', mapping)
    write_json(output / 'comparisons.json', comparisons)
    sources = [config[key] for key in required if config[key].is_file()]
    if stage != 'matrices':
        sources += [config['source_scripts'] / name for name in ['run_analysis.py', 'build_robustness_interpretation_analysis.py',
                      'build_meta_heterogeneity_analysis.py', 'build_leakage_aware_prioritization_analysis.py']]
    if stage == 'full': sources += [config['source_scripts'] / 'build_ecology_transfer_analysis.py']
    verify_unchanged(source_start_hashes)
    receipt = {'stage': stage, 'source_preflight_verified':True, 'sources_unchanged':True,
               'comparison_performed':bool(comparisons),
               'raw_shape': list(raw.shape), 'generated_matrix_shape': list(clr.shape),
               'pseudocount_percent_units': pseudocount, 'filter': 'prevalence >= 0.05; present in >= 3 studies',
               'unit': 'relative abundance percent; no renormalization', 'elapsed_seconds': time.perf_counter()-started,
               'source_manifest': [{'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha(path)} for path in sources],
               'output_manifest': [{'key': key, 'path': name, 'sha256': sha(output/name)} for key, name in mapping.items()],
               'packages': {name: importlib.metadata.version(name) for name in ['numpy','pandas','scipy','statsmodels','scikit-learn']},
               'model': models, 'warnings': recorded_warnings,
               'status': 'PASS' if all(item['pass'] for item in comparisons.values()) else 'DIFFERENCES_OR_PARTIAL'}
    write_json(output / 'generation_receipt.json', receipt)
    print(json.dumps({'status': receipt['status'], 'comparison': {key: value['pass'] for key, value in comparisons.items()},
                      'elapsed_seconds': receipt['elapsed_seconds'], 'output': str(output)}, indent=2), flush=True)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('stage', choices=['matrices', 'panels', 'full'])
    args = parser.parse_args()
    receipt = generate(load_config(args.config), args.stage)
    if receipt['status'] != 'PASS': raise SystemExit(2)


if __name__ == '__main__':
    main()
