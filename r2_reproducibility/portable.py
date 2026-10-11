"""Offline, path-configurable R2 replay. Run one explicit stage at a time.

This is a computational replay of completed analyses, not a new prospective
validation or a reconstruction of the original preregistration chronology.
"""
from pathlib import Path
import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import sys
from datetime import datetime, timezone

# Set before numerical imports; model-level RF uses n_jobs=1 as originally fixed.
for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[variable] = '2'
sys.dont_write_bytecode = True
PACKAGE = Path(__file__).resolve().parent
CODE = PACKAGE / 'code'
MARKER = '.portable-r2-replay.json'
PATH_KEYS = {'source_root', 'baseline_metadata', 'baseline_matrix', 'legacy_clr',
             'legacy_abundance', 'legacy_interpretation', 'legacy_membership',
             'old_clr_sets', 'output_root'}


def load_config(path):
    path = Path(path).resolve()
    config = json.loads(path.read_text(encoding='utf-8-sig'))
    unknown = set(config) - PATH_KEYS
    if unknown:
        raise ValueError(f'Unknown configuration keys: {sorted(unknown)}')
    return {key: (path.parent / value).resolve() for key, value in config.items() if value}


def check_output(config):
    output = config['output_root'].resolve()
    protected = [PACKAGE.resolve()]
    for key, value in config.items():
        if key != 'output_root':
            value = value.resolve()
            protected.append(value if key == 'source_root' else value.parent)
    for source in protected:
        if output == source or source in output.parents:
            raise ValueError('Replay output cannot be inside source inputs or this code package')
    if output.exists() and not (output / MARKER).exists():
        raise ValueError('Existing unmarked output directory refused; choose a fresh output')
    return output


def check_code(root=CODE):
    for name in ('statistics_reference.py', 'run_external.py', 'external_core.py',
                 'run_multicohort.py', 'multicohort_core.py'):
        if not (Path(root) / name).is_file():
            raise FileNotFoundError(f'Incomplete code package: {name}')


def missing_inputs(config, required):
    return [key for key in required if key not in config or not config[key].exists()]


def verify_contract(config, required, contract=PACKAGE / 'input_contract.tsv', overrides=None):
    import csv
    checked = []
    with Path(contract).open(encoding='utf-8', newline='') as handle:
        for row in csv.DictReader(handle, delimiter='\t'):
            key, _, suffix = row['input_key'].partition('/')
            if key not in required:
                continue
            path = config[key] / suffix if suffix else config[key]
            if overrides and row['input_key'] in overrides:
                path = Path(overrides[row['input_key']])
            dynamic_record = (row.get('hash_policy') == 'archive_checksums_verified_by_audit'
                              and row['input_key'] == 'source_root/zenodo_15069069.json')
            accepted = path.is_file() and (dynamic_record or sha(path) == row['sha256'])
            if (not accepted and path.is_file() and row.get('canonical_sha256')
                    and key in {'baseline_matrix', 'legacy_clr', 'legacy_abundance'}):
                accepted = canonical_matrix_sha(path) == row['canonical_sha256']
            if not accepted:
                raise ValueError(f'Frozen input mismatch: {row["input_key"]}; no substitution allowed')
            checked.append(row['input_key'])
    return checked


def canonical_matrix_sha(path):
    """Bind exact parsed axis identities/order and float64 matrix values.

    Supports faithful source reconstruction despite gzip timestamp and decimal
    serialization differences. This is not a fuzzy or numeric-tolerance gate.
    """
    import numpy as np
    import pandas as pd
    matrix = pd.read_csv(path, sep='\t', index_col=0, float_precision='round_trip')
    if matrix.index.has_duplicates or matrix.columns.has_duplicates:
        raise ValueError('Duplicate reconstructed matrix axes')
    values = np.asarray(matrix.to_numpy(), dtype='<f8')
    if not np.isfinite(values).all(): raise ValueError('Nonfinite reconstructed matrix')
    header = json.dumps({'rows': matrix.index.tolist(), 'columns': matrix.columns.tolist(),
                         'index_name': matrix.index.name}, ensure_ascii=False, separators=(',', ':')).encode('utf-8')
    return hashlib.sha256(header + b'\n' + values.tobytes(order='C')).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def start(config, stage):
    check_code()
    output = check_output(config)
    required = ['baseline_metadata']
    if stage.startswith(('mp4', 'multi')): required += ['source_root']
    elif stage.startswith('nested'): required += ['baseline_matrix']
    elif stage == 'historical-clr': required += ['legacy_clr', 'legacy_abundance', 'old_clr_sets']
    elif stage == 'historical-context': required += ['legacy_clr', 'legacy_membership']
    elif stage == 'historical-rf': required += ['legacy_clr', 'legacy_interpretation']
    missing = missing_inputs(config, required)
    if missing:
        raise FileNotFoundError(f'Missing required inputs: {missing}; no analysis started')
    verify_contract(config, required)
    # Marker binds explicit input locations; do not silently redirect a prior replay.
    bound = {key: str(value) for key, value in config.items()}
    marker = output / MARKER
    if marker.exists() and json.loads(marker.read_text())['configuration'] != bound:
        raise ValueError('Input configuration changed for this replay; use a new output folder')
    if not marker.exists():
        write_json(marker, {'configuration': bound, 'kind': 'post-hoc computational replay',
                            'created_utc': datetime.now(timezone.utc).isoformat(),
                            'original_chronology_recreated': False})
    return output


def configure_external(config, output):
    sys.path.insert(0, str(CODE))
    run = importlib.import_module('run_external')
    run.ROOT = output / 'mp4_initial'
    run.SOURCE = config['source_root']
    run.EXTRACTED = run.SOURCE / 'extracted'
    run.BASEMETA = config['baseline_metadata']
    run.REFERENCE = CODE / 'statistics_reference.py'
    run.INP = run.ROOT / '01_inputs'
    run.OUT = run.ROOT / '02_analysis/results'
    run.LOCK = run.ROOT / '02_analysis/model_lock'
    protocol = run.ROOT / '00_receipts/PROTOCOL.md'
    if not protocol.exists():
        protocol.parent.mkdir(parents=True, exist_ok=True)
        protocol.write_bytes((PACKAGE / 'protocols/mp4_replay.md').read_bytes())
    return run


def configure_multi(config, output):
    old = configure_external(config, output)
    run = importlib.import_module('run_multicohort')
    run.old = old
    run.ROOT = output / 'mp4_multicohort'
    run.OLD = old.ROOT
    run.INP = run.ROOT / '01_inputs'
    run.OUT = run.ROOT / '02_analysis/results'
    run.REC = run.ROOT / '00_receipts'
    protocol = run.REC / 'PROTOCOL.md'
    if not protocol.exists():
        protocol.parent.mkdir(parents=True, exist_ok=True)
        protocol.write_bytes((PACKAGE / 'protocols/multicohort_replay.md').read_bytes())
    return run


def run_nested(config, output, smoke=False, bootstrap=2000):
    sys.path.insert(0, str(CODE))
    module = importlib.import_module('statistics_reference')
    module.MATRIX = config['baseline_matrix']
    module.META = config['baseline_metadata']
    module.OUT = output / ('nested_smoke' if smoke else 'nested_full')
    if module.OUT.exists():
        raise ValueError('Nested destination exists; use a fresh output configuration')
    sys.argv = ['statistics_reference.py', '--bootstrap', str(bootstrap)] + (['--smoke'] if smoke else [])
    module.main()
    receipt = json.loads((module.OUT / 'nested_loso_receipt.json').read_text())
    if receipt['status'] != 'PASS':
        raise RuntimeError(f'Nested replay incomplete: {receipt["errors"]}')


def run_historical(config, output, stage, smoke=False):
    sys.path.insert(0, str(CODE))
    name = {'historical-clr': 'historical_clr_audit', 'historical-context': 'historical_context',
            'historical-rf': 'historical_rf_clr'}[stage]
    module = importlib.import_module(name)
    dest = output / stage.replace('-', '_')
    if dest.exists():
        raise ValueError('Historical replay destination exists; use a fresh output configuration')
    module.OUT = module.OUT_DIR = dest
    module.META = module.META_PATH = config['baseline_metadata']
    module.CLR = module.CLR_PATH = config['legacy_clr']
    if stage == 'historical-clr':
        module.ABUND = config['legacy_abundance']; module.OLD_SET = config['old_clr_sets']
    elif stage == 'historical-context':
        module.MEMBERSHIP = config['legacy_membership']
    else:
        module.INTERPRETATION_PATH = config['legacy_interpretation']
        sys.argv = ['historical_rf_clr.py', '--mode', 'smoke' if smoke else 'full']
    module.main()


def run_matched(config, output, smoke=False, bootstrap=2000):
    sys.path.insert(0, str(CODE))
    nested = output / ('nested_smoke' if smoke else 'nested_full')
    if not (nested / 'nested_loso_receipt.json').exists():
        raise FileNotFoundError('Run the corresponding nested stage first')
    normalization = importlib.import_module('normalize_membership')
    normalization.SRC = nested / 'nested_loso_panel_membership.tsv'
    normalization.DST = nested / 'nested_loso_panel_membership_long.tsv'
    normalization.main()
    module = importlib.import_module('matched_size')
    module.MATRIX = config['baseline_matrix']; module.META = config['baseline_metadata']
    module.NESTED = nested; module.HERE = nested / 'matched_size'
    if module.HERE.exists():
        raise ValueError('Matched-size destination exists; use a fresh output configuration')
    reference = importlib.import_module('statistics_reference')
    module.load_nested_module = lambda: reference
    sys.argv = ['matched_size.py', '--bootstrap', str(bootstrap)] + (['--smoke'] if smoke else [])
    module.main()
    receipt = json.loads((module.HERE / 'matched_size_receipt.json').read_text())
    if receipt['status'] != 'PASS':
        raise RuntimeError(f'Matched-size replay incomplete: {receipt["errors"]}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('stage', choices=['inspect', 'mp4-audit', 'mp4-fit', 'mp4-predict', 'mp4-evaluate', 'mp4-verify',
        'multi-audit', 'multi-predict', 'multi-evaluate', 'multi-replicate', 'nested', 'nested-matched', 'historical-clr', 'historical-context', 'historical-rf'])
    parser.add_argument('--cohort', choices=['LiuNN_2022', 'YangY_2021', 'IIGM_TU', 'YangJ_2020', 'NHSII', 'IIGM_CZ'])
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--bootstrap', type=int, default=2000, help='Nested branch only; changing 2000 does not reproduce historical CIs')
    args = parser.parse_args()
    config = load_config(args.config)
    output = start(config, args.stage)
    if args.stage == 'inspect':
        import csv
        rows = [{'input_key': key, 'bytes': value.stat().st_size, 'sha256': sha(value)}
                for key, value in config.items() if key != 'output_root' and value.is_file()]
        with (output / 'input_manifest.tsv').open('w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['input_key', 'bytes', 'sha256'], delimiter='\t')
            writer.writeheader(); writer.writerows(rows)
        write_json(output / 'environment.json', {'python': sys.version, 'platform': platform.platform(),
            'packages': {p: importlib.metadata.version(p) for p in ['numpy', 'pandas', 'scipy', 'statsmodels', 'scikit-learn', 'joblib']},
            'thread_limit': 2, 'historical_environment_reconstructed': False})
    elif args.stage.startswith('mp4-'):
        run = configure_external(config, output)
        getattr(run, args.stage.split('-')[1])()
    elif args.stage.startswith('multi-'):
        run = configure_multi(config, output)
        action = args.stage.split('-')[1]
        if action != 'audit' and not args.cohort: parser.error('--cohort required')
        getattr(run, action)(*([args.cohort] if action != 'audit' else []))
    elif args.stage == 'nested':
        run_nested(config, output, args.smoke, args.bootstrap)
    elif args.stage == 'nested-matched':
        run_matched(config, output, args.smoke, args.bootstrap)
    else:
        run_historical(config, output, args.stage, args.smoke)
    write_json(output / 'stage_receipts' / (args.stage + ('_' + args.cohort if args.cohort else '') + '.json'),
               {'stage': args.stage, 'cohort': args.cohort, 'status': 'completed',
                'completed_utc': datetime.now(timezone.utc).isoformat(), 'replay_not_new_validation': True,
                'author_manual_verification': 'not supplied', 'runner_sha256': sha(__file__)})


if __name__ == '__main__':
    main()

