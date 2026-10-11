"""Offline reconstruction of dated CRC accession identity intermediates."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
from pathlib import Path
import re

import pandas as pd
from identity_core import resolve_training

HERE = Path(__file__).resolve().parent
CONTRACT = HERE / 'identity_source_contract.json'
PROJECTS = {'PRJNA447983': {'ThomasAM_2018a', 'ThomasAM_2018b'},
            'PRJEB27928': {'WirbelJ_2018'}}
FIELDS = ['sample_uid', 'study_name', 'run_accession', 'sample_accession',
          'secondary_sample_accession', 'sample_alias', 'match_basis']
ENA_FIELDS = ['run_accession', 'study_accession', 'secondary_study_accession',
              'sample_accession', 'secondary_sample_accession', 'sample_alias',
              'sample_title', 'experiment_alias', 'library_name']
BASIS = 'same primary-source project; exact normalized supplied sample or subject alias'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_tsv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream, delimiter='\t')
        if not reader.fieldnames or len(set(reader.fieldnames)) != len(reader.fieldnames):
            raise ValueError('Duplicate or absent TSV header')
        rows = list(reader)
        if any(None in row or any(value is None for value in row.values()) for row in rows):
            raise ValueError('Malformed TSV row')
        return rows, reader.fieldnames


def tokens(row):
    def norm(value):
        return re.sub(r'[^A-Z0-9]', '', str(value).upper())
    values = []
    for key in ('sample_id', 'subject_id', 'NCBI_accession', 'Name', 'Sample Source ID',
                'run_accession', 'sample_accession', 'secondary_sample_accession',
                'sample_alias', 'experiment_alias', 'library_name'):
        values.extend(re.split(r'[;]+', row.get(key, '')))
    values.extend(re.findall(r'\b(?:V\d{9}|SAM[NED][A-Z]*\d+|[EDS]R[RS]\d+)\b',
                             row.get('sample_title', '')))
    return {norm(value) for value in values if value and len(norm(value)) >= 2
            and norm(value) not in {'UNSPECIFIED', 'NA', 'NONE', 'MISSING'}}


def build_enriched(original, snapshots):
    if set(snapshots) != set(PROJECTS):
        raise ValueError('Both pinned projects are required')
    original_by_uid = {}
    for row in original:
        uid = row['sample_uid']
        if not uid or uid in original_by_uid:
            raise ValueError('Duplicate or empty original UID')
        original_by_uid[uid] = row
    enriched = []
    run_keys = set()
    for project, studies in PROJECTS.items():
        local_index = defaultdict(set)
        for row in original:
            if row['study_name'] in studies:
                for token in tokens(row):
                    local_index[token].add(row['sample_uid'])
        for row in snapshots[project]:
            run = row['run_accession']
            if not re.fullmatch(r'[EDS]RR\d+', run):
                raise ValueError('Invalid public run accession')
            if run in run_keys:
                raise ValueError('Duplicate public run accession')
            run_keys.add(run)
            uids = {uid for token in tokens(row) for uid in local_index.get(token, set())}
            if len(uids) > 1:
                raise ValueError('Ambiguous original sample alias match in ' + project)
            for uid in uids:
                enriched.append({'sample_uid': uid, 'study_name': original_by_uid[uid]['study_name'],
                                 **{key: row[key] for key in FIELDS[2:6]}, 'match_basis': BASIS})
    return enriched


def validate_source(path, provenance, expected):
    path = Path(path)
    if path.stat().st_size != expected['bytes'] or sha(path) != expected['sha256']:
        raise ValueError('Source snapshot size/hash differs: ' + str(path))
    data = json.loads(Path(provenance).read_text(encoding='utf-8-sig'))
    if any(data.get(key) != expected[key] for key in ('bytes', 'sha256', 'url')) or not data.get('retrieved_utc'):
        raise ValueError('Snapshot provenance differs from frozen source')
    if expected.get('provenance_sha256') and sha(provenance) != expected['provenance_sha256']:
        raise ValueError('Snapshot provenance hash differs from frozen source')
    return data


def validate_output(output, inputs):
    out = Path(output).resolve()
    for source in inputs:
        source = Path(source).resolve()
        if out == source or out in source.parents or source in out.parents:
            raise ValueError('Output/input overlap is not permitted')
    if out.exists():
        raise ValueError('Output exists; preserve earlier generation')
    return out


def resolve_audit(training, enriched):
    return resolve_training(training, enriched)


def validate_ena_fields(fields):
    if fields != ENA_FIELDS:
        raise ValueError('ENA snapshot field order differs')
    return True


def load_config(path, stage):
    path = Path(path).resolve()
    data = json.loads(path.read_text(encoding='utf-8-sig'))
    keys = ({'original_metadata', 'project_snapshots', 'output_root'} if stage == 'enrich'
            else {'training_manifest', 'enriched_manifest', 'enrichment_receipt', 'output_root'})
    if set(data) != keys:
        raise ValueError('Configuration keys must be exactly: ' + str(sorted(keys)))
    paths = {key: (path.parent / value).resolve() for key, value in data.items() if key != 'project_snapshots'}
    if stage == 'enrich':
        if set(data['project_snapshots']) != set(PROJECTS):
            raise ValueError('Both pinned projects are required')
        paths['project_snapshots'] = {}
        for project, item in data['project_snapshots'].items():
            if set(item) != {'path', 'provenance'}:
                raise ValueError('Each project needs path and provenance')
            paths['project_snapshots'][project] = {key: (path.parent / value).resolve() for key, value in item.items()}
    return path, paths


def source_hashes(paths):
    return {str(path): sha(path) for path in paths}


def recheck(hashes):
    for path, digest in hashes.items():
        if sha(path) != digest:
            raise ValueError('Source changed during generation: ' + path)


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def enrich(config_path):
    config_path, cfg = load_config(config_path, 'enrich')
    contract = json.loads(CONTRACT.read_text(encoding='utf-8'))
    sources = [config_path, Path(__file__), CONTRACT, cfg['original_metadata']]
    for item in cfg['project_snapshots'].values():
        sources.extend(item.values())
    dest = validate_output(cfg['output_root'], sources)
    if sha(cfg['original_metadata']) != contract['original_metadata']['sha256']:
        raise ValueError('Original metadata hash differs from pinned historical source')
    original, fields = read_tsv(cfg['original_metadata'])
    if len(original) != 1395 or len(fields) != 45:
        raise ValueError('Original metadata dimensions differ')
    snapshots, versions = {}, {}
    for project in PROJECTS:
        item = cfg['project_snapshots'][project]
        versions[project] = validate_source(item['path'], item['provenance'], contract['projects'][project])
        snapshots[project], fields = read_tsv(item['path'])
        validate_ena_fields(fields)
    hashes = source_hashes(sources)
    rows = build_enriched(original, snapshots)
    if len(rows) != 373 or len({row['sample_uid'] for row in rows}) != 195:
        raise ValueError('Historical enriched row/UID counts differ')
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, FIELDS, delimiter='\t', lineterminator='\r\n')
    writer.writeheader(); writer.writerows(rows)
    payload = stream.getvalue().encode('utf-8')
    recheck(hashes)
    target = dest / '05_review/original_enriched_identity_manifest.tsv'
    target.parent.mkdir(parents=True)
    target.write_bytes(payload)
    recheck(hashes)
    receipt = {'stage': 'identity-enrich', 'status': 'PASS', 'created_utc': pd.Timestamp.now(tz='UTC').isoformat(),
               'inputs': hashes, 'source_versions': versions, 'project_studies': {p: sorted(s) for p,s in PROJECTS.items()},
               'generator_origin': contract['generator_origin'], 'rows': len(rows), 'unique_uids': 195,
               'unique_runs': len({r['run_accession'] for r in rows}),
               'rows_by_study': dict(Counter(r['study_name'] for r in rows)),
               'output': {'path': str(target), 'sha256': sha(target), 'fields': FIELDS},
               'scope': 'historical accession-linkage reproduction; no universal participant-independence claim'}
    write_json(dest / 'enrichment_receipt.json', receipt)
    return receipt


def audit(config_path):
    config_path, cfg = load_config(config_path, 'audit')
    contract = json.loads(CONTRACT.read_text(encoding='utf-8'))
    core = HERE / 'identity_core.py'
    sources = [config_path, Path(__file__), CONTRACT, core, cfg['training_manifest'],
               cfg['enriched_manifest'], cfg['enrichment_receipt']]
    dest = validate_output(cfg['output_root'], sources)
    if sha(core) != contract['identity_core_sha256']:
        raise ValueError('Identity core differs from supplied exact snapshot')
    if sha(cfg['training_manifest']) != contract['training_manifest_sha256']:
        raise ValueError('Historical training manifest hash differs')
    receipt = json.loads(cfg['enrichment_receipt'].read_text(encoding='utf-8'))
    if receipt.get('status') != 'PASS' or receipt.get('stage') != 'identity-enrich':
        raise ValueError('A successful generated enrichment receipt is required')
    if Path(receipt['output']['path']).resolve() != cfg['enriched_manifest'] or sha(cfg['enriched_manifest']) != receipt['output']['sha256']:
        raise ValueError('Enriched manifest does not match generation receipt')
    hashes = source_hashes(sources)
    training = pd.read_csv(cfg['training_manifest'], sep='\t', keep_default_na=False)
    enriched = pd.read_csv(cfg['enriched_manifest'], sep='\t', keep_default_na=False)
    out = resolve_audit(training, enriched)
    bases = out.resolution_basis.value_counts().to_dict()
    if len(out) != 1257 or int(out.accession_resolved.sum()) != 1214 or bases != {'direct':595,'biosample_id':508,'enriched':111,'alias_only':43}:
        raise ValueError('Historical identity count/basis constraints differ')
    if set(out.loc[~out.accession_resolved, 'study_name']) != {'WirbelJ_2018'}:
        raise ValueError('Historical excluded study differs')
    frames = {'training_identity_audit.tsv': out, 'training_manifest.tsv': out[out.accession_resolved],
              'excluded_training.tsv': out[~out.accession_resolved]}
    recheck(hashes)
    (dest / 'results').mkdir(parents=True)
    for name, frame in frames.items():
        frame.to_csv(dest / 'results' / name, sep='\t', index=False, float_format='%.17g', lineterminator='\r\n')
    recheck(hashes)
    result = {'stage':'identity-audit', 'status':'PASS', 'created_utc':pd.Timestamp.now(tz='UTC').isoformat(),
              'inputs': hashes, 'basis_counts':bases, 'full_n':1257, 'resolved_n':1214, 'alias_only_n':43,
              'files': {name:sha(dest / 'results' / name) for name in frames},
              'scope':'post-outcome identity sensitivity preparation; broader source/clinical audit not generated'}
    write_json(dest / 'audit_receipt.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('stage', choices=['enrich', 'audit'])
    args = parser.parse_args()
    result = globals()[args.stage](args.config)
    print(args.stage, result['status'])
