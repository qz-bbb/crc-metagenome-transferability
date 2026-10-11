"""Fixed post-outcome 2x2 decomposition. Writes only to a new output folder."""
import argparse
import hashlib
import itertools
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from definition_core import DEFINITIONS, ZERO_TOLERANCE, agreement, concordance, effect_grid

CLASSIC = ['Fusobacterium_nucleatum', 'Parvimonas_micra', 'Gemella_morbillorum', 'Dialister_pneumosintes', 'Peptostreptococcus_stomatis', 'Solobacterium_moorei', 'Bacteroides_fragilis']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path, **kwargs):
    return pd.read_csv(path, sep='\t', float_precision='round_trip', **kwargs)


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def profile(path, ids, features):
    a = read(path, index_col=0)
    if a.index.duplicated().any() or a.columns.duplicated().any() or pd.Index(ids).duplicated().any():
        raise ValueError('Duplicate input axis')
    if not set(ids).issubset(a.columns):
        raise ValueError('Metadata sample not present in profile')
    a = a.loc[a.index.str.rsplit('|', n=1).str[-1].str.startswith('s__'), list(ids)].T
    if not np.isfinite(a).all().all() or (a < 0).any().any() or not a.sum(axis=1).between(99.9, 100.1).all():
        raise ValueError('Source profiles must be finite nonnegative species percent abundances')
    present = np.array([f in a.columns for f in features])
    # Same frozen MP4 source universe only; missing cohort rows mean non-detection.
    # This is never MP3->MP4 identity matching or zero filling unresolved synonyms.
    return a.reindex(columns=features, fill_value=0).to_numpy(), present


def cohort_rows(raw, y, features, ref, pseudo, cohort, branch, present):
    effects, detected = effect_grid(raw, y, pseudo)
    rows = []
    for definition in DEFINITIONS:
        e = effects[definition]
        rows.append(pd.DataFrame(dict(taxon=features, cohort=cohort, branch=branch,
            definition=definition, effect=e, reference_sign=ref, detected=detected,
            explicit_profile_row=present, exact_zero=e == 0, numerical_zero=np.abs(e) <= ZERO_TOLERANCE,
            concordant=concordance(e, ref, detected), n_crc=int(y.sum()), n_control=int((y == 0).sum()),
            detection_crc=(raw[y == 1] > 0).mean(axis=0), detection_control=(raw[y == 0] > 0).mean(axis=0))))
    return pd.concat(rows, ignore_index=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config.resolve()
    cfg = json.loads(config_path.read_text(encoding='utf-8'))
    paths = {k: (config_path.parent / v).resolve() for k, v in cfg.items()}
    out = paths['output']
    if out.exists():
        raise ValueError('Refuse to overwrite an existing result directory')
    lock = paths['model_lock']
    prep = json.loads((lock / 'preprocessing.json').read_text())
    features, pseudo = prep['features'], prep['pseudocount']
    evidence = read(lock / 'feature_evidence.tsv').set_index('taxon').loc[features].reset_index()
    if len(features) != 800 or evidence.taxon.duplicated().any():
        raise ValueError('Wrong frozen feature universe')
    training = read(paths['training_manifest'])
    inventory = read(paths['inventory'])
    panels = read(lock / 'panels.tsv')
    if len(training) != 1257 or training.sample_uid.duplicated().any() or training.study_name.nunique() != 9:
        raise ValueError('Wrong frozen training population')
    inputs = [config_path, Path(__file__), Path(__file__).with_name('definition_core.py'),
        Path(__file__).with_name('PROTOCOL.md'), lock / 'preprocessing.json', lock / 'feature_evidence.tsv',
        lock / 'panels.tsv', paths['training_manifest'], paths['inventory'],
        paths['prior_sparsity'] / 'training_per_study_sparsity.tsv', paths['prior_sparsity'] / 'external_taxon_sparsity.tsv']
    stems = set(training.source_stem) | set(inventory.source_stem)
    for stem in sorted(stems):
        inputs.append(paths['extracted'] / 'metaphlan4_profiles' / f'{stem}.tsv')
        if stem in set(inventory.source_stem):
            inputs.append(paths['extracted'] / 'metadata' / f'{stem}.tsv')
    before = {str(p): sha(p) for p in inputs}
    out.mkdir(parents=True)
    save_json(out / 'start_receipt.json', dict(created_utc=datetime.now(timezone.utc).isoformat(),
        status='POST_OUTCOME_EXPLANATORY', inputs=before, zero_tolerance=ZERO_TOLERANCE,
        external_results_previously_seen=True, model_fitting=False))
    ref = np.sign(evidence.coef.to_numpy())
    train_rows, external_rows = [], []
    for stem, g in training.groupby('source_stem', sort=True):
        if g.study_name.nunique() != 1:
            raise ValueError('Source stem merges study labels')
        raw, present = profile(paths['extracted'] / 'metaphlan4_profiles' / f'{stem}.tsv', g.sample_id, features)
        y = g.study_condition.eq('CRC').astype(int).to_numpy()
        train_rows.append(cohort_rows(raw, y, features, ref, pseudo, stem, 'training', present))
    train = pd.concat(train_rows, ignore_index=True)
    for inv in inventory.itertuples():
        meta = read(paths['extracted'] / 'metadata' / f'{inv.source_stem}.tsv')
        meta = meta[meta['Study condition'].isin(['CRC', 'Control'])].sort_values('Sample Source ID')
        y = meta['Study condition'].eq('CRC').astype(int).to_numpy()
        if int(y.sum()) != inv.crc or int((y == 0).sum()) != inv.control:
            raise ValueError('External case/control count changed')
        raw, present = profile(paths['extracted'] / 'metaphlan4_profiles' / f'{inv.source_stem}.tsv', meta['Sample Source ID'], features)
        a = cohort_rows(raw, y, features, ref, pseudo, inv.cohort, 'external', present)
        a['role'] = inv.role
        external_rows.append(a)
    external = pd.concat(external_rows, ignore_index=True)
    summary = train.groupby(['taxon', 'definition'], sort=False).agg(
        eligible_studies=('concordant', 'size'), concordant_studies=('concordant', 'sum'),
        direction_agreement=('concordant', 'mean'), exact_zero_studies=('exact_zero', 'sum'),
        numerical_zero_studies=('numerical_zero', 'sum'), detected_studies=('detected', 'sum')).reset_index()
    if not summary.eligible_studies.eq(9).all():
        raise ValueError('Direction denominator drift')
    base = evidence.drop(columns='direction_agreement')
    summary = summary.merge(base, on='taxon', validate='many_to_one')
    summary['H'] = summary.re_q.lt(.1) & summary.i2.lt(75)
    summary['R'] = summary.inner_sign_agreement.ge(.9) & summary.inner_q_fraction.ge(.8)
    sets, members, counts = {}, [], []
    for definition, g in summary.groupby('definition', sort=False):
        g = g.sort_values(['q', 'taxon'])
        for cutoff in [.6, .7, .8]:
            a = g[g.q.lt(.1) & g.direction_agreement.ge(cutoff)]
            b = a[a.H & a.R]
            for criterion, selected, cap in [('DCP', a, 29), ('ICP', b, 18)]:
                sets[definition, cutoff, criterion] = set(selected.taxon)
                counts.append(dict(definition=definition, threshold=cutoff, criterion=criterion,
                    fdr_universe=800, association_eligible=int(g.q.lt(.1).sum()), direction_denominator=9,
                    n_eligible=len(selected), n_capped=min(cap, len(selected))))
                for rank, row in enumerate(selected.itertuples(), 1):
                    members.append(dict(definition=definition, threshold=cutoff, criterion=criterion,
                        rank=rank, taxon=row.taxon, q=row.q, agreement=row.direction_agreement, in_diagnostic_cap=rank <= cap))
    intersections = []
    for cutoff in [.6, .7, .8]:
        for criterion in ['DCP', 'ICP']:
            for left, right in itertools.combinations(DEFINITIONS, 2):
                a, b = sets[left, cutoff, criterion], sets[right, cutoff, criterion]
                intersections.append(dict(left=left, right=right, threshold=cutoff, criterion=criterion,
                    n_left=len(a), n_right=len(b), intersection=len(a & b), union=len(a | b),
                    jaccard=len(a & b)/len(a | b) if a | b else 1.))
    panel_rows = []
    for (cohort, definition), g in external.groupby(['cohort', 'definition'], sort=False):
        for panel, p in panels.groupby('panel', sort=False):
            s = g[g.taxon.isin(p.taxon)]
            if len(s) != len(p):
                raise ValueError('Incomplete frozen panel')
            panel_rows.append(dict(cohort=cohort, role=g.role.iloc[0], definition=definition, panel=panel,
                n_taxa=len(s), n_concordant=int(s.concordant.sum()), n_zero=int(s.numerical_zero.sum()),
                n_undetected=int((~s.detected).sum()), agreement=float(s.concordant.mean())))
    # Verify both previously computed effects, without reusing their calculations.
    checks = []
    for branch, current, oldname in [('training', train, 'training_per_study_sparsity.tsv'), ('external', external, 'external_taxon_sparsity.tsv')]:
        old = read(paths['prior_sparsity'] / oldname).set_index(['cohort', 'taxon'])
        for definition, oldcol, oldflag in [('raw_median', 'raw_median_difference', 'raw_concordant'), ('clr_mean', 'mean_clr_difference', 'clr_concordant')]:
            new = current[current.definition.eq(definition)].set_index(['cohort', 'taxon']).loc[old.index]
            delta = float(np.max(np.abs(new.effect.to_numpy() - old[oldcol].to_numpy())))
            np.testing.assert_allclose(new.effect, old[oldcol], rtol=1e-10, atol=1e-10)
            changes = int((new.concordant.to_numpy() != old[oldflag].to_numpy()).sum())
            checks.append(dict(branch=branch, definition=definition, n=len(new), max_effect_difference=delta,
                concordance_guard_changes=changes))
    raw = summary[summary.definition.eq('raw_median')].set_index('taxon').loc[features]
    np.testing.assert_allclose(raw.direction_agreement, evidence.direction_agreement, rtol=0, atol=1e-12)
    summary['species'] = summary.taxon.str.rsplit('|s__', n=1).str[-1]
    classic = summary[summary.species.isin(CLASSIC)].copy()
    outputs = {'training_effects.tsv': train, 'external_effects.tsv': external,
        'training_definition_summary.tsv': summary, 'candidate_counts.tsv': pd.DataFrame(counts),
        'candidate_membership.tsv': pd.DataFrame(members), 'definition_intersections.tsv': pd.DataFrame(intersections),
        'external_frozen_panel_summary.tsv': pd.DataFrame(panel_rows), 'classic_taxon_definitions.tsv': classic,
        'baseline_effect_replay.tsv': pd.DataFrame(checks)}
    for name, table in outputs.items():
        table.to_csv(out / name, sep='\t', index=False, lineterminator='\n', float_format='%.17g')
    if {str(p): sha(p) for p in inputs} != before:
        raise ValueError('An input changed during execution')
    receipt = dict(status='PASS', post_outcome_explanation=True, no_model_fit=True, no_new_p_values=True,
        training_samples=len(training), training_studies=9, features=800, external_cohorts=len(inventory),
        training_effect_rows=len(train), external_effect_rows=len(external), candidate_count_rows=len(counts),
        baseline_replay=checks, environment=dict(python=sys.version, platform=platform.platform(), numpy=np.__version__, pandas=pd.__version__),
        inputs=before, outputs={name:sha(out / name) for name in outputs})
    save_json(out / 'verification.json', receipt)
    print(json.dumps({k:v for k,v in receipt.items() if k not in ['inputs','outputs','environment']}, indent=2))


if __name__ == '__main__':
    main()

