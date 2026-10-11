# Mechanically selected historical AST; main is not an entrypoint.
from __future__ import annotations
import json
import math
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests

def parse_taxon(taxon: str) -> dict[str, str]:
    parts = {}
    for token in str(taxon).split('|'):
        if '__' in token:
            rank, value = token.split('__', 1)
            parts[rank] = value
    return {'taxon': taxon, 'genus': parts.get('g', ''), 'family': parts.get('f', ''), 'species': parts.get('s', taxon).replace('_', ' ')}

def guild_membership(clr: pd.DataFrame) -> pd.DataFrame:
    assoc = read_tsv(RESULTS / 'C' / 'cohort_adjusted_crc_control_association.tsv')
    assoc_map = assoc.set_index('taxon_long')['direction_adjusted'].to_dict()
    stable = set(read_tsv(RESULTS / 'M_robustness_interpretation' / 'stable_candidate_interpretation.tsv')['taxon_long'])
    rows = []
    rules = {'butyrate_SCFA_commensal_panel': {'genera': {'Roseburia', 'Faecalibacterium', 'Eubacterium', 'Anaerostipes', 'Fusicatenibacter', 'Agathobaculum', 'Gemmiger', 'Intestinimonas'}, 'rationale': 'literature-prior taxonomy-defined panel of commonly discussed butyrate/SCFA-associated commensal genera', 'caution': 'taxonomy-name panel only; no direct functional activity or butyrate-production measurement'}, 'oral_pathobiont_associated_panel': {'genera': {'Parvimonas', 'Peptostreptococcus', 'Fusobacterium', 'Gemella', 'Streptococcus', 'Solobacterium'}, 'rationale': 'literature-prior taxonomy-defined oral/pathobiont-associated CRC context panel', 'caution': 'taxonomy-name panel only; no oral-source or transmission inference'}}
    for taxon in clr.columns:
        parsed = parse_taxon(taxon)
        for guild, rule in rules.items():
            if parsed['genus'] in rule['genera']:
                rows.append({'guild': guild, 'taxon': taxon, 'genus': parsed['genus'], 'species': parsed['species'], 'present_in_retained_matrix': True, 'stable_candidate_yes_no': 'yes' if taxon in stable else 'no', 'direction_in_primary_association': assoc_map.get(taxon, ''), 'rationale': rule['rationale'], 'cautionary_note': rule['caution']})
        if parsed['species'] in {'Bacteroides fragilis', 'Escherichia coli'} or parsed['genus'] == 'Bilophila' or parsed['family'] == 'Enterobacteriaceae':
            rows.append({'guild': 'inflammation_Bacteroides_Enterobacteriaceae_panel', 'taxon': taxon, 'genus': parsed['genus'], 'species': parsed['species'], 'present_in_retained_matrix': True, 'stable_candidate_yes_no': 'yes' if taxon in stable else 'no', 'direction_in_primary_association': assoc_map.get(taxon, ''), 'rationale': 'conservative taxonomy-defined Bacteroides fragilis, Escherichia coli, Bilophila, and Enterobacteriaceae context panel', 'cautionary_note': 'exploratory taxonomic context only; no inflammation mechanism or functional inference'})
    out = pd.DataFrame(rows).drop_duplicates(['guild', 'taxon']).sort_values(['guild', 'genus', 'species'])
    counts = out.groupby('guild')['taxon'].transform('count')
    out['exploratory_due_to_small_panel'] = counts < 3
    return out
