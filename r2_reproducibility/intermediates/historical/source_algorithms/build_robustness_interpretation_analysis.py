# Mechanically selected historical AST; main is not an entrypoint.
from __future__ import annotations
import json
import math
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

def bh_fdr(pvalues: pd.Series) -> pd.Series:
    out = pd.Series(np.nan, index=pvalues.index, dtype=float)
    mask = pvalues.notna()
    if mask.any():
        out.loc[mask] = multipletests(pvalues.loc[mask].astype(float), method='fdr_bh')[1]
    return out

def genus_from_taxon(taxon: str) -> str:
    for part in str(taxon).replace(';', '|').split('|'):
        if part.startswith('g__'):
            return part.replace('g__', '')
    return ''

def species_from_taxon(taxon: str) -> str:
    for part in str(taxon).replace(';', '|').split('|'):
        if part.startswith('s__'):
            return part.replace('s__', '').replace('_', ' ')
    return str(taxon)

def fit_feature_ols(y: np.ndarray, design: pd.DataFrame) -> tuple[float, float, float]:
    if np.nanstd(y) == 0:
        return (np.nan, np.nan, np.nan)
    try:
        fit = sm.OLS(y, design, missing='drop').fit(cov_type='HC3')
        return (float(fit.params['condition_crc']), float(fit.bse['condition_crc']), float(fit.pvalues['condition_crc']))
    except Exception:
        return (np.nan, np.nan, np.nan)

def make_design(metadata: pd.DataFrame, sample_index: pd.Index) -> pd.DataFrame:
    meta = metadata.set_index('sample_uid').loc[sample_index]
    condition = (meta['study_condition'] == 'CRC').astype(int)
    study_dummies = pd.get_dummies(meta['study_name'], prefix='study', drop_first=True, dtype=float)
    design = pd.concat([condition.rename('condition_crc'), study_dummies], axis=1)
    return sm.add_constant(design, has_constant='add').astype(float)

def clr_transform(abundance: pd.DataFrame, pseudocount: float) -> pd.DataFrame:
    log_values = np.log(abundance + pseudocount)
    return log_values.sub(log_values.mean(axis=1), axis=0)

def association_screen(clr: pd.DataFrame, abundance: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    design = make_design(metadata, clr.index)
    meta = metadata.set_index('sample_uid').loc[clr.index]
    condition = meta['study_condition'] == 'CRC'
    rows = []
    for taxon in clr.columns:
        coef, se, pvalue = fit_feature_ols(clr[taxon].to_numpy(float), design)
        crc_vals = abundance.loc[condition.index[condition], taxon]
        ctl_vals = abundance.loc[condition.index[~condition], taxon]
        rows.append({'taxon_long': taxon, 'genus': genus_from_taxon(taxon), 'species_short': species_from_taxon(taxon), 'coef_crc_clr_adjusted': coef, 'se_hc3': se, 'pvalue': pvalue, 'prevalence_crc': float((crc_vals > 0).mean()), 'prevalence_control': float((ctl_vals > 0).mean())})
    assoc = pd.DataFrame(rows)
    assoc['qvalue_bh'] = bh_fdr(assoc['pvalue'])
    assoc['direction_sign'] = np.sign(pd.to_numeric(assoc['coef_crc_clr_adjusted'], errors='coerce'))
    return assoc.sort_values(['qvalue_bh', 'pvalue', 'taxon_long'], na_position='last')

def run_leave_one_cohort_association(abundance: pd.DataFrame, metadata: pd.DataFrame, baseline_assoc: pd.DataFrame, stable_taxa: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    cohorts = sorted(metadata['study_name'].astype(str).unique())
    taxa = baseline_assoc['taxon_long'].tolist()
    base_sign = baseline_assoc.set_index('taxon_long')['coef_crc_clr_adjusted'].astype(float).apply(np.sign)
    for left_out in cohorts:
        keep_ids = metadata.loc[metadata['study_name'].astype(str) != left_out, 'sample_uid'].tolist()
        subset_abundance = abundance.loc[keep_ids, taxa]
        min_positive = subset_abundance.where(subset_abundance > 0).min().min()
        assoc = association_screen(clr_transform(subset_abundance, float(min_positive / 2)), subset_abundance, metadata[metadata['sample_uid'].isin(keep_ids)])
        assoc = assoc.set_index('taxon_long')
        for taxon in stable_taxa:
            row = assoc.loc[taxon]
            rows.append({'taxon_long': taxon, 'genus': genus_from_taxon(taxon), 'species_short': species_from_taxon(taxon), 'left_out_cohort': left_out, 'coef_crc_clr_adjusted': float(row['coef_crc_clr_adjusted']), 'pvalue': float(row['pvalue']), 'qvalue_bh': float(row['qvalue_bh']), 'same_direction_as_baseline': bool(np.sign(float(row['coef_crc_clr_adjusted'])) == base_sign.loc[taxon]), 'fdr_lt_0_10': bool(float(row['qvalue_bh']) < 0.1)})
    loco = pd.DataFrame(rows)
    summary_rows = []
    for taxon, group in loco.groupby('taxon_long', sort=False):
        summary_rows.append({'taxon_long': taxon, 'genus': genus_from_taxon(taxon), 'species_short': species_from_taxon(taxon), 'leave_one_cohort_models': int(len(group)), 'same_direction_models': int(group['same_direction_as_baseline'].sum()), 'same_direction_fraction': float(group['same_direction_as_baseline'].mean()), 'fdr_lt_0_10_models': int(group['fdr_lt_0_10'].sum()), 'fdr_lt_0_10_fraction': float(group['fdr_lt_0_10'].mean()), 'min_qvalue': float(pd.to_numeric(group['qvalue_bh'], errors='coerce').min()), 'max_qvalue': float(pd.to_numeric(group['qvalue_bh'], errors='coerce').max())})
    summary = pd.DataFrame(summary_rows)
    summary['leave_one_cohort_robust'] = (summary['same_direction_fraction'] >= 0.9) & (summary['fdr_lt_0_10_fraction'] >= 0.8)
    return (loco, summary.sort_values(['leave_one_cohort_robust', 'same_direction_fraction', 'fdr_lt_0_10_fraction'], ascending=[False, False, False]))

def build_candidate_interpretation(stability: pd.DataFrame, covariate: pd.DataFrame, transport: pd.DataFrame, meta: pd.DataFrame, oral_panel: pd.DataFrame, microhd: pd.DataFrame, loco_summary: pd.DataFrame) -> pd.DataFrame:
    stable = stability[(pd.to_numeric(stability['adjusted_qvalue'], errors='coerce') < 0.1) & (pd.to_numeric(stability['stability_fraction'], errors='coerce') >= 0.6)].copy()
    oral_taxa = set(oral_panel['taxon_long'].astype(str))
    microhd_status = microhd.set_index('taxon_long')['benchmark_match_status'].to_dict()
    merged = stable.merge(covariate[['taxon_long', 'qvalue_bh_coef_crc_clr_study_age_bmi_gender', 'direction_concordant']], on='taxon_long', how='left').merge(transport[['taxon_long', 'transportability_rank', 'selected_fraction', 'mean_abs_elasticnet_coef']], on='taxon_long', how='left').merge(meta[['taxon_long', 'random_effect', 'random_ci_low', 'random_ci_high', 'qvalue_bh_random_effect', 'i2_percent', 'meta_stable_candidate']], on='taxon_long', how='left').merge(loco_summary[['taxon_long', 'same_direction_fraction', 'fdr_lt_0_10_fraction', 'leave_one_cohort_robust']], on='taxon_long', how='left')
    merged['direction'] = np.where(pd.to_numeric(merged['adjusted_coef'], errors='coerce') > 0, 'CRC_higher', 'control_higher')
    merged['oral_associated_taxonomy_panel'] = merged['taxon_long'].isin(oral_taxa)
    merged['microbiomehd_genus_match'] = merged['taxon_long'].map(microhd_status).fillna('NOT_IN_TOP100_CONTEXT')
    merged['covariate_supported'] = pd.to_numeric(merged['qvalue_bh_coef_crc_clr_study_age_bmi_gender'], errors='coerce') < 0.1
    merged['meta_supported'] = pd.to_numeric(merged['qvalue_bh_random_effect'], errors='coerce') < 0.1
    merged['elasticnet_supported'] = pd.to_numeric(merged['selected_fraction'], errors='coerce') >= 0.5
    merged['high_heterogeneity'] = pd.to_numeric(merged['i2_percent'], errors='coerce') >= 75
    layer_cols = ['covariate_supported', 'meta_supported', 'elasticnet_supported', 'leave_one_cohort_robust', 'oral_associated_taxonomy_panel']
    merged['support_layer_count'] = merged[layer_cols].fillna(False).astype(bool).sum(axis=1)
    merged['interpretation_group'] = np.select([merged['leave_one_cohort_robust'].fillna(False) & merged['meta_supported'].fillna(False) & ~merged['high_heterogeneity'].fillna(False), merged['high_heterogeneity'].fillna(False), merged['oral_associated_taxonomy_panel'].fillna(False) & (merged['direction'] == 'CRC_higher')], ['high_consistency_candidate', 'heterogeneous_candidate', 'oral_associated_context_candidate'], default='bounded_candidate')
    columns = ['taxon_long', 'genus', 'species_short', 'direction', 'adjusted_coef', 'adjusted_qvalue', 'stability_fraction', 'cohorts_same_direction', 'qvalue_bh_coef_crc_clr_study_age_bmi_gender', 'random_effect', 'qvalue_bh_random_effect', 'i2_percent', 'same_direction_fraction', 'fdr_lt_0_10_fraction', 'leave_one_cohort_robust', 'selected_fraction', 'transportability_rank', 'oral_associated_taxonomy_panel', 'microbiomehd_genus_match', 'support_layer_count', 'interpretation_group']
    return merged[columns].sort_values(['support_layer_count', 'stability_fraction', 'adjusted_qvalue'], ascending=[False, False, True])
