# Mechanically selected historical AST; main is not an entrypoint.
from __future__ import annotations
import math
from pathlib import Path
import numpy as np
import pandas as pd

def bh_fdr(pvalues: pd.Series) -> pd.Series:
    p = pd.to_numeric(pvalues, errors='coerce').to_numpy(dtype=float)
    out = np.full(len(p), np.nan, dtype=float)
    valid = np.isfinite(p)
    if not valid.any():
        return pd.Series(out, index=pvalues.index)
    valid_p = p[valid]
    order = np.argsort(valid_p)
    ranked = valid_p[order]
    n = len(ranked)
    q = ranked * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)
    valid_out = np.empty(n, dtype=float)
    valid_out[order] = q
    out[np.where(valid)[0]] = valid_out
    return pd.Series(out, index=pvalues.index)

def normal_pvalue(z: float) -> float:
    if not np.isfinite(z):
        return np.nan
    return float(math.erfc(abs(float(z)) / math.sqrt(2.0)))

def chi2_sf(value: float, df: int) -> float:
    try:
        from scipy.stats import chi2
        return float(chi2.sf(value, df))
    except Exception:
        return np.nan

def species_from_taxon(taxon: str) -> str:
    text = str(taxon)
    if '|s__' in text:
        return text.split('|s__')[-1].replace('_', ' ')
    return text

def genus_from_taxon(taxon: str) -> str:
    text = str(taxon)
    if '|g__' in text:
        return text.split('|g__')[-1].split('|')[0]
    species = species_from_taxon(text)
    return species.split()[0] if species else ''

def per_cohort_effects(clr: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    cohorts = sorted(metadata['study_name'].dropna().astype(str).unique())
    for taxon in clr.columns:
        values = clr[taxon]
        for cohort in cohorts:
            ids = metadata.index[metadata['study_name'].astype(str) == cohort]
            cohort_meta = metadata.loc[ids]
            crc_ids = cohort_meta.index[cohort_meta['study_condition'] == 'CRC']
            control_ids = cohort_meta.index[cohort_meta['study_condition'] == 'control']
            crc_vals = pd.to_numeric(values.loc[crc_ids], errors='coerce').dropna()
            control_vals = pd.to_numeric(values.loc[control_ids], errors='coerce').dropna()
            n_crc = int(len(crc_vals))
            n_control = int(len(control_vals))
            if n_crc < 2 or n_control < 2:
                rows.append({'taxon_long': taxon, 'genus': genus_from_taxon(taxon), 'species_short': species_from_taxon(taxon), 'cohort': cohort, 'n_crc': n_crc, 'n_control': n_control, 'effect_crc_minus_control': np.nan, 'se': np.nan, 'z_value': np.nan, 'pvalue': np.nan, 'status': 'INSUFFICIENT_GROUP_SIZE'})
                continue
            effect = float(crc_vals.mean() - control_vals.mean())
            variance = float(crc_vals.var(ddof=1) / n_crc + control_vals.var(ddof=1) / n_control)
            se = math.sqrt(variance) if variance > 0 else np.nan
            z_value = effect / se if np.isfinite(se) and se > 0 else np.nan
            rows.append({'taxon_long': taxon, 'genus': genus_from_taxon(taxon), 'species_short': species_from_taxon(taxon), 'cohort': cohort, 'n_crc': n_crc, 'n_control': n_control, 'effect_crc_minus_control': effect, 'se': se, 'z_value': z_value, 'pvalue': normal_pvalue(z_value), 'status': 'COMPUTED' if np.isfinite(z_value) else 'ZERO_OR_UNESTIMABLE_VARIANCE'})
    return pd.DataFrame(rows)

def random_effects_meta(effect: pd.Series, se: pd.Series) -> dict[str, float]:
    eff = pd.to_numeric(effect, errors='coerce').to_numpy(dtype=float)
    sev = pd.to_numeric(se, errors='coerce').to_numpy(dtype=float)
    valid = np.isfinite(eff) & np.isfinite(sev) & (sev > 0)
    eff = eff[valid]
    sev = sev[valid]
    k = int(len(eff))
    if k < 3:
        return {'k_cohorts': k, 'fixed_effect': np.nan, 'random_effect': np.nan, 'random_se': np.nan, 'random_ci_low': np.nan, 'random_ci_high': np.nan, 'z_value': np.nan, 'pvalue': np.nan, 'q_statistic': np.nan, 'q_df': np.nan, 'q_pvalue': np.nan, 'tau2_dl': np.nan, 'i2_percent': np.nan}
    weights = 1.0 / np.square(sev)
    fixed = float(np.sum(weights * eff) / np.sum(weights))
    q_stat = float(np.sum(weights * np.square(eff - fixed)))
    df = k - 1
    c_value = float(np.sum(weights) - np.sum(np.square(weights)) / np.sum(weights))
    tau2 = max(0.0, (q_stat - df) / c_value) if c_value > 0 else 0.0
    random_weights = 1.0 / (np.square(sev) + tau2)
    random = float(np.sum(random_weights * eff) / np.sum(random_weights))
    random_se = float(math.sqrt(1.0 / np.sum(random_weights)))
    z_value = random / random_se if random_se > 0 else np.nan
    i2 = max(0.0, (q_stat - df) / q_stat) * 100.0 if q_stat > 0 else 0.0
    return {'k_cohorts': k, 'fixed_effect': fixed, 'random_effect': random, 'random_se': random_se, 'random_ci_low': random - 1.96 * random_se, 'random_ci_high': random + 1.96 * random_se, 'z_value': z_value, 'pvalue': normal_pvalue(z_value), 'q_statistic': q_stat, 'q_df': df, 'q_pvalue': chi2_sf(q_stat, df), 'tau2_dl': tau2, 'i2_percent': i2}

def build_meta_table(per_cohort: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for taxon, group in per_cohort.groupby('taxon_long', sort=False):
        computed = group[group['status'] == 'COMPUTED'].copy()
        result = random_effects_meta(computed['effect_crc_minus_control'], computed['se'])
        effects = pd.to_numeric(computed['effect_crc_minus_control'], errors='coerce')
        random_direction = np.sign(result['random_effect']) if np.isfinite(result['random_effect']) else np.nan
        same_direction = int((np.sign(effects) == random_direction).sum()) if np.isfinite(random_direction) and random_direction != 0 else 0
        positive = int((effects > 0).sum())
        negative = int((effects < 0).sum())
        rows.append({'taxon_long': taxon, 'genus': genus_from_taxon(taxon), 'species_short': species_from_taxon(taxon), 'cohorts_computed': result['k_cohorts'], 'cohorts_positive': positive, 'cohorts_negative': negative, 'cohorts_same_random_direction': same_direction, 'same_random_direction_fraction': same_direction / result['k_cohorts'] if result['k_cohorts'] else np.nan, **result})
    meta = pd.DataFrame(rows)
    meta['qvalue_bh_random_effect'] = bh_fdr(meta['pvalue'])
    meta['random_direction'] = np.sign(meta['random_effect']).map({1.0: 'CRC_higher', -1.0: 'control_higher', 0.0: 'zero'})
    return meta.sort_values(['qvalue_bh_random_effect', 'pvalue', 'taxon_long'], na_position='last')
