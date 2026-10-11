# Mechanically selected historical AST; main is not an entrypoint.
from __future__ import annotations
import json
import math
import platform
from pathlib import Path
import numpy as np
import pandas as pd
import scipy
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def bh_fdr(pvalues: pd.Series) -> pd.Series:
    out = pd.Series(np.nan, index=pvalues.index, dtype=float)
    mask = pvalues.notna()
    if mask.any():
        out.loc[mask] = multipletests(pvalues.loc[mask].astype(float), method='fdr_bh')[1]
    return out

def genus_from_taxon(taxon: str) -> str:
    parts = str(taxon).replace(';', '|').split('|')
    for part in parts:
        if part.startswith('g__'):
            return part.replace('g__', '')
    return ''

def species_from_taxon(taxon: str) -> str:
    parts = str(taxon).replace(';', '|').split('|')
    for part in parts:
        if part.startswith('s__'):
            return part.replace('s__', '').replace('_', ' ')
    return ''

def fit_feature_ols(y: np.ndarray, x: pd.DataFrame, condition_name: str='condition_crc') -> tuple[float, float, float]:
    if np.nanstd(y) == 0:
        return (np.nan, np.nan, np.nan)
    try:
        model = sm.OLS(y, x, missing='drop').fit(cov_type='HC3')
        return (float(model.params[condition_name]), float(model.bse[condition_name]), float(model.pvalues[condition_name]))
    except Exception:
        return (np.nan, np.nan, np.nan)

def zscore_numeric(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors='coerce')
    sd = numeric.std(skipna=True)
    if sd == 0 or pd.isna(sd):
        return numeric * 0
    return (numeric - numeric.mean(skipna=True)) / sd

def finalize_design(design: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
    variable_cols = [col for col in design.columns if col != 'const' and design[col].nunique(dropna=False) > 1]
    design = sm.add_constant(design[variable_cols], has_constant='add').astype(float)
    matrix = design.to_numpy(dtype=float)
    return (design, int(np.linalg.matrix_rank(matrix)), int(matrix.shape[1]))

def main() -> None:
    design = pd.DataFrame({'condition_crc': condition.values}, index=clr.index)
    study_dummies = pd.get_dummies(study.loc[clr.index], prefix='study', drop_first=True, dtype=float)
    design = pd.concat([design, study_dummies.set_index(clr.index)], axis=1)
    design = sm.add_constant(design, has_constant='add').astype(float)
    design_path = PROCESSED / 'C' / 'cohort_adjusted_design_matrix.tsv.gz'
    assoc_rows = []
    for taxon in clr.columns:
        coef, se, pval = fit_feature_ols(clr[taxon].values.astype(float), design)
        crc_vals = filtered_abundance.loc[condition.index[condition == 1], taxon]
        ctl_vals = filtered_abundance.loc[condition.index[condition == 0], taxon]
        assoc_rows.append({'taxon_long': taxon, 'genus': genus_from_taxon(taxon), 'species_short': species_from_taxon(taxon), 'n_crc': int(crc_vals.shape[0]), 'n_control': int(ctl_vals.shape[0]), 'prevalence_crc': float((crc_vals > 0).mean()), 'prevalence_control': float((ctl_vals > 0).mean()), 'median_abundance_crc_percent': float(crc_vals.median()), 'median_abundance_control_percent': float(ctl_vals.median()), 'median_diff_crc_minus_control_percent': float(crc_vals.median() - ctl_vals.median()), 'coef_crc_clr_adjusted': coef, 'se_hc3': se, 'pvalue': pval})
    assoc = pd.DataFrame(assoc_rows)
    assoc['qvalue_bh'] = bh_fdr(assoc['pvalue'])
    assoc['direction_adjusted'] = np.sign(assoc['coef_crc_clr_adjusted']).map({1.0: 'CRC_higher', -1.0: 'control_higher', 0.0: 'zero'})
    assoc = assoc.sort_values(['qvalue_bh', 'pvalue', 'taxon_long'], na_position='last')
    stability_rows = []
    for taxon in assoc['taxon_long']:
        feature_assoc = assoc.loc[assoc['taxon_long'] == taxon].iloc[0]
        adjusted_sign = np.sign(feature_assoc['coef_crc_clr_adjusted'])
        tested = 0
        same = 0
        cohort_details = []
        for cohort in sorted(study.unique()):
            cohort_samples = study.index[study == cohort]
            cohort_meta = metadata.set_index('sample_uid').loc[cohort_samples]
            if set(cohort_meta['study_condition']) >= {'CRC', 'control'}:
                tested += 1
                crc_ids = cohort_meta.index[cohort_meta['study_condition'] == 'CRC']
                ctl_ids = cohort_meta.index[cohort_meta['study_condition'] == 'control']
                diff = float(filtered_abundance.loc[crc_ids, taxon].median() - filtered_abundance.loc[ctl_ids, taxon].median())
                direction = np.sign(diff)
                if adjusted_sign != 0 and direction == adjusted_sign:
                    same += 1
                cohort_details.append(f'{cohort}:{diff:.6g}')
        stability_rows.append({'taxon_long': taxon, 'genus': genus_from_taxon(taxon), 'species_short': species_from_taxon(taxon), 'adjusted_coef': feature_assoc['coef_crc_clr_adjusted'], 'adjusted_qvalue': feature_assoc['qvalue_bh'], 'cohorts_tested': tested, 'cohorts_same_direction': same, 'stability_fraction': float(same / tested) if tested else np.nan, 'per_cohort_median_diff_percent': ';'.join(cohort_details)})
    stability = pd.DataFrame(stability_rows).sort_values(['adjusted_qvalue', 'stability_fraction'], ascending=[True, False], na_position='last')
    schema_rows = []
    overlap = pd.DataFrame()
    try:
        mh_q = pd.read_csv(MICROBIOMEHD_Q, sep='\t').rename(columns={'Unnamed: 0': 'microbiomehd_taxon'})
        mh_eff = pd.read_csv(MICROBIOMEHD_EFFECTS, sep='\t').rename(columns={'Unnamed: 0': 'microbiomehd_taxon'})
        crc_cols = [c for c in mh_eff.columns if c.startswith('crc_')]
        crc_column_path = Path(write_tsv(pd.DataFrame({'crc_effect_column': crc_cols}), PROCESSED / 'E' / 'microbiomehd_crc_effect_columns.tsv'))
        mh_eff['genus'] = mh_eff['microbiomehd_taxon'].map(genus_from_taxon)
        mh_q['genus'] = mh_q['microbiomehd_taxon'].map(genus_from_taxon)
        top_candidates = assoc.head(min(100, len(assoc))).copy()
        top_candidates['rank'] = np.arange(1, len(top_candidates) + 1)
        top_candidates = top_candidates.merge(stability[['taxon_long', 'stability_fraction']], on='taxon_long', how='left')
        overlap = top_candidates.merge(mh_eff[['genus', 'microbiomehd_taxon'] + crc_cols], on='genus', how='left', suffixes=('', '_microbiomehd'))
        q_crc_cols = [c for c in mh_q.columns if c.startswith('crc_')]
        overlap = overlap.merge(mh_q[['genus'] + q_crc_cols].add_suffix('_q').rename(columns={'genus_q': 'genus'}), on='genus', how='left')
        schema_rows.append({'file': str(MICROBIOMEHD_Q), 'status': 'READABLE', 'rows': str(mh_q.shape[0]), 'columns': str(mh_q.shape[1]), 'crc_columns': ','.join(q_crc_cols)})
        schema_rows.append({'file': str(MICROBIOMEHD_EFFECTS), 'status': 'READABLE', 'rows': str(mh_eff.shape[0]), 'columns': str(mh_eff.shape[1]), 'crc_columns': ','.join(crc_cols)})
    except Exception as exc:
        crc_column_path = Path(write_tsv(pd.DataFrame(columns=['crc_effect_column']), PROCESSED / 'E' / 'microbiomehd_crc_effect_columns.tsv'))
        schema_rows.append({'file': str(MICROBIOMEHD_DIR), 'status': 'FAILED', 'rows': '', 'columns': '', 'crc_columns': '', 'error': str(exc)})
    schema_path = Path(write_tsv(pd.DataFrame(schema_rows), RESULTS / 'E' / 'microbiomehd_schema_audit.tsv'))
    if overlap.empty:
        overlap = pd.DataFrame(columns=['taxon_long', 'genus', 'status'])
    overlap['benchmark_match_status'] = np.where(overlap.get('microbiomehd_taxon', pd.Series(index=overlap.index)).notna(), 'GENUS_MATCH', 'NO_GENUS_MATCH')
    overlap_path = Path(write_tsv(overlap, RESULTS / 'E' / 'microbiomehd_top100_genus_overlap.tsv'))
    metadata_indexed = metadata.set_index('sample_uid').loc[clr.index].copy()
    completeness_specs = [('study_name', 'baseline adjustment block'), ('age', 'primary sensitivity covariate'), ('gender', 'primary sensitivity covariate'), ('BMI', 'primary sensitivity covariate'), ('country', 'context-only sensitivity covariate'), ('sequencing_platform', 'context-only sensitivity covariate'), ('DNA_extraction_kit', 'excluded: incomplete metadata'), ('disease_stage', 'excluded: sparse post-diagnosis field')]
    completeness_rows = []
    for col, model_role in completeness_specs:
        series = metadata_indexed[col] if col in metadata_indexed.columns else pd.Series(index=metadata_indexed.index, dtype=object)
        non_missing = int(series.notna().sum())
        completeness_rows.append({'covariate': col, 'non_missing_n': non_missing, 'total_n': int(len(metadata_indexed)), 'non_missing_fraction': float(non_missing / len(metadata_indexed)) if len(metadata_indexed) else np.nan, 'unique_non_missing': int(series.nunique(dropna=True)), 'model_role': model_role})
    covariate_completeness = pd.DataFrame(completeness_rows)
    covariate_completeness_path = Path(write_tsv(covariate_completeness, RESULTS / 'G' / 'covariate_completeness.tsv'))
    sens_meta = metadata_indexed.copy()
    sens_meta['age_z'] = zscore_numeric(sens_meta['age'])
    sens_meta['BMI_z'] = zscore_numeric(sens_meta['BMI'])
    eligible = sens_meta['age_z'].notna() & sens_meta['BMI_z'].notna() & sens_meta['gender'].notna() & sens_meta['study_condition'].isin(['CRC', 'control'])
    sensitivity_samples = sens_meta.index[eligible].tolist()
    if len(sensitivity_samples) < 100 or condition.loc[sensitivity_samples].nunique() < 2:
        raise SystemExit('covariate sensitivity design has too few complete CRC/control samples')
    base_sens = pd.DataFrame({'condition_crc': condition.loc[sensitivity_samples].astype(float), 'age_z': sens_meta.loc[sensitivity_samples, 'age_z'].astype(float), 'BMI_z': sens_meta.loc[sensitivity_samples, 'BMI_z'].astype(float)}, index=sensitivity_samples)
    gender_dummies = pd.get_dummies(sens_meta.loc[sensitivity_samples, 'gender'].astype(str), prefix='gender', drop_first=True, dtype=float)
    study_sens_dummies = pd.get_dummies(sens_meta.loc[sensitivity_samples, 'study_name'].astype(str), prefix='study', drop_first=True, dtype=float)
    covariate_design, covariate_rank, covariate_columns = finalize_design(pd.concat([base_sens, gender_dummies, study_sens_dummies], axis=1))
    covariate_design_path = PROCESSED / 'G' / 'study_age_bmi_gender_design_matrix.tsv.gz'
    context_sens = base_sens.copy()
    country_dummies = pd.get_dummies(sens_meta.loc[sensitivity_samples, 'country'].astype(str), prefix='country', drop_first=True, dtype=float)
    platform_dummies = pd.get_dummies(sens_meta.loc[sensitivity_samples, 'sequencing_platform'].astype(str), prefix='platform', drop_first=True, dtype=float)
    context_design, context_rank, context_columns = finalize_design(pd.concat([context_sens, gender_dummies, country_dummies, platform_dummies], axis=1))
    context_design_path = PROCESSED / 'G' / 'country_platform_no_study_design_matrix.tsv.gz'

    def run_covariate_screen(design_table: pd.DataFrame, sample_ids: list[str], coef_name: str) -> pd.DataFrame:
        rows = []
        for taxon in clr.columns:
            coef, se, pval = fit_feature_ols(clr.loc[sample_ids, taxon].values.astype(float), design_table)
            rows.append({'taxon_long': taxon, 'genus': genus_from_taxon(taxon), 'species_short': species_from_taxon(taxon), coef_name: coef, f'se_hc3_{coef_name}': se, f'pvalue_{coef_name}': pval})
        out = pd.DataFrame(rows)
        out[f'qvalue_bh_{coef_name}'] = bh_fdr(out[f'pvalue_{coef_name}'])
        out[f'direction_{coef_name}'] = np.sign(out[coef_name]).map({1.0: 'CRC_higher', -1.0: 'control_higher', 0.0: 'zero'})
        return out.sort_values([f'qvalue_bh_{coef_name}', f'pvalue_{coef_name}', 'taxon_long'], na_position='last')
    covariate_assoc = run_covariate_screen(covariate_design, sensitivity_samples, 'coef_crc_clr_study_age_bmi_gender')
    covariate_assoc_path = Path(write_tsv(covariate_assoc, RESULTS / 'G' / 'covariate_adjusted_crc_control_association.tsv'))
    context_assoc = run_covariate_screen(context_design, sensitivity_samples, 'coef_crc_clr_age_bmi_gender_country_platform')
    context_assoc_path = Path(write_tsv(context_assoc, RESULTS / 'G' / 'country_platform_no_study_sensitivity.tsv'))
    baseline_compare = assoc[['taxon_long', 'coef_crc_clr_adjusted', 'pvalue', 'qvalue_bh', 'direction_adjusted']].rename(columns={'coef_crc_clr_adjusted': 'coef_crc_clr_study_only', 'pvalue': 'pvalue_study_only', 'qvalue_bh': 'qvalue_bh_study_only', 'direction_adjusted': 'direction_study_only'})
    covariate_compare = baseline_compare.merge(covariate_assoc, on='taxon_long', how='left')
    covariate_compare['baseline_sign'] = np.sign(covariate_compare['coef_crc_clr_study_only'])
    covariate_compare['covariate_sign'] = np.sign(covariate_compare['coef_crc_clr_study_age_bmi_gender'])
    covariate_compare['direction_concordant'] = covariate_compare['baseline_sign'] == covariate_compare['covariate_sign']
    covariate_compare_path = Path(write_tsv(covariate_compare, RESULTS / 'G' / 'baseline_vs_covariate_sensitivity.tsv'))
    performance_rows = []
    prediction_rows = []
    coefficient_rows = []
    x_model = clr.to_numpy(dtype=float)
    y_model = condition.loc[clr.index].to_numpy(dtype=int)
    for held_out in sorted(metadata_indexed.loc[clr.index, 'study_name'].astype(str).unique()):
        test_mask = metadata_indexed.loc[clr.index, 'study_name'].astype(str).to_numpy() == held_out
        train_mask = ~test_mask
        y_train = y_model[train_mask]
        y_test = y_model[test_mask]
        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            performance_rows.append({'held_out_study': held_out, 'n_train': int(train_mask.sum()), 'n_test': int(test_mask.sum()), 'n_test_crc': int(y_test.sum()), 'n_test_control': int((1 - y_test).sum()), 'auroc': np.nan, 'average_precision': np.nan, 'status': 'SKIPPED_SINGLE_CLASS'})
            continue
        model = make_pipeline(StandardScaler(), LogisticRegression(solver='saga', l1_ratio=0.5, C=0.5, max_iter=2000, class_weight='balanced', random_state=42))
        model.fit(x_model[train_mask], y_train)
        prob = model.predict_proba(x_model[test_mask])[:, 1]
        auroc = float(roc_auc_score(y_test, prob))
        aupr = float(average_precision_score(y_test, prob))
        performance_rows.append({'held_out_study': held_out, 'n_train': int(train_mask.sum()), 'n_test': int(test_mask.sum()), 'n_test_crc': int(y_test.sum()), 'n_test_control': int((1 - y_test).sum()), 'auroc': auroc, 'average_precision': aupr, 'status': 'COMPUTED'})
        test_ids = clr.index[test_mask].tolist()
        for sample_uid, true_label, score in zip(test_ids, y_test, prob):
            prediction_rows.append({'sample_uid': sample_uid, 'held_out_study': held_out, 'true_condition': 'CRC' if int(true_label) == 1 else 'control', 'predicted_crc_probability': float(score)})
        coef = model.named_steps['logisticregression'].coef_.ravel()
        for taxon, value in zip(clr.columns, coef):
            coefficient_rows.append({'held_out_study': held_out, 'taxon_long': taxon, 'elasticnet_standardized_coef': float(value)})
    loso_performance = pd.DataFrame(performance_rows)
    loso_predictions = pd.DataFrame(prediction_rows)
    coefficient_table = pd.DataFrame(coefficient_rows)
    loso_performance_path = Path(write_tsv(loso_performance, RESULTS / 'H' / 'leave_one_study_elasticnet_performance.tsv'))
    loso_predictions_path = Path(write_tsv(loso_predictions, RESULTS / 'H' / 'leave_one_study_elasticnet_predictions.tsv'))
    if coefficient_table.empty:
        feature_weights = pd.DataFrame(columns=['taxon_long', 'mean_elasticnet_coef', 'mean_abs_elasticnet_coef', 'selected_fraction'])
    else:
        feature_weights = coefficient_table.groupby('taxon_long', as_index=False).agg(mean_elasticnet_coef=('elasticnet_standardized_coef', 'mean'), mean_abs_elasticnet_coef=('elasticnet_standardized_coef', lambda x: float(np.mean(np.abs(x)))), selected_fraction=('elasticnet_standardized_coef', lambda x: float(np.mean(np.abs(x) > 1e-08))))
        feature_weights['genus'] = feature_weights['taxon_long'].map(genus_from_taxon)
        feature_weights['species_short'] = feature_weights['taxon_long'].map(species_from_taxon)
        feature_weights = feature_weights.sort_values(['mean_abs_elasticnet_coef', 'selected_fraction'], ascending=[False, False])
    feature_weights_path = Path(write_tsv(feature_weights, RESULTS / 'H' / 'elasticnet_feature_weights.tsv'))
    transport = assoc[['taxon_long', 'genus', 'species_short', 'coef_crc_clr_adjusted', 'qvalue_bh']].merge(stability[['taxon_long', 'stability_fraction', 'cohorts_same_direction', 'cohorts_tested']], on='taxon_long', how='left').merge(covariate_compare[['taxon_long', 'qvalue_bh_coef_crc_clr_study_age_bmi_gender', 'direction_concordant']], on='taxon_long', how='left').merge(feature_weights[['taxon_long', 'mean_abs_elasticnet_coef', 'selected_fraction']], on='taxon_long', how='left')
    transport['neglog10_q'] = -np.log10(pd.to_numeric(transport['qvalue_bh'], errors='coerce').clip(lower=1e-300))
    transport['covariate_supported'] = (pd.to_numeric(transport['qvalue_bh_coef_crc_clr_study_age_bmi_gender'], errors='coerce') < 0.1) & (transport['direction_concordant'].astype(str).str.lower() == 'true')
    transport['transportability_score'] = transport['neglog10_q'].fillna(0) * transport['stability_fraction'].fillna(0) * np.where(transport['covariate_supported'], 1.0, 0.5) * (1.0 + transport['mean_abs_elasticnet_coef'].fillna(0))
    transport = transport.sort_values('transportability_score', ascending=False)
    transport['transportability_rank'] = np.arange(1, len(transport) + 1)
    transport_path = Path(write_tsv(transport, RESULTS / 'H' / 'transportability_score.tsv'))
