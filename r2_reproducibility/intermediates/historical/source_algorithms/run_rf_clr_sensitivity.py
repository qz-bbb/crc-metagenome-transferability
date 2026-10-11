# Mechanically selected historical AST; main is not an entrypoint.
from __future__ import annotations
import argparse
import hashlib
import json
import os
import platform
import sys
import time
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

def bh(values: pd.Series) -> np.ndarray:
    values = pd.to_numeric(values, errors='coerce').to_numpy(dtype=float)
    out = np.full(values.shape, np.nan)
    valid = np.isfinite(values)
    if valid.any():
        order = np.argsort(values[valid])
        ranked = values[valid][order]
        adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
        adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
        tmp = np.empty_like(adjusted)
        tmp[order] = np.clip(adjusted, 0, 1)
        out[valid] = tmp
    return out

def association_screen(clr: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    y = (meta['study_condition'].astype(str) == 'CRC').astype(float)
    design = pd.concat([pd.DataFrame({'CRC_status': y}, index=meta.index), pd.get_dummies(meta['study_name'].astype(str), prefix='study', drop_first=True, dtype=float)], axis=1)
    design = sm.add_constant(design, has_constant='add').astype(float)
    rows = []
    for taxon in clr.columns:
        fit = sm.OLS(clr[taxon].astype(float), design).fit(cov_type='HC3')
        rows.append({'taxon': taxon, 'coef_crc': float(fit.params['CRC_status']), 'se_hc3': float(fit.bse['CRC_status']), 'p_value': float(fit.pvalues['CRC_status'])})
    out = pd.DataFrame(rows)
    out['q_value_bh'] = bh(out['p_value'])
    out['direction'] = np.where(out['coef_crc'] >= 0, 'CRC_higher', 'control_higher')
    support = direction_support(clr, meta, out.set_index('taxon')['coef_crc'])
    return out.merge(support, on='taxon', how='left').sort_values(['q_value_bh', 'p_value', 'taxon'])

def direction_support(clr: pd.DataFrame, meta: pd.DataFrame, coef: pd.Series) -> pd.DataFrame:
    rows = []
    for taxon in clr.columns:
        global_sign = np.sign(float(coef.loc[taxon]))
        eligible = 0
        same = 0
        diffs: list[float] = []
        for study in sorted(meta['study_name'].astype(str).unique()):
            idx = meta['study_name'].astype(str) == study
            sub = meta.loc[idx]
            if sub['study_condition'].astype(str).nunique() < 2:
                continue
            crc = clr.loc[sub.index[sub['study_condition'].astype(str) == 'CRC'], taxon]
            ctrl = clr.loc[sub.index[sub['study_condition'].astype(str) != 'CRC'], taxon]
            diff = float(crc.mean() - ctrl.mean())
            diffs.append(diff)
            dsign = np.sign(diff)
            if dsign == 0 or global_sign == 0:
                continue
            eligible += 1
            same += int(dsign == global_sign)
        rows.append({'taxon': taxon, 'eligible_studies': eligible, 'same_direction_studies': same, 'dcp': same / eligible if eligible else np.nan, 'mean_study_clr_crc_minus_control': float(np.mean(diffs)) if diffs else np.nan})
    return pd.DataFrame(rows)

def select_panel(assoc: pd.DataFrame, threshold: float, cap: int | None=None) -> list[str]:
    keep = assoc[(assoc['q_value_bh'] < 0.1) & (assoc['dcp'] >= threshold)]
    selected = keep.sort_values(['q_value_bh', 'p_value', 'taxon'])['taxon'].tolist()
    return selected[:cap] if cap is not None else selected
