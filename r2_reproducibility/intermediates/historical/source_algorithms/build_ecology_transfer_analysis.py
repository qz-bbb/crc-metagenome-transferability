# Mechanically selected historical AST; main is not an entrypoint.
from __future__ import annotations
import json
import math
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests
ORAL_ASSOCIATED_GENERA = {'Actinomyces', 'Atopobium', 'Campylobacter', 'Dialister', 'Fusobacterium', 'Gemella', 'Granulicatella', 'Leptotrichia', 'Parvimonas', 'Peptostreptococcus', 'Porphyromonas', 'Prevotella', 'Selenomonas', 'Solobacterium', 'Streptococcus', 'Treponema', 'Veillonella'}
