"""Shared settings for the round-2 analyses run outside Colab (scripts/s1-s4)."""
import os
import pandas as pd, numpy as np, xgboost as xgb
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repository root
OUT = os.path.join(ROOT, 'results', 'rev2'); os.makedirs(OUT, exist_ok=True)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.utils.class_weight import compute_sample_weight
D = pd.read_csv(os.path.join(ROOT, 'data', 'D_single_era_matched.csv'))
D['address'] = D.address.str.lower()
FEAT = list(D.columns[D.columns.get_loc('total_vulnerabilities'):D.columns.get_loc('code_length')+1])
assert len(FEAT) == 29, FEAT
def groups(d):
    key = [c for c in FEAT if c != 'code_length']
    return (d[key].astype(int).astype(str).agg('|'.join, axis=1) + '|' + (d['code_length']//1000).astype(int).astype(str)).factorize()[0]
def xgbm(seed):
    return xgb.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.1, subsample=0.8,
                             colsample_bytree=0.8, eval_metric='logloss', random_state=seed, n_jobs=4)
def fit(m, X, y):
    m.fit(X, y, sample_weight=compute_sample_weight('balanced', y)); return m
