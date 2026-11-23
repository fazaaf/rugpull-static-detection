"""Append the CSV outputs of s1, s3 and s4 to results/rev2/rev2_analyses.xlsx (run after s1-s4)."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT, OUT
import pandas as pd
SHEETS = [('F_SHAP_full29', 'shap_full_29.csv', 0), ('G_SHAP_stability', 'shap_stability.csv', 0),
          ('H_cv_folds_threshold', 'cv_folds.csv', None), ('I_sensitivity_TL_Fee', 'I_sensitivity_type.csv', None),
          ('J_SHAP_RF_vs_XGB', 'J_shap_rf_vs_xgb.csv', 0), ('J_summary', 'J_summary.csv', 0),
          ('K_SHAP_by_compiler', 'K_shap_by_compiler.csv', 0), ('L_perturbation', 'L_perturbation.csv', None)]
with pd.ExcelWriter(f'{OUT}/rev2_analyses.xlsx', engine='openpyxl', mode='a', if_sheet_exists='replace') as w:
    for sheet, f, ix in SHEETS:
        pd.read_csv(f'{OUT}/{f}', index_col=ix).to_excel(w, sheet_name=sheet, index=ix is not None)
print('sheets written to', f'{OUT}/rev2_analyses.xlsx')
