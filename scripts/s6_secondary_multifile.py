"""Secondary analysis: XGBoost with feature-space grouping on the data that include multi-file contracts
(data/D_main_era_matched.csv; 1,901 rug pulls, 818 legitimate). Writes results/rev2/M_secondary_multifile.csv."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from sklearn.metrics import matthews_corrcoef, roc_auc_score
d = pd.read_csv(os.path.join(ROOT, 'data', 'D_main_era_matched.csv')).reset_index(drop=True)
X, y, g = d[FEAT], d.label.values, groups(d); r = []
for rep in range(5):
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=rep).split(X, y, g):
        p = fit(xgbm(rep), X.iloc[tr], y[tr]).predict_proba(X.iloc[te])[:, 1]; pr = (p >= .5).astype(int)
        r.append(dict(mcc=matthews_corrcoef(y[te], pr), spec=1 - pr[y[te] == 0].mean(), rec=pr[y[te] == 1].mean(), roc_auc=roc_auc_score(y[te], p)))
r = pd.DataFrame(r)
out = pd.DataFrame([{'n_rug': int(y.sum()), 'n_legit': int((y == 0).sum()), 'feature_clusters': len(set(g))} |
                    {k: f'{r[k].mean():.3f} ± {r[k].std():.3f}' for k in r}])
print(out.to_string(index=False)); out.to_csv(f'{OUT}/M_secondary_multifile.csv', index=False)
