import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from sklearn.metrics import matthews_corrcoef, confusion_matrix
d = D.reset_index(drop=True); X = d[FEAT]; y = d.label.values; g = groups(d)
print('n', len(d), 'rug', y.sum(), 'groups', len(set(g)))
rows, oof = [], []
for rep in range(5):
    for k,(tr,te) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=rep).split(X,y,g)):
        m = fit(xgbm(rep), X.iloc[tr], y[tr]); p = m.predict_proba(X.iloc[te])[:,1]
        # in-fold threshold: inner group CV on training part only
        Xi, yi, gi = X.iloc[tr].reset_index(drop=True), y[tr], g[tr]; pin = np.zeros(len(tr))
        for itr, ite in StratifiedGroupKFold(3, shuffle=True, random_state=100+rep).split(Xi, yi, gi):
            pin[ite] = fit(xgbm(rep), Xi.iloc[itr], yi[itr]).predict_proba(Xi.iloc[ite])[:,1]
        grid = np.round(np.arange(0.05, 0.96, 0.01), 2)
        thr = grid[np.argmax([matthews_corrcoef(yi, (pin>=t).astype(int)) for t in grid])]
        for t,name in [(0.5,'t050'),(thr,'tin')]:
            pr = (p>=t).astype(int); tn,fp,fn,tp = confusion_matrix(y[te],pr,labels=[0,1]).ravel()
            rows.append(dict(rep=rep,fold=k,rule=name,thr=t,mcc=matthews_corrcoef(y[te],pr),tn=tn,fp=fp,fn=fn,tp=tp))
        oof += [dict(rep=rep, fold=k, i=i, y=yy, p=pp, thr_in=thr) for i,yy,pp in zip(te,y[te],p)]
r = pd.DataFrame(rows); o = pd.DataFrame(oof)
r.to_csv(OUT+'/cv_folds.csv', index=False); o.to_csv(OUT+'/oof_per_rep.csv', index=False)
print(r.groupby('rule')[['mcc','thr']].agg(['mean','std']).round(3))
print((r.groupby('rule')[['tn','fp','fn','tp']].sum()/5).round(1))
# parity with published OOF
pub = pd.read_csv(ROOT+'/results/oof_xgb_final.csv'); pub['address']=pub.address.str.lower()
q = o.groupby('i').p.mean(); cmp = pd.DataFrame({'address': d.address.values[q.index], 'p_mine': q.values}).merge(pub[['address','p_rug']], on='address')
print('parity n', len(cmp), 'max abs diff', (cmp.p_mine-cmp.p_rug).abs().max())
pm = (q>=0.5).astype(int).values; yy = y[q.index]
print('mean-prob confusion (TN FP FN TP):', confusion_matrix(yy, pm).ravel())
