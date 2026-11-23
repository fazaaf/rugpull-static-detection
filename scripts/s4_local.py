import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import matthews_corrcoef, average_precision_score, roc_auc_score
from scipy.stats import spearmanr
d=D.reset_index(drop=True)
ls_=pd.read_csv(ROOT+'/data/label_sources.csv')
d=d.merge(ls_[['address','sun_type']].rename(columns={'sun_type':'Type'}),on='address',how='left')
def cv(dd, reps=5, model='xgb'):
    dd=dd.reset_index(drop=True); X=dd[FEAT]; y=dd.label.values; g=groups(dd); r=[]
    for rep in range(reps):
        for tr,te in StratifiedGroupKFold(5,shuffle=True,random_state=rep).split(X,y,g):
            m=fit(xgbm(rep),X.iloc[tr],y[tr]); p=m.predict_proba(X.iloc[te])[:,1]; pr=(p>=.5).astype(int)
            r.append(dict(mcc=matthews_corrcoef(y[te],pr),spec=1-pr[y[te]==0].mean(),rec=pr[y[te]==1].mean(),
                          pr_auc_legit=average_precision_score(1-y[te],1-p),roc=roc_auc_score(y[te],p)))
    r=pd.DataFrame(r); return {k:f'{r[k].mean():.3f} ± {r[k].std():.3f}' for k in r} | {'n_rug':int((dd.label==1).sum()),'n_legit':int((dd.label==0).sum())}
# ---- 1. TL+Fee sensitivity ----
S=[]
S.append({'set':'Primary (all rug pulls)'}|cv(d))
S.append({'set':'Transaction Limitation + Fee only'}|cv(d[(d.label==0)|d.Type.isin(['Transaction Limitation','Fee'])]))
S.append({'set':'Transaction Limitation only'}|cv(d[(d.label==0)|(d.Type=='Transaction Limitation')]))
S=pd.DataFrame(S); print(S.to_string()); S.to_csv(f'{OUT}/I_sensitivity_type.csv',index=False)

# ---- 2. RF SHAP vs XGB SHAP (rep 0 OOF) ----
X=d[FEAT]; y=d.label.values; g=groups(d)
svx=np.zeros((len(d),29)); svr=np.zeros((len(d),29))
for tr,te in StratifiedGroupKFold(5,shuffle=True,random_state=0).split(X,y,g):
    mx=fit(xgbm(0),X.iloc[tr],y[tr]); svx[te]=shap.TreeExplainer(mx)(X.iloc[te]).values
    mr=RandomForestClassifier(n_estimators=300,class_weight='balanced',random_state=0,n_jobs=-1).fit(X.iloc[tr],y[tr])
    v=shap.TreeExplainer(mr).shap_values(X.iloc[te], check_additivity=False)
    v=v[1] if isinstance(v,list) else (v[:,:,1] if v.ndim==3 else v); svr[te]=v
J=pd.DataFrame({'xgb_mean_abs_shap':np.abs(svx).mean(0),'rf_mean_abs_shap':np.abs(svr).mean(0),
                'xgb_dir':[np.corrcoef(X[f],svx[:,i])[0,1] for i,f in enumerate(FEAT)],
                'rf_dir':[np.corrcoef(X[f],svr[:,i])[0,1] for i,f in enumerate(FEAT)]},index=FEAT)
J['xgb_rank']=J.xgb_mean_abs_shap.rank(ascending=False).astype(int); J['rf_rank']=J.rf_mean_abs_shap.rank(ascending=False).astype(int)
J=J.sort_values('xgb_rank').round(3)
rho=spearmanr(J.xgb_mean_abs_shap,J.rf_mean_abs_shap)[0]
t10=len(set(J.nsmallest(10,'xgb_rank').index)&set(J.nsmallest(10,'rf_rank').index))
sign=(np.sign(J.xgb_dir)==np.sign(J.rf_dir)).mean()
print('\nRF vs XGB: spearman',round(rho,3),'| top10 overlap',t10,'| same direction',round(sign,3)); print(J.head(12).to_string())
J.to_csv(f'{OUT}/J_shap_rf_vs_xgb.csv')
pd.Series({'spearman_29':round(rho,3),'top10_overlap':t10,'direction_agreement':round(sign,3)}).to_csv(f'{OUT}/J_summary.csv')

# ---- 3. SHAP within compiler strata (model trained within stratum) ----
K={}
for v in [5,6,8]:
    dd=d[d.solc_minor==v].reset_index(drop=True); Xs=dd[FEAT]; ys=dd.label.values; gs=groups(dd); sv=np.zeros((len(dd),29))
    for tr,te in StratifiedGroupKFold(5,shuffle=True,random_state=0).split(Xs,ys,gs):
        sv[te]=shap.TreeExplainer(fit(xgbm(0),Xs.iloc[tr],ys[tr]))(Xs.iloc[te]).values
    K[f'0.{v}.x (rug {ys.sum()}, legit {(ys==0).sum()})']=pd.Series(np.abs(sv).mean(0),index=FEAT)
K=pd.DataFrame(K); Kr=K.rank(ascending=False).astype(int)
glob=pd.Series(np.abs(svx).mean(0),index=FEAT).rank(ascending=False).astype(int)
Kout=pd.concat([glob.rename('rank_all'),Kr.add_prefix('rank_')],axis=1).sort_values('rank_all')
print('\n',Kout.head(12).to_string())
for c in K: print(c,'spearman vs all:',round(spearmanr(K[c],np.abs(svx).mean(0))[0],3),'| top5:',list(K[c].nlargest(5).index))
Kout.to_csv(f'{OUT}/K_shap_by_compiler.csv')

# ---- 4. Perturbation test (feature-space, rug pulls in test folds only) ----
legmed=d[d.label==0][FEAT].median()
def scen(Xt, name):
    Z=Xt.copy()
    if name=='S1 optimization findings removed': Z['optimization']=0; Z['total_vulnerabilities']=Z.total_vulnerabilities-Xt.optimization
    if name=='S2 padding (+10 functions, +50% length)': Z['func_count']+=10; Z['code_length']=(Z.code_length*1.5).round()
    if name=='S3 structure set to legitimate median':
        for f in ['func_count','modifier_count','event_count','require_count','code_length']: Z[f]=legmed[f]
    if name=='S4 = S1 + S3':
        Z=scen(scen(Xt,'S1 optimization findings removed'),'S3 structure set to legitimate median')
    return Z
names=['S0 unchanged','S1 optimization findings removed','S2 padding (+10 functions, +50% length)','S3 structure set to legitimate median','S4 = S1 + S3']
P={n:[] for n in names}; FP={n:[] for n in names}
for rep in range(5):
    for tr,te in StratifiedGroupKFold(5,shuffle=True,random_state=rep).split(X,y,g):
        m=fit(xgbm(rep),X.iloc[tr],y[tr]); te_r=te[y[te]==1]
        for n in names: P[n].append((m.predict_proba(scen(X.iloc[te_r],n))[:,1]>=.5).mean())
L=pd.DataFrame({'scenario':names,'rug_recall':[f'{np.mean(P[n]):.3f} ± {np.std(P[n],ddof=1):.3f}' for n in names]})
print('\n',L.to_string()); L.to_csv(f'{OUT}/L_perturbation.csv',index=False)
