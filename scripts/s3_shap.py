import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
import shap, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.stats import spearmanr, kendalltau
d=D.reset_index(drop=True); X=d[FEAT]; y=d.label.values; g=groups(d)
SV={}
for rep in range(5):
    sv=np.zeros((len(d),len(FEAT)))
    for tr,te in StratifiedGroupKFold(5,shuffle=True,random_state=rep).split(X,y,g):
        m=fit(xgbm(rep),X.iloc[tr],y[tr]); sv[te]=shap.TreeExplainer(m)(X.iloc[te]).values
    SV[rep]=sv
imp=pd.DataFrame({f'rep{r}':np.abs(SV[r]).mean(0) for r in SV},index=FEAT)
pub=pd.read_csv(ROOT+'/results/shap_importance_final.csv').set_index('feature')
print('rep0 vs published max diff', (imp.rep0-pub.mean_abs_shap.reindex(FEAT)).abs().max())
imp['mean']=imp.mean(1); imp['sd']=imp[[f'rep{r}' for r in SV]].std(1)
ranks=imp[[f'rep{r}' for r in SV]].rank(ascending=False)
imp['rank_rep1']=ranks.rep0.astype(int); imp['rank_min']=ranks.min(1).astype(int); imp['rank_max']=ranks.max(1).astype(int)
imp['direction_r']=[np.corrcoef(X[f],SV[0][:,i])[0,1] if X[f].nunique()>1 else np.nan for i,f in enumerate(FEAT)]
imp=imp.sort_values('mean',ascending=False)
pairs=[(a,b) for a in range(5) for b in range(a+1,5)]
rho=[spearmanr(imp[f'rep{a}'],imp[f'rep{b}'])[0] for a,b in pairs]
tau10=[]
for a,b in pairs:
    top=set(imp[f'rep{a}'].nlargest(10).index)|set(imp[f'rep{b}'].nlargest(10).index)
    tau10.append(kendalltau(imp.loc[list(top),f'rep{a}'],imp.loc[list(top),f'rep{b}'])[0])
jac=[len(set(imp[f'rep{a}'].nlargest(5).index)&set(imp[f'rep{b}'].nlargest(5).index))/len(set(imp[f'rep{a}'].nlargest(5).index)|set(imp[f'rep{b}'].nlargest(5).index)) for a,b in pairs]
stab=dict(spearman_all29=f'{np.mean(rho):.3f} (min {np.min(rho):.3f})', kendall_top10=f'{np.mean(tau10):.3f} (min {np.min(tau10):.3f})', jaccard_top5=f'{np.mean(jac):.2f} (min {np.min(jac):.2f})')
print(stab)
# local stability: per-contract top-3 agreement across reps
top3=[np.argsort(-np.abs(SV[r]),axis=1)[:,:3] for r in SV]
agree=np.mean([[len(set(top3[a][i])&set(top3[b][i]))/3 for i in range(len(d))] for a,b in pairs])
stab['per_contract_top3_overlap']=round(float(agree),3); print(stab)
out=imp[['mean','sd','rank_rep1','rank_min','rank_max','direction_r']].round(3)
out.to_csv(f'{OUT}/shap_full_29.csv'); print(out.to_string())
pd.Series(stab).to_csv(f'{OUT}/shap_stability.csv')
# dependence plots (rep 1 = random_state 0, as in paper)
fig,axs=plt.subplots(2,2,figsize=(9,7)); sv=SV[0]
for ax,f,lab in zip(axs.ravel(),['optimization','code_length','has_owner_only','has_renounce'],['(a)','(b)','(c)','(d)']):
    i=FEAT.index(f); xv=X[f].values.astype(float)
    jit = np.random.default_rng(0).uniform(-.15,.15,len(xv)) if set(np.unique(xv))<={0,1} else 0
    for cls,mk,col in [(0,'o','white'),(1,'^','grey')]:
        s=y==cls; ax.scatter(xv[s]+(jit[s] if np.ndim(jit) else 0), sv[s,i], s=8, marker=mk, facecolors=col, edgecolors='black', linewidths=.3, alpha=.6, label='Legitimate' if cls==0 else 'Rug pull')
    ax.axhline(0,c='k',lw=.6,ls='--'); ax.set_xlabel(f); ax.set_ylabel('SHAP value (log-odds)'); ax.set_title(f'{lab} {f}',loc='left',fontsize=10)
    if f=='code_length': ax.set_xscale('log')
    if set(np.unique(xv))<={0,1}: ax.set_xticks([0,1])
axs[0,0].legend(frameon=False,fontsize=8); plt.tight_layout(); plt.savefig(f'{OUT}/fig_shap_dependence.png',dpi=300); plt.close()
