import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from sklearn.metrics import matthews_corrcoef, brier_score_loss, roc_auc_score
from sklearn.calibration import calibration_curve
from scipy.stats import mannwhitneyu
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
d = D.reset_index(drop=True); y = d.label.values
o = pd.read_csv(f'{OUT}/oof_per_rep.csv'); o['address'] = d.address.values[o.i]
o['pred'] = (o.p>=0.5).astype(int)

# ---------- A. label source / type sensitivity (per-rep mean ± sd) ----------
ls_ = pd.read_csv(ROOT+'/data/label_sources.csv')
lab = d[['address','year']].merge(ls_.rename(columns={'sun_type':'Type','source_group':'source'})[['address','Type','source','crpwarner_flag']], on='address', how='left')
lab['crp_flag'] = lab.crpwarner_flag
lab['in_crp'] = lab.crp_flag.notna()
lab['Type'] = lab.Type.fillna('Not in Sun et al. (CRPWarner)')
R = o[o.y==1].merge(lab, on='address')
def sub(col):
    t = R.groupby([col,'rep']).pred.mean().groupby(level=0).agg(['mean','std'])
    n = R[R.rep==0].groupby(col).size().rename('n')
    pr = R.groupby(col).p.mean().rename('mean_p')
    return pd.concat([n,t,pr],axis=1).sort_values('n',ascending=False).round(3)
A1 = sub('source'); A2 = sub('Type'); A3 = sub('in_crp'); A4 = R[R.in_crp].pipe(lambda x: x).groupby(['crp_flag','rep']).pred.mean().groupby(level=0).agg(['mean','std']).round(3)
A4['n'] = R[(R.rep==0)&R.in_crp].groupby('crp_flag').size()
print('\n== A1 source\n',A1,'\n== A2 type\n',A2,'\n== A3 in CRPWarner\n',A3,'\n== A4 CRP flag\n',A4)
yr = pd.crosstab(lab[d.label==1].source, lab[d.label==1].year); print('\n== source x year\n', yr)

# ---------- B. compiler breakdown ----------
o2 = o.merge(d[['address','solc_minor','is_proxy','year']], on='address')
rowsB=[]
for v,gq in o2.groupby('solc_minor'):
    per=[]
    for rep,gr in gq.groupby('rep'):
        yy,pp=gr.y.values,gr.pred.values
        rec = pp[yy==1].mean() if (yy==1).any() else np.nan; spec = 1-pp[yy==0].mean() if (yy==0).any() else np.nan
        mcc = matthews_corrcoef(yy,pp) if len(set(yy))==2 else np.nan
        per.append((rec,spec,mcc))
    per=np.array(per,dtype=float)
    rowsB.append(dict(solc=f'0.{int(v)}.x', n_rug=int((gq[gq.rep==0].y==1).sum()), n_legit=int((gq[gq.rep==0].y==0).sum()),
        recall=np.nanmean(per[:,0]), specificity=np.nanmean(per[:,1]), mcc=np.nanmean(per[:,2])))
B = pd.DataFrame(rowsB).round(3); print('\n== B compiler\n',B)

# ---------- C. calibration ----------
Cr=[]
for rep,gr in o.groupby('rep'):
    Cr.append(dict(rep=rep, brier=brier_score_loss(gr.y,gr.p), roc_auc=roc_auc_score(gr.y,gr.p)))
Cr=pd.DataFrame(Cr); prev=y.mean()
brier_ref = prev*(1-prev)
fig,ax=plt.subplots(1,2,figsize=(9,4))
for rep,gr in o.groupby('rep'):
    ft,mp = calibration_curve(gr.y,gr.p,n_bins=10,strategy='quantile'); ax[0].plot(mp,ft,'o-',ms=3,lw=1,alpha=.7,color='black' if rep==0 else 'grey',label='Repetition 1' if rep==0 else None)
ax[0].plot([0,1],[0,1],'k--',lw=.8); ax[0].set_xlabel('Mean predicted probability (rug pull)'); ax[0].set_ylabel('Observed fraction of rug pulls'); ax[0].set_title('(a) Reliability diagram, 10 quantile bins')
ax[1].hist(o[o.y==0].p,bins=20,range=(0,1),color='white',edgecolor='black',label='Legitimate',density=True)
ax[1].hist(o[o.y==1].p,bins=20,range=(0,1),color='grey',alpha=.6,edgecolor='black',label='Rug pull',density=True)
ax[1].axvline(.5,ls='--',c='k',lw=.8); ax[1].set_xlabel('Predicted probability'); ax[1].set_ylabel('Density'); ax[1].legend(frameon=False); ax[1].set_title('(b) Out-of-fold score distribution')
ax[0].legend(frameon=False); plt.tight_layout(); plt.savefig(f'{OUT}/fig_calibration.png',dpi=300); plt.close()
# ECE
def ece(yv,pv,b=10):
    q=pd.qcut(pv,b,duplicates='drop'); g=pd.DataFrame({'y':yv,'p':pv,'q':q}).groupby('q',observed=True); return (g.size()/len(yv)*(g.y.mean()-g.p.mean()).abs()).sum()
Cr['ece']=[ece(gr.y.values,gr.p.values) for _,gr in o.groupby('rep')]
cv = pd.read_csv(f'{OUT}/cv_folds.csv')
thr = cv[cv.rule=='tin'].thr
C = dict(brier=f"{Cr.brier.mean():.3f} ± {Cr.brier.std():.3f}", brier_noskill=round(brier_ref,3), ece=f"{Cr.ece.mean():.3f} ± {Cr.ece.std():.3f}",
         mcc_t050=f"{cv[cv.rule=='t050'].mcc.mean():.3f} ± {cv[cv.rule=='t050'].mcc.std():.3f}",
         mcc_infold=f"{cv[cv.rule=='tin'].mcc.mean():.3f} ± {cv[cv.rule=='tin'].mcc.std():.3f}",
         thr_infold_median=float(thr.median()), thr_infold_range=f"{thr.min():.2f}–{thr.max():.2f}")
print('\n== C calibration\n',C)

# ---------- D. class-wise descriptives ----------
rowsD=[]
for f in FEAT:
    a=d.loc[d.label==1,f]; b=d.loc[d.label==0,f]
    binary = set(d[f].dropna().unique())<= {0,1}
    U,p = mannwhitneyu(a,b,alternative='two-sided'); delta = 2*U/(len(a)*len(b))-1   # Cliff's delta (rug vs legit)
    fmt=lambda s: f"{s.mean()*100:.1f}%" if binary else f"{s.median():g} [{s.quantile(.25):g}–{s.quantile(.75):g}]"
    rowsD.append(dict(feature=f, type='binary' if binary else 'count', rug_pull=fmt(a), legitimate=fmt(b), cliffs_delta=round(delta,3), p_value=p))
Dd=pd.DataFrame(rowsD)
pv=Dd.p_value.values; o_=np.argsort(pv); m=len(pv); adj=np.empty(m); run=0
for r,i in enumerate(o_): run=max(run,min(1,(m-r)*pv[i])); adj[i]=run
Dd['p_holm']=adj
Dd['p_holm']=Dd.p_holm.map(lambda v:'<0.001' if v<0.001 else f'{v:.3f}'); Dd=Dd.drop(columns='p_value')
print('\n== D descriptives\n',Dd.to_string())

# ---------- E. legit ≥730 days sensitivity (rerun CV) ----------
from sklearn.model_selection import StratifiedGroupKFold
leg = pd.read_csv(ROOT+'/data/legit_candidates_processed_886.csv'); leg['address']=leg.address.str.lower()
pub = pd.read_csv(ROOT+'/results/oof_xgb_final.csv'); pub['address']=pub.address.str.lower()
life = pub.set_index('address').life_days
d2 = d.copy(); d2['life']=d2.address.map(life)
print('\nlegit life_days coverage', d2[d2.label==0].life.notna().mean())
E=[]
for cut in [365,730,1095]:
    dd = d2[(d2.label==1)|(d2.life>=cut)].reset_index(drop=True); X=dd[FEAT]; yy=dd.label.values; gg=groups(dd); m_=[]
    for rep in range(5):
        for tr,te in StratifiedGroupKFold(5,shuffle=True,random_state=rep).split(X,yy,gg):
            p=fit(xgbm(rep),X.iloc[tr],yy[tr]).predict_proba(X.iloc[te])[:,1]; pr=(p>=.5).astype(int)
            m_.append((matthews_corrcoef(yy[te],pr), 1-pr[yy[te]==0].mean(), pr[yy[te]==1].mean()))
    m_=np.array(m_); E.append(dict(min_activity_days=cut, n_legit=int((yy==0).sum()), mcc=f"{m_[:,0].mean():.3f} ± {m_[:,0].std(ddof=1):.3f}", specificity=round(m_[:,1].mean(),3), recall=round(m_[:,2].mean(),3)))
E=pd.DataFrame(E); print('\n== E legit activity\n',E)

with pd.ExcelWriter(f'{OUT}/rev2_analyses.xlsx') as w:
    A1.to_excel(w,'A1_source'); A2.to_excel(w,'A2_type'); A4.to_excel(w,'A3_CRPWarner_flag'); yr.to_excel(w,'A4_source_x_year')
    B.to_excel(w,'B_compiler',index=False); pd.DataFrame([C]).T.to_excel(w,'C_calibration'); Dd.to_excel(w,'D_descriptives',index=False); E.to_excel(w,'E_legit_activity',index=False)
