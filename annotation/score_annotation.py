"""Score the regular-expression indicators against the manual annotation (writes annotation_validation_results.xlsx)."""
import pandas as pd, openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter
import os
HERE=os.path.dirname(os.path.abspath(__file__))
_ann=pd.read_excel(os.path.join(HERE,'regex_annotation_sheet.xlsx'),sheet_name='annotate')
_key=pd.read_csv(os.path.join(HERE,'annotation_key.csv'))
A=_ann.merge(_key,on='id'); A=A[A.scored.astype(bool)].reset_index(drop=True)   # C01 was a training example
_F=pd.read_csv(os.path.join(HERE,'regex_V4_flags.csv')); _F['address']=_F.address.str.lower()
A['address']=A.address.str.lower(); A=A.merge(_F,on='address',how='left'); assert A.V4_has_mint.notna().all()
MAP={'has_max_tx':'max_tx_or_wallet_limit','has_blacklist':'blacklist','has_pause':'pause','has_fee_modifier':'fee_changeable_by_owner',
 'has_whitelist':'whitelist_or_fee_exclusion','has_transfer_lock':'transfer_lock_or_cooldown','has_mint':'mint_after_deployment',
 'has_owner_only':'owner_only_functions','has_renounce':'renounce_ownership','has_burn':'burn','has_selfdestruct':'selfdestruct','has_proxy':'proxy_or_delegatecall'}
PAT={'has_mint':r'\bmint\b|\b_mint\b|\bminttoken\b','has_blacklist':r'\bblacklist\b|\bblocklist\b|\bblocked\b','has_pause':r'\bpause\b|\bpaused\b|\bunpause\b',
 'has_selfdestruct':r'\bselfdestruct\b|\bsuicide\b','has_owner_only':r'\bonlyowner\b|\brequire.*owner\b','has_fee_modifier':r'\btaxfee\b|\bliquidityfee\b|\b_taxfee\b|\bsetfee\b',
 'has_max_tx':r'\bmaxtxamount\b|\b_maxtx\b|\bmaxwallet\b','has_whitelist':r'\bwhitelist\b|\bexcluded\b|\b_isexcluded\b','has_transfer_lock':r'\blocktime\b|\btimelock\b|\blocked\b',
 'has_proxy':r'\bdelegatecall\b|\bproxy\b|\bupgradeable\b','has_renounce':r'\brenounceownership\b','has_burn':r'\bburn\b|\b_burn\b|\bburntoken\b'}
WHY={'has_max_tx':'Keyword list misses most names in use: maxWalletSize/_maxWalletSize, maxTxnAmount, and the FLOKI _sellAmount limit (9 of the 15 misses left after V4). Python \\b also treats "_" as a letter, so _maxTxAmount failed in V0; V4 fixes this and recovers C03, C39, C51, C54.',
 'has_blacklist':'Keyword list misses the names actually used: bots[], _blackAddress, DENIED_ROLE, newun.',
 'has_pause':'Most switches are setTrading(bool)/tradingOpen, not pause/unpause.',
 'has_fee_modifier':'V4 recovers C02 and C14 (underscore-prefixed names). Still misses _taxFeeOnSell, updateSellFee, setFees, editSellFee, assembly-written fee (removeFee).',
 'has_whitelist':'_isExcludedFromFee does not match \\b_isexcluded\\b because the keyword runs on into a longer name (not fixed by V4); _whiteAddress/canSale missed. FP: the word "whitelist" in a misleading comment (C29).',
 'has_transfer_lock':'Keywords (locktime/timelock/locked) miss tradingOpen=false, enableTrading, transferDelay, cooldown.',
 'has_mint':'FP: internal _mint called only in the constructor.',
 'has_owner_only':'FP: "require(owner != address(0))" in standard _approve matches \\brequire.*owner\\b.',
 'has_renounce':'Misses renounceRole (C15), renounceContract (C28) and changeMinter(0) (C57).',
 'has_burn':'FP: internal _burn never called, and the string "burn amount exceeds balance".',
 'has_selfdestruct':'No contract in the sample contains selfdestruct.',
 'has_proxy':'FP: delegatecall only in an unused Address library, or the word "proxy"/"Upgradeable" in names and comments.'}
wb=openpyxl.Workbook()
H=Font(name='Arial',bold=True,color='FFFFFF'); HF=PatternFill('solid',start_color='1F3864'); N=Font(name='Arial'); B=Font(name='Arial',bold=True)
thin=Side(style='thin',color='BFBFBF'); BD=Border(left=thin,right=thin,top=thin,bottom=thin)
def hdr(ws,cols,widths):
    ws.append(cols)
    for c in ws[ws.max_row]: c.font=H; c.fill=HF; c.alignment=Alignment(wrap_text=True,vertical='top')
    for i,w in enumerate(widths,1): ws.column_dimensions[get_column_letter(i)].width=w
    ws.freeze_panes='A2'
# README
ws=wb.active; ws.title='README'
lines=[('Validation of regex indicators against manual annotation',),('',),
 ('Sample','60 contracts scored (C02-C61; C01 was a training example and is excluded). 45 rug pull, 15 legitimate.'),
 ('Annotation','Annotator: first author, blind to labels and regex output until annotation was complete; codebook + 10 rule decisions (see the annotation sheet, tab rule_decisions (English) or keputusan_aturan (original, Indonesian)). AI assistance was used for code navigation and rule clarification.'),
 ('Regex','The 12 indicators from text_feats (rev2lib.py), applied to text variants: V0 whole source, V1 comments stripped, V2 main contract only, V3 project contracts. V4 = V0 and V4m = V2 with one fix only: "_" and "$" count as separators at the edges of each keyword (Python \\b treats "_" as a letter). No keyword added or removed.'),
 ('Metrics','TP/FP/FN/TN are counts. Precision, recall, F1 and accuracy are formulas on those counts. Kappa = Cohen kappa between regex and annotation (computed in Python).'),
 ('Not scored','sell_restriction and mint_in_constructor_only have no regex counterpart.')]
for l in lines: ws.append(list(l))
ws['A1'].font=Font(name='Arial',bold=True,size=13)
for r in range(3,8): ws.cell(r,1).font=B; ws.cell(r,2).font=N; ws.cell(r,2).alignment=Alignment(wrap_text=True,vertical='top')
ws.column_dimensions['A'].width=14; ws.column_dimensions['B'].width=110
# validity
ws=wb.create_sheet('regex_validity')
hdr(ws,['indicator','mechanism (annotation column)','variant','n','present (annotation)','TP','FP','FN','TN','precision','recall','F1','accuracy','kappa'],[18,30,8,6,10,6,6,6,6,10,10,10,10,9])
def kappa(p,t):
    po=(p==t).mean(); pa=p.mean(); pb=t.mean(); pe=pa*pb+(1-pa)*(1-pb)
    return None if pe==1 else round((po-pe)/(1-pe),3)
for rx,m in MAP.items():
    t=(A[m]=='Y').astype(int)
    for v in ['V0','V1','V2','V3','V4','V4m']:
        p=A[f'{v}_{rx}'].astype(int)
        tp=int(((p==1)&(t==1)).sum()); fp=int(((p==1)&(t==0)).sum()); fn=int(((p==0)&(t==1)).sum()); tn=int(((p==0)&(t==0)).sum())
        r=ws.max_row+1
        ws.append([rx,m,v,len(t),int(t.sum()),tp,fp,fn,tn,
                   f'=IF(F{r}+G{r}=0,"",F{r}/(F{r}+G{r}))',f'=IF(F{r}+H{r}=0,"",F{r}/(F{r}+H{r}))',
                   f'=IF(J{r}="","",IF(K{r}="","",IF(J{r}+K{r}=0,"",2*J{r}*K{r}/(J{r}+K{r}))))',f'=(F{r}+I{r})/D{r}',kappa(p.values,t.values)])
        for c in ws[r]: c.font=N; c.border=BD
        for col in 'JKLM': ws[f'{col}{r}'].number_format='0.000'
        ws[f'N{r}'].number_format='0.000'
# failure modes
ws=wb.create_sheet('failure_modes_V0')
hdr(ws,['indicator','regex pattern (lower-cased source)','missed by regex (FN, annotation=Y)','false alarms (FP, annotation=N)','main reason'],[18,46,46,40,80])
for rx,m in MAP.items():
    t=A[m]=='Y'; p=A['V0_'+rx]==1
    ws.append([rx,PAT[rx],', '.join(A.id[t&~p]),', '.join(A.id[~t&p]),WHY[rx]])
    for c in ws[ws.max_row]: c.font=N; c.alignment=Alignment(wrap_text=True,vertical='top'); c.border=BD
# prevalence
ws=wb.create_sheet('prevalence_by_group')
mech=['sell_restriction','max_tx_or_wallet_limit','blacklist','pause','fee_changeable_by_owner','whitelist_or_fee_exclusion','transfer_lock_or_cooldown','mint_after_deployment','mint_in_constructor_only','owner_only_functions','renounce_ownership','burn','selfdestruct','proxy_or_delegatecall']
rug=A.group!='Legitimate'
hdr(ws,['mechanism','rug pull: Y','rug pull: n','rug pull: share','legitimate: Y','legitimate: n','legitimate: share'],[30,12,12,14,14,14,16])
for m in mech:
    r=ws.max_row+1
    ws.append([m,int((A[m][rug]=='Y').sum()),int(rug.sum()),f'=B{r}/C{r}',int((A[m][~rug]=='Y').sum()),int((~rug).sum()),f'=E{r}/F{r}'])
    for c in ws[r]: c.font=N; c.border=BD
    ws[f'D{r}'].number_format='0%'; ws[f'G{r}'].number_format='0%'
ws=wb.create_sheet('sell_by_type')
hdr(ws,['group (label source)','sell_restriction = Y','sell_restriction = N','n','share Y'],[34,20,20,8,10])
for g,sub in A.groupby('group'):
    r=ws.max_row+1; ws.append([g,int((sub.sell_restriction=='Y').sum()),int((sub.sell_restriction=='N').sum()),f'=B{r}+C{r}',f'=B{r}/D{r}'])
    for c in ws[r]: c.font=N; c.border=BD
    ws[f'E{r}'].number_format='0%'
# summary
ws=wb.create_sheet('annotation_summary')
hdr(ws,['item','value','contracts'],[46,10,80])
items=[('Contracts sharing an identical template (code)',None,'tradingOpen=false: C04 C18 C38 C40 C41 C45 C46 C54; FLOKI: C06 C26 C36 C37 C44 C48 C50 C59 C60; tradingOpen=true: C10 C16 C25 C52 C58 C61; generator: C11 C22 C32; Tenome: C07 C34 C56; removeFee: C21 C30; meme-2023: C39 C51'),
 ('Mechanisms outside the 14 columns',4,'C23 (balance seizure), C31 (hard-coded transfer destruction), C42 (burn of pair balance), C43 (balances set by external contract)'),
 ('Hidden mechanisms that evade common keywords',7,'C21, C24, C27, C30, C35, C42, C43'),
 ('Rule decisions recorded during annotation',10,'see tab rule_decisions (English) or keputusan_aturan (original) in the annotation sheet')]
for it in items:
    ws.append(list(it))
    for c in ws[ws.max_row]: c.font=N; c.alignment=Alignment(wrap_text=True,vertical='top'); c.border=BD
ws['B2']=33
wb.save(os.path.join(HERE,'annotation_validation_results.xlsx')); print('ok')
