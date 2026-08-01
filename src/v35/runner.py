from __future__ import annotations
import hashlib,json
from dataclasses import asdict
import numpy as np,pandas as pd,torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,average_precision_score,balanced_accuracy_score,brier_score_loss,confusion_matrix,fbeta_score,f1_score,matthews_corrcoef,precision_score,recall_score,roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from core import CFG,GLOBAL,LEVEL1,V32,VARIANTS,VERSION,availability,blend,choose_threshold,clip,engineer,load_cohort,logit,metric_row,mount_root,native,seed_all
from models import gated,level1_predictions,residuals

def calibrate(a,y,b):
    m=LogisticRegression(C=2.,max_iter=1000).fit(logit(a).reshape(-1,1),y)
    return clip(m.predict_proba(logit(a).reshape(-1,1))[:,1]),clip(m.predict_proba(logit(b).reshape(-1,1))[:,1]),{'coef':float(m.coef_[0,0]),'intercept':float(m.intercept_[0])}
def bootstrap(y,a,b,n,seed,la,lb):
    r=np.random.default_rng(seed); pos=np.where(y==1)[0]; neg=np.where(y==0)[0]; vals={k:[] for k in ['ROC_AUC','PR_AUC','Brier']}; point={'ROC_AUC':roc_auc_score(y,a)-roc_auc_score(y,b),'PR_AUC':average_precision_score(y,a)-average_precision_score(y,b),'Brier':brier_score_loss(y,a)-brier_score_loss(y,b)}
    for _ in range(n):
        q=np.r_[r.choice(pos,len(pos),True),r.choice(neg,len(neg),True)]; r.shuffle(q); vals['ROC_AUC'].append(roc_auc_score(y[q],a[q])-roc_auc_score(y[q],b[q])); vals['PR_AUC'].append(average_precision_score(y[q],a[q])-average_precision_score(y[q],b[q])); vals['Brier'].append(brier_score_loss(y[q],a[q])-brier_score_loss(y[q],b[q]))
    return [{'Model_A':la,'Model_B':lb,'Metric':k,'Delta_A_minus_B':float(point[k]),'CI95_Lower':float(np.quantile(v,.025)),'CI95_Upper':float(np.quantile(v,.975))} for k,v in vals.items()]

def run_fold(X,y,tr,te,rep,fold,cfg,out,device):
    seed=cfg.seed+rep*10000+fold*100; A=X.iloc[tr].reset_index(drop=True); B=X.iloc[te].reset_index(drop=True); ya,yb=y[tr],y[te]
    O,T,status=level1_predictions(A,ya,B,cfg,seed); av,at=availability(A),availability(B); train={}; test={}; art={'level1_status':status}
    w,s=blend(O[GLOBAL].to_numpy(),ya,cfg.blend_candidates,seed+1); train['A_Global'],test['A_Global'],cal=calibrate(O[GLOBAL].to_numpy()@w,ya,T[GLOBAL].to_numpy()@w); art['A_Global']={'weights':dict(zip(GLOBAL,w)),'score':s,'calibration':cal}
    w,s=blend(O[LEVEL1].to_numpy(),ya,cfg.blend_candidates,seed+2); train['B_ExpertBlend'],test['B_ExpertBlend'],cal=calibrate(O[LEVEL1].to_numpy()@w,ya,T[LEVEL1].to_numpy()@w); art['B_ExpertBlend']={'weights':dict(zip(LEVEL1,w)),'score':s,'calibration':cal}
    train['C_Gated'],test['C_Gated']=gated(O,ya,T,av,at,cfg,seed+3)
    train['D_MLPResidual'],test['D_MLPResidual'],train['E_MATResidual'],test['E_MATResidual']=residuals(O,av,ya,train['C_Gated'],T,at,test['C_Gated'],cfg,seed+4,device)
    rows=pd.DataFrame({'Original_Index':te,'Repeat':rep,'Outer_Fold':fold,'y_true':yb}); fm={}
    for v in VARIANTS:
        z=choose_threshold(ya,train[v],cfg); t=z['threshold']; rows[f'{v}_Probability']=test[v]; rows[f'{v}_Threshold']=t; rows[f'{v}_Prediction']=(test[v]>=t).astype(int); fm[v]=metric_row(yb,test[v],t); fm[v]['Training_Sensitivity']=z['sensitivity']; fm[v]['Training_Specificity']=z['specificity']
    rows.to_csv(out/f'fold_predictions_repeat{rep}_fold{fold}.csv',index=False)
    with open(out/f'fold_log_repeat{rep}_fold{fold}.json','w') as f: json.dump(native({'repeat':rep,'fold':fold,'artifacts':art,'metrics':fm}),f,indent=2)
    for v in VARIANTS:
        m=fm[v]; print(f" {v:17s} ROC={m['ROC_AUC']:.4f} PR={m['PR_AUC']:.4f} Sens={m['Recall_Sensitivity_TPR']:.3f} Spec={m['Specificity_TNR']:.3f} Brier={m['Brier']:.4f}")
    return rows

def summarize(P,cfg,out):
    rr=[]
    for rep in sorted(P.Repeat.unique()):
        S=P[P.Repeat==rep].sort_values('Original_Index'); y=S.y_true.to_numpy(int)
        for v in VARIANTS:
            p=S[f'{v}_Probability'].to_numpy(float); q=S[f'{v}_Prediction'].to_numpy(int); ts=S[f'{v}_Threshold'].to_numpy(float); tn,fp,fn,tp=confusion_matrix(y,q,labels=[0,1]).ravel(); r=metric_row(y,p,float(np.mean(ts))); r.update({'Repeat':int(rep),'Variant':v,'Accuracy':accuracy_score(y,q),'Balanced_Accuracy':balanced_accuracy_score(y,q),'Precision_PPV':precision_score(y,q,zero_division=0),'Recall_Sensitivity_TPR':recall_score(y,q,zero_division=0),'Specificity_TNR':tn/max(tn+fp,1),'NPV':tn/max(tn+fn,1),'F1':f1_score(y,q,zero_division=0),'F2':fbeta_score(y,q,beta=2,zero_division=0),'MCC':matthews_corrcoef(y,q) if len(np.unique(q))>1 else 0.,'TP':int(tp),'TN':int(tn),'FP':int(fp),'FN':int(fn),'Threshold':float(np.mean(ts))}); rr.append(r)
    R=pd.DataFrame(rr); R.to_csv(out/'repeat_metrics.csv',index=False)
    metrics=['Accuracy','Balanced_Accuracy','Precision_PPV','Recall_Sensitivity_TPR','Specificity_TNR','NPV','F1','F2','MCC','ROC_AUC','PR_AUC','Brier','Log_Loss','TP','TN','FP','FN','Threshold']; ss=[]
    for v in VARIANTS:
        z=R[R.Variant==v]; row={'Variant':v}
        for m in metrics: row[m+'_Mean']=float(z[m].mean()); row[m+'_SD']=float(z[m].std(ddof=1)) if len(z)>1 else 0.
        ss.append(row)
    S=pd.DataFrame(ss); S.to_csv(out/'summary_mean_std.csv',index=False)
    agg={'y_true':('y_true','first')}
    for v in VARIANTS: agg[f'{v}_Probability']=(f'{v}_Probability','mean'); agg[f'{v}_Threshold']=(f'{v}_Threshold','mean')
    C=P.groupby('Original_Index',as_index=False).agg(**agg).sort_values('Original_Index'); C.to_csv(out/'per_patient_consensus.csv',index=False); y=C.y_true.to_numpy(int); cm=[]
    for v in VARIANTS:
        r=metric_row(y,C[f'{v}_Probability'].to_numpy(float),float(C[f'{v}_Threshold'].mean())); r['Variant']=v; cm.append(r)
    M=pd.DataFrame(cm); M.to_csv(out/'consensus_metrics.csv',index=False)
    boots=[]
    for v in VARIANTS[1:]: boots+=bootstrap(y,C[f'{v}_Probability'].to_numpy(float),C['A_Global_Probability'].to_numpy(float),cfg.bootstrap_iterations,cfg.seed+len(boots),v,'A_Global')
    boots+=bootstrap(y,C['E_MATResidual_Probability'].to_numpy(float),C['C_Gated_Probability'].to_numpy(float),cfg.bootstrap_iterations,cfg.seed+991,'E_MATResidual','C_Gated'); boots+=bootstrap(y,C['E_MATResidual_Probability'].to_numpy(float),C['D_MLPResidual_Probability'].to_numpy(float),cfg.bootstrap_iterations,cfg.seed+993,'E_MATResidual','D_MLPResidual')
    B=pd.DataFrame(boots); B.to_csv(out/'paired_bootstrap.csv',index=False)
    comp=[]
    for _,r in M.iterrows():
        for m,b in V32.items(): comp.append({'Variant':r.Variant,'Metric':m,'V3_5':float(r[m]),'V3_2_Benchmark':b,'Delta_V35_minus_V32':float(r[m]-b)})
    pd.DataFrame(comp).to_csv(out/'comparison_vs_v32_descriptive.csv',index=False)
    E=M.set_index('Variant').loc['E_MATResidual']; G=M.set_index('Variant').loc['C_Gated']; q=B[(B.Model_A=='E_MATResidual')&(B.Model_B=='C_Gated')].set_index('Metric'); retain=bool((q.loc['PR_AUC','CI95_Lower']>0 or q.loc['ROC_AUC','CI95_Lower']>0) and E.Brier-G.Brier<=.002)
    audit={'retain_mat_as_performance_driver':retain,'decision':'Retain MAT residual' if retain else 'Do not claim MAT as the performance driver','consensus_deltas_mat_minus_gate':{'ROC_AUC':float(E.ROC_AUC-G.ROC_AUC),'PR_AUC':float(E.PR_AUC-G.PR_AUC),'Brier':float(E.Brier-G.Brier),'Balanced_Accuracy':float(E.Balanced_Accuracy-G.Balanced_Accuracy)},'paired_bootstrap':q.reset_index().to_dict('records')}
    with open(out/'mat_contribution_audit.json','w') as f: json.dump(native(audit),f,indent=2)
    Z=S.copy(); Z['Selection_Score']=.38*Z.PR_AUC_Mean+.30*Z.ROC_AUC_Mean+.20*Z.Balanced_Accuracy_Mean-.12*Z.Brier_Mean; ELI=Z[Z.Recall_Sensitivity_TPR_Mean>=.82]; ELI=ELI if len(ELI) else Z; best=ELI.sort_values('Selection_Score',ascending=False).iloc[0]
    dec={'exploratory_selected_variant':str(best.Variant),'selection_score':float(best.Selection_Score),'target_roc_auc_0_93_reached':bool(best.ROC_AUC_Mean>=.93),'development_repeats':cfg.repeats,'publication_requirement':'Lock the selected design and rerun with repeats=5 before primary paper claims.'}
    with open(out/'final_decision.json','w') as f: json.dump(native(dec),f,indent=2)
    print('\n=== V3.5 REPEATED NESTED-CV SUMMARY ==='); print(S[['Variant','ROC_AUC_Mean','PR_AUC_Mean','Accuracy_Mean','Balanced_Accuracy_Mean','Recall_Sensitivity_TPR_Mean','Specificity_TNR_Mean','F1_Mean','MCC_Mean','Brier_Mean']].to_string(index=False)); print('\n=== CONSENSUS METRICS ==='); print(M[['Variant','ROC_AUC','PR_AUC','Accuracy','Balanced_Accuracy','Recall_Sensitivity_TPR','Specificity_TNR','F1','MCC','Brier','TP','TN','FP','FN']].to_string(index=False)); print('\n=== MAT CONTRIBUTION AUDIT ==='); print(json.dumps(native(audit),indent=2)); print('\n=== EXPLORATORY FINAL DECISION ==='); print(json.dumps(native(dec),indent=2))

def preflight(cfg):
    y=np.r_[np.zeros(24,int),np.ones(12,int)]; P=np.random.default_rng(7).uniform(.05,.95,(36,4)); w,_=blend(P,y,40,7); assert np.isclose(w.sum(),1); choose_threshold(y,clip(P@w),cfg); print('PRE-FLIGHT PASSED: metrics, blend, threshold, dependencies.')
def main(cfg=CFG):
    seed_all(cfg.seed); root=mount_root(cfg); device='cuda' if torch.cuda.is_available() else 'cpu'; print(f'MAT-APPENDIX V3.5 CONTROLLED IMPROVEMENT | {VERSION}'); print('Device:',device); print('Development protocol: global baseline -> experts -> gate -> MLP residual -> MAT residual'); preflight(cfg)
    X0,ys,audit=load_cohort(cfg); X=engineer(X0); y=ys.to_numpy(int); fp=hashlib.sha256(json.dumps(native(asdict(cfg)),sort_keys=True).encode()).hexdigest()[:12]; out=root/f'v35_{fp}'; out.mkdir(parents=True,exist_ok=True); audit.to_csv(out/'leakage_audit.csv',index=False)
    with open(out/'config.json','w') as f: json.dump(native({'version':VERSION,**asdict(cfg)}),f,indent=2)
    print(f'Validated cohort: N={len(X)}, positives={int(y.sum())}, negatives={int((1-y).sum())}'); print('Predictor columns after controlled engineering:',X.shape[1]); print('Output:',out)
    cv=RepeatedStratifiedKFold(n_splits=cfg.outer_folds,n_repeats=cfg.repeats,random_state=cfg.seed); allrows=[]
    for k,(tr,te) in enumerate(cv.split(X,y)):
        rep=k//cfg.outer_folds+1; fold=k%cfg.outer_folds+1; cp=out/f'fold_predictions_repeat{rep}_fold{fold}.csv'; print(f'\n===== REPEAT {rep}/{cfg.repeats}, OUTER FOLD {fold}/{cfg.outer_folds} =====')
        if cp.exists():
            z=pd.read_csv(cp)
            if len(z)==len(te) and all(f'{v}_Probability' in z for v in VARIANTS): print('Loading completed fold checkpoint.'); allrows.append(z); continue
        z=run_fold(X,y,tr,te,rep,fold,cfg,out,device); allrows.append(z); pd.concat(allrows,ignore_index=True).to_csv(out/'all_outer_predictions_partial.csv',index=False)
    P=pd.concat(allrows,ignore_index=True); P.to_csv(out/'all_repeated_nested_predictions.csv',index=False); summarize(P,cfg,out); print('\nV3.5 complete. Results saved to:',out)
if __name__=='__main__': main()
