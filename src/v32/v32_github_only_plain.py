"""MAT-Appendix V3.2 GitHub-only cross-fitted constrained blend.
Run after V3.1 in the same Colab runtime.
"""
from pathlib import Path
import json, re
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    brier_score_loss, confusion_matrix, f1_score, fbeta_score,
    log_loss, matthews_corrcoef, precision_score, roc_auc_score,
)

SEED = 2029
MIN_SENS = 0.84
MIN_SPEC = 0.78
N_WEIGHTS = 30000
TOP_K = 5
THRESHOLDS = np.linspace(0.05, 0.95, 361)
OUT = Path('/content/drive/MyDrive/MAT-Appendix/v32_results')
if not Path('/content/drive/MyDrive').exists():
    OUT = Path('/content/MAT-Appendix/v32_results')
OUT.mkdir(parents=True, exist_ok=True)


def first_col(cols, patterns):
    for p in patterns:
        r = re.compile(p, re.I)
        for c in cols:
            if r.search(str(c).strip()):
                return c
    return None


def prob_col(s, name):
    if not pd.api.types.is_numeric_dtype(s):
        return False
    x = pd.to_numeric(s, errors='coerce').dropna().to_numpy(float)
    if len(x) == 0 or np.mean((x >= 0) & (x <= 1)) < 0.98:
        return False
    hints = ('prob','pred','oof','logistic','random','forest','extra','xgb',
             'catboost','mat','ensemble','stack')
    return any(h in str(name).lower() for h in hints)


def find_oof(ns):
    candidates = []
    for name, obj in ns.items():
        if not isinstance(obj, pd.DataFrame) or len(obj) < 50 or obj.shape[1] < 4:
            continue
        cols = list(obj.columns)
        y = first_col(cols, [r'^y_true$',r'^target$',r'^label$',r'^outcome$',
                             r'^severity$',r'^complicated$'])
        f = first_col(cols, [r'^outer_fold$',r'^outerfold$',r'^fold$',
                             r'^cv_fold$',r'^test_fold$'])
        if y is None or f is None:
            continue
        ps = [c for c in cols if c not in (y,f) and prob_col(obj[c], c)]
        if len(ps) >= 2:
            candidates.append((len(ps), len(obj), name, obj, y, f, ps))
    if not candidates:
        frames = [f'{k}: {v.shape}, {list(v.columns)[:12]}' for k,v in ns.items()
                  if isinstance(v, pd.DataFrame)]
        raise RuntimeError('No OOF DataFrame found. V3.1 must be completed in the '
                           'same runtime.\n' + '\n'.join(frames[:20]))
    candidates.sort(reverse=True, key=lambda z: (z[0], z[1]))
    _,_,name,df,y,f,ps = candidates[0]
    print(f'Using OOF DataFrame: {name}, shape={df.shape}')
    print('Probability columns:', ps)
    return df.copy(), y, f, ps


def clip(p):
    return np.clip(np.asarray(p, float), 1e-6, 1-1e-6)


def platt(y, p_train, p_test):
    try:
        def logits(p):
            p = clip(p)
            return np.log(p/(1-p)).reshape(-1,1)
        m = LogisticRegression(max_iter=2000, random_state=SEED)
        m.fit(logits(p_train), y)
        return clip(m.predict_proba(logits(p_train))[:,1]), clip(m.predict_proba(logits(p_test))[:,1])
    except Exception:
        return clip(p_train), clip(p_test)


def rank_models(y, X, k):
    rows=[]
    for c in X.columns:
        p=clip(X[c])
        rows.append((average_precision_score(y,p), roc_auc_score(y,p), c))
    rows.sort(reverse=True)
    selected=[c for _,_,c in rows[:k]]
    for token in ('random forest','xgboost','catboost','stack','ensemble','mat'):
        m=[c for c in X.columns if token in str(c).lower()]
        if m and m[0] not in selected:
            selected[-1]=m[0]
    return list(dict.fromkeys(selected))[:k]


def stats(y,p,ths):
    pred=p[:,None]>=ths[None,:]
    yy=y[:,None].astype(bool)
    tp=np.sum(pred & yy,0); fn=np.sum((~pred)&yy,0)
    tn=np.sum((~pred)&(~yy),0); fp=np.sum(pred&(~yy),0)
    sens=tp/np.maximum(tp+fn,1); spec=tn/np.maximum(tn+fp,1)
    bal=(sens+spec)/2
    den=np.sqrt(np.maximum((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn),1))
    mcc=((tp*tn)-(fp*fn))/den
    return sens,spec,bal,mcc


def search(y,M,rng):
    n=M.shape[1]
    weights=np.vstack([np.eye(n),np.full((1,n),1/n),
                       rng.dirichlet(np.full(n,0.7),N_WEIGHTS)])
    best=(-1e9,None,None,None)
    for start in range(0,len(weights),512):
        W=weights[start:start+512]
        P=M@W.T
        for j in range(P.shape[1]):
            p=clip(P[:,j]); pr=average_precision_score(y,p); roc=roc_auc_score(y,p)
            sens,spec,bal,mcc=stats(y,p,THRESHOLDS)
            penalty=4*((np.maximum(MIN_SENS-sens,0)**2)+(np.maximum(MIN_SPEC-spec,0)**2))
            score=.35*pr+.20*roc+.30*bal+.15*mcc-penalty
            score += ((sens>=MIN_SENS)&(spec>=MIN_SPEC))*0.01
            i=int(np.nanargmax(score))
            if score[i]>best[0]:
                best=(float(score[i]),W[j].copy(),float(THRESHOLDS[i]),
                      dict(train_pr=float(pr),train_roc=float(roc),
                           train_sens=float(sens[i]),train_spec=float(spec[i]),
                           train_bal=float(bal[i]),train_mcc=float(mcc[i])))
    return best[1],best[2],best[3]


def metrics(y,p,pred):
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    sens=tp/(tp+fn); spec=tn/(tn+fp); npv=tn/(tn+fn) if tn+fn else np.nan
    return dict(N=len(y),Positive_N=int(y.sum()),Accuracy=accuracy_score(y,pred),
        Balanced_Accuracy=balanced_accuracy_score(y,pred),
        Precision_PPV=precision_score(y,pred,zero_division=0),
        Recall_Sensitivity_TPR=sens,Specificity_TNR=spec,NPV=npv,
        F1=f1_score(y,pred,zero_division=0),F2=fbeta_score(y,pred,beta=2,zero_division=0),
        MCC=matthews_corrcoef(y,pred),ROC_AUC=roc_auc_score(y,p),
        PR_AUC=average_precision_score(y,p),Brier=brier_score_loss(y,p),
        Log_Loss=log_loss(y,p,labels=[0,1]),TP=int(tp),TN=int(tn),FP=int(fp),FN=int(fn))


df,ycol,fcol,pcols=find_oof(globals())
work=df[[ycol,fcol,*pcols]].copy()
work[ycol]=pd.to_numeric(work[ycol],errors='raise').astype(int)
for c in pcols: work[c]=pd.to_numeric(work[c],errors='raise')
if work.isna().any().any(): raise ValueError('Required OOF columns contain missing values.')
folds=sorted(work[fcol].unique())
if len(folds)<3: raise ValueError('At least 3 outer folds are required.')

y=work[ycol].to_numpy(int); p_oof=np.full(len(work),np.nan); t_oof=np.full(len(work),np.nan)
records=[]; rng=np.random.default_rng(SEED)
for held in folds:
    tr=(work[fcol]!=held); te=~tr
    yt=y[tr.to_numpy()]
    selected=rank_models(yt,work.loc[tr,pcols],min(TOP_K,len(pcols)))
    Mtr=np.zeros((tr.sum(),len(selected))); Mte=np.zeros((te.sum(),len(selected)))
    for j,c in enumerate(selected):
        Mtr[:,j],Mte[:,j]=platt(yt,work.loc[tr,c].to_numpy(),work.loc[te,c].to_numpy())
    w,t,s=search(yt,Mtr,rng)
    pt=clip(Mte@w); p_oof[te.to_numpy()]=pt; t_oof[te.to_numpy()]=t
    mt=metrics(y[te.to_numpy()],pt,(pt>=t).astype(int))
    rec=dict(Heldout_Outer_Fold=held,Selected_Models=selected,
             Weights={c:float(x) for c,x in zip(selected,w)},Threshold=t,**s,
             **{f'Heldout_{k}':v for k,v in mt.items()})
    records.append(rec)
    print(f'Fold {held}: sens={mt["Recall_Sensitivity_TPR"]:.3f}, '
          f'spec={mt["Specificity_TNR"]:.3f}, bal={mt["Balanced_Accuracy"]:.3f}, '
          f'PR-AUC={mt["PR_AUC"]:.3f}, threshold={t:.3f}')

pred=(p_oof>=t_oof).astype(int)
agg=metrics(y,p_oof,pred); agg['Model']='V3.2 Cross-Fitted Constrained Blend'
pred_df=work[[ycol,fcol]].copy(); pred_df['v32_probability']=p_oof
pred_df['v32_fold_threshold']=t_oof; pred_df['v32_prediction']=pred
pred_df.to_csv(OUT/'v32_cross_fitted_predictions.csv',index=False)
pd.DataFrame([agg]).to_csv(OUT/'v32_aggregate_metrics.csv',index=False)
with open(OUT/'v32_fold_weights_and_metrics.json','w') as f: json.dump(records,f,indent=2)
with open(OUT/'v32_aggregate_metrics.json','w') as f: json.dump(agg,f,indent=2)

print('\nV3.2 AGGREGATE METRICS')
for k,v in agg.items(): print(f'{k:28s}: {v:.6f}' if isinstance(v,float) else f'{k:28s}: {v}')
print('Saved to:',OUT)
v32_result={'aggregate_metrics':agg,'predictions':pred_df,'fold_records':records,'output_dir':OUT}
