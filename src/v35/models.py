from __future__ import annotations
import numpy as np, pandas as pd, torch, torch.nn as nn
from torch.utils.data import DataLoader,TensorDataset
from sklearn.ensemble import ExtraTreesClassifier,HistGradientBoostingClassifier,RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold,StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from core import Config,GLOBAL,LEVEL1,Prep,availability,blend,clip,indices,logit,miss_indices,modality,seed_all

def make_models(seed,posw):
    return {
      'RF':RandomForestClassifier(n_estimators=450,min_samples_leaf=3,max_features='sqrt',class_weight='balanced_subsample',n_jobs=-1,random_state=seed),
      'ET':ExtraTreesClassifier(n_estimators=500,min_samples_leaf=3,max_features=.7,class_weight='balanced',n_jobs=-1,random_state=seed),
      'HGB':HistGradientBoostingClassifier(learning_rate=.035,max_iter=280,max_leaf_nodes=15,min_samples_leaf=12,l2_regularization=1.5,random_state=seed),
      'XGB':XGBClassifier(n_estimators=380,max_depth=3,learning_rate=.025,min_child_weight=5,subsample=.82,colsample_bytree=.75,reg_alpha=.25,reg_lambda=2.5,gamma=.05,objective='binary:logistic',eval_metric='logloss',scale_pos_weight=posw,n_jobs=-1,random_state=seed),
      'CAT':CatBoostClassifier(iterations=420,depth=5,learning_rate=.025,l2_leaf_reg=6.,loss_function='Logloss',eval_metric='AUC',auto_class_weights='Balanced',random_seed=seed,verbose=False,allow_writing_files=False,thread_count=-1),
      'LR':LogisticRegression(C=.35,class_weight='balanced',max_iter=2000,solver='liblinear',random_state=seed)
    }
def fitpred(model,a,y,b,c):
    model.fit(a,y); return clip(model.predict_proba(b)[:,1]),clip(model.predict_proba(c)[:,1])
def topids(X,y,k,seed):
    m=ExtraTreesClassifier(n_estimators=280,min_samples_leaf=3,max_features='sqrt',class_weight='balanced',n_jobs=-1,random_state=seed).fit(X,y)
    return np.sort(np.argsort(m.feature_importances_)[-min(k,X.shape[1]):])

def level1_predictions(Xtr,ytr,Xte,cfg:Config,seed):
    sk=StratifiedKFold(cfg.inner_folds,shuffle=True,random_state=seed); o={n:np.full(len(ytr),np.nan) for n in LEVEL1}; t={n:np.zeros(len(Xte)) for n in LEVEL1}; status=[]
    for f,(i,v) in enumerate(sk.split(Xtr,ytr),1):
        p=Prep().fit(Xtr.iloc[i]); a,b,c=p.transform(Xtr.iloc[i]),p.transform(Xtr.iloc[v]),p.transform(Xte); yy=ytr[i]; pos=(len(yy)-yy.sum())/max(yy.sum(),1); mods=make_models(seed+f*101,pos)
        for n in ['RF','ET','XGB','CAT']:
            try: pv,pt=fitpred(mods[n],a,yy,b,c); s='ok'
            except Exception as e: pv=np.full(len(v),yy.mean()); pt=np.full(len(Xte),yy.mean()); s=f'fallback:{e}'
            o[n][v]=pv; t[n]+=pt/cfg.inner_folds; status.append({'fold':f,'model':n,'status':s})
        for n,k,key in [('XGB45',45,'XGB'),('ET30',30,'ET')]:
            q=topids(a,yy,k,seed+f*107+k)
            try: pv,pt=fitpred(make_models(seed+f*113+k,pos)[key],a[:,q],yy,b[:,q],c[:,q]); s='ok'
            except Exception as e: pv=np.full(len(v),yy.mean()); pt=np.full(len(Xte),yy.mean()); s=f'fallback:{e}'
            o[n][v]=pv; t[n]+=pt/cfg.inner_folds; status.append({'fold':f,'model':n,'status':s})
        specs=[('ClinicalLab',['clinical','laboratory'],'CAT'),('Ultrasound',['clinical','ultrasound'],'CAT'),('Missingness',[],'LR')]
        for n,gs,key in specs:
            q=miss_indices(p.names) if n=='Missingness' else indices(p.names,gs)
            if len(q)<2: pv=np.full(len(v),yy.mean()); pt=np.full(len(Xte),yy.mean()); s='fallback:few_features'
            else:
                try: pv,pt=fitpred(make_models(seed+f*131+len(q),pos)[key],a[:,q],yy,b[:,q],c[:,q]); s=f'ok:{len(q)}'
                except Exception as e: pv=np.full(len(v),yy.mean()); pt=np.full(len(Xte),yy.mean()); s=f'fallback:{e}'
            o[n][v]=pv; t[n]+=pt/cfg.inner_folds; status.append({'fold':f,'model':n,'status':s})
    O,T=pd.DataFrame(o),pd.DataFrame(t)
    if O.isna().any().any(): raise RuntimeError('NaN in level-1 OOF predictions')
    return O,T,status

def meta_matrix(P,A):
    prob=clip(P[LEVEL1].to_numpy(float)); dis=np.c_[prob.std(1),prob.max(1)-prob.min(1)]; us=prob[:,[LEVEL1.index('Ultrasound')]]*A[:,[0]]
    return np.c_[logit(prob),A,dis,us].astype(np.float32)
def gated(O,y,T,A,B,cfg,seed):
    X,Z=meta_matrix(O,A),meta_matrix(T,B); sk=StratifiedKFold(cfg.meta_folds,shuffle=True,random_state=seed); p=np.zeros(len(y)); q=np.zeros(len(T))
    for f,(i,v) in enumerate(sk.split(X,y),1):
        m=Pipeline([('s',StandardScaler()),('lr',LogisticRegression(C=.18,class_weight='balanced',max_iter=2000,random_state=seed+f))]).fit(X[i],y[i]); p[v]=m.predict_proba(X[v])[:,1]; q+=m.predict_proba(Z)[:,1]/cfg.meta_folds
    return clip(p),clip(q)

class MLP(nn.Module):
    def __init__(self,d,drop): super().__init__(); self.n=nn.Sequential(nn.Linear(d,32),nn.LayerNorm(32),nn.GELU(),nn.Dropout(drop),nn.Linear(32,16),nn.GELU(),nn.Dropout(drop),nn.Linear(16,1))
    def forward(self,x,b): return b+1.25*torch.tanh(self.n(x).squeeze(-1))
class MAT(nn.Module):
    def __init__(self,tokens,ctx,d,h,drop):
        super().__init__(); self.v=nn.Linear(2,d); self.e=nn.Embedding(tokens,d); lay=nn.TransformerEncoderLayer(d,h,d*2,drop,batch_first=True,activation='gelu',norm_first=True); self.enc=nn.TransformerEncoder(lay,1); self.c=nn.Sequential(nn.Linear(ctx,d),nn.GELU(),nn.Dropout(drop)); self.head=nn.Sequential(nn.LayerNorm(d*2),nn.Linear(d*2,d),nn.GELU(),nn.Dropout(drop),nn.Linear(d,1))
    def forward(self,x,r,c,b):
        ids=torch.arange(x.shape[1],device=x.device)[None].expand(x.shape[0],-1); z=self.enc(self.v(torch.stack([x,r],-1))+self.e(ids)); w=r[:,:,None].clamp(.05,1); z=(z*w).sum(1)/w.sum(1).clamp_min(1e-6); return b+1.25*torch.tanh(self.head(torch.cat([z,self.c(c)],1)).squeeze(-1))

def train_net(model,tr,y,es,ye,pred,cfg,seed,device):
    seed_all(seed); model=model.to(device); tt=[torch.tensor(x,dtype=torch.float32) for x in tr]; ds=TensorDataset(*tt,torch.tensor(y,dtype=torch.float32)); ld=DataLoader(ds,cfg.batch_size,shuffle=True,generator=torch.Generator().manual_seed(seed)); ev=[torch.tensor(x,dtype=torch.float32,device=device) for x in es]; pv=[torch.tensor(x,dtype=torch.float32,device=device) for x in pred]; opt=torch.optim.AdamW(model.parameters(),lr=cfg.lr,weight_decay=cfg.weight_decay); loss=nn.BCEWithLogitsLoss(); best=None; bl=1e9; pat=0
    for _ in range(cfg.epochs):
        model.train()
        for z in ld:
            *a,b=z; a=[x.to(device) for x in a]; b=b.to(device); opt.zero_grad(); l=loss(model(*a),b); l.backward(); nn.utils.clip_grad_norm_(model.parameters(),2.); opt.step()
        model.eval()
        with torch.no_grad(): vl=float(loss(model(*ev),torch.tensor(ye,dtype=torch.float32,device=device)))
        if vl<bl-1e-4: bl=vl; best={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}; pat=0
        else:
            pat+=1
            if pat>=cfg.patience: break
    if best: model.load_state_dict(best)
    model.eval()
    with torch.no_grad(): return clip(torch.sigmoid(model(*pv)).cpu().numpy())

def residuals(O,A,y,base,T,B,baset,cfg,seed,device):
    P=clip(O[LEVEL1].to_numpy(float)); Q=clip(T[LEVEL1].to_numpy(float)); X=meta_matrix(O,A); Z=meta_matrix(T,B); R=np.ones_like(P,np.float32); S=np.ones_like(Q,np.float32); u=LEVEL1.index('Ultrasound'); R[:,u]=A[:,0]; S[:,u]=B[:,0]
    sk=StratifiedKFold(cfg.meta_folds,shuffle=True,random_state=seed); mo=np.zeros(len(y)); ma=np.zeros(len(y)); mt=np.zeros(len(T)); mat=np.zeros(len(T))
    for f,(all_i,v) in enumerate(sk.split(X,y),1):
        ss=StratifiedShuffleSplit(1,test_size=.18,random_state=seed+f*17); a,b=next(ss.split(X[all_i],y[all_i])); i,e=all_i[a],all_i[b]; mean=X[i].mean(0,keepdims=True); std=X[i].std(0,keepdims=True)+1e-6
        xi,xe,xv,xt=(X[i]-mean)/std,(X[e]-mean)/std,(X[v]-mean)/std,(Z-mean)/std; bi,be,bv,bt=map(lambda z:logit(z).astype(np.float32),[base[i],base[e],base[v],baset])
        mo[v]=train_net(MLP(X.shape[1],cfg.dropout),[xi,bi],y[i],[xe,be],y[e],[xv,bv],cfg,seed+f*101,device); mt+=train_net(MLP(X.shape[1],cfg.dropout),[xi,bi],y[i],[xe,be],y[e],[xt,bt],cfg,seed+f*101,device)/cfg.meta_folds
        pi,pe,pv,pt=map(lambda z:logit(z).astype(np.float32),[P[i],P[e],P[v],Q]); ma[v]=train_net(MAT(len(LEVEL1),X.shape[1],cfg.d_model,cfg.heads,cfg.dropout),[pi,R[i],xi,bi],y[i],[pe,R[e],xe,be],y[e],[pv,R[v],xv,bv],cfg,seed+f*211,device); mat+=train_net(MAT(len(LEVEL1),X.shape[1],cfg.d_model,cfg.heads,cfg.dropout),[pi,R[i],xi,bi],y[i],[pe,R[e],xe,be],y[e],[pt,S,xt,bt],cfg,seed+f*211,device)/cfg.meta_folds
    return clip(mo),clip(mt),clip(ma),clip(mat)
