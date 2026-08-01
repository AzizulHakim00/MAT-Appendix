from __future__ import annotations
import json, random, subprocess, sys, warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Optional, Sequence
warnings.filterwarnings('ignore')

def ensure(pkg, imp=None):
    try: __import__(imp or pkg.replace('-','_'))
    except Exception: subprocess.check_call([sys.executable,'-m','pip','install','-q',pkg])
for p,i in [('ucimlrepo','ucimlrepo'),('xgboost','xgboost'),('catboost','catboost')]: ensure(p,i)
import numpy as np, pandas as pd, torch
from ucimlrepo import fetch_ucirepo
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.metrics import accuracy_score,average_precision_score,balanced_accuracy_score,brier_score_loss,confusion_matrix,f1_score,fbeta_score,log_loss,matthews_corrcoef,precision_score,recall_score,roc_auc_score

VERSION='3.5.0'
VARIANTS=['A_Global','B_ExpertBlend','C_Gated','D_MLPResidual','E_MATResidual']
GLOBAL=['RF','ET','XGB','CAT','XGB45','ET30']
LEVEL1=GLOBAL+['ClinicalLab','Ultrasound','Missingness']
V32={'Accuracy':.827214,'Balanced_Accuracy':.833874,'Precision_PPV':.617284,'Recall_Sensitivity_TPR':.847458,'Specificity_TNR':.820290,'F1':.714286,'F2':.788644,'MCC':.610130,'ROC_AUC':.883100,'PR_AUC':.701128,'Brier':.117716,'Log_Loss':.373076}

@dataclass
class Config:
    seed:int=2035; outer_folds:int=5; repeats:int=2; inner_folds:int=3; meta_folds:int=3
    blend_candidates:int=2500; bootstrap_iterations:int=1500
    min_sensitivity:float=.84; preferred_specificity:float=.80
    epochs:int=70; patience:int=10; batch_size:int=48; d_model:int=24; heads:int=2; dropout:float=.30
    lr:float=8e-4; weight_decay:float=3e-3
    drive_root:str='/content/drive/MyDrive/MAT-Appendix/v35_runs'; fallback_root:str='/content/MAT-Appendix/v35_runs'
    expected_n:int=463; expected_positive:int=118; expected_negative:int=345
CFG=Config()

def native(x:Any)->Any:
    if isinstance(x,dict): return {str(k):native(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)): return [native(v) for v in x]
    if isinstance(x,np.ndarray): return native(x.tolist())
    if isinstance(x,np.integer): return int(x)
    if isinstance(x,np.floating): return float(x) if np.isfinite(x) else None
    if isinstance(x,np.bool_): return bool(x)
    if isinstance(x,Path): return str(x)
    return x

def seed_all(seed:int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def mount_root(cfg:Config)->Path:
    try:
        from google.colab import drive
        if not Path('/content/drive/MyDrive').exists(): drive.mount('/content/drive')
    except Exception as e: print('Drive unavailable; using /content:',e)
    p=Path(cfg.drive_root if Path('/content/drive/MyDrive').exists() else cfg.fallback_root); p.mkdir(parents=True,exist_ok=True); return p

def norm(s:pd.Series)->pd.Series:
    return s.astype(str).str.strip().str.lower().str.replace(r'[^a-z0-9]+',' ',regex=True).str.strip()
def canon(s)->str: return str(s).strip().lower().replace(' ','_')
def find_col(cols:Sequence[str], names:Sequence[str])->Optional[str]:
    d={canon(c):str(c) for c in cols}
    for n in names:
        if canon(n) in d: return d[canon(n)]
    for c in cols:
        if any(canon(n) in canon(c) for n in names): return str(c)
    return None

LEAK_EXACT={'length_of_stay','management','severity','diagnosis','peritonitis','perforation','appendicular_abscess','abscess_location'}
LEAK_PART=('histology','histopath','pathology_result','operation','operative','surgery','postoperative','discharge','complication_label','gangren','perforat','abscess','peritonitis')
def load_cohort(cfg:Config):
    ds=fetch_ucirepo(id=938); X=ds.data.features.copy(); t=ds.data.targets.copy()
    dc=find_col(t.columns,['Diagnosis']); sc=find_col(t.columns,['Severity'])
    if dc is None or sc is None: raise RuntimeError(f'Target columns missing: {list(t.columns)}')
    d,s=norm(t[dc]),norm(t[sc]); keep=d.str.contains('appendicitis') & ~d.str.contains('no appendicitis') & s.isin(['complicated','uncomplicated'])
    X=X.loc[keep].reset_index(drop=True); y=(s.loc[keep].reset_index(drop=True)=='complicated').astype(int)
    keepcols=[]; audit=[]
    for c in X.columns:
        k=canon(c); reason=None
        if k in LEAK_EXACT: reason='target/post-decision/direct-endpoint'
        elif any(q in k for q in LEAK_PART) and 'lymph_node' not in k: reason='post-decision/pathology/direct-complication proxy'
        if reason: audit.append({'Feature':str(c),'Action':'DROP','Reason':reason})
        else: keepcols.append(c)
    X=X[keepcols].copy(); observed=(len(X),int(y.sum()),int((1-y).sum())); expected=(cfg.expected_n,cfg.expected_positive,cfg.expected_negative)
    if observed!=expected: raise RuntimeError(f'Corrected cohort mismatch: {observed} != {expected}')
    return X,y,pd.DataFrame(audit)

def modality(name:str)->str:
    n=canon(name)
    lab=('wbc','leuk','crp','neut','lymph','platelet','hemoglobin','haemoglobin','hematocrit','rbc','eryth','mcv','mch','rdw','bilirubin','creatin','sodium','potassium','urine','ketone')
    us=('appendix','diameter','ultrasound','sonograph','us_','_us','free_fluid','fluid','compress','hyperemia','echogenic','coprostasis','target_sign','lymph_node','bowel_wall','meteorism')
    clinical=('age','sex','bmi','height','weight','duration','pain','vomit','nausea','fever','temperature','rebound','guarding','tender','migration','anorexia','diarr','dysuria','stool','score','pas','alvarado','psoas','rovsing','cough','percussion')
    if any(k in n for k in lab): return 'laboratory'
    if any(k in n for k in us): return 'ultrasound'
    if any(k in n for k in clinical): return 'clinical'
    return 'other'
def num(s): return pd.to_numeric(s,errors='coerce')
def engineer(X:pd.DataFrame)->pd.DataFrame:
    z=X.copy(); groups={g:[c for c in z if modality(c)==g] for g in ['clinical','laboratory','ultrasound','other']}
    z['Missing_Total']=z.isna().sum(1).astype(float)
    for g,cs in groups.items(): z[f'Missing_{g.title()}']=z[cs].isna().sum(1).astype(float) if cs else 0.
    for c in list(z.columns):
        if z[c].isna().any(): z[f'{c}__Missing']=z[c].isna().astype(np.int8)
    def prod(name,a,b):
        x,y=find_col(z.columns,a),find_col(z.columns,b)
        if x and y: z[name]=num(z[x])*num(z[y])
    def ratio(name,a,b):
        x,y=find_col(z.columns,a),find_col(z.columns,b)
        if x and y: z[name]=num(z[x])/num(z[y]).replace(0,np.nan)
    prod('CRP_x_WBC',['CRP'],['WBC','Leukocytes']); prod('CRP_x_Neutrophils',['CRP'],['Neutrophil_Percentage','Neutrophils'])
    prod('CRP_x_AppendixDiameter',['CRP'],['Appendix_Diameter']); prod('WBC_x_AppendixDiameter',['WBC','Leukocytes'],['Appendix_Diameter'])
    prod('Duration_x_CRP',['Symptoms_Duration','Duration_of_Symptoms'],['CRP']); prod('Duration_x_WBC',['Symptoms_Duration','Duration_of_Symptoms'],['WBC','Leukocytes'])
    prod('Alvarado_x_PAS',['Alvarado_Score'],['Paedriatic_Appendicitis_Score','Pediatric_Appendicitis_Score'])
    ratio('CRP_WBC_Ratio',['CRP'],['WBC','Leukocytes']); ratio('NLR',['Neutrophil_Percentage','Neutrophils'],['Lymphocyte_Percentage','Lymphocytes'])
    return z

class Prep:
    def fit(self,X):
        self.nc=[c for c in X if pd.api.types.is_numeric_dtype(X[c]) or num(X[c]).notna().mean()>=.9]; self.cc=[c for c in X if c not in self.nc]
        npip=Pipeline([('imp',SimpleImputer(strategy='median',add_indicator=True)),('scale',RobustScaler())])
        cpip=Pipeline([('imp',SimpleImputer(strategy='most_frequent')),('oh',OneHotEncoder(handle_unknown='ignore',min_frequency=3,sparse_output=False))])
        self.t=ColumnTransformer([('num',npip,self.nc),('cat',cpip,self.cc)],sparse_threshold=0.,verbose_feature_names_out=True).fit(X)
        self.names=[str(x) for x in self.t.get_feature_names_out()]; return self
    def transform(self,X): return np.asarray(self.t.transform(X),dtype=np.float32)

def indices(names,groups): return np.asarray([i for i,n in enumerate(names) if modality(n) in set(groups)],int)
def miss_indices(names): return np.asarray([i for i,n in enumerate(names) if 'missing' in canon(n) or 'indicator' in canon(n)],int)
def clip(p): return np.clip(np.asarray(p,float),1e-6,1-1e-6)
def logit(p): p=clip(p); return np.log(p/(1-p))

def threshold_info(y,p,t):
    q=(p>=t).astype(int); tn,fp,fn,tp=confusion_matrix(y,q,labels=[0,1]).ravel(); se=tp/max(tp+fn,1); sp=tn/max(tn+fp,1)
    return {'threshold':float(t),'balanced':float((se+sp)/2),'sensitivity':float(se),'specificity':float(sp),'f1':float(f1_score(y,q,zero_division=0)),'mcc':float(matthews_corrcoef(y,q)) if len(np.unique(q))>1 else 0.}
def choose_threshold(y,p,cfg):
    grid=np.unique(np.r_[np.linspace(.08,.75,671),np.quantile(p,np.linspace(.02,.98,151))]); rows=[threshold_info(y,p,t) for t in grid]
    f=[r for r in rows if r['sensitivity']>=cfg.min_sensitivity and r['specificity']>=cfg.preferred_specificity] or [r for r in rows if r['sensitivity']>=cfg.min_sensitivity] or rows
    return max(f,key=lambda r:(r['balanced']+.05*r['f1']+.03*r['mcc'],r['specificity']))
def metric_row(y,p,t):
    p=clip(p); q=(p>=t).astype(int); tn,fp,fn,tp=confusion_matrix(y,q,labels=[0,1]).ravel(); sp=tn/max(tn+fp,1); npv=tn/max(tn+fn,1)
    return {'N':len(y),'Positive_N':int(y.sum()),'Accuracy':accuracy_score(y,q),'Balanced_Accuracy':balanced_accuracy_score(y,q),'Precision_PPV':precision_score(y,q,zero_division=0),'Recall_Sensitivity_TPR':recall_score(y,q,zero_division=0),'Specificity_TNR':sp,'NPV':npv,'F1':f1_score(y,q,zero_division=0),'F2':fbeta_score(y,q,beta=2,zero_division=0),'MCC':matthews_corrcoef(y,q) if len(np.unique(q))>1 else 0.,'ROC_AUC':roc_auc_score(y,p),'PR_AUC':average_precision_score(y,p),'Brier':brier_score_loss(y,p),'Log_Loss':log_loss(y,p,labels=[0,1]),'TP':int(tp),'TN':int(tn),'FP':int(fp),'FN':int(fn),'Threshold':float(t)}
def rank_score(y,p): return .52*average_precision_score(y,p)+.33*roc_auc_score(y,p)-.15*brier_score_loss(y,p)
def blend(P,y,n,seed):
    r=np.random.default_rng(seed); m=P.shape[1]; W=[np.ones(m)/m]+[np.eye(m)[i] for i in range(m)]
    for a in [.35,.7,1.,2.]: W.extend(r.dirichlet(np.full(m,a),size=max(1,n//4)))
    vals=[rank_score(y,clip(P@w)) for w in W]; j=int(np.argmax(vals)); return np.asarray(W[j]),float(vals[j])
def availability(X):
    us=[c for c in X if modality(c)=='ultrasound' and 'missing' not in canon(c)]; lab=[c for c in X if modality(c)=='laboratory' and 'missing' not in canon(c)]
    up=find_col(X.columns,['US_Performed','Ultrasound_Performed']); vis=find_col(X.columns,['Appendix_on_US','Appendix_Visible']); age=find_col(X.columns,['Age']); sex=find_col(X.columns,['Sex','Gender'])
    def binary(s):
        x=norm(s); a=np.full(len(s),.5); a[x.isin(['yes','true','1','performed','present','visible','male']).to_numpy()]=1.; a[x.isin(['no','false','0','not performed','absent','not visible','female']).to_numpy()]=0.; return a
    obs=(X[us].notna().sum(1).to_numpy()>0).astype(float) if us else np.zeros(len(X)); u=binary(X[up]) if up else obs; v=binary(X[vis]) if vis else obs
    a=num(X[age]).to_numpy(float) if age else np.zeros(len(X)); a=np.where(np.isnan(a),np.nanmedian(a) if not np.isnan(a).all() else 0,a); s=binary(X[sex]) if sex else np.full(len(X),.5)
    return np.c_[u,v,X.isna().mean(1),X[us].isna().mean(1) if us else 1.,X[lab].isna().mean(1) if lab else 1.,a,s].astype(np.float32)
