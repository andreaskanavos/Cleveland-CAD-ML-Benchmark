from __future__ import annotations
import argparse, math, random
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
from scipy.stats import friedmanchisquare, wilcoxon, norm
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, roc_curve
)
import shap

COLS = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal','num']
NOMINAL = ['cp','restecg','slope','thal']
BINARY = ['sex','fbs','exang']
CONT = ['age','trestbps','chol','thalach','oldpeak']
COUNT = ['ca']
MODEL_ORDER = ['Logistic Regression','Decision Tree','k-Nearest Neighbors','Random Forest',
               'Gradient Boosting','Support Vector Machine','Soft-Voting Ensemble','Multilayer Perceptron']


def load_data(path):
    raw = pd.read_csv(path, header=None, names=COLS, na_values='?')
    complete = raw.dropna().copy()
    complete['target'] = (complete['num'] > 0).astype(int)
    X = complete.drop(columns=['num','target'])
    y = complete['target']
    return raw, complete, X, y


def feature_groups(features):
    features=set(features)
    return ([c for c in NOMINAL if c in features], [c for c in BINARY if c in features],
            [c for c in CONT if c in features], [c for c in COUNT if c in features])


def preprocessor(features, scale_continuous):
    nom, binary, cont, count = feature_groups(features)
    trs=[]
    if nom: trs.append(('cat',OneHotEncoder(handle_unknown='ignore', sparse_output=False),nom))
    if cont: trs.append(('cont',StandardScaler() if scale_continuous else 'passthrough',cont))
    if binary: trs.append(('bin','passthrough',binary))
    if count: trs.append(('count','passthrough',count))
    return ColumnTransformer(trs, remainder='drop', verbose_feature_names_out=False)


def model_set(features, knn_k):
    return {
        'Logistic Regression': Pipeline([('prep',preprocessor(features,True)),('clf',LogisticRegression(C=1.0,solver='lbfgs',max_iter=1000))]),
        'Decision Tree': Pipeline([('prep',preprocessor(features,False)),('clf',DecisionTreeClassifier(criterion='gini',max_depth=5,min_samples_split=5,random_state=42))]),
        'k-Nearest Neighbors': Pipeline([('prep',preprocessor(features,True)),('clf',KNeighborsClassifier(n_neighbors=knn_k,weights='uniform',metric='euclidean'))]),
        'Random Forest': Pipeline([('prep',preprocessor(features,False)),('clf',RandomForestClassifier(n_estimators=200,max_features='sqrt',random_state=42,n_jobs=-1))]),
        'Gradient Boosting': Pipeline([('prep',preprocessor(features,False)),('clf',GradientBoostingClassifier(n_estimators=200,learning_rate=0.1,max_depth=3,subsample=0.8,random_state=42))]),
        'Support Vector Machine': Pipeline([('prep',preprocessor(features,True)),('clf',SVC(kernel='rbf',C=1.0,gamma='scale',probability=True,random_state=42))])
    }


def choose_knn(Xtr,ytr,features,seed):
    cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=seed)
    means={}
    for k in range(1,21):
        scores=[]
        for a,b in cv.split(Xtr,ytr):
            m=model_set(features,k)['k-Nearest Neighbors']
            m.fit(Xtr.iloc[a],ytr.iloc[a])
            scores.append(accuracy_score(ytr.iloc[b],m.predict(Xtr.iloc[b])))
        means[k]=float(np.mean(scores))
    # deterministic tie-break: smallest k among tied maxima
    best=max(means.values())
    selected=min(k for k,v in means.items() if np.isclose(v,best))
    return selected, means


def threshold_metrics(y,score,threshold=.5):
    pred=(np.asarray(score)>=threshold).astype(int)
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    return {
        'Accuracy':accuracy_score(y,pred),'Precision':precision_score(y,pred,zero_division=0),
        'Sensitivity':recall_score(y,pred,zero_division=0),'Specificity':tn/(tn+fp),
        'F1-score':f1_score(y,pred,zero_division=0),'ROC AUC':roc_auc_score(y,score),
        'TN':int(tn),'FP':int(fp),'FN':int(fn),'TP':int(tp)
    }


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True,warn_only=True)


class MLP(nn.Module):
    def __init__(self,n_in):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(n_in,64),nn.ReLU(),nn.Dropout(.30),
                               nn.Linear(64,32),nn.ReLU(),nn.Dropout(.20),nn.Linear(32,1))
    def forward(self,x): return self.net(x).squeeze(1)


def train_mlp(Xtr,ytr,Xeval,features,seed,max_epochs=200,patience=10,batch=32):
    seed_all(seed)
    pre=preprocessor(features,True)
    Xt=pre.fit_transform(Xtr); Xe=pre.transform(Xeval)
    ids=np.arange(len(ytr))
    tr,va=train_test_split(ids,test_size=.20,random_state=seed,stratify=np.asarray(ytr))
    xtr=torch.tensor(Xt[tr],dtype=torch.float32); ytr_t=torch.tensor(np.asarray(ytr)[tr],dtype=torch.float32)
    xva=torch.tensor(Xt[va],dtype=torch.float32); yva_t=torch.tensor(np.asarray(ytr)[va],dtype=torch.float32)
    model=MLP(Xt.shape[1]); opt=torch.optim.Adam(model.parameters(),lr=.001); lossf=nn.BCEWithLogitsLoss()
    best=None; best_loss=float('inf'); stale=0; completed=0
    for epoch in range(1,max_epochs+1):
        model.train(); perm=torch.randperm(len(xtr))
        for start in range(0,len(xtr),batch):
            ix=perm[start:start+batch]
            opt.zero_grad(); loss=lossf(model(xtr[ix]),ytr_t[ix]); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad(): vl=float(lossf(model(xva),yva_t).item())
        completed=epoch
        if vl < best_loss - 1e-12:
            best_loss=vl; best={k:v.detach().clone() for k,v in model.state_dict().items()}; stale=0
        else:
            stale += 1
            if stale>=patience: break
    model.load_state_dict(best); model.eval()
    with torch.no_grad(): score=torch.sigmoid(model(torch.tensor(Xe,dtype=torch.float32))).cpu().numpy()
    return score, completed


def soft_scores(fitted,X):
    names=['Logistic Regression','Random Forest','Gradient Boosting','Support Vector Machine']
    return np.mean([fitted[n].predict_proba(X)[:,1] for n in names],axis=0)


def hanley_mcneil_ci(y,score,alpha=.05):
    y=np.asarray(y); auc=roc_auc_score(y,score); n1=(y==1).sum(); n0=(y==0).sum()
    q1=auc/(2-auc); q2=2*auc*auc/(1+auc)
    se=math.sqrt((auc*(1-auc)+(n1-1)*(q1-auc*auc)+(n0-1)*(q2-auc*auc))/(n0*n1))
    z=norm.ppf(1-alpha/2)
    return max(0.,auc-z*se),min(1.,auc+z*se)


def stratified_bootstrap_ci(y,score,n_boot=5000,seed=42):
    y=np.asarray(y); score=np.asarray(score); rng=np.random.default_rng(seed)
    pos=np.flatnonzero(y==1); neg=np.flatnonzero(y==0)
    vals={k:[] for k in ['Accuracy','Precision','Sensitivity','Specificity','F1-score']}
    for _ in range(n_boot):
        idx=np.concatenate([rng.choice(neg,len(neg),replace=True),rng.choice(pos,len(pos),replace=True)])
        r=threshold_metrics(y[idx],score[idx])
        for k in vals: vals[k].append(r[k])
    return {k:(float(np.percentile(v,2.5)),float(np.percentile(v,97.5))) for k,v in vals.items()}


def holm_adjust(pvals):
    p=np.asarray(pvals,float); m=len(p); order=np.argsort(p); adj=np.empty(m); running=0.
    for rank,idx in enumerate(order):
        running=max(running,(m-rank)*p[idx]); adj[idx]=min(1.,running)
    return adj


def evaluate_task(X,y,out,prefix='',include_shap=False):
    features=list(X.columns)
    idx_tr,idx_te=train_test_split(np.arange(len(X)),test_size=.20,random_state=42,stratify=y)
    Xtr,Xte=X.iloc[idx_tr],X.iloc[idx_te]; ytr,yte=y.iloc[idx_tr],y.iloc[idx_te]

    k,kgrid=choose_knn(Xtr,ytr,features,42)
    pd.DataFrame({'k':list(kgrid),'mean_accuracy':[kgrid[q] for q in kgrid]}).to_csv(out/f'{prefix}knn_heldout_selection.csv',index=False)

    fitted={}; scores={}; rows=[]
    for name,m in model_set(features,k).items():
        m.fit(Xtr,ytr); fitted[name]=m; s=m.predict_proba(Xte)[:,1]; scores[name]=s
        r=threshold_metrics(yte,s); r['Model']=name; rows.append(r)
    s=soft_scores(fitted,Xte); scores['Soft-Voting Ensemble']=s; r=threshold_metrics(yte,s); r['Model']='Soft-Voting Ensemble'; rows.append(r)
    s,ep=train_mlp(Xtr,ytr,Xte,features,42); scores['Multilayer Perceptron']=s; r=threshold_metrics(yte,s); r['Model']='Multilayer Perceptron'; r['Completed epochs']=ep; rows.append(r)
    hold=pd.DataFrame(rows); hold.to_csv(out/f'{prefix}holdout_results.csv',index=False)

    if prefix=='':
        cirows=[]
        for name in MODEL_ORDER:
            s=scores[name]; r=threshold_metrics(yte,s); b=stratified_bootstrap_ci(yte,s,5000,42); alo,ahi=hanley_mcneil_ci(yte,s)
            q={'Model':name}
            for met in ['Accuracy','Precision','Sensitivity','Specificity','F1-score']:
                q[met]=r[met]; q[f'{met}_lo'],q[f'{met}_hi']=b[met]
            q['ROC AUC']=r['ROC AUC']; q['ROC AUC_lo']=alo; q['ROC AUC_hi']=ahi; cirows.append(q)
        pd.DataFrame(cirows).to_csv(out/'holdout_95ci.csv',index=False)

    outer=StratifiedKFold(10,shuffle=True,random_state=42)
    fold_rows=[]; selected=[]
    for f,(a,b) in enumerate(outer.split(X,y)):
        Xo,Xv=X.iloc[a],X.iloc[b]; yo,yv=y.iloc[a],y.iloc[b]
        kk,_=choose_knn(Xo,yo,features,42+f); selected.append(kk); ff={}
        for name,m in model_set(features,kk).items():
            m.fit(Xo,yo); ff[name]=m; s=m.predict_proba(Xv)[:,1]; rr=threshold_metrics(yv,s); rr.update(Model=name,Fold=f+1); fold_rows.append(rr)
        s=soft_scores(ff,Xv); rr=threshold_metrics(yv,s); rr.update(Model='Soft-Voting Ensemble',Fold=f+1); fold_rows.append(rr)
        s,_=train_mlp(Xo,yo,Xv,features,42+f); rr=threshold_metrics(yv,s); rr.update(Model='Multilayer Perceptron',Fold=f+1); fold_rows.append(rr)
    fr=pd.DataFrame(fold_rows); fr.to_csv(out/f'{prefix}fold_metrics.csv',index=False)
    summ=[]
    for name in MODEL_ORDER:
        g=fr[fr.Model==name]; row={'Model':name}
        for met in ['Accuracy','Precision','Sensitivity','Specificity','F1-score','ROC AUC']:
            row[met]=g[met].mean(); row[met+' SD']=g[met].std(ddof=1)
        summ.append(row)
    pd.DataFrame(summ).to_csv(out/f'{prefix}cross_validation_summary.csv',index=False)
    (out/f'{prefix}nested_knn_values.txt').write_text(', '.join(map(str,selected))+'\n',encoding='utf-8')

    if prefix=='':
        piv=fr.pivot(index='Fold',columns='Model',values='Accuracy')[MODEL_ORDER]
        stat,p=friedmanchisquare(*[piv[n].values for n in MODEL_ORDER])
        pairs=[]; raw=[]
        for a,b in combinations(MODEL_ORDER,2):
            w,pv=wilcoxon(piv[a],piv[b],zero_method='wilcox',alternative='two-sided',method='auto')
            pairs.append([a,b,w,pv]); raw.append(pv)
        adj=holm_adjust(raw)
        for row,a in zip(pairs,adj): row.append(a)
        pd.DataFrame(pairs,columns=['Model 1','Model 2','Wilcoxon W','Raw p','Holm-adjusted p']).to_csv(out/'wilcoxon_holm.csv',index=False)
        (out/'friedman.txt').write_text(f'Friedman chi-square = {stat:.6f}\ndf = 7\np = {p:.9f}\npaired comparisons = 28\ninferential metric = fold-level accuracy only\n',encoding='utf-8')

        rep=[]
        for seed in range(43,63):
            s,e=train_mlp(Xtr,ytr,Xte,features,seed); rr=threshold_metrics(yte,s); rr.update(seed=seed,epochs=e); rep.append(rr)
        pd.DataFrame(rep).to_csv(out/'mlp_repeated_runs_seeds_43_62.csv',index=False)

        # ROC curves
        plt.figure(figsize=(7,5))
        for name in MODEL_ORDER:
            fpr,tpr,_=roc_curve(yte,scores[name]); plt.plot(fpr,tpr,label=f'{name} (AUC={roc_auc_score(yte,scores[name]):.3f})')
        plt.plot([0,1],[0,1],'--',label='Random'); plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate'); plt.legend(fontsize=7); plt.tight_layout(); plt.savefig(out/'roc_curves.png',dpi=200); plt.close()

        # Confusion matrices stored as CSV for exact verification
        cm=[]
        for name in MODEL_ORDER:
            r=threshold_metrics(yte,scores[name]); cm.append({'Model':name,'TN':r['TN'],'FP':r['FP'],'FN':r['FN'],'TP':r['TP']})
        pd.DataFrame(cm).to_csv(out/'confusion_matrices.csv',index=False)

        # Tree feature importances and SHAP on fitted Gradient Boosting model
        for name in ['Random Forest','Gradient Boosting']:
            pipe=fitted[name]; names=pipe.named_steps['prep'].get_feature_names_out(); imp=pipe.named_steps['clf'].feature_importances_
            pd.DataFrame({'feature':names,'importance':imp}).sort_values('importance',ascending=False).to_csv(out/f'{name.lower().replace(" ","_")}_feature_importance.csv',index=False)
        if include_shap:
            gb=fitted['Gradient Boosting']; prep=gb.named_steps['prep']; model=gb.named_steps['clf']
            train_mat=prep.transform(Xtr); test_mat=prep.transform(Xte); fn=prep.get_feature_names_out()
            rng=np.random.default_rng(42); bg_idx=rng.choice(len(train_mat),size=min(100,len(train_mat)),replace=False); bg=train_mat[bg_idx]
            explainer=shap.TreeExplainer(model,data=bg,feature_perturbation='interventional',model_output='probability')
            sv=explainer(test_mat)
            vals=np.asarray(sv.values)
            pd.DataFrame({'feature':fn,'mean_abs_shap':np.abs(vals).mean(axis=0)}).sort_values('mean_abs_shap',ascending=False).to_csv(out/'shap_global_importance.csv',index=False)
            shap.summary_plot(vals,test_mat,feature_names=fn,show=False); plt.tight_layout(); plt.savefig(out/'shap_beeswarm.png',dpi=200,bbox_inches='tight'); plt.close()
            shap.summary_plot(vals,test_mat,feature_names=fn,plot_type='bar',show=False); plt.tight_layout(); plt.savefig(out/'shap_bar.png',dpi=200,bbox_inches='tight'); plt.close()
    return k


def main(data,out):
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    raw,complete,X,y=load_data(data)
    pd.DataFrame([{'original_records':len(raw),'excluded_missing':len(raw)-len(complete),'analytic_records':len(complete),
                   'CAD_negative':int((y==0).sum()),'CAD_positive':int((y==1).sum())}]).to_csv(out/'sample_flow.csv',index=False)
    k=evaluate_task(X,y,out,'',include_shap=True)
    evaluate_task(X.drop(columns=['ca']),y,out,'ca_exclusion_',include_shap=False)
    (out/'run_summary.txt').write_text(f'Held-out k selected by training-only five-fold stratified CV: {k}\n',encoding='utf-8')


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--data',required=True)
    ap.add_argument('--out',default='reproduced_outputs')
    args=ap.parse_args(); main(args.data,args.out)
