#!/usr/bin/env python
"""Compare ballistic, PCA-shape, known-phase, and paired-image eta inference."""
from __future__ import annotations
import argparse,json,sys,time
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(REPO))
from helpers.eta_learnability import *  # noqa: E402,F403

def make_split(data,seed,mode):
    n=len(data["phi_z0"])
    if mode=="shot": return deterministic_split(n,seed)
    from sklearn.model_selection import GroupShuffleSplit
    idx=np.arange(n); groups=data["run_id"]
    tv,test=next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=seed).split(idx,groups=groups))
    tr0,va0=next(GroupShuffleSplit(n_splits=1,test_size=.25,random_state=seed+1).split(tv,groups=groups[tv]))
    return tv[tr0],tv[va0],test

def regression(X,Y,train,test,alpha,degree=1):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler,PolynomialFeatures
    from sklearn.linear_model import Ridge
    xs=StandardScaler(); poly=PolynomialFeatures(degree,include_bias=False); ys=StandardScaler()
    xt=poly.fit_transform(xs.fit_transform(X[train])); yt=ys.fit_transform(Y[train])
    model=Ridge(alpha=alpha).fit(xt,yt)
    pred=ys.inverse_transform(model.predict(poly.transform(xs.transform(X[test]))))
    return pred

def pls_regression(X,Y,train,test,components):
    from sklearn.cross_decomposition import PLSRegression
    model=PLSRegression(n_components=min(components,len(train)-1,X.shape[1]),scale=True,max_iter=1000)
    model.fit(X[train],Y[train]); return model.predict(X[test])

def evaluate_dataset(cache_path,label,args,out):
    if cache_path.exists(): data={k:v for k,v in np.load(cache_path).items()}
    else:
        data=extract_paired_dataset(args.data_root,bins=args.bins,max_shots=args.max_shots,progress=100)
        cache_path.parent.mkdir(parents=True,exist_ok=True); np.savez_compressed(cache_path,**data)
    train,val,test=make_split(data,args.seed,args.split_mode)
    pc0,m0,_,_=fit_weighted_profile_pca(data["profile_z0"],data["weight_z0"],train,args.pcs)
    pc1,m1,_,_=fit_weighted_profile_pca(data["profile_z100"],data["weight_z100"],train,args.pcs)
    b0=data["summary_z0"][:,:5]; b1=data["summary_z100"][:,:5]
    f0=np.c_[b0,pc0]; f1=np.c_[b1,pc1]
    bright0=data["summary_z0"][:,5:8]; bright1=data["summary_z100"][:,5:8]
    ph0=np.c_[np.sin(data["phi_z0"]),np.cos(data["phi_z0"])]; ph1=np.c_[np.sin(data["phi_z100"]),np.cos(data["phi_z100"])]
    phase_aug=lambda f,ph: np.c_[f,ph,f*ph[:,0,None],f*ph[:,1,None]]
    paired=np.c_[f0,f1]
    feature_sets={"ballistic":(b0,b1),"ballistic_plus_brightness":(np.c_[b0,bright0],np.c_[b1,bright1]),"normalized_shape_pca":(f0,f1),"normalized_shape_pca_known_phase":(phase_aug(f0,ph0),phase_aug(f1,ph1)),"shape_plus_brightness":(np.c_[f0,bright0],np.c_[f1,bright1]),"paired_normalized_shape_pca":(paired,paired),"paired_known_phase":(phase_aug(paired,ph0),phase_aug(paired,ph1))}
    raw0=np.c_[b0,data["profile_z0"]]; raw1=np.c_[b1,data["profile_z100"]]
    rows=[]
    for port,eta,phi in (("Z0",data["eta_z0"],data["phi_z0"]),("Z100",data["eta_z100"],data["phi_z100"])):
        targets=derived_targets(eta); names=list(targets); Y=np.column_stack([targets[n] for n in names]); pidx=port=="Z100"
        for family,pair in feature_sets.items():
            pred=regression(pair[int(pidx)],Y,train,test,args.alpha,args.degree)
            for j,name in enumerate(names):
                y=Y[test,j]; mse=np.mean((y-pred[:,j])**2); var=np.var(y)
                rows.append({"dataset":label,"port":port,"feature_family":family,"target":name,"r2":1-mse/var if var else np.nan,"rmse":np.sqrt(mse),"prior_std":np.std(y),"rmse_over_prior":np.sqrt(mse/max(var,1e-300)),"n_train":len(train),"n_test":len(test)})
        for family,X in (("supervised_profile_pls",raw0 if port=="Z0" else raw1),("supervised_profile_pls_known_phase",phase_aug(raw0,ph0) if port=="Z0" else phase_aug(raw1,ph1))):
            pred=pls_regression(X,Y,train,test,args.pls_components)
            for j,name in enumerate(names):
                y=Y[test,j]; mse=np.mean((y-pred[:,j])**2); var=np.var(y)
                rows.append({"dataset":label,"port":port,"feature_family":family,"target":name,"r2":1-mse/var if var else np.nan,"rmse":np.sqrt(mse),"prior_std":np.std(y),"rmse_over_prior":np.sqrt(mse/max(var,1e-300)),"n_train":len(train),"n_test":len(test)})
        phase_y=np.c_[np.sin(phi),np.cos(phi)]
        phase_inputs={"phase_from_normalized_shape":pc0 if port=="Z0" else pc1,"phase_from_total_count":bright0[:,:1] if port=="Z0" else bright1[:,:1],"phase_from_global_state_fraction":bright0[:,1:] if port=="Z0" else bright1[:,1:],"phase_from_shape_and_brightness":np.c_[pc0,bright0] if port=="Z0" else np.c_[pc1,bright1]}
        for phase_family,Xphase in phase_inputs.items():
            pp=regression(Xphase,phase_y,train,test,args.alpha,args.degree)
            err=np.angle(np.exp(1j*(np.arctan2(pp[:,0],pp[:,1])-phi[test]))); phase_rmse=np.sqrt(np.mean(err**2))
            rows.append({"dataset":label,"port":port,"feature_family":phase_family,"target":"total_phase","r2":np.nan,"rmse":phase_rmse,"prior_std":np.pi/np.sqrt(3),"rmse_over_prior":phase_rmse/(np.pi/np.sqrt(3)),"n_train":len(train),"n_test":len(test)})
    np.savez_compressed(out/f"{label}_model_artifacts.npz",train_index=train,val_index=val,test_index=test,pca_scores_z0=pc0,pca_scores_z100=pc1,pca_components_z0=m0.components_,pca_components_z100=m1.components_,pca_variance_z0=m0.explained_variance_ratio_,pca_variance_z100=m1.explained_variance_ratio_)
    return pd.DataFrame(rows),data

def plot_results(table,out):
    targets=["mu_x0","mu_vx0","sigma_x","sigma_vx","hidden_mu_x_std","total_phase"]
    for dataset in table.dataset.unique():
        fig,axs=plt.subplots(2,3,figsize=(13,7),sharey=True)
        for ax,target in zip(axs.flat,targets):
            g=table[(table.dataset==dataset)&(table.target==target)]
            g.pivot(index="feature_family",columns="port",values="rmse_over_prior").plot.bar(ax=ax)
            ax.axhline(1,color="k",ls="--",lw=1); ax.set(title=target,ylabel="RMSE / test SD"); ax.tick_params(axis="x",rotation=35)
        fig.tight_layout(); fig.savefig(out/f"{dataset}_identifiability.png",dpi=160); plt.close(fig)

def main():
    p=argparse.ArgumentParser(); p.add_argument("--data-root",type=Path,required=True); p.add_argument("--label",required=True); p.add_argument("--output",type=Path,default=REPO/"results/eta_learnability"); p.add_argument("--cache",type=Path); p.add_argument("--bins",type=int,default=32); p.add_argument("--pcs",type=int,default=12); p.add_argument("--pls-components",type=int,default=8); p.add_argument("--alpha",type=float,default=10.); p.add_argument("--degree",type=int,choices=(1,2),default=1); p.add_argument("--seed",type=int,default=123); p.add_argument("--split-mode",choices=("shot","run"),default="shot"); p.add_argument("--max-shots",type=int); args=p.parse_args(); args.output.mkdir(parents=True,exist_ok=True)
    cache=args.cache or args.output/f"{args.label}_features_bins{args.bins}.npz"; started=time.time()
    table,data=evaluate_dataset(cache,args.label,args,args.output); table.to_csv(args.output/f"{args.label}_metrics.csv",index=False); plot_results(table,args.output)
    (args.output/f"{args.label}_run.json").write_text(json.dumps({"args":{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},"n_shots":len(data["phi_z0"]),"elapsed_seconds":time.time()-started},indent=2)+"\n")
    print(table[table.target.isin(["mu_x0","mu_vx0","sigma_x","sigma_vx","hidden_mu_x_std","total_phase"])].to_string(index=False))
if __name__=="__main__": main()
