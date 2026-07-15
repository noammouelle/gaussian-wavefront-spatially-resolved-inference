"""Streaming image features and leakage-safe PCA for eta learnability studies."""
from __future__ import annotations
from pathlib import Path
import h5py
import numpy as np

ETA_NAMES = ("mu_x0","mu_y0","mu_vx0","mu_vy0","sigma_x","sigma_y","sigma_vx","sigma_vy")
T_DET = 3.8


def deterministic_split(n, seed=123, train_fraction=.6, val_fraction=.2):
    """Shot-level split; valid because simulator nuisances are independent per shot."""
    if not 0 < train_fraction < 1 or not 0 <= val_fraction < 1-train_fraction:
        raise ValueError("invalid split fractions")
    order=np.random.default_rng(seed).permutation(n)
    a=int(train_fraction*n); b=int((train_fraction+val_fraction)*n)
    return order[:a],order[a:b],order[b:]


def downsample_sum(image,bins):
    n=image.shape[0]
    if image.shape != (n,n) or n%bins: raise ValueError("square image resolution must be divisible by bins")
    k=n//bins
    return image.reshape(bins,k,bins,k).sum((1,3),dtype=np.float64)


def shot_features(ground,excited,half_range,bins=32):
    """Return ballistic summaries and cloud-normalized contrast profile."""
    ground=np.asarray(ground); excited=np.asarray(excited); total=ground.astype(float)+excited
    n=ground.shape[0]; edges=np.linspace(-half_range,half_range,n+1); c=(edges[:-1]+edges[1:])/2
    count=max(total.sum(),1.); mx=(total*c[:,None]).sum()/count; my=(total*c[None,:]).sum()/count
    vx=max((total*(c[:,None]-mx)**2).sum()/count,0); vy=max((total*(c[None,:]-my)**2).sum()/count,0)
    cov=(total*(c[:,None]-mx)*(c[None,:]-my)).sum()/count
    state=excited.sum()/count
    lo,hi=-4.,4.; bw=(hi-lo)/bins
    bx=np.floor(((c-mx)/max(np.sqrt(vx),1e-30)-lo)/bw).astype(int)
    by=np.floor(((c-my)/max(np.sqrt(vy),1e-30)-lo)/bw).astype(int)
    ix=np.where((bx>=0)&(bx<bins))[0]; iy=np.where((by>=0)&(by<bins))[0]
    flat=(bx[ix,None]*bins+by[None,iy]).ravel()
    g=np.bincount(flat,weights=ground[np.ix_(ix,iy)].ravel(),minlength=bins*bins)
    e=np.bincount(flat,weights=excited[np.ix_(ix,iy)].ravel(),minlength=bins*bins)
    w=g+e; contrast=(e-g)/(w+1.); global_contrast=(w*contrast).sum()/max(w.sum(),1.)
    # Ballistic-only quantities precede state-dependent scalar summaries.
    summary=np.array([mx,my,np.sqrt(vx),np.sqrt(vy),cov,count,global_contrast,state])
    return summary,contrast,w


def extract_paired_dataset(data_root,bins=32,max_shots=None,progress=None):
    """Stream paired Z0/Z100 HDF5 images into compact arrays."""
    pairs=[]
    for z0 in sorted(Path(data_root).glob("run_*/Z0/data_IMG.h5")):
        z100=z0.parents[1]/"Z100"/"data_IMG.h5"
        if z100.exists(): pairs.append((z0,z100))
    if not pairs: raise FileNotFoundError(f"No paired images under {data_root}")
    rows={k:[] for k in ("summary_z0","summary_z100","profile_z0","profile_z100","weight_z0","weight_z100","eta_z0","eta_z100","phi_z0","phi_z100","delta_phi","run_id","shot_id")}
    done=0
    for z0,z100 in pairs:
        try:
            f0=h5py.File(z0); f1=h5py.File(z100)
        except (BlockingIOError,OSError):
            if 'f0' in locals() and f0: f0.close()
            continue
        with f0,f1:
            n=min(len(f0["phi0"]),len(f1["phi0"])); half=float(f0.attrs["image_half_range"])
            for j in range(n):
                s0,p0,w0=shot_features(f0["images_s0"][j],f0["images_s1"][j],half,bins)
                s1,p1,w1=shot_features(f1["images_s0"][j],f1["images_s1"][j],half,bins)
                rows["summary_z0"].append(s0); rows["summary_z100"].append(s1)
                rows["profile_z0"].append(p0); rows["profile_z100"].append(p1); rows["weight_z0"].append(w0); rows["weight_z100"].append(w1)
                rows["eta_z0"].append([f0[k][j] for k in ETA_NAMES]); rows["eta_z100"].append([f1[k][j] for k in ETA_NAMES])
                rows["phi_z0"].append(f0["phi0"][j]); rows["phi_z100"].append(f1["phi0"][j]); rows["delta_phi"].append(f1["delta_phi"][j])
                rows["run_id"].append(z0.parents[1].name); rows["shot_id"].append(j); done+=1
                if progress and done%progress==0: print(f"extracted {done} paired shots",flush=True)
                if max_shots and done>=max_shots: break
        if max_shots and done>=max_shots: break
    return {k:np.asarray(v) for k,v in rows.items()}


def fit_weighted_profile_pca(profiles,weights,train,n_components=12):
    """Fit training-only PCA of shape contrast, returning all-shot scores."""
    from sklearn.decomposition import PCA
    profiles=np.asarray(profiles,float); weights=np.asarray(weights,float)
    global_c=(weights*profiles).sum(1)/np.maximum(weights.sum(1),1)
    shape=profiles-global_c[:,None]
    mean=shape[train].mean(0); scale=shape[train].std(0); scale[scale==0]=1
    standardized=(shape-mean)/scale
    model=PCA(n_components=min(n_components,len(train),shape.shape[1]),svd_solver="randomized",random_state=0)
    model.fit(standardized[train])
    return model.transform(standardized),model,mean,scale


def derived_targets(eta):
    eta=np.asarray(eta); out=dict(zip(ETA_NAMES,eta.T))
    out.update({
        "mu_xf":eta[:,0]+T_DET*eta[:,2], "mu_yf":eta[:,1]+T_DET*eta[:,3],
        "sigma_xf":np.sqrt(eta[:,4]**2+(T_DET*eta[:,6])**2),
        "sigma_yf":np.sqrt(eta[:,5]**2+(T_DET*eta[:,7])**2),
        # Coordinates along final-mean-preserving fibre directions.
        "hidden_mu_x_std":(eta[:,2]/1e-5-T_DET*eta[:,0]/1e-5)/np.sqrt(1+T_DET**2),
        "hidden_mu_y_std":(eta[:,3]/1e-5-T_DET*eta[:,1]/1e-5)/np.sqrt(1+T_DET**2),
    })
    return out
