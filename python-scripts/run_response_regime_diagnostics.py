#!/usr/bin/env python
"""Run response-regime diagnostics on the repository's realistic PSMAPs."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import h5py, matplotlib.pyplot as plt, numpy as np, pandas as pd

REPO = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO))
from helpers.response_regime import *  # noqa: E402,F403

DEFAULT_MAPS = [REPO/"output-files/PSGRID4D_CONFOCAL_FINE_Z0.h5", REPO/"output-files/PSGRID4D_CONFOCAL_FINE_Z100.h5"]

def load_required(path):
    """Read the same HDF5 fields consumed by aispy.load_psmap, omitting huge path labels."""
    keys = ("atom_indices","initial_positions","initial_velocities","states","amp0","amp1","is_interfering","phase_shifts")
    with h5py.File(path) as f: return {k:f[k][...] for k in keys}

def trap_weights(coords):
    out=np.ones(len(coords))
    for i in range(4):
        ax=np.unique(coords[:,i]); w=np.gradient(ax); w[[0,-1]]*=.5
        out*=w[np.searchsorted(ax,coords[:,i])]
    return out

def cloud_projection(response,t,edges,sigmas,n_atoms):
    z=response.coordinates; density=np.prod(np.exp(-.5*(z/sigmas)**2)/(np.sqrt(2*np.pi)*sigmas),axis=1)
    weights=trap_weights(z)*density; weights/=weights.sum()
    bins,nb=final_bin_indices(z,t,edges,edges)
    a,q=aggregate_harmonics(response.a,response.q,bins,weights,nb)
    return weights,bins,a*n_atoms,q*n_atoms

def hidden_derivative(response,weights,bins,sigmas,t,n_atoms,scale):
    # Unit coordinate u changes mu_vx by scale and mu_x by -t*scale, preserving final mean.
    score=scale*(response.coordinates[:,2]/sigmas[2]**2-t*response.coordinates[:,0]/sigmas[0]**2)
    # Enforce the analytic final-density null bin by bin, removing finite-grid
    # and ROI leakage before attributing information to response variation.
    valid=bins>=0; nb=int(bins[valid].max()+1)
    mass=np.bincount(bins[valid],weights[valid],minlength=nb)
    moment=np.bincount(bins[valid],(weights*score)[valid],minlength=nb)
    score=score.copy(); score[valid]-=(moment/np.maximum(mass,1e-300))[bins[valid]]
    da,dq=aggregate_harmonics(response.a,response.q,bins,weights*score,nb)
    return n_atoms*da,n_atoms*dq

def analyze(path,cfg,out):
    name=path.stem.replace("PSGRID4D_CONFOCAL_FINE_","")
    response=harmonic_from_psmap(load_required(path),state=cfg["state"])
    bins_intr,nb=final_bin_indices(response.coordinates,cfg["t_det"],cfg["edges"],cfg["edges"])
    gen=generator_fields(response,cfg["t_det"])
    aeq,qeq=aggregate_harmonics(response.a,response.q,bins_intr,n_bins=nb)
    intrinsic=orbit_diagnostics(aeq,qeq)
    within=fibre_variation(response.q,bins_intr)
    weights,bins,a,q=cloud_projection(response,cfg["t_det"],cfg["edges"],cfg["sigmas"],cfg["n_atoms"])
    orbit=orbit_diagnostics(a,q); fisher=poisson_phase_information(a,q)
    da_u,dq_u=hidden_derivative(response,weights,bins,cfg["sigmas"],cfg["t_det"],cfg["n_atoms"],cfg["nuisance_scale"])
    prof=[]; degeneracy=[]
    for phi in fisher["phases"]:
        e=np.exp(1j*phi); mu=a+np.real(q*e); jp=np.real(1j*q*e); ju=da_u+np.real(dq_u*e)
        val,raw,removed=profiled_information(ju,jp,mu); prof.append(val); degeneracy.append(np.sqrt(removed/raw) if raw>0 else 0)
    prof=np.asarray(prof); hidden_snr=float(np.sqrt(np.median(prof)))

    # alpha scan (cloud projection recomputed because intrinsic phasors change)
    alpha_rows=[]
    for alpha in cfg["alphas"]:
        rr=apply_fibre_strength(response,bins_intr,alpha); wa,bb,aa,qq=cloud_projection(rr,cfg["t_det"],cfg["edges"],cfg["sigmas"],cfg["n_atoms"])
        d0,dq0=hidden_derivative(rr,wa,bb,cfg["sigmas"],cfg["t_det"],cfg["n_atoms"],cfg["nuisance_scale"])
        vals=[]
        for phi in fisher["phases"]:
            e=np.exp(1j*phi); mu=aa+np.real(qq*e); vals.append(profiled_information(d0+np.real(dq0*e),np.real(1j*qq*e),mu)[0])
        alpha_rows.append((alpha,float(np.sqrt(np.median(vals)))))
    alpha_min=next((x for x,s in alpha_rows if s>=1),None)

    krows=[]
    sigma_f=float(np.sqrt(cfg["sigmas"][0]**2+(cfg["t_det"]*cfg["sigmas"][2])**2))
    for k in cfg["kappas"]:
        rr=apply_shear(response,cfg["t_det"],k); _,_,aa,qq=cloud_projection(rr,cfg["t_det"],cfg["edges"],cfg["sigmas"],cfg["n_atoms"])
        oo,ff=orbit_diagnostics(aa,qq),poisson_phase_information(aa,qq)
        robust=ff["minimum"]>=cfg["min_phase_info"] and oo["axis_ratio"]>=cfg["min_axis_ratio"] and ff["min_median_ratio"]>=cfg["min_median_ratio"]
        krows.append((k,k*sigma_f/(2*np.pi),oo["axis_ratio"],ff["minimum"],ff["min_median_ratio"],ff["worst_phase_sigma"],robust))
    k_min=next((x[0] for x in krows if x[-1]),None)
    regime=classify_regime(gen["generator_violation"],hidden_snr,fisher,orbit,{"min_info":cfg["min_phase_info"],"axis_ratio":cfg["min_axis_ratio"],"min_median":cfg["min_median_ratio"]})

    # Required figures, compactly grouped per response.
    axes,_,qgrid=regular_grid(response); iy=len(axes[1])//2; ivy=len(axes[3])//2
    phase_slice=np.angle(qgrid[:,iy,:,ivy]); gx=np.abs(gen["Gq"][0][:,iy,:,ivy])
    X,V=np.meshgrid(axes[0]*1e3,axes[2]*1e3,indexing="ij")
    fig,ax=plt.subplots(1,2,figsize=(10,4)); ax[0].pcolormesh(X,V,phase_slice,shading="auto"); ax[0].set(title=f"{name} response phase",xlabel="x0 [mm]",ylabel="vx0 [mm/s]")
    for xf in np.linspace((axes[0]+cfg["t_det"]*axes[2]).min(),(axes[0]+cfg["t_det"]*axes[2]).max(),7): ax[0].plot(1e3*(xf-cfg["t_det"]*axes[2]),1e3*axes[2],"w-",lw=.5,alpha=.7)
    ax[1].pcolormesh(X,V,gx,shading="auto"); ax[1].set(title="|Gx q|",xlabel="x0 [mm]",ylabel="vx0 [mm/s]"); fig.tight_layout(); fig.savefig(out/f"{name}_response_geometry.png",dpi=160); plt.close(fig)
    fig,ax=plt.subplots(1,2,figsize=(10,4));
    for axi,k,title in [(ax[0],0,"original"),(ax[1],k_min or cfg["kappas"][-1],"added shear")]:
        rr=apply_shear(response,cfg["t_det"],k); _,_,aa,qq=cloud_projection(rr,cfg["t_det"],cfg["edges"],cfg["sigmas"],cfg["n_atoms"]); oo=orbit_diagnostics(aa,qq); u,s,v=np.linalg.svd(oo["orbit"]-oo["orbit"].mean(0),full_matrices=False); xy=(oo["orbit"]-oo["orbit"].mean(0))@v[:2].T; axi.plot(xy[:,0],xy[:,1]); axi.set(title=title,xlabel="PC1",ylabel="PC2",aspect="equal")
    fig.tight_layout(); fig.savefig(out/f"{name}_phase_orbits.png",dpi=160); plt.close(fig)
    fig,ax=plt.subplots(2,2,figsize=(10,7)); p=fisher["phases"]
    ax[0,0].plot(p,fisher["image"],label="image"); ax[0,0].plot(p,fisher["count"],label="count"); ax[0,0].plot(p,fisher["shape"],label="shape"); ax[0,0].legend(); ax[0,0].set(title="Phase Fisher information",xlabel="phase [rad]")
    ax[0,1].plot(p,prof); ax[0,1].set(title="Profiled hidden-mode information",xlabel="phase [rad]")
    ar=np.asarray(alpha_rows); ax[1,0].plot(ar[:,0],ar[:,1],"o-"); ax[1,0].axhline(1,color="k",ls="--"); ax[1,0].set(title="Fibre-breaking scan",xlabel="alpha",ylabel="hidden-mode SNR")
    kr=np.asarray([x[:-1] for x in krows],float); ax[1,1].plot(kr[:,1],kr[:,3],"o-",label="min information"); ax[1,1].plot(kr[:,1],kr[:,2],"o-",label="axis ratio"); ax[1,1].set(title="Shear scan",xlabel="fringes / final RMS"); ax[1,1].legend(); fig.tight_layout(); fig.savefig(out/f"{name}_operational_diagnostics.png",dpi=160); plt.close(fig)
    pd.DataFrame(alpha_rows,columns=["alpha","hidden_mode_snr"]).to_csv(out/f"{name}_alpha_scan.csv",index=False)
    pd.DataFrame(krows,columns=["kappa_rad_per_m","fringes_per_final_rms","axis_ratio","min_phase_info","min_median_ratio","worst_phase_sigma_rad","regime_iii"]).to_csv(out/f"{name}_kappa_scan.csv",index=False)
    return {"map":name,"regime":regime,"generator_violation":gen["generator_violation"],"within_fibre_fraction_q":within,"count_only_distance":intrinsic["count_only_distance"],"intrinsic_orbit_axis_ratio":intrinsic["axis_ratio"],"cloud_orbit_axis_ratio":orbit["axis_ratio"],"min_phase_information":fisher["minimum"],"median_phase_information":fisher["median"],"min_median_ratio":fisher["min_median_ratio"],"shape_information_fraction":fisher["shape_fraction"],"worst_phase_sigma_rad":fisher["worst_phase_sigma"],"median_profiled_hidden_info":float(np.median(prof)),"hidden_mode_snr":hidden_snr,"median_nuisance_phase_degeneracy":float(np.median(degeneracy)),"alpha_min_resolvable":alpha_min,"kappa_min_rad_per_m":k_min}

def main():
    p=argparse.ArgumentParser(); p.add_argument("--maps",nargs="*",type=Path,default=DEFAULT_MAPS); p.add_argument("--output",type=Path,default=REPO/"results/response_regime_diagnostics"); p.add_argument("--atoms",type=float,default=1e6); p.add_argument("--bins",type=int,default=24); args=p.parse_args(); args.output.mkdir(parents=True,exist_ok=True)
    t=3.8; sig=np.array([100e-6,100e-6,100e-6,100e-6]); sf=np.sqrt(sig[0]**2+(t*sig[2])**2); edges=np.linspace(-5*sf,5*sf,args.bins+1)
    cfg={"t_det":t,"state":0,"sigmas":sig,"n_atoms":args.atoms,"nuisance_scale":10e-6,"edges":edges,"alphas":np.linspace(0,4,17),"kappas":np.linspace(0,8*np.pi/sf,25),"min_phase_info":25.,"min_axis_ratio":.2,"min_median_ratio":.2}
    rows=[analyze(x,cfg,args.output) for x in args.maps]; pd.DataFrame(rows).to_csv(args.output/"summary.csv",index=False)
    with open(args.output/"summary.json","w") as f: json.dump({"configuration":{k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in cfg.items()},"maps":rows},f,indent=2)
    print(pd.DataFrame(rows).to_string(index=False))
if __name__=="__main__": main()
