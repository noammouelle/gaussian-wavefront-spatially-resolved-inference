#!/usr/bin/env python
"""
fit_surrogate_profiled_signal.py — Correct profiled pixel likelihood for science signal.

Architecture (from theory):
──────────────────────────────────────────────────────────────────────────────
Pre-computation  (once per shot × AI):
  1. Profile η̂_{iz} from the phi-AVERAGED sum image (n_g + n_e).
     The sum image is insensitive to φ_i (interference terms cancel), so the
     optimisation does not need a phase grid.
  2. Evaluate ACS = (A_g, Cc_g, Cs_g, A_e, Cc_e, Cs_e) = pixel_acs(η̂_{iz}).
     These are phi-independent: they encode how the pixel rates vary as
     λ(φ) = L × (A + Cc·cos φ + Cs·sin φ) for any absolute phase φ.

Per-evaluation  (every β = (As, Ac) call):
  1. δφ_i = As·sin(2πf·t_i) + Ac·cos(2πf·t_i)  for all shots.
  2. For each shot i and phase grid point k:
       Φ_{0,k} = φ_k             (Z0)
       Φ_{100,k} = φ_k + δφ_i   (Z100, rotation of ACS by δφ_i)
  3. logL_i(β, φ_k) = Σ_{z,s,b} [n·log λ(Φ) − λ(Φ)]   (Poisson)
  4. logL_i(β) = logsumexp_k[logL_i(β, φ_k)] − log K   (marginalise φ_i)
  5. logL(β) = Σ_i logL_i(β)                            (shots independent)

Signal fit: max_{As, Ac} logL(β)  — just 2 free parameters, smooth & fast.

The common laser phase φ_i is MARGINALISED (flat Uniform prior) — profiling
would absorb the signal into per-shot phase estimates.
Cloud nuisances η are PROFILED once (data sharply constrain them) with a
Gaussian prior from the dataset's shot-to-shot distribution.

Outputs
────────
  results/<stem>_surrogate_profiled_signal.jsonl  — per-run fit + truth
  results/surrogate_profile_tables/              — cached ACS tensors (.npz)
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path

import h5py
import numpy as np
from scipy.optimize import minimize

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "helpers"))
sys.path.insert(0, str(REPO / "python-scripts"))

from helpers import ImageShotDataset  # noqa: E402
from profile_cloud_nuisances import (  # noqa: E402
    SurrogatePixelACS, H5_KEYS, PARAM_NAMES, T_DET,
    _downsample, _HAS_GPU,
)

try:
    import cupy as cp
    from cupyx.scipy.special import logsumexp as _cp_lse
    _HAS_CUPY = True
except ImportError:
    cp = None
    _HAS_CUPY = False

from scipy.special import logsumexp as _np_lse

try:
    from aispy.psmap import load_psmap, PSMAPSurrogate
except ImportError:
    sys.path.insert(0, str(REPO.parents[1] / "local" / "aispy"))
    from aispy.psmap import load_psmap, PSMAPSurrogate


DEFAULT_RUN    = (
    REPO / "data"
    / "R80_N50_A1000000_muXStd10.0um_muVxStd10.0um_sigX100um_sigVx309um_"
      "sigXStd10.0um_sigVxStd10.0um_phi0random_sig_A0.100_f0.3000"
)
DEFAULT_PSMAP_Z0   = REPO / "output-files" / "PSGRID4D_CONFOCAL_Z0.h5"
DEFAULT_PSMAP_Z100 = REPO / "output-files" / "PSGRID4D_CONFOCAL_Z100.h5"


# ── Prior ──────────────────────────────────────────────────────────────────────

def _parse_prior(run_name: str) -> dict:
    """Extract per-shot theta prior parameters from the dataset name (SI units)."""
    def _um(pat): m = re.search(pat, run_name); return float(m.group(1))*1e-6 if m else None
    return {
        "mu_pos_std":     _um(r"muXStd([\d.]+)um")   or 10e-6,
        "mu_vel_std":     _um(r"muVxStd([\d.]+)um")  or 10e-6,
        "sigma_pos_mean": _um(r"sigX([\d.]+)um")      or 100e-6,
        "sigma_vel_mean": _um(r"sigVx([\d.]+)um")     or 309e-6,
        "sigma_pos_std":  _um(r"sigXStd([\d.]+)um")   or 10e-6,
        "sigma_vel_std":  _um(r"sigVxStd([\d.]+)um")  or 10e-6,
    }


def _log_prior(params, prior):
    """
    Gaussian log-prior on theta in log-sigma parameterisation.
    params[:4] = [mu_x0, mu_y0, mu_vx0, mu_vy0]  (linear)
    params[4:] = [log_sx0, log_sy0, log_svx0, log_svy0]  (log-space)
    """
    lp  = -0.5 * (params[0] / prior["mu_pos_std"])**2
    lp += -0.5 * (params[1] / prior["mu_pos_std"])**2
    lp += -0.5 * (params[2] / prior["mu_vel_std"])**2
    lp += -0.5 * (params[3] / prior["mu_vel_std"])**2
    sx0  = np.exp(params[4]); sy0  = np.exp(params[5])
    svx0 = np.exp(params[6]); svy0 = np.exp(params[7])
    lp += -0.5 * ((sx0  - prior["sigma_pos_mean"]) / prior["sigma_pos_std"])**2
    lp += -0.5 * ((sy0  - prior["sigma_pos_mean"]) / prior["sigma_pos_std"])**2
    lp += -0.5 * ((svx0 - prior["sigma_vel_mean"]) / prior["sigma_vel_std"])**2
    lp += -0.5 * ((svy0 - prior["sigma_vel_mean"]) / prior["sigma_vel_std"])**2
    return float(lp)


def _build_bounds_8(half_range, sigma_pos_max=5e-3, sigma_vel_max=5e-3,
                    com_vel_max=5e-3, sigma_pos_min=1e-6, sigma_vel_min=1e-6):
    return [
        (-half_range,           half_range),
        (-half_range,           half_range),
        (-com_vel_max,          com_vel_max),
        (-com_vel_max,          com_vel_max),
        (np.log(sigma_pos_min), np.log(sigma_pos_max)),
        (np.log(sigma_pos_min), np.log(sigma_pos_max)),
        (np.log(sigma_vel_min), np.log(sigma_vel_max)),
        (np.log(sigma_vel_min), np.log(sigma_vel_max)),
    ]


# ── Step 1: profile η̂ from joint marginal likelihood ──────────────────────────

def _profile_eta(evaluator, n_g, n_e, phi_grid, init_theta, bounds, prior,
                 maxiter, xp, lse_fn):
    """
    MAP estimate of η for one shot, marginalising over φ.

    Objective: MAP(η) = logsumexp_k[logP(n_g|λ_g(η,φ_k)) + logP(n_e|λ_e(η,φ_k))]
                        - log K  +  log_prior(η)
    """
    n_g_xp  = xp.asarray(n_g.astype(np.float64))
    n_e_xp  = xp.asarray(n_e.astype(np.float64))
    K       = len(phi_grid)
    log_K   = float(np.log(K))
    c_phi   = xp.asarray(np.cos(phi_grid))
    s_phi   = xp.asarray(np.sin(phi_grid))
    n_total = float((n_g + n_e).sum())

    def objective(params):
        theta = np.r_[params[:4], np.exp(params[4:])]
        if np.any(theta[4:] <= 0):
            return 1e300
        try:
            acs = evaluator.pixel_acs(theta)
        except Exception:
            return 1e300
        A_g, Cc_g, Cs_g, A_e, Cc_e, Cs_e = acs
        tot = float((A_g + A_e).sum())
        if tot <= 0:
            return 1e300
        L = n_total / tot

        ll = xp.zeros(K, dtype=xp.float64)
        for n_s, A_s, Cc_s, Cs_s in [
            (n_g_xp, A_g, Cc_g, Cs_g),
            (n_e_xp, A_e, Cc_e, Cs_e),
        ]:
            lam = L * xp.maximum(
                A_s[None] + Cc_s[None] * c_phi[:, None] + Cs_s[None] * s_phi[:, None],
                1e-300,
            )
            ll += (n_s[None] * xp.log(lam) - lam).sum(1)

        val = float(lse_fn(ll) - log_K) + _log_prior(params, prior)
        return -val

    p0  = np.r_[init_theta[:4], np.log(np.maximum(init_theta[4:], 1e-12))]
    res = minimize(objective, p0, method="L-BFGS-B", bounds=bounds,
                   options={"maxiter": maxiter, "ftol": 1e-10, "gtol": 1e-8, "maxls": 20})
    theta_hat = np.r_[res.x[:4], np.exp(res.x[4:])]
    return theta_hat, res


# ── Step 2: ACS at η̂ ──────────────────────────────────────────────────────────

def _acs_at_eta(evaluator, theta_hat, n_g, n_e, xp):
    """
    Compute ACS at η̂ and scale factor L.

    Returns dict with A_g, Cc_g, Cs_g, A_e, Cc_e, Cs_e (GPU arrays),
    L (scalar), n_g, n_e (GPU arrays).
    """
    acs      = evaluator.pixel_acs(theta_hat)
    A_g, Cc_g, Cs_g, A_e, Cc_e, Cs_e = acs
    A_plus   = A_g + A_e
    tot      = float(A_plus.sum())
    L        = float((n_g + n_e).sum()) / max(tot, 1e-300)
    return {
        "A_g":  A_g,  "Cc_g": Cc_g, "Cs_g": Cs_g,
        "A_e":  A_e,  "Cc_e": Cc_e, "Cs_e": Cs_e,
        "L":    L,
        "n_g":  xp.asarray(n_g.astype(np.float64)),
        "n_e":  xp.asarray(n_e.astype(np.float64)),
    }


# ── Step 3: marginal logL batch over all shots ─────────────────────────────────

def _marginal_logL(As, Ac, precomp_z0, precomp_z100, phi_grid,
                   t_shots, frequency, xp, lse_fn):
    """
    Full marginal log-likelihood over all shots.

    logL(β) = Σ_i  logsumexp_k [ logL_i(β, φ_k) ] − log K

    where logL_i(β, φ_k) = ll_Z0(φ_k) + ll_Z100(φ_k + δφ_i)
    and   ll_ZX(Φ) = Σ_{s,b} [n_{sb}·log λ_{sb}(Φ) − λ_{sb}(Φ)]

    Z100 ACS is rotated by δφ_i using the identity:
      cos(φ_k + δφ_i) = cos(φ_k)·cos(δφ_i) − sin(φ_k)·sin(δφ_i)
    so Cc_eff = Cc·cos(δφ_i) + Cs·sin(δφ_i)  (rotated cosine coeff)
       Cs_eff = −Cc·sin(δφ_i) + Cs·cos(δφ_i)
    """
    n_shots = len(t_shots)
    K       = len(phi_grid)
    log_K   = float(np.log(K))

    delta_phis = As * np.sin(2*np.pi*frequency*t_shots) + Ac * np.cos(2*np.pi*frequency*t_shots)
    phi_grid_xp = xp.asarray(phi_grid)
    c_phi  = xp.cos(phi_grid_xp)  # (K,)
    s_phi  = xp.sin(phi_grid_xp)

    total_ll = 0.0
    for i in range(n_shots):
        z0  = precomp_z0[i]
        z1  = precomp_z100[i]
        cd  = float(np.cos(delta_phis[i]))
        sd  = float(np.sin(delta_phis[i]))

        # Z0: ll at φ_k for all k  →  (K,)
        ll = xp.zeros(K, dtype=xp.float64)
        for (n_s, A_s, Cc_s, Cs_s, L_s) in [
            (z0["n_g"], z0["A_g"], z0["Cc_g"], z0["Cs_g"], z0["L"]),
            (z0["n_e"], z0["A_e"], z0["Cc_e"], z0["Cs_e"], z0["L"]),
        ]:
            lam = L_s * xp.maximum(A_s[None] + Cc_s[None]*c_phi[:, None] + Cs_s[None]*s_phi[:, None], 1e-300)
            ll += (n_s[None] * xp.log(lam) - lam).sum(1)

        # Z100: rotate ACS by δφ_i then compute ll at φ_k
        Cc_g1r = z1["Cc_g"]*cd + z1["Cs_g"]*sd
        Cs_g1r = -z1["Cc_g"]*sd + z1["Cs_g"]*cd
        Cc_e1r = z1["Cc_e"]*cd + z1["Cs_e"]*sd
        Cs_e1r = -z1["Cc_e"]*sd + z1["Cs_e"]*cd
        for (n_s, A_s, Cc_r, Cs_r, L_s) in [
            (z1["n_g"], z1["A_g"], Cc_g1r, Cs_g1r, z1["L"]),
            (z1["n_e"], z1["A_e"], Cc_e1r, Cs_e1r, z1["L"]),
        ]:
            lam = L_s * xp.maximum(A_s[None] + Cc_r[None]*c_phi[:, None] + Cs_r[None]*s_phi[:, None], 1e-300)
            ll += (n_s[None] * xp.log(lam) - lam).sum(1)

        total_ll += float(lse_fn(ll) - log_K)

    return total_ll


# ── Step 4: signal fit ─────────────────────────────────────────────────────────

def _fit_beta(precomp_z0, precomp_z100, phi_grid, t_shots, frequency,
              n_starts, amp_starts, maxiter, xp, lse_fn):
    """
    Maximise the 2-parameter marginal logL over β = (As, Ac).

    Multi-start L-BFGS-B with starts distributed over a range of amplitudes.
    As ≥ 0 breaks the (As, Ac) ↔ (−As, −Ac) degeneracy.
    """
    def objective(params):
        As, Ac = params
        val = _marginal_logL(As, Ac, precomp_z0, precomp_z100, phi_grid,
                             t_shots, frequency, xp, lse_fn)
        return -val if np.isfinite(val) else 1e300

    best = None
    for amp in amp_starts:
        for phase_start in np.linspace(-np.pi, np.pi, n_starts, endpoint=False):
            As0 = float(amp * np.cos(phase_start))
            Ac0 = float(amp * np.sin(phase_start))
            # As must be ≥ 0; reflect if needed
            if As0 < 0:
                As0, Ac0 = -As0, -Ac0
            res = minimize(
                objective, [As0, Ac0], method="L-BFGS-B",
                bounds=[(0.0, np.pi), (-np.pi, np.pi)],
                options={"maxiter": maxiter, "ftol": 1e-9, "gtol": 1e-6},
            )
            if best is None or res.fun < best.fun:
                best = res

    As_h, Ac_h = best.x
    return {
        "As_hat":    float(As_h),
        "Ac_hat":    float(Ac_h),
        "amp_hat":   float(np.hypot(As_h, Ac_h)),
        "phase_hat": float(np.arctan2(Ac_h, As_h)),
        "logL":      -float(best.fun),
        "converged": bool(best.success),
        "nit":       int(best.nit),
        "nfev":      int(best.nfev),
        "message":   str(best.message),
    }


# ── ACS cache ─────────────────────────────────────────────────────────────────

def _acs_cache_path(cache_dir, run_stem, run_idx, site, bins, n_quad, ntheta, max_shots):
    tag = f"{run_stem}_run{run_idx:03d}_{site}_b{bins}_q{n_quad}_k{ntheta}"
    if max_shots is not None:
        tag += f"_n{max_shots}"
    return cache_dir / f"{tag}_acs.npz"


def _build_or_load_acs(args, run_idx, site, evaluator, ds, bins, prior,
                       init_from_true, phi_grid, xp, lse_fn):
    """
    Profile η̂ for every shot (joint marginal over φ) and compute ACS tensors.
    Returns list of n_shots dicts, loads/saves NPZ cache.
    """
    cache = _acs_cache_path(
        args.profile_dir, Path(args.run_name).name,
        run_idx, site, bins, args.n_quad, args.ntheta, args.max_shots,
    )
    n_shots = ds.n_shots
    if args.max_shots is not None:
        n_shots = min(n_shots, args.max_shots)

    if cache.exists() and not args.rebuild_profiles:
        logging.info("  [%s] loading cached ACS %s", site, cache.name)
        d = np.load(cache)
        precomp = []
        for i in range(n_shots):
            entry = {k: xp.asarray(d[f"{k}_{i}"]) if k != "L" else float(d[f"L_{i}"])
                     for k in ("A_g", "Cc_g", "Cs_g", "A_e", "Cc_e", "Cs_e", "L", "n_g", "n_e")}
            if f"theta_hat_{i}" in d:
                entry["theta_hat"] = d[f"theta_hat_{i}"]
            precomp.append(entry)
        return precomp

    bounds = _build_bounds_8(
        ds.half_range,
        sigma_pos_max=args.sigma_pos_max,
        sigma_vel_max=args.sigma_vel_max,
        com_vel_max=args.com_vel_max,
    )
    prior_mean_theta = np.array([
        0.0, 0.0, 0.0, 0.0,
        prior["sigma_pos_mean"], prior["sigma_pos_mean"],
        prior["sigma_vel_mean"], prior["sigma_vel_mean"],
    ])

    logging.info("  [%s] profiling η̂ for %d shots (joint marginal over φ, K=%d) ...",
                 site, n_shots, len(phi_grid))
    t0      = time.perf_counter()
    precomp = []
    save_arrays = {}

    for shot in range(n_shots):
        img  = _downsample(ds[shot], bins)   # (2, bins, bins)
        n_g  = img[0].ravel(); n_e = img[1].ravel()

        # Initialisation
        if init_from_true:
            meta       = ds.meta(shot)
            init_theta = np.array([meta[H5_KEYS[nm]] for nm in PARAM_NAMES])
        else:
            init_theta = prior_mean_theta.copy()

        theta_hat, _res = _profile_eta(
            evaluator, n_g, n_e, phi_grid, init_theta, bounds, prior,
            args.profile_maxiter, xp, lse_fn)

        d = _acs_at_eta(evaluator, theta_hat, n_g, n_e, xp)
        d["theta_hat"] = theta_hat
        precomp.append(d)

        # Save to dict for NPZ (CPU numpy)
        for k in ("A_g", "Cc_g", "Cs_g", "A_e", "Cc_e", "Cs_e", "n_g", "n_e"):
            v = d[k]
            save_arrays[f"{k}_{shot}"] = cp.asnumpy(v) if _HAS_CUPY and isinstance(v, cp.ndarray) else np.asarray(v)
        save_arrays[f"L_{shot}"]         = np.array(d["L"])
        save_arrays[f"theta_hat_{shot}"] = theta_hat

        if (shot + 1) % 10 == 0 or shot + 1 == n_shots:
            logging.info("    %d/%d shots | %.1fs", shot + 1, n_shots, time.perf_counter() - t0)

    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **save_arrays)
    logging.info("  [%s] cached -> %s", site, cache.name)
    return precomp


# ── Per-run pipeline ───────────────────────────────────────────────────────────

def _run_one(run_path, run_idx, eval_z0, eval_z100, phi_grid, prior, args, xp, lse_fn):
    run_dir = run_path / f"run_{run_idx:03d}"
    ds_z0   = ImageShotDataset(str(run_dir / "Z0"   / "data_IMG.h5"))
    ds_z100 = ImageShotDataset(str(run_dir / "Z100" / "data_IMG.h5"))

    n_shots = ds_z0.n_shots
    if args.max_shots is not None:
        n_shots = min(n_shots, args.max_shots)

    # True signal from attrs (delta_phi[i] = amp*sin(2π*freq*i + phase))
    # As = amp*cos(phase), Ac = amp*sin(phase) in model As*sin + Ac*cos
    with h5py.File(run_dir / "Z100" / "data_IMG.h5") as f:
        sig_amp   = float(f.attrs.get("signal_amp",   np.nan))
        sig_freq  = float(f.attrs.get("signal_freq",  np.nan))
        sig_phase = float(f.attrs.get("signal_phase", 0.0))

    t_shots = np.arange(n_shots, dtype=float)

    As_true    = sig_amp * np.cos(sig_phase)
    Ac_true    = sig_amp * np.sin(sig_phase)
    amp_true   = float(sig_amp)
    phase_true = float(sig_phase)

    # Profile η̂ and precompute ACS (checkpointed)
    precomp_z0   = _build_or_load_acs(args, run_idx, "Z0",   eval_z0,   ds_z0,
                                       args.bins, prior, args.init_from_true,
                                       phi_grid, xp, lse_fn)
    precomp_z100 = _build_or_load_acs(args, run_idx, "Z100", eval_z100, ds_z100,
                                       args.bins, prior, args.init_from_true,
                                       phi_grid, xp, lse_fn)
    precomp_z0   = precomp_z0[:n_shots]
    precomp_z100 = precomp_z100[:n_shots]

    # Cloud-param RMSE (if theta_hat stored and init_from_true so true theta is available)
    eta_rmse = {}
    if all("theta_hat" in d for d in precomp_z0) and all("theta_hat" in d for d in precomp_z100):
        for site_label, ds_site, precomp_site in [("z0", ds_z0, precomp_z0),
                                                   ("z100", ds_z100, precomp_z100)]:
            errs = {nm: [] for nm in PARAM_NAMES}
            for shot, d in enumerate(precomp_site):
                meta = ds_site.meta(shot)
                true_theta = np.array([meta[H5_KEYS[nm]] for nm in PARAM_NAMES])
                for j, nm in enumerate(PARAM_NAMES):
                    errs[nm].append(d["theta_hat"][j] - true_theta[j])
            for nm in PARAM_NAMES:
                key = f"{nm}_{site_label}"
                eta_rmse[key + "_rmse"] = float(np.sqrt(np.mean(np.array(errs[nm])**2)))
                eta_rmse[key + "_bias"] = float(np.mean(errs[nm]))
        # Log summary
        for nm in PARAM_NAMES:
            scale = 1e6  # m or m/s -> µm or µm/s
            logging.info("  eta %s  z0  RMSE=%.2fµm  z100  RMSE=%.2fµm",
                         nm,
                         eta_rmse[f"{nm}_z0_rmse"] * scale,
                         eta_rmse[f"{nm}_z100_rmse"] * scale)

    # Fit β = (As, Ac)
    logging.info("Fitting β over %d shots ...", n_shots)
    t0  = time.perf_counter()
    fit = _fit_beta(
        precomp_z0, precomp_z100, phi_grid, t_shots, sig_freq,
        n_starts=args.signal_starts, amp_starts=args.amp_starts,
        maxiter=args.signal_maxiter, xp=xp, lse_fn=lse_fn,
    )
    logging.info("  done in %.1fs  converged=%s", time.perf_counter() - t0, fit["converged"])

    return {
        "run_idx":   run_idx,
        "n_shots":   n_shots,
        "signal_amp_true": sig_amp,
        "signal_freq":     sig_freq,
        "As_true":   float(As_true),
        "Ac_true":   float(Ac_true),
        "amp_true":  amp_true,
        "phase_true": phase_true,
        **fit,
        "As_err":    fit["As_hat"]    - float(As_true),
        "Ac_err":    fit["Ac_hat"]    - float(Ac_true),
        "amp_err":   fit["amp_hat"]   - amp_true,
        "phase_err": fit["phase_hat"] - phase_true,
        **eta_rmse,
    }


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_name", nargs="?", type=Path, default=DEFAULT_RUN)
    p.add_argument("--run-start", type=int, default=0)
    p.add_argument("--run-stop",  type=int, default=None)
    p.add_argument("--run-idx",   type=int, default=None,
                   help="Single run (overrides --run-start/stop).")
    p.add_argument("--psmap-z0",   type=Path, default=DEFAULT_PSMAP_Z0)
    p.add_argument("--psmap-z100", type=Path, default=DEFAULT_PSMAP_Z100)
    # Grid
    p.add_argument("--bins",      type=int, default=32)
    p.add_argument("--n-quad",    type=int, default=20_000)
    p.add_argument("--ntheta",    type=int, default=512,
                   help="Phase grid size K for φ_i marginalisation. "
                        "Rule of thumb: K ≥ 2√N_detected.")
    # Initialisation
    p.add_argument("--init-from-true", action="store_true",
                   help="Start η̂ profiling from the true theta (oracle). "
                        "If not set, uses prior mean.")
    # Prior overrides (SI units)
    p.add_argument("--mu-pos-std",     type=float, default=None)
    p.add_argument("--mu-vel-std",     type=float, default=None)
    p.add_argument("--sigma-pos-mean", type=float, default=None)
    p.add_argument("--sigma-vel-mean", type=float, default=None)
    p.add_argument("--sigma-pos-std",  type=float, default=None)
    p.add_argument("--sigma-vel-std",  type=float, default=None)
    # Bounds
    p.add_argument("--sigma-pos-max", type=float, default=5e-3)
    p.add_argument("--sigma-vel-max", type=float, default=5e-3)
    p.add_argument("--com-vel-max",   type=float, default=5e-3)
    # η profiling
    p.add_argument("--profile-maxiter", type=int, default=200)
    # Signal fit
    p.add_argument("--frequency",      type=float, default=0.3)
    p.add_argument("--signal-starts",  type=int,   default=6,
                   help="# phase starts in the multi-start signal optimiser.")
    p.add_argument("--amp-starts",     type=float, nargs="+",
                   default=[0.0, 0.03, 0.07, 0.10, 0.15],
                   help="Amplitude starting points for multi-start.")
    p.add_argument("--signal-maxiter", type=int,   default=500)
    # I/O
    p.add_argument("--max-shots",        type=int,  default=None)
    p.add_argument("--profile-dir",      type=Path,
                   default=REPO / "results" / "surrogate_profile_tables")
    p.add_argument("--output",           type=Path, default=None)
    p.add_argument("--rebuild-profiles", action="store_true")
    p.add_argument("--no-gpu",           action="store_true")
    p.add_argument("-v", "--verbose",    action="store_true")
    return p.parse_args()


def _setup_logging(verbose):
    (REPO / "logs").mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler()],
    )


def _infer_run_count(run_path):
    m = re.match(r"R(\d+)", run_path.name)
    if m: return int(m.group(1))
    dirs = sorted(run_path.glob("run_[0-9][0-9][0-9]"))
    return max(int(d.name.removeprefix("run_")) for d in dirs) + 1 if dirs else 1


def main():
    args = parse_args()
    _setup_logging(args.verbose)

    use_gpu = _HAS_GPU and _HAS_CUPY and not args.no_gpu
    xp      = cp if use_gpu else np
    lse_fn  = _cp_lse if use_gpu else _np_lse

    run_path = Path(args.run_name).expanduser().resolve()
    run_stem = run_path.name

    # Build prior (auto-parsed + CLI overrides)
    prior = _parse_prior(run_stem)
    for key, val in [("mu_pos_std",     args.mu_pos_std),
                     ("mu_vel_std",     args.mu_vel_std),
                     ("sigma_pos_mean", args.sigma_pos_mean),
                     ("sigma_vel_mean", args.sigma_vel_mean),
                     ("sigma_pos_std",  args.sigma_pos_std),
                     ("sigma_vel_std",  args.sigma_vel_std)]:
        if val is not None:
            prior[key] = val

    logging.info("Run: %s", run_stem)
    logging.info("GPU=%s  bins=%d  ntheta=%d  n_quad=%d", use_gpu, args.bins, args.ntheta, args.n_quad)
    logging.info("Prior: mu_pos=±%.1fµm  mu_vel=±%.1fµm/s  "
                 "sigma_pos=%.0f±%.0fµm  sigma_vel=%.0f±%.0fµm/s",
                 prior["mu_pos_std"]*1e6, prior["mu_vel_std"]*1e6,
                 prior["sigma_pos_mean"]*1e6, prior["sigma_pos_std"]*1e6,
                 prior["sigma_vel_mean"]*1e6, prior["sigma_vel_std"]*1e6)
    logging.info("Init: %s", "oracle (true theta)" if args.init_from_true else "prior mean")

    # Build evaluators (edges from run_000)
    _ds_tmp = ImageShotDataset(str(run_path / "run_000" / "Z0" / "data_IMG.h5"))
    _edges  = np.linspace(-_ds_tmp.half_range, _ds_tmp.half_range, args.bins + 1)
    del _ds_tmp

    logging.info("Loading PSMAPs ...")
    sur_z0   = PSMAPSurrogate(load_psmap(str(args.psmap_z0)),   T_DET, use_gpu=use_gpu)
    sur_z100 = PSMAPSurrogate(load_psmap(str(args.psmap_z100)), T_DET, use_gpu=use_gpu)
    eval_z0   = SurrogatePixelACS(sur_z0,   T_DET, _edges, _edges, n_quad=args.n_quad)
    eval_z100 = SurrogatePixelACS(sur_z100, T_DET, _edges, _edges, n_quad=args.n_quad)
    logging.info("Evaluators ready  n_bins=%d", eval_z0.n_bins)

    phi_grid = np.linspace(0.0, 2.0*np.pi, args.ntheta, endpoint=False)

    # Run range
    if args.run_idx is not None:
        run_ids = [args.run_idx]
    else:
        n_runs   = _infer_run_count(run_path)
        run_stop = n_runs if args.run_stop is None else args.run_stop
        run_ids  = list(range(args.run_start, run_stop))

    args.profile_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output or (REPO / "results" / f"{run_stem}_surrogate_profiled_signal.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    logging.info("Processing %d run(s): %s", len(run_ids), run_ids)

    results = []
    for run_idx in run_ids:
        logging.info("=== run_%03d ===", run_idx)
        t_run = time.perf_counter()
        result = _run_one(
            run_path, run_idx, eval_z0, eval_z100,
            phi_grid, prior, args, xp, lse_fn,
        )
        result["elapsed_s"] = round(time.perf_counter() - t_run, 1)
        results.append(result)

        logging.info(
            "run_%03d | amp: true=%.4f hat=%.4f err=%+.4f | "
            "phase: true=%.4f hat=%.4f err=%+.4f | "
            "logL=%.1f converged=%s | %.0fs",
            run_idx,
            result["amp_true"],   result["amp_hat"],   result["amp_err"],
            result["phase_true"], result["phase_hat"], result["phase_err"],
            result["logL"], result["converged"], result["elapsed_s"],
        )

    with out_path.open("w") as fh:
        for r in results:
            fh.write(json.dumps(r) + "\n")
    logging.info("Wrote %d rows -> %s", len(results), out_path)

    if len(results) > 1:
        def _rmse(k): return float(np.sqrt(np.mean([r[k]**2 for r in results])))
        def _bias(k): return float(np.mean([r[k] for r in results]))
        logging.info("=== Summary over %d runs ===", len(results))
        for k in ("amp_err", "phase_err", "As_err", "Ac_err"):
            logging.info("  %-12s  RMSE=%.5f  bias=%+.5f", k, _rmse(k), _bias(k))


if __name__ == "__main__":
    main()
