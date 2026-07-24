"""
Hybrid analytic/finite-diff gradient of SemiAnalyticPixelACS.pixel_acs w.r.t. theta.

Expensive part (_eval_fast, the PSMAP lookup): analytic, via the validated
_eval_fast_grad. Cheap part (theta -> sample points -> final pixel stats):
finite differences, since none of it touches the GPU PSMAP call, so it's
nearly free regardless of how many theta components we perturb.
"""
import numpy as np
import cupy as cp


def pixel_acs_value(acs, theta):
    """Plain-value call, byte-identical to acs.pixel_acs(theta) -- used as
    ground truth for validation."""
    return acs.pixel_acs(theta)


def _sample_points(acs, theta):
    """theta -> (x0,y0,vx0,vy0) sample points (n_bins*n_v,) each, and P_b (n_bins,).
    Pure elementary arithmetic -- no PSMAP call. Mirrors pixel_acs's own logic
    exactly (kept in sync by construction: same formulas, just factored out)."""
    mu_x, mu_y, mu_vx, mu_vy = (float(theta[0]), float(theta[1]), float(theta[2]), float(theta[3]))
    sx, sy, svx, svy = (float(theta[4]), float(theta[5]), float(theta[6]), float(theta[7]))
    T = acs.t_det
    mu_xf = mu_x + T * mu_vx
    mu_yf = mu_y + T * mu_vy
    sxf = float(np.sqrt(sx**2 + (T * svx)**2))
    syf = float(np.sqrt(sy**2 + (T * svy)**2))
    ndtr = acs._ndtr
    P_x = ndtr((acs._x_hi - mu_xf) / sxf) - ndtr((acs._x_lo - mu_xf) / sxf)
    P_y = ndtr((acs._y_hi - mu_yf) / syf) - ndtr((acs._y_lo - mu_yf) / syf)
    P_b = cp.asarray(P_x * P_y, dtype=cp.float64)

    svx_c = float(svx * sx / sxf)
    svy_c = float(svy * sy / syf)
    mu_vx_cond = mu_vx + (T * svx**2 / sxf**2) * (acs._xc - mu_xf)
    mu_vy_cond = mu_vy + (T * svy**2 / syf**2) * (acs._yc - mu_yf)
    mu_vx_g = cp.asarray(mu_vx_cond, dtype=cp.float64)
    mu_vy_g = cp.asarray(mu_vy_cond, dtype=cp.float64)

    vx0 = (mu_vx_g[:, None] + svx_c * acs.z2_x[None, :]).ravel()
    vy0 = (mu_vy_g[:, None] + svy_c * acs.z2_y[None, :]).ravel()
    xcb = acs.xc_bins; ycb = acs.yc_bins
    x0 = (xcb[:, None] - T * (mu_vx_g[:, None] + svx_c * acs.z2_x[None, :])).ravel()
    y0 = (ycb[:, None] - T * (mu_vy_g[:, None] + svy_c * acs.z2_y[None, :])).ravel()
    return x0, y0, vx0, vy0, P_b


def _finish(acs, dphi, amp0, amp1, P_b, n_bins):
    """(dphi,amp0,amp1) at sample points -> final (A_g,Cc_g,Cs_g,A_e,Cc_e,Cs_e)
    per-bin arrays. Mirrors the tail of pixel_acs (GH sum + port sum + P_b)."""
    inter = acs._port_inter[None]
    A_per = amp0**2 + amp1**2
    Cc_per = inter * 2.0 * amp0 * amp1 * cp.cos(dphi)
    Cs_per = -inter * 2.0 * amp0 * amp1 * cp.sin(dphi)

    def _gh(v):
        return (v.reshape(n_bins, acs.n_v) * acs.w2[None, :]).sum(1)

    A_g  = _gh(A_per[:, acs.s0_g].sum(-1))
    Cc_g = _gh(Cc_per[:, acs.s0_g].sum(-1))
    Cs_g = _gh(Cs_per[:, acs.s0_g].sum(-1))
    A_e  = _gh(A_per[:, acs.s1_g].sum(-1))
    Cc_e = _gh(Cc_per[:, acs.s1_g].sum(-1))
    Cs_e = _gh(Cs_per[:, acs.s1_g].sum(-1))
    return (P_b * A_g, P_b * Cc_g, P_b * Cs_g, P_b * A_e, P_b * Cc_e, P_b * Cs_e)


def pixel_acs_and_grad(acs, theta, h_theta):
    """
    Returns (A_g,Cc_g,Cs_g,A_e,Cc_e,Cs_e) each (n_bins,), and grad, an (8,6,n_bins)
    array: d(each of the 6 outputs)/d(theta_k), computed via ONE expensive
    _eval_fast_grad call (analytic in x0,y0,vx0,vy0) chained with cheap
    finite differences for theta -> sample points and the GH/port-sum tail
    (neither of which touches the GPU PSMAP call).
    """
    n_bins = acs.n_bins
    x0, y0, vx0, vy0, P_b = _sample_points(acs, theta)
    dphi, amp0, amp1, sp_grads = acs._eval_fast_grad(x0, y0, vx0, vy0)
    base = _finish(acs, dphi, amp0, amp1, P_b, n_bins)

    # d(final output)/d(sample point coord) via chain rule through _finish,
    # using the analytic d(dphi,amp0,amp1)/d(coord) -- cheap elementwise ops,
    # not re-touching the PSMAP call.
    def finish_from_coord_grad(d_dphi, d_amp0, d_amp1):
        inter = acs._port_inter[None]
        dA_per = 2*amp0*d_amp0 + 2*amp1*d_amp1
        dCc_per = inter*2.0*(d_amp0*amp1*cp.cos(dphi) + amp0*d_amp1*cp.cos(dphi) - amp0*amp1*cp.sin(dphi)*d_dphi)
        dCs_per = -inter*2.0*(d_amp0*amp1*cp.sin(dphi) + amp0*d_amp1*cp.sin(dphi) + amp0*amp1*cp.cos(dphi)*d_dphi)

        def _gh(v):
            return (v.reshape(n_bins, acs.n_v) * acs.w2[None, :]).sum(1)
        dA_g  = _gh(dA_per[:, acs.s0_g].sum(-1)); dCc_g = _gh(dCc_per[:, acs.s0_g].sum(-1)); dCs_g = _gh(dCs_per[:, acs.s0_g].sum(-1))
        dA_e  = _gh(dA_per[:, acs.s1_g].sum(-1)); dCc_e = _gh(dCc_per[:, acs.s1_g].sum(-1)); dCs_e = _gh(dCs_per[:, acs.s1_g].sum(-1))
        return (P_b*dA_g, P_b*dCc_g, P_b*dCs_g, P_b*dA_e, P_b*dCc_e, P_b*dCs_e)

    d_x0, d_amp0_x0, d_amp1_x0 = sp_grads['x0']
    out_dx0 = finish_from_coord_grad(d_x0, d_amp0_x0, d_amp1_x0)
    d_y0, d_amp0_y0, d_amp1_y0 = sp_grads['y0']
    out_dy0 = finish_from_coord_grad(d_y0, d_amp0_y0, d_amp1_y0)
    d_vx0, d_amp0_vx0, d_amp1_vx0 = sp_grads['vx0']
    out_dvx0 = finish_from_coord_grad(d_vx0, d_amp0_vx0, d_amp1_vx0)
    d_vy0, d_amp0_vy0, d_amp1_vy0 = sp_grads['vy0']
    out_dvy0 = finish_from_coord_grad(d_vy0, d_amp0_vy0, d_amp1_vy0)

    # cheap finite-diff of theta -> (x0,y0,vx0,vy0), no PSMAP call
    grad = np.zeros((8, 6, n_bins))
    for k in range(8):
        tp = theta.copy(); tp[k] += h_theta[k]
        tm = theta.copy(); tm[k] -= h_theta[k]
        x0p, y0p, vx0p, vy0p, P_bp = _sample_points(acs, tp)
        x0m, y0m, vx0m, vy0m, P_bm = _sample_points(acs, tm)
        dx0_dtheta  = ((x0p - x0m) / (2*h_theta[k]))[:, None]
        dy0_dtheta  = ((y0p - y0m) / (2*h_theta[k]))[:, None]
        dvx0_dtheta = ((vx0p - vx0m) / (2*h_theta[k]))[:, None]
        dvy0_dtheta = ((vy0p - vy0m) / (2*h_theta[k]))[:, None]
        # also need d(base output)/dtheta_k via the P_b(theta) and direct sample-point
        # dependence -- since we don't have an analytic dP_b/dtheta here, get the
        # FULL d(output)/dtheta_k via total finite difference of the CHEAP _finish
        # tail evaluated with the ANALYTIC dphi/amp0/amp1 Taylor-shifted by the
        # sample-point perturbation (first-order), avoiding a second PSMAP call.
        # First-order Taylor: dphi(pt+delta) ~ dphi(pt) + grad . delta
        ddphi = d_x0*dx0_dtheta + d_y0*dy0_dtheta + d_vx0*dvx0_dtheta + d_vy0*dvy0_dtheta
        damp0 = d_amp0_x0*dx0_dtheta + d_amp0_y0*dy0_dtheta + d_amp0_vx0*dvx0_dtheta + d_amp0_vy0*dvy0_dtheta
        damp1 = d_amp1_x0*dx0_dtheta + d_amp1_y0*dy0_dtheta + d_amp1_vx0*dvx0_dtheta + d_amp1_vy0*dvy0_dtheta
        out_p = finish_from_coord_grad(ddphi, damp0, damp1)
        # plus the direct P_b(theta) dependence (P_b multiplies everything; use FD for P_b itself, cheap)
        dP_b_dtheta = (P_bp - P_bm) / (2*h_theta[k])
        for j in range(6):
            base_over_Pb = base[j] / cp.maximum(P_b, 1e-300)
            grad[k, j, :] = (out_p[j] + dP_b_dtheta * base_over_Pb).get()

    return base, grad
