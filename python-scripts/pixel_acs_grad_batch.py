"""
Batched-across-shots version of pixel_acs_grad.py. Concatenates all N shots'
sample points into ONE _eval_fast_grad call instead of N sequential ones,
amortizing GPU launch/dispatch overhead.

All shapes explicit in comments -- this is the highest-risk part of the
session's speedup work, validated step by step against the already-correct
per-shot loop before trusting it.
"""
import numpy as np
import cupy as cp


def _sample_points_batch(acs, theta_batch):
    """theta_batch: (N,8) numpy -> x0,y0,vx0,vy0 each (N, n_bins*n_v) cupy flat,
    P_b (N, n_bins) cupy."""
    N = theta_batch.shape[0]
    mu_x, mu_y, mu_vx, mu_vy = (theta_batch[:,0], theta_batch[:,1], theta_batch[:,2], theta_batch[:,3])
    sx, sy, svx, svy = (theta_batch[:,4], theta_batch[:,5], theta_batch[:,6], theta_batch[:,7])
    T = acs.t_det
    mu_xf = mu_x + T*mu_vx          # (N,)
    mu_yf = mu_y + T*mu_vy
    sxf = np.sqrt(sx**2 + (T*svx)**2)   # (N,)
    syf = np.sqrt(sy**2 + (T*svy)**2)

    ndtr = acs._ndtr
    x_hi = acs._x_hi[None,:]; x_lo = acs._x_lo[None,:]   # (1, n_bins) CPU
    y_hi = acs._y_hi[None,:]; y_lo = acs._y_lo[None,:]
    P_x = ndtr((x_hi - mu_xf[:,None]) / sxf[:,None]) - ndtr((x_lo - mu_xf[:,None]) / sxf[:,None])  # (N, n_bins)
    P_y = ndtr((y_hi - mu_yf[:,None]) / syf[:,None]) - ndtr((y_lo - mu_yf[:,None]) / syf[:,None])
    P_b = cp.asarray(P_x * P_y, dtype=cp.float64)   # (N, n_bins)

    svx_c = svx * sx / sxf   # (N,)
    svy_c = svy * sy / syf
    xc = acs._xc[None,:]; yc = acs._yc[None,:]   # (1, n_bins) CPU (property is stored on GPU as xc_bins; use CPU copy consistently)
    mu_vx_cond = mu_vx[:,None] + (T*svx**2/sxf**2)[:,None] * (xc - mu_xf[:,None])   # (N, n_bins)
    mu_vy_cond = mu_vy[:,None] + (T*svy**2/syf**2)[:,None] * (yc - mu_yf[:,None])
    mu_vx_g = cp.asarray(mu_vx_cond, dtype=cp.float64)   # (N, n_bins)
    mu_vy_g = cp.asarray(mu_vy_cond, dtype=cp.float64)
    svx_c_g = cp.asarray(svx_c, dtype=cp.float64)   # (N,)
    svy_c_g = cp.asarray(svy_c, dtype=cp.float64)

    n_bins = acs.n_bins; n_v = acs.n_v
    xcb = acs.xc_bins[None,:]; ycb = acs.yc_bins[None,:]   # (1, n_bins) GPU

    # (N, n_bins, n_v)
    vx0_3 = mu_vx_g[:,:,None] + svx_c_g[:,None,None] * acs.z2_x[None,None,:]
    vy0_3 = mu_vy_g[:,:,None] + svy_c_g[:,None,None] * acs.z2_y[None,None,:]
    x0_3  = xcb[:,:,None] - T * vx0_3
    y0_3  = ycb[:,:,None] - T * vy0_3

    x0  = x0_3.reshape(N, n_bins*n_v)
    y0  = y0_3.reshape(N, n_bins*n_v)
    vx0 = vx0_3.reshape(N, n_bins*n_v)
    vy0 = vy0_3.reshape(N, n_bins*n_v)
    return x0, y0, vx0, vy0, P_b


def pixel_acs_value_batch(acs, theta_batch):
    """theta_batch (N,8) -> 6-tuple of (N, n_bins) arrays. Ground-truth-equivalent
    to looping pixel_acs_value per shot; validated against that loop."""
    N = theta_batch.shape[0]
    n_bins = acs.n_bins; n_v = acs.n_v
    x0, y0, vx0, vy0, P_b = _sample_points_batch(acs, theta_batch)   # each (N, n_bins*n_v) or (N,n_bins)

    dphi, amp0, amp1 = acs._eval_fast(x0.ravel(), y0.ravel(), vx0.ravel(), vy0.ravel())
    # dphi etc: (N*n_bins*n_v, nP)
    nP = dphi.shape[1]
    dphi = dphi.reshape(N, n_bins*n_v, nP)
    amp0 = amp0.reshape(N, n_bins*n_v, nP)
    amp1 = amp1.reshape(N, n_bins*n_v, nP)

    inter = acs._port_inter[None, None, :]   # (1,1,nP)
    A_per = amp0**2 + amp1**2
    Cc_per = inter * 2.0 * amp0 * amp1 * cp.cos(dphi)
    Cs_per = -inter * 2.0 * amp0 * amp1 * cp.sin(dphi)

    w2 = acs.w2[None, None, :]   # (1,1,n_v) -- will broadcast after reshape to (N,n_bins,n_v)
    def _gh(v_over_ports):   # v_over_ports: (N, n_bins*n_v)
        v3 = v_over_ports.reshape(N, n_bins, n_v)
        return (v3 * acs.w2[None, None, :]).sum(-1)   # (N, n_bins)

    s0 = acs.s0_g; s1 = acs.s1_g
    A_g  = _gh(A_per[:,:,s0].sum(-1));  Cc_g = _gh(Cc_per[:,:,s0].sum(-1));  Cs_g = _gh(Cs_per[:,:,s0].sum(-1))
    A_e  = _gh(A_per[:,:,s1].sum(-1));  Cc_e = _gh(Cc_per[:,:,s1].sum(-1));  Cs_e = _gh(Cs_per[:,:,s1].sum(-1))

    return (P_b*A_g, P_b*Cc_g, P_b*Cs_g, P_b*A_e, P_b*Cc_e, P_b*Cs_e)


def _finish_batch(acs, dphi, amp0, amp1, P_b, N, n_bins):
    """dphi,amp0,amp1: (N, n_bins*n_v, nP). P_b: (N, n_bins).
    -> 6-tuple of (N, n_bins)."""
    n_v = acs.n_v
    inter = acs._port_inter[None, None, :]
    A_per = amp0**2 + amp1**2
    Cc_per = inter * 2.0 * amp0 * amp1 * cp.cos(dphi)
    Cs_per = -inter * 2.0 * amp0 * amp1 * cp.sin(dphi)

    s0 = acs.s0_g; s1 = acs.s1_g
    def _gh(v_over_ports):
        v3 = v_over_ports.reshape(N, n_bins, n_v)
        return (v3 * acs.w2[None, None, :]).sum(-1)

    A_g  = _gh(A_per[:,:,s0].sum(-1));  Cc_g = _gh(Cc_per[:,:,s0].sum(-1));  Cs_g = _gh(Cs_per[:,:,s0].sum(-1))
    A_e  = _gh(A_per[:,:,s1].sum(-1));  Cc_e = _gh(Cc_per[:,:,s1].sum(-1));  Cs_e = _gh(Cs_per[:,:,s1].sum(-1))
    return (P_b*A_g, P_b*Cc_g, P_b*Cs_g, P_b*A_e, P_b*Cc_e, P_b*Cs_e)


def pixel_acs_and_grad_batch(acs, theta_batch, h_theta):
    """
    theta_batch: (N,8) numpy. Returns:
      base: 6-tuple of (N, n_bins) cupy arrays
      grad: (N, 8, 6, n_bins) numpy array -- d(each of 6 outputs)/d(theta_k), per shot.

    Same hybrid analytic(_eval_fast_grad)/cheap-FD structure as the single-shot
    pixel_acs_and_grad, batched across shots into ONE _eval_fast_grad call.
    """
    N = theta_batch.shape[0]
    n_bins = acs.n_bins; n_v = acs.n_v
    x0, y0, vx0, vy0, P_b = _sample_points_batch(acs, theta_batch)   # (N, n_bins*n_v) / (N,n_bins)

    dphi, amp0, amp1, sp_grads = acs._eval_fast_grad(x0.ravel(), y0.ravel(), vx0.ravel(), vy0.ravel())
    nP = dphi.shape[1]
    npts = n_bins * n_v
    dphi = dphi.reshape(N, npts, nP); amp0 = amp0.reshape(N, npts, nP); amp1 = amp1.reshape(N, npts, nP)
    sp_grads = {k: tuple(a.reshape(N, npts, nP) for a in v) for k, v in sp_grads.items()}

    base = _finish_batch(acs, dphi, amp0, amp1, P_b, N, n_bins)

    def finish_from_coord_grad(d_dphi, d_amp0, d_amp1):
        inter = acs._port_inter[None, None, :]
        dA_per = 2*amp0*d_amp0 + 2*amp1*d_amp1
        dCc_per = inter*2.0*(d_amp0*amp1*cp.cos(dphi) + amp0*d_amp1*cp.cos(dphi) - amp0*amp1*cp.sin(dphi)*d_dphi)
        dCs_per = -inter*2.0*(d_amp0*amp1*cp.sin(dphi) + amp0*d_amp1*cp.sin(dphi) + amp0*amp1*cp.cos(dphi)*d_dphi)
        s0 = acs.s0_g; s1 = acs.s1_g
        def _gh(v_over_ports):
            v3 = v_over_ports.reshape(N, n_bins, n_v)
            return (v3 * acs.w2[None, None, :]).sum(-1)
        dA_g  = _gh(dA_per[:,:,s0].sum(-1));  dCc_g = _gh(dCc_per[:,:,s0].sum(-1));  dCs_g = _gh(dCs_per[:,:,s0].sum(-1))
        dA_e  = _gh(dA_per[:,:,s1].sum(-1));  dCc_e = _gh(dCc_per[:,:,s1].sum(-1));  dCs_e = _gh(dCs_per[:,:,s1].sum(-1))
        return (P_b*dA_g, P_b*dCc_g, P_b*dCs_g, P_b*dA_e, P_b*dCc_e, P_b*dCs_e)

    d_x0, d_amp0_x0, d_amp1_x0 = sp_grads['x0']
    d_y0, d_amp0_y0, d_amp1_y0 = sp_grads['y0']
    d_vx0, d_amp0_vx0, d_amp1_vx0 = sp_grads['vx0']
    d_vy0, d_amp0_vy0, d_amp1_vy0 = sp_grads['vy0']

    grad = np.zeros((N, 8, 6, n_bins))
    for k in range(8):
        tp = theta_batch.copy(); tp[:, k] += h_theta[k]
        tm = theta_batch.copy(); tm[:, k] -= h_theta[k]
        x0p, y0p, vx0p, vy0p, P_bp = _sample_points_batch(acs, tp)
        x0m, y0m, vx0m, vy0m, P_bm = _sample_points_batch(acs, tm)
        dx0_dtheta  = (((x0p - x0m) / (2*h_theta[k]))).reshape(N, npts, 1)
        dy0_dtheta  = (((y0p - y0m) / (2*h_theta[k]))).reshape(N, npts, 1)
        dvx0_dtheta = (((vx0p - vx0m) / (2*h_theta[k]))).reshape(N, npts, 1)
        dvy0_dtheta = (((vy0p - vy0m) / (2*h_theta[k]))).reshape(N, npts, 1)

        ddphi = d_x0*dx0_dtheta + d_y0*dy0_dtheta + d_vx0*dvx0_dtheta + d_vy0*dvy0_dtheta
        damp0 = d_amp0_x0*dx0_dtheta + d_amp0_y0*dy0_dtheta + d_amp0_vx0*dvx0_dtheta + d_amp0_vy0*dvy0_dtheta
        damp1 = d_amp1_x0*dx0_dtheta + d_amp1_y0*dy0_dtheta + d_amp1_vx0*dvx0_dtheta + d_amp1_vy0*dvy0_dtheta
        out_p = finish_from_coord_grad(ddphi, damp0, damp1)

        dP_b_dtheta = (P_bp - P_bm) / (2*h_theta[k])   # (N, n_bins)
        for j in range(6):
            base_over_Pb = base[j] / cp.maximum(P_b, 1e-300)
            grad[:, k, j, :] = (out_p[j] + dP_b_dtheta * base_over_Pb).get()

    return base, grad
