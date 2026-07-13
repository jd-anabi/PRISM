"""
Simulation-based Fisher eigenbasis for the decorrelating reparameterization (Track A).

V = eigenvectors of the latent Fisher F = J^T J, where J is the standardized feature-Jacobian
w.r.t. the flow's latent coordinate z at the ground-truth operating point (perturb z -> T(z) ->
simulate spontaneous + forced -> 41 features). Rotating the flow's coordinate by V decorrelates
the (near-degenerate) posterior so the flow can calibrate it.

No trained posterior is needed -- F comes from the simulator alone, so this generalizes to any
model. V = I (REPARAM_ROTATE=False) recovers the plain pipeline exactly. Validated end-to-end by
scripts/reparam_selftest.py (orthogonality, bijection round-trip, decorrelation).
"""
import math

import numpy as np
import torch

from core.config import CHUNK_LEN, REPARAM_FISHER_M, REPARAM_FISHER_DZ, REPARAM_FISHER_POINTS
from core.SBI import pipeline
from core.SBI.statistics import FEATURE_LABELS
from core.SBI.reparam import build_inferred_bijection, fisher_eigenbasis


def build_latent_fisher_rotation(cfg, T=None, m: int = None, dz: float = None,
                                 latent_prior=None, n_points: int = None) -> torch.Tensor:
    """
    Decorrelating rotation V (P, P), P = ND + rescale dims, from the latent Fisher, AVERAGED over
    n_points operating points (GT + prior draws). Averaging makes the (single, linear) rotation
    valid across the prior rather than only at GT — the multiplicative degeneracies curve away from
    GT, so a GT-only V re-correlates off-GT (see the K=10 SBC redistribution finding). Pairs with
    the log-space box (REPARAM_LOG_PARAMS), which linearizes those degeneracies in the first place.

    :param cfg: SimConfig (provides model, params, rescale, forcing, time grid, device).
    :param T: the box bijection (build_inferred_bijection(cfg)); rebuilt if None.
    :param m: ensemble per latent perturbation (default config.REPARAM_FISHER_M).
    :param dz: latent central-difference step (default config.REPARAM_FISHER_DZ).
    :param latent_prior: latent inferred prior to draw the extra operating points from. If None,
                         only GT is used (original GT-only behavior, regardless of n_points).
    :param n_points: number of operating points GT + (n_points-1) prior draws (default
                     config.REPARAM_FISHER_POINTS). n_points=1 => GT only.
    :return: orthogonal V on cfg.hw.device; w = z @ V are the decorrelated flow coordinates.
    """
    T = T if T is not None else build_inferred_bijection(cfg)
    m = m or REPARAM_FISHER_M
    dz = dz if dz is not None else REPARAM_FISHER_DZ
    n_points = n_points or REPARAM_FISHER_POINTS
    dtype, device = cfg.hw.dtype, cfg.hw.device
    nd_dim = len(cfg.params_dict)
    P = nd_dim + len(cfg.rescale_params)
    N_obs = int(cfg.T_obs / cfg.dt_exp)
    forcing_gt = torch.tensor([[v for v, _ in cfg.force_params_dict.values()]], dtype=dtype, device=device)
    amp_v, freq_v, phase_v = (forcing_gt[:, cfg.forcing_idx[k]] for k in ("amp", "freq", "phase"))

    def feats(theta_row, mm):
        nd = theta_row[:nd_dim].unsqueeze(0).expand(mm, -1).contiguous()
        res = theta_row[nd_dim:]
        t_scale = float(res[cfg.rescale_idx["t_scale"]])
        subs = max(1, round((cfg.dt_exp / t_scale) / cfg.dt_nd_min))
        n_fine = min(cfg.steady_idx + N_obs * subs, len(cfg.t))
        t_fine = cfg.t[:n_fine]
        n_segs = max(1, math.ceil(n_fine / CHUNK_LEN))
        rv = res.unsqueeze(0).expand(mm, -1).contiguous()
        force = pipeline.build_nondim_sin_force_tensor(forcing_gt.expand(mm, -1), t_fine, rv,
                                                       cfg.forcing_idx, cfg.rescale_idx)

        def s(f):
            return pipeline.gen_obs(model=cfg.model, params=nd, t=t_fine,
                                    inits=cfg.inits_tensor.expand(mm, -1).contiguous(), force=f,
                                    n_segs=n_segs, steady_idx=cfg.steady_idx,
                                    state_dep_drift=cfg.state_dep_drift, batch_size=mm, dtype=dtype,
                                    device=device)[0][:, ::subs][:, :N_obs]
        torch.manual_seed(1); xf = s(force)
        torch.manual_seed(2); xs = s(torch.zeros_like(force))
        xsc = res[cfg.rescale_idx["x_scale"]].double()
        xof = res[cfg.rescale_idx["x_offset"]].double() if "x_offset" in cfg.rescale_idx else 0.0
        return pipeline.gen_stats(xsc * xs.double() + xof, xsc * xf.double() + xof, cfg.dt_exp,
                                  amp_v.expand(mm).double(), freq_v.expand(mm).double(),
                                  phase_v.expand(mm).double(), device=device).numpy()

    def fisher_at(theta_row):
        """Per-point standardized feature-Fisher F_k = J^T J, or None if features are non-finite."""
        f0 = feats(theta_row, max(4 * m, 128))
        if not np.isfinite(f0).all():
            return None
        fnoise = np.maximum(f0.std(0), 1e-9)
        z0 = T.inv(theta_row)
        J = np.zeros((len(FEATURE_LABELS), P))
        for i in range(P):
            zp = z0.clone(); zp[i] += dz
            zm = z0.clone(); zm[i] -= dz
            J[:, i] = (feats(T(zp), m).mean(0) - feats(T(zm), m).mean(0)) / (2 * dz) / fnoise
        return None if not np.isfinite(J).all() else (J.T @ J)

    # Operating points: GT first, then (n_points-1) prior draws (if a prior was provided).
    points = [cfg.ground_truth_tensor]
    if latent_prior is not None and n_points > 1:
        z_samp = latent_prior.sample((n_points - 1,)).to(device)
        points += [T(z_samp[k]) for k in range(z_samp.shape[0])]

    with torch.no_grad():
        F_accum = np.zeros((P, P)); n_used = 0
        for k, theta_row in enumerate(points):
            Fk = fisher_at(theta_row.to(device))
            if Fk is None:
                print(f"[fisher] operating point {k} gave non-finite features; skipping", flush=True)
                continue
            F_accum += Fk; n_used += 1
    if n_used == 0:
        raise RuntimeError("Fisher rotation: all operating points produced non-finite features.")
    print(f"[fisher] averaged simulation Fisher over {n_used}/{len(points)} operating points "
          f"(GT + {n_used - 1} prior draw(s))", flush=True)
    F = torch.tensor(F_accum / n_used, dtype=torch.float64, device=device)
    return fisher_eigenbasis(F).to(dtype)
