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

from core.config import CHUNK_LEN, REPARAM_FISHER_M, REPARAM_FISHER_DZ
from core.SBI import pipeline
from core.SBI.statistics import FEATURE_LABELS
from core.SBI.reparam import build_inferred_bijection, fisher_eigenbasis


def build_latent_fisher_rotation(cfg, T=None, m: int = None, dz: float = None) -> torch.Tensor:
    """
    Decorrelating rotation V (P, P), P = ND + rescale dims, from the GT latent Fisher.

    :param cfg: SimConfig (provides model, params, rescale, forcing, time grid, device).
    :param T: the box bijection (build_inferred_bijection(cfg)); rebuilt if None.
    :param m: ensemble per latent perturbation (default config.REPARAM_FISHER_M).
    :param dz: latent central-difference step (default config.REPARAM_FISHER_DZ).
    :return: orthogonal V on cfg.hw.device; w = z @ V are the decorrelated flow coordinates.
    """
    T = T if T is not None else build_inferred_bijection(cfg)
    m = m or REPARAM_FISHER_M
    dz = dz if dz is not None else REPARAM_FISHER_DZ
    dtype, device = cfg.hw.dtype, cfg.hw.device
    nd_dim = len(cfg.params_dict)
    P = nd_dim + len(cfg.rescale_params)
    N_obs = int(cfg.T_obs / cfg.dt_exp)
    forcing_gt = torch.tensor([[v for v, _ in cfg.force_params_dict.values()]], dtype=dtype, device=device)
    amp_v, freq_v, phase_v = (forcing_gt[:, cfg.forcing_idx[k]] for k in ("amp", "freq", "phase"))
    z_gt = T.inv(cfg.ground_truth_tensor)

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
        xsc = res[cfg.rescale_idx["x_scale"]].double(); xof = res[cfg.rescale_idx["x_offset"]].double()
        return pipeline.gen_stats(xsc * xs.double() + xof, xsc * xf.double() + xof, cfg.dt_exp,
                                  amp_v.expand(mm).double(), freq_v.expand(mm).double(),
                                  phase_v.expand(mm).double(), device=device).numpy()

    with torch.no_grad():
        fnoise = np.maximum(feats(cfg.ground_truth_tensor, max(4 * m, 128)).std(0), 1e-9)
        J = np.zeros((len(FEATURE_LABELS), P))
        for i in range(P):
            zp = z_gt.clone(); zp[i] += dz
            zm = z_gt.clone(); zm[i] -= dz
            J[:, i] = (feats(T(zp), m).mean(0) - feats(T(zm), m).mean(0)) / (2 * dz) / fnoise
    F = torch.tensor(J.T @ J, dtype=torch.float64, device=device)
    return fisher_eigenbasis(F).to(dtype)
