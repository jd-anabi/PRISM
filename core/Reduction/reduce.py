"""
Top-level orchestrator: NWK params + cell-file dimensional factors → ReductionRecord.

The single-point reduction map. Composes fixed_point → linear → cubic →
projection → rescaling into one call and packs every reported quantity into a
flat, immutable dataclass.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Literal

import numpy as np

from .fixed_point import solve_fixed_point
from .linear import jacobian_coeffs, solve_vieta_cubic, build_eigenvectors
from .cubic import lyapunov_coefficient
from .projection import project
from .rescaling import apply_stage2


# NWK parameter keys expected from cfg.params_dict (cell-file naming).
NWK_KEYS = ("k", "lam", "f_max", "tau", "tau_c", "c_0", "s",
            "delta_E", "beta", "n", "temp")


class ReductionFailure(RuntimeError):
    """Raised when the reduction map cannot be evaluated at a parameter point
    (fixed-point bracket fails, Vieta cubic has no admissible root, etc.)."""


@dataclass(frozen=True)
class ReductionRecord:
    """Flat record of all reported quantities for one operating point."""
    # Inputs (NWK ND params + dim factors)
    nwk_params: dict
    t_scale_nwk: float
    x_scale_nwk: float
    # Fixed point
    P_o_star: float
    xi_star: float
    x_star: float
    y_star: float
    c_star: float
    g: float
    # Phase A linear (NWK-ND units)
    a_1: float
    a_2: float
    a_3: float
    mu_N: float
    Omega0_N: float
    nu: float
    # Phase A cubic
    alpha_H_N: float
    beta_H_N: float
    g21_eff: complex
    # Phase B (NWK-ND units)
    Sigma_hat_barzz: float
    Sigma_hat_zz_mag: float
    sigma_x_N: float
    sigma_y_N: float
    theta_rot: float
    F_X_N: float
    F_Y_N: float
    # Stage 2 (fully-ND Hopf units)
    mu_tilde: float
    beta_tilde: float
    sigma_x_Hopf: float
    sigma_y_Hopf: float
    F_X_Hopf: float
    F_Y_Hopf: float
    t_scale_Hopf: float
    x_scale_Hopf: float
    # Regime
    regime: Literal["subcritical", "supercritical"]

    def to_flat_dict(self) -> dict:
        """Flat dict suitable for DataFrame row construction (complex split into re/im)."""
        d = asdict(self)
        # Flatten nested nwk_params into top-level keys with `nwk_` prefix to avoid collisions.
        nwk = d.pop("nwk_params")
        for k, v in nwk.items():
            d[f"nwk_{k}"] = v
        # numpy / complex → plain Python
        d["g21_eff_re"] = float(self.g21_eff.real)
        d["g21_eff_im"] = float(self.g21_eff.imag)
        d.pop("g21_eff")
        return d


def reduce_nwk_to_hopf(
    nwk_params: dict,
    t_scale_nwk: float,
    x_scale_nwk: float,
    F_amplitude: float = 1.0,
) -> ReductionRecord:
    """
    Run the full reduction pipeline at a single NWK operating point.

    :param nwk_params: dict containing every key in NWK_KEYS (cell-file naming:
                       k, lam, f_max, tau, tau_c, c_0, s, delta_E, beta, n, temp).
    :param t_scale_nwk: cell-file λ/K_gs value (e.g. ms).
    :param x_scale_nwk: cell-file D value (e.g. nm).
    :param F_amplitude: NWK forcing amplitude F̃ entering Phase B1. Default 1.0;
                        downstream code that wants forcing in physical units can
                        scale by the actual F0.

    :returns: ReductionRecord with every reported quantity.
    :raises ReductionFailure: if any phase fails (bracket, cubic, etc.).
    """
    missing = [k for k in NWK_KEYS if k not in nwk_params]
    if missing:
        raise ReductionFailure(f"nwk_params missing keys: {missing}")

    p = {k: float(nwk_params[k]) for k in NWK_KEYS}

    try:
        fp = solve_fixed_point(
            k=p["k"], f_max=p["f_max"], s=p["s"],
            beta=p["beta"], delta_E=p["delta_E"], c_0=p["c_0"],
        )
    except ValueError as e:
        raise ReductionFailure(f"Fixed-point solve failed: {e}") from e

    poly = jacobian_coeffs(k=p["k"], lam=p["lam"], tau=p["tau"], fp=fp)

    try:
        mu_N, Omega0_N, nu = solve_vieta_cubic(poly)
    except ValueError as e:
        raise ReductionFailure(f"Vieta cubic solve failed: {e}") from e

    mode = build_eigenvectors(
        k=p["k"], lam=p["lam"], tau=p["tau"],
        f_max=p["f_max"], s=p["s"], fp=fp,
        mu_H=mu_N, Omega0=Omega0_N, nu=nu,
    )

    cubic = lyapunov_coefficient(
        k=p["k"], lam=p["lam"], tau=p["tau"],
        f_max=p["f_max"], s=p["s"],
        fp=fp, poly=poly, mode=mode,
    )

    proj = project(
        fp=fp, mode=mode,
        lam=p["lam"], tau=p["tau"], tau_c=p["tau_c"],
        n=p["n"], beta=p["beta"], temp=p["temp"],
        F_amplitude=F_amplitude,
    )

    rescaled = apply_stage2(
        mu_N=mu_N, Omega0_N=Omega0_N,
        alpha_H_N=cubic.alpha_H_N, beta_H_N=cubic.beta_H_N,
        sigma_x_N=proj.sigma_x_N, sigma_y_N=proj.sigma_y_N,
        F_X_N=proj.F_X_N, F_Y_N=proj.F_Y_N,
        t_scale_nwk=t_scale_nwk, x_scale_nwk=x_scale_nwk,
    )

    regime = "subcritical" if mu_N < 0.0 else "supercritical"

    return ReductionRecord(
        nwk_params=p,
        t_scale_nwk=t_scale_nwk, x_scale_nwk=x_scale_nwk,
        P_o_star=fp.P_o_star, xi_star=fp.xi_star,
        x_star=fp.x_star, y_star=fp.y_star, c_star=fp.c_star,
        g=fp.g,
        a_1=poly.a_1, a_2=poly.a_2, a_3=poly.a_3,
        mu_N=mu_N, Omega0_N=Omega0_N, nu=nu,
        alpha_H_N=cubic.alpha_H_N, beta_H_N=cubic.beta_H_N, g21_eff=cubic.g21_eff,
        Sigma_hat_barzz=proj.Sigma_hat_barzz,
        Sigma_hat_zz_mag=float(np.abs(proj.Sigma_hat_zz)),
        sigma_x_N=proj.sigma_x_N, sigma_y_N=proj.sigma_y_N,
        theta_rot=proj.theta_rot,
        F_X_N=proj.F_X_N, F_Y_N=proj.F_Y_N,
        mu_tilde=rescaled.mu_tilde, beta_tilde=rescaled.beta_tilde,
        sigma_x_Hopf=rescaled.sigma_x_Hopf, sigma_y_Hopf=rescaled.sigma_y_Hopf,
        F_X_Hopf=rescaled.F_X_Hopf, F_Y_Hopf=rescaled.F_Y_Hopf,
        t_scale_Hopf=rescaled.t_scale_Hopf, x_scale_Hopf=rescaled.x_scale_Hopf,
        regime=regime,
    )
