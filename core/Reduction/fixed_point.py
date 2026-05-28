"""
Phase A — fixed point.

Solves the transcendental NWK fixed-point equation for the open-channel
probability P_o*, then derives the auxiliary fixed-point state variables and
the first three derivatives of P_o(ξ) used in subsequent phases.
"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq


_BRACKET_LO, _BRACKET_HI = -20.0, 20.0


@dataclass(frozen=True)
class FixedPoint:
    """All scalar fixed-point quantities needed by Phase A linear + cubic."""
    P_o_star: float
    xi_star: float
    x_star: float
    y_star: float
    c_star: float
    g: float            # β · P_o*(1-P_o*)
    P_o_pp: float       # P_o''*
    P_o_ppp: float      # P_o'''*
    A_gate: float
    rho_2: float


def _P_o(xi: float, beta: float, A_gate: float) -> float:
    return 1.0 / (1.0 + A_gate * np.exp(-beta * xi))


def _residual(xi: float, beta: float, A_gate: float, rho_2: float, const: float) -> float:
    return xi - _P_o(xi, beta, A_gate) * rho_2 - const


def solve_fixed_point(
    k: float, f_max: float, s: float,
    beta: float, delta_E: float,
) -> FixedPoint:
    """
    Solve the NWK fixed point and compute P_o derivatives at it.

    Inputs are the subset of NWK parameters that the deterministic fixed-point
    depends on: k (pivot stiffness), f_max (φ), s (S), beta, delta_E (ΔG).
    The drag ratio lam and time scales tau, tau_c do not enter at the fixed point.
    """
    A_gate = float(np.exp(delta_E + beta / 2.0))
    rho_2 = 1.0 - f_max * s
    const = f_max

    r_lo = _residual(_BRACKET_LO, beta, A_gate, rho_2, const)
    r_hi = _residual(_BRACKET_HI, beta, A_gate, rho_2, const)
    if r_lo * r_hi > 0:
        raise ValueError(
            f"Fixed-point bracket [{_BRACKET_LO}, {_BRACKET_HI}] does not contain a sign change "
            f"(r_lo={r_lo:.3e}, r_hi={r_hi:.3e}). Operating point may be off-branch."
        )

    xi_star = brentq(_residual, _BRACKET_LO, _BRACKET_HI,
                     args=(beta, A_gate, rho_2, const), xtol=1e-14, rtol=1e-14)

    P_o_star = _P_o(xi_star, beta, A_gate)
    g = beta * P_o_star * (1.0 - P_o_star)
    P_o_pp = (beta ** 2) * P_o_star * (1.0 - P_o_star) * (1.0 - 2.0 * P_o_star)
    P_o_ppp = (beta ** 3) * P_o_star * (1.0 - P_o_star) * (1.0 - 6.0 * P_o_star + 6.0 * P_o_star ** 2)

    # Auxiliary state variables (from NWK steady-state algebra):
    #   ċ = 0  →  c* = P_o*
    #   ẋ = 0  →  k·x* = P_o* - ξ*   →  x* = (P_o* - ξ*)/k
    #   ξ ≡ x − y                    →  y* = x* - ξ*
    c_star = P_o_star
    x_star = (P_o_star - xi_star) / k
    y_star = x_star - xi_star

    return FixedPoint(
        P_o_star=P_o_star, xi_star=xi_star,
        x_star=x_star, y_star=y_star, c_star=c_star,
        g=g, P_o_pp=P_o_pp, P_o_ppp=P_o_ppp,
        A_gate=A_gate, rho_2=rho_2,
    )
