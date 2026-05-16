"""
Phase A — cubic Lyapunov coefficient.

Computes Kuznetsov's first Lyapunov coefficient g_21^eff in the rank-1-simplified
form, then extracts the Hopf cubic parameters α_H^(N) and β_H^(N) in NWK-ND
time units.
"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np

from .fixed_point import FixedPoint
from .linear import JacobianPoly, LinearMode


@dataclass(frozen=True)
class CubicCoeffs:
    """Hopf cubic data in NWK-ND time units."""
    alpha_H_N: float           # -½ Re(g_21^eff)
    beta_H_N: float            # -½ Im(g_21^eff)
    g21_eff: complex
    P_calligraphic: complex    # the 𝒫 prefactor (for diagnostic / degenerate-locus checks)


def _p_J(sigma: complex, poly: JacobianPoly) -> complex:
    """Characteristic polynomial p_J(σ) = σ³ + a_1σ² + a_2σ + a_3."""
    a_1, a_2, a_3 = poly.a_1, poly.a_2, poly.a_3
    return sigma ** 3 + a_1 * sigma ** 2 + a_2 * sigma + a_3


def _R_resolvent(sigma: complex, k: float, lam: float, tau: float,
                 f_max: float, s: float, poly: JacobianPoly) -> complex:
    """
    Auxiliary resolvent 𝓡(σ) — see seed Phase A cubic.

      𝓡(σ) = [ (τ̄σ+1)·((1+λ̄)σ+K̄) − φS(σ+K̄) ] / [ λ̄τ̄·p_J(σ) ]
    """
    numer = (tau * sigma + 1.0) * ((1.0 + lam) * sigma + k) - f_max * s * (sigma + k)
    denom = lam * tau * _p_J(sigma, poly)
    return numer / denom


def lyapunov_coefficient(
    k: float, lam: float, tau: float, f_max: float, s: float,
    fp: FixedPoint, poly: JacobianPoly, mode: LinearMode,
) -> CubicCoeffs:
    """
    Compute g_21^eff = 𝒫 · (L·q_1)² · conj(L·q_1) · [ P_o'''* + 2(P_o''*)² 𝓡_0 + (P_o''*)² 𝓡_2 ]
    and read off α_H^(N), β_H^(N).

    𝓡_0 = 𝓡(2μ_H)  (real),  𝓡_2 = 𝓡(2Λ)  (complex).
    """
    Lambda = mode.Lambda
    one_minus_g = 1.0 - fp.g

    # 𝒫 prefactor (complex)
    P_calligraphic = (one_minus_g * mode.U
                      * (f_max * s * (1.0 + k + Lambda) - mode.U * (k + Lambda))
                      / mode.B_bo)

    R_0 = _R_resolvent(complex(2.0 * mode.mu_N, 0.0),
                       k, lam, tau, f_max, s, poly)
    R_2 = _R_resolvent(2.0 * Lambda, k, lam, tau, f_max, s, poly)

    bracket = (fp.P_o_ppp
               + 2.0 * (fp.P_o_pp ** 2) * R_0
               + (fp.P_o_pp ** 2) * R_2)

    g21_eff = (P_calligraphic
               * (mode.L_q1 ** 2)
               * np.conj(mode.L_q1)
               * bracket)

    alpha_H_N = -0.5 * float(g21_eff.real)
    beta_H_N = -0.5 * float(g21_eff.imag)

    return CubicCoeffs(
        alpha_H_N=alpha_H_N,
        beta_H_N=beta_H_N,
        g21_eff=complex(g21_eff),
        P_calligraphic=complex(P_calligraphic),
    )
