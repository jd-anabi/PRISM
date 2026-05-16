"""
Phase A — linear part.

Builds the Jacobian characteristic-polynomial coefficients (a_1, a_2, a_3),
solves the Vieta cubic for the bifurcation parameter μ_H and intrinsic
frequency Ω_0, and constructs the right/left eigenvectors of the slow complex
mode in the biorthogonal scaling p_1·q_1 = 1.
"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np

from .fixed_point import FixedPoint


# Re-used roots tolerance: numpy.roots returns complex roots even for real
# polynomials; treat a root as "real" if its imaginary part is below this.
_REAL_TOL = 1e-9


@dataclass(frozen=True)
class JacobianPoly:
    """Coefficients of the Jacobian characteristic polynomial σ³ + a_1σ² + a_2σ + a_3."""
    a_1: float
    a_2: float
    a_3: float


@dataclass(frozen=True)
class LinearMode:
    """Slow-mode eigendata in NWK-ND units. All intermediates kept for re-use in Phase A cubic."""
    mu_N: float                # real part of the complex eigenvalue pair
    Omega0_N: float            # imaginary part (intrinsic frequency)
    nu: float                  # 3rd (real) eigenvalue, = 2μ_H + a_1
    Lambda: complex            # μ_H + i·Ω_0
    # Eigenvector intermediates (all complex)
    A_sym: complex             # 1 + K̄ - g + Λ
    U: complex                 # 1 + Λτ̄
    D_aux: complex             # (1-g)U + gφS
    B_bo: complex              # biorthogonal bilinear product
    q1: np.ndarray             # (3,) complex right eigenvector  [1, A/(1-g), -g(K̄+Λ)/((1-g)U)]
    p1: np.ndarray             # (3,) complex left eigenvector (biorthogonal to q1)
    L_q1: complex              # slow-direction linear functional value: -(K̄+Λ)/(1-g)


def jacobian_coeffs(k: float, lam: float, tau: float, fp: FixedPoint) -> JacobianPoly:
    """
    Build (a_1, a_2, a_3) for the NWK characteristic polynomial.

    Formulas straight from the seed (Phase A linear):
      a_1 = (1+K̄-g) + (1-g)/λ̄ + 1/τ̄
      a_2 = K̄(1-g)/λ̄ + (1-g+K̄)/τ̄ + (1-gρ_2)/(λ̄τ̄)
      a_3 = K̄(1-gρ_2)/(λ̄τ̄)
    g and ρ_2 = 1 - φS come from the fixed-point object.
    """
    g = fp.g
    rho_2 = fp.rho_2
    a_1 = (1.0 + k - g) + (1.0 - g) / lam + 1.0 / tau
    a_2 = (k * (1.0 - g) / lam
           + (1.0 - g + k) / tau
           + (1.0 - g * rho_2) / (lam * tau))
    a_3 = k * (1.0 - g * rho_2) / (lam * tau)
    return JacobianPoly(a_1=a_1, a_2=a_2, a_3=a_3)


def solve_vieta_cubic(poly: JacobianPoly) -> tuple[float, float, float]:
    """
    Solve 8μ³ + 8a_1μ² + 2(a_1² + a_2)μ - (a_3 - a_1·a_2) = 0 for μ_H.

    Returns (μ_H, Ω_0, ν) for the real root that gives Ω_0² > 0 and ν > 0,
    where Ω_0² = a_2 + 3μ_H² + 2a_1μ_H and ν = 2μ_H + a_1.

    Raises ValueError if no admissible root exists (e.g., off-Hopf-manifold).
    """
    a_1, a_2, a_3 = poly.a_1, poly.a_2, poly.a_3
    coeffs = [8.0, 8.0 * a_1, 2.0 * (a_1 ** 2 + a_2), -(a_3 - a_1 * a_2)]
    roots = np.roots(coeffs)

    admissible: list[tuple[float, float, float]] = []
    for r in roots:
        if abs(r.imag) > _REAL_TOL:
            continue
        mu = float(r.real)
        Omega2 = a_2 + 3.0 * mu * mu + 2.0 * a_1 * mu
        nu = 2.0 * mu + a_1
        if Omega2 > 0.0 and nu > 0.0:
            admissible.append((mu, float(np.sqrt(Omega2)), nu))

    if not admissible:
        raise ValueError(
            f"No admissible Vieta root (Ω_0²>0 ∧ ν>0). Got roots={roots}, "
            f"a_1={a_1:.4g}, a_2={a_2:.4g}, a_3={a_3:.4g}."
        )
    # If multiple admissible roots, pick the one with smallest |μ_H| — closest
    # to the Hopf manifold, which is the physically relevant slow mode.
    admissible.sort(key=lambda t: abs(t[0]))
    return admissible[0]


def build_eigenvectors(k: float, lam: float, tau: float, f_max: float, s: float,
                       fp: FixedPoint, mu_H: float, Omega0: float, nu: float) -> LinearMode:
    """
    Construct the slow-mode right (q_1) and left (p_1) eigenvectors in
    biorthogonal scaling p_1·q_1 = 1, plus the intermediates A, U, D_aux, B_bo
    needed by Phase A cubic and Phase B projection.

    Option I scale-fixing: q_{1,x} = 1.
    """
    g = fp.g
    Lambda = complex(mu_H, Omega0)

    one_minus_g = 1.0 - g
    A_sym = 1.0 + k - g + Lambda
    U = 1.0 + Lambda * tau
    D_aux = one_minus_g * U + g * f_max * s
    B_bo = (D_aux * one_minus_g * U
            + lam * (A_sym ** 2) * (U ** 2)
            - g * f_max * s * tau * A_sym * (k + Lambda))

    # Right eigenvector q_1 (Option I, q_{1,x}=1)
    q1 = np.array([
        1.0 + 0j,
        A_sym / one_minus_g,
        -g * (k + Lambda) / (one_minus_g * U),
    ], dtype=np.complex128)

    # Left eigenvector p_1, biorthogonal to q_1 (p_1·q_1 = 1)
    p1 = np.array([
        one_minus_g * U * D_aux / B_bo,
        lam * one_minus_g * A_sym * (U ** 2) / B_bo,
        f_max * s * tau * one_minus_g * A_sym * U / B_bo,
    ], dtype=np.complex128)

    L_q1 = -(k + Lambda) / one_minus_g

    return LinearMode(
        mu_N=mu_H, Omega0_N=Omega0, nu=nu,
        Lambda=Lambda, A_sym=A_sym, U=U, D_aux=D_aux, B_bo=B_bo,
        q1=q1, p1=p1, L_q1=L_q1,
    )
