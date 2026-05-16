"""
Phase B — forcing & noise projection, plus noise-diagonalizing rotation.

Phase B1 projects the NWK forcing (on x̃ alone) onto the slow eigenspace.
Phase B2 builds the reduced complex covariances Σ̂_{z̄z} and Σ̂_{zz} from
NWK noise amplitudes propagated through p_1, extracts the diagonal Hopf
noise amplitudes, and rotates the forcing by θ = ½·arg(Σ̂_{zz}).
"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np

from .fixed_point import FixedPoint
from .linear import LinearMode


@dataclass(frozen=True)
class Projection:
    """Forcing + noise projection results in NWK-ND units (post-rotation for forcing)."""
    # Raw NWK noise amplitudes squared (standard SDE form)
    sigma_X_sq: float          # 2/(Nβ)
    sigma_Xa_sq: float         # 2(T_a/T)/(Nβ·λ̄)
    sigma_C_sq: float          # 2·τ̄_c·g / (Nβ·τ̄²)
    # Reduced complex covariances (factor 2/(Nβ) divided out)
    Sigma_hat_barzz: float     # always real ≥ 0
    Sigma_hat_zz: complex
    # Diagonal Hopf noise amplitudes (NWK-ND units)
    sigma_x_N: float
    sigma_y_N: float
    # Noise-diagonalizing rotation angle
    theta_rot: float
    # Rotated forcing amplitudes (NWK-ND units)
    F_X_N: float
    F_Y_N: float


def nwk_noise_amplitudes_squared(
    fp: FixedPoint,
    lam: float, tau: float, tau_c: float, n: float, beta: float, temp: float,
) -> tuple[float, float, float]:
    """
    Standalone helper: (σ̃_X², σ̃_Xa², σ̃_C²) — the three raw NWK noise variances
    in standard SDE form (per the seed Phase B2 noise amplitudes).
    """
    sigma_X_sq = 2.0 / (n * beta)
    sigma_Xa_sq = 2.0 * temp / (n * beta * lam)
    sigma_C_sq = 2.0 * tau_c * fp.g / (n * beta * tau ** 2)
    return sigma_X_sq, sigma_Xa_sq, sigma_C_sq


def project(
    fp: FixedPoint, mode: LinearMode,
    lam: float, tau: float, tau_c: float, n: float, beta: float, temp: float,
    F_amplitude: float,
) -> Projection:
    """
    Build the reduced covariances, diagonalize, and project the (x̃-only) forcing.

    Noise amplitudes in standard SDE form (factored 2/(Nβ) prefactor pulled out):
      σ̃_X²   = 2/(Nβ)                            → hat factor 1
      σ̃_Xa²  = 2(T_a/T)/(Nβ·λ̄)                  → hat factor (T_a/T)/λ̄
      σ̃_C²   = 2·τ̄_c·g / (Nβ·τ̄²)                → hat factor τ̄_c·g / τ̄²

    Forcing (NWK channel 0 only): F̃(t̃) = F̃·cos(ω̃·t̃ + φ).
    Pre-rotation: F̃_X^pre = F̃·Re(p_{1,x}), F̃_Y^pre = F̃·Im(p_{1,x}).
    Rotation by θ = ½ arg(Σ̂_{zz}) yields the diagonalized forcing.
    """
    p1x, p1y, p1z = mode.p1[0], mode.p1[1], mode.p1[2]

    sigma_X_sq, sigma_Xa_sq, sigma_C_sq = nwk_noise_amplitudes_squared(
        fp, lam=lam, tau=tau, tau_c=tau_c, n=n, beta=beta, temp=temp,
    )

    # Hat factors carry the projection weights for each NWK noise channel.
    w_y = temp / lam
    w_c = tau_c * fp.g / (tau ** 2)

    Sigma_hat_barzz = (np.abs(p1x) ** 2
                       + w_y * np.abs(p1y) ** 2
                       + w_c * np.abs(p1z) ** 2)
    # |.|^2 are real; cast to float for downstream typing.
    Sigma_hat_barzz = float(Sigma_hat_barzz.real if hasattr(Sigma_hat_barzz, "real") else Sigma_hat_barzz)

    Sigma_hat_zz = (p1x ** 2 + w_y * p1y ** 2 + w_c * p1z ** 2)

    # Diagonal Hopf noise amplitudes
    mag_zz = float(np.abs(Sigma_hat_zz))
    sigma_x_N_sq = (Sigma_hat_barzz + mag_zz) / (n * beta)
    sigma_y_N_sq = (Sigma_hat_barzz - mag_zz) / (n * beta)

    # Numerical guard: |Σ̂_{zz}| ≤ Σ̂_{z̄z} by Cauchy–Schwarz, but rounding can leak.
    sigma_y_N_sq = max(sigma_y_N_sq, 0.0)
    sigma_x_N = float(np.sqrt(sigma_x_N_sq))
    sigma_y_N = float(np.sqrt(sigma_y_N_sq))

    # Noise-diagonalizing rotation
    theta = 0.5 * float(np.angle(Sigma_hat_zz))

    # Rotated forcing: F̃_X^(N) = F·Re(e^{-iθ}·p_{1,x}), F̃_Y^(N) = F·Im(e^{-iθ}·p_{1,x})
    p1x_rot = np.exp(-1j * theta) * p1x
    F_X_N = float(F_amplitude * p1x_rot.real)
    F_Y_N = float(F_amplitude * p1x_rot.imag)

    return Projection(
        sigma_X_sq=sigma_X_sq, sigma_Xa_sq=sigma_Xa_sq, sigma_C_sq=sigma_C_sq,
        Sigma_hat_barzz=Sigma_hat_barzz,
        Sigma_hat_zz=complex(Sigma_hat_zz),
        sigma_x_N=sigma_x_N, sigma_y_N=sigma_y_N,
        theta_rot=theta,
        F_X_N=F_X_N, F_Y_N=F_Y_N,
    )
