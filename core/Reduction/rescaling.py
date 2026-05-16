"""
Stage 2 — rescaling NWK-ND Hopf quantities to fully-ND Hopf units, plus the
dimensional rescaling factors that connect experimental data to fully-ND Hopf
state variables.
"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Rescaling:
    """Stage-2 outputs in fully-ND Hopf units + dimensional rescaling factors."""
    mu_tilde: float                # μ_H^(N) / Ω_0^(N)
    beta_tilde: float              # β_H^(N) / α_H^(N)
    sigma_x_Hopf: float
    sigma_y_Hopf: float
    F_X_Hopf: float
    F_Y_Hopf: float
    t_scale_Hopf: float            # τ_Hopf = (λ/K_gs) / Ω_0^(N)  =  NWK_t_scale / Ω_0^(N)
    x_scale_Hopf: float            # ℓ_Hopf = D · √(Ω_0^(N) / α_H^(N)) = NWK_x_scale · same factor


def apply_stage2(
    mu_N: float, Omega0_N: float, alpha_H_N: float, beta_H_N: float,
    sigma_x_N: float, sigma_y_N: float,
    F_X_N: float, F_Y_N: float,
    t_scale_nwk: float, x_scale_nwk: float,
) -> Rescaling:
    """
    Apply ratios and rescaling factors per the seed (Stage 2).

      μ̃ = μ_H^(N) / Ω_0^(N)
      β̃ = β_H^(N) / α_H^(N)
      F̃_X^Hopf = F̃_X^(N) · √(α_H^(N) / (Ω_0^(N))³),  same for F̃_Y
      σ̃_x^Hopf = σ̃_x^(N) · √(α_H^(N)) / Ω_0^(N),    same for σ̃_y
      t_scale_Hopf = NWK_t_scale / Ω_0^(N)        (≡ λ / (K_gs · Ω_0^(N)))
      x_scale_Hopf = NWK_x_scale · √(Ω_0^(N) / α_H^(N))  (≡ D · √(Ω_0 / α_H))

    NWK_t_scale and NWK_x_scale are the cell-file rescale_params values
    (λ/K_gs in ms, D in nm respectively). We retain those units; the caller
    is responsible for any further unit harmonization.
    """
    if Omega0_N <= 0.0:
        raise ValueError(f"Omega0_N must be positive, got {Omega0_N}.")

    mu_tilde = mu_N / Omega0_N
    # β_H / α_H is well-defined even on the degenerate locus 𝒫=0 (both vanish
    # together); guard against pathological α_H=0.
    beta_tilde = float("nan") if alpha_H_N == 0.0 else beta_H_N / alpha_H_N

    # Forcing rescaling
    # If α_H_N is negative (subcritical Hopf: cubic stabilizes), √(α/Ω³) is imaginary.
    # The seed treats α_H_N as positive in supercritical and the linear-response
    # framework is the meaningful regime for FDT comparison anyway. To preserve
    # the formula as-given without raising, we work with √(|α_H_N|/Ω₀³) and
    # carry a sign flag implicitly via the regime field at the orchestrator level.
    F_rescale = np.sqrt(abs(alpha_H_N) / (Omega0_N ** 3))
    F_X_Hopf = float(F_X_N * F_rescale)
    F_Y_Hopf = float(F_Y_N * F_rescale)

    sigma_rescale = np.sqrt(abs(alpha_H_N)) / Omega0_N
    sigma_x_Hopf = float(sigma_x_N * sigma_rescale)
    sigma_y_Hopf = float(sigma_y_N * sigma_rescale)

    t_scale_Hopf = float(t_scale_nwk / Omega0_N)
    x_scale_Hopf = float(x_scale_nwk * np.sqrt(Omega0_N / abs(alpha_H_N))) if alpha_H_N != 0.0 else float("nan")

    return Rescaling(
        mu_tilde=mu_tilde, beta_tilde=beta_tilde,
        sigma_x_Hopf=sigma_x_Hopf, sigma_y_Hopf=sigma_y_Hopf,
        F_X_Hopf=F_X_Hopf, F_Y_Hopf=F_Y_Hopf,
        t_scale_Hopf=t_scale_Hopf, x_scale_Hopf=x_scale_Hopf,
    )
