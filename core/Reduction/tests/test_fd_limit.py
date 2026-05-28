"""
Unit test 2 — fluctuation-dissipation limit of the calcium noise amplitude.

When τ̄_c = τ̄, the calcium noise variance should reduce to
  σ̃_C² = 2·g / (N·β·τ̄)
where g = β·P_o*(1-P_o*). Test this against the standalone helper.
"""
import math

from core.Reduction.fixed_point import solve_fixed_point
from core.Reduction.projection import nwk_noise_amplitudes_squared


def test_fd_limit_tauc_equals_tau():
    # Use cell-file Nadrowski values
    k = 0.8; lam = 3.57; f_max = 1.06
    tau = 0.027
    s = 0.65; beta = 14.1; delta_E = 10.0
    n = 50; temp = 1.5

    tau_c = tau  # FD limit

    fp = solve_fixed_point(k=k, f_max=f_max, s=s, beta=beta, delta_E=delta_E)
    sigma_X_sq, sigma_Xa_sq, sigma_C_sq = nwk_noise_amplitudes_squared(
        fp, lam=lam, tau=tau, tau_c=tau_c, n=n, beta=beta, temp=temp,
    )

    expected_sigma_C_sq = 2.0 * fp.g / (n * beta * tau)
    assert math.isclose(sigma_C_sq, expected_sigma_C_sq, rel_tol=1e-12)

    # Also verify σ̃_X² and σ̃_Xa² match their seed formulas.
    assert math.isclose(sigma_X_sq, 2.0 / (n * beta), rel_tol=1e-12)
    assert math.isclose(sigma_Xa_sq, 2.0 * temp / (n * beta * lam), rel_tol=1e-12)
