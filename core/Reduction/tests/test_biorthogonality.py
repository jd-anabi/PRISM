"""
Unit test 3 — biorthogonality of left and right eigenvectors.

Verify that the constructed p_1, q_1 satisfy p_1 · q_1 = 1 at the cell-file
parameters (Nadrowski default). Tolerance is generous (1e-10) since the
algebra involves products of complex quantities.
"""
import numpy as np

from core.Reduction.fixed_point import solve_fixed_point
from core.Reduction.linear import jacobian_coeffs, solve_vieta_cubic, build_eigenvectors


def test_biorthogonality_at_cell_params():
    # Nadrowski cell file values
    k = 0.8; lam = 3.57; f_max = 1.06; tau = 0.027
    s = 0.65; beta = 14.1; delta_E = 10.0; c_0 = 0.0

    fp = solve_fixed_point(k=k, f_max=f_max, s=s, beta=beta, delta_E=delta_E, c_0=c_0)
    poly = jacobian_coeffs(k=k, lam=lam, tau=tau, fp=fp)
    mu, Omega0, nu = solve_vieta_cubic(poly)

    mode = build_eigenvectors(
        k=k, lam=lam, tau=tau, f_max=f_max, s=s,
        fp=fp, mu_H=mu, Omega0=Omega0, nu=nu,
    )

    inner = np.dot(mode.p1, mode.q1)
    assert abs(inner - 1.0) < 1e-10, f"|p_1·q_1 - 1| = {abs(inner - 1.0)} not < 1e-10"
