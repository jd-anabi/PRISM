"""
Unit test 1 — on-manifold consistency for the Vieta cubic.

When a_1·a_2 = a_3, the cubic 8μ³ + 8a_1μ² + 2(a_1²+a_2)μ - (a_3 - a_1·a_2) = 0
reduces to μ·(8μ² + 8a_1μ + 2(a_1²+a_2)) = 0. The trivial root μ = 0 should be
returned with Ω_0' = √a_2 and ν = a_1, assuming a_1 > 0, a_2 > 0.
"""
import math

import pytest

from core.Reduction.linear import JacobianPoly, solve_vieta_cubic


@pytest.mark.parametrize("a_1,a_2", [
    (1.0, 0.04),
    (2.5, 0.16),
    (0.5, 1.0),
])
def test_on_manifold_zero_root(a_1, a_2):
    a_3 = a_1 * a_2
    poly = JacobianPoly(a_1=a_1, a_2=a_2, a_3=a_3)
    mu, Omega0, nu = solve_vieta_cubic(poly)

    assert math.isclose(mu, 0.0, abs_tol=1e-10), f"expected μ=0, got {mu}"
    assert math.isclose(Omega0, math.sqrt(a_2), rel_tol=1e-10)
    assert math.isclose(nu, a_1, rel_tol=1e-10)
