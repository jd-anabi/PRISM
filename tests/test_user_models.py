"""User-defined model tests: parser/codegen, forcing kinds, registry, persistence, Simulate stream,
and the OS-accent / Inter-font appearance settings.

WHAT THESE LOCK DOWN
    * The sympy parse -> torch compile path reproduces a hand-written model (HopfModel) numerically,
      rejects everything outside the locked-down namespace, and pins g = sqrt(2*D).
    * pipeline.build_nondim_sin_force_tensor's refactor onto core/forcing.py stays NUMERICALLY
      IDENTICAL to the original math (SBI training data depends on it), and the new step/triangular/
      exponential kinds + the per-variable user force tensor follow the same nondimensionalization.
    * registry appends user models WITHOUT moving the built-ins (positional VALID_LABELS consumers),
      model_store's emitted Bounds/Cells/Units triple round-trips through the untouched config path,
      and a saved model streams end-to-end through the Simulate worker (incl. the divergence guard).

Round-trip tests write a throwaway model (name UMTEST*) into the real Resources tree and remove it in
a finally -- the exact code path the app takes, no path monkey-patching.

Run:  python -m pytest tests/test_user_models.py -v
      (or just: python tests/test_user_models.py)
"""
import math
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")   # must precede any PySide6 import
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib                                                 # noqa: E402
matplotlib.use("Agg")                                            # match the app (core/gui/__main__.py forces it)

import numpy as np                                                # noqa: E402
import torch                                                      # noqa: E402
from PySide6.QtGui import QPalette                                # noqa: E402
from PySide6.QtWidgets import QApplication                        # noqa: E402

from core import config, forcing, registry                        # noqa: E402
from core.Helpers import file_manager, model_store                # noqa: E402
from core.Models.hopf_model import HopfModel                      # noqa: E402
from core.Models.user_model import (ModelParseError, UserModel,   # noqa: E402
                                    parse_user_model)
from core.SBI import pipeline                                     # noqa: E402


def _app():
    return QApplication.instance() or QApplication([])


def _remove_user_model(name: str):
    try:
        model_store.delete_user_model(name)
    except Exception:                                             # noqa: BLE001 -- cleanup best-effort
        pass
    registry.unregister(name)


# ── parser / codegen ─────────────────────────────────────────────────────────────────────────────
def test_user_model_parse_matches_hopf():
    """A typed Hopf reproduces HopfModel.f/g numerically, and D -> sqrt(2*D) pins the noise map."""
    compiled = parse_user_model([
        {"name": "x", "drift": "mu*x - y - (x - beta*y)*(x^2 + y^2)", "D": "sx^2/2"},
        {"name": "y", "drift": "x + mu*y - (beta*x + y)*(x^2 + y^2)", "D": "sy^2/2"},
    ])
    assert compiled.var_names == ["x", "y"]
    assert compiled.param_names == ["mu", "beta", "sx", "sy"]     # first-appearance order (ctor order)

    torch.manual_seed(0)
    batch = 4
    mu, beta = torch.rand(batch), torch.rand(batch)
    sx, sy = torch.rand(batch) + 0.1, torch.rand(batch) + 0.1
    force = torch.randn(batch, 2, 5)
    um = UserModel(compiled, (mu, beta, sx, sy), force, batch_size=batch)
    hm = HopfModel(mu, beta, sx, sy, force, batch_size=batch)
    x = torch.randn(batch, 2)
    for t in range(5):
        assert torch.allclose(um.f(x, t), hm.f(x, t), atol=1e-6)
    assert torch.allclose(um.g(), hm.g(), atol=1e-6)              # sqrt(2 * s^2/2) == s


def test_user_model_parser_rejects_bad_input():
    bad_definitions = (
        [{"name": "x", "drift": "__import__('os').system('x')", "D": "0"}],   # attribute/dunder
        [{"name": "x", "drift": "x.diff(x)", "D": "0"}],                      # attribute access
        [{"name": "x", "drift": "foo(x)", "D": "0"}],                         # unknown function
        [{"name": "x", "drift": "lambda: 1", "D": "0"}],                      # not an expression
        [{"name": "x", "drift": "a[0]", "D": "0"}],                           # brackets
        [{"name": "t", "drift": "-t", "D": "0"}],                             # reserved name
        [{"name": "x", "drift": "", "D": "0"}],                               # empty drift
    )
    for variables in bad_definitions:
        try:
            parse_user_model(variables)
        except ModelParseError:
            pass
        else:
            raise AssertionError(f"not rejected: {variables}")


def test_state_dependent_noise_detection_and_gx():
    """A D that references state -> state_dep_noise True, g(None)==None cache skipped, g(x)=sqrt(2 D(x))
    per column, and state_dep_drift=True integrates finitely. Additive models keep g() cached and
    x-invariant (backward compatible)."""
    # multiplicative: D = 0.5*x^2 -> g = sqrt(2*0.5*x^2) = |x|
    c = parse_user_model([{"name": "x", "drift": "-k*x", "D": "0.5*x^2"}])
    assert c.state_dep_noise is True and c.param_names == ["k"]
    m = UserModel(c, (torch.ones(3),), torch.zeros(3, 1, 4), batch_size=3)
    assert m._g is None                                              # nothing cached for state-dep
    xq = torch.tensor([[2.0], [3.0], [-4.0]])
    assert torch.allclose(m.g(xq)[:, 0], xq[:, 0].abs())
    # a transient negative D is clamped, never NaN
    cneg = parse_user_model([{"name": "x", "drift": "-x", "D": "x"}])   # D=x can go negative
    mneg = UserModel(cneg, (), torch.zeros(2, 1, 4), batch_size=2)
    assert torch.isfinite(mneg.g(torch.tensor([[-5.0], [5.0]]))).all()
    # euler runs finite under the state-dependent branch
    from core.Solvers import sdeint
    res = sdeint.Solver().euler(
        UserModel(c, (torch.ones(1),), torch.zeros(1, 1, 101), batch_size=1),
        torch.tensor([[0.1]]), (0.0, 0.5), 101, state_dep_drift=True)
    assert torch.isfinite(res).all()

    # additive stays False + cached + x-invariant
    c2 = parse_user_model([{"name": "x", "drift": "-k*x", "D": "d0"}])
    assert c2.state_dep_noise is False
    m2 = UserModel(c2, (torch.ones(3), torch.full((3,), 0.02)), torch.zeros(3, 1, 4), batch_size=3)
    assert m2._g is not None and torch.allclose(m2.g(xq), m2.g())


def test_compiled_step_matches_eager():
    """The @torch.jit.script compiled_step (euler_compiled fast path) must reproduce the eager `euler`
    trajectory for additive AND state-dependent models -- same seed => same dW => identical math. Also:
    a model whose step wasn't built (None) exposes no compiled_step attr and falls back to eager."""
    from core.Models.user_model import build_compiled_step
    from core.Solvers import sdeint
    solver = sdeint.Solver()
    cases = [
        ([{"name": "x", "drift": "mu*x - x^3", "D": "d0"}], [0.5, 0.05], False),
        ([{"name": "x", "drift": "v", "D": "0"},
          {"name": "v", "drift": "-k*x - c*v", "D": "d0"}], [1.0, 0.2, 0.05], False),
        ([{"name": "x", "drift": "-k*x", "D": "0.5*x^2 + d0"}], [1.0, 0.01], True),
    ]
    for variables, pvals, sdd in cases:
        c = parse_user_model(variables)
        c.compiled_step_fn = build_compiled_step(c)
        assert c.compiled_step_fn is not None, variables
        n_vars, B = len(c.var_names), 5
        um = UserModel(c, tuple(torch.full((B,), v) for v in pvals), torch.zeros(B, n_vars, 50),
                       batch_size=B)
        assert hasattr(um, "compiled_step") and um.compiled_params() == um.params
        inits = torch.full((B, n_vars), 0.2)
        torch.manual_seed(3); a = solver.euler(um, inits, (0.0, 0.25), 50, state_dep_drift=sdd)
        torch.manual_seed(3); b = solver.euler_compiled(um, inits, (0.0, 0.25), 50, state_dep_drift=sdd)
        assert torch.allclose(a, b, atol=1e-4, rtol=1e-4), (variables, (a - b).abs().max().item())

    # No compiled step built -> no compiled_step attr -> __sols stays on eager euler (safe fallback).
    c0 = parse_user_model([{"name": "x", "drift": "-x", "D": "d0"}])   # compiled_step_fn stays None
    um0 = UserModel(c0, (torch.full((4,), 0.02),), torch.zeros(4, 1, 10), batch_size=4)
    assert not hasattr(um0, "compiled_step")
    assert torch.isfinite(solver.euler(um0, torch.zeros(4, 1), (0.0, 0.05), 10)).all()


def test_user_model_constant_and_zero_noise_normalization():
    """Constant/param-only expressions come back as scalars -- they must normalize to (batch,), zero-D
    channels must zero-pad g (the BP convention), and a negative constant D yields NaN g (so the SBI
    stability screen / builder smoke test drops that parameter set, rather than raising per-batch)."""
    compiled = parse_user_model([{"name": "x", "drift": "-x", "D": "0"},
                                 {"name": "y", "drift": "1.5", "D": "d0"}])
    d0 = 0.02
    um = UserModel(compiled, (torch.full((3,), d0),), torch.zeros(3, 2, 4), batch_size=3)
    assert um.g().shape == (3, 2)
    assert torch.all(um.g()[:, 0] == 0)
    assert torch.allclose(um.g()[:, 1], torch.full((3,), math.sqrt(2 * d0)))
    fx = um.f(torch.randn(3, 2), 0)
    assert fx.shape == (3, 2) and torch.all(fx[:, 1] == 1.5)
    # negative constant D -> NaN in that channel (screened downstream), no raise
    umn = UserModel(compiled, (torch.full((3,), -1.0),), torch.zeros(3, 2, 4), batch_size=3)
    assert torch.isnan(umn.g()[:, 1]).all()


def test_parameter_discovery_ignores_numeric_literals():
    """Scientific-notation literals ('1e-3') must not shed phantom parameters (e/e3/...) out of the
    mantissa tail, and parameters sympy simplifies away must be dropped from the positional list."""
    compiled = parse_user_model([{"name": "x", "drift": "-k*x + 1e-3", "D": "2.5e-4"}])
    assert compiled.param_names == ["k"], compiled.param_names
    um = UserModel(compiled, (torch.ones(2),), torch.zeros(2, 1, 3), batch_size=2)
    assert torch.allclose(um.f(torch.zeros(2, 1), 0)[:, 0], torch.full((2,), 1e-3))
    assert torch.allclose(um.g()[:, 0], torch.full((2,), math.sqrt(2 * 2.5e-4)))

    compiled = parse_user_model([{"name": "x", "drift": "2.5e-3*x", "D": "0"}])
    assert compiled.param_names == []
    compiled = parse_user_model([{"name": "x", "drift": "a - a + b*x", "D": "0"}])
    assert compiled.param_names == ["b"]                          # 'a' simplified away -> no dead column


def test_E_is_an_ordinary_parameter():
    """'E' must be a user parameter, not silently Euler's constant (physics names E are common)."""
    compiled = parse_user_model([{"name": "x", "drift": "E*x", "D": "0"}])
    assert compiled.param_names == ["E"]
    um = UserModel(compiled, (torch.full((2,), 3.0),), torch.zeros(2, 1, 3), batch_size=2)
    assert torch.allclose(um.f(torch.full((2, 1), 2.0), 0)[:, 0], torch.full((2,), 6.0))
    # exp() still covers the constant: exp(1) folds to Euler's number in the compiled tree.
    compiled = parse_user_model([{"name": "x", "drift": "exp(1) + 0*x", "D": "0"}])
    um = UserModel(compiled, (), torch.zeros(1, 1, 3), batch_size=1)
    assert abs(um.f(torch.zeros(1, 1), 0)[0, 0].item() - math.e) < 1e-6


def test_parser_internal_names_are_rejected():
    """Identifiers that shadow parse_expr's constructors (Float/Integer/...) must be refused with a
    clear message, not break every numeric literal downstream."""
    for bad in ("Float*x", "Integer + 2*x", "Rational*x", "Symbol*x", "Function*x"):
        try:
            parse_user_model([{"name": "x", "drift": bad, "D": "0"}])
        except ModelParseError as e:
            assert "reserved" in str(e), e
        else:
            raise AssertionError(f"not rejected: {bad}")
    try:
        parse_user_model([{"name": "Integer", "drift": "-Integer", "D": "0"}])
    except ModelParseError:
        pass
    else:
        raise AssertionError("variable named Integer not rejected")


# ── forcing ──────────────────────────────────────────────────────────────────────────────────────
def test_forcing_sin_matches_the_original_math():
    """The delegate must be numerically identical to the pre-refactor sinusoidal builder, in both the
    f_scale and the Hopf-style (x_scale/t_scale) nondim branches, incl. the amp_y second channel."""
    fp = torch.tensor([[3.0, 2.0, 0.4, 0.1, 1.7]])               # amp freq phase offset amp_y
    rp = torch.tensor([[62.14, 3.73, 10.0]])                     # x_scale t_scale f_scale
    fidx = {"amp": 0, "freq": 1, "phase": 2, "offset": 3, "amp_y": 4}
    t_nd = torch.linspace(0, 1, 50)
    t_dim = 3.73 * t_nd

    out = pipeline.build_nondim_sin_force_tensor(fp, t_nd, rp, fidx, {"x_scale": 0, "t_scale": 1, "f_scale": 2})
    carrier = torch.sin(2 * np.pi * 2.0 * t_dim + 0.4)
    assert out.shape == (1, 2, 50)
    assert torch.allclose(out[0, 0], (3.0 * carrier + 0.1) / 10.0, atol=1e-6)
    assert torch.allclose(out[0, 1], (1.7 * carrier + 0.1) / 10.0, atol=1e-6)

    out_hopf = pipeline.build_nondim_sin_force_tensor(fp, t_nd, rp, fidx, {"x_scale": 0, "t_scale": 1})
    assert torch.allclose(out_hopf[0, 0], (3.0 * carrier + 0.1) / (62.14 / 3.73), atol=1e-6)


def test_forcing_new_kinds_shapes_and_values():
    rp = torch.tensor([[62.14, 3.73, 10.0]])
    ridx = {"x_scale": 0, "t_scale": 1, "f_scale": 2}
    t_nd = torch.linspace(0, 1, 50)
    t_dim = 3.73 * t_nd

    fp = torch.tensor([[2.0, 1.5, 0.5]])                          # amp t0|tau offset
    out = forcing.build_nondim_force_tensor(fp, t_nd, rp, {"amp": 0, "t0": 1, "offset": 2}, ridx, kind="step")
    assert out.shape == (1, 1, 50)
    assert torch.allclose(out[0, 0], (0.5 + 2.0 * (t_dim >= 1.5).float()) / 10.0)

    out = forcing.build_nondim_force_tensor(fp, t_nd, rp, {"amp": 0, "tau": 1, "offset": 2}, ridx,
                                            kind="exponential", exp_sign=-1.0)
    assert torch.allclose(out[0, 0], (2.0 * torch.exp(-t_dim / 1.5) + 0.5) / 10.0, atol=1e-6)

    fp4 = torch.tensor([[3.0, 2.0, 0.4, 0.1]])                    # amp freq phase offset
    out = forcing.build_nondim_force_tensor(fp4, t_nd, rp, {"amp": 0, "freq": 1, "phase": 2, "offset": 3},
                                            ridx, kind="triangular")
    tri = (2 / np.pi) * torch.asin(torch.sin(2 * np.pi * 2.0 * t_dim + 0.4))
    assert torch.allclose(out[0, 0], (3.0 * tri + 0.1) / 10.0, atol=1e-6)


def test_user_force_tensor_maps_rows_and_zero_fills():
    class Spec:
        variables = [{"name": "x", "forcing": {"kind": "step", "params": {}, "sign": 1}},
                     {"name": "y", "forcing": None}]
    rp = torch.tensor([[62.14, 3.73, 10.0]])
    ridx = {"x_scale": 0, "t_scale": 1, "f_scale": 2}
    t_nd = torch.linspace(0, 1, 50)
    fp = torch.tensor([[2.0, 1.5, 0.5]])
    out = forcing.build_user_force_tensor(Spec(), fp, t_nd, rp, {"amp_x": 0, "t0_x": 1, "offset_x": 2}, ridx)
    assert out.shape == (1, 2, 50)
    assert torch.all(out[0, 1] == 0)
    ref = (0.5 + 2.0 * ((3.73 * t_nd) >= 1.5).float()) / 10.0     # suffixed lookup hit row 0
    assert torch.allclose(out[0, 0], ref)


# ── registry ─────────────────────────────────────────────────────────────────────────────────────
def test_registry_appends_and_unregisters_without_moving_builtins():
    n0 = len(config.VALID_MODELS)
    registry.register(registry.ModelSpec("UMTESTREG", ["a"], is_user_model=True, n_vars=1))
    try:
        assert config.VALID_MODELS[:3] == ["BP", "NADROWSKI", "HOPF"]     # positional consumers
        assert config.VALID_LABELS[config.VALID_MODELS.index("UMTESTREG")] == ["a"]
        assert registry.is_user_model("UMTESTREG")
        assert not registry.state_dep_drift("UMTESTREG")
        assert registry.state_dep_drift("NADROWSKI")
        assert registry.state_dep_drift("unknown-nadrowski-ish")          # legacy fallback
    finally:
        registry.unregister("UMTESTREG")
    assert len(config.VALID_MODELS) == n0 and "UMTESTREG" not in config.VALID_MODELS
    registry.unregister("NADROWSKI")                                       # built-ins are irremovable
    assert "NADROWSKI" in config.VALID_MODELS


def _doc(name="UMTEST", forcing_entry=None):
    return {
        "schema_version": 1,
        "name": name,
        "variables": [
            {"name": "x", "drift": "-k1*x", "D": "d0", "init": 0.1, "forcing": forcing_entry},
            {"name": "y", "drift": "-y + x", "D": "0", "init": 0.0, "forcing": None},
        ],
        "params": {"k1": 1.0, "d0": 0.05},
        "rescale": {"x_scale": 10.0, "t_scale": 0.01},
    }


def test_model_store_round_trip_emits_a_parseable_triple():
    """Save -> the emitted Bounds/Cells/Units parse through file_manager, param order follows the
    discovery order, the t_scale lower bound stays strictly positive (t_nd_max divides by it), and
    delete removes every artifact."""
    sin = {"kind": "sin", "params": {"amp": 0.5, "freq": 10.0, "phase": 0.0, "offset": 0.0}}
    name, folder = "UMTEST", "umtest"
    try:
        model_store.save_user_model(_doc(name, sin))
        b_params, b_rescale, b_forcing, _ = file_manager.parse_bounds_file(
            str(config.BOUNDS_PATH / folder / "default.txt"))
        assert list(b_params) == ["k1", "d0"]
        assert list(b_forcing) == ["amp_x", "freq_x", "phase_x", "offset_x"]
        lo, hi = b_rescale["t_scale"][1]
        assert lo == 0.005 and hi == 0.02                          # (v/2, 2v), strictly positive
        inits, _, v_rescale, _ = file_manager.parse_values_file(
            str(config.CELL_PATH / folder / "default.txt"))
        assert list(inits) == ["x_init", "y_init"]
        assert v_rescale == {"x_scale": 10.0, "t_scale": 0.01}
        assert set(file_manager.parse_units_file(str(config.UNITS_PATH / folder / "units.txt"))) == {"nm", "s"}

        doc = model_store.load_user_model(config.MODELS_PATH / f"{name}.json")
        assert doc["name"] == name and doc["schema_version"] == 2      # v1 _doc migrated on save/load
        assert doc["params"]["k1"] == model_store._param_entry(1.0)    # scalar -> placeholder box
        assert doc["params"]["d0"] == model_store._param_entry(0.05)
        try:
            model_store.validate_name("NADROWSKI")
        except ValueError:
            pass
        else:
            raise AssertionError("built-in name not refused")
    finally:
        _remove_user_model(name)
    assert not (config.MODELS_PATH / f"{name}.json").exists()
    for base in (config.BOUNDS_PATH, config.CELL_PATH, config.UNITS_PATH):
        assert not (base / folder).exists()


def _doc_v2(name="UMTESTV2", params=None):
    """A schema_version-2 doc: params carry {value, lo, hi} (per-parameter SBI bounds, S-1)."""
    return {
        "schema_version": 2,
        "name": name,
        "variables": [{"name": "x", "drift": "-k1*x", "D": "d0", "init": 0.1, "forcing": None}],
        "params": params or {"k1": {"value": 1.0, "lo": 0.5, "hi": 1.5},
                             "d0": {"value": 0.05, "lo": 0.01, "hi": 0.1}},
        "rescale": {"x_scale": 10.0, "t_scale": 0.01},
    }


def test_custom_param_bounds_emitted_verbatim():
    """A v2 doc's per-parameter (lo, hi) reach the Bounds file VERBATIM (not the _nd_bounds placeholder),
    and the value reaches the Cells file -- so a tightened SBI box actually flows into the pipeline."""
    name, folder = "UMTESTBND", "umtestbnd"
    try:
        model_store.save_user_model(_doc_v2(name))
        b_params, _, _, _ = file_manager.parse_bounds_file(str(config.BOUNDS_PATH / folder / "default.txt"))
        assert b_params["k1"][1] == (0.5, 1.5)                     # custom box, not _nd_bounds(1.0)=(0,2)
        assert b_params["d0"][1] == (0.01, 0.1)                    # not _nd_bounds(0.05)=(-0.95,1.05)
        _, vals, _, _ = file_manager.parse_values_file(str(config.CELL_PATH / folder / "default.txt"))
        assert vals["k1"] == 1.0 and vals["d0"] == 0.05
    finally:
        _remove_user_model(name)


def test_param_bounds_validation_rejects_bad_boxes():
    """lo >= hi and value-outside-[lo,hi] must fail at save time, persisting nothing."""
    name = "UMTESTBADBOX"
    for params, needle in (
        ({"k1": {"value": 1.0, "lo": 2.0, "hi": 1.0}, "d0": {"value": 0.05, "lo": 0.0, "hi": 0.1}}, "lo must be < hi"),
        ({"k1": {"value": 5.0, "lo": 0.0, "hi": 1.0}, "d0": {"value": 0.05, "lo": 0.0, "hi": 0.1}}, "outside its bounds"),
    ):
        try:
            model_store.save_user_model(_doc_v2(name, params))
        except ValueError as e:
            assert needle in str(e), e
        else:
            raise AssertionError(f"bad box not refused: {params}")
    assert not (config.MODELS_PATH / f"{name}.json").exists()


def test_v1_params_migrate_to_placeholder_boxes():
    """schema_version-1 scalar params migrate IN MEMORY to {value, lo, hi} placeholder boxes (non-mutating);
    a v2 doc is returned unchanged. This is what keeps old Resources/Models/*.json working after the bump."""
    v1 = {"schema_version": 1, "name": "X", "variables": [], "params": {"k": 2.0, "d0": 0.05},
          "rescale": {"x_scale": 10.0, "t_scale": 0.01}}
    out = model_store._normalize_to_v2(v1)
    assert out["schema_version"] == 2
    assert out["params"] == {"k": model_store._param_entry(2.0), "d0": model_store._param_entry(0.05)}
    assert v1["params"] == {"k": 2.0, "d0": 0.05}                  # input not mutated
    assert model_store._normalize_to_v2(out) is out               # already v2 -> unchanged


def test_builder_param_row_preserves_and_defaults():
    """The builder's per-parameter row: 'auto' reproduces _nd_bounds, a custom (min,max) survives a
    re-detect, and _validate refuses a value outside its bounds."""
    from core.gui.screens.model_builder_screen import ModelBuilderScreen, _ParamRow
    _app()
    r = _ParamRow(0.05)                                            # auto -> placeholder box
    assert r.auto.isChecked() and r.spec() == (0.05, *model_store._nd_bounds(0.05))
    r.set_spec(0.05, 0.01, 0.1)                                    # a custom box turns auto off
    assert not r.auto.isChecked() and r.spec() == (0.05, 0.01, 0.1)

    mb = ModelBuilderScreen()
    mb.vars_edit.setText("x")
    mb._set_variables()
    mb._var_rows[0].drift.setText("-k*x")
    mb._var_rows[0].noise.setText("d0")
    mb.name_edit.setText("UMTESTROW")
    mb._detect_params()
    assert set(mb._param_fields) == {"k", "d0"}
    mb._param_fields["d0"].set_spec(0.05, 0.01, 0.1)
    mb._detect_params()                                            # re-detect preserves the custom box
    assert mb._param_fields["d0"].spec() == (0.05, 0.01, 0.1)
    mb._param_fields["k"].set_spec(5.0, 0.0, 1.0)                  # value outside its box
    assert mb._validate() is None and "outside its bounds" in mb.status.text()


def test_model_store_rejects_unusable_values_and_names():
    """Values/names that would persist a registered-but-unstreamable model must fail at save time:
    t_scale past the transient budget, non-finite numbers, and Windows reserved device names."""
    doc = _doc("UMTESTVAL")
    for t_scale, needle in ((2.0, "t_scale must be below"), (float("inf"), "finite"),
                            (float("nan"), "finite")):
        try:
            model_store.save_user_model({**doc, "rescale": {"x_scale": 10.0, "t_scale": t_scale}})
        except ValueError as e:
            assert needle in str(e), e
        else:
            raise AssertionError(f"t_scale={t_scale} not refused")
    try:
        model_store.save_user_model({**doc, "params": {"k1": float("nan"), "d0": 0.05}})
    except ValueError as e:
        assert "finite" in str(e)
    else:
        raise AssertionError("nan param not refused")
    for name in ("NUL", "con", "Com3", "lpt9"):
        try:
            model_store.validate_name(name)
        except ValueError as e:
            assert "reserved Windows device name" in str(e), e
        else:
            raise AssertionError(f"reserved device name {name} not refused")
    assert not (config.MODELS_PATH / "UMTESTVAL.json").exists()   # nothing persisted by the refusals


def test_stale_bounds_file_is_detected():
    """A hand-edited JSON whose param discovery order no longer matches the emitted Bounds file must
    raise a clear out-of-sync error instead of silently mis-binding values by position."""
    import json as _json
    from core.gui.panels import simulate_runner as sr
    name = "UMTESTSYNC"
    doc = {"schema_version": 1, "name": name,
           "variables": [{"name": "x", "drift": "mu*x - nu*x^3", "D": "d0", "init": 0.1,
                          "forcing": None}],
           "params": {"mu": 2.0, "nu": 5.0, "d0": 0.01},
           "rescale": {"x_scale": 10.0, "t_scale": 0.01}}
    try:
        model_store.save_user_model(doc)
        json_path = config.MODELS_PATH / f"{name}.json"
        stale = _json.loads(json_path.read_text(encoding="utf-8"))
        stale["variables"][0]["drift"] = "-nu*x^3 + mu*x"          # same math, swapped discovery order
        json_path.write_text(_json.dumps(stale), encoding="utf-8")
        registry.load_user_models()
        try:
            sr.build_stream_config(name, str(config.CELL_PATH / name.lower() / "default.txt"))
        except ValueError as e:
            assert "out of sync" in str(e), e
        else:
            raise AssertionError("stale bounds file not detected")
    finally:
        _remove_user_model(name)
        registry.load_user_models()


def test_registry_load_collects_errors_without_raising():
    """One valid + one corrupt file: the valid one registers, the corrupt one lands in load_errors --
    a bad file must never brick startup (the CrossValPanel launch-guard rule)."""
    name = "UMTESTOK"
    config.MODELS_PATH.mkdir(parents=True, exist_ok=True)
    corrupt = config.MODELS_PATH / "UMTESTBAD.json"
    try:
        model_store.save_user_model(_doc(name))
        corrupt.write_text("{ not json", encoding="utf-8")
        registry.load_user_models()
        assert name in config.VALID_MODELS
        assert any(p.name == "UMTESTBAD.json" for p, _ in registry.load_errors)
    finally:
        corrupt.unlink(missing_ok=True)
        _remove_user_model(name)
        registry.load_user_models()                                # leave a clean registry behind
    assert name not in config.VALID_MODELS


# ── the Simulate path, end to end ────────────────────────────────────────────────────────────────
def test_user_model_streams_through_the_simulate_path():
    """Save -> register -> build_stream_config/plan_stream/run_simulation_stream emit finite frames,
    and a blow-up model raises the divergence RuntimeError instead of flatlining."""
    from core.gui.panels import simulate_runner as sr
    sin = {"kind": "sin", "params": {"amp": 0.5, "freq": 10.0, "phase": 0.0, "offset": 0.0}}
    name = "UMTEST"
    boom = "UMTESTBOOM"
    try:
        model_store.save_user_model(_doc(name, sin))
        model_store.save_user_model({
            "schema_version": 1, "name": boom,
            "variables": [{"name": "x", "drift": "x^3", "D": "0", "init": 2.0, "forcing": None}],
            "params": {}, "rescale": {"x_scale": 10.0, "t_scale": 0.01}})
        registry.load_user_models()

        cfg = sr.build_stream_config(name, str(config.CELL_PATH / name.lower() / "default.txt"))
        assert cfg.state_dep_drift is False and cfg.labels == ["k1", "d0"]
        plan = sr.plan_stream(cfg, 0.2)
        assert plan.user_spec is registry.get(name) and plan.n_channels == 2
        chunks = []
        sr.run_simulation_stream(cfg, 0.2, frame_steps=500, fps=0.0, emit_chunk=chunks.append)
        data = np.concatenate(chunks, axis=0)
        assert data.shape[1] == 2 and np.isfinite(data).all() and data.shape[0] > 50
        assert abs(data[:, 1]).max() < 100.0                       # x_scale=10 * O(1) ND state

        cfg_boom = sr.build_stream_config(boom, str(config.CELL_PATH / boom.lower() / "default.txt"))
        try:
            sr.run_simulation_stream(cfg_boom, 0.2, frame_steps=500, fps=0.0, emit_chunk=lambda c: None)
        except RuntimeError as e:
            assert "diverged" in str(e)
        else:
            raise AssertionError("blow-up not detected")
    finally:
        _remove_user_model(name)
        _remove_user_model(boom)


# ── GUI: combo refresh + builder guards ──────────────────────────────────────────────────────────
def test_combo_refresh_preserves_picker_selections():
    """A user-model save/delete must NOT reset the cell/bounds pickers when the panel's model
    selection did not change (the model-changed hook resets pickers to their first entry)."""
    import tempfile
    from core.gui import settings as gui_settings
    from core.gui.main_window import MainWindow
    from core.gui.panels.fdt_panel import FdtPanel
    _app()
    ini = tempfile.NamedTemporaryFile(suffix=".ini", delete=False)
    ini.close()
    gui_settings.use_ini_file(ini.name)
    try:
        window = MainWindow()
        fdt = window.panel(FdtPanel)
        assert fdt.cell_picker.combo.count() > 1, "needs >1 nadrowski cells to be meaningful"
        fdt.cell_picker.combo.setCurrentIndex(1)
        chosen = fdt.cell_picker.combo.currentText()
        registry.register(registry.ModelSpec("UMTESTCOMBO", ["a"], is_user_model=True, n_vars=1))
        try:
            window._on_user_models_changed()
            assert fdt.cell_picker.combo.currentText() == chosen   # unchanged model -> untouched picker
            # A DELETED selected model must still fall back and re-fire the hook.
            fdt.model_combo.setCurrentText("UMTESTCOMBO")
        finally:
            registry.unregister("UMTESTCOMBO")
        window._on_user_models_changed()
        assert fdt.model_combo.currentText() == "NADROWSKI"
        window.close()
    finally:
        gui_settings.use_ini_file(None)


def test_builder_validate_refuses_while_a_task_runs():
    """The smoke integration writes tqdm frames to the process-wide redirected streams; it must not
    run on the GUI thread while a worker owns them."""
    from core.gui.panels.base_panel import BasePanel
    from core.gui.screens.model_builder_screen import ModelBuilderScreen
    _app()
    mb = ModelBuilderScreen()
    mb.vars_edit.setText("x")
    mb._set_variables()
    mb._var_rows[0].drift.setText("-x")
    mb.name_edit.setText("UMTESTGUARD")
    mb._detect_params()
    BasePanel._running = True
    try:
        assert mb._validate() is None
        assert "task is running" in mb.status.text()
    finally:
        BasePanel._running = False
    assert mb._validate() is not None                              # and works again once idle


# ── appearance: OS accent + Inter toggle ─────────────────────────────────────────────────────────
def test_accent_tokens_and_palette_override():
    from core.gui import design
    base = design.tokens(False)
    t = design.tokens(False, "#AA3366")
    assert t["accent"] == "#AA3366" and t is not base
    assert t["accent_hover"] != t["accent"] and t["accent_press"] != t["accent"]
    assert design.tokens(False, "not-a-colour") is base            # invalid -> fixed Fluent blue
    assert design.tokens(False, "#EEEEEE")["on_accent"] == "#1B1B1B"   # light accent -> dark CTA text
    assert design.tokens(False, "#112233")["on_accent"] == "#FFFFFF"

    pal = design.build_palette(False, "#AA3366")
    assert pal.color(QPalette.Highlight).name().upper() == "#AA3366"   # LOAD-BEARING (custom paint)
    assert pal.color(QPalette.Mid).name().upper() == base["mid"].upper()
    assert "#AA3366" in design.build_qss(True, "#AA3366")
    assert "#AA3366" not in design.build_qss(True)


def test_system_accent_returns_a_hex_or_none():
    from core.gui import design
    accent = design.system_accent()
    assert accent is None or re.fullmatch(r"#[0-9A-F]{6}", accent), accent


def test_load_app_font_prefers_inter_when_forced():
    from core.gui import fonts
    app = _app()
    saved = app.font()
    try:
        assert fonts.load_app_font(app, prefer_inter=True) == "Inter"   # bundled Inter always registers
    finally:
        app.setFont(saved)


# ── icon set (B-e) ─────────────────────────────────────────────────────────────────────────────────
def test_icons_register_or_fallback():
    """The bundled icon font registers (or degrades to None), and every semantic name has a real
    codepoint glyph AND a non-empty unicode fallback."""
    from core.gui import icons
    _app()
    fam = icons.register()
    assert fam is None or isinstance(fam, str)
    assert isinstance(icons.available(), bool)
    for name, (glyph_cp, fallback) in icons.NAMES.items():
        assert glyph_cp and fallback, name
        assert icons.glyph(name) in (glyph_cp, fallback)


def test_apply_icon_never_blank():
    """apply_icon leaves a button non-blank in BOTH branches: the icon glyph when the font is present,
    the unicode fallback when it is monkeypatched away."""
    from PySide6.QtWidgets import QToolButton
    from core.gui import icons
    _app()
    for name in icons.NAMES:
        b = QToolButton()
        icons.apply_icon(b, name)
        assert b.text(), name
    real = icons.available
    icons.available = lambda: False                              # force the fallback path
    try:
        for name, (_glyph, fallback) in icons.NAMES.items():
            b = QToolButton()
            icons.apply_icon(b, name)
            assert b.text() == fallback, (name, b.text())
    finally:
        icons.available = real


def test_migrated_glyph_buttons_render():
    """The four migrated buttons (nav back/settings, picker refresh, help badge) never render blank."""
    import tempfile
    from PySide6.QtWidgets import QPushButton
    from core.gui.screens.nav_shell import NavShell
    from core.gui.widgets.artifact_picker import ArtifactPicker
    from core.gui.widgets.help_badge import HelpBadge
    _app()
    ns = NavShell()
    assert ns.btn_back.text() and ns.btn_settings.text()
    ap = ArtifactPicker(tempfile.mkdtemp())
    refresh = ap.findChild(QPushButton, "iconButton")
    assert refresh is not None and refresh.text()
    assert HelpBadge("some help text").text()


if __name__ == "__main__":
    _app()
    failures = 0
    for test_name, fn in sorted(globals().items()):
        if test_name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS  {test_name}")
            except AssertionError as e:
                failures += 1
                print(f"FAIL  {test_name}\n      {e}")
    print(f"\n{'ALL PASSED' if not failures else f'{failures} FAILURE(S)'}")
    raise SystemExit(1 if failures else 0)
