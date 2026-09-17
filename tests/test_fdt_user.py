"""FDT-for-user-models unit tests (FEATURE 1 v3 + B-d), Qt-free.

Locks down the generalized effective-temperature normalization and its gate:
  * campaigns.observable_noise_prefactor computes the per-model coupling/D_x -- n*beta for NADROWSKI,
    2/sigma_x^2 for HOPF, 2*tau_hb/eta_hb^2 for BP (experimental), 1/D_0 for an additive-noise user
    model -- and raises FDTModelError for a user model whose observable noise is multiplicative, zero,
    or negative.
  * registry.fdt_support admits the built-ins and additive-noise user models; it rejects user models
    with multiplicative / zero observable noise or intrinsic forcing.
  * campaigns._n_force_channels returns one channel per state variable for a user model.

No cell files / QApplication needed: a tiny fake cfg supplies only .model / .params_dict (and, for the
force-channel test, .inits_tensor / .force_params_dict), which is all these functions read.

Run:  pytest tests/test_fdt_user.py
"""
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch                                                       # noqa: E402

from core import registry                                         # noqa: E402
from core.FDT.campaigns import (observable_noise_prefactor, _n_force_channels,   # noqa: E402
                                FDTModelError)
from core.Models.user_model import parse_user_model               # noqa: E402


class _FakeCfg:
    """The subset of FDTConfig the FDT model helpers read."""
    def __init__(self, model, params=None, n_vars=None, force_params=None):
        self.model = model
        self.params_dict = {k: (v, None) for k, v in (params or {}).items()}   # name -> (value, bounds)
        if n_vars is not None:
            self.inits_tensor = torch.zeros(1, n_vars)
        self.force_params_dict = force_params or {}


def _register_user(name, variables):
    compiled = parse_user_model(variables)
    registry.register(registry.ModelSpec(
        name=name, labels=list(compiled.param_names), state_dep_drift=compiled.state_dep_noise,
        is_user_model=True, n_vars=len(compiled.var_names), variables=variables, compiled=compiled))
    return compiled


def test_observable_noise_prefactor_builtins():
    """The built-in prefactors reduce to the coded per-model formulas (Nadrowski = the historical value)."""
    assert observable_noise_prefactor(_FakeCfg("NADROWSKI", {"n": 50.0, "beta": 14.1})) == 50.0 * 14.1
    assert observable_noise_prefactor(_FakeCfg("HOPF", {"sigma_x": 0.1})) == 2.0 / 0.1 ** 2
    # BP (experimental): coupling 1/tau_hb, so prefactor = 2*tau_hb/eta_hb^2
    assert observable_noise_prefactor(_FakeCfg("BP", {"tau_hb": 1.0, "eta_hb": 0.5})) == 2.0 * 1.0 / 0.5 ** 2
    # a missing FDT parameter is a readable FDTModelError, not a bare KeyError
    try:
        observable_noise_prefactor(_FakeCfg("HOPF", {}))
    except FDTModelError as e:
        assert "sigma_x" in str(e), e
    else:
        raise AssertionError("missing HOPF param not reported")


def test_observable_noise_prefactor_user_additive():
    """An additive-noise user model's prefactor is 1/D_0 evaluated at the cell param values."""
    name = "FDTUADD"
    try:
        _register_user(name, [{"name": "x", "drift": "-k*x", "D": "Dx"}])
        pref = observable_noise_prefactor(_FakeCfg(name, {"k": 1.0, "Dx": 0.5}))
        assert pref == 1.0 / 0.5
    finally:
        registry.unregister(name)


def test_observable_noise_prefactor_user_rejects_multiplicative_zero_negative():
    name = "FDTUBAD"
    # multiplicative observable noise -> refused
    try:
        _register_user(name, [{"name": "x", "drift": "-k*x", "D": "0.5*x^2"}])
        try:
            observable_noise_prefactor(_FakeCfg(name, {"k": 1.0}))
        except FDTModelError as e:
            assert "multiplicative" in str(e), e
        else:
            raise AssertionError("multiplicative observable noise not refused")
    finally:
        registry.unregister(name)
    # zero and negative constant D_0 -> refused at runtime (D0 <= 0)
    try:
        _register_user(name, [{"name": "x", "drift": "-x", "D": "d0"}])
        for d0 in (0.0, -1.0):
            try:
                observable_noise_prefactor(_FakeCfg(name, {"d0": d0}))
            except FDTModelError as e:
                assert "non-positive" in str(e) or "zero" in str(e), e
            else:
                raise AssertionError(f"D0={d0} not refused")
    finally:
        registry.unregister(name)


def test_fdt_support_gate():
    """Built-ins are supported; user models are gated on additive, non-zero, unforced observable noise."""
    for m in ("NADROWSKI", "HOPF", "BP"):
        assert registry.fdt_support(m) == (True, ""), m
    assert registry.fdt_support("NOPE")[0] is False

    cases = [
        ("FDTGOK",  [{"name": "x", "drift": "-k*x", "D": "d0"}],                       True,  ""),
        ("FDTGMUL", [{"name": "x", "drift": "-k*x", "D": "0.5*x^2"}],                  False, "multiplicative"),
        ("FDTGZ",   [{"name": "x", "drift": "-x", "D": "0"}],                          False, "deterministic"),
        ("FDTGF",   [{"name": "x", "drift": "-k*x", "D": "d0",
                      "forcing": {"kind": "sin", "params": {}}}],                      False, "forcing"),
    ]
    for name, variables, ok, needle in cases:
        try:
            _register_user(name, variables)
            got_ok, reason = registry.fdt_support(name)
            assert got_ok is ok, (name, got_ok, reason)
            assert needle in reason, (name, reason)
        finally:
            registry.unregister(name)


def test_n_force_channels_user_vs_builtin():
    """A user model needs one force channel per state variable; built-ins keep the 1-or-2 convention."""
    name = "FDTNCH"
    try:
        _register_user(name, [{"name": "x", "drift": "v", "D": "0"},
                              {"name": "v", "drift": "-k*x", "D": "d0"}])
        assert _n_force_channels(_FakeCfg(name, n_vars=2)) == 2
    finally:
        registry.unregister(name)
    assert _n_force_channels(_FakeCfg("BP", force_params={})) == 1
    assert _n_force_channels(_FakeCfg("HOPF", force_params={"amp_y": 0.0})) == 2


def test_the_fdt_messages_are_records_with_their_own_levels(tmp_path, monkeypatch, caplog, capsys):
    """V4 in the FDT pipeline and the sweep study (piece 3). Banners, per-point progress, the sanity
    table and the saved-plot paths are information; a failed sanity verdict and a plot that had to
    drop non-finite points are warnings; a Campaign-2 point that FAILED and was recorded rather than
    raised is an error. Until piece 3 all of it was print(): the only severity marker was a hand-typed
    "WARNING:" word, and the per-point failure of an overnight sweep scrolled past at the level of its
    progress lines.

    Stubbed at the campaign seams, so nothing is simulated: two operating points, the first failing in
    Campaign 2. The sweep still completes, logs one error, warns about the count (a Python warning,
    unchanged) and says so. The hand-typed "WARNING: " word is gone from the sanity verdict because
    the level carries it -- on the tool it would have read "warning: WARNING: ...". No record starts or
    ends with a blank line, and nothing reaches stdout: the front ends' handlers put records there
    (Task 16)."""
    import logging

    import numpy as np
    import pytest

    from core.FDT import cross_validation as cv
    from core.FDT import fdt_pipeline, plots, sanity

    assert logging.getLogger("core").level == logging.INFO, "core/runs.py sets it at import"

    def records():
        return [(r.name, r.levelname, r.getMessage()) for r in caplog.records if r.name.startswith("core")]

    # ── the sweep study: information progress, one ERROR per failed point ─────────────────────────
    class _SweepCfg:
        model, n_freqs, ensemble_M, F0, psd_T_obs_nd = "STUB", 4, 2, 0.1, 1.0

        def with_overrides(self, **kw):
            return self

    calls = []

    def _campaign2(cfg_op, omegas, freqs_psd, G):
        calls.append(cfg_op)
        if len(calls) == 1:
            raise RuntimeError("stub out of memory")
        return torch.ones(4, dtype=torch.complex128), torch.full((4,), 2.0, dtype=torch.float64)

    monkeypatch.setattr(cv, "run_campaign1_psd",
                        lambda c: (torch.linspace(0.1, 3.0, 8, dtype=torch.float64),
                                   torch.ones(8, dtype=torch.float64)))
    monkeypatch.setattr(cv, "_detect_resonance", lambda omegas, G, w0: (1.0, True))
    monkeypatch.setattr(cv, "_build_common_grid",
                        lambda cfg, w0s, res: (torch.linspace(0.5, 2.0, 4, dtype=torch.float64), 1.0))
    monkeypatch.setattr(cv, "_campaign2_ratio", _campaign2)
    out_h5 = tmp_path / "s.h5"
    caplog.clear()
    with pytest.warns(UserWarning, match="1/2 operating points failed"):
        cv.run_fdt_param_sweep(_SweepCfg(), "s", np.array([0.0, 0.1]), {"temp": 1.0}, output_path=out_h5)
    got = records()
    name = "core.FDT.cross_validation"
    assert (name, "INFO", "--- Phase A (s sweep): spontaneous PSD + omega_0 detection ---") in got, got
    assert (name, "INFO", "  [A 1/2] s=0 (temp=1.0)") in got, got
    assert (name, "INFO", "      omega_0 = 1.0000") in got, got
    assert (name, "INFO", "Common grid: 4 pts spanning [0.5000, 2.0000] (omega_0_ref=1.0000)") in got, got
    assert (name, "ERROR", "      Campaign 2 FAILED: stub out of memory") in got, got
    assert (name, "INFO", "      T_eff/T peak = 2") in got, got
    assert got[-1] == (name, "INFO", f"s sweep complete (1/2 points). Saved to: {out_h5}"), got
    assert [lvl for _, lvl, _ in got].count("ERROR") == 1, got

    # ── the FDT run: a failed sanity verdict is a WARNING, without the hand-typed word ─────────────
    class _Hopf:
        model = "HOPF"

    monkeypatch.setattr(fdt_pipeline, "_out_dir", lambda: tmp_path)
    monkeypatch.setattr(fdt_pipeline, "run_all_sanity",
                        lambda cfg, passive_plot_path=None: {"linearity": (False, {"ratio": 0.5})})
    caplog.clear()
    fdt_pipeline.run_fdt(_Hopf(), skip_sanity=False, confirm_production=False)
    name = "core.FDT.fdt_pipeline"
    assert records() == [
        (name, "INFO", "Cell file natural-frequency estimate: omega_0 ~= 1.0 (ND Hopf natural frequency)"),
        (name, "WARNING", "One or more sanity checks failed (see metrics above)."),
        (name, "INFO", "Aborted by user."),
    ], records()

    # ── the sanity table: information, a FAIL row included, no blank-line records ──────────────────
    monkeypatch.setattr(sanity, "check_linearity", lambda c: (True, {"x": 1}))
    monkeypatch.setattr(sanity, "check_ensemble_convergence", lambda c: (False, {"x": 2}))
    monkeypatch.setattr(sanity, "check_psd_window", lambda c: (True, {"x": 3}))
    caplog.clear()
    res = sanity.run_all_sanity(_Hopf())
    assert res == {"linearity": (True, {"x": 1}), "ensemble_convergence": (False, {"x": 2}),
                   "psd_window": (True, {"x": 3})}
    got = records()
    assert {(n, lvl) for n, lvl, _ in got} == {("core.FDT.sanity", "INFO")}, got
    msgs = [m for _, _, m in got]
    assert len(msgs) == 15 and msgs[1] == "FDT Sanity Checks" and msgs[-1] == "=" * 60, msgs
    assert "  FAIL  metrics: {'x': 2}" in msgs and "=" * 60 + "\nSummary:" in msgs, msgs
    assert not any(m.startswith("\n") or m.endswith("\n") for m in msgs), msgs

    # ── the plot helpers: dropped non-finite points are WARNINGs ───────────────────────────────────
    caplog.clear()
    plots.plot_psd(np.array([1.0, 2.0, 3.0]), np.array([1.0, np.nan, 1.5]), save_path=tmp_path / "psd.png")
    plots.plot_eff_temp_ratio(np.array([1.0, 2.0, 3.0]), np.array([1.0, np.inf, 1.2]),
                              save_path=tmp_path / "ratio.png")
    assert records() == [
        ("core.FDT.plots", "WARNING", "plot_psd: dropping 1/3 non-finite PSD points."),
        ("core.FDT.plots", "WARNING", "plot_eff_temp_ratio: dropping 1/3 non-finite (NaN/inf) points."),
    ], records()

    assert capsys.readouterr().out == ""
