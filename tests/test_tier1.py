"""The tier-1 path: the box on which temperature is inferred and the force scale is derived from it.

``master_tier1.txt`` declares ``T`` in the rescale block where other boxes declare ``f_scale``. The
network is trained on T, and every simulation runs at the force scale T implies,

    f_scale = n * beta * k_B * T / x_scale,

which is 46.99 pN at the tier-1 cell (n=50, beta=14.1, x_scale=62.14 nm, T=300 K, with k_B in
pN*nm/K). The inferred index names ``T`` and no ``f_scale``, and a drive builder handed it raises
``RuntimeError`` rather than build a drive; a Hopf-style index, which names neither, still falls back
to ``x_scale / t_scale`` -- 16.66 at the same cell -- so a drive recorded without a declared force scale
means a caller handed a builder neither name. The retrain trains on this box, so every place a
simulation is set up from a draw is pinned here: the training rows, the calibration set, the Fisher
operating points, a simulated observation and its predictive checks, the two simulating
identifiability diagnostics, the Simulate panel's live runner, a narrowing round, and loading a prior
built on the master box. So is what the records say about it: the relation and the assumed parameters
in every record's config, and the force scale a simulated observation ran at.

Cheap by design: every fast test but the live runner's replaces ``pipeline.gen_obs`` with
``stand_in_gen_obs`` and records the force scale each drive is divided by with ``force_scale_spy``;
recordings are one to two seconds and batches are two of four rows. The live runner solves five
milliseconds for real. One slow test runs the real ``smoke`` chain on the tier-1 box.
"""
import re
import warnings
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch
from matplotlib import pyplot as plt
from sbi.utils import BoxUniform

from core import cli, config, orchestrator, registry
from core.SBI import pipeline, reparam, truncate
from core.SBI import training_checkpoint as tc
from core.SBI.run_guards import _log_params_for
from core.sim_config import SimConfig
from tests._fixtures import (_FakeDP, _nad_cfg, _prior_artifact, _tiny_nadrowski_gen_prior,
                             stub_calibration_battery)

TIER1_BOX = config.BOUNDS_PATH / "nadrowski" / "master_tier1.txt"
TIER1_CELL = config.CELL_PATH / "nadrowski" / "master_spont_tier1.txt"


def _close(title, fig):
    """A figure sink that closes what it is handed, so no test leaks a figure."""
    plt.close(fig)


def tier1_cfg(*, chi: bool = True, hw=None) -> SimConfig:
    """A real NADROWSKI SimConfig off the tier-1 box, bounds only (no cell), on the CPU unless ``hw``
    names another device. ``chi`` picks chi mode or, off, forced mode (the box has a Forcing section)."""
    labels = config.VALID_LABELS[config.VALID_MODELS.index("NADROWSKI")]
    return cli.make_sim_config("NADROWSKI", labels, registry.state_dep_drift("NADROWSKI"), str(TIER1_BOX),
                               chi_mode=chi, hw=hw or config.cpu_device())


def with_truth(cfg, *, amp: float = 3.0, freq: float = 0.05) -> SimConfig:
    """``cfg`` with the tier-1 cell's truth and inits loaded, and a real drive in place of the cell's
    all-zero one: a forced run needs a drive for the force scale to reach the trace at all. ``amp`` is
    in pN and ``freq`` in kHz, both inside the box's forcing bounds. Returns ``cfg``."""
    cli.load_and_validate_gt(cfg, str(TIER1_CELL))
    for name, value in (("amp", amp), ("freq", freq)):
        cfg.force_params_dict[name] = (value, cfg.force_params_dict[name][1])
    return cfg


def derived_force_scale(cfg) -> float:
    """The force scale the tier-1 relation gives at ``cfg``'s loaded truth, n * beta * k_B * T /
    x_scale in cell force units, read by name. 46.99 at the tier-1 cell."""
    nd, rescale = cfg.params_dict, cfg.rescale_params
    return nd["n"][0] * nd["beta"][0] * cfg.k_b_cell * rescale["T"][0] / rescale["x_scale"][0]


def twin_cfg(cfg):
    """The master-box twin of a tier-1 config with its truth loaded: the same cell, with the force
    scale the tier-1 relation derives declared as f_scale."""
    twin = _nad_cfg(chi_mode=cfg.chi_mode)
    twin.inject_ground_truth(
        dict(cfg.inits_dict), {k: v for k, (v, _) in cfg.params_dict.items()},
        {"x_scale": cfg.rescale_params["x_scale"][0], "t_scale": cfg.rescale_params["t_scale"][0],
         "f_scale": derived_force_scale(cfg)},
        {k: v for k, (v, _) in cfg.force_params_dict.items()})
    return twin


def _res(c):
    """``c``'s truth rescale block, (n_rescale,), in the box's own order."""
    return torch.tensor([v for v, _ in c.rescale_params.values()], dtype=c.hw.dtype)


def _jac_ctx(cfg, n_obs):
    """The identifiability jacobian's context for ``cfg``, built as ``identifiability_jacobian``
    builds it: the probe multipliers in chi mode, the cell's own drive otherwise."""
    from core import forcing
    from core.diagnostics import feature_sets, identifiability
    from core.SBI import chi
    return identifiability._JacCtx(
        cfg=cfg, n_obs=n_obs,
        mults=chi.chi_multipliers_for(cfg) if cfg.chi_mode else None,
        forcing_gt=(None if cfg.chi_mode else torch.tensor(
            [[v for v, _ in cfg.force_params_dict.values()]], dtype=cfg.hw.dtype)),
        keep_idx=feature_sets.summary_keep_idx(), feat_labels=feature_sets.feature_labels(cfg),
        n_force_ch=forcing.n_force_channels(cfg.model, cfg.forcing_idx, cfg.inits_tensor.shape[-1]))


def _narrow_prior(cfg, rel: float = 1e-4):
    """A physical box of relative half-width ``rel`` around the loaded truth, so every draw is the
    truth to about four figures and every simulation's force scale is known in advance. Every truth
    value of the tier-1 cell is positive, so the box is never inverted."""
    v = torch.tensor(cfg.ground_truth)
    return BoxUniform(v * (1 - rel), v * (1 + rel))


def force_scale_spy(monkeypatch) -> list:
    """Record the force scale every drive is divided by. Every sinusoidal and user drive goes
    through forcing.build_nondim_force_tensor. One record per call:
    {"declared": the index names f_scale, "f_scale": (B,) float64 on the CPU, "index": the index}.
    When nothing is declared, the record holds x_scale / t_scale, the builder's fallback."""
    from core import forcing
    real, seen = forcing.build_nondim_force_tensor, []

    def spy(forcing_params, t_nd, rescale_params, forcing_idx, rescale_idx, *a, **k):
        if "f_scale" in rescale_idx:
            f = rescale_params[:, rescale_idx["f_scale"]]
        else:
            f = rescale_params[:, rescale_idx["x_scale"]] / rescale_params[:, rescale_idx["t_scale"]]
        seen.append({"declared": "f_scale" in rescale_idx, "index": dict(rescale_idx),
                     "f_scale": f.detach().double().cpu().clone()})
        return real(forcing_params, t_nd, rescale_params, forcing_idx, rescale_idx, *a, **k)

    monkeypatch.setattr(forcing, "build_nondim_force_tensor", spy)
    return seen


def stand_in_gen_obs(model=None, params=None, t=None, inits=None, force=None, n_segs=None,
                     steady_idx=0, fixed_dict=None, state_dep_drift=False, batch_size=1,
                     var_idx=None, dtype=torch.float32, device=torch.device("cpu")):
    """A cheap stand-in for pipeline.gen_obs with its shape: the drive's first channel as the
    response, a unit oscillation at angular frequency 1 in ND time (so the passive trace has a
    spectral peak), and a little noise from torch's global stream."""
    n = t.shape[0] - steady_idx
    x = (force[:, 0, steady_idx:].to(dtype).expand(batch_size, n) + torch.sin(t[steady_idx:]).to(dtype)
         + 0.01 * torch.randn(batch_size, n, dtype=dtype, device=device))
    out = torch.zeros((1 if var_idx is not None else inits.shape[-1], batch_size, n), dtype=dtype,
                      device=device)
    out[0] = x
    return out


def _assert_every_drive_used_the_derived_scale(spy, f, rtol=2e-3):
    assert spy, "no drive was built"
    for e in spy:
        assert e["declared"] and "T" not in e["index"], e["index"]
        assert torch.allclose(e["f_scale"], torch.full_like(e["f_scale"], f), rtol=rtol), e["f_scale"]


_MASKED_PROBES = re.compile(r"chi: \d+/\d+ probes masked")


@contextmanager
def _only_masked_probe_warnings():
    """Capture every Python warning raised inside and check each one on exit. At one-to-two-second
    recordings a chi batch may mask a probe too short to lock in, and it says so with a count. Whether
    a given batch does depends on its unseeded (t_scale, T_obs) draw, so zero or more such lines are
    allowed, and nothing else is. Yields the captured list."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        yield caught
    others = [f"{w.category.__name__}: {w.message}" for w in caught
              if not (w.category is UserWarning and _MASKED_PROBES.search(str(w.message)))]
    assert not others, others


def test_the_tier1_box_parses_to_temperature_in_place_of_the_force_scale():
    """The tier-1 box declares T where other boxes declare f_scale, which switches the derivation on;
    the simulator's index renames T's column to f_scale, and the config carries what the relation
    needs (the ND index, for n and beta by name, and k_B in cell units). The master box has none of it."""
    from core.SBI import derived
    cfg = tier1_cfg()
    assert list(cfg.rescale_params) == ["x_scale", "t_scale", "T"]
    assert derived.uses_derived_f_scale(cfg.rescale_idx)
    assert cfg.sim_rescale_idx == {"x_scale": 0, "t_scale": 1, "f_scale": 2}
    nd_idx, k_b = cfg.tier1_args
    assert nd_idx == cfg.nd_idx and {"n", "beta"} <= set(nd_idx)
    assert k_b == pytest.approx(1.380649e-2, rel=1e-9)
    assert cfg.observation_mode == "chi" and tier1_cfg(chi=False).observation_mode == "forced"
    assert derived_force_scale(with_truth(tier1_cfg())) == pytest.approx(46.99, abs=0.01)
    master = _nad_cfg()
    assert master.tier1_args == (None, None) and master.sim_rescale_idx == master.rescale_idx


@pytest.mark.parametrize("chi", [True, False])
def test_training_rows_keep_temperature_while_every_simulation_gets_the_derived_force_scale(monkeypatch, chi):
    """The training target records the inferred temperature, and every drive the batch builds -- each
    chi probe, or the forced run -- is divided by the force scale that temperature implies. The only
    warning either mode may raise is the chi batch's masked-probe count; the forced run draws no probes,
    so it raises none."""
    cfg = with_truth(tier1_cfg(chi=chi))
    monkeypatch.setattr(pipeline, "gen_obs", stand_in_gen_obs)
    spy = force_scale_spy(monkeypatch)
    with _only_masked_probe_warnings() as caught:
        x, theta = pipeline.gen_training_data(
            cfg.model, _narrow_prior(cfg), orchestrator.build_forcing_prior(cfg), cfg.t, 4, 2,
            cfg.steady_idx, cfg.dt_nd_min, len(cfg.params_dict), cfg.forcing_idx, cfg.rescale_idx,
            dt_exp=cfg.dt_exp, t_min_exp=cfg.t_min_exp, t_max_exp=2 * cfg.t_min_exp,
            t_scale_bounds=cfg.t_scale_bounds, chi_mode=chi, chi_f0=cfg.chi_f0,
            chi_freq_bounds=cfg.chi_freq_bounds, chi_k_pad=cfg.chi_k_pad, chi_max_cycles=cfg.chi_max_cycles,
            n_vars=3, nd_idx=cfg.tier1_args[0], k_b_cell=cfg.tier1_args[1],
            dtype=cfg.hw.dtype, device=cfg.hw.device)
    assert chi or not caught, [str(w.message) for w in caught]
    i_T = len(cfg.params_dict) + cfg.rescale_idx["T"]
    assert theta.shape[1] == 13 and theta.shape[0] > 0
    assert torch.allclose(theta[:, i_T], torch.full_like(theta[:, i_T], 300.0), rtol=1e-3)
    _assert_every_drive_used_the_derived_scale(spy, derived_force_scale(cfg))


def test_the_calibration_set_simulates_at_the_derived_force_scale(monkeypatch):
    """The calibration set is drawn through the same generator as training, so its truths keep the
    temperature and its probes are driven at the derived force scale. Its only warning may be the chi
    batch's masked-probe count."""
    cfg = with_truth(tier1_cfg())
    cfg.t_max_exp = 2 * cfg.t_min_exp                      # a test config: keep the recordings short
    monkeypatch.setattr(pipeline, "gen_obs", stand_in_gen_obs)
    spy = force_scale_spy(monkeypatch)
    with _only_masked_probe_warnings():
        x_cal, theta = orchestrator._draw_calibration_set(
            cfg, _narrow_prior(cfg), None, orchestrator.build_forcing_prior(cfg), n_cal=8, cal_n_scales=2)
    i_T = len(cfg.params_dict) + cfg.rescale_idx["T"]
    assert torch.allclose(theta[:, i_T], torch.full_like(theta[:, i_T], 300.0), rtol=1e-3)
    _assert_every_drive_used_the_derived_scale(spy, derived_force_scale(cfg))


@pytest.mark.parametrize("chi", [True, False])
def test_the_fisher_operating_points_simulate_at_the_derived_force_scale(monkeypatch, chi):
    """The rotation's Fisher is built over the simulated experiment: its anchor is driven at the
    truth's derived force scale, and its central-difference arms, which move n, beta, x_scale and T by
    a latent step, stay near it -- never at the fallback, and never at the temperature itself. In chi
    mode every probe is such a drive, the anchor's first; in forced mode, the cell's own drive."""
    from core.SBI import decorrelate
    cfg = with_truth(tier1_cfg(chi=chi))
    cfg.T_obs = 1000.0                                      # cell units: 1 s at dt_exp = 1 ms
    monkeypatch.setattr(pipeline, "gen_obs", stand_in_gen_obs)
    spy = force_scale_spy(monkeypatch)
    V = decorrelate.build_latent_fisher_rotation(cfg, m=2, n_points=1)
    f = derived_force_scale(cfg)
    assert tuple(V.shape) == (13, 13)
    assert all(e["declared"] and "T" not in e["index"] for e in spy)
    assert torch.allclose(spy[0]["f_scale"], torch.full_like(spy[0]["f_scale"], f), rtol=1e-4)  # the anchor
    every = torch.cat([e["f_scale"] for e in spy])       # the +-dz arms move n, beta, x_scale, T a little
    assert bool(((every > 0.5 * f) & (every < 2.0 * f)).all())


@pytest.mark.parametrize("chi", [True, False])
def test_a_simulated_observation_and_its_predictive_checks_use_the_derived_force_scale(store, monkeypatch,
                                                                                         chi):
    """A simulated observation is driven at the truth's derived force scale, and so is every drive
    the inference builds from its posterior draws: the predictive-check bin's, and in forced mode the
    eye test's two central trajectories (in chi mode those run undriven, and the bin's drives are its
    probes). The posterior is a stand-in whose every draw is the truth. The only warning either mode may
    raise is a chi probe set's masked-probe count."""
    cfg = with_truth(tier1_cfg(chi=chi))
    cfg.T_obs = 1000.0                                      # cell units: 1 s at dt_exp = 1 ms
    monkeypatch.setattr(pipeline, "gen_obs", stand_in_gen_obs)
    spy = force_scale_spy(monkeypatch)
    f = derived_force_scale(cfg)
    with _only_masked_probe_warnings() as caught:
        obs = orchestrator.generate_observations(cfg, fig_sink=_close, name="tier1_obs")
        _assert_every_drive_used_the_derived_scale(spy, f, rtol=1e-5)
        spy.clear()

        gt = cfg.ground_truth_tensor
        post = SimpleNamespace(
            posterior=SimpleNamespace(x_obs_digest=None, truncation=None, T=None,
                                      sample=lambda shape, x=None, **k: gt.expand(shape[0], -1).clone()),
            manifest=SimpleNamespace(body={"mode": obs.mode, "conditioning": {"width": obs.width}}),
            id="post", name="", accepted=[])
        monkeypatch.setattr(orchestrator, "pairplot", lambda *a, **k: (plt.figure(), None))
        monkeypatch.setattr(orchestrator, "_emit_overlay_figures", lambda *a, **k: None)
        orchestrator.infer_and_visualize(cfg, post, obs, n_samples=4, fig_sink=_close)
    assert chi or not caught, [str(w.message) for w in caught]
    # Four draws fill one predictive-check bin (config.PPC_BIN_SIZE is 50). Forced: that bin's one
    # drive plus one drive each for the eye test's mean and median trajectories, 1 + 2 = 3. Chi: that
    # bin's probe set, one drive per probe at the observation's frequencies (cfg.chi_n_freqs is 6), and
    # the eye test's trajectories run undriven, 1 * 6 + 0 = 6.
    assert len(spy) == (6 if chi else 3), [e["index"] for e in spy]
    _assert_every_drive_used_the_derived_scale(spy, f, rtol=1e-5)


def test_a_narrowing_round_on_a_tier1_parent_judges_the_truth_in_inferred_coordinates(store, monkeypatch,
                                                                                       caplog):
    """A narrowing round on the tier-1 box keeps the derivation (its plan carries n and beta by name
    and k_B in cell units, and it announces the derived force scale), and it judges the truth against
    its region in the inferred coordinates, temperature included: no warning when the truth is inside,
    one warning naming the temperature's direction when it is outside, and each child records which
    of the two it was, with the temperature's latent coordinate."""
    cfg = with_truth(tier1_cfg(chi=False))
    cfg.reparam_rotate = False
    cfg.hw.batch_size = 4
    lp = store.load_prior(cfg, _prior_artifact(store, cfg, name="p").id)
    T = reparam.build_inferred_bijection(cfg, log_params=_log_params_for(cfg))
    i_T = 12
    c = float(T.inv(cfg.ground_truth_tensor.reshape(1, -1))[0, i_T])

    def region(lo, hi):
        return truncate.TruncationRegion([i_T], [lo], [hi], n_latent=13, V=None,
                                         probe=tc.bijection_probe(T, 13, device=cfg.hw.device),
                                         x_obs_digest="d" * 16)

    plans = []

    def fake_train_nn(plan, **kw):
        plans.append(plan)
        dp = _FakeDP()
        dp.prior = kw["prior"]                    # the prior wrapper build_posterior passed in
        return dp, {"training_loss": [1.0], "validation_loss": [1.0], "best_validation_loss": 1.0,
                    "epochs_trained": 1, "stop_after_epochs": 1}

    monkeypatch.setattr(orchestrator.pipeline, "train_nn", fake_train_nn)
    monkeypatch.setattr(orchestrator.pipeline, "gen_training_data",
                        lambda plan, **kw: (torch.zeros(8, 50), torch.zeros(8, 13)))
    budget = dict(num_runs=2, run_size_cap=4, hidden_features=8, num_transforms=1, stop_after_epochs=1,
                  checkpoint_every=0, fig_sink=_close)

    caplog.set_level("INFO", logger="core")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        inside = orchestrator.build_posterior(cfg, lp, None, True, truncation=region(c - 0.5, c + 0.5), **budget)
    assert not any("GROUND TRUTH" in str(w.message) for w in caught)
    assert plans[0].nd_idx == cfg.nd_idx and plans[0].k_b_cell == pytest.approx(cfg.k_b_cell)
    assert any(r.getMessage().startswith("[tier1] f_scale is DERIVED") for r in caplog.records)
    cont = store.get("posterior", inside.id).body["training"]["truth_containment"]
    assert [(e["direction"], e["inside"]) for e in cont] == [(12, True)], cont
    assert cont[0]["value"] == pytest.approx(c, abs=1e-6), cont
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        outside = orchestrator.build_posterior(cfg, lp, None, True, truncation=region(c + 1.0, c + 2.0), **budget)
    truth = [w for w in caught if "GROUND TRUTH" in str(w.message)]
    assert len(truth) == 1 and "direction 12" in str(truth[0].message)
    cont = store.get("posterior", outside.id).body["training"]["truth_containment"]
    assert [(e["direction"], e["inside"]) for e in cont] == [(12, False)], cont
    # The region is judged in inferred coordinates: the same truth with its temperature replaced by
    # the force scale it implies is not a point of the box at all.
    sim_truth = cfg.ground_truth_tensor.clone()
    sim_truth[i_T] = derived_force_scale(cfg)
    assert orchestrator._truth_outside_region(T, region(c - 0.5, c + 0.5), sim_truth) != []


def test_a_prior_built_on_the_master_box_loads_under_the_tier1_box(store):
    """The two boxes share their ND block, so a prior fitted on the master box loads under the tier-1
    box by design, and its rescale draws then follow the tier-1 box: temperature inside (280, 310)."""
    cfg = tier1_cfg(chi=False)
    lp = store.load_prior(cfg, _prior_artifact(store, _nad_cfg(), name="master_prior").id)
    draws = lp.prior.sample((64,))
    t_col = draws[:, len(cfg.params_dict) + cfg.rescale_idx["T"]]
    assert tuple(draws.shape) == (64, 13) and bool(((t_col >= 280.0) & (t_col <= 310.0)).all())


def test_for_simulation_derives_the_force_scale_on_a_tier1_box_and_passes_every_other_box_through():
    """On the tier-1 box the simulator's block carries the derived force scale in T's column, under an
    index that names it f_scale, and the inferred block it came from is left as it was. On the master
    box both come back as they went in."""
    from core.SBI import derived
    cfg = with_truth(tier1_cfg(chi=False))
    res = _res(cfg).unsqueeze(0)
    sim, idx = derived.for_simulation(cfg, cfg.params_tensor, res)
    assert idx == {"x_scale": 0, "t_scale": 1, "f_scale": 2}
    assert float(sim[0, 2]) == pytest.approx(derived_force_scale(cfg), rel=1e-5)
    assert torch.equal(sim[:, :2], res[:, :2]) and float(res[0, 2]) == 300.0   # the input is not written
    master = _nad_cfg()
    cli.load_and_validate_gt(master, str(config.CELL_PATH / "nadrowski" / "master_weak.txt"))
    res_m = _res(master).unsqueeze(0)
    sim_m, idx_m = derived.for_simulation(master, master.params_tensor, res_m)
    assert sim_m is res_m and idx_m == master.rescale_idx


def test_a_force_builder_handed_a_temperature_index_raises_before_anything_is_simulated(monkeypatch):
    """An index that names T and no f_scale is a programming error: both builders raise RuntimeError
    (never a refusal), naming the index and the relation, before a simulation is spent. The probe
    builder's own guard fires first, before its peak search, so the drive builder it calls later is
    never what raises. A Hopf-style index, which names neither, keeps its x_scale / t_scale form."""
    from core import forcing
    from core.refusals import Refusal
    from core.SBI import chi, chi_probes
    idx = {"x_scale": 0, "t_scale": 1, "T": 2}
    fp, fidx = torch.tensor([[1.0, 0.05, 0.0, 0.0]]), {"amp": 0, "freq": 1, "phase": 2, "offset": 3}
    t, res = torch.linspace(0.0, 1.0, 16), torch.tensor([[62.14, 3.73, 300.0]])
    relation = r"f_scale = N \* beta \* k_B \* T / x_scale"
    with pytest.raises(RuntimeError, match=relation) as e:
        forcing.build_nondim_sin_force_tensor(fp, t, res, fidx, idx)
    assert "'T'" in str(e.value) and not isinstance(e.value, Refusal)
    monkeypatch.setattr(pipeline, "gen_obs", lambda *a, **k: pytest.fail("a simulation ran"))
    monkeypatch.setattr(chi, "peak_freq", lambda *a, **k: pytest.fail("the peak search ran before the guard"))
    with pytest.raises(RuntimeError, match=relation):
        chi_probes.gen_chi_raw("NADROWSKI", torch.zeros(1, 10), res, torch.zeros(1, 64), t,
                               torch.zeros(1, 3), idx, 1, 0, 1, 64, 1.0, torch.tensor([0.1]), 0.15)
    hopf = forcing.build_nondim_sin_force_tensor(fp, t, res[:, :2], fidx, {"x_scale": 0, "t_scale": 1})
    assert tuple(hopf.shape) == (1, 1, 16)                  # a Hopf-style index keeps its fallback


def test_laplace_simulates_a_tier1_box_at_its_derived_force_scale(monkeypatch):
    """The Laplace diagnostic's features at a tier-1 point equal those of the master-box twin that
    declares the derived force scale, and every drive of the tier-1 call is built at that scale."""
    import numpy as np
    from core.diagnostics import identifiability
    cfg = with_truth(tier1_cfg(chi=False))
    twin = twin_cfg(cfg)
    monkeypatch.setattr(pipeline, "gen_obs", stand_in_gen_obs)
    spy = force_scale_spy(monkeypatch)
    force = torch.tensor([v for v, _ in cfg.force_params_dict.values()], dtype=cfg.hw.dtype)
    got, _, _ = identifiability._laplace_raw(cfg, cfg.params_tensor[0], _res(cfg), force, 4, True, 1000)
    n_tier1 = len(spy)
    want, _, _ = identifiability._laplace_raw(twin, twin.params_tensor[0], _res(twin), force, 4, True, 1000)
    assert np.allclose(got, want, rtol=1e-4, atol=1e-8, equal_nan=True)
    _assert_every_drive_used_the_derived_scale(spy[:n_tier1], derived_force_scale(cfg), rtol=1e-5)


@pytest.mark.parametrize("chi", [True, False])
def test_jacobian_simulates_a_tier1_box_at_its_derived_force_scale(monkeypatch, chi):
    """The jacobian diagnostic's features at the tier-1 truth equal those of the master-box twin that
    declares the derived force scale -- every probe in chi mode, the cell's drive in forced mode -- and
    every drive of the tier-1 call is built at that scale."""
    import numpy as np
    from core.diagnostics import identifiability
    cfg = with_truth(tier1_cfg(chi=chi))
    twin = twin_cfg(cfg)
    monkeypatch.setattr(pipeline, "gen_obs", stand_in_gen_obs)
    spy = force_scale_spy(monkeypatch)
    got, _, _ = identifiability._jacobian_features(_jac_ctx(cfg, 1000), cfg.params_tensor[0], _res(cfg),
                                                   4, True)
    n_tier1 = len(spy)
    want, _, _ = identifiability._jacobian_features(_jac_ctx(twin, 1000), twin.params_tensor[0], _res(twin),
                                                    4, True)
    assert np.allclose(got, want, rtol=1e-4, atol=1e-8, equal_nan=True)
    _assert_every_drive_used_the_derived_scale(spy[:n_tier1], derived_force_scale(cfg), rtol=1e-5)


def test_the_live_simulation_derives_the_force_scale_on_a_tier1_box(monkeypatch):
    """The Simulate panel's live runner plans with the simulator's block and index, keeps reading the
    length scale from the inferred block, and builds every frame's drive at the derived force scale."""
    from core.gui.panels import simulate_runner
    cfg = with_truth(tier1_cfg(chi=False))
    f = derived_force_scale(cfg)
    plan = simulate_runner.plan_stream(cfg, 0.005)
    assert plan.rescale_idx == {"x_scale": 0, "t_scale": 1, "f_scale": 2}
    assert float(plan.rescale_gt[0, 2]) == pytest.approx(f, rel=1e-5)
    assert plan.x_scale == pytest.approx(62.14)
    spy, chunks = force_scale_spy(monkeypatch), []
    simulate_runner.run_simulation_stream(cfg, 0.005, frame_steps=500, fps=0.0, emit_chunk=chunks.append)
    assert chunks
    _assert_every_drive_used_the_derived_scale(spy, f, rtol=1e-5)


def test_temperature_is_the_one_assumed_parameter_and_reports_carry_its_unit_and_mark():
    """On the tier-1 box temperature is the one assumed input, named by its key, and the labels the
    reports print carry its unit and the mark; every other label is the plotting label unchanged. The
    master box assumes nothing, so its report labels are its plotting labels."""
    cfg = tier1_cfg()
    keys = list(cfg.params_dict) + list(cfg.rescale_params)
    i_T = keys.index("T")
    assert cfg.assumed_params == ("T",) and len(cfg.report_labels) == len(keys) == 13
    assert cfg.report_labels[i_T] == cfg.inferred_labels[i_T] + " (assumed input, K)" == "$T$ (assumed input, K)"
    assert [lab for i, lab in enumerate(cfg.report_labels) if i != i_T] == \
           [lab for i, lab in enumerate(cfg.inferred_labels) if i != i_T]
    master = _nad_cfg()
    assert master.assumed_params == () and master.report_labels == master.inferred_labels


def test_the_posterior_summary_marks_the_assumed_parameter_and_the_corner_uses_the_report_labels():
    """Each summary entry says whether its parameter is an assumed input, beside the same three
    quantiles as before; the inference hands the summary the config's assumed keys and labels its
    corner plot with the report labels."""
    import ast, inspect, textwrap
    keys = list(tier1_cfg().params_dict) + ["x_scale", "t_scale", "T"]
    samples = torch.arange(3 * 13, dtype=torch.float32).reshape(3, 13)
    out = orchestrator._posterior_summary(samples, keys, ("T",))
    assert [e["name"] for e in out] == keys and [e["assumed"] for e in out] == [False] * 12 + [True]
    assert set(out[0]) == {"name", "q05", "median", "q95", "assumed"}
    assert out[0]["median"] == 13.0 and out[0]["q05"] == pytest.approx(1.3)
    tree = ast.parse(textwrap.dedent(inspect.getsource(orchestrator.infer_and_visualize)))
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)]
    pair = [c for c in calls if c.func.id == "pairplot"]
    assert len(pair) == 1
    assert ast.unparse({k.arg: k.value for k in pair[0].keywords}["labels"]) == "cfg.report_labels"
    assert [ast.unparse(c.args[2]) for c in calls if c.func.id == "_posterior_summary"] == ["cfg.assumed_params"]
    # The summary is also said once, as one info record, so the Infer tab's log pane carries it.
    described = [c for c in calls if c.func.id == "_describe_posterior_summary"]
    logged = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and ast.unparse(n.func) == "log.info"
              and any(a is d for a in n.args for d in described)]
    assert len(described) == 1 and len(logged) == 1, ast.unparse(tree)


def test_the_posterior_summary_text_marks_the_assumed_parameter():
    """The logged summary: a head line naming the three quantiles, then one row per parameter named
    by its key, and only temperature's row carries the mark and its unit."""
    keys = list(tier1_cfg().params_dict) + ["x_scale", "t_scale", "T"]
    samples = torch.arange(3 * 13, dtype=torch.float32).reshape(3, 13)
    text = orchestrator._describe_posterior_summary(orchestrator._posterior_summary(samples, keys, ("T",)))
    lines = text.splitlines()
    assert lines[0] == "[infer] posterior summary (5 % / median / 95 %):"
    assert [line.split()[0] for line in lines[1:]] == keys, text
    marked = [line for line in lines if "(assumed input" in line]
    assert len(marked) == 1 and marked[0].split()[0] == "T" and "(assumed input, K)" in marked[0], text


def test_describe_informativeness_marks_only_what_it_is_told_is_assumed():
    """The per-parameter line of an assumed parameter carries the mark, and nothing else changes: the
    joint total's head line is the same with or without it."""
    from core.SBI import analysis
    info = {"total_nats": 1.0, "sem_nats": 0.1, "n_used": 8, "n_dropped": 0, "param_names": ["k", "T"],
            "per_param": [0.2, 0.01], "per_direction": None, "n_decompose": 8}
    marked, plain = analysis.describe_informativeness(info, assumed=("T",)), analysis.describe_informativeness(info)
    assert "T +0.0100  (assumed input)" in marked and marked.count("(assumed input)") == 1
    assert "(assumed input)" not in plain and marked.splitlines()[0] == plain.splitlines()[0]


def test_the_calibration_table_plots_and_informativeness_mark_the_assumed_parameter(store, monkeypatch, caplog):
    """Temperature stays in the calibration's rank table, in both rank figures and in the
    informativeness block, each time marked as an assumed input: the table and the figures through the
    report labels, the informativeness block on temperature's own line only."""
    cfg = tier1_cfg()
    seen = stub_calibration_battery(monkeypatch, ks_pvals=[0.5] * 13)
    caplog.set_level("INFO", logger="core")
    cal = orchestrator.validate_calibration(cfg, seen["posterior"], seen["prior"], n_cal=8, cal_n_scales=2,
                                            fig_sink=_close, store=store)
    said = [r.getMessage() for r in caplog.records if r.name == "core.orchestrator"]
    assert said.count("  $T$ (assumed input, K): KS p=0.500  c2st_ranks=0.500  c2st_dap=0.500") == 1, said
    assert seen["rank_plot_labels"] == [cfg.report_labels, cfg.report_labels]
    info = [m for m in said if m.startswith("Informativeness")]
    assert len(info) == 1, said
    marked = [line for line in info[0].splitlines() if "(assumed input)" in line]
    assert len(marked) == 1 and marked[0].split()[0] == "T", info[0]
    assert "(assumed input)" in cal.results["informativeness"]["description"]


def test_the_predictive_check_and_the_best_fit_tables_mark_the_assumed_parameter(monkeypatch):
    """The predictive check's truth table and the two best-fit figures' parameter tables print the
    report labels, so temperature reads as an assumed input there as it does in every other report;
    the best-fit tables show a posterior estimate of it. A box that assumes nothing prints its
    plotting labels exactly as before."""
    import ast, inspect, textwrap
    from core.SBI import overlay
    tree = ast.parse(textwrap.dedent(inspect.getsource(orchestrator.infer_and_visualize)))
    ppc = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and ast.unparse(n.func) == "visualizers.plot_ppc"]
    assert len(ppc) == 1
    assert ast.unparse({k.arg: k.value for k in ppc[0].keywords}["param_names"]) == "cfg.report_labels"

    tables = []

    def best_fit(*a, param_labels=None, **k):
        tables.append(list(param_labels))
        return plt.figure()

    monkeypatch.setattr(overlay.visualizers, "plot_best_fit_overlay", best_fit)
    tier1, master = tier1_cfg(), _nad_cfg()
    for cfg in (tier1, master):
        g = torch.Generator().manual_seed(0)
        stats = torch.randn(6, 8, generator=g)
        overlay.emit_overlay_figures(cfg, torch.randn(1, 512, generator=g), torch.randn(6, 512, generator=g),
                                     stats, stats[0], torch.rand(6, len(cfg.inferred_labels), generator=g),
                                     False, _close)
    assert tables == [tier1.report_labels] * 2 + [master.inferred_labels] * 2, tables
    assert tables[0][-1] == "$T$ (assumed input, K)"


def test_the_calibration_verdict_judges_temperature_and_marks_it_assumed(store, monkeypatch, caplog):
    """Temperature stays in the verdict: its rank test failing fails the calibration. The verdict's
    record marks it as an assumed input, by its key, in the stored parameters and in its one row of
    the logged verdict."""
    seen = stub_calibration_battery(monkeypatch, ks_pvals=[0.5] * 12 + [0.001])
    caplog.set_level("INFO", logger="core")
    cal = orchestrator.validate_calibration(tier1_cfg(), seen["posterior"], seen["prior"], n_cal=8,
                                            cal_n_scales=2, seed=1, fig_sink=_close, store=store)
    v = cal.results["verdict"]
    assert v["passed"] is False
    assert v["parameters"][-1] == {"name": "T", "ks_p": 0.001, "passed": False, "assumed": True}
    said = [r.getMessage() for r in caplog.records if r.name == "core.orchestrator"]
    verdict = [m for m in said if m.startswith("[verdict] ")]
    assert len(verdict) == 1, said
    rows = [line for line in verdict[0].splitlines() if line.strip().startswith("T (assumed input)")]
    assert len(rows) == 1 and rows[0].rstrip().endswith("FAIL"), verdict[0]


def test_sbc_marks_the_assumed_parameter_in_its_table_and_its_records(store, monkeypatch, caplog):
    """The repeated SBC marks temperature in its KS table, in its pooled figure and in each
    per-parameter record, and keeps it in all three."""
    from core.diagnostics import sbc_repeats
    seen = stub_calibration_battery(monkeypatch, ks_pvals=[0.5] * 13)
    caplog.set_level("INFO", logger="core")
    d = sbc_repeats(tier1_cfg(), seen["posterior"], seen["prior"], repeats=2, n_cal=8, num_posterior_samples=10,
                    fig_sink=_close, store=store)
    assert [p["assumed"] for p in d.results["per_param"]] == [False] * 12 + [True]
    rows = [r.getMessage() for r in caplog.records if r.name == "core.diagnostics.sbc"]
    assert sum(m.startswith("T (assumed input)") for m in rows) == 1, rows
    assert seen["rank_plot_labels"][-1][-1] == "T (assumed input)"


def test_a_tier1_record_carries_the_constraint_and_the_assumed_parameters(store):
    """Every record written from a tier-1 config states the relation that derives the force scale, k_B in
    the cell's units, the temperature range and the assumed parameters; a record without tier 1 has none."""
    from core.artifacts import manifest as mf
    cfg = tier1_cfg()
    block = mf.config_from_cfg(cfg)
    assert block["tier1"] == {"relation": "f_scale = N * beta * k_B * T / x_scale",
                              "k_b_cell": cfg.k_b_cell, "T_range": [280.0, 310.0]}
    assert block["assumed_params"] == ["T"]
    m = store.get("prior", _prior_artifact(store, cfg, name="t1").id)
    assert m.config["tier1"] == block["tier1"] and m.config["assumed_params"] == ["T"]
    plain = mf.config_from_cfg(_nad_cfg())
    assert "tier1" not in plain and "assumed_params" not in plain


def test_a_simulated_tier1_observation_records_the_force_scale_it_ran_at(store, monkeypatch):
    """A simulated observation on the tier-1 box records the force scale it was really simulated at: the
    one the relation derives from its truth, which is also the one every drive it built was divided by."""
    from core.SBI import derived
    cfg = with_truth(tier1_cfg(chi=False))
    cfg.T_obs = 1000.0                                      # cell units: 1 s at dt_exp = 1 ms
    monkeypatch.setattr(pipeline, "gen_obs", stand_in_gen_obs)
    spy = force_scale_spy(monkeypatch)
    obs = orchestrator.generate_observations(cfg, fig_sink=_close, store=store)
    expected = derived.to_sim_rescale(cfg.params_tensor, _res(cfg).unsqueeze(0), cfg.rescale_idx,
                                      *cfg.tier1_args)[0, cfg.sim_rescale_idx["f_scale"]]
    got = obs.manifest.config["simulated_f_scale"]
    assert got == pytest.approx(float(expected), rel=1e-6) and 46.5 < got < 47.5   # n*beta*k_B*T/x_scale = 47.0 pN
    _assert_every_drive_used_the_derived_scale(spy, got, rtol=1e-6)


def test_a_simulated_observation_records_a_declared_force_scale_and_none_on_a_box_without_one(store, monkeypatch):
    """Off the tier-1 box the force scale a simulated observation ran at is the one the cell declares, and
    it is recorded all the same; a box whose simulator has no force scale at all records none."""
    monkeypatch.setattr(pipeline, "gen_obs", stand_in_gen_obs)
    master = _nad_cfg(chi_mode=False)
    cli.load_and_validate_gt(master, str(config.CELL_PATH / "nadrowski" / "master_weak.txt"))
    master.T_obs = 1000.0
    obs = orchestrator.generate_observations(master, fig_sink=_close, store=store)
    assert obs.manifest.config["simulated_f_scale"] == 10.0
    labels = config.VALID_LABELS[config.VALID_MODELS.index("NADROWSKI")]
    spont = cli.make_sim_config("NADROWSKI", labels, registry.state_dep_drift("NADROWSKI"),
                                str(config.BOUNDS_PATH / "nadrowski" / "master_spont.txt"), chi_mode=False,
                                hw=config.cpu_device())
    cli.load_and_validate_gt(spont, str(config.CELL_PATH / "nadrowski" / "master_spont.txt"))
    spont.T_obs = 1000.0
    assert "f_scale" not in spont.sim_rescale_idx
    obs = orchestrator.generate_observations(spont, fig_sink=_close, store=store)
    assert "simulated_f_scale" not in obs.manifest.config


@pytest.mark.slow
def test_smoke_runs_every_stage_on_the_tier1_box(tmp_path, monkeypatch, capsys):
    """The real chain -- prior, training, calibration, inference -- on the tier-1 box and cell, in chi
    mode, at tiny size; only the prior's stability sweep and the size of the rotation are bounded. It
    announces the derived force scale and the chi drive amplitude it implies, and the posterior it
    writes infers T, never f_scale."""
    from core.artifacts import ArtifactStore
    from core.tool import main
    monkeypatch.setattr(orchestrator.pipeline, "gen_prior", _tiny_nadrowski_gen_prior)
    # A small rotation. The default one is most of a CPU run's cost, and the full-size rotation on this
    # box is what the card run exercises. The training stage reads its default from these two names.
    monkeypatch.setattr(orchestrator, "REPARAM_FISHER_M", 8)
    monkeypatch.setattr(orchestrator, "REPARAM_FISHER_POINTS", 2)
    root = tmp_path / "tier1"
    assert main(["smoke", "--bounds", str(TIER1_BOX), "--device", "cpu", "--chi", "--cell", str(TIER1_CELL),
                 "--t-obs", "1.0", "--num-runs", "2", "--run-size", "8", "--n-cal", "20",
                 "--max-epochs", "1", "--save", "--store-root", str(root)]) == 0
    out = capsys.readouterr().out
    assert "[tier1] f_scale is DERIVED" in out and "[tier1] chi drive amplitude" in out
    keys = ArtifactStore(root).get("posterior", "smoke_posterior").body["transform"]["param_keys"]
    assert keys[-1] == "T" and "f_scale" not in keys
