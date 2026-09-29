"""The tier-1 path: the box on which temperature is inferred and the force scale is derived from it.

``master_tier1.txt`` declares ``T`` in the rescale block where other boxes declare ``f_scale``. The
network is trained on T, and every simulation runs at the force scale T implies,

    f_scale = n * beta * k_B * T / x_scale,

which is 46.99 pN at the tier-1 cell (n=50, beta=14.1, x_scale=62.14 nm, T=300 K, with k_B in
pN*nm/K). A drive builder handed the inferred index finds no ``f_scale`` in it and falls back to
``x_scale / t_scale`` -- 16.66 at the same cell -- without a word, so a test that sees 16.66 has found
that fallback. The retrain trains on this box, so every place a simulation is set up from a draw is
pinned here: the training rows, the calibration set, the Fisher operating points, a simulated
observation and its predictive checks, a narrowing round, and loading a prior built on the master box.

Cheap by design: every fast test replaces ``pipeline.gen_obs`` with ``stand_in_gen_obs`` and records
the force scale each drive is divided by with ``force_scale_spy``; recordings are one to two seconds
and batches are two of four rows. One slow test runs the real ``smoke`` chain on the tier-1 box.
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
from tests._fixtures import _FakeDP, _nad_cfg, _prior_artifact, _tiny_nadrowski_gen_prior

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
    a given batch does depends on its unseeded (t_scale, T) draw, so zero or more such lines are
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


def test_the_fisher_operating_points_simulate_at_the_derived_force_scale(monkeypatch):
    """The rotation's Fisher is built over the simulated experiment: its anchor is driven at the
    truth's derived force scale, and its central-difference arms, which move n, beta, x_scale and T by
    a latent step, stay near it -- never at the fallback, and never at the temperature itself."""
    from core.SBI import decorrelate
    cfg = with_truth(tier1_cfg(chi=False))
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


def test_a_simulated_observation_and_its_predictive_checks_use_the_derived_force_scale(store, monkeypatch):
    """A simulated observation is driven at the truth's derived force scale, and so is every
    simulation the inference runs from its posterior draws: the predictive-check bins and the eye
    test's two central trajectories. The posterior is a stand-in whose every draw is the truth."""
    cfg = with_truth(tier1_cfg(chi=False))
    cfg.T_obs = 1000.0                                      # cell units: 1 s at dt_exp = 1 ms
    monkeypatch.setattr(pipeline, "gen_obs", stand_in_gen_obs)
    spy = force_scale_spy(monkeypatch)
    f = derived_force_scale(cfg)
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
    _assert_every_drive_used_the_derived_scale(spy, f, rtol=1e-5)


def test_a_narrowing_round_on_a_tier1_parent_judges_the_truth_in_inferred_coordinates(store, monkeypatch,
                                                                                       caplog):
    """A narrowing round on the tier-1 box keeps the derivation (its plan carries n and beta by name
    and k_B in cell units, and it announces the derived force scale), and it judges the truth against
    its region in the inferred coordinates, temperature included: silent when the truth is inside, one
    warning naming the temperature's direction when it is outside."""
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
        orchestrator.build_posterior(cfg, lp, None, True, truncation=region(c - 0.5, c + 0.5), **budget)
    assert not any("GROUND TRUTH" in str(w.message) for w in caught)
    assert plans[0].nd_idx == cfg.nd_idx and plans[0].k_b_cell == pytest.approx(cfg.k_b_cell)
    assert any(r.getMessage().startswith("[tier1] f_scale is DERIVED") for r in caplog.records)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        orchestrator.build_posterior(cfg, lp, None, True, truncation=region(c + 1.0, c + 2.0), **budget)
    truth = [w for w in caught if "GROUND TRUTH" in str(w.message)]
    assert len(truth) == 1 and "direction 12" in str(truth[0].message)
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
