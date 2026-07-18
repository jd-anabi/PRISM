"""End-to-end SBI test for a no-forcing user-defined model (v2).

Drives the WHOLE spontaneous inference path at tiny sizes: build a stability-screened UserPrior, train a
short NPE posterior, run SBC/TARP calibration, infer on a simulated observation, and infer on a passive
recording. The stability sweep's production constants (50 iterations x batch, n_max=175000 flood-fill)
are far too slow for a test, so ``pipeline.gen_prior`` is monkeypatched to a tiny UserPrior screen -- the
same construct_prior call, small sizes. Everything else runs for real.

Also pins the built-in (Nadrowski) forcing path: generate_observations still yields the full-width,
Group-G-populated conditioning vector, so the spontaneous-only branching did not perturb it.

Run:  python tests/test_user_sbi.py      (or under pytest)
"""
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib                                                 # noqa: E402
matplotlib.use("Agg")

import torch                                                      # noqa: E402

from core import config, registry, orchestrator, cli             # noqa: E402
from core.Helpers import model_store                             # noqa: E402
from core.SBI.Priors.user_prior import UserPrior                 # noqa: E402
from core.SBI.statistics import FEATURE_LABELS                   # noqa: E402
from core.config import VALID_MODELS, VALID_LABELS               # noqa: E402

_N_GROUP_G = 11
_N_SPONT = len(FEATURE_LABELS) - _N_GROUP_G   # 30


def _tiny_gen_prior(model, t, global_batch_size, local_batch_size, segs, prior_bounds,
                    state_dep_drift=False, num_iterations=25, log_mask=None,
                    dtype=torch.float32, device=torch.device("cpu")):
    """A tiny stand-in for pipeline.gen_prior: the same UserPrior.construct_prior, small sizes."""
    p = UserPrior(registry.get(model), dtype, device)
    return p.construct_prior(t, len(prior_bounds), 32, 8, segs, prior_bounds,
                             t_global_scale=2, num_iterations=2, n_max=120, steady=False,
                             state_dep_drift=state_dep_drift, log_mask=log_mask)


def test_no_forcing_user_model_full_sbi_pipeline():
    """build_prior -> build_posterior -> generate_observations -> infer -> validate -> passive-infer."""
    name = "SBITEST"
    doc = {"schema_version": 1, "name": name,
           "variables": [{"name": "x", "drift": "-k*x", "D": "d0", "init": 0.5, "forcing": None}],
           "params": {"k": 1.0, "d0": 0.05}, "rescale": {"x_scale": 10.0, "t_scale": 0.01}}
    saved_gen_prior = orchestrator.pipeline.gen_prior
    saved_runs, saved_ncal = orchestrator.TRAINING_NUM_RUNS, orchestrator.SBC_N_CAL
    sink = lambda title, fig: None                                # noqa: E731
    try:
        model_store.save_user_model(doc)
        registry.load_user_models()
        assert registry.is_sbi_user_model(name) is True

        cfg = cli.make_sim_config(name, registry.get(name).labels, registry.state_dep_drift(name),
                                  str(config.BOUNDS_PATH / name.lower() / "default.txt"))
        cli.load_and_validate_gt(cfg, str(config.CELL_PATH / name.lower() / "default.txt"))
        cfg.hw = config.cpu_device()
        cfg.hw.batch_size = 8
        cfg.T_obs = 1.0
        assert cfg.has_forcing is False

        orchestrator.pipeline.gen_prior = _tiny_gen_prior
        orchestrator.TRAINING_NUM_RUNS = 2
        orchestrator.SBC_N_CAL = 60

        inferred_prior, force_prior = orchestrator.build_prior(cfg, None, True, save=False, fig_sink=sink)
        assert force_prior is None                               # no drive -> no forcing prior

        posterior, _ = orchestrator.build_posterior(cfg, inferred_prior, force_prior, None, True,
                                                    save=False, fig_sink=sink)

        x_dim, obs_stats, t_dim = orchestrator.generate_observations(cfg)
        assert obs_stats.shape[-1] == len(FEATURE_LABELS) + 1    # [S | log(T)], no forcing block
        assert torch.allclose(obs_stats[0, _N_SPONT:_N_SPONT + _N_GROUP_G], torch.zeros(_N_GROUP_G))
        assert torch.isfinite(obs_stats).all()

        orchestrator.infer_and_visualize(cfg, posterior, obs_stats, x_dim, t_dim, show_truth=True,
                                         fig_sink=sink)
        orchestrator.validate_calibration(cfg, posterior, inferred_prior, force_prior, fig_sink=sink)

        # passive experimental path: a single unforced recording, no drive / force units
        obs_stats_e, obs_data_e, t_dim_e = orchestrator.build_experiment_obs_spontaneous(
            cfg, x_dim[0].clone(), 1.0)
        assert obs_stats_e.shape[-1] == len(FEATURE_LABELS) + 1
        assert torch.allclose(obs_stats_e[0, _N_SPONT:_N_SPONT + _N_GROUP_G], torch.zeros(_N_GROUP_G))
        orchestrator.infer_and_visualize(cfg, posterior, obs_stats_e, obs_data_e, t_dim_e,
                                         show_truth=False, fig_sink=sink)
    finally:
        orchestrator.pipeline.gen_prior = saved_gen_prior
        orchestrator.TRAINING_NUM_RUNS, orchestrator.SBC_N_CAL = saved_runs, saved_ncal
        try:
            model_store.delete_user_model(name)
        except Exception:                                        # noqa: BLE001
            pass
        registry.unregister(name)


def test_builtin_forcing_path_unperturbed():
    """The spontaneous-only branching must leave the Nadrowski forcing path byte-compatible: a full-width
    conditioning vector [S(41) | log(T) | forcing] with Group G populated by the drive response."""
    labels = VALID_LABELS[VALID_MODELS.index("NADROWSKI")]
    cfg = cli.make_sim_config("NADROWSKI", labels, True,
                              str(config.BOUNDS_PATH / "nadrowski" / "cell.txt"))
    cli.load_and_validate_gt(cfg, str(config.CELL_PATH / "nadrowski" / "cell.txt"))
    cfg.hw = config.cpu_device()
    cfg.T_obs = 1000.0                                            # ms units -> 1 s of data
    assert cfg.has_forcing is True
    _, obs_stats, _ = orchestrator.generate_observations(cfg)
    n_forcing = len(cfg.force_params_dict)
    assert obs_stats.shape[-1] == len(FEATURE_LABELS) + 1 + n_forcing
    assert not torch.allclose(obs_stats[0, _N_SPONT:_N_SPONT + _N_GROUP_G], torch.zeros(_N_GROUP_G))
    assert torch.isfinite(obs_stats).all()


if __name__ == "__main__":
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
