"""Shared test helpers for the artifact-store suites (piece 1 of the 2026-09-10 hardening programme).

A plain module: importing it imports torch and sbi (``DirectPosterior`` has to be imported at module
level for ``_FakeDP`` to pickle) and does NOTHING ELSE -- no simulation, no file I/O, no torch
seeding, no store writes. Everything else here is a function or class definition whose own imports
stay lazy, exactly as they were before the move, so a bare ``import tests._fixtures`` costs the torch
import and nothing more.
Moved out of ``tests/test_artifact_store.py`` (Task 7) so ``tests/test_user_sbi.py``,
``tests/test_nav_and_gating.py`` and others can share them without importing a whole other test
module (which pytest would then also collect a second time under a different name).
"""
import torch

from core.artifacts import manifest as mf


def _nad_cfg(**over):
    """A real NADROWSKI SimConfig off the master bounds file, CPU device."""
    from core import cli, registry
    from core.config import BOUNDS_PATH, VALID_LABELS, VALID_MODELS
    from core import config
    labels = VALID_LABELS[VALID_MODELS.index("NADROWSKI")]
    cfg = cli.make_sim_config("NADROWSKI", labels, registry.state_dep_drift("NADROWSKI"),
                              str(BOUNDS_PATH / "nadrowski" / "master.txt"), **over)
    cfg.hw = config.cpu_device()
    return cfg


def _gmm_in_box(lows, highs, mask, seed=0, weights=None):
    """A tiny 2-component MixtureSameFamily pushed through a box bijection -- a real GMM prior payload,
    small enough to build and pickle in a test. ``weights`` replaces the two equal weights with a
    vector of its own length (one component per entry)."""
    from core.SBI import reparam
    torch.manual_seed(seed)
    d = len(lows)
    w = torch.tensor([0.5, 0.5]) if weights is None else torch.as_tensor(weights, dtype=torch.float32)
    n = int(w.numel())
    base = torch.distributions.MixtureSameFamily(
        torch.distributions.Categorical(probs=w),
        torch.distributions.MultivariateNormal(torch.randn(n, d), covariance_matrix=torch.eye(d).expand(n, d, d)))
    T = reparam.build_box_bijection(torch.tensor(lows, dtype=torch.float32), torch.tensor(highs, dtype=torch.float32), mask)
    return torch.distributions.TransformedDistribution(base, T)


def _prior_artifact(store, cfg, *, name="p", lows=None, highs=None, keys=None, model=None, seed=0,
                    mask=None):
    """A prior artifact with a real 2-component GMM payload, laid out exactly as build_prior writes it.

    ``mask`` records a DIFFERENT log-box mask in the manifest than the one the payload's box was
    actually built in. It is the only way to provoke load_prior's log-mask refusal: the mask is
    derived from the config, so a prior written from this config can never disagree with it by
    accident -- only a prior built when REPARAM_LOG_PARAMS (or a user model's box field) said
    something else can, and that is exactly what the refusal exists for.
    """
    from core.Helpers import file_manager
    from core.SBI.reparam import nd_log_mask
    from core.SBI.run_guards import _gmm_fingerprint, _log_params_for
    keys = keys or list(cfg.params_dict)
    lows = lows or [b[0] for _, b in cfg.params_dict.values()]
    highs = highs or [b[1] for _, b in cfg.params_dict.values()]
    box_mask = nd_log_mask(cfg, log_params=_log_params_for(cfg))
    recorded = [bool(v) for v in (box_mask.tolist() if mask is None else mask)]
    dist = _gmm_in_box(lows, highs, box_mask, seed)
    with store.create("prior", cfg, name=name) as w:
        file_manager.save_mix_dist(dist, str(w.payload("prior.pt")), model=model or cfg.model, param_keys=keys)
        if model:
            w.config["model"] = model
        w.fingerprints["gmm"] = _gmm_fingerprint(dist)
        w.body = {"gmm": {"n_components": 2, "param_keys": list(keys),
                          "box": {"nd_lows": [float(v) for v in lows], "nd_highs": [float(v) for v in highs],
                                  "log_mask": recorded}},
                  "sweep": {}, "stability": {"accepted_sets": None, "iterations": 1}}
    return w


from sbi.inference import DirectPosterior


class _FakeDP(DirectPosterior):
    """A DirectPosterior by type only (the loader's isinstance accepts it); module-level so it pickles."""
    def __init__(self):
        pass


def _set_path(w, path, value):
    target = w.config if path[0] == "config" else w.body
    for key in path[1 if path[0] == "config" else 0:-1]:
        target = target[key]
    target[path[-1]] = value


def _posterior_artifact(store, cfg, *, name="post", amortized=True, region=None, V=None, prior=None, over=None):
    """A posterior artifact with a _FakeDP payload, laid out exactly as build_posterior writes it."""
    keys = list(cfg.params_dict) + list(cfg.rescale_params)
    with store.create("posterior", cfg, name=name) as w:
        dp = _FakeDP()
        dp.prior = prior
        torch.save(dp, str(w.payload("posterior.pt")))
        w.parents = {"prior": "20260910T100000"}
        w.body = {
            "mode": cfg.observation_mode, "conditioning": mf.conditioning_block(cfg),
            "transform": {"param_keys": keys, "log_params": [],
                          "nd_lows": [float(b[0]) for _, b in cfg.params_dict.values()],
                          "nd_highs": [float(b[1]) for _, b in cfg.params_dict.values()],
                          "rescale_lows": [float(b[0]) for _, b in cfg.rescale_params.values()],
                          "rescale_highs": [float(b[1]) for _, b in cfg.rescale_params.values()],
                          "V": mf.tensor_to_json(V), "V_orientation": "columns",
                          "fisher_eigenvalues": None, "V_digest": mf.tensor_digest(V)},
            "amortized": amortized, "truncation": None if region is None else mf.region_to_json(region),
            "training": {}}
        for path, value in (over or {}).items():
            _set_path(w, path, value)
    return w


class _LoadedStub:
    """The LoadedPrior shape with no GMM: fails open through every fingerprint check."""
    id, name, fingerprint, force_prior = None, "", None, None
    def __init__(self, prior=None): self.prior = prior if prior is not None else object()


class _WriteFailed(RuntimeError):
    """Injected mid-write failure. Not OSError, so a handler that swallows disk errors cannot hide it."""


def _failing(real):
    """Wrap a serializer so it writes its bytes and THEN fails -- the tear that atomicity must absorb.

    Failing before writing anything would pass against a plain `torch.save` too: the destination is
    only clobbered once the writer has begun. The bytes have to land first for the test to mean
    anything.

    Signature-agnostic (`*a, **k`) because the two serialisers order their arguments differently --
    ``torch.save(obj, file)`` against ``np.savez(file, **arrays)``.
    """
    def _boom(*a, **k):
        real(*a, **k)
        raise _WriteFailed("disk full")
    return _boom


def _tiny_gen_prior(model, t, global_batch_size, local_batch_size, segs, prior_bounds,
                    state_dep_drift=False, num_iterations=25, log_mask=None,
                    dtype=torch.float32, device=torch.device("cpu"), **_kw):
    """A tiny stand-in for pipeline.gen_prior: the same UserPrior.construct_prior, small sizes.

    ``**_kw`` IS LOAD-BEARING AND THERE ARE TWO OF THESE STUBS. A stub installed over a function must
    tolerate arguments added to that function later, or the suite dies ~40 tests in with a TypeError
    raised deep inside build_prior -- which names only the stub it hit first, so fixing that one
    reveals the second on the next run. Adding n_max/step upstream cost an hour this way on
    2026-08-27. Deliberately NOT a hand-mirrored signature: that never checked anything (it failed as
    a TypeError, not an assertion), and gen_prior's real signature is asserted directly by
    test_n_max_and_step_are_no_longer_hidden_inside_gen_prior.
    """
    from core import registry
    from core.SBI.Priors.user_prior import UserPrior
    p = UserPrior(registry.get(model), dtype, device)
    return p.construct_prior(t, len(prior_bounds), 32, 8, segs, prior_bounds,
                             t_global_scale=2, num_iterations=2, n_max=120, steady=False,
                             state_dep_drift=state_dep_drift, log_mask=log_mask)


def install_sbitest():
    """Register the SBITEST user model and emit its Bounds/Cells/Units triple; ``(bounds, cell, teardown)``.

    A one-variable OU process with two ND parameters and no drive -- the smallest thing that can carry
    a real prior, a real flow and a real observation. ``save_user_model`` writes
    ``Resources/{Bounds,Cells,Units}/sbitest/`` and ``Resources/Models/SBITEST.json``, so the two paths
    returned are REAL INPUT FILES: that is what lets the command-line tool be driven with
    ``--bounds``/``--cell`` exactly as an operator would drive it, instead of against a hand-built
    config no flag could have produced. ``teardown`` removes exactly what this helper wrote (the
    SBITEST bounds/cell/units files and the model JSON) and unregisters the model; call it from a
    ``finally``. ``git status`` after a run is what shows ``Resources/`` clean -- the session conftest
    only snapshots and asserts ``Artifacts/``.
    """
    from core import config, registry
    from core.Helpers import model_store
    name = "SBITEST"
    doc = {"schema_version": 1, "name": name,
           "variables": [{"name": "x", "drift": "-k*x", "D": "d0", "init": 0.5, "forcing": None}],
           "params": {"k": 1.0, "d0": 0.05}, "rescale": {"x_scale": 10.0, "t_scale": 0.01}}
    model_store.save_user_model(doc)
    registry.load_user_models()

    def teardown():
        try:
            model_store.delete_user_model(name)
        except Exception:                                    # noqa: BLE001 -- best-effort cleanup
            pass
        registry.unregister(name)

    return (str(config.BOUNDS_PATH / name.lower() / "default.txt"),
            str(config.CELL_PATH / name.lower() / "default.txt"), teardown)


def build_tiny_run(store, hw=None):
    """A REAL SBITEST prior and posterior at tiny size, inside ``store`` (make it the default first).
    Mirrors test_user_sbi.test_no_forcing_user_model_full_sbi_pipeline's setup -- and its teardown:
    read that test's finally block and do the same in ``teardown``. ``hw`` is the DeviceConfig to run
    on (default the CPU); tests/test_gpu_paths.py passes config.detect_device() to reach the paths a
    CPU run cannot see."""
    from types import SimpleNamespace
    from core import cli, config, orchestrator, registry
    name = "SBITEST"
    bounds, cell, undo_install = install_sbitest()
    cfg = cli.make_sim_config(name, registry.get(name).labels, registry.state_dep_drift(name), bounds)
    cli.load_and_validate_gt(cfg, cell)
    cfg.hw = hw if hw is not None else config.cpu_device()
    cfg.hw.batch_size = 8
    cfg.T_obs = 1.0
    saved_gen_prior = orchestrator.pipeline.gen_prior
    orchestrator.pipeline.gen_prior = _tiny_gen_prior
    sink = lambda title, fig: None                                   # noqa: E731
    prior = orchestrator.build_prior(cfg, None, True, fig_sink=sink, name="tiny_prior")
    # EVERY knob is an argument. This fixture used to rebind orchestrator.TRAINING_NUM_RUNS,
    # SBC_N_CAL and TRAINING_CHECKPOINT_EVERY for its consumers; the cadence is now the session
    # fixture's job (tests/conftest.py::_checkpointing_off_unless_asked) and the sizes are each
    # consumer's own (they all pass num_runs=/n_cal= already).
    posterior = orchestrator.build_posterior(cfg, prior, None, True, fig_sink=sink, name="tiny_post",
                                             num_runs=2, hidden_features=8, num_transforms=1,
                                             stop_after_epochs=1)

    def other_prior():
        return orchestrator.build_prior(cfg, None, True, fig_sink=sink)   # another fit, another GMM

    def teardown():
        orchestrator.pipeline.gen_prior = saved_gen_prior
        undo_install()
    return SimpleNamespace(cfg=cfg, prior=prior, posterior=posterior, store=store, sink=sink,
                           other_prior=other_prior, teardown=teardown)
