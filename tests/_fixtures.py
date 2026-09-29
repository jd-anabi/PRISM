"""Shared test helpers: the artifact-store stand-ins and the GUI suites' helpers.

A plain module: importing it imports torch and sbi (``DirectPosterior`` has to be imported at module
level for ``_FakeDP`` to pickle) and does NOTHING ELSE -- no simulation, no file I/O, no torch
seeding, no store writes. Everything else here is a function or class definition whose own imports
stay lazy, exactly as they were before the move, so a bare ``import tests._fixtures`` costs the torch
import and nothing more.
Moved out of ``tests/test_artifact_store.py`` so ``tests/test_user_sbi.py``,
``tests/test_nav_and_gating.py`` and others can share them without importing a whole other test
module (which pytest would then also collect a second time under a different name).

It also holds what the GUI suites used to copy per file -- ``qt_app``, ``pump``, ``code_only``,
``PaneCapture`` -- and the dialog record ``SHOWN`` that tests/conftest.py's session guard appends
to. ONE import spelling, ``tests._fixtures`` (pytest.ini puts the repo root on the path; tests/ has
no __init__.py): ``from _fixtures import SHOWN`` would import this module a second time under
another name, with a second, empty SHOWN that the guard never touches.
"""
import ast
import inspect
import os
import textwrap
import time
from pathlib import Path

import torch

from core.artifacts import manifest as mf

# The top-level repository directories that hold PRODUCT code, and therefore the ones the source
# scans walk: among them the literal-path scan (tests/test_artifact_store.py), the rotation-reader
# and input() scans (both tests/test_conditioning_repair.py) and the refusal and message-box scans
# (tests/test_refusals.py). The command-line tool's own code lives under core/tool, so one root
# covers every scan and both front ends. tests/ and the gitignored archive/ are deliberately absent:
# they are not shipped code. test_the_source_scans_cover_every_code_directory keeps this set closed,
# so a new top-level package cannot appear and be scanned by nothing.
CODE_ROOTS: tuple[str, ...] = ("core",)

# The top-level repository files (outside any CODE_ROOTS directory) that hold product code, so the
# literal-path scan walks these too, and the same guard test keeps this set closed as well.
CODE_FILES: tuple[str, ...] = ("conftest.py",)

def backdate_tree(path, seconds: float = 3600.0) -> None:
    """Push a whole tree's modification times into the past.

    ``ArtifactStore.remove_incomplete`` refuses a directory whose tree was touched within
    ``store.RECENT_WRITE_SECONDS`` (the recency guard: an artifact's manifest is written LAST, so a
    run in flight -- possibly in another process -- looks exactly like a leftover). A
    test's leftover is seconds old, so every test that means one to be REMOVED says so here rather
    than sleeping five minutes or monkeypatching the guard away.
    """
    when = time.time() - seconds
    for p in [Path(path), *Path(path).rglob("*")]:
        os.utime(p, (when, when))


# Every QMessageBox a test did not fake itself, in order. tests/conftest.py::_no_modal_dialogs replaces
# QMessageBox.exec for the session with a record-and-return-0, and _clear_shown empties this list
# before each test, so SHOWN[-1] is the box the code under test just tried to show.
SHOWN: list = []


def qt_app():
    """The one QApplication a process may hold, built on first use (offscreen: the root conftest.py
    sets QT_QPA_PLATFORM before PySide6 is imported). The seven per-file ``_app`` copies were this."""
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def pump(app, seconds: float = 0.5) -> None:
    """Drive the event loop without app.exec(), so the pump's queued signals get delivered."""
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        app.processEvents()
        time.sleep(0.01)


def _strip_docstrings(tree):
    """Remove every docstring from a parsed tree, in place, and return it.

    ⚠ ``ast.unparse`` DROPS COMMENTS BUT KEEPS DOCSTRINGS -- they are real string expressions in the
    AST, not trivia. An earlier version of the helper claimed otherwise, and the claim went unnoticed
    because the checks that used it happened to forbid strings that appeared only in comments. It
    stopped being harmless the moment a check forbade `mem_get_info` in a function whose DOCSTRING
    explains why it does not use mem_get_info: the assertion matched the prose, exactly the
    false positive the parse was supposed to prevent (the same shape as the _local_map lesson).
    """
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if (isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                and isinstance(body, list) and body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)):
            body.pop(0)
            if not body:
                body.append(ast.Pass())
    return ast.fix_missing_locations(tree)


def code_only(obj) -> str:
    """Executable source of ``obj`` -- a function, method, class or whole MODULE -- with comments and
    docstrings both gone.

    IF YOU ASSERT ON SOURCE TEXT, PARSE IT FIRST. A check that "_local_map no longer hardcodes the
    CPU" failed on its first run against the very COMMENT that documents the fix, because the comment
    necessarily contains the string it forbids. The same false positive had already cost time once on
    the TSNPE runner check, and a THIRD time on 2026-08-28 -- see _strip_docstrings for why
    ast.unparse alone is not enough. The suites grew four copies of this (two _strip_docstrings,
    _code_only, _unparsed) and a fifth that stripped only the outermost docstring; this is the one.
    """
    return ast.unparse(_strip_docstrings(ast.parse(textwrap.dedent(inspect.getsource(obj)))))


class PaneCapture:
    """Records what a panel writes to its log pane, as ``(level, text)`` in order, INSTEAD of
    rendering it. ``PaneCapture(panel)`` replaces BOTH of that one pane's channels:

      * ``LogPane.append_line(text, level="info")`` -- the panel's own GUI-thread messages (a refusal,
        a warning, "Run cancelled.") and ``WorkerSignals.log``;
      * ``LogPane.append_lines(batch)`` -- ``WorkerSignals.log_batch``, one pump tick of the RUN's
        output, and therefore every print, every retired tqdm bar, every ``core`` record (through
        ``streams._PumpLogHandler``) and every Python warning (through ``redirect_streams``'
        ``showwarning``). That channel is the one the helper used to miss, so a test could assert on
        what the pane said while being blind to almost everything in it.

    TWO ORDERS, ONE LIST. A captured entry is ``(level, text)``: that is what the hand-written stubs
    this helper replaced appended and what every assertion in the suites reads. A batch's pairs arrive
    as ``(text, level)``, because that is the order ``WorkerSignals.log_batch`` carries
    (``core/gui/worker.py:27``; ``tests/test_vt_progress.py``'s handler test pins that payload
    directly). Each pair is therefore FLIPPED as it is appended, so ``.lines`` reads in one order --
    the order a user sees the lines in -- whichever channel each line came from. A test that wants one
    channel alone filters on what it knows only that channel says.

    ``BasePanel.dispatch`` resolves both attributes AT CONNECT TIME, so a capture installed before the
    dispatch sees both channels and one installed after sees neither. A batch is published by the
    pump's daemon thread at 15 Hz as a queued cross-thread signal, so a test must drive the event loop
    (``pump(app)``) before asserting on run output. The widget receives nothing, and the panel is a
    throwaway built by the test."""
    def __init__(self, panel):
        self.lines: list[tuple[str, str]] = []
        panel.log_pane.append_line = self._append
        panel.log_pane.append_lines = self._append_batch

    def _append(self, text: str, level: str = "info") -> None:
        self.lines.append((level, text))

    def _append_batch(self, batch) -> None:
        """One pump tick: each ``(text, level)`` appended as ``(level, text)``, in order. A falsy batch
        appends nothing, which is what ``LogPane.append_lines`` renders for one."""
        if not batch:
            return
        for text, level in batch:
            self.lines.append((level, text))


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


def _tiny_nadrowski_gen_prior(model, t, global_batch_size, local_batch_size, segs, prior_bounds,
                              state_dep_drift=False, num_iterations=25, log_mask=None,
                              dtype=torch.float32, device=torch.device("cpu"), **_kw):
    """Tiny stand-in for pipeline.gen_prior on the built-in Nadrowski: same construct_prior, small sizes."""
    from core.SBI.Priors import nadrowski_prior
    p = nadrowski_prior.NadrowskiPrior(dtype, device)
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
    from matplotlib import pyplot as plt
    from core import cli, config, orchestrator, registry
    name = "SBITEST"
    bounds, cell, undo_install = install_sbitest()
    saved_gen_prior = orchestrator.pipeline.gen_prior
    # A setup that fails must undo what it did: the conftest fixture reaches its own teardown only
    # after this returns, so without the except a failed build left the gen_prior stub installed on
    # core.orchestrator and SBITEST's files in the real Resources/ for the rest of the process.
    try:
        cfg = cli.make_sim_config(name, registry.get(name).labels, registry.state_dep_drift(name), bounds)
        cli.load_and_validate_gt(cfg, cell)
        cfg.hw = hw if hw is not None else config.cpu_device()
        cfg.hw.batch_size = 8
        cfg.T_obs = 1.0
        orchestrator.pipeline.gen_prior = _tiny_gen_prior
        # CLOSING: the writer's sink saves the PNG and forwards here, and only closes the figure itself
        # when nothing is forwarded -- a no-op sink leaked every figure every consumer drew.
        sink = lambda title, fig: plt.close(fig)                         # noqa: E731
        prior = orchestrator.build_prior(cfg, None, True, fig_sink=sink, name="tiny_prior")
        # EVERY knob is an argument. This fixture used to rebind orchestrator.TRAINING_NUM_RUNS,
        # SBC_N_CAL and TRAINING_CHECKPOINT_EVERY for its consumers; the cadence is now the session
        # fixture's job (tests/conftest.py::_checkpointing_off_unless_asked) and the sizes are each
        # consumer's own (they all pass num_runs=/n_cal= already).
        posterior = orchestrator.build_posterior(cfg, prior, None, True, fig_sink=sink, name="tiny_post",
                                                 num_runs=2, hidden_features=8, num_transforms=1,
                                                 stop_after_epochs=1)
    except BaseException:
        orchestrator.pipeline.gen_prior = saved_gen_prior
        undo_install()
        raise

    def other_prior():
        return orchestrator.build_prior(cfg, None, True, fig_sink=sink)   # another fit, another GMM

    def teardown():
        orchestrator.pipeline.gen_prior = saved_gen_prior
        undo_install()
    return SimpleNamespace(cfg=cfg, prior=prior, posterior=posterior, store=store, sink=sink,
                           other_prior=other_prior, teardown=teardown)


def stub_calibration_battery(monkeypatch, *, ks_pvals=None, tarp_ks_p=0.5) -> dict:
    """Stand in for everything validate_calibration and sbc_repeats simulate or sample. The
    stand-ins draw from torch's and numpy's global streams as the real calls do, so a seed is
    observable and nothing is simulated.

    ks_pvals: None draws each rank test's KS p-values from torch's stream; a flat list is returned
    by every rank test; a list of lists is consumed one list per rank test, in order. Returns a dict
    with the posterior and prior stand-ins and what the stand-ins saw: "x_cal" (each calibration set
    drawn), "numpy_at_draw" (numpy's state vector at each draw), "rank_plot_labels" (each rank
    figure's parameter labels)."""
    import numpy as np
    from types import SimpleNamespace
    from matplotlib import pyplot as plt
    from core import orchestrator as orch
    seen = {"x_cal": [], "numpy_at_draw": [], "rank_plot_labels": [],
            "posterior": SimpleNamespace(posterior=SimpleNamespace(truncation=None, x_obs_digest=None),
                                         id="post", name="", accepted=[]),
            "prior": SimpleNamespace(prior=None, force_prior=None, id="prior", name="", fingerprint=None)}
    queue = list(ks_pvals) if ks_pvals and isinstance(ks_pvals[0], (list, tuple)) else None

    def draw(cfg, vlp, T, force_prior, *, n_cal, cal_n_scales, chi_k_fixed=None):
        P = len(cfg.params_dict) + len(cfg.rescale_params)
        seen["numpy_at_draw"].append(np.random.get_state()[1].copy())
        x = torch.randn(n_cal, 5)
        seen["x_cal"].append(x.clone())
        theta = torch.rand(n_cal, P) + torch.as_tensor(np.random.standard_normal((n_cal, P)),
                                                       dtype=torch.float32)
        return x, theta

    def check_sbc(ranks, prior_samples, dap_samples, num_posterior_samples):
        P = prior_samples.shape[1]
        ks = (queue.pop(0) if queue is not None
              else list(ks_pvals) if ks_pvals is not None else torch.rand(P).tolist())
        return {"ks_pvals": ks, "c2st_ranks": [0.5] * P, "c2st_dap": [0.5] * P}

    def rank_plot(**k):
        seen["rank_plot_labels"].append(list(k.get("parameter_labels") or []))
        return plt.figure(), None

    def informativeness(post, theta_star, x_cal, prior, *, param_names=None, **k):
        n, P = theta_star.shape
        return {"total_nats": 1.0, "sem_nats": 0.1, "n_used": n, "n_dropped": 0,
                "param_names": param_names, "per_param": [0.1] * P, "per_direction": None,
                "n_decompose": n}

    for name, fn in {
        "_calibration_prior": lambda cfg, posterior, prior: (None, None, None),
        "_assert_prior_used_matches_posterior": lambda *a, **k: None,
        "_draw_calibration_set": draw,
        "run_sbc": lambda thetas, xs, posterior, num_posterior_samples, **k: (
            torch.randint(0, num_posterior_samples + 1, tuple(thetas.shape)), torch.zeros(tuple(thetas.shape))),
        "_sbc_reference_sample": lambda cfg, vlp, T, truncation, prior, theta_star: torch.zeros_like(theta_star),
        "check_sbc": check_sbc, "sbc_rank_plot": rank_plot,
        "run_tarp": lambda thetas, xs, posterior, **k: (torch.linspace(0, 1, 5), torch.linspace(0, 1, 5)),
        "check_tarp": lambda ecp, alpha: (0.0, tarp_ks_p),
        "plot_tarp": lambda *a, **k: plt.figure(),
    }.items():
        monkeypatch.setattr(orch, name, fn)
    monkeypatch.setattr(orch.analysis, "informativeness", informativeness)
    return seen


def build_browse_store(root):
    """One artifact of each of the EIGHT kinds plus the three bad-directory shapes, in a store at
    ``root``. Returns ``{kind: id, ..., "bad": (dir_name, dir_name, dir_name)}``.

    SECONDS, not minutes -- the browser suite must not reach for ``tiny_run``, whose cost is why
    ``screen_run`` is module-scoped. Every writer is handed ``cfg=None`` and every body is the
    smallest dict its kind's ``manifest.BODY_KEYS`` allows, so nothing here reads a bounds file,
    builds a SimConfig, fits a GMM or trains anything. Nothing ever loads these payloads (there are
    none): the browser only reads manifests. The one real cost is ``ArtifactWriter._commit``'s three
    ``git`` subprocesses per artifact.

    Each artifact is named ``browse_<kind>`` and carries a note, so the name and note columns have
    content. The simulation cache is the exception: it is ALWAYS unnamed
    (``write_simulation_manifest`` writes ``name=""``), so its row also covers ``Summary.label``'s
    ``(unnamed <id>)`` form. It is written at 3 of 4 batches and NOT complete, so its row carries real
    progress and ``finished`` is False while ``complete`` is True -- a valid manifest that is not a
    finished run, on disk. The 4 is the identity's ``n_runs``, which is where
    ``Summary.batches_planned`` comes from, so "3/4" on the row is read off the manifest and not
    assembled by the table. It is written with NO ``rows``, because that is the only state the real
    writer can be in mid-run: ``training_checkpoint.save`` passes none and only ``mark_complete``
    records them. A fixture that handed rows to an unfinished cache would be a shape no run
    produces, and would hide every "rows only once it finished" branch in both front ends.

    No artifact gets a ``log.txt``: the writer writes one only when a public entry is active
    (``runs.current_run_log()``), and this helper is not one. A test that wants records writes the
    file itself, which is exactly what ``read_log`` reads.

    The three bad directories all sit under ``priors/``, one per shape ``ArtifactStore._entries``
    classifies: no manifest at all, a manifest that will not parse, and a valid manifest of another
    kind (the calibration's own bytes, so the shape is real rather than hand-rolled).
    """
    from core.artifacts import ArtifactStore, write_simulation_manifest
    from core.artifacts.store import MANIFEST
    from core.SBI.training_checkpoint import identity_digest

    store = ArtifactStore(root)
    bodies = {
        "prior": {"gmm": {"n_components": 2, "param_keys": ["k"],
                          "box": {"nd_lows": [0.0], "nd_highs": [1.0], "log_mask": [False]}},
                  "sweep": {}, "stability": {"accepted_sets": None, "iterations": 1}},
        "posterior": {"mode": "chi", "conditioning": {"width": 61, "forcing_dim": 12},
                      "transform": {}, "amortized": True, "truncation": None, "training": {}},
        "observation": {"mode": "spontaneous", "conditioning": {"width": 50, "forcing_dim": 1},
                        "x_obs_digest": "0" * 16, "T_obs_cell": 4.5, "n_obs": None,
                        "forcing_vals": {}, "chi_obs_freqs": None, "source": {"kind": "bench"}},
        "calibration": {"results": {}},
        "inference": {"results": {}},
        "diagnostic": {"diagnostic": "identifiability", "variant": "laplace",
                       "settings": {}, "results": {}},
        # A FINISHED sweep, so the fdt row carries content in both of its own columns: a study name
        # and a point fraction with a failure in it. ``complete`` True because the writer commits
        # cleanly here; an UNFINISHED record is a different shape and the tests that need one build
        # it themselves rather than making this fixture carry two.
        "fdt": {"study": "sweep", "settings": {"n_freqs": 2, "preset": "fast"}, "seed": 7,
                "grid": None, "points": {"param": "S", "planned": 3, "done": 2, "failed": 1},
                "offgrid": {"blanks": 0, "of": 6}, "notices": [], "compared": None,
                "complete": True, "results": {"peak_ratio": 2.0}},
    }
    ids = {}
    for kind, body in bodies.items():
        with store.create(kind, None, name=f"browse_{kind}", note=f"the {kind} row") as w:
            w.body = dict(body)
        ids[kind] = w.id

    identity = {"format": "training-rows/2", "prior_fingerprint": "f" * 16, "n_runs": 4,
                "truncation": None}
    m = write_simulation_manifest(store.kind_dir("simulation") / identity_digest(identity), identity,
                                  batches_done=3, complete=False)
    ids["simulation"] = m.id

    priors = store.kind_dir("prior")
    bad = ("leftover_no_manifest", "leftover_bad_json", "leftover_wrong_kind")
    for name in bad:
        (priors / name).mkdir(parents=True, exist_ok=True)
    (priors / bad[1] / MANIFEST).write_text("{not json", encoding="utf-8")
    (priors / bad[2] / MANIFEST).write_text(
        (store.path("calibration", ids["calibration"]) / MANIFEST).read_text(encoding="utf-8"),
        encoding="utf-8")
    ids["bad"] = bad
    return ids


class _FdtStop(Exception):
    """Ends a record's write block where a cancel or a crash would, so ``build_fdt_record`` can leave
    an UNFINISHED record on disk without pretending to fail inside the store."""


def build_fdt_record(store, *, study="single", name="", note="", finished=True,
                     omegas=(0.5, 1.0, 2.0, 4.0), ratio=(1.0, 3.0, 1.2, 1.05),
                     omega_0=1.0, prefactor=2.0, settings=None, sweep_param="s", points=None,
                     seed=7):
    """One ``fdt`` record with a real manifest, a real ``data.h5`` and one figure, written in seconds.

    ``cfg=None``, like ``build_browse_store``'s rows: nothing here reads a bounds file, builds a
    SimConfig or simulates. The dataset names are ``cross_validation._fdt_measure``'s own vocabulary
    (``omega_grid``, ``T_eff_over_T``), which is what the single-cell run writes and what
    ``core.FDT.compare`` reads back. A ``study="sweep"`` record holds the sweep's layout instead: one
    group per entry of ``points``, swept in ``sweep_param``, each on the ``omegas`` axis; with no
    ``points`` it holds an empty group, a sweep in which no operating point finished.
    ``finished=False`` leaves the record unfinished on disk with its numbers already written -- the
    state the store keeps an interrupted record in, and the one a comparison refuses. ``seed`` is the
    seed the body records: 7 unless a test gives each record its own, which a repeats comparison
    reads -- records that share one are one run drawn twice.

    The figure is drawn on a bare ``matplotlib.figure.Figure``, never through pyplot, so a fixture
    that writes dozens of records leaves no open figure and touches no backend.

    :returns: the record's id.
    """
    import h5py
    import numpy as np
    from matplotlib.figure import Figure
    w = store.create("fdt", None, name=name, note=note)
    w.body = {"study": study, "settings": dict(settings or {"n_freqs": len(omegas), "F0": 0.05}),
              "seed": seed, "grid": None, "points": None, "offgrid": None, "notices": [],
              "compared": None, "complete": False, "results": None}
    try:
        with w:
            with h5py.File(w.payload("data.h5"), "w") as h5:
                h5.attrs["study"] = study
                h5.attrs["omega_0"] = float(omega_0)
                h5.attrs["prefactor"] = float(prefactor)
                if study == "sweep":
                    # The layout load_param_sweep reads, which is what the sweep record holds: one
                    # group per operating point, each with the shared omega/omega_0 axis and its own
                    # ratio row. `points` is [(param_value, ratio-row), ...].
                    h5.attrs["sweep_param"] = str(sweep_param)
                    h5.attrs["omega_0_ref"] = float(omega_0)
                    ops = h5.create_group("operating_points")
                    for idx, (value, row) in enumerate(points or []):
                        grp = ops.create_group(f"{idx:03d}")
                        grp.attrs["param_value"] = float(value)
                        grp.attrs["sweep_param"] = str(sweep_param)
                        grp.attrs["omega_0_resonance"] = float(omega_0)
                        grp.attrs["omega_0_ref"] = float(omega_0)
                        grp.attrs["is_resonant"] = True
                        grp.attrs["failed"] = False
                        grp.create_dataset("omega_norm", data=np.asarray(omegas, dtype=np.float64))
                        grp.create_dataset("omega_grid", data=np.asarray(omegas, dtype=np.float64))
                        grp.create_dataset("T_eff_over_T", data=np.asarray(row, dtype=np.float64))
                else:
                    h5.create_dataset("omega_grid", data=np.asarray(omegas, dtype=np.float64))
                    h5.create_dataset("T_eff_over_T", data=np.asarray(ratio, dtype=np.float64))
            fig = Figure(figsize=(2, 2))
            ax = fig.add_subplot()
            # a sweep's rows, not `ratio`: a sweep is built on its own `omegas`, which need not be as
            # long as the single-cell default ratio
            for row in ([r for _v, r in points or []] if study == "sweep" else [ratio]):
                ax.plot(omegas, row)
            fig.savefig(w.figure_path("T_eff over T"), dpi=40)
            if not finished:
                raise _FdtStop("interrupted after the numbers were written")
    except _FdtStop:
        pass
    return w.id


def compare_preflight_refusals(store) -> list:
    """Every refusal ``core.FDT.compare``'s per-mode PRE-FLIGHT makes, over records this writes into
    ``store``, as ``(label, mode, refs, options, field, words)``: ``words`` is a fragment of the
    refusal and ``field`` its key.

    These are the refusals a comparison can only make once it has READ the records it was given --
    whether their curves share a band, what constant a record holds, what a sweep swept and over what
    range, where its slice falls -- and so, before the pre-flight, were made by the drawer inside the
    comparison's already-open record. Shared by the API's and the tool's tests, so both drive the
    same inputs. A "hollow" record is finished and has a data file that holds none of its study's
    numbers."""
    import h5py

    def hollow(name, study):
        w = store.create("fdt", None, name=name)
        w.body = {"study": study, "settings": {}, "seed": 7, "grid": None, "points": None,
                  "offgrid": None, "notices": [], "compared": None, "complete": False,
                  "results": None}
        with w:
            with h5py.File(w.payload("data.h5"), "w") as h5:
                h5.attrs["study"] = study
        return w.id

    om = (0.5, 1.0, 2.0)
    a = build_fdt_record(store, name="pf_a")
    far = build_fdt_record(store, name="pf_far", omegas=(100.0, 200.0), ratio=(1.0, 1.0))
    single_hollow = hollow("pf_hollow", "single")
    no_constant = build_fdt_record(store, name="pf_nan", prefactor=float("nan"))
    zero_constant = build_fdt_record(store, name="pf_zero", prefactor=0.0)
    sweep_hollow = hollow("pf_sweep_hollow", "sweep")
    empty = build_fdt_record(store, name="pf_empty", study="sweep", omegas=om)
    low = build_fdt_record(store, name="pf_low", study="sweep", omegas=om,
                           points=[(0.0, (1.0, 1.0, 1.0)), (0.5, (1.0, 3.0, 1.1))])
    high = build_fdt_record(store, name="pf_high", study="sweep", omegas=om,
                            points=[(0.25, (1.0, 2.0, 1.0)), (0.75, (1.0, 6.0, 1.2))])
    apart = build_fdt_record(store, name="pf_apart", study="sweep", omegas=om,
                             points=[(2.0, (1.0, 2.0, 1.0)), (3.0, (1.0, 2.5, 1.0))])
    temp = build_fdt_record(store, name="pf_temp", study="sweep", omegas=om, sweep_param="temp",
                            points=[(1.0, (1.0, 2.0, 1.0)), (1.5, (1.0, 4.0, 1.1))])
    off_band = build_fdt_record(store, name="pf_off_band", study="sweep", omegas=(10.0, 20.0, 40.0),
                                points=[(0.25, (1.0, 2.0, 1.0)), (0.5, (1.0, 4.0, 1.1))])
    return [
        ("cells that share no band", "cells", [a, far], {}, "compare_records",
         "must share a frequency band"),
        ("repeats that share no band", "repeats", [a, far], {}, "compare_records",
         "must share a frequency band"),
        ("a data file without the curve", "cells", [a, single_hollow], {}, "compare_records",
         "'pf_hollow' has no 'omega_grid'"),
        ("no recorded constant", "renormalise", [no_constant], {"prefactor": 3.0},
         "compare_records", "'pf_nan' does not"),
        ("a recorded constant of zero", "renormalise", [zero_constant], {"prefactor": 3.0},
         "compare_records", "'pf_zero' records 0"),
        ("a sweep data file without its points", "sweeps", [sweep_hollow, low], {},
         "compare_records", "'pf_sweep_hollow' has no 'operating_points'"),
        ("no finished operating point", "sweeps", [empty, low], {}, "compare_records",
         "'pf_empty' has none that finished"),
        ("two swept parameters", "sweeps", [low, temp], {}, "compare_records",
         "these sweep s, temp"),
        ("no shared range", "sweeps", [low, apart], {}, "compare_records", "must overlap in s"),
        ("a slice outside the shared range", "sweeps", [low, high], {"at": 9.0}, "slice_at",
         "([0.25, 0.5] in s); got 9"),
        ("slice rows that share no band", "sweeps", [low, off_band], {}, "compare_records",
         "must share a frequency band"),
    ]


def artifact_screen(store):
    """The Artifacts screen wired to ``store`` and refreshed once.

    ``store`` travels through the screen's own ``_store`` / ``_resolved_store()`` seam -- the one
    ``StorePicker`` already uses and the picker tests already monkeypatch -- so no process default is
    swapped and nothing is patched. The screen reads the store only when it is shown or refreshed, so
    the ``refresh()`` here is what puts rows in the table.
    """
    from core.gui.screens.artifact_screen import ArtifactScreen
    qt_app()
    screen = ArtifactScreen(store=store)
    screen.refresh()
    return screen


def _same_value(a, b) -> bool:
    """Equality for one config field: tensors by shape, dtype and every element on the CPU (equal, or
    both NaN); dicts by type, key order and values; lists and tuples by type and elements; NaN equal to
    NaN; everything else by ``==``. Recursive, because a dict or tuple holding a tensor would make a
    plain ``==`` raise on the tensor's truth value instead of answering."""
    import math
    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        if not (isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor) and a.shape == b.shape
                and a.dtype == b.dtype):
            return False
        a, b = a.detach().cpu(), b.detach().cpu()
        if a.is_floating_point() or a.is_complex():
            # torch.equal says NaN != NaN, so an untouched tensor holding one would read as changed
            return bool(((a == b) | (torch.isnan(a) & torch.isnan(b))).all())
        return torch.equal(a, b)
    if isinstance(a, dict) and isinstance(b, dict):
        return type(a) is type(b) and list(a) == list(b) and all(_same_value(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(_same_value(x, y) for x, y in zip(a, b))
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    return a == b


def snapshot_cfg(cfg) -> dict:
    """Everything a run could write on ``cfg``, frozen: every key of ``cfg.__dict__`` except the cached
    properties (``core.sim_config._CACHED`` -- the time grid and the unit registry, which a read
    materialises and a copy recomputes, so their presence says nothing about a write), with tensors
    cloned to the CPU and every other value deep-copied, so a later write through a shared container
    cannot reach the snapshot. Works on any object with a ``__dict__``.

    The tests that pin that every public stage works on a private copy of its config take one before
    a public entry and hand it to ``assert_cfg_unchanged`` after, whether the entry returned, refused
    or raised.
    """
    import copy
    from core.sim_config import _CACHED
    return {key: (value.detach().cpu().clone() if isinstance(value, torch.Tensor) else copy.deepcopy(value))
            for key, value in vars(cfg).items() if key not in _CACHED}


def assert_cfg_unchanged(cfg, snap) -> None:
    """``cfg`` holds exactly the keys ``snap`` recorded and every value compares equal to its snapshot
    (``_same_value``). Keys count as well as values: a stage's write can ADD one (``chi_obs_freqs`` is
    not a dataclass field, and a freshly built config has no such key). The failure names every added,
    removed and changed field, with the before and after of each change -- which is what says which
    write escaped."""
    from core.sim_config import _CACHED
    now = {key: value for key, value in vars(cfg).items() if key not in _CACHED}
    added = sorted(set(now) - set(snap))
    removed = sorted(set(snap) - set(now))
    changed = [key for key in snap if key in now and not _same_value(now[key], snap[key])]
    assert not (added or removed or changed), (
        f"the caller's config was written: added {added}, removed {removed}, changed "
        + ("; ".join(f"{key}: {snap[key]!r} -> {now[key]!r}" for key in changed) or "[]"))
