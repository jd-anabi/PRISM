"""Shared test helpers: the artifact-store stand-ins (piece 1 of the 2026-09-10 hardening programme)
and, since piece 3, the GUI suites' helpers.

A plain module: importing it imports torch and sbi (``DirectPosterior`` has to be imported at module
level for ``_FakeDP`` to pickle) and does NOTHING ELSE -- no simulation, no file I/O, no torch
seeding, no store writes. Everything else here is a function or class definition whose own imports
stay lazy, exactly as they were before the move, so a bare ``import tests._fixtures`` costs the torch
import and nothing more.
Moved out of ``tests/test_artifact_store.py`` (Task 7) so ``tests/test_user_sbi.py``,
``tests/test_nav_and_gating.py`` and others can share them without importing a whole other test
module (which pytest would then also collect a second time under a different name).

Piece 3 (Task 9) added what the GUI suites used to copy per file -- ``qt_app``, ``pump``,
``code_only``, ``PaneCapture`` -- and the dialog record ``SHOWN`` that tests/conftest.py's session
guard appends to. ONE import spelling, ``tests._fixtures`` (pytest.ini puts the repo root on the
path; tests/ has no __init__.py): ``from _fixtures import SHOWN`` would import this module a second
time under another name, with a second, empty SHOWN that the guard never touches.
"""
import ast
import inspect
import textwrap
import time

import torch

from core.artifacts import manifest as mf

# The top-level repository directories that hold PRODUCT code, and therefore the only ones the source
# scans walk: the literal-path scan (tests/test_artifact_store.py), and the rotation-reader and
# input() scans (both tests/test_conditioning_repair.py). Only the rotation-reader scan ever walked
# scripts/ (the other two always covered "core" alone); piece 2 folded the scripts into
# `python -m core <subcommand>` and removed scripts/ -- the tool's own code lives under core/tool,
# so one root now covers every scan and both front ends. tests/ and the gitignored archive/ are
# deliberately absent: they are not shipped code. test_the_source_scans_cover_every_code_directory
# keeps this set closed, so a new top-level package cannot appear and be scanned by nothing.
CODE_ROOTS: tuple[str, ...] = ("core",)

# The top-level repository files (outside any CODE_ROOTS directory) that hold product code, so the
# literal-path scan walks these too, and the same guard test keeps this set closed as well.
CODE_FILES: tuple[str, ...] = ("conftest.py",)

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
    rendering it: ``PaneCapture(panel)`` replaces ``panel.log_pane.append_line`` on that one pane
    (LogPane.append_line(text, level="info")). The window's refusals and warnings are asserted off
    ``.lines``; the widget receives nothing, and the panel is a throwaway built by the test."""
    def __init__(self, panel):
        self.lines: list[tuple[str, str]] = []
        panel.log_pane.append_line = self._append

    def _append(self, text: str, level: str = "info") -> None:
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

    The V1 pins of piece 3 (spec §2.4) take one before a public entry and hand it to
    ``assert_cfg_unchanged`` after, whether the entry returned, refused or raised.
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
