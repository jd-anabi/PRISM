"""The command-line tool (``python -m core``), piece 2 of the 2026-09-10 hardening programme.

Everything runs IN PROCESS through ``main(argv)``. A subprocess would look more like the operator's
command line and would catch less: a handler that swaps the process default store and never restores
it, or a knob flag that quietly stops reaching its stage, is invisible from outside. Every test here
therefore asserts that ``default_store()`` is still the session sandbox when ``main`` returns, and the
knob tests read the keyword a recorder actually received rather than the run's output.
"""
import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.tool import build_parser, main


def _cfg(bounds):
    """The config flags every SBI subcommand takes. CPU always: the suite never asks for the card."""
    return ["--bounds", bounds, "--device", "cpu"]


class _Rec:
    """A stand-in that records every call and returns ``result``."""

    def __init__(self, result=None):
        self.calls, self.result = [], result

    def __call__(self, *a, **k):
        self.calls.append((a, k))
        return self.result


def _art(root, kind):
    """The shape ``config_args.report`` reads off a Loaded* wrapper."""
    return SimpleNamespace(kind=kind, path=Path(root) / f"{kind}s" / "rec__20260101T000000")


@pytest.fixture(scope="module", autouse=True)
def _session_default_store():
    """The session sandbox object, captured before ANY fixture below can run ``main`` for real.

    pytest instantiates same-scope fixtures with autouse ones first (see "Autouse fixtures are
    executed first within their scope" in the pytest docs), so this module-scoped autouse fixture is
    guaranteed to run before ``tool_env``/``tool_run`` -- both module-scoped but only explicitly
    requested -- regardless of test order or a ``-k`` selection. That matters because ``tool_run``
    calls the real ``main`` TWICE during its own fixture setup, before any test body -- and therefore
    before a function-scoped fixture's ``default_store()`` read -- ever runs. Capturing ``before``
    inside a function-scoped fixture would read the store AFTER those two calls, so a leak baked in
    during ``tool_run``'s setup would already be sitting in ``before`` and every later comparison
    would trivially pass (spec Sec. 8.4's gap). Recording it here, ahead of every module fixture, is what
    keeps ``default_store() is <this>`` a real check of the object the session started with.
    """
    from core.artifacts import default_store
    return default_store()


@pytest.fixture(scope="module")
def tool_env(_session_default_store, tmp_path_factory):
    """PRISM_ARTIFACTS at a temp root, the tiny gen_prior stub, and SBITEST installed as real input
    files -- all restored at module teardown. Yields ``(bounds, cell, root)``.

    Depends on ``_session_default_store`` (even though it does not use the value) so the capture above
    is guaranteed to happen first even if pytest's scope/autouse ordering were ever in doubt."""
    from core import orchestrator
    from tests._fixtures import _tiny_gen_prior, install_sbitest
    root = tmp_path_factory.mktemp("tool_artifacts")
    bounds, cell, teardown = install_sbitest()
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("PRISM_ARTIFACTS", str(root))
        mp.setattr(orchestrator.pipeline, "gen_prior", _tiny_gen_prior)
        try:
            yield str(bounds), str(cell), root
        finally:
            teardown()


@pytest.fixture(scope="module")
def tool_run(tool_env, _session_default_store):
    """A real prior (``tp``) and a real posterior (``tpost``) written BY THE TOOL, at tiny size.
    ``--checkpoint-every 1`` is explicit: the session default is off (tests/conftest.py).

    These are the only in-process calls to the real stages that happen OUTSIDE a test body (during
    module-fixture setup), so the store-survives check is repeated right here, immediately after them,
    rather than trusting the later function-scoped fixture alone to catch a leak from this setup."""
    from core.artifacts import default_store
    bounds, cell, root = tool_env
    assert main(["prior", *_cfg(bounds), "--name", "tp"]) == 0
    assert main(["train", *_cfg(bounds), "--prior", "tp", "--name", "tpost", "--num-runs", "2",
                 "--run-size", "8", "--hidden-features", "8", "--num-transforms", "1",
                 "--stop-after-epochs", "1", "--checkpoint-every", "1"]) == 0
    assert default_store() is _session_default_store, \
        "tool_run's own two real `main` calls left the tool's store as the process default"
    return bounds, cell, root


@pytest.fixture(autouse=True)
def _default_store_is_restored(_session_default_store):
    """``main`` runs its handler under ``use_store`` and must put the process default back. A tool that
    leaked its store would make every later test in the session write into the tool's root, and only a
    SINGLE-PROCESS gate could ever notice (CLAUDE.md); this makes it a per-test failure instead.

    Compared against ``_session_default_store`` -- captured once, before any module fixture's real
    ``main`` calls -- rather than a fresh ``default_store()`` read here, so a leak already baked into a
    per-test ``before`` by ``tool_run``'s fixture setup cannot hide behind it."""
    from core.artifacts import default_store
    yield
    assert default_store() is _session_default_store, "the tool left its own store as the process default"


def test_the_help_epilog_names_the_core_environment_settings():
    """D13: two roots and two core-level settings, named where an operator looks -- and nowhere else,
    because the tool itself reads none of them."""
    epilog = build_parser().epilog
    for name in ("PRISM_RESOURCES", "PRISM_ARTIFACTS", "PRISM_VRAM_CEILING_GIB", "PRISM_MEM_LOG_EVERY"):
        assert name in epilog, name


def test_the_tool_reads_no_environment_and_writes_no_knob():
    """D6: every setting is a flag that travels to its stage as a keyword.

    An ``os.environ`` read inside the tool would be a knob no flag names, and an assignment to a
    module constant is the trap CLAUDE.md spells out -- orchestrator binds config constants at import,
    so ``config.X = v`` changes nothing and the run silently uses the default. The entry is the one
    exception, and only for the two lines that MUST precede torch.
    """
    root = Path(__file__).resolve().parents[1]
    modules = sorted((root / "core" / "tool").rglob("*.py"))
    assert modules, "core/tool/ holds no modules"
    for py in modules:
        tree = ast.parse(py.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr in ("environ", "getenv"):
                pytest.fail(f"{py.relative_to(root)}:{node.lineno} reads the environment")
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Attribute) and (target.attr.isupper()
                                                              or target.attr == "batch_size"):
                        pytest.fail(f"{py.relative_to(root)}:{node.lineno} assigns {target.attr}")

    entry = ast.parse((root / "core" / "__main__.py").read_text(encoding="utf-8"))
    setdefaults, agg, core_imports = [], [], []
    for node in ast.walk(entry):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.args:
            first = getattr(node.args[0], "value", None)
            if node.func.attr == "setdefault" and first == "KMP_DUPLICATE_LIB_OK":
                setdefaults.append(node.lineno)
            if node.func.attr == "use" and first == "Agg":
                agg.append(node.lineno)
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [getattr(node, "module", None) or ""] + [a.name for a in node.names]
            if any(n.split(".")[0] == "core" for n in names):
                core_imports.append(node.lineno)
    assert len(setdefaults) == 1 and len(agg) == 1, (setdefaults, agg)
    assert core_imports, "core/__main__.py imports nothing from core"
    assert max(setdefaults + agg) < min(core_imports), \
        "KMP_DUPLICATE_LIB_OK and the Agg backend must be set BEFORE the first core import"


def test_every_prior_flag_reaches_build_prior_as_a_keyword(tool_env, monkeypatch):
    from core import orchestrator
    bounds, cell, root = tool_env
    rec = _Rec(_art(root, "prior"))
    monkeypatch.setattr(orchestrator, "build_prior", rec)

    assert main(["prior", *_cfg(bounds), "--name", "n1", "--note", "hello",
                 "--num-iterations", "7", "--sweep-batch", "9", "--max-sets", "11",
                 "--walk-step", "0.25", "--stability-units", "3.5",
                 "--min-cluster-size", "4", "--min-samples", "5"]) == 0
    (cfg, ref, build_new), kw = rec.calls[0]
    assert ref is None and build_new is True
    assert cfg.model == "SBITEST" and cfg.T_obs is None, "the builder never sets T_obs"
    assert cfg.hw.device.type == "cpu"
    assert kw["name"] == "n1" and kw["note"] == "hello" and kw["store"] is not None
    assert (kw["num_iterations"], kw["sweep_batch"], kw["max_sets"]) == (7, 9, 11)
    assert (kw["walk_step"], kw["stability_units"]) == (0.25, 3.5)
    assert (kw["min_cluster_size"], kw["min_samples"]) == (4, 5)

    rec.calls.clear()
    assert main(["prior", *_cfg(bounds)]) == 0
    assert set(rec.calls[0][1]) == {"name", "note", "fig_sink", "store"}, \
        "an unset knob must not be forwarded -- the stage's own default is the one default"


def test_every_train_flag_reaches_build_posterior_as_a_keyword(tool_env, monkeypatch):
    from core import orchestrator
    bounds, cell, root = tool_env
    loaded_prior = _art(root, "prior")
    prior_rec, post_rec = _Rec(loaded_prior), _Rec(_art(root, "posterior"))
    monkeypatch.setattr(orchestrator, "build_prior", prior_rec)
    monkeypatch.setattr(orchestrator, "build_posterior", post_rec)

    assert main(["train", *_cfg(bounds), "--prior", "tp", "--name", "n2", "--note", "b",
                 "--num-runs", "3", "--run-size", "8", "--hidden-features", "16",
                 "--num-transforms", "2", "--learning-rate", "0.001", "--stop-after-epochs", "4",
                 "--max-epochs", "6", "--fisher-m", "12", "--fisher-dz", "0.05",
                 "--fisher-points", "2", "--checkpoint-every", "1", "--resume", "require",
                 "--new-run"]) == 0
    (_, p_ref, p_new), _ = prior_rec.calls[0]
    assert p_ref == "tp" and p_new is False, "train LOADS its prior"
    (cfg, prior, ref, train_new), kw = post_rec.calls[0]
    assert prior is loaded_prior and ref is None and train_new is True
    assert set(kw) == {"name", "note", "fig_sink", "store", "new_run", "num_runs", "run_size_cap",
                       "hidden_features", "num_transforms", "learning_rate", "stop_after_epochs",
                       "max_num_epochs", "fisher_m", "fisher_dz", "fisher_points",
                       "checkpoint_every", "resume"}
    assert (kw["num_runs"], kw["run_size_cap"]) == (3, 8)
    assert (kw["hidden_features"], kw["num_transforms"]) == (16, 2)
    assert (kw["learning_rate"], kw["stop_after_epochs"], kw["max_num_epochs"]) == (0.001, 4, 6)
    assert (kw["fisher_m"], kw["fisher_dz"], kw["fisher_points"]) == (12, 0.05, 2)
    assert kw["checkpoint_every"] == 1 and kw["resume"] == "require" and kw["new_run"] is True

    post_rec.calls.clear()
    assert main(["train", *_cfg(bounds), "--prior", "tp"]) == 0
    kw = post_rec.calls[0][1]
    assert set(kw) == {"name", "note", "fig_sink", "store", "new_run"}
    assert kw["new_run"] is False, "a boolean is always passed; a value flag only when it is set"


def test_usage_errors_exit_2(tool_env, capsys):
    bounds, cell, root = tool_env
    assert main(["nosuchcommand"]) == 2
    assert main([]) == 2
    assert main(["prior", "--device", "cpu"]) == 2          # --bounds is required everywhere
    assert main(["train", *_cfg(bounds)]) == 2              # --prior is required
    capsys.readouterr()
    assert main(["--help"]) == 0
    assert "PRISM_ARTIFACTS" in capsys.readouterr().out


def test_a_taken_name_is_refused_with_exit_1_and_nothing_written(tool_run, capsys):
    from core.artifacts import ArtifactStore
    bounds, cell, root = tool_run
    store = ArtifactStore(root)
    before = [s.id for s in store.list("prior")]
    capsys.readouterr()
    assert main(["prior", *_cfg(bounds), "--name", "tp"]) == 1
    err = capsys.readouterr().err
    assert "refused: StoreError" in err and "already exists" in err and "raised at" in err
    assert [s.id for s in store.list("prior")] == before
