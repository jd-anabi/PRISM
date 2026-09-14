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


def test_parse_forced_and_recording_set_rules():
    """Section 3.6, as a unit: which recordings each observation mode takes, decided before anything
    is loaded or spent. The stub configs carry only the two fields the function is allowed to read."""
    from core.tool.config_args import UsageError, parse_forced, recording_set

    assert parse_forced("rec.npy") == ("rec.npy", None)
    assert parse_forced("rec.npy@12.5") == ("rec.npy", 12.5)
    assert parse_forced(r"C:\runs\a@b\rec.npy@40") == (r"C:\runs\a@b\rec.npy", 40.0), "split at the LAST @"
    with pytest.raises(UsageError, match="PATH@HZ"):
        parse_forced("rec.npy@fast")

    def _args(**over):
        d = dict(spont="s.npy", forced=None, drive=None, f0_si=None, t_obs_s=2.0)
        d.update(over)
        return SimpleNamespace(**d)

    chi = SimpleNamespace(observation_mode="chi", force_params_dict={"amp": None})
    with pytest.raises(UsageError, match="at least one"):
        recording_set(chi, _args(f0_si=1e-12))
    with pytest.raises(UsageError, match="frequency"):
        recording_set(chi, _args(forced=["a.npy"], f0_si=1e-12))
    with pytest.raises(UsageError, match="--f0-si"):
        recording_set(chi, _args(forced=["a.npy@10"]))
    rec = recording_set(chi, _args(forced=["a.npy@10", "b.npy@20"], f0_si=1e-12))
    assert rec.forced == (("a.npy", 10.0), ("b.npy", 20.0))
    assert rec.F0_si == 1e-12 and rec.T_obs_s == 2.0 and rec.forcing_params_si is None

    spont = SimpleNamespace(observation_mode="spontaneous", force_params_dict={})
    assert recording_set(spont, _args()).forced == ()
    with pytest.raises(UsageError, match="no Forcing section"):
        recording_set(spont, _args(forced=["a.npy"]))

    forced = SimpleNamespace(observation_mode="forced", force_params_dict={"amp": None, "freq": None})
    with pytest.raises(UsageError, match="exactly one"):
        recording_set(forced, _args(drive=["amp=1e-12", "freq=30"]))
    with pytest.raises(UsageError, match="exactly the drive"):
        recording_set(forced, _args(forced=["a.npy"], drive=["amp=1e-12"]))
    with pytest.raises(UsageError, match="NAME=VALUE"):
        recording_set(forced, _args(forced=["a.npy"], drive=["amp"]))
    rec = recording_set(forced, _args(forced=["a.npy"], drive=["amp=1e-12", "freq=30"]))
    assert rec.forced == (("a.npy", None),) and rec.forcing_params_si == {"amp": 1e-12, "freq": 30.0}


def test_infer_usage_errors_exit_2(tool_run, capsys):
    bounds, cell, root = tool_run
    base = ["infer", *_cfg(bounds), "--posterior", "tpost", "--t-obs", "1.0"]
    assert main([*base, "--cell", cell, "--spont", "x.npy"]) == 2     # mutually exclusive
    assert main([*base]) == 2                                        # and one of them is required
    capsys.readouterr()
    assert main([*base, "--spont", "x.npy", "--forced", "y.npy"]) == 2
    err = capsys.readouterr().err
    assert "usage:" in err and "no Forcing section" in err


def test_validate_and_simulated_infer(tool_run):
    """Both load the posterior AND the prior that posterior was trained on -- never a prior the
    operator names, which build_posterior could only ever refuse."""
    from core.artifacts import ArtifactStore
    bounds, cell, root = tool_run
    store = ArtifactStore(root)
    assert main(["validate", *_cfg(bounds), "--posterior", "tpost", "--n-cal", "60",
                 "--name", "tcal"]) == 0
    assert main(["infer", *_cfg(bounds), "--posterior", "tpost", "--cell", cell,
                 "--t-obs", "1.0", "--name", "tinf"]) == 0
    post_id, prior_id = store.get("posterior", "tpost").id, store.get("prior", "tp").id
    assert store.get("calibration", "tcal").parents == {"posterior": post_id, "prior": prior_id}
    inf = store.get("inference", "tinf")
    assert inf.parents["posterior"] == post_id and inf.parents["observation"]
    assert inf.body["results"]["accepted"] == []


def test_near_miss_is_refused_before_simulation(tool_run, capsys):
    """D7 on the command line: the 2026-09-11 incident, in which run 2 silently started a new cache
    under an identity one field away from run 1's, is now a refusal with no spend."""
    from core.artifacts import ArtifactStore
    bounds, cell, root = tool_run
    store = ArtifactStore(root)
    sims = Path(root) / "simulations"
    sims_before = {p.name for p in sims.iterdir()}
    posts_before = {s.id for s in store.list("posterior")}
    capsys.readouterr()
    assert main(["train", *_cfg(bounds), "--prior", "tp", "--num-runs", "3", "--run-size", "8",
                 "--hidden-features", "8", "--num-transforms", "1", "--stop-after-epochs", "1",
                 "--checkpoint-every", "1"]) == 1
    err = capsys.readouterr().err
    assert "ONE setting away" in err and "n_runs" in err and "--new-run" in err
    assert {p.name for p in sims.iterdir()} == sims_before, "a new cache was started anyway"
    assert {s.id for s in store.list("posterior")} == posts_before


@pytest.fixture(scope="module")
def tool_round(tool_run, _session_default_store):
    """One real TSNPE round through the tool: the observation it was drawn around, and ``tround``, a
    NON-AMORTIZED posterior. Shared by the round's own test and by the refusal test below, because a
    truncated artifact that the store will actually load has to be made by a real round."""
    from core.artifacts import ArtifactStore, default_store
    bounds, cell, root = tool_run
    store = ArtifactStore(root)
    assert main(["infer", *_cfg(bounds), "--posterior", "tpost", "--cell", cell,
                 "--t-obs", "1.0", "--name", "tround_inf"]) == 0
    obs = store.get("inference", "tround_inf").parents["observation"]
    # --directions 3: SBITEST's latent is 4 wide (2 ND params + 2 rescale), and tsnpe_round's own
    # default (truncate.DEFAULT_N_DIRECTIONS = 5) refuses on this tiny model ("5 directions requested
    # but the latent has 4") -- a real orchestrator guardrail, not a tool bug. 3 leaves one flat
    # direction, as the guardrail intends.
    assert main(["tsnpe", *_cfg(bounds), "--posterior", "tpost", "--observation", obs,
                 "--directions", "3", "--num-runs", "1", "--run-size", "8", "--hidden-features", "8",
                 "--num-transforms", "1", "--stop-after-epochs", "1", "--checkpoint-every", "1",
                 "--new-run", "--name", "tround"]) == 0
    assert default_store() is _session_default_store, \
        "tool_round's own two real `main` calls left the tool's store as the process default"
    return bounds, cell, root, obs


def test_tsnpe_subcommand_runs_a_round(tool_round):
    from core.artifacts import ArtifactStore
    bounds, cell, root, obs = tool_round
    store = ArtifactStore(root)
    m = store.get("posterior", "tround")
    assert m.body["amortized"] is False
    assert m.body["truncation"]["x_obs_digest"] == store.get("observation", obs).body["x_obs_digest"]
    assert m.parents["observation"] == obs
    assert m.parents["parent_posterior"] == store.get("posterior", "tpost").id
    assert m.parents["prior"] == store.get("prior", "tp").id, "a round records its BASE prior"


def test_validate_refuses_a_non_amortized_posterior_without_accept_truncated(tool_round, monkeypatch,
                                                                             capsys):
    from core import orchestrator
    bounds, cell, root, obs = tool_round
    argv = ["validate", *_cfg(bounds), "--posterior", "tround", "--n-cal", "60"]
    capsys.readouterr()
    assert main(argv) == 1
    err = capsys.readouterr().err
    assert "NOT AMORTIZED" in err and "--accept-truncated" in err and "raised at" in err

    rec = _Rec(_art(root, "calibration"))
    monkeypatch.setattr(orchestrator, "validate_calibration", rec)
    assert main([*argv, "--accept-truncated"]) == 0
    (cfg, posterior, prior), kw = rec.calls[0]
    assert posterior.posterior.truncation is not None
    assert posterior.accepted == ["truncated"], "the hatch used is recorded on the wrapper"
    assert kw["n_cal"] == 60


def test_ctrl_c_mid_simulation_keeps_the_committed_batches(tool_run, monkeypatch, capsys):
    """Ctrl-C is not a refusal and not a bug: the pipeline commits the batches it finished, the
    writer removes the artifact it had opened, and the next run with --resume require continues."""
    from core.SBI import pipeline
    from core.artifacts import ArtifactStore
    bounds, cell, root = tool_run
    store = ArtifactStore(root)
    posts_before = {s.id for s in store.list("posterior")}
    real, calls = pipeline._rows_with_oom_retry, []

    def _interrupt(fn, lo, hi, **kw):
        calls.append(1)
        if len(calls) == 3:
            raise KeyboardInterrupt
        return real(fn, lo, hi, **kw)

    monkeypatch.setattr(pipeline, "_rows_with_oom_retry", _interrupt)
    argv = ["train", *_cfg(bounds), "--prior", "tp", "--num-runs", "4", "--run-size", "8",
            "--hidden-features", "8", "--num-transforms", "1", "--stop-after-epochs", "1",
            "--checkpoint-every", "1", "--new-run"]
    capsys.readouterr()
    assert main(argv) == 130
    assert "interrupted" in capsys.readouterr().err
    assert {s.id for s in store.list("posterior")} == posts_before, "no half-posterior survived"
    partial = [m for m in (store.get("simulation", s.id) for s in store.list("simulation"))
               if m.body["complete"] is False]
    assert len(partial) == 1 and partial[0].body["batches_done"] == 2

    monkeypatch.undo()
    capsys.readouterr()
    assert main([*argv, "--resume", "require"]) == 0
    assert "resuming at batch 2/4" in capsys.readouterr().out


def test_every_validate_infer_and_tsnpe_flag_reaches_its_stage_as_a_keyword(tool_round, monkeypatch):
    """Spec 3.4: a flag's dest IS the stage's keyword name, for the three subcommands T11 did not
    cover. The real runs above would catch a wrong dest only for the flags they happen to pass -- and
    they pass none of these, so a rename like --posterior-samples -> num_posterior_samples could drift
    silently until a TypeError surfaced from **knobs() in front of an operator.
    """
    from core import orchestrator
    bounds, cell, root, obs = tool_round
    v = _Rec(_art(root, "calibration"))
    i = _Rec((_art(root, "observation"), _art(root, "inference")))
    t = _Rec(_art(root, "posterior"))
    monkeypatch.setattr(orchestrator, "validate_calibration", v)
    monkeypatch.setattr(orchestrator, "simulated_inference", i)
    monkeypatch.setattr(orchestrator, "tsnpe_round", t)

    assert main(["validate", *_cfg(bounds), "--posterior", "tpost", "--n-cal", "7",
                 "--cal-n-scales", "3", "--posterior-samples", "11"]) == 0
    assert v.calls[0][1]["n_cal"] == 7 and v.calls[0][1]["cal_n_scales"] == 3
    assert v.calls[0][1]["num_posterior_samples"] == 11

    assert main(["infer", *_cfg(bounds), "--posterior", "tpost", "--cell", cell,
                 "--t-obs", "2.5", "--n-samples", "13"]) == 0
    (_c, _p, t_obs), kw = i.calls[0]
    assert t_obs == 2.5 and kw["n_samples"] == 13

    assert main(["tsnpe", *_cfg(bounds), "--posterior", "tpost", "--observation", obs,
                 "--directions", "2", "--level", "0.99", "--learning-rate", "0.002",
                 "--max-epochs", "3"]) == 0
    kw = t.calls[0][1]
    assert (kw["n_directions"], kw["level"]) == (2, 0.99)
    assert (kw["learning_rate"], kw["max_num_epochs"]) == (0.002, 3)
