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

from core import config
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
    assert main(["validate", "--posterior", "x"]) == 2      # --bounds is required for validate too
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
    with pytest.raises(UsageError, match="frequency") as exc:
        recording_set(chi, _args(forced=["a.npy"], f0_si=1e-12))
    assert "(D9)" in str(exc.value), "spec 3.6 requires the guardrail id in the message"
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

    # R-L's ordering, PROVED rather than assumed: --posterior names a ref that does not exist, so the
    # only way this can return 2 (recording_set's UsageError) rather than 1
    # (load_posterior_and_prior's StoreError) is if the recording rules are checked BEFORE the load.
    # Moving recording_set after the load would silently turn this into a 1.
    bad_ref = ["infer", *_cfg(bounds), "--posterior", "nosuch", "--t-obs", "1.0"]
    capsys.readouterr()
    assert main([*bad_ref, "--spont", "x.npy", "--forced", "y.npy"]) == 2
    err = capsys.readouterr().err
    assert "usage:" in err and "no Forcing section" in err

    # --cell with any of the experimental-only recording flags: I3, refused before the load too.
    capsys.readouterr()
    assert main([*bad_ref, "--cell", cell, "--forced", "y.npy@10", "--f0-si", "1e-12"]) == 2
    err = capsys.readouterr().err
    assert "usage:" in err and "--forced" in err and "--f0-si" in err


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
    assert "this run 3" in err and "that cache 2" in err, \
        "the refusal must name BOTH values, not just the field -- read literally off the message"
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
    """Ctrl-C is not a refusal and not a bug: what this test actually proves is that the batches the
    pipeline already committed survive the interrupt, that no half-trained posterior is ever written
    (posts_before is unchanged after the 130), and that the same command with --resume require
    continues training from batch 2 rather than restarting."""
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
    cover. Pins the WHOLE keyword mapping each call produces -- ``set(kw) ==`` the exact set, as T11's
    own ``test_every_train_flag_reaches_build_posterior_as_a_keyword`` does -- not a sample of it:
    ``knobs()`` silently DROPS a dest that no longer matches a flag (a rename like ``--run-size``
    losing ``dest="run_size_cap"``, or a typo in ``getattr(args, "accept_other_observation", False)``),
    so checking only a few keys would stay green through that drift; only the exact set catches it.

    Also covers the two recording paths (``--cell`` and ``--spont``) and the accept hatches: what a
    composition (``simulated_inference``/``experimental_inference``) receives as ``accept=``, and --
    separately -- what ``load_posterior_and_prior`` itself receives for ``validate``/``tsnpe``, which
    reach it as the ONLY place they use ``accept`` (their own compositions take no ``accept=`` of their
    own).
    """
    from core.artifacts import Accept
    from core.SBI.observations import RecordingSet
    from core import orchestrator
    from core.tool import stages
    bounds, cell, root, obs = tool_round
    v = _Rec(_art(root, "calibration"))
    i = _Rec((_art(root, "observation"), _art(root, "inference")))
    e = _Rec((_art(root, "observation"), _art(root, "inference")))
    t = _Rec(_art(root, "posterior"))
    monkeypatch.setattr(orchestrator, "validate_calibration", v)
    monkeypatch.setattr(orchestrator, "simulated_inference", i)
    monkeypatch.setattr(orchestrator, "experimental_inference", e)
    monkeypatch.setattr(orchestrator, "tsnpe_round", t)

    # validate: every VALIDATE_KNOBS flag, a distinct value each.
    assert main(["validate", *_cfg(bounds), "--posterior", "tpost", "--n-cal", "7",
                 "--cal-n-scales", "3", "--posterior-samples", "11"]) == 0
    (_cfg1, _post1, _prior1), kw = v.calls[0]
    assert set(kw) == {"name", "note", "fig_sink", "store", "n_cal", "cal_n_scales",
                       "num_posterior_samples"}
    assert (kw["n_cal"], kw["cal_n_scales"], kw["num_posterior_samples"]) == (7, 3, 11)

    # infer --cell: the simulated composition's full keyword set.
    assert main(["infer", *_cfg(bounds), "--posterior", "tpost", "--cell", cell,
                 "--t-obs", "2.5", "--n-samples", "13"]) == 0
    (_c, _p, t_obs), kw = i.calls[0]
    assert t_obs == 2.5
    assert set(kw) == {"cell", "prior", "accept", "name", "note", "fig_sink", "store", "n_samples"}
    assert kw["n_samples"] == 13 and kw["cell"] == cell and kw["accept"] == Accept()

    # infer --spont: the experimental composition's full keyword set, and the RecordingSet SBITEST's
    # spontaneous mode (no Forcing section, per install_sbitest) actually accepts -- --spont alone, no
    # --forced/--drive/--f0-si. The path need not exist: nothing before the stub reads it.
    assert main(["infer", *_cfg(bounds), "--posterior", "tpost", "--spont", "s.npy",
                 "--t-obs", "3.0", "--n-samples", "17"]) == 0
    (_c2, _p2, rec), kw = e.calls[0]
    assert isinstance(rec, RecordingSet) and rec.spont == "s.npy" and rec.T_obs_s == 3.0
    assert set(kw) == {"accept", "name", "note", "fig_sink", "store", "n_samples"}
    assert kw["n_samples"] == 17 and kw["accept"] == Accept()

    # infer --cell with both accept flags: the composition sees the Accept they build.
    i.calls.clear()
    assert main(["infer", *_cfg(bounds), "--posterior", "tpost", "--cell", cell, "--t-obs", "2.5",
                 "--accept-truncated", "--accept-other-observation"]) == 0
    assert i.calls[0][1]["accept"] == Accept(truncated=True, other_observation=True)

    # tsnpe: every TSNPE_KNOBS flag, a distinct value each, plus --new-run.
    assert main(["tsnpe", *_cfg(bounds), "--posterior", "tpost", "--observation", obs,
                 "--directions", "2", "--level", "0.99", "--num-runs", "6", "--run-size", "9",
                 "--hidden-features", "10", "--num-transforms", "3", "--learning-rate", "0.002",
                 "--stop-after-epochs", "5", "--max-epochs", "3", "--checkpoint-every", "2",
                 "--resume", "require", "--new-run"]) == 0
    (_c3, _p3, _pr3, _obs3), kw = t.calls[0]
    assert set(kw) == {"name", "note", "fig_sink", "store", "new_run", "n_directions", "level",
                       "num_runs", "run_size_cap", "hidden_features", "num_transforms",
                       "learning_rate", "stop_after_epochs", "max_num_epochs", "checkpoint_every",
                       "resume"}
    assert (kw["n_directions"], kw["level"]) == (2, 0.99)
    assert (kw["num_runs"], kw["run_size_cap"]) == (6, 9)
    assert (kw["hidden_features"], kw["num_transforms"]) == (10, 3)
    assert (kw["learning_rate"], kw["stop_after_epochs"], kw["max_num_epochs"]) == (0.002, 5, 3)
    assert kw["checkpoint_every"] == 2 and kw["resume"] == "require" and kw["new_run"] is True

    # The accept hatch on validate/tsnpe reaches only load_posterior_and_prior (their own
    # compositions take no accept= of their own) -- so record what THAT receives, delegating to the
    # real implementation so the load still succeeds. It is imported into core.tool.stages'
    # namespace under its own name, so that is what a rename-proof monkeypatch targets.
    real_load = stages.load_posterior_and_prior
    load_calls = []

    def _record_load(cfg, ref, accept, store):
        load_calls.append(accept)
        return real_load(cfg, ref, accept, store)

    monkeypatch.setattr(stages, "load_posterior_and_prior", _record_load)

    assert main(["validate", *_cfg(bounds), "--posterior", "tpost", "--n-cal", "5",
                 "--accept-truncated"]) == 0
    assert load_calls[-1] == Accept(truncated=True)

    assert main(["validate", *_cfg(bounds), "--posterior", "tpost", "--n-cal", "5"]) == 0
    assert load_calls[-1] == Accept(), "Accept() arrives when the flag is not given"

    assert main(["tsnpe", *_cfg(bounds), "--posterior", "tpost", "--observation", obs,
                 "--directions", "2", "--accept-truncated"]) == 0
    assert load_calls[-1] == Accept(truncated=True)


def test_the_sbc_subcommand_forwards_every_knob_as_a_keyword(tmp_path, monkeypatch, capsys):
    """Each flag reaches sbc_repeats under the stage's own keyword name, and a flag left off is NOT
    passed -- so the default lives in one place (the function signature) instead of being restated by
    the tool. The posterior load is replaced: what is under test is the wiring, not the store."""
    from core import tool
    from core.artifacts import Accept
    from core.tool import diagnostics as tool_diag
    monkeypatch.setenv("PRISM_ARTIFACTS", str(tmp_path / "A"))
    seen = {}

    def _fake_pair(cfg, ref, accept, store):
        seen["ref"], seen["accept"] = ref, accept
        return "POST", "PRIOR"

    def _fake_sbc(cfg, posterior, prior, **kw):
        seen["args"] = (posterior, prior)
        seen["kw"] = kw
        # The shape `config_args.report` actually reads: `.kind` and `.path`, and it prints
        # `a.path.name`, so the directory name has to BE the last path segment.
        return SimpleNamespace(kind="diagnostic", path=tmp_path / "diagnostics" / "d__1")

    monkeypatch.setattr(tool_diag, "load_posterior_and_prior", _fake_pair)
    monkeypatch.setattr("core.diagnostics.sbc_repeats", _fake_sbc)
    bounds = str(config.BOUNDS_PATH / "nadrowski" / "master.txt")
    rc = tool.main(["sbc", "--bounds", bounds, "--device", "cpu", "--posterior", "tpost",
                    "--repeats", "3", "--n-cal", "40", "--posterior-samples", "70",
                    "--cal-n-scales", "2", "--seed", "5", "--accept-truncated",
                    "--name", "sbc1", "--note", "hello"])
    assert rc == 0
    assert seen["ref"] == "tpost" and seen["accept"] == Accept(truncated=True)
    assert seen["args"] == ("POST", "PRIOR")
    # The FULL set, not a sample of it (as T12's own tool tests pin VALIDATE_KNOBS/TSNPE_KNOBS,
    # tests/test_tool.py ~473-508): knobs() silently DROPS a dest that no longer matches a flag, so
    # checking only a few keys would stay green through a knob quietly no longer reaching sbc_repeats,
    # or through one being forwarded unconditionally (e.g. n_cal=args.n_cal beside **knobs(...)),
    # which would crash a real run on int(None) the moment the flag was left off.
    assert set(seen["kw"]) == {"name", "note", "fig_sink", "store", "repeats", "n_cal",
                               "num_posterior_samples", "cal_n_scales", "seed"}
    assert seen["kw"]["repeats"] == 3 and seen["kw"]["n_cal"] == 40
    assert seen["kw"]["num_posterior_samples"] == 70 and seen["kw"]["cal_n_scales"] == 2
    assert seen["kw"]["seed"] == 5 and seen["kw"]["name"] == "sbc1" and seen["kw"]["note"] == "hello"
    assert "chi_k_fixed" not in seen["kw"], "a flag left off must not be passed at all"
    assert "diagnostic d__1" in capsys.readouterr().out

    seen.clear()
    assert tool.main(["sbc", "--bounds", bounds, "--device", "cpu", "--posterior", "p",
                      "--chi", "--chi-k-fixed", "6"]) == 0
    assert set(seen["kw"]) == {"name", "note", "fig_sink", "store", "chi_k_fixed"}
    assert seen["kw"]["chi_k_fixed"] == 6 and seen["accept"] == Accept()
    assert "repeats" not in seen["kw"] and "seed" not in seen["kw"]


def test_identifiability_is_a_nested_subcommand_whose_modes_do_not_share_flags(tmp_path, monkeypatch):
    """The mode is a positional subparser, so a flag that means nothing to a mode is an argparse
    error (exit 2) rather than a silently ignored setting -- `rotation --cell x` is the case that
    matters, because rotation simulates nothing and a cell would never be read.

    R-T/flags-to-keywords: rotation and laplace declare the accept flag with `add_accept_flags` and
    build their Accept with `accept_from`, the single D8 definition (Tasks 12/16) -- so this also
    pins the FULL `set(kw)` each mode's handler forwards, one distinct value per knob, the way T12
    and T16's own tool tests pin VALIDATE_KNOBS/TSNPE_KNOBS/sbc's set: `knobs()` silently drops a
    dest that no longer matches a flag, so checking only a few keys would stay green through that."""
    import pytest
    from core import tool
    from core.tool import diagnostics as tool_diag
    monkeypatch.setenv("PRISM_ARTIFACTS", str(tmp_path / "A"))
    seen, seen_args = {}, {}
    monkeypatch.setattr(tool_diag, "load_posterior_and_prior", lambda cfg, ref, accept, store: ("P", "Q"))

    def _recorder(_n):
        # A NAMED function, not `seen.setdefault(_n, kw) or SimpleNamespace(...)`: setdefault returns
        # the stored value, and kw is never empty (every handler passes name, note, fig_sink and
        # store), so the `or` would short-circuit and hand `report` a dict, which has no `.kind`.
        def _rec(*a, **kw):
            seen[_n] = kw
            seen_args[_n] = a               # M13: the POSITIONAL cfg -- was previously ignored
            return SimpleNamespace(kind="diagnostic", path=tmp_path / "diagnostics" / "d__1")

        return _rec

    for fn in ("identifiability_rotation", "identifiability_laplace", "identifiability_jacobian"):
        monkeypatch.setattr(f"core.diagnostics.{fn}", _recorder(fn))
    # master.txt: there is no Bounds/nadrowski/master_weak.txt (master_weak is a cell).
    bounds = str(config.BOUNDS_PATH / "nadrowski" / "master.txt")
    cell = str(config.CELL_PATH / "nadrowski" / "master_weak.txt")

    assert tool.main(["identifiability", "rotation", "--bounds", bounds, "--device", "cpu",
                      "--posterior", "p", "--n-worst", "2", "--top-n", "5"]) == 0
    assert set(seen["identifiability_rotation"]) == {"name", "note", "fig_sink", "store",
                                                      "n_worst", "top_n"}
    assert seen["identifiability_rotation"]["n_worst"] == 2
    assert seen["identifiability_rotation"]["top_n"] == 5
    # M13: rotation's cfg must NOT carry a loaded ground truth (build_cfg's needs_gt=False for it) --
    # a bounds-only cfg's ground_truth raises, exactly like a config nobody ever pointed at a cell.
    with pytest.raises(ValueError):
        _ = seen_args["identifiability_rotation"][0].ground_truth

    assert tool.main(["identifiability", "laplace", "--bounds", bounds, "--device", "cpu",
                      "--posterior", "p", "--cell", cell, "--t-obs", "2.5",
                      "--n-points", "3", "--m", "5", "--m-noise", "17", "--rel", "0.03",
                      "--min-valid", "0.6", "--sd-identified", "0.4", "--seed", "9"]) == 0
    assert set(seen["identifiability_laplace"]) == {"name", "note", "fig_sink", "store", "n_points",
                                                     "m", "m_noise", "rel", "min_valid",
                                                     "sd_identified", "t_obs_s", "seed"}
    assert seen["identifiability_laplace"]["n_points"] == 3
    assert seen["identifiability_laplace"]["sd_identified"] == 0.4
    assert seen["identifiability_laplace"]["t_obs_s"] == 2.5
    assert seen["identifiability_laplace"]["m"] == 5
    assert seen["identifiability_laplace"]["m_noise"] == 17
    assert seen["identifiability_laplace"]["rel"] == 0.03
    assert seen["identifiability_laplace"]["min_valid"] == 0.6
    assert seen["identifiability_laplace"]["seed"] == 9
    # M13: laplace's cfg DOES need a loaded ground truth (build_cfg's needs_gt=True for it).
    assert seen_args["identifiability_laplace"][0].ground_truth is not None

    assert tool.main(["identifiability", "jacobian", "--bounds", bounds, "--device", "cpu",
                      "--cell", cell, "--t-obs", "3.0", "--m", "6", "--m-noise", "19",
                      "--rel", "0.04", "--min-valid", "0.7", "--zero-tol", "0.1",
                      "--noise-eps", "1e-5", "--seed", "11"]) == 0
    assert set(seen["identifiability_jacobian"]) == {"name", "note", "fig_sink", "store", "m",
                                                      "m_noise", "rel", "min_valid", "zero_tol",
                                                      "noise_eps", "t_obs_s", "seed"}
    assert seen["identifiability_jacobian"]["zero_tol"] == 0.1
    assert seen["identifiability_jacobian"]["noise_eps"] == 1e-5
    assert seen["identifiability_jacobian"]["m"] == 6 and seen["identifiability_jacobian"]["m_noise"] == 19
    assert seen["identifiability_jacobian"]["rel"] == 0.04
    assert seen["identifiability_jacobian"]["min_valid"] == 0.7
    assert seen["identifiability_jacobian"]["seed"] == 11
    assert "n_points" not in seen["identifiability_jacobian"]
    # M13: jacobian's cfg likewise needs a loaded ground truth (build_cfg's needs_gt=True for it too).
    assert seen_args["identifiability_jacobian"][0].ground_truth is not None

    assert tool.main(["identifiability", "rotation", "--bounds", bounds, "--device", "cpu",
                      "--posterior", "p", "--cell", cell]) == 2
    assert tool.main(["identifiability", "--bounds", bounds]) == 2


def test_the_ablation_subcommand_forwards_its_knobs(tmp_path, monkeypatch):
    from core import tool
    from core.artifacts import Accept
    from core.tool import diagnostics as tool_diag
    monkeypatch.setenv("PRISM_ARTIFACTS", str(tmp_path / "A"))
    seen = {}

    def _pair(cfg, ref, accept, store):
        seen["accept"] = accept
        return "POST", "PRIOR"

    def _rec(cfg, posterior, **kw):
        # Named, not `seen.setdefault("kw", kw) or SimpleNamespace(...)`: kw is never empty, so the
        # `or` would return the dict and `report` would fail on `dict.kind`.
        seen["kw"] = kw
        return SimpleNamespace(kind="diagnostic", path=tmp_path / "diagnostics" / "d__1")

    monkeypatch.setattr(tool_diag, "load_posterior_and_prior", _pair)
    monkeypatch.setattr("core.diagnostics.channel_ablation", _rec)
    bounds = str(config.BOUNDS_PATH / "nadrowski" / "master.txt")
    assert tool.main(["ablation", "--bounds", bounds, "--device", "cpu", "--posterior", "p",
                      "--rows", "500", "--n-sweep", "9", "--accept-truncated", "--name", "a1"]) == 0
    assert set(seen["kw"]) == {"name", "note", "fig_sink", "store", "rows", "n_sweep"}
    assert seen["kw"]["rows"] == 500 and seen["kw"]["n_sweep"] == 9 and seen["kw"]["name"] == "a1"
    assert seen["accept"] == Accept(truncated=True)
    seen.clear()
    assert tool.main(["ablation", "--bounds", bounds, "--device", "cpu", "--posterior", "p"]) == 0
    assert "rows" not in seen["kw"] and "n_sweep" not in seen["kw"]
    assert set(seen["kw"]) == {"name", "note", "fig_sink", "store"}


def test_smoke_runs_the_four_stages_and_the_resume_drill_is_loud(tool_env, tmp_path, capsys):
    """The four stages end to end at tiny size, then the drill the GPU gate of record runs (§3.8).

    Leg (b) is the resume: the SAME --num-runs against the SAME store, with the prior LOADED by name
    so prior_fingerprint is pinned. Leg (c) is the 2026-09-11 incident -- one field apart (n_runs) --
    which used to start a new cache and exit 0, and is now a refusal before the Fisher. Leg (d) is the
    stage banner: a stage that raises names itself before the traceback.
    """
    from core import orchestrator
    from core.artifacts import ArtifactStore, default_store
    from core.tool import main

    # tool_env is what installs SBITEST as real input files and points PRISM_ARTIFACTS at a temp
    # root; --store-root then puts this run's own store beside it.
    bounds, cell, _env_root = tool_env
    sandbox = default_store()
    root = tmp_path / "smoke"
    SCFG = [*_cfg(bounds), "--cell", cell]
    common = ["smoke", *SCFG, "--store-root", str(root), "--run-size", "8", "--t-obs", "1.0",
              "--checkpoint"]

    # (a) the full run: four stages, six artifacts.
    assert main([*common, "--num-runs", "2", "--n-cal", "60", "--max-epochs", "2", "--save"]) == 0
    out = capsys.readouterr().out
    for line in ("=== prior ===", "=== posterior ===", "=== validate ===", "=== infer ===",
                 "[smoke] ALL STAGES COMPLETED"):
        assert line in out, line
    s = ArtifactStore(root)
    assert [r.name for r in s.list("prior")] == ["smoke_prior"]
    assert [r.name for r in s.list("posterior")] == ["smoke_posterior"]
    for kind in ("simulation", "observation", "calibration", "inference"):
        assert len(s.list(kind)) == 1, kind

    # (b) the drill: same store, same budget, the prior loaded by name -> a resume, in seconds.
    drill = [*common, "--num-runs", "2", "--prior", "smoke_prior", "--stages", "prior,posterior"]
    assert main([*drill, "--resume", "require"]) == 0
    out = capsys.readouterr().out
    assert "resuming at batch 2/2" in out, out[-2000:]
    assert len(ArtifactStore(root).list("simulation")) == 1, "a resume must not key a new cache"

    # (c) one field away: refused before any simulation, naming the field and both values.
    assert main([*common, "--num-runs", "3", "--prior", "smoke_prior",
                 "--stages", "prior,posterior"]) == 1
    cap = capsys.readouterr()
    assert "n_runs" in cap.err and "new_run" in cap.err, cap.err
    assert len(ArtifactStore(root).list("simulation")) == 1, "nothing may be written by a refusal"

    # (d) a stage that raises names itself.
    def _boom(*a, **k):
        raise RuntimeError("calibration exploded")

    real = orchestrator.validate_calibration
    orchestrator.validate_calibration = _boom
    try:
        code = main([*drill, "--stages", "prior,posterior,validate", "--resume", "require"])
    finally:
        orchestrator.validate_calibration = real
    cap = capsys.readouterr()
    assert code == 1
    assert "*** FAILED in stage validate ***" in cap.out + cap.err

    assert default_store() is sandbox, "main must leave the session's default store installed"
