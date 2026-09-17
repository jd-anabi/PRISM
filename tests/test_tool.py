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
    teardown = None
    with pytest.MonkeyPatch.context() as mp:
        try:
            # INSIDE the try: an install that fails halfway (the model JSON written, a folder not) must
            # still be undone, and so must the monkeypatches -- the context manager restores those.
            bounds, cell, teardown = install_sbitest()
            mp.setenv("PRISM_ARTIFACTS", str(root))
            mp.setattr(orchestrator.pipeline, "gen_prior", _tiny_gen_prior)
            yield str(bounds), str(cell), root
        finally:
            if teardown is not None:
                teardown()
            else:
                # install_sbitest raised before handing back its teardown: remove whatever it wrote
                from core import registry
                from core.Helpers import model_store
                try:
                    model_store.delete_user_model("SBITEST")
                except Exception:                                # noqa: BLE001 -- best-effort cleanup
                    pass
                registry.unregister("SBITEST")


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


def _env_reads_and_knob_writes(tree) -> list:
    """``[(lineno, what)]`` for every environment read and every knob write in a parsed module.

    A knob is an attribute that is upper-case (a module constant) or ``batch_size``. The forms: a plain,
    augmented or annotated assignment, a tuple or list target (walked recursively), and
    ``setattr(x, "<knob>", v)`` with a string constant. An environment read is ``os.environ`` /
    ``os.getenv`` as attributes, or the bare names in a module that did ``from os import ...``."""
    def _knob(attr):
        return attr.isupper() or attr == "batch_size"

    def _targets(t):
        if isinstance(t, (ast.Tuple, ast.List)):
            for elt in t.elts:
                yield from _targets(elt)
        elif isinstance(t, ast.Starred):
            yield from _targets(t.value)
        else:
            yield t

    from_os = {a.asname or a.name for node in ast.walk(tree)
               if isinstance(node, ast.ImportFrom) and node.module == "os"
               for a in node.names if a.name in ("environ", "getenv")}
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in ("environ", "getenv"):
            found.append((node.lineno, "reads the environment"))
        if isinstance(node, ast.Name) and node.id in from_os:
            found.append((node.lineno, f"reads the environment ({node.id})"))
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
            targets = [node.target]
        else:
            targets = []
        for target in (t for tt in targets for t in _targets(tt)):
            if isinstance(target, ast.Attribute) and _knob(target.attr):
                found.append((node.lineno, f"assigns {target.attr}"))
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "setattr"
                and len(node.args) >= 2 and isinstance(node.args[1], ast.Constant)
                and isinstance(node.args[1].value, str) and _knob(node.args[1].value)):
            found.append((node.lineno, f"assigns {node.args[1].value} via setattr"))
    return found


def test_the_tool_reads_no_environment_and_writes_no_knob(tmp_path):
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
        for lineno, what in _env_reads_and_knob_writes(ast.parse(py.read_text(encoding="utf-8"))):
            pytest.fail(f"{py.relative_to(root)}:{lineno} {what}")

    # The checker's teeth: every write and read form it claims to catch, in a module of its own.
    forms = tmp_path / "forms.py"
    forms.write_text(
        "from os import getenv as ge, environ\n"
        "a, (config.TRAINING_NUM_RUNS, b) = 1, (2, 3)\n"
        "[cfg.hw.batch_size] = [4]\n"
        "config.SBC_N_CAL += 1\n"
        "config.CHI_MODE: bool = True\n"
        "setattr(config, 'TRAINING_RUN_SIZE', 8)\n"
        "x = ge('PRISM_X')\n"
        "y = environ['PRISM_Y']\n", encoding="utf-8")
    got = {ln for ln, _ in _env_reads_and_knob_writes(ast.parse(forms.read_text(encoding="utf-8")))}
    assert got >= {2, 3, 4, 5, 6, 7, 8}, sorted(got)
    assert not _env_reads_and_knob_writes(ast.parse("cfg.name = 'x'\nsetattr(cfg, 'name', 1)\n")), \
        "a lower-case attribute is not a knob"

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
    # V3 on the tool: ONE line, the message and the flag that answers it. No class name and no
    # [raised at ...] -- those were hedges for a bug disguised as a ValueError, which a dedicated
    # Refusal class no longer needs. fix_sentence owns the parentheses, so "((" would mean the
    # message carried its own copy of the flag.
    lines = [ln for ln in err.splitlines() if ln.startswith("prism prior: refused:")]
    assert len(lines) == 1, err
    assert "already exists" in lines[0] and lines[0].endswith("(--name)"), lines[0]
    assert "((" not in lines[0] and "raised at" not in lines[0] and "StoreError" not in lines[0], lines[0]
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
    assert "raised at" not in err, err          # a Refusal: the ladder prints no location for it
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
    # The flag comes from core/tool/fields.py's FLAG table, appended by main's ladder -- the store's
    # message itself no longer names it (test_artifact_store.py pins that it names no control).
    lines = [ln for ln in err.splitlines() if ln.startswith("prism validate: refused:")]
    assert len(lines) == 1, err
    assert "NOT AMORTIZED" in lines[0] and lines[0].endswith("(--accept-truncated)"), lines[0]
    assert "raised at" not in lines[0] and "Posterior tab" not in lines[0], lines[0]

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


def test_identifiability_rotation_refuses_a_posterior_without_a_rotation(tool_run, capsys):
    """Spec §8.4, through the REAL posterior load: no stub stands between the tool and the store.

    tool_run's tpost carries a rotation (the tool has no flag that turns it off), so the posterior
    under test is trained here, through the real build_posterior at tiny size with reparam_rotate off,
    into the tool's own store. It records no V, so there is no basis to decompose: a refusal (exit 1)
    naming the reason and the flag to change, with no `[raised at ...]` -- a Refusal is not a bug in
    disguise -- and no diagnostic written."""
    from matplotlib import pyplot as plt
    from core import cli, config as _config, orchestrator, registry
    from core.artifacts import ArtifactStore
    bounds, _cell, root = tool_run
    store = ArtifactStore(root)
    cfg = cli.make_sim_config("SBITEST", registry.get("SBITEST").labels, registry.state_dep_drift("SBITEST"),
                              bounds, hw=_config.cpu_device(), reparam_rotate=False)
    prior = orchestrator.build_prior(cfg, "tp", False, fig_sink=lambda title, fig: plt.close(fig),
                                     store=store)
    orchestrator.build_posterior(cfg, prior, None, True, name="tpost_norot", store=store,
                                 fig_sink=lambda title, fig: plt.close(fig), num_runs=2,
                                 run_size_cap=8, hidden_features=8, num_transforms=1,
                                 stop_after_epochs=1, checkpoint_every=0)
    assert store.get("posterior", "tpost_norot").body["transform"]["V"] is None
    before = [r.id for r in store.list("diagnostic")]
    capsys.readouterr()
    assert main(["identifiability", "rotation", *_cfg(bounds), "--posterior", "tpost_norot"]) == 1
    err = capsys.readouterr().err
    assert "no Fisher rotation" in err and "(--posterior)" in err and "raised at" not in err, err
    assert [r.id for r in ArtifactStore(root).list("diagnostic")] == before


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
    posts_before = len(ArtifactStore(root).list("posterior"))
    assert main([*common, "--num-runs", "3", "--prior", "smoke_prior",
                 "--stages", "prior,posterior"]) == 1
    cap = capsys.readouterr()
    assert "n_runs" in cap.err and "new_run" in cap.err, cap.err
    # The flag comes from core/tool/fields.py's table, appended by the ladder; the message itself
    # names only the keyword, and a Refusal prints with no class name and no [raised at ...].
    assert "--new-run" in cap.err and "raised at" not in cap.err, cap.err
    # K8, fix round 1: the comment always said "both values" -- pin them, read literally off the
    # message's own format (orchestrator._near_miss_lines: "this run <mine>, that cache <theirs>").
    # This run asked for --num-runs 3; leg (a) committed the cache at --num-runs 2.
    assert "this run 3" in cap.err and "that cache 2" in cap.err, cap.err
    assert len(ArtifactStore(root).list("simulation")) == 1, "nothing may be written by a refusal"
    assert len(ArtifactStore(root).list("posterior")) == posts_before, \
        "nothing may be written by a refusal"

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


def test_every_smoke_flag_reaches_its_stage_as_a_keyword(tool_env, tmp_path, monkeypatch):
    """K1, fix round 1 (review finding, spec Sec. 8.4): no test pinned smoke's own knob forwarding.
    Deleting ``run_size_cap=args.run_size_cap`` from smoke.py kept the four-stage test (and leg (b)
    of its drill) green, because both legs of THAT test fall back to the same hardware batch -- the
    GPU gate would then silently train at the card's batch and key a different simulation identity.
    This pins the exact ``set(kw)`` each of the four stage calls receives, one distinct non-default
    value per flag, the way T11/T16's own knob tests pin TRAIN_KNOBS/TSNPE_KNOBS/sbc's set.

    All four orchestrator calls smoke.py makes (build_prior, build_posterior, validate_calibration,
    simulated_inference) CAN be stubbed with recorders without re-implementing smoke -- none of
    smoke's own control flow (stage skipping, the width/finite checks, the banner) depends on a real
    return value beyond the shapes built here.
    """
    import torch
    from core import orchestrator
    from core.SBI.statistics import SUMMARY_WIDTH
    from core.tool import main

    bounds, cell, root = tool_env
    loaded_prior = _art(root, "prior")
    loaded_post = _art(root, "posterior")
    prior_rec = _Rec(loaded_prior)
    post_rec = _Rec(loaded_post)
    val_rec = _Rec(_art(root, "calibration"))
    infer_calls = []

    def _infer_rec(cfg, posterior, t_obs_s, **kw):
        infer_calls.append(((cfg, posterior, t_obs_s), kw))
        want = SUMMARY_WIDTH + 1 + orchestrator.expected_forcing_dim(cfg)
        obs = SimpleNamespace(width=want, x_obs=torch.zeros(1))
        return obs, _art(root, "inference")

    monkeypatch.setattr(orchestrator, "build_prior", prior_rec)
    monkeypatch.setattr(orchestrator, "build_posterior", post_rec)
    monkeypatch.setattr(orchestrator, "validate_calibration", val_rec)
    monkeypatch.setattr(orchestrator, "simulated_inference", _infer_rec)
    # --seed reaches the one seeded() context the whole stage sequence runs in
    import contextlib
    import core.diagnostics.rng
    seeds = []
    monkeypatch.setattr(core.diagnostics.rng, "seeded",
                        lambda seed, device: seeds.append((seed, device)) or contextlib.nullcontext())

    # --prior AND --save together: build_prior LOADS (ref given, build_new False) and is named ""
    # (a loaded prior is never renamed); build_posterior still gets "smoke_posterior" -- --save names
    # whatever THIS run builds, independent of whether the prior was loaded. --store-root keeps the
    # run out of %TEMP%: a successful run never removes the root main created for it.
    assert main(["smoke", *_cfg(bounds), "--cell", cell, "--num-runs", "3", "--run-size", "9",
                 "--n-cal", "17", "--max-epochs", "6", "--checkpoint", "--new-run",
                 "--resume", "require", "--t-obs", "2.5", "--seed", "5", "--prior", "smoke_prior",
                 "--save", "--store-root", str(tmp_path / "smoke")]) == 0

    (cfg1, ref1, build_new1), kw1 = prior_rec.calls[0]
    assert seeds == [(5, cfg1.hw.device)], seeds
    assert ref1 == "smoke_prior" and build_new1 is False
    assert kw1["name"] == ""
    assert set(kw1) == {"fig_sink", "store", "name"}

    (cfg2, prior2, ref2, train_new2), kw2 = post_rec.calls[0]
    assert prior2 is loaded_prior and ref2 is None and train_new2 is True
    assert kw2["name"] == "smoke_posterior"
    assert set(kw2) == {"fig_sink", "store", "name", "num_runs", "run_size_cap", "max_num_epochs",
                        "checkpoint_every", "new_run", "resume"}
    assert kw2["num_runs"] == 3
    assert kw2["run_size_cap"] == 9
    assert kw2["max_num_epochs"] == 6
    assert kw2["checkpoint_every"] == 1, "ck_every = max(1, num_runs // 2) = max(1, 3 // 2)"
    assert kw2["new_run"] is True
    assert kw2["resume"] == "require"

    (cfg3, post3, prior3), kw3 = val_rec.calls[0]
    assert post3 is loaded_post and prior3 is loaded_prior
    assert set(kw3) == {"fig_sink", "store", "n_cal"}
    assert kw3["n_cal"] == 17

    (cfg4, post4, t_obs4), kw4 = infer_calls[0]
    assert post4 is loaded_post and t_obs4 == 2.5
    assert set(kw4) == {"cell", "prior", "fig_sink", "store"}
    assert kw4["cell"] == cell and kw4["prior"] is loaded_prior


def test_smoke_rejects_an_unknown_stage_at_parse_time(tool_env, monkeypatch, capsys):
    """K2, fix round 1: an unknown --stages entry is now an argparse type= error, so it is refused
    DURING PARSING -- before main ever calls tempfile.mkdtemp or registry.load_user_models(). Pinned
    by call counts on both, not by a directory listing or the exit code alone: the OLD runtime-only
    check (still present in run_smoke as a second, defensive line -- see its own docstring) already
    returned exit 2 for this exact input, AND K3's own leaked-root cleanup would already remove the
    resulting empty prism_smoke_* directory after that runtime refusal -- so neither the exit code
    nor a clean tempdir listing can tell "refused before a root was resolved" apart from "refused
    after one was created and then cleaned up". Only the call counts can.
    """
    import tempfile
    from core import registry
    from core.tool import main

    bounds, cell, root = tool_env
    mkdtemp_calls = []
    real_mkdtemp = tempfile.mkdtemp
    monkeypatch.setattr(
        tempfile, "mkdtemp",
        lambda *a, **k: (mkdtemp_calls.append(1), real_mkdtemp(*a, **k))[1])
    load_calls = []
    monkeypatch.setattr(registry, "load_user_models", lambda: load_calls.append(1))

    capsys.readouterr()
    assert main(["smoke", *_cfg(bounds), "--cell", cell, "--stages", "prior,bogus"]) == 2
    err = capsys.readouterr().err
    assert "bogus" in err and "--stages" in err
    assert mkdtemp_calls == [], "a parse-time refusal must never call tempfile.mkdtemp"
    assert load_calls == [], "a parse-time refusal must never reach registry.load_user_models"


def test_smoke_leaves_no_leaked_temp_root_on_a_bad_bounds_file(tool_env, tmp_path, monkeypatch,
                                                                capsys):
    """K3, fix round 1: mkdtemp runs before build_cfg, so a bad --bounds used to leave an empty
    %TEMP%\\prism_smoke_* behind forever, its path never printed anywhere an operator would look.
    main() now removes an auto-created root when the handler fails AND the root is still empty; a
    non-empty root (Step 7's real one-stage check, or any run that got as far as writing anything)
    or a user-named --store-root is never touched -- pinned by the main four-stage smoke test's own
    --store-root runs, which still leave their directories on disk.

    tempfile.tempdir is monkeypatched to tmp_path (rather than reading %TEMP% itself) so this test
    counts in isolation from whatever else the real temp directory holds.
    """
    import tempfile
    from core.tool import main

    bounds, cell, root = tool_env
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    before = set(tmp_path.iterdir())
    bad_bounds = str(Path(bounds).parent / "does_not_exist.txt")
    capsys.readouterr()
    assert main(["smoke", "--bounds", bad_bounds, "--device", "cpu", "--cell", cell]) == 1
    after = set(tmp_path.iterdir())
    assert after == before, "a bad --bounds must not leave an empty prism_smoke_* directory behind"
    # §3.3's file rule at the build: ONE refusal line naming the input kind and the flag, not the
    # parser's bare FileNotFoundError with a [raised at file_manager.py:...] hedge
    err = capsys.readouterr().err
    lines = [ln for ln in err.splitlines() if ln.startswith("prism smoke: refused:")]
    assert len(lines) == 1, err
    assert "The bounds file was not found" in lines[0] and lines[0].endswith("(--bounds)"), lines[0]
    assert "raised at" not in lines[0] and "FileNotFoundError" not in lines[0], lines[0]


def test_a_missing_cell_is_refused_at_the_read_naming_the_flag(tool_env, capsys):
    """The cell half of §3.3's file rule, on a subcommand whose config build reads the truth
    (identifiability jacobian, through cli.load_and_validate_gt): a --cell that names no file is a
    Refusal(field="cell") printed as one line ending in the flag, before anything is spent."""
    bounds, cell, _root = tool_env
    missing = str(Path(cell).parent / "no_such_cell.txt")
    capsys.readouterr()
    assert main(["identifiability", "jacobian", "--bounds", bounds, "--device", "cpu",
                 "--cell", missing, "--t-obs", "1"]) == 1
    err = capsys.readouterr().err
    lines = [ln for ln in err.splitlines() if ln.startswith("prism identifiability: refused:")]
    assert len(lines) == 1, err
    assert "The cell file was not found" in lines[0] and lines[0].endswith("(--cell)"), lines[0]
    assert "raised at" not in lines[0] and "FileNotFoundError" not in lines[0], lines[0]


def test_smoke_ctrl_c_advice_depends_on_store_root():
    """K5, fix round 1: smoke keys its store on --store-root, not PRISM_ARTIFACTS, so the generic
    "the same command with --resume require continues them" advice is wrong for it -- re-issuing run
    1's own command line just re-BUILDS the prior. Unit-tested directly against the extracted helper
    (rather than only by driving a real KeyboardInterrupt through a real smoke run, as the existing
    Ctrl-C test does for train against a prior/posterior already on disk) because smoke's own
    prior/posterior build is the expensive real chain this subcommand exists to exercise.
    """
    from core.tool import _smoke_interrupt_advice

    auto = Path(r"C:\Temp\prism_smoke_abc")
    named = SimpleNamespace(store_root=r"C:\scratch\smoke", prior=None, save=True)
    advice = _smoke_interrupt_advice(named, Path(named.store_root))
    assert r"--store-root C:\scratch\smoke" in advice
    assert "--prior smoke_prior" in advice and "--stages prior,posterior" in advice
    assert "--resume require" in advice and "--checkpoint" in advice
    assert "--num-runs" in advice and "--run-size" in advice

    # without --save the prior this run built is unnamed: the advice must not name smoke_prior
    unsaved = SimpleNamespace(store_root=r"C:\scratch\smoke", prior=None, save=False)
    advice = _smoke_interrupt_advice(unsaved, Path(unsaved.store_root))
    assert "smoke_prior" not in advice and "cannot be resumed" not in advice, advice
    assert "_unnamed__" in advice and "--resume require" in advice

    # a loaded prior is the one to name, whether or not --save was given
    loaded = SimpleNamespace(store_root=r"C:\scratch\smoke", prior="P", save=True)
    assert "--prior P " in _smoke_interrupt_advice(loaded, Path(loaded.store_root))

    # the temp root is kept when it holds committed batches, and its path is the one to name
    unnamed = SimpleNamespace(store_root=None, prior=None, save=True)
    advice = _smoke_interrupt_advice(unnamed, auto)
    assert f"--store-root {auto}" in advice and "cannot be resumed" not in advice, advice


def test_smoke_ctrl_c_prints_store_specific_resume_advice(tool_env, tmp_path, monkeypatch, capsys):
    """K5 end to end: main()'s KeyboardInterrupt handler must pick the smoke-specific advice for
    smoke -- keyed on the --store-root FLAG (hasattr), not the subcommand name -- and must leave
    every other subcommand's generic advice untouched (test_ctrl_c_mid_simulation_keeps_the_committed_
    batches still asserts only "interrupted" is in stderr for train). Raising KeyboardInterrupt
    straight out of build_prior, rather than driving a real interrupt through a real simulation mid-
    flight, proves main() picked the right branch at no simulation cost.
    """
    from core import orchestrator
    from core.tool import main

    bounds, cell, _root = tool_env

    def _boom(*a, **k):
        raise KeyboardInterrupt

    monkeypatch.setattr(orchestrator, "build_prior", _boom)

    store = tmp_path / "named_store"
    capsys.readouterr()
    assert main(["smoke", *_cfg(bounds), "--cell", cell, "--store-root", str(store), "--save"]) == 130
    err = capsys.readouterr().err
    assert f"--store-root {store}" in err and "--prior smoke_prior" in err
    assert "--stages prior,posterior" in err and "--resume require" in err

    # no --store-root: the temp root's own path is named (it is kept when it holds batches). The
    # root is empty here, so main removes it after printing -- nothing is left in %TEMP%.
    capsys.readouterr()
    assert main(["smoke", *_cfg(bounds), "--cell", cell]) == 130
    err = capsys.readouterr().err
    assert "cannot be resumed" not in err and "prism_smoke_" in err, err
    assert "smoke_prior" not in err, "a run without --save built an unnamed prior"


def test_smoke_refuses_a_bad_cell_and_an_impossible_resume_before_the_prior(tool_env, tmp_path,
                                                                            monkeypatch, capsys):
    """F5/F6: smoke checks what it can from its flags before the prior build.

    The cell used to be parsed first inside the infer stage, after the prior (~100 s), the training
    and the calibration; a mistyped one then exited 1 and left orphans. And a --resume that can never
    work (no --checkpoint, or require without --prior, whose fresh fit has a new prior_fingerprint)
    was refused only after the prior, by an orchestrator message naming a flag smoke does not have."""
    from core import orchestrator
    from core.tool import main

    bounds, cell, _root = tool_env
    rec = _Rec(None)
    monkeypatch.setattr(orchestrator, "build_prior", rec)
    missing = tmp_path / "missing.txt"

    capsys.readouterr()
    assert main(["smoke", *_cfg(bounds), "--cell", str(missing), "--store-root", str(tmp_path / "s")]) == 1
    err = capsys.readouterr().err
    assert "missing.txt" in err, err
    assert rec.calls == [], "the prior was built before the cell was checked"

    assert main(["smoke", *_cfg(bounds), "--cell", cell, "--store-root", str(tmp_path / "s"),
                 "--resume", "require", "--prior", "p"]) == 2
    err = capsys.readouterr().err
    assert "--checkpoint" in err, err
    assert main(["smoke", *_cfg(bounds), "--cell", cell, "--store-root", str(tmp_path / "s"),
                 "--resume", "never"]) == 2
    assert "--checkpoint" in capsys.readouterr().err
    assert main(["smoke", *_cfg(bounds), "--cell", cell, "--store-root", str(tmp_path / "s"),
                 "--resume", "require", "--checkpoint"]) == 2
    err = capsys.readouterr().err
    assert "--prior" in err, err
    assert rec.calls == [], "an impossible --resume was refused only after the prior"


def test_fdt_and_crossval_flags_reach_their_builders(tool_env, monkeypatch, capsys):
    """Every fdt/crossval flag arrives at the builder it belongs to, and nothing simulates.

    The recorders stand in for the four callees; each handler imports them at CALL time, so the
    module attribute is what runs. This is the fast-gate coverage for the pair -- the real tiny-size
    run below may be slow-marked. ``tool_env`` is taken for its PRISM_ARTIFACTS redirect: ``main``
    mkdirs its store root before any handler runs, and the session teardown asserts the real
    ``Artifacts/`` gained nothing.

    Every ``kw`` a recorder receives is pinned by an exact-set (or exact-dict) comparison, never a
    sample of its keys: ``knobs()`` silently drops a dest that no longer matches a flag, so checking
    only a few keys would stay green through a knob quietly no longer reaching its builder.
    """
    from core import cli, config
    from core.FDT import cross_validation, fdt_pipeline
    from core.tool import main

    seen = {}

    def _fdt_cfg(model, state_dep_drift, cell_file, **kw):
        seen["make_fdt_config"] = (model, state_dep_drift, cell_file, kw)
        return "CFG"

    def _run_fdt(cfg, *, skip_sanity, confirm_production):
        seen["run_fdt"] = (cfg, skip_sanity, confirm_production)

    def _sweep_cfg(cell_file, **kw):
        seen["make_param_sweep_config"] = (cell_file, kw)
        return "CFG", "S", "T"

    def _study(cfg, s_grid, temp_grid):
        seen["run_param_study_cli"] = (cfg, s_grid, temp_grid)
        return Path("s.h5"), Path("t.h5")

    monkeypatch.setattr(cli, "make_fdt_config", _fdt_cfg)
    monkeypatch.setattr(fdt_pipeline, "run_fdt", _run_fdt)
    monkeypatch.setattr(cli, "make_param_sweep_config", _sweep_cfg)
    monkeypatch.setattr(cross_validation, "run_param_study_cli", _study)

    # Inputs resolve through config, never through the process working directory.
    cell = str(config.CELL_PATH / "hopf" / "cell.txt")
    nad = str(config.CELL_PATH / "nadrowski" / "master_spont.txt")
    assert main(["fdt", "--cell", cell, "--n-freqs", "3", "--ensemble-m", "7",
                 "--freqs-per-batch", "2", "--f0", "0.11", "--skip-sanity"]) == 0
    model, sdd, cell_file, kw = seen["make_fdt_config"]
    assert model == "HOPF" and sdd is False and cell_file == cell     # the model is the cell's folder
    assert kw == {"n_freqs": 3, "ensemble_M": 7, "freqs_per_batch": 2, "F0": 0.11}
    assert seen["run_fdt"] == ("CFG", True, True)                     # --no-production absent

    # M4, fix round 1: NADROWSKI's state-dependent (multiplicative) drift, next to HOPF's False above.
    seen.clear()
    assert main(["fdt", "--cell", nad]) == 0
    assert seen["make_fdt_config"][1] is True, "NADROWSKI has state-dependent drift"

    seen.clear()
    assert main(["fdt", "--cell", cell, "--model", "hopf", "--no-production"]) == 0
    assert seen["make_fdt_config"][3] == {}, "an unset knob must not be passed: the default is in cli"
    assert seen["run_fdt"] == ("CFG", False, False)

    # I1, fix round 1: NEITHER flag. The two cases above alone cannot catch confirm_production
    # cross-wired to skip_sanity: (skip_sanity=True, no_production=False) and (skip_sanity=False,
    # no_production=True) both coincidentally survive that bug (True/True and False/False again).
    # Only the both-False combination tells them apart -- it must come out (False, True).
    seen.clear()
    assert main(["fdt", "--cell", cell]) == 0
    assert seen["run_fdt"] == ("CFG", False, True), \
        "neither flag: sanity runs, then production proceeds by default"

    seen.clear()
    assert main(["crossval", "--cell", nad, "--s-grid", "0", "0.1", "2",
                 "--t-grid", "1", "1.1", "3", "--n-freqs", "2", "--ensemble-m", "8",
                 "--freqs-per-batch", "4", "--f0", "0.2"]) == 0
    cell_file, kw = seen["make_param_sweep_config"]
    assert cell_file == nad
    assert set(kw) == {"preset", "s_spec", "t_spec", "n_freqs", "ensemble_M", "freqs_per_batch", "F0"}
    assert kw["s_spec"] == (0.0, 0.1, 2) and kw["t_spec"] == (1.0, 1.1, 3)
    assert isinstance(kw["s_spec"][2], int), "np.linspace refuses a float num"
    assert kw["n_freqs"] == 2 and kw["ensemble_M"] == 8
    assert kw["freqs_per_batch"] == 4 and kw["F0"] == 0.2
    assert kw["preset"] == dict(cli.SWEEP_PRESETS["exploratory"])
    assert seen["run_param_study_cli"] == ("CFG", "S", "T")
    assert "s.h5" in capsys.readouterr().out

    seen.clear()
    assert main(["crossval", "--cell", nad, "--preset", "production",
                 "--s-grid", "0", "0.1", "2", "--t-grid", "1", "1.1", "2"]) == 0
    cell_file, kw = seen["make_param_sweep_config"]
    assert set(kw) == {"preset", "s_spec", "t_spec", "n_freqs", "ensemble_M"}, \
        "freqs_per_batch and F0 were left unset -- they must not be forwarded"
    assert kw["preset"] == dict(cli.SWEEP_PRESETS["production"])
    assert kw["n_freqs"] == cli.SWEEP_PRESETS["production"]["n_freqs"]
    assert kw["ensemble_M"] == cli.SWEEP_PRESETS["production"]["ensemble_M"]


def test_fdt_and_crossval_usage_errors(tool_env, capsys, monkeypatch):
    """A model FDT cannot run is a refusal (1); a malformed grid or a missing cell is usage (2).

    ``tool_env`` for the PRISM_ARTIFACTS redirect: even a refusal builds the store root first.
    """
    from core import config
    from core.tool import main

    cell = str(config.CELL_PATH / "hopf" / "cell.txt")
    assert main(["fdt", "--cell", cell, "--model", "NOPE"]) == 1
    err = capsys.readouterr().err
    assert "Unknown model" in err
    assert "pass a different --model" in err, "M1: the refusal says where the name came from"

    nad = str(config.CELL_PATH / "nadrowski" / "master_spont.txt")
    assert main(["crossval", "--cell", nad, "--s-grid", "0", "0.1", "2.5",
                 "--t-grid", "1", "1.1", "2"]) == 2
    assert "--s-grid" in capsys.readouterr().err
    assert main(["crossval", "--cell", nad, "--s-grid", "0", "0.1", "1",
                 "--t-grid", "1", "1.1", "2"]) == 2

    # M3, fix round 1: a bad --t-grid (valid --s-grid alongside it) must name --t-grid, not --s-grid.
    capsys.readouterr()
    assert main(["crossval", "--cell", nad, "--s-grid", "0", "0.1", "2",
                 "--t-grid", "1", "1.1", "1"]) == 2
    assert "--t-grid" in capsys.readouterr().err

    # M2, fix round 1: inf/nan point counts used to raise OverflowError/ValueError past main's usage-
    # error net (a traceback, "*** FAILED ***", exit 1) instead of naming the flag at exit 2.
    capsys.readouterr()
    assert main(["crossval", "--cell", nad, "--s-grid", "0", "0.1", "inf",
                 "--t-grid", "1", "1.1", "2"]) == 2
    err = capsys.readouterr().err
    assert "--s-grid" in err and "finite" in err

    capsys.readouterr()
    assert main(["crossval", "--cell", nad, "--s-grid", "0", "0.1", "nan",
                 "--t-grid", "1", "1.1", "2"]) == 2
    err = capsys.readouterr().err
    assert "--s-grid" in err and "finite" in err

    assert main(["fdt"]) == 2                       # --cell is required
    assert main(["crossval", "--cell", nad]) == 2   # both grids are required

    # F8: --skip-sanity with --no-production runs nothing, and used to run the full production sweep
    # silently. A usage error, before any config is built (a recorder stands in for the builder).
    from core import cli
    built = []
    monkeypatch.setattr(cli, "make_fdt_config", lambda *a, **k: built.append(1))
    capsys.readouterr()
    assert main(["fdt", "--cell", cell, "--skip-sanity", "--no-production"]) == 2
    err = capsys.readouterr().err
    assert "--skip-sanity" in err and "--no-production" in err, err
    assert built == [], "the refused pair built a config"


def test_crossval_preset_choices_match_sweep_presets():
    """M5, fix round 1: ``--preset``'s hard-coded choices stay hard-coded -- importing ``core.cli``
    while building the parser would cost a torch import on plain ``--help`` -- so this pins the two
    lists in sync instead of trusting them to agree by eye."""
    from core import cli
    from core.tool import build_parser

    action = next(a for a in build_parser().subcommands["crossval"]._actions if a.dest == "preset")
    assert tuple(action.choices) == tuple(cli.SWEEP_PRESETS)


def test_fdt_ctrl_c_gets_its_own_interrupt_note(tool_env, monkeypatch, capsys):
    """I2, fix round 1: fdt/crossval keep no cache and take no --resume, so main's generic advice
    ("if a [checkpoint] line above says batches were saved, ... --resume require") is simply wrong
    for them -- there is no cache to resume. The partial plots already on disk are the whole
    recovery story; re-running starts over. Every OTHER subcommand's Ctrl-C message is untouched
    (test_smoke_ctrl_c_prints_store_specific_resume_advice and
    test_ctrl_c_mid_simulation_keeps_the_committed_batches still pin the generic wording)."""
    from core import cli, config
    from core.FDT import fdt_pipeline
    from core.tool import main

    def _boom(cfg, *, skip_sanity, confirm_production):
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "make_fdt_config", lambda *a, **k: "CFG")
    monkeypatch.setattr(fdt_pipeline, "run_fdt", _boom)
    cell = str(config.CELL_PATH / "hopf" / "cell.txt")
    capsys.readouterr()
    assert main(["fdt", "--cell", cell]) == 130
    err = capsys.readouterr().err
    assert "interrupted" in err
    assert "<artifacts root>/fdt" in err and "stay on disk" in err and "from scratch" in err
    assert "--resume" not in err


@pytest.mark.slow
def test_fdt_and_crossval_run_at_tiny_size(tool_env, capsys):
    """The real pipelines, at the smallest sizes the flags allow, into the temp artifacts root.

    No new science: this asks only whether the two subcommands drive the campaigns end to end and
    put their outputs where the GUI panels put theirs. ``tool_env`` is what makes "the temp artifacts
    root" true: it sets PRISM_ARTIFACTS, and `config.artifacts_root()` -- which is what FDT and the
    sweep study write under -- reads that variable at every call. Without it the four PNGs and the
    two .h5 files land in the real ``Artifacts/`` and the session teardown fails.

    Measured 2026-09-15: 598.55 s on the CPU. fdt's own Campaign 1 (810k Euler steps) is one term,
    but crossval's four operating points (2 S-sweep + 2 T-sweep, ~410k steps each under the
    exploratory preset) are likely the bigger share of the total -- neither psd_T_obs_nd nor the
    sweep step count is a flag -- so it is slow-marked; the recorder test above keeps the fast-gate
    coverage. (Fix round 1, M7: corrected from "Campaign 1 alone", which undercounted crossval's
    share; not re-measured, since the PNG assertions added below are cheap globs.)
    """
    from core import config
    from core.tool import main

    cell = str(config.CELL_PATH / "hopf" / "cell.txt")
    assert main(["fdt", "--cell", cell, "--n-freqs", "2", "--ensemble-m", "8",
                 "--skip-sanity"]) == 0
    out = config.artifacts_root() / "fdt"
    for tag in ("fdt_ratio_", "chi_components_", "psd_", "spontaneous_trajectory_"):
        assert list(out.glob(f"{tag}*.png")), tag

    nad = str(config.CELL_PATH / "nadrowski" / "master_spont.txt")
    assert main(["crossval", "--cell", nad, "--s-grid", "0", "0.1", "2",
                 "--t-grid", "1", "1.1", "2", "--n-freqs", "2", "--ensemble-m", "8"]) == 0
    cv = config.artifacts_root() / "crossval"
    assert list(cv.glob("sweep_s_*.h5")) and list(cv.glob("sweep_temp_*.h5"))
    for tag in ("fdt3d_vs_S_", "fdt3d_vs_T_"):
        assert list(cv.glob(f"{tag}*.png")), tag
    assert "[prism crossval] S sweep:" in capsys.readouterr().out


def test_fdt_plot_functions_close_a_saved_figure_instead_of_show(tmp_path):
    """Commit B, fix round 1: every real caller (fdt_pipeline.py, sanity.py) always passes
    ``save_path``, so the old unconditional ``plt.show()`` was pure cost under the tool's Agg
    backend -- it does nothing there except print "FigureCanvasAgg is non-interactive, and thus
    cannot be shown" (four times per real ``fdt`` run, once per plot function) and it never closed
    the figure it drew, leaking one live figure per call for the life of the process. Close-when-
    saved is the right default; ``plt.show()`` survives for the no-``save_path`` case no current
    caller uses.
    """
    import warnings

    import numpy as np
    from matplotlib import pyplot as plt

    from core.FDT.plots import plot_psd

    before = len(plt.get_fignums())
    out = tmp_path / "psd.png"
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        plot_psd(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 1.5]), save_path=out)
    assert out.exists()
    assert not any("non-interactive" in str(w.message) for w in rec), \
        [str(w.message) for w in rec]
    assert len(plt.get_fignums()) == before


def test_the_tool_prints_a_refusal_with_its_flag_and_a_bug_with_a_traceback(tool_env, monkeypatch, capsys):
    """Spec section 1.2, "V3 and the tool's ladder": four rungs, told apart by TYPE.

    A Refusal is one operator line -- ``prism <cmd>: refused: <message> <fix>`` -- with the flag from
    core/tool/fields.py in parentheses when its field has one, and NOTHING after the message when
    the field is None or has no flag (fix_sentence owns the parentheses, so a bare message never
    ends in "()"). No class name, no [raised at ...]. A bare ValueError from a site piece 3 has not
    converted keeps today's hedged shape, class and location included, so an unconverted refusal
    still reads as a refusal. Anything else is a bug and prints the whole traceback. UsageError is
    now a Refusal too and MUST stay exit 2 with its own prefix: it is caught one rung earlier.

    The stage is stubbed to raise, so the ladder is exercised on the real path from main through
    the handler and build_cfg, not on a synthetic try/except."""
    from core import orchestrator
    from core.refusals import Refusal
    from core.tool.config_args import UsageError
    bounds, cell, root = tool_env
    argv = ["prior", *_cfg(bounds)]

    def _raising(exc):
        def _stage(*a, **k):
            raise exc
        return _stage

    def _prism_lines():
        err = capsys.readouterr().err
        return err, [ln for ln in err.splitlines() if ln.startswith("prism prior: ")]

    # (a) a field with a flag: the message, one space, the flag in parentheses, and nothing else
    monkeypatch.setattr(orchestrator, "build_prior",
                        _raising(Refusal("The observation length, in seconds, is blank.", field="t_obs")))
    capsys.readouterr()
    assert main(argv) == 1
    err, lines = _prism_lines()
    assert lines == ["prism prior: refused: The observation length, in seconds, is blank. (--t-obs)"], err
    assert "Traceback" not in err and "raised at" not in err and "Refusal" not in err, err

    # (b) field=None: the line ends at the message -- no parentheses at all
    monkeypatch.setattr(orchestrator, "build_prior", _raising(Refusal("Nothing to resume.")))
    assert main(argv) == 1
    err, lines = _prism_lines()
    assert lines == ["prism prior: refused: Nothing to resume."], err
    assert "(" not in lines[0] and ")" not in lines[0]

    # (c) a registered field with no flag (the chi band is fixed by measurement, D11): same as (b)
    monkeypatch.setattr(orchestrator, "build_prior", _raising(Refusal(
        "The chi frequency band (fixed by measurement) is not the one this posterior was trained under.",
        field="chi_freq_bounds")))
    assert main(argv) == 1
    err, lines = _prism_lines()
    assert lines == ["prism prior: refused: The chi frequency band (fixed by measurement) is not the one "
                     "this posterior was trained under."], err

    # (d) an unconverted ValueError keeps today's hedged shape: class name and innermost frame
    monkeypatch.setattr(orchestrator, "build_prior", _raising(ValueError("old style")))
    assert main(argv) == 1
    err, lines = _prism_lines()
    assert len(lines) == 1 and lines[0].startswith("prism prior: refused: ValueError: old style [raised at "), err
    assert lines[0].endswith("]") and "test_tool.py:" in lines[0] and "Traceback" not in err, err

    # (e) a bug: the traceback, then the FAILED line; never the word "refused"
    monkeypatch.setattr(orchestrator, "build_prior", _raising(RuntimeError("boom")))
    assert main(argv) == 1
    err, lines = _prism_lines()
    assert lines == ["prism prior: *** FAILED ***"], err
    assert "Traceback (most recent call last)" in err and "RuntimeError: boom" in err, err
    assert "refused" not in err, err

    # (f) UsageError is a Refusal and still exits 2 with the usage prefix: caught one rung earlier
    monkeypatch.setattr(orchestrator, "build_prior", _raising(UsageError("--x needs --y")))
    assert main(argv) == 2
    err, lines = _prism_lines()
    assert lines == ["prism prior: usage: --x needs --y"], err


def test_device_cuda_is_refused_when_unavailable(tool_env, monkeypatch, capsys):
    """`--device cuda` asks for the card explicitly, where `auto` would fall back to the CPU with a
    warning from the prior sweep (prior.py). The answer is detect_device()'s own -- CUDA absent, or a
    card below compute capability 8.0, both come back as a cpu DeviceConfig -- and a request it cannot
    honour is a Refusal at config build, before any stage runs: exit 1, `refused:`, the flag appended
    by the tool's own table, no traceback, and nothing written. Patched rather than skipped on a CUDA
    machine, so the refusal is exercised on every gate and not only on a laptop.

    And `cuda` resolves to detect_device()'s OWN DeviceConfig, never a hand-built one: its batch size
    and dtype enter the simulation identity, so `cuda` and `auto` on a qualifying card share one
    cache. make_sim_config is stubbed for that leg so no tensor is ever placed on a device the
    machine may not have."""
    import torch
    from core import cli, config
    from core.tool import config_args
    bounds, _cell, root = tool_env
    monkeypatch.setattr(config, "detect_device", config.cpu_device)
    manifests_before = sorted(Path(root).rglob("manifest.json"))
    capsys.readouterr()
    assert main(["prior", "--bounds", bounds, "--device", "cuda"]) == 1
    out, err = capsys.readouterr()
    assert "prism prior: refused:" in err and "(--device)" in err, err
    assert "'cuda' is not available" in err and "detected cpu" in err, err
    assert "Traceback" not in err and "raised at" not in err, err
    assert "[cfg]" not in out, "the refusal fires in make_cfg, before the banner"
    assert sorted(Path(root).rglob("manifest.json")) == manifests_before

    fake = config.DeviceConfig(device=torch.device("cuda"), dtype=torch.float32, batch_size=7)
    monkeypatch.setattr(config, "detect_device", lambda: fake)
    seen = {}

    def _rec(*a, **k):
        seen.update(k)
        return "CFG"

    monkeypatch.setattr(cli, "make_sim_config", _rec)
    args = build_parser().parse_args(["prior", "--bounds", bounds, "--device", "cuda"])
    assert config_args.make_cfg(args) == "CFG" and seen["hw"] is fake
    args = build_parser().parse_args(["prior", "--bounds", bounds, "--device", "auto"])
    assert config_args.make_cfg(args) == "CFG" and seen["hw"] is None, "auto keeps passing hw=None"


def test_the_tool_routes_info_to_stdout_and_warnings_to_stderr_and_removes_its_handlers(tool_env, monkeypatch, capsys):
    """V4 on the command line. An information record is stdout, plain -- the GPU recipe reads
    ``[checkpoint] resuming at batch k/n`` off stdout and must keep doing so once T17 makes it a record.
    A warning or an error is stderr with its level as a prefix, so an operator's ``2>err.log`` holds
    exactly what needs acting on.

    The handlers are installed for the handler call ONLY and removed in a finally: this suite calls
    ``main`` dozens of times in one process, and a handler left behind would print every later run's
    records twice, then three times. ``main`` runs twice here for that reason, and the ``core``
    logger's handler list is asserted EMPTY after each -- not "unchanged", which a leak from an
    earlier test could satisfy.

    The streams are resolved at EMIT time: capsys swaps sys.stdout/stderr per test, and a handler
    holding the stream it was built with would write into a buffer nobody reads."""
    import logging

    from core import orchestrator
    bounds, cell, root = tool_env
    log = logging.getLogger("core.orchestrator")

    def _prior(cfg, ref, build_new, **kw):
        log.info("[budget] 2 batches x 8 rows")
        log.warning("Adaptive batching: capped")
        log.error("[checkpoint] could not save on the way out")
        return _art(root, "prior")

    monkeypatch.setattr(orchestrator, "build_prior", _prior)
    core_logger = logging.getLogger("core")
    for _ in range(2):
        capsys.readouterr()
        assert main(["prior", *_cfg(bounds)]) == 0
        cap = capsys.readouterr()
        assert cap.out.count("[budget] 2 batches x 8 rows\n") == 1, cap.out
        assert "warning: Adaptive batching: capped\n" in cap.err, cap.err
        assert "error: [checkpoint] could not save on the way out\n" in cap.err, cap.err
        assert "Adaptive batching" not in cap.out and "[budget]" not in cap.err
        assert "info:" not in cap.out and "warning:" not in cap.out
        assert core_logger.handlers == [], core_logger.handlers


def test_the_core_logger_is_at_info_by_import_and_stays_so_after_main_and_a_redirect(tool_env, monkeypatch):
    """Python's root logger sits at WARNING. Without core/runs.py's one setLevel at import, the window
    handler and the artifact file would drop every information record -- and caplog, which sets its
    own level, would have hidden that in the suites. So the level is set ONCE, by import, and the two
    front ends install and remove handlers only: pinned on the level after a real ``main`` and a real
    redirect, and on their source, so a later "helpful" setLevel cannot creep in."""
    import logging

    import core.tool as tool
    from core import orchestrator, runs
    from core.gui import streams
    from core.gui.worker import WorkerSignals
    from core.tool import logging_console
    from tests._fixtures import code_only, qt_app
    bounds, cell, root = tool_env
    core_logger = logging.getLogger("core")
    assert runs.LOGGER is core_logger and core_logger.level == logging.INFO

    monkeypatch.setattr(orchestrator, "build_prior", _Rec(_art(root, "prior")))
    assert main(["prior", *_cfg(bounds)]) == 0
    assert core_logger.level == logging.INFO

    qt_app()
    with streams.redirect_streams(WorkerSignals()):
        assert core_logger.level == logging.INFO
    assert core_logger.level == logging.INFO

    for fn in (streams.redirect_streams, logging_console.console_handlers, tool.main):
        assert "setLevel" not in code_only(fn), f"{fn.__name__} touches the logger's level"


# The console lines the GPU smoke gate in CLAUDE.md and docs/STATE.md reads by eye (spec §4.5), each with
# the ONE kind of call that must carry it in its module: an information record (stdout on the tool), a
# warning record (stderr, behind "warning: "), a framing print (stdout: no file=) or a Python warning
# (stderr).
_GATE_LINES = (
    ("core.orchestrator", "Reusing the Fisher rotation stored with the training checkpoint", "log.info"),
    ("core.orchestrator", "[fisher] eigenvalue spread", "log.info"),
    ("core.SBI.pipeline", "[checkpoint] resuming at batch ", "log.info"),
    ("core.SBI.pipeline", "OOM at simulation batch", "log.warning"),
    ("core.SBI.pipeline", "OUTSIDE the simulator retry", "log.warning"),
    ("core.SBI.pipeline", "hit a device error", "log.warning"),
    ("core.SBI.chi_probes", "probes masked", "warnings.warn"),
    ("core.tool.smoke", "=== ", "print"),
    ("core.tool.smoke", "[ok] ", "print"),
    ("core.tool.smoke", "[smoke] ALL STAGES COMPLETED", "print"),
    ("core.tool.smoke", "[smoke] *** FAILED in stage", "print"),
)


def _calls_carrying(module_name: str, needle: str) -> list[str]:
    """Every call in ``module_name``'s executable source whose FIRST argument holds ``needle`` inside
    one of its string pieces, named by its callee as written (``"log.info"``, ``"print"``,
    ``"warnings.warn"``), with ``"(file=)"`` appended when the call passes ``file=``. Parsed from
    ``code_only``, so a comment or docstring quoting the line cannot answer for the code."""
    import importlib

    from tests._fixtures import code_only
    tree = ast.parse(code_only(importlib.import_module(module_name)))
    kinds = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        pieces = [c.value for c in ast.walk(node.args[0])
                  if isinstance(c, ast.Constant) and isinstance(c.value, str)]
        if not any(needle in p for p in pieces):
            continue
        kind = ast.unparse(node.func)
        if any(k.arg == "file" for k in node.keywords):
            kind += "(file=)"
        kinds.append(kind)
    return kinds


def test_the_gate_lines_keep_their_text_and_stream(capsys):
    """The GPU smoke gate is read BY EYE off the console (CLAUDE.md, docs/STATE.md): run 2 must show
    "Reusing the Fisher rotation stored with the training checkpoint" and "[checkpoint] resuming at
    batch 4/4", run 2b no "[fisher]" line, runs 1 and 3 no OOM line and their masked-probe counts, and
    the smoke driver frames every stage with "=== <stage> ===", "[ok] <stage> in Xs" and "[smoke] ALL
    STAGES COMPLETED". No suite assertion read most of these before piece 3, so the conversion to
    logging could have reworded one, or demoted an OOM notice to information on stdout, with every
    suite green and the gate silently unreadable.

    Two halves. The SOURCE: each line is carried in its own module by exactly one kind of call, the
    kind that puts it on its stream, and by no other. The ROUTE: under the tool's console handlers an
    information record lands on stdout as bare text and a warning record on stderr behind "warning: ",
    so a level in the source IS a stream on the command line. In the window the same level is the
    pane's plain line or triangle (Task 16)."""
    import logging

    from core.tool.logging_console import console_handlers

    for module_name, needle, kind in _GATE_LINES:
        kinds = _calls_carrying(module_name, needle)
        assert kinds and set(kinds) == {kind}, (module_name, needle, kinds)

    capsys.readouterr()
    pipeline_log = logging.getLogger("core.SBI.pipeline")
    with console_handlers():
        pipeline_log.info("[checkpoint] resuming at batch 4/4 from X (no rotation)")
        pipeline_log.warning("training batch 3/4: OOM at simulation batch 64; retrying in chunks of 32")
    out, err = capsys.readouterr()
    assert out == "[checkpoint] resuming at batch 4/4 from X (no rotation)\n", out
    assert err == "warning: training batch 3/4: OOM at simulation batch 64; retrying in chunks of 32\n", err


# Every module whose in-stage prints piece 3 converted: Task 16 (file_manager.list_dir), Task 17 (the
# orchestrator), Task 18 (core/SBI and the solver), Task 19 (the diagnostics, FDT and the plot helpers).
_CONVERTED = ("core/orchestrator.py", "core/SBI", "core/Solvers/sdeint.py", "core/diagnostics", "core/FDT",
              "core/Helpers/visualizers.py", "core/Helpers/file_manager.py")


def test_no_print_call_remains_in_the_converted_modules():
    """The definition of done for V4's conversion (spec §4.1): no print() call is left in a module whose
    messages are the pipeline's own voice. A print there reaches the window at the level of whichever
    STREAM it used, never reaches the artifact's log.txt, and on the command line ignores the
    stdout/stderr split -- the three defects the conversion exists to remove. Parsed, not grepped: a
    docstring or a comment that mentions print() is not a call. The tool's framing prints (core/tool)
    stay prints and are deliberately outside this set (spec §4.1)."""
    root = config.REPO_ROOT
    files = []
    for rel in _CONVERTED:
        path = root / rel
        files += sorted(path.rglob("*.py")) if path.is_dir() else [path]
    assert len(files) > 20 and all(f.is_file() for f in files), files
    left = []
    for f in files:
        tree = ast.parse(f.read_text(encoding="utf-8"), filename=str(f))
        left += [f"{f.relative_to(root).as_posix()}:{n.lineno}" for n in ast.walk(tree)
                 if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "print"]
    assert left == [], f"print() calls left in the converted modules: {sorted(left)}"
