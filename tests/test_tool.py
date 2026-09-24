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


def test_a_stage_note_that_breaks_the_rule_is_refused_before_the_stage_runs(tool_env, monkeypatch,
                                                                            capsys):
    """The stage ``--note`` (config_args.add_name_flags) used to reach the manifest unchecked: piece
    4's store contract leaves the note rule to each FRONT END (``ArtifactStore.set_note``:
    "require_note is that rule, and both front ends run it"), ``create`` stores a note as it is
    given, and only ``artifacts note`` ran the rule. So a 250-character note, or one with a newline,
    was written into a record that the tool's own ``artifacts note`` would then refuse to write back.

    ``main`` now judges it ONCE, before any handler runs, for every subcommand that has the flag
    (piece 5, the Task 29 ruling): a refusal is the ladder's one line ending ``(--note)``, exit 1,
    and nothing -- not the config build, not the stage -- has run. The stage is a recorder, so a
    refused note is asserted by what never reached it.
    """
    from core import orchestrator
    from core.refusals import NOTE_MAX_CHARS
    bounds, cell, root = tool_env
    rec = _Rec(_art(root, "prior"))
    monkeypatch.setattr(orchestrator, "build_prior", rec)

    for bad in ("a" * (NOTE_MAX_CHARS + 1), "two\nlines"):
        capsys.readouterr()
        assert main(["prior", *_cfg(bounds), "--note", bad]) == 1
        out, err = capsys.readouterr()
        lines = [ln for ln in err.splitlines() if ln.startswith("prism prior: refused:")]
        assert len(lines) == 1 and lines[0].endswith("(--note)"), err
        assert "raised at" not in err and "Traceback" not in err, err
        assert rec.calls == [], "a refused note reached the stage"
        assert "[cfg]" not in out, "the note is judged before the config is even built"

    assert main(["prior", *_cfg(bounds), "--note", "a good note"]) == 0
    assert rec.calls[-1][1]["note"] == "a good note", "a good note reaches the stage unchanged"
    # The rule's one transformation is the trim, as `artifacts note` already applies it, so the
    # manifest holds the same text whichever command wrote it.
    assert main(["prior", *_cfg(bounds), "--note", "  padded  "]) == 0
    assert rec.calls[-1][1]["note"] == "padded"


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
    smoke -- keyed on smoke's own ``args.temp_store_root`` property (piece 5, E11), neither on the
    subcommand name nor on the --store-root flag, which fdt and crossval declare too -- and must leave
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


def _fdt_cfg_stub():
    """What a recorder standing in for an FDT config builder hands back: the fields the FDT branch of
    ``manifest.config_from_cfg`` reads (``_fdt_config_from_cfg``), plus ``sources``, and NO
    ``observation_mode`` -- its absence is what selects that branch. The handler opens an ``fdt``
    record on the config before it runs anything (``store.create("fdt", cfg)``), and that real store
    path stays under test (ruling F6), so a bare placeholder string would fail there on ``cfg.model``.
    Coupled to that branch on purpose: a field it starts reading must be added here."""
    return SimpleNamespace(
        model="HOPF", state_dep_drift=False, params_dict={"sigma_x": (0.1, (0.0, 1.0))},
        rescale_params={}, n_freqs=60, freq_bounds=(0.1, 30.0), ensemble_M=256, freqs_per_batch=1,
        F0=0.05, burn_in_nd=100.0, T_obs_periods=30, dt_nd=0.01, psd_T_obs_nd=8000.0, seed=None,
        units_dict=("nm", "ms"), hw=config.cpu_device(), sources={})


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
    fdt_cfg = _fdt_cfg_stub()

    def _fdt_cfg(model, state_dep_drift, cell_file, **kw):
        seen["make_fdt_config"] = (model, state_dep_drift, cell_file, kw)
        return fdt_cfg

    def _run_fdt(cfg, *, skip_sanity, confirm_production, writer, seed=None):
        seen["run_fdt"] = (cfg, skip_sanity, confirm_production)
        seen["fdt_writer_kind"] = writer.kind
        seen["fdt_writer_entered"] = writer.dir.exists()
        return SimpleNamespace(id="rec", path=writer.dir)

    # The sweep's recorder hands back the same kind of stub (ruling F6): the handler opens BOTH records
    # on it with store.create("fdt", cfg) before the study runs, so a placeholder string would fail
    # there. preset_name is what a sweep config adds (P72).
    sweep_cfg = _fdt_cfg_stub()
    sweep_cfg.preset_name = "exploratory"

    def _sweep_cfg(cell_file, **kw):
        seen["make_param_sweep_config"] = (cell_file, kw)
        return sweep_cfg, "S", "T"

    def _study(cfg, *, s_grid, t_grid, writers, seed=None):
        seen["run_param_study_cli"] = (cfg, s_grid, t_grid)
        seen["crossval_writer_kinds"] = sorted(w.kind for w in writers.values())
        seen["crossval_writers_entered"] = [w.dir.exists() for w in writers.values()]
        return [SimpleNamespace(id="s.h5", path=writers["s"].dir),
                SimpleNamespace(id="t.h5", path=writers["temp"].dir)]

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
    assert seen["run_fdt"] == (fdt_cfg, True, True)                   # --no-production absent

    # M4, fix round 1: NADROWSKI's state-dependent (multiplicative) drift, next to HOPF's False above.
    seen.clear()
    assert main(["fdt", "--cell", nad]) == 0
    assert seen["make_fdt_config"][1] is True, "NADROWSKI has state-dependent drift"

    seen.clear()
    assert main(["fdt", "--cell", cell, "--model", "hopf", "--no-production"]) == 0
    assert seen["make_fdt_config"][3] == {}, "an unset knob must not be passed: the default is in cli"
    assert seen["run_fdt"] == (fdt_cfg, False, False)

    # I1, fix round 1: NEITHER flag. The two cases above alone cannot catch confirm_production
    # cross-wired to skip_sanity: (skip_sanity=True, no_production=False) and (skip_sanity=False,
    # no_production=True) both coincidentally survive that bug (True/True and False/False again).
    # Only the both-False combination tells them apart -- it must come out (False, True).
    seen.clear()
    assert main(["fdt", "--cell", cell]) == 0
    assert seen["run_fdt"] == (fdt_cfg, False, True), \
        "neither flag: sanity runs, then production proceeds by default"
    assert seen["fdt_writer_kind"] == "fdt", \
        "the record is created by the front end and ENTERED by the stage (spec §1.2)"
    assert seen["fdt_writer_entered"] is False, "the handler entered the writer the stage must enter"
    assert "[prism fdt] record rec at " in capsys.readouterr().out

    seen.clear()
    assert main(["crossval", "--cell", nad, "--s-grid", "0", "0.1", "2",
                 "--t-grid", "1", "1.1", "3", "--n-freqs", "2", "--ensemble-m", "8",
                 "--freqs-per-batch", "4", "--f0", "0.2"]) == 0
    cell_file, kw = seen["make_param_sweep_config"]
    assert cell_file == nad
    assert set(kw) == {"preset", "preset_name", "s_spec", "t_spec", "n_freqs", "ensemble_M",
                       "freqs_per_batch", "F0"}
    assert kw["preset_name"] == "exploratory", \
        "the resolved dict does not say which preset it is; body.settings must hold the name (§4.4)"
    assert kw["s_spec"] == (0.0, 0.1, 2) and kw["t_spec"] == (1.0, 1.1, 3)
    assert isinstance(kw["s_spec"][2], int), "np.linspace refuses a float num"
    assert kw["n_freqs"] == 2 and kw["ensemble_M"] == 8
    assert kw["freqs_per_batch"] == 4 and kw["F0"] == 0.2
    assert kw["preset"] == dict(cli.SWEEP_PRESETS["exploratory"])
    assert seen["run_param_study_cli"] == (sweep_cfg, "S", "T")
    assert seen["crossval_writer_kinds"] == ["fdt", "fdt"], "one record per swept parameter (§4.1)"
    assert seen["crossval_writers_entered"] == [False, False], \
        "the handler entered a writer each sweep must enter on its own (spec §1.2)"
    out = capsys.readouterr().out
    assert "s.h5" in out and "t.h5" in out, out

    seen.clear()
    assert main(["crossval", "--cell", nad, "--preset", "production",
                 "--s-grid", "0", "0.1", "2", "--t-grid", "1", "1.1", "2"]) == 0
    cell_file, kw = seen["make_param_sweep_config"]
    assert set(kw) == {"preset", "preset_name", "s_spec", "t_spec", "n_freqs", "ensemble_M"}, \
        "freqs_per_batch and F0 were left unset -- they must not be forwarded"
    assert kw["preset_name"] == "production"
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
    # M1, as V3's one line since piece 5: the hint says where the NAME came from and the ladder's
    # fix sentence says which flag answers it, so neither has to do the other's job.
    assert "--model named it" in err, "M1: the refusal says where the name came from"
    assert err.rstrip().endswith("(--model)"), err
    assert "ValueError" not in err and "raised at" not in err, err

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


def test_an_unsupported_model_named_by_the_cells_folder_is_one_refusal_line(tool_env, tmp_path,
                                                                            capsys):
    """The CELL-FOLDER branch of the unsupported-model hint -- the first of the two gaps
    docs/STATE.md names. Only the ``--model`` branch has ever been tested
    (test_fdt_and_crossval_usage_errors), and the two say different things on purpose: passing the
    wrong ``--model`` and standing a cell in the wrong folder are different mistakes with different
    fixes, and the operator cannot tell which one happened from the reason alone.

    V3 as well (spec §6.2): this used to be a bare ``ValueError``, so ``main``'s unconverted-refusal
    rung printed ``refused: ValueError: ... [raised at fdt.py:139]`` -- the class name and the raise
    site are a hedge for a bug disguised as a refusal, and a fielded ``Refusal`` no longer needs
    either. The cell file is never opened: the model gate runs before the builder, which is the
    point -- refuse before the spend.
    """
    from core.tool import main

    cell = tmp_path / "nosuchmodel" / "cell.txt"
    cell.parent.mkdir(parents=True)
    cell.write_text("", encoding="utf-8")

    capsys.readouterr()
    assert main(["fdt", "--cell", str(cell)]) == 1
    err = capsys.readouterr().err
    lines = [ln for ln in err.splitlines() if ln.startswith("prism fdt: refused:")]
    assert len(lines) == 1, err
    assert "Unknown model 'NOSUCHMODEL'" in lines[0], lines[0]
    assert "the cell's parent folder named it" in lines[0], lines[0]
    assert lines[0].endswith("(--model)"), lines[0]
    assert "ValueError" not in err and "raised at" not in err, err


def test_fdt_and_crossval_take_store_root_and_otherwise_follow_the_environment(
        tool_env, tmp_path, monkeypatch):
    """E11, first half: both analyses write records now, so both need the flag that says WHERE --
    and without it they must follow PRISM_ARTIFACTS like every subcommand but `smoke`, never a
    throwaway temp root nobody would think to look in.

    The handler is replaced by a recorder, so what is asserted is the root ``main`` OPENED THE STORE
    ON rather than anything a real campaign would write: the dispatch is the subject, and a real
    fdt run is minutes of simulation (it is `slow`-marked, in test_fdt_and_crossval_run_at_tiny_size).
    """
    from core import config
    from core.tool import fdt as fdt_mod
    from core.tool import main

    _bounds, cell, root = tool_env
    seen = {}

    def _rec(args, store):
        seen["root"] = Path(store.root).resolve()
        seen["flag"] = args.store_root
        return 0

    monkeypatch.setattr(fdt_mod, "run_fdt_cmd", _rec)
    monkeypatch.setattr(fdt_mod, "run_crossval", _rec)

    assert main(["fdt", "--cell", cell]) == 0
    assert seen["flag"] is None
    assert seen["root"] == Path(config.artifacts_root()).resolve(), \
        "with no --store-root, fdt must follow PRISM_ARTIFACTS, not a temp directory"
    assert seen["root"] == Path(root).resolve()

    named = tmp_path / "somewhere_else"
    assert main(["fdt", "--cell", cell, "--store-root", str(named)]) == 0
    assert seen["root"] == named.resolve() and seen["flag"] == str(named)

    nad = str(config.CELL_PATH / "nadrowski" / "master_spont.txt")
    assert main(["crossval", "--cell", nad, "--s-grid", "0", "0.1", "2",
                 "--t-grid", "1", "1.1", "2", "--store-root", str(named)]) == 0
    assert seen["root"] == named.resolve(), "crossval takes the same flag, with the same meaning"


def test_fdt_and_crossval_take_seed_and_hand_it_to_the_builder_and_the_run(tool_env, tmp_path,
                                                                           monkeypatch):
    """E7's command-line half (P79). A record carries the seed its run used; without the flag that
    supplies one, that seed can never be supplied back -- the defect E7 names. The ONE integer
    reaches the builder (through ``knobs``, so an unset flag forwards nothing and the builder's own
    default stands) and the run (explicitly, so None there means "draw one") -- the rule both panels
    follow (P12).

    ``ArtifactStore.create`` is replaced because the builder recorders return a placeholder, not an
    FDTConfig, and ``store.create("fdt", cfg)`` reads the run's settings off its cfg: the dispatch is
    the subject here, not the record."""
    from core import cli, config
    from core.artifacts import ArtifactStore
    from core.FDT import cross_validation, fdt_pipeline

    seen = {}

    def _create(self, kind, cfg=None, *, name="", note=""):
        return SimpleNamespace(kind=kind, id="rec", name=name, note=note, dir=tmp_path / "rec",
                               body={})

    def _fdt_cfg(model, state_dep_drift, cell_file, **kw):
        seen["fdt_builder"] = kw
        return "CFG"

    def _run_fdt(cfg, *, skip_sanity, confirm_production, writer, seed=None):
        seen["fdt_run"] = seed
        return SimpleNamespace(id="rec", name="", path=writer.dir)

    def _sweep_cfg(cell_file, **kw):
        seen["sweep_builder"] = kw
        return "CFG", "S", "T"

    def _study(cfg, *, s_grid, t_grid, writers, seed=None):
        seen["study"] = seed
        return [SimpleNamespace(id="s", name="", path=writers["s"].dir),
                SimpleNamespace(id="t", name="", path=writers["temp"].dir)]

    monkeypatch.setattr(ArtifactStore, "create", _create)
    monkeypatch.setattr(cli, "make_fdt_config", _fdt_cfg)
    monkeypatch.setattr(fdt_pipeline, "run_fdt", _run_fdt)
    monkeypatch.setattr(cli, "make_param_sweep_config", _sweep_cfg)
    monkeypatch.setattr(cross_validation, "run_param_study_cli", _study)

    cell = str(config.CELL_PATH / "hopf" / "cell.txt")
    nad = str(config.CELL_PATH / "nadrowski" / "master_spont.txt")
    assert main(["fdt", "--cell", cell, "--seed", "7"]) == 0
    assert seen["fdt_builder"].get("seed") == 7 and seen["fdt_run"] == 7, seen
    assert main(["crossval", "--cell", nad, "--s-grid", "0", "0.1", "2",
                 "--t-grid", "1", "1.1", "2", "--seed", "7"]) == 0
    assert seen["sweep_builder"].get("seed") == 7 and seen["study"] == 7, seen

    seen.clear()
    assert main(["fdt", "--cell", cell]) == 0
    assert "seed" not in seen["fdt_builder"], "an unset --seed forwards nothing to the builder"
    assert seen["fdt_run"] is None, "no --seed: the run draws one and records it (E7, P12)"


def test_fdt_and_crossval_declare_seed_and_store_root_by_name(capsys):
    """P79 and E11, pinned on THESE TWO parsers by name. The FLAG-table pin (tests/test_refusals.py)
    only asks that SOME subcommand defines ``--seed``, and ``smoke`` and the diagnostics already did
    -- so it stayed green while neither analysis took the flag its own refusals name. The ``--help``
    each prints is what an operator reads, so it is checked too.

    And the temp-root property is ``smoke``'s ALONE: every other subcommand must leave
    ``temp_store_root`` unset, or ``main`` would hand it a throwaway root, rmdir its root after a
    failure and print smoke's resume advice (E11)."""
    parser = build_parser()
    for name in ("fdt", "crossval"):
        sub = parser.subcommands[name]
        seed = sub._option_string_actions.get("--seed")
        assert seed is not None, f"{name} does not accept --seed"
        assert seed.dest == "seed" and seed.type is int and seed.default is None, (name, seed)
        root = sub._option_string_actions.get("--store-root")
        assert root is not None, f"{name} does not accept --store-root"
        assert root.dest == "store_root" and root.default is None, (name, root)

        capsys.readouterr()
        assert main([name, "--help"]) == 0
        out = capsys.readouterr().out
        assert "--seed" in out and "--store-root" in out, out

    flagged = sorted(name for name, sub in parser.subcommands.items()
                     if sub.get_default("temp_store_root"))
    assert flagged == ["smoke"], flagged


def test_an_fdt_run_that_fails_leaves_the_artifacts_root_where_it_found_it(tmp_path, monkeypatch,
                                                                          capsys):
    """E11's second half, and the accidental flip it exists to prevent. ``main`` removes an
    auto-created store root when the run did not succeed (``_remove_if_still_empty``) -- a safety
    net written for ``smoke``'s own ``mkdtemp`` directory. Keyed on the FLAG's presence, declaring
    ``--store-root`` on ``fdt`` would point that ``rmdir`` at the operator's real ``Artifacts/``
    root. ``rmdir`` refuses a non-empty directory, so nothing would be lost TODAY -- which is
    exactly why this needs a test rather than a reader: the hazard is invisible on any machine whose
    store already holds an artifact.

    PRISM_ARTIFACTS points at a directory that does not exist yet, so ``main`` creates it and the
    root is empty at the moment the cleanup would run: the one state in which the wrong dispatch
    actually deletes something.
    """
    from core import config
    from core.refusals import Refusal
    from core.tool import fdt as fdt_mod
    from core.tool import main

    root = tmp_path / "Artifacts"
    monkeypatch.setenv("PRISM_ARTIFACTS", str(root))
    assert not root.exists()

    def _refuse(args, store):
        raise Refusal("nothing here is real", field="cell")

    monkeypatch.setattr(fdt_mod, "run_fdt_cmd", _refuse)
    capsys.readouterr()
    assert main(["fdt", "--cell", str(tmp_path / "nadrowski" / "cell.txt")]) == 1
    assert root.is_dir(), \
        "a failed fdt run removed the operator's artifacts root (E11: auto_root is smoke's alone)"
    assert Path(config.artifacts_root()).resolve() == root.resolve()


def test_a_seed_the_generator_cannot_take_is_refused_naming_the_flag(tool_env, capsys):
    """``--seed`` is ``type=int``, so argparse takes any integer; its range is the builders' rule
    (A3). One above 2**64 - 1 used to pass that rule, open the record and then overflow the
    generator inside the run. Now the builder refuses it -- exit 1, one ``refused:`` line ending in
    the flag -- before the record is created: nothing new appears under the root's ``fdt/``."""
    _bounds, _cell, root = tool_env
    kind_dir = Path(root) / "fdt"
    before = sorted(kind_dir.iterdir()) if kind_dir.is_dir() else []

    hopf = str(config.CELL_PATH / "hopf" / "cell.txt")
    nad = str(config.CELL_PATH / "nadrowski" / "master_spont.txt")
    for cmd, argv in (("fdt", ["fdt", "--cell", hopf]),
                      ("crossval", ["crossval", "--cell", nad, "--s-grid", "0", "0.1", "2",
                                    "--t-grid", "1", "1.1", "2"])):
        capsys.readouterr()
        assert main([*argv, "--seed", "18446744073709551616"]) == 1, cmd
        err = capsys.readouterr().err
        lines = [ln for ln in err.splitlines() if ln.startswith(f"prism {cmd}: refused:")]
        assert len(lines) == 1, err
        assert "must be at most 18446744073709551615" in lines[0], lines[0]
        assert lines[0].endswith("(--seed)"), lines[0]
        assert "Overflow" not in err and "Traceback" not in err, err

    after = sorted(kind_dir.iterdir()) if kind_dir.is_dir() else []
    assert after == before, "a refused seed opened a record"


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
    for them -- there is no cache to resume. Since piece 5 the recovery story is the RECORD the run
    was writing: kept, marked unfinished, listed and deletable (E2); re-running still starts over,
    because nothing resumes. Every OTHER subcommand's Ctrl-C message is untouched
    (test_smoke_ctrl_c_prints_store_specific_resume_advice and
    test_ctrl_c_mid_simulation_keeps_the_committed_batches still pin the generic wording).

    Ruling F20: a static note cannot carry the record's id, so the handler prints it -- id and
    directory -- the moment ``store.create`` mints it, which is before anything can be interrupted;
    and the note says what an operator who passed ``--store-root`` must do first, because the
    ``artifacts`` commands read only PRISM_ARTIFACTS."""
    from core import cli, config
    from core.FDT import fdt_pipeline
    from core.tool import main

    def _boom(cfg, *, skip_sanity, confirm_production, writer, seed=None):
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "make_fdt_config", lambda *a, **k: _fdt_cfg_stub())
    monkeypatch.setattr(fdt_pipeline, "run_fdt", _boom)
    cell = str(config.CELL_PATH / "hopf" / "cell.txt")
    capsys.readouterr()
    assert main(["fdt", "--cell", cell]) == 130
    captured = capsys.readouterr()
    err = captured.err
    assert "interrupted" in err
    # E2, through the note: what survives an interrupt is a NAMED RECORD, kept and marked
    # unfinished -- not "the plots already written under <artifacts root>/fdt", which is where these
    # outputs stopped going when they became artifacts (spec §6.1).
    assert "unfinished" in err and "artifacts list fdt" in err, err
    assert "artifacts rm fdt" in err, "the note must say how to clear the record it just named"
    assert "from scratch" in err, err
    assert "<artifacts root>/fdt" not in err, "the flat output folder is gone"
    assert "--resume" not in err
    assert "--store-root" in err and "PRISM_ARTIFACTS" in err, err
    written = [ln for ln in captured.out.splitlines() if ln.startswith("[prism fdt] writing record ")]
    assert len(written) == 1, captured.out
    assert str(config.artifacts_root() / "fdt") in written[0], written[0]


def test_crossval_ctrl_c_names_each_record_and_how_to_clear_them(tool_env, monkeypatch, capsys):
    """The crossval twin of the test above (F20): a study opens TWO records, one per swept parameter
    (spec §4.1), so the handler prints one ``writing record`` line for each before the study runs,
    and the note is crossval's own -- one sweep's record may be finished, the one interrupted is kept
    unfinished, and a sweep that never started left nothing."""
    from core import cli, config
    from core.FDT import cross_validation
    from core.tool import main

    def _boom(*a, **k):
        raise KeyboardInterrupt

    sweep_cfg = _fdt_cfg_stub()
    sweep_cfg.preset_name = "exploratory"
    monkeypatch.setattr(cli, "make_param_sweep_config", lambda *a, **k: (sweep_cfg, "S", "T"))
    monkeypatch.setattr(cross_validation, "run_param_study_cli", _boom)
    nad = str(config.CELL_PATH / "nadrowski" / "master_spont.txt")
    capsys.readouterr()
    assert main(["crossval", "--cell", nad, "--s-grid", "0", "0.1", "2",
                 "--t-grid", "1", "1.1", "2"]) == 130
    captured = capsys.readouterr()
    err = captured.err
    assert err.startswith("prism crossval: interrupted:"), err
    assert "unfinished" in err and "artifacts list fdt" in err and "artifacts rm fdt" in err, err
    assert "--store-root" in err and "PRISM_ARTIFACTS" in err, err
    assert "from scratch" in err and "--resume" not in err, err
    assert "<artifacts root>/crossval" not in err, "the flat output folder is gone"
    written = [ln for ln in captured.out.splitlines()
               if ln.startswith("[prism crossval] writing record ")]
    assert len(written) == 2, captured.out
    assert "S sweep" in written[0] and "T_a/T sweep" in written[1], written


def _compare_root(tmp_path, monkeypatch, n=2):
    """A store of ``n`` finished single-cell records named ``cell_0`` ... with PRISM_ARTIFACTS pointing
    at it -- ``compare``, like the ``artifacts`` family, has no --store-root and reads only that
    variable. Returns ``(store, ids)``."""
    from core.artifacts import ArtifactStore
    from tests._fixtures import build_fdt_record
    root = tmp_path / "A"
    store = ArtifactStore(root)
    ids = [build_fdt_record(store, name=f"cell_{i}") for i in range(n)]
    monkeypatch.setenv("PRISM_ARTIFACTS", str(root))
    return store, ids


def _compare_stub(w, records, *, sink, **_options):
    """What every real mode does before it draws: the curves on the common grid, into data.h5."""
    from core.FDT import compare as cmp
    curves = [cmp.curve_of(r) for r in records]
    grid = cmp.common_grid(curves)
    values = [cmp.interpolate_onto(grid, c.omegas, c.ratio) for c in curves]
    cmp.write_curves(w, grid, values, [c.label for c in curves],
                     constants=[(c.omega_0, c.prefactor) for c in curves])
    return {"n_records": len(curves)}, []


def test_compare_is_a_nested_subcommand_whose_help_costs_no_torch():
    """Spec §7.1: one subcommand, four modes, ``identifiability``'s shape -- a parser per mode, so a
    flag that means nothing to a mode is an argparse error rather than a setting silently ignored:
    ``--prefactor`` exists on renormalise alone (and is required there), ``--at`` on sweeps alone.
    ``--record`` is repeatable and required on every mode. No configuration flags and no
    ``--store-root`` (the ``artifacts`` family's reasons: the root is PRISM_ARTIFACTS), and each mode
    carries its OWN interrupt note -- a comparison's record is kept on Ctrl-C and nothing resumes it
    (F56), so main's generic "--resume require" advice would be wrong. ``--help`` costs no torch
    import, checked in a FRESH interpreter because this process imported torch long ago."""
    import argparse
    import subprocess
    import sys

    from core.tool import fdt as tool_fdt

    p = build_parser().subcommands["compare"]
    modes = {name: sub for a in p._actions if isinstance(a, argparse._SubParsersAction)
             for name, sub in a.choices.items()}
    assert sorted(modes) == ["cells", "renormalise", "repeats", "sweeps"], sorted(modes)
    for name, mode in modes.items():
        record = mode._option_string_actions["--record"]
        assert record.required and isinstance(record, argparse._AppendAction), (name, record)
        assert "--name" in mode._option_string_actions and "--note" in mode._option_string_actions
        for flag in ("--bounds", "--cell", "--model", "--device", "--store-root", "--seed"):
            assert flag not in mode._option_string_actions, (name, flag)
        assert ("--prefactor" in mode._option_string_actions) == (name == "renormalise"), name
        assert ("--at" in mode._option_string_actions) == (name == "sweeps"), name
        assert mode.get_default("interrupt_note") == tool_fdt.COMPARE_INTERRUPT_NOTE, name
    assert modes["renormalise"]._option_string_actions["--prefactor"].required
    assert modes["sweeps"]._option_string_actions["--at"].default is None

    probe = ("import sys\n"
             "from core.tool import main\n"
             "rc = [main(['--help']), main(['compare', '--help']), main(['compare', 'cells', '--help'])]\n"
             "bad = sorted(m for m in sys.modules if m == 'torch' or m.startswith('torch.'))\n"
             "sys.exit(0 if rc == [0, 0, 0] and not bad else repr((rc, bad[:3])))\n")
    r = subprocess.run([sys.executable, "-c", probe], cwd=str(config.REPO_ROOT),
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "compare" in r.stdout and "--record" in r.stdout, r.stdout


def test_a_compare_ref_that_names_nothing_is_refused_naming_the_record_flag(tmp_path, monkeypatch,
                                                                            capsys):
    """F58. The store refuses a ref it cannot resolve under ``field="artifact"`` -- a key whose flag is
    None, because in the ``artifacts`` family the artifact is positional -- so a mistyped ``--record``
    used to be refused naming no flag at all. The comparison re-raises it under its own key, so the
    one line names the flag that answers it; and like every comparison refusal it lands before the
    record opens, leaving nothing on disk."""
    from core.FDT import compare as cmp
    store, ids = _compare_root(tmp_path, monkeypatch)
    monkeypatch.setitem(cmp._DRAWERS, "cells", _compare_stub)
    before = sorted(p.name for p in store.kind_dir("fdt").iterdir())
    capsys.readouterr()
    assert main(["compare", "cells", "--record", ids[0], "--record", "no_such_run"]) == 1
    err = capsys.readouterr().err
    lines = [ln for ln in err.splitlines() if ln.strip()]
    assert len(lines) == 1 and lines[0].startswith("prism compare: refused:"), err
    assert "'no_such_run' names no fdt record" in lines[0] and lines[0].endswith("(--record)"), err
    assert sorted(p.name for p in store.kind_dir("fdt").iterdir()) == before, "a refusal opened a record"


def test_compare_ctrl_c_keeps_the_record_and_says_nothing_resumes(tmp_path, monkeypatch, capsys):
    """F56, the comparison's own interrupt note. A comparison's record is progressive, so a Ctrl-C
    after it opened KEEPS it, marked unfinished (E2) -- the opposite of main's generic "the artifact
    being written was removed" -- and there is no cache and no --resume. The note says where the
    record is (the ``Writing comparison record`` line, printed the moment it opened, since fixed text
    cannot carry the id), how to list and remove it, that re-running draws a new one, and that a
    --name stays taken until the unfinished record is removed."""
    from core.artifacts import ArtifactStore
    from core.FDT import compare as cmp
    store, ids = _compare_root(tmp_path, monkeypatch)

    def _cancelled(w, records, *, sink):
        _compare_stub(w, records, sink=sink)
        raise KeyboardInterrupt

    monkeypatch.setitem(cmp._DRAWERS, "cells", _cancelled)
    capsys.readouterr()
    assert main(["compare", "cells", "--record", ids[0], "--record", ids[1], "--name", "ab"]) == 130
    captured = capsys.readouterr()
    err = captured.err
    assert err.startswith("prism compare: interrupted:"), err
    assert "KEEPS" in err and "unfinished" in err and "artifacts list fdt" in err, err
    assert "artifacts rm fdt" in err and "new record" in err and "--name" in err, err
    assert "--resume" not in err and "was removed" not in err, err
    kept = ArtifactStore(tmp_path / "A").load_fdt("ab")
    assert kept.body["complete"] is False and kept.body["study"] == "comparison", kept.body
    assert f"Writing comparison record {kept.id} at " in captured.out, captured.out


def test_a_comparison_whose_source_was_deleted_still_lists_through_the_tool(tmp_path, monkeypatch,
                                                                            capsys):
    """Spec §7.3/§8.2 and design ruling R5, through the command line: ``artifacts rm`` deletes a run a
    comparison drew without refusing (the sources are in the comparison's body, which the dependency
    check does not read), and the comparison is still listed, finished, by ``artifacts list fdt`` --
    while ``artifacts summary`` prints ``MISSING fdt [<id>]`` for the run that is gone."""
    from core.FDT import compare as cmp
    store, ids = _compare_root(tmp_path, monkeypatch)
    monkeypatch.setitem(cmp._DRAWERS, "cells", _compare_stub)
    assert main(["compare", "cells", "--record", "cell_0", "--record", "cell_1",
                 "--name", "both_cells"]) == 0
    assert main(["artifacts", "rm", "fdt", ids[0]]) == 0

    capsys.readouterr()
    assert main(["artifacts", "list", "fdt"]) == 0
    lines = capsys.readouterr().out.splitlines()
    header = next(ln for ln in lines if ln.split()[:1] == ["name"])
    start, end = header.index("finished"), header.index("note")
    row = next(ln for ln in lines if "both_cells" in ln)
    assert row[start:end].strip() == "yes", lines
    assert not any("cell_0" in ln for ln in lines), lines

    assert main(["artifacts", "summary", "fdt", "both_cells"]) == 0
    out = capsys.readouterr().out
    assert f"MISSING fdt [{ids[0]}]" in out and f"fdt cell_1 [{ids[1]}]" in out, out


def test_the_fdt_subcommands_no_longer_say_they_have_no_bounds_file():
    """Spec §3.2. Two sentences in this module claimed these analyses have no bounds file. They are
    false, and the record makes the falsehood expensive: ``cli.parse_cell`` DOES resolve one
    (``resolve_bounds_for_cell`` -- the same-named sibling, else the folder's master), and on the
    decoupled path that file "defines the param set + order" (core/cli.py:156-161). A record that
    did not name which bounds file resolved would not say which parameter set its numbers were
    measured under.

    A source scan, because these are DOCSTRINGS -- nothing executes them, so nothing else can catch
    them going stale. What is true and must stay said is that neither subcommand takes a ``--bounds``
    flag or an observation mode.
    """
    import inspect
    from core.tool import fdt

    text = inspect.getsource(fdt)
    assert "no bounds file" not in text, \
        "a bounds file DOES resolve for the cell; it is recorded by path and hash"
    assert "resolve_bounds_for_cell" in fdt.__doc__, fdt.__doc__
    assert "resolve" in inspect.getdoc(fdt.model_for_cell), inspect.getdoc(fdt.model_for_cell)


@pytest.mark.slow
def test_fdt_and_crossval_run_at_tiny_size(tool_env, capsys):
    """The real pipelines, at the smallest sizes the flags allow, writing real ``fdt`` records.

    No new science: this asks only whether the two subcommands drive the campaigns end to end and
    leave behind what piece 5 promises -- a record per run (two for the study), its numbers in
    ``data.h5``, its pictures under ``figures/``, and a body that says what ran. Until piece 5 it
    globbed ``artifacts_root()/fdt`` and ``/crossval`` for loose PNGs and .h5 files, which is
    exactly what E1 moved (spec section 8.3).

    Neither subcommand is given --store-root, deliberately: with the flag unset these two follow
    ``config.artifacts_root()`` and NOT a throwaway temp root (spec section 6.1), and this test is
    what keeps that true end to end. ``tool_env`` is what makes "the temp artifacts root" true --
    it sets PRISM_ARTIFACTS, which ``artifacts_root()`` reads at every call; without it the records
    land in the real ``Artifacts/`` and the session teardown fails.

    The single-cell leg runs a shipped NADROWSKI cell. Until piece 5 it ran the shipped Hopf cell,
    which the band check now refuses at the default band by design: its lowest probe, 0.1 x its
    spontaneous peak of about 0.268 (ND), lies below the spectrum's first resolved bin, 0.0383. That
    refusal is pinned in tests/test_fdt_user.py; here it would test a refusal instead of a record.
    The shipped Nadrowski cells clear that bin with a 1.3-1.4x margin.

    The study writes TWO records, one per swept parameter, carrying ONE seed (spec section 4.1): an
    activity sweep that failed entirely no longer costs the temperature sweep.

    Measured 2026-09-24: 452.03 s on the CPU. (The previous figure, 598.55 s recorded 2026-09-15,
    was stale against every gate since -- 243, 208 and 229 s -- so it is replaced by a measurement,
    not by a copied number.) It is longer than those gates because the single-cell leg now runs a
    Nadrowski cell: about 173 s of the total, against 138 s and 141 s for the two sweeps (the
    records' own timestamps).
    """
    import math

    import h5py

    from core import config
    from core.artifacts import ArtifactStore
    from core.tool import main

    cell = str(config.CELL_PATH / "nadrowski" / "master_spont.txt")
    capsys.readouterr()
    assert main(["fdt", "--cell", cell, "--n-freqs", "2", "--ensemble-m", "8",
                 "--skip-sanity"]) == 0
    out = capsys.readouterr().out

    store = ArtifactStore(config.artifacts_root())
    # ``list`` returns complete rows first, newest first, so [0] is this run even if a sibling test
    # in this module has left an older fdt record in the shared root.
    single = [s for s in store.list("fdt") if s.study == "single"]
    assert single and single[0].finished, single
    rec = store.load_fdt(single[0].id)
    # The record the tool named before it spent anything (ruling F20) is the one it wrote.
    assert f"[prism fdt] writing record {rec.id} at " in out, out[-2000:]
    assert rec.body["study"] == "single" and rec.body["complete"] is True
    assert isinstance(rec.body["seed"], int), rec.body["seed"]
    assert rec.body["settings"]["n_freqs"] == 2 and rec.body["settings"]["ensemble_M"] == 8
    assert rec.body["settings"]["skip_sanity"] is True
    # 2 frequencies and 8 trajectories are the thin-setting THRESHOLDS themselves, not below them
    # (spec section 3.3: "fewer than two grid frequencies, fewer than eight trajectories"), so this
    # run is not marked as a quick look.
    assert rec.body["notices"] == [], rec.body["notices"]
    assert rec.body["offgrid"]["of"] == 2, rec.body["offgrid"]
    assert rec.body["grid"]["n_freqs"] == 2, rec.body["grid"]
    # The numbers, not only the pictures (E1).
    assert "data.h5" in rec.manifest.payloads and rec.data_path.exists()
    # The contract's single-cell layout (P5, P6, P71), BY NAME: the names core.FDT.compare reads
    # back. The only other writer of this layout is a test fixture, so this is the one place a REAL
    # run's file is checked against the names its reader expects (rulings F21/F22).
    with h5py.File(rec.data_path, "r") as h5:
        assert h5.attrs["study"] == "single", dict(h5.attrs)
        assert float(h5.attrs["omega_0"]) > 0.0, dict(h5.attrs)
        assert math.isfinite(float(h5.attrs["prefactor"])), dict(h5.attrs)
        for key in ("omega_grid", "T_eff_over_T", "chi_prime", "chi_double_prime"):
            assert key in h5 and h5[key].shape == (2,), (key, list(h5))
        assert "PSD_omegas" in h5 and "PSD_G" in h5, list(h5)
        assert h5["PSD_omegas"].shape == h5["PSD_G"].shape, list(h5)
    # ... and read back by the reader itself: the comparison's own curve_of, on the record a real run
    # wrote rather than on the fixture's (F21/F22), labelled by the cell it measured.
    from core.FDT import compare as cmp
    curve = cmp.curve_of(store.load_fdt(rec.id))
    assert curve.id == rec.id and curve.label == "master_spont", curve
    assert curve.omegas.shape == curve.ratio.shape == (2,), curve
    assert curve.omega_0 > 0.0 and math.isfinite(curve.prefactor), curve
    # The four figures by name: ``writer.figure_path`` names each by the slug of its title.
    assert sorted(rec.manifest.figures) == [
        "figures/chi_components.png", "figures/effective_temperature_ratio.png",
        "figures/spontaneous_psd.png", "figures/spontaneous_trajectory.png"], rec.manifest.figures
    for fig in rec.manifest.figures:
        assert (rec.path / fig).exists(), fig
    # Provenance: the cell by path and hash, relative to Resources/ (provenance.file_ref).
    assert rec.manifest.inputs["cell"]["path"] == "Cells/nadrowski/master_spont.txt", \
        rec.manifest.inputs
    assert rec.manifest.inputs["cell"]["sha256"], rec.manifest.inputs
    assert rec.manifest.inputs["bounds"] is not None, rec.manifest.inputs

    assert main(["crossval", "--cell", cell, "--s-grid", "0", "0.1", "2",
                 "--t-grid", "1", "1.1", "2", "--n-freqs", "2", "--ensemble-m", "8"]) == 0
    out = capsys.readouterr().out

    sweeps = [s for s in store.list("fdt") if s.study == "sweep"]
    assert len(sweeps) == 2, sweeps
    assert all(s.finished for s in sweeps), sweeps
    # Every real operating point lands at this size (ruling F53): a failed point is a finding to
    # report, not a tolerance to allow.
    assert {s.points_planned for s in sweeps} == {2}, sweeps
    assert {s.points_done for s in sweeps} == {2}, sweeps
    assert {s.points_failed for s in sweeps} == {0}, sweeps

    recs = [store.load_fdt(s.id) for s in sweeps]
    assert {r.body["points"]["param"] for r in recs} == {"s", "temp"}, \
        [r.body["points"] for r in recs]
    assert len({r.body["seed"] for r in recs}) == 1, \
        "the study draws ONE seed and records it on both records (spec section 4.1)"
    for r in recs:
        param = r.body["points"]["param"]
        assert r.body["complete"] is True
        # A sweep's per-point grids live in data.h5, so the body's single grid block is null
        # (spec section 2.3).
        assert r.body["grid"] is None, r.body["grid"]
        assert r.body["settings"]["preset"] == "exploratory", r.body["settings"]
        assert "data.h5" in r.manifest.payloads and r.data_path.exists()
        # Each sweep plots itself into its own record (spec section 4.1).
        assert r.manifest.figures == [f"figures/fdt_ratio_vs_{param}.png"], r.manifest.figures
        assert (r.path / r.manifest.figures[0]).exists(), r.manifest.figures
        # The line the tool printed for this record before either sweep spent anything (F20) names
        # the sweep the record holds -- what the interrupt note sends the operator back to.
        line = [ln for ln in out.splitlines()
                if ln.startswith(f"[prism crossval] writing record {r.id} at ")]
        label = {"s": "S", "temp": "T_a/T"}[param]
        assert len(line) == 1 and line[0].endswith(f"(the {label} sweep)"), (line, out[-2000:])


def test_the_nadrowski_only_sanity_checks_are_selected_for_a_nadrowski_cell(monkeypatch, caplog):
    """``run_all_sanity`` keeps ``passive_baseline`` and ``high_freq_fdt`` for NADROWSKI and drops
    them for every other model (core/FDT/sanity.py: ``_runs_nadrowski_only_checks`` decides,
    ``_NADROWSKI_ONLY`` names the two), because both reason about parameters -- the s-feedback and
    the motor thermostat -- that only Nadrowski has. Until piece 5 the only end-to-end test of the
    ``fdt`` subcommand ran a HOPF cell with ``--skip-sanity``, so the branch that KEEPS the two was
    exercised nowhere and neither was the note that announces dropping them. Spec section 1's last
    bullet.

    The five check bodies are stubbed, so this asserts the SELECTION and the note, not the physics;
    the slow test below runs them for real.
    """
    from core.FDT import sanity

    seen = []

    def _stub(name):
        def _fn(cfg, **kw):
            seen.append(name)
            return True, {"stub": name}
        return _fn

    for fn_name in ("check_passive_baseline", "check_high_freq_fdt", "check_linearity",
                    "check_ensemble_convergence", "check_psd_window"):
        monkeypatch.setattr(sanity, fn_name, _stub(fn_name))

    class _Cfg:
        def __init__(self, model):
            self.model = model

    nadrowski = sanity.run_all_sanity(_Cfg("NADROWSKI"))
    assert list(nadrowski) == ["passive_baseline", "high_freq_fdt", "linearity",
                               "ensemble_convergence", "psd_window"], list(nadrowski)
    assert seen[:2] == ["check_passive_baseline", "check_high_freq_fdt"], seen
    # No caplog.set_level: the ``core`` logger is at INFO by import (core/runs.py).
    assert "Nadrowski-specific and are skipped" not in caplog.text, caplog.text

    seen.clear()
    caplog.clear()
    hopf = sanity.run_all_sanity(_Cfg("HOPF"))
    assert list(hopf) == ["linearity", "ensemble_convergence", "psd_window"], list(hopf)
    assert "check_passive_baseline" not in seen, seen
    assert "Nadrowski-specific and are skipped for HOPF" in caplog.text, caplog.text


@pytest.mark.slow
def test_fdt_runs_the_nadrowski_sanity_checks_end_to_end(tool_env, tmp_path, capsys):
    """The real sanity checks on a real Nadrowski cell, and the passive-baseline figure they draw.

    This is the gap spec section 1's last bullet names: the single-cell leg above runs with
    --skip-sanity (on a Hopf cell until piece 5, a Nadrowski one since), so
    ``check_passive_baseline`` and ``check_high_freq_fdt`` -- the two checks that decide whether the
    whole PSD / lock-in / noise-prefactor convention is right, and the only ones that draw a figure
    of their own -- had never executed under test. Everything is real: the campaigns, the checks,
    the production sweep and the record.

    Production is NOT skipped. ``--no-production`` would be cheaper, and its contract is fixed too
    (planning ruling P55: a finished record with ``grid`` and ``offgrid`` null) -- but it stops
    before Campaign 1, so on a Nadrowski cell with the checks on, the two campaigns, the three
    production figures and ``data.h5`` would go unexercised. This test runs the whole path and
    records what that costs.

    Its own --store-root, so the record cannot be confused with the one the tiny-size test above
    writes into the module's shared artifacts root. ``tool_env`` is still requested: it pins
    PRISM_ARTIFACTS at a temp root, so any write that escapes the store still misses the real
    ``Artifacts/``.

    Measured 2026-09-24 at 6210a05, run alone: 796.34 s on the CPU (13 min 22 s for the whole
    pytest process). Most of it is the checks: an earlier run cut off at 540 s had by then spent
    466 s on the passive-baseline check ALONE (the record was created at 12:41:35 and that check's
    figure written at 12:49:21). The seed was drawn in that measurement and is fixed since; the
    step count, and so the time, does not depend on it.
    """
    import warnings

    from core import config
    from core.artifacts import ArtifactStore
    from core.tool import main

    root = tmp_path / "store"
    cell = str(config.CELL_PATH / "nadrowski" / "master_spont.txt")
    # A FIXED seed, so a failure thirteen minutes in can be reproduced exactly; the record carries
    # the seed either way (E7), and it is checked below to be this one.
    seed = 20260924
    capsys.readouterr()
    # A sanity check EXCLUDES a probe the spontaneous spectrum does not resolve and says so in a
    # UserWarning (core/FDT/sanity.py); on this cell the passive check measured 18 of its 20 probes
    # (its figure, 2026-09-24). That notice is this path's own output, so it is caught here rather
    # than leaked into the session's warning count. It is not REQUIRED -- whether a probe falls off
    # the grid is physics, and this test is about the path -- but any OTHER warning is a finding, and
    # fails. The filters are inherited, not "always": what is recorded is exactly what would
    # otherwise have reached the session's warnings summary.
    with warnings.catch_warnings(record=True) as said:
        rc = main(["fdt", "--cell", cell, "--n-freqs", "2", "--ensemble-m", "8",
                   "--seed", str(seed), "--store-root", str(root)])
    captured = capsys.readouterr()
    out = captured.out
    assert rc == 0, captured.err[-2000:]
    others = [f"{w.category.__name__}: {w.message}" for w in said
              if "probe frequencies fall outside the Welch PSD grid" not in str(w.message)]
    assert not others, others

    # The two Nadrowski-only checks ran, and the note that announces dropping them did not appear.
    assert "[passive_baseline] true equilibrium (s=0): T_eff/T ~ 1" in out, out[-2000:]
    assert "[high_freq_fdt] high omega (s != 0): T_eff/T ~ 1" in out, out[-2000:]
    assert "Nadrowski-specific and are skipped" not in out
    assert "[PASS] passive_baseline" in out or "[FAIL] passive_baseline" in out, out[-2000:]

    rows = [s for s in ArtifactStore(root).list("fdt") if s.study == "single"]
    assert len(rows) == 1 and rows[0].finished, rows
    rec = ArtifactStore(root).load_fdt(rows[0].id)
    assert rec.body["settings"]["skip_sanity"] is False, rec.body["settings"]
    assert rec.body["seed"] == seed, rec.body["seed"]
    assert rec.body["complete"] is True
    # FIVE figures, not four: the passive-baseline plot is the one --skip-sanity never draws
    # (core/FDT/sanity.py's check_passive_baseline, ``save_plot_path``). Named as well as counted:
    # the count alone would stay green if another figure took its place.
    assert len(rec.manifest.figures) == 5, rec.manifest.figures
    assert "figures/passive_baseline_ratio.png" in rec.manifest.figures, rec.manifest.figures
    for fig in rec.manifest.figures:
        assert (rec.path / fig).exists(), fig


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


def _figure_is_saved_closed_and_silent(draw, out, monkeypatch):
    """The three properties the test above asserts for ``plot_psd``, as one helper the other three
    drawing functions reuse: the file is written, ``plt.show`` is never reached, no "non-interactive"
    warning is emitted, and the figure the call drew is closed again.

    ``plt.show`` is spied on rather than inferred from the absence of the warning: under a backend
    that IS interactive (a display-marked run, or a future matplotlib) the warning would simply not
    appear and the interactive half of the assertion would silently stop testing anything.
    """
    import warnings

    from matplotlib import pyplot as plt

    shown = []
    monkeypatch.setattr(plt, "show", lambda *a, **k: shown.append(True))
    before = len(plt.get_fignums())
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        draw(out)
    assert out.exists(), "save_path was given and nothing was written"
    assert shown == [], "plt.show() was called although save_path was given"
    assert not any("non-interactive" in str(w.message) for w in rec), \
        [str(w.message) for w in rec]
    assert len(plt.get_fignums()) == before, "the saved figure was left open"


def test_plot_eff_temp_ratio_closes_a_saved_figure_instead_of_show(tmp_path, monkeypatch):
    """The headline T_eff/T figure. ``bb22ac4`` gave all four of core/FDT/plots.py's drawing
    functions the close-when-saved branch and only ``plot_psd`` got a test -- the gap
    ``docs/STATE.md``'s piece-5 row hands on and spec section 8.2 closes.

    This one matters most of the four: a real ``fdt`` run draws it once at the end, and on a
    NADROWSKI cell with the sanity checks on the passive-baseline check draws it again
    (core/FDT/sanity.py, ``plot_eff_temp_ratio(... save_path=save_plot_path ...)``) -- so a
    regression here leaks two live figures per such run for the life of the process.
    """
    import numpy as np

    from core.FDT.plots import plot_eff_temp_ratio

    _figure_is_saved_closed_and_silent(
        lambda out: plot_eff_temp_ratio(np.array([1.0, 2.0, 3.0]), np.array([1.0, 1.2, 0.9]),
                                        save_path=out, omega_natural=2.0),
        tmp_path / "ratio.png", monkeypatch)


def test_plot_spontaneous_trajectory_closes_a_saved_figure_instead_of_show(tmp_path, monkeypatch):
    """The Campaign-1 diagnostic trace. Like ``plot_chi_components`` it has no non-finite filter of
    its own, so its save branch is reached on every input; and it is one of the two figures (with the
    spontaneous PSD, which piece 5 moved ahead of Campaign 2) that a run cancelled between the two
    campaigns has already drawn into its record.
    """
    import numpy as np

    from core.FDT.plots import plot_spontaneous_trajectory

    t = np.linspace(0.0, 10.0, 64)
    _figure_is_saved_closed_and_silent(
        lambda out: plot_spontaneous_trajectory(t, np.sin(t) * 0.1, save_path=out, burn_in=2.0),
        tmp_path / "traj.png", monkeypatch)


def test_plot_chi_components_closes_a_saved_figure_instead_of_show(tmp_path, monkeypatch):
    """The two-panel susceptibility figure. It is the one function that builds a MULTI-axes figure
    (``plt.subplots(2, 1, ...)``), so it is the one where "close the figure" and "close the axes"
    could plausibly come apart: ``plt.get_fignums()`` counts figures, and a two-panel figure left
    open counts once just like a one-panel one.
    """
    import numpy as np

    from core.FDT.plots import plot_chi_components

    chis = np.array([1.0 + 1.0j, 2.0 + 0.5j, 1.5 - 0.2j])
    _figure_is_saved_closed_and_silent(
        lambda out: plot_chi_components(np.array([1.0, 2.0, 3.0]), chis,
                                        save_path=out, omega_natural=2.0),
        tmp_path / "chi.png", monkeypatch)


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

    # ...and a run that ENDS IN A REFUSAL removes them too: the ladder's refusal line is printed after
    # the handler call has unwound, and a refusal that left its handlers installed would double every
    # later run's records exactly as a clean one would.
    from core.refusals import Refusal

    def _refused(cfg, ref, build_new, **kw):
        log.info("[budget] said before the refusal")
        raise Refusal("The artifact name is taken.", field="name")

    monkeypatch.setattr(orchestrator, "build_prior", _refused)
    for _ in range(2):
        capsys.readouterr()
        assert main(["prior", *_cfg(bounds)]) == 1
        cap = capsys.readouterr()
        assert cap.out.count("[budget] said before the refusal\n") == 1, cap.out
        assert "prism prior: refused: The artifact name is taken. (--name)" in cap.err, cap.err
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


# ── the artifacts family (piece 4, B9; design §4) ────────────────────────────────────────────────


@pytest.fixture
def browse_store(tmp_path, monkeypatch):
    """A store holding one artifact of each of the eight kinds plus three unusable directories, with
    PRISM_ARTIFACTS pointing at it.

    ``config.artifacts_root()`` reads the variable on EVERY call (core/config.py:185-189) and ``main``
    opens its store on it, so ``main(["artifacts", ...])`` reads exactly this root -- the same redirect
    test_the_sbc_subcommand_forwards_every_knob_as_a_keyword uses. Yields ``(root, ids)``, where
    ``ids`` is ``{kind: id}`` from tests/_fixtures.py::build_browse_store (Task 9's builder: the real
    writer at minimum size, seconds not minutes -- it must never reach for ``tiny_run``)."""
    from tests._fixtures import build_browse_store
    root = tmp_path / "A"
    ids = build_browse_store(root)
    monkeypatch.setenv("PRISM_ARTIFACTS", str(root))
    return root, ids


def test_the_artifacts_listing_shows_the_browsers_own_columns():
    """ONE FORMATTER PER FACT (design §1.2), kept honest across two front ends that cannot share code.
    The browser's ``cells_for``/``columns_for`` live in a Qt module, and importing it from
    ``core/tool/browse.py`` would pull PySide6 into ``python -m core --help`` -- so the tool keeps its
    own column list and the two COLUMN SETS are pinned against each other HERE rather than trusted to
    agree by eye. THIS TEST imports both front ends; the tool imports neither. A column added to the
    table without one added to the tool fails right here.

    The eight kinds are restated in the tool as a literal for the same reason (reading KIND_DIRS at
    parser-build time would import torch), so their order is pinned too -- the same shape as
    test_crossval_preset_choices_match_sweep_presets, which pins --preset's hard-coded choices against
    cli.SWEEP_PRESETS."""
    from core.artifacts.store import KIND_DIRS
    from core.gui.screens import artifact_screen as ascreen
    from core.gui.widgets.artifact_table import columns_for
    from core.tool import browse

    assert browse.KINDS == tuple(KIND_DIRS), "the eight kinds, in KIND_DIRS order"
    assert set(browse.COLUMNS) == set(browse.KINDS), "every kind has a column set, and no other"
    for kind in browse.KINDS:
        # The tool's literal is the GUI's own spelling, lower-cased (P11): one canonical column list,
        # Title-case in the window and lower-case in a terminal where the output may be piped.
        assert browse.COLUMNS[kind] == tuple(c.lower() for c in columns_for(kind)), kind
    # The THIRD deliberate duplication, and the only one that had no pin (tests lens, I2): the
    # sentence that tells an operator a training cache blocks a prior's delete because it was
    # GENERATED against it, rather than because it named it. Its own comment cites COLUMNS above as
    # the precedent for restating it -- so it is pinned the same way.
    assert browse._FINGERPRINT_DEPENDENT == ascreen._FINGERPRINT_DEPENDENT


def test_the_rendered_help_counts_the_kinds_correctly():
    """Checklist 18, a SILENT pin. Five phrases in core/tool/browse.py hand-write how many kinds
    there are -- two "all seven"s in EPILOG, the "<kind> is one of ..." line under them, and the
    ``kind`` help on ``list`` and on ``sweep``. None is generated, none was pinned, and a stale one
    prints a wrong help page to an operator with no failure anywhere.

    The scan is over the RENDERED help and not over the EPILOG constant, deliberately: two of the
    five live on argparse arguments and never appear in that string, so a test that read the
    constant would pass while `python -m core artifacts sweep --help` lied.
    """
    import argparse
    import re
    from core.tool import browse

    words = {6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten"}
    want = words[len(browse.KINDS)]

    p = build_parser().subcommands["artifacts"]
    modes = {name: sub for a in p._actions if isinstance(a, argparse._SubParsersAction)
             for name, sub in a.choices.items()}
    rendered = {"artifacts": p.format_help(),
                **{name: sub.format_help() for name, sub in modes.items()}}
    # Collapsed to single spaces. argparse wraps every help at the TERMINAL's width, and at some
    # widths (74-77 and 124-129 columns among them) it breaks sweep's "all eight" across two lines:
    # uncollapsed, this test fails in such a console for nothing, and a stale count broken the same
    # way would slip past the scan below.
    rendered = {name: " ".join(text.split()) for name, text in rendered.items()}

    for name, text in rendered.items():
        for found in re.findall(r"\ball (\w+)", text):
            assert found == want, f"{name} --help says 'all {found}' for {len(browse.KINDS)} kinds"
    assert f"all {want}" in rendered["artifacts"], "the epilog names the count at all"
    assert f"all {want}" in rendered["sweep"], "sweep's --help names the count at all"
    assert "<kind> is one of " + ", ".join(browse.KINDS) in rendered["artifacts"], \
        "the epilog lists the kinds by name, in KINDS order"


def test_artifacts_list_prints_one_line_per_artifact_with_its_kinds_facts(browse_store, capsys):
    """design §3.2's table, per kind: a posterior's mode, width and amortization; a cache's progress
    as a FRACTION (``batches_done``/``batches_planned``) and -- separately -- whether it FINISHED (B3:
    ``complete`` means "has a valid manifest", which is true of a cache from its first batch on). Read
    off the store's own rows rather than off literal names, so the assertions hold whatever
    build_browse_store names its artifacts."""
    from core.artifacts import ArtifactStore
    root, ids = browse_store
    s = ArtifactStore(root)

    capsys.readouterr()
    assert main(["artifacts", "list", "posterior"]) == 0
    out = capsys.readouterr().out
    assert "== posterior ==" in out, out
    post = next(r for r in s.list("posterior") if r.complete)
    line = next(ln for ln in out.splitlines() if post.label in ln)
    assert post.created in line, line
    assert str(post.width) in line and (post.mode or "") in line, line
    assert ("amortized" if post.amortized else "narrowed (TSNPE)") in line, line

    capsys.readouterr()
    assert main(["artifacts", "list", "simulation"]) == 0
    out = capsys.readouterr().out
    sim = next(r for r in s.list("simulation") if r.complete)
    line = next(ln for ln in out.splitlines() if sim.label in ln)
    assert f"{sim.batches_done}/{sim.batches_planned} batches" in line, line
    if sim.finished:
        assert f", {sum(sim.rows)} rows" in line, line
    else:
        assert "rows" not in line, "``save`` passes no rows, so a cache mid-run has none to show"
    assert ("yes" if sim.finished else "no") in line, line


def test_the_artifacts_listing_progress_cell_is_a_fraction_and_shows_rows_only_once_finished():
    """Both halves of the progress cell, on stand-in summaries: ``build_browse_store``'s cache is an
    unfinished one, so the FINISHED half has no fixture to come from: ``batches_planned`` (the body's
    ``identity["n_runs"]``, spec §12 row 2) makes the cell a FRACTION, and ``rows`` is written only by
    ``mark_complete`` -- ``save`` passes none -- so a cache mid-run shows its batches and no row
    count. A comma and plain ASCII, never the browser's middle dot: this is text a script may read."""
    from core.artifacts.store import Summary
    from core.tool import browse

    def summary(**kw):
        return Summary(kind="simulation", id="s", name="", created="2026-01-01 00:00:00", note="",
                       path=Path("."), complete=True, reason=None, **kw)

    assert browse._progress(summary(batches_done=3, batches_planned=4, rows=None)) == "3/4 batches"
    assert browse._progress(summary(batches_done=4, batches_planned=4, finished=True,
                                    rows=(48, 48))) == "4/4 batches, 96 rows"
    assert browse._progress(summary()) == "?/? batches", "a body with neither count still renders"


def test_the_artifacts_listing_points_cell_is_plain_ascii_and_a_dash_when_there_are_none():
    """The tool's twin of the browser's Points cell, on stand-in summaries: a comma and plain ASCII,
    never the browser's middle dot, because a script may read this line; "-" where a single run or a
    comparison has no operating points at all, as a diagnostic's missing variant already reads."""
    from core.artifacts.store import Summary
    from core.tool import browse

    def summary(**kw):
        return Summary(kind="fdt", id="f", name="", created="2026-01-01 00:00:00", note="",
                       path=Path("."), complete=True, reason=None, **kw)

    assert browse._fdt_points(summary(points_done=10, points_planned=12, points_failed=0)) == "10/12"
    assert browse._fdt_points(summary(points_done=10, points_planned=12, points_failed=2)) == \
        "10/12, 2 failed"
    assert browse._fdt_points(summary(points_done=10)) == "10/?", "a body with no planned total"
    assert browse._fdt_points(summary()) == "-"
    assert browse._cells("fdt", summary(study="sweep", points_done=2, points_planned=3,
                                        points_failed=1, finished=True)) == \
        ("(unnamed f)", "2026-01-01 00:00:00", "sweep", "2/3, 1 failed", "yes", "")
    assert browse._cells("fdt", summary())[2:5] == ("-", "-", "no")


def test_artifacts_list_with_no_kind_covers_every_kind_under_a_heading(browse_store, capsys):
    root, ids = browse_store
    capsys.readouterr()
    assert main(["artifacts", "list"]) == 0
    out = capsys.readouterr().out
    for kind in ("prior", "simulation", "posterior", "observation", "calibration", "inference",
                 "diagnostic"):
        assert f"== {kind} ==" in out, kind
    assert "nothing in:" not in out, "build_browse_store writes one artifact of every kind"


def test_an_empty_artifacts_listing_exits_0_and_a_bad_kind_exits_1(tmp_path, monkeypatch, capsys):
    """§4.3: an empty listing is 0, with a line saying there is nothing there. A script must be able to
    tell "nothing on disk" from "you asked for something wrong", and exiting 1 on an empty store would
    make the two indistinguishable. A kind that does not exist IS the second case: the store's own
    refusal, through the ladder's ``refused:`` rung at exit 1.

    The empty root need not exist beforehand -- ``main`` mkdirs the store root before any handler runs
    (core/tool/__init__.py:103)."""
    monkeypatch.setenv("PRISM_ARTIFACTS", str(tmp_path / "empty"))

    capsys.readouterr()
    assert main(["artifacts", "list"]) == 0
    out = capsys.readouterr().out
    named_empty = out.split("nothing in:")[1]
    for kind in ("prior", "simulation", "posterior", "observation", "calibration", "inference",
                 "diagnostic"):
        assert kind in named_empty, kind
    assert "holds no artifacts yet" in out, out

    capsys.readouterr()
    assert main(["artifacts", "list", "prior"]) == 0
    out = capsys.readouterr().out
    assert "== prior ==" in out and "nothing here yet" in out, out

    capsys.readouterr()
    assert main(["artifacts", "list", "nosuchkind"]) == 1
    err = capsys.readouterr().err
    lines = [ln for ln in err.splitlines() if ln.startswith("prism artifacts: refused:")]
    assert len(lines) == 1 and "unknown artifact kind" in lines[0], err
    assert "Traceback" not in err, err


def test_artifacts_list_puts_an_unusable_directory_last_with_its_reason(browse_store, capsys):
    """A leftover directory is the first thing an operator can act on that no front end has ever shown:
    ``get``/``path`` and the GUI picker all skip it. ``list`` already sorts incomplete rows last, and
    the row carries the ``dir_name`` because that is the only handle ``sweep`` can take (B7).

    NOTE on ordering among several leftovers: ``browse_store`` (build_browse_store) already seeds three
    ``leftover_*`` directories under ``priors/`` for the store shapes it covers, so this test's own
    leftover joins a total of four incomplete rows. ``ArtifactStore._entries`` sorts leftovers by plain
    directory name (test_artifact_store.py::test_remove_incomplete_is_not_defeated_by_a_degenerate_
    inode_volume relies on exactly this to put "aaa_leftover" before a real artifact), so "_unnamed__..."
    (an underscore, ASCII 95) sorts BEFORE "leftover_..." (ASCII 108) -- it is not guaranteed to be the
    last of several incomplete rows. What IS guaranteed, and what this asserts, is the property the
    store and ``_print_kind`` actually promise: every incomplete row comes after every complete one."""
    root, ids = browse_store
    leftover = root / "priors" / "_unnamed__20260101T000000"
    leftover.mkdir(parents=True, exist_ok=True)

    capsys.readouterr()
    assert main(["artifacts", "list", "prior"]) == 0
    lines = [ln for ln in capsys.readouterr().out.splitlines() if ln.strip()]
    hits = [i for i, ln in enumerate(lines) if "_unnamed__20260101T000000" in ln]
    assert len(hits) == 1, lines
    assert "no manifest.json" in lines[hits[0]], lines[hits[0]]
    assert lines[hits[0]].strip().startswith("incomplete"), lines[hits[0]]
    good_line = next(i for i, ln in enumerate(lines) if "browse_prior" in ln)
    assert hits[0] > good_line, f"an incomplete directory is listed after every complete row: {lines}"


def test_artifacts_show_prints_the_manifest_then_the_runs_records(browse_store, capsys, monkeypatch):
    """The manifest through the ONE renderer both front ends use (design §5), then the records with
    their ``HH:MM:SS level`` stamps, then a truncation notice when the tail was cut. Asserting the
    renderer's own output is IN the printed text is what keeps a second, drifting renderer from being
    written here."""
    from core.artifacts import ArtifactStore, render_manifest
    from core.artifacts.store import LOG_FILE
    from core.tool import browse
    root, ids = browse_store
    s = ArtifactStore(root)
    (s.path("prior", ids["prior"]) / LOG_FILE).write_text("10:00:00 info the sweep began\n",
                                                          encoding="utf-8")

    capsys.readouterr()
    assert main(["artifacts", "show", "prior", ids["prior"]]) == 0
    out = capsys.readouterr().out
    assert render_manifest(s.get("prior", ids["prior"])) in out, "the one renderer, not a second one"
    assert "== records ==" in out and "10:00:00 info the sweep began" in out, out

    # the tail, and the notice that says it is one. The cap is monkeypatched rather than met with a
    # megabyte of log: what is under test is the notice, not the byte count.
    monkeypatch.setattr(browse, "LOG_TAIL_BYTES", 40)
    (s.path("prior", ids["prior"]) / LOG_FILE).write_text("A" * 200 + "\n10:00:01 info the last line\n",
                                                          encoding="utf-8")
    capsys.readouterr()
    assert main(["artifacts", "show", "prior", ids["prior"]]) == 0
    out = capsys.readouterr().out
    assert "the log is longer" in out and "10:00:01 info the last line" in out, out


def test_artifacts_show_states_the_two_honest_gaps(browse_store, capsys):
    """design §3.3's two sentences, WORD FOR WORD the browser's, because one operator reads both. A
    cache has no log by design (it is written batch by batch across resumes and shared by every
    posterior that names it), and an artifact written with no run active has none either -- and
    ``read_log`` answers ``(None, False)`` for both, so the KIND is what tells them apart."""
    from core.artifacts import ArtifactStore
    from core.artifacts.store import LOG_FILE
    root, ids = browse_store
    s = ArtifactStore(root)

    capsys.readouterr()
    assert main(["artifacts", "show", "simulation", ids["simulation"]]) == 0
    out = capsys.readouterr().out
    assert "a training cache keeps no log" in out and "across resumes" in out, out

    (s.path("observation", ids["observation"]) / LOG_FILE).unlink(missing_ok=True)
    capsys.readouterr()
    assert main(["artifacts", "show", "observation", ids["observation"]]) == 0
    assert "written outside a run" in capsys.readouterr().out


def test_artifacts_show_names_a_directory_the_manifest_disagrees_with(browse_store, capsys):
    """``rename`` writes the manifest first and tolerates a refused directory move (the manifest is
    what resolves an artifact), which leaves a folder whose name disagrees with ``Manifest.dir_name``.
    ``show`` is the first place that is visible (design §3.3, §2.6)."""
    from core.artifacts import ArtifactStore
    root, ids = browse_store
    sub = ArtifactStore(root).path("calibration", ids["calibration"])
    sub.rename(sub.with_name("stale__" + sub.name.rsplit("__", 1)[-1]))

    capsys.readouterr()
    assert main(["artifacts", "show", "calibration", ids["calibration"]]) == 0
    out = capsys.readouterr().out
    assert "stale__" in out and "while its manifest says" in out, out


def test_artifacts_show_on_a_missing_ref_exits_1_through_the_refused_rung(browse_store, capsys):
    """§4.3: a missing or ambiguous ref is 1, through the existing ``refused:`` rung. The message
    already quotes the ref, which is exactly why core/tool/fields.py maps the ``artifact`` key to
    None -- the tool names the artifact positionally, so there is no option string to print, and
    fix_sentence owns the parentheses, so the line simply ends at the message."""
    root, ids = browse_store
    capsys.readouterr()
    assert main(["artifacts", "show", "prior", "nosuch"]) == 1
    err = capsys.readouterr().err
    lines = [ln for ln in err.splitlines() if ln.startswith("prism artifacts: refused:")]
    assert len(lines) == 1, err
    assert "nosuch" in lines[0] and "no complete prior artifact" in lines[0], lines[0]
    assert not lines[0].endswith("()"), lines[0]
    assert "Traceback" not in err and "raised at" not in err, err


def test_the_artifacts_family_takes_no_configuration_flags_and_its_help_costs_no_torch():
    """B9's two halves, pinned. (1) No configuration flags anywhere in the family: a listing must not
    be able to fail on a bounds file it does not need, and --store-root in particular is absent
    because ``main`` keys smoke's temp-root behaviour, its empty-root cleanup and its Ctrl-C advice on
    ``hasattr(args, "store_root")`` (core/tool/__init__.py:95-98, 121-122, 156-157). (2) ``--help``
    for the family imports no torch -- checked in a FRESH interpreter, because in this process torch
    is long since imported by the session fixtures, so a sys.modules check here would pass
    vacuously (the pattern of test_every_field_key_has_a_flag_..., leg (d))."""
    import argparse
    import subprocess
    import sys

    p = build_parser().subcommands["artifacts"]
    modes = {name: sub for a in p._actions if isinstance(a, argparse._SubParsersAction)
             for name, sub in a.choices.items()}
    assert set(modes) >= {"list", "show"}, sorted(modes)
    for name, parser in [("artifacts", p), *sorted(modes.items())]:
        for flag in ("--bounds", "--model", "--chi", "--chi-k", "--device", "--store-root",
                     "--accept-truncated"):
            assert flag not in parser._option_string_actions, (name, flag)

    probe = ("import sys\n"
             "from core.tool import main\n"
             "rc = main(['artifacts', '--help'])\n"
             "bad = sorted(m for m in sys.modules if m == 'torch' or m.startswith('torch.'))\n"
             "sys.exit(0 if rc == 0 and not bad else repr((rc, bad[:3])))\n")
    r = subprocess.run([sys.executable, "-c", probe], cwd=str(config.REPO_ROOT),
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stdout + r.stderr


def test_artifacts_note_sets_trims_and_clears_a_note(browse_store, capsys):
    """B5: the note is one line, trimmed, and a blank one CLEARS it. ``--note`` is a flag rather than
    a positional precisely so core/tool/fields.py can map the ``note`` key to a real option string --
    the table's values are pinned against build_parser()'s own option strings, both ways."""
    from core.artifacts import ArtifactStore
    root, ids = browse_store
    s = ArtifactStore(root)

    capsys.readouterr()
    assert main(["artifacts", "note", "prior", ids["prior"], "--note", "  the first fit  "]) == 0
    assert s.get("prior", ids["prior"]).note == "the first fit", "require_note trims, and only trims"
    assert "the first fit" in capsys.readouterr().out

    assert main(["artifacts", "note", "prior", ids["prior"], "--note", ""]) == 0
    assert s.get("prior", ids["prior"]).note == "", "a blank note clears it"


def test_a_note_that_breaks_the_rule_is_refused_naming_the_flag(browse_store, capsys):
    """V2 on the note: no clamp and no silent default. An over-long note is refused with BOTH numbers
    in the sentence, a newline is refused rather than flattened, and the manifest is untouched in
    either case. The flag comes from core/tool/fields.py's table, appended by main's ladder -- and
    only for a refusal about the note TEXT, which is ``require_note``'s (field="note"). The third leg
    is the other refusal this mode can raise: ``set_note``'s own "no complete artifact", which carries
    field="artifact" (spec §12 row 1), a key whose flag is None -- so that line ends at the message."""
    from core.artifacts import ArtifactStore
    from core.refusals import NOTE_MAX_CHARS
    root, ids = browse_store
    s = ArtifactStore(root)
    before = s.get("prior", ids["prior"]).note

    capsys.readouterr()
    assert main(["artifacts", "note", "prior", ids["prior"],
                 "--note", "a" * (NOTE_MAX_CHARS + 1)]) == 1
    err = capsys.readouterr().err
    lines = [ln for ln in err.splitlines() if ln.startswith("prism artifacts: refused:")]
    assert len(lines) == 1 and lines[0].endswith("(--note)"), err
    assert str(NOTE_MAX_CHARS) in lines[0] and str(NOTE_MAX_CHARS + 1) in lines[0], lines[0]
    assert "raised at" not in err and "Traceback" not in err, err

    capsys.readouterr()
    assert main(["artifacts", "note", "prior", ids["prior"], "--note", "a\nb"]) == 1
    assert "(--note)" in capsys.readouterr().err
    assert s.get("prior", ids["prior"]).note == before, "a refused note changed the manifest"

    # A ref that names no artifact is a DIFFERENT refusal: set_note's, with field="artifact" (spec
    # §12 row 1), whose entry in core/tool/fields.py is None because the tool names the artifact
    # positionally. fix_sentence owns the parentheses, so the line simply ends at the message -- no
    # "(--note)" here, because the note is not what is wrong.
    capsys.readouterr()
    assert main(["artifacts", "note", "prior", "nosuch", "--note", "a fine note"]) == 1
    err = capsys.readouterr().err
    lines = [ln for ln in err.splitlines() if ln.startswith("prism artifacts: refused:")]
    assert len(lines) == 1 and "nosuch" in lines[0], err
    assert not lines[0].rstrip().endswith(")"), f"no trailing parenthetical: {lines[0]}"
    assert "--note" not in lines[0], lines[0]


def test_artifacts_rm_offers_no_force_and_deletes_one_artifact(browse_store, capsys):
    """B6: delete is one artifact at a time and there is NO force in either front end, so the store's
    refusal is the last word. A diagnostic is the leaf case -- ``_PARENT_KEYS["diagnostic"]`` is
    empty, so nothing can ever name one as a parent."""
    import argparse
    from core.artifacts import ArtifactStore, StoreError
    root, ids = browse_store
    modes = {name: sub for a in build_parser().subcommands["artifacts"]._actions
             if isinstance(a, argparse._SubParsersAction) for name, sub in a.choices.items()}
    assert "--force" not in modes["rm"]._option_string_actions, "B6: no force in either front end"

    s = ArtifactStore(root)
    gone = s.path("diagnostic", ids["diagnostic"])
    capsys.readouterr()
    assert main(["artifacts", "rm", "diagnostic", ids["diagnostic"]]) == 0
    assert str(gone) in capsys.readouterr().out
    assert not gone.exists()
    with pytest.raises(StoreError):
        s.get("diagnostic", ids["diagnostic"])


def test_artifacts_rm_refuses_an_artifact_something_depends_on(tool_run, capsys):
    """The other half of B6: because no force is offered, an artifact anything depends on cannot be
    deleted AT ALL, and the refusal names every dependent -- in the TOOL's OWN WORDS (fix round 1,
    IMPORTANT 1), never the store's raw sentence, which ends "pass force=True to orphan them": a step
    nothing in either front end offers, so it must never reach an operator.

    Fix round 2: ``tool_run`` is MODULE-scoped, so every test in this file shares one store, and
    ``tp`` -- the one prior every SBI test in this module trains against -- accumulates a dependent
    for every posterior, calibration and cache an EARLIER-running test in this module built against
    it (``test_validate_and_simulated_infer``'s ``tcal`` among them). A hard-coded "2 artifact(s)"
    was true only when this test happened to run first, or alone -- so the expected COUNT and the
    expected IDS are read off ``s.dependents`` itself, which is the one ground truth that holds no
    matter what the rest of the module has built by the time this runs.

    Whole-piece review (tests lens, I2): that round-2 narrowing dropped the assertion on the REASON
    clause altogether -- only the ids were checked, so deleting ``-- {why}`` from the f-string would
    leave the suite green and the operator without the one sentence that says WHY a training cache
    blocks a prior's delete. Re-pinned here order-independently: every dependent's id is followed by
    a reason clause, and the reasons are the two the front ends agree on, COUNTED rather than
    sequenced (which dependent gets which is ``dependents()``'s business, not this test's)."""
    from core.artifacts import ArtifactStore
    from core.tool import browse
    bounds, cell, root = tool_run
    s = ArtifactStore(root)
    before = s.get("prior", "tp").id
    deps = s.dependents("prior", before)
    assert deps, "tool_run's tp is trained against and always has at least one dependent"

    capsys.readouterr()
    assert main(["artifacts", "rm", "prior", "tp"]) == 1
    err = capsys.readouterr().err
    lines = [ln for ln in err.splitlines() if ln.startswith("prism artifacts: refused:")]
    assert len(lines) == 1, err
    line = lines[0]
    assert "refusing to delete prior tp" in line, line
    assert f"{len(deps)} artifact(s) depend on it" in line, line
    for dep_kind, dep_id, _dep_name in deps:
        assert f"[{dep_id}]" in line, (dep_kind, dep_id, line)
        after = line.split(f"[{dep_id}]", 1)[1].lstrip()
        assert after.startswith("-- "), (dep_id, after[:120])
    reasons = (browse._FINGERPRINT_DEPENDENT, "it names this prior as a parent")
    assert sum(line.count(r) for r in reasons) == len(deps), (reasons, line)
    assert line.count(" -- ") == len(deps), line
    assert "Delete those first." in line, line
    assert "force" not in line.lower(), line
    assert s.get("prior", "tp").id == before, "a refusal removed something"


def test_artifacts_sweep_is_a_dry_run_until_yes_and_removes_only_manifest_less_directories(
        browse_store, capsys):
    """B7 with R1 and R4. One action removes every directory of a kind that has NO manifest at all,
    through a store call that CAN ONLY remove such a directory -- ``delete`` resolves through
    ``_find``, which never returns a manifest-less entry, and ``remove_incomplete`` is its
    complement. §4.3: nothing to remove is exit 0, and it says so rather than printing nothing.

    R4: the window ASKS before it removes and this did not -- no preview, no confirmation, and a
    reviewer's probe deleted two directories and printed their reasons afterwards. So the default is
    a DRY RUN: exactly what it would remove, removing nothing, and ``--yes`` performs it. That is a
    confirmation and not an override: B6 stands, there is still no ``--force`` and no way to reach a
    real artifact from here.

    R1: of the three shapes ``build_browse_store`` seeds under ``priors/`` (no manifest at all, a
    manifest that will not parse, a valid manifest of another kind), only the FIRST is a leftover a
    sweep may remove. The other two are reported and survive -- a manifest valid under a different
    SCHEMA reads as "no artifact here" to this build, and deleting one cost a reviewer's probe a real
    calibration with its payload."""
    from tests._fixtures import backdate_tree
    root, ids = browse_store
    leftover = root / "priors" / "_unnamed__20260101T000000"
    leftover.mkdir(parents=True, exist_ok=True)
    manifest_less = root / "priors" / ids["bad"][0]
    carries_a_manifest = [root / "priors" / n for n in ids["bad"][1:]]
    bad_names = set(ids["bad"]) | {leftover.name}
    kept = [p for p in (root / "priors").iterdir() if p.name not in bad_names]
    assert kept, "build_browse_store wrote a prior"
    backdate_tree(root / "priors")          # R3's guard is tested at the store; not the subject here

    # (a) the DRY RUN: says what it would remove, removes nothing, exits 0
    capsys.readouterr()
    assert main(["artifacts", "sweep", "prior"]) == 0
    out = capsys.readouterr().out
    assert "would remove" in out and leftover.name in out and manifest_less.name in out, out
    assert "--yes" in out, "the dry run must name the flag that performs it"
    assert "still being written" in out, \
        ("a preview run beside a live training would otherwise offer that training's own directory "
         "with no warning: the listing cannot tell a run in flight from a leftover (R3)")
    assert leftover.is_dir() and manifest_less.is_dir(), "a dry run removed something"

    # (b) --yes performs it, and only the manifest-less directories go
    capsys.readouterr()
    assert main(["artifacts", "sweep", "prior", "--yes"]) == 0
    out = capsys.readouterr().out
    assert "removed" in out and leftover.name in out and manifest_less.name in out, out
    assert not leftover.exists() and not manifest_less.exists()
    for d in carries_a_manifest:
        assert d.is_dir(), f"{d.name} carries a manifest.json and must never be swept"
        assert d.name in out, out
    assert all(p.exists() for p in kept), "sweep removed an artifact with a usable manifest"
    # RULED IN 5 (fix round 1): sweep_incomplete's own return has no reason, so the tool reads it off
    # `list` before removing and prints it after -- an operator must be able to tell "no manifest"
    # from "a manifest of another kind" after the fact, not just which directory went.
    assert "no manifest.json" in out, out
    assert "manifest declares kind" in out, out

    capsys.readouterr()
    assert main(["artifacts", "sweep", "prior", "--yes"]) == 0
    assert "nothing to sweep" in capsys.readouterr().out


def test_artifacts_sweep_offers_a_loose_file_and_never_a_records_payload(browse_store, capsys):
    """E10, first category. ``ArtifactStore._entries`` iterates DIRECTORIES only, so a file sitting
    directly inside a kind directory is invisible to every listing in both front ends -- and the
    owner's machine has two of them, the PNGs a pre-piece-5 fdt run left in ``Artifacts/fdt``. They
    carry no record of which cell or which settings produced them and no command could reach them.

    The complement matters as much as the category: ``loose_files`` reads the kind directory's own
    files and never descends, so a real artifact's payload -- which lives one level down, inside the
    record's folder -- is not offerable from here even in principle. The payload written below is
    what asserts that, and it must still be on disk after ``--yes``.
    """
    from tests._fixtures import backdate_tree
    root, ids = browse_store
    stray = root / "priors" / "fdt_ratio_20260915_153042.png"
    stray.write_bytes(b"a picture an older build left beside the records")
    record = next(p for p in (root / "priors").iterdir() if p.is_dir() and ids["prior"] in p.name)
    payload = record / "prior.pt"
    payload.write_bytes(b"a real artifact's payload")
    backdate_tree(root / "priors")           # the recency guard is tested on its own, below

    capsys.readouterr()
    assert main(["artifacts", "sweep", "prior"]) == 0
    out = capsys.readouterr().out
    assert f"would remove prior loose file {stray.name}" in out, out
    assert "prior.pt" not in out, "a valid record's payload was offered for removal"
    assert stray.is_file(), "a dry run removed something"

    capsys.readouterr()
    assert main(["artifacts", "sweep", "prior", "--yes"]) == 0
    out = capsys.readouterr().out
    assert f"removed prior loose file {stray.name}" in out, out
    assert not stray.exists()
    assert payload.is_file(), "the sweep reached inside a valid record's own folder"
    assert (record / "manifest.json").is_file(), "the record itself survived its neighbour's removal"


def test_a_legacy_directory_is_offered_by_the_all_kinds_sweep_only(browse_store, capsys):
    """E10's second category, and why it has a form of its own. A legacy directory -- a `crossval/`
    an older build wrote -- sits BESIDE the kind directories, under no kind, and the store never
    walks its own root, so it is invisible to every listing. ``sweep`` takes the kind POSITIONALLY
    (core/tool/browse.py's `sweep [<kind>]`), so only the form with no kind can offer something that
    belongs to none: a per-kind sweep that removed it would be answering a question nobody asked.

    And a per-kind sweep that finds nothing must not SAY there is no legacy directory either: it never
    read the store root, so that clause would claim what nobody checked -- with one sitting right
    there.
    """
    from core.artifacts.store import LEGACY_DIRS
    from tests._fixtures import backdate_tree
    root, ids = browse_store
    legacy = root / LEGACY_DIRS[0]
    legacy.mkdir()
    (legacy / "fdt3d_vs_S_20260915_153042.h5").write_bytes(b"an older build's sweep output")
    backdate_tree(root)

    capsys.readouterr()
    assert main(["artifacts", "sweep", "prior"]) == 0
    assert legacy.name not in capsys.readouterr().out, \
        "a per-kind sweep offered a directory that belongs to no kind"
    assert legacy.is_dir()

    capsys.readouterr()
    assert main(["artifacts", "sweep", "calibration"]) == 0
    out = capsys.readouterr().out
    assert "nothing to sweep" in out and "legacy" not in out, out

    # an EMPTY kind is no kind -- not a second spelling of the all-kinds form that skips the legacy read
    capsys.readouterr()
    assert main(["artifacts", "sweep", ""]) == 1
    assert "unknown artifact kind" in capsys.readouterr().err
    assert legacy.is_dir()

    capsys.readouterr()
    assert main(["artifacts", "sweep"]) == 0
    out = capsys.readouterr().out
    assert f"would remove legacy directory {legacy.name}" in out, out
    assert legacy.is_dir(), "a dry run removed something"

    capsys.readouterr()
    assert main(["artifacts", "sweep", "--yes"]) == 0
    assert f"removed legacy directory {legacy.name}" in capsys.readouterr().out
    assert not legacy.exists()


def test_a_loose_file_written_seconds_ago_is_refused_rather_than_swept(browse_store, capsys):
    """R3's guard, extended to the new categories (spec §6.3: "the recency guard applies"). A file
    inside a kind directory can be a run in flight writing its own figure as easily as it can be a
    leftover -- and a sweep from a second process has no ``BasePanel._running`` to consult. The
    removal refuses it, the tool reports it, and the exit code says the sweep did not do everything
    it offered.
    """
    from tests._fixtures import backdate_tree
    root, ids = browse_store
    backdate_tree(root / "priors")            # every leftover directory is old; only the file is new
    fresh = root / "priors" / "being_written_right_now.png"
    fresh.write_bytes(b"a run may still be writing this")

    capsys.readouterr()
    assert main(["artifacts", "sweep", "prior", "--yes"]) == 1
    cap = capsys.readouterr()
    assert fresh.is_file(), "a file written seconds ago was removed"
    assert f"could not remove prior loose file {fresh.name}" in cap.err, cap.err


def test_an_unfinished_fdt_record_is_listed_as_not_finished_and_never_swept(browse_store, capsys):
    """Controller ruling F7, the tool half of spec §8.2's "the leftover sweep never offers it" (the
    store half is tests/test_artifact_store.py's
    test_a_cancel_between_the_first_manifest_and_the_first_payload_keeps_the_record). E2 keeps an
    interrupted fdt record's folder with what it measured inside, and that folder carries a manifest
    from its first moment -- so it is an ARTIFACT, listed and honestly marked unfinished, never a
    leftover. Now that the sweep also reaches FILES inside a kind directory, the record's own payload
    is what this pins: it lives one level down, inside the record's folder, where ``loose_files``
    never looks.

    Backdated past the recency guard, so a sweep that did offer the record would also REMOVE it under
    --yes rather than be saved by the age check.
    """
    from core.artifacts import ArtifactStore
    from tests._fixtures import backdate_tree
    root, ids = browse_store
    w = ArtifactStore(root).create("fdt", None, name="stopped_fdt", note="interrupted")
    w.body = {"study": "single", "settings": {}, "seed": 3, "notices": []}
    with pytest.raises(RuntimeError):
        with w:
            w.payload("data.h5").write_bytes(b"the spontaneous spectrum, and nothing after it")
            raise RuntimeError("stopped")
    backdate_tree(root / "fdt")

    capsys.readouterr()
    assert main(["artifacts", "list", "fdt"]) == 0
    lines = capsys.readouterr().out.splitlines()
    header = next(ln for ln in lines if ln.split()[:1] == ["name"])
    start, end = header.index("finished"), header.index("note")   # the table is column-aligned

    def finished_cell(label):
        return next(ln for ln in lines if label in ln)[start:end].strip()

    assert finished_cell("stopped_fdt") == "no", lines
    assert finished_cell("browse_fdt") == "yes", "the fixture's finished record, for contrast"

    for argv in (["artifacts", "sweep", "fdt"], ["artifacts", "sweep", "fdt", "--yes"]):
        capsys.readouterr()
        assert main(argv) == 0
        cap = capsys.readouterr()
        assert "nothing to sweep" in cap.out, cap.out
        assert w.dir.name not in cap.out + cap.err and "data.h5" not in cap.out + cap.err, cap
    assert (w.dir / "data.h5").is_file() and (w.dir / "manifest.json").is_file(), \
        "an unfinished record, or the measurement inside it, was swept"


def test_a_sweep_that_could_not_remove_something_exits_1_naming_it(browse_store, monkeypatch, capsys):
    """§4.3: a sweep that removed everything is 0; one that could not remove a directory is 1, naming
    each failure. A held handle on Windows is what that stands for and it cannot be provoked on
    demand, so the two lists are injected on the store's own method -- what is under test is the
    ladder, not the removal.

    The injection also pins the CONTRACT R2 gave that method: it is handed the list of
    ``(kind, dir_name)`` the operator was shown, not a kind to re-scan."""
    from core.artifacts import ArtifactStore
    root, ids = browse_store
    seen = {}

    def _fake(self, entries):
        seen["entries"] = list(entries)
        return ([("prior", "gone__1")], [("posterior", "stuck__2", "PermissionError: in use")])

    monkeypatch.setattr(ArtifactStore, "sweep_incomplete", _fake)
    capsys.readouterr()
    assert main(["artifacts", "sweep", "--yes"]) == 1
    cap = capsys.readouterr()
    assert "gone__1" in cap.out, cap.out
    assert "stuck__2" in cap.err and "in use" in cap.err, cap.err
    assert ("prior", ids["bad"][0]) in seen["entries"], seen["entries"]
    assert all(len(e) == 2 for e in seen["entries"]), seen["entries"]
    assert not any(d in [e[1] for e in seen["entries"]] for d in ids["bad"][1:]), \
        "a directory that carries a manifest.json was handed to the removal"


def test_artifacts_summary_prints_the_lineage_or_writes_it_to_out(browse_store, tmp_path, capsys):
    """B10: the lineage report is a FILE, never a store kind of its own, and it comes out of the same
    renderer the browser's "Lineage report..." writes -- so the document a reviewer receives is the
    same whichever front end made it (design §5). ``--out`` writes exactly what stdout would have
    carried.

    The file is compared as BYTES, not as text: read_text would translate CRLF back to LF on the way
    in and pass whatever newline=None had written, which is precisely the drift §5 forbids and the
    GUI's own report test pins the same way (P23)."""
    from core.artifacts import ArtifactStore, render_lineage
    root, ids = browse_store
    want = render_lineage(ArtifactStore(root), "posterior", ids["posterior"])

    capsys.readouterr()
    assert main(["artifacts", "summary", "posterior", ids["posterior"]]) == 0
    assert want in capsys.readouterr().out, "the one renderer, not a second one"

    out_file = tmp_path / "lineage.txt"
    assert main(["artifacts", "summary", "posterior", ids["posterior"], "--out", str(out_file)]) == 0
    assert out_file.read_bytes() == want.encode("utf-8"), \
        "byte for byte what the browser's Lineage report writes: UTF-8, LF endings"
    assert str(out_file) in capsys.readouterr().out, "the path is named, as report() names an artifact's"


def test_the_artifacts_family_keeps_the_ladders_exit_codes(browse_store, capsys):
    """§4.3, one line per rung: a usage error is 2 (argparse, one rung earlier than every refusal); a
    missing ref, a bad kind and a bad note are 1; an empty listing and a sweep with nothing to remove
    are 0. ``--note`` being REQUIRED is part of this: a note must never be cleared by omission.

    There is no exit 3 anywhere in the tool and this family adds none: a BUG is the existing unhandled
    rung, which sets 1 with a traceback, so 1 covers a refusal and a bug alike (``main``'s own
    docstring: "1 a refusal or a bug") and no task here touches the ladder."""
    root, ids = browse_store
    assert main(["artifacts"]) == 2, "a mode is required"
    assert main(["artifacts", "nosuchmode"]) == 2
    assert main(["artifacts", "show", "prior"]) == 2, "<ref> is required"
    assert main(["artifacts", "note", "prior", ids["prior"]]) == 2, "--note is required"
    assert main(["artifacts", "rm", "prior", "nosuch"]) == 1
    assert main(["artifacts", "summary", "prior", "nosuch"]) == 1

    capsys.readouterr()
    assert main(["artifacts", "sweep", "nosuchkind"]) == 1
    err = capsys.readouterr().err
    lines = [ln for ln in err.splitlines() if ln.startswith("prism artifacts: refused:")]
    assert len(lines) == 1 and "nosuchkind" in lines[0], err
    assert not lines[0].rstrip().endswith(")"), \
        "the artifact key has no flag, so fix_sentence adds nothing and the line ends at the message"


def test_the_artifacts_family_has_all_six_modes_and_still_no_configuration_flags():
    """The closure of B9's list, and the extension of Task 13's own pin to the four modes that write:
    six modes, no configuration flag on any of them, and --note exactly where core/tool/fields.py
    says it is (that table is pinned against these very option strings, both ways)."""
    import argparse
    p = build_parser().subcommands["artifacts"]
    modes = {name: sub for a in p._actions if isinstance(a, argparse._SubParsersAction)
             for name, sub in a.choices.items()}
    assert set(modes) == {"list", "show", "note", "rm", "sweep", "summary"}, sorted(modes)
    for name, parser in [("artifacts", p), *sorted(modes.items())]:
        for flag in ("--bounds", "--model", "--chi", "--chi-k", "--device", "--store-root",
                     "--force", "--accept-truncated"):
            assert flag not in parser._option_string_actions, (name, flag)
    assert "--note" in modes["note"]._option_string_actions
    assert "--out" in modes["summary"]._option_string_actions
    # R4: the sweep's confirmation. Only the sweep has it -- it is not a global "don't ask me", and
    # nothing else in this family removes anything without naming one artifact.
    assert {name for name, m in modes.items() if "--yes" in m._option_string_actions} == {"sweep"}


# ── fix round 1 (post-implementation review) ─────────────────────────────────────────────────────


def test_artifacts_summary_write_failure_is_refused_not_a_crash(browse_store, tmp_path, capsys):
    """IMPORTANT 2 (fix round 1): the window wraps the identical write and reports it
    (``artifact_screen._lineage_report``'s own ``except OSError``); an operator's ``--out`` typo -- a
    directory, a read-only file -- must not escape ``write_text`` as an unhandled traceback and a
    failure banner. Writing to a DIRECTORY is the OSError every platform raises for free, no fixture
    needed."""
    root, ids = browse_store
    a_directory = str(tmp_path)                   # tmp_path exists and is a directory, not a file

    capsys.readouterr()
    assert main(["artifacts", "summary", "posterior", ids["posterior"], "--out", a_directory]) == 1
    out = capsys.readouterr()
    err = out.err
    lines = [ln for ln in err.splitlines() if ln.startswith("prism artifacts: refused:")]
    assert len(lines) == 1, err
    # repr(args.out) is what the message quotes, so backslashes come back doubled on Windows; the
    # directory's own NAME is what actually identifies it either way.
    assert Path(a_directory).name in lines[0], lines[0]
    assert "PermissionError" in lines[0] or "IsADirectoryError" in lines[0], lines[0]
    assert "Traceback" not in err and "raised at" not in err, err
    assert "lineage report" not in out.out, "a failed write prints no success line"


def test_note_and_rm_on_a_leftover_name_sweep_as_the_next_step(browse_store, capsys):
    """IMPORTANT 3 (fix round 1): a ref copied straight off ``list``'s own "incomplete ..." row
    cannot resolve through ``_find`` (which never returns a manifest-less entry), and the store's
    "no complete artifact" refusal said nothing about why, or what removes it -- unlike ``show``,
    which already states both honest gaps. ``note`` and ``rm`` now name ``sweep``, but ONLY for a ref
    that actually names one of THIS kind's leftovers; an ordinary typo still gets the plain refusal,
    with nothing invented about it.

    Whole-piece review (three lenses): the sentence is built as ``exc.message + hint``, and the
    store's message ENDS with the ref while the hint BEGAN with it -- so the ref printed twice, back
    to back, with no sentence break, in the one exit-1 line an operator sees after copying a ``list``
    row's "incomplete" name into ``note``. The substring assertions below could not see it; the
    count can."""
    root, ids = browse_store
    leftover_name = ids["bad"][0]                  # "leftover_no_manifest" -- build_browse_store's
    assert (root / "priors" / leftover_name).is_dir(), "build_browse_store seeds this one"

    for argv in (["artifacts", "note", "prior", leftover_name, "--note", "x"],
                ["artifacts", "rm", "prior", leftover_name]):
        capsys.readouterr()
        assert main(argv) == 1, argv
        err = capsys.readouterr().err
        lines = [ln for ln in err.splitlines() if ln.startswith("prism artifacts: refused:")]
        assert len(lines) == 1, err
        assert "sweep" in lines[0] and leftover_name in lines[0], lines[0]
        assert "python -m core artifacts sweep prior" in lines[0], lines[0]
        assert f". {leftover_name!r} is one of the leftovers" in lines[0], lines[0]
        assert f"{leftover_name!r} {leftover_name!r}" not in lines[0], lines[0]

    # An ordinary typo is not a leftover: no artifact by that name or id, and no directory by that
    # name either, so nothing invents a next step that is not true.
    capsys.readouterr()
    assert main(["artifacts", "rm", "prior", "nosuchref"]) == 1
    err = capsys.readouterr().err
    assert "sweep" not in err, err

    capsys.readouterr()
    assert main(["artifacts", "note", "prior", "nosuchref", "--note", "x"]) == 1
    err = capsys.readouterr().err
    assert "sweep" not in err, err


def test_the_shows_two_honest_gaps_are_word_for_word_the_browsers_own_constants(browse_store, capsys):
    """RULED IN 5 (fix round 1): the comment above ``_show``'s ``print`` claims its two "no log"
    sentences are the browser's OWN CONSTANTS, word for word -- and that claim drifted once already
    (this task's own first round silently fixed a stale copy, with no test to catch it). Pinned the
    way the column table already is
    (``test_the_artifacts_listing_shows_the_browsers_own_columns``): import both front ends and
    compare the actual strings, not eyeball the source a third time."""
    from core.artifacts import ArtifactStore
    from core.artifacts.store import LOG_FILE
    from core.gui.screens import artifact_screen as ascreen
    root, ids = browse_store
    s = ArtifactStore(root)

    capsys.readouterr()
    assert main(["artifacts", "show", "simulation", ids["simulation"]]) == 0
    assert ascreen._CACHE_NO_LOG in capsys.readouterr().out

    (s.path("observation", ids["observation"]) / LOG_FILE).unlink(missing_ok=True)
    capsys.readouterr()
    assert main(["artifacts", "show", "observation", ids["observation"]]) == 0
    assert ascreen._NO_RUN_LOG in capsys.readouterr().out


# ── piece 4, B14: the tool's root sink ───────────────────────────────────────────────────────────
def test_main_installs_the_root_handler_for_the_run_and_leaves_nothing_behind(tool_env, monkeypatch,
                                                                             capsys):
    """Every record from a logger OUTSIDE the ``core`` tree goes to STDERR, whatever its level,
    prefixed with the logger that said it. Deliberately unlike the window's level split: this tool's
    STDOUT carries results a script reads (the GPU recipe in CLAUDE.md greps ``[checkpoint] resuming
    at batch`` off it), so a library's chatter may never land there.

    ``logging.warning`` -- the module-level function, on the ROOT logger -- is sbi's own shape
    (sbi/samplers/rejection/rejection.py:336,359), and ``root`` names nothing, so for that ONE logger
    name the prefix falls back to ``record.module``, the basename of the file that logged:
    ``library: rejection:`` under sbi, and this test file's own stem here (the ``logging.warning``
    below is called from ``_prior``, which lives in this file). Walkthrough row D15 expects the module
    name.

    Two more things are pinned here, both of which only a repeated-call test can see: the handler is
    installed FOR the handler call and removed in a finally (``main`` runs dozens of times in one
    process under this suite, and the root's handler list must read the same after every call), and a
    ``core`` record still appears exactly once, on stdout -- ``console_handlers`` on the ``core``
    logger already emitted it, so the root sink's filter keeps it out (were the filter to let it
    through, the sink would write it to stdout a second time and the count of one would fail).

    What this test can NOT see is the basicConfig doubling itself: pytest keeps a handler of its own
    on the root for every test phase, so ``logging.warning`` never meets a handler-less root here
    and the ``[budget]``-not-on-stderr line cannot catch it. That precondition and its prevention
    are pinned in tests/test_refusals.py, with pytest's root handlers taken off."""
    import logging

    from core import logging_root, orchestrator
    bounds, cell, root = tool_env
    lib = logging.getLogger("a_library")
    lib.setLevel(logging.INFO)
    root_logger = logging.getLogger()
    before = root_logger.handlers[:]

    def _prior(cfg, ref, build_new, **kw):
        assert logging_root.installed() is True, "the handler must exist FOR the handler call"
        logging.warning("Only 0.5% proposal samples are accepted.")
        lib.info("a library information record")
        logging.getLogger("core.orchestrator").info("[budget] 2 batches x 8 rows")
        return _art(root, "prior")

    monkeypatch.setattr(orchestrator, "build_prior", _prior)
    here = Path(__file__).stem              # record.module: the file that called logging.warning
    try:
        for _ in range(2):
            capsys.readouterr()
            assert main(["prior", *_cfg(bounds)]) == 0
            cap = capsys.readouterr()
            assert cap.err.count(
                f"library: {here}: Only 0.5% proposal samples are accepted.\n") == 1, cap.err
            assert cap.err.count("library: a_library: a library information record\n") == 1, cap.err
            assert "library:" not in cap.out, f"a library record reached the results stream:\n{cap.out}"
            assert cap.out.count("[budget] 2 batches x 8 rows\n") == 1, cap.out
            assert "[budget]" not in cap.err, cap.err
            assert logging_root.installed() is False, "main left its root handler installed"
            assert root_logger.handlers == before, root_logger.handlers
    finally:
        logging_root.remove()
        lib.setLevel(logging.NOTSET)


def test_the_tools_root_sink_routes_a_core_record_as_the_console_handlers_would(capsys):
    """Fix round 1. A ``core`` record reaches the tool's root sink only when the console handlers are
    not attached (``main`` attaches them for the handler call alone), and it is PRISM's own voice, so
    it is written as those handlers would write it -- information bare on stdout, warning and above on
    stderr with the level as a prefix -- never ``library:``, and never lost. A library record goes to
    stderr whatever its level, its ``log.exception`` traceback included."""
    import logging
    import sys

    import core.tool as tool

    def _record(name, level, msg, exc_info=None):
        return logging.LogRecord(name, level, __file__, 1, msg, None, exc_info)

    capsys.readouterr()
    tool._library_record_sink(_record("core.probe", logging.INFO, "an information record"))
    tool._library_record_sink(_record("core.probe", logging.WARNING, "a warning"))
    tool._library_record_sink(_record("a_library", logging.INFO, "a library's note"))
    try:
        raise ValueError("the library's own failure")
    except ValueError:
        tool._library_record_sink(_record("a_library", logging.ERROR, "it failed", sys.exc_info()))
    cap = capsys.readouterr()
    assert cap.out == "an information record\n", cap.out
    assert cap.err.startswith("warning: a warning\nlibrary: a_library: a library's note\n"
                              "library: a_library: it failed\nTraceback (most recent call last):\n"), \
        cap.err
    assert cap.err.endswith("ValueError: the library's own failure\n"), cap.err
