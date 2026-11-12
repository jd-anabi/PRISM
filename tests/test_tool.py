"""The command-line tool (``python -m core``).

Everything runs IN PROCESS through ``main(argv)``. A subprocess would look more like the operator's
command line and would catch less: a handler that swaps the process default store and never restores
it, or a knob flag that quietly stops reaching its stage, is invisible from outside. Every test here
therefore asserts that ``default_store()`` is still the session sandbox when ``main`` returns, and the
knob tests read the keyword a recorder actually received rather than the run's output.
"""
import argparse
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
    would trivially pass. Recording it here, ahead of every module fixture, is what
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
    SINGLE-PROCESS gate could ever notice; this makes it a per-test failure instead.

    Compared against ``_session_default_store`` -- captured once, before any module fixture's real
    ``main`` calls -- rather than a fresh ``default_store()`` read here, so a leak already baked into a
    per-test ``before`` by ``tool_run``'s fixture setup cannot hide behind it."""
    from core.artifacts import default_store
    yield
    assert default_store() is _session_default_store, "the tool left its own store as the process default"


def test_the_help_epilog_names_the_core_environment_settings():
    """The two roots and the two core-level settings are named where an operator looks -- and nowhere
    else, because the tool itself reads none of them."""
    epilog = build_parser().epilog
    for name in ("PRISM_RESOURCES", "PRISM_ARTIFACTS", "PRISM_VRAM_CEILING_GIB", "PRISM_MEM_LOG_EVERY"):
        assert name in epilog, name


HOUSE_METAVARS = frozenset({"N", "X", "S", "K", "REF", "PATH", "NAME", "TEXT", "STAGE[,STAGE...]",
                            "PATH[@HZ]", "NAME=VALUE", ("MIN", "MAX", "N")})


def _parsers(parser=None, path=()):
    """Every parser under ``python -m core`` with its path as typed after it: the top level (path
    ""), each subcommand, and each mode of a subcommand that has modes."""
    parser = build_parser() if parser is None else parser
    yield " ".join(path), parser
    for a in parser._actions:
        if isinstance(a, argparse._SubParsersAction):
            for name, sub in a.choices.items():
                yield from _parsers(sub, (*path, name))


def _leaves(parser=None):
    """The parsers that run something: each subcommand without modes, and each mode."""
    for path, p in _parsers(parser):
        if path and not any(isinstance(a, argparse._SubParsersAction) for a in p._actions):
            yield path, p


def test_every_leaf_parser_has_a_description():
    """``<subcommand> --help`` and ``<subcommand> <mode> --help`` open with what the command does:
    every subcommand and mode parser carries a description, and it is the one-liner the listing
    above it shows, so the two never disagree."""
    seen = []
    for path, parser in _parsers():
        for a in parser._actions:
            if isinstance(a, argparse._SubParsersAction):
                listed = {ca.dest: ca.help for ca in a._choices_actions}
                for name, sub in a.choices.items():
                    where = f"{path} {name}".strip()
                    assert sub.description, f"`{where} --help` says nothing about what it does"
                    assert sub.description == listed[name], (where, sub.description, listed[name])
                    seen.append(where)
    assert {"prior", "smoke", "identifiability laplace", "probes band", "artifacts sweep"} <= set(seen)


def test_every_value_flag_has_a_house_placeholder():
    """A flag that takes a value says what it is and shows what to type from the house set -- N a
    count, X a real number, S seconds, K the probe count, REF an artifact's name or id, PATH, NAME,
    TEXT -- or lists its choices; never the internal destination name argparse prints otherwise."""
    bad, no_help = [], []
    for path, leaf in _leaves():
        for a in leaf._actions:
            if not a.option_strings or a.nargs == 0:
                continue
            if not a.help:
                no_help.append(f"{path} {a.option_strings[0]}")
            if a.choices is None and a.metavar not in HOUSE_METAVARS:
                bad.append(f"{path} {a.option_strings[0]}: {a.metavar!r}")
    assert not no_help, "value flags that say nothing:\n" + "\n".join(no_help)
    assert not bad, "value flags without a house placeholder:\n" + "\n".join(bad)
    shown = {(path, a.option_strings[0]): a.metavar
             for path, leaf in _leaves() for a in leaf._actions if a.option_strings}
    assert shown[("infer", "--f0-si")] == "X", "a force in newtons, not a count"
    assert shown[("tsnpe", "--level")] == "X"
    assert shown[("sbc", "--chi-k-fixed")] == "K", "the chi probe count, as --chi-k shows it"
    assert shown[("smoke", "--stages")] == "STAGE[,STAGE...]"


def _value_actions(parser):
    """The options that take a value, plus --chi/--no-chi, whose default is a value too; the on/off
    switches and --help state no default."""
    for a in parser._actions:
        if a.option_strings and (a.nargs != 0 or isinstance(a, argparse.BooleanOptionalAction)):
            yield a


def _owned_defaults() -> dict:
    """``{(leaf, flag): value}`` for every default the help table pins, read from what owns it."""
    import inspect
    from core.diagnostics import identifiability, probes, sbc

    def sig(fn, name):
        return str(inspect.signature(fn).parameters[name].default)

    lap, jac = identifiability.identifiability_laplace, identifiability.identifiability_jacobian
    return {
        ("sbc", "--seed"): sig(sbc.sbc_repeats, "seed"),
        ("sbc", "--n-cal"): sig(sbc.sbc_repeats, "n_cal"),
        ("sbc", "--posterior-samples"): sig(sbc.sbc_repeats, "num_posterior_samples"),
        ("identifiability laplace", "--seed"): sig(lap, "seed"),
        ("identifiability laplace", "--min-valid"): sig(lap, "min_valid"),
        ("identifiability laplace", "--sd-identified"): sig(lap, "sd_identified"),
        ("identifiability jacobian", "--seed"): sig(jac, "seed"),
        ("identifiability jacobian", "--min-valid"): sig(jac, "min_valid"),
        ("identifiability jacobian", "--zero-tol"): sig(jac, "zero_tol"),
        ("identifiability jacobian", "--noise-eps"): sig(jac, "noise_eps"),
        ("probes band", "--seed"): sig(probes.probe_band, "seed"),
        ("probes band", "--repeats"): sig(probes.probe_band, "repeats"),
        ("probes band", "--cv-max"): sig(probes.probe_band, "cv_max"),
        ("probes band", "--snr-min"): sig(probes.probe_band, "snr_min"),
        ("probes band", "--sup-min"): sig(probes.probe_band, "sup_min"),
        ("probes band", "--peak-window"): sig(probes.probe_band, "peak_window"),
        ("probes mask", "--num-runs"): sig(probes.probe_mask, "num_runs"),
        ("probes mask", "--run-size"): sig(probes.probe_mask, "run_size"),
        ("probes mask", "--seed"): sig(probes.probe_mask, "seed"),
        ("probes drive", "--t-obs"): sig(probes.probe_drive, "t_obs_s"),
        ("probes drive", "--repeats"): sig(probes.probe_drive, "repeats"),
        ("probes drive", "--detune"): sig(probes.probe_drive, "detune"),
        ("probes drive", "--free-min"): sig(probes.probe_drive, "free_min"),
        ("probes drive", "--captured-max"): sig(probes.probe_drive, "captured_max"),
        ("probes drive", "--peak-window"): sig(probes.probe_drive, "peak_window"),
        ("probes drive", "--clarity-min"): sig(probes.probe_drive, "clarity_min"),
        ("probes drive", "--seed"): sig(probes.probe_drive, "seed"),
        ("smoke", "--t-obs"): str(config.T_MIN_EXP_S),
    }


def _default_problems(parser) -> list:
    """One line per value flag whose help does not state its default the way its class requires."""
    from core.refusals import FIELDS, default_text
    from core.tool import fields as tool_fields
    from core.tool.help_defaults import DEFAULT_CLASS, default_for

    owned, chi_word = _owned_defaults(), ("--chi" if config.CHI_MODE else "--no-chi")
    problems = []
    for path, leaf in _leaves(parser):
        for a in _value_actions(leaf):
            opt = a.option_strings[0]
            where = f"{path} {opt}"
            if (path, opt) not in DEFAULT_CLASS:
                problems.append(f"{where}: no entry in core/tool/help_defaults.py DEFAULT_CLASS")
                continue
            cls, text = DEFAULT_CLASS[(path, opt)]
            shown = " ".join(leaf._get_formatter()._expand_help(a).split()) if a.help else ""
            if not shown:
                problems.append(f"{where}: no help")
                continue
            if cls == "none":
                want = None if text is None else problems.append(f"{where}: class none carries {text!r}")
            elif cls == "behaviour":
                if not text:
                    problems.append(f"{where}: a behaviour default needs its phrase")
                    continue
                want = default_text(text)
            elif cls == "value":
                if a.default not in (None, ""):
                    source, pinned = str(a.default), None
                elif opt == "--chi":
                    source = pinned = chi_word
                elif (path, opt) in owned:
                    source = pinned = owned[(path, opt)]
                else:
                    keys = [k for k, f in tool_fields.FLAG.items() if f == opt]
                    if len(keys) > 1:
                        keys = [k for k in keys if k == a.dest.lower()]
                    source, pinned = (FIELDS[keys[0]].default if len(keys) == 1 else None), None
                if source is None:
                    problems.append(f"{where}: a value default with no source; pin it in DEFAULT_CLASS")
                    continue
                if text != pinned:
                    problems.append(f"{where}: DEFAULT_CLASS says {text!r}, the code's default is {source!r}")
                want = default_text(source)
            else:
                problems.append(f"{where}: unknown class {cls!r}")
                continue
            if default_for(path, a) != want:
                problems.append(f"{where}: default_for gives {default_for(path, a)!r}, not {want!r}")
            count = shown.count("(default")
            if want is None and count:
                problems.append(f"{where}: states a default, but its class is none: {shown!r}")
            if want is not None and (count != 1 or not shown.endswith(want.strip())):
                problems.append(f"{where}: must end with {want.strip()!r}, stated once: {shown!r}")
            if a.dest == "name" and a.default == "" and "unnamed" not in shown:
                problems.append(f"{where}: must say what '' means (unnamed)")
            if a.dest == "note" and a.default == "" and "no note" not in shown:
                problems.append(f"{where}: must say what '' means (no note)")
    return problems


def test_every_value_flag_states_its_default_by_its_class():
    """Every flag that takes a value states its default one way: "(default X)" with the value the
    code really uses, "(default <phrase>)" where the default is a behaviour, or nothing for a
    required flag, an input and --name/--note. The value is computed here from what owns it -- the
    parser, the signature or constant the table pins, or the refusal registry through the flag's
    field key -- and compared exactly. A flag added without an entry in the table fails, naming the
    subcommand and the flag."""
    import inspect
    from core.diagnostics import identifiability
    from core.refusals import FIELDS
    from core.tool.help_defaults import DEFAULT_CLASS, LEAVES

    parser = build_parser()
    problems = _default_problems(parser)
    assert not problems, "\n".join(problems)
    walked = [path for path, _ in _leaves(parser)]
    assert sorted(LEAVES) == sorted(walked), sorted(set(walked) ^ set(LEAVES))
    present = {(path, a.option_strings[0]) for path, leaf in _leaves(parser) for a in _value_actions(leaf)}
    assert not set(DEFAULT_CLASS) - present, sorted(set(DEFAULT_CLASS) - present)
    # the registry pins these three against laplace only; the jacobian's help relies on them too
    jac = inspect.signature(identifiability.identifiability_jacobian).parameters
    for key in ("m", "m_noise", "rel"):
        assert str(jac[key].default) == FIELDS[key].default, key

    parser.subcommands["smoke"].add_argument("--brand-new", type=int, metavar="N", help="a new knob")
    named = [p for p in _default_problems(parser) if p.startswith("smoke --brand-new:")]
    assert named and "DEFAULT_CLASS" in named[0], _default_problems(parser)


def test_every_epilog_and_raw_description_fits_in_79_columns():
    """An epilog, and the description of a parser that prints it as typed, is wrapped by hand; a line
    wider than 79 columns wraps raggedly on an 80-column console."""
    wide = []
    for path, p in _parsers():
        texts = [p.epilog or ""]
        if p.formatter_class is argparse.RawDescriptionHelpFormatter:
            texts.append(p.description or "")
        wide += [f"{path or 'python -m core'}: {ln!r}" for t in texts for ln in t.splitlines() if len(ln) > 79]
    assert not wide, "\n".join(wide)


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
    """Every setting is a flag that travels to its stage as a keyword.

    An ``os.environ`` read inside the tool would be a knob no flag names, and an assignment to a
    module constant is the import-binding trap -- orchestrator binds config constants at import, so
    ``config.X = v`` changes nothing and the run silently uses the default. The entry is the one
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
    """The stage ``--note`` (config_args.add_name_flags) used to reach the manifest unchecked: the
    store's contract leaves the note rule to each FRONT END (``ArtifactStore.set_note``:
    "require_note is that rule, and both front ends run it"), ``create`` stores a note as it is
    given, and only ``artifacts note`` ran the rule. So a 250-character note, or one with a newline,
    was written into a record that the tool's own ``artifacts note`` would then refuse to write back.

    ``main`` now judges it ONCE, before any handler runs, for every subcommand that has the flag: a
    refusal is the ladder's one line ending ``(--note)``, exit 1, and nothing -- not the config
    build, not the stage -- has run. The stage is a recorder, so a refused note is asserted by what
    never reached it.
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
                 "--new-run", "--resume-fit", "kf"]) == 0
    (_, p_ref, p_new), _ = prior_rec.calls[0]
    assert p_ref == "tp" and p_new is False, "train LOADS its prior"
    (cfg, prior, ref, train_new), kw = post_rec.calls[0]
    assert prior is loaded_prior and ref is None and train_new is True
    assert set(kw) == {"name", "note", "fig_sink", "store", "new_run", "num_runs", "run_size_cap",
                       "hidden_features", "num_transforms", "learning_rate", "stop_after_epochs",
                       "max_num_epochs", "fisher_m", "fisher_dz", "fisher_points",
                       "checkpoint_every", "resume", "resume_fit"}
    assert (kw["num_runs"], kw["run_size_cap"]) == (3, 8)
    assert (kw["hidden_features"], kw["num_transforms"]) == (16, 2)
    assert (kw["learning_rate"], kw["stop_after_epochs"], kw["max_num_epochs"]) == (0.001, 4, 6)
    assert (kw["fisher_m"], kw["fisher_dz"], kw["fisher_points"]) == (12, 0.05, 2)
    assert kw["checkpoint_every"] == 1 and kw["resume"] == "require" and kw["new_run"] is True
    assert kw["resume_fit"] == "kf"

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


def test_the_help_lists_no_command_the_program_no_longer_has(capsys):
    """PRISM infers a cell's parameters and does nothing else, so the commands of the effective temperature -- its
    measurement, its sweep study and the comparisons of their records -- are neither parsed nor listed: the parser
    holds the eleven commands below and no other, and the help names none of the three."""
    import re
    parser = build_parser()
    commands, = [action.choices for action in parser._actions if isinstance(action, argparse._SubParsersAction)]
    assert set(commands) == {"prior", "train", "validate", "infer", "tsnpe", "sbc", "identifiability", "ablation",
                             "probes", "smoke", "artifacts"}
    capsys.readouterr()
    assert main(["--help"]) == 0
    said = capsys.readouterr().out
    assert [gone for gone in ("fdt", "crossval", "compare") if re.search(rf"(?<![\w-]){gone}(?![\w-])", said)] == []
    for gone in ("fdt", "crossval", "compare"):
        assert main([gone]) == 2, f"{gone} is still a command"
    capsys.readouterr()


def test_a_taken_name_is_refused_with_exit_1_and_nothing_written(tool_run, capsys):
    from core.artifacts import ArtifactStore
    bounds, cell, root = tool_run
    store = ArtifactStore(root)
    before = [s.id for s in store.list("prior")]
    capsys.readouterr()
    assert main(["prior", *_cfg(bounds), "--name", "tp"]) == 1
    err = capsys.readouterr().err
    # A refusal on the tool is ONE line: the message and the flag that answers it. No class name and no
    # [raised at ...] -- those were hedges for a bug disguised as a ValueError, which a dedicated
    # Refusal class no longer needs. fix_sentence owns the parentheses, so "((" would mean the
    # message carried its own copy of the flag.
    lines = [ln for ln in err.splitlines() if ln.startswith("prism prior: refused:")]
    assert len(lines) == 1, err
    assert "already exists" in lines[0] and lines[0].endswith("(--name)"), lines[0]
    assert "((" not in lines[0] and "raised at" not in lines[0] and "StoreError" not in lines[0], lines[0]
    assert [s.id for s in store.list("prior")] == before


def test_parse_forced_and_recording_set_rules():
    """The recording-set rules, as a unit: which recordings each observation mode takes, decided
    before anything is loaded or spent. The stub configs carry only the two fields the function is
    allowed to read."""
    import re
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
    msg = str(exc.value)
    assert "must state the frequency (Hz) it was driven at" in msg, msg
    assert not re.search(r"\([A-Z]\d+\)", msg), \
        f"the refusal carries an internal label in place of its reason: {msg}"
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

    # The recording rules come before the load, PROVED rather than assumed: --posterior names a ref
    # that does not exist, so the only way this can return 2 (recording_set's UsageError) rather than
    # 1 (load_posterior_and_prior's StoreError) is if the recording rules are checked BEFORE the load.
    # Moving recording_set after the load would silently turn this into a 1.
    bad_ref = ["infer", *_cfg(bounds), "--posterior", "nosuch", "--t-obs", "1.0"]
    capsys.readouterr()
    assert main([*bad_ref, "--spont", "x.npy", "--forced", "y.npy"]) == 2
    err = capsys.readouterr().err
    assert "usage:" in err and "no Forcing section" in err

    # --cell with any of the experimental-only recording flags: refused before the load too.
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


def test_validate_takes_a_seed_and_a_failing_verdict_still_exits_zero(tool_run, monkeypatch, capsys):
    """A failing calibration verdict is a result, not an error: the command prints the verdict, names
    the seed it ran with, writes the calibration with both in it and exits 0."""
    from core.artifacts import ArtifactStore
    from tests._fixtures import stub_calibration_battery
    bounds, cell, root = tool_run
    stub_calibration_battery(monkeypatch, ks_pvals=[0.5, 0.5, 0.5, 0.001])   # the test model infers four
    capsys.readouterr()
    assert main(["validate", *_cfg(bounds), "--posterior", "tpost", "--seed", "3", "--n-cal", "8",
                 "--name", "failing_cal"]) == 0
    out = capsys.readouterr().out
    assert "[verdict] FAIL" in out and "(calibration seed 3)" in out, out
    res = ArtifactStore(root).get("calibration", "failing_cal").body["results"]
    assert res["seed"] == 3 and res["verdict"]["passed"] is False


def test_near_miss_is_refused_before_simulation(tool_run, capsys):
    """A committed cache one setting away is refused before any simulation, on the command line. A
    second run once silently started a new cache under an identity one field away from the first
    run's; that is now a refusal with no spend."""
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


def test_train_resume_fit_continues_a_kept_training_and_names_the_flag_of_what_differs(tool_run, monkeypatch,
                                                                                         capsys):
    """A training stopped by Ctrl-C while its flow trains is kept, and the interrupt's advice names
    --resume-fit; resumed with a setting that differs from the kept training's, it is refused naming that
    setting's flag; resumed with nothing but the kept training, it writes the posterior under the name it
    was started with, and the kept training is gone. Placed after the near-miss test, which expects no
    cache but the module's own one setting away from its run. The resume reads back the kept training's
    six batches: a resume that lost its kept training would fall back on the configured budget of
    thousands, and fails at once instead of simulating for hours."""
    from core.artifacts import ArtifactStore
    from core.SBI import fit_state, pipeline
    bounds, cell, root = tool_run
    before = {s.id for s in ArtifactStore(root).list("fit")}
    real = fit_state.FitSession.save_pass

    def save_pass(self, take):
        real(self, take)
        if self.passes == 2:
            raise KeyboardInterrupt

    argv = ["train", *_cfg(bounds), "--prior", "tp"]
    capsys.readouterr()
    with monkeypatch.context() as patch:
        patch.setattr(fit_state.FitSession, "save_pass", save_pass)
        assert main([*argv, "--name", "tk", "--num-runs", "6", "--run-size", "8", "--hidden-features", "8",
                     "--num-transforms", "1", "--stop-after-epochs", "3", "--max-epochs", "8",
                     "--checkpoint-every", "1", "--new-run"]) == 130
    said = capsys.readouterr()
    assert "--resume-fit" in said.err, said.err
    kept, = [s for s in ArtifactStore(root).list("fit") if s.id not in before]
    assert f"is kept as fit {kept.id}" in said.out, said.out
    real_rows = pipeline.gen_training_data

    def rows(*args, **kwargs):
        batches = args[5] if len(args) > 5 else kwargs.get("n_runs")
        assert batches == 6, f"the resume reached the simulation with {batches} batches, not the kept training's 6"
        return real_rows(*args, **kwargs)

    monkeypatch.setattr(pipeline, "gen_training_data", rows)
    assert main([*argv, "--resume-fit", kept.id, "--hidden-features", "16"]) == 1
    err = capsys.readouterr().err
    refused = [ln for ln in err.splitlines() if ln.startswith("prism train: refused:")]
    assert len(refused) == 1 and refused[0].endswith("(--hidden-features)"), err
    assert main([*argv, "--resume-fit", kept.id]) == 0
    assert "[prism] posterior tk__" in capsys.readouterr().out
    assert [s for s in ArtifactStore(root).list("fit") if s.id not in before] == []


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


def test_the_training_stages_two_refusals_of_a_continuation_end_with_the_resume_flag(tool_run, capsys):
    """A training told to continue a simulation set it cannot continue is refused before anything is
    spent: the policy set to 'require' with no set kept at all (checkpointing off), and 'require' where
    no set of these settings is kept. Either line ends with the flag of the policy, the setting that was
    moved off its default; the first sentence names the cadence by its keyword. Nothing is written."""
    from core.artifacts import ArtifactStore
    bounds, cell, root = tool_run
    store = ArtifactStore(root)
    before = ({s.id for s in store.list("simulation")}, {s.id for s in store.list("posterior")})
    argv = ["train", *_cfg(bounds), "--prior", "tp", "--run-size", "8", "--hidden-features", "8",
            "--num-transforms", "1", "--stop-after-epochs", "1", "--resume", "require"]

    capsys.readouterr()
    assert main([*argv, "--num-runs", "2", "--checkpoint-every", "0"]) == 1
    err = capsys.readouterr().err
    lines = [ln for ln in err.splitlines() if ln.startswith("prism train: refused:")]
    assert len(lines) == 1, err
    assert "resume='require' needs checkpointing on" in lines[0] and "checkpoint_every=0" in lines[0], lines[0]
    assert lines[0].endswith(" (--resume)"), lines[0]

    assert main([*argv, "--num-runs", "5", "--checkpoint-every", "1"]) == 1
    err = capsys.readouterr().err
    assert "prism train: refused: resume='require' but there is no resumable cache at" in err, err
    assert err.rstrip().endswith(" (--resume)"), err
    assert "raised at" not in err and "Traceback" not in err, err
    assert ({s.id for s in store.list("simulation")}, {s.id for s in store.list("posterior")}) == before


def test_every_validate_infer_and_tsnpe_flag_reaches_its_stage_as_a_keyword(tool_round, monkeypatch):
    """A flag's dest IS the stage's keyword name, for the three subcommands the prior and train tests
    do not cover. Pins the WHOLE keyword mapping each call produces -- ``set(kw) ==`` the exact set, as
    ``test_every_train_flag_reaches_build_posterior_as_a_keyword`` does -- not a sample of it:
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

    # validate: every VALIDATE_KNOBS flag, a distinct value each, and the seed.
    assert main(["validate", *_cfg(bounds), "--posterior", "tpost", "--n-cal", "7",
                 "--cal-n-scales", "3", "--posterior-samples", "11", "--seed", "5"]) == 0
    (_cfg1, _post1, _prior1), kw = v.calls[0]
    assert set(kw) == {"name", "note", "fig_sink", "store", "n_cal", "cal_n_scales",
                       "num_posterior_samples", "seed"}
    assert (kw["n_cal"], kw["cal_n_scales"], kw["num_posterior_samples"]) == (7, 3, 11)
    assert kw["seed"] == 5
    # No --seed: the tool draws one and passes it, so the record always names the seed it ran with.
    assert main(["validate", *_cfg(bounds), "--posterior", "tpost"]) == 0
    drawn = v.calls[1][1]["seed"]
    assert type(drawn) is int and 0 <= drawn < 2 ** 31

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
                 "--resume", "require", "--new-run", "--resume-fit", "kf"]) == 0
    (_c3, _p3, _pr3, _obs3), kw = t.calls[0]
    assert set(kw) == {"name", "note", "fig_sink", "store", "new_run", "n_directions", "level",
                       "num_runs", "run_size_cap", "hidden_features", "num_transforms",
                       "learning_rate", "stop_after_epochs", "max_num_epochs", "checkpoint_every",
                       "resume", "resume_fit"}
    assert (kw["n_directions"], kw["level"]) == (2, 0.99)
    assert (kw["num_runs"], kw["run_size_cap"]) == (6, 9)
    assert (kw["hidden_features"], kw["num_transforms"]) == (10, 3)
    assert (kw["learning_rate"], kw["stop_after_epochs"], kw["max_num_epochs"]) == (0.002, 5, 3)
    assert kw["checkpoint_every"] == 2 and kw["resume"] == "require" and kw["new_run"] is True
    assert kw["resume_fit"] == "kf"

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
    # The FULL set, not a sample of it (as the validate/infer/tsnpe knob test above pins
    # VALIDATE_KNOBS/TSNPE_KNOBS): knobs() silently DROPS a dest that no longer matches a flag, so
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

    Every flag reaches its handler as a keyword: rotation and laplace declare the accept flag with
    `add_accept_flags` and build their Accept with `accept_from`, the single definition of how an
    --accept-* flag answers a narrowed or truncated load -- so this also pins the FULL `set(kw)` each
    mode's handler forwards, one distinct value per knob, the way the validate/infer/tsnpe and sbc
    knob tests pin VALIDATE_KNOBS/TSNPE_KNOBS/sbc's set: `knobs()` silently drops a dest that no
    longer matches a flag, so checking only a few keys would stay green through that."""
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
            seen_args[_n] = a               # the POSITIONAL cfg, which the keyword set alone misses
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
    # rotation's cfg must NOT carry a loaded ground truth (build_cfg's needs_gt=False for it) --
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
    # laplace's cfg DOES need a loaded ground truth (build_cfg's needs_gt=True for it).
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
    # jacobian's cfg likewise needs a loaded ground truth (build_cfg's needs_gt=True for it too).
    assert seen_args["identifiability_jacobian"][0].ground_truth is not None

    assert tool.main(["identifiability", "rotation", "--bounds", bounds, "--device", "cpu",
                      "--posterior", "p", "--cell", cell]) == 2
    assert tool.main(["identifiability", "--bounds", bounds]) == 2


def test_identifiability_rotation_refuses_a_posterior_without_a_rotation(tool_run, capsys):
    """Through the REAL posterior load: no stub stands between the tool and the store.

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


def _refused_lines(err: str) -> list[str]:
    return [line for line in err.splitlines() if ": refused: " in line]


def test_the_input_use_refusal_names_the_posterior_flag(tmp_path, monkeypatch, capsys):
    """`ablation` on a posterior whose training kept no simulation set prints one refusal line: the stage's sentence,
    then the flag that names the posterior, with no class name and no place in the code."""
    from core import tool
    from core.artifacts import ArtifactStore
    from core.tool import diagnostics as tool_diag
    from tests._fixtures import _nad_cfg, _posterior_artifact
    monkeypatch.setenv("PRISM_ARTIFACTS", str(tmp_path / "A"))
    store, cfg = ArtifactStore(tmp_path / "A"), _nad_cfg()
    loaded = store.load_posterior(cfg, _posterior_artifact(store, cfg, name="no_cache_post").id)
    monkeypatch.setattr(tool_diag, "load_posterior_and_prior", lambda c, ref, accept, s: (loaded, None))
    bounds = str(config.BOUNDS_PATH / "nadrowski" / "master.txt")
    assert tool.main(["ablation", "--bounds", bounds, "--device", "cpu", "--posterior", "no_cache_post"]) == 1
    err = capsys.readouterr().err
    assert _refused_lines(err) == [
        "prism ablation: refused: Posterior 'no_cache_post' names no simulation cache, so checkpointing was off for "
        "its training run and its conditioning rows no longer exist. The ranges a channel is swept over have to be "
        "the ranges THIS network was trained on; another cache's quantiles describe a different experiment. Retrain "
        "with checkpointing on, or pick a posterior that has one. (--posterior)"], err
    assert "ValueError" not in err and "[raised at" not in err, err


def test_the_across_the_prior_refusal_names_the_posterior_flag(tmp_path, monkeypatch, capsys):
    """`identifiability laplace` at more than one point, on a posterior that carries no training prior, prints one
    refusal line ending with the flag that names the posterior."""
    from core import tool
    from core.tool import diagnostics as tool_diag
    monkeypatch.setenv("PRISM_ARTIFACTS", str(tmp_path / "A"))
    no_prior = SimpleNamespace(name="no_prior_post", id="20260910T110000", latent=SimpleNamespace(prior=None),
                               posterior=SimpleNamespace(T=None))
    monkeypatch.setattr(tool_diag, "load_posterior_and_prior", lambda c, ref, accept, s: (no_prior, None))
    bounds = str(config.BOUNDS_PATH / "nadrowski" / "master.txt")
    cell = str(config.CELL_PATH / "nadrowski" / "master_weak.txt")
    assert tool.main(["identifiability", "laplace", "--bounds", bounds, "--device", "cpu", "--cell", cell,
                      "--t-obs", "4.5", "--posterior", "no_prior_post", "--n-points", "2"]) == 1
    err = capsys.readouterr().err
    assert _refused_lines(err) == [
        "prism identifiability: refused: this posterior carries no latent training prior (nothing with a `gen_dist` "
        "under its `.prior`), so the off-ground-truth points cannot be drawn from the distribution the flow was "
        "trained on. (--posterior)"], err
    assert "ValueError" not in err and "[raised at" not in err, err


def test_smoke_runs_the_four_stages_and_the_resume_drill_is_loud(tool_env, tmp_path, capsys):
    """The four stages end to end at tiny size, then the drill the GPU smoke gate runs.

    Leg (b) is the resume: the SAME --num-runs against the SAME store, with the prior LOADED by name
    so prior_fingerprint is pinned. Leg (c) is a run one field apart (n_runs), which once silently
    started a new cache and exited 0, and is now a refusal before the Fisher. Leg (d) is the
    stage banner: a stage that raises names itself before the traceback.
    """
    import re

    from core import orchestrator
    from core.artifacts import ArtifactStore, default_store
    from core.tool import main

    def framed(out):
        """The lines that frame each stage on the console, in order, each one's seconds left out."""
        return [re.sub(r" in \d+\.\ds$", " in _s", line) for line in out.splitlines()
                if line.startswith(("=== ", "[ok] ", "[skip] "))]

    # tool_env is what installs SBITEST as real input files and points PRISM_ARTIFACTS at a temp
    # root; --store-root then puts this run's own store beside it.
    bounds, cell, _env_root = tool_env
    sandbox = default_store()
    root = tmp_path / "smoke"
    SCFG = [*_cfg(bounds), "--cell", cell]
    common = ["smoke", *SCFG, "--store-root", str(root), "--run-size", "8", "--t-obs", "1.0",
              "--checkpoint"]

    # (a) the full run: four stages, six artifacts. Each stage is framed on the console by its banner
    # before it and its seconds after it, the lines the card gate is read by.
    assert main([*common, "--num-runs", "2", "--n-cal", "60", "--max-epochs", "2", "--save"]) == 0
    out = capsys.readouterr().out
    assert "[smoke] ALL STAGES COMPLETED" in out, out[-2000:]
    assert framed(out) == ["=== prior ===", "[ok] prior in _s", "=== posterior ===", "[ok] posterior in _s",
                           "=== validate ===", "[ok] validate in _s", "=== infer ===", "[ok] infer in _s"], out[-3000:]
    s = ArtifactStore(root)
    assert [r.name for r in s.list("prior")] == ["smoke_prior"]
    assert [r.name for r in s.list("posterior")] == ["smoke_posterior"]
    for kind in ("simulation", "observation", "calibration", "inference"):
        assert len(s.list(kind)) == 1, kind

    # (b) the drill: same store, same budget, the prior loaded by name -> a resume, in seconds. The two
    # stages it does not ask for are each said to be skipped, after the two it runs.
    drill = [*common, "--num-runs", "2", "--prior", "smoke_prior", "--stages", "prior,posterior"]
    assert main([*drill, "--resume", "require"]) == 0
    out = capsys.readouterr().out
    assert "resuming at batch 2/2" in out, out[-2000:]
    assert framed(out) == ["=== prior ===", "[ok] prior in _s", "=== posterior ===", "[ok] posterior in _s",
                           "[skip] validate", "[skip] infer"], out[-3000:]
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
    # Both values, as leg (c)'s comment says -- read literally off the message's own format
    # (orchestrator._near_miss_lines: "this run <mine>, that cache <theirs>").
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
    """Smoke's own knob forwarding, which no other test pins. Deleting
    ``run_size_cap=args.run_size_cap`` from smoke.py keeps the four-stage test (and leg (b) of its
    drill) green, because both legs of THAT test fall back to the same hardware batch -- the GPU
    gate would then silently train at the card's batch and key a different simulation identity. This
    pins the exact ``set(kw)`` each of the four stage calls receives, one distinct non-default value
    per flag, the way the train, tsnpe and sbc knob tests pin TRAIN_KNOBS/TSNPE_KNOBS/sbc's set.

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
    assert main(["smoke", *_cfg(bounds), "--cell", cell, "--num-runs", "8", "--run-size", "9",
                 "--n-cal", "17", "--max-epochs", "6", "--checkpoint", "--new-run",
                 "--resume", "require", "--t-obs", "2.5", "--seed", "5", "--prior", "smoke_prior",
                 "--hidden-features", "11", "--num-transforms", "2",
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
                        "checkpoint_every", "new_run", "resume", "hidden_features", "num_transforms"}
    assert kw2["hidden_features"] == 11 and kw2["num_transforms"] == 2
    assert kw2["num_runs"] == 8
    assert kw2["run_size_cap"] == 9
    assert kw2["max_num_epochs"] == 6
    assert kw2["checkpoint_every"] == 4, "ck_every = max(1, num_runs // 2) = max(1, 8 // 2)"
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

    # Without the network flags nothing is forwarded, so the stage keeps its own defaults.
    assert main(["smoke", *_cfg(bounds), "--cell", cell, "--stages", "prior,posterior",
                 "--store-root", str(tmp_path / "smoke2")]) == 0
    assert "hidden_features" not in post_rec.calls[-1][1] and "num_transforms" not in post_rec.calls[-1][1]


def test_smoke_rejects_an_unknown_stage_at_parse_time(tool_env, monkeypatch, capsys):
    """An unknown --stages entry is an argparse type= error, so it is refused
    DURING PARSING -- before main ever calls tempfile.mkdtemp or registry.load_user_models(). Pinned
    by call counts on both, not by a directory listing or the exit code alone: the OLD runtime-only
    check (still present in run_smoke as a second, defensive line -- see its own docstring) already
    returned exit 2 for this exact input, AND the leaked-root cleanup (the next test's) would remove the
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
    """mkdtemp runs before build_cfg, so a bad --bounds used to leave an empty
    %TEMP%\\prism_smoke_* behind forever, its path never printed anywhere an operator would look.
    main() now removes an auto-created root when the handler fails AND the root is still empty; a
    non-empty root (any run that got as far as writing anything)
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
    # The missing-file rule at the build: ONE refusal line naming the input kind and the flag, not the
    # parser's bare FileNotFoundError with a [raised at file_manager.py:...] hedge
    err = capsys.readouterr().err
    lines = [ln for ln in err.splitlines() if ln.startswith("prism smoke: refused:")]
    assert len(lines) == 1, err
    assert "The bounds file was not found" in lines[0] and lines[0].endswith("(--bounds)"), lines[0]
    assert "raised at" not in lines[0] and "FileNotFoundError" not in lines[0], lines[0]


def test_a_missing_cell_is_refused_at_the_read_naming_the_flag(tool_env, capsys):
    """The cell half of the missing-file rule, on a subcommand whose config build reads the truth
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
    """Smoke keys its store on --store-root, not PRISM_ARTIFACTS, so the generic
    "the same command with --resume require continues them" advice is wrong for it -- re-issuing the
    own command line of the first run, which builds the prior and saves it by name, just re-BUILDS
    the prior. Unit-tested directly against the extracted helper
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
    assert "--resume-fit" not in advice, "smoke takes no --resume-fit, so its advice never names it"
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
    """The same advice end to end: main()'s KeyboardInterrupt handler must pick the smoke-specific
    advice for smoke -- keyed on smoke's own ``args.temp_store_root`` property, neither on the
    subcommand name nor on the --store-root flag, which another subcommand could come to declare -- and must leave
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
    """Smoke checks what it can from its flags before the prior build.

    The cell used to be parsed first inside the infer stage, after the prior (~100 s), the training
    and the calibration; a mistyped one then exited 1 and left orphans. And a --resume that can never
    work (no --checkpoint, or require without --prior, whose fresh fit has a new prior_fingerprint)
    was refused only after the prior, by an orchestrator message naming a flag smoke does not have."""
    import re

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
    assert "an earlier --save run" in err and not re.search(r"\bruns? \d", err), err
    assert rec.calls == [], "an impossible --resume was refused only after the prior"


# ── the card test as a step of its own: the smoke command's four stages, called as the window will ──
def _card_stages(monkeypatch, cfg, *, writes=False):
    """Recorder stages on the orchestrator for the card test, each returning a stand-in of its record, and
    the seeding context replaced where the step reads it. With ``writes`` each stage first writes on the
    configuration it is handed, as a real stage writes on its own. Returns the calls by stage (and
    "seeds", the seeding context's), what each stage returns, and the width the mode conditions on."""
    import contextlib

    import torch

    import core.diagnostics.rng
    from core import orchestrator
    from core.SBI.statistics import SUMMARY_WIDTH
    want = SUMMARY_WIDTH + 1 + orchestrator.expected_forcing_dim(cfg)
    calls = {"prior": [], "posterior": [], "validate": [], "infer": [], "seeds": []}
    made = {"prior": "PRIOR", "posterior": "POST", "validate": "CAL",
            "infer": (SimpleNamespace(width=want, x_obs=torch.zeros(1)), "INF")}

    def stage(name):
        def call(*a, **k):
            calls[name].append((a, k))
            if writes:
                a[0].T_obs = 4242.0
                a[0].sources["cell"] = "written by the stage"
            return made[name]
        return call

    for name, attr in (("prior", "build_prior"), ("posterior", "build_posterior"),
                       ("validate", "validate_calibration"), ("infer", "simulated_inference")):
        monkeypatch.setattr(orchestrator, attr, stage(name))
    monkeypatch.setattr(core.diagnostics.rng, "seeded",
                        lambda seed, device: calls["seeds"].append((seed, device)) or contextlib.nullcontext())
    return calls, made, want


def _weak_cell() -> str:
    return str(config.CELL_PATH / "nadrowski" / "master_weak.txt")


def test_the_card_test_step_calls_the_four_stages_with_the_smoke_commands_keywords(monkeypatch):
    """The card test as one step, called as the window's screen will call it: the smoke command's four
    stages, each looked up on the orchestrator when it is called, with the keywords the command hands them
    (the sets the command's own keyword test pins), one distinct value per setting; the one seeding
    context, read where the command reads it; the command's defaults; and a result that holds what each
    stage returned, the seconds each took and the width the network conditions on. Every stage is handed
    the step's one working copy of the configuration, never the caller's, and no truth on it: the cell is
    checked on a copy of its own."""
    import inspect

    from core.diagnostics import card
    from tests._fixtures import _nad_cfg
    cfg = _nad_cfg()
    calls, made, want = _card_stages(monkeypatch, cfg)
    cell = _weak_cell()
    sink, store = object(), object()
    result = card.card_test(cfg, cell=cell, t_obs_s=2.5, seed=5, num_runs=8, run_size_cap=9, n_cal=17,
                            max_num_epochs=6, hidden_features=11, num_transforms=2, checkpoint=True, save=True,
                            prior_ref="smoke_prior", resume="require", new_run=True, fig_sink=sink, store=store)

    ((cfg1, ref1, build_new1), kw1), = calls["prior"]
    assert cfg1 is not cfg and not cfg1.has_ground_truth and "cell" not in cfg1.sources
    assert calls["seeds"] == [(5, cfg1.hw.device)]
    assert (ref1, build_new1) == ("smoke_prior", False) and kw1 == {"fig_sink": sink, "store": store, "name": ""}
    ((cfg2, prior2, ref2, train_new2), kw2), = calls["posterior"]
    assert cfg2 is cfg1 and (prior2, ref2, train_new2) == ("PRIOR", None, True)
    # The simulation set is committed every half of its batches: 4 of the 8, a value no setting here has.
    assert kw2 == {"fig_sink": sink, "store": store, "name": "smoke_posterior", "num_runs": 8, "run_size_cap": 9,
                   "max_num_epochs": 6, "checkpoint_every": 4, "new_run": True, "resume": "require",
                   "hidden_features": 11, "num_transforms": 2}
    ((cfg3, post3, prior3), kw3), = calls["validate"]
    assert cfg3 is cfg1 and (post3, prior3) == ("POST", "PRIOR")
    assert kw3 == {"fig_sink": sink, "store": store, "n_cal": 17}
    ((cfg4, post4, t_obs4), kw4), = calls["infer"]
    assert cfg4 is cfg1 and (post4, t_obs4) == ("POST", 2.5)
    assert kw4 == {"cell": cell, "prior": "PRIOR", "fig_sink": sink, "store": store}
    assert result.stages_run == card.STAGES and set(result.seconds) == set(card.STAGES)
    assert all(s >= 0.0 for s in result.seconds.values()), result.seconds
    assert (result.prior, result.posterior, result.calibration, result.inference) == ("PRIOR", "POST", "CAL", "INF")
    assert result.observation is made["infer"][0] and result.conditioning_width == want

    # At its defaults the step forwards no network size and no policy, so each stays the stage's own; it names
    # neither the prior it builds nor the posterior it trains; and every default is the smoke command's.
    for name in calls:
        calls[name].clear()
    card.card_test(cfg, cell=cell)
    (_c, ref, build_new), kw = calls["prior"][0]
    assert (ref, build_new, kw["name"]) == (None, True, ""), kw
    kw = calls["posterior"][0][1]
    assert kw["name"] == "" and not {"resume", "hidden_features", "num_transforms"} & set(kw), kw
    assert (kw["num_runs"], kw["run_size_cap"], kw["max_num_epochs"], kw["checkpoint_every"], kw["new_run"]) == \
        (4, 32, 5, 0, False)
    assert calls["validate"][0][1]["n_cal"] == 40 and calls["infer"][0][0][2] == config.T_MIN_EXP_S
    assert calls["seeds"][0][0] == 0
    # Told to save, it names both: the prior it builds and the posterior it trains.
    for name in calls:
        calls[name].clear()
    card.card_test(cfg, cell=cell, save=True)
    (_c, ref, build_new), kw = calls["prior"][0]
    assert (ref, build_new, kw["name"], calls["posterior"][0][1]["name"]) == \
        (None, True, "smoke_prior", "smoke_posterior")
    smoke = build_parser().subcommands["smoke"]
    step = inspect.signature(card.card_test).parameters
    for dest, keyword in (("seed", "seed"), ("num_runs", "num_runs"), ("run_size_cap", "run_size_cap"),
                          ("n_cal", "n_cal"), ("max_num_epochs", "max_num_epochs"), ("t_obs_s", "t_obs_s"),
                          ("hidden_features", "hidden_features"), ("num_transforms", "num_transforms"),
                          ("checkpoint", "checkpoint"), ("save", "save"), ("prior", "prior_ref"),
                          ("resume", "resume"), ("new_run", "new_run")):
        assert smoke.get_default(dest) == step[keyword].default, (dest, smoke.get_default(dest), step[keyword].default)
    assert ",".join(step["stages"].default) == smoke.get_default("stages")


def test_the_card_test_refuses_a_missing_cell_before_the_prior(monkeypatch, tmp_path):
    """The cell the inference simulates is read last, after the prior, the training and the calibration;
    the step checks it first, on a copy of its configuration, so a cell that names no file, or none at
    all, is refused before anything is spent, under the cell's own field. So is a list of stages naming one
    the card test does not have. A cell is read only when the inference is asked for, as the smoke command
    reads it."""
    from core.diagnostics import card
    from core.refusals import Refusal
    from tests._fixtures import _nad_cfg
    cfg = _nad_cfg()
    calls, _made, _want = _card_stages(monkeypatch, cfg)
    missing = str(tmp_path / "no_such_cell.txt")
    for cell in (missing, "", "   "):
        with pytest.raises(Refusal) as e:
            card.card_test(cfg, cell=cell)
        assert e.value.field == "cell", (cell, e.value.field, str(e.value))
    with pytest.raises(Refusal) as e:
        card.card_test(cfg, cell=_weak_cell(), stages=("prior", "bogus"))
    assert "bogus" in str(e.value) and e.value.field is None, str(e.value)
    assert all(v == [] for v in calls.values()), calls
    card.card_test(cfg, cell=missing, stages=("prior",))
    assert len(calls["prior"]) == 1, "a cell was read for a card test that does not infer"


def test_the_card_test_leaves_its_callers_configuration_as_it_found_it(monkeypatch, tmp_path):
    """The window's screen will hand the step a configuration it keeps. Each stage writes on the one it is
    handed; the step hands them a working copy of its own, so the caller's is as it was however the step
    ends: run through, refused before the prior, or failed in a stage."""
    from core import orchestrator
    from core.diagnostics import card
    from core.refusals import Refusal
    from tests._fixtures import _nad_cfg, assert_cfg_unchanged, snapshot_cfg
    cfg = _nad_cfg()
    calls, _made, _want = _card_stages(monkeypatch, cfg, writes=True)
    snap = snapshot_cfg(cfg)
    card.card_test(cfg, cell=_weak_cell())
    assert calls["infer"][0][0][0].T_obs == 4242.0, "the stages wrote nothing: this test would read nothing"
    assert_cfg_unchanged(cfg, snap)
    with pytest.raises(Refusal):
        card.card_test(cfg, cell=str(tmp_path / "no_such_cell.txt"))
    assert_cfg_unchanged(cfg, snap)

    def fail(c, *a, **k):
        c.T_obs = 4242.0
        raise RuntimeError("the calibration failed")

    monkeypatch.setattr(orchestrator, "validate_calibration", fail)
    with pytest.raises(RuntimeError, match="the calibration failed"):
        card.card_test(cfg, cell=_weak_cell())
    assert_cfg_unchanged(cfg, snap)


def test_the_card_test_tells_each_stage_once_in_order(monkeypatch, capsys, caplog):
    """The step tells its caller, on the calling thread, each stage once, in the order the stages run:
    "started" and then "finished" with the seconds it took, or "skipped" for a stage not asked for. A stage
    that fails is told as "failed", with its seconds, before the failure passes on, and nothing after it is
    told. With no prior there is nothing to train, and with no posterior nothing to calibrate or infer
    from: the step stops there, as the smoke command does, and tells nothing of the stages after. The step
    itself says nothing: no print and no record."""
    import logging
    import threading

    from core import orchestrator
    from core.diagnostics import card
    from tests._fixtures import _nad_cfg
    cfg = _nad_cfg()
    calls, _made, _want = _card_stages(monkeypatch, cfg)
    here, told = threading.get_ident(), []

    def on_stage(stage, event, seconds):
        assert threading.get_ident() == here
        told.append((stage, event, seconds))

    def run(**given):
        told.clear()
        result = card.card_test(cfg, cell=_weak_cell(), on_stage=on_stage, **given)
        for stage, event, seconds in told:
            assert (seconds is None) == (event in ("started", "skipped")), (stage, event, seconds)
            assert seconds is None or seconds >= 0.0, (stage, event, seconds)
        return result, [(stage, event) for stage, event, _ in told]

    caplog.set_level(logging.INFO, logger="core")
    caplog.clear()
    capsys.readouterr()
    result, events = run()
    assert events == [(s, e) for s in card.STAGES for e in ("started", "finished")], events
    assert [seconds for _s, e, seconds in told if e == "finished"] == [result.seconds[s] for s in card.STAGES]
    assert run(stages=("prior", "posterior"))[1] == [
        ("prior", "started"), ("prior", "finished"), ("posterior", "started"), ("posterior", "finished"),
        ("validate", "skipped"), ("infer", "skipped")]
    result, events = run(stages=("posterior", "validate"))
    assert events == [("prior", "skipped")] and result.prior is None and result.stages_run == ()
    result, events = run(stages=("prior",))
    assert events == [("prior", "started"), ("prior", "finished"), ("posterior", "skipped")]
    assert result.posterior is None and result.stages_run == ("prior",)

    def fail(*a, **k):
        raise RuntimeError("the calibration failed")

    monkeypatch.setattr(orchestrator, "validate_calibration", fail)
    told.clear()
    inferred = len(calls["infer"])
    with pytest.raises(RuntimeError, match="the calibration failed"):
        card.card_test(cfg, cell=_weak_cell(), on_stage=on_stage)
    assert [(s, e) for s, e, _ in told][-2:] == [("validate", "started"), ("validate", "failed")], told
    assert told[-1][2] >= 0.0 and len(calls["infer"]) == inferred, "a stage after the failure was run"
    assert capsys.readouterr() == ("", ""), "the step printed"
    assert [r.getMessage() for r in caplog.records] == [], "the step said something of its own"


def test_the_card_tests_stages_are_the_commands_stages():
    """The smoke command lists its stages without the step, so that reading its flags imports no torch: an
    unknown stage is refused while the flags are read, before anything is loaded. The two lists are one
    list, in one order. Checked in a fresh interpreter, since this one imported torch long ago."""
    import subprocess
    import sys

    from core.diagnostics import card
    from core.tool import smoke
    assert card.STAGES == smoke.STAGES == ("prior", "posterior", "validate", "infer")
    probe = ("import sys\n"
             "from core.tool import main\n"
             "rc = [main(['smoke', '--help']),\n"
             "      main(['smoke', '--bounds', 'b.txt', '--cell', 'c.txt', '--stages', 'prior,bogus'])]\n"
             "bad = sorted(m for m in sys.modules if m == 'torch' or m.startswith('torch.'))\n"
             "sys.exit(0 if rc == [0, 2] and not bad else repr((rc, bad[:3])))\n")
    r = subprocess.run([sys.executable, "-c", probe], cwd=str(config.REPO_ROOT),
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "--stages" in r.stdout and "bogus" in r.stderr, r.stdout + r.stderr


def test_the_card_test_refuses_a_continuation_it_cannot_make(monkeypatch):
    """A caller that did not check them first is refused, before the prior, the two continuations that can
    never work when the training is among the stages: a policy of 'require' or 'never' with no simulation
    set kept (no checkpoint), under the policy's field, and 'require' with a prior built afresh, whose new
    fit no kept set can match, under the prior's. Neither sentence names a flag. Without the training among
    the stages there is nothing to continue, and nothing is refused."""
    from core.diagnostics import card
    from core.refusals import Refusal
    from tests._fixtures import _nad_cfg
    cfg = _nad_cfg()
    calls, _made, _want = _card_stages(monkeypatch, cfg)
    for given, field in ((dict(resume="require", prior_ref="p"), "resume"), (dict(resume="never"), "resume"),
                         (dict(resume="require", checkpoint=True), "prior")):
        with pytest.raises(Refusal) as e:
            card.card_test(cfg, cell=_weak_cell(), **given)
        assert e.value.field == field, (given, e.value.field, str(e.value))
        assert "--" not in str(e.value), str(e.value)
    assert all(v == [] for v in calls.values()), calls
    card.card_test(cfg, cell=_weak_cell(), resume="require", checkpoint=True, prior_ref="p")
    card.card_test(cfg, cell=_weak_cell(), resume="never", stages=("prior",))
    assert len(calls["prior"]) == 2 and len(calls["posterior"]) == 1, calls


def test_the_tool_prints_a_refusal_with_its_flag_and_a_bug_with_a_traceback(tool_env, monkeypatch, capsys):
    """The tool's ladder: four rungs, told apart by TYPE.

    A Refusal is one operator line -- ``prism <cmd>: refused: <message> <fix>`` -- with the flag from
    core/tool/fields.py in parentheses when its field has one, and NOTHING after the message when
    the field is None or has no flag (fix_sentence owns the parentheses, so a bare message never
    ends in "()"). No class name, no [raised at ...]. A bare ValueError from a site not yet
    converted to a Refusal keeps today's hedged shape, class and location included, so an
    unconverted refusal still reads as a refusal. Anything else is a bug and prints the whole
    traceback. UsageError is now a Refusal too and MUST stay exit 2 with its own prefix: it is
    caught one rung earlier.

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

    # (c) a registered field with no flag (there is no chi override: the band is config.py's, fixed
    # by measurement): same as (b)
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
    """The console routing of logging records, on the command line. An information record is stdout,
    plain -- the GPU smoke gate reads ``[checkpoint] resuming at batch k/n`` off stdout and must keep
    doing so now that it is a record.
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


# The console lines the GPU smoke gate reads by eye, each with
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
    """The GPU smoke gate is read BY EYE off the console: the resume run (same store, same
    --num-runs, the prior loaded by name) must show "Reusing the Fisher rotation stored with the
    training checkpoint" and "[checkpoint] resuming at batch 4/4", the run one setting away no
    "[fisher]" line, the chi and forced runs no OOM line and the masked-probe run total, and the smoke
    driver frames every stage with "=== <stage> ===", "[ok] <stage> in Xs" and "[smoke] ALL
    STAGES COMPLETED". No suite assertion read most of these before, so the conversion to
    logging could have reworded one, or demoted an OOM notice to information on stdout, with every
    suite green and the gate silently unreadable.

    Two halves. The SOURCE: each line is carried in its own module by exactly one kind of call, the
    kind that puts it on its stream, and by no other. The ROUTE: under the tool's console handlers an
    information record lands on stdout as bare text and a warning record on stderr behind "warning: ",
    so a level in the source IS a stream on the command line. In the window the same level is the
    pane's plain line or triangle."""
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


# Every module whose in-stage prints were converted to logging records: file_manager.list_dir, the
# orchestrator, core/SBI and the solver, the diagnostics and the plot helpers.
_CONVERTED = ("core/orchestrator.py", "core/SBI", "core/Solvers/sdeint.py", "core/diagnostics",
              "core/Helpers/visualizers.py", "core/Helpers/file_manager.py")


def test_no_print_call_remains_in_the_converted_modules():
    """The definition of done for the conversion to logging: no print() call is left in a module whose
    messages are the pipeline's own voice. A print there reaches the window at the level of whichever
    STREAM it used, never reaches the artifact's log.txt, and on the command line ignores the
    stdout/stderr split -- the three defects the conversion exists to remove. Parsed, not grepped: a
    docstring or a comment that mentions print() is not a call. The tool's framing prints (core/tool)
    stay prints and are deliberately outside this set."""
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


def test_no_command_of_the_tool_passes_the_choice_for_a_taken_name():
    """A name another record takes while a command runs is refused at that record's write, as before: the
    tool never asks a stage to save the record without its name. Only the window asks, where the run's
    page can say so and offer to name it. Read on the text of every module of the tool, for the stages'
    own keyword."""
    import inspect
    from core import orchestrator
    root = Path(__file__).resolve().parents[1]
    modules = sorted((root / "core" / "tool").rglob("*.py"))
    assert modules, "core/tool/ holds no modules"
    assert "if_name_taken" in inspect.signature(orchestrator.build_posterior).parameters, \
        "the stages' keyword is spelled otherwise: this scan reads for nothing"
    said = [py.relative_to(root).as_posix() for py in modules if "if_name_taken" in py.read_text(encoding="utf-8")]
    assert said == [], f"the tool passes the choice for a taken name: {said}"


# ── the artifacts family ─────────────────────────────────────────────────────────────────────────


@pytest.fixture
def browse_store(tmp_path, monkeypatch):
    """A store holding one artifact of each of the nine kinds plus three unusable directories, with
    PRISM_ARTIFACTS pointing at it.

    ``config.artifacts_root()`` reads the variable on EVERY call and ``main``
    opens its store on it, so ``main(["artifacts", ...])`` reads exactly this root -- the same redirect
    test_the_sbc_subcommand_forwards_every_knob_as_a_keyword uses. Yields ``(root, ids)``, where
    ``ids`` is ``{kind: id}`` from tests/_fixtures.py::build_browse_store (the real
    writer at minimum size, seconds not minutes -- it must never reach for ``tiny_run``)."""
    from tests._fixtures import build_browse_store
    root = tmp_path / "A"
    ids = build_browse_store(root)
    monkeypatch.setenv("PRISM_ARTIFACTS", str(root))
    return root, ids


def test_the_artifacts_listing_shows_the_browsers_own_columns():
    """ONE FORMATTER PER FACT, kept honest across two front ends that cannot share code.
    The browser's ``cells_for``/``columns_for`` live in a Qt module, and importing it from
    ``core/tool/browse.py`` would pull PySide6 into ``python -m core --help`` -- so the tool keeps its
    own column list and the two COLUMN SETS are pinned against each other HERE rather than trusted to
    agree by eye. THIS TEST imports both front ends; the tool imports neither. A column added to the
    table without one added to the tool fails right here.

    The eight kinds are restated in the tool as a literal for the same reason (reading KIND_DIRS at
    parser-build time would import torch), so their order is pinned too."""
    from core.artifacts.store import KIND_DIRS
    from core.gui.screens import artifact_screen as ascreen
    from core.gui.widgets.artifact_table import columns_for
    from core.tool import browse

    assert browse.KINDS == tuple(KIND_DIRS), "the eight kinds, in KIND_DIRS order"
    assert set(browse.COLUMNS) == set(browse.KINDS), "every kind has a column set, and no other"
    for kind in browse.KINDS:
        # The tool's literal is the GUI's own spelling, lower-cased: one canonical column list,
        # Title-case in the window and lower-case in a terminal where the output may be piped.
        assert browse.COLUMNS[kind] == tuple(c.lower() for c in columns_for(kind)), kind
    # The THIRD deliberate duplication, and the only one that had no pin: the
    # sentence that tells an operator a training cache blocks a prior's delete because it was
    # GENERATED against it, rather than because it named it. Its own comment cites COLUMNS above as
    # the precedent for restating it -- so it is pinned the same way.
    assert browse._FINGERPRINT_DEPENDENT == ascreen._FINGERPRINT_DEPENDENT


def test_the_rendered_help_counts_the_kinds_correctly():
    """A SILENT pin. Five phrases in core/tool/browse.py hand-write how many kinds
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
    """The listing's table, per kind: a posterior's mode, width and amortization; a cache's progress
    as a FRACTION (``batches_done``/``batches_planned``) and -- separately -- whether it FINISHED
    (``complete`` means "has a valid manifest", which is true of a cache from its first batch on). Read
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
    ``identity["n_runs"]``) makes the cell a FRACTION, and ``rows`` is written only by
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


def test_artifacts_list_with_no_kind_covers_every_kind_under_a_heading(browse_store, capsys):
    """Every kind the tool lists -- read off ``browse.KINDS`` (pinned equal to the
    store's own kinds by test_the_artifacts_listing_shows_the_browsers_own_columns), never a literal
    that had fallen one kind behind."""
    from core.tool import browse
    root, ids = browse_store
    capsys.readouterr()
    assert main(["artifacts", "list"]) == 0
    out = capsys.readouterr().out
    assert "fdt" not in browse.KINDS
    for kind in browse.KINDS:
        assert f"== {kind} ==" in out, kind
    assert "nothing in:" not in out, "build_browse_store writes one artifact of every kind"


def test_artifacts_list_fit_prints_what_each_kept_training_is_for_its_passes_and_its_state(browse_store,
                                                                                          capsys):
    """A kept training's row: its label (always unnamed), when it began, the posterior it is to write
    ("-" when none is named yet), the passes it holds, its state's word and its note -- the window's
    columns, lower-cased. A record that says none of the three prints "-" for each, never "None"."""
    import re
    from core.artifacts.store import Summary
    from core.tool import browse
    root, ids = browse_store
    capsys.readouterr()
    assert main(["artifacts", "list", "fit"]) == 0
    out = capsys.readouterr().out
    assert "== fit ==" in out, out
    header = next(ln for ln in out.splitlines() if ln.split()[:1] == ["name"])
    assert header.split() == ["name", "created", "for", "passes", "state", "note"], header
    line = next(ln for ln in out.splitlines() if f"(unnamed {ids['fit']})" in ln)
    cells = re.split(r"\s{2,}", line.strip())
    assert cells[0] == f"(unnamed {ids['fit']})" and cells[2:] == ["-", "3", "kept when stopped",
                                                                    "the fit row"], cells
    bare = Summary(kind="fit", id="k", name="", created="2026-01-01 00:00:00", note="", path=Path("."),
                   complete=True, reason=None)
    assert browse._cells("fit", bare)[2:5] == ("-", "-", "-"), browse._cells("fit", bare)


def test_an_empty_artifacts_listing_exits_0_and_a_bad_kind_exits_1(tmp_path, monkeypatch, capsys):
    """An empty listing is 0, with a line saying there is nothing there. A script must be able to
    tell "nothing on disk" from "you asked for something wrong", and exiting 1 on an empty store would
    make the two indistinguishable. A kind that does not exist IS the second case: the store's own
    refusal, through the ladder's ``refused:`` rung at exit 1.

    The empty root need not exist beforehand -- ``main`` mkdirs the store root before any handler runs
    (core/tool/__init__.py, ``main``)."""
    monkeypatch.setenv("PRISM_ARTIFACTS", str(tmp_path / "empty"))

    capsys.readouterr()
    assert main(["artifacts", "list"]) == 0
    out = capsys.readouterr().out
    named_empty = out.split("nothing in:")[1]
    from core.tool import browse
    for kind in browse.KINDS:
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
    the row carries the ``dir_name`` because that is the only handle ``sweep`` can take.

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
    """The manifest through the ONE renderer both front ends use, then the records with
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
    """The two honest-gap sentences, WORD FOR WORD the browser's, because one operator reads both. A
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
    ``show`` is the first place that is visible.

    A simulation cache's folder is its digest whatever its name -- a rename writes its manifest and never
    moves its folder -- so a cache folder placed or renamed by hand is legitimate and gets no such note;
    the browser's detail pane leaves it out for the same reason."""
    from core.artifacts import ArtifactStore
    root, ids = browse_store
    sub = ArtifactStore(root).path("calibration", ids["calibration"])
    sub.rename(sub.with_name("stale__" + sub.name.rsplit("__", 1)[-1]))

    capsys.readouterr()
    assert main(["artifacts", "show", "calibration", ids["calibration"]]) == 0
    out = capsys.readouterr().out
    assert "stale__" in out and "while its manifest says" in out, out

    cache = ArtifactStore(root).path("simulation", ids["simulation"])
    cache.rename(cache.with_name("placed_by_hand"))
    assert ArtifactStore(root).path("simulation", ids["simulation"]).name == "placed_by_hand"
    assert main(["artifacts", "show", "simulation", ids["simulation"]]) == 0
    out = capsys.readouterr().out
    assert "while its manifest says" not in out, out


def test_artifacts_show_on_a_missing_ref_exits_1_through_the_refused_rung(browse_store, capsys):
    """A missing or ambiguous ref is 1, through the existing ``refused:`` rung. The message
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
    """The artifacts family's two halves, pinned. (1) No configuration flags anywhere in the family:
    a listing must not be able to fail on a bounds file it does not need, and --store-root in
    particular is absent because this family creates no second root and reads the one
    PRISM_ARTIFACTS names. It used to be absent for a second reason as well -- ``main`` keyed smoke's
    temp root, its empty-root cleanup and its Ctrl-C advice on ``hasattr(args, "store_root")`` --
    which was retired when those three became an explicit per-subcommand choice. (2) ``--help``
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
    """The note is one line, trimmed, and a blank one CLEARS it. ``--note`` is a flag rather than
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
    """No clamp and no silent default on the note. An over-long note is refused with BOTH numbers
    in the sentence, a newline is refused rather than flattened, and the manifest is untouched in
    either case. The flag comes from core/tool/fields.py's table, appended by main's ladder -- and
    only for a refusal about the note TEXT, which is ``require_note``'s (field="note"). The third leg
    is the other refusal this mode can raise: ``set_note``'s own "no complete artifact", which carries
    field="artifact", a key whose flag is None -- so that line ends at the message."""
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

    # A ref that names no artifact is a DIFFERENT refusal: set_note's, with field="artifact", whose
    # entry in core/tool/fields.py is None because the tool names the artifact
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
    """Delete is one artifact at a time and there is NO force in either front end, so the store's
    refusal is the last word. A diagnostic is the leaf case -- ``_PARENT_KEYS["diagnostic"]`` is
    empty, so nothing can ever name one as a parent."""
    import argparse
    from core.artifacts import ArtifactStore, StoreError
    root, ids = browse_store
    modes = {name: sub for a in build_parser().subcommands["artifacts"]._actions
             if isinstance(a, argparse._SubParsersAction) for name, sub in a.choices.items()}
    assert "--force" not in modes["rm"]._option_string_actions, "no force in either front end"

    s = ArtifactStore(root)
    gone = s.path("diagnostic", ids["diagnostic"])
    capsys.readouterr()
    assert main(["artifacts", "rm", "diagnostic", ids["diagnostic"]]) == 0
    assert str(gone) in capsys.readouterr().out
    assert not gone.exists()
    with pytest.raises(StoreError):
        s.get("diagnostic", ids["diagnostic"])


def test_artifacts_rm_refuses_an_artifact_something_depends_on(tool_run, capsys):
    """The other half of that rule: because no force is offered, an artifact anything depends on
    cannot be deleted AT ALL, and the refusal names every dependent -- in the TOOL's OWN WORDS, never
    the store's raw sentence, which ends "pass force=True to orphan them": a step
    nothing in either front end offers, so it must never reach an operator.

    ``tool_run`` is MODULE-scoped, so every test in this file shares one store, and
    ``tp`` -- the one prior every SBI test in this module trains against -- accumulates a dependent
    for every posterior, calibration and cache an EARLIER-running test in this module built against
    it (``test_validate_and_simulated_infer``'s ``tcal`` among them). A hard-coded "2 artifact(s)"
    was true only when this test happened to run first, or alone -- so the expected COUNT and the
    expected IDS are read off ``s.dependents`` itself, which is the one ground truth that holds no
    matter what the rest of the module has built by the time this runs.

    Reading the count and the ids off ``s.dependents`` must not drop the assertion on the REASON
    clause: with only the ids checked, deleting ``-- {why}`` from the f-string would leave the suite
    green and the operator without the one sentence that says WHY a training cache blocks a prior's
    delete. So it is pinned here order-independently: every dependent's id is followed by
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
    """One action removes every directory of a kind that has NO manifest at all,
    through a store call that CAN ONLY remove such a directory -- ``delete`` resolves through
    ``_find``, which never returns a manifest-less entry, and ``remove_incomplete`` is its
    complement. Nothing to remove is exit 0, and it says so rather than printing nothing.

    The window ASKS before it removes and this did not -- no preview, no confirmation, and a
    trial run deleted two directories and printed their reasons afterwards. So the default is
    a DRY RUN: exactly what it would remove, removing nothing, and ``--yes`` performs it. That is a
    confirmation and not an override: the no-force rule stands, there is still no ``--force`` and no
    way to reach a real artifact from here.

    Of the three shapes ``build_browse_store`` seeds under ``priors/`` (no manifest at all, a
    manifest that will not parse, a valid manifest of another kind), only the FIRST is a leftover a
    sweep may remove. The other two are reported and survive -- a manifest valid under a different
    SCHEMA reads as "no artifact here" to this build, and deleting one cost a trial run a real
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
    backdate_tree(root / "priors")          # the recency guard is tested at the store, not here

    # (a) the DRY RUN: says what it would remove, removes nothing, exits 0
    capsys.readouterr()
    assert main(["artifacts", "sweep", "prior"]) == 0
    out = capsys.readouterr().out
    assert "would remove" in out and leftover.name in out and manifest_less.name in out, out
    assert "--yes" in out, "the dry run must name the flag that performs it"
    assert "still being written" in out, \
        ("a preview run beside a live training would otherwise offer that training's own directory "
         "with no warning: the listing cannot tell a run in flight from a leftover")
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
    # sweep_incomplete's own return has no reason, so the tool reads it off
    # `list` before removing and prints it after -- an operator must be able to tell "no manifest"
    # from "a manifest of another kind" after the fact, not just which directory went.
    assert "no manifest.json" in out, out
    assert "manifest declares kind" in out, out

    capsys.readouterr()
    assert main(["artifacts", "sweep", "prior", "--yes"]) == 0
    assert "nothing to sweep" in capsys.readouterr().out


def test_artifacts_sweep_offers_a_loose_file_and_never_a_records_payload(browse_store, capsys):
    """The tidy-up's first category, the loose file. ``ArtifactStore._entries`` iterates DIRECTORIES
    only, so a file sitting directly inside a kind directory is invisible to every listing in both
    front ends; an older build's run once left two PNGs that way, in a kind directory. They
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
    """The second tidy-up category, and why it has its own form. A legacy directory -- a `crossval/`
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


def test_a_sweep_claims_nothing_it_could_not_read_and_its_help_names_every_category(
        tmp_path, monkeypatch, capsys):
    """The tool's half. "nothing to sweep: ... and no legacy directory sits beside them" printed
    even when the store root could not be read -- a line that contradicted the error printed beside
    it. Each clause is printed only when its read succeeded. The dry-run footnote named only "a
    directory that is still being written", though a recently written loose file and the legacy
    directory are refused too, and ``artifacts sweep --help`` never mentioned the legacy directory
    at all."""
    from core.artifacts import ArtifactStore

    root = tmp_path / "empty_root"
    root.mkdir()
    monkeypatch.setenv("PRISM_ARTIFACTS", str(root))

    def _unreadable(self):
        raise PermissionError(13, "Access is denied", str(self.root))

    monkeypatch.setattr(ArtifactStore, "legacy_dirs", _unreadable)
    capsys.readouterr()
    assert main(["artifacts", "sweep"]) == 1, "an unread root is a problem, not an empty one"
    cap = capsys.readouterr()
    assert "no legacy directory was swept" in cap.err, cap.err
    (line,) = [ln for ln in cap.out.splitlines() if "nothing to sweep" in ln]
    assert "legacy" not in line, line
    assert "every directory here carries a manifest.json" in line, "the reads that worked still say so"

    capsys.readouterr()
    assert main(["artifacts", "sweep", "--help"]) == 0
    assert "legacy directory" in " ".join(capsys.readouterr().out.split())

    # the dry run's footnote, printed under a loose FILE it previews, speaks of files too
    from tests._fixtures import backdate_tree
    stray = root / "priors" / "fdt_ratio_20260915_153042.png"
    stray.parent.mkdir()
    stray.write_bytes(b"a picture an older build left beside the records")
    backdate_tree(root / "priors")
    capsys.readouterr()
    assert main(["artifacts", "sweep", "prior"]) == 0
    out = capsys.readouterr().out
    assert f"would remove prior loose file {stray.name}" in out, out
    assert "a loose file, the legacy directory -- is refused at removal" in out, out


def test_a_loose_file_written_seconds_ago_is_refused_rather_than_swept(browse_store, capsys):
    """The recency guard, extended to the new categories. A file inside a kind directory can be a
    run in flight writing its own figure as easily as it can be a leftover -- and a sweep from a
    second process cannot ask the window's run controller whether a run is live. The removal
    refuses it, the tool reports it, and the exit code says the sweep did not do everything it
    offered."""
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


def test_a_sweep_that_could_not_remove_something_exits_1_naming_it(browse_store, monkeypatch, capsys):
    """A sweep that removed everything is 0; one that could not remove a directory is 1, naming
    each failure. A held handle on Windows is what that stands for and it cannot be provoked on
    demand, so the two lists are injected on the store's own method -- what is under test is the
    ladder, not the removal.

    The injection also pins that method's CONTRACT: it is handed the list of
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
    """The lineage report is a FILE, never a store kind of its own, and it comes out of the same
    renderer the browser's "Lineage report..." writes -- so the document a reviewer receives is the
    same whichever front end made it. ``--out`` writes exactly what stdout would have
    carried.

    The file is compared as BYTES, not as text: read_text would translate CRLF back to LF on the way
    in and pass whatever newline=None had written, which is precisely the drift the one-renderer rule
    forbids and the GUI's own report test pins the same way."""
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
    """One line per rung: a usage error is 2 (argparse, one rung earlier than every refusal); a
    missing ref, a bad kind and a bad note are 1; an empty listing and a sweep with nothing to remove
    are 0. ``--note`` being REQUIRED is part of this: a note must never be cleared by omission.

    There is no exit 3 anywhere in the tool and this family adds none: a BUG is the existing unhandled
    rung, which sets 1 with a traceback, so 1 covers a refusal and a bug alike (``main``'s own
    docstring: "1 a refusal or a bug") and this family does not touch the ladder."""
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
    """The closure of the family's list, and the extension of the no-configuration-flags pin above to
    the four modes that write:
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
    # The sweep's confirmation. Only the sweep has it -- it is not a global "don't ask me", and
    # nothing else in this family removes anything without naming one artifact.
    assert {name for name, m in modes.items() if "--yes" in m._option_string_actions} == {"sweep"}


# ── the artifacts family: a failed write, a leftover's name, the shared sentences ────────────────


def test_artifacts_summary_write_failure_is_refused_not_a_crash(browse_store, tmp_path, capsys):
    """The window wraps the identical write and reports it (``artifact_screen._lineage_report``'s
    own ``except OSError``); an operator's ``--out`` typo -- a directory, a read-only file -- must
    not escape ``write_text`` as an unhandled traceback and a failure banner. Writing to a DIRECTORY
    is the OSError every platform raises for free, no fixture needed."""
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
    """A ref copied straight off ``list``'s own "incomplete ..." row cannot resolve through
    ``_find`` (which never returns a manifest-less entry), and the store's "no complete artifact"
    refusal said nothing about why, or what removes it -- unlike ``show``, which already states both
    honest gaps. ``note`` and ``rm`` now name ``sweep``, but ONLY for a ref that actually names one
    of THIS kind's leftovers; an ordinary typo still gets the plain refusal, with nothing invented
    about it.

    The sentence is built as ``exc.message + hint``, and the
    store's message ENDS with the ref while the hint once BEGAN with it -- so the ref printed twice, back
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
    """The comment above ``_show``'s ``print`` claims its two "no log" sentences are the browser's
    OWN CONSTANTS, word for word -- and that claim drifted once already (a stale copy was fixed
    silently, with no test to catch it). Pinned the way the column table already is
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


# ── the tool's root sink ─────────────────────────────────────────────────────────────────────────
def test_main_installs_the_root_handler_for_the_run_and_leaves_nothing_behind(tool_env, monkeypatch,
                                                                             capsys):
    """Every record from a logger OUTSIDE the ``core`` tree goes to STDERR, whatever its level,
    prefixed with the logger that said it. Deliberately unlike the window's level split: this tool's
    STDOUT carries results a script reads (the GPU smoke gate greps ``[checkpoint] resuming
    at batch`` off it), so a library's chatter may never land there.

    ``logging.warning`` -- the module-level function, on the ROOT logger -- is sbi's own shape
    (sbi/samplers/rejection/rejection.py:336,359), and ``root`` names nothing, so for that ONE logger
    name the prefix falls back to ``record.module``, the basename of the file that logged:
    ``library: rejection:`` under sbi, and this test file's own stem here (the ``logging.warning``
    below is called from ``_prior``, which lives in this file). The module name is what a user reads
    on the console.

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
    """A ``core`` record reaches the tool's root sink only when the console handlers are
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


def _probe_band_argv():
    """``probes band`` on the master box and its spontaneous cell, CPU."""
    bounds = str(config.BOUNDS_PATH / "nadrowski" / "master.txt")
    cell = str(config.CELL_PATH / "nadrowski" / "master_spont.txt")
    return ["probes", "band", "--bounds", bounds, "--device", "cpu", "--cell", cell]


def test_probes_band_builds_a_chi_config_with_the_cell_and_forwards_every_knob(tmp_path, monkeypatch):
    """Every flag reaches the stage as its keyword, a list flag as a list of floats, and nothing else
    does; a bare call forwards only the four the handler always passes, so every default stays in the
    stage's signature. The positional config is in chi mode with the cell loaded, and its band and drive
    are config.py's: nothing on the command line reaches them."""
    from core import tool
    monkeypatch.setenv("PRISM_ARTIFACTS", str(tmp_path / "A"))
    calls = []

    def _rec(*a, **kw):
        calls.append((a, kw))
        return SimpleNamespace(kind="diagnostic", path=tmp_path / "diagnostics" / "b1__1")

    monkeypatch.setattr("core.diagnostics.probe_band", _rec)
    assert tool.main([*_probe_band_argv(), "--lengths", "1", "2.5", "--multipliers", "0.05", "0.3",
                      "--drives", "0.2", "--repeats", "6", "--cycle-caps", "8", "12", "--cv-max", "0.25",
                      "--phase-max", "0.6", "--snr-min", "2.5", "--sup-min", "0.4", "--peak-window", "0.08",
                      "--seed", "3", "--name", "b1", "--note", "hello"]) == 0
    (args, kw), = calls
    assert set(kw) == {"name", "note", "fig_sink", "store", "lengths", "multipliers", "drives", "repeats",
                       "cycle_caps", "cv_max", "phase_max", "snr_min", "sup_min", "peak_window", "seed"}
    lists = {k: kw[k] for k in ("lengths", "multipliers", "drives", "cycle_caps")}
    assert lists == {"lengths": [1.0, 2.5], "multipliers": [0.05, 0.3], "drives": [0.2], "cycle_caps": [8.0, 12.0]}
    assert all(isinstance(v, float) for vs in lists.values() for v in vs), lists
    assert (kw["repeats"], kw["cv_max"], kw["phase_max"], kw["snr_min"], kw["sup_min"], kw["peak_window"],
            kw["seed"], kw["name"], kw["note"]) == (6, 0.25, 0.6, 2.5, 0.4, 0.08, 3, "b1", "hello")
    cfg = args[0]
    assert cfg.chi_mode is True and cfg.ground_truth is not None
    assert (cfg.chi_f0, cfg.chi_freq_bounds) == (config.CHI_F0, config.CHI_FREQ_BOUNDS)

    calls.clear()
    assert tool.main(_probe_band_argv()) == 0
    (_, kw), = calls
    assert set(kw) == {"name", "note", "fig_sink", "store"}, kw


def _probe_mask_argv(prior="mp"):
    """``probes mask`` on the master box against a prior by reference, CPU."""
    bounds = str(config.BOUNDS_PATH / "nadrowski" / "master.txt")
    return ["probes", "mask", "--bounds", bounds, "--device", "cpu", "--prior", prior]


def test_the_probes_family_keeps_its_modes_apart_and_its_help_imports_no_torch(tmp_path, monkeypatch, capsys):
    """The family's modes take no chi flag and no flag of another mode: band and mask build their own
    chi configuration, drive builds one with chi mode off, and the band and drive any of them judges are
    config.py's. No parser in the family matches an abbreviated option, so a prefix of one of its flags
    is an error rather than that flag -- a habitual --chi-k on mask is refused, never read as
    --chi-k-fixed. Ctrl-C gets the family's own note -- nothing is kept and nothing resumes. ``--help``
    costs no torch import, checked in a FRESH interpreter because this process imported torch long ago."""
    import argparse
    import subprocess
    import sys

    from core.tool import probes as tool_probes
    monkeypatch.setenv("PRISM_ARTIFACTS", str(tmp_path / "A"))
    p = build_parser().subcommands["probes"]
    modes = {name: sub for a in p._actions if isinstance(a, argparse._SubParsersAction)
             for name, sub in a.choices.items()}
    assert set(modes) == {"band", "mask", "drive"}, sorted(modes)
    assert p.allow_abbrev is False and all(m.allow_abbrev is False for m in modes.values())
    band, mask, drive = modes["band"], modes["mask"], modes["drive"]
    for flag in ("--chi", "--no-chi", "--chi-k", "--chi-f0", "--chi-band", "--f0", "--prior", "--posterior",
                 "--num-runs", "--strengths", "--t-obs", "--detune", "--free-min", "--captured-max",
                 "--clarity-min"):
        assert flag not in band._option_string_actions, flag
    for flag in ("--chi", "--no-chi", "--chi-k", "--chi-f0", "--chi-band", "--f0", "--cell", "--lengths",
                 "--strengths", "--posterior", "--t-obs", "--detune", "--peak-window", "--repeats"):
        assert flag not in mask._option_string_actions, flag
    for flag in ("--chi", "--no-chi", "--chi-k", "--chi-f0", "--chi-band", "--f0", "--lengths", "--multipliers",
                 "--drives", "--cycle-caps", "--cv-max", "--phase-max", "--snr-min", "--sup-min", "--prior",
                 "--posterior", "--num-runs", "--run-size", "--chi-k-fixed"):
        assert flag not in drive._option_string_actions, flag
    for mode in (band, mask):
        assert mode.get_default("chi_mode") is True
    assert drive.get_default("chi_mode") is False and drive.get_default("chi_n_freqs") is None
    for mode in (band, mask, drive):
        assert mode.get_default("interrupt_note") == tool_probes.PROBES_INTERRUPT_NOTE
    assert drive._option_string_actions["--t-obs"].dest == "t_obs_s"
    # the probe count mask can hold is training's own range, from the floor of its per-batch draw
    assert f"from {config.CHI_K_MIN_TRAIN} to" in mask._option_string_actions["--chi-k-fixed"].help
    # mask's bounds file is checked against the prior it loads; every other subcommand keeps its sentence
    from core.tool import config_args
    mask_bounds = mask._option_string_actions["--bounds"].help
    assert "the prior" in mask_bounds and "posterior" not in mask_bounds, mask_bounds
    assert (band._option_string_actions["--bounds"].help == config_args.BOUNDS_HELP
            == build_parser().subcommands["train"]._option_string_actions["--bounds"].help)
    # drive loads no posterior: its bounds file must declare the Forcing section it suggests values for
    drive_bounds = drive._option_string_actions["--bounds"].help
    assert "Forcing section" in drive_bounds and "posterior" not in drive_bounds, drive_bounds

    capsys.readouterr()
    assert main(["probes"]) == 2
    err = capsys.readouterr().err
    assert "{band,mask,drive}" in err and "variant" not in err, "a bare probes names its modes, not the dest"
    assert main([*_probe_drive_argv(), "--no-chi"]) == 2
    assert main([*_probe_drive_argv(), "--strength", "0.1"]) == 2, "a prefix of --strengths is not --strengths"
    assert main([*_probe_drive_argv(), "--lengths", "1"]) == 2
    assert main([*_probe_band_argv(), "--chi"]) == 2
    assert main([*_probe_band_argv(), "--sup", "0.4"]) == 2, "a prefix of --sup-min is not --sup-min"
    assert main([*_probe_band_argv(), "--cycle", "8"]) == 2, "a prefix of --cycle-caps is not --cycle-caps"
    assert main([*_probe_mask_argv(), "--chi-k", "6"]) == 2, "--chi-k is not a prefix of --chi-k-fixed"
    assert main([*_probe_mask_argv(), "--chi", "6"]) == 2

    def _interrupted(*a, **k):
        raise KeyboardInterrupt

    monkeypatch.setattr("core.diagnostics.probe_band", _interrupted)
    capsys.readouterr()
    assert main(_probe_band_argv()) == 130
    err = capsys.readouterr().err
    assert err.startswith("prism probes: interrupted: a probe check writes its record only when it "
                          "finishes"), err
    assert "--resume" not in err, err

    probe = ("import sys\n"
             "from core.tool import main\n"
             "rc = [main(['probes', '--help']), main(['probes', 'band', '--help']),\n"
             "      main(['probes', 'mask', '--help']), main(['probes', 'drive', '--help'])]\n"
             "bad = sorted(m for m in sys.modules if m == 'torch' or m.startswith('torch.'))\n"
             "sys.exit(0 if rc == [0, 0, 0, 0] and not bad else repr((rc, bad[:3])))\n")
    r = subprocess.run([sys.executable, "-c", probe], cwd=str(config.REPO_ROOT),
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "--cycle-caps" in r.stdout and "band" in r.stdout and "--chi-k-fixed" in r.stdout, r.stdout
    assert "--clarity-min" in r.stdout and "drive" in r.stdout, r.stdout


def test_each_probes_mode_takes_exactly_its_own_flags():
    """Past the framing every subcommand shares -- the configuration, the name and note, the seed and
    help -- each mode's flags are exactly its own, so a flag registered on the wrong mode fails here by
    name. --cell, --repeats and --peak-window belong to band and drive, never to mask. The drive
    strengths' default phrase states the fixed grid it describes."""
    import argparse
    from core.diagnostics import probes
    from core.tool.help_defaults import BEHAVIOUR, DEFAULT_CLASS
    framing = {"-h", "--help", "--bounds", "--model", "--device", "--name", "--note", "--seed"}
    p = build_parser().subcommands["probes"]
    modes = {name: sub for a in p._actions if isinstance(a, argparse._SubParsersAction)
             for name, sub in a.choices.items()}
    own = {name: set(sub._option_string_actions) - framing for name, sub in modes.items()}
    assert own == {
        "band": {"--cell", "--lengths", "--multipliers", "--drives", "--repeats", "--cycle-caps", "--cv-max",
                 "--phase-max", "--snr-min", "--sup-min", "--peak-window"},
        "mask": {"--prior", "--num-runs", "--run-size", "--chi-k-fixed"},
        "drive": {"--cell", "--t-obs", "--repeats", "--detune", "--strengths", "--free-min", "--captured-max",
                  "--peak-window", "--clarity-min"},
    }, own
    assert all(framing - {"-h", "--help"} <= set(sub._option_string_actions) for sub in modes.values())
    # the drive's run count and clarity threshold say what pure noise does to the clarity, and its
    # recording length that one the time grid cannot hold is refused
    helps = {flag: modes["drive"]._option_string_actions[flag].help
             for flag in ("--repeats", "--clarity-min", "--t-obs")}
    assert "pure noise" in helps["--repeats"].lower() and "pure noise" in helps["--clarity-min"].lower(), helps
    assert "time grid" in helps["--t-obs"] and "refused" in helps["--t-obs"], helps
    assert DEFAULT_CLASS[("probes drive", "--strengths")] == (
        BEHAVIOUR, "sixteen from 0.01 to 20, plus the configured chi drive")
    grid = probes._DRIVE_STRENGTHS
    assert (len(grid), grid[0], grid[-1]) == (16, 0.01, 20.0) and list(grid) == sorted(set(grid))


def _probe_drive_argv():
    """``probes drive`` on the master box and its spontaneous cell, CPU."""
    bounds = str(config.BOUNDS_PATH / "nadrowski" / "master.txt")
    cell = str(config.CELL_PATH / "nadrowski" / "master_spont.txt")
    return ["probes", "drive", "--bounds", bounds, "--device", "cpu", "--cell", cell]


def test_probes_drive_builds_a_non_chi_config_with_the_cell_and_forwards_every_knob(tmp_path, monkeypatch):
    """Every flag reaches the stage as its keyword -- --t-obs as t_obs_s, the strengths as a list of
    floats -- and nothing else does; a bare call forwards only the four the handler always passes, so
    every default stays in the stage's signature. The positional config has chi mode off, the cell
    loaded, and config.py's chi drive and band: nothing on the command line reaches them."""
    from core import tool
    monkeypatch.setenv("PRISM_ARTIFACTS", str(tmp_path / "A"))
    calls = []

    def _rec(*a, **kw):
        calls.append((a, kw))
        return SimpleNamespace(kind="diagnostic", path=tmp_path / "diagnostics" / "d1__1")

    monkeypatch.setattr("core.diagnostics.probe_drive", _rec)
    assert tool.main([*_probe_drive_argv(), "--t-obs", "2.5", "--repeats", "6", "--detune", "1.3",
                      "--strengths", "0.05", "0.5", "4", "--free-min", "0.6", "--captured-max", "0.2",
                      "--peak-window", "0.03", "--clarity-min", "4.5", "--seed", "3", "--name", "d1",
                      "--note", "hello"]) == 0
    (args, kw), = calls
    assert set(kw) == {"name", "note", "fig_sink", "store", "t_obs_s", "repeats", "detune", "strengths",
                       "free_min", "captured_max", "peak_window", "clarity_min", "seed"}
    assert kw["strengths"] == [0.05, 0.5, 4.0] and all(isinstance(v, float) for v in kw["strengths"])
    assert (kw["t_obs_s"], kw["repeats"], kw["detune"], kw["free_min"], kw["captured_max"], kw["peak_window"],
            kw["clarity_min"], kw["seed"], kw["name"], kw["note"]) == (2.5, 6, 1.3, 0.6, 0.2, 0.03, 4.5, 3, "d1",
                                                                       "hello")
    cfg = args[0]
    assert cfg.chi_mode is False and cfg.has_forcing and cfg.ground_truth is not None
    assert (cfg.chi_f0, cfg.chi_freq_bounds) == (config.CHI_F0, config.CHI_FREQ_BOUNDS)

    calls.clear()
    assert tool.main(_probe_drive_argv()) == 0
    (_, kw), = calls
    assert set(kw) == {"name", "note", "fig_sink", "store"}, kw
    assert tool.main([*_probe_drive_argv()[:-2]]) == 2 and len(calls) == 1, "--cell is required"


def test_probes_mask_loads_its_prior_by_reference_and_forwards_every_knob(tmp_path, monkeypatch):
    """The prior is loaded from the store by name, never built: an unknown one is refused and nothing
    is written. Every flag reaches the stage as its keyword, a bare call forwards only the four the
    handler always passes, and the configuration is in chi mode with no cell."""
    from core import tool
    from core.artifacts import ArtifactStore
    from tests._fixtures import _nad_cfg, _prior_artifact
    monkeypatch.setenv("PRISM_ARTIFACTS", str(tmp_path / "A"))
    _prior_artifact(ArtifactStore(tmp_path / "A"), _nad_cfg(chi_mode=True), name="mp")
    calls = []

    def _rec(*a, **kw):
        calls.append((a, kw))
        return SimpleNamespace(kind="diagnostic", path=tmp_path / "diagnostics" / "m1__1")

    monkeypatch.setattr("core.diagnostics.probe_mask", _rec)
    assert tool.main([*_probe_mask_argv(), "--num-runs", "3", "--run-size", "5", "--chi-k-fixed", "6",
                      "--seed", "4", "--name", "m1"]) == 0
    (args, kw), = calls
    assert set(kw) == {"name", "note", "fig_sink", "store", "num_runs", "run_size", "chi_k_fixed", "seed"}
    assert (kw["num_runs"], kw["run_size"], kw["chi_k_fixed"], kw["seed"], kw["name"]) == (3, 5, 6, 4, "m1")
    cfg, prior = args
    assert prior.name == "mp" and cfg.chi_mode is True and not cfg.inits_dict
    assert (cfg.chi_f0, cfg.chi_freq_bounds) == (config.CHI_F0, config.CHI_FREQ_BOUNDS)

    calls.clear()
    assert tool.main(_probe_mask_argv()) == 0
    (_, kw), = calls
    assert set(kw) == {"name", "note", "fig_sink", "store"}, kw

    calls.clear()
    assert tool.main(_probe_mask_argv("nope")) == 1 and calls == []
    assert [d.name.split("__")[0] for d in (tmp_path / "A" / "priors").iterdir()] == ["mp"]
    assert tool.main([*_probe_mask_argv(), "--cell", "x"]) == 2 and calls == []


def _grid_leaks(tree) -> list:
    """``[(lineno, what)]`` for every way a probe grid could reach training in a parsed module: a
    ``replace``/``dataclasses.replace`` or ``make_sim_config`` call, a ``chi_f0``/``chi_freq_bounds``
    keyword anywhere but ``gen_training_data`` with config's own value, and an assignment to a
    ``chi_``/``CHI_`` attribute."""
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            fn = ast.unparse(node.func)
            if fn in ("replace", "dataclasses.replace") or fn.endswith("make_sim_config"):
                found.append((node.lineno, f"calls {fn}"))
            for kw in node.keywords:
                if kw.arg in ("chi_f0", "chi_freq_bounds") and not (
                        fn.endswith("gen_training_data") and ast.unparse(kw.value) == f"cfg.{kw.arg}"):
                    found.append((node.lineno, f"passes {kw.arg}={ast.unparse(kw.value)} to {fn}"))
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
            targets = [node.target]
        else:
            targets = []
        for target in targets:
            for sub in ast.walk(target):
                if isinstance(sub, ast.Attribute) and sub.attr.lower().startswith("chi_"):
                    found.append((node.lineno, f"assigns {sub.attr}"))
    return found


def test_the_probe_modules_write_no_setting_and_hand_no_grid_to_training():
    """The probe checks measure with their own grids, and those grids reach the simulator only as a
    drive's frequency and amplitude. So the family's modules read no environment, assign no setting,
    build no configuration of their own, and hand the training generator nothing but the config's own
    band and drive."""
    root = Path(__file__).resolve().parents[1]
    for rel in ("core/diagnostics/probes.py", "core/diagnostics/probe_math.py", "core/tool/probes.py"):
        tree = ast.parse((root / rel).read_text(encoding="utf-8"))
        assert _env_reads_and_knob_writes(tree) == [], rel
        assert _grid_leaks(tree) == [], rel

    # the checker's teeth: each form it claims to catch, and the one call it allows
    teeth = ast.parse("cfg2 = dataclasses.replace(cfg, chi_f0=0.3)\n"
                      "cli.make_sim_config('NADROWSKI', labels, drift, bounds)\n"
                      "pipeline.gen_training_data(chi_f0=0.3, chi_freq_bounds=cfg.chi_freq_bounds)\n"
                      "cfg.chi_freq_bounds = (0.1, 0.5)\n"
                      "config.CHI_F0 += 1\n"
                      "pipeline.gen_training_data(chi_f0=cfg.chi_f0, chi_freq_bounds=cfg.chi_freq_bounds)\n")
    assert sorted({ln for ln, _ in _grid_leaks(teeth)}) == [1, 2, 3, 4, 5], _grid_leaks(teeth)
