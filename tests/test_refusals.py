"""The refusals module: one exception kind, the field registry and the rule functions (piece 3 of the
2026-09-10 hardening programme, design §3.1).

Torch-free by construction and pinned so: the window runs these rules on the GUI thread at the click
and the tool runs them before any stage import, so ``core.refusals`` must cost nothing to import.
The one test here that imports ``core.config`` (and with it torch) is the default pin, and it is the
only one: the registry's defaults are literal strings, because the module cannot read ``config.py``
without importing torch, so that test is what keeps the two from drifting.
"""
import ast
import math
import re
import subprocess
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from core.refusals import (FIELDS, Field, Refusal, describe, refuse, require_at_least, require_between,
                           require_choice, require_file, require_finite, require_given, require_positive)

REPO = Path(__file__).resolve().parents[1]

# The registry's initial key set, exactly (design §3.1). The front-end tables (Task 4) map every one
# of these, and a key raised anywhere under core/ must be here. A later task that adds a key adds it
# here too -- the set is closed on purpose, so a misspelt key cannot slip in beside its twin.
BASE_KEYS = (
    "t_obs", "num_runs", "run_size_cap", "n_directions", "hpd_level", "n_cal", "cal_n_scales",
    "num_posterior_samples", "n_samples", "num_iterations", "sweep_batch", "max_sets", "walk_step",
    "stability_units", "min_cluster_size", "min_samples", "hidden_features", "num_transforms",
    "learning_rate", "stop_after_epochs", "max_num_epochs", "fisher_m", "fisher_dz", "fisher_points",
    "checkpoint_every", "resume", "new_run", "accept_truncated", "accept_other_observation", "name",
    "bounds", "cell", "units", "recording_spont", "recording_forced", "recording_probe",
    "drive_amplitude", "drive_frequency", "drive_phase", "chi_f0_si", "chi_n_freqs", "chi_k_pad",
    "chi_max_cycles", "chi_f0", "chi_freq_bounds", "device", "model", "observation", "posterior", "prior",
)
TOOL_ONLY_KEYS = ("repeats", "n_points", "n_worst", "top_n", "m", "m_noise", "rel", "min_valid", "rows",
                  "n_sweep", "chi_k_fixed")


def _shape(exc, key):
    """The shape every rule refusal shares (design §3.1): it starts with the field's description as
    a sentence, ends with a period, carries the field key and the message on the exception, and ends
    with the default clause exactly when the field has a default. Returns the message for the
    caller's own pin on the words in between."""
    msg = str(exc)
    what = describe(key)
    assert exc.field == key, (exc.field, key)
    assert exc.message == msg
    assert msg.startswith(what[0].upper() + what[1:]), msg
    assert msg.endswith("."), msg
    default = FIELDS[key].default
    if default is None:
        assert "(default" not in msg, msg
    else:
        assert msg.endswith(f" (default {default})."), msg
    return msg


def test_a_refusal_is_a_value_error_carrying_its_message_and_a_keyword_only_field():
    """``Refusal`` is what both front ends route to the yellow "Check your inputs" box or the
    ``refused:`` line, so its two attributes are the whole interface: ``message`` (also ``str(exc)``,
    so an existing ``str(e.value)`` pin keeps reading it) and ``field``, None when no single control
    answers it. A ValueError, so the ``except ValueError`` callers that predate the conversion keep
    working. ``field`` is keyword-only: a positional second argument would read as ValueError's own
    args and the key would vanish into ``exc.args``."""
    text = "The HPD level must be between 0 and 1 (exclusive); got 1 (default 0.999)."
    e = Refusal(text, field="hpd_level")
    assert isinstance(e, ValueError)
    assert str(e) == e.message == text
    assert e.field == "hpd_level"
    assert Refusal("no single control answers this").field is None
    with pytest.raises(TypeError):
        Refusal("x", "hpd_level")
    with pytest.raises(Refusal, match="no single control"):
        raise Refusal("no single control answers this")


def test_the_registry_holds_exactly_the_initial_keys_with_neutral_descriptions():
    """Every key raised anywhere under core/ is here (Task 4's AST scan pins that side); this side
    pins the registry itself: the closed initial set, and that each entry is a frozen ``Field`` whose
    description is a lower-case noun phrase with no trailing period (it is spliced into "<What> must
    be ..." and "<What> is blank ...") that names no box, tab, flag or button (V3: neutral; the front
    ends add the control), and whose default is a string or None -- never a number, so a message
    shows the default as the operator would type it. ``describe`` is the public reader and refuses
    an unknown key with a KeyError: a message can only be built for a field a front end can map."""
    assert set(FIELDS) == set(BASE_KEYS) | set(TOOL_ONLY_KEYS)
    assert len(FIELDS) == len(BASE_KEYS) + len(TOOL_ONLY_KEYS) == 61, "a key is listed twice above"
    control_words = re.compile(r"\b(tab|box|flag|button|click|tick|dialog)\b")
    for key, f in FIELDS.items():
        assert isinstance(f, Field) and f.key == key, key
        assert f.what and f.what[0].islower() and not f.what.endswith("."), (key, f.what)
        assert not control_words.search(f.what), (key, f.what)
        assert f.default is None or (isinstance(f.default, str) and f.default), (key, f.default)
        assert describe(key) == f.what
    assert describe("t_obs") == "the observation length, in seconds"
    assert FIELDS["t_obs"].default == "none: it must be given"
    assert FIELDS["run_size_cap"].what == "the rows-per-batch cap (0 = automatic)"
    assert FIELDS["new_run"] == Field("new_run", "consent to start a new simulation cache", None)
    with pytest.raises(KeyError):
        describe("t_obs_seconds")
    with pytest.raises(FrozenInstanceError):
        FIELDS["t_obs"].default = "1"


def test_a_blank_is_refused_by_every_rule_with_the_default_in_the_sentence():
    """V2: a blank box is a refusal, never a zero or a default. ``None`` is the blank (what
    ``value_or_none()`` returns for an empty or half-typed box), every rule refuses it before it looks
    at anything else, and the sentence names the default so the operator knows what to type. A field
    with no default gets the short form. The two "0 = automatic" boxes accept a typed 0 only."""
    rules = [
        lambda: require_given("t_obs", None),
        lambda: require_positive("t_obs", None),
        lambda: require_finite("t_obs", None),
        lambda: require_at_least("t_obs", None, 1),
        lambda: require_between("t_obs", None, 0.0, 1.0),
        lambda: require_choice("t_obs", None, ("a", "b")),
    ]
    for rule in rules:
        with pytest.raises(Refusal) as e:
            rule()
        assert _shape(e.value, "t_obs") == (
            "The observation length, in seconds is blank (default none: it must be given).")
    with pytest.raises(Refusal) as e:
        require_at_least("num_runs", None, 1)
    assert _shape(e.value, "num_runs") == "The number of training batches is blank (default 5000)."
    with pytest.raises(Refusal) as e:
        require_finite("drive_phase", None)
    assert _shape(e.value, "drive_phase") == "The drive phase is blank."
    assert require_at_least("run_size_cap", 0, 0) == 0
    assert require_at_least("sweep_batch", 0, 0) == 0
    with pytest.raises(Refusal) as e:
        require_at_least("sweep_batch", None, 0)
    assert _shape(e.value, "sweep_batch") == "The candidates per sweep round (0 = automatic) is blank (default 0)."


def test_require_given_returns_what_it_was_given():
    """The weakest rule: present or blank. It coerces nothing and judges nothing else -- an empty
    name is GIVEN (the store's name rule is the store's), a policy string is given -- so a caller
    that only needs "the box was filled in" binds exactly what came in."""
    assert require_given("t_obs", 4.5) == 4.5
    assert require_given("name", "") == ""
    assert require_given("resume", "auto") == "auto"
    assert require_given("run_size_cap", 0) == 0


def test_require_positive_refuses_zero_negative_and_non_finite_and_returns_a_float():
    """t_obs, the two drives, the walk step, the learning rate, the Fisher step: "finite, > 0". A zero
    is the refusal the observation length needed most -- a blank read as 0 used to reach
    math.log(0.0) AFTER the simulation was spent -- and NaN and the infinities are refused by the same
    sentence, because ``nan > 0`` is False and ``inf`` is not a length. The value comes back as a
    float so the stage binds the coerced number, formatted with :g so 5000.0 reads as 5000."""
    assert require_positive("t_obs", 4.5) == 4.5
    assert isinstance(require_positive("t_obs", 3), float)
    for bad, shown in ((0, "0"), (-2.5, "-2.5"), (0.0, "0"), (float("nan"), "nan"), (float("inf"), "inf"),
                       (-math.inf, "-inf")):
        with pytest.raises(Refusal) as e:
            require_positive("t_obs", bad)
        assert _shape(e.value, "t_obs") == (
            f"The observation length, in seconds must be greater than 0; got {shown} "
            f"(default none: it must be given).")
    with pytest.raises(Refusal) as e:
        require_positive("learning_rate", 0)
    assert _shape(e.value, "learning_rate") == "The learning rate must be greater than 0; got 0 (default 0.001)."
    with pytest.raises(Refusal) as e:
        require_positive("drive_frequency", -1)
    assert _shape(e.value, "drive_frequency") == "The drive frequency must be greater than 0; got -1."


def test_require_at_least_refuses_below_the_minimum_and_returns_an_int():
    """Counts are "integer >= 1", the two automatic boxes and the checkpoint cadence "integer >= 0",
    the minimum cluster size "integer >= 2": one rule with the minimum as its argument, and the rule
    text is "at least <minimum>" so the existing "at least 1" pins in the store suite keep reading.
    Returns the int the stage binds (a "7" from a text source and a 3.0 both come back as ints)."""
    assert require_at_least("num_runs", 1, 1) == 1
    assert require_at_least("num_runs", "7", 1) == 7
    assert isinstance(require_at_least("num_runs", 3.0, 1), int)
    with pytest.raises(Refusal) as e:
        require_at_least("num_runs", 0, 1)
    assert _shape(e.value, "num_runs") == "The number of training batches must be at least 1; got 0 (default 5000)."
    with pytest.raises(Refusal) as e:
        require_at_least("min_cluster_size", 1, 2)
    assert _shape(e.value, "min_cluster_size") == "The minimum cluster size must be at least 2; got 1 (default 50)."
    with pytest.raises(Refusal) as e:
        require_at_least("run_size_cap", -5, 0)
    assert _shape(e.value, "run_size_cap") == (
        "The rows-per-batch cap (0 = automatic) must be at least 0; got -5 (default 0).")
    with pytest.raises(Refusal) as e:
        require_at_least("repeats", 0, 1)
    assert _shape(e.value, "repeats") == "The number of SBC repeats must be at least 1; got 0 (default 10)."


def test_require_between_honours_open_and_closed_ends_and_treats_nan_as_outside():
    """The HPD level is 0 < x < 1 (both ends open: 0 truncates everything, 1 nothing); the chi probe
    count and slots are closed intervals. " (exclusive)" appears when either end is open, and the
    bounds print with :g so 0.0 and 1.0 read as 0 and 1. NaN compares False against every bound and
    is refused as outside rather than slipping through as "not below and not above"."""
    assert require_between("hpd_level", 0.5, 0.0, 1.0, open_lo=True, open_hi=True) == 0.5
    assert require_between("hpd_level", 1.0, 0.0, 1.0) == 1.0
    assert require_between("hpd_level", 0, 0, 1) == 0.0
    assert isinstance(require_between("hpd_level", 1, 0, 1), float)
    for bad in (0.0, 1.0):
        with pytest.raises(Refusal) as e:
            require_between("hpd_level", bad, 0.0, 1.0, open_lo=True, open_hi=True)
        assert _shape(e.value, "hpd_level") == (
            f"The HPD level must be between 0 and 1 (exclusive); got {bad:g} (default 0.999).")
    with pytest.raises(Refusal) as e:
        require_between("hpd_level", 1.5, 0.0, 1.0)
    assert _shape(e.value, "hpd_level") == "The HPD level must be between 0 and 1; got 1.5 (default 0.999)."
    with pytest.raises(Refusal) as e:
        require_between("hpd_level", 0.0, 0.0, 1.0, open_lo=True)
    assert "(exclusive)" in _shape(e.value, "hpd_level")
    with pytest.raises(Refusal) as e:
        require_between("hpd_level", float("nan"), 0.0, 1.0)
    assert _shape(e.value, "hpd_level") == "The HPD level must be between 0 and 1; got nan (default 0.999)."
    with pytest.raises(Refusal) as e:
        require_between("chi_n_freqs", 25, 2, 24)
    assert _shape(e.value, "chi_n_freqs") == (
        "The number of chi probe frequencies must be between 2 and 24; got 25 (default 6).")


def test_require_finite_refuses_nan_and_inf_and_returns_a_float():
    """The drive phase is the one field that may be zero or negative but must be a number: the rule
    admits every finite value, sign included, and refuses NaN and both infinities with the given
    value in repr so "nan" and "inf" are visible words in the sentence."""
    assert require_finite("drive_phase", -3.25) == -3.25
    assert require_finite("drive_phase", 0) == 0.0
    assert isinstance(require_finite("drive_phase", 0), float)
    for bad in (float("nan"), float("inf"), -math.inf):
        with pytest.raises(Refusal) as e:
            require_finite("drive_phase", bad)
        assert _shape(e.value, "drive_phase") == f"The drive phase must be a finite number; got {bad!r}."
    with pytest.raises(Refusal) as e:
        require_finite("walk_step", math.nan)
    assert _shape(e.value, "walk_step") == "The random-walk step must be a finite number; got nan (default 0.01)."


def test_require_choice_refuses_an_unknown_option_and_lists_the_choices():
    """The resume policy is one of auto/require/never and the device one of auto/cpu/cuda; today the
    first is refused by keyword name inside the stage. The sentence lists the choices in the order
    the caller gives them and shows the given value in repr, so a stray space or case is visible."""
    assert require_choice("resume", "auto", ("auto", "require", "never")) == "auto"
    with pytest.raises(Refusal) as e:
        require_choice("resume", "sometimes", ("auto", "require", "never"))
    assert _shape(e.value, "resume") == (
        "The resume policy must be one of auto, require, never; got 'sometimes' (default auto).")
    with pytest.raises(Refusal) as e:
        require_choice("device", "gpu", ("auto", "cpu", "cuda"))
    assert _shape(e.value, "device") == "The compute device must be one of auto, cpu, cuda; got 'gpu' (default auto)."


def test_require_file_refuses_a_blank_and_a_missing_path_naming_the_input_kind(tmp_path):
    """The bounds, cell and units files and the three recordings: "given and the file exists",
    checked at the click and again at stage entry, so ``File not found: <path>`` inside the worker
    and ``the spont recording was not found: ''`` never happen. ``what`` names the input kind in the
    blank sentence ("no cell file was given"); without it the field's own description stands in, which
    reads awkwardly for a key whose description already says "file" -- callers pass ``what``. A
    directory is not the file. The path comes back as a str, so a Path caller binds the same thing a
    flag caller does."""
    f = tmp_path / "cell.txt"
    f.write_text("k 1\n", encoding="utf-8")
    assert require_file("cell", str(f), "cell") == str(f)
    assert require_file("cell", f, "cell") == str(f)
    for blank in (None, "", "   "):
        with pytest.raises(Refusal) as e:
            require_file("cell", blank, "cell")
        assert _shape(e.value, "cell") == "The cell file is blank: no cell file was given."
    with pytest.raises(Refusal) as e:
        require_file("recording_spont", "", "passive recording")
    assert _shape(e.value, "recording_spont") == (
        "The passive recording is blank: no passive recording file was given.")
    with pytest.raises(Refusal) as e:
        require_file("bounds", None)
    assert _shape(e.value, "bounds") == "The bounds file is blank: no the bounds file file was given."
    missing = str(tmp_path / "nowhere.txt")
    with pytest.raises(Refusal) as e:
        require_file("cell", missing, "cell")
    assert _shape(e.value, "cell") == f"The cell file was not found: {missing!r}."
    with pytest.raises(Refusal) as e:
        require_file("recording_forced", tmp_path, "driven recording")
    assert _shape(e.value, "recording_forced") == f"The driven recording was not found: {str(tmp_path)!r}."


def test_refuse_appends_the_default_clause_to_the_callers_sentence_and_binds_the_field():
    """For the refusals whose rule is not one of the require_* shapes -- a direction count above the
    latent width, a device that is not there -- the caller writes the sentence and ``refuse`` adds
    the field's default clause and key, so those messages carry the default the same way. The
    caller's sentence carries its own period and the clause follows it. A field with no default gets
    the sentence unchanged. It never returns, and an unregistered key is a KeyError, as everywhere."""
    with pytest.raises(Refusal) as e:
        refuse("device", "The compute device 'cuda' is not available on this machine (detected cpu).")
    assert str(e.value) == (
        "The compute device 'cuda' is not available on this machine (detected cpu). (default auto)")
    assert e.value.field == "device"
    with pytest.raises(Refusal) as e:
        refuse("cell", "The cell names a parameter the bounds file does not.")
    assert str(e.value) == "The cell names a parameter the bounds file does not."
    assert e.value.field == "cell"
    with pytest.raises(KeyError):
        refuse("not_a_field", "x")


def test_the_module_is_torch_free_and_imports_only_the_standard_library():
    """The window runs the rules on the GUI thread at the click and the tool before any stage import,
    so ``core.refusals`` must cost nothing: no torch, no ``core.config`` (which imports torch). Two
    pins: the module's import statements name only the four standard-library modules the design
    allows, and a fresh interpreter that imports it has neither torch nor core.config loaded. The
    subprocess is the real pin -- in this process torch is long since imported by the session
    fixtures, so a sys.modules check here would pass vacuously."""
    src = (REPO / "core" / "refusals.py").read_text(encoding="utf-8")
    imported = set()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Import):
            imported |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module)
    assert imported <= {"dataclasses", "math", "os", "typing"}, imported
    probe = ("import sys; import core.refusals; "
             "bad = sorted(m for m in sys.modules if m == 'torch' or m.startswith('torch.') or m == 'core.config'); "
             "sys.exit(repr(bad) if bad else 0)")
    r = subprocess.run([sys.executable, "-c", probe], cwd=str(REPO), capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr


def test_every_registry_default_is_the_trees_own_default():
    """THE ONE TEST HERE THAT IMPORTS core.config (and with it torch). The registry cannot read
    config.py without pulling torch into the click path, so its defaults are literal strings; this is
    what keeps them honest. Every default that config.py owns is pinned as ``str(constant)`` -- the
    same spelling the message shows -- and every other default against the object whose signature
    owns it: the truncation defaults, the stages' keyword defaults, the diagnostics' signatures and
    the tool's ``--device`` default. A constant retuned in config.py without this registry following
    it fails here, not in a message that names a default nobody set. The last assertion closes the
    set: no default exists that this test did not look at."""
    import argparse
    import inspect

    from core import config, orchestrator
    from core.diagnostics import ablation, identifiability, sbc
    from core.SBI import truncate
    from core.tool import config_args

    owned_by_config = {
        "num_runs": "TRAINING_NUM_RUNS", "run_size_cap": "TRAINING_RUN_SIZE", "n_cal": "SBC_N_CAL",
        "cal_n_scales": "CAL_N_SCALES", "num_iterations": "PRIOR_SWEEP_ITERATIONS",
        "sweep_batch": "PRIOR_SWEEP_BATCH", "max_sets": "PRIOR_SWEEP_MAX_SETS", "walk_step": "PRIOR_SWEEP_STEP",
        "stability_units": "STABILITY_SWEEP_ND_UNITS", "min_cluster_size": "PRIOR_CLUSTER_MIN_SIZE",
        "min_samples": "PRIOR_CLUSTER_MIN_SAMPLES", "hidden_features": "NSF_HIDDEN_FEATURES",
        "num_transforms": "NSF_NUM_TRANSFORMS", "learning_rate": "TRAINING_LEARNING_RATE",
        "stop_after_epochs": "TRAINING_STOP_AFTER_EPOCHS", "max_num_epochs": "TRAINING_MAX_NUM_EPOCHS",
        "fisher_m": "REPARAM_FISHER_M", "fisher_dz": "REPARAM_FISHER_DZ", "fisher_points": "REPARAM_FISHER_POINTS",
        "checkpoint_every": "TRAINING_CHECKPOINT_EVERY", "chi_n_freqs": "CHI_N_FREQS", "chi_k_pad": "CHI_K_PAD",
        "chi_max_cycles": "CHI_MAX_CYCLES",
    }
    for key, const in owned_by_config.items():
        assert FIELDS[key].default == str(getattr(config, const)), (key, const, FIELDS[key].default)

    def _default(fn, name):
        return inspect.signature(fn).parameters[name].default

    owned_by_a_signature = {
        "n_directions": str(truncate.DEFAULT_N_DIRECTIONS), "hpd_level": str(truncate.DEFAULT_HPD),
        "resume": str(_default(orchestrator.build_posterior, "resume")),
        "num_posterior_samples": str(_default(orchestrator.validate_calibration, "num_posterior_samples")),
        "n_samples": str(_default(orchestrator.infer_and_visualize, "n_samples")),
        "repeats": str(_default(sbc.sbc_repeats, "repeats")),
        "n_worst": str(_default(identifiability.identifiability_rotation, "n_worst")),
        "top_n": str(_default(identifiability.identifiability_rotation, "top_n")),
        "n_points": str(_default(identifiability.identifiability_laplace, "n_points")),
        "m": str(_default(identifiability.identifiability_laplace, "m")),
        "m_noise": str(_default(identifiability.identifiability_laplace, "m_noise")),
        "rel": str(_default(identifiability.identifiability_laplace, "rel")),
        "rows": str(_default(ablation.channel_ablation, "rows")),
        "n_sweep": str(_default(ablation.channel_ablation, "n_sweep")),
    }
    for key, expected in owned_by_a_signature.items():
        assert FIELDS[key].default == expected, (key, expected, FIELDS[key].default)
    p = argparse.ArgumentParser()
    config_args.add_config_flags(p)
    assert FIELDS["device"].default == p.get_default("device") == "auto"

    looked_at = set(owned_by_config) | set(owned_by_a_signature) | {"device"}
    rest = {k: f.default for k, f in FIELDS.items() if k not in looked_at}
    assert rest == {k: ("none: it must be given" if k == "t_obs" else None) for k in rest}, rest


def test_core_runs_imports_without_torch():
    """core/runs.py is imported by every public entry point and by BOTH front ends, and the tool's
    entry sets KMP_DUPLICATE_LIB_OK and the Agg backend BEFORE any torch import (CLAUDE.md). So the
    module must cost the standard library only -- which is also what lets public_entry duck-type
    the config instead of isinstance-checking SimConfig. core/__init__.py is empty, so a fresh
    interpreter tells the truth about what `import core.runs` pulls in."""
    import subprocess
    import sys
    from pathlib import Path

    repo = Path(__file__).resolve().parents[1]
    proc = subprocess.run(
        [sys.executable, "-c",
         "import sys, core.runs; assert 'torch' not in sys.modules, 'core.runs imports torch'"],
        cwd=str(repo), capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr


def test_the_core_logger_is_at_info_by_import():
    """V4 (spec §1.2, "V4 and the logger level"). Python's root logger sits at WARNING, so a `core`
    logger left at NOTSET inherits it and drops every information record before any handler sees
    it: the window's pane and the artifact's log.txt would carry warnings only, while
    `caplog.set_level` in a suite hid the loss. core/runs.py sets the level ONCE at import; the
    window's handler and the tool's `main` install and remove handlers and never touch it. Task 16
    extends this pin across `main` and a stream redirect."""
    import logging

    from core import runs

    assert runs.LOGGER is logging.getLogger("core")
    assert runs.LOGGER.level == logging.INFO
    assert logging.getLogger("core.orchestrator").getEffectiveLevel() == logging.INFO, \
        "a child inherits INFO: every module's `logging.getLogger(__name__)` is covered"
    assert runs.LOGGER.propagate is True, "caplog reads the records off the root logger"
    assert not any(isinstance(h, runs._RunLogHandler) for h in runs.LOGGER.handlers), \
        "no buffer is attached outside a run"


def test_capture_run_pushes_one_buffer_for_nested_calls():
    """V4 (spec §4.4). A composition writes two artifacts: the observation's log.txt holds the
    records up to its commit, the inference's holds the WHOLE composition's. So the public stages a
    composition calls must join the composition's buffer, not open their own: capture_run pushes
    only when nothing is active on this thread, and a nested exit pops nothing. The buffer is
    popped on EVERY outer exit, an exception's included, so a refused or crashed run leaves no
    handler on the logger and no tee on warnings.showwarning."""
    import logging
    import re
    import warnings

    import pytest

    from core import runs

    child = logging.getLogger("core.some_stage")
    assert runs.current_run_log() is None
    with runs.capture_run() as outer:
        assert runs.current_run_log() is outer
        child.info("outer says hello")
        with runs.capture_run() as inner:
            assert inner is outer, "a nested public entry joins the active buffer"
            child.warning("inner warns")
            child.debug("never recorded")             # below INFO: dropped at the logger
        assert runs.current_run_log() is outer, "the inner exit pops nothing"
        child.error("outer fails")
    assert runs.current_run_log() is None
    stamp = r"\d{2}:\d{2}:\d{2}"
    assert [re.sub(stamp, "T", ln) for ln in outer.lines] == [
        "T info outer says hello", "T warning inner warns", "T error outer fails"]
    assert outer.text() == "\n".join(outer.lines) + "\n"
    assert runs.RunLog().text() == "", "an empty buffer renders as an empty file"

    # popped on the way out of an exception, with the handler and the warnings hook gone
    prev_hook = warnings.showwarning
    with pytest.raises(RuntimeError, match="mid-run"):
        with runs.capture_run():
            assert warnings.showwarning is not prev_hook, "the tee is installed while active"
            raise RuntimeError("mid-run")
    assert runs.current_run_log() is None
    assert warnings.showwarning is prev_hook
    assert not any(isinstance(h, runs._RunLogHandler) for h in runs.LOGGER.handlers)


def test_the_run_buffer_tees_python_warnings_and_calls_the_previous_hook():
    """V4 (spec §1.2, "V4 and Python warnings"). The judgement channel -- PreflightWarning -- is
    what a reviewer wants in an artifact's log.txt, so the buffer tees warnings.showwarning while
    it is active. It TEES: the hook that was there before (the window's pane hook under a run,
    pytest's recorder under pytest.warns, Python's stderr printer otherwise) is still called with
    the same six arguments, and is restored on detach, so the tee nests cleanly inside either.
    The PreflightWarning import is local: the module under test stays torch-free, and the test
    process already holds core.orchestrator through the session fixtures."""
    import re
    import warnings

    import pytest

    from core import runs
    from core.orchestrator import PreflightWarning

    seen = []
    with warnings.catch_warnings():                   # restores showwarning and the filters on exit
        warnings.simplefilter("always")
        warnings.showwarning = lambda *a, **k: seen.append(a)
        mine = warnings.showwarning
        with runs.capture_run() as log:
            warnings.warn("T_obs is outside the training range", PreflightWarning)
        assert warnings.showwarning is mine, "detach restores the hook it found"
    assert len(seen) == 1 and str(seen[0][0]) == "T_obs is outside the training range" \
        and seen[0][1] is PreflightWarning, "the previous hook still receives the warning"
    assert re.fullmatch(r"\d{2}:\d{2}:\d{2} warning PreflightWarning: T_obs is outside the training range",
                        log.lines[0]), log.lines

    # under pytest.warns the previous hook is pytest's recorder, and it still records
    with pytest.warns(PreflightWarning, match="training range"):
        with runs.capture_run() as log2:
            warnings.warn("T_obs is outside the training range", PreflightWarning)
    assert len(log2.lines) == 1 and "PreflightWarning" in log2.lines[0]


def test_the_public_entry_decorator_passes_a_sentinel_through():
    """V1 (spec §2.2, §2.4). The decorator replaces the config argument -- the first positional, or
    the `cfg` keyword -- with its copy_for_run() and runs the call inside capture_run(). It is
    duck-typed, so core/runs.py never imports torch, and a stub or an object() sentinel (the gate
    tests put one on session.cfg) passes through untouched. functools.wraps keeps the name, the
    docstring, the signature and the source, which the AST pins on the stages read."""
    import inspect
    import logging

    import pytest

    from core import runs

    class _Duck:
        """A config stand-in whose copy_for_run returns a NEW object that remembers its origin."""
        def __init__(self, origin=None):
            self.origin = origin

        def copy_for_run(self):
            return _Duck(origin=self)

    seen = {}

    @runs.public_entry
    def stage(cfg, prior, *, num_runs=None, boom=False):
        """the stage's own doc"""
        seen["cfg"], seen["log"] = cfg, runs.current_run_log()
        if boom:
            raise RuntimeError("inside the stage")
        return cfg

    d = _Duck()
    out = stage(d, "prior", num_runs=3)
    assert out is not d and out.origin is d, "the stage received a copy of the caller's config"
    assert isinstance(seen["log"], runs.RunLog), "the call ran inside a run buffer"
    assert runs.current_run_log() is None, "and the buffer was popped on return"

    out = stage(cfg=d, prior="prior")
    assert out is not d and out.origin is d, "the keyword form is copied too"

    s = object()
    assert stage(s, "prior") is s, "a sentinel with no copy_for_run passes through untouched"
    assert stage(prior="prior", cfg=s) is s

    with pytest.raises(RuntimeError, match="inside the stage"):
        stage(d, "prior", boom=True)
    assert runs.current_run_log() is None, "popped on an exception too"

    assert stage.__wrapped__.__name__ == "stage" and stage.__doc__ == "the stage's own doc"
    assert list(inspect.signature(stage).parameters) == ["cfg", "prior", "num_runs", "boom"]
    assert inspect.getsource(stage).lstrip().startswith("@runs.public_entry"), \
        "getsource follows __wrapped__ and returns the decorated original"

    # inside an outer capture -- a composition's -- the stage's records land in the OUTER buffer
    with runs.capture_run() as outer:
        @runs.public_entry
        def logging_stage(cfg):
            logging.getLogger("core.logging_stage").info("stage record")

        logging_stage(_Duck())
    assert any(ln.endswith("info stage record") for ln in outer.lines), outer.lines


# ── The tool's table, and the closure of the registry over the tree (piece 3, Task 4) ────────────
# The rule functions take the key POSITIONALLY (require_positive("t_obs", v)); a refusal raised directly
# carries it as field="…". Both shapes are scanned. `describe` is in the list although
# core/tool/config_args.py has a describe(cfg, ...) of its own: that one's first argument is a config,
# never a string literal, so a name match cannot mistake it for the registry's describe(key).
_RULE_CALLS = ("describe", "refuse", "require_given", "require_finite", "require_positive",
               "require_at_least", "require_between", "require_choice", "require_file")


def _field_key_literals(tree) -> list:
    """Every string literal used as a refusal field key in a parsed module: the ``field="…"`` keyword
    of ANY call, and the first positional argument of a call to one of ``_RULE_CALLS`` (by the
    callee's bare name, whether ``require_file(...)`` or ``refusals.require_file(...)``).
    ``(lineno, key)`` pairs. A ``field=None`` or a key passed as a variable is not a literal and is
    not returned."""
    import ast
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        for kw in node.keywords:
            if kw.arg == "field" and isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                out.append((node.lineno, kw.value.value))
        fn = node.func
        name = fn.id if isinstance(fn, ast.Name) else fn.attr if isinstance(fn, ast.Attribute) else ""
        if (name in _RULE_CALLS and node.args and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)):
            out.append((node.lineno, node.args[0].value))
    return out


def test_every_field_key_has_a_flag_and_every_key_literal_under_core_is_registered():
    """V3's tool half, and the closure of the registry. A Refusal names its field by a key and says
    nothing about flags; the tool turns the key into a flag with ONE table, so:

    (a) the table knows every registry key and no other -- a key that reached the ladder unmapped
        would print a refusal with no way out named, and an entry for a key nobody raises is a flag
        that can go stale unseen;
    (b) every flag it names is one build_parser defines, read off the REAL parsers (option strings,
        modes included) rather than a copy of the help text, because the flag is what the operator
        types next. The one key whose flag is not its own name spelled with dashes is pinned by name;
    (c) fix_sentence owns the parentheses and never raises: it runs while a refusal is being printed;
    (d) the module imports in a bare interpreter with neither torch nor Qt -- the ladder prints
        refusals, and one of them can be that the card was not there;
    (e)(f) every key LITERAL under core/ is registered. The rule functions look a key up at call
        time, so a typo would surface as a KeyError from inside a refusal, on the one path no test
        walked. Found by the same AST walk the literal-path scan uses (test_artifact_store.py), over
        the same CODE_ROOTS + CODE_FILES. The scanner is checked on a snippet FIRST: with no field=
        literal in the tree yet (Task 5 adds the first), the tree walk alone would pass vacuously.
    """
    import argparse
    import ast
    import subprocess
    import sys
    from pathlib import Path
    from core.refusals import FIELDS
    from core.tool import build_parser
    import core.tool.fields as tool_fields

    # (a) the same key set, in both directions
    assert set(tool_fields.FLAG) == set(FIELDS), (
        f"only in FLAG: {sorted(set(tool_fields.FLAG) - set(FIELDS))}; "
        f"only in FIELDS: {sorted(set(FIELDS) - set(tool_fields.FLAG))}")

    # (b) every flag is a real option string on some subcommand, or on a mode of one
    def _walk(parser):
        yield parser
        for a in parser._actions:
            if isinstance(a, argparse._SubParsersAction):
                for sub in a.choices.values():
                    yield from _walk(sub)

    real = {opt for top in build_parser().subcommands.values()
            for p in _walk(top) for opt in p._option_string_actions}
    unreal = {k: f for k, f in tool_fields.FLAG.items() if f is not None and f not in real}
    assert not unreal, f"FLAG names an option no subcommand defines: {unreal}"
    assert all(f is None or f.startswith("--") for f in tool_fields.FLAG.values())
    assert {k for k, f in tool_fields.FLAG.items() if f is None} == {
        "units", "chi_k_pad", "chi_max_cycles", "chi_f0", "chi_freq_bounds"}
    assert tool_fields.FLAG["num_posterior_samples"] == "--posterior-samples"    # the tree's spelling
    assert tool_fields.FLAG["hpd_level"] == "--level" and tool_fields.FLAG["n_directions"] == "--directions"
    assert tool_fields.FLAG["recording_probe"] == tool_fields.FLAG["recording_forced"] == "--forced"
    assert (tool_fields.FLAG["drive_amplitude"] == tool_fields.FLAG["drive_frequency"]
            == tool_fields.FLAG["drive_phase"] == "--drive")
    assert tool_fields.FLAG["max_num_epochs"] == "--max-epochs" and tool_fields.FLAG["run_size_cap"] == "--run-size"

    # (c) fix_sentence owns the parentheses and never raises
    assert tool_fields.fix_sentence("t_obs") == "(--t-obs)"
    assert tool_fields.fix_sentence("new_run") == "(--new-run)"
    assert tool_fields.fix_sentence("repeats") == "(--repeats)"
    assert tool_fields.fix_sentence("units") == ""
    assert tool_fields.fix_sentence(None) == ""
    assert tool_fields.fix_sentence("no_such_key") == ""

    # (d) torch-free and Qt-free in a fresh interpreter
    root = Path(__file__).resolve().parents[1]
    probe = ("import sys; import core.tool.fields; "
             "print('torch' in sys.modules, any(m.startswith('PySide6') for m in sys.modules))")
    r = subprocess.run([sys.executable, "-c", probe], cwd=root, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0 and r.stdout.split() == ["False", "False"], r.stdout + r.stderr

    # (e) the scanner, on a snippet: one key of each shape it must see, and the shapes it must not
    snippet = ast.parse(
        'raise Refusal("x", field="t_obs")\n'                      # 1: field= on a raise
        'store.StoreError("y", field="no_such_box")\n'             # 2: field= on an attribute call
        'require_positive("walk_step", v)\n'                       # 3: a rule, bare name
        'refusals.require_at_least("nor_this", n, 1)\n'            # 4: a rule, through the module
        'describe("hpd_level")\n'                                  # 5: describe(key)
        'Refusal("z", field=None)\n'                                # 6: None is not a literal key
        'refuse(key, "w")\n'                                       # 7: a variable is not a literal
        'describe(cfg, cell="c")\n'                                # 8: config_args.describe's shape
        'other(field="not_a_refusal_kw")\n')                       # 9: field= on ANY call counts
    assert [k for _, k in sorted(_field_key_literals(snippet))] == [
        "t_obs", "no_such_box", "walk_step", "nor_this", "hpd_level", "not_a_refusal_kw"]

    # (f) the tree: every key literal under CODE_ROOTS and CODE_FILES is a registered key
    from tests._fixtures import CODE_FILES, CODE_ROOTS
    files = [py for sub in CODE_ROOTS for py in sorted((root / sub).rglob("*.py"))]
    files += [root / name for name in CODE_FILES]
    assert len(files) >= 20, f"the scan walked only {len(files)} files -- CODE_ROOTS is {CODE_ROOTS}"
    unregistered = []
    for py in files:
        for ln, key in _field_key_literals(ast.parse(py.read_text(encoding="utf-8"))):
            if key not in FIELDS:
                unregistered.append(f"{py.relative_to(root)}:{ln}: {key!r}")
    assert not unregistered, ("refusal keys used under core/ but absent from core.refusals.FIELDS:\n"
                              + "\n".join(unregistered))
