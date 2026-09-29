"""The refusals module: one exception kind, the field registry and the rule functions.

The MODULE is torch-free by construction and pinned so in a fresh interpreter
(test_the_module_is_torch_free_and_imports_only_the_standard_library): the window runs these rules
on the GUI thread at the click and the tool runs them before any stage import, so ``core.refusals``
must cost nothing to import. This FILE is not torch-free, and no longer claims to be. The default pin
imports ``core.config`` (and with it torch) because the registry's defaults are literal strings --
the module cannot read ``config.py`` without importing torch -- so that test is what keeps the two
from drifting. It is not the only one: the domain-error scan, the run buffer's warning test and the
cell-wording tests (which call ``core.cli`` and ``core.sim_config``) reach torch-importing modules
too. Each imports them inside the test, so the file's own top-level imports stay the module under
test and the standard library.
"""
import ast
import math
import re
import subprocess
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from core.refusals import (FIELDS, NOTE_MAX_CHARS, Field, Refusal, describe, refuse, require_at_least,
                           require_below, require_between, require_choice, require_file, require_finite,
                           require_given, require_note, require_positive)

REPO = Path(__file__).resolve().parents[1]

# The registry's key set, exactly. The front-end tables map every one of these, and a key raised
# anywhere under core/ must be here. A change that adds a key adds it here too -- the set is closed
# on purpose, so a misspelt key cannot slip in beside its twin.
BASE_KEYS = (
    "t_obs", "num_runs", "run_size_cap", "n_directions", "hpd_level", "n_cal", "cal_n_scales",
    "num_posterior_samples", "n_samples", "num_iterations", "sweep_batch", "max_sets", "walk_step",
    "stability_units", "min_cluster_size", "min_samples", "hidden_features", "num_transforms",
    "learning_rate", "stop_after_epochs", "max_num_epochs", "fisher_m", "fisher_dz", "fisher_points",
    "checkpoint_every", "resume", "new_run", "accept_truncated", "accept_other_observation", "name",
    "bounds", "cell", "units", "recording_spont", "recording_forced", "recording_probe",
    "drive_amplitude", "drive_frequency", "drive_phase", "chi_f0_si", "chi_n_freqs", "chi_k_pad",
    "chi_max_cycles", "chi_f0", "chi_freq_bounds", "device", "model", "observation", "posterior", "prior",
    # the secondary analyses: the eight knobs the FDT and sweep screens and their two subcommands
    # expose, then the five FDT settings neither front end exposes -- registered all the same.
    "n_freqs", "ensemble_m", "freqs_per_batch", "f0", "preset", "s_grid", "t_grid", "seed",
    "freq_bounds", "burn_in_nd", "t_obs_periods", "dt_nd", "psd_t_obs_nd",
    "artifact", "note",
    # the model builder
    "param_value", "param_min", "param_max", "init", "x_scale", "t_scale", "forcing_value",
    # the live simulation
    "frame_steps", "fps",
    # comparing saved FDT records -- which records, and the two mode settings the comparison checks
    # before its record opens
    "compare_records", "prefactor", "slice_at",
)
TOOL_ONLY_KEYS = ("repeats", "n_points", "n_worst", "top_n", "m", "m_noise", "rel", "min_valid", "rows",
                  "n_sweep", "chi_k_fixed",
                  # the probe checks: the seed, the band check's four grids, its repeat count and its
                  # five pass thresholds, and the mask audit's batch count and batch size
                  "probe_seed", "probe_lengths", "probe_multipliers", "probe_drives", "band_repeats",
                  "probe_cycle_caps", "cv_max", "phase_max", "snr_min", "sup_min", "band_peak_window",
                  "mask_num_runs", "mask_run_size")


def _shape(exc, key):
    """The shape every rule refusal shares: it starts with the field's description as a sentence,
    ends with a period, carries the field key and the message on the exception, and ends with the
    default clause exactly when the field has a default. Returns the message for the caller's own pin
    on the words in between."""
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
    """Every key raised anywhere under core/ is here (the key-literal scan further down pins that
    side); this side pins the registry itself: the closed initial set, and that each entry is a frozen
    ``Field`` whose description is a lower-case noun phrase with no trailing period (it is spliced
    into "<What> must be ..." and "<What> is blank ...") that names no box, tab, flag or button
    (neutral: the front ends add the control), and whose default is a string or None -- never a
    number, so a message shows the default as the operator would type it. ``describe`` is the public
    reader and refuses an unknown key with a KeyError: a message can only be built for a field a
    front end can map."""
    assert set(FIELDS) == set(BASE_KEYS) | set(TOOL_ONLY_KEYS)
    assert len(FIELDS) == len(BASE_KEYS) + len(TOOL_ONLY_KEYS) == 101, "a key is listed twice above"
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
    # the artifact browser's two: the artifact it acts on, and its note. Both descriptions obey the
    # wording ban above (the regex on f.what), and neither has a default -- a note has no default
    # text and an artifact is chosen, not defaulted.
    assert FIELDS["artifact"] == Field("artifact", "the artifact", None)
    assert FIELDS["note"] == Field("note", "the note", None)
    with pytest.raises(KeyError):
        describe("t_obs_seconds")
    with pytest.raises(FrozenInstanceError):
        FIELDS["t_obs"].default = "1"


def test_one_phrase_says_the_cell_is_missing_what_the_bounds_file_declares():
    """The cell not supplying something the bounds file declares is ONE rule, and the dry run
    (cli.validate_gt_file) and the injection (SimConfig._fill_checked) worded it two ways. One
    builder now produces the phrase. It is a FRAGMENT -- lower case, no trailing period -- because
    the dry run's problems are joined with "; " (infer_tab._on_cell_changed) while the refusal puts
    "Cell file is " in front and a period after. The label is used as given, never pluralised, and a
    list is plain comma-separated names, never a repr'd Python list."""
    from core.refusals import missing_values_phrase
    assert missing_values_phrase("ND parameters", ["k_gs", "gamma"]) == \
        "missing ND parameters the bounds file requires: k_gs, gamma"
    assert missing_values_phrase("rescale parameters", ("x_scale",)) == \
        "missing rescale parameters the bounds file requires: x_scale"
    assert "[" not in missing_values_phrase("forcing parameters", ["amp"])
    phrase = missing_values_phrase("ND parameters", ["k_gs"])
    assert phrase[0].islower() and not phrase.endswith(".")


def test_the_two_cell_sites_both_splice_the_one_phrase():
    """The other half of that rule: each of the two sites builds its wording through the phrase and
    does not re-type the sentence beside the call. Asserted on executable source, because what would
    regress is somebody re-typing the sentence. cli._merge_vals_bounds is a known third wording,
    left alone on purpose (it carries the cell path), so it is not in this list."""
    from core import cli, sim_config
    from tests._fixtures import code_only
    for obj in (cli.validate_gt_file, sim_config.SimConfig._fill_checked):
        src = code_only(obj)
        assert "missing_values_phrase(" in src, obj
        assert "the bounds file requires" not in src, obj


def test_the_dry_run_and_the_injection_word_a_missing_value_identically(monkeypatch):
    """The cell refusal's wording is identical from load_and_validate_gt and from the dry run: the
    refusal is exactly "Cell file is " + the dry run's problem + ".". Both are reached without a
    bounds file: the dry run through a stand-in config carrying the three dicts it reads and a
    stubbed values parser; the injection through the static SimConfig._fill_checked, called with the
    label inject_ground_truth passes."""
    import types
    from collections import OrderedDict
    from core import cli
    from core.sim_config import SimConfig

    monkeypatch.setattr(cli.file_manager, "parse_values_file",
                        lambda path: ({"x": 0.0}, {}, {}, {}))
    declared = OrderedDict(k_gs=(None, (0.0, 1.0)))
    cfg = types.SimpleNamespace(params_dict=OrderedDict(declared), rescale_params=OrderedDict(),
                                force_params_dict=OrderedDict())
    problems = cli.validate_gt_file(cfg, "unused_cell.txt")
    assert problems == ["missing ND parameters the bounds file requires: k_gs"], problems
    with pytest.raises(Refusal) as exc:
        SimConfig._fill_checked("ND parameters", {}, OrderedDict(declared), check_bounds=True)
    assert exc.value.field == "cell"
    assert exc.value.message == f"Cell file is {problems[0]}."


def test_a_blank_is_refused_by_every_rule_with_the_default_in_the_sentence():
    """A blank box is a refusal, never a zero or a default. ``None`` is the blank (what
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


def test_require_below_refuses_an_inverted_or_blank_pair_and_returns_two_floats():
    """The sweep grids and the FDT frequency band are ORDERED PAIRS, and none of the existing eight
    rules covers that shape: require_between judges one value against fixed bounds, and a pair whose
    own two ends are the thing being judged has no bound to compare to. An inverted or degenerate
    pair is not hypothetical: a typed (0.0, 0.0) would make np.linspace produce a sweep of one
    repeated value rather than refuse. A BLANK end is another matter -- value() would read it as a
    real 0, which passes here whenever the other end is above it -- so every caller reads its ends
    through value_or_none and a blank arrives as None.

    Both ends are refused blank and non-finite first, so the message never reads "0 and nan"; the
    pair comes back as floats, which is what the caller binds."""
    assert require_below("s_grid", 0.0, 1.0) == (0.0, 1.0)
    assert all(isinstance(v, float) for v in require_below("s_grid", 0, 1))
    for lo, hi, shown in ((1.0, 0.0, "1 and 0"), (0.0, 0.0, "0 and 0"), (2.5, 2.5, "2.5 and 2.5")):
        with pytest.raises(Refusal) as e:
            require_below("s_grid", lo, hi)
        assert _shape(e.value, "s_grid") == (
            f"The activity sweep grid must have its lower bound below its upper bound; got {shown}.")
    with pytest.raises(Refusal) as e:
        require_below("t_grid", None, 2.0)
    assert _shape(e.value, "t_grid") == "The temperature sweep grid is blank."
    with pytest.raises(Refusal) as e:
        require_below("t_grid", 1.0, float("nan"))
    assert _shape(e.value, "t_grid") == "The temperature sweep grid must be a finite number; got nan."


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
    """The bounds and cell files and the three recordings: "given and the file exists". The bounds
    file is checked where the config is built (cli.make_sim_config: the Prior tab's click and every
    subcommand's first act), the cell where its truth is read (cli.load_and_validate_gt), at the Infer
    tab's click and at simulated_inference's entry, and the recordings at the click and again at
    stage entry, so ``File not found: <path>`` and ``the spont recording was not found: ''`` are
    reached only by a race. (The units file has its own refusal, cli.resolve_units_file.) ``what``
    names the input kind in the blank sentence ("no cell file was given"); without it the field's own
    description stands in, which reads awkwardly for a key whose description already says "file" --
    callers pass ``what``. A directory is not the file. The path comes back as a str, so a Path caller
    binds the same thing a flag caller does."""
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


def test_require_note_trims_one_line_and_refuses_a_break_or_the_limit():
    """A note is ONE line, at most NOTE_MAX_CHARS characters, and blank clears it.

    Surrounding whitespace is TRIMMED -- the one transformation this module allows, because it changes
    no meaning -- and everything else is refused rather than fixed: a newline, a carriage return
    or a tab inside the text refuses, and so does a note over the limit, whose sentence gives BOTH the
    limit and the length given so the operator knows how much to cut. An all-whitespace note is not a
    refusal: it is how a note is CLEARED, and it comes back as "". The limit lives here and not in
    config.py: this module is torch-free on purpose and a note limit is not a science constant."""
    assert NOTE_MAX_CHARS == 200
    assert require_note("note", "  kept for the paper  ") == "kept for the paper"
    assert require_note("note", "kept") == "kept"
    assert require_note("note", "") == ""
    assert require_note("note", "   ") == "", "an all-whitespace note clears it"
    assert require_note("note", " \t \n ") == "", "so does one that is only a break"
    assert require_note("note", None) == "", "a blank box clears it too"
    assert require_note("note", "x" * NOTE_MAX_CHARS) == "x" * NOTE_MAX_CHARS
    assert require_note("note", "  " + "x" * NOTE_MAX_CHARS + "  ") == "x" * NOTE_MAX_CHARS, \
        "the length is measured AFTER the trim"
    with pytest.raises(Refusal) as e:
        require_note("note", "x" * (NOTE_MAX_CHARS + 1))
    assert _shape(e.value, "note") == "The note must be at most 200 characters; got 201."
    for bad in ("a\nb", "a\rb", "a\tb"):
        with pytest.raises(Refusal) as e:
            require_note("note", bad)
        assert _shape(e.value, "note") == f"The note must be one line; got {bad!r}."
    assert e.value.field == "note", "the key the caller passed travels on the refusal"


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
    from core.refusals import default_clause, default_text
    assert default_clause("num_runs") == default_text("5000") == " (default 5000)"
    assert default_clause("max_num_epochs") == " (default no ceiling)"
    assert default_clause("min_valid") == " (default 0.5)"
    assert default_clause("new_run") == ""


def test_the_seed_rule_carries_the_field_key_it_is_given():
    """One seed rule for every seeded run, keyed on the field the caller names: the probe checks'
    seed has its own key, whose default is 0, while every other caller keeps ``seed``. Both ends of the
    range refuse under the key given, with that key's default in the sentence."""
    from core.rng import SEED_MAX, require_seed
    for bad in (-1, SEED_MAX + 1):
        with pytest.raises(Refusal) as e:
            require_seed(bad, key="probe_seed")
        assert e.value.field == "probe_seed", (bad, e.value.field)
        assert str(e.value).rstrip(".").endswith("(default 0)"), str(e.value)
    with pytest.raises(Refusal) as e:
        require_seed(-1)
    assert e.value.field == "seed"
    assert require_seed(7, key="probe_seed") == 7


def test_the_module_is_torch_free_and_imports_only_the_standard_library():
    """The window runs the rules on the GUI thread at the click and the tool before any stage import,
    so ``core.refusals`` must cost nothing: no torch, no ``core.config`` (which imports torch). Two
    pins: the module's import statements name only the five standard-library modules it may use
    (``warnings`` because PreflightWarning and its "always" filter live here, so the FDT path can
    raise one without the SBI stack), and a fresh interpreter that imports it has neither torch nor
    core.config loaded. The subprocess is the real pin -- in this process torch is long since
    imported by the session fixtures, so a sys.modules check here would pass vacuously."""
    src = (REPO / "core" / "refusals.py").read_text(encoding="utf-8")
    imported = set()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Import):
            imported |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module)
    assert imported <= {"dataclasses", "math", "os", "typing", "warnings"}, imported
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
    owns it: the truncation defaults, the stages' keyword defaults, the diagnostics' signatures, the
    tool's ``--device`` and ``crossval --preset`` defaults, FDTConfig's own field defaults, and the
    Simulate panel's two frame boxes as a built panel shows them. A constant retuned in config.py
    without this registry following it fails here, not in a message that names a default nobody set.
    The last assertion closes the set: no default exists that this test did not look at."""
    import argparse
    import inspect

    from core import config, orchestrator
    from core.diagnostics import ablation, identifiability, probes, sbc
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
        if key == "max_num_epochs":
            continue                                    # the one named exception, just below
        assert FIELDS[key].default == str(getattr(config, const)), (key, const, FIELDS[key].default)
    # The epoch cap is the largest 32-bit integer, which the trainer reads as no cap at all; the
    # help and the refusal line say so in words rather than print the number.
    assert config.TRAINING_MAX_NUM_EPOCHS == 2**31 - 1
    assert FIELDS["max_num_epochs"].default == "no ceiling"

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
        "min_valid": str(_default(identifiability.identifiability_laplace, "min_valid")),
        "rows":str(_default(ablation.channel_ablation, "rows")),
        "n_sweep": str(_default(ablation.channel_ablation, "n_sweep")),
        "probe_seed": str(_default(probes.probe_band, "seed")),
        "band_repeats": str(_default(probes.probe_band, "repeats")),
        "cv_max": str(_default(probes.probe_band, "cv_max")),
        "snr_min": str(_default(probes.probe_band, "snr_min")),
        "sup_min": str(_default(probes.probe_band, "sup_min")),
        "band_peak_window": str(_default(probes.probe_band, "peak_window")),
        "mask_num_runs": str(_default(probes.probe_mask, "num_runs")),
        "mask_run_size": str(_default(probes.probe_mask, "run_size")),
    }
    for key, expected in owned_by_a_signature.items():
        assert FIELDS[key].default == expected, (key, expected, FIELDS[key].default)
    # the probe checks share one seed key, so its default must be every mode's
    assert _default(probes.probe_mask, "seed") == _default(probes.probe_band, "seed")
    p = argparse.ArgumentParser()
    config_args.add_config_flags(p)
    assert FIELDS["device"].default == p.get_default("device") == "auto"

    # The FDT knobs' defaults are FDTConfig's own -- make_fdt_config's keyword defaults equal them --
    # and every key whose sweep value comes from the preset also says a sweep takes the preset's value
    # (the band and the two durations as well as n_freqs / ensemble_m). Each preset is pinned to carry
    # those keys, so the suffix cannot outlive the fact it states.
    import dataclasses
    from core import cli
    from core.config import FDTConfig
    from core.tool import build_parser
    fdt = {f.name: f.default for f in dataclasses.fields(FDTConfig)}
    owned_by_fdt_config = {"freqs_per_batch": "freqs_per_batch", "f0": "F0", "burn_in_nd": "burn_in_nd",
                           "dt_nd": "dt_nd"}
    for key, name in owned_by_fdt_config.items():
        assert FIELDS[key].default == str(fdt[name]), (key, name, FIELDS[key].default)
    set_by_the_preset = {"n_freqs": "n_freqs", "ensemble_m": "ensemble_M",
                         "t_obs_periods": "T_obs_periods", "psd_t_obs_nd": "psd_T_obs_nd"}
    for key, name in set_by_the_preset.items():
        assert FIELDS[key].default == f"{fdt[name]}, or the preset's in a sweep", (key, FIELDS[key].default)
    lo, hi = fdt["freq_bounds"]
    assert FIELDS["freq_bounds"].default == f"{lo} to {hi}, or the preset's in a sweep", \
        FIELDS["freq_bounds"].default
    for preset_name, preset in cli.SWEEP_PRESETS.items():
        assert {"freq_bounds", *set_by_the_preset.values()} <= set(preset), (preset_name, preset)
    for name in ("n_freqs", "ensemble_M", "freqs_per_batch", "F0"):
        assert _default(cli.make_fdt_config, name) == fdt[name], name
    assert (FIELDS["preset"].default == build_parser().subcommands["crossval"].get_default("preset")
            == next(iter(cli.SWEEP_PRESETS)))

    # The live simulation's two frame settings have no constant and no signature default -- the
    # panel's own boxes are constructed with them (IntField(2000), IntField(30)), so the box text of a
    # freshly built panel is what owns them. The session's settings file is empty per
    # test (tests/conftest.py), so nothing restored over the construction defaults.
    from core.gui.panels.simulate_panel import SimulatePanel
    from tests._fixtures import qt_app
    qt_app()
    sim = SimulatePanel()
    owned_by_the_simulate_panel = {"frame_steps": sim.frame_steps.text(), "fps": sim.fps.text()}
    for key, box_text in owned_by_the_simulate_panel.items():
        assert FIELDS[key].default == box_text, (key, box_text, FIELDS[key].default)

    looked_at = (set(owned_by_config) | set(owned_by_a_signature) | {"device"} | set(owned_by_fdt_config)
                 | set(set_by_the_preset) | {"freq_bounds", "preset"} | set(owned_by_the_simulate_panel))
    rest = {k: f.default for k, f in FIELDS.items() if k not in looked_at}
    no_constant = {"t_obs": "none: it must be given", "seed": "none: one is drawn and recorded"}
    assert rest == {k: no_constant.get(k) for k in rest}, rest


def test_core_runs_imports_without_torch():
    """core/runs.py is imported by every public entry point and by BOTH front ends, and the tool's
    entry sets KMP_DUPLICATE_LIB_OK and the Agg backend BEFORE any torch import. So the module must
    cost the standard library only -- which is also what lets public_entry duck-type the config
    instead of isinstance-checking SimConfig. core/__init__.py is empty, so a fresh interpreter tells
    the truth about what `import core.runs` pulls in."""
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
    """Python's root logger sits at WARNING, so a `core` logger left at NOTSET inherits it and drops
    every information record before any handler sees it: the window's pane and the artifact's
    log.txt would carry warnings only, while `caplog.set_level` in a suite hid the loss.
    core/runs.py sets the level ONCE at import; the window's handler and the tool's `main` install
    and remove handlers and never touch it.
    tests/test_tool.py::test_the_core_logger_is_at_info_by_import_and_stays_so_after_main_and_a_redirect
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
    """A composition writes two artifacts: the observation's log.txt holds the records up to its
    commit, the inference's holds the WHOLE composition's. So the public stages a composition calls
    must join the composition's buffer, not open their own: capture_run pushes only when nothing is
    active on this thread, and a nested exit pops nothing. The buffer is popped on EVERY outer exit,
    an exception's included, so a refused or crashed run leaves no handler on the logger and no tee
    on warnings.showwarning."""
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
    """The judgement channel -- PreflightWarning -- is what a reader wants in an artifact's log.txt,
    so the buffer tees warnings.showwarning while it is active. It TEES: the hook that was there
    before (the window's pane hook under a run, pytest's recorder under pytest.warns, Python's stderr
    printer otherwise) is still called with the same six arguments, and is restored on detach, so the
    tee nests cleanly inside either.
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


def test_the_run_buffer_is_per_thread_in_fact_not_only_in_name():
    """RunLog documented itself as per-thread, but its handler sits on the PROCESS-WIDE ``core``
    logger and its tee replaces the PROCESS-WIDE ``warnings.showwarning`` -- so a record or a warning
    raised from a second thread while a run is active on the main thread used to land in that run's
    buffer too. ``attach`` now records the attaching thread
    (``threading.get_ident()``), and both ``_RunLogHandler.emit`` and ``RunLog._showwarning`` check it
    before appending -- a record from the main thread still lands; one from another thread does not,
    though it must still reach whatever hook was installed before the buffer's own (here pytest's
    recorder).

    ``pytest.warns`` wraps the OUTSIDE of ``capture_run``, not the inside: entering ``pytest.warns``
    resets ``warnings.showwarning`` to its own recorder (Lib/warnings.py, ``_pytest/recwarn.py``), so
    nesting it INSIDE ``capture_run`` would tear out the buffer's tee before the other thread ever
    warns, and the owner-thread check on ``_showwarning`` would go untested -- the ``active.lines``
    assertion below would hold even with that check removed. Nested this way instead, ``pytest.warns``
    passing proves the warning reached the previous hook (the tee forwarded it), and the ``lines``
    assertion proves it was dropped from THIS buffer. A main-thread warning is checked too, so the
    test cannot pass merely because the tee is missing altogether -- and the match pattern covers BOTH
    messages, because ``WarningsChecker.__exit__`` re-``warn``s every recorded warning that does not
    match, which would otherwise leak the main-thread one into the suite's warning summary."""
    import logging
    import threading
    import warnings

    from core import runs

    log = logging.getLogger("core.tests.runs_threading")
    with pytest.warns(RuntimeWarning, match=r"from (another|the main) thread"):
        with runs.capture_run() as active:
            other_done = threading.Event()

            def _other_thread():
                log.info("from another thread")
                warnings.warn("from another thread", RuntimeWarning)
                other_done.set()

            t = threading.Thread(target=_other_thread)
            t.start()
            t.join(timeout=5)
            assert other_done.is_set(), "the other thread never finished"
            assert not any("from another thread" in line for line in active.lines), active.lines

            log.info("from the main thread")
            assert any("info from the main thread" in line for line in active.lines), active.lines

            warnings.warn("warning from the main thread", RuntimeWarning)
            assert any("warning RuntimeWarning: warning from the main thread" in line
                       for line in active.lines), active.lines


def test_the_public_entry_decorator_passes_a_sentinel_through():
    """Every public stage works on a private copy of its config. The decorator replaces the config
    argument -- the first positional, or the `cfg` keyword -- with its copy_for_run() and runs the
    call inside capture_run(). It is duck-typed, so core/runs.py never imports torch, and a stub or
    an object() sentinel (the gate tests put one on session.cfg) passes through untouched.
    functools.wraps keeps the name, the docstring, the signature and the source, which the AST pins
    on the stages read."""
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


# ── The tool's table, and the closure of the registry over the tree ──────────────────────────────
# The rule functions take the key POSITIONALLY (require_positive("t_obs", v)); a refusal raised directly
# carries it as field="…". Both shapes are scanned. `describe` is in the list although
# core/tool/config_args.py has a describe(cfg, ...) of its own: that one's first argument is a config,
# never a string literal, so a name match cannot mistake it for the registry's describe(key).
_RULE_CALLS = ("describe", "refuse", "require_given", "require_finite", "require_positive",
               "require_at_least", "require_between", "require_below", "require_choice", "require_file",
               # require_note("note", ...) has call sites in both measurement panels and the tool,
               # which the scan once missed
               "require_note",
               # the probe checks' own routes to a key: the list rule and the sentence opener of
               # core/diagnostics/probes.py, each taking the key first
               "_positive_list", "_what")


def _field_key_literals(tree) -> list:
    """Every string literal used as a refusal field key in a parsed module: the ``field="…"`` keyword
    of ANY call, the ``key="…"`` keyword a keyed rule takes (``require_seed(seed, key="probe_seed")``),
    and the first positional argument of a call to one of ``_RULE_CALLS`` (by the callee's bare name,
    whether ``require_file(...)`` or ``refusals.require_file(...)``). ``(lineno, key)`` pairs. A
    ``field=None`` or a key passed as a variable is not a literal and is not returned."""
    import ast
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        for kw in node.keywords:
            if (kw.arg in ("field", "key") and isinstance(kw.value, ast.Constant)
                    and isinstance(kw.value.value, str)):
                out.append((node.lineno, kw.value.value))
        fn = node.func
        name = fn.id if isinstance(fn, ast.Name) else fn.attr if isinstance(fn, ast.Attribute) else ""
        if (name in _RULE_CALLS and node.args and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)):
            out.append((node.lineno, node.args[0].value))
    return out


# The five QMessageBox class methods that open a box WITHOUT an instance. They are C++ statics, so
# patching the instance method ``exec`` (tests/conftest.py::_no_modal_dialogs) does not reach them.
_STATIC_BOXES = ("warning", "information", "question", "critical", "about")


def _static_message_box_calls(tree) -> list:
    """``(lineno, spelling)`` for every ``QMessageBox.<static>(...)`` call in a parsed module,
    however the class was imported (``QMessageBox.warning``, ``QtWidgets.QMessageBox.warning``).

    An INSTANCE call -- ``box = QMessageBox(self)`` then ``box.exec()`` -- is not one of these: the
    receiver has to be the class name itself."""
    import ast
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr not in _STATIC_BOXES:
            continue
        owner = node.func.value
        name = owner.attr if isinstance(owner, ast.Attribute) else getattr(owner, "id", None)
        if name == "QMessageBox":
            out.append((node.lineno, f"QMessageBox.{node.func.attr}"))
    return out


def test_no_static_message_box_is_called_anywhere_under_core():
    """The guard tests/conftest.py ASSERTS AS FACT, finally asserted.

    ``_no_modal_dialogs`` patches the INSTANCE method ``QMessageBox.exec``. The five statics are C++
    class methods and escape it, and offscreen each spins a nested event loop nothing ever closes --
    so a static added anywhere under ``core/`` STALLS the suite past the ten-minute tool-call limit
    instead of failing it. No timeout plugin is installed; the failure a reviewer would have to
    diagnose is a stack dump of a stuck process.

    ``main_window.py``'s last three statics were converted to instance boxes, and then "Nothing
    under core/ calls the statics any more ... so this guard covers every box the GUI shows" went
    into that fixture's docstring -- with nothing checking it. The repository has four scans of this
    shape already (the literal-path scan, the code-directory closure, the checkpoint-commit print
    scan and the field-key registry scan); this is the fifth, over the same
    ``CODE_ROOTS + CODE_FILES``.

    The scanner is checked on a SNIPPET first: with no static left in the tree, the tree walk alone
    passes vacuously and would go on passing if the walk stopped working."""
    import ast
    from pathlib import Path

    from tests._fixtures import CODE_FILES, CODE_ROOTS

    snippet = ast.parse(
        'QMessageBox.warning(self, "t", "m")\n'              # 1: the bare import spelling
        'QtWidgets.QMessageBox.question(self, "t", "m")\n'   # 2: through the module
        'box = QMessageBox(self)\n'                          # 3: an instance is what is wanted
        'box.setText("m")\n'
        'box.exec()\n'
        'logging.warning("not a dialog at all")\n'           # 4: the same attribute name elsewhere
        'self.log_pane.information("nor this")\n')
    assert [s for _ln, s in _static_message_box_calls(snippet)] == [
        "QMessageBox.warning", "QMessageBox.question"], _static_message_box_calls(snippet)

    root = Path(__file__).resolve().parents[1]
    files = [py for sub in CODE_ROOTS for py in sorted((root / sub).rglob("*.py"))]
    files += [root / name for name in CODE_FILES]
    assert len(files) >= 20, f"the scan walked only {len(files)} files -- CODE_ROOTS is {CODE_ROOTS}"
    found = []
    for py in files:
        for ln, spelling in _static_message_box_calls(ast.parse(py.read_text(encoding="utf-8"))):
            found.append(f"{py.relative_to(root)}:{ln}: {spelling}")
    assert not found, ("a static QMessageBox call escapes tests/conftest.py's dialog guard and HANGS "
                       "the offscreen suite instead of failing it. Build the box as an instance and "
                       "show it with .exec():\n" + "\n".join(found))


def test_every_field_key_has_a_flag_and_every_key_literal_under_core_is_registered():
    """The tool's half of the field-key rule, and the closure of the registry. A Refusal names its
    field by a key and says nothing about flags; the tool turns the key into a flag with ONE table,
    so:

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
        the same CODE_ROOTS + CODE_FILES. The scanner is checked on a snippet FIRST: a walk that
        found no literal at all would pass vacuously.
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
    # A SUBSET assertion, not an equality: these six answer to no option string today
    # and must keep doing so -- the units are declared per model, the four chi constants are
    # config.py's, and the artifact is positional. A later key with no flag is a new fact about that
    # key, not a regression in these six, and an equality here turns every such addition into a
    # false red in a file that has nothing to do with it.
    assert {"units", "chi_k_pad", "chi_max_cycles", "chi_f0", "chi_freq_bounds", "artifact"} <= \
        {k for k, f in tool_fields.FLAG.items() if f is None}
    assert tool_fields.FLAG["s_grid"] == "--s-grid" and tool_fields.FLAG["seed"] == "--seed"
    assert tool_fields.FLAG["num_posterior_samples"] == "--posterior-samples"    # the tree's spelling
    assert tool_fields.FLAG["hpd_level"] == "--level" and tool_fields.FLAG["n_directions"] == "--directions"
    assert tool_fields.FLAG["recording_probe"] == tool_fields.FLAG["recording_forced"] == "--forced"
    assert (tool_fields.FLAG["drive_amplitude"] == tool_fields.FLAG["drive_frequency"]
            == tool_fields.FLAG["drive_phase"] == "--drive")
    assert tool_fields.FLAG["max_num_epochs"] == "--max-epochs" and tool_fields.FLAG["run_size_cap"] == "--run-size"
    # the note is a flag so this table can name one (config_args.add_name_flags), and the
    # artifact is positional, so its entry is None and fix_sentence adds NOTHING -- set_note's
    # "no complete <kind> artifact named or id'd ..." refusal carries field="artifact", and the
    # ladder's line for it therefore ends at the message, with no trailing parenthetical
    assert tool_fields.FLAG["note"] == "--note" and tool_fields.fix_sentence("note") == "(--note)"
    assert tool_fields.FLAG["artifact"] is None and tool_fields.fix_sentence("artifact") == ""

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
        'other(field="not_a_refusal_kw")\n'                        # 9: field= on ANY call counts
        'require_below("nor_this_pair", lo, hi)\n'                 # 10: the ordered-pair rule
        'require_seed(seed, key="nor_this_seed")\n'                # 11: a keyed rule's key=
        '_positive_list("nor_this_list", values)\n'                # 12: the probe checks' list rule
        'sorted(rows, key=len)\n')                                 # 13: a key= that is no literal
    assert [k for _, k in sorted(_field_key_literals(snippet))] == [
        "t_obs", "no_such_box", "walk_step", "nor_this", "hpd_level", "not_a_refusal_kw", "nor_this_pair",
        "nor_this_seed", "nor_this_list"]

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


def test_every_domain_error_is_a_refusal_and_carries_a_field():
    """The six errors the tree already raises for "something asked for that the program will not
    do" -- a bad or taken name, a stale manifest, a unit pint cannot resolve, a cell FDT cannot
    run, a model definition that does not parse, a flag combination argparse cannot express --
    become Refusals, so ONE ladder on the tool and ONE routing in the window tell a refusal from a
    bug by TYPE instead of guessing from ValueError. Each keeps its
    class (every ``except StoreError`` in the tree still holds) and its docstring, and each takes
    ``field=`` through Refusal.__init__, so a raise site can name the control that answers it.

    The imports are inside the test: this module is torch-free by contract, and core.artifacts
    imports core.config, which imports torch."""
    from core.refusals import Refusal
    from core.artifacts.manifest import ManifestError
    from core.artifacts.store import StoreError
    from core.cli import UnitParseError
    from core.FDT.campaigns import FDTModelError
    from core.Models.user_model import ModelParseError
    from core.tool.config_args import UsageError

    for cls in (StoreError, ManifestError, UnitParseError, FDTModelError, ModelParseError, UsageError):
        assert issubclass(cls, Refusal) and issubclass(cls, ValueError), cls.__name__
        e = cls("why", field="name")
        assert isinstance(e, Refusal) and str(e) == "why" and e.message == "why", cls.__name__
        assert e.field == "name", cls.__name__
        assert cls("why").field is None, f"{cls.__name__}: field must default to None"
        assert cls.__doc__ and cls.__doc__.strip(), f"{cls.__name__} lost its docstring"


# ── THE root-logger handler ──────────────────────────────────────────────────────────────────────
def test_the_root_handler_exists_from_startup_so_basicconfig_never_fires():
    """WHY NO SUITE HAS EVER SEEN THIS DEFECT: pytest attaches a handler to the root logger for every
    test phase (_pytest/logging.py, ``catching_logs.__enter__``, around line 349), so
    ``len(root.handlers) == 0`` is never true during a test and ``basicConfig`` cannot fire. This test
    therefore takes pytest's root handlers off, asserts the defect's PRECONDITION -- a handler-less
    root logger makes ``logging.warning`` install a StreamHandler of its own -- asserts that
    ``logging_root.install`` prevents exactly that, and puts them back in a finally -- WHOLESALE
    (``root.handlers[:] = ...``), so a failed assertion between the deliberate trigger and its manual
    cleanup cannot leave basicConfig's StreamHandler on the root for the rest of the process.

    Then the filter's rule, during a run: a record from the ``core`` tree reaches its OWN handler once
    and the root sink NOT AT ALL (that handler already emitted it), while a library's record is
    handed to the sink once, with its level and its logger's name intact.
    """
    import logging

    from core import logging_root, runs

    root = logging.getLogger()
    pytest_handlers = root.handlers[:]
    core_seen, sink_seen = [], []

    class _Count(logging.Handler):
        def emit(self, record):
            core_seen.append((record.name, record.levelname, record.getMessage()))

    own = _Count()
    runs.LOGGER.addHandler(own)            # stands in for the front end's own handler on ``core``
    try:
        for h in pytest_handlers:
            root.removeHandler(h)
        assert root.handlers == [], "the precondition needs a handler-less root logger"

        logging.warning("a library warning, with no handler installed")
        assert len(root.handlers) == 1 and isinstance(root.handlers[0], logging.StreamHandler), \
            "logging.warning no longer calls basicConfig: the defect this handler prevents is gone"
        for h in root.handlers[:]:
            root.removeHandler(h)

        logging_root.install(sink_seen.append)
        assert logging_root.installed() is True
        logging.warning("a library warning, with THE root handler installed")
        assert len(root.handlers) == 1, [type(h).__name__ for h in root.handlers]
        assert [r.getMessage() for r in sink_seen] == \
            ["a library warning, with THE root handler installed"], sink_seen
        assert sink_seen[-1].levelno == logging.WARNING and sink_seen[-1].name == "root", \
            (sink_seen[-1].levelno, sink_seen[-1].name)

        sink_seen.clear()
        logging.getLogger("core.probe").info("the pipeline's own voice")
        assert sink_seen == [], "a core record reached the root sink: the filter is not dropping it"
        assert core_seen == [("core.probe", "INFO", "the pipeline's own voice")], core_seen

        logging_root.remove()
        assert logging_root.installed() is False and root.handlers == []
    finally:
        logging_root.remove()
        runs.LOGGER.removeHandler(own)
        root.handlers[:] = pytest_handlers


def test_caplog_still_sees_core_records_while_the_root_handler_is_installed(caplog):
    """The drop is a ``logging.Filter`` ON THE HANDLER, never ``core.propagate = False``: ``caplog``
    reads off the root logger and the propagation is pinned by
    test_the_core_logger_is_at_info_by_import above, and ``RunLog`` plus both front-end handlers sit
    on the ``core`` logger itself, so a filter leaves all three untouched.

    A stand-in handler sits on ``core`` here, as a run's would: with none attached the rule hands a
    ``core`` record to the sink as well (the between-runs case, pinned in the test below)."""
    import logging

    from core import logging_root, runs

    class _RunsHandler(logging.Handler):
        def emit(self, record):
            pass

    seen = []
    stand_in = _RunsHandler()
    runs.LOGGER.addHandler(stand_in)
    logging_root.install(seen.append)
    try:
        with caplog.at_level(logging.INFO, logger="core"):
            logging.getLogger("core.probe").info("a record caplog must still see")
            logging.getLogger("a_library").warning("and one the sink may have")
    finally:
        logging_root.remove()
        runs.LOGGER.removeHandler(stand_in)
    assert "a record caplog must still see" in caplog.text
    assert [r.getMessage() for r in seen] == ["and one the sink may have"], seen


def test_the_root_sink_takes_exactly_what_no_handler_below_the_root_has_emitted():
    """The filter's rule is "no logger between the record's own and the root (the root excluded) has
    a handler that already emitted it", replacing a rule by logger NAME that lost three kinds of
    record. Run as a plain script would run it -- pytest's root handlers off, put back
    WHOLESALE in the finally -- because what is under test is what reaches the root.

    (a) a ``core`` WARNING with NO ``core`` handler attached (the window between runs) reaches the
        sink instead of vanishing, and renders in the tool's own shape: ``warning: ...``, never
        ``library:``; an information record renders bare;
    (b) the same record with a ``core`` handler attached (a run) does NOT reach the sink;
    (c) a library with a handler of its OWN (pytensor adds one at import) is emitted once, by that
        handler, and not a second time by the sink -- while a ``NullHandler``, which emits nothing,
        does not count, so a library that added only that (pint does) still reaches the sink once;
    (d) a library's ``log.exception`` carries its traceback, with ``library:`` on the first line only.
    """
    import logging

    from core import logging_root, runs

    root = logging.getLogger()
    pytest_handlers = root.handlers[:]
    seen = []
    own_seen = []

    class _Own(logging.Handler):
        def emit(self, record):
            own_seen.append(record.getMessage())

    chatty = logging.getLogger("a_chatty_library")          # installs its own, as pytensor does
    polite = logging.getLogger("a_polite_library")          # installs a NullHandler, as pint does
    lib = logging.getLogger("a_library")
    chatty_own, polite_null = _Own(), logging.NullHandler()
    core_own = _Own()
    try:
        root.handlers[:] = []
        chatty.addHandler(chatty_own)
        polite.addHandler(polite_null)
        logging_root.install(seen.append)
        assert runs.LOGGER.handlers == [], \
            f"a core handler is attached before the between-runs case: {runs.LOGGER.handlers}"

        # (a) no core handler: the record reaches the sink, in PRISM's own shape
        logging.getLogger("core.probe").warning("the pipeline's own voice, between runs")
        logging.getLogger("core.probe").info("and its information record")
        assert [logging_root.render(r) for r in seen] == [
            "warning: the pipeline's own voice, between runs", "and its information record"], seen

        # (b) a core handler attached: it emitted the record, so the sink stays silent
        seen.clear()
        runs.LOGGER.addHandler(core_own)
        logging.getLogger("core.probe").warning("the pipeline's own voice, during a run")
        runs.LOGGER.removeHandler(core_own)
        assert seen == [], [r.getMessage() for r in seen]
        assert own_seen == ["the pipeline's own voice, during a run"], own_seen

        # (c) a library's own handler emitted it; a NullHandler emitted nothing
        seen.clear()
        own_seen.clear()
        chatty.warning("printed by its own handler")
        polite.warning("printed by nobody but the sink")
        assert own_seen == ["printed by its own handler"], own_seen
        assert [logging_root.render(r) for r in seen] == \
            ["library: a_polite_library: printed by nobody but the sink"], seen

        # (d) a traceback survives, prefixed on its first line only
        seen.clear()
        try:
            raise ValueError("the library's own failure")
        except ValueError:
            lib.exception("it failed")
        assert len(seen) == 1, seen
        text = logging_root.render(seen[0])
        lines = text.splitlines()
        assert lines[0] == "library: a_library: it failed", lines
        assert lines[1] == "Traceback (most recent call last):", lines
        assert lines[-1] == "ValueError: the library's own failure", lines
        assert [ln for ln in lines if ln.startswith("library:")] == lines[:1], text
    finally:
        logging_root.remove()
        runs.LOGGER.removeHandler(core_own)
        chatty.removeHandler(chatty_own)
        polite.removeHandler(polite_null)
        root.handlers[:] = pytest_handlers


def test_the_run_log_handler_does_not_count_as_having_emitted_a_record():
    """``runs._RunLogHandler`` appends to a LIST. It emits to nobody, which is exactly the standing a
    ``NullHandler`` already had in ``_already_emitted`` -- and it did not have it. So while ANY run
    was active the sink stayed silent for every ``core`` record, and two things followed:

      * a run whose console redirect was DECLINED showed the operator nothing live, although the
        record still survived into log.txt -- the case the module docstring claimed worked;
      * a ``core`` record raised from a thread OTHER than the run's owner reached NOTHING AT ALL:
        ``_RunLogHandler.emit`` drops it (that buffer holds this run's own lines, by thread) and the
        root sink suppressed it as already emitted.

    A handler that DOES emit still suppresses the duplicate -- that is the whole point of the walk --
    so the third leg attaches one beside the run log and the sink goes quiet again.
    """
    import logging
    import threading

    from core import logging_root, runs

    root = logging.getLogger()
    pytest_handlers = root.handlers[:]
    seen = []
    log = runs.RunLog()
    emitted = []

    class _Emits(logging.Handler):
        def emit(self, record):
            emitted.append(record.getMessage())

    pump_like = _Emits()
    try:
        root.handlers[:] = []
        logging_root.install(seen.append)
        assert runs.LOGGER.handlers == [], \
            f"a core handler is attached before this test attaches one: {runs.LOGGER.handlers}"
        log.attach()

        logging.getLogger("core.probe").warning("a record whose console redirect was declined")
        assert [logging_root.render(r) for r in seen] == \
            ["warning: a record whose console redirect was declined"], seen
        assert any("a record whose console redirect was declined" in ln for ln in log.lines), \
            log.lines

        seen.clear()
        threads_say = threading.Thread(
            target=lambda: logging.getLogger("core.probe").warning("from another thread"))
        threads_say.start()
        threads_say.join()
        assert [logging_root.render(r) for r in seen] == ["warning: from another thread"], seen
        assert not any("from another thread" in ln for ln in log.lines), \
            ("the run log is per thread; this record has nowhere else to go", log.lines)

        seen.clear()
        runs.LOGGER.addHandler(pump_like)
        logging.getLogger("core.probe").warning("a record the pane already showed")
        assert seen == [], [r.getMessage() for r in seen]
        assert emitted == ["a record the pane already showed"], emitted
    finally:
        runs.LOGGER.removeHandler(pump_like)
        log.detach()
        logging_root.remove()
        root.handlers[:] = pytest_handlers


def test_logging_root_imports_only_the_standard_librarys_logging():
    """``python -m core --help`` must stay torch-free and the tool imports this module at its top, so
    the same pin the refusals module carries applies here: the import statements name ``logging`` and
    nothing else, a fresh interpreter that imports it has no torch loaded, and no logger's level is
    touched (core/runs.py sets the ``core`` level once, at import, and nothing else ever does)."""
    src = (REPO / "core" / "logging_root.py").read_text(encoding="utf-8")
    imported = set()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Import):
            imported |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module)
    assert imported == {"logging"}, imported
    calls = [ast.unparse(n.func) for n in ast.walk(ast.parse(src))
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)]
    assert not any(c.endswith("setLevel") for c in calls), calls
    probe = ("import sys, core.logging_root; "
             "bad = sorted(m for m in sys.modules if m == 'torch' or m.startswith('torch.')); "
             "sys.exit(repr(bad) if bad else 0)")
    r = subprocess.run([sys.executable, "-c", probe], cwd=str(REPO), capture_output=True, text=True,
                       timeout=120)
    assert r.returncode == 0, r.stderr
