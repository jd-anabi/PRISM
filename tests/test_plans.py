"""The plan reader: a hand-edited file of settings, read whole or refused with every problem named; and the window's
side of plans: the list on step 1, the fill of steps 1 to 3, where each value came from, the save and the run's
history.

Nothing at the top of the file reaches torch or builds a widget: ``core/plans.py`` is standard library on purpose,
and the tests that hold it against the rest of the program, or build the window, import what they need inside the
test. Every file is written byte for byte (``_plan``), since ``write_text`` would turn each line end
into this computer's and hide the difference between one editor's file and another's.
"""
import ast
import hashlib
import subprocess
import sys
from pathlib import Path

import pytest

from core import plans
from core.plans import ALLOWED, CLASSES, HEADINGS, PlanRefusal, plan_text, read_plan
from core.refusals import (FIELDS, Refusal, _what, require_at_least, require_between, require_positive)

REPO = Path(__file__).resolve().parents[1]

ABOUT = "The 13-parameter retrain on the temperature box: 10,000 batches, network 256 by 10."
EXAMPLE = f"""# About this plan
{ABOUT}

# Model and options
chi_mode = on                  # Kind of observation: Frequency response

# Prior
bounds = master_tier1.txt      # Bounds file

# Training
num_runs = 10000               # Batches
hidden_features = 256          # Network width
num_transforms = 10            # Network depth

# Narrowing round
hidden_features = 256          # Network width
num_transforms = 10            # Network depth
"""
EXAMPLE_VALUES = {("Model and options", "chi_mode"): True, ("Prior", "bounds"): "master_tier1.txt",
                  ("Training", "num_runs"): 10000, ("Training", "hidden_features"): 256,
                  ("Training", "num_transforms"): 10, ("Narrowing round", "hidden_features"): 256,
                  ("Narrowing round", "num_transforms"): 10}
EXAMPLE_REMARKS = {("Model and options", "chi_mode"): "Kind of observation: Frequency response",
                   ("Prior", "bounds"): "Bounds file", ("Training", "num_runs"): "Batches",
                   ("Training", "hidden_features"): "Network width", ("Training", "num_transforms"): "Network depth",
                   ("Narrowing round", "hidden_features"): "Network width",
                   ("Narrowing round", "num_transforms"): "Network depth"}

OPTIONS = "# Model and options\nchi_mode = on\n"
THE_HEADINGS = ("The headings are: About this plan, Model and options, Prior, Training, Calibration, Inference, "
                "Narrowing round.")


def _plan(tmp_path, text, name="retrain") -> Path:
    path = tmp_path / f"{name}.txt"
    path.write_bytes(text.encode("utf-8"))
    return path


def _refused(tmp_path, text) -> PlanRefusal:
    with pytest.raises(PlanRefusal) as refused:
        read_plan(_plan(tmp_path, text))
    return refused.value


def _rule(rule, *arguments, **keywords) -> str:
    """What a registry rule says, so a sentence below is the rule's own and not a copy that could drift."""
    with pytest.raises(Refusal) as refused:
        rule(*arguments, **keywords)
    return refused.value.message


def test_a_plan_reads_back_what_was_written(tmp_path):
    """A plan as the window writes one with its description added by hand: every kind of line, the remarks
    not read, the description kept, and only what the file gives."""
    path = _plan(tmp_path, EXAMPLE)
    plan = read_plan(path)
    assert plan.values == EXAMPLE_VALUES
    assert plan.values[("Training", "num_runs")] == 10000
    assert plan.values[("Model and options", "chi_mode")] is True
    assert plan.values[("Prior", "bounds")] == "master_tier1.txt"
    assert plan.about == ABOUT
    assert (plan.name, plan.path) == ("retrain", path)
    assert len(plan.sha256) == 64
    assert read_plan(str(path)).path == path


def test_training_and_the_narrowing_round_share_names_under_their_own_headings(tmp_path):
    """Two steps take the same names with different values, so the values are keyed by heading and name:
    keyed by name the second would replace the first, or read as set twice."""
    plan = read_plan(_plan(tmp_path, "# Training\nnum_runs = 10000\n\n# Narrowing round\nnum_runs = 2000\n"))
    assert plan.values == {("Training", "num_runs"): 10000, ("Narrowing round", "num_runs"): 2000}


def test_every_problem_is_listed_with_its_line_and_nothing_is_returned(tmp_path):
    """All of them at once, in file order, each with its line, in one refusal whose field is the file. A
    setting under a misspelt heading is still judged, so one slip does not turn into a list of false ones."""
    refusal = _refused(tmp_path, "# About this plan\nThree things are wrong below.\n\n# Trainng\n\n"
                                 "num_runs 10000\nnum_run = 10000\n")
    assert isinstance(refusal, Refusal)
    assert refusal.field == "plan"
    assert refusal.message == str(refusal) == "This plan cannot be used: 3 problems in retrain.txt"
    assert [problem.line for problem in refusal.problems] == [4, 6, 7]
    assert all(problem.message.startswith(f"line {problem.line}: ") for problem in refusal.problems)
    assert refusal.problems[2].message == "line 7: 'num_run' is not a setting PRISM has. The nearest is num_runs."
    single = _refused(tmp_path, "# Training\nnum_runs = abc\n")
    assert single.message == "This plan cannot be used: 1 problem in retrain.txt"


def test_a_line_under_a_misspelt_heading_is_still_judged(tmp_path):
    """What the line itself can be told is told at the first reading: its value, and a name a plan may not hold."""
    problems = _refused(tmp_path, "# Trainng\nnum_runs = abc\n").problems
    assert [problem.line for problem in problems] == [1, 2]
    assert problems[1].message == f"line 2: {_what('num_runs')} must be a whole number; got 'abc'."
    problems = _refused(tmp_path, "# Trainng\nseed = 1\n").problems
    assert [(problem.line, problem.key) for problem in problems] == [(1, None), (2, "seed")]


@pytest.mark.parametrize("text, line, key, sentence", [
    pytest.param("\n# Trainng\n", 2, None, f"'# Trainng' is not a heading a plan has. {THE_HEADINGS}",
                 id="unknown heading"),
    pytest.param("# Training\n# num_runs = 10000\n", 2, None,
                 f"'# num_runs = 10000' is not a heading a plan has. {THE_HEADINGS}", id="a setting commented out"),
    pytest.param("# Training\nnum_runs 10000\n", 2, None,
                 "'num_runs 10000' is not a setting line. Write name = value under a heading.", id="not a setting"),
    pytest.param("num_runs = 10000\n# Training\n", 1, None,
                 "'num_runs = 10000' is not a setting line. Write name = value under a heading.",
                 id="above the first heading"),
    pytest.param("# Training\nnum_run = 1\n", 2, None, "'num_run' is not a setting PRISM has. The nearest is num_runs.",
                 id="unknown with a near name"),
    pytest.param("# Training\nmemory = 1\n", 2, None, "'memory' is not a setting PRISM has.",
                 id="unknown with none near"),
    pytest.param("# Training\nruns = 1\n", 2, None, "'runs' is not a setting PRISM has.", id="a name just too far off"),
    pytest.param("# Training\nNum_Runs = 1\n", 2, None,
                 "'Num_Runs' is not a setting PRISM has. The nearest is num_runs.", id="a name in another case"),
    pytest.param("# Training\nsed = 1\n", 2, None, "'sed' is not a setting PRISM has. The nearest is seed.",
                 id="the nearest is one a plan cannot hold"),
    pytest.param("# Training\nseed = 1\n", 2, "seed",
                 f"{_what('seed')} cannot be set by a plan: it belongs to one run.", id="not allowed"),
    pytest.param("# Training\nbounds = a.txt\n", 2, "bounds",
                 f"{_what('bounds')} belongs under '# Prior', not '# Training'.", id="under the wrong heading"),
    pytest.param("# Prior\nnum_runs = 1\n", 2, "num_runs",
                 f"{_what('num_runs')} belongs under '# Training', not '# Prior'.", id="a name two headings hold"),
    pytest.param("# Training\n\nnum_runs = 1\nnum_runs = 2\n", 4, "num_runs",
                 f"{_what('num_runs')} is set twice under '# Training' (first on line 3).", id="named twice"),
    pytest.param("# Training\nnum_runs = abc\n", 2, "num_runs",
                 f"{_what('num_runs')} must be a whole number; got 'abc'.", id="not whole"),
    pytest.param("# Training\nnum_runs = 10,000\n", 2, "num_runs",
                 f"{_what('num_runs')} must be a whole number; got '10,000'.", id="a whole number with separators"),
    pytest.param("# Training\nnum_runs = 1_000\n", 2, "num_runs",
                 f"{_what('num_runs')} must be a whole number; got '1_000'.", id="a form Python would take"),
    pytest.param("# Training\nnum_runs =\n", 2, "num_runs",
                 f"{_what('num_runs')} must be a whole number; got ''.", id="a whole number left out"),
    pytest.param("# Training\nlearning_rate = 1,5\n", 2, "learning_rate",
                 f"{_what('learning_rate')} must be a number; got '1,5'.", id="not a number"),
    pytest.param("# Training\nlearning_rate = inf\n", 2, "learning_rate",
                 f"{_what('learning_rate')} must be a number; got 'inf'.", id="a word Python would take as a number"),
    pytest.param("# Model and options\nreparam_rotate = yes\n", 2, "reparam_rotate",
                 f"{_what('reparam_rotate')} must be on or off; got 'yes'.", id="not on or off"),
    pytest.param("# Model and options\nchi_mode = ON\n", 2, "chi_mode",
                 f"{_what('chi_mode')} must be on or off; got 'ON'.", id="a switch in capitals"),
    pytest.param("# Prior\nbounds =\n", 2, "bounds",
                 f"{_what('bounds')} must name a file in the model's bounds folder; got ''.",
                 id="a bounds name left out"),
    pytest.param("# Model and options\nunits =     # Units\n", 2, "units",
                 f"{_what('units')} are blank: write at least one unit, or leave the line out.",
                 id="the units left out"),
    pytest.param("# Training\nnum_runs = 0\n", 2, "num_runs", _rule(require_at_least, "num_runs", 0, 1),
                 id="below its least"),
    pytest.param("# Training\nrun_size_cap = -1\n", 2, "run_size_cap", _rule(require_at_least, "run_size_cap", -1, 0),
                 id="below its least of nought"),
    pytest.param("# Prior\nmin_cluster_size = 1\n", 2, "min_cluster_size",
                 _rule(require_at_least, "min_cluster_size", 1, 2), id="below its least of two"),
    pytest.param("# Training\nlearning_rate = 0\n", 2, "learning_rate", _rule(require_positive, "learning_rate", 0.0),
                 id="not above nought"),
    pytest.param("# Narrowing round\nhpd_level = 1\n", 2, "hpd_level",
                 _rule(require_between, "hpd_level", 1.0, 0, 1, open_lo=True, open_hi=True), id="outside its band"),
    pytest.param("# Narrowing round\nhpd_level = 0\n", 2, "hpd_level",
                 _rule(require_between, "hpd_level", 0.0, 0, 1, open_lo=True, open_hi=True), id="below its band"),
    pytest.param(f"{OPTIONS}chi_n_freqs = 1\n", 3, "chi_n_freqs", _rule(require_at_least, "chi_n_freqs", 1, 2),
                 id="fewer probes than two"),
    pytest.param(f"{OPTIONS}chi_k_pad = 1\n", 3, "chi_k_pad", _rule(require_at_least, "chi_k_pad", 1, 2),
                 id="fewer slots than two"),
    pytest.param(f"{OPTIONS}chi_k_pad = 24\nchi_n_freqs = 30\n", 4, "chi_n_freqs",
                 f"{_what('chi_n_freqs')} must be at most 24; got 30 (default 6).", id="too many probes"),
    pytest.param(f"{OPTIONS}chi_k_pad = 30\n", 3, "chi_k_pad",
                 f"{_what('chi_k_pad')} must be at most 24; got 30 (default 12).", id="too many slots"),
    pytest.param(f"{OPTIONS}chi_max_cycles = 2\n", 3, "chi_max_cycles",
                 f"{_what('chi_max_cycles')} must be greater than 2, the floor below which a probe is masked; "
                 "got 2 (default 20).", id="a ceiling that does not clear the floor"),
    pytest.param(f"{OPTIONS}chi_max_cycles = 1e999\n", 3, "chi_max_cycles",
                 f"{_what('chi_max_cycles')} must be a finite number; got inf (default 20.0).",
                 id="a ceiling past what a float holds"),
    pytest.param(f"{OPTIONS}chi_n_freqs = 14\n", 3, "chi_n_freqs",
                 f"{_what('chi_n_freqs')} must be at most the number of slots, 12 (the built-in); got 14.",
                 id="more probes than the built-in slots"),
    pytest.param(f"{OPTIONS}chi_k_pad = 4\n", 3, "chi_n_freqs",
                 f"{_what('chi_n_freqs')} must be at most the number of slots, 4; got 6 (the built-in).",
                 id="fewer slots than the built-in probes"),
    pytest.param(f"{OPTIONS}chi_k_pad = 4\n\nchi_n_freqs = 6\n", 5, "chi_n_freqs",
                 f"{_what('chi_n_freqs')} must be at most the number of slots, 4; got 6.",
                 id="both counts in the file, said on the probes' line"),
])
def test_each_kind_of_problem_says_the_sentence_of_the_table(tmp_path, text, line, key, sentence):
    """One case for each thing a hand-edited plan can get wrong: the one problem, its line, the setting it
    concerns, and the sentence after "line N: ". Each opens with the registry's phrase for the key, and the
    range sentences are the registry's own rules' (a rule's words are not repeated here)."""
    problems = _refused(tmp_path, text).problems
    assert len(problems) == 1, problems
    assert (problems[0].line, problems[0].key) == (line, key)
    assert problems[0].message == f"line {line}: {sentence}"


@pytest.mark.parametrize("switch", ["", "chi_mode = off\n"], ids=["no switch", "the switch off"])
@pytest.mark.parametrize("name, value", [("chi_n_freqs", 4), ("chi_k_pad", 12), ("chi_max_cycles", 20)])
def test_each_chi_setting_needs_the_switch(tmp_path, switch, name, value):
    """With the switch off the window sends none of the three, so a plan that holds one would say what no run reads."""
    problems = _refused(tmp_path, f"# Model and options\n{switch}{name} = {value}\n").problems
    assert [(problem.heading, problem.key) for problem in problems] == [("Model and options", name)]
    assert problems[0].line == 2 + bool(switch)
    assert problems[0].message.endswith(f": {_what(name)} needs chi_mode = on in the same plan.")


def test_a_count_equal_to_the_slots_is_read(tmp_path):
    """"At most" holds at the end it names, each way round: the probes up to the built-in slots, the slots down to the
    built-in probes."""
    assert read_plan(_plan(tmp_path, f"{OPTIONS}chi_n_freqs = 12\n")).values[("Model and options", "chi_n_freqs")] == 12
    assert read_plan(_plan(tmp_path, f"{OPTIONS}chi_k_pad = 6\n")).values[("Model and options", "chi_k_pad")] == 6


@pytest.mark.parametrize("text, line, key", [
    pytest.param("# Training\nchi_mode = on\n\n# Model and options\nchi_k_pad = 12\n", 2, "chi_mode",
                 id="the switch under the wrong heading"),
    pytest.param("# Model and optins\nchi_mode = on\n\n# Model and options\nchi_k_pad = 12\n", 1, None,
                 id="the switch under a misspelt heading"),
    pytest.param("# Prior\nchi_k_pad = 24\n\n# Model and options\nchi_mode = on\nchi_n_freqs = 14\n", 2, "chi_k_pad",
                 id="the slots under the wrong heading"),
    pytest.param("# Model and options\nchi_n_freqs = 14\n", 2, "chi_n_freqs", id="past the slots with no switch"),
    pytest.param("# Model and options\nchi_mode = off\nchi_k_pad = 4\n", 3, "chi_k_pad",
                 id="below the probes with the switch off"),
    pytest.param("chi_mode = on\n# Model and options\nchi_k_pad = 12\n", 1, None,
                 id="the switch above the first heading"),
    pytest.param("chi_k_pad = 24\n# Model and options\nchi_mode = on\nchi_n_freqs = 14\n", 1, None,
                 id="the slots above the first heading"),
])
def test_one_slip_is_one_problem(tmp_path, text, line, key):
    """A file that is wrong in one place is told one thing: not that it lacks a line it holds, and not of a comparison
    that means nothing without the switch. A line refused wherever it stands is not read by the rules across lines."""
    problems = _refused(tmp_path, text).problems
    assert [(problem.line, problem.key) for problem in problems] == [(line, key)]


NOT_FOUND = "The plan was not found: '{path}'."
NOT_UTF8 = "The plan could not be read as UTF-8 text: '{path}'. Save it as UTF-8."


@pytest.mark.parametrize("make_file, sentence", [
    pytest.param(lambda folder: folder / "gone.txt", NOT_FOUND, id="missing"),
    pytest.param(lambda folder: folder, NOT_FOUND, id="a folder"),
    pytest.param(lambda folder: _bytes(folder, b"# About this plan\n\xe9t\xe9\n"), NOT_UTF8, id="an accent in ANSI"),
    pytest.param(lambda folder: _bytes(folder, "# Training\n".encode("utf-16")), NOT_UTF8, id="UTF-16"),
])
def test_a_file_that_cannot_be_read_is_one_problem_with_no_line(tmp_path, make_file, sentence):
    """The list offers every file in the folder, and a file can go between the list and the press, or stand there
    in an encoding this reader does not take. Neither is a traceback, and a file that is there is not called gone."""
    path = make_file(tmp_path)
    with pytest.raises(PlanRefusal) as refused:
        read_plan(path)
    assert refused.value.field == "plan"
    assert refused.value.message == f"This plan cannot be used: 1 problem in {path.name}"
    assert [(problem.line, problem.key, problem.message) for problem in refused.value.problems] == [
        (None, None, sentence.format(path=path))]


def _bytes(folder, data) -> Path:
    path = folder / "retrain.txt"
    path.write_bytes(data)
    return path


@pytest.mark.parametrize("key, reason", [
    ("chi_f0", "it is fixed by measurement"), ("chi_freq_bounds", "it is fixed by measurement"),
    ("device", "it belongs to the computer"),
    ("name", "it belongs to one run"), ("note", "it belongs to one run"), ("seed", "it belongs to one run"),
    ("new_run", "it is a consent"), ("accept_truncated", "it is a consent"),
    ("accept_other_observation", "it is a consent"), ("resume", "it is a consent"),
    ("prior", "it is a pick, not a setting"), ("resume_fit", "it is a pick, not a setting"),
    ("t_obs", "it is a pick, not a setting"), ("chi_f0_si", "it is a pick, not a setting"),
    ("recording_probe", "it is a pick, not a setting"), ("drive_amplitude", "it is a pick, not a setting"),
    ("model", "it is the folder the plan stands in"),
    ("n_freqs", "it is not a setting of the inference steps"), ("plan", "it is not a setting of the inference steps"),
    # named like a recording's drive, but the drive check's own knob
    ("drive_t_obs", "it is not a setting of the inference steps"),
    ("drive_repeats", "it is not a setting of the inference steps"),
])
def test_what_a_plan_may_not_hold_is_refused_with_its_reason(tmp_path, key, reason):
    """A registered name that is no plan setting is told apart from a misspelt one: it says why, and the
    problem carries the key, under any heading."""
    problems = _refused(tmp_path, f"# Training\n{key} = 1\n").problems
    assert [(problem.line, problem.key) for problem in problems] == [(2, key)]
    assert problems[0].message == f"line 2: {_what(key)} cannot be set by a plan: {reason}."


def test_a_whole_number_may_carry_its_sign(tmp_path):
    assert read_plan(_plan(tmp_path, "# Training\nnum_runs = +5\n")).values == {("Training", "num_runs"): 5}


def test_a_kind_is_checked_before_the_registrys_rule(tmp_path):
    """The rule alone would have read this value and truncated it to 10000: the reader refuses the text."""
    problems = _refused(tmp_path, "# Training\nnum_runs = 10000.5\n").problems
    assert [problem.message for problem in problems] == [
        f"line 2: {_what('num_runs')} must be a whole number; got '10000.5'."]


def test_every_registered_key_is_allowed_or_refused_for_a_stated_reason():
    """STANDING. Every key of the registry is a setting a plan holds or has its reason not to be, so a key
    added to the registry cannot slip past the reader unclassified, and which heading holds which name is written
    out here and not read back from the table under test. The picks are listed by name: "every key that begins
    with drive" would take in the drive check's own knobs."""
    names = {name for held in ALLOWED.values() for name in held}
    assert not names & set(CLASSES)
    assert names | set(CLASSES) == set(FIELDS)
    assert ALLOWED == {
        "Model and options": ("chi_mode", "chi_n_freqs", "chi_k_pad", "chi_max_cycles", "reparam_rotate", "units"),
        "Prior": ("bounds", "num_iterations", "sweep_batch", "max_sets", "walk_step", "stability_units",
                  "min_cluster_size", "min_samples"),
        "Training": ("num_runs", "run_size_cap", "hidden_features", "num_transforms", "learning_rate",
                     "stop_after_epochs", "max_num_epochs", "checkpoint_every", "fisher_m", "fisher_dz",
                     "fisher_points", "expected_epochs"),
        "Calibration": ("n_cal", "cal_n_scales", "num_posterior_samples"),
        "Inference": ("n_samples",),
        "Narrowing round": ("n_directions", "hpd_level", "num_runs", "run_size_cap", "hidden_features",
                            "num_transforms", "learning_rate", "stop_after_epochs", "max_num_epochs",
                            "checkpoint_every")}
    assert tuple(ALLOWED) == HEADINGS[1:]
    assert [len(ALLOWED[heading]) for heading in HEADINGS[1:]] == [6, 8, 12, 3, 1, 10]
    assert sum(len(held) for held in ALLOWED.values()) == 40 and len(names) == 32
    reasons = {}
    for key, reason in CLASSES.items():
        reasons.setdefault(reason, set()).add(key)
    assert {reason: len(keys) for reason, keys in reasons.items()} == {
        "it is fixed by measurement": 2, "it belongs to the computer": 1, "it belongs to one run": 3,
        "it is a consent": 4, "it is a pick, not a setting": 13, "it is the folder the plan stands in": 1,
        "it is not a setting of the inference steps": 58}
    assert reasons["it is a pick, not a setting"] == {
        "prior", "posterior", "observation", "cell", "resume_fit", "t_obs", "chi_f0_si", "recording_spont",
        "recording_forced", "recording_probe", "drive_amplitude", "drive_frequency", "drive_phase"}


# One value for each allowed name: a value the rules allow, with the built-ins beside it.
ONE_VALUE_EACH = {
    "chi_mode": True, "chi_n_freqs": 2, "chi_k_pad": 24, "chi_max_cycles": 2.5, "reparam_rotate": False,
    "units": "nm ms pN kHz", "bounds": "sub/master_tier1.txt", "num_iterations": 1, "sweep_batch": 0, "max_sets": 1,
    "walk_step": 1e-05, "stability_units": 1000.0, "min_cluster_size": 2, "min_samples": 1, "num_runs": 1,
    "run_size_cap": 0, "hidden_features": 1, "num_transforms": 1, "learning_rate": 0.001, "stop_after_epochs": 1,
    "max_num_epochs": 2147483647, "checkpoint_every": 0, "fisher_m": 1, "fisher_dz": 0.1, "fisher_points": 1,
    "expected_epochs": 1, "n_cal": 1, "cal_n_scales": 1, "num_posterior_samples": 1, "n_samples": 1,
    "n_directions": 1, "hpd_level": 0.999}
PAIRS = [(heading, name) for heading, names in ALLOWED.items() for name in names]
DESCRIPTION = "The first line of a description.\nIts second line, with a = sign, is still free text."


def _with_the_switch(values) -> dict:
    """The three chi settings are refused without the switch in the same plan."""
    chi = any(name in ("chi_n_freqs", "chi_k_pad", "chi_max_cycles") for _heading, name in values)
    return {("Model and options", "chi_mode"): True, **values} if chi else values


@pytest.mark.parametrize("pair", PAIRS, ids=lambda pair: f"{pair[0]}: {pair[1]}")
def test_read_plan_of_plan_text_gives_the_values_and_the_description_back(tmp_path, pair):
    """Every setting a plan can hold, one at a time, with a description of two lines: what is written reads
    back equal and of the same type, so a saved plan means what the forms held."""
    assert set(ONE_VALUE_EACH) == {name for _heading, name in PAIRS}
    values = _with_the_switch({pair: ONE_VALUE_EACH[pair[1]]})
    plan = read_plan(_plan(tmp_path, plan_text(values, DESCRIPTION)))
    assert plan.values == values
    assert type(plan.values[pair]) is type(ONE_VALUE_EACH[pair[1]])
    assert plan.about == DESCRIPTION


# The least of every whole number, written out here and not read from the module.
LEAST = {
    "chi_n_freqs": 2, "chi_k_pad": 2, "num_iterations": 1, "sweep_batch": 0, "max_sets": 1, "min_cluster_size": 2,
    "min_samples": 1, "num_runs": 1, "run_size_cap": 0, "hidden_features": 1, "num_transforms": 1,
    "stop_after_epochs": 1, "max_num_epochs": 1, "checkpoint_every": 0, "fisher_m": 1, "fisher_points": 1,
    "expected_epochs": 1, "n_cal": 1, "cal_n_scales": 1, "num_posterior_samples": 1, "n_samples": 1, "n_directions": 1}


@pytest.mark.parametrize("pair", [pair for pair in PAIRS if pair[1] in LEAST], ids=lambda pair: f"{pair[0]}: {pair[1]}")
def test_one_below_every_whole_numbers_least_is_refused(tmp_path, pair):
    """Each under each heading that holds it: the value the rule's own sentence names, one step past what the
    table allows."""
    assert set(LEAST) == {name for name, value in ONE_VALUE_EACH.items() if type(value) is int}
    heading, name = pair
    below = LEAST[name] - 1
    top = OPTIONS if heading == "Model and options" else f"# {heading}\n"
    problems = _refused(tmp_path, f"{top}{name} = {below}\n").problems
    assert [problem.message.split(": ", 1)[1] for problem in problems] == [
        _rule(require_at_least, name, below, LEAST[name])]


def test_every_plan_setting_at_once_reads_back(tmp_path):
    """The whole table in one file, under its headings in order, and the switch's other way."""
    values = {pair: ONE_VALUE_EACH[pair[1]] for pair in PAIRS}
    text = plan_text(values, DESCRIPTION)
    assert read_plan(_plan(tmp_path, text)).values == values
    assert [line[2:] for line in text.splitlines() if line.startswith("#")] == list(HEADINGS)
    switched = {("Model and options", "chi_mode"): False}
    assert read_plan(_plan(tmp_path, plan_text(switched))).values == switched


@pytest.mark.parametrize("value", [1e-05, 1e+16, 1.5e-07, 0.30000000000000004, 1.0, 123456789.12345679, 5e-324],
                         ids=str)
def test_every_form_python_writes_a_number_in_reads_back_the_same(tmp_path, value):
    """``str`` writes a small or a large float with an exponent: the number form takes it, and the value comes
    back to the last bit."""
    pair = ("Training", "learning_rate")
    assert read_plan(_plan(tmp_path, plan_text({pair: value}))).values == {pair: value}


def test_plan_text_writes_the_form_a_person_reads():
    """The file as a person meets it: the headings in their order, the settings in the table's, each remark
    in its column, and no heading without a value, however the values were handed in."""
    assert plan_text(EXAMPLE_VALUES, ABOUT, EXAMPLE_REMARKS) == EXAMPLE
    assert plan_text(dict(reversed(list(EXAMPLE_VALUES.items()))), ABOUT, EXAMPLE_REMARKS) == EXAMPLE
    assert plan_text({("Training", "num_runs"): 10000}) == "# Training\nnum_runs = 10000\n"
    assert plan_text({}) == ""
    assert plan_text({}, "  One line.  \n\n   \nAnother.\n") == "# About this plan\nOne line.\nAnother.\n"
    long = {("Prior", "bounds"): "a_name_much_longer_than_the_remark_column.txt"}
    assert plan_text(long, remarks={("Prior", "bounds"): "Bounds file"}) == (
        "# Prior\nbounds = a_name_much_longer_than_the_remark_column.txt # Bounds file\n")


@pytest.mark.parametrize("values, about, key", [
    pytest.param({}, "A first line.\n# Training", None, id="a description line that starts with the mark"),
    pytest.param({("Model and options", "units"): "nm #ms"}, "", "units", id="a value that holds a remark"),
    pytest.param({("Model and options", "units"): "nm\t#ms"}, "", "units", id="a value that holds a tab and the mark"),
    pytest.param({("Model and options", "units"): "#nm"}, "", "units", id="a value that starts with the mark"),
    pytest.param({("Model and options", "units"): "  "}, "", "units", id="blank units"),
    pytest.param({("Prior", "bounds"): ""}, "", "bounds", id="a blank bounds name"),
    pytest.param({("Training", "learning_rate"): float("inf")}, "", "learning_rate", id="an infinity"),
    pytest.param({("Training", "learning_rate"): float("nan")}, "", "learning_rate", id="not a number"),
    pytest.param({("Training", "num_runs"): 10000.0}, "", "num_runs", id="a float for a whole number"),
    pytest.param({("Training", "num_runs"): None}, "", "num_runs", id="nothing for a whole number"),
    pytest.param({("Model and options", "chi_k_pad"): 12}, "", "chi_k_pad", id="a chi setting with no switch"),
    pytest.param({("Training", "bounds"): "a.txt"}, "", "bounds", id="a setting under the wrong heading"),
    pytest.param({("Prior", "bounds"): "../a.txt"}, "", "bounds", id="a bounds name that leaves the folder"),
])
def test_plan_text_refuses_what_could_not_be_read_back(values, about, key):
    """The inverse holds or nothing is written: a file the reader would refuse is a refusal at the saving, and not a
    surprise at the next reading."""
    with pytest.raises(PlanRefusal) as refused:
        plan_text(values, about)
    assert refused.value.field == "plan"
    assert [problem.key for problem in refused.value.problems] == [key]
    assert all(problem.line is None for problem in refused.value.problems)


BOUNDS, UNITS, RUNS = ("Prior", "bounds"), ("Model and options", "units"), ("Training", "num_runs")
SWITCH = ("Model and options", "chi_mode")
LINE_ENDS = " nm\nchi_k_pad = 4\nchi_n_freqs = 4"


@pytest.mark.parametrize("values, remarks, key, sentence", [
    pytest.param({BOUNDS: None}, None, "bounds", f"{_what('bounds')} would not read back as it was given; got None.",
                 id="no bounds file picked"),
    pytest.param({UNITS: None}, None, "units", f"{_what('units')} would not read back as it was given; got None.",
                 id="no units"),
    pytest.param({UNITS: ("nm", "ms")}, None, "units",
                 f"{_what('units')} would not read back as it was given; got ('nm', 'ms').",
                 id="the units as the tuple a stage takes"),
    pytest.param({("About this plan", "num_runs"): 1}, None, None,
                 "The description would not read back as it was given.", id="a pair under the description"),
    pytest.param({SWITCH: True, UNITS: LINE_ENDS}, None, "units",
                 f"{_what('units')} would not read back as it was given; got {LINE_ENDS!r}.",
                 id="a value that holds line ends"),
    pytest.param({RUNS: 1}, {RUNS: "Batches\nhidden_features = 8"}, "hidden_features",
                 f"{_what('hidden_features')} would not read back as it was given; got None.",
                 id="a remark that holds a line end"),
    pytest.param({("training", "num_runs"): 1}, None, "num_runs",
                 f"{_what('num_runs')} would not read back as it was given; got 1.", id="a heading in another case"),
    pytest.param({RUNS: "10000"}, None, "num_runs",
                 f"{_what('num_runs')} would not read back as it was given; got '10000'.", id="a whole number as text"),
])
def test_plan_text_refuses_what_would_read_back_as_something_else(values, remarks, key, sentence):
    """The reader would take each of these without a problem and hand back something else, so reading the text back
    for problems alone would save a plan that does not say what was given. What it reads is compared, and the sentence
    shows the value as it was given."""
    with pytest.raises(PlanRefusal) as refused:
        plan_text(values, "", remarks)
    assert refused.value.message == "This plan cannot be used: 1 problem"
    assert [(problem.line, problem.key, problem.message) for problem in refused.value.problems] == [
        (None, key, sentence)]


def test_what_would_not_read_back_is_said_in_the_order_of_the_file():
    """Several values that would read back as something else are told as the reader tells its problems, top of the
    file first, in whatever order they were handed in."""
    with pytest.raises(PlanRefusal) as refused:
        plan_text({("Calibration", "n_cal"): "5", BOUNDS: None, UNITS: None})
    assert [(problem.key, problem.message) for problem in refused.value.problems] == [
        ("units", f"{_what('units')} would not read back as it was given; got None."),
        ("bounds", f"{_what('bounds')} would not read back as it was given; got None."),
        ("n_cal", f"{_what('n_cal')} would not read back as it was given; got '5'.")]


def test_a_refusal_at_the_saving_says_the_readers_sentence_with_no_line():
    """There is no file yet, so no line to name: the sentence is the one the reader would have given."""
    with pytest.raises(PlanRefusal) as refused:
        plan_text({("Training", "learning_rate"): float("inf"), ("Training", "num_runs"): 0})
    assert refused.value.message == "This plan cannot be used: 2 problems"
    assert [problem.message for problem in refused.value.problems] == [
        _rule(require_at_least, "num_runs", 0, 1), f"{_what('learning_rate')} must be a number; got 'inf'."]


@pytest.mark.parametrize("values, about, sentence", [
    pytest.param({}, "# Training", "A line of the description cannot start with #; got '# Training'.",
                 id="a description line"),
    pytest.param({("Model and options", "units"): "nm #ms"}, "",
                 f"{_what('units')} cannot hold ' #'; got 'nm #ms'.", id="a value"),
    pytest.param({("Model and options", "units"): "nm\t#ms"}, "",
                 f"{_what('units')} cannot hold ' #'; got 'nm\t#ms'.", id="a value with a tab, shown as written"),
    pytest.param({("Model and options", "units"): "#nm"}, "",
                 f"{_what('units')} cannot hold ' #'; got '#nm'.", id="a value that opens with the mark"),
])
def test_what_only_the_saving_can_refuse_says_so_in_its_own_sentence(values, about, sentence):
    """The reader has no sentence for these: the file would read without a problem and mean something else. The
    refusal says "1 problem" with no file, there being none yet."""
    with pytest.raises(PlanRefusal) as refused:
        plan_text(values, about)
    assert refused.value.message == "This plan cannot be used: 1 problem"
    assert [problem.message for problem in refused.value.problems] == [sentence]


def test_what_reads_back_equal_is_written_though_it_was_handed_in_another_form():
    """The comparison is of what the setting means: a whole number handed for a number reads back equal, and a text is
    written stripped, as a description's lines are."""
    assert plan_text({UNITS: "  nm ms  "}) == "# Model and options\nunits = nm ms\n"
    assert plan_text({("Training", "learning_rate"): 1}) == "# Training\nlearning_rate = 1\n"


def test_a_plan_saved_with_a_byte_order_mark_reads_as_any_other(tmp_path):
    """Notepad can write a byte-order mark, and without its removal the first heading would read as a line
    above the first heading. The hash is of the file's own bytes, mark and all, so a plan changed after the
    fill is seen by what is on the disk."""
    plain = read_plan(_plan(tmp_path, EXAMPLE))
    marked_path = tmp_path / "marked.txt"
    marked_path.write_bytes(b"\xef\xbb\xbf" + EXAMPLE.encode("utf-8"))
    marked = read_plan(marked_path)
    assert (marked.values, marked.about) == (plain.values, plain.about)
    assert marked.sha256 == hashlib.sha256(marked_path.read_bytes()).hexdigest() != plain.sha256


@pytest.mark.parametrize("line_end", ["\r\n", "\r"], ids=["Windows", "old Mac"])
def test_a_plan_with_another_editors_line_ends_reads_as_any_other(tmp_path, line_end):
    """Notepad ends a line with two characters. The values and the description are the same, and a problem is
    on the line the editor shows."""
    plain = read_plan(_plan(tmp_path, EXAMPLE))
    other = read_plan(_plan(tmp_path, EXAMPLE.replace("\n", line_end), name="other"))
    assert (other.values, other.about) == (plain.values, plain.about)
    problems = _refused(tmp_path, f"# Training{line_end}{line_end}num_runs = abc{line_end}").problems
    assert problems[0].line == 3


def test_a_file_named_against_the_saving_rule_reads_as_any_other(tmp_path):
    """The rule on a plan's name is the saving's: a file a person named and put in the folder by hand is read,
    its name the file's stem."""
    assert read_plan(_plan(tmp_path, "# Training\nnum_runs = 1\n", name="retrain v2 (old)")).name == "retrain v2 (old)"


def test_a_setting_written_under_the_description_is_description(tmp_path):
    """Every line under the first heading is free text, so a setting put there applies nothing and refuses
    nothing: the picker's count of settings is what shows it."""
    plan = read_plan(_plan(tmp_path, "# About this plan\nnum_runs = 10000\n"))
    assert (plan.values, plan.about) == ({}, "num_runs = 10000")


def test_a_file_with_no_setting_is_a_plan_of_usual_values(tmp_path):
    """No setting is required: an empty file, a file of blank lines, and a description alone are plans, whose
    every setting means the built-in value."""
    for text in ("", "\n  \n\t\n"):
        plan = read_plan(_plan(tmp_path, text))
        assert (plan.values, plan.about) == ({}, "")
    assert read_plan(_plan(tmp_path, "# About this plan\nOnly words.\n")).about == "Only words."


def test_a_heading_is_read_without_regard_to_case_and_may_come_twice(tmp_path):
    """A heading typed in another case is that heading, and its values are filed under its one spelling; a
    heading that comes back goes on where it left off, so a setting repeated across the two is still set twice.
    A line of two marks, or a remark on a heading, is no heading."""
    plan = read_plan(_plan(tmp_path, "#TRAINING\nnum_runs = 1\n\n#  narrowing ROUND\nnum_runs = 2\n\n"
                                     "# Training\nhidden_features = 8\n"))
    assert plan.values == {("Training", "num_runs"): 1, ("Narrowing round", "num_runs"): 2,
                           ("Training", "hidden_features"): 8}
    problems = _refused(tmp_path, "# Training\nnum_runs = 1\n\n# Prior\n\n# TRAINING\nnum_runs = 2\n").problems
    assert [(problem.line, problem.heading) for problem in problems] == [(7, "Training")]
    assert problems[0].message.endswith("is set twice under '# Training' (first on line 2).")
    assert [problem.line for problem in _refused(tmp_path, "## Training\n\n# Training # budget\n").problems] == [1, 3]


def test_a_remark_is_never_read_and_the_mark_needs_white_space_before_it(tmp_path):
    """After a value, white space and the mark end it, whatever follows: a tab lines a remark up as readily as a
    space, and left in a text setting it would be read as the setting. Without the white space the mark is part of
    the value, and a value left out before a remark is left out."""
    plan = read_plan(_plan(tmp_path, "# Prior\nbounds = master.txt   # = 12 # and more\n"
                                     "# Model and options\nunits = nm ms\n"))
    assert plan.values == {("Prior", "bounds"): "master.txt", ("Model and options", "units"): "nm ms"}
    plan = read_plan(_plan(tmp_path, "# Prior\nbounds = master.txt\t# Bounds file\n"
                                     "# Model and options\nunits = nm ms\t# Units\n"
                                     "# Training\nnum_runs = 10000\t# Batches\n"))
    assert plan.values == {("Prior", "bounds"): "master.txt", ("Model and options", "units"): "nm ms",
                           ("Training", "num_runs"): 10000}
    assert read_plan(_plan(tmp_path, "# Training\nnum_runs=10000 # no spaces at the sign\n")).values == {
        ("Training", "num_runs"): 10000}
    problems = _refused(tmp_path, "# Training\nnum_runs = 10000# Batches\n"
                                  "hidden_features =   # Network width\n").problems
    assert [problem.message for problem in problems] == [
        f"line 2: {_what('num_runs')} must be a whole number; got '10000# Batches'.",
        f"line 3: {_what('hidden_features')} must be a whole number; got ''."]


def test_the_rules_across_lines_keep_to_file_order_and_do_not_judge_a_refused_line_twice(tmp_path):
    """A problem found only at the end of the file is listed where its line is. A value already refused is not
    judged again: a switch with a bad word does not also call every chi line lost, and a count refused on its own
    line is not compared with the other. A chi line with a bad value and no switch is two things to correct, and says
    both."""
    problems = _refused(tmp_path, "# Model and options\nchi_k_pad = 12\n\n# Training\nnum_runs = abc\n").problems
    assert [problem.line for problem in problems] == [2, 5]
    problems = _refused(tmp_path, "# Model and options\nchi_mode = maybe\nchi_k_pad = 12\n").problems
    assert [problem.key for problem in problems] == ["chi_mode"]
    problems = _refused(tmp_path, f"{OPTIONS}chi_n_freqs = abc\nchi_k_pad = 4\n").problems
    assert [problem.key for problem in problems] == ["chi_n_freqs"]
    problems = _refused(tmp_path, f"{OPTIONS}chi_n_freqs = 14\nchi_k_pad = abc\n").problems
    assert [problem.key for problem in problems] == ["chi_k_pad"]
    problems = _refused(tmp_path, "# Model and options\nchi_k_pad = abc\n").problems
    assert [(problem.line, problem.key) for problem in problems] == [(2, "chi_k_pad"), (2, "chi_k_pad")]


def test_a_whole_number_of_absurd_length_is_a_problem_and_no_traceback(tmp_path):
    """Python will not read a text of over 4300 digits as a number: the reader says so as any other wrong kind."""
    digits = "9" * 5000
    problems = _refused(tmp_path, f"# Training\nnum_runs = {digits}\n").problems
    assert problems[0].message == f"line 2: {_what('num_runs')} must be a whole number; got '{digits}'."


@pytest.mark.parametrize("name", ["../../Cells/nadrowski/x.txt", "C:/x.txt", "C:x.txt", "/x.txt", "//server/x.txt",
                                  "sub/../../x.txt", "..", "./master.txt", "sub//master.txt", "sub/./master.txt",
                                  "sub/"])
def test_a_bounds_name_that_leaves_the_folder_is_a_problem(tmp_path, name):
    """The window checks that the named file exists under the model's folder, and that check alone would accept a
    name that climbs out of it, or one the computer resolves and the list shows under no such name. Either is refused
    here, with its line."""
    problems = _refused(tmp_path, f"# Prior\nbounds = {name}\n").problems
    assert [(problem.line, problem.key) for problem in problems] == [(2, "bounds")]
    assert problems[0].message == (f"line 2: {_what('bounds')} must name a file in the model's bounds folder; "
                                   f"got '{name}'.")


@pytest.mark.parametrize("name", ["a\\b.txt", "sub\\master.txt", "C:\\x.txt"])
def test_a_backslash_is_told_to_be_a_forward_slash(tmp_path, name):
    """On Windows it is the natural way to write a path, and the file may well be there, so the sentence says what to
    change."""
    problems = _refused(tmp_path, f"# Prior\nbounds = {name}\n").problems
    assert [problem.message for problem in problems] == [
        f"line 2: {_what('bounds')} must name a file in the model's bounds folder; got '{name}'. "
        "Write a sub-folder with a forward slash."]


@pytest.mark.parametrize("name", ["master.txt", "sub/master.txt", "master.v2.txt"])
def test_a_bounds_name_inside_the_folder_is_read_as_written(tmp_path, name):
    assert read_plan(_plan(tmp_path, f"# Prior\nbounds = {name}\n")).values == {("Prior", "bounds"): name}


def test_the_limits_no_rule_holds_are_the_configurations_own():
    """STANDING. The reader may not import ``core/config.py`` or the file manager, both of which reach torch,
    so these are literals, and this is what keeps them honest. The built-ins of the two chi counts are the
    registry's, pinned against the configuration with the registry's other defaults."""
    from core import config
    from core.Helpers import file_manager
    assert plans.CHI_K_MAX == config.CHI_K_MAX
    assert plans.CHI_MIN_CYCLES == config.CHI_MIN_CYCLES
    assert plans.NUMBER_FORM == file_manager.FLOAT_REGEX


def test_the_plan_reader_imports_neither_torch_nor_the_toolkit_nor_the_configuration():
    """STANDING. The window reads a plan on its own thread, so ``core.plans`` costs the standard library and
    the registry. Two pins: every import statement of the file, closed to those, and a fresh interpreter that
    imports it holding none of the three. The subprocess is the real pin: in this process they are long since
    loaded."""
    tree = ast.parse((REPO / "core" / "plans.py").read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, "written out in full, so this list reads what the file reads"
            imported.add(node.module)
    assert "core.refusals" in imported
    assert {name.split(".")[0] for name in imported - {"core.refusals"}} <= set(sys.stdlib_module_names), imported
    probe = ("import sys; import core.plans; "
             "bad = sorted(m for m in sys.modules if m.split('.')[0] in ('torch', 'PySide6') or m == 'core.config'); "
             "sys.exit(repr(bad) if bad else 0)")
    result = subprocess.run([sys.executable, "-c", probe], cwd=str(REPO), capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr


def test_the_plans_folder_stands_beside_the_other_inputs():
    """The one line the configuration gains. It names the folder and is read by nothing in the engine."""
    from core import config
    assert config.PLANS_PATH == config.RESOURCES_ROOT / "Plans"
    source = (REPO / "core" / "config.py").read_text(encoding="utf-8")
    assert len([line for line in source.splitlines() if "PLANS_PATH" in line]) == 1


def _names_a_plan(path) -> bool:
    """Whether a file names the plans folder or imports the reader, however the import is written."""
    source = path.read_text(encoding="utf-8")
    imports = [node for node in ast.walk(ast.parse(source)) if isinstance(node, (ast.Import, ast.ImportFrom))]
    modules = [*(alias.name for node in imports for alias in node.names),
               *(getattr(node, "module", None) or "" for node in imports)]
    return "PLANS_PATH" in source or any(module.split(".")[-1] == "plans" for module in modules)


def test_nothing_outside_the_window_reads_a_plan():
    """STANDING. Filling the forms is the only thing a plan does, so no stage and no command reads one. The reader and
    the folder are the window's: under the engine only their own two files name them."""
    core = REPO / "core"
    files = [path for path in core.rglob("*.py") if path.relative_to(core).parts[0] != "gui"
             and path not in (core / "plans.py", core / "config.py")]
    assert len(files) >= 20
    assert [path.relative_to(REPO).as_posix() for path in files if _names_a_plan(path)] == []


def test_the_plan_key_names_the_file_to_correct_on_screen(tmp_path):
    """A refusal's field chooses the words under it. Here the thing to correct is the file, so the window says
    that, and the command line, which reads no plan, adds nothing."""
    from core.gui import fields
    from core.tool import fields as tool_fields
    refusal = _refused(tmp_path, "# Trainng\n")
    assert fields.fix_sentence(refusal.field) == "Correct the file, then choose the plan again."
    assert tool_fields.fix_sentence(refusal.field) == ""


# ══ the window's side: the list, the fill, where each value came from, the save and the history ══
# Each test below builds the window, so what it needs of the toolkit, the engine and the window is imported inside it;
# and each hands the window a folder of plans of its own (``_the_plans``), so that no test lists or writes the real one.

ABOUT_RETRAIN = "The larger network on the temperature box."
RETRAIN = f"""# About this plan
{ABOUT_RETRAIN}

# Model and options
chi_mode = on                  # Kind of observation: Frequency response
chi_n_freqs = 5                # Driven recordings per observation

# Prior
bounds = master_tier1.txt      # Bounds file
num_iterations = 40            # Sweep rounds

# Training
num_runs = 10000               # Batches
hidden_features = 256          # Network width
learning_rate = 0.0005         # Learning rate

# Narrowing round
num_transforms = 10            # Network depth
"""
#: What RETRAIN fills, by the step and the field; every other row of the three steps takes its usual value.
FILLED = {("model", "chi_mode"): True, ("model", "chi_n_freqs"): 5, ("prior", "bounds"): "master_tier1.txt",
          ("prior", "num_iterations"): 40, ("training", "num_runs"): 10000, ("training", "hidden_features"): 256,
          ("training", "learning_rate"): 0.0005}
#: A value of every row of the three steps that is neither its usual one nor RETRAIN's (the kind of observation, which
#: has two, takes the one RETRAIN does not hold), each as its field takes it.
ELSEWHERE = {("model", "chi_mode"): False, ("model", "chi_n_freqs"): 7, ("model", "chi_k_pad"): 14,
             ("model", "chi_max_cycles"): 25.0, ("model", "reparam_rotate"): False, ("model", "units"): "kHz ms nm pN",
             ("prior", "bounds"): "master.txt", ("prior", "num_iterations"): 45, ("prior", "sweep_batch"): 64,
             ("prior", "max_sets"): 1000, ("prior", "walk_step"): 0.02, ("prior", "stability_units"): 500.0,
             ("prior", "min_cluster_size"): 60, ("prior", "min_samples"): 12, ("training", "num_runs"): 9500,
             ("training", "run_size_cap"): 2048, ("training", "hidden_features"): 192,
             ("training", "num_transforms"): 6, ("training", "learning_rate"): 0.002,
             ("training", "stop_after_epochs"): 30, ("training", "max_num_epochs"): 300,
             ("training", "expected_epochs"): 40, ("training", "checkpoint_every"): 25, ("training", "fisher_m"): 32,
             ("training", "fisher_dz"): 0.2, ("training", "fisher_points"): 6}
#: The places a line of the window may break at, which a sentence read back leaves out.
BREAK = "​"


def _usual() -> dict:
    """The usual value of every row of the three steps but the bounds file, which has none: core/config.py's."""
    from core import config
    return {("model", "chi_mode"): False, ("model", "chi_n_freqs"): config.CHI_N_FREQS,
            ("model", "chi_k_pad"): config.CHI_K_PAD, ("model", "chi_max_cycles"): config.CHI_MAX_CYCLES,
            ("model", "reparam_rotate"): bool(config.REPARAM_ROTATE), ("model", "units"): None,
            ("prior", "num_iterations"): config.PRIOR_SWEEP_ITERATIONS, ("prior", "sweep_batch"): 0,
            ("prior", "max_sets"): config.PRIOR_SWEEP_MAX_SETS, ("prior", "walk_step"): config.PRIOR_SWEEP_STEP,
            ("prior", "stability_units"): float(config.STABILITY_SWEEP_ND_UNITS),
            ("prior", "min_cluster_size"): config.PRIOR_CLUSTER_MIN_SIZE,
            ("prior", "min_samples"): config.PRIOR_CLUSTER_MIN_SAMPLES,
            ("training", "num_runs"): config.TRAINING_NUM_RUNS, ("training", "run_size_cap"): config.TRAINING_RUN_SIZE,
            ("training", "hidden_features"): config.NSF_HIDDEN_FEATURES,
            ("training", "num_transforms"): config.NSF_NUM_TRANSFORMS,
            ("training", "learning_rate"): config.TRAINING_LEARNING_RATE,
            ("training", "stop_after_epochs"): config.TRAINING_STOP_AFTER_EPOCHS, ("training", "max_num_epochs"): None,
            ("training", "expected_epochs"): None, ("training", "checkpoint_every"): config.TRAINING_CHECKPOINT_EVERY,
            ("training", "fisher_m"): config.REPARAM_FISHER_M, ("training", "fisher_dz"): config.REPARAM_FISHER_DZ,
            ("training", "fisher_points"): config.REPARAM_FISHER_POINTS}


def _the_plans(tmp_path, monkeypatch, files=None, model="nadrowski") -> Path:
    """The test's own folder of plans, the one a window built from now on lists and writes, and in it the folder of
    ``model`` holding ``files``, each written byte for byte under its name; that folder, made only when files are
    handed."""
    from core.gui import plans as window_plans
    root = tmp_path / "plans"
    monkeypatch.setattr(window_plans, "PLAN_FOLDER", root)
    folder = root / model
    if files:
        folder.mkdir(parents=True)
        for name, text in files.items():
            _plan(folder, text, name)
    return folder


def _the_window(tmp_path, monkeypatch, files=None, *, clock=None):
    """A whole window in sight, off the screen, on the test's own clock, with the test's own plans: step 1 shown and
    read. (window, screen, the first step, the model's folder of plans, the clock)."""
    from core.gui import design
    from tests._fixtures import build_window, qt_app
    from tests._look_helpers import _settle
    from tests._runs import Clock, at, controller_on
    from tests._window_helpers import shown_off_screen
    folder = _the_plans(tmp_path, monkeypatch, files)
    qt_app()
    clock = clock or Clock(at(8, 12, 4))
    controller_on(tmp_path, clock)
    window = build_window()
    design.set_room((3400, 1380))
    window.setMinimumSize(1, 1)
    shown_off_screen(window, 1300, 820)
    window.go_to("inference", "model")
    _settle()
    screen = window.part("inference")
    return window, screen, screen.config_panel, folder, clock


def _listed(page) -> list:
    """What the list of plans holds, as it reads now."""
    return [page.plan_choice.itemText(at) for at in range(page.plan_choice.count())]


def _opened(page) -> list:
    """The list of plans opened, as a person opens it, and shut: what it then holds."""
    page.plan_choice.showPopup()
    page.plan_choice.hidePopup()
    return _listed(page)


def _choose(page, name) -> None:
    """Open the list and choose ``name`` in it, "No plan" for None."""
    _opened(page)
    found = page.plan_choice.findData(name) if name is not None else 0
    assert found >= 0, (name, _listed(page))
    page.plan_choice.setCurrentIndex(found)


def _pages(screen) -> dict:
    """The three steps a plan fills, by their step."""
    return {page.step: page for page in screen.step_pages()}


def _values(screen) -> dict:
    """Every row a plan fills, by its step and its field, as it reads now."""
    return {(step, key): row.value() for step, page in _pages(screen).items()
            for key, row in page.plan_fields().items()}


def _origins(screen) -> dict:
    """Where every row a plan fills says its value came from, by its step and its field."""
    return {(step, key): page.form.rows[key].origin.text().replace(BREAK, "") for step, page in _pages(screen).items()
            for key in page.plan_fields()}


def _set(screen, values) -> None:
    """Each row of ``values`` set as a person would leave it: a bounds file selected in its list."""
    pages = _pages(screen)
    for (step, key), value in values.items():
        if key == "bounds":
            pages[step].bounds_picker.refresh()
            pages[step].bounds_picker.restore_key(value)
        else:
            pages[step].plan_fields()[key].set_value(value)


def _press_fill(page, monkeypatch, name="retrain", *, answer=None) -> list:
    """Choose ``name`` in the list ("No plan" for None) and press the fill button, answering a question with the answer
    of the words ``answer``, or with nothing, as the guard does; every box asked."""
    from tests._dialogs import all_asked, choose, forget_asked
    _choose(page, name)
    forget_asked()
    if answer is not None:
        choose(monkeypatch, words=answer)
    assert page.fill_button.isEnabled(), page.fill_reason.text()
    page.fill_button.click()
    return all_asked()


def _shown_again(window) -> None:
    """Step 1 shown again from another step, as a person comes back to it: what another step's form changed is drawn."""
    window.go_to("inference", "training")
    window.go_to("inference", "model")


def _said(line) -> str:
    """What a label or a sentence of the window says, with the places a line may break at left out."""
    text = line.plain_text() if hasattr(line, "plain_text") else line.text()
    return text.replace(BREAK, "")


def _facts(page) -> list:
    """What "Before you start" says, as it reads now."""
    return [row.sentence.plain_text() for row in page.foot.facts.rows()]


def _plan_fact(page) -> str:
    """What "Before you start" says of the plan in use, as it reads now."""
    return next(said for said in _facts(page)
                if said.startswith(("Information. No plan", "Information. Plan ", "Information. Settings from")))


#: Under the save's list while step 2's list stands in another model's folder than the model shown.
OTHER_BOUNDS = ("The bounds list holds another model's files, not those of the model shown, so this plan names no "
                "bounds file.")


def _bounds_of_your_own(tmp_path, monkeypatch) -> Path:
    """A folder of bounds files of the test's own, Nadrowski's copied, bound where both pages read it: so that a test
    can remove a file, or give two models a file of one name, and no real file changes. Bound before the window is
    built."""
    import shutil
    from core.config import BOUNDS_PATH
    from core.gui.panels.inference import config_tab, prior_tab
    bounds = tmp_path / BOUNDS_PATH.name
    shutil.copytree(BOUNDS_PATH / "nadrowski", bounds / "nadrowski")
    monkeypatch.setattr(config_tab, "BOUNDS_PATH", bounds)
    monkeypatch.setattr(prior_tab, "BOUNDS_PATH", bounds)
    return bounds


def _a_plan_of_hopf(folder) -> Path:
    """Beside the test's own folder of plans for Nadrowski, Hopf's, holding one plan that names its bounds file."""
    hopf = folder.parent / "hopf"
    hopf.mkdir()
    return _plan(hopf, "# Prior\nbounds = cell.txt\n", "hopf_bounds")


def test_choosing_in_the_list_changes_nothing(tmp_path, monkeypatch, store):
    """Choosing a plan in the list reads it and says what it holds; it changes no form, no origin, no plan line, nothing
    in use, nothing remembered and no step's state: only the fill button fills."""
    from PySide6.QtCore import QSettings
    window, screen, page, _folder, _clock = _the_window(tmp_path, monkeypatch, {"retrain": RETRAIN})
    written = []
    kept = QSettings.setValue
    monkeypatch.setattr(QSettings, "setValue", lambda qs, key, value: (written.append(key), kept(qs, key, value))[1])

    def held():
        return (_values(screen), _origins(screen), screen.step_states(), screen.plan_source,
                [_said(one.plan_line) for one in screen.step_pages()])

    before = held()
    _choose(page, "retrain")
    assert page.plan_choice.currentText() == "retrain"
    assert _said(page.plan_about) == f"{ABOUT_RETRAIN} 8 settings in 4 steps."
    assert held() == before
    assert _said(page.plan_in_use) == "In use: no plan."
    assert page.form.rows["plan"].line.words == "Changed, not applied. In use: no plan."
    assert written == [], written


def test_the_folder_is_read_again_each_time_the_list_opens(tmp_path, monkeypatch, store):
    """The list holds "No plan", then the model's plans by name in the order of the alphabet, read as it opens: a plan
    written since comes in, one taken away goes, the plan shown stays while it is there. Open, it is at least as wide
    as its longest name, which the box shows cut short in its middle, whole in its tip."""
    long_name = "retrain_on_the_temperature_box_at_the_larger_network_second_try"
    window, screen, page, folder, _clock = _the_window(tmp_path, monkeypatch, {"retrain": RETRAIN})
    assert _opened(page) == ["No plan", "retrain"]
    _plan(folder, RETRAIN, "Another")
    _plan(folder, RETRAIN, long_name)
    (folder / "Zeta.TXT").write_bytes(RETRAIN.encode("utf-8"))
    assert _listed(page) == ["No plan", "retrain"], "the list was read before it opened"
    assert _opened(page) == ["No plan", "Another", "retrain", long_name, "Zeta"]
    (folder / "Zeta.TXT").unlink()
    page.plan_choice.setCurrentIndex(page.plan_choice.findData(long_name))
    page.plan_choice.showPopup()
    try:
        wide = page.plan_choice.fontMetrics().horizontalAdvance(long_name)
        assert page.plan_choice.view().width() >= wide, (page.plan_choice.view().width(), wide)
    finally:
        page.plan_choice.hidePopup()
    assert page.plan_choice.toolTip() == long_name and page.plan_choice.shown_text() != long_name
    (folder / "Another.txt").unlink()
    assert _opened(page) == ["No plan", "retrain", long_name] and page.plan_choice.currentText() == long_name
    (folder / f"{long_name}.txt").unlink()
    assert _opened(page) == ["No plan", "retrain"] and page.plan_choice.currentText() == "No plan"


@pytest.mark.parametrize("factor", [1.0, 1.5])
def test_a_long_plan_name_keeps_the_first_step_inside_the_least_window(factor, tmp_path, monkeypatch, store):
    """A plan is named by hand, at any length. Shown in the list at the window's least size, with the fold open and the
    plan's table open, its name is cut short in its middle in the box and whole in its tip, and the form stays inside
    its viewport: the body never scrolls sideways. The program's font, set for the measure, is put back."""
    from PySide6.QtGui import QFont
    from core.gui import design, fonts
    from tests._fixtures import build_window, qt_app
    from tests._look_helpers import _settle, _the_look, text_size
    from tests._runs import Clock, at, controller_on
    from tests._window_helpers import shown_off_screen
    name = "a_plan_named_at_length_for_the_retrain_measured_on_the_second_of_november_2026_by_hand"
    _the_plans(tmp_path, monkeypatch, {name: RETRAIN})
    app = qt_app()
    kept = QFont(app.font())
    try:
        with text_size(factor), _the_look(False):
            fonts.load_app_font(app)
            controller_on(tmp_path, Clock(at(8, 12, 4)))
            window = build_window()
            design.set_room((3400, 1380))
            window.setMinimumSize(1, 1)
            page = window.part("inference").config_panel
            shown_off_screen(window, design.scaled(900), design.scaled(600))
            window.go_to("inference", "model")
            _settle()
            _choose(page, name)
            page.holds_button.click()
            page.form.fold.set_open(True)
            _settle()
            port = page.scroller.viewport()
            assert page.scroller.body.minimumSizeHint().width() <= port.width(), (
                page.scroller.body.minimumSizeHint().width(), port.width())
            for part in (page.plan_choice, page.holds_table, page.plan_about, page.fill_button):
                assert part.mapTo(port, part.rect().topRight()).x() < port.width(), type(part).__name__
            assert page.plan_choice.shown_text() != name and "…" in page.plan_choice.shown_text()
            assert page.plan_choice.toolTip() == name
            window.hide()
    finally:
        app.setFont(kept)


def test_a_hand_named_file_is_listed_and_a_plan_with_problems_fills_nothing(tmp_path, monkeypatch, store):
    """Every .txt of the model's folder is listed, a hand-named one too. One with problems says so with each problem on
    its line, greys the fill button with the reason beside it, and fills nothing, never a part of a plan, at a press
    that reaches the page all the same. "Open the plans folder" is greyed until the folder exists, then opens it."""
    from core.gui import plans as window_plans
    opened = []
    monkeypatch.setattr(window_plans, "open_folder", opened.append)
    window, screen, page, folder, _clock = _the_window(tmp_path, monkeypatch)
    assert _opened(page) == ["No plan"] and not page.open_button.isEnabled()
    broken = "# Training\nnum_runs = 9000\nnum_run = 5\n\n# Trainng\nhidden_features = 64\n"
    folder.mkdir(parents=True)
    _plan(folder, broken, "retrain v2")
    (folder / "notes.md").write_text("not a plan", encoding="utf-8")
    assert _opened(page) == ["No plan", "retrain v2"] and page.open_button.isEnabled()
    before = _values(screen)
    _choose(page, "retrain v2")
    assert page.plan_trouble.sentence.plain_text() == "This plan cannot be used: 2 problems in retrain v2.txt"
    shown = [_said(line) for line in page.plan_problems if not line.isHidden()]
    assert shown == ["line 3: 'num_run' is not a setting PRISM has. The nearest is num_runs.",
                     f"line 5: '# Trainng' is not a heading a plan has. {THE_HEADINGS}"]
    assert not page.fill_button.isEnabled()
    assert page.fill_reason.text() == "Correct the file, then choose the plan again."
    assert page.holds_button.isHidden() and page.plan_about.isHidden()
    page._fill_pressed()
    assert _values(screen) == before and screen.plan_source is None
    page.open_button.click()
    assert opened == [folder]


def test_a_plan_with_many_problems_says_eight_and_counts_the_rest_each_breaking_anywhere(tmp_path, monkeypatch, store):
    """A file of notes is a plan with a problem on every line, each echoing its line whole: eight are said, each cut at
    its length and breaking anywhere, so that a line of one long word does not widen the page, and the rest are
    counted."""
    notes = "\n".join(["# Training"] + [f"note {at}" for at in range(11)] + ["y" * 600])
    window, screen, page, _folder, _clock = _the_window(tmp_path, monkeypatch, {"notes": notes})
    _choose(page, "notes")
    assert page.plan_trouble.sentence.plain_text() == "This plan cannot be used: 12 problems in notes.txt"
    shown = [_said(line) for line in page.plan_problems if not line.isHidden()]
    assert len(shown) == 9 and shown[-1] == "And 4 more."
    assert shown[0] == "line 2: 'note 0' is not a setting line. Write name = value under a heading."
    long = page._shown_plan.problems[-1].message
    assert len(long) > 300
    page._shown_plan = type(page._shown_plan)("", (page._shown_plan.problems[-1],))
    page.redraw()
    said = page.plan_problems[0].text()
    assert said.replace(BREAK, "") == long[:299] + "…" and BREAK in said
    port = page.scroller.viewport()
    assert page.scroller.body.minimumSizeHint().width() <= port.width()


def test_the_picker_opens_on_no_plan_and_says_what_was_last_used(tmp_path, monkeypatch, store):
    """At launch the list shows "No plan", never what was chosen before. Beside it while it shows "No plan", the plan
    last used with the model shown and when, or that its file is gone; nothing when the last step applied, or the last
    run started, had no plan, nor for another model's plan. Hopf's plan last used has a name of its own, so that a hint
    read under another model's key would show."""
    from datetime import datetime
    from core.gui import settings
    from core.gui.remembered import LAST_USED_AT_KEY, LAST_USED_KEY
    from tests._fixtures import build_window
    from tests._look_helpers import _settle
    from tests._window_helpers import shown_off_screen
    used = datetime(2026, 10, 5, 14, 2).astimezone()
    _the_plans(tmp_path, monkeypatch, {"retrain": RETRAIN})

    def opened(remembered):
        qs = settings.settings()
        for key, value in remembered.items():
            qs.setValue(key, value)
        qs.sync()
        window = build_window()
        window.setMinimumSize(1, 1)
        shown_off_screen(window, 1300, 820)
        window.go_to("inference", "model")
        _settle()
        page = window.part("inference").config_panel
        return page, page.form.rows["plan"].origin.text().replace(BREAK, "")

    page, hint = opened({LAST_USED_KEY.format("nadrowski"): "retrain",
                         LAST_USED_AT_KEY.format("nadrowski"): used.isoformat()})
    assert (page.plan_choice.currentText(), hint) == ("No plan", "Last used: retrain, 5 Oct 2026, 14:02")
    assert _said(page.plan_line) == "No plan"
    _choose(page, "retrain")
    assert page.form.rows["plan"].origin.text() == "", "the hint stood beside a plan chosen"
    assert opened({LAST_USED_KEY.format("nadrowski"): "gone_plan"})[1] == (
        "Last used: gone_plan (the file is no longer there)")
    assert opened({LAST_USED_KEY.format("nadrowski"): "retrain", LAST_USED_AT_KEY.format("nadrowski"): "?"})[1] == (
        "Last used: retrain")
    assert opened({LAST_USED_KEY.format("nadrowski"): "", LAST_USED_AT_KEY.format("nadrowski"): ""})[1] == ""
    assert opened({LAST_USED_KEY.format("hopf"): "zz_hopf"})[1] == ""
    plans_of = tmp_path / "plans" / "nadrowski"
    (plans_of / "retrain.txt").rename(plans_of / "Retrain.txt")
    assert opened({LAST_USED_KEY.format("nadrowski"): "retrain", LAST_USED_AT_KEY.format("nadrowski"): used.isoformat()
                   })[1] == "Last used: Retrain, 5 Oct 2026, 14:02", "a plan renamed in another case was said gone"
    page.model_combo.setCurrentText("HOPF")
    page.read_facts()
    assert page.form.rows["plan"].origin.text().replace(BREAK, "") == "Last used: zz_hopf (the file is no longer there)"


def test_one_press_fills_steps_one_to_three(tmp_path, monkeypatch, store):
    """Every row of steps 1 to 3 that a plan fills takes the plan's value, else its usual one, from wherever each stood:
    off its usual value and off the plan's, or put back from last time. The bounds file is selected in the model
    applied, its limits no longer typed. Nothing is said to be kept any longer, and the plan line under each heading
    names the plan, and once step 1 is applied, no change from it."""
    from core.gui import plans as window_plans
    window, screen, page, _folder, _clock = _the_window(tmp_path, monkeypatch, {"retrain": RETRAIN})
    page.foot.main_button.click()
    _set(screen, ELSEWHERE)
    prior, training = screen.prior_panel, screen.posterior_panel
    prior.typed_tick.setChecked(True)
    training.form.rows["num_runs"].kept = True
    assert _values(screen) == {**ELSEWHERE, ("prior", "bounds"): window_plans.TYPED}
    _press_fill(page, monkeypatch, answer="Fill the forms")
    usual = _usual()
    wanted = {pair: FILLED.get(pair, usual.get(pair)) for pair in ELSEWHERE}
    assert _values(screen) == wanted
    assert [pair for pair in ELSEWHERE if wanted[pair] == ELSEWHERE[pair]] == [], "a row was not moved by the fill"
    assert not prior.typed_tick.isChecked() and prior.bounds_picker.key() == "master_tier1.txt"
    assert not any(row.kept for one in screen.step_pages() for row in one.plan_fields().values())
    assert (screen.plan_source.kind, screen.plan_source.name) == ("plan", "retrain")
    assert [_said(one.plan_line) for one in screen.step_pages()] == [
        "Plan: retrain", "Plan: retrain, 1 change", "Plan: retrain, 1 change"], "step 1 holds what it held"
    page.foot.main_button.click()
    assert [_said(one.plan_line) for one in screen.step_pages()] == ["Plan: retrain"] * 3


def test_a_fill_starts_nothing_applies_nothing_and_replaces_no_session(tmp_path, monkeypatch, store):
    """A fill asks nothing more than whether to replace what was typed, starts no run, applies no step and replaces no
    session: what the steps hold are the very objects they were, and the records picked in the lists stay picked. The
    forms the fill changed now differ from what step 1 and step 2 hold, and the rows that compare them say so: "Changed,
    not applied" under the kind of observation and the bounds file, in the rail and in the chain Home reads."""
    from types import SimpleNamespace
    from core.config import BOUNDS_PATH
    from core.gui import chain
    from core.gui.run.controller import RunController
    from core.gui.screens.inference_screen import InferenceScreen
    from core.gui.session import SbiSession
    window, screen, page, _folder, _clock = _the_window(tmp_path, monkeypatch, {"retrain": RETRAIN})
    page.foot.main_button.click()
    cfg = SimpleNamespace(model="NADROWSKI", sources={"bounds": str(BOUNDS_PATH / "nadrowski" / "master.txt")},
                          params_dict={}, rescale_params={}, force_params_dict={}, observation_mode="spontaneous",
                          check_unit_consistency=lambda: [])
    prior = SimpleNamespace(kind="prior", id="p_old", name="old_prior", manifest=None)
    posterior = SimpleNamespace(kind="posterior", id="post_old", name="retrain", manifest=None)
    observation = SimpleNamespace(kind="observation", id="obs_old", name="cell_a")
    screen.session = session = SbiSession(draft=screen.session.draft, cfg=cfg, inf_prior=prior, posterior=posterior,
                                          observation=observation)
    screen.prior_panel.bounds_picker.restore_key("master.txt")
    screen.refresh_gates()
    held = (session.draft, session.cfg, session.inf_prior, session.posterior, session.observation)
    picked = (screen.prior_panel.prior_picker.key(), screen.posterior_panel.post_picker.key())
    assert screen.pending() == frozenset()
    started, drafted = [], []
    monkeypatch.setattr(RunController, "dispatch", lambda held_, *args, **kwargs: started.append(args))
    monkeypatch.setattr(InferenceScreen, "new_draft", lambda screen_, draft: drafted.append(draft))
    asked = _press_fill(page, monkeypatch, answer="Fill the forms")
    assert [one.title for one in asked] == ["Fill the forms from plan retrain?"] and started == drafted == []
    assert screen.session is session and (session.draft, session.cfg, session.inf_prior, session.posterior,
                                          session.observation) == held
    assert all(now is then for now, then in zip((session.draft, session.cfg, session.inf_prior, session.posterior,
                                                  session.observation), held))
    assert (screen.prior_panel.prior_picker.key(), screen.posterior_panel.post_picker.key()) == picked
    assert screen.pending() == {"model", "prior"}
    states = screen.step_states()
    assert states[0].pending and states[1].pending
    for step in ("model", "prior"):
        assert "not applied" in window.shell.rail.row(step).accessibleName(), step
    assert chain.step_in_hand(states) not in ("model", "prior"), "Home's next step went back to a step that holds"
    assert page.form.rows["chi_mode"].line.words == "Changed, not applied. In use: Set by the bounds file."
    assert screen.prior_panel.form.rows["bounds"].line.words == "Changed, not applied. In use: master.txt."
    training = _facts(screen.posterior_panel)[-1]
    assert training.startswith("Information. Plan retrain, with 2 changes: ") and training.endswith(
        "Kind of observation set by the bounds file (plan frequency response); Bounds file master.txt (plan "
        "master_tier1.txt)."), training


def test_a_fill_that_would_replace_a_typed_value_asks_first(tmp_path, monkeypatch, store):
    """A fill that would replace a value a person typed or that was put back from last time asks first: its title, its
    sentence counting what the plan fills, and each value it would replace in full, "Keep what is there" preselected,
    the answer of Escape. That answer leaves every field as it was, and fills nothing; the other fills. A value already
    at its usual one, or at the plan's, is not asked about."""
    from tests._dialogs import choose
    window, screen, page, _folder, _clock = _the_window(tmp_path, monkeypatch, {"retrain": RETRAIN})
    page.foot.main_button.click()
    training, prior = screen.posterior_panel, screen.prior_panel
    _set(screen, {("training", "num_runs"): 9500, ("prior", "bounds"): "master.txt", ("model", "units"): "kHz ms",
                  ("training", "hidden_features"): 256, ("training", "fisher_m"): 48})
    training.form.rows["run_size_cap"].set_value(512)
    training.form.rows["run_size_cap"].kept = True
    before = _values(screen)
    asked = _press_fill(page, monkeypatch)
    assert len(asked) == 1, asked
    question = asked[0]
    assert (question.title, question.sentence) == (
        "Fill the forms from plan retrain?",
        "Filling from plan retrain sets every setting a plan can hold: 7 from the plan, the rest to their usual "
        "values.")
    assert question.items == ("Units: kHz ms becomes the model's units file", "Bounds file: master.txt becomes "
                              "master_tier1.txt", "Batches: 9,500 becomes 10,000", "Simulations per batch: 512 becomes "
                              "automatic")
    assert (question.answers, question.default, question.safe, question.loses_work) == (
        ("Fill the forms", "Keep what is there"), "Keep what is there", "Keep what is there", False)
    assert _values(screen) == before and screen.plan_source is None
    choose(monkeypatch, other=True)
    page.fill_button.click()
    assert _values(screen)[("training", "num_runs")] == 10000 and screen.plan_source.name == "retrain"
    _plan(_folder, "# Training\nnum_runs = 7000\n", "fewer")
    assert _press_fill(page, monkeypatch, "fewer") == [], "a value at the plan's in use was asked about"
    assert _values(screen)[("training", "num_runs")] == 7000


def test_the_fill_holds_the_plans_hash_as_read_and_the_moment_of_the_press(tmp_path, monkeypatch, store):
    """The plan in use after "Fill the forms from this plan" holds the hash of its file as the press read it and the
    moment of the press on the window's clock."""
    window, screen, page, _folder, clock = _the_window(tmp_path, monkeypatch, {"retrain": RETRAIN})
    clock.advance(90)
    _press_fill(page, monkeypatch, answer="Fill the forms")
    source = screen.plan_source
    assert (source.name, source.sha256, source.at) == (
        "retrain", hashlib.sha256(RETRAIN.encode("utf-8")).hexdigest(), clock.now), source


def test_keep_what_is_there_keeps_looking_for_the_gone_plan(tmp_path, monkeypatch, store):
    """The plan in use's file removed, a value typed, the usual values pressed and answered "Keep what is there":
    nothing is filled and the plan is still looked for, so it is shown again when its file comes back."""
    window, screen, page, folder, _clock = _the_window(tmp_path, monkeypatch, {"retrain": RETRAIN})
    _press_fill(page, monkeypatch, answer="Fill the forms")
    (folder / "retrain.txt").unlink()
    page.read_facts()
    screen.posterior_panel.form.rows["num_runs"].set_value(9500)
    asked = _press_fill(page, monkeypatch, None, answer="Keep what is there")
    assert [one.title for one in asked] == ["Return the forms to the usual values?"], asked
    assert screen.plan_source.name == "retrain"
    _plan(folder, RETRAIN, "retrain")
    page.read_facts()
    assert page.plan_choice.currentData() == "retrain", page.plan_choice.currentText()


def test_return_to_the_usual_values_resets_steps_one_to_three(tmp_path, monkeypatch, store):
    """With "No plan" shown the button returns every row of steps 1 to 3 to its usual value, asking first with its own
    words when it would replace what was typed; the bounds file is left as it is, and no plan is in use after it."""
    window, screen, page, _folder, _clock = _the_window(tmp_path, monkeypatch, {"retrain": RETRAIN})
    page.foot.main_button.click()
    _press_fill(page, monkeypatch, answer="Fill the forms")
    screen.posterior_panel.form.rows["num_runs"].set_value(9500)
    assert page.fill_button.text() == "Fill the forms from this plan"
    _choose(page, None)
    assert page.fill_button.text() == "Return the forms to the usual values"
    asked = _press_fill(page, monkeypatch, None, answer="Return to the usual values")
    assert [(one.title, one.sentence, one.items, one.answers, one.default) for one in asked] == [(
        "Return the forms to the usual values?", "This sets every setting a plan can hold to its usual value.",
        ("Batches: 9,500 becomes 5,000",), ("Return to the usual values", "Keep what is there"),
        "Keep what is there")]
    usual = _usual()
    assert _values(screen) == {**usual, ("prior", "bounds"): "master_tier1.txt"}
    assert screen.plan_source is None and [_said(one.plan_line) for one in screen.step_pages()] == ["No plan"] * 3
    assert _said(page.plan_in_use) == "In use: no plan."
    assert _press_fill(page, monkeypatch, None) == [], "the forms at their usual values asked again"


def test_the_later_headings_are_shown_not_applied_yet_and_counted(tmp_path, monkeypatch, store):
    """What the plan holds is shown, folded, as a table of four columns, a row a setting in the file's order: its step,
    its label with its name in the file, its value in the plan and its usual value; a setting of a step that is not
    filled yet says "not applied yet". The count beside the plan's description counts every setting and every step;
    the fill's question counts the settings it fills, and the facts and the fill leave the later ones alone."""
    from core import config
    window, screen, page, _folder, _clock = _the_window(tmp_path, monkeypatch, {"retrain": RETRAIN})
    _choose(page, "retrain")
    assert page.holds_button.text() == "Show what it holds" and page.holds_table.isHidden()
    page.holds_button.click()
    assert not page.holds_table.isHidden()
    from PySide6.QtCore import Qt
    model = page.holds_table.view.model()
    header = [model.headerData(at, Qt.Horizontal) for at in range(4)]
    rows = [[model.index(row, at).data() for at in range(4)] for row in range(model.rowCount())]
    assert header == ["Step", "Setting", "In the plan", "Usual"]
    assert rows == [
        ["Model and options", "Kind of observation (chi_mode)", "frequency response", "set by the bounds file"],
        ["Model and options", "Driven recordings per observation (chi_n_freqs)", "5", f"{config.CHI_N_FREQS}"],
        ["Prior", "Bounds file (bounds)", "master_tier1.txt", "none"],
        ["Prior", "Sweep rounds (num_iterations)", "40", f"{config.PRIOR_SWEEP_ITERATIONS}"],
        ["Training", "Batches (num_runs)", "10,000", f"{config.TRAINING_NUM_RUNS:,}"],
        ["Training", "Network width (hidden_features)", "256", f"{config.NSF_HIDDEN_FEATURES}"],
        ["Training", "Learning rate (learning_rate)", "0.0005", f"{config.TRAINING_LEARNING_RATE}"],
        ["Narrowing round", "Network depth (num_transforms)", "10", "not applied yet"]], rows
    assert _said(page.plan_about) == f"{ABOUT_RETRAIN} 8 settings in 4 steps."
    asked = _press_fill(page, monkeypatch, answer="Fill the forms")
    assert asked == [] or asked[0].sentence.startswith("Filling from plan retrain sets every setting a plan can hold: "
                                                       "7 from the plan")
    assert screen.posterior_panel.form.rows["num_transforms"].value() == config.NSF_NUM_TRANSFORMS
    assert ("Narrowing round", "num_transforms") not in screen.plan_source.values


def test_a_plans_bounds_file_waits_until_its_model_is_applied(tmp_path, monkeypatch, store):
    """Filled while the model on step 1's form is not the one applied, a plan's bounds file is not selected in the list
    of another model's folder: it waits, the limits typed by hand no longer read, and when step 1 applies the model the
    plan was filled for, the file is selected by its name in that model's folder. A file chosen in the list takes its
    place."""
    from tests._dialogs import choose
    window, screen, page, folder, _clock = _the_window(tmp_path, monkeypatch, {"retrain": RETRAIN})
    prior = screen.prior_panel
    prior.typed_tick.setChecked(True)
    before = prior.bounds_picker.key()
    _press_fill(page, monkeypatch, answer="Fill the forms")
    assert screen.session.draft is None and prior.bounds_picker.key() == before
    assert not prior.typed_tick.isChecked()
    assert _said(prior.waiting_line) == "master_tier1.txt is selected when step 1, Model and options, is applied."
    assert not prior.waiting_line.isHidden() and _origins(screen)[("prior", "bounds")] == "from plan retrain"
    page.foot.main_button.click()
    assert prior.bounds_picker.key() == "master_tier1.txt" and prior.gone_line.isHidden()
    assert prior.waiting_line.isHidden() and _origins(screen)[("prior", "bounds")] == "from plan retrain"
    assert not prior.form.rows["bounds"].kept
    _a_plan_of_hopf(folder)
    page.model_combo.setCurrentText("HOPF")
    _press_fill(page, monkeypatch, "hopf_bounds", answer="Fill the forms")
    assert _said(prior.waiting_line) == "cell.txt is selected when step 1, Model and options, is applied."
    prior.bounds_picker.restore_key("master_spont.txt")
    assert prior.waiting_line.isHidden(), "a file chosen in the list left a plan's file waiting"
    _press_fill(page, monkeypatch, "hopf_bounds", answer="Fill the forms")
    assert not prior.waiting_line.isHidden()
    choose(monkeypatch, words="Apply and start a new session")
    page.foot.main_button.click()
    assert prior.bounds_picker.key() == "cell.txt" and prior.gone_line.isHidden() and prior.waiting_line.isHidden()


def test_step_two_names_the_file_its_build_reads_while_a_plans_file_waits(tmp_path, monkeypatch, store):
    """While a plan's bounds file waits for the model it was filled for, step 2's "Before you start" and its plan line
    name the file the build reads, the one the list selects in the folder of the model applied, with that folder, as a
    departure from the plan's; the row still says its value came from the plan; and the press builds on the file
    selected."""
    from core import orchestrator
    from core.gui.run.controller import controller
    from tests._fixtures import qt_app
    from tests._runs import StandIn, turn_until
    app = qt_app()
    window, screen, page, folder, _clock = _the_window(tmp_path, monkeypatch, {"retrain": RETRAIN})
    _a_plan_of_hopf(folder)
    page.foot.main_button.click()
    prior = screen.prior_panel
    _set(screen, {("prior", "bounds"): "master.txt"})
    page.model_combo.setCurrentText("HOPF")
    _press_fill(page, monkeypatch, "hopf_bounds", answer="Fill the forms")
    assert (prior._pending, prior.bounds_picker.key()) == ("cell.txt", "master.txt")
    window.go_to("inference", "prior")
    prior.show_view("build")
    prior.read_facts()
    assert _plan_fact(prior) == (
        "Information. Plan hopf_bounds, with 1 change: Bounds file master.txt in nadrowski (plan cell.txt).")
    assert _said(prior.plan_line) == "Plan: hopf_bounds, 1 change"
    assert _origins(screen)[("prior", "bounds")] == "from plan hopf_bounds"
    stand_in = StandIn(word="Prior build")
    monkeypatch.setattr(orchestrator, "build_prior", stand_in)
    prior.names.name.setText("a_prior")
    prior._main_clicked()
    assert controller().is_live() and stand_in.started.wait(5), prior.foot.reason.text()
    built = stand_in.calls[-1][0][0]
    stand_in.fail(RuntimeError("ended by the test"))
    turn_until(app, lambda: not controller().is_live(), 10.0)
    assert Path(built.sources["bounds"]).name == "master.txt", built.sources


def test_a_waiting_file_is_a_departure_though_the_file_selected_has_its_name(tmp_path, monkeypatch, store):
    """Two models' folders each hold a bounds file of one name. With one model applied and that file selected, a plan of
    the other model naming its own file of that name is filled: the file the build reads is the applied model's, named
    with its folder, so "Before you start" and the plan line count it as a change, and the press builds on it."""
    import shutil
    from core import orchestrator
    from core.gui.run.controller import controller
    from tests._fixtures import qt_app
    from tests._runs import StandIn, turn_until
    app = qt_app()
    bounds = _bounds_of_your_own(tmp_path, monkeypatch)
    (bounds / "hopf").mkdir()
    shutil.copy(bounds / "nadrowski" / "master.txt", bounds / "hopf" / "master.txt")
    window, screen, page, folder, _clock = _the_window(tmp_path, monkeypatch, {"retrain": RETRAIN})
    (folder.parent / "hopf").mkdir()
    _plan(folder.parent / "hopf", "# Prior\nbounds = master.txt\n", "same_name")
    page.foot.main_button.click()
    prior = screen.prior_panel
    _set(screen, {("prior", "bounds"): "master.txt"})
    page.model_combo.setCurrentText("HOPF")
    _press_fill(page, monkeypatch, "same_name", answer="Fill the forms")
    assert (prior.waiting_file(), prior.bounds_picker.key()) == ("master.txt", "master.txt")
    window.go_to("inference", "prior")
    prior.show_view("build")
    prior.read_facts()
    assert _plan_fact(prior) == (
        "Information. Plan same_name, with 1 change: Bounds file master.txt in nadrowski (plan master.txt).")
    assert _said(prior.plan_line) == "Plan: same_name, 1 change"
    assert _origins(screen)[("prior", "bounds")] == "from plan same_name"
    stand_in = StandIn(word="Prior build")
    monkeypatch.setattr(orchestrator, "build_prior", stand_in)
    prior.names.name.setText("a_prior")
    prior._main_clicked()
    assert controller().is_live() and stand_in.started.wait(5), prior.foot.reason.text()
    built = stand_in.calls[-1][0][0]
    stand_in.fail(RuntimeError("ended by the test"))
    turn_until(app, lambda: not controller().is_live(), 10.0)
    assert Path(built.sources["bounds"]).parent.name == "nadrowski", built.sources


def test_limits_typed_take_the_place_of_a_waiting_file(tmp_path, monkeypatch, store):
    """Limits typed by hand take a waiting plan's file's place as a file chosen in the list does: the file waits no
    longer, and step 1 applied leaves the list as it was, the limits typed still read."""
    window, screen, page, _folder, _clock = _the_window(tmp_path, monkeypatch, {"retrain": RETRAIN})
    prior = screen.prior_panel
    _set(screen, {("prior", "bounds"): "master.txt"})
    _press_fill(page, monkeypatch, answer="Fill the forms")
    assert prior._pending == "master_tier1.txt" and prior.bounds_picker.key() == "master.txt", prior._pending
    prior.show_view("build")
    prior.typed_tick.setChecked(True)
    assert prior._pending is None and prior.waiting_line.isHidden(), "limits typed left a plan's file waiting"
    page.foot.main_button.click()
    assert (prior.bounds_picker.key(), prior.typed_tick.isChecked()) == ("master.txt", True), (
        prior.bounds_picker.key(), prior.typed_tick.isChecked())


def test_a_waiting_file_is_found_in_the_folder_of_the_model_applied(tmp_path, monkeypatch, store):
    """A plan of another model than the one applied, filled with its model on the form, waits: its file is not looked
    for in the folder the list shows. Step 1 applying that model selects the file in that model's own folder."""
    from tests._dialogs import choose
    window, screen, page, folder, _clock = _the_window(tmp_path, monkeypatch, {"retrain": RETRAIN})
    _a_plan_of_hopf(folder)
    page.foot.main_button.click()
    prior = screen.prior_panel
    page.model_combo.setCurrentText("HOPF")
    _press_fill(page, monkeypatch, "hopf_bounds", answer="Fill the forms")
    waited = (prior._pending, prior.waiting_line.isHidden(), prior.gone_line.isHidden())
    choose(monkeypatch, words="Apply and start a new session")
    page.foot.main_button.click()
    applied = (prior.bounds_picker.key(), prior.gone_line.isHidden(), _said(prior.gone_line))
    assert waited == ("cell.txt", False, True), waited
    assert applied[:2] == ("cell.txt", True), applied


def test_a_waiting_file_goes_with_the_model_it_was_filled_for(tmp_path, monkeypatch, store):
    """Another model chosen on step 1's form lets the plan in use go, and its waiting bounds file with it: the line
    under the list goes, and the model applied next has its own file back, kept from last time, with nothing said of the
    plan's."""
    from core.gui import settings
    from core.gui.remembered import BOUNDS_KEY, BOUNDS_MODEL_KEY
    qs = settings.settings()
    qs.setValue(BOUNDS_KEY, "cell.txt")
    qs.setValue(BOUNDS_MODEL_KEY, "HOPF")
    qs.sync()
    window, screen, page, _folder, _clock = _the_window(tmp_path, monkeypatch, {"retrain": RETRAIN})
    prior = screen.prior_panel
    _press_fill(page, monkeypatch, answer="Fill the forms")
    before = (prior._pending, _said(prior.waiting_line), prior.waiting_line.isHidden())
    page.model_combo.setCurrentText("HOPF")
    chosen = (prior._pending, prior.waiting_line.isHidden(), screen.plan_source, _origins(screen)[("prior", "bounds")])
    page.foot.main_button.click()
    applied = (prior.bounds_picker.key(), prior.gone_line.isHidden(), prior.waiting_line.isHidden(),
               _origins(screen)[("prior", "bounds")])
    assert before[0] == "master_tier1.txt" and not before[2], before
    assert chosen == (None, True, None, ""), chosen
    assert applied == ("cell.txt", True, True, "kept from last time"), applied


def test_a_plans_bounds_file_gone_by_the_apply_is_said_in_the_plans_words(tmp_path, monkeypatch, store):
    """A plan's bounds file that is no longer in its model's folder when step 1 applies that model, removed since the
    fill, is said in the plan's own words, never as the file used last time, and the list does not select it."""
    import shutil
    from core.config import BOUNDS_PATH
    from core.gui.panels.inference import config_tab, prior_tab
    bounds = tmp_path / BOUNDS_PATH.name
    shutil.copytree(BOUNDS_PATH / "nadrowski", bounds / "nadrowski")
    monkeypatch.setattr(config_tab, "BOUNDS_PATH", bounds)
    monkeypatch.setattr(prior_tab, "BOUNDS_PATH", bounds)
    window, screen, page, _folder, _clock = _the_window(tmp_path, monkeypatch, {"retrain": RETRAIN})
    prior = screen.prior_panel
    _press_fill(page, monkeypatch, answer="Fill the forms")
    assert prior._pending == "master_tier1.txt"
    (bounds / "nadrowski" / "master_tier1.txt").unlink()
    page.foot.main_button.click()
    assert prior.bounds_picker.key() != "master_tier1.txt" and not prior.gone_line.isHidden()
    assert _said(prior.gone_line) == "The plan's bounds file, master_tier1.txt, is not in this model's bounds folder."


def test_the_windows_two_checks_are_listed_as_the_readers_are(tmp_path, monkeypatch):
    """Two checks need more than the file: its bounds file is in the model's own folder, judged against that folder's
    listing, so that a name in another case, which Windows would find, is refused as the list would not select it, and
    said to stand there in other capitals; and its units resolve. Each is a problem of the plan, in the reader's own
    form, under the refusal's own sentence, its field the plan. A plan the reader refuses is refused with the reader's
    problems alone."""
    from core import plans as engine_plans
    from core.config import BOUNDS_PATH
    from core.gui import plans as window_plans
    two = "# Model and options\nunits = kHz notaunit\n\n# Prior\nbounds = nothere.txt\n"
    folder = _the_plans(tmp_path, monkeypatch, {"two": two, "case": "# Prior\nbounds = MASTER.TXT\n"})
    bounds = BOUNDS_PATH / "nadrowski"
    if sys.platform == "win32":
        assert (bounds / "MASTER.TXT").is_file(), "Windows no longer finds a name in another case"
    with pytest.raises(engine_plans.PlanRefusal) as refused:
        window_plans.read_checked(folder / "two.txt", bounds_folder=bounds)
    said = [(problem.key, problem.message) for problem in refused.value.problems]
    assert refused.value.field == "plan" and refused.value.message == "This plan cannot be used: 2 problems in two.txt"
    assert said[0] == ("bounds", "The bounds file nothere.txt is not in this model's bounds folder.")
    assert said[1][0] == "units" and said[1][1].startswith("The units are not usable: ")
    with pytest.raises(engine_plans.PlanRefusal) as refused:
        window_plans.read_checked(folder / "case.txt", bounds_folder=bounds)
    assert [problem.message for problem in refused.value.problems] == [
        "The bounds file MASTER.TXT is not in this model's bounds folder; master.txt is. A plan names its file with "
        "the same capitals."]
    assert window_plans.read_checked(_plan(folder, "# Prior\nbounds = master.txt\n", "fine"),
                                     bounds_folder=bounds).values == {("Prior", "bounds"): "master.txt"}
    with pytest.raises(engine_plans.PlanRefusal) as refused:
        window_plans.read_checked(_plan(folder, "# Prior\nbounds = nothere.txt\nnum_run = 5\n", "both"),
                                  bounds_folder=bounds)
    assert [problem.message for problem in refused.value.problems] == [
        "line 3: 'num_run' is not a setting PRISM has. The nearest is num_runs."], "the reader's problems stood alone"


def test_a_model_changed_returns_the_list_to_no_plan(tmp_path, monkeypatch, store):
    """The list follows the model shown on step 1's form: another model chosen there shows "No plan", lists that
    model's own plans as it opens, and leaves no plan in use; the values on the forms stay as they are, and say they
    are changed from the usual."""
    window, screen, page, folder, _clock = _the_window(tmp_path, monkeypatch, {"retrain": RETRAIN})
    hopf = folder.parent / "hopf"
    hopf.mkdir()
    _plan(hopf, "# Training\nnum_runs = 3000\n", "hopf_plan")
    _press_fill(page, monkeypatch, answer="Fill the forms")
    assert screen.plan_source.name == "retrain"
    page.model_combo.setCurrentText("HOPF")
    assert _listed(page) == ["No plan"] and screen.plan_source is None
    assert _values(screen)[("training", "num_runs")] == 10000
    assert _origins(screen)[("training", "num_runs")] == "changed from the usual"
    assert [_said(one.plan_line) for one in screen.step_pages()] == ["No plan"] * 3
    assert _opened(page) == ["No plan", "hopf_plan"]
    page.model_combo.setCurrentText("NADROWSKI")
    assert _listed(page) == ["No plan"] and _opened(page) == ["No plan", "retrain"]


def test_the_origin_of_every_row_follows_the_source_in_use(tmp_path, monkeypatch, store):
    """Beside every row a plan fills stands where its value came from, worked out after every edit, fill and apply: its
    usual value, or changed from it; put back from last time; from the plan, or changed from it; and as a record's or a
    run's when one filled the forms. A bounds file has no usual value, and says nothing until a source holds one."""
    from core.gui import plans as window_plans
    from tests._runs import at
    window, screen, page, _folder, _clock = _the_window(tmp_path, monkeypatch, {"retrain": RETRAIN})
    training = screen.posterior_panel
    row = training.form.rows["num_runs"]
    assert set(_origins(screen).values()) == {"usual", ""}
    assert _origins(screen)[("prior", "bounds")] == ""
    row.set_value(7000)
    assert _said(row.origin) == "changed from the usual"
    row.kept = True
    training.redraw()
    assert _said(row.origin) == "kept from last time"
    _press_fill(page, monkeypatch, answer="Fill the forms")
    assert _said(row.origin) == "from plan retrain"
    assert _origins(screen) == {pair: "from plan retrain" if pair in FILLED else "usual" for pair in _origins(screen)}
    row.set_value(9500)
    assert _said(row.origin) == "changed from the plan"
    page.foot.main_button.click()
    assert _said(row.origin) == "changed from the plan" and _said(page.form.rows["chi_mode"].origin) == (
        "from plan retrain")
    record = window_plans.PlanSource("record", "retrain_posterior", None, "", at(8, 12, 4),
                                     {("Training", "num_runs"): 9500}, ident="20261008T120400")
    screen.take_plan(record)
    assert _said(row.origin) == "from record retrain_posterior"
    assert _said(training.form.rows["hidden_features"].origin) == "changed from the usual"
    row.set_value(9600)
    assert _said(row.origin) == "changed from the record"
    screen.take_plan(window_plans.PlanSource("run", "retrain_3", None, "", at(8, 12, 4),
                                             {("Training", "num_runs"): 9600}, ident="20261008T130000"))
    assert _said(row.origin) == "from run retrain_3"
    assert [_said(one.plan_line) for one in screen.step_pages()][2].startswith("Settings from run retrain_3")


def test_the_hash_is_read_once_until_the_file_changes(tmp_path, monkeypatch, store):
    """The file of the plan in use is read for its hash at the steps' readings, once until it is written again: a
    reading after a write that keeps its size and moves its moment reads it again, and so does one that changes its
    size."""
    import os
    from core.gui import plans as window_plans
    window, screen, page, folder, _clock = _the_window(tmp_path, monkeypatch, {"retrain": RETRAIN})
    _press_fill(page, monkeypatch, answer="Fill the forms")
    hashed = []
    kept = window_plans._hash_of
    monkeypatch.setattr(window_plans, "_hash_of", lambda path: (hashed.append(path), kept(path))[1])
    path = folder / "retrain.txt"
    for _ in range(3):
        page.read_facts()
        screen.posterior_panel.read_facts()
    assert hashed == [path] and screen.plan_file == screen.plan_source.sha256
    moment = os.stat(path).st_mtime_ns
    path.write_bytes(RETRAIN.replace("10000", "20000").encode("utf-8"))
    os.utime(path, ns=(moment + 5_000_000_000, moment + 5_000_000_000))
    assert len(path.read_bytes()) == len(RETRAIN.encode("utf-8"))
    page.read_facts()
    page.read_facts()
    assert hashed == [path, path] and screen.plan_file not in (None, screen.plan_source.sha256)
    path.write_bytes(RETRAIN.replace("10000", "200000").encode("utf-8"))
    os.utime(path, ns=(moment + 5_000_000_000, moment + 5_000_000_000))
    page.read_facts()
    assert hashed == [path, path, path]


def _a_training_starts(screen, monkeypatch, tmp_path, during=None) -> dict:
    """A prior and its configuration put in use, the Training page shown, a name typed and its press made, the stage a
    stand-in run that is released at once, ``during`` called while it is live: the run's history as it was first
    written."""
    from core import orchestrator
    from core.artifacts import default_store
    from core.gui.run import history
    from core.gui.run.controller import controller
    from core.gui.session import SbiSession
    from tests._fixtures import _nad_cfg, _prior_artifact, qt_app
    from tests._runs import StandIn, turn_until
    app = qt_app()
    page = screen.posterior_panel
    if screen.session.inf_prior is None:
        cfg = _nad_cfg()
        made = _prior_artifact(default_store(), cfg, name=f"retrain_prior_{len(default_store().list('prior'))}").id
        screen.session = SbiSession(draft=screen.session.draft, cfg=cfg,
                                    inf_prior=default_store().load_prior(cfg, made))
        screen.refresh_gates()
    screen.window().go_to("inference", "training")
    page.names.name.setText("retrain_3")
    page.read_facts()
    stand_in = StandIn(word="Training")
    monkeypatch.setattr(orchestrator, "build_posterior", stand_in)
    page._main_clicked()
    assert controller().is_live() and stand_in.started.wait(5), page.foot.reason.text()
    first = history.read(tmp_path / "runs" / f"{controller().status.run}.json")
    if during is not None:
        during()
    stand_in.release()
    turn_until(app, lambda: not controller().is_live(), 10.0)
    return first


def test_a_plan_changed_or_removed_after_the_fill_is_said_and_stops_nothing(tmp_path, monkeypatch, store):
    """A plan written again after its fill is said at the next reading, the forms keeping what was filled; one taken
    away is said too; neither stops the start, and the run's history says which: the file's hash then, changed, or
    "missing" with none."""
    window, screen, page, folder, _clock = _the_window(tmp_path, monkeypatch, {"retrain": RETRAIN})
    _press_fill(page, monkeypatch, answer="Fill the forms")
    page.foot.main_button.click()
    filled = screen.plan_source.sha256
    path = folder / "retrain.txt"
    path.write_bytes((RETRAIN + "max_num_epochs = 300\n").encode("utf-8"))
    screen.posterior_panel.read_facts()
    changed = ("Attention. retrain.txt has been changed since its settings were filled in on 8 Oct 2026, 12:04. The "
               "forms hold the earlier values; fill again to take the changes.")
    assert changed in _facts(screen.posterior_panel)
    assert screen.posterior_panel.form.rows["max_num_epochs"].value() is None
    entry = _a_training_starts(screen, monkeypatch, tmp_path)["plan"]
    assert (entry["changed"], entry["filled_sha256"], entry["sha256"] != filled) == (True, filled, True), entry
    path.unlink()
    screen.posterior_panel.read_facts()
    assert ("Attention. retrain.txt is no longer in the plans folder. The forms hold the values filled from it on 8 "
            "Oct 2026, 12:04.") in _facts(screen.posterior_panel)
    entry = _a_training_starts(screen, monkeypatch, tmp_path)["plan"]
    assert (entry["changed"], entry["sha256"], entry["name"]) == ("missing", None, "retrain"), entry


def test_the_run_history_records_the_plan(tmp_path, monkeypatch, store):
    """A training's history keeps the plan in use as the run starts: its name, its file and its hash then, the hash and
    the moment of the fill, and whether the file changed since; for each setting the run reads, where its value came
    from and the plan's value where the plan holds it. With no plan in use it keeps none; with a record as the source,
    its kind, id and name. Names and numbers only."""
    import json
    from core.gui import plans as window_plans
    from tests._runs import at
    from core.gui import settings
    from core.gui.remembered import LAST_USED_KEY
    from tests._dialogs import choose
    long_name = "retrain_on_the_temperature_box"
    window, screen, page, folder, _clock = _the_window(tmp_path, monkeypatch, {long_name: RETRAIN})
    assert _a_training_starts(screen, monkeypatch, tmp_path)["plan"] is None
    window.go_to("inference", "model")
    _press_fill(page, monkeypatch, long_name, answer="Fill the forms")
    screen.posterior_panel.form.rows["hidden_features"].set_value(192)
    first = _a_training_starts(screen, monkeypatch, tmp_path)
    sha = screen.plan_source.sha256
    assert first["plan"] == {"kind": "plan", "name": long_name, "path": str((folder / f"{long_name}.txt").resolve()),
                             "sha256": sha, "filled_sha256": sha, "filled_at": at(8, 12, 4).isoformat(),
                             "changed": False}, first["plan"]
    rows = {row["name"]: row for row in first["settings"]["rows"]}
    assert {key: (rows[key]["value"], rows[key]["origin"], rows[key]["plan"], rows[key]["usual"])
            for key in ("num_runs", "hidden_features", "learning_rate", "stop_after_epochs")} == {
        "num_runs": (10000, f"from plan {long_name}", 10000, 5000),
        "hidden_features": (192, "changed from the plan", 256, 128),
        "learning_rate": (0.0005, f"from plan {long_name}", 0.0005, 0.001),
        "stop_after_epochs": (20, "usual", None, 20)}
    json.dumps(first["plan"]), json.dumps(first["settings"])
    window.go_to("inference", "model")
    choose(monkeypatch, words="Apply and start a new session")
    page.foot.main_button.click()
    qs = settings.settings()
    qs.sync()
    assert qs.value(LAST_USED_KEY.format("nadrowski")) == long_name
    screen.take_plan(window_plans.PlanSource("record", "retrain_posterior", None, "", at(8, 12, 4),
                                             {("Training", "num_runs"): 10000}, ident="20261008T120400"))
    assert _a_training_starts(screen, monkeypatch, tmp_path)["plan"] == {
        "kind": "record", "id": "20261008T120400", "name": "retrain_posterior"}
    qs.sync()
    assert qs.value(LAST_USED_KEY.format("nadrowski")) == "", "a record was remembered as the plan last used"


def test_the_last_used_keys_follow_what_was_applied_or_started(tmp_path, monkeypatch, store):
    """The plan last used with the model, and when, is written as step 1 is applied, a prior or a posterior comes into
    use, or a training starts with it in use, never as it is filled; a step applied or a run started with no plan in use
    writes both blank, which takes the hint away."""
    from core.gui import settings
    from core.gui.remembered import LAST_USED_AT_KEY, LAST_USED_KEY
    from tests._runs import at

    def remembered():
        qs = settings.settings()
        qs.sync()
        return qs.value(LAST_USED_KEY.format("nadrowski")), qs.value(LAST_USED_AT_KEY.format("nadrowski"))

    window, screen, page, _folder, clock = _the_window(tmp_path, monkeypatch, {"retrain": RETRAIN})
    _press_fill(page, monkeypatch, answer="Fill the forms")
    assert remembered() == (None, None), "a plan was remembered as it was filled"
    clock.advance(60)
    page.foot.main_button.click()
    assert remembered() == ("retrain", at(8, 12, 5).isoformat())
    clock.advance(60)
    started = []
    _a_training_starts(screen, monkeypatch, tmp_path, during=lambda: started.append(remembered()))
    assert started == [("retrain", at(8, 12, 6).isoformat())], "the training's start did not remember the plan"
    assert remembered() == ("retrain", at(8, 12, 6).isoformat())
    window.go_to("inference", "model")
    _press_fill(page, monkeypatch, None, answer="Return to the usual values")
    assert screen.plan_source is None
    assert remembered() == ("retrain", at(8, 12, 6).isoformat()), "a return to the usual values was remembered"
    _a_training_starts(screen, monkeypatch, tmp_path, during=lambda: started.append(remembered()))
    assert started[1:] == [("", "")] and remembered() == ("", "")
    window.go_to("inference", "model")
    page.read_facts()
    assert page.form.rows["plan"].origin.text() == ""


def test_the_foot_of_the_first_step_says_when_it_read_only_while_a_plan_is_in_use(tmp_path, monkeypatch, store):
    """Step 1 reads the file of the plan in use: while one is, the line of when it read and "Check again" stand, at the
    next drawing, and "Check again" is a stop of the Tab key; with none they go, at the next drawing too."""
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication
    from tests._look_helpers import _settle
    from tests._window_helpers import in_front
    window, screen, page, _folder, _clock = _the_window(tmp_path, monkeypatch, {"retrain": RETRAIN})
    foot = page.foot
    assert page.reads_the_disk is False and foot.reading.isHidden() and foot.check_again.isHidden()
    _press_fill(page, monkeypatch, answer="Fill the forms")
    assert page.reads_the_disk is True and not foot.reading.isHidden() and not foot.check_again.isHidden()
    page.read_facts()
    assert foot.reading.text() == "The saved records were read at 12:04."
    _settle()
    with in_front(window):
        page.open_button.setFocus(Qt.TabFocusReason)
        stops = []
        for _ in range(12):
            QTest.keyClick(QApplication.focusWidget(), Qt.Key_Tab)
            _settle()
            stops.append(QApplication.focusWidget())
        assert foot.check_again in stops and stops.index(foot.check_again) < stops.index(foot.main_button)
    _press_fill(page, monkeypatch, None, answer="Return to the usual values")
    assert page.reads_the_disk is False and foot.reading.isHidden() and foot.check_again.isHidden()


def test_a_held_enter_answers_neither_the_fill_nor_the_save(tmp_path, monkeypatch):
    """In the whole program, whose ring follows the keyboard as it does for a person: Enter held down as the fill's
    question or the save's dialog opens answers neither, each still standing after the repeats. Enter pressed afresh
    then gives the question its preselected answer, "Keep what is there", which fills nothing; the save's dialog, a good
    name typed in it so that its answer stands ready, takes nothing from the repeats, and nothing is saved."""
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    from tests._dialogs import all_asked, forget_asked, held_enter
    from tests._fixtures import whole_program
    from tests._look_helpers import _arrive, _settle
    from tests._runs import Clock, at, controller_on
    from tests._window_helpers import in_front, shown_off_screen
    folder = _the_plans(tmp_path, monkeypatch, {"retrain": RETRAIN})
    monkeypatch.setenv("PRISM_ARTIFACTS", str(tmp_path / "Artifacts"))
    with whole_program() as (_app, window):
        controller_on(tmp_path, Clock(at(8, 12, 4)))
        shown_off_screen(window, 1300, 820)
        window.go_to("inference", "model")
        _settle()
        screen = window.part("inference")
        page = screen.config_panel
        screen.posterior_panel.form.rows["num_runs"].set_value(9500)
        page.redraw()
        _choose(page, "retrain")
        seen = []
        forget_asked()
        with in_front(window):
            for button, held in ((page.fill_button, {}), (page.save_button, {"line": "zz_held", "afresh": False})):
                held_enter(monkeypatch, seen, **held)
                button.setFocus(Qt.TabFocusReason)
                _arrive(button, Qt.TabFocusReason)
                QTest.keyClick(button, Qt.Key_Return)
        assert [one.title for one in all_asked()] == ["Fill the forms from plan retrain?", "Save as a plan"]
        assert seen == [(True, False), (True, True)], "a held Enter answered a box, or Enter afresh did not"
        assert screen.plan_source is None and _values(screen)[("training", "num_runs")] == 9500
        assert sorted(path.name for path in folder.iterdir()) == ["retrain.txt"]
        window.hide()


def test_no_plan_is_read_while_a_run_is_live(tmp_path, monkeypatch, store):
    """While a run is live, another page's, nothing of the plans is read: the list opened, a plan chosen, the presses
    reached by code, the readings, "Check again" and the store's change read no folder and no file, and no press asks a
    question. The run's end reads the file of the plan in use once."""
    from core.gui import plans as window_plans
    from core.gui.panels.reduction_panel import ReductionPanel
    from core.gui.run.controller import controller
    from tests._dialogs import all_asked, forget_asked
    from tests._fixtures import qt_app, readings_of_a_run
    from tests._runs import StandIn, start_run, turn_until
    app = qt_app()
    window, screen, page, folder, _clock = _the_window(tmp_path, monkeypatch, {"retrain": RETRAIN})
    _press_fill(page, monkeypatch, answer="Fill the forms")
    _plan(folder, RETRAIN, "another")
    listed = _listed(page)
    calls = readings_of_a_run(monkeypatch)
    stand_in = StandIn(word="Reduction")
    start_run(window.panel(ReductionPanel), stand_in)
    forget_asked()
    page.plan_choice.showPopup()
    page.plan_choice.hidePopup()
    page._plan_chosen()
    page._fill_pressed()
    page._save_pressed()
    assert all_asked() == [], "a press asked a question while a run was live"
    page._read_the_plans()
    for one in screen.step_pages():
        one.read_facts()
        one.foot.check_again.click()
        one.read_timer.timeout.emit()
        one.stale_store()
        one.redraw()
    screen.read_plan_file()
    assert calls == [] and _listed(page) == listed, calls
    stand_in.release()
    turn_until(app, lambda: not controller().is_live())
    assert (calls.count("file_sha"), calls.count("list_plans"), calls.count("save_plan")) == (1, 1, 0), calls
    assert window_plans.PLAN_FOLDER == tmp_path / "plans"


def test_a_fill_with_what_was_put_back_kept_leaves_it_and_says_where_it_came_from(tmp_path, monkeypatch, store):
    """Filled keeping what was put back, a value put back from last time and not touched since is left as it is: it
    reads "changed from the plan" where the plan holds the setting and something else, and "kept from last time" where
    the plan does not hold it. Every other row is filled."""
    from core.config import BOUNDS_PATH
    from core.gui import plans as window_plans
    from core.gui import settings
    from core.gui.remembered import NUM_RUNS_KEY, RUN_SIZE_KEY
    from tests._runs import at
    qs = settings.settings()
    qs.setValue(NUM_RUNS_KEY, "4000")
    qs.setValue(RUN_SIZE_KEY, "64")
    qs.sync()
    window, screen, page, folder, _clock = _the_window(tmp_path, monkeypatch, {"retrain": RETRAIN})
    plan = window_plans.read_checked(folder / "retrain.txt", bounds_folder=BOUNDS_PATH / "nadrowski")
    values = {pair: value for pair, value in plan.values.items() if pair[0] in window_plans.FILLED}
    window_plans.fill_forms(screen, values, window_plans.PlanSource("plan", "retrain", plan.path, plan.sha256,
                                                                    at(8, 12, 4), values), keep_remembered=True)
    rows = screen.posterior_panel.form.rows
    assert (rows["num_runs"].value(), rows["run_size_cap"].value()) == (4000, 64)
    assert (rows["num_runs"].origin.text(), rows["run_size_cap"].origin.text()) == (
        "changed from the plan", "kept from last time")
    assert rows["hidden_features"].value() == 256 and rows["hidden_features"].origin.text() == "from plan retrain"


def test_a_value_put_back_and_filled_over_with_the_same_number_is_the_plans(tmp_path, monkeypatch, store):
    """Filled over, a value put back from last time is no longer said to be kept, even where the plan holds the same
    number, which changes nothing in the field: it reads "from plan retrain"; and one the plan does not hold, set back
    to its usual value, reads "usual"."""
    from core.gui import settings
    from core.gui.remembered import NUM_RUNS_KEY, RUN_SIZE_KEY
    qs = settings.settings()
    qs.setValue(NUM_RUNS_KEY, "10000")
    qs.setValue(RUN_SIZE_KEY, "64")
    qs.sync()
    window, screen, page, _folder, _clock = _the_window(tmp_path, monkeypatch, {"retrain": RETRAIN})
    rows = screen.posterior_panel.form.rows
    assert (rows["num_runs"].origin.text(), rows["run_size_cap"].origin.text()) == ("kept from last time",) * 2
    _press_fill(page, monkeypatch, answer="Fill the forms")
    assert (rows["num_runs"].value(), rows["num_runs"].kept, rows["num_runs"].origin.text()) == (
        10000, False, "from plan retrain")
    assert (rows["run_size_cap"].kept, rows["run_size_cap"].origin.text()) == (False, "usual")


def test_a_fill_while_a_run_is_live_changes_nothing(tmp_path, monkeypatch, store):
    """A fill asked for by code while a run is live changes no form and puts no plan in use."""
    from core.gui import plans as window_plans
    from core.gui.panels.reduction_panel import ReductionPanel
    from tests._runs import running
    window, screen, page, _folder, _clock = _the_window(tmp_path, monkeypatch, {"retrain": RETRAIN})
    before = _values(screen)
    with running(window.panel(ReductionPanel)):
        done = window_plans.fill_forms(screen, {("Training", "num_runs"): 10000},
                                       window_plans.PlanSource("plan", "retrain", None, "", None, {}))
    assert done is False and _values(screen) == before and screen.plan_source is None


def test_a_plans_number_of_batches_has_no_ceiling_and_is_filled_as_it_is(tmp_path, monkeypatch, store):
    """The reader sets no ceiling on the batches: a plan's 1,000,000,000 is filled as it is, said grouped, read by the
    page's rule as a whole number of at least 1, and named in "Before you start"."""
    huge = "# Training\nnum_runs = 1000000000\n"
    window, screen, page, _folder, _clock = _the_window(tmp_path, monkeypatch, {"huge": huge})
    _press_fill(page, monkeypatch, "huge")
    training = screen.posterior_panel
    row = training.form.rows["num_runs"]
    assert (row.value(), row.widget.text(), row.in_error()) == (1_000_000_000, "1,000,000,000", False)
    assert "num_runs" not in training.problems()
    assert _facts(training)[-1] == "Information. Plan huge. Its 1 setting is in use as written."


def test_the_process_holds_the_suites_own_folder_of_plans(tmp_path):
    """For the whole run of the suites the window's folder of plans is the session's own, not the real one beside the
    other inputs, and the process's state names it."""
    from core import config
    from core.gui import plans as window_plans
    from tests._fixtures import process_state, qt_app
    assert window_plans.PLAN_FOLDER != config.PLANS_PATH and window_plans.PLAN_FOLDER.is_dir()
    assert process_state(qt_app())["plans folder"] == window_plans.PLAN_FOLDER


def _spontaneous(screen) -> None:
    page = screen.config_panel
    page.rot_choice.setCurrentIndex(page.rot_choice.findData(False))
    page.form.rows["chi_n_freqs"].set_value(4)
    screen.prior_panel.bounds_picker.restore_key("master.txt")
    screen.posterior_panel.form.rows["num_runs"].set_value(7000)


def _frequency_response(screen) -> None:
    page = screen.config_panel
    page.kind_choice.setCurrentIndex(page.kind_choice.findData(True))
    for key, value in (("chi_n_freqs", 4), ("chi_k_pad", 10), ("chi_max_cycles", 30.5)):
        page.form.rows[key].set_value(value)


def _typed_units(screen) -> None:
    screen.config_panel.plan_fields()["units"].set_value("pN nm ms kHz")


def _typed_limits(screen) -> None:
    prior = screen.prior_panel
    prior.show_view("build")
    prior.bounds_picker.restore_key("master.txt")
    prior.typed_tick.setChecked(True)
    prior.form.rows["num_iterations"].set_value(30)


def _a_changed_row(screen) -> None:
    page = screen.config_panel
    page.foot.main_button.click()
    page.kind_choice.setCurrentIndex(page.kind_choice.findData(True))


SAVED = [
    ("spontaneous", _spontaneous, {("Model and options", "reparam_rotate"): False, ("Prior", "bounds"): "master.txt",
                                   ("Training", "num_runs"): 7000},
     ["Rotation (reparam_rotate): off", "Bounds file (bounds): master.txt", "Batches (num_runs): 7,000"], ""),
    ("frequency response", _frequency_response, {
        ("Model and options", "chi_mode"): True, ("Model and options", "chi_n_freqs"): 4,
        ("Model and options", "chi_k_pad"): 10, ("Model and options", "chi_max_cycles"): 30.5},
     ["Kind of observation (chi_mode): frequency response", "Driven recordings per observation (chi_n_freqs): 4",
      "Slots for driven recordings (chi_k_pad): 10", "Lock-in limit (chi_max_cycles): 30.5"], ""),
    ("typed units", _typed_units, {("Model and options", "units"): "pN nm ms kHz"},
     ["Units (units): kHz ms nm pN"], ""),
    ("typed limits", _typed_limits, {("Prior", "num_iterations"): 30}, ["Sweep rounds (num_iterations): 30"],
     "The bounds were typed by hand, so this plan names no bounds file."),
    ("a changed row", _a_changed_row, {("Model and options", "chi_mode"): True},
     ["Kind of observation (chi_mode): frequency response"],
     "Saved as the forms show them, though not applied: Kind of observation."),
]


@pytest.mark.parametrize("set_up, values, items, beneath", [case[1:] for case in SAVED],
                         ids=[case[0] for case in SAVED])
def test_saving_writes_only_what_differs_and_every_file_it_can_write_reads_back(set_up, values, items, beneath,
                                                                                tmp_path, monkeypatch, store):
    """"Save as a plan…" writes every setting of steps 1 to 3 that differs from its usual value, each with its label as
    its remark and the words the form shows where the file's differ, and the bounds file when one is selected, never
    limits typed by hand; the driven recordings' three only with the frequency response. Its dialog lists what it will
    write, and says when the bounds were typed and which settings it writes as the forms show them though not applied.
    The file reads back as the values it was given."""
    from tests._dialogs import all_asked, answer_in_turn
    window, screen, page, folder, _clock = _the_window(tmp_path, monkeypatch)
    set_up(screen)
    _shown_again(window)
    answer_in_turn(monkeypatch, ("Save", "zz_saved"))
    page.save_button.click()
    dialog = all_asked()[-1]
    assert (dialog.title, dialog.sentence, dialog.label, dialog.answers) == (
        "Save as a plan", "Only the settings that differ from their usual values are written. A setting left out "
        "follows the usual value.", "Plan name", ("Save", "Cancel"))
    assert (list(dialog.items), dialog.beneath) == (items, beneath)
    written = read_plan(folder / "zz_saved.txt")
    assert written.values == values and written.about == ""
    remarks = [line.split(" # ", 1)[1] for line in (folder / "zz_saved.txt").read_text(encoding="utf-8").splitlines()
               if " = " in line]
    if ("Model and options", "chi_mode") in values:
        assert remarks[0] == "Kind of observation: Frequency response", remarks
    else:
        assert remarks == [item.split(" (", 1)[0] for item in items], remarks


def test_saving_never_replaces_without_asking(tmp_path, monkeypatch, store):
    """A name already in use, in any case, is never written over without asking: the question names the file and when it
    was last changed, "Keep the existing plan" preselected, the other drawn as losing work. Kept, the file is as it was;
    replaced, it holds the new plan."""
    import os
    from datetime import datetime
    from tests._dialogs import all_asked, answer_in_turn, forget_asked
    window, screen, page, folder, _clock = _the_window(tmp_path, monkeypatch, {"retrain": RETRAIN})
    path = folder / "retrain.txt"
    os.utime(path, (datetime(2026, 10, 3, 14, 2).timestamp(),) * 2)
    kept = path.read_bytes()
    screen.posterior_panel.form.rows["num_runs"].set_value(7000)
    _shown_again(window)
    answer_in_turn(monkeypatch, ("Save", "Retrain"), ("Keep the existing plan", None))
    page.save_button.click()
    question = all_asked()[-1]
    assert (question.title, question.sentence, question.answers, question.default, question.loses_work) == (
        "Replace plan retrain?", "retrain.txt was last changed on 3 Oct 2026, 14:02. The earlier file cannot be "
        "brought back.", ("Replace it", "Keep the existing plan"), "Keep the existing plan", True)
    assert path.read_bytes() == kept and screen.plan_source is None
    forget_asked()
    answer_in_turn(monkeypatch, ("Save", "retrain"), ("Replace it", None))
    page.save_button.click()
    assert [one.title for one in all_asked()] == ["Save as a plan", "Replace plan retrain?"]
    assert sorted(entry.name for entry in folder.iterdir()) == ["retrain.txt"]
    assert read_plan(path).values == {("Training", "num_runs"): 7000}


def test_nothing_to_save_greys_the_button_with_its_reason(tmp_path, monkeypatch, store):
    """With every setting at its usual value and no bounds file selected there is nothing to write: the button is greyed
    and says why beside it, as the forms change on step 1, and on another step when step 1 is shown again."""
    window, screen, page, _folder, _clock = _the_window(tmp_path, monkeypatch)
    assert not page.save_button.isEnabled()
    assert page.save_reason.text() == "Every setting is at its usual value, so there is nothing to save."
    screen.posterior_panel.form.rows["learning_rate"].set_value(0.002)
    assert not page.save_button.isEnabled(), "step 1 not in sight was drawn"
    _shown_again(window)
    assert page.save_button.isEnabled() and page.save_reason.text() == ""
    rate = screen.posterior_panel.form.rows["learning_rate"]
    rate.set_value(rate.usual())
    screen.prior_panel.bounds_picker.restore_key("master.txt")
    _shown_again(window)
    assert page.save_button.isEnabled(), "a bounds file selected is something to save"
    screen.prior_panel.bounds_picker.restore_key("")
    _shown_again(window)
    assert not page.save_button.isEnabled()


def test_a_plans_name_is_judged_as_it_is_typed(tmp_path, monkeypatch, store):
    """The name is letters, digits, _ and -, at most 40 of them, judged as it is typed: a name that breaks the rule says
    why under the line and greys "Save"; one that keeps it can be saved."""
    from tests._dialogs import answer_in_turn
    window, screen, page, folder, _clock = _the_window(tmp_path, monkeypatch)
    screen.posterior_panel.form.rows["num_runs"].set_value(7000)
    _shown_again(window)
    judged = []

    def typed(dialog) -> None:
        for name in ("retrain v2", "a" * 41, "retrain/2", "x" * 40, "retrain-2_b"):
            dialog.field.setText(name)
            judged.append((name, dialog.wrong, dialog.other_button.isEnabled()))

    answer_in_turn(monkeypatch, typed)
    page.save_button.click()
    rule = "A plan's name is letters, digits, _ and -, at most 40 characters."
    assert judged == [("retrain v2", rule, False), ("a" * 41, rule, False), ("retrain/2", rule, False),
                      ("x" * 40, "", True), ("retrain-2_b", "", True)]
    assert not folder.exists()


def test_the_save_says_to_commit_the_file(tmp_path, monkeypatch, store):
    """A plan saved becomes the plan in use, shown in the list, with no value changed, and the page says to commit the
    file before a long run."""
    from tests._dialogs import answer_in_turn
    window, screen, page, folder, _clock = _the_window(tmp_path, monkeypatch)
    screen.posterior_panel.form.rows["num_runs"].set_value(7000)
    _shown_again(window)
    before = _values(screen)
    answer_in_turn(monkeypatch, ("Save", "zz_saved"))
    page.save_button.click()
    assert _values(screen) == before
    assert (screen.plan_source.kind, screen.plan_source.name, screen.plan_source.path) == (
        "plan", "zz_saved", folder / "zz_saved.txt")
    assert page.plan_choice.currentText() == "zz_saved" and page.open_button.isEnabled()
    assert _said(page.plan_in_use) == "In use: plan zz_saved, filled 8 Oct 2026, 12:04."
    assert page.in_use.said.sentence.plain_text() == (
        "Information. Saved as plan zz_saved. Commit this file before a long run, or the run's records are marked as "
        "made from a changed tree.")
    assert [_said(one.plan_line) for one in screen.step_pages()] == ["Plan: zz_saved"] * 3
    assert screen.posterior_panel.form.rows["num_runs"].origin.text() == "from plan zz_saved"


def test_a_plan_that_cannot_be_written_says_so_and_saves_nothing(tmp_path, monkeypatch, store):
    """What the forms hold that a plan cannot hold is told, every problem in its sentence, and nothing is saved: a value
    left blank, a value its rule refuses, a text that holds a remark's mark."""
    from core.refusals import _what
    from tests._dialogs import all_asked, answer_in_turn
    window, screen, page, folder, _clock = _the_window(tmp_path, monkeypatch)
    training = screen.posterior_panel.form.rows
    training["num_runs"].widget.setText("")
    _shown_again(window)
    answer_in_turn(monkeypatch, ("Save", "zz_saved"), ("Save", "zz_saved"), ("Save", "zz_saved"))
    page.save_button.click()
    told = all_asked()[-1]
    assert (told.title, told.sentence, told.marker, told.answers, told.items) == (
        "Save as a plan", "This plan cannot be saved: 1 problem", "err", ("OK",), ("Batches is blank.",))
    training["num_runs"].set_value(7000)
    screen.config_panel.plan_fields()["units"].set_value("kHz #ms")
    page.save_button.click()
    told = all_asked()[-1]
    assert (told.sentence, told.items) == ("This plan cannot be saved: 1 problem", (
        f"{_what('units')} cannot hold ' #'; got 'kHz #ms'.",)), told
    screen.config_panel.plan_fields()["units"].set_value(None)
    training["hidden_features"].set_value(0)
    page.save_button.click()
    told = all_asked()[-1]
    assert told.sentence == "This plan cannot be saved: 1 problem" and "got 0" in told.items[0], told
    assert not folder.exists()


def test_a_save_whose_write_fails_is_told_and_changes_nothing(tmp_path, monkeypatch, store):
    """A plan whose file cannot be written is told in the save's own question, its file and the system's reason, as an
    error with one answer; nothing changes: no plan comes into use, no line says a plan was saved, the forms stay, and
    no part file is left. Both a write the system refuses for a file another program holds open, on Windows, and any
    other refusal of the system are told so, never by the last-resort box."""
    from core.gui import plans as window_plans
    from tests._dialogs import all_asked, answer_in_turn
    window, screen, page, folder, _clock = _the_window(tmp_path, monkeypatch, {"retrain": RETRAIN})
    screen.posterior_panel.form.rows["num_runs"].set_value(7000)
    _shown_again(window)
    before, kept, writes = _values(screen), (folder / "retrain.txt").read_bytes(), window_plans.save_plan

    def refused(*_args):
        raise PermissionError("the disk said no")

    monkeypatch.setattr(window_plans, "save_plan", refused)
    answer_in_turn(monkeypatch, ("Save", "zz_saved"))
    page.save_button.click()
    told = all_asked()[-1]
    assert (told.title, told.marker, told.sentence, told.answers) == (
        "Save as a plan", "err", "This plan cannot be saved: zz_saved.txt could not be written (the disk said no).",
        ("OK",)), told
    assert screen.plan_source is None and page.in_use.said.isHidden() and _values(screen) == before
    monkeypatch.setattr(window_plans, "save_plan", writes)
    if sys.platform != "win32":
        return                                   # a file held open is replaced elsewhere: the refusal is Windows'
    answer_in_turn(monkeypatch, ("Save", "retrain"), ("Replace it", None))
    with open(folder / "retrain.txt", "rb"):
        page.save_button.click()
    told = all_asked()[-1]
    assert (told.marker, told.sentence) == (
        "err", "This plan cannot be saved: retrain.txt could not be written (Access is denied)."), told
    assert screen.plan_source is None and page.in_use.said.isHidden() and _values(screen) == before
    assert sorted(path.name for path in folder.iterdir()) == ["retrain.txt"], "a part file was left"
    assert (folder / "retrain.txt").read_bytes() == kept


def test_with_its_only_plan_gone_the_forms_can_still_return_to_the_usual_values(tmp_path, monkeypatch, store):
    """A folder whose only plan is the plan in use: its file removed, the list holds "No plan" alone, and the button
    still returns the forms to their usual values, asking first, as it does, when a value typed would be replaced, the
    sentence that the plan is gone beside it."""
    window, screen, page, folder, _clock = _the_window(tmp_path, monkeypatch, {"retrain": RETRAIN})
    _press_fill(page, monkeypatch, answer="Fill the forms")
    (folder / "retrain.txt").unlink()
    page.read_facts()
    assert _listed(page) == ["No plan"]
    assert (page.fill_button.text(), page.fill_button.isEnabled(), page.fill_reason.text()) == (
        "Return the forms to the usual values", True, "The plan in use is no longer in the plans folder.")
    from tests._dialogs import all_asked, choose, forget_asked
    screen.posterior_panel.form.rows["num_runs"].set_value(9500)
    forget_asked()
    choose(monkeypatch, words="Return to the usual values")
    page.fill_button.click()
    assert [(one.title, one.items) for one in all_asked()] == [
        ("Return the forms to the usual values?", ("Batches: 9,500 becomes 5,000",))]
    assert screen.plan_source is None and screen.posterior_panel.form.rows["num_runs"].value() == 5000


def test_a_field_that_does_not_read_is_said_as_such_in_a_view_that_hides_it(tmp_path, monkeypatch, store):
    """Step 2's seven settings are hidden in its view of a saved prior, and the save still writes them: a field there
    that holds a text that does not read is told by its own sentence, never as blank, a whole number's and a
    decimal's."""
    from tests._dialogs import all_asked, answer_in_turn
    window, screen, page, _folder, _clock = _the_window(tmp_path, monkeypatch, {"retrain": RETRAIN})
    prior = screen.prior_panel
    prior.form.rows["num_iterations"].widget.setText("1,5")
    prior.form.rows["walk_step"].widget.setText("1,5")
    prior.show_view("saved")
    assert not prior.form.rows["num_iterations"].is_shown()
    _shown_again(window)
    answer_in_turn(monkeypatch, ("Save", "zz_saved"))
    page.save_button.click()
    assert all_asked()[-1].items == ("Sweep rounds must be a number; got '1,5'.",
                                     "Random-walk step: Use a point as the decimal sign."), all_asked()[-1].items


def test_with_nothing_applied_the_saves_sentence_names_no_model_applied(tmp_path, monkeypatch, store):
    """With nothing applied, step 2's list stands on Nadrowski's folder: a file chosen there and another model shown,
    the save writes no bounds file and says why without naming a model applied."""
    from tests._dialogs import all_asked, answer_in_turn
    window, screen, page, folder, _clock = _the_window(tmp_path, monkeypatch, {"retrain": RETRAIN})
    hopf = _a_plan_of_hopf(folder).parent
    assert screen.session.draft is None
    _set(screen, {("prior", "bounds"): "master.txt"})
    page.model_combo.setCurrentText("HOPF")
    screen.posterior_panel.form.rows["num_runs"].set_value(7000)
    _shown_again(window)
    answer_in_turn(monkeypatch, ("Save", "zz_hopf"))
    page.save_button.click()
    assert all_asked()[-1].beneath == OTHER_BOUNDS, all_asked()[-1].beneath
    assert read_plan(hopf / "zz_hopf.txt").values == {("Training", "num_runs"): 7000}


@pytest.mark.skipif(sys.platform != "win32", reason="Windows keeps one file for both spellings of a name")
def test_a_plan_in_use_renamed_in_case_alone_stays_in_use(tmp_path, monkeypatch, store):
    """A plan in use whose file is renamed in its capitals alone is still the plan in use: the list shows it as the
    folder names it, nothing under it says it changed or is gone, and the fill button is the plan's."""
    window, screen, page, folder, _clock = _the_window(tmp_path, monkeypatch, {"retrain": RETRAIN})
    _press_fill(page, monkeypatch, answer="Fill the forms")
    (folder / "retrain.txt").rename(folder / "Retrain.txt")
    page.read_facts()
    row = page.form.rows["plan"]
    assert (page.plan_choice.currentText(), row.line.words, page.fill_reason.text()) == ("Retrain", "", "")
    assert (page.fill_button.text(), page.fill_button.isEnabled()) == ("Fill the forms from this plan", True)
    assert screen.plan_source.name == "retrain" and not _facts(page)[-1].startswith("Attention."), _facts(page)


def test_the_usual_values_pressed_with_the_plan_in_use_gone_look_for_it_no_more(tmp_path, monkeypatch, store):
    """The plan in use's file removed, the forms returned to their usual values: when the file comes back the list stays
    on "No plan", nothing under it, and no plan is in use."""
    window, screen, page, folder, _clock = _the_window(tmp_path, monkeypatch, {"retrain": RETRAIN})
    _press_fill(page, monkeypatch, answer="Fill the forms")
    (folder / "retrain.txt").unlink()
    page.read_facts()
    assert page.fill_button.text() == "Return the forms to the usual values"
    _press_fill(page, monkeypatch, None)
    assert screen.plan_source is None
    _plan(folder, RETRAIN, "retrain")
    page.read_facts()
    assert (page.plan_choice.currentText(), page.form.rows["plan"].line.words) == ("No plan", ""), (
        page.plan_choice.currentText(), page.form.rows["plan"].line.words)
    assert screen.plan_source is None


def test_two_spellings_of_one_plan_listed_keep_the_plan_in_use_its_own(tmp_path, monkeypatch, store):
    """A file system that keeps case may list two plans whose names differ in case alone: the plan in use is the one of
    its own spelling, kept shown as the list is read again, and the other spelling chosen reads changed, not applied.
    Where only the other spelling is listed, as Windows lists one file, the plan stands under it."""
    from core.gui import plans as window_plans
    from core.gui.panels.inference import config_tab
    assert config_tab._listed_as(("Retrain", "retrain"), "retrain") == "retrain"
    assert config_tab._listed_as(("Retrain", "retrain"), "Retrain") == "Retrain"
    assert config_tab._listed_as(("Retrain",), "retrain") == "Retrain"
    assert config_tab._listed_as(("other",), "retrain") is None and config_tab._listed_as(("retrain",), None) is None
    window, screen, page, folder, _clock = _the_window(tmp_path, monkeypatch, {"retrain": RETRAIN})
    _press_fill(page, monkeypatch, answer="Fill the forms")
    listed = window_plans.list_plans
    monkeypatch.setattr(window_plans, "list_plans", lambda where: ("Retrain", *listed(where)))
    page.read_facts()
    assert _listed(page) == ["No plan", "Retrain", "retrain"]
    assert (page.plan_choice.currentData(), page.form.rows["plan"].line.words) == ("retrain", "")
    page.plan_choice.setCurrentIndex(page.plan_choice.findData("Retrain"))
    assert page.form.rows["plan"].line.words == "Changed, not applied. In use: plan retrain, filled 8 Oct 2026, 12:04."


@pytest.mark.skipif(sys.platform != "win32", reason="Windows keeps one file for both spellings of a name")
def test_a_plan_in_use_renamed_in_case_then_removed_is_said_to_be_gone(tmp_path, monkeypatch, store):
    """The plan in use renamed in its capitals alone, then removed: the list falls back to "No plan" with nothing under
    it, and beside the fill button the plan in use is said to be no longer in the folder."""
    window, screen, page, folder, _clock = _the_window(tmp_path, monkeypatch, {"retrain": RETRAIN})
    _press_fill(page, monkeypatch, answer="Fill the forms")
    (folder / "retrain.txt").rename(folder / "Retrain.txt")
    page.read_facts()
    assert page.plan_choice.currentText() == "Retrain"
    (folder / "Retrain.txt").unlink()
    page.read_facts()
    assert (page.plan_choice.currentText(), page.form.rows["plan"].line.words, page.fill_reason.text()) == (
        "No plan", "", "The plan in use is no longer in the plans folder.")


def test_a_plans_bounds_file_waiting_in_the_folder_the_list_stands_in_is_no_change(tmp_path, monkeypatch, store):
    """With nothing applied the list of bounds files stands in Nadrowski's folder: a plan of Nadrowski filled there, its
    bounds file selected in the list, waits for step 1 and is the very file the build reads, so step 2 says the plan is
    in use as written, and its plan line counts no change."""
    window, screen, page, folder, _clock = _the_window(tmp_path, monkeypatch,
                                                       {"bounds": "# Prior\nbounds = master.txt\n"})
    prior = screen.prior_panel
    _set(screen, {("prior", "bounds"): "master.txt"})
    _press_fill(page, monkeypatch, "bounds")
    assert screen.session.draft is None and prior.waiting_file() == "master.txt"
    assert _plan_fact(prior) == "Information. Plan bounds. Its 1 setting is in use as written.", _plan_fact(prior)
    assert _said(prior.plan_line) == "Plan: bounds"


def test_with_no_file_selected_the_save_says_nothing_of_the_bounds_list(tmp_path, monkeypatch, store):
    """With another model shown than the one applied and no file selected in step 2's list, the save has no bounds file
    to leave out, and says nothing of the list."""
    from tests._dialogs import all_asked, answer_in_turn
    window, screen, page, folder, _clock = _the_window(tmp_path, monkeypatch, {"retrain": RETRAIN})
    _a_plan_of_hopf(folder)
    page.foot.main_button.click()
    _set(screen, {("prior", "bounds"): ""})
    assert screen.prior_panel.bounds_picker.key() == ""
    page.model_combo.setCurrentText("HOPF")
    screen.posterior_panel.form.rows["num_runs"].set_value(7000)
    _shown_again(window)
    answer_in_turn(monkeypatch, ("Save", "zz_hopf"))
    page.save_button.click()
    assert all_asked()[-1].beneath == "", all_asked()[-1].beneath


def test_a_gone_plan_is_not_chosen_again_in_another_models_list(tmp_path, monkeypatch, store):
    """The plan in use's file removed, another model chosen on step 1's form lets the plan go: a plan of that model of
    the same name is not chosen in its list unasked."""
    window, screen, page, folder, _clock = _the_window(tmp_path, monkeypatch, {"retrain": RETRAIN})
    _press_fill(page, monkeypatch, answer="Fill the forms")
    hopf = folder.parent / "hopf"
    hopf.mkdir()
    _plan(hopf, "# Prior\nbounds = cell.txt\n", "retrain")
    (folder / "retrain.txt").unlink()
    page.read_facts()
    page.model_combo.setCurrentText("HOPF")
    page.read_facts()
    assert (page.plan_choice.currentText(), screen.plan_source) == ("No plan", None), page.plan_choice.currentText()


def test_a_name_in_two_other_cases_is_said_not_to_be_there(tmp_path, monkeypatch):
    """A folder that lists two files whose names differ from the plan's only in case, as a file system that keeps case
    may, names neither: the plain sentence stands."""
    from core.gui import plans as window_plans
    monkeypatch.setattr(window_plans.file_manager, "list_dir", lambda folder: ["MASTER.txt", "Master.TXT"])
    assert [problem.message for problem in window_plans.window_problems(
        {("Prior", "bounds"): "master.txt"}, bounds_folder=tmp_path)] == [
        "The bounds file master.txt is not in this model's bounds folder."]


def test_the_save_refuses_what_its_own_window_would_refuse(tmp_path, monkeypatch, store):
    """Before it writes, the save holds what it would write to the window's two checks against the model shown: typed
    units that do not resolve, and a bounds file not in the model's bounds folder (a plan's file removed since its
    fill), are told in the save's refusal with the checks' own sentences, and nothing is written."""
    import shutil
    from core.config import BOUNDS_PATH
    from core.gui.panels.inference import config_tab, prior_tab
    from tests._dialogs import all_asked, answer_in_turn
    bounds = tmp_path / BOUNDS_PATH.name
    shutil.copytree(BOUNDS_PATH / "nadrowski", bounds / "nadrowski")
    monkeypatch.setattr(config_tab, "BOUNDS_PATH", bounds)
    monkeypatch.setattr(prior_tab, "BOUNDS_PATH", bounds)
    window, screen, page, folder, _clock = _the_window(tmp_path, monkeypatch, {"retrain": RETRAIN})
    page.units_choice.setCurrentIndex(page.units_choice.findData("typed"))
    page.units_text.setText("nm notaunit")
    answer_in_turn(monkeypatch, ("Save", "zz_saved"))
    page.save_button.click()
    told = all_asked()[-1]
    assert (told.title, told.sentence) == ("Save as a plan", "This plan cannot be saved: 1 problem"), told
    assert len(told.items) == 1 and told.items[0].startswith("The units are not usable: "), told.items
    page.units_choice.setCurrentIndex(page.units_choice.findData(None))
    _press_fill(page, monkeypatch, answer="Fill the forms")
    assert screen.prior_panel.waiting_file() == "master_tier1.txt"
    (bounds / "nadrowski" / "master_tier1.txt").unlink()
    answer_in_turn(monkeypatch, ("Save", "zz_saved"))
    page.save_button.click()
    told = all_asked()[-1]
    assert told.items == ("The bounds file master_tier1.txt is not in this model's bounds folder.",), told.items
    assert sorted(path.name for path in folder.iterdir()) == ["retrain.txt"]


def test_a_save_with_another_model_shown_writes_no_file_of_the_model_applied(tmp_path, monkeypatch, store):
    """While the model shown on step 1 is not the one applied, step 2's list holds the applied model's files: the save
    writes no bounds file from it, and its dialog says why. A plan's file of the model shown, waiting, is written."""
    from tests._dialogs import all_asked, answer_in_turn
    window, screen, page, folder, _clock = _the_window(tmp_path, monkeypatch, {"retrain": RETRAIN})
    hopf = _a_plan_of_hopf(folder).parent
    page.foot.main_button.click()
    _set(screen, {("prior", "bounds"): "master.txt"})
    page.model_combo.setCurrentText("HOPF")
    screen.posterior_panel.form.rows["num_runs"].set_value(7000)
    _shown_again(window)
    answer_in_turn(monkeypatch, ("Save", "zz_hopf"))
    page.save_button.click()
    dialog = all_asked()[-1]
    assert dialog.beneath == OTHER_BOUNDS, dialog.beneath
    assert read_plan(hopf / "zz_hopf.txt").values == {("Training", "num_runs"): 7000}
    _press_fill(page, monkeypatch, "hopf_bounds", answer="Fill the forms")
    screen.posterior_panel.form.rows["num_runs"].set_value(7000)
    _shown_again(window)
    answer_in_turn(monkeypatch, ("Save", "zz_waiting"))
    page.save_button.click()
    assert all_asked()[-1].beneath == "", all_asked()[-1].beneath
    assert read_plan(hopf / "zz_waiting.txt").values == {("Prior", "bounds"): "cell.txt",
                                                         ("Training", "num_runs"): 7000}


def test_a_field_that_does_not_read_is_said_as_such(tmp_path, monkeypatch, store):
    """A number field that holds a text that does not read, a comma for the decimal point, is not blank: the save's
    refusal says the row's own sentence, by its label, and the fill's question names the text as typed."""
    from tests._dialogs import all_asked, answer_in_turn
    window, screen, page, _folder, _clock = _the_window(tmp_path, monkeypatch, {"retrain": RETRAIN})
    screen.posterior_panel.form.rows["learning_rate"].widget.setText("1,5")
    _shown_again(window)
    answer_in_turn(monkeypatch, ("Save", "zz_saved"))
    page.save_button.click()
    assert all_asked()[-1].items == ("Learning rate: Use a point as the decimal sign.",), all_asked()[-1].items
    asked = _press_fill(page, monkeypatch)
    assert asked[-1].items == ("Learning rate: 1,5 becomes 0.0005",), asked[-1].items


def test_the_saved_line_goes_at_the_next_plan_press(tmp_path, monkeypatch, store):
    """Step 1's line that a plan was saved, and is to be committed, stands until the next press of the plans' row: a
    fill takes it away."""
    from tests._dialogs import answer_in_turn
    window, screen, page, _folder, _clock = _the_window(tmp_path, monkeypatch, {"retrain": RETRAIN})
    screen.posterior_panel.form.rows["num_runs"].set_value(7000)
    _shown_again(window)
    answer_in_turn(monkeypatch, ("Save", "zz_saved"))
    page.save_button.click()
    assert not page.in_use.said.isHidden()
    _press_fill(page, monkeypatch, "retrain", answer="Fill the forms")
    assert page.in_use.said.isHidden(), page.in_use.said.sentence.plain_text()


def test_a_plan_in_use_whose_file_is_gone_is_said_by_its_fact_alone(tmp_path, monkeypatch, store):
    """The plan in use's file removed, the list falls back to "No plan" at its next reading, and no line says the list
    was changed: the Attention fact says why, and beside the button, which returns the forms to their usual values as it
    does with "No plan" chosen, a sentence says the plan is gone. The file back, the list shows the plan again. A choice
    the person makes in the list, with the list shut too, is said as before."""
    window, screen, page, folder, _clock = _the_window(tmp_path, monkeypatch, {"retrain": RETRAIN, "another": RETRAIN})
    _press_fill(page, monkeypatch, answer="Fill the forms")
    kept = (folder / "retrain.txt").read_bytes()
    (folder / "retrain.txt").unlink()
    page.read_facts()
    row = page.form.rows["plan"]
    assert (page.plan_choice.currentText(), row.line.words) == ("No plan", ""), row.line.words
    assert (page.fill_button.text(), page.fill_button.isEnabled(), page.fill_reason.text()) == (
        "Return the forms to the usual values", True, "The plan in use is no longer in the plans folder.")
    assert _facts(page)[-1].startswith("Attention. retrain.txt is no longer in the plans folder."), _facts(page)
    (folder / "retrain.txt").write_bytes(kept)
    page.read_facts()
    assert (page.plan_choice.currentText(), row.line.words, page.fill_button.isEnabled()) == ("retrain", "", True)
    (folder / "retrain.txt").unlink()
    page.read_facts()
    page.plan_choice.setCurrentIndex(page.plan_choice.findData("another"))   # with the arrow keys, the list shut
    page.plan_choice.setCurrentIndex(0)
    assert row.line.words.startswith("Changed, not applied. In use: plan retrain"), row.line.words
    assert (page.fill_button.text(), page.fill_button.isEnabled()) == ("Return the forms to the usual values", True)


def test_nine_problems_are_said_whole_and_ten_counted_after_eight(tmp_path, monkeypatch, store):
    """A plan of nine problems has them all said: a line that would count one more says it instead. Ten are eight said
    and the rest counted."""
    nine = "\n".join(["# Training"] + [f"note {at}" for at in range(9)])
    ten = "\n".join(["# Training"] + [f"note {at}" for at in range(10)])
    window, screen, page, _folder, _clock = _the_window(tmp_path, monkeypatch, {"nine": nine, "ten": ten})
    _choose(page, "nine")
    shown = [_said(line) for line in page.plan_problems if not line.isHidden()]
    assert len(shown) == 9 and shown[-1] == (
        "line 10: 'note 8' is not a setting line. Write name = value under a heading."), shown
    _choose(page, "ten")
    shown = [_said(line) for line in page.plan_problems if not line.isHidden()]
    assert len(shown) == 9 and shown[-1] == "And 2 more.", shown


def test_a_source_with_no_moment_says_none(tmp_path, monkeypatch, store):
    """A source that keeps no moment of its fill is named without one: "In use", the line under the list, and the
    Attention fact of its file removed say no ", filled …" and no "on …"."""
    from core.gui import plans as window_plans
    window, screen, page, folder, _clock = _the_window(tmp_path, monkeypatch, {"retrain": RETRAIN})
    path = folder / "retrain.txt"
    screen.take_plan(window_plans.PlanSource("plan", "retrain", path, window_plans.file_sha(path, {}), None,
                                             {("Training", "num_runs"): 10000}))
    page.redraw()
    assert _said(page.plan_in_use) == "In use: plan retrain.", _said(page.plan_in_use)
    assert page.form.rows["plan"].line.words == "Changed, not applied. In use: plan retrain."
    changed = window_plans.PlanSource("record", "retrain_posterior", None, "", None, {}, ident="20261008T120400")
    screen.take_plan(changed)
    page.redraw()
    assert _said(page.plan_in_use) == "In use: settings from record retrain_posterior."
    screen.take_plan(window_plans.PlanSource("plan", "retrain", path, "a" * 64, None,
                                             {("Training", "num_runs"): 10000}))
    path.unlink()
    page.read_facts()
    assert _facts(page)[-1] == (
        "Attention. retrain.txt is no longer in the plans folder. The forms hold the values filled from it.")
    path.write_text("# Training\nnum_runs = 9000\n", encoding="utf-8")
    page.read_facts()
    assert _facts(page)[-1] == ("Attention. retrain.txt has been changed since its settings were filled in. The forms "
                                "hold the earlier values; fill again to take the changes.")


def test_the_training_reads_the_rotation_step_one_holds(tmp_path, monkeypatch, store):
    """Training's "Before you start" reads step 1 as step 1 holds it: the rotation and the units changed on step 1's
    form, not applied, are no departure of the training's run."""
    window, screen, page, _folder, _clock = _the_window(tmp_path, monkeypatch, {"retrain": RETRAIN})
    _press_fill(page, monkeypatch, answer="Fill the forms")
    page.foot.main_button.click()
    training = screen.posterior_panel
    training.redraw()
    held = _plan_fact(training)
    page.rot_choice.setCurrentIndex(page.rot_choice.findData(False))
    page.units_choice.setCurrentIndex(page.units_choice.findData("typed"))
    page.units_text.setText("kHz ms nm pN")
    training.redraw()
    assert held.startswith("Information. Plan retrain"), held
    assert _plan_fact(training) == held, (held, _plan_fact(training))


def test_the_rotations_three_are_no_departure_while_unused(tmp_path, monkeypatch, store):
    """With the rotation off in the configuration held, Training's run reads none of the rotation's three settings: one
    set off its usual value is no departure."""
    from types import SimpleNamespace
    from core.config import BOUNDS_PATH
    from core.gui.session import SbiSession
    window, screen, page, _folder, _clock = _the_window(tmp_path, monkeypatch)
    page.foot.main_button.click()
    cfg = SimpleNamespace(model="NADROWSKI", sources={"bounds": str(BOUNDS_PATH / "nadrowski" / "master.txt")},
                          params_dict={}, rescale_params={}, force_params_dict={}, observation_mode="spontaneous",
                          check_unit_consistency=lambda: [], reparam_rotate=False)
    prior = SimpleNamespace(kind="prior", id="p_old", name="old_prior", manifest=None)
    screen.session = SbiSession(draft=screen.session.draft, cfg=cfg, inf_prior=prior)
    screen.refresh_gates()
    training = screen.posterior_panel
    training.form.rows["fisher_m"].set_value(32)
    training.redraw()
    assert _plan_fact(training) == "Information. No plan. Every setting is at its usual value.", _plan_fact(training)


def test_the_driven_three_are_no_departure_with_the_kind_off(tmp_path, monkeypatch, store):
    """With the spontaneous kind on step 1's form, a run reads none of the driven recordings' settings: one set off its
    usual value is no departure."""
    window, screen, page, _folder, _clock = _the_window(tmp_path, monkeypatch)
    page.form.rows["chi_n_freqs"].set_value(4)
    page.redraw()
    assert page.form.rows["chi_n_freqs"].value() == 4 and page.kind_choice.currentData() is False
    assert _plan_fact(page) == "Information. No plan. Every setting is at its usual value.", _plan_fact(page)


def test_the_saved_view_of_step_two_reads_none_of_the_seven(tmp_path, monkeypatch, store):
    """Step 2's build reads its seven settings and a load reads none: one set off its usual value is a departure in the
    build view and none in the saved view."""
    window, screen, page, _folder, _clock = _the_window(tmp_path, monkeypatch)
    page.foot.main_button.click()
    prior = screen.prior_panel
    prior.show_view("build")
    prior.form.rows["num_iterations"].set_value(30)
    prior.redraw()
    built = _plan_fact(prior)
    prior.show_view("saved")
    prior.redraw()
    assert built.startswith("Information. No plan. 1 setting differs from the usual: Sweep rounds 30"), built
    assert _plan_fact(prior) == "Information. No plan. Every setting is at its usual value.", _plan_fact(prior)


def test_a_plan_under_the_inputs_is_recorded_by_its_path_under_them(tmp_path, monkeypatch, store):
    """A plan that stands under the inputs is recorded by its path there, as the other inputs are."""
    from core import config
    window, screen, page, _folder, _clock = _the_window(tmp_path, monkeypatch, {"retrain": RETRAIN})
    _press_fill(page, monkeypatch, answer="Fill the forms")
    monkeypatch.setattr(config, "RESOURCES_ROOT", tmp_path)
    assert screen.plan_entry()["path"] == "plans/nadrowski/retrain.txt", screen.plan_entry()


def test_a_trainings_start_writes_the_keys_of_the_model_held(tmp_path, monkeypatch, store):
    """A training's start writes the plan last used with the model step 1 holds, never the model the form shows: with
    another model shown, that model's hint is left as it was."""
    from core.gui import settings
    from core.gui.remembered import LAST_USED_KEY
    window, screen, page, _folder, _clock = _the_window(tmp_path, monkeypatch, {"retrain": RETRAIN})
    _press_fill(page, monkeypatch, answer="Fill the forms")
    page.foot.main_button.click()
    page.model_combo.setCurrentText("HOPF")
    _a_training_starts(screen, monkeypatch, tmp_path)
    qs = settings.settings()
    qs.sync()
    assert (qs.value(LAST_USED_KEY.format("nadrowski")), qs.value(LAST_USED_KEY.format("hopf"))) == ("", None), (
        qs.value(LAST_USED_KEY.format("nadrowski")), qs.value(LAST_USED_KEY.format("hopf")))


def test_two_plans_alike_in_moment_and_size_keep_their_own_hashes(tmp_path):
    """Two plan files of the same moment and size, as a copy on a coarse file system makes them, are read for their own
    hashes."""
    import os
    from core.gui import plans as window_plans
    one = _plan(tmp_path, "# Training\nnum_runs = 7000\n", "one")
    two = _plan(tmp_path, "# Training\nnum_runs = 8000\n", "two")
    moment = os.stat(one).st_mtime_ns
    os.utime(two, ns=(moment, moment))
    cache = {}
    assert window_plans.file_sha(one, cache) != window_plans.file_sha(two, cache)


def test_a_models_plans_are_listed_and_a_taken_name_is_found_without_case(tmp_path):
    """The plans of a model's folder, one folder for each model in lower case, are its .txt files, a suffix in capitals
    too, by their names in the order of the alphabet without regard to case; a folder not there lists none. A plan's
    file is the folder's own, and the plan a save would replace is found whatever the case of the name typed."""
    from core.gui import plans as window_plans
    folder = window_plans.model_folder(tmp_path, "NADROWSKI")
    assert folder == tmp_path / "nadrowski"
    assert window_plans.list_plans(folder) == () and window_plans.taken(folder, "retrain") is None
    folder.mkdir()
    for name in ("retrain.txt", "Zeta.TXT", "another.txt", "notes.md"):
        (folder / name).write_bytes(b"# Training\n")
    (folder / "sub.txt").mkdir()
    assert window_plans.list_plans(folder) == ("another", "retrain", "Zeta")
    assert window_plans.plan_path(folder, "Zeta") == folder / "Zeta.TXT"
    assert window_plans.plan_path(folder, "gone") == folder / "gone.txt"
    assert window_plans.taken(folder, "RETRAIN") == folder / "retrain.txt"
    assert window_plans.taken(folder, "zeta") == folder / "Zeta.TXT" and window_plans.taken(folder, "notes") is None


def test_a_plan_files_hash_is_read_once_until_it_is_written_again(tmp_path, monkeypatch):
    """The hash of a plan's file is read once and kept until the file's moment of writing or its size changes: a write
    that keeps the size and moves the moment is read again. A file gone has none, and nothing kept."""
    import os
    from core.gui import plans as window_plans
    hashed = []
    kept = window_plans._hash_of
    monkeypatch.setattr(window_plans, "_hash_of", lambda path: (hashed.append(path), kept(path))[1])
    path, cache = _plan(tmp_path, RETRAIN), {}
    sha = hashlib.sha256(RETRAIN.encode("utf-8")).hexdigest()
    assert [window_plans.file_sha(path, cache) for _ in range(3)] == [sha] * 3 and hashed == [path]
    moment = os.stat(path).st_mtime_ns
    path.write_bytes(RETRAIN.replace("10000", "20000").encode("utf-8"))
    os.utime(path, ns=(moment + 5_000_000_000, moment + 5_000_000_000))
    assert window_plans.file_sha(path, cache) not in (None, sha) and hashed == [path, path]
    path.unlink()
    assert window_plans.file_sha(path, cache) is None and path not in cache


def test_the_fill_reads_the_plan_again_at_its_press(tmp_path, monkeypatch, store):
    """The press reads and checks the plan again: a plan written since it was chosen fills what it holds now, and one
    that has come to have a problem fills nothing and says it."""
    window, screen, page, folder, _clock = _the_window(tmp_path, monkeypatch, {"retrain": RETRAIN})
    _choose(page, "retrain")
    _plan(folder, RETRAIN.replace("num_runs = 10000", "num_runs = 20000"))
    asked = _press_fill(page, monkeypatch)
    assert asked == [] and _values(screen)[("training", "num_runs")] == 20000
    _plan(folder, RETRAIN.replace("num_runs = 10000", "num_runs = many"))
    page.fill_button.click()
    assert _values(screen)[("training", "num_runs")] == 20000
    assert page.plan_trouble.sentence.plain_text() == "This plan cannot be used: 1 problem in retrain.txt"
    assert not page.fill_button.isEnabled()


def test_save_plan_writes_beside_its_file_and_moves_it_into_place(tmp_path, monkeypatch):
    """A plan is written into a file beside it, then moved into place in one step, so that it is never left half
    written; the folder is made when it is not there, and nothing is left beside the plan."""
    import os
    from core.gui import plans as window_plans
    moved = []
    kept = os.replace
    monkeypatch.setattr(os, "replace", lambda source, target: (moved.append((Path(source), Path(target))),
                                                                kept(source, target))[1])
    folder = tmp_path / "plans" / "nadrowski"
    path = window_plans.save_plan(folder, "retrain", "# Training\nnum_runs = 7000\n")
    assert path == folder / "retrain.txt" and path.read_bytes() == b"# Training\nnum_runs = 7000\n"
    assert len(moved) == 1 and moved[0][1] == path and moved[0][0].parent == folder and moved[0][0] != path
    assert sorted(entry.name for entry in folder.iterdir()) == ["retrain.txt"]
    monkeypatch.setattr(os, "replace", lambda source, target: (_ for _ in ()).throw(OSError("the disk is full")))
    with pytest.raises(OSError):
        window_plans.save_plan(folder, "retrain", "# Training\nnum_runs = 9000\n")
    assert sorted(entry.name for entry in folder.iterdir()) == ["retrain.txt"], "a file was left beside the plan"
    assert path.read_bytes() == b"# Training\nnum_runs = 7000\n"


def test_the_load_view_says_the_plan_and_reads_none_of_the_training(tmp_path, monkeypatch, store):
    """The load view's "Before you start" says the plan in use too, of the settings a load reads: step 1's and step 2's
    as they are held, never the training's form, which a load does not read."""
    window, screen, page, _folder, _clock = _the_window(tmp_path, monkeypatch, {"retrain": RETRAIN})
    page.foot.main_button.click()
    training = screen.posterior_panel
    training.form.rows["num_runs"].set_value(9500)
    assert _facts(training)[-1] == ("Information. No plan. 1 setting differs from the usual: Batches 9,500 (usual "
                                   "5,000).")
    training.show_view("load")
    assert _facts(training)[-1] == "Information. No plan. Every setting is at its usual value."
    page.kind_choice.setCurrentIndex(page.kind_choice.findData(True))
    page.foot.main_button.click()
    training.redraw()
    assert _facts(training)[-1] == ("Information. No plan. 1 setting differs from the usual: Kind of observation "
                                   "frequency response (usual set by the bounds file).")


@pytest.mark.parametrize("factor", [1.85, 2.0])
def test_what_a_plan_holds_is_drawn_whole_at_large_text(factor, tmp_path, monkeypatch, store):
    """At 185 and 200 % text the table of what a plan holds draws every word whole: a column fitted to its words is as
    wide as its widest word in the face it is drawn in, and a long name with no space, a sub-folder's path, breaks
    inside onto lines of its own, its row as tall as its lines; the table stays inside the form, which stays inside its
    viewport. The program's font, set for the measure, is put back."""
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QFont, QFontMetricsF
    from core import plans as engine_plans
    from core.gui import design, fonts
    from core.gui.widgets import rule_table
    from tests._fixtures import build_window, qt_app
    from tests._look_helpers import _settle, _the_look, text_size
    from tests._runs import Clock, at, controller_on
    from tests._window_helpers import shown_off_screen
    _the_plans(tmp_path, monkeypatch)
    long_path = "a_sub_folder_of_the_bounds_for_the_cell_of_the_second_of_november/" + "master_" * 10 + "tier1.txt"
    plan = engine_plans.Plan("long", Path("long.txt"), "", "", {
        ("Model and options", "chi_mode"): True, ("Prior", "bounds"): long_path, ("Training", "num_runs"): 10000,
        ("Narrowing round", "hpd_level"): 0.999})
    app = qt_app()
    kept = QFont(app.font())
    try:
        with text_size(factor), _the_look(False):
            fonts.load_app_font(app)
            controller_on(tmp_path, Clock(at(8, 12, 4)))
            window = build_window()
            design.set_room((3400, 1380))
            window.setMinimumSize(1, 1)
            page = window.part("inference").config_panel
            shown_off_screen(window, design.scaled(1000), design.scaled(700))
            window.go_to("inference", "model")
            _settle()
            page.holds_table.set_rows(page._holds_rows(plan))
            page.holds_table.setVisible(True)
            _settle()
            view, model = page.holds_table.view, page.holds_table.view.model()
            line, pad = design.scaled(design.TYPE["body"][1]), design.scaled(design.SIZE["table_pad"])
            for row in range(model.rowCount()):
                for column in range(4):
                    cell = model.index(row, column).data(Qt.UserRole)
                    lead, gap = rule_table._pads(column, 4)
                    room = view.columnWidth(column) - lead - gap
                    layout, lines = rule_table._wrapped(cell.text, rule_table._face("text"), room, line)
                    laid = sum(layout.lineAt(at).textLength() for at in range(lines))
                    assert laid == len(layout.text()), (factor, row, column, cell.text)
                    assert view.rowHeight(row) >= lines * line + 2 * pad, (factor, row, column)
                    widest = max(QFontMetricsF(rule_table._face("text")).horizontalAdvance(word)
                                 for word in cell.text.split())
                    if column != 1 and len(cell.text) < 40:
                        assert widest <= room + 0.5, (factor, column, cell.text, widest, room)
            assert view.rowHeight(1) > 2 * line + 2 * pad, "the long path did not break onto lines of its own"
            port = page.scroller.viewport()
            assert page.holds_table.mapTo(port, page.holds_table.rect().topRight()).x() < port.width()
            assert page.scroller.body.minimumSizeHint().width() <= port.width()
            window.hide()
    finally:
        app.setFont(kept)

