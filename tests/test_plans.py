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


def test_the_windows_two_checks_are_listed_as_the_readers_are(tmp_path, monkeypatch):
    """Two checks need more than the file: its bounds file is in the model's own folder, judged against that folder's
    listing, so that a name in another case, which Windows would find, is refused as the list would not select it; and
    its units resolve. Each is a problem of the plan, in the reader's own form, under the refusal's own sentence, its
    field the plan. A plan the reader refuses is refused with the reader's problems alone."""
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
        "The bounds file MASTER.TXT is not in this model's bounds folder."]
    assert window_plans.read_checked(_plan(folder, "# Prior\nbounds = master.txt\n", "fine"),
                                     bounds_folder=bounds).values == {("Prior", "bounds"): "master.txt"}
    with pytest.raises(engine_plans.PlanRefusal) as refused:
        window_plans.read_checked(_plan(folder, "# Prior\nbounds = nothere.txt\nnum_run = 5\n", "both"),
                                  bounds_folder=bounds)
    assert [problem.message for problem in refused.value.problems] == [
        "line 3: 'num_run' is not a setting PRISM has. The nearest is num_runs."], "the reader's problems stood alone"


def test_the_process_holds_the_suites_own_folder_of_plans(tmp_path):
    """For the whole run of the suites the window's folder of plans is the session's own, not the real one beside the
    other inputs, and the process's state names it."""
    from core import config
    from core.gui import plans as window_plans
    from tests._fixtures import process_state, qt_app
    assert window_plans.PLAN_FOLDER != config.PLANS_PATH and window_plans.PLAN_FOLDER.is_dir()
    assert process_state(qt_app())["plans folder"] == window_plans.PLAN_FOLDER


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
