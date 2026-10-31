"""The plan reader: a hand-edited file of settings, read whole or refused with every problem named.

No widget here, and nothing at the top of the file that reaches torch: ``core/plans.py`` is standard library
on purpose, and the few tests that hold it against the rest of the program import what they compare inside
the test. Every file is written byte for byte (``_plan``), since ``write_text`` would turn each line end
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
    pytest.param("# Model and options\nchi_k_pad = 12\n", 2, "chi_k_pad",
                 f"{_what('chi_k_pad')} needs chi_mode = on in the same plan.", id="a chi setting with no switch"),
    pytest.param("# Model and options\nchi_mode = off\nchi_max_cycles = 20\n", 3, "chi_max_cycles",
                 f"{_what('chi_max_cycles')} needs chi_mode = on in the same plan.", id="a chi setting, switch off"),
    pytest.param(f"{OPTIONS}chi_n_freqs = 14\n", 3, "chi_n_freqs",
                 f"{_what('chi_n_freqs')} must be at most the number of slots, 12; got 14.",
                 id="more probes than the built-in slots"),
    pytest.param(f"{OPTIONS}chi_k_pad = 4\n", 3, "chi_n_freqs",
                 f"{_what('chi_n_freqs')} must be at most the number of slots, 4; got 6.",
                 id="fewer slots than the built-in probes"),
])
def test_each_kind_of_problem_says_the_sentence_of_the_table(tmp_path, text, line, key, sentence):
    """One case for each thing a hand-edited plan can get wrong: the one problem, its line, the setting it
    concerns, and the sentence after "line N: ". Each opens with the registry's phrase for the key, and the
    range sentences are the registry's own rules' (a rule's words are not repeated here)."""
    problems = _refused(tmp_path, text).problems
    assert len(problems) == 1, problems
    assert (problems[0].line, problems[0].key) == (line, key)
    assert problems[0].message == f"line {line}: {sentence}"


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


def test_a_kind_is_checked_before_the_registrys_rule(tmp_path):
    """The rule alone would have read this value and truncated it to 10000: the reader refuses the text."""
    problems = _refused(tmp_path, "# Training\nnum_runs = 10000.5\n").problems
    assert [problem.message for problem in problems] == [
        f"line 2: {_what('num_runs')} must be a whole number; got '10000.5'."]


def test_every_registered_key_is_allowed_or_refused_for_a_stated_reason():
    """STANDING. Every key of the registry is a setting a plan holds or has its reason not to be, so a key
    added to the registry cannot slip past the reader unclassified. The picks are listed by name: "every key
    that begins with drive" would take in the drive check's own knobs."""
    names = {name for held in ALLOWED.values() for name in held}
    assert not names & set(CLASSES)
    assert names | set(CLASSES) == set(FIELDS)
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


# One value for each allowed name: its least, unless the rules across lines refuse that with the built-ins beside it.
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
    """The inverse holds or nothing is written: a file the reader would refuse, or one it would read as
    something else, is a refusal at the saving and not a surprise at the next reading."""
    with pytest.raises(PlanRefusal) as refused:
        plan_text(values, about)
    assert refused.value.field == "plan"
    assert [problem.key for problem in refused.value.problems] == [key]
    assert all(problem.line is None for problem in refused.value.problems)


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


def test_a_value_written_with_spaces_around_it_is_written_without_them():
    assert plan_text({("Model and options", "units"): "  nm ms  "}) == "# Model and options\nunits = nm ms\n"


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
    judged again: a switch with a bad word does not also call every chi line lost, and a count already out of
    range is not also compared with the slots."""
    problems = _refused(tmp_path, "# Model and options\nchi_k_pad = 12\n\n# Training\nnum_runs = abc\n").problems
    assert [problem.line for problem in problems] == [2, 5]
    problems = _refused(tmp_path, "# Model and options\nchi_mode = maybe\nchi_k_pad = 12\n").problems
    assert [problem.key for problem in problems] == ["chi_mode"]
    problems = _refused(tmp_path, f"{OPTIONS}chi_n_freqs = abc\nchi_k_pad = 4\n").problems
    assert [problem.key for problem in problems] == ["chi_n_freqs"]
    problems = _refused(tmp_path, f"{OPTIONS}chi_n_freqs = 14\nchi_k_pad = abc\n").problems
    assert [problem.key for problem in problems] == ["chi_k_pad"]


def test_a_whole_number_of_absurd_length_is_a_problem_and_no_traceback(tmp_path):
    """Python will not read a text of over 4300 digits as a number: the reader says so as any other wrong kind."""
    digits = "9" * 5000
    problems = _refused(tmp_path, f"# Training\nnum_runs = {digits}\n").problems
    assert problems[0].message == f"line 2: {_what('num_runs')} must be a whole number; got '{digits}'."


@pytest.mark.parametrize("name", ["../../Cells/nadrowski/x.txt", "C:/x.txt", "C:x.txt", "/x.txt", "//server/x.txt",
                                  "a\\b.txt", "sub/../../x.txt", ".."])
def test_a_bounds_name_that_leaves_the_folder_is_a_problem(tmp_path, name):
    """The window checks that the named file exists under the model's folder, and that check alone would accept
    a name that climbs out of it. A name that could leave the folder is refused here, with its line."""
    problems = _refused(tmp_path, f"# Prior\nbounds = {name}\n").problems
    assert [(problem.line, problem.key) for problem in problems] == [(2, "bounds")]
    assert problems[0].message == (f"line 2: {_what('bounds')} must name a file in the model's bounds folder; "
                                   f"got '{name}'.")


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


def test_the_plan_key_names_the_file_to_correct_on_screen(tmp_path):
    """A refusal's field chooses the words under it. Here the thing to correct is the file, so the window says
    that, and the command line, which reads no plan, adds nothing."""
    from core.gui import fields
    from core.tool import fields as tool_fields
    refusal = _refused(tmp_path, "# Trainng\n")
    assert fields.fix_sentence(refusal.field) == "Correct the file, then choose the plan again."
    assert tool_fields.fix_sentence(refusal.field) == ""
