"""Run plans: a hand-edited text file of science settings for the inference steps.

A plan stands at ``Resources/Plans/<model>/<name>.txt``, one folder per model, and reads like its
neighbours the cell files: ``#`` heading lines and ``name = value`` lines, each name a key of the refusal
registry. What stands under the first heading is free text, read back as the plan's description, and what
follows white space and ``#`` on a setting line is a remark, never read:

    # About this plan
    The temperature box at the larger network.

    # Model and options
    chi_mode = on                  # Kind of observation: Frequency response

    # Prior
    bounds = master_tier1.txt      # Bounds file

    # Training
    num_runs = 10000               # Batches
    hidden_features = 256          # Network width

A setting the file does not hold means the built-in value.

WHY STRICT. The readers of the cell and bounds files skip what they do not know, and a heading they do not
know silently ends a section. A plan decides days of work on the card, so every line here is understood or
named as a problem with its line, all of them at once, and a plan with any problem is refused whole: never
part of a plan.

WHY THE KIND COMES FIRST. The registry's rules convert with ``int()`` and ``float()``: a text that is no
number raises a bare ValueError there, and a float handed to ``require_at_least`` is truncated without a
word, so 10000.5 would become 10000. The reader checks a value's form itself and hands the rule only what
has that form, so the sentence a person reads is the rule's own or this module's, never a traceback.

STANDARD LIBRARY ONLY, like ``core/refusals.py``: the window reads a plan on its own thread, and the command
line could read one before torch is imported. This module reads one file, no store and no ``core/config.py``,
so the two limits that are constants there and the number form of ``core/Helpers/file_manager.py`` are
literals here, held equal by tests/test_plans.py. Whether the bounds file exists and the units resolve takes
more than this file, so the window checks that. No stage reads a plan.

``plan_text`` is the inverse as pure text, and reads its own text back before it returns it, so reader and
writer cannot drift. The window does the writing and the asking.
"""
import difflib
import hashlib
import re
from dataclasses import dataclass, replace
from itertools import groupby
from pathlib import Path

from core.refusals import (FIELDS, Refusal, _what, default_clause, default_text, require_at_least,
                           require_between, require_finite, require_positive)

# The number form of the cell and bounds files, read with float(): the decimal mark is a point whatever
# Windows' number format is.
NUMBER_FORM = r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?'
# The two limits no registry rule holds: the most probe slots a network may have, and the cycles below which
# a probe is masked, which a lock-in ceiling must clear.
CHI_K_MAX = 24
CHI_MIN_CYCLES = 2

HEADINGS: tuple[str, ...] = ("About this plan", "Model and options", "Prior", "Training", "Calibration",
                             "Inference", "Narrowing round")
_ABOUT, _OPTIONS = HEADINGS[:2]

_FIT = ("num_runs", "run_size_cap", "hidden_features", "num_transforms", "learning_rate", "stop_after_epochs",
        "max_num_epochs", "checkpoint_every")

# Which names each heading holds, in the order a plan is written. Training and the narrowing round share
# names, which is why a plan's values are keyed by heading and name.
ALLOWED: dict[str, tuple[str, ...]] = {
    _OPTIONS: ("chi_mode", "chi_n_freqs", "chi_k_pad", "chi_max_cycles", "reparam_rotate", "units"),
    "Prior": ("bounds", "num_iterations", "sweep_batch", "max_sets", "walk_step", "stability_units",
              "min_cluster_size", "min_samples"),
    "Training": (*_FIT, "fisher_m", "fisher_dz", "fisher_points", "expected_epochs"),
    "Calibration": ("n_cal", "cal_n_scales", "num_posterior_samples"),
    "Inference": ("n_samples",),
    "Narrowing round": ("n_directions", "hpd_level", *_FIT),
}

# Why every other registered key is not in a plan, in the words after "cannot be set by a plan: ". A key added
# to the registry is named here or in ALLOWED in the same commit: no rest is computed, or a new key would be
# classified by nobody's decision, and tests/test_plans.py closes the two against the registry.
_REASONS: dict[str, tuple[str, ...]] = {
    "it is fixed by measurement": ("chi_f0", "chi_freq_bounds"),
    "it belongs to the computer": ("device",),
    "it belongs to one run": ("name", "note", "seed"),
    "it is a consent": ("new_run", "accept_truncated", "accept_other_observation", "resume"),
    "it is a pick, not a setting": ("prior", "posterior", "observation", "cell", "resume_fit", "t_obs", "chi_f0_si",
                                    "recording_spont", "recording_forced", "recording_probe",
                                    "drive_amplitude", "drive_frequency", "drive_phase"),
    "it is the folder the plan stands in": ("model",),
    "it is not a setting of the inference steps": (
        "plan", "artifact", "compare_records", "prefactor", "slice_at",
        # the two secondary analyses, and the five settings neither front end exposes
        "n_freqs", "ensemble_m", "freqs_per_batch", "f0", "preset", "s_grid", "t_grid",
        "freq_bounds", "burn_in_nd", "t_obs_periods", "dt_nd", "psd_t_obs_nd",
        # the live simulation and the model builder
        "frame_steps", "fps", "param_value", "param_min", "param_max", "init", "x_scale", "t_scale",
        "forcing_value",
        # the diagnostics and the probe checks; the drive check's own knobs are not a recording's drive
        "repeats", "n_points", "n_worst", "top_n", "m", "m_noise", "rel", "min_valid", "rows", "n_sweep",
        "chi_k_fixed", "probe_seed", "probe_lengths", "probe_multipliers", "probe_drives", "band_repeats",
        "probe_cycle_caps", "cv_max", "phase_max", "snr_min", "sup_min", "band_peak_window", "mask_num_runs",
        "mask_run_size", "drive_t_obs", "drive_repeats", "drive_detune", "drive_strengths", "free_min",
        "captured_max", "drive_peak_window", "clarity_min"),
}
CLASSES: dict[str, str] = {key: reason for reason, keys in _REASONS.items() for key in keys}

# The kinds. Every allowed name not listed here is a whole number.
_SWITCHES = ("chi_mode", "reparam_rotate")
_NUMBERS = ("learning_rate", "fisher_dz", "walk_step", "stability_units", "hpd_level", "chi_max_cycles")
# The least a whole number may be, where that is not 1. The two chi counts hold the window's floor, not the
# engine's own 1.
_FLOORS = {"run_size_cap": 0, "checkpoint_every": 0, "sweep_batch": 0, "min_cluster_size": 2, "chi_n_freqs": 2,
           "chi_k_pad": 2}
_AT_MOST_THE_SLOTS = ("chi_n_freqs", "chi_k_pad")
# These mean nothing with the switch off: the window sends none of them then.
_ONLY_WITH_CHI_MODE = ("chi_n_freqs", "chi_k_pad", "chi_max_cycles")

_BY_CASE = {heading.lower(): heading for heading in HEADINGS}
_FILE_ORDER = {pair: place for place, pair in enumerate(
    (heading, name) for heading, names in ALLOWED.items() for name in names)}
_LINE_END = re.compile(r"\r\n|\r|\n")
_SETTING = re.compile(r"(\w+)\s*=\s*(.*?)\s*")
_WHOLE = re.compile(r"[+-]?\d+")
_NUMBER = re.compile(NUMBER_FORM)
_REMARK = re.compile(r"\s#")     # white space, not only a space: a remark is lined up with a tab as readily
_NEAREST = 0.8                  # how close a name must be to be offered, difflib's own measure
_REMARK_COLUMN = 30             # a remark is written after this many columns, so the remarks of a plan line up


@dataclass(frozen=True)
class Plan:
    """A plan read whole. ``values`` holds only what the file gives, each checked, keyed by heading and name; the
    units are their words as one text, as ``plan_text`` takes them."""
    name: str
    path: Path
    sha256: str
    about: str
    values: dict[tuple[str, str], object]


@dataclass(frozen=True)
class PlanProblem:
    """One thing wrong: its line (None when the whole file is at fault), the heading in force (None when the
    reader does not know one), the setting it concerns (None when no setting is named) and the sentence, which
    opens with "line N: " where a line is known."""
    line: int | None
    heading: str | None
    key: str | None
    message: str


class PlanRefusal(Refusal):
    """Every problem of one plan. The field is always ``plan``, the thing to correct being the file; each
    problem carries its own setting's key."""

    def __init__(self, message: str, problems: tuple[PlanProblem, ...]):
        super().__init__(message, field="plan")
        self.problems = tuple(problems)


def _counted(problems) -> str:
    return f"{len(problems)} problem{'' if len(problems) == 1 else 's'}"


def _refusal(path: Path, problems: tuple[PlanProblem, ...]) -> PlanRefusal:
    return PlanRefusal(f"This plan cannot be used: {_counted(problems)} in {path.name}", problems)


def read_plan(path) -> Plan:
    """The whole plan at ``path``, or one ``PlanRefusal`` listing every problem. It writes nothing and looks at
    no second file. The byte-order mark Notepad can write is dropped; the hash is of the file's own bytes."""
    path = Path(path)
    try:
        data = path.read_bytes()
    except OSError:
        raise _refusal(path, (PlanProblem(None, None, None, f"The plan was not found: '{path}'."),)) from None
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise _refusal(path, (PlanProblem(None, None, None, f"The plan could not be read as UTF-8 text: '{path}'. "
                                                            "Save it as UTF-8."),)) from None
    about, values, problems = _read(text)
    if problems:
        raise _refusal(path, tuple(replace(p, message=f"line {p.line}: {p.message}") for p in problems))
    return Plan(path.stem, path, hashlib.sha256(data).hexdigest(), about, values)


def plan_text(values, about: str = "", remarks=None) -> str:
    """The text of a plan holding ``values`` (keyed as ``Plan.values``) and the description ``about``, each
    setting followed by its remark from ``remarks``, keyed the same way. ``read_plan`` of this text gives
    ``values`` and ``about`` back, so what could not be read back is a ``PlanRefusal`` and not a file: a
    description line that starts with ``#``, a value that holds a remark's mark, whatever the reader would refuse,
    and whatever it would read as something else. A description's lines are stripped and its blank lines dropped."""
    remarks = remarks or {}
    lines = [line.strip() for line in about.splitlines() if line.strip()]
    problems = [PlanProblem(None, _ABOUT, None, f"A line of the description cannot start with #; got '{line}'.")
                for line in lines if line.startswith("#")]
    blocks = [[f"# {_ABOUT}", *lines]] if lines else []
    # a pair the reader would not take is written all the same, last, so that reading it back refuses it
    for heading, held in groupby(_in_file_order(values), key=lambda pair: pair[0]):
        block = [f"# {heading}"]
        for pair in held:
            name, written = pair[1], _written(values[pair])
            if _REMARK.search(f" {written}"):
                problems.append(PlanProblem(None, heading, name, f"{_what(name)} cannot hold ' #'; got '{written}'."))
            line = f"{name} = {written}"
            block.append(f"{line:<{_REMARK_COLUMN}} # {remarks[pair]}" if remarks.get(pair) else line)
        blocks.append(block)
    text = "\n".join("\n".join(block) + "\n" for block in blocks)
    if not problems:
        back, read, found = _read(text)
        problems = [replace(problem, line=None) for problem in found] or _differences(values, lines, read, back)
    if problems:
        raise PlanRefusal(f"This plan cannot be used: {_counted(problems)}", tuple(problems))
    return text


def _in_file_order(pairs) -> list:
    return sorted(pairs, key=lambda pair: _FILE_ORDER.get(pair, len(_FILE_ORDER)))


def _written(value) -> str:
    return "on" if value is True else "off" if value is False else str(value).strip()


def _differences(values, about: list[str], read: dict, back: str) -> list[PlanProblem]:
    """What the reader took without a problem and would hand back as something else. A value is compared for what it
    means: a text stripped, as it is written, and a whole number handed for a number is the same number. Only the
    first kind of difference is said, since one slip can show as several: a setting nobody handed in is looked for
    last, and a pair under the description shows as the description."""
    if back != "\n".join(about):
        return [PlanProblem(None, _ABOUT, None, "The description would not read back as it was given.")]
    wanted = {pair: value.strip() if isinstance(value, str) else value for pair, value in values.items()}
    differ = [pair for pair, value in wanted.items() if pair not in read or read[pair] != value]
    return [PlanProblem(None, pair[0], pair[1],
                        f"{_what(pair[1])} would not read back as it was given; got {values.get(pair)!r}.")
            for pair in _in_file_order(differ or [pair for pair in read if pair not in wanted])]


def _read(text: str) -> tuple[str, dict[tuple[str, str], object], list[PlanProblem]]:
    """The description, the values and every problem of a plan's text, the problems in file order, each
    sentence without its line. A setting under a heading the reader does not know is still judged on its own
    line, so one misspelt heading does not hide what stands under it, nor call every line there a fault. ``unread``
    gathers each name met on a setting line and not accepted, for the rules across lines."""
    about, values, problems, set_on, unread = [], {}, [], {}, set()
    heading, past_the_top = None, False
    for number, raw in enumerate(_LINE_END.split(text), 1):
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            past_the_top = True
            heading = _BY_CASE.get(line[1:].strip().lower())
            if heading is None:
                problems.append(PlanProblem(number, None, None, f"'{line}' is not a heading a plan has. "
                                                                f"The headings are: {', '.join(HEADINGS)}."))
            continue
        if heading == _ABOUT:
            about.append(line)
            continue
        setting = _SETTING.fullmatch(_REMARK.split(line, maxsplit=1)[0])
        if setting is None or not past_the_top:
            problems.append(PlanProblem(number, heading, None, f"'{line}' is not a setting line. "
                                                               "Write name = value under a heading."))
            if setting is not None:
                unread.add(setting.group(1))
            continue
        name, written = setting.groups()
        refused = _refused_name(name, heading, number, set_on)
        if refused is not None:
            problems.append(PlanProblem(number, heading, *refused))
            unread.add(name)
            continue
        try:
            checked = _checked(name, written)
        except Refusal as refusal:
            problems.append(PlanProblem(number, heading, refusal.field, refusal.message))
            unread.add(name)
            continue
        if heading is None:
            unread.add(name)
        else:
            values[(heading, name)] = checked
    problems += _across_lines(values, set_on, unread)
    problems.sort(key=lambda problem: problem.line)
    return "\n".join(about), values, problems


def _refused_name(name: str, heading: str | None, number: int, set_on: dict) -> tuple[str | None, str] | None:
    """The key and the sentence of why this name cannot stand on this line, or None. ``set_on`` gains the line
    a setting is first set on under its own heading. A name two headings hold is sent to the first of them. The
    nearest name is looked for without case, every key being lower-case, though only the exact name is a setting."""
    if name not in FIELDS:
        nearest = difflib.get_close_matches(name.lower(), FIELDS, n=1, cutoff=_NEAREST)
        return None, f"'{name}' is not a setting PRISM has." + (f" The nearest is {nearest[0]}." if nearest else "")
    if name in CLASSES:
        return name, f"{_what(name)} cannot be set by a plan: {CLASSES[name]}."
    if heading is None:
        return None
    if name not in ALLOWED[heading]:
        home = next(place for place in ALLOWED if name in ALLOWED[place])
        return name, f"{_what(name)} belongs under '# {home}', not '# {heading}'."
    first = set_on.setdefault((heading, name), number)
    if first != number:
        return name, f"{_what(name)} is set twice under '# {heading}' (first on line {first})."
    return None


def _checked(name: str, text: str):
    """One setting's value from its text, or a ``Refusal`` whose field is ``name``: the kind first, then the
    range, by the registry's own rule where it has one."""
    if name in _SWITCHES:
        if text not in ("on", "off"):
            raise Refusal(f"{_what(name)} must be on or off; got '{text}'.", field=name)
        return text == "on"
    if name == "units":
        if not text:
            raise Refusal(f"{_what(name)} are blank: write at least one unit, or leave the line out.", field=name)
        return text
    if name == "bounds":
        return _bounds_name(text)
    if name in _NUMBERS:
        if _NUMBER.fullmatch(text) is None:
            raise Refusal(f"{_what(name)} must be a number; got '{text}'.", field=name)
        return _in_range(name, float(text))
    try:
        number = int(text) if _WHOLE.fullmatch(text) else None
    except ValueError:                  # Python refuses a text of over 4300 digits
        number = None
    if number is None:
        raise Refusal(f"{_what(name)} must be a whole number; got '{text}'.", field=name)
    number = require_at_least(name, number, _FLOORS.get(name, 1))
    if name in _AT_MOST_THE_SLOTS and number > CHI_K_MAX:
        raise Refusal(f"{_what(name)} must be at most {CHI_K_MAX}; got {number}{default_clause(name)}.", field=name)
    return number


def _in_range(name: str, number: float) -> float:
    if name == "hpd_level":
        return require_between(name, number, 0, 1, open_lo=True, open_hi=True)
    if name == "chi_max_cycles":
        number = require_finite(name, number)
        if number <= CHI_MIN_CYCLES:
            built_in = default_text(f"{float(FIELDS[name].default):g}")
            raise Refusal(f"{_what(name)} must be greater than {CHI_MIN_CYCLES}, the floor below which a probe is "
                          f"masked; got {number:g}{built_in}.", field=name)
        return number
    return require_positive(name, number)


def _bounds_name(text: str) -> str:
    """A name under the model's bounds folder, with forward slashes as the picker shows it. One that could
    leave that folder is refused here, and so is an empty or a ``.`` part, which the computer resolves and the list
    shows under no such name: the window's check that the file exists would accept either."""
    if "\\" in text or ":" in text or {"", ".", ".."} & set(text.split("/")):
        advice = " Write a sub-folder with a forward slash." if "\\" in text else ""
        raise Refusal(f"{_what('bounds')} must name a file in the model's bounds folder; got '{text}'.{advice}",
                      field="bounds")
    return text


def _across_lines(values: dict, set_on: dict, unread: set) -> list[PlanProblem]:
    """The two rules that need more than one line, so that one slip gives one problem. ``set_on`` holds the line of
    every setting under its own heading, ``values`` those accepted, and ``unread`` each name met on a line not
    accepted, wherever it stood. So a chi setting is not called lost when the fault is the switch's own line, a count
    not read is not compared, and a line told it needs the switch is not told of the slots as well, a comparison that
    means nothing without it. A chi line with a bad value and no switch has two things to correct, and says both."""
    found, where = [], {name: line for (heading, name), line in set_on.items() if heading == _OPTIONS}
    if "chi_mode" not in unread and values.get((_OPTIONS, "chi_mode")) is not True:
        found += [PlanProblem(where[name], _OPTIONS, name, f"{_what(name)} needs chi_mode = on in the same plan.")
                  for name in _ONLY_WITH_CHI_MODE if name in where]

    def count(name: str) -> int | None:
        """The plan's value, the built-in when the file holds none, None when a line of it was not accepted."""
        return None if name in unread else values.get((_OPTIONS, name), int(FIELDS[name].default))

    def said(name: str) -> str:
        """A number as the sentence shows it: one the file does not hold says so, or a person looks for a line."""
        return f"{count(name)}" if name in where else f"{count(name)} (the built-in)"

    probes, slots = count("chi_n_freqs"), count("chi_k_pad")
    if not found and probes is not None and slots is not None and probes > slots:
        found.append(PlanProblem(where.get("chi_n_freqs", where.get("chi_k_pad")), _OPTIONS, "chi_n_freqs",
                                 f"{_what('chi_n_freqs')} must be at most the number of slots, {said('chi_k_pad')}; "
                                 f"got {said('chi_n_freqs')}."))
    return found
