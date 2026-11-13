"""Each check's form: the rows its page shows under its result, as a table keyed by the check's key; one builder that
makes a check's form from its rows, with the name and the note of the record its run writes; the reading of a form's
rows through the rules the check's stage applies again; what a form hands its stage, by the stage's own names; and the
run a press of a check starts, as its page describes it to the run's controller.

A check whose rows are not here yet cannot be run from the window, and its button says so. Nothing on a check's form is
remembered between launches: every row opens at its usual value. A row is held to its stage's own rule, called and never
copied: the line under a field says it in the window's words, from the row's label, and a press that reaches the stage
with it is refused in the stage's own sentence.
"""
import re
from collections.abc import Mapping
from dataclasses import dataclass

from PySide6.QtWidgets import QLineEdit

from core import config
from core.config import CELL_PATH
from core.refusals import Refusal, require_at_least, require_between, require_positive
from core.rng import SEED_MAX, require_seed

from ... import check_readings, fields
from ...check_catalogue import entry
from ...check_readings import before_you_start  # noqa: F401 -- "Before you start" of a check, words with no widget
from ...run.model import SIMULATION_ROW, RunSpec, Said, Stage
from ...texts import number_text
from ...widgets.artifact_picker import ArtifactPicker
from ...widgets.form import Form, NameAndNote, TickedField
from ...widgets.labeled_inputs import FloatField, IntField
from .base import TRAINING_STAGES
from .help_text import HELP

#: What greys the button of a check whose form is not built yet.
NOT_BUILT_TEXT = "This check cannot be run from the window yet."
#: What a check's list of cell files shows while it holds none, and the fact that says so.
CHOOSE_CELL = "Choose a cell file"
NO_CELL_TEXT = "Choose a cell file."
NO_CELL_BLOCKS = "no cell file is chosen"
#: The line under a recording length left blank, which the two identifiability checks must be given.
TYPE_A_LENGTH_TEXT = "Type a recording length."
#: The tick of a fixed number of driven recordings that stands for none, and the line under a number of them outside
#: the posterior's slots.
MIXED_WORDS = "Mixed, as in training"
SLOTS_TEXT = "{} must be a whole number from 1 to {}."
#: The line under a list of numbers that holds something else, and under one that holds a number at or below 0.
NUMBERS_TEXT = "{} must be numbers separated by commas or spaces."
EACH_ABOVE_TEXT = "{} must each be greater than 0."
#: The line under a seed: blank, not a whole number, outside the one rule of a seed.
SEED_BLANK_TEXT = "{} is blank."
SEED_NOT_WHOLE_TEXT = "{} must be a whole number; got {!r}."
SEED_RULE_TEXT = "{} must be a whole number from 0 to {}."
#: The lines under a whole number and a number that break their rule.
AT_LEAST_TEXT = "{} must be at least {}."
ABOVE_TEXT = "{} must be greater than 0."
#: Where the window shows a check's runs, and the stage of a check that says no stages of its own.
PLACE = ("inference", "checks")
MEASURING = "Measuring"
#: The calibration's stages: its test datasets simulated, counted by the progress row that counts a simulation; the
#: rank test, from the line that ends the simulation; the joint coverage test; and the saving, from its last reading.
CALIBRATION_STAGES = (Stage("Simulating the test datasets", counts=SIMULATION_ROW),
                      Stage("Rank test", begins=("patho_total",)),
                      Stage("Joint coverage test", begins=("sbc_uniformity",)),
                      Stage("Saving", begins=("tarp_atc",)))
#: What the repeated calibration says while it ranks a repeat, which reports no progress and takes no stop until it
#: ends: from the line that ends a repeat's simulation to the line that ends the repeat.
RANKING_WORDS = ("Ranking repeat {done} of {of}: no progress is reported during this part, and Stop takes effect when "
                 "it ends")
REPEATED_STAGES = (Stage(MEASURING, said=Said("patho_total", "sbc_repeat", RANKING_WORDS, "repeats")),)
#: The timing of a short fit's stages, each begun at the lines a training's begins at: the rotation, the simulation
#: counted by the row that counts a training's, and the fit, counted by its epochs.
TIMING_STAGES = (Stage("Rotation", TRAINING_STAGES[0].begins),
                 Stage("Simulation", TRAINING_STAGES[1].begins, TRAINING_STAGES[1].counts),
                 Stage("Fit", TRAINING_STAGES[2].begins, TRAINING_STAGES[2].counts))
#: The kinds of row a check's form holds, and those whose value is no size of the run: a cell file is loaded on the
#: check's own configuration, and a seed changes no cost.
KINDS = ("whole", "number", "lengths", "cell", "seed", "blank_seed", "epochs", "ticked_count", "network")
#: "epochs" is a whole number of epochs the stage is sent one less of; "network", the networks a check fits, only shown.
NOT_SIZES = ("cell", "seed", "blank_seed")
#: The usual sizes of the checks below, each as its stage takes it when none is handed: the identifiability check's
#: weakest directions and largest contributions; what the observations can pin down, its simulations for each
#: comparison and for the noise level, its step and the least of the second, which the stage refuses below 10; the
#: calibration repeated, its repeats; the drive study, its recording length in seconds, its simulations for each
#: strength and its drive frequency, as a multiple of the cell's own; identifiability across the prior, its points.
WEAKEST_DIRECTIONS = 3
LARGEST_CONTRIBUTIONS = 4
COMPARISON_SIMULATIONS = 32
NOISE_SIMULATIONS = 128
NOISE_LEAST = 10
STEP_SHARE = 0.02
REPEATS = 10
DRIVE_LENGTH_S = 5.0
DRIVE_SIMULATIONS = 16
DRIVE_DETUNE = 1.4
POINTS_ACROSS = 6
#: The timing of a short fit, at the retraining runbook's sizes for timing an epoch before a full training: its batches
#: and its epochs; and the least number of epochs, one more than the least limit the stage takes, since the limit it is
#: sent is one less than the epochs it trains.
TIMING_BATCHES = 20
TIMING_EPOCHS = 10
LEAST_EPOCHS = 2
EPOCHS_LABEL = "Epochs"
EPOCHS_HELP = "At least 2 epochs: the limit sent is one less."
#: The network the training stage builds when it is handed none, read from its settings, which the timing times after
#: the Training form's; and the two rows that are a check's network, which its sizes hold as one.
BUILT_IN_NETWORK = (config.NSF_HIDDEN_FEATURES, config.NSF_NUM_TRANSFORMS)
NETWORK_ROWS = ("hidden_features", "num_transforms")
#: What the timing's row that is only shown says of the networks it times, and of a network as it is said.
NETWORKS_TEXT = "Networks: {}, as step 3's form holds them, then the built-in {}."
NETWORK_ONE_TEXT = "Network: {}, as step 3's form holds it, which is the built-in one."
NETWORK_UNREAD_TEXT = "Networks: as step 3's form holds them, which cannot be used now, then the built-in {}."
NETWORK_WORDS = "{} by {}"
#: What separates the numbers of a list typed in one field.
_BETWEEN = re.compile(r"[,\s]+")


@dataclass(frozen=True)
class FormRow:
    """One row of a check's form: the ``setting`` it hands the check's stage, under the key the stage refuses it by,
    labelled as the window's table of controls names that key unless ``label`` is handed; its ``kind`` of field (one of
    ``KINDS``), the ``usual`` value it opens at, whether it is ``folded`` among the rarely changed, its ``unit``, the
    ``least`` value its rule takes, its ``help_text``, the stage's ``rule`` it is held to ("at_least", "positive",
    "seed", "lengths", "slots"; "" for a cell file), the stage's ``argument`` it is handed as, where that name is not the
    key's, and for a seed the stage adds to for each of its runs, ``adds``, the row whose count those runs are."""
    setting: str
    label: str | None
    kind: str
    usual: object
    folded: bool = False
    unit: str = ""
    least: int = 1
    help_text: str = ""
    rule: str = ""
    argument: str = ""
    adds: str = ""


#: The rarely changed rows of the two identifiability checks that simulate, which they share.
_SIMULATED = (FormRow("m", None, "whole", COMPARISON_SIMULATIONS, folded=True, rule="at_least"),
              FormRow("m_noise", None, "whole", NOISE_SIMULATIONS, folded=True, least=NOISE_LEAST, rule="at_least"),
              FormRow("rel", None, "number", STEP_SHARE, folded=True, rule="positive"),
              FormRow("seed", None, "seed", 0, folded=True, least=0, rule="seed"))
#: Each check's rows, by its key, each opening at the size its stage takes as usual; the audit at the least size that
#: counts, the retraining runbook's, not the stage's.
FORMS: dict[str, tuple[FormRow, ...]] = {
    "band": (FormRow("cell", None, "cell", None),
             FormRow("probe_lengths", None, "lengths", None, folded=True, unit="s", rule="lengths",
                     argument="lengths"),
             FormRow("band_repeats", None, "whole", check_readings.BAND_LEAST_REPEATS, folded=True, least=2,
                     rule="at_least", argument="repeats"),
             FormRow("probe_seed", None, "seed", 0, folded=True, least=0, rule="seed", argument="seed")),
    "audit": (FormRow("mask_num_runs", None, "whole", check_readings.AUDIT_LEAST_BATCHES, rule="at_least",
                      argument="num_runs"),
              FormRow("mask_run_size", None, "whole", check_readings.AUDIT_LEAST_PER_BATCH, folded=True,
                      rule="at_least", argument="run_size"),
              FormRow("probe_seed", None, "seed", 0, folded=True, least=0, rule="seed", argument="seed")),
    "calibration": (FormRow("n_cal", None, "whole", config.SBC_N_CAL, help_text=HELP["cal_n"], rule="at_least"),
                    FormRow("cal_n_scales", None, "whole", config.CAL_N_SCALES, help_text=HELP["cal_scales"],
                            rule="at_least"),
                    FormRow("num_posterior_samples", None, "whole", check_readings.CALIBRATION_LEAST_SAMPLES,
                            folded=True, rule="at_least"),
                    FormRow("seed", None, "blank_seed", None, folded=True, least=0, rule="seed")),
    "input_use": (FormRow("rows", None, "whole", check_readings.INPUT_USE_LEAST_ROWS, folded=True, rule="at_least"),
                  FormRow("n_sweep", None, "whole", check_readings.INPUT_USE_LEAST_SWEEP, folded=True, least=2,
                          rule="at_least")),
    "identifiability": (FormRow("n_worst", None, "whole", WEAKEST_DIRECTIONS, folded=True, rule="at_least"),
                        FormRow("top_n", None, "whole", LARGEST_CONTRIBUTIONS, folded=True, rule="at_least")),
    "pin_down": (FormRow("cell", None, "cell", None),
                 FormRow("t_obs", None, "number", None, unit="s", rule="positive", argument="t_obs_s"),
                 *_SIMULATED),
    "fixed_number": (FormRow("chi_k_fixed", None, "ticked_count", None, rule="slots"),
                     FormRow("repeats", None, "whole", REPEATS, rule="at_least"),
                     FormRow("n_cal", None, "whole", config.SBC_N_CAL, unit="per repeat", rule="at_least"),
                     FormRow("cal_n_scales", None, "whole", config.CAL_N_SCALES, folded=True, rule="at_least"),
                     FormRow("num_posterior_samples", None, "whole", check_readings.CALIBRATION_LEAST_SAMPLES,
                             folded=True, rule="at_least"),
                     FormRow("seed", None, "seed", 0, folded=True, least=0, rule="seed", adds="repeats")),
    "drive_study": (FormRow("cell", None, "cell", None),
                    FormRow("drive_t_obs", None, "number", DRIVE_LENGTH_S, unit="s", rule="positive",
                            argument="t_obs_s"),
                    FormRow("drive_repeats", None, "whole", DRIVE_SIMULATIONS, folded=True, least=2, rule="at_least",
                            argument="repeats"),
                    FormRow("drive_detune", None, "number", DRIVE_DETUNE, folded=True, rule="positive",
                            argument="detune"),
                    FormRow("drive_strengths", None, "lengths", None, folded=True, rule="lengths",
                            argument="strengths"),
                    FormRow("probe_seed", None, "seed", 0, folded=True, least=0, rule="seed", argument="seed")),
    "across_prior": (FormRow("cell", None, "cell", None),
                     FormRow("t_obs", None, "number", None, unit="s", rule="positive", argument="t_obs_s"),
                     FormRow("n_points", None, "whole", POINTS_ACROSS, rule="at_least"),
                     *_SIMULATED),
    "timing": (FormRow("num_runs", None, "whole", TIMING_BATCHES, rule="at_least"),
               FormRow("epochs", EPOCHS_LABEL, "epochs", TIMING_EPOCHS, least=LEAST_EPOCHS, help_text=EPOCHS_HELP,
                       rule="epochs"),
               FormRow("network", None, "network", None, folded=True)),
}


class _Numbers(QLineEdit):
    """Numbers typed in one field, commas or spaces between them: read as None when blank, as the numbers when each
    reads as one, else as the text it holds, which no rule takes: separators with no number among them too."""

    def value_or_none(self):
        text = self.text().strip()
        if not text:
            return None
        try:
            numbers = [float(part) for part in _BETWEEN.split(text) if part]
        except ValueError:
            return text
        return numbers or text


def _field(row: FormRow, form: Form):
    """The field of ``row``, made for ``form``: a whole number grouped, a seed copied as typed, a number, numbers in one
    field, or the list of the cell files of a model's folder, which keeps its selection and opens on none."""
    if row.kind in ("whole", "epochs"):
        return IntField(row.usual, grouped=True)
    if row.kind in ("seed", "blank_seed"):
        return IntField(row.usual)
    if row.kind == "number":
        return FloatField(row.usual)
    if row.kind == "lengths":
        return _Numbers()
    if row.kind == "ticked_count":
        return TickedField(IntField(row.usual, grouped=True), MIXED_WORDS, None)
    if row.kind == "cell":
        form.cell_picker = ArtifactPicker(CELL_PATH / "nadrowski", parent=form, keeps_selection=True,
                                          hint=CHOOSE_CELL)
        return form.cell_picker.combo
    raise ValueError(f"no field of the kind {row.kind!r}")


def build_form(key: str) -> Form:
    """The form of the check ``key``, its rows at their usual values, and for a check that writes a record the name and
    the note of that record (``form.names``; None for none); a check with no rows has an empty form. The page that
    holds it shows it, or keeps an empty one out of sight, once it stands inside it: a form with no parent made visible
    would be a window of its own."""
    form = Form()
    form.cell_picker, form.names = None, None
    rows = FORMS.get(key, ())
    for row in rows:
        if row.kind == "network":            # only shown: its words are the page's, at each drawing
            form.add_text(row.setting, "", "", folded=row.folded)
            continue
        form.add(row.setting, _field(row, form), row.unit, usual=row.usual, folded=row.folded,
                 help_text=row.help_text, label=row.label)
    if rows and entry(key).kind:
        form.names = NameAndNote(entry(key).kind, required=False)
        form.add_name(form.names)
    return form


def cell_picker(form: Form) -> "ArtifactPicker | None":
    """The list of cell files of a check's form, or None for a form with no cell file."""
    return getattr(form, "cell_picker", None)


# ── the rules ──
def seed_problem(text: str, setting: str, *, blank: bool, added: int = 0) -> str:
    """The sentence under the field of the seed ``setting`` that holds ``text``: that it is blank (nothing, with
    ``blank``: a seed left blank is drawn at the press), that it holds no whole number, or that it breaks the one rule of
    a seed, which the engine's own ``require_seed`` decides, for the seed and, with ``added``, for the largest the stage
    makes of it by adding (the bound said is then what that leaves); "" when it passes."""
    label, words = fields.label(setting), text.strip().replace(",", "")
    if not words:
        return "" if blank else SEED_BLANK_TEXT.format(label)
    try:
        value = int(words)
    except ValueError:
        return SEED_NOT_WHOLE_TEXT.format(label, text.strip())
    try:
        require_seed(value, key=setting)
        require_seed(value + added, key=setting)
    except Refusal:
        return SEED_RULE_TEXT.format(label, SEED_MAX - added)
    return ""


def _rule(one: FormRow, value, *, slots=None, added: int = 0):
    """The stage's own rule for the row ``one`` applied to ``value``: the value it takes, or its refusal; a seed the
    stage adds to, ``added`` at most, held so that its largest passes too."""
    if one.rule == "at_least":
        return require_at_least(one.setting, value, one.least)
    if one.rule == "epochs":                 # the stage is sent a limit one less, held to the stage's own rule
        return require_at_least("max_num_epochs", None if value is None else value - 1, one.least - 1) + 1
    if one.rule == "positive":
        return require_positive(one.setting, value)
    if one.rule == "seed":
        taken = require_seed(value, key=one.setting)
        if added:
            require_seed(value + added, key=one.setting)
        return taken
    if one.rule == "lengths":
        return [require_positive(one.setting, part) for part in value]
    if one.rule == "slots":
        return value if value is None or slots is None else int(require_between(one.setting, value, 1, slots))
    return value


def _rule_line(one: FormRow, label: str, *, slots=None) -> str:
    """The rule of ``one`` as the line under its field says it, from its label."""
    if one.rule in ("at_least", "epochs"):
        return AT_LEAST_TEXT.format(label, one.least)
    if one.rule == "lengths":
        return EACH_ABOVE_TEXT.format(label)
    if one.rule == "slots":
        return SLOTS_TEXT.format(label, slots)
    return ABOVE_TEXT.format(label)


def _press_words(one: FormRow, value, line: str, *, slots=None, added: int = 0) -> str:
    """What a press refuses the row ``one`` with: the stage's own sentence for ``value``, blank among them; the line
    under the field where the field holds what no rule of the stage can be handed."""
    try:
        _rule(one, value, slots=slots, added=added)
    except Refusal as refused:
        return refused.message
    except (TypeError, ValueError):
        pass
    return line


def _problem(one: FormRow, row, *, slots=None, added: int = 0) -> "tuple[str, str] | None":
    """The rule the row ``row`` of ``one`` breaks now: the line under its field and what a press says; None for none."""
    label, value = row.label_text(), row.value()
    if one.kind in ("seed", "blank_seed"):
        line = seed_problem(row.widget.text(), one.setting, blank=one.kind == "blank_seed", added=added)
        return None if not line else (line, _press_words(one, value, line, slots=slots, added=added))
    if one.kind == "lengths":
        if value is None:
            return None
        if isinstance(value, str):
            line = NUMBERS_TEXT.format(label)
            return line, line
    else:
        said = row.reading()
        if said is not None:
            typed = getattr(row.widget, "field", row.widget).text().strip()
            if not typed and one.setting == "t_obs":
                said = TYPE_A_LENGTH_TEXT
            return said, said if typed else _press_words(one, None, said, slots=slots)
        if value is None:
            return None
    try:
        _rule(one, value, slots=slots)
    except Refusal as refused:
        return _rule_line(one, label, slots=slots), refused.message
    return None


def problems(key: str, form: Form, *, slots=None) -> dict:
    """Every rule the form of the check ``key`` breaks now, by field key in the form's order: the line under the field,
    in the window's words from its row's label, and what a press is refused with, the stage's own sentence. A cell file
    breaks no rule here: its absence is a fact of its own, and its stage reads the file at the press."""
    found = {}
    for one in FORMS.get(key, ()):
        row = form.rows[one.setting]
        if one.kind == "cell" or not row.is_shown():
            continue
        problem = _problem(one, row, slots=slots, added=_added(one, form))
        if problem is not None:
            found[one.setting] = problem
    return found


def _added(one: FormRow, form: Form) -> int:
    """The most the stage adds to the seed of ``one`` for its runs: one less than the count its ``adds`` row holds, as
    the repeated calibration seeds its repeat r at the seed and r; 0 for a seed it uses as it is, or while that count is
    not a whole number from 1."""
    count = form.rows[one.adds].value() if one.adds else None
    return count - 1 if isinstance(count, int) and not isinstance(count, bool) and count >= 1 else 0


def _handed(one: FormRow, value):
    """``value`` as the stage is handed it: a whole number as one, a number as a real, numbers as a list of reals."""
    if value is None:
        return None
    if one.kind in ("whole", "seed", "blank_seed", "ticked_count", "epochs"):
        return int(value)
    if one.kind == "number":
        return float(value)
    if one.kind == "lengths":
        return [float(part) for part in value]
    return value


def call_arguments(key: str, form: Form, *, slots=None) -> dict:
    """What the form of the check ``key`` hands its stage, each by the stage's own name for it, read through the
    stage's own rules; the first rule broken is refused, and nothing is handed. The cell file is no argument: it is
    loaded on the check's own configuration."""
    for setting, (_line, said) in problems(key, form, slots=slots).items():
        raise Refusal(said, field=setting)
    return {one.argument or one.setting: _handed(one, form.rows[one.setting].value())
            for one in FORMS.get(key, ()) if one.kind not in ("cell", "network")}


def _as_sizes(key: str, read) -> dict:
    """The sizes of the check ``key``, each row's read by ``read``: every row but a cell file, a seed and what is only
    shown, the two rows of a network held as one, ``network``, the width and the depth."""
    sizes = {}
    for one in FORMS.get(key, ()):
        if one.kind in NOT_SIZES or one.kind == "network":
            continue
        if one.setting in NETWORK_ROWS:
            sizes["network"] = [read(name) for name in NETWORK_ROWS]
        else:
            sizes[one.setting] = read(one)
    return sizes


def sizes_of(key: str, form: Form) -> dict:
    """The sizes the form of the check ``key`` holds, by field key, as they read: every row but a cell file, a seed and
    what is only shown, a network's two rows as one, ``network``."""
    return _as_sizes(key, lambda one: form.rows[getattr(one, "setting", one)].value())


def sizes_from_call(key: str, kwargs: Mapping) -> dict:
    """The sizes a call of the stage of the check ``key`` was handed, by field key, as ``sizes_of`` names them."""
    def handed(one):
        name = one if isinstance(one, str) else one.argument or one.setting
        return kwargs.get(name)
    return _as_sizes(key, handed)


def network_words(network) -> str:
    """A network's width and depth as the window says them: "256 by 10"."""
    return NETWORK_WORDS.format(*(number_text(part) for part in network))


def timing_networks(trained) -> tuple:
    """The networks the timing of a short fit times, in their order: ``trained``, as the Training form holds it, then
    the built-in one; the one alone when the two are the same."""
    trained = tuple(trained)
    return (trained,) if trained == BUILT_IN_NETWORK else (trained, BUILT_IN_NETWORK)


def networks_line(trained) -> str:
    """What the timing's row of its networks says: the Training form's ``trained`` (None while it holds no network that
    can be used) and the built-in one."""
    if trained is None:
        return NETWORK_UNREAD_TEXT.format(network_words(BUILT_IN_NETWORK))
    if tuple(trained) == BUILT_IN_NETWORK:
        return NETWORK_ONE_TEXT.format(network_words(trained))
    return NETWORKS_TEXT.format(network_words(trained), network_words(BUILT_IN_NETWORK))


# ── the run ──
def check_spec(key: str, *, sizes: dict, facts: Mapping = {}, plan: dict | None = None) -> RunSpec:
    """The run of the check ``key`` at ``sizes``, as its page describes it: a check, at Checks, named for the check,
    stopped by its stop button, in its stages (the calibration's four; any other one, "Measuring"), writing its one
    kind of record, its run keeping the ``plan`` in use and its key and sizes, and no settings, which its record holds;
    with ``facts``, what is known of it before it says it."""
    one = entry(key)
    stages = (CALIBRATION_STAGES if key == "calibration" else REPEATED_STAGES if key == "fixed_number"
              else TIMING_STAGES if key == "timing" else (Stage(MEASURING),))
    # A check that keeps nothing says so to its run, which then says nothing of a training not kept; the timing's fit
    # ends at the limit its epochs set, or by the training's own patience before it.
    kept_nothing = {"keeps_nothing": True} if key in check_readings.KEEPS_NOTHING else {}
    epochs = sizes.get("epochs") if key == "timing" else None
    timed = isinstance(epochs, int) and not isinstance(epochs, bool)
    return RunSpec("Check", PLACE, name=one.name, stop_button=one.stop_words, stages=stages,
                   kinds=(one.kind,) if one.kind else (), facts={**facts, **kept_nothing}, plan=plan,
                   check={"key": key, "sizes": dict(sizes)},
                   patience=config.TRAINING_STOP_AFTER_EPOCHS if timed else None,
                   epoch_limit=epochs - 1 if timed else None)
