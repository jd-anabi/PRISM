"""Step 5, Inference: the posterior in use run on one observation, a cell simulated now or recordings of one's own, and
each parameter's estimate with its uncertainty said once the run is over.

The step holds the observation an inference made, and holds it only when the run returns it. Two modes, chosen at the
top of the form: "A simulated cell", a cell file of the model in use or its values typed by hand, and "Your own
recordings", whose rows are those the posterior in use reads: a spontaneous recording only, or one driven recording and
its drive, or a frequency response, a table of driven recordings, each a file and the frequency it was driven at, and
the drive's amplitude. With no posterior in use the second mode says so in place of a form. The two modes never mix:
the rows of the other mode stand off the page and are neither judged nor passed.

The recording length and the drive amplitude open blank at every launch. Nothing is remembered as it is typed.

The page reads the configuration in use as it draws, never as another page hands it one: a change of what is in use
marks the page to read again, at once while it is shown, else at its next showing. Its reading lists the cell files of
the model in use, checks the cell chosen against the configuration in use, and finds the label of the observation a
narrowed posterior was narrowed around; nothing of it while a run is live.
"""
import inspect
import math

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QCheckBox, QHBoxLayout, QLabel, QPushButton, QRadioButton, QVBoxLayout, QWidget

from core import cli, config, orchestrator
from core.artifacts import default_store
from core.config import CELL_PATH
from core.Helpers import file_manager, labels
from core.refusals import Refusal, require_at_least, require_file, require_finite, require_positive
from core.SBI.observations import RecordingSet

from ... import design, record_words, settings
from ...fields import label
from ...remembered import DRIVEN_FILE_KEY, DRIVEN_FREQUENCY_KEY, DRIVEN_SIZE_KEY
from ...run.model import RunSpec
from ...step_facts import (CELL_PROBLEM_TEXT, INFER_BLANKS, INFER_REMOVE, NO_OBSERVATION_WORDS, NOT_MEASURED_TEXT,
                           PLAN_NEEDS_LENGTH_TEXT, PLAN_NEEDS_SPONTANEOUS_TEXT, PLANNED_TEXT, Fact, driven_reading,
                           inference_facts, inference_uses, observation_in_use)
from ...widgets.artifact_picker import ArtifactPicker
from ...widgets.form import NameAndNote, OtherName, RowTable, _dress
from ...widgets.labeled_inputs import FloatField, IntField, PathField
from ...widgets.param_grid import ValuesGrid
from .help_text import HELP
from .step_page import StepPage

TITLE = "Inference"
SENTENCE = "Runs the posterior on one observation and reports each parameter's estimate with its uncertainty."
#: The row that chooses the mode, its two answers by the mode each names, and the label of the tick that types a cell.
RUN_ON = "Run on"
MODES = (("simulated", "A simulated cell"), ("recordings", "Your own recordings"))
TYPED_LABEL = "Edit the values by hand"
CHOOSE_CELL = "Choose a cell file"
#: What the second mode says of the form it shows, by the kind of observation the posterior in use reads; and what it
#: says in place of a form while no posterior is in use.
READS_TEXT = {"chi": "This posterior reads a frequency response.",
              "forced": "This posterior reads one driven recording.",
              "spontaneous": "This posterior reads a spontaneous recording only."}
NO_FORM_TEXT = "The form depends on the posterior. Load or train one in step 3."
#: Under the recording length: the lengths the posterior in use was trained on.
TRAINED_TEXT = "The posterior was trained on lengths from {} to {} s."
ADD_WORDS = "Add a driven recording"
PLAN_WORDS = "Plan the driven recordings…"
RUN_WORDS = "Run inference"
STOP_WORDS = "Stop the inference…"
#: The labels of the two names, the inference's and its observation's.
INFERENCE_NAME = "Name of the inference"
OBSERVATION_NAME = "Name of the observation"
SECONDS, NEWTONS, HERTZ = "s", "N", "Hz"
#: The words beside the main button while a field breaks its rule.
SETTING_BLOCKS = "a setting cannot be used"
VALUES_BLOCKS = "the values cannot be used"
#: What a drive's parameter the table does not name says when it is left blank, and beside the main button.
DRIVE_BLANK_TEXT = "Type a value for {}."
DRIVE_BLOCKS = "the drive is not complete"
#: What the line under a driven recording says of a file or a frequency that cannot be used yet.
NO_FILE_TEXT = "Choose the file of this driven recording."
NO_FREQUENCY_TEXT = "Type the frequency this recording was driven at."
BAD_FREQUENCY_TEXT = "The frequency must be greater than 0 Hz."
#: What "Before you start" says of a driven recording the engine would refuse, its reason under its row.
REFUSED_ROW_TEXT = "A driven recording cannot be used: its reason stands under it."
REFUSED_ROW_BLOCKS = "a driven recording cannot be used"
#: The least width of the table of driven recordings, in the look's pixels at the ordinary text size: a path of the
#: usual length on its line, with its remove button beside it.
DRIVEN_LEAST = 330
#: The drive's own rows by the forcing name each answers, with the field key a refusal names it by; a forcing name the
#: table does not name is a row of its own, labelled from the name and its unit.
_DRIVE_FIELD = {"amp": "drive_amplitude", "freq": "drive_frequency", "phase": "drive_phase"}
#: The rows each mode shows: the simulated cell's; the recordings', each with the kinds of observation that read it.
_SIMULATED_ROWS = ("cell", "cell_typed")
_RECORDING_ROWS = {"recording_spont": ("chi", "forced", "spontaneous"), "recording_forced": ("forced",),
                   "recording_probe": ("chi",), "chi_f0_si": ("chi",)}
#: The number rows a run reads and the engine's rule for each: the floor of require_at_least, or None for
#: require_positive, or "finite" for require_finite.
_RULES = {"t_obs": None, "chi_f0_si": None, "n_samples": 1, "drive_amplitude": None, "drive_frequency": None,
          "drive_phase": "finite"}
# Where the window shows this step's runs: Parameter inference, its step Inference.
_PLACE = ("inference", "infer")


def _draws() -> int:
    """The posterior draws an inference makes unless told otherwise: the stage's own number, read off its signature."""
    return inspect.signature(orchestrator.infer_and_visualize).parameters["n_samples"].default


def _drive_label(name: str) -> str:
    """The row label of the drive's parameter ``name``: the table's for the three it names, else built from the name and
    its unit as the table builds those three."""
    key = _DRIVE_FIELD.get(name)
    if key is not None:
        return label(key)
    return labels.gui_forcing_label(name, config.FORCING_DISPLAY_UNITS.get(name, ""))


def posterior_kind(posterior, cfg) -> "str | None":
    """The kind of observation the posterior in use reads: its record's, else the configuration's; None while there is
    no posterior in use."""
    if posterior is None:
        return None
    body = getattr(getattr(posterior, "manifest", None), "body", None)
    mode = body.get("mode") if isinstance(body, dict) else None
    return mode if mode in READS_TEXT else getattr(cfg, "observation_mode", None)


def posterior_slots(posterior, cfg) -> "int | None":
    """The driven recordings the posterior in use takes at most: its record's slots, else the configuration's."""
    body = getattr(getattr(posterior, "manifest", None), "body", None)
    conditioning = body.get("conditioning") if isinstance(body, dict) else None
    held = conditioning.get("chi_k_pad") if isinstance(conditioning, dict) else None
    if held is None:
        held = getattr(cfg, "chi_k_pad", None)
    try:
        return int(held)
    except (TypeError, ValueError):
        return None


def trained_lengths(posterior, cfg) -> tuple:
    """The shortest and the longest recording the posterior in use was trained on, in seconds: from its record, in the
    cell's units of time, else the usual range."""
    made_for = getattr(getattr(posterior, "manifest", None), "config", None)
    made_for = made_for if isinstance(made_for, dict) else {}
    try:
        per_second = float(cfg.get_unit_conversion_factor("s"))
        low, high = float(made_for["t_min_exp"]) / per_second, float(made_for["t_max_exp"]) / per_second
        if math.isfinite(low) and math.isfinite(high) and 0 < low < high:
            return low, high
    except Exception:                        # noqa: BLE001 -- what a record or a configuration cannot say is the usual
        pass
    return config.T_MIN_EXP_S, config.T_MAX_EXP_S


def _rule_line(text: str, value, rule) -> "str | None":
    """The line under a number row whose value breaks the engine's rule, in the row's own words; None when it passes."""
    try:
        if rule == "finite":
            require_finite("drive_phase", value)
        elif rule is None:
            require_positive("t_obs", value)
        else:
            require_at_least("n_samples", value, rule)
        return None
    except Refusal:
        if rule == "finite":
            return f"{text} must be a finite number."
        return f"{text} must be greater than 0." if rule is None else f"{text} must be at least {rule}."


class _Modes(QWidget):
    """The two modes as two buttons side by side, both names in sight: ``value_or_none`` is the mode chosen, by its key,
    and ``changed`` says another was chosen. The Tab key stops at the pair once, at the first, and the arrows move
    between them."""

    changed = Signal()

    def __init__(self) -> None:
        super().__init__()
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(design.scaled(design.SIZE["row_gap"]))
        self.buttons = {}
        for key, words in MODES:
            button = QRadioButton(words)
            button.toggled.connect(self._toggled)
            row.addWidget(button)
            self.buttons[key] = button
        row.addStretch(1)
        self.buttons[MODES[0][0]].setChecked(True)

    def _toggled(self, on: bool) -> None:
        if on:
            self.changed.emit()

    def value_or_none(self) -> str:
        return next(key for key, button in self.buttons.items() if button.isChecked())

    def set_value(self, key) -> None:
        if key in self.buttons:
            self.buttons[key].setChecked(True)

    setText = set_value

    def text(self) -> str:
        return self.buttons[self.value_or_none()].text()

    def stops(self) -> list:
        """The pair is one stop of the Tab key, its first button: the toolkit's way with two buttons of one choice."""
        return [self.buttons[MODES[0][0]]]


class _Driven(QWidget):
    """One driven recording, a file and the frequency it was driven at, as one widget: a row removed takes both with it,
    so a file never stands beside another's frequency. The file on its line, the frequency on the next."""

    changed = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.path, self.hz = PathField(), FloatField(None)
        _dress(self.hz)
        box = QVBoxLayout(self)
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(design.scaled(design.GAP["field_message"]))
        box.addWidget(self.path)
        line = QHBoxLayout()
        line.setContentsMargins(0, 0, 0, 0)
        line.setSpacing(design.scaled(design.SIZE["row_gap"]))
        line.addWidget(QLabel("at"))
        line.addWidget(self.hz)
        line.addWidget(QLabel(HERTZ))
        line.addStretch(1)
        box.addLayout(line)

    def value(self) -> tuple:
        """The file and the frequency, each None while it is blank: ``(path, Hz)``."""
        return self.path.value_or_none(), self.hz.value_or_none()

    def set_value(self, value) -> None:
        path, hz = value if isinstance(value, (tuple, list)) and len(value) == 2 else (None, None)
        self.path.set_value(path)
        self.hz.setText("" if hz is None else str(hz))

    def typed(self) -> bool:
        """Whether anything is typed in the row, its file or its frequency."""
        return bool(self.path.edit.text().strip() or self.hz.text().strip())

    def problem(self) -> str:
        """What stops the row from being run, in a sentence; "" for nothing. Whether it is long enough, or in the band,
        is the planner's to say."""
        if not self.path.edit.text().strip():
            return NO_FILE_TEXT
        hz = self.hz.value_or_none()
        if hz is None:
            return NO_FREQUENCY_TEXT
        try:
            require_positive("recording_probe", hz)
        except Refusal:
            return BAD_FREQUENCY_TEXT
        return ""

    def stops(self) -> list:
        return [*self.path.stops(), self.hz]


class InferPanel(StepPage):
    """Step 5. The form's rows are read by their field keys: ``infer_mode`` chooses the mode; ``cell`` lists the cell
    files of the model in use (``cell_picker``), with the values typed by hand folded under ``cell_typed``
    (``values_grid``); ``recording_probe`` is the table of driven recordings (``driven``). ``names`` holds the
    inference's name, the observation's and the note."""

    def __init__(self, screen, parent=None):
        super().__init__(screen, TITLE, SENTENCE, step="infer", parent=parent)
        self._held_for = None                # what was in use as the page last read: the configuration, the posterior
        self._cell_problems: list = []       # why the cell chosen does not fit the configuration, as the reading found
        self._cell_read = None               # the cell file and the configuration that reading was made for
        self._observation_label = ""         # the observation a narrowed posterior was narrowed around, by its label
        self._drives: dict = {}              # the drive's rows made so far, by forcing name: their form keys
        self._typed_from = None              # the cell file the values typed by hand were last filled from
        self._planned: dict = {}             # what the planner read of each driven recording: what it held, its line
        form = self.form
        self.modes = _Modes()
        form.add("infer_mode", self.modes, usual="simulated", label=RUN_ON)
        self.reads_line = form.add_sentence("", under="infer_mode")
        self.reads_line.setVisible(False)
        self.cell_picker = ArtifactPicker(CELL_PATH / "nadrowski", parent=self, keeps_selection=True, hint=CHOOSE_CELL)
        form.add("cell", self.cell_picker.combo, help_text=HELP["cell"])
        # The draws first in the fold, which stands in both modes; the values typed by hand after them, in the first.
        draws = _draws()
        form.add("n_samples", IntField(draws, grouped=True), usual=draws, folded=True, help_text=HELP["n_samples"])
        self.typed_tick = QCheckBox()
        self.values_grid = ValuesGrid(scrolls=False)
        typed = form.add("cell_typed", self.typed_tick, usual=False, folded=True, label=TYPED_LABEL)
        typed.add_under(self.values_grid)
        typed.hear(self.values_grid)
        self.values_grid.setVisible(False)
        form.add_path("recording_spont", PathField(), help_text=HELP["spont"])
        form.add_path("recording_forced", PathField(), help_text=HELP["forced"])
        for name in _DRIVE_FIELD:
            self._drive_row(name)
        self.driven = RowTable(_Driven, typed=lambda row: row.typed(), question=INFER_REMOVE, add_words=ADD_WORDS,
                               least_width=design.scaled(DRIVEN_LEAST))
        form.add_table("recording_probe", self.driven)
        self.plan_button = QPushButton(PLAN_WORDS)
        self.plan_button.clicked.connect(self._plan_pressed)
        self.plan_holder = QWidget()
        planning = QHBoxLayout(self.plan_holder)
        planning.setContentsMargins(0, 0, 0, 0)
        planning.addWidget(self.plan_button)
        planning.addStretch(1)
        form.rows["recording_probe"].add_under(self.plan_holder)
        self.plan_said = form.add_sentence("", under="recording_probe")
        self.plan_said.setVisible(False)
        form.add("chi_f0_si", FloatField(None), NEWTONS, help_text=HELP["chi_f0_si"])
        form.add("t_obs", FloatField(None), SECONDS, help_text=HELP["tobs"])
        self.trained_line = form.add_sentence("", under="t_obs")
        self._names = NameAndNote("inference", required=False, label=INFERENCE_NAME,
                                  other=OtherName("observation_name", "observation", OBSERVATION_NAME, True))
        self.use_names(self._names)
        self.restore_settings(settings.settings())
        self.modes.changed.connect(self._mode_changed)
        self.typed_tick.toggled.connect(self._typed_toggled)
        self.cell_picker.combo.currentIndexChanged.connect(self._cell_chosen)
        self._show_the_mode()
        self.refresh_local_gates()

    # ── the rows ──
    def _drive_row(self, name: str) -> str:
        """The drive's row of the forcing name ``name``, made the first time the name is seen, blank and off the page
        until the posterior in use reads a driven recording: its form key."""
        key = self._drives.get(name)
        if key is None:
            key = _DRIVE_FIELD.get(name) or f"drive_{name}"
            self.form.add(key, FloatField(None), help_text=HELP["forcing"],
                          label=None if name in _DRIVE_FIELD else _drive_label(name))
            self._drives[name] = key
            self.form.rows[key].set_shown(False)
        return key

    def mode(self) -> str:
        """The mode chosen: "simulated" or "recordings"."""
        return self.modes.value_or_none()

    def _held(self) -> tuple:
        """The configuration and the posterior in use, each None when none is held."""
        session = getattr(self._screen, "session", None)
        return getattr(session, "cfg", None), getattr(session, "posterior", None)

    def kind(self) -> "str | None":
        """The kind of observation the posterior in use reads; None while no posterior is in use."""
        cfg, posterior = self._held()
        return posterior_kind(posterior, cfg)

    def _forcing_names(self) -> tuple:
        """The drive's parameters of the configuration in use, by name, in its order."""
        names = getattr(self._held()[0], "force_params_dict", None)
        return tuple(names) if isinstance(names, dict) else ()

    def _show_the_mode(self) -> None:
        """The rows of the mode chosen on the page, those of the other mode off it; in the second, the rows the
        posterior in use reads, or with no posterior none and the sentence that says why."""
        simulated = self.mode() == "simulated"
        kind = self.kind()
        rows = self.form.rows
        for key in _SIMULATED_ROWS:
            rows[key].set_shown(simulated)
        self.values_grid.setVisible(simulated and self.typed_tick.isChecked())
        for key, kinds in _RECORDING_ROWS.items():
            rows[key].set_shown(not simulated and kind in kinds)
        forcing = set(self._forcing_names()) if kind == "forced" else set()
        for name in forcing:
            self._drive_row(name)
        for name, key in self._drives.items():
            rows[key].set_shown(not simulated and name in forcing)
        rows["t_obs"].set_shown(simulated or kind is not None)
        said = "" if simulated else READS_TEXT.get(kind, NO_FORM_TEXT)
        self.reads_line.set_text(said)
        self.reads_line.setVisible(bool(said))
        cfg, posterior = self._held()
        low, high = trained_lengths(posterior, cfg)
        self.trained_line.set_text(TRAINED_TEXT.format(f"{low:g}", f"{high:g}"))

    def _mode_changed(self) -> None:
        self._show_the_mode()
        self._a_stop_joined()

    def _typed_toggled(self, on: bool) -> None:
        """The cell's values typed by hand, under the tick: filled from the cell file chosen the first time, and from
        nothing when none is chosen."""
        if on and self._typed_from != self.cell_picker.selected_path():
            self._fill_the_grid()
        self.values_grid.setVisible(on and self.mode() == "simulated")
        self._a_stop_joined()

    def _fill_the_grid(self) -> None:
        """The values typed by hand filled from the cell file chosen; a file that cannot be read leaves them as they
        were."""
        path = self.cell_picker.selected_path()
        if not path:
            return
        try:
            self.values_grid.load(*file_manager.parse_values_file(path))
        except Exception:                    # noqa: BLE001 -- the file's trouble is said by the reading
            return
        self._typed_from = path
        self._a_stop_joined()

    def _cell_chosen(self) -> None:
        """Another cell file chosen: the values typed by hand follow it while they are shown."""
        if self.typed_tick.isChecked():
            self._fill_the_grid()

    # ── what a page answers ──
    def main_words(self) -> str:
        return RUN_WORDS

    def in_use_words(self) -> tuple:
        """The observation in use: its label, and where it came from and its length."""
        observation = getattr(getattr(self._screen, "session", None), "observation", None)
        body = getattr(getattr(observation, "manifest", None), "body", None)
        return observation_in_use(record_words.label_of(observation) if observation is not None else "", body)

    def _shown_rows(self) -> list:
        """The keys of the rows on the page, in the form's order."""
        return [key for key, row in self.form.rows.items() if row.is_shown()]

    def _field_lines(self) -> tuple:
        """Each number row on the page that breaks its rule, with its line under it, and the keys of the rows on the
        page left blank. A blank says nothing under its row: "Before you start" says what to fill."""
        rows, broken, blank = self.form.rows, {}, []
        for key in self._ruled():
            row = rows[key]
            said = row.reading()
            if said is None:
                said = _rule_line(row.label_text(), row.value(), self._rule_of(key))
            elif not row.widget.text().strip() and (key in INFER_BLANKS or key.startswith("drive_")):
                blank.append(key if key in INFER_BLANKS else (DRIVE_BLANK_TEXT.format(row.label_text()),
                                                              DRIVE_BLOCKS))
                continue
            if said is not None:
                broken[key] = said
        return broken, blank

    def _ruled(self) -> list:
        """The number rows on the page that a run reads, each held to a rule of the engine's, in the form's order."""
        return [key for key in self._shown_rows() if key in _RULES or key.startswith("drive_")]

    @staticmethod
    def _rule_of(key: str):
        """The engine's rule of the number row ``key``, as ``_RULES`` writes it: a drive's parameter the table does not
        name is any finite number."""
        return _RULES.get(key, "finite")

    def _driven_lines(self) -> tuple:
        """Each driven recording's line: what stops it from being run; else what the planner last read of it, while the
        row holds what it held then; else nothing. Whether any is incomplete, and whether any the engine would refuse."""
        incomplete = refused = False
        for row in self.driven.rows():
            problem = row.problem()
            if problem:
                incomplete = True
                self.driven.set_reading(row, "err", problem)
                continue
            planned = self._planned.get(id(row))
            if planned is not None and planned[0] == row.value():
                refused = refused or planned[1] == "err"
                self.driven.set_reading(row, planned[1], planned[2])
            else:
                self.driven.set_reading(row, "", "")
        return incomplete, refused

    def facts_from_form(self) -> tuple:
        """The facts of the step as the form, the session and the last reading stand: what it runs on, the cell's fit,
        the rows left blank and the rules broken, under their rows too, and the names. Reads nothing."""
        cfg, posterior = self._held()
        simulated = self.mode() == "simulated"
        broken, blank = self._field_lines()
        for key in self._ruled():
            self.form.rows[key].set_line("err" if key in broken else "", broken.get(key, ""))
        kind = self.kind()
        uses = ""
        if posterior is not None:
            uses = inference_uses(record_words.label_of(posterior), kind, posterior_slots(posterior, cfg))
        cell_problem, grid = "", []
        if simulated:
            if self.typed_tick.isChecked():
                grid = [Fact("err", said, VALUES_BLOCKS) for said in self.values_grid.problems()]
            elif not self.cell_picker.selected_path():
                blank.insert(0, "cell")
            elif self._cell_read == (self.cell_picker.selected_path(), cfg) and self._cell_problems:
                cell_problem = CELL_PROBLEM_TEXT.format("; ".join(self._cell_problems))
        elif kind is not None:
            rows = self.form.rows
            if not rows["recording_spont"].widget.edit.text().strip():
                blank.insert(0, "recording_spont")
            if kind == "forced" and not rows["recording_forced"].widget.edit.text().strip():
                blank.insert(1, "recording_forced")
            if kind == "chi":
                incomplete, refused = self._driven_lines()
                if not self.driven.rows():
                    blank.append("recording_probe")
                elif incomplete:
                    blank.append("driven_row")
                if refused:
                    blank.append((REFUSED_ROW_TEXT, REFUSED_ROW_BLOCKS))
        names = [fact for fact in (self._names.name_fact(), self._names.other_fact(), self._names.note_fact()) if fact]
        field_facts = [*grid, *(Fact("err", line, SETTING_BLOCKS) for line in broken.values())]
        chi = not simulated and kind == "chi"
        return inference_facts(mode=self.mode(), uses=uses, cell_problem=cell_problem,
                               slots=posterior_slots(posterior, cfg) if chi else None,
                               rows=len(self.driven.rows()) if chi else 0, blanks=blank, name_facts=names,
                               field_facts=field_facts)

    # ── the reading of the disk ──
    def read_disk(self) -> None:
        """The cell files of the model in use listed again, the choice kept while the folder holds it; the cell chosen
        checked against the configuration in use; and the label of the observation a narrowed posterior was narrowed
        around."""
        cfg, posterior = self._held()
        self._held_for = (cfg, posterior)
        draft = getattr(getattr(self._screen, "session", None), "draft", None)
        model = getattr(draft, "model", None) or getattr(cfg, "model", None)
        if isinstance(model, str) and model:
            folder = CELL_PATH / model.lower()
            if folder != self.cell_picker.base_path:
                self.cell_picker.repoint(folder, "")
            else:
                self.cell_picker.refresh()
        path = self.cell_picker.selected_path()
        self._cell_problems = []
        if cfg is not None and path:
            try:
                self._cell_problems = list(cli.validate_gt_file(cfg, path))
            except Exception as error:       # noqa: BLE001 -- a check that cannot be made is said, never raised
                self._cell_problems = [f"it cannot be checked: {type(error).__name__}: {error}"]
        self._cell_read = (path, cfg)
        self._observation_label = ""
        if getattr(getattr(posterior, "posterior", None), "x_obs_digest", None) is not None:
            parents = getattr(getattr(posterior, "manifest", None), "parents", None)
            ref = parents.get("observation") if isinstance(parents, dict) else None
            try:
                self._observation_label = (record_words.label_of(default_store().get("observation", ref)) if ref
                                           else NO_OBSERVATION_WORDS)
            except Exception:                # noqa: BLE001 -- a record no longer in Records is said so
                self._observation_label = NO_OBSERVATION_WORDS

    def refresh_local_gates(self) -> None:
        """What the step holds and what follows from it, the rows of the mode for the posterior in use; and a change of
        what is in use since the last reading marks the page to read again: at once while it is shown, else at its next
        showing."""
        self._show_the_mode()
        read = self._held_for
        if read is not None and any(now is not then for now, then in zip(self._held(), read)):
            self._held_for = None
            if self.isVisible():
                self.read_soon()
            else:
                self.stale = True
        super().refresh_local_gates()

    # ── the press ──
    def _plan_pressed(self) -> None:
        """"Plan the driven recordings…", on the window's thread: the cell's own frequency measured from the spontaneous
        recording, with the recording length typed, and said under the table with the band; only a frequency left blank
        filled from the band, never one typed, a zero among them; and under each row what the engine will do with it,
        from its own verdict. A blank length, or no spontaneous recording, is said under the table and plans nothing."""
        from core.SBI import chi
        cfg = self._held()[0]
        if cfg is None or self.kind() != "chi":
            return
        try:
            t_obs = require_positive("t_obs", self.form.rows["t_obs"].value())
        except Refusal:
            self._say_planned(PLAN_NEEDS_LENGTH_TEXT)
            return
        try:
            path = require_file("recording_spont", self.form.rows["recording_spont"].value(), "recording")
        except Refusal:
            self._say_planned(PLAN_NEEDS_SPONTANEOUS_TEXT)
            return
        try:
            trace = file_manager.load_experimental_data(path, dtype=cfg.hw.dtype)
            own = float(chi.peak_freq(trace.unsqueeze(0), cfg.dt_exp))
            low, high = chi.band_hz(cfg, own)
        except Exception as error:           # noqa: BLE001 -- a planner never breaks the page: what failed is said
            self._say_planned(NOT_MEASURED_TEXT.format(f"{type(error).__name__}: {error}"))
            return
        per_second = cfg.get_unit_conversion_factor("s")
        samples = max(1, int(round(t_obs * per_second / cfg.dt_exp)))
        blank = [row for row in self.driven.rows() if row.hz.value_or_none() is None]
        if blank:
            grid = chi.chi_multipliers(n_freqs=len(blank), bounds=cfg.chi_freq_bounds).tolist()
            for row, multiple in zip(blank, grid):
                row.hz.setText(f"{multiple * own * per_second:.4g}")
        self._planned = {}
        for row in self.driven.rows():
            hz = row.hz.value_or_none()
            if row.problem() or hz is None:
                continue
            verdict = chi.probe_verdict(cfg, own, hz, samples)
            self._planned[id(row)] = (row.value(), *driven_reading(verdict, least_cycles=config.CHI_MIN_CYCLES))
        self._say_planned(PLANNED_TEXT.format(f"{own * per_second:.3g}", f"{low:.2g}", f"{high:.2g}"))

    def _say_planned(self, sentence: str) -> None:
        """What the planner says under the table, and the facts drawn again with its readings."""
        self.plan_said.set_text(sentence)
        self.plan_said.setVisible(bool(sentence))
        self.redraw()

    def _read_inputs(self) -> dict:
        """What the run reads of the form, through the engine's rules: the first that breaks is refused, and nothing is
        dispatched."""
        rows = self.form.rows
        values = {"t_obs": require_positive("t_obs", rows["t_obs"].value())}
        if self.mode() == "simulated":
            if self.typed_tick.isChecked():
                problems = self.values_grid.problems()
                if problems:
                    raise Refusal(" ".join(problems), field="cell")
                values["gt_values"] = self.values_grid.to_dicts()
                return values
            values["cell"] = require_file("cell", self.cell_picker.selected_path(), "cell")
            return values
        kind = self.kind()
        if kind == "chi":
            values["chi_f0_si"] = require_positive("chi_f0_si", rows["chi_f0_si"].value())
        if kind == "forced":
            values["drive"] = self._read_drive()
        values["recording_spont"] = require_file("recording_spont", rows["recording_spont"].value(), "recording")
        if kind == "forced":
            values["recording_forced"] = require_file("recording_forced", rows["recording_forced"].value(),
                                                      "recording")
        if kind == "chi":
            if not self.driven.rows():
                raise Refusal(INFER_BLANKS["recording_probe"][0], field="recording_probe")
            values["recording_probe"] = tuple((require_file("recording_probe", path, "recording"),
                                               require_positive("recording_probe", hz))
                                              for path, hz in rows["recording_probe"].value())
        return values

    def _read_drive(self) -> dict:
        """The drive's rows as ``{forcing name: value}``: amplitude and frequency above 0, the phase finite, any other a
        finite number."""
        out = {}
        for name in self._forcing_names():
            value = self.form.rows[self._drive_row(name)].value()
            field = _DRIVE_FIELD.get(name)
            if field == "drive_phase":
                out[name] = require_finite(field, value)
            elif field is not None:
                out[name] = require_positive(field, value)
            elif value is None:
                raise Refusal(f"{_drive_label(name)} is blank.")
            elif not math.isfinite(value):
                raise Refusal(f"{_drive_label(name)} must be a finite number.")
            else:
                out[name] = value
        return out

    def _prior_handed(self):
        """The prior in use, handed to a simulated inference only when it is the one the posterior was trained with: the
        cell is then checked against what training saw; else None."""
        session = getattr(self._screen, "session", None)
        prior, posterior = getattr(session, "inf_prior", None), getattr(session, "posterior", None)
        parents = getattr(getattr(posterior, "manifest", None), "parents", None)
        trained_with = parents.get("prior") if isinstance(parents, dict) else None
        return prior if prior is not None and trained_with and getattr(prior, "id", None) == trained_with else None

    def main_pressed(self) -> None:
        """Run the inference on the configuration and the posterior in use, with what the form holds."""
        if self._held()[1] is None:
            return
        try:
            values = self._read_inputs()
        except Refusal as refused:
            self._refusal(refused)
            return
        self._start(values, None)

    def _start(self, values: dict, accept) -> bool:
        """Dispatch the inference: a simulated cell, or the recordings as one set; the names and the draws. True when
        its run was started."""
        cfg, posterior = self._held()
        names = self._names
        common = dict(accept=accept, n_samples=self.form.rows["n_samples"].value(), name=names.name_value(),
                      note=names.note_value(), observation_name=names.other_value(), if_name_taken="unnamed",
                      provide_fig_sink=True, on_result=self._on_inference)
        if self.mode() == "simulated":
            return self.dispatch(orchestrator.simulated_inference, cfg, posterior, values["t_obs"],
                                 cell=values.get("cell"), gt_values=values.get("gt_values"),
                                 prior=self._prior_handed(), **common)
        kind = self.kind()
        if kind == "chi":
            rec = RecordingSet(spont=values["recording_spont"], forced=values["recording_probe"],
                               T_obs_s=values["t_obs"], F0_si=values["chi_f0_si"])
        elif kind == "forced":
            rec = RecordingSet(spont=values["recording_spont"], forced=((values["recording_forced"], None),),
                               T_obs_s=values["t_obs"], forcing_params_si=values["drive"])
        else:
            rec = RecordingSet(spont=values["recording_spont"], T_obs_s=values["t_obs"])
        return self.dispatch(orchestrator.experimental_inference, cfg, posterior, rec, **common)

    def run_spec(self, fn, *args, **kwargs) -> RunSpec:
        """An inference, of a simulated cell or of recordings: it writes an observation and the inference made on it."""
        if fn not in (orchestrator.simulated_inference, orchestrator.experimental_inference):
            return super().run_spec(fn, *args, **kwargs)
        return RunSpec("Inference", _PLACE, name=kwargs.get("name", ""), stop_button=STOP_WORDS,
                       kinds=("observation", "inference"))

    # ── what is remembered ──
    def restore_settings(self, qs) -> None:
        """The driven recordings the last inference started with, every one, put back as the table's rows, never run;
        one blank row when none is remembered."""
        try:
            count = int(settings.get_str(qs, DRIVEN_SIZE_KEY, "0") or 0)
        except ValueError:
            count = 0
        driven = []
        for at in range(max(count, 0)):
            path = settings.get_str(qs, DRIVEN_FILE_KEY.format(at), "")
            try:
                hz = float(settings.get_str(qs, DRIVEN_FREQUENCY_KEY.format(at), ""))
            except ValueError:
                hz = None
            driven.append((path or None, hz))
        self.driven.set_value(driven or [(None, None)])

    def _on_inference(self, payload) -> None:
        """The run returned its observation and the inference made on it, while it is still live: the observation comes
        into use."""
        observation, _inference = payload
        self._screen.take_observation(observation)
