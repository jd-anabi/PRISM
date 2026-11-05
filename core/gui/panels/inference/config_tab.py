"""Step 1, Model and options: the model, the kind of observation and the settings that go with them, applied at once as
what every later step is made for.

The step holds a draft and never a configuration: a configuration cannot be built before a bounds file, which declares
the parameters inferred and so the observation mode, and the Prior step chooses that. Applying reads the fields through
the page's one table of rules, which the line under each field reads as it is typed; asks before it lets go of the
records the session holds; and replaces the session with one that holds the new draft. Every row picks what the step
holds, so a choice that differs from it reads "Changed, not applied" under its field, and in the rail.

The drive amplitude and the frequency band used in training are measurements, shown as text: no draft carries them, so
every configuration takes core/config.py's. The three settings of the driven recordings stand only while the kind is the
frequency response, and open at core/config.py's values at every launch. The model and the kind are put back as they
were last applied, as choices; nothing else is read from the settings file, and nothing is written but by applying.

Plans stand under "Model", since a plan belongs to a model: the list of the model's plans, read again each time it
opens and at the page's readings; what the plan shown holds, or why it cannot be used; and the three presses. Choosing
in the list changes nothing. "Fill the forms from this plan" fills steps 1 to 3, asking first when it would replace a
value a person typed or that was put back; with "No plan" shown it returns them to the usual values. "Save as a plan…"
writes the settings that differ from their usual values. The plan in use is the screen's, so it outlives the session
applying replaces; a model chosen on the form leaves no plan in use and the values as they stand. Nothing of the folder
is read while a run is live, and the choice in the list is never remembered: every launch opens on "No plan".
"""
import math
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QSize, Qt
from PySide6.QtWidgets import QHBoxLayout, QLabel, QLineEdit, QPushButton, QSizePolicy, QVBoxLayout, QWidget

from core import cli, config, registry
from core import plans as engine_plans
from core.config import BOUNDS_PATH, CHI_K_MAX, CHI_MIN_CYCLES, VALID_LABELS, VALID_MODELS
from core.Helpers import file_manager
from core.refusals import Refusal, default_clause

from ... import chain, design, fields, keys, plans, settings, style, texts
from ...remembered import KIND_KEY, LAST_USED_AT_KEY, LAST_USED_KEY, MODEL_KEY
from ...reported import reported
from ...run.controller import controller
from ...session import ConfigDraft
from ...step_facts import (CHANGED_TEXT, NOT_HELD, Fact, SelectionRow, held_value, model_facts, model_in_use,
                           units_words)
from ...widgets import labeled_inputs
from ...widgets.artifact_picker import LIST_MOST
from ...widgets.ask import ask, ask_text
from ...widgets.choice import Choice
from ...widgets.form import TickedField, _may_wrap
from ...widgets.labeled_inputs import FloatField, IntField
from ...widgets.marker import painted_icon
from ...widgets.message_row import MessageRow
from ...widgets.rule_table import Column, RuleTable, TextCell
from .help_text import HELP
from .step_page import StepPage

TITLE = "Model and options"
SENTENCE = "Choose the model and the kind of observation. Everything after this step is made for these choices."
APPLY = "Apply model and options"
NOTHING_YET = "Choose the model and the kind of observation, then apply them."
GO_TO_PRIOR = "Go to Prior"
KINDS = (("Set by the bounds file", False), ("Frequency response", True))
ROTATIONS = (("On", True), ("Off", False))
UNIT_SOURCES = (("From the model's units file", "file"), ("Typed here", "typed"))
FREQUENCY_RESPONSE_TEXT = ("Frequency response (χ(ω)): one spontaneous recording and several driven recordings, each at "
                           "one frequency.")
BOUNDS_FILE_TEXT = ("Set by the bounds file: a spontaneous recording, and one driven recording when the file has a "
                    "forcing section.")
SLOTS_TEXT = "Slots: the most driven recordings one observation can hold. Fixed in the posterior once it is trained."
LOCK_IN_TEXT = ("Lock-in limit: the longest stretch of a driven recording, in drive cycles, used to measure the "
                "response.")
TYPED_TEXT = "Typed units cannot continue a simulation set made with the model's units file, even when they match it."
AMPLITUDE_TEXT = "Drive amplitude in training: {:g} (no unit)."
BAND_TEXT = "Frequency band in training: {:g} to {:g} times the cell's own frequency."
MEASURED = "fixed by measurement"
KEPT = "kept from last time"
UNITS_FROM = "{}, from {}"
NO_UNITS_FILE = "This model has no units file: {}"
#: Units held from a units file that gave no words, as a line under the row names them.
THE_UNITS_FILE = "the model's units file"
TYPED_HERE = "typed here"
FORCING_TEXT = "This model has external forcing, so parameter inference cannot use it."
NO_PARAMETERS_TEXT = "This model has no free parameters to infer, so parameter inference cannot use it."
#: Said while the model chosen is one of your own that parameter inference can take: the place is the window's own words.
CALIBRATION_TEXT = ("The checks are not tuned for a model of your own: read their results in {}, before you rely on its "
                    "posterior.")
#: The words beside the main button while a field breaks a rule of the page.
SETTING_BLOCKS = "a setting cannot be used"
QUESTION = "Start a new session?"
QUESTION_TEXT = ("Applying releases what this session holds: {}. A different model, units or kind of observation makes "
                 "them invalid. Each stays in Records and can be loaded again.")
KEEP = "Keep this session"
APPLY_ANSWER = "Apply and start a new session"
#: The fewest driven recordings an observation holds, and the fewest slots.
LEAST_DRIVEN = 2
#: The rows the page's rules read, in the order a press reads them.
RULED = ("chi_n_freqs", "chi_k_pad", "chi_max_cycles", "units")
#: The settings of the frequency response: shown, and read, only while that kind is chosen.
DRIVEN = ("chi_n_freqs", "chi_k_pad", "chi_max_cycles", "chi_f0", "chi_freq_bounds")
#: The three of them a draft carries.
DRIVEN_HELD = ("chi_n_freqs", "chi_k_pad", "chi_max_cycles")

# ── plans ──
PLAN_LABEL = "Plan"
NO_PLAN_ITEM = "No plan"
LAST_USED_TEXT = "Last used: {}, {}"
LAST_USED_UNDATED_TEXT = "Last used: {}"
LAST_USED_GONE_TEXT = "Last used: {} (the file is no longer there)"
ONE_SETTING = "1 setting"
SETTINGS = "{} settings"
ONE_STEP = "1 step"
STEPS = "{} steps"
HOLDS_TEXT = "{} in {}"
SHOW_HOLDS = "Show what it holds"
HOLDS_COLUMNS = ("Step", "Setting", "In the plan", "Usual")
NOT_APPLIED_YET = "not applied yet"
IN_USE_TEXT = "In use: {}."
PLAN_FILLED_WORDS = "plan {}, filled {}"
SOURCE_FILLED_WORDS = "settings from {} {}, filled {}"
#: The two, for a source that keeps no moment of its fill.
PLAN_WORDS = "plan {}"
SOURCE_WORDS = "settings from {} {}"
NO_PLAN_WORDS = "no plan"
FILL_WORDS = "Fill the forms from this plan"
USUAL_WORDS = "Return the forms to the usual values"
SAVE_WORDS = "Save as a plan…"
OPEN_WORDS = "Open the plans folder"
#: Beside the fill button while the plan shown cannot be used: the table's own sentence for a plan to correct.
FILL_BLOCKED_TEXT = fields.CONTROL["plan"]
#: Beside the fill button while the plan in use is no longer in the folder: the list fell back to "No plan", and the
#: button returns the forms to their usual values, as it does with "No plan" chosen.
FILL_GONE_TEXT = "The plan in use is no longer in the plans folder."
NOTHING_TO_SAVE_TEXT = "Every setting is at its usual value, so there is nothing to save."
FILL_TITLE = "Fill the forms from plan {}?"
FILL_ONE_TEXT = ("Filling from plan {} sets every setting a plan can hold: 1 from the plan, the rest to their usual "
                 "values.")
FILL_TEXT = ("Filling from plan {} sets every setting a plan can hold: {} from the plan, the rest to their usual "
             "values.")
KEEP_WHAT_IS_THERE = "Keep what is there"
FILL_ANSWER = "Fill the forms"
USUAL_TITLE = "Return the forms to the usual values?"
USUAL_TEXT = "This sets every setting a plan can hold to its usual value."
USUAL_ANSWER = "Return to the usual values"
SAVE_TITLE = "Save as a plan"
SAVE_TEXT = ("Only the settings that differ from their usual values are written. A setting left out follows the usual "
             "value.")
SAVE_LABEL = "Plan name"
SAVE_ANSWER = "Save"
WRITTEN_ITEM = "{} ({}): {}"
TYPED_BOUNDS_TEXT = "The bounds were typed by hand, so this plan names no bounds file."
OTHER_BOUNDS_TEXT = ("The bounds list holds another model's files, not those of the model shown, so this plan names no "
                     "bounds file.")
NOT_APPLIED_SAVED_TEXT = "Saved as the forms show them, though not applied: {}."
REPLACE_TITLE = "Replace plan {}?"
REPLACE_TEXT = "{} was last changed on {}. The earlier file cannot be brought back."
KEEP_PLAN = "Keep the existing plan"
REPLACE_ANSWER = "Replace it"
SAVED_TEXT = ("Saved as plan {}. Commit this file before a long run, or the run's records are marked as made from a "
              "changed tree.")
CANNOT_SAVE_TEXT = "This plan cannot be saved: {}"
NOT_WRITTEN_TEXT = "This plan cannot be saved: {} could not be written ({})."
BLANK_TEXT = "{} is blank."
OK_TEXT = "OK"
MORE_TEXT = "And {} more."
#: How many of a plan's problems are shown under the list, and how much of each: a file of notes gives one a line, each
#: echoing its line whole, and a page that grew with them would push every setting out of reach. One more is said
#: whole in the line that would count it.
PROBLEMS_SHOWN = 8
PROBLEM_MOST = 300


class _ModelChoice(Choice):
    """The list of models: each shown by its name on screen, spoken of by its engine name. ``addItems``,
    ``currentText`` and ``setCurrentText`` take and give engine names, as every list of models in the window does."""

    def addItems(self, names) -> None:
        for name in names:
            self.addItem(chain.model_label(name), name)

    def currentText(self) -> str:
        name = self.currentData()
        return name if isinstance(name, str) else super().currentText()

    def setCurrentText(self, name) -> None:
        found = self.findData(name)
        if found >= 0:
            self.setCurrentIndex(found)


def _choice(items) -> Choice:
    choice = Choice()
    for words, value in items:
        choice.addItem(words, value)
    return choice


def _words_of(items, value) -> str:
    """The words a list shows for ``value``, or the value itself when no item holds it."""
    return next((words for words, held in items if held == value), str(value))


def _one_plan(one, other) -> bool:
    """Whether two names are the one plan: the same, or the same without regard to case, as Windows keeps one file for
    both; None is "No plan"."""
    return one == other or (one is not None and other is not None and one.casefold() == other.casefold())


def _listed_as(names, name) -> "str | None":
    """The name in ``names`` the plan ``name`` stands under: its own where it is listed, else the first listed in
    another case, as Windows keeps one file for both; None for "No plan" and for a plan not listed. A file system that
    keeps case may list both spellings, and the plan is then its own."""
    if name is None:
        return None
    if name in names:
        return name
    return next((listed for listed in names if listed.casefold() == name.casefold()), None)


def _counted(count: int, one: str, many: str) -> str:
    return one if count == 1 else many.format(count)


class _PlanChoice(Choice):
    """The list of a model's plans, "No plan" first: as wide as its longest name up to the cap of a list of names, and
    read again, by ``before_opening``, each time it opens."""

    def __init__(self) -> None:
        super().__init__(most=LIST_MOST)
        self.before_opening = None

    def showPopup(self):
        if callable(self.before_opening):
            self.before_opening()
        super().showPopup()


class _UnitsField:
    """The units as a plan holds them, over the units row: the words typed here as one text, or None for the model's
    units file."""

    key = "units"

    def __init__(self, page) -> None:
        self._page, self._row = page, page.form.rows["units"]

    @property
    def kept(self) -> bool:
        return self._row.kept

    @kept.setter
    def kept(self, on: bool) -> None:
        self._row.kept = on

    def value(self):
        return " ".join(self._page.units_text.text().split()) if self._page._typed() else None

    def set_value(self, value) -> None:
        """The model's units file for None, else the words typed."""
        choice = self._page.units_choice
        if value is None:
            choice.setCurrentIndex(choice.findData("file"))
            return
        choice.setCurrentIndex(choice.findData("typed"))
        self._page.units_text.setText(str(value))

    def usual(self):
        return None

    def label_text(self) -> str:
        return self._row.label_text()

    def set_origin(self, text: str) -> None:
        self._row.set_origin(text)

    def in_error(self) -> bool:
        return self._row.in_error()

    def words_of(self, value) -> str:
        return THE_UNITS_FILE if value is None else units_words(str(value).split())


class ConfigPanel(StepPage):
    """Step 1: a model and its options, held as a draft once applied. ``model_combo`` lists the models; ``kind_choice``,
    ``rot_choice`` and ``units_choice`` are the other choices, and ``units_text`` the units typed here. ``plan_choice``
    lists the model's plans, with ``fill_button``, ``save_button`` and ``open_button`` under it."""

    def __init__(self, screen, parent=None):
        super().__init__(screen, TITLE, SENTENCE, step="model", parent=parent)
        self._units_read: dict = {}
        self._seeded = ""                    # what the box of typed units was last filled with from a units file
        self._shown_plan = None              # the plan the list shows, as it was last read, or why it cannot be used
        self._listed_for = None              # the model whose folder the list of plans was last read for
        self._folder_there = False           # whether that folder stood there at that reading
        self._hint = ""                      # the plan last used with the model shown, beside the list
        self._gone = None                    # the plan the list showed until its file went, "No plan" shown since
        self.model_combo = _ModelChoice()
        self.model_combo.addItems(VALID_MODELS)
        self.model_combo.setCurrentText("NADROWSKI")
        self.model_combo.currentIndexChanged.connect(self._model_chosen)
        self.kind_choice = _choice(KINDS)
        # Heard before its row hears it: the settings of a kind stand before the page judges them.
        self.kind_choice.currentIndexChanged.connect(self._kind_chosen)
        self.rot_choice = _choice(ROTATIONS)
        self.rot_choice.setCurrentIndex(self.rot_choice.findData(bool(config.REPARAM_ROTATE)))
        self.units_choice = _choice(UNIT_SOURCES)
        self.units_text = QLineEdit()
        self.units_text.setPlaceholderText("e.g. kHz ms nm pN")
        form = self.form
        form.add("model", self.model_combo, usual="NADROWSKI", help_text=HELP["model"])
        self._build_the_plans_block()
        form.add("chi_mode", self.kind_choice, usual=False, help_text=HELP["chi_mode"])
        form.add_sentence(FREQUENCY_RESPONSE_TEXT, under="chi_mode")
        form.add_sentence(BOUNDS_FILE_TEXT, under="chi_mode")
        form.add("chi_n_freqs", IntField(config.CHI_N_FREQS, grouped=True), usual=config.CHI_N_FREQS, folded=True,
                 help_text=HELP["chi_k"])
        form.add("chi_k_pad", IntField(config.CHI_K_PAD, grouped=True), usual=config.CHI_K_PAD, folded=True,
                 help_text=HELP["chi_k_pad"])
        form.add_sentence(SLOTS_TEXT, under="chi_k_pad")
        form.add("chi_max_cycles", FloatField(config.CHI_MAX_CYCLES), "drive cycles", usual=config.CHI_MAX_CYCLES,
                 folded=True, help_text=HELP["chi_max_cycles"])
        form.add_sentence(LOCK_IN_TEXT, under="chi_max_cycles")
        form.add_text("chi_f0", AMPLITUDE_TEXT.format(config.CHI_F0), MEASURED, folded=True)
        form.add_text("chi_freq_bounds", BAND_TEXT.format(*config.CHI_FREQ_BOUNDS), MEASURED, folded=True)
        form.add("reparam_rotate", self.rot_choice, usual=bool(config.REPARAM_ROTATE), folded=True,
                 help_text=HELP["reparam_rotate"])
        units = form.add("units", self.units_choice, usual="file", folded=True, help_text=HELP["units"])
        self.units_line = form.add_sentence("", under="units")
        units.add_under(self.units_text)
        units.hear(self.units_text)
        self.typed_note = form.add_sentence(TYPED_TEXT, under="units")
        self.units_choice.currentIndexChanged.connect(self._units_chosen)
        self._units_field = _UnitsField(self)
        self.restore_settings(settings.settings())
        self._kind_chosen()
        self._units_chosen()
        self.refresh_local_gates()

    @property
    def reads_the_disk(self) -> bool:
        """Whether a plan is in use, whose file the page reads: the line of when it read, and "Check again", stand only
        then."""
        return getattr(getattr(self._screen, "plan_source", None), "kind", None) == "plan"

    # ── what the step holds ──
    def _draft(self):
        """The draft the step holds, or None; None too on a page with no screen."""
        return getattr(getattr(self._screen, "session", None), "draft", None)

    def _chi(self) -> bool:
        return self.kind_choice.currentData() is True

    def _typed(self) -> bool:
        return self.units_choice.currentData() == "typed"

    def _units_override(self):
        """What the draft's units are: None for the model's units file, else the words typed here."""
        return tuple(self.units_text.text().split()) if self._typed() else None

    def _units_of(self, model) -> tuple:
        """The model's units file as the units line says it, read once each time the model is chosen, never as the page
        is drawn: its words, its path from the resources folder, and why it could not be read."""
        if model not in self._units_read:
            try:
                path = cli.resolve_units_file(model)
                tokens = file_manager.parse_units_file(path)
                where = Path(path).resolve().relative_to(Path(config.RESOURCES_ROOT).resolve()).as_posix()
                self._units_read[model] = (units_words(tokens), where, "")
            except Refusal as refused:
                self._units_read[model] = ("", "", refused.message)
            except Exception as error:               # noqa: BLE001 -- a model's units file is the model's own trouble
                self._units_read[model] = ("", "", f"{type(error).__name__}: {error}")
        return self._units_read[model]

    def _units_said(self, override, model) -> str:
        """Units held as a sentence names them, and where they come from: "kHz ms nm pN, from the model's file"."""
        if override is None:
            words = self._units_of(model)[0]
            return UNITS_FROM.format(words, "the model's file") if words else ""
        if isinstance(override, (tuple, list)):
            return f"{units_words(override)}, {TYPED_HERE}"
        return ""

    def _unusable(self, model) -> str:
        """Why parameter inference cannot take ``model``, or "" when it can."""
        if registry.is_user_model(model) and not registry.is_sbi_user_model(model):
            return FORCING_TEXT if registry.user_model_has_forcing(model) else NO_PARAMETERS_TEXT
        return ""

    # ── the rules ──
    def problems(self) -> dict:
        """Every rule of the page the form breaks now, by field key in the order a press reads them: the line under the
        field, built from its row's label, and what the press says, the same with the value it found and the usual one.
        The settings of the frequency response are read only while that kind is chosen, as the draft reads them; typed
        units only while they are typed."""
        rows, found = self.form.rows, {}

        def label(key) -> str:
            return rows[key].label_text()

        def rule(key, words, value) -> tuple:
            shown = f"{value:g}" if isinstance(value, float) else str(value)
            return f"{label(key)} must be {words}.", f"{label(key)} must be {words}; got {shown}{default_clause(key)}."

        if self._chi():
            numbers = {}
            for key in ("chi_n_freqs", "chi_k_pad", "chi_max_cycles"):
                said = rows[key].reading()
                if said is None:
                    numbers[key] = rows[key].value()
                elif said.endswith(" is blank."):
                    found[key] = (said, f"{said[:-1]}{default_clause(key)}.")
                else:
                    found[key] = (said, said)
            for key in ("chi_n_freqs", "chi_k_pad"):
                value = numbers.get(key)
                if value is not None and value < LEAST_DRIVEN:
                    found[key] = rule(key, f"at least {LEAST_DRIVEN}", value)
                elif value is not None and value > CHI_K_MAX:
                    found[key] = rule(key, f"at most {CHI_K_MAX}", value)
            most, pad = numbers.get("chi_n_freqs"), numbers.get("chi_k_pad")
            if most is not None and pad is not None and not found.keys() & {"chi_n_freqs", "chi_k_pad"} and most > pad:
                found["chi_n_freqs"] = rule("chi_n_freqs", f"at most the {label('chi_k_pad').lower()}, {pad}", most)
            cycles = numbers.get("chi_max_cycles")
            if cycles is not None and not math.isfinite(cycles):
                found["chi_max_cycles"] = rule("chi_max_cycles", "a finite number", cycles)
            elif cycles is not None and cycles <= CHI_MIN_CYCLES:
                found["chi_max_cycles"] = rule("chi_max_cycles", f"greater than {CHI_MIN_CYCLES:g}", cycles)
        if self._typed():
            typed = self._units_override()
            if not typed:
                found["units"] = (f"{label('units')} typed here are blank.",
                                  f"{label('units')} typed here are blank: type at least one unit, or use the model's "
                                  "units file.")
            else:
                try:
                    cli.units_to_factors(typed)
                except Refusal as refused:
                    found["units"] = (f"{label('units')} typed here cannot be used: {refused.message}",) * 2
                except Exception as error:           # noqa: BLE001 -- what a units library raises on a word it cannot read
                    found["units"] = (f"{label('units')} typed here cannot be used: {error}",) * 2
        return {key: found[key] for key in RULED if key in found}

    def _read(self) -> dict:
        """What a draft is made of, read through the rules: the first rule broken is refused, and nothing changes."""
        for key, (_line, said) in self.problems().items():
            raise Refusal(said, field=key)
        chi, rows = self._chi(), self.form.rows
        return {"units": self._units_override(),
                **{key: rows[key].value() if chi else None for key in ("chi_n_freqs", "chi_k_pad", "chi_max_cycles")}}

    # ── what a page answers ──
    def main_words(self) -> str:
        return APPLY

    def nothing_to_apply(self) -> bool:
        """A draft is held and every choice shows what it holds."""
        return self._draft() is not None and not any(row.changed() for row in self.selection_rows())

    def selection_rows(self) -> tuple:
        """Every row of the step: each picks what the step holds. The settings of the frequency response are compared
        only while both the draft and the form are of that kind, as they are read only then."""
        draft, rows = self._draft(), self.form.rows

        def held(name):
            return lambda: held_value(draft, name)

        def driven(name):
            return lambda: held_value(draft, name) if held_value(draft, "chi_mode") is True and self._chi() else NOT_HELD

        def rotation():
            value = held_value(draft, "reparam_rotate")
            return value if isinstance(value, bool) else NOT_HELD

        def number(key):
            return lambda value: f"{value:g} {rows[key].unit.text()}".rstrip() if isinstance(value, float) else str(value)

        model = getattr(draft, "model", None)
        return (SelectionRow("model", held("model"), self.model_combo.currentText, chain.model_label),
                SelectionRow("chi_mode", held("chi_mode"), self.kind_choice.currentData,
                             lambda value: _words_of(KINDS, value)),
                *(SelectionRow(key, driven(key), rows[key].value, number(key))
                  for key in ("chi_n_freqs", "chi_k_pad", "chi_max_cycles")),
                SelectionRow("reparam_rotate", rotation, self.rot_choice.currentData,
                             lambda value: _words_of(ROTATIONS, value)),
                SelectionRow("units", held("units_override"), self._units_override,
                             lambda value: self._units_said(value, model) or THE_UNITS_FILE))

    def plan_fields(self) -> dict:
        rows = self.form.rows
        return {**{key: rows[key] for key in ("chi_mode", *DRIVEN_HELD, "reparam_rotate")}, "units": self._units_field}

    def plan_read(self, applied: bool) -> dict:
        """What a run reads of this step: the draft applied, or the form as a draft would read it; the driven
        recordings' settings only with the frequency response."""
        if not applied:
            fields_ = self.plan_fields()
            return {key: fields_[key].value() for key in ("chi_mode", *(DRIVEN_HELD if self._chi() else ()),
                                                          "reparam_rotate", "units")}
        draft = self._draft()
        if draft is None:
            return {}
        chi = getattr(draft, "chi_mode", None) is True
        read = {"chi_mode": chi, **{key: getattr(draft, key, None) for key in (DRIVEN_HELD if chi else ())}}
        rotate, units = getattr(draft, "reparam_rotate", None), getattr(draft, "units_override", None)
        if isinstance(rotate, bool):
            read["reparam_rotate"] = rotate
        read["units"] = " ".join(str(unit) for unit in units) if isinstance(units, (tuple, list)) else None
        return read

    def facts_from_form(self) -> tuple:
        """Each rule broken, under its field and as a fact; where a value put back came from; that the model cannot be
        used, when it cannot; and, for a model of your own it can use, that its checks are not tuned."""
        found = self.problems()
        for key, row in self.form.rows.items():
            if row.widget is None or key == "plan":
                continue
            row.set_line("err" if key in found else "", found[key][0] if key in found else "")
            row.set_origin(KEPT if row.kept else "")
        self._draw_the_plans_block()
        model = self.model_combo.currentText()
        own = ((Fact("info", CALIBRATION_TEXT.format(keys.place_words("checks"))),)
               if registry.is_user_model(model) and not self._unusable(model) else ())
        return model_facts(usable_reason=self._unusable(model),
                           field_facts=(*(Fact("err", line, SETTING_BLOCKS) for line, _said in found.values()), *own),
                           plan_lines=self.plan_lines())

    def in_use_words(self) -> tuple:
        draft = self._draft()
        if draft is None:
            return "", NOTHING_YET
        return model_in_use(draft, self._units_said(getattr(draft, "units_override", None), getattr(draft, "model", None)))

    def main_pressed(self) -> None:
        self._build_config()

    def refresh_local_gates(self) -> None:
        """What the step holds and what follows from it, and the way on to the next step once a model is applied."""
        super().refresh_local_gates()
        held = self._draft() is not None
        self.set_next(GO_TO_PRIOR if held else "", "prior" if held else None)

    # ── the choices ──
    def _model_chosen(self, _index: int) -> None:
        self._on_model_changed(self.model_combo.currentText())

    def _on_model_changed(self, model: str) -> None:
        """A model chosen, by a person or by the list read again: its units file read again, so that a file written since
        says its units; its units line; and the box of typed units filled from it while the units come from that file, or
        while the box still holds what a file filled it with, so that typing them starts from the file in use. Units
        typed here are never written over."""
        self.form.rows["model"].kept = False
        self._model_left()
        self._units_read.pop(model, None)
        words, where, trouble = self._units_of(model)
        self.units_line.set_text(UNITS_FROM.format(words, where) if words else NO_UNITS_FILE.format(trouble))
        if not self._typed() or self.units_text.text() == self._seeded:
            self.units_text.setText(words)
            self._seeded = words
        self._edited()

    def _kind_chosen(self, _index: int = 0) -> None:
        """The settings of the frequency response stand only while it is chosen."""
        for key in DRIVEN:
            self.form.rows[key].set_shown(self._chi())

    def _units_chosen(self, _index: int = 0) -> None:
        """The line of the model's units file while it is chosen; the box of typed units, and what typing them means,
        while they are typed."""
        typed = self._typed()
        self.units_line.setVisible(not typed)
        self.units_text.setVisible(typed)
        self.typed_note.setVisible(typed)

    # ── applying ──
    def _build_config(self) -> None:
        """Apply: read the fields through the rules, ask before letting go of what the session holds, and hold the new
        draft. A refusal and a kept session change nothing. The button greys as it applies: pressed by the keyboard,
        which the ring on it says before the question can take the keyboard, it hands the keyboard on to the next step."""
        follower = style.current_follower()
        by_keyboard = follower is not None and follower.target() is self.foot.main_button
        model = self.model_combo.currentText()
        if self._unusable(model):
            return                                   # the main button is greyed, with its reason, for this model
        try:
            values = self._read()
        except Refusal as refused:
            self._refusal(refused)
            return
        draft = self._draft_of(model, values)
        held = self._screen.session_contents()
        if held and not ask(self, QUESTION, QUESTION_TEXT.format("; ".join(held)), KEEP, APPLY_ANSWER):
            return
        self._screen.new_draft(draft)
        if by_keyboard and not self.next_button.isHidden():
            self.next_button.setFocus(Qt.TabFocusReason)

    def _draft_of(self, model: str, values: dict) -> ConfigDraft:
        """The draft of ``model`` from what the rules read, with the kind and the rotation the form shows."""
        return ConfigDraft(
            model=model, labels=VALID_LABELS[VALID_MODELS.index(model)], state_dep_drift=registry.state_dep_drift(model),
            units_override=values["units"], chi_mode=self._chi(), chi_n_freqs=values["chi_n_freqs"], chi_f0=None,
            chi_freq_bounds=None, chi_k_pad=values["chi_k_pad"], chi_max_cycles=values["chi_max_cycles"],
            reparam_rotate=self.rot_choice.currentData())

    def apply_quietly(self, model: str, options: dict, *, fill=None) -> bool:
        """Apply ``model`` with ``options`` as "Apply model and options" does, asking nothing, for a caller that asked
        its own question first or lets go of nothing held: the model on the form; then ``fill``, when handed, which
        fills the forms once the model stands, since a model chosen on the form lets the plan in use go and a fill made
        before would go with it; then each setting of ``options`` by its name in a draft (``chi_mode``, the three of the
        driven recordings, ``reparam_rotate``, and ``units_override``, the units typed or None for the model's file),
        every other setting of the step, and one ``options`` gives as None, at its usual value; then the rules read the
        form as a press reads it, and a value one refuses is told as a press's refusal is and nothing is applied. True
        when the draft is held. Nothing while a run is live, for a model the list does not hold, or for one parameter
        inference cannot take."""
        if controller().is_live() or self.model_combo.findData(model) < 0 or self._unusable(model):
            return False
        self.model_combo.setCurrentText(model)
        if fill is not None:
            fill()
        rows = self.form.rows
        for key in ("chi_mode", *DRIVEN_HELD, "reparam_rotate"):
            value = options.get(key)
            rows[key].set_value(rows[key].usual() if value is None else value)
        units = options.get("units_override")
        self._units_field.set_value(None if units is None else " ".join(str(unit) for unit in units))
        try:
            values = self._read()
        except Refusal as refused:
            self._refusal(refused)
            return False
        self._screen.new_draft(self._draft_of(model, values))
        return True

    # ── plans ──
    def _build_the_plans_block(self) -> None:
        """The row of the plans, under "Model": the list, the hint of the plan last used beside it, and under it what
        the plan shown holds or why it cannot be used, what is in use, and the three presses. Nothing is read here: the
        list holds "No plan" until the page's first reading."""
        self.plan_choice = _PlanChoice()
        self.plan_choice.addItem(NO_PLAN_ITEM, None)
        self.plan_choice.before_opening = self._read_the_plans
        self.plan_choice.currentIndexChanged.connect(self._plan_chosen)
        row = self.form.add("plan", self.plan_choice, label=PLAN_LABEL)
        self.plan_about = self.form.add_sentence("", under="plan")
        self.plan_about.setVisible(False)
        block = QWidget()
        box = QVBoxLayout(block)
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(design.scaled(design.GAP["field_message"]))
        self.plan_trouble = MessageRow("err", "")
        self.plan_problems = []
        for _ in range(PROBLEMS_SHOWN + 1):
            line = QLabel()
            line.setTextFormat(Qt.PlainText)
            line.setWordWrap(True)
            line.setContentsMargins(design.scaled(design.SIZE["marker"]) + design.scaled(design.GAP["message_marker"]),
                                    0, 0, 0)
            self.plan_problems.append(line)
        self.holds_button = style.set_class(QPushButton(SHOW_HOLDS), cls="fold")
        self.holds_button.setIcon(painted_icon("chev_right"))
        self.holds_button.setIconSize(QSize(*(design.scaled(n) for n in design.GAP["fold_icon"])))
        self.holds_button.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        self.holds_button.clicked.connect(self._holds_pressed)
        self.holds_table = RuleTable([Column(HOLDS_COLUMNS[0], "fit", wrap=True), Column(HOLDS_COLUMNS[1]),
                                      Column(HOLDS_COLUMNS[2], "fit", wrap=True),
                                      Column(HOLDS_COLUMNS[3], "fit", wrap=True)], selectable=False)
        self.holds_table.setVisible(False)
        self.fill_button = QPushButton(FILL_WORDS)
        self.fill_button.clicked.connect(self._fill_pressed)
        self.fill_reason = style.set_class(QLabel(), cls="small")
        self.save_button = QPushButton(SAVE_WORDS)
        self.save_button.clicked.connect(self._save_pressed)
        self.save_reason = style.set_class(QLabel(), cls="small")
        self.open_button = QPushButton(OPEN_WORDS)
        self.open_button.clicked.connect(self._open_pressed)
        for part in (self.plan_trouble, *self.plan_problems):
            box.addWidget(part)
            part.setVisible(False)
        box.addWidget(self.holds_button, 0, Qt.AlignLeft)
        box.addWidget(self.holds_table)
        row.add_under(block)
        self.plan_in_use = self.form.add_sentence("", under="plan")
        buttons = QWidget()
        box = QVBoxLayout(buttons)
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(design.scaled(design.GAP["field_message"]))
        for button, reason in ((self.fill_button, self.fill_reason), (self.save_button, self.save_reason),
                               (self.open_button, None)):
            line = QHBoxLayout()
            line.setContentsMargins(0, 0, 0, 0)
            line.setSpacing(design.scaled(design.SIZE["row_gap"]))
            line.addWidget(button)
            if reason is not None:
                reason.setTextFormat(Qt.PlainText)
                reason.setWordWrap(True)
                line.addWidget(reason, 1)
            line.addStretch(0 if reason is not None else 1)
            box.addLayout(line)
        row.add_under(buttons)

    def _plans_folder(self) -> "Path | None":
        """The folder of the plans of the model shown on the form; None on a page with no screen."""
        held = getattr(self._screen, "plans_folder", None)
        return None if held is None else plans.model_folder(held, self.model_combo.currentText())

    def _step_pages(self) -> tuple:
        """The steps a plan fills, as the screen holds them; none on a page with no screen."""
        pages = getattr(self._screen, "step_pages", None)
        return pages() if callable(pages) else ()

    def _model_left(self) -> None:
        """Another model shown on the form: its plans are not read yet, the list shows "No plan", and no plan is in use,
        nor a plan's bounds file waiting on step 2; the values on the forms stay as they are."""
        if not hasattr(self, "plan_choice"):
            return
        self.plan_choice.blockSignals(True)
        try:
            while self.plan_choice.count() > 1:
                self.plan_choice.removeItem(1)
            self.plan_choice.setCurrentIndex(0)
        finally:
            self.plan_choice.blockSignals(False)
        self._shown_plan, self._listed_for, self._folder_there, self._hint, self._gone = None, None, False, "", None
        screen = self._screen
        prior = getattr(screen, "prior_panel", None)
        if prior is not None:
            prior.let_the_plans_file_go()
        if getattr(screen, "plan_source", None) is not None:
            screen.take_plan(None)

    def _read_the_plans(self) -> None:
        """The model's plans listed again, the plan shown kept while it stands there, under its own name or else under
        the name the folder lists it by in another case, else "No plan" until its file is back; whether the folder
        stands there; the plan last used with the model, found without regard to case, and whether its file is still
        there; and the plan shown read again. Nothing while a run is live."""
        folder = self._plans_folder()
        if controller().is_live() or folder is None:
            return
        model = self.model_combo.currentText()
        names = plans.list_plans(folder)
        choice, shown = self.plan_choice, self.plan_choice.currentData()
        if shown is None and self._gone is not None:
            shown = self._gone                   # the plan whose file went is shown again once its file is back
        found = _listed_as(names, shown)
        choice.blockSignals(True)
        try:
            while choice.count() > 1:
                choice.removeItem(1)
            for name in names:
                choice.addItem(name, name)
                choice.setItemData(choice.count() - 1, name, Qt.ToolTipRole)
            choice.setCurrentIndex(max(0, choice.findData(found)) if found is not None else 0)
        finally:
            choice.blockSignals(False)
        choice.say_whole()
        self._gone = shown if shown is not None and found is None else None
        self._listed_for, self._folder_there = model, folder.is_dir()
        last = settings.get_str(settings.settings(), LAST_USED_KEY.format(model.lower()), "").strip()
        when = settings.get_str(settings.settings(), LAST_USED_AT_KEY.format(model.lower()), "").strip()
        last_found = next((name for name in names if name.casefold() == last.casefold()), None)
        if not last:
            self._hint = ""
        elif last_found is None:
            self._hint = LAST_USED_GONE_TEXT.format(last)
        else:
            try:
                self._hint = LAST_USED_TEXT.format(last_found,
                                                   texts.date_text(datetime.fromisoformat(when).astimezone()))
            except (TypeError, ValueError, OverflowError, OSError):
                self._hint = LAST_USED_UNDATED_TEXT.format(last_found)
        self._read_the_plan_shown()
        self._draw_the_plans_block()

    def _read_the_plan_shown(self) -> None:
        """The plan the list shows, read and checked, or why it cannot be used; none for "No plan"."""
        name, folder = self.plan_choice.currentData(), self._plans_folder()
        if name is None or folder is None:
            self._shown_plan = None
            return
        try:
            self._shown_plan = plans.read_checked(plans.plan_path(folder, name),
                                                  bounds_folder=BOUNDS_PATH / self.model_combo.currentText().lower())
        except engine_plans.PlanRefusal as refused:
            self._shown_plan = refused

    def _plan_chosen(self, _index: int = 0) -> None:
        """Another plan chosen in the list: it is read, so that what it holds or why it cannot be used is said. The
        forms do not change."""
        self._gone = None
        if not controller().is_live():
            self._read_the_plan_shown()
        self.redraw()

    def _holds_pressed(self) -> None:
        opening = self.holds_table.isHidden()
        self.holds_table.setVisible(opening)
        self.holds_button.setIcon(painted_icon("chev_down" if opening else "chev_right"))

    def _draw_the_plans_block(self) -> None:
        """What the row of the plans says, as the list, the plan shown, the source in use and the forms stand: reads
        nothing."""
        if not hasattr(self, "plan_choice"):
            return
        row, shown, screen = self.form.rows["plan"], self._shown_plan, self._screen
        source = getattr(screen, "plan_source", None)
        listed = self.plan_choice.currentData()
        row.set_origin(self._hint if listed is None else "")
        in_use = source.name if source is not None and source.kind == "plan" else None
        gone = in_use is not None and listed is None and _one_plan(self._gone, in_use)
        in_the_list = _listed_as([self.plan_choice.itemData(at) for at in range(1, self.plan_choice.count())], in_use)
        held, changed = self._in_use_words(source), listed != (in_the_list or in_use) and not gone
        row.set_line("warn" if changed else "", CHANGED_TEXT.format(held) if changed else "")
        self.plan_in_use.set_text(IN_USE_TEXT.format(held))
        refused = isinstance(shown, engine_plans.PlanRefusal)
        held = shown if isinstance(shown, engine_plans.Plan) else None
        self.plan_about.set_text(self._about_words(held) if held is not None else "")
        self.plan_about.setVisible(held is not None)
        self.plan_trouble.set_sentence(shown.message if refused else "")
        self.plan_trouble.setVisible(refused)
        problems = list(shown.problems) if refused else []
        whole = len(problems) if len(problems) <= PROBLEMS_SHOWN + 1 else PROBLEMS_SHOWN
        for at, line in enumerate(self.plan_problems):
            if at < whole:
                said = problems[at].message
                line.setText(_may_wrap(said if len(said) <= PROBLEM_MOST else said[:PROBLEM_MOST - 1] + "…"))
                line.setVisible(True)
            elif at == PROBLEMS_SHOWN and len(problems) > whole:
                line.setText(MORE_TEXT.format(len(problems) - whole))
                line.setVisible(True)
            else:
                line.setVisible(False)
        self.holds_button.setVisible(held is not None)
        if held is not None:
            self.holds_table.set_rows(self._holds_rows(held))
        else:
            self.holds_table.setVisible(False)
            self.holds_button.setIcon(painted_icon("chev_right"))
        self.fill_button.setText(USUAL_WORDS if listed is None else FILL_WORDS)
        self.fill_button.setEnabled(not refused)
        self.fill_reason.setText(FILL_BLOCKED_TEXT if refused else FILL_GONE_TEXT if gone else "")
        values, _remarks, wrong = self._to_save()
        nothing = not values and not wrong
        self.save_button.setEnabled(not nothing)
        self.save_reason.setText(NOTHING_TO_SAVE_TEXT if nothing else "")
        self.open_button.setEnabled(self._folder_there)

    @staticmethod
    def _in_use_words(source) -> str:
        """The source in use as "In use: …" names it: "no plan", "plan retrain, filled 6 Oct 2026, 12:04", with no
        moment for a source that keeps none."""
        if source is None:
            return NO_PLAN_WORDS
        if source.at is None:
            return PLAN_WORDS.format(source.name) if source.kind == "plan" else SOURCE_WORDS.format(source.kind,
                                                                                                    source.name)
        when = texts.date_text(source.at)
        if source.kind == "plan":
            return PLAN_FILLED_WORDS.format(source.name, when)
        return SOURCE_FILLED_WORDS.format(source.kind, source.name, when)

    @staticmethod
    def _about_words(plan) -> str:
        """What the plan shown says of itself, and how many settings it holds in how many steps."""
        steps = {heading for heading, _name in plan.values}
        counted = HOLDS_TEXT.format(_counted(len(plan.values), ONE_SETTING, SETTINGS),
                                    _counted(len(steps), ONE_STEP, STEPS))
        return f"{plan.about} {counted}.".strip() if plan.about else f"{counted}."

    def _holds_rows(self, plan) -> list:
        """What the plan shown holds, a row a setting in the file's order: the step, the setting by its label with its
        name in the file beside it, the plan's value, and the usual one; a setting of a later step is not applied
        yet."""
        rows_of = {plans.HEADING_OF[page.step]: page.plan_fields() for page in self._step_pages()}
        rows = []
        for (heading, name), value in plan.values.items():
            row = rows_of.get(heading, {}).get(name)
            label = row.label_text() if row is not None else self._label_of(name)
            plan_words = plans.value_words(row, value) if row is not None else plans.value_words(None, value)
            usual = NOT_APPLIED_YET if row is None else (plans.NONE_WORDS if name in plans.LEFT_ALONE
                                                         else plans.value_words(row, row.usual()))
            rows.append([TextCell(heading, wrap=True), TextCell(f"{label} ({name})", wrap=True),
                         TextCell(plan_words, wrap=True), TextCell(usual, wrap=True)])
        return rows

    @staticmethod
    def _label_of(name: str) -> str:
        try:
            return fields.label(name)
        except KeyError:
            return name

    def _to_save(self) -> tuple:
        """What a save would write, keyed as a plan's values are: every setting of steps 1 to 3 whose form differs from
        its usual value, and the bounds file when one is selected in the folder of the model shown, or waits; the driven
        recordings' settings only with the frequency response. With each setting's remark: its label, and the words on
        the form where the file's differ. And what is wrong with the settings that hold no value: blank, or the row's
        own sentence for a text that does not read."""
        values, remarks, wrong = {}, {}, []
        for page in self._step_pages():
            heading = plans.HEADING_OF.get(page.step)
            for name, row in page.plan_fields().items():
                if name in DRIVEN_HELD and page is self and not self._chi():
                    continue
                if name in plans.LEFT_ALONE and self._bounds_of_another_model():
                    continue
                value = row.value()
                if value is plans.TYPED or (name in plans.LEFT_ALONE and value is None):
                    continue
                if name not in plans.LEFT_ALONE and plans.same(value, row.usual()):
                    continue
                if value is None or value == "":
                    wrong.append(self._unread_words(row))
                    continue
                values[(heading, name)] = value
                shown = plans.shown_words(row, value)
                remarks[(heading, name)] = row.label_text() if shown is None else f"{row.label_text()}: {shown}"
        return values, remarks, wrong

    @staticmethod
    def _unread_words(row) -> str:
        """Why a row holds no value a plan can write: blank, or the row's own sentence for a text that does not read,
        named by the row's label where the sentence names none; read from the field itself, so that a row a view hides
        says it too."""
        label, widget = row.label_text(), getattr(row, "widget", None)
        field = widget.field if isinstance(widget, TickedField) else widget
        typed = plans.typed_words(row)
        said = labeled_inputs.reading(field, label) if typed and isinstance(field, (FloatField, IntField)) else None
        if not said:
            return BLANK_TEXT.format(label)
        return said if said.startswith(label) else f"{label}: {said}"

    def _bounds_of_another_model(self) -> bool:
        """Whether step 2's list selects a file in another model's folder than the model shown: the list holds the files
        of the model applied, or Nadrowski's while nothing is applied. A plan's file that waits is the shown model's."""
        prior = getattr(self._screen, "prior_panel", None)
        if prior is None or prior.waiting_file() is not None:
            return False
        shown = BOUNDS_PATH / self.model_combo.currentText().lower()
        return bool(prior.bounds_picker.key()) and Path(prior.bounds_picker.base_path) != Path(shown)

    def _fill_pressed(self) -> None:
        """Fill the forms of steps 1 to 3 from the plan shown, read and checked again now, or return them to their usual
        values with "No plan" shown: asked first when a value a person typed or that was put back would be replaced,
        "Keep what is there" preselected. Nothing while a run is live."""
        screen, name = self._screen, self.plan_choice.currentData()
        if controller().is_live() or screen is None:
            return
        self.in_use.say(None)
        if name is None:
            values, source = {}, plans.PlanSource("usual", "", None, "", self.now(), {})
            title, sentence, answer = USUAL_TITLE, USUAL_TEXT, USUAL_ANSWER
        else:
            self._read_the_plan_shown()
            if not isinstance(self._shown_plan, engine_plans.Plan):
                self.redraw()
                return
            plan = self._shown_plan
            values = {pair: value for pair, value in plan.values.items() if pair[0] in plans.FILLED}
            source = plans.PlanSource("plan", plan.name, plan.path, plan.sha256, self.now(), values)
            title = FILL_TITLE.format(plan.name)
            sentence = (FILL_ONE_TEXT.format(plan.name) if len(values) == 1
                        else FILL_TEXT.format(plan.name, len(values)))
            answer = FILL_ANSWER
        replaced = plans.would_replace(screen, values)
        if replaced and not ask(self, title, sentence, KEEP_WHAT_IS_THERE, answer, items=replaced):
            return
        self._gone = None                        # the plan whose file went is no longer looked for
        plans.fill_forms(screen, values, source)

    def _save_pressed(self) -> None:
        """Save the settings that differ from their usual values as a plan of the model shown, under a name typed and
        judged as it is typed, never over a plan of that name without asking, and never a plan the window's own checks
        would refuse. The plan saved is then the plan in use, and nothing on the forms changes; a write that fails is
        said, and changes nothing. Nothing while a run is live."""
        screen = self._screen
        if controller().is_live() or screen is None:
            return
        self.in_use.say(None)
        values, remarks, wrong = self._to_save()
        try:
            if wrong:
                raise engine_plans.PlanRefusal("", tuple(engine_plans.PlanProblem(None, None, None, said)
                                                         for said in wrong))
            text = engine_plans.plan_text(values, "", remarks)
            checked = plans.window_problems(values, bounds_folder=BOUNDS_PATH / self.model_combo.currentText().lower())
            if checked:
                raise engine_plans.PlanRefusal("", tuple(checked))
        except engine_plans.PlanRefusal as refused:
            count = f"{len(refused.problems)} problem{'' if len(refused.problems) == 1 else 's'}"
            ask(self, SAVE_TITLE, CANNOT_SAVE_TEXT.format(count), OK_TEXT, marker="err",
                items=[problem.message for problem in refused.problems])
            return
        fields_of = {plans.HEADING_OF[page.step]: page.plan_fields() for page in screen.step_pages()}
        items = [WRITTEN_ITEM.format(fields_of[heading][name].label_text(), name,
                                     plans.value_words(fields_of[heading][name], value))
                 for (heading, name), value in values.items()]
        beneath = []
        if screen.prior_panel.plan_fields()["bounds"].value() is plans.TYPED:
            beneath.append(TYPED_BOUNDS_TEXT)
        elif self._bounds_of_another_model():
            beneath.append(OTHER_BOUNDS_TEXT)
        pending = [page.plan_fields()[selection.key].label_text() for page in screen.step_pages()
                   for selection in page.selection_rows()
                   if selection.key in page.plan_fields() and selection.changed()]
        if pending:
            beneath.append(NOT_APPLIED_SAVED_TEXT.format(", ".join(pending)))
        name = ask_text(self, SAVE_TITLE, SAVE_TEXT, SAVE_LABEL, SAVE_ANSWER,
                        check=lambda typed: None if plans.NAME_RULE.fullmatch(typed) else plans.NAME_TEXT,
                        items=items, beneath=" ".join(beneath) or None)
        if not name:
            return
        folder = self._plans_folder()
        there = plans.taken(folder, name)
        if there is not None:
            try:
                changed = texts.date_text(datetime.fromtimestamp(there.stat().st_mtime).astimezone())
            except OSError:
                changed = ""
            if not ask(self, REPLACE_TITLE.format(there.stem), REPLACE_TEXT.format(there.name, changed), KEEP_PLAN,
                       REPLACE_ANSWER, loses_work=True):
                return
        try:
            path = plans.save_plan(folder, name, text)
        except OSError as error:
            reason = error.strerror or error
            ask(self, SAVE_TITLE, NOT_WRITTEN_TEXT.format(f"{name}.txt", reason), OK_TEXT, marker="err")
            return
        source = plans.PlanSource("plan", name, path, plans.file_sha(path, {}) or "", self.now(), dict(values))
        screen.take_plan(source)
        self._read_the_plans()
        self.plan_choice.setCurrentIndex(max(0, self.plan_choice.findData(name)))
        self.own_line(SAVED_TEXT.format(name))

    def _open_pressed(self) -> None:
        plans.open_folder(self._plans_folder())

    def read_disk(self) -> None:
        """The model's plans, listed again, and the plan the list shows, read again."""
        self._read_the_plans()

    @reported(QWidget.showEvent)
    def showEvent(self, event):
        """Shown again with nothing owed, the page is drawn again: a page of another step may have changed what a save
        would write."""
        reads = self.stale
        super().showEvent(event)
        if not reads:
            self.redraw()

    # ── what is remembered ──
    def restore_settings(self, qs) -> None:
        """The model and the kind last applied, put back as choices and marked kept from last time until they are
        touched. Nothing else is read, and nothing is applied: a model no longer listed, or a kind that cannot be read,
        leaves its row as it opens."""
        rows = self.form.rows
        model = settings.get_str(qs, MODEL_KEY, "")
        known = self.model_combo.findData(model) >= 0
        if known:
            self.model_combo.setCurrentText(model)
        self._on_model_changed(self.model_combo.currentText())
        raw = str(qs.value(KIND_KEY, "")).strip().lower()
        kind = {"1": True, "true": True, "0": False, "false": False}.get(raw)
        if kind is not None:
            self.kind_choice.setCurrentIndex(self.kind_choice.findData(kind))
        rows["model"].kept, rows["chi_mode"].kept = known, kind is not None
