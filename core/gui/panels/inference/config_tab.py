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
"""
import math
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLineEdit

from core import cli, config, registry
from core.config import CHI_K_MAX, CHI_MIN_CYCLES, VALID_LABELS, VALID_MODELS
from core.Helpers import file_manager
from core.refusals import Refusal, default_clause

from ... import chain, keys, settings, style
from ...remembered import KIND_KEY, MODEL_KEY
from ...session import ConfigDraft
from ...step_facts import NOT_HELD, Fact, SelectionRow, held_value, model_facts, model_in_use, units_words
from ...widgets.ask import ask
from ...widgets.choice import Choice
from ...widgets.labeled_inputs import FloatField, IntField
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


class ConfigPanel(StepPage):
    """Step 1: a model and its options, held as a draft once applied. ``model_combo`` lists the models; ``kind_choice``,
    ``rot_choice`` and ``units_choice`` are the other choices, and ``units_text`` the units typed here."""

    reads_the_disk = False

    def __init__(self, screen, parent=None):
        super().__init__(screen, TITLE, SENTENCE, step="model", parent=parent)
        self._units_read: dict = {}
        self.model_combo = _ModelChoice()
        self.model_combo.addItems(VALID_MODELS)
        self.model_combo.setCurrentText("NADROWSKI")
        self.model_combo.currentIndexChanged.connect(self._model_chosen)
        self.kind_choice = _choice(KINDS)
        self.rot_choice = _choice(ROTATIONS)
        self.rot_choice.setCurrentIndex(self.rot_choice.findData(bool(config.REPARAM_ROTATE)))
        self.units_choice = _choice(UNIT_SOURCES)
        self.units_text = QLineEdit()
        self.units_text.setPlaceholderText("e.g. kHz ms nm pN")
        form = self.form
        form.add("model", self.model_combo, usual="NADROWSKI", help_text=HELP["model"])
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
        self.kind_choice.currentIndexChanged.connect(self._kind_chosen)
        self.units_choice.currentIndexChanged.connect(self._units_chosen)
        self.restore_settings(settings.settings())
        self._kind_chosen()
        self._units_chosen()
        self.refresh_local_gates()

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
        """The model's units file as the units line says it, read once for each model: its words, its path from the
        resources folder, and why it could not be read."""
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
                             lambda value: self._units_said(value, model)))

    def plan_fields(self) -> dict:
        return {key: self.form.rows[key] for key in ("chi_mode", "chi_n_freqs", "chi_k_pad", "chi_max_cycles",
                                                      "reparam_rotate", "units")}

    def facts_from_form(self) -> tuple:
        """Each rule broken, under its field and as a fact; where a value put back came from; that the model cannot be
        used, when it cannot; and, for a model of your own it can use, that its checks are not tuned."""
        found = self.problems()
        for key, row in self.form.rows.items():
            if row.widget is None:
                continue
            row.set_line("err" if key in found else "", found[key][0] if key in found else "")
            row.set_origin(KEPT if row.kept else "")
        model = self.model_combo.currentText()
        own = ((Fact("info", CALIBRATION_TEXT.format(keys.place_words("checks"))),)
               if registry.is_user_model(model) and not self._unusable(model) else ())
        return model_facts(usable_reason=self._unusable(model),
                           field_facts=(*(Fact("err", line, SETTING_BLOCKS) for line, _said in found.values()), *own))

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
        """A model chosen, by a person or by the list read again: its units line, and the box of typed units filled from
        its units file, so that switching to typed units starts from what the file declares."""
        self.form.rows["model"].kept = False
        words, where, trouble = self._units_of(model)
        self.units_line.set_text(UNITS_FROM.format(words, where) if words else NO_UNITS_FILE.format(trouble))
        self.units_text.setText(words)
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
        draft = ConfigDraft(
            model=model, labels=VALID_LABELS[VALID_MODELS.index(model)], state_dep_drift=registry.state_dep_drift(model),
            units_override=values["units"], chi_mode=self._chi(), chi_n_freqs=values["chi_n_freqs"], chi_f0=None,
            chi_freq_bounds=None, chi_k_pad=values["chi_k_pad"], chi_max_cycles=values["chi_max_cycles"],
            reparam_rotate=self.rot_choice.currentData())
        held = self._screen.session_contents()
        if held and not ask(self, QUESTION, QUESTION_TEXT.format("; ".join(held)), KEEP, APPLY_ANSWER):
            return
        self._screen.new_draft(draft)
        if by_keyboard and not self.next_button.isHidden():
            self.next_button.setFocus(Qt.TabFocusReason)

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
