"""Step 3, Training: the training that makes the simulation set and fits the network to it, set up, named and started;
or a posterior saved before, loaded.

The step holds a posterior, and holds it only when a run returns one: the press hands the run the configuration and the
prior the session holds, and the posterior held before stays in use through the run. A run refused, failed, stopped or
withdrawn changes nothing that is held, and the name and the note stand as typed.

Two views, each with its one main button: the training, and "Use a saved posterior", which lists the saved posteriors,
says of the one selected what its record holds and compares the prior it was trained with to the prior in use, and
loads it with the prior in use. A posterior from a narrowing round asks first, "Cancel" preselected, and the consent
holds for that load alone. Loading while a posterior is held asks no question about the posterior held: it goes only
when the loaded one comes back.

The form holds what the training stage reads, each as an argument of its call: the batches and the simulations of each,
the network, whether a saved simulation set is continued, and, folded, the fit's settings, how often the simulations are
saved and the rotation's three. A setting a training will not read is shown as such and is neither judged nor passed:
the rotation's three while the rotation is off or the saved set's own is reused. While the simulations are not saved the
choice of continuing stands at its first choice, greyed, and that choice is what is passed. "Before you start" says what
the run uses, what it makes, what it does with the saved sets, each set of the model it would not continue with what it
differs in by the names on screen, how long it should take and the memory it may use.

A training that kept trainings match asks first, "Cancel" preselected, naming each with where it stopped, and starts a
new one only on its answer, which leaves each kept training as it is. Else a training that would simulate from
zero while saved sets of its model exist asks first, "Cancel" preselected, listing each set with what it differs in;
with "Never" a set these very settings would continue heads the list. A yes hands the stage the consent to start a new
set; so does a yes to the same question asked again when the stage refuses a new set it finds one setting away from
another, whose own sentence is then not shown. Never both questions.

Nothing is read while a run is live: the saved simulation sets, the kept trainings, the run history, the saved
posteriors' records and the records of earlier trainings are read by the page's reading alone, and the list of saved
posteriors is read again and moved to a posterior that came into use once its run has ended, whichever page's run
brought it: this page's, or a narrowing round's. The batches and the
simulations per batch are remembered as the text of their boxes once a training has started with them; the posterior
that comes into use is remembered as it comes, and is put back as the list's choice at the next launch.
"""
import math
from dataclasses import replace
from datetime import datetime

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from core import config, orchestrator
from core.artifacts import Accept, default_store
from core.refusals import Refusal, default_clause, require_at_least, require_positive

from ... import continuing, design, fields, fonts, plans, record_words, settings, style, texts
from ...memory_limit import computer, limit_words, memory_limit_in_force
from ...remembered import NUM_RUNS_KEY, POSTERIOR_KEY, RUN_SIZE_KEY, put_many
from ...reported import reported
from ...run import ends, history
from ...run.controller import controller
from ...run.model import RunSpec
from ...run.view import SelectableLines, WrappedWords
from ...step_facts import (CANCEL, CONTINUING, LOAD_POSTERIOR_WORDS, NO_POSTERIOR_TEXT, NO_PRIOR_WORDS, START_WORDS,
                           Fact, SelectionRow, budget_words, held_value, how_long_words, kept_words, load_facts,
                           narrowed_question, new_set_question, new_training_question, of_this_model,
                           posterior_in_use, saved_posterior_words, set_words, stopped_words, training_facts,
                           uses_words)
from ...widgets.artifact_picker import StorePicker
from ...widgets.ask import ask
from ...widgets.choice import Choice
from ...widgets.form import NameAndNote, TickedField
from ...widgets.labeled_inputs import FloatField, IntField
from .base import training_spec
from .help_text import HELP
from .prior_tab import bounds_in_use, recorded_file
from .resume import resume_call, resume_draft
from .step_page import StepPage

TITLE = "Training"
TO_LOAD = "Use a saved posterior instead…"
TO_TRAINING = "Set up a training instead…"
CHOOSE_POSTERIOR = "Choose a saved posterior"
SENTENCE = ("Makes the simulation set, then fits the network to it. The result is saved as a posterior: the trained "
            "network that turns an observation into parameter estimates with their uncertainty.")
#: "Continue or start new": what each choice says, and the policy the training stage is handed for it.
RESUME_CHOICES = (("Continue the saved simulation set if there is one", "auto"),
                  ("Only continue: do not start if there is nothing to continue", "require"),
                  ("Only start new: do not start if a saved simulation set would be continued", "never"))
NETWORK_TEXT = "Width and depth open at {} and {} each time PRISM starts. A plan can set them."
LIMIT_TEXT = "A limit of N lets the fit run N + 1 epochs."
LIMIT_KEPT_TEXT = "A fit that reaches the limit keeps the network of its last epoch, which may not be its best."
EXPECTED_TEXT = ("Used only to say when the fit should end. It never limits the fit. Left blank, PRISM takes the epochs "
                 "of the last finished training of this kind in Records.")
NO_EARLIER_TEXT = "No earlier training of this kind: the run's page will say only when the fit cannot end before."
NEVER_TEXT = ("No simulation set is kept or continued. A stop or a failure during simulation loses every simulation; a "
              "stop or a failure during the fit loses the fit, which cannot be tried again without simulating again; "
              "and the check 'Use of each part of the input' cannot be run on the posterior.")
REUSED_TEXT = "Not used: this training reuses the saved simulation set's rotation."
ROTATION_OFF_TEXT = "Not used: rotation is off."
PRIOR_IN_USE_TEXT = "Prior in use: {}"
NEW_POSTERIOR = "The new posterior"
GO_TO_CHECKS = "Go to Checks"
STOP_WORDS = "Stop training…"
KEPT = "kept from last time"
EPOCHS = "epochs"
SAVED_WITHOUT_NAME = "It was saved without its name: '{}' was taken while the run was going."
#: The words beside the main button while a field breaks a rule of the page.
SETTING_BLOCKS = "a setting cannot be used"
#: The hint under "Name" after "Continue from batch N" filled the forms.
EARLIER_NAME_TEXT = "The earlier run was named {}."
#: The name a kept training with none is asked for at Resume, and one whose name another posterior has taken since.
NAME_TITLE = "Name this training"
NAME_TEXT = "This kept training has no name yet. A posterior needs one."
NAME_TAKEN_TEXT = "The name '{}' is now taken by another posterior. This training needs another one."
#: The list of kept trainings: its heading, its two buttons, what a row says of a training a run of this window trains,
#: and Delete's question.
KEPT_HEAD = "Kept trainings"
RESUME_WORDS = "Resume"
DELETE_WORDS = "Delete"
RUNNING_TEXT = "Running: a run of this window is training it."
USES_TEXT = "It uses the settings it was kept with."
DELETE_TITLE = "Delete this kept training?"
DELETE_TEXT = "The {} epochs kept are deleted; the next training starts from its first epoch."
DELETE_ONE_TEXT = "The 1 epoch kept is deleted; the next training starts from its first epoch."
DELETE_UNCOUNTED_TEXT = "The epochs kept are deleted; the next training starts from its first epoch."
KEEP_IT, DELETE_IT = "Keep it", "Delete it"

# THE SETTINGS A TRAINING READS, in the order the stage checks them: the field key, which is also build_posterior's
# keyword but for the expected epochs, which only the run's page reads; and the rule's floor, an int for
# require_at_least, None for require_positive. ONE table, read by the line under each field as it is typed and at the
# press, so the two can never name different limits.
_RULES = (
    ("num_runs", 1),
    ("run_size_cap", 0),                     # a TYPED 0 = follow the hardware batch, as "Automatic" does
    ("hidden_features", 1),
    ("num_transforms", 1),
    ("learning_rate", None),
    ("stop_after_epochs", 1),
    ("max_num_epochs", 1),                   # "No limit" passes nothing
    ("expected_epochs", 1),                  # blank: the last training's, or none
    ("checkpoint_every", 0),                 # "Never" = 0
    ("fisher_m", 1),
    ("fisher_dz", None),
    ("fisher_points", 1),
)
#: The rotation's three: read only while the rotation is computed afresh.
_ROTATION = ("fisher_m", "fisher_dz", "fisher_points")
#: The budget's three: the reading of the saved simulation sets is made at what they hold.
_BUDGET = ("num_runs", "run_size_cap", "checkpoint_every")
_FLOORS = dict(_RULES)
#: The rows a blank leaves at their choice of nothing, which no rule refuses.
_BLANK_CHOSEN = ("expected_epochs",)

#: The rows only a training reads: its settings and the choice of continuing.
_TRAINING_ROWS = ("resume", *(key for key, _ in _RULES))

# Where the window shows this step's runs: Parameter inference, its step Training.
_PLACE = ("inference", "training")


def _check(key: str, value, minimum):
    """Run one setting's rule: require_at_least with its floor, or require_positive when the floor is None. Returns the
    value; raises Refusal(field=key) on a blank or an out-of-rule value."""
    return require_positive(key, value) if minimum is None else require_at_least(key, value, minimum)


def _rule_words(minimum) -> str:
    """The rule as the line under a field says it: the words the refusal's own message uses after "must be"."""
    return "greater than 0" if minimum is None else f"at least {minimum}"


def _shown(value) -> str:
    return f"{value:g}" if isinstance(value, float) else str(value)


def _quietly(read, *args, **kwargs):
    """What ``read`` answers, or None when it raises: a history written by hand may hold what no reading foresaw, and the
    page's reading never raises."""
    try:
        return read(*args, **kwargs)
    except Exception:                        # noqa: BLE001 -- what cannot be read says nothing
        return None


def _made(created) -> str:
    """When a record was made, as the window writes a moment: in local time; "" when its record says no moment."""
    try:
        return texts.date_text(datetime.fromisoformat(created).astimezone())
    except (TypeError, ValueError, OverflowError, OSError):
        return ""


def _plain(value):
    """``value`` as a run's history holds it: a name, a number, a truth value or nothing as it is; a list of them as a
    list; anything else by its words."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_plain(one) for one in value]
    return str(value)


def _label_of(kind: str, manifest) -> str:
    """A record's label as Records gives it, from its manifest: its name, or its kind and when it was made."""
    try:
        made = datetime.fromisoformat(manifest.created).astimezone()
    except (TypeError, ValueError, OverflowError, OSError):
        made = None
    return record_words.record_label(kind, getattr(manifest, "name", "") or "", made)


class PosteriorPanel(StepPage):
    """Step 3. The form's rows are read by their field keys; ``resume_choice`` is "Continue or start new"; the line of
    the prior in use stands in the block for the run. ``post_picker`` lists the saved posteriors; ``view`` is
    "training" or "load"."""

    def __init__(self, screen, parent=None):
        super().__init__(screen, TITLE, SENTENCE, step="training", parent=parent)
        self.view = "training"
        self._record: dict = {}              # the saved posterior selected, as the last reading read its record
        self._matches = ()                   # the kept trainings the last reading found that match the form
        self._reading = None                 # the saved simulation sets, as the last reading found them at its budget
        self._never_reading = None           # with "Never", the same at the built-in cadence: which sets exist
        self._sets_said: dict = {}           # each saved set the last reading found, by its name: its words
        self._prior_labels: dict = {}        # each prior those sets were made with, by its fingerprint: its label
        self._per_batch = self._per_epoch = None  # the run history's paces, for this size
        self._expected = None                # the epochs the last finished training of this kind ran, and its words
        self._looked = False                 # whether the last reading looked for that training in Records
        self._last = self._measured = None   # the last run on this computer, and the last that measured its batches
        self._history_read = False           # whether a reading has read the run history yet
        self._made = None                    # the posterior a run of this page returned, until the run has ended
        self._handed = None                  # the name and the note this page's last training was handed
        self._moved_to = ""                  # the posterior the list was last moved to at a run's end, by its id
        self._lists_wait = False             # the reading after the typing stops reads no kept training
        self._resuming = None                # the kept training a Resume is starting, until its run is dispatched
        self._last_resumed = None            # the id of the kept training this page's last Resume started
        self._kept_rows: list = []           # the list's rows as drawn: (offer, its widgets)
        self._said_on_row: dict = {}         # a refusal said on a kept training's row, by its id, until the next reading
        self.switch_button = QPushButton()
        self.switch_button.clicked.connect(self._switch)
        self.head.add_action(self.switch_button)
        self.post_picker = StorePicker("posterior", parent=self, keeps_selection=True, hint=CHOOSE_POSTERIOR)
        self.resume_choice = Choice()
        for words, policy in RESUME_CHOICES:
            self.resume_choice.addItem(words, policy)
        self._saves = TickedField(IntField(config.TRAINING_CHECKPOINT_EVERY, grouped=True), "Never", 0)
        # Heard before its row hears it: the choice of continuing is fixed before the page judges the form.
        self._saves.changed.connect(self._saves_changed)
        form = self.form
        form.add("posterior", self.post_picker.combo)
        self.posterior_line = form.add_sentence("", under="posterior")
        self.posterior_line.setVisible(False)
        form.add("num_runs", IntField(config.TRAINING_NUM_RUNS, grouped=True), usual=config.TRAINING_NUM_RUNS,
                 help_text=HELP["num_runs"])
        form.add("run_size_cap", TickedField(IntField(config.TRAINING_RUN_SIZE, grouped=True), "Automatic", 0),
                 usual=config.TRAINING_RUN_SIZE, help_text=HELP["run_size"])
        form.add("hidden_features", IntField(config.NSF_HIDDEN_FEATURES, grouped=True), usual=config.NSF_HIDDEN_FEATURES,
                 help_text=HELP["flow_hidden"])
        form.add("num_transforms", IntField(config.NSF_NUM_TRANSFORMS, grouped=True), usual=config.NSF_NUM_TRANSFORMS,
                 help_text=HELP["flow_transforms"])
        form.add_sentence(NETWORK_TEXT.format(config.NSF_HIDDEN_FEATURES, config.NSF_NUM_TRANSFORMS),
                          under="num_transforms")
        form.add("resume", self.resume_choice, usual="auto")
        self.never_line = form.add_sentence(NEVER_TEXT, under="resume")
        self.never_line.setVisible(False)
        form.add("learning_rate", FloatField(config.TRAINING_LEARNING_RATE), usual=config.TRAINING_LEARNING_RATE,
                 folded=True, help_text=HELP["flow_lr"])
        form.add("stop_after_epochs", IntField(config.TRAINING_STOP_AFTER_EPOCHS, grouped=True),
                 usual=config.TRAINING_STOP_AFTER_EPOCHS, folded=True, help_text=HELP["flow_patience"])
        form.add("max_num_epochs", TickedField(IntField(None, grouped=True), "No limit", None), EPOCHS, usual=None,
                 folded=True)
        form.add_sentence(LIMIT_TEXT, under="max_num_epochs")
        form.add_sentence(LIMIT_KEPT_TEXT, under="max_num_epochs")
        form.add("expected_epochs", IntField(None, grouped=True), EPOCHS, usual=None, folded=True)
        form.add_sentence(EXPECTED_TEXT, under="expected_epochs")
        self.no_earlier_line = form.add_sentence(NO_EARLIER_TEXT, under="expected_epochs")
        self.no_earlier_line.setVisible(False)
        form.add("checkpoint_every", self._saves, usual=config.TRAINING_CHECKPOINT_EVERY, folded=True)
        form.add("fisher_m", IntField(config.REPARAM_FISHER_M, grouped=True), usual=config.REPARAM_FISHER_M,
                 folded=True, help_text=HELP["fisher_m"])
        form.add("fisher_dz", FloatField(config.REPARAM_FISHER_DZ), usual=config.REPARAM_FISHER_DZ, folded=True,
                 help_text=HELP["fisher_dz"])
        form.add("fisher_points", IntField(config.REPARAM_FISHER_POINTS, grouped=True),
                 usual=config.REPARAM_FISHER_POINTS, folded=True, help_text=HELP["fisher_points"])
        self.rotation_line = form.add_sentence("", under="fisher_points")
        self.rotation_line.setVisible(False)
        form.set_side_heading(NEW_POSTERIOR)
        self._names = NameAndNote("posterior", required=True)
        self.use_names(self._names)
        self.prior_row = form.add_text("prior", "", "", side=True)
        self.kept_block = QWidget()
        self._kept_lines = QVBoxLayout(self.kept_block)
        self._kept_lines.setContentsMargins(0, 0, 0, 0)
        self._kept_lines.setSpacing(design.scaled(design.GAP["head_table"]))
        self.kept_head = style.set_class(QLabel(KEPT_HEAD), cls="head")
        self.kept_head.setFont(fonts.inter("head"))
        self._kept_lines.addWidget(self.kept_head)
        self.kept_block.setVisible(False)
        self._body.insertWidget(self._body.indexOf(self.in_use) + 1, self.kept_block)
        self.restore_settings(settings.settings())
        self._apply_view("training")
        self.refresh_local_gates()

    # ── what the step holds ──
    def _held(self) -> tuple:
        """The configuration and the prior a training is made with, or ``(None, None)``: the one is held only with the
        other, and a session that holds one alone holds nothing a training can be made with."""
        session = getattr(self._screen, "session", None)
        cfg, prior = getattr(session, "cfg", None), getattr(session, "inf_prior", None)
        return (cfg, prior) if cfg is not None and prior is not None else (None, None)

    def _never(self) -> bool:
        """Whether the simulations of this training are not saved: "Never", or a typed 0."""
        return self._saves.value_or_none() == 0

    def _resume(self) -> str:
        """The policy "Continue or start new" hands the stage: always the first while the simulations are not saved."""
        return "auto" if self._never() else (self.resume_choice.currentData() or "auto")

    def _rotation_unused(self) -> str:
        """Why the rotation's three settings are not read by this training, in a sentence; "" when they are: while the
        rotation is off in the configuration held, and while the reading that stands says the saved set is continued,
        whose own rotation is reused."""
        cfg = self._held()[0]
        rotate = getattr(cfg, "reparam_rotate", None)
        if rotate is not None and not rotate:
            return ROTATION_OFF_TEXT
        if not self._never() and getattr(self._standing_reading(), "checkpoint", None) in CONTINUING:
            return REUSED_TEXT
        return ""

    def _ruled(self) -> tuple:
        """The settings the press reads, by field key, in the table's order: the rotation's only while they are used."""
        unused = bool(self._rotation_unused())
        return tuple((key, floor) for key, floor in _RULES if not (unused and key in _ROTATION))

    def _budget(self) -> "tuple[int, int, int] | None":
        """The batches, the simulations per batch and the batches between two saves the form holds, each passing its
        rule; None while any does not. Judged from those three rows alone, which no other setting changes."""
        rows = self.form.rows
        if any(self._row_problem(key, _FLOORS[key]) is not None for key in _BUDGET):
            return None
        values = tuple(rows[key].value() for key in _BUDGET)
        return None if None in values else values

    def _standing_reading(self):
        """The last reading of the saved simulation sets while the budget it was made at can be read: None while a box
        of the budget is out of its rule. The facts and the rotation's three both stand on it."""
        return self._reading if self._budget() is not None else None

    # ── what a page answers ──
    def main_words(self) -> str:
        """"Load this posterior"; or, with "Only continue" and the saved set of these settings unfinished, "Continue from
        batch N" at the batches the reading found, and with it whole "Fit again on the saved simulations"; else "Start
        training"."""
        if self.view == "load":
            return LOAD_POSTERIOR_WORDS
        reading = self._standing_reading()
        if not self._never() and self._resume() == "require" and reading is not None:
            state = getattr(reading, "checkpoint", None)
            if state == "resume_partial":
                return continuing.CONTINUE_BUTTON.format(texts.number_text(getattr(reading, "batches_done", 0) or 0))
            if state == "resume_complete":
                return continuing.FIT_AGAIN_BUTTON
        return START_WORDS

    def nothing_to_apply(self) -> bool:
        """The load view holds the posterior selected: a load would change nothing."""
        return (self.view == "load" and getattr(self.session, "posterior", None) is not None
                and not any(row.changed() for row in self.selection_rows()))

    def selection_rows(self) -> tuple:
        """In the load view, the saved posterior selected, against the posterior held."""
        if self.view != "load":
            return ()
        posterior = getattr(self.session, "posterior", None)
        return (SelectionRow("posterior", lambda: held_value(posterior, "id"), self.post_picker.key,
                             self._posterior_words),)

    def _posterior_words(self, value) -> str:
        found = self.post_picker.combo.findData(value)
        if found >= 0:
            return self.post_picker.combo.itemText(found)
        return record_words.label_of(getattr(self.session, "posterior", None)) or str(value)

    def plan_fields(self) -> dict:
        return {key: self.form.rows[key] for key, _ in _RULES}

    def plan_read(self, applied: bool) -> dict:
        """What a training reads of this step's form: its settings, the rotation's three only while they are read. A
        load reads none of them."""
        if applied or self.view == "load":
            return {}
        return {key: self.form.rows[key].value() for key, _ in self._ruled()}

    def in_use_words(self) -> tuple:
        posterior = getattr(self.session, "posterior", None)
        if posterior is None:
            return "", NO_POSTERIOR_TEXT
        manifest = getattr(posterior, "manifest", None)
        body, made_for = getattr(manifest, "body", None), getattr(manifest, "config", None)
        parents = getattr(manifest, "parents", None)
        prior = self._held()[1]
        trained_with = parents.get("prior") if isinstance(parents, dict) else None
        named = ""
        if prior is not None and trained_with is not None and trained_with == getattr(prior, "id", None):
            named = getattr(prior, "name", None) or record_words.label_of(prior)
        body = body if isinstance(body, dict) else {}
        return posterior_in_use(record_words.label_of(posterior), body.get("training"), made_for, named,
                                narrowed=body.get("amortized") is False)

    def problems(self) -> dict:
        """Every rule the form breaks now, by field key in the order a press reads them: the line under the field, built
        from its row's label, and what the press says, the same with the value it found and the usual one. A setting the
        training will not read breaks none, and a blank expected epochs is a choice. The load view says none of them:
        its facts are its own."""
        found = {}
        for key, minimum in self._ruled():
            problem = self._row_problem(key, minimum)
            if problem is not None:
                found[key] = problem
        return found

    def _row_problem(self, key: str, minimum) -> "tuple[str, str] | None":
        """The rule the row ``key`` breaks now, as ``problems`` gives it; None for none."""
        row = self.form.rows[key]
        if key in _BLANK_CHOSEN and not row.widget.text().strip():
            return None
        said = row.reading()
        if said is not None:
            return said, (f"{said[:-1]}{default_clause(key)}." if said.endswith(" is blank.") else said)
        value = row.value()
        if value is None:
            return None
        if isinstance(value, float) and not math.isfinite(value):
            line = f"{row.label_text()} must be a finite number."
        else:
            try:
                _check(key, value, minimum)
                return None
            except Refusal:
                line = f"{row.label_text()} must be {_rule_words(minimum)}."
        return line, f"{line[:-1]}; got {_shown(value)}{default_clause(key)}."

    def facts_from_form(self) -> tuple:
        """The facts of the step as the form, the session and the last reading stand: the rules under their fields and
        what stands under them; what the run uses and makes; what it does with the saved sets; how long it takes; the
        memory limit and the last run; the other steps' changes not applied; the name and the note. In the load view,
        the saved posterior selected as the last reading read it, against the prior in use."""
        if self.view == "load":
            return self._load_facts()
        found = self.problems()
        unused = self._rotation_unused()
        for key, row in self.form.rows.items():
            if row.widget is None or key in ("name", "note"):
                continue
            if key in found:
                row.set_line("err", found[key][0])
            else:
                row.set_line("", "")
            row.set_origin(KEPT if row.kept else "")
        for key in _ROTATION:
            self.form.rows[key].widget.setEnabled(not unused)
        self.rotation_line.set_text(unused)
        self.rotation_line.setVisible(bool(unused))
        cfg, prior = self._held()
        reading = self._standing_reading()
        rows = self.form.rows
        batches = None if "num_runs" in found else rows["num_runs"].value()
        width = getattr(reading, "width", None)
        typed = rows["expected_epochs"].value() if "expected_epochs" not in found else None
        self.no_earlier_line.setVisible(self._looked and typed is None and self._expected is None)
        uses = budget = how_long = ""
        if cfg is not None:
            uses = uses_words(model=getattr(cfg, "model", None), bounds=bounds_in_use(cfg),
                              prior=getattr(prior, "name", None) or record_words.label_of(prior),
                              kind=getattr(cfg, "observation_mode", None), slots=getattr(cfg, "chi_k_pad", None),
                              rotate=getattr(cfg, "reparam_rotate", None))
            budget = budget_words(batches, width, getattr(cfg, "observation_mode", None),
                                  getattr(cfg, "chi_k_pad", None))
        if reading is not None and batches is not None:
            state, done = getattr(reading, "checkpoint", None), getattr(reading, "batches_done", 0) or 0
            to_do = 0 if state == "resume_complete" and not self._never() else (
                batches - done if state == "resume_partial" and not self._never() else batches)
            epochs = typed if typed is not None else (self._expected[0] if self._expected else None)
            how_long = how_long_words(per_batch=self._per_batch, per_epoch=self._per_epoch, batches=to_do,
                                      epochs=epochs)
        memory = limit_words(memory_limit_in_force())
        if self._history_read:
            memory = f"{memory} {ends.last_run_sentence(self._last, self._measured)}"
        field_facts = [Fact("err", line, SETTING_BLOCKS) for line, _said in found.values()]
        names = [fact for fact in (self._names.name_fact(), self._names.note_fact()) if fact]
        differences = {one.name: self._differs(one) for one in of_this_model(getattr(reading, "sibling_diffs", ()))}
        return training_facts(needs_prior=prior is None, uses=uses, budget=budget, reading=reading,
                              never=self._never(), resume=self._resume(), batches=batches, differences=differences,
                              how_long=how_long, memory=memory, others_pending=self._others_pending(),
                              name_facts=names, field_facts=field_facts, plan_lines=self.plan_lines())

    def _load_facts(self) -> tuple:
        """The facts of the load view: what blocks a load, and the prior the posterior selected was trained with
        against the prior in use, as the last reading read its record."""
        for key, row in self.form.rows.items():
            if row.widget is not None and key not in ("name", "note"):
                row.set_origin(KEPT if row.kept else "")
        prior = self._held()[1]
        record = self._record if self._record.get("key") == self.post_picker.key() else {}
        return load_facts(needs_prior=prior is None, chosen=bool(self.post_picker.key()),
                          saved=self.post_picker.has_entries(), error=record.get("error", ""),
                          record_model=record.get("model"), model=getattr(self._held()[0], "model", None),
                          trained_with=record.get("trained_with"), trained_with_label=record.get("prior_label", ""),
                          in_use=getattr(prior, "id", None),
                          in_use_label=getattr(prior, "name", None) or record_words.label_of(prior),
                          others_pending=self._others_pending(), plan_lines=self.plan_lines())

    def _differs(self, one) -> tuple:
        """What the saved set ``one`` differs in, line by line, by the names on screen: its limits by the parameters of
        the configuration held, its prior by the label the last reading found for it."""
        cfg = self._held()[0]
        names = [*(getattr(cfg, "params_dict", None) or {}), *(getattr(cfg, "rescale_params", None) or {})]
        return fields.difference_lines(getattr(one, "fields", ()), prior_label=self._prior_labels.get, param_keys=names)

    # ── the reading of the disk ──
    def read_disk(self) -> None:
        """The run history (the last run on this computer and the last that measured its batches); and, while a prior is
        held and the budget passes its rules, the saved simulation sets at this budget, the run history's paces for this
        size, and the epochs the last finished training of this kind ran: only then has it looked for one."""
        if not self._lists_wait and self._screen is not None and hasattr(self._screen, "read_offers"):
            self._said_on_row = {}
            self._screen.read_offers()
        if self.view == "load":
            self._read_the_record()
            return
        folder, here = controller().folder(), computer()
        self._last, self._measured = history.last_on(folder, here), history.last_measured_on(folder, here)
        self._history_read = True
        self._reading = self._never_reading = self._per_batch = self._per_epoch = self._expected = None
        self._sets_said, self._prior_labels, self._matches = {}, {}, ()
        self._looked = False
        cfg, prior = self._held()
        budget = self._budget()
        if prior is None or budget is None:
            return
        num_runs, cap, cadence = budget
        self._reading = orchestrator.training_reading(cfg, prior, num_runs=num_runs, run_size_cap=cap,
                                                      checkpoint_every=cadence)
        if self._never():
            # The simulations are not saved, so the reading reads no set: the sets that exist are read at the cadence
            # the saves are usually made at, for the question a start asks.
            self._never_reading = orchestrator.training_reading(
                cfg, prior, num_runs=num_runs, run_size_cap=cap,
                checkpoint_every=self.form.rows["checkpoint_every"].usual())
        self._read_the_sets(prior)
        if getattr(self._reading, "checkpoint", None) == "resume_complete":
            # A kept training matches only a set of these settings that is whole: only then can one match.
            values = self._stage_values()
            self._matches = orchestrator.matching_fits(
                cfg, prior, num_runs=num_runs, run_size_cap=cap, checkpoint_every=cadence,
                **{key: values.get(key) for key in ("hidden_features", "num_transforms", "learning_rate",
                                                    "stop_after_epochs", "max_num_epochs")})
        width = getattr(self._reading, "width", None)
        if width:
            self._per_batch = _quietly(history.estimate, folder, word="Training", per="batch",
                                       size={"per_batch": width})
            network = tuple(self.form.rows[key].value() for key in ("hidden_features", "num_transforms"))
            if None not in network and not self.problems().keys() & {"hidden_features", "num_transforms"}:
                self._per_epoch = _quietly(history.estimate, folder, word="Training", per="epoch",
                                           size={"network": network, "simulations": num_runs * width})
        try:
            spec = self.run_spec(orchestrator.build_posterior, cfg, prior, None, True, **self._stage_values())
        except Exception:                    # noqa: BLE001 -- a configuration that cannot say its kind has no last training
            return
        # The last finished training's alone: the epochs typed are the form's, read at each drawing.
        self._expected = controller().expected_epochs(replace(spec, expected_epochs=None))
        self._looked = True

    def _timer_fired(self) -> None:
        """The reading after the typing stops reads no kept training: typing makes none."""
        self._lists_wait = True
        try:
            super()._timer_fired()
        finally:
            self._lists_wait = False

    def kept_offers(self) -> tuple:
        """The kept trainings the list shows, as the screen last read them, newest first."""
        return tuple(offer for offer in (getattr(self._screen, "offers", tuple)() or ())
                     if offer.kind in ("resume", "round"))

    def redraw(self) -> None:
        """What every page draws, then the list of kept trainings as last read."""
        super().redraw()
        self._draw_the_kept()

    def _draw_the_kept(self) -> None:
        """One row for each kept training: its label, what it says, why it cannot be resumed or the command that resumes
        a kept narrowing round, then "Resume" and "Delete"; a kept training a live run of this window trains reads
        "running" and offers nothing. Reads nothing; rows made again only when the list's trainings change, and the Tab
        key's order then with them."""
        offers, live = self.kept_offers(), controller().is_live()
        training = (getattr(getattr(controller().model, "spec", None), "kwargs", None) or {}).get("resume_fit") \
            if live else None
        if [offer for offer, _parts in self._kept_rows] != list(offers):
            for _offer, parts in self._kept_rows:
                # Hidden and taken out of the list, never set loose: a row set loose stands in no window, and the
                # keyboard could be handed on to its buttons until it is deleted.
                parts["row"].hide()
                self._kept_lines.removeWidget(parts["row"])
                parts["row"].deleteLater()
            self._kept_rows = [(offer, self._kept_row(offer)) for offer in offers]
            for _offer, parts in self._kept_rows:
                self._kept_lines.addWidget(parts["row"])
            self._a_stop_joined()
        for offer, parts in self._kept_rows:
            running = offer.kept == training
            said = self._said_on_row.get(offer.kept, "")
            parts["why"].setText(RUNNING_TEXT if running else said or offer.why_not)
            parts["why"].setVisible(running or bool(said or offer.why_not))
            parts["resume"].setVisible(offer.kind == "resume" and not offer.why_not and not running and not said)
            parts["resume"].setEnabled(not live)
            parts["delete"].setVisible(not running)
            parts["delete"].setEnabled(not live)
        self.kept_block.setVisible(bool(self._kept_rows))

    def _kept_row(self, offer) -> dict:
        """The widgets of one row of the list."""
        row = QWidget()
        lines = QVBoxLayout(row)
        lines.setContentsMargins(0, 0, 0, 0)
        lines.setSpacing(design.scaled(design.GAP["field_message"]))
        # Every line wraps within the column, inside a long name or path where it must, so that no row widens the page.
        label, sentence = WrappedWords(), WrappedWords()
        why, uses = style.set_class(WrappedWords(), cls="quiet"), style.set_class(WrappedWords(), cls="quiet")
        label.setText(f"{offer.label}: {offer.how}")
        sentence.setText(offer.sentence)
        # Whatever the forms hold: a Resume runs on the kept training's own record.
        uses.setText(USES_TEXT if offer.kind == "resume" and not offer.why_not else "")
        uses.setVisible(bool(uses.text()))
        parts = {"row": row, "label": label, "sentence": sentence, "why": why, "uses": uses}
        for part in (label, sentence, uses, why):
            lines.addWidget(part)
        if offer.command:
            command = SelectableLines()
            command.setFont(fonts.fixed())
            command.setText(offer.command)
            command.setFocusPolicy(Qt.StrongFocus)
            lines.addWidget(command)
            parts["command"] = command
        buttons = QHBoxLayout()
        buttons.setContentsMargins(0, 0, 0, 0)
        parts["resume"] = QPushButton(RESUME_WORDS)
        parts["resume"].clicked.connect(lambda _checked=False, ident=offer.kept: self.resume_pressed(ident))
        parts["delete"] = QPushButton(DELETE_WORDS)
        parts["delete"].clicked.connect(lambda _checked=False, ident=offer.kept: self.delete_pressed(ident))
        buttons.addWidget(parts["resume"])
        buttons.addWidget(parts["delete"])
        buttons.addStretch(1)
        lines.addLayout(buttons)
        return parts

    def region_stops(self) -> list:
        """The list's own stops, in its order: each row's command, then its two buttons."""
        stops = []
        for _offer, parts in self._kept_rows:
            stops += [part for part in (parts.get("command"), parts["resume"], parts["delete"]) if part is not None]
        return stops

    def delete_pressed(self, ident: str) -> None:
        """"Delete" on a kept training's row: asked first, "Keep it" preselected; deleted from Records, the list read
        again, and the keyboard where the row stood: on the first button of the row now in its place, its "Resume", or
        its "Delete" for a row that offers no Resume, else the row before's, else the step's heading; whatever the row
        said before. A Delete the store refuses is said on the row, and the keyboard stays. Nothing while a run is
        live."""
        if controller().is_live():
            return
        offer = next((offer for offer in self.kept_offers() if offer.kept == ident), None)
        epochs = getattr(offer, "epoch", None)
        sentence = (DELETE_UNCOUNTED_TEXT if not isinstance(epochs, int) or epochs <= 0 else
                    DELETE_ONE_TEXT if epochs == 1 else DELETE_TEXT.format(texts.number_text(epochs)))
        if not ask(self, DELETE_TITLE, sentence, KEEP_IT, DELETE_IT, loses_work=True):
            return
        at = next((index for index, (row, _parts) in enumerate(self._kept_rows) if row.kept == ident), 0)
        self._said_on_row.pop(ident, None)      # what the row said before, a refused Resume's, is no refusal of this
        try:
            default_store().delete("fit", ident)
        except Exception as error:           # noqa: BLE001 -- said on the row, as any refusal of the store is
            self._said_on_row[ident] = str(getattr(error, "message", None) or error)
        self._screen.read_offers()
        if ident not in self._said_on_row:
            # A turn later: the rows the reading made again come into sight first.
            QTimer.singleShot(0, self, lambda: self._keyboard_after_delete(at))

    def _keyboard_after_delete(self, at: int) -> None:
        """The keyboard after a row's training was deleted, the row at ``at`` gone: where it stood in the list, on the
        first button of the row now in its place (its "Resume", or its "Delete" for a row that offers no Resume), else
        on the first button of the row before, else on the step's heading."""
        for index in (at, at - 1):
            if 0 <= index < len(self._kept_rows):
                parts = self._kept_rows[index][1]
                for button in (parts["resume"], parts["delete"]):
                    if not button.isHidden() and button.isEnabled():
                        button.setFocus(Qt.OtherFocusReason)
                        return
        self.help_button.setFocus(Qt.OtherFocusReason)

    def resume_pressed(self, ident: str) -> None:
        """"Resume" on a kept training's row: the screen's Resume, the same as Home's and a past view's."""
        self._screen.resume_kept(ident)

    def resume(self, kept, name: str = "") -> bool:
        """Start the kept training ``kept`` again, on its own record: the configuration and the prior the session holds,
        put in use from that record, and the call naming the kept training (and the name asked for, when its record holds
        none). True when its run was started."""
        cfg, prior = self._held()
        if prior is None or controller().is_live():
            return False
        self._resuming, self._last_resumed = kept, kept.id
        try:
            return self.dispatch(orchestrator.build_posterior, cfg, prior, None, True, **resume_call(kept, name),
                                 if_name_taken="unnamed", provide_fig_sink=True, on_result=self._on_posterior)
        finally:
            self._resuming = None

    def say_on_row(self, ident: str, sentence: str) -> None:
        """Say ``sentence`` on the kept training's row, in place of its Resume, until the list is next read."""
        self._said_on_row[ident] = sentence
        self.redraw()

    def continue_run(self, offer, settings_: dict) -> None:
        """The forms filled from a training's history, for "Continue from batch N" or "Fit again on the saved
        simulations": "Only continue" chosen, the name and the note blank, the earlier run's name as the hint, and the
        page read, so that its main button says what the press does."""
        self.show_view("training")
        index = self.resume_choice.findData("require")
        if index >= 0:
            self.resume_choice.setCurrentIndex(index)
        self._names.name.clear()
        self._names.note.clear()
        earlier = settings_.get("name") if isinstance(settings_.get("name"), str) else ""
        self._names.set_hint(EARLIER_NAME_TEXT.format(earlier) if earlier else "")
        self.read_facts()

    def _read_the_record(self) -> None:
        """The saved posteriors listed again, keeping the one selected; and its record as the line under the list says
        it and the load's facts read it: when it was made and for which model, what it reads, its network, its
        simulations, the prior it was trained with by its label, the bounds file, and, from a narrowing round, the
        observation it was narrowed around by its label. What cannot be read is said, never raised."""
        self.post_picker.refresh()
        key = self.post_picker.key()
        self._record = {"key": key}
        self.posterior_line.set_text("")
        self.posterior_line.setVisible(False)
        if not key:
            return
        store = default_store()
        try:
            manifest = store.get("posterior", key)
            body = dict(manifest.body)
        except Exception as error:           # noqa: BLE001 -- a record that cannot be read is said, not raised
            self._record["error"] = f"The posterior cannot be read: {type(error).__name__}: {error}"
            return
        parents, made_for = dict(manifest.parents or {}), dict(manifest.config or {})
        training = body.get("training") if isinstance(body.get("training"), dict) else {}
        record = self._record
        record.update(model=made_for.get("model"), trained_with=parents.get("prior"),
                      label=_label_of("posterior", manifest), narrowed=body.get("amortized") is False, prior_label="",
                      observation="")
        for kind, ref, slot in (("prior", parents.get("prior"), "prior_label"),
                                ("observation", parents.get("observation") if record["narrowed"] else None,
                                 "observation")):
            if ref:
                try:
                    record[slot] = _label_of(kind, store.get(kind, ref))
                except Exception:            # noqa: BLE001 -- a record no longer in Records is said so
                    pass
        runs, size = training.get("n_runs"), training.get("run_size")
        simulations = runs * size if isinstance(runs, int) and isinstance(size, int) else None
        inputs = manifest.inputs if isinstance(manifest.inputs, dict) else {}
        self.posterior_line.set_text(saved_posterior_words(
            made=_made(manifest.created), model=made_for.get("model"), kind=body.get("mode"),
            network=(training.get("hidden_features"), training.get("num_transforms")), simulations=simulations,
            prior=record["prior_label"] or (NO_PRIOR_WORDS if parents.get("prior") else ""),
            bounds=recorded_file(inputs.get("bounds")), narrowed=record["narrowed"], observation=record["observation"]))
        self.posterior_line.setVisible(True)

    def _read_the_sets(self, prior) -> None:
        """How the question names each saved set of the model the readings found: by its label in Records, the prior it
        was made with by that prior's label, with when it was made; and the label of each prior those sets differ in.
        What cannot be read is named without it."""
        held = getattr(prior, "fingerprint", None)
        in_use = getattr(prior, "name", None) or record_words.label_of(prior)
        labels = {held: in_use} if held else {}
        sets = [*of_this_model(getattr(self._reading, "sibling_diffs", ())),
                *of_this_model(getattr(self._never_reading, "sibling_diffs", ()))]
        if not sets:
            self._prior_labels = labels
            return
        store = default_store()
        for one in sets:
            for field in getattr(one, "fields", ()) or ():
                if field.get("field") == "prior_fingerprint":
                    for value in (field.get("mine"), field.get("theirs")):
                        if value and value not in labels:
                            labels[value] = self._prior_label(store, value)
        self._prior_labels = labels
        for one in sets:
            made = ""
            try:
                made = _made(store.get("simulation", one.name).created)
            except Exception:                # noqa: BLE001 -- a set whose record cannot be read is named without when
                pass
            theirs = next((field.get("theirs") for field in getattr(one, "fields", ()) or ()
                           if field.get("field") == "prior_fingerprint"), held)
            label = record_words.record_label("simulation", "", None, prior_label=labels.get(theirs))
            self._sets_said[one.name] = set_words(label, made, one.batches, one.complete)

    @staticmethod
    def _prior_label(store, fingerprint) -> "str | None":
        """The label in Records of the prior whose fingerprint is ``fingerprint``; None when no prior there has it."""
        try:
            found = store.find_prior_by_fingerprint(fingerprint)
        except Exception:                    # noqa: BLE001 -- a store that cannot be read names no prior
            return None
        if found is None:
            return None
        try:
            made = datetime.fromisoformat(found.created).astimezone()
        except (TypeError, ValueError, OverflowError, OSError):
            made = None
        return record_words.record_label("prior", found.name, made)

    def _from_zero(self) -> "tuple | None":
        """The question a start asks before it simulates from zero while saved sets of its model exist, as
        ``new_set_question`` gives it, from the last reading; None when it would continue its own set, or when no set of
        its model is saved or could be read. With "Never" the reading at the usual cadence says which sets exist, and a
        set these very settings would continue heads the list."""
        never = self._never()
        reading = self._never_reading if never else self._standing_reading()
        if reading is None:
            return None
        state, own = getattr(reading, "checkpoint", None), ""
        if state in CONTINUING:
            if not never:
                return None
            held = self._held()[1]
            in_use = getattr(held, "name", None) or record_words.label_of(held)
            own = set_words(record_words.record_label("simulation", "", None, prior_label=in_use), "",
                            reading.batches_done, state == "resume_complete")
        sets = [(self._sets_said.get(one.name) or set_words(record_words.record_label("simulation", "", None), "",
                                                             one.batches, one.complete), self._differs(one))
                for one in of_this_model(getattr(reading, "sibling_diffs", ()))]
        if not own and not sets:
            return None
        return new_set_question(batches=self.form.rows["num_runs"].value(), never=never, own=own, sets=sets)

    def _stage_values(self) -> dict:
        """The settings the stage would be handed now, as far as they pass their rules: for the run's facts."""
        found, rows = self.problems(), self.form.rows
        return {key: rows[key].value() for key, _ in self._ruled()
                if key not in found and key != "expected_epochs" and rows[key].value() is not None}

    # ── the views ──
    def show_view(self, view: str) -> None:
        """Show "training", a training to set up and start, or "load", a saved posterior to load: each its rows and its
        main button, its facts read once the choice has settled."""
        self._apply_view(view)
        self._edited()

    def _apply_view(self, view: str) -> None:
        if view not in ("training", "load"):
            raise ValueError(f"no such view of the Training step: {view!r}")
        self.view = view
        load = view == "load"
        self.switch_button.setText(style.access_key(TO_TRAINING if load else TO_LOAD))
        rows = self.form.rows
        rows["posterior"].set_shown(load)
        for key in _TRAINING_ROWS:
            rows[key].set_shown(not load)
        for row in self._names.rows.values():
            row.set_shown(not load)
        self.form.side.setVisible(not load)
        self.names = None if load else self._names
        # The kept trainings under what the step holds while a training is set up, beside the choice to start one; in
        # the load view after the saved posteriors, which that view is for.
        self._body.removeWidget(self.kept_block)
        self._body.insertWidget(self._body.indexOf(self.form if load else self.in_use) + 1, self.kept_block)
        self._a_stop_joined()

    def region_after_the_form(self) -> bool:
        """The kept trainings stand after the form in the load view."""
        return self.view == "load"

    def _switch(self) -> None:
        self.show_view("training" if self.view == "load" else "load")

    def set_controls_enabled(self, enabled: bool) -> None:
        """A run locks the form, the foot's two buttons, and the way between the two views."""
        self.switch_button.setEnabled(enabled)
        super().set_controls_enabled(enabled)

    # ── the choices ──
    def _saves_changed(self) -> None:
        """"Never", or a typed 0: no simulation set is continued, so "Continue or start new" stands at its first choice,
        greyed, with the reason under it."""
        never = self._never()
        if never:
            self.resume_choice.setCurrentIndex(0)
        self.resume_choice.setEnabled(not never)
        self.never_line.setVisible(never)

    # ── the press ──
    def _read_knobs(self) -> dict:
        """The settings the stage is handed, read through the rules: the first broken is refused, and nothing changes. The
        epoch limit only when one is typed, and the rotation's three only when they are read."""
        for key, (_line, said) in self.problems().items():
            raise Refusal(said, field=key)
        rows = self.form.rows
        return {key: rows[key].value() for key, _ in self._ruled()
                if key != "expected_epochs" and not (key == "max_num_epochs" and rows[key].value() is None)}

    def main_pressed(self) -> None:
        """Start the training on the configuration and the prior held, or load the posterior selected: nothing is held
        anew until the run returns its posterior. A training asks first when kept trainings match it, and else when it
        would simulate from zero beside saved sets of its model; never both. The batches and the simulations per batch
        are remembered once a training has started."""
        if self.view == "load":
            self._load()
            return
        cfg, prior = self._held()
        if prior is None:
            return                               # the facts say why, and the button is greyed with them
        try:
            knobs = self._read_knobs()
        except Refusal as refused:
            self._refusal(refused)
            return
        new_run = False
        if self._matches:
            title, sentence, items, beneath, other = new_training_question(
                [kept_words(kept.posterior_name or _label_of("fit", kept),
                            stopped_words(kept.state, kept.stopped_on_purpose, kept.stop_reason), kept.passes,
                            _made(kept.saved_at)) for kept in self._matches])
            if not ask(self, title, sentence, CANCEL, other, items=items, beneath=beneath):
                return
        else:
            question = self._from_zero()
            if question is not None:
                if not self._asked_from_zero(question):
                    return
                new_run = True
        self._start(knobs, new_run)

    def _load(self) -> bool:
        """Load the saved posterior selected with the configuration and the prior held, asked first when it comes from
        a narrowing round, the consent then handed for this load alone. True when its run was started."""
        cfg, prior = self._held()
        key = self.post_picker.key()
        if prior is None or not key:
            return False
        accept = Accept()
        record = self._record if self._record.get("key") == key else {}
        if record.get("narrowed"):
            title, sentence, beneath, other = narrowed_question(record.get("label", key), record.get("observation", ""))
            if not ask(self, title, sentence, CANCEL, other, beneath=beneath):
                return False
            accept = Accept(truncated=True)
        return self.dispatch(orchestrator.build_posterior, cfg, prior, key, False, accept=accept,
                             on_result=self._on_posterior)

    def load_record(self, ref: str) -> bool:
        """Load the saved posterior ``ref`` as "Load this posterior" does, for a caller of the window's own: the load
        view shown and read, so that the list holds a record saved since by another program, ``ref`` selected, the
        facts read, and the load started unless they block it, asking first when it comes from a narrowing round. True
        when its run was started. While a run is live it changes nothing: neither the view, nor the list, nor what is
        held."""
        if controller().is_live():
            return False
        self._by_keyboard = False                # no key pressed this load
        self.show_view("load")
        self.read_facts()
        self.post_picker.restore_key(ref)
        self.read_facts()
        if self.post_picker.key() != ref or not self.foot.main_button.isEnabled():
            return False
        return self._load()

    def _asked_from_zero(self, question) -> bool:
        """Ask the question before simulating from zero, "Cancel" preselected; True for the answer that starts."""
        title, sentence, items, beneath, other = question
        return ask(self, title, sentence, CANCEL, other, items=items, beneath=beneath)

    def _start(self, knobs: dict, new_run: bool) -> None:
        """Start the training on the configuration and the prior held, with ``knobs`` and the consent to start a new
        simulation set, ``new_run``; once it has started, the name and the note it was handed are kept, and the batches
        and the simulations per batch remembered."""
        cfg, prior = self._held()
        handed = (self._names.name_value(), self._names.note_value())
        started = self.dispatch(orchestrator.build_posterior, cfg, prior, None, True, name=handed[0], note=handed[1],
                                resume=self._resume(), new_run=new_run, if_name_taken="unnamed", provide_fig_sink=True,
                                on_result=self._on_posterior, **knobs)
        if started:
            self._handed = handed
            put_many([(NUM_RUNS_KEY, str(knobs["num_runs"])), (RUN_SIZE_KEY, str(knobs["run_size_cap"]))]
                     + self._screen.last_used(getattr(getattr(self.session, "draft", None), "model", None)))

    def _refusal(self, exc) -> None:
        """A refusal, as every page tells one; but the stage's refusal of a new simulation set beside a saved one a
        setting away is no box: the page reads again and asks the question before simulating from zero, the stage's
        sentence unshown, and a yes starts the training again with the consent. A field that now breaks its rule is told
        as any refusal is."""
        if getattr(exc, "field", None) == "resume_fit" and self._last_resumed:
            # A Resume refused (a race, a torn record, a process training it): said on the kept training's row,
            # beside its "Delete", and the list read again.
            said = str(getattr(exc, "message", None) or exc)
            self._screen.read_offers()
            self.say_on_row(self._last_resumed, said)
        if getattr(exc, "field", None) != "new_run" or controller().is_live() or self._held()[1] is None:
            super()._refusal(exc)
            return
        self.read_facts()
        try:
            knobs = self._read_knobs()
        except Refusal as refused:
            super()._refusal(refused)
            return
        question = self._from_zero() or new_set_question(batches=knobs["num_runs"], never=self._never())
        if self._asked_from_zero(question):
            self._start(knobs, True)

    def run_spec(self, fn, *args, **kwargs) -> RunSpec:
        """A training, which writes a posterior, its simulation set and its kept training, and may be tried again: with
        the epochs typed for its time, the plan in use and the settings it was started with. Or a posterior's load,
        which writes nothing, keeps no file and stops without a question (its fourth argument says which, as the stage
        reads it)."""
        if fn is not orchestrator.build_posterior:
            return super().run_spec(fn, *args, **kwargs)
        if args[3] and self._resuming is not None:
            kept = self._resuming
            record = getattr(kept, "config", None) or {}
            given = {key: record.get(key) for key in ("stop_after_epochs", "max_num_epochs", "hidden_features",
                                                      "num_transforms", "checkpoint_every")}
            given.update(kwargs)
            given["name"] = kwargs.get("name") or getattr(kept, "posterior_name", "") or ""
            return training_spec("Training", _PLACE, STOP_WORDS, args[0], given, narrowed=False,
                                 plan=None, settings=self._kept_settings(kept, given["name"]))
        if args[3]:
            row = self.form.rows["expected_epochs"]
            typed = None if "expected_epochs" in self.problems() else row.value()
            return training_spec("Training", _PLACE, STOP_WORDS, args[0], kwargs, narrowed=False,
                                 expected_epochs=typed, plan=self._screen.plan_entry() if self._screen else None,
                                 settings=self.settings_entry())
        return RunSpec("Posterior load", _PLACE, stop_button="Stop loading", loads_only=True, keeps_files=False,
                       stop_asks=False)

    def _kept_settings(self, kept, name: str) -> dict:
        """What a Resume is started with, as its run's history keeps it: the kept training's own record, never the
        forms: its model and its options, its bounds file and its prior, the name the run is called by (``name``: its
        record's, or the one the Resume asked for), and one row for each setting of the fit it was kept with."""
        record = getattr(kept, "config", None) or {}
        cfg, prior = self._held()
        rows = [{"step": "training", "name": key, "label": self.form.rows[key].label_text(), "value": _plain(record[key]),
                 "origin": "kept training", "plan": None, "usual": _plain(self.form.rows[key].usual())}
                for key, _ in _RULES if key in record]
        options = {name: _plain(record.get(name)) for name in ("chi_mode", "chi_k_pad", "chi_max_cycles",
                                                               "reparam_rotate")}
        # The units as the kept training's record names them: its units file, by path when not the model's own.
        options["units_override"] = _plain(resume_draft(kept).units_override)
        return {"model": _plain(record.get("model")), "chi_mode": _plain(record.get("chi_mode")), "options": options,
                "bounds": _plain(bounds_in_use(cfg)), "prior": _plain(getattr(prior, "id", None)), "resume": None,
                "name": name, "note": getattr(kept, "note", "") or "",
                "rows": rows, "resume_fit": getattr(kept, "id", None)}

    def settings_entry(self) -> dict:
        """What this training is started with, as its run's history keeps it: names and numbers only. The model and its
        options as step 1 holds them, the bounds file and the prior in use, the choice of continuing, the name and the
        note, and one row for each setting the training reads, with its label, its value, where the value came from, the
        value the source in use holds for it and its usual one."""
        session = self.session
        draft = getattr(session, "draft", None)
        cfg, prior = self._held()
        options = {name: _plain(getattr(draft, name, None))
                   for name in ("chi_mode", "chi_n_freqs", "chi_k_pad", "chi_max_cycles", "reparam_rotate",
                                "units_override")}
        source = getattr(self._screen, "plan_source", None)
        rows = [{"step": "training", "name": key, "label": row.label_text(), "value": _plain(row.value()),
                 "origin": plans.origin(row, source, kept=row.kept, heading=plans.HEADING_OF[self.step]),
                 "plan": _plain(plans.plan_value(source, plans.HEADING_OF[self.step], key)),
                 "usual": _plain(row.usual())}
                for key, row in ((key, self.form.rows[key]) for key, _ in self._ruled())]
        return {"model": _plain(getattr(draft, "model", None)), "chi_mode": _plain(getattr(draft, "chi_mode", None)),
                "options": options, "bounds": _plain(bounds_in_use(cfg)), "prior": _plain(getattr(prior, "id", None)),
                "resume": self._resume(), "name": self._names.name_value(), "note": self._names.note_value(),
                "rows": rows}

    def _on_posterior(self, payload) -> None:
        """The run returned its posterior, trained or loaded, while it is still live: it comes into use, and nothing is
        read until the run has ended."""
        self._made = payload
        self._screen.take_posterior(payload)

    def _run_ended(self, status) -> None:
        """A run has ended. One of this page's that returned its posterior, or any that ended with another posterior
        held than the one the list was last moved to, as a narrowing round's brings one: the list of saved posteriors
        read again and moved to the posterior held, when it is listed, so that the list says what the step holds. After
        a training of this page's, a word when the record came back under another name than the one its call asked for
        (the ended run's own, which "Continue" and "Resume" repeat), and the name and the note each emptied while it
        still holds what the call was handed, so that one typed since stands. Then the reading."""
        made, self._made = self._made, None
        held = getattr(made if made is not None else getattr(self.session, "posterior", None), "id", "") or ""
        if made is not None or (held and held != self._moved_to):
            self.post_picker.refresh()
            if self.post_picker.combo.findData(held) >= 0:
                self.post_picker.restore_key(held)
            self._moved_to = held
        if made is not None and not getattr(status, "loading", False):
            asked = getattr(status, "name", "") or ""
            if asked and getattr(made, "name", "") != asked:
                self.in_use.say(Fact("warn", SAVED_WITHOUT_NAME.format(asked)))
            name, note = self._handed or ("", "")
            if self._names.name_value() == name:
                self._names.name.clear()
                self._names.set_hint("")
            if self._names.note.text().strip() == note:
                self._names.note.clear()
        super()._run_ended(status)

    def refresh_local_gates(self) -> None:
        """What the step holds and what follows from it, the prior a training would use, and the way on to Checks once a
        posterior is in use."""
        prior = self._held()[1]
        self.prior_row.set_value(PRIOR_IN_USE_TEXT.format(getattr(prior, "name", None)
                                                          or record_words.label_of(prior)) if prior is not None else "")
        self.prior_row.set_shown(prior is not None)
        super().refresh_local_gates()
        held = getattr(self.session, "posterior", None) is not None
        self.set_next(GO_TO_CHECKS if held else "", "checks" if held else None)

    @reported(QWidget.showEvent)
    def showEvent(self, event):
        """Shown again with nothing owed, the page is drawn again, since what the other steps hold may have changed;
        the load view lists the saved posteriors again first, unless a run is live, so that one saved meanwhile by
        another program comes in. Nothing else is read."""
        reads = self.stale
        super().showEvent(event)
        if not reads:
            if self.view == "load" and not controller().is_live():
                self.post_picker.refresh()
            self.redraw()

    # ── what is remembered ──
    def restore_settings(self, qs) -> None:
        """The batches and the simulations per batch the last training started with, put back as their boxes' text,
        never read anywhere else, and the posterior that came into use last, put back as the list's choice, never
        loaded; each marked kept from last time until it is touched. A value that is no whole number, and a posterior
        no longer listed, is passed over."""
        posterior = settings.get_str(qs, POSTERIOR_KEY, "")
        self.post_picker.restore_key(posterior)
        self.form.rows["posterior"].kept = bool(posterior) and self.post_picker.key() == posterior
        for key, name in ((NUM_RUNS_KEY, "num_runs"), (RUN_SIZE_KEY, "run_size_cap")):
            raw = settings.get_str(qs, key, "").strip()
            try:
                value = int(raw)
            except ValueError:
                continue
            row = self.form.rows[name]
            row.set_value(value)
            row.kept = True
