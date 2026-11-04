"""Step 3, Training: the training that makes the simulation set and fits the network to it, set up, named and started.

The step holds a posterior, and holds it only when a training's run returns one: the press hands the run the
configuration and the prior the session holds, and the posterior held before stays in use through the run. A run
refused, failed, stopped or withdrawn changes nothing that is held, and the name and the note stand as typed.

The form holds what the training stage reads, each as an argument of its call: the batches and the simulations of each,
the network, whether a saved simulation set is continued, and, folded, the fit's settings, how often the simulations are
saved and the rotation's three. A setting a training will not read is shown as such and is neither judged nor passed:
the rotation's three while the rotation is off or the saved set's own is reused. While the simulations are not saved the
choice of continuing stands at its first choice, greyed, and that choice is what is passed. "Before you start" says what
the run uses, what it makes, what it does with the saved sets, how long it should take and the memory it may use.

Nothing is read while a run is live: the saved simulation sets, the run history and the records of earlier trainings are
read by the page's reading alone. The batches and the simulations per batch are remembered as the text of their boxes
once a training has started with them; the posterior that comes into use is remembered as it comes.
"""
import math
from dataclasses import replace

from PySide6.QtWidgets import QWidget

from core import config, orchestrator
from core.refusals import Refusal, default_clause, require_at_least, require_positive

from ... import keys, record_words, settings
from ...chain import model_label
from ...memory_limit import computer, limit_words, memory_limit_in_force
from ...remembered import NUM_RUNS_KEY, RUN_SIZE_KEY, put_many
from ...reported import reported
from ...run import ends, history
from ...run.controller import controller
from ...run.model import RunSpec
from ...step_facts import (CONTINUING, NO_POSTERIOR_TEXT, START_WORDS, Fact, budget_words, how_long_words,
                           pending_words, posterior_in_use, training_facts, uses_words)
from ...widgets.choice import Choice
from ...widgets.form import NameAndNote, TickedField
from ...widgets.labeled_inputs import FloatField, IntField
from .base import training_spec
from .help_text import HELP
from .prior_tab import bounds_in_use
from .step_page import StepPage

TITLE = "Training"
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


def _plain(value):
    """``value`` as a run's history holds it: a name, a number, a truth value or nothing as it is; a list of them as a
    list; anything else by its words."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_plain(one) for one in value]
    return str(value)


class PosteriorPanel(StepPage):
    """Step 3. The form's rows are read by their field keys; ``resume_choice`` is "Continue or start new"; the line of
    the prior in use stands in the block for the run."""

    def __init__(self, screen, parent=None):
        super().__init__(screen, TITLE, SENTENCE, step="training", parent=parent)
        self._reading = None                 # the saved simulation sets, as the last reading found them at its budget
        self._per_batch = self._per_epoch = None  # the run history's paces, for this size
        self._expected = None                # the epochs the last finished training of this kind ran, and its words
        self._looked = False                 # whether the last reading looked for that training in Records
        self._last = self._measured = None   # the last run on this computer, and the last that measured its batches
        self._history_read = False           # whether a reading has read the run history yet
        self._made = None                    # the posterior a run of this page returned, until the run has ended
        self._handed = None                  # the name and the note this page's last training was handed
        self.resume_choice = Choice()
        for words, policy in RESUME_CHOICES:
            self.resume_choice.addItem(words, policy)
        self._saves = TickedField(IntField(config.TRAINING_CHECKPOINT_EVERY, grouped=True), "Never", 0)
        # Heard before its row hears it: the choice of continuing is fixed before the page judges the form.
        self._saves.changed.connect(self._saves_changed)
        form = self.form
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
        self.restore_settings(settings.settings())
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
        return START_WORDS

    def plan_fields(self) -> dict:
        return {key: self.form.rows[key] for key, _ in _RULES}

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
        return posterior_in_use(record_words.label_of(posterior),
                                body.get("training") if isinstance(body, dict) else None, made_for, named)

    def problems(self) -> dict:
        """Every rule the form breaks now, by field key in the order a press reads them: the line under the field, built
        from its row's label, and what the press says, the same with the value it found and the usual one. A setting the
        training will not read breaks none, and a blank expected epochs is a choice."""
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
        memory limit and the last run; the other steps' changes not applied; the name and the note."""
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
        return training_facts(needs_prior=prior is None, uses=uses, budget=budget, reading=reading,
                              never=self._never(), resume=self._resume(), batches=batches, how_long=how_long,
                              memory=memory, others_pending=self._others_pending(), name_facts=names,
                              field_facts=field_facts)

    def _others_pending(self) -> list:
        """The other steps whose page shows changes they have not applied, each with what this run uses of it."""
        screen = self._screen
        try:
            pending = screen.pending() if screen is not None else frozenset()
        except Exception:                    # noqa: BLE001 -- a page's facts never raise
            return []
        session, said = getattr(screen, "session", None), []
        held = {"model": lambda: model_label(getattr(getattr(session, "draft", None), "model", None)),
                "prior": lambda: record_words.label_of(getattr(session, "inf_prior", None))}
        for step in ("model", "prior"):
            if step in pending:
                said.append(pending_words(keys.step_number(step), keys.step(step).name, held[step]()))
        return said

    # ── the reading of the disk ──
    def read_disk(self) -> None:
        """The run history (the last run on this computer and the last that measured its batches); and, while a prior is
        held and the budget passes its rules, the saved simulation sets at this budget, the run history's paces for this
        size, and the epochs the last finished training of this kind ran: only then has it looked for one."""
        folder, here = controller().folder(), computer()
        self._last, self._measured = history.last_on(folder, here), history.last_measured_on(folder, here)
        self._history_read = True
        self._reading = self._per_batch = self._per_epoch = self._expected = None
        self._looked = False
        cfg, prior = self._held()
        budget = self._budget()
        if prior is None or budget is None:
            return
        num_runs, cap, cadence = budget
        self._reading = orchestrator.training_reading(cfg, prior, num_runs=num_runs, run_size_cap=cap,
                                                      checkpoint_every=cadence)
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

    def _stage_values(self) -> dict:
        """The settings the stage would be handed now, as far as they pass their rules: for the run's facts."""
        found, rows = self.problems(), self.form.rows
        return {key: rows[key].value() for key, _ in self._ruled()
                if key not in found and key != "expected_epochs" and rows[key].value() is not None}

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
        """Start the training on the configuration and the prior held: nothing is held anew until the run returns its
        posterior. The batches and the simulations per batch are remembered once the run has started."""
        cfg, prior = self._held()
        if prior is None:
            return                               # the facts say why, and the button is greyed with them
        try:
            knobs = self._read_knobs()
        except Refusal as refused:
            self._refusal(refused)
            return
        handed = (self._names.name_value(), self._names.note_value())
        started = self.dispatch(orchestrator.build_posterior, cfg, prior, None, True, name=handed[0], note=handed[1],
                                resume=self._resume(), new_run=False, if_name_taken="unnamed", provide_fig_sink=True,
                                on_result=self._on_posterior, **knobs)
        if started:
            self._handed = handed
            put_many([(NUM_RUNS_KEY, str(knobs["num_runs"])), (RUN_SIZE_KEY, str(knobs["run_size_cap"]))])

    def run_spec(self, fn, *args, **kwargs) -> RunSpec:
        """A training, which writes a posterior, its simulation set and its kept training, and may be tried again: with
        the epochs typed for its time, the plan in use and the settings it was started with. Or a posterior's load,
        which writes nothing, keeps no file and stops without a question (its fourth argument says which, as the stage
        reads it)."""
        if fn is not orchestrator.build_posterior:
            return super().run_spec(fn, *args, **kwargs)
        if args[3]:
            row = self.form.rows["expected_epochs"]
            typed = None if "expected_epochs" in self.problems() else row.value()
            return training_spec("Training", _PLACE, STOP_WORDS, args[0], kwargs, narrowed=False,
                                 expected_epochs=typed, plan=self._screen.plan_entry() if self._screen else None,
                                 settings=self.settings_entry())
        return RunSpec("Posterior load", _PLACE, stop_button="Stop loading", loads_only=True, keeps_files=False,
                       stop_asks=False)

    def settings_entry(self) -> dict:
        """What this training is started with, as its run's history keeps it: names and numbers only. The model and its
        options as step 1 holds them, the bounds file and the prior in use, the choice of continuing, the name and the
        note, and one row for each setting the training reads, with its label, its value, where the value came from and
        its usual one."""
        session = self.session
        draft = getattr(session, "draft", None)
        cfg, prior = self._held()
        options = {name: _plain(getattr(draft, name, None))
                   for name in ("chi_mode", "chi_n_freqs", "chi_k_pad", "chi_max_cycles", "reparam_rotate",
                                "units_override")}
        rows = [{"step": "training", "name": key, "label": row.label_text(), "value": _plain(row.value()),
                 "origin": row.origin.text(), "plan": None, "usual": _plain(row.usual())}
                for key, row in ((key, self.form.rows[key]) for key, _ in self._ruled())]
        return {"model": _plain(getattr(draft, "model", None)), "chi_mode": _plain(getattr(draft, "chi_mode", None)),
                "options": options, "bounds": _plain(bounds_in_use(cfg)), "prior": _plain(getattr(prior, "id", None)),
                "resume": self._resume(), "name": self._names.name_value(), "note": self._names.note_value(),
                "rows": rows}

    def _on_posterior(self, payload) -> None:
        """The run returned its posterior, while it is still live: it comes into use, and nothing is read until the run
        has ended."""
        self._made = payload
        self._screen.take_posterior(payload)

    def _run_ended(self, status) -> None:
        """A run has ended. One of this page's that returned its posterior: a word when the record came back under
        another name than the one its call asked for (the ended run's own, which "Continue" and "Resume" repeat), and
        the name and the note each emptied while it still holds what the call was handed, so that one typed since
        stands. Then the reading."""
        made, self._made = self._made, None
        if made is not None:
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
        it reads nothing."""
        reads = self.stale
        super().showEvent(event)
        if not reads:
            self.redraw()

    # ── what is remembered ──
    def restore_settings(self, qs) -> None:
        """The batches and the simulations per batch the last training started with, put back as their boxes' text,
        never read anywhere else, and marked kept from last time until they are touched. A value that is no whole
        number is passed over."""
        for key, name in ((NUM_RUNS_KEY, "num_runs"), (RUN_SIZE_KEY, "run_size_cap")):
            raw = settings.get_str(qs, key, "").strip()
            try:
                value = int(raw)
            except ValueError:
                continue
            row = self.form.rows[name]
            row.set_value(value)
            row.kept = True
