"""Step 6, Narrowing round: a second posterior trained for one observation only. The prior is cut down to the region the
posterior in use finds believable for that observation, and training is repeated there: the proposal is the prior
restricted to that region, never the posterior. The result is valid only near that observation.

The step holds nothing of its own: a round's posterior comes into use in step 3, through the screen's one way in, once
its run returns it, and the posterior it came from stays in Records. The form holds the observation the round is drawn
around, the region's level and how many directions it narrows, the budget of the round's simulations, the network and,
folded, the fit's settings: each passed to the round as an argument, the network always, so that what is on the page
is what trains. Their rules are a training's own, read from the one table both pages read.

The list of observations holds the observations of the configuration's kind of observation and width; with a posterior
in use that was narrowed around one observation, those it cannot be used on are listed greyed. An observation an
inference makes is chosen in the list once that run has ended, the line under it saying what it replaced. Nothing is
read while a run is live: the observations' records, the run history and the kept sets are read by the page's reading
alone. The observation chosen and the budget are remembered once a round has started with them.
"""
import json
import math
import traceback
from pathlib import Path

from PySide6.QtCore import Qt

from core import config, orchestrator
from core.artifacts import ArtifactStore, default_store
from core.artifacts.store import MANIFEST
from core.refusals import Refusal, require_at_least, require_between
from core.SBI import truncate

from ... import check_lineage, plans, record_words, settings
from ...memory_limit import computer, limit_words, memory_limit_in_force
from ...remembered import OBSERVATION_KEY, put_many
from ...run import ends, history
from ...run.controller import controller
from ...run.model import RunSpec
from ...step_facts import (CANCEL, CANNOT_BE_USED_BLOCKS, CANNOT_BE_USED_TEXT, CARRIED_TEXT, CHANGED_TO_TEXT,
                           CHECK_THIS_WORDS,
                           DIRECTIONS_TEXT, FIT_RULES, NETWORK_DIFFERS_TEXT, NETWORK_HELD_TEXT, NEW_SET_ROUND_TEXT,
                           NEW_SET_ROUND_TITLE, NO_OBSERVATION_WORDS, NO_PRIOR_WORDS, REGION_TEXT, REGION_WARN_BELOW,
                           ROTATION_REUSED_TEXT, ROUND_CAME_FROM_TEXT, ROUND_IN_USE_TEXT, ROUND_KEPT_TEXT,
                           ROUND_START_WORDS, SAME_OBSERVATION_WORDS, Fact, fit_row_problem, pressed_reading,
                           round_facts, round_how_long, round_uses)
from ...widgets.artifact_picker import StorePicker
from ...widgets.ask import ask
from ...widgets.form import NameAndNote, TickedField
from ...widgets.labeled_inputs import FloatField, IntField
from ...widgets.result_region import NextStep
from .base import training_spec
from .help_text import HELP
from .posterior_tab import EARLIER_NAME_TEXT
from .step_page import StepPage

TITLE = "Narrowing round"
SENTENCE = ("Trains a second posterior for one observation only: the prior is cut down to the region the posterior in "
            "use finds believable for that observation, and training is repeated there (truncated sequential neural "
            "posterior estimation, TSNPE). The result is valid only near that observation.")
CHOOSE_OBSERVATION = "Choose an observation"
STOP_WORDS = "Stop the narrowing round…"
#: The run of an observation's load, a link of a chain of loads.
LOAD_WORD = "Observation load"
STOP_LOADING = "Stop loading"
EPOCHS = "epochs"
#: The words beside the main button while a field breaks its rule.
SETTING_BLOCKS = "a setting cannot be used"
#: What the settings file keeps of a round, written once a round has started with them; the observation too, under the
#: key of the observation last in use.
NUM_RUNS_KEY = "inference_tsnpe/num_runs"
RUN_SIZE_KEY = "inference_tsnpe/run_size_cap"
#: The fit's settings a round reads, a training's own and under its rules, in the order the stage checks them.
_FIT = ("num_runs", "run_size_cap", "hidden_features", "num_transforms", "learning_rate", "stop_after_epochs",
        "max_num_epochs", "checkpoint_every")
#: The rows a plan fills, in the form's order.
PLAN_ROWS = ("hpd_level", "n_directions", *_FIT)
# Where the window shows this step's runs: Parameter inference, its step Narrowing round.
_PLACE = ("inference", "narrowing")
# A list's row as it can be chosen, and the role under which the list's items keep their flags.
_CHOOSABLE = Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled
_FLAGS_ROLE = Qt.ItemDataRole.UserRole - 1


def observation_digests(store) -> dict:
    """Each observation's id and the digest its record holds, its manifest read once by its folder: a counted reading,
    made by the page's reading alone. An observation whose record cannot be read is left out."""
    found = {}
    try:
        listed = store.list("observation")
    except Exception:                        # noqa: BLE001 -- a store that cannot be read gives nothing
        return found
    for summary in listed:
        try:
            manifest = json.loads((Path(summary.path) / MANIFEST).read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            continue
        body = manifest.get("body") if isinstance(manifest, dict) else None
        digest = body.get("x_obs_digest") if isinstance(body, dict) else None
        if isinstance(digest, str) and digest:
            found[str(summary.id)] = digest
    return found


def _plain(value):
    """``value`` as a run's history holds it: a name, a number, a truth value or nothing as it is; a list of them as a
    list; anything else by its words."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_plain(one) for one in value]
    return str(value)


def _whole(value) -> "int | None":
    try:
        return int(value) if value is not None and not isinstance(value, bool) else None
    except (TypeError, ValueError):
        return None


class TSNPEPanel(StepPage):
    """Step 6. The form's rows are read by their field keys; ``obs_picker`` lists the observations. The page reads what
    the checks need at its own readings, as the Training page does, so the screen does not read them again while it is
    in sight."""

    #: Whether the page's reading reads what the checks need, which the screen then leaves to it while it is in sight.
    reads_the_checks = True

    def __init__(self, screen, parent=None):
        super().__init__(screen, TITLE, SENTENCE, step="narrowing", parent=parent)
        self._digests: dict = {}             # each observation's digest, as the last reading read them
        self._digests_read = False           # whether the last reading read them (a narrowed posterior in use)
        self._preview = None                 # the training preview at the budget, as the last reading made it
        self._per_epoch = None               # the run history's pace of a round's epoch at this size
        self._expected = None                # the epochs the last narrowing round of this kind ran, and its words
        self._last = self._measured = None   # the last run on this computer, and the last that measured its batches
        self._history_read = False
        self._trained_with = ""              # the label of the prior the posterior in use was trained with
        self._followed = None                # the observation in use the list last moved to, by its id
        self._carried = None                 # an observation a next step carried, chosen at the next reading
        self._changed = ""                   # what the line under the list says of an observation it was moved to
        self._remembered = ""                # the observation the last round started with
        self._made = None                    # the posterior a round of this page returned, until its run has ended
        self._handed = None                  # the parent posterior and the observation the last round was handed
        self._call = None                    # what the last round was dispatched with, for its start again
        self._bodies: dict = {}              # the body of the observation chosen at the last reading, by its id
        self._continuing = None              # the offer a "Continue from batch N" set the page to, until a round starts
        self.obs_picker = StorePicker("observation", parent=self, keeps_selection=True, hint=CHOOSE_OBSERVATION,
                                      row_filter=self._of_this_kind)
        self.obs_picker.combo.currentIndexChanged.connect(self._picked)
        form = self.form
        form.add("observation", self.obs_picker.combo, help_text=HELP["tsnpe_obs"])
        form.add("hpd_level", FloatField(truncate.DEFAULT_HPD), usual=truncate.DEFAULT_HPD, help_text=HELP["tsnpe_hpd"])
        form.add("n_directions", IntField(truncate.DEFAULT_N_DIRECTIONS, grouped=True),
                 usual=truncate.DEFAULT_N_DIRECTIONS, help_text=HELP["tsnpe_dirs"])
        form.add("num_runs", IntField(config.TRAINING_NUM_RUNS, grouped=True), usual=config.TRAINING_NUM_RUNS,
                 help_text=HELP["num_runs"])
        form.add("run_size_cap", TickedField(IntField(config.TRAINING_RUN_SIZE, grouped=True), "Automatic", 0),
                 usual=config.TRAINING_RUN_SIZE, help_text=HELP["run_size"])
        form.add("hidden_features", IntField(config.NSF_HIDDEN_FEATURES, grouped=True),
                 usual=config.NSF_HIDDEN_FEATURES, help_text=HELP["flow_hidden"])
        form.add("num_transforms", IntField(config.NSF_NUM_TRANSFORMS, grouped=True), usual=config.NSF_NUM_TRANSFORMS,
                 help_text=HELP["flow_transforms"])
        self.network_line = form.add_sentence("", under="num_transforms")
        self.network_line.setVisible(False)
        form.add("learning_rate", FloatField(config.TRAINING_LEARNING_RATE), usual=config.TRAINING_LEARNING_RATE,
                 folded=True, help_text=HELP["flow_lr"])
        form.add("stop_after_epochs", IntField(config.TRAINING_STOP_AFTER_EPOCHS, grouped=True),
                 usual=config.TRAINING_STOP_AFTER_EPOCHS, folded=True, help_text=HELP["flow_patience"])
        form.add("max_num_epochs", TickedField(IntField(None, grouped=True), "No limit", None), EPOCHS, usual=None,
                 folded=True)
        form.add("checkpoint_every", TickedField(IntField(config.TRAINING_CHECKPOINT_EVERY, grouped=True), "Never", 0),
                 usual=config.TRAINING_CHECKPOINT_EVERY, folded=True)
        self.rotation_row = form.add_text("rotation", ROTATION_REUSED_TEXT, "")
        self._names = NameAndNote("posterior", required=True)
        self.use_names(self._names)
        self.restore_settings(settings.settings())
        self.refresh_local_gates()

    # ── what is in use ──
    def _held(self) -> tuple:
        """The configuration, the prior and the posterior in use, each None when none is held."""
        session = getattr(self._screen, "session", None)
        return (getattr(session, "cfg", None), getattr(session, "inf_prior", None),
                getattr(session, "posterior", None))

    def _narrowed_digest(self) -> "str | None":
        """The digest of the observation the posterior in use was narrowed around; None for one that was not."""
        return getattr(getattr(self._held()[2], "posterior", None), "x_obs_digest", None)

    def _of_this_kind(self, summary) -> bool:
        """Whether the observation ``summary`` is of the configuration's kind of observation and width: only those can be
        used with it. With no configuration in use every observation is listed."""
        cfg = self._held()[0]
        if cfg is None:
            return True
        mode = getattr(cfg, "observation_mode", None)
        if mode is not None and getattr(summary, "mode", None) not in (None, mode):
            return False
        try:
            from core.artifacts.manifest import conditioning_block
            width = int(conditioning_block(cfg)["width"])
        except Exception:                    # noqa: BLE001 -- a configuration that cannot say its width lists by kind
            return True
        held = getattr(summary, "width", None)
        return held in (None, 0) or int(held) == width

    def _usable(self, ident: str) -> bool:
        """Whether the observation ``ident`` can be used by the posterior in use: any, unless it was narrowed around one,
        and then only that one, by its digest, as the last reading read the digests."""
        wanted = self._narrowed_digest()
        return wanted is None or not self._digests_read or self._digests.get(ident) == wanted

    def _grey(self) -> None:
        """The observations the posterior in use cannot be used on, greyed in the list with why. Each row is told through
        the list's own data, its flags under the role its items keep them in, and no item of the list is held: the list
        deletes its items as it lists the store again, and a handle kept on one would outlive it. A row the round can
        use keeps the list's own tip: the list is listed afresh before each greying."""
        combo = self.obs_picker.combo
        for at in range(combo.count()):
            ident = combo.itemData(at)
            if ident is None:
                continue
            usable = self._usable(str(ident))
            combo.setItemData(at, (_CHOOSABLE if usable else Qt.ItemFlag.NoItemFlags).value, _FLAGS_ROLE)
            if not usable:
                combo.setItemData(at, CANNOT_BE_USED_TEXT + ".", Qt.ItemDataRole.ToolTipRole)

    def observation_on_offer(self) -> bool:
        """Whether the list holds an observation the round can be drawn around, as the last reading found them."""
        combo = self.obs_picker.combo
        return any(combo.itemData(at) is not None and self._usable(str(combo.itemData(at)))
                   for at in range(combo.count()))

    def _follow_the_observation(self) -> None:
        """The list moved to the observation in use when it is not the one it last moved to: an observation an inference
        made, or one loaded, chosen; a choice it replaces is said on the line under the list. Never while a run is live:
        the page's reading and a run's end call this."""
        observation = getattr(getattr(self._screen, "session", None), "observation", None)
        ident = getattr(observation, "id", None)
        if not ident or ident == self._followed:
            return
        before = self.obs_picker.key()
        self.obs_picker.refresh()
        if self.obs_picker.combo.findData(ident) < 0:
            return
        before_words = self.obs_picker.combo.itemText(self.obs_picker.combo.findData(before)) if before else ""
        self.obs_picker.restore_key(ident)
        self._followed = ident
        self._changed = (CHANGED_TO_TEXT.format(record_words.label_of(observation), before_words)
                         if before and before != ident else "")
        self.form.rows["observation"].kept = False

    def _picked(self, _index) -> None:
        """The list's choice moved: the line of a pick the window made goes once another observation is chosen. A
        relisting that keeps the choice says nothing, so the line stands through the page's readings."""
        if self._changed and self.obs_picker.key() != self._followed:
            self._changed = ""

    def take_carried(self, carries) -> None:
        """What a next step carries here: the observation it names, chosen in the list. Nothing is asked: the round holds
        nothing of its own to let go. One the list does not hold is looked for in the store at once, unless a run is
        live; one not found then is held, said under the list, and chosen at the step's next reading of its list."""
        ident = carries.get("observation") if hasattr(carries, "get") else None
        if not ident:
            return
        if self.obs_picker.combo.findData(ident) < 0 and not controller().is_live():
            self.obs_picker.refresh()
            self._grey()
        if self.obs_picker.combo.findData(ident) >= 0:
            self.obs_picker.restore_key(ident)
            self._changed, self._carried = "", None
        else:
            self._carried = str(ident)
        self.redraw()

    # ── what a plan fills ──
    def plan_fields(self) -> dict:
        """The rows a plan fills, in the form's order: the region's level and its directions, then the fit's settings.
        The observation is a pick, and no plan holds it."""
        return {key: self.form.rows[key] for key in PLAN_ROWS}

    def plan_read(self, applied: bool) -> dict:
        """What a narrowing round reads of this step: its form as it holds it, the epoch limit's blank among it; the
        step holds nothing a later run reads."""
        return {} if applied else {key: row.value() for key, row in self.plan_fields().items()}

    # ── what a page answers ──
    def main_words(self) -> str:
        """"Start the narrowing round"; while the page is set to continue a round's simulation set, the offer's own
        words."""
        return self._continuing.button if self._continuing is not None else ROUND_START_WORDS

    def in_use_words(self) -> tuple:
        """What the round would be drawn from: the posterior in use, by its label."""
        posterior = self._held()[2]
        if posterior is None:
            return "", "Load or train a posterior in step 3 to narrow it."
        return record_words.label_of(posterior), "A narrowing round draws its region from this posterior."

    def _never(self) -> bool:
        return self.form.rows["checkpoint_every"].value() == 0

    def problems(self) -> dict:
        """Every rule the form breaks now, by field key in the order a press reads them: the line under the field and
        what the press says, the stage's own sentence. The fit's settings by a training's own rules; the region's level
        inside (0, 1) and the directions at least 1 and at most the model's parameters."""
        rows, found = self.form.rows, {}
        level = rows["hpd_level"]
        said, value = level.reading(), level.value()
        if said is None and value is not None:
            try:
                require_between("hpd_level", value, 0.0, 1.0, open_lo=True, open_hi=True)
            except Refusal as refused:
                found["hpd_level"] = (f"{level.label_text()} must be greater than 0 and less than 1.", refused.message)
        elif said is not None:
            found["hpd_level"] = (said, pressed_reading("hpd_level", said))
        directions = rows["n_directions"]
        said, value, count = directions.reading(), directions.value(), self._parameters()
        if said is None and value is not None:
            try:
                require_at_least("n_directions", value, 1)
                if count is not None and value > count:
                    found["n_directions"] = (DIRECTIONS_TEXT.format(count),) * 2
            except Refusal as refused:
                found["n_directions"] = (f"{directions.label_text()} must be at least 1.", refused.message)
        elif said is not None:
            found["n_directions"] = (said, pressed_reading("n_directions", said))
        for key in _FIT:
            row = rows[key]
            problem = fit_row_problem(key, row.value(), label=row.label_text(), reading=row.reading())
            if problem is not None:
                found[key] = problem
        return found

    def _parameters(self) -> "int | None":
        """How many parameters the model in use has: the most directions a region can narrow."""
        cfg = self._held()[0]
        try:
            return len(cfg.params_dict) + len(cfg.rescale_params)
        except Exception:                    # noqa: BLE001 -- a configuration that cannot say counts nothing
            return None

    def _network(self) -> tuple:
        rows = self.form.rows
        return rows["hidden_features"].value(), rows["num_transforms"].value()

    def _network_words(self) -> tuple:
        """What the line under the network says of the posterior in use's own network: its kind and its sentence."""
        training = getattr(getattr(self._held()[2], "manifest", None), "body", None)
        training = training.get("training") if isinstance(training, dict) else None
        held = tuple(_whole((training or {}).get(key)) for key in ("hidden_features", "num_transforms"))
        if None in held:
            return "", ""
        if held != tuple(self._network()):
            return "warn", NETWORK_DIFFERS_TEXT.format(*held, *self._network())
        return "info", NETWORK_HELD_TEXT.format(*held)

    def facts_from_form(self) -> tuple:
        """The facts of the step as the form, the session and the last reading stand. Reads nothing."""
        cfg, prior, posterior = self._held()
        rows, found = self.form.rows, self.problems()
        for key, row in rows.items():
            if row.widget is None or key in ("name", "note", "observation"):
                continue
            row.set_line("err" if key in found else "", found[key][0] if key in found else "")
        level = rows["hpd_level"].value()
        if "hpd_level" not in found and level is not None and level < REGION_WARN_BELOW:
            rows["hpd_level"].set_line("warn", REGION_TEXT.format(REGION_WARN_BELOW))
        kind, said = self._network_words()
        self.network_line.set_text(said)
        self.network_line.setVisible(bool(said))
        if kind == "warn" and "num_transforms" not in found:
            rows["num_transforms"].set_line("warn", said)
            self.network_line.setVisible(False)
        key = self.obs_picker.key()
        observation_facts = []
        if key and not self._usable(key):
            rows["observation"].set_line("err", CANNOT_BE_USED_TEXT + ".")
            observation_facts.append(Fact("err", CANNOT_BE_USED_TEXT + ".", CANNOT_BE_USED_BLOCKS))
        elif self._carried:
            rows["observation"].set_line("info", CARRIED_TEXT.format(self._carried))
        else:
            rows["observation"].set_line("warn" if self._changed else "", self._changed)
        needs = [name for name, held in (("posterior", posterior), ("prior", prior), ("observation", key)) if not held]
        uses = prior_own = ""
        if posterior is not None and prior is not None and key:
            observation = self.obs_picker.selection_text()
            uses = round_uses(record_words.label_of(posterior), record_words.label_of(prior), observation,
                              self._observation_body(key))
        if posterior is not None and prior is not None:
            parents = getattr(getattr(posterior, "manifest", None), "parents", None)
            trained_with = parents.get("prior") if isinstance(parents, dict) else None
            in_use = record_words.label_of(prior)
            prior_own = in_use if not trained_with or trained_with == getattr(prior, "id", None) else (
                self._trained_with or NO_PRIOR_WORDS, in_use)
        simulations = None
        preview = self._preview if "num_runs" not in found and "run_size_cap" not in found else None
        if preview is not None and getattr(preview, "width", None):
            simulations = rows["num_runs"].value() * preview.width
        how_long = round_how_long(self._expected, self._per_epoch) if self._history_read else ""
        memory = limit_words(memory_limit_in_force())
        if self._history_read:
            memory = f"{memory} {ends.last_run_sentence(self._last, self._measured)}"
        names = [fact for fact in (self._names.name_fact(), self._names.note_fact()) if fact]
        field_facts = [*observation_facts, *(Fact("err", line, SETTING_BLOCKS) for line, _said in found.values())]
        continuing = self._continuing
        continued = continuing.batch if continuing is not None and continuing.kind == "continue" else None
        return round_facts(needs=needs, uses=uses, prior_own=prior_own or None, simulations=simulations,
                           continuing_from=continued, checks=self._pre_run(), fit="never" if self._never() else "saved", how_long=how_long,
                           memory=memory, others_pending=self._others_pending(), name_facts=names,
                           field_facts=field_facts, plan_lines=self.plan_lines())

    def _observation_body(self, ident: str):
        return self._bodies.get(ident)

    def _pre_run_rows(self) -> "tuple | None":
        """The checks' rows for the prior in use alone, as the screen last read them; None before its first reading."""
        rows = getattr(self._screen, "pre_run_rows", None)
        return rows() if callable(rows) else None

    def _pre_run(self) -> tuple:
        rows = self._pre_run_rows()
        if rows is None or self._held()[1] is None:
            return ()
        return (check_lineage.pre_run_fact(rows),)

    # ── the reading of the disk ──
    def read_disk(self) -> None:
        """What the checks need, read by the screen; the observations listed again and, with a posterior narrowed around
        one in use, each one's digest; the list moved to the observation in use; the label of the prior the posterior
        was trained with; the training preview at the budget, for the simulations the round makes; and the run history:
        the last run on this computer, and the pace of an epoch and the epochs of the last narrowing round of this
        kind."""
        screen = self._screen
        if screen is not None and hasattr(screen, "read_checks"):
            screen.read_checks()
        cfg, prior, posterior = self._held()
        store = default_store()
        self.obs_picker.refresh()
        self._digests_read = self._narrowed_digest() is not None
        self._digests = observation_digests(store) if self._digests_read else {}
        if not self.obs_picker.key():
            ident = getattr(getattr(getattr(screen, "session", None), "observation", None), "id", None)
            self.obs_picker.restore_key(ident or self._remembered)
            self.form.rows["observation"].kept = not ident and bool(self.obs_picker.key())
        self._follow_the_observation()
        carried, self._carried = self._carried, None
        if carried and self.obs_picker.combo.findData(carried) >= 0:
            self.obs_picker.restore_key(carried)
            self._changed = ""
        self._grey()
        key, self._bodies = self.obs_picker.key(), {}
        if key:
            try:
                self._bodies[key] = store.get("observation", key).body
            except Exception:                # noqa: BLE001 -- an observation that cannot be read is named alone
                pass
        parents = getattr(getattr(posterior, "manifest", None), "parents", None)
        ref = parents.get("prior") if isinstance(parents, dict) else None
        self._trained_with = ""
        if ref and ref != getattr(prior, "id", None):
            try:
                self._trained_with = record_words.label_of(_Record(store.get("prior", ref)))
            except Exception:                # noqa: BLE001 -- a prior no longer in Records is said so
                self._trained_with = ""
        folder, here = controller().folder(), computer()
        self._last, self._measured = history.last_on(folder, here), history.last_measured_on(folder, here)
        self._history_read = True
        self._preview = self._per_epoch = self._expected = None
        rows, found = self.form.rows, self.problems()
        if cfg is None or prior is None or found.keys() & {"num_runs", "run_size_cap"}:
            return
        self._preview = orchestrator.training_preview(cfg, prior, num_runs=rows["num_runs"].value(),
                                                      run_size_cap=rows["run_size_cap"].value())
        width = getattr(self._preview, "width", None)
        if width and not found.keys() & {"hidden_features", "num_transforms"}:
            try:
                self._per_epoch = history.estimate(folder, word="Narrowing round", per="epoch",
                                                   size={"network": tuple(self._network()),
                                                         "simulations": rows["num_runs"].value() * width})
            except Exception:                # noqa: BLE001 -- a history written by hand says nothing
                self._per_epoch = None
        try:
            spec = training_spec("Narrowing round", _PLACE, STOP_WORDS, cfg, self._stage_values(), narrowed=True)
            self._expected = controller().expected_epochs(spec)
        except Exception:                    # noqa: BLE001 -- a configuration that cannot say its kind has no last round
            self._expected = None

    def _stage_values(self) -> dict:
        """The fit's settings the round would be handed now, as far as they pass their rules."""
        rows, found = self.form.rows, self.problems()
        return {key: rows[key].value() for key in _FIT if key not in found and rows[key].value() is not None}

    def _run_ended(self, status) -> None:
        """A run has ended. One of this page's that returned its posterior: its result said. Then the reading, which
        lists the observations again and moves the list to the observation in use, which the controller's order makes
        safe: no run is live by now. The objects the last start was handed go, but at a refusal made before anything was
        spent, whose question may start the same call again."""
        if getattr(status, "outcome", None) is not None:
            self._call = None
        made, self._made = self._made, None
        handed, self._handed = self._handed, None
        if made is not None:
            self._say_the_result(made, handed)
        super()._run_ended(status)

    # ── the press ──
    def _read_knobs(self) -> dict:
        """The settings the round is handed, read through the rules: the first broken is refused, and nothing changes.
        The epoch limit only when one is typed."""
        for key, (_line, said) in self.problems().items():
            raise Refusal(said, field=key)
        rows = self.form.rows
        knobs = {key: rows[key].value() for key in _FIT if not (key == "max_num_epochs" and rows[key].value() is None)}
        knobs["level"], knobs["n_directions"] = rows["hpd_level"].value(), rows["n_directions"].value()
        return knobs

    def main_pressed(self) -> None:
        """Start the narrowing round on the configuration, the posterior and the prior in use, around the observation
        chosen, loaded here, at the press: a refusal of the load is said as any refusal is, a fault as a fault. A start
        after the frequency band or the audit of the prior in use read "Failed" asks first, "Cancel" preselected."""
        cfg, prior, posterior = self._held()
        if posterior is None or prior is None or not self.obs_picker.key():
            return
        try:
            knobs = self._read_knobs()
        except Refusal as refused:
            self._refusal(refused)
            return
        rows = self._pre_run_rows()
        failed = None if rows is None else check_lineage.pre_run_question(rows, start=self.main_words())
        if failed is not None:
            title, text, other = failed
            if not ask(self, title, text, CANCEL, other):
                return
        try:
            observation = default_store().load_observation(cfg, self.obs_picker.key())
        except Refusal as refused:
            self._refusal(refused)
            return
        except Exception as error:           # noqa: BLE001 -- a fault is told as a fault
            self._on_error(error, traceback.format_exc())
            return
        self._start(cfg, posterior, prior, observation, knobs, new_run=False)

    def _start(self, cfg, posterior, prior, observation, knobs: dict, *, new_run: bool) -> bool:
        """Dispatch the round with ``knobs`` and the consent to start a new set, ``new_run``; once it has started, the
        observation chosen and the budget are remembered. True when its run was started."""
        names = self._names
        call = dict(knobs)
        call.update(resume="require" if self._continuing is not None else "auto", new_run=new_run,
                    name=names.name_value(), note=names.note_value(), if_name_taken="unnamed")
        self._call = (cfg, posterior, prior, observation, knobs)
        started = self.dispatch(orchestrator.tsnpe_round, cfg, posterior, prior, observation, provide_fig_sink=True,
                                on_result=self._on_round, **call)
        if started:
            self._continuing = None
            self._handed = (posterior, observation)
            model = getattr(getattr(getattr(self._screen, "session", None), "draft", None), "model", None)
            put_many([(OBSERVATION_KEY, self.obs_picker.key()), (NUM_RUNS_KEY, str(knobs["num_runs"])),
                      (RUN_SIZE_KEY, str(knobs["run_size_cap"]))] + self._screen.last_used(model))
        return started

    def continue_run(self, offer, settings_: dict) -> None:
        """The round's form filled from its history, for "Continue from batch N" or "Fit again on the saved
        simulations": the region's level and its directions the run's, the name and the note blank, the earlier run's
        name as the hint, the main button the offer's, and the next round handed the policy that must continue; the
        page then read."""
        rows = self.form.rows
        for key, name in (("hpd_level", "level"), ("n_directions", "directions")):
            value = settings_.get(name)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                rows[key].set_value(value)
        self._continuing = offer
        self._names.name.clear()
        self._names.note.clear()
        earlier = settings_.get("name") if isinstance(settings_.get("name"), str) else ""
        self._names.set_hint(EARLIER_NAME_TEXT.format(earlier) if earlier else "")
        self.read_facts()

    def _refusal(self, exc) -> None:
        """A refusal, as every page tells one; but the stage's refusal of a new simulation set beside a saved one a
        setting away is a question, the stage's own sentence in it: "Start the narrowing round" dispatches the same call
        again with the consent and the objects already loaded; "Cancel", preselected, starts nothing. Answered, the
        objects go, but those a new start holds."""
        if getattr(exc, "field", None) != "new_run" or self._call is None or controller().is_live():
            super()._refusal(exc)
            return
        (cfg, posterior, prior, observation, knobs), self._call = self._call, None
        sentence = f"{getattr(exc, 'message', None) or exc} {NEW_SET_ROUND_TEXT}"
        if ask(self, NEW_SET_ROUND_TITLE, sentence, CANCEL, ROUND_START_WORDS):
            self._start(cfg, posterior, prior, observation, knobs, new_run=True)

    def load_observation(self, ident: str) -> bool:
        """Load the saved observation ``ident`` on the configuration in use, as a chain's link: a run that only loads,
        keeping no history and no file, its record put in use as it returns. True when its run was started; nothing
        while a run is live or with no configuration in use."""
        cfg = self._held()[0]
        if cfg is None or not ident or controller().is_live():
            return False
        return self.dispatch(ArtifactStore.load_observation, default_store(), cfg, ident,
                             on_result=self._observation_loaded)

    def _observation_loaded(self, observation) -> None:
        """The observation's load returned its record, while its run is still live: it comes into use through the
        screen's one way in; the list moves to it at the reading after the run's end."""
        self._screen.take_observation(observation)

    def run_spec(self, fn, *args, **kwargs):
        """A narrowing round: a training of its own, on the region, so its run is a training's, narrowed; its first
        total of unusable driven recordings compared with the audit of the prior in use that counted as it starts, as
        the screen last read the checks; its history keeps what it was started with. An observation's load only
        loads."""
        if fn is ArtifactStore.load_observation:
            return RunSpec(LOAD_WORD, _PLACE, stop_button=STOP_LOADING, loads_only=True, keeps_files=False,
                           stop_asks=False)
        if fn is not orchestrator.tsnpe_round:
            return super().run_spec(fn, *args, **kwargs)
        rows = self._pre_run_rows()
        prior = args[2] if len(args) > 2 else None
        plan = self._screen.plan_entry() if hasattr(self._screen, "plan_entry") else None
        return training_spec("Narrowing round", _PLACE, STOP_WORDS, args[0], kwargs, narrowed=True, plan=plan,
                             settings=self.settings_entry(), audit=check_lineage.counted_audit(rows) if rows else None,
                             prior=getattr(prior, "id", None),
                             prior_words=record_words.phrase_of("prior", prior) if prior is not None else "")

    def settings_entry(self) -> dict:
        """What this round is started with, as its run's history keeps it: names and numbers only. The model and its
        options as step 1 holds them, the bounds file, the prior, the posterior it is drawn from and the observation by
        their ids, the region's level and its directions, the name and the note, and one row for each fit's setting
        the round reads, with its label, its value, where the value came from, the value the source in use holds for it
        and its usual one."""
        from .prior_tab import bounds_in_use
        session = getattr(self._screen, "session", None)
        draft = getattr(session, "draft", None)
        cfg, prior, posterior = self._held()
        options = {name: _plain(getattr(draft, name, None))
                   for name in ("chi_mode", "chi_n_freqs", "chi_k_pad", "chi_max_cycles", "reparam_rotate",
                                "units_override")}
        source, heading = getattr(self._screen, "plan_source", None), plans.HEADING_OF[self.step]
        rows = [{"step": self.step, "name": key, "label": row.label_text(), "value": _plain(row.value()),
                 "origin": plans.origin(row, source, kept=row.kept, heading=heading),
                 "plan": _plain(plans.plan_value(source, heading, key)), "usual": _plain(row.usual())}
                for key, row in ((key, self.form.rows[key]) for key in _FIT)]
        return {"model": _plain(getattr(draft, "model", None)), "chi_mode": _plain(getattr(draft, "chi_mode", None)),
                "options": options, "bounds": _plain(bounds_in_use(cfg) if cfg is not None else None),
                "prior": _plain(getattr(prior, "id", None)), "parent_posterior": _plain(getattr(posterior, "id", None)),
                "observation": self.obs_picker.key() or None,
                "level": _plain(self.form.rows["hpd_level"].value()),
                "directions": _plain(self.form.rows["n_directions"].value()),
                "name": self._names.name_value(), "note": self._names.note_value(), "rows": rows}

    def _on_round(self, payload) -> None:
        """The round returned its posterior, while its run is still live: it comes into use through the screen's one way
        in, which remembers it as the posterior last in use; nothing is read until the run has ended."""
        self._made = payload
        self._screen.take_posterior(payload)

    def _say_the_result(self, made, handed) -> None:
        """The round's result, once its run has ended: the posterior in use from step 3 on, the observation it was
        narrowed around and how much of the prior's draws its region kept; the posterior it came from, which stays in
        Records; and the next steps."""
        parent, observation = handed if handed else (None, None)
        body = getattr(getattr(made, "manifest", None), "body", None)
        training = body.get("training") if isinstance(body, dict) else None
        acceptance = (training or {}).get("tsnpe_acceptance") if isinstance(training, dict) else None
        said = ROUND_IN_USE_TEXT.format(record_words.label_of(made),
                                        record_words.label_of(observation) if observation else NO_OBSERVATION_WORDS)
        try:
            share = float(acceptance)
            if math.isfinite(share):
                said = f"{said} {ROUND_KEPT_TEXT.format(math.floor(share * 100 + 0.5))}"
        except (TypeError, ValueError):
            pass
        facts = [Fact("info", said)]
        if parent is not None:
            facts.append(Fact("info", ROUND_CAME_FROM_TEXT.format(record_words.label_of(parent))))
        self.result.set_sentences(facts)
        self.result.set_next_steps((NextStep(CHECK_THIS_WORDS, "checks"), NextStep(SAME_OBSERVATION_WORDS, "infer")))
        name, note = self._names.name_value(), self._names.note_value()
        if name and getattr(made, "name", "") == name:
            self._names.name.clear()
        if note:
            self._names.note.clear()

    # ── what is remembered ──
    def restore_settings(self, qs) -> None:
        """The observation the last round started with, put back as the list's choice while the session holds none, and
        the budget, as its boxes' text; each marked kept from last time until it is touched. A value that is no whole
        number is passed over; the region's level and directions, and the network, open at their usual values."""
        self._remembered = settings.get_str(qs, OBSERVATION_KEY, "")
        self.obs_picker.restore_key(self._remembered)
        self.form.rows["observation"].kept = bool(self._remembered) and self.obs_picker.key() == self._remembered
        for key, name in ((NUM_RUNS_KEY, "num_runs"), (RUN_SIZE_KEY, "run_size_cap")):
            raw = settings.get_str(qs, key, "").strip()
            try:
                value = int(raw)
            except ValueError:
                continue
            row = self.form.rows[name]
            row.set_value(value)
            row.kept = True


class _Record:
    """A manifest as a record's label reads it: its kind, its name and its manifest."""

    def __init__(self, manifest):
        self.kind, self.name, self.manifest = manifest.kind, manifest.name, manifest
