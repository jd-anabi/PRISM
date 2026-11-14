"""Parameter inference: its six steps over ONE shared SbiSession, each step a page of its own.

    Model and options -> Prior -> Training -> Checks -> Inference -> Narrowing round

The first step records the MODEL-level choices (model, units, chi/rotation knobs) as a ConfigDraft. The Prior
step picks the BOUNDS file and turns that draft into the SimConfig, because the bounds file is what declares
which parameters are inferred, in what order and over what range -- a SimConfig cannot exist without it,
and the choice of bounds is also what selects the observation mode (a Forcing section or not).

The screen owns the session (the single source of truth); the steps read it through their ``_screen``
back-reference and never cache it. No step is ever disabled: one that cannot start yet still opens and can be
read, its buttons greyed by its own gate. ``refresh_gates`` runs every step's own gate after every run of a
step, then works out where each of the six steps stands, which the rail and Home show, and says so through
``states_changed``.

The screen also keeps the plan in use: where the forms of the steps were last filled from, which outlives the session
that step 1 replaces as it is applied, and what the steps' last reading found of that plan's file.

And it runs a chain of loads: step 1 applied with a model and its options, then each later step's own "Load this …" of
a saved record, one after the other, each once the one before has returned its record; Home's "Continue with these"
is one.
"""
import json
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QVBoxLayout, QWidget

from core import config

from .. import chain, check_lineage, check_readings, continuing, keys, record_words, remembered
from .. import plans as window_plans
from ..load_chain import LoadChain
from ..panels.inference import resume as resuming
from ..panels.inference.prior_tab import bounds_in_use
from ..panels.inference_tabs import (ConfigPanel, InferPanel, PosteriorPanel, PriorPanel,
                                     TSNPEPanel, ValidatePanel)
from ..run.controller import controller
from ..session import SbiSession
from ..widgets.adaptive_stack import AdaptiveStack

#: A plan's setting of step 1 by its name in a draft, where the two differ.
_DRAFT_NAMES = {"units": "units_override"}
#: The question a Continue or a Resume asks when the session holds a record.
CONTINUE_TITLE = "{}?"
CONTINUE_LETS_GO_TEXT = "This lets go of {}. Nothing is deleted: each stays in Records."
KEEP_SESSION = "Keep this session"
CONTINUE_OTHER = "Continue and start a new session"
FIT_AGAIN_OTHER = "Fit again and start a new session"
RESUME_TITLE = "Resume this training?"
RESUME_OTHER = "Resume and start a new session"
RESUME_WORDS = "Resume"
CONTINUE_WITH_WORDS = "Continue with these"
#: The question "Continue with these" asks when it would set what was chosen on the forms and not applied.
CONTINUE_WITH_TITLE = "Continue with these?"
CONTINUE_WITH_TEXT = ("This applies and loads what was in use last time. It sets these, which were chosen on the forms "
                      "and not applied:")
KEEP_WHAT_IS_THERE = "Keep what is there"
REPLACED_TEXT = "{}: {} becomes {}"
NO_PLAN_WORDS = "No plan"
#: Said under the heading of the page a press opened, when its chain of loads stopped before its end.
CHAIN_STOPPED_TEXT = "{} stopped at {}: {} was not {}. {} says why."
#: Said so when step 1 cannot take the model the chain names, which the step never shows, and so cannot say why itself.
CHAIN_CANNOT_TAKE_TEXT = "{} stopped at {}: {}."
#: Said where "Continue from batch N" was pressed, when what it would load is gone: nothing is let go.
CONTINUE_CANNOT_TEXT = "{} cannot start: {}."
CONTINUE_PRIOR_GONE = "the prior it was trained with is no longer in Records"
CONTINUE_BOUNDS_GONE = "the bounds file {} is no longer there"


@dataclass(frozen=True)
class ChecksHeld:
    """The screen's last reading of what the checks need: what was in use then, every check record read, the run
    history of the records folder in use, oldest first, each check's row and the summary, when it was read, and the
    text of each record whose result reads its text, by the record's folder (what the observations can pin down)."""
    held: check_lineage.Held
    records: tuple
    histories: tuple
    rows: tuple
    summary: check_lineage.ChecksSummary
    read_at: datetime
    texts: Mapping = field(default_factory=dict)


def _ident(record) -> "str | None":
    """A record's id, or None for none or a record that holds none."""
    ident = getattr(record, "id", None)
    return ident if isinstance(ident, str) and ident else None


#: The file of a simulation set's newest saved state, as the engine's reading of a set names the file it read.
NEWEST_STATE = "state.pt"
#: The checks whose result reads its record's text as well as its manifest.
TEXT_READ = ("pin_down",)


def prior_label(store, ident: "str | None") -> str:
    """The label of the prior whose id is ``ident``, as Records names it, from the store's listing of priors; "" when it
    is no longer there. A reading of the disk: the screen makes it only with no run live."""
    if not ident:
        return ""
    try:
        found = next((one for one in store.list("prior") if one.id == ident), None)
    except Exception:                            # noqa: BLE001 -- a store that cannot list its priors names none
        return ""
    if found is None:
        return ""
    try:
        made = datetime.fromisoformat(found.created).astimezone()
    except (TypeError, ValueError, OverflowError, OSError):
        made = None
    return record_words.record_label("prior", found.name or "", made)


def _completed(text: "str | None") -> "datetime | None":
    """When a simulation set was completed, in local time, from its manifest's text: its making and the seconds it took;
    None when the text does not say both."""
    try:
        manifest = json.loads(text)
        made = datetime.fromisoformat(manifest["created"])
        seconds = manifest["body"]["wall_seconds"]
        if made.tzinfo is None or isinstance(seconds, bool) or not isinstance(seconds, (int, float)):
            return None
        return (made + timedelta(seconds=float(seconds))).astimezone()
    except (TypeError, ValueError, KeyError, OverflowError):
        return None


def _the_sets_training(simulation: "str | None") -> tuple:
    """Of the simulation set whose id is ``simulation``, as the run history holds its runs: the comparison with the audit
    the newest training or narrowing round of it keeps, with the answers given to it and that run's id (None for none),
    and whether any run of the history names the set. A reading of the run history, by the set's id: the screen makes
    it only with no run live."""
    if not simulation:
        return None, False
    from ..run import history
    from ..run.questions import TRAINING_WORDS
    try:
        runs = history.runs_for_set_id(controller().folder(), simulation)
    except OSError:                              # a folder of runs that cannot be read names no set
        return None, False
    for record in reversed(runs):
        compared = (record.get("health") or {}).get("comparison")
        if record.get("word") in TRAINING_WORDS and isinstance(compared, Mapping):
            return {**compared, "run": record.get("id")}, True
    return None, bool(runs)


def _name_trouble(store, typed: str) -> "str | None":
    """Why the store would refuse ``typed`` as a posterior's name, in the window's words; None when it is free, or when
    the store cannot be read now (the training stage reads it again at its entry)."""
    from ..widgets.form import name_trouble
    return name_trouble("posterior", typed.strip(), store=store)


class InferenceScreen(QWidget):
    """Parameter inference: its six steps as pages over one shared SbiSession.

    Drives the workflow from a model to a narrowing round and owns its gating: ``refresh_gates`` runs every
    step's own gate after every run of a step, then the steps' states. Remembers the model and the kind of observation
    when the first step is applied; each other panel saves its own settings group. ``states_changed`` carries the six
    steps' states each time they are worked out again.
    ``pages`` holds the six steps in their order, its height following the step shown; ``show_step`` shows one
    (``show_page`` is the same) and ``current_step`` (or ``current_page``) names the one shown.
    """
    states_changed = Signal(object)
    # A page asks the window to go somewhere: a part, and a page of it or None.
    navigate = Signal(str, object)
    # A page's "Go to the run" was pressed.
    go_to_run = Signal()

    def __init__(self, parent=None, *, plans: "Path | None" = None):
        super().__init__(parent)
        self.session = SbiSession()
        self._states = ()
        self.pending_shown = frozenset()         # the steps the states last worked out read as changed and not applied
        # The folder of the plans, one folder for each model under it; a test hands its own.
        self.plans_folder = Path(plans) if plans is not None else Path(window_plans.PLAN_FOLDER)
        self.plan_source = None                  # where the forms were last filled from: a plan, a record or a run
        self.plan_model = None                   # the model on step 1's form as the forms were filled from it
        self.plan_file = window_plans.UNREAD     # the plan file in use as the last reading found it: its hash, or None
        self._plan_hashes: dict = {}             # each plan file's hash, kept until the file is written again
        self.loads = None                        # the last chain of loads started, going or ended
        # A sentence said under the heading of one place, "home" or a step's key, until the next press or run: where a
        # chain of loads stopped, or why a press could not start one.
        self._notice = ("", "")
        # The configuration a Resume put in use, with the device and the width it runs on: (the configuration, the kind
        # of device, the kept training's width or None), until step 1 or step 2 is applied again.
        self._resumed = None
        # What the run history and the kept trainings offer, as last read: none before the first reading, and the last
        # kept while a run is live.
        self._offers: tuple = ()
        # The last reading of what the checks need (``read_checks``): none before the first, the last kept while a run
        # is live; the configuration the checks' words of it were last worked out for, with them; and the checks' rows
        # last worked out, with what was in use for them.
        self.checks_held: ChecksHeld | None = None
        self._cfg_words: tuple = (None, None, None)
        self._rows_for: tuple = (None, None)

        self.config_panel = ConfigPanel(self)
        self.prior_panel = PriorPanel(self)
        self.posterior_panel = PosteriorPanel(self)
        self.validate_panel = ValidatePanel(self)
        self.infer_panel = InferPanel(self)
        # The narrowing round sits AFTER Infer, and the order is the workflow: a round needs an observation,
        # and the Infer step is what records one (an amortized posterior has none at save time).
        self.tsnpe_panel = TSNPEPanel(self)
        # The six steps as pages, in their order and with no tab bar: the rail is the way between them. The
        # stack's height follows the step shown, because the six pages differ wildly -- Validate is one label
        # and one button, Infer in chi mode is 10-25 rows -- and a stack sized to the largest left the short
        # pages a large dead gap.
        self.pages = AdaptiveStack()
        for panel in self.panels():
            self.pages.addWidget(panel)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.pages, 1)

        controller().status_changed.connect(self._status_changed)
        controller().run_ended.connect(self._run_ended)
        self.refresh_gates()

    def panels(self):
        return [self.config_panel, self.prior_panel, self.posterior_panel,
                self.validate_panel, self.infer_panel, self.tsnpe_panel]

    def step_pages(self) -> tuple:
        """The steps a plan fills, in their order: all six, Checks by the calibration's form; while the screen is being
        built, those built so far."""
        return tuple(getattr(self, name) for name in ("config_panel", "prior_panel", "posterior_panel",
                                                      "validate_panel", "infer_panel", "tsnpe_panel")
                     if hasattr(self, name))

    def set_small(self, on: bool) -> None:
        """The low window, or the ordinary one again: told to every step that offers it."""
        for panel in self.panels():
            told = getattr(panel, "set_small", None)
            if callable(told):
                told(on)

    def first_focus(self):
        """The control the step shown names for the keyboard to start at; None when it names none, and the keyboard
        then starts at its first setting."""
        first = getattr(self.pages.currentWidget(), "first_focus", None)
        return first() if callable(first) else None

    def stale_store(self) -> None:
        """The store changed: each step that reads it reads again now when it is shown, else at its next show; and what
        the checks need is read again now, by a step in sight that reads the checks, else by the screen."""
        for panel in self.panels():
            told = getattr(panel, "stale_store", None)
            if callable(told):
                told()
        self._read_checks_unless_shown()

    def _run_ended(self, _status) -> None:
        """A run has ended: what the checks need is read again, by a step in sight that reads the checks at a run's end,
        else by the screen."""
        self._read_checks_unless_shown()

    def _read_checks_unless_shown(self) -> None:
        if not any(getattr(panel, "reads_the_checks", False) and panel.isVisible() for panel in self.panels()):
            self.read_checks()

    # ── what the checks need ──
    def read_checks(self) -> None:
        """The one reading of what the checks need, held as ``checks_held``: every check record's manifest, the run
        history of the records folder in use, and for the posterior in use the counts of unusable driven recordings its
        simulation set holds (from its saved state, else from the posterior's own text) and whether the engine refuses
        the prior in use with it; then the steps' states worked out again. Nothing while a run is live, the last reading
        standing."""
        if controller().is_live():
            return
        from core.artifacts import default_store
        from ..run import history
        store = default_store()
        records = check_lineage.read_records(store)
        histories = tuple(history.runs_of(controller().folder(), records_root=str(store.root)))
        held = self._checks_in_use(read=store)
        rows = check_lineage.rows(held, records, histories)
        # What the observations can pin down says from its text from which recording length the low end of the band
        # holds enough cycles: the text of the record that decides it, read by its folder.
        texts = {row.record.folder: check_lineage.read_text(row.record.folder) for row in rows
                 if row.key in TEXT_READ and row.record is not None}
        self.checks_held = ChecksHeld(held, records, histories, rows, check_lineage.summary(rows, held=held),
                                      controller().now(), texts)
        self._rows_for = (held, rows)
        self.recompute_states()

    def checks_in_use(self) -> check_lineage.Held:
        """What is in use, as the checks' rules read it, with what the last reading found of the posterior's set and of
        the prior it was trained on when the posterior is the one it read; reads nothing."""
        return self._checks_in_use()

    def check_rows(self) -> "tuple | None":
        """Each check's row, from the last reading with what the session holds now, reading nothing: the counts of the
        posterior's set and the engine's word on the prior with it are kept from the last reading for the same records,
        else not known. None before the first reading."""
        if self.checks_held is None:
            return None
        held = self._checks_in_use()
        kept_for, kept = self._rows_for
        if kept_for != held:
            kept = check_lineage.rows(held, self.checks_held.records, self.checks_held.histories)
            self._rows_for = (held, kept)
        return kept

    def pre_run_rows(self) -> "tuple | None":
        """The checks' rows for what a training started now would use, from the last reading, reading nothing: the
        prior in use alone, never the prior a posterior held beside it was trained on, so that the frequency band and
        the audit said before a training are the configuration's and the prior's in use. None before the first
        reading."""
        if self.checks_held is None:
            return None
        held = replace(self._checks_in_use(), posterior=None, posterior_prior=None, simulation=None)
        return check_lineage.rows(held, self.checks_held.records, self.checks_held.histories)

    def checks_summary(self) -> "check_lineage.ChecksSummary | None":
        """The checks' summary the chain of steps and Home read, worked out from the last reading with what the session
        holds now, reading nothing; None before the first reading."""
        found = self.check_rows()
        return None if found is None else check_lineage.summary(found, held=self._checks_in_use())

    def _cfg_checks(self, cfg) -> tuple:
        """The fields of the configuration block a record made on ``cfg`` holds, by the engine's own writer, and the
        settings of the frequency response a probe check's record states, by the engine's own (None without it, or for
        a configuration that cannot say them); worked out once for each configuration put in use. Reads nothing."""
        if self._cfg_words[0] is cfg and cfg is not None:
            return self._cfg_words[1], self._cfg_words[2]
        block = configured = None
        if cfg is not None:
            try:
                from core.artifacts.manifest import config_from_cfg
                whole = config_from_cfg(cfg)
                block = {name: whole.get(name) for name in check_lineage.CONFIG_FIELDS}
            except Exception:                    # noqa: BLE001 -- a configuration that cannot say them matches nothing
                block = None
            if getattr(cfg, "chi_mode", False) is True:
                try:
                    from core.diagnostics.probes import _configured
                    configured = _configured(cfg)
                except Exception:                # noqa: BLE001 -- as above
                    configured = None
        self._cfg_words = (cfg, block, configured)
        return block, configured

    def _checks_in_use(self, *, read=None) -> check_lineage.Held:
        """What is in use, as the checks' rules read it. Handed the store to ``read``, the posterior's set and text are
        read and the engine is asked about the prior in use with the posterior; else those are kept from the last
        reading for the same prior, posterior and set, and are not known for others."""
        session = self.session
        cfg, prior, posterior = (getattr(session, name, None) for name in ("cfg", "inf_prior", "posterior"))
        manifest = getattr(posterior, "manifest", None)
        parents, body = getattr(manifest, "parents", None), getattr(manifest, "body", None)
        parents = parents if isinstance(parents, dict) else {}
        body = body if isinstance(body, dict) else {}
        transform = body.get("transform") if isinstance(body.get("transform"), dict) else {}
        simulation = parents.get("simulation") if isinstance(parents.get("simulation"), str) else None
        trained_on = parents.get("prior") if isinstance(parents.get("prior"), str) else None
        block, configured = self._cfg_checks(cfg)
        chi = getattr(cfg, "chi_mode", None) if cfg is not None else None
        ids = (_ident(prior), _ident(posterior), simulation)
        if read is not None:
            refused, counts, done, when = self._read_the_pair_and_the_set(read, prior, posterior, simulation)
            words = prior_label(read, trained_on) if refused else ""
            answered, named = _the_sets_training(simulation)
            same = True
        else:
            last = self.checks_held.held if self.checks_held is not None else None
            same = last is not None and (last.prior, last.posterior, last.simulation) == ids
            refused, counts, done, when, words, answered, named = (
                (last.pair_refused, last.set_counts, last.set_done, last.set_when, last.posterior_prior_words,
                 last.answered, last.set_in_history) if same else ("", None, False, None, "", None, False))
        return check_lineage.Held(
            prior=ids[0], posterior=ids[1], posterior_prior=trained_on, simulation=simulation,
            chi=chi if isinstance(chi, bool) else None, config=block, configured=configured,
            rotated=posterior is not None and transform.get("V") is not None,
            narrowed=getattr(getattr(posterior, "posterior", None), "truncation", None) is not None,
            nadrowski=getattr(cfg, "model", None) == "NADROWSKI", forcing=getattr(cfg, "has_forcing", False) is True,
            pair_refused=refused, set_counts=counts, set_done=done,
            prior_words=record_words.phrase_of("prior", prior) if prior is not None else "",
            posterior_words=record_words.phrase_of("posterior", posterior) if posterior is not None else "",
            set_when=when, set_read=same, posterior_prior_words=words, answered=answered, set_in_history=named)

    @staticmethod
    def _read_the_pair_and_the_set(store, prior, posterior, simulation) -> tuple:
        """What the reading finds of the posterior in use: the engine's sentence refusing the prior in use with it (""
        when it passes, or cannot tell); its set's counts of unusable driven recordings, from the set's saved state when
        that is the set's newest state and holds one pair of counts for each batch done, else from the posterior's own
        text; whether the set is complete; and when it was completed. With no posterior, nothing. The state kept from
        before the newest, which is read when the newest cannot be, counts an older number of batches than the set
        holds, so its counts are not taken for the set's."""
        if posterior is None:
            return "", None, False, None
        refused = ""
        try:
            from core.SBI.run_guards import _assert_prior_used_matches_posterior
            _assert_prior_used_matches_posterior(getattr(posterior, "posterior", None), getattr(prior, "prior", None),
                                                 "The calibration")
        except Exception as error:               # noqa: BLE001 -- the engine's refusal, or a pair it cannot read
            refused = str(getattr(error, "message", None) or error) if getattr(error, "field", None) == "prior" else ""
        counts, done, when = None, False, None
        if simulation is not None:
            from core.SBI import training_checkpoint
            folder = store.kind_dir("simulation") / simulation
            state = training_checkpoint.peek(folder) or {}
            masked, batches = state.get("chi_masked"), state.get("batches_done")
            done = state.get("complete") is True
            try:
                if (state.get("_state_file") == NEWEST_STATE and isinstance(masked, list)
                        and isinstance(batches, int) and batches > 0 and len(masked) == batches
                        and all(isinstance(pair, (list, tuple)) and len(pair) == 2 for pair in masked)):
                    counts = (int(sum(pair[0] for pair in masked)), int(sum(pair[1] for pair in masked)), batches,
                              check_readings.WHOLE_SET_SCOPES[0])
            except (TypeError, ValueError):      # a state of another shape holds no counts to read
                counts = None
            when = _completed(check_lineage.read_text(str(folder), check_lineage.MANIFEST))
        if counts is None:
            folder = getattr(posterior, "path", None)
            counts = check_readings.first_masked(check_lineage.read_text(str(folder))) if folder is not None else None
        return refused, counts, done, when

    def pending(self) -> frozenset[str]:
        """The steps whose page shows a choice other than what the step holds: each page that has rows that pick what
        its step holds is asked."""
        return frozenset(panel.step for panel in self.panels()
                         if any(row.changed() for row in getattr(panel, "selection_rows", tuple)()))

    def show_step(self, name: str) -> None:
        """Show the step whose key is ``name``, whether or not it can start; KeyError, naming it, for a key that
        is no step's."""
        self.pages.setCurrentIndex(keys.step_number(name) - 1)

    show_page = show_step

    def open_with(self, step: str, carries) -> None:
        """A next step of a result was pressed: the step ``step`` opens, then its page takes what the next step
        ``carries``, on a page that takes anything. KeyError, naming it, for a key
        that is no step's, before anything moves."""
        page = self.panels()[keys.step_number(step) - 1]
        self.navigate.emit("inference", step)
        take = getattr(page, "take_carried", None)
        if callable(take):
            take(carries)

    def current_step(self) -> str:
        """The key of the step shown: the first, Model and options, until another is shown."""
        return keys.STEPS[self.pages.currentIndex()].key

    current_page = current_step

    def session_contents(self) -> list[str]:
        """Plain phrases for the records the session holds -- the prior, the posterior and the observation -- in the
        steps' order, each by its label; empty when a ``new_draft`` would let go of nothing.

        THE DRAFT IS DELIBERATELY NOT IN THE LIST. ``new_draft`` replaces the draft, and replacing it is the point of
        applying the first step: a list that named it would make every deliberate change of model look like a loss.
        Nor is the configuration built from it, which is never held without its prior.

        NOTHING HERE MAY RAISE: each record is read as its label reads it, which a bare ``object()`` passes. The phrases
        carry no code's names and name no control, since the first step's question quotes them."""
        s = self.session
        held = [(word, getattr(s, slot, None)) for word, slot in
                (("prior", "inf_prior"), ("posterior", "posterior"), ("observation", "observation"))]
        return [record_words.phrase_of(word, record) for word, record in held if record is not None]

    def new_draft(self, draft):
        """Config applied: replace the WHOLE session (a different model or unit system invalidates
        every artifact) and repoint the Prior step's bounds picker at the new model's folder.

        THE DESTRUCTIVE ONE OF THE SCREEN'S TWO ENTRY POINTS, AND THE CONFIRMATION IS NOW THE GUARD.
        The first step asks before it calls here whenever ``session_contents()`` is non-empty,
        defaulting to keeping the session, so a mis-aimed Apply can no longer drop a prior, a
        posterior and a recorded observation in silence -- which is the half of mixing up replacing
        the session's configuration and installing into it that an operator could not defend
        against.

        Nothing is deleted either way: every stage writes its artifact at completion, so what this
        drops is the session's HANDLES, and the artifacts stay on disk to be selected again. Its
        in-place twin is ``install_config``.

        The model and the kind of observation applied are remembered here, in one write, as they come into use: the
        next launch puts them back as choices. A value the settings file could not give back as it was written, from a
        draft that is not the window's own, is not written. The sentence a press left goes: step 1 applied is the next
        press, and a chain that applies it says where it stops after that.
        """
        self._notice = ("", "")                  # drawn with the gates, worked out again below
        self.session = SbiSession(draft=draft)
        model, kind = getattr(draft, "model", None), getattr(draft, "chi_mode", None)
        if self.plan_source is not None and model != self.plan_model:
            # Applied with another model than the one the forms were filled for: the plan in use goes. A model only
            # shown on the form lets nothing go. A plan's bounds file that waits is selected for its own model alone.
            self.plan_source, self.plan_model, self.plan_file = None, None, window_plans.UNREAD
        self.prior_panel.on_draft_set(draft)
        picks = [(key, value) for key, value, fits in ((remembered.MODEL_KEY, model, isinstance(model, str)),
                                                       (remembered.KIND_KEY, kind, isinstance(kind, bool))) if fits]
        picks += self.last_used(model)
        if picks:
            remembered.put_many(picks)
        self.refresh_gates()

    def install_config(self, cfg):
        """A configuration set IN PLACE, then the gates and the states worked out again: each step reads the
        configuration in use as it draws. The Prior step no longer calls this: its configuration comes into use with
        its prior, through ``take_prior``. Its destructive twin ``new_draft`` is the one the first step confirms."""
        self._fan_out(cfg)
        self.refresh_gates()

    def _fan_out(self, cfg) -> None:
        """Hold ``cfg``: the steps whose rows depend on it read it from the session as they draw."""
        self.session.cfg = cfg

    def take_prior(self, cfg, loaded) -> None:
        """A prior has come back from its run, made on ``cfg``, the very configuration the Prior step handed the run:
        what depended on the prior held is let go, the two come into use together, the bounds file, with the model it
        came into use with, and the prior are remembered as the picks last in use, and the gates and the states are
        worked out once. Called while the run is still live, so it writes the session and nothing else reads: the Prior
        step reads its lists again when the run has ended."""
        session = self.session
        session.reset_downstream("prior")
        session.inf_prior = loaded
        self._fan_out(cfg)
        key = bounds_in_use(cfg)
        picks = [(remembered.PRIOR_KEY, getattr(loaded, "id", None))]
        if key:
            picks[:0] = [(remembered.BOUNDS_KEY, key), (remembered.BOUNDS_MODEL_KEY, getattr(cfg, "model", None))]
        remembered.put_many([(name, value) for name, value in picks if isinstance(value, str) and value]
                            + self.last_used(getattr(getattr(session, "draft", None), "model", None)))
        self.refresh_gates()

    def take_posterior(self, loaded) -> None:
        """A posterior has come back from its run, a training's, a load's or a narrowing round's, made with the
        configuration and the prior held: what depended on the posterior held is let go, the new one comes into use, it
        is remembered as the posterior last in use, and the gates and the states are worked out once. Called while the
        run is still live, so it writes the session and nothing else reads: the Training step reads, and moves its list
        to the posterior, when the run has ended. The posterior held before stays in use through the run."""
        session = self.session
        session.reset_downstream("posterior")
        session.posterior = loaded
        ident = getattr(loaded, "id", None)
        picks = [(remembered.POSTERIOR_KEY, ident)] if isinstance(ident, str) and ident else []
        remembered.put_many(picks + self.last_used(getattr(getattr(session, "draft", None), "model", None)))
        self.refresh_gates()

    def take_observation(self, observation) -> None:
        """An observation has come back from an inference's run or a load: it comes into use, it is remembered as the
        observation last in use, and the gates and the states are worked out once. Called while the run is still live,
        so it writes the session and nothing else reads."""
        self.session.observation = observation
        ident = getattr(observation, "id", None)
        picks = [(remembered.OBSERVATION_KEY, ident)] if isinstance(ident, str) and ident else []
        remembered.put_many(picks + self.last_used(getattr(getattr(self.session, "draft", None), "model", None)))
        self.refresh_gates()

    # ── a sentence a press leaves ──
    def notice_on(self, where: str) -> str:
        """The sentence said under the heading of ``where`` ("home", or a step's key): where the last chain of loads
        stopped, or why a press could not start one; "" for none."""
        return self._notice[1] if self._notice[0] == where else ""

    def set_notice(self, where: str, sentence: str) -> None:
        """Say ``sentence`` under the heading of ``where`` until the next press or run; the steps and Home drawn
        again."""
        self._notice = (where, sentence)
        self.refresh_gates()

    def clear_notice(self) -> None:
        """The next press or run: the sentence goes."""
        if self._notice[1]:
            self._notice = ("", "")
            self.refresh_gates()

    def _status_changed(self, status) -> None:
        """A run has begun: the sentence a press left goes."""
        if getattr(status, "live", False):
            self.clear_notice()

    def resumed_on(self) -> "tuple[str, int | None] | None":
        """The kind of device ("cpu", "cuda") a Resume put the session on, and the kept training's width when it is
        its own, while the configuration that Resume put in use is held: until step 1 or step 2 is applied again. None
        else."""
        held = self._resumed
        if held is None or held[0] is not getattr(self.session, "cfg", None):
            return None
        return held[1], held[2]

    # ── what a reopened window offers ──
    def offers(self) -> tuple:
        """What the run history and the kept trainings offer, newest first, as last read (``read_offers``)."""
        return self._offers

    def seed_offers(self, offers) -> None:
        """What the run history and the kept trainings offered at the window's last reading, held as the screen's own
        until it reads again, and the steps' states worked out with it. Nothing is read."""
        self._offers = tuple(offers or ())
        self.recompute_states()

    def read_offers(self) -> None:
        """Read what the run history and the kept trainings offer, and work out the steps' states again; the Training
        step draws its list of kept trainings from it. While a run is live nothing is read and the last reading
        stands."""
        if controller().is_live():
            return
        from core.artifacts import default_store
        self._offers = resuming.read_offers(default_store(), controller().folder())
        self.recompute_states()
        self.posterior_panel.redraw()

    def _lets_go(self, title: str, other: str) -> bool:
        """Whether what the session holds may be let go for a Continue or a Resume: asked first, "Keep this session"
        preselected, when the session holds a record; True with nothing held."""
        held = self.session_contents()
        if not held:
            return True
        from ..widgets.ask import ask
        return ask(self, title, CONTINUE_LETS_GO_TEXT.format(", ".join(held)), KEEP_SESSION, other)

    def _continue_problems(self, model: str, bounds: str, prior) -> list:
        """What a Continue's chain would find gone, each in a few words: the run's prior, no longer in Records, and its
        bounds file, no longer in the model's bounds folder (or where its path names it). Read with no run live."""
        from core.artifacts import default_store
        from ..panels.inference.prior_tab import BOUNDS_PATH
        found = []
        try:
            if not isinstance(prior, str) or not prior:
                raise LookupError(prior)
            default_store().get("prior", prior)
        except Exception:                    # noqa: BLE001 -- deleted, torn, or never named
            found.append(CONTINUE_PRIOR_GONE)
        path = Path(bounds) if Path(bounds).is_absolute() else BOUNDS_PATH / model.lower() / bounds
        if not path.is_file():
            found.append(CONTINUE_BOUNDS_GONE.format(Path(bounds).name))
        return found

    def continue_from(self, offer, where: str = "home") -> "LoadChain | None":
        """"Continue from batch N", or "Fit again on the saved simulations", for the training of the run ``offer`` names,
        after a reopening: its history read again, asked first when the session holds a record, then step 1 applied with
        the model and the options the run was started with, the Training step's forms filled from its settings once the
        model stands, its bounds file selected and its prior loaded by step 2's own load, and, once the prior is loaded,
        the Training step set to continue: "Only continue", the name and the note blank with the earlier name as the
        hint. The press is the next press: what an earlier press left under a heading goes first. Its prior and its
        bounds file are read then: one gone is said under the Training step's heading, the step shown with its form when
        ``where``, the place the press was made, is "home" (a past view of the Training step, "training", stands where
        it is and says it too), and nothing is let go. Once the step is shown, its form is in sight, not the page of a
        run that is over. The chain of loads is handed back; None when nothing started: a run live, a history gone or no
        longer naming its settings, its prior or its bounds file gone, or the question answered "Keep this session"."""
        from ..run import history
        if controller().is_live() or offer is None or not offer.run:
            return None
        self.clear_notice()
        record = history.read(controller().folder() / f"{offer.run}.json") or {}
        settings_ = record.get("settings")
        if not isinstance(settings_, dict) or not isinstance(settings_.get("bounds"), str) or not settings_["bounds"]:
            return None
        model = settings_.get("model")
        if not isinstance(model, str) or not model:
            return None
        gone = self._continue_problems(model, settings_["bounds"], settings_.get("prior"))
        if gone:
            # Read before the question: what would stop the chain after the session was let go is said now, at the head
            # of the step the press fills, and nothing is let go.
            self.set_notice("training", CONTINUE_CANNOT_TEXT.format(offer.button, "; and ".join(gone)))
            if where == "home":
                self.navigate.emit("inference", "training")
                self.posterior_panel.to_the_form()
            return None
        other = FIT_AGAIN_OTHER if offer.kind == "fit_again" else CONTINUE_OTHER
        if not self._lets_go(CONTINUE_TITLE.format(offer.button), other):
            return None
        rows = settings_.get("rows") if isinstance(settings_.get("rows"), list) else []
        values = {(window_plans.HEADING_OF["training"], row["name"]): row.get("value") for row in rows
                  if isinstance(row, dict) and isinstance(row.get("name"), str)}
        source = window_plans.PlanSource("run", str(record.get("name") or offer.run), None, "", controller().now(),
                                         values, ident=offer.run)
        options = settings_.get("options") if isinstance(settings_.get("options"), dict) else {}
        options = {**options, "chi_mode": settings_.get("chi_mode") is True}
        units = options.get("units_override")
        # Units typed are a list of words in the history; a units file other than the model's, its path.
        options["units_override"] = (tuple(str(unit) for unit in units) if isinstance(units, list)
                                     else units if isinstance(units, str) and units else None)

        def fill() -> None:
            self.config_panel.fill_with(source)

        loads = self.load_chain(model, options, settings_["bounds"], settings_.get("prior"), fill=fill,
                                press=offer.button, where="training")
        loads.finished.connect(lambda: self.posterior_panel.continue_run(offer, settings_))
        self.navigate.emit("inference", "training")
        self.posterior_panel.to_the_form()
        return loads

    def resume_kept(self, ident: str) -> "LoadChain | None":
        """"Resume" for the kept training ``ident``, from Home, the Training step's list or a past view: what stops it
        said on its row before anything is read; a name asked for when its record holds none, or when another posterior
        has taken its name since ("Cancel" starts nothing); asked first when the session holds a record; then its model
        and options, its bounds file and its prior put in use from its record by a chain of loads, the prior loaded on
        the configuration its record rebuilds, and, once the prior is loaded, the training resumed on its own record.
        The step's form is in sight once it is shown, not the page of a run that is over. The press is the next press:
        what an earlier press left under a heading goes first. The chain is handed back; None when nothing started."""
        from core import config as prism_config
        from core.artifacts import default_store
        from core.SBI import fit_state
        from ..widgets.ask import ask_text
        if controller().is_live():
            return None
        self.clear_notice()
        if self.posterior_panel._past_view is not None:
            self.posterior_panel.past_view.say("")       # what a press on a past view left there goes too
        store = default_store()
        try:
            kept = fit_state.kept_fit(store, ident)
        except Exception as error:           # noqa: BLE001 -- gone or torn since the list was read
            self.navigate.emit("inference", "training")
            self.posterior_panel.to_the_form()
            self.posterior_panel.say_on_row(ident, str(getattr(error, "message", None) or error))
            return None
        problems = (continuing.as_sentence(kept.why_not),) if not kept.can_resume else resuming.resume_problems(
            kept, device=prism_config.detect_device(), store=store)
        if problems:
            # Shown first: the step's reading as it comes into sight reads the list again, which says no sentence.
            self.navigate.emit("inference", "training")
            self.posterior_panel.to_the_form()
            self.posterior_panel.say_on_row(ident, " ".join(problems))
            return None
        name = ""
        # A name another posterior has taken since: the stage would refuse it, and its sentence names a new training's.
        taken = bool(kept.posterior_name) and _name_trouble(store, kept.posterior_name) is not None
        if not kept.posterior_name or taken:
            from ..panels.inference.posterior_tab import NAME_TAKEN_TEXT, NAME_TEXT, NAME_TITLE
            from ..widgets.form import NAME_ID_TEXT, NAME_SHAPE_TEXT
            sentence = NAME_TAKEN_TEXT.format(kept.posterior_name) if taken else NAME_TEXT
            name = ask_text(self, NAME_TITLE, sentence, "Name", "Resume",
                            check=lambda typed: _name_trouble(store, typed),
                            room=(NAME_SHAPE_TEXT, NAME_ID_TEXT)) or ""
            if not name:
                return None
        if not self._lets_go(RESUME_TITLE, RESUME_OTHER):
            return None
        record = kept.config or {}
        # The units as its record names them: the model's file (None), another file by its path, or the units typed.
        options = {"chi_mode": bool(record.get("chi_mode")), "chi_k_pad": record.get("chi_k_pad"),
                   "chi_max_cycles": record.get("chi_max_cycles"), "reparam_rotate": record.get("reparam_rotate"),
                   "units_override": resuming.resume_draft(kept).units_override}
        cfg = resuming.resume_config(kept)
        width = record.get("run_size")
        own = isinstance(width, int) and not isinstance(width, bool) and width > 0
        self._resumed = (cfg, getattr(getattr(getattr(cfg, "hw", None), "device", None), "type", None),
                         width if own else None)
        loads = self.load_chain(record.get("model"), options, resuming.bounds_key(kept), kept.parents.get("prior"),
                                candidate=cfg, press=RESUME_WORDS, where="training")
        loads.finished.connect(lambda: self.posterior_panel.resume(kept, name))
        self.navigate.emit("inference", "training")
        self.posterior_panel.to_the_form()
        return loads

    # ── a chain of loads ──
    def load_chain(self, model, options, bounds_file, prior, posterior=None, observation=None, *, fill=None,
                   candidate=None, press: str = "", where: str = ""):
        """Step 1 applied with ``model`` and ``options``, asking nothing (``ConfigPanel.apply_quietly``, which calls
        ``fill`` once the model stands on the form), ``bounds_file`` then selected in step 2's list when one is named,
        and the saved ``prior`` loaded by step 2's own "Load this prior", then, once it has returned its record, the saved
        ``posterior`` by step 3's, then, once that has returned its record, the saved ``observation`` by the narrowing
        round's load: a chain of loads, started and handed back. A load that does not start or does not finish ends the
        chain, and what came before it stays. A step 1 that cannot be applied, or a chain asked for while another goes,
        starts nothing, and the chain stops at once. ``candidate``
        is the configuration the prior is loaded with, when the caller holds one (a kept training's, rebuilt from its
        record), else step 2 builds its own. With ``press``, the words of the button that asked for the chain, a chain
        that stops before its end says so under the heading of ``where``, the place the press opened ("home", or a
        step's key), naming the step it stopped at and what was not loaded, until the next press or run; and for a model
        step 1 cannot take, which its page never shows, why."""
        loads = LoadChain(self)
        going = self.loads is not None and self.loads.is_going()
        if not going:
            self.loads = loads
            self.clear_notice()
            if press:
                cannot = self.config_panel.cannot_take(model)
                loads.at = ("model", f"the model {chain.model_label(model)}", "applied")
                loads.stopped.connect(lambda why: self._chain_stopped(loads, press, where, why, cannot))
        applied = not going and self.config_panel.apply_quietly(model, options, fill=fill)
        if applied and (bounds_file or candidate is not None):
            # A configuration handed in stands for bounds the list may not hold: the list then selects none.
            self.prior_panel.bounds_picker.restore_key(bounds_file or "")
        links = [] if applied else [lambda: False]

        def at(step, kind, ref, call):
            def link() -> bool:
                loads.at = (step, self._record_named(kind, ref), "loaded")
                return call()
            return link

        if applied and prior:
            links.append(at("prior", "prior", prior, lambda: self.prior_panel.load_record(prior, candidate=candidate)))
            if posterior:
                links.append(at("training", "posterior", posterior,
                                lambda: self.posterior_panel.load_record(posterior)))
                if observation:
                    links.append(at("narrowing", "observation", observation,
                                    lambda: self.tsnpe_panel.load_observation(observation)))
        loads.start(links)
        return loads

    def _record_named(self, kind: str, ref) -> str:
        """A saved prior, posterior or observation as a sentence names it, by its label in its step's list: "the prior
        retrain_prior"; "the prior" alone when the list does not hold it."""
        picker = {"prior": self.prior_panel.prior_picker, "posterior": self.posterior_panel.post_picker,
                  "observation": self.tsnpe_panel.obs_picker}[kind]
        found = picker.combo.findData(ref)
        label = picker.combo.itemText(found) if found >= 0 else ""
        return f"the {kind} {label}".rstrip() if label else f"the {kind}"

    def _chain_stopped(self, loads, press: str, where: str, why: str, cannot: str = "") -> None:
        """A chain of loads a press asked for stopped before its end: the step it stopped at, and what was not loaded,
        said under the heading of the page the press opened; at step 1, for a model it ``cannot`` take, why. A chain
        stopped because the window closed says nothing."""
        if why == "closed" or getattr(loads, "at", None) is None:
            return
        step, what, done = loads.at
        name = keys.step(step).name
        if step == "model" and cannot:
            self.set_notice(where, CHAIN_CANNOT_TAKE_TEXT.format(press, name, cannot))
            return
        self.set_notice(where, CHAIN_STOPPED_TEXT.format(press, name, what, done, name))

    def continue_with(self, picks):
        """What was in use last time, applied and loaded as one press of Home's "Continue with these" asks: the plan
        last used, read and checked again now, filled into the forms once the model stands, keeping the values put back
        from last time and asking nothing, the press being the person's own; step 1 applied with the model, the kind of
        observation remembered and the plan's other settings of that step, every other at its usual value; then the
        bounds file selected and the prior, the posterior and the observation loaded, each by its own step's load. A
        plan gone, or with
        problems, since Home read it is left out, and the rest goes on. The chain of loads is handed back. While a run
        is live it reads nothing, neither the plan nor its folder, changes nothing, and hands back a chain that stops at
        once. When it would set a choice made on the forms and not applied (step 1's model, kind, rotation, units or
        plan shown, and each setting the plan fills on the other steps), it asks first, "Keep what is there"
        preselected, as the plan's own fill does; kept, nothing is read further and None is handed back."""
        if controller().is_live():
            return self.load_chain(picks.model, {}, None, None)
        plan = remembered.last_plan(picks.model, picks.plan, plans_folder=self.plans_folder) if picks.plan else None
        options, fill, source = {}, None, None
        if plan is not None:
            source = window_plans.source_of(plan, controller().now())
            options = {_DRAFT_NAMES.get(name, name): value for (heading, name), value in source.values.items()
                       if heading == window_plans.HEADING_OF["model"]}

            def fill() -> None:
                self.config_panel.fill_with(source, keep_remembered=True)

        options["chi_mode"] = picks.chi_mode is True
        if options.get("units_override") is not None:
            options["units_override"] = tuple(str(options["units_override"]).split())
        replaced = self._continuing_replaces(picks.model, options, source)
        if replaced:
            from ..widgets.ask import ask
            if not ask(self, CONTINUE_WITH_TITLE, CONTINUE_WITH_TEXT, KEEP_WHAT_IS_THERE, CONTINUE_WITH_WORDS,
                       items=replaced):
                return None
        return self.load_chain(picks.model, options, picks.bounds, picks.prior, picks.posterior,
                               getattr(picks, "observation", None), fill=fill, press=CONTINUE_WITH_WORDS, where="home")

    def _continuing_replaces(self, model, options: dict, source) -> tuple:
        """What "Continue with these" would set that a person chose on the forms and did not apply, or that was put
        back and is not what the press puts in use: the model on step 1's form, each of step 1's settings, the plan
        shown in its list, and, with a plan, each setting of steps 2 and 3 the plan fills over what was typed, as
        "Rotation: off becomes on". A value at its usual one, or put back from last time and kept by the press, is no
        choice of a person's. With no plan last used, a value filled from the plan in use and not applied is one: the
        press sets step 1 whatever that plan holds, and the plan stays in use."""
        page = self.config_panel
        found = []
        row, shown = page.form.rows["model"], page.model_combo.currentText()
        if shown != model and not row.at_usual() and not row.kept:
            found.append(REPLACED_TEXT.format(row.label_text(), chain.model_label(shown), chain.model_label(model)))
        listed = page.plan_choice.currentData()
        after = source.name if source is not None else (listed if shown == model else None)
        if listed is not None and listed != after:
            found.append(REPLACED_TEXT.format(page.form.rows["plan"].label_text(), listed, after or NO_PLAN_WORDS))
        heading = window_plans.HEADING_OF["model"]
        values = {(heading, name): value for name, value in (("chi_mode", options.get("chi_mode")),
                                                             *((key, options.get(key)) for key in
                                                               ("chi_n_freqs", "chi_k_pad", "chi_max_cycles",
                                                                "reparam_rotate")))
                  if value is not None}
        units = options.get("units_override")
        values[(heading, "units")] = " ".join(str(unit) for unit in units) if isinstance(units, (tuple, list)) else None
        if source is not None:
            values = {**source.values, **values}
        found += window_plans.would_replace(self, values, steps=None if source is not None else ("model",),
                                            keep_remembered=True, source_kept=source is not None)
        return tuple(found)

    # ── the plan in use ──
    def take_plan(self, source) -> None:
        """The forms were filled from ``source``, or returned to their usual values (None): it is the plan in use, and
        every step is drawn again, its origins and its plan line with it. A plan's bounds file that waits for another
        model than this source's, or with no source in use, goes with the plan it came from. Nothing is applied and
        nothing is read."""
        self.plan_source = source
        self.plan_model = None if source is None else self.config_panel.model_combo.currentText()
        self.plan_file = source.sha256 if source is not None and source.kind == "plan" else window_plans.UNREAD
        waiting = self.prior_panel.waiting_model()
        if waiting is not None and (source is None or waiting != self.plan_model):
            self.prior_panel.let_the_plans_file_go()
        self.refresh_gates()

    def read_plan_file(self) -> None:
        """What the plan file in use holds now, read at each step's reading: its hash, read again only when the file has
        been written since, or None when it is no longer there. Nothing while a run is live."""
        source = self.plan_source
        if source is not None and source.kind == "plan" and source.path is not None and not controller().is_live():
            self.plan_file = window_plans.file_sha(source.path, self._plan_hashes)

    def last_used(self, model) -> list:
        """The two remembered keys of the plan last used with ``model``: its name and when, or both blank when no plan
        is in use, which takes the hint away. Written with what comes into use or starts, never as a plan is filled;
        nothing for no model."""
        if not isinstance(model, str) or not model:
            return []
        source = self.plan_source
        used = source is not None and source.kind == "plan"
        return [(remembered.LAST_USED_KEY.format(model.lower()), source.name if used else ""),
                (remembered.LAST_USED_AT_KEY.format(model.lower()), controller().now().isoformat() if used else "")]

    def plan_entry(self) -> "dict | None":
        """The plan in use, as a run's history keeps it, read as the run starts: none; for a plan, its name, its file
        and that file's hash now, the hash and the moment of the fill, and whether the file has changed since,
        "missing" when it is no longer there, which stops nothing; for a record or a run, its kind, id and name. Names
        and numbers only."""
        source = self.plan_source
        if source is None:
            return None
        if source.kind != "plan":
            return {"kind": source.kind, "id": source.ident, "name": source.name}
        now = window_plans.file_ref(source.path, relative_to=config.RESOURCES_ROOT, missing_ok=True)
        changed = "missing" if now["sha256"] is None else now["sha256"] != source.sha256
        return {"kind": "plan", "name": source.name, "path": now["path"], "sha256": now["sha256"],
                "filled_sha256": source.sha256, "filled_at": source.at.isoformat() if source.at is not None else None,
                "changed": changed}

    def refresh_gates(self):
        """Run every step's own gate, which greys the buttons of a step that cannot start yet (no step is ever
        disabled as a whole), then work out the six steps' states again. The states come after the gates; the narrowing
        round's list of observations, which its state reads, is listed by the round's own reading, never by a gate."""
        for panel in self.panels():
            panel.refresh_local_gates()
        self.recompute_states()

    def recompute_states(self) -> None:
        """Work out again where each of the six steps stands, from the session, the step the live run is in
        (its place, as the run's status gives it) and whether it is stopping or waits to be tried again, what the run
        history and the kept trainings offer step 3 as last read, the checks' summary from their last reading, and
        whether the narrowing round has an observation to choose, and say so through ``states_changed``. No panel's
        gate is run: ``refresh_gates`` runs them first. Nothing is read."""
        held = controller()
        status = held.status
        place = status.place if status.live else None
        live = place[1] if place is not None and place[0] == "inference" else None
        self.pending_shown = self.pending()
        self._states = chain.step_states(self.session, live=live, stopping=status.state == "stopping",
                                         pending=self.pending_shown, checks=self.checks_summary(),
                                         observation_on_offer=self.tsnpe_panel.observation_on_offer(),
                                         label=record_words.label_of, retrying=status.state == "retrying",
                                         kept=continuing.for_the_steps(self._offers))
        self.states_changed.emit(self._states)

    def step_states(self) -> tuple:
        """The six steps' states as last worked out, in the steps' order."""
        return self._states
