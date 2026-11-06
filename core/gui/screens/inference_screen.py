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

The screen also keeps the plan in use: where the forms of steps 1 to 3 were last filled from, which outlives the session
that step 1 replaces as it is applied, and what the steps' last reading found of that plan's file.

And it runs a chain of loads: step 1 applied with a model and its options, then each later step's own "Load this …" of
a saved record, one after the other, each once the one before has returned its record; Home's "Continue with these"
is one.
"""
from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QVBoxLayout, QWidget

from core import config

from .. import chain, continuing, keys, record_words, remembered
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
        self.plan_file = window_plans.UNREAD     # the plan file in use as the last reading found it: its hash, or None
        self._plan_hashes: dict = {}             # each plan file's hash, kept until the file is written again
        self.loads = None                        # the last chain of loads started, going or ended
        # What the run history and the kept trainings offer, as last read: none before the first reading, and the last
        # kept while a run is live.
        self._offers: tuple = ()

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

        self.refresh_gates()

    def panels(self):
        return [self.config_panel, self.prior_panel, self.posterior_panel,
                self.validate_panel, self.infer_panel, self.tsnpe_panel]

    def step_pages(self) -> tuple:
        """The steps a plan fills, in their order: Model and options, Prior and Training; while the screen is being
        built, those built so far."""
        return tuple(getattr(self, name) for name in ("config_panel", "prior_panel", "posterior_panel")
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
        """The store changed: each step that reads it reads again now when it is shown, else at its next show."""
        for panel in self.panels():
            told = getattr(panel, "stale_store", None)
            if callable(told):
                told()

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
        draft that is not the window's own, is not written.
        """
        self.session = SbiSession(draft=draft)
        self.prior_panel.on_draft_set(draft)
        model, kind = getattr(draft, "model", None), getattr(draft, "chi_mode", None)
        picks = [(key, value) for key, value, fits in ((remembered.MODEL_KEY, model, isinstance(model, str)),
                                                       (remembered.KIND_KEY, kind, isinstance(kind, bool))) if fits]
        picks += self.last_used(model)
        if picks:
            remembered.put_many(picks)
        self.refresh_gates()

    def install_config(self, cfg):
        """A configuration set IN PLACE, fanned out to the steps whose pickers and fields depend on it, then the gates
        and the states worked out again. The Prior step no longer calls this: its configuration comes into use with its
        prior, through ``take_prior``. Its destructive twin ``new_draft`` is the one the first step confirms."""
        self._fan_out(cfg)
        self.refresh_gates()

    def _fan_out(self, cfg) -> None:
        """Hold ``cfg`` and hand it to the steps whose pickers and fields depend on it."""
        self.session.cfg = cfg
        self.infer_panel.on_config_built(cfg)

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

    # ── what a reopened window offers ──
    def offers(self) -> tuple:
        """What the run history and the kept trainings offer, newest first, as last read (``read_offers``)."""
        return self._offers

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

    # ── a chain of loads ──
    def load_chain(self, model, options, bounds_file, prior, posterior=None, observation=None, *, fill=None):
        """Step 1 applied with ``model`` and ``options``, asking nothing (``ConfigPanel.apply_quietly``, which calls
        ``fill`` once the model stands on the form), ``bounds_file`` then selected in step 2's list when one is named,
        and the saved ``prior`` loaded by step 2's own "Load this prior", then, once it has returned its record, the saved
        ``posterior`` by step 3's: a chain of loads, started and handed back. A load that does not start or does not
        finish ends the chain, and what came before it stays. A step 1 that cannot be applied, or a chain asked for while
        another goes, starts nothing, and the chain stops at once. An observation's link is not built yet."""
        if observation is not None:
            raise ValueError("a chain of loads takes no observation yet")
        loads = LoadChain(self)
        going = self.loads is not None and self.loads.is_going()
        if not going:
            self.loads = loads
        applied = not going and self.config_panel.apply_quietly(model, options, fill=fill)
        if applied and bounds_file:
            self.prior_panel.bounds_picker.restore_key(bounds_file)
        links = [] if applied else [lambda: False]
        if applied and prior:
            links.append(lambda: self.prior_panel.load_record(prior))
            if posterior:
                links.append(lambda: self.posterior_panel.load_record(posterior))
        loads.start(links)
        return loads

    def continue_with(self, picks):
        """What was in use last time, applied and loaded as one press of Home's "Continue with these" asks: the plan last
        used, read and checked again now, filled into the forms once the model stands, keeping the values put back from
        last time and asking nothing, the press being the person's own; step 1 applied with the model, the kind of
        observation remembered and the plan's other settings of that step, every other at its usual value; then the
        bounds file selected and the prior and the posterior loaded, each by its own step. A plan gone, or with problems,
        since Home read it is left out, and the rest goes on. The chain of loads is handed back. While a run is live it
        reads nothing, neither the plan nor its folder, changes nothing, and hands back a chain that stops at once."""
        if controller().is_live():
            return self.load_chain(picks.model, {}, None, None)
        plan = remembered.last_plan(picks.model, picks.plan, plans_folder=self.plans_folder) if picks.plan else None
        options, fill = {}, None
        if plan is not None:
            source = window_plans.source_of(plan, controller().now())
            options = {_DRAFT_NAMES.get(name, name): value for (heading, name), value in source.values.items()
                       if heading == window_plans.HEADING_OF["model"]}

            def fill() -> None:
                self.config_panel.fill_with(source, keep_remembered=True)

        options["chi_mode"] = picks.chi_mode is True
        if options.get("units_override") is not None:
            options["units_override"] = tuple(str(options["units_override"]).split())
        return self.load_chain(picks.model, options, picks.bounds, picks.prior, picks.posterior, fill=fill)

    # ── the plan in use ──
    def take_plan(self, source) -> None:
        """The forms were filled from ``source``, or returned to their usual values (None): it is the plan in use, and
        every step is drawn again, its origins and its plan line with it. Nothing is applied and nothing is read."""
        self.plan_source = source
        self.plan_file = source.sha256 if source is not None and source.kind == "plan" else window_plans.UNREAD
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
        disabled as a whole), then work out the six steps' states again. The states come after the gates: the
        narrowing round's list of observations, which its state reads, is refreshed by its gate."""
        for panel in self.panels():
            panel.refresh_local_gates()
        self.recompute_states()

    def recompute_states(self) -> None:
        """Work out again where each of the six steps stands, from the session, the step the live run is in
        (its place, as the run's status gives it) and whether it is stopping or waits to be tried again, what the run
        history and the kept trainings offer step 3 as last read, and whether the narrowing round has an observation to
        choose, and say so through ``states_changed``. No panel's gate is run: ``refresh_gates`` runs them first."""
        held = controller()
        status = held.status
        place = status.place if status.live else None
        live = place[1] if place is not None and place[0] == "inference" else None
        self.pending_shown = self.pending()
        self._states = chain.step_states(self.session, live=live, stopping=status.state == "stopping",
                                         pending=self.pending_shown,
                                         observation_on_offer=self.tsnpe_panel.obs_picker.has_entries(),
                                         label=record_words.label_of, retrying=status.state == "retrying",
                                         kept=continuing.for_the_steps(self._offers))
        self.states_changed.emit(self._states)

    def step_states(self) -> tuple:
        """The six steps' states as last worked out, in the steps' order."""
        return self._states
