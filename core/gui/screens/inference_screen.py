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
"""
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QVBoxLayout, QWidget

from .. import chain, keys, record_words, remembered
from ..panels.inference.prior_tab import bounds_in_use
from ..panels.inference_tabs import (ConfigPanel, InferPanel, PosteriorPanel, PriorPanel,
                                     TSNPEPanel, ValidatePanel)
from ..run.controller import controller
from ..session import SbiSession
from ..widgets.adaptive_stack import AdaptiveStack


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

    def __init__(self, parent=None):
        super().__init__(parent)
        self.session = SbiSession()
        self._states = ()
        self.pending_shown = frozenset()         # the steps the states last worked out read as changed and not applied

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
        what depended on the prior held is let go, the two come into use together, the bounds file and the prior are
        remembered as the picks last in use, and the gates and the states are worked out once. Called while the run is
        still live, so it writes the session and nothing else reads: the Prior step reads its lists again when the run
        has ended."""
        session = self.session
        session.reset_downstream("prior")
        session.inf_prior = loaded
        self._fan_out(cfg)
        key = bounds_in_use(cfg)
        picks = [(remembered.PRIOR_KEY, getattr(loaded, "id", None))]
        if key:
            picks.insert(0, (remembered.BOUNDS_KEY, key))
        remembered.put_many([(name, value) for name, value in picks if isinstance(value, str) and value])
        self.refresh_gates()

    def refresh_gates(self):
        """Run every step's own gate, which greys the buttons of a step that cannot start yet (no step is ever
        disabled as a whole), then work out the six steps' states again. The states come after the gates: the
        narrowing round's list of observations, which its state reads, is refreshed by its gate."""
        for panel in self.panels():
            panel.refresh_local_gates()
        self.recompute_states()

    def recompute_states(self) -> None:
        """Work out again where each of the six steps stands, from the session, the step the live run is in
        (its place, as the run's status gives it) and whether it is stopping or waits to be tried again, the batch
        the latest training of this launch kept in its simulation set when it was stopped or failed, and whether
        the narrowing round has an observation to choose, and say so through ``states_changed``. No panel's gate is
        run: ``refresh_gates`` runs them first."""
        held = controller()
        status = held.status
        place = status.place if status.live else None
        live = place[1] if place is not None and place[0] == "inference" else None
        self.pending_shown = self.pending()
        self._states = chain.step_states(self.session, live=live, stopping=status.state == "stopping",
                                         ended=held.training_kept(), pending=self.pending_shown,
                                         observation_on_offer=self.tsnpe_panel.obs_picker.has_entries(),
                                         label=record_words.label_of, retrying=status.state == "retrying")
        self.states_changed.emit(self._states)

    def step_states(self) -> tuple:
        """The six steps' states as last worked out, in the steps' order."""
        return self._states
