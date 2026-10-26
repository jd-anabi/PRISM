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

from .. import chain, keys, record_words
from ..panels.inference_tabs import (ConfigPanel, InferPanel, PosteriorPanel, PriorPanel,
                                     TSNPEPanel, ValidatePanel)
from ..run.controller import controller
from ..session import SbiSession
from ..widgets.adaptive_stack import AdaptiveStack


def _held_ref(wrapper) -> str:
    """How ``session_contents`` names one held artifact: its name in quotes, or ``(unnamed, id
    <id>)`` -- the wording the Prior, Training and Inference steps' own "ready" lines already use, so the
    log and the line agree.

    ``(unidentified)`` for a stand-in carrying neither. That branch is not defensive padding: the
    gate tests put a bare ``object()`` on every session field, and everything this function feeds is
    a derived status line, which may never raise.
    """
    name = getattr(wrapper, "name", "") or ""
    if name:
        return f"'{name}'"
    ref = getattr(wrapper, "id", "") or ""
    return f"(unnamed, id {ref})" if ref else "(unidentified)"


class InferenceScreen(QWidget):
    """Parameter inference: its six steps as pages over one shared SbiSession.

    Drives the workflow from a model to a narrowing round and owns its gating: ``refresh_gates`` runs every
    step's own gate after every run of a step, then the steps' states. Persists nothing itself -- each panel saves its
    own settings group. ``states_changed`` carries the six steps' states each time they are worked out again.
    ``pages`` holds the six steps in their order, its height following the step shown; ``show_step`` shows one
    (``show_page`` is the same) and ``current_step`` (or ``current_page``) names the one shown.
    """
    states_changed = Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.session = SbiSession()
        self._states = ()

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
        """Plain phrases for what the session holds -- the config, the prior, the posterior and the
        observation -- in pipeline order; empty when a ``new_draft`` would lose nothing.

        THE DRAFT IS DELIBERATELY NOT IN THE LIST. ``new_draft`` replaces the draft, and replacing it
        is the entire point of pressing Apply: a list that named it would make every deliberate model
        change look like a loss, and the guard built on this list would then interrupt every one of
        them. What Apply discards that nobody asked it to discard is the work DOWNSTREAM of
        the draft -- the built config, the prior, the posterior and the recorded observation -- which
        is exactly ``SbiSession.reset_downstream``'s subject and exactly this list.

        NOTHING HERE MAY RAISE. Every field is read through ``getattr``: the gate tests put a bare
        ``object()`` on ``cfg``, ``inf_prior`` and ``posterior``, and this feeds a derived status line
        on the first step, which must never be able to take a step down through ``refresh_gates()``
        (the rule ``_TrainingBudgetMixin._sync_budget`` states for its own three lines).

        The phrases carry no code identifiers and name no control: the first step's line and the
        confirmation both quote them verbatim, and each front end says where to look for itself.
        """
        s = self.session
        held: list[str] = []
        if s.cfg is not None:
            model = getattr(s.cfg, "model", "") or "an unnamed model"
            mode = getattr(s.cfg, "observation_mode", "") or ""
            held.append(f"the built config for {model}"
                        + (f", {mode} observations" if mode else ""))
        if s.inf_prior is not None:
            held.append(f"the prior {_held_ref(s.inf_prior)}")
        if s.posterior is not None:
            held.append(f"the posterior {_held_ref(s.posterior)}")
        if s.observation is not None:
            held.append(f"the observation {_held_ref(s.observation)}")
        return held

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
        """
        self.session = SbiSession(draft=draft)
        self.prior_panel.on_draft_set(draft)
        self.refresh_gates()

    def install_config(self, cfg):
        """Prior stage: bounds chosen, so the SimConfig now exists. Set it IN PLACE and fan out to
        the steps whose pickers/fields depend on it.

        THE IN-PLACE ONE OF THE TWO ENTRY POINTS, and in place BECAUSE OF WHEN IT IS CALLED: the
        Prior stage installs the config as the first step of building the prior, so a new session
        here would wipe the draft that built this config and the artifact the stage is about to
        store -- mid-stage, with nothing to confirm against, because the operator pressed a button
        that promised to build a prior and not to start over. That asymmetry is the difference
        between replacing the session's configuration and installing into it: its destructive twin
        ``new_draft`` is the one the first step confirms, and this one is never confirmed at all.
        """
        self.session.cfg = cfg
        self.infer_panel.on_config_built(cfg)
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
        (its place, as the run's status gives it) and whether it is stopping, and whether the narrowing round has
        an observation to choose, and say so through ``states_changed``. No panel's gate is run:
        ``refresh_gates`` runs them first."""
        status = controller().status
        place = status.place if status.live else None
        live = place[1] if place is not None and place[0] == "inference" else None
        self._states = chain.step_states(self.session, live=live, stopping=status.state == "stopping",
                                         observation_on_offer=self.tsnpe_panel.obs_picker.has_entries(),
                                         label=record_words.label_of)
        self.states_changed.emit(self._states)

    def step_states(self) -> tuple:
        """The six steps' states as last worked out, in the steps' order."""
        return self._states
