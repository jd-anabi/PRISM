"""The six steps of parameter inference as rules with no widget: what each step holds or waits for, the step
in hand, and Home's one next step.

Pure and import-safe: each rule reads the session it is handed through ``getattr`` and never raises on what a
field holds (a test puts a bare ``object()`` on every field), and nothing here imports the toolkit or the
engine, so the rail, Home and the refusal tables can share it without a window.

The checks are not counted yet: until they are, Checks reads "Needs a prior" while no prior is in use, "Needs a
posterior" while no posterior is, and "To come" otherwise (a calibration needs a posterior and the prior it was
trained against), and it is never the step in hand, so after a training Home points to Inference.
"""
from dataclasses import dataclass
from typing import Callable

from . import keys

# the word each marker is read with
_WORDS = {"done": "Done", "now": "Now", "later": "To come", "warn": "Attention", "err": "Error", "half": "Unfinished"}

# what a step in hand is doing; Checks and the narrowing round are never in hand
_DOING = {"model": "Choose a model", "prior": "Choose a prior", "training": "Setting up", "infer": "Setting up"}

# the steps that can be in hand, in order: the narrowing round is optional, and Checks waits until a check
# that applies and is still open can be counted
_IN_HAND = ("model", "prior", "training", "infer")

# the built-in models, as the window names them; a model of your own keeps the name it was stored under
_MODEL_NAMES = {"NADROWSKI": "Nadrowski", "HOPF": "Hopf", "BP": "BP"}


@dataclass(frozen=True)
class StepState:
    """Where one step stands. ``detail`` is what Home's table says the step holds or waits for, whole;
    ``rail_words`` is the step's second line in the rail. A step that is done holds a record or a model, and its
    detail is that name; any other detail is words, which a sentence writes with a small first letter."""
    key: str
    number: int
    marker: str
    word: str
    detail: str
    rail_words: str
    can_start: bool
    done: bool

    @property
    def spoken(self) -> str:
        """The step's whole state in one sentence, as its tool tip and its accessible name read it: "Step 5,
        Inference. To come: needs a posterior", "Step 1, Model and options. Done: Nadrowski", "Step 6, Narrowing
        round. To come"."""
        said = f"Step {self.number}, {keys.step(self.key).name}. {self.word}"
        if not self.detail:
            return said
        detail = self.detail if self.done else self.detail[:1].lower() + self.detail[1:]
        return f"{said}: {detail}"


def model_label(name) -> str:
    """A model as the window names it: "Nadrowski", "Hopf" and "BP" for the built-in models, a model of your own
    as it was stored, and "unnamed" for anything that is not a name."""
    if isinstance(name, str) and name:
        return _MODEL_NAMES.get(name, name)
    return "unnamed"


def _narrowed(posterior) -> bool:
    """Whether the posterior in use came from a narrowing round: its network carries the region it was
    narrowed to."""
    return getattr(getattr(posterior, "posterior", None), "truncation", None) is not None


def _state(key: str, marker: str, detail: str, rail_words: str, can_start: bool) -> StepState:
    return StepState(key, keys.step_number(key), marker, _WORDS[marker], detail, rail_words, can_start,
                     marker == "done")


def step_states(session, *, live: str | None = None, stopping: bool = False, ended: int | None = None,
                pending: frozenset = frozenset(), checks=None, observation_on_offer: bool = False,
                label: Callable[[object], str] | None = None) -> tuple[StepState, ...]:
    """The six steps' states, in the steps' order, from what ``session`` holds.

    ``live`` is the key of the step whose run is live, and ``stopping`` whether that run is stopping;
    ``ended`` the batch the latest training of this launch kept when it was stopped or failed; ``pending``
    the keys of the steps whose choice was changed and not applied; ``observation_on_offer`` whether the
    narrowing round has an observation to choose. ``label`` names a record the session holds; with none, a
    record is named by its own ``name``, or "unnamed" when it has none. ``checks`` is accepted and not read
    while the checks are not counted.

    For each step the first rule that holds decides: a live run, a change not applied, what the step holds,
    what it needs first, a training that ended early (step 3), and last "Now" for the step in hand or "To
    come" for any other."""
    draft = getattr(session, "draft", None)
    prior = getattr(session, "inf_prior", None)
    posterior = getattr(session, "posterior", None)
    observation = getattr(session, "observation", None)
    narrowed = _narrowed(posterior)

    def named(record) -> str:
        if label is not None:
            return label(record)
        name = getattr(record, "name", None)
        return name if isinstance(name, str) and name else "unnamed"

    holds = {
        "model": None if draft is None else model_label(getattr(draft, "model", None)),
        "prior": None if prior is None else named(prior),
        "training": None if posterior is None else named(posterior) + (", narrowed" if narrowed else ""),
        "checks": None,
        "infer": None if observation is None else named(observation),
        "narrowing": named(posterior) if narrowed else None,
    }
    needs = {
        "model": None,
        "prior": None if draft is not None else "Needs a model",
        "training": None if prior is not None else "Needs a prior",
        "checks": "Needs a prior" if prior is None else "Needs a posterior" if posterior is None else None,
        "infer": None if posterior is not None else "Needs a posterior",
        "narrowing": ("Needs a posterior" if posterior is None else "Needs a prior" if prior is None
                      else None if observation_on_offer else "Needs an observation"),
    }

    states, waiting = [], set()
    for place in keys.STEPS:
        key, can_start = place.key, needs[place.key] is None
        if key == live:
            words = "Stopping" if stopping else "Running"
            states.append(_state(key, "now", words, f"Now: {words.lower()}", can_start))
        elif key in pending:
            states.append(_state(key, "warn", "Changed, not applied", "Changed, not applied", can_start))
        elif holds[key] is not None:
            states.append(_state(key, "done", holds[key], f"Done: {holds[key]}", can_start))
        elif not can_start:
            states.append(_state(key, "later", needs[key], needs[key], False))
        elif key == "training" and ended is not None:
            batch = format(ended, ",")
            states.append(_state(key, "half", f"Simulations saved to batch {batch}", f"Unfinished: batch {batch}",
                                 True))
        else:
            states.append(_state(key, "later", "", "To come", True))
            waiting.add(key)
    in_hand = step_in_hand(states)
    if in_hand in waiting:
        doing = _DOING[in_hand]
        states[keys.step_number(in_hand) - 1] = _state(in_hand, "now", doing, f"Now: {doing.lower()}", True)
    return tuple(states)


def step_in_hand(states) -> str | None:
    """The key of the first step that is neither done nor unable to start, among the steps that can be in hand:
    never Checks while the checks are not counted, never the narrowing round, which is optional. None when no
    such step is open."""
    for state in states:
        if state.key in _IN_HAND and not state.done and state.can_start:
            return state.key
    return None


@dataclass(frozen=True)
class Next:
    """Home's one next step: its heading, the sentence under it (none when the heading says it all), its main
    button's words, and the part and page the button opens."""
    heading: str
    sentence: str
    button: str
    part: str
    page: str | None


# the next step while a step is in hand, by the step
_NEXT = {
    "model": Next("Next: choose a model", "Nothing is loaded: PRISM starts every launch with nothing loaded.",
                  "Open Model and options", "inference", "model"),
    "prior": Next("Next: build or load a prior", "", "Open Prior", "inference", "prior"),
    "training": Next("Next: start a training", "The Training page says what it will cost before it starts.",
                     "Set up training", "inference", "training"),
    "infer": Next("Next: run the posterior on a cell or on your recordings", "", "Open Inference", "inference",
                  "infer"),
}


def next_step(session, states, *, live_place: tuple[str, str | None] | None = None) -> Next:
    """The one next step Home offers. The first that holds decides: a run is live (``live_place`` is the part
    and page it runs in); then the step in hand among ``states``, which reads the steps in their order (no
    model, no prior, no posterior, no observation); otherwise nothing is waiting, and the narrowing round is
    offered, or Records once the posterior in use is a narrowed one."""
    if live_place is not None:
        part, page = live_place
        where = keys.place_words(f"{part}/{page}" if page else part)
        return Next("Next: nothing until the run ends",
                    f"A run is going in {where}, and no other run can start until it ends. You can still move "
                    "around and read results. Keep PRISM open; you can minimise it.", "Go to the run", part, page)
    in_hand = step_in_hand(states)
    if in_hand is not None:
        return _NEXT[in_hand]
    if _narrowed(getattr(session, "posterior", None)):
        return Next("Next: nothing is waiting", "A narrowing round was made: the posterior in use is the narrowed "
                    "one.", "Open Records", "records", None)
    return Next("Next: nothing is waiting", "A narrowing round is optional: it trains a second posterior that is "
                "valid only near one observation.", "Open Narrowing round", "inference", "narrowing")


def first_time(*, has_records: bool, has_runs: bool, session) -> bool:
    """Whether Home opens on its first-time page: no record of any kind in the store, no run in the history of
    the records folder in use, and nothing in the session (no model chosen)."""
    return not has_records and not has_runs and getattr(session, "draft", None) is None
