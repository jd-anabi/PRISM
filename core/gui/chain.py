"""The six steps of parameter inference as rules with no widget: what each step holds or waits for, the step
in hand, and Home's one next step.

Pure and import-safe: each rule reads the session it is handed through ``getattr`` and never raises on what a
field holds (a test puts a bare ``object()`` on every field), and nothing here imports the toolkit or the
engine, so the rail, Home and the refusal tables can share it without a window.

Checks reads the checks' summary the screen hands once it has read the checks' records: how many deciding checks
passed, of how many, and whether one failed, needs a closer look or is still open. Handed none, Checks reads as before
the checks were counted: "Needs a prior" while no prior is in use, "Needs a posterior" while no posterior is, and "To
come" otherwise, and it is never the step in hand.
"""
from dataclasses import dataclass, replace
from typing import Callable

from . import keys

# the word each marker is read with
_WORDS = {"done": "Done", "now": "Now", "later": "To come", "warn": "Attention", "err": "Error", "half": "Unfinished"}

# what a step in hand is doing; Checks says its count instead, and the narrowing round is never in hand
_DOING = {"model": "Choose a model", "prior": "Choose a prior", "training": "Setting up", "infer": "Setting up"}

# the steps that can be in hand, in order: the narrowing round is optional, and Checks is in hand only while its checks
# are counted and a deciding check that applies is still open
_IN_HAND = ("model", "prior", "training", "checks", "infer")

# the built-in models, as the window names them; a model of your own keeps the name it was stored under
_MODEL_NAMES = {"NADROWSKI": "Nadrowski", "HOPF": "Hopf", "BP": "BP"}

# the kind of observation applied, as Home's sentence of what was in use last time says it, and its last words
_FREQUENCY_RESPONSE = "the frequency response as the kind of observation"
_SET_BY_THE_FILE = "the kind of observation set by the bounds file"
_NOTHING_UNTIL_PRESSED = "Nothing is loaded until you press the button."


@dataclass(frozen=True)
class StepState:
    """Where one step stands. ``detail`` is what Home's table says the step holds or waits for, whole;
    ``rail_words`` is the step's second line in the rail. A step that is done holds a record or a model, and its
    detail is that name; any other detail is words, which a sentence writes with a small first letter. ``pending``
    says the step holds something and a choice on its page differs from it. ``may_be_in_hand`` says the step can be
    the step in hand at all: Checks only while its checks are counted."""
    key: str
    number: int
    marker: str
    word: str
    detail: str
    rail_words: str
    can_start: bool
    done: bool
    pending: bool = False
    may_be_in_hand: bool = True

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


def _state(key: str, marker: str, detail: str, rail_words: str, can_start: bool, pending: bool = False,
           may_be_in_hand: bool = True) -> StepState:
    return StepState(key, keys.step_number(key), marker, _WORDS[marker], detail, rail_words, can_start,
                     marker == "done", pending, may_be_in_hand)


def _counted(checks) -> "tuple[str | None, str, str, str, bool]":
    """What Checks says of the checks' summary ``checks``, read by its attributes: its marker, None while a deciding
    check that applies is open and none failed or needs a closer look; its words in the rail; its detail, the headline
    with the summary's second line in brackets after it; its count, "2 of 6 passed"; and whether it may be the step in
    hand. With none open, failed or to look closer at, a deciding check no run can pass for the posterior in use reads
    "Attention" and leaves the step in hand to the next step: it is not passed, and no run of this step can pass it."""
    count = f"{getattr(checks, 'passed', 0)} of {getattr(checks, 'of', 0)} passed"
    headline, second = getattr(checks, "headline", ""), getattr(checks, "second_line", "")
    detail = f"{headline} ({second})" if headline and second else str(headline or "")
    cannot = getattr(checks, "cannot_pass", 0)
    if getattr(checks, "failed", False) is True:
        return "err", f"Error: {count}", detail, count, True
    if getattr(checks, "look_closer", False) is True:
        return "warn", f"Attention: {count}", detail, count, True
    if getattr(checks, "open", True) is False and isinstance(cannot, int) and not isinstance(cannot, bool) and cannot:
        return "warn", f"Attention: {count}", detail, count, False
    if getattr(checks, "open", True) is False:
        return "done", f"Done: {count}", detail, count, True
    return None, f"To come: {count}", detail, count, True


def step_states(session, *, live: str | None = None, stopping: bool = False, pending: frozenset = frozenset(),
                checks=None, observation_on_offer: bool = False, label: Callable[[object], str] | None = None,
                retrying: bool = False, kept=None, kept_round=None) -> tuple[StepState, ...]:
    """The six steps' states, in the steps' order, from what ``session`` holds.

    ``live`` is the key of the step whose run is live, ``stopping`` whether that run is stopping, and ``retrying``
    whether it waits to be tried again; ``pending`` the keys of the steps whose choice was changed and not applied;
    ``observation_on_offer`` whether the narrowing round has an observation to choose. ``label`` names a record the
    session holds; with none, a record is named by its own ``name``, or "unnamed" when it has none. ``checks`` is the
    checks' summary (read by its attributes, as ``core.gui.check_lineage.ChecksSummary`` holds them), None while the
    checks have not been read: with it Checks needs only a prior, and reads "Error", "Attention" or "Done" with its
    count when a deciding check failed, needs a closer look, or every one that applies passed, else its count, in hand
    while one is open; with none open and one that no run can pass for the posterior in use, "Attention" with its count,
    never in hand; with none it reads as before the checks were counted. ``kept`` is what the run history and the kept
    trainings offer step 3 (read by its attributes, as ``core.gui.continuing.Offer`` holds them; None for nothing): a
    kept training to resume, after the epoch it holds, or a training to continue, or stopped, with the batches its
    simulation set holds; anything else it offers changes no step. ``kept_round`` is what they offer step 6, the same
    way: a kept narrowing round to resume, after the epoch it holds, or a narrowing round to continue, with the batches
    its set holds.

    For each step the first rule that holds decides: a live run, a change not applied, what the step holds, for steps
    3 and 6 what is offered to them, which needs nothing first since a kept training names its own records and a
    continuation loads them, what the step needs first, and last "Now" for the step in hand or "To come" for any
    other."""
    draft = getattr(session, "draft", None)
    prior = getattr(session, "inf_prior", None)
    posterior = getattr(session, "posterior", None)
    observation = getattr(session, "observation", None)
    narrowed = _narrowed(posterior)
    counted = checks is not None

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
        "checks": ((None if getattr(checks, "can_check", False) is True else "Needs a prior") if counted
                   else "Needs a prior" if prior is None else "Needs a posterior" if posterior is None else None),
        "infer": None if posterior is not None else "Needs a posterior",
        "narrowing": ("Needs a posterior" if posterior is None else "Needs a prior" if prior is None
                      else None if observation_on_offer else "Needs an observation"),
    }

    states, waiting = [], {}
    for place in keys.STEPS:
        key, can_start = place.key, needs[place.key] is None
        may = key != "checks" or counted              # Checks is in hand only while its checks are counted
        if key == live:
            words = "Stopping" if stopping else "Waiting to try again" if retrying else "Running"
            states.append(_state(key, "now", words, f"Now: {words.lower()}", can_start, may_be_in_hand=may))
        elif key in pending:
            states.append(_state(key, "warn", "Changed, not applied", "Changed, not applied", can_start,
                                 holds[key] is not None, may))
        elif holds[key] is not None:
            states.append(_state(key, "done", holds[key], f"Done: {holds[key]}", can_start))
        elif key == "training" and _unfinished(kept) is not None:
            states.append(_state(key, "half", *_unfinished(kept), True))
        elif key == "narrowing" and _round_unfinished(kept_round) is not None:
            states.append(_state(key, "half", *_round_unfinished(kept_round), True))
        elif not can_start:
            states.append(_state(key, "later", needs[key], needs[key], False, may_be_in_hand=may))
        elif key == "checks" and counted:
            marker, words, detail, count, may = _counted(checks)
            states.append(_state(key, marker or "later", detail, words, True, may_be_in_hand=may))
            if marker is None:
                waiting[key] = (detail, f"Now: {count}")
        else:
            states.append(_state(key, "later", "", "To come", True, may_be_in_hand=may))
            doing = _DOING.get(key, "")
            waiting[key] = (doing, f"Now: {doing.lower()}")
    in_hand = step_in_hand(states)
    if in_hand in waiting:
        detail, words = waiting[in_hand]
        states[keys.step_number(in_hand) - 1] = _state(in_hand, "now", detail, words, True)
    return tuple(states)


def _unfinished(kept) -> "tuple[str, str] | None":
    """What step 3 says of what is offered to it, its detail and its words in the rail; None for nothing it says, and
    for anything offered to step 6."""
    if getattr(kept, "step", "training") != "training":
        return None
    kind = getattr(kept, "kind", None)
    if kind == "resume" and isinstance(getattr(kept, "epoch", None), int):
        epoch = format(kept.epoch, ",")
        return f"Training kept after epoch {epoch}", f"Unfinished: epoch {epoch}"
    if kind in ("continue", "stopped") and isinstance(getattr(kept, "batch", None), int):
        batch = format(kept.batch, ",")
        return f"Simulations saved to batch {batch}", f"Unfinished: batch {batch}"
    return None


def _round_unfinished(kept) -> "tuple[str, str] | None":
    """What step 6 says of what is offered to it, its detail and its words in the rail; None for nothing it says."""
    if getattr(kept, "step", None) != "narrowing":
        return None
    kind = getattr(kept, "kind", None)
    if kind == "round" and isinstance(getattr(kept, "epoch", None), int):
        epoch = format(kept.epoch, ",")
        return f"Narrowing round kept after epoch {epoch}", f"Unfinished: epoch {epoch}"
    if kind == "continue" and isinstance(getattr(kept, "batch", None), int):
        batch = format(kept.batch, ",")
        return f"Simulations saved to batch {batch}", f"Unfinished: batch {batch}"
    return None


def step_in_hand(states) -> str | None:
    """The key of the first step that is neither done nor unable to start, among the steps that can be in hand:
    never Checks while its checks are not counted, never the narrowing round, which is optional, and never a step
    that holds something with a choice changed and not applied, since what it holds is what the next steps read.
    None when no such step is open."""
    for state in states:
        if (state.key in _IN_HAND and getattr(state, "may_be_in_hand", True) and not state.done
                and not state.pending and state.can_start):
            return state.key
    return None


@dataclass(frozen=True)
class Next:
    """Home's one next step: its heading, the sentence under it (none when the heading says it all), its main
    button's words, and the part and page the button opens; for a run whose end has not been seen, the run, whose
    page the button opens and whose end it acknowledges; and for a button that does something in place of opening a
    place, what it does (``"continue_with_these"``: apply and load what was in use last time;
    ``"continue_run"``: fill the forms from a training's history and continue it, or fit it again; ``"resume"``: resume
    a kept training; ``"open_round"``: open the Narrowing round page on a kept round's row, resuming nothing;
    ``"continue_round"``: fill the forms from a narrowing round's history and continue it, or fit it again) and the
    offer it takes (``offer``); and ``link``, the words of a link beside the button that opens the run's page, for a
    run whose end has not been seen and whose offer the button takes."""
    heading: str
    sentence: str
    button: str
    part: str
    page: str | None
    run: str | None = None
    action: str | None = None
    link: str | None = None
    offer: object = None


# the next step while a step is in hand, by the step
_NEXT = {
    "model": Next("Next: choose a model", "Nothing is loaded: PRISM starts every launch with nothing loaded.",
                  "Open Model and options", "inference", "model"),
    "prior": Next("Next: build or load a prior", "", "Open Prior", "inference", "prior"),
    "training": Next("Next: start a training", "The Training page says what it will cost before it starts.",
                     "Set up training", "inference", "training"),
    "checks": Next("Next: check the posterior", "", "Open Checks", "inference", "checks"),
    "infer": Next("Next: run the posterior on a cell or on your recordings", "", "Open Inference", "inference",
                  "infer"),
}


def next_step(session, states, *, run=None, picks=None, offers=(), checks=None) -> Next:
    """The one next step Home offers. ``run`` is the run's status (read by its attributes: its state, word and name,
    place, the reason it needs someone and how it ended, in words), None for none; ``picks`` what was in use last time
    (read by its attributes, as ``last_time_sentence`` reads them), None for nothing; ``offers`` what the run history
    and the kept trainings offer, newest first (each read by its attributes, as ``core.gui.continuing.Offer`` holds
    it); ``checks`` the checks' summary, whose headline is the sentence while Checks is the step in hand (read by its
    attributes; None for none). The first that holds decides, for a run with a place: a live run needs someone (its
    reason, and the button goes to it); a run is live (nothing until it ends: it is running, stopping or waiting to be
    tried again); a run has ended and nobody has seen how (how and when it ended and what it kept, and the button opens
    its page and acknowledges it: "See the result" for a finished run, "Go to the run" for a stopped one, "See what happened" for a
    failure or a run not seen to end; but for a failure or a run not seen to end that something is offered for, the
    button takes the offer and "See what happened" stands beside it). Then the newest offer that can be taken, a
    run stopped with its set unfinished passed over: continue a training or a narrowing round, fit it again, resume a
    kept training, or for a kept narrowing round the page that lists it. Then, with no model chosen and a model in use
    last time, what was in use last time, which the button applies and loads, staying on Home. Then the step in hand
    among ``states``, which reads the steps in their order (no model, no prior, no posterior, a deciding check that
    applies still open, no observation); otherwise nothing is waiting, and the narrowing round is offered, or Records
    once the posterior in use is a narrowed one."""
    place = getattr(run, "place", None)
    if place is not None:
        part, page = place
        what, who = run.word.lower(), f"{run.word} {run.name}".strip()
        if run.live and run.needs_you:
            return Next(f"Next: look at the {what}", run.needs_you, "See why", part, page)
        if run.live:
            doing = ("is stopping" if run.state == "stopping" else "is waiting to try again"
                     if run.state == "retrying" else "is running")
            return Next(f"Next: nothing until the {what} ends",
                        f"{who} {doing}, and no other run can start until it ends. You can still move around and read "
                        "results. Keep PRISM open; you can minimise it.", "Go to the run", part, page)
        if run.unacknowledged_end:
            if run.outcome == "finished":
                return Next(f"Next: see the {what}", run.sentence, "See the result", part, page, run.run)
            own = next((offer for offer in offers if getattr(offer, "run", None) == run.run and _takes(offer)), None)
            if own is not None and run.outcome in ("failed", "not seen to end"):
                return Next(f"Next: see what happened to the {what}", run.sentence, own.button, part, page, run.run,
                            action=action_of(own), link=SEE_WHAT_HAPPENED, offer=own)
            return Next(f"Next: see what happened to the {what}", run.sentence,
                        "Go to the run" if run.outcome == "stopped" else SEE_WHAT_HAPPENED, part, page, run.run)
    offer = next((offer for offer in offers if _takes(offer)), None)
    if offer is not None:
        return _offered(offer)
    if getattr(session, "draft", None) is None and isinstance(getattr(picks, "model", None), str) and picks.model:
        return Next("Next: continue with what was in use last time", last_time_sentence(picks), "Continue with these",
                    "home", None, action="continue_with_these")
    in_hand = step_in_hand(states)
    if in_hand == "checks":
        headline = getattr(checks, "headline", "")
        return replace(_NEXT["checks"], sentence=headline if isinstance(headline, str) else "")
    if in_hand is not None:
        return _NEXT[in_hand]
    if _narrowed(getattr(session, "posterior", None)):
        return Next("Next: nothing is waiting", "A narrowing round was made: the posterior in use is the narrowed "
                    "one.", "Open Records", "records", None)
    return Next("Next: nothing is waiting", "A narrowing round is optional: it trains a second posterior that is "
                "valid only near one observation.", "Open Narrowing round", "inference", "narrowing")


#: The words of the link beside the button that takes the offer of a run whose end has not been seen.
SEE_WHAT_HAPPENED = "See what happened"
#: What the button does for each offer it takes, by the offer's kind and the step whose page takes it.
_ACTIONS = {("continue", "training"): "continue_run", ("fit_again", "training"): "continue_run",
            ("resume", "training"): "resume", ("round", "narrowing"): "open_round",
            ("continue", "narrowing"): "continue_round", ("fit_again", "narrowing"): "continue_round"}
#: Home's button for a kept narrowing round: it opens the round's page on its row.
OPEN_ROUND_WORDS = "Open Narrowing round"


def action_of(offer) -> "str | None":
    """What a button that takes ``offer`` does, read from its kind and its step (read by their attributes); None for an
    offer no button takes."""
    return _ACTIONS.get((getattr(offer, "kind", None), getattr(offer, "step", "training")))


def _takes(offer) -> bool:
    """Whether Home's button can take ``offer``: one to continue, fit again, resume or open that nothing stops."""
    return action_of(offer) is not None and not getattr(offer, "why_not", "")


def _offered(offer) -> Next:
    """Home's next step for the offer ``offer``."""
    action = action_of(offer)
    if offer.kind == "round":
        heading = (f"Next: resume the narrowing round {offer.label}" if getattr(offer, "named", True)
                   else "Next: resume the narrowing round")
        return Next(heading, offer.sentence, OPEN_ROUND_WORDS, "inference", "narrowing", action=action, offer=offer)
    if offer.step == "narrowing":
        return Next("Next: continue the narrowing round", offer.sentence, offer.button, "inference", "narrowing",
                    action=action, offer=offer)
    label = offer.label
    if offer.kind != "resume":
        heading = "Next: continue the training"
    elif getattr(offer, "named", True):
        heading = f"Next: resume the training {label}".rstrip()
    else:                                        # Records' label of an unnamed one, read as part of the sentence
        heading = f"Next: resume the {label[:1].lower()}{label[1:]}"
    return Next(heading, offer.sentence, offer.button, "inference", "training", action=action, offer=offer)


def last_time_sentence(picks) -> str:
    """What was in use last time, in one sentence, by the names the window shows, exactly what one press applies and
    loads: the model, the kind of observation the press applies with it, the bounds file, the prior, the posterior, the
    observation and the plan, each left out when ``picks`` holds none (read by its attributes: ``model``, ``chi_mode``,
    ``bounds``, ``prior`` with ``prior_words``, ``posterior`` with ``posterior_words``, ``observation`` with
    ``observation_words``, ``plan``). A kind that could not be read is
    applied as the bounds file's, and named so."""
    said = [f"the model {model_label(picks.model)}",
            _FREQUENCY_RESPONSE if getattr(picks, "chi_mode", None) is True else _SET_BY_THE_FILE]
    bounds = getattr(picks, "bounds", None)
    if bounds:
        said.append("the bounds file " + str(bounds).replace("\\", "/"))
    for record, words in (("prior", "prior_words"), ("posterior", "posterior_words"),
                          ("observation", "observation_words")):
        if getattr(picks, record, None) and getattr(picks, words, ""):
            said.append(getattr(picks, words))
    if getattr(picks, "plan", None):
        said.append(f"the plan {picks.plan}")
    return f"Last time: {', '.join(said)}. {_NOTHING_UNTIL_PRESSED}"


def first_time(*, has_records: bool, has_runs: bool, session) -> bool:
    """Whether Home opens on its first-time page: no record of any kind in the store, no run in the history of
    the records folder in use, and nothing in the session (no model chosen)."""
    return not has_records and not has_runs and getattr(session, "draft", None) is None
