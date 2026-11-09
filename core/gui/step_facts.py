"""What a step says before it starts, as values: a fact and the order facts stand in, the reason beside the main
button, the line that says when the saved records were read, where a field's value came from, and how long something
takes, in a person's words; a row that picks what a step holds; what the first step holds and says, and what the
second does.

A fact is a marker with its lead word, one or two plain sentences, and, for what stops the start, the few words that go
beside the button. Every "Error." and every "Blocked." blocks and nothing else does, and a fact that says otherwise is
refused as it is made, so that a marker never disagrees with the button.

Nothing here imports the toolkit or the engine: a step's facts are worked out and tested as values, with no widget.
"""
import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime

from .chain import model_label

#: The lead word of each marker, as a fact is shown.
LEADS = {"err": "Error.", "lock": "Blocked.", "warn": "Attention.", "info": "Information.", "fine": "Fine."}
#: The markers that stop the start.
BLOCKING = ("err", "lock")
#: Said among a page's own lines when it asks for a run while one is live, and beside a main button greyed by one.
LIVE_TEXT = "A run is live. Only one runs at a time."
#: What a page says of what it has not read yet, having been first opened while a run is live.
NOT_READ_YET_TEXT = "A run is live. Not read yet; read when it ends."
READ_TEXT = "The saved records were read at {}."
READ_LIVE_TEXT = "A run is live. Read at {}; read again when it ends."
NOTHING_TO_APPLY_TEXT = "Nothing to apply: these choices are in use."
#: The words beside the button of a start the engine refused.
REFUSED_BLOCKS = "the last start was refused"
ONE_BLOCKS_TEXT = "Cannot start yet: {}."
SEVERAL_BLOCK_TEXT = "Cannot start yet: {}, and {} more above."
#: The same, for a block that neither waiting nor an edit on the page lifts: the thing chosen itself cannot be used.
ONE_STOPS_TEXT = "Cannot start: {}."
SEVERAL_STOP_TEXT = "Cannot start: {}, and {} more above."


@dataclass(frozen=True)
class Fact:
    """One line of "Before you start": ``kind``, one of ``LEADS``; ``sentence``; and ``blocks``, the few words that go
    beside the button, which an "err" or a "lock" fact holds and no other does. ValueError for one that breaks this."""
    kind: str
    sentence: str
    blocks: str = ""

    def __post_init__(self):
        if self.kind not in LEADS:
            raise ValueError(f"not a marker of a fact: {self.kind!r}")
        if (self.kind in BLOCKING) != bool(self.blocks.strip()):
            raise ValueError(f"a {self.kind!r} fact {'must hold' if self.kind in BLOCKING else 'holds no'} words beside "
                             f"the button: {self.blocks!r}")


def refused_facts(sentences: Iterable[str], blocks: str = "") -> tuple[Fact, ...]:
    """What a refusal says among the facts, each of ``sentences`` the engine's own: an "Error." that stops the start, as
    the mark under its field does, for a marker never disagrees with the button; beside the button ``blocks``, or, for a
    refused start, that it was."""
    return tuple(Fact("err", sentence, blocks or REFUSED_BLOCKS) for sentence in sentences)


def ordered(facts: Iterable[Fact]) -> tuple[Fact, ...]:
    """The facts as they stand: those that block first, then the others, each group in the order handed in."""
    facts = tuple(facts)
    return tuple(fact for fact in facts if fact.blocks) + tuple(fact for fact in facts if not fact.blocks)


def reason_beside(facts: Sequence[Fact], *, live: bool, nothing_to_apply: bool) -> str:
    """The words beside the main button: a live run first, since nothing else can be done about that; then what blocks,
    the first by its words and the others counted; then that there is nothing to apply; else none."""
    if live:
        return LIVE_TEXT
    blocking = [fact.blocks for fact in ordered(facts) if fact.blocks]
    if not blocking:
        return NOTHING_TO_APPLY_TEXT if nothing_to_apply else ""
    final = blocking[0] in FINAL_BLOCKS          # the lead is the first block's, the one its words name
    if len(blocking) == 1:
        return (ONE_STOPS_TEXT if final else ONE_BLOCKS_TEXT).format(blocking[0])
    return (SEVERAL_STOP_TEXT if final else SEVERAL_BLOCK_TEXT).format(blocking[0], len(blocking) - 1)


def reading_words(read_at: datetime | None, *, live: bool) -> str:
    """The small line at the right of "Before you start": when the saved records were read, ``read_at`` (None for not
    yet), and that they are read again when a live run ends. A page not read and with no run live says nothing: it reads
    as it is shown."""
    if read_at is None:
        return NOT_READ_YET_TEXT if live else ""
    return (READ_LIVE_TEXT if live else READ_TEXT).format(f"{read_at.hour:02d}:{read_at.minute:02d}")


def origin_words(*, held: bool, at_source: bool, at_usual: bool, kept: bool, source: str = "plan",
                 name: str = "") -> str:
    """Where a field's value came from. ``held``: the source in use, a "plan", a "record" or a "run" called ``name``,
    holds the setting, and ``at_source`` says whether the form still has its value; that wins. Else ``kept``: put back
    from the settings file and not touched since. Else ``at_usual``: the form has the built-in value."""
    if held:
        return f"from {source} {name}".rstrip() if at_source else f"changed from the {source}"
    if kept:
        return "kept from last time"
    return "usual" if at_usual else "changed from the usual"


def _rounded(value: float) -> int:
    """``value`` to the nearest whole number, a half up: Python's own ``round`` takes a half to the even neighbour."""
    return math.floor(value + 0.5)


def _in_halves(halves: int, unit: str, one: str) -> str:
    """A length counted in halves of ``unit``, whose single is said ``one``: "a minute and a half", "2 hours"."""
    whole, half = divmod(halves, 2)
    if whole == 0:
        return f"half {one}"
    if whole == 1:
        return f"{one} and a half" if half else one
    return f"{whole} and a half {unit}s" if half else f"{whole} {unit}s"


def about_words(seconds: float) -> str:
    """How long something takes as a person says it: to the half minute below ten minutes, to the minute below an hour,
    to the half hour above. "" for a length that is no length, since a clock may be set back between a run's two
    ends and a history file is only as good as the file."""
    if not math.isfinite(seconds) or seconds <= 0:
        return ""
    if seconds < 600:
        halves = max(1, _rounded(seconds / 30))
        return "about " + _in_halves(halves, "minute", "a minute")
    minutes = _rounded(seconds / 60)
    if minutes < 60:
        return f"about {minutes} minutes"
    return "about " + _in_halves(_rounded(seconds / 1800), "hour", "an hour")


class _NotHeld:
    """What a selection row's step holds of it while it holds nothing that says the row's value."""

    def __repr__(self) -> str:
        return "NOT_HELD"


#: What a selection row's ``in_use`` answers while the step holds nothing it could compare the row with.
NOT_HELD = _NotHeld()
CHANGED_TEXT = "Changed, not applied. In use: {}."


def held_value(holder, name: str) -> object:
    """``name`` of what a step holds, read as a session's fields are read; ``NOT_HELD`` while it holds nothing, or
    holds something that has no such value."""
    return NOT_HELD if holder is None else getattr(holder, name, NOT_HELD)


@dataclass(frozen=True)
class SelectionRow:
    """A row whose value picks what its step holds. ``key`` is its field key; ``in_use`` answers what the step holds of
    it, or ``NOT_HELD``; ``chosen``, what the row shows now; ``words``, a value held as the line under the row names it.
    A row reads changed only against something held, so it never does while nothing is in use."""
    key: str
    in_use: Callable[[], object]
    chosen: Callable[[], object]
    words: Callable[[object], str] = str

    def changed(self) -> bool:
        """Whether the row shows something other than what is in use. Never raises: a value no comparison can be made
        with is no change a person made."""
        held = self.in_use()
        if held is NOT_HELD:
            return False
        try:
            return bool(self.chosen() != held)
        except Exception:                    # noqa: BLE001 -- what a session holds may be anything
            return False

    def line(self) -> str:
        """What the line under the row says while it reads changed: "Changed, not applied. In use: Nadrowski."."""
        return CHANGED_TEXT.format(self.words(self.in_use()))


def units_words(tokens: Iterable[str]) -> str:
    """Units as a sentence names them: in the order of the alphabet, whatever order a file or a person gave them."""
    return " ".join(sorted((str(token) for token in tokens), key=lambda token: (token.casefold(), token)))


#: The words beside the main button of the first step while the model chosen cannot be used.
UNUSABLE_BLOCKS = "this model cannot be used"


def model_in_use(draft, units_said: str) -> tuple[str, str]:
    """What the first step holds, as "In use" says it: the model's name, and one sentence of the kind of observation,
    the rotation and ``units_said``, the units and where they come from; ``("", "")`` while it holds nothing. Read
    from the draft held, as a session's fields are read: what it does not say is left out."""
    if draft is None:
        return "", ""
    parts = []
    numbers = [getattr(draft, name, None) for name in ("chi_n_freqs", "chi_k_pad", "chi_max_cycles")]
    if getattr(draft, "chi_mode", None) is True:
        if all(isinstance(number, (int, float)) and not isinstance(number, bool) for number in numbers):
            parts.append(f"Frequency response, {numbers[0]} driven recordings in {numbers[1]} slots, lock-in limit "
                         f"{numbers[2]:g} drive cycles.")
        else:
            parts.append("Frequency response.")
    elif getattr(draft, "chi_mode", None) is False:
        parts.append("Set by the bounds file.")
    rotation = getattr(draft, "reparam_rotate", None)
    if isinstance(rotation, bool):
        parts.append("Rotation on." if rotation else "Rotation off.")
    if units_said:
        parts.append(f"Units {units_said}.")
    return model_label(getattr(draft, "model", None)), " ".join(parts)


#: What step 1's "In use" adds after a Resume, while the session runs as the resumed training was kept.
RESUMED_TEXT = "Runs on {} as the resumed training was kept{}, until step 1 or step 2 is applied again."
RESUMED_WIDTH = ", {:,} simulations at a time"
DEVICE_WORDS = {"cpu": "the processor", "cuda": "the graphics card"}


def resumed_words(device: "str | None", width: "int | None") -> str:
    """The sentence step 1's "In use" adds after a Resume: the device the session now uses, by the kind of device
    ("cpu", "cuda"), and the width, when it is the kept training's own; "" for a device of no kind known."""
    if device not in DEVICE_WORDS:
        return ""
    return RESUMED_TEXT.format(DEVICE_WORDS[device], RESUMED_WIDTH.format(width) if width else "")


def model_facts(*, usable_reason: str, field_facts: Sequence[Fact], plan_lines: Sequence[Fact] = ()) -> tuple[Fact, ...]:
    """The facts of the first step: that the model chosen cannot be used, when ``usable_reason`` says why; then those of
    its fields; then what the plan in use says. Those that block come first."""
    lead = (Fact("err", usable_reason, UNUSABLE_BLOCKS),) if usable_reason else ()
    return ordered((*lead, *field_facts, *plan_lines))


# ── the second step, Prior ──
#: The longest a prior's build is believed to have taken: two times of a run further apart than this, or a length
#: below nothing, come from a clock that was set wrong, and say nothing of how long a build takes.
LONGEST_BUILD_S = 7 * 24 * 3600
BUILD_WORDS = "Build a new prior"
LOAD_WORDS = "Load this prior"
NEEDS_A_MODEL = "Needs a model: apply one in step 1, Model and options."
NEEDS_A_MODEL_BLOCKS = "needs a model"
NO_BOUNDS_TEXT = "Choose a bounds file."
NO_BOUNDS_BLOCKS = "no bounds file is chosen"
BOUNDS_BLOCKS = "the bounds file cannot be read"
NO_PRIOR_TEXT = "Choose a saved prior."
NO_PRIOR_BLOCKS = "no saved prior is chosen"
NONE_SAVED_TEXT = "No prior is saved yet: build one with 'Build a new prior instead…'."
NONE_SAVED_BLOCKS = "no prior is saved yet"
PRIOR_BLOCKS = "the prior cannot be read"
OTHER_MODEL_TEXT = "This prior was made for the model {}."
OTHER_MODEL_BLOCKS = "this prior is for another model"
TYPED_BY_HAND = "limits typed by hand"
SAME_FILE_TEXT = "Built on {}, the bounds file selected."
OTHER_FILE_TEXT = "Built on {}; the bounds file selected is {}."
CHECK_PASSES_TEXT = ("PRISM's own check compares only the model's parameters and their limits, which these two files "
                     "share, so the load goes through.")
CHECK_REFUSES_TEXT = ("PRISM's own check compares only the model's parameters and their limits, which differ between "
                      "these two files, so PRISM will refuse the load.")
EDITED_TEXT = "Built on {} as it was then; the file has been edited since."
EDITED_REFUSES_TEXT = "Its parameters or their limits have changed with it, so PRISM will refuse the load."
TYPED_TEXT = "Built on limits typed by hand."
TYPED_REFUSES_TEXT = ("The bounds file selected differs from them in the model's parameters or their limits, so PRISM "
                      "will refuse the load.")
NO_SETS_TEXT = "No saved simulation set was made with this prior."
ONE_SET_TEXT = "1 saved simulation set was made with this prior; it holds {:,} batches."
SETS_TEXT = "{} saved simulation sets were made with this prior; the largest holds {:,} batches."
COST_TEXT = "Tests {:,} candidates ({:,} rounds of {:,}){}, then walks outward until {:,} stable sets are collected."
STEPS_TEXT = ", each integrated over {:,} steps"
NEVER_BUILT_TEXT = "No prior has been built on this computer yet, so the time is not known."
NO_LENGTH_TEXT = "The last build on this computer has no length that can be read, so the time is not known."
TOOK_TEXT = "{}: the last build on this computer took {}."
ONE_OTHER_TEXT = ("A new prior cannot continue a saved simulation set: the saved set for this model was made with another "
                  "prior.")
OTHERS_TEXT = ("A new prior cannot continue a saved simulation set: the {} saved sets for this model were made with other "
               "priors.")
#: How a posterior trained on a bounds file reads, by the kind of observation a configuration made from it has.
READS = {"chi": "a frequency response", "forced": "a spontaneous recording and one driven recording",
         "spontaneous": "a spontaneous recording only"}
PARAMETERS_TEXT = "{} parameters: {} of the model ({}) and {} {} ({})."
MODEL_ONLY_TEXT = "{} parameters, all of the model ({})."
FORCING_TEXT = "It has a forcing section."
NO_FORCING_TEXT = "It has no forcing section."
READS_TEXT = "With the kind of observation applied in step 1, a posterior trained on it reads: {}."
SAVED_PRIOR_TEXT = "Made {}, for {}: {} parameters, built on {}."
PRIOR_IN_USE_TEXT = "Built on {}, {} parameters. Saved as a record."
NOTHING_HELD_TEXT = "Choose a bounds file, then load a saved prior or build a new one."


def length_words(seconds: float) -> str:
    """A length a run took, to the second, as a clock says it: "45 s", "1 min 32 s", "12 min", "1 h 5 min"."""
    whole = max(0, math.floor(seconds + 0.5))
    hours, rest = divmod(whole, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours} h {minutes} min" if minutes else f"{hours} h"
    if minutes:
        return f"{minutes} min {secs} s" if secs else f"{minutes} min"
    return f"{secs} s"


def build_length(took) -> "float | None":
    """How long the last build took, in seconds, from ``took``, the run history's answer (its length and its name, or
    None for no build): None for none, and for a length a clock set wrong gave, below nothing or past the longest a build
    is believed to take."""
    try:
        seconds = float(took[0])
    except (TypeError, ValueError, IndexError, OverflowError):
        return None
    return seconds if math.isfinite(seconds) and 0 < seconds <= LONGEST_BUILD_S else None


def build_words(took) -> str:
    """The build's main button, named for its outcome, with how long the last build took when that can be read: "Build
    a new prior (about a minute and a half)"."""
    seconds = build_length(took)
    return f"{BUILD_WORDS} ({about_words(seconds)})" if seconds is not None else BUILD_WORDS


def bounds_words(params: Sequence[str], scales: Sequence[str], forcing: bool, mode: str) -> str:
    """What a bounds file is, under its list: its parameters, the model's and the scales, named in the file's order;
    whether it has a forcing section; and how a posterior trained on it reads, by ``mode``, the kind of observation of a
    configuration made from it. A file with no scale names none."""
    scale = "scale" if len(scales) == 1 else "scales"
    if scales:
        named = PARAMETERS_TEXT.format(len(params) + len(scales), len(params), ", ".join(params), len(scales), scale,
                                       ", ".join(scales))
    else:
        named = MODEL_ONLY_TEXT.format(len(params), ", ".join(params))
    return " ".join((named, FORCING_TEXT if forcing else NO_FORCING_TEXT, READS_TEXT.format(READS.get(mode, mode))))


def saved_prior_words(made: str, model: str, parameters: int, built_on: "str | None") -> str:
    """A saved prior, under its list: when it was made, its model by its name, how many parameters it was made for and
    the bounds file it was built on."""
    return SAVED_PRIOR_TEXT.format(made, model_label(model), parameters, TYPED_BY_HAND if built_on is None else built_on)


def prior_in_use(label: str, built_on: "str | None", parameters: int) -> tuple[str, str]:
    """What the second step holds, as "In use" says it: the prior's label, and the bounds file its record was built on,
    with how many parameters it holds; ``("", "")`` while it holds none."""
    if not label:
        return "", ""
    return label, PRIOR_IN_USE_TEXT.format(TYPED_BY_HAND if built_on is None else built_on, parameters)


def _bounds_fact(built_on: "str | None", selected: str, edited: bool, same_box: bool) -> Fact:
    """The bounds file a saved prior was built on, against the one selected: said, never blocking, for the engine is
    the one that refuses a load."""
    if built_on is None:
        return Fact("info", TYPED_TEXT) if same_box else Fact("warn", f"{TYPED_TEXT} {TYPED_REFUSES_TEXT}")
    if built_on != selected:
        return Fact("warn", f"{OTHER_FILE_TEXT.format(built_on, selected)} "
                            f"{CHECK_PASSES_TEXT if same_box else CHECK_REFUSES_TEXT}")
    if edited:
        return Fact("warn", EDITED_TEXT.format(built_on) + ("" if same_box else f" {EDITED_REFUSES_TEXT}"))
    return Fact("fine", SAME_FILE_TEXT.format(built_on))


def _sets_fact(batches: Sequence[int]) -> Fact:
    """The simulation sets made with a saved prior, by the batches each holds."""
    if not batches:
        return Fact("info", NO_SETS_TEXT)
    if len(batches) == 1:
        return Fact("info", ONE_SET_TEXT.format(batches[0]))
    return Fact("info", SETS_TEXT.format(len(batches), max(batches)))


def _length_fact(took) -> Fact:
    """How long a build takes: what the last build on this computer took."""
    if took is None:
        return Fact("info", NEVER_BUILT_TEXT)
    seconds = build_length(took)
    if seconds is None:
        return Fact("info", NO_LENGTH_TEXT)
    said = about_words(seconds)
    return Fact("info", TOOK_TEXT.format(said[:1].upper() + said[1:], length_words(seconds)))


def prior_facts(*, view: str, needs: bool = False, bounds_error: str = "", bounds_blocks: str = BOUNDS_BLOCKS,
                prior_error: str = "", prior_blocks: str = PRIOR_BLOCKS, record_model: "str | None" = None,
                model: "str | None" = None, built_on: "str | None" = None, selected: str = "", edited: bool = False,
                same_box: bool = True, sets_of_prior: "Sequence[int] | None" = None, rounds: "int | None" = None,
                batch: "int | None" = None, stable: "int | None" = None, steps: "int | None" = None, took=None,
                sets_for_model: int = 0, attention: Sequence[str] = (), others_pending: Sequence[str] = (),
                name_facts: Sequence[Fact] = (), field_facts: Sequence[Fact] = (),
                plan_lines: Sequence[Fact] = ()) -> tuple[Fact, ...]:
    """The facts of the second step, in the order of its table, those that block first. ``view`` is "saved" or
    "build". ``needs``: no model is applied, so nothing can be made. ``bounds_error`` and ``prior_error`` say why the
    bounds file or the saved prior chosen cannot be used, each with the words beside the button. A saved prior:
    ``record_model``, the model its record was made for, against ``model``, the one applied; ``built_on``, the bounds
    file it was built on (None for limits typed by hand), against ``selected``, the file the list selects, "" for none,
    with ``edited`` (the same file, edited since) and ``same_box`` (its parameters and their limits are the selected
    file's); with no file selected the record is compared with none, and what it was built on is said only in "In use"
    and under the list of saved priors; ``sets_of_prior``, the batches of
    each saved simulation set made with it, None for not read. A build: ``rounds``, ``batch`` and ``stable`` give what it
    costs, ``steps`` the steps each candidate is integrated over when they are known, any of the three None leaves the
    cost out; ``took``, the run history's last build, how long it takes; ``sets_for_model``, the saved simulation sets of
    the model, which a new prior cannot continue. Then ``attention``, what the configuration in use asks a person to
    know; ``others_pending``, the sentences of the steps before it whose changes are not applied, which do not block;
    the name's and the fields' facts; and what the plan in use says."""
    facts = [Fact("lock", NEEDS_A_MODEL, NEEDS_A_MODEL_BLOCKS)] if needs else []
    if bounds_error:
        facts.append(Fact("err", bounds_error, bounds_blocks))
    if view == "saved":
        if prior_error:
            facts.append(Fact("err", prior_error, prior_blocks))
        elif record_model is not None and model is not None and record_model != model:
            facts.append(Fact("err", OTHER_MODEL_TEXT.format(model_label(record_model)), OTHER_MODEL_BLOCKS))
        else:
            if record_model is not None and not bounds_error and selected:
                facts.append(_bounds_fact(built_on, selected, edited, same_box))
            if sets_of_prior is not None:
                facts.append(_sets_fact(sets_of_prior))
    else:
        if None not in (rounds, batch, stable):
            facts.append(Fact("info", COST_TEXT.format(rounds * batch, rounds, batch,
                                                       STEPS_TEXT.format(steps) if steps else "", stable)))
        facts.append(_length_fact(took))
        if sets_for_model == 1:
            facts.append(Fact("info", ONE_OTHER_TEXT))
        elif sets_for_model > 1:
            facts.append(Fact("info", OTHERS_TEXT.format(sets_for_model)))
    facts += [Fact("warn", sentence) for sentence in attention]
    facts += [Fact("warn", said) for said in others_pending if said]
    return ordered((*facts, *name_facts, *field_facts, *plan_lines))


# ── the third step, Training ──
START_WORDS = "Start training"
NEEDS_A_PRIOR = "Needs a prior: build or load one in step 2, Prior."
NEEDS_A_PRIOR_BLOCKS = "needs a prior"
USES_TEXT = "Model {}; bounds file {}; prior {}; {}; rotation {}."
#: What a training's observation is, by the kind of observation of the configuration in use.
USES_KIND = {"chi": "frequency response with {} slots", "forced": "a spontaneous recording and one driven recording",
             "spontaneous": "a spontaneous recording only"}
BUDGET_TEXT = "{:,} batches of {:,} make {:,} simulations. Each simulates {}."
#: What each simulation of a training simulates, by the kind of observation; the frequency response's driven
#: recordings run from the fewest an observation holds to its slots.
EACH_SIMULATES = {"chi": "one spontaneous recording and {} to {} driven recordings",
                  "forced": "one spontaneous recording and one driven recording",
                  "spontaneous": "one spontaneous recording"}
LEAST_DRIVEN = 2
DRIVEN_EACH_TEXT = "one spontaneous recording and its driven recordings"
CONTINUES_TEXT = "Continues the saved simulation set: {:,} of {:,} batches are done."
NO_SIMULATIONS_TEXT = "No simulations this time: the saved {:,} are used."
NO_SIMULATIONS_BATCHES_TEXT = "No simulations this time: the saved {:,} batches are used."
NONE_SAVED_SET_TEXT = "Starts a new simulation set; none is saved for this model yet."
ONE_NOT_MATCHING_TEXT = "Starts a new simulation set. 1 saved set for this model does not match these settings:"
NOT_MATCHING_TEXT = "Starts a new simulation set. {} saved sets for this model do not match these settings:"
SET_DIFFERS_TEXT = "the set of {:,} batches ({}) differs in {}"
SET_SAID_TEXT = "the set of {:,} batches ({}): {}"
#: A saved set that differs in nothing: only a folder renamed by hand can be one.
NO_DIFFERENCE_TEXT = "No setting differs: its folder was renamed by hand, so a training does not find it."
DIFFERENCE_TEXT = "{}: {} here, {} in this set"
NOT_SAVED_TEXT = "The simulations of this training are not saved, so no saved simulation set is continued."
SETS_UNREAD_TEXT = "The saved simulation sets could not be read: {}."
NOTHING_TO_CONTINUE_TEXT = "There is no saved simulation set to continue."
NOTHING_TO_CONTINUE_BLOCKS = "there is nothing to continue"
WOULD_CONTINUE_TEXT = "A saved simulation set would be continued."
WOULD_CONTINUE_BLOCKS = "a saved simulation set would be continued"
FIT_SAVED_TEXT = "The fit is saved after every epoch, so a stop or a failure keeps it."
READ_BACK_TEXT = "A resume first reads the simulation set back, about four minutes at full size."
FIT_LOST_TEXT = ("A training whose simulation set is not kept cannot be resumed: a stop or a failure during the fit "
                 "loses it.")
SIMULATION_TAKES_TEXT = "Simulation: {}: {}."
FIT_TAKES_TEXT = "Fit: {} for {:,} epochs: {}."
EPOCH_TAKES_TEXT = "Fit: {}."
NO_ESTIMATE_TEXT = ("No time estimate: no training of this size has run on this computer, and the timing of a short "
                    "fit has not been run.")
PENDING_TEXT = "Step {}, {}, has changes that are not applied. This run uses what is in use: {}."
POSTERIOR_IN_USE_TEXT = "{:,} epochs, network {} by {}, on {:,} simulations, prior {}. Saved as a record."
POSTERIOR_SAVED_TEXT = "Saved as a record."
FROM_ROUND_TEXT = "From a narrowing round."
LIMIT_REACHED_TEXT = "The fit reached the epoch limit; the network kept is its last epoch's."
NO_POSTERIOR_TEXT = "Start a training to make a posterior."
#: The states of the saved simulation sets in which a training continues the set of its own settings.
CONTINUING = ("resume_partial", "resume_complete")


def _whole(value) -> "int | None":
    """``value`` as a whole number above nothing a sentence can count, or None: a truth value, a fraction, a number
    a float cannot hold, or anything else is none."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        return None
    return value


def uses_words(*, model, bounds, prior: str, kind, slots, rotate) -> str:
    """What a training uses, in one sentence: the model by its name, the bounds file (``""`` for limits typed by hand),
    the prior by its label, the kind of observation with its slots, and the rotation."""
    if kind == "chi":
        held = _whole(slots)
        observed = USES_KIND["chi"].format(f"{held:,}") if held else "frequency response"
    else:
        observed = USES_KIND.get(kind, str(kind))
    return USES_TEXT.format(model_label(model), bounds or TYPED_BY_HAND, prior, observed,
                            "on" if rotate is True else "off" if rotate is False else "unknown")


def budget_words(batches, width, kind, slots) -> str:
    """How many simulations a training makes, from its ``batches`` and the simulations of each, ``width``, and what each
    simulates by the kind of observation; "" while either count is not known."""
    batches, width = _whole(batches), _whole(width)
    if batches is None or width is None:
        return ""
    if kind == "chi":
        held = _whole(slots)
        each = EACH_SIMULATES["chi"].format(LEAST_DRIVEN, held) if held is not None else DRIVEN_EACH_TEXT
    else:
        each = EACH_SIMULATES.get(kind, "one observation")
    return BUDGET_TEXT.format(batches, width, batches * width, each)


def _raw_lines(fields) -> tuple:
    """What a saved set differs in, by the names its identity gives the entries, for a set the page named no lines
    of."""
    return tuple(DIFFERENCE_TEXT.format(one.get("field"), one.get("mine"), one.get("theirs")) if isinstance(one, dict)
                 else str(one) for one in (fields or ()))


def of_this_model(diffs) -> list:
    """The saved sets of ``diffs`` made for the model in use: a set whose model differs is another model's."""
    try:
        held = list(diffs or ())
    except TypeError:
        return []
    return [one for one in held
            if not any(isinstance(field, dict) and field.get("field") == "model" for field in getattr(one, "fields", ()) or ())]


def _unmatched_fact(sets, differences) -> Fact:
    """The saved sets of this model a new set is started beside, each with what it differs in: the lines the page
    named for it in ``differences``, by the set's name, else its identity's own names."""
    head = ONE_NOT_MATCHING_TEXT if len(sets) == 1 else NOT_MATCHING_TEXT.format(len(sets))
    said = []
    for one in sets:
        batches = getattr(one, "batches", None)
        count = batches if isinstance(batches, int) and not isinstance(batches, bool) else 0
        state = "complete" if getattr(one, "complete", False) else "partial"
        fields = getattr(one, "fields", ()) or ()
        lines = differences.get(getattr(one, "name", None)) if isinstance(differences, dict) else None
        lines = tuple(lines) if lines else _raw_lines(fields)
        said.append(SET_DIFFERS_TEXT.format(count, state, "; ".join(lines)) if fields and lines
                    else SET_SAID_TEXT.format(count, state, "; ".join(lines) or NO_DIFFERENCE_TEXT))
    return Fact("warn", " ".join((head, *(f"{one[:1].upper()}{one[1:].rstrip('.')}." for one in said))))


def continue_facts(reading, *, never: bool, resume: str, batches, differences=None) -> list:
    """What a training does with the saved simulation sets, as the last reading found them, ``reading`` (a training's
    reading of them, or None for none made); and what the choice "Continue or start new", ``resume``, stops. With
    ``never`` the simulations are not saved, and nothing is read. ``differences``: what each set of this model that
    does not match differs in, as the page names it, by the set's name."""
    if never:
        return [Fact("warn", NOT_SAVED_TEXT)]
    if reading is None:
        return []
    error = getattr(reading, "checkpoint_error", None)
    if error:
        return [Fact("warn", SETS_UNREAD_TEXT.format(str(error).rstrip(".")))]
    state, facts = getattr(reading, "checkpoint", None), []
    done = _whole(getattr(reading, "batches_done", None))
    if state == "resume_partial" and done is not None and _whole(batches) is not None:
        facts.append(Fact("fine", CONTINUES_TEXT.format(done, batches)))
    elif state == "resume_complete" and done is not None:
        width = _whole(getattr(reading, "width", None))
        facts.append(Fact("fine", NO_SIMULATIONS_TEXT.format(done * width) if width is not None
                          else NO_SIMULATIONS_BATCHES_TEXT.format(done)))
    elif state in ("new", "new_with_siblings"):
        sets = of_this_model(getattr(reading, "sibling_diffs", ()))
        facts.append(_unmatched_fact(sets, differences) if sets else Fact("info", NONE_SAVED_SET_TEXT))
    if resume == "require" and state in ("new", "new_with_siblings"):
        facts.append(Fact("err", NOTHING_TO_CONTINUE_TEXT, NOTHING_TO_CONTINUE_BLOCKS))
    elif resume == "never" and state in CONTINUING:
        facts.append(Fact("err", WOULD_CONTINUE_TEXT, WOULD_CONTINUE_BLOCKS))
    return facts


def _pace(estimate) -> "tuple[float, str] | None":
    """A pace the run history gives, its seconds and its words, or None for one that says no length."""
    try:
        seconds, words = estimate
        seconds = float(seconds)
    except (TypeError, ValueError, OverflowError):
        return None
    return (seconds, str(words)) if math.isfinite(seconds) and seconds > 0 else None


def how_long_words(*, per_batch=None, per_epoch=None, batches=0, epochs=None) -> str:
    """How long a training should take: its ``batches`` still to simulate at the pace ``per_batch``, and its fit at the
    pace ``per_epoch`` for ``epochs``, each pace as the run history says it, its seconds and its words; each half left
    out when it is not known, and both together said to be unknown."""
    said = []
    pace, count = _pace(per_batch), _whole(batches)
    if pace is not None and count is not None:
        about = about_words(pace[0] * count)
        if about:
            said.append(SIMULATION_TAKES_TEXT.format(about, pace[1]))
    pace, held = _pace(per_epoch), _whole(epochs)
    if pace is not None:
        about = about_words(pace[0] * held) if held is not None else ""
        said.append(FIT_TAKES_TEXT.format(about, held, pace[1]) if about else EPOCH_TAKES_TEXT.format(pace[1]))
    return " ".join(said) or NO_ESTIMATE_TEXT


def pending_words(number: int, name: str, held: str) -> str:
    """Another step whose page shows changes it has not applied: this run uses what that step holds, ``held``."""
    return PENDING_TEXT.format(number, name, held)


def posterior_in_use(label: str, training, config, prior: str, narrowed: bool = False) -> tuple[str, str]:
    """What the third step holds, as "In use" says it: the posterior's label, and from its record how many epochs it
    was fitted for, its network, the simulations it was trained on and its prior, that it came from a narrowing round
    when ``narrowed``, and that a fit that ran past its limit, which a fit reaches only by running one epoch more, kept
    its last epoch's network; ``("", "")`` while it holds none. What its record does not hold is left out."""
    if not label:
        return "", ""
    training = training if isinstance(training, dict) else {}
    config = config if isinstance(config, dict) else {}
    epochs, runs, size = (_whole(training.get(key)) for key in ("epochs_trained", "n_runs", "run_size"))
    width, depth = (_whole(training.get(key)) for key in ("hidden_features", "num_transforms"))
    if None in (epochs, runs, size, width, depth) or not prior:
        said = POSTERIOR_SAVED_TEXT
    else:
        said = POSTERIOR_IN_USE_TEXT.format(epochs, width, depth, runs * size, prior)
    if narrowed:
        said = f"{said[:-len(POSTERIOR_SAVED_TEXT)]}{FROM_ROUND_TEXT} {POSTERIOR_SAVED_TEXT}"
    limit = _whole(config.get("max_num_epochs"))
    if epochs is not None and limit is not None and epochs > limit:
        said = f"{said} {LIMIT_REACHED_TEXT}"
    return label, said


# ── the load view ──
LOAD_POSTERIOR_WORDS = "Load this posterior"
NO_POSTERIOR_CHOSEN_TEXT = "Choose a saved posterior."
NO_POSTERIOR_CHOSEN_BLOCKS = "no saved posterior is chosen"
NO_POSTERIOR_SAVED_TEXT = "No posterior is saved yet: start a training with 'Set up a training instead…'."
NO_POSTERIOR_SAVED_BLOCKS = "no posterior is saved yet"
POSTERIOR_BLOCKS = "the posterior cannot be read"
OTHER_MODEL_POSTERIOR_TEXT = "This posterior was made for the model {}."
OTHER_MODEL_POSTERIOR_BLOCKS = "this posterior is for another model"
TRAINED_WITH_TEXT = "Trained with {}, the prior in use."
TRAINED_OTHER_TEXT = ("Trained with {}; the prior in use is {}. Calibration and a narrowing round will refuse this "
                      "pair. "
                      "Load {} in step 2 first.")
TRAINED_GONE_TEXT = ("Trained with a prior no longer in Records; the prior in use is {}. Calibration and a narrowing "
                     "round will refuse this pair.")
MADE_FOR_WORDS = "Made {}, for {}"
FOR_WORDS = "For {}"
NETWORK_WORDS = "network {} by {}"
SIMULATIONS_WORDS = "{:,} simulations"
TRAINED_ON_WORDS = "trained with {}, on {}"
ROUND_AROUND_TEXT = "From a narrowing round around {}."
OBSERVATION_WORDS = "the observation {}"
NO_OBSERVATION_WORDS = "an observation no longer in Records"
NO_PRIOR_WORDS = "a prior no longer in Records"


def saved_posterior_words(*, made: str, model, kind, network=None, simulations=None, prior: str = "",
                          bounds: "str | None" = None, narrowed: bool = False, observation: str = "") -> str:
    """A saved posterior, under its list, from its record: when it was made and for which model; the kind of
    observation it reads, its network's width by depth, the simulations it was trained on, the prior it was trained
    with and the bounds file it was built on (None for limits typed by hand); and, from a narrowing round, the
    observation it was narrowed around by its label ("" when its record is no longer in Records). What its record does
    not hold is left out, when it was made too (``made`` "")."""
    parts = [READS.get(kind, str(kind))] if kind else []
    held = [_whole(one) for one in (network or (None, None))]
    if None not in held:
        parts.append(NETWORK_WORDS.format(*held))
    if _whole(simulations) is not None:
        parts.append(SIMULATIONS_WORDS.format(simulations))
    if prior:
        parts.append(TRAINED_ON_WORDS.format(prior, TYPED_BY_HAND if bounds is None else bounds))
    head = MADE_FOR_WORDS.format(made, model_label(model)) if made else FOR_WORDS.format(model_label(model))
    said = f"{head}: {'; '.join(parts)}." if parts else f"{head}."
    if narrowed:
        around = OBSERVATION_WORDS.format(observation) if observation else NO_OBSERVATION_WORDS
        said = f"{said} {ROUND_AROUND_TEXT.format(around)}"
    return said


def load_facts(*, needs_prior: bool = False, chosen: bool = True, saved: bool = True, error: str = "",
               record_model=None, model=None, trained_with=None, trained_with_label: str = "", in_use=None,
               in_use_label: str = "", others_pending: Sequence[str] = (),
               plan_lines: Sequence[Fact] = ()) -> tuple[Fact, ...]:
    """The facts of the third step's load view, those that block first. ``needs_prior``: no prior is in use, and a
    load reads the prior first. ``chosen``: a saved posterior is selected, and ``saved``: one is saved at all.
    ``error``: why the record selected cannot be read. ``record_model``, the model its record was made for, against
    ``model``, the one in use. Then the prior it was trained with, ``trained_with`` by its id and its label (empty when
    it is no longer in Records), against the prior in use, ``in_use`` by its id and its label: never blocking, for the
    load goes through and what refuses the pair is a later step. Then the other steps' changes not applied, and what
    the plan in use says. Reads nothing, and raises on nothing a record holds."""
    facts = [Fact("lock", NEEDS_A_PRIOR, NEEDS_A_PRIOR_BLOCKS)] if needs_prior else []
    if not chosen:
        facts.append(Fact("err", NO_POSTERIOR_CHOSEN_TEXT, NO_POSTERIOR_CHOSEN_BLOCKS) if saved
                     else Fact("err", NO_POSTERIOR_SAVED_TEXT, NO_POSTERIOR_SAVED_BLOCKS))
    elif error:
        facts.append(Fact("err", error, POSTERIOR_BLOCKS))
    elif record_model is not None and model is not None and record_model != model:
        facts.append(Fact("err", OTHER_MODEL_POSTERIOR_TEXT.format(model_label(record_model)),
                          OTHER_MODEL_POSTERIOR_BLOCKS))
    elif not needs_prior and trained_with is not None:
        if trained_with == in_use:
            facts.append(Fact("fine", TRAINED_WITH_TEXT.format(in_use_label)))
        elif trained_with_label:
            facts.append(Fact("warn", TRAINED_OTHER_TEXT.format(trained_with_label, in_use_label, trained_with_label)))
        else:
            facts.append(Fact("warn", TRAINED_GONE_TEXT.format(in_use_label)))
    facts += [Fact("warn", said) for said in others_pending if said]
    return ordered((*facts, *plan_lines))


# ── the question before loading a posterior from a narrowing round ──
NARROWED_TITLE = "Load a posterior from a narrowing round?"
NARROWED_TEXT = ("{} was narrowed around {}. It is valid only near that observation; elsewhere it has seen no training "
                 "simulations and gives confident answers that are wrong.")
NARROWED_THEN = ("If you load it: calibration runs on the narrowed prior; inference on any other observation asks "
                 "first; "
                 "a further narrowing round can only be drawn around the same observation.")


def narrowed_question(label: str, observation: str) -> tuple:
    """The question asked before a posterior from a narrowing round is loaded: its title, its sentence, the sentence
    beneath it, and the answer that loads it; ``observation`` is the observation it was narrowed around by its label,
    "" when its record is no longer in Records."""
    around = OBSERVATION_WORDS.format(observation) if observation else NO_OBSERVATION_WORDS
    return NARROWED_TITLE, NARROWED_TEXT.format(label, around), NARROWED_THEN, LOAD_POSTERIOR_WORDS


# ── the choice of a kept training that matches ──
NEW_TRAINING_TITLE = "Start a new training?"
ONE_KEPT_TEXT = "1 kept training matches these settings and can be resumed:"
KEPT_TEXT = "{} kept trainings match these settings and can be resumed:"
LEAVES_IT_TEXT = "A new training leaves it as it is. To resume it, cancel and press Resume beside it on this page."
LEAVES_THEM_TEXT = ("A new training leaves them as they are. To resume one, cancel and press Resume beside it on this "
                    "page.")
NEW_TRAINING_ANSWER = "Start a new training"
AFTER_ERROR_TEXT = "{} after the error '{}'"


def stopped_words(state, on_purpose, reason) -> str:
    """How a kept training stopped, in words: the listing's own word for its state, and, for one an error stopped, the
    error as its record words it."""
    said = str(state or "stopped unexpectedly")
    if on_purpose is False and isinstance(reason, str) and reason.strip():
        return AFTER_ERROR_TEXT.format(said, reason.strip())
    return said


def kept_words(label: str, how: str, passes, saved: str) -> str:
    """A kept training as the choice names it: its label, how it stopped, the epoch it was saved after and when, in the
    window's words: "retrain_2: kept when stopped, after epoch 41, saved 3 Oct 2026, 03:12"."""
    parts = [how]
    if _whole(passes) is not None:
        parts.append(f"after epoch {passes}")
    if saved:
        parts.append(f"saved {saved}")
    return f"{label}: {', '.join(parts)}"


def new_training_question(kept: Sequence[str]) -> tuple:
    """The question asked before a training starts anew while kept trainings match its settings: its title, its
    sentence, its items, ``kept`` as ``kept_words`` names each, the sentence beneath them and the answer that starts."""
    one = len(kept) == 1
    return (NEW_TRAINING_TITLE, ONE_KEPT_TEXT if one else KEPT_TEXT.format(len(kept)), tuple(kept),
            LEAVES_IT_TEXT if one else LEAVES_THEM_TEXT, NEW_TRAINING_ANSWER)


# ── the question before simulating from zero ──
NEW_SET_TITLE = "Start a new simulation set?"
NEW_SET_TEXT = ("This training will simulate {:,} batches from zero. Saved simulation sets for this model exist, and "
                "none matches these settings.")
NEW_SET_NEVER_TEXT = ("This training will simulate {:,} batches from zero. Saved simulation sets for this model exist, "
                      "and none will be continued.")
SET_BACK_TEXT = "To continue one of them, cancel and set these back."
NEW_SET_ANSWER = "Start training on a new simulation set"
NOT_SAVED_ANSWER = "Start training without saving the simulations"
CANCEL = "Cancel"
OWN_SET_TEXT = "No setting differs. It is not continued because the simulations of this training are not saved."
SET_IS_TEXT = "{:,} batches, {}"


def set_words(label: str, made: str, batches, complete: bool) -> str:
    """A saved simulation set as the question names it, by its label in Records, then in brackets when it was made,
    when that is known, the batches it holds and whether it is whole: "Simulation set for retrain_prior (2 Oct 2026,
    09:10; 10,000 batches, complete)"."""
    count = _whole(batches) or 0
    held = SET_IS_TEXT.format(count, "complete" if complete else "partial")
    return f"{label} ({made}; {held})" if made else f"{label} ({held})"


def new_set_question(*, batches, never: bool, own: str = "", sets=()) -> tuple:
    """The question asked before a training simulates from zero while saved sets of its model exist: its title, its
    sentence, its items, the sentence beneath them, and the answer that starts it. ``own``: with ``never``, the set
    these very settings would continue, as ``set_words`` names it, which heads the list; ``sets``: each other set as
    ``set_words`` names it with what it differs in, line by line. Values in, values out."""
    items = []
    if own:
        items += [f"{own}:", OWN_SET_TEXT]
    for named, lines in sets:
        items += [f"{named} differs in:", *lines]
    text = (NEW_SET_NEVER_TEXT if never else NEW_SET_TEXT).format(_whole(batches) or 0)
    return NEW_SET_TITLE, text, tuple(items), SET_BACK_TEXT, NOT_SAVED_ANSWER if never else NEW_SET_ANSWER


def training_facts(*, needs_prior: bool = False, uses: str = "", budget: str = "", reading=None, never: bool = False,
                   resume: str = "auto", batches=None, differences=None, how_long: str = "", memory: str = "",
                   others_pending: Sequence[str] = (), name_facts: Sequence[Fact] = (),
                   field_facts: Sequence[Fact] = (), plan_lines: Sequence[Fact] = ()) -> tuple[Fact, ...]:
    """The facts of the third step, in the order of its table, those that block first. ``needs_prior``: no prior is in
    use, so nothing can be trained. ``uses`` and ``budget``: what the run uses and how many simulations it makes, said
    when known. ``reading``: the last reading of the saved simulation sets for this budget, or None; ``never``: the
    simulations are not saved; ``resume``, the choice "Continue or start new", and ``batches``, the batches typed, for
    what the sets say and what the choice stops; ``differences``, what each set that does not match differs in, by the
    set's name, as the page names it. Then that the fit is saved and how a resume reads back, or that it cannot be
    resumed; ``how_long`` and ``memory``, each one sentence or more; ``others_pending``, the sentences of the other
    steps that hold changes not applied; the name's and the fields' facts; and what the plan in use says. Reads nothing,
    and raises on nothing a reading or a session holds."""
    facts = [Fact("lock", NEEDS_A_PRIOR, NEEDS_A_PRIOR_BLOCKS)] if needs_prior else []
    facts += [Fact("info", said) for said in (uses, budget) if said]
    facts += continue_facts(reading, never=never, resume=resume, batches=batches, differences=differences)
    facts += [Fact("warn", FIT_LOST_TEXT)] if never else [Fact("info", FIT_SAVED_TEXT), Fact("info", READ_BACK_TEXT)]
    facts += [Fact("info", said) for said in (how_long, memory) if said]
    facts += [Fact("warn", said) for said in others_pending if said]
    return ordered((*facts, *name_facts, *field_facts, *plan_lines))


#: The blocks that neither waiting nor an edit on the page lifts, beside which the button reads "Cannot start:": a model
#: parameter inference cannot take, a file or a record that cannot be read, a record made for another model, and nothing
#: to continue. Every other block is lifted on the page (a field, a choice, a name, a step applied, a record chosen or
#: made) and reads "Cannot start yet:".
FINAL_BLOCKS = frozenset({UNUSABLE_BLOCKS, BOUNDS_BLOCKS, PRIOR_BLOCKS, OTHER_MODEL_BLOCKS, NOTHING_TO_CONTINUE_BLOCKS,
                          POSTERIOR_BLOCKS, OTHER_MODEL_POSTERIOR_BLOCKS})
