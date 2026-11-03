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


def refused_facts(sentences: Iterable[str]) -> tuple[Fact, ...]:
    """What a refusal says among the facts, each of ``sentences`` the engine's own: an "Error." that stops the start, as
    the mark under its field does, for a marker never disagrees with the button."""
    return tuple(Fact("err", sentence, REFUSED_BLOCKS) for sentence in sentences)


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
    if len(blocking) == 1:
        return ONE_BLOCKS_TEXT.format(blocking[0])
    if blocking:
        return SEVERAL_BLOCK_TEXT.format(blocking[0], len(blocking) - 1)
    return NOTHING_TO_APPLY_TEXT if nothing_to_apply else ""


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
                sets_for_model: int = 0, attention: Sequence[str] = (), name_facts: Sequence[Fact] = (),
                field_facts: Sequence[Fact] = (), plan_lines: Sequence[Fact] = ()) -> tuple[Fact, ...]:
    """The facts of the second step, in the order of its table, those that block first. ``view`` is "saved" or
    "build". ``needs``: no model is applied, so nothing can be made. ``bounds_error`` and ``prior_error`` say why the
    bounds file or the saved prior chosen cannot be used, each with the words beside the button. A saved prior:
    ``record_model``, the model its record was made for, against ``model``, the one applied; ``built_on``, the bounds
    file it was built on (None for limits typed by hand), against ``selected``, with ``edited`` (the same file, edited
    since) and ``same_box`` (its parameters and their limits are the selected file's); ``sets_of_prior``, the batches of
    each saved simulation set made with it, None for not read. A build: ``rounds``, ``batch`` and ``stable`` give what it
    costs, ``steps`` the steps each candidate is integrated over when they are known, any of the three None leaves the
    cost out; ``took``, the run history's last build, how long it takes; ``sets_for_model``, the saved simulation sets of
    the model, which a new prior cannot continue. Then ``attention``, what the configuration in use asks a person to
    know; the name's and the fields' facts; and what the plan in use says."""
    facts = [Fact("lock", NEEDS_A_MODEL, NEEDS_A_MODEL_BLOCKS)] if needs else []
    if bounds_error:
        facts.append(Fact("err", bounds_error, bounds_blocks))
    if view == "saved":
        if prior_error:
            facts.append(Fact("err", prior_error, prior_blocks))
        elif record_model is not None and model is not None and record_model != model:
            facts.append(Fact("err", OTHER_MODEL_TEXT.format(model_label(record_model)), OTHER_MODEL_BLOCKS))
        else:
            if record_model is not None and not bounds_error:
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
    return ordered((*facts, *name_facts, *field_facts, *plan_lines))
