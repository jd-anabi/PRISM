"""What a step says before it starts, as values: a fact and the order facts stand in, the reason beside the main
button, the line that says when the saved records were read, where a field's value came from, and how long something
takes, in a person's words; and a row that picks what a step holds.

A fact is a marker with its lead word, one or two plain sentences, and, for what stops the start, the few words that go
beside the button. Every "Error." and every "Blocked." blocks and nothing else does, and a fact that says otherwise is
refused as it is made, so that a marker never disagrees with the button.

Nothing here imports the toolkit or the engine: a step's facts are worked out and tested as values, with no widget.
"""
import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime

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
