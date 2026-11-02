"""What belongs to this computer and not to the science, set in Settings and remembered for this computer: the memory
limit, the most graphics-card memory one batch of simulations may plan to use, and the wish to keep the computer awake
while a run is live.

This is the one module of the window that assigns ``core.config.SIM_VRAM_CEILING_GIB``. Every other setting reaches
its stage as an argument; this one the planner reads live at every batch (``pipeline.vram_ceiling_gib``), so a plain
assignment reaches every stage that simulates. It splits batches and costs time, and changes neither the rows nor a
simulation set's identity. The variable ``PRISM_VRAM_CEILING_GIB``, when it holds a number, wins over it, as it does for
the planner; what the screen says is in force is the planner's own answer, so the two cannot disagree.

Both are stored under this computer's name, so that a settings file carried to another machine applies no limit there:
``computer/<name>/memory_limit_gib``, the text ``none`` or a number, and ``computer/<name>/keep_awake``. Reading them
reads the settings file and never the card. The command line reads none of this. The request itself is the run
controller's, the one place that asks Windows to keep the computer awake.
"""
import math
import os
import socket
from dataclasses import dataclass

from core import config
from core.SBI import pipeline

from . import remembered, settings

#: The run's line on the limit in force, filled with the limit and where it was set; and the line when there is none.
LIMIT_TEXT = "Memory limit: {} GiB for each batch of simulations ({})."
NO_LIMIT_TEXT = "No memory limit is set on this computer."
#: Where the limit in force was set, as the run's line says it.
SET_HERE_TEXT = "set in Settings, This computer"
SET_BY_VARIABLE_TEXT = f"set by {pipeline.VRAM_CEILING_ENV}"
#: What the group says when the remembered limit cannot be read.
NOT_READ_TEXT = "The remembered limit could not be read; no limit is in force."
#: What stands for this computer's name in a key when it has none to give.
NO_NAME = "unnamed"
#: What the stored limit holds when no limit is chosen.
NO_LIMIT_STORED = "none"
#: The bytes in one GiB: the planner multiplies a limit by this and takes a whole number of the answer, which it cannot
#: of a number so large that the product is no longer finite.
GIB_BYTES = 2 ** 30


@dataclass(frozen=True)
class InForce:
    """The memory limit in force: in GiB, None for no limit; ``source``, where it was set ("setting", "environment" or
    "none"); and ``raw``, what the variable of the environment holds as it stands ("" for unset)."""
    gib: float | None
    source: str
    raw: str


def computer() -> str:
    """This computer's name, the one every record carries; "" when it cannot say: nothing stops for it."""
    try:
        return socket.gethostname()
    except (OSError, ValueError):
        return ""


def memory_limit_in_force() -> InForce:
    """The limit in force, by the planner's own answer. The variable counts when it holds something and ``float`` takes
    it, the planner's test: 0 and a negative among them, which the planner reads as no limit. A variable that holds no
    number is passed over, and the constant is read here as the planner reads it, because the planner would log its
    warning again at every draw of a page."""
    raw = os.environ.get(pipeline.VRAM_CEILING_ENV, "")
    by_variable = _a_number(raw)
    gib = pipeline.vram_ceiling_gib() if by_variable else _constant()
    source = "environment" if by_variable else "setting" if gib > 0 else "none"
    return InForce(gib if gib > 0 else None, source, raw)


def chosen() -> float | None:
    """The limit chosen in the window, whatever the variable says; None for no limit."""
    return _constant() or None


def limit_words(force: InForce) -> str:
    """The limit ``force`` as a run says it, and where it was set."""
    if force.gib is None:
        return NO_LIMIT_TEXT
    source = SET_BY_VARIABLE_TEXT if force.source == "environment" else SET_HERE_TEXT
    return LIMIT_TEXT.format(f"{force.gib:g}", source)


def limit_line(force: InForce, held: float | None, read: bool) -> str:
    """The line under the two options: what is in force and why, ``held`` being the limit remembered and ``read``
    whether it could be read."""
    named = pipeline.VRAM_CEILING_ENV
    if force.source == "environment":
        meaning = "" if force.gib is not None else ", which means no limit"
        here = (f"the limit remembered here ({held:g} GiB) is not in use" if held is not None
                else "no limit is remembered here")
        return f"Set outside PRISM by {named} = {force.raw.strip()}{meaning}. It takes precedence; {here}."
    said = NOT_READ_TEXT if not read and force.gib is None else f"In force: {_force_words(force)}."
    if force.raw.strip():
        return f"{named} is set to '{force.raw}', which is not a number. PRISM ignores it. {said}"
    return said


def locked_line(gib: float | None) -> str:
    """The line under the two options while a run is live, ``gib`` being the limit that run started with."""
    started = "no limit" if gib is None else f"a limit of {gib:g} GiB"
    return f"Locked while a run is live. This run started with {started}."


def judged(text: str, force: InForce) -> tuple[float | None, str]:
    """What the field's ``text`` comes to, with the limit ``force`` in force: the number it sets and "", or None and
    the line beside the field that says why it is not applied."""
    unchanged = f"In force, unchanged: {_force_words(force)}."
    text = text.strip()
    if not text:
        return None, f"The field is empty. {unchanged} To run without a limit, choose 'No limit'."
    if "," in text:
        return None, f"Use a point as the decimal sign. {unchanged}"
    try:
        gib = float(text)
    except ValueError:
        return None, f"That is not a number. {unchanged}"
    if not math.isfinite(gib):
        return None, f"That is not a number. {unchanged}"
    if gib <= 0:
        return None, "The limit must be above 0. To run without a limit, choose 'No limit'."
    if not math.isfinite(gib * GIB_BYTES):
        return None, f"That is too large. {unchanged}"
    return gib, ""


def set_memory_limit(gib: float | None) -> bool:
    """Make ``gib`` the limit, None for no limit, and remember it for this computer. True when it was remembered."""
    config.SIM_VRAM_CEILING_GIB = 0.0 if gib is None else float(gib)
    return remembered.put(_key("memory_limit_gib"), NO_LIMIT_STORED if gib is None else f"{gib:g}")


def stored_limit() -> tuple[float | None, bool]:
    """The limit remembered for this computer, None for none, and whether it could be read: absent or ``none`` is no
    limit, and read; anything else that is no limit one can set is no limit, and not read. Reads the settings file."""
    held = settings.settings().value(_key("memory_limit_gib"), None)
    if held is None or held == NO_LIMIT_STORED:
        return None, True
    try:
        gib = float(held)
    except (TypeError, ValueError):
        return None, False
    return (gib, True) if math.isfinite(gib * GIB_BYTES) and gib > 0 else (None, False)


def keep_awake_wish() -> bool:
    """Whether this computer is to stay awake while a run is live, as remembered: yes unless it was turned off."""
    return settings.get_bool(settings.settings(), _key("keep_awake"), True)


def set_keep_awake_wish(on: bool) -> bool:
    """Remember the wish for this computer, and hand it to the run's controller, which acts on it at once while a run
    is live. True when it was remembered."""
    from .run.controller import controller           # here: the controller imports this module
    controller().set_keep_awake(on)
    return remembered.put(_key("keep_awake"), "1" if on else "0")


def apply_remembered() -> None:
    """What this computer remembers, in force: called once as the program starts, before the window is built. A limit
    that is absent, ``none`` or cannot be read assigns nothing. Reads the settings file alone."""
    from .run.controller import controller           # here: the controller imports this module
    gib, _read = stored_limit()
    if gib is not None:
        config.SIM_VRAM_CEILING_GIB = gib
    controller().set_keep_awake(keep_awake_wish())


def _key(name: str) -> str:
    """The settings key of ``name`` for this computer; ``unnamed`` holds the place of a name it cannot give."""
    return f"computer/{computer() or NO_NAME}/{name}"


def _force_words(force: InForce) -> str:
    """The limit ``force`` in two or three words: "6 GiB", or "no limit"."""
    return "no limit" if force.gib is None else f"{force.gib:g} GiB"


def _a_number(raw: str) -> bool:
    """Whether the variable's text ``raw`` is a number as the planner reads one."""
    if not raw.strip():
        return False
    try:
        float(raw)
    except ValueError:
        return False
    return True


def _constant() -> float:
    """The constant as the planner reads it, with nothing logged: above nought, or nought."""
    try:
        return max(0.0, float(config.SIM_VRAM_CEILING_GIB or 0.0))
    except (TypeError, ValueError):
        return 0.0
