"""What each of the twelve checks is, as values: its key and number, the name the window calls it by and the sentence
under that name, its group and its role, the words of its button and of its stop button, the kind of record it writes
and, for a check record, which check and which form of it.

Nothing here imports the toolkit, torch or the engine's settings, so the chain of steps, Home and the pages read it with
no window. Every entry is built positionally, and every function here takes a check's key positionally.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class CheckEntry:
    """One check. ``key`` names it everywhere in the window; ``number`` is its place in the list, from 1; ``name`` and
    ``sentence`` are what the list and its page say of it; ``group`` is one of ``GROUPS`` and ``role`` one of ``ROLES``;
    ``button`` the words of the button that runs it ("" for a check that has none) and ``stop_words`` those of the
    button that stops it; ``kind`` the kind of record it writes ("" for none in the store), and for a check record
    ``diagnostic`` and ``variant``, as the record's body names them (a variant of None takes every form)."""
    key: str
    number: int
    name: str
    sentence: str
    group: str
    role: str
    button: str
    stop_words: str
    kind: str
    diagnostic: str | None
    variant: str | None


#: The four groups of the list, in their order, each named by when its checks are made.
GROUPS = ("Before the run", "After simulating", "After the fit", "Tools")
#: What a check does for a posterior: it decides whether the posterior passes, it informs, or it is a tool.
ROLES = ("Decides", "Informs", "Tool")
#: The words of a check's button once a record of it exists.
AGAIN_WORDS = "Run again"
#: The words of the button of a check that has no words of its own, and of the button that stops one.
RUN_THIS_WORDS = "Run this check"
STOP_THIS_WORDS = "Stop this check…"

#: The twelve, in the list's order.
CHECKS: tuple[CheckEntry, ...] = (
    CheckEntry("band", 1, "Frequency band",
               "Can driven recordings across the frequency band, at the set drive amplitude, be measured on this cell "
               "without the drive taking over the cell's own oscillation?",
               "Before the run", "Decides", "Run the frequency-band check", STOP_THIS_WORDS, "diagnostic", "probes",
               "band"),
    CheckEntry("audit", 2, "Unusable driven recordings: audit",
               "What share of the driven recordings that training will simulate from this prior is unusable, and is "
               "that share in the expected range?",
               "Before the run", "Decides", "Run the audit", "Stop the audit…", "diagnostic", "probes", "mask"),
    CheckEntry("comparison", 3, "Unusable driven recordings: simulation set against the audit",
               "Does the share of unusable driven recordings in the finished simulation set agree with the audit?",
               "After simulating", "Decides", "", STOP_THIS_WORDS, "", None, None),
    CheckEntry("calibration", 4, "Calibration", "Is the posterior honest about its uncertainty?",
               "After the fit", "Decides", "Run the calibration", "Stop the calibration…", "calibration", None, None),
    CheckEntry("input_use", 5, "Use of each part of the input",
               "Does the network respond to every part of an observation?",
               "After the fit", "Decides", "Run the input-use check", STOP_THIS_WORDS, "diagnostic", "ablation", None),
    CheckEntry("identifiability", 6, "Identifiability of the posterior",
               "Is it on record how strongly the posterior pins down each direction in parameter space?",
               "After the fit", "Decides", "Run the identifiability check", STOP_THIS_WORDS, "diagnostic",
               "identifiability", "rotation"),
    CheckEntry("pin_down", 7, "What the observations can pin down",
               "At this cell and recording length, which parameters do the observations pin down, and which move an "
               "observation alike?",
               "Before the run", "Informs", RUN_THIS_WORDS, STOP_THIS_WORDS, "diagnostic", "identifiability",
               "jacobian"),
    CheckEntry("timing", 8, "Timing of a short fit",
               "How long does one epoch of the fit take on this computer, and so how long should the full fit take?",
               "Before the run", "Informs", RUN_THIS_WORDS, STOP_THIS_WORDS, "", None, None),
    CheckEntry("fixed_number", 9, "Calibration repeated at fixed numbers of driven recordings",
               "Does the rank test still pass when every observation holds the same number of driven recordings?",
               "After the fit", "Informs", RUN_THIS_WORDS, STOP_THIS_WORDS, "diagnostic", "sbc", None),
    CheckEntry("card_test", 10, "Card test", "Does every stage run end to end on this graphics card?",
               "Tools", "Tool", RUN_THIS_WORDS, STOP_THIS_WORDS, "", None, None),
    CheckEntry("drive_study", 11, "Drive study",
               "How hard can this cell be driven before the drive takes over its own oscillation?",
               "Tools", "Tool", RUN_THIS_WORDS, STOP_THIS_WORDS, "diagnostic", "probes", "drive"),
    CheckEntry("across_prior", 12, "Identifiability across the prior",
               "Could recordings pin each parameter down at points across the prior, whatever the posterior learned?",
               "Tools", "Tool", RUN_THIS_WORDS, STOP_THIS_WORDS, "diagnostic", "identifiability", "laplace"),
)
#: The six checks that decide whether a posterior passes, in the list's order.
DECIDING: tuple[str, ...] = tuple(one.key for one in CHECKS if one.role == "Decides")

#: Each entry by its key.
_BY_KEY = {one.key: one for one in CHECKS}


def entry(key: str) -> CheckEntry:
    """The entry of the check whose key is ``key``; KeyError, naming it, for a key that is no check's."""
    if key not in _BY_KEY:
        raise KeyError(f"no check has the key {key!r}")
    return _BY_KEY[key]
