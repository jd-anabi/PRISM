"""What a record is called on screen: one word for each kind of record, and one label for each record, the
same in every list, table and sentence that names it.

A named record is called by its name. An unnamed one is called by its kind and the moment it was made, in local
time, as the window writes a date ("Unnamed calibration, 3 Oct 2026, 14:02"); a simulation set, which has no
name until it is renamed, by the prior it was simulated for. Imports no torch and nothing of the engine: the
date's words are the look's.
"""
from datetime import datetime

from .design import date_text

# one word for one record of each kind the store keeps
KIND_WORDS: dict[str, str] = {
    "prior": "prior",
    "simulation": "simulation set",
    "posterior": "posterior",
    "observation": "observation",
    "calibration": "calibration",
    "inference": "inference result",
    "diagnostic": "check",
    "fdt": "effective-temperature measurement",
}

# an effective-temperature record is a measurement on one cell, a sweep study's sweep, or a comparison of them
_STUDY_WORDS = {"single": "effective-temperature measurement", "sweep": "sweep", "comparison": "comparison"}


def kind_word(kind: str, study: str | None = None) -> str:
    """The word for one record of ``kind``; for an effective-temperature record, the word for its ``study``
    ("single", "sweep" or "comparison") when it is given. "record" for a kind the store does not keep."""
    if kind == "fdt" and isinstance(study, str) and study in _STUDY_WORDS:
        return _STUDY_WORDS[study]
    return KIND_WORDS.get(kind, "record") if isinstance(kind, str) else "record"


def record_label(kind: str, name: str, created: datetime | None, *, prior_label: str | None = None,
                 study: str | None = None) -> str:
    """A record's label: its name when it has one; a simulation set with none is "Simulation set for
    <prior_label>", or "Simulation set for a prior that is no longer here" when its prior is not found; any
    other record with none is "Unnamed <kind word>" and the moment it was made, written as handed in (a moment
    of None leaves the date out). ``study`` is an effective-temperature record's, for its kind word."""
    if name:
        return name
    if kind == "simulation":
        return f"Simulation set for {prior_label or 'a prior that is no longer here'}"
    words = f"Unnamed {kind_word(kind, study)}"
    return f"{words}, {date_text(created)}" if created is not None else words


def label_of(loaded) -> str:
    """The label of a record the session holds (a loaded prior, posterior, observation or the like): its kind,
    its name and the moment its manifest says it was made, turned into local time. Never raises: a field that
    is missing or not what a record holds is read as absent."""
    kind = getattr(loaded, "kind", None)
    name = getattr(loaded, "name", None)
    created = None
    try:
        created = datetime.fromisoformat(getattr(getattr(loaded, "manifest", None), "created", None)).astimezone()
    except (TypeError, ValueError, OverflowError, OSError):
        pass
    body = getattr(loaded, "body", None)
    study = body.get("study") if isinstance(body, dict) else None
    return record_label(kind if isinstance(kind, str) else "", name if isinstance(name, str) else "", created,
                        study=study)
