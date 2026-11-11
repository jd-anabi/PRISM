"""A check's result as a widget: from its record's manifest and its folder, or, for a check that keeps nothing but its
result, from its run in the run history. The verdict sentence with the marker of its state, the table, and the figures
the record holds, each captioned in the window's words, in a result region; the engine's own sentences and words are
never shown. A check's page shows it; Records can host the same block.

A block reads the record's figures from its folder as it is made, so it is made only while no run is live: the Checks
page makes its blocks in its reading of the disk, never as it draws its facts.
"""
import re
from collections.abc import Mapping
from pathlib import Path

from core.artifacts.store import slug

from ... import check_readings
from ...check_catalogue import entry
from ...step_facts import Fact
from ...widgets.result_region import ResultRegion

#: The calibration's three figures by the engine's titles, each made a file name as the store makes one, and the words
#: the window captions it with.
FIGURE_CAPTIONS: dict[str, str] = {slug("SBC ranks (CDF)"): "Rank test, cumulative",
                                   slug("SBC ranks (histogram)"): "Rank test, histogram",
                                   slug("TARP coverage"): "Joint coverage test"}
#: A figure the window has no words for, captioned by its check and its place among the record's figures.
FIGURE_TEXT = "{}, figure {}"
#: The number the store adds to a figure's file name when the record holds another of the same title.
_AGAIN = re.compile(r"-\d+$")


def figures_of(key: str, manifest: Mapping, folder: "Path | str") -> list:
    """Each figure the record holds, as ``(its caption, its path)``, in the record's order: a figure of the
    calibration by its words, any other by its check's name and its place."""
    listed = manifest.get("figures")
    found = []
    for at, name in enumerate(listed if isinstance(listed, list) else (), start=1):
        if not isinstance(name, str):
            continue
        stem = _AGAIN.sub("", Path(name).stem)
        found.append((FIGURE_CAPTIONS.get(stem, FIGURE_TEXT.format(entry(key).name, at)), Path(folder) / name))
    return found


def result_block(key: str, manifest: Mapping, folder: "Path | str", *, units: Mapping[str, str] = {},
                 force_unit: str = "", others_open: int = 0, prior_words: str = "",
                 text: "str | None" = None) -> ResultRegion:
    """The result of the record of the check ``key`` whose manifest is ``manifest``, in the folder ``folder``: its
    verdict sentence as a message row with the marker of the state the record reads, its table, and its figures,
    captioned. ``units``, each parameter's unit by its name; ``force_unit``, the cell's unit of force; ``others_open``,
    for a passed calibration, how many other deciding checks that apply have not passed; ``prior_words``, the audit's
    prior as a sentence names it; ``text``, the record's text, for what the observations can pin down."""
    region = ResultRegion()
    state = check_readings.read(key, manifest).state
    region.set_verdict(check_readings.STATE_MARKERS.get(state, "later"),
                       check_readings.verdict_sentence(key, manifest, others_open=others_open,
                                                       prior_words=prior_words, force_unit=force_unit))
    region.set_table(check_readings.table_words(key, manifest, units=units, force_unit=force_unit, text=text))
    region.set_pictures(figures_of(key, manifest, folder))
    return region


def history_block(key: str, entry_: Mapping) -> ResultRegion:
    """The result of a check that keeps nothing but its result, from its run ``entry_`` in the run history: its name and
    how the run ended, with the marker of a check done, or of a run that did not finish."""
    region = ResultRegion()
    finished = isinstance(entry_, Mapping) and entry_.get("outcome") == "finished"
    region.set_verdict("done" if finished else "err", check_readings.run_sentence(key, entry_))
    return region


def comparison_block(sentence: str, state: str, *, narrowed: bool = False) -> ResultRegion:
    """The comparison after simulating, which writes no record: its sentence with the marker of its state, and for a
    posterior from a narrowing round that the audit measured the whole prior."""
    region = ResultRegion()
    region.set_verdict(check_readings.STATE_MARKERS.get(state, "later"), sentence)
    if narrowed:
        region.set_sentences((Fact("info", check_readings.NARROWED_COMPARISON_TEXT),))
    return region
