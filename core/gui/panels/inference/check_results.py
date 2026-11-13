"""A check's result as a widget: from its record's manifest and its folder, or, for a check that keeps nothing but its
result, from its run in the run history. The verdict sentence with the marker of its state, the table, and the figures
the record holds, each captioned in the window's words, in a result region; the engine's own sentences and words are
never shown. A check's page shows it; Records can host the same block.

A block reads the record's figures from its folder as it is made, so it is made only while no run is live: the Checks
page makes its blocks in its reading of the disk, never as it draws its facts. A block is made inside its check's page
(``parent``) before anything fills it: one made with no parent would be a window of its own, shown as its first words
are said.
"""
import re
from collections.abc import Mapping
from pathlib import Path

from PySide6.QtWidgets import QPushButton

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


def result_block(key: str, manifest: Mapping, folder: "Path | str", *, parent, units: Mapping[str, str] = {},
                 force_unit: str = "", others_open: int = 0, prior_words: str = "",
                 text: "str | None" = None) -> ResultRegion:
    """The result of the record of the check ``key`` whose manifest is ``manifest``, in the folder ``folder``: its
    verdict sentence as a message row with the marker of the state the record reads, its table, and its figures,
    captioned. ``units``, each parameter's unit by its name; ``force_unit``, the cell's unit of force; ``others_open``,
    for a passed calibration, how many other deciding checks that apply have not passed; ``prior_words``, the audit's
    prior as a sentence names it; ``text``, the record's text, for what the observations can pin down; made inside
    ``parent``."""
    region = ResultRegion(parent)
    state = check_readings.read(key, manifest).state
    region.set_verdict(check_readings.STATE_MARKERS.get(state, "later"),
                       check_readings.verdict_sentence(key, manifest, others_open=others_open,
                                                       prior_words=prior_words, force_unit=force_unit))
    region.set_table(check_readings.table_words(key, manifest, units=units, force_unit=force_unit, text=text))
    if key == "drive_study":
        region.set_lines(check_readings.forcing_lines(manifest))
    region.set_pictures(figures_of(key, manifest, folder))
    return region


def history_block(key: str, entry_: Mapping, *, parent) -> ResultRegion:
    """The result of a check that keeps nothing but its result, from its run ``entry_`` in the run history: its name and
    how the run ended, with the marker of a check done, or of a run that did not finish; made inside ``parent``."""
    region = ResultRegion(parent)
    finished = isinstance(entry_, Mapping) and entry_.get("outcome") == "finished"
    region.set_verdict("done" if finished else "err", check_readings.run_sentence(key, entry_))
    return region


def timing_block(runs, *, parent, expected=None, planned: "int | None" = None) -> ResultRegion:
    """The result of the timing of a short fit, from the runs of one press in the run history, oldest first: from those
    that finished, each network's time per epoch, the table and the projection for the training step 3's form holds,
    at ``planned`` simulations for ``expected`` epochs (with where they come from); when none finished, how the last
    run ended. Made inside ``parent``."""
    finished = [one for one in runs if isinstance(one, Mapping) and one.get("outcome") == "finished"
                and ((one.get("check") or {}).get("result") or {})]
    if not finished:                         # none finished, or one written before a check kept its result
        return history_block("timing", runs[-1], parent=parent)
    sentence, table, projection = check_readings.timing_words(finished, expected=expected, planned=planned)
    region = ResultRegion(parent)
    region.set_verdict("done", sentence)
    region.set_table(table)
    if projection:
        region.set_sentences((Fact("info", projection),))
    return region


def card_block(run: Mapping, *, parent, earlier: "Mapping | None" = None) -> ResultRegion:
    """The result of a card test, from its ``run`` in the run history beside ``earlier``, the run of it that finished
    before: done in how long, or the stage it failed in, each stage's seconds beside the last time's, and what it read
    of its training; never its calibration's verdict. Made inside ``parent``."""
    marker, sentence, table, lines = check_readings.card_words(run, earlier)
    region = ResultRegion(parent)
    region.set_verdict(marker, sentence)
    region.set_table(table)
    region.set_sentences(tuple(Fact("info", line) for line in lines))
    return region


def comparison_block(sentence: str, state: str, *, parent, narrowed: bool = False, table=None, accept=None,
                     accept_greyed: bool = False, kept: str = "") -> ResultRegion:
    """The comparison after simulating, which writes no record: its sentence with the marker of its state; for a
    posterior from a narrowing round, that the audit measured the whole prior; ``kept``, the answer to keep training
    beyond the last band, with its moment ("" for none); its ``table``, the set's and the audit's shares (None for
    none); and in the band to look closer at, "Accept this difference", pressed into ``accept``, or greyed, with why,
    for a set no run history names (``accept_greyed``). Made inside ``parent``."""
    region = ResultRegion(parent)
    region.set_verdict(check_readings.STATE_MARKERS.get(state, "later"), sentence)
    said = ([Fact("info", check_readings.NARROWED_COMPARISON_TEXT)] if narrowed else []) + (
        [Fact("info", kept)] if kept else [])
    if said:
        region.set_sentences(tuple(said))
    if table is not None:
        region.set_table(table)
    if accept is not None or accept_greyed:
        button = QPushButton(check_readings.ACCEPT_WORDS, region)
        if accept is not None:
            button.clicked.connect(accept)
        button.setEnabled(accept is not None)
        region.add_action(button, "" if accept is not None else check_readings.NO_HISTORY_TEXT)
    return region
