"""The pieces every rebuilt page is made of: a fact and its order, the words beside the main button, where a value came
from, the form with its rows, its fold and its name and note, the foot, "In use", and the page that holds them, with
when it reads and how it hosts a run. They are read here on a stand-in page as awkward as a real one
(``tests/_fixtures.build_step_page``), in a whole window where a page's place in it matters.
"""
import itertools

import pytest

from core.gui.run.controller import LIVE_TEXT, NOT_READ_YET_TEXT
from core.gui.step_facts import LEADS, Fact, about_words, ordered, origin_words, reading_words, reason_beside
from tests._runs import at

BLOCKS = "the name cannot be used"


# ── a fact, as values ──
def test_every_error_and_blocked_fact_blocks_and_nothing_else_does():
    """A marker never disagrees with the button: an "Error." and a "Blocked." hold the words that go beside it and no other
    fact does, and a fact that says otherwise cannot be made. Words of nothing but spaces are no words. Blocking facts come
    first, each group in the order handed in."""
    for kind in ("err", "lock"):
        assert Fact(kind, "A sentence.", BLOCKS).blocks == BLOCKS
        for words in ("", "   "):
            with pytest.raises(ValueError):
                Fact(kind, "A sentence.", words)
    for kind in ("warn", "info", "fine"):
        assert Fact(kind, "A sentence.").blocks == ""
        with pytest.raises(ValueError):
            Fact(kind, "A sentence.", BLOCKS)
    with pytest.raises(ValueError):
        Fact("done", "A marker of the rail is no fact's.")
    assert LEADS == {"err": "Error.", "lock": "Blocked.", "warn": "Attention.", "info": "Information.", "fine": "Fine."}
    fine, info, one, two = Fact("fine", "a"), Fact("info", "b"), Fact("lock", "c", "x"), Fact("err", "d", "y")
    assert ordered([fine, one, info, two]) == (one, two, fine, info)


@pytest.mark.parametrize("facts, live, nothing, words", [
    ((), False, False, ""),
    ((), False, True, "Nothing to apply: these choices are in use."),
    ((Fact("err", "a", BLOCKS),), False, False, "Cannot start yet: the name cannot be used."),
    ((Fact("err", "a", BLOCKS), Fact("lock", "b", "needs a prior")), False, False,
     "Cannot start yet: the name cannot be used, and 1 more above."),
    ((Fact("err", "a", BLOCKS), Fact("lock", "b", "needs a prior"), Fact("err", "c", "z")), False, False,
     "Cannot start yet: the name cannot be used, and 2 more above."),
    ((Fact("fine", "a"), Fact("err", "b", BLOCKS)), False, False, "Cannot start yet: the name cannot be used."),
    ((Fact("err", "a", BLOCKS),), False, True, "Cannot start yet: the name cannot be used."),
    ((Fact("err", "a", BLOCKS),), True, True, "A run is live. Only one runs at a time."),
    ((), True, False, "A run is live. Only one runs at a time."),
], ids=range(9))
def test_the_reason_beside_the_button_for_each_state(facts, live, nothing, words):
    """The table of what stands beside a greyed button, exact, a live run first and what blocks before nothing to apply.
    The live words are the ones a page says when it asks for a run while one is live."""
    assert reason_beside(facts, live=live, nothing_to_apply=nothing) == words
    assert reason_beside((), live=True, nothing_to_apply=False) == LIVE_TEXT


def test_the_reading_line_says_when_the_records_were_read():
    """The three sentences, the time from the moment handed in, and nothing for a page not read with no run live: it reads
    as it is shown."""
    moment = at(8, 9, 4)
    assert reading_words(moment, live=False) == "The saved records were read at 09:04."
    assert reading_words(moment, live=True) == "A run is live. Read at 09:04; read again when it ends."
    assert reading_words(None, live=True) == NOT_READ_YET_TEXT == "A run is live. Not read yet; read when it ends."
    assert reading_words(None, live=False) == ""


SOURCES = {"plan": ("retrain", "from plan retrain", "changed from the plan"),
           "record": ("retrain_posterior", "from record retrain_posterior", "changed from the record"),
           "run": ("retrain_3", "from run retrain_3", "changed from the run")}


def test_each_field_has_exactly_one_origin_over_every_combination():
    """Where the value came from, over every combination of what can be said of a field: where the source in use holds the
    setting, its two rows win; else a value put back and not touched since is kept from last time, whether or not it
    equals the usual; else the usual or a change from it. Every combination has one answer, and each of the table's
    seven words stands."""
    seen = set()
    for held, at_source, at_usual, kept, source in itertools.product((True, False), (True, False), (True, False),
                                                                     (True, False), SOURCES):
        name, unchanged, changed = SOURCES[source]
        words = origin_words(held=held, at_source=at_source, at_usual=at_usual, kept=kept, source=source, name=name)
        want = ((unchanged if at_source else changed) if held else "kept from last time" if kept
                else "usual" if at_usual else "changed from the usual")
        assert words == want, (held, at_source, at_usual, kept, source)
        seen.add(words)
    assert seen == {"usual", "changed from the usual", "kept from last time", *(w for s in SOURCES.values() for w in s[1:])}


@pytest.mark.parametrize("seconds, words", [
    (1, "about half a minute"), (15, "about half a minute"), (30, "about half a minute"), (44, "about half a minute"),
    (45, "about a minute"), (60, "about a minute"), (75, "about a minute and a half"), (92, "about a minute and a half"),
    (120, "about 2 minutes"), (150, "about 2 and a half minutes"), (570, "about 9 and a half minutes"),
    (585, "about 10 minutes"), (600, "about 10 minutes"), (629, "about 10 minutes"), (631, "about 11 minutes"),
    (3540, "about 59 minutes"), (3570, "about an hour"), (3599, "about an hour"), (3600, "about an hour"),
    (5400, "about an hour and a half"), (7200, "about 2 hours"), (8999, "about 2 and a half hours"),
    (9000, "about 2 and a half hours"), (90_000, "about 25 hours"), (0, ""), (-3.0, ""), (float("nan"), ""),
    (float("inf"), "")])
def test_about_words_round_a_length_as_a_person_would(seconds, words):
    """To the half minute below ten minutes, to the minute below an hour, to the half hour above, a half going up as a
    person rounds, and an hour never "60 minutes". A length that is no length says nothing."""
    assert about_words(seconds) == words
