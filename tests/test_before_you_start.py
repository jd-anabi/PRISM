"""What a step says in "Before you start", as values: each row of the step's table, its sentence word for word, and what
blocks. No widget is built here."""
import math

import pytest

from core.gui.step_facts import (LONGEST_BUILD_S, Fact, bounds_words, build_length, build_words, length_words,
                                 prior_facts, prior_in_use, reason_beside, saved_prior_words)

#: What every saved view's facts are read with but the row a test looks at: a saved prior of the model applied, built
#: on the file selected as it is now.
SAVED = dict(view="saved", record_model="NADROWSKI", model="NADROWSKI", built_on="master_tier1.txt",
             selected="master_tier1.txt")
#: What every build view's facts are read with but the row a test looks at.
BUILD = dict(view="build", rounds=50, batch=2048, stable=175000)


def _said(facts) -> list:
    return [(fact.kind, fact.sentence) for fact in facts]


# ── the saved view ──
@pytest.mark.parametrize("built_on, selected, edited, same_box, said", [
    ("master_tier1.txt", "master_tier1.txt", False, True,
     ("fine", "Built on master_tier1.txt, the bounds file selected.")),
    ("master.txt", "master_tier1.txt", False, True,
     ("warn", "Built on master.txt; the bounds file selected is master_tier1.txt. PRISM's own check compares only the "
              "model's parameters and their limits, which these two files share, so the load goes through.")),
    ("master.txt", "master_spont.txt", False, False,
     ("warn", "Built on master.txt; the bounds file selected is master_spont.txt. PRISM's own check compares only the "
              "model's parameters and their limits, which differ between these two files, so PRISM will refuse the "
              "load.")),
    ("master_tier1.txt", "master_tier1.txt", True, True,
     ("warn", "Built on master_tier1.txt as it was then; the file has been edited since.")),
    ("master_tier1.txt", "master_tier1.txt", True, False,
     ("warn", "Built on master_tier1.txt as it was then; the file has been edited since. Its parameters or their limits "
              "have changed with it, so PRISM will refuse the load.")),
    (None, "master_tier1.txt", False, True, ("info", "Built on limits typed by hand.")),
    (None, "master_tier1.txt", False, False,
     ("warn", "Built on limits typed by hand. The bounds file selected differs from them in the model's parameters or "
              "their limits, so PRISM will refuse the load.")),
], ids=["the file selected", "another file that shares the box", "another file that does not", "the file edited",
        "the file edited, its box with it", "limits typed by hand", "limits typed by hand, another box"])
def test_a_saved_priors_bounds_file_against_the_one_selected(built_on, selected, edited, same_box, said):
    """The bounds file a saved prior was built on, said against the one selected, never blocking: the engine is the one
    that refuses a load, and the line says when it will."""
    facts = prior_facts(**{**SAVED, "built_on": built_on, "selected": selected, "edited": edited,
                           "same_box": same_box})
    assert _said(facts) == [said]
    assert not any(fact.blocks for fact in facts)


@pytest.mark.parametrize("batches, sentence", [
    ([], "No saved simulation set was made with this prior."),
    ([3989], "1 saved simulation set was made with this prior; it holds 3,989 batches."),
    ([4, 10000], "2 saved simulation sets were made with this prior; the largest holds 10,000 batches."),
])
def test_the_sets_made_with_a_saved_prior(batches, sentence):
    """How many saved simulation sets were made with the prior selected, and the largest's batches; nothing at all when
    they were not read."""
    assert _said(prior_facts(**SAVED, sets_of_prior=batches))[1:] == [("info", sentence)]
    assert len(prior_facts(**SAVED, sets_of_prior=None)) == 1
    unread = prior_facts(**SAVED, bounds_error="Choose a bounds file.", bounds_blocks="no bounds file is chosen",
                         sets_of_prior=batches)
    assert [fact.kind for fact in unread] == ["err", "info"], "a file not chosen was compared with the record"


def test_a_prior_of_another_model_blocks_and_says_which():
    """A prior made for another model greys the load with that reason, its model by its name; what it was built on is
    then no matter."""
    facts = prior_facts(**{**SAVED, "record_model": "BP"}, sets_of_prior=[5])
    assert _said(facts) == [("err", "This prior was made for the model BP.")]
    assert reason_beside(facts, live=False, nothing_to_apply=False) == "Cannot start yet: this prior is for another model."


# ── the build view ──
@pytest.mark.parametrize("steps, sentence", [
    (None, "Tests 102,400 candidates (50 rounds of 2,048), then walks outward until 175,000 stable sets are collected."),
    (40000, "Tests 102,400 candidates (50 rounds of 2,048), each integrated over 40,000 steps, then walks outward until "
            "175,000 stable sets are collected."),
])
def test_what_a_build_costs_and_when_the_steps_are_said(steps, sentence):
    """What a build costs, from the three settings it reads, with the steps each candidate is integrated over only when
    a configuration is there to say them; never a "?" in their place, and nothing at all while a setting cannot be
    read."""
    assert _said(prior_facts(**BUILD, steps=steps))[0] == ("info", sentence)
    for missing in ("rounds", "batch", "stable"):
        assert not any(fact.sentence.startswith("Tests") for fact in prior_facts(**{**BUILD, missing: None}))


@pytest.mark.parametrize("took, sentence, words", [
    (None, "No prior has been built on this computer yet, so the time is not known.", "Build a new prior"),
    ((92.0, "retrain_prior"), "About a minute and a half: the last build on this computer took 1 min 32 s.",
     "Build a new prior (about a minute and a half)"),
    ((45.0, ""), "About a minute: the last build on this computer took 45 s.", "Build a new prior (about a minute)"),
    ((3725.0, ""), "About an hour: the last build on this computer took 1 h 2 min.", "Build a new prior (about an hour)"),
    ((-60.0, ""), "The last build on this computer has no length that can be read, so the time is not known.",
     "Build a new prior"),
    ((0.0, ""), "The last build on this computer has no length that can be read, so the time is not known.",
     "Build a new prior"),
    ((LONGEST_BUILD_S + 1.0, ""),
     "The last build on this computer has no length that can be read, so the time is not known.", "Build a new prior"),
    ((math.nan, ""), "The last build on this computer has no length that can be read, so the time is not known.",
     "Build a new prior"),
], ids=["none", "92 s", "45 s", "an hour", "a clock set back", "nothing", "past a week", "not a number"])
def test_how_long_a_build_takes(took, sentence, words):
    """How long a build takes is what the last build on this computer took, rounded as a person says it in the button's
    bracket and to the second in the fact. A length a clock set wrong gave, below nothing or past a week, is no length:
    no bracket, and never a negative time."""
    facts = prior_facts(**BUILD, took=took)
    assert ("info", sentence) in _said(facts)
    assert build_words(took) == words


def test_a_length_is_said_to_the_second():
    assert [length_words(seconds) for seconds in (0, 45, 60, 92, 600, 3600, 3725, 7200.4)] == [
        "0 s", "45 s", "1 min", "1 min 32 s", "10 min", "1 h", "1 h 2 min", "2 h"]
    assert [build_length(took) for took in (None, (), ("x",), (5.0, ""), (-5.0, ""), (math.inf, ""))] == [
        None, None, None, 5.0, None, None]


@pytest.mark.parametrize("sets, sentence", [
    (0, None),
    (1, "A new prior cannot continue a saved simulation set: the saved set for this model was made with another prior."),
    (2, "A new prior cannot continue a saved simulation set: the 2 saved sets for this model were made with other "
        "priors."),
])
def test_a_new_prior_cannot_continue_a_saved_set(sets, sentence):
    said = [fact.sentence for fact in prior_facts(**BUILD, sets_for_model=sets) if "cannot continue" in fact.sentence]
    assert said == ([] if sentence is None else [sentence])


# ── what blocks ──
def test_what_stops_a_load_or_a_build_comes_first_with_its_words():
    """No model applied blocks, as the step's need, with its own words; no bounds file chosen, a file that cannot be
    read and no saved prior chosen are errors; each comes before every other fact and names its words beside the
    button."""
    facts = prior_facts(view="saved", needs=True, bounds_error="Choose a bounds file.",
                        bounds_blocks="no bounds file is chosen", prior_error="Choose a saved prior.",
                        prior_blocks="no saved prior is chosen", sets_of_prior=[2])
    assert _said(facts)[:3] == [("lock", "Needs a model: apply one in step 1, Model and options."),
                                ("err", "Choose a bounds file."), ("err", "Choose a saved prior.")]
    assert reason_beside(facts, live=False, nothing_to_apply=False) == "Cannot start yet: needs a model, and 2 more above."
    build = prior_facts(**BUILD, attention=["The units say so."], name_facts=[Fact("err", "Type a name.",
                                                                                     "a name is needed")])
    assert _said(build)[0] == ("err", "Type a name.") and ("warn", "The units say so.") in _said(build)


# ── what is said of a file and of a prior ──
@pytest.mark.parametrize("mode, reads", [
    ("chi", "a frequency response"), ("forced", "a spontaneous recording and one driven recording"),
    ("spontaneous", "a spontaneous recording only")])
def test_what_a_bounds_file_is_is_said_in_one_line(mode, reads):
    assert bounds_words(["k", "lam"], ["x_scale", "t_scale", "T"], mode == "forced", mode) == (
        "5 parameters: 2 of the model (k, lam) and 3 scales (x_scale, t_scale, T). "
        + ("It has a forcing section. " if mode == "forced" else "It has no forcing section. ")
        + f"With the kind of observation applied in step 1, a posterior trained on it reads: {reads}.")
    assert bounds_words(["k"], ["x_scale"], True, "forced").startswith("2 parameters: 1 of the model (k) and 1 scale (")


def test_a_bounds_file_with_no_scale_names_none():
    """A bounds file with no section of scales, as each of Bp's, says its parameters are all the model's, and names no
    scale."""
    from core.config import BOUNDS_PATH
    from core.Helpers import file_manager
    params, scales, forcing, _ = file_manager.parse_bounds_file(str(BOUNDS_PATH / "bp" / "cell_1.txt"))
    assert (len(params), len(scales), len(forcing)) == (17, 0, 0)
    said = bounds_words(list(params), list(scales), bool(forcing), "spontaneous")
    assert said == (f"17 parameters, all of the model ({', '.join(params)}). It has no forcing section. With the kind "
                    "of observation applied in step 1, a posterior trained on it reads: a spontaneous recording only.")
    assert bounds_words(["k"], ["x_scale"], False, "chi").startswith(
        "2 parameters: 1 of the model (k) and 1 scale (x_scale).")


def test_a_prior_is_said_by_what_it_was_built_on():
    assert prior_in_use("retrain_prior", "master_tier1.txt", 13) == (
        "retrain_prior", "Built on master_tier1.txt, 13 parameters. Saved as a record.")
    assert prior_in_use("Unnamed prior, 1 Oct 2026, 22:29", None, 12)[1] == (
        "Built on limits typed by hand, 12 parameters. Saved as a record.")
    assert prior_in_use("", "master.txt", 3) == ("", "")
    assert saved_prior_words("1 Oct 2026, 22:29", "NADROWSKI", 13, "master_tier1.txt") == (
        "Made 1 Oct 2026, 22:29, for Nadrowski: 13 parameters, built on master_tier1.txt.")
