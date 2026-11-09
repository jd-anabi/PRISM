"""What a step says in "Before you start", as values: each row of the step's table, its sentence word for word, and what
blocks; and what it says of the plan in use, read off stand-ins of the steps. No widget is built here."""
import math

import pytest

from core.gui.step_facts import (LONGEST_BUILD_S, Fact, bounds_words, budget_words, build_length, build_words,
                                 how_long_words, length_words, pending_words, posterior_in_use, prior_facts,
                                 prior_in_use, reason_beside, saved_prior_words, training_facts, uses_words)

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
    assert reason_beside(facts, live=False, nothing_to_apply=False) == "Cannot start: this prior is for another model."


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


# ── the third step, Training ──
def _reading(**changed):
    """A training's reading of the saved simulation sets, as the engine hands one: no set, the cadence on."""
    from core.orchestrator import TrainingReading
    held = dict(checkpoint="new", batches_done=0, siblings="", sibling_diffs=(), cadence=50, checkpoint_error=None,
                width=2048)
    held.update(changed)
    return TrainingReading(**held)


def _set(batches, *fields, complete=True, name="eda5c2a1e468"):
    """A saved simulation set beside the one a training would make, with what it differs in, as the engine says it:
    each of ``fields`` a ``(field, mine, theirs)``."""
    from core.SBI.training_checkpoint import SiblingDifference
    return SiblingDifference(name, batches, complete,
                             tuple({"field": key, "mine": mine, "theirs": theirs} for key, mine, theirs in fields))


#: What every training's facts are read with but the row a test looks at.
TRAINING = dict(reading=_reading(), batches=10000)
#: The two facts every training whose simulations are kept says of its fit.
FIT_KEPT = [("info", "The fit is saved after every epoch, so a stop or a failure keeps it."),
            ("info", "A resume first reads the simulation set back, about four minutes at full size.")]


def test_what_the_run_uses_is_named():
    """The model by its name, the bounds file (or the limits typed by hand), the prior by its label, the kind of
    observation with the frequency response's slots, and the rotation, each as the configuration in use holds it."""
    assert uses_words(model="NADROWSKI", bounds="master_tier1.txt", prior="retrain_prior", kind="chi", slots=12,
                      rotate=True) == ("Model Nadrowski; bounds file master_tier1.txt; prior retrain_prior; frequency "
                                       "response with 12 slots; rotation on.")
    assert uses_words(model="HOPF", bounds="", prior="Unnamed prior, 1 Oct 2026, 22:29", kind="forced", slots=12,
                      rotate=False) == ("Model Hopf; bounds file limits typed by hand; prior Unnamed prior, 1 Oct 2026, "
                                        "22:29; a spontaneous recording and one driven recording; rotation off.")
    assert uses_words(model="NADROWSKI", bounds="sub/master.txt", prior="p", kind="spontaneous", slots=None,
                      rotate=None).endswith("; a spontaneous recording only; rotation unknown.")
    facts = training_facts(uses="Model Nadrowski; one more.", **TRAINING)
    assert facts[0] == Fact("info", "Model Nadrowski; one more.")


@pytest.mark.parametrize("kind, slots, each", [
    ("chi", 12, "one spontaneous recording and 2 to 12 driven recordings"),
    ("chi", 6, "one spontaneous recording and 2 to 6 driven recordings"),
    ("forced", 12, "one spontaneous recording and one driven recording"),
    ("spontaneous", 12, "one spontaneous recording"),
], ids=["frequency response", "frequency response of six slots", "a driven recording", "spontaneous"])
def test_the_simulations_are_counted_for_each_kind_of_observation(kind, slots, each):
    """The batches typed times the simulations of each, which the reading's width gives, grouped by thousands; and what
    each simulation simulates, by the kind of observation and, for the frequency response, its slots."""
    assert budget_words(10000, 2048, kind, slots) == (
        f"10,000 batches of 2,048 make 20,480,000 simulations. Each simulates {each}.")
    assert budget_words(None, 2048, kind, slots) == "" and budget_words(10000, None, kind, slots) == ""
    assert budget_words(0, 2048, kind, slots) == "" and budget_words(True, 2048, kind, slots) == ""


@pytest.mark.parametrize("reading, differences, said", [
    (_reading(checkpoint="resume_partial", batches_done=3989), None,
     ("fine", "Continues the saved simulation set: 3,989 of 10,000 batches are done.")),
    (_reading(checkpoint="resume_complete", batches_done=10000), None,
     ("fine", "No simulations this time: the saved 20,480,000 are used.")),
    (_reading(checkpoint="resume_complete", batches_done=10000, width=None), None,
     ("fine", "No simulations this time: the saved 10,000 batches are used.")),
    (_reading(), None, ("info", "Starts a new simulation set; none is saved for this model yet.")),
    (_reading(checkpoint="new_with_siblings", sibling_diffs=(_set(4000, ("n_runs", 10000, 4000)),)),
     {"eda5c2a1e468": ("Batches: 10,000 here, 4,000 in this set",)},
     ("warn", "Starts a new simulation set. 1 saved set for this model does not match these settings: The set of 4,000 "
              "batches (complete) differs in Batches: 10,000 here, 4,000 in this set.")),
    (_reading(checkpoint="new_with_siblings", sibling_diffs=(
        _set(4000, ("n_runs", 10000, 4000), ("reparam_rotate", True, False), ("nd_lows", [0.1], [0.2])),
        _set(12, ("run_size", 2048, 1024), complete=False, name="b"))),
     {"eda5c2a1e468": ("Batches: 10,000 here, 4,000 in this set", "Bounds file: lower limit of k: 0.1 here, 0.2 in "
                       "this "
                       "set", "Rotation: on here, off in this set")},
     ("warn", "Starts a new simulation set. 2 saved sets for this model do not match these settings: The set of 4,000 "
              "batches (complete) differs in Batches: 10,000 here, 4,000 in this set; Bounds file: lower limit of k: "
              "0.1 "
              "here, 0.2 in this set; Rotation: on here, off in this set. The set of 12 batches (partial) differs in "
              "run_size: 2048 here, 1024 in this set.")),
    (_reading(checkpoint="new_with_siblings", sibling_diffs=(_set(4000),)),
     {"eda5c2a1e468": ("No setting differs: its folder was renamed by hand, so a training does not find it.",)},
     ("warn", "Starts a new simulation set. 1 saved set for this model does not match these settings: The set of 4,000 "
              "batches (complete): No setting differs: its folder was renamed by hand, so a training does not find "
              "it.")),
    (_reading(checkpoint="new_with_siblings", sibling_diffs=(_set(4000),)), None,
     ("warn", "Starts a new simulation set. 1 saved set for this model does not match these settings: The set of 4,000 "
              "batches (complete): No setting differs: its folder was renamed by hand, so a training does not find "
              "it.")),
    (_reading(checkpoint="new_with_siblings", sibling_diffs=(_set(4000, ("model", "NADROWSKI", "HOPF")),)), None,
     ("info", "Starts a new simulation set; none is saved for this model yet.")),
], ids=["continues a set", "a complete set", "a complete set of no known width", "none saved", "one set that differs",
        "two sets, the second unnamed by the page", "a set that differs in nothing",
        "a set that differs in nothing, no line named by the page", "a set of another model"])
def test_continue_or_new_says_each_state(reading, differences, said):
    """What the training does with the saved simulation sets, as the reading at this budget found them, word for word: it
    continues the set of these settings, uses a complete one, or starts a new one beside none of this model, or beside
    every set of this model that does not match, each with everything it differs in, as the page names it by the set's
    name, and by the identity's own names for a set it named nothing of. A set that differs in nothing says what the
    page says of it. A set of another model is not counted. None of these blocks."""
    facts = training_facts(**{**TRAINING, "reading": reading, "differences": differences})
    assert _said(facts) == [said, *FIT_KEPT]
    assert not any(fact.blocks for fact in facts)


@pytest.mark.parametrize("never, own, sets, text, items, other", [
    (False, "", [("Simulation set for retrain_prior (2 Oct 2026, 09:10; 10,000 batches, complete)",
                  ("Batches: 4,000 here, 10,000 in this set", "Rotation: on here, off in this set")),
                 ("Simulation set for a prior that is no longer here (12 batches, partial)",
                  ("Prior: retrain_prior here, a prior no longer in Records in this set",))],
     "This training will simulate 4,000 batches from zero. Saved simulation sets for this model exist, and none "
     "matches "
     "these settings.",
     ("Simulation set for retrain_prior (2 Oct 2026, 09:10; 10,000 batches, complete) differs in:",
      "Batches: 4,000 here, 10,000 in this set", "Rotation: on here, off in this set",
      "Simulation set for a prior that is no longer here (12 batches, partial) differs in:",
      "Prior: retrain_prior here, a prior no longer in Records in this set"),
     "Start training on a new simulation set"),
    (True, "Simulation set for retrain_prior (4,000 batches, partial)",
     [("Simulation set for retrain_prior (2 Oct 2026, 09:10; 10,000 batches, complete)",
       ("Batches: 4,000 here, 10,000 in this set",))],
     "This training will simulate 4,000 batches from zero. Saved simulation sets for this model exist, and none "
     "will be "
     "continued.",
     ("Simulation set for retrain_prior (4,000 batches, partial):",
      "No setting differs. It is not continued because the simulations of this training are not saved.",
      "Simulation set for retrain_prior (2 Oct 2026, 09:10; 10,000 batches, complete) differs in:",
      "Batches: 4,000 here, 10,000 in this set"),
     "Start training without saving the simulations"),
], ids=["the sets saved", "with Never, the set it will not continue first"])
def test_the_question_before_simulating_from_zero_names_each_set_and_what_it_differs_in(never, own, sets, text, items,
                                                                                         other):
    """The question asked before a training simulates from zero while saved sets of its model exist, as values: its
    title, the batches it will simulate, each set by its label with when it was made, its batches and whether it is
    whole, then what it differs in line by line; with "Never" the set these very settings would continue heads the list,
    the sentence says none will be continued and the answer says the simulations are not saved."""
    from core.gui.step_facts import new_set_question, set_words
    assert set_words("Simulation set for retrain_prior", "2 Oct 2026, 09:10", 10000, True) == (
        "Simulation set for retrain_prior (2 Oct 2026, 09:10; 10,000 batches, complete)")
    assert set_words("Simulation set for retrain_prior", "", 12, False) == (
        "Simulation set for retrain_prior (12 batches, partial)")
    assert new_set_question(batches=4000, never=never, own=own, sets=sets) == (
        "Start a new simulation set?", text, items, "To continue one of them, cancel and set these back.", other)


# ── the third step's load view ──
#: What every load's facts are read with but the row a test looks at: a posterior of the model in use, selected, trained
#: with the prior in use.
LOADED = dict(record_model="NADROWSKI", model="NADROWSKI", trained_with="p1", trained_with_label="retrain_prior",
              in_use="p1", in_use_label="retrain_prior")


@pytest.mark.parametrize("changed, said", [
    ({}, [("fine", "Trained with retrain_prior, the prior in use.")]),
    ({"trained_with": "p0", "trained_with_label": "old_prior"},
     [("warn", "Trained with old_prior; the prior in use is retrain_prior. Calibration and a narrowing round will "
               "refuse "
               "this pair. Load old_prior in step 2 first.")]),
    ({"trained_with": "p0", "trained_with_label": ""},
     [("warn", "Trained with a prior no longer in Records; the prior in use is retrain_prior. Calibration and a "
               "narrowing "
               "round will refuse this pair.")]),
    ({"trained_with": None}, []),
    ({"needs_prior": True}, [("lock", "Needs a prior: build or load one in step 2, Prior.")]),
    ({"chosen": False}, [("err", "Choose a saved posterior.")]),
    ({"chosen": False, "saved": False},
     [("err", "No posterior is saved yet: start a training with 'Set up a training instead…'.")]),
    ({"error": "The posterior cannot be read: StoreError: gone."}, [("err", "The posterior cannot be read: StoreError: "
                                                                            "gone.")]),
    ({"record_model": "HOPF"}, [("err", "This posterior was made for the model Hopf.")]),
    ({"others_pending": ["Step 2, Prior, has changes that are not applied. This run uses what is in use: "
                         "retrain_prior."]},
     [("fine", "Trained with retrain_prior, the prior in use."),
      ("warn", "Step 2, Prior, has changes that are not applied. This run uses what is in use: retrain_prior.")]),
], ids=["the prior in use", "another prior", "a prior no longer in Records", "a record that names no prior",
        "no prior in use", "nothing chosen", "nothing saved", "a record that cannot be read", "another model",
        "another step's change"])
def test_the_load_view_compares_the_prior(changed, said):
    """The facts of the load view: the prior the posterior selected was trained with, against the prior in use, said and
    never blocking; a posterior trained with another prior names it and says to load it in step 2; with no prior in use,
    a lock and no comparison; nothing chosen, nothing saved, a record that cannot be read, or one of another model,
    each an error with its words beside the button. Another step's change not applied is said."""
    from core.gui.step_facts import load_facts
    facts = load_facts(**{**LOADED, **changed})
    assert _said(facts) == said
    blocks = {"needs_prior": "needs a prior", "chosen": "no saved posterior is chosen",
              "error": "the posterior cannot be read", "record_model": "this posterior is for another model"}
    held = [fact.blocks for fact in facts if fact.blocks]
    named = next((key for key in blocks if key in changed), None)
    if named == "chosen" and changed.get("saved") is False:
        assert held == ["no posterior is saved yet"]
    else:
        assert held == ([blocks[named]] if named else []), held


@pytest.mark.parametrize("given, said", [
    (dict(made="3 Oct 2026, 03:12", model="NADROWSKI", kind="chi", network=(256, 10), simulations=20480000,
          prior="retrain_prior", bounds="master.txt"),
     "Made 3 Oct 2026, 03:12, for Nadrowski: a frequency response; network 256 by 10; 20,480,000 simulations; trained "
     "with retrain_prior, on master.txt."),
    (dict(made="3 Oct 2026, 03:12", model="NADROWSKI", kind="forced", network=(128, 8), simulations=320000,
          prior="a prior no longer in Records", bounds=None, narrowed=True, observation="cell_a"),
     "Made 3 Oct 2026, 03:12, for Nadrowski: a spontaneous recording and one driven recording; network 128 by 8; "
     "320,000 simulations; trained with a prior no longer in Records, on limits typed by hand. From a narrowing round "
     "around the observation cell_a."),
    (dict(made="3 Oct 2026, 03:12", model="NADROWSKI", kind="spontaneous", narrowed=True),
     "Made 3 Oct 2026, 03:12, for Nadrowski: a spontaneous recording only. From a narrowing round around an "
     "observation "
     "no longer in Records."),
    (dict(made="3 Oct 2026, 03:12", model="NADROWSKI", kind=None, network=(None, 8), simulations=True),
     "Made 3 Oct 2026, 03:12, for Nadrowski."),
    (dict(made="", model="NADROWSKI", kind="spontaneous", network=(128, 8)),
     "For Nadrowski: a spontaneous recording only; network 128 by 8."),
    (dict(made="", model="NADROWSKI", kind=None), "For Nadrowski."),
], ids=["a training's", "a narrowing round's, on limits typed by hand", "a narrowing round's that says little",
        "a record that says nothing more", "a record whose moment cannot be read",
        "a record that says only its model"])
def test_a_saved_posteriors_record_is_said_under_its_list(given, said):
    """Under the list of saved posteriors, from the record selected: when it was made and for which model, the kind of
    observation it reads, its network, its simulations, the prior it was trained with and the bounds file it was built
    on, and, from a narrowing round, the observation it was narrowed around; what its record does not say is left
    out."""
    from core.gui.step_facts import saved_posterior_words
    assert saved_posterior_words(**given) == said


def test_a_narrowed_posterior_in_use_says_it_came_from_a_narrowing_round():
    """A posterior from a narrowing round says so in "In use", whatever else its record says."""
    training = {"epochs_trained": 131, "n_runs": 10000, "run_size": 2048, "hidden_features": 256, "num_transforms": 10}
    assert posterior_in_use("retrain_round1", training, {}, "retrain_prior", narrowed=True) == (
        "retrain_round1", "131 epochs, network 256 by 10, on 20,480,000 simulations, prior retrain_prior. From a "
                          "narrowing round. Saved as a record.")
    assert posterior_in_use("retrain_round1", {}, {}, "", narrowed=True) == (
        "retrain_round1", "From a narrowing round. Saved as a record.")
    assert posterior_in_use("retrain", training, {}, "retrain_prior")[1].endswith(
        "prior retrain_prior. Saved as a record.")


def test_the_narrowed_question_says_what_loading_it_means():
    """Before a posterior from a narrowing round is loaded, its question says around which observation it was narrowed,
    what it is valid for and what loading it means, in its own words; the answer that loads it is the page's button."""
    from core.gui.step_facts import narrowed_question
    assert narrowed_question("retrain_round1", "cell_a") == (
        "Load a posterior from a narrowing round?",
        "retrain_round1 was narrowed around the observation cell_a. It is valid only near that observation; elsewhere "
        "it "
        "has seen no training simulations and gives confident answers that are wrong.",
        "If you load it: calibration runs on the narrowed prior; inference on any other observation asks first; a "
        "further narrowing round can only be drawn around the same observation.",
        "Load this posterior")
    assert narrowed_question("Unnamed posterior, 3 Oct 2026, 03:12", "")[1].startswith(
        "Unnamed posterior, 3 Oct 2026, 03:12 was narrowed around an observation no longer in Records. It is valid")


@pytest.mark.parametrize("state, on_purpose, reason, how", [
    ("kept when stopped", True, "cancelled", "kept when stopped"),
    ("stopped unexpectedly", False, "RuntimeError: the card ran out of memory",
     "stopped unexpectedly after the error 'RuntimeError: the card ran out of memory'"),
    ("stopped unexpectedly", None, None, "stopped unexpectedly"),
    ("stopped unexpectedly", False, "  ", "stopped unexpectedly"),
    (None, None, None, "stopped unexpectedly"),
], ids=["kept on purpose", "stopped by an error", "a stop never recorded", "an error with no words", "no state"])
def test_a_kept_training_is_named_with_how_it_stopped_and_when(state, on_purpose, reason, how):
    """A kept training as the choice before a new training names it: its label, how it stopped in the listing's words
    and, when an error stopped it, the error as its record words it, never the record's raw word for a stop on purpose;
    the epoch it was saved after, and when, in the window's words."""
    from core.gui.step_facts import kept_words, new_training_question, stopped_words
    assert stopped_words(state, on_purpose, reason) == how
    assert kept_words("retrain_2", how, 41, "3 Oct 2026, 03:12") == f"retrain_2: {how}, after epoch 41, saved 3 Oct " \
                                                                       f"2026, 03:12"
    assert kept_words("retrain_2", how, None, "") == f"retrain_2: {how}"
    one = new_training_question(["retrain_2: kept when stopped, after epoch 41, saved 3 Oct 2026, 03:12"])
    assert one == ("Start a new training?", "1 kept training matches these settings and can be resumed:",
                   ("retrain_2: kept when stopped, after epoch 41, saved 3 Oct 2026, 03:12",),
                   "A new training leaves it as it is. To resume it, cancel and press Resume beside it on this page.",
                   "Start a new training")
    two = new_training_question(["a: kept when stopped", "b: stopped unexpectedly"])
    assert two[1:4] == ("2 kept trainings match these settings and can be resumed:",
                        ("a: kept when stopped", "b: stopped unexpectedly"),
                        "A new training leaves them as they are. To resume one, cancel and press Resume beside it on "
                        "this page.")


def test_with_never_nothing_saved_is_continued_and_nothing_is_read():
    """With "Never" the simulations are not saved, so no set is continued, whatever a reading says; the reading of the
    built-in cadence is not this training's."""
    for reading in (None, _reading(checkpoint="resume_partial", batches_done=3989)):
        facts = training_facts(reading=reading, never=True, batches=10000)
        assert _said(facts)[0] == (
            "warn", "The simulations of this training are not saved, so no saved simulation set is continued.")


def test_the_fit_is_said_to_be_saved_and_a_resume_to_read_back():
    """While the simulation set is kept, the fit is said to be saved after every epoch, and a resume to read the set back
    first; no fact says a fit cannot be resumed."""
    facts = training_facts(**TRAINING)
    assert _said(facts)[-2:] == FIT_KEPT
    assert not any("cannot be resumed" in fact.sentence for fact in facts)


def test_with_never_the_fit_is_said_to_be_lost():
    """With "Never" no set is kept, so a stop or a failure during the fit loses it, said as an attention and blocking
    nothing; nothing says the fit is saved."""
    facts = training_facts(reading=_reading(checkpoint="off", cadence=0), never=True, batches=10000)
    assert _said(facts) == [
        ("warn", "The simulations of this training are not saved, so no saved simulation set is continued."),
        ("warn", "A training whose simulation set is not kept cannot be resumed: a stop or a failure during the fit "
                 "loses it.")]
    assert not any(fact.blocks for fact in facts)


#: How the run history says the pace of an epoch.
EPOCH_PACE = (3780.0, "the last training of this size, retrain, took about 63 minutes per epoch")


@pytest.mark.parametrize("per_batch, per_epoch, batches, epochs, said", [
    ((9.7, "the last training of this size, retrain, took 9.7 s per batch"), EPOCH_PACE, 10000, 130,
     "Simulation: about 27 hours: the last training of this size, retrain, took 9.7 s per batch. Fit: about 136 and a "
     "half hours for 130 epochs: the last training of this size, retrain, took about 63 minutes per epoch."),
    ((9.7, "the last training of this size took 9.7 s per batch"), EPOCH_PACE, 6011, None,
     "Simulation: about 16 hours: the last training of this size took 9.7 s per batch. Fit: the last training of this "
     "size, retrain, took about 63 minutes per epoch."),
    (None, (60.0, "it took about a minute per epoch"), 10000, 131,
     "Fit: about 2 hours for 131 epochs: it took about a minute per epoch."),
    ((9.7, "words"), None, 0, 131, "No time estimate: no training of this size has run on this computer, and the "
                                   "timing of a short fit has not been run."),
    (None, None, 10000, 131, "No time estimate: no training of this size has run on this computer, and the timing of "
                             "a short fit has not been run."),
    ((float("nan"), "words"), (-1.0, "x"), 10000, 131, "No time estimate: no training of this size has run on this "
                                                       "computer, and the timing of a short fit has not been run."),
], ids=["both paces and the epochs", "the epochs unknown", "the fit alone", "nothing left to simulate",
        "no pace", "paces that are no length"])
def test_the_fits_length_has_three_sources_and_never_limits_it(per_batch, per_epoch, batches, epochs, said):
    """How long the training takes, from the run history's paces of this size, each with the history's own words of
    where it comes from: the batches still to simulate, and the epochs the fit is expected to run, typed, or the last
    finished training's, or not known, when the pace of an epoch is said alone. A pace that is no length says nothing.
    The fact never blocks."""
    assert how_long_words(per_batch=per_batch, per_epoch=per_epoch, batches=batches, epochs=epochs) == said
    facts = training_facts(how_long=said, **TRAINING)
    assert ("info", said) in _said(facts) and not any(fact.blocks for fact in facts)


def test_the_memory_lines_say_what_is_in_force_and_what_the_last_run_measured():
    """The memory limit as the run's line says it, where it was set, and the last run's sentence after it, led by
    "Information."; the build of a prior says no memory line."""
    from core.gui.memory_limit import InForce, limit_words
    from core.gui.run import ends
    for force, words in ((InForce(None, "none", ""), "No memory limit is set on this computer."),
                         (InForce(6.0, "setting", ""),
                          "Memory limit: 6 GiB for each batch of simulations (set in Settings, This computer)."),
                         (InForce(4.5, "environment", "4.5"),
                          "Memory limit: 4.5 GiB for each batch of simulations (set by PRISM_VRAM_CEILING_GIB).")):
        assert limit_words(force) == words
    last = {"name": "retrain", "memory_limit": {"gib": 6.0, "source": "setting", "computer": "here"},
            "health": {"memory_peak": 5.4}}
    memory = f"{limit_words(InForce(6.0, 'setting', ''))} {ends.last_run_sentence(last)}"
    facts = training_facts(memory=memory, **TRAINING)
    assert ("info", memory) in _said(facts)
    assert memory.endswith("The last run on this computer (retrain) ran with a limit of 6 GiB. Its largest batch held "
                           "5.4 GiB.")
    assert not any("Memory limit" in fact.sentence for fact in prior_facts(**BUILD))


def test_a_choice_not_applied_elsewhere_is_said_and_does_not_block():
    """A step whose page shows a change it has not applied is named, with what this run uses of it, as an attention: the
    run uses what is in use."""
    said = pending_words(2, "Prior", "retrain_prior")
    assert said == "Step 2, Prior, has changes that are not applied. This run uses what is in use: retrain_prior."
    facts = training_facts(others_pending=[said, pending_words(1, "Model and options", "Nadrowski")], **TRAINING)
    assert _said(facts)[-2:] == [
        ("warn", said),
        ("warn", "Step 1, Model and options, has changes that are not applied. This run uses what is in use: Nadrowski.")]
    assert not any(fact.blocks for fact in facts)
    model = pending_words(1, "Model and options", "Nadrowski")
    for view in (BUILD, SAVED):                      # a prior's build and load read the model step 1 holds
        facts = prior_facts(others_pending=[model], **view)
        assert _said(facts)[-1] == ("warn", model) and not any(fact.blocks for fact in facts), _said(facts)


@pytest.mark.parametrize("state", ["new", "new_with_siblings"])
def test_only_continue_with_nothing_to_continue_blocks_with_its_reason(state):
    """"Only continue" with no set of these settings to continue blocks, with its words beside the button; with a set to
    continue it blocks nothing."""
    facts = training_facts(reading=_reading(checkpoint=state), resume="require", batches=10000)
    assert facts[0] == Fact("err", "There is no saved simulation set to continue.", "there is nothing to continue")
    assert reason_beside(facts, live=False, nothing_to_apply=False) == "Cannot start: there is nothing to continue."
    for done in ("resume_partial", "resume_complete"):
        assert not any(fact.blocks for fact in training_facts(reading=_reading(checkpoint=done, batches_done=4),
                                                              resume="require", batches=10000))


@pytest.mark.parametrize("state", ["resume_partial", "resume_complete"])
def test_only_start_new_with_a_set_to_continue_blocks_with_its_reason(state):
    """"Only start new" with a set of these settings that would be continued blocks, with its words; with none it blocks
    nothing, and neither choice blocks while the simulations are not saved or the sets could not be read."""
    facts = training_facts(reading=_reading(checkpoint=state, batches_done=4), resume="never", batches=10000)
    assert facts[0] == Fact("err", "A saved simulation set would be continued.", "a saved simulation set would be "
                                                                                 "continued")
    assert not any(fact.blocks for fact in training_facts(reading=_reading(), resume="never", batches=10000))
    for resume in ("require", "never"):
        assert not any(fact.blocks for fact in training_facts(reading=_reading(checkpoint=state, batches_done=4),
                                                              resume=resume, never=True, batches=10000))
        assert not any(fact.blocks for fact in training_facts(
            reading=_reading(checkpoint_error="OSError: the disk is gone"), resume=resume, batches=10000))


def test_a_failed_reading_is_said_and_does_not_block():
    """A reading that could not read the saved sets says so, with the error, as an attention; nothing else is said of the
    sets, and the training is not blocked."""
    facts = training_facts(reading=_reading(checkpoint_error="OSError: the disk is gone."), batches=10000)
    assert _said(facts) == [("warn", "The saved simulation sets could not be read: OSError: the disk is gone."),
                            *FIT_KEPT]


def test_needs_a_prior_blocks_first():
    """With no prior in use nothing can be trained: blocked, first, with its words beside the button."""
    facts = training_facts(needs_prior=True, name_facts=[Fact("err", "Type a name.", "a name is needed")])
    assert facts[0] == Fact("lock", "Needs a prior: build or load one in step 2, Prior.", "needs a prior")
    assert reason_beside(facts, live=False, nothing_to_apply=False) == (
        "Cannot start yet: needs a prior, and 1 more above.")


def test_the_facts_function_never_raises_on_what_it_cannot_read():
    """What a reading or a session holds may be anything: the facts are said of what can be read, and nothing raises."""
    from types import SimpleNamespace
    odd = (None, object(), SimpleNamespace(), SimpleNamespace(checkpoint="resume_partial", batches_done="many"),
           SimpleNamespace(checkpoint="resume_partial", batches_done=236),
           SimpleNamespace(checkpoint="resume_complete", batches_done=4, width="wide"),
           SimpleNamespace(checkpoint="new_with_siblings", sibling_diffs=object()),
           SimpleNamespace(checkpoint="new_with_siblings", sibling_diffs=(object(), SimpleNamespace(fields=(1, None)))),
           SimpleNamespace(checkpoint="new", sibling_diffs=(SimpleNamespace(batches=True, fields=({"field": None},)),)))
    for reading in odd:
        for batches in (None, 10000, "ten", -3, float("nan")):
            for resume in ("auto", "require", "never", None):
                training_facts(reading=reading, batches=batches, resume=resume)
    assert how_long_words(per_batch=("fast", "words"), per_epoch=object(), batches="all", epochs=object())
    assert budget_words("many", object(), None, "twelve") == ""
    assert uses_words(model=object(), bounds=None, prior="", kind=None, slots=object(), rotate="yes")
    assert posterior_in_use("retrain", "not a dict", None, "") == ("retrain", "Saved as a record.")


@pytest.mark.parametrize("training, config, said", [
    ({"epochs_trained": 131, "n_runs": 10000, "run_size": 2048, "hidden_features": 256, "num_transforms": 10},
     {"max_num_epochs": 2147483647},
     "131 epochs, network 256 by 10, on 20,480,000 simulations, prior retrain_prior. Saved as a record."),
    ({"epochs_trained": 201, "n_runs": 4, "run_size": 64, "hidden_features": 128, "num_transforms": 8},
     {"max_num_epochs": 200},
     "201 epochs, network 128 by 8, on 256 simulations, prior retrain_prior. Saved as a record. The fit reached the "
     "epoch limit; the network kept is its last epoch's."),
    ({"epochs_trained": 150, "n_runs": 4, "run_size": 64, "hidden_features": 128, "num_transforms": 8},
     {"max_num_epochs": 200}, "150 epochs, network 128 by 8, on 256 simulations, prior retrain_prior. Saved as a record."),
    ({"epochs_trained": 200, "n_runs": 4, "run_size": 64, "hidden_features": 128, "num_transforms": 8},
     {"max_num_epochs": 200}, "200 epochs, network 128 by 8, on 256 simulations, prior retrain_prior. Saved as a record."),
    ({"epochs_trained": None}, {}, "Saved as a record."),
], ids=["no limit", "the limit reached", "a limit not reached", "as many epochs as the limit", "a record that says too "
                                                                                            "little"])
def test_what_the_step_holds_says_its_training(training, config, said):
    """"In use" after a training: its epochs, its network, the simulations it was trained on and its prior, from its
    record; a fit that ran past a limit set, which it does only by running one epoch more, says it kept its last epoch's
    network, and one that ran as many epochs as the limit ended by itself; what the record does not hold is left out."""
    assert posterior_in_use("retrain_posterior", training, config, "retrain_prior") == ("retrain_posterior", said)
    assert posterior_in_use("", training, config, "retrain_prior") == ("", "")


# ── the plan in use ──
class _Row:
    """A row a plan fills, as the plan's sentences read it: its key, its label, its usual value and its value now."""

    def __init__(self, key, label, usual, value, *, wrong=False):
        self.key, self._label, self._usual, self._value, self._wrong = key, label, usual, value, wrong
        self.kept = False

    def label_text(self):
        return self._label

    def usual(self):
        return self._usual

    def value(self):
        return self._value

    def in_error(self):
        return self._wrong


class _Step:
    """A step a plan fills, as the plan's sentences read it: its rows, what its form holds and what it holds applied."""

    def __init__(self, step, rows, applied=None):
        self.step, self._rows, self._applied = step, {row.key: row for row in rows}, applied

    def plan_fields(self):
        return self._rows

    def plan_read(self, applied):
        if applied:
            return dict(self._applied or {})
        return {key: row.value() for key, row in self._rows.items()}


def _screen(source=None, *, batches=10000, width=256, rotate=True, applied_rotate=True, state="unread"):
    """A screen of the three steps as the plan's sentences read it: the batches and the network width typed on Training,
    the rotation chosen on step 1 and applied there, ``source`` in use and ``state`` the plan file's last reading: its
    hash, None for a file gone, or not read yet."""
    from types import SimpleNamespace
    from core.gui import plans
    model = _Step("model", [_Row("reparam_rotate", "Rotation", True, rotate)], {"reparam_rotate": applied_rotate})
    prior = _Step("prior", [_Row("bounds", "Bounds file", None, "master.txt")], {"bounds": "master.txt"})
    training = _Step("training", [_Row("num_runs", "Batches", 5000, batches),
                                  _Row("hidden_features", "Network width", 128, width)])
    return SimpleNamespace(plan_source=source, plan_file=plans.UNREAD if state == "unread" else state,
                           step_pages=lambda: (model, prior, training))


def _source(kind="plan", name="retrain", values=None):
    from datetime import datetime
    from pathlib import Path
    from core.gui import plans
    held = values if values is not None else {("Training", "num_runs"): 10000, ("Training", "hidden_features"): 256,
                                              ("Prior", "bounds"): "master.txt"}
    return plans.PlanSource(kind, name, Path("plans/nadrowski/retrain.txt") if kind == "plan" else None, "a" * 64,
                            datetime(2026, 10, 6, 12, 4), held)


def _plan_said(screen, headings=("Model and options", "Prior", "Training")):
    from core.gui import plans
    return [(fact.kind, fact.sentence) for fact in plans.plan_facts(screen, headings)]


def test_plan_facts_say_each_sentence():
    """What "Before you start" says of the plan, word for word: no plan with every setting usual, or with each departure
    from the usual named; a plan whose settings are in use as written, counted, or each departure from it named, against
    the plan's value; a record or a run as the source; a plan file changed or removed since its fill. Every departure
    named, none cut, in the steps' order; only the changed file's is "Attention.", and none blocks."""
    assert _plan_said(_screen(batches=5000, width=128)) == [("info", "No plan. Every setting is at its usual value.")]
    assert _plan_said(_screen(rotate=False, applied_rotate=False)) == [
        ("info", "No plan. 3 settings differ from the usual: Rotation off (usual on); Batches 10,000 (usual 5,000); "
                 "Network width 256 (usual 128).")]
    assert _plan_said(_screen(batches=5000, width=128, applied_rotate=False)) == [
        ("info", "No plan. 1 setting differs from the usual: Rotation off (usual on).")]
    assert _plan_said(_screen(_source())) == [("info", "Plan retrain. Its 3 settings are in use as written.")]
    assert _plan_said(_screen(_source(values={("Training", "num_runs"): 10000}), width=128)) == [
        ("info", "Plan retrain. Its 1 setting is in use as written.")]
    assert _plan_said(_screen(_source(values={("Calibration", "n_cal"): 100}), batches=5000, width=128)) == [
        ("info", "Plan retrain. This step reads none of its settings.")]
    assert _plan_said(_screen(_source(), batches=9500, width=128, applied_rotate=False)) == [
        ("info", "Plan retrain, with 3 changes: Rotation off (usual on); Batches 9,500 (plan 10,000); Network width "
                 "128 (plan 256).")]
    assert _plan_said(_screen(_source(), batches=9500)) == [
        ("info", "Plan retrain, with 1 change: Batches 9,500 (plan 10,000).")]
    assert _plan_said(_screen(_source("record", "retrain_posterior"), batches=9500)) == [
        ("info", "Settings from record retrain_posterior, with 1 change: Batches 9,500 (record 10,000).")]
    assert _plan_said(_screen(_source("run", "retrain_3"))) == [
        ("info", "Settings from run retrain_3. Its 3 settings are in use as written.")]
    assert _plan_said(_screen(_source(), state="b" * 64)) == [
        ("info", "Plan retrain. Its 3 settings are in use as written."),
        ("warn", "retrain.txt has been changed since its settings were filled in on 6 Oct 2026, 12:04. The forms hold "
                 "the earlier values; fill again to take the changes.")]
    assert _plan_said(_screen(_source(), state=None)) == [
        ("info", "Plan retrain. Its 3 settings are in use as written."),
        ("warn", "retrain.txt is no longer in the plans folder. The forms hold the values filled from it on 6 Oct "
                 "2026, 12:04.")]
    assert _plan_said(_screen(_source(), state="a" * 64)) == [
        ("info", "Plan retrain. Its 3 settings are in use as written.")]
    from dataclasses import replace
    from pathlib import Path
    record = replace(_source("record", "retrain_posterior"), path=Path("retrain_posterior.json"))
    assert _plan_said(_screen(record, state="b" * 64))[1:] == [], "a record's source was said to be a changed file"


def test_plan_facts_read_the_form_of_their_own_step_and_what_the_others_hold():
    """A step's run reads its own form and what the earlier steps hold: the last of the headings handed is read from its
    form, each earlier one from what its step holds; a step the run does not read says nothing; a setting in error on
    the form is no departure, its own line says it; a bounds file the source does not hold has no usual to depart
    from."""
    from core.gui import plans
    screen = _screen(rotate=False, applied_rotate=True, batches=5000, width=128)
    assert _plan_said(screen) == [("info", "No plan. Every setting is at its usual value.")]
    assert _plan_said(screen, ("Model and options",)) == [
        ("info", "No plan. 1 setting differs from the usual: Rotation off (usual on).")]
    wrong = _screen(batches=0, width=128)
    wrong.step_pages()[2].plan_fields()["num_runs"]._wrong = True
    assert _plan_said(wrong) == [("info", "No plan. Every setting is at its usual value.")]
    assert plans.plan_line(_screen(_source(), batches=9500), ("Model and options", "Prior", "Training")) == (
        "Plan: retrain, 1 change")
    assert plans.plan_line(_screen(_source(), batches=9500, width=64), ("Model and options", "Prior", "Training")) == (
        "Plan: retrain, 2 changes")
    assert plans.plan_line(_screen(_source()), ("Model and options",)) == "Plan: retrain"
    assert plans.plan_line(_screen(), ("Model and options",)) == "No plan"
    assert plans.plan_line(_screen(_source("record", "retrain_posterior")), ("Model and options",)) == (
        "Settings from record retrain_posterior")


# ── the lead beside the button ──
#: The blocks that neither waiting nor an edit on the page lifts: a model inference cannot take, a file or a record
#: that cannot be read, a record made for another model, nothing to continue.
FINAL = ("this model cannot be used", "the bounds file cannot be read", "the prior cannot be read",
         "this prior is for another model", "there is nothing to continue", "the posterior cannot be read",
         "this posterior is for another model")
#: The blocks a person lifts on the page: a field, a choice, a name, a step applied, a record chosen or made.
YET = ("a setting cannot be used", "the limits typed cannot be used", "the last start was refused", "needs a model",
       "no bounds file is chosen", "no saved prior is chosen", "no prior is saved yet", "needs a prior",
       "a saved simulation set would be continued", "no saved posterior is chosen", "no posterior is saved yet",
       "the name cannot be used", "a name is needed", "the note cannot be used",
       "the units file chosen cannot be used")


def test_every_block_the_pages_say_is_known_to_be_lifted_on_the_page_or_not():
    """Every block a page of the three steps can say beside its button, read from the names of the facts' module, the
    form's and the three pages' that end in BLOCKS, is either lifted on the page and reads "Cannot start yet:", or is
    not and reads "Cannot start:". A block added without a place in one of the two fails here."""
    from core.gui import step_facts
    from core.gui.panels.inference import config_tab, posterior_tab, prior_tab
    from core.gui.widgets import form
    said = {value for module in (step_facts, form, config_tab, prior_tab, posterior_tab)
            for name, value in vars(module).items() if name.endswith("BLOCKS") and isinstance(value, str)}
    said |= {form.NEEDS_A_NAME}
    assert set(step_facts.FINAL_BLOCKS) == set(FINAL)
    assert said == set(FINAL) | set(YET), (sorted(said - set(FINAL) - set(YET)), sorted((set(FINAL) | set(YET)) - said))


@pytest.mark.parametrize("words", FINAL + YET)
def test_the_lead_beside_the_button_says_whether_the_page_can_lift_the_block(words):
    """One block: "Cannot start:" for a block no wait or edit on the page lifts, "Cannot start yet:" for one the page
    lifts. Two: the lead is the first's, the one its words name. A run live says so, whatever blocks."""
    lead = "Cannot start:" if words in FINAL else "Cannot start yet:"
    one = (Fact("err", "A sentence.", words),)
    assert reason_beside(one, live=False, nothing_to_apply=False) == f"{lead} {words}."
    other = "a name is needed" if words in FINAL else "the prior cannot be read"
    two = (Fact("err", "A sentence.", words), Fact("lock", "Another.", other))
    assert reason_beside(two, live=False, nothing_to_apply=False) == f"{lead} {words}, and 1 more above."
    assert reason_beside(two, live=True, nothing_to_apply=False) == "A run is live. Only one runs at a time."


# ── after a Resume ──
@pytest.mark.parametrize("device, width, said", [
    ("cpu", 2048, "Runs on the processor as the resumed training was kept, 2,048 simulations at a time, until step 1 "
                  "or step 2 is applied again."),
    ("cuda", 16384, "Runs on the graphics card as the resumed training was kept, 16,384 simulations at a time, until "
                    "step 1 or step 2 is applied again."),
    ("cpu", None, "Runs on the processor as the resumed training was kept, until step 1 or step 2 is applied again."),
    ("meta", 64, ""), (None, 64, "")])
def test_step_one_says_the_device_and_the_width_a_resume_runs_on(device, width, said):
    """After a Resume step 1's "In use" adds the device the session now uses and, when it is the kept training's own,
    its width, grouped by thousands; a device of no kind known says nothing."""
    from core.gui.step_facts import resumed_words
    assert resumed_words(device, width) == said
