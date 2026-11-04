"""What a step says in "Before you start", as values: each row of the step's table, its sentence word for word, and what
blocks. No widget is built here."""
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


@pytest.mark.parametrize("reading, said", [
    (_reading(checkpoint="resume_partial", batches_done=3989),
     ("fine", "Continues the saved simulation set: 3,989 of 10,000 batches are done.")),
    (_reading(checkpoint="resume_complete", batches_done=10000),
     ("fine", "No simulations this time: the saved 20,480,000 are used.")),
    (_reading(checkpoint="resume_complete", batches_done=10000, width=None),
     ("fine", "No simulations this time: the saved 10,000 batches are used.")),
    (_reading(), ("info", "Starts a new simulation set; none is saved for this model yet.")),
    (_reading(checkpoint="new_with_siblings", sibling_diffs=(_set(4000, ("n_runs", 10000, 4000)),)),
     ("warn", "Starts a new simulation set. 1 saved set for this model does not match these settings: The set of 4,000 "
              "batches (complete) differs in Batches: 10000 here, 4000 in that set.")),
    (_reading(checkpoint="new_with_siblings", sibling_diffs=(
        _set(4000, ("n_runs", 10000, 4000), ("reparam_rotate", True, False), ("nd_lows", [0.1], [0.2])),
        _set(12, ("run_size", 2048, 1024), complete=False, name="b"))),
     ("warn", "Starts a new simulation set. 2 saved sets for this model do not match these settings: The set of 4,000 "
              "batches (complete) differs in Batches: 10000 here, 4000 in that set; Rotation: True here, False in that "
              "set; nd_lows: [0.1] here, [0.2] in that set. The set of 12 batches (partial) differs in Simulations per "
              "batch: 2048 here, 1024 in that set.")),
    (_reading(checkpoint="new_with_siblings", sibling_diffs=(_set(4000, ("model", "NADROWSKI", "HOPF")),)),
     ("info", "Starts a new simulation set; none is saved for this model yet.")),
], ids=["continues a set", "a complete set", "a complete set of no known width", "none saved", "one set that differs",
        "two sets", "a set of another model"])
def test_continue_or_new_says_each_state(reading, said):
    """What the training does with the saved simulation sets, as the reading at this budget found them, word for word: it
    continues the set of these settings, uses a complete one, or starts a new one beside none of this model, or beside
    every set of this model that does not match, each with everything it differs in, by the names this step can give and
    the identity's own else. A set of another model is not counted. None of these blocks."""
    facts = training_facts(**{**TRAINING, "reading": reading})
    assert _said(facts) == [said, *FIT_KEPT]
    assert not any(fact.blocks for fact in facts)


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


@pytest.mark.parametrize("state", ["new", "new_with_siblings"])
def test_only_continue_with_nothing_to_continue_blocks_with_its_reason(state):
    """"Only continue" with no set of these settings to continue blocks, with its words beside the button; with a set to
    continue it blocks nothing."""
    facts = training_facts(reading=_reading(checkpoint=state), resume="require", batches=10000)
    assert facts[0] == Fact("err", "There is no saved simulation set to continue.", "there is nothing to continue")
    assert reason_beside(facts, live=False, nothing_to_apply=False) == "Cannot start yet: there is nothing to continue."
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
