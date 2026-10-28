"""Stopping a run, and closing the window during one: the question the stop button asks for each kind of run and
each stage, what the page, the foot and Home say while the stop is pending, withdrawing it, keeping or discarding a
training the stop kept, where the keyboard goes meanwhile, the close's question and the window that waits for its
run to end before it closes with every window it opened, Windows' end of a session (its question, "shut down anyway"
and "cancel"), a figure shown as its picture while a run is live, and a figure the window could not show.

A live run here is the stand-in run of tests/_runs.py, started through a real panel's own dispatch; the window's
clock is moved by hand, and every question is answered through the session's guard or ``choose``.
"""
import logging
import threading
import warnings
from datetime import datetime, timedelta, timezone

import pytest

from core.gui import this_computer
from core.gui.run import questions
from core.gui.run.controller import controller
from core.gui.run.model import SILENCE_NEEDS_YOU_S, SIMULATION_ROW, RunSpec
from tests._fixtures import (all_asked, asked, build_fit_record, build_run_host, build_window, choose,
                             forget_asked, hold_open, qt_app)
from tests._runs import Clock, RunText, StandIn, start_run, turn_until

# local time with its zone, as the window's clock gives it
_ZONE = timezone(timedelta(hours=-7))
# a simulation set's folder as the engine names it
_FOLDER = r"C:\Users\J\PRISM\Artifacts\simulations\_unnamed__20261008T190211"
# a training's lines, as the engine writes them
_BUDGET = "[budget] 10,000 batches x 2,048 rows = 20,480,000 training rows"
_ROTATING = "Computing decorrelating Fisher rotation (REPARAM_ROTATE=True)..."
_ROTATED = "[fisher] eigenvalue spread (best/worst direction): 412"
_WRITING = f"[checkpoint] writing to {_FOLDER} every 50 batches (~10.3 GiB total, 120.5 GiB free)"
_COMPLETE = (f"[checkpoint] complete: 10000 batches in {_FOLDER}. Safe to delete once the posterior is saved; keeping "
             f"it lets you retrain the flow without re-simulating.")
_NOT_SAVED = "[fit] saving is switched off: this training is not saved as it goes and cannot be resumed."
# what the stop of a training in its fit says while it is pending, an epoch having taken 63 minutes
_STOPPING_FIT = "Stopping: takes effect when this epoch ends, up to about 63 minutes."


def _at(day, hour, minute, second=0) -> datetime:
    """A moment of October 2026 in the test's local time (the 8th is a Thursday)."""
    return datetime(2026, 10, day, hour, minute, second, tzinfo=_ZONE)


def _epoch(epoch, since=0, took=3780.0) -> str:
    """An epoch's record, as the training call writes it."""
    return (f"[fit] epoch {epoch}: training loss -3, validation loss -3.1, took {took:.1f} s, epochs since the best "
            f"{since}")


def _controller_on(tmp_path, clock):
    """The process's run controller, its clock and its folder of runs this test's own; the end of the test puts
    both back."""
    held = controller()
    held.use_clock(clock)
    held.use_folder(tmp_path / "runs")
    return held


def _training(word="Training", **changed) -> RunSpec:
    """A training as its page hands it to its run: its stages, its patience and its epoch limit."""
    from core.gui.panels.inference.base import TRAINING_STAGES
    button = "Stop training…" if word == "Training" else "Stop the narrowing round…"
    return RunSpec(**{"word": word, "place": ("inference", "training"), "name": "retrain_3", "stop_button": button,
                      "stages": TRAINING_STAGES, "kinds": ("posterior", "simulation", "fit"), "simulation_set": True,
                      "retries": True, "patience": 20, "epoch_limit": 300,
                      "facts": {"model": "NADROWSKI", "observation": "chi", "narrowed": word != "Training",
                                "network": (256, 10)}, **changed})


def _say(app, stand_in, text, *lines, level="info") -> None:
    """Have the stand-in run say ``lines``, and turn the event loop until the run's full text holds the last."""
    for line in lines:
        stand_in.say(line, level)
    turn_until(app, lambda: lines[-1] in text.texts())


def _to_the_simulation(app, stand_in, text, held, done=6_150) -> None:
    """A training brought to its simulation, ``done`` of its 10,000 batches simulated."""
    _say(app, stand_in, text, _BUDGET, _ROTATING, _ROTATED, _WRITING)
    stand_in.row(SIMULATION_ROW, done, 10_000)
    turn_until(app, lambda: held.model.batches_done() == done)


def _to_the_fit(app, stand_in, text, clock, epochs, *, saved=True) -> None:
    """A training brought to its fit, one epoch's time later, its records ``epochs`` ((epoch, since the best), ...)
    said; its fit not saved as it goes when not ``saved``."""
    _say(app, stand_in, text, _BUDGET, _ROTATING, _ROTATED, _WRITING, *(() if saved else (_NOT_SAVED,)), _COMPLETE)
    clock.advance(3780)
    if epochs:
        _say(app, stand_in, text, *(_epoch(epoch, since) for epoch, since in epochs))


def _note(kept_id, passes=62, resumed=False, saved_here=True, processes=1):
    from core.SBI.fit_state import KeptNote
    return KeptNote(kept_id, passes, resumed, saved_here, processes)


def _ended(app, held) -> None:
    turn_until(app, lambda: not held.is_live(), 10)


def _note_the_release(monkeypatch, folder, run, text, view) -> list:
    """Note, as the one-PRISM claim is let go, what the history of the run ``run`` in ``folder`` says of its end, the
    ends the run's controller has said (heard by ``text``, the last one) and the mode of the run's page ``view``; the
    list the note is put in."""
    from core.gui import one_instance
    from core.gui.run import history
    released, real = [], one_instance.release

    def release():
        if not released:
            record = history.read(folder / f"{run}.json")
            released.append((None if record is None else record["outcome"],
                             [status.outcome for status in text.ended][-1:], view.mode))
        real()

    monkeypatch.setattr(one_instance, "release", release)
    return released


# ── the stop's question ──
# The table of what each stop asks, by kind of run and stage: the spec, how the run is brought to its stage, and the
# question: its title, its sentence, what stands beneath it, the other answer, the safe one, and whether the other
# loses work.
_THREE_EPOCHS = ((1, 0), (2, 1), (3, 2))
_CASES = {
    "training, its rotation": (
        _training(), "rotation", ("Stop training?", "Nothing has been simulated yet. The rotation will be computed "
                                  "again next time.", "", "Stop training", "Keep training", False)),
    "training, its simulation": (
        _training(), "simulation", ("Stop training?", "The 6,150 batches simulated so far are kept; the batch in "
                                    "progress is not. You can continue from batch 6,150.", "",
                                    "Stop and keep the simulations", "Keep training", False)),
    "training, its simulation, no batch yet": (
        _training(), "no batch", ("Stop training?", "Nothing has been simulated yet; the batch in progress is not "
                                  "kept.", "", "Stop and keep the simulations", "Keep training", False)),
    "training, its simulation, one batch": (
        _training(), "one batch", ("Stop training?", "The 1 batch simulated so far is kept; the batch in progress is "
                                   "not. You can continue from batch 1.", "", "Stop and keep the simulations",
                                   "Keep training", False)),
    "training, its rotation off": (
        _training(facts={"model": "NADROWSKI", "observation": "chi", "narrowed": False, "network": (256, 10),
                         "rotated": False}), "budget",
        ("Stop training?", "Nothing has been simulated yet.", "", "Stop training", "Keep training", False)),
    "training, its fit": (
        _training(), "fit", ("Stop training?", "The simulations are kept. The fit stops when the epoch in progress "
                             "ends, up to about 63 minutes from now; every epoch finished so far is kept.", "",
                             "Stop training", "Keep training", False)),
    "training, its fit, an epoch that may be its last": (
        _training(), "last epoch", ("Stop training?", "The simulations are kept. The fit stops when the epoch in "
                                    "progress ends, up to about 63 minutes from now; every epoch finished so far is "
                                    "kept.", "This epoch may be the last: if validation does not improve, the fit ends "
                                    "by itself at about Thu 8 Oct, 21:08.", "Stop training", "Keep training", False)),
    "training, its fit not saved as it goes": (
        _training(), "fit not saved", ("Stop training?", "The simulations are kept. The fit stops when the epoch in "
                                       "progress ends; this training is not saved as it goes, so the fit so far is "
                                       "lost.", "", "Stop and lose the fit", "Keep training", True)),
    "narrowing round": (
        _training("Narrowing round"), "rotation", ("Stop the narrowing round?", "Nothing has been simulated yet. The "
                                                   "rotation will be computed again next time.", "", "Stop training",
                                                   "Keep training", False)),
    "prior build": (
        RunSpec("Prior build", ("inference", "prior"), stop_button="Stop the prior build…"), None,
        ("Stop the prior build?", "Nothing of the prior is kept; it is built again from the start next time.", "",
         "Stop the build", "Keep building", True)),
    "check": (
        RunSpec("Check", ("inference", "checks"), name="Calibration", stop_button="Stop the calibration…"), None,
        ("Stop the calibration?", "A check writes its record only when it finishes, so nothing of this one is kept.",
         "", "Stop and discard it", "Keep it running", True)),
    "inference": (
        RunSpec("Inference", ("inference", "infer"), stop_button="Stop the inference…"), None,
        ("Stop the inference?", "The inference is not kept. An observation it has already written stays in Records.",
         "", "Stop the inference", "Keep it running", True)),
    "measurement": (
        RunSpec("Measurement", ("temperature", "one_cell"), name="cell_a_3oct", stop_button="Stop measurement…"),
        4_320, ("Stop measurement?", "It has run for 1 hour 12 minutes. The record cell_a_3oct is kept, marked "
                "unfinished, with what it has measured so far. A measurement cannot be continued; to finish it, it is "
                "measured again from the start.", "", "Stop and keep what is measured", "Keep measuring", True)),
    "sweep study": (
        RunSpec("Sweep study", ("temperature", "sweep"), name="study", stop_button="Stop sweep study…"), 20_400,
        ("Stop sweep study?", "It has run for 5 hours 40 minutes. What the study has measured is kept, marked "
         "unfinished; a study cannot be continued.", "", "Stop the study", "Keep measuring", True)),
    "comparison": (
        RunSpec("Comparison", ("temperature", "one_cell"), stop_button="Stop the comparison…"), None,
        ("Stop the comparison?", "A comparison writes its record only when it finishes, so nothing of this one is "
         "kept.", "", "Stop the comparison", "Keep comparing", True)),
    "reduction map": (
        RunSpec("Reduction map", ("reduction", None), stop_button="Stop computing…"), None,
        ("Stop computing the map?", "The map will not be shown. If its table is already written, that file stays in "
         "the reduction folder.", "", "Stop computing", "Keep computing", True)),
    "video": (
        RunSpec("Video", ("live", None), stop_button="Stop saving…", keeps_files=False), None,
        ("Stop saving the video?", "The unfinished file is removed. The trace stays and can be saved again.", "",
         "Stop saving", "Keep saving", True)),
}


@pytest.mark.parametrize("case", list(_CASES))
def test_the_stop_button_names_the_run_and_asks_what_its_stage_keeps(case, tmp_path):
    """The run page's stop button names the run, and asks before it stops anything, in the words of the run and of
    the stage it is in: what is kept and what is lost, the safe answer first (the default, Enter's and Escape's, the
    answer a question nobody answers gives), and the other drawn as losing work only where it loses what cannot be
    had again. Left unanswered, nothing is stopped."""
    spec, stage, (title, sentence, beneath, other, safe, loses) = _CASES[case]
    app = qt_app()
    clock = Clock(_at(8, 19, 2, 11))
    held, host, stand_in = _controller_on(tmp_path, clock), build_run_host(), StandIn()
    text = RunText(held)
    start_run(host, stand_in, spec=spec)
    if stage == "rotation":
        _say(app, stand_in, text, _BUDGET, _ROTATING)
    elif stage == "budget":
        _say(app, stand_in, text, _BUDGET)
    elif stage == "simulation":
        _to_the_simulation(app, stand_in, text, held)
    elif stage == "no batch":
        _say(app, stand_in, text, _BUDGET, _ROTATING, _ROTATED, _WRITING)
    elif stage == "one batch":
        _to_the_simulation(app, stand_in, text, held, done=1)
    elif stage == "fit":
        _to_the_fit(app, stand_in, text, clock, _THREE_EPOCHS)
    elif stage == "last epoch":
        _to_the_fit(app, stand_in, text, clock, [(epoch, epoch - 1) for epoch in range(1, 21)])
    elif stage == "fit not saved":
        _to_the_fit(app, stand_in, text, clock, _THREE_EPOCHS, saved=False)
    elif stage is not None:
        clock.advance(stage)
    view = host.run_view
    assert view.stop_button.text() == spec.stop_button, "the stop button does not name the run"
    forget_asked()
    view.stop_button.click()
    question = asked()
    assert (question.kind, question.title, question.sentence, question.beneath) == ("question", title, sentence,
                                                                                    beneath), question
    assert (question.answers, question.default, question.safe, question.loses_work) == ((other, safe), safe, safe,
                                                                                        loses), question
    assert (len(all_asked()), held.is_live(), held.status.stop_requested, held._run.token.requested.is_set()) == (
        1, True, False, False), "the stop asked more than once, or stopped the run with no answer"
    stand_in.release()
    _ended(app, held)


# ── stopping, and withdrawing a stop ──
def test_a_stop_during_the_fit_lands_when_the_epoch_ends_and_says_so_meanwhile(tmp_path, monkeypatch, store):
    """A stop asked for during the fit is taken when the epoch in progress ends: the record that would carry it is
    never said. Meanwhile the run is stopping, and says when the stop is taken -- before the fit's first epoch, at
    that epoch's end; once an epoch is measured, at that epoch's end, up to about the time an epoch takes -- on its
    page, where the stop button is greyed and "Don't stop after all" stands beside it, at the rail's foot, whose
    title is that the run is stopping, on Home, which says the run is stopping, and on the run's step in the rail.
    "Don't stop after all" puts each back as the run says it while running: the foot then says the stage the run is
    in."""
    from core.gui.widgets.page import Sentence
    from tests._window_helpers import shown_off_screen
    app = qt_app()
    build_fit_record(store)                         # a record: Home states its next step, not its first page
    clock = Clock(_at(8, 19, 2, 11))
    held, window = _controller_on(tmp_path, clock), build_window()
    shown_off_screen(window)
    host, text = window.part("inference").posterior_panel, RunText(held)
    choose(monkeypatch, words="Stop training")
    young = StandIn()
    start_run(host, young, spec=_training())
    _to_the_fit(app, young, text, clock, ())
    host.run_view.stop_button.click()
    assert host.run_view.sentence.text() == "Stopping: takes effect when this epoch ends.", (
        "no epoch measured, and the stop's line names a time")
    _ended(app, held)

    stand_in = StandIn(takes_stops=False)
    start_run(host, stand_in, spec=_training())
    _to_the_fit(app, stand_in, text, clock, _THREE_EPOCHS)
    view, foot, home = host.run_view, window.shell.rail.foot, window.part("home")
    step = window.shell.rail.row("training")

    def home_says():
        return [label.plain_text() for label in home.findChildren(Sentence) if label.isVisibleTo(home)]

    view.stop_button.click()
    assert (held.status.state, held.status.stop_requested, held._run.token.requested.is_set()) == (
        "stopping", True, True)
    assert (view.sentence.text(), view.stop_button.isEnabled(), view.withdraw_button.isVisibleTo(view),
            view.withdraw_button.text()) == (_STOPPING_FIT, False, True, "Don't stop after all")
    assert (foot._marker, foot._title, foot._words, foot.is_button()) == (
        "now", "Training is stopping", _STOPPING_FIT, True), "the foot does not say the run is stopping"
    assert step.accessibleName() == "Step 3, Training. Now: stopping", "the rail does not say the run is stopping"
    window.go_to("home")
    assert ("Training retrain_3 is stopping, and no other run can start until it ends. You can still move around and "
            "read results. Keep PRISM open; you can minimise it.") in home_says(), home_says()
    view.withdraw_button.click()
    assert (held.status.state, foot._title, foot._words, step.accessibleName()) == (
        "running", "Training is running", "Fitting, epoch 3. Settings are locked.", "Step 3, Training. Now: running"), (
        "after 'Don't stop after all' the foot or the rail still says the run is stopping", foot._title, foot._words)
    assert ("Training retrain_3 is running, and no other run can start until it ends. You can still move around and "
            "read results. Keep PRISM open; you can minimise it.") in home_says(), home_says()
    view.stop_button.click()
    assert (held.status.state, foot._title) == ("stopping", "Training is stopping")
    last = _epoch(4, 3)
    stand_in.say(last)
    _ended(app, held)
    assert (stand_in.stopped_at, last in text.texts(), [status.outcome for status in text.ended][-1]) == (
        last, False, "stopped"), "the stop was not taken at the epoch's record, or the record was said"


def test_a_stop_can_be_withdrawn_until_it_lands(tmp_path, monkeypatch):
    """"Don't stop after all" withdraws a stop until it is taken: the run goes on, its stop no longer asked, the page
    says so in the run's own word, and the stop button can be pressed again. Once the run has begun to stop, a
    withdraw changes nothing and the page says it came too late; the run stops. The stop button is greyed, with no
    question, while the posterior is saved after the fit has ended."""
    app = qt_app()
    clock = Clock(_at(8, 19, 2, 11))
    held, host = _controller_on(tmp_path, clock), build_run_host()
    text, view = RunText(held), host.run_view
    stand_in = StandIn(takes_stops=False)
    start_run(host, stand_in, spec=_training())
    _to_the_simulation(app, stand_in, text, held)
    choose(monkeypatch, words="Stop and keep the simulations")
    view.stop_button.click()
    token = held._run.token
    assert (held.status.state, token.requested.is_set(), view.withdraw_button.isVisibleTo(view)) == (
        "stopping", True, True)
    view.withdraw_button.click()
    assert (held.status.state, held.status.stop_requested, token.requested.is_set(), token.fired) == (
        "running", False, False, False), "the stop was not withdrawn"
    assert (view.notice.text(), view.notice.isVisibleTo(view), view.withdraw_button.isVisibleTo(view),
            view.stop_button.isEnabled()) == ("The stop was withdrawn. The training goes on.", True, False, True)
    _say(app, stand_in, text, "a line after the withdraw")
    assert held.is_live() and stand_in.stopped_at is None, "a withdrawn stop was taken"

    view.stop_button.click()
    assert view.notice.text() == "", "the earlier withdraw was still said once a stop was asked again"
    stand_in.say("the line at which the stop is taken")
    assert stand_in.stopped.wait(10), "the stop was not taken"
    assert token.fired and held.is_live(), "the run's end was heard before the window turned"
    view.withdraw_button.click()                    # the end not heard yet: the run has begun to stop
    assert (token.fired, view.notice.text()) == (True, "Too late: the run had already begun to stop.")
    _ended(app, held)
    assert [status.outcome for status in text.ended][-1] == "stopped"

    saving = StandIn()
    start_run(host, saving, spec=_training())
    _to_the_fit(app, saving, text, clock, [(1, 0), (2, 20)])          # 20 epochs since the best: the fit has ended
    forget_asked()
    assert (held.model.stage, view.stop_button.isVisibleTo(view), view.stop_button.isEnabled(),
            view.sentence.text()) == (
        "Saving", True, False, "The fit ended after 2 epochs. The posterior is being saved.")
    view.stop_button.click()
    assert (all_asked(), held.status.stop_requested) == ([], False), "a greyed stop button asked or stopped"
    saving.release()
    _ended(app, held)


def test_a_withdraw_between_the_checks_two_readings_wins_every_time():
    """The stop's token reads whether a stop is asked for twice before it is taken: once with no lock, then again
    under its lock, which a withdraw takes too. Held, by a gate in place of the lock, between the two readings, the
    run's thread meets a withdraw made meanwhile at the second reading: the withdraw says it came in time, nothing
    is raised, and nothing fired. The other way round, a stop taken first is not withdrawn. The gate holds the run's
    thread alone: the withdraw passes it, and takes the lock all the same, each way round (the gate notes the thread
    of each who enters)."""
    from core.gui.streams import CancelToken, WorkerCancelled

    class Gate:
        """The token's lock, holding the run's thread at its door until it is opened; any other thread passes. The
        name of each thread that enters is noted, in the order they came."""

        def __init__(self, held_thread_name):
            self._inner, self._held = threading.Lock(), held_thread_name
            self.reached, self.opened = threading.Event(), threading.Event()
            self.entered = []

        def __enter__(self):
            self.entered.append(threading.current_thread().name)
            if threading.current_thread().name == self._held:
                self.reached.set()
                self.opened.wait(10)
            self._inner.acquire()
            return self

        def __exit__(self, *exc):
            self._inner.release()
            return False

    mine = threading.current_thread().name
    for withdraw_first in (True, False):
        token, said = CancelToken(), {}
        token._lock = gate = Gate("the run")
        if not withdraw_first:
            gate.opened.set()

        def run():
            token.arm()
            try:
                token.check()
                said["raised"] = False
            except WorkerCancelled:
                said["raised"] = True

        token.requested.set()
        thread = threading.Thread(target=run, name="the run")
        thread.start()
        if withdraw_first:
            assert gate.reached.wait(10), "the check took no lock before it fired"
            said["withdrawn"] = token.withdraw()
            gate.opened.set()
            thread.join(10)
            assert (said, token.fired, token.requested.is_set(), mine in gate.entered) == (
                {"withdrawn": True, "raised": False}, False, False, True), (said, gate.entered)
        else:
            thread.join(10)
            said["withdrawn"] = token.withdraw()
            assert (said, token.fired, token.requested.is_set(), gate.entered) == (
                {"raised": True, "withdrawn": False}, True, True, ["the run", mine]), (said, gate.entered)


def test_live_simulations_and_a_loads_stop_ask_nothing_and_cannot_be_withdrawn(tmp_path):
    """A live simulation's stop and a load's stop lose nothing, so they ask nothing: the stop is asked for at once,
    no "Don't stop after all" is offered, and a withdraw is refused; each run ends stopped."""
    app = qt_app()
    held = _controller_on(tmp_path, Clock(_at(8, 19, 2, 11)))
    host, text = build_run_host(), RunText(held)
    view = host.run_view
    live = StandIn(word="Live simulation", takes_stops=False)
    start_run(host, live, spec=RunSpec("Live simulation", ("live", None), stop_button="Stop simulation",
                                       keeps_files=False, stop_asks=False, own_view=True))
    forget_asked()
    view.stop_button.click()
    assert (all_asked(), held.status.stop_requested, view.withdraw_button.isVisibleTo(view)) == ([], True, False)
    assert (held.withdraw_stop(), held._run.token.requested.is_set(), held.status.stop_requested) == (
        False, True, True), "a live simulation's stop was withdrawn"
    live.say("the next frame")
    _ended(app, held)
    loads = StandIn(word="Prior load", takes_stops=False)
    start_run(host, loads, spec=RunSpec("Prior load", ("inference", "prior"), stop_button="Stop loading",
                                        loads_only=True, keeps_files=False, stop_asks=False))
    assert questions.stop_question(held.status, held.model) is None, "a load's stop asks something"
    held.request_stop()
    assert (held.withdraw_stop(), held._run.token.requested.is_set()) == (False, True), "a load's stop was withdrawn"
    loads.say("the next line")
    _ended(app, held)
    assert [status.outcome for status in text.ended] == ["stopped", "stopped"], text.ended
    assert all_asked() == []


# ── what a stop keeps ──
def test_a_stop_during_the_simulation_keeps_no_training_and_says_so(tmp_path, monkeypatch):
    """A training stopped during its simulation, before its fit saved anything, hands back no kept training: its page
    asks nothing about keeping one, and says, under the sentence that it was stopped, that none was kept."""
    app = qt_app()
    held, host = _controller_on(tmp_path, Clock(_at(8, 19, 2, 11))), build_run_host()
    text, view = RunText(held), host.run_view
    stand_in = StandIn()
    start_run(host, stand_in, spec=_training())
    _to_the_simulation(app, stand_in, text, held)
    choose(monkeypatch, words="Stop and keep the simulations")
    view.stop_button.click()
    _ended(app, held)
    assert (view.mode, view.sentence.text(), view.notice.text(), view.notice.isVisibleTo(view),
            view.keep_block.isVisibleTo(view)) == (
        "ended", "Training was stopped on Thu 8 Oct, 19:02, during simulation. No simulation was saved.",
        "No training was kept: the fit had not begun.", True, False)
    assert held.status.kept_fit is None


def test_once_a_stop_in_the_fit_lands_the_page_asks_to_keep_the_training(tmp_path, monkeypatch, store):
    """Once a stop in the fit has landed with a note of the training it kept, the run's page itself -- never a box --
    asks whether to keep it, from the note's numbers: "Keep" first, then "Discard", drawn as losing work. Unanswered the
    training stays, and the question stays with the page. Discard deletes the kept training, says so, and asks no more;
    the run's end then says, on its page and in its history, that the training was discarded; Keep deletes nothing and
    asks no more; a training the records no longer hold is said to be gone already, and asked about no more. A training
    kept by a run before the last is deleted by its id in that run's history, from the records that history names,
    whatever records are in use now, and its end's words then say so on its page; and in the status, the foot and
    Home's next step whenever that end is the status again, given back by a run that ended into idle since (a load of
    the page's own, which leaves the page and its offer as they were, or a run that keeps no file, ended before the
    Discard or still live at it). When that history cannot be read, nothing is deleted, the page says so, and the
    question stays."""
    from core.artifacts import ArtifactStore, use_store
    from core.gui import chain
    from core.gui.run import ends, history
    app = qt_app()
    clock = Clock(_at(8, 19, 2, 11))
    held, host = _controller_on(tmp_path, clock), build_run_host()
    text, view = RunText(held), host.run_view
    choose(monkeypatch, words="Stop training")

    def stopped_in_the_fit(note):
        stand_in = StandIn()
        stand_in.kept_fit = note
        start_run(host, stand_in, spec=_training())
        _to_the_fit(app, stand_in, text, clock, _THREE_EPOCHS)
        turn_until(app, lambda: len(held.model.epochs()) == 3)       # this run's own records, not an earlier run's
        forget_asked()
        view.stop_button.click()
        _ended(app, held)
        return held.status.run

    def kept():
        return [summary.id for summary in store.list("fit")]

    def discarded(at):
        """The end's words once the training a stop in the fit at ``at`` kept is discarded."""
        return (f"Training was stopped on {at}, during the fit. The 20,480,000 simulations are kept; the training was "
                f"discarded, as you asked.")

    first = build_fit_record(store, passes=62)
    run = stopped_in_the_fit(_note(first))
    assert [box.title for box in all_asked()] == ["Stop training?"], "a box asked to keep the training"
    assert (view.mode, view.keep_block.isVisibleTo(view), view.keep_words.text(), view.keep_beneath.text()) == (
        "ended", True, "Keep this training so you can resume it later?",
        "It stopped after epoch 62; the 62 epochs are kept until you discard them.")
    assert [(button.text(), button.property("cls")) for button in (view.keep_button, view.discard_button)] == [
        ("Keep", "primary"), ("Discard", "danger")]
    view.back_button.click()
    view.how_link.click()
    assert view.keep_block.isVisibleTo(view) and kept() == [first], "an unanswered question lost the training"
    said = "Training was stopped on Thu 8 Oct, 20:05, during the fit. The 20,480,000 simulations are kept; "
    assert (view.sentence.text(), held.status.kept) == (
        said + "the training is kept up to epoch 62, so you can resume it.",
        "The 20,480,000 simulations are kept; the training is kept up to epoch 62, so you can resume it.")
    view.discard_button.click()
    assert (kept(), view.notice.text(), view.keep_block.isVisibleTo(view)) == (
        [], "The training was discarded.", False)
    assert (view.sentence.text(), held.status.sentence, history.read(tmp_path / "runs" / f"{run}.json")["fit"][
        "discarded"]) == (said + "the training was discarded, as you asked.",
                          said + "the training was discarded, as you asked.", True), (
        "the end's words or its history still say the discarded training is kept")
    view.back_button.click()
    view.how_link.click()
    assert not view.keep_block.isVisibleTo(view), "the question was asked again"

    second = build_fit_record(store, passes=62)
    stopped_in_the_fit(_note(second))
    view.keep_button.click()
    assert (kept(), view.keep_block.isVisibleTo(view), view.notice.text()) == ([second], False, "")

    stopped_in_the_fit(_note("20261011T121000"))           # a training the store does not hold
    view.discard_button.click()
    assert (view.notice.text(), view.keep_block.isVisibleTo(view), kept()) == (
        "The training was already gone.", False, [second]), "a training already gone was not said to be"

    def later_run():
        """A run of another page, so that the controller holds the page's run no more."""
        later = StandIn()
        start_run(build_run_host(), later, spec=RunSpec("Check", ("inference", "checks"), keeps_files=False))
        later.release()
        _ended(app, held)

    third = build_fit_record(store, passes=62)
    run = stopped_in_the_fit(_note(third))
    assert history.read(tmp_path / "runs" / f"{run}.json")["fit"]["kept_id"] == third
    later_run()
    with use_store(ArtifactStore(tmp_path / "Elsewhere")):           # the records in use now are others
        view.discard_button.click()
    assert (sorted(kept()), view.notice.text(), view.keep_block.isVisibleTo(view)) == (
        [second], "The training was discarded.", False), (
        "the training an earlier run kept was not deleted by its history's id, from the records it names")
    assert history.read(tmp_path / "runs" / f"{run}.json")["fit"]["discarded"] is True, (
        "an earlier run's history does not say its training was discarded")
    assert (view.sentence.text(), held.status.sentence) == (discarded("Thu 8 Oct, 23:14"),) * 2, (
        "the page or the status still says the training an earlier run kept can be resumed")

    fourth = build_fit_record(store, passes=62)
    run = stopped_in_the_fit(_note(fourth))
    later_run()
    (tmp_path / "runs" / f"{run}.json").unlink()                    # its history is gone
    view.discard_button.click()
    assert (sorted(kept()), view.notice.text(), view.keep_block.isVisibleTo(view)) == (
        sorted([second, fourth]), "Nothing was discarded: the run's history that names the training cannot be read.",
        True), "a training whose run's history cannot be read was said to be discarded"

    loaded = build_fit_record(store, passes=62)
    run = stopped_in_the_fit(_note(loaded))
    loads = StandIn(word="Posterior load")
    start_run(host, loads, spec=RunSpec("Posterior load", ("inference", "training"), stop_button="Stop loading",
                                        loads_only=True, keeps_files=False, stop_asks=False))
    loads.release()
    _ended(app, held)
    assert (held.status.run, view.mode, view.resume_button.isVisibleTo(view), view.keep_block.isVisibleTo(view)) == (
        run, "ended", True, True), "a load of the page's own took the stopped run's end, its page or its offer"
    view.discard_button.click()
    gone = discarded("Fri 9 Oct, 01:20")
    assert (sorted(kept()), view.sentence.text(), held.status.sentence, ends.foot(held.status, clock()),
            chain.next_step(None, {}, run=held.status).sentence) == (
        sorted([second, fourth]), gone, gone, ("half", "Training stopped", "Fri 9 Oct, 01:20. The 20,480,000 "
                                               "simulations are kept; the training was discarded, as you asked."),
        gone), "after a load, the page, the status, the foot or Home still say the training can be resumed"

    meanwhile = build_fit_record(store, passes=62)
    stopped_in_the_fit(_note(meanwhile))
    later = StandIn()
    start_run(build_run_host(), later, spec=RunSpec("Check", ("inference", "checks"), keeps_files=False))
    view.discard_button.click()                                      # while a run of another page is live
    later.release()
    _ended(app, held)
    assert (sorted(kept()), view.sentence.text(), held.status.sentence) == (
        sorted([second, fourth]), discarded("Fri 9 Oct, 02:23"), discarded("Fri 9 Oct, 02:23")), (
        "a run live at the Discard gave back, as it ended, the end's words that say the training is kept")


def test_discard_of_a_resumed_training_says_it_deletes_every_epoch_kept_since_it_began(tmp_path, monkeypatch, store):
    """A training that was resumed and kept nothing new in this process before it was stopped is the earlier
    training as it was: the question says that Discard deletes the whole of it, its epochs and the sessions that
    saved them, from the note; one resumed that saved, the same. Discard deletes it."""
    app = qt_app()
    clock = Clock(_at(8, 19, 2, 11))
    held, host = _controller_on(tmp_path, clock), build_run_host()
    text, view = RunText(held), host.run_view
    choose(monkeypatch, words="Stop training")
    for passes, saved_here, said in (
            (62, False, "Nothing new was saved since it was resumed. Discard deletes the whole training: 62 epochs, "
                        "saved over 2 sessions."),
            (64, True, "Discard deletes the whole training: 64 epochs, saved over 2 sessions.")):
        resumed = build_fit_record(store, passes=passes)
        stand_in = StandIn()
        stand_in.kept_fit = _note(resumed, passes=passes, resumed=True, saved_here=saved_here, processes=2)
        start_run(host, stand_in, spec=_training())
        _to_the_fit(app, stand_in, text, clock, _THREE_EPOCHS)
        view.stop_button.click()
        _ended(app, held)
        assert (view.keep_block.isVisibleTo(view), view.keep_words.text(), view.keep_beneath.text()) == (
            True, "Keep this training so you can resume it later?", said)
        view.discard_button.click()
        assert [summary.id for summary in store.list("fit")] == [] and view.notice.text() == (
            "The training was discarded.")
    assert questions.keep_question(_note("x", passes=1)) == (
        "Keep this training so you can resume it later?", "It stopped after epoch 1; the 1 epoch is kept until you "
                                                          "discard it.")


# ── the keyboard ──
def test_the_keyboard_stays_on_the_runs_page_through_a_stop(tmp_path, monkeypatch, store):
    """After a start made with a click the keyboard is on the run page's stop button, and Space there asks first:
    unanswered, nothing is stopped. A stop answered with Space hands the keyboard to "Don't stop after all", not
    out of the page; Space there withdraws the stop and hands it back to the stop button. Once a stop has landed
    the keyboard goes to "Keep" while the page asks about the training the stop kept, and from there, answered, to
    the page's sentence; with nothing kept, to the sentence at once. A keyboard on a control that stays at the end
    (the full text's own) stays there. While the run waits to be tried again, the keyboard on the stop button goes to
    "Stop trying", which stands in its place, and comes back to it when the next try begins."""
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication, QPushButton
    from tests._window_helpers import in_front, shown_off_screen
    app = qt_app()
    clock = Clock(_at(8, 19, 2, 11))
    held, host = _controller_on(tmp_path, clock), build_run_host()
    text, view = RunText(held), host.run_view
    start, runs = QPushButton("Start the run"), []
    host.controls_layout.addWidget(start)
    start.clicked.connect(lambda: start_run(host, runs[-1], spec=_training()))
    shown_off_screen(host, 1000, 700)

    def focus():
        return QApplication.focusWidget()

    with in_front(host):
        runs.append(StandIn(takes_stops=False))
        runs[-1].kept_fit = _note(build_fit_record(store, passes=62))
        QTest.mouseClick(start, Qt.LeftButton)          # a click: the keyboard goes to the stop button
        assert focus() is view.stop_button
        _to_the_fit(app, runs[-1], text, clock, _THREE_EPOCHS)
        forget_asked()
        QTest.keyClick(view.stop_button, Qt.Key_Space)
        assert ([box.title for box in all_asked()], held.status.stop_requested, focus()) == (
            ["Stop training?"], False, view.stop_button), "Space on the stop button stopped the run with no question"
        choose(monkeypatch, words="Stop training")
        QTest.keyClick(view.stop_button, Qt.Key_Space)
        assert (held.status.stop_requested, focus()) == (True, view.withdraw_button), "the keyboard left the page"
        QTest.keyClick(view.withdraw_button, Qt.Key_Space)
        assert (held.status.stop_requested, focus()) == (False, view.stop_button)
        QTest.keyClick(view.stop_button, Qt.Key_Space)
        runs[-1].say("the line at which the stop is taken")
        _ended(app, held)
        assert focus() is view.keep_button, "the keyboard did not go to Keep"
        QTest.keyClick(view.keep_button, Qt.Key_Space)
        assert focus() is view.sentence, "the keyboard did not go to the page's sentence once Keep was answered"

        view.back_button.click()
        runs.append(StandIn())
        QTest.mouseClick(start, Qt.LeftButton)          # a click: the keyboard goes to the stop button
        _to_the_simulation(app, runs[-1], text, held)
        choose(monkeypatch, words="Stop and keep the simulations")
        QTest.keyClick(view.stop_button, Qt.Key_Space)
        _ended(app, held)
        assert focus() is view.sentence, "with nothing kept, the keyboard did not go to the page's sentence"

        view.back_button.click()
        runs.append(StandIn())
        QTest.mouseClick(start, Qt.LeftButton)          # a click: the keyboard goes to the stop button
        view.text_button.click()
        view.find_button.click()
        assert focus() is view.find_field
        runs[-1].fail(RuntimeError("the run broke"))
        _ended(app, held)
        assert focus() is view.find_field, "the end took the keyboard from a control that stays"

        view.back_button.click()
        runs.append(StandIn())
        QTest.mouseClick(start, Qt.LeftButton)          # a click: the keyboard goes to the stop button
        runs[-1].fail(RuntimeError("CUDA error: out of memory"))         # a failure the window tries again
        turn_until(app, lambda: held.status.state == "retrying")
        assert focus() is view.stop_trying_button, "while the run waits, the keyboard did not go to 'Stop trying'"
        clock.advance(60)
        held.tick()
        turn_until(app, lambda: len(runs[-1].calls) == 2 and held.status.state == "running")
        assert focus() is view.stop_button, "at the try, the keyboard did not come back to the stop button"
        runs[-1].release()
        _ended(app, held)


# ── closing during a run ──
def test_closing_during_a_run_asks_and_keeps_running_by_default(tmp_path, monkeypatch):
    """Closing the window while a run is live asks first, naming the run, how long it has run, where it is and what
    a stop keeps; "Keep running", the safe answer, is what a question nobody answers gives: the window stays, and
    the run goes on. With a stop asked for already, it asks whether to close once the run has stopped, "Wait" the
    safe answer. The second answer is drawn as losing work where the stop's own is. A training in its simulation
    is said to be there, at the batch it has reached; a measurement says once how long it has run."""
    from core.gui.panels.reduction_panel import ReductionPanel
    from tests._window_helpers import shown_off_screen
    app = qt_app()
    clock = Clock(_at(8, 19, 2, 11))
    held, window = _controller_on(tmp_path, clock), build_window()
    shown_off_screen(window)
    host, text = window.panel(ReductionPanel), RunText(held)
    stand_in = StandIn(takes_stops=False)
    start_run(host, stand_in, spec=_training())
    _say(app, stand_in, text, _BUDGET, _ROTATING, _ROTATED, _WRITING, _COMPLETE)
    clock.advance(62 * 3780)                          # 62 epochs of 63 minutes: 2 days 17 hours
    _say(app, stand_in, text, *(_epoch(epoch, epoch % 5) for epoch in range(1, 63)))
    forget_asked()
    window.close()
    question = asked()
    assert (question.title, question.sentence, question.beneath) == (
        "Close PRISM?", "Training retrain_3 is running.",
        "It has been running for 2 days 17 hours and is in its fit, at epoch 62. The simulations are kept. The fit "
        "stops when the epoch in progress ends, up to about 63 minutes from now; every epoch finished so far is kept.")
    assert (question.answers, question.default, question.safe, question.loses_work) == (
        ("Stop the run and close", "Keep running"), "Keep running", "Keep running", False)
    assert (window.isVisible(), held.is_live(), held.status.stop_requested, held.status.closing) == (
        True, True, False, False), "a close nobody answered closed the window or stopped the run"
    choose(monkeypatch, words="Stop training")
    host.run_view.stop_button.click()
    choose(monkeypatch)                               # the guard again: nobody answers
    window.close()
    question = asked()
    assert (question.sentence, question.beneath, question.answers, question.default, question.loses_work) == (
        "Training retrain_3 is stopping.", _STOPPING_FIT, ("Close when it has stopped", "Wait"), "Wait", False)
    assert (window.isVisible(), held.is_live(), held.status.closing) == (True, True, False)
    stand_in.say("the line at which the stop is taken")
    _ended(app, held)
    check = StandIn()
    start_run(host, check, spec=RunSpec("Check", ("inference", "checks"), name="Calibration",
                                        stop_button="Stop the calibration…"))
    window.close()
    assert (asked().sentence, asked().beneath, asked().loses_work) == (
        "Check Calibration is running.", "It has been running for less than a minute. A check writes its record only "
                                         "when it finishes, so nothing of this one is kept.", True)
    check.release()
    _ended(app, held)
    measuring = StandIn()
    start_run(host, measuring, spec=RunSpec("Measurement", ("temperature", "one_cell"), name="cell_a_3oct",
                                            stop_button="Stop measurement…"))
    clock.advance(4_320)
    window.close()
    assert (asked().sentence, asked().beneath, asked().loses_work) == (
        "Measurement cell_a_3oct is running.", "It has been running for 1 hour 12 minutes. The record cell_a_3oct is "
                                               "kept, marked unfinished, with what it has measured so far. A "
                                               "measurement cannot be continued; to finish it, it is measured again "
                                               "from the start.", True)
    measuring.release()
    _ended(app, held)
    simulating = StandIn()
    start_run(host, simulating, spec=_training())
    _to_the_simulation(app, simulating, text, held)
    window.close()
    assert (asked().sentence, asked().beneath, asked().loses_work) == (
        "Training retrain_3 is running.", "It has been running for less than a minute and is in its simulation, at "
                                          "batch 6,150. The 6,150 batches simulated so far are kept; the batch in "
                                          "progress is not. You can continue from batch 6,150.", False)
    simulating.release()
    _ended(app, held)
    window.hide()


def test_stop_the_run_and_close_waits_for_the_end_then_closes_every_window(tmp_path, monkeypatch):
    """"Stop the run and close" asks the run to stop and keeps the window open, on the run's page, which says that
    PRISM closes when the run has stopped; "Don't stop after all" withdraws the stop and the close both. Asked again,
    the window waits: when the run has ended its history is written, then the one-PRISM claim is let go, the figure
    window a page opened is closed, and the window closes, once all of the run's end has been said: its history, its
    end heard and its page drawn ended. The whole program then leaves no run to acknowledge and no close waiting for
    one."""
    from core.gui import one_instance
    from tests._fixtures import whole_program
    from tests._window_helpers import shown_off_screen
    app = qt_app()
    monkeypatch.setenv("PRISM_ARTIFACTS", str(tmp_path / "Artifacts"))
    clock = Clock(_at(8, 19, 2, 11))
    with whole_program() as (_app, window):
        held = _controller_on(tmp_path, clock)
        claim = one_instance.current()
        assert claim is not None, "the whole program holds no claim"
        shown_off_screen(window)
        host, text = window.part("inference").posterior_panel, RunText(held)
        host.figure_stack.add_figure("a figure", _picture())
        host.figure_stack._pop_out(host.figure_stack.widget(0))
        figure = next(iter(host.figure_stack._windows))
        stand_in = StandIn(takes_stops=False)
        start_run(host, stand_in, spec=_training())
        _say(app, stand_in, text, _BUDGET)
        view = host.run_view
        choose(monkeypatch, words="Stop the run and close")
        window.close()
        assert (window.isVisible(), held.status.state, held.status.closing, one_instance.current() is claim) == (
            True, "stopping", True, True)
        assert (view.closing_line.text(), view.closing_line.isVisibleTo(view),
                view.withdraw_button.isVisibleTo(view)) == (
            "PRISM closes when the run has stopped.", True, True)
        view.withdraw_button.click()
        assert (held.status.state, held.status.closing, window._close_at_end, view.closing_line.isVisibleTo(view),
                window.isVisible()) == (
            "running", False, False, False, True), "'Don't stop after all' did not withdraw the close"
        _say(app, stand_in, text, "a line after the withdraw")
        assert held.is_live() and window.isVisible()

        window.close()
        app.processEvents()                           # what the run said before is heard
        clock.advance(SILENCE_NEEDS_YOU_S + 1)        # then a long silence while the run stops: its status changes
        held.tick()
        silent = held.status.needs_you
        app.processEvents()
        assert (silent is not None, window.isVisible(), one_instance.current() is claim) == (True, True, True), (
            "the window closed, or let go of the claim, before its stopping run had ended")
        released = _note_the_release(monkeypatch, tmp_path / "runs", held.status.run, text, view)
        stand_in.say("the line at which the stop is taken")
        turn_until(app, lambda: not window.isVisible(), 10)
        assert (released, one_instance.current(), _open(figure), held.is_live()) == (
            [("stopped", ["stopped"], "ended")], None, False, False), (
            "the claim was let go before the run's history, its end and its page said how it ended, or a window "
            "outlived the close")
    assert (held.is_live(), held.status.state, held.status.closing) == (False, "idle", False)


def test_the_window_does_not_close_under_a_retry_pause_or_a_pending_stop(tmp_path, monkeypatch):
    """A run starting, running, waiting to be tried again or stopping is live: a close asked for meanwhile is
    refused, whatever is answered short of the close's second answer, and the window keeps the one-PRISM claim; a
    pause that runs out while the close's question is open lets the next try begin, and the answer that keeps the
    run changes nothing; a try of a run that saved no simulation yet is the run's own call as it was; a stopping run's
    close waits for its end, the claim held until then."""
    from core.gui import one_instance
    from tests._fixtures import whole_program
    from tests._window_helpers import shown_off_screen
    app = qt_app()
    monkeypatch.setenv("PRISM_ARTIFACTS", str(tmp_path / "Artifacts"))
    with whole_program() as (_app, window):
        clock = Clock(_at(8, 19, 2, 11))
        held = _controller_on(tmp_path, clock)
        claim = one_instance.current()
        shown_off_screen(window)
        host, text = window.part("inference").posterior_panel, RunText(held)
        stand_in, seen = StandIn(takes_stops=False), []

        def closed_now():
            window.close()
            seen.append((held.status.state, window.isVisible(), one_instance.current() is claim))

        tried = StandIn()
        start_run(host, tried, spec=_training())
        _say(app, tried, text, "the line the run says before it runs out of memory")
        tried.fail(RuntimeError("CUDA error: out of memory"))          # a failure the window tries again
        turn_until(app, lambda: held.status.state == "retrying")
        closed_now()                                 # nobody answers
        assert (asked().sentence, asked().beneath.endswith("No simulation is saved yet."), seen) == (
            "Training retrain_3 is waiting to try again.", True, [("retrying", True, True)]), (
            "the window closed under a retry's pause, or its question does not say that nothing is saved yet")

        def the_pause_runs_out():
            clock.advance(60)
            held.tick()

        choose(monkeypatch, words="Keep running", during=the_pause_runs_out)
        closed_now()
        turn_until(app, lambda: len(tried.calls) == 2)
        assert (seen[-1], held.is_live(), held.status.stop_requested) == (("running", True, True), True, False), (
            "a pause that ran out while the close's question was open stopped the run or closed the window")
        assert tried.calls[1] == tried.calls[0], (
            "a try of a run whose set held no batch did not repeat its call as it was")
        tried.release()
        _ended(app, held)
        seen.clear()
        choose(monkeypatch)                          # the guard again: nobody answers

        start_run(host, stand_in, spec=_training())
        closed_now()                                 # nobody answers
        _say(app, stand_in, text, _BUDGET)
        closed_now()
        choose(monkeypatch, words="Stop training")
        host.run_view.stop_button.click()
        choose(monkeypatch, words="Wait")
        closed_now()
        assert seen == [("starting", True, True), ("running", True, True), ("stopping", True, True)], (
            "the window closed, or let go of the claim, under a live run")
        choose(monkeypatch, words="Close when it has stopped")
        closed_now()
        app.processEvents()
        assert (seen[-1], held.status.closing, window.isVisible(), one_instance.current() is claim) == (
            ("stopping", True, True), True, True, True), "the window did not wait for a stopping run"
        stand_in.say("the line at which the stop is taken")
        _ended(app, held)
        turn_until(app, lambda: one_instance.current() is None and not window.isVisible(), 10)


def test_an_answer_given_after_the_run_ended_is_taken_as_too_late(tmp_path, monkeypatch):
    """A run that ends while one of its questions is open: the training finishes by itself while "Stop training?" is
    open, and the run fails while the close's question is open. The answer, given after, is too late: nothing raises,
    nothing is stopped, the close is not carried out, and the page shows the run's end and says the stop came too
    late."""
    from core.gui.panels.reduction_panel import ReductionPanel
    from tests._window_helpers import shown_off_screen
    app = qt_app()
    held, window = _controller_on(tmp_path, Clock(_at(8, 19, 2, 11))), build_window()
    shown_off_screen(window)
    host, text = window.panel(ReductionPanel), RunText(held)
    view = host.run_view
    for how, answer, ask in (("finishes", "Stop training", view.stop_button.click),
                             ("fails", "Stop the run and close", window.close)):
        stand_in, done = StandIn(), []

        def ends(stand_in=stand_in, how=how, done=done):
            if done:
                return
            done.append(how)
            if how == "finishes":
                stand_in.release()
            else:
                stand_in.fail(RuntimeError("the run broke"))
            _ended(app, held)

        start_run(host, stand_in, spec=_training())
        _say(app, stand_in, text, _BUDGET)
        choose(monkeypatch, words=answer, during=ends)
        ask()
        outcome = "finished" if how == "finishes" else "failed"
        assert (done, [status.outcome for status in text.ended][-1], held.is_live()) == ([how], outcome, False)
        assert (window.isVisible(), window._close_at_end, held.status.closing) == (True, False, False), (
            "a close answered after the run ended was carried out")
        assert (view.notice.text(), "The run stopped: a stop was asked for." in text.texts()) == (
            "Too late: the run had already finished.", False), how
    assert (view.mode, view.sentence.text()) == ("ended", "Training retrain_3 failed.")
    window.hide()


def test_a_close_while_the_posterior_is_saved_stops_nothing_and_closes_once_it_is_saved(tmp_path, monkeypatch):
    """While a training saves its posterior, which no stop interrupts, the close asks whether to close once it is
    saved: "Keep running" the safe answer, "Close when it is saved" not drawn as losing work. That answer asks no stop
    of the run, its token never set, and keeps the window open on the run's page, which says that PRISM closes when
    the posterior is saved and offers "Keep PRISM open" where a stop offers "Don't stop after all". "Keep PRISM open",
    pressed with Space, withdraws the close and hands the keyboard to the page's sentence: the run finishes, the
    window stays, and what the page said goes with the run's end. Asked again, the window waits: the run finishes,
    all of its end said, then the one-PRISM claim is let go, the figure window a page opened is closed, and the
    window closes."""
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication
    from core.gui import one_instance
    from tests._fixtures import whole_program
    from tests._window_helpers import in_front, shown_off_screen
    app = qt_app()
    monkeypatch.setenv("PRISM_ARTIFACTS", str(tmp_path / "Artifacts"))
    clock = Clock(_at(8, 19, 2, 11))
    with whole_program() as (_app, window):
        held = _controller_on(tmp_path, clock)
        claim = one_instance.current()
        shown_off_screen(window)
        host = window.part("inference").posterior_panel
        view = host.run_view
        host.figure_stack.add_figure("a figure", _picture())
        host.figure_stack._pop_out(host.figure_stack.widget(0))
        figure = next(iter(host.figure_stack._windows))

        def saving():
            """A training brought to the saving of its posterior, its fit ended after two epochs, its lines heard by a
            text of its own; a stop asked of it would be taken at once."""
            stand_in, text = StandIn(), RunText(held)
            start_run(host, stand_in, spec=_training())
            _to_the_fit(app, stand_in, text, clock, [(1, 0), (2, 20)])
            assert held.model.stage == "Saving"
            return stand_in, held._run.token, text

        stand_in, token, text = saving()
        forget_asked()
        window.close()                                # nobody answers
        question = asked()
        assert (question.title, question.sentence, question.beneath) == (
            "Close PRISM?", "Training retrain_3 is running.",
            "It has been running for 1 hour 3 minutes and is saving its posterior.")
        assert (question.answers, question.default, question.loses_work) == (
            ("Close when it is saved", "Keep running"), "Keep running", False)
        assert (window.isVisible(), held.status.closing, token.requested.is_set()) == (True, False, False)
        choose(monkeypatch, words="Close when it is saved")
        window.close()
        assert (window.isVisible(), held.status.state, held.status.closing, token.requested.is_set(),
                window._close_at_end) == (True, "running", True, False, True), (
            "the close during the save asked a stop, or the window did not wait")
        assert (view.closing_line.text(), view.closing_line.isVisibleTo(view), view.keep_open_button.isVisibleTo(view),
                view.withdraw_button.isVisibleTo(view), view.stop_button.isEnabled()) == (
            "PRISM closes when the posterior is saved.", True, True, False, False)
        with in_front(window):
            view.keep_open_button.setFocus(Qt.OtherFocusReason)
            assert QApplication.focusWidget() is view.keep_open_button
            QTest.keyClick(view.keep_open_button, Qt.Key_Space)
            focus = QApplication.focusWidget()
        assert (held.status.closing, window._close_at_end, view.closing_line.isVisibleTo(view),
                view.keep_open_button.isVisibleTo(view), view.notice.text()) == (
            False, False, False, False, "PRISM stays open."), "'Keep PRISM open' did not withdraw the close"
        assert focus is view.sentence, "the keyboard did not go to the page's sentence once the close was withdrawn"
        stand_in.release()
        _ended(app, held)
        app.processEvents()
        assert (window.isVisible(), one_instance.current() is claim, token.requested.is_set(),
                [status.outcome for status in text.ended][-1], view.notice.text()) == (
            True, True, False, "finished", ""), (
            "a withdrawn close closed the window, the save was stopped, or the page still said PRISM stays open")

        stand_in, token, text = saving()
        window.close()                                # answered "Close when it is saved" again
        released = _note_the_release(monkeypatch, tmp_path / "runs", held.status.run, text, view)
        stand_in.release()
        turn_until(app, lambda: not window.isVisible(), 10)
        assert (released, token.requested.is_set(), one_instance.current(), _open(figure), held.is_live()) == (
            [("finished", ["finished"], "ended")], False, None, False, False), (
            "the save was stopped, the claim was let go before the run's history, its end and its page said it "
            "finished, or a window outlived the close")
    assert (held.is_live(), held.status.state, held.status.closing) == (False, "idle", False)


def test_an_answer_given_once_the_save_has_begun_stops_nothing(tmp_path, monkeypatch):
    """A question asked during the fit and answered once the fit has ended and the posterior is being saved: the stop
    button's "Stop training" stops nothing, and the page says that the save cannot be stopped; the close's "Stop the
    run and close" stops nothing either, and the window waits for the save, then closes."""
    from core.gui import one_instance
    from tests._fixtures import whole_program
    from tests._window_helpers import shown_off_screen
    app = qt_app()
    monkeypatch.setenv("PRISM_ARTIFACTS", str(tmp_path / "Artifacts"))
    clock = Clock(_at(8, 19, 2, 11))
    with whole_program() as (_app, window):
        held = _controller_on(tmp_path, clock)
        shown_off_screen(window)
        host = window.part("inference").posterior_panel
        view = host.run_view
        for answer, ask in (("Stop training", view.stop_button.click), ("Stop the run and close", window.close)):
            stand_in, text = StandIn(), RunText(held)
            start_run(host, stand_in, spec=_training())
            _to_the_fit(app, stand_in, text, clock, [(1, 0)])
            token = held._run.token

            def saves(stand_in=stand_in):
                _say(app, stand_in, text, _epoch(2, 20))     # 20 epochs since the best: the fit ends, the save begins

            choose(monkeypatch, words=answer, during=saves)
            ask()
            assert (held.model.stage, held.status.stop_requested, token.requested.is_set()) == (
                "Saving", False, False), f"{answer!r}, answered once the save had begun, asked a stop"
            if answer == "Stop training":
                assert (view.notice.text(), held.status.closing, window._close_at_end) == (
                    "Too late: the posterior was already being saved, which cannot be stopped.", False, False)
                stand_in.release()
                _ended(app, held)
                assert [status.outcome for status in text.ended][-1] == "finished"
        assert (held.status.closing, window._close_at_end, view.closing_line.text(),
                view.keep_open_button.isVisibleTo(view)) == (
            True, True, "PRISM closes when the posterior is saved.", True), "the window did not wait for the save"
        stand_in.release()
        turn_until(app, lambda: not window.isVisible(), 10)
        assert ([status.outcome for status in text.ended][-1], token.requested.is_set(), one_instance.current()) == (
            "finished", False, None), "the save was stopped, or the window did not close once it was saved"


def test_a_close_that_waits_is_not_carried_out_when_the_run_fails(tmp_path, monkeypatch):
    """A close that waits for its run is carried out only once the run has stopped or finished. A run that fails
    instead, after "Stop the run and close", and a save that fails after "Close when it is saved", leave the window
    open on the run's page, which says the failure, with no box, and that PRISM stayed open, the event loop turning:
    nothing closes the window, and the one-PRISM claim is held. So does "Stop trying" after a save that ran out of
    memory under "Close when it is saved", a close that did not end the tries. A question of the person's own that is
    open when a stopped run ends is answered first; the window then closes."""
    import time
    from core.gui import one_instance
    from core.gui.widgets.ask import ask
    from tests._fixtures import whole_program
    from tests._window_helpers import shown_off_screen
    app = qt_app()
    monkeypatch.setenv("PRISM_ARTIFACTS", str(tmp_path / "Artifacts"))
    clock = Clock(_at(8, 19, 2, 11))

    def turned_past_the_windows_look():
        """The event loop turned for twice the time the window waits between two looks for an open box: a close that
        waited for such a box has had its turn."""
        from core.gui.main_window import _ANSWERED_EVERY_MS
        began = time.monotonic()
        turn_until(app, lambda: time.monotonic() - began >= 2 * _ANSWERED_EVERY_MS / 1000)

    with whole_program() as (_app, window):
        held = _controller_on(tmp_path, clock)
        claim = one_instance.current()
        shown_off_screen(window)
        host = window.part("inference").posterior_panel
        view = host.run_view
        for answer, broke in (("Stop the run and close", "the run broke"),
                              ("Close when it is saved", "the posterior could not be written")):
            stand_in, text = StandIn(takes_stops=False), RunText(held)
            start_run(host, stand_in, spec=_training())
            if answer == "Close when it is saved":
                _to_the_fit(app, stand_in, text, clock, [(1, 0), (2, 20)])      # the fit has ended: the save
            else:
                _say(app, stand_in, text, _BUDGET)
            choose(monkeypatch, words=answer)
            window.close()
            assert (window._close_at_end, held.status.closing) == (True, True), answer
            forget_asked()
            stand_in.fail(RuntimeError(broke))
            _ended(app, held)
            _settled(app)
            turned_past_the_windows_look()
            assert (all_asked(), view.failure.isVisible(), view.failure_message.text()) == ([], True, broke), (
                f"{answer!r}: the failure was told in a box, or not on the run's page")
            assert (window.isVisible(), one_instance.current() is claim, window._close_at_end, window.current_place(),
                    view.mode, view.sentence.text(), view.notice.text(), view.notice.isVisible()) == (
                True, True, False, ("inference", "training"), "ended", "Training retrain_3 failed.",
                "PRISM stayed open: the run failed.", True), answer
            view.back_button.click()

        stand_in, text = StandIn(takes_stops=False), RunText(held)
        start_run(host, stand_in, spec=_training())
        _to_the_fit(app, stand_in, text, clock, [(1, 0), (2, 20)])          # the fit has ended: the save
        choose(monkeypatch, words="Close when it is saved")
        window.close()
        stand_in.fail(RuntimeError("CUDA error: out of memory"))             # the save is tried again
        turn_until(app, lambda: held.status.state == "retrying")
        view.stop_trying_button.click()
        _ended(app, held)
        _settled(app)
        turned_past_the_windows_look()
        assert (window.isVisible(), one_instance.current() is claim, view.notice.text(), view.notice.isVisible()) == (
            True, True, "PRISM stayed open: the run failed.", True), (
            "'Stop trying' under a close that waited for the save closed the window, or the page did not say so")
        view.back_button.click()

        stand_in, text = StandIn(takes_stops=False), RunText(held)
        start_run(host, stand_in, spec=_training())
        _say(app, stand_in, text, _BUDGET)
        choose(monkeypatch, words="Stop the run and close")
        window.close()
        seen = []

        def ends():
            stand_in.say("the line at which the stop is taken")
            _ended(app, held)

        hold_open(monkeypatch, seen, during=ends)
        released = _note_the_release(monkeypatch, tmp_path / "runs", held.status.run, text, view)
        ask(window, "A question of its own", "Asked while the run stops.", "Cancel", "Go on")
        assert (seen, released) == ([(True, True)], []), (
            "the close the run's end made closed a question still open, or the window with it")
        turn_until(app, lambda: not window.isVisible(), 10)
        assert released == [("stopped", ["stopped"], "ended")], (
            "the window did not close once the question was answered")


def test_a_yes_to_the_close_shows_the_runs_page_wherever_the_close_was_asked(tmp_path, monkeypatch):
    """Whatever place is in sight when the window is closed during a run, a yes to the close -- "Stop the run and
    close", "Close when it has stopped", "Close when it is saved" -- brings the run's page into sight, where its line
    says that PRISM closes when the run has stopped, or when the posterior is saved, and where the close can be
    withdrawn."""
    from tests._window_helpers import shown_off_screen
    app = qt_app()
    clock = Clock(_at(8, 19, 2, 11))
    held, window = _controller_on(tmp_path, clock), build_window()
    shown_off_screen(window)
    host, text = window.part("inference").posterior_panel, RunText(held)
    view = host.run_view

    def closed_from(place, answer):
        window.go_to(*place)
        choose(monkeypatch, words=answer)
        window.close()
        return window.current_place(), view.closing_line.isVisible(), view.closing_line.text()

    stand_in = StandIn(takes_stops=False)
    start_run(host, stand_in, spec=_training())
    _say(app, stand_in, text, _BUDGET)
    assert closed_from(("home",), "Stop the run and close") == (
        ("inference", "training"), True, "PRISM closes when the run has stopped."), "the run's page is not in sight"
    assert view.withdraw_button.isVisible()
    view.withdraw_button.click()
    choose(monkeypatch, words="Stop training")
    view.stop_button.click()
    assert closed_from(("records",), "Close when it has stopped") == (
        ("inference", "training"), True, "PRISM closes when the run has stopped."), "the run's page is not in sight"
    assert view.withdraw_button.isVisible()
    view.withdraw_button.click()
    stand_in.release()
    _ended(app, held)

    saving = StandIn()
    start_run(host, saving, spec=_training())
    _to_the_fit(app, saving, text, clock, [(1, 0), (2, 20)])
    assert closed_from(("temperature", "one_cell"), "Close when it is saved") == (
        ("inference", "training"), True, "PRISM closes when the posterior is saved."), "the run's page is not in sight"
    assert view.keep_open_button.isVisible()
    view.keep_open_button.click()
    saving.release()
    _ended(app, held)
    window.hide()


def test_a_close_while_a_close_waits_asks_nothing_and_the_window_closes_at_the_end(tmp_path, monkeypatch):
    """A close of the window while a close already waits for its run -- for the save, or for a stop -- asks nothing
    again and brings the run's page into sight, so that no question stands open to take the close the run's end
    makes: when the run ends, even where a question would have stayed open past its end, the window closes."""
    from core.gui import one_instance
    from tests._fixtures import whole_program
    from tests._window_helpers import shown_off_screen
    app = qt_app()
    monkeypatch.setenv("PRISM_ARTIFACTS", str(tmp_path / "Artifacts"))
    clock = Clock(_at(8, 19, 2, 11))
    with whole_program() as (_app, window):
        held = _controller_on(tmp_path, clock)
        shown_off_screen(window)
        host = window.part("inference").posterior_panel
        view = host.run_view
        saving, text = StandIn(), RunText(held)
        start_run(host, saving, spec=_training())
        _to_the_fit(app, saving, text, clock, [(1, 0), (2, 20)])
        choose(monkeypatch, words="Close when it is saved")
        window.close()
        window.go_to("home")
        forget_asked()
        window.close()
        assert (all_asked(), window.current_place(), window._close_at_end) == (
            [], ("inference", "training"), True), "a close while the save's close waited asked again"
        view.keep_open_button.click()
        saving.release()
        _ended(app, held)
        app.processEvents()
        assert window.isVisible()

        stand_in = StandIn(takes_stops=False)
        start_run(host, stand_in, spec=_training())
        _say(app, stand_in, text, _BUDGET)
        choose(monkeypatch, words="Stop the run and close")
        window.close()
        window.go_to("home")

        def ends():
            """The run ends while the question stands open, and the event loop turns on before it is answered."""
            stand_in.say("the line at which the stop is taken")
            _ended(app, held)
            _settled(app)

        choose(monkeypatch, words="Wait", during=ends)
        forget_asked()
        window.close()
        assert (all_asked(), window.current_place()) == ([], ("inference", "training")), (
            "a close while the stop's close waited asked again")
        stand_in.say("the line at which the stop is taken")
        turn_until(app, lambda: not window.isVisible(), 10)
        assert (one_instance.current(), held.is_live()) == (None, False)


def test_the_end_of_a_test_stops_a_training_it_left_live_in_its_save(tmp_path):
    """The end of a test stops a run it left live (tests/_built.py's end of a test) even while a training saves its
    posterior, which takes no stop asked through the controller: the run's own token is set, the stand-in takes the
    stop at its next line, and the end is handled within the end's wait, so that no later run is refused."""
    from tests import _built
    from tests._runs import STOP_LINE
    app = qt_app()
    clock = Clock(_at(8, 19, 2, 11))
    held, host = _controller_on(tmp_path, clock), build_run_host()
    at_start = _built.noted()
    stand_in, text = StandIn(), RunText(held)
    start_run(host, stand_in, spec=_training())
    _to_the_fit(app, stand_in, text, clock, [(1, 0), (2, 20)])
    assert held.model.stage == "Saving"
    with pytest.raises(AssertionError, match="ended with a run still live; it was asked to stop, and its end was"):
        _built.end_of_test(at_start)
    assert (stand_in.stopped_at, held.is_live()) == (STOP_LINE, False), "the end of the test did not stop the save"


# ── Windows' end of a session: its question, "shut down anyway" and "cancel" ──
# The watch on Windows' end of a session as its module defines it, read as this file is loaded: in a test not marked
# for the real screen the session's guard stands in for it.
WATCH_SESSION_END = this_computer.watch_session_end


class _Manager:
    """The toolkit's session manager, as Windows' question hands it to the window: how often it was cancelled."""

    def __init__(self):
        self.cancelled = 0

    def cancel(self):
        self.cancelled += 1


def _windows_log(caplog) -> list:
    """What the window said in its log: each line, its level, and the thread it was said on."""
    return [(record.getMessage(), record.levelname, record.threadName) for record in caplog.records
            if record.name == "core.gui.main_window"]


def _histories_written(monkeypatch, held) -> list:
    """Each write of a run's history from now on, as the outcome, the stop's reason and the end it wrote, the outcome
    and the end it wrote of the run's last try, and whether the run was live as it was written; the list they are put
    in."""
    from core.gui.run import history
    written, write = [], history.write

    def noted(folder, record):
        last = record["tries"][-1] if record["tries"] else {}
        written.append((record["outcome"], record["stop_reason"], record["ended"], last.get("outcome"),
                        last.get("ended"), held.is_live()))
        return write(folder, record)

    monkeypatch.setattr(history, "write", noted)
    return written


def test_from_the_moment_windows_asks_no_question_is_shown(tmp_path, caplog):
    """When Windows asks whether the session may end, for a sign-out or a shutdown, the window says so in its log
    and, while a run is live, answers Windows that PRISM blocks the end; from then on its close asks nothing. The
    toolkit closes a shown window inside Windows' end message: the close is refused while a run is live, so that
    PRISM and its run go on until Windows ends them, and accepted when none is, with no question either way. With no
    run live, Windows is not answered that PRISM blocks."""
    from core.gui.panels.reduction_panel import ReductionPanel
    from tests._window_helpers import shown_off_screen
    app = qt_app()
    held, window = _controller_on(tmp_path, Clock(_at(8, 19, 2, 11))), build_window()
    shown_off_screen(window)
    host, text, stand_in = window.panel(ReductionPanel), RunText(held), StandIn()
    start_run(host, stand_in, spec=_training())
    _to_the_simulation(app, stand_in, text, held)
    blocked, unblocked = _Manager(), _Manager()
    forget_asked()
    with caplog.at_level(logging.INFO, logger="core.gui.main_window"):
        window.session_asked(blocked)
        window.close()
        assert (blocked.cancelled, all_asked(), window.isVisible(), held.is_live(), held.status.stop_requested) == (
            1, [], True, True, False), "Windows was not answered that PRISM blocks, or the close asked or stopped"
        stand_in.release()
        _ended(app, held)
        window.session_asked(unblocked)
        window.close()
    assert (unblocked.cancelled, all_asked(), window.isVisible()) == (0, [], False), (
        "with no run live Windows was answered that PRISM blocks, or the close asked or was refused")
    assert _windows_log(caplog) == [("Windows asked to end the session.", "INFO", "MainThread")] * 2


def test_shut_down_anyway_stops_the_run_once_and_cancel_never(tmp_path, monkeypatch, caplog):
    """"Cancel" stops nothing: the window's log says the end of the session was cancelled, and a close asks its question
    again. "Shut down anyway" stops the live run as a stop asked by the person does, once however often it is heard,
    during the wait for the run's end too. From that moment the run's history says the run, and the try it is in, was
    stopped, when, and that Windows shut down, with the moment the run was last heard, and its text says the stop was
    asked for that reason, so both hold it should Windows end PRISM before the run has ended; then the window's thread
    waits for the run's end, the event loop turning, and returns once the run has ended, stopped, its history and its
    try then holding the moment it ended. Such a run counts as stopped by the person: Home offers to continue it no more
    than it offers a stop's."""
    from PySide6.QtCore import QTimer
    from core.artifacts import ArtifactStore
    from core.gui.panels.reduction_panel import ReductionPanel
    from core.gui.run import history
    from tests._runs import STOP_LINE
    from tests._window_helpers import shown_off_screen
    app = qt_app()
    clock = Clock(_at(8, 19, 2, 11))
    held, window = _controller_on(tmp_path, clock), build_window()
    shown_off_screen(window)
    folder, host = tmp_path / "runs", window.panel(ReductionPanel)
    check = StandIn()
    start_run(host, check, spec=RunSpec("Check", ("inference", "checks"), name="Calibration",
                                        stop_button="Stop the calibration…"))
    token = held._run.token
    with caplog.at_level(logging.INFO, logger="core.gui.main_window"):
        window.session_asked(_Manager())
        window.session_ending(False)
    assert (held.is_live(), held.status.stop_requested, token.requested.is_set()) == (True, False, False), (
        "a cancelled end of the session stopped the run")
    assert _windows_log(caplog) == [("Windows asked to end the session.", "INFO", "MainThread"),
                                    ("Windows' end of the session was cancelled.", "INFO", "MainThread")]
    forget_asked()
    window.close()
    assert (asked().title, asked().sentence, window.isVisible(), held.is_live()) == (
        "Close PRISM?", "Check Calibration is running.", True, True), "after a cancel the close asked nothing"
    check.release()
    _ended(app, held)

    written = _histories_written(monkeypatch, held)
    heard, noted = [], history.write
    monkeypatch.setattr(history, "write", lambda folder, record: (
        heard.append((record["outcome"], record["last_heard"] is not None, held.is_live())), noted(folder, record))[1])
    text, stand_in = RunText(held), StandIn()
    start_run(host, stand_in, spec=_training())
    _to_the_simulation(app, stand_in, text, held)
    run = held.status.run
    clock.advance(600)
    window.session_asked(_Manager())
    heard_at = clock()
    QTimer.singleShot(0, lambda: (clock.advance(5), window.session_ending(True)))   # heard again while it waits
    window.session_ending(True)
    assert (held.is_live(), stand_in.stopped_at) == (False, STOP_LINE), "the window did not wait for the run's end"
    window.session_ending(True)
    record = history.read(folder / f"{run}.json")
    assert (len(stand_in.calls), [status.outcome for status in text.ended], record["outcome"],
            record["stop_reason"], datetime.fromisoformat(record["ended"]), record["tries"][-1]["outcome"],
            datetime.fromisoformat(record["tries"][-1]["ended"])) == (
        1, ["stopped"], "stopped", "Windows shut down", clock(), "stopped", clock()), (
        "the run's end did not write how and when the run and its try ended")
    assert ("stopped", "Windows shut down", heard_at, "stopped", heard_at, True) in written, (
        "the history did not say the run and its try were stopped, when, and why, while the run was still live")
    assert text.texts("window").count("A stop was asked for: Windows shut down.") == 1
    assert ("stopped", True, True) in heard and (record["last_heard"], heard[-1]) == (
        None, ("stopped", False, False)), (
        "while the stop waited the history did not hold when the run was last heard, or the end kept it")
    assert history.continuable(record, store=ArtifactStore(tmp_path / "records"), folder=folder) is None


def test_shut_down_anyway_while_the_posterior_is_saved_stops_nothing_and_waits_for_the_save(tmp_path, monkeypatch):
    """A training saving its posterior is not stopped, whoever asks, Windows too: "shut down anyway" sets no stop and
    its text says the save goes on, and the window's thread waits for the save as long as Windows lets PRISM live.
    Until the run has ended its history says it, and the try it is in, was stopped because Windows shut down, which is
    what stays should Windows end PRISM first (the finished training is kept, and resuming it writes the posterior);
    once the posterior is saved the run has finished, and its history says so, with no reason for a stop and the
    moment the save ended."""
    from PySide6.QtCore import QTimer
    from core.gui.run import history
    from tests._window_helpers import shown_off_screen
    app = qt_app()
    clock = Clock(_at(8, 19, 2, 11))
    held, window = _controller_on(tmp_path, clock), build_window()
    shown_off_screen(window)
    written = _histories_written(monkeypatch, held)
    host, text, stand_in = build_run_host(), RunText(held), StandIn()
    start_run(host, stand_in, spec=_training())
    _to_the_fit(app, stand_in, text, clock, [(1, 0), (2, 20)])
    assert held.model.stage == "Saving"
    token, run = held._run.token, held.status.run
    clock.advance(30)
    window.session_asked(_Manager())
    heard_at = clock()
    QTimer.singleShot(0, lambda: (clock.advance(5), stand_in.release()))   # the save ends while the window waits
    window.session_ending(True)
    record = history.read(tmp_path / "runs" / f"{run}.json")
    assert (held.is_live(), token.requested.is_set(), stand_in.stopped_at, [status.outcome for status in text.ended],
            record["outcome"], record["stop_reason"], datetime.fromisoformat(record["ended"]),
            record["tries"][-1]["outcome"], datetime.fromisoformat(record["tries"][-1]["ended"])) == (
        False, False, None, ["finished"], "finished", None, clock(), "finished", clock()), (
        "the save was stopped, or the window did not wait for it, or a finished run's history kept the reason, or did "
        "not say when the run and its try ended")
    assert ("stopped", "Windows shut down", heard_at, "stopped", heard_at, True) in written, (
        "until the save ended the history did not say the run and its try were stopped because Windows shut down")
    assert text.texts("window").count(
        "Windows shut down: the posterior is still being saved, which no stop interrupts.") == 1


def test_the_session_watch_hears_shut_down_anyway_apart_from_an_ordinary_quit_and_hears_a_cancel():
    """The watch on Windows' end of a session. The toolkit's quit, heard with no question of Windows before it, is an
    ordinary quit and calls nothing; after Windows' question, the toolkit's session manager handed on, it is "shut
    down anyway". Windows' end message saying the session does not end is "cancel", heard once for its question
    however many of the process's windows Windows sends it to; a quit after it is ordinary again. The filter consumes
    no message, and hands on no other: the end message saying the session ends, Windows' question, any other message,
    another kind of event. The filter is on the application only from Windows' question to its cancel, since a filter
    written in Python is handed every message of the window's thread: once however many questions come, off at the
    cancel, and kept once the session ends. Removed, the watch is off the application. Its signals are raised on a
    stand-in of the application: the one application of the suites is heard by others too (the live view's plotting
    library cleans up as it quits), so on it the watch's connections are only counted, from a reading taken first."""
    import ctypes
    from ctypes import wintypes
    from PySide6.QtCore import QByteArray, QObject, Signal

    class Application(QObject):
        """The application as the watch uses it: its question and its quit, and the native filters it holds."""
        aboutToQuit = Signal()
        commitDataRequest = Signal(object)

        def __init__(self):
            super().__init__()
            self.filters = []

        def installNativeEventFilter(self, one):
            self.filters.append(one)

        def removeNativeEventFilter(self, one):
            self.filters.remove(one)

    app, stand_in = qt_app(), Application()
    asked_with, ending, manager = [], [], _Manager()
    signals = ("2aboutToQuit()", "2commitDataRequest(QSessionManager&)")
    before = [app.receivers(signal) for signal in signals]
    real = this_computer.SessionWatch(app, asked=asked_with.append, ending=ending.append)
    try:
        assert [app.receivers(signal) for signal in signals] == [count + 1 for count in before]
    finally:
        real.remove()
    assert [app.receivers(signal) for signal in signals] == before, "the watch was left on the application"
    watch = this_computer.SessionWatch(stand_in, asked=asked_with.append, ending=ending.append)

    def message(number, wparam=0, kind=b"windows_generic_MSG"):
        msg = wintypes.MSG()
        msg.message, msg.wParam = number, wparam
        return watch.filter.nativeEventFilter(QByteArray(kind), ctypes.addressof(msg))

    try:
        assert stand_in.filters == [], "the filter was handed the application's messages before Windows' question"
        stand_in.aboutToQuit.emit()
        assert (asked_with, ending, stand_in.filters) == ([], [], []), (
            "an ordinary quit was heard as Windows ending the session")
        stand_in.commitDataRequest.emit(manager)       # Windows' question, the toolkit's session manager handed on
        assert stand_in.filters == [watch.filter], "Windows' question did not put the filter on the application"
        stand_in.aboutToQuit.emit()
        assert (asked_with, ending, stand_in.filters) == ([manager], [True], [watch.filter])
        stand_in.commitDataRequest.emit(manager)
        assert stand_in.filters == [watch.filter], "a second question put the filter on the application again"
        assert (message(0x0016), ending, stand_in.filters) == ((False, 0), [True, False], []), (
            "a cancel was not heard, or left the filter on the application")
        assert (message(0x0016), ending, stand_in.filters) == ((False, 0), [True, False], []), (
            "a cancel was heard twice")
        stand_in.aboutToQuit.emit()
        stand_in.commitDataRequest.emit(manager)
        assert [message(0x0016, 1), message(0x0011), message(0x000F),
                message(0x0016, kind=b"windows_dispatcher_MSG")] == [(False, 0)] * 4
        assert (ending, stand_in.filters) == ([True, False], [watch.filter]), (
            f"a quit after a cancel, or another message, was heard: {ending}")
    finally:
        watch.remove()
    stand_in.commitDataRequest.emit(manager)
    stand_in.aboutToQuit.emit()
    assert (len(asked_with), ending, stand_in.filters) == (3, [True, False], []), (
        "the watch was left on the application")


def test_off_the_real_windows_platform_no_watch_on_the_sessions_end_is_installed():
    """Off Windows, and off the toolkit's real ``windows`` platform as here, the watch on Windows' end of a session is
    installed nowhere, as every other request made of Windows answers False there: none is handed back, the process
    holds none, and nothing more hears the application's question or its quit. The watch is asked for as its module
    defines it."""
    app = qt_app()
    signals = ("2aboutToQuit()", "2commitDataRequest(QSessionManager&)")
    before, asked_with, ending = [app.receivers(signal) for signal in signals], [], []
    watch = WATCH_SESSION_END(app, asked=asked_with.append, ending=ending.append)
    try:
        assert (watch, this_computer.watching(), [app.receivers(signal) for signal in signals]) == (
            None, None, before), "off the real windows platform the watch on Windows' end of a session was installed"
    finally:
        if watch is not None:
            watch.remove()


# ── the figure windows ──
def _picture(width=40, height=30) -> bytes:
    """A picture of a figure, small unless a size is handed, as the run's figure sink hands one over."""
    from PySide6.QtCore import QBuffer, QByteArray, QIODevice
    from PySide6.QtGui import QImage
    image = QImage(width, height, QImage.Format_RGB32)
    image.fill(0xFFFFFF)
    data = QByteArray()
    buffer = QBuffer(data)
    buffer.open(QIODevice.WriteOnly)
    image.save(buffer, "PNG")
    return bytes(data)


def _figure_pickle() -> bytes:
    """A figure of the figure library, built off any state machine of its own and pickled, as the sink ships one."""
    import pickle
    from matplotlib.figure import Figure
    fig = Figure()
    axes = fig.add_subplot()
    axes.plot([0, 1, 2], [0, 1, 4])
    return pickle.dumps(fig)


def _pop_button(stack, index):
    """The "Pop out" button of the figure at ``index`` of ``stack``."""
    from PySide6.QtWidgets import QPushButton
    return next(button for button in stack.widget(index).findChildren(QPushButton) if button.text() == "Pop out")


def _pop(stack, index):
    """Press "Pop out" on the figure at ``index`` of ``stack``; the window it opened."""
    before = set(stack._windows)
    _pop_button(stack, index).click()
    (opened,) = set(stack._windows) - before
    return opened


def _open(widget) -> bool:
    """Whether ``widget`` is still there and in sight: a window closed with its own deferred delete is neither."""
    import shiboken6
    return shiboken6.isValid(widget) and widget.isVisible()


def test_every_popped_out_figure_window_closes_with_the_main_one(tmp_path):
    """A figure popped out of a page is a window of its own that no window owns, interactive or a picture, so that it
    stands where a person puts it, over the main window or under it, and is not minimised with it; when the main
    window closes, every such window closes with it, a minimised one too, so that nothing keeps the program running
    unseen."""
    from core.gui.panels.reduction_panel import ReductionPanel
    from core.gui.widgets.figure_window import ImageZoomWindow, InteractiveFigureWindow
    from tests._window_helpers import shown_off_screen
    app = qt_app()
    window = build_window()
    shown_off_screen(window)
    stack = window.panel(ReductionPanel).figure_stack
    stack.add_figure("interactive", _picture(), _figure_pickle())
    stack.add_figure("a picture", _picture())
    opened = [_pop(stack, 0), _pop(stack, 1), _pop(stack, 1)]
    opened[2].showMinimized()
    app.processEvents()
    assert [type(figure) for figure in opened] == [InteractiveFigureWindow, ImageZoomWindow, ImageZoomWindow]
    assert [(figure.isWindow(), figure.parentWidget(), figure.isVisible()) for figure in opened] == [
        (True, None, True)] * 3, "a figure window is owned, or not in sight"
    assert opened[2].isMinimized()
    window.close()
    assert (window.isVisible(), [_open(figure) for figure in opened]) == (False, [False, False, False]), (
        "a figure window outlived the main window's close")


# ── a figure while a run is live, and a figure the window could not show ──
def _settled(app) -> None:
    """Turn the event loop until what is due now has been done: two rounds of the loop's own zero-time timers, so
    that a draw a resize put off to the next round has had its turn."""
    from PySide6.QtCore import QTimer
    for _ in range(2):
        done = []
        QTimer.singleShot(0, lambda: done.append(True))
        turn_until(app, lambda: bool(done))


def test_a_figure_popped_out_during_a_run_opens_as_its_picture(tmp_path):
    """While a run is live, a figure popped out of a page opens as its picture, in the image viewer, even when an
    interactive copy could be built: the window's thread lays out no figure while a run builds its own; a window no
    window owns, as at any time. Its button says so meanwhile, for a figure that arrived before the run and one that
    arrived during it. Once the run has ended, it opens interactive again, and its button says that."""
    from core.gui.panels.reduction_panel import ReductionPanel
    from core.gui.widgets.figure_window import ImageZoomWindow, InteractiveFigureWindow
    app = qt_app()
    held, window = _controller_on(tmp_path, Clock(_at(8, 19, 2, 11))), build_window()
    host = window.panel(ReductionPanel)
    stack = host.figure_stack
    stack.add_figure("interactive", _picture(), _figure_pickle())
    interactive, held_tip, picture = ("Open an interactive, zoomable copy in a new window",
                                      "Open a zoomable picture in a new window; an interactive copy opens once the run "
                                      "has ended", "Open a zoomable copy in a new window")
    assert _pop_button(stack, 0).toolTip() == interactive
    stand_in = StandIn()
    start_run(host, stand_in, spec=RunSpec("Reduction map", ("reduction", None), stop_button="Stop computing…"))
    stack.add_figure("arrived during the run", _picture(), _figure_pickle())
    stack.add_figure("a picture", _picture())
    assert [_pop_button(stack, index).toolTip() for index in range(3)] == [held_tip, held_tip, picture], (
        "the pop-out's button promises an interactive copy while a run is live")
    during = _pop(stack, 0)
    assert (type(during), during.parentWidget()) == (ImageZoomWindow, None)
    stand_in.release()
    _ended(app, held)
    assert [_pop_button(stack, index).toolTip() for index in range(3)] == [interactive, interactive, picture]
    after = _pop(stack, 0)
    assert type(after) is InteractiveFigureWindow
    for figure in (during, after):
        figure.close()


def test_a_figure_window_open_when_a_run_starts_shows_its_picture_until_the_run_ends(tmp_path):
    """An interactive figure window open when a run starts shows its figure's picture from the run's start, scaled
    down to the room the window has, and lays out nothing of its figure while the run is live: a resize, a key or a
    change of the screen's scale reaches no drawing of it. It keeps its size, a picture larger than the window being
    no reason to grow, and can be made smaller meanwhile. When the run has ended it is interactive again, at the view
    it had and at the size it was given. A window whose figure has no picture held says, in its place, that the
    figure is drawn again when the run has ended."""
    from PySide6.QtCore import QEvent, Qt
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication
    from core.gui.panels.reduction_panel import ReductionPanel
    from core.gui.widgets.figure_window import WAIT_TEXT
    from tests._window_helpers import shown_off_screen
    app = qt_app()
    held, window = _controller_on(tmp_path, Clock(_at(8, 19, 2, 11))), build_window()
    shown_off_screen(window)
    host = window.panel(ReductionPanel)
    stack = host.figure_stack
    stack.add_figure("interactive", _picture(1_800, 1_200), _figure_pickle())
    stack.add_figure("no picture", b"", _figure_pickle())
    figure, bare = _pop(stack, 0), _pop(stack, 1)
    figure.resize(900, 650)
    axes = figure._fig.axes[0]
    axes.set_xlim(0.5, 1.5)                          # zoomed in, as a person leaves it
    figure.canvas.setFocus(Qt.MouseFocusReason)      # the keyboard on the figure, as a click on it leaves it
    _settled(app)
    drawn = []
    real = figure.canvas.draw
    figure.canvas.draw = lambda: drawn.append("drawn") or real()
    stand_in = StandIn()
    start_run(host, stand_in, spec=RunSpec("Reduction map", ("reduction", None), stop_button="Stop computing…"))
    assert (figure.picture.isVisibleTo(figure), figure.canvas.isVisibleTo(figure), figure.toolbar.isVisibleTo(figure),
            figure.picture.pixmap().isNull()) == (True, False, False, False), "the run's start did not show the picture"
    assert (bare.picture.text(), bare.picture.isVisibleTo(bare)) == (WAIT_TEXT, True)
    _settled(app)
    assert ((figure.width(), figure.height()), figure.picture.pixmap().width() <= figure.picture.width()) == (
        (900, 650), True), "a held window grew to its picture's size, or its picture is not scaled to its room"
    figure.resize(700, 480)
    QTest.keyClick(figure.focusWidget() or figure, Qt.Key_H)    # where a person's key goes: "home", a redraw
    _settled(app)
    assert ((figure.width(), figure.height()), drawn) == ((700, 480), []), (
        "a held window could not be made smaller, or the figure was drawn while the run was live")
    figure.canvas._set_device_pixel_ratio(2.0)       # the figure library's ratio as a screen of another scale leaves it
    QApplication.sendEvent(figure.windowHandle(), QEvent(QEvent.DevicePixelRatioChange))
    _settled(app)
    assert drawn == [], "a change of the screen's scale drew the figure while the run was live"
    stand_in.release()
    _ended(app, held)
    _settled(app)
    assert (figure.picture.isVisibleTo(figure), figure.canvas.isVisibleTo(figure), figure.toolbar.isVisibleTo(figure),
            tuple(axes.get_xlim()), (figure.width(), figure.height())) == (False, True, True, (0.5, 1.5), (700, 480)), (
        "the window is not interactive again, at its view and its size")
    turn_until(app, lambda: bool(drawn))
    for each in (figure, bare):
        each.close()


def test_a_figure_handed_over_after_the_window_is_gone_cannot_remove_a_finished_record(tmp_path):
    """The window's figure sink is called inside the writer of the record the figure is saved in, and that writer
    removes its record on any exception. A picture that cannot be made, a hand-over that fails, and a hand-over to a
    window that is gone each become one warning, the figure's title and the error's kind and message in it, and the
    record is written whole with its figure; the figure is closed. The same failure met again is said again, by the
    filter of the window's own, never dropped as a repeat. A stop still passes, as at any point where one is taken,
    and the figure is closed then too."""
    import io
    import matplotlib.pyplot as plt
    import shiboken6
    from core.artifacts import ArtifactStore
    from core.Helpers import visualizers
    from core.gui.panels.base_panel import FigureNotShown, _png_fig_sink
    from core.gui.streams import WorkerCancelled
    from core.gui.worker import WorkerSignals
    qt_app()
    store = ArtifactStore(tmp_path / "Artifacts")
    real = visualizers.save_figure

    class Failing:
        def emit(self, *args):
            raise RuntimeError("the hand-over failed")

    gone = WorkerSignals()
    figure_signal = gone.figure
    shiboken6.delete(gone)

    def render_fails(fig, target, **kwargs):
        if isinstance(target, io.BytesIO):              # the window's picture, never the record's own file
            raise ValueError("the picture could not be made")
        return real(fig, target, **kwargs)

    legs = (("a render that fails", render_fails, WorkerSignals().figure, "ValueError: the picture could not be made"),
            ("an emit that fails", real, Failing(), "RuntimeError: the hand-over failed"),
            ("a window that is gone", real, figure_signal, "RuntimeError: Signal source has been deleted"))
    for leg, save, signal, why in legs:
        visualizers.save_figure = save
        try:
            fig, _axes = plt.subplots()
            again, _axes = plt.subplots()
            with warnings.catch_warnings(record=True) as said:
                with store.create("calibration", None) as writer:
                    writer.body = {"results": {}}
                    writer.fig_sink(_png_fig_sink(signal))("Training loss", fig)
                _png_fig_sink(signal)("Training loss", again)       # the same failure once more: said again
        finally:
            visualizers.save_figure = real
        assert [(warning.category, str(warning.message)) for warning in said] == [
            (FigureNotShown,
             f"The figure 'Training loss' could not be shown ({why}). The record is not affected.")] * 2, leg
        assert (plt.fignum_exists(fig.number), plt.fignum_exists(again.number)) == (False, False), (
            f"{leg}: a figure was left open")
    records = store.list("calibration")
    assert len(records) == 3 and all(summary.complete for summary in records), "a record was removed"

    def stopped(fig, target, **kwargs):
        if isinstance(target, io.BytesIO):
            raise WorkerCancelled()
        return real(fig, target, **kwargs)

    fig, _axes = plt.subplots()
    visualizers.save_figure = stopped
    try:
        with pytest.raises(WorkerCancelled):
            with store.create("calibration", None) as writer:
                writer.body = {"results": {}}
                writer.fig_sink(_png_fig_sink(WorkerSignals().figure))("Training loss", fig)
    finally:
        visualizers.save_figure = real
    assert plt.fignum_exists(fig.number) is False and len(store.list("calibration")) == 3
