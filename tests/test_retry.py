"""A run's failure, as a state of the run's page and never a box: what failed and when, what was saved and what was
lost, the error's message, and its kind and traceback when asked for; the health line of a training whose save
failed during its fit; the window's tries after the graphics card ran out of memory, their pauses, and when only
reopening PRISM can help; and "Continue from batch N" and "Resume the training" on the page of a failed or stopped
training.

A live run here is the stand-in run of tests/_runs.py, started through a real panel's own dispatch and failing with
the error a test hands it, the graphics card's lack of memory as an error chain the engine reads as one; a training's
simulation set is a folder of the test's own whose state, written as the engine writes it, holds the batches the test
says; the window's clock is moved by hand, and a pause runs out when the test moves the clock past it and looks
(``tick``), never by a real wait.
"""
import gc
import weakref
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from PySide6.QtWidgets import QLabel

from core.gui.run import retry
from core.gui.run.controller import controller
from core.gui.run.model import SIMULATION_ROW, RunSpec
from core.refusals import Refusal
from tests._fixtures import all_asked, build_fit_record, build_run_host, choose, qt_app
from tests._runs import Clock, RunText, StandIn, start_run, turn_until

# local time with its zone, as the window's clock gives it
_ZONE = timezone(timedelta(hours=-7))
# a training's lines, as the engine writes them
_BUDGET = "[budget] 10,000 batches x 2,048 rows = 20,480,000 training rows"
_ROTATING = "Computing decorrelating Fisher rotation (REPARAM_ROTATE=True)..."
_ROTATED = "[fisher] eigenvalue spread (best/worst direction): 412"
_NOT_SAVED = "[fit] saving is switched off: this training is not saved as it goes and cannot be resumed."


def _at(day, hour, minute, second=0) -> datetime:
    """A moment of October 2026 in the test's local time (the 3rd is a Saturday)."""
    return datetime(2026, 10, day, hour, minute, second, tzinfo=_ZONE)


def _controller_on(tmp_path, clock):
    """The process's run controller, its clock and its folder of runs this test's own; the end of the test puts
    both back."""
    held = controller()
    held.use_clock(clock)
    held.use_folder(tmp_path / "runs")
    return held


def _training(word="Training", **changed) -> RunSpec:
    """A training as its page hands it to its run: its stages, its patience and its epoch limit, tried again by the
    window."""
    from core.gui.panels.inference.base import TRAINING_STAGES
    button = "Stop training…" if word == "Training" else "Stop the narrowing round…"
    return RunSpec(**{"word": word, "place": ("inference", "training"), "name": "retrain_3", "stop_button": button,
                      "stages": TRAINING_STAGES, "kinds": ("posterior", "simulation", "fit"), "simulation_set": True,
                      "retries": True, "patience": 20, "epoch_limit": 300,
                      "facts": {"model": "NADROWSKI", "observation": "chi", "narrowed": word != "Training",
                                "network": (256, 10)}, **changed})


def _set(tmp_path, batches, *, complete=False):
    """A simulation set's folder whose state, written as the engine writes it, holds ``batches`` saved batches."""
    import torch
    folder = tmp_path / "simulations" / "_unnamed__20261003T030211"
    folder.mkdir(parents=True, exist_ok=True)
    torch.save({"batches_done": batches, "complete": complete}, folder / "state.pt")
    return folder


def _writing(folder) -> str:
    return f"[checkpoint] writing to {folder} every 50 batches (~10.3 GiB total, 120.5 GiB free)"


def _complete(folder) -> str:
    return (f"[checkpoint] complete: 10000 batches in {folder}. Safe to delete once the posterior is saved; keeping "
            f"it lets you retrain the flow without re-simulating.")


def _epoch(epoch, since=0, took=3780.0) -> str:
    """An epoch's record, as the training call writes it."""
    return (f"[fit] epoch {epoch}: training loss -3, validation loss -3.1, took {took:.1f} s, epochs since the best "
            f"{since}")


def _say(app, stand_in, text, *lines, level="info") -> None:
    """Have the stand-in run say ``lines``, and turn the event loop until the run's full text holds the last once
    more than before: an earlier run of the test may have said the same line."""
    said = text.texts().count(lines[-1])
    for line in lines:
        stand_in.say(line, level)
    turn_until(app, lambda: text.texts().count(lines[-1]) > said)


def _to_the_simulation(app, stand_in, text, held, folder, done) -> None:
    """A training brought to its simulation in ``folder``, ``done`` of its 10,000 batches simulated."""
    _say(app, stand_in, text, _BUDGET, _ROTATING, _ROTATED, _writing(folder))
    stand_in.row(SIMULATION_ROW, done, 10_000)
    turn_until(app, lambda: held.model.batches_done() == done)


def _to_the_fit(app, stand_in, text, clock, folder, epochs, *, saved=True) -> None:
    """A training brought to its fit on the set in ``folder``, one epoch's time later, its records ``epochs``
    ((epoch, since the best), ...) said; its fit not saved as it goes when not ``saved``."""
    _say(app, stand_in, text, _BUDGET, _ROTATING, _ROTATED, _writing(folder), *(() if saved else (_NOT_SAVED,)),
         _complete(folder))
    clock.advance(3780)
    if epochs:
        _say(app, stand_in, text, *(_epoch(epoch, since) for epoch, since in epochs))


def _note(kept_id, passes, *, resumed=False, saved_here=True, processes=1):
    from core.SBI.fit_state import KeptNote
    return KeptNote(kept_id, passes, resumed, saved_here, processes)


def _ended(app, held) -> None:
    turn_until(app, lambda: not held.is_live(), 10)


def _failure(view) -> str:
    """The failure's sentence as its page says it, the lead beside its marker included."""
    return view.failure_row.sentence.plain_text()


def _health(view) -> list:
    """The rows of the page's health table, as written in their cells."""
    table = view.health.view.model()
    return [tuple(table.index(row, column).data().strip() for column in range(3)) for row in range(table.rowCount())]


def _out_of_memory(what="the simulation of batch 237"):
    """An error chain the engine reads as the graphics card running out of memory: the card's own error, under the
    error raised from it."""
    try:
        raise RuntimeError("CUDA error: out of memory")
    except RuntimeError as card:
        try:
            raise RuntimeError(f"{what} failed") from card
        except RuntimeError as raised:
            return raised


def _device_failure():
    """The graphics card's library reporting a failure of the card that is no lack of memory."""
    import torch
    return torch.AcceleratorError("CUDA error: an illegal memory access was encountered")


def _start_with(host, stand_in, spec, *args, **kwargs) -> None:
    """Start ``stand_in`` through ``host``'s own dispatch with the positional arguments a page's call hands its
    stage, ``spec`` standing in for the host's run spec for this one start."""
    host.run_spec = lambda fn, *a, **k: spec
    try:
        assert host.dispatch(stand_in, *args, **kwargs), "the stand-in run was not started"
    finally:
        del host.run_spec


def _waiting(app, held) -> None:
    """Turn the event loop until the run waits to be tried again."""
    turn_until(app, lambda: held.status.state == "retrying")


def _tried(app, held, stand_in, calls) -> None:
    """Turn the event loop until the stand-in run has been called ``calls`` times and that try is running."""
    turn_until(app, lambda: len(stand_in.calls) == calls and held.status.state == "running")


def _waits(pause, which, batch) -> str:
    """What the page of a training that waits to be tried again says, the graphics card having run out of memory in
    its simulation, ``batch`` batches saved."""
    return (f"The graphics card ran out of memory at batch {batch + 1}. Trying again in {pause}: the {which} of three "
            f"tries. The simulations up to batch {batch} are saved.")


_REOPEN = ("Error. Trying again here did not help: this copy of PRISM still holds the graphics card's memory. Close "
           "PRISM and open it again")


# ── the failure page ──
def test_a_failure_is_a_state_of_the_run_page_and_no_box(tmp_path):
    """A run that fails says so on its page, never in a box: beside the error's marker, what failed and when, what
    was saved and what was lost; under it the error's own message, quoted in the fixed-width face, and that the
    full error is in the run's file; the error's kind and traceback only behind "Show the full error", and in the
    run's text file. A run of another kind says what failed and when, with the error's message. A run that only loads
    a record has no page of its own: its failure is told in the box, as a failure at a click is."""
    from core.gui import fonts
    app = qt_app()
    clock = Clock(_at(3, 3, 2, 11))
    held, host = _controller_on(tmp_path, clock), build_run_host()
    text, view = RunText(held), host.run_view
    folder = _set(tmp_path, 236)
    stand_in = StandIn()
    start_run(host, stand_in, spec=_training())
    _to_the_simulation(app, stand_in, text, held, folder, 236)
    clock.advance(600)                                  # 03:12
    stand_in.fail(OSError(28, "No space left on device"))
    _ended(app, held)
    assert (all_asked(), view.mode, held.status.outcome, view.failure.isVisibleTo(view),
            view.failure_row.marker.kind) == ([], "ended", "failed", True, "err"), (
        "the failure was told in a box, or its page does not say it")
    assert _failure(view) == ("Error. Training failed on Sat 3 Oct, 03:12, during simulation: [Errno 28] No space left "
                              "on device. Saved: the simulations up to batch 236. Lost: nothing else."), _failure(view)
    assert (view.failure_message.text(), view.failure_message.font().family(), view.file_line.text(),
            view.file_line.isVisibleTo(view), view.full_error_button.text(), view.full_error.isVisibleTo(view)) == (
        "[Errno 28] No space left on device", fonts.fixed().family(), "The full error is in the run's file.", True,
        "Show the full error", False)
    in_sight = [label.text() for label in view.findChildren(QLabel) if label.isVisibleTo(view)] + [_failure(view)]
    assert not any("OSError" in words for words in in_sight), "the error's kind is shown outside the full error"
    view.full_error_button.click()
    shown = view.full_error.toPlainText()
    assert (view.full_error.isVisibleTo(view), view.full_error_button.text(), shown.startswith("Traceback"),
            "OSError: [Errno 28] No space left on device" in shown) == (True, "Hide the full error", True, True), shown
    filed = (tmp_path / "runs" / f"{held.status.run}.txt").read_text(encoding="utf-8")
    assert "Traceback" in filed and "OSError: [Errno 28] No space left on device" in filed, filed

    check = StandIn(word="Check")
    start_run(host, check, spec=RunSpec("Check", ("inference", "checks"), name="Calibration",
                                        stop_button="Stop the calibration…"))
    assert not view.failure.isVisibleTo(view), "the earlier run's failure stayed on the page of the next"
    clock.advance(60)
    check.fail(ValueError("the coverage could not be computed."))
    _ended(app, held)
    assert (all_asked(), _failure(view), view.failure_message.text(), view.full_error.isVisibleTo(view)) == (
        [], "Error. Check failed on Sat 3 Oct, 03:13: the coverage could not be computed.",
        "the coverage could not be computed.", False)

    loads = StandIn(word="Prior load")
    start_run(host, loads, spec=RunSpec("Prior load", ("inference", "prior"), stop_button="Stop loading",
                                        loads_only=True, keeps_files=False, stop_asks=False))
    loads.fail(ValueError("the prior's file could not be read"))
    _ended(app, held)
    assert [(box.kind, box.sentence) for box in all_asked()] == [("error", "the prior's file could not be read")], (
        "a run that only loads a record, which has no page of its own, did not tell its failure in the box")


def test_a_failed_save_of_the_training_shows_on_its_health_line(tmp_path):
    """A save of the training that fails during the fit never stops it: the page's health says that the last save
    failed and the epoch the kept training stands at, or that nothing is kept yet, its state "Look closer", and a
    message says why; the next save that holds clears the row."""
    app = qt_app()
    clock = Clock(_at(3, 3, 2, 11))
    held, host = _controller_on(tmp_path, clock), build_run_host()
    text, view = RunText(held), host.run_view
    folder = _set(tmp_path, 10_000, complete=True)
    stand_in = StandIn()
    start_run(host, stand_in, spec=_training())
    _to_the_fit(app, stand_in, text, clock, folder, ())

    def saved():
        return next(row for row in _health(view) if row[0] == "Training saved")

    def said():
        return held.model.messages()[0].sentence

    _say(app, stand_in, text, "[fit] the last save failed: OSError: disk full; nothing is kept yet.", level="warning")
    assert (saved(), said()) == (
        ("Training saved", "nothing kept yet; the last save failed", "Look closer"),
        "The last save of the training failed (OSError: disk full); nothing of it is kept yet.")
    _say(app, stand_in, text, _epoch(1), "[fit] saved after pass 1.")
    assert saved() == ("Training saved", "after epoch 1", "Fine")
    _say(app, stand_in, text, _epoch(2, 1))
    _say(app, stand_in, text, "[fit] the last save failed: OSError: disk full; the kept training stands at pass 1.",
         level="warning")
    assert (saved(), said()) == (
        ("Training saved", "epoch 1; the last save failed", "Look closer"),
        "The last save of the training failed (OSError: disk full); the kept training stands at epoch 1.")
    assert view.message_rows()[0].sentence.plain_text().endswith(said()), "the message is not on the page"
    _say(app, stand_in, text, _epoch(3, 2), "[fit] saved after pass 3.")
    assert (saved(), held.is_live()) == (("Training saved", "after epoch 3", "Fine"), True), (
        "the next save did not clear the row, or the failed save stopped the fit")
    stand_in.release()
    _ended(app, held)


# ── the tries ──
def test_the_window_tries_again_three_times_after_the_cards_memory_ran_out(tmp_path, monkeypatch):
    """After the graphics card ran out of memory, the window tries the training again, the run waiting between the
    tries, live, with nothing computing: after 1 minute, then, while each try keeps nothing more, after 3 minutes and
    10 minutes, counted on the window's clock. Meanwhile the page says what failed, when it tries again and which of
    the three tries that is, and what is saved, and "Stop trying" stands in the stop button's place. A try is the
    run's own call in a new worker, its keywords as they were but the set's policy that must continue, its result
    handed where the run's was, in the same text and history files, each try opened by its line in the text; a try
    that keeps a batch more starts the count again. "Stop trying" ends the run failed, said last in its text, and
    its page then offers to continue from the batches the set holds. A run whose stop was asked for before it ran
    out of memory is not tried again."""
    from core.gui.run import history
    app = qt_app()
    clock = Clock(_at(3, 3, 2, 11))
    held, host = _controller_on(tmp_path, clock), build_run_host()
    text, view = RunText(held), host.run_view
    folder, results = _set(tmp_path, 236), []
    stand_in = StandIn()
    start_run(host, stand_in, spec=_training(), name="retrain_3", num_runs=10_000, on_result=results.append)
    _to_the_simulation(app, stand_in, text, held, folder, 236)
    clock.advance(600)                                  # 03:12
    stand_in.fail(_out_of_memory())
    _waiting(app, held)

    def page():
        return (held.status.state, held.status.state_word, held.is_live(), view.sentence.text(),
                view.stop_trying_button.isVisibleTo(view), view.stop_button.isVisibleTo(view),
                view.failure.isVisibleTo(view))

    def waits(pause, which, batch):
        return ("retrying", "trying again", True, _waits(pause, which, batch), True, False, False)

    assert (page(), all_asked(), view.stop_trying_button.text()) == (waits("1 minute", "first", 236), [],
                                                                     "Stop trying")
    clock.advance(59)
    held.tick()
    assert (len(stand_in.calls), page()) == (1, waits("1 minute", "first", 236)), "a try began before its pause"
    clock.advance(1)
    held.tick()
    _tried(app, held, stand_in, 2)
    held.tick()
    assert (view.stop_button.isVisibleTo(view), view.stop_trying_button.isVisibleTo(view),
            [one.sentence for one in held.model.messages() if one.sentence.startswith("No output")]) == (
        True, False, []), "the try's start was not heard as the run's sign of life"
    stand_in.row(SIMULATION_ROW, 300, 10_000)
    turn_until(app, lambda: held.model.batches_done() == 300)
    _set(tmp_path, 300)                                 # the try saved batches up to 300, then ran out of memory
    stand_in.fail(_out_of_memory("the simulation of batch 301"))
    _waiting(app, held)
    assert page() == waits("1 minute", "first", 300), "a try that kept a batch more did not start the count again"
    clock.advance(60)
    held.tick()
    _tried(app, held, stand_in, 3)
    stand_in.fail(_out_of_memory("the simulation of batch 301"))
    _waiting(app, held)
    assert page() == waits("3 minutes", "second", 300)
    clock.advance(61)
    held.tick()
    assert page() == waits("2 minutes", "second", 300), "the page did not count down to the next try"
    clock.advance(59)
    held.tick()
    assert (len(stand_in.calls), page()) == (3, waits("1 minute", "second", 300))
    clock.advance(60)                                   # 03:17
    held.tick()
    _tried(app, held, stand_in, 4)
    stand_in.fail(_out_of_memory("the simulation of batch 301"))
    _waiting(app, held)
    assert page() == waits("10 minutes", "third", 300)
    base = {"name": "retrain_3", "num_runs": 10_000}
    assert stand_in.calls == [((), base)] + [((), {**base, "resume": "require"})] * 3, stand_in.calls

    view.stop_trying_button.click()
    _ended(app, held)
    assert (held.status.state, held.status.outcome, all_asked(), results) == ("ended", "failed", [], [])
    assert (_failure(view), view.failure_message.text(), view.continue_button.isVisibleTo(view),
            view.continue_button.text(), view.stop_trying_button.isVisibleTo(view)) == (
        "Error. Training failed on Sat 3 Oct, 03:17, during simulation: the graphics card ran out of memory at batch "
        "301. Saved: the simulations up to batch 300. Lost: nothing else.", "the simulation of batch 301 failed", True,
        "Continue from batch 300", False)
    run = held.status.run
    said = [line for line in text.texts() if line.startswith("Try ")]
    filed = (tmp_path / "runs" / f"{run}.txt").read_text(encoding="utf-8").splitlines()
    assert said == ["Try 2 begins here.", "Try 3 begins here.", "Try 4 begins here."], said
    assert ([line[20:] for line in filed if " window Try " in line], filed[-1][20:]) == (
        [f"window {line}" for line in said], f"window {retry.STOPPED_TRYING_TEXT}"), filed[-3:]
    record = history.read(tmp_path / "runs" / f"{run}.json")
    assert [(datetime.fromisoformat(one["started"]), one["outcome"], one["failure"], one["batches_before"],
             one["batches_after"]) for one in record["tries"]] == [
        (_at(3, 3, 2, 11), "failed", "RuntimeError", 0, 236), (_at(3, 3, 13, 11), "failed", "RuntimeError", 236, 300),
        (_at(3, 3, 14, 11), "failed", "RuntimeError", 300, 300),
        (_at(3, 3, 17, 11), "failed", "RuntimeError", 300, 300)], record["tries"]
    assert (record["outcome"], record["failure"]["kind"], record["failure"]["reopen"]) == (
        "failed", "out of memory", False)

    asked_to_stop, before = StandIn(takes_stops=False), len(text.statuses)
    start_run(host, asked_to_stop, spec=_training())
    _to_the_simulation(app, asked_to_stop, text, held, folder, 300)
    choose(monkeypatch, words="Stop and keep the simulations")
    view.stop_button.click()
    asked_to_stop.fail(_out_of_memory("the simulation of batch 301"))      # before the stop is taken
    _ended(app, held)
    assert ([status.state for status in text.statuses[before:] if status.state == "retrying"],
            len(asked_to_stop.calls), held.status.outcome, view.noted.isVisibleTo(view)) == ([], 1, "failed", True), (
        "a run whose stop was asked for was tried again after its failure")


_REPORTED_AT_ONCE = {
    "a full disk": (lambda: OSError(28, "No space left on device"), "[Errno 28] No space left on device"),
    "a programming error": (lambda: TypeError("unsupported operand type(s) for +: 'int' and 'NoneType'"),
                            "unsupported operand type(s) for +: 'int' and 'NoneType'"),
    "a refusal after the spend": (lambda: Refusal("The read-back of the posterior was refused.", field="name"),
                                  "The read-back of the posterior was refused."),
    "the card's memory, in a check": (_out_of_memory, "the simulation of batch 237 failed"),
}


@pytest.mark.parametrize("failure", list(_REPORTED_AT_ONCE))
def test_any_other_failure_is_reported_at_once(failure, tmp_path):
    """A failure that the graphics card's lack of memory did not cause fails the same way again, so the window says
    it at once: no pause and no try, the run ended failed, its page saying what failed and offering to continue
    from the batches the simulation set holds. A run that is no training is not tried again, whatever its failure."""
    app = qt_app()
    clock = Clock(_at(3, 3, 2, 11))
    held, host = _controller_on(tmp_path, clock), build_run_host()
    text, view = RunText(held), host.run_view
    error, said = _REPORTED_AT_ONCE[failure]
    stand_in = StandIn()
    if failure.endswith("in a check"):
        start_run(host, stand_in, spec=RunSpec("Check", ("inference", "checks"), name="Calibration"))
    else:
        start_run(host, stand_in, spec=_training())
        _to_the_simulation(app, stand_in, text, held, _set(tmp_path, 236), 236)
    stand_in.fail(error())
    _ended(app, held)
    assert ([status.state for status in text.statuses if status.state == "retrying"], len(stand_in.calls),
            held.status.outcome, all_asked(), view.failure.isVisibleTo(view), view.failure_message.text()) == (
        [], 1, "failed", [], True, said), "the failure was tried again, told in a box, or not said on its page"
    if failure.endswith("in a check"):
        assert (_failure(view), view.continue_button.isVisibleTo(view)) == (
            f"Error. Check failed on Sat 3 Oct, 03:02: {said}.", False)
        from core.SBI.pipeline import _is_oom
        assert [retry.is_tried_again(error(), spec, is_oom=_is_oom) for spec in (
            RunSpec("Check", ("inference", "checks"), name="Calibration"), _training())] == [False, True], (
            "the rule of the tries does not read the run's own word on whether it is tried again")
    else:
        assert (view.continue_button.isVisibleTo(view), view.continue_button.text()) == (
            True, "Continue from batch 236")


def test_when_only_reopening_can_help_the_page_says_so(tmp_path, store):
    """When trying again in this copy of PRISM cannot help, the page says so, the run ended failed with no more
    tries and nothing to press: after three tries in a row that kept nothing more, each out of memory again; and at
    once after a failure in which the graphics card failed in a way that is no lack of memory, a first failure as
    well as a try's, which is never tried again. The page says how the training goes on: started again once PRISM is
    open again, it continues the simulation set from the batches it holds; a training kept in its fit is resumed
    from the command line. Neither "Continue from batch N" nor "Resume the training" is offered: either would run in
    this copy of PRISM."""
    from core.gui.run import history
    app = qt_app()
    clock = Clock(_at(3, 3, 2, 11))
    held, host = _controller_on(tmp_path, clock), build_run_host()
    text, view = RunText(held), host.run_view
    folder = _set(tmp_path, 236)
    reopen = (f"{_REOPEN}, then start the same training: it continues the saved simulations from batch 236.")

    def reopened(message):
        return (held.status.state, held.status.outcome, held.status.reopen, all_asked(), _failure(view),
                view.failure_message.text(), view.continue_button.isVisibleTo(view),
                view.resume_button.isVisibleTo(view), view.stop_trying_button.isVisibleTo(view)) == (
            "ended", "failed", True, [], reopen, message, False, False, False)

    stand_in = StandIn()
    start_run(host, stand_in, spec=_training())
    _to_the_simulation(app, stand_in, text, held, folder, 236)
    stand_in.fail(_out_of_memory())
    for pause, calls in ((60, 2), (180, 3), (600, 4)):  # three tries in a row that keep nothing more
        _waiting(app, held)
        clock.advance(pause)
        held.tick()
        _tried(app, held, stand_in, calls)
        stand_in.fail(_out_of_memory())
    _ended(app, held)
    assert reopened("the simulation of batch 237 failed"), (_failure(view), held.status)
    record = history.read(tmp_path / "runs" / f"{held.status.run}.json")
    assert (record["failure"]["reopen"], len(record["tries"])) == (True, 4)

    device = StandIn()
    start_run(host, device, spec=_training())
    _to_the_simulation(app, device, text, held, folder, 236)
    device.fail(_out_of_memory())
    _waiting(app, held)
    clock.advance(60)
    held.tick()
    _tried(app, held, device, 2)
    device.fail(_device_failure())
    _ended(app, held)
    assert (len(device.calls), reopened("CUDA error: an illegal memory access was encountered")) == (2, True), (
        _failure(view))

    first = StandIn()
    start_run(host, first, spec=_training())
    _to_the_simulation(app, first, text, held, folder, 236)
    first.fail(_device_failure())
    _ended(app, held)
    assert (len(first.calls), reopened("CUDA error: an illegal memory access was encountered")) == (1, True), (
        "a first failure of the graphics card that is no lack of memory was tried again, or offered to continue",
        _failure(view))

    kept = build_fit_record(store, passes=2)
    in_the_fit = StandIn()
    in_the_fit.kept_fit = _note(kept, 2)
    start_run(host, in_the_fit, spec=_training())
    _to_the_fit(app, in_the_fit, text, clock, _set(tmp_path, 10_000, complete=True), [(1, 0), (2, 1)])
    in_the_fit.fail(_device_failure())
    _ended(app, held)
    assert (len(in_the_fit.calls), held.status.reopen, _failure(view), view.resume_button.isVisibleTo(view),
            view.continue_button.isVisibleTo(view)) == (
        1, True, f"{_REOPEN}. The training is kept up to epoch 2; until the Training page offers to resume it, it is "
                 f"resumed from the command line: python -m core train --resume-fit {kept}, with the same --bounds, "
                 f"--prior and --chi it was trained with.", False, False), _failure(view)


# ── continuing and resuming from the run's page ──
def test_continue_from_batch_n_repeats_the_runs_own_call_with_the_must_continue_policy(tmp_path, monkeypatch):
    """"Continue from batch N" stands on the page of a training that failed, and of one that was stopped, while its
    simulation set is unfinished and holds a batch, N the batches the set holds as read from it once the run has
    ended. Pressed, it starts the run's own call again as the page's next run, with files of its own: its keywords
    as they were, the policy that must continue (``resume="require"``) the one new argument and N in none of them,
    its result handed where the first run's was. A run stopped before its set held a batch offers nothing; neither
    does a page drawn from a run's files alone; and while another run is live, the press starts nothing and the page
    says so."""
    from core.gui.run import history
    from core.gui.run.controller import LIVE_TEXT
    app = qt_app()
    clock = Clock(_at(3, 3, 2, 11))
    held, host = _controller_on(tmp_path, clock), build_run_host()
    text, view = RunText(held), host.run_view
    folder, results = _set(tmp_path, 236), []
    base = {"name": "retrain_3", "note": "the retrain", "num_runs": 10_000}
    failed = StandIn()
    start_run(host, failed, spec=_training(), on_result=results.append, **base)
    _to_the_simulation(app, failed, text, held, folder, 236)
    failed.fail(OSError(28, "No space left on device"))
    _ended(app, held)
    first = held.status.run
    assert (view.continue_button.isVisibleTo(view), view.continue_button.text(),
            view.resume_button.isVisibleTo(view)) == (True, "Continue from batch 236", False)
    record, path = history.read(tmp_path / "runs" / f"{first}.json"), tmp_path / "runs" / f"{first}.txt"
    view.continue_button.click()
    turn_until(app, lambda: len(failed.calls) == 2)
    assert (held.is_live(), held.status.run != first, held.status.name, view.mode,
            view.continue_button.isVisibleTo(view)) == (True, True, "retrain_3", "live", False)
    assert failed.calls == [((), base), ((), {**base, "resume": "require"})], failed.calls
    assert "236" not in repr(failed.calls[1]), "the batch the set holds was handed to the stage"
    failed.release("the posterior")
    _ended(app, held)
    assert (results, held.status.outcome, sorted(p.name for p in (tmp_path / "runs").glob("*.json"))) == (
        ["the posterior"], "finished", sorted([f"{first}.json", f"{held.status.run}.json"]))

    _set(tmp_path, 6_150)
    stopped = StandIn()
    start_run(host, stopped, spec=_training(), **base)
    _to_the_simulation(app, stopped, text, held, folder, 6_150)
    choose(monkeypatch, words="Stop and keep the simulations")
    view.stop_button.click()
    _ended(app, held)
    assert (held.status.outcome, view.continue_button.isVisibleTo(view), view.continue_button.text()) == (
        "stopped", True, "Continue from batch 6,150")
    other, elsewhere = StandIn(word="Check"), build_run_host()
    start_run(elsewhere, other, spec=RunSpec("Check", ("inference", "checks"), keeps_files=False))
    view.continue_button.click()
    assert (len(stopped.calls), view.notice.text(), view.continue_button.isVisibleTo(view)) == (1, LIVE_TEXT, True), (
        "a continue started while another run was live, or the page did not say why it did not")
    other.release()
    _ended(app, held)
    view.continue_button.click()
    turn_until(app, lambda: len(stopped.calls) == 2)
    assert stopped.calls[1] == ((), {**base, "resume": "require"}), stopped.calls
    stopped.release()
    _ended(app, held)

    early = StandIn()
    start_run(host, early, spec=_training(), **base)
    _say(app, early, text, _BUDGET, _ROTATING)
    choose(monkeypatch, words="Stop training")
    view.stop_button.click()
    _ended(app, held)
    assert (held.status.outcome, view.continue_button.isVisibleTo(view), view.resume_button.isVisibleTo(view)) == (
        "stopped", False, False), "a run stopped before its set held a batch offered to continue"

    view.show_past(record, path)
    assert (view.mode, view.continue_button.isVisibleTo(view), view.resume_button.isVisibleTo(view)) == (
        "past", False, False), "a page drawn from a run's files offered to continue it"


def test_resume_the_training_repeats_the_runs_own_call_with_the_kept_training_named(tmp_path, monkeypatch, store):
    """"Resume the training" stands on the page of a training stopped in its fit whose stop kept the training, beside
    the question of keeping it, as a secondary button, "Keep" being the main one (pressed, it is the answer Keep: the
    training stays kept), the main button once the question is answered, and on the page of one that failed during
    its fit. Pressed, it starts the run's own call again as the page's next run: its keywords as they
    were, the kept training named (``resume_fit``) the one new argument; a narrowing round's call names no number of
    directions and no HPD level, which it resumes with from the kept training's own record. A resume the stage
    refuses before anything is spent, for a reason the kept training gives, is said on the run's page, with "Back to
    the form", in no box. Once Discard has deleted the kept training, nothing is offered to resume."""
    app = qt_app()
    clock = Clock(_at(3, 3, 2, 11))
    held, host = _controller_on(tmp_path, clock), build_run_host()
    text, view = RunText(held), host.run_view
    folder = _set(tmp_path, 10_000, complete=True)
    base = {"name": "retrain_3", "num_runs": 10_000}
    choose(monkeypatch, words="Stop training")
    kept = build_fit_record(store, passes=3)
    stand_in = StandIn()
    stand_in.kept_fit = _note(kept, 3)
    start_run(host, stand_in, spec=_training(), **base)
    _to_the_fit(app, stand_in, text, clock, folder, [(1, 0), (2, 1), (3, 2)])
    view.stop_button.click()
    _ended(app, held)
    assert (held.status.outcome, view.keep_block.isVisibleTo(view), view.resume_button.isVisibleTo(view),
            view.resume_button.text(), view.continue_button.isVisibleTo(view)) == (
        "stopped", True, True, "Resume the training", False)
    assert (view.keep_button.property("cls"), view.resume_button.property("cls")) == ("primary", ""), (
        "while Keep is asked, Keep is not the one main button")
    view.resume_button.click()
    turn_until(app, lambda: len(stand_in.calls) == 2)
    assert (stand_in.calls[1], [summary.id for summary in store.list("fit")]) == (
        ((), {**base, "resume_fit": kept}), [kept]), "the resume's call, or the kept training, is not as it was"
    stand_in.fail(Refusal(f"The kept training {kept} cannot be resumed: the code that builds or trains the network "
                          f"has changed since.", field="resume_fit"))
    _ended(app, held)
    assert (all_asked(), view.mode, view.sentence.text(), view.back_button.isVisibleTo(view),
            view.failure.isVisibleTo(view), host.results.currentWidget() is view) == (
        [box for box in all_asked() if box.title == "Stop training?"], "ended",
        f"The kept training {kept} cannot be resumed: the code that builds or trains the network has changed since.",
        True, False, True), "a refused resume was told in a box, or not on its page"
    view.back_button.click()
    assert host.results.currentWidget() is host.figures

    narrowing = StandIn(word="Narrowing round")
    narrowed = {**base, "n_directions": 4, "level": 0.95}
    narrowing.kept_fit = _note(kept, 2)
    start_run(host, narrowing, spec=_training("Narrowing round"), **narrowed)
    _to_the_fit(app, narrowing, text, clock, folder, [(1, 0), (2, 1)])
    narrowing.fail(TypeError("unsupported operand type(s) for +: 'int' and 'NoneType'"))
    _ended(app, held)
    assert (view.resume_button.isVisibleTo(view), view.keep_block.isVisibleTo(view)) == (True, False)
    view.resume_button.click()
    turn_until(app, lambda: len(narrowing.calls) == 2)
    assert narrowing.calls[1] == ((), {**narrowed, "n_directions": None, "level": None, "resume_fit": kept}), (
        narrowing.calls)
    narrowing.release()
    _ended(app, held)

    answered = StandIn()
    answered.kept_fit = _note(kept, 3)
    start_run(host, answered, spec=_training(), **base)
    _to_the_fit(app, answered, text, clock, folder, [(1, 0), (2, 1), (3, 2)])
    view.stop_button.click()
    _ended(app, held)
    view.keep_button.click()
    assert (view.keep_block.isVisibleTo(view), view.resume_button.isVisibleTo(view),
            view.resume_button.property("cls")) == (False, True, "primary"), (
        "once Keep is answered, Resume is not the main button")

    discarded = StandIn()
    discarded.kept_fit = _note(kept, 3)
    start_run(host, discarded, spec=_training(), **base)
    _to_the_fit(app, discarded, text, clock, folder, [(1, 0), (2, 1), (3, 2)])
    view.stop_button.click()
    _ended(app, held)
    view.discard_button.click()
    assert (store.list("fit"), view.resume_button.isVisibleTo(view), held.resume_training(held.status.run)) == (
        [], False, False), "a training Discard deleted was still offered to resume"


def test_a_failure_during_the_fit_offers_resume_and_says_what_was_saved(tmp_path, store):
    """A training that fails during its fit says so, at the epoch the training it kept stands at (the note's, never
    the records the run said, one of which may be an epoch whose save did not hold), that the simulations and the
    training up to that epoch are saved and that at most the epoch in progress is lost, and offers to resume it; the
    history's fit says the same epochs. One whose fit is not saved as it goes says the fit is lost; one whose fit had
    kept no epoch yet says so; neither offers anything."""
    from core.gui.run import history
    app = qt_app()
    clock = Clock(_at(3, 3, 2, 11))
    held, host = _controller_on(tmp_path, clock), build_run_host()
    text, view = RunText(held), host.run_view
    folder = _set(tmp_path, 10_000, complete=True)
    kept = build_fit_record(store, passes=41)
    error = "unsupported operand type(s) for +: 'int' and 'NoneType'"
    for note, saved, epochs, when, said, resumes in (
            (_note(kept, 41), True, [(n, 0) for n in range(1, 43)], "04:05",
             f"during the fit, at epoch 41: {error}. Saved: the simulations, and the training up to epoch 41. Lost: at "
             f"most the epoch in progress.", True),
            (None, False, [(1, 0)], "05:08",
             f"during the fit: {error}. Saved: the simulations. Lost: the fit, which is not saved as it goes.", False),
            (None, True, (), "06:11",
             f"during the fit: {error}. Saved: the simulations. Lost: the epoch in progress; no epoch of the fit was "
             f"kept.", False)):
        stand_in = StandIn()
        stand_in.kept_fit = note
        start_run(host, stand_in, spec=_training())
        _to_the_fit(app, stand_in, text, clock, folder, epochs, saved=saved)
        stand_in.fail(TypeError(error))
        _ended(app, held)
        assert (all_asked(), _failure(view), view.resume_button.isVisibleTo(view),
                view.continue_button.isVisibleTo(view)) == (
            [], f"Error. Training failed on Sat 3 Oct, {when}, {said}", resumes, False), _failure(view)
    assert history.read(tmp_path / "runs" / f"20261003T030211-training.json")["fit"]["epochs"] == 41, (
        "the history's fit holds the records seen, not the kept training's epochs")


def test_a_failure_while_the_posterior_is_saved_says_the_finished_training_is_kept(tmp_path, store):
    """A training whose fit finished but whose posterior could not be written says so, and that the finished
    training is kept, so that resuming it writes the posterior without training again, and offers to resume it; one
    whose training was not saved as it went says that none was kept, and offers nothing."""
    app = qt_app()
    clock = Clock(_at(3, 3, 2, 11))
    held, host = _controller_on(tmp_path, clock), build_run_host()
    text, view = RunText(held), host.run_view
    folder = _set(tmp_path, 10_000, complete=True)
    kept = build_fit_record(store, passes=2, complete=True)
    for saved, note, said in ((True, _note(kept, 2), "The fit finished but its record could not be written. The "
                                                     "finished training is kept: resuming it writes the posterior "
                                                     "without training again."),
                              (False, None, "The fit finished but its record could not be written; the training was "
                                            "not kept.")):
        stand_in = StandIn()
        stand_in.kept_fit = note
        start_run(host, stand_in, spec=_training())
        _to_the_fit(app, stand_in, text, clock, folder, [(1, 0), (2, 20)], saved=saved)   # the fit has ended
        assert held.model.stage == "Saving"
        stand_in.fail(OSError(13, "Permission denied"))
        _ended(app, held)
        assert (all_asked(), _failure(view), view.failure_message.text(), view.resume_button.isVisibleTo(view),
                view.continue_button.isVisibleTo(view)) == (
            [], f"Error. {said}", "[Errno 13] Permission denied", saved, False), saved


def test_a_try_after_the_fit_ran_out_of_memory_resumes_the_kept_training(tmp_path, store):
    """After the graphics card ran out of memory during the fit, a try is the run's own call with the training it
    kept named (``resume_fit``), as Resume is, and nothing else new: neither the set's policy nor anything of a
    narrowing round's region; meanwhile the page says the epochs kept, the note's. The try resumes the kept training
    from its kept epoch: an epoch an earlier try recorded past it, whose save did not hold, leaves the curve, and the
    best epoch is counted again from the kept one. A try that keeps more epochs and then runs out of memory again
    starts a new count; a try that fails before it names the kept training keeps the one named before; when three
    tries in a row keep none, the page says only reopening PRISM helps, and that the
    kept training is resumed from the command line until its page offers it: the tool's command for the run,
    ``train`` for a training, ``tsnpe`` for a narrowing round, with the flags of what the run's own call trained
    from, or, where the call does not tell them, words that say to give them as they were."""
    app = qt_app()
    clock = Clock(_at(3, 3, 2, 11))
    held, host = _controller_on(tmp_path, clock), build_run_host()
    text, view = RunText(held), host.run_view
    folder = _set(tmp_path, 10_000, complete=True)
    kept = build_fit_record(store, passes=2)
    base = {"name": "retrain_3", "num_runs": 10_000}
    cfg = SimpleNamespace(sources={"bounds": "Resources/Bounds/nadrowski/master.txt"}, chi_mode=True,
                          hw=SimpleNamespace(device=SimpleNamespace(type="cuda")))
    prior = SimpleNamespace(kind="prior", id="20261001T150211", name="retrain_prior")
    stand_in = StandIn()
    stand_in.kept_fit = _note(kept, 2)
    _start_with(host, stand_in, _training(), cfg, prior, None, True, **base)
    _to_the_fit(app, stand_in, text, clock, folder, [(1, 1), (2, 0), (3, 0)])    # epoch 3's save did not hold
    stand_in.fail(_out_of_memory("the fit's epoch 4"))
    _waiting(app, held)

    def waits(pause, which, epoch):
        return (f"The graphics card ran out of memory during the fit, at epoch {epoch}. Trying again in {pause}: the "
                f"{which} of three tries. The epochs up to {epoch} are kept.")

    assert view.sentence.text() == waits("1 minute", "first", 2)
    clock.advance(60)
    held.tick()
    _tried(app, held, stand_in, 2)
    called = (cfg, prior, None, True)
    assert stand_in.calls[1] == (called, {**base, "resume_fit": kept}), stand_in.calls
    _say(app, stand_in, text, f"[fit] resuming the kept training {kept} after pass 2 (best validation loss -3.1, 0 "
                              f"passes since the best); its simulation set is read back first.")
    assert ([record.epoch for record in held.model.epochs()], held.model.best_epoch(),
            [record.epoch for record in view.loss.plot.records], view.loss.plot.best) == ([1, 2], 2, [1, 2], 2), (
        "an epoch whose save did not hold stayed on the curve, or the best was counted from it")
    _say(app, stand_in, text, _epoch(3, 1), _epoch(4, 2))
    stand_in.kept_fit = _note(kept, 4, resumed=True, processes=2)
    stand_in.fail(_out_of_memory("the fit's epoch 5"))
    _waiting(app, held)
    assert view.sentence.text() == waits("1 minute", "first", 4), "a try that kept more epochs did not start again"
    stand_in.kept_fit = None                            # each try below fails as the kept training is read back
    for pause, calls, after in ((60, 3, ("3 minutes", "second")), (180, 4, ("10 minutes", "third")), (600, 5, None)):
        clock.advance(pause)
        held.tick()
        _tried(app, held, stand_in, calls)
        stand_in.fail(_out_of_memory("the fit's epoch 5"))
        if after is not None:
            _waiting(app, held)
            assert view.sentence.text() == waits(*after, 4)
    _ended(app, held)
    assert (held.status.reopen, _failure(view), view.resume_button.isVisibleTo(view)) == (
        True, f"{_REOPEN}. The training is kept up to epoch 4; until the Training page offers to resume it, it is "
              f"resumed from the command line: python -m core train --resume-fit {kept} --bounds "
              f"Resources/Bounds/nadrowski/master.txt --prior retrain_prior --chi.", False), _failure(view)
    assert stand_in.calls[1:] == [(called, {**base, "resume_fit": kept})] * 4, stand_in.calls

    narrowing = StandIn(word="Narrowing round")
    narrowed = {**base, "n_directions": 4, "level": 0.95}
    narrowing.kept_fit = _note(kept, 2)
    typed = SimpleNamespace(sources={}, chi_mode=True, hw=None)          # bounds typed in by hand: no file to name
    parent, observation = SimpleNamespace(name="retrain_3", id="p1"), SimpleNamespace(name="", id="o1")
    _start_with(host, narrowing, _training("Narrowing round"), typed, parent, prior, observation, **narrowed)
    _to_the_fit(app, narrowing, text, clock, folder, [(1, 0), (2, 1)])
    narrowing.fail(_out_of_memory("the fit's epoch 3"))
    _waiting(app, held)
    clock.advance(60)
    held.tick()
    _tried(app, held, narrowing, 2)
    assert narrowing.calls[1] == ((typed, parent, prior, observation),
                                  {**narrowed, "n_directions": None, "level": None, "resume_fit": kept})
    narrowing.fail(_device_failure())
    _ended(app, held)
    assert _failure(view) == (f"{_REOPEN}. The training is kept up to epoch 2; until the Narrowing round page offers "
                              f"to resume it, it is resumed from the command line: python -m core tsnpe --resume-fit "
                              f"{kept}, with the same --bounds, --posterior, --observation and --chi it was trained "
                              f"with."), _failure(view)
    chosen = SimpleNamespace(sources={"bounds": r"C:\PRISM files\master.txt"}, chi_mode=False,
                             hw=SimpleNamespace(device=SimpleNamespace(type="cpu")))
    narrowed_parent = SimpleNamespace(name="", id="p2", posterior=SimpleNamespace(truncation=object()))
    spec = _training("Narrowing round").with_call(len, (chosen, narrowed_parent, prior, observation), {})
    assert retry.resume_command(spec, "k1") == (
        'python -m core tsnpe --resume-fit k1 --bounds "C:\\PRISM files\\master.txt" --posterior p2 --observation o1 '
        '--no-chi --accept-truncated --device cpu'), "the command names a flag that is not the call's"


def test_a_runs_call_is_kept_only_while_its_page_offers_to_continue_or_resume_it(tmp_path):
    """A run's own call -- the configuration, the prior and the tensors its arguments hold -- is let go at its end,
    but for a training whose page offers to continue or resume it: that call is kept until the page starts its next
    run. A finished training, a training whose failure only reopening can help, and a run of another kind let go of
    it at their end."""
    app = qt_app()
    clock = Clock(_at(3, 3, 2, 11))
    held, host = _controller_on(tmp_path, clock), build_run_host()
    text = RunText(held)
    folder = _set(tmp_path, 236)

    class Payload:
        """What a run's arguments hold, for the test to see when it is let go."""

    def ended(spec, end):
        payload = Payload()
        kept = weakref.ref(payload)
        stand_in = StandIn(word=spec.word)
        start_run(host, stand_in, spec=spec, payload=payload)
        del payload
        if spec.simulation_set:
            _to_the_simulation(app, stand_in, text, held, folder, 236)
        end(stand_in)
        _ended(app, held)
        stand_in.calls.clear()                          # the stand-in's own record of what it was handed
        gc.collect()
        return kept

    def tries_to_the_end(stand_in):
        stand_in.fail(_out_of_memory())
        for pause, calls in ((60, 2), (180, 3), (600, 4)):
            _waiting(app, held)
            clock.advance(pause)
            held.tick()
            _tried(app, held, stand_in, calls)
            stand_in.fail(_out_of_memory())

    offered = ended(_training(), lambda stand_in: stand_in.fail(OSError(28, "No space left on device")))
    assert offered() is not None, "the call of a run its page offers to continue was let go"
    let_go = [ended(_training(), lambda stand_in: stand_in.release()),
              ended(_training(), tries_to_the_end),
              ended(RunSpec("Check", ("inference", "checks")), lambda stand_in: stand_in.fail(OSError("x")))]
    gc.collect()
    assert ([one() is None for one in let_go], offered()) == ([True, True, True], None), (
        "a run's call was kept at its end with nothing to offer, or kept past its page's next run")


def test_closing_the_window_during_a_pause_ends_the_tries_and_then_closes_it(tmp_path, monkeypatch):
    """Closing the window while a training waits to be tried again asks first, in the words of the pause, and a
    question nobody answers keeps the window, the one-PRISM claim and the run's tries. "Stop the run and close" ends
    the tries: the run ends failed, its history says so, and only then is the claim let go and the window closed."""
    from core.gui import one_instance
    from core.gui.panels.reduction_panel import ReductionPanel
    from core.gui.run import history
    from tests._fixtures import asked, whole_program
    from tests._window_helpers import shown_off_screen
    app = qt_app()
    monkeypatch.setenv("PRISM_ARTIFACTS", str(tmp_path / "Artifacts"))
    clock = Clock(_at(3, 3, 2, 11))
    with whole_program() as (_app, window):
        held = _controller_on(tmp_path, clock)
        claim = one_instance.current()
        shown_off_screen(window)
        host, text = window.panel(ReductionPanel), RunText(held)
        stand_in = StandIn()
        start_run(host, stand_in, spec=_training())
        _to_the_simulation(app, stand_in, text, held, _set(tmp_path, 236), 236)
        stand_in.fail(_out_of_memory())
        _waiting(app, held)
        window.close()
        question = asked()
        assert (question.title, question.sentence, question.beneath, question.answers, question.default,
                question.loses_work) == (
            "Close PRISM?", "Training retrain_3 is waiting to try again.", _waits("1 minute", "first", 236),
            ("Stop the run and close", "Keep running"), "Keep running", False)
        assert (window.isVisible(), held.status.state, one_instance.current() is claim) == (True, "retrying", True), (
            "a close nobody answered closed the window, ended the tries or let go of the claim")
        run, released = held.status.run, []
        real = one_instance.release

        def release():
            if not released:
                record = history.read(tmp_path / "runs" / f"{run}.json")
                released.append((None if record is None else record["outcome"], host.run_view.notice.text()))
            real()

        monkeypatch.setattr(one_instance, "release", release)
        choose(monkeypatch, words="Stop the run and close")
        window.close()
        turn_until(app, lambda: not window.isVisible(), 10)
        assert (released, one_instance.current(), held.is_live(), len(stand_in.calls)) == (
            [("failed", "")], None, False, 1), (
            "the claim was let go before the run's history said how it ended, a try began, or the page said PRISM "
            "stayed open as it closed")
