"""A run's failure, as a state of the run's page and never a box: what failed and when, what was saved and what was
lost, the error's message, and its kind and traceback when asked for; the health line of a training whose save
failed during its fit; the window's tries after the graphics card ran out of memory, their pauses, and when only
reopening PRISM can help; and "Continue from batch N" and "Resume the training" on the page of a failed or stopped
training.

A live run here is the stand-in run of tests/_runs.py, started through a real panel's own dispatch and failing with
the error a test hands it, the graphics card's lack of memory as an error chain the engine reads as one, and the
card's own errors, in the simulation, under the simulator's error as the simulation raises them; a training's
simulation set is a folder of the test's own whose state, written as the engine writes it, holds the batches the test
says, which may be fewer than the run's text counted, as the engine saves the set only every so many batches; the
window's clock is moved by hand, and a pause runs out when the test moves the clock past it and looks (``tick``),
never by a real wait.
"""
import gc
import warnings
import weakref
from datetime import datetime
from types import SimpleNamespace

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel

from core.gui.fields import fix_sentence
from core.gui.run import retry
from core.gui.run.model import SIMULATION_ROW, RunSpec
from core.refusals import Refusal
from tests._fixtures import all_asked, build_fit_record, build_run_host, choose, qt_app
from tests._runs import (BUDGET as _BUDGET, ROTATING as _ROTATING, Clock, RunText, StandIn, at as _at,
                         controller_on as _controller_on, ended as _ended, epoch_line as _epoch, note as _note,
                         say as _say, start_run, to_the_fit as _to_the_fit,
                         to_the_simulation as _to_the_simulation, training_spec as _training, turn_until)

# the graphics card's own words for a failure that is no lack of memory, two lines of them; and the words the
# simulator's error gives before the card's
_ILLEGAL = ("CUDA error: an illegal memory access was encountered\nCUDA kernel errors might be asynchronously reported "
            "at some other API call, so the stacktrace below might be incorrect.")
_SOLVER = ("NadrowskiSDE euler_compiled failed after 4500 steps (batch=2048, segs=1, device=cuda:0, "
           "dtype=torch.float32)")


def _set(tmp_path, batches, *, complete=False):
    """A simulation set's folder whose state, written as the engine writes it, holds ``batches`` saved batches."""
    import torch
    folder = tmp_path / "simulations" / "_unnamed__20261003T030211"
    folder.mkdir(parents=True, exist_ok=True)
    torch.save({"batches_done": batches, "complete": complete}, folder / "state.pt")
    return folder


def _failure(view) -> str:
    """The failure's block as its page says it, beside its marker, under the page's sentence that says the run
    failed."""
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


def _card(message):
    """The graphics card's library reporting a failure of the card itself, in ``message``."""
    import torch
    return torch.AcceleratorError(message)


def _simulated(card):
    """``card``, the graphics card's own error, as the simulation raises it: under the simulator's error, whose
    message gives the solver's words, then the card's error's kind and whole message."""
    from core.Simulator.simulator import SimulationError
    error = SimulationError(f"{_SOLVER}: {type(card).__name__}: {card}")
    error.__cause__ = card
    return error


def _device_failure(in_the_simulation=True):
    """The graphics card's library reporting a failure of the card that is no lack of memory, in its own two lines:
    as the simulation raises it, or, in the fit, as it is."""
    card = _card(_ILLEGAL)
    return _simulated(card) if in_the_simulation else card


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


def _waits(pause, which, batch, at=None) -> str:
    """What the page of a training that waits to be tried again says, the graphics card having run out of memory in
    its simulation at batch ``at`` (by default the one after the batches saved), ``batch`` batches saved."""
    return (f"The graphics card ran out of memory at batch {at or batch + 1}. Trying again in {pause}: the {which} of "
            f"three tries. The simulations up to batch {batch} are saved.")


# the opening of a page that only reopening PRISM can help: after tries that did not help, and after a failure of the
# graphics card that is no lack of memory, filled with the error's message's first line
_REOPEN = "Trying again here did not help: this copy of PRISM still holds the graphics card's memory."
_DEVICE = "The graphics card failed in a way only reopening PRISM can clear: {}."
# what a page says when its offer is pressed after its session came to hold another model or another prior
_CHANGED = "The model or its prior has changed since this run began; start it again from the form."


def _said_alone(view) -> str:
    """The command line the page of a run names, read off the page's words as a person copies it: from the tool's name
    to the end of the words it stands in; "" when the page names none."""
    said = [_failure(view)] + [label.text() for label in view.failure.findChildren(QLabel)
                               if label.isVisibleTo(view) and label.textFormat() == Qt.PlainText]
    return next((words[words.index("python -m core "):] for words in said if "python -m core " in words), "")


def _parsed(command):
    """The command-line tool's own parser's reading of ``command``, the words before its subcommand left out; and
    what the parser said when it refused it ("" when it took it)."""
    import contextlib
    import io
    import shlex
    from core.tool import build_parser
    heard = io.StringIO()
    try:
        with contextlib.redirect_stderr(heard):
            return build_parser().parse_args(shlex.split(command)[3:]), ""
    except SystemExit:
        return None, heard.getvalue()


# ── the failure page ──
def test_a_failure_is_a_state_of_the_run_page_and_no_box(tmp_path):
    """A run that fails says so on its page, never in a box, and says it once: the page's sentence says the run
    failed, and beside the error's marker the failure's block says when and where, what was saved and what was lost,
    never the run's word and "failed" again; under it the error's own message, quoted in the fixed-width face, only
    where it adds to the block (a full disk's message is the block's cause word for word), and that the full error is
    in the run's file; the error's kind and traceback only behind "Show the full error", and in the run's text file.
    What the foot and Home are told keeps the run's word and "failed". A run of another kind says when it failed,
    with the error's message: the block quotes its first line alone, and the whole message stands beneath. A run
    that only loads a record has no page of its own: its failure is told in the box, as a failure at a click is."""
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
    said = ("during simulation: [Errno 28] No space left on device. Saved: the simulations up to batch 236. Lost: "
            "nothing else.")
    assert (view.sentence.text(), _failure(view), held.status.sentence) == (
        "Training retrain_3 failed.", f"On Sat 3 Oct, 03:12, {said}", f"Training failed on Sat 3 Oct, 03:12, {said}"), (
        "the failure is said twice on its page, or the foot and Home are not told which run failed",
        view.sentence.text(), _failure(view), held.status.sentence)
    assert (view.failure_message.text(), view.failure_message.isVisibleTo(view), view.failure_message.font().family(),
            view.file_line.text(), view.file_line.isVisibleTo(view), view.full_error_button.text(),
            view.full_error.isVisibleTo(view)) == (
        "[Errno 28] No space left on device", False, fonts.fixed().family(), "The full error is in the run's file.",
        True, "Show the full error", False), "a full disk's message, the block's own words, is quoted beneath it"
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
    check.fail(ValueError("the coverage could not be computed.\n12 of 200 rows were not finite."))
    _ended(app, held)
    assert (all_asked(), view.sentence.text(), _failure(view), view.failure_message.text(),
            view.failure_message.isVisibleTo(view), view.full_error.isVisibleTo(view)) == (
        [], "Check Calibration failed.", "On Sat 3 Oct, 03:13: the coverage could not be computed.",
        "the coverage could not be computed.\n12 of 200 rows were not finite.", True, False), (
        "the block quotes more than the message's first line, or the message beneath, which adds its second line, "
        "is not whole or not in sight")

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
    tries, live, with nothing computing and every panel's controls locked: after 1 minute, then, while each try keeps
    nothing more, after 3 minutes and 10 minutes, counted on the window's clock. Meanwhile the page says what failed,
    when it tries again and which of the three tries that is, and what is saved, and "Stop trying" stands in the stop
    button's place. The card's own lack of memory, as the simulation raises it, is tried again too. A try is the run's
    own call in a new worker, its keywords as they were but the set's policy that must continue, its result handed
    where the run's was, in the same text and history files, each try opened by its line in the text; a try that
    keeps a batch more in the simulation set starts the count again, one whose run's text counted batches the set
    does not hold does not. The page names the batch that was in progress, the one after the last its text counted,
    and says what is saved from the set. The history keeps each try: when it began and ended, how, and what the set
    and the kept training held before and after it. "Stop trying" ends the run failed, said last in its text, and its
    page then offers to continue from the batches the set holds. A run whose stop was asked for before it ran out of
    memory is not tried again."""
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

    def waits(pause, which, batch, at=None):
        return ("retrying", "trying again", True, _waits(pause, which, batch, at), True, False, False)

    assert (retry.TRIES, retry.PAUSES_S) == (3, (60.0, 180.0, 600.0)), (
        "the tries are not three, after 1, 3 and 10 minutes")
    assert (page(), all_asked(), view.stop_trying_button.text(), host.controls.isEnabled()) == (
        waits("1 minute", "first", 236), [], "Stop trying", False), (
        "the run does not wait to be tried again, or a panel's controls were unlocked while it waits")
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
    stand_in.row(SIMULATION_ROW, 340, 10_000)           # the try simulated batches up to 340 and saved none of them
    turn_until(app, lambda: held.model.batches_done() == 340)
    stand_in.fail(_simulated(_card("CUDA error: out of memory")))       # the card's own lack of memory
    _waiting(app, held)
    assert page() == waits("3 minutes", "second", 300, 341), (
        "a try whose set holds no batch more started the count again, the card's own lack of memory was not tried "
        "again, or the page does not name the batch in progress")
    clock.advance(61)
    held.tick()
    assert page() == waits("2 minutes", "second", 300, 341), "the page did not count down to the next try"
    clock.advance(118)                                  # 179 s into the pause of 3 minutes
    held.tick()
    assert (len(stand_in.calls), page()) == (3, waits("1 minute", "second", 300, 341)), "a try began before its pause"
    clock.advance(1)                                    # 03:17
    held.tick()
    _tried(app, held, stand_in, 4)
    stand_in.row(SIMULATION_ROW, 300, 10_000)           # the try's bar, from the batches the set holds
    turn_until(app, lambda: held.model.batches_done() == 300)
    stand_in.fail(_out_of_memory("the simulation of batch 301"))
    _waiting(app, held)
    assert page() == waits("10 minutes", "third", 300)
    base = {"name": "retrain_3", "num_runs": 10_000}
    assert stand_in.calls == [((), base)] + [((), {**base, "resume": "require"})] * 3, stand_in.calls

    view.stop_trying_button.click()
    _ended(app, held)
    assert (held.status.state, held.status.outcome, all_asked(), results) == ("ended", "failed", [], [])
    assert (_failure(view), view.failure_message.text(), view.failure_message.isVisibleTo(view),
            view.continue_button.isVisibleTo(view), view.continue_button.text(),
            view.stop_trying_button.isVisibleTo(view)) == (
        "On Sat 3 Oct, 03:17, during simulation: the graphics card ran out of memory at batch 301. Saved: the "
        "simulations up to batch 300. Lost: nothing else.", "the simulation of batch 301 failed", True, True,
        "Continue from batch 300", False)
    run = held.status.run
    said = [line for line in text.texts() if line.startswith("Try ")]
    filed = (tmp_path / "runs" / f"{run}.txt").read_text(encoding="utf-8").splitlines()
    assert said == ["Try 2 begins here.", "Try 3 begins here.", "Try 4 begins here."], said
    assert ([line[20:] for line in filed if " window Try " in line], filed[-1][20:]) == (
        [f"window {line}" for line in said], f"window {retry.STOPPED_TRYING_TEXT}"), filed[-3:]
    record = history.read(tmp_path / "runs" / f"{run}.json")
    assert [(datetime.fromisoformat(one["started"]), datetime.fromisoformat(one["ended"]), one["outcome"],
             one["failure"], one["batches_before"], one["batches_after"], one["epochs_before"], one["epochs_after"])
            for one in record["tries"]] == [
        (_at(3, 3, 2, 11), _at(3, 3, 12, 11), "failed", "RuntimeError", 0, 236, None, 0),
        (_at(3, 3, 13, 11), _at(3, 3, 13, 11), "failed", "RuntimeError", 236, 300, 0, 0),
        (_at(3, 3, 14, 11), _at(3, 3, 14, 11), "failed", "SimulationError", 300, 300, 0, 0),
        (_at(3, 3, 17, 11), _at(3, 3, 17, 11), "failed", "RuntimeError", 300, 300, 0, 0)], record["tries"]
    assert (record["outcome"], record["failure"]["kind"], record["failure"]["reopen"]) == (
        "failed", "out of memory", False)

    finishing = StandIn()
    start_run(host, finishing, spec=_training(), on_result=results.append)
    _to_the_simulation(app, finishing, text, held, folder, 300)
    finishing.fail(_out_of_memory("the simulation of batch 301"))
    _waiting(app, held)
    clock.advance(60)                                   # 03:18
    held.tick()
    _tried(app, held, finishing, 2)
    finishing.release("the posterior")
    _ended(app, held)
    assert (held.status.outcome, results) == ("finished", ["the posterior"]), (
        "a try's result was not handed where the run's was")

    asked_to_stop, before = StandIn(takes_stops=False), len(text.statuses)
    start_run(host, asked_to_stop, spec=_training())
    _set(tmp_path, 200)                                 # the set holds 200 batches; the run's text counts 236
    _to_the_simulation(app, asked_to_stop, text, held, folder, 236)
    choose(monkeypatch, words="Stop and keep the simulations")
    view.stop_button.click()
    asked_to_stop.fail(_out_of_memory())               # before the stop is taken
    _ended(app, held)
    assert ([status.state for status in text.statuses[before:] if status.state == "retrying"],
            len(asked_to_stop.calls), held.status.outcome, view.noted.isVisibleTo(view)) == ([], 1, "failed", True), (
        "a run whose stop was asked for was tried again after its failure")
    assert (_failure(view), view.continue_button.text()) == (
        "On Sat 3 Oct, 03:18, during simulation: the graphics card ran out of memory at batch 237. Saved: the "
        "simulations up to batch 200. Lost: nothing else.", "Continue from batch 200"), (
        "the cause does not name the batch in progress, or what is saved is not what the set holds")


def test_a_run_waiting_to_be_tried_again_keeps_the_computer_awake_and_shutting_down_anyway_ends_its_tries(
        tmp_path, windows_asked):
    """A run that waits to be tried again is live: the computer is kept awake, asked once as the run starts and let
    sleep only at its end, and the reason against ending the session stays registered, through each pause and the
    try after it, each asked from the window's thread. Windows' question meanwhile is answered that PRISM blocks the
    end. "Shut down anyway" ends the tries, as any stop asked while the run waits does: the window's wait returns, the
    run ends failed, its history says that Windows shut down and when the run ended, its text says the stop's reason
    and then that the tries stopped, and the computer is let sleep, the reason withdrawn."""
    from PySide6.QtCore import QTimer
    from core.gui.run import history
    from tests._fixtures import build_window
    app = qt_app()
    clock = Clock(_at(3, 3, 2, 11))
    held = _controller_on(tmp_path, clock)
    window, host = build_window(), build_run_host()
    text, stand_in, folder = RunText(held), StandIn(), _set(tmp_path, 236)
    simulating = ("Training retrain_3 is simulating. Shutting down stops it; the simulations done so far are saved "
                  "first.")

    def asked_of_windows():
        """Each request to keep the computer awake, and each reason registered (None: withdrawn), so far."""
        assert all(thread == "MainThread" for _name, _args, thread in windows_asked), windows_asked
        return ([args[0] for name, args, _thread in windows_asked if name == "keep_awake"],
                [args[1] for name, args, _thread in windows_asked if name == "block_shutdown"])

    def standing(when):
        awake, reasons = asked_of_windows()
        assert (awake, reasons[-1:], None in reasons) == ([True], [simulating], False), (
            f"{when}: the computer was let sleep, or the reason withdrawn: {awake}, {reasons}")

    start_run(host, stand_in, spec=_training())
    _to_the_simulation(app, stand_in, text, held, folder, 236)
    stand_in.fail(_out_of_memory())
    _waiting(app, held)
    standing("the first pause")
    clock.advance(60)
    held.tick()
    _tried(app, held, stand_in, 2)
    standing("the second try")
    stand_in.fail(_out_of_memory("the simulation of batch 237"))
    _waiting(app, held)
    standing("the second pause")

    cancelled = []
    window.session_asked(SimpleNamespace(cancel=lambda: cancelled.append(True)))
    assert cancelled == [True], "while the run waits to be tried again Windows was not answered that PRISM blocks"
    QTimer.singleShot(0, lambda: clock.advance(5))      # the tries end a moment after "shut down anyway"
    late = QTimer()                                      # were the tries not ended, the next try begins and ends it
    late.setSingleShot(True)
    late.timeout.connect(lambda: (clock.advance(600), held.tick()))
    late.start(10_000)
    try:
        window.session_ending(True)
    finally:
        late.stop()
    record = history.read(tmp_path / "runs" / f"{held.status.run}.json")
    assert (held.is_live(), held.status.outcome, len(stand_in.calls), record["outcome"], record["stop_reason"],
            datetime.fromisoformat(record["ended"])) == (False, "failed", 2, "failed", "Windows shut down", clock()), (
        "shutting down anyway during a pause did not end the tries, or the history did not say why and when")
    assert text.texts("window")[-2:] == ["A stop was asked for: Windows shut down.", retry.STOPPED_TRYING_TEXT]
    awake, reasons = asked_of_windows()
    assert (awake, reasons[-2:], reasons.count(None)) == ([True, False], [simulating, None], 1), (
        "the run's end did not let the computer sleep, or did not withdraw the reason")


def test_stop_trying_pressed_as_the_pause_runs_out_begins_no_try(tmp_path):
    """"Stop trying" pressed in the turn of the event loop in which the pause runs out, the window's look at its
    clock coming before the tries have ended, begins no try: the run ends failed, as a press earlier in the pause
    ends it."""
    app = qt_app()
    clock = Clock(_at(3, 3, 2, 11))
    held, host = _controller_on(tmp_path, clock), build_run_host()
    text, view = RunText(held), host.run_view
    stand_in = StandIn()
    start_run(host, stand_in, spec=_training())
    _to_the_simulation(app, stand_in, text, held, _set(tmp_path, 236), 236)
    stand_in.fail(_out_of_memory())
    _waiting(app, held)
    clock.advance(60)                                   # the pause has run out, and nothing has looked yet
    view.stop_trying_button.click()
    held.tick()                                         # the timer's look, in the turn of the press
    turn_until(app, lambda: not held.is_live() or len(stand_in.calls) == 2)
    if held.is_live():                                  # a try began after all: it is let end, as a later run needs
        stand_in.release()
        _ended(app, held)
    assert (len(stand_in.calls), held.status.outcome) == (1, "failed"), (
        "a try began though 'Stop trying' was pressed in the turn its pause ran out")


_REPORTED_AT_ONCE = {
    "a full disk": (lambda: OSError(28, "No space left on device"), "[Errno 28] No space left on device", False),
    "a programming error": (lambda: TypeError("unsupported operand type(s) for +: 'int' and 'NoneType'"),
                            "unsupported operand type(s) for +: 'int' and 'NoneType'", False),
    "a refusal after the spend": (lambda: Refusal("The read-back of the posterior was refused.", field="name"),
                                  f"The read-back of the posterior was refused. {fix_sentence('name')}", True),
    "the card's memory, in a check": (_out_of_memory, "the simulation of batch 237 failed", False),
}


@pytest.mark.parametrize("failure", list(_REPORTED_AT_ONCE))
def test_any_other_failure_is_reported_at_once(failure, tmp_path):
    """A failure that the graphics card's lack of memory did not cause fails the same way again, so the window says
    it at once: no pause and no try, the run ended failed, its page saying what failed and offering to continue
    from the batches the simulation set holds; a refusal made once something was spent is quoted with where to fix
    it, as the window says a refusal. The error's message stands beneath only where it adds to what the failure's
    block says: a refusal's place to fix it does, a message the block quotes whole does not. A run that is no training
    is not tried again, whatever its failure."""
    app = qt_app()
    clock = Clock(_at(3, 3, 2, 11))
    held, host = _controller_on(tmp_path, clock), build_run_host()
    text, view = RunText(held), host.run_view
    error, said, adds = _REPORTED_AT_ONCE[failure]
    stand_in = StandIn()
    if failure.endswith("in a check"):
        start_run(host, stand_in, spec=RunSpec("Check", ("inference", "checks"), name="Calibration"))
    else:
        start_run(host, stand_in, spec=_training())
        _to_the_simulation(app, stand_in, text, held, _set(tmp_path, 236), 236)
    stand_in.fail(error())
    _ended(app, held)
    assert ([status.state for status in text.statuses if status.state == "retrying"], len(stand_in.calls),
            held.status.outcome, all_asked(), view.failure.isVisibleTo(view), view.failure_message.text(),
            view.failure_message.isVisibleTo(view)) == ([], 1, "failed", [], True, said, adds), (
        "the failure was tried again, told in a box, or not said on its page, or its message is quoted beneath where "
        "it adds nothing, or not where it adds")
    if failure.endswith("in a check"):
        assert (_failure(view), view.continue_button.isVisibleTo(view)) == (f"On Sat 3 Oct, 03:02: {said}.", False)
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
    well as a try's, in the simulation, where the card's error comes under the simulator's own, as in the fit, which
    is never tried again, and whose history says the card failed. The page names the cause: that trying again did
    not help, or that the card failed, quoting the first line of its error. It says how the training goes on: started
    again once PRISM is open again, it continues the simulation set from the batches it holds; a training kept in its
    fit is resumed from the command line, the command set off on its own beneath the sentence. Neither "Continue from
    batch N" nor "Resume the training" is offered: either would run in this copy of PRISM."""
    from core.gui.run import history
    app = qt_app()
    clock = Clock(_at(3, 3, 2, 11))
    held, host = _controller_on(tmp_path, clock), build_run_host()
    text, view = RunText(held), host.run_view
    folder = _set(tmp_path, 236)
    again = ("Close PRISM and open it again, then start the same training: it continues the saved simulations from "
             "batch 236.")
    simulated = f"{_SOLVER}: AcceleratorError: {_ILLEGAL}"
    device_opening = _DEVICE.format(simulated.splitlines()[0])

    def reopened(message, opening=_REOPEN):
        return (held.status.state, held.status.outcome, held.status.reopen, all_asked(), _failure(view),
                view.failure_message.text(), view.failure_message.isVisibleTo(view),
                view.continue_button.isVisibleTo(view), view.resume_button.isVisibleTo(view),
                view.stop_trying_button.isVisibleTo(view)) == (
            "ended", "failed", True, [], f"{opening} {again}", message, True, False, False, False)

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
    assert (len(device.calls), reopened(simulated, device_opening)) == (2, True), _failure(view)

    first = StandIn()
    start_run(host, first, spec=_training())
    _to_the_simulation(app, first, text, held, folder, 236)
    first.fail(_device_failure())
    _ended(app, held)
    assert (len(first.calls), reopened(simulated, device_opening)) == (1, True), (
        "a first failure of the graphics card that is no lack of memory was tried again, offered to continue, or its "
        "page named another cause", _failure(view))
    assert history.read(tmp_path / "runs" / f"{held.status.run}.json")["failure"]["kind"] == "device", (
        "the history does not say the graphics card failed")

    kept = build_fit_record(store, passes=2)
    in_the_fit = StandIn()
    in_the_fit.kept_fit = _note(kept, 2)
    start_run(host, in_the_fit, spec=_training())
    _to_the_fit(app, in_the_fit, text, clock, _set(tmp_path, 10_000, complete=True), [(1, 0), (2, 1)])
    in_the_fit.fail(_device_failure(in_the_simulation=False))
    _ended(app, held)
    assert (len(in_the_fit.calls), held.status.reopen, _failure(view), _said_alone(view),
            view.resume_button.isVisibleTo(view), view.continue_button.isVisibleTo(view)) == (
        1, True, f"{_DEVICE.format(_ILLEGAL.splitlines()[0])} Close PRISM and open it again. The training is kept "
                 f"up to epoch 2; until the Training page offers to resume it, it is resumed from the command line "
                 f"with the command below, given the same --bounds, --prior and --chi it was trained with.",
        f"python -m core train --resume-fit {kept}", False, False), _failure(view)


def _laid_out(label):
    """A label's words laid out at its width, as they may break: inside a word that is wider than a line, as a long
    path is, and nowhere else but between words; the lines one under the other from the label's top left."""
    from PySide6.QtCore import QPointF
    from PySide6.QtGui import QTextLayout, QTextOption
    layout = QTextLayout(label.text().replace("\n", chr(0x2028)), label.font())
    option = QTextOption()
    option.setWrapMode(QTextOption.WrapAtWordBoundaryOrAnywhere)
    layout.setTextOption(option)
    layout.beginLayout()
    top = 0.0
    while True:
        line = layout.createLine()
        if not line.isValid():
            break
        line.setLineWidth(label.width())
        line.setPosition(QPointF(0, top))
        top += line.height()
    layout.endLayout()
    return layout


def _lines_needed(label) -> tuple:
    """How many lines a label's words take at its width, and how high they stand, laid out as ``_laid_out`` lays
    them."""
    layout = _laid_out(label)
    return layout.lineCount(), sum(layout.lineAt(index).height() for index in range(layout.lineCount()))


def _at_the_end_of(layout, index):
    """The point at the end of the words of a line of ``layout``, and the place in the words it is nearest, counted
    as the layout counts them."""
    from PySide6.QtCore import QPoint
    line = layout.lineAt(index)
    x = line.naturalTextWidth() + 1
    return QPoint(round(x), round(line.y() + line.height() / 2)), line.xToCursor(x)


def _menu_of(label) -> tuple:
    """The right button's menu of ``label``, opened as a person opens it, then closed: each entry's words, whether it
    is a line, and whether it can be pressed; whether it stands in sight, is modal, and is deleted when it closes;
    whether it ran a loop of its own, inside which a timer closes it; whether the label kept its selection while the
    menu was open; and whether any menu is left once it has closed."""
    from PySide6.QtCore import QEvent, QPoint, QTimer
    from PySide6.QtGui import QContextMenuEvent
    from PySide6.QtWidgets import QApplication, QMenu
    looped = []

    def close_inside():
        looped.append(True)
        for menu in label.findChildren(QMenu):
            menu.close()

    closer = QTimer()
    closer.setSingleShot(True)
    closer.timeout.connect(close_inside)
    closer.start(0)                          # turned only by a loop the menu runs before the event returns
    point = QPoint(2, 2)
    QApplication.sendEvent(label, QContextMenuEvent(QContextMenuEvent.Mouse, point, label.mapToGlobal(point)))
    closer.stop()
    menus = label.findChildren(QMenu)
    entries = [(action.text(), action.isSeparator(), action.isEnabled()) for menu in menus for action in menu.actions()]
    kind = [(menu.isVisible(), menu.isModal(), menu.testAttribute(Qt.WA_DeleteOnClose)) for menu in menus]
    kept = label.hasSelectedText()
    for menu in menus:
        menu.close()
    QApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    return entries, kind, bool(looped), kept, bool(label.findChildren(QMenu))


def test_a_long_command_and_a_long_message_wrap_within_a_narrow_page_and_the_command_is_copied_whole(tmp_path,
                                                                                                    store, monkeypatch):
    """On a run page 326 px wide, as at the window's least size, the command that resumes a training only reopening
    PRISM can help, naming its bounds file by a long path, takes as many lines as it needs, broken inside the path
    where it must: its label is as tall as those lines. A person selects it with the mouse, pressing at its first
    character and letting go at the end of its last line: the press gives it the keyboard, and the keyboard's copy,
    sent where the keyboard is, copies the command exactly, one line with nothing added. Neither the move of the drag
    nor a repaint lays its words out again. Its right button's menu is the toolkit's own for a selectable label, entry
    for entry: "Copy", a line, "Select All", each pressable as the toolkit's is, with a selection and without one, and
    for no words; a popup, not modal and running no loop of its own, deleted once it closes, while which the selection
    stays. The selection ends when the keyboard goes to another control; a double click selects the command whole. A
    failure's own message beneath, a long path among its words and in it a character outside the Basic Multilingual
    Plane, wraps the same way; a drag across that character copies exactly what is highlighted, and a double click
    highlights it to its end. The clipboard is put back as it was."""
    from PySide6.QtCore import QPoint
    from PySide6.QtGui import QTextLayout
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication
    from core.gui.run import view as view_module
    from tests._look_helpers import _settle
    from tests._window_helpers import in_front, shown_off_screen
    app = qt_app()
    clock = Clock(_at(3, 3, 2, 11))
    held, host = _controller_on(tmp_path, clock), build_run_host()
    text, view = RunText(held), host.run_view
    view.setFixedWidth(326)
    shown_off_screen(host, 1300, 1000)
    bounds = (r"C:\Users\J\OneDrive - Hair Bundle Lab\PRISM inputs\Bounds\nadrowski"
              r"\master_tier1_spontaneous_and_driven_wide_box_october_retrain.txt")
    cfg = SimpleNamespace(sources={"bounds": bounds}, chi_mode=True,
                          hw=SimpleNamespace(device=SimpleNamespace(type="cuda")))
    prior = SimpleNamespace(kind="prior", id="20261001T150211", name="retrain_prior")
    kept = build_fit_record(store, passes=2)
    in_the_fit = StandIn()
    in_the_fit.kept_fit = _note(kept, 2)
    _start_with(host, in_the_fit, _training(), cfg, prior, None, True, name="retrain_3")
    _to_the_fit(app, in_the_fit, text, clock, _set(tmp_path, 10_000, complete=True), [(1, 0), (2, 1)])
    in_the_fit.fail(_device_failure(in_the_simulation=False))
    _ended(app, held)
    _settle()
    line = view.command_line
    command = f'python -m core train --resume-fit {kept} --bounds "{bounds}" --prior retrain_prior --chi'
    lines, needed = _lines_needed(line)
    assert (line.text(), line.isVisibleTo(view)) == (command, True), line.text()
    assert lines > 1 and line.width() <= view.width() and line.height() >= needed - 1, (
        "the command is not wrapped within the page", lines, line.height(), needed, line.width())
    clipboard = QApplication.clipboard()
    was = clipboard.text()
    toolkit = QLabel(host)                                   # the toolkit's own selectable label, for its menu
    toolkit.setTextInteractionFlags(Qt.TextSelectableByMouse)
    toolkit.show()
    try:
        with in_front(host):
            layout = _laid_out(line)
            first = QPoint(1, round(layout.lineAt(0).height() / 2))
            last, end = _at_the_end_of(layout, layout.lineCount() - 1)
            QTest.mousePress(line, Qt.LeftButton, Qt.NoModifier, first)
            assert QApplication.focusWidget() is line, ("a press does not give the command the keyboard",
                                                        QApplication.focusWidget())
            laid = []

            class Counted(QTextLayout):
                def __init__(self, *told):
                    laid.append(told[0] if told else "")
                    super().__init__(*told)

            with monkeypatch.context() as patch:
                patch.setattr(view_module, "QTextLayout", Counted)
                QTest.mouseMove(line, last)
                line.grab()
                line.grab()
            QTest.mouseRelease(line, Qt.LeftButton, Qt.NoModifier, last)
            assert laid == [], ("the command's words were laid out again", len(laid))
            clipboard.clear()
            QTest.keyClick(QApplication.focusWidget(), Qt.Key_C, Qt.ControlModifier)
            assert (end, clipboard.text()) == (len(command), command), (
                "the command dragged over and copied with the keyboard is not the command", end, repr(clipboard.text()))
            toolkit.setText(command)
            toolkit.setSelection(0, len(command))
            ours, reference = _menu_of(line), _menu_of(toolkit)
            assert ours == reference and reference[1:] == ([(True, False, True)], False, True, False), (
                "the right button's menu is not the toolkit's", ours, reference)
            view.back_button.setFocus(Qt.TabFocusReason)
            assert not line.hasSelectedText(), ("the command keeps its selection after the keyboard left it",
                                                line.selectedText())
            toolkit.setSelection(0, 0)
            assert _menu_of(line)[:3] == _menu_of(toolkit)[:3], "the menu with no selection"
            QTest.mouseDClick(line, Qt.LeftButton, Qt.NoModifier, first)
            assert (QApplication.focusWidget(), line.selectedText()) == (line, command), (
                "a double click does not select the command whole", QApplication.focusWidget(), line.selectedText())
            with warnings.catch_warnings():                  # the toolkit marks the call as old, and keeps it
                warnings.simplefilter("ignore", DeprecationWarning)
                QApplication.setActiveWindow(None)           # the window loses the keyboard, to another program
                away = line.selectedText()
                QApplication.setActiveWindow(host)           # and is brought back
            assert away == command, ("the command lost its selection as the window lost the keyboard", away)
        view.back_button.click()
        path = (f"C:/Users/J/PRISM/Artifacts/simulations/_unnamed__20261003T030211/{chr(0x1F4C1)}"
                f"{'very_long_batch_name_' * 3}6150.pt")
        unwritten = StandIn()
        start_run(host, unwritten, spec=_training())
        _to_the_simulation(app, unwritten, text, held, _set(tmp_path, 236), 236)
        unwritten.fail(RuntimeError(f"the batch could not be written to its file\n{path}"))    # the path adds to it
        _ended(app, held)
        _settle()
        message = view.failure_message
        lines, needed = _lines_needed(message)
        assert path in message.text() and lines > 1 and message.width() <= view.width(), (lines, message.text())
        assert message.height() >= needed - 1, ("the message is not wrapped within the page", lines, message.height(),
                                                needed)
        words = message.text().encode("utf-16-le")
        units = words.find(chr(0x1F4C1).encode("utf-16-le")) // 2                       # where the character is
        layout = _laid_out(message)
        holds = next(index for index in range(layout.lineCount())
                     if layout.lineAt(index).textStart() + layout.lineAt(index).textLength() > units)
        last, end = _at_the_end_of(layout, holds + 1)               # the end of the line after the character's
        assert units + 2 <= end < len(words) // 2, ("the drag does not cross the character, or takes it all", units,
                                                     end, len(words) // 2)
        with in_front(host):
            QTest.mousePress(message, Qt.LeftButton, Qt.NoModifier, QPoint(1, round(layout.lineAt(0).height() / 2)))
            QTest.mouseMove(message, last)
            QTest.mouseRelease(message, Qt.LeftButton, Qt.NoModifier, last)
            marked = [(shown.start, shown.length) for shown in message._marked()]
            clipboard.clear()
            QTest.keyClick(QApplication.focusWidget(), Qt.Key_C, Qt.ControlModifier)
            highlighted = words[:2 * end].decode("utf-16-le")
            assert (marked, clipboard.text()) == ([(0, end)], highlighted), (
                "the copy is not what is highlighted", marked, end, repr(clipboard.text()), repr(highlighted))
            QTest.mouseDClick(message, Qt.LeftButton, Qt.NoModifier, last)
            marked = [(shown.start, shown.length) for shown in message._marked()]
            assert marked == [(0, len(words) // 2)], ("the whole message is not highlighted", marked, len(words) // 2)
        message.setText("")
        toolkit.setText("")
        assert _menu_of(message)[:3] == _menu_of(toolkit)[:3], "the menu for no words"
    finally:
        clipboard.setText(was)
        toolkit.deleteLater()


# ── continuing and resuming from the run's page ──
def test_continue_from_batch_n_repeats_the_runs_own_call_with_the_must_continue_policy(tmp_path, monkeypatch):
    """"Continue from batch N" stands on the page of a training that failed, and of one that was stopped, while its
    simulation set is unfinished and holds a batch, N the batches the set holds as read from it once the run has
    ended, which may be fewer than the run's text counted, and the page says that the simulations up to N are
    saved; a load from the same page meanwhile, which is handed no page, leaves the offer and its call as they are.
    Pressed, it starts the run's own call again as the page's next run, with files of its own: its keywords
    as they were, the policy that must continue (``resume="require"``) the one new argument and N in none of them,
    its result handed where the first run's was; the page, which held the failed run's figure, keeps no frame for the
    new run's none from the moment it takes the run. A run stopped before its set held a batch offers nothing; neither
    does a page drawn from a run's files alone; and while another run is live, the press starts nothing and the page
    says so."""
    from core.gui.run import history
    from core.gui.run.controller import LIVE_TEXT
    app = qt_app()
    clock = Clock(_at(3, 3, 2, 11))
    held, host = _controller_on(tmp_path, clock), build_run_host()
    text, view = RunText(held), host.run_view
    folder, results = _set(tmp_path, 200), []          # the set holds 200 batches; the run's text counted 236
    base = {"name": "retrain_3", "note": "the retrain", "num_runs": 10_000}
    failed = StandIn()
    start_run(host, failed, spec=_training(), on_result=results.append, **base)
    host.figure_stack.add_figure("retrain_3's loss", b"")
    _to_the_simulation(app, failed, text, held, folder, 236)
    failed.fail(OSError(28, "No space left on device"))
    _ended(app, held)
    first = held.status.run
    assert (view.continue_button.isVisibleTo(view), view.continue_button.text(), view.resume_button.isVisibleTo(view),
            _failure(view).endswith("Saved: the simulations up to batch 200. Lost: nothing else.")) == (
        True, "Continue from batch 200", False, True), ("N is not the batches the set holds", _failure(view))
    record, path = history.read(tmp_path / "runs" / f"{first}.json"), tmp_path / "runs" / f"{first}.txt"
    loads = StandIn(word="Posterior load")                # a load from the same page, which is handed no page
    start_run(host, loads, spec=RunSpec("Posterior load", ("inference", "training"), stop_button="Stop loading",
                                        loads_only=True, keeps_files=False, stop_asks=False))
    loads.release()
    _ended(app, held)
    assert (view.continue_button.isVisibleTo(view), view.notice.text(), view.figures.isVisibleTo(view)) == (
        True, "", True), "a load from the page took its offer away, or said something on it, or hid the run's figure"
    shown, framed = host.show_run, []
    # the page read as it takes the new run, before the run's controller draws it
    monkeypatch.setattr(host, "show_run", lambda: (shown(), framed.append(view.figures.isVisibleTo(view))))
    view.continue_button.click()
    assert framed == [False], "the page, holding the figures already, kept their frame for a run that has made none"
    turn_until(app, lambda: len(failed.calls) == 2)
    assert (held.is_live(), held.status.run != first, held.status.name, view.mode,
            view.continue_button.isVisibleTo(view)) == (True, True, "retrain_3", "live", False)
    assert failed.calls == [((), base), ((), {**base, "resume": "require"})], failed.calls
    assert "200" not in repr(failed.calls[1]), "the batch the set holds was handed to the stage"
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
    the form", in no box; once back at the form, that run has ended, and a figure that comes then is none of its own,
    which "How the run went" does not show. Once Discard has deleted the kept training, nothing is offered to
    resume."""
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
    host.figure_stack.add_figure("opened after the refusal", b"")       # a figure of no run: a saved record's
    view.how_link.click()
    assert (view.figures.isVisibleTo(view), host.figure_stack.in_sight()) == (False, 0), (
        "a figure that came after the refused resume shows on that run's page as its own")
    view.back_button.click()

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


def test_continue_and_resume_start_nothing_once_the_page_holds_another_model_or_prior(tmp_path, monkeypatch, store):
    """"Continue from batch N" and "Resume the training" repeat the run's own call, made with the model and the prior
    its page's session held then. While the session still holds them, the press starts the run; once it holds another
    model (one applied in step 1) or another prior (one built or loaded), the press starts nothing: the page says why
    and offers it no more, and the question it asks of the training a stop kept stays asked."""
    from core.gui.session import ConfigDraft, SbiSession
    from tests._fixtures import build_inference
    app = qt_app()
    clock = Clock(_at(3, 3, 2, 11))
    cfg, prior = SimpleNamespace(model="NADROWSKI"), SimpleNamespace(kind="prior", id="20261001T150211",
                                                                     name="retrain_prior")
    screen = build_inference(session=SbiSession(cfg=cfg, inf_prior=prior))
    held, host = _controller_on(tmp_path, clock), screen.posterior_panel
    text, view = RunText(held), host.run_view
    failed, folder = StandIn(), _set(tmp_path, 236)
    _start_with(host, failed, _training(), cfg, prior, None, True)
    _to_the_simulation(app, failed, text, held, folder, 236)
    failed.fail(OSError(28, "No space left on device"))
    _ended(app, held)
    view.continue_button.click()                        # the session holds the model and the prior of the call
    turn_until(app, lambda: len(failed.calls) == 2)
    _to_the_simulation(app, failed, text, held, folder, 236)
    failed.fail(OSError(28, "No space left on device"))
    _ended(app, held)
    assert view.continue_button.isVisibleTo(view), "the continued run's page does not offer to continue it"
    screen.new_draft(ConfigDraft(model="NADROWSKI", labels=[], state_dep_drift=False))     # a model applied again
    view.continue_button.click()
    assert (held.is_live(), len(failed.calls), view.notice.text(), view.continue_button.isVisibleTo(view)) == (
        False, 2, _CHANGED, False), "a continue ran in a session that holds another model"

    choose(monkeypatch, words="Stop training")
    stopped = StandIn()
    stopped.kept_fit = _note(build_fit_record(store, passes=3), 3)
    screen.session.cfg, screen.session.inf_prior = cfg, prior
    _start_with(host, stopped, _training(), cfg, prior, None, True)
    _to_the_fit(app, stopped, text, clock, _set(tmp_path, 10_000, complete=True), [(1, 0), (2, 1), (3, 2)])
    view.stop_button.click()
    _ended(app, held)
    screen.session.inf_prior = SimpleNamespace(kind="prior", id="20261002T150211", name="another_prior")
    view.resume_button.click()
    assert (held.is_live(), len(stopped.calls), view.notice.text(), view.resume_button.isVisibleTo(view),
            view.keep_block.isVisibleTo(view)) == (False, 1, _CHANGED, False, True), (
        "a resume ran in a session that holds another prior, or the question of keeping the training went")


def test_a_failure_during_the_fit_offers_resume_and_says_what_was_saved(tmp_path, store):
    """A training that fails during its fit says so, at the epoch the training it kept stands at (the note's, never
    the records the run said, one of which may be an epoch whose save did not hold), that the simulations and the
    training up to that epoch are saved and that at most the epoch in progress is lost, and offers to resume it; the
    history's fit says the same epochs, and its try, the one the run ended with, what the kept training and the set
    held after it, as a try that was tried again says them. One whose fit is not saved as it goes says the fit is
    lost; one whose fit had kept no epoch yet says so; neither offers anything."""
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
                view.continue_button.isVisibleTo(view)) == ([], f"On Sat 3 Oct, {when}, {said}", resumes, False), (
            _failure(view))
    record = history.read(tmp_path / "runs" / f"20261003T030211-training.json")
    assert (record["fit"]["epochs"], record["tries"][-1]["epochs_after"], record["tries"][-1]["batches_after"]) == (
        41, 41, 10_000), "the history's fit or its last try holds the records seen, not the kept training and the set"


def test_a_failed_runs_history_keeps_its_failure_once_its_end_is_seen_or_set_aside(tmp_path, store):
    """Once a run has ended, every later write of its history reads the file back and changes only what that write
    is for. A failed training's history keeps its failure -- its kind, its message, whether only reopening PRISM
    helps -- and each try's failure, once its end is seen, its page keeping its failure and the command it names;
    and the past view drawn from that history names the cause: so for a training out of memory in its fit after the
    tries that did not help, and for one that failed with an error of its own. The same once a run that started
    while the failure waited to be seen sets it aside. A history spoiled after the run's end is written whole
    again, from what its end wrote, once its end is seen."""
    from core.gui.run import history
    app = qt_app()
    clock = Clock(_at(3, 3, 2, 11))
    held, host = _controller_on(tmp_path, clock), build_run_host()
    text, view = RunText(held), host.run_view
    folder = _set(tmp_path, 10_000, complete=True)
    kept = build_fit_record(store, passes=2)
    own = "unsupported operand type(s) for +: 'int' and 'NoneType'"

    def out_of_memory_in_the_fit():
        stand_in = StandIn()
        stand_in.kept_fit = _note(kept, 2)
        start_run(host, stand_in, spec=_training())
        _to_the_fit(app, stand_in, text, clock, folder, [(1, 0), (2, 1)])
        stand_in.fail(_out_of_memory("the fit's epoch 3"))
        stand_in.kept_fit = None                        # three tries in a row that keep nothing more
        for pause, calls in ((60, 2), (180, 3), (600, 4)):
            _waiting(app, held)
            clock.advance(pause)
            held.tick()
            _tried(app, held, stand_in, calls)
            stand_in.fail(_out_of_memory("the fit's epoch 3"))
        _ended(app, held)

    def an_error_of_its_own():
        stand_in = StandIn()
        start_run(host, stand_in, spec=_training())
        _to_the_fit(app, stand_in, text, clock, folder, [(1, 0)])
        stand_in.fail(TypeError(own))
        _ended(app, held)

    def ended():
        """The run that ended: its id, its history file, what that file says, and what its page says."""
        run = held.status.run
        path = tmp_path / "runs" / f"{run}.json"
        return run, path, history.read(path), (_failure(view), _said_alone(view), view.failure.isVisibleTo(view))

    for fail, cause, failure in ((out_of_memory_in_the_fit, "the graphics card ran out of memory",
                                  {"type": "RuntimeError", "message": "the fit's epoch 3 failed",
                                   "kind": "out of memory", "reopen": True}),
                                 (an_error_of_its_own, own,
                                  {"type": "TypeError", "message": own, "kind": "other", "reopen": False})):
        fail()
        run, path, at_the_end, page = ended()
        assert (at_the_end["failure"], at_the_end["tries"][-1]["failure"], page[2]) == (
            failure, failure["type"], True), at_the_end
        held.acknowledge(run)
        assert history.read(path) == {**at_the_end, "acknowledged": True}, (
            "seeing the end changed its history", history.read(path))
        assert ended()[3] == page, "seeing the end changed the failure its page says"
        past = build_run_host().past_view
        past.show_past(history.read(path), tmp_path / "runs" / f"{run}.txt", clock=held.now)
        assert f"during the fit: {cause}." in past.sentence.text(), past.sentence.text()

    an_error_of_its_own()                               # its end waits to be seen as the next run starts
    run, path, at_the_end, _page = ended()
    later = StandIn()
    start_run(host, later, spec=_training())
    later.release()
    _ended(app, held)
    assert (held.status.run != run, history.read(path)) == (True, {**at_the_end, "acknowledged": True}), (
        "setting the end aside changed its history", history.read(path))

    an_error_of_its_own()
    run, path, at_the_end, _page = ended()
    path.write_text("not a run's history", encoding="utf-8")       # spoiled after the run's end
    held.acknowledge(run)
    assert history.read(path) == {**at_the_end, "acknowledged": True}, (
        "a history spoiled after the run's end was not written again from what its end wrote")


def test_a_failure_while_the_posterior_is_saved_says_the_training_is_kept_with_its_fit_complete(tmp_path, store):
    """A training whose fit finished but whose posterior could not be written says so, and that the training is kept
    with its fit complete, so that resuming it writes the posterior without training again, and offers to resume it;
    one whose training was not saved as it went says that none was kept, and offers nothing. One that ran out of the
    graphics card's memory as its posterior was saved says, while it waits to be tried again, that the training is
    kept with its fit complete."""
    app = qt_app()
    clock = Clock(_at(3, 3, 2, 11))
    held, host = _controller_on(tmp_path, clock), build_run_host()
    text, view = RunText(held), host.run_view
    folder = _set(tmp_path, 10_000, complete=True)
    kept = build_fit_record(store, passes=2, complete=True)
    for saved, note, said in ((True, _note(kept, 2), "The fit finished but its record could not be written. The "
                                                     "training is kept with its fit complete: resuming it writes the "
                                                     "posterior without training again."),
                              (False, None, "The fit finished but its record could not be written; the training was "
                                            "not kept.")):
        stand_in = StandIn()
        stand_in.kept_fit = note
        start_run(host, stand_in, spec=_training())
        _to_the_fit(app, stand_in, text, clock, folder, [(1, 0), (2, 20)], saved=saved)   # the fit has ended
        assert held.model.stage == "Saving"
        stand_in.fail(OSError(13, "Permission denied"))
        _ended(app, held)
        assert (all_asked(), _failure(view), view.failure_message.text(), view.failure_message.isVisibleTo(view),
                view.resume_button.isVisibleTo(view), view.continue_button.isVisibleTo(view)) == (
            [], said, "[Errno 13] Permission denied", True, saved, False), saved
    stand_in = StandIn()
    stand_in.kept_fit = _note(kept, 2)
    start_run(host, stand_in, spec=_training())
    _to_the_fit(app, stand_in, text, clock, folder, [(1, 0), (2, 20)])
    assert held.model.stage == "Saving"
    stand_in.fail(_out_of_memory("the posterior's save"))
    _waiting(app, held)
    assert view.sentence.text() == ("The graphics card ran out of memory as the posterior was being saved. Trying "
                                    "again in 1 minute: the first of three tries. The training is kept with its fit "
                                    "complete."), view.sentence.text()
    view.stop_trying_button.click()
    _ended(app, held)


def test_a_try_after_the_fit_ran_out_of_memory_resumes_the_kept_training(tmp_path, store):
    """After the graphics card ran out of memory during the fit, a try is the run's own call with the training it
    kept named (``resume_fit``), as Resume is, and nothing else new: neither the set's policy nor anything of a
    narrowing round's region; meanwhile the page and the history's fit say the epochs kept, the note's. The try
    resumes the kept training from its kept epoch: an epoch an earlier try recorded past it, whose save did not
    hold, leaves the curve, and the best epoch is counted again from the kept one. A try that keeps more epochs and
    then runs out of memory again starts a new count; a try that fails before it names the kept training keeps the
    one named before; when three tries in a row keep none, the page says only reopening PRISM helps, and that the
    kept training is resumed from the command line until its page offers it: the tool's command for the run,
    ``train`` for a training, ``tsnpe`` for a narrowing round, with the flags of what the run's own call trained
    from, or, where the call does not tell them, words that say to give them as they were. The command stands on its
    own beneath the sentence, so that copied off the page it is one the command-line tool takes."""
    from core.gui.run import history
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

    assert (view.sentence.text(), history.read(tmp_path / "runs" / f"{held.status.run}.json")["fit"]["epochs"]) == (
        waits("1 minute", "first", 2), 2), "while the run waits, the page or its history names the records' epochs"
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
    command = (f"python -m core train --resume-fit {kept} --bounds Resources/Bounds/nadrowski/master.txt --prior "
               f"retrain_prior --chi")
    assert (held.status.reopen, _failure(view), _said_alone(view), view.resume_button.isVisibleTo(view)) == (
        True, f"{_REOPEN} Close PRISM and open it again. The training is kept up to epoch 4; until the Training page "
              f"offers to resume it, it is resumed from the command line with the command below.", command, False), (
        _failure(view))
    args, said = _parsed(_said_alone(view))
    assert (args is not None and (args.cmd, args.resume_fit, args.bounds, args.prior, args.chi_mode)) == (
        "train", kept, "Resources/Bounds/nadrowski/master.txt", "retrain_prior", True), (
        "the command copied off the page is not one the command-line tool takes", said)
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
    narrowing.fail(_device_failure(in_the_simulation=False))
    _ended(app, held)
    assert (_failure(view), _said_alone(view)) == (
        f"{_DEVICE.format(_ILLEGAL.splitlines()[0])} Close PRISM and open it again. The training is kept up to "
        f"epoch 2; until the Narrowing round page offers to resume it, it is resumed from the command line with the "
        f"command below, given the same --bounds, --posterior, --observation and --chi it was trained with.",
        f"python -m core tsnpe --resume-fit {kept}"), _failure(view)
    chosen = SimpleNamespace(sources={"bounds": r"C:\PRISM files\master.txt"}, chi_mode=False,
                             hw=SimpleNamespace(device=SimpleNamespace(type="cpu")))
    narrowed_parent = SimpleNamespace(name="", id="p2", posterior=SimpleNamespace(truncation=object()))
    spec = _training("Narrowing round").with_call(len, (chosen, narrowed_parent, prior, observation), {})
    assert retry.resume_command(spec, "k1") == (
        'python -m core tsnpe --resume-fit k1 --bounds "C:\\PRISM files\\master.txt" --posterior p2 --observation o1 '
        '--no-chi --accept-truncated --device cpu'), "the command names a flag that is not the call's"
    args, said = _parsed(retry.resume_command(spec, "k1"))
    assert args is not None and (args.cmd, args.resume_fit, args.bounds, args.posterior, args.observation,
                                 args.chi_mode, args.accept_truncated, args.device) == (
        "tsnpe", "k1", r"C:\PRISM files\master.txt", "p2", "o1", False, True, "cpu"), (
        "the narrowing round's command is not one the command-line tool takes", said)


def test_a_runs_call_is_kept_only_while_its_page_offers_to_continue_or_resume_it(tmp_path):
    """A run's own call -- the configuration, the prior and the tensors its arguments hold -- is let go at its end,
    but for a training whose page offers to continue or resume it: that call is kept until the page starts its next
    run, or goes. A finished training, a training whose failure only reopening can help, and a run of another kind let
    go of it at their end."""
    from PySide6.QtCore import QCoreApplication, QEvent
    app = qt_app()
    clock = Clock(_at(3, 3, 2, 11))
    held, host = _controller_on(tmp_path, clock), build_run_host()
    text = RunText(held)
    folder = _set(tmp_path, 236)

    class Payload:
        """What a run's arguments hold, for the test to see when it is let go."""

    def ended(spec, end, on=host):
        payload = Payload()
        kept = weakref.ref(payload)
        stand_in = StandIn(word=spec.word)
        start_run(on, stand_in, spec=spec, payload=payload)
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
    gone = build_run_host()
    left = ended(_training(), lambda stand_in: stand_in.fail(OSError(28, "No space left on device")), gone)
    assert left() is not None, "the call of a run its page offers to continue was let go"
    gone.deleteLater()                                  # the page goes, its offer with it
    QCoreApplication.sendPostedEvents(gone, QEvent.DeferredDelete)
    del gone
    gc.collect()
    assert left() is None, "the call of a run whose page went was kept"
    let_go = [ended(_training(), lambda stand_in: stand_in.release()),
              ended(_training(), tries_to_the_end),
              ended(RunSpec("Check", ("inference", "checks")), lambda stand_in: stand_in.fail(OSError("x")))]
    gc.collect()
    assert ([one() is None for one in let_go], offered()) == ([True, True, True], None), (
        "a run's call was kept at its end with nothing to offer, or kept past its page's next run")


def test_closing_the_window_during_a_pause_ends_the_tries_and_then_closes_it(tmp_path, monkeypatch):
    """While a training waits to be tried again, the rail's foot, its step in the rail and Home say that it waits to
    try again, not that it runs. Closing the window meanwhile asks first, in the words of the pause, and a question
    nobody answers keeps the window, the one-PRISM claim and the run's tries. "Stop the run and close" ends the tries:
    the run ends failed, its history says so, and only then is the claim let go and the window closed."""
    from core.artifacts import default_store
    from core.gui import one_instance
    from core.gui.run import history
    from core.gui.widgets.page import Sentence
    from tests._fixtures import asked, whole_program
    from tests._window_helpers import shown_off_screen
    app = qt_app()
    monkeypatch.setenv("PRISM_ARTIFACTS", str(tmp_path / "Artifacts"))
    clock = Clock(_at(3, 3, 2, 11))
    with whole_program() as (_app, window):
        held = _controller_on(tmp_path, clock)
        claim = one_instance.current()
        build_fit_record(default_store())               # a record: Home states its next step, not its first page
        shown_off_screen(window)
        host, text = window.part("inference").posterior_panel, RunText(held)
        foot, home, step = window.shell.rail.foot, window.part("home"), window.shell.rail.row("training")
        stand_in = StandIn()
        start_run(host, stand_in, spec=_training())
        _to_the_simulation(app, stand_in, text, held, _set(tmp_path, 236), 236)
        stand_in.fail(_out_of_memory())
        _waiting(app, held)
        window.go_to("home")
        assert (foot._title, foot._words, step.accessibleName()) == (
            "Training is waiting to try again", "Nothing runs until the next try. Settings are locked.",
            "Step 3, Training. Now: waiting to try again"), (
            "the foot or the rail does not say the run waits to try again", foot._title, foot._words)
        assert ("Training retrain_3 is waiting to try again, and no other run can start until it ends. You can still "
                "move around and read results. Keep PRISM open; you can minimise it.") in [
            label.plain_text() for label in home.findChildren(Sentence) if label.isVisibleTo(home)], (
            "Home does not say the run waits to try again")
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


# ── what a reopened window can offer ──
_ROOT = "C:/records/here"
_SETTINGS = {"model": "NADROWSKI", "chi_mode": True, "bounds": "master.txt", "prior": "20261002T101500",
             "options": {"chi_mode": True}, "rows": []}


def _history(run, day, outcome, *, root=_ROOT, word="Training", folder="eda5c2a1e468", batches=10_000,
             settings=_SETTINGS, kept=None, **more):
    """A run's history as the window writes it, of the records folder ``root``, started on ``day`` of October 2026 at
    noon and ended an hour later, its simulation set ``folder`` of ``batches`` in all; its two counts of the set left
    at numbers no reading should take for the set's own (the set's count at the start and at the end)."""
    return {"id": run, "word": word, "name": f"{word.lower()}_{day}", "records_root": root, "outcome": outcome,
            "started": _at(day, 12, 0).isoformat(), "ended": _at(day, 13, 0).isoformat(),
            "simulation_set": {"folder": folder, "batches": batches, "at_start": 7, "at_end": 9},
            "settings": settings, "fit": None if kept is None else {"kept_id": kept}, **more}


def _kept(ident, *, state="stopped unexpectedly", can_resume=True, why_not=None, simulation="eda5c2a1e468",
          region=None, saved="2026-10-04T10:00:00+00:00", passes=41, name="retrain_3", on_purpose=None, reason=None,
          prior="20261002T101500", config=None, inputs=None, parents=None):
    """A kept training as the store's listing hands it (``fit_state.KeptFit``), saved at ``saved``."""
    from pathlib import Path
    from core.SBI.fit_state import KeptFit
    return KeptFit(id=ident, path=Path("kept") / ident, created="2026-10-03T08:00:00+00:00", posterior_name=name,
                   note="", state=state, passes=passes, best_validation_loss=-3.1, passes_since_best=2, saved_at=saved,
                   parents=parents or {"prior": prior, "simulation": simulation}, inputs=inputs or {}, identity={},
                   region=region, config=config or {"chi_mode": True, "device": "cuda"}, can_resume=can_resume,
                   why_not=why_not, stopped_on_purpose=on_purpose, stopped_at=None, stop_reason=reason)


def _offered(histories=(), kept=(), *, sets=None, root=_ROOT):
    """The offers of ``histories`` and ``kept``, each set holding what ``sets`` says by its folder's name (none: no set);
    and the folders whose count was read."""
    from core.gui import continuing
    read, sets = [], sets or {}
    found = continuing.offers(histories, kept, records_root=root,
                              set_batches=lambda folder: (read.append(folder), sets.get(folder))[1],
                              prior_label=lambda ident: "retrain_prior" if ident == "20261002T101500" else None)
    return found, read


@pytest.mark.parametrize("case, want", [
    ("failed", ("continue", 6150)), ("not seen to end", ("continue", 6150)), ("stopped", ("stopped", 6150)),
    ("finished", None), ("running", None), ("a later training", None), ("a later training of another folder", ("continue", 6150)),
    ("a later prior build", ("continue", 6150)), ("complete with no kept training", ("fit_again", 10_000)),
    ("complete with a kept training on it", ("resume", None)), ("complete, stopped", None), ("set gone", None),
    ("set empty", None), ("another records folder", None), ("no settings", None), ("settings not a dict", None),
    ("settings naming no bounds file", None), ("a narrowing round", None)])
def test_the_offers_follow_the_rule_of_each_end(case, want):
    """The newest training of the records folder in use offers, by how it ended and what its simulation set holds now:
    failed or not seen to end with its set unfinished and its settings recorded, "Continue from batch N" at the set's
    own count, whatever the history's two counts say; its set whole and no kept training to resume on it, "Fit again";
    stopped with its set unfinished, what only the steps say. A later training of that folder leaves it nothing; a later
    run of another kind or another folder does not count. A history with no settings, or settings that name no bounds
    file, a set gone or empty, another records folder and a narrowing round offer nothing to press."""
    outcome = case if case in ("failed", "not seen to end", "stopped", "finished", "running") else "failed"
    sets = {"eda5c2a1e468": 6150}
    histories = [_history("20261003T120000-training", 3, outcome)]
    kept = []
    if case == "a later training":
        histories.append(_history("20261005T120000-training", 5, "finished", folder="0a1b2c3d4e5f"))
    elif case == "a later training of another folder":
        histories.append(_history("20261005T120000-training", 5, "running", root="C:/records/elsewhere"))
    elif case == "a later prior build":
        histories.append(_history("20261005T120000-prior-build", 5, "finished", word="Prior build"))
    elif case.startswith("complete"):
        sets = {"eda5c2a1e468": 10_000}
        if case == "complete, stopped":
            histories = [_history("20261003T120000-training", 3, "stopped")]
        if case == "complete with a kept training on it":
            kept = [_kept("20261003T130500", saved="2026-10-03T19:30:00+00:00")]    # kept during the run, before its end
    elif case == "set gone":
        sets = {}
    elif case == "set empty":
        sets = {"eda5c2a1e468": 0}
    elif case == "another records folder":
        histories = [_history("20261003T120000-training", 3, "failed", root="C:/records/elsewhere")]
    elif case == "no settings":
        histories = [_history("20261003T120000-training", 3, "failed", settings=None)]
    elif case == "settings not a dict":
        histories = [_history("20261003T120000-training", 3, "failed", settings="master.txt")]
    elif case == "settings naming no bounds file":
        histories = [_history("20261003T120000-training", 3, "failed", settings={**_SETTINGS, "bounds": ""})]
    elif case == "a narrowing round":
        histories = [_history("20261003T120000-narrowing-round", 3, "failed", word="Narrowing round")]
    found, _read = _offered(histories, kept, sets=sets)
    pressable = [(offer.kind, offer.batch) for offer in found if not offer.why_not]
    assert (pressable[0] if pressable else None) == want, found


def test_the_offers_read_only_the_newest_trainings_set_and_name_it_by_its_end():
    """Only the set of the newest training of the folder in use is read, and only for an end that can offer something;
    the offer names that run, its batch is the set's own count, and its sentence says how and when it ended and what is
    saved, at the set's count."""
    histories = [_history("20261003T120000-training", 3, "failed", folder="older"),
                 _history("20261004T120000-training", 4, "not seen to end"),
                 _history("20261005T120000-training", 5, "running", root="C:/records/elsewhere", folder="other")]
    found, read = _offered(histories, sets={"eda5c2a1e468": 236, "older": 5})
    assert read == ["eda5c2a1e468"], read
    offer, = found
    assert (offer.kind, offer.run, offer.batch, offer.button, offer.how) == (
        "continue", "20261004T120000-training", 236, "Continue from batch 236", "not seen to end")
    assert offer.sentence == ("The training training_4 not seen to end on 4 Oct 2026, 13:00. The simulations up to "
                              "batch 236 are saved."), offer.sentence
    _found, read = _offered([_history("20261004T120000-training", 4, "finished")], sets={"eda5c2a1e468": 236})
    assert read == [], "a set was read for an end that offers nothing"


def test_a_training_kept_on_purpose_is_offered_as_one_cut_off_is_saying_how():
    """Every kept training is offered, newest first, whether it was stopped unexpectedly, kept when stopped, or stopped
    by an error, which is quoted as its record words it; its sentence says after which epoch and when it was last
    saved, its best loss and how long ago, the set and the prior it trains on, and that a resume reads the set back
    first; one whose fit is finished says resuming writes the posterior alone; and one that cannot be resumed says why
    and offers no button. The run that kept it is named when a history says so."""
    from core.gui import continuing
    kept = [_kept("20261003T130500", state="kept when stopped", on_purpose=True, reason="KeyboardInterrupt",
                  saved="2026-10-04T10:00:00+00:00"),
            _kept("20261003T140500", reason="RuntimeError: the card ran out of memory", on_purpose=False,
                  saved="2026-10-05T10:00:00+00:00", name=""),
            _kept("20261003T150500", state="finished", saved="2026-10-03T10:00:00+00:00", prior="gone"),
            _kept("20261003T160500", can_resume=False, why_not="its simulation set eda5c2a1e468 is not in the store",
                  saved="2026-10-06T10:00:00+00:00")]
    histories = [_history("20261003T120000-training", 3, "stopped", kept="20261003T130500")]
    found, _read = _offered(histories, kept, sets={"eda5c2a1e468": 10_000})
    assert [offer.kept for offer in found] == ["20261003T160500", "20261003T140500", "20261003T130500",
                                               "20261003T150500"], found
    gone, error, purpose, finished = found
    assert (gone.why_not, gone.button, continuing.for_home(found)) == (
        "its simulation set eda5c2a1e468 is not in the store", "", error)
    assert purpose.how == "kept when stopped" and purpose.run == "20261003T120000-training"
    assert purpose.sentence.startswith("It was kept when stopped after epoch 41, last saved "), purpose.sentence
    assert error.how == "stopped unexpectedly after the error 'RuntimeError: the card ran out of memory'"
    assert error.label.startswith("Unnamed unfinished training, ") and error.sentence.endswith(
        ", after the error 'RuntimeError: the card ran out of memory'. Best validation loss -3.10, 2 epochs ago. It "
        "trains on the simulation set eda5c2a1e468 and the prior retrain_prior. Resuming first reads the simulation "
        "set back, about four minutes at full size."), error.sentence
    assert finished.sentence.endswith("It trains on the simulation set eda5c2a1e468 and a prior no longer in Records. "
                                      "Its fit is finished; resuming writes the posterior without training again.")


def test_a_kept_round_is_listed_with_its_command_and_no_resume():
    """A kept narrowing round offers the command line that resumes it, which the tool's own parser takes, naming the
    kept training, the bounds file, the parent posterior, the observation and the kind of observation as its record
    holds them, the processor when it trained there; Home names it, the steps' state does not."""
    from core.gui import continuing
    kept = _kept("20261003T130500", region={"lo": [0.0]}, config={"chi_mode": False, "device": "cpu"},
                 inputs={"bounds": {"path": "Bounds/nadrowski/master weak.txt", "sha256": "0" * 64}},
                 parents={"prior": "p", "simulation": "s", "parent_posterior": "20261002T120000",
                          "observation": "20261002T130000"})
    found, _read = _offered((), [kept])
    offer, = found
    assert (offer.kind, offer.button, continuing.for_home(found), continuing.for_the_steps(found)) == (
        "round", "Open Training", offer, None)
    assert offer.command == ('python -m core tsnpe --resume-fit 20261003T130500 --bounds "Resources/Bounds/nadrowski/'
                             'master weak.txt" --posterior 20261002T120000 --observation 20261002T130000 --no-chi '
                             '--device cpu'), offer.command
    args, refused = _parsed(offer.command)
    assert refused == "" and (args.resume_fit, args.posterior, args.observation) == (
        "20261003T130500", "20261002T120000", "20261002T130000"), refused
    assert offer.sentence.endswith("It is resumed from the command line until its page offers it: ")


def test_of_two_trainings_started_in_one_second_the_later_is_the_newest():
    """Two trainings of one records folder started in one second: the second's id carries its count ("-2"), and it is
    the newest, whatever order their files are listed in, so a stop of the second is said and the first's finished end
    offers nothing."""
    first = _history("20261003T120000-training", 3, "finished")
    second = {**_history("20261003T120000-training-2", 3, "stopped"), "started": first["started"]}
    for histories in ([first, second], [second, first]):
        found, _read = _offered(histories, sets={"eda5c2a1e468": 300})
        assert [(offer.kind, offer.run) for offer in found] == [("stopped", "20261003T120000-training-2")], found


def test_home_and_the_steps_take_the_newest_offer_each_can_say():
    """Home names the newest offer that can be taken, passing over a training stopped with its set unfinished; the
    steps' state says the newest kept training to resume, training to continue, or stopped one, never a set to fit
    again or a kept round."""
    from core.gui import continuing
    stopped = _history("20261005T120000-training", 5, "stopped")
    found, _read = _offered([stopped], [_kept("20261003T130500", saved="2026-10-04T10:00:00+00:00")],
                            sets={"eda5c2a1e468": 300})
    assert [offer.kind for offer in found] == ["stopped", "resume"]
    assert (continuing.for_home(found).kind, continuing.for_the_steps(found).kind) == ("resume", "stopped")
    again, _read = _offered([_history("20261005T120000-training", 5, "failed")], sets={"eda5c2a1e468": 10_000})
    assert (continuing.for_home(again).kind, continuing.for_the_steps(again)) == ("fit_again", None)


class _Unsaid(RuntimeError):
    """An error whose message cannot be had: asked for its words, it raises."""

    def __str__(self):
        raise ValueError("this error's message cannot be had")


# The last test of the file: a run whose end is held up stays live, and no run after it in this process could start.
def test_an_error_whose_message_cannot_be_had_still_ends_its_run(tmp_path):
    """A training that fails with an error whose message cannot be had -- which the engine's rule of the graphics
    card's memory cannot read either -- still ends, failed, and its page says that the error gave no message."""
    app = qt_app()
    clock = Clock(_at(3, 3, 2, 11))
    held, host = _controller_on(tmp_path, clock), build_run_host()
    text, view = RunText(held), host.run_view
    stand_in = StandIn()
    start_run(host, stand_in, spec=_training())
    _to_the_simulation(app, stand_in, text, held, _set(tmp_path, 236), 236)
    stand_in.fail(_Unsaid())
    _ended(app, held)
    assert (held.status.outcome, view.failure.isVisibleTo(view), view.failure_message.text()) == (
        "failed", True, retry.NO_MESSAGE), "the run's end was held up, or its page does not say its failure"
