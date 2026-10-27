"""A run's failure, as a state of the run's page and never a box: what failed and when, what was saved and what was
lost, the error's message, and its kind and traceback when asked for; and the health line of a training whose save
failed during its fit.

A live run here is the stand-in run of tests/_runs.py, started through a real panel's own dispatch and failing with
the error a test hands it; a training's simulation set is a folder of the test's own whose state, written as the
engine writes it, holds the batches the test says; the window's clock is moved by hand.
"""
from datetime import datetime, timedelta, timezone

from PySide6.QtWidgets import QLabel

from core.gui.run import retry
from core.gui.run.controller import controller
from core.gui.run.model import SIMULATION_ROW, RunSpec
from tests._fixtures import all_asked, build_run_host, qt_app
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
