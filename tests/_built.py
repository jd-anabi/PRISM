"""The record of what the builders of tests/_fixtures.py made, so that each window, part, inference screen and
bare run host is let go when the test that built it ends.

At module level this file imports the standard library and nothing else -- the toolkit inside its functions
only, torch and the program nowhere -- as tests/_dialogs.py does. ``BUILT`` holds a weak reference to each thing
a builder made, in the order they were made, so the record itself keeps nothing alive. tests/conftest.py's
``_built_widgets_let_go`` takes ``noted()`` when a test is set up, the record's length among it, and at the
test's teardown hands it to ``end_of_test``, which deletes what was made since (``let_go``). What a fixture of
wider scope built (the screen of a module's run, a panel a module builds once) was made before that length was
noted, so it is left alone and outlives the test, as it is meant to. tests/_fixtures.py reaches the record
through this module and binds no container of its own, so that its builders' own test can still say that
_fixtures keeps nothing it built. What tests/_runs.py's ``RunText`` hears of the run's controller is noted here
too, so that it hears no run after its test.

The end of every test also puts the process's run controller back as the next test must find it: a run left live
is stopped and waited for, and a run that ended is acknowledged (``end_of_test``, ``put_back_the_run``).
"""
import sys
import time
import weakref

BUILT: list = []             # one weak reference for each thing a builder made, the oldest first

# How long the end of a test waits, at most, for a run the test left live to stop once it is asked to. A run
# stops at its next line of output: the slowest the suites start, a simulated inference on the tiny posterior,
# stopped within 1.1 s of the ask wherever in it the ask came, and the suites' own tests of a cancel wait 10 s.
STOPS_WITHIN = 10.0


def keep(widget):
    """Note ``widget`` among what the builders made, and hand it back."""
    BUILT.append(weakref.ref(widget))
    return widget


def let_go(since: int) -> None:
    """Delete everything a builder made after the first ``since`` entries of the record, the newest first, each by
    its own deferred delete, and take those entries out. One that is already gone (dropped by its test, deleted
    by its parent or by the test itself) is passed over. Never closed: closing a window writes settings."""
    import gc
    import shiboken6
    from PySide6.QtCore import QCoreApplication, QEvent
    made = BUILT[since:]
    del BUILT[since:]
    for ref in reversed(made):
        widget = ref()
        if widget is not None and shiboken6.isValid(widget):
            widget.deleteLater()
            QCoreApplication.sendPostedEvents(widget, QEvent.DeferredDelete)
        del widget
    if made:
        gc.collect()             # the wrappers of what went, and the cycles that held it, go now too


def _controller():
    """The process's run controller, reached where its module is kept; None when that module was never loaded."""
    runs = sys.modules.get("core.gui.run.controller")
    return None if runs is None else runs.controller()


def a_run_is_live() -> bool:
    """Whether a run is live in this process, as its run controller says. False when the controller's module was
    never loaded."""
    held = _controller()
    return held is not None and held.is_live()


def put_back_the_run() -> None:
    """The process's run controller as the next test must find it: a run that ended acknowledged, so that the
    controller is idle, and the clock and the folder of runs a test handed it let go. Nothing while a run is
    live, nor when the controller's module was never loaded."""
    held = _controller()
    if held is None or held.is_live():
        return
    if held.status.unacknowledged_end and held.status.run is not None:
        held.acknowledge(held.status.run)
    held.use_clock(None)
    held.use_folder(None)


def noted() -> tuple:
    """What the end of the test now set up is held to: how many entries the record holds, and whether a run is
    live already (one an earlier test left live, which failed for it)."""
    return len(BUILT), a_run_is_live()


def end_of_test(at_start: tuple, within: float = STOPS_WITHIN) -> None:
    """Let go what the builders made since ``at_start`` (what ``noted`` gave when the test was set up), put the run
    controller back as the next test must find it (``put_back_the_run``), and fail the test, by raising, when it
    left a run live.

    Such a run is asked to stop first, through the controller (a training saving its posterior, which takes no stop
    asked so, through its own token), and the event loop is turned until its end has been handled, ``within``
    seconds at most: a run whose end came after its panel was deleted would end without the page it was started
    from. If it has not stopped by then, what the test made is taken out of the record and kept alive, not deleted,
    so that the run's end still finds its panel, and every later run is refused until it stops.
    A run that was live already when the test was set up is an earlier test's, and is not waited for again."""
    since, live_before = at_start
    left_live = a_run_is_live() and not live_before
    waited = 0.0
    if left_live:
        from PySide6.QtCore import QCoreApplication
        held = _controller()
        held.request_stop()
        token = getattr(getattr(held, "_run", None), "token", None)
        if token is not None and not token.requested.is_set():
            token.requested.set()                # a training saving its posterior: no stop the controller is asked for
        began = time.monotonic()
        while a_run_is_live() and time.monotonic() - began < within:
            QCoreApplication.processEvents()
            time.sleep(0.01)
        waited = time.monotonic() - began
    stopped = not (left_live and a_run_is_live())
    put_back_the_run()
    if stopped:
        let_go(since)
    else:
        del BUILT[since:]
    if left_live:
        raise AssertionError(
            "the test ended with a run still live; it was asked to stop, and "
            + (f"its end was handled {waited:.1f} s later" if stopped else
               f"it had not stopped {within:g} s later, so what the test built is kept for its end to find, "
               f"and every later run is refused until it stops"))
