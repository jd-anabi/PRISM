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
_fixtures keeps nothing it built.
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


def a_run_is_live() -> bool:
    """Whether a panel's run is live in this process: the flag every run sets while it goes, read where the
    panels' module keeps it. False when that module was never loaded."""
    panels = sys.modules.get("core.gui.panels.base_panel")
    return panels is not None and panels.BasePanel._running


def noted() -> tuple:
    """What the end of the test now set up is held to: how many entries the record holds, and whether a run is
    live already (one an earlier test left live, which failed for it)."""
    return len(BUILT), a_run_is_live()


def end_of_test(at_start: tuple, within: float = STOPS_WITHIN) -> None:
    """Let go what the builders made since ``at_start`` (what ``noted`` gave when the test was set up), and fail
    the test, by raising, when it left a run live.

    Such a run is asked to stop first, and the event loop is turned until its end has been handled, ``within``
    seconds at most: the run's end, come after its panel was deleted, would raise on the panel before it put back
    the flag that says a run is live, and every later run would be refused. If it has not stopped by then, what
    the test made is taken out of the record and kept alive, not deleted, so that the run's end still finds its
    panel. A deleted panel then leaves the set every panel enrols in, which the next run walks to lock every
    panel's controls: a panel that ran something is still held in Python once deleted, by its run's signals.
    A run that was live already when the test was set up is an earlier test's, and is not waited for again."""
    since, live_before = at_start
    left_live = a_run_is_live() and not live_before
    waited = 0.0
    if left_live:
        from PySide6.QtCore import QCoreApplication
        sys.modules["core.gui.panels.base_panel"].BasePanel.request_cancel_all()
        began = time.monotonic()
        while a_run_is_live() and time.monotonic() - began < within:
            QCoreApplication.processEvents()
            time.sleep(0.01)
        waited = time.monotonic() - began
    stopped = not (left_live and a_run_is_live())
    try:
        if stopped:
            let_go(since)
        else:
            del BUILT[since:]
    finally:
        panels = sys.modules.get("core.gui.panels.base_panel")
        if panels is not None:
            import shiboken6
            for panel in list(panels.BasePanel._instances):
                if not shiboken6.isValid(panel):
                    panels.BasePanel._instances.discard(panel)
    if left_live:
        raise AssertionError(
            "the test ended with a run still live; it was asked to stop, and "
            + (f"its end was handled {waited:.1f} s later" if stopped else
               f"it had not stopped {within:g} s later, so what the test built is kept for its end to find, "
               f"and every later run is refused until it stops"))
