"""The record of what the builders of tests/_fixtures.py made, so that each window, part, inference screen and
bare run host is let go when the test that built it ends.

At module level this file imports the standard library and nothing else -- the toolkit inside its functions
only, torch and the program nowhere -- as tests/_dialogs.py does. ``BUILT`` holds a weak reference to each thing
a builder made, in the order they were made, so the record itself keeps nothing alive. tests/conftest.py's
``_built_widgets_let_go`` notes the record's length when a test is set up, and at the test's teardown hands
that length to ``let_go``, which deletes what was made since. What a fixture of wider scope built (the screen of
a module's run, a panel a module builds once) was made before that length was noted, so it is left alone and
outlives the test, as it is meant to. tests/_fixtures.py reaches the record through this module and binds no
container of its own, so that its builders' own test can still say that _fixtures keeps nothing it built.
"""
import weakref

BUILT: list = []             # one weak reference for each thing a builder made, the oldest first


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
