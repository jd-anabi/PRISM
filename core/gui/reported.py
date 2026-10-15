"""Drawing code that reports its own errors.

The main window keeps the interpreter while it handles an event, a repaint among them: its ``event`` is
written in Python for that, so that the parts of the look written in Python are entered without waiting
for the interpreter once per call while a run's thread is working. One thing goes wrong with it. When
Python code the toolkit calls from inside that handling raises, and the toolkit enters another method
written in Python before the handling returns, the binding drops the error without a word, where it used
to reach ``sys.excepthook`` and the window's "Unhandled error" box. An error of the last such method the
handling enters is still reported, under a longer first line. In practice the loss is what happens to
whatever is raised while the window is painted, and to a page's ``showEvent`` or ``hideEvent`` when the
window comes back from minimised or goes there. So the methods this program writes that the toolkit calls
while a window is drawn, laid out, shown or hidden carry ``reported``: a ``paintEvent``, an override of the
style, the event filter that hears every paint, a ``resizeEvent``, a page's ``showEvent`` and ``hideEvent``,
and likewise a delegate's or an icon engine's ``paint`` and what a model answers in every repaint (``data``,
``headerData``, ``flags``, ``rowCount``, ``columnCount``) when the program has one. Each hands its own error
to the hook, and answers with a fallback so that the drawing goes on.

Not guarded is a drawing library's own code inside the main window, which this program cannot decorate:
today what the live simulation's plotting widget (core/gui/widgets/live_hair_bundle.py) inherits from its
library. An error raised there during a repaint is dropped in the same way, and reported only when no
other method written in Python is entered after it. What this program itself writes in that class
carries the guard like any other.

What is kept of an error is words and a count, never the error. An error keeps its traceback, the
traceback the frames of the failing call, and a frame its locals: in a paint event that is the live
painter, which the toolkit would then find still active at every later paint of that widget. And the
window runs for days, so a method that fails at every repaint must not add an entry at every repaint.

Nothing of the toolkit or the engine is imported here.
"""
import contextlib
import functools
import sys

# One (name, words) for the FIRST error raised under each name since the record was last emptied. The name
# is a guarded method's module and qualified name ("core.gui.style.FocusRing.paintEvent": two modules may
# each hold a class of one name), that name with " (its fallback)" for an error its fallback raised, or
# "sys.excepthook" for one the hook raised; the words are the error's type and its text.
ERRORS: list = []
COUNTS: dict = {}            # name: how many errors were raised under it since then, the listed one included
_EXPECTED: list = []         # one list for every open ``expected`` block, the innermost last


def forget() -> None:
    """Empty the record and the counts: every method's next failure is its first again."""
    ERRORS.clear()
    COUNTS.clear()


def _failed(name, error) -> bool:
    """Count one error under ``name``. The first since the record was emptied is listed as words, and the
    answer is then True; a later one is counted and nothing more."""
    COUNTS[name] = COUNTS.get(name, 0) + 1
    if COUNTS[name] > 1:
        return False
    try:
        words = f"{type(error).__name__}: {error}"
    except Exception:                        # noqa: BLE001 -- an error whose own text fails
        words = type(error).__name__
    ERRORS.append((name, words))
    for caught in _EXPECTED:
        caught.append((name, words))
    return True


def reported(fallback=None):
    """Guard a method the toolkit calls while a window is drawn or laid out.

    The method runs as before. If it raises, the failure is counted in ``COUNTS``; the first one of that
    method since ``forget()`` is listed in ``ERRORS`` and handed to ``sys.excepthook``, looked up at that
    moment, so that the window's own hook is the one called, and later ones are not: a paint that fails at
    every repaint shows one box, not one a frame. The answer is then ``fallback(self, *args, **kwargs)``
    when ``fallback`` can be called, and ``fallback`` itself when it cannot. A keyboard interrupt and an
    exit pass through. Nothing else comes out: an error the hook raises is counted under "sys.excepthook",
    and one the fallback raises under the method's name with " (its fallback)", the answer then being None.

    When the call returns nothing here holds the error, its traceback or a frame of the failing call.
    """
    def guard(method):
        name = f"{method.__module__}.{method.__qualname__}"

        @functools.wraps(method)
        def guarded(self, *args, **kwargs):
            try:
                return method(self, *args, **kwargs)
            except Exception as error:       # noqa: BLE001 -- whatever it is, the binding would drop it
                # counted before the hook is called: its box runs a loop, which may paint again
                if _failed(name, error) and not _EXPECTED:
                    try:
                        sys.excepthook(type(error), error, error.__traceback__)
                    except Exception as second:      # noqa: BLE001
                        _failed("sys.excepthook", second)
            # Out of the handler: the error is let go, and with it the failing call's frame and its locals,
            # before the fallback draws.
            if not callable(fallback):
                return fallback
            try:
                return fallback(self, *args, **kwargs)
            except Exception as error:       # noqa: BLE001
                _failed(f"{name} (its fallback)", error)
                return None
        return guarded
    return guard


@contextlib.contextmanager
def expected():
    """For a test that makes a guarded method raise on purpose. Inside the block a failure is counted and
    listed as always and handed to no hook; on the way out the record and the counts are as they were on
    the way in. Yields the list of the (name, words) listed inside the block: the first failure under each
    name that had none before it. A repeat is not in that list; it is in ``COUNTS`` while the block is open."""
    errors, counts, caught = list(ERRORS), dict(COUNTS), []
    _EXPECTED.append(caught)
    try:
        yield caught
    finally:
        _EXPECTED.pop()
        ERRORS[:] = errors
        COUNTS.clear()
        COUNTS.update(counts)
