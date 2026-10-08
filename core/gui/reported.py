"""Drawing code that reports its own errors.

The main window keeps the interpreter while it handles an event, a repaint among them: its ``event`` is
written in Python for that, so that the parts of the look written in Python are entered without waiting
for the interpreter once per call while a run's thread is working. One thing goes wrong with it. An error
raised by Python code the toolkit calls from inside that handling -- a ``paintEvent``, an override of the
style, an event filter hearing a paint, a ``resizeEvent`` -- is then dropped by the binding without a
word, where it used to reach ``sys.excepthook`` and the window's "Unhandled error" box. So every method
the toolkit calls while a window is drawn or laid out carries ``reported``: it hands its own error to the
hook, and answers with a fallback so that the drawing goes on.

Nothing of the toolkit or the engine is imported here.
"""
import contextlib
import functools
import sys

# One (name, error) for every error a guarded method raised since the record was last emptied; the name
# is the method's qualified name, or "sys.excepthook" for an error the hook itself raised.
ERRORS: list = []
_SHOWN: set = set()          # the names handed to the hook since then
_EXPECTED: list = []         # one list for every open ``expected`` block, the innermost last


def forget() -> None:
    """Empty the record and the set of names already shown."""
    ERRORS.clear()
    _SHOWN.clear()


def _record(name, error) -> None:
    ERRORS.append((name, error))
    for caught in _EXPECTED:
        caught.append((name, error))


def reported(fallback=None):
    """Guard a method the toolkit calls while a window is drawn or laid out.

    The method runs as before. If it raises, the error is recorded in ``ERRORS``; the first time for that
    method since ``forget()`` it is handed to ``sys.excepthook``, looked up at that moment, so that the
    window's own hook is the one called, and later times not: a paint that fails at every repaint shows
    one box, not one a frame. The answer is then ``fallback(self, *args, **kwargs)`` when ``fallback`` can
    be called, and ``fallback`` itself when it cannot. A keyboard interrupt and an exit pass through.
    Nothing else comes out: an error the hook raises is recorded under "sys.excepthook", and one the
    fallback raises under the method's name, the answer then being None.
    """
    def guard(method):
        name = method.__qualname__

        @functools.wraps(method)
        def guarded(self, *args, **kwargs):
            try:
                return method(self, *args, **kwargs)
            except Exception as error:       # noqa: BLE001 -- whatever it is, the binding would drop it
                _record(name, error)
                if name not in _SHOWN and not _EXPECTED:
                    _SHOWN.add(name)         # before the hook: its box runs a loop, which may paint again
                    try:
                        sys.excepthook(type(error), error, error.__traceback__)
                    except Exception as second:      # noqa: BLE001
                        _record("sys.excepthook", second)
            if not callable(fallback):
                return fallback
            try:
                return fallback(self, *args, **kwargs)
            except Exception as error:       # noqa: BLE001
                _record(name, error)
                return None
        return guarded
    return guard


@contextlib.contextmanager
def expected():
    """For a test that makes a guarded method raise on purpose. Inside the block an error is recorded as
    always and handed to no hook; on the way out the record and the names already shown are as they
    were on the way in. Yields the list of the (name, error) caught inside the block."""
    errors, shown, caught = list(ERRORS), set(_SHOWN), []
    _EXPECTED.append(caught)
    try:
        yield caught
    finally:
        _EXPECTED.pop()
        ERRORS[:] = errors
        _SHOWN.clear()
        _SHOWN.update(shown)
