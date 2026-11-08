"""A chain of loads: links, each a step's own load of one saved record, started one after the other, each only once
the run of the one before has ended having returned its record. What an earlier link loaded stays when a later one
does not start or does not finish: the chain stops there, and says why.

A link is a callable that starts its load and answers whether its run was started, as a page's ``dispatch`` answers.
The next link starts a turn of the event loop after the run's end has been said, never inside it: the end of a run has
lines of its own still to run once it has said so, its page's among them, and a box that tells a refusal or a failure
opens after it. A chain hears the process's run controller only while it goes, and says how it ended a turn after it
ended, so that a caller that connects to a chain just started hears even a first link that started nothing.
"""
from collections import deque
from collections.abc import Callable, Sequence

from PySide6.QtCore import QObject, QTimer, Signal

from .run.controller import controller

#: Why a chain stopped when a link started no run: a run was live, what the link's step read blocks the load, its
#: record is no longer listed, or a question the link asked was answered "Cancel". The step's own page says which.
NOT_STARTED = "not started"
#: Why a chain stopped when a link's run was refused before it began, which gives no outcome of its own.
REFUSED = "refused"
#: Why a chain stopped when a link raised as it started: the error goes where every error goes.
RAISED = "raised"
#: Why a chain stopped when its window closed while it went, or Windows ended the session.
CLOSED = "closed"


class LoadChain(QObject):
    """A chain of loads. ``start(links)`` starts the first link at once; ``stop()`` ends it where it stands.
    ``finished`` says every link's run returned its record; ``stopped(why)`` says one did not: ``NOT_STARTED``,
    ``RAISED``, how its run ended ("failed", "stopped", or ``REFUSED``), or ``CLOSED``. ``why`` holds the same once
    the chain has ended ("" when it finished), and ``is_going`` whether it goes. A chain goes once: a second ``start``
    raises."""

    finished = Signal()
    stopped = Signal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._links: deque = deque()
        self._going = False
        self._started = False
        self._said = False
        self.why: "str | None" = None
        # What the link in hand puts in use, for a caller that says where a chain stopped: (the step's key, the words
        # that name it, "applied" or "loaded"); None for a caller that says nothing.
        self.at: "tuple[str, str, str] | None" = None

    def is_going(self) -> bool:
        """Whether a link's run is live or the next link waits for its turn."""
        return self._going

    def start(self, links: Sequence[Callable[[], bool]]) -> None:
        """Start the chain of ``links``, the first one now."""
        if self._started:
            raise RuntimeError("a chain of loads goes once")
        self._started, self._going, self._links = True, True, deque(links)
        controller().run_ended.connect(self._ended)
        self._next()

    def stop(self, why: str = CLOSED) -> None:
        """End the chain where it stands, while it goes: no later link starts, and a link's run that is live goes on to
        its own end. Its window closing, or Windows ending the session, ends it so. A chain that has ended in this turn,
        its end not yet said, is stopped too: it says the stop, and ``finished`` never comes."""
        if self._going:
            self._end(why)
        elif self._started and not self._said:
            self.why = why

    def _next(self) -> None:
        """The next link, or the end when none is left."""
        if not self._going:
            return
        if not self._links:
            self._end("")
            return
        link = self._links.popleft()
        try:
            started = link()
        except BaseException:
            self._end(RAISED)
            raise
        if not started:
            self._end(NOT_STARTED)

    def _ended(self, status) -> None:
        """A run has ended: the link's own, for one runs at a time. Finished, the next link a turn later; else the
        chain stops with how it ended."""
        if not self._going:
            return
        outcome = getattr(status, "outcome", None)
        if outcome == "finished":
            QTimer.singleShot(0, self, self._next)
        else:
            self._end(outcome or REFUSED)

    def _end(self, why: str) -> None:
        """The chain ends, once: it hears the controller no more, and says how a turn later. A chain that has ended
        already is left as it ended."""
        if not self._going:
            return
        self._going, self._links, self.why = False, deque(), why
        controller().run_ended.disconnect(self._ended)
        QTimer.singleShot(0, self, self._say_the_end)

    def _say_the_end(self) -> None:
        self._said = True
        if self.why:
            self.stopped.emit(self.why)
        else:
            self.finished.emit()
