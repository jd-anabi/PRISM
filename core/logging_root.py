"""THE root-logger handler, installed by both front ends at start-up (piece 4, B14, spec §7.1).

WHAT WAS WRONG. ``logging.warning(...)`` -- the module-level function, on the ROOT logger -- calls
``basicConfig()`` whenever the root logger has no handlers (logging/__init__.py, ``warning``), and
``basicConfig`` installs a real ``StreamHandler`` bound to ``sys.stderr`` AT CONSTRUCTION. The
``core`` logger propagates (core/runs.py; pinned, because ``caplog`` reads records off the root), so
from that moment every ``core`` record is emitted TWICE -- once by the front end's own handler, once
by the root's. Under the window it is worse than doubling: the stream that handler captured is the
RUN's ``_SignalStream``, so the duplicate arrives in the pane at the error stream's ``warning``
level, and once the run ends the handler keeps writing into a STOPPED pump, where lines are appended
and never published. After one trigger, library records go MISSING rather than doubling.

The trigger is not hypothetical. sbi's ``accept_reject_sample`` calls ``logging.warning`` at
sbi/samplers/rejection/rejection.py:336 and :359 when fewer than ``warn_acceptance=0.01`` of its
proposals are accepted, and that function is on the path of EVERY posterior draw PRISM makes.

WHAT THIS DOES. ``install(sink)`` puts ONE handler on the root logger:
  * it exists from start-up, which is why ``basicConfig`` never fires and no second copy of a
    ``core`` record can ever appear;
  * a record from the ``core`` tree is DROPPED here -- it already has the front end's own handler and
    the artifact's log.txt, both of which read off the ``core`` logger -- and every other logger's
    record is handed to ``sink`` exactly once.

The drop is a ``logging.Filter`` ON THE HANDLER, never a change to ``core.propagate``: ``caplog``
reads off root and tests/test_refusals.py pins that propagation, and ``RunLog`` and both front-end
handlers sit on the ``core`` logger itself, so a filter leaves all three untouched.

``sink`` is handed the LogRecord, not a formatted line, so each front end splits by level and
resolves its own streams AT EMIT TIME -- the rule core/tool/logging_console.py already states, and
the one ``basicConfig``'s handler broke.

Standard library only (so ``python -m core --help`` stays torch-free), and no logger's level is
touched anywhere in this module: core/runs.py owns the ``core`` logger's level, once, at import.
"""
import logging


class _DropCore(logging.Filter):
    """False for a record from the ``core`` logger or any of its children: it has its own handlers."""

    def filter(self, record: logging.LogRecord) -> bool:
        return not (record.name == "core" or record.name.startswith("core."))


class _SinkHandler(logging.Handler):
    """Hands every record that survives ``_DropCore`` to ``sink(record)``.

    No level of its own (NOTSET): whether a library emits a record at all is that library's logger's
    business, and the root logger's own WARNING level already gates ``logging.warning``.
    """

    def __init__(self, sink):
        super().__init__()
        self._sink = sink
        self.addFilter(_DropCore())

    def emit(self, record: logging.LogRecord) -> None:
        # ``except Exception``, deliberately not BaseException: under the window a library record is
        # written through ``_SignalStream.write``, which is a cancel checkpoint like every other
        # write, and the WorkerCancelled it raises must sail through to Worker.run.
        try:
            self._sink(record)
        except Exception:                     # noqa: BLE001 -- logging's contract: emit never raises
            self.handleError(record)


_installed: "_SinkHandler | None" = None


def install(sink) -> None:
    """Install THE root handler for this process. A second install replaces the first."""
    global _installed
    remove()
    _installed = _SinkHandler(sink)
    logging.getLogger().addHandler(_installed)


def remove() -> None:
    """Remove it if one is installed. Idempotent, so a ``finally`` can always call it."""
    global _installed
    if _installed is not None:
        logging.getLogger().removeHandler(_installed)
        _installed = None


def installed() -> bool:
    """True between ``install`` and ``remove``."""
    return _installed is not None
