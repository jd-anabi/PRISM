"""THE root-logger handler, installed by both front ends at start-up.

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

WHAT THIS DOES. ``install(sink)`` puts ONE handler on the root logger. It exists from start-up,
which is why ``basicConfig`` never fires. It hands ``sink`` a record ONLY IF NO LOGGER BETWEEN THE
RECORD'S OWN AND THE ROOT (the root excluded) HAS A HANDLER THAT ALREADY EMITTED IT -- the walk goes
up ``parent`` and stops where ``propagate`` is False, exactly as ``Logger.callHandlers`` does. That
one rule covers every case, where a rule by logger NAME lost records:
  * a ``core`` record DURING A RUN (the window's pump handler, the tool's console handlers, the
    artifact's log.txt -- all on the ``core`` logger) is already out, so the sink stays silent and
    no second copy can appear;
  * a ``core`` record that no handler below the root would SHOW -- the window between runs, a run
    whose console redirect was declined (its ``RunLog`` handler only fills a buffer), a record from
    a thread other than the run's owner (that buffer is per thread and drops it) -- reaches the sink
    instead of vanishing; ``render`` gives it the tool's own shape (bare for information,
    ``warning: `` and so on above), never ``library:``;
  * a library that installed a handler of its OWN (pytensor does, at import, when no handler is
    found above its logger -- which is before the window is built) has already printed its record,
    so the sink does not print it a second time;
  * a handler that EMITS NOWHERE does not count. A ``logging.NullHandler`` is the standard case:
    Python's logging HOWTO tells a library to add one so it stays quiet UNTIL the application
    configures logging; this root handler is that configuration, so such a library's warning
    (pint's, pint/util.py) is shown once. ``core.runs._RunLogHandler`` is the other: it appends to
    an artifact's ``log.txt`` buffer and shows nobody anything, so it carries ``emits_nowhere`` and
    is skipped by the same rule. Without that, EVERY ``core`` record during any run was suppressed
    here because a run log was attached -- which is right when the pane or the console also has a
    handler (they do the showing) and wrong when neither does, the two cases named above.
The handler level test is ``Logger.callHandlers``'s own (``record.levelno >= handler.level``). A
handler's own filters are not consulted: calling another handler's filter a second time could have
side effects, and none of the handlers in play filters.

The decision is a ``logging.Filter`` ON THE HANDLER, never a change to ``core.propagate``:
``caplog`` reads off root and tests/test_refusals.py pins that propagation, and ``RunLog`` and both
front-end handlers sit on the ``core`` logger itself, so a filter leaves all three untouched.

``sink`` is handed the LogRecord, not a formatted line, so each front end picks its stream AT EMIT
TIME -- the rule core/tool/logging_console.py already states, and the one ``basicConfig``'s handler
broke. ``render(record)`` is the one text both front ends write: a ``logging.Formatter``, so a
``log.exception`` traceback and a ``stack_info`` are included, with the prefix on the FIRST line.

Standard library only (so ``python -m core --help`` stays torch-free), and no logger's level is
touched anywhere in this module: core/runs.py owns the ``core`` logger's level, once, at import.
"""
import logging

_FORMATTER = logging.Formatter("%(message)s")


def is_prism(record: logging.LogRecord) -> bool:
    """True for a record from the ``core`` logger or any of its children: PRISM's own voice."""
    return record.name == "core" or record.name.startswith("core.")


def render(record: logging.LogRecord) -> str:
    """The text for one record, traceback and stack included, with no trailing newline.

    A library record reads ``library: <who>: <message>``, named by its logger EXCEPT for ``root``:
    the trigger this handler exists for is ``logging.warning``, the module-level function, whose
    logger is the root logger, so ``record.name`` would read ``root`` and name nothing. For that one
    name the prefix falls back to ``record.module``, the basename of the file that logged -- sbi's
    leakage warning reads ``library: rejection:`` (sbi/samplers/rejection/rejection.py), and that is
    the line the window's log pane shows for it.

    A ``core`` record reaches a sink only when no ``core`` handler was attached, and it is PRISM's own
    voice, so it takes the tool's own console shape (core/tool/logging_console.py): the bare message
    below WARNING, ``<level name lower-cased>: `` at WARNING and above -- never ``library:``.
    """
    text = _FORMATTER.format(record)
    if is_prism(record):
        return text if record.levelno < logging.WARNING else f"{record.levelname.lower()}: {text}"
    who = record.name if record.name != "root" else record.module
    return f"library: {who}: {text}"


def _emits_nowhere(handler: logging.Handler) -> bool:
    """True for a handler that shows the record to nobody, so having "had" it is not having emitted
    it: a ``logging.NullHandler``, and any handler that says so with ``emits_nowhere`` (PRISM's run
    log, which only fills an artifact's log.txt buffer -- and does so per THREAD, so it can drop the
    record outright)."""
    return isinstance(handler, logging.NullHandler) or getattr(handler, "emits_nowhere", False)


def _already_emitted(record: logging.LogRecord) -> bool:
    """True if a logger between the record's own and the root (the root excluded) has a handler that
    ``Logger.callHandlers`` gave it to -- one that emits nowhere excepted (``_emits_nowhere``)."""
    root = logging.getLogger()
    logger = logging.getLogger(record.name)          # the root itself for ``logging.warning``
    while logger is not None and logger is not root:
        for handler in logger.handlers:
            if not _emits_nowhere(handler) and record.levelno >= handler.level:
                return True
        if not logger.propagate:
            break
        logger = logger.parent
    return False


class _NotYetEmitted(logging.Filter):
    """False for a record some handler below the root already emitted (the module docstring's rule)."""

    def filter(self, record: logging.LogRecord) -> bool:
        return not _already_emitted(record)


class _SinkHandler(logging.Handler):
    """Hands every record that survives ``_NotYetEmitted`` to ``sink(record)``.

    No level of its own (NOTSET): whether a library emits a record at all is that library's logger's
    business, and the root logger's own WARNING level already gates ``logging.warning``.
    """

    def __init__(self, sink):
        super().__init__()
        self._sink = sink
        self.addFilter(_NotYetEmitted())

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
