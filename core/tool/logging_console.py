"""The tool's two logging handlers: information to stdout, warnings and errors to stderr with the
level as a prefix (piece 3, V4, spec §4.3). Torch-free; imported by ``core.tool.__init__`` only.

Both resolve ``sys.stdout``/``sys.stderr`` AT EMIT TIME, never at construction: capsys swaps them
per test and the window swaps them per run, and a handler holding the stream it was built with would
write into a buffer that is no longer anybody's. Neither touches the logger's level: core/runs.py
set it once at import.
"""
import logging
import sys
from contextlib import contextmanager

from core import runs  # noqa: F401 -- sets the ``core`` logger to INFO at import (spec §1.2)


class _LiveStreamHandler(logging.Handler):
    """``%(message)s`` to ``getattr(sys, sys_attr)``, for records with ``min_level <= levelno <=
    max_level``; with ``prefix`` the line is ``<levelname lower-cased>: <message>``."""

    def __init__(self, *, sys_attr: str, min_level: int = logging.NOTSET,
                 max_level: int = logging.CRITICAL, prefix: bool = False):
        super().__init__(level=min_level)
        self._sys_attr, self._max_level, self._prefix = sys_attr, max_level, prefix
        self.setFormatter(logging.Formatter("%(message)s"))

    def emit(self, record):
        if record.levelno > self._max_level:
            return
        try:
            line = self.format(record)
            if self._prefix:
                line = f"{record.levelname.lower()}: {line}"
            stream = getattr(sys, self._sys_attr)
            stream.write(line + "\n")
            stream.flush()
        except Exception:                     # noqa: BLE001 -- logging's contract: emit never raises
            self.handleError(record)


@contextmanager
def console_handlers():
    """The two handlers on the ``core`` logger for the duration of one handler call, removed in a
    finally: ``main`` runs repeatedly in one process under the suite, and a handler left behind would
    print every later run's records once more per leak."""
    out = _LiveStreamHandler(sys_attr="stdout", max_level=logging.INFO)
    err = _LiveStreamHandler(sys_attr="stderr", min_level=logging.WARNING, prefix=True)
    logger = logging.getLogger("core")
    logger.addHandler(out)
    logger.addHandler(err)
    try:
        yield
    finally:
        logger.removeHandler(out)
        logger.removeHandler(err)
