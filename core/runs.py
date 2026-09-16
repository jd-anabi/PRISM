"""The run boundary of the core (piece 3, V1 and V4): torch-free, imported by every public entry
point and by both front ends.

Two jobs, one decorator. `public_entry` wraps every public function that takes a SimConfig and may
write to it or to an artifact (spec §2.2 lists the fifteen):

- THE PRIVATE COPY (V1). The window builds ONE SimConfig at Build/Load prior, and every later run
  used to write onto that same object -- the observation length, the loaded cell's truth, a
  recording's drive values and probe count -- and nothing cleared them: a bench chi inference with
  three probes made the next simulated chi inference simulate three, and an amortized training
  after an inference anchored its Fisher rotation on the leaked truth. The decorator replaces the
  config argument with `cfg.copy_for_run()`, so the stage works on a copy and the caller's object
  is exactly what it was, whether the run succeeds, is refused or crashes. Duck-typed
  (`hasattr(cfg, "copy_for_run")`) so this module never imports torch, and so a stub or an
  `object()` sentinel passes through untouched.
- THE RUN LOG (V4). Records emitted since the OUTERMOST public entry began are buffered per thread,
  formatted `HH:MM:SS level message`, together with every Python warning raised meanwhile
  (PreflightWarning judgements included); ArtifactWriter._commit writes the buffer so far to
  `log.txt` in every artifact the entry commits. A composition and the stages it calls share one
  buffer: `capture_run` pushes only when none is active on this thread.

The `core` logger's level is set to INFO here, ONCE, at import. Python's root logger sits at
WARNING, so without this the window's handler and the artifact file would drop every information
record while `caplog` hid it in the suites. Nothing else touches the level: the window's handler
and the tool's `main` install and remove handlers only.
"""
import datetime
import functools
import logging
import threading
import warnings
from contextlib import contextmanager

LOGGER = logging.getLogger("core")
LOGGER.setLevel(logging.INFO)


def _stamp() -> str:
    return datetime.datetime.now().strftime("%H:%M:%S")


class _RunLogHandler(logging.Handler):
    """Appends every record the `core` logger passes at INFO and above to one RunLog."""

    def __init__(self, log: "RunLog"):
        super().__init__(level=logging.INFO)
        self._log = log

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._log.lines.append(f"{_stamp()} {record.levelname.lower()} {record.getMessage()}")
        except Exception:                      # noqa: BLE001 -- a bad format string must not kill a run
            self.handleError(record)


class RunLog:
    """Lines of one outermost public entry: `HH:MM:SS level message`, plus every Python warning."""

    def __init__(self):
        self.lines: list[str] = []
        self._handler = _RunLogHandler(self)
        self._prev_showwarning = None

    def attach(self) -> None:
        LOGGER.addHandler(self._handler)
        self._prev_showwarning = warnings.showwarning
        warnings.showwarning = self._showwarning

    def detach(self) -> None:
        LOGGER.removeHandler(self._handler)
        warnings.showwarning = self._prev_showwarning

    def _showwarning(self, message, category, filename, lineno, file=None, line=None):
        """The tee: record the warning, then hand it to whatever hook was installed before -- the
        window's log-pane hook under a run, pytest's recorder under `pytest.warns`, Python's own
        stderr printer otherwise. detach restores that hook, so the tee nests cleanly inside any of
        them."""
        self.lines.append(f"{_stamp()} warning {getattr(category, '__name__', 'Warning')}: {message}")
        if self._prev_showwarning is not None:
            self._prev_showwarning(message, category, filename, lineno, file, line)

    def text(self) -> str:
        return "\n".join(self.lines) + ("\n" if self.lines else "")


_active = threading.local()                    # _active.log: RunLog | None


def current_run_log() -> "RunLog | None":
    """The buffer of the outermost public entry running on THIS thread; None outside one."""
    return getattr(_active, "log", None)


@contextmanager
def capture_run():
    """Push a RunLog if none is active on this thread; else yield the active one and pop nothing.

    So a composition and the public stages it calls share one buffer, and each artifact they commit
    carries the records since the COMPOSITION began (the inference's file is a superset of the
    observation's). Popped -- handler removed, warnings hook restored -- on every exit, an
    exception's included, so a refused or crashed run leaves nothing installed on the process.
    """
    active = current_run_log()
    if active is not None:
        yield active
        return
    log = RunLog()
    log.attach()
    _active.log = log
    try:
        yield log
    finally:
        _active.log = None
        log.detach()


def public_entry(fn):
    """Decorate a public entry point: the config argument (the first positional, or the `cfg`
    keyword) is replaced by its `copy_for_run()` when it has one, and the call runs inside
    `capture_run()`.

    `functools.wraps`, so `inspect.signature` follows `__wrapped__` and `inspect.getsource` returns
    the original's source with the decorator in its `decorator_list`: the AST pins on the stages
    keep finding their needles. A composition looks its stages up in module globals at call time,
    so a monkeypatched stage name still takes effect inside it.
    """
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if "cfg" in kwargs:
            if hasattr(kwargs["cfg"], "copy_for_run"):
                kwargs = {**kwargs, "cfg": kwargs["cfg"].copy_for_run()}
        elif args and hasattr(args[0], "copy_for_run"):
            args = (args[0].copy_for_run(), *args[1:])
        with capture_run():
            return fn(*args, **kwargs)
    return wrapper
