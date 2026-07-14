"""Route worker-thread stdout/stderr + warnings to Qt signals, so the CLI-first pipeline's tqdm bars,
print() diagnostics and warnings render in the GUI's progress/log widgets instead of a console.

Two layers:
  * core.gui.vt.StreamRouter turns each write() chunk into structured events (a progress row upsert,
    a row retirement, or a completed log line). It is Qt-free and does no I/O.
  * _Pump coalesces those events on a daemon thread and emits them to Qt at PUMP_HZ.

The pump is load-bearing, not a nicety. tqdm's own `mininterval` does NOT bound the redraw rate here:
set_description() (tqdm/std.py:1382) and reset() (:1360) both call refresh() -> display() with no time
gate at all, and core/SBI/Priors/{bp,hopf,nadrowski}_prior.py call both on EVERY sweep iteration. Left
unthrottled, that floods the GUI's event queue with cross-thread queued signals.
"""
import sys
import threading
import time
import warnings
from contextlib import contextmanager

from .vt import StreamRouter

PUMP_HZ = 15.0
_TICK = 1.0 / PUMP_HZ

# redirect_streams swaps sys.stdout/stderr PROCESS-WIDE. BasePanel._busy only guards one panel, and
# Phase 2 adds more tabs onto the same QThreadPool -- two concurrent redirects would nest, and the
# inner one's restore would leave sys.stdout pointing at the outer worker's dead stream forever.
# Second and later redirects decline and let their output go to the real console.
_REDIRECT = threading.Lock()


class WorkerCancelled(BaseException):
    """Unwinds a cooperatively-cancelled worker run.

    Derived from BaseException, NOT Exception, ON PURPOSE: the pipeline is full of `except Exception`
    (in sbi, in core/FDT/cross_validation.py, in Worker.run itself) and a cancel must sail straight
    through all of them to reach Worker.run, which catches it by name. Verified: a BaseException
    subclass is not caught by `except Exception`.
    """


class CancelToken:
    """The cancel flag for one run, shared by the run's stdout and stderr streams.

    `requested` is set from the GUI thread (the Cancel button); `fired` is the LATCH -- once the raise
    has happened once, later writes pass through normally. The latch matters because tqdm's __iter__
    does `finally: self.close()`, which re-enters write() DURING the unwind; raising again there would
    replace the clean traceback with a chained one and abort the teardown writes that carry a
    leave=True bar's final frame into the log. One raise, then quiet.

    The raise is restricted to the OWNER (worker) thread. tqdm runs a TMonitor DAEMON thread that
    force-refreshes a bar gone quiet for >=maxinterval (10s) -- that refresh writes to our stream too.
    Without the owner guard, the monitor thread would reach check() first, consume the latch, raise
    WorkerCancelled where nobody catches it (killing the monitor), and leave the worker thread to sail
    past a fired latch -- silently losing the cancel for the whole run.
    """

    def __init__(self):
        self.requested = threading.Event()
        self.fired = False
        self._owner: int | None = None

    def arm(self) -> None:
        """Bind the token to the calling thread as the only one allowed to raise. Called from
        redirect_streams, which runs on the worker thread."""
        self._owner = threading.get_ident()

    def check(self) -> None:
        if not self.requested.is_set() or self.fired:
            return
        if self._owner is not None and threading.get_ident() != self._owner:
            return                            # a non-worker writer (tqdm's monitor) -- do not consume
        self.fired = True
        raise WorkerCancelled()


def reset_tqdm_lock() -> None:
    """Recover tqdm's global write lock after a cancel unwound through a bar's redraw.

    tqdm's refresh() (tqdm/std.py:1346-1349) does a MANUAL acquire/display/release, not a `with`:
        self._lock.acquire(); self.display(); self._lock.release()
    Our cancel raises from inside display()'s write(), so the release is skipped and the lock leaks --
    after which the NEXT tqdm.__new__ (which acquires that lock) deadlocks. We can't release a lock we
    may not own, so we ABANDON it: drop the class-level sub-locks and install a fresh write lock. Safe
    because only one run happens at a time (the app-wide guard) and this runs after the run unwound, so
    no bar is live. Verified to clear the deadlock.
    """
    try:
        from tqdm import tqdm
        from tqdm.std import TRLock, TqdmDefaultWriteLock
        TqdmDefaultWriteLock.mp_lock = None
        TqdmDefaultWriteLock.th_lock = TRLock()
        tqdm.set_lock(TqdmDefaultWriteLock())
    except Exception:                     # noqa: BLE001 -- best-effort recovery, never mask the cancel
        pass


class _Pump:
    """Collect router events; emit them to Qt at PUMP_HZ from a daemon thread.

    Progress rows are last-wins (a dict): intermediate frames within a tick are dropped, not queued --
    progress is idempotent state, not a log. Log lines are order-preserving and never dropped.
    """

    def __init__(self, signals):
        self._signals = signals
        self._lock = threading.Lock()
        self._rows: dict[tuple, object] = {}
        self._logs: list[tuple[str, str]] = []
        self._dirty = False
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="gui-progress-pump", daemon=True)
        self._thread.start()

    def sink(self, kind, payload):
        """Called synchronously on the WRITER thread. Does no Qt work and never blocks on Qt."""
        with self._lock:
            if kind == "row":
                self._rows[payload.key] = payload
            elif kind == "retire":
                self._rows.pop(payload, None)
            elif kind == "log":
                self._logs.append(payload)
            self._dirty = True

    def _take(self):
        with self._lock:
            if not self._dirty:
                return None, None
            self._dirty = False
            rows = tuple(sorted(self._rows.values(), key=lambda s: (s.key[0], s.row)))
            logs, self._logs = self._logs, []
            return rows, logs

    def _publish(self):
        rows, logs = self._take()
        if rows is None:
            return
        # Emitted OUTSIDE the lock. redirect_streams is process-wide, so a GUI-thread print() also
        # lands in write(); emitting under the lock would let Qt pick a DirectConnection and run
        # widget code while we hold it.
        try:
            if logs:
                self._signals.log_batch.emit(logs)
            self._signals.rows.emit(rows)
        except RuntimeError:
            # "Signal source has been deleted" -- the window was closed while this run was still going,
            # so the QApplication (and our WorkerSignals) are gone. There is nobody left to tell.
            self._stop.set()

    def _loop(self):
        while True:
            stopping = self._stop.wait(_TICK)
            self._publish()
            if stopping:
                return

    def stop(self):
        """Publish whatever is left, then shut down.

        The final publish runs ON THE PUMP THREAD (we set the flag and join) rather than inline on the
        caller's thread. That matters: if the caller is the GUI thread, Qt resolves a same-thread emit
        as a DirectConnection and its slots run IMMEDIATELY -- ahead of every tick still sitting in the
        event queue -- which silently scrambles the log into teardown-first order.
        """
        self._stop.set()
        self._thread.join(timeout=5.0)


class _SignalStream:
    """File-like stdout/stderr replacement. Feeds a StreamRouter; the pump does all the Qt work.

    DO NOT add a `fileno()` or an `encoding` attribute -- tqdm probes for them, and finding them
    re-enables the screen-shape probe (ncols/nrows) and flips `ascii`, which changes the frame format
    core.gui.vt parses. See the invariant note in that module.
    """

    def __init__(self, pump, name: str, level: str = "info", cancel: "CancelToken | None" = None):
        self._pump = pump
        self._level = level
        self._cancel = cancel
        self.router = StreamRouter(name, pump.sink, level=level)
        self._broken = False

    def write(self, text: str):
        if not text:
            return 0
        # The cancel check sits BEFORE the try below: WorkerCancelled is a BaseException, so that
        # `except Exception` would not catch it anyway, but keeping it outside makes the intent explicit
        # -- a cancel is not a "parser broke" degradation. Every print() and every tqdm redraw funnels
        # through here, so this is the pipeline's cancellation checkpoint, reaching even inside sbi's
        # fit loop (it prints an epoch counter every epoch).
        if self._cancel is not None:
            self._cancel.check()
        if self._broken:                      # degraded: dumb line split, but never lose output
            self._pump.sink("log", (text.rstrip(), self._level))
            return len(text)
        try:
            self.router.feed(text)
        except Exception:                     # noqa: BLE001
            # tqdm calls fp.write() from inside a 4-deep loop and does NOT catch (its
            # DisableOnWriteError only swallows OSError(errno=5)/ValueError('closed')), so a raise
            # here would kill a multi-hour run. Degrade instead.
            self._broken = True
            self._pump.sink("log", ("[progress parser disabled after an internal error]", "warning"))
        return len(text)

    def flush(self):
        pass

    def isatty(self):
        return False


@contextmanager
def redirect_streams(signals, cancel: "CancelToken | None" = None):
    """Swap sys.stdout/stderr for signal-emitting streams and route warnings.warn to the log.

    `cancel`, when given, is shared by both streams: a set-and-not-yet-fired token makes the next
    write() (i.e. the next print or tqdm redraw) raise WorkerCancelled. Yields the _Pump (or None if
    another redirect already owns the process's streams) so the caller can drain() it before emitting
    its result.
    """
    if not _REDIRECT.acquire(blocking=False):
        signals.log.emit("Another task already owns the console; this run's output is not captured.",
                         "warning")
        yield None
        return

    # Everything from here to the release is inside the try: _Pump() starts a thread, and a failure to
    # do so (thread exhaustion in a torch/BLAS process, or interpreter shutdown) would otherwise leave
    # _REDIRECT held for the rest of the process -- after which EVERY later run takes the decline branch
    # above and silently loses all of its GUI output.
    pump = out = err = None
    old_out, old_err, old_showwarning = sys.stdout, sys.stderr, warnings.showwarning
    if cancel is not None:
        cancel.arm()                          # this runs on the worker thread -> the only one that raises
    try:
        pump = _Pump(signals)
        out = _SignalStream(pump, "out", "info", cancel)
        err = _SignalStream(pump, "err", "warning", cancel)
        sys.stdout, sys.stderr = out, err

        def _showwarning(message, category, filename, lineno, file=None, line=None):
            pump.sink("log", (f"{getattr(category, '__name__', 'Warning')}: {message}", "warning"))

        warnings.showwarning = _showwarning
        yield pump
    finally:
        # Settle both routers (retiring live rows and flushing partial lines) while the streams are
        # still ours, then stop the pump -- so nothing a late/GC'd tqdm writes can reach a dead widget.
        for stream in (out, err):
            if stream is None:
                continue
            try:
                stream.router.close()
            except Exception:                 # noqa: BLE001 -- teardown must not mask a worker error
                pass
        sys.stdout, sys.stderr = old_out, old_err
        warnings.showwarning = old_showwarning
        if pump is not None:
            pump.stop()
        # If the cancel fired, it raised from inside a tqdm redraw and may have leaked tqdm's write
        # lock; recover it here on the worker thread so the NEXT run's first bar does not deadlock.
        if cancel is not None and cancel.fired:
            reset_tqdm_lock()
        _REDIRECT.release()
