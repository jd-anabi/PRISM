"""The dated text file of one run: every finished line the run writes, stamped, as it is written.

A line reads ``2026-10-08 19:02:11 warning text``: the run clock's local date and time, the line's level and its
text, one space between each. The further lines of a text written whole (a traceback) are each indented by the
stamp's width, ``STAMP_WIDTH``, so they stand under the text of the line they belong to. The levels are ``LEVELS``:
the three of the engine's records, ``err`` for what a library prints on the error stream, and ``window`` for the
window's own lines.

The writer is called from two sides. ``line`` is the run's line sink: it runs on whichever thread wrote the line,
under the lock of the stream pump that collects the run's output, and it is reached while the engine writes inside
a save that must not be interrupted. So it never raises, never prints, never logs and is never a point where a
requested stop is taken: a raise would stop the run halfway through a save, and anything it said would come back
into the same pump, under the lock already held. A file that cannot be written breaks the writer for the rest of
the run; ``line`` hands back, once, the window's sentence saying so, for the pump to queue. ``head``,
``separator``, ``outcome``, ``sync`` and ``close`` are called by the window or by the worker that ran the run. The
writer holds a lock of its own for all of them.

Every line is flushed to the operating system as it is written, so a run whose process is killed loses no finished
line. The disk itself is asked to hold the file (``os.fsync``) only by ``sync``, at most every ``FSYNC_EVERY_S``
seconds, which the window calls from a timer, and by ``close``, each outside the writer's lock: the disk's time must
never stall the run's own print or record. A power cut loses at most that interval's lines.

The standard library alone: no torch, nothing of the toolkit.
"""
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Callable

#: The levels a line carries: the engine's records' three, a print on the error stream, the window's own line.
LEVELS = ("info", "warning", "error", "err", "window")
#: The least time, in seconds of the run's clock, between two requests that the disk hold the file.
FSYNC_EVERY_S = 30.0
#: The width of a line's date and time with the space after them; a text's further lines are indented by it.
STAMP_WIDTH = 20

#: The window's line, at warning, when the file cannot be written; filled with the error's kind and message.
BROKEN_TEXT = "The run's text file could not be written ({}); the full text is still on the page."


class RunTextFile:
    """The text file of one run, at ``path``. ``clock`` gives the moment each line is stamped with, in local
    time. The file and its folder are made at the first line, so making the writer never fails. ``broken`` is
    True once a write has failed; nothing is written after that."""

    def __init__(self, path: Path, *, clock: Callable[[], datetime]):
        self.path = Path(path)
        self.broken = False
        self._clock = clock
        self._lock = threading.Lock()
        self._file = None
        self._synced = None                   # when the disk was last asked to hold the file
        self._closed = False

    def line(self, text: str, level: str) -> str | None:
        """Write one finished line of the run. Returns None; at the write that broke the file, the window's
        sentence saying so, once. Never raises."""
        with self._lock:
            return self._write(text, level)

    def head(self, *, word: str, name: str, run: str, memory_limit: str) -> None:
        """The run's first lines, the window's own: which run this is (its name left out when it has none), and
        the memory limit in force as the window words it (left out when it hands none)."""
        with self._lock:
            self._write(f"{word} {name}, run {run}." if name else f"{word}, run {run}.", "window")
            if memory_limit:
                self._write(memory_limit, "window")

    def separator(self, attempt: int) -> None:
        """The line that opens a try of the run after its first."""
        with self._lock:
            self._write(f"Try {attempt} begins here.", "window")

    def outcome(self, text: str, level: str, detail: str = "") -> None:
        """How the run ended, which the worker writes before it says anything to the window: ``text`` at
        ``level``, with ``detail`` (a traceback) as its further lines."""
        with self._lock:
            self._write(f"{text}\n{detail}" if detail else text, level)

    def sync(self) -> None:
        """Ask the disk to hold what is written, when ``FSYNC_EVERY_S`` seconds of the clock have passed since it
        was last asked (at once the first time). The request is made outside the writer's lock, on a copy of the
        file's handle, so a line written meanwhile never waits for the disk. Never raises."""
        with self._lock:
            if self._file is None or self._closed or self.broken:
                return
            try:
                now = self._clock()
                if self._synced is not None and (now - self._synced).total_seconds() < FSYNC_EVERY_S:
                    return
                self._file.flush()
                handle = os.dup(self._file.fileno())
            except Exception:                 # noqa: BLE001 -- a request that cannot be made is skipped
                return
            self._synced = now
        try:
            os.fsync(handle)
        except Exception:                     # noqa: BLE001 -- the next request tries again
            pass
        finally:
            try:
                os.close(handle)
            except OSError:
                pass

    def close(self) -> None:
        """Write what is left, ask the disk to hold it and close the file. Never raises; once closed, the writer
        writes nothing more. The file is taken from the writer under its lock and finished outside it: nothing
        else can reach the file once it is taken, and a line arriving meanwhile is turned away at once instead of
        waiting for the disk."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            file, self._file = self._file, None
        if file is None:
            return
        try:
            file.flush()
            os.fsync(file.fileno())
        except Exception:                     # noqa: BLE001 -- a file that cannot be closed cleanly is still let go
            pass
        finally:
            try:
                file.close()
            except Exception:                 # noqa: BLE001
                pass

    def _write(self, text: str, level: str) -> str | None:
        """One line, stamped, its further lines indented; flushed. Called under the writer's lock."""
        if self.broken or self._closed:
            return None
        try:
            if self._file is None:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self._file = open(self.path, "a", encoding="utf-8", errors="backslashreplace")
            first, *more = str(text).splitlines() or [""]
            stamp = self._clock().strftime("%Y-%m-%d %H:%M:%S")
            self._file.write("".join([f"{stamp} {level} {first}\n",
                                      *(f"{' ' * STAMP_WIDTH}{further}\n" for further in more)]))
            self._file.flush()
            return None
        except Exception as e:                # noqa: BLE001 -- the writer never raises into a run
            self.broken = True
            return BROKEN_TEXT.format(f"{type(e).__name__}: {e}")
