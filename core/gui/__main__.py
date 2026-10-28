"""Entry point: ``python -m core.gui``, from any working directory (``core/config.py`` resolves ``Resources/`` and the
artifact store from its own location; ``PRISM_RESOURCES`` / ``PRISM_ARTIFACTS`` override them).

What runs first is cheap, so that a second start of PRISM, which only knocks on the open one, ends at once: this
module imports ``os`` and ``sys`` alone, sets ``KMP_DUPLICATE_LIB_OK`` before anything can load torch (torch and MKL
each ship an OpenMP runtime, and without it the first simulation aborts with no traceback), and ``main`` claims the
one PRISM before it imports the window, which loads torch and the engine.

The figure library is told to draw with Agg before any module that draws is imported: figures are still made, and
the window embeds them, while a stray ``plt.show()`` in the engine's plotters does nothing, which is what lets the
pipeline run on a background thread.

Started with no console -- by ``pythonw``, as the launcher starts it, or detached from its terminal, as ``run.sh``
starts it -- the program has no standard streams, or none that anybody reads. They are then bound to
``window-log.txt`` beside the settings file, which takes what the window prints and logs outside a run, the records
another thread logs during one, the toolkit's own messages, an error nothing else caught and a crash of the
interpreter itself. A start that fails before its window is shown writes the failure there with its trace, and says
so in a plain Windows message that names the file: not in a window of the toolkit's, which may be what failed.

Run as ``python -m core.gui`` this file is the module ``__main__``; it hands over at once to itself imported under
its own name, ``core.gui.__main__``, so that the binding the window reads (``bound_log``) is kept in the one module
the window and the suites import.
"""
import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

LOG_NAME = "window-log.txt"
COULD_NOT_START = "PRISM could not start. The details are in {}."

# While the standard streams are bound to the window's log, what the binding put in place and what it found there
# (a ``_Binding``); None otherwise.
_bound = None


class _Binding:
    """What ``bind_streams`` put in place and what it found there, for ``unbind_streams`` to put back."""

    def __init__(self, path, log, streams):
        self.path = path
        self.log = log
        self.streams = streams
        self.fault = False            # the fault handler was free, and the binding took it
        self.qt = None                # the toolkit's message handler the binding installed
        self.qt_found = None          # ... and the one it found there, which ``unbind_streams`` puts back


def window_log_path():
    """The window's log: ``window-log.txt`` in the folder of the settings file, the folder made, and the file, when
    they are not there yet; in the system's temporary folder when that folder cannot be made or that file cannot be
    opened to be written."""
    import tempfile
    from pathlib import Path
    from core.gui import settings
    beside = Path(settings.settings().fileName()).parent / LOG_NAME
    try:
        beside.parent.mkdir(parents=True, exist_ok=True)
        with open(beside, "a", encoding="utf-8"):
            pass
    except OSError:
        return Path(tempfile.gettempdir()) / LOG_NAME
    return beside


def bind_streams(log_path) -> None:
    """Bind the standard streams to the window's log at ``log_path``: both to the one file, opened to append, in
    UTF-8, written out at every line, which this launch opens with a dated line naming its interpreter. A crash of
    the interpreter itself is written there by the fault handler, when nothing holds that handler yet (an interpreter
    told to write its crashes elsewhere goes on doing so), and the toolkit's own messages are written there with
    their kind. What was in place is kept for ``unbind_streams``. Bound already, nothing is done."""
    global _bound
    if _bound is not None:
        return
    import faulthandler
    import time
    from pathlib import Path
    log = open(log_path, "a", encoding="utf-8", errors="backslashreplace", buffering=1)
    log.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} PRISM started ({sys.executable}).\n")
    binding = _Binding(Path(log_path), log, (sys.stdout, sys.stderr))
    _bound = binding
    sys.stdout = sys.stderr = log
    if not faulthandler.is_enabled():
        faulthandler.enable(file=log)
        binding.fault = True
    try:
        from PySide6.QtCore import qInstallMessageHandler
    except ImportError:
        return                        # the toolkit cannot be loaded, so none of its messages can come
    binding.qt = _qt_messages_to(log)
    binding.qt_found = qInstallMessageHandler(binding.qt)


def _qt_messages_to(log):
    """The toolkit's message handler that writes each of its messages to ``log`` with its kind, "Qt warning: ...",
    and raises nothing, on whichever thread the toolkit calls it."""
    def written(kind, context, message):
        word = getattr(kind, "name", str(kind)).removeprefix("Qt").removesuffix("Msg").lower()
        try:
            log.write(f"Qt {word}: {message}\n")
        except Exception:                                 # noqa: BLE001 -- a message that cannot be written is lost
            pass
    return written


def bound_log():
    """The file the standard streams are bound to, while they are (PRISM started with no console); else None."""
    return None if _bound is None else _bound.log


def unbind_streams() -> None:
    """Put back what ``bind_streams`` found: the toolkit's message handler, the fault handler and each standard
    stream that is still the window's log; the log is closed when both were. Called last as the program ends, and by
    a test after it binds. Not bound, nothing is done."""
    global _bound
    if _bound is None:
        return
    binding, _bound = _bound, None
    if binding.qt is not None:
        from PySide6.QtCore import qInstallMessageHandler
        qInstallMessageHandler(binding.qt_found)
    if binding.fault:
        import faulthandler
        faulthandler.disable()
    given_back = 0
    if sys.stdout is binding.log:
        sys.stdout = binding.streams[0]
        given_back += 1
    if sys.stderr is binding.log:
        sys.stderr = binding.streams[1]
        given_back += 1
    if given_back == 2:               # nothing stands in front of the log that could hand it back later
        binding.log.close()


def _native_box(text: str) -> None:
    """Show ``text`` in a plain Windows message with PRISM's name as its title and an error's mark, and wait for its
    OK. The one place PRISM shows such a message: a start that failed before its window was shown, when the
    toolkit's own boxes cannot be trusted. No test calls it, since it waits for a person. Off Windows nothing is
    shown: the log holds the words."""
    if sys.platform != "win32":
        return
    try:
        import ctypes
        ctypes.windll.user32.MessageBoxW(None, text, "PRISM", 0x10 | 0x10000)   # the error's mark, in front
    except Exception:                                     # noqa: BLE001 -- a message that cannot be shown is lost
        pass


def _could_not_start() -> None:
    """Write the failure being handled, with its trace, to the window's log -- the one the streams are bound to;
    else the one beside the settings file, and the trace to the console too -- and say in a plain Windows message
    that PRISM could not start, and where the details are."""
    import tempfile
    import time
    import traceback
    from pathlib import Path
    trace = traceback.format_exc()
    words = f"{time.strftime('%Y-%m-%d %H:%M:%S')} PRISM could not start.\n{trace}"
    if _bound is not None:
        path = _bound.path
        try:
            _bound.log.write(words)
        except Exception:                                 # noqa: BLE001 -- the message is still shown
            pass
    else:
        try:
            path = window_log_path()
        except Exception:                                 # noqa: BLE001 -- the toolkit itself may be what failed
            path = Path(tempfile.gettempdir()) / LOG_NAME
        try:
            with open(path, "a", encoding="utf-8", errors="backslashreplace") as log:
                log.write(words)
        except OSError:
            pass
        if sys.stderr is not None:
            try:
                sys.stderr.write(trace)
                sys.stderr.flush()
            except (OSError, ValueError):
                pass
    _native_box(COULD_NOT_START.format(path))


def main() -> int:
    """Start PRISM, and return the status the process ends with.

    In this order: with no output or no error stream, or started detached (``PRISM_DETACHED``), the streams are
    bound to the window's log; the figure library draws with Agg; the Windows shell is told PRISM is an application
    of its own, and the application is made, with PRISM's names; the one PRISM is claimed. A start handed over to an
    open PRISM says so in one line and ends with 0, one that gives up ends with 1, both having built nothing and
    imported nothing of the window. Only then is the window's module imported and the program built, handed the
    application and the claim; the window is shown and the event loop runs. A failure before the window is shown is
    written to the log and said in a plain Windows message, and ends with 1. Whatever the end, the claim is let go
    and the streams are put back, last."""
    try:
        try:
            if sys.stdout is None or sys.stderr is None or os.environ.get("PRISM_DETACHED"):
                bind_streams(window_log_path())
            import matplotlib
            matplotlib.use("Agg")
            from PySide6.QtWidgets import QApplication
            from core.gui import app_icon, one_instance, settings
            # The Windows shell's grouping key, before the application and so before any window: set late, it leaves
            # the taskbar's button showing the interpreter's icon instead of PRISM's.
            app_icon.set_windows_app_user_model_id()
            app = QApplication.instance() or QApplication(sys.argv)
            app.setOrganizationName(settings.ORG)
            app.setApplicationName(settings.APP)
            claimed = one_instance.claim(settings.settings().fileName())
            if claimed.case in one_instance.SAID:
                one_instance.say(claimed.case)
                return 1 if claimed.case == "give_up" else 0
            from core.gui.app import build_app
            app, window = build_app(app=app, claimed=claimed)
            window.show()
        except Exception:                                 # noqa: BLE001 -- a failure to start must still be said
            _could_not_start()
            return 1
        return app.exec()
    finally:
        # The window's close let go of the claim as it was accepted; a loop that ended otherwise, or a start that
        # failed, lets go of it here. Letting go twice does nothing.
        held = sys.modules.get("core.gui.one_instance")
        if held is not None:
            held.release()
        unbind_streams()


if __name__ == "__main__":
    from core.gui.__main__ import main as _main
    raise SystemExit(_main())
