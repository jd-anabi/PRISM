"""This computer: every request PRISM makes of Windows for a run, and what Windows' end of a session tells it.

While a run is live the computer is asked to stay awake (``keep_awake``), and Windows is given the reason it should
not end the session (``block_shutdown``): the words Windows lists beside PRISM when a sign-out or a shutdown is asked
for. The window's button on the taskbar is flashed with ``flash``. Each request is made on the window's thread: the
run's controller makes them from its slots and its own timer, which live there. Windows keeps a request to stay awake
only while the thread that made it lives, and a reason belongs to the window it is registered on. ``state`` is the
last request this process made of each.

``watch_session_end`` hears Windows end the session, a sign-out or a shutdown, all on the window's thread and inside
Windows' own messages. Windows' question comes first, as the toolkit's ``commitDataRequest``. "Shut down anyway"
comes as the toolkit's ``aboutToQuit``, inside Windows' end message, after a question; an ordinary quit raises the
same signal with no question before it. "Cancel" gives the toolkit no signal at all: only a filter of the native
messages sees it, as Windows' end message saying the session does not end. The process holds one watch at a time
(``watching``).

Every call answers False, never an error: off Windows, off the toolkit's real ``windows`` platform (the suites run
off the screen) and when Windows refuses. Windows is reached in one place, ``_windows``. No torch is loaded here.
"""
import sys

from PySide6.QtCore import QAbstractNativeEventFilter, QObject

#: What ``SetThreadExecutionState`` is asked: the computer kept awake until asked again, then the request ended.
ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001
#: The most characters of a reason Windows takes: a longer one is refused whole, so longer words are cut to this,
#: their last character an ellipsis.
REASON_CHARS = 256
#: Windows' end message, and the kind of event the toolkit hands a native filter one of Windows' messages as.
WM_ENDSESSION = 0x0016
NATIVE_MESSAGE = b"windows_generic_MSG"

# The last request this process made: the computer kept awake, and the reason registered (None for none).
_requested = {"awake": False, "reason": None}
# The process's watch on Windows' end of a session, while there is one.
_watch = None


def state() -> tuple[bool, str | None]:
    """The last request this process made: whether the computer is asked to stay awake, and the reason registered
    against ending the session as Windows was handed it (None for none)."""
    return _requested["awake"], _requested["reason"]


def keep_awake(on: bool) -> bool:
    """Ask Windows to keep the computer awake (``on``), or end that request; the request is this thread's. True when
    Windows took it."""
    _requested["awake"] = bool(on)
    if not _on_windows():
        return False
    try:
        import ctypes
        call = _windows("kernel32", "SetThreadExecutionState")
        call.argtypes, call.restype = [ctypes.c_uint32], ctypes.c_uint32
        return bool(call((ES_CONTINUOUS | ES_SYSTEM_REQUIRED) if on else ES_CONTINUOUS))
    except Exception:                       # noqa: BLE001 -- a request Windows did not take is no run's failure
        return False


def block_shutdown(window, reason: str | None) -> bool:
    """Give Windows ``reason`` as the reason not to end the session, registered on ``window``'s handle in place of the
    one registered there before (withdrawn first); None withdraws it. Words longer than Windows takes are cut
    (``fitted``). True when Windows took the request."""
    words = None if reason is None else fitted(reason)
    _requested["reason"] = words
    if window is None or not _on_windows():
        return False
    try:
        from ctypes import wintypes
        handle = int(window.winId())
        destroy = _windows("user32", "ShutdownBlockReasonDestroy")
        destroy.argtypes, destroy.restype = [wintypes.HWND], wintypes.BOOL
        withdrawn = bool(destroy(handle))
        if words is None:
            return withdrawn
        create = _windows("user32", "ShutdownBlockReasonCreate")
        create.argtypes, create.restype = [wintypes.HWND, wintypes.LPCWSTR], wintypes.BOOL
        return bool(create(handle, words))
    except Exception:                       # noqa: BLE001 -- a reason Windows did not take is no run's failure
        return False


def fitted(reason: str) -> str:
    """``reason`` as Windows takes it: whole when it fits in ``REASON_CHARS``, else cut there, the last character
    kept an ellipsis."""
    return reason if len(reason) <= REASON_CHARS else reason[:REASON_CHARS - 1] + "…"


def flash(window) -> bool:
    """Flash ``window``'s button on the taskbar until the window is brought to the front. True when Windows was
    asked."""
    if window is None or not _on_windows():
        return False
    try:
        from PySide6.QtWidgets import QApplication
        QApplication.alert(window)
        return True
    except Exception:                       # noqa: BLE001 -- a flash Windows did not take is nobody's failure
        return False


def watching():
    """The process's watch on Windows' end of a session, or None."""
    return _watch


def watch_session_end(app, *, asked, ending) -> "SessionWatch":
    """Hear Windows end the session from now on (``SessionWatch``): ``asked(manager)`` at Windows' question,
    ``ending(True)`` at "shut down anyway", ``ending(False)`` at "cancel". The process's watch, in place of any it
    held."""
    global _watch
    if _watch is not None:
        _watch.remove()
    _watch = SessionWatch(app, asked=asked, ending=ending)
    return _watch


class SessionWatch(QObject):
    """A watch on Windows' end of a session. Windows' question reaches ``asked`` with the toolkit's session manager,
    whose ``cancel`` answers Windows that PRISM blocks the end. The toolkit's ``aboutToQuit`` reaches ``ending(True)``
    only when a question was heard since the last cancel: "shut down anyway", never an ordinary quit. One native
    filter (``filter``) sees Windows' end message; one saying the session does not end, after a question, reaches
    ``ending(False)`` once (Windows may send it to each of the process's windows), and the filter consumes no message.
    ``remove`` takes all of it off the application."""

    def __init__(self, app, *, asked, ending):
        super().__init__()
        self._app, self._asked, self._ending = app, asked, ending
        self.heard = False                       # a question was heard since the last cancel
        self._removed = False
        self.filter = _EndFilter(self._cancelled)
        app.commitDataRequest.connect(self._question)
        app.aboutToQuit.connect(self._quitting)
        app.installNativeEventFilter(self.filter)

    def _question(self, manager) -> None:
        self.heard = True
        self._asked(manager)

    def _quitting(self) -> None:
        if self.heard:
            self._ending(True)

    def _cancelled(self) -> None:
        if self.heard:
            self.heard = False
            self._ending(False)

    def remove(self) -> None:
        """The watch off the application, once: its two signals and its filter. The process holds it no more."""
        global _watch
        if self._removed:
            return
        self._removed = True
        self._app.commitDataRequest.disconnect(self._question)
        self._app.aboutToQuit.disconnect(self._quitting)
        self._app.removeNativeEventFilter(self.filter)
        if _watch is self:
            _watch = None


class _EndFilter(QAbstractNativeEventFilter):
    """Sees each of Windows' messages before the toolkit handles it, and hands on Windows' end message saying the
    session does not end ("cancel"). It never consumes a message, and never raises into the toolkit."""

    def __init__(self, cancelled):
        super().__init__()
        self._cancelled = cancelled

    def nativeEventFilter(self, event_type, message):
        try:
            if bytes(event_type) == NATIVE_MESSAGE:
                from ctypes import wintypes
                msg = wintypes.MSG.from_address(int(message))
                if msg.message == WM_ENDSESSION and not msg.wParam:
                    self._cancelled()
        except Exception:                   # noqa: BLE001 -- a message read wrong is passed on as it came
            pass
        return False, 0


def _on_windows() -> bool:
    """Whether this process can make a request of Windows for its window: on Windows, with the toolkit's application
    made on its real ``windows`` platform. Before the application is made the toolkit names its default platform,
    which is that one, so the application is asked for first."""
    if sys.platform != "win32":
        return False
    from PySide6.QtGui import QGuiApplication
    return QGuiApplication.instance() is not None and QGuiApplication.platformName() == "windows"


def _windows(library: str, name: str):
    """The function ``name`` of Windows' ``library`` ("kernel32", "user32"): the one place this module reaches
    Windows. A library object of this module's own, so that no other caller's description of the same function is
    touched."""
    import ctypes
    return getattr(ctypes.WinDLL(library, use_last_error=True), name)
