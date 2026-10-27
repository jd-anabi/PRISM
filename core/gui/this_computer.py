"""This computer: every request PRISM makes of Windows for a run.

While a run is live the computer is asked to stay awake (``keep_awake``), and Windows is given the reason it should
not end the session (``block_shutdown``): the words Windows lists beside PRISM when a sign-out or a shutdown is asked
for. The window's button on the taskbar is flashed with ``flash``. Each request is made on the window's thread: the
run's controller makes them from its slots and its own timer, which live there. Windows keeps a request to stay awake
only while the thread that made it lives, and a reason belongs to the window it is registered on. ``state`` is the
last request this process made of each.

Every call answers False, never an error: off Windows, off the toolkit's real ``windows`` platform (the suites run
off the screen) and when Windows refuses. Windows is reached in one place, ``_windows``. No torch is loaded here.
"""
import sys

#: What ``SetThreadExecutionState`` is asked: the computer kept awake until asked again, then the request ended.
ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001
#: The most characters of a reason Windows takes: a longer one is refused whole, so longer words are cut to this,
#: their last character an ellipsis.
REASON_CHARS = 256

# The last request this process made: the computer kept awake, and the reason registered (None for none).
_requested = {"awake": False, "reason": None}


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
