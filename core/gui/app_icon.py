"""The application / window icon, loaded from the PNG set in assets/app.

Why not the SVG directly: Qt's ``svg`` IMAGE-FORMAT plugin is absent in this environment, so
``QIcon("prism.svg")`` returns a NULL icon and the app silently keeps Qt's default mark with nothing
logged anywhere. assets/app/build_app_icon.py rasterises the SVG source to PNGs offline (via the QtSvg
*module*, which is a different thing and is available); this only ever loads those PNGs. Same
source-plus-committed-binary split as the icon font in icons.py.

Missing or unreadable PNGs are not an error: an app with no icon still runs, and refusing to start over
decoration would be absurd. The icon is simply empty, exactly as it was before this module existed.
"""
import sys
from pathlib import Path

from PySide6.QtGui import QIcon, QPixmap

_APP_DIR = Path(__file__).resolve().parent / "assets" / "app"
_ICO = _APP_DIR / "prism.ico"
_APP_USER_MODEL_ID = "PRISM.PRISM.DesktopApp.1"
_class_icons = []      # HICONs handed to the window class: they must outlive every window, so never freed


def app_icon() -> QIcon:
    """A multi-resolution QIcon, or an empty one if the assets are unavailable. Every size found is
    added so Qt picks the nearest rather than downscaling 256 to 16, which turns a stroked mark to
    mush -- see the note in assets/app/prism.svg."""
    icon = QIcon()
    for png in sorted(_APP_DIR.glob("prism-*.png")):
        pm = QPixmap(str(png))
        if not pm.isNull():
            icon.addPixmap(pm)
    return icon


def set_windows_app_user_model_id() -> None:
    """Tell the Windows shell this process is its own application.

    Without it the taskbar groups PRISM's windows under the HOST INTERPRETER's identity (python.exe),
    which is what pinning and jump lists key on. It is a shell grouping key, so it has to be set
    BEFORE the first window is created. It does NOT decide the button's icon -- measured 2026-09-11:
    with or without it the button showed the same glyph; see set_windows_class_icon for what does.
    No-op off Windows, and best-effort everywhere: a failure here costs taskbar grouping, which is
    never worth stopping a launch for.
    """
    if sys.platform != "win32":
        return
    try:
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(_APP_USER_MODEL_ID)
    except Exception:                                   # noqa: BLE001 -- decoration, never fatal
        pass


def set_windows_class_icon(window) -> bool:
    """Make PRISM's mark the icon of Qt's Win32 WINDOW CLASS, so the taskbar's fallback is ours.

    HOW THE TASKBAR PICKS A BUTTON'S ICON (measured on Windows 11 26200, 2026-09-11, the display
    walkthrough's row 1): when the native window first shows, the shell asks it for its icon with a
    short timeout; if the window's thread does not answer in time it falls back to the window CLASS
    icon and caches that for the button's lifetime. Qt registers its class with the stock generic
    "application" glyph because the host executable, python.exe, carries no IDI_ICON1 resource.
    PRISM's first show keeps the GUI thread busy for ~150 ms AFTER the native show (Qt lays out the
    whole widget tree), the query times out, and the generic glyph is what the user saw -- while a
    minimal window, idle right after showing, got the mark, and the same minimal window blocked for
    4 s after showing got the glyph. setWindowIcon is not the lever: the title bar and Alt-Tab were
    right all along. With the class icon set, even the 4 s block showed the mark.

    Must run BEFORE the first show; ``winId()`` creates the HWND hidden, which is all the class
    needs. The class is shared by every Qt top-level window in the process, so dialogs inherit it.
    The HICONs come from assets/app/prism.ico (LoadImage cannot read a PNG) and live for the process.
    Returns True when the class icon was set; False (never raises) off Windows, without the real
    ``windows`` platform plugin (offscreen tests), without the .ico, or on any Win32 failure.
    """
    if sys.platform != "win32" or not _ICO.is_file():
        return False
    try:
        from PySide6.QtGui import QGuiApplication
        if QGuiApplication.platformName() != "windows":
            return False
        import ctypes
        from ctypes import wintypes
        user32 = ctypes.windll.user32
        user32.LoadImageW.restype = wintypes.HANDLE
        user32.LoadImageW.argtypes = [wintypes.HINSTANCE, wintypes.LPCWSTR, wintypes.UINT,
                                      ctypes.c_int, ctypes.c_int, wintypes.UINT]
        user32.SetClassLongPtrW.restype = ctypes.c_void_p
        user32.SetClassLongPtrW.argtypes = [wintypes.HWND, ctypes.c_int, ctypes.c_void_p]
        IMAGE_ICON, LR_LOADFROMFILE = 1, 0x10
        GCLP_HICON, GCLP_HICONSM = -14, -34
        SM_CXICON, SM_CXSMICON = 11, 49
        hwnd = int(window.winId())
        cx, cxs = user32.GetSystemMetrics(SM_CXICON), user32.GetSystemMetrics(SM_CXSMICON)
        big = user32.LoadImageW(None, str(_ICO), IMAGE_ICON, cx, cx, LR_LOADFROMFILE)
        small = user32.LoadImageW(None, str(_ICO), IMAGE_ICON, cxs, cxs, LR_LOADFROMFILE)
        if not big or not small:
            return False
        user32.SetClassLongPtrW(hwnd, GCLP_HICON, big)
        user32.SetClassLongPtrW(hwnd, GCLP_HICONSM, small)
        _class_icons.extend((big, small))
        return True
    except Exception:                                   # noqa: BLE001 -- decoration, never fatal
        return False
