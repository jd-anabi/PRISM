"""What the window's suites share: the session a window holds, and the ways a test shows a window where no screen
shows it, brings it to the front, presses a key where a person's key goes and notes the focus given.

At its head this file imports the standard library and the window's session class, which loads neither the toolkit
nor torch; each toolkit name is imported inside the function that uses it, so that a suite that may not load the
toolkit or the engine at its head can import it, and the look's fresh-interpreter test reads every import in it.
Nothing here builds a window: the builders of tests/_fixtures.py do that, and a test hands the window in.
"""
import contextlib
import warnings

from core.gui.session import SbiSession


def session(**held) -> SbiSession:
    """The window's own session class holding what is named, and nothing in any other field: its five fields are
    ``draft``, ``cfg``, ``inf_prior``, ``posterior`` and ``observation``, and a name it does not have is refused."""
    return SbiSession(**held)


def shown_off_screen(window, width=None, height=None):
    """Show ``window`` where no screen shows it, at a size when one is handed, and let its layouts settle."""
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication
    window.setAttribute(Qt.WA_DontShowOnScreen)
    if width is not None:
        window.resize(width, height)
    window.show()
    QApplication.processEvents()


@contextlib.contextmanager
def in_front(widget):
    """``widget`` as the window in front, as the screen makes a window when a person brings it there. Shown where
    no screen shows it, the window is brought there by telling the toolkit so, through a call the toolkit keeps
    though it marks it as old, whose warning is not let through; the focus the window gives is then delivered with
    its reason, as the screen delivers it. At the end no window is in front."""
    from PySide6.QtWidgets import QApplication

    def tell(front):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            QApplication.setActiveWindow(front)
    tell(widget)
    try:
        yield widget
    finally:
        tell(None)


def press(window, sequence):
    """Press the key ``sequence`` ("Ctrl+3", "Shift+F6") where a person's key goes: at the control that holds the
    keyboard in ``window``, or at the window itself when none does."""
    from PySide6.QtGui import QKeySequence
    from PySide6.QtTest import QTest
    combination = QKeySequence(sequence)[0]
    QTest.keyClick(window.focusWidget() or window, combination.key(), combination.keyboardModifiers())


def given(monkeypatch):
    """Every focus given from now on, each as the widget and the reason it was given for, in their order."""
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QWidget
    given, kept = [], QWidget.setFocus

    def set_focus(widget, *reason):
        given.append((widget, reason[0] if reason else Qt.OtherFocusReason))
        return kept(widget, *reason)

    monkeypatch.setattr(QWidget, "setFocus", set_focus)
    return given
