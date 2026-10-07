"""The one place PRISM's own code opens a file chooser.

Every "Save…", "Save As…" and "Browse…" button calls ``choose_file``. It hands the system's dialog
the title, the start value and the filter exactly as given and returns the path, or "" when the
dialog was cancelled. It adds no extension, remembers no folder and stores nothing: what is done
with the path is the caller's.

The two static calls are looked up on the class EACH TIME the function runs, never bound to a name
when this module is imported. The suites replace both for the whole session, because a chooser
nobody answers would otherwise open for real and stall the run; a name bound at import would go on
calling whatever stood on the class at that moment.

One chooser does not come through here: the toolbar matplotlib gives a popped-out figure opens its
own, from matplotlib's code.
"""
from PySide6.QtWidgets import QFileDialog


def choose_file(parent, title: str, start: str = "", filter: str = "", *, save: bool) -> str:
    """Ask for a file to save to (``save=True``) or to open (``save=False``).

    ``parent`` is the widget the dialog belongs to, or None. ``start`` is empty or a suggested file
    name; ``filter`` is in Qt's own form, entries joined by ``;;``. Returns the chosen path, or ""
    when the dialog was cancelled."""
    ask = QFileDialog.getSaveFileName if save else QFileDialog.getOpenFileName
    path, _chosen_filter = ask(parent, title, start, filter)
    return path
