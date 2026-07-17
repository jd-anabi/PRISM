"""A small circular "?" help badge shown next to a configurable option's name.

Hover shows the description as a tooltip; clicking pins it at the cursor (so a click/touch user, or
anyone who wants to read it without holding the mouse still, gets it too). Colours come from the
palette, so it reads correctly in both light and dark themes.
"""
from PySide6.QtCore import Qt
from PySide6.QtGui import QCursor
from PySide6.QtWidgets import QHBoxLayout, QLabel, QToolButton, QToolTip, QWidget

from core.Helpers import labels as _labels

_STYLE = (
    "QToolButton { border: 1px solid palette(mid); border-radius: 8px; color: palette(mid);"
    " font-weight: bold; font-size: 10px; padding: 0px; }"
    "QToolButton:hover { border-color: palette(highlight); color: palette(highlight); }"
)


class HelpBadge(QToolButton):
    """A 16 px circled '?' whose tooltip (and click popup) is ``text``."""

    def __init__(self, text: str, parent=None):
        super().__init__(parent)
        self._text = text
        self.setText("?")
        self.setToolTip(text)
        self.setFixedSize(16, 16)
        self.setCursor(Qt.WhatsThisCursor)
        self.setFocusPolicy(Qt.NoFocus)         # a help hint must never steal tab focus from the form
        self.setStyleSheet(_STYLE)
        self.clicked.connect(self._pin)

    def _pin(self):
        QToolTip.showText(QCursor.pos(), self._text, self)


def help_label(text: str, help_text: str) -> QWidget:
    """The option name + a trailing HelpBadge, packaged as a widget usable as a QFormLayout row label."""
    holder = QWidget()
    layout = QHBoxLayout(holder)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(4)
    layout.addWidget(QLabel(_labels.pretty_gui(text)))     # HTML/Unicode rich text (Qt AutoText renders it)
    layout.addWidget(HelpBadge(help_text))
    layout.addStretch(1)
    return holder


def add_help_row(form, text: str, widget, help_text: str) -> None:
    """``form.addRow(label + badge, widget)`` -- or a plain string label when ``help_text`` is empty.

    Both branches route the label through ``labels.pretty_gui`` so config-option names render as Qt rich
    text (e.g. F0 -> F<sub>0</sub>, T_a/T -> T<sub>a</sub>/T) with zero call-site churn."""
    if help_text:
        form.addRow(help_label(text, help_text), widget)
    else:
        form.addRow(_labels.pretty_gui(text), widget)


def with_badge(widget, help_text: str) -> QWidget:
    """A widget that carries its own label (e.g. a checkbox) followed by a trailing HelpBadge, for use
    as a single spanning ``form.addRow(with_badge(cb, "…"))`` row."""
    holder = QWidget()
    layout = QHBoxLayout(holder)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(4)
    layout.addWidget(widget)
    layout.addWidget(HelpBadge(help_text))
    layout.addStretch(1)
    return holder
