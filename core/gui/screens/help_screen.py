"""Help: what each part's settings mean, in a reference assembled from the user-facing panel module docstrings and
their per-control ``HELP`` dicts -- filling the gap where the app had no section-level "what is this for" text --
and under it the window's keys, each with what it does. ``help_text`` builds that reference."""
import html
import importlib

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFrame, QLabel, QScrollArea, QTextBrowser, QVBoxLayout, QWidget

from .. import design, fonts, keys, style
from ..design import HELP_VIEWER_MIN_H
from ..reported import reported
from ..widgets.rule_table import Column, RuleTable, TextCell

# (module path, human title) for the five user-facing sections, in display order. Each module opens with
# a "<MODE> mode: ..." docstring line and (most) carry a per-control HELP dict.
_HELP_SECTIONS = (
    ("core.gui.panels.reduction_panel", "Reduction Map"),
    ("core.gui.panels.fdt_panel", "FDT Analysis"),
    ("core.gui.panels.crossval_panel", "Cross-validation (sweep study)"),
    ("core.gui.panels.inference_tabs", "Parameter Inference"),
    ("core.gui.panels.simulate_panel", "Simulate"),
)


def _first_paragraph(doc: str) -> str:
    """The first paragraph of a docstring (up to the first blank line) -- the user-facing summary."""
    out = []
    for line in doc.strip().splitlines():
        if not line.strip():
            break
        out.append(line.strip())
    return " ".join(out)


def help_text() -> str:
    """The help reference as rich text: for each user-facing section, its title, the first paragraph of its
    module's docstring and its per-control ``HELP`` entries. A module that cannot be imported is left out."""
    parts = []
    for modpath, title in _HELP_SECTIONS:
        try:
            mod = importlib.import_module(modpath)     # already imported by MainWindow -> cache hit
        except Exception:                              # noqa: BLE001 -- Help is best-effort
            continue
        parts.append(f"<h3>{html.escape(title)}</h3>")
        summary = _first_paragraph(mod.__doc__ or "")
        if summary:
            parts.append(f"<p>{html.escape(summary)}</p>")
        help_dict = getattr(mod, "HELP", None)
        if isinstance(help_dict, dict) and help_dict:
            items = "".join(
                f"<li><b>{html.escape(str(k))}</b>: {html.escape(str(v))}</li>"
                for k, v in help_dict.items())
            parts.append(f"<ul>{items}</ul>")
    return "".join(parts)


class HelpScreen(QWidget):
    """Help: a heading over the help text, which lays itself out, scrolls itself and can be selected, and under it
    the window's keys, a table of each key or run of keys and what it does, as the key map lists them. The text
    takes the room the table leaves; in a window too low for the text's least height and the table, the two
    scroll together under the heading."""

    def __init__(self, parent=None):
        super().__init__(parent)
        heading = QLabel("Help")
        heading.setProperty("type", "heading")
        self.viewer = QTextBrowser()
        self.viewer.setHtml(help_text())
        self.viewer.setOpenExternalLinks(True)
        self.viewer.setMinimumHeight(HELP_VIEWER_MIN_H)
        keys_heading = QLabel("Keys")
        keys_heading.setTextFormat(Qt.PlainText)
        style.set_class(keys_heading, cls="head")             # what the sheet's rule for a heading matches
        # The font is given here, when the label is built, and nowhere later: a font handed to a widget after a
        # sheet has dressed it is dropped at the next change of sheet.
        keys_heading.setFont(fonts.inter("head"))
        keys_heading.setFixedHeight(design.scaled(design.TYPE["head"][1]))
        self.keys_table = RuleTable([Column("Key", "fit"), Column("What it does")],
                                    [[TextCell(key), TextCell(does, wrap=True)] for key, does in keys.KEYS],
                                    selectable=False)
        body = QWidget()
        self._lines = QVBoxLayout(body)
        self._lines.setContentsMargins(0, 0, 0, 0)
        self._lines.setSpacing(0)
        self._lines.addWidget(self.viewer, 1)
        self._lines.addSpacing(design.scaled(design.GAP["block"]))
        self._lines.addWidget(keys_heading)
        self._lines.addSpacing(design.scaled(design.GAP["head_table"]))
        self._lines.addWidget(self.keys_table)
        self.middle = QScrollArea()
        # The frame is switched off here, at build: the sheet gives a line to a scrolling area that still has the
        # frame it is born with.
        self.middle.setFrameShape(QFrame.NoFrame)
        self.middle.setWidgetResizable(True)
        self.middle.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.middle.setFocusPolicy(Qt.NoFocus)                # a stop for the Tab key only while it scrolls
        self.middle.setProperty("ring", "inset")              # its focus ring is drawn just inside its edge
        self.middle.setWidget(body)
        # Handing the body over switches on its ground, and the view fills itself with the window's colour: both
        # are switched off again, so that the page's own ground shows through.
        body.setAutoFillBackground(False)
        self.middle.viewport().setAutoFillBackground(False)
        # A bound method, never a function that holds the screen: connected to the area's own bar.
        self.middle.verticalScrollBar().rangeChanged.connect(self._range_changed)
        layout = QVBoxLayout(self)
        layout.addWidget(heading)
        layout.addWidget(self.middle, 1)

    @reported()
    def _range_changed(self, low, high):
        """Hears the middle's scroll bar. While there is something to scroll, the middle is a stop for the Tab key, so
        that the keyboard can scroll it, and what it holds keeps the room of the middle's focus ring on every side,
        so that the ring crosses none of it, the bar a gap beside it too."""
        scrolls = high > low
        room = design.SIZE["focus_ring"] + design.SIZE["focus_gap"] if scrolls else 0
        self._lines.setContentsMargins(room, room, room + (design.SIZE["scroll_gap"] if scrolls else 0), room)
        self.middle.setFocusPolicy(Qt.TabFocus if scrolls else Qt.NoFocus)

    def first_focus(self) -> QWidget:
        """The control the keyboard starts at: the help text, which the keys scroll."""
        return self.viewer
