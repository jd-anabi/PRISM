"""One check's page, beside the Checks list or in its place: the check's name, the sentence under it, its state line,
the result of the record that counts, its earlier runs, and its form. The foot of the Checks list holds its "Before you
start" and its button while the page is open.

The page reads nothing: the list hands it its row, its result and its earlier runs at the list's own readings, and its
form is the list's to lock while a run is live. It is made inside the list's page, out of sight until it is opened.
"""
from collections.abc import Sequence

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from ... import check_readings, design, fonts, style
from ...check_catalogue import CheckEntry
from ...widgets.form import Form
from ...widgets.message_row import MessageRow
from ...widgets.page import Sentence

#: The heading over a check's earlier runs, one line each under it.
EARLIER_HEAD = "Earlier runs"


class CheckPage(QWidget):
    """The page of the check ``entry``: ``heading``, ``sentence``, ``state_line``, the result in ``result`` (None for
    none), ``earlier`` under ``earlier_head``, and ``form``, the check's form, in sight while it holds rows."""

    def __init__(self, entry: CheckEntry, form: Form, parent=None) -> None:
        super().__init__(parent)
        self.entry, self.key, self.form = entry, entry.key, form
        self.result: "QWidget | None" = None
        self.result_source = None            # what the result was made from: a record's folder, or a run's id
        box = QVBoxLayout(self)
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(design.scaled(design.GAP["block"]))
        self.heading = style.set_class(QLabel(entry.name), cls="head")
        self.heading.setTextFormat(Qt.PlainText)
        self.heading.setFont(fonts.inter("head"))
        self.heading.setWordWrap(True)
        box.addWidget(self.heading)
        self.sentence = Sentence(entry.sentence)
        box.addWidget(self.sentence)
        self.state_line = MessageRow("later", "")
        self.state_line.setVisible(False)
        box.addWidget(self.state_line)
        self._result_box = QVBoxLayout()
        self._result_box.setContentsMargins(0, 0, 0, 0)
        box.addLayout(self._result_box)
        self.earlier_head = style.set_class(QLabel(EARLIER_HEAD), cls="small")
        self.earlier_head.setTextFormat(Qt.PlainText)
        self.earlier = style.set_class(QLabel(), cls="small")
        self.earlier.setTextFormat(Qt.PlainText)
        self.earlier.setWordWrap(True)
        for part in (self.earlier_head, self.earlier):
            part.setVisible(False)
            box.addWidget(part)
        box.addWidget(form)
        form.setVisible(bool(form.rows))

    def show_row(self, row, result: "QWidget | None", *, earlier: Sequence[str] = (), source=None) -> None:
        """Show the check's ``row``: its state line with its state's marker, ``result`` (None for none; the result
        shown before is let go unless it is the one handed again), and the lines of its ``earlier`` runs, newest first.
        ``source`` is what the result was made from."""
        if row is None:
            self.state_line.setVisible(False)
        else:
            self.state_line.set_kind(check_readings.STATE_MARKERS.get(row.state, "later"))
            self.state_line.set_sentence(row.line)
            self.state_line.setVisible(bool(row.line))
        if result is not self.result:
            if self.result is not None:
                self._result_box.removeWidget(self.result)
                self.result.hide()
                self.result.deleteLater()
            self.result = result
            if result is not None:
                self._result_box.addWidget(result)
        self.result_source = source if result is not None else None
        self.earlier.setText("\n".join(earlier))
        for part in (self.earlier_head, self.earlier):
            part.setVisible(bool(earlier))

    def fit(self, width: int) -> None:
        """The page stands ``width`` wide: its form told so."""
        self.form.fit(width)

    def stops(self) -> list:
        """What the Tab key stops at before the form, which the foot's keys take while the page is open: the result's
        own."""
        return list(self.result.stops()) if self.result is not None and hasattr(self.result, "stops") else []
