"""A page's form: its rows, "Rarely changed", and the name and note of the record a run makes.

A row is a label, a field, a unit and where the value came from, read left to right, with one line under the field. A
page adds a row by its field key, so that one table names it, and keeps it in ``Form.rows``, which is how a refusal
finds its row and how a plan fills it. What a page reads of a row is what the field reads as, never its text: a grouped
number is rewritten as it is typed, so its text is heard twice, and a listener reads the field.

The fold holds the settings changed least often. It is shut at every launch, with their values in one line under it;
a folded field in error opens it, and it cannot be shut again until the error has gone. A row a page hides, because its
model does not use it, is not on the page: the fold neither counts it nor names it, and its error opens nothing.

The name is checked 300 ms after the last keystroke and again at the press, since that check reads the store's
records of its kind; never while a run is live, so that an edit waiting as a run starts is read when it ends. The note
is checked at every keystroke and reads nothing.
"""
import re

from PySide6.QtCore import QEvent, QSize, Qt, QTimer, Signal
from PySide6.QtWidgets import (QCheckBox, QComboBox, QGridLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QSizePolicy,
                               QVBoxLayout, QWidget)

from core.artifacts import default_store
from core.Helpers import labels
from core.refusals import Refusal, require_note

from .. import design, fields, fonts, style
from ..reported import reported
from ..run.controller import controller
from ..step_facts import Fact
from . import labeled_inputs
from .choice import Choice
from .help_badge import HelpBadge
from .labeled_inputs import _NumberField
from .marker import Marker, painted_icon
from .page import _BREAK, _BREAK_AFTER, Columns, Sentence
from .ruled import Ruled, SectionHead

FOLD_WORDS = "Rarely changed ({count})"
FOLD_NOT_USUAL = "Rarely changed ({count}, {m} not usual)"
FOLD_CHANGED = "Rarely changed ({count}, {n} changed)"
FOLD_NOT_USUAL_CHANGED = "Rarely changed ({count}, {m} not usual, {n} changed)"
MAIN_HEAD = "Main settings"
#: How long after the last keystroke a name is checked, in milliseconds.
READ_AFTER_MS = 300
NAME_FREE_TEXT = "Fine. This name is free."
TYPE_A_NAME_TEXT = "Type a name."
NAME_BLOCKS = "the name cannot be used"
NEEDS_A_NAME = "a name is needed"
NOTE_BLOCKS = "the note cannot be used"
#: What a row of a kind the form does not know says of a field that reads as nothing.
BLANK_WORDS = "blank"
#: A word this long in an origin, a record's name, may break: no origin column at the window's least width is narrower.
_LONG_WORD = re.compile(r"\S{17,}")
_PIECE = re.compile(rf"[^\s{_BREAK}]{{16}}(?=[^\s{_BREAK}])")


def _may_wrap(words: str) -> str:
    """``words`` with places to break inside a long word: after a slash, an underscore or a full stop, and in what is still
    too long between them."""
    def broken(word) -> str:
        return _PIECE.sub(lambda piece: piece.group() + _BREAK, _BREAK_AFTER.sub(_BREAK, word.group()))

    return _LONG_WORD.sub(broken, words)


def _column(*parts) -> QVBoxLayout:
    """A column with no margin and no gap of its own, holding ``parts``."""
    box = QVBoxLayout()
    box.setContentsMargins(0, 0, 0, 0)
    box.setSpacing(0)
    for part in parts:
        box.addWidget(part)
    return box


def _grid(label_w: int) -> QGridLayout:
    """A form's grid: a label column of the look's width, and the rest for what a row holds."""
    grid = QGridLayout()
    grid.setContentsMargins(0, design.scaled(design.SIZE["row_gap"]), 0, 0)
    grid.setHorizontalSpacing(design.scaled(design.SIZE["row_gap"]))
    grid.setVerticalSpacing(design.scaled(design.SIZE["row_gap"]))
    grid.setColumnMinimumWidth(0, design.scaled(label_w))
    grid.setColumnStretch(1, 1)
    return grid


class TickedField(QWidget):
    """A field beside a tick box that stands for a value of its own: ``TickedField(IntField("0"), "Automatic", 0)``.
    Ticked, the field is greyed, keeps what it held, and the value is ``ticked``; the box is ticked at the start when
    the field reads as that value. "No limit" and a blank box both read None from ``value_or_none``, so a caller that
    must tell them apart reads ``ticked_box`` first. It hands the keyboard to neither by a focus proxy: a greyed field
    would take the keyboard from it, and a widget with one is a stop of the Tab key at its own place in the order, which
    the order of a page cannot move. A row says which of the two takes the keyboard."""

    changed = Signal()

    def __init__(self, field: QWidget, words: str, ticked: object) -> None:
        super().__init__()
        self.field, self.stands_for = field, ticked
        self.ticked_box = QCheckBox(words)
        self._row = QHBoxLayout(self)
        self._row.setContentsMargins(0, 0, 0, 0)
        self._row.setSpacing(design.scaled(design.SIZE["row_gap"]))
        self._row.addWidget(field)
        self._row.addWidget(self.ticked_box)
        self._row.addStretch(1)
        self.ticked_box.setChecked(field.value_or_none() == ticked)
        field.setEnabled(not self.ticked_box.isChecked())
        self.ticked_box.toggled.connect(self._ticked_changed)
        field.textChanged.connect(self.changed)

    def set_unit(self, unit: QWidget) -> None:
        """Stand the row's unit where it belongs: after the field it counts, before the box."""
        self._row.insertWidget(1, unit)

    def _ticked_changed(self, on: bool) -> None:
        self.field.setEnabled(not on)
        self.changed.emit()

    def is_ticked(self) -> bool:
        return self.ticked_box.isChecked()

    def value_or_none(self):
        """The value the box stands for while it is ticked, else what the field reads as."""
        return self.stands_for if self.is_ticked() else self.field.value_or_none()

    def set_value(self, value) -> None:
        """Set the value: the box ticked for the value it stands for, else the field told ``value``."""
        self.ticked_box.setChecked(value == self.stands_for)
        if value != self.stands_for:
            self.field.setText("" if value is None else str(value))

    def words(self) -> str:
        """The value as it is shown: the box's words while it is ticked, else the field's text."""
        return self.ticked_box.text() if self.is_ticked() else self.field.text().strip()


def _number_of(widget: QWidget) -> "_NumberField | None":
    """The number field ``widget`` is or holds, or None."""
    field = widget.field if isinstance(widget, TickedField) else widget
    return field if isinstance(field, _NumberField) else None


def _dress(widget: QWidget) -> None:
    """A number field as a form stands it: right aligned, its digits of equal width, the look's width. The font is given
    here, as the field is built into its form: a font handed to a widget a sheet has dressed is dropped at the next
    change of sheet."""
    field = _number_of(widget)
    if field is not None:
        field.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        field.setFont(fonts.inter("body", tabular=True))
        field.setFixedWidth(design.scaled(design.SIZE["number_w"]))


class _Line(QWidget):
    """The one line under a field: its marker and a sentence in the lettering face, hidden while it says nothing."""

    def __init__(self) -> None:
        super().__init__()
        self.kind, self.words = "", ""
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)                   # a widget's own layout is born with the style's margins
        row.setSpacing(design.scaled(design.GAP["marker_text"]))
        beside = QVBoxLayout()
        beside.setContentsMargins(0, design.scaled(design.GAP["marker_drop"]), 0, 0)
        self.marker = Marker("fine")
        beside.addWidget(self.marker)
        beside.addStretch(1)
        row.addLayout(beside)
        self.label = QLabel()
        self.label.setTextFormat(Qt.PlainText)
        self.label.setWordWrap(True)
        row.addWidget(self.label, 1)
        self.setVisible(False)

    def say(self, kind: str, words: str) -> None:
        self.kind, self.words = kind, words
        if kind:
            self.marker.set_kind(kind)
        self.label.setText(words)

    def says(self) -> bool:
        return bool(self.kind and self.words)


class Row:
    """One setting of a form: ``key``, ``widget``, what the form asks of it. ``value`` is what the field reads as: a
    number or None for a number field, the item's data or else its words for a list, whether it is ticked for a tick
    box, the text for a text field or a value that is only shown. ``usual`` is the value the field opens at. ``kept``
    says a page put the value back from what it remembers, and it has not been touched since: any edit ends it."""

    def __init__(self, key: str, widget: QWidget | None, label: str, usual, *, unit: str = "", edited=None, marked=None,
                 help_text: str = "") -> None:
        self.key, self.widget = key, widget
        self._label, self._usual, self._unit = label, usual, unit
        self._on_edit, self._on_mark = edited, marked
        self._shown, self._marked, self._blocks = True, False, False
        self.kept = False
        self.label = QLabel(labels.pretty_gui(label))
        self.label.setWordWrap(True)
        self.label.setMinimumHeight(design.scaled(design.SIZE["field_h"]))
        self.badge = HelpBadge(help_text) if help_text else None
        self.origin = style.set_class(QLabel(), cls="small")
        self.origin.setTextFormat(Qt.PlainText)
        self.origin.setWordWrap(True)
        self.unit = QLabel(unit)
        self.unit.setTextFormat(Qt.PlainText)
        self.unit.setVisible(bool(unit))
        self.line = _Line()
        self.under: list = []
        self._heard: list = []
        self._parts = [self.label, self.origin]
        self._in_sight: list = []
        if widget is not None:
            self.label.setBuddy(widget.field if isinstance(widget, TickedField) else widget)
            _dress(widget)
            self._listen(widget)

    def _listen(self, widget: QWidget) -> None:
        """Hear every change of the field, however its kind says one."""
        number = _number_of(widget)
        if number is not None:
            number.error_changed.connect(self._field_error)
        said = (widget.currentIndexChanged if isinstance(widget, QComboBox) else
                widget.toggled if isinstance(widget, QCheckBox) else
                widget.textChanged if isinstance(widget, QLineEdit) else getattr(widget, "changed", None))
        if said is not None:
            said.connect(self._edited)

    def _field_error(self, sentence: str) -> None:
        """The field put itself in error, or a caller did, or the error was taken back: said under it."""
        if sentence:
            self._say("err", sentence)
        elif self.line.kind == "err":
            self._say("", "")

    def _say(self, kind: str, sentence: str) -> None:
        """Say the line, while the row stands on the page, and tell the form: what stands under a field is no edit, so a
        page may say it as it draws its facts."""
        self.line.say(kind, sentence)
        self.line.setVisible(self._shown and self.line.says())
        if self._on_mark is not None:
            self._on_mark()

    def _edited(self) -> None:
        """A refusal's sentence stands under the field until the field is edited; a value put back is no longer that."""
        self.kept = False
        if self._marked:
            self._marked = self._blocks = False
            self._set_line("", "")
        if self._on_edit is not None:
            self._on_edit()

    def hear(self, widget: QWidget) -> None:
        """Hear ``widget``'s edits as the field's own: a field under the row that the row's value stands on. A box of
        text so heard is outlined with the row in error."""
        self._listen(widget)
        self._heard.append(widget)

    # ── what a page reads and sets ──
    def label_text(self) -> str:
        """The row's label, as the table writes it."""
        return self._label

    def value(self):
        """What the field reads as."""
        widget = self.widget
        if widget is None:
            return self._label
        read = getattr(widget, "value_or_none", None)
        if callable(read):
            return read()
        if isinstance(widget, QComboBox):
            data = widget.currentData()
            return widget.currentText() if data is None else data
        if isinstance(widget, QCheckBox):
            return widget.isChecked()
        return widget.text()

    def set_value(self, value) -> None:
        """Show ``value``, as a plan fills a form: nothing is judged here. In a list, the item that holds it; a list
        that holds no such item stays as it stands."""
        widget = self.widget
        if widget is None:
            self._label = str(value)
            self.label.setText(self._label)
        elif isinstance(widget, TickedField):
            widget.set_value(value)
        elif isinstance(widget, QComboBox):
            found = widget.findData(value)
            found = found if found >= 0 else widget.findText(str(value))
            if found >= 0:
                widget.setCurrentIndex(found)
        elif isinstance(widget, QCheckBox):
            widget.setChecked(bool(value))
        else:
            widget.setText("" if value is None else str(value))

    def usual(self):
        """The built-in value the field opens at."""
        return self._usual

    def at_usual(self) -> bool:
        """Whether the field reads as its built-in value: as it reads, never as it is typed, so that "1e-3" is 0.001 and
        a box that reads as nothing is not a number's usual value. A blank box and a tick that stands for nothing both read
        None, so a ticked field's tick is compared first."""
        widget = self.widget
        if isinstance(widget, TickedField) and widget.is_ticked() != (self._usual == widget.stands_for):
            return False
        return self.value() == self._usual

    def words(self) -> str:
        """The value as a sentence says it, with its unit."""
        widget = self.widget
        if widget is None:
            return self._label
        if isinstance(widget, QComboBox):
            said = widget.currentText()
        elif isinstance(widget, QCheckBox):
            said = "on" if widget.isChecked() else "off"
        elif isinstance(widget, TickedField) and widget.is_ticked():
            return widget.words()
        else:
            said = widget.words() if isinstance(widget, TickedField) else widget.text().strip()
        return f"{said} {self._unit}".rstrip() if said else BLANK_WORDS

    def set_origin(self, text: str) -> None:
        """Say where the value came from, on as many lines as it takes: a record's name has no end."""
        self.origin.setText(_may_wrap(text))

    def set_line(self, kind: str, sentence: str) -> None:
        """Say ``sentence`` under the field with the marker of ``kind``; an empty one says nothing. In error the field
        is outlined too, and a number field says the sentence as its tool tip. What the field says of itself, a comma's
        sentence, stands while the field holds it: it is no page's to take away. Nor is a refusal's sentence, while it stops
        the start: a page says its lines as it draws, at every edit, and its line comes at the first drawing after the field
        is edited. Once the facts have been read again a line with words takes its place, and an empty one still leaves
        it."""
        if self._marked and (self._blocks or not sentence):
            return
        self._marked = False
        self._set_line(kind, sentence)

    def _set_line(self, kind: str, sentence: str) -> None:
        number = None if self.widget is None else _number_of(self.widget)
        if number is not None:
            number.set_error((sentence if kind == "err" else "") or number.own_error())
            if number.error():
                kind, sentence = "err", number.error()
        elif isinstance(self.widget, QLineEdit):
            style.set_class(self.widget, state="error" if kind == "err" else "")
        for heard in self._heard:
            if isinstance(heard, QLineEdit):
                style.set_class(heard, state="error" if kind == "err" else "")
        self._say(kind, sentence)

    def mark(self, sentence: str) -> None:
        """A refusal of this row: its sentence under the field, stopping the start, until the field is edited."""
        self._set_line("err", sentence)
        self._marked = self._blocks = True

    def refused(self) -> str:
        """The sentence of the refusal that stops the start here, "" for none: a row off the page stops nothing."""
        return self.line.words if self._blocks and self._shown else ""

    def ease(self) -> None:
        """The facts were read again since the refusal: it stops the start no longer, since what cures it may stand in
        another field, or on another page, and the press asks the engine again. Its sentence stands until the field is
        edited, no longer as an error, for a marker never disagrees with the button."""
        if self._blocks:
            self._blocks = False
            self._set_line("warn", self.line.words)

    def in_error(self) -> bool:
        """Whether the row stands on the page in error."""
        return self._shown and self.line.kind == "err"

    def reading(self) -> "str | None":
        """What is wrong with the number the field holds, in a sentence, or None; a row off the page holds nothing wrong."""
        number = None if self.widget is None else _number_of(self.widget)
        if number is None or not self._shown or (isinstance(self.widget, TickedField) and self.widget.is_ticked()):
            return None
        return labeled_inputs.reading(number, self._label)

    def stops(self) -> list:
        """The controls the Tab key stops at in this row, in their order, what stands under the field among them."""
        stops = []
        for part in (self.widget, *self.under):
            if part is not None:
                stops += [stop for stop in ([part] if part.focusPolicy() & Qt.TabFocus else part.findChildren(QWidget))
                          if stop.focusPolicy() & Qt.TabFocus]
        return stops

    def focus_target(self) -> "QWidget | None":
        """The control that takes the keyboard for this row, or None for a row with no control that can."""
        widget = self.widget
        if isinstance(widget, TickedField):
            widget = widget.ticked_box if widget.is_ticked() else widget.field
        return widget if widget is not None and widget.isEnabled() else None

    # ── on the page ──
    def is_shown(self) -> bool:
        return self._shown

    def set_shown(self, on: bool) -> None:
        """Stand on the page, or not: a row its model does not use is hidden whole, and what of it stood in sight comes
        back with it."""
        on = bool(on)
        if on == self._shown:
            return
        self._shown = on
        if on:
            for part in self._in_sight:
                part.setVisible(True)
            self.line.setVisible(self.line.says())
        else:
            parts = [*self._parts, self.unit, *self.under]
            parts += [part for part in (self.badge, self.widget) if part is not None]
            self._in_sight = [part for part in parts if not part.isHidden()]
            for part in (*parts, self.line):
                part.setVisible(False)
        if self._on_mark is not None:
            self._on_mark()

    def stand_in(self, grid: QGridLayout, at: int) -> None:
        """Take the row ``at`` of ``grid``: the label and its badge in the first column, and in the second the field with
        its unit and origin, the line under them. A value that is only shown is a sentence, with no column of its own,
        and its origin stands after it."""
        left = QHBoxLayout()
        left.setContentsMargins(0, 0, 0, 0)
        left.setSpacing(design.scaled(design.GAP["marker_text"]))
        left.addWidget(self.label, 1 if self.badge is None else 0)
        if self.badge is not None:
            left.addWidget(self.badge, 0, Qt.AlignVCenter)
            left.addStretch(1)
        beside = QHBoxLayout()
        beside.setContentsMargins(0, 0, 0, 0)
        beside.setSpacing(design.scaled(design.SIZE["row_gap"]))
        if self.widget is None:
            beside.addLayout(left)
            beside.setStretch(0, 1)
        else:
            grid.addLayout(left, at, 0, Qt.AlignTop)
            beside.addWidget(self.widget)
        if isinstance(self.widget, TickedField):
            self.widget.set_unit(self.unit)
        else:
            beside.addWidget(self.unit)
        beside.addWidget(self.origin)
        if self.widget is not None:
            beside.addStretch(1)
        self.cell = QVBoxLayout()
        self.cell.setContentsMargins(0, 0, 0, 0)
        self.cell.setSpacing(design.scaled(design.GAP["field_message"]))
        self.cell.addLayout(beside)
        self.cell.addWidget(self.line)
        if self.widget is None:
            grid.addLayout(self.cell, at, 0, 1, 2)
        else:
            grid.addLayout(self.cell, at, 1)

    def add_under(self, widget: QWidget) -> None:
        """Hold ``widget`` under the field, with the row: an explanation."""
        self.under.append(widget)
        self.cell.addWidget(widget)
        if not self._shown:
            self._in_sight.append(widget)
            widget.setVisible(False)


class Fold(Ruled):
    """The rarely changed settings: a button, the line of their values while it is shut, and the rows while it is open."""

    def __init__(self) -> None:
        above, between, below = design.GAP["fold"]
        super().__init__(top=design.SIZE["hair"], bottom=design.SIZE["hair"], colour="rule_light",
                         gap_top=design.scaled(above), gap_bottom=design.scaled(below))
        self._rows: list = []
        self._changed_rows: tuple = ()
        self._open = False
        self.button = style.set_class(QPushButton(), cls="fold")
        self.button.setIcon(painted_icon("chev_right"))
        self.button.setIconSize(QSize(*(design.scaled(n) for n in design.GAP["fold_icon"])))
        self.button.setFixedHeight(design.scaled(design.SIZE["line_h"]))
        self.button.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        self.button.clicked.connect(self._pressed)
        self.box.addWidget(self.button, 0, Qt.AlignLeft)     # as wide as its words: so is its ring
        self.summary = style.set_class(QLabel(), cls="small")
        self.summary.setTextFormat(Qt.PlainText)
        self.summary.setWordWrap(True)
        self.summary.setContentsMargins(design.scaled(design.GAP["fold_indent"]), design.scaled(between), 0, 0)
        self.box.addWidget(self.summary)
        self.body = QWidget()
        self.grid = _grid(design.SIZE["label_w"][0])
        self.body.setLayout(self.grid)
        self.body.setVisible(False)
        self.box.addWidget(self.body)
        self.setVisible(False)

    def is_open(self) -> bool:
        return self._open

    def rows(self) -> list:
        return list(self._rows)

    def _in_error(self) -> bool:
        return any(row.in_error() for row in self._rows)

    def _pressed(self) -> None:
        self.set_open(not self._open)

    def set_open(self, on: bool) -> None:
        """Open or shut the fold; one that holds a field in error stays open."""
        self._open = bool(on) or self._in_error()
        self.button.setIcon(painted_icon("chev_down" if self._open else "chev_right"))
        self.body.setVisible(self._open)
        self.summary.setVisible(not self._open)

    def add(self, row: Row) -> None:
        row.stand_in(self.grid, len(self._rows))
        self._rows.append(row)

    def stops(self) -> list:
        """The button, then what the fold holds, in the order the Tab key takes them."""
        return [self.button, *(stop for row in self._rows for stop in row.stops())]

    def set_changed(self, rows) -> None:
        """The rows of the form whose choice differs from what the step holds: those the fold holds are counted on its
        button. The fold opens for none of them, so one a person shut stays shut."""
        self._changed_rows = tuple(rows)
        self.judge()

    def judge(self) -> None:
        """What the button and the line of values say of the rows that stand on the page, those not usual first, and how
        many of them are changed and not applied; a field in error opens the fold."""
        rows = [row for row in self._rows if row.is_shown() and row.widget is not None]
        apart = [row for row in rows if not row.at_usual()]
        changed = len([row for row in rows if row in self._changed_rows])
        count = "1 setting" if len(rows) == 1 else f"{len(rows)} settings"
        words = FOLD_NOT_USUAL_CHANGED if apart and changed else FOLD_NOT_USUAL if apart else (
            FOLD_CHANGED if changed else FOLD_WORDS)
        self.button.setText(words.format(count=count, m=len(apart), n=changed))
        said = apart + [row for row in rows if row not in apart]
        self.summary.setText("; ".join(f"{row.label_text()} {row.words()}" for row in said))
        self.setVisible(any(row.is_shown() for row in self._rows))
        if self._in_error() and not self._open:
            self.set_open(True)


class NameAndNote(QWidget):
    """The block of a run that makes a record: "Name" and "Note". A required name left blank blocks with its own words;
    a name the store refuses is outlined, with the store's sentence under it, and blocks; a note its rule refuses does
    the same. ``changed`` says the block's verdict may have moved. Nothing here is emptied but by ``clear``."""

    changed = Signal()

    def __init__(self, kind: str, *, required: bool, store=None, read_after_ms: int = READ_AFTER_MS) -> None:
        super().__init__()
        self.kind, self.required, self._store = kind, required, store
        self._refused = self._free = self._note_refused = ""
        self.name, self.note = QLineEdit(), QLineEdit()
        self.hint = style.set_class(QLabel(), cls="quiet")
        self.hint.setTextFormat(Qt.PlainText)
        self.hint.setWordWrap(True)
        self.hint.setVisible(False)
        grid = QGridLayout(self)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(design.scaled(design.SIZE["row_gap"]))
        grid.setVerticalSpacing(design.scaled(design.SIZE["row_gap"]))
        grid.setColumnMinimumWidth(0, design.scaled(design.SIZE["label_w"][1]))
        grid.setColumnStretch(1, 1)
        self.rows = {"name": Row("name", self.name, "Name", None), "note": Row("note", self.note, "Note", None)}
        for at, row in enumerate(self.rows.values()):
            row.stand_in(grid, at)
        self.rows["name"].add_under(self.hint)
        self.read_timer = QTimer(self)
        self.read_timer.setSingleShot(True)
        self.read_timer.setInterval(read_after_ms)
        self.read_timer.timeout.connect(self.check_now)
        self.name.textChanged.connect(self._name_edited)
        self.note.textChanged.connect(self._note_edited)

    def name_value(self) -> str:
        """The name as it is sent: the text, stripped."""
        return self.name.text().strip()

    def note_value(self) -> str:
        """The note as it is sent: what its rule hands back; "" for a note it refuses."""
        return "" if self._note_refused else self.note.text().strip()

    def name_fact(self) -> "Fact | None":
        """What stops the start about the name, as a fact; None for nothing."""
        if self._refused:
            return Fact("err", self._refused, NAME_BLOCKS)
        if self.required and not self.name_value():
            return Fact("err", TYPE_A_NAME_TEXT, NEEDS_A_NAME)
        return None

    def note_fact(self) -> "Fact | None":
        """What stops the start about the note, as a fact; None for nothing."""
        return Fact("err", self._note_refused, NOTE_BLOCKS) if self._note_refused else None

    def _name_edited(self) -> None:
        self._free = ""
        if self.name_value():
            self.read_timer.start()
        else:
            self.read_timer.stop()
            self._refused = ""
        self._say_name()

    def check_now(self) -> None:
        """Read the name against the store, now, unless a run is live: what is waiting then is read by the page when the
        run ends, through this. A store that cannot be read says nothing: the stage reads it again at its entry."""
        self.read_timer.stop()
        name = self.name_value()
        if controller().is_live() or not name:
            return
        self._refused, self._free = "", NAME_FREE_TEXT
        try:
            (self._store or default_store()).assert_name_free(self.kind, name)
        except Refusal as refused:
            self._refused, self._free = refused.message, ""
        except OSError:
            self._free = ""
        self._say_name()

    def _say_name(self) -> None:
        row = self.rows["name"]
        if self._refused:
            row.set_line("err", self._refused)
        else:
            row.set_line("fine" if self._free else "", self._free)
        self.changed.emit()

    def _note_edited(self) -> None:
        """The note's rule, at every keystroke; it reads nothing."""
        try:
            require_note("note", self.note.text())
            self._note_refused = ""
        except Refusal as refused:
            self._note_refused = refused.message
        self.rows["note"].set_line("err" if self._note_refused else "", self._note_refused)
        self.changed.emit()

    def clear(self) -> None:
        """Both fields emptied and the hint dropped, nothing read: only after a run that made its record."""
        self.name.clear()
        self.note.clear()
        self.set_hint("")

    def set_hint(self, sentence: str) -> None:
        """A quiet line under "Name"; an empty one takes no room."""
        self.hint.setText(sentence)
        self.hint.setVisible(bool(sentence))


class Form(QWidget):
    """The form of a page: "Main settings" with the fold under them, and beside them, while the page is wide enough, the
    block for this run. ``rows`` holds main, folded and block rows alike, by field key. ``edited`` says a field changed."""

    edited = Signal()
    joined = Signal()                        # a row joined the form

    def __init__(self) -> None:
        super().__init__()
        self.rows: dict[str, Row] = {}
        self._width: "int | None" = None         # the width the form was last fitted to
        self.head = SectionHead(MAIN_HEAD)
        self._grid = _grid(design.SIZE["label_w"][0])
        self._main_rows, self._side_rows = [], []
        self._side_count = 0
        self.fold = Fold()
        left = QWidget()
        column = _column(self.head)
        column.addLayout(self._grid)
        column.addSpacing(design.scaled(design.GAP["form_to_fold"]))
        column.addWidget(self.fold)
        column.addStretch(1)
        left.setLayout(column)
        self._left = left
        left.installEventFilter(self)            # a row's words beside its field changed: its list held again
        self.side_head = SectionHead("")
        self._side_grid = _grid(design.SIZE["label_w"][1])
        self.side = QWidget()
        column = _column(self.side_head)
        column.addLayout(self._side_grid)
        column.addStretch(1)
        self.side.setLayout(column)
        self.side.setVisible(False)
        self.columns = Columns(left, self.side, stretch=design.GAP["setup_columns"])
        self.setLayout(_column(self.columns))

    def set_side_heading(self, words: str) -> None:
        """The heading of the block for this run."""
        self.side_head.set_text(words)

    def _changed(self) -> None:
        self.fold.judge()
        self.edited.emit()

    def add(self, field_key: str, widget: QWidget, unit: str = "", *, usual=None, folded: bool = False,
            help_text: str = "", label: "str | None" = None, side: bool = False) -> Row:
        """Add the setting ``field_key``, labelled as the table names it unless ``label`` is handed in, for a row the
        table does not name: among the main settings, in the fold, or in the block for this run."""
        row = Row(field_key, widget, fields.label(field_key) if label is None else label, usual, unit=unit,
                  edited=self._changed, marked=self.fold.judge, help_text=help_text)
        self._place(row, folded=folded, side=side)
        self.rows[field_key] = row
        self.fold.judge()
        self.joined.emit()
        return row

    def add_text(self, field_key: str, text: str, origin: str, *, folded: bool = False, side: bool = False) -> Row:
        """A value the window may not change: words and where they came from, no field."""
        row = Row(field_key, None, text, None, marked=self.fold.judge)
        row.set_origin(origin)
        self._place(row, folded=folded, side=side)
        self.rows[field_key] = row
        self.fold.judge()
        return row

    def add_sentence(self, sentence: str, *, under: str, quiet: bool = True) -> Sentence:
        """One sentence under the row ``under``, which stands and goes with it. KeyError for a row not added yet."""
        said = Sentence(sentence, quiet=quiet)
        self.rows[under].add_under(said)
        return said

    def add_name(self, block: NameAndNote) -> None:
        """Put the name and note in the block for this run: their rows are the form's."""
        self._side_grid.addWidget(block, self._side_count, 0, 1, 2)
        self._side_count += 1
        self.side.setVisible(True)
        self.rows.update(block.rows)
        self._side_rows += block.rows.values()

    def _place(self, row: Row, *, folded: bool, side: bool) -> None:
        if folded:
            self.fold.add(row)
        elif side:
            row.stand_in(self._side_grid, self._side_count)
            self._side_count += 1
            self._side_rows.append(row)
            self.side.setVisible(True)
        else:
            row.stand_in(self._grid, len(self._main_rows))
            self._main_rows.append(row)

    def fit(self, width: int) -> None:
        """Set the two columns side by side when ``width`` has room, else one above the other; and hold each list among
        the main settings that has not the room for its longest item beside its label to the room it has, so that the
        main settings stand within ``width`` at the window's least size."""
        self._width = width
        self._hold_the_lists()
        self.columns.fit(width)

    def _hold_the_lists(self) -> None:
        """Each list among the main settings held to the room its row leaves it at the width last fitted to: the label's
        column, the gaps, and the unit and the origin that stand beside it; let go where its longest item fits."""
        if self._width is None:
            return
        gap = self._grid.horizontalSpacing()
        label = self._grid.columnMinimumWidth(0)
        for row in self._main_rows:
            box = row.widget
            if not isinstance(box, Choice):
                continue
            beside = sum(part.minimumSizeHint().width() + gap for part in (row.unit, row.origin) if not part.isHidden())
            room = self._width - label - gap - beside
            box.hold_to(room if box._whole_width() > room else None)

    @reported(False)
    def eventFilter(self, watched, event):
        if event.type() == QEvent.Type.LayoutRequest and watched is getattr(self, "_left", None):
            self._hold_the_lists()
        return False

    def stops(self) -> list:
        """Every control of the form the Tab key stops at, in the order the page is read: the fold's own button before
        what it holds."""
        return [*(stop for row in self._main_rows for stop in row.stops()), *self.fold.stops(),
                *(stop for row in self._side_rows for stop in row.stops())]

    def refused(self) -> tuple:
        """The sentences of the refusals that stop the start, in the order the page reads."""
        return tuple(sentence for sentence in (row.refused() for row in self._in_order()) if sentence)

    def ease(self) -> None:
        """The facts were read again: no refusal stops the start any longer."""
        for row in self.rows.values():
            row.ease()

    def _in_order(self) -> list:
        """The rows in the order the page reads: the main settings, the fold, the block for this run."""
        return [*self._main_rows, *self.fold.rows(), *self._side_rows]

    def first_control(self) -> "QWidget | None":
        """The first setting that can take the keyboard, in the order the page reads, or None."""
        for row in self._in_order():
            target = row.focus_target()
            if target is not None and row.is_shown() and target.isVisibleTo(self):
                return target
        return None
