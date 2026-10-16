"""The rail down the window's left edge: PRISM's name, a row for each part, the six steps of parameter inference
as a chain under their part while it is open, and the run's state at its foot.

Every row is painted here from the look's table, in the colours of the theme in force when it paints, and asks
the lettering for its font when it paints and when it is measured: no row sets a font on itself, so no change of
class or of sheet can take one away. One row at a time carries the accent's tint and bar, the open step while
parameter inference is open and otherwise the open part's own row; the open part stands on the page's ground
between two hairlines. No row has a look of its own under the pointer.

The list of rows scrolls between the name and the foot when the window is too low for it, and the row of the
place in sight is brought into view. The list is one stop of the Tab key: the row that has it is the one the
arrows have moved to, or else the current one, and the focus ring stands just inside its edge.
"""
from PySide6.QtCore import QEvent, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QFont, QFontMetricsF, QPainter, QTextLayout, QTextOption
from PySide6.QtWidgets import QAbstractButton, QFrame, QLabel, QScrollArea, QSizePolicy, QVBoxLayout, QWidget

from .. import chain, design, fonts, keys, paint
from ..reported import reported

_ELLIPSIS = chr(0x2026)              # what the toolkit ends a cut line with
_LINES = ("solid", "dotted", None)   # the line that leaves a step: solid after a step that is done


def _px(name: str) -> int:
    """A length of the look's size table at the text size in force."""
    return design.scaled(design.SIZE[name])


def _tool_tip(words: str, place: keys.Place) -> str:
    """A row's tool tip: its words, then the keys that open its place."""
    return f"{words} ({', '.join(place.keys)})" if place.keys else words


def _rail_keys(row, event) -> bool:
    """The keys a row of the rail hands to the rail it stands in: True when the rail took the key."""
    holder = row.parentWidget()
    while holder is not None and not isinstance(holder, Rail):
        holder = holder.parentWidget()
    return holder is not None and holder._key(row, event)


class PartRow(QAbstractButton):
    """One part of the program in the rail: its name, set in from the rail's edge. The current row stands on
    the accent's tint with the accent's bar at its left and its name semibold; the open part's row keeps the
    semibold name while one of its steps is the current row instead."""

    def __init__(self, place: keys.Place):
        super().__init__()
        self.place = place
        self._current = self._open = self._small = False
        self.setText(place.name)
        self.setAccessibleName(place.name)
        self.setToolTip(_tool_tip(place.name, place))
        self.setFocusPolicy(Qt.NoFocus)              # the rail gives its one stop of the Tab key to one row
        self.setProperty("ring", "inset")            # the ring stands just inside the row's edge
        self.setProperty("ring_left", 0)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)     # as wide as the list, never wider
        self.setFixedHeight(_px("nav_h"))

    def set_current(self, on: bool) -> None:
        """Mark this row as the place in sight, or take the mark off."""
        self._current = bool(on)
        self.setProperty("ring_left", design.SIZE["bar"] if self._current else 0)     # the ring leaves the bar whole
        self.update()

    def _set_open(self, on: bool) -> None:
        """The part is open while one of its steps is the current row: its name stays semibold."""
        self._open = bool(on)
        self.update()

    def set_small(self, on: bool) -> None:
        """The low window's shorter row, or the usual one again."""
        self._small = bool(on)
        self.setFixedHeight(_px("nav_h_small" if self._small else "nav_h"))

    @reported(QAbstractButton.sizeHint)
    def sizeHint(self):
        return QSize(_px("rail"), _px("nav_h_small" if self._small else "nav_h"))

    def keyPressEvent(self, event):
        if not _rail_keys(self, event):
            super().keyPressEvent(event)

    @reported()
    def paintEvent(self, _event):
        painter = QPainter(self)
        try:                                 # a raise below must not leave a live painter on the row
            rect = QRectF(self.rect())
            if self._current:
                painter.fillRect(rect, design.colour("accent_tint"))
                paint.fill_px(painter, rect, design.colour("accent"), left=design.SIZE["bar"])
            font = fonts.inter("strong" if (self._current or self._open) else "body")
            painter.setFont(font)
            painter.setPen(design.colour("ink"))
            pad = _px("nav_pad")
            room = max(0.0, rect.width() - 2 * pad)
            painter.drawText(QRectF(pad, 0, room, rect.height()), Qt.AlignLeft | Qt.AlignVCenter,
                             QFontMetricsF(font).elidedText(self.place.name, Qt.ElideRight, room))
        finally:
            painter.end()


class StepRow(QAbstractButton):
    """One step of the chain: its marker, its number and name, its state in words on a second line, and the
    line that runs on to the next step. Words too long for the row are cut at their end with an ellipsis; its
    tool tip and its accessible name carry them whole. A step that has not started has its name in the second
    ink, unless it is the current row. The number is written in digits of equal width, so that every step's
    name starts at one x."""

    def __init__(self, place: keys.Place):
        super().__init__()
        self._place = place
        self._number = keys.step_number(place.key)
        self._state = None
        self._current = self._one_line = False
        self._out, self._into = None, False
        self.setText(place.name)
        self.setFocusPolicy(Qt.NoFocus)
        self.setProperty("ring", "inset")
        self.setProperty("ring_left", 0)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self.setFixedHeight(_px("step_h"))
        self._name_it()

    def _name_it(self) -> None:
        """The whole state in words, as the tool tip and the accessible name give it."""
        spoken = f"Step {self._number}, {self._place.name}" if self._state is None else self._state.spoken
        self.setAccessibleName(spoken)
        self.setToolTip(_tool_tip(spoken, self._place))

    def set_state(self, state: chain.StepState | None) -> None:
        """Show where the step stands: a ``StepState`` of the chain, or None for nothing known yet."""
        self._state = state
        self._name_it()
        self.update()

    def set_current(self, on: bool) -> None:
        """Mark this step as the place in sight, or take the mark off."""
        self._current = bool(on)
        self.setProperty("ring_left", design.SIZE["bar"] if self._current else 0)
        self.update()

    def set_line(self, out: str | None, into: bool) -> None:
        """The line that leaves this step for the next, "solid", "dotted" or None (the last step has none),
        and whether the line arriving from the step above is solid; the first step has no line arriving."""
        if out not in _LINES:
            raise ValueError(f"not a line between steps: {out!r}")
        self._out, self._into = out, bool(into)
        self.update()

    def set_one_line(self, on: bool) -> None:
        """The low window's row of one line, with no state words, or the usual two lines again."""
        self._one_line = bool(on)
        self.setFixedHeight(_px("step_h_one" if self._one_line else "step_h"))
        self.update()

    @reported(QAbstractButton.sizeHint)
    def sizeHint(self):
        return QSize(_px("rail"), _px("step_h_one" if self._one_line else "step_h"))

    def keyPressEvent(self, event):
        if not _rail_keys(self, event):
            super().keyPressEvent(event)

    def _line(self, painter, style: str, top: float, bottom: float) -> None:
        """The line between two steps, down the middle of the markers, on whole device pixels: solid in the
        rule's colour, or dotted in the third ink, each dot and each gap as long as the line is wide."""
        weight = design.SIZE["rule"]
        left = design.scaled(design.STEP["line_x"]) - weight / 2
        if style == "solid":
            paint.fill_px(painter, QRectF(left, top, weight, bottom - top), design.colour("rule"), left=weight)
            return
        colour, y = design.colour("ink_3"), top
        while y < bottom:
            paint.fill_px(painter, QRectF(left, y, weight, min(weight, bottom - y)), colour, left=weight)
            y += 2 * weight

    @reported()
    def paintEvent(self, _event):
        painter = QPainter(self)
        try:                                 # a raise below must not leave a live painter on the row
            rect = QRectF(self.rect())
            width = rect.width()
            step = {name: design.scaled(value) for name, value in design.STEP.items() if not isinstance(value, tuple)}
            if self._current:
                painter.fillRect(rect, design.colour("accent_tint"))
                paint.fill_px(painter, rect, design.colour("accent"), left=design.SIZE["bar"])
            if self._number > 1:
                self._line(painter, "solid" if self._into else "dotted", 0, step["line_in"])
            if self._out is not None:
                self._line(painter, self._out, step["line_from"], rect.height())
            marker, words = ("later", "") if self._state is None else (self._state.marker, self._state.rail_words)
            x, y = (design.scaled(n) for n in design.STEP["marker_one" if self._one_line else "marker"])
            side = _px("marker")
            paint.paint_marker(painter, marker, QRectF(x, y, side, side))
            kind = "strong" if self._current else "body"
            font, digits = fonts.inter(kind), fonts.inter(kind, tabular=True)
            metrics = QFontMetricsF(font)
            painter.setFont(digits)
            top = step["name_y_one" if self._one_line else "name_y"]
            line = design.scaled(design.TYPE["body"][1])
            number = str(self._number)
            painter.setPen(design.colour("ink_2"))
            painter.drawText(QRectF(step["text_x"], top, width - step["text_x"], line),
                             Qt.AlignLeft | Qt.AlignVCenter, number)
            x = step["text_x"] + QFontMetricsF(digits).horizontalAdvance(number) + step["number_gap"]
            room = max(0.0, width - x - step["pad_right"])
            painter.setFont(font)
            painter.setPen(design.colour("ink_2" if (marker == "later" and not self._current) else "ink"))
            painter.drawText(QRectF(x, top, room, line), Qt.AlignLeft | Qt.AlignVCenter,
                             metrics.elidedText(self._place.name, Qt.ElideRight, room))
            if not self._one_line and words:
                small = fonts.inter("small")
                painter.setFont(small)
                painter.setPen(design.colour("ink" if self._current else "ink_2"))
                room = max(0.0, width - step["text_x"] - step["pad_right"])
                painter.drawText(QRectF(step["text_x"], step["words_y"], room, design.scaled(design.TYPE["small"][1])),
                                 Qt.AlignLeft | Qt.AlignVCenter,
                                 QFontMetricsF(small).elidedText(words, Qt.ElideRight, room))
        finally:
            painter.end()


def _lines(text: str, font: QFont, width: float, most: int) -> list[str]:
    """``text`` laid out in lines no wider than ``width``, broken between words (and inside a word too long for
    a line), at most ``most`` of them: what does not fit is cut at the end of the last line with an ellipsis."""
    layout = QTextLayout(text, font)
    option = QTextOption()
    option.setWrapMode(QTextOption.WrapAtWordBoundaryOrAnywhere)
    layout.setTextOption(option)
    layout.beginLayout()
    spans = []
    while True:
        line = layout.createLine()
        if not line.isValid():
            break
        line.setLineWidth(width)
        spans.append((line.textStart(), line.textLength()))
    layout.endLayout()
    lines = [text[start:start + length].rstrip() for start, length in spans]
    if len(lines) > most:
        rest = text[spans[most - 1][0]:]
        lines = lines[:most - 1] + [QFontMetricsF(font).elidedText(rest, Qt.ElideRight, width)]
    return lines


class RunFoot(QWidget):
    """The run's state at the foot of the rail: a hairline, a marker, a semibold title of one or two lines and
    up to two lines of words, each cut at its end with an ellipsis after its second line. Room is always kept
    for two lines of words, so the foot changes its height only when its title takes a second line. The words
    are laid out and written in digits of equal width, so that a clock among them keeps its lines and its width
    as it ticks. While a run is shown it is one button, which the Tab key reaches and Enter, Space and a click
    press; otherwise it is no control at all."""
    clicked = Signal()

    def __init__(self):
        super().__init__()
        self._marker, self._title, self._words, self._button = "none", "", "", False
        self.setProperty("ring", "inset")            # a button as wide as the rail: the ring just inside its edge
        self.setProperty("ring_left", 0)
        self.set_state("none", "Nothing is running", "Only one run at a time.")

    def _room(self) -> int:
        """The width the title and the words are laid out in: the rail's, less the margins and the marker."""
        left, _top, right, _bottom = (design.scaled(n) for n in design.SIZE["run_line"])
        return _px("rail") - left - _px("marker") - _px("run_line_gap") - right

    def _laid_out(self) -> tuple[list[str], list[str]]:
        room = self._room()
        title = _lines(self._title, fonts.inter("strong"), room, 2) if self._title else []
        words = _lines(self._words, fonts.inter("small", tabular=True), room, 2) if self._words else []
        return title, words

    def set_state(self, marker: str, title: str, words: str = "", *, button: bool = False) -> None:
        """Show a state: one of the look's markers, the title, the words under it, and whether the foot is a
        button that goes to the run. A marker the look does not have raises ValueError."""
        if marker not in paint.MARKERS:
            raise ValueError(f"not a marker: {marker!r}")
        self._marker, self._title, self._words, self._button = marker, title, words, bool(button)
        self.setFocusPolicy(Qt.TabFocus if self._button else Qt.NoFocus)
        if not self._button and self.window().focusWidget() is self:     # the window in front or not
            self.clearFocus()
        title_lines, word_lines = self._laid_out()
        _left, top, _right, bottom = (design.scaled(n) for n in design.SIZE["run_line"])
        title_h, words_h = design.scaled(design.TYPE["strong"][1]), design.scaled(design.TYPE["small"][1])
        self.setFixedHeight(design.SIZE["hair"] + top + max(1, len(title_lines)) * title_h + 2 * words_h + bottom)
        whole = f"{title}. {words}" if words else title
        self.setAccessibleName(whole)
        self.setToolTip(whole if any(line.endswith(_ELLIPSIS) for line in title_lines + word_lines) else "")
        self.update()

    def is_button(self) -> bool:
        """Whether the foot is a button now."""
        return self._button

    @reported(QWidget.sizeHint)
    def sizeHint(self):
        return QSize(_px("rail"), self.height())

    def keyPressEvent(self, event):
        if self._button and event.key() in (Qt.Key_Return, Qt.Key_Enter, Qt.Key_Space):
            if not event.isAutoRepeat():             # a key held down presses the foot once
                self.clicked.emit()
            return
        super().keyPressEvent(event)

    @reported(QWidget.event)
    def event(self, event):
        kind = event.type()
        if self._button and kind in (QEvent.MouseButtonPress, QEvent.MouseButtonRelease) and (
                event.button() == Qt.LeftButton):
            if kind == QEvent.MouseButtonRelease and self.rect().contains(event.position().toPoint()):
                self.clicked.emit()
            return True
        return super().event(event)

    @reported()
    def paintEvent(self, _event):
        painter = QPainter(self)
        try:                                 # a raise below must not leave a live painter on the foot
            paint.fill_px(painter, QRectF(self.rect()), design.colour("rule_light"), top=design.SIZE["hair"])
            left, top, _right, _bottom = (design.scaled(n) for n in design.SIZE["run_line"])
            top += design.SIZE["hair"]
            side = _px("marker")
            paint.paint_marker(painter, self._marker, QRectF(left, top + _px("run_marker_drop"), side, side))
            x, room = left + side + _px("run_line_gap"), self._room()
            title_lines, word_lines = self._laid_out()
            for kind, ink, lines in (("strong", "ink", title_lines), ("small", "ink_2", word_lines)):
                painter.setFont(fonts.inter(kind, tabular=kind == "small"))
                painter.setPen(design.colour(ink))
                line = design.scaled(design.TYPE[kind][1])
                for text in lines:
                    painter.drawText(QRectF(x, top, room, line), Qt.AlignLeft | Qt.AlignVCenter, text)
                    top += line
        finally:
            painter.end()


class _List(QWidget):
    """What the rail's list scrolls: its rows on the rail's grey, the rail's right edge as a hairline, and the
    open part's rows on the page's ground between two hairlines, joined to the page. Where the open part stands
    last, on the foot, the foot's own hairline is the lower of the two."""

    def __init__(self):
        super().__init__()
        self.open_block = None

    @reported()
    def paintEvent(self, _event):
        painter = QPainter(self)
        try:                                 # a raise below must not leave a live painter on the list
            rect = QRectF(self.rect())
            hair, quiet = design.SIZE["hair"], design.colour("rule_light")
            painter.fillRect(rect, design.colour("margin"))
            paint.fill_px(painter, rect, quiet, right=hair)
            block = self.open_block
            if block is not None:
                band = QRectF(0, block.y(), rect.width(), block.height())
                painter.fillRect(band, design.colour("paper"))
                last = block.y() + block.height() >= self.height()
                paint.fill_px(painter, band, quiet, top=hair, bottom=0 if last else hair)
        finally:
            painter.end()


class _Scroll(QScrollArea):
    """The list's scrolling area. Each time its height changes -- a lower window, the small layout, a foot whose
    title takes a second line -- the current row is brought into view again."""

    @reported(QScrollArea.resizeEvent)
    def resizeEvent(self, event):
        super().resizeEvent(event)                   # first: the list takes the area's new size
        rail = self.parentWidget()
        if isinstance(rail, Rail):
            rail._keep_in_view()


class Rail(QWidget):
    """The rail: PRISM's name, the list of the parts with the chain of steps under parameter inference, and
    the run's state at its foot. ``opened`` is emitted with a part's key and None, or with "inference" and a
    step's key, when a row is pressed; ``left`` when Escape is pressed in the rail, for the page to take the
    keyboard. Up, Down, Home and End move between the rows in sight, in their order; Enter and Space open the
    row that has the keyboard. While ``opened`` is said, ``opened_by_key`` tells a row opened by Enter or Space
    from one opened by a click."""
    opened = Signal(str, object)
    left = Signal()

    def __init__(self):
        super().__init__()
        self.setFixedWidth(_px("rail"))
        self._part = self._step = None
        self._small = False
        self._states = ()
        self._by_key = False                 # True only while a row that Enter or Space opened is said
        column = QVBoxLayout(self)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(0)

        self.brand = QLabel("PRISM")
        self.brand.setTextFormat(Qt.PlainText)
        # The name's font is given here, when the label is built, and nowhere later: a font handed to a widget
        # after a sheet has dressed it is dropped at the next change of sheet.
        face = fonts.inter("brand")
        face.setLetterSpacing(QFont.PercentageSpacing, design.BRAND_SPACING)
        self.brand.setFont(face)
        column.addWidget(self.brand)

        self._list = _List()
        rows = QVBoxLayout(self._list)
        rows.setContentsMargins(0, 0, 0, 0)
        rows.setSpacing(0)
        self._parts, self._steps, self._blocks = {}, {}, {}
        for place in keys.PARTS:
            if place.key == "temperature":
                rows.addSpacing(design.scaled(design.GAP["rail_rest"]))
            elif place.key == "records":
                rows.addSpacing(design.scaled(design.GAP["rail_group"]))
            elif place.key == "models":
                rows.addStretch(1)
            block = QWidget()                    # the part's row, and for parameter inference its chain
            inside = QVBoxLayout(block)
            inside.setContentsMargins(0, 0, 0, 0)
            inside.setSpacing(0)
            row = PartRow(place)
            row.clicked.connect(self._pressed)
            inside.addWidget(row)
            self._parts[place.key] = row
            self._blocks[place.key] = block
            if place.key == "inference":
                for step in keys.STEPS:
                    step_row = StepRow(step)
                    step_row.clicked.connect(self._pressed)
                    step_row.set_line(None if step is keys.STEPS[-1] else "dotted", False)
                    step_row.hide()
                    inside.addWidget(step_row)
                    self._steps[step.key] = step_row
            rows.addWidget(block)

        self._scroll = _Scroll()
        # The frame is switched off here, at build: the sheet gives a line to a scrolling area that still has
        # the frame it is born with.
        self._scroll.setFrameShape(QFrame.NoFrame)
        self._scroll.setWidgetResizable(True)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._scroll.setFocusPolicy(Qt.NoFocus)          # the list's one stop of the Tab key is a row
        self._scroll.viewport().setAutoFillBackground(False)
        self._scroll.setWidget(self._list)
        column.addWidget(self._scroll, 1)

        self.foot = RunFoot()
        column.addWidget(self.foot)
        self._roving = self._parts["home"]
        self._rove(self._roving)
        self.set_small(False)

    # what the rail shows
    def set_place(self, part: str, step: str | None = None) -> None:
        """Mark the place in sight: the part, and the step while parameter inference is open; the chain shows
        only then. A part the rail does not list marks no row, a step it does not list leaves its part's row
        marked, and a page of a part (a page of effective temperature) marks its part's row. The current row is
        brought into view."""
        self._part = part if part in self._parts else None
        self._step = step if (self._part == "inference" and step in self._steps) else None
        for row in self._steps.values():
            row.setVisible(self._part == "inference")
        for key, row in self._parts.items():
            row.set_current(key == self._part and self._step is None)
            row._set_open(key == self._part and self._step is not None)
        for key, row in self._steps.items():
            row.set_current(key == self._step)
        hair = design.SIZE["hair"]
        for key, block in self._blocks.items():
            # room for the open part's two hairlines; the lower one of the last part is the foot's own
            edge = hair if key == self._part else 0
            block.layout().setContentsMargins(0, edge, 0, 0 if key == keys.PARTS[-1].key else edge)
        self._list.open_block = self._blocks.get(self._part)
        self._list.update()
        current = self._current_row()
        if current is not None:
            self._rove(current)
        elif self._roving.isHidden():                # the stop stood on a step of the folded chain
            self._rove(self._parts["home"])
        self._fit_steps()
        if current is not None:
            self._scroll.ensureWidgetVisible(current, 0, 0)

    def set_step_states(self, states) -> None:
        """Show each step's state, a ``StepState`` of the chain for each step; the line leaving a step is solid
        once that step is done."""
        self._states = tuple(state for state in states if state.key in self._steps)
        known = {state.key: state for state in self._states}
        done = [key in known and known[key].done for key in self._steps]
        for number, (key, row) in enumerate(self._steps.items()):
            row.set_state(known.get(key))
            last = number == len(self._steps) - 1
            row.set_line(None if last else ("solid" if done[number] else "dotted"), number > 0 and done[number - 1])
        self._fit_steps()

    def set_small(self, on: bool) -> None:
        """The low window: a shorter name and shorter rows, and a step keeps its words only when it is the step
        in hand, the first that cannot start or the open one."""
        self._small = bool(on)
        left, top, right, bottom = (design.scaled(n) for n in design.SIZE["brand_small" if self._small else "brand"])
        self.brand.setContentsMargins(left, top, right, bottom)
        self.brand.setFixedHeight(top + design.scaled(design.TYPE["brand"][1]) + bottom)
        for row in self._parts.values():
            row.set_small(self._small)
        self._fit_steps()

    def _fit_steps(self) -> None:
        kept = {chain.step_in_hand(self._states), self._step,
                next((state.key for state in self._states if not state.can_start), None)}
        for key, row in self._steps.items():
            row.set_one_line(self._small and key not in kept)
        self._fit()
        self._keep_in_view()

    def _fit(self) -> None:
        """Lay the list out now and size it as its scrolling area will, so that a row can be brought into view
        before the event loop turns again. Each part's rows are laid out first: a step that takes its second line,
        or gives it up, changes the height of its part's block, which the list's own layout reads."""
        for block in self._blocks.values():
            block.layout().activate()
        self._list.layout().activate()
        view = self._scroll.viewport().size()
        self._list.resize(view.width(), max(view.height(), self._list.minimumSizeHint().height()))

    def _keep_in_view(self) -> None:
        """Bring the current row into view again, after the rows or the room for them changed."""
        current = self._current_row()
        if current is not None and not current.isHidden():
            self._scroll.ensureWidgetVisible(current, 0, 0)

    # what other parts ask of it
    def row(self, name: str):
        """The row of a part or of a step, by its key; KeyError for a key the rail does not list."""
        if name in self._parts:
            return self._parts[name]
        return self._steps[name]

    def visible_rows(self) -> list:
        """The rows in sight, top to bottom: every part, and the steps while their part is open."""
        shown = []
        for key, row in self._parts.items():
            shown.append(row)
            if key == "inference":
                shown += [step for step in self._steps.values() if not step.isHidden()]
        return shown

    def take_keyboard(self) -> None:
        """Give the keyboard to the rail, as a key does: at the current row, or the row it last had."""
        row = self._current_row() or self._roving
        self._rove(row)
        self._scroll.ensureWidgetVisible(row, 0, 0)
        row.setFocus(Qt.ShortcutFocusReason)

    def opened_by_key(self) -> bool:
        """While ``opened`` is said: True when Enter or Space in the rail opened the row, False when a click did."""
        return self._by_key

    # the keyboard inside the rail
    def _current_row(self):
        if self._step is not None:
            return self._steps[self._step]
        return self._parts.get(self._part)

    def _rove(self, row) -> None:
        """Make ``row`` the list's one stop of the Tab key."""
        for other in list(self._parts.values()) + list(self._steps.values()):
            other.setFocusPolicy(Qt.TabFocus if other is row else Qt.NoFocus)
        self._roving = row

    def _key(self, row, event) -> bool:
        """A key pressed on one of the rows: True when the rail took it. Left and Right are taken and do
        nothing: there is nothing beside a row, and the keyboard stays in the rail."""
        key = event.key()
        if key == Qt.Key_Escape:
            self.left.emit()
            return True
        if key in (Qt.Key_Return, Qt.Key_Enter, Qt.Key_Space):
            if not event.isAutoRepeat():             # a key held down opens the row once
                self._by_key = True
                try:
                    row.click()
                finally:
                    self._by_key = False
            return True
        if key in (Qt.Key_Left, Qt.Key_Right):
            return True
        if key not in (Qt.Key_Up, Qt.Key_Down, Qt.Key_Home, Qt.Key_End):
            return False
        shown = self.visible_rows()
        at = shown.index(row) if row in shown else 0
        target = shown[{Qt.Key_Up: max(0, at - 1), Qt.Key_Down: min(len(shown) - 1, at + 1),
                        Qt.Key_Home: 0, Qt.Key_End: len(shown) - 1}[key]]
        self._rove(target)
        self._scroll.ensureWidgetVisible(target, 0, 0)
        target.setFocus(Qt.TabFocusReason)
        return True

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:             # pressed on the foot, the rail's other control
            self.left.emit()
            return
        super().keyPressEvent(event)

    def _pressed(self) -> None:
        row = self.sender()
        if isinstance(row, StepRow):
            self.opened.emit("inference", row._place.key)
        elif isinstance(row, PartRow):
            self.opened.emit(row.place.key, None)

    @reported()
    def paintEvent(self, _event):
        painter = QPainter(self)
        try:                                 # a raise below must not leave a live painter on the rail
            rect = QRectF(self.rect())
            painter.fillRect(rect, design.colour("margin"))
            paint.fill_px(painter, rect, design.colour("rule_light"), right=design.SIZE["hair"])
        finally:
            painter.end()
