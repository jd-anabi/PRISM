"""A table with three ruled lines: a heavy one above, a hairline under its header, a heavy one below.

There are no lines between rows and none between columns. A cell is a small description (words, a value or
a state) and one delegate paints all three kinds; the header paints its own words and the hairline under
them, in one stroke across every column. A column is fixed, fitted to its widest cell, or takes the room
left. The table is as tall as its rows and scrolls nothing itself: the page that holds it scrolls. Every
row's height is set from its cells.

Nothing is built when this module is imported, and no colour or font is kept between two paints: each is
read from the look's table when a part is painted or measured, so a change of theme takes a repaint and
nothing more. The table puts nothing in order: a header that sorts says which column and which way, and the
page that owns the rows orders them and hands them in again.
"""
import math
from dataclasses import dataclass

from PySide6.QtCore import QPointF, QRectF, QSize, Qt
from PySide6.QtGui import QFontMetricsF, QPainter, QPen, QStandardItem, QStandardItemModel, QTextLayout, QTextOption
from PySide6.QtWidgets import (QAbstractItemView, QFrame, QHeaderView, QSizePolicy, QStyle, QStyledItemDelegate,
                               QTableView)

from .. import design, fonts, paint
from ..reported import reported
from .ruled import Ruled


@dataclass(frozen=True)
class TextCell:
    """Words in a cell, at its left. Too long for its column they are cut, or set in as many lines as they
    need when ``wrap`` is set."""
    text: str
    quiet: bool = False       # secondary lettering
    elide: str = "end"        # "end" for a note, "middle" for a name
    wrap: bool = False


@dataclass(frozen=True)
class ValueCell:
    """A value in a cell, at its right. It is never shortened with an ellipsis: in a column too narrow for
    it, its left end is not drawn."""
    figure: str               # semibold
    unit: str = ""            # plain, one space after the figure
    second: str = ""          # a second line in the small size


@dataclass(frozen=True)
class StateCell:
    """A state in a cell: its marker and, after it, the word for it. The word is never shortened with an
    ellipsis: in a column too narrow for it, its right end is not drawn."""
    kind: str                 # one of paint.MARKERS
    word: str


@dataclass(frozen=True)
class Column:
    """One column of a table: the words over it and how wide it is."""
    title: str
    width: int | str | None = None    # pixels at ordinary text size, "fit", or None for the room left
    right: bool = False               # the header's words at the right, as over a column of values


# One row to each lettered part of a table: the row of the look's type table it is set in, a weight that
# replaces that row's own, and whether its digits are of equal width, as where numbers stand under one another.
_FACES = {
    "text": ("body", None, False),
    "selected": ("strong", None, False),      # the first cell of a selected row, where it holds words
    "word": ("body", None, False),            # the word beside a state's marker
    "figure": ("strong", None, True),
    "unit": ("body", None, False),
    "second": ("small", None, True),          # a value's second line
    "title": ("small", 600, False),           # the words over a column
}


def _face(part: str):
    """The font one lettered part of a table is set in, at the text size in force. It is asked where the part
    is painted and where it is measured, so the two agree; no font is kept."""
    kind, weight, equal_digits = _FACES[part]
    return fonts.inter(kind, weight, tabular=equal_digits)


def _pads(column: int, count: int) -> tuple[int, int]:
    """The room a column keeps clear of its words, in a table of ``count`` columns: before them in the first
    column, and after them in every column. Between two columns that is the space between them; the last
    column keeps what the first keeps before its words, less the bar, so that the ring round the current row
    crosses no word at either end."""
    lead = design.scaled(design.SIZE["table_lead"]) if column == 0 else 0
    if column == count - 1:
        gap = design.scaled(design.SIZE["table_lead"]) - design.SIZE["bar"]
    else:
        gap = design.scaled(design.SIZE["table_col_gap"])
    return lead, gap


def _wrapped(text: str, font, width: float, height: int):
    """Words laid out in lines no wider than ``width``, each ``height`` high: the layout, ready to be drawn
    from the top left of its place, and the number of its lines."""
    layout = QTextLayout(text, font)
    option = QTextOption()
    option.setWrapMode(QTextOption.WordWrap)
    layout.setTextOption(option)
    layout.beginLayout()
    count = 0
    while True:
        line = layout.createLine()
        if not line.isValid():
            break
        line.setLineWidth(width)
        line.setPosition(QPointF(0, count * height + (height - line.height()) / 2))
        count += 1
    layout.endLayout()
    return layout, count


def _words(cell) -> str:
    """A cell's plain words, as assistive technology reads them."""
    if isinstance(cell, ValueCell):
        return f"{cell.figure} {cell.unit}" if cell.unit else cell.figure
    return cell.word if isinstance(cell, StateCell) else cell.text


def _checked_columns(columns) -> list:
    """The columns as a list, or ValueError when more than one of them takes the room left."""
    columns = list(columns)
    takers = sum(1 for column in columns if column.width is None)
    if takers > 1:
        raise ValueError(f"only one column of a table can take the room left: {takers} have no width")
    return columns


def _checked_rows(rows, count: int) -> list:
    """The rows as lists of cells for a table of ``count`` columns. A row of another length raises ValueError,
    a cell that is none of the three descriptions TypeError, a state with no marker of its kind ValueError."""
    rows = [list(row) for row in rows]
    for row in rows:
        if len(row) != count:
            raise ValueError(f"a row of {len(row)} cells in a table of {count} columns")
        for cell in row:
            if not isinstance(cell, (TextCell, ValueCell, StateCell)):
                raise TypeError(f"not a cell of a table: {cell!r}")
            if isinstance(cell, StateCell) and cell.kind not in paint.MARKERS:
                raise ValueError(f"not a marker: {cell.kind!r}")
    return rows


class RuleDelegate(QStyledItemDelegate):
    """Paints every cell of a ruled table from its description, and a selected row's ground with it: the
    tint, a line of the accent above and below, and the accent's bar at the left of the first cell, whose
    words are then semibold. Built on the view it serves. A cell whose painting fails is drawn by the toolkit
    itself, from the words the model keeps."""

    @reported(QStyledItemDelegate.sizeHint)
    def sizeHint(self, option, index):
        view = self.parent()
        return QSize(view._column_width(index.column()), view._cell_height(index))

    @reported(QStyledItemDelegate.paint)
    def paint(self, painter, option, index):
        painter.save()
        try:                                 # the view paints the cells after this one with the same painter
            cell = index.data(Qt.UserRole)
            rect = QRectF(option.rect)
            selected = bool(option.state & QStyle.State_Selected)
            first = index.column() == 0
            if selected:
                accent = design.colour("accent")
                painter.fillRect(rect, design.colour("accent_tint"))
                hair = design.SIZE["hair"]
                paint.fill_px(painter, rect, accent, top=hair, bottom=hair,
                              left=design.SIZE["bar"] if first else 0)
            lead, gap = _pads(index.column(), index.model().columnCount())
            pad = design.scaled(design.SIZE["table_pad"])
            height = design.scaled(design.TYPE["body"][1])
            box = rect.adjusted(lead, pad, -gap, -pad)
            line = QRectF(box.left(), box.top(), box.width(), height)
            if isinstance(cell, StateCell):
                side = design.scaled(design.SIZE["marker"])
                paint.paint_marker(painter, cell.kind, QRectF(box.left(), box.top() + (height - side) / 2, side, side))
                painter.setFont(_face("word"))
                painter.setPen(design.colour("ink"))
                painter.drawText(line.adjusted(side + design.scaled(design.SIZE["state_gap"]), 0, 0, 0),
                                 Qt.AlignLeft | Qt.AlignVCenter, cell.word)
            elif isinstance(cell, ValueCell):
                plain = _face("unit")
                unit = f" {cell.unit}" if cell.unit else ""
                painter.setPen(design.colour("ink"))
                painter.setFont(plain)
                painter.drawText(line, Qt.AlignRight | Qt.AlignVCenter, unit)
                painter.setFont(_face("figure"))
                painter.drawText(line.adjusted(0, 0, -QFontMetricsF(plain).horizontalAdvance(unit), 0),
                                 Qt.AlignRight | Qt.AlignVCenter, cell.figure)
                if cell.second:
                    painter.setFont(_face("second"))
                    painter.setPen(design.colour("ink_2"))
                    under = QRectF(box.left(), box.top() + height, box.width(), design.scaled(design.TYPE["small"][1]))
                    painter.drawText(under, Qt.AlignRight | Qt.AlignVCenter, cell.second)
            elif isinstance(cell, TextCell):
                font = _face("selected" if selected and first else "text")
                painter.setFont(font)
                painter.setPen(design.colour("ink_2" if cell.quiet else "ink"))
                if cell.wrap:
                    layout, _lines = _wrapped(cell.text, font, box.width(), height)
                    layout.draw(painter, box.topLeft())
                else:
                    mode = Qt.ElideMiddle if cell.elide == "middle" else Qt.ElideRight
                    painter.drawText(line, Qt.AlignLeft | Qt.AlignVCenter,
                                     QFontMetricsF(font).elidedText(cell.text, mode, box.width()))
        finally:
            painter.restore()


class RuleHeader(QHeaderView):
    """The header of a ruled table. It paints its own words, small and semibold, and one hairline under
    them across every column. With ``sortable`` its sections can be clicked, and the column the rows are in
    order of wears a chevron after its words, up for ascending and down for descending; before anything was
    sorted no column wears one. It says which column and which way through the toolkit's own signal for a
    changed sort indicator and never orders the rows itself. A section whose painting fails is drawn by the
    toolkit itself."""

    def __init__(self, sortable: bool = False, parent=None):
        super().__init__(Qt.Horizontal, parent)
        self.sortable = bool(sortable)
        self.setFixedHeight(design.scaled(design.TYPE["small"][1]) + 2 * design.scaled(design.SIZE["table_pad"])
                            + design.SIZE["hair"])
        self.setHighlightSections(False)
        self.setMinimumSectionSize(design.scaled(design.SIZE["table_section_least"]))
        self.setSectionsClickable(self.sortable)
        # The toolkit's own indicator is never shown, and a fresh header would name its first column as the
        # sorted one: the chevron is painted here, from the column and the order the header holds.
        self.setSortIndicatorShown(False)
        self.setSortIndicator(-1, Qt.AscendingOrder)
        self.viewport().setAutoFillBackground(False)         # the header stands on the page's own ground

    @reported(QHeaderView.paintSection)
    def paintSection(self, painter, rect, index):
        if not rect.isValid():
            return
        painter.save()
        try:                                 # the header paints its other sections with the same painter
            model = self.model()
            lead, gap = _pads(index, self.count())
            pad = design.scaled(design.SIZE["table_pad"])
            box = QRectF(rect).adjusted(lead, pad, -gap, -(pad + design.SIZE["hair"]))
            chevron = self.sortable and self.sortIndicatorSection() == index
            side, before = design.scaled(design.SIZE["marker"]), design.scaled(design.SIZE["icon_gap"])
            room = box.adjusted(0, 0, -(before + side), 0) if chevron else box    # the chevron stays in its section
            font = _face("title")
            metrics = QFontMetricsF(font)
            title = model.headerData(index, Qt.Horizontal)
            words = metrics.elidedText("" if title is None else str(title), Qt.ElideRight, max(0.0, room.width()))
            aligned = model.headerData(index, Qt.Horizontal, Qt.TextAlignmentRole)
            at_right = aligned is not None and bool(aligned & Qt.AlignRight)
            painter.setFont(font)
            painter.setPen(design.colour("ink"))
            painter.drawText(room, (Qt.AlignRight if at_right else Qt.AlignLeft) | Qt.AlignVCenter, words)
            if chevron:                      # one group with the words: it follows them, at either end
                after = room.right() if at_right else room.left() + metrics.horizontalAdvance(words)
                up = self.sortIndicatorOrder() == Qt.AscendingOrder
                paint.paint_marker(painter, "chev_up" if up else "chev_down",
                                   QRectF(after + before, box.top() + (box.height() - side) / 2, side, side))
        finally:
            painter.restore()

    @reported()
    def paintEvent(self, event):
        super().paintEvent(event)            # first: no painter of this method stands while the base paints
        painter = QPainter(self.viewport())
        try:
            paint.fill_px(painter, QRectF(self.viewport().rect()), design.colour("rule"), bottom=design.SIZE["hair"])
        finally:
            painter.end()


class RuleView(QTableView):
    """The rows of a ruled table: ``columns`` are ``Column`` descriptions and each row a list of cell
    descriptions, one to a column. It is as tall as its header and its rows and has no scroll bar. With the
    keyboard focus shown it rings its current row, clear of a selected row's bar, and never the whole table.
    Tab leaves it."""

    def __init__(self, columns, *, selectable: bool = True, sortable: bool = False, parent=None):
        columns = _checked_columns(columns)                  # refused before anything is built
        super().__init__(parent)
        self._columns = columns
        self._widths = [None] * len(columns)                 # None: the column that takes the room left
        self._focus_shown = False
        model = QStandardItemModel(0, len(columns), self)
        for index, column in enumerate(columns):
            model.setHeaderData(index, Qt.Horizontal, column.title)
            model.setHeaderData(index, Qt.Horizontal, Qt.AlignRight if column.right else Qt.AlignLeft,
                                Qt.TextAlignmentRole)
        header = RuleHeader(sortable, self)
        self.setHorizontalHeader(header)
        self.setModel(model)
        self.setItemDelegate(RuleDelegate(self))
        self.setFrameShape(QFrame.NoFrame)
        self.setShowGrid(False)
        self.setWordWrap(False)
        self.verticalHeader().hide()
        self.verticalHeader().setMinimumSectionSize(design.scaled(design.SIZE["table_row"]))
        self.verticalHeader().setSectionResizeMode(QHeaderView.Fixed)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setTabKeyNavigation(False)                      # Tab goes on to the next control
        self.viewport().setAutoFillBackground(False)         # the rows stand on the page's own ground
        self.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.setSelectionMode(QAbstractItemView.SingleSelection if selectable else QAbstractItemView.NoSelection)
        self.setFocusPolicy(Qt.StrongFocus if selectable else Qt.NoFocus)
        self.setEditTriggers(QAbstractItemView.NoEditTriggers)
        for index, column in enumerate(columns):
            header.setSectionResizeMode(index, QHeaderView.Stretch if column.width is None else QHeaderView.Fixed)
        self._fit_columns()
        self._set_heights()

    def set_rows(self, rows) -> None:
        """Replace the rows. A row with a cell too few or too many, a cell that is no description and a state
        with no marker are refused before anything is changed. Afterwards nothing is selected, the fitted
        columns are measured again and every row has its height. The model stays the same one."""
        rows = _checked_rows(rows, len(self._columns))
        model = self.model()
        model.removeRows(0, model.rowCount())
        for row in rows:
            items = []
            for cell in row:
                item = QStandardItem(_words(cell))
                item.setData(cell, Qt.UserRole)
                item.setEditable(False)
                items.append(item)
            model.appendRow(items)
        self._fit_columns()
        self._set_heights()

    def set_focus_shown(self, on: bool) -> None:
        """Show the keyboard focus, as a ring round the current row, or stop showing it."""
        self._focus_shown = bool(on)
        self.viewport().update()

    def _fit(self, column: int) -> int:
        """A fitted column's width: its title or its widest cell, whichever is wider, held to the widest a
        fitted column may be, then the room it keeps clear of its words; never under a column's least width.
        The first column's words are measured semibold, which is how a selected row sets them."""
        metrics = {}

        def wide(part, words):
            if not words:
                return 0.0
            if part not in metrics:
                metrics[part] = QFontMetricsF(_face(part))
            return metrics[part].horizontalAdvance(words)

        need = wide("title", self._columns[column].title)
        if self.horizontalHeader().sortable:                 # room for the chevron after the title
            need += design.scaled(design.SIZE["icon_gap"]) + design.scaled(design.SIZE["marker"])
        model = self.model()
        for row in range(model.rowCount()):
            cell = model.index(row, column).data(Qt.UserRole)
            if isinstance(cell, ValueCell):
                unit = wide("unit", f" {cell.unit}") if cell.unit else 0.0
                need = max(need, wide("figure", cell.figure) + unit, wide("second", cell.second))
            elif isinstance(cell, StateCell):
                marker = design.scaled(design.SIZE["marker"]) + design.scaled(design.SIZE["state_gap"])
                need = max(need, marker + wide("selected" if column == 0 else "word", cell.word))
            else:
                need = max(need, wide("selected" if column == 0 else "text", cell.text))
        lead, gap = _pads(column, len(self._columns))
        width = min(design.scaled(design.SIZE["table_fit_most"]), math.ceil(need)) + lead + gap
        return max(width, design.scaled(design.SIZE["table_section_least"]))

    def _fit_columns(self) -> None:
        """Give every column that has a width of its own that width, and the view its least width."""
        header = self.horizontalHeader()
        for index, column in enumerate(self._columns):
            if column.width is not None:
                width = self._fit(index) if column.width == "fit" else design.scaled(column.width)
                self._widths[index] = width
                header.resizeSection(index, width)
        taken = sum(width for width in self._widths if width is not None)
        self.setMinimumWidth(taken + design.scaled(design.SIZE["table_stretch_least"]))

    def _column_width(self, column: int) -> int:
        """A column's width as it WILL be: the one that takes the room left is not yet resized when the rows
        are measured."""
        if self._widths[column] is not None:
            return self._widths[column]
        taken = sum(width for width in self._widths if width is not None)
        return max(design.scaled(design.SIZE["table_stretch_least"]), self.width() - taken)

    def _cell_height(self, index) -> int:
        """The height one cell asks of its row: a line of words, or as many lines as its wrapped words take
        at the width its column will have, or a line and a small one under it; the space above and below; and
        never under a row's least height. Wrapped words in the first column are measured semibold where a row
        can be selected, which is how a selected row sets them; a table that selects nothing never sets them
        so, and measures them plain."""
        cell = index.data(Qt.UserRole)
        line = design.scaled(design.TYPE["body"][1])
        lines = line
        if isinstance(cell, TextCell) and cell.wrap:
            lead, gap = _pads(index.column(), len(self._columns))
            semibold = index.column() == 0 and self.selectionMode() != QAbstractItemView.NoSelection
            font = _face("selected" if semibold else "text")
            lines = line * _wrapped(cell.text, font, self._column_width(index.column()) - lead - gap, line)[1]
        elif isinstance(cell, ValueCell) and cell.second:
            lines = line + design.scaled(design.TYPE["small"][1])
        return max(lines + 2 * design.scaled(design.SIZE["table_pad"]), design.scaled(design.SIZE["table_row"]))

    def _set_heights(self) -> None:
        """Set every row's height from its cells, then the view's own: its header and its rows."""
        model = self.model()
        total = self.horizontalHeader().height()
        for row in range(model.rowCount()):
            height = max(self._cell_height(model.index(row, column)) for column in range(model.columnCount()))
            self.setRowHeight(row, height)
            total += height
        self.setFixedHeight(total)

    @reported(QTableView.resizeEvent)
    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._set_heights()                  # wrapped words take other lines at another width

    @reported()
    def paintEvent(self, event):
        super().paintEvent(event)            # first: no painter of this method stands while the base paints
        index = self.currentIndex()
        if not (self._focus_shown and index.isValid()):
            return
        painter = QPainter(self.viewport())
        try:
            painter.setRenderHint(QPainter.Antialiasing, True)
            ring = design.SIZE["focus_ring"]
            inset = design.SIZE["row_ring_inset"] + ring / 2          # the pen's middle: half its width further in
            row = QRectF(0, self.rowViewportPosition(index.row()), self.viewport().width(),
                         self.rowHeight(index.row()))
            painter.setPen(QPen(design.colour("focus"), ring))
            painter.setBrush(Qt.NoBrush)
            painter.drawRoundedRect(row.adjusted(design.SIZE["bar"] + inset, inset, -inset, -inset),
                                    design.SIZE["radius"], design.SIZE["radius"])
        finally:
            painter.end()


class RuleTable(Ruled):
    """A table between its two heavy rules. ``view`` holds the rows and ``header`` is the view's header. A
    page hears a selection through ``view.selectionModel()`` and, from a header that sorts, a click through
    ``header.sortIndicatorChanged``. A table that is not ``selectable`` takes neither a selection nor the
    keyboard focus."""

    def __init__(self, columns, rows=(), *, selectable: bool = True, sortable: bool = False):
        columns = _checked_columns(columns)                  # both refused before anything is built
        rows = _checked_rows(rows, len(columns))
        super().__init__(top=design.SIZE["rule"], bottom=design.SIZE["rule"])
        self.view = RuleView(columns, selectable=selectable, sortable=sortable)
        self.header = self.view.horizontalHeader()
        self.box.addWidget(self.view)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)     # as tall as its rows, wherever it stands
        self.view.set_rows(rows)

    def set_rows(self, rows) -> None:
        """Replace the rows; a row or a cell that cannot stand is refused before anything is changed, and
        afterwards nothing is selected."""
        self.view.set_rows(rows)

    def select_row(self, row: int | None) -> None:
        """Select a row and make it the current one, which the ring is drawn round; None selects nothing and
        leaves the current row where it was. A row that is not there selects nothing."""
        if row is None:
            self.view.clearSelection()
        else:
            self.view.selectRow(row)

    def selected_row(self) -> int | None:
        """The selected row, or None."""
        rows = self.view.selectionModel().selectedRows()
        return rows[0].row() if rows else None
