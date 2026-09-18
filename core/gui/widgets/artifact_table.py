"""The artifact browser's table: the ONE formatter for a store row, and the flat sortable tree that
shows it (piece 4, B2, design §3.2).

``columns_for`` and ``cells_for`` are pure functions over a ``core.artifacts.store.Summary`` and read
NOTHING -- no store, no manifest, no disk. That is what B2 buys: ``Summary`` carries every fact a row
shows, so a listing costs one directory scan and a row costs nothing. It is also why the pickers'
selection line (§6.3) is built from these and not from a second reader: what must not happen is two
surfaces disagreeing about whether a cache is finished (§1.2).

There was no item view anywhere in the repository before this piece and ``core/gui/design.py`` styled
none, so the QSS for ``QTreeView``/``QTreeView::item``/``QHeaderView::section`` lands there in the
same commit as this file.
"""
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QAbstractItemView, QTreeWidget, QTreeWidgetItem

# Every kind leads with these two.
_LEADING = ("Name", "Created")

# Per kind, the columns BETWEEN "Created" and "Note" -- what only that kind has (design §3.2's
# table). CLOSED against store.KIND_DIRS by the tests: an eighth kind must appear here or fail there.
# A diagnostic shows `variant` and not `mode`, deliberately: it records its kind under `variant` and
# has no conditioning geometry, so a mode column could only ever be blank for it (§1.3).
_EXTRA_COLUMNS: dict[str, tuple[str, ...]] = {
    "prior": (),
    "simulation": ("Progress", "Finished"),
    "posterior": ("Mode", "Width", "Amortized"),
    "observation": ("Mode", "Width"),
    "calibration": (),
    "inference": (),
    "diagnostic": ("Variant",),
}

# "Created", descending: newest first, which is the order ArtifactStore.list already returns. Used on
# the first fill and whenever a remembered sort names a column the current kind does not have. (That
# an INCOMPLETE row sorts last is not this constant's doing -- _Row.__lt__ below enforces it under
# every column and in both directions.) A plain int for the order, never a Qt.SortOrder: see
# ArtifactTable.sort_state.
DEFAULT_SORT = (1, 1)


def columns_for(kind: str) -> tuple[str, ...]:
    """The header labels for one kind's table: "Name", "Created", the kind's own, then "Note".

    KeyError for an unknown kind -- loud on purpose, exactly as ``core/gui/fields.py::label`` is: the
    kind comes from the screen's own selector, which is built from ``KIND_DIRS``, so a kind this
    table does not know is a programming mistake and not something an operator can type.
    """
    return _LEADING + _EXTRA_COLUMNS[kind] + ("Note",)


def _progress(s) -> str:
    """A cache's progress: ``"3/4 batches"`` while it runs, ``"4/4 batches · 96 rows"`` when done.

    The fraction, as in design §3.2's example, and it still costs no second read: BOTH halves are on
    the Summary, because the store task carries the planned total across as ``batches_planned`` (from
    ``body["identity"]["n_runs"]``, ``core/artifacts/identity.py:54``) beside ``batches_done``. A
    manifest that records no planned count -- nothing writes one without it today, but a hand-edited
    or an older manifest can -- falls back to the count alone rather than printing "3/None".

    The ROW count is a separate matter: ``rows`` is written only by ``mark_complete``
    (``core/SBI/training_checkpoint.py:346``), so a running cache does not know how many rows it has
    and the cell adds them only once they exist. The cell says what it knows and no more.
    """
    if s.batches_done is None:
        return ""
    n = int(s.batches_done)
    if s.batches_planned is None:
        head = f"{n} {'batch' if n == 1 else 'batches'}"
    else:
        head = f"{n}/{int(s.batches_planned)} batches"
    if not s.rows:
        return head
    return f"{head} · {sum(int(r) for r in s.rows)} rows"


def _amortization(s) -> str:
    """``amortized`` or ``narrowed (TSNPE)``; blank when the manifest records neither.

    A column names BOTH states, unlike the picker's visible item text (§6.3), which suffixes the
    exception only: a column that is blank for the common case reads as missing data, whereas a
    dropdown entry that is silent reads as the norm.
    """
    if s.amortized is None:
        return ""
    return "amortized" if s.amortized else "narrowed (TSNPE)"


def cells_for(kind: str, s) -> tuple[str, ...]:
    """One row's cells in ``columns_for(kind)`` order, every one a string.

    An INCOMPLETE row -- a directory with no usable manifest, which ``ArtifactStore.list`` returns
    with ``complete=False`` and a ``reason`` -- has no id, no name, no created and no note, so it is
    identified by its ``dir_name`` (the handle ``remove_incomplete`` takes) and carries its reason in
    the third column, in place of the kind's own. Six blank cells would read as an artifact with
    nothing in it; for the three kinds whose third column is "Note" the reason simply sits there,
    which is the only column such a row has to give it.
    """
    columns = columns_for(kind)
    if not s.complete:
        head = (s.dir_name, "", s.reason or "")
        return head + ("",) * (len(columns) - len(head))
    middle: tuple[str, ...] = ()
    if kind == "simulation":
        middle = (_progress(s), "yes" if s.finished else "no")
    elif kind == "posterior":
        middle = (s.mode or "", "" if s.width is None else str(s.width), _amortization(s))
    elif kind == "observation":
        middle = (s.mode or "", "" if s.width is None else str(s.width))
    elif kind == "diagnostic":
        middle = (s.variant or "",)
    return (s.label, s.created) + middle + (s.note or "",)


class _Row(QTreeWidgetItem):
    """One row, which sorts INCOMPLETE LAST under every column and in both directions.

    Two things make this a class rather than a blank cell that happens to sort well. The rank is on
    completeness BEFORE the column's own value, so a sort by Name cannot float a leftover directory
    ("halfwritten") above a real artifact ("newer"); and Qt inverts the whole comparison for a
    descending sort, which would put the incomplete rows on top, so the completeness bit is flipped
    with the order and comes out the same way round either way. Design §3.2 states "incomplete rows
    sort last" as a property of the table -- this is where it is a property and not an accident of an
    incomplete row's blank ``created``.

    The item carries the row's INDEX into ``ArtifactTable._rows``, not the Summary itself: Qt item
    data is a QVariant and only PySide6's wrapping makes an arbitrary object survive it, while
    sorting reorders the ITEMS and never that list.
    """

    def __init__(self, cells, index: int, complete: bool):
        super().__init__(list(cells))
        self.setData(0, Qt.UserRole, int(index))
        self._complete = bool(complete)

    def _key(self, col: int, descending: bool):
        """``(rank, the column's own text)``. ``!= descending`` is the XOR that cancels Qt's
        reversal, so the incomplete rows sit at the bottom under an ascending and a descending sort
        alike."""
        return ((not self._complete) != descending, self.text(col))

    def __lt__(self, other):
        tree = self.treeWidget()
        col = tree.sortColumn() if tree is not None else 0
        if col < 0:                                 # nothing sorted yet: rank by the first column
            col = 0
        if tree is None or not isinstance(other, _Row):     # neither happens; cheap to survive
            return self.text(col) < other.text(col)
        # .value, not int(): int(Qt.SortOrder) raises TypeError in PySide6 6.9.3.
        descending = bool(tree.header().sortIndicatorOrder().value)
        return self._key(col, descending) < other._key(col, descending)


class ArtifactTable(QTreeWidget):
    """A flat, sortable, read-only table over one kind's rows, one row selectable at a time.

    It holds the ``Summary`` objects it was given and hands the selected one back, so a caller never
    parses a cell back into a fact: the detail pane needs the id, Delete needs the kind and the id,
    and Sweep needs the ``dir_name``, none of which is on screen in full.

    ``selection_changed`` is the one signal the screen listens to. Selecting a row emits it, and so
    does a refresh, which clears the table and therefore the selection -- ``current_summary()`` is
    None until something is selected again. That is deliberate: remembering an id across a refresh is
    the dangling state design §3.4 protects the store pickers from.
    """
    selection_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._kind = ""
        self._rows: list = []
        # A fresh QTreeWidget carries ONE column and reports sortColumn() == 0 for it, which
        # ``sort_state`` would hand back as a remembered "sort by Name ascending" on the very first
        # fill -- the one order the default exists to avoid. Emptying the header makes "before the
        # first fill" the state it looks like, and set_rows sets the real count a moment later.
        self.setColumnCount(0)
        self.setRootIsDecorated(False)      # flat: no expander column in front of "Name"
        self.setUniformRowHeights(True)
        self.setAlternatingRowColors(True)
        self.setAllColumnsShowFocus(True)
        # A note is edited in its own box, with require_note run at the click (B5), so a double-click
        # on a cell must not open an editor that writes nothing and refuses nothing.
        self.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.setSelectionMode(QAbstractItemView.SingleSelection)
        self.setSortingEnabled(True)
        self.itemSelectionChanged.connect(self.selection_changed.emit)

    def kind(self) -> str:
        """The kind currently shown, or "" before the first fill."""
        return self._kind

    def set_rows(self, kind: str, rows: list) -> None:
        """Replace the contents with one kind's ``Summary`` rows, keeping the sort the user chose.

        Sorting is switched OFF while the items go in and back on afterwards: with it on, Qt re-sorts
        on every single insert, which is both quadratic and enough to move a row out from under the
        item being added. The sort is then re-applied, falling back to ``DEFAULT_SORT`` when the
        remembered column is not in this kind's header -- the kinds have different widths.

        The items are ``_Row``s, which is what keeps an incomplete row at the bottom whatever the
        sort, and each one carries its index into ``self._rows``.
        """
        col, order = self.sort_state()
        self.setSortingEnabled(False)
        self.clear()
        self._kind, self._rows = kind, list(rows)
        columns = columns_for(kind)
        self.setColumnCount(len(columns))
        self.setHeaderLabels(list(columns))
        for i, s in enumerate(self._rows):
            self.addTopLevelItem(_Row(cells_for(kind, s), i, s.complete))
        self.setSortingEnabled(True)
        if not 0 <= col < len(columns):
            col, order = DEFAULT_SORT
        self.apply_sort_state(col, order)
        for c in range(len(columns)):
            self.resizeColumnToContents(c)

    def current_summary(self):
        """The selected row's ``Summary``, or None when nothing is selected."""
        chosen = self.selectedItems()
        if not chosen:
            return None
        i = chosen[0].data(0, Qt.UserRole)
        return self._rows[i] if isinstance(i, int) and 0 <= i < len(self._rows) else None

    # ── what the screen remembers (§3.5: the kind and the sort, never the selection) ─────────────
    def sort_state(self) -> tuple[int, int]:
        """``(column, order)`` as PLAIN ints -- 0 ascending, 1 descending.

        Plain in both directions, and not merely as a convenience: QSettings stores ints and hands
        them back as strings, and a Qt enum does not survive that round trip. The order is read as
        ``.value`` off the indicator rather than as ``int(...)``, because ``int(Qt.SortOrder)`` raises
        ``TypeError`` in PySide6 6.9.3 -- the pinned version -- so no caller and nothing here ever
        converts a Qt enum.

        ``DEFAULT_SORT`` before the first fill, where there is no header to read a sort off, and
        likewise when Qt reports no sorted column (-1).
        """
        if self.columnCount() == 0:
            return DEFAULT_SORT
        col = self.sortColumn()
        if not 0 <= col < self.columnCount():
            return DEFAULT_SORT
        return int(col), self.header().sortIndicatorOrder().value

    def apply_sort_state(self, col: int, order: int) -> None:
        """Sort by ``col`` in ``order`` -- 0 ascending, 1 descending, both PLAIN ints.

        Anything ``int()`` accepts, so the screen hands back exactly what QSettings gave it (a
        string) and never builds a ``Qt.SortOrder`` of its own; ``sort_state`` is the same seam in
        reverse.

        A column outside the current header is IGNORED rather than clamped to 0: a remembered sort
        may name a column this kind does not have, and quietly re-sorting by "Name" is not what the
        user chose. ``set_rows`` is where that case is handled, by substituting ``DEFAULT_SORT``.
        """
        col, order = int(col), int(order)
        if not 0 <= col < self.columnCount():
            return
        indicator = Qt.DescendingOrder if order else Qt.AscendingOrder
        # THE INDICATOR FIRST, then the sort. QTreeWidget.sortItems sorts the model and only then
        # updates the header, so a _Row reading the indicator mid-comparison would see the PREVIOUS
        # order and hoist the incomplete rows to the top. Setting the indicator on a sorting-enabled
        # view already re-sorts (QTreeView connects the header's sortIndicatorChanged), so sortItems
        # below is the brace to that belt: same column, same order, so it is not a second ordering.
        self.header().setSortIndicator(col, indicator)
        self.sortItems(col, indicator)
