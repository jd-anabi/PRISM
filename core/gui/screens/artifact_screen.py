"""The Artifacts screen: read the artifact store from inside the app (piece 4, §3; B1, B2).

A fifth Home tile and a PLAIN QWidget -- deliberately NOT a BasePanel. A BasePanel enrols itself in
``BasePanel._instances`` and ``_set_busy`` disables every instance's controls column while ANY run is
live, so a browser built on one would grey out the moment a training started and you could not read
the log of the thing you were waiting for. Reading is never blocked here; the actions that CHANGE the
store check ``BasePanel._running`` for themselves (§3.4).

The store is reached through ``_resolved_store()`` -- ``store or the process default``, the same seam
``StorePicker`` uses -- so a test hands one in and no process default is swapped.

Remembered (V5, §3.5): the kind last viewed, and the sort column and order. NOT the selected
artifact: a remembered id that has since been deleted is exactly the dangling selection §3.4 removes
from the pickers.
"""
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QComboBox, QHBoxLayout, QLabel, QPushButton, QSplitter, QVBoxLayout,
                               QWidget)

from core.artifacts.store import KIND_DIRS

from .. import settings
from ..widgets.artifact_table import DEFAULT_SORT, ArtifactTable, columns_for

# The kind selector's visible text, in KIND_DIRS order; each item's userData is the kind key itself,
# which is what the store, the settings key and the table all speak.
KIND_LABELS = {
    "prior": "Priors",
    "simulation": "Simulation caches (training rows)",
    "posterior": "Posteriors",
    "observation": "Observations",
    "calibration": "Calibrations",
    "inference": "Inferences",
    "diagnostic": "Diagnostics",
}


class ArtifactScreen(QWidget):
    """The artifact browser: one kind at a time in a sortable table, over a status line.

    Emits ``store_changed`` after any change it makes to the store; MainWindow connects that to
    ``_refresh_store_pickers`` (B8), because ``StorePicker.restore_key`` silently keeps whatever is
    current when the saved id has vanished -- deliberate for a picker, and a defect the moment
    something can delete.
    """

    store_changed = Signal()

    def __init__(self, store=None, parent=None):
        super().__init__(parent)
        self._store = store
        # (column, order), both plain ints; restored here, applied in refresh. It opens at the
        # TABLE's own default rather than at (0, 0): the table falls back to DEFAULT_SORT whenever it
        # has no sort to honour, so a screen that started anywhere else would silently re-sort every
        # first listing by Name and "newest first" would never be what a launch shows.
        self._sort = DEFAULT_SORT

        heading = QLabel("Artifacts")
        heading.setProperty("type", "heading")     # Fluent type ramp (global QSS)

        self.kind_combo = QComboBox()
        for kind in KIND_DIRS:
            self.kind_combo.addItem(KIND_LABELS[kind], userData=kind)

        self.btn_refresh = QPushButton("Refresh")
        self.btn_refresh.clicked.connect(self.refresh)

        kind_row = QHBoxLayout()
        kind_row.addWidget(QLabel("Kind"))
        kind_row.addWidget(self.kind_combo, 1)
        kind_row.addWidget(self.btn_refresh)

        self.table = ArtifactTable()

        # One child today; the detail pane joins it beside the table (§3.3).
        self.split = QSplitter(Qt.Horizontal)
        self.split.setChildrenCollapsible(False)
        self.split.addWidget(self.table)

        self.status = QLabel("")
        self.status.setWordWrap(True)

        layout = QVBoxLayout(self)
        layout.addWidget(heading)
        layout.addLayout(kind_row)
        layout.addWidget(self.split, 1)
        layout.addWidget(self.status)

        # Restore BEFORE connecting currentIndexChanged: setCurrentIndex fires it, and a refresh()
        # during __init__ would read the store at launch -- the start-up work §1.2 keeps off that
        # path. Same construction-never-fires rule the Settings screen's radios follow.
        self.restore_settings(settings.settings())
        self.kind_combo.currentIndexChanged.connect(lambda _i: self.refresh())

    # ── the listing (§3.2) ────────────────────────────────────────────────────
    def kind(self) -> str:
        """The kind key currently selected."""
        return str(self.kind_combo.currentData())

    def showEvent(self, event):
        """List on every visit, so an artifact written since the last look is there. NOT from
        __init__: MainWindow builds this screen at launch, and a directory scan per launch is exactly
        the start-up cost that lost the taskbar icon on 2026-09-11 (§1.2)."""
        super().showEvent(event)
        self.refresh()

    def refresh(self) -> None:
        """Re-list the selected kind. Idempotent, and the only place the store is read for the table."""
        kind = self.kind()
        if self.table.topLevelItemCount():
            # The user's own header click, kept across the rebuild -- but only off a table that HAS
            # rows. A table that has never been filled reports DEFAULT_SORT (or a sort it is HOLDING
            # for its first fill), never the one restore_settings just read out of QSettings, so an
            # unguarded capture here would overwrite that before _apply_sort below could use it and
            # "the sort survives a relaunch" would be quietly false.
            self._sort = self.table.sort_state()
        try:
            rows = self._resolved_store().list(kind)
        except Exception as e:                 # noqa: BLE001 -- REPORTED, never swallowed
            # StorePicker.refresh lists nothing on an unreadable root, which makes "there is nothing
            # here" and "I could not look" identical. The browser must tell them apart (§3.2), so the
            # error and its class go on the status line and the table is emptied.
            self.table.set_rows(kind, [])
            self._set_status(f"Could not read the {kind} artifacts: {type(e).__name__}: {e}",
                             error=True)
            return
        self.table.set_rows(kind, rows)
        self._apply_sort(kind)
        if not rows:
            self._set_status("Nothing here yet.")
            return
        bad = sum(1 for s in rows if not s.complete)
        self._set_status(f"{len(rows)} row(s): {len(rows) - bad} complete, {bad} incomplete.")

    def _apply_sort(self, kind: str) -> None:
        """Re-apply the remembered sort after a rebuild, clamped to THIS kind's column count: the
        columns differ per kind (§3.2), so a column remembered while viewing posteriors can be past
        the end of a calibration's."""
        col, order = self._sort
        if 0 <= col < len(columns_for(kind)):
            self.table.apply_sort_state(col, order)

    def _set_status(self, text: str, error: bool = False) -> None:
        """One line, with a ⚠ prefix when it is trouble -- ModelBuilderScreen._set_status's pattern."""
        self.status.setText(("⚠ " if error else "") + text)

    def _resolved_store(self):
        """This screen's store, or the process default -- StorePicker._resolved_store's seam."""
        from core.artifacts import resolve_store
        return resolve_store(self._store)

    # ── persistence (§3.5) ────────────────────────────────────────────────────
    def save_settings(self, qs) -> None:
        """The kind and the sort, and nothing else.

        MainWindow._save_state calls this BY NAME: ``_all_panels()`` is panel-typed and this screen is
        a QWidget (B1), so it gets no sweep for free -- which is also what keeps it out of
        ``_refresh_model_combos``.

        A table that HAS BEEN FILLED is the sort, whether or not that fill produced any rows: a kind
        with nothing in it still has a header, and a sort clicked on it is the user's choice like any
        other. One that was never filled reports DEFAULT_SORT whatever was restored, so a window
        closed without ever opening this screen must save what it read rather than the default it
        never showed. ``columnCount()`` is the exact question -- ``set_rows`` is what gives the table
        its columns -- where ``topLevelItemCount()`` would answer "did that kind have any artifacts"
        and silently discard a header click on an empty one.
        """
        col, order = self.table.sort_state() if self.table.columnCount() else self._sort
        qs.beginGroup("artifacts")
        qs.setValue("kind", self.kind())
        qs.setValue("sort_col", str(int(col)))
        qs.setValue("sort_order", str(int(order)))
        qs.endGroup()

    def restore_settings(self, qs) -> None:
        """What save_settings wrote. A kind an older build wrote and this one does not know is
        IGNORED, not restored -- findData returns -1 and the first kind stands. A missing or
        unreadable sort falls back to the TABLE's default, so a first launch lists newest-first."""
        qs.beginGroup("artifacts")
        kind = settings.get_str(qs, "kind", "")
        col = settings.get_int(qs, "sort_col", DEFAULT_SORT[0])
        order = settings.get_int(qs, "sort_order", DEFAULT_SORT[1])
        qs.endGroup()
        i = self.kind_combo.findData(kind)
        if i >= 0:
            self.kind_combo.setCurrentIndex(i)
        self._sort = (col, order)
