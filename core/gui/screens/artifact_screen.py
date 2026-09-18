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

The detail pane is TEXT ONLY (B4): no figure rendering, no open-the-folder button, no "use this" jump
into a stage tab. All three were considered and declined on 2026-09-17 (§1.3) and are additive on top
of ``_detail_text`` if they are ever wanted.
"""
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFontDatabase
from PySide6.QtWidgets import (QComboBox, QFileDialog, QHBoxLayout, QLabel, QPlainTextEdit,
                               QPushButton, QSplitter, QVBoxLayout, QWidget)

from core.artifacts import render_manifest
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

# The tail of a run's records the pane shows (§3.3). A ceiling, not a budget: a training run's log is
# unbounded and the pane is a text box.
LOG_MAX_BYTES = 1 << 20

# read_log answers None (no file at all) and "" (a run that said nothing) differently, deliberately
# (§2.2), and the pane states BOTH: piece 3's invariant is that silence is a record too, and a blank
# pane under a Records heading reads as a bug rather than as an answer. None has two causes, and the
# cache's is its own -- it has no writer, so no log.txt is ever written beside it.
_CACHE_NO_LOG = ("a training cache keeps no log: it is written batch by batch across resumes and "
                 "shared by every posterior that names it")
_NO_RUN_LOG = "written outside a run"
_EMPTY_LOG = "the run recorded nothing"
_RECORDS_HEADER = "── the run's records (log.txt) ──"


def _stale_folder_note(actual: str, expected: str) -> str:
    """Said when Summary.dir_name disagrees with the manifest's own dir_name (§3.3).

    ArtifactStore.rename writes the manifest FIRST and moves the directory SECOND, and tolerates a
    PermissionError on the move (store.py:430-441) because the manifest is what resolves an artifact.
    Nothing is lost when that happens and nothing has ever said it happened.

    Never said for the simulation kind -- ``_detail_text`` guards the call, and the reason is there.
    """
    return (f"This artifact's folder is called {actual!r}, but its own manifest says {expected!r}. A "
            f"rename writes the manifest first and moves the directory second, and the move can be "
            f"refused (a handle held open on Windows). The manifest is what resolves an artifact, so "
            f"nothing is lost -- only the folder name is out of date.")


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

        # Read-only, text only (B4): the manifest rendered, then the run's records. A fixed-pitch
        # face, because the manifest is rendered in aligned columns and the records carry HH:MM:SS
        # stamps -- both ragged out in a proportional font. No wrapping, for the same reason.
        self.detail = QPlainTextEdit()
        self.detail.setReadOnly(True)
        self.detail.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.detail.setFont(QFontDatabase.systemFont(QFontDatabase.FixedFont))

        self.btn_save = QPushButton("Save…")
        self.btn_save.setToolTip("Write what is shown here to a text file")
        self.btn_save.setEnabled(False)
        self.btn_save.clicked.connect(self._save_shown)

        # The pane's button row, kept as an attribute rather than a local: Task 12's "Lineage
        # report…" button mounts into THIS layout, beside Save, because both write a file about the
        # selected artifact.
        self.detail_actions = QHBoxLayout()
        self.detail_actions.addStretch(1)
        self.detail_actions.addWidget(self.btn_save)

        detail_side = QWidget()
        detail_layout = QVBoxLayout(detail_side)
        detail_layout.setContentsMargins(0, 0, 0, 0)
        detail_layout.addWidget(self.detail, 1)
        detail_layout.addLayout(self.detail_actions)
        self.split.addWidget(detail_side)
        self.split.setStretchFactor(1, 1)
        self.table.selection_changed.connect(self._on_selection_changed)

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
        if self.table.columnCount():
            # The user's own header click, kept across the rebuild -- but only off a table that HAS
            # rows. An empty one still reports DEFAULT_SORT (never anything negative), so an
            # unguarded capture here would overwrite the sort restore_settings just read out of
            # QSettings before _apply_sort below could use it, and "the sort survives a relaunch"
            # would be quietly false.
            self._sort = self.table.sort_state()
        try:
            rows = self._resolved_store().list(kind)
        except Exception as e:                 # noqa: BLE001 -- REPORTED, never swallowed
            # StorePicker.refresh lists nothing on an unreadable root, which makes "there is nothing
            # here" and "I could not look" identical. The browser must tell them apart (§3.2), so the
            # error and its class go on the status line and the table is emptied.
            self.table.set_rows(kind, [])
            self._on_selection_changed()
            self._set_status(f"Could not read the {kind} artifacts: {type(e).__name__}: {e}",
                             error=True)
            return
        self.table.set_rows(kind, rows)
        self._apply_sort(kind)
        # Explicitly, not off the signal: set_rows leaves nothing selected, and a table that was
        # already empty emits no change -- so a stale pane would outlive the rows it described.
        self._on_selection_changed()
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

    # ── the detail view (§3.3, B4) ────────────────────────────────────────────
    def _on_selection_changed(self) -> None:
        """Re-render the pane for whatever is selected now (nothing -> an empty pane)."""
        s = self.table.current_summary()
        try:
            self.detail.setPlainText(self._detail_text(s))
        except Exception as e:                 # noqa: BLE001 -- reported, never swallowed
            # The row was listed and the artifact has gone since, or its manifest stopped parsing
            # (another process, a half-finished copy). Say which rather than show a blank pane.
            self.detail.setPlainText("")
            self._set_status(f"Could not read the selected artifact: {type(e).__name__}: {e}",
                             error=True)
        self.btn_save.setEnabled(bool(self.detail.toPlainText()))

    def _detail_text(self, s) -> str:
        """Everything the pane shows for one row, as one string -- which is what Save writes (B10).

        Text only (B4): the manifest rendered by the ONE renderer both front ends use
        (core/artifacts/report.py), the stale-folder note when there is one, then the run's records
        verbatim with their HH:MM:SS level stamps -- or, when there are none, WHICH kind of none it
        is (§3.3). Two store reads and no other side effect.
        """
        if s is None:
            return ""
        if not s.complete:
            # Nothing can load this directory, so there is no manifest to render: what there is to
            # say is why it was not read, where it is, and what it is called.
            return "\n".join((f"{s.kind} — an incomplete directory, which nothing can load.",
                              f"folder:  {s.dir_name}",
                              f"path:    {s.path}",
                              f"reason:  {s.reason}"))
        store = self._resolved_store()
        m = store.get(s.kind, s.id)
        parts = [render_manifest(m)]
        # NOT for the simulation kind: Manifest.dir_name is the bare digest for a cache
        # (manifest.py:73-77) and write_simulation_manifest writes into whatever directory it is
        # handed, so a cache folder named anything else is legitimate -- the note would fire on every
        # hand-placed cache and report a half-failed rename that never happened.
        if s.kind != "simulation" and s.dir_name != m.dir_name:
            parts += ["", _stale_folder_note(s.dir_name, m.dir_name)]
        text, truncated = store.read_log(s.kind, s.id, max_bytes=LOG_MAX_BYTES)
        parts += ["", _RECORDS_HEADER]
        if text is None:
            parts.append(_CACHE_NO_LOG if s.kind == "simulation" else _NO_RUN_LOG)
        elif truncated:
            parts.append(f"(only the last {LOG_MAX_BYTES} bytes are shown; the file is longer)")
            parts.append(text)
        elif not text:
            # read_log says "" for a run that said nothing and None for no file at all (§2.2).
            # Silence is a record too, so the pane says so rather than leave the heading bare.
            parts.append(_EMPTY_LOG)
        else:
            parts.append(text)
        return "\n".join(parts)

    def _save_shown(self) -> None:
        """B10's first half: what is on screen, to a file the operator picks.

        A FILE, never an eighth store kind -- a report DESCRIBES the store and must not be mistaken
        for something the store holds. The dialog is QFileDialog.getSaveFileName, the same call
        simulate_panel._save_video and figure_window use; cancelling returns "" and writes nothing.
        """
        text = self.detail.toPlainText()
        if not text:
            return                     # the button is disabled with nothing selected; belt and braces
        s = self.table.current_summary()
        suggested = f"{s.kind}_{(s.name or s.id) if s.complete else s.dir_name}.txt"
        path, _ = QFileDialog.getSaveFileName(self, "Save what is shown", suggested,
                                              "Text file (*.txt)")
        if not path:
            return
        if not path.lower().endswith(".txt"):
            path += ".txt"
        try:
            Path(path).write_text(text, encoding="utf-8", newline="\n")
        except OSError as e:
            self._set_status(f"Could not write {path}: {e}", error=True)
            return
        self._set_status(f"Saved what is shown to {Path(path).name}.")

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
