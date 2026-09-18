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
from PySide6.QtWidgets import (QComboBox, QFileDialog, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
                               QMessageBox, QPlainTextEdit, QPushButton, QSplitter, QVBoxLayout,
                               QWidget)

from core.artifacts import render_manifest
from core.artifacts.store import KIND_DIRS
from core.refusals import NOTE_MAX_CHARS, Refusal, require_note

from .. import settings
from ..design import SPACE
from ..panels.base_panel import BasePanel
from ..widgets.artifact_table import DEFAULT_SORT, ArtifactTable, columns_for
from ..widgets.refusal_box import show_refusal

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

# The one dependent that names nothing. A cache's directory is keyed on the prior's GMM, so
# ArtifactStore.dependents reports it whether or not the manifest records the parent link -- and a
# bare list of ids gives an operator no way to tell that one from a child that named it.
_FINGERPRINT_DEPENDENT = ("a training cache was generated against this prior and its rows are "
                          "meaningless without it")


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


def _delete_prompt(s) -> tuple:
    """``(text, informative)`` for the confirmation: what goes, and what cannot come back.

    An UNFINISHED cache names its committed BATCHES: ``finished`` is not ``complete`` (B3), and those
    batches are the only thing deleting one destroys that a later run could not simply remake.

    Batches and not rows, deliberately (P2): ``rows`` is written by ``mark_complete`` alone --
    ``training_checkpoint.save`` passes none -- so every real mid-run cache has ``rows is None``, and
    a ``sum(())`` here would print a confident, false "0 rows". Where the row counts come from is
    said instead.
    """
    lines = [f"id {s.id}"]
    if s.kind == "simulation" and not s.finished:
        lines.append(f"This training cache is UNFINISHED: {s.batches_done} committed batch(es). "
                     "Deleting it throws those batches away and a later run starts from zero. "
                     "(Only the batch count is known while a cache is running: the rows are "
                     "recorded when the cache finishes.)")
    lines.append(f"This removes {s.path} and everything in it, and cannot be undone.")
    return f"Delete {s.kind} {s.label}?", "\n".join(lines)


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

        # The actions row sits directly above the status line.
        self.layout().insertWidget(self.layout().count() - 1, self._build_actions())
        self.table.selection_changed.connect(self._sync_actions)
        self._sync_actions()

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

    # ── the actions (§3.4; B5, B6, B7, B8) ────────────────────────────────────
    def _build_actions(self) -> QWidget:
        """The Note box and the Delete/Sweep buttons: one row under the listing.

        A plain group box, not the model builder's sticky action bar -- this screen does not scroll a
        form of unbounded length, so nothing can fall below the fold.
        """
        box = QGroupBox("Actions")
        row = QHBoxLayout(box)
        self.note_edit = QLineEdit()
        self.note_edit.setPlaceholderText(
            f"one line, at most {NOTE_MAX_CHARS} characters; empty clears it")
        # NO setMaxLength: B5 REFUSES an over-long note and names both numbers. A maxLength would
        # silently truncate it instead, which is the clamp V2 exists to forbid.
        self.btn_note = QPushButton("Set")
        self.btn_note.clicked.connect(self._set_note)
        self.btn_delete = QPushButton("Delete…")
        self.btn_delete.clicked.connect(self._delete)
        self.btn_sweep = QPushButton("Sweep this kind…")
        self.btn_sweep.setToolTip("Remove this kind's directories that have no usable manifest")
        self.btn_sweep.clicked.connect(lambda: self._sweep(all_kinds=False))
        self.btn_sweep_all = QPushButton("Sweep all kinds…")
        self.btn_sweep_all.clicked.connect(lambda: self._sweep(all_kinds=True))
        row.addWidget(QLabel("Note"))
        row.addWidget(self.note_edit, 1)
        row.addWidget(self.btn_note)
        row.addSpacing(SPACE[2])
        row.addWidget(self.btn_delete)
        row.addSpacing(SPACE[2])
        row.addWidget(self.btn_sweep)
        row.addWidget(self.btn_sweep_all)
        return box

    def _sync_actions(self) -> None:
        """Enable what the selected row can answer, and show its note.

        An incomplete directory has no manifest, so there is no note to rewrite and nothing for
        ``delete`` to resolve (it goes through ``_find``, which only ever returns manifest-bearing
        entries): the sweep is what removes those, which is also why the sweep is the one action that
        needs no selection at all.
        """
        s = self.table.current_summary()
        live = s is not None and s.complete
        self.note_edit.setEnabled(live)
        self.btn_note.setEnabled(live)
        self.btn_delete.setEnabled(live)
        self.note_edit.setText(s.note if live else "")

    def _after_change(self) -> None:
        """Re-list the kind and announce the change (B8). The three StorePickers are the WINDOW's to
        refresh: it is the one party that knows all of them, exactly as for the model combos.

        Its CALLERS set their own status line AFTER calling this, never before: ``refresh`` ends by
        reporting the kind's row counts (or, when the kind cannot be read, the error), so a sentence
        written first would be overwritten by the re-list it triggered and the operator would be
        told the row count in answer to a Delete.
        """
        self.refresh()
        self._sync_actions()
        self.store_changed.emit()

    def _refuse_while_running(self, doing: str) -> bool:
        """True when a run is live: the status line says so and the caller returns (B6).

        Every WRITE goes through this and nothing that only READS does. The wording is the window's
        own, from the two model-builder sites (model_builder_screen.py:371, :447); the status line is
        this screen's surface, so there is no dialog to dismiss either.
        """
        if BasePanel._running:
            self._set_status(f"A task is running -- wait for it to finish before {doing}.", error=True)
            return True
        return False

    def _selected(self, doing: str):
        """The selected COMPLETE row, or None with the refusal already shown.

        Two different answers: nothing selected is a fielded refusal (the yellow box's fix sentence
        names the list), while a selected LEFTOVER is not a refusal at all -- there is no artifact
        there to act on, and saying so on the status line points at the sweep without pretending a
        rule was broken.
        """
        s = self.table.current_summary()
        if s is None:
            show_refusal(self, Refusal("No artifact is selected.", field="artifact"))
            return None
        if not s.complete:
            self._set_status(f"{s.dir_name} has no usable manifest ({s.reason}), so there is nothing "
                             f"to {doing}: it is one of the leftovers a sweep removes.", error=True)
            return None
        return s

    def _set_note(self) -> None:
        """B5: one trimmed line, at most NOTE_MAX_CHARS; blank clears it.

        The rule runs at the click, in ``core.refusals``, so ``python -m core artifacts note``
        refuses the same note with the same sentence; the front end only adds where to fix it.
        """
        if self._refuse_while_running("setting a note"):
            return
        s = self._selected("annotate")
        if s is None:
            return
        try:
            note = require_note("note", self.note_edit.text())
            self._resolved_store().set_note(s.kind, s.id, note)
        except Refusal as exc:      # StoreError is one: the row can have gone since it was listed,
            show_refusal(self, exc)  # and since Task 2 that refusal carries field="note" too
            self._set_status(exc.message, error=True)
            return
        self._after_change()        # before the sentence, never after: see _after_change
        self._set_status(f"Set the note on {s.kind} {s.label}." if note
                         else f"Cleared the note on {s.kind} {s.label}.")

    def _dependents_refusal(self, store, s, deps) -> Refusal:
        """The sentence for a delete that nothing can perform: the artifact, then every artifact
        that depends on it WITH WHY.

        ``field="artifact"`` sends the operator to the list to delete the children first -- the same
        key the store's own dependents refusal carries since Task 5. This one exists for the WORDING
        alone: the store's message ends "pass force=True to orphan them", which no front end offers
        (B6), and a bare list of ids cannot say which dependent named this artifact and which the
        store found by fingerprint.
        """
        parents = {}
        for kind in sorted({k for k, _, _ in deps}):
            for row in store.list(kind):
                parents[(kind, row.id)] = row.parents
        lines = []
        for kind, id_, name in deps:
            held = parents.get((kind, id_)) or {}
            # dependents()'s first pass matches parents[<kind key>] == id; its second adds simulation
            # caches by fingerprint alone. "prior" is the only parent key a prior can occupy, so a
            # cache that does not hold it there came from that second pass.
            fingerprint_only = (s.kind == "prior" and kind == "simulation"
                                and held.get("prior") != s.id)
            why = _FINGERPRINT_DEPENDENT if fingerprint_only else f"it names this {s.kind} as a parent"
            lines.append(f"  {kind} {name or '(unnamed)'} [{id_}]: {why}.")
        return Refusal(
            f"Refusing to delete {s.kind} {s.label} [{s.id}]: {len(deps)} artifact(s) depend on it.\n"
            + "\n".join(lines)
            + "\nDelete those first. Nothing here can orphan them.", field="artifact")

    def _delete(self) -> None:
        """One artifact, no force, TWO outcomes (B6).

        ``dependents`` is read here, before anything is asked, because ``store.delete`` refuses
        anything with dependents and no front end offers ``force=True``: a confirmation for such an
        artifact could only ever be followed by a failure. With dependents this is a refusal naming
        every one of them; without, a confirmation that defaults to No. The store reads
        ``dependents`` again inside ``delete`` and stays the last word -- two directory scans, which
        is irrelevant at this scale.
        """
        if self._refuse_while_running("deleting an artifact"):
            return
        s = self._selected("delete")
        if s is None:
            return
        store = self._resolved_store()
        deps = store.dependents(s.kind, s.id)
        if deps:
            show_refusal(self, self._dependents_refusal(store, s, deps))
            self._set_status(f"{s.kind} {s.label} has {len(deps)} dependent(s) and was not deleted.",
                             error=True)
            return
        text, detail = _delete_prompt(s)
        box = QMessageBox(self)                     # an INSTANCE dialog: the statics escape the
        box.setIcon(QMessageBox.Warning)            # suite's guard and hang it offscreen
        box.setWindowTitle("Delete artifact")
        box.setText(text)
        box.setInformativeText(detail)
        box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        box.setDefaultButton(QMessageBox.No)        # the safe branch, and what exec() == 0 reads as
        if box.exec() != QMessageBox.Yes:
            self._set_status(f"{s.kind} {s.label} was not deleted.")
            return
        try:
            store.delete(s.kind, s.id)              # never force, from either front end (B6)
        # A row that went, or dependents that arrived, since the list -- shown as the store wrote
        # it, fix sentence and all (its refusals are fielded).
        except Refusal as exc:
            show_refusal(self, exc)
            self._set_status(exc.message, error=True)
            return
        self._after_change()        # before the sentence, never after: see _after_change
        self._set_status(f"Deleted {s.kind} {s.label}.")

    def _incomplete(self, store, kind) -> tuple:
        """``([(kind, dir_name, reason)], [str])``: what a sweep would remove, and any kind that
        could not be read at all -- an unreadable directory is not an empty one (§3.2).

        Read off ``list``, so the confirmation shows exactly the rows the table calls incomplete.
        """
        out, problems = [], []
        for k in ([kind] if kind else list(KIND_DIRS)):     # Task 9's import; the seven, in order
            try:
                rows = store.list(k)
            except Exception as e:      # noqa: BLE001 -- an unreadable kind is reported, not fatal
                problems.append(f"The {k} directory could not be read: {e}")
                continue
            out += [(k, row.dir_name, row.reason) for row in rows if not row.complete]
        return out, problems

    def _sweep(self, *, all_kinds: bool) -> None:
        """B7: remove every directory of this kind -- or of all seven -- that has no usable manifest.

        The candidates are listed BEFORE anything is removed, and ``sweep_incomplete`` is what
        removes them: a call that can only ever touch a directory ``_entries`` classifies as
        incomplete, the complement of ``delete``, which can only ever touch a real artifact.
        """
        if self._refuse_while_running("removing leftover directories"):
            return
        store = self._resolved_store()
        kind = None if all_kinds else self.kind_combo.currentData()
        cands, problems = self._incomplete(store, kind)
        tail = (" " + " ".join(problems)) if problems else ""
        if not cands:
            where = "any kind's" if kind is None else f"{kind}"
            self._set_status(f"Nothing to remove: every {where} directory has a usable manifest."
                             + tail, error=bool(problems))
            return
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Warning)
        box.setWindowTitle("Remove leftover directories")
        box.setText(f"Remove {len(cands)} director{'y' if len(cands) == 1 else 'ies'} with no "
                    f"usable manifest?")
        box.setInformativeText("\n".join(f"{k}/{d} — {why}" for k, d, why in cands)
                               + "\n\nA directory that holds a manifest is never touched by this.")
        box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        box.setDefaultButton(QMessageBox.No)
        if box.exec() != QMessageBox.Yes:
            self._set_status("Nothing was removed." + tail, error=bool(problems))
            return
        removed, failed = store.sweep_incomplete(kind)
        said = f"Removed {len(removed)} of {len(cands)} leftover director" \
               f"{'y' if len(cands) == 1 else 'ies'}."
        for k, d, why in failed:
            said += f" {k}/{d} could not be removed: {why}."
        self._after_change()        # before the sentence, never after: see _after_change
        self._set_status(said + tail, error=bool(failed or problems))

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
