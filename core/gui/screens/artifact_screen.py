"""The Artifacts screen: read the artifact store from inside the app.

A fifth Home tile and a PLAIN QWidget -- deliberately NOT a BasePanel. A BasePanel enrols itself in
``BasePanel._instances`` and ``_set_busy`` disables every instance's controls column while ANY run is
live, so a browser built on one would grey out the moment a training started and you could not read
the log of the thing you were waiting for. Reading is never blocked here; the actions that CHANGE the
store check ``BasePanel._running`` for themselves.

The store is reached through ``_resolved_store()`` -- ``store or the process default``, the same seam
``StorePicker`` uses -- so a test hands one in and no process default is swapped.

Remembered: the kind last viewed, and the sort column and order -- selections only. NOT the selected
artifact: a remembered id that has since been deleted is exactly the dangling selection that
re-listing the pickers after every change removes from them.

The detail pane is TEXT ONLY: no figure rendering, no open-the-folder button, no "use this" jump
into a stage tab. All three were considered and declined on 2026-09-17 and are additive on top of
``_detail_text`` if they are ever wanted.
"""
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFontDatabase
from PySide6.QtWidgets import (QComboBox, QFileDialog, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
                               QMessageBox, QPlainTextEdit, QPushButton, QSplitter, QVBoxLayout,
                               QWidget)

from core.artifacts import render_lineage, render_manifest
from core.artifacts.store import KIND_DIRS, NO_MANIFEST_REASON
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
    # comparisons too: a comparison is a record of this kind, and both measurement panels send the
    # operator here to find one
    "fdt": "FDT measurements, sweeps and comparisons",
}

# The tail of a run's records the pane shows. A ceiling, not a budget: a training run's log is
# unbounded and the pane is a text box.
LOG_MAX_BYTES = 1 << 20

# read_log answers None (no file at all) and "" (a run that said nothing) differently, deliberately,
# and the pane states BOTH: silence is a record too -- a run that said nothing still writes an empty
# log.txt -- and a blank pane under a Records heading reads as a bug rather than as an answer. None
# has two causes, and the cache's is its own -- it has no writer, so no log.txt is ever written
# beside it.
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
    """Said when Summary.dir_name disagrees with the manifest's own dir_name.

    ArtifactStore.rename writes the manifest FIRST and moves the directory SECOND, and tolerates a
    PermissionError on the move because the manifest is what resolves an artifact. Nothing is lost
    when that happens and nothing has ever said it happened.

    Never said for the simulation kind -- ``_detail_text`` guards the call, and the reason is there.
    """
    return (f"This artifact's folder is called {actual!r}, but its own manifest says {expected!r}. A "
            f"rename writes the manifest first and moves the directory second, and the move can be "
            f"refused (a handle held open on Windows). The manifest is what resolves an artifact, so "
            f"nothing is lost -- only the folder name is out of date.")


# An fdt body's ``study`` as ``_unfinished_note`` names it; its docstring says why a phrase.
_STUDY_PHRASES = {"single": "single-cell measurement", "sweep": "sweep", "comparison": "comparison"}


def _unfinished_note(s) -> str:
    """The one sentence that says what deleting an UNFINISHED record of this kind destroys.

    TWO kinds can be unfinished, not one: a training cache, whose manifest exists from its first
    batch because it is resumable, and an fdt record, whose folder survives a cancel or a crash so
    that what it measured is still readable. The old form named the cache by hand and would
    have deleted a half-measured sweep behind a prompt that said nothing about it.

    The cache names its committed BATCHES and not its rows, deliberately: ``rows`` is written by
    ``mark_complete`` alone -- ``training_checkpoint.save`` passes none -- so every real mid-run
    cache has ``rows is None``, and a ``sum(())`` here would print a confident, false "0 rows".

    An fdt record is described by what its study holds, because one sentence for all three was false
    for some of them:

    - a MEASUREMENT (a single-cell run or a sweep) names its operating points where it has them --
      the done count alone when the planned total is not recorded, as the table's Points cell does,
      never "4 of ?" -- and says there is no resume: unlike a cache, whose batches a later run
      continues from, it is measured again from the beginning. Its figures and log are
      "any it had written": a single-cell run cancelled during its spontaneous campaign has no
      figure yet.
    - a COMPARISON measures nothing -- it draws records already measured -- so deleting one loses no
      measurement, and running it again redraws it (an interrupted one stays on disk and a re-run
      writes a new one). "Measuring again from the start" would be false for it.

    The study is named as a phrase, not as the body's bare word: "This single record" reads as "this
    one record". A null or unrecognised study is a "measurement", which claims nothing it cannot back.

    An fdt record still being WRITTEN -- by a run in another window or at a terminal -- looks exactly
    like an interrupted one until it finishes, and nothing stops it being deleted (there is no
    cross-process lock; ``_refuse_while_running`` sees only this process). A flat "the run was
    interrupted or it failed" was false for it, and deleting on the strength of it killed the live
    run, so every fdt branch says so and says what to do first.
    """
    if s.kind == "simulation":
        return (f"This training cache is UNFINISHED: {s.batches_done} committed batch(es). "
                "Deleting it throws those batches away and a later run starts from zero. "
                "(Only the batch count is known while a cache is running: the rows are "
                "recorded when the cache finishes.)")
    if s.kind == "fdt":
        why = ("its run was interrupted or failed, or it is still being written by a run in another "
               "window or at a terminal, which looks exactly the same until it finishes. If one is "
               "still running, let it finish or stop it before deleting.")
        if s.study == "comparison":
            return (f"This comparison record is UNFINISHED: {why} It holds whatever it had drawn "
                    "when it stopped. Deleting it loses nothing measured: running the comparison "
                    "again redraws it from the records it compares.")
        if s.points_done is None:
            held = "whatever it had measured when it stopped"
        elif s.points_planned is None:
            held = f"{s.points_done} operating point(s)"
        else:
            held = f"{s.points_done} of {s.points_planned} operating point(s)"
        return (f"This {_STUDY_PHRASES.get(s.study, 'measurement')} record is UNFINISHED: {why} "
                f"It holds {held} and any figures and log it had written. There is no resume for "
                "these runs -- deleting it means measuring again from the start.")
    return ("This record is UNFINISHED: its folder holds only what the run had written when it "
            "stopped.")


def _delete_prompt(s) -> tuple:
    """``(text, informative)`` for the confirmation: what goes, and what cannot come back.

    An UNFINISHED artifact names what is half-written, whichever kind it is -- ``finished`` is not
    ``complete``, and what a half-written record holds is the only thing deleting it destroys
    that a later run could not simply remake. ``_unfinished_note`` is where each kind's sentence is.

    ``s.complete and not s.finished`` rather than ``not s.finished`` alone: an INCOMPLETE row (a
    directory with no usable manifest) is not an artifact at all and ``finished`` is False for it. It
    never reaches this button (``_sync_actions`` enables Delete on complete rows only); the leftover
    sweep removes one only when it has no manifest file at all, and leaves one whose manifest this
    build cannot read for the operator (``ArtifactStore.remove_incomplete``).
    """
    lines = [f"id {s.id}"]
    if s.complete and not s.finished:
        lines.append(_unfinished_note(s))
    lines.append(f"This removes {s.path} and everything in it, and cannot be undone.")
    return f"Delete {s.kind} {s.label}?", "\n".join(lines)


class ArtifactScreen(QWidget):
    """The artifact browser: one kind at a time in a sortable table, over a status line.

    Emits ``store_changed`` after any change it makes to the store; MainWindow connects that to
    ``_refresh_store_pickers``, because ``StorePicker.restore_key`` silently keeps whatever is
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
        # True while refresh() is rebuilding the table, so _sync_actions ignores the selection
        # changes that rebuild produces. Set here, not in _build_actions: showEvent can refresh.
        self._relisting = False

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

        # One child today; the detail pane joins it beside the table.
        self.split = QSplitter(Qt.Horizontal)
        self.split.setChildrenCollapsible(False)
        self.split.addWidget(self.table)

        # Read-only, text only: the manifest rendered, then the run's records. A fixed-pitch
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

        # The pane's button row, kept as an attribute rather than a local: the "Lineage report…"
        # button mounts into THIS layout, beside Save, because both write a file about the selected
        # artifact.
        self.detail_actions = QHBoxLayout()
        self.detail_actions.addStretch(1)
        self.detail_actions.addWidget(self.btn_save)

        self.btn_lineage = QPushButton("Lineage report…")
        self.btn_lineage.setToolTip("Write this artifact and its parents, back through the chain, "
                                    "to a text file")
        self.btn_lineage.clicked.connect(self._lineage_report)
        self.detail_actions.addWidget(self.btn_lineage)

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
        # during __init__ would read the store at launch -- start-up work kept off that path (see
        # showEvent). Same construction-never-fires rule the Settings screen's radios follow.
        self.restore_settings(settings.settings())
        self.kind_combo.currentIndexChanged.connect(lambda _i: self.refresh())

    # ── the listing ───────────────────────────────────────────────────────────
    def kind(self) -> str:
        """The kind key currently selected."""
        return str(self.kind_combo.currentData())

    def showEvent(self, event):
        """List on every visit, so an artifact written since the last look is there. NOT from
        __init__: MainWindow builds this screen at launch, and a directory scan per launch is exactly
        the start-up cost that lost the taskbar icon on 2026-09-11."""
        super().showEvent(event)
        self.refresh()

    def refresh(self) -> bool:
        """Re-list the selected kind. Idempotent, and the only place the store is read for the table.

        Returns whether the store could be READ -- which is what ``_after_change`` needs in order to
        know whose sentence belongs on the status line. A caller that only re-lists may ignore it;
        the two signals wired to this method do.

        The rebuild is MARKED while it runs, because a ``QTreeWidget``'s ``clear()`` does not go
        straight from "this row is selected" to "nothing is selected": removing the selected row
        makes Qt move the selection to an ADJACENT row first, so the screen sees a spurious change
        to a DIFFERENT artifact halfway through. Acting on that rewrote the note box out of the
        neighbour's manifest and threw away a note typed and not yet applied. ``_sync_actions``
        ignores selection changes while the mark is set, and is called once by hand when the rebuild
        is over -- the same reason this method has always called ``_on_selection_changed`` itself.
        """
        self._relisting = True
        try:
            return self._relist()
        finally:
            self._relisting = False
            self._sync_actions()

    def _relist(self) -> bool:
        """``refresh``'s body, run under its rebuild mark. Nothing else calls this."""
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
            # here" and "I could not look" identical. The browser must tell them apart, so the
            # error and its class go on the status line and the table is emptied.
            self.table.set_rows(kind, [])
            self._on_selection_changed()
            self._set_status(f"Could not read the {kind} artifacts: {type(e).__name__}: {e}",
                             error=True)
            return False
        self.table.set_rows(kind, rows)
        self._apply_sort(kind)
        # Explicitly, not off the signal: set_rows leaves nothing selected, and a table that was
        # already empty emits no change -- so a stale pane would outlive the rows it described.
        self._on_selection_changed()
        if not rows:
            self._set_status("Nothing here yet.")
            return True
        bad = sum(1 for s in rows if not s.complete)
        self._set_status(f"{len(rows)} row(s): {len(rows) - bad} complete, {bad} incomplete.")
        return True

    def _apply_sort(self, kind: str) -> None:
        """Re-apply the remembered sort after a rebuild, clamped to THIS kind's column count: the
        columns differ per kind, so a column remembered while viewing posteriors can be past
        the end of a calibration's."""
        col, order = self._sort
        if 0 <= col < len(columns_for(kind)):
            self.table.apply_sort_state(col, order)

    # ── the detail view ───────────────────────────────────────────────────────
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
        """Everything the pane shows for one row, as one string -- which is what Save writes.

        Text only: the manifest rendered by the ONE renderer both front ends use
        (core/artifacts/report.py), the stale-folder note when there is one, then the run's records
        verbatim with their HH:MM:SS level stamps -- or, when there are none, WHICH kind of none it
        is. Two store reads and no other side effect.
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
            # read_log says "" for a run that said nothing and None for no file at all.
            # Silence is a record too, so the pane says so rather than leave the heading bare.
            parts.append(_EMPTY_LOG)
        else:
            parts.append(text)
        return "\n".join(parts)

    def _save_shown(self) -> None:
        """One half of the saveable run summary: what is on screen, to a file the operator picks.

        A FILE, never a store kind of its own -- a report DESCRIBES the store and must not be mistaken
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

    def _lineage_report(self) -> None:
        """The other half of the run summary: write the selected artifact's lineage to a file the
        operator names.

        A report DESCRIBES the store and is never a kind of its own in it: nothing here writes
        into the artifact root -- ``render_lineage`` reads manifests and this writes its text
        wherever the operator says. UTF-8 with LF endings, which is byte for byte what
        ``python -m core artifacts summary --out`` writes, so the document cannot say which front end
        made it.

        NOT guarded by ``BasePanel._running``: this is a read, and reading is never refused.

        The two checks below duplicate what keeps ``btn_lineage`` disabled in ``_sync_actions``
        (nothing selected, or a leftover with no manifest) -- deliberately: a disabled button is a
        courtesy that stops a click before it starts, but this method is also reachable directly (a
        test calls it, and a future caller might), and the guard is the actual rule. Do not delete
        these as "unreachable through the button" -- the button merely tries not to make them fire.
        """
        s = self.table.current_summary()
        if s is None:
            show_refusal(self, Refusal("No artifact is selected.", field="artifact"))
            return
        if not s.complete:
            # A leftover directory has no manifest, so there is no chain to walk -- and unlike the
            # note and the delete, this is not a rule anybody broke, so it goes on the status line.
            self._set_status(f"{s.dir_name} has no usable manifest ({s.reason}), so it has no "
                             f"lineage to report.", error=True)
            return
        try:
            text = render_lineage(self._resolved_store(), s.kind, s.id)
        except Refusal as exc:               # a parent that cannot be read at all
            show_refusal(self, exc)
            self._set_status(exc.message, error=True)
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save lineage report",
                                              f"{s.kind}-{s.name or s.id}-lineage.txt",
                                              "Text file (*.txt)")
        if not path:
            return
        if not path.lower().endswith(".txt"):
            path += ".txt"                   # the courtesy simulate_panel._save_video does for .mp4
        try:
            Path(path).write_text(text, encoding="utf-8", newline="\n")
        except OSError as e:
            self._set_status(f"Could not write the lineage report: {e}", error=True)
            return
        self._set_status(f"Wrote the lineage report for {s.kind} {s.label} to {path}.")

    # ── the actions ───────────────────────────────────────────────────────────
    def _build_actions(self) -> QWidget:
        """The Note box and the Delete/Sweep buttons: one row under the listing.

        A plain group box, not the model builder's sticky action bar -- this screen does not scroll a
        form of unbounded length, so nothing can fall below the fold.
        """
        box = QGroupBox("Actions")
        row = QHBoxLayout(box)
        # The (kind, id) whose note is in the box, so _sync_actions can tell a real change of
        # artifact from a re-list that lands back on the same one. None = no artifact's note.
        self._noted = None
        self.note_edit = QLineEdit()
        self.note_edit.setPlaceholderText(
            f"one line, at most {NOTE_MAX_CHARS} characters; empty clears it")
        # NO setMaxLength: the note rule REFUSES an over-long note and names both numbers. A
        # maxLength would silently truncate it instead -- a clamp; a bad input is refused at the
        # click, never repaired.
        self.btn_note = QPushButton("Set")
        self.btn_note.clicked.connect(self._set_note)
        self.btn_delete = QPushButton("Delete…")
        self.btn_delete.clicked.connect(self._delete)
        self.btn_sweep = QPushButton("Sweep this kind…")
        self.btn_sweep.setToolTip("Remove this kind's directories that have no manifest at all, and "
                                  "any loose file inside its folder")
        self.btn_sweep.clicked.connect(lambda: self._sweep(all_kinds=False))
        self.btn_sweep_all = QPushButton("Sweep all kinds…")
        self.btn_sweep_all.setToolTip("Remove every kind's directories that have no manifest at all, "
                                      "any loose file inside a kind's folder, and a legacy directory "
                                      "an older build wrote beside them")
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
        needs no selection at all. The lineage report is in the same boat as delete, not as Save --
        Save is enabled off the pane's TEXT, which an incomplete row also has (its reason, its path),
        but a lineage walk needs a resolvable artifact to start from, so it follows ``live`` exactly
        the way ``_lineage_report``'s own guard does: disabled with nothing selected, disabled on a
        leftover, enabled only on a complete row. The button and the guard read the same condition, so
        they cannot disagree.

        The note box is repopulated only when the selected ARTIFACT changed. Two reasons, and the
        second is the one that bites: a re-list clears the table's selection before the user picks a
        row again, so an unconditional rewrite discarded a note typed and not yet applied even when
        the selection came back to the same artifact. The transient no-selection state therefore
        leaves the box (and ``_noted``) alone -- it is disabled meanwhile -- while a selection that
        lands on a DIFFERENT artifact, or on a leftover that has no note at all, does rewrite it.
        """
        if self._relisting:
            # A rebuild's intermediate selections are Qt's bookkeeping, not the user's choice;
            # refresh() calls this once itself when the rebuild is done. See refresh's docstring.
            return
        s = self.table.current_summary()
        live = s is not None and s.complete
        self.note_edit.setEnabled(live)
        self.btn_note.setEnabled(live)
        self.btn_delete.setEnabled(live)
        self.btn_lineage.setEnabled(live)
        if s is None:
            return                  # the transient state a re-list passes through: keep the draft
        ref = (s.kind, s.id) if live else None
        if ref != self._noted:
            self.note_edit.setText(s.note if live else "")
            self._noted = ref

    def _after_change(self, said: str, error: bool = False) -> None:
        """Re-list the kind, tell the rest of the app, and THEN say what the action did.

        The action hands its sentence here instead of writing the status line itself, which is what
        makes the wrong order unrepresentable. Two ways to get it wrong, both closed here:

        * A sentence written BEFORE the re-list is lost: ``refresh`` ends by reporting the kind's row
          counts, so the operator would be told a row count in answer to a Delete.
        * A sentence written AFTER the re-list UNCONDITIONALLY hides a read failure. When the re-list
          cannot read the store, ``refresh`` has already put ``Could not read the <kind> artifacts:
          ...`` on the line, and THAT message wins. The action did succeed -- the store is the record
          of that -- but "there is nothing here" and "I could not look" are the one distinction this
          screen exists to keep apart, and a cheerful "Deleted ..." over an emptied table
          would erase it.

        So: the success sentence is set only when the re-list could read the store. The three
        StorePickers are told either way -- the change happened, whether or not the re-list saw it --
        and they are the WINDOW's to refresh, the one party that knows all of them.
        """
        read_ok = self.refresh()
        self._sync_actions()
        self.store_changed.emit()
        if read_ok:
            self._set_status(said, error=error)

    def _refuse_while_running(self, doing: str) -> bool:
        """True when a run is live: the status line says so and the caller returns.

        Every WRITE goes through this and nothing that only READS does. The wording is the window's
        own, from the two model-builder sites (ModelBuilderScreen._validate and _save); the status
        line is this screen's surface, so there is no dialog to dismiss either.
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
        """The note rule: one trimmed line, at most NOTE_MAX_CHARS; blank clears it.

        The rule is ``core.refusals.require_note`` and it runs at the click, so the command-line
        tool's own note flag (the ``"note"`` row of ``core/tool/fields.py``) refuses the same note
        with the same sentence; this front end only adds where to fix it.
        """
        if self._refuse_while_running("setting a note"):
            return
        s = self._selected("annotate")
        if s is None:
            return
        try:
            note = require_note("note", self.note_edit.text())
            self._resolved_store().set_note(s.kind, s.id, note)
        except Refusal as exc:      # a refused note text (field="note"), or the store's refusal
            show_refusal(self, exc)  # for a row gone since it was listed -- field="artifact", so
                                     # the box names the list, not the note box
            self._set_status(exc.message, error=True)
            return
        except Exception as e:      # noqa: BLE001 -- reported, never raised out of a click
            # The convention every other entry point here follows (refresh, _on_selection_changed):
            # a disk that will not take the manifest goes on the status line, not out of a slot into
            # the application's last-resort red box.
            self._set_status(f"Could not write the note on {s.kind} {s.label}: "
                             f"{type(e).__name__}: {e}", error=True)
            return
        # What was STORED, i.e. the TRIMMED text -- not what was typed. This box is the one place the
        # note is shown, so leaving an untrimmed draft in it would misreport the manifest.
        self.note_edit.setText(note)
        self._after_change(f"Set the note on {s.kind} {s.label}." if note
                           else f"Cleared the note on {s.kind} {s.label}.")

    def _dependents_refusal(self, store, s, deps) -> Refusal:
        """The sentence for a delete that nothing can perform: the artifact, then every artifact
        that depends on it WITH WHY.

        ``field="artifact"`` sends the operator to the list to delete the children first -- the same
        key the store's own dependents refusal carries. This one exists for the WORDING alone: the
        store's message ends "pass force=True to orphan them", which no front end offers, and a bare
        list of ids cannot say which dependent named this artifact and which the store found by
        fingerprint.
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
        """One artifact, no force, TWO outcomes.

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
        # Both directory scans -- dependents() here and the list() _dependents_refusal does to say
        # WHY each dependent depends -- under one broad guard, the convention refresh() and
        # _on_selection_changed already follow: an unreadable root must report on the status line,
        # not turn a Delete click into an unhandled slot exception and the app's red box.
        try:
            deps = store.dependents(s.kind, s.id)
            refusal = self._dependents_refusal(store, s, deps) if deps else None
        except Exception as e:                      # noqa: BLE001 -- reported, never swallowed
            self._set_status(f"Could not read what depends on {s.kind} {s.label}, so nothing was "
                             f"deleted: {type(e).__name__}: {e}", error=True)
            return
        if refusal is not None:
            show_refusal(self, refusal)
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
            store.delete(s.kind, s.id)              # never force, from either front end
        # A row that went, or dependents that arrived, since the list -- shown as the store wrote
        # it, fix sentence and all (its refusals are fielded).
        except Refusal as exc:
            show_refusal(self, exc)
            self._set_status(exc.message, error=True)
            return
        except Exception as e:                      # noqa: BLE001 -- reported, never raised
            # A held handle can leave the directory half removed, so the table and the pickers are
            # re-read rather than left describing a tree that has moved under them.
            self._after_change(f"Could not delete {s.kind} {s.label}: {type(e).__name__}: {e}",
                               error=True)
            return
        self._after_change(f"Deleted {s.kind} {s.label}.")

    def _incomplete(self, store, kind) -> tuple:
        """``(candidates, kept, problems)``: what a sweep would remove, what it will LEAVE ALONE
        although the table calls it incomplete, and any kind that could not be read at all -- an
        unreadable directory is not an empty one. The first two are ``[(kind, dir_name, reason)]``,
        the third is ``[str]``.

        Read off ``list``, so the confirmation shows exactly the rows the table calls incomplete --
        narrowed to the ONE reason a sweep may act on: no manifest.json at all. A manifest that
        exists and will not parse, or that declares another kind, is something nobody here
        understands; those rows stay in the listing and are named on the status line instead, which
        is why they are returned rather than silently dropped.

        ``kind is None`` means all kinds, EXPLICITLY -- never "whatever a falsy kind means". A
        truthiness test here is how a one-kind sweep could have widened to every kind.
        """
        out, kept, problems = [], [], []
        for k in (list(KIND_DIRS) if kind is None else [kind]):   # every kind, in KIND_DIRS order
            try:
                rows = store.list(k)
            except Exception as e:      # noqa: BLE001 -- an unreadable kind is reported, not fatal
                # With a next step, like every other sentence on this screen: an unreadable kind is
                # not an empty one, and the operator can act on it.
                problems.append(f"The {k} directory could not be read ({type(e).__name__}: {e}), so "
                                f"no {k} leftover can be swept; check that folder's permissions on "
                                f"disk and sweep again.")
                continue
            for row in rows:
                if row.complete:
                    continue
                target = out if row.reason == NO_MANIFEST_REASON else kept
                target.append((k, row.dir_name, row.reason))
        return out, kept, problems

    def _loose_and_legacy(self, store, kind) -> tuple:
        """``(loose, legacy, loose_problems, legacy_problems)``: the two categories no listing
        shows, loose files and legacy directories, beside ``_incomplete``'s directories, and each
        category's unreadable parts APART, so a "nothing to remove" line claims only what was read.

        ``loose`` is ``[(kind, LooseFile)]`` -- a FILE sitting directly inside a kind directory,
        which no artifact accounts for. ``_entries`` iterates directories only (store.py's
        ``for sub in ... if p.is_dir()``), so such a file is invisible to every listing and no front
        end could see or clear one until the sweeps learned to (a run stamped 20260915_153042 once
        left two in ``Artifacts/fdt``). ``loose_files`` reads the kind directory's own files and
        NEVER DESCENDS, so nothing inside a record's folder is reachable from here -- the same
        complement rule ``remove_incomplete`` has against ``delete``.

        ``legacy`` is ``[name]`` and is filled for the ALL-KINDS sweep alone: a legacy directory
        sits BESIDE the kind directories, under no kind, so the per-kind button has nothing to say
        about one. ``kind is None`` means all kinds, EXPLICITLY, exactly as ``_incomplete`` reads it.

        Both reads are guarded per kind: an unreadable directory is reported and does not stop the
        others (an unreadable kind is not an empty one).
        """
        loose, legacy, loose_problems, legacy_problems = [], [], [], []
        for k in (list(KIND_DIRS) if kind is None else [kind]):
            try:
                loose += [(k, f) for f in store.loose_files(k)]
            except Exception as e:      # noqa: BLE001 -- an unreadable kind is reported, not fatal
                loose_problems.append(f"The files in the {k} directory could not be read "
                                      f"({type(e).__name__}: {e}), so no {k} loose file can be "
                                      f"swept; check that folder's permissions on disk and sweep "
                                      f"again.")
        if kind is None:
            try:
                legacy = list(store.legacy_dirs())
            except Exception as e:      # noqa: BLE001 -- reported, never swallowed
                legacy_problems.append(f"The store root could not be read ({type(e).__name__}: {e}), "
                                       f"so no legacy directory can be swept; check its permissions "
                                       f"on disk and sweep again.")
        return loose, legacy, loose_problems, legacy_problems

    def _sweep(self, *, all_kinds: bool) -> None:
        """Remove every directory of this kind -- or of all kinds -- that has NO manifest at all,
        and the two categories ``_loose_and_legacy`` reads.

        TWO MORE CATEGORIES, so this screen can clear what the tool's ``artifacts sweep`` can clear,
        by the same rules: a loose FILE directly inside a kind directory, and -- from "Sweep all
        kinds…" ALONE -- a legacy directory beside the kind directories. Each is named in the
        confirmation and removed from the list that confirmation showed, one call per item through
        ``remove_loose`` / ``remove_legacy``, each of which applies the recency guard itself.

        The candidates are listed BEFORE anything is removed, and THAT LIST is what is removed: each
        entry goes by name through ``remove_incomplete``, the call that can only ever touch a
        directory ``_entries`` classifies as manifest-less -- the complement of ``delete``, which can
        only ever touch a real artifact.

        The removal used to re-scan, so the confirmation did not bind it -- a directory created
        while the dialog sat open was removed although the operator never saw it ("Removed 2 of 1
        leftover directories", probed). Anything that appeared since is simply not in the list, and
        the count sentence is true by construction.

        The removal is GUARDED, like every other entry point on this screen. Unguarded, a kind
        that could not be read threw out of the click after earlier kinds' leftovers were already
        gone: the application's red box, no status line, and ``store_changed`` never emitted, so the
        three pickers went on showing rows that had just been deleted.
        """
        # "leftovers", not "leftover directories": the sweep removes loose files and the legacy
        # directory too.
        if self._refuse_while_running("removing leftovers"):
            return
        store = self._resolved_store()
        # self.kind(), the one accessor everything else on this screen reads, and None ONLY for the
        # all-kinds button: reading currentData() here gave a second answer that disagreed with
        # kind() whenever it was falsy, and a falsy kind means "all kinds" downstream.
        kind = None if all_kinds else self.kind()
        cands, kept, dir_problems = self._incomplete(store, kind)
        loose, legacy, loose_problems, legacy_problems = self._loose_and_legacy(store, kind)
        problems = dir_problems + loose_problems + legacy_problems
        tail = (" " + " ".join(problems)) if problems else ""
        if kept:
            # Named, never silently dropped: the table shows these rows as incomplete, so a sweep
            # that left them there without a word would read as a sweep that failed.
            tail += (" Left alone, because a directory that carries a manifest.json is reported and "
                     "never removed: "
                     + "; ".join(f"{k}/{d} ({why})" for k, d, why in kept) + ".")
        if not cands and not loose and not legacy:
            # Each clause only where its read SUCCEEDED: the problems in the tail say what could not
            # be read, and "none here" beside "could not be read" contradicted itself. The legacy
            # clause only where the store root was READ: a per-kind sweep never looks beside the
            # kind directories, so "no legacy directory" would claim what nobody checked -- and a
            # crossval/ may well sit there. The tool's sentence does the same.
            said = [text for ok, text in (
                (not dir_problems, ("every directory of every kind carries a manifest.json"
                                    if kind is None else
                                    f"every {kind} directory carries a manifest.json")),
                (not loose_problems, ("no loose file sits inside a kind's folder" if kind is None
                                      else f"no loose file sits inside the {kind} directory")),
                (kind is None and not legacy_problems,
                 "no legacy directory sits beside the kind directories")) if ok]
            joined = (" and ".join(said) if len(said) < 3
                      else ", ".join(said[:-1]) + ", and " + said[-1])
            self._set_status("Nothing to remove" + (f": {joined}." if said else
                                                    " among what could be read.") + tail,
                             error=bool(problems))
            return
        total = len(cands) + len(loose) + len(legacy)
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Warning)
        box.setWindowTitle("Remove what no artifact accounts for")
        box.setText(f"Remove {total} item{'' if total == 1 else 's'} that no artifact accounts for?")
        box.setInformativeText(
            "\n".join([*(f"{k}/{d} — {why}" for k, d, why in cands),
                       *(f"{k}/{f.name} — a loose file inside the {k} directory, {f.size} bytes, "
                         f"part of no artifact" for k, f in loose),
                       *(f"{name}/ — a directory an older build wrote beside the kind directories"
                         for name in legacy)])
            + "\n\nA directory that carries a manifest.json of any kind is never touched by this. "
              "Nothing inside a record's own folder is offered either. Nor is anything that is "
              "still being written: an artifact's manifest is written last, so a run in flight — "
              "in another window or at a terminal — looks exactly like a leftover until it "
              "commits, and a recently written directory or file is refused rather than removed.")
        box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        box.setDefaultButton(QMessageBox.No)
        if box.exec() != QMessageBox.Yes:
            self._set_status("Nothing was removed." + tail, error=bool(problems))
            return
        files_gone, files_failed, legacy_gone, legacy_failed = [], [], [], []
        try:
            removed, failed = store.sweep_incomplete([(k, d) for k, d, _ in cands])
            # The confirmation binds the two new categories too: `loose` and `legacy` were read
            # BEFORE the dialog and THOSE LISTS are walked, so anything that appeared while it sat
            # open is simply not in them. One call per item, because neither remover has a batch
            # form -- each carries its own refusal (the recency guard among them), which is reported
            # and never stops the rest, exactly as sweep_incomplete does for directories.
            for k, f in loose:
                try:
                    store.remove_loose(k, f.name)
                except (Refusal, OSError) as e:        # StoreError is a Refusal, and fielded
                    files_failed.append((f"{k}/{f.name}",
                                         getattr(e, "message", f"{type(e).__name__}: {e}")))
                else:
                    files_gone.append(f.name)
            for name in legacy:
                try:
                    store.remove_legacy(name)
                except (Refusal, OSError) as e:
                    legacy_failed.append((name, getattr(e, "message", f"{type(e).__name__}: {e}")))
                else:
                    legacy_gone.append(name)
        except Exception as e:                  # noqa: BLE001 -- reported, never raised out of a click
            self._after_change(f"The sweep stopped part-way: {type(e).__name__}: {e}. Refresh to see "
                               f"what is left." + tail, error=True)
            return
        # One count per category that HAD candidates: a sweep that found only a loose file
        # does not report "Removed 0 of 0 leftover directories" before saying what it did.
        said = []
        if cands:
            said.append(f"Removed {len(removed)} of {len(cands)} leftover director"
                        f"{'y' if len(cands) == 1 else 'ies'}.")
            said += [f"{k}/{d} could not be removed: {why}." for k, d, why in failed]
        if loose:
            said.append(f"Removed {len(files_gone)} of {len(loose)} loose "
                        f"file{'' if len(loose) == 1 else 's'}.")
            said += [f"{what} could not be removed: {why}." for what, why in files_failed]
        if legacy:
            said.append(f"Removed {len(legacy_gone)} of {len(legacy)} legacy "
                        f"director{'y' if len(legacy) == 1 else 'ies'}.")
            said += [f"{what} could not be removed: {why}." for what, why in legacy_failed]
        self._after_change(" ".join(said) + tail,
                           error=bool(failed or files_failed or legacy_failed or problems))

    def _set_status(self, text: str, error: bool = False) -> None:
        """One line, with a ⚠ prefix when it is trouble -- ModelBuilderScreen._set_status's pattern."""
        self.status.setText(("⚠ " if error else "") + text)

    def _resolved_store(self):
        """This screen's store, or the process default -- StorePicker._resolved_store's seam."""
        from core.artifacts import resolve_store
        return resolve_store(self._store)

    # ── persistence ───────────────────────────────────────────────────────────
    def save_settings(self, qs) -> None:
        """The kind and the sort, and nothing else.

        MainWindow._save_state calls this BY NAME: ``_all_panels()`` is panel-typed and this screen is
        a QWidget, so it gets no sweep for free -- which is also what keeps it out of
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
