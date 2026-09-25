"""Reading an ``fdt`` record back onto a panel: what it says, and where its pictures are.

TWO SCREENS, ONE VIEWER. The FDT and CrossVal panels both show an earlier run (spec §5.4) and both
must show it the same way, so the wording (``record_summary``), the listing (``record_figures``) and
the viewer itself -- what the line describes, when figures open, which tabs close (``RecordViewer``)
-- live here once, and each panel keeps a one-line ``_show_record``. Nothing here touches the store's
load path: a selection change is a combo event, and ``load_fdt`` verifies the payload hash -- on a
sweep's ``data.h5`` that is the whole study re-read for one keystroke. The line needs the manifest
(``store.get``) and the figures the directory (``store.path``), both answered from manifests alone.

Everything here feeds a READ-ONLY label or an image tab, so none of it may raise: a formatter that
throws inside a ``currentIndexChanged`` slot brings the panel down with it, and a panel that fails in
``__init__`` takes ``MainWindow`` and the whole launch with it (the hazard
``CrossValPanel._on_cell_changed`` documents).
"""
import os
from pathlib import Path

from PySide6.QtCore import QObject, Qt, QTimer
from PySide6.QtWidgets import QLabel

from core.artifacts import default_store

# The ONE title formatter. The live watcher names a figure this way as it lands
# (core/gui/plot_watcher.py), so a re-opened figure must be named by the same function or the same
# picture carries two names depending on whether you watched it arrive.
from ..plot_watcher import _title as png_title


def details_label() -> QLabel:
    """A read-only, word-wrapped, plain-text block under a record picker.

    Word-wrapped and PlainText for the reason the inference tabs' derived labels are
    (core/gui/panels/inference/base.py): these carry generated strings that can be long -- a cell
    path and a settings line -- and an unwrapped label widens the whole controls column, which is the
    defect that once put a permanent horizontal scrollbar on the crossval panel.
    """
    lab = QLabel()
    lab.setWordWrap(True)
    lab.setTextFormat(Qt.PlainText)
    return lab


def record_summary(m) -> str:
    """The four facts spec §5.4 names, off an ``fdt`` manifest: when it was written and under which
    seed, its cell, its settings, and its notices -- plus a line when the run did not finish.

    Written to be unraisable on a manifest that is missing anything: ``manifest.validate`` checks
    body KEY SETS and top-level types only, so every value below can legitimately be null and a
    hand-edited one can be anything at all -- which is why each block is type-checked before it is
    read, and a ``notices`` that is not a list is taken as one sentence or as none, never iterated.

    A NULL IS NEVER PRINTED. A null in a record means "not recorded" or "does not apply" -- every
    sweep records ``skip_sanity`` and ``confirm_production`` null, because they are run_fdt's
    arguments and a sweep takes neither -- so a setting whose value is null is left out, and a count
    line is written only when the counts it states are known: "3 of None probe frequencies came back
    blank" is not a sentence that is true.
    """
    body = getattr(m, "body", None)
    body = body if isinstance(body, dict) else {}
    created, seed = getattr(m, "created", None), body.get("seed")
    lines = [f"Written {created if created is not None else '?'}"
             + (f" · seed {seed}" if seed is not None else " · seed not recorded")]
    inputs = getattr(m, "inputs", None)
    cell = inputs.get("cell") if isinstance(inputs, dict) else None
    lines.append(f"Cell: {cell['path']}" if isinstance(cell, dict) and cell.get("path")
                 else "Cell: (no cell file recorded)")
    settings = body.get("settings")
    if isinstance(settings, dict):
        known = [f"{k}={settings[k]}" for k in sorted(settings, key=str) if settings[k] is not None]
        if known:
            lines.append(", ".join(known))
    points = body.get("points")
    points = points if isinstance(points, dict) else None
    offgrid = body.get("offgrid")
    offgrid = offgrid if isinstance(offgrid, dict) else {}
    if offgrid.get("blanks") is not None and offgrid.get("of") is not None:
        # A sweep's count is over every probe of every point it MEASURED on its common grid (the
        # whole-piece review's M2), so a bare "of N probe frequencies" would read as one grid's size
        # when it is the whole study's.
        where = " across the operating points it measured" if points is not None else ""
        lines.append(f"{offgrid['blanks']} of {offgrid['of']} probe frequencies{where} came back "
                     f"blank.")
    if points is not None and points.get("done") is not None and points.get("planned") is not None:
        failed, param = points.get("failed"), points.get("param")
        lines.append(f"{points['done']} of {points['planned']} operating points measured"
                     + (f" ({failed} failed)" if failed is not None else "")
                     + (f", sweeping {param}" if param is not None else "") + ".")
    notices = body.get("notices")
    if isinstance(notices, str):
        notices = [notices]
    for notice in (notices if isinstance(notices, list) else []):
        if notice is not None:
            lines.append(f"Notice: {notice}")
    if not body.get("complete", True):
        lines.append("This run did not finish.")
    return "\n".join(lines)


def record_figures(record_dir) -> list:
    """``[(title, absolute path)]`` for every PNG in ``<record>/figures``, in name order.

    ``figures/`` is the artifact directory's one subdirectory and holds only PNGs
    (``core/artifacts/store.py``), so a glob is the whole listing -- ``data.h5`` and the rest are
    payloads and sit directly in the record directory, not here. A record with no figures -- a run
    that was cancelled before it drew one, whose folder E2 keeps -- is an empty list, never an error.
    """
    figs = Path(record_dir) / "figures"
    if not figs.is_dir():
        return []
    return [(png_title(p.name), str(p)) for p in sorted(figs.glob("*.png"))]


def _same_file(path) -> "str | None":
    """One spelling per file, so two routes to the same PNG compare equal."""
    return None if path is None else os.path.normcase(os.path.abspath(str(path)))


class RecordViewer(QObject):
    """The saved-run viewer under one record picker: the line it fills, and the figure tabs it opened.

    THE LINE FOLLOWS EVERY INDEX CHANGE -- a pick, a refresh, a delete in the Artifacts screen, a
    run's result slot moving the picker -- because it must describe whatever the picker shows. It
    costs one manifest scan (``store.get``).

    FIGURES OPEN ONLY ON A USER'S PICK. ``QComboBox.activated`` is emitted for the keyboard and the
    popup and never for a programmatic index change, so figures open from that signal alone.
    ``StorePicker.refresh`` -- which MainWindow runs on every change the Artifacts screen makes, and
    the Rescan button runs -- passes three index changes on its way back to the same record (-1, row
    0, the record). Taking those for choices opened a record nobody chose, re-opened tabs the user
    had closed and moved the current tab. ``StorePicker`` itself is left alone: the inference tabs
    rely on ``currentIndexChanged`` covering ``refresh()``. A launch opens nothing (F47): the viewer
    is built after restore_settings, fills the line once, and nothing emits ``activated`` there.

    ITS OWN TABS CLOSE when the SETTLED selection is no longer the record they show -- a delete in the
    Artifacts screen, a run's result slot moving the picker -- and when a run finishes
    (``close_figures``, which each panel hands its dispatch). "Settled" is judged one event-loop turn
    after an index change, by one coalesced ``QTimer.singleShot(0, ...)`` comparing the picker's key
    with the key whose figures are open, so a refresh's passing states are never taken for a move.
    Only the viewer's own tabs ever go (F48, never ``clear_all()``): a comparison drawn on the stack
    (Task 40) and a run's figures survive every pick.

    NO FILE IS OPENED TWICE. A PNG another tab already shows -- a run's watcher put it up -- is left to
    that tab. A page is recorded only if the stack actually grew, so a stubbed ``add_png`` never gets
    a neighbour's tab adopted as the viewer's.

    ``busy`` answers "is this panel's run live?". While it is, nothing opens: the stack then holds
    what the run's watcher is landing.
    """

    def __init__(self, picker, label, stack, *, busy):
        # Parented to the picker, so the coalesced timer dies with the widget it reads.
        super().__init__(picker)
        self._picker, self._label, self._stack, self._busy = picker, label, stack, busy
        self._opened: list = []              # the pages this viewer opened and may close (F48)
        self._open_key = None                # the record those pages show, or None
        self._settle_queued = False
        self.show_record(figures=False)      # F47: the restored selection's line, no figures
        picker.combo.currentIndexChanged.connect(self._on_index_changed)
        picker.combo.activated.connect(self._on_activated)

    def show_record(self, *, figures: bool = True) -> None:
        """Describe the selected record on the line and -- with ``figures`` and no run live -- open its
        figures in place of the ones this viewer opened before. The panels' ``_show_record``."""
        # key(), never selected()[1]: that is True for an EMPTY picker too, with no sentinel to be new.
        ref = self._picker.key()
        if not ref:
            self._label.setText("")
        else:
            try:
                self._label.setText(record_summary(default_store().get("fdt", ref)))
            except Exception as e:                   # noqa: BLE001 -- see the module docstring
                self._label.setText(f"(could not read the record: {e})")
        if figures and not self._busy():
            self._open(ref)

    def close_figures(self) -> None:
        """Close the tabs this viewer opened that are still open, and nothing else on the stack.

        The pages are looked for among the tabs still OPEN, never handed to Qt: a tab the user closed
        by hand has had its page deleted, and touching that wrapper raises."""
        mine = {id(page) for page in self._opened}
        for i in reversed(range(self._stack.count())):
            page = self._stack.widget(i)
            if id(page) in mine:
                self._stack.removeTab(i)             # what FigureStack's own close button does
                page.deleteLater()
        self._opened, self._open_key = [], None

    def _on_index_changed(self, _index) -> None:
        self.show_record(figures=False)
        if not self._settle_queued:
            self._settle_queued = True
            QTimer.singleShot(0, self, self._settle)

    def _on_activated(self, _index) -> None:
        if not self._busy():
            self._open(self._picker.key())

    def _settle(self) -> None:
        self._settle_queued = False
        if self._open_key is not None and self._picker.key() != self._open_key:
            self.close_figures()

    def _open(self, ref) -> None:
        self.close_figures()
        if not ref:
            return
        try:
            record_dir = default_store().path("fdt", ref)
        except Exception:                            # noqa: BLE001 -- the line already says why
            return
        shown = {_same_file(self._stack.png_path(i)) for i in range(self._stack.count())}
        for title, png in record_figures(record_dir):
            if _same_file(png) in shown:
                continue
            before = self._stack.count()
            self._stack.add_png(title, png)
            if self._stack.count() > before:
                self._opened.append(self._stack.widget(self._stack.count() - 1))
        self._open_key = ref
