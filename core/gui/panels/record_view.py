"""Reading an ``fdt`` record back onto a panel: what it says, and where its pictures are.

TWO SCREENS, ONE WORDING. The FDT and CrossVal panels both show an earlier run (spec §5.4) and both
must show it the same way, so the rendering lives here rather than twice. Nothing here touches the
store's load path: a selection change is a combo event, and ``load_fdt`` verifies the payload hash --
on a sweep's ``data.h5`` that is the whole study re-read for one keystroke. The callers pass the
manifest and the directory, which ``store.get`` and ``store.path`` answer from manifests alone.

Every function here feeds a READ-ONLY label or an image tab, so none of them may raise: a formatter
that throws inside a ``currentIndexChanged`` slot brings the panel down with it, and a panel that
fails in ``__init__`` takes ``MainWindow`` and the whole launch with it (the hazard
``CrossValPanel._on_cell_changed`` documents).
"""
import os
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel

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
    """
    body = getattr(m, "body", None)
    body = body if isinstance(body, dict) else {}
    seed = body.get("seed")
    lines = [f"Written {getattr(m, 'created', '?')}"
             + (f" · seed {seed}" if seed is not None else " · seed not recorded")]
    inputs = getattr(m, "inputs", None)
    cell = inputs.get("cell") if isinstance(inputs, dict) else None
    lines.append(f"Cell: {cell['path']}" if isinstance(cell, dict) and cell.get("path")
                 else "Cell: (no cell file recorded)")
    settings = body.get("settings")
    if isinstance(settings, dict) and settings:
        lines.append(", ".join(f"{k}={settings[k]}" for k in sorted(settings, key=str)))
    points = body.get("points")
    offgrid = body.get("offgrid")
    if isinstance(offgrid, dict):
        # A sweep's count is over EVERY planned point's slot on its common grid (P78), so a bare
        # "of N probe frequencies" would read as one grid's size when it is the whole study's.
        where = " across its operating points" if isinstance(points, dict) else ""
        lines.append(f"{offgrid.get('blanks')} of {offgrid.get('of')} probe frequencies{where} came "
                     f"back blank.")
    if isinstance(points, dict):
        lines.append(f"{points.get('done')} of {points.get('planned')} operating points measured "
                     f"({points.get('failed')} failed), sweeping {points.get('param')}.")
    notices = body.get("notices")
    if isinstance(notices, str):
        notices = [notices]
    for notice in (notices if isinstance(notices, list) else []):
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


def reopen_figures(stack, opened: list, record_dir) -> list:
    """Close the figure tabs a viewer ``opened`` for the previous selection, open ``record_dir``'s,
    and return the pages this call opened -- the list the caller keeps and hands back next time.
    ``record_dir`` None (nothing selected, or a record the store could not read) opens nothing.

    ONLY ITS OWN TABS GO (ruling F48). The stack is shared: a run's watcher lands figures in it and a
    comparison is drawn into it (Task 40), and clearing it to show the next selection would throw
    the comparison away the moment you pick the next record to append to it. A tab the user already
    closed is skipped: the pages are looked for among the tabs still OPEN, never handed to Qt,
    because a closed tab's page is deleted and touching its wrapper raises.

    NO FILE IS OPENED TWICE. A PNG another tab already shows is left to that tab -- the run that just
    finished put its figures up through its watcher, and choosing its record again must not double
    them. A page is recorded only if the stack actually grew, so a stubbed ``add_png`` never gets a
    neighbour's tab adopted as the viewer's.
    """
    mine = {id(page) for page in opened}
    for i in reversed(range(stack.count())):
        page = stack.widget(i)
        if id(page) in mine:
            stack.removeTab(i)                   # what FigureStack's own close button does
            page.deleteLater()
    if record_dir is None:
        return []
    shown = {_same_file(stack.png_path(i)) for i in range(stack.count())}
    added = []
    for title, png in record_figures(record_dir):
        if _same_file(png) in shown:
            continue
        before = stack.count()
        stack.add_png(title, png)
        if stack.count() > before:
            added.append(stack.widget(stack.count() - 1))
    return added
