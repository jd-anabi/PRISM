"""Comparing saved FDT records (piece 5, spec §7; decision E8).

A comparison is itself an ``fdt`` record, with ``body.study = "comparison"`` (§7.3): ``body.compared``
names the mode and the records it drew, its figures are its output, and its ``data.h5`` holds the
common grid and the interpolated curves. It names its sources in the BODY and not in ``parents``,
because the parents block is a flat ``{key: id}`` map read as ``m.parents.get(pk) == id_``, so it
cannot carry an arbitrary number of ids without widening the store's contract for every kind (spec
§1.2). Deleting a record a comparison drew is therefore not refused; ``report.render_lineage``
resolves the body's ids and prints ``MISSING`` for one the store no longer holds -- which is why the
body records each source's RESOLVED id, never the ref the caller typed: the store resolves a name as
well as an id, and a later run that took a deleted source's name must not read as that source.

ONE PUBLIC ENTRY, four modes. Every run detects its own resonance and builds its grid around it, so
two runs of the same cell land on different frequencies; each mode therefore goes through the same
three steps -- load and check the records, interpolate onto a common grid, write the record -- and
differs only in what it draws. The modes register themselves in ``_DRAWERS``.

REFUSED BEFORE THE RECORD OPENS. The record is progressive (spec §2.2): it exists from the moment
its writer is entered, and an exception after that KEEPS it (E2). So everything a comparison can
refuse from what it was given -- the mode, its settings, the arity, each record -- is refused before
``open_record``, and a refused comparison leaves nothing on disk. An interrupted one keeps its
record, unfinished; nothing resumes it, and re-running draws a new one.

BLANKS ARE NEVER INTERPOLATED ACROSS. A frequency a run could not measure comes back blank (E9), and
a comparison that blended over one would put back exactly the fabricated tail the off-grid fix
removed. A target point whose bracketing samples include a blank is blank, and the record says how
many there were, in its notices and on the axis.

LIGHT ON PURPOSE: nothing here imports ``core.diagnostics``, ``core.orchestrator`` or an ``sbi``
submodule. A comparison draws saved numbers; the inference stack would cost every run about two
seconds and two false pytensor "g++" lines (tests/test_fdt_compare.py pins it in a fresh interpreter).
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from core.artifacts import StoreError, resolve_store
from core.refusals import refuse, require_finite, require_positive
from core.runs import public_entry

# The comparison's own report -- which records it drew, how much of the common grid was usable -- is
# information (piece 3, V4); the record's log.txt keeps it beside the manifest.
log = logging.getLogger(__name__)

#: ``mode -> (the study its records must carry, the fewest records it draws, the most or None)``.
#: The arity is a real input class and not a nicety: one cell is not a comparison, and a sweep drawn
#: as a single-cell run reads a layout that is not there. A ceiling as well as a floor (F59), because
#: a mode that draws one record, or two, would otherwise be handed more and silently draw the first.
MODE_RULES: dict[str, tuple[str, int, "int | None"]] = {
    "cells": ("single", 2, None),
    "repeats": ("single", 2, None),
    "renormalise": ("single", 1, 1),
    "sweeps": ("sweep", 2, 2),
}

#: ``mode -> the keyword settings it accepts``. A setting a mode has no use for is refused rather
#: than recorded: ``compare`` writes the options it was given into ``body.settings``, and a typo
#: recorded there would describe a comparison that never happened.
MODE_OPTIONS: dict[str, tuple[str, ...]] = {
    "cells": (), "repeats": (), "renormalise": ("prefactor",), "sweeps": ("at",),
}

#: ``mode -> drawer(w, records, *, sink, **options) -> (results, notices)``. Filled by the mode
#: modules below as each lands; a mode with no drawer is refused, which also covers a mode string
#: that reached here without passing through the tool's own choices.
#:
#: A drawer runs inside the ENTERED writer and keeps its contract: every figure goes through
#: ``sink`` and the numbers through ``write_curves``, and ``results`` holds finite numbers only
#: (``_num``) -- a manifest refuses NaN, and this ratio legitimately carries it (spec §2.3).
_DRAWERS: dict = {}

#: What a single-cell record's ``data.h5`` calls its two curves (spec §2.3). The names are
#: ``cross_validation._fdt_measure``'s own vocabulary, which the sweep's file already uses, so one
#: reader's words describe both files.
OMEGA_GRID, RATIO = "omega_grid", "T_eff_over_T"

#: Appended to the frequency axis whenever a drawn curve has blanks, so the picture says what the
#: record's notices say.
BLANK_AXIS_NOTE = "gaps are frequencies a run did not measure; they are never interpolated across"


@dataclass(frozen=True)
class Curve:
    """One record's ratio curve, on its OWN frequency grid."""
    id: str
    name: str
    label: str              # the legend entry: the cell file's stem, else the record's own label
    omegas: np.ndarray
    ratio: np.ndarray
    omega_0: float
    prefactor: float


def _num(x):
    """A finite float, or None -- a manifest refuses a non-finite number in the body
    (``manifest._check_finite``) and this ratio legitimately carries NaN (spec §2.3). The same rule
    as ``orchestrator._num``, restated here so this module needs no orchestrator import."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _label_of(rec) -> str:
    """A record's legend entry: its cell file's stem, else its name, else its id. ``inputs.cell`` is
    None for a record written without a configuration, so the fallbacks are not decoration."""
    cell = (rec.manifest.inputs or {}).get("cell")
    if isinstance(cell, dict) and cell.get("path"):
        return Path(str(cell["path"])).stem
    return rec.name or rec.id


def _checked_options(mode: str, options: dict) -> dict:
    """The mode's settings, judged BEFORE the record opens (F57) and returned as the floats the
    record will carry. The first manifest is written at ``__enter__`` with these values in
    ``body.settings``, so one the manifest cannot hold -- a NaN -- would otherwise fail inside the
    writer and leave a folder behind, and a blank or non-positive constant would be refused only once
    the record existed. The drawers keep their own check as a backstop."""
    checked = dict(options)
    if mode == "renormalise":
        checked["prefactor"] = require_positive("prefactor", options.get("prefactor"))
    elif mode == "sweeps" and options.get("at") is not None:
        checked["at"] = require_finite("slice_at", options["at"])
    return checked


def _held(store, ref: str) -> bool:
    """Does ``ref`` name or id a record the store can resolve? Asked apart from ``load_fdt`` because
    that call refuses a missing record and an INCONSISTENT one (a payload that no longer hashes to
    its manifest) with one exception type, and only the first is the comparison's to reword."""
    try:
        store.get("fdt", ref)
    except StoreError:
        return False
    return True


def load_records(store, refs, mode: str) -> list:
    """The records ``refs`` names, checked against ``mode``'s own rules before anything is drawn.

    Refuses, each under ``compare_records`` and naming the record by its own name (or id), never by
    the ref it was given: too few or too many records for the mode, a ref that names no record, a
    record of the wrong study, a record whose run did not finish, and a record with no numbers beside
    its manifest. An unfinished record (E2 leaves those on disk with a valid manifest) can reach here
    only from the tool, by id or name: the window's picker offers finished records only.
    """
    store = resolve_store(store)
    want_study, fewest, most = MODE_RULES[mode]
    refs = [str(r) for r in refs]
    if len(refs) < fewest:
        refuse("compare_records",
               f"The saved runs to compare must be at least {fewest} for a {mode} comparison; "
               f"got {len(refs)}.")
    if most is not None and len(refs) > most:
        refuse("compare_records",
               f"The saved runs to compare must be at most {most} for a {mode} comparison; "
               f"got {len(refs)}.")
    out = []
    for ref in refs:
        if not _held(store, ref):
            # The store's own refusal carries field="artifact", which would send the window to the
            # Artifacts screen and name no flag at all; the control that answers it is this list (F58).
            refuse("compare_records", f"The saved runs to compare must exist; {ref!r} names no fdt record.")
        rec = store.load_fdt(ref)
        who = rec.name or rec.id
        study = rec.body.get("study")
        if study != want_study:
            refuse("compare_records",
                   f"The saved runs to compare must all be {want_study} runs for a {mode} "
                   f"comparison; {who!r} is a {study} run.")
        if not rec.body.get("complete"):
            refuse("compare_records",
                   f"The saved runs to compare must have finished; {who!r} did not (it was "
                   f"interrupted, or it is still running), so the numbers it holds are partial.")
        if rec.data_path is None or not Path(rec.data_path).is_file():
            refuse("compare_records",
                   f"The saved runs to compare must hold their numbers; {who!r} has no data file "
                   f"beside its manifest, so there is nothing to draw.")
        out.append(rec)
    return out


def curve_of(rec) -> Curve:
    """A single-cell record's ratio curve, read back from its ``data.h5``."""
    with h5py.File(rec.data_path, "r") as h5:
        for key in (OMEGA_GRID, RATIO):
            if key not in h5:
                refuse("compare_records",
                       f"The saved runs to compare must hold their numbers; the data file of "
                       f"{(rec.name or rec.id)!r} has no {key!r} in it.")
        omegas = np.asarray(h5[OMEGA_GRID][...], dtype=np.float64)
        ratio = np.asarray(h5[RATIO][...], dtype=np.float64)
        omega_0 = float(h5.attrs.get("omega_0", math.nan))
        prefactor = float(h5.attrs.get("prefactor", math.nan))
    return Curve(id=rec.id, name=rec.name, label=_label_of(rec), omegas=omegas, ratio=ratio,
                 omega_0=omega_0, prefactor=prefactor)


def common_grid(curves: list) -> np.ndarray:
    """Log-spaced over the INTERSECTION of the curves' spans, with N the smallest point count among
    them (spec §7.2). Curves that share no band at all are refused: an empty picture is not an
    answer.

    The two end points are the span's own numbers, not their round trip through the logarithm:
    ``exp(log(3.0))`` is one ulp above 3.0, and an end point one ulp outside a record's span is
    outside it for ``interpolate_onto`` -- a blank the run never had.
    """
    lo = max(float(np.min(c.omegas)) for c in curves)
    hi = min(float(np.max(c.omegas)) for c in curves)
    n = min(int(np.size(c.omegas)) for c in curves)
    if not (lo > 0.0) or not (hi > lo) or n < 2:
        refuse("compare_records",
               f"The saved runs to compare must overlap in frequency; the intersection of their "
               f"grids is [{lo:g}, {hi:g}] over {n} point(s).")
    grid = np.exp(np.linspace(math.log(lo), math.log(hi), n))
    grid[0], grid[-1] = lo, hi
    return grid


def interpolate_onto(grid, omegas, values) -> np.ndarray:
    """``values`` on ``grid``, linear in log-omega. A target point whose bracketing samples include a
    blank -- or that lies outside the source's span -- is blank.

    The blank mask is carried by an INDICATOR interpolated the same way, never by trusting NaN to
    propagate through ``np.interp``: numpy retries a non-finite result from the other bracket, so
    "the NaN comes out anyway" is not a property to lean on for the guarantee §7.2 states.
    """
    x = np.log(np.asarray(omegas, dtype=np.float64))
    y = np.asarray(values, dtype=np.float64)
    order = np.argsort(x)
    x, y = x[order], y[order]
    t = np.log(np.asarray(grid, dtype=np.float64))
    finite = np.isfinite(y)
    keep = np.interp(t, x, finite.astype(np.float64), left=0.0, right=0.0)
    out = np.interp(t, x, np.where(finite, y, 0.0), left=np.nan, right=np.nan)
    out[keep < 1.0] = np.nan
    return out


def blank_notice(values: list) -> tuple:
    """``(blanks, notices)`` -- how many common-grid points are blank in at least one curve, and the
    sentence the record keeps when there are any (spec §7.2: "the drawing says so, in the axis label
    and in the record's notices")."""
    stack = np.stack([np.asarray(v, dtype=np.float64) for v in values], axis=0)
    blanks = int((~np.isfinite(stack)).any(axis=0).sum())
    if blanks == 0:
        return 0, []
    return blanks, [f"{blanks} of {stack.shape[1]} points on the common grid are blank in at least "
                    f"one run: a point whose bracketing samples include a blank is left blank, "
                    f"never interpolated across."]


def open_record(store, mode: str, records: list, *, name: str, note: str, settings: dict):
    """The comparison's own record, CREATED and not entered.

    ``create`` mints the id and runs ``assert_name_free`` before anything is spent, and the body is
    set between ``create()`` and the ``with``: the progressive mode's first manifest carries every
    body key, so the facts already known have to be there before ``__enter__`` writes it (spec §2.2).
    ``compared`` carries each LOADED record's own id and name, whatever ref reached ``compare``.
    """
    w = store.create("fdt", None, name=name, note=note)
    w.body = {
        "study": "comparison",
        "settings": {"mode": mode, "records": len(records), **settings},
        "seed": None,          # a comparison measures nothing; it draws what was measured
        "grid": None, "points": None, "offgrid": None,
        "notices": [],
        "compared": {"mode": mode,
                     "records": [{"kind": "fdt", "id": r.id, "name": r.name} for r in records]},
        "complete": False,
        "results": None,
    }
    return w


def write_curves(w, grid, values: list, labels: list, *, constants: list) -> None:
    """The comparison's ``data.h5``: the common grid and one interpolated curve per record, each
    carrying the label the picture used (spec §7.3).

    ``constants`` is one ``(omega_0, prefactor)`` pair per curve -- the constants THAT curve's numbers
    were computed with, which for a renormalised curve is not its source's recorded prefactor. The
    layout contract puts ``omega_0`` and ``prefactor`` at the root of every ``data.h5``; a comparison
    has no single value for either, so the root holds NaN (not applicable) and each curve carries its
    own (F55). Required, not defaulted: a drawer that forgot them would write NaN for every curve and
    nothing would say so. ``strict``, so a drawer that hands one list short is a loud bug and never a
    curve silently dropped from the file.
    """
    with h5py.File(w.payload("data.h5"), "w") as h5:
        h5.attrs["study"] = "comparison"
        h5.attrs["mode"] = w.body["compared"]["mode"]
        h5.attrs["omega_0"] = math.nan
        h5.attrs["prefactor"] = math.nan
        h5.create_dataset("omega_common", data=np.asarray(grid, dtype=np.float64))
        group = h5.create_group("curves")
        for i, (label, vals, (omega_0, prefactor)) in enumerate(zip(labels, values, constants,
                                                                    strict=True)):
            d = group.create_dataset(f"{i:03d}", data=np.asarray(vals, dtype=np.float64))
            d.attrs["label"] = str(label)
            d.attrs["omega_0"] = float(omega_0)
            d.attrs["prefactor"] = float(prefactor)


def finish_record(w, *, results: dict, notices) -> None:
    """Fill in what the drawing found and re-write the manifest. ``complete`` is the writer's to set,
    on the clean exit (spec §2.2 step 4).

    Called once, inside the entered writer and after the drawer has handed out its figures and its
    payload, so the one refresh a comparison makes never precedes what it records."""
    w.body["results"] = results
    w.body["notices"] = list(notices)
    w.refresh()


@public_entry
def compare(mode: str, refs, *, name: str = "", note: str = "", fig_sink=None, store=None,
            **options):
    """Draw one comparison of saved ``fdt`` records and write it as a record of its own.

    :param mode: one of ``MODE_RULES`` -- cells, repeats, renormalise, sweeps.
    :param refs: the records to draw, by name or id. The arity and the study each mode needs are
                 ``MODE_RULES``', and are refused before anything is created.
    :param options: the mode's own settings (``MODE_OPTIONS``): renormalise's ``prefactor``, sweeps'
                 ``at``. A setting the mode has no use for is refused, and a value the mode cannot use
                 is refused before the record opens.
    :returns: the comparison record.
    """
    store = resolve_store(store)
    if mode not in MODE_RULES:
        refuse("compare_records",
               f"The saved runs to compare cannot be drawn as a {mode!r} comparison; this build "
               f"draws {', '.join(sorted(MODE_RULES))}.")
    unknown = sorted(set(options) - set(MODE_OPTIONS[mode]))
    if unknown:
        refuse("compare_records",
               f"The saved runs to compare were given settings a {mode} comparison has no use for: "
               f"{', '.join(unknown)}.")
    drawer = _DRAWERS.get(mode)
    if drawer is None:
        refuse("compare_records",
               f"The saved runs to compare cannot be drawn: this build has no {mode} comparison "
               f"(it draws {', '.join(sorted(_DRAWERS)) or 'none yet'}).")
    options = _checked_options(mode, options)
    records = load_records(store, refs, mode)
    log.info(f"Comparing {len(records)} record(s) as a {mode} comparison: "
             f"{', '.join(r.name or r.id for r in records)}")
    w = open_record(store, mode, records, name=name, note=note, settings=options)
    with w:
        # The record's id, on screen and in its own log the moment it exists: the tool's interrupt
        # note is fixed text and points back at this line, which is all that can name the record
        # an interrupt leaves behind.
        log.info(f"Writing comparison record {w.id} at {w.dir}")
        results, notices = drawer(w, records, sink=w.fig_sink(fig_sink), **options)
        for sentence in notices:
            log.warning(sentence)
        finish_record(w, results=results, notices=notices)
    return store.load_fdt(w.id)
