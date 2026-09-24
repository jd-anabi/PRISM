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
import re
from dataclasses import dataclass, replace
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


def _cell_key(rec) -> tuple:
    """The cell a record measured, as the components of the path its manifest records with the
    file's stem last -- ``Cells/shm/default.txt`` is ``("Cells", "shm", "default")`` -- or, for a
    record that names no cell, its name (or id) alone, the same fallback ``_label_of`` takes. Split
    on either separator: a cell outside ``Resources/`` is recorded as an absolute path, which on
    Windows is written with backslashes."""
    cell = (rec.manifest.inputs or {}).get("cell")
    if isinstance(cell, dict) and cell.get("path"):
        parts = [p for p in re.split(r"[\\/]", str(cell["path"])) if p]
        if parts:
            return (*parts[:-1], Path(parts[-1]).stem)
    return (rec.name or rec.id,)


def _shortest_names(keys: list) -> dict:
    """Each of the DISTINCT ``keys``' shortest tail that no other key ends in, joined with "/":
    ``default`` while no other cell is called that, ``shm/default`` beside ``shm2/default``. A key
    that is wholly the tail of another -- a run's bare name, or a path that ends a longer one --
    keeps all of itself and the longer key grows past it, so no two keys come out with one name."""
    names = {}
    for key in keys:
        others = [k for k in keys if k != key]
        depth = next((d for d in range(1, len(key) + 1) if all(k[-d:] != key[-d:] for k in others)),
                     len(key))
        names[key] = "/".join(key[-depth:])
    return names


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
    only from the tool, by id or name: the window's picker offers finished records only. One run
    named twice is refused too, so the arity counts distinct runs.

    A bare string for ``refs`` is a TypeError, not a refusal: iterated, it would be one ref per
    character. Neither front end can produce one (``--record`` appends to a list, and the window's
    comparison list returns one), so only a caller's bug gets here, and it should read as a bug.
    """
    if isinstance(refs, (str, bytes)):
        raise TypeError(f"compare takes a list of refs, not a single {type(refs).__name__} "
                        f"({refs!r}); pass [{refs!r}] for one record")
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
    out, seen = [], set()
    for ref in refs:
        if not _held(store, ref):
            # The store's own refusal carries field="artifact", which would send the window to the
            # Artifacts screen and name no flag at all; the control that answers it is this list (F58).
            refuse("compare_records", f"The saved runs to compare must exist; {ref!r} names no fdt record.")
        rec = store.load_fdt(ref)
        who = rec.name or rec.id
        if rec.id in seen:
            # The arity above counts REFS, and a ref is an id OR a name: one run named twice -- by its
            # id twice, or by its id and its name -- would meet "at least 2" alone, and a "repeats"
            # of one run would record a zero-width band as two runs agreeing. Judged on the RESOLVED
            # id, so both spellings are caught.
            refuse("compare_records", f"The saved runs to compare must be different runs; {who!r} is "
                                      f"named twice.")
        seen.add(rec.id)
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


def labelled_curves(records: list) -> tuple:
    """``(curves, cells)``: each record's curve, labelled so that no two curves in one picture share a
    legend entry, and each record's cell under the name its label uses.

    A cell is named by its file's stem, as ``_label_of`` names it, until two DIFFERENT cells share
    one -- ``Cells/shm/default.txt`` and ``Cells/shm2/default.txt`` are both "default" -- and then by
    as much of its path as tells them apart ("shm/default"): a legend showing one name twice would
    say one cell was measured twice. Two runs of the SAME cell share its name honestly, so their
    labels add the run ("default (run_a)") and each curve can still be found. ``cells`` is what a
    mode reports as the cells it drew (P60), so it follows the same rule and counts cells, not runs.
    """
    keys = [_cell_key(r) for r in records]
    names = _shortest_names(list(dict.fromkeys(keys)))
    cells = [names[k] for k in keys]
    curves = [replace(curve_of(rec), label=cell if keys.count(key) == 1
                      else f"{cell} ({rec.name or rec.id})")
              for rec, key, cell in zip(records, keys, cells)]
    return curves, cells


def common_grid(curves: list) -> np.ndarray:
    """Log-spaced over the INTERSECTION of the curves' spans, with N the smallest point count among
    them (spec §7.2). Curves that share no band at all are refused: an empty picture is not an
    answer.

    The two end points are the span's own numbers, not their round trip through the logarithm:
    ``exp(log(3.0))`` is one ulp above 3.0, and an end point one ulp outside a record's span is
    outside it for ``interpolate_onto`` -- a blank the run never had.

    With no shared band, the refusal names the run that STARTS last and the run that ENDS first:
    the shared band is empty exactly when the one starts at or above where the other ends, so those
    two share nothing, which is the fact to act on. (The run with the narrowest span need not be one
    of them -- it can overlap every other run while two wider ones miss each other.) One run whose
    grid is a single frequency is both, and spans no band itself.
    """
    starts = [float(np.min(c.omegas)) for c in curves]
    ends = [float(np.max(c.omegas)) for c in curves]
    last, first = int(np.argmax(starts)), int(np.argmin(ends))
    lo, hi = starts[last], ends[first]
    n = min(int(np.size(c.omegas)) for c in curves)
    if not (hi > lo):          # also n < 2: a one-frequency grid starts where it ends
        if last == first:
            who = curves[last].name or curves[last].id
            refuse("compare_records",
                   f"The saved runs to compare must share a frequency band; {who!r} spans no band "
                   f"(its grid runs from {lo:g} to {hi:g}).")
        late, early = (curves[i].name or curves[i].id for i in (last, first))
        refuse("compare_records",
               f"The saved runs to compare must share a frequency band; these share none: {late!r} "
               f"starts at {lo:g} and {early!r} ends at {hi:g}.")
    if not (lo > 0.0):
        refuse("compare_records",
               f"The saved runs to compare must be measured at positive frequencies; every one of "
               f"them starts at {lo:g} or below, where a log-spaced grid cannot begin.")
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


def peak_of(curve: Curve, grid, values) -> dict:
    """One record's line in ``body.results.per_record``: where its ratio peaked on the common grid,
    and how high. Every float goes through ``_num``, so an all-blank curve records nulls rather than
    reaching the manifest writer, which refuses a non-finite number outright."""
    vals = np.asarray(values, dtype=np.float64)
    if not np.isfinite(vals).any():
        return {"id": curve.id, "label": curve.label, "peak_ratio": None, "peak_omega": None}
    i = int(np.nanargmax(vals))
    return {"id": curve.id, "label": curve.label,
            "peak_ratio": _num(vals[i]), "peak_omega": _num(np.asarray(grid)[i])}


def ratio_axes(title: str, blanks: int):
    """A figure and axes for T_eff/T against frequency, drawn the way ``plots.plot_eff_temp_ratio``
    draws its own: log x, symlog y so the chi''=0 crossing is visible on both sides of zero, and the
    two reference lines in ``axes.edgecolor`` -- figure CHROME follows the theme, which is why it is
    not a hardcoded gray (core/FDT/plots.py says why at length)."""
    from matplotlib import pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.set_xscale("log")
    ax.set_yscale("symlog", linthresh=1.0)
    ax.axhline(1.0, color=plt.rcParams["axes.edgecolor"], linestyle="--", linewidth=0.8,
               label=r"$T_{\rm eff}/T = 1$ (equilibrium)")
    ax.axhline(0.0, color=plt.rcParams["axes.edgecolor"], linestyle=":", linewidth=0.6)
    xlabel = r"$\tilde\omega$ (ND), log scale"
    if blanks:
        xlabel += f"  --  {BLANK_AXIS_NOTE}"
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"$T_{\rm eff}(\tilde\omega) / T$, symlog scale")
    ax.set_title(title)
    ax.grid(False)
    return fig, ax


def draw_cells(w, records, *, sink):
    """Several single-cell records' ratio curves on one axis, labelled by cell (spec §7.1)."""
    curves, _cells = labelled_curves(records)
    grid = common_grid(curves)
    values = [interpolate_onto(grid, c.omegas, c.ratio) for c in curves]
    blanks, notices = blank_notice(values)
    write_curves(w, grid, values, [c.label for c in curves],
                 constants=[(c.omega_0, c.prefactor) for c in curves])
    fig, ax = ratio_axes("FDT ratio by cell", blanks)
    for curve, vals in zip(curves, values):
        ax.plot(grid, vals, marker="o", markersize=4, linewidth=1.0, label=curve.label)
    ax.legend()
    fig.tight_layout()
    sink("FDT ratio by cell", fig)
    results = {"n_records": len(curves), "n_grid": int(grid.size), "blanks": blanks,
               "per_record": [peak_of(c, grid, v) for c, v in zip(curves, values)]}
    log.info(f"Common grid: {grid.size} pts spanning [{grid[0]:.4f}, {grid[-1]:.4f}]; "
             f"{blanks} blank point(s)")
    return results, notices


def draw_repeats(w, records, *, sink):
    """Repeats of one cell, with the spread ACROSS them as a band (spec §7.1, E7).

    The band is the envelope -- the lowest and the highest repeat at each frequency -- and not a
    standard deviation: with the two or three repeats this is written for, an SD is a number computed
    from too little to mean what its name promises, while the envelope is exactly "what the runs
    disagreed by". Plain ``min``/``max`` over the stack, so a point blank in ANY repeat is blank in
    the band: an envelope narrowed by a missing run would understate the error it exists to show.
    """
    curves, cell_of = labelled_curves(records)
    grid = common_grid(curves)
    values = [interpolate_onto(grid, c.omegas, c.ratio) for c in curves]
    blanks, notices = blank_notice(values)
    # P60: "the same cell" is not enforced -- a record's cell lives in its manifest's inputs and
    # nothing stops a caller passing two -- so the mode REPORTS which cells it drew: in the legend
    # (each curve is labelled by its cell) and here, in the record's notices. Named by
    # ``labelled_curves``' rule, so two different cells that share a file name count as two.
    cells = sorted(set(cell_of))
    notices = list(notices) + [
        f"The repeats drawn come from {len(cells)} cell(s): {', '.join(cells)}. The band is the "
        f"spread across these runs, and it is one cell's measurement error only when every run is "
        f"of the same cell."]
    stack = np.stack(values, axis=0)
    lo, hi, mean = stack.min(axis=0), stack.max(axis=0), stack.mean(axis=0)
    write_curves(w, grid, values, [c.label for c in curves],
                 constants=[(c.omega_0, c.prefactor) for c in curves])
    with h5py.File(w.payload("data.h5"), "a") as h5:
        band = h5.create_group("band")
        for key, arr in (("lo", lo), ("hi", hi), ("mean", mean)):
            band.create_dataset(key, data=np.asarray(arr, dtype=np.float64))
    fig, ax = ratio_axes("FDT ratio across repeats", blanks)
    ax.fill_between(grid, lo, hi, alpha=0.25, color="steelblue", label="spread across the repeats")
    ax.plot(grid, mean, color="steelblue", linewidth=1.4, label="mean of the repeats")
    for curve, vals in zip(curves, values):
        ax.plot(grid, vals, linewidth=0.6, alpha=0.7, label=curve.label)
    ax.legend()
    fig.tight_layout()
    sink("FDT ratio across repeats", fig)
    widest = hi - lo
    results = {"n_records": len(curves), "n_grid": int(grid.size), "blanks": blanks,
               "cells": cells,
               "widest": _num(np.nanmax(widest)) if np.isfinite(widest).any() else None,
               "band": {"lo": [_num(v) for v in lo], "hi": [_num(v) for v in hi],
                        "mean": [_num(v) for v in mean]},
               "per_record": [peak_of(c, grid, v) for c, v in zip(curves, values)]}
    return results, notices


def draw_renormalise(w, records, *, sink, prefactor):
    """One record's ratio recomputed with a supplied normalisation constant, drawn against the
    original (spec §7.1).

    An exact rescaling, not a re-measurement: ``spectral.eff_temp_ratio`` is
    ``prefactor * omega * G / (4 chi'')``, linear in the constant, so the new curve is the recorded
    one times the ratio of the two constants and nothing is simulated. A blank stays blank (NaN
    times any scale is NaN): no constant measures a frequency the run could not. The record's OWN
    grid is the common grid here -- there is one record, so there is nothing to interpolate onto,
    and interpolating a curve onto a regenerated copy of its own grid would only add float error.

    The supplied constant was judged before the record opened (``_checked_options``, F57); judging
    it again here is the backstop for a caller that reaches the drawer another way. The RECORDED
    constant can only be judged once the record's numbers are read, and it is judged before anything
    is written, so a refusal here still leaves no record behind (the writer removes a progressive
    record refused before its first payload or figure). A record that holds no constant (NaN)
    cannot be undone, and one that holds zero or less is no normalisation constant -- it is
    ``coupling / D_x``, which is positive -- and a ratio computed with zero is zero everywhere,
    which no scale brings back.
    """
    want = require_positive("prefactor", prefactor)
    curves, _cells = labelled_curves(records)
    curve = curves[0]
    who = records[0].name or records[0].id
    if math.isnan(curve.prefactor):
        refuse("compare_records",
               f"The saved runs to compare must record the normalisation constant they used; "
               f"{who!r} does not, so its ratio cannot be recomputed with another one.")
    if not (math.isfinite(curve.prefactor) and curve.prefactor > 0):
        refuse("compare_records",
               f"The saved runs to compare must record a finite normalisation constant greater "
               f"than 0 to be renormalised; {who!r} records {curve.prefactor:g}, so its ratio "
               f"cannot be recomputed with another one.")
    scale = want / curve.prefactor
    grid = np.asarray(curve.omegas, dtype=np.float64)
    original = np.asarray(curve.ratio, dtype=np.float64)
    renormalised = original * scale
    labels = [f"{curve.label} (recorded, {curve.prefactor:g})",
              f"{curve.label} (renormalised, {want:g})"]
    values = [original, renormalised]
    blanks, notices = blank_notice(values)
    # Each curve carries the constant ITS numbers were computed with (F55): the original its
    # record's own, the renormalised one the constant supplied -- so a reader of the file never has
    # to know which of the two curves was rescaled to read either correctly.
    write_curves(w, grid, values, labels,
                 constants=[(curve.omega_0, curve.prefactor), (curve.omega_0, want)])
    fig, ax = ratio_axes("FDT ratio renormalised", blanks)
    for label, vals, style in zip(labels, values, ("--", "-")):
        ax.plot(grid, vals, style, marker="o", markersize=3, linewidth=1.0, label=label)
    ax.legend()
    fig.tight_layout()
    sink("FDT ratio renormalised", fig)
    log.info(f"Renormalised {curve.label} from {curve.prefactor:g} to {want:g} "
             f"(x{scale:g}); nothing was re-simulated")
    results = {"n_records": 1, "n_grid": int(grid.size), "blanks": blanks,
               "prefactor": _num(want), "prefactor_recorded": _num(curve.prefactor),
               "scale": _num(scale),
               "per_record": [peak_of(curve, grid, renormalised)]}
    return results, notices


_DRAWERS["cells"] = draw_cells
_DRAWERS["repeats"] = draw_repeats
_DRAWERS["renormalise"] = draw_renormalise
