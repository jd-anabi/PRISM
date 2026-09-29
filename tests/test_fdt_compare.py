"""Comparing saved FDT records.

This suite covers the shared machinery -- the common grid, the interpolation that never blends a
blank away, and what a comparison refuses before it draws anything -- and then one test per mode.

Nothing here simulates. Every record is written straight into a store by ``build_fdt_record``, the
way ``build_browse_store`` writes its rows (``cfg=None``, the smallest body the kind allows), so the
whole file costs the writer's git calls and no solver at all.

Run:  pytest tests/test_fdt_compare.py
"""
import logging
import math
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib                                                  # noqa: E402
matplotlib.use("Agg")                                              # match the app; no interactive path

import h5py                                                        # noqa: E402
import numpy as np                                                 # noqa: E402
import pytest                                                      # noqa: E402

from core.artifacts import ArtifactStore                           # noqa: E402
from core.FDT import compare as cmp                                # noqa: E402
from core.refusals import Refusal                                  # noqa: E402
from tests._fixtures import build_fdt_record, compare_preflight_refusals  # noqa: E402


def _store(tmp_path):
    return ArtifactStore(tmp_path / "artifacts")


def _closing():
    """A sink that records the titles it is handed and closes each figure, like the store writer's
    own sink does when nothing is forwarded."""
    seen = []

    def _sink(title, fig):
        from matplotlib import pyplot as plt
        seen.append(title)
        plt.close(fig)
    return seen, _sink


def _curves_stub(w, records, *, sink, **_options):
    """A drawer that does what every real mode does first -- read each record's curve, put them on
    the common grid and write the comparison's ``data.h5`` -- and draws nothing. Stands in for a mode
    whose drawer has not landed, so this file pins the shared record and not any one picture."""
    curves = [cmp.curve_of(r) for r in records]
    grid = cmp.common_grid(curves)
    values = [cmp.interpolate_onto(grid, c.omegas, c.ratio) for c in curves]
    cmp.write_curves(w, grid, values, [c.label for c in curves],
                     constants=[(c.omega_0, c.prefactor) for c in curves])
    return {"n_records": len(curves)}, []


def _record_dirs(store):
    """Every directory under the store's fdt kind, finished or not: what a refusal must not add to."""
    d = store.kind_dir("fdt")
    return sorted(p.name for p in d.iterdir() if p.is_dir()) if d.is_dir() else []


def test_the_common_grid_is_the_log_spaced_intersection_at_the_smallest_point_count():
    """Every run detects its own resonance and builds its grid around it, so two runs of
    the same cell land on different frequencies; drawing them on one axis without a common grid would
    compare a value at 1.9 with a value at 2.1 and call the difference a result. The intersection is
    what both runs actually measured, and the smallest point count is the only one neither has to be
    invented for."""
    a = cmp.Curve("a", "a", "a", np.array([1.0, 2.0, 4.0, 8.0]), np.ones(4), 1.0, 1.0)
    b = cmp.Curve("b", "b", "b", np.array([2.0, 4.0, 8.0]), np.ones(3), 1.0, 1.0)
    grid = cmp.common_grid([a, b])
    assert grid.shape == (3,), grid
    assert grid[0] == pytest.approx(2.0) and grid[-1] == pytest.approx(8.0)
    # log-spaced: the ratio between neighbours is constant
    assert grid[1] / grid[0] == pytest.approx(grid[2] / grid[1])
    # no overlap at all is a refusal, not an empty picture
    far = cmp.Curve("c", "c", "c", np.array([100.0, 200.0]), np.ones(2), 1.0, 1.0)
    with pytest.raises(Refusal) as e:
        cmp.common_grid([a, far])
    assert e.value.field == "compare_records" and "share a frequency band" in str(e.value), e.value


def test_runs_that_share_no_band_are_refused_naming_two_that_share_nothing():
    """The refusal used to print "the intersection of their grids is [100, 8]" -- an interval with
    its ends the wrong way round, naming no run -- which is true only to a reader who already knows
    how it was computed. It says instead that the runs share no band and names the two that prove it:
    the run that STARTS last and the run that ENDS first. Those two are always disjoint when the
    shared band is empty, whereas the run with the narrowest span need not be: below, ``mid`` is the
    narrowest and overlaps both others, and it is ``lo`` and ``hi`` that share nothing. A run whose
    own grid is a single frequency spans no band at all, and is named as the reason."""
    def curve(name, om):
        return cmp.Curve(name, name, name, np.asarray(om, dtype=float), np.ones(len(om)), 1.0, 1.0)

    a, far = curve("a", [1.0, 2.0, 4.0, 8.0]), curve("far", [100.0, 200.0])
    with pytest.raises(Refusal) as e:
        cmp.common_grid([a, far])
    msg = str(e.value)
    assert e.value.field == "compare_records", e.value
    assert "share none" in msg and "'far' starts at 100" in msg and "'a' ends at 8" in msg, msg
    assert "[100, 8]" not in msg, "the inverted interval is gone"

    lo, mid, hi = curve("lo", [1.0, 4.0]), curve("mid", [3.5, 5.5]), curve("hi", [5.0, 8.0])
    with pytest.raises(Refusal) as e:
        cmp.common_grid([lo, mid, hi])
    msg = str(e.value)
    assert "'hi' starts at 5" in msg and "'lo' ends at 4" in msg and "'mid'" not in msg, msg

    one = curve("one", [2.0])
    with pytest.raises(Refusal) as e:
        cmp.common_grid([a, one])
    msg = str(e.value)
    assert "'one' spans no band" in msg and "'a'" not in msg, msg

    # a shared band that reaches down to zero cannot carry a log-spaced grid: said, not a log(0)
    with pytest.raises(Refusal) as e:
        cmp.common_grid([curve("z", [0.0, 2.0]), curve("neg", [-1.0, 3.0])])
    assert e.value.field == "compare_records" and "positive frequencies" in str(e.value), e.value
    assert "starts at 0 or below" in str(e.value), e.value


def test_the_common_grid_ends_exactly_on_the_shared_span_so_its_end_points_are_never_blank():
    """``exp(log(x))`` is not ``x`` in floating point: for a span ending at 3.0 it lands one ulp
    ABOVE 3.0 (and for one starting at 0.1, one ulp above that). A common-grid end point one ulp
    outside a record's own span is outside it as far as the interpolation is concerned -- blank --
    so a comparison of two clean curves would lose its last point for nothing, and report the loss
    as a gap the runs never had. The ends are therefore the span's own numbers, not their round
    trip through the logarithm."""
    om = np.array([0.1, 1.0, 3.0])
    clean = cmp.Curve("a", "a", "a", om, np.array([1.0, 2.0, 1.5]), 1.0, 1.0)
    grid = cmp.common_grid([clean, clean])
    assert grid[0] == 0.1 and grid[-1] == 3.0, grid.tolist()
    vals = cmp.interpolate_onto(grid, om, clean.ratio)
    assert np.isfinite(vals).all(), vals.tolist()
    assert vals[0] == 1.0 and vals[-1] == 1.5, vals.tolist()
    assert cmp.blank_notice([vals, vals]) == (0, []), "two clean curves report no blank"


def test_no_target_point_is_interpolated_across_a_blank():
    """The whole point of the off-grid fix: a frequency the run could not measure
    comes back blank, and a comparison that quietly interpolated over it would put the fabricated tail
    back -- the defect ``_interp_log`` had. A target between two good
    samples is a blend; a target whose bracket includes a blank, or that lies outside the source's
    span, is blank. Asserted through an interpolated INDICATOR rather than by trusting NaN to survive
    ``np.interp``, which retries a non-finite result from the other bracket."""
    om = np.array([1.0, 2.0, 4.0, 8.0])
    vals = np.array([1.0, np.nan, 3.0, 4.0])
    out = cmp.interpolate_onto(np.array([1.0, 1.5, 2.0, 3.0, 4.0, 16.0]), om, vals)
    assert out[0] == pytest.approx(1.0), "a target ON a good sample keeps that sample"
    assert math.isnan(out[1]), "bracketed by the blank at 2.0"
    assert math.isnan(out[2]), "the blank itself"
    assert math.isnan(out[3]), "bracketed by the blank at 2.0 on the other side"
    assert out[4] == pytest.approx(3.0)
    assert math.isnan(out[5]), "outside the source's span"
    # and a curve with no blanks is interpolated normally, in LOG omega
    clean = cmp.interpolate_onto(np.array([2.0]), np.array([1.0, 4.0]), np.array([0.0, 2.0]))
    assert clean[0] == pytest.approx(1.0), "linear in log-omega: log2 is halfway between log1 and log4"
    # the notice counts a point once however many curves are blank there, and says so in words
    blanks, notices = cmp.blank_notice([out, np.ones(6)])
    assert blanks == 4 and len(notices) == 1 and notices[0].startswith("4 of 6 points"), notices


def test_a_comparison_refuses_one_record_another_study_and_an_unfinished_one(tmp_path, monkeypatch):
    """A comparison draws only records whose complete is true. Every mode's arity is a real input
    class -- one cell is not a comparison, and a sweep drawn as a single-cell run reads a layout that
    is not there -- and an interrupted run keeps its folder, so half-written records sit on disk
    carrying a valid manifest and the flag can hand one over by id or name (the window's picker offers
    finished records only). Each refusal names what it got and carries the field both front ends map
    to their own control."""
    store = _store(tmp_path)
    # these refusals are load_records', and every one lands before a drawer runs: a stub stands in for
    # the real drawer so the test pins the refusals and not the picture
    monkeypatch.setitem(cmp._DRAWERS, "cells", lambda w, records, *, sink: ({}, []))
    one = build_fdt_record(store, name="a")
    sweep = build_fdt_record(store, name="sw", study="sweep")
    part = build_fdt_record(store, name="half", finished=False)

    with pytest.raises(Refusal) as e:
        cmp.compare("cells", [one], store=store)
    assert e.value.field == "compare_records" and "at least 2" in str(e.value), e.value
    assert "got 1" in str(e.value), e.value

    with pytest.raises(Refusal) as e:
        cmp.compare("cells", [one, sweep], store=store)
    assert "'sw'" in str(e.value) and "sweep" in str(e.value), e.value

    with pytest.raises(Refusal) as e:
        cmp.compare("cells", [one, part], store=store)
    assert "'half'" in str(e.value) and "did not" in str(e.value), e.value

    # an unfinished record is listed and loadable -- it is the comparison that refuses it, not the store
    assert store.load_fdt(part).body["complete"] is False
    # a setting the mode has no use for is refused rather than recorded
    with pytest.raises(Refusal) as e:
        cmp.compare("cells", [one, one], prefactor=3.0, store=store)
    assert "prefactor" in str(e.value), e.value


def test_every_refusal_lands_before_the_comparison_opens_its_record(tmp_path, monkeypatch, caplog):
    """Refuse before the spend. A progressive record exists from the moment its writer is entered,
    and the writer keeps it through an exception -- so a refusal raised after that point would leave
    a comparison record behind that holds nothing, or, refused before its first write, announce
    a record ("Writing comparison record <id> at <dir>") for a directory the writer then removes.
    Every refusal a comparison can make from what it was GIVEN is therefore made before the record
    opens: each one leaves the kind directory exactly as it found it and logs no such line.

    What was given is the mode and its settings (a mode this build does not draw, a setting the mode
    has no use for, a blank or non-positive normalisation constant, a non-finite slice point); the
    arity, both ways (a mode that draws one record or two is never handed more and silently draws the
    first), counted in distinct RUNS (one run named twice is one run); each record (one that names
    nothing, one of another study, one that did not finish, one with no numbers beside its manifest);
    and -- the mode's PRE-FLIGHT, which reads the records once they load -- what they hold: curves
    that share no band, a data file without its curve, a recorded constant that cannot be undone, a
    sweep with no finished operating point, two sweeps of different parameters or of ranges that do
    not meet, a slice point outside the range they share, and slice rows that share no band. The
    drawers are stubs here, so a pre-flight refusal cannot be the drawer's. The ref that names
    nothing is refused under the comparison's own field, not the store's ``artifact``, so both front
    ends name the control that answers it."""
    caplog.set_level(logging.INFO, logger="core")
    store = _store(tmp_path)
    for mode in cmp.MODE_RULES:
        monkeypatch.setitem(cmp._DRAWERS, mode, _curves_stub)
    a = build_fdt_record(store, name="a")
    b = build_fdt_record(store, name="b")
    s1 = build_fdt_record(store, name="s1", study="sweep")
    s2 = build_fdt_record(store, name="s2", study="sweep")
    s3 = build_fdt_record(store, name="s3", study="sweep")
    half = build_fdt_record(store, name="half", finished=False)
    # a FINISHED single-cell record that never wrote numbers -- a run stopped after its sanity checks
    # has this shape -- so the store loads it (no payload is recorded, so none is verified)
    with store.create("fdt", None, name="bare") as w:
        w.body = {"study": "single", "settings": {}, "seed": 7, "notices": []}

    cases = [
        ("no such mode", ("overlay", [a, b], {}), "compare_records", "'overlay'"),
        ("unused setting", ("cells", [a, b], {"at": 1.0}), "compare_records", "no use for: at"),
        ("blank constant", ("renormalise", [a], {}), "prefactor", "blank"),
        ("zero constant", ("renormalise", [a], {"prefactor": 0.0}), "prefactor", "greater than 0"),
        ("nan constant", ("renormalise", [a], {"prefactor": math.nan}), "prefactor", "nan"),
        ("infinite slice", ("sweeps", [s1, s2], {"at": math.inf}), "slice_at", "finite"),
        ("too few", ("cells", [a], {}), "compare_records", "at least 2"),
        ("too many for one", ("renormalise", [a, b], {"prefactor": 3.0}), "compare_records",
         "at most 1 for a renormalise comparison; got 2"),
        ("too many for two", ("sweeps", [s1, s2, s3], {}), "compare_records",
         "at most 2 for a sweeps comparison; got 3"),
        ("names nothing", ("cells", [a, "no_such_run"], {}), "compare_records",
         "'no_such_run' names no fdt record"),
        ("other study", ("sweeps", [s1, a], {}), "compare_records", "'a' is a single run"),
        ("unfinished", ("repeats", [a, half], {}), "compare_records", "'half' did not"),
        ("no numbers", ("cells", [a, "bare"], {}), "compare_records", "'bare' has no data file"),
        # the arity counts RUNS, not refs: one run named twice -- by its id twice, by its id and its
        # name, or as both halves of a sweeps pair -- is one run, and a one-run "repeats" would record
        # a zero-width band as two runs agreeing
        ("one id twice", ("repeats", [a, a], {}), "compare_records", "'a' is named twice"),
        ("id and name", ("cells", [a, "a"], {}), "compare_records", "'a' is named twice"),
        ("one sweep twice", ("sweeps", [s1, s1], {}), "compare_records", "'s1' is named twice"),
    ]
    cases += [(label, (mode, refs, options), field, words)
              for label, mode, refs, options, field, words in compare_preflight_refusals(store)]
    before = _record_dirs(store)
    for label, (mode, refs, options), field, words in cases:
        caplog.clear()
        with pytest.raises(Refusal) as e:
            cmp.compare(mode, refs, store=store, **options)
        assert e.value.field == field, (label, e.value.field, str(e.value))
        assert words in str(e.value), (label, str(e.value))
        assert _record_dirs(store) == before, f"{label}: a refused comparison left a record behind"
        opened = [r.getMessage() for r in caplog.records
                  if r.getMessage().startswith("Writing comparison record")]
        assert not opened, f"{label}: refused after the record opened: {opened}"

    # A bare string is not a list of refs: iterated, a run's id would be read as one ref per
    # character ("'2' names no fdt record"). No front end can hand one over -- --record appends, and
    # the window's list returns a list -- so it is the caller's bug, raised as one (a TypeError, which
    # both front ends show as a bug) rather than dressed up as a refusal of the runs.
    with pytest.raises(TypeError, match="a list of refs"):
        cmp.compare("cells", a, store=store)
    assert _record_dirs(store) == before

    # the no-drawer guard names what this build DOES draw, once the table is back to real
    monkeypatch.delitem(cmp._DRAWERS, "cells")
    with pytest.raises(Refusal) as e:
        cmp.compare("cells", [a, b], store=store)
    assert e.value.field == "compare_records" and "no cells comparison" in str(e.value), e.value
    assert "(it draws renormalise, repeats, sweeps)" in str(e.value), e.value
    assert _record_dirs(store) == before


def test_each_drawer_backstops_its_preflight_in_the_same_words(tmp_path, monkeypatch, caplog):
    """The pre-flight is where these refusals LAND; the drawers keep them as a backstop, for a caller
    that reaches a drawer another way. Each check is one helper that both call, so a refusal has one
    wording whichever route raises it: with the pre-flight table emptied, the REAL drawer refuses the
    same input with the same field and the same sentence. And it refuses before its first write:
    the record it opened -- the log names it, which is the proof the drawer ran -- is removed
    by the writer, because nothing was written into it."""
    caplog.set_level(logging.INFO, logger="core")
    store = _store(tmp_path)
    cases = compare_preflight_refusals(store)
    before = _record_dirs(store)
    for label, mode, refs, options, _field, _words in cases:
        with pytest.raises(Refusal) as first:
            cmp.compare(mode, refs, store=store, **options)
        caplog.clear()
        with monkeypatch.context() as m:
            m.setattr(cmp, "_PREFLIGHT", {})
            with pytest.raises(Refusal) as backstop:
                cmp.compare(mode, refs, store=store, **options)
        assert (backstop.value.field, str(backstop.value)) == (first.value.field, str(first.value)), \
            label
        assert any(r.getMessage().startswith("Writing comparison record") for r in caplog.records), \
            f"{label}: the drawer never ran, so this is not its backstop"
        assert _record_dirs(store) == before, f"{label}: the drawer wrote before it refused"


def test_a_comparison_is_a_record_that_names_the_mode_and_the_runs_it_drew(tmp_path, monkeypatch):
    """A comparison is an ``fdt`` record of its own, with ``study = "comparison"``, its
    sources in ``body.compared`` and NOT in ``parents`` (the parents block is a flat {key: id} map and
    cannot carry an arbitrary number of ids), the common grid and the interpolated curves
    in ``data.h5``, and its figures as its output. Driven through a stub drawer, so this pins the
    record and not any one mode's picture.

    Every ``data.h5`` carries ``omega_0`` and ``prefactor`` at its root (the layout contract); for a
    comparison neither has one value, so the root holds NaN and each curve carries the constants of
    the record it came from -- which is what a renormalised curve needs to be read correctly."""
    store = _store(tmp_path)
    a = build_fdt_record(store, name="cell_a")
    b = build_fdt_record(store, name="cell_b", omegas=(1.0, 2.0, 4.0), ratio=(1.0, 2.0, 1.1),
                         omega_0=1.5, prefactor=2.5)

    def _stub(w, records, *, sink):
        results, _ = _curves_stub(w, records, sink=sink)
        from matplotlib import pyplot as plt
        sink("Stub comparison", plt.figure())
        return results, ["a notice the record keeps"]

    monkeypatch.setitem(cmp._DRAWERS, "cells", _stub)
    rec = cmp.compare("cells", [a, b], name="ab", note="two cells", store=store)

    assert rec.body["study"] == "comparison" and rec.body["complete"] is True
    assert rec.name == "ab" and rec.manifest.note == "two cells"
    assert rec.body["compared"]["mode"] == "cells"
    assert [r["id"] for r in rec.body["compared"]["records"]] == [a, b]
    assert [r["name"] for r in rec.body["compared"]["records"]] == ["cell_a", "cell_b"]
    assert [r["kind"] for r in rec.body["compared"]["records"]] == ["fdt", "fdt"]
    assert rec.body["settings"] == {"mode": "cells", "records": 2}, rec.body["settings"]
    assert rec.manifest.parents == {}, "a comparison names its sources in the body, never in parents"
    assert rec.body["results"] == {"n_records": 2} and rec.body["notices"] == ["a notice the record keeps"]
    assert rec.body["seed"] is None and rec.body["points"] is None and rec.body["grid"] is None
    assert rec.body["offgrid"] is None
    assert rec.manifest.figures == ["figures/stub_comparison.png"]
    assert (rec.path / "figures" / "stub_comparison.png").is_file()
    assert rec.manifest.payloads["data.h5"], "the payload is hashed at the commit, like every other kind"
    assert (rec.path / "log.txt").is_file(), "public_entry: the run's records land beside the manifest"
    log_text = (rec.path / "log.txt").read_text(encoding="utf-8")
    assert f"Writing comparison record {rec.id}" in log_text, log_text
    with h5py.File(rec.data_path, "r") as h5:
        assert h5.attrs["study"] == "comparison" and h5.attrs["mode"] == "cells"
        assert math.isnan(h5.attrs["omega_0"]) and math.isnan(h5.attrs["prefactor"]), dict(h5.attrs)
        assert h5["omega_common"].shape == (3,)
        assert sorted(h5["curves"]) == ["000", "001"]
        assert h5["curves"]["000"].attrs["label"] == "cell_a"
        assert h5["curves"]["001"].attrs["label"] == "cell_b"
        assert (h5["curves"]["000"].attrs["omega_0"], h5["curves"]["000"].attrs["prefactor"]) == (1.0, 2.0)
        assert (h5["curves"]["001"].attrs["omega_0"], h5["curves"]["001"].attrs["prefactor"]) == (1.5, 2.5)
    # the comparison is listed like any other record, as a comparison and finished
    row = next(s for s in store.list("fdt") if s.id == rec.id)
    assert row.study == "comparison" and row.finished, row


def test_a_comparison_records_the_ids_it_resolved_never_the_refs_it_was_given(tmp_path, monkeypatch):
    """The store resolves a ref by NAME as well as by id, and ``render_lineage`` looks a compared entry
    up by id. A comparison that stored the ref as typed would, handed ``cell_a``, record a NAME in the
    id slot -- and once that run was deleted and another took its name, the comparison's lineage
    would name the newcomer as its source. So each entry is the loaded record's own id and name."""
    store = _store(tmp_path)
    monkeypatch.setitem(cmp._DRAWERS, "cells", _curves_stub)
    a = build_fdt_record(store, name="cell_a")
    b = build_fdt_record(store, name="cell_b")
    rec = cmp.compare("cells", ["cell_a", "cell_b"], store=store)
    assert [(r["id"], r["name"]) for r in rec.body["compared"]["records"]] == \
        [(a, "cell_a"), (b, "cell_b")], rec.body["compared"]

    # the source deleted, and a new run given its name: the comparison still points at the one it drew
    store.delete("fdt", "cell_a")
    newcomer = build_fdt_record(store, name="cell_a")
    from core.artifacts import render_lineage
    text = render_lineage(store, "fdt", rec.id)
    assert f"MISSING fdt [{a}]" in text, text
    assert newcomer not in text, "the lineage names the run that took the deleted source's name"


def test_a_comparison_whose_source_was_deleted_still_lists_and_loads(tmp_path, monkeypatch):
    """Deleting a run a comparison drew is NOT refused -- the sources live in the body, which the
    dependency check does not read -- and the comparison stays what it was: listed, loadable, and
    honest about the gap, its lineage printing ``MISSING fdt [<id>]`` for the run that is gone and
    the ordinary line for the one that is not."""
    store = _store(tmp_path)
    monkeypatch.setitem(cmp._DRAWERS, "cells", _curves_stub)
    a = build_fdt_record(store, name="cell_a")
    b = build_fdt_record(store, name="cell_b")
    rec = cmp.compare("cells", [a, b], name="ab", store=store)

    store.delete("fdt", a)                                  # not refused: nothing names it as a parent

    rows = {s.id: s for s in store.list("fdt")}
    assert a not in rows and rec.id in rows and rows[rec.id].finished, rows
    again = store.load_fdt(rec.id)
    assert again.body["compared"]["records"][0]["id"] == a, "the record still says what it drew"
    with h5py.File(again.data_path, "r") as h5:
        assert sorted(h5["curves"]) == ["000", "001"], "and still holds what it drew"
    from core.artifacts import render_lineage
    text = render_lineage(store, "fdt", rec.id)
    assert f"MISSING fdt [{a}]" in text, text
    assert f"fdt cell_b [{b}]" in text, text


def test_an_interrupted_comparison_keeps_its_record_unfinished(tmp_path, monkeypatch):
    """For the comparison's own record, too: a cancel after the record opened leaves it on disk, marked
    unfinished, holding what was written -- the same rule every fdt record follows, and the one the
    tool's interrupt note describes. Nothing resumes it: the records it drew are untouched and
    re-running draws a new comparison."""
    store = _store(tmp_path)
    a = build_fdt_record(store, name="cell_a")
    b = build_fdt_record(store, name="cell_b")

    def _cancelled(w, records, *, sink):
        _curves_stub(w, records, sink=sink)
        raise KeyboardInterrupt

    monkeypatch.setitem(cmp._DRAWERS, "cells", _cancelled)
    with pytest.raises(KeyboardInterrupt):
        cmp.compare("cells", [a, b], name="ab", store=store)
    kept = store.load_fdt("ab")
    assert kept.body["complete"] is False and kept.body["study"] == "comparison", kept.body
    assert kept.data_path is not None, "the curves written before the cancel are kept"
    assert [r["id"] for r in kept.body["compared"]["records"]] == [a, b]
    for ref in (a, b):
        assert store.load_fdt(ref).body["complete"] is True, "a comparison never touches its sources"


def test_the_compare_path_loads_no_inference_machinery(tmp_path):
    """As for an FDT run, so for the comparison: importing any ``core.diagnostics`` submodule runs
    that package's ``__init__``, which loads the SBI stack and prints two false pytensor "g++" lines
    at the head of the run. A comparison draws saved numbers and needs none of it. RUN in a fresh
    interpreter -- this process holds the orchestrator through the session fixtures, so a
    ``sys.modules`` check here would pass vacuously -- over records this process wrote. The bare
    ``sbi`` package is the store's (``provenance.env_info`` reads its version into every manifest);
    every other ``sbi`` module is the inference stack and is forbidden here. The sweeps mode is drawn
    too, by its REAL drawer: it reads its records through ``cross_validation.load_param_sweep``, the
    one import on this path that brings a measurement module (and the simulators) with it."""
    root = tmp_path / "artifacts"
    store = ArtifactStore(root)
    ids = [build_fdt_record(store, name=f"cell_{i}") for i in range(2)]
    ids += [build_fdt_record(store, name=f"sweep_{i}", study="sweep", omegas=(0.5, 1.0, 2.0),
                             points=[(0.1 * i, (1.0, 2.0, 1.0)), (0.5 + 0.1 * i, (1.0, 3.0, 1.0))])
            for i in range(2)]
    probe = (
        "import sys\n"
        "from core.artifacts import ArtifactStore\n"
        "from core.FDT import compare as cmp\n"
        "def draw(w, records, *, sink):\n"
        "    curves = [cmp.curve_of(r) for r in records]\n"
        "    grid = cmp.common_grid(curves)\n"
        "    values = [cmp.interpolate_onto(grid, c.omegas, c.ratio) for c in curves]\n"
        "    cmp.write_curves(w, grid, values, [c.label for c in curves],\n"
        "                     constants=[(c.omega_0, c.prefactor) for c in curves])\n"
        "    return {}, []\n"
        "cmp._DRAWERS['cells'] = draw\n"
        "rec = cmp.compare('cells', sys.argv[2:4], store=ArtifactStore(sys.argv[1]))\n"
        "assert rec.body['complete'] is True, rec.body\n"
        "rec = cmp.compare('sweeps', sys.argv[4:6], store=ArtifactStore(sys.argv[1]))\n"
        "assert rec.body['complete'] is True, rec.body\n"
        "bad = sorted(m for m in sys.modules\n"
        "             if m in ('core.diagnostics', 'core.orchestrator', 'pytensor')\n"
        "             or (m.startswith('sbi.') and m != 'sbi.__version__'))\n"
        "sys.exit(0 if not bad else repr(bad))\n")
    r = subprocess.run([sys.executable, "-c", probe, str(root), *ids],
                       cwd=str(Path(__file__).resolve().parents[1]), capture_output=True, text=True,
                       timeout=300, env={**os.environ, "MPLBACKEND": "Agg", "KMP_DUPLICATE_LIB_OK": "TRUE",
                                         "PRISM_ARTIFACTS": str(root)})
    assert r.returncode == 0, r.stdout + r.stderr
    assert "pytensor" not in r.stderr, r.stderr


def test_compare_cells_draws_every_record_on_one_axis_and_records_its_peak(tmp_path):
    """Several cells' ratio curves on ONE axis, labelled by cell. The labels matter as much
    as the curves -- a picture of four unlabelled traces answers nothing -- and the peak of each is in
    the record, so the comparison can be read back without re-opening the figure. The two records here
    measure different bands on purpose: the drawn grid is their intersection, not either one's
    own."""
    store = _store(tmp_path)
    a = build_fdt_record(store, name="master_spont", omegas=(0.5, 1.0, 2.0, 4.0),
                         ratio=(1.0, 5.0, 1.3, 1.1))
    b = build_fdt_record(store, name="master_weak", omegas=(1.0, 2.0, 4.0),
                         ratio=(1.0, 2.0, 1.2))
    seen, sink = _closing()
    rec = cmp.compare("cells", [a, b], name="two_cells", fig_sink=sink, store=store)

    assert seen == ["FDT ratio by cell"], seen
    assert rec.manifest.figures == ["figures/fdt_ratio_by_cell.png"]
    res = rec.body["results"]
    assert res["n_records"] == 2 and res["n_grid"] == 3 and res["blanks"] == 0
    assert [p["label"] for p in res["per_record"]] == ["master_spont", "master_weak"]
    # the common grid is [1, 2, 4] -- the intersection [1, 4] at the smaller point count, 3:
    # master_spont's peak at omega=1 is on it, master_weak's at omega=2
    assert res["per_record"][0]["peak_omega"] == pytest.approx(1.0)
    assert res["per_record"][1]["peak_omega"] == pytest.approx(2.0)
    assert res["per_record"][1]["peak_ratio"] == pytest.approx(2.0)
    with h5py.File(rec.data_path, "r") as h5:
        assert [h5["curves"][k].attrs["label"] for k in sorted(h5["curves"])] == \
            ["master_spont", "master_weak"]


def test_compare_cells_keeps_a_blank_blank_and_says_so_in_the_record(tmp_path):
    """A blank carried through a comparison. A run that could not measure a frequency reports a
    blank, and the comparison must not blend it away: the blank stays blank, the count is in the
    record's notices, and the notices are what a user reads. A silently interpolated gap is the
    fabricated tail the off-grid fix removed, put back one layer up."""
    store = _store(tmp_path)
    a = build_fdt_record(store, name="a", omegas=(1.0, 2.0, 4.0, 8.0),
                         ratio=(1.0, float("nan"), 1.3, 1.1))
    b = build_fdt_record(store, name="b", omegas=(1.0, 2.0, 4.0, 8.0), ratio=(1.0, 2.0, 1.2, 1.05))
    _seen, sink = _closing()
    rec = cmp.compare("cells", [a, b], fig_sink=sink, store=store)
    assert rec.body["results"]["blanks"] == 1, rec.body["results"]
    assert rec.body["notices"] and "never interpolated across" in rec.body["notices"][0]
    with h5py.File(rec.data_path, "r") as h5:
        drawn = h5["curves"]["000"][...]
    assert math.isnan(drawn[1]) and not np.isnan(drawn[[0, 2, 3]]).any(), drawn


def test_compare_repeats_draws_the_spread_across_the_runs_as_a_band(tmp_path):
    """Repeats of one cell run at different seeds, and the spread across them IS the
    measurement error -- the number that says whether a difference between two cells means anything.
    The band is the envelope across the repeats at each frequency, so a point blank in ANY repeat is
    blank in the band: an envelope silently narrowed by a missing run would understate exactly the
    error it exists to show."""
    store = _store(tmp_path)
    ids = [build_fdt_record(store, name=f"rep{i}", omegas=(1.0, 2.0, 4.0), ratio=r)
           for i, r in enumerate([(1.0, 3.0, 1.0), (1.2, 5.0, 1.1), (0.8, 4.0, 0.9)])]
    seen, sink = _closing()
    rec = cmp.compare("repeats", ids, fig_sink=sink, store=store)

    assert seen == ["FDT ratio across repeats"], seen
    res = rec.body["results"]
    assert res["n_records"] == 3
    # at omega = 2 the three repeats give 3, 5 and 4: the band spans 3 to 5 and the mean is 4
    assert res["band"]["lo"][1] == pytest.approx(3.0)
    assert res["band"]["hi"][1] == pytest.approx(5.0)
    assert res["band"]["mean"][1] == pytest.approx(4.0)
    assert res["widest"] == pytest.approx(2.0), "the widest spread over the grid"
    with h5py.File(rec.data_path, "r") as h5:
        assert sorted(h5["curves"]) == ["000", "001", "002"]
        assert list(h5["band"]) == ["hi", "lo", "mean"]
    assert res["cells"] == ["rep0", "rep1", "rep2"], "no cell input on the fixture: the label is the name"
    assert any("rep0, rep1, rep2" in n for n in rec.body["notices"]), rec.body["notices"]


def test_a_point_blank_in_any_repeat_is_blank_in_the_band(tmp_path):
    """The band's own promise, driven: one repeat that could not measure omega = 2 blanks the band
    there -- lo, hi and mean alike -- rather than narrowing it to the two runs that did, which would
    report a smaller measurement error than the runs have. The record holds null where the band is
    blank (a manifest refuses NaN), the file holds NaN, the widest spread is taken over the points the
    band does have, and the blank is counted in the notices like any other."""
    store = _store(tmp_path)
    ids = [build_fdt_record(store, name=f"rep{i}", omegas=(1.0, 2.0, 4.0), ratio=r)
           for i, r in enumerate([(1.0, 3.0, 1.0), (1.2, float("nan"), 1.6), (0.8, 4.0, 0.9)])]
    _seen, sink = _closing()
    rec = cmp.compare("repeats", ids, fig_sink=sink, store=store)

    band = rec.body["results"]["band"]
    assert band["lo"][1] is None and band["hi"][1] is None and band["mean"][1] is None, band
    assert band["lo"][2] == pytest.approx(0.9) and band["hi"][2] == pytest.approx(1.6), band
    assert rec.body["results"]["widest"] == pytest.approx(0.7), "taken over the points it has"
    assert rec.body["results"]["blanks"] == 1
    assert "never interpolated across" in rec.body["notices"][0], rec.body["notices"]
    with h5py.File(rec.data_path, "r") as h5:
        assert all(math.isnan(h5["band"][k][1]) for k in ("lo", "hi", "mean"))


def _record_of_cell(store, monkeypatch, cell_path: str, **kwargs) -> str:
    """A ``build_fdt_record`` whose manifest names a cell file, the way a real run's does.

    The fixture writes with no configuration, so its manifests name no cell and a comparison labels
    each curve by the record's name. A real run's ``inputs.cell`` is ``{"path", "sha256"}`` with the
    path relative to ``Resources/`` (``provenance.file_ref``); the writer's inputs are replaced with
    exactly that for this one record, rather than building a full FDT configuration from a real cell
    just to have its path recorded."""
    from core.artifacts.store import ArtifactWriter
    inputs = {"bounds": None, "units": None, "model": None,
              "cell": {"path": cell_path, "sha256": "ab" * 32}}
    with monkeypatch.context() as m:
        m.setattr(ArtifactWriter, "_inputs", lambda self, *, warn: dict(inputs))
        return build_fdt_record(store, **kwargs)


def test_two_cells_that_share_a_file_name_never_share_a_legend_entry(tmp_path, monkeypatch):
    """A curve is labelled by its cell's file stem, and stems are not unique: ``Cells/shm/default.txt``
    and ``Cells/shm2/default.txt`` are two different cells, both called "default". A legend that
    showed "default" twice would put two cells' curves under one name -- the picture would claim the
    same cell measured twice. So colliding names grow the folder they sit in ("shm/default"), a stem
    no other cell shares stays short, and two runs of the SAME cell (which share its name honestly)
    are told apart by the run. ``repeats`` names the cells it drew by the same rule, so its
    notice counts two cells here, not one "default"."""
    store = _store(tmp_path)
    shm = _record_of_cell(store, monkeypatch, "Cells/shm/default.txt", name="run_shm")
    again = _record_of_cell(store, monkeypatch, "Cells/shm/default.txt", name="run_shm_again")
    shm2 = _record_of_cell(store, monkeypatch, "Cells/shm2/default.txt", name="run_shm2")
    other = _record_of_cell(store, monkeypatch, "Cells/nadrowski/master_spont.txt", name="run_ms")
    assert store.load_fdt(shm).manifest.inputs["cell"]["path"] == "Cells/shm/default.txt"

    legends = {}

    def _sink(title, fig):
        from matplotlib import pyplot as plt
        legends[title] = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
        plt.close(fig)

    def labels_of(rec):
        with h5py.File(rec.data_path, "r") as h5:
            drawn = [h5["curves"][k].attrs["label"] for k in sorted(h5["curves"])]
        assert drawn == [p["label"] for p in rec.body["results"]["per_record"]], drawn
        return drawn

    rec = cmp.compare("cells", [shm, shm2, other], fig_sink=_sink, store=store)
    assert labels_of(rec) == ["shm/default", "shm2/default", "master_spont"]
    drawn = legends["FDT ratio by cell"]
    assert len(set(drawn)) == len(drawn), drawn
    assert {"shm/default", "shm2/default", "master_spont"} <= set(drawn), drawn

    # two runs of one cell beside a different cell of the same file name
    rec = cmp.compare("repeats", [shm, again, shm2], fig_sink=_sink, store=store)
    assert labels_of(rec) == ["shm/default (run_shm)", "shm/default (run_shm_again)", "shm2/default"]
    assert rec.body["results"]["cells"] == ["shm/default", "shm2/default"], rec.body["results"]
    assert any("2 cell(s): shm/default, shm2/default" in n for n in rec.body["notices"]), \
        rec.body["notices"]
    drawn = legends["FDT ratio across repeats"]
    assert len(set(drawn)) == len(drawn), drawn

    # repeats of one cell and nothing else: one cell, by its short name, and each run told apart
    rec = cmp.compare("repeats", [shm, again], fig_sink=_sink, store=store)
    assert labels_of(rec) == ["default (run_shm)", "default (run_shm_again)"]
    assert rec.body["results"]["cells"] == ["default"], rec.body["results"]
    assert any("1 cell(s): default." in n for n in rec.body["notices"]), rec.body["notices"]


def _lines_sink():
    """``(lines, sink)``: a sink that keeps each figure's first axes' labelled lines --
    ``{title: {label: Line2D}}`` -- and closes the figure."""
    lines = {}

    def _sink(title, fig):
        from matplotlib import pyplot as plt
        lines[title] = {ln.get_label(): ln for ln in fig.axes[0].get_lines()}
        plt.close(fig)
    return lines, _sink


def test_repeats_of_one_cell_say_so_as_information_and_draw_every_measured_point(tmp_path,
                                                                                  monkeypatch,
                                                                                  caplog):
    """The repeats mode reports which cells it drew, in the record's notices -- and ``compare``
    logged every notice at WARNING, so EVERY correct comparison of one cell's repeats warned: a
    warning on every run trains the owner to ignore warnings. The one-cell sentence is kept in the
    notices and logged as information; drawing more than one cell is still the warning it should be.

    And the repeat curves and their mean were drawn as lines with no marker, so a measured point
    with a blank on each side -- a segment of one point -- was not drawn at all. Every curve now marks
    its points: here the first repeat measured the two ends of the grid and not its middle."""
    store = _store(tmp_path)
    a = _record_of_cell(store, monkeypatch, "Cells/shm/default.txt", name="run_a", seed=1,
                        omegas=(1.0, 2.0, 4.0), ratio=(1.0, float("nan"), 2.0))
    b = _record_of_cell(store, monkeypatch, "Cells/shm/default.txt", name="run_b", seed=2,
                        omegas=(1.0, 2.0, 4.0), ratio=(1.2, 3.0, 2.2))
    lines, sink = _lines_sink()
    with caplog.at_level(logging.INFO, logger="core"):
        rec = cmp.compare("repeats", [a, b], fig_sink=sink, store=store)
    said = [(r.levelno, r.getMessage()) for r in caplog.records if r.name == "core.FDT.compare"]
    cells = [(lvl, m) for lvl, m in said if "come from 1 cell(s)" in m]
    assert [lvl for lvl, _ in cells] == [logging.INFO], said
    assert any("come from 1 cell(s): default." in n for n in rec.body["notices"]), rec.body["notices"]

    drawn = lines["FDT ratio across repeats"]
    first = drawn["default (run_a)"]
    assert np.isnan(first.get_ydata()[1]) and np.isfinite(first.get_ydata()[[0, 2]]).all(), \
        "the premise: two measured points with a blank between them"
    for label in ("default (run_a)", "default (run_b)", "mean of the repeats"):
        assert drawn[label].get_marker() == "o", f"{label!r} draws a lone point as nothing"

    # two different cells: still a warning, and still in the notices
    other = _record_of_cell(store, monkeypatch, "Cells/shm2/default.txt", name="run_c", seed=3,
                            omegas=(1.0, 2.0, 4.0), ratio=(1.1, 3.1, 2.1))
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="core"):
        rec = cmp.compare("repeats", [a, other], fig_sink=sink, store=store)
    warned = [r.getMessage() for r in caplog.records
              if r.name == "core.FDT.compare" and r.levelno == logging.WARNING]
    assert any("come from 2 cell(s)" in m for m in warned), warned
    assert any("come from 2 cell(s)" in n for n in rec.body["notices"]), rec.body["notices"]


def test_repeats_that_share_a_seed_are_named_as_one_run_drawn_twice(tmp_path, caplog):
    """A shared seed is a notice, never a refusal. Repeats of a cell use DIFFERENT seeds, and their
    spread is the measurement error. Two records of one seed and one setting are one run drawn twice
    -- a deliberate reproducibility check -- and their band has zero width; the record reported that
    as the measurement error without a word. The mode now groups the records by seed, names the ones
    that share one in the notices (logged as a warning: it changes what the band means) and records
    every seed in ``results``."""
    store = _store(tmp_path)
    ids = [build_fdt_record(store, name=name, seed=seed, omegas=(1.0, 2.0, 4.0), ratio=(1.0, 3.0, 1.0))
           for name, seed in (("twin_a", 7), ("twin_b", 7), ("other", 8))]
    _seen, sink = _closing()
    with caplog.at_level(logging.INFO, logger="core"):
        rec = cmp.compare("repeats", ids, fig_sink=sink, store=store)
    assert rec.body["results"]["seeds"] == [7, 7, 8], rec.body["results"]
    said = [n for n in rec.body["notices"] if "share seed" in n]
    assert len(said) == 1, rec.body["notices"]
    assert "'twin_a' and 'twin_b' share seed 7" in said[0] and "other" not in said[0], said
    assert "one run drawn twice" in said[0] and "not a measurement error" in said[0], said
    assert any("share seed 7" in r.getMessage() for r in caplog.records
               if r.levelno == logging.WARNING), "a shared seed changes what the band means"

    # every seed its own: nothing to say
    distinct = [build_fdt_record(store, name=f"rep{i}", seed=i, omegas=(1.0, 2.0, 4.0),
                                 ratio=(1.0, 3.0, 1.0)) for i in range(2)]
    rec = cmp.compare("repeats", distinct, fig_sink=sink, store=store)
    assert not [n for n in rec.body["notices"] if "share seed" in n], rec.body["notices"]
    assert rec.body["results"]["seeds"] == [0, 1]


def test_a_renormalised_blank_is_described_without_an_interpolation_it_never_did(tmp_path):
    """The blank notice every mode shares speaks of "the common grid" and of blanks "never
    interpolated across", and renormalise draws the record on its OWN grid and interpolates nothing
    -- so its record described a step that never ran. It says what is true of a rescaling: the run
    did not measure those points, and no constant brings them back. The interpolating modes keep
    their sentence."""
    store = _store(tmp_path)
    one = build_fdt_record(store, name="gappy", omegas=(1.0, 2.0, 4.0), ratio=(1.0, float("nan"), 1.5))
    _seen, sink = _closing()
    rec = cmp.compare("renormalise", [one], prefactor=3.0, fig_sink=sink, store=store)
    (said,) = [n for n in rec.body["notices"] if "blank" in n]
    assert "interpolat" not in said and "common grid" not in said, said
    assert said.startswith("1 of 3 points of the run's own grid are blank"), said
    assert "no constant brings them back" in said, said
    blanks, notices = cmp.blank_notice([np.array([1.0, np.nan])])
    assert blanks == 1 and "never interpolated across" in notices[0], "the default is unchanged"


def test_sweeps_that_do_not_overlap_are_refused_naming_each_sweep_and_its_range(tmp_path):
    """The old refusal, "Their ranges do not meet", named neither sweep nor range,
    and the ranges are taken over the operating points that FINISHED -- so two sweeps whose grids
    overlap on paper can be refused, and the operator could not tell why. The refusal names each
    sweep and the range of its finished points."""
    store = _store(tmp_path)
    om = (0.5, 1.0, 2.0)
    low = build_fdt_record(store, name="s_low", study="sweep", omegas=om,
                           points=[(0.0, (1.0, 1.0, 1.0)), (0.5, (1.0, 3.0, 1.1))])
    apart = build_fdt_record(store, name="s_apart", study="sweep", omegas=om,
                             points=[(2.0, (1.0, 2.0, 1.0)), (3.0, (1.0, 2.5, 1.0))])
    with pytest.raises(Refusal) as e:
        cmp.compare("sweeps", [low, apart], store=store)
    msg = str(e.value)
    assert e.value.field == "compare_records" and msg.startswith("The saved runs to compare must "
                                                                 "overlap in s"), msg
    assert "'s_low' covers [0, 0.5]" in msg and "'s_apart' covers [2, 3]" in msg, msg


def test_a_sweep_slice_prints_the_values_it_compared_to_the_places_it_compared_them(tmp_path):
    """Rows are compared rounded to twelve decimal places and were printed with ``:g`` -- six
    significant digits -- so a slice point typed with more digits produced a sentence that
    contradicts itself ("sliced at s = 0.1, the nearest ... to the slice point s = 0.1"). The
    sentence, the slice figure's title and the log line print to the precision the comparison is
    made at."""
    both = cmp._rows_notice("s", 0.1000001, ["a", "b"], [0.1, 0.1])
    assert both and "sliced at s = 0.1," in both[0] and "slice point s = 0.1000001" in both[0], both
    apart = cmp._rows_notice("s", 0.1, ["a", "b"], [0.1000001, 0.1000002])
    assert "'a' at s = 0.1000001" in apart[0] and "'b' at s = 0.1000002" in apart[0], apart

    store = _store(tmp_path)
    om = (0.5, 1.0, 2.0)
    pts = [(0.1, (1.0, 2.0, 1.0)), (0.3, (1.0, 4.0, 1.0))]
    a = build_fdt_record(store, name="p_a", study="sweep", omegas=om, points=pts)
    b = build_fdt_record(store, name="p_b", study="sweep", omegas=om, points=pts)
    titles = []

    def _sink(title, fig):
        from matplotlib import pyplot as plt
        titles.append(fig.axes[0].get_title())
        plt.close(fig)

    cmp.compare("sweeps", [a, b], at=0.1000001, fig_sink=_sink, store=store)
    assert "Sweep slice at s = 0.1000001" in titles, titles


def test_compare_renormalise_rescales_one_run_and_draws_it_against_the_original(tmp_path):
    """T_eff/T is LINEAR in the normalisation constant -- spectral.eff_temp_ratio is
    ``prefactor * omega * G / (4 chi'')`` -- so recomputing it with another constant is an exact
    rescaling of what the record already holds, and nothing is re-simulated. That is why the run
    records the prefactor it used: without it the stored ratio cannot be undone, and this
    mode would have to guess. A record that carries no usable constant is refused, naming it, rather
    than silently rescaling from a NaN."""
    store = _store(tmp_path)
    one = build_fdt_record(store, name="cellA", omegas=(1.0, 2.0, 4.0), ratio=(1.0, 4.0, 1.5),
                           prefactor=2.0)
    seen, sink = _closing()
    rec = cmp.compare("renormalise", [one], prefactor=3.0, fig_sink=sink, store=store)

    assert seen == ["FDT ratio renormalised"], seen
    assert rec.body["settings"]["prefactor"] == 3.0
    res = rec.body["results"]
    assert res["prefactor"] == 3.0 and res["prefactor_recorded"] == 2.0
    assert res["scale"] == pytest.approx(1.5)
    with h5py.File(rec.data_path, "r") as h5:
        labels = [h5["curves"][k].attrs["label"] for k in sorted(h5["curves"])]
        renormalised = h5["curves"]["001"][...]
    assert labels == ["cellA (recorded, 2)", "cellA (renormalised, 3)"], labels
    assert renormalised == pytest.approx([1.5, 6.0, 2.25])

    # a blank box and a non-positive constant are both refused, by name, before the comparison's
    # record is created -- so they leave no record behind
    for bad in (None, 0.0, -1.0):
        with pytest.raises(Refusal) as e:
            cmp.compare("renormalise", [one], prefactor=bad, store=store)
        assert e.value.field == "prefactor", (bad, e.value.field)
    assert len(store.list("fdt")) == 2, "a refused comparison leaves no record behind"

    no_pref = build_fdt_record(store, name="cellB", prefactor=float("nan"))
    before = _record_dirs(store)
    with pytest.raises(Refusal) as e:
        cmp.compare("renormalise", [no_pref], prefactor=3.0, store=store)
    assert e.value.field == "compare_records" and "'cellB'" in str(e.value), e.value
    # the recorded constant is judged before the comparison writes anything, so the writer removes
    # the record it opened for the refused comparison (a progressive record refused before its first
    # payload or figure): nothing is left behind
    assert _record_dirs(store) == before, "a refused comparison left a record behind"

    # a constant that IS recorded but is no normalisation constant (coupling / D_x is positive) is
    # refused too, saying what the record holds: "does not record one" would be false of it, and a
    # ratio computed with zero is zero everywhere, which no rescaling can undo
    zero = build_fdt_record(store, name="cellC", prefactor=0.0)
    negative = build_fdt_record(store, name="cellD", prefactor=-2.0)
    before = _record_dirs(store)
    for ref, who, value in ((zero, "cellC", 0.0), (negative, "cellD", -2.0)):
        with pytest.raises(Refusal) as e:
            cmp.compare("renormalise", [ref], prefactor=3.0, store=store)
        msg = str(e.value)
        assert e.value.field == "compare_records" and f"'{who}' records {value:g}" in msg, msg
    assert _record_dirs(store) == before, "a refused comparison left a record behind"


def _stub_measurement(monkeypatch):
    """Both campaigns of a REAL ``run_fdt`` replaced by deterministic arithmetic, so two runs of one
    cell measure identical spectra and susceptibilities and differ only in the constant applied.

    The spectrum stops at 10 while the default probe grid (0.1..30 around the peak at 1.0) reaches
    30, so the top probes come back blank, as a real run's do past the spectrum's Nyquist frequency;
    chi'' rises with the frequency, so the ratio is a curve and not a constant."""
    import torch
    from core.FDT import fdt_pipeline

    omegas_psd = torch.linspace(0.0, 10.0, 401, dtype=torch.float64)
    G = torch.exp(-((omegas_psd - 1.0) ** 2) / 0.02) + 1e-6
    t = torch.linspace(0.0, 1.0, 8, dtype=torch.float64)

    def _c1(cfg, return_trajectory=False):
        return (omegas_psd, G, t, torch.zeros_like(t)) if return_trajectory else (omegas_psd, G)

    def _c2(cfg, omegas):
        om = omegas.to(torch.float64)
        return torch.complex(1.0 / (1.0 + om), 0.5 + om)

    monkeypatch.setattr(fdt_pipeline, "run_campaign1_psd", _c1)
    monkeypatch.setattr(fdt_pipeline, "run_campaign2_chi", _c2)
    for name in ("plot_psd", "plot_eff_temp_ratio", "plot_chi_components",
                 "plot_spontaneous_trajectory"):
        monkeypatch.setattr(fdt_pipeline, name,
                            lambda *a, save_path=None, **k: Path(save_path).write_bytes(b"\x89PNG"))


def test_renormalising_a_real_run_equals_measuring_it_again_with_the_other_constant(store,
                                                                                     monkeypatch):
    """The physics behind the mode, checked against records ``run_fdt`` itself wrote rather than the
    fixture's. A Nadrowski cell is measured twice over identical (stubbed) campaigns: once with its
    own constant n*beta, once with that constant replaced by another. Renormalising the first record
    to the second constant must give the second run's ratio -- that is the question the mode answers
    ("what would this run have said with a different constant") -- and it must do so as the exact
    product recorded * (new / recorded) wherever the run measured something, with every blank still
    blank: a blank is a frequency the run could not measure, and no constant brings it back.

    The drawn grid is the record's OWN grid, bit for bit, and each curve in the comparison's file
    carries the constant ITS numbers were computed with."""
    from core import cli, config, registry
    from core.FDT import fdt_pipeline
    from core.FDT.campaigns import observable_noise_prefactor

    _stub_measurement(monkeypatch)
    cell = str(config.CELL_PATH / "nadrowski" / "master_spont.txt")
    cfg = cli.make_fdt_config("NADROWSKI", registry.state_dep_drift("NADROWSKI"), cell,
                              n_freqs=12, ensemble_M=8)
    own = observable_noise_prefactor(cfg)
    want = 2.5 * own
    measured = fdt_pipeline.run_fdt(cfg, skip_sanity=True, confirm_production=True,
                                    writer=store.create("fdt", cfg, name="measured"), seed=5)
    monkeypatch.setattr(fdt_pipeline, "observable_noise_prefactor", lambda _cfg: want)
    again = fdt_pipeline.run_fdt(cfg, skip_sanity=True, confirm_production=True,
                                 writer=store.create("fdt", cfg, name="measured_again"), seed=5)

    _seen, sink = _closing()
    rec = cmp.compare("renormalise", ["measured"], prefactor=want, fig_sink=sink, store=store)

    with h5py.File(measured.data_path, "r") as h5:
        grid, recorded = h5["omega_grid"][...], h5["T_eff_over_T"][...]
        assert float(h5.attrs["prefactor"]) == own, "the run records the constant it applied"
        omega_0 = float(h5.attrs["omega_0"])
    with h5py.File(again.data_path, "r") as h5:
        remeasured = h5["T_eff_over_T"][...]
        assert float(h5.attrs["prefactor"]) == want
    blank = np.isnan(recorded)
    assert blank.any() and not blank.all(), recorded.tolist()
    assert np.array_equal(np.isnan(remeasured), blank), "one grid, one spectrum: the same blanks"

    with h5py.File(rec.data_path, "r") as h5:
        drawn_grid = h5["omega_common"][...]
        original, renormalised = h5["curves"]["000"][...], h5["curves"]["001"][...]
        constants = [(float(h5["curves"][k].attrs["omega_0"]),
                      float(h5["curves"][k].attrs["prefactor"])) for k in ("000", "001")]
        labels = [h5["curves"][k].attrs["label"] for k in ("000", "001")]
    assert np.array_equal(drawn_grid, grid), "the record's own grid, never a regenerated copy"
    assert np.array_equal(original, recorded, equal_nan=True), "the original is drawn as recorded"
    assert np.array_equal(np.isnan(renormalised), blank), "blanks stay blanks, and nothing else is"
    assert np.array_equal(renormalised[~blank], recorded[~blank] * (want / own)), "exact rescaling"
    assert renormalised[~blank] == pytest.approx(remeasured[~blank], rel=1e-12), \
        "the rescaled ratio is what a run with the other constant measured"
    assert constants == [(omega_0, own), (omega_0, want)], constants
    assert labels == [f"master_spont (recorded, {own:g})", f"master_spont (renormalised, {want:g})"]

    res = rec.body["results"]
    assert res["prefactor_recorded"] == own and res["prefactor"] == want
    assert res["scale"] == pytest.approx(2.5) and res["blanks"] == int(blank.sum()), res
    assert res["n_records"] == 1 and res["n_grid"] == grid.size, res
    peak = res["per_record"][0]
    assert peak["peak_ratio"] == pytest.approx(float(np.nanmax(renormalised)))
    assert rec.body["notices"] and "no constant brings them back" in rec.body["notices"][0], \
        "renormalise interpolates nothing, and its notice says so"
    assert [r["id"] for r in rec.body["compared"]["records"]] == [measured.id]


def test_compare_sweeps_draws_both_surfaces_and_one_slice_through_them(tmp_path):
    """The fourth mode. Two sweeps answer "does FDT come back" along one parameter each, and
    the question this mode exists for is whether they agree -- which is read at ONE operating point,
    not off two surfaces side by side. The slice point defaults to the middle of the range the two
    sweeps share, is recorded, and each sweep contributes the row nearest it (the two grids are set
    independently, so an exact match is not something to require). Two sweeps of DIFFERENT parameters
    are refused: "the same operating point" means nothing across an S sweep and a temperature one."""
    store = _store(tmp_path)
    om = (0.5, 1.0, 2.0)
    a = build_fdt_record(store, name="s_low", study="sweep", omegas=om, sweep_param="s",
                         points=[(0.0, (1.0, 1.0, 1.0)), (0.5, (1.0, 3.0, 1.1))])
    b = build_fdt_record(store, name="s_high", study="sweep", omegas=om, sweep_param="s",
                         points=[(0.25, (1.0, 2.0, 1.0)), (0.75, (1.0, 6.0, 1.2))])
    seen, sink = _closing()
    rec = cmp.compare("sweeps", [a, b], fig_sink=sink, store=store)

    assert seen == ["Sweep surfaces", "Sweep slice"], seen
    res = rec.body["results"]
    assert res["param"] == "s"
    # the shared range is [0.25, 0.5]; its middle is 0.375, and the nearest rows are 0.5 and 0.25
    assert res["at"] == pytest.approx(0.375)
    assert [p["param_value"] for p in res["per_record"]] == [0.5, 0.25]
    assert res["per_record"][0]["peak_ratio"] == pytest.approx(3.0)
    with h5py.File(rec.data_path, "r") as h5:
        assert sorted(h5["curves"]) == ["000", "001"]
        assert h5["curves"]["000"].attrs["label"].startswith("s_low")

    # an operating point outside the range they share is refused, naming the setting
    with pytest.raises(Refusal) as e:
        cmp.compare("sweeps", [a, b], at=9.0, store=store)
    assert e.value.field == "slice_at" and "0.25" in str(e.value), e.value

    # and two different swept parameters are refused, naming both
    t = build_fdt_record(store, name="t_sweep", study="sweep", omegas=om, sweep_param="temp",
                         points=[(1.0, (1.0, 2.0, 1.0)), (1.5, (1.0, 4.0, 1.1))])
    with pytest.raises(Refusal) as e:
        cmp.compare("sweeps", [a, t], store=store)
    assert e.value.field == "compare_records" and "temp" in str(e.value), e.value


def _sweep_sink():
    """``(lines, sink)``: a sink that closes every figure and, for the surfaces figure, records each
    surface's dashed horizontal line -- ``{surface title: [heights]}`` -- which is where that sweep
    was sliced. The colorbars' axes carry no title and are skipped."""
    lines = {}

    def _sink(title, fig):
        from matplotlib import pyplot as plt
        if title == "Sweep surfaces":
            lines.update({ax.get_title(): [float(ln.get_ydata()[0]) for ln in ax.get_lines()
                                           if ln.get_linestyle() == "--"]
                          for ax in fig.axes if ax.get_title()})
        plt.close(fig)
    return lines, _sink


def test_a_sweep_slice_breaks_a_near_tie_by_its_rule_and_says_which_rows_it_compared(tmp_path):
    """Each sweep contributes its row NEAREST the slice point, and "nearest" must not be decided by
    float noise: from 0.2, the rows 0.1 and 0.3 are 0.1 and 0.09999999999999998 away, and a raw
    comparison picks 0.3 for a reason nobody chose. Distances are compared rounded, and a tie goes to
    the LOWER value, the rule ``sweep_slice`` states. A slice that could not take a row AT the slice
    point says so in the record's notices, naming the row it took; and each surface's dashed line
    marks the row that sweep was actually sliced at, which is not the slice point when no row sits
    on it. A slice point both sweeps measured has nothing to explain."""
    store = _store(tmp_path)
    om = (0.5, 1.0, 2.0)
    pts = [(0.1, (1.0, 2.0, 1.0)), (0.3, (1.0, 4.0, 1.0))]
    a = build_fdt_record(store, name="tie_a", study="sweep", omegas=om, points=pts)
    b = build_fdt_record(store, name="tie_b", study="sweep", omegas=om, points=pts)
    assert abs(0.3 - 0.2) < abs(0.1 - 0.2), "the float noise this test is about"

    lines, sink = _sweep_sink()
    rec = cmp.compare("sweeps", [a, b], at=0.2, fig_sink=sink, store=store)
    assert [p["param_value"] for p in rec.body["results"]["per_record"]] == [0.1, 0.1]
    assert lines == {"tie_a": [0.1], "tie_b": [0.1]}, lines
    said = [n for n in rec.body["notices"] if "sliced at" in n]
    assert len(said) == 1 and "Both sweeps were sliced at s = 0.1" in said[0], rec.body["notices"]
    assert "slice point s = 0.2" in said[0], said

    lines.clear()
    rec = cmp.compare("sweeps", [a, b], at=0.3, fig_sink=sink, store=store)
    assert [p["param_value"] for p in rec.body["results"]["per_record"]] == [0.3, 0.3]
    assert lines == {"tie_a": [0.3], "tie_b": [0.3]}, lines
    assert not any("sliced at" in n for n in rec.body["notices"]), rec.body["notices"]


def _resonance_of(cfg_op) -> float:
    """The stubbed cell's resonance at an operating point: it moves with S, as a real cell's does."""
    return 1.0 + 2.0 * float(cfg_op.params_dict["s"][0])


def _stub_sweep(monkeypatch, *, fail_s=()):
    """Both campaigns of a REAL ``run_fdt_param_sweep`` replaced by one physical law.

    The spontaneous spectrum peaks at the resonance ``1 + 2 S`` and the detector takes that peak, so
    a sweep's reference ``omega_0_ref`` -- its LARGEST resonance -- depends on the range of S it
    covers, as a real sweep's does: two sweeps over different ranges divide their axes by different
    numbers. The ratio is ``1 + S + 0.5 log(omega / (1 + 2 S))``, a law of the ABSOLUTE frequency and
    S alone, and linear in log-omega, so interpolation reproduces it exactly and one operating point
    measured by two sweeps must read the same from both. A driven campaign at an S in ``fail_s``
    raises, as a point that failed: the sweep records it failed, with an ``error`` attribute and no
    response data, and goes on. The sweep's own figure is stubbed to the file it was handed."""
    import torch
    from core.FDT import cross_validation as cv

    freqs = torch.linspace(0.05, 70.0, 1400, dtype=torch.float64)

    def _c1(cfg_op):
        return freqs, torch.exp(-((freqs - _resonance_of(cfg_op)) ** 2) / 0.02) + 1e-6

    def _c2(cfg_op, omegas, freqs_psd, G):
        s = float(cfg_op.params_dict["s"][0])
        if any(abs(s - f) < 1e-9 for f in fail_s):
            raise RuntimeError(f"stub driven-campaign failure at S = {s:g}")
        om = omegas.to(torch.float64)
        return (torch.ones(om.shape, dtype=torch.complex128),
                1.0 + s + 0.5 * torch.log(om / _resonance_of(cfg_op)))

    monkeypatch.setattr(cv, "run_campaign1_psd", _c1)
    monkeypatch.setattr(cv, "_detect_resonance",
                        lambda omegas, G, w0: (float(omegas[int(torch.argmax(G))]), True))
    monkeypatch.setattr(cv, "_campaign2_ratio", _c2)
    monkeypatch.setattr(cv, "plot_fdt_3d_vs_param",
                        lambda *a, save_path=None, **k: Path(save_path).write_bytes(b"\x89PNG"))


def test_compare_sweeps_reads_one_measurement_the_same_from_both_sweeps(store, monkeypatch):
    """Records ``run_fdt_param_sweep`` itself wrote (campaigns stubbed by one physical law), so the
    fixture's layout is checked against the sweep's own. Two S sweeps of ONE cell over different
    ranges: each divides its axis by its OWN largest resonance, so the two references differ, and a
    slice drawn on those normalised axes would read one measurement as two disagreeing ones. The
    slice is drawn on the ABSOLUTE frequency axis instead, where the same operating point measured by
    both sweeps gives one curve.

    The default slice point is 0.25, which both sweeps straddle: sweep_a's rows 0.2 and 0.3 are
    0.05000000000000002 and 0.04999999999999999 away, and only the stated tie rule (rounded
    distances, the lower value) sends both sweeps to the same row, 0.2. A point whose driven campaign
    failed -- kept in the file with an ``error`` and no response data -- is never the row sliced,
    however near it sits; and when the two rows differ, the record says so, naming both.

    Two sweeps of the same cell share its name, so their curves are told apart by the run
    (``labelled_curves``' rule), and each curve carries its own constants: ``omega_0`` its
    row's own resonance, the ``prefactor`` NaN (a sweep records one constant for the cell and none
    per operating point), plus its swept value and its sweep's reference as provenance."""
    from core import cli, config
    from core.FDT import cross_validation as cv

    _stub_sweep(monkeypatch, fail_s=(0.25,))
    cell = str(config.CELL_PATH / "nadrowski" / "master_spont.txt")

    def sweep(name, lo, hi):
        cfg, s_grid, _t = cli.make_param_sweep_config(
            cell, preset=dict(cli.SWEEP_PRESETS["exploratory"]), preset_name="exploratory",
            s_spec=(lo, hi, 4), t_spec=(1.0, 1.1, 2), n_freqs=8, ensemble_M=2, seed=1)
        return cv.run_fdt_param_sweep(cfg, "s", s_grid, {"temp": 1.0},
                                      writer=store.create("fdt", cfg, name=name))

    a = sweep("sweep_a", 0.0, 0.3)              # S = 0, 0.1, 0.2, 0.3
    b = sweep("sweep_b", 0.2, 0.5)              # S = 0.2, 0.3, 0.4, 0.5
    with pytest.warns(UserWarning, match="operating points failed"):
        c = sweep("sweep_c", 0.15, 0.45)        # S = 0.15, 0.25 (its driven campaign fails), ...
    assert (a.body["points"]["failed"], b.body["points"]["failed"], c.body["points"]["failed"]) == \
        (0, 0, 1)

    def reference(rec):
        with h5py.File(rec.data_path, "r") as h5:
            return float(h5.attrs["omega_0_ref"])

    def row_at(rec, s):
        return next(r for r in cv.load_param_sweep(rec.data_path) if abs(r["param_value"] - s) < 1e-9)

    # the precondition: each sweep's reference is its own largest resonance, and they differ
    assert reference(a) == pytest.approx(1.6, abs=0.06), reference(a)
    assert reference(b) == pytest.approx(2.0, abs=0.06), reference(b)

    lines, sink = _sweep_sink()
    rec = cmp.compare("sweeps", ["sweep_a", "sweep_b"], fig_sink=sink, store=store)
    res = rec.body["results"]
    assert res["param"] == "s" and res["shared_range"] == pytest.approx([0.2, 0.3]), res
    assert res["at"] == pytest.approx(0.25)
    assert [p["param_value"] for p in res["per_record"]] == pytest.approx([0.2, 0.2]), res
    labels = ["master_spont (sweep_a)", "master_spont (sweep_b)"]
    assert set(lines) == set(labels) and all(v == pytest.approx([0.2]) for v in lines.values()), \
        "each surface marks the row it was sliced at, not the slice point 0.25"
    assert any("Both sweeps were sliced at s = 0.2" in n for n in rec.body["notices"]), \
        rec.body["notices"]
    with h5py.File(rec.data_path, "r") as h5:
        grid = h5["omega_common"][...]
        curves = [h5["curves"][k][...] for k in ("000", "001")]
        attrs = [dict(h5["curves"][k].attrs) for k in ("000", "001")]
    both = np.isfinite(curves[0]) & np.isfinite(curves[1])
    assert both.sum() >= 5, curves
    assert curves[0][both] == pytest.approx(curves[1][both], rel=1e-9, abs=1e-12), \
        "one operating point, measured by two sweeps, reads the same from both"
    assert curves[0][both] == pytest.approx(1.2 + 0.5 * np.log(grid[both] / 1.4), rel=1e-9), \
        "and it is the measurement, on the absolute axis"
    # the intersection of the two sweeps' ABSOLUTE grids: sweep_b's start (0.2 x its lowest
    # resonance, 1.4) and sweep_a's end (30 x its highest, 1.6), in the preset's freq_bounds
    assert grid[0] == pytest.approx(0.2 * 1.4, rel=0.05), grid
    assert grid[-1] == pytest.approx(30 * 1.6, rel=0.05), grid
    for attr, rec_of in zip(attrs, (a, b)):
        assert attr["label"].startswith("master_spont (sweep_"), attr
        assert attr["omega_0"] == row_at(rec_of, 0.2)["omega_0_resonance"], "the row's own resonance"
        assert math.isnan(attr["prefactor"]), attr
        assert attr["param_value"] == pytest.approx(0.2) and attr["omega_0_ref"] == reference(rec_of)
    assert [r["id"] for r in rec.body["compared"]["records"]] == [a.id, b.id]

    # sweep_c's point 0.25 failed: from 0.24 it is by far the nearest, and it is never the row sliced
    lines.clear()
    rec = cmp.compare("sweeps", ["sweep_a", "sweep_c"], at=0.24, fig_sink=sink, store=store)
    res = rec.body["results"]
    assert res["shared_range"] == pytest.approx([0.15, 0.3]), res
    assert [p["param_value"] for p in res["per_record"]] == pytest.approx([0.2, 0.15]), res
    assert lines["master_spont (sweep_a)"] == pytest.approx([0.2]), lines
    assert lines["master_spont (sweep_c)"] == pytest.approx([0.15]), lines
    said = [n for n in rec.body["notices"] if "different operating points" in n]
    assert len(said) == 1, rec.body["notices"]
    assert "'master_spont (sweep_a)' at s = 0.2" in said[0], said
    assert "'master_spont (sweep_c)' at s = 0.15" in said[0] and "s = 0.24" in said[0], said
