"""Comparing saved FDT records (piece 5, spec §7; decision E8).

The comparison facility is the last part of the piece and nothing before it depends on it (E12).
This suite covers the shared machinery -- the common grid, the interpolation that never blends a
blank away, and what a comparison refuses before it draws anything -- and then one test per mode.

Nothing here simulates. Every record is written straight into a store by ``build_fdt_record``, the
way ``build_browse_store`` writes its rows (``cfg=None``, the smallest body the kind allows), so the
whole file costs the writer's git calls and no solver at all.

Run:  pytest tests/test_fdt_compare.py
"""
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
from tests._fixtures import build_fdt_record                       # noqa: E402


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
    """Spec §7.2. Every run detects its own resonance and builds its grid around it, so two runs of
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
    """Spec §7.2, and the whole point of the off-grid fix (E9): a frequency the run could not measure
    comes back blank, and a comparison that quietly interpolated over it would put the fabricated tail
    back -- the exact defect §1 of the spec measured in ``_interp_log``. A target between two good
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
    """Review Focus item 5, and spec §7.1's "a comparison draws only records whose complete is true".
    Every mode's arity is a real input class -- one cell is not a comparison, and a sweep drawn as a
    single-cell run reads a layout that is not there -- and E2 means half-written records sit on disk
    carrying a valid manifest, so the flag can hand one over by id or name (the window's picker offers
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


def test_every_refusal_lands_before_the_comparison_opens_its_record(tmp_path, monkeypatch):
    """Refuse before the spend (P49, F57-F59). A progressive record exists from the moment its writer
    is entered, and E2 keeps it through an exception -- so a refusal raised after that point would
    leave a comparison record behind that holds nothing. Every refusal a comparison can make from
    what it was GIVEN is therefore made before the record opens, and each one leaves the kind
    directory exactly as it found it:

    the mode and its settings (a mode this build does not draw, a setting the mode has no use for, a
    blank or non-positive normalisation constant, a non-finite slice point); the arity, both ways (a
    mode that draws one record or two is never handed more and silently draws the first), counted in
    distinct RUNS (one run named twice is one run); and each record (one that names nothing, one of
    another study, one that did not finish, one with no numbers beside its manifest). The ref that names nothing is refused under the comparison's own
    field, not the store's ``artifact``, so both front ends name the control that answers it (F58)."""
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
    before = _record_dirs(store)
    for label, (mode, refs, options), field, words in cases:
        with pytest.raises(Refusal) as e:
            cmp.compare(mode, refs, store=store, **options)
        assert e.value.field == field, (label, e.value.field, str(e.value))
        assert words in str(e.value), (label, str(e.value))
        assert _record_dirs(store) == before, f"{label}: a refused comparison left a record behind"

    # A bare string is not a list of refs: iterated, a run's id would be read as one ref per
    # character ("'2' names no fdt record"). No front end can hand one over -- --record appends, and
    # the window's list returns a list -- so it is the caller's bug, raised as one (a TypeError, which
    # both front ends show as a bug) rather than dressed up as a refusal of the runs.
    with pytest.raises(TypeError, match="a list of refs"):
        cmp.compare("cells", a, store=store)
    assert _record_dirs(store) == before

    # the no-drawer guard names what this build DOES draw (P11), once the table is back to real
    monkeypatch.delitem(cmp._DRAWERS, "cells")
    with pytest.raises(Refusal) as e:
        cmp.compare("cells", [a, b], store=store)
    assert e.value.field == "compare_records" and "no cells comparison" in str(e.value), e.value
    assert "(it draws renormalise, repeats, sweeps)" in str(e.value), e.value
    assert _record_dirs(store) == before


def test_a_comparison_is_a_record_that_names_the_mode_and_the_runs_it_drew(tmp_path, monkeypatch):
    """Spec §7.3: a comparison is an ``fdt`` record of its own, with ``study = "comparison"``, its
    sources in ``body.compared`` and NOT in ``parents`` (the parents block is a flat {key: id} map and
    cannot carry an arbitrary number of ids -- spec §1.2), the common grid and the interpolated curves
    in ``data.h5``, and its figures as its output. Driven through a stub drawer, so this pins the
    record and not any one mode's picture.

    Every ``data.h5`` carries ``omega_0`` and ``prefactor`` at its root (the layout contract); for a
    comparison neither has one value, so the root holds NaN and each curve carries the constants of
    the record it came from (F55) -- which is what a renormalised curve needs to be read correctly."""
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
    """Spec §7.3 and §8.2, design ruling R5: deleting a run a comparison drew is NOT refused -- the
    sources live in the body, which the dependency check does not read -- and the comparison stays
    what it was: listed, loadable, and honest about the gap, its lineage printing
    ``MISSING fdt [<id>]`` for the run that is gone and the ordinary line for the one that is not."""
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
    """E2, for the comparison's own record: a cancel after the record opened leaves it on disk, marked
    unfinished, holding what was written -- the same rule every fdt record follows, and the one the
    tool's interrupt note describes. Nothing resumes it (F56): the records it drew are untouched and
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
    """Task 17's lesson, applied to the comparison: importing any ``core.diagnostics`` submodule runs
    that package's ``__init__``, which loads the SBI stack and prints two false pytensor "g++" lines
    at the head of the run. A comparison draws saved numbers and needs none of it. RUN in a fresh
    interpreter -- this process holds the orchestrator through the session fixtures, so a
    ``sys.modules`` check here would pass vacuously -- over records this process wrote. The bare
    ``sbi`` package is the store's (``provenance.env_info`` reads its version into every manifest);
    every other ``sbi`` module is the inference stack and is forbidden here."""
    root = tmp_path / "artifacts"
    store = ArtifactStore(root)
    ids = [build_fdt_record(store, name=f"cell_{i}") for i in range(2)]
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
        "rec = cmp.compare('cells', sys.argv[2:], store=ArtifactStore(sys.argv[1]))\n"
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
    """Spec §7.1: several cells' ratio curves on ONE axis, labelled by cell. The labels matter as much
    as the curves -- a picture of four unlabelled traces answers nothing -- and the peak of each is in
    the record, so the comparison can be read back without re-opening the figure. The two records here
    measure different bands on purpose: the drawn grid is their intersection (§7.2), not either one's
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
    """E9 carried through a comparison. A run that could not measure a frequency reports a blank, and
    the comparison must not blend it away: the blank stays blank, the count is in the record's
    notices, and the notices are what the walkthrough row reads. A silently interpolated gap is the
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
    """Spec §7.1 and E7: repeats of one cell run at different seeds, and the spread across them IS the
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
    are told apart by the run. ``repeats`` names the cells it drew (P60) by the same rule, so its
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
