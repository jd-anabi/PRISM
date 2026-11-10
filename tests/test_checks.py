"""The checks as values: what each of the twelve is, how a record of a deciding check is read, which records count for
what is in use, the headline over the six deciding checks and the summary the chain and Home read, the comparison of
the unusable driven recordings a training simulated with its prior's audit, and the line "Before you start" says of the
two checks made before a training.

No widget: the rules are pure functions over plain values, handed records as the store writes them, either read back
from the test's own store or written out here as the engine writes them.
"""
import ast
import inspect
import itertools
import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from core.gui import check_catalogue, check_lineage, check_readings
from core.gui.check_lineage import CheckRow, Held, RecordRead
from core.gui.check_readings import Reading
from core.gui.texts import date_text
from tests._runs import SCOPE_PROCESS as PROCESS
from tests._runs import SCOPE_WHOLE as WHOLE

REPO = Path(__file__).resolve().parents[1]

PRIOR, OTHER_PRIOR = "20261001T222935", "20261001T222936"
POSTERIOR, OTHER_POSTERIOR = "20261003T031200", "20261003T031201"
SET = "eda5c2a1e468"
#: The configuration block a record made on the configuration in use holds, the fields the rules compare.
CONFIG = {"model": "NADROWSKI", "param_keys": ["k", "lam", "f_max", "x_scale", "t_scale", "T"],
          "nd_lows": [0.01, 0.1, 0.05], "nd_highs": [5.0, 50.0, 30.0], "rescale_lows": [10.0, 1.0, 280.0],
          "rescale_highs": [100.0, 40.0, 310.0]}
#: The settings of the frequency response a probe check's record states, as the engine writes them.
CONFIGURED = {"chi_f0": 0.15, "chi_freq_bounds": [0.03, 0.3], "chi_k_pad": 12, "chi_min_cycles": 2.0,
              "chi_max_cycles": 20.0, "probe_count": 6}


def _at(day, hour=12, minute=0):
    """A local moment of October 2026."""
    return datetime(2026, 10, day, hour, minute).astimezone()


def _said(moment) -> str:
    return date_text(moment.astimezone())


def _held(**over) -> Held:
    """What is in use: a posterior trained with the frequency response on the prior in use, its set's counts whole, the
    rotation on; each field changed by ``over``."""
    fields = dict(prior=PRIOR, posterior=POSTERIOR, posterior_prior=PRIOR, simulation=SET, chi=True, config=CONFIG,
                  configured=CONFIGURED, rotated=True, narrowed=False, nadrowski=True, forcing=True, pair_refused="",
                  set_counts=(8_390, 22_144, 100, WHOLE), set_done=True, prior_words="the prior retrain_prior",
                  posterior_words="the posterior retrain", set_when=_at(3, 9, 0))
    fields.update(over)
    return Held(**fields)


def _record(check: str, *, created=None, ident=None, parents=None, config=None, cell="Cells/nadrowski/a.txt",
            body=None, **given) -> RecordRead:
    """One record of the check ``check`` as the store writes its manifest: its kind, its check and its form, its
    parents (the posterior and the prior in use for a record of a posterior, the prior for an audit), the configuration
    block, the input cell file, and the body ``check_results`` gives with ``given``; ``created`` a local moment."""
    from tests._fixtures import check_results
    one = check_catalogue.entry(check)
    created = created or _at(3, 14, 2)
    stamp = f"{created.astimezone(timezone.utc):%Y%m%dT%H%M%S}"
    if parents is None:
        parents = ({"prior": PRIOR} if check == "audit" else {"posterior": POSTERIOR, "prior": PRIOR}
                   if one.kind == "calibration" or check in ("input_use", "identifiability", "fixed_number",
                                                             "across_prior") else {})
    held = dict(check_results(check, **given)) if body is None else dict(body)
    if one.kind == "diagnostic":
        variant = one.variant if one.variant is not None else ("pooled" if check == "fixed_number" else None)
        held = {"diagnostic": one.diagnostic, "variant": variant, **held}
    manifest = {"kind": one.kind, "id": ident or stamp, "name": "", "created": created.astimezone(timezone.utc).isoformat(),
                "parents": parents, "config": dict(CONFIG if config is None else config),
                "inputs": {"cell": None if cell is None else {"path": cell, "sha256": "ab" * 32}}, "body": held}
    return RecordRead(f"/records/{check}/{ident or stamp}", manifest)


def _row(rows, key) -> CheckRow:
    return next(row for row in rows if row.key == key)


# ── the twelve ──
def test_the_twelve_checks_stand_in_their_order_with_their_words():
    """Twelve checks in the list's order, each under its key, its number, the name and sentence the window gives it,
    its group, its role, its button's and its stop button's words, the kind of record it writes and which check record
    it is; the groups and the roles in their order, the six that decide, and the words once a record exists."""
    checks = check_catalogue.CHECKS
    assert [(one.key, one.number) for one in checks] == list(zip(
        ("band", "audit", "comparison", "calibration", "input_use", "identifiability", "pin_down", "timing",
         "fixed_number", "card_test", "drive_study", "across_prior"), range(1, 13)))
    assert [(one.name, one.sentence) for one in checks] == [
        ("Frequency band", "Can driven recordings across the frequency band, at the set drive amplitude, be measured on "
                           "this cell without the drive taking over the cell's own oscillation?"),
        ("Unusable driven recordings: audit", "What share of the driven recordings that training will simulate from "
                                              "this prior is unusable, and is that share in the expected range?"),
        ("Unusable driven recordings: simulation set against the audit",
         "Does the share of unusable driven recordings in the finished simulation set agree with the audit?"),
        ("Calibration", "Is the posterior honest about its uncertainty?"),
        ("Use of each part of the input", "Does the network respond to every part of an observation?"),
        ("Identifiability of the posterior",
         "Is it on record how strongly the posterior pins down each direction in parameter space?"),
        ("What the observations can pin down", "At this cell and recording length, which parameters do the "
                                               "observations pin down, and which move an observation alike?"),
        ("Timing of a short fit",
         "How long does one epoch of the fit take on this computer, and so how long should the full fit take?"),
        ("Calibration repeated at fixed numbers of driven recordings",
         "Does the rank test still pass when every observation holds the same number of driven recordings?"),
        ("Card test", "Does every stage run end to end on this graphics card?"),
        ("Drive study", "How hard can this cell be driven before the drive takes over its own oscillation?"),
        ("Identifiability across the prior",
         "Could recordings pin each parameter down at points across the prior, whatever the posterior learned?")]
    assert [(one.group, one.role) for one in checks] == (
        [("Before the run", "Decides")] * 2 + [("After simulating", "Decides")] + [("After the fit", "Decides")] * 3
        + [("Before the run", "Informs")] * 2 + [("After the fit", "Informs")] + [("Tools", "Tool")] * 3)
    assert [one.button for one in checks] == [
        "Run the frequency-band check", "Run the audit", "", "Run the calibration", "Run the input-use check",
        "Run the identifiability check"] + ["Run this check"] * 6
    assert [one.stop_words for one in checks] == [
        "Stop this check…", "Stop the audit…", "Stop this check…", "Stop the calibration…"] + ["Stop this check…"] * 8
    assert [(one.kind, one.diagnostic, one.variant) for one in checks] == [
        ("diagnostic", "probes", "band"), ("diagnostic", "probes", "mask"), ("", None, None),
        ("calibration", None, None), ("diagnostic", "ablation", None), ("diagnostic", "identifiability", "rotation"),
        ("diagnostic", "identifiability", "jacobian"), ("", None, None), ("diagnostic", "sbc", None), ("", None, None),
        ("diagnostic", "probes", "drive"), ("diagnostic", "identifiability", "laplace")]
    assert check_catalogue.GROUPS == ("Before the run", "After simulating", "After the fit", "Tools")
    assert check_catalogue.ROLES == ("Decides", "Informs", "Tool")
    assert check_catalogue.DECIDING == ("band", "audit", "comparison", "calibration", "input_use", "identifiability")
    assert check_catalogue.AGAIN_WORDS == "Run again"
    assert all(check_catalogue.entry(one.key) is one for one in checks)
    with pytest.raises(KeyError, match="calibrate"):
        check_catalogue.entry("calibrate")


# ── reading a deciding check ──
_READ_CASES = {
    "band, both hold": ("band", {}, "Passed"),
    "band, the band not judged": ("band", {"band_holds": None}, "Look closer"),
    "band, the drive not judged": ("band", {"drive_holds": None}, "Look closer"),
    "band, neither judged": ("band", {"band_holds": None, "drive_holds": None}, "Look closer"),
    "band, the band fails": ("band", {"band_holds": False}, "Failed"),
    "band, the drive fails": ("band", {"drive_holds": False}, "Failed"),
    "band, one fails and one not judged": ("band", {"band_holds": False, "drive_holds": None}, "Failed"),
    "audit at 37.9 per cent": ("audit", {"masked_fraction": 0.379}, "Passed"),
    "audit 7 points over": ("audit", {"masked_fraction": 0.44}, "Passed"),
    "audit 7 points under": ("audit", {"masked_fraction": 0.30}, "Passed"),
    "audit 9 points over": ("audit", {"masked_fraction": 0.46}, "Look closer"),
    "audit at 47.1 per cent": ("audit", {"masked_fraction": 0.471}, "Look closer"),
    "audit 10 points under": ("audit", {"masked_fraction": 0.27}, "Look closer"),
    "audit 11 points over": ("audit", {"masked_fraction": 0.48}, "Look closer"),
    "audit at 50 per cent": ("audit", {"masked_fraction": 0.50}, "Failed"),
    "audit 13 points under": ("audit", {"masked_fraction": 0.24}, "Failed"),
    "audit with non-finite lock-ins": ("audit", {"masked_fraction": 0.379, "nonfinite": 3}, "Look closer"),
    "audit far and with lock-ins": ("audit", {"masked_fraction": 0.50, "nonfinite": 3}, "Failed"),
    "audit whose invariants fail": ("audit", {"invariants_hold": False}, "Failed"),
    "audit whose attribution fails": ("audit", {"attribution_holds": False}, "Failed"),
    "audit whose cross-check fails": ("audit", {"cross_check_holds": False}, "Failed"),
    "calibration that passed": ("calibration", {"passed": True}, "Passed"),
    "calibration that failed": ("calibration", {"passed": False}, "Failed"),
    "input use, both healthy": ("input_use", {}, "Passed"),
    "input use, the first named part compressed": ("input_use", {"a1": "compressed"}, "Failed"),
    "input use, the second named part invisible": (
        "input_use", {"d3": "*** INVISIBLE -- the whole physical range does nothing ***"}, "Failed"),
    "input use, the first named part non-finite": (
        "input_use", {"a1": "NON-FINITE -- this perturbation drove the network to a NaN/Inf output"}, "Failed"),
    "input use, a third part invisible": ("input_use", {"invisible": 1}, "Look closer"),
    "input use, a third part non-finite": ("input_use", {"nonfinite": 1}, "Look closer"),
    "the rotation's directions on record": ("identifiability", {}, "Passed"),
    "the rotation's directions not kept": ("identifiability", {"eigenvalues": None}, "Failed"),
    "what the observations pin down": ("pin_down", {}, "Done"),
    "calibration at a fixed number": ("fixed_number", {}, "Done"),
    "the drive study": ("drive_study", {}, "Done"),
    "identifiability across the prior": ("across_prior", {}, "Done"),
}


@pytest.mark.parametrize("check, given, state", list(_READ_CASES.values()), ids=list(_READ_CASES))
def test_each_deciding_reading_follows_its_rule(check, given, state):
    """Each deciding check's record reads its state by its rule: the band by its two verdicts, the audit by its share
    against 37 % (within 8 points fine, beyond 12 failed), its three consistency records and its non-finite lock-ins,
    the calibration by its verdict, the use of each part of the input by its two named parts and the parts invisible or
    non-finite, and the identifiability of the posterior by its directions being on record; an informing check's or a
    tool's record reads "Done". Each record is at its usual size and counts."""
    from tests._fixtures import check_results
    assert check_readings.read(check, {"body": check_results(check, **given)}) == Reading(state)


def test_a_record_that_cannot_be_read_never_counts_and_raises_nothing():
    """A record whose deciding field is missing, of another shape or of no true-or-false, reads "Not run yet", says its
    record cannot be read and does not count; the comparison and the two checks that keep nothing write no record."""
    from tests._fixtures import check_results
    unread = Reading("Not run yet", "Its record cannot be read, so it does not count.", counts=False)
    band, audit = check_results("band"), check_results("audit")
    sizeless = {key: value for key, value in audit["settings"].items() if key != "run_size"}
    for check, manifest in (("band", {}), ("band", {"body": None}),
                            ("band", {"body": {**band, "results": {"band_holds": "yes", "drive_holds": True}}}),
                            ("audit", {"body": {"settings": {"num_runs": "100"}, "results": {}}}),
                            ("audit", {"body": {**audit, "settings": sizeless}}),
                            ("calibration", {"body": {"results": {"n_cal": 2000, "cal_n_scales": 200,
                                                                  "num_posterior_samples": 1000, "verdict": {}}}}),
                            ("input_use", {"body": {**check_results("input_use"), "results": {"channels": []}}}),
                            ("identifiability", {"body": {"results": {"eigenvalues": "none"}}})):
        assert check_readings.read(check, manifest) == unread, (check, manifest)
    for check in ("comparison", "timing", "card_test"):
        with pytest.raises(KeyError):
            check_readings.read(check, {"body": {}})


_REDUCED = {
    "a calibration of 40 test datasets": ("calibration", {"n_cal": 40},
                                          "Run only at a reduced size (40 test datasets; the usual is 2,000), which does "
                                          "not count."),
    "a calibration at 150 operating points": ("calibration", {"cal_n_scales": 150},
                                              "Run only at a reduced size (150 operating points; the usual is 200), "
                                              "which does not count."),
    "a calibration of 500 samples": ("calibration", {"samples": 500},
                                     "Run only at a reduced size (500 samples per dataset; the usual is 1,000), which "
                                     "does not count."),
    "an audit of 12 batches": ("audit", {"num_runs": 12},
                               "Run only at a reduced size (12 batches; the usual is 100), which does not count."),
    "an audit of 1 batch": ("audit", {"num_runs": 1},
                            "Run only at a reduced size (1 batch; the usual is 100), which does not count."),
    "an audit of 100 batches of 8": ("audit", {"run_size": 8},
                                     "Run only at a reduced size (100 batches of 8 simulations; the usual is 100 of "
                                     "32), which does not count."),
    "an audit of 1 batch of 1": ("audit", {"num_runs": 1, "run_size": 1},
                                 "Run only at a reduced size (1 batch of 1 simulation; the usual is 100 of 32), which "
                                 "does not count."),
    "a band of 8 simulations per point": ("band", {"repeats": 8},
                                          "Run only at a reduced size (8 simulations per point; the usual is 24), which "
                                          "does not count."),
    "an input use of 1,000 simulations": ("input_use", {"rows": 1_000},
                                          "Run only at a reduced size (1,000 simulations read; the usual is 200,000), "
                                          "which does not count."),
    "an input use of 9 points per sweep": ("input_use", {"n_sweep": 9},
                                           "Run only at a reduced size (9 points per sweep; the usual is 33), which does "
                                           "not count."),
}
#: For each check, the keyword that makes its record come out failed.
_FAILS = {"calibration": {"passed": False}, "audit": {"masked_fraction": 0.5}, "band": {"band_holds": False},
          "input_use": {"a1": "compressed"}}


@pytest.mark.parametrize("check, smaller, words", list(_REDUCED.values()), ids=list(_REDUCED))
def test_a_record_made_at_reduced_size_never_decides(check, smaller, words):
    """A record made at less than its check's usual size reads "Not run yet", says its size beside the usual one, and
    never counts, whichever way it came out. Its row says so while no other record counts; beside an older record that
    counts, the older decides, newest or not, and the headline counts what that one read."""
    from tests._fixtures import check_results
    for outcome in ({}, _FAILS[check]):
        reading = check_readings.read(check, {"body": check_results(check, **smaller, **outcome)})
        assert reading == Reading("Not run yet", words, counts=False), (outcome, reading)
    small = _record(check, created=_at(12, 14, 5), **smaller)
    rows = check_lineage.rows(_held(), [small], ())
    assert (_row(rows, check).state, _row(rows, check).line, _row(rows, check).record) == (
        "Not run yet", f"Not run yet. {words}", None)
    older = _record(check, created=_at(11, 9, 30), **_FAILS[check])
    rows = check_lineage.rows(_held(), [small, older], ())
    assert (_row(rows, check).state, _row(rows, check).record) == ("Failed", older)
    assert check_lineage.summary(rows, held=_held()).failed is True
    usual = {"calibration": {"n_cal": 2000, "cal_n_scales": 200, "samples": 1000},
             "audit": {"num_runs": 100, "run_size": 32},
             "band": {"repeats": 24}, "input_use": {"rows": 200_000, "n_sweep": 33}}[check]
    assert check_readings.read(check, {"body": check_results(check, **usual)}).counts, "the usual size did not count"


def test_a_band_made_with_other_thresholds_or_an_audit_at_a_fixed_number_does_not_count():
    """A frequency-band record judged with thresholds other than the usual ones, or with the phase judged, measured
    something else, and an audit made at one fixed number of driven recordings is not the reference: neither counts."""
    from tests._fixtures import check_results
    band = check_results("band")
    for name, value in (("cv_max", 0.3), ("snr_min", 2.0), ("sup_min", 0.4), ("peak_window", 0.2), ("phase_max", 0.5)):
        held = {**band, "settings": {**band["settings"], name: value}}
        assert check_readings.read("band", {"body": held}) == Reading(
            "Not run yet", "Run with other thresholds than the usual ones, which does not count.", counts=False), name
    audit = check_results("audit")
    fixed = {**audit, "settings": {**audit["settings"], "chi_k_fixed": 4}}
    assert check_readings.read("audit", {"body": fixed}) == Reading(
        "Not run yet", "Run at a fixed number of driven recordings, which does not count.", counts=False)


def test_a_record_counts_only_for_the_record_it_was_made_for(tmp_path):
    """A record counts only for the record it was made for, by id and never by name: a calibration of another posterior
    counts for nothing, even named as the posterior in use is; a calibration renamed in Records still counts, its id
    kept; an audit of another prior counts for nothing, and an audit of the prior the posterior was trained on counts
    though another prior is in use; a frequency-band record of another configuration, or of other settings of the
    frequency response, counts for nothing."""
    from core.artifacts import ArtifactStore
    from tests._fixtures import write_calibration, write_diagnostic, check_results
    store = ArtifactStore(tmp_path / "Artifacts")
    held = _held()
    write_calibration(store, None, posterior=OTHER_POSTERIOR, prior=PRIOR, name="retrain")
    write_diagnostic(store, None, diagnostic="probes", variant="mask", parents={"prior": OTHER_PRIOR},
                     name="retrain_prior", **check_results("audit"))
    rows = check_lineage.rows(held, check_lineage.read_records(store), ())
    assert (_row(rows, "calibration").state, _row(rows, "audit").state) == ("Not run yet", "Not run yet"), rows
    ours = write_calibration(store, None, posterior=POSTERIOR, prior=PRIOR, name="first_try")
    store.rename("calibration", ours, "renamed_since")
    rows = check_lineage.rows(held, check_lineage.read_records(store), ())
    assert _row(rows, "calibration").state == "Passed", "a renamed record no longer counted"
    assert _row(rows, "calibration").record.manifest["name"] == "renamed_since"
    write_diagnostic(store, None, diagnostic="probes", variant="mask", parents={"prior": PRIOR},
                     **check_results("audit", masked_fraction=0.5))
    for in_use, trained_on, state in ((PRIOR, PRIOR, "Failed"), (OTHER_PRIOR, PRIOR, "Failed"),
                                      (PRIOR, OTHER_PRIOR, "Passed")):
        found = _row(check_lineage.rows(_held(prior=in_use, posterior_prior=trained_on),
                                        check_lineage.read_records(store), ()), "audit")
        assert found.state == state, (in_use, trained_on, found)
    found = _row(check_lineage.rows(_held(posterior=None, prior=OTHER_PRIOR), check_lineage.read_records(store), ()),
                 "audit")
    assert found.state == "Passed", "with no posterior the audit of the prior in use did not count"
    audit = check_results("audit")
    drive = {**audit, "settings": {**audit["settings"],
                                   "configured": {**audit["settings"]["configured"], "chi_f0": 0.2}}}
    assert _row(check_lineage.rows(held, [_record("audit", body=drive)], ()), "audit").state == "Not run yet", (
        "an audit at another drive amplitude counted")
    assert _row(check_lineage.rows(held, [_record("audit")], ()), "audit").state == "Passed"
    other = {**CONFIG, "nd_highs": [5.0, 50.0, 31.0]}
    band = [_record("band", config=other), _record("band", body=_with_configured({**CONFIGURED, "chi_f0": 0.2}))]
    assert _row(check_lineage.rows(held, band, ()), "band").state == "Not run yet"
    assert _row(check_lineage.rows(held, [_record("band")], ()), "band").state == "Passed"


def _with_configured(configured) -> dict:
    from tests._fixtures import check_results
    band = check_results("band")
    return {**band, "settings": {**band["settings"], "configured": configured}}


def test_the_newest_record_decides_and_says_when_an_earlier_one_differed():
    """Of several records that count, the newest decides, and its line says when the newest earlier one that counts came
    out otherwise; one made at a reduced size is no earlier run. Two made in the same second are told apart by the id
    the store gives the later one, whatever order they are listed in; the others are named, newest first."""
    failed = _record("calibration", created=_at(11, 9, 30), passed=False)
    passed = _record("calibration", created=_at(12, 14, 5))
    small = _record("calibration", created=_at(13, 8, 0), n_cal=40, passed=False)
    for listed in ([failed, passed, small], [small, passed, failed]):
        row = _row(check_lineage.rows(_held(), listed, ()), "calibration")
        assert (row.state, row.line, row.record, row.when) == (
            "Passed", f"Passed, {_said(_at(12, 14, 5))}. An earlier run failed ({_said(_at(11, 9, 30))}).", passed,
            _at(12, 14, 5)), listed
        assert row.earlier == (small.manifest["id"], failed.manifest["id"])
    alike = _record("calibration", created=_at(12, 15, 0), passed=False)
    row = _row(check_lineage.rows(_held(), [failed, passed, alike], ()), "calibration")
    assert row.line == f"Failed, {_said(_at(12, 15, 0))}. An earlier run passed ({_said(_at(12, 14, 5))}).", row
    same = _at(14, 10, 0)
    first = _record("calibration", created=same, ident="20261014T170000", passed=False)
    second = _record("calibration", created=same, ident="20261014T170000-2")
    tenth = _record("calibration", created=same, ident="20261014T170000-10", passed=False)
    for listed in ([first, second], [second, first]):
        assert _row(check_lineage.rows(_held(), listed, ()), "calibration").record is second, listed
    for listed in ([first, second, tenth], [tenth, second, first]):
        assert _row(check_lineage.rows(_held(), listed, ()), "calibration").record is tenth, listed
    look = _record("audit", created=_at(12, 14, 5), masked_fraction=0.46)
    plain = _record("audit", created=_at(11, 9, 30))
    assert _row(check_lineage.rows(_held(), [plain, look], ()), "audit").line == (
        f"Look closer, {_said(_at(12, 14, 5))}. An earlier run passed ({_said(_at(11, 9, 30))}).")
    assert _row(check_lineage.rows(_held(), [look, _record("audit", created=_at(13, 9, 0))], ()), "audit").line == (
        f"Passed, {_said(_at(13, 9, 0))}. An earlier run needed a closer look ({_said(_at(12, 14, 5))}).")


def test_the_band_is_judged_by_its_worst_cell():
    """The frequency band is judged on each cell by that cell's newest record that counts, and reads the worst of the
    cells, dated by the record that decides it: a newer record of the same cell replaces an older one, and a cell's
    record never stands for another cell's."""
    a_passed = _record("band", created=_at(12, 9, 0), cell="Cells/nadrowski/a.txt")
    b_failed = _record("band", created=_at(11, 9, 0), cell="Cells/nadrowski/b.txt", band_holds=False)
    row = _row(check_lineage.rows(_held(), [a_passed, b_failed], ()), "band")
    assert (row.state, row.record, row.when) == ("Failed", b_failed, _at(11, 9, 0))
    b_passed = _record("band", created=_at(13, 9, 0), cell="Cells/nadrowski/b.txt")
    row = _row(check_lineage.rows(_held(), [a_passed, b_failed, b_passed], ()), "band")
    assert (row.state, row.record) == ("Passed", b_passed), row
    a_look = _record("band", created=_at(14, 9, 0), cell="Cells/nadrowski/a.txt", drive_holds=None)
    row = _row(check_lineage.rows(_held(), [a_passed, b_failed, b_passed, a_look], ()), "band")
    assert (row.state, row.record) == ("Look closer", a_look), row
    small = _record("band", created=_at(15, 9, 0), cell="Cells/nadrowski/c.txt", repeats=8, band_holds=False)
    row = _row(check_lineage.rows(_held(), [a_passed, small], ()), "band")
    assert (row.state, row.record) == ("Passed", a_passed), "a cell with no record that counts was judged"


# ── the headline ──
_OPEN_STATES = ("Passed", "Look closer", "Failed", "Not run yet")


def _rows_of(states) -> tuple:
    """The twelve rows with the six deciding ones in ``states`` and the others not run."""
    states = list(states) + ["Not run yet"] * 6
    return tuple(CheckRow(one.key, state, f"{state}.", None, None, (), "")
                 for one, state in zip(check_catalogue.CHECKS, states))


def _expected(states, *, posterior: bool, chi: bool) -> dict:
    """What the headline, its sentence and the summary say of six deciding states, worked out here from the states
    alone."""
    names = [one.name for one in check_catalogue.CHECKS[:6]]
    applies = [state != "Does not apply" for state in states]
    passed = states.count("Passed")
    trained_without = posterior and not chi
    of = sum(applies) if trained_without else 6
    head = (f"{passed} of {of} deciding checks passed; {6 - of} do not apply to this posterior" if trained_without
            else f"{passed} of 6 deciding checks passed")
    failed = [name for name, state in zip(names, states) if state == "Failed"]
    closer = [name for name, state in zip(names, states) if state == "Look closer"]
    still = sum(1 for state, apply in zip(states, applies) if apply and state != "Passed")
    if failed:
        sentence = "Failed: " + ", ".join(failed) + "."
    elif closer:
        sentence = "Look closer: " + ", ".join(closer) + "."
    elif not posterior:
        sentence = ("Three of the six do not apply to a posterior trained without the frequency response." if not chi
                    else "The other four are made after simulating and after the fit.")
    elif still:
        sentence = (f"Inference can be run now; {still} deciding checks are still open." if still > 1
                    else "Inference can be run now; 1 deciding check is still open.")
    else:
        sentence = ("This posterior has passed every deciding check that applies to it." if trained_without
                    else "This posterior has passed every deciding check.")
    waiting = sum(state in ("Waiting for a posterior", "Waiting for the simulations") for state in states)
    second = (f"{len(failed)} failed" if failed else f"{len(closer)} to look closer at" if closer
              else ("1 waits for step 3" if waiting == 1 else f"{waiting} wait for step 3") if waiting
              else f"{states.count('Not run yet')} not run yet" if "Not run yet" in states else "")
    return {"headline": head, "sentence": sentence, "passed": passed, "of": of, "failed": bool(failed),
            "look_closer": bool(closer), "open": bool(still), "can_check": True, "second_line": second}


def _cases():
    """Every combination of the deciding states a posterior can hold, with the frequency response and without, and the
    ones with no posterior."""
    for states in itertools.product(_OPEN_STATES, repeat=6):
        yield list(states), True, True
    for states in itertools.product(_OPEN_STATES, repeat=3):
        yield ["Does not apply"] * 3 + list(states), True, False
    for states in itertools.product(_OPEN_STATES, repeat=2):
        yield list(states) + ["Waiting for the simulations"] + ["Waiting for a posterior"] * 3, False, True
    yield ["Does not apply"] * 3 + ["Waiting for a posterior"] * 3, False, False


def test_the_headline_over_every_combination_of_states():
    """For every combination of the six deciding states, with a posterior trained with the frequency response, one
    trained without it, and with none, the headline, its sentence and every field of the summary are what the rules
    give from the states alone: n of 6, or n of 3 with the three that do not apply said for a posterior trained without
    it; the first sentence that applies; and the first count of the second line."""
    seen = 0
    for states, posterior, chi in _cases():
        held = _held(posterior=POSTERIOR if posterior else None, chi=chi)
        rows = _rows_of(states)
        want = _expected(states, posterior=posterior, chi=chi)
        said = check_lineage.summary(rows, held=held)
        assert check_lineage.headline(rows, held=held) == (want["headline"], want["sentence"]), (states, posterior, chi)
        assert {name: getattr(said, name) for name in want} == want, (states, posterior, chi)
        seen += 1
    assert seen == 4 ** 6 + 4 ** 3 + 4 ** 2 + 1
    nothing = check_lineage.rows(_held(prior=None), (), ())
    said = check_lineage.summary(nothing, held=_held(prior=None))
    assert {row.state for row in nothing} == {"Waiting for a prior"}
    assert (said.headline, said.sentence, said.can_check, said.second_line) == (
        "0 of 6 deciding checks passed", "Nothing can be checked yet. Build or load a prior in step 2.", False, "")


def test_the_headlines_sentence_is_the_first_that_applies():
    """The seven sentences under the headline, each where it is the first that applies: with no prior, any failed
    (named in the list's order), any to look closer at, three that do not apply with no posterior, no posterior, a
    check still open (one, and several), and every one that applies passed (of six, and of three)."""
    def sentence(states, **over):
        held = _held(**over)
        return check_lineage.headline(_rows_of(states), held=held)[1]

    six = ["Passed"] * 6
    assert sentence(six, prior=None) == "Nothing can be checked yet. Build or load a prior in step 2."
    assert sentence(["Failed", "Passed", "Passed", "Failed", "Look closer", "Passed"]) == (
        "Failed: Frequency band, Calibration.")
    assert sentence(["Passed", "Look closer", "Passed", "Passed", "Passed", "Not run yet"]) == (
        "Look closer: Unusable driven recordings: audit.")
    assert sentence(["Does not apply"] * 3 + ["Waiting for a posterior"] * 3, posterior=None, chi=False) == (
        "Three of the six do not apply to a posterior trained without the frequency response.")
    assert sentence(["Passed", "Not run yet", "Waiting for the simulations"] + ["Waiting for a posterior"] * 3,
                    posterior=None) == "The other four are made after simulating and after the fit."
    assert sentence(["Passed"] * 5 + ["Not run yet"]) == "Inference can be run now; 1 deciding check is still open."
    assert sentence(["Passed"] * 4 + ["Not compared", "Not run yet"]) == (
        "Inference can be run now; 2 deciding checks are still open.")
    assert sentence(six) == "This posterior has passed every deciding check."
    assert sentence(["Does not apply"] * 3 + ["Passed"] * 3, chi=False) == (
        "This posterior has passed every deciding check that applies to it.")


def test_the_summarys_second_line_counts_the_first_that_applies():
    """The second line says the first of what failed, what to look closer at, what waits for step 3 and what is not run
    yet, counted; a check waiting for the audit is not run yet, one not compared is none of them; and none, nothing."""
    def second(states, **over):
        return check_lineage.summary(_rows_of(states), held=_held(**over)).second_line

    assert second(["Failed", "Look closer", "Failed", "Not run yet", "Passed", "Passed"]) == "2 failed"
    assert second(["Look closer", "Passed", "Passed", "Not run yet", "Passed", "Passed"]) == "1 to look closer at"
    assert second(["Passed", "Not run yet", "Waiting for the simulations"] + ["Waiting for a posterior"] * 3,
                  posterior=None) == "4 wait for step 3"
    assert second(["Does not apply"] * 3 + ["Waiting for a posterior"] * 3, posterior=None, chi=False) == (
        "3 wait for step 3")
    assert second(["Passed", "Passed", "Waiting for the audit", "Not run yet", "Passed", "Passed"]) == "2 not run yet"
    assert second(["Passed", "Passed", "Not compared", "Passed", "Passed", "Passed"]) == ""
    assert second(["Passed"] * 6) == ""


# ── the comparison after simulating ──
_UNUSABLE, _TOTAL = 57_703_562, 143_831_040
_SHARE = _UNUSABLE / _TOTAL


@pytest.mark.parametrize("points, state", [(2.2, "Passed"), (2.99, "Passed"), (3.01, "Look closer"),
                                           (3.8, "Look closer"), (4.99, "Look closer"), (5.01, "Failed"),
                                           (5.2, "Failed"), (-3.8, "Look closer"), (-5.2, "Failed")])
def test_the_comparison_bands_from_the_counts(points, state):
    """The simulation set's share is taken from its two counts, never from the share rounded to one decimal: within 3
    points of the audit is fine, up to 5 a reason to look closer, beyond 5 the advice is to stop, on either side."""
    compared = check_readings.compare(unusable=_UNUSABLE, total=_TOTAL, scope=WHOLE, audit_fraction=_SHARE - points / 100,
                                      batches=10_000)
    assert compared.state == state
    assert compared.difference == pytest.approx(abs(points), abs=1e-9)
    assert check_readings.compare(unusable=_UNUSABLE, total=_TOTAL, scope=PROCESS, audit_fraction=_SHARE - points / 100,
                                  batches=10_000).state == state, "the scope of a set that is not kept is the whole set"


def test_the_comparisons_sentences_and_messages_are_the_windows():
    """Each band's sentence for the run's health row and the check, and its message: none within 3 points; to look
    closer at, the difference; beyond 5, the advice to stop, in the words of a fit saved after every epoch, or of a
    training that keeps no simulation set. A difference just above a limit is never said as the limit."""
    def at(points, **more):
        return check_readings.compare(unusable=_UNUSABLE, total=_TOTAL, scope=WHOLE,
                                      audit_fraction=_SHARE - points / 100, batches=10_000, **more)

    assert (at(2.2).sentence, at(2.2).message) == (
        "Unusable driven recordings 40.1 %. The audit read 37.9 %; within 3 points is fine.", "")
    assert (at(3.8).sentence, at(3.8).message) == (
        "Unusable driven recordings 40.1 %. The audit read 36.3 %; they differ by 3.8 points; up to 5 is a reason to "
        "look closer.",
        "The simulation set and the audit differ by 3.8 points. Look closer before leaving the fit to run: step 4, "
        "Checks, shows the per-batch figures of both.")
    assert (at(5.2).sentence, at(5.2).message) == (
        "Unusable driven recordings 40.1 %. The audit read 34.9 %; they differ by 5.2 points; beyond 5 the advice is to "
        "stop.",
        "The advice is to stop the fit and find out why. The fit is saved after every epoch, so a stop keeps it and "
        "loses at most the epoch in progress; the training can be resumed later.")
    assert at(5.2, kept_set=False).message == (
        "The advice is to stop the fit and find out why. This training keeps no simulation set, so a stop loses its "
        "simulations and its fit.")
    assert "differ by 3.01 points" in at(3.01).sentence and "differ by 5.01 points" in at(5.01).sentence
    assert "differ by 3.04 points" in at(3.04).sentence


def test_the_comparison_at_the_edges_of_its_counts():
    """Nothing counted, a total of nought or one that reads "none counted", says so and compares nothing; a count of
    part of the set says which batches it covers and which were saved without their counts; no audit says there is
    none to compare with; an audit made at fewer than its usual batches counts for nothing, so the check waits for the
    audit; and nothing handed in raises, nor divides by nought."""
    none = check_readings.compare(unusable=0, total=0, scope=WHOLE, audit_fraction=0.379)
    assert (none.state, none.sentence, none.message, none.difference) == (
        "Not compared", "Not compared: no driven recordings were counted.",
        "Not compared: no driven recordings were counted.", None)
    from tests._runs import masked_line, masked_none, scope_part
    text = f"10:01:02 info [chi] something else\n10:01:03 info {masked_none(WHOLE)}\n"
    assert check_readings.first_masked(text) == (0, 0, 0, WHOLE)
    unusable, total, batches, scope = check_readings.first_masked(
        f"10:01:03 info {masked_line(9_600, 24_000, 240, scope_part(760))}\n10:05:00 info "
        f"{masked_line(1, 10, 1, PROCESS)}\n")
    part = check_readings.compare(unusable=unusable, total=total, scope=scope, audit_fraction=0.379, batches=batches)
    assert (part.state, part.sentence) == (
        "Not compared", "Not compared: the count covers 240 batches, not the whole simulation set; the first 760 were "
        "saved without their counts.")
    for counted in ((0, 0, 0, WHOLE), (unusable, total, batches, scope)):
        alone = check_readings.compare(unusable=counted[0], total=counted[1], scope=counted[3], audit_fraction=None,
                                       batches=counted[2])
        assert alone.state == "Not compared", f"no audit was said of counts that compare nothing: {alone}"
    waiting = check_readings.compare(unusable=_UNUSABLE, total=_TOTAL, scope=WHOLE, audit_fraction=None,
                                     prior_words="the prior retrain_prior")
    assert (waiting.state, waiting.sentence, waiting.difference) == (
        "Waiting for the audit", "There is no audit of the prior retrain_prior to compare with. The simulation set holds "
        "40.1 % unusable driven recordings. Run the audit in step 4, Checks, when this run has ended.", None)
    small = _record("audit", num_runs=12)
    rows = check_lineage.rows(_held(set_counts=(_UNUSABLE, _TOTAL, 10_000, WHOLE)), [small], ())
    assert (_row(rows, "audit").state, _row(rows, "comparison").state) == ("Not run yet", "Waiting for the audit")
    for odd in (dict(unusable=None, total=None, scope=None, audit_fraction=None),
                dict(unusable=1, total=-5, scope=WHOLE, audit_fraction=0.3),
                dict(unusable=True, total=True, scope=WHOLE, audit_fraction=True),
                dict(unusable=3, total=10, scope=7, audit_fraction="0.3"),
                dict(unusable=3, total=10, scope="", audit_fraction=0.3, batches=None)):
        assert check_readings.compare(**odd).state in check_readings.STATES, odd
    assert check_readings.first_masked(None) is None and check_readings.first_masked("no total here") is None


def test_an_accepted_difference_reads_passed_with_its_words():
    """A difference to look closer at that was accepted reads "Passed", its sentence saying what was accepted and when,
    with no message; an acceptance with no moment is none, and a difference beyond 5 is never turned by one."""
    when = datetime(2026, 10, 3, 9, 14).astimezone()

    def at(points, accepted):
        return check_readings.compare(unusable=_UNUSABLE, total=_TOTAL, scope=WHOLE, audit_fraction=_SHARE - points / 100,
                                      accepted=accepted)

    for moment in (when, when.isoformat()):
        taken = at(3.8, {"at": moment})
        assert (taken.state, taken.sentence, taken.message) == (
            "Passed", "Unusable driven recordings 40.1 %. The audit read 36.3 %; they differ by 3.8 points; up to 5 is a "
            "reason to look closer. A difference of 3.8 points was accepted on 3 Oct 2026, 09:14.", ""), moment
    assert at(3.8, {}).state == at(3.8, {"at": "not a moment"}).state == "Look closer"
    assert at(5.2, {"at": when}).state == "Failed"


def test_check_three_reads_only_the_counts_that_are_there():
    """The comparison after simulating, before a training has written it: with no posterior in use it waits for the
    simulations unless the newest training of the prior in use says how it compared; with a posterior it compares the
    counts the reading found against the audit that counts, says it could not read them when none were found, says a
    count of part of the set is not compared, and does not apply without the frequency response. No "Passed" or
    "Failed" comes from counts that are not there."""
    audit = _record("audit")
    rows = check_lineage.rows(_held(posterior=None), [audit], ({"word": "Training", "health": {"failed": 0}},))
    assert _row(rows, "comparison").state == "Waiting for the simulations"
    written = {"word": "Training", "settings": {"prior": PRIOR}, "health": {"comparison": {"state": "Look closer"}}}
    other = {"word": "Training", "settings": {"prior": OTHER_PRIOR}, "health": {"comparison": {"state": "Passed"}}}
    assert _row(check_lineage.rows(_held(posterior=None), [audit], (written, other)), "comparison").state == (
        "Look closer"), "the newest training of another prior was read"
    found = _row(check_lineage.rows(_held(set_counts=(8_412, 22_144, 100, WHOLE)), [audit], ()), "comparison")
    assert (found.state, found.when, found.line) == ("Passed", _at(3, 9, 0), f"Passed, {_said(_at(3, 9, 0))}.")
    found = _row(check_lineage.rows(_held(set_counts=(9_700, 22_144, 100, WHOLE)), [audit], ()), "comparison")
    assert found.state == "Failed"
    for counts, line in ((None, "Not compared: the counts of unusable driven recordings could not be read."),
                         ((0, 0, 0, WHOLE), "Not compared: no driven recordings were counted."),
                         ((900, 2_200, 10, check_readings_part()), "Not compared: the count covers 10 batches, not the "
                          "whole simulation set; the first 90 were saved without their counts.")):
        found = _row(check_lineage.rows(_held(set_counts=counts), [audit], ()), "comparison")
        assert (found.state, found.line, found.when) == ("Not compared", line, None), counts
    assert _row(check_lineage.rows(_held(set_counts=(8_412, 22_144, 100, WHOLE)), [], ()), "comparison").state == (
        "Waiting for the audit")
    assert _row(check_lineage.rows(_held(chi=False), [audit], ()), "comparison").state == "Does not apply"
    for counts in (None, (0, 0, 0, WHOLE), (900, 2_200, 10, check_readings_part())):
        for records in ([audit], [], [_record("audit", masked_fraction=0.5)]):
            state = _row(check_lineage.rows(_held(set_counts=counts), records, ()), "comparison").state
            assert state not in ("Passed", "Failed"), (counts, records)


def check_readings_part() -> str:
    from tests._runs import scope_part
    return scope_part(90)


# ── what is read, and how ──
def test_every_reading_rule_is_a_named_constant_with_its_reason():
    """No number a deciding check is judged by stands bare in a rule: inside the module's functions no number is
    written but 0, 1 and 100; and every constant of the module, named in capitals, has a line beginning "#:" above it,
    or above the constants written with it."""
    path = REPO / "core" / "gui" / "check_readings.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    numbers = [(node.lineno, node.value) for function in ast.walk(tree)
               if isinstance(function, (ast.FunctionDef, ast.Lambda))
               for node in ast.walk(function)
               if isinstance(node, ast.Constant) and isinstance(node.value, (int, float))
               and not isinstance(node.value, bool) and node.value not in (0, 1, 100)]
    assert numbers == [], f"a bare number in a reading rule: {numbers}"
    lines = source.splitlines()
    constant = {node.lineno for node in tree.body if isinstance(node, (ast.Assign, ast.AnnAssign))
                for target in (node.targets if isinstance(node, ast.Assign) else [node.target])
                if isinstance(target, ast.Name) and target.id.lstrip("_").isupper()}
    spans = {line for node in tree.body if isinstance(node, (ast.Assign, ast.AnnAssign)) and node.lineno in constant
             for line in range(node.lineno, node.end_lineno + 1)}
    bare = []
    for number in sorted(constant):
        above = number - 1
        while above in spans:
            above -= 1
        if not lines[above - 1].startswith("#:"):
            bare.append(lines[number - 1])
    assert bare == [], f"a constant with no reason above it: {bare}"
    assert len(constant) > 20, "the scan found the constants"


def test_the_least_sizes_are_the_engines_usual_sizes():
    """Each least size and threshold a record is judged by is the engine's own usual value: the calibration's datasets
    and operating points, which the stage takes from the settings when its argument is left at None, and its samples,
    the frequency band's simulations per point and thresholds, the audit's simulations in each batch, and the
    input-use check's simulations and points, each the default of its stage's argument."""
    from core import config, orchestrator
    from core.diagnostics.ablation import channel_ablation
    from core.diagnostics.probes import probe_band, probe_mask
    from tests._fixtures import code_only
    calibration = inspect.signature(orchestrator.validate_calibration.__wrapped__
                                    if hasattr(orchestrator.validate_calibration, "__wrapped__")
                                    else orchestrator.validate_calibration).parameters
    assert (calibration["n_cal"].default, calibration["cal_n_scales"].default) == (None, None)
    code = code_only(orchestrator.validate_calibration)
    assert "SBC_N_CAL if n_cal is None else n_cal" in code and "CAL_N_SCALES if cal_n_scales is None" in code
    assert (check_readings.CALIBRATION_LEAST_DATASETS, check_readings.CALIBRATION_LEAST_POINTS) == (
        config.SBC_N_CAL, config.CAL_N_SCALES) == (2000, 200)
    assert check_readings.CALIBRATION_LEAST_SAMPLES == calibration["num_posterior_samples"].default == 1000
    band = inspect.signature(probe_band).parameters
    assert (check_readings.BAND_LEAST_REPEATS, check_readings.BAND_CV_MAX, check_readings.BAND_SNR_MIN,
            check_readings.BAND_SUP_MIN, check_readings.BAND_PEAK_WINDOW) == tuple(
        band[name].default for name in ("repeats", "cv_max", "snr_min", "sup_min", "peak_window")) == (
        24, 0.2, 3.0, 0.5, 0.1)
    assert band["phase_max"].default is None
    assert check_readings.AUDIT_LEAST_PER_BATCH == inspect.signature(probe_mask).parameters["run_size"].default == 32
    ablation = inspect.signature(channel_ablation).parameters
    assert (check_readings.INPUT_USE_LEAST_ROWS, check_readings.INPUT_USE_LEAST_SWEEP) == (
        ablation["rows"].default, ablation["n_sweep"].default) == (200_000, 33)


_FRESH = """
import importlib
import sys
importlib.import_module(sys.argv[1])
from datetime import datetime, timezone
from core.gui import check_catalogue, check_lineage, check_readings
whole = check_readings.WHOLE_SET_SCOPES[0]
held = check_lineage.Held(prior="p", posterior="q", posterior_prior="p", simulation="s", chi=True, config=None,
                          configured=None, rotated=False, narrowed=False, nadrowski=True, forcing=False,
                          pair_refused="no", set_counts=(1, 4, 1, whole), set_done=True, prior_words="the prior p",
                          posterior_words="the posterior q")
manifest = {"kind": "calibration", "id": "20261003T120000", "created": datetime(2026, 10, 3, tzinfo=timezone.utc).isoformat(),
            "parents": {"posterior": "q", "prior": "p"},
            "body": {"results": {"n_cal": 40, "cal_n_scales": 200, "num_posterior_samples": 1000,
                                 "verdict": {"passed": True}}}}
records = (check_lineage.RecordRead("x", manifest), check_lineage.RecordRead("y", {}))
history = ({"word": "Training", "check": {"key": "timing"}, "outcome": "finished", "started": "2026-10-03T12:00:00+00:00"},)
for one in (held, check_lineage.Held(**{**held.__dict__, "prior": None}),
            check_lineage.Held(**{**held.__dict__, "posterior": None, "chi": False})):
    rows = check_lineage.rows(one, records, history)
    check_lineage.summary(rows, held=one)
    check_lineage.pre_run_fact(rows)
    check_lineage.pre_run_question(rows)
for key in ("band", "audit", "calibration", "input_use", "identifiability", "pin_down"):
    check_readings.read(key, {"body": {}})
for scope in (whole, "part"):
    for fraction in (None, 0.2, 0.3, 0.9):
        check_readings.compare(unusable=1, total=4, scope=scope, audit_fraction=fraction, accepted={"at": "x"})
check_readings.first_masked("10:00:00 info [chi] masked probes: none counted, " + whole)
check_lineage.read_records(None)
check_lineage.read_text("no folder here")
check_catalogue.entry("band")
roots = ("PySide6", "torch", "core.config", "core.SBI", "core.artifacts", "core.gui.record_words", "core.gui.design")
bad = sorted(name for name in sys.modules if any(name == root or name.startswith(root + ".") for root in roots))
sys.exit(repr(bad) if bad else 0)
"""


@pytest.mark.parametrize("module", ["core.gui.check_catalogue", "core.gui.check_readings", "core.gui.check_lineage"])
def test_the_checks_rules_import_neither_the_toolkit_nor_torch(module):
    """The three modules of the checks' rules load neither the toolkit, nor torch, nor the engine's settings, nor the
    store, the engine's training code or the window's record words (which draw through the toolkit): each module's
    import statements name only the standard library and the window's own modules of rules and words; and a fresh
    interpreter that imports one of them first, alone, then the others, and calls every rule down its branches, holds
    none of them."""
    allowed = {"collections.abc", "dataclasses", "datetime", "json", "pathlib", "re", ".", ".check_catalogue",
               ".check_readings", ".run", ".step_facts", ".texts"}
    path = REPO / (module.replace(".", "/") + ".py")
    named = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            named |= {alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            named.add("." * node.level + (node.module or ""))
    assert named <= allowed, named - allowed
    r = subprocess.run([sys.executable, "-c", _FRESH, module], cwd=str(REPO), capture_output=True, text=True,
                       timeout=120)
    assert r.returncode == 0, r.stdout + r.stderr


def test_a_manifest_that_cannot_be_read_is_passed_over_and_said(tmp_path):
    """Every check record of the store is read, each manifest once by its folder: one torn, and one whose folder holds
    no manifest, are kept by their folders with nothing read, so that they can be said, and never count; the others are
    read whole; the rows raise nothing. A record's text is read by its folder, and one that is not there is None."""
    from core.artifacts import ArtifactStore
    from tests._fixtures import check_results, write_calibration, write_diagnostic
    store = ArtifactStore(tmp_path / "Artifacts")
    good = write_calibration(store, None, posterior=POSTERIOR, prior=PRIOR)
    write_diagnostic(store, None, diagnostic="probes", variant="mask", parents={"prior": PRIOR},
                     **check_results("audit"))
    torn = write_calibration(store, None, posterior=POSTERIOR, prior=PRIOR,
                             results=check_results("calibration", passed=False)["results"])
    torn_folder = store.path("calibration", torn)
    (torn_folder / "manifest.json").write_bytes(b'{"kind": "calibration", "id": ')
    (store.kind_dir("diagnostic") / "an_empty_folder").mkdir()
    records = check_lineage.read_records(store)
    assert len(records) == 4
    read = [record for record in records if record.manifest]
    assert sorted(record.manifest["kind"] for record in read) == ["calibration", "diagnostic"]
    said = sorted(Path(record.folder).name for record in records if not record.manifest)
    assert said == sorted(["an_empty_folder", torn_folder.name]), said
    rows = check_lineage.rows(_held(), records, ())
    assert _row(rows, "calibration").state == "Passed" and _row(rows, "calibration").record.manifest["id"] == good
    assert check_lineage.read_text(str(tmp_path / "nowhere")) is None
    assert json.loads(check_lineage.read_text(str(store.path("calibration", good)), "manifest.json"))["id"] == good
    assert check_lineage.read_records(store, kinds=("not a kind",)) == ()


def test_each_check_that_cannot_apply_or_run_says_why():
    """A check that cannot apply reads "Does not apply" with its reason, and one that waits says what for: the first
    three without the frequency response; what the observations pin down for another model and, without the frequency
    response, without a forcing section; the drive study without one; identifiability across the prior with the
    frequency response; the checks of a posterior with none; the input use of a posterior trained without its
    simulations and the identifiability of one trained without the rotation, which read "Not run yet"; and with no
    prior every row waits for one."""
    rows = check_lineage.rows(_held(chi=False, forcing=True), (), ())
    assert [(row.state, row.reason) for row in rows[:3]] == [("Does not apply",
                                                              "Applies only with the frequency response.")] * 3
    rows = check_lineage.rows(_held(nadrowski=False), (), ())
    assert (_row(rows, "pin_down").state, _row(rows, "pin_down").reason) == (
        "Does not apply", "Applies only to the Nadrowski model.")
    rows = check_lineage.rows(_held(chi=False, forcing=False), (), ())
    assert [(_row(rows, key).state, _row(rows, key).reason) for key in ("pin_down", "drive_study", "across_prior")] == [
        ("Does not apply", "Needs a bounds file with a forcing section.")] * 3
    assert _row(check_lineage.rows(_held(), (), ()), "across_prior").reason == "Does not apply with the frequency response."
    rows = check_lineage.rows(_held(posterior=None, chi=False), (), ())
    assert [(_row(rows, key).state, _row(rows, key).reason) for key in ("calibration", "input_use", "identifiability",
                                                                         "fixed_number", "across_prior")] == [
        ("Waiting for a posterior", "Needs a posterior: train one in step 3, or load a saved one.")] * 5
    rows = check_lineage.rows(_held(simulation=None, rotated=False), (), ())
    assert (_row(rows, "input_use").state, _row(rows, "input_use").line) == (
        "Not run yet", "Not run yet. This posterior was trained without keeping its simulations, and this check reads "
                       "them.")
    assert (_row(rows, "identifiability").state, _row(rows, "identifiability").line) == (
        "Not run yet", "Not run yet. This posterior was trained without the rotation, so there are no directions to "
                       "read.")
    assert _row(check_lineage.rows(_held(pair_refused="the engine's sentence"), (), ()), "calibration").reason == (
        "The prior in use is not the one this posterior was trained on. Load it in step 2.")
    assert {(row.state, row.reason) for row in check_lineage.rows(_held(prior=None), (), ())} == {
        ("Waiting for a prior", "Needs a prior: build or load one in step 2.")}


def test_the_informing_checks_and_the_tools_read_done_from_their_own_records():
    """What the observations pin down and the drive study count by the configuration in use, the calibration at fixed
    numbers and identifiability across the prior by the posterior; the timing and the card test read "Done" from the
    newest run of each in the history that finished, dated by its start; each else reads "Not run yet", and none ever
    reads "Passed" or "Failed"."""
    records = [_record("pin_down", created=_at(5)), _record("drive_study", created=_at(6)),
               _record("fixed_number", created=_at(7)), _record("across_prior", created=_at(8))]
    histories = ({"word": "Check", "check": {"key": "timing"}, "outcome": "finished", "started": _at(9).isoformat()},
                 {"word": "Check", "check": {"key": "card_test"}, "outcome": "failed", "started": _at(10).isoformat()},
                 {"word": "Check", "check": None, "outcome": "finished", "started": _at(11).isoformat()})
    rows = check_lineage.rows(_held(chi=False), records, histories)
    assert [(_row(rows, key).state, _row(rows, key).when) for key in
            ("pin_down", "timing", "fixed_number", "card_test", "drive_study", "across_prior")] == [
        ("Done", _at(5)), ("Done", _at(9)), ("Done", _at(7)), ("Not run yet", None), ("Done", _at(6)), ("Done", _at(8))]
    elsewhere = check_lineage.rows(_held(chi=False, config={**CONFIG, "model": "HOPF"}, posterior=OTHER_POSTERIOR),
                                   records, ())
    assert [_row(elsewhere, key).state for key in ("pin_down", "fixed_number", "drive_study", "across_prior")] == [
        "Not run yet"] * 4


# ── before a training ──
def test_the_pre_run_fact_for_each_state():
    """"Before you start" says "Fine." when the frequency band and the audit both passed, with their dates;
    "Information." when neither applies; else "Attention.", naming each that has not passed in the list's order, and
    after one failed that starting will ask first. It never blocks. The question a start asks is there only after one
    failed: its title and its other answer from the button's words, its text naming each that failed."""
    def rows(band, audit, *, band_at=_at(3, 14, 2), audit_at=_at(3, 14, 40)):
        made = []
        for key, state, when in (("band", band, band_at), ("audit", audit, audit_at)):
            made.append(CheckRow(key, state, "", when if state in ("Passed", "Look closer", "Failed") else None, None,
                                 (), ""))
        return tuple(made) + tuple(CheckRow(one.key, "Not run yet", "", None, None, (), "")
                                   for one in check_catalogue.CHECKS[2:])

    from core.gui.step_facts import Fact
    band_at, audit_at = _said(_at(3, 14, 2)), _said(_at(3, 14, 40))
    assert check_lineage.pre_run_fact(rows("Passed", "Passed")) == Fact(
        "fine", f"Frequency band: Passed, {band_at}. Unusable driven recordings: audit: Passed, {audit_at}.")
    assert check_lineage.pre_run_fact(rows("Does not apply", "Does not apply")) == Fact(
        "info", "The two pre-run checks do not apply: they need the frequency response.")
    assert check_lineage.pre_run_fact(rows("Passed", "Not run yet")) == Fact(
        "warn", "Unusable driven recordings: audit: Not run yet.")
    assert check_lineage.pre_run_fact(rows("Not run yet", "Not run yet")) == Fact(
        "warn", "Frequency band: Not run yet. Unusable driven recordings: audit: Not run yet.")
    assert check_lineage.pre_run_fact(rows("Failed", "Not run yet")) == Fact(
        "warn", f"Frequency band: Failed, {band_at}. Unusable driven recordings: audit: Not run yet. Starting will ask "
                "first.")
    assert check_lineage.pre_run_fact(rows("Look closer", "Passed")) == Fact(
        "warn", f"Frequency band: Look closer, {band_at}.")
    for band, audit in itertools.product(_OPEN_STATES + ("Does not apply", "Waiting for a prior"), repeat=2):
        assert check_lineage.pre_run_fact(rows(band, audit)).kind in ("fine", "info", "warn"), (band, audit)
        failed = "Failed" in (band, audit)
        assert (check_lineage.pre_run_question(rows(band, audit)) is not None) == failed, (band, audit)
    assert check_lineage.pre_run_question(rows("Failed", "Passed")) == (
        "Start training although a check failed?",
        f"Frequency band: Failed, {band_at}. Step 4, Checks, says why.", "Start training anyway")
    assert check_lineage.pre_run_question(rows("Failed", "Failed"), start="Continue from batch 236") == (
        "Continue from batch 236 although a check failed?",
        f"Frequency band: Failed, {band_at}. Unusable driven recordings: audit: Failed, {audit_at}. Step 4, Checks, "
        "says why.", "Continue from batch 236 anyway")
