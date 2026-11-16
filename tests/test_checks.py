"""The checks: what each of the twelve is, how a record of a deciding check is read, which records count for what is in
use, the headline over the six deciding checks and the summary the chain and Home read, the comparison of the unusable
driven recordings a training simulated with its prior's audit, and the line "Before you start" says of the two checks
made before a training; then the Checks list, a check's page and the result of every check, and what a check's page
never shows.

The rules are pure functions over plain values, handed records as the store writes them, either read back from the
test's own store or written out here as the engine writes them; the list and its pages are read in a whole window shown
off the screen, over records written in the test's own store.
"""
import ast
import contextlib
import inspect
import itertools
import json
import subprocess
import sys
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from core.gui import check_catalogue, check_lineage, check_readings
from core.gui.check_lineage import CheckRow, Held, RecordRead
from core.gui.check_readings import Reading
from core.gui.run.controller import controller
from core.gui.texts import date_text
from tests._fixtures import qt_app
from tests._look_helpers import _settle, text_size
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
    "an audit of 100 batches of 31": ("audit", {"run_size": 31},
                                      "Run only at a reduced size (100 batches of 31 simulations; the usual is 100 of "
                                      "32), which does not count."),
    "an audit of 99 batches": ("audit", {"num_runs": 99},
                               "Run only at a reduced size (99 batches; the usual is 100), which does not count."),
    "a band of 23 simulations per point": ("band", {"repeats": 23},
                                           "Run only at a reduced size (23 simulations per point; the usual is 24), "
                                           "which does not count."),
    "a calibration of 1,999 test datasets": ("calibration", {"n_cal": 1_999},
                                             "Run only at a reduced size (1,999 test datasets; the usual is 2,000), "
                                             "which does not count."),
    "a calibration at 199 operating points": ("calibration", {"cal_n_scales": 199},
                                              "Run only at a reduced size (199 operating points; the usual is 200), "
                                              "which does not count."),
    "a calibration of 999 samples": ("calibration", {"samples": 999},
                                     "Run only at a reduced size (999 samples per dataset; the usual is 1,000), which "
                                     "does not count."),
    "an input use of 199,999 simulations": ("input_use", {"rows": 199_999},
                                            "Run only at a reduced size (199,999 simulations read; the usual is "
                                            "200,000), which does not count."),
    "an input use of 32 points per sweep": ("input_use", {"n_sweep": 32},
                                            "Run only at a reduced size (32 points per sweep; the usual is 33), which "
                                            "does not count."),
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
    the store gives the later one, whatever order they are listed in and whatever they are named; the others are named,
    newest first, by their ids."""
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
    named = [RecordRead(one.folder, {**one.manifest, "name": name}) for one, name in ((first, "z_first"),
                                                                                      (second, "a_second"))]
    for listed in (named, named[::-1]):
        row = _row(check_lineage.rows(_held(), listed, ()), "calibration")
        assert (row.record, row.earlier) == (named[1], ("20261014T170000",)), "a record was told by its name"
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
    row = _row(check_lineage.rows(_held(), [b_failed, a_passed, b_passed], ()), "band")
    assert row.earlier == (a_passed.manifest["id"], b_failed.manifest["id"]), "the earlier records not newest first"
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


def _with_never(states):
    """``_rows_of`` with the rows no run can pass for the posterior in use given their reasons: the comparison not
    compared, and the input use and the identifiability not run as a posterior trained without what they read."""
    reasons = {"input_use": check_lineage.NO_SIMULATIONS_TEXT, "identifiability": check_lineage.NO_ROTATION_TEXT}
    made = []
    for row in _rows_of([state if state != "never" else "Not run yet" for state in states]):
        never = row.key in check_catalogue.DECIDING and states[check_catalogue.DECIDING.index(row.key)] == "never"
        unread = row.key == "comparison" and states[check_catalogue.DECIDING.index(row.key)] == "unread"
        if (never or unread) and row.key == "comparison":
            row = CheckRow(row.key, "Not compared", "Not compared.", None, None, (), "")
        elif never:
            row = CheckRow(row.key, row.state, row.line, None, None, (), reasons[row.key])
        made.append(row)
    return tuple(made)


def _never_cases():
    """Every combination of the deciding states a posterior with the frequency response can hold, a check no run can
    pass among the comparison's, the input use's and the identifiability's states."""
    open_ = _OPEN_STATES
    for states in itertools.product(open_, open_, (*open_, "never"), open_, (*open_, "never"), (*open_, "never")):
        yield list(states)


def _expected_never(states) -> dict:
    """What the headline, its sentence and the summary say of six deciding states of a posterior trained with the
    frequency response, among them "never" for a check no run can pass for it, worked out here from the states alone:
    such a check is not passed, counts in the six, holds nothing open, is named in the sentence once nothing failed or
    needs a closer look, and is counted last on the second line."""
    names = [one.name for one in check_catalogue.CHECKS[:6]]
    passed = states.count("Passed")
    failed = [name for name, state in zip(names, states) if state == "Failed"]
    closer = [name for name, state in zip(names, states) if state == "Look closer"]
    never = [name for name, state in zip(names, states) if state == "never"]
    still = sum(1 for state in states if state not in ("Passed", "never"))
    opened = ("" if not still else "Inference can be run now; 1 deciding check is still open." if still == 1
              else f"Inference can be run now; {still} deciding checks are still open.")
    quoted = [f"'{name}'" for name in never]                 # each set off; "or" before the last
    named = f"{', '.join(quoted[:-1])} or {quoted[-1]}" if len(quoted) > 1 else "".join(quoted)
    if failed:
        sentence = "Failed: " + ", ".join(failed) + "."
    elif closer:
        sentence = "Look closer: " + ", ".join(closer) + "."
    elif opened and never:
        sentence = f"{opened} No run can pass {named} for this posterior."
    elif opened:
        sentence = opened
    elif never:
        sentence = f"Inference can be run now; no run can pass {named} for this posterior."
    else:
        sentence = "This posterior has passed every deciding check."
    second = (f"{len(failed)} failed" if failed else f"{len(closer)} to look closer at" if closer
              else f"{states.count('Not run yet')} not run yet" if "Not run yet" in states
              else f"{len(never)} that no run can pass" if never else "")
    return {"headline": f"{passed} of 6 deciding checks passed", "sentence": sentence, "passed": passed, "of": 6,
            "failed": bool(failed), "look_closer": bool(closer), "open": bool(still), "can_check": True,
            "second_line": second, "cannot_pass": len(never)}


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
    trained without it, and with none, and for every combination with a check no run can pass among them, or with the
    comparison not compared because its counts could not be read, which holds the step open, the headline,
    its sentence and every field of the summary are what the rules give from the states alone: n of 6, or n of 3 with
    the three that do not apply said for a posterior trained without it; the first sentence that applies; and the first
    count of the second line."""
    seen = 0
    for states, posterior, chi in _cases():
        held = _held(posterior=POSTERIOR if posterior else None, chi=chi)
        rows = _rows_of(states)
        want = _expected(states, posterior=posterior, chi=chi)
        said = check_lineage.summary(rows, held=held)
        assert check_lineage.headline(rows, held=held) == (want["headline"], want["sentence"]), (states, posterior, chi)
        assert {name: getattr(said, name) for name in want} == want, (states, posterior, chi)
        seen += 1
    for states in _never_cases():
        rows, held, want = _with_never(states), _held(), _expected_never(states)
        said = check_lineage.summary(rows, held=held)
        assert check_lineage.headline(rows, held=held) == (want["headline"], want["sentence"]), states
        assert {name: getattr(said, name) for name in want} == want, states
        seen += 1
        if states[2] != "never":
            continue
        # The same comparison not compared because the set's counts could not be read: a reading may lift that, so it
        # holds the step open, as a check still to pass.
        unread = [*states[:2], "unread", *states[3:]]
        rows, held, want = _with_never(unread), _held(set_counts=None), _expected_never(unread)
        said = check_lineage.summary(rows, held=held)
        assert check_lineage.headline(rows, held=held) == (want["headline"], want["sentence"]), unread
        assert {name: getattr(said, name) for name in want} == want, unread
        seen += 1
    assert seen == 4 ** 6 + 4 ** 3 + 4 ** 2 + 1 + 4 * 4 * 5 * 4 * 5 * 5 + 4 * 4 * 4 * 5 * 5
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


def test_a_check_no_run_can_pass_is_set_off_in_quotes_where_the_sentence_names_it():
    """The headline's sentence names a deciding check that no run can pass for the posterior in use in passing, set off
    in quotes as the window sets off a name inside a sentence, so that the sentence reads: one alone, two joined by
    "or", three with commas and "or" before the last. The checks that failed or need a closer look stand after a colon,
    where a name reads, as they did."""
    def sentence(states):
        return check_lineage.headline(_with_never(states), held=_held())[1]

    assert sentence(["Passed", "Passed", "Passed", "Passed", "never", "Not run yet"]) == (
        "Inference can be run now; 1 deciding check is still open. No run can pass 'Use of each part of the input' "
        "for this posterior.")
    assert sentence(["Passed", "Passed", "Passed", "Passed", "never", "never"]) == (
        "Inference can be run now; no run can pass 'Use of each part of the input' or 'Identifiability of the "
        "posterior' for this posterior.")
    assert sentence(["Passed", "Passed", "never", "Passed", "never", "never"]) == (
        "Inference can be run now; no run can pass 'Unusable driven recordings: simulation set against the audit', "
        "'Use of each part of the input' or 'Identifiability of the posterior' for this posterior.")
    assert sentence(["Failed", "Passed", "never", "Failed", "never", "Passed"]) == "Failed: Frequency band, Calibration."


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
    assert second(["Passed", "Passed", "Not compared", "Passed", "Passed", "Passed"]) == "1 that no run can pass"
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
    waiting = _row(check_lineage.rows(_held(set_counts=(8_412, 22_144, 100, WHOLE)), [], ()), "comparison")
    assert (waiting.state, waiting.when, waiting.line) == ("Waiting for the audit", None, "Waiting for the audit."), (
        "the comparison after simulating was dated while it waits for the audit")
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
    allowed = {"collections.abc", "dataclasses", "datetime", "json", "pathlib", "re", "statistics", ".",
               ".check_catalogue", ".check_readings", ".run", ".step_facts", ".texts"}
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
        "The prior in use is not the one this posterior was trained on, and that prior is no longer in Records.")
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
    twice = (histories[0], {"word": "Check", "check": {"key": "timing"}, "outcome": "finished",
                            "started": _at(12).isoformat()})
    assert _row(check_lineage.rows(_held(chi=False), records, twice), "timing").when == _at(12), "an older run decided"
    elsewhere = check_lineage.rows(_held(chi=False, config={**CONFIG, "model": "HOPF"}, posterior=OTHER_POSTERIOR),
                                   records, ())
    assert [_row(elsewhere, key).state for key in ("pin_down", "fixed_number", "drive_study", "across_prior")] == [
        "Not run yet"] * 4


def test_each_earlier_run_says_its_cell_its_number_and_why_it_does_not_count():
    """A check's earlier runs, newest first, each say what it read and when: a frequency band its cell, so that the runs
    of two cells are told apart; one judged with other thresholds, and an audit at a fixed number of driven recordings,
    that it does not count; a calibration repeated at fixed numbers how many driven recordings it held, or that they
    were mixed as in training. Each is looked for among its own check's records: a record of another kind with the same
    id is not read for it."""
    from tests._fixtures import check_results
    band, audit = check_results("band"), check_results("audit")
    other = {**band, "settings": {**band["settings"], "cv_max": 0.3}}
    fixed = {**audit, "settings": {**audit["settings"], "chi_k_fixed": 4}}
    records = [_record("band", created=_at(13, 9, 0), cell="Cells/nadrowski/a.txt"),
               _record("band", created=_at(12, 9, 0), cell="Cells/nadrowski/b.txt"),
               _record("band", created=_at(11, 9, 0), cell="Cells/nadrowski/a.txt", body=other),
               _record("audit", created=_at(13, 9, 0)), _record("audit", created=_at(12, 9, 0), body=fixed),
               _record("fixed_number", created=_at(13, 9, 0)),
               _record("fixed_number", created=_at(12, 9, 0), chi_k_fixed=4)]
    rows = {row.key: row for row in check_lineage.rows(_held(), records, ())}

    def lines(key):
        return check_lineage.earlier_lines(rows[key], records)

    assert lines("band") == (f"Passed, {_said(_at(12, 9, 0))}; cell b",
                             f"Run with other thresholds, {_said(_at(11, 9, 0))}; does not count; cell a"), lines("band")
    assert lines("audit") == (f"Run at a fixed number of driven recordings, {_said(_at(12, 9, 0))}; does not count",)
    assert lines("fixed_number") == (f"Done, {_said(_at(12, 9, 0))}; with 4 driven recordings",), lines("fixed_number")
    use = [_record("input_use", created=_at(13, 9, 0)), _record("input_use", created=_at(12, 9, 0), ident="shared"),
           _record("calibration", created=_at(12, 9, 0), ident="shared", passed=False)]
    row = _row(check_lineage.rows(_held(), use, ()), "input_use")
    assert check_lineage.earlier_lines(row, use) == (f"Passed, {_said(_at(12, 9, 0))}",), "another kind's record was read"


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


# ── the Checks list and a check's page ──
@contextlib.contextmanager
def _checks_page(tmp_path, *, chi=True, prior=True, posterior=True, records=None, size=(1300, 820), shown=True,
                 model="NADROWSKI", **made):
    """The Checks page in a whole window, in sight off the screen at ``size`` when ``shown``, on the test's own clock and
    store: the configuration of the master bounds file, with the frequency response or without it (``chi``), the first
    step applied with it, a real prior of it in use with ``prior`` and with ``posterior`` a stand-in posterior trained on
    it that says what the checks read of it (``_held_posterior``'s keywords in ``made``), the records ``records`` writes
    first (handed the store, the configuration, the prior's id and the posterior's), and the page read:
    (window, screen, page, store)."""
    from core.artifacts import ArtifactStore, use_store
    from core.gui.session import SbiSession
    from tests._fixtures import _nad_cfg, _prior_artifact, build_window, qt_app
    from tests._look_helpers import _settle
    from tests._runs import Clock, at, controller_on
    from tests._window_helpers import shown_off_screen
    qt_app()
    store = ArtifactStore(tmp_path / "Artifacts")
    with use_store(store):
        controller_on(tmp_path, Clock(at(8, 12, 4)))
        window = build_window()
        screen = window.part("inference")
        cfg = _nad_cfg(chi_mode=chi)
        if model != "NADROWSKI":
            cfg.model = model
        held = store.load_prior(cfg, _prior_artifact(store, cfg, name="retrain_prior").id) if prior else None
        loaded = _held_posterior(tmp_path, prior=held, **made) if prior and posterior else None
        if records is not None:
            records(store, cfg, getattr(held, "id", None), getattr(loaded, "id", None))
        first = screen.config_panel
        first.kind_choice.setCurrentIndex(first.kind_choice.findData(chi))
        first.foot.main_button.click()               # step 1 applied by its own button, so that it holds what it shows
        screen.session = SbiSession(draft=screen.session.draft, cfg=cfg, inf_prior=held, posterior=loaded)
        screen.prior_panel.bounds_picker.restore_key("master.txt")
        screen.refresh_gates()
        if shown:
            shown_off_screen(window, *size)
            window.go_to("inference", "checks")
            _settle()
        try:
            yield window, screen, screen.validate_panel, store
        finally:
            window.hide()


def _held_posterior(tmp_path, *, prior, ident=POSTERIOR, name="retrain", simulation=None, rotated=True,
                    other_prior=False, narrowed=False, text=""):
    """A posterior in use as a stand-in that says what the checks read of it: trained on ``prior``, with the network's
    own prior that prior's or another (``other_prior``), its simulation set ``simulation`` (None: not kept), the
    rotation or not, narrowed in a round or not, and the text of its record (``text``) in a folder of its own."""
    import torch
    from torch.distributions import Categorical, MixtureSameFamily, Normal
    folder = tmp_path / f"posterior_{ident}"
    folder.mkdir(exist_ok=True)
    (folder / "log.txt").write_bytes(text.encode("utf-8"))
    parents = {"prior": prior.id, **({"simulation": simulation} if simulation else {})}
    manifest = SimpleNamespace(parents=parents, body={"transform": {"V": [[1.0]] if rotated else None}},
                               created="2026-10-03T09:00:00+00:00")
    own = (MixtureSameFamily(Categorical(torch.ones(2)), Normal(torch.tensor([0.0, 9.0]), torch.ones(2)))
           if other_prior else prior.prior)
    network = SimpleNamespace(latent=SimpleNamespace(prior=own), truncation=object() if narrowed else None)
    return SimpleNamespace(kind="posterior", id=ident, name=name, path=folder, manifest=manifest, posterior=network)


def _shown_rows(page) -> list:
    """The whole table's rows as they are drawn: each row's cells' words, a heading's alone."""
    model = page.listing.whole.view.model()
    return [tuple(str(model.data(model.index(row, column)) or "") for column in range(model.columnCount()))
            for row in range(model.rowCount())]


def _texts(widget) -> list:
    """Every label's words in ``widget`` that stands in sight, in the order they were made."""
    from PySide6.QtWidgets import QLabel
    return [label.text() for label in widget.findChildren(QLabel) if label.isVisibleTo(widget) and label.text()]


def test_the_list_shows_twelve_checks_in_four_groups_with_role_state_and_last_run(tmp_path):
    """One ruled table, its columns "Check", "Role", "State" and "Last run": the four groups' headings as rows across it,
    each check under its group in the list's order, with its role, the state its row reads and the date of the record
    that decides it. Before the checks are first read the names stand alone."""
    def records(store, cfg, prior, posterior):
        from tests._fixtures import write_calibration
        write_calibration(store, cfg, posterior=posterior, prior=prior, created=_at(12, 14, 5))

    with _checks_page(tmp_path, records=records) as (_window, screen, page, _store):
        table = page.listing.whole.view
        assert [table.model().headerData(at, Qt.Horizontal) for at in range(4)] == ["Check", "Role", "State",
                                                                                  "Last run"]
        shown = _shown_rows(page)
        groups = [row[0] for at, row in enumerate(shown) if at in page.listing.headings]
        assert groups == list(check_catalogue.GROUPS) and len(shown) == 16
        for at in page.listing.headings:
            assert table.columnSpan(at, 0) == 4, at
        rows = {row.key: row for row in screen.check_rows()}
        for one in check_catalogue.CHECKS:
            cells = shown[page.listing.row_of[one.key]]
            assert cells[:3] == (one.name, one.role, rows[one.key].state), one.key
        assert shown[page.listing.row_of["calibration"]][3] == _said(_at(12, 14, 5))
        assert shown[page.listing.row_of["band"]][3] == ""
        order = [page.listing.key_at[at] for at in sorted(page.listing.key_at)]
        assert order == [one.key for group in check_catalogue.GROUPS for one in check_catalogue.CHECKS
                         if one.group == group]
    from core.gui.panels.inference.validate_tab import _Listing
    from tests._fixtures import qt_app
    qt_app()
    bare = _Listing()
    try:
        model = bare.whole.view.model()
        assert [str(model.data(model.index(bare.row_of["band"], column)) or "") for column in range(4)] == [
            "Frequency band", "Decides", "", ""], "a state was drawn before the checks were read"
    finally:
        bare.deleteLater()


@pytest.mark.parametrize("held", ["nothing", "a prior", "a prior, the frequency response off", "a posterior",
                                  "a posterior trained without the frequency response", "a posterior of another prior"])
def test_the_heading_sentence_and_in_use_follow_what_is_held(held, tmp_path):
    """The heading names the posterior in use, or, with a prior alone, the prior a training would be made on, or neither;
    the sentence under it counts the deciding checks that apply to the posterior, all six with no posterior whatever
    the frequency response; and "In use" names the posterior and the prior it was trained on, or both when the prior
    in use is another, or the prior alone."""
    chi = held not in ("a posterior trained without the frequency response", "a prior, the frequency response off")
    with _checks_page(tmp_path, chi=chi, prior=held != "nothing", posterior=held.startswith("a posterior"),
                      shown=False) as (_window, screen, page, _store):
        if held == "a posterior of another prior":
            screen.session.posterior.manifest.parents["prior"] = OTHER_PRIOR
            page.refresh_local_gates()
        want = {"nothing": ("Checks", "A posterior passes when all six deciding checks have passed.",
                            "In use: nothing yet."),
                "a prior": ("Checks before training on retrain_prior", "A posterior passes when all six deciding "
                            "checks have passed.", "In use: prior retrain_prior"),
                "a prior, the frequency response off": (
                    "Checks before training on retrain_prior", "A posterior passes when all six deciding checks have "
                    "passed.", "In use: prior retrain_prior"),
                "a posterior": ("Checks of retrain", "A posterior passes when all six deciding checks have passed.",
                                "In use: posterior retrain, trained on prior retrain_prior"),
                "a posterior trained without the frequency response": (
                    "Checks of retrain", "A posterior passes when the three deciding checks that apply to it have "
                    "passed.", "In use: posterior retrain, trained on prior retrain_prior"),
                "a posterior of another prior": ("Checks of retrain", "A posterior passes when all six deciding "
                                                 "checks have passed.", "In use: posterior retrain and prior "
                                                                        "retrain_prior")}[held]
        assert (page.head.title.text(), page.head.sentence.plain_text(), page.in_use.line_text()) == want


def _label_size(label) -> float:
    """How large a label's words are set: its font's pixels, or its points."""
    font = label.font()
    return font.pixelSize() if font.pixelSize() > 0 else font.pointSizeF()


def test_the_headline_is_the_pages_only_large_line(tmp_path):
    """With no check open, the headline is the one line of the page's body set in the heading's type, its size and its
    weight, the count of the deciding checks passed, and the sentence under it is in the book face; the page's own
    heading stands above the body. Before the checks are first read there is no headline."""
    from PySide6.QtWidgets import QLabel
    from core.gui import fonts
    from core.gui.widgets.page import Sentence
    with _checks_page(tmp_path) as (_window, screen, page, _store):
        heading = fonts.inter("head")
        big = heading.pixelSize() if heading.pixelSize() > 0 else heading.pointSizeF()
        large = [label for label in page.scroller.body.findChildren(QLabel)
                 if label.isVisibleTo(page) and label.text() and _label_size(label) >= big
                 and label.font().weight() >= heading.weight()]
        assert large == [page.headline], [label.text() for label in large]
        assert page.headline.text() == screen.checks_summary().headline == "0 of 6 deciding checks passed"
        assert isinstance(page.headline_sentence, Sentence) and page.headline_sentence.isVisibleTo(page)
        assert page.headline_sentence.plain_text() == screen.checks_summary().sentence
        assert page.headline_sentence.font().weight() < heading.weight()
        screen.checks_held = None
        page.refresh_local_gates()
        assert page.headline_block.isHidden() and page.headline.text() == ""


def _row_point(view, row: int):
    """The middle of the row ``row`` of ``view``, in its viewport."""
    return view.visualRect(view.model().index(row, 0)).center()


@pytest.mark.parametrize("factor", [1.0, 1.5])
def test_choosing_a_check_opens_its_page_beside_the_list_or_in_its_place(factor, tmp_path):
    """A click on a check's row, Space or Enter on it, opens its page: beside the list, the list told narrow to its name
    and its state, where the page's body is at least the width the page names, grown with the text; in the list's place
    under "All checks" where it is narrower; the switch reading the width alone. The calibration's first setting takes
    the keyboard, and "All checks" on a check with no setting; "All checks" brings the whole list back, the keyboard on
    the row of the check left. A group's heading opens nothing."""
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication
    from core.gui import design
    from tests._look_helpers import _settle, text_size
    from tests._window_helpers import in_front
    with text_size(factor), _checks_page(tmp_path, size=(design.scaled(1400), design.scaled(600))) as (
            window, _screen, page, _store):
        assert page.list_detail.beside_width == design.scaled(890)
        calibration = page.check_pages["calibration"]
        with in_front(window):
            view = page.listing.whole.view
            QTest.mouseClick(view.viewport(), Qt.LeftButton, pos=_row_point(view, page.listing.headings[1]))
            _settle()
            assert page.open_key is None, "a group's heading opened a check"
            QTest.mouseClick(view.viewport(), Qt.LeftButton, pos=_row_point(view, page.listing.row_of["calibration"]))
            _settle()
            assert (page.open_key, page.list_detail.beside(), page.listing.narrowed) == ("calibration", True, True)
            assert page.listing.narrow.isVisibleTo(page) and not page.listing.whole.isVisibleTo(page), (
                "beside a check's page the whole table stayed in sight")
            assert [column.wrap for column in page.listing.narrow.view._columns] == [False, True], (
                "the narrow table's state does not wrap")
            assert page.listing.width() == design.scaled(436), (factor, page.listing.width())
            page.read_facts()
            assert (page.listing.narrow.selected_row(), page.listing.narrow.view.currentIndex().row()) == (
                page.listing.row_of["calibration"],) * 2, "a reading lost the open check's row"
            assert page.list_detail.detail is calibration and calibration.isVisibleTo(page)
            assert QApplication.focusWidget() is calibration.form.rows["n_cal"].widget
            assert page.form is calibration.form, "the foot was not handed the check's form"
            narrow = page.listing.narrow.view
            narrow.setFocus(Qt.TabFocusReason)
            narrow.setCurrentIndex(narrow.model().index(page.listing.row_of["band"], 0))
            QTest.keyClick(narrow, Qt.Key_Return)
            _settle()
            assert page.open_key == "band" and page.list_detail.detail is page.check_pages["band"]
            narrow.setFocus(Qt.TabFocusReason)
            narrow.setCurrentIndex(narrow.model().index(page.listing.row_of["audit"], 0))
            QTest.keyClick(narrow, Qt.Key_Space)
            _settle()
            assert page.open_key == "audit"
            window.resize(design.scaled(900), design.scaled(800))
            _settle()
            assert page.open_key == "audit" and not page.list_detail.beside()
            assert not page.listing.isVisibleTo(page) and page.list_detail.back.isVisibleTo(page)
            assert page.check_pages["audit"].isVisibleTo(page)
            page.list_detail.back.setFocus(Qt.TabFocusReason)
            QTest.keyClick(page.list_detail.back, Qt.Key_Space)
            _settle()
            assert page.open_key is None and page.list_detail.detail is None and page.form is page._own_form
            assert page.listing.isVisibleTo(page) and not page.listing.narrowed
            whole = page.listing.whole.view
            assert QApplication.focusWidget() is whole and whole.currentIndex().row() == page.listing.row_of["audit"]
            page.open_check("calibration")
            _settle()
            assert QApplication.focusWidget() is calibration.form.rows["n_cal"].widget
            page.open_check("comparison")
            _settle()
            assert QApplication.focusWidget() is page.list_detail.back, "a check with no setting left the keyboard"
        with pytest.raises(KeyError):
            page.open_check("calibrate")


def test_a_checks_page_reads_name_sentence_state_result_earlier_runs_and_form_in_that_order(tmp_path):
    """A check's page, top to bottom: its name, the sentence under it, its state line with its state's marker, the
    result of the record that decides it, its earlier runs newest first, each with its state and its date, a run at a
    reduced size said not to count, and its form."""
    from core.gui.check_readings import STATE_MARKERS

    def records(store, cfg, prior, posterior):
        from tests._fixtures import check_results, write_calibration
        write_calibration(store, cfg, posterior=posterior, prior=prior, created=_at(10, 15, 0),
                          results=check_results("calibration", n_cal=40)["results"])
        write_calibration(store, cfg, posterior=posterior, prior=prior, created=_at(11, 9, 30),
                          results=check_results("calibration", passed=False)["results"])
        write_calibration(store, cfg, posterior=posterior, prior=prior, created=_at(12, 14, 5),
                          results=_calibration_of(["k", "x_scale", "t_scale"], ks=[0.5] * 3))

    with _checks_page(tmp_path, records=records) as (_window, _screen, page, _store):
        page.open_check("calibration")
        one = page.check_pages["calibration"]
        parts = [one.heading, one.sentence, one.state_line, one.result, one.earlier_head, one.earlier, one.form]
        assert all(part.isVisibleTo(page) for part in parts), [part.isVisibleTo(page) for part in parts]
        tops = [part.mapTo(one, part.rect().topLeft()).y() for part in parts]
        assert tops == sorted(tops) and len(set(tops)) == len(tops), tops
        assert (one.heading.text(), one.sentence.plain_text()) == (
            "Calibration", "Is the posterior honest about its uncertainty?")
        assert one.state_line.sentence.plain_text() == (f"Passed, {_said(_at(12, 14, 5))}. An earlier run failed "
                                                        f"({_said(_at(11, 9, 30))}).")
        assert one.state_line.kind == STATE_MARKERS["Passed"] == "fine"
        assert one.result.verdict.sentence.plain_text().startswith("Calibration: passed.")
        assert one.earlier_head.text() == "Earlier runs"
        assert one.earlier.text().split("\n") == [f"Failed, {_said(_at(11, 9, 30))}",
                                                  f"Run at a reduced size, {_said(_at(10, 15, 0))}; does not count"]
        assert list(one.form.rows) == ["n_cal", "cal_n_scales", "num_posterior_samples", "seed", "name", "note"]
        assert [row[:2] for row in one.result.table_words().rows[:3]] == [("k", ""), ("x_scale", "nm"),
                                                                         ("t_scale", "ms")], "the units were not read"


#: Each check's greyed button, as values: the check, what is in use that holds it there, the facts that grey it and
#: the words beside it.
_GREYED = {
    "the calibration with no posterior": (
        "calibration", dict(posterior=None), [("lock", "Needs a posterior: train one in step 3, or load a saved one.")],
        "Cannot start yet: needs a posterior."),
    "the calibration with the prior of another": (
        "calibration", dict(pair_refused="refused", posterior_prior_words="older_prior"),
        [("lock", "The prior in use is not the one this posterior was trained on (older_prior). Load it in step 2.")],
        "Cannot start yet: the prior in use is not the posterior's own."),
    "the calibration with the prior of another, its record gone": (
        "calibration", dict(pair_refused="refused"),
        [("lock", "The prior in use is not the one this posterior was trained on, and that prior is no longer in "
                  "Records.")],
        "Cannot start yet: the prior in use is not the posterior's own."),
    "the calibration with no prior": (
        "calibration", dict(prior=None), [("lock", "Needs a prior: build or load one in step 2, Prior.")],
        "Cannot start yet: needs a prior."),
    "the input use, trained without its simulations": (
        "input_use", dict(simulation=None),
        [("lock", "This posterior was trained without keeping its simulations, and this check reads them.")],
        "Cannot start: the posterior was trained without keeping its simulations."),
    "the identifiability, trained without the rotation": (
        "identifiability", dict(rotated=False),
        [("lock", "This posterior was trained without the rotation, so there are no directions to read.")],
        "Cannot start: the posterior was trained without the rotation."),
    "the frequency band without the frequency response": (
        "band", dict(chi=False), [("lock", "Applies only with the frequency response.")],
        "Cannot start: it applies only with the frequency response."),
    "the audit without the frequency response": (
        "audit", dict(chi=False), [("lock", "Applies only with the frequency response.")],
        "Cannot start: it applies only with the frequency response."),
    "what the observations pin down, of another model": (
        "pin_down", dict(nadrowski=False), [("lock", "Applies only to the Nadrowski model.")],
        "Cannot start: it applies only to the Nadrowski model."),
    "what the observations pin down, with no forcing and no frequency response": (
        "pin_down", dict(chi=False, forcing=False), [("lock", "Needs a bounds file with a forcing section.")],
        "Cannot start: the bounds file has no forcing section."),
    "the drive study with no forcing section": (
        "drive_study", dict(forcing=False), [("lock", "Needs a bounds file with a forcing section.")],
        "Cannot start: the bounds file has no forcing section."),
    "identifiability across the prior with the frequency response": (
        "across_prior", dict(), [("lock", "Does not apply with the frequency response.")],
        "Cannot start: it does not apply with the frequency response."),
    "identifiability across the prior with no forcing section": (
        "across_prior", dict(chi=False, forcing=False), [("lock", "Needs a bounds file with a forcing section.")],
        "Cannot start: the bounds file has no forcing section."),
    "the calibration repeated, with the prior of another": (
        "fixed_number", dict(pair_refused="refused", posterior_prior_words="older_prior"),
        [("lock", "The prior in use is not the one this posterior was trained on (older_prior). Load it in step 2.")],
        "Cannot start yet: the prior in use is not the posterior's own."),
    "the calibration, free to run": ("calibration", dict(), [], ""),
}


#: The words beside a check's greyed button for each reason, by the start of its sentence, and the lead they read alone:
#: a load lifts the first three; waiting lifts none of the others.
_BLOCK_WORDS = (("Needs a prior", "needs a prior", "Cannot start yet:"),
                ("Needs a posterior", "needs a posterior", "Cannot start yet:"),
                ("The prior in use is not", "the prior in use is not the posterior's own", "Cannot start yet:"),
                ("This posterior was trained without keeping", "the posterior was trained without keeping its simulations",
                 "Cannot start:"),
                ("This posterior was trained without the rotation", "the posterior was trained without the rotation",
                 "Cannot start:"),
                ("Applies only with", "it applies only with the frequency response", "Cannot start:"),
                ("Applies only to the Nadrowski", "it applies only to the Nadrowski model", "Cannot start:"),
                ("Needs a bounds file", "the bounds file has no forcing section", "Cannot start:"),
                ("Does not apply with", "it does not apply with the frequency response", "Cannot start:"))


@pytest.mark.parametrize("model, bounds, applies", [
    ("NADROWSKI", "master.txt", {"pin_down": True, "drive_study": True, "across_prior": True}),
    ("HOPF", "cell.txt", {"pin_down": False, "drive_study": True, "across_prior": True}),
    ("NADROWSKI", "master_spont.txt", {"pin_down": False, "drive_study": False, "across_prior": False})],
    ids=["Nadrowski with its forcing section", "Hopf", "Nadrowski with no forcing section"])
def test_the_checks_read_the_model_and_the_forcing_section_of_the_configuration_in_use(model, bounds, applies, tmp_path):
    """What the checks are told of the configuration in use is read off the configuration itself, as the screen reads
    the checks: another model than Nadrowski's makes "What the observations can pin down" not apply, and a bounds file
    with no forcing section makes it, the drive study and identifiability across the prior not apply, each in its row's
    words; with Nadrowski's master bounds file each applies."""
    from core import cli, config, registry
    from core.gui.session import SbiSession
    folder = {"NADROWSKI": "nadrowski", "HOPF": "hopf"}[model]
    cfg = cli.make_sim_config(model, config.VALID_LABELS[config.VALID_MODELS.index(model)],
                              registry.state_dep_drift(model), str(config.BOUNDS_PATH / folder / bounds))
    cfg.hw = config.cpu_device()
    with _checks_page(tmp_path, chi=False, shown=False) as (_window, screen, _page, _store):
        screen.session = SbiSession(draft=screen.session.draft, cfg=cfg, inf_prior=screen.session.inf_prior,
                                    posterior=screen.session.posterior)
        screen.read_checks()
        rows = {row.key: row for row in screen.check_rows()}
        said = {key: (rows[key].state != "Does not apply", rows[key].reason) for key in applies}
        why = {"pin_down": check_lineage.NEEDS_FORCING_TEXT if model == "NADROWSKI" else check_lineage.NADROWSKI_ONLY_TEXT,
               "drive_study": check_lineage.NEEDS_FORCING_TEXT, "across_prior": check_lineage.NEEDS_FORCING_TEXT}
        assert said == {key: (on, "" if on else why[key]) for key, on in applies.items()}, said


@pytest.mark.parametrize("case", list(_GREYED))
def test_each_checks_button_is_greyed_with_the_reason_of_its_table(case):
    """Each check's button is greyed by the reason its row gives, each in its table's exact words: a block a load lifts,
    the prior, the posterior or the posterior's own prior, says "Cannot start yet:"; one that waiting will not lift, a
    posterior trained without what the check reads, a model, a kind of observation or a bounds file the check does not
    apply to, says "Cannot start:". The prior a posterior was trained on is named
    when the reading found its record, and said without its name when it is gone. Each reason has its own words beside
    the button, read alone after their lead. A live run is said first."""
    from core.gui.step_facts import reason_beside
    key, given, facts, beside = _GREYED[case]
    row = _row(check_lineage.rows(_held(**given), (), ()), key)
    made = check_lineage.button_facts(row)
    assert [(fact.kind, fact.sentence) for fact in made] == facts, made
    assert reason_beside(made, live=False, nothing_to_apply=False) == beside
    for fact in made:
        _start, words, lead = next(one for one in _BLOCK_WORDS if fact.sentence.startswith(one[0]))
        assert fact.blocks == words, (fact.sentence, fact.blocks)
        assert reason_beside((fact,), live=False, nothing_to_apply=False) == f"{lead} {words}.", fact
    assert reason_beside(made, live=True, nothing_to_apply=False) == "A run is live. Only one runs at a time."


@pytest.mark.parametrize("pair", ["the posterior's own prior", "another prior, its record kept",
                                  "another prior, its record gone", "a prior of another id and the same fingerprint"])
def test_the_prior_pair_greys_the_calibration_by_the_engines_guard_and_names_the_posteriors_prior(pair, tmp_path):
    """On the page, the calibration's button is greyed for the prior in use only where the engine's own guard refuses
    the pair, by the prior's fingerprint: greyed beside a posterior trained on another prior, which is named when its
    record is in the store and said without its name when it is gone; live beside its own prior, and beside a prior of
    another id with the same fingerprint, which the engine takes."""
    from tests._fixtures import _prior_artifact, check_results, write_diagnostic
    kept = {}

    def records(store, cfg, prior, posterior):
        kept["older"] = store.load_prior(cfg, _prior_artifact(store, cfg, name="older_prior", seed=5).id)
        write_diagnostic(store, cfg, diagnostic="probes", variant="mask", parents={"prior": kept["older"].id},
                         **check_results("audit"))

    with _checks_page(tmp_path, records=records) as (_window, screen, page, store):
        held = screen.session.inf_prior
        if pair != "the posterior's own prior":
            older = kept["older"]
            screen.session.posterior = _held_posterior(tmp_path, prior=older)
            if pair == "a prior of another id and the same fingerprint":
                screen.session.posterior.posterior.latent.prior = held.prior
            if pair == "another prior, its record gone":
                store.delete("prior", older.id, force=True)
        page.read_facts()
        page.open_check("calibration")
        live = pair in ("the posterior's own prior", "a prior of another id and the same fingerprint")
        assert page.foot.main_button.isEnabled() == live, (pair, page.foot.reason.text())
        assert page.foot.reason.text() == ("" if live else "Cannot start yet: the prior in use is not the posterior's "
                                                           "own.")
        facts = [fact.sentence for fact in page.facts_from_form()]
        # a prior gone from Records cannot be loaded, and its sentence asks for nothing
        sentence = ("The prior in use is not the one this posterior was trained on, and that prior is no longer in "
                    "Records." if pair == "another prior, its record gone" else
                    "The prior in use is not the one this posterior was trained on (older_prior). Load it in step 2.")
        assert (sentence in facts) == (not live), facts
        if pair == "another prior, its record kept":
            audit = page.check_pages["audit"].result.verdict.sentence.plain_text()
            assert "simulated from the prior older_prior were unusable" in audit, audit


def test_the_foot_with_no_check_open_offers_the_next_check(tmp_path):
    """With no check open the foot's main button opens the first deciding check that applies, has not passed, does not
    wait for a posterior or for the simulations, and that a run can still pass, named for it; when only checks that
    wait for those are left, before a training, it reads "Go to Training" and opens step 3; and it goes to Inference
    once none is left. It says no fact of its own, and with no prior in use it is greyed with the reason of a step not
    done. A check open reads its own button's words, and "Run again" once a record of it exists; on the page of a check
    with no button of its own, the comparison's, the foot passes over that check, which stands open already, to the
    next, to Training or to Inference."""
    from tests._fixtures import check_results, write_diagnostic
    from tests._runs import masked_none
    nothing_counted = f"10:00:00 info {masked_none(WHOLE)}\n"     # a posterior whose set counted no driven recording
    with _checks_page(tmp_path, prior=False) as (_window, _screen, page, _store):
        assert (page.foot.main_button.text().replace("&", ""), page.foot.main_button.isEnabled(),
                page.foot.reason.text()) == ("Open Frequency band", False, "Cannot start yet: needs a prior.")
        assert [(fact.kind, fact.sentence) for fact in page.facts_from_form()] == [
            ("lock", "Needs a prior: build or load one in step 2, Prior.")]
    with _checks_page(tmp_path / "held", posterior=False) as (window, screen, page, store):
        assert page.foot.main_button.text().replace("&", "") == "Open Frequency band"
        assert page.foot.main_button.isEnabled() and page.facts_from_form() == ()
        page.foot.main_button.click()
        assert page.open_key == "band" and page.list_detail.detail is page.check_pages["band"]
        page.open_check(None)
        prior = screen.session.inf_prior.id
        write_diagnostic(store, screen.session.cfg, diagnostic="probes", variant="band",
                         cell=str(_cell_file()), **check_results("band"))
        write_diagnostic(store, screen.session.cfg, diagnostic="probes", variant="mask", parents={"prior": prior},
                         **check_results("audit"))
        page.read_facts()
        # before a training, what is left waits for a posterior or for the simulations, which step 3 makes
        assert [(row.key, row.state) for row in screen.check_rows() if row.key in check_catalogue.DECIDING
                and row.state != "Passed"] == [("comparison", "Waiting for the simulations"),
                                               ("calibration", "Waiting for a posterior"),
                                               ("input_use", "Waiting for a posterior"),
                                               ("identifiability", "Waiting for a posterior")]
        assert page.foot.main_button.text().replace("&", "") == "Go to Training", "a check that waits was offered"
        page.open_check("comparison")
        assert page.foot.main_button.text().replace("&", "") == "Go to Training", (
            "the comparison's page offered a check that waits for training")
        page.foot.main_button.click()
        assert window.current_place() == ("inference", "training")
        window.go_to("inference", "checks")
        page.open_check(None)
        screen.session.posterior = _held_posterior(tmp_path, prior=screen.session.inf_prior, text=nothing_counted)
        page.read_facts()
        assert _row(screen.check_rows(), "comparison").state == "Not compared"
        assert page.foot.main_button.text().replace("&", "") == "Open Calibration", (
            "a check no run can pass was offered next")
        page.open_check("comparison")
        assert page.foot.main_button.text().replace("&", "") == "Open Calibration", (
            "the comparison's page offered itself")
        page.open_check(None)
        page.foot.main_button.click()
        assert page.open_key == "calibration"
        assert page.foot.main_button.text().replace("&", "") == "Run the calibration"
        page.open_check(None)
        for key, made in (("calibration", None), ("input_use", None), ("identifiability", None)):
            posterior = screen.session.posterior.id
            if key == "calibration":
                from tests._fixtures import write_calibration
                write_calibration(store, screen.session.cfg, posterior=posterior, prior=prior)
            else:
                one = check_catalogue.entry(key)
                write_diagnostic(store, screen.session.cfg, diagnostic=one.diagnostic, variant=one.variant,
                                 parents={"posterior": posterior, "prior": prior}, **check_results(key))
        page.read_facts()
        page.open_check("calibration")
        assert page.foot.main_button.text().replace("&", "") == "Run again"
        page.open_check(None)
        assert page.foot.main_button.text().replace("&", "") == "Go to Inference"
        screen.session.posterior = _held_posterior(tmp_path, prior=screen.session.inf_prior)    # counts not read
        page.read_facts()
        assert _row(screen.check_rows(), "comparison").line == (
            "Not compared: the counts of unusable driven recordings could not be read.")
        assert page.foot.main_button.text().replace("&", "") == (
            "Open Unusable driven recordings: simulation set against the audit"), "counts not read closed the step"
        page.open_check("comparison")
        assert page.foot.main_button.text().replace("&", "") == "Go to Inference"
        page.foot.main_button.click()
        assert window.current_place() == ("inference", "infer")


def _cell_file() -> Path:
    """A cell file the store can record a record was made on: the master's spontaneous cell, read and never written."""
    from core import config
    return Path(config.RESOURCES_ROOT) / "Cells" / "nadrowski" / "master_spont.txt"


#: The fields a record holds the engine's own words in: its verdicts and readings, its reasons, summary and caveats,
#: the calibration's caveat and description, the repeated calibration's scope, the drive study's notes and the audit's
#: dominant cause.
_OWN_WORDS = ("verdict", "reading", "reasons", "summary", "caveats", "caveat", "description", "scope", "note", "notes",
              "dominant_cause")
#: The checks whose records hold none of the engine's own words: the engine writes only numbers and names for them.
_NO_OWN_WORDS = ("pin_down", "across_prior")


def _own_words(found, said: set) -> set:
    """Every string a record holds under one of ``_OWN_WORDS``, at any depth: in a list under such a field, or in a
    field of that name inside any other, the calibration's verdict holding its caveat."""
    if isinstance(found, dict):
        for name, value in found.items():
            if name in _OWN_WORDS:
                for one in (value if isinstance(value, list) else [value]):
                    if isinstance(one, str) and one:
                        said.add(one)
            _own_words(value, said)
    elif isinstance(found, list):
        for one in found:
            _own_words(one, said)
    return said


def _block_words(region) -> list:
    """Every word a result region says: its verdict, its sentences, each cell of its table, its caption and its footnote,
    and its pictures' captions."""
    words = [region.verdict.sentence.plain_text()] if region.verdict is not None else []
    words += [fact.sentence for fact in region.sentences()]
    table = region.table_words()
    if table is not None:
        words += [cell for row in table.rows for cell in row] + [table.caption, table.footnote]
    words += [caption.text() for caption, _picture in region.pictures]
    return [word for word in words if word]


#: Each check's result from the record each fixture writes by default, the sentence it begins with.
_RESULT_SENTENCES = {
    "band": "Frequency band: passed for cell master_spont. Every frequency inside the band (0.03 to 0.3 times the "
            "cell's own) can be measured at every recording length tried, and the drive amplitude 0.15 does not take "
            "over the cell.",
    "audit": "Unusable driven recordings, audit: passed. 37.9 % of 22,400 driven recordings simulated from the prior "
             "retrain_prior were unusable. The expected share is 37 %; within 8 points is fine. Every one was too "
             "short or too slow to hold two cycles of the drive.",
    "calibration": "Calibration: passed. All 2 parameters pass the rank test, and the joint coverage test passes. Made "
                   "on 2,000 test datasets at 200 operating points (pairs of time scale and recording length), seed "
                   "1840273.",
    "input_use": "Use of each part of the input: passed. The network responds to 50 of the 50 parts of an "
                 "observation's summary; every part varies in the training simulations; none is invisible to it.",
    "identifiability": "Identifiability of the posterior: passed. The record holds how strongly each of the 6 "
                       "directions is pinned down: the strongest 333 times the weakest. This check passes when those "
                       "figures are on record; the table says what they are.",
    "pin_down": "What the observations can pin down: done. At 4.5 s, every parameter has local information and no "
                "pair moves an observation alike.",
    "fixed_number": "Calibration repeated at fixed numbers of driven recordings: done. With the number of driven "
                    "recordings mixed, as in training, 10 of 10 repeats pass the rank test. This is the rank half "
                    "only; the joint coverage test is Calibration's.",
    "drive_study": "Drive study: done. The cell keeps its own oscillation up to a drive of 0.3 pN and is captured from "
                   "1.6 pN.",
    "across_prior": "Identifiability across the prior: done, at 6 points across the prior. The table gives each "
                    "parameter's spread at each point and the share of points where recordings pin it down.",
}


@pytest.mark.parametrize("key", [one.key for one in check_catalogue.CHECKS])
def test_every_result_block_says_its_record_in_the_windows_words(key, tmp_path):
    """Each check's result begins with its sentence in the window's words, a message row with the marker of the state
    its record reads: of a record, from its manifest and its folder; of the comparison, from the counts; of a check that
    keeps nothing but its result, from its run in the history. No string of the record's own verdicts, readings or
    reasons stands in it, and a string of several words of them nowhere within its words. Each record holds such
    words in every field the engine writes them in, so that a result that said them would be seen."""
    from PySide6.QtWidgets import QWidget
    from core.artifacts import ArtifactStore
    from core.gui.panels.inference import check_results as blocks
    from tests._fixtures import check_results, qt_app, write_calibration, write_diagnostic
    qt_app()
    host = QWidget()                                 # what a check's page is to its result
    if key == "comparison":
        compared = check_readings.compare(unusable=8_412, total=22_144, scope=WHOLE, audit_fraction=0.3789,
                                          batches=100)
        region = blocks.comparison_block(check_readings.compared_sentence(compared, unusable=8_412, total=22_144,
                                                                          audit_fraction=0.3789), compared.state,
                                         parent=host)
        said = ("Unusable driven recordings, simulation set against the audit: passed. 38.0 % in the simulation set; "
                "the audit read 37.9 %. The difference is 0.1 points: within 3 is fine, up to 5 look closer, beyond 5 "
                "stop.")
        assert (region.verdict.kind, region.verdict.sentence.plain_text()) == ("fine", said)
        host.deleteLater()
        return
    if key in ("timing", "card_test"):
        for outcome, marker, word in (("finished", "done", "done"), ("failed", "err", "failed")):
            region = blocks.history_block(key, {"check": {"key": key}, "outcome": outcome}, parent=host)
            assert (region.verdict.kind, region.verdict.sentence.plain_text()) == (
                marker, f"{check_catalogue.entry(key).name}: {word}."), outcome
        host.deleteLater()
        return
    store = ArtifactStore(tmp_path / "Artifacts")
    one = check_catalogue.entry(key)
    if key == "calibration":
        ident = write_calibration(store, None, posterior=POSTERIOR, prior=PRIOR)
    else:
        variant = "pooled" if key == "fixed_number" else one.variant
        ident = write_diagnostic(store, None, diagnostic=one.diagnostic, variant=variant, parents={"prior": PRIOR},
                                 cell=str(_cell_file()) if key == "band" else None, **check_results(key))
    folder = store.path(one.kind, ident)
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    region = blocks.result_block(key, manifest, folder, parent=host, prior_words="the prior retrain_prior",
                                 force_unit="pN")
    try:
        state = check_readings.read(key, manifest).state
        assert region.verdict.kind == check_readings.STATE_MARKERS[state]
        assert region.verdict.sentence.plain_text() == _RESULT_SENTENCES[key]
        words = _block_words(region)
        own = _own_words(manifest, set())
        assert bool(own) == (key not in _NO_OWN_WORDS), (key, own)
        assert not [mine for mine in own if mine.lower() in {word.lower() for word in words}], own
        assert not [mine for mine in own if " " in mine and any(mine.lower() in word.lower() for word in words)]
    finally:
        host.deleteLater()


def _calibration_of(names, *, ks, assumed=(), seed=1840273, tarp=0.5, accepted=()):
    """A calibration's results as the stage writes them, over the parameters ``names`` with the rank tests' p-values
    ``ks``, by the engine's own verdict."""
    from core import config, orchestrator
    return {"sbc": {"per_param": [{"name": name, "ks_p": p, "c2st_ranks": 0.5, "c2st_dap": 0.5}
                                  for name, p in zip(names, ks)]},
            "tarp": {"atc": 0.0123, "ks_p": tarp, "num_bins": config.TARP_NUM_BINS},
            "informativeness": {"total_nats": 2.31, "sem_nats": 0.05, "per_param": None, "per_direction": None,
                                "n_used": 2000, "n_dropped": 0, "description": "the engine's own description"},
            "kept_fraction": None, "n_cal": 2000, "cal_n_scales": 200, "num_posterior_samples": 1000,
            "verdict": orchestrator.calibration_verdict(ks, names, list(assumed), tarp), "seed": seed,
            "accepted": list(accepted)}


def test_the_calibrations_result_names_its_verdict_and_its_seed(tmp_path):
    """The calibration's result: its sentence names how many parameters pass the rank test, whether the joint coverage
    test passes, its sizes and the seed it was drawn with; its table holds a row for each parameter with its unit, its
    p-value, its result and the window's remark for an assumed input and for the time scale, then "All together" and the
    reading of what it learned; the footnote says the rank test's threshold from the record's own count of parameters
    and level, never a fixed number; and its three pictures are captioned in the window's words. A failed one names the
    parameters that fail, and one made in a narrowed region says so."""
    from core.artifacts import ArtifactStore
    from core.gui.panels.inference.check_results import FIGURE_CAPTIONS, result_block
    from core.artifacts.store import slug
    from PySide6.QtWidgets import QWidget
    from tests._fixtures import qt_app, write_calibration
    qt_app()
    host = QWidget()                                 # what the calibration's page is to its result
    store = ArtifactStore(tmp_path / "Artifacts")
    names = ["k", "lam", "f_max", "tau", "tau_c", "s", "a", "b", "c", "d", "x_scale", "t_scale", "T"]
    ident = write_calibration(store, None, posterior=POSTERIOR, prior=PRIOR,
                              results=_calibration_of(names, ks=[0.5] * 13, assumed=["T"]),
                              figures=("SBC ranks (CDF)", "SBC ranks (histogram)", "TARP coverage"))
    folder = store.path("calibration", ident)
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    region = result_block("calibration", manifest, folder, parent=host,
                          units={"x_scale": "nm", "t_scale": "ms", "T": "K"})
    try:
        assert region.verdict.sentence.plain_text() == (
            "Calibration: passed. All 13 parameters pass the rank test, and the joint coverage test passes. Made on "
            "2,000 test datasets at 200 operating points (pairs of time scale and recording length), seed 1840273.")
        table = region.table_words()
        assert table.columns == ("Parameter", "Unit", "p-value", "Result", "Remark") and table.right == {2}
        assert [row[0] for row in table.rows] == [*names, "All together", "A reading, not a test", "Information "
                                                  "gained", "Its standard error"]
        assert table.rows[0] == ("k", "", "0.5", "Passes", "")
        assert table.rows[names.index("t_scale")] == (
            "t_scale", "ms", "0.5", "Passes", "Rests on the operating points, not the datasets, so it has less power")
        assert table.rows[names.index("T")] == ("T", "K", "0.5", "Passes", "An assumed input")
        assert table.rows[13] == ("All together", "", "0.5", "Passes", "The joint coverage test; it passes at p = 0.05 "
                                                                      "or more")
        assert table.rows[15:] == (("Information gained", "nats", "2.31", "", ""),
                                   ("Its standard error", "nats", "0.05", "", "")) and 14 in table.headings
        assert table.footnote == "Rank test: a parameter passes at p = 0.0038 or more (0.05 divided by 13)."
        assert [caption.text() for caption, _picture in region.pictures] == [
            "Rank test, cumulative", "Rank test, histogram", "Joint coverage test"]
        assert set(FIGURE_CAPTIONS) == {slug("SBC ranks (CDF)"), slug("SBC ranks (histogram)"), slug("TARP coverage")}
    finally:
        region.deleteLater()
    twelve = write_calibration(store, None, posterior=POSTERIOR, prior=PRIOR, created=_at(13, 9, 0),
                               results=_calibration_of(names[:12], ks=[0.5] * 10 + [0.001, 0.002], tarp=0.01,
                                                       accepted=["truncated"]),
                               figures=("SBC ranks (CDF)", "another figure"))
    folder = store.path("calibration", twelve)
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    region = result_block("calibration", manifest, folder, parent=host)
    try:
        assert region.verdict.sentence.plain_text() == (
            "Calibration: failed. 2 of 12 parameters fail the rank test: x_scale, t_scale; the joint coverage test "
            "fails. Made on 2,000 test datasets at 200 operating points (pairs of time scale and recording length), "
            "seed 1840273. It was made inside the region its posterior was narrowed to.")
        assert region.verdict.kind == "err"
        assert region.table_words().footnote == ("Rank test: a parameter passes at p = 0.0042 or more (0.05 divided "
                                                 "by 12).")
        assert [caption.text() for caption, _picture in region.pictures] == ["Rank test, cumulative",
                                                                             "Calibration, figure 2"]
    finally:
        host.deleteLater()


def test_a_checks_result_is_made_at_the_pages_reading_and_never_as_it_draws(tmp_path, monkeypatch):
    """A check's result, figures and all, is made at the page's own reading of the disk, never as the page draws: a
    calibration written while the page is out of sight and read by the screen shows no result and reads no figure when
    the page is drawn; the page's reading makes it, its figures read; drawn again, or the check opened, it is the same
    result, kept, and no figure is read again."""
    from tests._fixtures import readings_of_a_run, write_calibration
    with _checks_page(tmp_path) as (window, screen, page, store):
        window.go_to("inference", "prior")
        _settle()
        write_calibration(store, screen.session.cfg, posterior=screen.session.posterior.id,
                          prior=screen.session.inf_prior.id,
                          figures=("SBC ranks (CDF)", "SBC ranks (histogram)", "TARP coverage"))
        screen.read_checks()
        calls = readings_of_a_run(monkeypatch)
        page.refresh_local_gates()
        one = page.check_pages["calibration"]
        assert _row(screen.check_rows(), "calibration").state == "Passed"
        assert one.result is None and "a record's figures" not in calls, calls
        window.go_to("inference", "checks")
        _settle()
        page.read_facts()
        assert one.result is not None and len(one.result.pictures) == 3, "the page's reading made no result"
        made, read = one.result, calls.count("a record's figures")
        assert read >= 1, calls
        page.refresh_local_gates()
        page.open_check("calibration")
        page.redraw()
        assert one.result is made and calls.count("a record's figures") == read, calls


def test_no_result_is_ever_a_window_of_its_own(tmp_path, monkeypatch):
    """STANDING. Every result a reading makes, of each check that has one, a record's, the comparison's and a run's from
    the history, is made inside its check's page, and each check's form stands inside its page before it is shown:
    none is a window of its own at any moment it is shown, as a widget made with no parent is, shown on the screen as a
    window as its first words are said. A result that is a verdict alone, the comparison's and a run's, stands in sight
    on its page."""
    from core.gui.run import history
    from core.gui.widgets.form import Form
    from core.gui.widgets.result_region import ResultRegion
    from tests._fixtures import write_simulation_set
    shown, forms = [], []
    real, real_form = ResultRegion.setVisible, Form.setVisible

    def recorded(region, visible):
        if visible:
            shown.append(region.isWindow())
        real(region, visible)

    def recorded_form(form, visible):
        if visible:
            forms.append(form.isWindow())
        real_form(form, visible)

    monkeypatch.setattr(ResultRegion, "setVisible", recorded)
    monkeypatch.setattr(Form, "setVisible", recorded_form)
    with _checks_page(tmp_path, records=_every_record) as (_window, screen, page, store):
        folder = write_simulation_set(store.kind_dir("simulation"), {"a set of": "the comparison"}, 100,
                                      complete=True, chi_masked=[(84, 221)] * 100)
        screen.session.posterior = _held_posterior(tmp_path, prior=screen.session.inf_prior, simulation=folder.name)
        for at, key in enumerate(("timing", "card_test")):
            history.write(tmp_path / "runs", {"id": f"2026100812000{at}-check", "word": "Check", "name": "",
                                              "place": ["inference", "checks"], "records_root": str(store.root),
                                              "started": _at(8, 12, at), "ended": _at(8, 12, at + 1),
                                              "outcome": "finished", "check": {"key": key}})
        page.read_facts()
        made = sorted(key for key, one in page.check_pages.items() if one.result is not None)
        decided = [row.key for row in screen.check_rows() if row.record is not None]
        assert len(decided) >= 6 and made == sorted([*decided, "comparison", "timing", "card_test"]), made
        assert shown and not any(shown), f"{shown.count(True)} results were shown as windows of their own"
        assert forms and not any(forms), f"{forms.count(True)} forms were shown as windows of their own"
        for key in ("comparison", "timing", "card_test"):
            page.open_check(key)
            _settle()
            one = page.check_pages[key]
            assert one.result.verdict is not None and one.result.isVisibleTo(page), key


def test_a_kept_result_is_made_again_when_its_words_would_change(tmp_path, monkeypatch):
    """A result is kept while it would say the same, and made again at the next reading when what its words say beside
    its record moved: the calibration's count of the other deciding checks still open, after one of them passed, its
    figures read again then and only then."""
    from tests._fixtures import check_results, readings_of_a_run, write_calibration, write_diagnostic

    def records(store, cfg, prior, posterior):
        write_calibration(store, cfg, posterior=posterior, prior=prior, figures=("SBC ranks (CDF)",))
        write_diagnostic(store, cfg, diagnostic="probes", variant="band", cell=str(_cell_file()),
                         **check_results("band"))
        write_diagnostic(store, cfg, diagnostic="probes", variant="mask", parents={"prior": prior},
                         **check_results("audit"))

    alone = "Calibration alone does not show that the posterior is sound: {}"
    with _checks_page(tmp_path, chi=False, records=records, simulation=SET) as (_window, screen, page, store):
        one = page.check_pages["calibration"]
        assert one.result.verdict.sentence.plain_text().endswith(alone.format("2 deciding checks are still open."))
        calls = readings_of_a_run(monkeypatch)
        page.read_facts()
        assert calls.count("a record's figures") == 0, "a result that would say the same was made again"
        posterior, prior = screen.session.posterior.id, screen.session.inf_prior.id
        entry = check_catalogue.entry("identifiability")
        write_diagnostic(store, screen.session.cfg, diagnostic=entry.diagnostic, variant=entry.variant,
                         parents={"posterior": posterior, "prior": prior}, **check_results("identifiability"))
        page.read_facts()
        assert one.result.verdict.sentence.plain_text().endswith(alone.format("1 deciding check is still open.")), (
            one.result.verdict.sentence.plain_text())
        assert calls.count("a record's figures") == 1, calls


def test_a_kept_result_is_made_again_when_the_units_in_use_change(tmp_path):
    """A record's result says each parameter's unit and the cell's unit of force as the configuration in use names
    them, so it is kept only while those stay: once the units in use change, the next reading makes the result again in
    the new ones, and one that would say the same is kept."""
    from tests._fixtures import check_results, write_calibration, write_diagnostic

    def records(store, cfg, prior, posterior):
        write_calibration(store, cfg, posterior=posterior, prior=prior,
                          results=_calibration_of(["k", "x_scale", "t_scale"], ks=[0.5] * 3))
        write_diagnostic(store, cfg, diagnostic="probes", variant="drive", cell=str(_cell_file()),
                         **check_results("drive_study"))

    with _checks_page(tmp_path, records=records) as (_window, screen, page, _store):
        calibration, drive = page.check_pages["calibration"], page.check_pages["drive_study"]
        made = (calibration.result, drive.result)
        assert [row[:2] for row in calibration.result.table_words().rows[:2]] == [("k", ""), ("x_scale", "nm")]
        said = drive.result.verdict.sentence.plain_text()
        assert screen.session.cfg.force_unit in said, said
        screen.session.cfg.length_unit, screen.session.cfg.force_unit = "um", "nN"
        page.read_facts()
        assert calibration.result is not made[0] and drive.result is not made[1], "a result kept in the old units"
        assert [row[:2] for row in calibration.result.table_words().rows[:2]] == [("k", ""), ("x_scale", "um")]
        assert "nN" in drive.result.verdict.sentence.plain_text(), drive.result.verdict.sentence.plain_text()
        made = (calibration.result, drive.result)
        page.read_facts()
        assert (calibration.result, drive.result) == made, "a result that would say the same was made again"


def test_the_comparison_is_read_afterwards_from_the_sets_counts_or_the_log(tmp_path, monkeypatch):
    """With no run live, the comparison after simulating is worked out afresh from the posterior's simulation set: the
    counts its saved state holds for each batch, summed while there is one pair for each batch done; else, with the set
    gone, the first total of unusable driven recordings in the posterior's own text. Its page says the two shares and
    their difference; for a posterior from a narrowing round, that the audit measured the whole prior. While a run is
    live nothing of the set is read: the page shows what it read last, and while a training is live, that the
    comparison is counted when its simulation ends."""
    from core.artifacts import default_store
    from tests._fixtures import check_results, readings_of_a_run, write_diagnostic, write_simulation_set
    from tests._look_helpers import _settle
    from tests._runs import StandIn, masked_line, start_run, training_spec, turn_until

    def records(store, cfg, prior, posterior):
        write_diagnostic(store, cfg, diagnostic="probes", variant="mask", parents={"prior": prior},
                         **check_results("audit"))

    app = qt_app()
    with _checks_page(tmp_path, records=records) as (window, screen, page, store):
        folder = write_simulation_set(store.kind_dir("simulation"), {"a set of": "the comparison"}, 100,
                                      complete=True, chi_masked=[(84, 221)] * 100)
        prior = screen.session.inf_prior
        screen.session.posterior = _held_posterior(tmp_path, prior=prior, simulation=folder.name, text=(
            f"10:00:00 info {masked_line(8_400, 20_000, 100, WHOLE)}\n"))
        page.read_facts()
        page.open_check("comparison")
        one = page.check_pages["comparison"]
        assert _row(screen.check_rows(), "comparison").state == "Passed"
        assert one.result.verdict.sentence.plain_text() == (
            "Unusable driven recordings, simulation set against the audit: passed. 38.0 % in the simulation set; the "
            "audit read 37.9 %. The difference is 0.1 points: within 3 is fine, up to 5 look closer, beyond 5 stop.")
        assert one.result.sentences() == (), "a posterior trained on the whole prior was said to be narrowed"
        import shutil
        shutil.rmtree(folder)
        page.read_facts()
        assert _row(screen.check_rows(), "comparison").state == "Look closer"
        assert one.result.verdict.sentence.plain_text().startswith(
            "Unusable driven recordings, simulation set against the audit: look closer. 42.0 % in the simulation set; "
            "the audit read 37.9 %. The difference is 4.1 points"), "the posterior's text was not read with the set gone"
        assert _row(screen.check_rows(), "comparison").state == "Look closer"
        screen.session.posterior = _held_posterior(tmp_path, prior=prior, ident=OTHER_POSTERIOR, narrowed=True,
                                                   text=f"10:00:00 info {masked_line(8_400, 22_144, 100, WHOLE)}\n")
        page.read_facts()
        assert [fact.sentence for fact in one.result.sentences()] == [
            "The audit measured the whole prior; this round simulated inside a narrowed region."]
        calls = readings_of_a_run(monkeypatch)
        stand_in = StandIn(word="Other run")
        start_run(window.part("inference").tsnpe_panel, stand_in)
        shown = one.result
        page.read_facts()
        page.stale_store()
        screen.read_checks()
        page.refresh_local_gates()
        _settle()
        assert calls == [] and one.result is shown, f"read while a run is live: {calls}"
        stand_in.release()
        turn_until(app, lambda: not controller().is_live())
        _settle()
        calls.clear()                        # what the run's end read, as the end of every run reads
        training = StandIn(word="Training")
        start_run(window.part("inference").tsnpe_panel, training, spec=training_spec(place=("inference", "narrowing")))
        page.refresh_local_gates()
        _settle()
        assert one.result.verdict.sentence.plain_text() == "Counted when simulation ends.", (
            "the comparison was worked out while a training is live")
        assert calls == [], f"read while a training is live: {calls}"
        training.release()
        turn_until(app, lambda: not controller().is_live())


def test_check_3_is_counted_later_only_where_it_applies(tmp_path):
    """While a training is live, the comparison after simulating says on its page it is counted when the simulation
    ends only where the check applies: with the frequency response off it does not apply and its page says nothing
    more."""
    from tests._runs import StandIn, start_run, training_spec, turn_until
    app = qt_app()
    with _checks_page(tmp_path, chi=False) as (window, _screen, page, _store):
        training = StandIn(word="Training")
        start_run(window.part("inference").tsnpe_panel, training, spec=training_spec(place=("inference", "narrowing")))
        page.refresh_local_gates()
        _settle()
        one = page.check_pages["comparison"]
        assert one.state_line.sentence.plain_text().startswith("Does not apply.")
        assert one.result is None, one.result.verdict.sentence.plain_text()
        training.release()
        turn_until(app, lambda: not controller().is_live())


def test_check_3_waiting_for_the_audit_says_which_check_to_run(tmp_path):
    """With the posterior's set counted and no audit of its prior that counts, the comparison after simulating says on
    its page, with no run live, that there is no audit to compare with, the set's share, and which check of the list to
    run; not the run's own words for after its end."""
    from tests._fixtures import write_simulation_set
    with _checks_page(tmp_path) as (_window, screen, page, store):
        folder = write_simulation_set(store.kind_dir("simulation"), {"a set of": "the comparison"}, 100,
                                      complete=True, chi_masked=[(84, 221)] * 100)
        screen.session.posterior = _held_posterior(tmp_path, prior=screen.session.inf_prior, simulation=folder.name)
        page.read_facts()
        assert _row(screen.check_rows(), "comparison").state == "Waiting for the audit"
        one = page.check_pages["comparison"]
        assert one.result.verdict.sentence.plain_text() == (
            "There is no audit of the prior retrain_prior to compare with. The simulation set holds 38.0 % unusable "
            "driven recordings; run \"Unusable driven recordings: audit\" in this list to compare them.")


def test_check_3_is_counted_later_during_a_narrowing_round(tmp_path):
    """While a narrowing round is live, the comparison after simulating says on its page it is counted when the round's
    simulation ends, as during a training, and nothing is worked out from the counts read before."""
    from tests._fixtures import check_results, write_diagnostic, write_simulation_set
    from tests._runs import StandIn, start_run, training_spec, turn_until

    def records(store, cfg, prior, posterior):
        write_diagnostic(store, cfg, diagnostic="probes", variant="mask", parents={"prior": prior},
                         **check_results("audit"))

    app = qt_app()
    with _checks_page(tmp_path, records=records) as (window, screen, page, store):
        folder = write_simulation_set(store.kind_dir("simulation"), {"a set of": "the comparison"}, 100,
                                      complete=True, chi_masked=[(84, 221)] * 100)
        screen.session.posterior = _held_posterior(tmp_path, prior=screen.session.inf_prior, simulation=folder.name)
        page.read_facts()
        one = page.check_pages["comparison"]
        assert one.result.verdict.sentence.plain_text().startswith(
            "Unusable driven recordings, simulation set against the audit: passed.")
        round_ = StandIn(word="Narrowing round")
        start_run(window.part("inference").tsnpe_panel, round_, spec=training_spec("Narrowing round", place=("inference", "narrowing")))
        page.refresh_local_gates()
        _settle()
        assert one.result.verdict.sentence.plain_text() == "Counted when simulation ends."
        round_.release()
        turn_until(app, lambda: not controller().is_live())


def _heading_labels(widget) -> list:
    """Every label in ``widget`` set as a heading: by the heading's class, or at its size or larger."""
    from PySide6.QtWidgets import QLabel
    from core.gui import fonts
    heading = fonts.inter("head")
    big = heading.pixelSize() if heading.pixelSize() > 0 else heading.pointSizeF()
    return [label for label in widget.findChildren(QLabel)
            if label.property("cls") == "head" or (_label_size(label) >= big and label.font().weight() >= heading.weight())]


def _every_record(store, cfg, prior, posterior) -> None:
    """A passing record of each check that writes one, made for what is in use."""
    from tests._fixtures import check_results, write_calibration, write_diagnostic
    write_calibration(store, cfg, posterior=posterior, prior=prior,
                      figures=("SBC ranks (CDF)", "SBC ranks (histogram)", "TARP coverage"))
    for one in check_catalogue.CHECKS:
        if one.kind != "diagnostic":
            continue
        parents = ({"prior": prior} if one.key == "audit" else {"posterior": posterior, "prior": prior}
                   if one.key in ("input_use", "identifiability", "fixed_number", "across_prior") else {})
        write_diagnostic(store, cfg, diagnostic=one.diagnostic,
                         variant="pooled" if one.key == "fixed_number" else one.variant, parents=parents,
                         cell=str(_cell_file()) if one.key in ("band", "pin_down", "drive_study") else None,
                         **check_results(one.key))


def test_a_checks_verdict_is_a_message_row_never_a_heading(tmp_path):
    """STANDING. No check's verdict is a heading, boxed or coloured: with a passing record of every check, no label of
    the page set as a heading holds "PASS" or "Passed", on the list or on any check's page; each verdict is the first
    message row of its result, its sentence beginning with the check's name."""
    from core.gui.widgets.message_row import MessageRow
    with _checks_page(tmp_path, records=_every_record) as (_window, _screen, page, _store):
        for key in [None, *(one.key for one in check_catalogue.CHECKS)]:
            page.open_check(key)
            loud = [label.text() for label in _heading_labels(page) if label.isVisibleTo(page)
                    and ("PASS" in label.text() or "Passed" in label.text())]
            assert loud == [], (key, loud)
        for key, one in page.check_pages.items():
            if one.result is None:
                continue
            assert isinstance(one.result.verdict, MessageRow), key
            assert one.result.verdict.sentence.plain_text().startswith(check_readings.said_name(key)), key


def test_no_sentence_says_a_posterior_passed_below_every_check_that_applies():
    """STANDING. Over every combination of the deciding states, a check no run can pass among them: the headline's
    sentence says the posterior passed only when every deciding check that applies passed, and nothing below that
    says it, a check no run can pass included."""
    seen = 0
    for states in _never_cases():
        rows = _with_never(states)
        held = _held()
        _head, sentence = check_lineage.headline(rows, held=held)
        every = all(row.state in ("Passed", "Does not apply") for row in rows if row.key in check_catalogue.DECIDING)
        assert ("passed every deciding check" in sentence) == every, (states, sentence)
        seen += 1
    assert seen == 4 * 4 * 5 * 4 * 5 * 5
    for states, posterior, chi in _cases():
        rows = _rows_of(states)
        held = _held(posterior=POSTERIOR if posterior else None, chi=chi)
        _head, sentence = check_lineage.headline(rows, held=held)
        every = posterior and all(state in ("Passed", "Does not apply") for state in states)
        assert ("passed every deciding check" in sentence) == every, (states, posterior, chi, sentence)


@pytest.mark.parametrize("others", [0, 1, 2, "one no run can pass"])
def test_a_passed_calibration_with_others_open_says_it_alone_is_not_enough(others, tmp_path):
    """Under a passed calibration, while other deciding checks that apply are still open, its result says the
    calibration alone does not show the posterior is sound, counting them; with none open it says nothing of it; a
    check no run can pass for the posterior, its input use when it was trained without keeping its simulations, is not
    counted among them; a failed calibration never says it."""
    never = others == "one no run can pass"
    written = 0 if never else 2 - others
    from tests._fixtures import check_results, write_calibration, write_diagnostic

    def records(store, cfg, prior, posterior):
        write_calibration(store, cfg, posterior=posterior, prior=prior)
        write_diagnostic(store, cfg, diagnostic="probes", variant="band", cell=str(_cell_file()),
                         **check_results("band"))
        write_diagnostic(store, cfg, diagnostic="probes", variant="mask", parents={"prior": prior},
                         **check_results("audit"))
        for key in ("input_use", "identifiability")[:written]:
            one = check_catalogue.entry(key)
            write_diagnostic(store, cfg, diagnostic=one.diagnostic, variant=one.variant,
                             parents={"posterior": posterior, "prior": prior}, **check_results(key))

    with _checks_page(tmp_path, chi=False, records=records, simulation=None if never else SET) as (
            _window, screen, page, _store):
        assert [_row(screen.check_rows(), key).state for key in check_catalogue.DECIDING[3:]].count("Passed") == (
            1 if never else 3 - others)
        assert check_lineage.no_run_passes(_row(screen.check_rows(), "input_use"), held=screen.checks_in_use()) is never
        said = page.check_pages["calibration"].result.verdict.sentence.plain_text()
        alone = {0: "", 1: " Calibration alone does not show that the posterior is sound: 1 deciding check is still "
                           "open.",
                 2: " Calibration alone does not show that the posterior is sound: 2 deciding checks are still open."}
        assert said == ("Calibration: passed. All 2 parameters pass the rank test, and the joint coverage test passes. "
                        "Made on 2,000 test datasets at 200 operating points (pairs of time scale and recording "
                        "length), seed 1840273." + alone[1 if never else others])
    failed = check_readings.verdict_sentence("calibration", {"body": check_results("calibration", passed=False)},
                                             others_open=2)
    assert "alone does not show" not in failed


def test_no_passed_or_failed_for_an_informing_check_a_tool_or_a_reduced_record(tmp_path):
    """STANDING. An informing check or a tool reads "Done" once run, and its result says "done", never "Passed",
    "Failed", "passed" or "failed", whatever its record holds; a record made at a reduced size is no pass nor failure:
    its row reads "Not run yet" with why, its page shows no result for it, and its sentence says it does not count."""
    from tests._fixtures import check_results, write_calibration

    def records(store, cfg, prior, posterior):
        _every_record(store, cfg, prior, posterior)
        write_calibration(store, cfg, posterior=posterior, prior=prior, created=_at(30, 9, 0),
                          results=check_results("calibration", n_cal=40, passed=False)["results"])

    informing = [one.key for one in check_catalogue.CHECKS if one.role != "Decides"]
    with _checks_page(tmp_path, chi=False, records=records) as (_window, screen, page, _store):
        rows = {row.key: row for row in screen.check_rows()}
        for key in informing:
            assert rows[key].state in ("Done", "Not run yet", "Does not apply"), (key, rows[key])
            one = page.check_pages[key]
            if one.result is not None:
                said = one.result.verdict.sentence.plain_text()
                assert ": done" in said and not {"passed", "failed", "Passed", "Failed"} & set(
                    said.replace(".", " ").replace(",", " ").split()), (key, said)
        assert rows["calibration"].state == "Passed", "the older record that counts no longer decided"
    with _checks_page(tmp_path / "reduced", records=lambda store, cfg, prior, posterior: write_calibration(
            store, cfg, posterior=posterior, prior=prior,
            results=check_results("calibration", n_cal=40, passed=False)["results"])) as (_window, screen, page, _s):
        row = _row(screen.check_rows(), "calibration")
        assert (row.state, row.record) == ("Not run yet", None)
        assert row.line == ("Not run yet. Run only at a reduced size (40 test datasets; the usual is 2,000), which "
                            "does not count.")
        assert page.check_pages["calibration"].result is None, "a reduced record's result was shown"
        assert page.check_pages["calibration"].earlier.text().endswith("; does not count")
    reduced = check_readings.verdict_sentence("calibration", {"body": check_results("calibration", n_cal=40)})
    assert reduced == ("Calibration: not run yet. Run only at a reduced size (40 test datasets; the usual is 2,000), "
                       "which does not count.")


def test_the_engines_verdict_line_is_shown_only_in_the_full_text(tmp_path, monkeypatch):
    """STANDING. The engine's verdict line of a calibration, "[verdict] PASS: ...", is never one of the window's words:
    while the calibration runs and after it, no label of the window and no message of the run's page holds "PASS",
    and the run's full text holds the line as the engine wrote it."""
    from PySide6.QtWidgets import QLabel
    from core import orchestrator
    from tests._runs import RunText, StandIn, say, turn_until
    verdict = ("[verdict] PASS: every parameter's rank test at KS p >= 0.05/13 = 0.003846 and the joint coverage test "
               "at KS p >= 0.05 (calibration seed 1840273)")
    stand_in = StandIn(word="Check")
    monkeypatch.setattr(orchestrator, "validate_calibration", stand_in)
    app = qt_app()

    def shown(window, page) -> list:
        """Every label of the window in sight that holds "PASS", and every line of the run's page but its full text's."""
        labels = [label.text() for label in window.findChildren(QLabel)
                  if label.isVisible() and "PASS" in label.text()
                  and not page.run_view.full_text.isAncestorOf(label)]
        return labels

    with _checks_page(tmp_path) as (window, _screen, page, _store):
        text = RunText(controller())
        page.open_check("calibration")
        page._main_clicked()
        assert stand_in.started.wait(5)
        say(app, stand_in, text, verdict)
        _settle()
        assert page.results.currentWidget() is page.run_view
        assert shown(window, page) == [], shown(window, page)
        assert any(verdict in line for line in page.run_view.full_text.lines()), "the full text lost the line"
        stand_in.release()
        turn_until(app, lambda: not controller().is_live())
        _settle()
        assert shown(window, page) == [], shown(window, page)


def test_a_record_that_moves_or_cannot_be_read_is_said_and_raises_nothing(tmp_path):
    """A record that moves under the page, or cannot be read, is read again when Records changes the store, said in
    words, and raises nothing, the drawing guard silent: a calibration renamed still counts, by its id; a torn manifest
    is said on the list as a record that counts for no check; a calibration whose manifest holds no results, and one of
    an older shape, read "Not run yet" saying their record cannot be read; the posterior in use whose record went from
    under it is checked with what is read, its counts not; and with no posterior every row waits for one."""
    from core.gui import reported
    from tests._fixtures import write_calibration
    kept = {}

    def records(store, cfg, prior, posterior):
        kept["calibration"] = write_calibration(store, cfg, posterior=posterior, prior=prior, created=_at(12, 14, 5),
                                                name="first_try")

    with _checks_page(tmp_path, records=records) as (window, screen, page, store):
        reported.forget()
        store.rename("calibration", kept["calibration"], "renamed_since")
        window._store_changed()
        assert _row(screen.check_rows(), "calibration").state == "Passed", "a renamed record no longer counted"
        torn = write_calibration(store, screen.session.cfg, posterior=POSTERIOR, prior=PRIOR, created=_at(13, 9, 0))
        torn_folder = store.path("calibration", torn)
        (torn_folder / "manifest.json").write_bytes(b'{"kind": "calibration", "id": ')
        window._store_changed()
        said = [fact.sentence for fact in page.facts_from_form()]
        assert f"The record {torn_folder.name} cannot be read, so it counts for no check." in said, said
        assert _row(screen.check_rows(), "calibration").state == "Passed"
        for minute, shape in ((0, {"results": {"n_cal": 2000, "cal_n_scales": 200, "num_posterior_samples": 1000}}),
                              (1, {"verdict": "PASS", "seed": 1})):
            ident = write_calibration(store, screen.session.cfg, posterior=screen.session.posterior.id,
                                      prior=screen.session.inf_prior.id, created=_at(14, 9, minute))
            folder = store.path("calibration", ident)
            manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
            manifest["body"] = shape
            (folder / "manifest.json").write_bytes(json.dumps(manifest).encode("utf-8"))
            window._store_changed()
            row = _row(screen.check_rows(), "calibration")
            assert row.state == "Passed", "a record that cannot be read decided"
            assert f"Its record cannot be read, {_said(_at(14, 9, minute))}; does not count" in (
                page.check_pages["calibration"].earlier.text().splitlines())
        posterior = screen.session.posterior
        import shutil
        shutil.rmtree(posterior.path)
        window._store_changed()
        assert _row(screen.check_rows(), "comparison").line == (
            "Not compared: the counts of unusable driven recordings could not be read.")
        screen.session.posterior = None
        screen.refresh_gates()
        window._store_changed()
        assert {row.state for row in screen.check_rows() if row.key in ("calibration", "input_use",
                                                                         "identifiability")} == {"Waiting for a posterior"}
        page.open_check("calibration")
        page.redraw()
        assert reported.ERRORS == [] and reported.COUNTS == {}, reported.ERRORS


def test_the_calibrations_stop_asks_with_keep_it_running_preselected(tmp_path, monkeypatch):
    """The calibration started from its page is a check's run named for it: its stop button reads "Stop the
    calibration…", and pressed it asks before it stops anything, "Keep it running" preselected, the other answer drawn
    as losing work, saying a check writes its record only when it finishes; kept running, nothing stops. The figure it
    draws stands on its run's page alone: once it is over the page shows no figure area of its own."""
    from core import orchestrator
    from tests._dialogs import asked
    from tests._runs import StandIn, turn_until
    stand_in = StandIn(word="Check")
    monkeypatch.setattr(orchestrator, "validate_calibration", stand_in)
    app = qt_app()
    with _checks_page(tmp_path) as (_window, _screen, page, _store):
        page.open_check("calibration")
        page._main_clicked()
        assert stand_in.started.wait(5)
        _settle()
        stop = page.run_view.stop_button
        assert stop.text().replace("&", "") == "Stop the calibration…"
        stop.click()
        _settle()
        question = asked()
        assert (question.title, question.sentence, question.safe, question.loses_work) == (
            "Stop the calibration?", "A check writes its record only when it finishes, so nothing of this one is kept.",
            "Keep it running", True)
        assert question.answers == ("Stop and discard it", "Keep it running") and question.default == "Keep it running"
        assert controller().is_live() and stand_in.stopped_at is None
        stand_in.figure("SBC ranks (CDF)")
        turn_until(app, lambda: page.figure_stack.count() == 1)
        stand_in.release()
        turn_until(app, lambda: not controller().is_live())
        _settle()
        assert page.scroller.isAncestorOf(page.figure_stack) and page.figure_stack.count() == 1
        assert not page.figure_stack.isVisibleTo(page), "the run's figure stood on the page after its run"


def test_the_checks_page_writes_nothing_to_the_settings_file(tmp_path, monkeypatch):
    """STANDING. The Checks page remembers nothing: shown, read, a check opened, its form edited, a check run and the
    window closed, the keys written are exactly those a session that never opens the Checks page writes, whatever their
    group: no key of its own, and nothing of a check's form, the calibration's sizes included."""
    from tests._window_helpers import shown_off_screen
    from PySide6.QtCore import QSettings
    from core import orchestrator
    from tests._runs import StandIn, turn_until
    written = []
    real = QSettings.setValue
    monkeypatch.setattr(QSettings, "setValue", lambda held, key, value: (written.append(key), real(held, key, value)))
    stand_in = StandIn(word="Check")
    monkeypatch.setattr(orchestrator, "validate_calibration", stand_in)
    app = qt_app()
    with _checks_page(tmp_path / "without", shown=False) as (window, _screen, _page, _store):
        shown_off_screen(window, 1300, 820)
        window.go_to("inference", "prior")
        _settle()
        window._save_state()
    without, written[:] = set(written), []
    with _checks_page(tmp_path / "with") as (window, _screen, page, _store):
        page.open_check("calibration")
        page.form.rows["n_cal"].widget.setText("40")
        page.open_check("band")
        page.open_check("calibration")
        page._main_clicked()
        assert stand_in.started.wait(5)
        stand_in.release()
        turn_until(app, lambda: not controller().is_live())
        window._save_state()
    assert without and set(written) == without, (sorted(set(written) - without), sorted(without - set(written)))


def test_a_checks_form_is_locked_while_a_run_is_live_whichever_is_in_sight(tmp_path):
    """STANDING. While a run is live every check's form is locked, the one in sight and every other, a check opened
    during the run shows its form locked, and the table can still be read and a check opened; when the run ends every
    form is unlocked, the one in sight at its start included."""
    from tests._runs import StandIn, start_run, turn_until
    app = qt_app()
    with _checks_page(tmp_path) as (window, _screen, page, _store):
        page.open_check("calibration")
        stand_in = StandIn(word="Other run")
        start_run(window.part("inference").tsnpe_panel, stand_in)
        _settle()
        assert not any(one.form.isEnabled() for one in page.check_pages.values())
        assert page.listing.isEnabled() and page.list_detail.isEnabled()
        page.open_check(None)
        page.open_check("audit")
        page.open_check("calibration")
        assert not page.form.isEnabled() and not page.check_pages["calibration"].form.rows["n_cal"].widget.isEnabled()
        stand_in.release()
        turn_until(app, lambda: not controller().is_live())
        _settle()
        assert all(one.form.isEnabled() for one in page.check_pages.values())
        assert page.check_pages["calibration"].form.rows["n_cal"].widget.isEnabled()


def test_the_checks_page_says_the_earlier_steps_changes_not_applied(tmp_path):
    """A choice of step 2 not applied is said on the list's facts and in the calibration's "Before you start" as a
    warning naming what the run uses, and the calibration's button stays live: a check reads what the earlier steps
    hold."""
    with _checks_page(tmp_path) as (_window, screen, page, _store):
        assert [fact for fact in page.facts_from_form() if fact.kind == "warn"] == []
        prior_page = screen.prior_panel
        prior_page.bounds_picker.restore_key("master_tier1.txt")
        prior_page._edited()
        assert "prior" in screen.pending()
        sentence = "Step 2, Prior, has changes that are not applied. This run uses what is in use: retrain_prior."
        assert [(fact.kind, fact.sentence) for fact in page.facts_from_form()] == [("warn", sentence)]
        page.open_check("calibration")
        page.redraw()
        assert ("warn", sentence) in [(fact.kind, fact.sentence) for fact in page.facts_from_form()]
        assert page.foot.main_button.isEnabled() and page.foot.reason.text() == ""


@pytest.mark.parametrize("in_use", ["HOPF", "NADROWSKI"])
def test_the_checks_read_the_records_of_the_model_in_use_alone(in_use, tmp_path):
    """The frequency band's and the audit's records count for the model they were made on alone: Nadrowski's records
    count nothing while another model's configuration and prior are in use, and the other model's count nothing while
    Nadrowski's are; each row then reads as it reads with no record."""
    from core.artifacts import ArtifactStore
    from tests._fixtures import _nad_cfg, check_results, write_diagnostic
    other = ArtifactStore(tmp_path / "elsewhere")

    def records(store, cfg, prior, posterior):
        made_on = _nad_cfg(chi_mode=True)
        if in_use == "NADROWSKI":
            made_on.model = "HOPF"
            prior = "20261001T000001"
        else:
            cfg.model = "HOPF"
        write_diagnostic(store, made_on, diagnostic="probes", variant="band", cell=str(_cell_file()),
                         **check_results("band"))
        write_diagnostic(store, made_on, diagnostic="probes", variant="mask",
                         parents={"prior": prior if in_use == "NADROWSKI" else "20261001T000002"},
                         **check_results("audit"))

    with _checks_page(tmp_path, records=records, posterior=False) as (_window, screen, page, _store):
        rows = {row.key: row for row in screen.check_rows()}
        assert (rows["band"].state, rows["band"].line, rows["audit"].state, rows["audit"].line) == (
            "Not run yet", "Not run yet.", "Not run yet", "Not run yet."), (rows["band"], rows["audit"])
        assert screen.checks_summary().passed == 0
    del other


def test_a_checks_run_hands_its_plan_and_no_settings(tmp_path, monkeypatch):
    """The calibration's run is handed the calibration's call, the configuration, the posterior and the prior in use
    and the form's sizes with a seed drawn afresh at each press; its run is a check named for the calibration, writing
    one calibration record, and keeps the plan in use and the check's key and sizes, and no settings."""
    import random
    from core import orchestrator
    from core.gui import plans
    from core.gui.run.controller import RunController
    seen = {}
    monkeypatch.setattr(RunController, "dispatch", lambda held, fn, *args, **kwargs: seen.update(
        fn=fn, args=args, kwargs=kwargs) or True)
    drawn = iter((1840273, 77))
    monkeypatch.setattr(random, "randrange", lambda top: next(drawn) if top == 2 ** 31 else 0)
    with _checks_page(tmp_path) as (_window, screen, page, _store):
        screen.plan_source = plans.PlanSource("record", "retrain_prior", None, "", None, {}, ident=PRIOR)
        page.open_check("calibration")
        page.form.rows["n_cal"].widget.setText("2500")
        page._main_clicked()
        spec = seen["kwargs"]["spec"]
        assert seen["fn"] is orchestrator.validate_calibration
        assert seen["args"] == (screen.session.cfg, screen.session.posterior, screen.session.inf_prior)
        assert (seen["kwargs"]["n_cal"], seen["kwargs"]["cal_n_scales"]) == (2500, 200)
        assert seen["kwargs"]["seed"] == 1840273 and seen["kwargs"]["provide_fig_sink"] is True
        assert (spec.word, spec.place, spec.name, spec.stop_button, spec.kinds) == (
            "Check", ("inference", "checks"), "Calibration", "Stop the calibration…", ("calibration",))
        assert spec.plan == {"kind": "record", "id": PRIOR, "name": "retrain_prior"} == screen.plan_entry()
        assert spec.settings is None
        assert spec.check == {"key": "calibration", "sizes": {"n_cal": 2500, "cal_n_scales": 200,
                                                              "num_posterior_samples": 1000}}
        page._main_clicked()
        assert seen["kwargs"]["seed"] == 77, "the seed of the press before was handed again"


def test_a_check_no_run_can_pass_no_longer_holds_the_step_open(tmp_path):
    """A deciding check that no run can pass for the posterior in use, the input use of a posterior trained without
    keeping its simulations, the identifiability of one trained without the rotation and the comparison not compared,
    is shown not passed with its reason and counts in the headline's denominator, but no longer holds the step open:
    with every other deciding check passed the rail reads "Attention" with the count, Checks is no longer the step in
    hand and Home goes on to Inference, the headline's sentence names what no run can pass, and the foot goes to
    Inference. A check a load, a run or a reading can still lift holds the step open as before: the prior of another, a
    record at a reduced size, one that cannot be read, a comparison waiting for the audit, a comparison whose counts
    could not be read, a posterior just come into use."""
    from core.gui import chain
    from tests._fixtures import check_results, write_calibration, write_diagnostic
    from tests._runs import masked_none
    nothing_counted = f"10:00:00 info {masked_none(WHOLE)}\n"     # a posterior whose set counted no driven recording

    def records(store, cfg, prior, posterior):
        write_diagnostic(store, cfg, diagnostic="probes", variant="band", cell=str(_cell_file()),
                         **check_results("band"))
        write_diagnostic(store, cfg, diagnostic="probes", variant="mask", parents={"prior": prior},
                         **check_results("audit"))
        write_calibration(store, cfg, posterior=posterior, prior=prior)
        one = check_catalogue.entry("identifiability")
        write_diagnostic(store, cfg, diagnostic=one.diagnostic, variant=one.variant,
                         parents={"posterior": posterior, "prior": prior}, **check_results("identifiability"))

    with _checks_page(tmp_path, records=records, text=nothing_counted) as (window, screen, page, _store):
        rows = {row.key: row for row in screen.check_rows()}
        assert (rows["comparison"].state, rows["input_use"].state) == ("Not compared", "Not run yet")
        assert rows["comparison"].line == "Not compared: no driven recordings were counted."
        summary = screen.checks_summary()
        assert (summary.passed, summary.of, summary.open, summary.cannot_pass) == (4, 6, False, 2)
        assert summary.headline == "4 of 6 deciding checks passed"
        assert summary.sentence == ("Inference can be run now; no run can pass 'Unusable driven recordings: "
                                    "simulation set against the audit' or 'Use of each part of the input' for this "
                                    "posterior.")
        assert summary.second_line == "2 that no run can pass"
        checks = screen.step_states()[3]
        assert (checks.marker, checks.rail_words, checks.done) == ("warn", "Attention: 4 of 6 passed", False)
        assert chain.step_in_hand(screen.step_states()) == "infer"
        nxt = chain.next_step(screen.session, screen.step_states(), checks=summary)
        assert (nxt.heading, nxt.button) == ("Next: run the posterior on a cell or on your recordings", "Open Inference")
        assert page.foot.main_button.text().replace("&", "") == "Go to Inference"
        page.open_check("input_use")
        assert page.check_pages["input_use"].state_line.sentence.plain_text() == (
            "Not run yet. This posterior was trained without keeping its simulations, and this check reads them.")
        held = screen.checks_in_use()
        for lifted in (replace(rows["calibration"], state="Not run yet", record=None,
                               reason=check_lineage.OTHER_PRIOR_GONE_TEXT),
                       replace(rows["calibration"], state="Not run yet", record=None,
                               line="Not run yet. Run only at a reduced size (40 test datasets; the usual is 2,000), "
                                    "which does not count."),
                       replace(rows["calibration"], state="Not run yet", record=None,
                               line=f"Not run yet. {check_readings.UNREADABLE_TEXT}"),
                       replace(rows["comparison"], state="Waiting for the audit")):
            assert not check_lineage.no_run_passes(lifted, held=held), lifted
            found = tuple(lifted if row.key == lifted.key else row for row in screen.check_rows())
            assert check_lineage.summary(found, held=held).open is True, lifted
        for unread in (replace(held, set_read=False), replace(held, posterior=None), replace(held, set_counts=None)):
            assert not check_lineage.no_run_passes(rows["comparison"], held=unread)
        kept = screen.session.posterior
        screen.session.posterior = _held_posterior(tmp_path, prior=screen.session.inf_prior, ident=OTHER_POSTERIOR,
                                                   text=nothing_counted)
        screen.refresh_gates()
        assert screen.checks_in_use().set_read is False, "a posterior just come into use was taken for read"
        assert not check_lineage.no_run_passes(_row(screen.check_rows(), "comparison"), held=screen.checks_in_use())
        screen.read_checks()
        assert screen.checks_in_use().set_read is True
        assert check_lineage.no_run_passes(_row(screen.check_rows(), "comparison"), held=screen.checks_in_use())
        screen.session.posterior = kept
        screen.read_checks()
        window.go_to("home", None)
        _settle()
        home = window.part("home")
        assert home.next_heading.label.text() == "Next: run the posterior on a cell or on your recordings"


@pytest.mark.parametrize("factor", [1.0, 1.5])
def test_the_keyboards_row_of_the_list_stays_in_view(factor, tmp_path):
    """At the window's least size the list of sixteen rows is higher than the page's view: the row the keyboard moves
    to, from the first check down to the last and back up, passing over the groups' headings, stands whole in the view
    with its ring's room, and so does the row it stands on when the Tab key brings the keyboard back to the list."""
    from PySide6.QtTest import QTest
    from core.gui import design
    from core.gui.widgets.page import _ring_room
    from tests._window_helpers import in_front
    with text_size(factor), _checks_page(tmp_path, size=(design.scaled(900), design.scaled(600))) as (
            window, _screen, page, _store):
        view, port = page.listing.whole.view, page.scroller.viewport()
        assert view.height() > port.height(), "the list was not higher than its view"

        def row_in_view() -> bool:
            room = _ring_room()
            rect = view.visualRect(view.currentIndex()).adjusted(0, -room, 0, room)
            top = view.viewport().mapTo(port, rect.topLeft()).y()
            return 0 <= top and top + rect.height() <= port.height()

        checks = len(check_catalogue.CHECKS)
        with in_front(window):
            view.setFocus(Qt.TabFocusReason)
            view.setCurrentIndex(view.model().index(page.listing.row_of[check_catalogue.CHECKS[0].key], 0))
            _settle()
            for _ in range(checks - 1):
                QTest.keyClick(view, Qt.Key_Down)
                _settle()
                assert view.currentIndex().row() in page.listing.key_at, "the keyboard stopped on a heading"
                assert row_in_view(), view.currentIndex().row()
            assert view.currentIndex().row() == view.model().rowCount() - 1
            for _ in range(checks - 1):
                QTest.keyClick(view, Qt.Key_Up)
                _settle()
                assert view.currentIndex().row() in page.listing.key_at, "the keyboard stopped on a heading"
                assert row_in_view(), view.currentIndex().row()
            assert view.currentIndex().row() == page.listing.row_of[check_catalogue.CHECKS[0].key]
            view.setCurrentIndex(view.model().index(view.model().rowCount() - 1, 0))
            page.help_button.setFocus(Qt.TabFocusReason)
            page.scroller.verticalScrollBar().setValue(0)
            _settle()
            QTest.keyClick(page.help_button, Qt.Key_Tab)
            _settle()
            assert QApplication.focusWidget() is view and row_in_view()


def test_a_groups_heading_is_no_row_a_click_or_the_keyboard_stops_on(tmp_path):
    """A group's heading stands in the list as its heading and nothing more: a click on it selects nothing and moves no
    ring onto it, and the arrow keys and Home pass over it to a check's row; Space and Enter act on a check's row."""
    from PySide6.QtTest import QTest
    from tests._window_helpers import in_front
    with _checks_page(tmp_path) as (window, _screen, page, _store):
        view, headings, rows = page.listing.whole.view, page.listing.headings, page.listing.row_of
        with in_front(window):
            view.setFocus(Qt.TabFocusReason)
            view.setCurrentIndex(view.model().index(rows["calibration"], 0))
            page.listing.whole.select_row(None)
            QTest.mouseClick(view.viewport(), Qt.LeftButton, pos=_row_point(view, headings[1]))
            _settle()
            assert page.open_key is None and page.listing.whole.selected_row() is None
            assert view.currentIndex().row() == rows["calibration"], "a click put the ring on a heading"
            first = rows[check_catalogue.CHECKS[0].key]
            view.setCurrentIndex(view.model().index(first, 0))
            QTest.keyClick(view, Qt.Key_Up)
            assert view.currentIndex().row() == first, "Up stopped on the first group's heading"
            QTest.keyClick(view, Qt.Key_Home, Qt.ControlModifier)
            assert view.currentIndex().row() == first, "Home stopped on the first group's heading"
            last = max(key_row for key_row in page.listing.key_at if key_row < headings[1])
            view.setCurrentIndex(view.model().index(last, 0))
            QTest.keyClick(view, Qt.Key_Down)
            assert view.currentIndex().row() == headings[1] + 1, "Down stopped on a heading"
            QTest.keyClick(view, Qt.Key_Up)
            assert view.currentIndex().row() == last, "Up stopped on a heading"
            QTest.keyClick(view, Qt.Key_Return)
            _settle()
            assert page.open_key == page.listing.key_at[last]


def test_a_groups_heading_is_drawn_as_a_heading_over_its_checks(tmp_path, monkeypatch):
    """Each group's heading ("Before the run", "After simulating", "After the fit", "Tools") is drawn as a heading in the
    whole table and in the narrow one: set as the table's header sets its words, small and semibold, with a little space
    above it, so that its row stands taller than a check's row of one line by that space; none of its cells can be
    chosen, so that neither a click nor the arrow keys stop on it; and a check's row is drawn in the body's type."""
    from PySide6.QtCore import QRect
    from PySide6.QtGui import QImage, QPainter
    from PySide6.QtWidgets import QStyleOptionViewItem
    from core.gui import design
    from core.gui.widgets import rule_table as rt
    with _checks_page(tmp_path) as (_window, _screen, page, _store):
        page.open_check("band")
        _settle()
        real = rt._face
        for table in (page.listing.whole, page.listing.narrow):
            view, model = table.view, table.view.model()
            one_line = view.rowHeight(page.listing.row_of["card_test"])          # "Card test": one line anywhere
            for at in page.listing.headings:
                cell = model.index(at, 0).data(Qt.UserRole)
                assert isinstance(cell, rt.TextCell) and cell.heading, (at, cell)
                assert view.rowHeight(at) == one_line + design.scaled(design.GAP["head_table"]), (at, view.rowHeight(at))
                assert all(model.flags(model.index(at, column)) == Qt.NoItemFlags
                           for column in range(model.columnCount())), at
            for row, faces in ((page.listing.headings[0], ["title"]), (page.listing.row_of["card_test"], ["text"])):
                asked = []
                monkeypatch.setattr(rt, "_face", lambda part, asked=asked: asked.append(part) or real(part))
                image = QImage(600, 80, QImage.Format_ARGB32)
                painter = QPainter(image)
                try:
                    option = QStyleOptionViewItem()
                    option.rect = QRect(0, 0, 600, view.rowHeight(row))
                    view.itemDelegate().paint(painter, option, model.index(row, 0))
                finally:
                    painter.end()
                monkeypatch.setattr(rt, "_face", real)
                assert asked == faces, (row, asked)


def _a_failed_band(store, cfg, _prior, _posterior) -> None:
    """A frequency band that failed for the master's spontaneous cell: its result is long, its table of every
    frequency among it."""
    from tests._fixtures import check_results, write_diagnostic
    write_diagnostic(store, cfg, diagnostic="probes", variant="band", cell=str(_cell_file()),
                     **check_results("band", band_holds=False))


def _top_in(widget, body) -> int:
    """Where ``widget``'s top stands in ``body``, the page's scrolling body."""
    from PySide6.QtCore import QPoint
    return widget.mapTo(body, QPoint(0, 0)).y()


@pytest.mark.parametrize("factor", [1.0, 1.5])
def test_beside_a_long_result_the_list_stands_at_the_top_and_each_row_is_as_tall_as_its_words(factor, tmp_path):
    """At the window's opening size, beside a check's page whose result is long (a frequency band that failed, its
    table of every frequency among it), the list of checks stands at the top of its column, level with the page beside
    it, whatever that page holds. Every row of the list a check's page stands beside is as tall as its words take as
    they are drawn, the name of a row that is not selected in the body's weight: no row keeps the room of a line its
    words do not take."""
    from core.gui import design
    from core.gui.widgets import rule_table as rt
    with text_size(factor), _checks_page(tmp_path, posterior=False, records=_a_failed_band,
                                         size=(design.scaled(1300), design.scaled(820))) as (_window, _screen, page,
                                                                                            _store):
        page.open_check("band")
        _settle()
        detail, body, table = page.check_pages["band"], page.scroller.body, page.listing.narrow
        assert page.list_detail.beside() and detail.height() > table.height(), (
            "the band's page did not stand beside the list, taller than it")
        assert abs(_top_in(table, body) - _top_in(detail, body)) <= 4, (
            factor, _top_in(table, body), _top_in(detail, body))
        view = page.listing.narrow.view
        line, pad = design.scaled(design.TYPE["body"][1]), design.scaled(design.SIZE["table_pad"])
        marker = design.scaled(design.SIZE["marker"]) + design.scaled(design.SIZE["state_gap"])
        count = view.model().columnCount()
        for row, key in page.listing.key_at.items():
            if row == page.listing.row_of["band"]:
                continue                     # the open check's row, its name set semibold
            name, state = (view.model().index(row, column).data(Qt.UserRole) for column in range(2))
            lead, gap = rt._pads(0, count)
            lines = rt._wrapped(name.text, rt._face("text"), view.columnWidth(0) - lead - gap, line)[1]
            if isinstance(state, rt.StateCell):
                lead, gap = rt._pads(1, count)
                lines = max(lines, rt._wrapped(state.word, rt._face("word"), view.columnWidth(1) - lead - gap - marker,
                                               line)[1])
            want = max(lines * line + 2 * pad, design.scaled(design.SIZE["table_row"]))
            assert view.rowHeight(row) == want, (factor, key, view.rowHeight(row), want)


def test_a_drawing_keeps_the_keyboards_row_and_the_open_checks_row(tmp_path):
    """The list drawn again, at a reading or as the page draws, keeps the row the keyboard is on, by its check, in the
    whole list and beside a check's page; and beside a check's page the open check's row stays selected."""
    from tests._window_helpers import in_front
    with _checks_page(tmp_path, size=(1400, 820)) as (window, _screen, page, _store):
        whole, rows = page.listing.whole.view, page.listing.row_of
        with in_front(window):
            whole.setFocus(Qt.TabFocusReason)
            whole.setCurrentIndex(whole.model().index(rows["identifiability"], 0))
            page.read_facts()
            page.refresh_local_gates()
            assert whole.currentIndex().row() == rows["identifiability"], "a drawing dropped the keyboard's row"
            page.open_check("calibration")
            _settle()
            narrow = page.listing.narrow
            narrow.view.setFocus(Qt.TabFocusReason)
            narrow.view.setCurrentIndex(narrow.view.model().index(rows["band"], 0))
            page.read_facts()
            assert narrow.view.currentIndex().row() == rows["band"], "a drawing dropped the keyboard's row"
            assert narrow.selected_row() == rows["calibration"], "a drawing dropped the open check's row"


def test_the_tab_key_follows_a_checks_page_moved_by_a_resize(tmp_path):
    """A check opened, then the window made narrower and wider across the width where its page moves between beside the
    list and its place: each time the Tab key goes from the heading's button on as the page now stands, to "All
    checks" with the page in the list's place, to the list beside the page."""
    from PySide6.QtTest import QTest
    from core.gui import design
    from tests._window_helpers import in_front
    with _checks_page(tmp_path, size=(design.scaled(1400), design.scaled(820))) as (window, _screen, page, _store):
        page.open_check("band")
        _settle()
        assert page.list_detail.beside()
        with in_front(window):
            for width, beside, after in ((900, False, page.list_detail.back), (1400, True, page.listing.narrow.view),
                                         (900, False, page.list_detail.back)):
                window.resize(design.scaled(width), design.scaled(820))
                _settle()
                assert page.list_detail.beside() is beside, width
                page.help_button.setFocus(Qt.TabFocusReason)
                QTest.keyClick(page.help_button, Qt.Key_Tab)
                _settle()
                assert QApplication.focusWidget() is after, (width, QApplication.focusWidget())


def test_a_checks_button_reads_run_again_once_any_record_of_it_exists(tmp_path):
    """The calibration's button reads "Run again" once a record of it exists, one at a reduced size that does not count
    among them; with none, its own words."""
    from tests._fixtures import check_results, write_calibration
    with _checks_page(tmp_path) as (_window, _screen, page, _store):
        page.open_check("calibration")
        assert page.foot.main_button.text().replace("&", "") == "Run the calibration"

    def records(store, cfg, prior, posterior):
        write_calibration(store, cfg, posterior=posterior, prior=prior,
                          results=check_results("calibration", n_cal=40)["results"])

    with _checks_page(tmp_path / "reduced", records=records) as (_window, screen, page, _store):
        assert _row(screen.check_rows(), "calibration").state == "Not run yet"
        page.open_check("calibration")
        assert page.foot.main_button.text().replace("&", "") == "Run again"


def test_a_checks_page_tells_its_form_the_width_it_stands_in(tmp_path):
    """A check's page, told the width it stands in beside the list or in its place, tells its form the same, so that a
    form that lays itself out by its width is laid out for it."""
    from core.gui import design
    with _checks_page(tmp_path, size=(design.scaled(1400), design.scaled(820))) as (window, _screen, page, _store):
        one = page.check_pages["calibration"]
        told, page_fit = [], one.fit
        one.fit = lambda width: (told.append(width), page_fit(width))[1]
        page.open_check("calibration")
        _settle()
        assert told and one.form._width == told[-1], (told, one.form._width)
        window.resize(design.scaled(900), design.scaled(820))
        _settle()
        assert one.form._width == told[-1] == page.list_detail._width, (told, one.form._width)


def test_the_two_readings_of_the_checks_are_made_only_where_they_are_needed(tmp_path, monkeypatch):
    """At a reading, the label of the prior the posterior was trained on is read only when the engine refuses the pair,
    and of the checks' records only the text of the one that decides what the observations can pin down is read."""
    from core.gui import check_lineage as lineage
    from core.gui.screens import inference_screen
    with _checks_page(tmp_path, records=_every_record) as (_window, screen, page, _store):
        labels, texts = [], []
        real_label, real_text = inference_screen.prior_label, lineage.read_text
        monkeypatch.setattr(inference_screen, "prior_label", lambda *args: (labels.append(args), real_label(*args))[1])
        monkeypatch.setattr(lineage, "read_text", lambda folder: (texts.append(folder), real_text(folder))[1])
        page.read_facts()
        decides = _row(screen.check_rows(), "pin_down").record.folder
        records = {record.folder for record in screen.checks_held.records}
        assert labels == [], "the label was read with the pair taken"
        assert [folder for folder in texts if folder in records] == [decides], texts


def test_each_checks_page_is_made_inside_the_list_out_of_sight_and_reading_nothing(tmp_path, monkeypatch):
    """The twelve check pages are made with the list, each its page's child from the start and out of sight until it
    is opened, with a form of its own, in sight only while it holds rows; the list's own form holds nothing and stands
    out of sight; and building the page reads nothing and starts no timer."""
    from PySide6.QtCore import QTimer
    from core.gui.panels.inference.check_page import CheckPage
    from core.gui.panels.inference.validate_tab import ValidatePanel
    from tests._fixtures import build_step_page, build_window, qt_app, readings_of_a_run
    qt_app()
    window = build_window()
    window.part("inference")
    calls = readings_of_a_run(monkeypatch)
    page = build_step_page(ValidatePanel, window=window, step="checks")
    _settle()
    assert calls == [] and not [timer for timer in page.findChildren(QTimer) if timer.isActive()]
    assert list(page.check_pages) == [one.key for one in check_catalogue.CHECKS]
    for key, one in page.check_pages.items():
        assert isinstance(one, CheckPage) and one.parent() is page and one.isHidden(), key
        assert one.form.isHidden() == (not one.form.rows), key
    assert not page._own_form.rows and page._own_form.isHidden()


#: The walk of the Checks list at the larger size, run alone in a process of its own.
CHECKS_WALKED_ALONE = ("tests/test_pages.py::test_tab_walks_the_page_in_view_at_every_size"
                       "[(1300, 820)-1.0-checks-the calibration open-fold shut]")


def test_the_checks_lists_walk_run_alone_ends_its_process_cleanly(tmp_path):
    """The walk of the Checks list with the calibration's page open beside it, run alone in a process of its own, ends
    that process as a test run ends. Twelve check pages, each with a form, stand in the window: a page with a form made
    with no parent and never shown, standing while the program's style sheet was changed and put back and let go after
    its window, once corrupted the process's memory as the test ended (the exit 0xC0000374). The process sees the card
    as the run that starts it does. It is a canary of that crash and no more: it does not make the crash happen, and
    the rule that every check's page is made inside the list is held by the test of how the pages are made."""
    import os
    import subprocess
    root = Path(__file__).resolve().parents[1]
    environment = {**os.environ, "QT_QPA_PLATFORM": "offscreen", "KMP_DUPLICATE_LIB_OK": "TRUE",
                   "PRISM_ARTIFACTS": str(tmp_path / "Artifacts")}
    for name in ("PRISM_VRAM_CEILING_GIB", "PYTEST_CURRENT_TEST", "PYTEST_ADDOPTS"):
        environment.pop(name, None)
    (tmp_path / "Artifacts").mkdir()
    child = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", CHECKS_WALKED_ALONE],
                           cwd=root, env=environment, capture_output=True, text=True, encoding="utf-8", timeout=300)
    code = child.returncode & 0xFFFFFFFF
    assert code == 0, (f"the process ended with 0x{code:08X}" + (", a corrupted heap" if code == 0xC0000374 else ""),
                       child.stdout[-2000:], child.stderr[-2000:])
    assert "1 passed" in child.stdout, child.stdout[-2000:]


# ── the forms and runs of the checks ──
#: The cell file a check that reads one is given, in the model's own folder: one that loads on the master bounds file.
CELL_KEY = "master_spont.txt"
#: What a dispatch is handed that is the controller's and no argument of the stage.
_CONTROLLERS = ("spec", "host", "on_result", "provide_fig_sink", "provide_stream", "on_chunk", "watch_dir")
#: Each check whose form is built: its stage by the module that holds it and its name there, and what the stage is
#: handed at the form's usual values (the calibration's seed, left blank, drawn) and at changed ones, each by the
#: stage's own name, the cell typed and the name and note as a press hands them.
_HANDED = {
    "band": (("core.diagnostics.probes", "probe_band"), {"lengths": None, "repeats": 24, "seed": 0},
             {"probe_lengths": "1, 2.5", "band_repeats": "30", "probe_seed": "7"},
             {"lengths": [1.0, 2.5], "repeats": 30, "seed": 7}),
    "audit": (("core.diagnostics.probes", "probe_mask"), {"num_runs": 100, "run_size": 32, "seed": 0},
              {"mask_num_runs": "120", "mask_run_size": "16", "probe_seed": "3"},
              {"num_runs": 120, "run_size": 16, "seed": 3}),
    "calibration": (("core.orchestrator", "validate_calibration"),
                    {"n_cal": 2000, "cal_n_scales": 200, "num_posterior_samples": 1000, "seed": 1840273},
                    {"n_cal": "2500", "cal_n_scales": "150", "num_posterior_samples": "500", "seed": "11"},
                    {"n_cal": 2500, "cal_n_scales": 150, "num_posterior_samples": 500, "seed": 11}),
    "input_use": (("core.diagnostics.ablation", "channel_ablation"), {"rows": 200_000, "n_sweep": 33},
                  {"rows": "1000", "n_sweep": "5"}, {"rows": 1000, "n_sweep": 5}),
    "identifiability": (("core.diagnostics.identifiability", "identifiability_rotation"), {"n_worst": 3, "top_n": 4},
                        {"n_worst": "2", "top_n": "6"}, {"n_worst": 2, "top_n": 6}),
}


def _engine(where):
    import importlib
    module, name = where
    return getattr(importlib.import_module(module), name)


def _ready(page, key) -> None:
    """The check ``key`` open, its form at its usual values but a cell chosen where it reads one."""
    from core.gui.panels.inference import check_forms
    page.open_check(key)
    picker = check_forms.cell_picker(page.check_pages[key].form)
    if picker is not None:
        picker.restore_key(CELL_KEY)
    page.redraw()


def _tripwire(monkeypatch) -> dict:
    """The run controller's start, recording what it is handed and starting nothing."""
    from core.gui.run.controller import RunController
    seen = {"calls": 0}

    def started(held, fn, *args, **kwargs):
        seen.update(fn=fn, args=args, kwargs=kwargs, calls=seen["calls"] + 1)
        return True
    monkeypatch.setattr(RunController, "dispatch", started)
    return seen


@pytest.mark.parametrize("key", list(_HANDED))
def test_each_deciding_check_hands_its_stage_the_forms_arguments(key, tmp_path, monkeypatch):
    """A press of a deciding check hands its own stage, by the stage's own names, what its form holds: at the usual
    values and at changed ones, with the record's name and note, the figures' sink asked for, and the configuration,
    the posterior and the prior as the stage takes them; no setting of the engine is written."""
    import random
    from core import config
    from core.gui.panels.inference import check_forms
    seen = _tripwire(monkeypatch)
    monkeypatch.setattr(random, "randrange", lambda top: 1840273 if top == 2 ** 31 else 0)
    where, usual, typed, changed = _HANDED[key]
    settings = {name: repr(value) for name, value in vars(config).items() if name.isupper()}
    with _checks_page(tmp_path, simulation=SET) as (_window, screen, page, _store):
        session = screen.session
        _ready(page, key)
        for values, wanted in ((None, usual), (typed, changed)):
            form = page.check_pages[key].form
            if values is not None:
                for setting, text in values.items():
                    form.rows[setting].widget.setText(text)
                form.names.name.setText("a_check")
                form.names.note.setText("why it was run")
            page._main_clicked()
            assert seen["fn"] is _engine(where), (key, seen.get("fn"))
            handed = {name: value for name, value in seen["kwargs"].items() if name not in _CONTROLLERS}
            named = ("a_check", "why it was run") if values is not None else ("", "")
            assert handed == {**wanted, "name": named[0], "note": named[1]}, (key, handed)
            assert seen["kwargs"]["spec"].check == {"key": key, "sizes": check_forms.sizes_of(key, form)}, key
            assert seen["kwargs"]["provide_fig_sink"] is True
            given = {"band": 1, "audit": 2, "calibration": 3, "input_use": 2, "identifiability": 2}[key]
            assert len(seen["args"]) == given, seen["args"]
            if key in ("band", "audit"):
                assert seen["args"][0] is not session.cfg and seen["args"][0].chi_mode is True
            else:
                assert seen["args"][0] is session.cfg
            assert seen["args"][1:] == {"band": (), "audit": (session.inf_prior,),
                                        "calibration": (session.posterior, session.inf_prior),
                                        "input_use": (session.posterior,),
                                        "identifiability": (session.posterior,)}[key]
    assert {name: repr(value) for name, value in vars(config).items() if name.isupper()} == settings


@pytest.mark.parametrize("limits", ["a bounds file", "limits typed by hand"])
def test_a_check_builds_its_own_configuration_and_never_writes_the_sessions(limits, tmp_path, monkeypatch):
    """The frequency band, the audit and the drive study run on a configuration of their own, made as the session's was,
    from the step 1 settings applied and the bounds file in use or the limits typed by hand, the first two with the
    frequency response and the drive study without it, on the session configuration's device: the cell is loaded on it
    alone, the audit's holds none, and the session's configuration and its step 1 settings are as they were."""
    from collections import OrderedDict
    seen = _tripwire(monkeypatch)
    with _checks_page(tmp_path) as (_window, screen, page, _store):
        held = screen.session.cfg
        if limits == "limits typed by hand":
            held = screen.session.draft.make_config(bounds_dicts=tuple(
                OrderedDict((name, (None, ends)) for name, (_value, ends) in section.items())
                for section in (held.params_dict, held.rescale_params, held.force_params_dict)), hw=held.hw)
            screen.session.cfg = held
            assert "bounds" not in held.sources
        before = {name: repr(getattr(held, name)) for name in (
            "sources", "params_dict", "rescale_params", "force_params_dict", "chi_mode", "T_obs")}
        draft, drafted = screen.session.draft, repr(screen.session.draft)
        for key in ("band", "audit", "drive_study"):
            _ready(page, key)
            page._main_clicked()
            made = seen["args"][0]
            assert made is not held and made.chi_mode is (key != "drive_study"), key
            assert made.hw.device == held.hw.device, key
            assert [(name, ends) for name, (_v, ends) in made.params_dict.items()] == [
                (name, ends) for name, (_v, ends) in held.params_dict.items()], key
            assert made.sources.get("bounds") == held.sources.get("bounds"), key
            assert (made.sources.get("cell") is not None) == (key != "audit"), (key, made.sources)
        assert {name: repr(getattr(held, name)) for name in before} == before
        assert screen.session.draft is draft and repr(draft) == drafted, "the step 1 settings were written"


def test_the_calibrations_seed_left_blank_is_drawn_and_named_in_its_result(tmp_path, monkeypatch):
    """The calibration's seed opens blank: a press draws one afresh and hands it, so that its record names it, and the
    next press draws another; a seed typed is handed as typed."""
    import random
    seen = _tripwire(monkeypatch)
    drawn = iter((1840273, 77))
    monkeypatch.setattr(random, "randrange", lambda top: next(drawn) if top == 2 ** 31 else 0)
    with _checks_page(tmp_path) as (_window, _screen, page, _store):
        _ready(page, "calibration")
        assert page.form.rows["seed"].widget.text() == "" and page.form.rows["seed"].at_usual()
        page._main_clicked()
        assert seen["kwargs"]["seed"] == 1840273
        page._main_clicked()
        assert seen["kwargs"]["seed"] == 77
        page.form.rows["seed"].widget.setText("5")
        page._main_clicked()
        assert seen["kwargs"]["seed"] == 5
    region = check_results_block(tmp_path, seed=1840273)
    assert ", seed 1840273" in region, region


def check_results_block(tmp_path, *, seed) -> str:
    """The verdict of a calibration's result whose record holds ``seed``."""
    from core.gui.check_readings import verdict_sentence
    return verdict_sentence("calibration", {"body": {"results": _calibration_of(["k"], ks=[0.5], seed=seed)}})


#: Every number of the deciding checks' forms that a rule holds, with what it is typed as, the line under its field and
#: what a press would be refused with: blank, below its least and, for a seed, beyond the one rule of a seed.
#: What each kind of field is typed with that a rule refuses: a whole number blank and one below its least; a real
#: number blank, at 0 and below; a list of numbers holding a word, a 0, a number below 0 and a separator alone; a seed
#: blank (but the calibration's, which a press draws), below 0 and beyond the largest the rule takes.
_TYPED = {"whole": lambda one: ("", str(one.least - 1)), "epochs": lambda one: ("", str(one.least - 1)),
          "number": lambda one: ("", "0", "-1"),
          "lengths": lambda one: ("x", "0", "-1", ","), "seed": lambda one: ("", "-1", str(2 ** 64)),
          "blank_seed": lambda one: ("-1", str(2 ** 64))}
_BROKEN = [(key, one.setting, text)
           for key, rows in __import__("core.gui.panels.inference.check_forms", fromlist=["FORMS"]).FORMS.items()
           for one in rows if one.kind in _TYPED for text in _TYPED[one.kind](one)]


def _broken_line(one, text, *, added=0) -> str:
    """The line under the field of the row ``one`` typed with ``text``, in the window's words from its label."""
    from core.gui import fields as gui_fields
    from core.rng import SEED_MAX
    label = one.label or gui_fields.label(one.setting)
    if one.kind == "lengths":
        return (f"{label} must be numbers separated by commas or spaces." if text in ("x", ",")
                else f"{label} must each be greater than 0.")
    if not text:
        return "Type a recording length." if one.setting == "t_obs" else f"{label} is blank."
    if "seed" in one.kind:
        return f"{label} must be a whole number from 0 to {SEED_MAX - added}."
    if one.kind == "number":
        return f"{label} must be greater than 0."
    return f"{label} must be at least {one.least}."


@pytest.mark.parametrize("key, setting, text", _BROKEN, ids=[f"{key}-{setting}-{text or 'blank'}"
                                                             for key, setting, text in _BROKEN])
def test_a_blank_or_bad_number_is_said_under_its_field_and_blocks(key, setting, text, tmp_path, monkeypatch):
    """Every number, list of numbers and seed of every check's form, blank or breaking its stage's rule (the least read
    from its row), is said under its field in the window's words, from its row's label, the rule decided by the stage's
    own (``require_at_least``, ``require_positive``, ``require_seed``); the button is greyed "Cannot start yet: a setting
    cannot be used."; a press dispatches nothing, and a start reached all the same is refused under the field's key in
    the stage's own sentence, or, for a list that holds no numbers, which no rule of the stage can be handed, in the
    line's."""
    from core.gui.panels.inference import check_forms
    seen = _tripwire(monkeypatch)
    refused = []
    # with the frequency response, but for identifiability across the prior, which does not apply with it
    with _checks_page(tmp_path, chi=key != "across_prior", simulation=SET) as (_window, _screen, page, _store):
        _ready(page, key)
        if setting != "t_obs" and "t_obs" in page.form.rows:
            _typed_into(page.form, {"t_obs": "2.5"})
        one = next(row for row in check_forms.FORMS[key] if row.setting == setting)
        row = page.form.rows[setting]
        row.widget.setText(text)
        page.redraw()
        repeats = page.form.rows["repeats"].value() - 1 if one.adds else 0
        line = _broken_line(one, text, added=repeats)
        assert (row.line.kind, row.line.words) == ("err", line), (row.line.kind, row.line.words)
        assert not page.foot.main_button.isEnabled()
        assert page.foot.reason.text() == "Cannot start yet: a setting cannot be used."
        page._main_clicked()
        assert seen["calls"] == 0
        page._refusal = refused.append
        getattr(page, f"_start_{key}")()
        assert seen["calls"] == 0 and len(refused) == 1 and refused[0].field == setting, refused
        if one.kind == "lengths" and text in ("x", ","):
            assert refused[0].message == line
        else:
            assert refused[0].message != line and refused[0].message.startswith("The "), refused[0].message


def test_the_repeated_calibrations_largest_seed_leaves_room_for_every_repeat(tmp_path, monkeypatch):
    """The calibration repeated at fixed numbers seeds its repeat r at its seed and r, so its seed is held to the one
    rule of a seed for its last repeat too, the engine's own rule called on that one: at 10 repeats the largest seed
    that passes is nine below the largest the rule takes, and the line under the field says the bound the repeats leave;
    at one repeat the largest itself passes."""
    from core.rng import SEED_MAX
    seen = _tripwire(monkeypatch)
    with _checks_page(tmp_path, simulation=SET) as (_window, _screen, page, _store):
        _ready(page, "fixed_number")
        row = page.form.rows["seed"]
        row.widget.setText(str(SEED_MAX - 8))
        page.redraw()
        assert (row.line.kind, row.line.words) == ("err", f"Seed must be a whole number from 0 to {SEED_MAX - 9}.")
        page._main_clicked()
        assert seen["calls"] == 0
        row.widget.setText(str(SEED_MAX - 9))
        page._main_clicked()
        assert seen["calls"] == 1 and seen["kwargs"]["seed"] == SEED_MAX - 9
        page.form.rows["repeats"].widget.setText("1")
        row.widget.setText(str(SEED_MAX))
        page._main_clicked()
        assert seen["calls"] == 2 and seen["kwargs"]["seed"] == SEED_MAX


#: What "Before you start" says of each deciding check at its usual sizes, after its time and with the memory lines
#: where the check simulates on the graphics card.
_COUNTS = {"band": "Simulates 24 recordings at each recording length and frequency of the band: five lengths chosen "
                   "for this cell.",
           "audit": "Simulates 100 batches of 32, as training would: 3,200 simulations.",
           "calibration": "Simulates 200 batches, holding 2,000 test datasets, then draws 1,000 samples for each.",
           "input_use": "Reads records and simulates nothing; takes seconds.",
           "identifiability": "Reads records and simulates nothing; takes seconds.",
           "pin_down": "Simulates 32 recordings for each comparison and 128 for the noise level.",
           "fixed_number": "Runs 10 calibrations of 2,000 test datasets each, about 10 times the calibration's time.",
           "drive_study": "Simulates 16 recordings at each of 17 strengths.",
           "across_prior": "Simulates at 6 points across the prior, at each as the check of what the observations can "
                           "pin down does at one."}


@pytest.mark.parametrize("key", list(_COUNTS))
def test_before_you_start_says_the_counts_the_time_the_memory_and_what_is_written(key):
    """In its order: what the run simulates or reads, in counts; how long it took here last at these sizes, or that it
    was not timed here; the memory limit and the last run, for a check that simulates on the card and for no other;
    that nothing else can run meanwhile; and the one record it writes."""
    from core.gui.panels.inference import check_forms
    sizes = {one.setting: one.usual for one in check_forms.FORMS[key] if one.kind not in check_forms.NOT_SIZES}
    memory = ("Memory limit: none.", "The last run on this computer ran with no limit.")
    said = check_forms.before_you_start(key, sizes=sizes, last=None, memory=memory)
    card = key not in ("input_use", "identifiability")
    assert [(fact.kind, fact.sentence) for fact in said] == [
        ("info", _COUNTS[key]), ("info", "Not timed on this computer yet."),
        *([("info", " ".join(memory))] if card else []),
        ("info", "Nothing else can run until it ends."), ("info", "One record, under Records.")]
    last = {"check": {"key": key, "sizes": sizes, "seconds": 43 * 60 + 20}}
    assert check_forms.before_you_start(key, sizes=sizes, last=last, memory=())[1].sentence == (
        "Took 43 minutes here last time at these sizes.")
    if key == "band":
        typed = check_forms.before_you_start(key, sizes={**sizes, "probe_lengths": [1.0, 2.0, 3.0]}, last=None,
                                             memory=())
        assert typed[0].sentence == ("Simulates 24 recordings at each recording length and frequency of the band: 3 "
                                     "recording lengths.")


def test_a_checks_time_is_this_computers_own_from_the_history(tmp_path, monkeypatch):
    """How long a check took is read from this computer's finished runs of it, held from the page's reading: two runs at
    two sizes on this computer, and the form edited from one to the other, says each its own time, reading nothing at
    the edit; at a third size, and for a run on another computer only, it was not timed here."""
    from core.gui.memory_limit import computer
    from core.gui.run import history
    from tests._fixtures import readings_of_a_run
    here = computer()

    def ran(at, n_cal, seconds, on):
        history.write(tmp_path / "runs", {
            "id": f"2026100812000{at}-check", "word": "Check", "name": "Calibration", "place": ["inference", "checks"],
            "records_root": str(tmp_path / "Artifacts"), "started": _at(8, 12, at), "ended": _at(8, 13, at),
            "outcome": "finished", "memory_limit": {"gib": None, "source": "none", "computer": on},
            "check": {"key": "calibration", "sizes": {"n_cal": n_cal, "cal_n_scales": 200,
                                                      "num_posterior_samples": 1000}, "seconds": seconds}})

    ran(1, 2000, 43 * 60, here)
    ran(2, 1000, 20 * 60, here)
    ran(3, 3000, 90 * 60, "another computer")
    with _checks_page(tmp_path) as (_window, _screen, page, _store):
        page.open_check("calibration")
        page.read_facts()

        def time_said():
            return next(fact.sentence for fact in page.facts_from_form()
                        if "here last time" in fact.sentence or fact.sentence.startswith("Not timed"))

        assert time_said() == "Took 43 minutes here last time at these sizes."
        calls = readings_of_a_run(monkeypatch)

        def edited(text):
            """``n_cal`` typed, and a reading the edit started fired at once rather than after its pause."""
            page.form.rows["n_cal"].widget.setText(text)
            if page.read_timer.isActive():
                page.read_timer.timeout.emit()
            return time_said()

        assert edited("1000") == "Took 20 minutes here last time at these sizes."
        assert edited("3000") == "Not timed on this computer yet."
        assert edited("1500") == "Not timed on this computer yet."
        _settle()
        assert calls == [], calls


def test_the_input_use_refusals_are_shown_as_refusals_naming_the_saved_posterior(tmp_path, monkeypatch):
    """A refusal the input-use check gives under the posterior's key is shown as every refusal is: "Check your inputs",
    the engine's sentence, and where to fix it, "Saved posterior" in step 3; the run's page says no failure, and the
    check's state is as it was."""
    from core.diagnostics import ablation
    from core.refusals import Refusal
    from tests._dialogs import asked
    from tests._runs import StandIn, turn_until
    sentence = "Posterior 'retrain' names no simulation cache, so checkpointing was off for its training run."
    stand_in = StandIn(word="Check")
    monkeypatch.setattr(ablation, "channel_ablation", stand_in)
    app = qt_app()
    with _checks_page(tmp_path, simulation=SET) as (_window, screen, page, _store):
        before = _row(screen.check_rows(), "input_use").state
        _ready(page, "input_use")
        page._main_clicked()
        assert stand_in.started.wait(5)
        stand_in.fail(Refusal(sentence, field="posterior"))
        turn_until(app, lambda: not controller().is_live())
        _settle()
        box = asked()
        assert (box.kind, box.title, box.sentence, box.beneath) == (
            "refusal", "Check your inputs", sentence, "Set it in 'Saved posterior' in step 3, Training.")
        assert controller().status.outcome is None, "the refusal was told as a failed run"
        page.read_facts()
        assert _row(screen.check_rows(), "input_use").state == before


@pytest.mark.parametrize("ending", ["refused", "failed", "stopped", "pressed twice"])
def test_a_check_that_ends_any_way_but_well_leaves_the_state_as_it_was(ending, tmp_path, monkeypatch):
    """A check refused, failed or stopped, or pressed twice as it starts, keeps nothing and starts no second run: one
    dispatch, no record written, its row's state and the headline as they were."""
    from core import orchestrator
    from core.refusals import Refusal
    from tests._dialogs import answer_in_turn
    from tests._runs import StandIn, turn_until
    stand_in = StandIn(word="Check")
    monkeypatch.setattr(orchestrator, "validate_calibration", stand_in)
    app = qt_app()
    with _checks_page(tmp_path) as (_window, screen, page, store):
        before = (_row(screen.check_rows(), "calibration").state, screen.checks_summary().headline)
        _ready(page, "calibration")
        page._main_clicked()
        if ending == "pressed twice":
            page._main_clicked()
        assert stand_in.started.wait(5)
        if ending == "refused":
            stand_in.fail(Refusal("The number of calibration datasets must be at least 1; got 0.", field="n_cal"))
        elif ending == "failed":
            stand_in.fail(RuntimeError("the simulation broke"))
        elif ending == "stopped":
            answer_in_turn(monkeypatch, ("Stop and discard it", None))
            page.run_view.stop_button.click()
        else:
            stand_in.release()
        turn_until(app, lambda: not controller().is_live(), 10.0)
        _settle()
        page.read_facts()
        assert len(stand_in.calls) == 1, stand_in.calls
        assert [summary.id for summary in store.list("calibration")] == []
        assert (_row(screen.check_rows(), "calibration").state, screen.checks_summary().headline) == before


@pytest.mark.parametrize("key, words", [("calibration", "Stop the calibration"), ("audit", "Stop the audit"),
                                        ("band", "Stop this check")])
def test_a_checks_stop_asks_with_keep_it_running_preselected(key, words):
    """A check's stop asks its own question: its title the stop button's words asked, that a check writes its record
    only when it finishes, "Keep it running" the safe answer, and "Stop and discard it" drawn as losing work."""
    from core.gui.panels.inference import check_forms
    from core.gui.run import questions
    from core.gui.run.model import RunModel
    spec = check_forms.check_spec(key, sizes={})
    assert spec.stop_button == f"{words}…"
    asked = questions.stop_question(None, RunModel(spec, clock=lambda: _at(8)))
    assert (asked.title, asked.sentence, asked.safe, asked.other, asked.loses_work) == (
        f"{words}?", "A check writes its record only when it finishes, so nothing of this one is kept.",
        "Keep it running", "Stop and discard it", True)


def test_the_calibrations_stages_begin_at_their_lines():
    """The calibration's four stages begin at their lines, in order: its test datasets simulated from the start and
    counted by the bar of the simulation, the rank test at the line that ends it, the joint coverage test and the
    saving at the readings that begin them; every other check has one stage, "Measuring"."""
    from core.gui import vt
    from core.gui.panels.inference import check_forms
    from core.gui.run.model import RunModel
    from tests._runs import CALIBRATION_LINES
    model = RunModel(check_forms.check_spec("calibration", sizes={}), clock=lambda: _at(8))
    frame, *lines = CALIBRATION_LINES
    model.rows([vt.parse_bar(("err", 0), 0, frame)])
    began = [model.stage]
    assert model.doing() == "Simulating, batch 3 of 10", model.doing()
    for line in lines:
        model.feed([(line, "info")])
        began.append(model.stage)
    assert began == ["Simulating the test datasets", "Rank test", "Joint coverage test", "Saving"], began
    assert [stage.name for stage in check_forms.check_spec("band", sizes={}).stages] == ["Measuring"]


def test_every_check_form_row_is_named_from_the_control_table(tmp_path):
    """STANDING. Every row of every check's form whose setting the table of controls names is labelled with that key's
    one label, but a row that hands a label of its own, which no key of the table holds (a check's own "Epochs"), and
    a row that is only shown, which says its words whole and names no control (the networks a check fits); read on
    each check's page as it opens."""
    from core.gui import fields as gui_fields
    from core.gui.panels.inference import check_forms
    with _checks_page(tmp_path) as (_window, _screen, page, _store):
        for key, rows in check_forms.FORMS.items():
            page.open_check(key)
            for one in rows:
                said = page.form.rows[one.setting].label_text()
                if one.kind == "network":
                    assert page.form.rows[one.setting].widget is None and one.setting not in gui_fields.CONTROL, key
                    assert said.startswith(("Networks: ", "Network: ")), (key, said)
                elif one.label is None:
                    assert said == gui_fields.label(one.setting), (key, one.setting, said)
                else:
                    assert not isinstance(gui_fields.CONTROL.get(one.setting), tuple), (key, one.setting)
                    assert said == one.label, (key, one.setting, said)


def test_the_check_forms_remember_nothing(tmp_path, monkeypatch):
    """STANDING. A check's form keeps nothing between launches: every form edited and a check run, the window closed
    and built again, each row of each check opens at its usual value, and nothing of a check is written."""
    from PySide6.QtCore import QSettings
    from core import orchestrator
    from core.gui.panels.inference import check_forms
    from tests._runs import StandIn, turn_until
    written = []
    real = QSettings.setValue
    monkeypatch.setattr(QSettings, "setValue", lambda held, key, value: (written.append(key), real(held, key, value)))
    stand_in = StandIn(word="Check")
    monkeypatch.setattr(orchestrator, "validate_calibration", stand_in)
    app = qt_app()
    with _checks_page(tmp_path / "untouched") as (window, _screen, page, _store):
        window._save_state()
    untouched, written[:] = set(written), []
    with _checks_page(tmp_path) as (window, _screen, page, _store):
        for key, rows in check_forms.FORMS.items():
            page.open_check(key)
            for one in rows:
                widget = page.form.rows[one.setting].widget
                if hasattr(widget, "setText"):
                    widget.setText("7")
            if page.form.names is not None:          # a check that keeps nothing names no record
                page.form.names.name.setText("kept_nowhere")
        page.open_check("calibration")
        page._main_clicked()
        assert stand_in.started.wait(5)
        stand_in.release()
        turn_until(app, lambda: not controller().is_live())
        window._save_state()
    with _checks_page(tmp_path / "again") as (_window, _screen, page, _store):
        for key, rows in check_forms.FORMS.items():
            form = page.check_pages[key].form
            settings = [one for one in rows if one.kind not in ("cell", "network")]
            assert [(one.setting, form.rows[one.setting].at_usual()) for one in settings] == [
                (one.setting, True) for one in settings], key
            assert form.names is None or (form.names.name_value(), form.names.note_value()) == ("", ""), key
            assert (form.names is None) == (key in check_readings.KEEPS_NOTHING), key
    assert set(written) <= untouched, sorted(set(written) - untouched)


def test_every_check_form_row_is_a_keyword_of_the_function_its_check_starts(tmp_path, monkeypatch):
    """STANDING. Each row of a check's form hands the stage its check's start dispatches a keyword that stage takes, by
    the stage's own name for it: read off that function's signature, a row's argument, or its setting where the stage
    names it so, is a parameter passed by its keyword; a cell file is no argument, loaded on the check's own
    configuration. Each check's press is followed to the function it dispatches. And each row opens at the default of
    that parameter, but the few whose usual value is the window's own, each with its reason: the audit's batches, at
    the least that counts; the two calibrations' sizes the stage takes from the settings when it is handed none, as the
    window opens them; the timing's batches and epochs, the retraining runbook's, and the card test's sizes, the engine's
    card test's own (held to them in its own test), which their calls have no default for. A row that is only shown
    hands nothing by its own name: the timing's networks are handed as a network's width and depth. The card test's
    cell is handed by its path, since the engine's card test reads it itself."""
    import inspect
    from core import config
    from core.gui.panels.inference import check_forms
    own = {("audit", "mask_num_runs"): check_readings.AUDIT_LEAST_BATCHES,
           ("calibration", "n_cal"): config.SBC_N_CAL, ("calibration", "cal_n_scales"): config.CAL_N_SCALES,
           ("fixed_number", "cal_n_scales"): config.CAL_N_SCALES,
           ("timing", "num_runs"): check_forms.TIMING_BATCHES, ("timing", "epochs"): check_forms.TIMING_EPOCHS,
           **{("card_test", setting): usual for setting, usual in (
               ("t_obs", check_forms.CARD_LENGTH_S), ("num_runs", check_forms.CARD_BATCHES),
               ("run_size_cap", check_forms.CARD_PER_BATCH), ("n_cal", check_forms.CARD_DATASETS),
               ("epochs", check_forms.CARD_EPOCHS), ("hidden_features", check_forms.BUILT_IN_NETWORK[0]),
               ("num_transforms", check_forms.BUILT_IN_NETWORK[1]), ("seed", 0))}}
    seen = _tripwire(monkeypatch)
    pressed = set()
    for chi in (True, False):
        with _checks_page(tmp_path / str(chi), chi=chi, simulation=SET) as (_window, _screen, page, _store):
            for key, rows in check_forms.FORMS.items():
                if key in pressed or (key == "across_prior") == chi or (key in ("band", "audit") and not chi):
                    continue
                seen.pop("fn", None)
                _ready(page, key)
                _typed_into(page.form, {"t_obs": "2.5"} if "t_obs" in page.form.rows else {})
                page._main_clicked()
                assert "fn" in seen, (key, page.foot.reason.text())
                pressed.add(key)
                taken = inspect.signature(seen["fn"]).parameters
                for one in rows:
                    if one.kind == "cell":            # the card test reads its cell itself, handed by its path
                        assert one.argument == "" and (one.setting in seen["kwargs"]) == (key == "card_test"), key
                        continue
                    if one.kind == "network":
                        assert one.setting not in seen["kwargs"] and {"hidden_features", "num_transforms"} <= set(
                            seen["kwargs"]) <= set(taken) | set(_CONTROLLERS), key
                        continue
                    name = one.argument or one.setting
                    assert name in taken and taken[name].kind in (inspect.Parameter.KEYWORD_ONLY,
                                                                  inspect.Parameter.POSITIONAL_OR_KEYWORD), (key, name)
                    assert name in seen["kwargs"], (key, name)
                    usual = own.get((key, one.setting), taken[name].default)
                    assert one.usual == usual, (key, one.setting, one.usual, taken[name].default)
    assert pressed == set(check_forms.FORMS), sorted(set(check_forms.FORMS) - pressed)


def test_a_drawing_of_the_list_never_scrolls_the_page(tmp_path):
    """The list keeps the row the keyboard is on through a drawing, and a drawing never scrolls the page: the page left
    where the person scrolled it with the wheel, the row the keyboard is on out of sight, stays there through every
    drawing and reading; a move of the keyboard brings its row into view."""
    from PySide6.QtTest import QTest
    from tests._window_helpers import in_front
    with _checks_page(tmp_path, size=(900, 600)) as (window, _screen, page, _store):
        view, bar = page.listing.view(), page.scroller.verticalScrollBar()
        assert bar.maximum() > 0, "the page does not scroll at this size"
        with in_front(window):
            view.setFocus(Qt.TabFocusReason)
            view.setCurrentIndex(view.model().index(page.listing.row_of["drive_study"], 0))
            _settle()
            bar.setValue(0)
            _settle()
            for redraw in (page.refresh_local_gates, page.read_facts, page.redraw):
                redraw()
                _settle()
                assert bar.value() == 0, (redraw.__name__, bar.value())
                assert view.currentIndex().row() == page.listing.row_of["drive_study"], redraw.__name__
            QTest.keyClick(view, Qt.Key_Up)
            _settle()
            assert bar.value() > 0, "a move of the keyboard left its row out of view"


def test_a_drafts_configuration_is_made_for_the_device_handed_and_else_as_before(monkeypatch):
    """The settings of step 1 make a configuration for the device a caller hands, as a check's own is made on the
    session configuration's; a caller that hands none, as every other does, leaves the device to the builder's own
    choice, as before."""
    from core import cli
    from core.gui.session import ConfigDraft
    handed = []
    monkeypatch.setattr(cli, "make_sim_config", lambda *args, **kwargs: handed.append(kwargs.get("hw", "none")))
    draft = ConfigDraft(model="NADROWSKI", labels=["x"], state_dep_drift=False)
    draft.make_config("bounds.txt")
    draft.make_config(bounds_dicts=({}, {}, {}))
    draft.make_config("bounds.txt", hw="the card")
    assert handed == [None, None, "the card"], handed


# ── the forms and runs of the informing checks and the tools ──
#: Each informing check and tool with a form: whether its window has the frequency response, its stage, what a press
#: hands at its form's usual values with only what must be typed typed (a cell, a recording length) and at changed
#: ones, by the stage's own names, and how many of the call's first arguments are the stage's.
_HANDED_TOOLS = {
    "pin_down": (True, ("core.diagnostics.identifiability", "identifiability_jacobian"), {"t_obs": "2.5"},
                 {"t_obs_s": 2.5, "m": 32, "m_noise": 128, "rel": 0.02, "seed": 0},
                 {"t_obs": "4", "m": "20", "m_noise": "50", "rel": "0.05", "seed": "9"},
                 {"t_obs_s": 4.0, "m": 20, "m_noise": 50, "rel": 0.05, "seed": 9}),
    "fixed_number": (True, ("core.diagnostics.sbc", "sbc_repeats"), {},
                     {"chi_k_fixed": None, "repeats": 10, "n_cal": 2000, "cal_n_scales": 200,
                      "num_posterior_samples": 1000, "seed": 0},
                     {"chi_k_fixed": "3", "repeats": "4", "n_cal": "500", "cal_n_scales": "50",
                      "num_posterior_samples": "200", "seed": "8"},
                     {"chi_k_fixed": 3, "repeats": 4, "n_cal": 500, "cal_n_scales": 50, "num_posterior_samples": 200,
                      "seed": 8}),
    "drive_study": (True, ("core.diagnostics.probes", "probe_drive"), {},
                    {"t_obs_s": 5.0, "repeats": 16, "detune": 1.4, "strengths": None, "seed": 0},
                    {"drive_t_obs": "3", "drive_repeats": "8", "drive_detune": "1.6", "drive_strengths": "0.1 0.5 2",
                     "probe_seed": "4"},
                    {"t_obs_s": 3.0, "repeats": 8, "detune": 1.6, "strengths": [0.1, 0.5, 2.0], "seed": 4}),
    "across_prior": (False, ("core.diagnostics.identifiability", "identifiability_laplace"), {"t_obs": "2.5"},
                     {"t_obs_s": 2.5, "n_points": 6, "m": 32, "m_noise": 128, "rel": 0.02, "seed": 0},
                     {"t_obs": "3", "n_points": "4", "m": "16", "m_noise": "40", "rel": "0.01", "seed": "2"},
                     {"t_obs_s": 3.0, "n_points": 4, "m": 16, "m_noise": 40, "rel": 0.01, "seed": 2}),
}


def _typed_into(form, values) -> None:
    """Type ``values`` into ``form``, by field key: a fixed number of driven recordings unticked first."""
    for setting, text in values.items():
        widget = form.rows[setting].widget
        if hasattr(widget, "ticked_box"):
            widget.ticked_box.setChecked(False)
            widget = widget.field
        widget.setText(text)


@pytest.mark.parametrize("key", list(_HANDED_TOOLS))
def test_each_informing_check_and_tool_hands_its_stage_the_forms_arguments(key, tmp_path, monkeypatch):
    """A press of an informing check or a tool hands its own stage, by the stage's own names, what its form holds: at
    the usual values, but what must be typed, and at changed ones, with the record's name and note and the figures'
    sink asked for; what the observations can pin down and the drive study on a configuration of their own (the drive
    study's without the frequency response, as its stage drives the cell itself), the calibration repeated on the
    session's with the posterior and the prior, identifiability across the prior on its own with the posterior; no
    setting of the engine is written."""
    from core import config
    from core.gui.panels.inference import check_forms
    seen = _tripwire(monkeypatch)
    chi, where, needed, usual, typed, changed = _HANDED_TOOLS[key]
    settings = {name: repr(value) for name, value in vars(config).items() if name.isupper()}
    with _checks_page(tmp_path, chi=chi, simulation=SET) as (_window, screen, page, _store):
        session = screen.session
        _ready(page, key)
        form = page.check_pages[key].form
        _typed_into(form, needed)
        for values, wanted in ((None, usual), (typed, changed)):
            if values is not None:
                _typed_into(form, values)
                form.names.name.setText("a_tool")
            page._main_clicked()
            assert seen.get("fn") is _engine(where), (key, page.foot.reason.text())
            handed = {name: value for name, value in seen["kwargs"].items() if name not in _CONTROLLERS}
            assert handed == {**wanted, "name": "a_tool" if values is not None else "", "note": ""}, (key, handed)
            assert seen["kwargs"]["spec"].check == {"key": key, "sizes": check_forms.sizes_of(key, form)}, key
            assert seen["kwargs"]["provide_fig_sink"] is True
            made, *rest = seen["args"]
            if key == "fixed_number":
                assert made is session.cfg and rest == [session.posterior, session.inf_prior]
            else:
                assert made is not session.cfg and made.sources.get("cell") is not None, key
                assert made.chi_mode is (False if key == "drive_study" else chi), key
                assert rest == ([session.posterior] if key == "across_prior" else []), key
            seen.pop("fn")
    assert {name: repr(value) for name, value in vars(config).items() if name.isupper()} == settings


@pytest.mark.parametrize("key", ["pin_down", "across_prior"])
def test_the_recording_length_of_the_two_identifiability_checks_opens_blank_and_blocks(key, tmp_path, monkeypatch):
    """The recording length of what the observations can pin down and of identifiability across the prior opens blank,
    says "Type a recording length." under its field and blocks, so that the stage's own fallback is never used; a length
    typed lets the press through."""
    seen = _tripwire(monkeypatch)
    with _checks_page(tmp_path, chi=key == "pin_down") as (_window, _screen, page, _store):
        _ready(page, key)
        row = page.form.rows["t_obs"]
        assert row.widget.text() == "" and row.unit.text() == "s"
        assert (row.line.kind, row.line.words) == ("err", "Type a recording length.")
        assert not page.foot.main_button.isEnabled()
        assert page.foot.reason.text() == "Cannot start yet: a setting cannot be used."
        page._main_clicked()
        assert seen["calls"] == 0
        row.widget.setText("2.5")
        page._main_clicked()
        assert seen["calls"] == 1 and seen["kwargs"]["t_obs_s"] == 2.5


def test_mixed_as_in_training_sends_no_fixed_number_and_a_number_is_held_to_the_slots(tmp_path, monkeypatch):
    """The fixed number of driven recordings opens ticked "Mixed, as in training", which sends none; unticked, a whole
    number from 1 to the slots of the configuration in use, said under its field otherwise, and blocked; a number with
    no frequency response in use is greyed, which waiting will not lift."""
    seen = _tripwire(monkeypatch)
    with _checks_page(tmp_path) as (_window, screen, page, _store):
        _ready(page, "fixed_number")
        row = page.form.rows["chi_k_fixed"]
        assert row.widget.is_ticked() and row.widget.ticked_box.text() == "Mixed, as in training"
        page._main_clicked()
        assert seen["kwargs"]["chi_k_fixed"] is None
        slots = screen.session.cfg.chi_k_pad
        for text, line in (("0", f"Fixed number of driven recordings must be a whole number from 1 to {slots}."),
                           (str(slots + 1), f"Fixed number of driven recordings must be a whole number from 1 to "
                                            f"{slots}."), (str(slots), "")):
            _typed_into(page.form, {"chi_k_fixed": text})
            page.redraw()
            assert row.line.words == line, (text, row.line.words)
            assert page.foot.main_button.isEnabled() == (not line), text
        page._main_clicked()
        assert seen["kwargs"]["chi_k_fixed"] == slots
    with _checks_page(tmp_path / "without", chi=False) as (_window, _screen, page, _store):
        _ready(page, "fixed_number")
        assert page.foot.main_button.isEnabled(), page.foot.reason.text()
        _typed_into(page.form, {"chi_k_fixed": "2"})
        page.redraw()
        assert page.foot.reason.text() == "Cannot start: it applies only with the frequency response."


@pytest.mark.parametrize("key, chi", [("pin_down", True), ("fixed_number", True), ("across_prior", False)])
def test_every_seed_a_stage_does_not_refuse_is_held_to_the_one_rule_of_a_seed(key, chi, tmp_path, monkeypatch):
    """A seed no stage refuses itself is held by the window to the engine's one rule of a seed, called, not copied:
    the largest it takes passes (for the repeated calibration, the largest its last repeat's seed leaves), and the random
    generator runs at the largest; one beyond, one below 0 and a number with a fraction are said under the field, in the
    window's words from its label, and block; nothing is dispatched."""
    import torch
    from core.rng import SEED_MAX, seeded
    seen = _tripwire(monkeypatch)
    with _checks_page(tmp_path, chi=chi, simulation=SET) as (_window, _screen, page, _store):
        _ready(page, key)
        _typed_into(page.form, {"t_obs": "2.5"} if key != "fixed_number" else {})
        row = page.form.rows["seed"]
        bound = SEED_MAX - (page.form.rows["repeats"].value() - 1 if key == "fixed_number" else 0)
        for text, line in ((str(bound + 1), f"Seed must be a whole number from 0 to {bound}."),
                           ("-1", f"Seed must be a whole number from 0 to {bound}."),
                           ("2.5", "Seed must be a whole number; got '2.5'.")):
            row.widget.setText(text)
            page.redraw()
            assert (row.line.kind, row.line.words) == ("err", line), (text, row.line.words)
            page._main_clicked()
            assert seen["calls"] == 0, text
        largest = SEED_MAX - (page.form.rows["repeats"].value() - 1 if key == "fixed_number" else 0)
        row.widget.setText(str(largest))
        page._main_clicked()
        assert seen["calls"] == 1 and seen["kwargs"]["seed"] == largest
    with seeded(SEED_MAX, torch.device("cpu")):
        torch.rand(1)


def test_the_repeated_calibrations_stage_line_says_no_progress_is_reported():
    """The calibration repeated at fixed numbers has one stage, "Measuring", said in other words while it ranks a
    repeat: from the line that ends a repeat's simulation to the line that ends the repeat, which repeat of how many,
    that no progress is reported then and that a stop waits for its end; between two, its own name again."""
    from core.gui.panels.inference import check_forms
    from core.gui.run.model import RunModel
    from tests._runs import CALIBRATION_LINES
    spec = check_forms.check_spec("fixed_number", sizes={"repeats": 10, "n_cal": 2000})
    model = RunModel(spec, clock=lambda: _at(8))
    simulated = CALIBRATION_LINES[1]
    ranking = ("Ranking repeat {} of 10: no progress is reported during this part, and Stop takes effect when it "
               "ends")
    said = []
    for repeat in (1, 2, 3):
        model.feed([("[sbc] probe-count stratum: POOLED over the training mixture", "info")] if repeat == 1 else [])
        said.append(model.doing())
        model.feed([(simulated, "info")])
        said.append(model.doing())
        model.feed([(f"[sbc] repeat {repeat}/10: n_valid=1995  worst KS p=0.0123", "info")])
    said.append(model.doing())
    assert said == ["Measuring", ranking.format(1), "Measuring", ranking.format(2), "Measuring", ranking.format(3),
                    "Measuring"], said
    assert [stage.name for stage in spec.stages] == ["Measuring"]


def test_the_drive_studys_forcing_lines_can_be_copied(tmp_path):
    """The drive study's result shows the Forcing sections its record suggests, as lines a cell file and a bounds file
    take, in the fixed face and selectable, with "Copy" beside them, which puts them on the clipboard exactly; the
    clipboard is put back as it was."""
    from core.gui import fonts
    from tests._fixtures import check_results, write_diagnostic
    app = qt_app()
    suggested = {"free_running": {"amp": 0.25, "freq": 0.0316, "phase": 0.0, "offset": 0.0},
                 "captured": {"amp": 1.5, "freq": 0.0316, "phase": 0.0, "offset": 0.0},
                 "box": {"amp": [0.0, 4.5], "freq": [0.003, 0.3], "phase": [0.0, 6.28319], "offset": [-50.0, 50.0]}}

    def records(store, cfg, prior, posterior):
        body = check_results("drive_study")
        body["results"]["suggested_forcing"] = suggested
        write_diagnostic(store, cfg, diagnostic="probes", variant="drive", cell=str(_cell_file()), **body)

    clipboard = app.clipboard()
    before = clipboard.text()
    try:
        with _checks_page(tmp_path, records=records) as (_window, _screen, page, _store):
            page.open_check("drive_study")
            region = page.check_pages["drive_study"].result
            lines = ("# Forcing Parameters: the strongest drive that keeps the cell's own oscillation\namp = 0.25\n"
                     "freq = 0.0316\nphase = 0\noffset = 0\n\n"
                     "# Forcing Parameters: the weakest drive that captures the cell\namp = 1.5\nfreq = 0.0316\n"
                     "phase = 0\noffset = 0\n\n"
                     "# Forcing Parameters of a bounds file\namp in (0, 4.5)\nfreq in (0.003, 0.3)\n"
                     "phase in (0, 6.28319)\noffset in (-50, 50)")
            assert region.lines.text() == lines and region.lines.isVisibleTo(page)
            assert region.lines.font().fixedPitch() and region.lines.font().families() == fonts.fixed().families()
            assert region.lines.textInteractionFlags() & Qt.TextSelectableByMouse
            assert region.copy_button.text() == "Copy" and region.copy_button in region.stops()
            region.copy_button.click()
            assert clipboard.text() == lines
    finally:
        clipboard.setText(before)


def test_an_informing_check_or_a_tool_reads_done_once_run(tmp_path):
    """Each of the four reads "Done" once a record of it counts, never "Passed", and its button "Run again"."""
    from tests._fixtures import check_results, write_diagnostic
    keys = ("pin_down", "fixed_number", "drive_study", "across_prior")

    def records(store, cfg, prior, posterior):
        for key in keys:
            one = check_catalogue.entry(key)
            parents = {"posterior": posterior, "prior": prior} if key in ("fixed_number", "across_prior") else None
            write_diagnostic(store, cfg, diagnostic=one.diagnostic, variant=one.variant or "pooled",
                             cell=str(_cell_file()), **({"parents": parents} if parents else {}),
                             **check_results(key))

    for chi, held in ((True, keys[:3]), (False, keys[3:])):
        with _checks_page(tmp_path / str(chi), chi=chi, records=records) as (_window, screen, page, _store):
            for key in held:
                assert _row(screen.check_rows(), key).state == "Done", (key, _row(screen.check_rows(), key))
                page.open_check(key)
                assert page.foot.main_button.text().replace("&", "") == "Run again", key


def test_a_tools_greyed_reasons_follow_the_model_the_mode_and_the_bounds_file(tmp_path):
    """On the page, an informing check's or a tool's button is greyed for what it cannot apply to, each beside it as
    "Cannot start:", which waiting will not lift: what the observations can pin down of a model not Nadrowski's, or with
    neither the frequency response nor a forcing section; the drive study and identifiability across the prior with no
    forcing section; identifiability across the prior with the frequency response; a fixed number of driven recordings
    without it."""
    cases = (("pin_down", dict(nadrowski=False), "Cannot start: it applies only to the Nadrowski model."),
             ("pin_down", dict(chi=False, forcing=False), "Cannot start: the bounds file has no forcing section."),
             ("drive_study", dict(forcing=False), "Cannot start: the bounds file has no forcing section."),
             ("across_prior", dict(chi=True), "Cannot start: it does not apply with the frequency response."),
             ("across_prior", dict(chi=False, forcing=False), "Cannot start: the bounds file has no forcing section."))
    with _checks_page(tmp_path, chi=False) as (_window, screen, page, _store):
        real = screen._checks_in_use
        for key, over, beside in cases:
            screen._checks_in_use = lambda *args, over=over, **kwargs: replace(real(*args, **kwargs), **over)
            _ready(page, key)
            _typed_into(page.form, {"t_obs": "2.5"} if "t_obs" in page.form.rows else {})
            page.redraw()
            assert page.foot.reason.text() == beside, (key, over, page.foot.reason.text())
        screen._checks_in_use = real
        _ready(page, "fixed_number")
        _typed_into(page.form, {"chi_k_fixed": "2"})
        page.redraw()
        assert page.foot.reason.text() == "Cannot start: it applies only with the frequency response."


# ── the forms' rules, the audit's greying and the result's width ──
#: The Forcing sections a drive study's record suggests, as a record of the engine holds them.
_SUGGESTED = {"free_running": {"amp": 0.25, "freq": 0.0316, "phase": 0.0, "offset": 0.0},
              "captured": {"amp": 1.5, "freq": 0.0316, "phase": 0.0, "offset": 0.0},
              "box": {"amp": [0.0, 4.5], "freq": [0.003, 0.3], "phase": [0.0, 6.28319], "offset": [-50.0, 50.0]}}


def _a_drive_study(store, cfg, prior, posterior) -> None:
    """A drive study's record that suggests ``_SUGGESTED``."""
    from tests._fixtures import check_results, write_diagnostic
    body = check_results("drive_study")
    body["results"]["suggested_forcing"] = _SUGGESTED
    write_diagnostic(store, cfg, diagnostic="probes", variant="drive", cell=str(_cell_file()), **body)


def test_the_audit_is_greyed_while_its_record_could_not_count(tmp_path, monkeypatch):
    """With a posterior in use trained on another prior than the prior in use, the audit's button is greyed with the
    very reason the calibration gives in that state, by the same reading of the engine's guard, the posterior's own
    prior named: its row counts only audits of that prior, and a press would audit the prior in use. With no posterior
    in use it audits the prior in use."""
    from core.diagnostics import probes
    from tests._fixtures import _prior_artifact
    seen = _tripwire(monkeypatch)
    kept = {}

    def records(store, cfg, prior, posterior):
        kept["older"] = store.load_prior(cfg, _prior_artifact(store, cfg, name="older_prior", seed=5).id)

    sentence = "The prior in use is not the one this posterior was trained on (older_prior). Load it in step 2."
    with _checks_page(tmp_path, records=records) as (_window, screen, page, _store):
        screen.session.posterior = _held_posterior(tmp_path, prior=kept["older"])
        page.read_facts()
        for key in ("audit", "calibration"):
            page.open_check(key)
            assert page.foot.reason.text() == "Cannot start yet: the prior in use is not the posterior's own.", key
            assert sentence in [fact.sentence for fact in page.facts_from_form()], key
        page.open_check("audit")
        page._main_clicked()
        assert seen["calls"] == 0
        screen.session.posterior = None
        page.read_facts()
        page.open_check("audit")
        assert page.foot.main_button.isEnabled(), page.foot.reason.text()
        page._main_clicked()
        assert seen["fn"] is probes.probe_mask and seen["args"][1] is screen.session.inf_prior


@pytest.mark.parametrize("factor", [1.0, 1.5])
def test_the_drive_studys_result_stands_inside_the_page_at_the_least_size(factor, tmp_path):
    """At the window's least size, at the ordinary text and at one and a half times it, the drive study's result holds
    its page's width: the body is no wider than what shows it, and the lines to copy and "Copy" stand wholly inside it,
    a line too long for the column wrapped on the page and copied whole."""
    from PySide6.QtCore import QPoint, QRect
    from core.gui import design
    with text_size(factor):
        with _checks_page(tmp_path, records=_a_drive_study, size=(design.scaled(900), design.scaled(600))) as (
                _window, _screen, page, _store):
            page.open_check("drive_study")
            _settle()
            region = page.check_pages["drive_study"].result
            port = page.scroller.viewport()
            assert region is not None and region.lines.isVisibleTo(page)
            assert page.scroller.widget().width() <= port.width(), (page.scroller.widget().width(), port.width())
            assert page.scroller.horizontalScrollBar().maximum() == 0
            for part in (region.lines, region.copy_button):
                edge = QRect(part.mapTo(port, QPoint(0, 0)), part.size())
                assert 0 <= edge.left() and edge.right() < port.width(), (type(part).__name__, edge, port.width())
            assert region.lines.text().split("\n")[0] == ("# Forcing Parameters: the strongest drive that keeps the "
                                                          "cell's own oscillation")


def test_a_list_of_numbers_holds_numbers_and_a_size_in_error_says_no_count(tmp_path, monkeypatch):
    """Only a truly blank list of numbers means the engine's own choice: one that holds separators and no number is a
    bad list, said under its field, and blocks; "2,5" is two numbers. While a size a count reads is in error, "Before you
    start" says no count of it: the line under its field says what is wrong."""
    seen = _tripwire(monkeypatch)
    with _checks_page(tmp_path) as (_window, _screen, page, _store):
        _ready(page, "band")
        lengths = page.form.rows["probe_lengths"]
        for text in (",", " , ", ", ,"):
            lengths.widget.setText(text)
            page.redraw()
            assert lengths.line.words == "Recording lengths must be numbers separated by commas or spaces.", text
            assert not [fact for fact in page.facts_from_form() if fact.sentence.startswith("Simulates")], text
            assert not page.foot.main_button.isEnabled(), text
        lengths.widget.setText("2,5")
        page.redraw()
        assert ("Simulates 24 recordings at each recording length and frequency of the band: 2 recording lengths."
                in [fact.sentence for fact in page.facts_from_form()])
        page._main_clicked()
        assert seen["kwargs"]["lengths"] == [2.0, 5.0]
        _ready(page, "calibration")
        page.form.rows["n_cal"].widget.setText("0")
        page.redraw()
        assert page.form.rows["n_cal"].line.words == "Test datasets must be at least 1."
        assert not [fact for fact in page.facts_from_form() if fact.sentence.startswith("Simulates")]


@pytest.mark.parametrize("raised, told", [(OSError("no such file"), "bounds"), (ValueError("not a cell"), "cell"),
                                          (RuntimeError("a fault of the program"), "bug")])
def test_a_checks_own_configuration_that_fails_is_told_as_what_failed(raised, told, tmp_path, monkeypatch):
    """A check's own configuration that cannot be made is told as what failed: a bounds file or a cell file that cannot be
    read, as a refusal of its field, "Check your inputs" with where to fix it; any other failure as the fault of the
    program it is, as every page tells one; nothing is dispatched either way."""
    from core import cli
    from core.gui import fields as gui_fields
    from core.gui.session import ConfigDraft
    from tests._dialogs import asked

    def failing(*_args, **_kwargs):
        raise raised

    seen = _tripwire(monkeypatch)
    with _checks_page(tmp_path) as (_window, _screen, page, _store):
        _ready(page, "band")
        monkeypatch.setattr(cli if told == "cell" else ConfigDraft,
                            "load_and_validate_gt" if told == "cell" else "make_config", failing)
        page._main_clicked()
        box = asked()
        if told == "bug":
            assert (box.kind, box.title, box.marker) != ("refusal", "Check your inputs", "warn"), box
            assert "a fault of the program" in box.sentence, box
        else:
            what = "bounds file" if told == "bounds" else "cell file"
            assert (box.kind, box.title) == ("refusal", "Check your inputs"), box
            assert box.sentence.startswith(f"The {what} cannot be read: {type(raised).__name__}: "), box.sentence
            assert box.beneath == gui_fields.fix_sentence(told, here="checks"), box.beneath
        assert seen["calls"] == 0


def test_before_you_start_stands_in_its_order_on_a_checks_page(tmp_path):
    """A check's "Before you start" in its order: what the run does, its time, the memory lines, that nothing else can
    run meanwhile and what it writes; the earlier steps' changes not applied; the name's and the note's facts; the
    greyed reasons; then a setting in error; and last what it says of the plan in use."""
    from core.gui.widgets.form import NAME_SHAPE_TEXT
    with _checks_page(tmp_path, posterior=False) as (_window, screen, page, _store):
        page.open_check("calibration")
        prior_page = screen.prior_panel
        prior_page.bounds_picker.restore_key("master_tier1.txt")
        prior_page._edited()
        page.form.names.name.setText("not a name!")
        page.form.names.check_now()
        page.form.rows["n_cal"].widget.setText("0")
        page.redraw()
        facts = page.facts_from_form()
        assert [fact.kind for fact in facts] == ["info"] * 4 + ["warn", "err", "lock", "err", "info"], facts
        assert (facts[0].sentence, facts[2].sentence, facts[3].sentence) == (
            "Not timed on this computer yet.", "Nothing else can run until it ends.", "One record, under Records.")
        assert facts[1].sentence.endswith("No run on this computer has been recorded yet."), facts[1].sentence
        assert facts[4].sentence.startswith("Step 2, Prior, has changes that are not applied.")
        assert (facts[5].sentence, facts[6].sentence, facts[7].sentence) == (
            NAME_SHAPE_TEXT, "Needs a posterior: train one in step 3, or load a saved one.",
            "Test datasets must be at least 1.")
        assert facts[8].sentence.startswith("No plan."), facts[8].sentence


def test_before_the_history_is_read_no_time_is_said(tmp_path):
    """Until the page has read this computer's run history, "Before you start" says nothing of how long a check took:
    neither a time nor that it was not timed; once read, it says one."""
    with _checks_page(tmp_path, shown=False) as (_window, _screen, page, _store):
        page.open_check("calibration")
        page.redraw()
        said = [fact.sentence for fact in page.facts_from_form()]
        assert not [one for one in said if "here last time" in one or one.startswith("Not timed")], said
        page.read_facts()
        assert "Not timed on this computer yet." in [fact.sentence for fact in page.facts_from_form()]


def test_a_checks_page_says_the_memory_lines_for_a_check_on_the_card_alone(tmp_path):
    """On a check's page, "Before you start" says the memory limit in force and the last run on this computer, as the
    page's reading found them, for a check that simulates on the graphics card, and nothing of them for one that reads
    records and simulates nothing."""
    from core.gui.memory_limit import limit_words, memory_limit_in_force
    from core.gui.run.ends import NO_LAST_RUN_TEXT
    memory = f"{limit_words(memory_limit_in_force())} {NO_LAST_RUN_TEXT}"
    with _checks_page(tmp_path, simulation=SET) as (_window, _screen, page, _store):
        for key, card in (("calibration", True), ("input_use", False)):
            page.open_check(key)
            page.redraw()
            assert (memory in [fact.sentence for fact in page.facts_from_form()]) == card, (key, page.facts_from_form())


def test_no_widget_a_checks_reading_or_a_runs_end_makes_is_shown_as_a_window_of_its_own(tmp_path, monkeypatch):
    """STANDING, beside the builders' guard. What the Checks page makes after it is built, every check's result at a
    reading with a record of each kind and again at a check's run's end, is made inside its page: none is shown as a
    window of its own, but the window itself and the questions the dialogs' guard records."""
    from PySide6.QtWidgets import QMessageBox, QWidget
    from core import orchestrator
    from core.gui.main_window import MainWindow
    from core.gui.widgets.ask import Ask
    from tests._runs import StandIn, turn_until
    from tests._window_helpers import shown_off_screen
    stand_in = StandIn(word="Check")
    monkeypatch.setattr(orchestrator, "validate_calibration", stand_in)
    app = qt_app()
    shown = []
    real = {name: getattr(QWidget, name) for name in ("setVisible", "show", "setHidden")}

    def heard(widget, on: bool) -> None:
        if on and widget.isWindow() and not isinstance(widget, (MainWindow, Ask, QMessageBox)):
            words = getattr(widget, "text", None)
            shown.append((type(widget).__name__, words() if callable(words) else "", widget.parent() is None))

    with _checks_page(tmp_path, records=_every_record, shown=False) as (window, _screen, page, _store):
        monkeypatch.setattr(QWidget, "setVisible", lambda widget, on: (heard(widget, on), real["setVisible"](widget, on)))
        monkeypatch.setattr(QWidget, "show", lambda widget: (heard(widget, True), real["show"](widget)))
        monkeypatch.setattr(QWidget, "setHidden", lambda widget, on: (heard(widget, not on), real["setHidden"](widget, on)))
        shown_off_screen(window, 1300, 820)
        window.go_to("inference", "checks")           # the page's first reading, which makes every result
        _settle()
        made = sum(1 for one in page.check_pages.values() if one.result is not None)
        _ready(page, "calibration")
        page._main_clicked()
        assert stand_in.started.wait(5)
        stand_in.release()
        turn_until(app, lambda: not controller().is_live())
        _settle()
        for name, method in real.items():
            monkeypatch.setattr(QWidget, name, method)
    assert made >= 6, made
    assert shown == [], f"{len(shown)} widgets were shown as windows of their own: {shown}"


# ── the two checks that keep nothing but their result ──
@pytest.mark.parametrize("ending", ["finished", "failed", "stopped", "refused once", "never goes"])
def test_the_scratch_folder_is_removed_however_the_check_ends(ending, tmp_path, monkeypatch, caplog):
    """A scratch store is a folder of its own among the computer's temporary files, and goes however its block ends:
    finished, failed, or stopped as a run is stopped; a removal Windows refuses once is tried again after its pause
    and the folder goes; one that never goes is tried as often as the rule says, then said once, as a warning that
    names the folder, and the block's own end passes on unchanged."""
    import shutil
    import tempfile
    from core.gui import check_scratch
    from core.gui.streams import WorkerCancelled
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    real, tries, pauses = shutil.rmtree, [], []

    def rmtree(path, *args, **kwargs):
        tries.append(Path(path))
        if ending == "never goes" or (ending == "refused once" and len(tries) == 1):
            raise PermissionError(13, "held open by another program", str(path))
        real(path, *args, **kwargs)
    monkeypatch.setattr(shutil, "rmtree", rmtree)
    raised = {"failed": RuntimeError("the fit broke"), "stopped": WorkerCancelled()}.get(ending)
    seen = {}
    with caplog.at_level("WARNING", logger="core.gui.check_scratch"):
        try:
            with check_scratch.scratch_store(pause=pauses.append) as store:
                seen["folder"] = folder = Path(store.root)
                assert folder.parent == tmp_path and folder.name.startswith(check_scratch.SCRATCH_PREFIX), folder
                (folder / "posteriors").mkdir()
                (folder / "posteriors" / "posterior.pt").write_bytes(b"held")
                if raised is not None:
                    raise raised
        except (RuntimeError, WorkerCancelled) as caught:
            assert caught is raised
    folder = seen["folder"]
    said = [record for record in caplog.records if record.name == "core.gui.check_scratch"]
    if ending == "never goes":
        assert folder.exists() and len(tries) == check_scratch.RMTREE_TRIES, tries
        assert pauses == [check_scratch.RMTREE_PAUSE_S] * (check_scratch.RMTREE_TRIES - 1), pauses
        assert [record.levelname for record in said] == ["WARNING"] and str(folder) in said[0].getMessage()
        real(folder)
    else:
        assert not folder.exists() and said == [], said
        assert (len(tries), pauses) == ((2, [check_scratch.RMTREE_PAUSE_S]) if ending == "refused once" else (1, []))


def _a_posterior(created="2026-10-08T19:11:14+00:00", epochs=10):
    """What the training stage hands back, as a short fit reads it: the moment its record was made and its epochs."""
    return SimpleNamespace(manifest=SimpleNamespace(created=created, body={"training": {"epochs_trained": epochs}}))


def _said_at(text, moment) -> None:
    """``text`` said by the engine's logger as a record made at ``moment``, seconds since the epoch."""
    import logging
    logger = logging.getLogger("core")
    record = logger.makeRecord("core.SBI.pipeline", logging.INFO, __file__, 0, text, (), None)
    record.created = moment
    logger.handle(record)


@contextlib.contextmanager
def _a_store_of_its_own(tmp_path, made):
    """A store standing in for the scratch store, which ``made`` keeps."""
    from core.artifacts import ArtifactStore
    store = ArtifactStore(tmp_path / f"scratch_{len(made)}")
    made.append(store)
    yield store


def test_the_short_fit_hands_build_posterior_exactly_these_arguments(tmp_path, monkeypatch):
    """A short fit makes one call of the training stage, in the store it is handed: the configuration, the prior in use
    handed as the object, no record named and a new training; the batches, the simulations per batch and the network as
    handed, an epoch limit one less than its epochs, its simulations not kept, so that no saved set is read or written
    and no training is kept as it goes, and the figures' sink; nothing else. It hands back its result alone, with the
    press that started it, and names no record."""
    from core import orchestrator
    from core.gui import check_scratch
    seen, made = [], []
    monkeypatch.setattr(orchestrator, "build_posterior", lambda *args, **kwargs: (seen.append((args, kwargs)),
                                                                                  _a_posterior())[1])
    cfg, prior, sink = object(), object(), object()
    fit = check_scratch.short_fit(cfg, prior, num_runs=20, run_size_cap=512, epochs=10, hidden_features=256,
                                  num_transforms=10, press="a1b2", fig_sink=sink,
                                  store_factory=lambda: _a_store_of_its_own(tmp_path, made))
    (args, kwargs), = seen
    assert args == (cfg, prior, None, True) and args[1] is prior, args
    assert kwargs == {"store": made[0], "num_runs": 20, "run_size_cap": 512, "hidden_features": 256,
                      "num_transforms": 10, "max_num_epochs": 9, "checkpoint_every": 0, "resume": "auto",
                      "fig_sink": sink}, kwargs
    assert fit.check_result["press"] == "a1b2" and fit.check_result["network"] == [256, 10]
    assert [getattr(fit, part, None) for part in ("kind", "id", "name")] == [None, None, None]


def test_the_short_fits_span_runs_from_the_end_of_generation_to_the_posteriors_created(tmp_path, monkeypatch):
    """The span runs from the first record that ends the generation of the training's rows (the total of failed
    simulations, or of unusable driven recordings) to the moment the posterior's record was made; seconds per epoch is
    the span over the epochs trained, and the median epoch the median of the epochs' own seconds; the simulations are
    the budget's. A record not made gives no span, and no epoch trained no seconds per epoch: None, and nothing
    raises."""
    from core import orchestrator
    from core.gui import check_scratch
    from tests._runs import BUDGET, epoch_line, masked_line
    began = datetime(2026, 10, 8, 19, 10, 0, tzinfo=timezone.utc).timestamp()
    handed = {}

    def build(*_args, **_kwargs):
        _said_at(BUDGET, began - 600)
        _said_at("[patho] run total: 0 pathological of 40,960 simulated trajectories (0.0000%) -- 0 non-finite, 0 "
                 "exactly constant, 0 over 1e+15", began)
        _said_at(masked_line(410, 1000, 20, WHOLE), began + 1)
        for epoch, took in enumerate((7.0, 7.4, 9.0), start=1):
            _said_at(epoch_line(epoch, took=took), began + 10 * epoch)
        return handed["posterior"]
    monkeypatch.setattr(orchestrator, "build_posterior", build)

    def fit(**changed):
        handed["posterior"] = _a_posterior(**changed)
        return check_scratch.short_fit(None, None, num_runs=20, run_size_cap=512, epochs=10, hidden_features=256,
                                       num_transforms=10, store_factory=lambda: _a_store_of_its_own(tmp_path, []))
    made = fit(created="2026-10-08T19:11:14+00:00", epochs=10).check_result
    assert made == {"network": [256, 10], "simulations": 20_480_000, "epochs": 10, "span": pytest.approx(74.0),
                    "per_epoch": pytest.approx(7.4), "median_epoch": 7.4, "press": ""}, made
    assert fit(created=None, epochs=10).check_result | {"press": ""} == {
        **made, "span": None, "per_epoch": None}
    assert fit(created="2026-10-08T19:11:14+00:00", epochs=0).check_result["per_epoch"] is None


def _step_three(screen, width, depth) -> None:
    """Step 3's form holding the network ``width`` by ``depth``."""
    rows = screen.posterior_panel.form.rows
    rows["hidden_features"].widget.setText(str(width))
    rows["num_transforms"].widget.setText(str(depth))


@pytest.mark.parametrize("ending", ["two networks", "the same network", "stopped", "failed", "refused",
                                    "a run live by then", "a stop taken too late", "a close"])
def test_the_timing_is_two_runs_one_per_network_and_one_when_they_are_equal(ending, tmp_path, monkeypatch):
    """One press of the timing of a short fit starts a run for step 3's network and, once that run has finished and its
    end has been published and drawn, one turn later, a second for the built-in network, each with its own history,
    the two at equal sizes and with one press in their results; one run when step 3's network is the built-in one.
    A first run stopped, failed or refused starts no second; nor does one asked to stop that finished all the same, the
    stop taken too late, nor one the window's close asked to stop; nor one whose end finds another run live by the
    turn its second would start in. The page holds no configuration and no prior of the press once no fit of it is to
    come."""
    import tempfile
    from core import orchestrator
    from core.gui.run import history
    from core.gui.run.model import RunSpec
    from core.refusals import Refusal
    from tests._dialogs import answer_in_turn, asked
    from tests._fixtures import build_run_host
    from tests._runs import Clock, StandIn, at, controller_on, start_run, turn_until
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    late = ending in ("a stop taken too late", "a close")
    stand_in, other = StandIn(word="Check", takes_stops=not late), StandIn(word="Run")
    monkeypatch.setattr(orchestrator, "build_posterior", stand_in)
    app = qt_app()
    order = []
    held = controller()

    def heard(status) -> None:
        order.append(("ended", status.word, status.outcome))
        if ending == "a run live by then" and len(order) == 2:
            start_run(build_run_host(), other, spec=RunSpec("Run", None))

    try:
        with _checks_page(tmp_path) as (_window, screen, page, store):
            # heard after the page's own, so that another run starts between its end and the turn its second would
            # start in
            held.run_ended.connect(heard)
            clock = Clock(at(8, 12, 5))
            controller_on(tmp_path, clock)
            if ending != "the same network":
                _step_three(screen, 256, 10)
            real = page._time_a_fit
            page._time_a_fit = lambda network: (order.append(("fit", tuple(network), held.status.state)),
                                                real(network))[1]
            page.open_check("timing")
            page._main_clicked()
            assert stand_in.started.wait(5)
            if ending == "stopped":
                answer_in_turn(monkeypatch, ("Stop and discard it", None))
                page.run_view.stop_button.click()
            elif ending == "failed":
                stand_in.fail(RuntimeError("the fit broke"))
            elif ending == "refused":
                stand_in.fail(Refusal("The number of training batches must be at least 1; got 0.", field="num_runs"))
            elif late:
                held.request_stop(closing=ending == "a close")
                turn_until(app, lambda: held.status.stop_requested)
                stand_in.release(_a_posterior())
            else:
                clock.advance(90)
                stand_in.release(_a_posterior())
            if ending == "two networks":
                turn_until(app, lambda: len(stand_in.calls) == 2, 10.0)
                stand_in.release(_a_posterior())
            if ending == "a run live by then":
                turn_until(app, lambda: other.started.is_set(), 10.0)
                _settle()                            # the turn the second would have started in has passed
                other.release(None)
            turn_until(app, lambda: not held.is_live(), 10.0)
            _settle()
            if ending == "refused":
                assert asked().kind == "refusal"
            runs = [one for one in history.runs_of(tmp_path / "runs", records_root=str(store.root))
                    if (one.get("check") or {}).get("key") == "timing"]
            kept = [name for name, value in vars(page).items() if value is screen.session.inf_prior or (
                isinstance(value, dict) and any(isinstance(one, tuple) and screen.session.inf_prior in one
                                                for one in value.values()))]
    finally:
        held.run_ended.disconnect(heard)
    assert kept == [], f"the page holds the press's prior: {kept}"
    networks = [(kwargs["hidden_features"], kwargs["num_transforms"]) for _args, kwargs in stand_in.calls]
    if ending == "two networks":
        assert networks == [(256, 10), (128, 8)], networks
        assert order == [("fit", (256, 10), "idle"), ("ended", "Check", "finished"), ("fit", (128, 8), "ended"),
                         ("ended", "Check", "finished")], order
        assert [one["check"]["result"]["network"] for one in runs] == [[256, 10], [128, 8]]
        assert len({one["check"]["result"]["press"] for one in runs}) == 1 and runs[0]["check"]["result"]["press"]
        assert runs[0]["check"]["sizes"] == runs[1]["check"]["sizes"] == {
            "num_runs": 20, "epochs": 10, "run_size_cap": 0, "network": [[256, 10], [128, 8]]}
    else:
        assert networks == [(128, 8) if ending == "the same network" else (256, 10)], networks
        assert [step for step in order if step[0] == "fit"] == [order[0]], order
        assert len(runs) == (0 if ending == "refused" else 1), runs
        if late:
            assert runs[0]["outcome"] == "finished", "the stop was not taken too late"
    assert [path.name for path in tmp_path.glob("prism_check_*")] == [], "a scratch folder was left"


def test_the_timings_time_last_time_is_found_at_the_same_sizes(tmp_path):
    """"Before you start" of the timing says how long its last press took on this computer at the sizes its form and
    step 3's hold, the runs of that press together, though their results name the press that the sizes never hold; at
    other sizes it was not timed here; a newer press one of whose fits did not finish is passed over. It trains two short
    fits, or one when step 3's network is the built-in one, and keeps nothing but the result."""
    from core.gui.memory_limit import computer
    from tests._fixtures import write_history
    sizes = {"num_runs": 20, "epochs": 10, "run_size_cap": 0, "network": [[256, 10], [128, 8]]}

    def ran(at, seconds, press, network, outcome="finished"):
        write_history(tmp_path / "runs", id=f"2026100812{at:02d}00-check", word="Check", started=_at(8, 12, at),
                      outcome=outcome, memory_limit={"gib": None, "source": "none", "computer": computer()},
                      check={"key": "timing", "sizes": sizes, "seconds": seconds, "stages": {},
                             "result": {"network": network, "press": press}})

    ran(1, 30.0, "first", [256, 10])
    ran(2, 100.0, "second", [256, 10])
    ran(3, 80.0, "second", [128, 8])
    ran(4, 20.0, "third", [256, 10])
    ran(5, 5.0, "third", [128, 8], outcome="failed")
    with _checks_page(tmp_path) as (_window, screen, page, _store):
        _step_three(screen, 256, 10)
        page.open_check("timing")
        page.read_facts()
        said = [fact.sentence for fact in page.facts_from_form()]
        assert said[:2] == ["Trains two short fits, one per network, each on 20 batches for 10 epochs; each computes "
                            "its own rotation first.", "Took 3 minutes here last time at these sizes."], said
        assert said[said.index("Nothing else can run until it ends.") + 1] == "Nothing is kept but the result.", said
        page.form.rows["epochs"].widget.setText("12")
        assert "Not timed on this computer yet." in [fact.sentence for fact in page.facts_from_form()]
        page.form.rows["epochs"].widget.setText("10")
        _step_three(screen, 128, 8)
        page.redraw()
        said = [fact.sentence for fact in page.facts_from_form()]
        assert said[:2] == ["Trains one short fit, on 20 batches for 10 epochs; it computes its own rotation first.",
                            "Not timed on this computer yet."], said
        assert page.form.rows["network"].label_text() == (
            "Network: 128 by 8, as step 3's form holds it, which is the built-in one.")


def _timing_runs(per_epoch=(7.4, 5.2)) -> list:
    """The two runs of one press of the timing, as their histories hold them: step 3's network, then the built-in."""
    return [{"id": f"2026100812{at}000-check", "outcome": "finished",
             "check": {"key": "timing", "seconds": 90.0,
                       "result": {"network": network, "simulations": 40_960, "epochs": 10, "span": pace * 10,
                                  "per_epoch": pace, "median_epoch": pace - 0.1, "press": "one"}}}
            for at, (network, pace) in enumerate((([256, 10], per_epoch[0]), ([128, 8], per_epoch[1])), start=1)]


def test_the_timings_result_reads_both_runs_and_projects_with_trainings_expected_epochs(tmp_path):
    """The timing's result reads the two runs of its newest press: each network's time per epoch on the simulations
    timed, a row for each in its table, and the projection for the training step 3's form holds, by the retraining
    runbook's rule (seconds per epoch times the simulations planned over those timed, times the epochs expected),
    naming where its epochs came from; with no epochs expected, it says to type them in step 3."""
    from core.gui.check_readings import NO_EPOCHS_TEXT, newest_press, timing_words
    from core.gui.panels.inference.check_results import timing_block
    runs = _timing_runs()
    last = (130, "about 130 epochs: the last training of this kind, retrain_aug, ran 130")
    sentence, table, projection = timing_words(runs, expected=last, planned=20_480_000)
    assert sentence == ("Timing of a short fit: done. One epoch of the 256 × 10 network took 7.4 s on 40,960 "
                        "simulations; of the 128 × 8 network, 5.2 s.")
    assert (table.columns, table.rows) == (
        ("Network", "Epochs timed", "Span", "Seconds per epoch", "Median epoch"),
        (("256 × 10", "10", "1 min 14 s", "7.4", "7.3 s"), ("128 × 8", "10", "52.0 s", "5.2", "5.1 s")))
    assert projection == ("About 134 hours of fitting, if the fit takes 130 epochs (the last training of this kind, "
                          "retrain_aug).")
    assert timing_words(runs, expected=None, planned=20_480_000)[2] == NO_EPOCHS_TEXT == (
        "There is no epoch count to project with. Type one in \"Expected epochs\" in step 3.")
    older = [{**one, "check": {**one["check"], "result": {**one["check"]["result"], "press": "older"}}}
             for one in runs[:1]]
    assert newest_press([*older, *runs], "timing") == tuple(runs)
    with _checks_page(tmp_path) as (_window, screen, page, _store):
        screen.posterior_panel.form.rows["expected_epochs"].widget.setText("130")
        region = timing_block(runs, parent=page, expected=screen.posterior_panel.expected(), planned=20_480_000)
        assert [fact.sentence for fact in region.sentences()] == [
            "About 134 hours of fitting, if the fit takes 130 epochs (the number given on the Training page)."]
        assert region.verdict.sentence.plain_text() == sentence
        screen.posterior_panel.form.rows["expected_epochs"].widget.setText("")
        assert screen.posterior_panel.expected() is None
        region = timing_block(runs, parent=page, expected=screen.posterior_panel.expected(), planned=20_480_000)
        assert [fact.sentence for fact in region.sentences()] == [NO_EPOCHS_TEXT]


def test_the_not_kept_message_is_not_shown_for_a_check_that_keeps_nothing():
    """A check that keeps nothing but its result keeps no training, so its run's page says nothing of a training not
    kept as it goes; a training's run says it."""
    from core.gui.panels.inference import check_forms
    from core.gui.run.model import RunModel
    from tests._runs import NOT_SAVED, training_spec
    not_kept = ("[fit] the simulation set is not kept (checkpointing off), so this training is not saved as it goes "
                "and cannot be resumed.")
    timing = RunModel(check_forms.check_spec("timing", sizes={"epochs": 10}), clock=lambda: _at(8))
    training = RunModel(training_spec(), clock=lambda: _at(8))
    for model in (timing, training):
        model.feed([(not_kept, "info"), (NOT_SAVED, "info")])
    assert timing.messages() == ()
    assert [message.kind for message in training.messages()] == ["info", "info"]


def _a_card_result(**seconds):
    """What the engine's card test hands back, as a card test reads it: each stage's seconds and the stages run."""
    seconds = seconds or {"prior": 118.0, "posterior": 150.0, "validate": 40.0, "infer": 28.0}
    return SimpleNamespace(seconds=seconds, stages_run=tuple(seconds), prior=object(), posterior=object())


def test_the_card_test_hands_its_one_call_the_forms_arguments_and_a_scratch_store(tmp_path, monkeypatch):
    """A press of the card test hands its one call what its form holds, by the call's own names, on a configuration of
    its own made as the session's, with no cell loaded on it, the cell chosen handed by its path, and the page's own
    way of hearing its stages; the call hands the engine's card test those, its epoch limit one less than its epochs,
    nothing kept or saved by a name, and a store of its own. The form's usual sizes are the engine's card test's own."""
    import inspect
    from core import diagnostics
    from core.diagnostics.card import card_test
    from core.gui import check_scratch
    from core.gui.panels.inference import check_forms
    seen = _tripwire(monkeypatch)
    with _checks_page(tmp_path) as (_window, screen, page, _store):
        _ready(page, "card_test")
        form = page.check_pages["card_test"].form
        cell = check_forms.cell_picker(form).selected_path()
        for typed, changed in (({}, {}), ({"t_obs": "2", "n_cal": "20", "seed": "5"},
                                          {"t_obs_s": 2.0, "n_cal": 20, "seed": 5})):
            _typed_into(form, typed)
            page._main_clicked()
            assert seen["fn"] is check_scratch.card_check, page.foot.reason.text()
            handed = {name: value for name, value in seen["kwargs"].items() if name not in _CONTROLLERS}
            assert handed == {"cell": cell, "t_obs_s": 1.0, "seed": 0, "num_runs": 4, "run_size_cap": 32, "n_cal": 40,
                              "epochs": 6, "hidden_features": 128, "num_transforms": 8,
                              "on_stage": page._card_stages.tell, **changed}, handed
            made, = seen["args"]
            assert made is not screen.session.cfg and made.sources.get("cell") is None
            assert made.chi_mode is screen.session.cfg.chi_mode
            assert seen["kwargs"]["spec"].check == {"key": "card_test", "sizes": check_forms.sizes_of("card_test",
                                                                                                      form)}
    called, stores = [], []
    monkeypatch.setattr(diagnostics, "card_test", lambda *args, **kwargs: (called.append((args, kwargs)),
                                                                           _a_card_result())[1])
    cfg, hear, sink = object(), print, object()
    made = check_scratch.card_check(cfg, cell="c.txt", t_obs_s=1.0, seed=0, num_runs=4, run_size_cap=32, n_cal=40,
                                    epochs=6, hidden_features=128, num_transforms=8, on_stage=hear, fig_sink=sink,
                                    store_factory=lambda: _a_store_of_its_own(tmp_path, stores))
    (args, kwargs), = called
    assert args == (cfg,) and kwargs == {"cell": "c.txt", "t_obs_s": 1.0, "seed": 0, "num_runs": 4, "run_size_cap": 32,
                                         "n_cal": 40, "max_num_epochs": 5, "hidden_features": 128,
                                         "num_transforms": 8, "checkpoint": False, "save": False, "on_stage": hear,
                                         "fig_sink": sink, "store": stores[0]}, kwargs
    assert made.check_result == {"seconds": {"prior": 118.0, "posterior": 150.0, "validate": 40.0, "infer": 28.0},
                                 "stages_run": ["prior", "posterior", "validate", "infer"], "unusable": None,
                                 "split": 0}
    usual = inspect.signature(card_test).parameters
    assert (check_forms.CARD_BATCHES, check_forms.CARD_PER_BATCH, check_forms.CARD_DATASETS,
            check_forms.CARD_EPOCHS - 1) == tuple(usual[name].default for name in (
                "num_runs", "run_size_cap", "n_cal", "max_num_epochs")) and usual["seed"].default == 0


def test_a_scratch_check_writes_nothing_in_the_real_store(tmp_path, monkeypatch):
    """STANDING. The timing of a short fit and the card test, each run through the window with a stage of the engine's
    standing in that writes into the store it is handed, write nothing into the store in use: every call is handed a
    store of its own among the computer's temporary files, which is gone when the run has ended, and the process's
    store is never swapped."""
    import tempfile
    from core import diagnostics, orchestrator
    from core.artifacts import store as artifact_store
    from core.artifacts.store import KIND_DIRS
    from tests._runs import StandIn, turn_until
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path / "temp"))
    (tmp_path / "temp").mkdir()
    app = qt_app()
    handed = []

    def writing(stand_in):
        def stage(*args, **kwargs):
            folder = Path(kwargs["store"].root)
            handed.append(folder)
            (folder / "made_here.txt").write_text("a record of its own", encoding="utf-8")
            return stand_in(*args, **kwargs)
        return stage

    fit, card = StandIn(word="Check"), StandIn(word="Check")
    monkeypatch.setattr(orchestrator, "build_posterior", writing(fit))
    monkeypatch.setattr(diagnostics, "card_test", writing(card))
    with _checks_page(tmp_path) as (_window, _screen, page, store):
        def listing():
            return {kind: sorted(summary.id for summary in store.list(kind)) for kind in KIND_DIRS}
        before = listing()

        def never(*_args, **_kwargs):
            raise AssertionError("the process's store was swapped")
        for module in (artifact_store, __import__("core.artifacts", fromlist=["use_store"])):
            monkeypatch.setattr(module, "use_store", never)
            monkeypatch.setattr(module, "set_default_store", never)
        for key, stand_in, result in (("timing", fit, _a_posterior()), ("card_test", card, _a_card_result())):
            _ready(page, key)
            page._main_clicked()
            assert stand_in.started.wait(5), (key, page.foot.reason.text())
            stand_in.release(result)
            turn_until(app, lambda: not controller().is_live(), 10.0)
            _settle()
        assert listing() == before
    assert len(handed) == 2 and all(folder.parent == tmp_path / "temp" and folder != Path(store.root)
                                    for folder in handed), handed
    assert list((tmp_path / "temp").iterdir()) == [], "a scratch folder was left"


def test_the_card_tests_stage_line_follows_its_stages(tmp_path, monkeypatch):
    """The card test's stages, told by the engine on the worker's thread, are heard on the window's: each begun
    begins on the run's page in its words, in turn, the one before it done; each event is kept as it was told."""
    import threading
    from core import diagnostics
    from core.gui.run.view import RunView
    from tests._runs import StandIn, turn_until
    stand_in = StandIn(word="Check")
    monkeypatch.setattr(diagnostics, "card_test", stand_in)
    threads = []
    real = RunView.set_stage
    monkeypatch.setattr(RunView, "set_stage", lambda view, name: (threads.append(threading.current_thread()),
                                                                  real(view, name))[1])
    app = qt_app()
    with _checks_page(tmp_path) as (_window, _screen, page, _store):
        _ready(page, "card_test")
        page._main_clicked()
        assert stand_in.started.wait(5)
        said = []
        for stage, words in (("prior", "Prior"), ("posterior", "Training"), ("validate", "Calibration"),
                             ("infer", "Inference")):
            stand_in.stage(stage, "started")
            turn_until(app, lambda: controller().model.stage == words)
            said.append([(word.name, word.marker) for word in controller().model.stage_line()])
            stand_in.stage(stage, "finished", 2.0)
        stand_in.release(_a_card_result())
        turn_until(app, lambda: not controller().is_live(), 10.0)
        heard = list(page.card_heard)
    names = ["Prior", "Training", "Calibration", "Inference"]
    assert said == [[(name, "now" if at == now else "done" if at < now else "later") for at, name in enumerate(names)]
                    for now in range(4)], said
    assert [(stage, event) for stage, event, _seconds in heard] == [
        (stage, event) for stage in ("prior", "posterior", "validate", "infer") for event in ("started", "finished")]
    assert len(threads) == 4 and all(thread is threading.main_thread() for thread in threads), threads


def test_the_card_tests_page_hears_any_order_of_its_stages_and_never_raises(tmp_path):
    """What the card test's run is handed to tell its stages only says them, and never raises, whatever it is told and
    in whatever order, with no run of the page's own live: the page keeps each, and begins nothing on the run's page
    but for a run of its own."""
    told = (("infer", "finished", 1.0), ("prior", "started", None), ("prior", "started", None),
            ("no such stage", "failed", "not a number"), (None, 3, object()))
    with _checks_page(tmp_path) as (_window, _screen, page, _store):
        before = page.run_view.stage_line.words()
        for one in told:
            page._card_stages.tell(*one)
        _settle()
        assert [event[:2] for event in page.card_heard] == [(str(stage), str(event)) for stage, event, _ in told]
        assert page.run_view.stage_line.words() == before


def _card_run(at, outcome, *, seconds=None, stage=None, stages=None, unusable=None, split=0):
    """A card test's run as its history holds it."""
    result = {} if seconds is None else {"seconds": seconds, "stages_run": list(seconds), "unusable": unusable,
                                         "split": split}
    return {"id": f"2026100812{at:02d}00-check", "word": "Check", "outcome": outcome, "stage": stage,
            "started": _at(8, 12, at).isoformat(), "check": {"key": "card_test", "seconds": 340.0,
                                                             "stages": stages or {}, "result": result}}


def test_the_card_tests_own_reading_of_its_training(tmp_path, monkeypatch):
    """The card test reads its training's share of unusable driven recordings from the first total its run says, a
    later one (its calibration's) changing nothing, and counts each batch run in smaller pieces once, however often and
    by either way it was halved; its sentence of the share is said from the two named numbers it is read against."""
    from core import diagnostics
    from core.gui import check_scratch
    from tests._runs import masked_line
    tag = "training batch {}/4 [t_scale=0.8, T=1.0, n_fine=4096, N_points=2048, rows=32]"
    split = (f"{tag.format(3)}: OOM at simulation batch 32; retrying in chunks of 16, 3.71 GiB reported free "
             f"(optimistic on Windows). Original: CUDA out of memory",
             f"{tag.format(3)}: OOM with 32 rows OUTSIDE the simulator retry; re-running this batch in halves of 16. "
             f"Original: CUDA out of memory",
             f"{tag.format(4)}: OOM at simulation batch 32; retrying in chunks of 16. Original: CUDA out of memory")

    def card_test(*_args, **_kwargs):
        for line in (masked_line(412, 1000, 4, WHOLE), *split, masked_line(5, 1000, 1, WHOLE)):
            _said_at(line, 0.0)
        return _a_card_result()
    monkeypatch.setattr(diagnostics, "card_test", card_test)
    made = check_scratch.card_check(object(), cell="c.txt", t_obs_s=1.0, seed=0, num_runs=4, run_size_cap=32, n_cal=40,
                                    epochs=6, hidden_features=128, num_transforms=8, on_stage=print,
                                    store_factory=lambda: _a_store_of_its_own(tmp_path, []))
    assert (made.check_result["unusable"], made.check_result["split"]) == (0.412, 2), made.check_result
    run = _card_run(1, "finished", seconds={"prior": 1.0}, unusable=0.412)
    monkeypatch.setattr(check_readings, "CARD_SAYS_NOTHING_POINTS", 10)
    assert check_readings.card_words(run)[3][0] == ("Unusable driven recordings in its training: 41.2 %. At this size a "
                                                   "share within about 10 points of 37 % says nothing either way.")


def test_the_card_tests_result_says_done_or_where_it_failed(tmp_path):
    """The card test's result says every stage ran on the card and in how long, or the stage it failed in; its table
    gives each stage's seconds beside the last finished run's; it says the share of unusable driven recordings its
    training simulated, with what that share means at this size, the batches split for lack of memory, and what the
    test shows; its calibration's verdict never. Its row reads "Done" only from a run that finished."""
    from core.gui.panels.inference.check_results import card_block
    earlier = _card_run(1, "finished", seconds={"prior": 120.0, "posterior": 160.0, "validate": 41.0, "infer": 30.0})
    done = _card_run(2, "finished", seconds={"prior": 118.0, "posterior": 150.0, "validate": 40.0, "infer": 28.0},
                     unusable=0.412, split=0)
    failed = _card_run(3, "failed", stage="Training", stages={"Prior": 118.0, "Training": 30.5})
    with _checks_page(tmp_path) as (_window, _screen, page, _store):
        region = card_block(done, parent=page, earlier=earlier)
        assert (region.verdict.kind, region.verdict.sentence.plain_text()) == (
            "done", "Card test: done. Every stage ran on the graphics card in 5 min 36 s.")
        assert region.table_words().rows == (("Prior", "1 min 58 s", "2 min"), ("Training", "2 min 30 s", "2 min 40 s"),
                                             ("Calibration", "40.0 s", "41.0 s"), ("Inference", "28.0 s", "30.0 s"))
        assert [fact.sentence for fact in region.sentences()] == [
            "Unusable driven recordings in its training: 41.2 %. At this size a share within about 12 points of 37 % "
            "says nothing either way.", "Batches split for lack of memory: 0.",
            "This shows that the chain runs. It says nothing about calibration: at these sizes the test has no power."]
        region = card_block(failed, parent=page, earlier=done)
        assert (region.verdict.kind, region.verdict.sentence.plain_text()) == ("err", "Card test: failed in Training.")
        assert region.table_words().rows[:2] == (("Prior", "1 min 58 s", "1 min 58 s"),
                                                 ("Training", "30.5 s", "2 min 30 s"))
        assert not any("calibrat" in word.lower() and "passed" in word.lower() for word in _block_words(region))
    for histories, state in (((failed,), "Not run yet"), ((done, failed), "Done")):
        assert _row(check_lineage.rows(_held(), (), histories), "card_test").state == state, histories


def test_the_card_tests_before_you_start_says_its_four_stages_its_time_and_the_memory():
    """"Before you start" of the card test says, in one sentence, that it builds its prior at full size and runs three
    small stages after it on the card, how long it took here last time, or that it has not run here, and that nothing
    else can run and nothing is kept; then the memory lines. Before the history is read, no time is said."""
    from core.gui.panels.inference import check_forms
    memory = ("Memory limit: none.", "The last run on this computer ran with no limit.")
    head = ("This builds a prior at full size and then trains a small posterior, calibrates it and runs one inference "
            "on a simulated cell: four real stages, the last three at very small sizes, on the graphics card. It takes "
            "minutes")
    tail = ". Nothing else can run meanwhile, and nothing it makes is kept."
    for last, read, words in ((None, True, "; it has not run on this computer yet"),
                              ({"check": {"seconds": 336.0}}, True, " (5 min 36 s here last time)"),
                              ({"check": {"seconds": 336.0}}, False, "")):
        said = check_forms.before_you_start("card_test", sizes={}, last=last, memory=memory, read=read)
        assert [(fact.kind, fact.sentence) for fact in said] == [("info", head + words + tail),
                                                                 ("info", " ".join(memory))], said


def test_the_card_tests_network_is_step_threes_until_it_is_changed(tmp_path):
    """The card test's network width and depth are step 3's form's while they stand at their usual value, which they
    take; a width typed on the card test's form stays what was typed when step 3's changes."""
    with _checks_page(tmp_path) as (_window, screen, page, _store):
        rows = page.check_pages["card_test"].form.rows
        _step_three(screen, 256, 10)
        page.open_check("card_test")
        assert [(rows[key].value(), rows[key].at_usual()) for key in ("hidden_features", "num_transforms")] == [
            (256, True), (10, True)]
        rows["hidden_features"].widget.setText("64")
        _step_three(screen, 300, 12)
        page.refresh_local_gates()
        assert [rows[key].value() for key in ("hidden_features", "num_transforms")] == [64, 12]


def test_no_check_says_it_cannot_be_run_from_the_window(tmp_path):
    """STANDING. Every check with a button has its form and its start, and none says it cannot be run from the window:
    its words are gone from the window's code, and each check's page opens with its main button named."""
    from core.gui.check_catalogue import CHECKS
    from core.gui.panels.inference import check_forms, validate_tab
    buttons = {one.key for one in CHECKS if one.button}
    assert buttons == set(check_forms.FORMS) == set(validate_tab._STARTS), sorted(buttons ^ set(check_forms.FORMS))
    assert not hasattr(check_forms, "NOT_BUILT_TEXT")
    said = [path.name for path in (REPO / "core" / "gui").rglob("*.py")
            if "cannot be run from the window" in path.read_text(encoding="utf-8")]
    assert said == [], said
    with _checks_page(tmp_path) as (_window, _screen, page, _store):
        for key in sorted(buttons):
            page.open_check(key)
            assert page.foot.main_button.text().replace("&", "") in (check_catalogue.entry(key).button,
                                                                    check_catalogue.AGAIN_WORDS), key
            assert not any("cannot be run" in fact.sentence for fact in page.facts_from_form()), key


# ── the comparison after simulating: its answers ──
#: The training that made the posterior's simulation set, as its history file is named.
_TRAINED = "20261002T081000-training"


def _an_audit(store, cfg, prior, _posterior, *, fraction=0.379) -> None:
    from tests._fixtures import check_results, write_diagnostic
    write_diagnostic(store, cfg, diagnostic="probes", variant="mask", parents={"prior": prior},
                     **check_results("audit", masked_fraction=fraction))


def _audit_ids(store) -> list:
    """The audits the store holds, oldest first."""
    return [summary.id for summary in reversed(store.list("diagnostic"))]


def _a_set_of(tmp_path, store, screen, *, masked=92, named=True, compared=None, comparison=True) -> Path:
    """The posterior in use trained on a simulation set whose batches each counted ``masked`` of 221 driven
    recordings unusable (92: 41.6 %, 3.7 points from the audit's 37.9 %), and, when ``named``, the history of the
    training that made it, which keeps ``compared`` as its comparison with the audit (none at all without
    ``comparison``, as a history written before runs kept one)."""
    from core.gui.run import history
    from tests._fixtures import write_history, write_simulation_set
    folder = write_simulation_set(store.kind_dir("simulation"), {"a set of": "the comparison"}, 100, complete=True,
                                  chi_masked=[(masked, 221)] * 100)
    screen.session.posterior = _held_posterior(tmp_path, prior=screen.session.inf_prior, simulation=folder.name)
    if named:
        entry = {name: None for name in history.COMPARISON_FIELDS}
        entry.update({"unusable": masked * 100, "total": 22_100, "scope": WHOLE, "batches": 100, "batch_median": 41.0,
                      "batch_low": 35.5, "batch_high": 47.2, "state": "Look closer", **(compared or {})})
        from core.gui.panels.inference.base import TRAINING_STAGES
        write_history(tmp_path / "runs", id=_TRAINED, word="Training", started=_at(2, 8, 10), ended=_at(3, 9),
                      outcome="finished", records_root=str(store.root), simulation_set={"folder": str(folder)},
                      stages=[[stage.name, list(stage.begins), stage.counts] for stage in TRAINING_STAGES],
                      health={"failed": 0, "simulated": 22_100, **({"comparison": entry} if comparison else {})})
    return folder


def test_an_acceptance_is_written_into_the_training_history_during_and_after_the_run(tmp_path):
    """A difference to look closer at shows the comparison's table and "Accept this difference". Pressed after the
    training, it writes the acceptance, with its moment, the audit it is of and the difference, into the history of the
    training that made the posterior's set; from then the check reads "Passed" and its state line and its sentence say
    the difference was accepted, and when. Pressed while a training is live, its run is answered so, the run's history
    keeps it, and its page reads the same words."""
    from core.gui.run import history
    from core.gui.texts import date_text
    from tests._runs import StandIn, masked_line, start_run, training_spec, turn_until
    app = qt_app()
    with _checks_page(tmp_path, records=_an_audit) as (window, screen, page, store):
        audit, = _audit_ids(store)
        _a_set_of(tmp_path, store, screen, compared={"audit": audit, "audit_fraction": 0.379})
        page.read_facts()
        page.open_check("comparison")
        one = page.check_pages["comparison"]
        assert _row(screen.check_rows(), "comparison").state == "Look closer"
        assert one.result.table_words().rows[0] == ("Simulation set", "41.6 %", "22,100", "100", "41.0 %",
                                                    "35.5 to 47.2 %")
        assert one.result.table_words().rows[1][:2] == ("The audit", "37.9 %")
        button, = one.result.actions()
        assert (button.text(), button.isEnabled()) == ("Accept this difference", True)
        button.click()
        accepted = history.read(tmp_path / "runs" / f"{_TRAINED}.json")["health"]["comparison"]["accepted"]
        assert (accepted["at"], accepted["audit"], round(accepted["difference"], 2)) == (
            _at(8, 12, 4).isoformat(), audit, 3.73), accepted
        said = f"A difference of 3.7 points was accepted on {date_text(_at(8, 12, 4).astimezone())}."
        row = _row(screen.check_rows(), "comparison")
        assert row.state == "Passed" and row.line.endswith(said), row.line
        assert one.result.verdict.sentence.plain_text().endswith(said)
        assert one.result.actions() == [], "a difference accepted asked to be accepted again"
        training = StandIn(word="Training")
        start_run(window.part("inference").tsnpe_panel, training, spec=training_spec(
            place=("inference", "narrowing"), facts={"model": "NADROWSKI", "observation": "chi", "narrowed": False,
                                                    "network": (256, 10), "cadence": 50,
                                                    "audit": {"id": audit, "fraction": 0.379, "batches": 100}}))
        assert training.started.wait(5)
        run = controller().status.run
        training.say(masked_line(4163, 10_000, 3, WHOLE))
        turn_until(app, lambda: controller().model.comparison() is not None)
        page.refresh_local_gates()
        assert one.result.verdict.sentence.plain_text().endswith("up to 5 is a reason to look closer.")
        button, = one.result.actions()
        button.click()
        assert "accepted" in controller().model.answers()
        assert one.result.verdict.sentence.plain_text().endswith(said)
        assert history.read(tmp_path / "runs" / f"{run}.json")["health"]["comparison"]["accepted"]["audit"] == audit
        training.release()
        turn_until(app, lambda: not controller().is_live())


def test_while_a_training_is_live_the_comparisons_page_gives_one_answer(tmp_path):
    """While a training is live, the comparison after simulating answers once on its page: the list's row (in the whole
    table and the narrow one), the page's state line and the result under them are all the running training's, from its
    run's model in memory: "Running" until its simulation ends, then its own comparison, here a difference to look
    closer at, though the records read before it began say "Passed". The headline still counts the records."""
    from tests._runs import StandIn, masked_line, start_run, training_spec, turn_until
    app = qt_app()
    with _checks_page(tmp_path, records=_an_audit) as (window, screen, page, store):
        audit, = _audit_ids(store)
        _a_set_of(tmp_path, store, screen, masked=86, compared={"audit": audit, "audit_fraction": 0.379,
                                                                 "state": "Passed"})
        page.read_facts()
        page.open_check("comparison")
        one = page.check_pages["comparison"]
        assert _row(screen.check_rows(), "comparison").state == "Passed"
        headline = page.headline.text()

        def said() -> tuple:
            at = page.listing.row_of["comparison"]
            whole = page.listing.whole.view.model().index(at, 2).data(Qt.UserRole)
            narrow = page.listing.narrow.view.model().index(at, 1).data(Qt.UserRole)
            return (whole.word, narrow.word, one.state_line.sentence.plain_text(),
                    one.result.verdict.sentence.plain_text())

        training = StandIn(word="Training")
        start_run(window.part("inference").tsnpe_panel, training, spec=training_spec(
            place=("inference", "narrowing"), facts={"model": "NADROWSKI", "observation": "chi", "narrowed": False,
                                                    "network": (256, 10), "cadence": 50,
                                                    "audit": {"id": audit, "fraction": 0.379, "batches": 100}}))
        assert training.started.wait(5)
        page.refresh_local_gates()
        assert said() == ("Running", "Running", "Running.", "Counted when simulation ends."), said()
        training.say(masked_line(4163, 10_000, 3, WHOLE))
        turn_until(app, lambda: controller().model.comparison() is not None)
        page.refresh_local_gates()
        whole, narrow, line, verdict = said()
        assert (whole, narrow, line) == ("Look closer", "Look closer", "Look closer."), said()
        assert verdict.endswith("up to 5 is a reason to look closer."), verdict
        assert page.headline.text() == headline == screen.checks_summary().headline, "the headline left the records"
        assert _row(screen.check_rows(), "comparison").state == "Passed", "the reading's row was changed"
        training.release()
        turn_until(app, lambda: not controller().is_live())


@pytest.mark.parametrize("factor", [1.0, 1.5])
def test_the_comparisons_table_beside_the_list_keeps_its_row_names_whole(factor, tmp_path):
    """At the window's opening size, at both text sizes, the comparison's page of a difference to look closer at stands
    beside the list, and its table's first column, which names each row ("Simulation set", "The audit"), is at least as
    wide as its longest word: the table keeps its row names whole and, where its block is narrower than the table,
    scrolls sideways inside it rather than crush them. The table's block stands within the page."""
    from PySide6.QtCore import QPoint
    from PySide6.QtGui import QFontMetricsF
    from core.gui import design
    from core.gui.widgets import rule_table as rt
    with text_size(factor), _checks_page(tmp_path, records=_an_audit,
                                         size=(design.scaled(1300), design.scaled(820))) as (_window, screen, page,
                                                                                            store):
        audit, = _audit_ids(store)
        _a_set_of(tmp_path, store, screen, compared={"audit": audit, "audit_fraction": 0.379})
        page.read_facts()
        page.open_check("comparison")
        _settle()
        one = page.check_pages["comparison"]
        view = one.result.table.view
        lead, gap = rt._pads(0, view.model().columnCount())
        words = [word for row in one.result.table_words().rows for word in row[0].split()]
        widest = max(QFontMetricsF(rt._face("text")).horizontalAdvance(word) for word in words)
        assert view.columnWidth(0) - lead - gap >= widest, (factor, view.columnWidth(0) - lead - gap, widest)
        assert page.list_detail.beside(), (factor, "the comparison's page left the list's side")
        block = one.result.table_block
        assert block.isVisibleTo(one) and block.mapTo(one, QPoint(block.width(), 0)).x() <= one.width(), (
            factor, block.width(), one.width())


def test_an_acceptance_lapses_when_another_audit_counts(tmp_path):
    """An acceptance is of the audit it names: once another audit counts for the prior, the check reads the difference
    from that audit again, to look closer at, and says no acceptance."""
    with _checks_page(tmp_path, records=_an_audit) as (_window, screen, page, store):
        first, = _audit_ids(store)
        _a_set_of(tmp_path, store, screen, compared={
            "audit": first, "audit_fraction": 0.379,
            "accepted": {"at": _at(3, 9, 14).isoformat(), "audit": first, "difference": 3.73}})
        page.read_facts()
        assert _row(screen.check_rows(), "comparison").state == "Passed"
        _an_audit(store, screen.session.cfg, screen.session.inf_prior.id, None, fraction=0.375)
        page.read_facts()
        row = _row(screen.check_rows(), "comparison")
        assert row.state == "Look closer" and "accepted" not in row.line, row.line


def test_accept_is_greyed_for_a_set_with_no_history(tmp_path):
    """For a simulation set no run history names, made outside the window, "Accept this difference" is greyed, and the
    page says why: there is no run history to keep the acceptance in."""
    from core.gui.check_readings import NO_HISTORY_TEXT
    with _checks_page(tmp_path, records=_an_audit) as (_window, screen, page, store):
        _a_set_of(tmp_path, store, screen, named=False)
        page.read_facts()
        page.open_check("comparison")
        one = page.check_pages["comparison"]
        button, = one.result.actions()
        assert not button.isEnabled() and NO_HISTORY_TEXT in _texts(one.result)
        assert NO_HISTORY_TEXT == ("This simulation set was made outside the window: there is no run history to keep "
                                   "the acceptance in.")
        assert one.result.table_words().rows[0][4:] == ("", ""), "the set's figures by batch were made up"


def test_accept_is_greyed_for_a_set_whose_trainings_kept_no_comparison(tmp_path):
    """For a simulation set the run history names, whose trainings ran before the window kept their comparison with the
    audit, "Accept this difference" is greyed with a sentence true of that set: there is no comparison to accept into.
    Its table claims no figure by batch the history does not hold."""
    from core.gui.check_readings import NO_COMPARISON_TEXT, NO_HISTORY_TEXT
    with _checks_page(tmp_path, records=_an_audit) as (_window, screen, page, store):
        _a_set_of(tmp_path, store, screen, comparison=False)
        page.read_facts()
        page.open_check("comparison")
        one = page.check_pages["comparison"]
        button, = one.result.actions()
        said = _texts(one.result)
        assert not button.isEnabled() and NO_COMPARISON_TEXT in said and NO_HISTORY_TEXT not in said, said
        assert one.result.table_words().rows[0][4:] == ("", "")


def test_the_answer_to_keep_training_is_shown_on_the_comparisons_page(tmp_path):
    """The answer "Keep training" beyond the last band is said on the comparison's page with its moment; it changes no
    state: the check still reads "Failed"."""
    from core.gui.texts import date_text
    with _checks_page(tmp_path, records=_an_audit) as (_window, screen, page, store):
        audit, = _audit_ids(store)
        kept = "2026-10-02T09:14:00-07:00"
        _a_set_of(tmp_path, store, screen, masked=100, compared={"audit": audit, "state": "Failed",
                                                                  "kept_training": kept})
        page.read_facts()
        page.open_check("comparison")
        one = page.check_pages["comparison"]
        assert _row(screen.check_rows(), "comparison").state == "Failed"
        assert [fact.sentence for fact in one.result.sentences()] == [
            f"You chose to keep training on {date_text(datetime.fromisoformat(kept).astimezone())}."]


def test_keep_training_is_not_asked_again_at_a_reopening(tmp_path):
    """A training whose history keeps the answer "Keep training", its window closed and opened again: nothing asks for
    the answer again, on the run's past view (no "Keep training", no stop beside it), in the status that Home and the
    rail's foot read, or anywhere; the comparison's page names the answer and its moment."""
    from core.gui.texts import date_text
    from tests._fixtures import build_window
    from tests._runs import masked_line
    kept = "2026-10-02T09:14:00-07:00"
    with _checks_page(tmp_path, records=_an_audit) as (_window, screen, page, store):
        audit, = _audit_ids(store)
        _a_set_of(tmp_path, store, screen, masked=100, compared={"audit": audit, "audit_fraction": 0.379,
                                                                  "state": "Failed", "kept_training": kept})
        (tmp_path / "runs" / f"{_TRAINED}.txt").write_text(
            f"2026-10-02 09:10:00 info {masked_line(22_100, 49_000, 100, WHOLE)}\n", encoding="utf-8")
        again = build_window()
        assert controller().status.needs_you is None
        host = again.part("inference").posterior_panel
        assert controller().show_past(_TRAINED, host)
        past = host.past_view
        assert past.answers.isHidden() and not past.keep_training_button.isVisibleTo(past), "the past asked"
        assert [row.state for row in past._model.health() if row.measure == "Unusable driven recordings"] == ["Failed"]
        page.read_facts()
        page.open_check("comparison")
        said = [fact.sentence for fact in page.check_pages["comparison"].result.sentences()]
        assert said == [f"You chose to keep training on {date_text(datetime.fromisoformat(kept).astimezone())}."]
        again.hide()
