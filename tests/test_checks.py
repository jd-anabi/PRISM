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
    if failed:
        sentence = "Failed: " + ", ".join(failed) + "."
    elif closer:
        sentence = "Look closer: " + ", ".join(closer) + "."
    elif opened and never:
        sentence = f"{opened} No run can pass {', '.join(never)} for this posterior."
    elif opened:
        sentence = opened
    elif never:
        sentence = f"Inference can be run now; no run can pass {', '.join(never)} for this posterior."
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
        "check 3 was dated while it waits for the audit")
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
        assert list(one.form.rows) == ["n_cal", "cal_n_scales"]
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
        [("lock", "The prior in use is not the one this posterior was trained on. Load it in step 2.")],
        "Cannot start yet: the prior in use is not the posterior's own."),
    "the calibration with no prior": (
        "calibration", dict(prior=None), [("lock", "Needs a prior: build or load one in step 2, Prior.")],
        "Cannot start yet: needs a prior."),
    "the input use, trained without its simulations": (
        "input_use", dict(simulation=None),
        [("lock", "This check cannot be run from the window yet."),
         ("lock", "This posterior was trained without keeping its simulations, and this check reads them.")],
        "Cannot start: this check cannot be run from the window yet, and 1 more above."),
    "the identifiability, trained without the rotation": (
        "identifiability", dict(rotated=False),
        [("lock", "This check cannot be run from the window yet."),
         ("lock", "This posterior was trained without the rotation, so there are no directions to read.")],
        "Cannot start: this check cannot be run from the window yet, and 1 more above."),
    "the frequency band without the frequency response": (
        "band", dict(chi=False), [("lock", "This check cannot be run from the window yet."),
                                  ("lock", "Applies only with the frequency response.")],
        "Cannot start: this check cannot be run from the window yet, and 1 more above."),
    "the audit without the frequency response": (
        "audit", dict(chi=False), [("lock", "This check cannot be run from the window yet."),
                                   ("lock", "Applies only with the frequency response.")],
        "Cannot start: this check cannot be run from the window yet, and 1 more above."),
    "what the observations pin down, of another model": (
        "pin_down", dict(nadrowski=False), [("lock", "This check cannot be run from the window yet."),
                                            ("lock", "Applies only to the Nadrowski model.")],
        "Cannot start: this check cannot be run from the window yet, and 1 more above."),
    "what the observations pin down, with no forcing and no frequency response": (
        "pin_down", dict(chi=False, forcing=False),
        [("lock", "This check cannot be run from the window yet."),
         ("lock", "Needs a bounds file with a forcing section.")],
        "Cannot start: this check cannot be run from the window yet, and 1 more above."),
    "the drive study with no forcing section": (
        "drive_study", dict(forcing=False), [("lock", "This check cannot be run from the window yet."),
                                             ("lock", "Needs a bounds file with a forcing section.")],
        "Cannot start: this check cannot be run from the window yet, and 1 more above."),
    "identifiability across the prior with the frequency response": (
        "across_prior", dict(), [("lock", "This check cannot be run from the window yet."),
                                 ("lock", "Does not apply with the frequency response.")],
        "Cannot start: this check cannot be run from the window yet, and 1 more above."),
    "identifiability across the prior with no forcing section": (
        "across_prior", dict(chi=False, forcing=False),
        [("lock", "This check cannot be run from the window yet."),
         ("lock", "Needs a bounds file with a forcing section.")],
        "Cannot start: this check cannot be run from the window yet, and 1 more above."),
    "the calibration repeated, with the prior of another": (
        "fixed_number", dict(pair_refused="refused", posterior_prior_words="older_prior"),
        [("lock", "This check cannot be run from the window yet."),
         ("lock", "The prior in use is not the one this posterior was trained on (older_prior). Load it in step 2.")],
        "Cannot start: this check cannot be run from the window yet, and 1 more above."),
    "the timing, which the window cannot run yet": (
        "timing", dict(), [("lock", "This check cannot be run from the window yet.")],
        "Cannot start: this check cannot be run from the window yet."),
    "the calibration, free to run": ("calibration", dict(), [], ""),
}


#: The words beside a check's greyed button for each reason, by the start of its sentence, and the lead they read alone:
#: a load lifts the first three; waiting lifts none of the others.
_BLOCK_WORDS = (("Needs a prior", "needs a prior", "Cannot start yet:"),
                ("Needs a posterior", "needs a posterior", "Cannot start yet:"),
                ("The prior in use is not", "the prior in use is not the posterior's own", "Cannot start yet:"),
                ("This check cannot be run", "this check cannot be run from the window yet", "Cannot start:"),
                ("This posterior was trained without keeping", "the posterior was trained without keeping its simulations",
                 "Cannot start:"),
                ("This posterior was trained without the rotation", "the posterior was trained without the rotation",
                 "Cannot start:"),
                ("Applies only with", "it applies only with the frequency response", "Cannot start:"),
                ("Applies only to the Nadrowski", "it applies only to the Nadrowski model", "Cannot start:"),
                ("Needs a bounds file", "the bounds file has no forcing section", "Cannot start:"),
                ("Does not apply with", "it does not apply with the frequency response", "Cannot start:"))


@pytest.mark.parametrize("case", list(_GREYED))
def test_each_checks_button_is_greyed_with_the_reason_of_its_table(case):
    """Each check's button is greyed by the reason its row gives, each in its table's exact words: a block a load lifts,
    the prior, the posterior or the posterior's own prior, says "Cannot start yet:"; one that waiting will not lift, a
    posterior trained without what the check reads, a model, a kind of observation or a bounds file the check does not
    apply to, and a check the window cannot run yet, says "Cannot start:". The prior a posterior was trained on is named
    when the reading found its record, and said without its name when it is gone. Each reason has its own words beside
    the button, read alone after their lead. A live run is said first."""
    from core.gui.panels.inference.check_forms import FORMS, NOT_BUILT_TEXT
    from core.gui.step_facts import reason_beside
    key, given, facts, beside = _GREYED[case]
    row = _row(check_lineage.rows(_held(**given), (), ()), key)
    made = check_lineage.button_facts(row, not_built="" if key in FORMS else NOT_BUILT_TEXT)
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
        named = "" if pair == "another prior, its record gone" else " (older_prior)"
        assert page.foot.reason.text() == ("" if live else "Cannot start yet: the prior in use is not the posterior's "
                                                           "own.")
        facts = [fact.sentence for fact in page.facts_from_form()]
        sentence = f"The prior in use is not the one this posterior was trained on{named}. Load it in step 2."
        assert (sentence in facts) == (not live), facts
        if pair == "another prior, its record kept":
            audit = page.check_pages["audit"].result.verdict.sentence.plain_text()
            assert "simulated from the prior older_prior were unusable" in audit, audit


def test_the_foot_with_no_check_open_offers_the_next_check(tmp_path):
    """With no check open the foot's main button opens the first deciding check that applies, has not passed and that a
    run can still pass, named for it, and goes to Inference once none is left; it says no fact of its own, and with no
    prior in use it is greyed with the reason of a step not done. A check open reads its own button's words, and "Run
    again" once a record of it exists; on the page of a check with no button of its own, the comparison's, the foot
    passes over that check, which stands open already, to the next, or to Inference."""
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
        assert page.foot.main_button.text().replace("&", "") == (
            "Open Unusable driven recordings: simulation set against the audit")
        page.foot.main_button.click()
        assert page.open_key == "comparison"
        assert page.foot.main_button.text().replace("&", "") == "Open Calibration", (
            "the comparison's page offered itself")
        page.foot.main_button.click()
        assert page.open_key == "calibration"
        page.open_check(None)
        screen.session.posterior = _held_posterior(tmp_path, prior=screen.session.inf_prior, text=nothing_counted)
        page.read_facts()
        assert _row(screen.check_rows(), "comparison").state == "Not compared"
        assert page.foot.main_button.text().replace("&", "") == "Open Calibration", (
            "a check no run can pass was offered next")
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


#: What a record of each check holds that says its verdict, its reading or its reasons in the engine's own words.
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
        stand_in = StandIn(word="Reduction")
        start_run(window.part("reduction"), stand_in)
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
        start_run(window.part("reduction"), training, spec=training_spec(place=("reduction", None)))
        page.refresh_local_gates()
        _settle()
        assert one.result.verdict.sentence.plain_text() == "Counted when simulation ends.", (
            "the comparison was worked out while a training is live")
        assert calls == [], f"read while a training is live: {calls}"
        training.release()
        turn_until(app, lambda: not controller().is_live())


def test_check_3_is_counted_later_only_where_it_applies(tmp_path):
    """While a training is live, check 3's page says it is counted when the simulation ends only where the check
    applies: with the frequency response off it does not apply and its page says nothing more."""
    from tests._runs import StandIn, start_run, training_spec, turn_until
    app = qt_app()
    with _checks_page(tmp_path, chi=False) as (window, _screen, page, _store):
        training = StandIn(word="Training")
        start_run(window.part("reduction"), training, spec=training_spec(place=("reduction", None)))
        page.refresh_local_gates()
        _settle()
        one = page.check_pages["comparison"]
        assert one.state_line.sentence.plain_text().startswith("Does not apply.")
        assert one.result is None, one.result.verdict.sentence.plain_text()
        training.release()
        turn_until(app, lambda: not controller().is_live())


def test_check_3_waiting_for_the_audit_says_which_check_to_run(tmp_path):
    """With the posterior's set counted and no audit of its prior that counts, check 3's page says, with no run live,
    that there is no audit to compare with, the set's share, and which check of the list to run; not the run's own
    words for after its end."""
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
    """While a narrowing round is live, check 3's page says it is counted when the round's simulation ends, as during a
    training, and nothing is worked out from the counts read before."""
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
        start_run(window.part("reduction"), round_, spec=training_spec("Narrowing round", place=("reduction", None)))
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
        stand_in = StandIn(word="Reduction")
        start_run(window.part("reduction"), stand_in)
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
    from core.gui.run.controller import RunController
    seen = {}
    monkeypatch.setattr(RunController, "dispatch", lambda held, fn, *args, **kwargs: seen.update(
        fn=fn, args=args, kwargs=kwargs) or True)
    drawn = iter((1840273, 77))
    monkeypatch.setattr(random, "randrange", lambda top: next(drawn) if top == 2 ** 31 else 0)
    with _checks_page(tmp_path) as (_window, screen, page, _store):
        screen.plan_source = SimpleNamespace(kind="prior", ident=PRIOR, name="retrain_prior")
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
        assert spec.plan == {"kind": "prior", "id": PRIOR, "name": "retrain_prior"} == screen.plan_entry()
        assert spec.settings is None
        assert spec.check == {"key": "calibration", "sizes": {"n_cal": 2500, "cal_n_scales": 200}}
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
        assert summary.sentence == ("Inference can be run now; no run can pass Unusable driven recordings: simulation "
                                    "set against the audit, Use of each part of the input for this posterior.")
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
                               reason=check_lineage.OTHER_PRIOR_TEXT),
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
