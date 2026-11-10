"""How a check's record is read, as values: the state a record of each deciding check reads, by the rules a deciding
check is judged by, and the comparison of the share of unusable driven recordings a training's simulation set holds
with the audit of its prior.

The engine decides the frequency band and the calibration itself, and their records hold the verdict. The audit, the
comparison, the use of each part of the input and the identifiability of the posterior are judged by rules that live
here, each number a named constant with its reason; they are science numbers that live outside the engine. A record
made at less than its check's usual size never decides, whichever way it came out.

Nothing here imports the toolkit, torch or the engine's settings: only the window's table of the engine's lines and
its words for a date and a number, beside the standard library.
"""
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime

from .run import readings
from .texts import date_text, number_text

#: Every state a check's row can read, as the list and a check's page show it.
STATES = ("Not run yet", "Running", "Passed", "Look closer", "Failed", "Waiting for a prior", "Waiting for a posterior",
          "Waiting for the simulations", "Waiting for the audit", "Not compared", "Does not apply", "Done")

#: The frequency band's usual simulations per point: fewer measure each point too loosely to say that it holds.
BAND_LEAST_REPEATS = 24
#: The thresholds that define what "holds" means for the frequency band: the largest spread of the lock-in across
#: repeats, the least signal over its noise, the least share of the cell's own oscillation kept, and the window around
#: the cell's peak; a record made with others measured something else.
BAND_CV_MAX = 0.2
BAND_SNR_MIN = 3.0
BAND_SUP_MIN = 0.5
BAND_PEAK_WINDOW = 0.1
#: The share of unusable driven recordings training is expected to simulate, as a fraction: the reading the retraining
#: runbook gives for an audit of 100 batches of 32 simulations.
AUDIT_EXPECTED_SHARE = 0.37
#: How far from the expected share, in points, the audit may read and pass; and beyond how many points it fails. Between
#: the two it is a reason to look closer. The runbook's reading, for an audit of 100 batches of 32 simulations.
AUDIT_FINE_POINTS = 8.0
AUDIT_FAILED_POINTS = 12.0
#: The audit's usual size, in batches: the runbook's reading assumes 100 batches, and fewer read the share too loosely.
AUDIT_LEAST_BATCHES = 100
#: The audit's usual simulations in each batch, the engine's own: the runbook's reading assumes batches of 32, and
#: smaller ones draw fewer driven recordings than the share is read against.
AUDIT_LEAST_PER_BATCH = 32
#: How far the simulation set's share may lie from the audit's, in points, and still be fine; and beyond how many points
#: the advice is to stop the fit. Between the two it is a reason to look closer. The owner's limits.
COMPARISON_FINE_POINTS = 3.0
COMPARISON_FAILED_POINTS = 5.0
#: The calibration's usual sizes: its test datasets, its operating points (pairs of time scale and recording length) and
#: its samples for each dataset. A calibration made smaller cannot tell an honest posterior from a slightly dishonest
#: one, so it never decides.
CALIBRATION_LEAST_DATASETS = 2000
CALIBRATION_LEAST_POINTS = 200
CALIBRATION_LEAST_SAMPLES = 1000
#: The two parts of the input whose reading decides the input-use check: the mean of the first driven recording's
#: amplitude and the bimodality of the spontaneous recording, the parts the runbook's gate names.
INPUT_USE_PARTS = ("A1_mean", "D3_bimodality")
#: The engine's reading of a part of the input that the network responds to as it does to the typical part.
HEALTHY = "healthy"
#: The input-use check's usual sizes: the simulations it reads and the points of each part's sweep. Fewer say too
#: little of the parts that matter least.
INPUT_USE_LEAST_ROWS = 200_000
INPUT_USE_LEAST_SWEEP = 33
#: The scopes of a training's total of unusable driven recordings that cover the whole simulation set: every batch of a
#: kept set, or a set that is not kept and was simulated whole by the one process. Any other covers part of it.
WHOLE_SET_SCOPES = ("every committed batch of the simulation cache", "this process only (no simulation cache)")
#: How a part's scope names the batches that were saved without their counts.
FIRST_UNCOUNTED = re.compile(r"the cache's first (?P<first>\d[\d,]*) batches")
#: A record's text: each line stamped with its time and its level.
STAMPED = re.compile(r"\d{2}:\d{2}:\d{2} (?P<level>info|warning|error) (?P<text>.*)", re.DOTALL)
#: How the training's total of unusable driven recordings begins, from the window's own table of the engine's lines: a
#: line that does not begin so is not read for it.
MASKED_MARK = next(row.form for row in readings.ROWS if row.name == "chi_total").partition("<")[0]
#: The most decimals a difference is said to: past this it is said as it rounds.
MOST_DECIMALS = 6

#: What a record says when its check's deciding field cannot be read.
UNREADABLE_TEXT = "Its record cannot be read, so it does not count."
#: What a record made at less than its check's usual size says.
REDUCED_TEXT = "Run only at a reduced size ({}; the usual is {}), which does not count."
#: What a frequency-band record made with other thresholds says.
OTHER_THRESHOLDS_TEXT = "Run with other thresholds than the usual ones, which does not count."
#: What an audit made at one fixed number of driven recordings says: the reference the comparison needs is the mixture
#: training draws.
FIXED_AUDIT_TEXT = "Run at a fixed number of driven recordings, which does not count."

#: The comparison's sentences and messages.
NO_AUDIT_TEXT = ("There is no audit of {} to compare with. The simulation set holds {} % unusable driven recordings. "
                 "Run the audit in step 4, Checks, when this run has ended.")
NONE_COUNTED_TEXT = "Not compared: no driven recordings were counted."
PART_TEXT = "Not compared: the count covers {} batches, not the whole simulation set."
PART_FIRST_TEXT = ("Not compared: the count covers {} batches, not the whole simulation set; the first {} were saved "
                   "without their counts.")
FINE_TEXT = "Unusable driven recordings {} %. The audit read {} %; within 3 points is fine."
CLOSER_TEXT = ("Unusable driven recordings {} %. The audit read {} %; they differ by {} points; up to 5 is a reason to "
               "look closer.")
FAILED_TEXT = ("Unusable driven recordings {} %. The audit read {} %; they differ by {} points; beyond 5 the advice is "
               "to stop.")
ACCEPTED_TEXT = "A difference of {} points was accepted on {}."
CLOSER_MESSAGE = ("The simulation set and the audit differ by {} points. Look closer before leaving the fit to run: "
                  "step 4, Checks, shows the per-batch figures of both.")
STOP_KEPT_MESSAGE = ("The advice is to stop the fit and find out why. The fit is saved after every epoch, so a stop "
                     "keeps it and loses at most the epoch in progress; the training can be resumed later.")
STOP_NOT_KEPT_MESSAGE = ("The advice is to stop the fit and find out why. This training keeps no simulation set, so a "
                         "stop loses its simulations and its fit.")


@dataclass(frozen=True)
class Reading:
    """What one record of a check reads: its ``state``, one of ``STATES``; ``words`` said after the state ("" for
    none); and whether it ``counts``: a record that does not count never decides, and its row reads "Not run yet"."""
    state: str
    words: str = ""
    counts: bool = True


@dataclass(frozen=True)
class Comparison:
    """The share of unusable driven recordings in a training's simulation set against its prior's audit: the
    ``difference`` in points (None when nothing is compared), the ``state`` it reads, the ``sentence`` the run's
    health row and the check say, and the ``message`` its page says with it ("" for none)."""
    difference: float | None
    state: str
    sentence: str
    message: str


def read(key: str, manifest: Mapping) -> Reading:
    """The reading of one record of the check ``key``, from its manifest as the store wrote it. Never raises on what a
    manifest holds: one whose deciding field cannot be read does not count. KeyError for a check that writes no record
    (the comparison, the timing and the card test)."""
    reader = _READERS[key]
    try:
        body = manifest["body"]
        return reader(body)
    except (KeyError, TypeError, ValueError, AttributeError, IndexError):
        return Reading("Not run yet", UNREADABLE_TEXT, counts=False)


def _whole(value) -> int:
    """A whole number a record holds; ValueError for anything else."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"not a whole number: {value!r}")
    return value


def _real(value) -> float:
    """A real number a record holds; ValueError for anything else."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"not a number: {value!r}")
    return float(value)


def _flag(value) -> bool:
    """A true or false a record holds; ValueError for anything else."""
    if not isinstance(value, bool):
        raise ValueError(f"not true or false: {value!r}")
    return value


def _counted(count: int, one: str, many: str) -> str:
    """``count`` said with what it counts: ``one``, or ``many`` for any other number."""
    return f"{number_text(count)} {one if count == 1 else many}"


def _reduced(done: int, least: int, one: str, many: str) -> Reading | None:
    """The reading of a record made at ``done`` where the usual size is ``least``, each counting ``one``, or ``many``;
    None when it is at least the usual size."""
    if done >= least:
        return None
    return Reading("Not run yet", REDUCED_TEXT.format(_counted(done, one, many), number_text(least)), counts=False)


def _band(body) -> Reading:
    settings, results = body["settings"], body["results"]
    smaller = _reduced(_whole(settings["repeats"]), BAND_LEAST_REPEATS, "simulation per point", "simulations per point")
    if smaller is not None:
        return smaller
    usual = (BAND_CV_MAX, BAND_SNR_MIN, BAND_SUP_MIN, BAND_PEAK_WINDOW)
    held = tuple(_real(settings[name]) for name in ("cv_max", "snr_min", "sup_min", "peak_window"))
    if held != usual or settings["phase_max"] is not None:
        return Reading("Not run yet", OTHER_THRESHOLDS_TEXT, counts=False)
    holds = (results["band_holds"], results["drive_holds"])
    if any(value is not None and not isinstance(value, bool) for value in holds):
        raise ValueError(f"the band's verdicts are no true or false: {holds!r}")
    if False in holds:
        return Reading("Failed")
    return Reading("Passed") if None not in holds else Reading("Look closer")


def _audit(body) -> Reading:
    settings, results = body["settings"], body["results"]
    batches, per_batch = _whole(settings["num_runs"]), _whole(settings["run_size"])
    if per_batch < AUDIT_LEAST_PER_BATCH:
        size = f"{_counted(batches, 'batch', 'batches')} of {_counted(per_batch, 'simulation', 'simulations')}"
        usual = f"{number_text(AUDIT_LEAST_BATCHES)} of {number_text(AUDIT_LEAST_PER_BATCH)}"
        return Reading("Not run yet", REDUCED_TEXT.format(size, usual), counts=False)
    smaller = _reduced(batches, AUDIT_LEAST_BATCHES, "batch", "batches")
    if smaller is not None:
        return smaller
    if settings["chi_k_fixed"] is not None:
        return Reading("Not run yet", FIXED_AUDIT_TEXT, counts=False)
    holds = (_flag(results["invariants"]["hold"]), _flag(results["attribution"]["holds"]),
             _flag(results["cross_check"]["holds"]))
    away = abs(_real(results["masked_fraction"]) - AUDIT_EXPECTED_SHARE) * 100
    lock_ins = _whole(results["causes"]["non_finite_lock_in"]["count"])
    if not all(holds) or away > AUDIT_FAILED_POINTS:
        return Reading("Failed")
    if away > AUDIT_FINE_POINTS or lock_ins > 0:
        return Reading("Look closer")
    return Reading("Passed")


def _calibration(body) -> Reading:
    results = body["results"]
    for key, least, one, many in (("n_cal", CALIBRATION_LEAST_DATASETS, "test dataset", "test datasets"),
                                  ("cal_n_scales", CALIBRATION_LEAST_POINTS, "operating point", "operating points"),
                                  ("num_posterior_samples", CALIBRATION_LEAST_SAMPLES, "sample per dataset",
                                   "samples per dataset")):
        smaller = _reduced(_whole(results[key]), least, one, many)
        if smaller is not None:
            return smaller
    return Reading("Passed") if _flag(results["verdict"]["passed"]) else Reading("Failed")


def _input_use(body) -> Reading:
    settings, results = body["settings"], body["results"]
    for key, least, one, many in (("rows", INPUT_USE_LEAST_ROWS, "simulation read", "simulations read"),
                                  ("n_sweep", INPUT_USE_LEAST_SWEEP, "point per sweep", "points per sweep")):
        smaller = _reduced(_whole(settings[key]), least, one, many)
        if smaller is not None:
            return smaller
    verdicts = {channel["label"]: channel["verdict"] for channel in results["channels"]}
    parts = [verdicts[label] for label in INPUT_USE_PARTS]
    counts = results["counts"]
    if any(verdict != HEALTHY for verdict in parts):
        return Reading("Failed")
    if _whole(counts["invisible"]) or _whole(counts["nonfinite"]):
        return Reading("Look closer")
    return Reading("Passed")


def _rotation(body) -> Reading:
    eigenvalues = body["results"]["eigenvalues"]
    if eigenvalues is None:
        return Reading("Failed")
    if not isinstance(eigenvalues, list):
        raise ValueError(f"the eigenvalues are no list: {eigenvalues!r}")
    return Reading("Passed")


def _informs(_body) -> Reading:
    return Reading("Done")


#: How a record of each check that writes one is read.
_READERS = {"band": _band, "audit": _audit, "calibration": _calibration, "input_use": _input_use,
            "identifiability": _rotation, "pin_down": _informs, "fixed_number": _informs, "drive_study": _informs,
            "across_prior": _informs}


def _share(fraction: float) -> str:
    """A share as the comparison says it: per cent, to one decimal."""
    return f"{fraction * 100:.1f}"


def _points(difference: float, limit: float) -> str:
    """A difference in points as the comparison says it: to one decimal, or as many more as it takes for a difference
    above ``limit`` not to read as ``limit`` itself."""
    decimals = 1
    while float(f"{difference:.{decimals}f}") <= limit < difference and decimals < MOST_DECIMALS:
        decimals += 1
    return f"{difference:.{decimals}f}"


def _when(moment) -> str:
    """A moment handed as a time or as its text, in the window's words, in local time; "" for neither."""
    if isinstance(moment, str):
        try:
            moment = datetime.fromisoformat(moment)
        except ValueError:
            return ""
    if not isinstance(moment, datetime):
        return ""
    return date_text(moment.astimezone() if moment.tzinfo is not None else moment)


def compare(*, unusable: int, total: int, scope: str, audit_fraction: float | None, accepted: Mapping | None = None,
            batches: int | None = None, prior_words: str = "the prior in use", kept_set: bool = True) -> Comparison:
    """The simulation set's ``unusable`` of ``total`` driven recordings, over ``batches`` batches and of the ``scope``
    the engine's total names, against ``audit_fraction``, the audit's share of its prior (``prior_words``, as a
    sentence names it), None when no audit counts. The difference is taken from the two counts, never the rounded share,
    in points and unrounded. The first that holds decides: nothing counted; a count of part of the set; no audit; then
    the bands. ``accepted``, the acceptance of a difference to look closer at (its moment under "at"), turns that band
    into a pass. ``kept_set`` says whether the training keeps its simulation set, which the advice to stop says. Never
    raises, and never divides by nought."""
    if not isinstance(total, int) or isinstance(total, bool) or total <= 0 or not isinstance(unusable, int):
        return Comparison(None, "Not compared", NONE_COUNTED_TEXT, NONE_COUNTED_TEXT)
    if scope not in WHOLE_SET_SCOPES:
        found = FIRST_UNCOUNTED.search(scope) if isinstance(scope, str) else None
        counted = number_text(batches) if isinstance(batches, int) else "some"
        sentence = (PART_FIRST_TEXT.format(counted, found["first"]) if found is not None
                    else PART_TEXT.format(counted))
        return Comparison(None, "Not compared", sentence, sentence)
    share = unusable / total
    if not isinstance(audit_fraction, (int, float)) or isinstance(audit_fraction, bool):
        sentence = NO_AUDIT_TEXT.format(prior_words, _share(share))
        return Comparison(None, "Waiting for the audit", sentence, sentence)
    difference = abs(share - audit_fraction) * 100
    ours, theirs = _share(share), _share(audit_fraction)
    if difference <= COMPARISON_FINE_POINTS:
        return Comparison(difference, "Passed", FINE_TEXT.format(ours, theirs), "")
    if difference <= COMPARISON_FAILED_POINTS:
        said = _points(difference, COMPARISON_FINE_POINTS)
        sentence = CLOSER_TEXT.format(ours, theirs, said)
        moment = _when(accepted.get("at")) if isinstance(accepted, Mapping) else ""
        if moment:
            return Comparison(difference, "Passed", f"{sentence} {ACCEPTED_TEXT.format(said, moment)}", "")
        return Comparison(difference, "Look closer", sentence, CLOSER_MESSAGE.format(said))
    said = _points(difference, COMPARISON_FAILED_POINTS)
    return Comparison(difference, "Failed", FAILED_TEXT.format(ours, theirs, said),
                      STOP_KEPT_MESSAGE if kept_set else STOP_NOT_KEPT_MESSAGE)


def first_masked(text: str) -> tuple[int, int, int, str] | None:
    """The first total of unusable driven recordings in a record's text, as the window's table of the engine's lines
    reads it: the unusable, the total, the batches counted and the scope; a total that counted none reads 0, 0 and 0.
    None when the text holds none, or is no text."""
    if not isinstance(text, str):
        return None
    for line in text.splitlines():
        if MASKED_MARK not in line:
            continue
        stamped = STAMPED.fullmatch(line.strip())
        said, level = (stamped["text"], stamped["level"]) if stamped is not None else (line, "info")
        reading = readings.read(said, level)
        if reading is None:
            continue
        if reading.name == "chi_total":
            values = reading.values
            return values["masked"], values["total"], values["batches"], values["scope"]
        if reading.name == "chi_none":
            return 0, 0, 0, reading.values["scope"]
    return None
