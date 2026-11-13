"""How a check's record is read, as values: the state a record of each deciding check reads, by the rules a deciding
check is judged by, and the comparison of the share of unusable driven recordings a training's simulation set holds
with the audit of its prior; what each check's record says, in the window's words: the sentence its result begins
with and its table; and what "Before you start" says of a check's run before it is started.

The engine decides the frequency band and the calibration itself, and their records hold the verdict. The audit, the
comparison, the use of each part of the input and the identifiability of the posterior are judged by rules that live
here, each number a named constant with its reason; they are science numbers that live outside the engine. A record
made at less than its check's usual size never decides, whichever way it came out.

Nothing here imports the toolkit, torch or the engine's settings: only the window's table of the engine's lines, the
twelve checks, its words for a date and a number and its table as words, beside the standard library.
"""
import re
import statistics
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import PureWindowsPath

from .check_catalogue import entry
from .run import readings
from .step_facts import Fact, TableWords
from .texts import date_text, duration_text, number_text

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
#: The same, as the comparison's page says it afterwards, with no run live: the audit's row is the one to run.
NO_AUDIT_AFTER_TEXT = ("There is no audit of {} to compare with. The simulation set holds {} % unusable driven "
                       "recordings; run \"{}\" in this list to compare them.")
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


# ── what a record says, in the window's words ──
#: The marker each state is drawn with: waiting or not run yet, running, passed, to look closer at or not compared,
#: failed, not applying, and an informing check's or a tool's done.
STATE_MARKERS = {"Not run yet": "later", "Running": "now", "Passed": "fine", "Look closer": "warn", "Failed": "err",
                 "Waiting for a prior": "later", "Waiting for a posterior": "later",
                 "Waiting for the simulations": "later", "Waiting for the audit": "later", "Not compared": "warn",
                 "Does not apply": "none", "Done": "done"}
#: How a verdict sentence says a deciding check's state after the check's name; an informing check or a tool says it
#: is done.
VERDICT_WORDS = {"Passed": "passed", "Look closer": "look closer", "Failed": "failed"}
DONE_WORDS = "done"
#: What a result says when the record holds less than its sentence reads.
SHORT_TEXT = "{}: {}."

#: The frequency band's sentence: the cell, the band and the drive amplitude, each clause by its verdict (true, false,
#: or not judged).
BAND_HEAD_TEXT = "{}: {} for cell {}."
BAND_CLAUSES = {
    True: "Every frequency inside the band ({} to {} times the cell's own) can be measured at every recording length "
          "tried",
    False: "Not every frequency inside the band ({} to {} times the cell's own) can be measured at every recording "
           "length tried",
    None: "Whether the frequencies inside the band ({} to {} times the cell's own) can be measured was not judged"}
DRIVE_CLAUSES = {True: "the drive amplitude {} does not take over the cell",
                 False: "the drive amplitude {} takes over the cell",
                 None: "whether the drive amplitude {} takes over the cell was not judged"}
#: The audit's sentence.
AUDIT_SHARE_TEXT = "{} % of {} driven recordings simulated from {} were unusable."
AUDIT_EXPECTED_TEXT = "The expected share is {} %; within {} points is fine."
AUDIT_ALL_SHORT_TEXT = "Every one was too short or too slow to hold {} of the drive."
AUDIT_SOME_SHORT_TEXT = "{} were too short or too slow to hold {} of the drive, and {} gave no finite reading."
AUDIT_UNHELD_TEXT = "One of its consistency records does not hold, so its share cannot be relied on."
#: The comparison's sentence once it is made.
COMPARED_TEXT = ("{}: {}. {} % in the simulation set; the audit read {} %. The difference is {} points: within 3 is "
                 "fine, up to 5 look closer, beyond 5 stop.")
#: The narrowing round's sentence under the comparison.
NARROWED_COMPARISON_TEXT = "The audit measured the whole prior; this round simulated inside a narrowed region."
#: The calibration's sentence: the rank test, the joint coverage test, and what it was made on.
RANKS_ALL_TEXT = "All {} parameters pass the rank test"
RANKS_SOME_TEXT = "{} of {} parameters fail the rank test: {}"
JOINT_WORDS = {True: "passes", False: "fails"}
CALIBRATION_MADE_TEXT = ("Made on {} test datasets at {} operating points (pairs of time scale and recording "
                         "length){}.")
SEED_TEXT = ", seed {}"
NARROWED_CALIBRATION_TEXT = "It was made inside the region its posterior was narrowed to."
ALONE_TEXT = "Calibration alone does not show that the posterior is sound: {} still open."
#: The input-use check's sentence.
INPUT_USE_TEXT = "The network responds to {} of the {} parts of an observation's summary; {}; {}{}."
CONSTANT_WORDS = ("every part varies in the training simulations", "1 never varies in the training simulations",
                  "{} never vary in the training simulations")
INVISIBLE_WORDS = ("none is invisible to it", "1 is invisible to it", "{} are invisible to it")
NONFINITE_WORDS = ("", "; 1 drives it to no finite output", "; {} drive it to no finite output")
NAMED_PARTS_TEXT = "Of the two parts it is judged by, {} do not respond as the typical part does."
#: The identifiability of the posterior's sentence.
DIRECTIONS_TEXT = ("The record holds how strongly each of the {} directions is pinned down: the strongest {} times the "
                   "weakest. This check passes when those figures are on record; the table says what they are.")
NO_DIRECTIONS_TEXT = "The record does not hold how strongly the directions are pinned down: its figures were not kept."
#: The informing checks' and the tools' sentences.
PIN_DOWN_TEXT = "At {} s, {} and {}."
NO_INFORMATION_WORDS = ("every parameter has local information", "1 parameter has no local information",
                        "{} parameters have no local information")
ALIKE_WORDS = ("no pair moves an observation alike", "1 pair moves an observation alike",
               "{} pairs move an observation alike")
FIXED_TEXT = ("{}, {} of {} repeats pass the rank test. This is the rank half only; the joint coverage test is "
              "Calibration's.")
FIXED_WORDS = "With {} driven recordings"
MIXED_WORDS = "With the number of driven recordings mixed, as in training"
DRIVE_BOTH_TEXT = "The cell keeps its own oscillation up to a drive of {} and is captured from {}."
DRIVE_FREE_TEXT = "The cell keeps its own oscillation up to a drive of {}; no drive tried captures it."
DRIVE_CAPTURED_TEXT = "The cell is captured from a drive of {}; it keeps its own oscillation at no drive tried."
DRIVE_NONE_TEXT = "No drive tried was judged free-running or captured."
NOT_OSCILLATING_TEXT = "The cell shows no clear oscillation of its own, so no drive was judged."
ACROSS_TEXT = ("{}: {}, at {} points across the prior. The table gives each parameter's spread at each point and the "
               "share of points where recordings pin it down.")
#: A check that keeps nothing but its result, said from the run that made it.
RUN_WORDS = {"finished": "done", "failed": "failed", "stopped": "stopped", "not seen to end": "not seen to end"}

#: The frequency band's table.
BAND_COLUMNS = ("Multiple of the cell's own frequency", "Inside the band", "Result", "Reasons")
BAND_FREQUENCIES_HEAD = "Each frequency"
BAND_DRIVES_HEAD = "Each drive amplitude"
YES_NO = {True: "Yes", False: "No"}
BAND_RESULTS = ("Measured at every length", "Measured within the cycle ceiling", "Not measured at every length")
DRIVE_RESULTS = {True: "Does not take over the cell", None: "Not judged"}
DRIVE_TAKES_TEXT = "Takes over the cell at {} inside the band"
#: What a point of the band says when it does not pass, built from its measures against the thresholds it was judged
#: by, and where it stood.
POINT_CAUSES = ("varies between runs", "weak over the noise", "taken over by the drive")
POINT_MASKED = "too fast to sample at {} s"
POINT_UNMEASURED = "not measured"
POINT_AT = "{} at {} s"
POINT_FULL = " without the ceiling"
POINT_DRIVE = ", drive {}"
#: The audit's table.
AUDIT_COLUMNS = ("Cause", "Driven recordings", "Share")
AUDIT_RIGHT = frozenset({1, 2})
AUDIT_CAUSES = (("too_slow_at_band_top", "Too slow to hold {} of the drive"),
                ("shortened_by_duration_draw", "Too short to hold {} of the drive"),
                ("non_finite_lock_in", "Gave no finite reading"))
AUDIT_ALL = "All unusable"
PER_BATCH_TEXT = "Per batch: mean {} %, spread {} points, over {}."
BATCH_WORDS = ("no batch", "1 batch", "{} batches")
#: The calibration's table.
CALIBRATION_COLUMNS = ("Parameter", "Unit", "p-value", "Result", "Remark")
CALIBRATION_RIGHT = frozenset({2})
PASS_WORDS = {True: "Passes", False: "Fails"}
TOGETHER = "All together"
TOGETHER_REMARK = "The joint coverage test; it passes at p = {} or more"
ASSUMED_REMARK = "An assumed input"
TIME_SCALE = "t_scale"
TIME_SCALE_REMARK = "Rests on the operating points, not the datasets, so it has less power"
RANK_FOOTNOTE = "Rank test: a parameter passes at p = {} or more ({} divided by {})."
READING_HEAD = "A reading, not a test"
INFORMATION_ROWS = (("total_nats", "Information gained"), ("sem_nats", "Its standard error"))
NATS = "nats"
#: The input-use check's table, weakest first, and its readings of each part.
INPUT_USE_COLUMNS = ("Part", "Response", "Against the typical part", "Range in training", "Reading")
INPUT_USE_RIGHT = frozenset({1, 2})
PART_READINGS = {"healthy": "Responds as the typical part does", "compressed": "Responds weakly",
                 "severely compressed": "Responds very weakly"}
INVISIBLE_READING = "Invisible to the network"
NONFINITE_READING = "Drives it to no finite output"
CONSTANT_READING = "Never varies in training"
RANGE_TEXT = "{} to {}"
#: The identifiability of the posterior's table.
DIRECTION_COLUMNS = ("Parameter or direction", "Figure", "Reading")
DIRECTION_RIGHT = frozenset({1})
PARAMETERS_HEAD = "Each parameter: its share in the weakest directions"
DIRECTIONS_HEAD = "Each direction: its strength and its largest contributions"
DIRECTION_NAME = "Direction {}"
PARAMETER_READINGS = {"good": "Well pinned down", "moderate": "Moderately pinned down", "poor": "Poorly pinned down",
                      "very poor": "Very poorly pinned down", "UNMEASURED": "Not pinned down"}
SHARE_TEXT = "{} %"
#: What the observations can pin down: its table, and the line of its text that says from which recording length the
#: driven recordings at the low end of the band hold enough cycles.
PIN_DOWN_COLUMNS = ("Parameter or pair", "Reading")
NO_INFORMATION_HEAD = "No local information"
ALIKE_HEAD = "Move an observation alike"
PAIR_TEXT = "{} and {}"
ALIKE_TEXT = "Alike to {}"
LOW_EDGE = re.compile(r"low edge \S+x clears (?P<cycles>[0-9.]+) cycles at T_obs >= (?P<seconds>[0-9.e+-]+) s")
LOW_EDGE_TEXT = "Driven recordings at the low end of the band hold {} from {} s."
#: The calibration repeated at fixed numbers' table.
FIXED_COLUMNS = ("Parameter", "Median p-value", "Lowest", "Repeats below 0.05")
FIXED_RIGHT = frozenset({1, 2, 3})
#: The drive study's table.
DRIVE_COLUMNS = ("Strength", "Force", "Own oscillation kept", "Reading")
DRIVE_RIGHT = frozenset({1, 2})
DRIVE_READINGS = {"free-running": "Keeps its own oscillation", "captured": "Captured by the drive",
                  "in between": "Partly kept"}
#: Identifiability across the prior's table.
ACROSS_COLUMNS = ("Parameter", "Spread at each point", "Median", "Points pinned down")
ACROSS_RIGHT = frozenset({2, 3})
#: The cycles of the drive a sentence says in a word: the engine's usual floor.
CYCLE_WORDS = {2: "two cycles"}
#: The most significant figures a measured number is said to.
FIGURES = 3
#: From how large, and below how small, a number is said with its power of ten; and the figures of a large ratio, as
#: a sentence says how much the strongest direction is pinned down beyond the weakest.
POWER_FROM = 10_000
SMALL_FROM = 0.001
RATIO_FIGURES = 2


def said_name(key: str) -> str:
    """A check's name as its sentences begin with it: a name in two parts is said with a comma between them."""
    return entry(key).name.replace(": ", ", ")


def _figure(value, figures: int = FIGURES) -> str:
    """A measured number to ``figures`` significant figures, with its power of ten when it is very large or very small
    ("1.2e6", "1.05e-9"), its thousands grouped otherwise; "" for none."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return ""
    size = abs(value)
    if size and (size >= POWER_FROM or size < SMALL_FROM):
        mantissa, power = f"{value:.{figures - 1}e}".split("e")
        return f"{mantissa}e{int(power)}"
    rounded = float(f"{value:.{figures}g}")
    return number_text(int(rounded) if rounded.is_integer() else rounded)


def _ratio(value: float) -> str:
    """A ratio as a sentence says it: with its power of ten when it is large ("1.2e6"), else to its figures."""
    return _figure(value, RATIO_FIGURES) if value >= POWER_FROM else _figure(value)


def _counts_said(count: int, words: tuple) -> str:
    """``count`` by the three forms of ``words``: none, one, and several."""
    none, one, several = words
    return none if count == 0 else one if count == 1 else several.format(number_text(count))


def _cycles(value) -> str:
    """How many cycles of the drive, as a sentence says it."""
    return CYCLE_WORDS.get(value, f"{value:g} cycles")


def _cell_name(manifest: Mapping) -> str:
    """The name of the cell file a record was made on, without its folder or its ending; "" for none."""
    inputs = manifest.get("inputs")
    cell = inputs.get("cell") if isinstance(inputs, Mapping) else None
    path = cell.get("path") if isinstance(cell, Mapping) else None
    return PureWindowsPath(path).stem if isinstance(path, str) and path else ""


def _body(manifest: Mapping) -> Mapping:
    body = manifest.get("body")
    return body if isinstance(body, Mapping) else {}


def verdict_sentence(key: str, manifest: Mapping, *, others_open: int = 0, prior_words: str = "",
                     force_unit: str = "") -> str:
    """The sentence a check's result begins with, from its record's manifest, in the window's words and never the
    engine's own: the check's name, its state, and what its record holds. ``others_open``, for a passed calibration,
    how many other deciding checks that apply have not passed; ``prior_words``, the audit's prior as a sentence names
    it; ``force_unit``, the cell's unit of force, for the drive study. A record that holds less than its sentence reads
    is said by its name and its state alone, and a deciding record that does not count by its name and what its row
    says of it. Never raises; KeyError for a check that writes no record."""
    reading = read(key, manifest)
    name = said_name(key)
    if entry(key).role == "Decides" and reading.state not in VERDICT_WORDS:
        return f"{SHORT_TEXT.format(name, reading.state.lower())} {reading.words}".rstrip()
    word = VERDICT_WORDS[reading.state] if entry(key).role == "Decides" else DONE_WORDS
    try:
        return _SENTENCES[key](name, word, _body(manifest), manifest, others_open=others_open,
                               prior_words=prior_words, force_unit=force_unit)
    except (KeyError, TypeError, ValueError, AttributeError, IndexError, ZeroDivisionError):
        return SHORT_TEXT.format(name, word)


def _band_sentence(name, word, body, manifest, **_given) -> str:
    configured, results = body["settings"]["configured"], body["results"]
    low, high = (number_text(float(value)) for value in configured["chi_freq_bounds"])
    drive = number_text(float(configured["chi_f0"]))
    cell = _cell_name(manifest)
    head = BAND_HEAD_TEXT.format(name, word, cell) if cell else SHORT_TEXT.format(name, word)
    return (f"{head} {BAND_CLAUSES[results['band_holds']].format(low, high)}, and "
            f"{DRIVE_CLAUSES[results['drive_holds']].format(drive)}.")


def _audit_sentence(name, word, body, _manifest, *, prior_words, **_given) -> str:
    settings, results = body["settings"], body["results"]
    cycles = _cycles(settings["configured"]["chi_min_cycles"])
    said = [SHORT_TEXT.format(name, word),
            AUDIT_SHARE_TEXT.format(f"{_real(results['masked_fraction']) * 100:.1f}",
                                    number_text(_whole(results["probes"])), prior_words or "its prior"),
            AUDIT_EXPECTED_TEXT.format(f"{AUDIT_EXPECTED_SHARE * 100:g}", f"{AUDIT_FINE_POINTS:g}")]
    masked = _whole(results["masked"])
    lock_ins = _whole(results["causes"]["non_finite_lock_in"]["count"])
    if masked and not lock_ins:
        said.append(AUDIT_ALL_SHORT_TEXT.format(cycles))
    elif masked:
        said.append(AUDIT_SOME_SHORT_TEXT.format(number_text(masked - lock_ins), cycles, number_text(lock_ins)))
    holds = (results["invariants"]["hold"], results["attribution"]["holds"], results["cross_check"]["holds"])
    if not all(value is True for value in holds):
        said.append(AUDIT_UNHELD_TEXT)
    return " ".join(said)


def _calibration_sentence(name, word, body, _manifest, *, others_open, **_given) -> str:
    results = body["results"]
    verdict = results["verdict"]
    parameters = verdict["parameters"]
    failing = [one["name"] for one in parameters if one["passed"] is not True]
    joint = verdict["tarp"]["passed"] is True
    if failing:
        ranks = f"{RANKS_SOME_TEXT.format(len(failing), len(parameters), ', '.join(failing))}; the joint coverage " \
                f"test {JOINT_WORDS[joint]}."
    else:
        ranks = (f"{RANKS_ALL_TEXT.format(len(parameters))}, {'and' if joint else 'but'} the joint coverage test "
                 f"{JOINT_WORDS[joint]}.")
    seed = results.get("seed")
    made = CALIBRATION_MADE_TEXT.format(number_text(_whole(results["n_cal"])), number_text(_whole(results["cal_n_scales"])),
                                        SEED_TEXT.format(seed) if isinstance(seed, int) and not isinstance(seed, bool)
                                        else "")
    said = [SHORT_TEXT.format(name, word), ranks, made]
    if "truncated" in (results.get("accepted") or ()):
        said.append(NARROWED_CALIBRATION_TEXT)
    if word == VERDICT_WORDS["Passed"] and others_open > 0:
        said.append(ALONE_TEXT.format("1 deciding check is" if others_open == 1
                                      else f"{number_text(others_open)} deciding checks are"))
    return " ".join(said)


def _input_use_sentence(name, word, body, _manifest, **_given) -> str:
    results = body["results"]
    counts = results["counts"]
    total, constant = _whole(counts["total"]), _whole(counts["constant"])
    invisible, nonfinite = _whole(counts["invisible"]), _whole(counts["nonfinite"])
    parts = INPUT_USE_TEXT.format(number_text(total - constant - invisible - nonfinite), number_text(total),
                                  _counts_said(constant, CONSTANT_WORDS), _counts_said(invisible, INVISIBLE_WORDS),
                                  _counts_said(nonfinite, NONFINITE_WORDS))
    said = [SHORT_TEXT.format(name, word), parts]
    verdicts = {channel["label"]: channel["verdict"] for channel in results["channels"]}
    weak = [label for label in INPUT_USE_PARTS if verdicts.get(label) != HEALTHY]
    if weak:
        said.append(NAMED_PARTS_TEXT.format(" and ".join(weak)))
    return " ".join(said)


def _identifiability_sentence(name, word, body, _manifest, **_given) -> str:
    eigenvalues = body["results"]["eigenvalues"]
    if eigenvalues is None:
        return f"{SHORT_TEXT.format(name, word)} {NO_DIRECTIONS_TEXT}"
    strengths = [_real(value) for value in eigenvalues]
    return (f"{SHORT_TEXT.format(name, word)} "
            f"{DIRECTIONS_TEXT.format(number_text(len(strengths)), _ratio(max(strengths) / min(strengths)))}")


def _pin_down_sentence(name, word, body, _manifest, **_given) -> str:
    results = body["results"]
    length = f"{_real(results['T_obs_s']):g}"
    said = PIN_DOWN_TEXT.format(length, _counts_said(len(results["unmeasurable"]), NO_INFORMATION_WORDS),
                                _counts_said(len(results["degenerate_pairs"]), ALIKE_WORDS))
    return f"{SHORT_TEXT.format(name, word)} {said}"


def _fixed_sentence(name, word, body, _manifest, **_given) -> str:
    fixed, verdict = body["settings"]["chi_k_fixed"], body["results"]["rank_verdict"]
    repeats = verdict["per_repeat"]
    how = MIXED_WORDS if fixed is None else FIXED_WORDS.format(number_text(_whole(fixed)))
    return (f"{SHORT_TEXT.format(name, word)} "
            f"{FIXED_TEXT.format(how, number_text(sum(one is True for one in repeats)), number_text(len(repeats)))}")


def _drive_sentence(name, word, body, _manifest, *, force_unit, **_given) -> str:
    results = body["results"]
    if results["oscillating"] is not True:
        return f"{SHORT_TEXT.format(name, word)} {NOT_OSCILLATING_TEXT}"

    def force(pick):
        return None if pick is None else f"{_figure(_real(pick['cell_force']))} {force_unit}".rstrip()

    free, captured = force(results["strongest_free_running"]), force(results["weakest_captured"])
    said = (DRIVE_BOTH_TEXT.format(free, captured) if free and captured else DRIVE_FREE_TEXT.format(free) if free
            else DRIVE_CAPTURED_TEXT.format(captured) if captured else DRIVE_NONE_TEXT)
    return f"{SHORT_TEXT.format(name, word)} {said}"


def _across_sentence(name, word, body, _manifest, **_given) -> str:
    return ACROSS_TEXT.format(name, word, number_text(len(body["results"]["points"])))


#: How each check that writes a record says its result.
_SENTENCES = {"band": _band_sentence, "audit": _audit_sentence, "calibration": _calibration_sentence,
              "input_use": _input_use_sentence, "identifiability": _identifiability_sentence,
              "pin_down": _pin_down_sentence, "fixed_number": _fixed_sentence, "drive_study": _drive_sentence,
              "across_prior": _across_sentence}


def compared_sentence(compared: Comparison, *, unusable: int, total: int, audit_fraction,
                      prior_words: str = "the prior in use") -> str:
    """The comparison after simulating as its page says it afterwards: its state and the two shares with their
    difference; waiting for the audit, that there is none of ``prior_words`` to compare with and which check to run;
    else, nothing compared or a part of the set, the comparison's own sentence."""
    if compared.state == "Waiting for the audit" and isinstance(total, int) and total > 0:
        return NO_AUDIT_AFTER_TEXT.format(prior_words, _share(unusable / total), entry("audit").name)
    if compared.difference is None or compared.state not in VERDICT_WORDS:
        return compared.sentence
    limit = COMPARISON_FINE_POINTS if compared.difference <= COMPARISON_FAILED_POINTS else COMPARISON_FAILED_POINTS
    return COMPARED_TEXT.format(said_name("comparison"), VERDICT_WORDS[compared.state], _share(unusable / total),
                                _share(audit_fraction), _points(compared.difference, limit))


def run_sentence(key: str, entry_: Mapping) -> str:
    """The sentence of a check that keeps nothing but its result, from the run of it the history holds: its name and
    how the run ended."""
    outcome = entry_.get("outcome") if isinstance(entry_, Mapping) else None
    return SHORT_TEXT.format(said_name(key), RUN_WORDS.get(outcome, RUN_WORDS["not seen to end"]))


def table_words(key: str, manifest: Mapping, *, units: Mapping[str, str] = {}, force_unit: str = "",
                text: str | None = None) -> TableWords | None:
    """The table of a check's result, from its record's manifest: its columns in the window's words, every row, and
    the record's own strings never shown. ``units``, each parameter's unit by its name; ``force_unit``, the cell's
    unit of force; ``text``, the record's text, for what the observations can pin down. None for a check whose record
    holds no table, or a record that holds less than its table reads. Never raises."""
    maker = _TABLES.get(key)
    if maker is None:
        return None
    try:
        return maker(_body(manifest), units=units, force_unit=force_unit, text=text)
    except (KeyError, TypeError, ValueError, AttributeError, IndexError, ZeroDivisionError):
        return None


def _point_causes(measure: Mapping, sup, settings: Mapping) -> str:
    """What a point of the band misses, by its measures against the thresholds it was judged by."""
    misses = [words for words, missed in zip(POINT_CAUSES, (
        isinstance(measure.get("cv"), (int, float)) and measure["cv"] > settings["cv_max"],
        isinstance(measure.get("snr"), (int, float)) and measure["snr"] < settings["snr_min"],
        isinstance(sup, (int, float)) and sup < settings["sup_min"])) if missed]
    return ", ".join(misses) or POINT_UNMEASURED


def _band_reasons(points, multiplier, settings) -> str:
    """Why the frequency ``multiplier`` is not measured at every length, from its points, the lengths of each reason
    together; "" when it is."""
    several = len(settings.get("drives") or ()) > 1
    lengths: dict = {}
    for point in points:
        if point.get("multiplier") != multiplier:
            continue
        capped, full = point.get("capped") or {}, point.get("full") or {}
        drive = POINT_DRIVE.format(f"{_real(point['drive']):g}") if several and "drive" in point else ""
        if capped.get("verdict") == "masked":
            why = (POINT_MASKED, drive)
        elif capped.get("verdict") != "pass":
            why = (POINT_AT.format(_point_causes(capped, point.get("sup"), settings), "{}"), drive)
        elif full.get("verdict") != "pass":
            why = (POINT_AT.format(_point_causes(full, point.get("sup"), settings), "{}") + POINT_FULL, drive)
        else:
            continue
        said = lengths.setdefault(why, [])
        length = f"{_real(point['length_s']):g}"
        if length not in said:
            said.append(length)
    said = "; ".join(words.format(_listed(lengths_)) + drive for (words, drive), lengths_ in lengths.items())
    return said[:1].upper() + said[1:]


def _listed(words: list) -> str:
    """Words listed as a sentence lists them: the last after "and"."""
    return words[0] if len(words) == 1 else f"{', '.join(words[:-1])} and {words[-1]}"


def _band_table(body, **_given) -> TableWords:
    settings, results = body["settings"], body["results"]
    points = results.get("points") or ()
    measured, capped, not_measured = BAND_RESULTS
    rows, headings = [], set()
    headings.add(len(rows))
    rows.append((BAND_FREQUENCIES_HEAD, "", "", ""))
    for one in results["frequencies"]:
        result = (measured if one["passes_full"] is True else capped if one["passes_capped"] is True
                  else not_measured)
        rows.append((f"{_real(one['multiplier']):g}", YES_NO[one["in_band"] is True], result,
                     _band_reasons(points, one["multiplier"], settings)))
    headings.add(len(rows))
    rows.append((BAND_DRIVES_HEAD, "", "", ""))
    for one in results["drives"]:
        captured = one["captured_in_band"]
        result = (DRIVE_TAKES_TEXT.format(_counts_said(len(captured), ("", "1 point", "{} points")))
                  if captured else DRIVE_RESULTS[True])
        rows.append((f"{_real(one['drive']):g}", "", result, ""))
    return TableWords(BAND_COLUMNS, tuple(rows), headings=frozenset(headings))


def _audit_table(body, **_given) -> TableWords:
    settings, results = body["settings"], body["results"]
    cycles = _cycles(settings["configured"]["chi_min_cycles"])
    causes = results["causes"]
    rows = [(words.format(cycles) if "{}" in words else words, number_text(_whole(causes[cause]["count"])),
             SHARE_TEXT.format(f"{_real(causes[cause]['share']) * 100:.1f}")) for cause, words in AUDIT_CAUSES]
    rows.append((AUDIT_ALL, number_text(_whole(results["masked"])),
                 SHARE_TEXT.format(f"{_real(results['masked_fraction']) * 100:.1f}")))
    footnote = ""
    per_batch = (results.get("per_batch") or {}).get("fractions")
    if isinstance(per_batch, list) and per_batch:
        shares = [_real(value) * 100 for value in per_batch]
        footnote = PER_BATCH_TEXT.format(f"{statistics.fmean(shares):.1f}", f"{statistics.pstdev(shares):.1f}",
                                         _counts_said(len(shares), BATCH_WORDS))
    return TableWords(AUDIT_COLUMNS, tuple(rows), right=AUDIT_RIGHT, footnote=footnote)


def _calibration_table(body, *, units, **_given) -> TableWords:
    results = body["results"]
    verdict = results["verdict"]
    caveat = bool(verdict.get("caveat"))
    rows = []
    for one in verdict["parameters"]:
        remarks = [words for words, holds in ((ASSUMED_REMARK, one.get("assumed") is True),
                                              (TIME_SCALE_REMARK, caveat and one["name"] == TIME_SCALE)) if holds]
        rows.append((one["name"], units.get(one["name"], ""), _figure(one["ks_p"]), PASS_WORDS[one["passed"] is True],
                     "; ".join(remarks)))
    tarp, alpha = verdict["tarp"], _real(verdict["alpha"])
    rows.append((TOGETHER, "", _figure(tarp["ks_p"]), PASS_WORDS[tarp["passed"] is True],
                 TOGETHER_REMARK.format(f"{alpha:g}")))
    headings = set()
    information = results.get("informativeness")
    if isinstance(information, Mapping):
        headings.add(len(rows))
        rows.append((READING_HEAD, "", "", "", ""))
        rows += [(words, NATS, _figure(information.get(name)), "", "") for name, words in INFORMATION_ROWS]
    footnote = RANK_FOOTNOTE.format(f"{_real(verdict['threshold']):.2g}", f"{alpha:g}",
                                    number_text(len(verdict["parameters"])))
    return TableWords(CALIBRATION_COLUMNS, tuple(rows), right=CALIBRATION_RIGHT, headings=frozenset(headings),
                      footnote=footnote)


def _part_reading(verdict) -> str:
    """The window's reading of the engine's verdict on one part of the input."""
    if not isinstance(verdict, str):
        return ""
    if verdict in PART_READINGS:
        return PART_READINGS[verdict]
    if "INVISIBLE" in verdict:
        return INVISIBLE_READING
    if verdict.startswith("NON-FINITE"):
        return NONFINITE_READING
    return CONSTANT_READING if verdict.startswith("constant") else ""


def _input_use_table(body, **_given) -> TableWords:
    rows = tuple((one["label"], _figure(one["max_disp"]), _figure(one["rel_median"]),
                  RANGE_TEXT.format(_figure(one["p1"]), _figure(one["p99"])) if one.get("p1") is not None else "",
                  _part_reading(one["verdict"])) for one in body["results"]["channels"])
    return TableWords(INPUT_USE_COLUMNS, rows, right=INPUT_USE_RIGHT)


def _identifiability_table(body, **_given) -> TableWords:
    results = body["results"]
    rows, headings = [], {0}
    rows.append((PARAMETERS_HEAD, "", ""))
    rows += [(one["name"], SHARE_TEXT.format(f"{_real(one['bottom_share']) * 100:.0f}"),
              PARAMETER_READINGS.get(one.get("verdict"), "")) for one in results["per_param"]]
    if results.get("eigenvalues") is not None:
        headings.add(len(rows))
        rows.append((DIRECTIONS_HEAD, "", ""))
        rows += [(DIRECTION_NAME.format(_whole(one["index"]) + 1), _ratio(_real(one["eigenvalue"])),
                  ", ".join(f"{part['name']} {_real(part['loading']):.2f}" for part in one["loadings"]))
                 for one in results["directions"]]
    return TableWords(DIRECTION_COLUMNS, tuple(rows), right=DIRECTION_RIGHT, headings=frozenset(headings))


def low_edge_words(text: str | None) -> str:
    """From a record's text of what the observations can pin down, the recording length from which the driven
    recordings at the low end of the band hold enough cycles, as a sentence; "" when the text says none."""
    found = LOW_EDGE.search(text) if isinstance(text, str) else None
    if found is None:
        return ""
    return LOW_EDGE_TEXT.format(_cycles(float(found["cycles"])), found["seconds"])


def _pin_down_table(body, *, text, **_given) -> TableWords | None:
    results = body["results"]
    unmeasurable, pairs = results["unmeasurable"], results["degenerate_pairs"]
    caption = low_edge_words(text)
    if not unmeasurable and not pairs and not caption:
        return None
    rows, headings = [], set()
    if unmeasurable:
        headings.add(len(rows))
        rows.append((NO_INFORMATION_HEAD, ""))
        rows += [(str(name), "") for name in unmeasurable]
    if pairs:
        headings.add(len(rows))
        rows.append((ALIKE_HEAD, ""))
        rows += [(PAIR_TEXT.format(one["a"], one["b"]), ALIKE_TEXT.format(_figure(one["cos"]))) for one in pairs]
    return TableWords(PIN_DOWN_COLUMNS, tuple(rows), headings=frozenset(headings), caption=caption)


def _fixed_table(body, **_given) -> TableWords:
    rows = tuple((one["name"], _figure(one["ks_p_median"]), _figure(one["ks_p_min"]),
                  SHARE_TEXT.format(f"{_real(one['frac_ks_below_05']) * 100:.0f}"))
                 for one in body["results"]["per_param"])
    return TableWords(FIXED_COLUMNS, rows, right=FIXED_RIGHT)


def _drive_table(body, *, force_unit, **_given) -> TableWords:
    rows = tuple((f"{_real(one['strength']):g}", f"{_figure(one['cell_force'])} {force_unit}".rstrip(),
                  SHARE_TEXT.format(f"{_real(one['own_peak_ratio']) * 100:.0f}")
                  if isinstance(one.get("own_peak_ratio"), (int, float)) else "",
                  DRIVE_READINGS.get(one.get("verdict"), "")) for one in body["results"]["strengths"])
    return TableWords(DRIVE_COLUMNS, rows, right=DRIVE_RIGHT)


def _across_table(body, **_given) -> TableWords:
    rows = tuple((one["name"], ", ".join(_figure(value) for value in one["sd"]), _figure(one["median_sd"]),
                  SHARE_TEXT.format(f"{_real(one['frac_identified']) * 100:.0f}"))
                 for one in body["results"]["per_param"])
    return TableWords(ACROSS_COLUMNS, rows, right=ACROSS_RIGHT)


#: How each check that writes a record lays out its table.
_TABLES = {"band": _band_table, "audit": _audit_table, "calibration": _calibration_table,
           "input_use": _input_use_table, "identifiability": _identifiability_table, "pin_down": _pin_down_table,
           "fixed_number": _fixed_table, "drive_study": _drive_table, "across_prior": _across_table}


# ── "Before you start": what a check's run does, said before it is started ──
#: The calibration's counts: one batch for each operating point, holding the test datasets, then the samples drawn for
#: each dataset.
CALIBRATION_COUNTS_TEXT = "Simulates {} batches, holding {} test datasets, then draws {} samples for each."
#: The frequency band's counts, at the recording lengths the engine chooses for the cell when none is typed (five, its
#: own number), or at those typed.
BAND_COUNTS_TEXT = "Simulates {} recordings at each recording length and frequency of the band: {}."
BAND_CHOSEN_WORDS = "five lengths chosen for this cell"
BAND_TYPED_WORDS = ("{} recording length", "{} recording lengths")
#: The audit's counts: batches simulated as training simulates them.
AUDIT_COUNTS_TEXT = "Simulates {} batches of {}, as training would: {} simulations."
#: What the observations can pin down: its simulations for each comparison and for the noise level.
PIN_DOWN_COUNTS_TEXT = "Simulates {} recordings for each comparison and {} for the noise level."
#: The calibration repeated at fixed numbers: its calibrations, each of its test datasets.
FIXED_COUNTS_TEXT = "Runs {} calibrations of {} test datasets each, about {} times the calibration's time."
#: The drive study: its simulations at each strength, at the strengths typed or at the engine's own sixteen, to which it
#: adds the set drive amplitude.
DRIVE_COUNTS_TEXT = "Simulates {} recordings at each of {} strengths."
DRIVE_USUAL_STRENGTHS = 16
#: Identifiability across the prior: its points, at each of which it simulates as the check of what the observations
#: can pin down does at one.
ACROSS_COUNTS_TEXT = ("Simulates at {} points across the prior, at each as the check of what the observations can pin "
                      "down does at one.")
#: What a check that reads records and simulates nothing says of its cost.
READS_ONLY_TEXT = "Reads records and simulates nothing; takes seconds."
READS_ONLY = ("input_use", "identifiability")
#: How long a check took on this computer the last time it finished at the sizes the form holds, from the run history;
#: and what is said when it never has.
TOOK_TEXT = "Took {} here last time at these sizes."
NOT_TIMED_TEXT = "Not timed on this computer yet."
#: The one-run rule, and what a check's run writes.
ONE_AT_A_TIME_TEXT = "Nothing else can run until it ends."
ONE_RECORD_TEXT = "One record, under Records."
#: The checks that simulate on the graphics card: beside them the memory limit in force and the last run are said.
ON_THE_CARD = ("band", "audit", "calibration", "pin_down", "fixed_number", "drive_study", "across_prior")


def _is_whole(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _counts_of(key: str, sizes: Mapping) -> str:
    """The counts of the run of the check ``key`` at the ``sizes`` its form holds, by field key: "" while a size its
    sentence says is not a whole number, or for a check that says none."""
    if key in READS_ONLY:
        return READS_ONLY_TEXT
    if key == "calibration":
        held = [sizes.get(name) for name in ("cal_n_scales", "n_cal", "num_posterior_samples")]
        return CALIBRATION_COUNTS_TEXT.format(*map(number_text, held)) if all(map(_is_whole, held)) else ""
    if key == "band":
        repeats, lengths = sizes.get("band_repeats"), sizes.get("probe_lengths")
        if "probe_lengths" not in sizes or not _is_whole(repeats) or not (
                lengths is None or isinstance(lengths, (list, tuple))):
            return ""
        typed = BAND_TYPED_WORDS[len(lengths) != 1].format(number_text(len(lengths))) if lengths else ""
        return BAND_COUNTS_TEXT.format(number_text(repeats), typed or BAND_CHOSEN_WORDS)
    if key == "audit":
        batches, each = sizes.get("mask_num_runs"), sizes.get("mask_run_size")
        if not (_is_whole(batches) and _is_whole(each)):
            return ""
        return AUDIT_COUNTS_TEXT.format(number_text(batches), number_text(each), number_text(batches * each))
    if key == "pin_down":
        each, noise = sizes.get("m"), sizes.get("m_noise")
        return PIN_DOWN_COUNTS_TEXT.format(number_text(each), number_text(noise)) if (
            _is_whole(each) and _is_whole(noise)) else ""
    if key == "fixed_number":
        repeats, datasets = sizes.get("repeats"), sizes.get("n_cal")
        return FIXED_COUNTS_TEXT.format(number_text(repeats), number_text(datasets), number_text(repeats)) if (
            _is_whole(repeats) and _is_whole(datasets)) else ""
    if key == "drive_study":
        each, strengths = sizes.get("drive_repeats"), sizes.get("drive_strengths")
        if "drive_strengths" not in sizes or not _is_whole(each) or not (
                strengths is None or isinstance(strengths, (list, tuple))):
            return ""
        return DRIVE_COUNTS_TEXT.format(number_text(each), number_text(
            len(strengths) if strengths else DRIVE_USUAL_STRENGTHS + 1))
    if key == "across_prior":
        points = sizes.get("n_points")
        return ACROSS_COUNTS_TEXT.format(number_text(points)) if _is_whole(points) else ""
    return ""


#: The drive study's suggested Forcing sections, as a cell file and a bounds file write them: each drive the record
#: suggests, under a heading the cell file reads as its Forcing section, then the limits for a bounds file.
FORCING_HEAD_TEXT = "# Forcing Parameters: {}"
FORCING_SUGGESTED = (("free_running", "the strongest drive that keeps the cell's own oscillation"),
                     ("captured", "the weakest drive that captures the cell"))
FORCING_BOX_HEAD = "# Forcing Parameters of a bounds file"
#: The order a cell file writes its drive in; a name not here follows, in the record's order.
FORCING_ORDER = ("amp", "freq", "phase", "offset")


def _in_order(given: Mapping) -> list:
    return sorted(given.items(), key=lambda pair: FORCING_ORDER.index(pair[0]) if pair[0] in FORCING_ORDER
                  else len(FORCING_ORDER))


def forcing_lines(manifest: Mapping) -> str:
    """The Forcing sections a drive study's record suggests, as lines a cell file and a bounds file take: each drive
    under its heading, "amp = 0.2", then the bounds file's limits, "amp in (0, 6)"; "" for a record that suggests none.
    The engine writes them to no file."""
    try:
        suggested = _body(manifest)["results"]["suggested_forcing"]
    except (KeyError, TypeError):
        return ""
    if not isinstance(suggested, Mapping):
        return ""
    parts = []
    for key, words in FORCING_SUGGESTED:
        drive = suggested.get(key)
        if isinstance(drive, Mapping) and drive:
            parts.append("\n".join([FORCING_HEAD_TEXT.format(words),
                                     *(f"{name} = {_real(value):.10g}" for name, value in _in_order(drive))]))
    box = suggested.get("box")
    if isinstance(box, Mapping) and box:
        parts.append("\n".join([FORCING_BOX_HEAD, *(f"{name} in ({_real(ends[0]):.10g}, {_real(ends[-1]):.10g})"
                                                     for name, ends in _in_order(box))]))
    return "\n\n".join(parts)


def before_you_start(key: str, *, sizes: Mapping, last: Mapping | None, memory: tuple[str, ...],
                     read: bool = True) -> tuple[Fact, ...]:
    """What "Before you start" says of a run of the check ``key`` at the ``sizes`` its form holds, by field key, in its
    order: what the run simulates or reads, in counts (left out while a size it says is missing, a row in error leaving
    its key out); how long it took the ``last`` time it finished at these sizes on this computer, a run's history (None
    for none: a run whose time its history does not hold is no information), nothing while the history is not ``read``
    yet; the ``memory`` lines, the limit in force and the last run on this computer, for a check that simulates on the
    card; that nothing else can run meanwhile; and what it writes."""
    said = []
    counts = _counts_of(key, sizes)
    if counts:
        said.append(Fact("info", counts))
    held = last.get("check") if isinstance(last, Mapping) else None
    seconds = held.get("seconds") if isinstance(held, Mapping) else None
    timed = isinstance(seconds, (int, float)) and not isinstance(seconds, bool) and seconds >= 0
    if read:
        said.append(Fact("info", TOOK_TEXT.format(duration_text(seconds)) if timed else NOT_TIMED_TEXT))
    lines = " ".join(line for line in memory if line)
    if key in ON_THE_CARD and lines:
        said.append(Fact("info", lines))
    said += [Fact("info", ONE_AT_A_TIME_TEXT), Fact("info", ONE_RECORD_TEXT)]
    return tuple(said)
