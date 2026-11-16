"""Which records count for what is in use, the state each check's row reads, the headline over the six deciding checks
and the summary the chain of steps and Home read, and the line "Before you start" says of the two checks made before a
training: as values.

A record counts only for the record it was made for, matched by id and never by name: a record can be renamed in Records
and keeps its id. Of several, the newest that counts decides, and the row says when an earlier one came out
differently; the frequency band is judged on each cell by that cell's newest, and reads the worst of them. What is in
use reaches here as plain values (``Held``), worked out by the screen: nothing here reads the session, and nothing here
imports the toolkit, torch or the engine's settings. A deciding check that no run can pass for the posterior in use,
since the posterior was made without what it reads, is not passed and no longer holds the checks open; a check that a
run or a load can still pass holds them open. What stops a check's button, and the words beside it, follow from its
row.

The two readings of the disk here are the ones the screen makes, with no run live: every check record's manifest, each
read once by its folder, and one record's text or manifest by its folder. Neither goes through the store's lookup by
id, which reads every manifest of its kind again for each record looked up.
"""
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from . import check_readings, step_facts
from .check_catalogue import CHECKS, DECIDING, entry
from .step_facts import Fact
from .texts import date_text

#: The file a record's manifest is in, under its folder.
MANIFEST = "manifest.json"
#: The fields of a record's configuration block that say it was made on the configuration in use: the model, its
#: parameters and their ranges.
CONFIG_FIELDS = ("model", "param_keys", "nd_lows", "nd_highs", "rescale_lows", "rescale_highs")
#: A record whose moment cannot be read is the oldest.
_EARLIEST = datetime.min.replace(tzinfo=timezone.utc)
#: The states a deciding check reads once a record of it counts, from the best to the worst.
_DECIDED = ("Passed", "Look closer", "Failed")
#: What the second line counts as waiting for step 3, which the Checks list's main button passes over, and as not run
#: yet.
WAITING_FOR_TRAINING = ("Waiting for a posterior", "Waiting for the simulations")
_NOT_RUN = ("Not run yet", "Waiting for the audit")

#: What a row says when the check cannot run or cannot apply, before or beside its state.
NEEDS_PRIOR_TEXT = "Needs a prior: build or load one in step 2."
NEEDS_POSTERIOR_TEXT = "Needs a posterior: train one in step 3, or load a saved one."
OTHER_PRIOR_TEXT = "The prior in use is not the one this posterior was trained on. Load it in step 2."
OTHER_PRIOR_NAMED_TEXT = "The prior in use is not the one this posterior was trained on ({}). Load it in step 2."
NO_SIMULATIONS_TEXT = "This posterior was trained without keeping its simulations, and this check reads them."
NO_ROTATION_TEXT = "This posterior was trained without the rotation, so there are no directions to read."
FREQUENCY_ONLY_TEXT = "Applies only with the frequency response."
NADROWSKI_ONLY_TEXT = "Applies only to the Nadrowski model."
NEEDS_FORCING_TEXT = "Needs a bounds file with a forcing section."
NOT_WITH_FREQUENCY_TEXT = "Does not apply with the frequency response."
UNREAD_COUNTS_TEXT = "Not compared: the counts of unusable driven recordings could not be read."
#: How a row says an earlier record that counts came out differently from the one that decides.
EARLIER_TEXT = "An earlier run {} ({})."
EARLIER_WORDS = {"Passed": "passed", "Failed": "failed", "Look closer": "needed a closer look"}

#: The headline and the sentence under it.
HEADLINE_TEXT = "{} of {} deciding checks passed"
HEADLINE_SOME_TEXT = "{} of {} deciding checks passed; {} do not apply to this posterior"
NOTHING_YET_TEXT = "Nothing can be checked yet. Build or load a prior in step 2."
FAILED_TEXT = "Failed: {}."
CLOSER_TEXT = "Look closer: {}."
THREE_DO_NOT_APPLY_TEXT = "Three of the six do not apply to a posterior trained without the frequency response."
OTHER_FOUR_TEXT = "The other four are made after simulating and after the fit."
ONE_OPEN_TEXT = "Inference can be run now; 1 deciding check is still open."
SEVERAL_OPEN_TEXT = "Inference can be run now; {} deciding checks are still open."
ALL_PASSED_TEXT = "This posterior has passed every deciding check."
ALL_THAT_APPLY_TEXT = "This posterior has passed every deciding check that applies to it."
NO_RUN_PASSES_TEXT = "Inference can be run now; no run can pass {} for this posterior."
OPEN_AND_NO_RUN_TEXT = "{} No run can pass {} for this posterior."
#: A check's name inside a sentence that names it in passing, set off as the window sets off a name in a sentence; two
#: or more are joined by commas, and "or" before the last.
NAMED_IN_PASSING = "'{}'"
#: The reasons a deciding check's row gives that no run can lift for the posterior in use: it was made without what
#: the check reads.
CANNOT_PASS_REASONS = (NO_SIMULATIONS_TEXT, NO_ROTATION_TEXT)
#: The summary's second line.
FAILED_COUNT_TEXT = "{} failed"
CLOSER_COUNT_TEXT = "{} to look closer at"
ONE_WAITS_TEXT = "1 waits for step 3"
SEVERAL_WAIT_TEXT = "{} wait for step 3"
NOT_RUN_COUNT_TEXT = "{} not run yet"
NO_RUN_COUNT_TEXT = "{} that no run can pass"

#: The two checks made before a training, as "Before you start" says them, and the question a start asks after one
#: failed.
PRE_RUN = ("band", "audit")
NEED_THE_FREQUENCY_TEXT = "The two pre-run checks do not apply: they need the frequency response."
ASKS_FIRST_TEXT = "Starting will ask first."
QUESTION_TITLE = "{} although a check failed?"
QUESTION_WHY_TEXT = "Step 4, Checks, says why."
QUESTION_OTHER = "{} anyway"


@dataclass(frozen=True)
class RecordRead:
    """One check record as the reading found it: its ``folder``, and its ``manifest`` as the store wrote it, or an
    empty one for a record whose manifest could not be read, which is passed over and can be said."""
    folder: str
    manifest: Mapping


@dataclass(frozen=True)
class Held:
    """What is in use, as plain values the screen worked out: the ids of the prior and the posterior in use, of the
    prior the posterior was trained on and of its simulation set (None for none); whether the configuration is of the
    frequency response (None with none); the six fields of its configuration block that a record made on it holds
    (``CONFIG_FIELDS``) and the settings of the frequency response a probe check's record states (None without it);
    whether the posterior was trained with the rotation and narrowed in a round, whether the model is Nadrowski and the
    bounds file has a forcing section; the engine's sentence refusing the prior in use with the posterior ("" when it
    passes, or was not read); the set's counts of unusable driven recordings, its unusable, its total, its batches
    counted and the scope the engine names them by (None when not read or not there), whether the set is complete and
    when it was completed; the prior and the posterior as a sentence names them; whether the posterior's set and text
    were read for the posterior now in use (``set_read``: False between a posterior coming into use and the next
    reading); the label of the prior the posterior was trained on, as a reading found it ("" when not read, or
    when that prior's record is gone); and of the training that made the posterior's set, as the run history holds it:
    the comparison its history keeps with the answers given to it and the id of that run (answered; None for none),
    and whether any run of the history names the set (set_in_history)."""
    prior: str | None
    posterior: str | None
    posterior_prior: str | None
    simulation: str | None
    chi: bool | None
    config: Mapping | None
    configured: Mapping | None
    rotated: bool
    narrowed: bool
    nadrowski: bool
    forcing: bool
    pair_refused: str
    set_counts: tuple[int, int, int, str] | None
    set_done: bool
    prior_words: str
    posterior_words: str
    set_when: datetime | None = None
    set_read: bool = True
    posterior_prior_words: str = ""
    answered: "Mapping | None" = None
    set_in_history: bool = False


@dataclass(frozen=True)
class CheckRow:
    """One check's row: its ``key``, the ``state`` it reads (one of ``check_readings.STATES``), its state ``line`` as
    its page says it, ``when`` the record that decides was made (or the set completed, or the run started; None for
    none), the ``record`` that decides (None for none), the ids of its ``earlier`` records, newest first, and the
    ``reason`` the check cannot run or cannot apply now ("" for none)."""
    key: str
    state: str
    line: str
    when: datetime | None
    record: RecordRead | None
    earlier: tuple[str, ...]
    reason: str


@dataclass(frozen=True)
class ChecksSummary:
    """What the chain of steps and Home read of the checks: how many deciding checks ``passed``, ``of`` how many (six,
    or the three that apply to a posterior trained without the frequency response); whether any ``failed`` or reads
    ``look_closer``; whether one that applies is still ``open``, a run able to pass it; whether anything ``can_check``
    yet (a prior is in use); the ``second_line``, the first of what failed, what to look closer at, what waits for step
    3, what is not run yet and what no run can pass ("" for none); the ``headline`` and the ``sentence`` under it; and
    ``cannot_pass``, how many deciding checks no run can pass for the posterior in use, none of them passed and none
    of them open."""
    passed: int
    of: int
    failed: bool
    look_closer: bool
    open: bool
    can_check: bool
    second_line: str
    headline: str
    sentence: str
    cannot_pass: int = 0


# ── the two readings ──
def read_records(store, kinds: Sequence[str] = ("diagnostic", "calibration")) -> tuple[RecordRead, ...]:
    """Every record of ``kinds`` in ``store``, each manifest read once from the folder the store's listing hands back. A
    record whose manifest cannot be read, or whose folder holds none, is kept with an empty manifest, so that it can be
    said and is never counted; a kind the store cannot list gives nothing. Never raises on what the disk holds."""
    found = []
    for kind in kinds:
        try:
            listed = store.list(kind)
        except Exception:                        # noqa: BLE001 -- a store that cannot be listed holds no record
            continue
        for summary in listed:
            folder, manifest = Path(summary.path), {}
            try:
                held = json.loads((folder / MANIFEST).read_text(encoding="utf-8"))
                manifest = held if isinstance(held, dict) else {}
            except (OSError, ValueError):
                pass
            found.append(RecordRead(str(folder), manifest))
    return tuple(found)


def read_text(folder: str, name: str = "log.txt") -> str | None:
    """The text of the file ``name`` in a record's ``folder``, read by the folder; None when it is not there or cannot
    be read."""
    try:
        return (Path(folder) / name).read_text(encoding="utf-8", errors="replace")
    except (OSError, TypeError, ValueError):
        return None


# ── which records count ──
def _ident(record: RecordRead) -> str:
    ident = record.manifest.get("id")
    return ident if isinstance(ident, str) else Path(record.folder).name


def _moment(record: RecordRead) -> datetime | None:
    """When the record was made, as its manifest says, with its zone; None when it does not say."""
    try:
        moment = datetime.fromisoformat(record.manifest.get("created"))
    except (TypeError, ValueError):
        return None
    return moment if moment.tzinfo is not None else None


def _newest_first(record: RecordRead) -> tuple:
    """The order of records, newest first once reversed: by the moment each was made, then, for two made in the same
    moment, by its id, whose number after the moment the store raises for a record made in a moment already taken."""
    moment = _moment(record)
    stem, _, number = _ident(record).partition("-")
    return moment is not None, moment or _EARLIEST, stem, int(number) if number.isdigit() else 1


def _local(record: RecordRead | None) -> datetime | None:
    moment = None if record is None else _moment(record)
    return None if moment is None else moment.astimezone()


def _body(record: RecordRead) -> Mapping:
    body = record.manifest.get("body")
    return body if isinstance(body, Mapping) else {}


def _settings(record: RecordRead) -> Mapping:
    settings = _body(record).get("settings")
    return settings if isinstance(settings, Mapping) else {}


def _parent(record: RecordRead, kind: str):
    parents = record.manifest.get("parents")
    return parents.get(kind) if isinstance(parents, Mapping) else None


def _of_check(one, records: Sequence[RecordRead]) -> list[RecordRead]:
    """The records of the check ``one``: of its kind, and for a check record of its check and its form."""
    found = []
    for record in records:
        manifest = record.manifest
        if not manifest or manifest.get("kind") != one.kind:
            continue
        body = _body(record)
        if one.diagnostic is not None and body.get("diagnostic") != one.diagnostic:
            continue
        if one.variant is not None and body.get("variant") != one.variant:
            continue
        found.append(record)
    return found


def _same_config(record: RecordRead, config: Mapping | None) -> bool:
    """Whether the record was made on the configuration in use: its model, parameters and ranges."""
    block = record.manifest.get("config")
    if not isinstance(block, Mapping) or not isinstance(config, Mapping):
        return False
    return all(config.get(name) is not None and block.get(name) == config.get(name) for name in CONFIG_FIELDS)


def _same_configured(record: RecordRead, configured: Mapping | None, *, but_the_count: bool = False) -> bool:
    """Whether the record's settings of the frequency response are those in use; for an audit, which draws the count of
    driven recordings as training does, all but that count."""
    stated = _settings(record).get("configured")
    if not isinstance(stated, Mapping) or not isinstance(configured, Mapping):
        return False
    if but_the_count:
        return ({name: value for name, value in stated.items() if name != "probe_count"}
                == {name: value for name, value in configured.items() if name != "probe_count"})
    return dict(stated) == dict(configured)


def _date(moment: datetime | None) -> str:
    return "" if moment is None else date_text(moment)


def _line(state: str, when: datetime | None, then: str = "") -> str:
    said = f"{state}, {_date(when)}." if when is not None else f"{state}."
    return f"{said} {then}" if then else said


def _row(key: str, state: str, then: str = "", *, reason: str = "") -> CheckRow:
    """A row no record decides."""
    return CheckRow(key, state, _line(state, None, then), None, None, (), reason)


def live_row(key: str, state: str) -> CheckRow:
    """The row of the check ``key`` as a live run's own reading decides it, no record among it: its ``state`` and its
    state line alone."""
    return _row(key, state)


def _decided(one, belonging: Sequence[RecordRead], *, reason: str = "") -> CheckRow:
    """The row of the check ``one`` from its own records: the newest that counts decides, and the newest earlier one
    that counts and came out otherwise is said; with none that counts, "Not run yet", with what the newest that does not
    count says, or ``reason``."""
    newest = sorted(belonging, key=_newest_first, reverse=True)
    read = [(record, check_readings.read(one.key, record.manifest)) for record in newest]
    counting = [(record, reading) for record, reading in read if reading.counts]
    if not counting:
        said = next((reading.words for _record, reading in read if reading.words), "")
        return CheckRow(one.key, "Not run yet", _line("Not run yet", None, said or reason), None, None,
                        tuple(_ident(record) for record in newest), reason)
    record, reading = counting[0]
    when, then = _local(record), ""
    for other, earlier in counting[1:]:
        if earlier.state != reading.state:
            then = EARLIER_TEXT.format(EARLIER_WORDS.get(earlier.state, earlier.state.lower()), _date(_local(other)))
            break
    return CheckRow(one.key, reading.state, _line(reading.state, when, then), when, record,
                    tuple(_ident(other) for other in newest if other is not record), reason)


def _cell(record: RecordRead):
    inputs = record.manifest.get("inputs")
    cell = inputs.get("cell") if isinstance(inputs, Mapping) else None
    return cell.get("path") if isinstance(cell, Mapping) else None


def _band(one, held: Held, records, _histories, _made) -> CheckRow:
    if held.chi is False:
        return _row(one.key, "Does not apply", FREQUENCY_ONLY_TEXT, reason=FREQUENCY_ONLY_TEXT)
    belonging = [record for record in _of_check(one, records)
                 if _same_config(record, held.config) and _same_configured(record, held.configured)]
    cells: dict = {}
    for record in belonging:
        cells.setdefault(_cell(record), []).append(record)
    decided = [_decided(one, group) for group in cells.values()]
    judged = [row for row in decided if row.state in _DECIDED]
    if not judged:
        return _decided(one, belonging)
    worst = max(judged, key=lambda row: (_DECIDED.index(row.state), row.when or _EARLIEST))
    others = tuple(_ident(record) for record in sorted(belonging, key=_newest_first, reverse=True)
                   if record is not worst.record)
    return CheckRow(worst.key, worst.state, worst.line, worst.when, worst.record, others, worst.reason)


def _audit(one, held: Held, records, _histories, _made) -> CheckRow:
    if held.chi is False:
        return _row(one.key, "Does not apply", FREQUENCY_ONLY_TEXT, reason=FREQUENCY_ONLY_TEXT)
    prior = held.posterior_prior if held.posterior is not None and held.posterior_prior else held.prior
    # With a posterior in use the row counts the audits of its own prior, and a press audits the prior in use: while the
    # engine refuses the two as a pair, a press would write a record that counts for nothing, so the audit is greyed as
    # the two checks of that pair are.
    return _decided(one, [record for record in _of_check(one, records) if _parent(record, "prior") == prior
                          and _same_configured(record, held.configured, but_the_count=True)],
                    reason=_other_prior(held) if held.posterior is not None else "")


def _newest_training(held: Held, histories: Sequence[Mapping]) -> Mapping | None:
    """The newest training of the prior in use in the run history handed in, live or ended."""
    found = None
    for record in histories:
        if not isinstance(record, Mapping):
            continue
        settings = record.get("settings")
        prior = settings.get("prior") if isinstance(settings, Mapping) else None
        if record.get("word") == "Training" and prior in (None, held.prior):
            found = record
    return found


def audit_fraction(found: Sequence[CheckRow]):
    """The share of unusable driven recordings the audit that decides its row read, from the rows; None when no
    audit decides it."""
    audit = next((row for row in found if row.key == "audit"), None)
    if audit is None or audit.record is None or audit.state not in _DECIDED:
        return None
    results = _body(audit.record).get("results")
    return results.get("masked_fraction") if isinstance(results, Mapping) else None


def counted_audit(found: Sequence[CheckRow]) -> "dict | None":
    """The audit that decides the audit's row among ``found``, as a run is handed it to compare its simulation set
    with, names and numbers: its record's id, the share it read and the batches it was made at; None when no audit
    decides it."""
    audit = next((row for row in found if row.key == "audit"), None)
    fraction = audit_fraction(found)
    if audit is None or fraction is None:
        return None
    batches = _settings(audit.record).get("num_runs")
    return {"id": _ident(audit.record), "fraction": fraction,
            "batches": batches if isinstance(batches, int) and not isinstance(batches, bool) else None}


def comparison_of(held: Held, found: Sequence[CheckRow]) -> "check_readings.Comparison | None":
    """The comparison of the posterior's simulation set with the audit, from the set's counts the reading found and the
    audit that decides among ``found``, the rows made before it; None with no posterior in use or no counts."""
    if held.posterior is None or held.set_counts is None:
        return None
    unusable, total, batches, scope = held.set_counts
    return check_readings.compare(unusable=unusable, total=total, scope=scope, audit_fraction=audit_fraction(found),
                                  accepted=acceptance(held, found), batches=batches,
                                  prior_words=held.prior_words or "the prior in use",
                                  kept_set=held.simulation is not None)


def acceptance(held: Held, found: Sequence[CheckRow]) -> "Mapping | None":
    """The acceptance of a difference the history of the training that made the posterior's set keeps, while it is of
    the audit that counts now among found; None for none, and once another audit counts."""
    accepted = held.answered.get("accepted") if isinstance(held.answered, Mapping) else None
    audit = counted_audit(found)
    if not isinstance(accepted, Mapping) or audit is None or accepted.get("audit") != audit["id"]:
        return None
    return accepted


def _comparison(one, held: Held, _records, histories, made) -> CheckRow:
    if held.chi is False:
        return _row(one.key, "Does not apply", FREQUENCY_ONLY_TEXT, reason=FREQUENCY_ONLY_TEXT)
    if held.posterior is None:
        health = (_newest_training(held, histories) or {}).get("health")
        compared = health.get("comparison") if isinstance(health, Mapping) else None
        state = compared.get("state") if isinstance(compared, Mapping) else None
        if state in check_readings.STATES:
            return _row(one.key, state)
        return _row(one.key, "Waiting for the simulations")
    compared = comparison_of(held, made)
    if compared is None:
        return CheckRow(one.key, "Not compared", UNREAD_COUNTS_TEXT, None, None, (), "")
    if compared.state == "Not compared":
        return CheckRow(one.key, compared.state, compared.sentence, None, None, (), "")
    when = held.set_when if compared.state in _DECIDED else None
    then = check_readings.accepted_words(compared, acceptance(held, made))
    return CheckRow(one.key, compared.state, _line(compared.state, when, then), when, None, (), "")


def _of_posterior(one, held: Held, records, *, reason: str = "") -> CheckRow:
    if held.posterior is None:
        return _row(one.key, "Waiting for a posterior", reason=NEEDS_POSTERIOR_TEXT)
    own = [record for record in _of_check(one, records) if _parent(record, "posterior") == held.posterior]
    return _decided(one, own, reason=reason)


def _other_prior(held: Held) -> str:
    """Why the prior in use cannot be checked with the posterior, naming the prior it was trained on when the reading
    found it; "" when the engine takes the pair."""
    if not held.pair_refused:
        return ""
    return (OTHER_PRIOR_NAMED_TEXT.format(held.posterior_prior_words) if held.posterior_prior_words
            else OTHER_PRIOR_TEXT)


def _calibration(one, held: Held, records, _histories, _made) -> CheckRow:
    return _of_posterior(one, held, records, reason=_other_prior(held))


def _input_use(one, held: Held, records, _histories, _made) -> CheckRow:
    return _of_posterior(one, held, records, reason="" if held.simulation is not None else NO_SIMULATIONS_TEXT)


def _identifiability(one, held: Held, records, _histories, _made) -> CheckRow:
    return _of_posterior(one, held, records, reason="" if held.rotated else NO_ROTATION_TEXT)


def _of_configuration(one, held: Held, records) -> CheckRow:
    return _decided(one, [record for record in _of_check(one, records) if _same_config(record, held.config)])


def _pin_down(one, held: Held, records, _histories, _made) -> CheckRow:
    if not held.nadrowski:
        return _row(one.key, "Does not apply", NADROWSKI_ONLY_TEXT, reason=NADROWSKI_ONLY_TEXT)
    if not held.chi and not held.forcing:
        return _row(one.key, "Does not apply", NEEDS_FORCING_TEXT, reason=NEEDS_FORCING_TEXT)
    return _of_configuration(one, held, records)


def _of_history(one, _held: Held, _records, histories, _made) -> CheckRow:
    """A check that keeps nothing but its result: "Done" from the newest run of it in the history that finished."""
    for record in reversed(tuple(histories)):
        held = record.get("check") if isinstance(record, Mapping) else None
        if isinstance(held, Mapping) and held.get("key") == one.key and record.get("outcome") == "finished":
            try:
                when = datetime.fromisoformat(record.get("started")).astimezone()
            except (TypeError, ValueError):
                when = None
            return CheckRow(one.key, "Done", _line("Done", when), when, None, (), "")
    return _row(one.key, "Not run yet")


def _fixed_number(one, held: Held, records, _histories, _made) -> CheckRow:
    return _of_posterior(one, held, records, reason=_other_prior(held))


def _drive_study(one, held: Held, records, _histories, _made) -> CheckRow:
    if not held.forcing:
        return _row(one.key, "Does not apply", NEEDS_FORCING_TEXT, reason=NEEDS_FORCING_TEXT)
    return _of_configuration(one, held, records)


def _across_prior(one, held: Held, records, _histories, _made) -> CheckRow:
    if held.chi:
        return _row(one.key, "Does not apply", NOT_WITH_FREQUENCY_TEXT, reason=NOT_WITH_FREQUENCY_TEXT)
    if not held.forcing:
        return _row(one.key, "Does not apply", NEEDS_FORCING_TEXT, reason=NEEDS_FORCING_TEXT)
    return _of_posterior(one, held, records)


#: How each check's row is made.
_ROWS = {"band": _band, "audit": _audit, "comparison": _comparison, "calibration": _calibration,
         "input_use": _input_use, "identifiability": _identifiability, "pin_down": _pin_down, "timing": _of_history,
         "fixed_number": _fixed_number, "card_test": _of_history, "drive_study": _drive_study,
         "across_prior": _across_prior}


def rows(held: Held, records: Sequence[RecordRead], histories: Sequence[Mapping]) -> tuple[CheckRow, ...]:
    """Every check's row, in the list's order, for what is ``held``, from the check ``records`` and the run
    ``histories`` of the records folder in use, oldest first. With no prior in use every row waits for one."""
    if held.prior is None:
        return tuple(_row(one.key, "Waiting for a prior", reason=NEEDS_PRIOR_TEXT) for one in CHECKS)
    made: list = []
    for one in CHECKS:
        made.append(_ROWS[one.key](one, held, records, histories, made))
    return tuple(made)


# ── the headline and the summary ──
def _deciding(found) -> list:
    return [row for row in found if row.key in DECIDING]


def no_run_passes(row: CheckRow, *, held: Held) -> bool:
    """Whether no run can pass the deciding check of ``row`` for the posterior in use, so that the row no longer holds
    the step open though it does not count as passed: the use of each part of the input for a posterior trained without
    keeping its simulations, the identifiability of one trained without the rotation, each with no record that counts,
    and the comparison after simulating not compared for what the posterior's simulation set is (nothing counted, or a
    part of it), as a reading of it found it. A row that passed or does not apply, a check that does not decide, every
    row while no posterior is in use, and the comparison whose counts could not be read, are none of these: a load, a
    run or a reading lifts what they wait for."""
    if held.posterior is None or row.key not in DECIDING or row.state in ("Passed", "Does not apply"):
        return False
    if row.key == "comparison":
        return row.state == "Not compared" and held.set_read and held.set_counts is not None
    return row.record is None and row.reason in CANNOT_PASS_REASONS


def _still_open(deciding, held: Held) -> list:
    """The deciding rows that apply and have not passed, a run or a load able to pass each."""
    return [row for row in deciding if row.state not in ("Passed", "Does not apply")
            and not no_run_passes(row, held=held)]


def _counts(found, held: Held) -> tuple[int, int, int]:
    """How many deciding checks passed, of how many, and how many do not apply to a posterior trained without the
    frequency response (none counted otherwise: with no such posterior the headline is of all six)."""
    deciding = _deciding(found)
    passed = sum(row.state == "Passed" for row in deciding)
    trained_without = held.posterior is not None and held.chi is False
    away = sum(row.state == "Does not apply" for row in deciding) if trained_without else 0
    return passed, len(DECIDING) - away, away


def _in_passing(names: Sequence[str]) -> str:
    """Checks' ``names`` as a sentence names them in passing: each set off, the last joined by "or"; "" for none."""
    quoted = [NAMED_IN_PASSING.format(name) for name in names]
    return " or ".join([", ".join(quoted[:-1]), quoted[-1]]) if len(quoted) > 1 else "".join(quoted)


def headline(found: Sequence[CheckRow], *, held: Held) -> tuple[str, str]:
    """The headline over the deciding checks and the sentence under it, the first that applies."""
    passed, of, away = _counts(found, held)
    said = HEADLINE_SOME_TEXT.format(passed, of, away) if away else HEADLINE_TEXT.format(passed, of)
    deciding = _deciding(found)

    def names(state: str) -> str:
        return ", ".join(entry(row.key).name for row in deciding if row.state == state)

    if held.prior is None:
        return said, NOTHING_YET_TEXT
    if names("Failed"):
        return said, FAILED_TEXT.format(names("Failed"))
    if names("Look closer"):
        return said, CLOSER_TEXT.format(names("Look closer"))
    if held.posterior is None:
        none_apply = sum(row.state == "Does not apply" for row in deciding)
        return said, THREE_DO_NOT_APPLY_TEXT if none_apply == 3 else OTHER_FOUR_TEXT
    still = len(_still_open(deciding, held))
    never = _in_passing([entry(row.key).name for row in deciding if no_run_passes(row, held=held)])
    opened = (ONE_OPEN_TEXT if still == 1 else SEVERAL_OPEN_TEXT.format(still)) if still else ""
    if opened and never:
        return said, OPEN_AND_NO_RUN_TEXT.format(opened, never)
    if opened:
        return said, opened
    if never:
        return said, NO_RUN_PASSES_TEXT.format(never)
    return said, ALL_THAT_APPLY_TEXT if away else ALL_PASSED_TEXT


def _second_line(deciding, held: Held) -> str:
    never = [row for row in deciding if no_run_passes(row, held=held)]
    for states, one, several in ((("Failed",), FAILED_COUNT_TEXT, FAILED_COUNT_TEXT),
                                 (("Look closer",), CLOSER_COUNT_TEXT, CLOSER_COUNT_TEXT),
                                 (WAITING_FOR_TRAINING, ONE_WAITS_TEXT, SEVERAL_WAIT_TEXT),
                                 (_NOT_RUN, NOT_RUN_COUNT_TEXT, NOT_RUN_COUNT_TEXT)):
        count = sum(row.state in states and row not in never for row in deciding)
        if count:
            return (one if count == 1 else several).format(count)
    return NO_RUN_COUNT_TEXT.format(len(never)) if never else ""


def summary(found: Sequence[CheckRow], *, held: Held) -> ChecksSummary:
    """What the chain of steps and Home read of the checks, from their rows. A deciding check no run can pass for the
    posterior in use is not passed, and no longer holds the checks open."""
    passed, of, _away = _counts(found, held)
    deciding = _deciding(found)
    said, sentence = headline(found, held=held)
    return ChecksSummary(passed, of, any(row.state == "Failed" for row in deciding),
                         any(row.state == "Look closer" for row in deciding), bool(_still_open(deciding, held)),
                         held.prior is not None, _second_line(deciding, held), said, sentence,
                         sum(no_run_passes(row, held=held) for row in deciding))


# ── before a training ──
def _said(row: CheckRow) -> str:
    name = entry(row.key).name
    return f"{name}: {row.state}, {_date(row.when)}." if row.when is not None else f"{name}: {row.state}."


def pre_run_fact(found: Sequence[CheckRow]) -> Fact:
    """What "Before you start" says of the frequency band and the audit for the prior in use: "Fine." when both
    passed, "Information." when neither applies, otherwise "Attention." naming each that has not passed, and saying
    that starting will ask first after one failed. Never blocks."""
    by_key = {row.key: row for row in found}
    two = [by_key[key] for key in PRE_RUN if key in by_key]
    if two and all(row.state == "Passed" for row in two):
        return Fact("fine", " ".join(_said(row) for row in two))
    if two and all(row.state == "Does not apply" for row in two):
        return Fact("info", NEED_THE_FREQUENCY_TEXT)
    said = [_said(row) for row in two if row.state != "Passed"]
    if any(row.state == "Failed" for row in two):
        said.append(ASKS_FIRST_TEXT)
    return Fact("warn", " ".join(said))


def pre_run_question(found: Sequence[CheckRow], *, start: str = "Start training") -> tuple[str, str, str] | None:
    """The question a start asks first when the frequency band or the audit reads "Failed", with ``start`` the words of
    the button pressed: its title, its text and its other answer; None when neither failed."""
    failed = [row for row in found if row.key in PRE_RUN and row.state == "Failed"]
    if not failed:
        return None
    return (QUESTION_TITLE.format(start), " ".join([*(_said(row) for row in failed), QUESTION_WHY_TEXT]),
            QUESTION_OTHER.format(start))


# ── a check's earlier runs ──
#: How a check's page says each earlier run: what it read and when, and for one that does not count why, by what its
#: reading says.
EARLIER_LINE_TEXT = "{}, {}"
EARLIER_UNCOUNTED_TEXT = "{}, {}; does not count"
EARLIER_WHY = ((check_readings.REDUCED_TEXT.split(" (", 1)[0], "Run at a reduced size"),
               (check_readings.OTHER_THRESHOLDS_TEXT, "Run with other thresholds"),
               (check_readings.FIXED_AUDIT_TEXT, "Run at a fixed number of driven recordings"),
               (check_readings.UNREADABLE_TEXT, "Its record cannot be read"))
EARLIER_CELL_TEXT = "{}; cell {}"
EARLIER_VARIANT_TEXT = "{}; {}"
EARLIER_FIXED_WORDS = "with {} driven recordings"
EARLIER_MIXED_WORDS = "mixed, as in training"


def earlier_lines(row: CheckRow, records: Sequence[RecordRead]) -> tuple[str, ...]:
    """The earlier runs of the check of ``row``, one line each, newest first: what each read and when, why one that
    does not count does not, the cell of a frequency band, and how many driven recordings a calibration repeated at
    fixed numbers held. A record no longer among ``records`` is passed over; each is looked for among the check's own
    records, since two kinds of record may hold the same id."""
    by_id = {_ident(record): record for record in _of_check(entry(row.key), records)}
    said = []
    for ident in row.earlier:
        record = by_id.get(ident)
        if record is None:
            continue
        reading = check_readings.read(row.key, record.manifest)
        when = _date(_local(record))
        why = next((words for start, words in EARLIER_WHY if reading.words.startswith(start)), "")
        if why:
            line = EARLIER_UNCOUNTED_TEXT.format(why, when) if when else f"{why}; does not count"
        else:
            line = EARLIER_LINE_TEXT.format(reading.state, when) if when else reading.state
        if row.key == "band" and _cell(record):
            line = EARLIER_CELL_TEXT.format(line, Path(str(_cell(record)).replace("\\", "/")).stem)
        if row.key == "fixed_number":
            fixed = _settings(record).get("chi_k_fixed")
            line = EARLIER_VARIANT_TEXT.format(line, EARLIER_MIXED_WORDS if fixed is None
                                               else EARLIER_FIXED_WORDS.format(fixed))
        said.append(line)
    return tuple(said)


# ── a check's button ──
#: What stops a check's button for each reason its row gives, as a blocking fact: its sentence, and the words beside
#: the button. A prior or a posterior that a load brings lifts the first three; the others are what the posterior,
#: the model, the kind of observation or the bounds file is, which waiting will not change.
_BUTTON_FACTS = {NEEDS_PRIOR_TEXT: (step_facts.NEEDS_A_PRIOR, step_facts.NEEDS_A_PRIOR_BLOCKS),
                 NEEDS_POSTERIOR_TEXT: (NEEDS_POSTERIOR_TEXT, step_facts.NEEDS_A_POSTERIOR_BLOCKS),
                 NO_SIMULATIONS_TEXT: (NO_SIMULATIONS_TEXT, step_facts.NO_SIMULATIONS_KEPT_BLOCKS),
                 NO_ROTATION_TEXT: (NO_ROTATION_TEXT, step_facts.NO_ROTATION_BLOCKS),
                 FREQUENCY_ONLY_TEXT: (FREQUENCY_ONLY_TEXT, step_facts.FREQUENCY_ONLY_BLOCKS),
                 NADROWSKI_ONLY_TEXT: (NADROWSKI_ONLY_TEXT, step_facts.NADROWSKI_ONLY_BLOCKS),
                 NEEDS_FORCING_TEXT: (NEEDS_FORCING_TEXT, step_facts.NO_FORCING_BLOCKS),
                 NOT_WITH_FREQUENCY_TEXT: (NOT_WITH_FREQUENCY_TEXT, step_facts.NOT_WITH_FREQUENCY_BLOCKS)}


def button_facts(row: CheckRow) -> tuple[Fact, ...]:
    """What stops the button of the check of ``row``, as blocking facts: the reason its row gives. A prior in use that
    is not the posterior's own is said with that prior's name when the reading found it. None for a check its row
    leaves free to run."""
    facts = []
    if row.reason.startswith(OTHER_PRIOR_TEXT.split(".", 1)[0]):
        facts.append(Fact("lock", row.reason, step_facts.OTHER_PRIOR_BLOCKS))
    elif row.reason in _BUTTON_FACTS:
        facts.append(Fact("lock", *_BUTTON_FACTS[row.reason]))
    return tuple(facts)
