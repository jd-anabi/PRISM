"""Which records count for what is in use, the state each check's row reads, the headline over the six deciding checks
and the summary the chain of steps and Home read, and the line "Before you start" says of the two checks made before a
training: as values.

A record counts only for the record it was made for, matched by id and never by name: a record can be renamed in Records
and keeps its id. Of several, the newest that counts decides, and the row says when an earlier one came out
differently; the frequency band is judged on each cell by that cell's newest, and reads the worst of them. What is in
use reaches here as plain values (``Held``), worked out by the screen: nothing here reads the session, and nothing here
imports the toolkit, torch or the engine's settings.

The two readings of the disk here are the ones the screen makes, with no run live: every check record's manifest, each
read once by its folder, and one record's text or manifest by its folder. Neither goes through the store's lookup by
id, which reads every manifest of its kind again for each record looked up.
"""
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from . import check_readings
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
#: What the second line counts as waiting for step 3, and as not run yet.
_WAITING_FOR_TRAINING = ("Waiting for a posterior", "Waiting for the simulations")
_NOT_RUN = ("Not run yet", "Waiting for the audit")

#: What a row says when the check cannot run or cannot apply, before or beside its state.
NEEDS_PRIOR_TEXT = "Needs a prior: build or load one in step 2."
NEEDS_POSTERIOR_TEXT = "Needs a posterior: train one in step 3, or load a saved one."
OTHER_PRIOR_TEXT = "The prior in use is not the one this posterior was trained on. Load it in step 2."
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
#: The summary's second line.
FAILED_COUNT_TEXT = "{} failed"
CLOSER_COUNT_TEXT = "{} to look closer at"
ONE_WAITS_TEXT = "1 waits for step 3"
SEVERAL_WAIT_TEXT = "{} wait for step 3"
NOT_RUN_COUNT_TEXT = "{} not run yet"

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
    when it was completed; and the prior and the posterior as a sentence names them."""
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
    ``look_closer``; whether one that applies is still ``open``; whether anything ``can_check`` yet (a prior is in use);
    the ``second_line``, the first of what failed, what to look closer at, what waits for step 3 and what is not run yet
    ("" for none); and the ``headline`` and the ``sentence`` under it."""
    passed: int
    of: int
    failed: bool
    look_closer: bool
    open: bool
    can_check: bool
    second_line: str
    headline: str
    sentence: str


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
    return _decided(one, [record for record in _of_check(one, records) if _parent(record, "prior") == prior
                          and _same_configured(record, held.configured, but_the_count=True)])


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
    if held.set_counts is None:
        return CheckRow(one.key, "Not compared", UNREAD_COUNTS_TEXT, None, None, (), "")
    audit = made[DECIDING.index("audit")]
    fraction = None
    if audit.record is not None and audit.state in _DECIDED:
        results = _body(audit.record).get("results")
        fraction = results.get("masked_fraction") if isinstance(results, Mapping) else None
    unusable, total, batches, scope = held.set_counts
    compared = check_readings.compare(unusable=unusable, total=total, scope=scope, audit_fraction=fraction,
                                      batches=batches, prior_words=held.prior_words or "the prior in use",
                                      kept_set=held.simulation is not None)
    if compared.state == "Not compared":
        return CheckRow(one.key, compared.state, compared.sentence, None, None, (), "")
    when = held.set_when if compared.state in _DECIDED else None
    return CheckRow(one.key, compared.state, _line(compared.state, when), when, None, (), "")


def _of_posterior(one, held: Held, records, *, reason: str = "") -> CheckRow:
    if held.posterior is None:
        return _row(one.key, "Waiting for a posterior", reason=NEEDS_POSTERIOR_TEXT)
    own = [record for record in _of_check(one, records) if _parent(record, "posterior") == held.posterior]
    return _decided(one, own, reason=reason)


def _calibration(one, held: Held, records, _histories, _made) -> CheckRow:
    return _of_posterior(one, held, records, reason=OTHER_PRIOR_TEXT if held.pair_refused else "")


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
    return _of_posterior(one, held, records, reason=OTHER_PRIOR_TEXT if held.pair_refused else "")


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


def _counts(found, held: Held) -> tuple[int, int, int]:
    """How many deciding checks passed, of how many, and how many do not apply to a posterior trained without the
    frequency response (none counted otherwise: with no such posterior the headline is of all six)."""
    deciding = _deciding(found)
    passed = sum(row.state == "Passed" for row in deciding)
    trained_without = held.posterior is not None and held.chi is False
    away = sum(row.state == "Does not apply" for row in deciding) if trained_without else 0
    return passed, len(DECIDING) - away, away


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
    still = sum(row.state not in ("Passed", "Does not apply") for row in deciding)
    if still:
        return said, ONE_OPEN_TEXT if still == 1 else SEVERAL_OPEN_TEXT.format(still)
    return said, ALL_THAT_APPLY_TEXT if away else ALL_PASSED_TEXT


def _second_line(deciding) -> str:
    for states, one, several in ((("Failed",), FAILED_COUNT_TEXT, FAILED_COUNT_TEXT),
                                 (("Look closer",), CLOSER_COUNT_TEXT, CLOSER_COUNT_TEXT),
                                 (_WAITING_FOR_TRAINING, ONE_WAITS_TEXT, SEVERAL_WAIT_TEXT),
                                 (_NOT_RUN, NOT_RUN_COUNT_TEXT, NOT_RUN_COUNT_TEXT)):
        count = sum(row.state in states for row in deciding)
        if count:
            return (one if count == 1 else several).format(count)
    return ""


def summary(found: Sequence[CheckRow], *, held: Held) -> ChecksSummary:
    """What the chain of steps and Home read of the checks, from their rows."""
    passed, of, _away = _counts(found, held)
    deciding = _deciding(found)
    said, sentence = headline(found, held=held)
    return ChecksSummary(passed, of, any(row.state == "Failed" for row in deciding),
                         any(row.state == "Look closer" for row in deciding),
                         any(row.state not in ("Passed", "Does not apply") for row in deciding),
                         held.prior is not None, _second_line(deciding), said, sentence)


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
