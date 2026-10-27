"""How a run ended, in words, and the foot of the rail in each state of a run.

Once a run has ended, its page, the foot of the rail and Home say how and when it ended, and what it kept, until
someone has seen it: ``words`` reads both from the run's history (``core/gui/run/history.py``) and the moment now,
never from the run's text, so that a run of an earlier launch is said as one of this launch is. ``status_of`` is the
status such a history file gives, which the foot and Home read after PRISM is opened again, and ``foot`` the foot's
marker, title and words for any status, read from the status alone. No sentence says "finished" after a stop or a
failure.

Words only: no toolkit, no engine and none of the engine's settings are imported here, so the words are read and
tested with no window.
"""
from datetime import datetime

from ..texts import date_text, duration_text, number_text
from .model import OUTCOMES, RunStatus
from .questions import TRAINING_WORDS, what_a_stop_keeps
from .retry import NO_MESSAGE, OUT_OF_MEMORY, WAITING_LINE, WAITING_TITLE

#: Said after the sentence of a run that finished with a stop asked of it too late to stop anything; the run's text
#: says it too, as the run ends.
TOO_LATE_TEXT = "A stop was asked for, but the run had already finished."
#: What the foot says while nothing runs and no end waits to be seen.
IDLE_TITLE = "Nothing is running"
IDLE_WORDS = "Only one run at a time."
#: The foot's title while a run only loads a record.
LOADING_TITLE = "Loading…"
#: What the foot's words end with while a run is live.
LOCKED_TEXT = "Settings are locked."
#: What a stopped run kept when PRISM closed before the run's end was written, and what a run that was not seen to
#: end kept.
SAVED_BY_THEN_TEXT = "What it had saved by then is kept."
#: How a run ended, as Home's "Last runs" marks it and says it in a word, by the outcome its history holds; a run
#: still running is said so.
HOW = {"finished": ("done", "Finished"), "stopped": ("half", "Stopped"), "failed": ("err", "Failed"),
       "not seen to end": ("err", "Not seen to end"), "running": ("now", "Running")}

# The word for the record a run wrote, by the store's kind, as the end's sentence names it.
_RECORDS = {"prior": "prior", "posterior": "posterior", "observation": "observation", "calibration": "calibration",
            "inference": "inference", "diagnostic": "check", "fdt": "record", "simulation": "simulation set"}
# Where a training was when it was stopped, failed or was last heard, by the name of its stage.
_DURING = {"Rotation": "during the rotation", "Simulation": "during simulation", "Fitting": "during the fit",
           "Saving": "while its posterior was being saved"}
_FAILED_AT = {"Rotation": "before its simulation", "Simulation": "during simulation", "Fitting": "during the fit",
              "Saving": "while its posterior was being saved"}


def words(record: dict, now: datetime, *, too_late: bool = False) -> tuple[str, str]:
    """How the run of the history ``record`` ended, as its page says it once it has ended, and what it kept, as the
    foot says it after the date ("" when the foot says the date alone). ``now`` is the moment the date is written
    against: within seven days a date carries its weekday. ``too_late``: a stop was asked of a run that had already
    finished, which its sentence then says.

    Finished: when, after how long, the record it saved (and that it was saved without the name asked for, when
    another record took that name while the run was going), and how the fit ended, when it ended by itself.
    Stopped: when, and where the run was, or that Windows shut down, then what the stop kept. Failed: when, where,
    the error's message, then what was kept. Not seen to end: when the run was last heard, and where it was."""
    word = str(record.get("word") or "Run")
    outcome = record.get("outcome")
    ended, started = _moment(record.get("ended")), _moment(record.get("started"))
    date = date_text(ended, now) if ended is not None else "a moment not known"
    if outcome == "finished":
        took = (f", after {duration_text((ended - started).total_seconds())}"
                if ended is not None and started is not None else "")
        said = [f"{word} finished on {date}{took}."] + _saved(record) + _fit_ended(record)
        if too_late:
            said.append(TOO_LATE_TEXT)
        return " ".join(said), ""
    if outcome == "stopped":
        kept = _stop_kept(record)
        if record.get("stop_reason"):
            how = ", when Windows shut down."
        elif word in TRAINING_WORDS:
            how = f", {_DURING.get(record.get('stage'), 'as it started')}."
        else:
            how = "."
        return " ".join(part for part in (f"{word} was stopped on {date}{how}", kept) if part), kept
    if outcome == "failed":
        kept = _failure_kept(record)
        where = f", {_FAILED_AT.get(record.get('stage'), 'before its simulation')}" if word in TRAINING_WORDS else ""
        said = f"{word} failed on {date}{where}: {_cause(record)}."
        return " ".join(part for part in (said, kept) if part), kept
    heard = _moment(record.get("last_heard"))
    when = date_text(heard, now) if heard is not None else "at a moment not known"
    at = _last_heard_at(record, word)
    kept = SAVED_BY_THEN_TEXT if word in TRAINING_WORDS else ""
    return f"PRISM closed while this {word.lower()} was running; last heard {when}{at}.", kept


def status_of(record: dict, now: datetime) -> RunStatus:
    """The status the history ``record`` of an ended run gives, as the foot and Home read it after PRISM is opened
    again: ended, or not seen to end, with how and when, what it kept and its words written against ``now``."""
    outcome = record.get("outcome") if record.get("outcome") in OUTCOMES else "not seen to end"
    sentence, kept = words({**record, "outcome": outcome}, now)
    place = record.get("place")
    return RunStatus(state="not_seen" if outcome == "not seen to end" else "ended",
                     word=str(record.get("word") or ""), name=str(record.get("name") or ""),
                     run=record.get("id") if isinstance(record.get("id"), str) else None,
                     place=tuple(place) if isinstance(place, (list, tuple)) and len(place) == 2 else None,
                     stage=record.get("stage") if isinstance(record.get("stage"), str) else None,
                     outcome=outcome, started=_moment(record.get("started")), ended=_moment(record.get("ended")),
                     last_heard=_moment(record.get("last_heard")), kept=kept, sentence=sentence,
                     acknowledged=bool(record.get("acknowledged")))


def foot(status: RunStatus, now: datetime) -> tuple[str, str, str]:
    """The foot's marker, title and words for ``status``, read from the status alone, ``now`` the moment a date is
    written against: nothing running; a run live (loading, waiting to be tried again, needing someone, stopping, or
    running with the stage in a few words); an end not yet seen (finished, stopped, or failed, a run not seen to end
    among the failed, dated when it was last heard); and nothing running again once the end has been seen."""
    word = status.word or "Run"
    if status.live:
        if status.loading:
            return "now", LOADING_TITLE, ""
        if status.state == "retrying":
            return "now", WAITING_TITLE.format(word), WAITING_LINE
        if status.needs_you:
            return "warn", f"{word} needs you", status.needs_you
        if status.state == "stopping":
            return "now", f"{word} is stopping", status.stage_words
        return "now", f"{word} is running", f"{status.stage_words or 'Running'}. {LOCKED_TEXT}"
    if status.unacknowledged_end:
        if status.outcome == "finished":
            return "done", f"{word} finished", f"{date_text(status.ended, now)}." if status.ended else ""
        when = status.last_heard if status.outcome == "not seen to end" else status.ended
        said = " ".join(part for part in ((f"{date_text(when, now)}." if when else ""), status.kept) if part)
        if status.outcome == "stopped":
            return "half", f"{word} stopped", said
        return "err", f"{word} failed", said
    return "none", IDLE_TITLE, IDLE_WORDS


# ── the parts ──
def _saved(record: dict) -> list[str]:
    """The record a finished run saved, by its kind and its name, and that it was saved without its name when
    another record took the name asked for while the run was going."""
    result = record.get("result") if isinstance(record.get("result"), dict) else {}
    kind, name, asked = result.get("kind"), result.get("name") or "", result.get("asked") or ""
    if not kind:
        return []
    thing = _RECORDS.get(kind, "record")
    said = [f"The {thing} {name} is saved." if name else f"The {thing} is saved."]
    if asked and name != asked:
        said.append(f"It was saved without its name: '{asked}' was taken while the run was going.")
    return said


def _fit_ended(record: dict) -> list[str]:
    """How a fit ended by itself: validation no longer improving for its patience, earlier than the estimate when
    there was one; or its epoch limit reached before it had stopped improving."""
    fit = record.get("fit") if isinstance(record.get("fit"), dict) else {}
    epochs, best, how, expected = fit.get("epochs"), fit.get("best_epoch"), fit.get("ended_by"), fit.get("expected")
    if not _count(epochs) or not _count(best):
        return []
    if how == "patience":
        earlier = (f", earlier than the estimate of about {number_text(expected)}"
                   if _count(expected) and epochs < expected else "")
        return [f"The fit ended at epoch {number_text(epochs)}{earlier}: validation had not improved for "
                f"{number_text(epochs - best)} epochs (best at epoch {number_text(best)})."]
    if how == "limit":
        return [f"The fit stopped at its epoch limit after {number_text(epochs)} epochs, before it had stopped "
                f"improving. The saved network is the last epoch's, not the best (epoch {number_text(best)})."]
    return []


def _stop_kept(record: dict) -> str:
    """What a stopped run kept: a training's by the stage it was stopped in, any other run's what its stop keeps. A
    training whose end PRISM did not write (Windows ended PRISM first) kept what it had saved by then."""
    word = str(record.get("word") or "")
    if word not in TRAINING_WORDS:
        return what_a_stop_keeps(word, str(record.get("name") or ""))
    if not _end_written(record):
        return SAVED_BY_THEN_TEXT
    stage, fit = record.get("stage"), record.get("fit") if isinstance(record.get("fit"), dict) else {}
    if stage == "Simulation":
        batches = _batches(record)
        if not batches:
            return "No simulation was saved."
        return (f"The {_counted(batches, 'batch')} simulated so far {'is' if batches == 1 else 'are'} kept; you can "
                f"continue from batch {number_text(batches)}.")
    if stage == "Fitting":
        simulations = _simulations(record)
        held = f"The {number_text(simulations)} simulations are kept" if simulations else "The simulations are kept"
        if fit.get("discarded"):
            return f"{held}; the training was discarded, as you asked."
        if fit.get("kept_id") and _count(fit.get("passes")):
            return f"{held}; the training is kept up to epoch {number_text(fit['passes'])}, so you can resume it."
        return f"{held}."
    if stage == "Saving":
        if fit.get("kept_id") and not fit.get("discarded"):
            return "The finished training is kept; resuming it writes the posterior."
        return "The simulations are kept."
    return "Nothing was simulated."


def _failure_kept(record: dict) -> str:
    """What a failed training kept: its simulations, and its training up to the epoch kept; nothing said for a run
    of another kind."""
    if str(record.get("word") or "") not in TRAINING_WORDS:
        return ""
    fit = record.get("fit") if isinstance(record.get("fit"), dict) else {}
    if fit.get("kept_id") and _count(fit.get("passes")):
        return f"The simulations and the training up to epoch {number_text(fit['passes'])} are kept."
    batches = _batches(record)
    if record.get("stage") == "Simulation" and batches:
        return f"The simulations up to batch {number_text(batches)} are kept."
    if record.get("stage") in ("Fitting", "Saving"):
        return "The simulations are kept."
    return "No simulation was saved."


def _cause(record: dict) -> str:
    """Why a run failed, as a clause: the graphics card running out of memory, or the first line of the error's
    message."""
    failure = record.get("failure") if isinstance(record.get("failure"), dict) else {}
    if failure.get("kind") == "out of memory":
        return OUT_OF_MEMORY
    said = str(failure.get("message") or "").strip().splitlines()
    said = said[0].strip() if said else ""
    if said.endswith(".") and not said.endswith(".."):
        said = said[:-1]
    return said or NO_MESSAGE


def _last_heard_at(record: dict, word: str) -> str:
    """Where a run not seen to end was when it was last heard, after a comma: a training's stage, with the batch of
    its simulation or the epoch of its fit; another run's stage by its name; nothing when none had begun."""
    stage = record.get("stage")
    if not isinstance(stage, str) or not stage:
        return ""
    if word not in TRAINING_WORDS:
        return f", {stage[:1].lower()}{stage[1:]}"
    if stage == "Simulation":
        held = record.get("simulation_set") if isinstance(record.get("simulation_set"), dict) else {}
        done, total = held.get("at_end"), held.get("batches")
        if _count(done) and _count(total):
            return f", simulation, batch {number_text(done)} of {number_text(total)}"
        return ", simulation"
    if stage == "Fitting":
        fit = record.get("fit") if isinstance(record.get("fit"), dict) else {}
        return f", the fit, epoch {number_text(fit['epochs'])}" if _count(fit.get("epochs")) else ", the fit"
    return {"Rotation": ", the rotation", "Saving": ", saving its posterior"}.get(stage, f", {stage.lower()}")


def _end_written(record: dict) -> bool:
    """Whether the run's own end was written into its history: a run stopped when Windows shut down is written as
    stopped from that moment, with the moment it was last heard at each write, until its own end is written; a
    history that still holds that moment was left by a PRISM Windows ended first."""
    return record.get("last_heard") is None


def _batches(record: dict) -> int:
    """The batches a run's simulation set held at its end, as its last try left them; nothing when not known."""
    tries = record.get("tries") if isinstance(record.get("tries"), list) else []
    last = tries[-1] if tries and isinstance(tries[-1], dict) else {}
    held = last.get("batches_after")
    return held if _count(held) else 0


def _simulations(record: dict) -> int | None:
    """The simulations a training's fit was fitted on: as its fit counted them, or as its simulation set holds them."""
    fit = record.get("fit") if isinstance(record.get("fit"), dict) else {}
    if _count(fit.get("simulations")):
        return fit["simulations"]
    held = record.get("simulation_set") if isinstance(record.get("simulation_set"), dict) else {}
    if _count(held.get("batches")) and _count(held.get("per_batch")):
        return held["batches"] * held["per_batch"]
    return None


def _counted(count: int, unit: str) -> str:
    """``count`` of ``unit`` as the window writes it: "1 batch", "6,150 batches"."""
    plural = "es" if unit == "batch" else "s"
    return f"{number_text(count)} {unit}{'' if count == 1 else plural}"


def _count(value) -> bool:
    """Whether ``value`` is a whole number above nought, as a count the history holds is."""
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _moment(value) -> datetime | None:
    """A moment of a history, as the history writes it (text with its offset) or as it was handed to the writer (a
    time with its zone); None for anything else."""
    if isinstance(value, datetime):
        moment = value
    else:
        try:
            moment = datetime.fromisoformat(value)
        except (TypeError, ValueError):
            return None
    return moment if moment.tzinfo is not None else None
