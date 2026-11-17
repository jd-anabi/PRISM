"""What a reopened window can offer, worked out from the run history and the kept trainings, with no toolkit, no torch and
no reading of its own: each kept training, resumable or not and why, a narrowing round's on its own page; the newest
training, and the newest narrowing round, that failed or was not seen to end with its simulation set unfinished,
continued from the batch the set holds; the same with its set whole and no kept training to resume on it, fitted
again; and the newest that was stopped with its set unfinished, which the steps say and Home passes over. Every offer
is said in the window's words: how it stopped, where and when; and each names the step whose page takes it.

The reading is the caller's (``core/gui/panels/inference/resume.py``'s ``read_offers``), never while a run is live:
the simulation sets' counts are handed in, read from each set's own state, since neither of the history's two numbers
is the set's count for a run not seen to end.
"""
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Iterable

from .keys import place_words
from .step_facts import stopped_words
from .texts import date_text, number_text

#: The kinds of an offer: a kept training to resume; a kept narrowing round, resumed on its own page; a training or a
#: narrowing round to continue from the batch its set holds; one whose set is whole, to fit again; and one stopped with
#: its set unfinished, which only the steps say.
KINDS = ("resume", "round", "continue", "fit_again", "stopped")
#: The words of the run history that a training and a narrowing round are recorded under, and the step of each.
TRAINING = "Training"
ROUND = "Narrowing round"
STEP_OF = {TRAINING: "training", ROUND: "narrowing"}
#: The buttons of each offer that can be pressed.
RESUME_BUTTON = "Resume"
CONTINUE_BUTTON = "Continue from batch {}"
FIT_AGAIN_BUTTON = "Fit again on the saved simulations"
#: A kept training with no name, as Records labels one.
UNNAMED = "Unnamed unfinished training"
#: The sentences of a kept training.
READ_BACK_TEXT = "Resuming first reads the simulation set back, about four minutes at full size."
FINISHED_TEXT = "Its fit is finished; resuming writes the posterior without training again."
GONE_PRIOR = "a prior no longer in Records"
GONE_OBSERVATION = "an observation no longer in Records"
GONE_POSTERIOR = "a posterior no longer in Records"
#: What a kept narrowing round's row says its Resume loads.
LOADS_TEXT = "Resume loads {}, its {} and {}, and puts them in use."
#: The heading of each step's list of what it keeps, and where the past view of a run says a kept training, or a kept
#: narrowing round, that cannot be resumed can be deleted, after saying, against what the run kept then, that it
#: cannot be resumed now.
KEPT_HEADS = {"training": "Kept trainings", "narrowing": "Kept narrowing rounds"}
DELETED_WHERE_TEXT = "It can be deleted under '{}' in {}."
NOT_NOW_TEXT = "It cannot be resumed now."
#: How a history's training ended, as an offer's sentence says it.
HOW = {"failed": "failed", "not seen to end": "was not seen to end", "stopped": "stopped"}


@dataclass(frozen=True)
class Offer:
    """One thing a reopened window can offer. ``kind`` is one of ``KINDS``; ``run`` the id of the run whose history it
    comes from, or for a kept training the newest run that kept it, None when no history names it; ``kept`` the kept
    training's id; ``batch`` the batches its simulation set holds, read from the set; ``epoch`` the epochs a kept
    training holds; ``sentence`` what Home and the Training step say of it; ``why_not`` why a kept training cannot be
    resumed now, "" when it can; ``label`` its name, or a kept training's label; ``how`` how it stopped, in words;
    ``at`` the moment it is ordered by; ``named`` whether a kept training has a name of its own; ``step`` the step whose
    page takes it, "training", or "narrowing" for a narrowing round's; for a kept narrowing round, ``loads`` what its
    Resume loads and puts in use, in a sentence ("" while one of them is no longer in Records), and ``match`` what a
    new round's form is matched against: its prior, its parent posterior and its observation by their ids, its region's
    level, its batches, simulations per batch and network."""
    kind: str
    run: "str | None"
    kept: "str | None"
    batch: "int | None"
    sentence: str
    why_not: str
    epoch: "int | None" = None
    label: str = ""
    how: str = ""
    at: "datetime | None" = None
    named: bool = True
    step: str = "training"
    loads: str = ""
    match: tuple = ()

    @property
    def button(self) -> str:
        """The words of the button that takes the offer, "" for one that has none."""
        if self.why_not:
            return ""
        if self.kind in ("resume", "round"):
            return RESUME_BUTTON
        if self.kind == "continue":
            return CONTINUE_BUTTON.format(number_text(self.batch or 0))
        if self.kind == "fit_again":
            return FIT_AGAIN_BUTTON
        return ""


def _moment(value) -> "datetime | None":
    """A time as a record holds it, with its zone, in local time; None for anything else."""
    if isinstance(value, datetime):
        moment = value
    else:
        try:
            moment = datetime.fromisoformat(value)
        except (TypeError, ValueError):
            return None
    if moment.tzinfo is None:
        return None
    try:
        return moment.astimezone()
    except (OverflowError, OSError):
        return None


def _when(moment: "datetime | None") -> str:
    """A moment as the window writes one in a sentence: local time, the year form; "" for none."""
    return "" if moment is None else date_text(moment)


def _same_folder(one, other) -> bool:
    if not isinstance(one, str) or not isinstance(other, str) or not one or not other:
        return False
    return os.path.normcase(os.path.abspath(one)) == os.path.normcase(os.path.abspath(other))


def _repeat(ident) -> int:
    """Which of the runs started in one second a run is: its id's own count, 1 for the first, whose id has none
    (``history.run_id`` gives the second ``-2``), so that of two runs started in one second the later counts as the
    newer."""
    stem, _dash, count = str(ident or "").rpartition("-")
    return int(count) if stem and count.isdigit() else 1


def _whole(value) -> "int | None":
    return value if type(value) is int else None


def kept_label(kept) -> str:
    """A kept training as Records labels it: its name, or "Unnamed unfinished training" and when it was made."""
    name = getattr(kept, "posterior_name", "") or ""
    if isinstance(name, str) and name:
        return name
    made = _moment(getattr(kept, "created", None))
    return f"{UNNAMED}, {_when(made)}" if made is not None else UNNAMED


def as_sentence(clause) -> str:
    """A clause as a sentence: its first letter a capital, a full stop at its end; "" for none."""
    clause = str(clause or "").strip()
    if not clause:
        return ""
    return clause[:1].upper() + clause[1:] + ("" if clause.endswith((".", "!", "?")) else ".")


def resume_sentence(kept, prior_label: "str | None" = None, *, around: "str | None" = None,
                    narrowed_from: "str | None" = None, resumable: bool = True) -> str:
    """What Home and the step that lists it say of a kept training: how it stopped, after which epoch and when it was
    last saved, its best validation loss and how many epochs ago, for a narrowing round's the observation it trains
    around and the posterior it was narrowed from, the simulation set and the prior it trains on, and that a resume
    reads the set back first; for one whose fit is finished, that resuming writes the posterior alone. Nothing of what
    a resume does for one that cannot be resumed (``resumable`` False), whose row says why instead. ``prior_label``,
    ``around`` and ``narrowed_from`` are how a sentence names the prior, the observation and the parent posterior ("the
    prior retrain_prior", or an unnamed one by when it was made), None for one no longer in Records."""
    state = getattr(kept, "state", "") or ""
    passes = _whole(getattr(kept, "passes", None))
    saved = _when(_moment(getattr(kept, "saved_at", None)))
    after = f" after epoch {number_text(passes)}" if passes is not None else ""
    last = f", last saved {saved}" if saved else ""
    if state == "kept when stopped":
        opening = f"It was kept when stopped{after}{last}."
    elif state == "finished":
        opening = f"Its fit ended{after}{last}."
    else:
        reason = getattr(kept, "stop_reason", None)
        error = (getattr(kept, "stopped_on_purpose", None) is False and isinstance(reason, str)
                 and bool(reason.strip()))
        opening = f"It stopped unexpectedly{after}{last}" + (f", after the error '{reason.strip()}'." if error
                                                             else ".")
    said = [opening]
    best, since = getattr(kept, "best_validation_loss", None), _whole(getattr(kept, "passes_since_best", None))
    if isinstance(best, (int, float)) and not isinstance(best, bool):
        ago = ("at its last epoch" if since == 0 else "1 epoch ago" if since == 1
               else f"{number_text(since)} epochs ago" if since is not None else "")
        said.append(f"Best validation loss {best:.2f}" + (f", {ago}." if ago else "."))
    parents = getattr(kept, "parents", None) or {}
    simulation = parents.get("simulation") if isinstance(parents, dict) else None
    prior = prior_label or GONE_PRIOR
    if getattr(kept, "region", None) is not None:
        said.append(f"It trains around {around or GONE_OBSERVATION}, narrowed from {narrowed_from or GONE_POSTERIOR}"
                    + (f", on the simulation set {simulation} and {prior}." if isinstance(simulation, str) and simulation
                       else f", on {prior}."))
    elif isinstance(simulation, str) and simulation:
        said.append(f"It trains on the simulation set {simulation} and {prior}.")
    if resumable:
        said.append(FINISHED_TEXT if state == "finished" else READ_BACK_TEXT)
    return " ".join(said)


def not_resumable_words(offer) -> str:
    """What the past view of a run says of the kept training, or the kept narrowing round, ``offer`` holds that cannot
    be resumed, under the run's own sentence, which says what the run kept then: that it cannot be resumed now, why,
    and where it can be deleted; "" for one that can be."""
    why = getattr(offer, "why_not", "")
    if not why:
        return ""
    step = getattr(offer, "step", "training")
    return f"{NOT_NOW_TEXT} {why} {DELETED_WHERE_TEXT.format(KEPT_HEADS[step], place_words(step))}"


def _kept_offer(kept, runs: dict, prior_label, label, gone) -> Offer:
    """A kept training as an offer: a narrowing round's, on its page, or another's, with why it cannot be resumed where
    it cannot, as a sentence: the record's own reason, else what of the records its Resume loads is gone (``gone``),
    as its press says it; its own sentence then says nothing of what a resume does."""
    region = getattr(kept, "region", None)
    why = (" ".join(gone(kept)) if getattr(kept, "can_resume", False)
           else as_sentence(getattr(kept, "why_not", None) or "it cannot be resumed"))
    ident = getattr(kept, "id", None)
    parents = getattr(kept, "parents", None) or {}
    parents = parents if isinstance(parents, dict) else {}

    def named(kind, key, reader):
        ref = parents.get(key)
        return reader(kind, ref) if isinstance(ref, str) and ref else None

    prior = named("prior", "prior", lambda _kind, ref: prior_label(ref))
    loads, match = "", ()
    if region is not None:
        around, narrowed_from = named("observation", "observation", label), named("posterior", "parent_posterior", label)
        sentence = resume_sentence(kept, prior, around=around, narrowed_from=narrowed_from, resumable=not why)
        if around and narrowed_from and prior:
            loads = LOADS_TEXT.format(narrowed_from, prior[4:] if prior.startswith("the ") else prior, around)
        record = getattr(kept, "config", None) or {}
        record = record if isinstance(record, dict) else {}
        level = region.get("level") if isinstance(region, dict) else None
        match = (parents.get("prior"), parents.get("parent_posterior"), parents.get("observation"), level,
                 *(record.get(key) for key in ("num_runs", "run_size_cap", "hidden_features", "num_transforms")))
    else:
        sentence = resume_sentence(kept, prior, resumable=not why)
    how = stopped_words(getattr(kept, "state", None), getattr(kept, "stopped_on_purpose", None),
                        getattr(kept, "stop_reason", None))
    at = _moment(getattr(kept, "saved_at", None)) or _moment(getattr(kept, "created", None))
    name = getattr(kept, "posterior_name", "")
    return Offer("round" if region is not None else "resume", runs.get(ident), ident, None, sentence, why,
                 epoch=_whole(getattr(kept, "passes", None)), label=kept_label(kept), how=how, at=at,
                 named=isinstance(name, str) and bool(name), step="narrowing" if region is not None else "training",
                 loads=loads, match=match)


def _history_offer(record: dict, kept_on: set, set_batches) -> "Offer | None":
    """The offer the newest training, or the newest narrowing round, of the records folder in use makes, by how it
    ended and what its set holds now; None for none. A narrowing round's history offers to continue only when its
    settings name the posterior it was narrowed from and the observation it was trained around, as a round's run
    hands them: one run before they were kept offers nothing to press."""
    outcome = record.get("outcome")
    if outcome not in ("failed", "not seen to end", "stopped"):
        return None
    word = record.get("word")
    step = STEP_OF.get(word, "training")
    held = record.get("simulation_set")
    held = held if isinstance(held, dict) else {}
    folder, total = held.get("folder"), _whole(held.get("batches"))
    if not isinstance(folder, str) or not folder:
        return None
    done = set_batches(folder)
    if not isinstance(done, int) or isinstance(done, bool) or done <= 0:
        return None
    whole = total is not None and done >= total
    at = _moment(record.get("ended")) or _moment(record.get("last_heard")) or _moment(record.get("started"))
    name = record.get("name") if isinstance(record.get("name"), str) else ""
    how = HOW[outcome]
    run = record.get("id") if isinstance(record.get("id"), str) else None
    if outcome == "stopped":
        if whole:
            return None
        return Offer("stopped", run, None, done, "", "", label=name, how=how, at=at, step=step)
    settings = record.get("settings")
    bounds = settings.get("bounds") if isinstance(settings, dict) else None
    if not isinstance(bounds, str) or not bounds:
        return None                              # nothing to fill the forms from: Home says how it ended
    if step == "narrowing" and not all(isinstance(settings.get(key), str) and settings.get(key)
                                       for key in ("parent_posterior", "observation")):
        return None                              # a round run before its settings were kept: Home says how it ended
    what = "narrowing round" if step == "narrowing" else "training"
    if whole:
        # The history holds the set's folder as the run's line names it, its whole path; a kept training, its name. A
        # set a kept training is on offers no fit again: its kept training is resumed instead, or, where a record that
        # training's Resume loads is gone, the press would refuse a fit again for that same record.
        if os.path.basename(os.path.normpath(folder)) in kept_on:
            return None
        return Offer("fit_again", run, None, done, _ended_sentence(name, how, at, done, whole=True, what=what), "",
                     label=name, how=how, at=at, step=step)
    return Offer("continue", run, None, done, _ended_sentence(name, how, at, done, whole=False, what=what), "",
                 label=name, how=how, at=at, step=step)


def _ended_sentence(name: str, how: str, at, done: int, *, whole: bool, what: str = "training") -> str:
    """What Home says of a training, or a narrowing round (``what``), to continue or to fit again: how and when it
    ended, and what its set keeps."""
    who = f"The {what} {name}" if name else f"The last {what}"
    when = f" on {_when(at)}" if at is not None else ""
    if whole:
        return f"{who} {how}{when}. Its simulation set is whole: {number_text(done)} batches are saved."
    return f"{who} {how}{when}. The simulations up to batch {number_text(done)} are saved."


def offers(histories: Iterable[dict], kept_rows: Iterable, *, records_root: str,
           set_batches: Callable[[str], "int | None"],
           prior_label: Callable[[str], "str | None"] = lambda _ident: None,
           label: Callable[[str, str], "str | None"] = lambda _kind, _ident: None,
           gone: Callable[[object], Iterable[str]] = lambda _kept: ()) -> tuple[Offer, ...]:
    """Every offer, newest first: each kept training (``kept_rows``, read by their attributes as ``KeptFit`` holds them),
    resumable or not, a narrowing round's on its own page; and, of the history of the records folder ``records_root``
    (``histories``, as ``history.read`` gives them, of any records folder and any order), the newest training and the
    newest narrowing round, each by the same rule: failed or not seen to end with a history that names its settings and
    its bounds file (a round's, its parent posterior and its observation too), it offers to continue from the batch its
    set holds (``set_batches``, the set's own count by its folder as the history holds it, its whole path, None for no
    set), or, its set whole and no kept training on that set to resume, to fit again; stopped with its set unfinished, it
    offers what the steps alone say. A later run of the same word of that folder, whatever its end, leaves an earlier one
    nothing; a run of the other word does not count. ``prior_label`` names a kept training's prior by its id, as a
    sentence names it, None for one no longer in Records; ``label`` names a kept narrowing round's observation and
    parent posterior by their kind and id, the same way; ``gone`` says, in sentences, what of the records a kept
    training's Resume loads is no longer in Records, and a kept training it says any of offers no Resume."""
    records = [record for record in histories if isinstance(record, dict)]
    kept_rows = list(kept_rows)
    newest, runs = [], {}
    for word in (TRAINING, ROUND):
        ran = sorted(((moment, record) for record in records
                      if record.get("word") == word and _same_folder(record.get("records_root"), records_root)
                      and (moment := _moment(record.get("started"))) is not None),
                     key=lambda pair: (pair[0], _repeat(pair[1].get("id"))))
        for _moment_at, record in ran:
            fit = record.get("fit") if isinstance(record.get("fit"), dict) else {}
            if isinstance(fit.get("kept_id"), str) and isinstance(record.get("id"), str):
                runs[fit["kept_id"]] = record["id"]
        if ran:
            newest.append(ran[-1][1])
    kept = [_kept_offer(row, runs, prior_label, label, gone) for row in kept_rows]
    # The sets of the kept trainings whose own records let them be resumed, one whose Resume would load a record no
    # longer in Records among them: a fit again on that set would be refused for the same record.
    kept_on = {getattr(row, "parents", {}).get("simulation") for row in kept_rows
               if getattr(row, "can_resume", False) and isinstance(getattr(row, "parents", None), dict)}
    found = list(kept)
    for record in newest:
        offer = _history_offer(record, kept_on, set_batches)
        if offer is not None:
            found.append(offer)
    oldest = datetime.min.replace(tzinfo=datetime.now().astimezone().tzinfo)
    return tuple(sorted(found, key=lambda offer: offer.at or oldest, reverse=True))


def for_home(found: Iterable[Offer]) -> "Offer | None":
    """The offer Home's next step names: the newest that can be taken, a run stopped on purpose with its set unfinished
    passed over."""
    return next((offer for offer in found if not offer.why_not and offer.kind != "stopped"), None)


def for_the_steps(found: Iterable[Offer]) -> "Offer | None":
    """The offer the Training step's state says: the newest kept training that can be resumed, or training to continue
    or stopped with its set unfinished; a set to fit again and anything of a narrowing round change no step of it."""
    return next((offer for offer in found if not offer.why_not and offer.step == "training"
                 and offer.kind in ("resume", "continue", "stopped")), None)


def for_the_round(found: Iterable[Offer]) -> "Offer | None":
    """The offer the Narrowing round step's state says: the newest kept narrowing round that can be resumed, or
    narrowing round to continue; None for none."""
    return next((offer for offer in found if not offer.why_not and offer.step == "narrowing"
                 and offer.kind in ("round", "continue")), None)
