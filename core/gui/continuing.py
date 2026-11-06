"""What a reopened window can offer, worked out from the run history and the kept trainings, with no toolkit, no torch and
no reading of its own: each kept training, resumable or not and why; the newest training that failed or was not seen
to end with its simulation set unfinished, continued from the batch the set holds; the same with its set whole and no
kept training to resume on it, fitted again; and the newest training that was stopped with its set unfinished, which
the steps say and Home passes over. Every offer is said in the window's words: how it stopped, where and when, and
for a kept narrowing round the command line that resumes it, until its page offers it.

The reading is the caller's (``core/gui/panels/inference/resume.py``'s ``read_offers``), never while a run is live:
the simulation sets' counts are handed in, read from each set's own state, since neither of the history's two numbers
is the set's count for a run not seen to end.
"""
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Iterable

from .step_facts import stopped_words
from .texts import date_text, number_text

#: The kinds of an offer: a kept training to resume; a kept narrowing round, resumed from the command line until its
#: page offers it; a training to continue from the batch its set holds; one whose set is whole, to fit again; and one
#: stopped with its set unfinished, which only the steps say.
KINDS = ("resume", "round", "continue", "fit_again", "stopped")
#: The word of the run history that a training is recorded under.
TRAINING = "Training"
#: The buttons of each offer that can be pressed.
RESUME_BUTTON = "Resume"
CONTINUE_BUTTON = "Continue from batch {}"
FIT_AGAIN_BUTTON = "Fit again on the saved simulations"
ROUND_BUTTON = "Open Training"
#: A kept training with no name, as Records labels one.
UNNAMED = "Unnamed unfinished training"
#: The sentences of a kept training.
READ_BACK_TEXT = "Resuming first reads the simulation set back, about four minutes at full size."
FINISHED_TEXT = "Its fit is finished; resuming writes the posterior without training again."
GONE_PRIOR = "a prior no longer in Records"
ROUND_TEXT = "It is resumed from the command line until its page offers it: "
#: How a history's training ended, as an offer's sentence says it.
HOW = {"failed": "failed", "not seen to end": "was not seen to end", "stopped": "stopped"}
#: The flag a kept narrowing round's command gains when its parent posterior is itself narrowed.
ACCEPT_TRUNCATED = "--accept-truncated"


@dataclass(frozen=True)
class Offer:
    """One thing a reopened window can offer. ``kind`` is one of ``KINDS``; ``run`` the id of the run whose history it
    comes from, or for a kept training the newest run that kept it, None when no history names it; ``kept`` the kept
    training's id; ``batch`` the batches its simulation set holds, read from the set; ``epoch`` the epochs a kept
    training holds; ``sentence`` what Home and the Training step say of it; ``why_not`` why a kept training cannot be
    resumed now, "" when it can; ``label`` its name, or a kept training's label; ``how`` how it stopped, in words;
    ``command`` the command line that resumes a kept narrowing round; ``at`` the moment it is ordered by; ``named``
    whether a kept training has a name of its own."""
    kind: str
    run: "str | None"
    kept: "str | None"
    batch: "int | None"
    sentence: str
    why_not: str
    epoch: "int | None" = None
    label: str = ""
    how: str = ""
    command: str = ""
    at: "datetime | None" = None
    named: bool = True

    @property
    def button(self) -> str:
        """The words of the button that takes the offer, "" for one that has none."""
        if self.why_not:
            return ""
        if self.kind == "resume":
            return RESUME_BUTTON
        if self.kind == "continue":
            return CONTINUE_BUTTON.format(number_text(self.batch or 0))
        if self.kind == "fit_again":
            return FIT_AGAIN_BUTTON
        if self.kind == "round":
            return ROUND_BUTTON
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


def resume_sentence(kept, prior_label: "str | None" = None) -> str:
    """What Home and the Training step say of a kept training: how it stopped, after which epoch and when it was last
    saved, its best validation loss and how many epochs ago, the simulation set and the prior it trains on, and that a
    resume reads the set back first; for one whose fit is finished, that resuming writes the posterior alone.
    ``prior_label`` is how a sentence names the prior ("the prior retrain_prior", or an unnamed one by when it was
    made), None for one no longer in Records."""
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
    if isinstance(simulation, str) and simulation:
        said.append(f"It trains on the simulation set {simulation} and {prior}.")
    said.append(FINISHED_TEXT if state == "finished" else READ_BACK_TEXT)
    return " ".join(said)


def round_command(kept, *, narrowed_parent: bool = False) -> str:
    """The command line that resumes the kept narrowing round ``kept``, given the bounds file, the parent posterior,
    the observation and the kind of observation it was trained with, as its record holds them, the acceptance of a
    parent posterior that is itself narrowed (``narrowed_parent``), and the processor when it trained there."""
    inputs = getattr(kept, "inputs", None) or {}
    parents = getattr(kept, "parents", None) or {}
    config = getattr(kept, "config", None) or {}
    words = [f"python -m core tsnpe --resume-fit {getattr(kept, 'id', '')}"]
    bounds = inputs.get("bounds") if isinstance(inputs, dict) else None
    path = bounds.get("path") if isinstance(bounds, dict) else None
    if isinstance(path, str) and path:
        shown = path if os.path.isabs(path) else f"Resources/{path}"
        words.append(f"--bounds {_quoted(shown)}")
    for flag, key in (("--posterior", "parent_posterior"), ("--observation", "observation")):
        ref = parents.get(key) if isinstance(parents, dict) else None
        if isinstance(ref, str) and ref:
            words.append(f"{flag} {_quoted(ref)}")
    chi = config.get("chi_mode") if isinstance(config, dict) else None
    if isinstance(chi, bool):
        words.append("--chi" if chi else "--no-chi")
    if narrowed_parent:
        words.append(ACCEPT_TRUNCATED)
    if isinstance(config, dict) and config.get("device") == "cpu":
        words.append("--device cpu")
    return " ".join(words)


def _quoted(word: str) -> str:
    return f'"{word}"' if " " in word else word


def _kept_offer(kept, runs: dict, prior_label, parent_narrowed) -> Offer:
    """A kept training as an offer: a narrowing round's or another's, with why it cannot be resumed where it cannot, as
    a sentence."""
    region = getattr(kept, "region", None)
    why = "" if getattr(kept, "can_resume", False) else as_sentence(getattr(kept, "why_not", None)
                                                                   or "it cannot be resumed")
    ident = getattr(kept, "id", None)
    parents = getattr(kept, "parents", None) or {}
    prior = parents.get("prior") if isinstance(parents, dict) else None
    label = prior_label(prior) if isinstance(prior, str) and prior else None
    parent = parents.get("parent_posterior") if isinstance(parents, dict) else None
    narrowed = isinstance(parent, str) and bool(parent) and bool(parent_narrowed(parent))
    command = round_command(kept, narrowed_parent=narrowed) if region is not None else ""
    sentence = resume_sentence(kept, label)
    if region is not None:
        sentence = f"{sentence} {ROUND_TEXT}"
    how = stopped_words(getattr(kept, "state", None), getattr(kept, "stopped_on_purpose", None),
                        getattr(kept, "stop_reason", None))
    at = _moment(getattr(kept, "saved_at", None)) or _moment(getattr(kept, "created", None))
    name = getattr(kept, "posterior_name", "")
    return Offer("round" if region is not None else "resume", runs.get(ident), ident, None, sentence, why,
                 epoch=_whole(getattr(kept, "passes", None)), label=kept_label(kept), how=how, command=command, at=at,
                 named=isinstance(name, str) and bool(name))


def _history_offer(record: dict, kept_on: set, set_batches) -> "Offer | None":
    """The offer the newest training of the records folder in use makes, by how it ended and what its set holds now;
    None for none."""
    outcome = record.get("outcome")
    if outcome not in ("failed", "not seen to end", "stopped"):
        return None
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
        return Offer("stopped", run, None, done, "", "", label=name, how=how, at=at)
    settings = record.get("settings")
    bounds = settings.get("bounds") if isinstance(settings, dict) else None
    if not isinstance(bounds, str) or not bounds:
        return None                              # nothing to fill the forms from: Home says how it ended
    if whole:
        # The history holds the set's folder as the run's line names it, its whole path; a kept training, its name.
        if os.path.basename(os.path.normpath(folder)) in kept_on:
            return None                          # its kept training is resumed instead
        return Offer("fit_again", run, None, done, _ended_sentence(name, how, at, done, whole=True), "", label=name,
                     how=how, at=at)
    return Offer("continue", run, None, done, _ended_sentence(name, how, at, done, whole=False), "", label=name,
                 how=how, at=at)


def _ended_sentence(name: str, how: str, at, done: int, *, whole: bool) -> str:
    """What Home says of a training to continue or to fit again: how and when it ended, and what its set keeps."""
    who = f"The training {name}" if name else "The last training"
    when = f" on {_when(at)}" if at is not None else ""
    if whole:
        return f"{who} {how}{when}. Its simulation set is whole: {number_text(done)} batches are saved."
    return f"{who} {how}{when}. The simulations up to batch {number_text(done)} are saved."


def offers(histories: Iterable[dict], kept_rows: Iterable, *, records_root: str,
           set_batches: Callable[[str], "int | None"],
           prior_label: Callable[[str], "str | None"] = lambda _ident: None,
           parent_narrowed: Callable[[str], bool] = lambda _ident: False) -> tuple[Offer, ...]:
    """Every offer, newest first: each kept training (``kept_rows``, read by their attributes as ``KeptFit`` holds them),
    resumable or not, a narrowing round's with its command; and the newest training of the history of the records
    folder ``records_root`` (``histories``, as ``history.read`` gives them, of any records folder and any order), which,
    failed or not seen to end with a history that names its settings and its bounds file, offers to continue from the
    batch its set holds (``set_batches``, the set's own count by its folder's name, None for no set), or, its set whole
    and no kept training on that set to resume, to fit again; stopped with its set unfinished, it offers what the steps
    alone say. A later training of that folder, whatever its end, leaves an earlier one nothing. ``prior_label`` names a
    kept training's prior by its id, as a sentence names it, None for one no longer in Records; ``parent_narrowed``
    says whether a kept narrowing round's parent posterior, by its id, is itself narrowed."""
    records = [record for record in histories if isinstance(record, dict)]
    kept_rows = list(kept_rows)
    trainings = sorted(((moment, record) for record in records
                        if record.get("word") == TRAINING and _same_folder(record.get("records_root"), records_root)
                        and (moment := _moment(record.get("started"))) is not None),
                       key=lambda pair: (pair[0], _repeat(pair[1].get("id"))))
    runs = {}
    for _moment_at, record in trainings:
        fit = record.get("fit") if isinstance(record.get("fit"), dict) else {}
        if isinstance(fit.get("kept_id"), str) and isinstance(record.get("id"), str):
            runs[fit["kept_id"]] = record["id"]
    kept = [_kept_offer(row, runs, prior_label, parent_narrowed) for row in kept_rows]
    kept_on = {getattr(row, "parents", {}).get("simulation") for row in kept_rows
               if getattr(row, "can_resume", False) and isinstance(getattr(row, "parents", None), dict)}
    found = list(kept)
    if trainings:
        offer = _history_offer(trainings[-1][1], kept_on, set_batches)
        if offer is not None:
            found.append(offer)
    oldest = datetime.min.replace(tzinfo=datetime.now().astimezone().tzinfo)
    return tuple(sorted(found, key=lambda offer: offer.at or oldest, reverse=True))


def for_home(found: Iterable[Offer]) -> "Offer | None":
    """The offer Home's next step names: the newest that can be taken, a training stopped on purpose with its set
    unfinished passed over."""
    return next((offer for offer in found if not offer.why_not and offer.kind != "stopped"), None)


def for_the_steps(found: Iterable[Offer]) -> "Offer | None":
    """The offer the Training step's state says: the newest kept training that can be resumed, or training to continue
    or stopped with its set unfinished; a set to fit again and a kept narrowing round change no step."""
    return next((offer for offer in found if not offer.why_not and offer.kind in ("resume", "continue", "stopped")),
                None)
