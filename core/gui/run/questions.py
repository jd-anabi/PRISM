"""What a stop and a close ask, and what a page says while its run stops: the question the stop button asks for each
kind of run and each stage, the question the window asks when it is closed during a run, the question of keeping a
training a stop kept, and the line that says when a stop asked for is taken. Words only: no toolkit, no engine and
none of the engine's settings are imported here, so the words are read and tested with no window.

Each question's first answer is the safe one: it changes nothing, it is the default, and Enter and Escape mean it.
The other answer is drawn as losing work where taking it loses what cannot be had again. A run whose stop loses
nothing (a live simulation, a run that only loads a record) is stopped with no question.
"""
from dataclasses import dataclass

from ..texts import date_text, duration_text, number_text

#: The page's button that withdraws a stop asked for, and the close that waits for it.
WITHDRAW_TEXT = "Don't stop after all"
#: The page's line while the window waits for its run to end before it closes.
CLOSING_TEXT = "PRISM closes when the run has stopped."
#: The page's line while the window waits for a training to save its posterior, which no stop interrupts, before
#: it closes; the page's button that withdraws that close, no stop having been asked; and what it leaves said.
CLOSING_SAVED_TEXT = "PRISM closes when the posterior is saved."
KEEP_OPEN_TEXT = "Keep PRISM open"
KEPT_OPEN_TEXT = "PRISM stays open."
#: Said on the page when a stop is withdrawn in time, filled with the run's word.
WITHDRAWN_TEXT = "The stop was withdrawn. The {} goes on."
#: Said on the page when a stop is withdrawn too late, the run having begun to stop.
TOO_LATE_TEXT = "Too late: the run had already begun to stop."
#: Said on the page when a stop is asked for after the run has ended.
FINISHED_TEXT = "Too late: the run had already finished."
#: Said on the page when a stop is asked for once the training has begun to save its posterior.
SAVING_TEXT = "Too late: the posterior was already being saved, which cannot be stopped."
#: Said under the ended sentence of a training stopped before its fit kept anything.
NOT_KEPT_TEXT = "No training was kept: the fit had not begun."
#: The question of keeping a training a stop kept, its two answers, and what Discard leaves said.
KEEP_QUESTION = "Keep this training so you can resume it later?"
KEEP_TEXT = "Keep"
DISCARD_TEXT = "Discard"
DISCARDED_TEXT = "The training was discarded."
#: The words of the close question.
CLOSE_TITLE = "Close PRISM?"
KEEP_RUNNING = "Keep running"
STOP_AND_CLOSE = "Stop the run and close"
WAIT = "Wait"
CLOSE_WHEN_STOPPED = "Close when it has stopped"
CLOSE_WHEN_SAVED = "Close when it is saved"
#: A training and a narrowing round, which simulate and fit: their stop keeps what each stage leaves.
TRAINING_WORDS = ("Training", "Narrowing round")

# The question of each other kind of run, by its word: title, what its stop keeps, the safe answer, the other.
_OTHERS = {
    "Prior build": ("Stop the prior build?",
                    "Nothing of the prior is kept; it is built again from the start next time.",
                    "Keep building", "Stop the build"),
    "Check": ("Stop the calibration?", "A check writes its record only when it finishes, so nothing of this one is "
              "kept.", "Keep it running", "Stop and discard it"),
    "Inference": ("Stop the inference?", "The inference is not kept. An observation it has already written stays in "
                  "Records.", "Keep it running", "Stop the inference"),
    "Measurement": ("Stop measurement?", "The record {name}is kept, marked unfinished, with what it has measured so "
                    "far. A measurement cannot be continued; to finish it, it is measured again from the start.",
                    "Keep measuring", "Stop and keep what is measured"),
    "Sweep study": ("Stop sweep study?", "What the study has measured is kept, marked unfinished; a study cannot be "
                    "continued.", "Keep measuring", "Stop the study"),
    "Comparison": ("Stop the comparison?", "A comparison writes its record only when it finishes, so nothing of this "
                   "one is kept.", "Keep comparing", "Stop the comparison"),
    "Reduction map": ("Stop computing the map?", "The map will not be shown. If its table is already written, that "
                      "file stays in the reduction folder.", "Keep computing", "Stop computing"),
    "Video": ("Stop saving the video?", "The unfinished file is removed. The trace stays and can be saved again.",
              "Keep saving", "Stop saving"),
}
# The two kinds of run whose stop question says first how long the run has run.
_SAYS_HOW_LONG = ("Measurement", "Sweep study")
# A run the window has no words of its own for (a test's bare run): its stop keeps nothing it has not written.
_ANY = ("Stop the run?", "What the run has not written yet is not kept.", "Keep it running", "Stop the run")


@dataclass(frozen=True)
class Question:
    """One question: its title, its sentence and the words beneath it ("" for none), the safe answer and the other,
    and whether the other loses work."""
    title: str
    sentence: str
    beneath: str
    safe: str
    other: str
    loses_work: bool


def stop_question(status, model) -> "Question | None":
    """The question the stop button asks, for the run and the stage the run is in now; None when the stop asks
    nothing: a run whose stop loses nothing, and a training saving its posterior, whose stop button is greyed."""
    spec = model.spec
    if not spec.stop_asks:
        return None
    if spec.word in TRAINING_WORDS:
        return _training_question(spec, model)
    title, kept, safe, other = _OTHERS.get(spec.word, _ANY)
    return Question(title, _ran(spec, model) + _kept(spec, kept), "", safe, other, True)


def close_question(status, model) -> Question:
    """The question the window asks when it is closed during a run: whether to stop the run and close once it has
    stopped; with a stop asked for already, whether to close once it has stopped; and, for a training saving its
    posterior, which no stop interrupts, whether to close once it is saved. The second answer is drawn as losing
    work where the stop's own is."""
    who = f"{status.word} {status.name}".strip()
    if status.stop_requested:
        return Question(CLOSE_TITLE, f"{who} is stopping.", stopping_line(status, model), WAIT, CLOSE_WHEN_STOPPED,
                        False)
    beneath = [f"It has been running for {duration_text((model.now() - model.started).total_seconds())}"
               f"{_where(model)}."]
    if model.spec.word in TRAINING_WORDS and model.stage_kind() == "saving":
        return Question(CLOSE_TITLE, f"{who} is running.", beneath[0], KEEP_RUNNING, CLOSE_WHEN_SAVED, False)
    stop = stop_question(status, model)
    loses = False
    if stop is not None:
        spec = model.spec
        kept = stop.sentence if spec.word in TRAINING_WORDS else _kept(spec, _OTHERS.get(spec.word, _ANY)[1])
        beneath += [kept] + ([stop.beneath] if stop.beneath else [])
        loses = stop.loses_work
    return Question(CLOSE_TITLE, f"{who} is running.", " ".join(beneath), KEEP_RUNNING, STOP_AND_CLOSE, loses)


def keep_question(note) -> tuple[str, str]:
    """The question of keeping the training a stop kept, read from its note (``passes``, ``resumed``, ``saved_here``,
    ``processes``), never from the records the window saw: the question, and what stands beneath it."""
    epochs, sessions = _counted(note.passes, "epoch"), _counted(note.processes, "session")
    if not note.resumed:
        them = "it" if note.passes == 1 else "them"
        beneath = (f"It stopped after epoch {number_text(note.passes)}; the {epochs} "
                   f"{'is' if note.passes == 1 else 'are'} kept until you discard {them}.")
    elif not note.saved_here:
        beneath = (f"Nothing new was saved since it was resumed. Discard deletes the whole training: {epochs}, saved "
                   f"over {sessions}.")
    else:
        beneath = f"Discard deletes the whole training: {epochs}, saved over {sessions}."
    return KEEP_QUESTION, beneath


def stopping_line(status, model) -> str:
    """When a stop asked for is taken, as the page, the foot and Home say it while the run stops: the run's own
    sentence while it is stopping."""
    return model.sentence()


# ── the parts ──
def _training_question(spec, model) -> "Question | None":
    title = "Stop training?" if spec.word == "Training" else "Stop the narrowing round?"
    kind = model.stage_kind()
    if kind == "saving":
        return None
    if kind == "simulation":
        done = model.batches_done() or 0
        if done:
            batches = _counted(done, "batch")
            sentence = (f"The {batches} simulated so far {'is' if done == 1 else 'are'} kept; the batch in progress "
                        f"is not. You can continue from batch {number_text(done)}.")
        else:
            sentence = "Nothing has been simulated yet; the batch in progress is not kept."
        return Question(title, sentence, "", "Keep training", "Stop and keep the simulations", False)
    if kind == "fit":
        if model.fit_not_saved():
            return Question(title, "The simulations are kept. The fit stops when the epoch in progress ends; this "
                                   "training is not saved as it goes, so the fit so far is lost.", "", "Keep training",
                            "Stop and lose the fit", True)
        bound = model.stop_bound()
        when = f", up to about {bound} from now" if bound is not None else ""
        last = model.last_epoch_ends()
        beneath = ("" if last is None else
                   f"This epoch may be the last: if validation does not improve, the fit ends by itself at about "
                   f"{date_text(last, model.now())}.")
        return Question(title, f"The simulations are kept. The fit stops when the epoch in progress ends{when}; every "
                               f"epoch finished so far is kept.", beneath, "Keep training", "Stop training", False)
    sentence = "Nothing has been simulated yet."
    if spec.facts.get("rotated") is not False:
        sentence += " The rotation will be computed again next time."
    return Question(title, sentence, "", "Keep training", "Stop training", False)


def _ran(spec, model) -> str:
    """How long the run has run, for the kinds of run whose stop says it first."""
    if spec.word not in _SAYS_HOW_LONG:
        return ""
    return f"It has run for {duration_text((model.now() - model.started).total_seconds())}. "


def _kept(spec, kept: str) -> str:
    """What the stop of a run of another kind keeps, its record named where the sentence names one."""
    return kept.replace("{name}", f"{spec.name} " if spec.name else "")


def _where(model) -> str:
    """Where a training is, as the close question says it after how long it has run; nothing before its
    simulation, whose stop sentence says it, and nothing for another run."""
    if model.spec.word not in TRAINING_WORDS:
        return ""
    kind = model.stage_kind()
    if kind == "simulation":
        done = model.batches_done()
        return " and is in its simulation" + ("" if not done else f", at batch {number_text(done)}")
    if kind == "fit":
        epochs = model.epochs()
        return " and is in its fit" + ("" if not epochs else f", at epoch {number_text(epochs[-1].epoch)}")
    if kind == "saving":
        return " and is saving its posterior"
    return ""


def _counted(count: int, unit: str) -> str:
    """``count`` of ``unit`` as the window writes it: "1 epoch", "62 epochs", "6,150 batches"."""
    plural = "es" if unit == "batch" else "s"
    return f"{number_text(count)} {unit}{'' if count == 1 else plural}"
