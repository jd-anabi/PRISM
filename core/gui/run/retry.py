"""A run's failure as its page says it: what failed and when, what was saved and what was lost, in the window's own
words, and the words of the button that continues or resumes the run where its page offers one. Words only: no
toolkit, no engine and none of the engine's settings are imported here, so the words are read and tested with no
window. Whether a failure is the graphics card running out of memory is the engine's own rule, handed in
(``is_oom``): the engine's rule loads the graphics library, which this module never does.

Under the sentence the page quotes the error's own message in the fixed-width face; the error's kind and its
traceback are shown only when asked for ("Show the full error"), and are in the run's file. No box is shown for a
run's failure.
"""
from ..texts import date_text, number_text
from .questions import TRAINING_WORDS

#: Said under the error's message on the page of a run that keeps a file.
FILE_TEXT = "The full error is in the run's file."
#: The button that shows the error's kind and traceback, and the words it takes while they are shown.
SHOW_ERROR_TEXT = "Show the full error"
HIDE_ERROR_TEXT = "Hide the full error"
#: Said on the page of a run that failed after a stop had been asked for: the failure is told as a failure, never as
#: the stop, and the person who asked for the stop is told both.
NOTED_TEXT = "A stop was asked for before the run failed."
#: The buttons that continue a run from the batches its simulation set holds, and resume the training it kept.
CONTINUE_TEXT = "Continue from batch {}"
RESUME_TEXT = "Resume the training"
#: The cause, when the graphics card ran out of memory; and the error's message, when it gives none.
OUT_OF_MEMORY = "the graphics card ran out of memory"
NO_MESSAGE = "the error gave no message"
#: The page of a training whose fit finished but whose posterior could not be written: with the training kept, and
#: with none kept (its fit was not saved as it went).
SAVING_KEPT_TEXT = ("The fit finished but its record could not be written. The finished training is kept: resuming "
                    "it writes the posterior without training again.")
SAVING_LOST_TEXT = "The fit finished but its record could not be written; the training was not kept."


def message(exc) -> str:
    """The error's own message, whole, as the page quotes it; ``NO_MESSAGE`` when it gives none or its message
    cannot be had."""
    try:
        said = str(exc).strip()
    except Exception:                         # noqa: BLE001 -- an error whose message cannot be had is said so
        said = ""
    return said or NO_MESSAGE


def failure_words(status, model, exc, *, is_oom=None) -> tuple[str, str]:
    """The failure page's sentence for a run that failed, as its ended ``status`` and its ``model`` read it, ``exc``
    the error it failed with; and the words of the button that continues or resumes it, "" for none. The sentence
    says what failed and when, what was saved and what was lost; its numbers are the simulation set's batches as the
    status holds them (``continue_from``, read from the set once the run has ended) and the epochs of the training
    it kept (the note's passes), never the records the run's text holds. ``is_oom`` is the engine's rule of a
    failure that is the graphics card running out of memory; None: no failure is."""
    spec = model.spec
    word, kind, note = spec.word, model.stage_kind(), status.kept_fit
    when = date_text(status.ended, model.now())
    said = _clause(message(exc).splitlines()[0])
    if word not in TRAINING_WORDS:
        return f"{word} failed on {when}: {said}.", ""
    oom = is_oom is not None and exc is not None and bool(is_oom(exc))
    cause = OUT_OF_MEMORY if oom else said
    if kind == "saving":
        return (SAVING_KEPT_TEXT, RESUME_TEXT) if note is not None else (SAVING_LOST_TEXT, "")
    if kind == "fit":
        if note is not None:
            epochs = number_text(note.passes)
            return (f"{word} failed on {when}, during the fit, at epoch {epochs}: {cause}. Saved: the simulations, "
                    f"and the training up to epoch {epochs}. Lost: at most the epoch in progress.", RESUME_TEXT)
        lost = ("the fit, which is not saved as it goes" if model.fit_not_saved()
                else "the epoch in progress; no epoch of the fit was kept")
        return f"{word} failed on {when}, during the fit: {cause}. Saved: the simulations. Lost: {lost}.", ""
    batches = status.continue_from or 0
    where = "during simulation" if kind == "simulation" else "before its simulation"
    if oom and kind == "simulation":
        cause = f"{OUT_OF_MEMORY} at batch {number_text(batches + 1)}"
    if not batches:
        return f"{word} failed on {when}, {where}: {cause}. No simulation was saved.", ""
    return (f"{word} failed on {when}, {where}: {cause}. Saved: the simulations up to batch {number_text(batches)}. "
            f"Lost: nothing else.", CONTINUE_TEXT.format(number_text(batches)))


def _clause(text: str) -> str:
    """A message as a clause of the page's sentence: its own closing full stop left out, the sentence's own closes
    it."""
    return text[:-1] if text.endswith(".") and not text.endswith("..") else text
