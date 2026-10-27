"""A run's failure as its page says it, and the window's tries after the graphics card ran out of memory: what failed
and when, what was saved and what was lost, in the window's own words, and the words of the button that continues or
resumes the run where its page offers one; which failures are tried again, after which pause, and when only closing
PRISM and opening it again can help. Words and rules only: no toolkit, no engine and none of the engine's settings
are imported here, so they are read and tested with no window. Whether a failure is the graphics card running out of
memory is the engine's own rule, handed in (``is_oom``): the engine's rule loads the graphics library, which this
module never does.

Under the sentence the page quotes the error's own message in the fixed-width face; the error's kind and its
traceback are shown only when asked for ("Show the full error"), and are in the run's file. No box is shown for a
run's failure.

The window tries a training or a narrowing round again after the graphics card ran out of memory, ``TRIES`` times
in a row at most, after the pauses ``PAUSES_S``; a try that kept a batch or an epoch more than the failure before it
starts the count again. A try is the run's own call: the simulation set continued once it holds a batch, or, when
the run kept a training, that training resumed. A failure of the graphics card that is no lack of memory is never
tried again, its first time included: the card is then usually lost to this copy of PRISM, and only reopening PRISM
can help, which the page says, naming that cause, as it names the tries that did not help after the card ran out of
memory; the command line that resumes a kept training outside the window stands on its own beneath the page's
sentence (``reopen_command``), to be copied whole. Any other failure is said at once.

"Continue from batch N" and "Resume the training" repeat the run's own call, made with the configuration and the
prior its page's session held then: only into that session (``made_in``).
"""
import math

from ..texts import date_text, number_text
from .questions import TRAINING_WORDS

#: The tries the window makes in a row, after the graphics card ran out of memory, before only reopening PRISM can
#: help; and the pause before each, in seconds of the window's clock.
TRIES = 3
PAUSES_S = (60.0, 180.0, 600.0)
#: The button that ends the tries while the window waits for the next.
STOP_TRYING_TEXT = "Stop trying"
#: Said in the run's text, last, when the tries are stopped.
STOPPED_TRYING_TEXT = "No more tries: the tries were stopped, and the run ends failed."
#: The opening of the page when trying again in this copy of PRISM did not help; the opening when the graphics card
#: failed in a way that is no lack of memory, filled with the first line of the error's message; and what follows
#: either.
REOPEN_TEXT = "Trying again here did not help: this copy of PRISM still holds the graphics card's memory."
DEVICE_TEXT = "The graphics card failed in a way only reopening PRISM can clear: {}."
AGAIN_TEXT = "Close PRISM and open it again"
#: Said on a page whose "Continue from batch N" or "Resume the training" is pressed once its session holds another
#: configuration or another prior than the run's own call was made with: nothing starts, and the offer goes.
CHANGED_TEXT = "The model or its prior has changed since this run began; start it again from the form."
#: What the rail's foot says while a run waits to be tried again: its title, filled with the run's word, and the line
#: under it.
WAITING_TITLE = "{} is waiting to try again"
WAITING_LINE = "Nothing runs until the next try. Settings are locked."
_ORDINALS = ("first", "second", "third", "fourth", "fifth", "sixth")
_COUNTS = ("no", "one", "two", "three", "four", "five", "six")

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


def is_device_failure(exc) -> bool:
    """Whether the graphics card's library reported a failure of the card itself: its device error anywhere in the
    exception's chain, told by the class's name, so that the library is never loaded here."""
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        if any(kind.__name__ == "AcceleratorError" for kind in type(exc).__mro__):
            return True
        exc = exc.__cause__ or exc.__context__
    return False


def is_tried_again(exc, spec, *, is_oom) -> bool:
    """Whether the window tries a run's failure again: a training or a narrowing round (its spec says the window
    tries it again) that failed because the graphics card ran out of memory. Any other failure is said at once: a
    programming error, a full disk or a refusal fails the same way again."""
    return bool(spec.retries) and exc is not None and bool(is_oom(exc))


def gained(kept_before, kept_after) -> bool:
    """Whether a try kept more than there was at the failure before it: a batch more in the simulation set, or an
    epoch more in the training it kept. Each is a pair: the batches the set holds, and the epochs the kept training
    holds (its note's passes), 0 for none."""
    return kept_after[0] > kept_before[0] or kept_after[1] > kept_before[1]


def only_reopening(exc, *, kept_before, kept_after, is_oom, tried: int = TRIES) -> bool:
    """Whether only closing PRISM and opening it again can help a failure: the graphics card failed in a way that is
    not a lack of memory (its library's device error in the chain, not out of memory), at a run's first failure as at
    a try's; or a try ran out of memory and kept neither a batch nor an epoch more than there was at the failure
    before it (``kept_before``, then ``kept_after``, each a pair as ``gained`` reads it; the same pair twice at a
    first failure), and it was the last of ``TRIES`` tries in a row that kept nothing more (``tried``: the tries made
    since the run last kept more)."""
    if is_device_failure(exc) and not is_oom(exc):
        return True
    return bool(is_oom(exc)) and not gained(kept_before, kept_after) and tried >= TRIES


def continued_call(spec) -> dict:
    """The keywords of the run's own call, continued from the batches its simulation set holds: the policy that must
    continue (``resume="require"``) is the one new argument, and the batch it continues from is the set's to say."""
    return {**spec.kwargs, "resume": "require"}


def resumed_call(spec, kept: str) -> dict:
    """The keywords of the run's own call, resuming the training it kept: that training named (``resume_fit``) is the
    one new argument. A narrowing round resumes in the region it was trained in, which the kept training's own record
    holds, so its call names no number of directions and no HPD level: the round refuses both beside a kept
    training."""
    kwargs = {**spec.kwargs, "resume_fit": kept}
    if spec.word == "Narrowing round":
        kwargs.update({name: None for name in ("n_directions", "level") if name in kwargs})
    return kwargs


def failure_words(status, model, exc, *, is_oom=None) -> tuple[str, str]:
    """The page's sentence for a run that failed, as its ``status`` and its ``model`` read it, ``exc`` the error it
    failed with; and the words of the button that continues or resumes it, or ends its tries, "" for none. While the
    run waits to be tried again (``status.state`` "retrying"): what failed, when the next try begins and which of
    the tries it is, and what is saved; once it has ended: what failed and when, what was saved and what was lost, or,
    when only reopening PRISM can help (``status.reopen``), that, why (the tries did not help, or the graphics card
    failed, the first line of its error quoted), and how the run goes on after it. The numbers are the simulation
    set's batches as the status holds them (``continue_from``, read from the set once the run's worker has ended) and
    the epochs of the training it kept (its note's passes), never the records the run's text holds; but the batch that
    was in progress when the graphics card ran out of memory is the one after the last the run's text counted, which
    the set may not hold yet (``_failing``). ``is_oom`` is the engine's rule of a failure that is the graphics card
    running out of memory; None: none is."""
    spec = model.spec
    word, kind, note = spec.word, model.stage_kind(), status.kept_fit
    if status.state == "retrying":
        return _retrying(status, model, kind, note), STOP_TRYING_TEXT
    when = date_text(status.ended, model.now())
    said = _clause(message(exc).splitlines()[0])
    if word not in TRAINING_WORDS:
        return f"{word} failed on {when}: {said}.", ""
    oom = is_oom is not None and exc is not None and bool(is_oom(exc))
    if status.reopen:
        device = exc is not None and is_device_failure(exc) and not oom
        opening = DEVICE_TEXT.format(said) if device else REOPEN_TEXT
        return _reopening(spec, kind, note, status.continue_from or 0, opening), ""
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
        cause = f"{OUT_OF_MEMORY} at batch {number_text(_failing(batches, model))}"
    if not batches:
        return f"{word} failed on {when}, {where}: {cause}. No simulation was saved.", ""
    return (f"{word} failed on {when}, {where}: {cause}. Saved: the simulations up to batch {number_text(batches)}. "
            f"Lost: nothing else.", CONTINUE_TEXT.format(number_text(batches)))


def _retrying(status, model, kind, note) -> str:
    """The sentence of a run that waits to be tried again: what failed, when the next try begins (the time left, in
    whole minutes counted up) and which of the tries it is, and what is kept."""
    left = 0.0 if status.try_at is None else max((status.try_at - model.now()).total_seconds(), 0.0)
    minutes = max(1, math.ceil(left / 60))
    when = (f"Trying again in {minutes} minute{'' if minutes == 1 else 's'}: the "
            f"{_ORDINALS[max(status.trying, 1) - 1]} of {_COUNTS[TRIES]} tries.")
    if kind == "saving":
        kept = "The finished training is kept." if note is not None else "The simulations are saved."
        return f"The graphics card ran out of memory as the posterior was being saved. {when} {kept}"
    if kind == "fit":
        if note is None:
            return f"The graphics card ran out of memory during the fit. {when} The simulations are saved."
        epochs = number_text(note.passes)
        return (f"The graphics card ran out of memory during the fit, at epoch {epochs}. {when} The epochs up to "
                f"{epochs} are kept.")
    batches = status.continue_from or 0
    at = f" at batch {number_text(_failing(batches, model))}" if kind == "simulation" else ""
    saved = (f"The simulations up to batch {number_text(batches)} are saved." if batches
             else "No simulation is saved yet.")
    return f"{OUT_OF_MEMORY[0].upper()}{OUT_OF_MEMORY[1:]}{at}. {when} {saved}"


def _failing(batches: int, model) -> int:
    """The batch that was in progress when a run failed in its simulation: the one after the last the run's text
    counted, which the simulation set, saved only every so many batches, may not hold yet (``batches``, the set's);
    never one the set holds."""
    return max(batches, model.batches_done() or 0) + 1


def reopen_command(status, model) -> str:
    """The command line that the page of a training only reopening PRISM can help sets off on its own beneath its
    sentence, in the fixed-width face, so that it is copied whole with nothing of the sentence: the training it kept
    resumed outside the window (``resume_command``). "" for every other page."""
    spec = model.spec
    if (not status.reopen or status.kept_fit is None or spec.word not in TRAINING_WORDS
            or model.stage_kind() not in ("fit", "saving")):
        return ""
    return resume_command(spec, status.kept_fit.id)


def resume_command(spec, kept: str) -> str:
    """The command line that resumes the kept training ``kept`` outside the window, built from the run's own call
    (``spec``'s): the tool's command for the run (``train`` for a training, ``tsnpe`` for a narrowing round), the kept
    training named, then the flags the training was started with, read off the call (``_flags``). Where the call does
    not tell one of them (bounds typed in by hand, a call with no arguments), the command names the kept training
    alone, and ``untold`` names the flags to give as they were."""
    narrowing = spec.word == "Narrowing round"
    command = f"python -m core {'tsnpe' if narrowing else 'train'} --resume-fit {kept}"
    flags = _flags(spec)
    return command if flags is None else " ".join([command] + flags)


def untold(spec) -> str:
    """The flags of the run's command line that its own call does not tell, as the page names them for a person to
    give as they were ("--bounds, --prior and --chi"); "" when the call tells them all."""
    if _flags(spec) is not None:
        return ""
    return ("--bounds, --posterior, --observation and --chi" if spec.word == "Narrowing round"
            else "--bounds, --prior and --chi")


def _flags(spec) -> list[str] | None:
    """The flags a training was started with, read off the run's own call: its bounds file, its prior (a narrowing
    round's parent posterior and observation, and the acceptance of a parent that is itself narrowed), its kind of
    observation, and the device when it was the processor, which the tool does not pick by itself where a graphics
    card is. None when the call does not tell one of them."""
    narrowing = spec.word == "Narrowing round"
    args = tuple(spec.args)
    cfg = args[0] if args else None
    bounds = (getattr(cfg, "sources", None) or {}).get("bounds")
    chi = getattr(cfg, "chi_mode", None)
    device = getattr(getattr(getattr(cfg, "hw", None), "device", None), "type", None)
    named = ((("--posterior", 1), ("--observation", 3)) if narrowing else (("--prior", 1),))
    refs = [(flag, _ref(args[at] if len(args) > at else None)) for flag, at in named]
    if bounds is None or not isinstance(chi, bool) or any(ref is None for _flag, ref in refs):
        return None
    flags = [f"--bounds {_quoted(str(bounds))}"] + [f"{flag} {_quoted(ref)}" for flag, ref in refs]
    flags.append("--chi" if chi else "--no-chi")
    parent = getattr(args[1], "posterior", None) if narrowing else None
    if getattr(parent, "truncation", None) is not None:
        flags.append("--accept-truncated")
    if device == "cpu":
        flags.append("--device cpu")
    return flags


def made_in(spec, session) -> bool:
    """Whether ``session``, the session of the page that offers to continue or resume a run, still holds what the
    run's own call was made with: its configuration (the call's first argument) and its prior (a training's second
    argument, a narrowing round's third), the very records, not records like them. A page with no session, and a call
    that holds neither, have nothing to compare."""
    args = tuple(spec.args)
    at = 2 if spec.word == "Narrowing round" else 1
    if session is None or len(args) <= at:
        return True
    return args[0] is getattr(session, "cfg", None) and args[at] is getattr(session, "inf_prior", None)


def _ref(record) -> str | None:
    """A record the run's call holds, as the tool names one: by its name, else by its id; None for none."""
    if record is None:
        return None
    return getattr(record, "name", None) or getattr(record, "id", None) or None


def _quoted(word: str) -> str:
    """A word of a command line, in quotes when it holds a space."""
    return f'"{word}"' if " " in word else word


def _reopening(spec, kind, note, batches: int, opening: str) -> str:
    """The page of a training that only reopening PRISM can help: ``opening``, which says why, then that PRISM is
    closed and opened again, and how the training goes on after it: the kept training resumed, until its page offers
    it, from the command line, with the command the page sets off beneath the sentence (``reopen_command``); or the
    same training started again, which continues the simulation set from the batches it holds."""
    word = spec.word
    if note is not None and kind in ("fit", "saving"):
        given = untold(spec)
        tail = f", given the same {given} it was trained with" if given else ""
        return (f"{opening} {AGAIN_TEXT}. The training is kept up to epoch {number_text(note.passes)}; until the "
                f"{word} page offers to resume it, it is resumed from the command line with the command below{tail}.")
    if batches:
        return (f"{opening} {AGAIN_TEXT}, then start the same {word.lower()}: it continues the saved simulations from "
                f"batch {number_text(batches)}.")
    return f"{opening} {AGAIN_TEXT}, then start the same {word.lower()}."


def _clause(text: str) -> str:
    """A message as a clause of the page's sentence: its own closing full stop left out, the sentence's own closes
    it."""
    return text[:-1] if text.endswith(".") and not text.endswith("..") else text
