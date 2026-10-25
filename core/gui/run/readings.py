"""What the run's page reads off the engine's lines: one table, a row for each line it reads, and the window's own
sentence for each warning and error it can say in its own words.

A row holds the line's exact form as the engine writes it, ``<name>`` standing for what varies, and the line is
matched on its whole shape, never on how it begins: the record of an epoch and the save that follows it both begin
``[fit]``. A line no row matches is text, never an error. Each row names the engine module that writes its line,
and the engine's own function that builds it where there is one, as dotted paths written out: nothing of the engine
is imported here, and neither is the toolkit, so the table can be read and tested with no window and no engine
loaded. A test holds each row to the module it names, so a line reworded in the engine fails it.

``message`` says which rows are messages on the run's page: a warning or an error, and the two lines of
information that say a training is not saved as it goes. Any other warning or error is a message too, said from
the engine as it stands. A print (a line at ``info`` from the output stream, or at ``err``) never is.
"""
import re
from dataclasses import dataclass

from ..texts import number_text

# What each placeholder of a form matches, and how its text is read. A whole number may carry the commas the
# engine writes between thousands; a real number is any the engine prints, an infinity and a non-number among
# them; a text is the shortest that lets the rest of the line match, and two of them may be empty.
_WHOLE = r"-?\d[\d,]*"
_REAL = r"[-+]?(?:inf|nan|\d[\d,]*(?:\.\d+)?(?:e[-+]?\d+)?)"
_TEXT = r".+?"
_MAYBE = r".*?"
_SHARE = r"[\d.]+%"
# A batch as the engine names it inside a line: the training batch, counted from one, with the settings it was
# simulated at. A library's warning comes with the name of its kind before it.
_BATCH_TAG = r"(?:\w+: )?training batch (?P<batch>\d[\d,]*)/(?P<batches>\d[\d,]*) \[[^\]]*\]"

_PLACES: dict[str, str] = {
    **{name: _WHOLE for name in (
        "batches", "rows", "simulations", "k", "n", "every", "a", "b", "new", "nonfinite", "constant",
        "overflow", "bad", "masked", "total", "epoch", "since", "passes", "attempt", "attempts", "half", "i",
        "kept", "dropped", "probes")},
    **{name: _REAL for name in (
        "spread", "need", "free", "share", "low", "high", "median", "peak", "reserved", "available", "card",
        "training_loss", "validation_loss", "seconds", "wait", "magnitude", "cycles", "value")},
    **{name: _SHARE for name in ("lower", "upper")},
    **{name: _TEXT for name in (
        "dir", "id", "error", "reason", "scope", "where", "basis", "rotation", "note", "path", "what", "how",
        "matches", "progress", "param", "fixed", "cap")},
    **{name: _MAYBE for name in ("free_note", "complete")},
    "tag": _BATCH_TAG,
}
# The placeholders, and the groups of the batch's name, read as numbers; every other is kept as written.
_WHOLES = frozenset(name for name, rx in _PLACES.items() if rx == _WHOLE) | {"batch", "batches"}
_REALS = frozenset(name for name, rx in _PLACES.items() if rx == _REAL)
_PLACEHOLDER = re.compile(r"<(\w+)>")


@dataclass(frozen=True)
class Reading:
    """One line read: the row's name, what its placeholders held (whole numbers and real numbers read as numbers,
    the rest as written) and the level the line arrived at."""
    name: str
    values: dict
    level: str


@dataclass(frozen=True)
class Row:
    """One line the window reads. ``form`` is the line as the engine writes it, ``<name>`` for what varies;
    ``pattern`` the expression made from it; ``module`` the engine module that writes it and ``formatter`` the
    engine's function that builds it, if any, both dotted paths; ``sentence`` the window's own words for it, with
    the same placeholders; ``message``: "warning", "error", "information" or None."""
    name: str
    form: str
    pattern: re.Pattern
    module: str
    sentence: str | None
    formatter: str | None
    message: str | None


def _pattern(form: str) -> re.Pattern:
    """The expression that matches a line of ``form`` whole: its fixed words as they are, each placeholder as
    ``_PLACES`` gives it."""
    parts, at = [], 0
    for found in _PLACEHOLDER.finditer(form):
        parts.append(re.escape(form[at:found.start()]))
        name = found.group(1)
        parts.append(_PLACES[name] if name == "tag" else f"(?P<{name}>{_PLACES[name]})")
        at = found.end()
    parts.append(re.escape(form[at:]))
    return re.compile("".join(parts), re.DOTALL)


def _row(name, form, module, *, sentence=None, formatter=None, message=None) -> Row:
    return Row(name, form, _pattern(form), module, sentence, formatter, message)


_ORCH, _PIPE, _FIT, _TRAIN = "core.orchestrator", "core.SBI.pipeline", "core.SBI.fit_state", "core.SBI.train"
_FDT, _SWEEP = "core.FDT.fdt_pipeline", "core.FDT.cross_validation"
# Said where a line of the training's fit-saving holds an error the window has no words of its own for.
_NOT_SAVED = "a stop or a failure during its fit loses the fit."

ROWS: tuple[Row, ...] = (
    # the training's totals, its rotation and its simulation
    _row("budget", "[budget] <batches> batches x <rows> rows = <simulations> training rows", _ORCH),
    _row("rotation_begins", "Computing decorrelating Fisher rotation (REPARAM_ROTATE=True)...", _ORCH),
    _row("rotation_done", "[fisher] eigenvalue spread (best/worst direction): <spread>", _ORCH),
    _row("rotation_reused", "Reusing the Fisher rotation stored with the training checkpoint (<k>/<n> "
         "batches<complete>) — NOT recomputing it: the rotation's operating points are not reproducible across "
         "processes, so a fresh V would put the reused rows in a different coordinate than the targets stored "
         "beside them.", _ORCH),
    _row("rotation_parent", "[tsnpe] basis: reusing the PARENT posterior's rotation carried with the region "
         "(<basis>); the Fisher is NOT recomputed for a truncated round -- a fresh V would put the box on axes it "
         "was never measured on.", _ORCH),
    _row("simulation_writing", "[checkpoint] writing to <dir> every <every> batches (~<need> GiB total, <free> GiB "
         "free)", _PIPE),
    _row("simulation_resuming", "[checkpoint] resuming at batch <k>/<n> from <dir> (<rotation>)", _PIPE),
    _row("simulation_complete", "[checkpoint] complete: <n> batches in <dir>. Safe to delete once the posterior is "
         "saved; keeping it lets you retrain the flow without re-simulating.", _PIPE),
    _row("winsor", "[winsor] clipped <n> of <total> summary elements (<share>%) to their per-column "
         "<lower>/<upper> percentiles", "core.SBI.summaries"),
    # the fit
    _row("fit_epoch", "[fit] epoch <epoch>: training loss <training_loss>, validation loss <validation_loss>, took "
         "<seconds> s, epochs since the best <since>", _TRAIN, formatter="core.SBI.train.fit_epoch_line"),
    _row("sbi_converged", "Neural network successfully converged after <n> epochs.", "sbi.inference.trainers.base"),
    # failed simulations, unusable driven recordings, the graphics card's memory
    _row("patho_batch", "[patho] batch <k>: <new> new pathological trajectorie(s) -- <nonfinite> non-finite, "
         "<constant> exactly constant, <overflow> over <magnitude> in magnitude, of <rows> simulated", _PIPE,
         sentence="Failed simulations: <new> more in batch <batch>.", message="warning"),
    _row("patho_total", "[patho] run total: <bad> pathological of <rows> simulated trajectories (<share>%) -- "
         "<nonfinite> non-finite, <constant> exactly constant, <overflow> over <magnitude>", _PIPE),
    _row("chi_batch", "<tag>: chi: <dropped>/<probes> probes masked (below <cycles> drive cycles, at/above "
         "Nyquist, out of band, or a non-finite lock-in).", "core.SBI.chi_probes",
         sentence="Unusable driven recordings were reported in <counted> (latest: batch <batch>, <dropped> of "
                  "<probes>).", message="warning"),
    _row("chi_total", "[chi] masked probes: <masked> of <total> (<share>%) over <batches> batches, <scope>; per "
         "batch <low>-<high>%, median <median>%", _PIPE,
         formatter="core.SBI.pipeline._BatchProbeLedger.summary_line"),
    _row("chi_none", "[chi] masked probes: none counted, <scope>", _PIPE,
         formatter="core.SBI.pipeline._BatchProbeLedger.summary_line"),
    _row("mem", "[mem] <where>: peak allocated <peak> GiB, peak reserved <reserved> GiB, <available>/<card> GiB "
         "reported free (optimistic on Windows), learned cap <cap>", _PIPE),
    _row("mem_unavailable", "[mem] <where>: memory statistics unavailable (<error>)", _PIPE),
    _row("oom_split_simulation", "<tag>: OOM at simulation batch <rows>; retrying in chunks of "
         "<half><free_note>. Original: <note>", _PIPE,
         sentence="Batch <batch> did not fit in the graphics card's memory and was run in smaller pieces.",
         message="warning"),
    _row("oom_split_rows", "<tag>: OOM with <rows> rows OUTSIDE the simulator retry; re-running this batch in "
         "halves of <half><free_note>. Original: <note>", _PIPE,
         sentence="Batch <batch> did not fit in the graphics card's memory and was run in smaller pieces.",
         message="warning"),
    _row("batch_failed", "<tag>: batch FAILED after both halving retries (<attempt>/<attempts>)<free_note>. "
         "Waiting <wait>s and re-running the whole batch. Original: <note>", _PIPE,
         sentence="Batch <batch> failed for lack of graphics-card memory and is being run again after a pause.",
         message="error"),
    # a stop during the simulation, and a save put off
    _row("stopping", "[checkpoint] stopping: saving <n> completed batches (<a> -> <b>) before unwinding…", _PIPE),
    _row("save_put_off", "[checkpoint] deferring the write at batch <k>: no RNG snapshot. The rows are still held "
         "and go out at the next boundary.", _PIPE,
         sentence="The save of the simulations at batch <k> was put off to the next save.", message="warning"),
    # the training saved as it goes, kept, resumed
    _row("fit_saving_off", "[fit] saving is switched off: this training is not saved as it goes and cannot be "
         "resumed.", _ORCH, sentence=f"This training is not saved as it goes: {_NOT_SAVED}", message="information"),
    _row("fit_not_kept", "[fit] the simulation set is not kept (checkpointing off), so this training is not saved "
         "as it goes and cannot be resumed.", _ORCH,
         sentence=f"Its simulation set is not kept (checkpointing is off), so this training is not saved as it "
                  f"goes: {_NOT_SAVED}", message="information"),
    _row("fit_listing_failed", "[fit] the kept trainings could not be listed to name the ones that match these "
         "settings (<error>); this run starts a new one.", _ORCH, message="warning"),
    _row("fit_matches", "[fit] <kept> kept training(s) match these settings and can be resumed: <matches>; this "
         "run starts a new one.", _ORCH),
    _row("fit_resume_fallback", "[fit] the newest saved pass of the kept training <id> could not be read; it "
         "resumes from the pass before it, pass <passes>.", _FIT,
         sentence="The newest saved epoch of the kept training could not be read; it resumes from the epoch before "
                  "it, epoch <passes>.", message="warning"),
    _row("fit_resuming", "[fit] resuming the kept training <id> after pass <passes> (<progress>); its simulation "
         "set is read back first.", _FIT),
    _row("fit_mark_failed", "[fit] could not mark the kept training <id> as one that cannot be resumed (<error>).",
         _FIT, message="warning"),
    _row("fit_folder_left", "[fit] could not remove the folder <path>, which holds no kept training (<error>); the "
         "store lists it as a leftover to clear.", _FIT,
         sentence="A folder of a training that was not kept could not be removed; Records lists it as a leftover "
                  "to clear.", message="warning"),
    _row("fit_saved", "[fit] saved after pass <passes>.", _FIT),
    _row("fit_save_failed", "[fit] the last save failed: <reason>; the kept training stands at pass <passes>.", _FIT,
         sentence="The last save of the training failed (<reason>); the kept training stands at epoch <passes>.",
         message="warning"),
    _row("fit_save_failed_nothing_kept", "[fit] the last save failed: <reason>; nothing is kept yet.", _FIT,
         sentence="The last save of the training failed (<reason>); nothing of it is kept yet.",
         message="warning"),
    _row("fit_stopped_nothing_kept", "[fit] the training stopped before its fit saved a pass, so no kept training "
         "was made; the simulation set keeps the batches it committed.", _FIT),
    _row("fit_stopped_not_saved_here", "[fit] the training stopped before it saved a pass in this process; it is "
         "kept as fit <id>, as it was after pass <passes>.", _FIT),
    _row("fit_stopped_kept", "[fit] the training stopped after pass <passes> and is kept as fit <id> (<how>).",
         _FIT),
    _row("fit_stop_write_failed", "[fit] could not write the kept training's <what> as it stopped (<error>).", _FIT,
         message="warning"),
    _row("fit_finish_mark_failed", "[fit] could not mark the kept training <id> finished (<error>).", _FIT,
         message="warning"),
    _row("fit_remove_failed", "[fit] the kept training <id> could not be removed after its posterior was written "
         "(<error>); it can be deleted.", _FIT, message="warning"),
    _row("fit_no_summary", "[fit] this sbi's trainer has no per-pass summary to read, so no line is written per "
         "epoch.", _TRAIN,
         sentence="This version of the training library gives no line per epoch, so the fit's epochs, pace and loss "
                  "are not shown.", message="warning"),
    _row("fit_record_skipped", "[fit] an epoch's record could not be written (<error>), so the epochs of this "
         "training are not all said.", _TRAIN,
         sentence="An epoch's record could not be read from the training library (<error>), so not every epoch "
                  "is shown.", message="warning"),
    # the window's own file
    _row("text_file_broken", "The run's text file could not be written (<error>); the full text is still on the "
         "page.", "core.gui.run.textfile",
         sentence="The run's text file could not be written (<error>); the full text is still on the page.",
         message="warning"),
    # One cell and Sweep study
    _row("fdt_sanity_skipped", "Skipping sanity checks.", _FDT),
    _row("fdt_campaign_spontaneous", "Campaign 1: spontaneous fluctuations -> PSD", _FDT),
    _row("fdt_campaign_driven", "Campaign 2: forced response -> chi via lock-in", _FDT),
    _row("sweep_phase", "--- Phase A (<param> sweep): spontaneous PSD + omega_0 detection ---", _SWEEP),
    _row("sweep_point_a", "[A <i>/<n>] <param>=<value> (<fixed>)", _SWEEP),
    _row("sweep_point_b", "[B <i>/<n>] <param>=<value>", _SWEEP),
)
NAMES: tuple[str, ...] = tuple(row.name for row in ROWS)
_BY_NAME = {row.name: row for row in ROWS}
# What a row's sentence says that its line does not hold as written: a batch the engine counts from nought is
# said counted from one, as every other batch the window names is; the batches a grouped message stands for.
_SAID = {"patho_batch": lambda values: {"batch": values["k"] + 1},
         "chi_batch": lambda values: {"counted": f"{number_text(values['count'])} batch"
                                                f"{'' if values['count'] == 1 else 'es'}"}}


def read(text: str, level: str) -> Reading | None:
    """The reading of one line, or None when no row matches it whole. The line is read without the spaces around
    it; a whole number loses its commas, and a real number is read as one."""
    line = text.strip()
    for row in ROWS:
        found = row.pattern.fullmatch(line)
        if found is not None:
            return Reading(row.name, {name: _value(name, said) for name, said in found.groupdict().items()}, level)
    return None


def sentence(reading: Reading, *, count: int = 1) -> str | None:
    """The window's own sentence for a reading, its placeholders filled in (a number as the window writes one), or
    None when its row has none. ``count`` is how many such lines a grouped message stands for."""
    row = _BY_NAME[reading.name]
    if row.sentence is None:
        return None
    values = {**reading.values, "count": count}
    values.update(_SAID.get(row.name, lambda _: {})(values))
    return _PLACEHOLDER.sub(lambda found: _said(values[found.group(1)]), row.sentence)


def _value(name: str, said: str | None):
    if said is None:
        return None
    if name in _WHOLES:
        return int(said.replace(",", ""))
    if name in _REALS:
        return float(said.replace(",", ""))
    return said


def _said(value) -> str:
    return number_text(value) if isinstance(value, int) and not isinstance(value, bool) else str(value)
