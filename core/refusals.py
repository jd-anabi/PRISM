"""Refusals: the one exception kind for "asked for, and will not be done", and the field rules that
raise it before anything is spent (piece 3 of the 2026-09-10 hardening programme, design §3.1).

WHY ONE KIND. The window flattened every exception to a red "Error" box with a traceback behind
Details, and the tool guessed from the type: any ValueError printed as a refusal. A refusal and a
crash are different things -- one is answered by changing an input, the other by fixing the code --
so ``Refusal`` is the type both front ends route to the yellow "Check your inputs" box or the
``refused:`` line, and everything else stays a bug with a traceback.

WHY A FIELD KEY. The message is neutral: it names the setting in plain words, the allowed values,
what was given and the default, and never a box, tab, flag or button (a Python keyword may be named:
it is the core API's own word). Each front end appends its own "how to fix here" from a table keyed
by ``field`` (core/gui/fields.py, core/tool/fields.py), so a renamed control is renamed in one place
and a core message can never go stale about it.

WHY BLANK IS A REFUSAL. ``FloatField.value()`` returned 0 on a blank box and the tabs clamped,
defaulted or forwarded the zero: a blank observation length reached ``math.log(0.0)`` AFTER the
simulation was spent, and the TSNPE tab's blank direction count was refused with a sentence naming
neither the box nor the default. Here ``None`` is the blank and is refused as one, with the default
in the sentence; the two "0 = automatic" fields accept a typed 0 only.

TORCH-FREE, deliberately: this module imports only the standard library, so the window can run the
rules on the GUI thread at the click, the tool can run them before ``--help`` would cost a torch
import, and ``core.config`` (which imports torch) is never pulled in. The defaults are therefore
literal strings, pinned against ``core.config``'s constants by tests/test_refusals.py.
"""
import math
import os
from dataclasses import dataclass
from typing import NoReturn


class Refusal(ValueError):
    """Something asked for that the program will not do.

    ``field`` names the setting at fault by a key every front end can map to its own control or flag
    (``FIELDS``); None for a refusal no single control answers. A ValueError, so the callers that
    still catch one keep working while the tree is converted; ``str(exc)`` is the message.
    """

    def __init__(self, message: str, *, field: str | None = None):
        super().__init__(message)
        self.message = message
        self.field = field


@dataclass(frozen=True)
class Field:
    """One setting the rules can refuse: its key, its neutral description (a lower-case noun phrase,
    spliced into "<What> must be ..." and "<What> is blank ...") and its default as the message shows
    it. ``default`` is None when the rule has no default: a consent, a file, a name, a fixed value."""
    key: str
    what: str
    default: str | None


# The registry: every key raised anywhere under core/ is here (Task 4's AST scan pins it), and both
# front-end tables map every key here. The words are the design's (§3.1); each default is the str()
# of the constant named beside it, so a message shows the default as the operator would type it.
FIELDS: dict[str, Field] = {f.key: f for f in (
    Field("t_obs", "the observation length, in seconds", "none: it must be given"),
    Field("num_runs", "the number of training batches", "5000"),                       # TRAINING_NUM_RUNS
    Field("run_size_cap", "the rows-per-batch cap (0 = automatic)", "0"),              # TRAINING_RUN_SIZE
    Field("n_directions", "the number of directions to truncate", "5"),                # truncate.DEFAULT_N_DIRECTIONS
    Field("hpd_level", "the HPD level", "0.999"),                                      # truncate.DEFAULT_HPD
    Field("n_cal", "the number of calibration datasets", "2000"),                      # SBC_N_CAL
    Field("cal_n_scales", "the number of calibration operating points", "200"),        # CAL_N_SCALES
    Field("num_posterior_samples", "the number of posterior samples per calibration dataset", "1000"),
    Field("n_samples", "the number of posterior samples", "1000"),                     # infer_and_visualize
    Field("num_iterations", "the number of prior-sweep rounds", "50"),                 # PRIOR_SWEEP_ITERATIONS
    Field("sweep_batch", "the candidates per sweep round (0 = automatic)", "0"),       # PRIOR_SWEEP_BATCH
    Field("max_sets", "the maximum number of accepted parameter sets", "175000"),      # PRIOR_SWEEP_MAX_SETS
    Field("walk_step", "the random-walk step", "0.01"),                                # PRIOR_SWEEP_STEP
    Field("stability_units", "the stability duration, in ND units", "1000"),           # STABILITY_SWEEP_ND_UNITS
    Field("min_cluster_size", "the minimum cluster size", "50"),                       # PRIOR_CLUSTER_MIN_SIZE
    Field("min_samples", "the minimum samples per cluster", "10"),                     # PRIOR_CLUSTER_MIN_SAMPLES
    Field("hidden_features", "the number of hidden features", "128"),                  # NSF_HIDDEN_FEATURES
    Field("num_transforms", "the number of flow transforms", "8"),                     # NSF_NUM_TRANSFORMS
    Field("learning_rate", "the learning rate", "0.001"),                              # TRAINING_LEARNING_RATE
    Field("stop_after_epochs", "the early-stop patience, in epochs", "20"),            # TRAINING_STOP_AFTER_EPOCHS
    Field("max_num_epochs", "the maximum number of epochs", "2147483647"),             # TRAINING_MAX_NUM_EPOCHS
    Field("fisher_m", "the Fisher ensemble size per perturbation", "48"),              # REPARAM_FISHER_M
    Field("fisher_dz", "the Fisher central-difference step", "0.1"),                   # REPARAM_FISHER_DZ
    Field("fisher_points", "the number of Fisher operating points", "8"),              # REPARAM_FISHER_POINTS
    Field("checkpoint_every", "the checkpoint cadence, in batches (0 = off)", "50"),   # TRAINING_CHECKPOINT_EVERY
    Field("resume", "the resume policy", "auto"),                                      # build_posterior
    Field("new_run", "consent to start a new simulation cache", None),
    Field("accept_truncated", "consent to load a non-amortized posterior", None),
    Field("accept_other_observation", "consent to infer on another observation", None),
    Field("name", "the artifact name", None),
    Field("bounds", "the bounds file", None),
    Field("cell", "the cell file", None),
    Field("units", "the units", None),
    Field("recording_spont", "the passive recording", None),
    Field("recording_forced", "the driven recording", None),
    Field("recording_probe", "the probe recording", None),
    Field("drive_amplitude", "the drive amplitude", None),
    Field("drive_frequency", "the drive frequency", None),
    Field("drive_phase", "the drive phase", None),
    Field("chi_f0_si", "the physical chi drive amplitude, in newtons", None),
    Field("chi_n_freqs", "the number of chi probe frequencies", "6"),                  # CHI_N_FREQS
    Field("chi_k_pad", "the number of chi probe slots", "12"),                         # CHI_K_PAD
    Field("chi_max_cycles", "the chi lock-in ceiling, in cycles", "20.0"),             # CHI_MAX_CYCLES
    Field("chi_f0", "the chi drive amplitude (fixed by measurement)", None),
    Field("chi_freq_bounds", "the chi frequency band (fixed by measurement)", None),
    Field("device", "the compute device", "auto"),                                     # --device
    Field("model", "the model", None),
    Field("observation", "the observation", None),
    Field("posterior", "the posterior", None),
    Field("prior", "the prior", None),
    # tool-only (the diagnostics); no window control, CONTROL[key] is None in core/gui/fields.py
    Field("repeats", "the number of SBC repeats", "10"),                               # sbc_repeats
    Field("n_points", "the number of operating points", "6"),                          # identifiability_laplace
    Field("n_worst", "the number of worst directions to report", "3"),                 # identifiability_rotation
    Field("top_n", "the number of top features per parameter", "4"),                   # identifiability_rotation
    Field("m", "the ensemble size", "32"),                                             # identifiability_laplace
    Field("m_noise", "the noise-floor ensemble size", "128"),                          # identifiability_laplace
    Field("rel", "the relative perturbation", "0.02"),                                 # identifiability_laplace
    Field("min_valid", "the minimum valid fraction", None),
    Field("rows", "the number of rows", "200000"),                                     # channel_ablation
    Field("n_sweep", "the number of sweep points", "33"),                              # channel_ablation
    Field("chi_k_fixed", "the fixed chi probe count", None),
)}


def describe(key: str) -> str:
    """The neutral noun phrase for a field key ("the observation length, in seconds"). KeyError on
    an unregistered key: a message can only be built for a field a front end can map to a control."""
    return FIELDS[key].what


def _what(key: str) -> str:
    """describe(key) as the start of a sentence."""
    what = describe(key)
    return what[0].upper() + what[1:]


def _default_clause(key: str) -> str:
    """" (default 5000)", or "" for a field with no default."""
    default = FIELDS[key].default
    return "" if default is None else f" (default {default})"


def refuse(key: str, message: str) -> NoReturn:
    """A Refusal for ``key`` with the field's default clause appended to the caller's own sentence,
    which carries its own period. For the refusals whose rule is not one of the ``require_*`` shapes
    below: a direction count above the latent width, a device that is not there."""
    raise Refusal(message + _default_clause(key), field=key)


def _blank(key: str) -> NoReturn:
    raise Refusal(f"{_what(key)} is blank{_default_clause(key)}.", field=key)


def require_given(key: str, value):
    """None (a blank box) is refused; anything else is returned as given, uncoerced and unjudged."""
    if value is None:
        _blank(key)
    return value


def require_finite(key: str, value) -> float:
    """A blank, NaN or an infinity is refused; the value is returned as a float. The one rule for a
    field that may be zero or negative (the drive phase)."""
    if value is None:
        _blank(key)
    v = float(value)
    if not math.isfinite(v):
        raise Refusal(f"{_what(key)} must be a finite number; got {value!r}{_default_clause(key)}.", field=key)
    return v


def require_positive(key: str, value) -> float:
    """A blank, a non-finite value or one at or below 0 is refused; the value is returned as a float.
    Formatted with :g, so 5000.0 reads as 5000 and NaN as nan."""
    if value is None:
        _blank(key)
    v = float(value)
    if not math.isfinite(v) or v <= 0:
        raise Refusal(f"{_what(key)} must be greater than 0; got {v:g}{_default_clause(key)}.", field=key)
    return v


def require_at_least(key: str, value, minimum: int) -> int:
    """A blank or a value below ``minimum`` is refused; the value is returned as an int. ``minimum``
    is 1 for a count, 0 for the two "0 = automatic" fields and the checkpoint cadence, 2 for the
    minimum cluster size. The rule text is "at least <minimum>"."""
    if value is None:
        _blank(key)
    v = int(value)
    if v < minimum:
        raise Refusal(f"{_what(key)} must be at least {minimum}; got {v}{_default_clause(key)}.", field=key)
    return v


def require_between(key: str, value, lo, hi, *, open_lo: bool = False, open_hi: bool = False) -> float:
    """A blank or a value outside [lo, hi] is refused (an open end excludes its bound; NaN fails
    every comparison and so is outside every interval); the value is returned as a float."""
    if value is None:
        _blank(key)
    v = float(value)
    inside = (lo < v if open_lo else lo <= v) and (v < hi if open_hi else v <= hi)
    if not inside:
        ends = " (exclusive)" if (open_lo or open_hi) else ""
        raise Refusal(f"{_what(key)} must be between {lo:g} and {hi:g}{ends}; got {v:g}"
                      f"{_default_clause(key)}.", field=key)
    return v


def require_choice(key: str, value, choices: tuple) -> str:
    """A blank or a value outside ``choices`` is refused, the choices listed in the caller's order;
    the value is returned as a str."""
    if value is None:
        _blank(key)
    v = str(value)
    if v not in choices:
        raise Refusal(f"{_what(key)} must be one of {', '.join(choices)}; got {value!r}"
                      f"{_default_clause(key)}.", field=key)
    return v


def require_file(key: str, path, what: str | None = None) -> str:
    """A blank path or one that names no file is refused; the path is returned as a str. ``what`` is
    the input kind the blank sentence names ("cell", "bounds", "passive recording"); without it the
    field's own description stands in. A directory is not the file."""
    if path is None or not os.fspath(path).strip():
        kind = what or describe(key)
        raise Refusal(f"{_what(key)} is blank: no {kind} file was given{_default_clause(key)}.", field=key)
    p = os.fspath(path)
    if not os.path.isfile(p):
        raise Refusal(f"{_what(key)} was not found: {p!r}{_default_clause(key)}.", field=key)
    return p
