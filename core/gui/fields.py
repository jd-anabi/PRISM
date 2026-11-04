"""The window's half of the refusal-naming rule: which control answers each refusal field.

A ``Refusal`` names the setting at fault by a key from ``core.refusals.FIELDS`` and never a box,
tab, button or flag. ``BasePanel._refusal`` puts ``fix_sentence(exc.field)`` under the message in
the yellow "Check your inputs" box, and the inference tabs build their static rows from
``label(key)`` -- ``add_help_row(form, label("t_obs"), ...)`` -- so a control is named in ONE place
and a renamed one cannot leave a stale sentence behind. ``add_help_row`` applies
``labels.pretty_gui`` to whatever it is given (``core/gui/widgets/help_badge.py``), so the strings
here are the plain forms ("T_obs (s)"), and the read-back pin compares ``pretty_gui(label(key))``
against the QLabels shown at the place a tuple names.

Three shapes of entry:

* ``(place, label)`` for a box or a picker: ``place`` is a place of the window by its key in
  ``core/gui/keys.py`` -- a step ("infer"), a part that has neither steps nor pages ("live") or a
  page of a part ("temperature/sweep") -- and the label is the row as the panel passes it to
  ``add_help_row``. A tuple of places is how one input names every place it appears, in the window's
  order (the parts as the rail lists them, a part's steps or pages as it shows them): the cell picker
  is in five of them.
* one sentence, for a consent, a dialog, a table, the name box, a value fixed by measurement, a
  control in Records, whose list and Note box are not form rows, the model builder's forcing
  fields, whose rows are labelled by parameter name, or a plan, whose file is what is corrected; the
  sentence quotes the control's own text ("Run on a different observation") and names its places as
  a box's sentence does.
* ``None`` for a key the window has no control for: the tool-only diagnostics knobs, the three
  settings the window never exposes (the device and the two sample counts), and the five FDT
  settings neither front end exposes.

Four settings stand only on the Training page, for its own run: the checkpoint cadence, the resume
policy, the epoch limit and the expected epochs. A page of the old kind, whose run reads none of them
from a box, names no box for them (``ON_TRAINING_ALONE``).

A saved simulation set that a training would not continue is told by what it differs in, each entry of its identity
by its name on screen (``IDENTITY_ROWS``, ``difference_lines``): a list compared entry by entry, each limit by its
parameter's name, every value whole.

A place is written in the words the rail's places are written in (``keys.place_words``), and several
are joined as ``keys.places_words`` joins them, so a place is named in one table and its joining is
written in one place. ``num_runs`` and ``run_size_cap`` sit in both training and the narrowing round
with the same label, so their place is a tuple of both and the sentence names both: one place would
send a user refused in the narrowing round to training. The narrowing round's row of the second reads
"(0 = automatic)" after the label, where the Training page has its tick "Automatic". The three drive keys
are static strings built exactly as the Infer tab builds its rows (``labels.gui_forcing_label``
with ``config.FORCING_DISPLAY_UNITS``); ``_rebuild_forcing_fields`` keeps deriving its rows from
the config's forcing names, so a model with an ``amp_y`` or ``offset`` drive shows rows this table
does not name. ``tests/test_nav_and_gating.py`` pins the key set against the registry both ways,
the places against the built window, and the sentences verbatim.
"""
from core import config
from core.Helpers import labels
from core.refusals import FIELDS

from . import keys

_FIXED = "Fixed by measurement: it cannot be changed in the window."


def _drive(name: str) -> str:
    """The Infer tab's own row label for a forcing parameter (infer_tab._rebuild_forcing_fields)."""
    return labels.gui_forcing_label(name, config.FORCING_DISPLAY_UNITS[name])


def _in(place) -> str:
    """Where a control is, as a sentence says it: ``"in step 5, Inference"``, and for a tuple its places in
    their turn as ``keys.places_words`` joins them: ``"in step 3, Training, or step 6, Narrowing round"``,
    ``"in step 5, Inference, or in Live simulation"``. KeyError for a place the window does not have."""
    return "in " + keys.places_words(place if isinstance(place, tuple) else (place,))


CONTROL: dict[str, tuple[str | tuple[str, ...], str] | str | None] = {
    # the observation and the training budget
    "t_obs": (("infer", "live"), "T_obs (s)"),   # Live simulation's box too
    "num_runs": (("training", "narrowing"), "Batches"),
    "run_size_cap": (("training", "narrowing"), "Simulations per batch"),
    "checkpoint_every": ("training", "Save the simulations every … batches"),
    "resume": ("training", "Continue or start new"),
    "new_run": (f"Answer 'Start training on a new simulation set' in the question 'Start training' asks "
                f"{_in('training')}, or tick 'Start a new simulation even if a cache one setting away exists' "
                f"{_in('narrowing')}."),
    # the kept training to resume: no control yet, its Resume comes with the window's training page
    "resume_fit": None,
    # the narrowing round
    "n_directions": ("narrowing", "Directions truncated"),
    "hpd_level": ("narrowing", "HPD level"),
    "observation": ("narrowing", "Observation"),
    # calibration and inference sizes
    "n_cal": ("checks", "Calibration datasets"),
    "cal_n_scales": ("checks", "(t_scale, T_obs) operating points"),
    "num_posterior_samples": None,
    "n_samples": None,
    # the prior sweep
    "num_iterations": ("prior", "Sweep rounds"),
    "sweep_batch": ("prior", "Candidates per round"),
    "max_sets": ("prior", "Stable sets to collect"),
    "walk_step": ("prior", "Random-walk step"),
    "stability_units": ("prior", "Stability test length"),
    "min_cluster_size": ("prior", "Smallest cluster"),
    "min_samples": ("prior", "Cluster density"),
    # the flow and the Fisher rotation
    "hidden_features": ("training", "Network width"),
    "num_transforms": ("training", "Network depth"),
    "learning_rate": ("training", "Learning rate"),
    "stop_after_epochs": ("training", "End the fit after … epochs without improvement"),
    "max_num_epochs": ("training", "Epoch limit"),
    "fisher_m": ("training", "Rotation: simulations per perturbation"),
    "fisher_dz": ("training", "Rotation: step"),
    "fisher_points": ("training", "Rotation: parameter sets averaged"),
    # consents and names
    "accept_truncated": f"Confirm the load in the dialog {_in('training')}.",
    "accept_other_observation": f"Tick 'Run on a different observation' {_in('infer')}.",
    # A name and a note are typed in several boxes, and each sentence lists them all: 'Name' on step 2 names the
    # prior a build makes, 'Name' on step 3 the posterior a training makes, and a run's record and a comparison's
    # are named in boxes of their own on the two pages of Effective temperature, so a taken comparison name is sent
    # to the 'Comparison name' box, not the run's.
    "name": (f"Choose another name in 'Name' {_in(('prior', 'training'))}, or in the 'Record name' or 'Comparison "
             f"name' box {_in(('temperature/one_cell', 'temperature/sweep'))}."),
    # Records. Sentences, not (place, label), although Records is a place: the list is not a box, and
    # the Note box sits in its action row, not in a form row the read-back pin reads.
    "artifact": f"Select an artifact in the list {_in('records')}.",
    "note": (f"Edit it in the Note box {_in('records')}, or in the 'Note' or 'Comparison note' "
             f"box {_in(('temperature/one_cell', 'temperature/sweep'))}."),
    # A sentence and not a (place, label) pair: the comparison list is one control that appears on
    # both pages of Effective temperature, under a different row on each. Both rows are quoted, and the
    # verb is "Choose", not "Add": a mode's ceiling ("at most 2") and a run named twice are fixed by
    # REMOVING one.
    "compare_records": (f"Choose the records in the 'Runs to compare' list {_in('temperature/one_cell')}, "
                        f"or in the 'Sweeps to compare' list {_in('temperature/sweep')}."),
    # The two mode settings, each a box the comparison controls build: the renormalise mode's
    # constant on One cell, the sweeps mode's slice point on Sweep study.
    "prefactor": ("temperature/one_cell", "Normalisation constant"),
    "slice_at": ("temperature/sweep", "Slice at"),
    # inputs. The cell picker and the model combo appear in several places at once and each entry
    # names them ALL: a bad cell chosen for the effective temperature used to be answered with step 5
    # alone. ``units`` is NOT widened -- one units control exists in the whole application (the
    # choice of step 1); everywhere else the units file is resolved from the model and there is no
    # control to name.
    "bounds": ("prior", "Bounds file"),
    "cell": (("infer", "temperature/one_cell", "temperature/sweep", "live", "reduction"), "Cell"),
    "units": ("model", "Units"),
    "model": (("model", "temperature/one_cell", "live"), "Model"),
    "device": None,
    "prior": ("prior", "Saved prior"),
    # The Training page makes a posterior and lists none until it can load one: the cure is another training.
    "posterior": f"Train another posterior {_in('training')}.",
    # recordings and the drive
    "recording_spont": (f"Select the passive recording {_in('infer')} (the 'Spontaneous' box, or the "
                        "'Passive' box for a χ observation)."),
    "recording_forced": ("infer", "Forced"),
    "recording_probe": f"Pick the probe's recording in the χ probe table {_in('infer')}.",
    "drive_amplitude": ("infer", _drive("amp")),
    "drive_frequency": ("infer", _drive("freq")),
    "drive_phase": ("infer", _drive("phase")),
    "chi_f0_si": ("infer", "Drive F₀ (N)"),
    # the kind of observation, its driven recordings, and the rotation
    "chi_mode": ("model", "Kind of observation"),
    "chi_n_freqs": ("model", "Driven recordings per observation"),
    "chi_k_pad": ("model", "Slots for driven recordings"),
    "chi_max_cycles": ("model", "Lock-in limit"),
    "chi_f0": _FIXED,
    "chi_freq_bounds": _FIXED,
    "reparam_rotate": ("model", "Rotation"),
    # a run plan: the thing to correct is the file, so the sentence names no box
    "plan": "Correct the file, then choose the plan again.",
    "expected_epochs": ("training", "Expected epochs (for the time estimate only)"),
    # the two secondary analyses, one page each: the label is the row the panel builds, so label(key)
    # keeps a box and its hint sentence from drifting apart; the Seed rows too are built from
    # label("seed") on both panels.
    # The two labels are the words the rows SHOW: the raw forms "ensemble_M" and "freqs_per_batch"
    # rendered as "M_ensemble" and "freqs / batch", so the fix sentence named a box the page does not
    # show. pretty_gui still renders "M_ensemble" with its subscript; no QSettings key is a label
    # (each panel writes its own literal key names).
    "n_freqs": (("temperature/one_cell", "temperature/sweep"), "n_freqs"),
    "ensemble_m": (("temperature/one_cell", "temperature/sweep"), "M_ensemble"),
    "freqs_per_batch": (("temperature/one_cell", "temperature/sweep"), "freqs / batch"),
    "f0": (("temperature/one_cell", "temperature/sweep", "reduction"), "F0 (ND forcing amplitude)"),
    "preset": ("temperature/sweep", "Preset"),
    "s_grid": ("temperature/sweep", "S grid  (T_a/T = 1)"),
    "t_grid": ("temperature/sweep", "T_a/T grid  (S = 0)"),
    "seed": (("temperature/one_cell", "temperature/sweep"), "Seed"),
    # the five FDT settings NEITHER front end exposes: no control, so fix_sentence returns "" and
    # the refusal names the setting and offers no fix
    "freq_bounds": None, "burn_in_nd": None, "t_obs_periods": None, "dt_nd": None, "psd_t_obs_nd": None,
    # the model builder, in Your own models. The parameter rows repeat per parameter, so the label is
    # the row's own ("value" / "min" / "max") and the message names which parameter.
    "param_value": ("models", "value"),
    "param_min": ("models", "min"),
    "param_max": ("models", "max"),
    "init": ("models", "init"),
    "x_scale": ("models", "x_scale (nm)"),
    "t_scale": ("models", "t_scale (s)"),
    # ONE key for every forcing field: the forcing rows' labels are the parameter NAMES (amp, freq,
    # tau, ...), built per kind, so there is no one label(key) to build a row from.
    "forcing_value": ("Set it in the forcing parameter's own box, beneath the variable's 'forcing' "
                      f"choice, {_in('models')}."),
    # the live simulation
    "frame_steps": ("live", "Steps / frame"),
    "fps": ("live", "Max FPS"),
    # tool-only: the diagnostics' knobs; no window sentence
    "repeats": None, "n_points": None, "n_worst": None, "top_n": None, "m": None, "m_noise": None,
    "rel": None, "min_valid": None, "rows": None, "n_sweep": None, "chi_k_fixed": None,
    # tool-only: the probe checks run from the command line alone
    "probe_seed": None, "probe_lengths": None, "probe_multipliers": None, "probe_drives": None,
    "band_repeats": None, "probe_cycle_caps": None, "cv_max": None, "phase_max": None, "snr_min": None,
    "sup_min": None, "band_peak_window": None, "mask_num_runs": None, "mask_run_size": None,
    "drive_t_obs": None, "drive_repeats": None, "drive_detune": None, "drive_strengths": None,
    "free_min": None, "captured_max": None, "drive_peak_window": None, "clarity_min": None,
}


#: The settings that stand on the Training page alone, for its own run: a page of the old kind names no box for them,
#: since its run reads none of them from one.
ON_TRAINING_ALONE = frozenset({"checkpoint_every", "resume", "max_num_epochs", "expected_epochs"})

#: What a page of the look says where the table's own sentence lists every old page's box.
ON_THIS_PAGE: dict[str, str] = {
    "new_run": "Answer the question that the start button asks on this page.",
    "name": "Choose another name in 'Name' on this page.",
    "note": "Change it in 'Note' on this page.",
}


#: A saved simulation set's identity as a person reads it, row by row: the entries a row folds into its one line, what
#: the line is called, and the step where a person changes them ("" where no page of the window does). ``model`` stands
#: in no row: a set of another model is that model's own, and is not listed.
IDENTITY_ROWS: tuple[tuple[tuple[str, ...], str, str], ...] = (
    (("n_runs",), "Batches", "training"),
    (("run_size",), "Simulations per batch", "training"),
    (("prior_fingerprint",), "Prior", "prior"),
    (("param_keys", "nd_lows", "nd_highs", "rescale_lows", "rescale_highs"), "Bounds file", "prior"),
    (("spontaneous_only",), "Bounds file: forcing section", "prior"),
    (("chi_mode", "mode"), "Kind of observation", "model"),
    (("chi_k_pad",), "Slots for driven recordings", "model"),
    (("chi_max_cycles",), "Lock-in limit", "model"),
    (("reparam_rotate",), "Rotation", "model"),
    (("units_sha256", "dt_exp", "t_min_exp", "t_max_exp"), "Units", "model"),
    (("t_scale_bounds", "dt_nd_min", "n_grid", "steady_idx"), "Time grid", "prior"),
    (("log_params",), "Parameters on a logarithmic range", ""),
    (("chi_f0",), "Drive amplitude in training", ""),
    (("chi_freq_bounds",), "Frequency band in training", ""),
    (("truncation",), "Made by a narrowing round: a training never continues it.", ""),
    (("device", "dtype"), "This computer", ""),
    (("format", "summary_flags", "feature_set_version", "chi_layout", "chi_elem_w"), "Made by another version of PRISM",
     ""),
)
#: How a kind of observation reads in a difference.
_KINDS = {"chi": "a frequency response", "forced": "a spontaneous recording and one driven recording",
          "spontaneous": "a spontaneous recording only"}
#: How each entry of the time grid reads on each side.
_GRID = {"t_scale_bounds": "t_scale from {}", "dt_nd_min": "smallest step {}", "n_grid": "{} points",
         "steady_idx": "steady part from point {}"}
DIFFERS_TEXT = "{}: {} here, {} in this set"
NO_DIFFERENCE_TEXT = "No setting differs: its folder was renamed by hand, so a training does not find it."
NO_PRIOR_TEXT = "a prior no longer in Records"
UNITS_TEXT = "Units: the units in use here are not those of this set"
OTHER_VERSION_TEXT = "Made by another version of PRISM"
MADE_ON_TEXT = "This computer: made on {}"
PRECISION_TEXT = "This computer: made with numbers of another precision ({})"


def _said(value) -> str:
    """A value of an identity as a line says it, whole: a whole number grouped, a fraction as briefly as it can be
    written exactly enough, a list entry by entry, nothing as "none"."""
    if value is None:
        return "none"
    if isinstance(value, bool):
        return "on" if value else "off"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        return f"{value:g}"
    if isinstance(value, (list, tuple)):
        return ", ".join(_said(one) for one in value) or "none"
    return str(value)


def _span(value) -> str:
    """A pair of limits as "0.1 to 1"; anything else as ``_said`` says it."""
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return f"{_said(value[0])} to {_said(value[1])}"
    return _said(value)


def _limits(differs: dict, param_keys) -> list:
    """The limits a set differs in, one line for each parameter whose limit differs, named from ``param_keys``, the
    parameters of this training, the model's first and the scales after them: both limits when both differ, else the
    one that does. A limit with no name to give is named by its place."""
    keys, lines = list(param_keys or ()), []
    for low, high in (("nd_lows", "nd_highs"), ("rescale_lows", "rescale_highs")):
        ends = {name: differs[name] for name in (low, high) if name in differs}
        if not ends:
            continue
        count = max(len(end.get(side) or ()) for end in ends.values() for side in ("mine", "theirs"))
        start = 0 if low == "nd_lows" else max(len(keys) - count, 0)
        for at in range(count):
            said = {}
            for name, end in ends.items():
                mine, theirs = (end.get(side) or () for side in ("mine", "theirs"))
                one, two = (values[at] if at < len(values) else None for values in (mine, theirs))
                if one != two:
                    said[name] = (one, two)
            if not said:
                continue
            index = start + at
            called = keys[index] if index < len(keys) else f"parameter {index + 1}"
            if len(said) == 2:
                lines.append(DIFFERS_TEXT.format(f"Bounds file: limits of {called}",
                                                 *(f"{_said(said[low][side])} to {_said(said[high][side])}"
                                                   for side in (0, 1))))
            else:
                (name, (one, two)), = said.items()
                end = "lower" if name == low else "upper"
                lines.append(DIFFERS_TEXT.format(f"Bounds file: {end} limit of {called}", _said(one), _said(two)))
    return lines


def difference_lines(fields, *, prior_label, param_keys=None) -> tuple:
    """What a saved simulation set differs in, one line for each row of ``IDENTITY_ROWS`` it differs in, in the rows'
    order, then any entry no row names, by its own name: ``fields`` holds one ``{"field", "mine", "theirs"}`` for each
    entry of the identity that differs, as the engine lists them. A list is compared entry by entry, and only the
    entries that differ are said; a limit is named by its parameter, from the parameter names this training holds,
    ``param_keys`` (a limit is named by its place when they are not handed); a prior by its label, ``prior_label`` of
    its fingerprint, or as no longer in Records when that answers nothing. Nothing is cut short. A set that differs in
    nothing is a folder renamed by hand, and is said so. ``model`` gives no line."""
    differs = {}
    for one in fields or ():
        if isinstance(one, dict) and isinstance(one.get("field"), str):
            differs[one["field"]] = one
    if not differs:
        return (NO_DIFFERENCE_TEXT,)

    def both(key):
        return differs[key].get("mine"), differs[key].get("theirs")

    lines, named = [], {"model"}
    for keys_, shown, _place in IDENTITY_ROWS:
        named |= set(keys_)
        here = [key for key in keys_ if key in differs]
        if not here:
            continue
        if keys_[0] in ("n_runs", "run_size", "chi_k_pad", "chi_max_cycles", "reparam_rotate", "chi_f0", "log_params"):
            lines.append(DIFFERS_TEXT.format(shown, *(_said(value) for value in both(keys_[0]))))
        elif keys_[0] == "prior_fingerprint":
            lines.append(DIFFERS_TEXT.format(shown, *((prior_label(value) or NO_PRIOR_TEXT) for value in
                                                      both("prior_fingerprint"))))
        elif keys_[0] == "param_keys":
            if "param_keys" in differs:
                lines.append(DIFFERS_TEXT.format("Bounds file: parameters", *(_said(list(value or ())) for value in
                                                                                both("param_keys"))))
            else:
                lines += _limits(differs, param_keys)
        elif keys_[0] == "spontaneous_only":
            lines.append(DIFFERS_TEXT.format(shown, *("none" if value else "one"
                                                     for value in both("spontaneous_only"))))
        elif keys_[0] == "chi_mode":
            if "mode" in differs:
                words = [_KINDS.get(value, _said(value)) for value in both("mode")]
            else:
                words = ["a frequency response" if value else "not a frequency response" for value in both("chi_mode")]
            lines.append(DIFFERS_TEXT.format(shown, *words))
        elif keys_[0] == "units_sha256":
            lines.append(UNITS_TEXT)
        elif keys_[0] == "t_scale_bounds":
            lines.append(f"{shown}: " + "; ".join(
                f"{_GRID[key].format(_span(differs[key].get('mine')))} here, "
                f"{_GRID[key].format(_span(differs[key].get('theirs')))} in this set"
                for key in keys_ if key in differs))
        elif keys_[0] == "chi_freq_bounds":
            lines.append(DIFFERS_TEXT.format(shown, *(_span(value) for value in both("chi_freq_bounds"))))
        elif keys_[0] == "truncation":
            lines.append(shown)
        elif keys_[0] == "device":
            if "device" in differs:
                theirs = differs["device"].get("theirs")
                lines.append(MADE_ON_TEXT.format("the graphics card" if theirs == "cuda" else "the processor"))
            else:
                lines.append(PRECISION_TEXT.format(_said(differs["dtype"].get("theirs"))))
        else:
            lines.append(OTHER_VERSION_TEXT)
    for key in sorted(set(differs) - named):
        lines.append(DIFFERS_TEXT.format(key, *(_said(value) for value in both(key))))
    return tuple(lines)


def label(key: str) -> str:
    """The row label of a box entry; the inference tabs build their static rows from it.

    KeyError for a sentence or None entry (there is no single box to label) and for a key the
    registry does not know -- loud on purpose: this runs when a tab is BUILT, so a wrong key fails
    at construction and never inside a refusal."""
    entry = CONTROL.get(key)
    if isinstance(entry, tuple):
        return entry[1]
    if key not in FIELDS:
        raise KeyError(f"{key!r} is not a registered refusal field (core.refusals.FIELDS)")
    raise KeyError(f"{key!r} has no box: its window control is {entry!r}")


def fix_sentence(key: str | None, here: str | None = None) -> str:
    """What the yellow box says under the message: ``"Set it in the 'T_obs (s)' box in step 5, Inference,
    or in Live simulation."`` for a box entry, the sentence itself for a sentence entry, ``""`` for
    ``field=None``, a None entry or an unknown key. It runs while a refusal is being shown, so it never
    raises for a key of the table: every place the table names is a place of the window, and the pin
    renders every entry.

    ``here`` is the place a page of the look stands on, a step or a part, whose rows are named without the word for a
    box: ``"Set it in 'Batches' on this page."`` when the row stands there, and else
    ``"Set it in 'Batches' in step 3, Training, or step 6, Narrowing round."``; a sentence entry says what
    ``ON_THIS_PAGE`` gives for it, when it gives anything. With no ``here`` a page not rebuilt yet keeps its own
    words, and names no box of ``ON_TRAINING_ALONE``."""
    if key is None or (here is None and key in ON_TRAINING_ALONE):
        return ""
    entry = CONTROL.get(key)
    if isinstance(entry, tuple):
        place, text = entry
        if here is None:
            return f"Set it in the '{text}' box {_in(place)}."
        return f"Set it in '{text}' " + ("on this page." if here in (place if isinstance(place, tuple) else (place,))
                                         else f"{_in(place)}.")
    if here is not None and key in ON_THIS_PAGE:
        return ON_THIS_PAGE[key]
    return entry or ""
