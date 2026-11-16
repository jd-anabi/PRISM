"""The window's half of the refusal-naming rule: which control answers each refusal field.

A ``Refusal`` names the setting at fault by a key from ``core.refusals.FIELDS`` and never a box,
tab, button or flag. ``BasePanel._refusal`` puts ``fix_sentence(exc.field)`` under the message in
the yellow "Check your inputs" box, and the inference tabs build their static rows from
``label(key)`` -- ``add_help_row(form, label("t_obs"), ...)`` -- so a control is named in ONE place
and a renamed one cannot leave a stale sentence behind. ``add_help_row`` and a form's row apply
``labels.pretty_gui`` to whatever they are given (``core/gui/widgets/help_badge.py``), so the strings
here are the plain forms, a unit standing beside its field and never in its label, and the read-back
pin compares ``pretty_gui(label(key))`` against the QLabels shown at the place a tuple names.

Three shapes of entry:

* ``(place, label)`` for a box or a picker: ``place`` is a place of the window by its key in
  ``core/gui/keys.py`` -- a step ("infer") or a part that has no steps ("models") -- and the label is
  the row as the panel passes it to ``add_help_row``. A tuple of places is how one input names every
  place it appears, in the window's order (the parts as the rail lists them, a part's steps as it
  shows them): the batch count is in three of them.
* one sentence, for a consent, a dialog, a table, the name box, a value fixed by measurement, a
  control in Records, whose list and Note box are not form rows, the model builder's forcing
  fields, whose rows are labelled by parameter name, or a plan, whose file is what is corrected; the
  sentence quotes the control's own text ("Load this posterior") and names its places as a box's
  sentence does.
* ``None`` for a key the window has no control for: the device, which the window never exposes, and the
  settings of the checks that no check's form offers.

Two settings stand only on the Training page, for its own run: the resume policy and the expected epochs; with no
step named, no box is named for them (``ON_TRAINING_ALONE``).

A saved simulation set that a training would not continue is told by what it differs in, each entry of its identity
by its name on screen (``IDENTITY_ROWS``, ``difference_lines``): a list compared entry by entry, each limit by its
parameter's name, every value whole.

A place is written in the words the rail's places are written in (``keys.place_words``), and several
are joined as ``keys.places_words`` joins them, so a place is named in one table and its joining is
written in one place. ``num_runs`` and ``run_size_cap`` sit in training, in the narrowing round and on
a check's form with the same label, so their place is a tuple of every one and the sentence names each:
one place would send a user refused in the narrowing round to training. The three drive keys
are static strings built exactly as the Inference page builds its drive's rows (``labels.gui_forcing_label``
with ``config.FORCING_DISPLAY_UNITS``); the page makes a row for each forcing name of the configuration
in use, so a model with an ``amp_y`` or ``offset`` drive shows rows this table does not name.
``tests/test_nav_and_gating.py`` pins the key set against the registry both ways, the places against the
built window, and the sentences verbatim.
"""
from core import config
from core.Helpers import labels
from core.refusals import FIELDS

from . import keys
from .step_facts import NO_DIFFERENCE_TEXT, ROUND_START_WORDS

_FIXED = "Fixed by measurement: it cannot be changed in the window."


def _drive(name: str) -> str:
    """The Inference page's own row label for a forcing parameter of the drive."""
    return labels.gui_forcing_label(name, config.FORCING_DISPLAY_UNITS[name])


def _in(place) -> str:
    """Where a control is, as a sentence says it: ``"in step 5, Inference"``, and for a tuple its places in
    their turn as ``keys.places_words`` joins them: ``"in step 3, Training, or step 6, Narrowing round"``,
    ``"in step 5, Inference, or in Records"``. KeyError for a place the window does not have."""
    return "in " + keys.places_words(place if isinstance(place, tuple) else (place,))


CONTROL: dict[str, tuple[str | tuple[str, ...], str] | str | None] = {
    # the observation and the training budget
    "t_obs": (("checks", "infer"), "Recording length"),
    "num_runs": (("training", "checks", "narrowing"), "Batches"),
    "run_size_cap": (("training", "checks", "narrowing"), "Simulations per batch"),
    "checkpoint_every": (("training", "narrowing"), "Save the simulations every … batches"),
    "resume": ("training", "Continue or start new"),
    "new_run": (f"Answer 'Start training on a new simulation set' in the question 'Start training' asks "
                f"{_in('training')}, or 'Start the narrowing round' in the question that 'Start the narrowing round' "
                f"asks {_in('narrowing')}."),
    # the kept training to resume: the Training step lists every kept training with its Resume
    "resume_fit": f"Look at the kept trainings {_in('training')}.",
    # the narrowing round
    "n_directions": ("narrowing", "Directions narrowed"),
    "hpd_level": ("narrowing", "Region level"),
    "observation": ("narrowing", "Observation"),
    # calibration and inference sizes
    "n_cal": ("checks", "Test datasets"),
    "cal_n_scales": ("checks", "Operating points"),
    "num_posterior_samples": ("checks", "Samples per dataset"),
    "n_samples": ("infer", "Posterior draws"),
    # the prior sweep
    "num_iterations": ("prior", "Sweep rounds"),
    "sweep_batch": ("prior", "Candidates per round"),
    "max_sets": ("prior", "Stable sets to collect"),
    "walk_step": ("prior", "Random-walk step"),
    "stability_units": ("prior", "Stability test length"),
    "min_cluster_size": ("prior", "Smallest cluster"),
    "min_samples": ("prior", "Cluster density"),
    # the flow and the Fisher rotation
    "hidden_features": (("training", "checks", "narrowing"), "Network width"),
    "num_transforms": (("training", "checks", "narrowing"), "Network depth"),
    "learning_rate": (("training", "narrowing"), "Learning rate"),
    "stop_after_epochs": (("training", "narrowing"), "End the fit after … epochs without improvement"),
    "max_num_epochs": (("training", "narrowing"), "Epoch limit"),
    "fisher_m": ("training", "Rotation: simulations per perturbation"),
    "fisher_dz": ("training", "Rotation: step"),
    "fisher_points": ("training", "Rotation: parameter sets averaged"),
    # consents and names
    "accept_truncated": (f"Answer 'Load this posterior' in the question that loading a posterior from a narrowing "
                         f"round asks {_in('training')}."),
    "accept_other_observation": f"Answer the question that 'Run inference' asks {_in('infer')}.",
    # A name is typed in three boxes, and the sentence lists them: 'Name' on step 2 names the prior a build makes,
    # 'Name' on step 3 the posterior a training makes, and 'Name of the inference' on step 5 the inference.
    "name": (f"Choose another name in 'Name' {_in(('prior', 'training'))}, or in 'Name of the inference' "
             f"{_in('infer')}."),
    # the observation an inference makes, named beside the inference
    "observation_name": ("infer", "Name of the observation"),
    # Records. Sentences, not (place, label), although Records is a place: the list is not a box, and
    # the Note box sits in its action row, not in a form row the read-back pin reads.
    "artifact": f"Select an artifact in the list {_in('records')}.",
    "note": f"Change it in 'Note' {_in('infer')}, or in the Note box {_in('records')}.",
    # inputs. ``units`` is NOT widened -- one units control exists in the whole application (the choice of step 1);
    # everywhere else the units file is resolved from the model and there is no control to name.
    "bounds": ("prior", "Bounds file"),
    "cell": (("checks", "infer"), "Cell file"),
    "units": ("model", "Units"),
    "model": ("model", "Model"),
    "device": None,
    "prior": ("prior", "Saved prior"),
    "posterior": ("training", "Saved posterior"),
    # recordings and the drive
    "recording_spont": ("infer", "Spontaneous recording"),
    "recording_forced": ("infer", "Driven recording"),
    "recording_probe": ("infer", "Driven recordings"),
    "drive_amplitude": ("infer", _drive("amp")),
    "drive_frequency": ("infer", _drive("freq")),
    "drive_phase": ("infer", _drive("phase")),
    "chi_f0_si": ("infer", "Drive amplitude"),
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
    # the random seed of a check's run; the probe checks refuse theirs under a key of their own, under the same label,
    # and no check's form holds both
    "seed": ("checks", "Seed"),
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
    # the checks' own settings, on each check's form
    "rows": ("checks", "Simulations read"),
    "n_sweep": ("checks", "Points per sweep"),
    "n_worst": ("checks", "Weakest directions totalled"),
    "top_n": ("checks", "Largest contributions shown"),
    "probe_seed": ("checks", "Seed"),
    "probe_lengths": ("checks", "Recording lengths"),
    "band_repeats": ("checks", "Simulations per point"),
    "mask_num_runs": ("checks", "Batches to audit"),
    "mask_run_size": ("checks", "Simulations per batch"),
    "m": ("checks", "Simulations per comparison"),
    "m_noise": ("checks", "Simulations for the noise level"),
    "rel": ("checks", "Step, as a share of each range"),
    "repeats": ("checks", "Repeats"),
    "chi_k_fixed": ("checks", "Fixed number of driven recordings"),
    "n_points": ("checks", "Points across the prior"),
    # the drive study's recording length reads as the recording length does, on a form that holds no other
    "drive_t_obs": ("checks", "Recording length"),
    "drive_repeats": ("checks", "Simulations per strength"),
    "drive_detune": ("checks", "Drive frequency, × the cell's own"),
    "drive_strengths": ("checks", "Strengths"),
    # the checks' settings no form offers: what decides a check's verdict
    "min_valid": None, "probe_multipliers": None, "probe_drives": None, "probe_cycle_caps": None, "cv_max": None,
    "phase_max": None, "snr_min": None, "sup_min": None, "band_peak_window": None, "free_min": None,
    "captured_max": None, "drive_peak_window": None, "clarity_min": None,
}


#: The settings that stand on the Training page alone, for its own run: with no step named, no box is named for them.
ON_TRAINING_ALONE = frozenset({"resume", "expected_epochs"})

#: What a page of the look says where the table's own sentence lists every old page's box.
ON_THIS_PAGE: dict[str, str] = {
    "new_run": "Answer the question that the start button asks on this page.",
    "name": "Choose another name in 'Name' on this page.",
    "accept_truncated": "Answer 'Load this posterior' in the question that loading a posterior from a narrowing round "
                        "asks on this page.",
    "note": "Change it in 'Note' on this page.",
    "accept_other_observation": "Answer the question that the start button asks on this page.",
    "resume_fit": "Look at the kept narrowing rounds on this page.",
}
#: The steps whose own question or box a sentence of ``ON_THIS_PAGE`` answers: on another page the table's sentence,
#: which names the step, is said. The name and the note stand on the five steps that make a record; the kept narrowing
#: rounds are listed on the Narrowing round step, the kept trainings on Training, whose sentence is the table's.
_RECORDED = ("prior", "training", "checks", "infer", "narrowing")
ASKED_ON = {"new_run": ("training", "narrowing"), "accept_truncated": ("training",),
            "accept_other_observation": ("infer",), "name": _RECORDED, "note": _RECORDED, "resume_fit": ("narrowing",)}
#: What a part says of a setting it holds itself, where the table's sentence lists every place that holds it.
ON_A_PART = {("note", "records"): f"Edit it in the Note box {_in('records')}."}
#: What a step says of a refusal of its own run under a setting the table names on another step's form, which its page
#: does not show: a narrowing round's "Continue from batch N" is refused under the policy that Training's "Continue or
#: start new" sets, and on the round's page a press of its main button, named in the words it reads, then starts a new
#: round.
ON_A_STEP = {("resume", "narrowing"): f"Press '{ROUND_START_WORDS}' on this page to start a new round from its first "
                                      "batch."}


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
NO_PRIOR_TEXT = "a prior no longer in Records"
UNITS_TEXT = "Units: the units in use here are not those of this set"
OTHER_VERSION_TEXT = "Made by another version of PRISM"
MADE_ON_TEXT = "This computer: made on {}"
PRECISION_TEXT = "This computer: made with numbers of another precision ({})"


def _said(value) -> str:
    """A value of an identity as a line says it, whole: a whole number grouped, a fraction in the shortest form that
    is exactly it (a whole one without its ".0"), a list entry by entry, nothing as "none"."""
    if value is None:
        return "none"
    if isinstance(value, bool):
        return "on" if value else "off"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        said = repr(value)
        return said[:-2] if said.endswith(".0") else said
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
    """What the yellow box says under the message: ``"Set it in the 'Bounds file' box in step 2, Prior."``
    for a box entry, the sentence itself for a sentence entry, ``""`` for
    ``field=None``, a None entry or an unknown key. It runs while a refusal is being shown, so it never
    raises for a key of the table: every place the table names is a place of the window, and the pin
    renders every entry.

    ``here`` is the place a page of the look stands on, a step or a part, whose rows are named without the word for a
    box: ``"Set it in 'Batches' on this page."`` when the row stands there, and else
    ``"Set it in 'Batches' in step 3, Training, or step 6, Narrowing round."``; a sentence entry says what
    ``ON_THIS_PAGE`` gives for it, when it gives anything and the page is one ``ASKED_ON`` names for it, else the
    table's own; a part says ``ON_A_PART``'s sentence of what it holds itself, and a step ``ON_A_STEP``'s of a refusal
    of its own run under a setting its page does not show. With no ``here`` the table's own words stand, and no box of
    ``ON_TRAINING_ALONE`` is named."""
    if key is None or (here is None and key in ON_TRAINING_ALONE):
        return ""
    if (key, here) in ON_A_PART:
        return ON_A_PART[(key, here)]
    if (key, here) in ON_A_STEP:
        return ON_A_STEP[(key, here)]
    entry = CONTROL.get(key)
    if isinstance(entry, tuple):
        place, text = entry
        if here is None:
            return f"Set it in the '{text}' box {_in(place)}."
        return f"Set it in '{text}' " + ("on this page." if here in (place if isinstance(place, tuple) else (place,))
                                         else f"{_in(place)}.")
    if here is not None and key in ON_THIS_PAGE and here in ASKED_ON.get(key, (here,)):
        return ON_THIS_PAGE[key]
    return entry or ""
