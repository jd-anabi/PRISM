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
  control in Records, whose list and Note box are not form rows, or the model builder's forcing
  fields, whose rows are labelled by parameter name; the sentence quotes the control's own text
  ("Run on a different observation") and names its places as a box's sentence does.
* ``None`` for a key the window has no control for: the tool-only diagnostics knobs, and the six
  settings the window never exposes (the checkpoint cadence, the resume policy, the device, the
  two sample counts, the epoch ceiling), and the five FDT settings neither front end exposes.

A place is written in the words the rail's places are written in (``keys.place_words``), and several
are joined as ``keys.places_words`` joins them, so a place is named in one table and its joining is
written in one place. ``num_runs`` and ``run_size_cap`` sit in both training and the narrowing round
with the same label, so their place is a tuple of both and the sentence names both: one place would
send a user refused in the narrowing round to training. The three drive keys
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

_FIXED = "Fixed by measurement: change it in config.py, deliberately."


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
    "run_size_cap": (("training", "narrowing"), "Max rows per batch (0 = auto)"),
    "checkpoint_every": None,
    "resume": None,
    "new_run": (f"Answer 'Start a new run anyway' in the dialog {_in('training')}, or tick 'Start a "
                f"new simulation even if a cache one setting away exists' {_in('narrowing')}."),
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
    "num_iterations": ("prior", "Global rounds"),
    "sweep_batch": ("prior", "Candidates per round (0 = auto)"),
    "max_sets": ("prior", "Max accepted sets"),
    "walk_step": ("prior", "Random-walk step"),
    "stability_units": ("prior", "Stability duration (ND units)"),
    "min_cluster_size": ("prior", "Min cluster size"),
    "min_samples": ("prior", "Min samples"),
    # the flow and the Fisher rotation
    "hidden_features": ("training", "Hidden features"),
    "num_transforms": ("training", "Transforms"),
    "learning_rate": ("training", "Learning rate"),
    "stop_after_epochs": ("training", "Early-stop patience"),
    "max_num_epochs": None,
    "fisher_m": ("training", "Ensemble per perturbation"),
    "fisher_dz": ("training", "Central-difference step"),
    "fisher_points": ("training", "Operating points"),
    # consents and names
    "accept_truncated": f"Confirm the load in the dialog {_in('training')}.",
    "accept_other_observation": f"Tick 'Run on a different observation' {_in('infer')}.",
    # A name and a note are typed in several boxes, and each sentence lists them all: the Save box
    # beside the Save button of steps 2 and 3 names the prior or the posterior held, and a run's record
    # and a comparison's are named in boxes of their own on the same two pages, so a taken comparison
    # name is sent to the 'Comparison name' box, not the run's.
    "name": (f"Choose another name in the Save box {_in(('prior', 'training'))}, or in the 'Record name' or "
             f"'Comparison name' box {_in(('temperature/one_cell', 'temperature/sweep'))}."),
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
    # toggle of step 1); everywhere else the units file is resolved from the model and there is no
    # control to name.
    "bounds": ("prior", "Bounds"),
    "cell": (("infer", "temperature/one_cell", "temperature/sweep", "live", "reduction"), "Cell"),
    "units": ("model", "Units"),
    "model": (("model", "temperature/one_cell", "live"), "Model"),
    "device": None,
    "prior": ("prior", "Prior"),
    "posterior": ("training", "Posterior"),
    # recordings and the drive
    "recording_spont": (f"Select the passive recording {_in('infer')} (the 'Spontaneous' box, or the "
                        "'Passive' box for a χ observation)."),
    "recording_forced": ("infer", "Forced"),
    "recording_probe": f"Pick the probe's recording in the χ probe table {_in('infer')}.",
    "drive_amplitude": ("infer", _drive("amp")),
    "drive_frequency": ("infer", _drive("freq")),
    "drive_phase": ("infer", _drive("phase")),
    "chi_f0_si": ("infer", "Drive F₀ (N)"),
    # chi conditioning
    "chi_n_freqs": ("model", "χ probes per observation"),
    "chi_k_pad": ("model", "χ probe slots (capacity)"),
    "chi_max_cycles": ("model", "χ lock-in ceiling (cycles)"),
    "chi_f0": _FIXED,
    "chi_freq_bounds": _FIXED,
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


def fix_sentence(key: str | None) -> str:
    """What the yellow box says under the message: ``"Set it in the 'T_obs (s)' box in step 5, Inference,
    or in Live simulation."`` for a box entry, the sentence itself for a sentence entry, ``""`` for
    ``field=None``, a None entry or an unknown key. It runs while a refusal is being shown, so it never
    raises for a key of the table: every place the table names is a place of the window, and the pin
    renders every entry."""
    if key is None:
        return ""
    entry = CONTROL.get(key)
    if isinstance(entry, tuple):
        place, text = entry
        return f"Set it in the '{text}' box {_in(place)}."
    return entry or ""
