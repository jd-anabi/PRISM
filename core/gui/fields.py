"""The window's half of V3 (piece 3 spec §3.2): which control answers each refusal field.

A ``Refusal`` names the setting at fault by a key from ``core.refusals.FIELDS`` and never a box,
tab, button or flag. ``BasePanel._refusal`` puts ``fix_sentence(exc.field)`` under the message in
the yellow "Check your inputs" box, and the inference tabs build their static rows from
``label(key)`` -- ``add_help_row(form, label("t_obs"), ...)`` -- so a control is named in ONE place
and a renamed one cannot leave a stale sentence behind. ``add_help_row`` applies
``labels.pretty_gui`` to whatever it is given (``core/gui/widgets/help_badge.py``), so the strings
here are the plain forms ("T_obs (s)"), and the read-back pin compares ``pretty_gui(label(key))``
against the QLabel on the tab a tuple names.

Three shapes of entry:

* ``(tab, label)`` for a box or a picker: the tab title exactly as ``InferenceScreen`` shows it
  (Config, Prior, Posterior, Validate, Infer, TSNPE) and the row label as the tab passes it to
  ``add_help_row``.
* one sentence, for a consent, a dialog, a table, the name box, or a value fixed by measurement;
  the sentence quotes the control's own text ("Run on a different observation").
* ``None`` for a key the window has no control for: the tool-only diagnostics knobs, and the six
  settings the window never exposes (the checkpoint cadence, the resume policy, the device, the
  two sample counts, the epoch ceiling).

``num_runs`` and ``run_size_cap`` sit on both the Posterior and TSNPE tabs with the same label, so
their tab slot is a tuple of both and the sentence names both: one tab would send a user refused on
the TSNPE tab to the Posterior tab. The three drive keys
are static strings built exactly as the Infer tab builds its rows (``labels.gui_forcing_label``
with ``config.FORCING_DISPLAY_UNITS``); ``_rebuild_forcing_fields`` keeps deriving its rows from
the config's forcing names, so a model with an ``amp_y`` or ``offset`` drive shows rows this table
does not name. ``tests/test_nav_and_gating.py`` pins the key set against the registry both ways,
the tab names against the built screen, and the sentences verbatim.
"""
from core import config
from core.Helpers import labels
from core.refusals import FIELDS

_FIXED = "Fixed by measurement: change it in config.py, deliberately."


def _drive(name: str) -> str:
    """The Infer tab's own row label for a forcing parameter (infer_tab._rebuild_forcing_fields)."""
    return labels.gui_forcing_label(name, config.FORCING_DISPLAY_UNITS[name])


CONTROL: dict[str, tuple[str | tuple[str, ...], str] | str | None] = {
    # the observation and the training budget
    "t_obs": ("Infer", "T_obs (s)"),
    "num_runs": (("Posterior", "TSNPE"), "Batches"),
    "run_size_cap": (("Posterior", "TSNPE"), "Max rows per batch (0 = auto)"),
    "checkpoint_every": None,
    "resume": None,
    "new_run": ("Answer 'Start a new run anyway' in the dialog on the Posterior tab, or tick 'Start a "
                "new simulation even if a cache one setting away exists' on the TSNPE tab."),
    # TSNPE
    "n_directions": ("TSNPE", "Directions truncated"),
    "hpd_level": ("TSNPE", "HPD level"),
    "observation": ("TSNPE", "Observation"),
    # calibration and inference sizes
    "n_cal": ("Validate", "Calibration datasets"),
    "cal_n_scales": ("Validate", "(t_scale, T) operating points"),
    "num_posterior_samples": None,
    "n_samples": None,
    # the prior sweep
    "num_iterations": ("Prior", "Global rounds"),
    "sweep_batch": ("Prior", "Candidates per round (0 = auto)"),
    "max_sets": ("Prior", "Max accepted sets"),
    "walk_step": ("Prior", "Random-walk step"),
    "stability_units": ("Prior", "Stability duration (ND units)"),
    "min_cluster_size": ("Prior", "Min cluster size"),
    "min_samples": ("Prior", "Min samples"),
    # the flow and the Fisher rotation
    "hidden_features": ("Posterior", "Hidden features"),
    "num_transforms": ("Posterior", "Transforms"),
    "learning_rate": ("Posterior", "Learning rate"),
    "stop_after_epochs": ("Posterior", "Early-stop patience"),
    "max_num_epochs": None,
    "fisher_m": ("Posterior", "Ensemble per perturbation"),
    "fisher_dz": ("Posterior", "Central-difference step"),
    "fisher_points": ("Posterior", "Operating points"),
    # consents and names
    "accept_truncated": "Confirm the load in the dialog on the Posterior tab.",
    "accept_other_observation": "Tick 'Run on a different observation' on the Infer tab.",
    "name": "Choose another name in the Save box.",
    # inputs
    "bounds": ("Prior", "Bounds"),
    "cell": ("Infer", "Cell"),
    "units": ("Config", "Units"),
    "model": ("Config", "Model"),
    "device": None,
    "prior": ("Prior", "Prior"),
    "posterior": ("Posterior", "Posterior"),
    # recordings and the drive
    "recording_spont": ("Select the passive recording on the Infer tab (the 'Spontaneous' box on the "
                        "driven page, the 'Passive' box on the χ page)."),
    "recording_forced": ("Infer", "Forced"),
    "recording_probe": "Pick the probe's recording in the χ probe table on the Infer tab.",
    "drive_amplitude": ("Infer", _drive("amp")),
    "drive_frequency": ("Infer", _drive("freq")),
    "drive_phase": ("Infer", _drive("phase")),
    "chi_f0_si": ("Infer", "Drive F₀ (N)"),
    # chi conditioning
    "chi_n_freqs": ("Config", "χ probes per observation"),
    "chi_k_pad": ("Config", "χ probe slots (capacity)"),
    "chi_max_cycles": ("Config", "χ lock-in ceiling (cycles)"),
    "chi_f0": _FIXED,
    "chi_freq_bounds": _FIXED,
    # tool-only: the diagnostics' knobs; no window sentence
    "repeats": None, "n_points": None, "n_worst": None, "top_n": None, "m": None, "m_noise": None,
    "rel": None, "min_valid": None, "rows": None, "n_sweep": None, "chi_k_fixed": None,
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
    """What the yellow box says under the message: ``"Set it in the 'T_obs (s)' box on the Infer
    tab."`` for a box entry, the sentence itself for a sentence entry, ``""`` for ``field=None``, a
    None entry or an unknown key. Never raises: it runs while a refusal is being shown."""
    if key is None:
        return ""
    entry = CONTROL.get(key)
    if isinstance(entry, tuple):
        tab, text = entry
        where = " or ".join(tab) if isinstance(tab, tuple) else tab
        return f"Set it in the '{text}' box on the {where} tab."
    return entry or ""
