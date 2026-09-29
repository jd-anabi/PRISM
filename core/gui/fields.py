"""The window's half of the refusal-naming rule: which control answers each refusal field.

A ``Refusal`` names the setting at fault by a key from ``core.refusals.FIELDS`` and never a box,
tab, button or flag. ``BasePanel._refusal`` puts ``fix_sentence(exc.field)`` under the message in
the yellow "Check your inputs" box, and the inference tabs build their static rows from
``label(key)`` -- ``add_help_row(form, label("t_obs"), ...)`` -- so a control is named in ONE place
and a renamed one cannot leave a stale sentence behind. ``add_help_row`` applies
``labels.pretty_gui`` to whatever it is given (``core/gui/widgets/help_badge.py``), so the strings
here are the plain forms ("T_obs (s)"), and the read-back pin compares ``pretty_gui(label(key))``
against the QLabel on the tab a tuple names.

Three shapes of entry:

* ``(place, label)`` for a box or a picker: ``place`` is a tab title exactly as its section shows it
  -- the six inference tabs (Config, Prior, Posterior, Validate, Infer, TSNPE), "FDT analysis",
  "Sweep study cross-validation", "NWK → Hopf reduction map", "Live simulation" -- or one of
  ``SCREENS``, and the label is the row as the panel passes it to ``add_help_row``. A tuple of
  places is how one input names every place it appears: the cell picker is on five of them.
* one sentence, for a consent, a dialog, a table, the name box, a value fixed by measurement, a
  control on the Artifacts screen, whose list and Note box are not form rows, or the model builder's
  forcing fields, whose rows are labelled by parameter name; the sentence quotes the control's own
  text ("Run on a different observation").
* ``None`` for a key the window has no control for: the tool-only diagnostics knobs, and the six
  settings the window never exposes (the checkpoint cadence, the resume policy, the device, the
  two sample counts, the epoch ceiling), and the five FDT settings neither front end exposes.

``num_runs`` and ``run_size_cap`` sit on both the Posterior and TSNPE tabs with the same label, so
their tab slot is a tuple of both and the sentence names both: one tab would send a user refused on
the TSNPE tab to the Posterior tab. The three drive keys
are static strings built exactly as the Infer tab builds its rows (``labels.gui_forcing_label``
with ``config.FORCING_DISPLAY_UNITS``); ``_rebuild_forcing_fields`` keeps deriving its rows from
the config's forcing names, so a model with an ``amp_y`` or ``offset`` drive shows rows this table
does not name. ``tests/test_nav_and_gating.py`` pins the key set against the registry both ways,
the places against the built window, and the sentences verbatim.
"""
from core import config
from core.Helpers import labels
from core.refusals import FIELDS

#: Place strings that name a SCREEN rather than a tab. ``fix_sentence`` renders "on the <place>
#: screen" for these and "on the <place> tab" for everything else. The Artifacts browser and the
#: model builder are whole screens with no tab widget; the tab titles outside Parameter Inference --
#: "NWK → Hopf reduction map", "FDT analysis", "Sweep study cross-validation", "Live simulation" --
#: are ORDINARY tab entries, because those sections do host a tab widget.
SCREENS: frozenset = frozenset({"Artifacts", "Model Builder"})

_FIXED = "Fixed by measurement: change it in config.py, deliberately."


def _drive(name: str) -> str:
    """The Infer tab's own row label for a forcing parameter (infer_tab._rebuild_forcing_fields)."""
    return labels.gui_forcing_label(name, config.FORCING_DISPLAY_UNITS[name])


CONTROL: dict[str, tuple[str | tuple[str, ...], str] | str | None] = {
    # the observation and the training budget
    "t_obs": (("Infer", "Live simulation"), "T_obs (s)"),   # the Simulate panel's box too
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
    "cal_n_scales": ("Validate", "(t_scale, T_obs) operating points"),
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
    # A name and a note are typed in several boxes, and each sentence lists them all: a run's record
    # and a comparison's are named in boxes of their own on the same two tabs, so a taken
    # comparison name is sent to the 'Comparison name' box, not the run's.
    "name": ("Choose another name in the Save box, or in the 'Record name' or 'Comparison name' box "
             "on the FDT analysis or Sweep study cross-validation tab."),
    # the artifact browser. Sentences, not (place, label), although the Artifacts screen is a place
    # in SCREENS: the list is not a box, and the Note box sits in the screen's action row, not in a
    # form row the read-back pin reads.
    "artifact": "Select an artifact in the list on the Artifacts screen.",
    "note": ("Edit it in the Note box on the Artifacts screen, or in the 'Note' or 'Comparison note' "
             "box on the FDT analysis or Sweep study cross-validation tab."),
    # A sentence and not a (place, label) pair: the comparison list is one control that appears on
    # two tabs of the FDT section, under a different row on each. Both rows are quoted, and the verb
    # is "Choose", not "Add": a mode's ceiling ("at most 2") and a run named twice are fixed by
    # REMOVING one.
    "compare_records": ("Choose the records in the 'Runs to compare' list on the FDT analysis tab, or "
                        "in the 'Sweeps to compare' list on the Sweep study cross-validation tab."),
    # The two mode settings, each a box the comparison controls build: the renormalise mode's
    # constant on the FDT analysis tab, the sweeps mode's slice point on the cross-validation tab.
    # Sentences until those boxes existed; the words fix_sentence renders did not change.
    "prefactor": ("FDT analysis", "Normalisation constant"),
    "slice_at": ("Sweep study cross-validation", "Slice at"),
    # inputs. The cell picker and the model combo appear on several places at once and each entry
    # names them ALL: a bad cell chosen on the FDT analysis tab used to be answered with "the
    # Infer tab". ``units`` is NOT widened -- one units control exists in the whole application (the
    # Config tab's toggle); everywhere else the units file is resolved from the model and there is
    # no control to name.
    "bounds": ("Prior", "Bounds"),
    "cell": (("Infer", "FDT analysis", "Sweep study cross-validation",
              "NWK → Hopf reduction map", "Live simulation"), "Cell"),
    "units": ("Config", "Units"),
    "model": (("Config", "FDT analysis", "Live simulation"), "Model"),
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
    # the two secondary analyses. Ordinary tuple entries now that a place may be any section's tab
    # title: the label is the row the panel builds, so label(key) keeps a box and its hint sentence
    # from drifting apart; the Seed rows too are built from label("seed") on both panels.
    # The two labels are the words the rows SHOW: the raw forms "ensemble_M" and "freqs_per_batch"
    # rendered as "M_ensemble" and "freqs / batch", so the fix sentence named a box the tab does not
    # show. pretty_gui still renders "M_ensemble" with its subscript; no QSettings key is a label
    # (each panel writes its own literal key names).
    "n_freqs": (("FDT analysis", "Sweep study cross-validation"), "n_freqs"),
    "ensemble_m": (("FDT analysis", "Sweep study cross-validation"), "M_ensemble"),
    "freqs_per_batch": (("FDT analysis", "Sweep study cross-validation"), "freqs / batch"),
    "f0": (("FDT analysis", "Sweep study cross-validation", "NWK → Hopf reduction map"),
           "F0 (ND forcing amplitude)"),
    "preset": ("Sweep study cross-validation", "Preset"),
    "s_grid": ("Sweep study cross-validation", "S grid  (T_a/T = 1)"),
    "t_grid": ("Sweep study cross-validation", "T_a/T grid  (S = 0)"),
    "seed": (("FDT analysis", "Sweep study cross-validation"), "Seed"),
    # the five FDT settings NEITHER front end exposes: no control, so fix_sentence returns "" and
    # the refusal names the setting and offers no fix
    "freq_bounds": None, "burn_in_nd": None, "t_obs_periods": None, "dt_nd": None, "psd_t_obs_nd": None,
    # the model builder. A SCREEN, not an inference tab -- fix_sentence renders
    # "on the Model Builder screen" for a place in SCREENS. The parameter rows repeat per parameter,
    # so the label is the row's own ("value" / "min" / "max") and the message names which parameter.
    "param_value": ("Model Builder", "value"),
    "param_min": ("Model Builder", "min"),
    "param_max": ("Model Builder", "max"),
    "init": ("Model Builder", "init"),
    "x_scale": ("Model Builder", "x_scale (nm)"),
    "t_scale": ("Model Builder", "t_scale (s)"),
    # ONE key for every forcing field: the forcing rows' labels are the parameter NAMES (amp, freq,
    # tau, ...), built per kind, so there is no one label(key) to build a row from.
    "forcing_value": ("Set it in the forcing parameter's own box, beneath the variable's 'forcing' "
                      "choice, on the Model Builder screen."),
    # the live simulation
    "frame_steps": ("Live simulation", "Steps / frame"),
    "fps": ("Live simulation", "Max FPS"),
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


def _where(place) -> str:
    """The place phrase: ``"the Infer tab"``, ``"the Posterior or TSNPE tab"``, ``"the Artifacts
    screen"``, and for a mixed tuple ``"the Infer tab or the Model Builder screen"``.

    An all-tab or all-screen tuple shares ONE noun, which keeps the sentences byte-identical
    wherever they are quoted ("on the Posterior or TSNPE tab"); only a tuple that genuinely mixes
    the two spells the noun out per place, because "the Infer or Model Builder tab" would be a lie
    about one of them."""
    names = place if isinstance(place, tuple) else (place,)
    screens = [n in SCREENS for n in names]
    if all(screens):
        return f"the {' or '.join(names)} screen"
    if not any(screens):
        return f"the {' or '.join(names)} tab"
    return " or ".join(f"the {n} {'screen' if n in SCREENS else 'tab'}" for n in names)


def fix_sentence(key: str | None) -> str:
    """What the yellow box says under the message: ``"Set it in the 'T_obs (s)' box on the Infer or
    Live simulation tab."`` for a box entry, the sentence itself for a sentence entry, ``""`` for
    ``field=None``, a None entry or an unknown key. Never raises: it runs while a refusal is being
    shown."""
    if key is None:
        return ""
    entry = CONTROL.get(key)
    if isinstance(entry, tuple):
        place, text = entry
        return f"Set it in the '{text}' box on {_where(place)}."
    return entry or ""
