"""The command-line tool's half of the refusal-naming rule: which flag answers each refusal field.

A ``Refusal`` names the setting at fault by a key from ``core.refusals.FIELDS`` and says nothing
about any front end. ``main``'s ladder (``core/tool/__init__.py``) appends ``fix_sentence(e.field)``
to its ``refused:`` line, so the operator reads the flag to change next without the core knowing
that flags exist. This table is the ONLY place under ``core/tool`` where a refusal key is paired
with a flag: a renamed flag is renamed here, and ``tests/test_refusals.py`` pins every value against
``build_parser()``'s real option strings and the key set against the registry, both ways.

Torch-free and Qt-free on purpose -- it imports nothing from ``core`` -- because the ladder must be
able to print a refusal whose cause is that torch or the card was not available, and the suite
imports this module in a fresh interpreter and asserts that torch is absent afterwards.

``None`` marks a key no option string answers: the units (the tool declares none), the four chi
constants only ``config.py`` sets (``chi_k_pad``, ``chi_max_cycles``, ``chi_f0``,
``chi_freq_bounds``), ``artifact``, which the ``artifacts`` subcommand names POSITIONALLY
(``artifacts show <kind> <ref>``) -- there is no flag to print, and the refusal's own sentence already
quotes the ref -- the five FDT settings neither front end exposes (the frequency band, the
burn-in, the two durations and the step), the model builder's seven numeric fields, which live on
a window-only screen, and the live simulation's two frame settings, on a window-only panel. One
repeatable flag can carry several keys:
``--forced PATH[@HZ]`` is both the driven recording and a chi probe's, ``--drive NAME=VALUE`` is the
amplitude, frequency and phase.
"""

FLAG: dict[str, str | None] = {
    # the observation and the training budget
    "t_obs": "--t-obs",
    "num_runs": "--num-runs",
    "run_size_cap": "--run-size",
    "checkpoint_every": "--checkpoint-every",
    "resume": "--resume",
    "new_run": "--new-run",
    # TSNPE
    "n_directions": "--directions",
    "hpd_level": "--level",
    "observation": "--observation",
    # calibration and inference sizes
    "n_cal": "--n-cal",
    "cal_n_scales": "--cal-n-scales",
    "num_posterior_samples": "--posterior-samples",     # dest=num_posterior_samples; the flag is shorter
    "n_samples": "--n-samples",
    # the prior sweep
    "num_iterations": "--num-iterations",
    "sweep_batch": "--sweep-batch",
    "max_sets": "--max-sets",
    "walk_step": "--walk-step",
    "stability_units": "--stability-units",
    "min_cluster_size": "--min-cluster-size",
    "min_samples": "--min-samples",
    # the flow and the Fisher rotation
    "hidden_features": "--hidden-features",
    "num_transforms": "--num-transforms",
    "learning_rate": "--learning-rate",
    "stop_after_epochs": "--stop-after-epochs",
    "max_num_epochs": "--max-epochs",
    "fisher_m": "--fisher-m",
    "fisher_dz": "--fisher-dz",
    "fisher_points": "--fisher-points",
    # consents and names
    "accept_truncated": "--accept-truncated",
    "accept_other_observation": "--accept-other-observation",
    "name": "--name",
    # the artifact browser: `artifacts note <kind> <ref> --note TEXT`. The note text is a FLAG (and
    # not a positional) precisely so this table can name one; the artifact itself is positional.
    "artifact": None,
    "note": "--note",                                   # add_name_flags defines it beside --name
    # comparing saved records: `compare <mode> --record REF --record REF`, one flag per record, and
    # the one setting each of two modes takes (renormalise's constant, sweeps' slice point)
    "compare_records": "--record",
    "prefactor": "--prefactor",
    "slice_at": "--at",
    # the model builder: a window-only screen, so no option string answers any of these
    "param_value": None, "param_min": None, "param_max": None, "init": None,
    "x_scale": None, "t_scale": None, "forcing_value": None,
    # the live simulation: a window-only panel; the tool has no streaming subcommand
    "frame_steps": None, "fps": None,
    # the two secondary analyses: `fdt` and `crossval` share the four resolution knobs, and the
    # grids and the preset are the sweep's alone. --seed exists on smoke and the simulating
    # diagnostics, and on these two as well.
    "n_freqs": "--n-freqs",
    "ensemble_m": "--ensemble-m",
    "freqs_per_batch": "--freqs-per-batch",
    "f0": "--f0",
    "preset": "--preset",
    "s_grid": "--s-grid",
    "t_grid": "--t-grid",
    "seed": "--seed",
    # the five FDT settings neither front end exposes: no option string answers them
    "freq_bounds": None, "burn_in_nd": None, "t_obs_periods": None, "dt_nd": None, "psd_t_obs_nd": None,
    # inputs
    "bounds": "--bounds",
    "cell": "--cell",
    "units": None,
    "model": "--model",
    "device": "--device",
    "prior": "--prior",
    "posterior": "--posterior",
    # recordings and the drive
    "recording_spont": "--spont",
    "recording_forced": "--forced",
    "recording_probe": "--forced",
    "drive_amplitude": "--drive",
    "drive_frequency": "--drive",
    "drive_phase": "--drive",
    "chi_f0_si": "--f0-si",
    # chi conditioning: one flag; the rest is config.py's
    "chi_n_freqs": "--chi-k",
    "chi_k_pad": None,
    "chi_max_cycles": None,
    "chi_f0": None,
    "chi_freq_bounds": None,
    # tool-only: the diagnostics' knobs
    "repeats": "--repeats",
    "n_points": "--n-points",
    "n_worst": "--n-worst",
    "top_n": "--top-n",
    "m": "--m",
    "m_noise": "--m-noise",
    "rel": "--rel",
    "min_valid": "--min-valid",
    "rows": "--rows",
    "n_sweep": "--n-sweep",
    "chi_k_fixed": "--chi-k-fixed",
    # tool-only: the probe checks. Their own flag names, never the training drive's; --seed and
    # --repeats answer two keys each, as --drive answers three
    "probe_seed": "--seed",
    "probe_lengths": "--lengths",
    "probe_multipliers": "--multipliers",
    "probe_drives": "--drives",
    "band_repeats": "--repeats",
    "probe_cycle_caps": "--cycle-caps",
    "cv_max": "--cv-max",
    "phase_max": "--phase-max",
    "snr_min": "--snr-min",
    "sup_min": "--sup-min",
    "band_peak_window": "--peak-window",
}


def fix_sentence(key: str | None) -> str:
    """``"(--t-obs)"`` for a key with a flag; ``""`` for ``field=None``, for a key with no flag and
    for a key this table does not know. Owns the parentheses, so the ladder's line reads
    ``refused: <message> (--t-obs)`` and simply ends at the message when there is nothing to add.
    Never raises: it runs while an error is being reported, and an unregistered key is the scan's
    job to catch (tests/test_refusals.py), not a reason to turn a refusal into a traceback."""
    if key is None:
        return ""
    flag = FLAG.get(key)
    return f"({flag})" if flag else ""
