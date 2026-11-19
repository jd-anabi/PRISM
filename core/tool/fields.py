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
quotes the ref -- the model builder's seven numeric fields, which live on a window-only screen, the
name of the
observation an inference makes (``observation_name``), which ``infer`` never gives, so its observation
stays unnamed, and three keys of a run plan (``plan``, ``reparam_rotate``, ``expected_epochs``): the tool
reads no plan, so none of them is ever a refusal of the tool's, whatever flag sets the same setting. The
plan's fourth, ``chi_mode``, the kind of observation, is a refusal of the tool's too: a kept training
resumed in another kind is refused under it, and ``--chi`` (or ``--no-chi``) is the flag that answers it.
One repeatable flag can carry several keys:
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
    "resume_fit": "--resume-fit",
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
    # the observation an inference makes: infer gives it no name, so no flag answers its name
    "observation_name": None,
    # the artifact browser: `artifacts note <kind> <ref> --note TEXT`. The note text is a FLAG (and
    # not a positional) precisely so this table can name one; the artifact itself is positional.
    "artifact": None,
    "note": "--note",                                   # add_name_flags defines it beside --name
    # the model builder: a window-only screen, so no option string answers any of these
    "param_value": None, "param_min": None, "param_max": None, "init": None,
    "x_scale": None, "t_scale": None, "forcing_value": None,
    # --seed: smoke's and the simulating diagnostics'
    "seed": "--seed",
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
    # a run plan: the tool reads none; the kind of observation is also a kept training's, which --chi sets
    "plan": None, "chi_mode": "--chi", "reparam_rotate": None, "expected_epochs": None,
    # the diagnostics' knobs
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
    # the probe checks: their own flag names, never the training drive's; --seed,
    # --num-runs, --run-size, --peak-window and --t-obs answer two keys each, and --repeats three, as
    # --drive answers three
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
    "mask_num_runs": "--num-runs",
    "mask_run_size": "--run-size",
    "drive_t_obs": "--t-obs",
    "drive_repeats": "--repeats",
    "drive_detune": "--detune",
    "drive_strengths": "--strengths",
    "free_min": "--free-min",
    "captured_max": "--captured-max",
    "drive_peak_window": "--peak-window",
    "clarity_min": "--clarity-min",
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
