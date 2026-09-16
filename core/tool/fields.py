"""The command-line tool's half of V3 (piece 3 spec §3.2): which flag answers each refusal field.

A ``Refusal`` names the setting at fault by a key from ``core.refusals.FIELDS`` and says nothing
about any front end. ``main``'s ladder (``core/tool/__init__.py``) appends ``fix_sentence(e.field)``
to its ``refused:`` line, so the operator reads the flag to change next without the core knowing
that flags exist. This table is the ONLY place under ``core/tool`` where a refusal key is paired
with a flag: a renamed flag is renamed here, and ``tests/test_refusals.py`` pins every value against
``build_parser()``'s real option strings and the key set against the registry, both ways.

Torch-free and Qt-free on purpose -- it imports nothing from ``core`` -- because the ladder must be
able to print a refusal whose cause is that torch or the card was not available, and the suite
imports this module in a fresh interpreter and asserts that torch is absent afterwards.

``None`` marks a key no subcommand can set: the units (the tool declares none) and the four chi
constants only ``config.py`` sets (``chi_k_pad``, ``chi_max_cycles``, ``chi_f0``,
``chi_freq_bounds``). One repeatable flag can carry several keys: ``--forced PATH[@HZ]`` is both the
driven recording and a chi probe's, ``--drive NAME=VALUE`` is the amplitude, frequency and phase.
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
