"""
Prompt-free configuration builders, shared by PRISM's two front ends.

The interactive prompts this module was named for went with the retired prompt CLI. What is
left is pure: cell / bounds / units parsing, the unit conversion factors, and the ``make_*_config``
builders that the GUI (``core/gui``) and the command-line tool (``core/tool``) each call with values
they obtained their own way. The module keeps the name ``core.cli``.
"""
import math
from collections import OrderedDict
from pathlib import Path

import pint

from core import config          # live-read config.CHI_MODE / CHI_N_FREQS (avoid import-time snapshot)
from core import registry        # fdt_support, the FDT model gate; imports only config and torch
from .config import (
    SimConfig, FDTConfig, detect_device, cpu_device,
    DT_EXP_S, T_MIN_EXP_S, T_MAX_EXP_S,
    BOUNDS_PATH, UNITS_PATH,
)
from .Helpers import file_manager
from .refusals import (Refusal, describe, missing_values_phrase, refuse, require_at_least,
                       require_below, require_between, require_choice, require_file, require_finite,
                       require_positive)
from .rng import require_seed


class UnitParseError(Refusal):
    """Raised when a cell/units file names a unit pint can't resolve.

    Previously the parsers printed an error and called exit(), which killed the whole process --
    fatal for a GUI. They now raise this instead; each front end catches it -- the command-line tool
    turns it into a refusal (exit 1), and the GUI surfaces it as an error dialog.
    """


# ── Per-model input files ────────────────────────────────────────────────────
def resolve_units_file(model: str) -> str:
    """
    Auto-resolve the per-model units file (no prompt): Resources/Units/<model>/units.txt.

    :raises Refusal: (field "units") if the units file for this model is missing.
    """
    path = UNITS_PATH / model.lower() / "units.txt"
    if not path.exists():
        raise Refusal(f"Missing units file for model '{model}': expected {path}", field="units")
    return str(path)


# ── Post-training inference ─────────────────────────────────────────────────
def validate_gt_file(cfg: SimConfig, cell_path: str) -> list:
    """Non-mutating dry run of ``load_and_validate_gt``: the problems that would make it raise.

    Lets a GUI validate a cell the moment it is PICKED, on the GUI thread, instead of discovering the
    failure inside a worker halfway through an inference run. Mirrors SimConfig._fill_checked's rules --
    MISSING parameters are fatal, extras are ignored, and only ND/rescale are range-checked (forcing is
    the known drive, whose range is deliberately not enforced).

    :return: human-readable problem strings; empty means the cell can be injected.
    """
    try:
        inits, param_vals, rescale_vals, forcing_vals = file_manager.parse_values_file(cell_path)
    except Exception as e:                            # noqa: BLE001 -- surfaced to the user as-is
        return [f"could not parse the cell file: {e}"]
    problems = []
    for label, vals, cfg_dict, check_bounds in (
            ("ND parameter", param_vals, cfg.params_dict, True),
            ("rescale parameter", rescale_vals, cfg.rescale_params, True),
            ("forcing parameter", forcing_vals, cfg.force_params_dict, False)):
        missing = sorted(set(cfg_dict) - set(vals))
        if missing:
            # the phrase takes its label as given; this loop's labels are singular because the
            # out-of-bounds line below reads "ND parameter k = ...", so the plural is spelled here
            problems.append(missing_values_phrase(f"{label}s", missing))
            continue
        if check_bounds:
            problems += [f"{label} {n} = {vals[n]:g} is outside its bounds ({lo:g}, {hi:g})"
                         for n, (_, (lo, hi)) in cfg_dict.items() if not (lo <= vals[n] <= hi)]
    if not inits:
        problems.append("the cell file declares no initial conditions")
    return problems


def load_and_validate_gt(cfg: SimConfig, cell_path: str) -> list:
    """
    Parse a cell file's VALUES + initial conditions and inject them into cfg, validating (via
    SimConfig.inject_ground_truth) that the ND/rescale values lie within the bounds file's bounds.

    :return: names of cell values the bounds file does not declare, which were therefore IGNORED
             (e.g. f_scale + the drive when a forced cell is paired with spontaneous bounds). Empty in
             the usual matched case; callers may surface it, and existing callers can ignore it.
    :raises Refusal: (field "cell") for a blank path or one that names no file, before the parser
             (a file that exists but does not parse keeps the parser's behaviour).
    """
    require_file("cell", cell_path, "cell")
    inits, param_vals, rescale_vals, forcing_vals = file_manager.parse_values_file(cell_path)
    cfg.sources["cell"] = str(cell_path)
    return cfg.inject_ground_truth(inits, param_vals, rescale_vals, forcing_vals)


# Display-only SI unit hints, indexed by forcing param name (the GUI's drive fields).
# DERIVED from config.FORCING_SI_UNITS, the authoritative conversion table, so the two cannot drift.
INFERENCE_PROMPT_UNITS = config.FORCING_DISPLAY_UNITS

# ── Cell-file parsing (shared by the FDT/REDUCTION/CROSSVAL builders + the CrossVal panel) ───
def _merge_vals_bounds(vals: dict, bounds: OrderedDict,
                       label: str, cell_file: str) -> OrderedDict:
    """
    Merge cell VALUES with bounds-file BOUNDS into {name: (val, (lo, hi))}, iterating the bounds
    dict so param order follows the bounds file (the single source of truth for the set + order).

    Params present in the cell but absent from `bounds` are DROPPED (this keeps bp's rescale/forcing
    empty -- its bounds file has only the ND section). Every bounds param must have a value in the
    cell, else a clear error (a None in slot 0 would later crash params_tensor). Bounds are NOT
    range-checked here -- that enforcement lives in SimConfig.inject_ground_truth (the SBI path).
    """
    missing = [name for name in bounds if name not in vals]
    if missing:
        raise Refusal(
            f"Cell file '{cell_file}' is missing value(s) for {label} required by the bounds file: {missing}.",
            field="cell")
    merged = OrderedDict()
    for name, (_, bnds) in bounds.items():
        merged[name] = (vals[name], bnds)
    return merged


#: Bounds file every cell in a model folder falls back to when it has no same-named sibling.
#: Named, not guessed: a folder with several cells sharing one box needs a declared default, and
#: silently picking "the only file present" would break the moment a second one appeared.
MASTER_BOUNDS_NAME = "master.txt"


def resolve_bounds_for_cell(cell_file: str, model: str | None = None) -> Path | None:
    """Bounds file governing ``cell_file``, or None if neither candidate exists.

    Resolution order:
      1. the same-named sibling ``Bounds/<model>/<cell>.txt`` -- the original one-box-per-cell
         convention, so every pre-existing cell resolves exactly as it always did;
      2. ``Bounds/<model>/master.txt`` -- the shared box, for a model whose cells are variations on
         one parameter set (the three Nadrowski master cells differ only in their Forcing section,
         and giving each a byte-identical copy of the same box is what let the old per-cell files
         drift apart in the first place).

    Returning None rather than raising keeps the LEGACY inline-bounds path reachable: a cell that
    carries its own bounds has no file in Bounds/ at all, and that is not an error.
    """
    p = Path(cell_file)
    model = (model or p.parent.name).lower()
    sibling = BOUNDS_PATH / model / p.name
    if sibling.exists():
        return sibling
    master = BOUNDS_PATH / model / MASTER_BOUNDS_NAME
    return master if master.exists() else None


def cell_sources(cell_file: str, model: str | None = None) -> dict:
    """The ``sources`` block for a cell: its own path, the bounds file that RESOLVES for it, the
    units file, and the model NAME. Beside :func:`parse_cell`, which keeps its 7-tuple.

    parse_cell resolves both paths and throws them away (``bounds_path`` and ``units_path`` are
    local to it), so an FDT record had no way to say which box its parameter set came from. The two
    candidates resolve DIFFERENTLY -- a same-named sibling, else the folder's master.txt -- and only
    one of them governed the run.

    Both come back None on the LEGACY inline-bounds branch, and the condition is parse_cell's own
    (``bounds_path is not None and units_path.exists()``): the units file is only read when the
    decoupled path is taken, so reporting it alone would name a file the run never opened.
    """
    p = Path(cell_file)
    name = (model or p.parent.name)
    bounds_path = resolve_bounds_for_cell(cell_file, name.lower())
    units_path = UNITS_PATH / name.lower() / "units.txt"
    decoupled = bounds_path is not None and units_path.exists()
    return {"cell": str(cell_file),
            "bounds": str(bounds_path) if decoupled else None,
            "units": str(units_path) if decoupled else None,
            "model": name.upper()}


def parse_cell(cell_file: str, model: str | None = None):
    """
    Parse a cell file into the 7-tuple used by the FDT/REDUCTION/CROSSVAL config builders
    (``make_fdt_config``, ``make_reduction_config``, ``make_param_sweep_config``) and by the CrossVal
    panel's prefill from the chosen cell, then run pint unit conversion.

    Cell files hold VALUES only (bounds + units are decoupled) and live in per-model subfolders:
    Resources/Cells/<model>/<cell>.txt. If a bounds file RESOLVES for the cell (see
    :func:`resolve_bounds_for_cell`) AND Resources/Units/<model>/units.txt exists, use the DECOUPLED
    path: the bounds file defines the param set + order, the cell supplies the values, units come
    from the units file. Otherwise fall back to the legacy parse_model_file (bounds + units read
    inline from the cell).

    :param cell_file: path to the cell file (Resources/Cells/<model>/<cell>.txt).
    :param model: model name for resolving the bounds/units files; derived from the cell's parent
                  folder (e.g. '.../Cells/nadrowski/master_weak.txt' -> 'nadrowski') when None.
    :return: (inits_dict, params_dict, rescale_params, force_params_dict,
             units_dict, si_factors, s_to_cell)
    """
    p = Path(cell_file)
    if model is None:
        model = p.parent.name
    model = model.lower()

    bounds_path = resolve_bounds_for_cell(cell_file, model)
    units_path = UNITS_PATH / model / "units.txt"    # Units/<model>/units.txt

    if bounds_path is not None and units_path.exists():
        # Decoupled path: bounds file = param set + order; cell = values; units file = units.
        b_params, b_rescale, b_forcing, _ = file_manager.parse_bounds_file(str(bounds_path))
        inits_dict, v_params, v_rescale, v_forcing = file_manager.parse_values_file(cell_file)
        params_dict = _merge_vals_bounds(v_params, b_params, "ND parameters", cell_file)
        rescale_params = _merge_vals_bounds(v_rescale, b_rescale, "rescale parameters", cell_file)
        force_params_dict = _merge_vals_bounds(v_forcing, b_forcing, "forcing parameters", cell_file)
        units_dict = file_manager.parse_units_file(str(units_path))
        si_factors, s_to_cell = units_to_factors(units_dict)
        return inits_dict, params_dict, rescale_params, force_params_dict, units_dict, si_factors, s_to_cell

    # ── Legacy fallback: bounds + units read inline from the cell (models without decoupled files) ──
    inits_dict, params_dict, rescale_params, force_params_dict, units_dict = file_manager.parse_model_file(cell_file)

    ureg = config.unit_registry()      # shared singleton: building one parses pint's full unit file
    try:
        si_factors = [ureg(unit).to_base_units().magnitude for unit in units_dict]
    except pint.UndefinedUnitError as e:
        raise UnitParseError(f"{e}. Unrecognized unit in cell file '{cell_file}'.", field="cell")

    time_unit = None
    for unit_str in units_dict:
        try:
            if ureg.Quantity(1, unit_str).check("[time]"):
                time_unit = unit_str
                break
        except pint.UndefinedUnitError:
            continue
    if time_unit is None:
        raise Refusal("Could not detect time unit from cell file. Ensure t_scale has a time unit.",
                      field="cell")

    s_to_cell = ureg.Quantity(1, "s").to(time_unit).magnitude
    return inits_dict, params_dict, rescale_params, force_params_dict, units_dict, si_factors, s_to_cell


# ── Units → conversion factors (standalone helper for the decoupled bounds/units path) ──────
def units_to_factors(units: tuple) -> tuple[list[float], float]:
    """
    From a set of unit strings compute (si_factors, s_to_cell). Standalone so `parse_cell` stays
    byte-for-byte untouched (it is shared by the FDT/REDUCTION/CROSSVAL builders + the CrossVal
    panel). Also called by ``make_sim_config``, and by the Config tab to refuse a typed unit token it
    cannot resolve before a run starts.
    """
    ureg = config.unit_registry()      # shared singleton: building one parses pint's full unit file
    try:
        si_factors = [ureg(unit).to_base_units().magnitude for unit in units]
    except pint.UndefinedUnitError as e:
        raise UnitParseError(f"{e}. Unrecognized unit in the units file.", field="units")
    time_unit = None
    for unit_str in units:
        try:
            if ureg.Quantity(1, unit_str).check("[time]"):
                time_unit = unit_str
                break
        except pint.UndefinedUnitError:
            continue
    if time_unit is None:
        raise Refusal("Could not detect a time unit from the units file. Include a time unit (e.g. ms).",
                      field="units")
    return si_factors, ureg.Quantity(1, "s").to(time_unit).magnitude


# ── Pure config cores (no prompts) — shared by `python -m core` and the GUI ────────────
def make_sim_config(model: str, labels: list[str], state_dep_drift: bool, bounds_file: str = None, *,
                    bounds_dicts=None, units_override=None,
                    chi_mode: bool | None = None, chi_n_freqs: int | None = None,
                    chi_f0: float | None = None, chi_freq_bounds: tuple | None = None,
                    chi_k_pad: int | None = None, chi_max_cycles: float | None = None,
                    reparam_rotate: bool | None = None,
                    hw: "DeviceConfig | None" = None) -> SimConfig:
    """
    Build a bounds-only SimConfig (no prompts) from a chosen model + bounds file. Ground-truth values,
    initial conditions, and T_obs are filled later (only for simulated inference). Shared by
    the command-line tool (core/tool) and the GUI's SBI config form.

    The chi(omega) knobs are explicit keyword args so a GUI can set them PER CONFIG; each falls back to
    the live ``config.CHI_*`` module value when None (the command-line tool passes only chi_mode and
    chi_n_freqs, and keeps this behaviour for the rest).
    They are stored ON the config, so a posterior trained from it is self-describing.

    ``units_override`` DECLARES what the numbers in the bounds/cell files mean; it never converts them.
    Pass a path to a units file, or an iterable of unit tokens (e.g. ``("nm", "ms", "pN", "kHz")``).
    None keeps the per-model default ``Resources/Units/<model>/units.txt``.

    ``bounds_dicts`` is the hand-entered alternative to ``bounds_file``: a
    ``(params, rescale, forcing)`` triple in ``parse_bounds_file``'s shape. Exactly one of the two must
    be given. Parameter ORDER within each dict is load-bearing (simulators bind columns positionally),
    so it is preserved verbatim.

    ``hw`` is the DeviceConfig to run on; None detects one. The GUI passes nothing and keeps
    detect_device(); the command-line tool builds it from --device. It is not a cosmetic setting --
    the device and dtype are part of the simulation identity (core/artifacts/identity.py:72-73).
    """
    if (bounds_file is None) == (bounds_dicts is None):
        raise Refusal("make_sim_config needs exactly one of bounds_file or bounds_dicts.")
    if bounds_dicts is not None:
        params_dict, rescale_params, force_params_dict = (OrderedDict(d) for d in bounds_dicts)
    else:
        # The file is checked by its input kind first, so both front ends refuse a missing one
        # with the field key; the parser's own FileNotFoundError is then reached only by a race.
        require_file("bounds", bounds_file, "bounds")
        params_dict, rescale_params, force_params_dict, _ = file_manager.parse_bounds_file(bounds_file)
    units_path = None
    if units_override is None:
        units_path = resolve_units_file(model)
        units_dict = file_manager.parse_units_file(units_path)
    elif isinstance(units_override, (str, Path)):
        units_path = str(units_override)
        units_dict = file_manager.parse_units_file(str(units_override))
    else:
        units_dict = tuple(str(u) for u in units_override)
    _, s_to_cell = units_to_factors(units_dict)

    # convert experimental constants from seconds to cell-file time units (training T-range)
    return SimConfig(
        model=model,
        labels=labels,
        state_dep_drift=state_dep_drift,
        inits_dict=OrderedDict(),               # filled from a cell file only if inferring on a simulation
        params_dict=params_dict,                # {name: (None, (lo,hi))} until a cell injects values
        rescale_params=rescale_params,
        force_params_dict=force_params_dict,
        units_dict=units_dict,
        # multi-frequency chi(omega) conditioning; explicit arg wins, else the live module value
        chi_mode=config.CHI_MODE if chi_mode is None else bool(chi_mode),
        chi_n_freqs=config.CHI_N_FREQS if chi_n_freqs is None else int(chi_n_freqs),
        chi_f0=config.CHI_F0 if chi_f0 is None else float(chi_f0),
        chi_freq_bounds=(config.CHI_FREQ_BOUNDS if chi_freq_bounds is None
                         else tuple(chi_freq_bounds)),
        chi_k_pad=config.CHI_K_PAD if chi_k_pad is None else int(chi_k_pad),
        chi_max_cycles=(config.CHI_MAX_CYCLES if chi_max_cycles is None else float(chi_max_cycles)),
        reparam_rotate=(config.REPARAM_ROTATE if reparam_rotate is None else bool(reparam_rotate)),
        dt_exp=DT_EXP_S * s_to_cell,
        t_min_exp=T_MIN_EXP_S * s_to_cell,
        t_max_exp=T_MAX_EXP_S * s_to_cell,
        T_obs=None,                             # observation duration is supplied later, at the
                                                 # inference step (the GUI's field, or --t-obs)
        hw=detect_device() if hw is None else hw,
        sources={k: str(v) for k, v in (("bounds", bounds_file), ("units", units_path)) if v},
    )


# ── Pure config core (FDT) ───────────────────────────────────────────────────
def check_fdt_settings(cfg: FDTConfig) -> None:
    """Refuse the five FDT settings neither front end exposes: the frequency band, the burn-in, the
    two durations and the integration step.

    They are parameters of neither builder -- they arrive from the dataclass defaults, from the
    closed preset, or from ``with_overrides`` -- so a bad one is a hand-edited preset or a caller's
    bug, not a mistyped control. They are registered in ``core.refusals.FIELDS`` like every key and
    map to ``None`` in BOTH front-end tables: each refusal names the setting and offers no fix,
    because there is no box and no flag to name.
    """
    require_positive("dt_nd", cfg.dt_nd)
    require_positive("psd_t_obs_nd", cfg.psd_T_obs_nd)
    require_positive("t_obs_periods", cfg.T_obs_periods)
    # A zero burn-in is a well-defined setting, not a broken one, and a floor sits only where the
    # computation would otherwise be undefined (a setting merely too thin to trust warns). Not the
    # count rule, require_at_least: it coerces with int(), so -0.5 would pass as 0 and NaN would
    # raise a bare ValueError. Finite first, so NaN and an infinity get that rule's own sentence; then
    # the CLOSED half-line, which keeps 0 legal and reports a negative as it was given. Closed, not an
    # open upper end (open_hi=True): require_between marks any open end "(exclusive)" without saying
    # which, and beside a legal 0 that reads as though 0 were excluded.
    require_between("burn_in_nd", require_finite("burn_in_nd", cfg.burn_in_nd), 0.0, math.inf)
    lo, hi = cfg.freq_bounds
    require_positive("freq_bounds", lo)
    require_below("freq_bounds", lo, hi)


def make_fdt_config(model: str, state_dep_drift: bool, cell_file: str, *,
                    n_freqs: int = 60, ensemble_M: int = 256, freqs_per_batch: int = 1,
                    F0: float = 0.05, seed: "int | None" = None) -> FDTConfig:
    """Build an FDTConfig (no prompts) from a model + cell file + FDT knobs. Shared by the command-line
    tool (core/tool) and the GUI's FDT form.

    The checks live HERE and not on FDTConfig and not in the screens, so both front ends
    inherit one wording. The knobs, the cell and the model are checked before ``parse_cell``: every
    one of those checks is free, and the four knobs are what a blank field turns into a zero -- 0
    frequencies is an empty figure and exit 0, 0 trajectories is a ZeroDivisionError, 0 frequencies
    per call is an unbounded loop that looks like a hang, and F0 = 0 divides by zero in the lock-in.
    The five settings no front end exposes are not arguments here, so they are checked on the built
    object, by :func:`check_fdt_settings`.
    """
    n_freqs = require_at_least("n_freqs", n_freqs, 1)
    ensemble_M = require_at_least("ensemble_m", ensemble_M, 1)
    freqs_per_batch = require_at_least("freqs_per_batch", freqs_per_batch, 1)
    F0 = require_positive("f0", F0)
    if seed is not None:
        seed = require_seed(seed)
    require_file("cell", cell_file, "cell")
    ok, reason = registry.fdt_support(model)
    if not ok:
        # NOT require_choice: fdt_support is a predicate returning a tailored diagnostic sentence
        # per model, not a list of choices, and that sentence is the useful half of the refusal.
        refuse("model", reason)
    (inits_dict, params_dict, rescale_params, force_params_dict,
     units_dict, _, _) = parse_cell(cell_file, model=model)
    cfg = FDTConfig(
        model=model,
        state_dep_drift=state_dep_drift,
        inits_dict=inits_dict,
        params_dict=params_dict,
        rescale_params=rescale_params,
        force_params_dict=force_params_dict,
        units_dict=units_dict,
        n_freqs=n_freqs,
        ensemble_M=ensemble_M,
        freqs_per_batch=freqs_per_batch,
        F0=F0,
        hw=cpu_device(),  # FDT: sequential SDE loop at M~256 is ~3.4x faster on CPU than GPU
        sources=cell_sources(cell_file, model),
        seed=seed,
    )
    check_fdt_settings(cfg)
    return cfg


def make_reduction_config(cell_file: str, *, F0: float = 0.05) -> FDTConfig:
    """Build a reduction-map FDTConfig (no prompts) from a cell file. Model is fixed to NADROWSKI
    (the reduction is Nadrowski-specific). The GUI's Reduction form is its only caller -- there is no
    reduction subcommand."""
    (inits_dict, params_dict, rescale_params, force_params_dict,
     units_dict, _, _) = parse_cell(cell_file, model="NADROWSKI")
    return FDTConfig(
        model="NADROWSKI",
        state_dep_drift=True,
        inits_dict=inits_dict,
        params_dict=params_dict,
        rescale_params=rescale_params,
        force_params_dict=force_params_dict,
        units_dict=units_dict,
        F0=F0,
        hw=detect_device(),
    )


# ── Sweep-study resolution presets ───────────────────────────────────────────
# Drive the FDT resolution knobs for the parameter-sweep study. The exploratory
# preset is a fast/coarse pass to confirm the FDT-restoration trend before a full
# overnight run; production is the publication-quality resolution. The dominant
# cost is Campaign 2's low-frequency drive points (cost ~ 1/omega), so the
# exploratory preset raises freq_bounds[0] and trims n_freqs / T_obs_periods /
# psd_T_obs_nd while keeping ensemble_M=256 so the trend stays clean above noise.
# Public: the GUI's CrossVal panel builds its preset combo and knob defaults from this table.
SWEEP_PRESETS = {
    "exploratory": dict(freq_bounds=(0.2, 30.0), n_freqs=30, T_obs_periods=20,
                        psd_T_obs_nd=4000.0, ensemble_M=256, points=8),
    "production":  dict(freq_bounds=(0.1, 30.0), n_freqs=60, T_obs_periods=30,
                        psd_T_obs_nd=8000.0, ensemble_M=256, points=12),
}


def _check_grid(key: str, spec: tuple) -> tuple:
    """One sweep axis, ``(min, max, N)``, as ``np.linspace`` needs it: both ends finite, ``N`` a
    whole number of at least 2, and the minimum below the maximum.

    Checked as a WHOLE, because an end is only wrong against the other one. A blank box must
    therefore arrive as None, never as 0: 0 is a legal sweep end, so a blank read as 0 under a
    positive max passes "min below max" as a sweep nobody typed. The window's grid row reads its
    boxes through value_or_none, so each blank part is refused here AS blank, naming which one --
    its minimum, its maximum or its point count (a blank count used to read "needs at least 2
    points; got 0", a zero nobody typed).
    """
    lo, hi, n = spec
    what = describe(key)
    what = what[0].upper() + what[1:]
    for part, value in (("minimum", lo), ("maximum", hi), ("point count", n)):
        if value is None:
            refuse(key, f"{what} is incomplete: its {part} is blank.")
    n = int(n)
    if n < 2:
        refuse(key, f"{what} needs at least 2 points; got {n}.")
    lo, hi = require_below(key, lo, hi)
    return lo, hi, n


def make_param_sweep_config(cell_file: str, *, preset: dict, preset_name: str,
                            s_spec: tuple, t_spec: tuple,
                            n_freqs: int | None = None, ensemble_M: int | None = None,
                            freqs_per_batch: int | None = None, F0: float | None = None,
                            seed: "int | None" = None) -> tuple["FDTConfig", "np.ndarray", "np.ndarray"]:
    """Build (FDTConfig, s_grid, temp_grid) for the sweep study (no prompts). ``preset`` supplies the
    advanced resolution levers (freq_bounds / T_obs_periods / psd_T_obs_nd); ``s_spec``/``t_spec`` are
    (min, max, n_points). Model fixed to NADROWSKI. Shared by the command-line tool (core/tool) + the GUI.

    ``preset_name`` is the name of the preset ``preset`` was resolved from. Both front ends pick the
    preset from a closed list and then pass the resolved DICT, dropping the name -- and the record's
    ``settings["preset"]`` has to hold the name, because the dict alone does not say which of the two
    a reader is looking at.

    Each unset knob falls back to the preset (or to FDTConfig's own default), here rather than at
    each call site, so the window and the tool cannot fall back differently.
    """
    import numpy as np  # local import — keep top-of-file lean
    require_choice("preset", preset_name, tuple(SWEEP_PRESETS))
    n_freqs = require_at_least("n_freqs", preset["n_freqs"] if n_freqs is None else n_freqs, 1)
    ensemble_M = require_at_least(
        "ensemble_m", preset["ensemble_M"] if ensemble_M is None else ensemble_M, 1)
    freqs_per_batch = require_at_least(
        "freqs_per_batch", 1 if freqs_per_batch is None else freqs_per_batch, 1)
    F0 = require_positive("f0", 0.05 if F0 is None else F0)
    if seed is not None:
        seed = require_seed(seed)
    s_spec = _check_grid("s_grid", s_spec)
    t_spec = _check_grid("t_grid", t_spec)
    if t_spec[0] < 0:
        # A negative T_a/T is unphysical, and it is the one known way a sweep point diverges -- the
        # model takes a square root of it, which torch answers with NaN rather than an error. 0
        # stays legal (no active noise is a meaningful operating point) and S has no floor; any other
        # diverging point is a failed point in the sweep itself.
        refuse("t_grid", f"The temperature sweep grid must not reach below 0: T_a/T is a ratio of two "
                         f"temperatures, so a negative value is unphysical, and the simulation "
                         f"diverges there; got a minimum of {t_spec[0]:g}.")
    require_file("cell", cell_file, "cell")
    (inits_dict, params_dict, rescale_params, force_params_dict,
     units_dict, _, _) = parse_cell(cell_file, model="NADROWSKI")
    s_grid = np.linspace(*s_spec)
    temp_grid = np.linspace(*t_spec)
    cfg = FDTConfig(
        model="NADROWSKI",
        state_dep_drift=True,
        inits_dict=inits_dict,
        params_dict=params_dict,
        rescale_params=rescale_params,
        force_params_dict=force_params_dict,
        units_dict=units_dict,
        n_freqs=n_freqs,
        freq_bounds=preset["freq_bounds"],
        ensemble_M=ensemble_M,
        freqs_per_batch=freqs_per_batch,
        F0=F0,
        T_obs_periods=preset["T_obs_periods"],
        psd_T_obs_nd=preset["psd_T_obs_nd"],
        hw=cpu_device(),  # sweep: sequential SDE loop at M~256 is ~3.4x faster on CPU than GPU
        sources=cell_sources(cell_file, "NADROWSKI"),
        seed=seed,
        preset_name=preset_name,
    )
    check_fdt_settings(cfg)
    return cfg, s_grid, temp_grid
