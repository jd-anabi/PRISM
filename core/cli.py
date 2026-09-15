"""
Prompt-free configuration builders, shared by PRISM's two front ends.

The interactive prompts this module was named for went with the prompt CLI (piece 2, D1). What is
left is pure: cell / bounds / units parsing, the unit conversion factors, and the ``make_*_config``
builders that the GUI (``core/gui``) and the command-line tool (``core/tool``) each call with values
they obtained their own way. The module keeps the name ``core.cli``.
"""
from collections import OrderedDict
from pathlib import Path

import pint

from core import config          # live-read config.CHI_MODE / CHI_N_FREQS (avoid import-time snapshot)
from .config import (
    SimConfig, FDTConfig, detect_device, cpu_device,
    DT_EXP_S, T_MIN_EXP_S, T_MAX_EXP_S,
    BOUNDS_PATH, UNITS_PATH,
)
from .Helpers import file_manager


class UnitParseError(ValueError):
    """Raised when a cell/units file names a unit pint can't resolve.

    Previously the parsers printed an error and called exit(), which killed the whole process --
    fatal for a GUI. They now raise this instead; each front end catches it -- the command-line tool
    turns it into a refusal (exit 1), and the GUI surfaces it as an error dialog.
    """


# ── Per-model input files ────────────────────────────────────────────────────
def resolve_units_file(model: str) -> str:
    """
    Auto-resolve the per-model units file (no prompt): Resources/Units/<model>/units.txt.

    :raises FileNotFoundError: if the units file for this model is missing.
    """
    path = UNITS_PATH / model.lower() / "units.txt"
    if not path.exists():
        raise FileNotFoundError(f"Missing units file for model '{model}': expected {path}")
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
            problems.append(f"missing {label}(s) the bounds file requires: {', '.join(missing)}")
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
    """
    inits, param_vals, rescale_vals, forcing_vals = file_manager.parse_values_file(cell_path)
    cfg.sources["cell"] = str(cell_path)
    return cfg.inject_ground_truth(inits, param_vals, rescale_vals, forcing_vals)


# Display-only SI unit hints, indexed by forcing param name (the GUI's drive fields).
# DERIVED from config.FORCING_SI_UNITS, the authoritative conversion table, so the two cannot drift.
INFERENCE_PROMPT_UNITS = config.FORCING_DISPLAY_UNITS

# ── Cell-file parsing (shared by FDT/REDUCTION/CROSSVAL modes + the scripts) ─────────────────
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
        raise ValueError(
            f"Cell file '{cell_file}' is missing value(s) for {label} required by the bounds file: {missing}."
        )
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


def parse_cell(cell_file: str, model: str | None = None):
    """
    Parse a cell file into the 7-tuple used by the FDT/REDUCTION/CROSSVAL config builders and the
    diagnostic scripts, then run pint unit conversion.

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
        raise UnitParseError(f"{e}. Unrecognized unit in cell file '{cell_file}'.")

    time_unit = None
    for unit_str in units_dict:
        try:
            if ureg.Quantity(1, unit_str).check("[time]"):
                time_unit = unit_str
                break
        except pint.UndefinedUnitError:
            continue
    if time_unit is None:
        raise ValueError("Could not detect time unit from cell file. Ensure t_scale has a time unit.")

    s_to_cell = ureg.Quantity(1, "s").to(time_unit).magnitude
    return inits_dict, params_dict, rescale_params, force_params_dict, units_dict, si_factors, s_to_cell


# ── Units → conversion factors (standalone helper for the decoupled bounds/units path) ──────
def units_to_factors(units: tuple) -> tuple[list[float], float]:
    """
    From a set of unit strings compute (si_factors, s_to_cell). Standalone so `parse_cell` stays
    byte-for-byte untouched (it is shared by the FDT/REDUCTION/CROSSVAL builders + the scripts).
    """
    ureg = config.unit_registry()      # shared singleton: building one parses pint's full unit file
    try:
        si_factors = [ureg(unit).to_base_units().magnitude for unit in units]
    except pint.UndefinedUnitError as e:
        raise UnitParseError(f"{e}. Unrecognized unit in the units file.")
    time_unit = None
    for unit_str in units:
        try:
            if ureg.Quantity(1, unit_str).check("[time]"):
                time_unit = unit_str
                break
        except pint.UndefinedUnitError:
            continue
    if time_unit is None:
        raise ValueError("Could not detect a time unit from the units file. Include a time unit (e.g. ms).")
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
        raise ValueError("make_sim_config needs exactly one of bounds_file or bounds_dicts.")
    if bounds_dicts is not None:
        params_dict, rescale_params, force_params_dict = (OrderedDict(d) for d in bounds_dicts)
    else:
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
def make_fdt_config(model: str, state_dep_drift: bool, cell_file: str, *,
                    n_freqs: int = 60, ensemble_M: int = 256, freqs_per_batch: int = 1,
                    F0: float = 0.05) -> FDTConfig:
    """Build an FDTConfig (no prompts) from a model + cell file + FDT knobs. Shared by the command-line
    tool (core/tool) and the GUI's FDT form."""
    (inits_dict, params_dict, rescale_params, force_params_dict,
     units_dict, _, _) = parse_cell(cell_file, model=model)
    return FDTConfig(
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
    )


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


def make_param_sweep_config(cell_file: str, *, preset: dict, s_spec: tuple, t_spec: tuple,
                            n_freqs: int, ensemble_M: int, freqs_per_batch: int = 1,
                            F0: float = 0.05) -> tuple["FDTConfig", "np.ndarray", "np.ndarray"]:
    """Build (FDTConfig, s_grid, temp_grid) for the sweep study (no prompts). ``preset`` supplies the
    advanced resolution levers (freq_bounds / T_obs_periods / psd_T_obs_nd); ``s_spec``/``t_spec`` are
    (min, max, n_points). Model fixed to NADROWSKI. Shared by the command-line tool (core/tool) + the GUI."""
    import numpy as np  # local import — keep top-of-file lean
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
    )
    return cfg, s_grid, temp_grid
