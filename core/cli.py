"""
Interactive CLI prompts for the SBI pipeline.

This is the ONLY module that calls input() / print() for user interaction.
To build a GUI, replace this module with one that provides the same function signatures.
"""
import warnings
from collections import OrderedDict
from pathlib import Path

import pint

from .config import (
    SimConfig, FDTConfig, detect_device, cpu_device,
    DT_EXP_S, T_MIN_EXP_S, T_MAX_EXP_S,
    VALID_MODELS, VALID_LABELS,
    CELL_PATH, BOUNDS_PATH, UNITS_PATH, PRIOR_PATH, POSTERIOR_PATH,
)
from .Helpers import helpers, file_manager


class UnitParseError(ValueError):
    """Raised when a cell/units file names a unit pint can't resolve.

    Previously the parsers printed an error and called exit(), which killed the whole process --
    fatal for a GUI. They now raise this instead; the CLI boundary (core/__main__) catches it and
    exits cleanly, while the GUI surfaces it as an error dialog.
    """


# ── Model & cell file selection ──────────────────────────────────────────────
def select_model() -> tuple[str, list[str], bool]:
    """
    Prompt the user to choose a model.

    :return: (model_name, labels, state_dep_drift)
    """
    helpers.clear_screen()
    print("Available models:")
    for idx, model in enumerate(VALID_MODELS):
        print(f"  ({idx + 1}) {model}")
    model_num = int(input("\nWhich model would you like to run? Select a number: "))
    model = VALID_MODELS[model_num - 1]
    labels = VALID_LABELS[model_num - 1]
    state_dep_drift = "nadrowski" in model.lower()
    if model not in VALID_MODELS:
        raise ValueError(f"Invalid model selection. Please choose from {VALID_MODELS}.")
    helpers.clear_screen()
    return model, labels, state_dep_drift

def select_cell_file() -> str:
    """
    Prompt the user to choose a cell configuration file.

    :return: Full path to the chosen cell file.
    """
    print("Available cell files:")
    cell_files = file_manager.list_dir(str(CELL_PATH))
    file_num = int(input("\nFile number for model parameters: "))
    helpers.clear_screen()
    return str(CELL_PATH / cell_files[file_num - 1])

def select_bounds_file() -> str:
    """
    Prompt the user to choose a parameter-BOUNDS file (the SBI startup input).

    :return: Full path to the chosen bounds file.
    """
    print("Available bounds files:")
    bounds_files = file_manager.list_dir(str(BOUNDS_PATH))
    file_num = int(input("\nFile number for parameter BOUNDS: "))
    helpers.clear_screen()
    return str(BOUNDS_PATH / bounds_files[file_num - 1])

def resolve_units_file(model: str) -> str:
    """
    Auto-resolve the per-model units file (no prompt): Resources/Units/<model>/units.txt.

    :raises FileNotFoundError: if the units file for this model is missing.
    """
    path = UNITS_PATH / model.lower() / "units.txt"
    if not path.exists():
        raise FileNotFoundError(f"Missing units file for model '{model}': expected {path}")
    return str(path)

# ── Time / segmentation parameters ──────────────────────────────────────────
def get_time_params() -> float:
    """
    Prompt for observation duration.

    :return: T_obs_seconds
    """
    T_obs_s = float(input("Observation duration T_obs (seconds): "))
    helpers.clear_screen()
    return T_obs_s

# ── Prior / posterior selection ──────────────────────────────────────────────
def select_or_build_prior() -> tuple[str | None, bool]:
    """
    Ask the user whether to load an existing prior or build a new one.

    :return: (filename_or_None, build_new). If build_new is True, filename is None.
    """
    print("Available priors: ")
    saved = file_manager.list_dir(str(PRIOR_PATH), keep=lambda f: f.endswith(".pt"))
    try:
        if len(saved) > 0:
            idx = int(input(
                "\nWhich prior would you like to use? "
                "Select a file number ('0' if you want to make from scratch): "
            )) - 1
            if idx == -1:
                raise ValueError
            helpers.clear_screen()
            return saved[idx], False
        else:
            raise ValueError
    except ValueError:
        helpers.clear_screen()
        return None, True

def select_or_train_posterior() -> tuple[str | None, bool]:
    """
    Ask the user whether to load an existing posterior or train a new one.

    :return: (filename_or_None, train_new). If train_new is True, filename is None.
    """
    print("Available posteriors: ")
    # Show only loadable posteriors -- hide the .rot.pt reparam sidecars and .loss.npz curves that
    # live alongside each <name>.pt (picking one would fail to load).
    saved = file_manager.list_dir(str(POSTERIOR_PATH),
                                  keep=lambda f: f.endswith(".pt") and not f.endswith(".rot.pt"))
    try:
        if len(saved) > 0:
            idx = int(input(
                "\nWhich posterior would you like to use? "
                "Select a file number (or '0' if you would like to make it from scratch): "
            )) - 1
            if idx == -1:
                raise ValueError
            helpers.clear_screen()
            return saved[idx], False
        else:
            raise ValueError
    except ValueError:
        helpers.clear_screen()
        return None, True

def prompt_save_name(artifact_type: str) -> str:
    """
    Ask the user for a filename when saving a prior or posterior.

    :param artifact_type: Human-readable label, e.g. "prior" or "posterior".
    :return: The name entered by the user (without extension).
    """
    return input(f"Enter a name for the {artifact_type} file: ")


# ── Post-training inference ─────────────────────────────────────────────────
def select_inference_mode() -> str:
    """
    Ask whether / how to infer after training + calibration.

    :return: "simulated" (a cell-file ground-truth point), "experimental" (real recordings), or "none".
    """
    print("\nInfer on a dataset now?")
    print("  (1) Simulated dataset  (a cell file's ground-truth point)")
    print("  (2) Experimental data  (real recordings)")
    print("  (3) Neither            (stop after calibration)")
    choice = input("\nSelect a number: ").strip()
    helpers.clear_screen()
    return {"1": "simulated", "2": "experimental", "3": "none"}.get(choice, "none")

def load_and_validate_gt(cfg: SimConfig, cell_path: str) -> None:
    """
    Parse a cell file's VALUES + initial conditions and inject them into cfg, validating (via
    SimConfig.inject_ground_truth) that the ND/rescale values lie within the bounds file's bounds.
    """
    inits, param_vals, rescale_vals, forcing_vals = file_manager.parse_values_file(cell_path)
    cfg.inject_ground_truth(inits, param_vals, rescale_vals, forcing_vals)


# Display-only SI unit hints, indexed by forcing param name. Used to label the
# CLI prompt; the authoritative SI-unit map lives in orchestrator._FORCING_SI_UNITS.
_INFERENCE_PROMPT_UNITS = {
    "amp":    "N",
    "amp_y":  "N",   # Hopf y-channel amplitude (shares freq/phase/offset with x)
    "freq":   "Hz",
    "phase":  "rad",
    "offset": "N",
}

def get_inference_inputs(force_param_names: list[str]) -> tuple[str, str, float, dict]:
    """
    Prompt for the inputs needed to run inference on real experimental data.

    All inputs are in SI units; conversion to cell file units happens in the
    caller via SimConfig.get_unit_conversion_factor().

    :param force_param_names: Forcing parameter names from the cell file (e.g.
                              ["amp", "freq", "phase", "offset"] for Nadrowski/BP, or
                              ["amp", "amp_y", "freq", "phase", "offset"] for Hopf).
    :return: (spont_path, forced_path, T_obs_seconds, forcing_params_si). The forcing dict
             has one entry per name in force_param_names.
    """
    spont_path = input("Path to SPONTANEOUS (unforced) recording (.csv or .npy): ").strip()
    forced_path = input("Path to FORCED (driven) recording (.csv or .npy): ").strip()
    T_obs_s = float(input("Observation duration T_obs (seconds): "))
    print("\nForcing parameters (in SI units):")
    forcing_params_si: dict = {}
    for name in force_param_names:
        unit = _INFERENCE_PROMPT_UNITS.get(name, "")
        unit_str = f" ({unit})" if unit else ""
        forcing_params_si[name] = float(input(f"  {name}{unit_str}: "))
    helpers.clear_screen()
    return spont_path, forced_path, T_obs_s, forcing_params_si

# ── Mode selection (top-level) ──────────────────────────────────────────────
def select_mode() -> str:
    """
    Top-level prompt: which analysis mode to run.

    :return: "FDT", "SBI", "REDUCTION", or "CROSSVAL".
    """
    helpers.clear_screen()
    print("Available analysis modes:")
    print("  (1) FDT analysis")
    print("  (2) SBI parameter fitting")
    print("  (3) NWK→Hopf reduction map")
    print("  (4) FDT parameter-sweep study (S and T_a/T)")
    choice_str = input("\nWhich mode? Select a number: ").strip()
    helpers.clear_screen()
    if choice_str == "1":
        return "FDT"
    if choice_str == "2":
        return "SBI"
    if choice_str == "3":
        return "REDUCTION"
    if choice_str == "4":
        return "CROSSVAL"
    raise ValueError(f"Invalid mode selection: {choice_str}.")


# ── Small input helpers ─────────────────────────────────────────────────────
def _prompt_int(label: str, default: int) -> int:
    ans = input(f"{label} [{default}]: ").strip()
    return int(ans) if ans else default

def _prompt_float(label: str, default: float) -> float:
    ans = input(f"{label} [{default}]: ").strip()
    return float(ans) if ans else default


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


def _parse_cell(cell_file: str, model: str | None = None):
    """
    Parse a cell file into the 7-tuple used by the FDT/REDUCTION/CROSSVAL config builders and the
    diagnostic scripts, then run pint unit conversion.

    Cell files hold VALUES only (bounds + units are decoupled) and live in per-model subfolders:
    Resources/Cells/<model>/<cell>.txt. If the sibling Resources/Bounds/<model>/<cell>.txt AND
    Resources/Units/<model>/units.txt both exist, use the DECOUPLED path: the bounds file defines the
    param set + order, the cell supplies the values, units come from the units file. Otherwise fall
    back to the legacy parse_model_file (bounds + units read inline from the cell).

    :param cell_file: path to the cell file (Resources/Cells/<model>/<cell>.txt).
    :param model: model name for resolving the bounds/units files; derived from the cell's parent
                  folder (e.g. '.../Cells/nadrowski/cell_2.txt' -> 'nadrowski') when None.
    :return: (inits_dict, params_dict, rescale_params, force_params_dict,
             units_dict, si_factors, s_to_cell)
    """
    p = Path(cell_file)
    if model is None:
        model = p.parent.name
    model = model.lower()

    bounds_path = BOUNDS_PATH / model / p.name       # Bounds/<model>/<cell>.txt (sibling of the cell)
    units_path = UNITS_PATH / model / "units.txt"    # Units/<model>/units.txt

    if bounds_path.exists() and units_path.exists():
        # Decoupled path: bounds file = param set + order; cell = values; units file = units.
        b_params, b_rescale, b_forcing, _ = file_manager.parse_bounds_file(str(bounds_path))
        inits_dict, v_params, v_rescale, v_forcing = file_manager.parse_values_file(cell_file)
        params_dict = _merge_vals_bounds(v_params, b_params, "ND parameters", cell_file)
        rescale_params = _merge_vals_bounds(v_rescale, b_rescale, "rescale parameters", cell_file)
        force_params_dict = _merge_vals_bounds(v_forcing, b_forcing, "forcing parameters", cell_file)
        units_dict = file_manager.parse_units_file(str(units_path))
        si_factors, s_to_cell = _units_to_factors(units_dict)
        return inits_dict, params_dict, rescale_params, force_params_dict, units_dict, si_factors, s_to_cell

    # ── Legacy fallback: bounds + units read inline from the cell (models without decoupled files) ──
    inits_dict, params_dict, rescale_params, force_params_dict, units_dict = file_manager.parse_model_file(cell_file)

    ureg = pint.UnitRegistry()
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
def _units_to_factors(units: tuple) -> tuple[list[float], float]:
    """
    From a set of unit strings compute (si_factors, s_to_cell). Standalone so `_parse_cell` stays
    byte-for-byte untouched (it is shared by the FDT/REDUCTION/CROSSVAL builders + the scripts).
    """
    ureg = pint.UnitRegistry()
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


# ── Pure config cores (no prompts) — shared by the CLI builders below and the GUI ────────────
def make_sim_config(model: str, labels: list[str], state_dep_drift: bool, bounds_file: str) -> SimConfig:
    """
    Build a bounds-only SimConfig (no prompts) from a chosen model + bounds file. Ground-truth values,
    initial conditions, and T_obs are filled later (only for simulated inference). Shared by
    build_sim_config (CLI) and the GUI's SBI config form.
    """
    units_file = resolve_units_file(model)
    params_dict, rescale_params, force_params_dict, _ = file_manager.parse_bounds_file(bounds_file)
    units_dict = file_manager.parse_units_file(units_file)
    si_factors, s_to_cell = _units_to_factors(units_dict)

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
        si_factors=si_factors,
        dt_exp=DT_EXP_S * s_to_cell,
        t_min_exp=T_MIN_EXP_S * s_to_cell,
        t_max_exp=T_MAX_EXP_S * s_to_cell,
        T_obs=None,                             # observation duration is prompted at the inference step
        hw=detect_device(),
    )


# ── Top-level config builder (SBI mode) ─────────────────────────────────────
def build_sim_config() -> SimConfig:
    """
    Interactive setup for SBI parameter fitting. Prompts ONLY for a model and a parameter-BOUNDS file;
    the units file is auto-resolved per model. No cell file and no T_obs at startup — ground-truth
    values, initial conditions, and T_obs are supplied later, only if the user chooses to infer on a
    simulated observation (see orchestrator.run + cli.load_and_validate_gt).
    """
    model, labels, state_dep_drift = select_model()
    bounds_file = select_bounds_file()
    return make_sim_config(model, labels, state_dep_drift, bounds_file)


# ── Pure config core (FDT) ───────────────────────────────────────────────────
def make_fdt_config(model: str, state_dep_drift: bool, cell_file: str, *,
                    n_freqs: int = 60, ensemble_M: int = 256, freqs_per_batch: int = 1,
                    F0: float = 0.05) -> FDTConfig:
    """Build an FDTConfig (no prompts) from a model + cell file + FDT knobs. Shared by build_fdt_config
    (CLI) and the GUI's FDT form."""
    (inits_dict, params_dict, rescale_params, force_params_dict,
     units_dict, si_factors, _) = _parse_cell(cell_file, model=model)
    return FDTConfig(
        model=model,
        state_dep_drift=state_dep_drift,
        inits_dict=inits_dict,
        params_dict=params_dict,
        rescale_params=rescale_params,
        force_params_dict=force_params_dict,
        units_dict=units_dict,
        si_factors=si_factors,
        n_freqs=n_freqs,
        ensemble_M=ensemble_M,
        freqs_per_batch=freqs_per_batch,
        F0=F0,
        hw=cpu_device(),  # FDT: sequential SDE loop at M~256 is ~3.4x faster on CPU than GPU
    )


# ── Top-level config builder (FDT mode) ─────────────────────────────────────
def build_fdt_config() -> FDTConfig:
    """
    Interactive setup for FDT analysis. Prompts for model and cell file like the
    SBI mode, then for FDT-specific knobs (n_freqs, ensemble_M, F0, freqs_per_batch).
    """
    model, _labels, state_dep_drift = select_model()
    cell_file = select_cell_file()

    print("\nFDT knobs (press Enter to accept default):")
    n_freqs = _prompt_int("  n_freqs", 60)
    ensemble_M = _prompt_int("  ensemble_M", 256)
    freqs_per_batch = _prompt_int("  freqs_per_batch (Campaign 2 packing)", 1)
    F0 = _prompt_float("  F0 (ND forcing amplitude)", 0.05)
    helpers.clear_screen()

    return make_fdt_config(model, state_dep_drift, cell_file, n_freqs=n_freqs,
                           ensemble_M=ensemble_M, freqs_per_batch=freqs_per_batch, F0=F0)


# ── Top-level config builder (Reduction-map mode) ────────────────────────────
def build_reduction_config() -> FDTConfig:
    """
    Interactive setup for the NWK→Hopf analytical reduction map.

    The reduction map is Nadrowski-specific by construction, so the model is
    fixed to NADROWSKI. Only the cell file (which carries the ND parameters
    and dimensional rescaling factors) and an optional forcing amplitude F0
    need to be supplied — no FDT-specific simulation knobs are relevant here.
    """
    print("Reduction map: model fixed to NADROWSKI (NWK→Hopf reduction).")
    cell_file = select_cell_file()

    print("\nReduction-map knobs (press Enter to accept default):")
    F0 = _prompt_float("  F0 (NWK forcing amplitude for Phase B1)", 0.05)
    helpers.clear_screen()

    return make_reduction_config(cell_file, F0=F0)


def make_reduction_config(cell_file: str, *, F0: float = 0.05) -> FDTConfig:
    """Build a reduction-map FDTConfig (no prompts) from a cell file. Model is fixed to NADROWSKI
    (the reduction is Nadrowski-specific). Shared by build_reduction_config (CLI) and the GUI form."""
    (inits_dict, params_dict, rescale_params, force_params_dict,
     units_dict, si_factors, _) = _parse_cell(cell_file, model="NADROWSKI")
    return FDTConfig(
        model="NADROWSKI",
        state_dep_drift=True,
        inits_dict=inits_dict,
        params_dict=params_dict,
        rescale_params=rescale_params,
        force_params_dict=force_params_dict,
        units_dict=units_dict,
        si_factors=si_factors,
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
_SWEEP_PRESETS = {
    "exploratory": dict(freq_bounds=(0.2, 30.0), n_freqs=30, T_obs_periods=20,
                        psd_T_obs_nd=4000.0, ensemble_M=256, points=8),
    "production":  dict(freq_bounds=(0.1, 30.0), n_freqs=60, T_obs_periods=30,
                        psd_T_obs_nd=8000.0, ensemble_M=256, points=12),
}

def _select_sweep_preset() -> dict:
    """
    Prompt for the sweep-study resolution preset. Defaults to exploratory.

    :return: the chosen preset's knob dict (a copy of the _SWEEP_PRESETS entry).
    """
    print("\nSweep preset:")
    print("  (1) Exploratory — fast/coarse; confirm the restoration trend (~3-4x faster)")
    print("  (2) Production  — full resolution")
    choice = input("\nWhich preset? Select a number [1]: ").strip() or "1"
    name = "production" if choice == "2" else "exploratory"
    print(f"Using the {name} preset.")
    return dict(_SWEEP_PRESETS[name])


# ── Top-level config builder (FDT parameter-sweep study) ─────────────────────
def build_param_sweep_config() -> tuple["FDTConfig", "np.ndarray", "np.ndarray"]:
    """
    Interactive setup for the FDT parameter-sweep study.

    Two sweeps probe FDT restoration on the Nadrowski model:
      - S sweep  (T_a/T = 1 held): vary S; FDT restored as S -> 0.
      - T sweep  (S = 0 held):     vary T_a/T; FDT restored as T_a/T -> 1.

    Returns (cfg, s_grid, temp_grid). cfg carries NWK params + FDT knobs.
    """
    print("FDT parameter-sweep study: model fixed to NADROWSKI.")

    cell_file = select_cell_file()
    (_i, params_dict, _r, _f, _u, _si, _s) = _parse_cell(cell_file, model="NADROWSKI")
    cell_s = params_dict["s"][0]
    cell_temp = params_dict["temp"][0]
    print(f"\nCell-file values: S = {cell_s:.4f},  T_a/T = {cell_temp:.4f}")

    # Resolution preset (drives the FDT knobs + default grid density for BOTH sweeps).
    preset = _select_sweep_preset()

    print("\nS sweep grid (T_a/T held at 1; FDT restored as S -> 0):")
    s_min = _prompt_float("  S_min", 0.0)
    s_max = _prompt_float("  S_max", cell_s)
    s_points = _prompt_int("  S n_points", preset["points"])

    print("\nT_a/T sweep grid (S held at 0; FDT restored as T_a/T -> 1):")
    t_min = _prompt_float("  T_min", 1.0)
    t_max = _prompt_float("  T_max", cell_temp)
    t_points = _prompt_int("  T n_points", preset["points"])

    print("\nFDT knobs (press Enter to accept preset default):")
    n_freqs = _prompt_int("  n_freqs", preset["n_freqs"])
    ensemble_M = _prompt_int("  ensemble_M", preset["ensemble_M"])
    freqs_per_batch = _prompt_int("  freqs_per_batch (Campaign 2 packing)", 1)
    F0 = _prompt_float("  F0 (ND forcing amplitude)", 0.05)
    print(f"  freq_bounds={preset['freq_bounds']}, T_obs_periods={preset['T_obs_periods']}, "
          f"psd_T_obs_nd={preset['psd_T_obs_nd']}  (from preset)")
    helpers.clear_screen()

    return make_param_sweep_config(
        cell_file, preset=preset, s_spec=(s_min, s_max, s_points), t_spec=(t_min, t_max, t_points),
        n_freqs=n_freqs, ensemble_M=ensemble_M, freqs_per_batch=freqs_per_batch, F0=F0)


def make_param_sweep_config(cell_file: str, *, preset: dict, s_spec: tuple, t_spec: tuple,
                            n_freqs: int, ensemble_M: int, freqs_per_batch: int = 1,
                            F0: float = 0.05) -> tuple["FDTConfig", "np.ndarray", "np.ndarray"]:
    """Build (FDTConfig, s_grid, temp_grid) for the sweep study (no prompts). ``preset`` supplies the
    advanced resolution levers (freq_bounds / T_obs_periods / psd_T_obs_nd); ``s_spec``/``t_spec`` are
    (min, max, n_points). Model fixed to NADROWSKI. Shared by build_param_sweep_config (CLI) + the GUI."""
    import numpy as np  # local import — keep top-of-file lean
    (inits_dict, params_dict, rescale_params, force_params_dict,
     units_dict, si_factors, _) = _parse_cell(cell_file, model="NADROWSKI")
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
        si_factors=si_factors,
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
