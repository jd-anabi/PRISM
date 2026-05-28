"""
Interactive CLI prompts for the SBI pipeline.

This is the ONLY module that calls input() / print() for user interaction.
To build a GUI, replace this module with one that provides the same function signatures.
"""
import warnings

import pint

from .config import (
    SimConfig, FDTConfig, detect_device, cpu_device,
    DT_EXP_S, T_MIN_EXP_S, T_MAX_EXP_S,
    VALID_MODELS, VALID_LABELS,
    CELL_PATH, PRIOR_PATH, POSTERIOR_PATH,
)
from .Helpers import helpers, file_manager

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
    saved = file_manager.list_dir(str(PRIOR_PATH))
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
    saved = file_manager.list_dir(str(POSTERIOR_PATH))
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


# ── Inference on real experimental data ────────────────────────────────────
def select_or_skip_inference() -> bool:
    """
    Ask whether to run inference on a real experimental recording.

    :return: True if user wants to run inference, False to skip.
    """
    response = input("\nRun inference on real experimental data? (y/N): ").strip().lower()
    helpers.clear_screen()
    return response in ("y", "yes")


# Display-only SI unit hints, indexed by forcing param name. Used to label the
# CLI prompt; the authoritative SI-unit map lives in orchestrator._FORCING_SI_UNITS.
_INFERENCE_PROMPT_UNITS = {
    "amp":    "N",
    "amp_y":  "N",   # Hopf y-channel amplitude (shares freq/phase/offset with x)
    "freq":   "Hz",
    "phase":  "rad",
    "offset": "N",
}

def get_inference_inputs(force_param_names: list[str]) -> tuple[str, float, dict]:
    """
    Prompt for the inputs needed to run inference on real experimental data.

    All inputs are in SI units; conversion to cell file units happens in the
    caller via SimConfig.get_unit_conversion_factor().

    :param force_param_names: Forcing parameter names from the cell file (e.g.
                              ["amp", "freq", "phase", "offset"] for Nadrowski/BP, or
                              ["amp", "amp_y", "freq", "phase", "offset"] for Hopf).
    :return: (data_file_path, T_obs_seconds, forcing_params_si). The forcing dict has
             one entry per name in force_param_names.
    """
    data_path = input("Path to experimental data file (.csv or .npy): ").strip()
    T_obs_s = float(input("Observation duration T_obs (seconds): "))
    print("\nForcing parameters (in SI units):")
    forcing_params_si: dict = {}
    for name in force_param_names:
        unit = _INFERENCE_PROMPT_UNITS.get(name, "")
        unit_str = f" ({unit})" if unit else ""
        forcing_params_si[name] = float(input(f"  {name}{unit_str}: "))
    helpers.clear_screen()
    return data_path, T_obs_s, forcing_params_si

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


# ── Cell-file parsing (shared by SBI and FDT modes) ─────────────────────────
def _parse_cell(cell_file: str):
    """
    Parse a cell file and run pint unit conversion.

    :param cell_file: path to the cell file.
    :return: (inits_dict, params_dict, rescale_params, force_params_dict,
             units_dict, si_factors, s_to_cell)
    """
    inits_dict, params_dict, rescale_params, force_params_dict, units_dict = file_manager.parse_model_file(cell_file)

    ureg = pint.UnitRegistry()
    try:
        si_factors = [ureg(unit).to_base_units().magnitude for unit in units_dict]
    except pint.UndefinedUnitError as e:
        print(f"Error: {e}. Unrecognized units.")
        exit()

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


# ── Top-level config builder (SBI mode) ─────────────────────────────────────
def build_sim_config() -> SimConfig:
    """
    Run the full interactive setup flow for SBI parameter fitting and return a populated SimConfig.

    Steps:
      1. Select model
      2. Select cell file & parse parameters
      3. Convert units to SI
      4. Prompt for time / segmentation params
    """
    model, labels, state_dep_drift = select_model()
    cell_file = select_cell_file()

    (inits_dict, params_dict, rescale_params, force_params_dict,
     units_dict, si_factors, s_to_cell) = _parse_cell(cell_file)

    # convert experimental constants from seconds to cell file time units
    dt_exp = DT_EXP_S * s_to_cell
    t_min_exp = T_MIN_EXP_S * s_to_cell
    t_max_exp = T_MAX_EXP_S * s_to_cell

    # time / observation parameters
    T_obs_s = get_time_params()
    T_obs = T_obs_s * s_to_cell

    # Check T_obs against training range (warn if out of distribution)
    if T_obs_s < T_MIN_EXP_S:
        warnings.warn(
            f"T_obs={T_obs_s:.2f}s is below the training range minimum "
            f"T_MIN_EXP_S={T_MIN_EXP_S:.2f}s. The network has not been trained "
            f"on recordings this short and may extrapolate poorly.",
            stacklevel=2,
        )
    elif T_obs_s > T_MAX_EXP_S:
        warnings.warn(
            f"T_obs={T_obs_s:.2f}s exceeds the training range maximum "
            f"T_MAX_EXP_S={T_MAX_EXP_S:.2f}s. The network has not been trained "
            f"on recordings this long and may extrapolate poorly.",
            stacklevel=2,
        )

    return SimConfig(
        model=model,
        labels=labels,
        state_dep_drift=state_dep_drift,
        inits_dict=inits_dict,
        params_dict=params_dict,
        rescale_params=rescale_params,
        force_params_dict=force_params_dict,
        units_dict=units_dict,
        si_factors=si_factors,
        dt_exp=dt_exp,
        t_min_exp=t_min_exp,
        t_max_exp=t_max_exp,
        T_obs=T_obs,
        hw=detect_device(),
    )


# ── Top-level config builder (FDT mode) ─────────────────────────────────────
def build_fdt_config() -> FDTConfig:
    """
    Interactive setup for FDT analysis. Prompts for model and cell file like the
    SBI mode, then for FDT-specific knobs (n_freqs, ensemble_M, F0, freqs_per_batch).
    """
    model, _labels, state_dep_drift = select_model()
    cell_file = select_cell_file()

    (inits_dict, params_dict, rescale_params, force_params_dict,
     units_dict, si_factors, _) = _parse_cell(cell_file)

    print("\nFDT knobs (press Enter to accept default):")
    n_freqs = _prompt_int("  n_freqs", 60)
    ensemble_M = _prompt_int("  ensemble_M", 256)
    freqs_per_batch = _prompt_int("  freqs_per_batch (Campaign 2 packing)", 1)
    F0 = _prompt_float("  F0 (ND forcing amplitude)", 0.05)
    helpers.clear_screen()

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


# ── Top-level config builder (Reduction-map mode) ────────────────────────────
def build_reduction_config() -> FDTConfig:
    """
    Interactive setup for the NWK→Hopf analytical reduction map.

    The reduction map is Nadrowski-specific by construction, so the model is
    fixed to NADROWSKI. Only the cell file (which carries the ND parameters
    and dimensional rescaling factors) and an optional forcing amplitude F0
    need to be supplied — no FDT-specific simulation knobs are relevant here.
    """
    model = "NADROWSKI"
    state_dep_drift = True
    print("Reduction map: model fixed to NADROWSKI (NWK→Hopf reduction).")

    cell_file = select_cell_file()

    (inits_dict, params_dict, rescale_params, force_params_dict,
     units_dict, si_factors, _) = _parse_cell(cell_file)

    print("\nReduction-map knobs (press Enter to accept default):")
    F0 = _prompt_float("  F0 (NWK forcing amplitude for Phase B1)", 0.05)
    helpers.clear_screen()

    return FDTConfig(
        model=model,
        state_dep_drift=state_dep_drift,
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
    import numpy as np  # local import — keep top-of-file lean
    model = "NADROWSKI"
    state_dep_drift = True
    print("FDT parameter-sweep study: model fixed to NADROWSKI.")

    cell_file = select_cell_file()
    (inits_dict, params_dict, rescale_params, force_params_dict,
     units_dict, si_factors, _) = _parse_cell(cell_file)

    cell_s = params_dict["s"][0]
    cell_temp = params_dict["temp"][0]
    print(f"\nCell-file values: S = {cell_s:.4f},  T_a/T = {cell_temp:.4f}")

    # Resolution preset (drives the FDT knobs + default grid density for BOTH sweeps).
    preset = _select_sweep_preset()

    print("\nS sweep grid (T_a/T held at 1; FDT restored as S -> 0):")
    s_min = _prompt_float("  S_min", 0.0)
    s_max = _prompt_float("  S_max", cell_s)
    s_points = _prompt_int("  S n_points", preset["points"])
    s_grid = np.linspace(s_min, s_max, s_points)

    print("\nT_a/T sweep grid (S held at 0; FDT restored as T_a/T -> 1):")
    t_min = _prompt_float("  T_min", 1.0)
    t_max = _prompt_float("  T_max", cell_temp)
    t_points = _prompt_int("  T n_points", preset["points"])
    temp_grid = np.linspace(t_min, t_max, t_points)

    print("\nFDT knobs (press Enter to accept preset default):")
    n_freqs = _prompt_int("  n_freqs", preset["n_freqs"])
    ensemble_M = _prompt_int("  ensemble_M", preset["ensemble_M"])
    freqs_per_batch = _prompt_int("  freqs_per_batch (Campaign 2 packing)", 1)
    F0 = _prompt_float("  F0 (ND forcing amplitude)", 0.05)
    # Advanced resolution levers taken directly from the preset (not prompted).
    freq_bounds = preset["freq_bounds"]
    T_obs_periods = preset["T_obs_periods"]
    psd_T_obs_nd = preset["psd_T_obs_nd"]
    print(f"  freq_bounds={freq_bounds}, T_obs_periods={T_obs_periods}, "
          f"psd_T_obs_nd={psd_T_obs_nd}  (from preset)")
    helpers.clear_screen()

    cfg = FDTConfig(
        model=model,
        state_dep_drift=state_dep_drift,
        inits_dict=inits_dict,
        params_dict=params_dict,
        rescale_params=rescale_params,
        force_params_dict=force_params_dict,
        units_dict=units_dict,
        si_factors=si_factors,
        n_freqs=n_freqs,
        freq_bounds=freq_bounds,
        ensemble_M=ensemble_M,
        freqs_per_batch=freqs_per_batch,
        F0=F0,
        T_obs_periods=T_obs_periods,
        psd_T_obs_nd=psd_T_obs_nd,
        hw=cpu_device(),  # sweep: sequential SDE loop at M~256 is ~3.4x faster on CPU than GPU
    )
    return cfg, s_grid, temp_grid
