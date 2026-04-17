"""
Interactive CLI prompts for the SBI pipeline.

This is the ONLY module that calls input() / print() for user interaction.
To build a GUI, replace this module with one that provides the same function signatures.
"""
import warnings

import pint

from .config import (
    SimConfig, detect_device,
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


def get_inference_inputs() -> tuple[str, float, dict]:
    """
    Prompt for the inputs needed to run inference on real experimental data.

    All inputs are in SI units; conversion to cell file units happens in the
    caller via SimConfig.get_unit_conversion_factor().

    :return: (data_file_path, T_obs_seconds, forcing_params_si)
             forcing_params_si has keys "amp" (N), "freq" (Hz), "phase" (rad), "offset" (N).
    """
    data_path = input("Path to experimental data file (.csv or .npy): ").strip()
    T_obs_s = float(input("Observation duration T_obs (seconds): "))
    print("\nForcing parameters (in SI units):")
    amp = float(input("  Amplitude (N): "))
    freq = float(input("  Frequency (Hz): "))
    phase = float(input("  Phase (rad): "))
    offset = float(input("  Offset (N): "))
    helpers.clear_screen()
    forcing_params_si = {"amp": amp, "freq": freq, "phase": phase, "offset": offset}
    return data_path, T_obs_s, forcing_params_si

# ── Top-level config builder ────────────────────────────────────────────────
def build_sim_config() -> SimConfig:
    """
    Run the full interactive setup flow and return a populated SimConfig.

    Steps:
      1. Select model
      2. Select cell file & parse parameters
      3. Convert units to SI
      4. Prompt for time / segmentation params
    """
    model, labels, state_dep_drift = select_model()
    cell_file = select_cell_file()

    inits_dict, params_dict, rescale_params, force_params_dict, units_dict = file_manager.parse_model_file(cell_file)

    # unit conversion
    ureg = pint.UnitRegistry()
    try:
        si_factors = [ureg(unit).to_base_units().magnitude for unit in units_dict]
    except pint.UndefinedUnitError as e:
        print(f"Error: {e}. Unrecognized units.")
        exit()

    # detect time unit from cell file (find which unit has time dimensionality)
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

    # convert experimental constants from seconds to cell file time units
    s_to_cell = ureg.Quantity(1, "s").to(time_unit).magnitude
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
