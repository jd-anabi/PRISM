"""
Interactive CLI prompts for the SBI pipeline.

This is the ONLY module that calls input() / print() for user interaction.
To build a GUI, replace this module with one that provides the same function signatures.
"""
import pint

from .config import (
    SimConfig, detect_device,
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
def get_time_params() -> tuple[float, float, float, int]:
    """
    Prompt for simulation time parameters.

    :return: (t_max, dt, steady_pct, n_segs)
    """
    t_max = int(input("Max time: "))
    dt = float(input("Time step: "))
    steady_pct = float(input("Percentage of data that is transient (%): ").replace("%", "")) / 100.0
    n_segs = int(input("Number of segments to divide time series into: "))
    helpers.clear_screen()
    return float(t_max), dt, steady_pct, n_segs

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

    # time parameters
    t_max, dt, steady_pct, n_segs = get_time_params()

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
        t_max=t_max,
        dt=dt,
        steady_pct=steady_pct,
        n_segs=n_segs,
        hw=detect_device(),
    )
