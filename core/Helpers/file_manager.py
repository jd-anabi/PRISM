import os
import re
from collections import OrderedDict

import torch

# --- Regex Definitions ---
# Float Value (Scientific Notation)
FLOAT_REGEX = r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?'
# Flexible Assignment: name [opt_units] = val
ASSIGNMENT_PATTERN = re.compile(fr'^\s*(?P<name>\w+)\s*(?:\(\s*(?P<units>[^)]+)\s*\))?\s*=\s*(?P<val>{FLOAT_REGEX})\s*$')
# Flexible Bounds: name [opt_units] = val [opt_in] (bounds)
BOUNDS_PATTERN = re.compile(fr'^\s*(?P<name>\w+)\s*(?:\(\s*(?P<units>[^)]+)\s*\))?\s*=\s*(?P<val>{FLOAT_REGEX})\s+(?:in\s+)?[\[\(](?P<tup>.*?)[\]\)]\s*$')

def parse_model_file(file_name: str) -> tuple:
    """
    Parses a model configuration file to extract initialization variables, parameters, rescaling values,
    forcing parameters, and associated unit types. The function processes a file with sections defined
    by specific headers, and categorizes the data into corresponding dictionaries or structures.

    :param file_name: The path to the model file to be parsed.
    :return: A tuple containing extracted model data.
        - ``init_conditions``: An ordered dictionary of initial conditions mapping variable names to their values.
        - ``parameters``: An ordered dictionary where each key is a parameter name, and the value is a tuple of
          its initial value and bounds.
        - ``forcing_params``: An ordered dictionary of time-dependent forcing parameters.
        - ``collected_units``: A tuple of unit strings found during processing.
        If `nd` is True, the tuple includes:
        - ``init_conditions``: An ordered dictionary of initial conditions.
        - ``parameters``: Parameter data with values and bounds.
        - ``rescale_params``: Rescaling data for specific variables.
        - ``forcing_params``: Forcing parameter data.
        - ``collected_units``: Unit strings found during processing.
    :rtype: tuple
    """
    # --- Data Structures ---
    init_conditions = OrderedDict()
    parameters = OrderedDict()  # Format: {name: (val, (min, max))}
    rescale_params = OrderedDict()
    forcing_params = OrderedDict()
    collected_units = set()

    # --- State/Section Management ---
    current_section = None

    # split string content into lines (simulating file read)
    try:
        with open(file_name, 'r', encoding='utf-8') as file:
            lines = file.read().strip().split('\n')
    except FileNotFoundError:
        raise FileNotFoundError("File not found")

    def process_units(match_obj):
        if match_obj.group('units'):
            raw_units = match_obj.group('units').split()
            for u in raw_units:
                base_unit = u.split('^')[0]
                collected_units.add(base_unit)

    for line in lines:
        line = line.strip()
        if not line:
            continue  # skip empty lines

        # --- Section Detection ---
        if line.startswith("#"):
            if "Initial Conditions" in line:
                current_section = "INIT"
            elif "Parameters" in line and "Forcing" not in line:
                if line.startswith("# Dimensional"):
                    current_section = "RESCALE"
                else:
                    current_section = "PARAM"
            elif "Forcing Parameters" in line:
                current_section = "FORCING"
            continue

        # 1. Initial Conditions (Using ASSIGNMENT_PATTERN)
        if current_section == "INIT":
            match = ASSIGNMENT_PATTERN.search(line)
            if match:
                init_conditions[match.group('name')] = float(match.group('val'))
                process_units(match)

        # 2. Parameters, Forcing, Rescale (Using BOUNDS_PATTERN)
        elif current_section in ["PARAM", "FORCING", "RESCALE"]:
            match = BOUNDS_PATTERN.search(line)
            if match:
                name = match.group('name')
                val = float(match.group('val'))
                bounds = tuple(float(x) for x in re.findall(FLOAT_REGEX, match.group('tup')))
                if current_section == "PARAM":
                    target_dict = parameters
                elif current_section == "FORCING":
                    target_dict = forcing_params
                else:
                    target_dict = rescale_params
                target_dict[name] = (val, bounds)
                process_units(match)

    return init_conditions, parameters, rescale_params, forcing_params, tuple(collected_units)

def list_dir(files_dir: str, return_list: bool = True) -> list[str] | list[None]:
    """
    Lists all files in the specified directory and its subdirectories, with an option to return a list of files.

    The function walks through the directory tree starting from the given directory. It prints the directory structure
    with files ordered and numbered. Optionally, it returns a list of all files found.

    :param files_dir: Path to the directory that needs to be traversed.
    :type files_dir: str
    :param return_list: A flag indicating whether to return the list of files. If True, the list of files is returned.
        Default is True.
    :type return_list: bool
    :return: A list of all files in the directory and its subdirectories if `return_list` is True; otherwise, None.
    :rtype: list[str] | None
    """
    # list files in directory
    model_files = [""]
    file_num = 1
    for root, dirs, files in os.walk(files_dir):
        level = root.replace(files_dir, "").count(os.sep)
        indent = " " * 2 * level
        print(f"{indent}{os.path.basename(root)}")
        subindent = " " * 2 * (level + 1)
        for file in files:
            model_files.append(file)
            print(f"{subindent}({file_num}) {file}")
            file_num += 1
    model_files.pop(0)
    if return_list:
        return model_files
    return []

def save_mix_dist(dist: torch.distributions.MixtureSameFamily, filename: str):
    """
    Saves the parameters of a mixture of distributions to a file, including the
    means, covariances, and weights of the mixture components. The data is saved
    in a serialized format using PyTorch.

    :param dist: A mixture distribution of type `torch.distributions.MixtureSameFamily`.
    :param filename: The file path where the mixture distribution should be saved.
    :type filename: str
    :return: None
    """
    data_to_save = {'means': dist.component_distribution.loc, 'covariances': dist.component_distribution.covariance_matrix, 'weights': dist.mixture_distribution.probs}
    torch.save(data_to_save, filename)

def load_mix_dist(filename: str, device: torch.device = torch.device('cpu')) -> torch.distributions.MixtureSameFamily:
    """
    Loads a pre-saved mixture distribution from a file and reconstructs it using
    torch.distributions. The file is expected to include means, covariances, and weights
    that define a `MixtureSameFamily` distribution. This utility ensures the loaded
    distribution is ready for further computations or sampling.

    :param filename: The path to the file containing the serialized mixture distribution.
    :param device: The device on which the loaded data should be placed. Defaults to CPU.
    :return: A reconstructed torch.distributions.MixtureSameFamily object representing
        the mixture distribution.
    :rtype: torch.distributions.MixtureSameFamily
    """
    data = torch.load(filename, map_location=device)
    means = data['means']
    covs = data['covariances']
    weights = data['weights']

    # reconstruct the distribution
    comp_dist = torch.distributions.MultivariateNormal(means, covariance_matrix=covs)
    mix_dist = torch.distributions.Categorical(probs=weights)
    prior = torch.distributions.MixtureSameFamily(mix_dist, comp_dist)

    return prior
