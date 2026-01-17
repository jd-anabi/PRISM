import re

import torch

# --- Regex Definitions ---
# Float Value (Scientific Notation)
FLOAT_REGEX = r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?'
# Flexible Assignment: name [opt_units] = val
ASSIGNMENT_PATTERN = re.compile(fr'^\s*(?P<name>\w+)\s*(?:\(\s*(?P<units>[^)]+)\s*\))?\s*=\s*(?P<val>{FLOAT_REGEX})\s*$')
# Flexible Bounds: name [opt_units] = val [opt_in] (bounds)
BOUNDS_PATTERN = re.compile(fr'^\s*(?P<name>\w+)\s*(?:\(\s*(?P<units>[^)]+)\s*\))?\s*=\s*(?P<val>{FLOAT_REGEX})\s+(?:in\s+)?\((?P<tup>.*?)\)\s*$')

def parse_model_file(file_name: str, nd: bool = False) -> tuple:
    # --- Data Structures ---
    init_conditions = {}
    parameters = {}  # Format: {name: (val, (min, max))}
    rescale_params = {}
    forcing_params = {}
    collected_units = set()

    # --- State/Section Management ---
    current_section = None

    # split string content into lines (simulating file read)
    try:
        with open(file_name, 'r', encoding='utf-8') as file:
            lines = file.read().strip().split('\n')
    except FileNotFoundError:
        raise FileNotFoundError("File not found")

    for line in lines:
        line = line.strip()
        if not line:
            continue  # skip empty lines

        # --- Section Detection ---
        if line.startswith("#"):
            if "Initial Conditions" in line:
                current_section = "INIT"
            elif "Parameters" in line and "Forcing" not in line:
                if "Dimensional" not in line:
                    current_section = "PARAM"
                else:
                    current_section = "RESCALE"
            elif "Forcing Parameters" in line:
                current_section = "FORCING"
            continue

        # --- Helper to process units ---
        def process_units(match_obj):
            if match_obj.group('units'):
                raw_units = match_obj.group('units').split()
                for u in raw_units:
                    # Clean exponent: ms^-2 -> ms
                    base_unit = u.split('^')[0]
                    collected_units.add(base_unit)

        # 1. Initial Conditions & Forcing (Using ASSIGNMENT_PATTERN)
        if current_section in ["INIT", "FORCING", "RESCALE"]:
            match = ASSIGNMENT_PATTERN.search(line)
            if match:
                if current_section == "INIT":
                    target_dict = init_conditions
                elif current_section == "FORCING":
                    target_dict = forcing_params
                else:
                    target_dict = rescale_params
                target_dict[match.group('name')] = float(match.group('val'))
                process_units(match)

        # 2. Parameters (Using BOUNDS_PATTERN)
        elif current_section == "PARAM":
            match = BOUNDS_PATTERN.search(line)
            if match:
                name = match.group('name')
                val = float(match.group('val'))
                # Extract bounds
                bounds = tuple(float(x) for x in re.findall(FLOAT_REGEX, match.group('tup')))
                parameters[name] = (val, bounds)
                process_units(match)

    if not nd:
        return init_conditions, parameters, forcing_params, tuple(collected_units)
    return init_conditions, parameters, rescale_params, forcing_params, collected_units

def save_mix_dist(dist: torch.distributions.MixtureSameFamily, filename: str):
    data_to_save = {'means': dist.component_distribution.loc, 'covariances': dist.component_distribution.covariance_matrix, 'weights': dist.mixture_distribution.probs}
    torch.save(data_to_save, filename)

def load_mix_dist(filename: str, device: torch.device = torch.device('cpu')) -> torch.distributions.MixtureSameFamily:
    data = torch.load(filename, map_location=device)
    means = data['means']
    covs = data['covariances']
    weights = data['weights']

    # reconstruct the distribution
    comp_dist = torch.distributions.MultivariateNormal(means, covariance_matrix=covs)
    mix_dist = torch.distributions.Categorical(probs=weights)
    prior = torch.distributions.MixtureSameFamily(mix_dist, comp_dist)

    return prior
