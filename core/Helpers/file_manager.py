import re

import torch

# --- Regex Definitions ---
FLOAT_REGEX = r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?' # pattern for: value
SIMPLE_PATTERN = re.compile(fr'^\s*(?P<name>\w+)\s*=\s*(?P<val>{FLOAT_REGEX})\s*$') # pattern for: key = value
BOUNDS_PATTERN = re.compile(fr'^\s*(?P<name>\w+)\s*=\s*(?P<val>{FLOAT_REGEX})\s+in\s+\((?P<tup>.*?)\)\s*$') # pattern for: key = value in (min, max)
UNIT_PATTERN = re.compile(fr'^\s*(?P<name>\w+)\s*\(\s*(?P<units>[^)]+)\s*\)\s*=\s*(?P<val>{FLOAT_REGEX})\s*$') # pattern for: key (units) = value

def parse_model_file(file_name: str, nd: bool = False) -> tuple:
    # --- Data Structures ---
    init_conditions = {}  # {str: float}
    non_dim_params = {}  # {str: (float, (float, float))}
    dim_params = {}  # {str: float}
    forcing_params = {}  # {str: float}
    collected_units = set() # we use a set initially to avoid duplicate units (e.g., 'nm' appearing twice)

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
        if line.startswith('#'):
            if nd:
                if 'Non-dimensional Initial Conditions' in line:
                    current_section = "INIT"
                elif 'Non-dimensional Parameters' in line:
                    current_section = "NON_DIM"
                elif 'Dimensional Parameters' in line:
                    current_section = "PARAMS"
                elif 'Forcing Parameters' in line:
                    current_section = "FORCING"
                continue
            else:
                if 'Initial Conditions' in line:
                    current_section = "INIT"
                elif 'Parameters' in line:
                    current_section = "PARAMS"
                elif 'Forcing Parameters' in line:
                    current_section = "FORCING"
                continue

        # --- Section Processing ---
        match current_section:
            # 1. Non-dimensional Initial Conditions (Simple floats)
            case 'INIT':
                match = SIMPLE_PATTERN.search(line)
                if match:
                    init_conditions[match.group('name')] = float(match.group('val'))

            # 2. Non-dimensional Parameters (Float + Tuple of Bounds)
            case 'NON_DIM':
                match = BOUNDS_PATTERN.search(line)
                if match:
                    name = match.group('name')
                    val = float(match.group('val'))
                    bounds_str = match.group('tup') # extract bounds from tuple string
                    bounds = tuple(float(x) for x in re.findall(FLOAT_REGEX, bounds_str))

                    # store as value + bounds pair
                    non_dim_params[name] = (val, bounds)

            # 3. Dimensional Parameters and Forcing Parameters. These sections are similar: they might have units (UNIT_PATTERN) or they might be simple assignments (SIMPLE_PATTERN)
            case 'DIM' | 'FORCING':
                target_dict = dim_params if current_section == 'DIM' else forcing_params

                # unit pattern
                match = UNIT_PATTERN.search(line)
                if match:
                    target_dict[match.group('name')] = float(match.group('val'))
                    raw_units = match.group('units').split()  # e.g., ['mg', 'nm', 'ms^-2']
                    for u in raw_units:
                        # split on '^' and take the first part to remove exponents (e.g., 'ms^-2' -> 'ms')
                        base_unit = u.split('^')[0]
                        collected_units.add(base_unit)
                    continue

                # fallback to simple pattern (e.g., 'phase = 0')
                match = SIMPLE_PATTERN.search(line)
                if match:
                    target_dict[match.group('name')] = float(match.group('val'))

    # Convert set to tuple
    final_units = tuple(collected_units)

    return init_conditions, non_dim_params, dim_params, forcing_params, final_units

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
