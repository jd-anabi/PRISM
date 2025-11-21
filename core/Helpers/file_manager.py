import re

import torch

SN_PATTERN = re.compile(r'[\s=]+([+-]?(?:0|[1-9]\d*)(?:\.\d*)?(?:[eE][+\-]?\d+)?)$')  # use pattern matching to extract values (scientific notation)
PAR_PATTERN = re.compile(r'\((.*?)\)') # use pattern matching to extract value within parentheses
UNIT_PATTERN = re.compile(r'[a-zA-Z]+') # use pattern matching to extract units

def read_model_file(file: str) -> tuple[list[float], list[float], list[float], list[float], list[str]]:
    x0: list[float] = []
    params: list[float] = []
    rescale_params: list[float] = []
    forcing_params: list[float] = []
    units: list[str] = []

    line = 0
    invalid_lines = (0, 23, 24, 33, 34)
    with open(file, mode='r') as txtfile:
        for row in txtfile:
            if line not in invalid_lines:
                # non-dimensional parameters
                if line <= 22:
                    val = float(re.findall(SN_PATTERN, row.strip())[0])
                    if line <= 5:
                        x0.append(val)
                    else:
                        params.append(val)
                # dimensional parameters
                elif line <= 32:
                    val = float(re.findall(SN_PATTERN, row.strip())[0])
                    curr_units = [unit for par in re.findall(PAR_PATTERN, row.strip()) for unit in
                                  re.findall(UNIT_PATTERN, par)]
                    rescale_params.append(val)
                    for unit in curr_units:
                        if unit not in units:
                            units.append(unit)
                # forcing parameters
                else:
                    val = float(re.findall(SN_PATTERN, row.strip())[0])
                    forcing_params.append(val)
            line = line + 1
    return x0, params, rescale_params, forcing_params, units

def save_mix_dist(dist: torch.distributions.MixtureSameFamily, filename: str):
    data_to_save = {"means": dist.component_distribution.loc, "covariances": dist.component_distribution.covariance_matrix, "weights": dist.mixture_distribution.probs}
    torch.save(data_to_save, filename)

def load_mix_dist(filename: str, device: torch.device = torch.device('cpu')) -> torch.distributions.MixtureSameFamily:
    data = torch.load(filename, map_location=device)
    means = data["means"]
    covs = data["covariances"]
    weights = data["weights"]

    # reconstruct the distribution
    comp_dist = torch.distributions.MultivariateNormal(means, covariance_matrix=covs)
    mix_dist = torch.distributions.Categorical(probs=weights)
    prior = torch.distributions.MixtureSameFamily(mix_dist, comp_dist)

    return prior
