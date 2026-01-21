import torch
from torch.distributions import MixtureSameFamily

from core.Simulator import dim_simulator, nd_simulator, nadrowski_simulator
from core.SBI import statistics
from core.SBI.Priors import dim_prior, nd_prior

def gen_obs(sim: str, params: list, t: torch.Tensor, inits: list, force: torch.Tensor, n_segs: int, steady_idx: int, dtype: torch.dtype = torch.float32, device: torch.device = torch.device("cpu")):
    """
    Function in the SBI pipeline that generates observational data to test the posterior extracted by the neural network.
    :param sim: name of the simulator; currently only valid ones are "Dimensional", "Non-dimensional", and "Nadrowski"
    :param params: the parameters to use for the simulation
    :param t: the time to run the simulation for
    :param inits: the initial values to use for the state variables of the simulation
    :param force: the force tensor
    :param n_segs: the number of segments to divide the time series into for memory efficient simulations
    :param steady_idx: the index that corresponds to the steady state
    :param dtype: the data type to use
    :param device: the device to run the simulator on
    :return: tensor of shape (number of state variables, t[steady_idx:].shape[0])
    :raise: ValueError if sim is not a valid one
    """
    valid_sims = ["Dimensional", "Non-dimensional", "Nadrowski"]
    if sim not in valid_sims:
        raise ValueError(f"Invalid simulator: {sim}")

    # move to device
    t = t.to(dtype=dtype, device=device)

    # construct parameters and initial conditions tensor
    params_tensor = torch.tensor(params, dtype=dtype, device=device).unsqueeze(0) # shape: (1, number of parameters)
    inits_tensor = torch.tensor(inits, dtype=dtype, device=device).unsqueeze(0) # shape: (1, number of state variables)

    simulator = None
    match sim:
        case "Dimensional":
            simulator = dim_simulator.DimSimulator(params_tensor, force, inits_tensor, t, segs=n_segs)
        case "Non-dimensional":
            simulator = nd_simulator.NDSimulator(params_tensor, force, inits_tensor, t, segs=n_segs)
        case "Nadrowski":
            simulator = nadrowski_simulator.NadrowskiSimulator(params_tensor, force, inits_tensor, t, segs=n_segs)
        case _:
            raise ValueError(f"Invalid simulator: {sim}")

    obs = simulator.simulate()[:, 0, 0, steady_idx:]
    return obs

def gen_stats(x: torch.Tensor, dt: float, n_bands: int = 10, n_lags: int = 10, pacf_lags: int = 10, downsamples: tuple = (1000, 1000, 1000, 1000)) -> torch.Tensor:
    """
    Function in the SBI pipeline that generates summary statistics
    :param x: the input data to compress to a set of summary statistics
    :param dt: the time step
    :param n_bands: the number of frequency bands
    :param n_lags: the number of lags
    :param pacf_lags: the number of lags for PACF calculation
    :param downsamples: the downsampling factor for computationally expensive statistics
    :return:
    """
    stats = statistics.SummaryStatistics(x, dt)
    return stats.compute_statistics(n_bands, n_lags, pacf_lags, downsamples=downsamples)

def gen_prior(model: str, t: torch.Tensor, global_batch_size: int, local_batch_size: int, segs: int, prior_bounds: list, dtype: torch.dtype = torch.float32,device: torch.device = torch.device('cpu')) -> MixtureSameFamily:
    """
    Function in the SBI pipeline that generates the prior
    :param model: which model to use
    :param t: the time series
    :param global_batch_size: the batch size for the global sweep
    :param local_batch_size: the batch size for the local sweep
    :param segs: the number of segments to divide the time series into for memory efficient simulations
    :param prior_bounds: the bounds for the parameters in the prior
    :param dtype: the data type to use
    :param device: the device to run the prior constructor on
    :return:
    """
    n_params = len(prior_bounds)
    prior = None

    match model:
        case "Dimensional":
            prior = dim_prior.DimPrior(dtype, device)
        case "Non-dimensional":
            prior = nd_prior.NDPrior(dtype, device)
        case "Nadrowski":
            raise ValueError(f"Invalid model (at the moment): {model}")
        case _:
            raise ValueError(f"Invalid model: {model}")

    with torch.no_grad():
        prior = prior.construct_prior(t, n_params, global_batch_size, local_batch_size // (2 ** 6), segs, prior_bounds, t_global_scale=2, num_iterations=300, n_max=175000)

    return prior