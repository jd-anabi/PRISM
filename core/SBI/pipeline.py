import torch
from torch.distributions import MixtureSameFamily

from .Priors import hopf_prior
from core.Simulator import dim_simulator, nd_simulator, nadrowski_simulator, hopf_simulator
from core.SBI import statistics
from core.SBI.Priors import dim_prior, nd_prior

def gen_obs(sim: str, params: torch.Tensor, t: torch.Tensor, inits: torch.Tensor, force: torch.Tensor, n_segs: int, steady_idx: int,
            batch_size: int = 1, dtype: torch.dtype = torch.float32, device: torch.device = torch.device("cpu")):
    """
    Function in the SBI pipeline that generates observational data to test the posterior extracted by the neural network.
    :param sim: name of the simulator; currently only valid ones are "Dimensional", "Non-dimensional", and "Nadrowski"
    :param params: the parameters to use for the simulation
    :param t: the time to run the simulation for
    :param inits: the initial values to use for the state variables of the simulation
    :param force: the force tensor
    :param n_segs: the number of segments to divide the time series into for memory efficient simulations
    :param steady_idx: the index that corresponds to the steady state
    :param batch_size: the batch size for the simulation
    :param dtype: the data type to use
    :param device: the device to run the simulator on
    :return: tensor of shape (number of state variables, batch size, t[steady_idx:].shape[0])
    :raise: ValueError if sim is not a valid one
    """
    if params.shape[0] != batch_size or inits.shape[0] != batch_size:
        raise ValueError(f"Batch size: {batch_size} cannot differ from dim 0 of parameters tensor or initial conditions tensor")

    valid_sims = ["Dimensional", "Non-dimensional", "Nadrowski", "Hopf"]
    if sim not in valid_sims:
        raise ValueError(f"Invalid simulator: {sim}")

    # move to device
    t = t.to(dtype=dtype, device=device)

    simulator = None
    match sim:
        case "Dimensional":
            simulator = dim_simulator.DimSimulator(params, force, inits, t, segs=n_segs, batch_size=batch_size, device=device)
        case "Non-dimensional":
            simulator = nd_simulator.NDSimulator(params, force, inits, t, segs=n_segs, batch_size=batch_size, device=device)
        case "Hopf":
            simulator = hopf_simulator.HopfSimulator(params, force, inits, t, segs=n_segs, batch_size=batch_size, device=device)
        case "Nadrowski":
            simulator = nadrowski_simulator.NadrowskiSimulator(params, force, inits, t, segs=n_segs, batch_size=batch_size, device=device)
        case _:
            raise ValueError(f"Invalid simulator: {sim}")

    obs = simulator.simulate()[:, 0, :, steady_idx:]
    return obs

def gen_stats(x: torch.Tensor, dt: float, n_bands: int = 20, n_lags: int = 20, pacf_lags: int = 20, downsamples: tuple = (2000, 2000, 2000, 2000)) -> torch.Tensor:
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
        case "Hopf":
            prior = hopf_prior.HopfPrior(dtype, device)
        case "Nadrowski":
            raise ValueError(f"Invalid model (at the moment): {model}")
        case _:
            raise ValueError(f"Invalid model: {model}")

    with torch.no_grad():
        prior = prior.construct_prior(t, n_params, global_batch_size, local_batch_size // (2 ** 6), segs, prior_bounds, t_global_scale=2, num_iterations=50, n_max=175000, steady=False)

    return prior