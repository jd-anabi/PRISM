import warnings

import torch
import numpy as np
from tqdm import tqdm
from sbi.inference.posteriors.base_posterior import NeuralPosterior
from sbi.inference import NPE
from sbi.neural_nets import posterior_nn

from core.Helpers import helpers
from .Priors import dim_prior, nd_prior, hopf_prior
from core.Simulator import dim_simulator, nd_simulator, nadrowski_simulator, hopf_simulator
from core.SBI import statistics

def gen_obs(sim: str, params: torch.Tensor, t: torch.Tensor, inits: torch.Tensor, force: torch.Tensor, n_segs: int, steady_idx: int,
            batch_size: int = 1, dtype: torch.dtype = torch.float32, device: torch.device = torch.device("cpu")):
    """
    Generates observations based on specified simulation type, parameters, and other input data.

    This function initializes a simulator based on the chosen simulation type and configuration. It
    validates the batch size of input tensors and ensures that the simulation type is supported.
    The specified simulator is used to simulate observations, and the processed observation data
    is returned.

    :param sim: The type of simulator to use. Must be one of ["Dimensional", "Non-dimensional", "Nadrowski", "Hopf"].
    :param params: Tensor containing simulation parameters. The first dimension must match the given batch size.
    :param t: Tensor specifying the time points for the simulation. Its data type and device are set during processing.
    :param inits: Tensor containing initial conditions for the simulation. The first dimension must match the batch size.
    :param force: Tensor specifying the forces acting during the simulation.
    :param n_segs: The number of segments in the simulation. Used for configuration of the simulator.
    :param steady_idx: The index representing steady-state time points for slicing simulation results.
    :param batch_size: Number of simulation batches to process. Default is 1.
    :param dtype: Data type of tensors during processing. Default is `torch.float32`.
    :param device: The device on which simulations are run, such as "cpu" or "cuda". Default is "cpu".

    :return: Tensor containing simulated observations after processing using the selected simulator. Shape: (number of variables, batch size, steady state time points).
    """
    if params.shape[0] != batch_size or inits.shape[0] != batch_size:
        raise ValueError(f"Batch size: {batch_size} cannot differ from dim 0 of parameters tensor or initial conditions tensor")

    valid_sims = ["Dimensional", "Non-dimensional", "Nadrowski", "Hopf"]
    if sim not in valid_sims:
        raise ValueError(f"Invalid simulator: {sim}")

    # move to the specified device
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

def gen_stats(x: torch.Tensor, dt: float, n_bands: int = 20, n_lags: int = 20, pacf_lags: int = 20, device: torch.device = torch.device('cpu')) -> torch.Tensor:
    """
    Generate statistical features from input data using the given parameters.

    This function computes a set of statistical features by utilizing the
    `SummaryStatistics` facility. It supports options to partition data
    into frequency bands, calculate lags, and apply partial autocorrelation
    function (PACF). The final statistics are influenced by the specified
    downsampling rates for efficiency.

    :param x: The input data tensor from which features will be computed.
    :type x: torch.Tensor
    :param dt: The time step resolution for the input data.
    :type dt: float
    :param n_bands: Number of frequency bands for spectral features. Defaults to 20.
    :type n_bands: int, optional
    :param n_lags: Number of temporal lags for autocorrelation features. Defaults to 20.
    :type n_lags: int, optional
    :param pacf_lags: Number of lags for the partial autocorrelation function. Defaults to 20.
    :type pacf_lags: int, optional
    :param downsamples: Tuple specifying downsample rates for different statistical features.
     Defaults to (2000, 2000, 2000, 2000).
    :type downsamples: tuple
    :param comp_device: The device on which the computation should be performed for statistical features. Defaults to torch.device('cpu').
    :type comp_device: torch.device
    :param device: The device on which the statistics should be moved to. Defaults to torch.device('cpu').
    :type device: torch.device
    :return: A tensor containing the computed statistical features. Shape: (batch size, number of statistics).
    :rtype: torch.Tensor
    """
    stats = statistics.SummaryStatistics(x.to(device), dt)
    return stats.compute_statistics(n_bands, n_lags, pacf_lags)

def gen_prior(model: str, t: torch.Tensor, global_batch_size: int, local_batch_size: int, segs: int, prior_bounds: list,
              num_iterations: int = 25, dtype: torch.dtype = torch.float32,device: torch.device = torch.device('cpu')) -> torch.distributions.MixtureSameFamily:
    """
    Generates a prior distribution based on the given model and parameters.

    The function constructs a prior distribution using the specified model type
    and parameters. It supports different models, including "Dimensional",
    "Non-dimensional", and "Hopf". For the "Nadrowski" model or any invalid model
    input, it raises a ValueError. The prior generation process involves a series
    of calculations and iterations executed without gradient computation.

    :param model: Specifies the type of model to use for prior generation. Accepted
                  values include "Dimensional", "Non-dimensional", and "Hopf".
    :param t: A tensor representing the input time vector used in the prior
              construction process.
    :param global_batch_size: Global batch size to be considered during the prior
                              generation.
    :param local_batch_size: Local batch size to be used in the computation.
    :param segs: Number of segmentation points for prior construction.
    :param prior_bounds: A list of bounding values defining the range of the prior
                         parameters.
    :param num_iterations: Number of iterations to be performed in the process.
                           Defaults to 25.
    :param dtype: Data type to be used for tensor computations.
                  Defaults to torch.float32.
    :param device: Device on which the computation should run.
                   Defaults to torch.device('cpu').

    :return: A torch.distributions.MixtureSameFamily object representing the
             constructed prior distribution.
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
        prior = prior.construct_prior(t, n_params, global_batch_size, local_batch_size // (2 ** 6), segs, prior_bounds, t_global_scale=2, num_iterations=num_iterations, n_max=175000, steady=False)

    return prior

def gen_training_data(sim: str, prior: torch.distributions.Distribution, t: torch.Tensor,
                      run_size: int, n_runs: int, n_segs: int, steady_idx: int, dt: float,
                      dtype: torch.dtype = torch.float32, device: torch.device = torch.device('cpu')) -> tuple:
    """
    Generates synthetic training data for a dynamical system simulation using specified parameters
    and a probabilistic prior for generating initial conditions. This function supports different
    types of simulators, handles device configuration, and computes training statistics for the
    simulated observations.

    :param sim: The type of simulator to use. Supported options are "Dimensional",
                "Non-dimensional", "Nadrowski", and "Hopf".
    :param prior: A probabilistic distribution used for sampling initial parameters of
                  the system.
    :param t: A time tensor specifying the time points for the simulation.
    :param run_size: The size of the batch to simulate for each run.
    :param n_runs: The number of independent simulation runs to perform.
    :param n_segs: The number of segments in the generated data for certain simulators.
    :param steady_idx: The index within the time tensor from which the system is assumed
                       to reach a steady state.
    :param dt: The time discretization step used for generating statistics from simulated
               observations.
    :param dtype: (Optional) The data type for tensors used during simulation. Defaults
                  to torch.float32.
    :param device: (Optional) The device on which tensors will be allocated and computations
                   will run. Defaults to torch.device('cpu').
    :return: A tuple containing two elements:
             - A list of training data statistics computed for each simulation run.
             - A list of parameter tensors used for generating the simulated data.
    """
    valid_sims = ["Dimensional", "Non-dimensional", "Nadrowski", "Hopf"]
    if sim not in valid_sims:
        raise ValueError(f"Invalid simulator: {sim}")

    inits = None
    match sim:
        case "Dimensional":
            inits = helpers.concat(np.random.randint(0, 10, size=(run_size, 2)), np.random.randint(0, 1, size=(run_size, 3)))  # size: (run_size, 5)
            inits = torch.tensor(inits, dtype=dtype, device=device)
        case "Non-dimensional":
            inits = helpers.concat(np.random.randint(0, 10, size=(run_size, 2)), np.random.randint(0, 1, size=(run_size, 3)))  # size: (run_size, 5)
            inits = torch.tensor(inits, dtype=dtype, device=device)
        case "Hopf":
            inits = torch.tensor(np.random.randint(0, 10, size=(run_size, 2)), dtype=dtype, device=device)
        case "Nadrowski":
            raise ValueError(f"Invalid simulator (at the moment): {sim}")
        case _:
            raise ValueError(f"Invalid simulator: {sim}")

    # move to the specified device
    t = t.to(dtype=dtype, device=device)

    training_data = []
    thetas = []

    with torch.no_grad():
        for _ in tqdm(range(n_runs), desc=f"Generating training data", leave=False):
            curr_thetas = prior.sample((run_size,)).to(device=device, dtype=dtype)
            data = gen_obs(sim=sim, params=curr_thetas, t=t, inits=inits,
                           force=torch.zeros((1, t.shape[0]), dtype=dtype, device=device), n_segs=n_segs, steady_idx=steady_idx,
                           batch_size=run_size, dtype=dtype, device=device)[0, :, :]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                training_stats = gen_stats(data, dt, device=device)
                del data
                training_data.append(training_stats)
            thetas.append(curr_thetas)
            del training_stats

    training_data_tensor = torch.cat(training_data, dim=0)
    thetas_tensor = torch.cat(thetas, dim=0)

    return training_data_tensor, thetas_tensor

def train_nn(thetas: torch.Tensor, data: torch.Tensor, model: str,
             prior: torch.distributions.Distribution, embedding_net: torch.nn.Module,
             batch_size: int = 128, device: torch.device = torch.device('cpu')) -> NeuralPosterior:
    """
    Trains a neural posterior estimation (NPE) model using the given simulation data, prior distribution,
    and embedding network. This function constructs a neural posterior density estimator, trains it on
    the simulation data, and returns the resulting posterior distribution object.

    :param thetas: Simulated parameter values, represented as a PyTorch tensor of shape
        (num_simulations, parameter_dim).
    :param data: Simulated observation data, represented as a PyTorch tensor of shape
        (num_simulations, observation_dim).
    :param model: A string specifying the type of neural network model to use for density estimation.
        For example, 'maf', 'mdn', etc.
    :param prior: The prior distribution over parameters, represented as an instance of PyTorch's
        `torch.distributions.Distribution` class.
    :param embedding_net: A PyTorch module that specifies the embedding network to process the observation
        data before feeding it into the density estimator neural network.
    :param batch_size: The size of each training batch used during density estimator training. Default is 128.
    :param device: The computational device to be used (e.g., CPU or CUDA). Default is `torch.device('cpu')`.
    :return: A trained neural posterior distribution, represented as an instance of the `NeuralPosterior` class.
    """
    neural_posterior = posterior_nn(model=model, embedding_net=embedding_net)
    infer = NPE(prior=prior, density_estimator=neural_posterior, device=str(device))

    # train the density estimator
    density_estimator = infer.append_simulations(thetas, data).train(training_batch_size=batch_size)

    # build the posterior
    return infer.build_posterior(density_estimator)