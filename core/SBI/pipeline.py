import warnings

import torch
import numpy as np
from tqdm import tqdm
from sbi.inference.posteriors import DirectPosterior
from sbi.inference import SNPE
from sbi.neural_nets import posterior_nn

from core.Helpers import helpers
from .Priors import dim_prior, nd_prior, hopf_prior, nadrowski_prior
from core.Simulator import dim_simulator, nd_simulator, nadrowski_simulator, hopf_simulator
from core.SBI import statistics

def gen_obs(model: str, params: torch.Tensor, t: torch.Tensor, inits: torch.Tensor, force: torch.Tensor, n_segs: int, steady_idx: int,
            fixed_dict: dict = None, batch_size: int = 1, dtype: torch.dtype = torch.float32, device: torch.device = torch.device("cpu")):
    """
    Generates observations based on specified simulation type, parameters, and other input data.

    This function initializes a simulator based on the chosen simulation type and configuration. It
    validates the batch size of input tensors and ensures that the simulation type is supported.
    The specified simulator is used to simulate observations, and the processed observation data
    is returned.

    :param model: The type of model to use. Must be one of ["dimensional", "non-dimensional", "nadrowski", "hopf"].
    :param params: Tensor containing simulation parameters. The first dimension must match the given batch size.
    :param t: Tensor specifying the time points for the simulation. Its data type and device are set during processing.
    :param inits: Tensor containing initial conditions for the simulation. The first dimension must match the batch size.
    :param force: Tensor specifying the forces acting during the simulation.
    :param n_segs: The number of segments in the simulation. Used for configuration of the simulator.
    :param steady_idx: The index representing steady-state time points for slicing simulation results.
    :param fixed_dict: Dictionary of fixed parameters for the model.
    :param batch_size: Number of simulation batches to process. Default is 1.
    :param dtype: Data type of tensors during processing. Default is `torch.float32`.
    :param device: The device on which simulations are run, such as "cpu" or "cuda". Default is "cpu".

    :return: Tensor containing simulated observations after processing using the selected simulator. Shape: (number of variables, batch size, steady state time points).
    :rtype: torch.Tensor

    :raises ValueError: If the batch size of input tensors does not match the first dimension of the parameters tensor or initial conditions tensor.
    :raises ValueError: If the specified model is not supported.
    """
    if params.shape[0] != batch_size or inits.shape[0] != batch_size:
        raise ValueError(f"Batch size: {batch_size} cannot differ from dim 0 of parameters tensor or initial conditions tensor")

    valid_models = ["dimensional", "non-dimensional", "nadrowski", "hopf"]
    if model.lower() not in valid_models:
        raise ValueError(f"Invalid simulator: {model}")

    full_params = params
    if fixed_dict is not None:
        n_full = params.shape[1] + len(fixed_dict)
        full_params = torch.empty((params.shape[0], n_full), dtype=params.dtype, device=params.device)
        free_idx = 0
        for i in range(n_full):
            if i in fixed_dict:
                full_params[:, i] = fixed_dict[i]
            else:
                full_params[:, i] = params[:, free_idx]
                free_idx += 1
        del params

    # move to the specified device
    t = t.to(dtype=dtype, device=device)

    simulator = None
    match model.lower():
        case "dimensional":
            simulator = dim_simulator.DimSimulator(full_params, force, inits, t, segs=n_segs, batch_size=batch_size, device=device)
        case "non-dimensional":
            simulator = nd_simulator.NDSimulator(full_params, force, inits, t, segs=n_segs, batch_size=batch_size, device=device)
        case "hopf":
            simulator = hopf_simulator.HopfSimulator(full_params, force, inits, t, segs=n_segs, batch_size=batch_size, device=device)
        case "nadrowski":
            simulator = nadrowski_simulator.NadrowskiSimulator(full_params, force, inits, t, segs=n_segs, batch_size=batch_size, device=device)
        case _:
            raise ValueError(f"Invalid simulator: {model}")

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
    :rtype: torch.distributions.MixtureSameFamily

    :raises ValueError: If the specified model is not supported.
    """
    valid_models = ["dimensional", "non-dimensional", "nadrowski", "hopf"]
    if model.lower() not in valid_models:
        raise ValueError(f"Invalid simulator: {model}")

    n_params = len(prior_bounds)
    prior = None

    match model.lower():
        case "dimensional":
            prior = dim_prior.DimPrior(dtype, device)
        case "non-dimensional":
            prior = nd_prior.NDPrior(dtype, device)
        case "hopf":
            prior = hopf_prior.HopfPrior(dtype, device)
        case "nadrowski":
            prior = nadrowski_prior.NadrowskiPrior(dtype, device)
        case _:
            raise ValueError(f"Invalid model: {model}")

    with torch.no_grad():
        prior = prior.construct_prior(t, n_params, global_batch_size, local_batch_size, segs, prior_bounds, t_global_scale=2, num_iterations=num_iterations, n_max=175000, steady=False)

    return prior

def gen_training_data(model: str, prior: torch.distributions.Distribution, t: torch.Tensor,
                      run_size: int, n_runs: int, n_segs: int, steady_idx: int, dt: float, proposal: torch.distributions.Distribution = None,
                      fixed_dict: dict = None, dtype: torch.dtype = torch.float32, device: torch.device = torch.device('cpu')) -> tuple:
    """
    Generates synthetic training data for a dynamical system simulation using specified parameters
    and a probabilistic prior for generating initial conditions. This function supports different
    types of simulators, handles device configuration, and computes training statistics for the
    simulated observations.

    :param model: The type of model to use. Supported options are "dimensional",
                "non-dimensional", "nadrowski", and "hopf".
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
    :param proposal: (Optional) A proposal distribution for sampling. Defaults to None.
    :param fixed_dict: (Optional) Dictionary of fixed parameters for the model. Defaults to None.
    :param dtype: (Optional) The data type for tensors used during simulation. Defaults
                  to torch.float32.
    :param device: (Optional) The device on which tensors will be allocated and computations
                   will run. Defaults to torch.device('cpu').
    :return: A tuple containing two elements:
             - A list of training data statistics computed for each simulation run.
             - A list of parameter tensors used for generating the simulated data.
    :rtype: tuple

    :raises ValueError: If the specified model is not supported.
    """
    valid_models = ["dimensional", "non-dimensional", "nadrowski", "hopf"]
    if model.lower() not in valid_models:
        raise ValueError(f"Invalid simulator: {model}")

    inits = None
    match model.lower():
        case "dimensional":
            inits = helpers.concat(np.random.randint(0, 10, size=(run_size, 2)), np.random.randint(0, 1, size=(run_size, 3)))  # size: (run_size, 5)
            inits = torch.tensor(inits, dtype=dtype, device=device)
        case "non-dimensional":
            inits = helpers.concat(np.random.randint(0, 10, size=(run_size, 2)), np.random.randint(0, 1, size=(run_size, 3)))  # size: (run_size, 5)
            inits = torch.tensor(inits, dtype=dtype, device=device)
        case "hopf":
            inits = torch.tensor(np.random.randint(0, 10, size=(run_size, 2)), dtype=dtype, device=device)
        case "nadrowski":
            inits = helpers.concat(np.random.randint(0, 10, size=(run_size, 2)), np.random.randint(0, 1, size=(run_size, 1)))  # size: (run_size, 3)
            inits = torch.tensor(inits, dtype=dtype, device=device)
        case _:
            raise ValueError(f"Invalid simulator: {model}")

    # move to the specified device
    t = t.to(dtype=dtype, device=device)

    training_data = []
    thetas = []

    sampling_dist = prior if proposal is None else proposal

    with torch.no_grad():
        for _ in tqdm(range(n_runs), desc=f"Generating training data", leave=False):
            curr_thetas = sampling_dist.sample((run_size,)).to(device=device, dtype=dtype)
            data = gen_obs(model=model, params=curr_thetas, t=t, inits=inits,
                           force=torch.zeros((run_size, t.shape[0]), dtype=dtype, device=device), n_segs=n_segs, steady_idx=steady_idx,
                           fixed_dict=fixed_dict, batch_size=run_size, dtype=dtype, device=device)[0, :, :]
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

def train_nn(training_params: dict, model: str, prior: torch.distributions.Distribution, embedding_net: torch.nn.Module,
             x_obs: torch.Tensor = None, theta_obs: torch.Tensor = None, num_rounds: int = 1, return_diagnostics: bool = False,
             fixed_dict: dict = None, batch_size: int = 128, device: torch.device = torch.device('cpu')) -> DirectPosterior | tuple[DirectPosterior, dict]:
    """
    Trains a neural posterior distribution using either Neural Posterior Estimation (NPE) or Sequential Neural Posterior
    Estimation (SNPE), depending on the number of training runs specified. The method automates simulation-based
    learning by generating synthetic data, training a density estimator, and refining a posterior iteratively if multiple
    training runs are performed.

    :param training_params: A dictionary of parameters required to generate training data. These parameters are used as input
        for the data generation function. Check @gen_training_data for details of the order of the parameters.
    :param model: The type of neural density estimator to use, specified as a string. It determines the architecture of the
        neural network approximating the posterior distribution.
    :param prior: The prior distribution over parameters, given as a `torch.distributions.Distribution` object.
    :param embedding_net: A neural network module that is used to compute embeddings of the data.
    :param x_obs: Observed data given as a `torch.Tensor`. Required when performing SNPE (i.e., `num_runs > 1`). Defaults
        to None.
    :param theta_obs: Observed parameters given as a `torch.Tensor`. Required when returning diagnostics. Defaults to None.
    :param num_rounds: The number of sequential training runs. If greater than 1, Sequential Neural Posterior Estimation (SNPE)
        is performed. Defaults to 1.
    :param return_diagnostics: Whether to return additional diagnostics such as loss values during training. Defaults to False.
    :param fixed_dict: Dictionary of fixed parameters for the model. Defaults to None.
    :param batch_size: Batch size for training the density estimator during each run. Defaults to 128.
    :param device: Device on which the computations should be performed (e.g., 'cpu' or 'cuda'). Defaults to 'cpu'.
    :return: A `NeuralPosterior` object representing the trained posterior distribution. If 'return_diagnostics = True', return a tuple containing
        the posterior and diagnostics.
    """
    if num_rounds > 1 and x_obs is None:
        raise ValueError("x_obs must be specified for SNPE algorithm")

    neural_posterior = posterior_nn(model=model, embedding_net=embedding_net)
    infer = SNPE(prior=prior, density_estimator=neural_posterior, device=str(device))

    proposal = None # set up initial proposal distribution
    posterior = None

    # diagnostics storage
    diagnostics = {
        "log_prob_true": [],
        "posterior_means": [],
        "posterior_stds": [],
    }

    for _ in tqdm(range(num_rounds), desc=f"Training neural posterior", leave=False):
        # train the density estimator
        data, thetas = gen_training_data(training_params["model"], training_params["prior"], training_params["t"],
                                         training_params["run_size"], training_params["num_runs"], training_params["n_segs"],
                                         training_params["steady_idx"], training_params["dt"], proposal=proposal,
                                         fixed_dict=fixed_dict, dtype=training_params["dtype"], device=training_params["device"])  # initial (data, thetas) training pairs

        # filter data
        nan_mask = torch.isfinite(data).all(dim=1)
        safe_magnitude_mask = (torch.abs(data) < 1e15).all(dim=1)
        valid_idx = nan_mask & safe_magnitude_mask
        thetas = thetas[valid_idx]
        data = data[valid_idx]

        infer.append_simulations(thetas, data, proposal=proposal)
        density_estimator = infer.train(training_batch_size=batch_size)
        posterior = infer.build_posterior(density_estimator)

        # compute diagnostics after each round
        if return_diagnostics and x_obs is not None:
            x_obs_device = x_obs.to(device)

            # log probability of ground truth
            if theta_obs is not None:
                theta_true_device = theta_obs.to(device)
                if theta_true_device.dim() == 1:
                    theta_true_device = theta_true_device.unsqueeze(0)
                log_prob = posterior.log_prob(theta_true_device, x=x_obs_device).item()
                diagnostics["log_prob_true"].append(log_prob)

            # posterior mean and std from samples
            samples = posterior.sample((10000,), x=x_obs_device)
            diagnostics["posterior_means"].append(samples.mean(dim=0).cpu())
            diagnostics["posterior_stds"].append(samples.std(dim=0).cpu())

        # need to now check if num_runs > 1: if so, then that is equivalent to SNPE, and if not, that is equivalent to NPE
        if num_rounds > 1:
            proposal = posterior.set_default_x(x_obs.to(device)) # if SNPE, the user has to specify x_obs

    assert isinstance(posterior, DirectPosterior), f"Expected DirectPosterior, got {type(posterior)}"

    if return_diagnostics:
        return posterior, diagnostics
    return posterior