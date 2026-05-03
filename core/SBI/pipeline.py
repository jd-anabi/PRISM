import math
import warnings
from collections import OrderedDict

import torch
import numpy as np
from tqdm import tqdm
from sbi.inference.posteriors import DirectPosterior
from sbi.inference.posteriors.base_posterior import NeuralPosterior
from sbi.inference import SNPE
from sbi.neural_nets import posterior_nn
from torch.distributions.transforms import Transform

from core.Helpers import helpers
from core.config import CHUNK_LEN, N_ND_MAX
from .Priors import bp_prior, hopf_prior, nadrowski_prior
from core.Simulator import bp_simulator, nadrowski_simulator, hopf_simulator
from core.SBI import statistics

VALID_SIMS: dict = {"bp":        bp_simulator.BPSimulator,
                    "nadrowski": nadrowski_simulator.NadrowskiSimulator,
                    "hopf":      hopf_simulator.HopfSimulator}

VALID_PRIORS: dict = {"bp":        bp_prior.BPPrior,
                      "nadrowski": nadrowski_prior.NadrowskiPrior,
                      "hopf":      hopf_prior.HopfPrior}

INIT_SHAPES: dict = {"bp":        (2, 3),
                     "nadrowski": (2, 1),
                     "hopf":      (2, 0)}

def build_nondim_sin_force_tensor(
    forcing_params: torch.Tensor,
    t_nd: torch.Tensor,
    rescale_params: torch.Tensor,
    forcing_idx: dict,
    rescale_idx: dict,
) -> torch.Tensor:
    """
    Build a batch of non-dimensional sinusoidal force tensors.

    Constructs F_dim(t_dim) = amp * sin(2pi * freq * t_dim + phase) + offset
    in dimensional space, then nondimensionalizes via
    F_nd = (F_dim - f_offset) / f_scale.

    :param forcing_params: Forcing parameter values, shape (batch, n_forcing).
    :param t_nd: Non-dimensional time vector, shape (T,).
    :param rescale_params: Rescaling parameter values, shape (batch, n_rescale).
    :param forcing_idx: Maps forcing param names to column indices in forcing_params,
                        e.g. {"amp": 0, "freq": 1, "phase": 2, "offset": 3}. If "amp_y"
                        is present, a second forcing channel is built sharing freq, phase,
                        and offset with the x-channel but using its own amplitude.
    :param rescale_idx: Maps rescale param names to column indices in rescale_params,
                        e.g. {"t_scale": 3, "t_offset": 2, "f_scale": 7, "f_offset": 6}.
                        If "f_scale" is absent (Hopf-style nondim), f_scale is derived
                        as x_scale / t_scale and f_offset is taken as 0 — both follow
                        algebraically from F_ND = F_dim / (l * omega_0) with l = x_scale
                        and 1/omega_0 = t_scale.
    :return: Non-dimensional force tensor, shape (batch, n_force_channels, T) where
             n_force_channels = 2 if "amp_y" in forcing_idx else 1.
    """
    # extract forcing params as (batch, 1) for broadcasting against (1, T)
    amp    = forcing_params[:, forcing_idx["amp"]].unsqueeze(1)
    freq   = forcing_params[:, forcing_idx["freq"]].unsqueeze(1)
    phase  = forcing_params[:, forcing_idx["phase"]].unsqueeze(1)
    offset = forcing_params[:, forcing_idx["offset"]].unsqueeze(1)

    # extract rescale params as (batch, 1)
    t_scale  = rescale_params[:, rescale_idx["t_scale"]].unsqueeze(1)
    t_offset = rescale_params[:, rescale_idx["t_offset"]].unsqueeze(1)
    if "f_scale" in rescale_idx:
        f_scale  = rescale_params[:, rescale_idx["f_scale"]].unsqueeze(1)
        f_offset = rescale_params[:, rescale_idx["f_offset"]].unsqueeze(1)
    else:
        # Hopf-style nondim: F_ND = F_dim / (l * omega_0) -> f_scale = x_scale / t_scale,
        # f_offset = 0. Cell file omits f_scale/f_offset from the rescale block since
        # they're algebraic combinations of the inferred length and time scales, not
        # independent inferred dimensions.
        x_scale  = rescale_params[:, rescale_idx["x_scale"]].unsqueeze(1)
        f_scale  = x_scale / t_scale
        f_offset = torch.zeros_like(f_scale)

    # t_nd is (T,) -> (1, T) for broadcasting
    t = t_nd.unsqueeze(0)

    # nd -> dim time, then evaluate the shared sinusoidal carrier
    t_dim = helpers.rescale(t, t_scale, t_offset)             # (batch, T)
    sin_term = torch.sin(2 * np.pi * freq * t_dim + phase)    # (batch, T)

    # x-channel: dim -> nd force
    f_x_nd = (amp * sin_term + offset - f_offset) / f_scale   # (batch, T)

    if "amp_y" in forcing_idx:
        # y-channel shares freq, phase, offset, f_scale, f_offset with x;
        # only amp differs. Used by ND Hopf where the latent y also gets forcing.
        amp_y = forcing_params[:, forcing_idx["amp_y"]].unsqueeze(1)
        f_y_nd = (amp_y * sin_term + offset - f_offset) / f_scale
        return torch.stack([f_x_nd, f_y_nd], dim=1)           # (batch, 2, T)

    return f_x_nd.unsqueeze(1)                                # (batch, 1, T)

def gen_obs(model: str, params: torch.Tensor, t: torch.Tensor, inits: torch.Tensor, force: torch.Tensor,
            n_segs: int, steady_idx: int, fixed_dict: dict = None, state_dep_drift: bool = False,
            batch_size: int = 1, dtype: torch.dtype = torch.float32, device: torch.device = torch.device("cpu")):
    """
    Generates observations based on specified simulation type, parameters, and other input data.

    This function initializes a simulator based on the chosen simulation type and configuration. It
    validates the batch size of input tensors and ensures that the simulation type is supported.
    The specified simulator is used to simulate observations, and the processed observation data
    is returned.

    :param model: The type of model to use. Must be one of ["bp", "nadrowski", "hopf"].
    :param params: Tensor containing simulation parameters. The first dimension must match the given batch size.
    :param t: Tensor specifying the time points for the simulation. Its data type and device are set during processing.
    :param inits: Tensor containing initial conditions for the simulation. The first dimension must match the batch size.
    :param force: Tensor specifying the forces acting during the simulation.
    :param n_segs: The number of segments in the simulation. Used for configuration of the simulator.
    :param steady_idx: The index representing steady-state time points for slicing simulation results.
    :param fixed_dict: Dictionary of fixed parameters for the model.
    :param state_dep_drift: Whether to use state-dependent drift for the simulator.
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

    if VALID_SIMS.get(model.lower()) is None:
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

    simulator_cls = VALID_SIMS[model.lower()]
    simulator = simulator_cls(full_params, force, inits, t, segs=n_segs, batch_size=batch_size, device=device)

    obs = simulator.simulate(state_dep_drift=state_dep_drift)[:, 0, :, steady_idx:].clone()
    return obs

def gen_stats(x: torch.Tensor, dt: float | torch.Tensor , n_bands: int = 20, n_lags: int = 20, pacf_lags: int = 20,
              device: torch.device = torch.device('cpu'), stats_batch_size: int = 256) -> torch.Tensor:
    """
    Generate statistical features from input data using the given parameters.

    Computes statistics in sub-batches on the target device to keep GPU FFT
    performance while avoiding OOM on large datasets. Each sub-batch result
    is moved to CPU immediately.

    :param x: The input data tensor from which features will be computed (on CPU).
    :type x: torch.Tensor
    :param dt: The time step resolution for the input data.
    :type dt: float
    :param n_bands: Number of frequency bands for spectral features. Defaults to 20.
    :type n_bands: int, optional
    :param n_lags: Number of temporal lags for autocorrelation features. Defaults to 20.
    :type n_lags: int, optional
    :param pacf_lags: Number of lags for the partial autocorrelation function. Defaults to 20.
    :type pacf_lags: int, optional
    :param device: The device on which to compute statistics. Defaults to torch.device('cpu').
    :type device: torch.device
    :param stats_batch_size: Number of samples to process per sub-batch on GPU. Defaults to 256.
    :type stats_batch_size: int

    :return: A tensor containing the computed statistical features. Shape: (batch size, number of statistics).
    :rtype: torch.Tensor
    """
    total = x.shape[0]
    results = []
    for start in range(0, total, stats_batch_size):
        end = min(start + stats_batch_size, total)
        x_sub = x[start:end].to(device)
        dt_sub = dt
        if isinstance(dt, torch.Tensor):
            dt_sub = dt[start:end].to(device)
        stats = statistics.SummaryStatistics(x_sub, dt_sub)
        result = stats.compute_statistics(n_bands, n_lags, pacf_lags)
        results.append(result.cpu())
        del stats, x_sub, result
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    return torch.cat(results, dim=0)

def gen_prior(model: str, t: torch.Tensor, global_batch_size: int, local_batch_size: int, segs: int, prior_bounds: list,
              state_dep_drift: bool = False, num_iterations: int = 25,
              dtype: torch.dtype = torch.float32, device: torch.device = torch.device('cpu')) -> torch.distributions.MixtureSameFamily:
    """
    Generates a prior distribution based on the given model and parameters.

    The function constructs a prior distribution using the specified model type
    and parameters. It supports different models, including "BP", "Nadrowski",
    and "Hopf". For any invalid model input, it raises a ValueError. The prior
    generation process involves a series of calculations and iterations executed
    without gradient computation.

    :param model: Specifies the type of model to use for prior generation. Accepted
                  values include "BP", "Nadrowski", and "Hopf".
    :param t: A tensor representing the input time vector used in the prior
              construction process.
    :param global_batch_size: Global batch size to be considered during the prior
                              generation.
    :param local_batch_size: Local batch size to be used in the computation.
    :param segs: Number of segmentation points for prior construction.
    :param prior_bounds: A list of bounding values defining the range of the prior
                         parameters.
    :param state_dep_drift: Boolean flag indicating whether to include state-dependent drift in the prior.
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
    if VALID_PRIORS.get(model.lower()) is None:
        raise ValueError(f"Invalid simulator: {model}")

    n_params = len(prior_bounds)

    prior_cls = VALID_PRIORS[model.lower()]
    prior = prior_cls(dtype, device)

    with torch.no_grad():
        prior = prior.construct_prior(t, n_params, global_batch_size, local_batch_size, segs, prior_bounds,
                                      t_global_scale=2, num_iterations=num_iterations, n_max=175000, steady=False, state_dep_drift=state_dep_drift)

    return prior

def gen_training_data(model: str, prior: torch.distributions.Distribution, forcing_prior: torch.distributions.Distribution,
                      t: torch.Tensor, run_size: int, n_runs: int, steady_idx: int, dt_nd_min: float,
                      nd_dim: int, forcing_idx: dict, rescale_idx: dict,
                      dt_exp: float = None, t_min_exp: float = None, t_max_exp: float = None,
                      t_scale_bounds: tuple[float, float] = None,
                      proposal: DirectPosterior = None, theta_transform: Transform | None = None,
                      fixed_dict: dict = None, state_dep_drift: bool = False, dtype: torch.dtype = torch.float32, device: torch.device = torch.device('cpu')) -> tuple:
    """
    Generate synthetic training data for the SBI posterior using batch-by-scale strategy.

    Each batch shares a single (t_scale_k, T_k) pair sampled via Sobol sequence over the
    2D space [t_scale_lo, t_scale_hi] x [t_min_exp, t_max_exp]. Within a batch, the 11 ND
    parameters and (D, K_gs*D) vary per-simulation, but t_scale is overridden to the
    batch-level value. The pre-simulated ND trajectory is subsampled to dt_nd_k = dt_exp / t_scale_k
    and truncated to T_nd_k = T_k / t_scale_k points, so that after rescaling every simulation
    has physical duration T_k at sampling rate 1/dt_exp. Summary statistics are computed with
    the fixed dt_exp, and log(T_k) is appended to the conditioning vector.

    If theta_transform is provided, `prior` is interpreted as a LATENT prior. Samples z
    from it, applies theta_transform(z) to get physical θ for the simulator, and stores
    the latent z as the training target. The override of t_scale to the batch-level value
    is performed in physical space, after which the latent is recomputed via
    theta_transform.inv so the stored z corresponds exactly to what the simulator saw.

    If theta_transform is None, `prior` is physical and the legacy path is taken.

    :param model: Name of the simulation model (e.g. "nadrowski", "hopf").
    :param prior: Prior distribution over inferred parameters (ND x rescale product prior).
    :param forcing_prior: Prior distribution over dimensional forcing parameters, sampled
                          independently every batch regardless of SNPE round.
    :param t: Pre-simulated ND time tensor at finest resolution (dt_nd_min), shape (T_full,).
    :param run_size: Number of simulations per batch.
    :param n_runs: Number of batches to generate.
    :param steady_idx: Index where transient ends and steady-state begins (at full resolution).
    :param dt_nd_min: Finest ND time step of the pre-simulated trajectory.
    :param nd_dim: Number of ND model parameters; used to split inferred params into
                   theta_nd [:nd_dim] and theta_rescale [nd_dim:].
    :param forcing_idx: Maps forcing param names to column indices,
                        e.g. {"amp": 0, "freq": 1, "phase": 2, "offset": 3}.
    :param rescale_idx: Maps rescale param names to column indices,
                        e.g. {"t_scale": 3, "t_offset": 2, "f_scale": 7, "f_offset": 6}.
    :param dt_exp: Fixed experimental sampling interval (seconds).
    :param t_min_exp: Shortest experimental recording duration (seconds).
    :param t_max_exp: Longest experimental recording duration (seconds).
    :param t_scale_bounds: (lo, hi) bounds on the t_scale rescaling parameter.
    :param proposal: Proposal distribution for SNPE rounds 2+. If None, samples from prior.
    :param theta_transform: Optional transformation function for physical parameters.
    :param fixed_dict: Optional dict mapping ND parameter indices to fixed values for
                       conditional posterior estimation.
    :param state_dep_drift: Whether the model uses state-dependent drift.
    :param dtype: Tensor data type. Defaults to torch.float32.
    :param device: Computation device. Defaults to CPU.
    :return: Tuple of (training_data, thetas) where training_data has shape
             (n_runs * run_size, n_stats + n_forcing + 1) and thetas has shape
             (n_runs * run_size, nd_dim + rescale_dim).
    """
    if model.lower() not in VALID_SIMS:
        raise ValueError(f"Invalid simulator: {model}")

    n_pos, n_prob = INIT_SHAPES[model.lower()]
    if n_prob > 0:
        inits = torch.tensor(
            helpers.concat(np.array(np.random.randint(0, 10, size=(run_size, n_pos))),
                           np.array(np.random.randint(0, 1, size=(run_size, n_prob)))),
            dtype=dtype, device=device)
    else:
        inits = torch.tensor(np.random.randint(0, 10, size=(run_size, n_pos)), dtype=dtype, device=device)

    # move to the specified device
    t = t.to(dtype=dtype, device=device)

    training_data = []
    thetas = []

    sampling_dist = prior if proposal is None else proposal

    # --- Stratified sampling of batch-level (t_scale, T) pairs with pre-filter ---
    t_scale_lo, t_scale_hi = t_scale_bounds
    log_t_scale_lo, log_t_scale_hi = math.log(t_scale_lo), math.log(t_scale_hi)
    log_T_lo, log_T_hi = math.log(t_min_exp), math.log(t_max_exp)

    def _draw_and_filter(n_candidates: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Draw Sobol candidates, filter by N_ND_MAX, return (t_scales, Ts) that fit."""
        pts = sobol.draw(n_candidates)
        cand_t_scales = torch.exp(log_t_scale_lo + pts[:, 0] * (log_t_scale_hi - log_t_scale_lo))
        cand_Ts = torch.exp(log_T_lo + pts[:, 1] * (log_T_hi - log_T_lo))
        dt_nd_cand = dt_exp / cand_t_scales
        subsample_cand = torch.clamp(torch.round(dt_nd_cand / dt_nd_min), min=1).long()
        N_points_cand = (cand_Ts / dt_exp).long()
        n_fine_cand = steady_idx + N_points_cand * subsample_cand
        valid = n_fine_cand <= N_ND_MAX
        return cand_t_scales[valid], cand_Ts[valid]

    sobol = torch.quasirandom.SobolEngine(dimension=2, scramble=True)
    oversample = 3
    valid_t_scales, valid_Ts = _draw_and_filter(n_runs * oversample)
    # Fallback: keep drawing more candidates until we have enough valid ones
    while valid_t_scales.shape[0] < n_runs:
        more_t_scales, more_Ts = _draw_and_filter(n_runs * oversample)
        valid_t_scales = torch.cat([valid_t_scales, more_t_scales])
        valid_Ts = torch.cat([valid_Ts, more_Ts])
    batch_t_scales = valid_t_scales[:n_runs]
    batch_Ts = valid_Ts[:n_runs]

    with torch.no_grad():
        for batch_k in tqdm(range(n_runs), desc="Generating training data", leave=False):
            # --- Batch-level scale and duration (unchanged) ---
            t_scale_k = batch_t_scales[batch_k].item()
            T_k = batch_Ts[batch_k].item()
            T_nd_k = T_k / t_scale_k
            dt_nd_k = dt_exp / t_scale_k
            subsample_factor = max(1, round(dt_nd_k / dt_nd_min))
            N_points_k = int(T_nd_k / dt_nd_k)
            n_fine_total = steady_idx + N_points_k * subsample_factor
            t_fine = t[:n_fine_total]
            n_segs_k = max(1, math.ceil(n_fine_total / CHUNK_LEN))

            # 1. Sample inferred params. If theta_transform given, sampling_dist is latent.
            curr_thetas_raw = sampling_dist.sample((run_size,)).to(device=device, dtype=dtype)
            if theta_transform is not None:
                # prior is latent; lift to physical for the simulator
                curr_thetas_phys = theta_transform(curr_thetas_raw)
            else:
                curr_thetas_phys = curr_thetas_raw

            curr_thetas_nd      = curr_thetas_phys[:, :nd_dim]
            curr_thetas_rescale = curr_thetas_phys[:, nd_dim:]
            curr_thetas_forcing = forcing_prior.sample((run_size,)).to(device=device, dtype=dtype)

            # Override t_scale to the batch-level value (in PHYSICAL space)
            curr_thetas_rescale[:, rescale_idx["t_scale"]] = t_scale_k

            # Recompute the latent to reflect the override; this is the training target.
            if theta_transform is not None:
                curr_thetas_latent = theta_transform.inv(curr_thetas_phys)
            else:
                curr_thetas_latent = curr_thetas_phys

            # 2. Build nondimensional force tensor at fine resolution (uses PHYSICAL rescale)
            force = build_nondim_sin_force_tensor(
                curr_thetas_forcing, t_fine, curr_thetas_rescale, forcing_idx, rescale_idx
            )

            # 3. Simulate with physical ND params
            x_nd_fine = gen_obs(
                model=model, params=curr_thetas_nd, t=t_fine, inits=inits,
                force=force, n_segs=n_segs_k, steady_idx=steady_idx,
                fixed_dict=fixed_dict, state_dep_drift=state_dep_drift,
                batch_size=run_size, dtype=dtype, device=device,
            )[0, :, :]
            x_nd = x_nd_fine[:, ::subsample_factor][:, :N_points_k]
            del x_nd_fine, force

            # 4. Redimensionalize (uses PHYSICAL rescale)
            x_scale  = curr_thetas_rescale[:, rescale_idx["x_scale"]].unsqueeze(1)
            x_offset = curr_thetas_rescale[:, rescale_idx["x_offset"]].unsqueeze(1)
            x_dim = helpers.rescale(x_nd, x_scale, x_offset)
            del x_nd

            # 5. Stats + conditioning (unchanged)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                training_stats = gen_stats(x_dim.cpu(), dt_exp, device=device)
                log_T_k_tensor = torch.full((run_size, 1), math.log(T_k), dtype=dtype)
                training_stats = torch.cat((training_stats, curr_thetas_forcing.cpu(), log_T_k_tensor), dim=-1)
                training_data.append(training_stats)

            # 6. Collect LATENT targets (not physical)
            thetas.append(curr_thetas_latent.cpu())
            if device.type == "cuda":
                torch.cuda.empty_cache()

    training_data_tensor = torch.cat(training_data, dim=0)
    thetas_tensor = torch.cat(thetas, dim=0)
    return training_data_tensor, thetas_tensor

def train_nn(training_params: dict, model: str, prior: torch.distributions.Distribution, embedding_net: torch.nn.Module,
             forcing_prior: torch.distributions.Distribution, nd_dim: int, forcing_idx: dict, rescale_idx: dict,
             x_obs: torch.Tensor = None, theta_obs: torch.Tensor = None, num_rounds: int = 1, return_diagnostics: bool = False, theta_transform: Transform | None = None,
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
        data, thetas = gen_training_data(
            training_params["model"], training_params["prior"], forcing_prior, training_params["t"],
            training_params["run_size"], training_params["num_runs"],
            training_params["steady_idx"], training_params["dt_nd_min"],
            nd_dim, forcing_idx, rescale_idx,
            dt_exp=training_params["dt_exp"], t_min_exp=training_params["t_min_exp"],
            t_max_exp=training_params["t_max_exp"], t_scale_bounds=training_params["t_scale_bounds"],
            proposal=proposal,
            theta_transform=theta_transform,
            fixed_dict=fixed_dict,
            state_dep_drift=training_params.get("state_dep_drift", False),
            dtype=training_params["dtype"], device=training_params["device"],
        )

        # filter data
        nan_mask = torch.isfinite(data).all(dim=1)
        safe_magnitude_mask = (torch.abs(data) < 1e15).all(dim=1)
        valid_idx = nan_mask & safe_magnitude_mask
        thetas = thetas[valid_idx]
        data = data[valid_idx]

        infer.append_simulations(thetas, data, proposal=proposal)
        density_estimator = infer.train(training_batch_size=batch_size)
        posterior = infer.build_posterior(density_estimator)
        assert isinstance(posterior, DirectPosterior), f"Expected DirectPosterior, got {type(posterior)}"

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
            assert x_obs is not None, "x_obs must be specified for SNPE algorithm"
            proposal = posterior.set_default_x(x_obs.to(device)) # if SNPE, the user has to specify x_obs

    assert isinstance(posterior, DirectPosterior), f"Expected DirectPosterior, got {type(posterior)}"
    if return_diagnostics:
        return posterior, diagnostics
    return posterior