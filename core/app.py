import os
import sys
import math
import time

from sbi.inference import DirectPosterior
from torch.distributions import MixtureSameFamily

os.environ["KMP_DUPLICATE_LIB_OK"]="True"

import pint
import torch
import numpy as np
from matplotlib import pyplot as plt
from sbi.analysis import pairplot

from .Helpers import helpers, visualizers, file_manager
from .SBI import embedded_network, pipeline, analysis
from .SBI.Priors import sbi_prior_wrapper

# === PYTORCH INITIALIZATION ===
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    if (torch.cuda.get_device_properties(DEVICE).major, torch.cuda.get_device_properties(DEVICE).minor) < (8, 0):
        DEVICE = torch.device("cpu")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

if DEVICE.type == "cuda":
    DTYPE = torch.float32
else:
    DTYPE = torch.float32

if DEVICE.type == "cuda" and DTYPE == torch.float32:
    BATCH_SIZE = 2**11
elif DEVICE.type == "cuda" and DTYPE == torch.float64:
    BATCH_SIZE = 2**10
else:
    BATCH_SIZE = 2**6

# === PATHS ===
if sys.platform == "win32":
    CELL_PATH = os.getcwd() + "\\Resources\\Cells\\"
    PRIOR_PATH = os.getcwd() + "\\Resources\\Priors\\"
    POSTERIOR_PATH = os.getcwd() + "\\Resources\\Posteriors\\"
    PLOT_PATH = os.getcwd() + "\\Resources\\Plots\\"
else:
    CELL_PATH = os.getcwd() + "/Resources/Cells/"
    PRIOR_PATH = os.getcwd() + "/Resources/Priors/"
    POSTERIOR_PATH = os.getcwd() + "/Resources/Posteriors/"
    PLOT_PATH = os.getcwd() + "/Resources/Plots/"

# === LABELS FOR PLOTTING ===
HOPF_LABELS = [r"$\mu$", r"$\omega$", r"$\alpha$", r"$\beta$", r"$\epsilon_x$", r"$\epsilon_y$"]
DIM_LABELS = [r"$\lambda_x$", r"$\lambda_y$", r"$\lambda_{sf}$", r"$k_{sf}", r"k_{sp}",
              r"$k_{gs, min}$", r"$k_{gs, max}$", r"$k_{es}", r"$x_{sf}$", r"$x_{es}$", r"$x_{sp}$", r"$x_c$",
              r"$d$", r"$n$", r"$\gamma$", r"$c_{min}$", r"$s_{min}$", r"$c_{max}$", r"$s_{max}$",
              r"$k_{m, +}$", r"$k_{r, +}", r"$k_{m, -}$", r"$k_{r, -}$", r"$Ca2_{x, in}$", r"$ca2_{x, ex}$",
              r"$v_m$", r"$v_{ref}$", r"$z$", r"$r_m$", r"$r_r$", r"$\Delta_e$", r"$\tau_0$", r"$T$", r"$\epsilon$"]
ND_LABELS = [r"$\tau_{hb}$", r"$\tau_m$", r"$\tau_{gs}$", r"$\tau_t$",
             r"$C_{min}$", r"$S_{min}$", r"$S_{max}$", r"$Ca^2_m$", r"$Ca^2_{gs}$",
             r"$U_{gs,\ max}$", r"$\Delta E$", r"$k_{gs, \text{ ratio}}$",
             r"$\chi_{hb}$", r"$\chi_a$", r"$x_c$", r"$\eta_{hb}$", r"$\eta_{a}$"]
NADROWSKI_LABELS = [r"$\lambda$", r"$\lambda_y$", r"$\tau$", r"$k_{gs}$", r"$k_{sp}$",
                    r"$d$", r"$f_{max}$", r"$c_0$", r"$c_m$", r"$S$",
                    r"$n$", r"$\Delta E$", r"$T$", r"$T_{eff}$", r"$\tau_c$"]
ND_NADROWSKI_LABELS = [r"$\kappa$", r"$\lambda$", r"$f_{\text{max}}$", r"$\tau$", r"$\tau_c$",
                       r"$c_0$", r"$S$", r"$\Delta E$", r"$\beta", r"$n$", r"$T$"]

# === VALID MODELS ===
VALID_MODELS = ["DIMENSIONAL", "NON-DIMENSIONAL", "NADROWSKI", "ND NADROWSKI", "HOPF"]
VALID_LABELS = [DIM_LABELS, ND_LABELS, NADROWSKI_LABELS, ND_NADROWSKI_LABELS, HOPF_LABELS]

# === ENSEMBLE VARIABLES ===
UNIQUE_FREQS = 2**6 # number of unique frequencies
ENSEMBLE_SIZE = 2**7 if DEVICE.type == "cuda:0" else 2**5 # ensemble size for each frequency
FPB = BATCH_SIZE // ENSEMBLE_SIZE # number of frequencies per batch
ITERATIONS = int(UNIQUE_FREQS / FPB)
K_B = 1.380649e-23  # m^2 kg s^-2 K^-1

def setup() -> tuple:
    """
    Load model parameters from a cell file and compute SI unit conversion factors.

    Prompts the user to select a cell file from Resources/Cells/, parses its initial conditions,
    model parameters, forcing parameters, and units, then converts units to SI using pint.

    :return: Tuple of (inits_dict, params_dict, force_params_dict, units_dict, si_factors).
    """
    # figure out which model to run
    helpers.clear_screen()
    print("Available models:")
    for model_idx, model in enumerate(VALID_MODELS):
        print(f"({model_idx + 1}) {model}")
    model_num = int(input("\nWhich model would you like to run? Select a number: "))
    model = VALID_MODELS[model_num - 1]
    labels = VALID_LABELS[model_num - 1]
    state_dep_drift = False
    if "nadrowski" in model.lower():
        state_dep_drift = True
    if model not in VALID_MODELS:
        raise ValueError(f"Invalid model selection. Please choose from {VALID_MODELS}.")
    helpers.clear_screen()

    # list files in the cell directory
    print("Available cell files:")
    cell_files = file_manager.list_dir(CELL_PATH)

    # read in model parameters
    file_num = int(input("\nFile number for model parameters: "))
    helpers.clear_screen()
    cell_file = CELL_PATH + cell_files[file_num - 1]
    inits_dict, params_dict, rescale_params, force_params_dict, units_dict = file_manager.parse_model_file(cell_file)

    # need to construct dictionary now that constructs factors to convert current units to SI units
    ureg = pint.UnitRegistry()
    try:
        si_factors = [ureg(unit).to_base_units().magnitude for unit in units_dict]
    except pint.UndefinedUnitError as e:
        print(f"Error: {e}. Unrecognized units.")
        exit()

    return inits_dict, params_dict, rescale_params, force_params_dict, units_dict, si_factors, model, labels, state_dep_drift

def run(inits_dict: dict, params_dict: dict, rescale_params: dict,
        force_params_dict: dict, units_dict: dict, si_factors: list,
        model: str, labels: list, state_dep_drift: bool):
    """
    Execute the full SBI pipeline: generate synthetic observations, construct or load a prior
    and posterior, then validate the posterior with PPC, SBC, expected coverage, and a
    posterior-vs-truth overlay plot.

    :param inits_dict: Initial conditions for the model state variables.
    :param params_dict: Model parameters; each value is [ground_truth, [lower_bound, upper_bound]].
    :param rescale_params: Rescaling parameters for non-dimensionalization.
    :param force_params_dict: Forcing/stimulus parameters (unused when force is zero).
    :param units_dict: Unit strings for each parameter (used for SI conversion).
    :param si_factors: Precomputed SI conversion factors from setup().
    :param model: Name of the model (e.g. "Hopf").
    :param labels: List of parameter labels for plotting.
    :param state_dep_drift: Whether the model has state-dependent drift.
    """
    # === GENERATE SYNTHETIC DATA ===
    t_max = int(input("Max time: "))
    dt = float(input("Time step: "))
    t = torch.linspace(0, t_max, int(t_max / dt), dtype=DTYPE, device=DEVICE)

    steady_percentage = float(input("Percentage of data that is transient (%): ").replace("%", "")) / 100.0
    steady_idx = int(steady_percentage * len(t))

    segs = int(input("Number of segments to divide time series into: "))
    helpers.clear_screen()

    force = torch.zeros((BATCH_SIZE, t.shape[0]), dtype=DTYPE, device=DEVICE) # no forcing

    param_vals = list(params_dict.values())
    params = torch.tensor([row[0] for row in param_vals], dtype=DTYPE).unsqueeze(0)
    inits = torch.tensor(list(inits_dict.values()), dtype=DTYPE).unsqueeze(0)

    obs_data = pipeline.gen_obs(model=model, params=params, t=t, inits=inits, force=force[0].unsqueeze(0),
                                n_segs=segs, steady_idx=steady_idx, state_dep_drift=state_dep_drift)[0, :, :]
    obs_stats = pipeline.gen_stats(obs_data, dt)
    visualizers.plot(t[steady_idx:].cpu().detach().numpy(), obs_data[0, :].cpu().detach().numpy())

    # === PRIOR CONSTRUCTION ===
    prior = _construct_prior(prior_bounds=[row[1] for row in param_vals], t=t, segs=math.ceil(segs / 2), model=model,
                             labels=labels, state_dep_drift=state_dep_drift)

    # === POSTERIOR CONSTRUCTION ===
    ground_truth = [row[0] for row in params_dict.values()]
    ground_truth_tensor = torch.tensor(ground_truth, dtype=DTYPE, device=DEVICE)

    """"# fix alpha and condition new prior
    alpha_idx = 2
    alpha_fixed = ground_truth[alpha_idx]
    ground_truth_alpha_fixed = ground_truth[:alpha_idx] + ground_truth[alpha_idx+1:]
    ground_truth_alpha_fixed_tensor = torch.tensor(ground_truth_alpha_fixed, dtype=DTYPE, device=DEVICE)

    comp_dist = cast(dist.MultivariateNormal, prior.component_distribution)
    mix_weights = prior.mixture_distribution.probs
    component_means, component_covs = comp_dist.loc, comp_dist.covariance_matrix
    weights, means, covs = helpers.condition_gmm_on_param(mix_weights, component_means, component_covs, alpha_idx, alpha_fixed)
    mixture = dist.Categorical(probs=weights.to(device=DEVICE))
    comp = dist.MultivariateNormal(means.to(device=DEVICE), covariance_matrix=covs.to(device=DEVICE))
    cond_prior = dist.MixtureSameFamily(mixture, comp)

    fixed_dict = {alpha_idx: alpha_fixed}"""
    posterior, pos_diagnostics = _construct_posterior(sim_model=model, prior=prior, t=t, obs_stats=obs_stats, ground_truth_tensor=ground_truth_tensor,
                                                      segs=segs, steady_idx=steady_idx, dt=dt, net_model="maf", training_runs=300,
                                                      num_rounds=1, state_dep_drift=state_dep_drift)
    helpers.clear_screen()

    # === VALIDATION ===
    # visualize posterior with corner plot
    samples = posterior.sample((1000,), x=obs_stats.to(DEVICE))
    fig, ax = pairplot(samples.cpu().numpy(), points=np.array([ground_truth]), labels=labels,)
    plt.show()

    # validate posterior with PPC, SBC, and expected coverage plots
    x_sims = pipeline.gen_obs(model=model, params=samples, t=t, inits=inits.expand(samples.shape[0], -1),
                              force=torch.zeros((samples.shape[0], t.shape[0]), dtype=DTYPE, device=DEVICE), n_segs=segs, steady_idx=steady_idx,
                              state_dep_drift=state_dep_drift, batch_size=samples.shape[0], dtype=DTYPE, device=DEVICE)[0, :, :]
    sim_stats = pipeline.gen_stats(x_sims, dt, device=DEVICE)
    results = analysis.posterior_predictive_check(obs_stats.squeeze(), sim_stats)

    x_cal, theta_star = analysis.gen_cal_data(model=model, prior=prior, t=t, n_segs=segs, steady_idx=steady_idx, dt=dt, n_cal=1000,
                                            state_dep_drift=state_dep_drift, dtype=DTYPE, device=DEVICE)
    ranks = analysis.compute_sbc_ranks(posterior, theta_star, x_cal, m=1000, device=DEVICE)
    alphas = analysis.compute_expected_coverage(posterior, theta_star, x_cal, m=1000, dtype=DTYPE, device=DEVICE)

    sbc_plot = visualizers.plot_sbc(ranks, param_names=labels, m=1000, fig_size=(7, 12))
    expected_cov_plot = visualizers.plot_expected_coverage(alphas, fig_size=(7, 20))
    plt.show()

    # "eye test"
    # get MAP parameters
    log_probs = posterior.log_prob(samples, x=obs_stats.to(DEVICE))
    map_params = samples[log_probs.argmax()].unsqueeze(0)

    # Simulate from MAP
    x_map = pipeline.gen_obs(model=model, params=map_params, t=t, inits=inits,
                             force=force[0].unsqueeze(0), n_segs=segs,
                             steady_idx=steady_idx, state_dep_drift=state_dep_drift)[0, 0, :]

    # simulate from several posterior samples (reuse x_sims from PPC)
    t_plot = t[steady_idx:].cpu().numpy()
    fig = visualizers.plot_posterior_vs_truth(
        t=t_plot,
        x_true=obs_data[0, :].cpu().numpy(),
        x_map=x_map.cpu().numpy(),
        x_samples=x_sims.cpu().numpy(),  # shape (N, T)
        n_show=10
    )
    plt.show()

# === PRIVATE FUNCTIONS ===
def _construct_prior(prior_bounds, t, segs, model, labels, state_dep_drift=False) -> MixtureSameFamily:
    """
    Load an existing prior from disk or construct one from scratch using a two-phase
    sweep (global + local) over the parameter space.

    :param prior_bounds: List of [lower, upper] bounds for each parameter.
    :param t: Time tensor for simulation during prior construction.
    :param segs: Number of time segments for stability filtering.
    :param model: Name of the model (e.g. "Hopf").
    :param labels: Labels for the parameter space
    :return: MixtureSameFamily prior distribution over model parameters.
    """
    print("Available priors: ")
    saved_priors = file_manager.list_dir(PRIOR_PATH)
    try:
        if len(saved_priors) > 0:
            prior_idx = int(input(f"\nWhich prior would you like to use? Select an file number ('0' if you want to make from scratch): ")) - 1
            if prior_idx == -1:
                raise ValueError
            prior_path = PRIOR_PATH + saved_priors[prior_idx]
            prior = file_manager.load_mix_dist(prior_path, device=DEVICE)
            helpers.clear_screen()
            visualizers.visualize_dist(prior, labels=labels)
        else:
            raise ValueError
    except ValueError:
        helpers.clear_screen()
        print("No prior found. Going to construct prior from scratch.")
        time.sleep(5)
        helpers.clear_screen()
        prior = pipeline.gen_prior(model=model, t=t, global_batch_size=BATCH_SIZE,
                                   local_batch_size=(BATCH_SIZE // 2),
                                   segs=math.ceil(segs / 2), prior_bounds=prior_bounds,
                                   state_dep_drift=state_dep_drift,
                                   num_iterations=50, dtype=DTYPE, device=DEVICE)
        prior_file_name = input("Enter a name for the prior file: ")
        file_manager.save_mix_dist(prior, PRIOR_PATH + prior_file_name + ".pt")
        corner_plot_path = PLOT_PATH + prior_file_name + ".png"
        visualizers.visualize_dist(prior, labels=labels, save_path=corner_plot_path)
    return prior

def _construct_posterior(sim_model: str, prior: MixtureSameFamily, t: torch.Tensor, obs_stats: torch.Tensor,
                         ground_truth_tensor: torch.Tensor, segs: int, steady_idx: int, dt: float, net_model: str = "maf",
                         training_runs: int = 15, run_size: int = BATCH_SIZE, num_rounds: int = 1, fixed_dict: dict = None,
                         state_dep_drift: bool = False, dtype: torch.dtype = DTYPE, device: torch.device = DEVICE) -> tuple[DirectPosterior, dict]:
    """
    Load an existing posterior from disk or train a new one using SNPE.

    Constructs an embedding network to compress summary statistics, wraps the prior for
    sbi compatibility, and trains a neural density estimator (MAF by default). The trained
    posterior is saved to Resources/Posteriors/.

    :param sim_model: Name of the simulation model (e.g. "Hopf").
    :param prior: Prior distribution over model parameters.
    :param t: Time tensor for generating training simulations.
    :param obs_stats: Observed summary statistics, shape (1, n_stats).
    :param ground_truth_tensor: Ground truth parameter values, shape (n_params,).
    :param segs: Number of time segments per simulation.
    :param steady_idx: Index where transient ends and steady state begins.
    :param dt: Simulation time step.
    :param net_model: Density estimator architecture ("maf", "nsf", etc.).
    :param training_runs: Number of simulation batches to generate for training.
    :param run_size: Batch size for each training run.
    :param num_rounds: Number of sequential SNPE rounds.
    :param fixed_dict: Dictionary of fixed parameters for the model. Defaults to None.
    :param dtype: Tensor data type.
    :param device: Computation device.
    :return: Tuple of (trained DirectPosterior, diagnostics dict or None if loaded from disk).
    """
    pos_diagnostics = None
    print("Available posteriors: ")
    saved_posteriors = file_manager.list_dir(POSTERIOR_PATH)
    try:
        if len(saved_posteriors) > 0:
            posterior_idx = int(input(f"\nWhich posterior would you like to use? Select an file number (or '0' if you would like to make it from scratch): ")) - 1
            if posterior_idx == -1:
                raise ValueError
            posterior_path = POSTERIOR_PATH + saved_posteriors[posterior_idx]
            posterior = torch.load(posterior_path, weights_only=False)
            helpers.clear_screen()
        else:
            raise ValueError
    except ValueError:
        helpers.clear_screen()
        training_params = {"model": sim_model, "prior": prior, "t": t, "run_size": run_size, "num_runs": training_runs,
                                "n_segs": segs, "steady_idx": steady_idx, "dt": dt, "state_dep_drift": state_dep_drift,
                                "dtype": dtype, "device": device}
        # === SNPE ===
        # set up an embedded network
        input_dim = obs_stats.shape[1]
        embedded_net = embedded_network.EmbeddedNet(input_dim, 3 * input_dim // 2, (5 * input_dim // 2, 2 * input_dim))

        # set up the SBI prior
        sbi_prior = sbi_prior_wrapper.SBIPriorWrapper(prior)

        # train the neural network
        posterior, pos_diagnostics = pipeline.train_nn(training_params, model=net_model, prior=sbi_prior,
                                                       embedding_net=embedded_net, x_obs=obs_stats,
                                                       theta_obs=ground_truth_tensor, num_rounds=num_rounds,
                                                       return_diagnostics=True, fixed_dict=fixed_dict, batch_size=int(2 ** 7),
                                                       device=device)
        # save the posterior
        posterior_file_name = input("Enter a name for the posterior file: ")
        torch.save(posterior, POSTERIOR_PATH + posterior_file_name + ".pt")
    assert isinstance(posterior, DirectPosterior)
    assert isinstance(pos_diagnostics, dict)
    return posterior, pos_diagnostics