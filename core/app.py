import os
import sys
import math
import time

from dipy.segment.clusteringspeed import DTYPE
from pyro.infer.mcmc.util import diagnostics
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
    DEVICE = torch.device("cuda:0")
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
    BATCH_SIZE = 2**12
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

# === ENSEMBLE VARIABLES ===
UNIQUE_FREQS = 2**6 # number of unique frequencies
ENSEMBLE_SIZE = 2**7 if DEVICE.type == "cuda" else 2**5 # ensemble size for each frequency
FPB = BATCH_SIZE // ENSEMBLE_SIZE # number of frequencies per batch
ITERATIONS = int(UNIQUE_FREQS / FPB)
K_B = 1.380649e-23  # m^2 kg s^-2 K^-1

def setup() -> tuple:
    # list files in the cell directory
    cell_files = file_manager.list_dir(CELL_PATH)

    # read in model parameters
    file_num = int(input("\nFile number for model parameters: "))
    helpers.clear_screen()
    cell_file = CELL_PATH + cell_files[file_num - 1]
    inits_dict, params_dict, force_params_dict, units_dict = file_manager.parse_model_file(cell_file)

    # need to construct dictionary now that constructs factors to convert current units to SI units
    ureg = pint.UnitRegistry()
    try:
        si_factors = [ureg(unit).to_base_units().magnitude for unit in units_dict]
    except pint.UndefinedUnitError as e:
        print(f"Error: {e}. Unrecognized units.")
        exit()

    return inits_dict, params_dict, force_params_dict, units_dict, si_factors

def construct_prior(prior_bounds, t, segs) -> MixtureSameFamily:
    print("Available priors: ")
    saved_priors = file_manager.list_dir(PRIOR_PATH)
    if len(saved_priors) > 0:
        prior_idx = int(input(f"\nWhich prior would you like to use? Select an file number: ")) - 1
        prior_path = PRIOR_PATH + saved_priors[prior_idx]
        prior = file_manager.load_mix_dist(prior_path, device=DEVICE)
        helpers.clear_screen()
    else:
        helpers.clear_screen()
        print("No prior found. Going to construct prior from scratch.")
        time.sleep(5)
        helpers.clear_screen()
        prior = pipeline.gen_prior(model="Hopf", t=t, global_batch_size=BATCH_SIZE,
                                   local_batch_size=(BATCH_SIZE // (2 ** 6)),
                                   segs=math.ceil(segs / 2), prior_bounds=prior_bounds, dtype=DTYPE, device=DEVICE)
        prior_file_name = input("Enter a name for the prior file: ")
        file_manager.save_mix_dist(prior, PRIOR_PATH + prior_file_name + ".pt")
        corner_plot_path = PLOT_PATH + prior_file_name + ".png"
        visualizers.visualize_dist(prior, labels=HOPF_LABELS, save_path=corner_plot_path)
    return prior

def run():
    # === SETUP ===
    inits_dict, params_dict, force_params_dict, units_dict, si_factors = setup()

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

    obs_data = pipeline.gen_obs(model="Hopf", params=params, t=t, inits=inits, force=force[0].unsqueeze(0), n_segs=segs, steady_idx=steady_idx)[0, :, :]
    obs_stats = pipeline.gen_stats(obs_data, dt)
    visualizers.plot(t[steady_idx:].cpu().detach().numpy(), obs_data[0, :].cpu().detach().numpy())

    # === PRIOR CONSTRUCTION ===
    prior = construct_prior(prior_bounds=[row[1] for row in param_vals], t=t, segs=math.ceil(segs / 2))
    print("Available priors: ")
    saved_priors = file_manager.list_dir(PRIOR_PATH)
    if len(saved_priors) > 0:
        prior_idx = int(input(f"\nWhich prior would you like to use? Select an file number: ")) - 1
        prior_path = PRIOR_PATH + saved_priors[prior_idx]
        prior = file_manager.load_mix_dist(prior_path, device=DEVICE)
        helpers.clear_screen()
    else:
        print("No prior found. Going to construct prior from scratch.")
        time.sleep(5)
        helpers.clear_screen()
        prior_bounds = [row[1] for row in param_vals]
        prior = pipeline.gen_prior(model="Hopf", t=t, global_batch_size=BATCH_SIZE, local_batch_size=(BATCH_SIZE // (2**6)),
                                         segs=math.ceil(segs / 2), prior_bounds=prior_bounds, dtype=DTYPE, device=DEVICE)
        prior_file_name = input("Enter a name for the prior file: ")
        file_manager.save_mix_dist(prior, PRIOR_PATH + prior_file_name + ".pt")
        corner_plot_path = PLOT_PATH + prior_file_name + ".png"
        visualizers.visualize_dist(prior, labels=HOPF_LABELS, save_path=corner_plot_path)

    # === GET TRAINING DATA ===
    ground_truth = [row[0] for row in params_dict.values()]
    ground_truth_tensor = torch.tensor(ground_truth, dtype=DTYPE, device=DEVICE)
    pos_diagnostics = None
    print("Available posteriors: ")
    saved_posteriors = file_manager.list_dir(POSTERIOR_PATH)
    if len(saved_posteriors) > 0:
        posterior_idx = int(input(f"\nWhich posterior would you like to use? Select an file number (or '0' if you would like to make it from scratch): ")) - 1
        if posterior_idx == -1:
            hopf_training_params = {"model": "Hopf", "prior": prior, "t": t, "run_size": BATCH_SIZE, "num_runs": 300,
                                    "n_segs": segs,
                                    "steady_idx": steady_idx, "dt": dt, "dtype": DTYPE, "device": DEVICE}
            # === SNPE ===
            # set up an embedded network
            input_dim = obs_stats.shape[1]
            embedded_net = embedded_network.EmbeddedNet(input_dim, 3 * input_dim // 2,
                                                        (5 * input_dim // 2, 2 * input_dim))

            # set up the SBI prior
            sbi_prior = sbi_prior_wrapper.SBIPriorWrapper(prior)

            # train the neural network
            posterior, pos_diagnostics = pipeline.train_nn(hopf_training_params, model="maf", prior=sbi_prior,
                                          embedding_net=embedded_net, x_obs=obs_stats, theta_obs=ground_truth_tensor,
                                          num_runs=1, return_diagnostics=True, batch_size=int(2 ** 7), device=DEVICE)

            # save the posterior
            posterior_file_name = input("Enter a name for the posterior file: ")
            torch.save(posterior, POSTERIOR_PATH + posterior_file_name + ".pt")
        else:
            posterior_path = POSTERIOR_PATH + saved_posteriors[posterior_idx]
            posterior = torch.load(posterior_path, weights_only=False)
    else:
        hopf_training_params = {"model": "Hopf", "prior": prior, "t": t, "run_size": BATCH_SIZE, "num_runs": 15, "n_segs": segs,
                                "steady_idx": steady_idx, "dt": dt, "dtype": DTYPE, "device": DEVICE}
        # === SNPE ===
        # set up an embedded network
        input_dim = obs_stats.shape[1]
        embedded_net = embedded_network.EmbeddedNet(input_dim, 3 * input_dim // 2, (5 * input_dim // 2, 2 * input_dim))

        # set up the SBI prior
        sbi_prior = sbi_prior_wrapper.SBIPriorWrapper(prior)

        # train the neural network
        posterior, pos_diagnostics = pipeline.train_nn(hopf_training_params, model="maf", prior=sbi_prior,
                                                   embedding_net=embedded_net, x_obs=obs_stats,
                                                   theta_obs=ground_truth_tensor,
                                                   num_runs=3, return_diagnostics=True, batch_size=int(2 ** 7),
                                                   device=DEVICE)

        # save the posterior
        posterior_file_name = input("Enter a name for the posterior file: ")
        torch.save(posterior, POSTERIOR_PATH + posterior_file_name + ".pt")

    # visualize and validate posterior
    samples = posterior.sample((1000,), x=obs_stats.to(DEVICE))
    fig, ax = pairplot(samples.cpu().numpy(), points=np.array([ground_truth]), labels=HOPF_LABELS)
    plt.show()

    x_sims = pipeline.gen_obs(model="hopf", params=samples, t=t, inits=inits.expand(samples.shape[0], -1),
                              force=torch.zeros((samples.shape[0], t.shape[0]), dtype=DTYPE, device=DEVICE), n_segs=segs, steady_idx=steady_idx,
                              batch_size=samples.shape[0], dtype=DTYPE, device=DEVICE)[0, :, :]
    sim_stats = pipeline.gen_stats(x_sims, dt, device=DEVICE)
    results = analysis.posterior_predictive_check(obs_stats.squeeze().to(DEVICE), sim_stats)
    print(f"Posterior predictive check: {results}")

    x_cal, theta_star = analysis.gen_cal_data(model="hopf", prior=prior, t=t, n_segs=segs, steady_idx=steady_idx, dt=dt, n_cal=1000, dtype=DTYPE, device=DEVICE)
    ranks = analysis.compute_sbc_ranks(posterior, theta_star, x_cal, m=1000, device=DEVICE)
    alphas = analysis.compute_expected_coverage(posterior, theta_star, x_cal, m=1000, dtype=DTYPE, device=DEVICE)

    param_names = [r'\mu', r'\omega', r'\alpha', r'\beta', r'\varepsilon_x', r'\varepsilon_y']
    sbc_plot = visualizers.plot_sbc(ranks, param_names=param_names, m=1000, fig_size=(7, 12))
    expected_cov_plot = visualizers.plot_expected_coverage(alphas, fig_size=(7, 20))
    plt.show()

    # "eye test"
    # get MAP parameters
    log_probs = posterior.log_prob(samples, x=obs_stats.to(DEVICE))
    map_params = samples[log_probs.argmax()].unsqueeze(0)

    # Simulate from MAP
    x_map = pipeline.gen_obs(model="hopf", params=map_params, t=t, inits=inits,
                             force=force[0].unsqueeze(0), n_segs=segs,
                             steady_idx=steady_idx)[0, 0, :]

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