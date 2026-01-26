import os
import sys
import math
import time
import warnings

from dipy.segment.clusteringspeed import DTYPE

os.environ["KMP_DUPLICATE_LIB_OK"]="True"

import pint
import torch
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from sbi import utils
from sbi.inference import SNPE
from sbi.analysis import pairplot
from sbi.neural_nets import posterior_nn

from .Helpers import helpers, visualizers, file_manager
from .SBI import statistics, embedded_network, pipeline
from .SBI.Priors import nd_prior, sbi_prior_wrapper
from .Simulator import nd_simulator, nadrowski_simulator, hopf_simulator

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
    BATCH_SIZE = 2**15
elif DEVICE.type == "cuda" and DTYPE == torch.float64:
    BATCH_SIZE = 2**10
else:
    BATCH_SIZE = 2**6

# ensemble variables needed
UNIQUE_FREQS = 2**6 # number of unique frequencies
ENSEMBLE_SIZE = 2**7 if DEVICE.type == "cuda" else 2**5 # ensemble size for each frequency
FPB = BATCH_SIZE // ENSEMBLE_SIZE # number of frequencies per batch
ITERATIONS = int(UNIQUE_FREQS / FPB)

K_B = 1.380649e-23  # m^2 kg s^-2 K^-1

def run():
    # --- SETUP --- #
    # construct OS dependent directory paths
    if sys.platform == "win32":
        cell_path = os.getcwd() + "\\Resources\\Cells\\"
        prior_path = os.getcwd() + "\\Resources\\Priors\\"
        posterior_path = os.getcwd() + "\\Resources\\Posteriors\\"
    else:
        cell_path = os.getcwd() + "/Resources/Cells/"
        prior_path = os.getcwd() + "/Resources/Priors/"
        posterior_path = os.getcwd() + "/Resources/Posteriors/"

    # list files in directory
    model_files = [""]
    file_num = 1
    for root, dirs, files in os.walk(cell_path):
        level = root.replace(cell_path, "").count(os.sep)
        indent = " " * 2 * level
        print(f"{indent}{os.path.basename(root)}")
        subindent = " " * 2 * (level + 1)
        for file in files:
            model_files.append(file)
            print(f"{subindent}({file_num}) {file}")
            file_num += 1
    model_files.pop(0)

    # read in model parameters
    file_num = int(input("\nFile number for model parameters: "))
    helpers.clear_screen()
    file = cell_path + model_files[file_num - 1]
    inits, params, force_params, units = file_manager.parse_model_file(file)

    # need to construct dictionary now that constructs factors to convert current units to SI units
    ureg = pint.UnitRegistry()
    try:
        si_factors = [ureg(unit).to_base_units().magnitude for unit in units]
    except pint.UndefinedUnitError as e:
        print(f"Error: {e}. Unrecognized units.")
        exit()

    # --- GENERATE "OBSERVABLE" DATA --- #
    t_max = int(input("Max time: "))
    dt = float(input("Time step: "))
    steady_percentage = float(input("Percentage of data that is transient (%): ").replace("%", "")) / 100.0
    segs = int(input("Number of segments to divide time series into: "))
    helpers.clear_screen()

    t = torch.linspace(0, t_max, int(t_max / dt), dtype=DTYPE, device=DEVICE)
    force = torch.zeros((BATCH_SIZE, t.shape[0]), dtype=DTYPE, device=DEVICE) # no forcing

    param_list = list(params.values())
    params_tensor = torch.tensor([row[0] for row in param_list], dtype=DTYPE).unsqueeze(0)
    inits_tensor = torch.tensor(list(inits.values()), dtype=DTYPE).unsqueeze(0)

    steady_idx = int(steady_percentage * len(t))
    obs_data = pipeline.gen_obs(sim="Hopf", params=params_tensor, t=t, inits=inits_tensor, force=force[0].unsqueeze(0), n_segs=segs, steady_idx=steady_idx)[0, :, :]
    obs_stats = pipeline.gen_stats(obs_data, dt)
    visualizers.plot(t[steady_idx:].cpu().detach().numpy(), obs_data[0, :].cpu().detach().numpy())

    # --- PRIOR CONSTRUCTION --- #
    print("Checking for prior...")
    time.sleep(1)
    mixed_prior_path = prior_path + "mixed_prior_dist.pt"
    try:
        helpers.clear_screen()
        mixed_prior = file_manager.load_mix_dist(mixed_prior_path, device=DEVICE)
        print("Prior found")
        time.sleep(1)
        helpers.clear_screen()
    except FileNotFoundError as e:
        print(f"Error: {e}. Going to construct prior from scratch.")
        time.sleep(5)
        helpers.clear_screen()
        prior_bounds = [row[1] for row in param_list]
        mixed_prior = pipeline.gen_prior(model="Hopf", t=t, global_batch_size=BATCH_SIZE, local_batch_size=(BATCH_SIZE // (2**6)),
                                         segs=math.ceil(segs / 2), prior_bounds=prior_bounds, dtype=DTYPE, device=DEVICE)
        file_manager.save_mix_dist(mixed_prior, mixed_prior_path)
    corner_plot_path = prior_path + "mixed_prior_dist.png"

    hopf_labels = [r"$\mu$", r"$\omega$", r"$\alpha$", r"$\beta$", r"$\epsilon_x$", r"$\epsilon_y$"]
    dim_labels = [r"$\lambda_x$", r"$\lambda_y$", r"$\lambda_{sf}$", r"$k_{sf}", r"k_{sp}",
                  r"$k_{gs, min}$", r"$k_{gs, max}$", r"$k_{es}", r"$x_{sf}$", r"$x_{es}$", r"$x_{sp}$", r"$x_c$",
                  r"$d$", r"$n$", r"$\gamma$", r"$c_{min}$", r"$s_{min}$", r"$c_{max}$", r"$s_{max}$",
                  r"$k_{m, +}$", r"$k_{r, +}", r"$k_{m, -}$", r"$k_{r, -}$", r"$Ca2_{x, in}$", r"$ca2_{x, ex}$",
                  r"$v_m$", r"$v_{ref}$", r"$z$", r"$r_m$", r"$r_r$", r"$\Delta_e$", r"$\tau_0$", r"$T$", r"$\epsilon$"]
    nd_labels = [r"$\tau_{hb}$", r"$\tau_m$", r"$\tau_{gs}$", r"$\tau_t$",
                 r"$C_{min}$", r"$S_{min}$", r"$S_{max}$", r"$Ca^2_m$", r"$Ca^2_{gs}$",
                 r"$U_{gs,\ max}$", r"$\Delta E$", r"$k_{gs, \text{ ratio}}$",
                 r"$\chi_{hb}$", r"$\chi_a$", r"$x_c$", r"$\eta_{hb}$", r"$\eta_{a}$"]
    visualizers.visualize_dist(mixed_prior, labels=hopf_labels, save_path=corner_plot_path)

    hopf_posterior_path = posterior_path + "hopf_posterior.pt"
    try:
        posterior = torch.load(hopf_posterior_path, map_location=torch.device("cpu"))
    except FileNotFoundError as e:
        # --- SUMMARY STATISTICS --- #
        num_runs = 10
        all_training_stats = []
        all_thetas = []
        init_pos = torch.tensor(np.random.randint(0, 10, size=(BATCH_SIZE, 2)), dtype=DTYPE, device=DEVICE)
        #batch_inits = helpers.concat(np.random.randint(0, 10, size=(BATCH_SIZE, 2)), np.random.randint(0, 1, size=(BATCH_SIZE, 3)))  # size: (BATCH_SIZE, 5)
        #batch_inits = torch.tensor(batch_inits, dtype=DTYPE, device=DEVICE)
        with torch.no_grad():
            for _ in tqdm(range(num_runs), desc=f"Calculating summary statistics for {num_runs} runs", leave=False):
                curr_thetas = mixed_prior.sample((BATCH_SIZE,)).to(device=DEVICE, dtype=DTYPE)
                training_data = pipeline.gen_obs(sim="Hopf", params=curr_thetas, t=t, inits=init_pos,
                                                 force=force[0].unsqueeze(0), n_segs=segs, steady_idx=steady_idx,
                                                 batch_size=BATCH_SIZE, dtype=DTYPE, device=DEVICE)[0, :, :]
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    training_stats = pipeline.gen_stats(training_data, dt)
                    all_training_stats.append(training_stats)
                all_thetas.append(curr_thetas)
                del training_stats
        summary_stats = torch.cat(all_training_stats, dim=0)
        thetas = torch.cat(all_thetas, dim=0)


        # --- SNPE --- #
        # filter data
        nan_mask = torch.isfinite(summary_stats).all(dim=1)
        safe_magnitude_mask = (torch.abs(summary_stats) < 1e15).all(dim=1)
        valid_idx = nan_mask & safe_magnitude_mask
        thetas = thetas[valid_idx]
        summary_stats = summary_stats[valid_idx]

        # set up embedded network
        input_dim = summary_stats.shape[1]
        output_dim = input_dim // 4
        layer_dims = (3 * input_dim // 2, input_dim // 2)
        embedded_net = embedded_network.EmbeddedNet(input_dim, output_dim, layer_dims)

        # set up snpe with embedded network
        priors = []
        prior_bounds = []
        for vals in params.values():
            prior_bounds.append(vals[1])
        #steady_idx = [i for i in range(len(prior_bounds)) if i != 3]
        #prior_bounds.pop(3)  # shape: (1, 16)
        for curr_bounds in prior_bounds:
            curr_prior = utils.BoxUniform(low=torch.ones(1) * curr_bounds[0], high=torch.ones(1) * curr_bounds[1])
            priors.append(curr_prior)
        sbi_prior = utils.MultipleIndependent(priors, device=str(DEVICE))

        #safe_prior = sbi_prior_wrapper.SBIPriorWrapper(mixed_prior)

        neural_posterior = posterior_nn(model="maf", embedding_net=embedded_net)
        inference = SNPE(prior=sbi_prior, density_estimator=neural_posterior, device=str(DEVICE))

        # train the density estimator
        density_estimator = inference.append_simulations(thetas, summary_stats).train(training_batch_size=int(2**7))

        # build the posterior
        posterior = inference.build_posterior(density_estimator)

        # save the posterior
        torch.save(posterior, hopf_posterior_path)

    # visualize and validate posterior
    posterior = torch.load(hopf_posterior_path, map_location=torch.device("cpu"))
    ground_truth = [row[0] for row in params]
    samples = posterior.sample((1000,), x=obs_stats)
    fig, ax = pairplot(samples.cpu().numpy(), points=np.array(ground_truth), labels=hopf_labels)
    plt.show()