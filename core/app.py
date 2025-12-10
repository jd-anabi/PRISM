import os
import sys
import math
import time
from typing import Dict

os.environ['KMP_DUPLICATE_LIB_OK']='True'

import pint
import torch
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from sbi import utils
from sbi.inference import SNPE
from sbi.analysis import pairplot
from sbi.neural_nets import posterior_nn
from sbi.neural_nets.embedding_nets import CNNEmbedding

from .Helpers import helpers, model_helpers, visualizers, file_manager
from .SBI import prior, statistics, embedded_network
from .Simulator import simulator

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
    if (torch.cuda.get_device_properties(DEVICE).major, torch.cuda.get_device_properties(DEVICE).minor) < (8, 0):
        DEVICE = torch.device('cpu')
elif torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
else:
    DEVICE = torch.device('cpu')

DTYPE = torch.float64 if DEVICE.type == 'cuda' or DEVICE.type == 'cpu' else torch.float32
BATCH_SIZE = 2**10 if DEVICE.type == 'cuda' else 2**7

# ensemble variables needed
UNIQUE_FREQS = 2**6 # number of unique frequencies
ENSEMBLE_SIZE = 2**7 if DEVICE.type == 'cuda' else 2**5 # ensemble size for each frequency
FPB = BATCH_SIZE // ENSEMBLE_SIZE # number of frequencies per batch
ITERATIONS = int(UNIQUE_FREQS / FPB)

K_B = 1.380649e-23  # m^2 kg s^-2 K^-1

def run():
    # --- SETUP --- #
    # construct OS dependent directory for model parameters directory
    if sys.platform == 'win32':
        model_params_dir = '\\Cells\\'
    else:
        model_params_dir = '/Cells/'

    # list files in directory
    model_files = ['']
    file_num = 1
    direct = os.getcwd() + '\\Cells' if sys.platform == 'win32' else os.getcwd() + '/Cells'
    for root, dirs, files in os.walk(direct):
        level = root.replace(direct, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 2 * (level + 1)
        for file in files:
            model_files.append(file)
            print(f"{subindent}({file_num}) {file}")
            file_num += 1
    model_files.pop(0)

    # read in model parameters
    file_num = int(input('File number for model parameters: '))
    helpers.clear_screen()
    file = os.getcwd() + model_params_dir + model_files[file_num - 1]
    inits, params, rescaling_params, force_params, units = file_manager.parse_model_file(file)

    # need to construct dictionary now that constructs factors to convert current units to SI units
    ureg = pint.UnitRegistry()
    try:
        si_factors = [ureg(unit).to_base_units().magnitude for unit in units]
    except pint.UndefinedUnitError as e:
        print(f"Error: {e}. Unrecognized units.")
        exit()

    # --- TIME AND FREQUENCY ARRAY CALCULATIONS --- #
    # non-dimensional time
    dt = 1e-3
    t_max = int(input("Max time: "))
    helpers.clear_screen()
    ts = (0, t_max)
    n = int((ts[-1] - ts[0]) / dt)
    t_nd = torch.linspace(ts[0], ts[-1], n, dtype=DTYPE, device=DEVICE)
    steady_id = int(0.7 * len(t_nd))
    segs = math.ceil(ts[-1] / 100)

    # rescale time to dimensional
    t = model_helpers.rescale_t(t_nd, rescaling_params['t_offset'], rescaling_params['t_scale'])
    dt = float(t[1] - t[0])  # rescale dt

    # steady-state index for analysis later
    n = t.shape[0]
    n_steady = t[steady_id:].shape[0]

    # frequency arrays (only positive values)
    pos_freqs = torch.fft.rfftfreq(n_steady, dt)

    # --- PRIOR CONSTRUCTION --- #
    prior_path = os.getcwd() + '\\Priors\\mixed_prior_dist.pt' if sys.platform == 'win32' else os.getcwd() + '/Priors/mixed_prior_dist.pt'
    try:
        prior_dist = file_manager.load_mix_dist(prior_path)
    except FileNotFoundError as e:
        print(f"Error: {e}. Going to construct prior from scratch.")
        time.sleep(5)
        helpers.clear_screen()
        prior_bounds = []
        for bounds in params.values():
            prior_bounds.append(bounds[1])
        prior_dist = prior.Prior(DTYPE, DEVICE)
        with torch.no_grad():
            prior_dist = prior_dist.construct_prior(t_nd, 17, 2 * BATCH_SIZE, BATCH_SIZE // 2, segs, prior_bounds, t_global_scale=1, num_iterations=50)
        file_manager.save_mix_dist(prior_dist, "mixed_prior_dist.pt")
    corner_plot_path = os.getcwd() + '\\Priors\\mixed_prior_dist.png' if sys.platform == 'win32' else os.getcwd() + '/Priors/mixed_prior_dist.png'
    parameter_labels = [r'$\tau_{hb}$', r'$\tau_m$', r'$\tau_{gs}$', r'$\tau_t$',
                        r'$C_{min}$', r'$S_{min}$', r'$S_{max}$', r'$Ca^2_m$', r'$Ca^2_{gs}$',
                        r'$U_{gs,\ max}$', r'$\Delta E$', r'$k_{gs, \text{ ratio}}$',
                        r'$\chi_{hb}$', r'$\chi_a$', r'$x_c$', r'$\eta_{hb}$', r'$\eta_{a}$']
    visualizers.visualize_dist(prior_dist, labels=[label for label in parameter_labels if label != r'$\tau_t$'], save_path=corner_plot_path)

    # --- SUMMARY STATISTICS --- #
    init_pos = np.random.randint(0, 10, size=(BATCH_SIZE, 2))
    init_probs = np.random.randint(0, 1, size=(BATCH_SIZE, 3))
    inits = helpers.concat(init_pos, init_probs)  # size: (BATCH_SIZE, 5)
    inits = torch.tensor(inits, dtype=DTYPE, device=DEVICE)
    force = torch.zeros((BATCH_SIZE, t.shape[0]), dtype=DTYPE, device=DEVICE)

    num_runs = 1
    all_stats = []
    all_thetas = []
    with torch.no_grad():
        for _ in tqdm(range(num_runs), desc=f"Calculating summary statistics for {num_runs} runs", leave=False):
            curr_thetas = prior_dist.sample((BATCH_SIZE,)).to(device=DEVICE, dtype=DTYPE)
            sim = simulator.Simulator(curr_thetas, force, inits, t, segs=segs, batch_size=BATCH_SIZE, device=DEVICE)
            x_sims = sim.simulate()[0, 0, :, steady_id:]  # shape: (BATCH_SIZE, len(t))
            stats = statistics.SummaryStatistics(x_sims, dt)
            del x_sims
            all_stats.append(stats.compute_statistics(n_bands=20, n_lags=20, pacf_lags=10))
            all_thetas.append(curr_thetas)
    summary_stats = torch.cat(all_stats, dim=0)
    thetas = torch.cat(all_thetas, dim=0)

    # --- SNPE --- #
    # set up embedded network
    input_dim = summary_stats.shape[1]
    output_dim = input_dim // 4
    layer_dims = (3 * input_dim // 2, input_dim // 2)
    embedded_net = embedded_network.EmbeddedNet(input_dim, output_dim, layer_dims)

    # set up snpe with embedded network
    neural_posterior = posterior_nn(model='maf', embedding_net=embedded_net)
    inference = SNPE(prior=prior_dist, density_estimator=neural_posterior)

    # train the density estimator
    density_estimator = inference.append_simulations(thetas.to(dtype=torch.float32), summary_stats.to(dtype=torch.float32)).train()

    # build the posterior
    posterior = inference.build_posterior(density_estimator)

    # visualize the posterior
    cpu_device = torch.device('cpu')
    cpu_dtype = torch.float32
    obs_params = torch.tensor([value[0] for value in params.values()], dtype=cpu_dtype, device=cpu_device).unsqueeze(-1)
    sim_obs = simulator.Simulator(obs_params, force[0:1].to(dtype=cpu_dtype, device=cpu_device), inits[0:1].to(dtype=cpu_dtype, device=cpu_device),
                                t.to(dtype=cpu_dtype, device=cpu_device), segs=segs, batch_size=1, device=cpu_device)

    # visualize and validate posterior
    x_obs = sim_obs.simulate()[0, 0, :, n_steady:].to(dtype=DTYPE, device=DEVICE).unsqueeze(0)
    stats_obs = statistics.SummaryStatistics(x_obs, dt).compute_statistics(n_bands=20, n_lags=20, pacf_lags=10)
    samples = posterior.sample((1000,), x=stats_obs)
    fig, ax = pairplot(samples, points=obs_params.squeeze(-1), labels=parameter_labels)
    plt.show()