import os
import sys
import math
import time
from typing import Dict

os.environ['KMP_DUPLICATE_LIB_OK']='True'

import pint
import torch
import numpy as np
from sbi import utils
from sbi.inference import NPE
from sbi.analysis import pairplot
from sbi.neural_nets import posterior_nn
from sbi.neural_nets.embedding_nets import CNNEmbedding

from .Helpers import helpers, model_helpers, visualizers, file_manager
from .SBI import prior, statistics
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
BATCH_SIZE = 2**10 if DEVICE.type == 'cuda' else 2**6

# ensemble variables needed
UNIQUE_FREQS = 2**6 # number of unique frequencies
ENSEMBLE_SIZE = 2**7 if DEVICE.type == 'cuda' else 2**5 # ensemble size for each frequency
FPB = BATCH_SIZE // ENSEMBLE_SIZE # number of frequencies per batch
ITERATIONS = int(UNIQUE_FREQS / FPB)

K_B = 1.380649e-23  # m^2 kg s^-2 K^-1

def run():
    # -------------------------- BEGIN SETUP -------------------------- #
    # construct OS dependent directory for model parameters directory
    if sys.platform == 'win32':
        model_params_dir = '\\Model Parameters\\'
    else:
        model_params_dir = '/Model Parameters/'

    # list files in directory
    model_files = ['']
    file_num = 1
    direct = os.getcwd() + '\\Model Parameters' if sys.platform == 'win32' else os.getcwd() + '/Model Parameters'
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
    x0, params, rescale_params, forcing_params, units = file_manager.read_model_file(file)
    amp, phase, offset = forcing_params

    # need to construct dictionary now that converts current units to SI units
    ureg = pint.UnitRegistry()
    try:
        factors = [ureg(unit).to_base_units().magnitude for unit in units]
        units_rescale: Dict[str, float] = {'distance': factors[0], 'mass': factors[1], 'time': factors[2]}
    except pint.UndefinedUnitError as e:
        print(f"Error: {e}. Unrecognized units.")
        exit()
    # -------------------------- END SETUP -------------------------- #

    # ------------- BEGIN RESCALING CALCULATIONS ------------- #
    # set up dictionary for model parameters and rescaling parameters
    parameters_with_bounds: Dict[str, tuple] = {'tau_hb': (params[0], (0.01, 100)), 'tau_m': (params[1], (0, 1000)),
                                                'tau_gs': (params[2], (0, 1000)), 'tau_t': (params[3], (0, 10)),
                                                'c_min': (params[4], (0, 1)), 's_min': (params[5], (0, 1)), 's_max': (params[6], (0, 1)),
                                                'ca2_m': (params[7], (0, 10)), 'ca2_gs': (params[8], (0, 1000)),
                                                'u_gs_max': (params[9], (0, 1000)), 'delta_e': (params[10], (0, 10)), 'k_gs_ratio': (params[11], (0, 1)),
                                                'chi_hb': (params[12], (0, 10)), 'chi_a': (params[13], (0, 10)), 'x_c': (params[14], (0, 100)),
                                                'eta_hb': (params[15], (0.001, 0.05)), 'eta_a': (params[16], (0.001, 0.05))}
    parameter_labels = [r'$\tau_{hb}$', r'$\tau_m$', r'$\tau_{gs}$', r'$\tau_t$',
                        r'$C_{min}$', r'$S_{min}$', r'$S_{max}$', r'$Ca2_m$', r'$Ca2_{gs}$',
                        r'$U_{gs,\ max}$', r'$\Delta E$', r'$k_{gs, \text{ ratio}}$',
                        r'$\chi_{hb}$', r'$\chi_a$', r'$x_c$', r'$\eta_{hb}$', r'$\eta_{a}$']
    hb_rescale_params: Dict[str, float] = {'gamma': rescale_params[0], 'd': rescale_params[1],
                                           'x_sp': rescale_params[2], 'k_sp': rescale_params[3],
                                           'k_gs_max': rescale_params[4], 's_max': rescale_params[5],
                                           't_0': rescale_params[6], 'alpha': rescale_params[7]}
    hb_rescale_params.update({'s_max_nd': params[6], 'chi_hb': params[12],
                              'chi_a': params[13]})  # need to add in non-dimensional parameters for rescaling too

    # rescaling parameters needed for time and data
    t_rescale_params = [hb_rescale_params['k_gs_max'], hb_rescale_params['s_max'], hb_rescale_params['t_0'],
                        hb_rescale_params['s_max_nd'], hb_rescale_params['chi_a']]
    x_rescale_params = [hb_rescale_params['gamma'], hb_rescale_params['d'], hb_rescale_params['x_sp'],
                        hb_rescale_params['k_sp'], hb_rescale_params['alpha'], hb_rescale_params['chi_hb']]
    # ------------- END RESCALING CALCULATIONS ------------- #

    # ------------- BEGIN TIME AND FREQUENCY ARRAY CALCULATIONS ------------- #
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
    t = model_helpers.rescale_t(t_nd, *t_rescale_params)
    dt = float(t[1] - t[0])  # rescale dt

    # steady-state index for analysis later
    n = t.shape[0]
    n_steady = t[steady_id:].shape[0]

    # frequency arrays (only positive values)
    pos_freqs = torch.fft.rfftfreq(n_steady, dt)
    # ------------- END TIME AND FREQUENCY ARRAY CALCULATIONS ------------- #

    # -------------------- BEGIN PRIOR CONSTRUCTION -------------------- #
    prior_path = os.getcwd() + '\\Priors\\mixed_prior_dist.pt' if sys.platform == 'win32' else os.getcwd() + '/Priors/mixed_prior_dist.pt'
    try:
        prior_dist = file_manager.load_mix_dist(prior_path)
    except FileNotFoundError as e:
        print(f"Error: {e}. Going to construct prior from scratch.")
        time.sleep(5)
        helpers.clear_screen()
        prior_bounds = []
        for bounds in parameters_with_bounds.values():
            prior_bounds.append(bounds[1])
        prior_dist = prior.Prior(DTYPE, DEVICE)
        prior_dist = prior_dist.construct_prior(t_nd, 17, 10 * BATCH_SIZE, 10 * BATCH_SIZE, segs, prior_bounds, t_global_scale=1, num_iterations=50)
        file_manager.save_mix_dist(prior_dist, "mixed_prior_dist.pt")
    corner_plot_path = os.getcwd() + '\\Priors\\mixed_prior_dist.png' if sys.platform == 'win32' else os.getcwd() + '/Priors/mixed_prior_dist.png'
    visualizers.visualize_dist(prior_dist, labels=[label for label in parameter_labels if label != r'$\tau_t$'], save_path=corner_plot_path)
    # -------------------- END PRIOR CONSTRUCTION -------------------- #

    # -------------------- BEGIN SUMMARY STATISTICS -------------------- #
    thetas = prior_dist.sample((BATCH_SIZE,)).to(device=DEVICE, dtype=DTYPE)
    init_pos = np.random.randint(0, 10, size=(BATCH_SIZE, 2))
    init_probs = np.random.randint(0, 1, size=(BATCH_SIZE, 3))
    inits = helpers.concat(init_pos, init_probs)  # size: (BATCH_SIZE, 5)
    inits = torch.tensor(inits, dtype=DTYPE, device=DEVICE)
    force = torch.zeros((BATCH_SIZE, t.shape[0]), dtype=DTYPE, device=DEVICE)

    sim = simulator.Simulator(thetas, force, inits, t, segs=segs, batch_size=BATCH_SIZE, device=DEVICE)
    x = sim.simulate()[0, 0, :, :]  # shape: (BATCH_SIZE, len(t))

    stats = statistics.SummaryStatistics(x, dt)
    print(stats.get_summary_statistics())
    # -------------------- END SUMMARY STATISTICS -------------------- #