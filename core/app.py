import os
import sys
import math
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

from .Helpers import fdt, helpers, model_helpers, stats
from .Simulator import simulator
from .SBI import prior

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
    file = os.getcwd() + model_params_dir + model_files[file_num - 1]
    x0, params, rescale_params, forcing_params, units = helpers.read_model_file(file)
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
    prior_bounds = []
    for bounds in parameters_with_bounds.values():
        prior_bounds.append(bounds[1])
    prior_dist = prior.Prior(DTYPE, DEVICE)
    prior_dist = prior_dist.construct_prior(t_nd, 17, 10 * BATCH_SIZE, BATCH_SIZE, segs, prior_bounds, t_global_scale=100, num_iterations=300)
    print(prior_dist)
    # -------------------- END PRIOR CONSTRUCTION -------------------- #

    ''''# -------------------- BEGIN NATURAL FREQUENCY CALCULATION -------------------- #
    # set up simulator for spontaneous oscillations
    # initial conditions
    init_pos = np.random.randint(0, 10, size=(BATCH_SIZE, 2))
    init_probs = np.random.randint(0, 1, size=(BATCH_SIZE, 3))
    inits = helpers.concat(init_pos, init_probs)  # size: (BATCH_SIZE, 5)
    inits = torch.tensor(inits, dtype=DTYPE, device=DEVICE)
    inits_0 = inits[0, :].unsqueeze(0)

    sim = simulator.Simulator(torch.tensor(params, dtype=DTYPE, device=torch.device('cpu')).unsqueeze(0),
                              fdt.force(t_nd, 0, 0, 0, 0, 1),
                              inits_0.to(torch.device('cpu')), t_nd.to(torch.device('cpu')), segs=segs)
    x0 = sim.simulate()[0, 0, 0]
    t_dim = model_helpers.rescale_t(t_nd, *t_rescale_params).cpu().detach().numpy()
    x0_dim = model_helpers.rescale_x(x0, *x_rescale_params).cpu().detach().numpy()
    helpers.plot(units_rescale['time'] * t_dim[steady_id:], units_rescale['distance'] * x0_dim[steady_id:], labels=(r'Time (s)', r'$x_{0}$ (m)'))
    x0 = x0[steady_id:]
    print(x0.unsqueeze(0))
    x0_summary_stats = stats.get_summary_statistics(x0.unsqueeze(0), dt, n=30)
    omega_center = 2 * np.pi * pos_freqs[torch.argmax(torch.abs(torch.fft.rfft(x0 - torch.mean(x0))))]
    print(f"Frequency of spontaneous oscillations: {omega_center / (2 * np.pi * units_rescale['time'])} Hz")

    priors = []
    for param_vals in parameters_with_bounds.values():
        curr_prior = utils.BoxUniform(low=torch.ones(1) * param_vals[0] / 2, high=torch.ones(1) * 3 * param_vals[0] / 2)
        priors.append(curr_prior)
    prior = utils.MultipleIndependent(priors, device=str(DEVICE))
    thetas = prior.sample((BATCH_SIZE,)).to(dtype=DTYPE)
    for i in range(BATCH_SIZE):
        thetas[i, 3] = 0

    sim = simulator.Simulator(thetas, fdt.force(t_nd, 0, 1, 0, 0, BATCH_SIZE), inits, t_nd, segs=segs, batch_size=BATCH_SIZE, device=DEVICE)
    x = sim.simulate()[0, 0, :, :]
    x = x[:, steady_id:]

    print(x)
    summary_stats = torch.zeros((BATCH_SIZE, x0_summary_stats.shape[1]), dtype=torch.float32, device=torch.device('cpu'))
    step = summary_stats.shape[0] // 4
    for i in range(0, summary_stats.shape[0], step):
        start, end = i, min(i + step, summary_stats.shape[0])
        summary_stats[start:end] = stats.get_summary_statistics(x[start:end], dt, n=30)
    print(summary_stats)
    embedding_net = CNNEmbedding(input_shape=(1, n))
    neural_posterior = posterior_nn(model='nsf', embedding_net=embedding_net)
    inference = NPE(prior=prior, device=str(DEVICE), density_estimator=neural_posterior)
    density_estimator = inference.append_simulations(thetas.to(dtype=torch.float32), summary_stats.to(dtype=torch.float32)).train(training_batch_size=128, show_train_summary=True)
    posterior = inference.build_posterior(density_estimator=density_estimator)
    samples = posterior.sample((1000,), x=x0_summary_stats)
    pairplot(samples)'''