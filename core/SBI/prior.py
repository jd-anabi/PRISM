import math

import numpy as np
import torch
from sbi import utils

from core.Helpers import helpers
from core.Simulator import simulator

class Prior:
    def __init__(self, dtype: torch.dtype = torch.float32,device: torch.device = torch.device('cpu')):
        self.dtype = dtype
        self.device = device

    # ------------------ PRIVATE METHODS ------------------
    def __global_map(self, t: torch.Tensor, n_params: int, prior_bounds: tuple, segs: int, batch_size: int, num_iterations: int, steady: bool) -> list:
        if batch_size % num_iterations != 0:
            raise ValueError('batch_size must be divisible by num_iterations')
        curr_batch_size = batch_size // num_iterations

        wide_prior = utils.BoxUniform(low=torch.ones(n_params) * prior_bounds[0], high=torch.ones(n_params) * prior_bounds[1], device=str(self.device))
        thetas = wide_prior.sample((batch_size,))
        if steady:
            for i in range(thetas.shape[0]):
                thetas[i, 3] = 0

        init_pos = np.random.randint(0, 10, size=(curr_batch_size, 2))
        init_probs = np.random.randint(0, 1, size=(curr_batch_size, 3))
        inits = helpers.concat(init_pos, init_probs)  # size: (BATCH_SIZE, 5)
        inits = torch.tensor(inits, dtype=self.dtype, device=self.device)
        force = torch.zeros((curr_batch_size, t.shape[0]), dtype=self.dtype, device=self.device)
        stable_params = []

        for i in range(num_iterations - 1):
            sim = simulator.Simulator(thetas[i*curr_batch_size:(i+1)*curr_batch_size], force, inits, t, segs=segs, batch_size=curr_batch_size, device=self.device)
            x = sim.simulate()[0, 0, :, :]
            is_valid = torch.isfinite(x).all(dim=1)
            stable_params.append(thetas[is_valid])
        return stable_params