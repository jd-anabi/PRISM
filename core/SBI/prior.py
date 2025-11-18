import math

import numpy as np
import torch
from sbi import utils
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from core.Helpers import helpers
from core.Simulator import simulator

class Prior:
    def __init__(self, dtype: torch.dtype = torch.float32,device: torch.device = torch.device('cpu')):
        self.dtype = dtype
        self.device = device

    # ------------------ PUBLIC METHODS ------------------
    def construct_prior(self, t: torch.Tensor, n_params, batch_size_limit, num_iterations, steady: bool) -> torch.distributions.Distribution:
        segs = math.ceil(t[-1] / 100)
        n_sims = batch_size_limit * num_iterations

        # do global sweep and find number of "islands"
        progress_bar = tqdm(total=4, desc="Doing global sweep for prior construction...")
        stable_params = self.__global_map(t[:(t.shape[0] // 100)], n_params, (-1000, 1000), segs, n_sims, num_iterations, steady)
        stable_params_arr = np.array(stable_params)
        progress_bar.update(1)
        scaler = StandardScaler()
        stable_params_scaled = scaler.fit_transform(stable_params_arr)
        progress_bar.update(2)
        db = DBSCAN(eps=2.5, min_samples=50).fit(stable_params_scaled)
        progress_bar.update(3)
        labels = db.labels_
        n_clusters = len(labels) - (1 if -1 in labels else 0)
        progress_bar.update(4)
        progress_bar.close()

        # do local sweep
        progress_bar = tqdm(total=1, desc="Doing local sweep for prior construction...")
        accepted_params = np.array(self.__local_map(t, stable_params, batch_size_limit, n_params, int(2e5), 0.05, segs, steady))
        progress_bar.update(1)
        progress_bar.close()

        # finally construct prior using Gaussian-Mixture Model'
        progress_bar = tqdm(total=5, desc="Constructing prior...")
        gmm = GaussianMixture(n_components=n_clusters, covariance_type='full').fit(accepted_params)
        progress_bar.update(1)
        means = torch.tensor(gmm.means_, dtype=self.dtype, device=self.device)
        cov = torch.tensor(gmm.covariances_, dtype=self.dtype, device=self.device)
        weights = torch.tensor(gmm.weights_, dtype=self.dtype, device=self.device)
        progress_bar.update(2)
        comp_dist = torch.distributions.MultivariateNormal(means, covariance_matrix=cov)
        progress_bar.update(3)
        mix_dist = torch.distributions.Categorical(probs=weights)
        progress_bar.update(4)
        prior = torch.distributions.MixtureSameFamily(mix_dist, comp_dist)
        progress_bar.update(5)
        progress_bar.close()

        return prior

    # ------------------ PRIVATE METHODS ------------------
    def __global_map(self, t: torch.Tensor, n_params: int, prior_bounds: tuple, segs: int, batch_size: int, num_iterations: int, steady: bool) -> list:
        t = t.to(dtype=self.dtype, device=self.device)
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

    @staticmethod
    def __local_map(t: torch.Tensor, stable_params: list, batch_size: int, n_params: int, n_max: int, step: float, segs: int, steady: bool) -> set:
        # gpu variables
        dtype = torch.float32
        device = torch.device('cpu')
        t = t.to(dtype=dtype, device=device)

        # algorithm variables
        queue = list(stable_params)
        accepted_params = set(stable_params)

        # check if steady-state
        steps = torch.full((batch_size,), step, dtype=dtype, device=device)
        if steady:
            for i in range(steps.shape[0]):
                steps[i] = 0

        # SDE variable
        init_pos = np.random.randint(0, 10, size=(batch_size, 2))
        init_probs = np.random.randint(0, 1, size=(batch_size, 3))
        inits = helpers.concat(init_pos, init_probs)  # size: (BATCH_SIZE, 5)
        inits = torch.tensor(inits, dtype=dtype, device=device)
        force = torch.zeros((batch_size, t.shape[0]), dtype=dtype, device=device)

        # begin algorithm
        while len(queue) != 0 and len(accepted_params) <= n_max:
            thetas = torch.tensor(queue.pop(0), dtype=dtype, device=device) + torch.randn((batch_size, n_params), dtype=dtype, device=device) * steps.unsqueeze(1)
            sim = simulator.Simulator(thetas, force, inits, t, segs=segs, batch_size=batch_size, device=device)
            x = sim.simulate()[0, 0, :, :]
            is_valid = torch.isfinite(x).all(dim=1)
            for i in range(batch_size):
                if is_valid[i]:
                    contains = False
                    stable_point = thetas[i]
                    for theta in accepted_params:
                        if torch.equal(theta, stable_point):
                            contains = True
                    if not contains:
                        accepted_params.add(stable_point)
                        queue.append(stable_point)
                        n_max += 1

        return accepted_params