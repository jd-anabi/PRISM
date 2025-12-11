from collections import deque

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

    # --- PUBLIC METHODS --- #
    def construct_prior(self, t: torch.Tensor, n_params: int, global_batch_size: int, local_batch_size: int, segs: int, prior_bounds: list[tuple],
                        t_global_scale: int = 1, num_iterations: int = 25, steady: bool = True, n_max: int = 200000, step: float = 0.01) -> torch.distributions.MixtureSameFamily:
        n_sims = global_batch_size * num_iterations

        # do global sweep and find number of "islands"
        stable_params = self.__global_map(t[:(t.shape[0] // t_global_scale)], n_params, prior_bounds, segs, n_sims, num_iterations, steady)
        stable_params_arr = np.array(stable_params)
        scaler = StandardScaler()

        stable_params_scaled = scaler.fit_transform(stable_params_arr)
        db = DBSCAN(eps=2.5, min_samples=50).fit(stable_params_scaled)
        labels = db.labels_
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        if n_clusters < 1:
            print('No clusters found. Defaulting to 1 cluster')
            n_clusters = 1

        # do local sweep
        accepted_params = np.array(self.__local_map(t, stable_params, local_batch_size, n_params, n_max, step, segs, steady))

        # safety check
        if accepted_params.shape[0] < n_clusters:
            raise ValueError(f"Not enough stable parameter sets ({accepted_params.shape[0]}) to fit {n_clusters} GMM components")

        # finally construct prior using Gaussian-Mixture Model'
        progress_bar = tqdm(total=5, desc="Constructing prior...")
        gmm = GaussianMixture(n_components=n_clusters, covariance_type='full').fit(accepted_params)
        progress_bar.update()
        means = torch.tensor(gmm.means_, dtype=self.dtype, device=self.device)
        cov = torch.tensor(gmm.covariances_, dtype=self.dtype, device=self.device)
        weights = torch.tensor(gmm.weights_, dtype=self.dtype, device=self.device)
        progress_bar.update()
        comp_dist = torch.distributions.MultivariateNormal(means, covariance_matrix=cov)
        progress_bar.update()
        mix_dist = torch.distributions.Categorical(probs=weights)
        progress_bar.update()
        prior = torch.distributions.MixtureSameFamily(mix_dist, comp_dist)
        progress_bar.update()
        progress_bar.close()

        return prior

    # --- PRIVATE METHODS --- #
    def __global_map(self, t: torch.Tensor, n_params: int, prior_bounds: list[tuple], segs: int, batch_size: int, num_iterations: int, steady: bool) -> list:
        t = t.to(dtype=self.dtype, device=self.device)
        if batch_size % num_iterations != 0:
            raise ValueError('batch_size must be divisible by num_iterations')
        curr_batch_size = batch_size // num_iterations

        priors = []
        for curr_bounds in prior_bounds:
            curr_prior = utils.BoxUniform(low=torch.ones(1) * curr_bounds[0], high=torch.ones(1) * curr_bounds[1])
            priors.append(curr_prior)
        wide_prior = utils.MultipleIndependent(priors, device=str(self.device))
        thetas = wide_prior.sample((batch_size,)).to(dtype=self.dtype)
        if steady:
            thetas = torch.cat((thetas[:, :3], thetas[:, 4:]), dim=1)
            n_params -= 1

        init_pos = np.random.randint(0, 10, size=(curr_batch_size, 2))
        init_probs = np.random.randint(0, 1, size=(curr_batch_size, 3))
        inits = helpers.concat(init_pos, init_probs)  # size: (BATCH_SIZE, 5)
        inits = torch.tensor(inits, dtype=self.dtype, device=self.device)
        force = torch.zeros((curr_batch_size, t.shape[0]), dtype=self.dtype, device=self.device)
        stable_params = []

        num_added = 0
        added_params_progress_bar = tqdm(total=(num_iterations - 1), desc=f"Added {num_added} sets to accepted parameters during global sweep", leave=False)
        with torch.no_grad():
            for i in range(num_iterations - 1):
                curr_thetas = thetas[i*curr_batch_size:(i+1)*curr_batch_size]
                sim = simulator.Simulator(curr_thetas, force, inits, t, segs=segs, batch_size=curr_batch_size, device=self.device)
                x = sim.simulate()[0, 0, :, :] # shape: (curr_batch_size, len(t))
                is_valid = torch.isfinite(x).all(dim=1)
                valid_params = curr_thetas[is_valid]
                if valid_params.shape[0] > 0:
                    num_added += valid_params.shape[0]
                added_params_progress_bar.update()
                added_params_progress_bar.set_description(f"Added {num_added} sets to accepted parameters during global sweep")
                del x
                stable_params.extend(valid_params.detach().cpu().tolist())
        added_params_progress_bar.close()
        return list(set(stable_params))

    @staticmethod
    def __local_map(t: torch.Tensor, stable_params: list, batch_size: int, n_params: int, n_max: int, step: float, segs: int, steady: bool) -> list:
        # cpu variables
        dtype = torch.float32
        device = torch.device('cpu')
        t = t.to(dtype=dtype, device=device)

        # algorithm variables
        queue = deque(stable_params)
        accepted_params = set([tuple(p) for p in stable_params])

        # check if steady-state
        if steady:
            n_params -= 1

        # SDE variable
        init_pos = np.random.randint(0, 10, size=(batch_size, 2))
        init_probs = np.random.randint(0, 1, size=(batch_size, 3))
        inits = helpers.concat(init_pos, init_probs)  # size: (BATCH_SIZE, 5)
        inits = torch.tensor(inits, dtype=dtype, device=device)
        force = torch.zeros((batch_size, t.shape[0]), dtype=dtype, device=device)

        # begin algorithm
        num_added = 0
        added_params_progress_bar = tqdm(total=batch_size, desc=f"Added {num_added} sets to accepted parameters during local sweep. Total parameter sets: {len(accepted_params)}. Number of parameter sets to check: {len(queue)}", leave=False)
        with torch.no_grad():
            while len(queue) != 0 and len(accepted_params) <= n_max:
                thetas = torch.tensor(queue.popleft(), dtype=dtype, device=device) + torch.randn((batch_size, n_params), dtype=dtype, device=device) * step
                sim = simulator.Simulator(thetas, force, inits, t, segs=segs, batch_size=batch_size, device=device)
                x = sim.simulate()[0, 0, :, :] # shape: (batch_size, len(t))
                is_valid = torch.isfinite(x).all(dim=1)
                for i in range(batch_size):
                    if is_valid[i]:
                        stable_point = tuple(thetas[i].tolist())
                        if not stable_point in accepted_params:
                            accepted_params.add(stable_point)
                            queue.append(stable_point)
                            num_added += 1
                    added_params_progress_bar.update()
                added_params_progress_bar.reset()
                added_params_progress_bar.set_description(f"Added {num_added} sets to accepted parameters during local sweep. Total parameter sets: {len(accepted_params)}. Number of parameter sets to check: {len(queue)}")
                del x
        added_params_progress_bar.close()

        return list(accepted_params)