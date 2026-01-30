import numpy as np
import torch
import hdbscan
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from abc import ABC, abstractmethod

class Prior(ABC):
    def __init__(self, dtype: torch.dtype = torch.float32,device: torch.device = torch.device('cpu')):
        self.dtype = dtype
        self.device = device

    # --- PUBLIC METHODS --- #
    def construct_prior(self, t: torch.Tensor, n_params: int, global_batch_size: int, local_batch_size: int, segs: int, prior_bounds: list[tuple],
                        t_global_scale: int = 1, num_iterations: int = 25, steady: bool = True, n_max: int = 200000, step: float = 0.01) -> torch.distributions.MixtureSameFamily:
        n_sims = global_batch_size * num_iterations

        # do global sweep and find number of "islands"
        stable_params = self._global_map(t[:(t.shape[0] // t_global_scale)], n_params, prior_bounds, segs, n_sims, num_iterations, steady)

        # do local sweep
        accepted_params = np.array(self._local_map(t, stable_params, local_batch_size, n_params, n_max, step, segs, steady))

        scaler = StandardScaler()
        stable_params_scaled = scaler.fit_transform(accepted_params)
        clusterer = hdbscan.HDBSCAN(min_cluster_size=50, min_samples=10)
        labels = clusterer.fit_predict(stable_params_scaled)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        if n_clusters < 1:
            print('No clusters found. Defaulting to 1 cluster')
            n_clusters = 1
        else:
            print(f'Found {n_clusters} clusters')

        # safety check
        if accepted_params.shape[0] < n_clusters:
            raise ValueError(
                f"Not enough stable parameter sets ({accepted_params.shape[0]}) to fit {n_clusters} GMM components")

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
    @abstractmethod
    def _global_map(self, t: torch.Tensor, n_params: int, prior_bounds: list[tuple], segs: int, batch_size: int, num_iterations: int, steady: bool) -> list:
        pass

    @staticmethod
    @abstractmethod
    def _local_map(t: torch.Tensor, stable_params: list, batch_size: int, n_params: int, n_max: int, step: float, segs: int, steady: bool) -> list:
        pass