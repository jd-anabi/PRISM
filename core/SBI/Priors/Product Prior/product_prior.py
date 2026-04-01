import torch
from torch.distributions import Distribution

class ProductPrior(Distribution):
    def __init__(self, distributions: list, dims: list):
        """
        Product prior over independent groups of parameters.

        :param distributions: list of distribution objects, each supporting
                              .sample() and .log_prob()
        :param dims: list of ints, the dimensionality of each distribution.
                     Must sum to the total parameter dimension.
        """
        self.distributions = distributions
        self.dims = dims
        self._total_dim = sum(dims)
        super().__init__()

    def sample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        samples = [d.sample(sample_shape) for d in self.distributions]
        return torch.cat(samples, dim=-1)

    def log_prob(self, value: torch.Tensor) -> int:
        log_probs = []
        idx = 0
        for d, dim in zip(self.distributions, self.dims):
            log_probs.append(d.log_prob(value[..., idx:idx + dim]))
            idx += dim
        return sum(log_probs)