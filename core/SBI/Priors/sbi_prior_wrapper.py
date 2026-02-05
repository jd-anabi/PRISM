import torch
from torch import distributions as dist

class SBIPriorWrapper(dist.Distribution):
    def __init__(self, gen_dist: torch.distributions.Distribution, n_samples: int = 10000):
        self.gen_dist = gen_dist
        super().__init__(batch_shape=gen_dist.batch_shape, event_shape=gen_dist.event_shape)

        # precompute samples and mean/stddev
        samples = self.gen_dist.sample((n_samples,))
        self._mean = samples.mean(dim=0)
        self._stddev = samples.std(dim=0)

    @property
    def support(self):
        return dist.constraints.independent(dist.constraints.real, 1)

    @property
    def mean(self):
        return self._mean

    @property
    def stddev(self):
        return self._stddev

    def log_prob(self, value):
        return self.gen_dist.log_prob(value)

    def sample(self, sample_shape=torch.Size()):
        return self.gen_dist.sample(sample_shape)