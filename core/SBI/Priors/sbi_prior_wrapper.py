import torch
from torch import distributions as dist

class SBIPriorWrapper(dist.Distribution):
    def __init__(self, gen_dist: torch.distributions.Distribution):
        self.gen_dist = gen_dist
        super().__init__(batch_shape=gen_dist.batch_shape, event_shape=gen_dist.event_shape)

    @property
    def support(self):
        return dist.constraints.independent(dist.constraints.real, 1)

    def log_prob(self, value):
        return self.gen_dist.log_prob(value)

    def sample(self, sample_shape=torch.Size()):
        return self.gen_dist.sample(sample_shape)