"""
Log-uniform distribution compatible with sbi's MultipleIndependent.to().

sbi's device-move path reconstructs each marginal via `type(dist)(**params)` where
`params` is extracted using the distribution's `arg_constraints` keys. Standard
`torch.distributions.TransformedDistribution(Uniform(log(low), log(high)), ExpTransform())`
has `arg_constraints = {}`, so the reconstruction fails. This class exposes
`low` and `high` as reconstructible args, so sbi can move it to a device cleanly.
"""
import torch
from torch.distributions import Distribution, constraints


class LogUniform(Distribution):
    """
    Log-uniform distribution on (low, high): x such that log(x) ~ Uniform(log(low), log(high)).

    Compatible with sbi's MultipleIndependent.to() because:
      - `arg_constraints` has 'low' and 'high' — extractable keys for reconstruction
      - `__init__` accepts low, high as keyword arguments

    Density: p(x) = 1 / (x * (log(high) - log(low))) for low <= x <= high.
    """
    arg_constraints = {
        "low": constraints.positive,
        "high": constraints.positive,
    }
    support = constraints.positive
    has_rsample = True

    def __init__(self, low, high, validate_args=None):
        self.low = low
        self.high = high
        if isinstance(low, torch.Tensor):
            batch_shape = low.shape
        else:
            batch_shape = torch.Size()
        super().__init__(batch_shape=batch_shape, validate_args=validate_args)

    @property
    def log_low(self) -> torch.Tensor:
        return torch.log(self.low)

    @property
    def log_high(self) -> torch.Tensor:
        return torch.log(self.high)

    def sample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        with torch.no_grad():
            return self.rsample(sample_shape)

    def rsample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        shape = self._extended_shape(sample_shape)
        u = torch.rand(shape, dtype=self.low.dtype, device=self.low.device)
        return torch.exp(self.log_low + u * (self.log_high - self.log_low))

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        if self._validate_args:
            self._validate_sample(value)
        # p(x) = 1 / (x * (log_high - log_low))  =>  log p(x) = -log(x) - log(log_high - log_low)
        return -torch.log(value) - torch.log(self.log_high - self.log_low)

    @property
    def mean(self) -> torch.Tensor:
        # E[X] = (high - low) / (log(high) - log(low))
        return (self.high - self.low) / (self.log_high - self.log_low)

    @property
    def variance(self) -> torch.Tensor:
        # E[X^2] = (high^2 - low^2) / (2 * (log(high) - log(low)))
        second_moment = (self.high ** 2 - self.low ** 2) / (2 * (self.log_high - self.log_low))
        return second_moment - self.mean ** 2

    def cdf(self, value: torch.Tensor) -> torch.Tensor:
        return (torch.log(value) - self.log_low) / (self.log_high - self.log_low)

    def icdf(self, value: torch.Tensor) -> torch.Tensor:
        return torch.exp(self.log_low + value * (self.log_high - self.log_low))
