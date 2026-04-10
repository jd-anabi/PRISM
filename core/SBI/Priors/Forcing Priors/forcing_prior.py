import torch
from torch.distributions import Uniform, TransformedDistribution, ExpTransform

from sbi.utils import MultipleIndependent


class ForcingPrior:
    def __init__(self, dtype: torch.dtype = torch.float32, device: torch.device = torch.device('cpu')):
        self.dtype = dtype
        self.device = device

    def construct_prior(self, bounds: list[tuple], types: tuple[str]) -> MultipleIndependent:
        """
        Construct a prior over forcing parameters with uniform or log-uniform marginals.

        :param bounds: List of (low, high) tuples, one per parameter.
        :param types: Tuple of distribution types, one per parameter ("uniform" or "log-uni").
        :return: A MultipleIndependent distribution over all forcing parameters.
        """
        if len(bounds) != len(types):
            raise ValueError(f"bounds length ({len(bounds)}) must match types length ({len(types)})")

        allowed = {"uniform", "log-uni"}
        for t in types:
            if t not in allowed:
                raise ValueError(f"Invalid type '{t}'. Allowed types are: {allowed}")

        marginals = []
        for i in range(len(bounds)):
            low = torch.tensor([bounds[i][0]], dtype=self.dtype, device=self.device)
            high = torch.tensor([bounds[i][1]], dtype=self.dtype, device=self.device)

            if types[i] == "uniform":
                marginals.append(Uniform(low, high))
            elif types[i] == "log-uni":
                base = Uniform(torch.log(low), torch.log(high))
                marginals.append(TransformedDistribution(base, ExpTransform()))

        return MultipleIndependent(marginals)
