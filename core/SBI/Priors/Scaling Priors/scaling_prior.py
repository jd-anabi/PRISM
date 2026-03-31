import torch
from torch.distributions import Uniform, TransformedDistribution, ExpTransform

from sbi.utils import MultipleIndependent

class ScalingPrior:
    def __init__(self, dtype: torch.dtype = torch.float32, device: torch.device = torch.device('cpu')):
        self.dtype = dtype
        self.device = device

    def construct_prior(self, bounds: list[tuple], types: tuple[str]) -> MultipleIndependent:
        if len(types) != 3:
            raise ValueError(f"types must be of length 3, got {len(types)}")
        if len(bounds) != 3:
            raise ValueError(f"bounds must be of length 3, got {len(bounds)}")

        allowed = {"uniform", "log-uni"}
        for t in types:
            if t not in allowed:
                raise ValueError(f"Invalid type '{t}'. Allowed types are: {allowed}")

        marginals = []
        for i in range(3):
            low = torch.tensor(bounds[i][0], dtype=self.dtype, device=self.device)
            high = torch.tensor(bounds[i][1], dtype=self.dtype, device=self.device)

            if types[i] == "uniform":
                marginals.append(Uniform(low, high))
            elif types[i] == "log-uni":
                base = Uniform(torch.log(low), torch.log(high))
                marginals.append(TransformedDistribution(base, ExpTransform()))

        return MultipleIndependent(marginals)
