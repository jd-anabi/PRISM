from abc import abstractmethod, ABC

import torch
from sbi.utils import MultipleIndependent


class ForcingPrior(ABC):
    def __init__(self, dtype: torch.dtype = torch.float32, device: torch.device = torch.device('cpu')):
        self.dtype = dtype
        self.device = device

    @abstractmethod
    def construct_prior(self, bounds: list[tuple], types: tuple[str]) -> MultipleIndependent:
        pass