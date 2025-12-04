import torch
from torch import nn

class EmbeddedNet(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
