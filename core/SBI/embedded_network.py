from torch import nn

class EmbeddedNet(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, layer_dims: tuple):
        """
        Embedded neural network to compress summary statistics to lower-dimensional space
        :param input_dim: the input dimension (initial dimension of summary statistics)
        :param output_dim: the output dimension (desired compressed dimension)
        :param layer_dims: the dimensions of each intermediate layer in the neural network
        """
        super().__init__()
        self.net = nn.Sequential(
            # Layer 1: input dim -> first layer dim
            nn.Linear(input_dim, layer_dims[0]), # standard sully connected layer, learns linear combination of inputs
            nn.LayerNorm(layer_dims[0]), # stats have different scales, this forces stats to have a standard scale (mean = 0, variance = 1)
            nn.ReLU(), # rectified linear unit, introduces non-linearity between statistics

            # Layer 2: first layer dim -> second layer dim
            nn.Linear(layer_dims[0], layer_dims[1]),
            nn.LayerNorm(layer_dims[1]),
            nn.ReLU(),

            # Layer 3: second layer dim -> output dim
            nn.Linear(layer_dims[1], output_dim)
        )

    def forward(self, x):
        """
        Defines the forward pass of the embedded network
        :param x: input
        :return: output
        """
        return self.net(x)