from torch import nn
import torch

class EmbeddedNet(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, layer_dims: tuple,
                 forcing_dim: int = 0, forcing_layer_dims: tuple = None,
                 merge_layer_dim: int = None):
        """
        Embedding network with optional conditioning on forcing parameters.

        :param input_dim: width of the leading conditioning block routed to the summary
            pathway (summary statistics; in this pipeline log(T) is grouped here too)
        :param output_dim: final output dimension
        :param layer_dims: hidden layer dims for the summary pathway
        :param forcing_dim: width of the trailing conditioning block routed to the forcing
            pathway, e.g. the forcing parameters (0 = unconditioned)
        :param forcing_layer_dims: hidden layer dims for the forcing pathway
        :param merge_layer_dim: hidden layer dim for the merge pathway
        """
        super().__init__()

        self.input_dim = input_dim
        self.forcing_dim = forcing_dim
        self.conditioned = forcing_dim > 0

        # Summary pathway
        self.summary_net = nn.Sequential(
            nn.Linear(input_dim, layer_dims[0]),
            nn.LayerNorm(layer_dims[0]),
            nn.GELU(),

            nn.Linear(layer_dims[0], layer_dims[1]),
            nn.LayerNorm(layer_dims[1]),
            nn.GELU(),
        )

        if self.conditioned:
            if forcing_layer_dims is None or merge_layer_dim is None:
                raise ValueError(
                    "forcing_layer_dims and merge_layer_dim must be provided when forcing_dim > 0"
                )

            self.forcing_net = nn.Sequential(
                nn.Linear(forcing_dim, forcing_layer_dims[0]),
                nn.LayerNorm(forcing_layer_dims[0]),
                nn.GELU(),

                nn.Linear(forcing_layer_dims[0], forcing_layer_dims[1]),
                nn.LayerNorm(forcing_layer_dims[1]),
                nn.GELU(),
            )

            self.merge_net = nn.Sequential(
                nn.Linear(layer_dims[1] + forcing_layer_dims[1], merge_layer_dim),
                nn.LayerNorm(merge_layer_dim),
                nn.GELU(),

                nn.Linear(merge_layer_dim, output_dim),
            )
        else:
            # No forcing: just project summary output to final dimension
            self.output_net = nn.Linear(layer_dims[1], output_dim)

    def forward(self, x):
        if self.conditioned:
            s = x[:, :self.input_dim]
            f = x[:, self.input_dim:]
            return self.merge_net(torch.cat([
                self.summary_net(s),
                self.forcing_net(f)
            ], dim=-1))
        else:
            return self.output_net(self.summary_net(x))