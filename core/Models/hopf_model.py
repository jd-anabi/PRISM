import torch

class HopfModel:
    def __init__(self, mu: torch.Tensor, beta: torch.Tensor, sigma_x: torch.Tensor, sigma_y: torch.Tensor,
                 force: torch.Tensor, batch_size: int, device: torch.device = torch.device('cpu'), dtype: torch.dtype = torch.float32):
        # sde model parameters
        self.batch_size = batch_size
        self.device = device
        self.dtype = dtype

        # parameters
        self.mu = mu.to(dtype=self.dtype, device=self.device)
        self.beta = beta.to(dtype=self.dtype, device=self.device)
        self.sigma_x = sigma_x.to(dtype=self.dtype, device=self.device)
        self.sigma_y = sigma_y.to(dtype=self.dtype, device=self.device)

        # force parameters
        self.force = force.to(dtype=self.dtype, device=self.device)

    def f(self, x, t) -> torch.Tensor:
        dx = self._x_dot(x[:, 0], x[:, 1])
        dy = self._y_dot(x[:, 0], x[:, 1])
        dx = dx + self.force[:, 0, t]   # x-channel forcing
        dy = dy + self.force[:, 1, t]   # y-channel forcing (shared freq/phase/offset, distinct amp)
        dx = torch.stack((dx, dy), dim=1)
        return dx

    def g(self) -> torch.Tensor:
        x_noise = self._x_noise()
        y_noise = self._y_noise()
        dsigma = torch.stack((x_noise, y_noise), dim=0)
        dsigma = torch.atleast_2d(torch.transpose(dsigma, -1, 0))
        return torch.diag_embed(dsigma)

    # --- SDEs --- #
    def _x_dot(self, x, y) -> torch.Tensor:
        linear_term = self.mu * x - y
        cubic_term = (x - self.beta * y) * (torch.pow(x, 2) + torch.pow(y, 2))
        return linear_term - cubic_term

    def _y_dot(self, x, y) -> torch.Tensor:
        linear_term = x + self.mu * y
        cubic_term = (self.beta * x + y) * (torch.pow(x, 2) + torch.pow(y, 2))
        return linear_term - cubic_term

    # --- NOISE --- #
    def _x_noise(self) -> torch.Tensor:
        return self.sigma_x

    def _y_noise(self) -> torch.Tensor:
        return self.sigma_y