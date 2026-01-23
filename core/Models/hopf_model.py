import torch

K_B: float = 1.38e-23 # m^2 kg s^-2 K^-1

class HopfModel:
    def __init__(self, mu: torch.Tensor, omega: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor, epsilon: torch.Tensor,
                 force: torch.Tensor, batch_size: int, device: torch.device = torch.device('cpu'), dtype: torch.dtype = torch.float32):
        # sde model parameters
        self.batch_size = batch_size
        self.device = device
        self.dtype = dtype

        # parameters
        self.mu = mu.to(dtype=self.dtype, device=self.device)
        self.omega = omega.to(dtype=self.dtype, device=self.device)
        self.alpha = alpha.to(dtype=self.dtype, device=self.device)
        self.beta = beta.to(dtype=self.dtype, device=self.device)

        # force parameters
        self.force = force.to(dtype=self.dtype, device=self.device)

    def f(self, x, t) -> torch.Tensor:
        dx = self._x_dot(x[:, 0], x[:, 1])
        dy = self._y_dot(x[:, 0], x[:, 1], x[:, 2])
        dx = dx + self.force[:, t] / self.lam
        dx = torch.stack((dx, dy), dim=1)
        return dx

    def g(self, x) -> torch.Tensor:
        x_noise = self._x_noise()
        y_noise = self._y_noise()
        dsigma = torch.stack((x_noise, y_noise), dim=0)
        dsigma = torch.atleast_2d(torch.transpose(dsigma, -1, 0))
        return torch.diag_embed(dsigma)

    # --- SDEs --- #
    def _x_dot(self, x, y) -> torch.Tensor:
        linear_term = self.mu * x - self.omega * y
        quadratic_term = self.beta * torch.pow(x, 2) * y - self.alpha * x * torch.pow(y, 2)
        cubic_term = -1 * self.alpha * torch.pow(x, 3) + self.beta * torch.pow(y, 3)
        return linear_term + quadratic_term + cubic_term

    def _y_dot(self, x, y, c) -> torch.Tensor:
        linear_term = self.omega * x + self.mu * y
        quadratic_term = -1 * self.alpha * torch.pow(x, 2) * y - self.beta * x * torch.pow(y, 2)
        cubic_term = -1 * self.beta * torch.pow(x, 3) - self.alpha * torch.pow(y, 3)
        return linear_term + quadratic_term + cubic_term

    # --- NOISE --- #
    def _x_noise(self) -> torch.Tensor:
        return self.eta_x * torch.sqrt(2 * K_B * self.temp * self.lam)

    def _y_noise(self) -> torch.Tensor:
        return self.eta_y * torch.sqrt(2 * K_B * self.temp_eff * self.lam_y)