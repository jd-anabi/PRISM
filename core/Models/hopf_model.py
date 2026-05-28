import torch


@torch.jit.script
def _hopf_compiled_step(
    x: torch.Tensor, force_step: torch.Tensor, dW: torch.Tensor,
    mu: torch.Tensor, beta: torch.Tensor, g: torch.Tensor,
    dt: float, sqrt_dt: float,
) -> torch.Tensor:
    """One Euler-Maruyama step for Hopf, JIT-scripted for Python-overhead removal."""
    x0 = x[:, 0]
    x1 = x[:, 1]
    r2 = x0 * x0 + x1 * x1
    dx = mu * x0 - x1 - (x0 - beta * x1) * r2 + force_step[:, 0]
    dy = x0 + mu * x1 - (beta * x0 + x1) * r2 + force_step[:, 1]
    drift = torch.stack((dx, dy), dim=1)
    return x + drift * dt + g * dW * sqrt_dt


class HopfModel:
    # Module-level compiled step exposed for sdeint.euler_compiled.
    compiled_step = staticmethod(_hopf_compiled_step)

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
        return self.f_pure(x, self.force[:, :, t])

    def f_pure(self, x: torch.Tensor, force_step: torch.Tensor) -> torch.Tensor:
        """Drift as a pure function of state and a pre-sliced force vector."""
        x0 = x[:, 0]
        x1 = x[:, 1]
        r2 = x0 * x0 + x1 * x1
        dx = self._x_dot(x0, x1, r2) + force_step[:, 0]
        dy = self._y_dot(x0, x1, r2) + force_step[:, 1]
        return torch.stack((dx, dy), dim=1)

    def g(self) -> torch.Tensor:
        # Diagonal noise as a (batch, d) vector; solver multiplies elementwise.
        return torch.stack((self.sigma_x, self.sigma_y), dim=-1)

    def compiled_params(self) -> tuple:
        """Tuple of tensor params for `compiled_step` (after x, force_step, dW)."""
        return (self.mu, self.beta, self.g())

    # --- SDEs --- #
    def _x_dot(self, x, y, r2) -> torch.Tensor:
        return self.mu * x - y - (x - self.beta * y) * r2

    def _y_dot(self, x, y, r2) -> torch.Tensor:
        return x + self.mu * y - (self.beta * x + y) * r2

    # --- NOISE --- #
    def _x_noise(self) -> torch.Tensor:
        return self.sigma_x

    def _y_noise(self) -> torch.Tensor:
        return self.sigma_y