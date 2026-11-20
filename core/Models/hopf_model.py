import torch


@torch.jit.script
def _hopf_compiled_step(
    x: torch.Tensor, force_step: torch.Tensor, dW: torch.Tensor,
    mu: torch.Tensor, beta: torch.Tensor, g: torch.Tensor,
    dt: torch.Tensor, sqrt_dt: torch.Tensor,
) -> torch.Tensor:
    """One Euler-Maruyama step for Hopf, JIT-scripted for Python-overhead removal."""
    x0 = x[:, 0]
    x1 = x[:, 1]
    r2 = x0 * x0 + x1 * x1
    dx = mu * x0 - x1 - (x0 - beta * x1) * r2 + force_step[:, 0]
    dy = x0 + mu * x1 - (beta * x0 + x1) * r2 + force_step[:, 1]
    drift = torch.stack((dx, dy), dim=1)
    return x + drift * dt + g * dW * sqrt_dt


@torch.jit.script
def _hopf_drift(x: torch.Tensor, force_step: torch.Tensor, mu: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    """The drift at ``x`` under the drive ``force_step``, in the plain compiled step's operation order."""
    x0 = x[:, 0]
    x1 = x[:, 1]
    r2 = x0 * x0 + x1 * x1
    dx = mu * x0 - x1 - (x0 - beta * x1) * r2 + force_step[:, 0]
    dy = x0 + mu * x1 - (beta * x0 + x1) * r2 + force_step[:, 1]
    return torch.stack((dx, dy), dim=1)


@torch.jit.script
def _hopf_compiled_corrected_step(
    x: torch.Tensor, force_now: torch.Tensor, force_next: torch.Tensor, dW: torch.Tensor,
    mu: torch.Tensor, beta: torch.Tensor, g: torch.Tensor,
    dt: torch.Tensor, sqrt_dt: torch.Tensor,
) -> torch.Tensor:
    """One stochastic Heun step for Hopf (``core.Solvers.sdeint.Solver.heun``), JIT-scripted: the plain step as a
    prediction, the drift again where it lands under the drive at the step's end, the two averaged; the noise is
    constant, so its average is itself."""
    f0 = _hopf_drift(x, force_now, mu, beta)
    e0 = g * dW * sqrt_dt
    xp = x + f0 * dt + e0
    f1 = _hopf_drift(xp, force_next, mu, beta)
    return x + 0.5 * (f0 + f1) * dt + 0.5 * (e0 + g * dW * sqrt_dt)


class HopfModel:
    # Module-level compiled steps exposed for sdeint.euler_compiled and sdeint.heun_compiled.
    compiled_step = staticmethod(_hopf_compiled_step)
    compiled_corrected_step = staticmethod(_hopf_compiled_corrected_step)
    # The corrected step integrates this model's equation: its noise is constant
    # (core.Simulator.simulator.Simulator.simulate).
    admits_corrected_step = True

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