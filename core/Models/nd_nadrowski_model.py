import torch

class NDNadrowskiModel:
    def __init__(self, k: torch.Tensor, lam: torch.Tensor, f: torch.Tensor, tau: torch.Tensor, tau_c: torch.Tensor,
                 c_0: torch.Tensor, s: torch.Tensor, delta_e: torch.Tensor, beta: torch.Tensor, n: torch.Tensor, temp: torch.Tensor,
                 force: torch.Tensor, batch_size: int, device: torch.device = torch.device('cpu'), dtype: torch.dtype = torch.float32):
        # sde model parameters
        self.batch_size = batch_size
        self.device = device
        self.dtype = dtype

        # parameters
        self.k = k.to(self.device)
        self.lam = lam.to(self.device)
        self.f_max = f.to(self.device)
        self.tau = tau.to(self.device)
        self.tau_c = tau_c.to(self.device)
        self.c_0 = c_0.to(self.device)
        self.s = s.to(self.device)
        self.delta_e = delta_e.to(self.device)
        self.beta = beta.to(self.device)
        self.n = n.to(self.device)
        self.temp = temp.to(self.device)

        # force parameters
        self.force = force.to(self.device)

        # subsuming parameters
        self.a = torch.exp(self.delta_e + self.beta / 2)

    def f(self, x, t) -> torch.Tensor:
        dx = self._x_dot(x[:, 0], x[:, 1])
        dy = self._y_dot(x[:, 0], x[:, 1], x[:, 2])
        dc = self._c_dot(x[:, 0], x[:, 1], x[:, 2])
        dx = dx + self.force[:, t]
        dx = torch.stack((dx, dy, dc), dim=1)
        return dx

    def g(self, x) -> torch.Tensor:
        x_noise = self._x_noise()
        y_noise = self._y_noise()
        c_noise = self._c_noise(x[:, 0], x[:, 1])
        dsigma = torch.stack((x_noise, y_noise, c_noise), dim=0)
        dsigma = torch.atleast_2d(torch.transpose(dsigma, -1, 0))
        return torch.diag_embed(dsigma)

    # --- SDEs --- #
    def _x_dot(self, x, y) -> torch.Tensor:
        x_gs = (x - y - self.__p_t0(x, y))
        x_sp = self.k * x
        return -1 * (x_gs + x_sp)

    def _y_dot(self, x, y, c) -> torch.Tensor:
        x_gs = (x - y - self.__p_t0(x, y))
        f_c = self.f_max * (1 - self.s * c)
        return (x_gs - f_c) / self.lam

    def _c_dot(self, x, y, c) -> torch.Tensor:
        return (self.c_0 - c + self.__p_t0(x, y)) / self.tau

    # --- NOISE --- #
    def _x_noise(self) -> torch.Tensor:
        return torch.sqrt(2 / (self.n * self.beta))

    def _y_noise(self) -> torch.Tensor:
        return torch.sqrt(2 * self.temp / (self.n * self.beta * self.lam))

    def _c_noise(self, x, y) -> torch.Tensor:
        return torch.sqrt(2 * self.tau_c * self.__p_t0(x, y) * (1 - self.__p_t0(x, y)) / self.n) / self.tau

    # --- PRIVATE --- #
    def __p_t0(self, x, y):
        return 1 / (1 + self.a * torch.exp(-1 * self.beta * (x - y)))