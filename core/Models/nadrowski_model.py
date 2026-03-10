import torch

K_B: float = 1.38e-23 # m^2 kg s^-2 K^-1

class NadrowskiModel:
    def __init__(self, lam: torch.Tensor, lam_y: torch.Tensor, tau: torch.Tensor,
                 k_gs: torch.Tensor, k_sp: torch.Tensor, d: torch.Tensor, f_max: torch.Tensor,
                 c_0: torch.Tensor, c_m: torch.Tensor, s: torch.Tensor, n: torch.Tensor,
                 delta_e: torch.Tensor, temp: torch.Tensor, temp_eff: torch.Tensor, tau_c: torch.Tensor,
                 force: torch.Tensor, batch_size: int, device: torch.device = torch.device('cpu'), dtype: torch.dtype = torch.float32):
        # sde model parameters
        self.batch_size = batch_size
        self.device = device
        self.dtype = dtype

        # parameters
        self.lam = lam.to(self.device)
        self.lam_y = lam_y.to(self.device)
        self.tau = tau.to(self.device)
        self.k_gs = k_gs.to(self.device)
        self.k_sp = k_sp.to(self.device)
        self.d = d.to(self.device)
        self.f_max = f_max.to(self.device) # 429 and 352 pN
        self.c_0 = c_0.to(self.device)
        self.c_m = c_m.to(self.device)
        self.s = s.to(self.device) # 0.95 and 0.65 work, 0 <= S <= 1
        self.n = n.to(self.device)
        self.delta_e = delta_e.to(self.device)
        self.temp = temp.to(self.device)
        self.temp_eff = temp_eff.to(self.device)
        self.tau_c = tau_c.to(self.device)

        # force parameters
        self.force = force.to(self.device)

        # subsuming parameters
        self.a = torch.exp((self.delta_e + self.k_gs * self.d**2 / (2 * self.n)) / (K_B * self.temp))
        self.delta = self.n * K_B * self.temp / (self.k_gs * self.d)

    def f(self, x, t) -> torch.Tensor:
        dx = self._x_dot(x[:, 0], x[:, 1])
        dy = self._y_dot(x[:, 0], x[:, 1], x[:, 2])
        dc = self._c_dot(x[:, 0], x[:, 1], x[:, 2])
        dx = dx + self.force[:, t] / self.lam
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
        x_gs = -1 * self.k_gs * (x - y - self.d * self.__p0(x, y))
        x_sp = -1 * self.k_sp * x
        return (x_gs + x_sp) / self.lam

    def _y_dot(self, x, y, c) -> torch.Tensor:
        x_gs = self.k_gs * (x - y - self.d * self.__p0(x, y))
        f = self.f_max * (1 - self.s * c / self.c_m)
        return -1 * (x_gs + f) / self.lam_y

    def _c_dot(self, x, y, c) -> torch.Tensor:
        return self.c_0 - c + self.__p0(x, y)

    # --- NOISE --- #
    def _x_noise(self) -> torch.Tensor:
        return torch.sqrt(2 * K_B * self.temp / self.lam)

    def _y_noise(self) -> torch.Tensor:
        return torch.sqrt(2 * K_B * self.temp_eff * self.temp / self.lam_y)

    def _c_noise(self, x, y) -> torch.Tensor:
        return torch.sqrt(2 * self.c_m**2 * self.__p0(x, y) * (1 - self.__p0(x, y)) * self.tau_c / self.n) / self.tau

    # --- PRIVATE --- #
    def __p0(self, x, y):
        return 1 / (1 + self.a * torch.exp(-1 * (x - y) / self.delta))