import torch

K_B: float = 1.38e-23 # m^2 kg s^-2 K^-1

class NadrowskiModel:
    def __init__(self, lam: torch.Tensor, lam_y: torch.Tensor, tau: torch.Tensor, tau_t: torch.Tensor,
                 k_gs: torch.Tensor, k_sp: torch.Tensor, d: torch.Tensor, gamma: torch.Tensor,
                 c_0: torch.Tensor, n: torch.Tensor, n_a: torch.Tensor, delta_e: torch.Tensor, temp_eff: torch.Tensor,
                 eta_hb: torch.Tensor, eta_a: torch.Tensor, force: torch.Tensor,
                 batch_size: int, device: torch.device = torch.device('cpu'), dtype: torch.dtype = torch.float32):
        # sde model parameters
        self.batch_size = batch_size
        self.device = device
        self.dtype = dtype

        # parameters
        self.lam = lam.to(self.device)
        self.lam_y = lam_y.to(self.device)
        self.tau = tau.to(self.device)
        self.tau_t = tau_t.to(self.device)
        self.k_gs = k_gs.to(self.device)
        self.k_sp = k_sp.to(self.device)
        self.d = d.to(self.device)
        self.gamma = gamma.to(self.device)
        self.c_0 = c_0.to(self.device)
        self.n = n.to(self.device)
        self.n_a = n_a.to(self.device)
        self.delta_e = delta_e.to(self.device)
        self.temp_eff = temp_eff.to(self.device)
        self.eta_hb = eta_hb.to(self.device)
        self.eta_a = eta_a.to(self.device)

        # force parameters
        self.force = force.to(self.device)

        # subsuming parameters
        self.d_swing = self.d / self.gamma
        self.a = torch.exp((self.delta_e + self.k_gs * self.d_swing**2 / (2 * self.n)) / (K_B * self.temp_eff))
        self.delta = self.n * K_B * self.temp_eff / (self.k_gs * self.d)

    def f(self, x, t) -> torch.Tensor:
        dx_hb = self._x_hb_dot(x[:, 0], x[:, 1], x[:, 3], x[:, 4])
        dx_a = self._x_a_dot(x[:, 0], x[:, 1], x[:, 2], x[:, 3], x[:, 4])
        dp_m = self._p_m_dot(x[:, 2], x[:, 4])
        dp_gs = self._p_gs_dot(x[:, 3], x[:, 4])
        dp_t = self.__p_t_dot(x[:, 0], x[:, 1], x[:, 3], x[:, 4])
        dx_hb = dx_hb + self.force[:, t] / self.tau_hb
        dx = torch.stack((dx_hb, dx_a, dp_m, dp_gs, dp_t), dim=1)
        return dx

    def g(self) -> torch.Tensor:
        hb_noise = self._hb_noise()
        a_noise = self._a_noise()
        dsigma = torch.stack((hb_noise, a_noise, torch.zeros_like(hb_noise), torch.zeros_like(hb_noise), torch.zeros(hb_noise)), dim=0)
        dsigma = torch.atleast_2d(torch.transpose(dsigma, -1, 0))
        return torch.diag_embed(dsigma)

    # --- SDEs --- #
    def _x_hb_dot(self, x_hb, x_a, p_gs, p_t) -> torch.Tensor:
        x_gs = self.chi_hb * x_hb - self.chi_a * x_a + self.x_c
        k_gs = 1 - p_gs * self.k_gs_offset
        f_gs = k_gs * (x_gs - p_t)
        return -1 * (f_gs + x_hb) / self.tau_hb

    def _x_a_dot(self, x_hb, x_a, p_m, p_gs, p_t) -> torch.Tensor:
        c = 1 - p_m * self.c_offset
        s = self.s_min + p_m * self.s_offset
        x_gs = self.chi_hb * x_hb - self.chi_a * x_a + self.x_c
        k_gs = 1 - p_gs * self.k_gs_offset
        f_gs = k_gs * (x_gs - p_t)
        return self.s_max * s * (f_gs - x_a) - self.c_max * c

    def _p_m_dot(self, p_m, p_t) -> torch.Tensor:
        return (self.ca2_m * p_t * (1 - p_m) - p_m) / self.tau_m

    def _p_gs_dot(self, p_gs, p_t) -> torch.Tensor:
        return (self.ca2_gs * p_t * (1 - p_gs) - p_gs) / self.tau_gs

    def __p_t_dot(self, x_hb, x_a, p_gs, p_t) -> torch.Tensor:
        k_gs = 1 - p_gs * self.k_gs_offset
        x_gs = self.chi_hb * x_hb - self.chi_a * x_a + self.x_c
        arg = -1 * self.u_gs_max * k_gs * (x_gs - 0.5)
        p_t0 = 1 / (1 + self.E_exp * torch.exp(arg))
        return (p_t0 - p_t) / self.tau_t

    # --- NOISE --- #
    def _hb_noise(self) -> torch.Tensor:
        return self.eta_hb / self.tau_hb

    def _a_noise(self) -> torch.Tensor:
        return -1 * self.s_max * self.s_min * self.eta_a