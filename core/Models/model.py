from typing import Union

import torch

K_B: float = 1.38e-23 # m^2 kg s^-2 K^-1
Q: float = 1.6e-19 # C

class BPModel:
    def __init__(self, lam_x: torch.Tensor, lam_y: torch.Tensor, lam_sf: torch.Tensor, k_sf: torch.Tensor, k_sp: torch.Tensor,
                 k_gs_min: torch.Tensor, k_gs_max: torch.Tensor, k_es: torch.Tensor, x_sf: torch.Tensor, x_es: torch.Tensor,
                 x_sp: torch.Tensor, x_c: torch.Tensor, d: torch.Tensor, n: torch.Tensor, gamma: torch.Tensor,
                 c_min: torch.Tensor, s_min: torch.Tensor, c_max: torch.Tensor, s_max: torch.Tensor,
                 k_m_plus: torch.Tensor, k_r_plus: torch.Tensor, k_m_minus: torch.Tensor, k_r_minus: torch.Tensor,
                 trans_perm: torch.Tensor, ca2_x_in: torch.Tensor, ca2_x_ex: torch.Tensor, v_m: torch.Tensor,
                 ref_pot: torch.Tensor, diff_const: Union[torch.Tensor, torch.float32], valence: Union[torch.Tensor, torch.float32],
                 r_m: torch.Tensor, r_r: torch.Tensor, delta_e: torch.Tensor, tau_0: torch.Tensor, temp: torch.Tensor, epsilon: torch.Tensor,
                 force: torch.Tensor, batch_size: int, device: torch.device = torch.device('cpu'), dtype: torch.dtype = torch.float32):
        # sde model parameters
        self.batch_size = batch_size
        self.device = device
        self.dtype = dtype

        # parameters
        self.lam_x = lam_x.to(self.device)
        self.lam_y = lam_y.to(self.device)
        self.lam_sf = lam_sf.to(self.device)
        self.k_sf = k_sf.to(self.device)
        self.k_sp = k_sp.to(self.device)
        self.k_gs_min = k_gs_min.to(self.device)
        self.k_gs_max = k_gs_max.to(self.device)
        self.k_es = k_es.to(self.device)
        self.x_sf = x_sf.to(self.device)
        self.x_es = x_es.to(self.device)
        self.x_sp = x_sp.to(self.device)
        self.x_c = x_c.to(self.device)
        self.d = d.to(self.device)
        self.n = n.to(self.device)
        self.gamma = gamma.to(self.device)
        self.c_min = c_min.to(self.device)
        self.s_min = s_min.to(self.device)
        self.c_max = c_max.to(self.device)
        self.s_max = s_max.to(self.device)
        self.k_m_plus = k_m_plus.to(self.device)
        self.k_r_plus = k_r_plus.to(self.device)
        self.k_m_minus = k_m_minus.to(self.device)
        self.k_r_minus = k_r_minus.to(self.device)
        self.trans_perm = trans_perm.to(self.device)
        self.ca2_x_in = ca2_x_in.to(self.device)
        self.ca2_x_ex = ca2_x_ex.to(self.device)
        self.v_m = v_m.to(self.device)
        self.ref_pot = ref_pot.to(self.device)
        self.diff_const = diff_const.to(self.device)
        self.valence = valence.to(self.device)
        self.r_m = r_m.to(self.device)
        self.r_r = r_r.to(self.device)
        self.delta_e = delta_e.to(self.device)
        self.tau_0 = tau_0.to(self.device)
        self.temp = temp.to(self.device)
        self.epsilon = epsilon.to(self.device)

        # force parameters
        self.force = force.to(self.device)

        # subsuming parameters
        self.delta_c = self.c_max - self.c_min
        self.delta_s = self.s_max - self.s_min
        self.delta_k_gs = self.k_gs_max - self.k_gs_min
        self.a = torch.exp(self.delta_e / (K_B * self.temp))
        self.lam = self.lam_x + self.lam_sf

        # transduction channel parameters
        trans_exp = torch.exp(-1 * (Q / K_B) * (self.valence * self.v_m / self.temp))
        trans_cond = (Q**2 / K_B) * (self.trans_perm * self.valence**2 / self.temp) * ((self.ca2_x_in - self.ca2_x_ex * trans_exp) / (1 - trans_exp)) # conduction
        trans_current = trans_cond * (self.v_m - self.ref_pot) # current - from Ohm's Law

        # Ca2+ concentrations
        ca2 = -1 * trans_current / (2 * torch.pi * Q * self.valence * self.diff_const)
        self.ca2_m = ca2 / self.r_m
        self.ca2_r = ca2 / self.r_r

    def f(self, x, t) -> torch.Tensor:
        dx = self._x_dot(x[:, 0], x[:, 1], x[:, 3], x[:, 4])
        dy = self._y_dot(x[:, 0], x[:, 1], x[:, 2], x[:, 3], x[:, 4])
        dp_m = self._p_m_dot(x[:, 2])
        dp_r = self._p_r_dot(x[:, 3])
        dp_t = self._p_t_dot(x[:, 0], x[:, 1], x[:, 3], x[:, 4])
        dx = dx + self.force[:, t] / self.lam
        dx = torch.stack((dx, dy, dp_m, dp_r, dp_t), dim=1)
        return dx

    def g(self) -> torch.Tensor:
        x_noise = self._x_noise()
        y_noise = self._y_noise()
        dsigma = torch.stack((x_noise, y_noise, torch.zeros_like(x_noise), torch.zeros_like(x_noise), torch.zeros_like(x_noise)), dim=0)
        dsigma = torch.atleast_2d(torch.transpose(dsigma, -1, 0))
        return torch.diag_embed(dsigma)

    # --- SDEs --- #
    def _x_dot(self, x, y, p_r, p_t) -> torch.Tensor:
        stimulus = self.k_sf * (self.x_sf - x)
        pivot = self.k_sp * (x - self.x_sp)
        k_gs = self.k_gs_min + self.delta_k_gs * p_r
        f_gs = self.n * self.gamma * k_gs * (self.gamma * x - y + self.x_c - p_t * self.d)
        return -1 * (stimulus + pivot + f_gs) / self.lam

    def _y_dot(self, x, y, p_m, p_r, p_t) -> torch.Tensor:
        c = self.c_min + (1 - p_m) * self.delta_c
        s = self.s_min + p_m * self.delta_s
        k_gs = self.k_gs_min + self.delta_k_gs * p_r
        f_gs = k_gs * (self.gamma * x - y + self.x_c - p_t * self.d)
        f_es = self.k_es * (y - self.x_es)
        return -1 * (c - s * (f_gs - f_es))

    def _p_m_dot(self, p_m) -> torch.Tensor:
        return self.k_m_plus * self.ca2_m * (1 - p_m) - self.k_m_minus * p_m

    def _p_r_dot(self, p_r) -> torch.Tensor:
        return self.k_r_plus * self.ca2_r * (1 - p_r) - self.k_r_minus * p_r

    def _p_t_dot(self, x, y, p_r, p_t) -> torch.Tensor:
        k_gs = self.k_gs_min + self.delta_k_gs * p_r
        delta_x_t = K_B * self.temp / (k_gs * self.d)
        channel_disp = self.gamma * x - y + self.x_c - self.d / 2
        denom = torch.cosh(channel_disp / (2 * delta_x_t) + torch.log(self.a) / 2)
        tau_t = self.tau_0 / denom
        p_t_steady = 1 / (1 + self.a * torch.exp(-1 * channel_disp / delta_x_t))
        return (p_t_steady - p_t) / tau_t

    # --- NOISE --- #
    def _x_noise(self) -> torch.Tensor:
        return self.epsilon * torch.sqrt(2 * K_B * self.temp * self.lam_x) / self.lam

    def _y_noise(self) -> torch.Tensor:
        return self.epsilon * torch.sqrt(2 * K_B * self.temp / self.lam_y)