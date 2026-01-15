from typing import Union

import torch

from core.Models import model

K_B: float = 1.38e-23 # m^2 kg s^-2 K^-1
Q: float = 1.6e-19 # C

class BPModelSteady(model.BPModel):
    def __init__(self, lam_x: torch.Tensor, lam_y: torch.Tensor, lam_sf: torch.Tensor, k_sf: torch.Tensor, k_sp: torch.Tensor,
                 k_gs_min: torch.Tensor, k_gs_max: torch.Tensor, k_es: torch.Tensor, x_sf: torch.Tensor, x_es: torch.Tensor,
                 x_sp: torch.Tensor, x_c: torch.Tensor, d: torch.Tensor, n: torch.Tensor, gamma: torch.Tensor,
                 c_min: torch.Tensor, s_min: torch.Tensor, c_max: torch.Tensor, s_max: torch.Tensor,
                 k_m_plus: torch.Tensor, k_r_plus: torch.Tensor, k_m_minus: torch.Tensor, k_r_minus: torch.Tensor,
                 trans_perm: torch.Tensor, ca2_x_in: torch.Tensor, ca2_x_ex: torch.Tensor, v_m: torch.Tensor,
                 ref_pot: torch.Tensor, diff_const: Union[torch.Tensor, torch.float32], valence: Union[torch.Tensor, torch.float32],
                 r_m: torch.Tensor, r_r: torch.Tensor, delta_e: torch.Tensor, temp: torch.Tensor, epsilon: torch.Tensor,
                 force: torch.Tensor, batch_size: int, device: torch.device = torch.device('cpu'), dtype: torch.dtype = torch.float32):
        super().__init__(lam_x, lam_y, lam_sf, k_sf, k_sp, k_gs_min, k_gs_max, k_es, x_sf, x_es,
                         x_sp, x_c, d, n, gamma, c_min, s_min, c_max, s_max, k_m_plus, k_r_plus, k_m_minus, k_r_minus,
                         trans_perm, ca2_x_in, ca2_x_ex, v_m, ref_pot, diff_const, valence, r_m, r_r, delta_e,
                         torch.zeros_like(lam_x, dtype=dtype, device=device), temp, epsilon,
                         force, batch_size, device, dtype)

    def f(self, x, t) -> torch.Tensor:
        p_t_steady = self.__p_t_steady(x[:, 0], x[:, 1], x[:, 3])
        dx = self._x_dot(x[:, 0], x[:, 1], x[:, 3], p_t_steady)
        dy = self._y_dot(x[:, 0], x[:, 1], x[:, 2], x[:, 3], p_t_steady)
        dp_m = self._p_m_dot(x[:, 2])
        dp_r = self._p_r_dot(x[:, 3])
        dx = dx + self.force[:, t] / self.lam
        dx = torch.stack((dx, dy, dp_m, dp_r), dim=1)
        return dx

    def g(self) -> torch.Tensor:
        x_noise = self._x_noise()
        y_noise = self._y_noise()
        dsigma = torch.stack((x_noise, y_noise, torch.zeros_like(x_noise), torch.zeros_like(x_noise)), dim=0)
        dsigma = torch.atleast_2d(torch.transpose(dsigma, -1, 0))
        return torch.diag_embed(dsigma)

    def __p_t_steady(self, x, y, p_r) -> torch.Tensor:
        k_gs = self.k_gs_min + self.delta_k_gs * p_r
        delta_x_t = K_B * self.temp / (k_gs * self.d)
        channel_disp = self.gamma * x - y + self.x_c - self.d / 2
        p_t_steady = 1 / (1 + self.a * torch.exp(-1 * channel_disp / delta_x_t))
        return p_t_steady