import torch

from core.Models import bp_model

class BpModelSteady(bp_model.BpModel):
    def __init__(self, tau_hb: torch.Tensor, tau_m: torch.Tensor, tau_gs: torch.Tensor,
                 c_min: torch.Tensor, s_min: torch.Tensor, s_max: torch.Tensor, ca2_m: torch.Tensor,
                 ca2_gs: torch.Tensor, u_gs_max: torch.Tensor, delta_e: torch.Tensor, k_gs_ratio: torch.Tensor,
                 chi_hb: torch.Tensor, chi_a: torch.Tensor, x_c: torch.Tensor, eta_hb: torch.Tensor,
                 eta_a: torch.Tensor, force: torch.Tensor, batch_size: int, device: torch.device = torch.device('cpu'),
                 dtype: torch.dtype = torch.float32):
        super().__init__(tau_hb, tau_m, tau_gs, torch.zeros(batch_size),
                         c_min, s_min, s_max, ca2_m, ca2_gs, u_gs_max,
                         delta_e, k_gs_ratio, chi_hb, chi_a, x_c,
                         eta_hb, eta_a, force, batch_size, device, dtype)

    def f(self, x, t: int = 0) -> torch.Tensor:
        p_t0 = self.__p_t0(x[:, 0], x[:, 1], x[:, 3])
        dx_hb = self._x_hb_dot(x[:, 0], x[:, 1], x[:, 3], p_t0)
        dx_a = self._x_a_dot(x[:, 0], x[:, 1], x[:, 2], x[:, 3], p_t0)
        dp_m = self._p_m_dot(x[:, 2], p_t0)
        dp_gs = self._p_gs_dot(x[:, 3], p_t0)
        dx_hb = dx_hb + self.force[:, t] / self.tau_hb
        dx = torch.stack((dx_hb, dx_a, dp_m, dp_gs), dim=1)
        return dx

    def g(self) -> torch.Tensor:
        hb_noise = self._hb_noise()
        a_noise = self._a_noise()
        dsigma = torch.stack((hb_noise, a_noise, torch.zeros_like(hb_noise), torch.zeros_like(hb_noise)), dim=0)
        dsigma = torch.atleast_2d(torch.transpose(dsigma, -1, 0))
        return torch.diag_embed(dsigma)

    # --- STEADY-STATE OPEN-CHANNEL PROBABILITY --- #
    def __p_t0(self, x_hb, x_a, p_gs) -> torch.Tensor:
        k_gs = 1 - p_gs * self.k_gs_offset
        x_gs = self.chi_hb * x_hb - self.chi_a * x_a + self.x_c
        arg = -1 * self.u_gs_max * k_gs * (x_gs - 0.5)
        return 1 / (1 + self.E_exp * torch.exp(arg))