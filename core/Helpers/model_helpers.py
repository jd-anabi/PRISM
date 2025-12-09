import torch

def rescale_x(x_nd: torch.Tensor, x_offset: float, x_scale: float) -> torch.Tensor:
    """
    Rescaling the hair-bundle displacement
    :param x_nd: the hair bundle position
    :param x_offset: the hair bundle displacement offset
    :param x_scale: the hair bundle displacement multiplicative scale
    :return: the rescaled hair-bundle displacement
    """
    x = x_scale * x_nd + x_offset
    return x

def irescale_x(x: torch.Tensor, x_offset: float, x_scale: float) -> torch.Tensor:
    """
    Rescaling the hair-bundle displacement from dimensional -> non-dimensional
    :param x: the hair bundle position
    :param x_offset: the hair bundle displacement offset
    :param x_scale: the hair bundle displacement multiplicative scale
    :return: the rescaled hair-bundle displacement
    """
    x_nd = (x - x_offset) / x_scale
    return x_nd

def rescale_t(t_nd: torch.Tensor, t_offset: float, t_scale: float) -> torch.Tensor:
    """
    Rescaling the time array
    :param t_nd: the time array
    :param t_offset: the time offset
    :param t_scale: the time multiplicative scale
    :return:  the rescaled time
    """
    t = t_scale * t_nd + t_offset
    return t

def irescale_f(force: torch.Tensor, scale: float) -> torch.Tensor:
    """
    Rescaling the stimulus force from dimensional -> non-dimensional
    :param force: the stimulus force position
    :param scale: the stimulus force multiplicative scale
    :return: the rescaled stimulus force
    """
    force_nd = scale * force
    return force_nd

'''def irescale_f_params(omegas: torch.Tensor, amp: float, phase: float, offset: float,
                      gamma: float, d: float, k_sp: float, chi_hb: float,
                      k_gs_max: float, s_max: float, s_max_nd: float, chi_a: float, t_0: float) -> tuple[torch.Tensor, float, torch.Tensor, float]:
    """
    Rescale the stimulus force (sinusoidal force)  parameters from dimensional -> non-dimensional
    :param amp: amplitude
    :param omegas: frequencies
    :param phase: phase
    :param offset: offset
    :param gamma: geometric conversion factor
    :param d: distance of gating spring relaxation on channel opening
    :param k_sp: stiffness of stereociliary pivots
    :param chi_hb: non-dimensional parameter for non-dimensional hair bundle displacement
    :param k_gs_max: maximum stiffness of gating spring
    :param s_max: maximum slipping rate
    :param s_max_nd: non-dimensional maximum slipping rate
    :param chi_a: non-dimensional parameter for non-dimensional adaptation motor displacement
    :param t_0: time offset
    :return: the rescaled stimulus force parameters
    """
    alpha = gamma / (chi_hb * k_sp * d)
    t_prime = chi_a * s_max_nd / (k_gs_max * s_max)
    amp_nd = alpha * amp
    offset_nd = alpha * offset
    omega_nd = t_prime * omegas
    phases_nd = phase - t_0 * omegas
    return omega_nd, amp_nd, phases_nd, offset_nd'''