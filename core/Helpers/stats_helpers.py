import torch

import gen_helpers as helpers

def _get_moments(x: torch.Tensor, n: int) -> torch.Tensor:
    """
    Gets the first n moments of an input signal x
    :param x: input signal (shape: batch size x time steps)
    :param n: number of moments to calculate; first moment = raw, second moment = central, third and higher moments = standardized
    :return: 2D tensor of moments from order 1 to n for each batch
    """
    d = 1 if x.shape[0] > 1 else 0
    moments = torch.zeros((x.shape[0], n), dtype=x.dtype, device=x.device)
    if n < 1:
        return torch.tensor(1, dtype=x.dtype, device=x.device)
    mean = torch.mean(x, dim=d)
    moments[:, 0] = mean
    if n > 1:
        var = torch.var(x, dim=d)
        moments[:, 1] = var
    if n > 2:
        for i in range(3, n + 1):
            z_score = (x - mean) / torch.sqrt(moments[:, 1])
            moments[:, i] = torch.mean(torch.pow(z_score, i), dim=d)
    return moments

def _get_acf_at_lags(x: torch.Tensor, n: int) -> torch.Tensor:
    """
    Gets n evenly distributed values of the autocorrelation of an input signal x
    :param x: the input signal (shape: batch size x time steps)
    :param n: number of time lags
    :return: 2D tensor of the autocorrelation of n time lags
    """
    if n > x.shape[1]:
        raise ValueError("n cannot be greater than the length of the time series")
    d = 1 if x.shape[0] > 1 else 0
    xf = torch.fft.rfft(x - torch.mean(x, dim=d, keepdim=True), n=2*x.shape[-1], dim=d)
    acf_at_lags = torch.fft.irfft(torch.abs(xf)**2, dim=d)[:, :x.shape[-1]]
    lag_ids = torch.tensor(helpers.get_even_ids(x.shape[-1], n), dtype=torch.long, device=x.device)
    acf_at_lags = torch.index_select(acf_at_lags, 1, lag_ids)
    acf_at_lags = acf_at_lags / acf_at_lags[:, 0]
    return acf_at_lags

def _get_psd(x: torch.Tensor, dt: float) -> torch.Tensor:
    """
    Gets the PSD of an input signal x
    :param x: the input signal (shape: batch size x time steps)
    :param dt: the time step to compute PSD
    :return: 2D tensor of the PSD
    """
    d = 1 if x.shape[0] > 1 else 0
    xf = torch.fft.rfft(x - torch.mean(x, dim=d, keepdim=True), dim=d)
    psd = xf ** 2 * dt / (xf.shape[-1])
    return psd

def _get_psd_at_lags(x: torch.Tensor, n: int, dt: float) -> torch.Tensor:
    """
    Gets n evenly distributed values of the PSD of an input signal x
    :param x: the input signal (shape: batch size x time steps)
    :param n: number of time lags
    :param dt: time step to compute PSD
    :return: 2D tensor of the PSD of n time lags
    """
    if n > x.shape[1]:
        raise ValueError("n cannot be greater than the length of the time series")
    psd_at_lags = _get_psd(x, dt)
    lag_ids = torch.tensor(helpers.get_even_ids(x.shape[-1], n), dtype=torch.long, device=x.device)
    psd_at_lags = torch.index_select(psd_at_lags, 1, lag_ids)
    return psd_at_lags

def _get_psd_peak_features(x: torch.Tensor, dt: float) -> torch.Tensor:
    """
    Gets the (peak frequency, height) pair of the PSD of an input signal x
    :param x: the input signal (shape: batch size x time steps)
    :param dt: time step to compute PSD
    :return: 1D tensor of the (peak frequency, height) for each batch
    """
    d = 1 if x.shape[0] > 1 else 0
    psd = _get_psd(x, dt)
    freqs = torch.fft.rfftfreq(x.shape[-1], d=dt, dtype=x.dtype, device=x.device)
    max_indices = torch.argmax(psd, dim=d)
    maxes = torch.zeros((x.shape[0], 2), dtype=x.dtype, device=x.device)
    for i in range(x.shape[0]):
        peak_power = psd[i, max_indices[i]]
        peak_freq = freqs[max_indices[i]]
        fwhm_ids = torch.where(psd[i, :] >= peak_power / 2)[0]
        q_factor = peak_freq / (freqs[fwhm_ids[-1]] - freqs[fwhm_ids[0]])
        maxes[i] = torch.tensor([peak_freq, peak_power, q_factor], dtype=x.dtype, device=x.device)
    return maxes

def _get_total_power(x: torch.Tensor, dt: float) -> torch.Tensor:
    """
    Gets the total power of an input signal x
    :param x: the input signal (shape: batch size x time steps)
    :param dt: the time step to compute total power
    :return: 2D tensor of the total power of each batch signal
    """
    d = 1 if x.shape[0] > 1 else 0
    psd = _get_psd(x, dt)
    pwr = torch.sum(psd / (x.shape[-1] * dt), dim=d)
    return pwr