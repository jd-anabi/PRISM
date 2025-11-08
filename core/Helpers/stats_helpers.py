import torch
import statistics

import gen_helpers as helpers

# STATIC STATISTICS
def _get_moments(x: torch.Tensor, order: int) -> torch.Tensor:
    """
    Gets the first n moments of an input signal x
    :param x: input signal (shape: batch size x time steps)
    :param order: number of moments to calculate; first moment = raw, second moment = central, third and higher moments = standardized
    :return: 2D tensor of moments from order 1 to n for each batch
    """
    d = 1 if x.shape[0] > 1 else 0
    moments = torch.zeros((x.shape[0], order), dtype=x.dtype, device=x.device)
    if order < 1:
        return torch.tensor(1, dtype=x.dtype, device=x.device)
    mean = torch.mean(x, dim=d)
    moments[:, 0] = mean
    if order > 1:
        var = torch.var(x, dim=d)
        moments[:, 1] = var
    if order > 2:
        for i in range(3, order + 1):
            z_score = (x - mean) / torch.sqrt(moments[:, 1])
            moments[:, i] = torch.mean(torch.pow(z_score, i), dim=d)
    return moments

def _pdf_features(x: torch.Tensor, nbins: int) -> torch.Tensor:
    """
    Gets the peak locations, valley depth, and peak ratio of the probability density function of x
    :param x: the input signal (shape: batch size x time steps)
    :param nbins: the number of bins to use
    :return: the peak locations, valley depth, and peak ratio of the probability density function
    """
    pdf_features = torch.zeros((x.shape[0], 4), dtype=x.dtype, device=x.device)
    for i in range(x.shape[0]):
        pdf, bin_edges = torch.histogram(x[i], bins=nbins, density=True)
        peaks = torch.topk(pdf, 2)
        valley = torch.topk(pdf, 1, largest=False)
        pdf_features[i, 0] = peaks[0]
        pdf_features[i, 1] = peaks[1]
        pdf_features[i, 2] = valley[0]
        pdf_features[i, 3] = peaks[0] / peaks[1]
    return pdf_features

# DYNAMIC STATISTICS (TIME-DOMAIN)
def _acf_at_lags(x: torch.Tensor, nlags: int) -> torch.Tensor:
    """
    Gets n evenly distributed values of the autocorrelation of an input signal x
    :param x: the input signal (shape: batch size x time steps)
    :param nlags: number of time lags
    :return: 2D tensor of the autocorrelation of n time lags
    """
    if nlags > x.shape[1]:
        raise ValueError("n cannot be greater than the length of the time series")
    d = 1 if x.shape[0] > 1 else 0
    xf = torch.fft.rfft(x - torch.mean(x, dim=d, keepdim=True), n=2*x.shape[-1], dim=d)
    acf_at_lags = torch.fft.irfft(torch.abs(xf)**2, dim=d)[:, :x.shape[-1]]
    lag_ids = torch.tensor(helpers.get_even_ids(x.shape[-1], nlags), dtype=torch.long, device=x.device)
    acf_at_lags = torch.index_select(acf_at_lags, 1, lag_ids)
    acf_at_lags = acf_at_lags / acf_at_lags[:, 0]
    return acf_at_lags

def _crossing_stats(x: torch.Tensor, dt: float, boundary: float = 0) -> torch.Tensor:
    """
    Gets the mean and standard deviation of the crossing times of some boundary of an input signal x
    :param x: the input signal (shape: batch size x time steps)
    :param dt: the time step
    :param boundary: the boundary of the crossings
    :return: the mean and standard deviation of the zero crossing times
    """
    crossing_stats = torch.zeros((x.shape[0], 2), dtype=x.dtype, device=x.device)
    x = x - boundary
    for i in range(x.shape[0]):
        crossing_times = []
        x_batch = x[i, :]
        crossing_ids = (x_batch[:-1] * x_batch[1:] < 0).nonzero().squeeze(-1)
        if crossing_ids.shape[0] < 2:
            crossing_stats[i, 0] = float('inf')
            crossing_stats[i, 1] = 0
        x_batch = x_batch[crossing_ids]

# DYNAMIC STATISTICS (FREQUENCY DOMAIN)
def _psd(x: torch.Tensor, dt: float) -> torch.Tensor:
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

def _psd_peak_features(x: torch.Tensor, dt: float) -> torch.Tensor:
    """
    Gets the (peak frequency, height, q factor) pair of the PSD of an input signal x
    :param x: the input signal (shape: batch size x time steps)
    :param dt: time step to compute PSD
    :return: 1D tensor of the (peak frequency, height, q factor) for each batch
    """
    d = 1 if x.shape[0] > 1 else 0
    psd = _psd(x, dt)
    freqs = torch.fft.rfftfreq(x.shape[-1], d=dt, dtype=x.dtype, device=x.device)
    max_indices = torch.argmax(psd, dim=d)
    peaks = torch.zeros((x.shape[0], 2), dtype=x.dtype, device=x.device)
    for i in range(x.shape[0]):
        peak_power = psd[i, max_indices[i]]
        peak_freq = freqs[max_indices[i]]
        fwhm_ids = torch.where(psd[i, :] >= peak_power / 2)[0]
        q_factor = peak_freq / (freqs[fwhm_ids[-1]] - freqs[fwhm_ids[0]])
        peaks[i] = torch.tensor([peak_freq, peak_power, q_factor], dtype=x.dtype, device=x.device)
    return peaks

def _binned_psd_pwr(x: torch.Tensor, nbins: int, dt: float) -> torch.Tensor:
    """
    Calculate the PSD of an input signal x and return the power in n evenly distributed bins
    :param x: the input signal (shape: batch size x time steps)
    :param nbins: number of bins to compute power
    :param dt: time step to compute PSD
    :return: 2D tensor of the PSD of n frequency bins
    """
    if nbins > x.shape[1]:
        raise ValueError("n cannot be greater than the length of the time series")
    lag_ids = torch.tensor(helpers.get_even_ids(x.shape[-1], nbins + 1), dtype=torch.long, device=x.device)
    binned_psd = torch.zeros((x.shape[0], nbins), dtype=x.dtype, device=x.device)
    for i in range(nbins):
        binned_psd[:, i] = _psd_pwr(x, dt, (lag_ids[i], lag_ids[i + 1]))
    return binned_psd

def _psd_pwr(x: torch.Tensor, dt: float, id_bounds: tuple) -> torch.Tensor:
    """
    Gets the total power of an input signal x
    :param x: the input signal (shape: batch size x time steps)
    :param dt: the time step to compute total power
    :param id_bounds: the bounds to integrate the PSD over (in units of array indices)
    :return: 2D tensor of the total power of each batch signal
    """
    d = 1 if x.shape[0] > 1 else 0
    psd = _psd(x, dt)
    n = id_bounds[-1] - id_bounds[0]
    pwr = torch.sum(psd[:, id_bounds[0]:id_bounds[-1]] / (n * dt), dim=d)
    return pwr