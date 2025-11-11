from typing import Union

import torch
import pybispectra as pyi
import antropy as ap
import numpy as np
from scipy import signal

from ..Helpers import gen_helpers as helpers

def get_summary_statistics(x: torch.Tensor, dt: float) -> torch.Tensor:
    """
    Get the set of summary statistics
    :param x: the input signal (shape: batch size x time steps)
    :param dt: the time step
    :return: summary statistics
    """
    n = 30
    if x.shape[-1] < 100:
        n = 1

    # static stats
    moments = _moments(x, 4)
    pdf_features = _pdf_features(x, n)

    # dynamic stats (time-domain)
    acf_at_lags = _acf_at_lags(x, n)
    zero_crossing_stats = _crossing_stats(x, dt, 0)
    cramer_crossing_stats = _crossing_stats(x, dt, pdf_features[:, 2])
    sample_entropy = _sample_entropy(x)

    # dynamic stats (frequency-domain)
    psd_peak_stats = _psd_peak_features(x, dt)
    binned_psd_pwr = _binned_psd_pwr(x, n, dt)

    # dynamic stats (phase-domain)
    analytic_signal_stats = _analytic_signal_stats(x)
    bicoherence = _mean_bicoherence(x, dt)

    summary_stats = [moments, pdf_features, acf_at_lags, zero_crossing_stats,
                   cramer_crossing_stats, sample_entropy, psd_peak_stats,
                   binned_psd_pwr, analytic_signal_stats, bicoherence]
    return torch.cat(summary_stats, dim=1)

# STATIC STATISTICS
def _moments(x: torch.Tensor, order: int) -> torch.Tensor:
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
        var = torch.var(x, dim=d, keepdim=True)
        moments[:, 1] = var[:, 0]
        if order > 2:
            z_score = (x - torch.mean(x, dim=d, keepdim=True)) / torch.sqrt(var)
            for i in range(3, order + 1):
                moments[:, i - 1] = torch.mean(torch.pow(z_score, i), dim=d)
    return moments

def _pdf_features(x: torch.Tensor, nbins: int) -> torch.Tensor:
    """
    Gets the peak locations, valley depth, and peak ratio of the probability density function of x
    :param x: the input signal (shape: batch size x time steps)
    :param nbins: the number of bins to use
    :return: the peak locations, valley depth, and peak ratio of the probability density function
    """
    pdf_features = torch.zeros((x.shape[0], 4), dtype=x.dtype, device=x.device)
    x_np = x.cpu().detach().numpy()
    for i in range(x.shape[0]):
        counts, bin_edges = np.histogram(x_np[i], bins=nbins, density=True)
        bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2
        peaks, _ = signal.find_peaks(counts, height=0.01)
        if len(peaks) == 2:
            peak_locs = bin_centers[peaks]
            peak_heights = counts[peaks]
            valley_idx = np.argmin(counts[peaks[0]:peaks[1]]) + peaks[0]
            valley_depth = counts[valley_idx]
            peak_ratio = peak_heights[0] / (peak_heights[1] + 1e-7) # offset added for stability
            pdf_features[i, :] = torch.tensor([peak_locs[0], peak_locs[1], valley_depth, peak_ratio])
        elif len(peaks) == 1:
            peak_loc = bin_centers[peaks[0]]
            pdf_features[i, :] = torch.tensor([peak_loc, peak_loc, 0.0, 1.0])
        else:
            pdf_features[i, :] = torch.tensor([0.0, 0.0, 0.0, 1.0])
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

def _crossing_stats(x: torch.Tensor, dt: float, boundary: Union[float, torch.Tensor] = 0) -> torch.Tensor:
    """
    Gets the mean and standard deviation of the crossing times of some boundary of an input signal x
    :param x: the input signal (shape: batch size x time steps)
    :param dt: the time step
    :param boundary: the boundary of the crossings
    :return: the mean and standard deviation of the zero crossing times
    """
    crossing_stats = torch.zeros((x.shape[0], 2), dtype=x.dtype, device=x.device)
    if isinstance(boundary, float):
        x = x - boundary
    else:
        x = x - boundary.unsqueeze(1)
    for i in range(x.shape[0]):
        x_curr_batch = x[i, :]
        crossing_ids = (x_curr_batch[:-1] * x_curr_batch[1:] < 0).nonzero().squeeze(-1)
        if crossing_ids.shape[0] < 2:
            crossing_stats[i, 0] = float('nan')
            crossing_stats[i, 1] = 0
        t = crossing_ids * dt
        t_next = (crossing_ids + 1.0) * dt
        x_abs = torch.abs(x_curr_batch[crossing_ids])
        x_abs_next = torch.abs(x_curr_batch[crossing_ids + 1])
        crossing_time = (t * x_abs_next + t_next * x_abs) / (x_abs + x_abs_next)
        dwell_time = crossing_time[1:] - crossing_time[:-1]
        crossing_stats[i, 0] = torch.mean(dwell_time)
        crossing_stats[i, 1] = torch.std(dwell_time)
    return crossing_stats

def _sample_entropy(x: torch.Tensor, m: int = 2, r: float = None) -> torch.Tensor:
    """
    Gets the sample entropy of an input signal x
    :param x: the input signal (shape: batch size x time steps)
    :return: the sample entropy of an input signal x
    """
    x_np = x.cpu().detach().numpy()
    sampen = []
    for i in range(x_np.shape[0]):
        se = ap.sample_entropy(x_np[i], order=m, tolerance=r) if r is not None else ap.sample_entropy(x_np[i], order=m)
        sampen.append(se)
    return torch.tensor(sampen, dtype=x.dtype, device=x.device).unsqueeze(1)

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
    psd = torch.abs(xf) ** 2 * dt / (xf.shape[-1])
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
    peaks = torch.zeros((x.shape[0], 3), dtype=x.dtype, device=x.device)
    for i in range(x.shape[0]):
        peak_power = psd[i, max_indices[i]]
        peak_freq = freqs[max_indices[i]]
        q_factor = 0.0  # Default value
        try:
            fwhm_ids = torch.where(psd[i, :] >= peak_power / 2)[0]
            if fwhm_ids.shape[0] >= 2:
                bandwidth = freqs[fwhm_ids[-1]] - freqs[fwhm_ids[0]]
                if bandwidth > 1e-6:
                    q_factor = peak_freq / bandwidth
        except Exception:
            pass
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

def _psd_pwr(x: torch.Tensor, dt: float, id_bounds: tuple, norm: bool = False) -> torch.Tensor:
    """
    Gets the total power of an input signal x
    :param x: the input signal (shape: batch size x time steps)
    :param dt: the time step to compute total power
    :param id_bounds: the bounds to integrate the PSD over (in units of array indices)
    :param norm: whether to normalize the PSD
    :return: 2D tensor of the total power of each batch signal
    """
    d = 1 if x.shape[0] > 1 else 0
    psd = _psd(x, dt)
    norm_constant = (id_bounds[-1] - id_bounds[0]) * dt if norm else 1
    pwr = torch.sum(psd[:, id_bounds[0]:id_bounds[-1]] / norm_constant, dim=d)
    return pwr

# DYNAMIC STATISTICS (PHASE DOMAIN)
def _analytic_signal_stats(x: torch.Tensor) -> torch.Tensor:
    """
    Calculates the mean and standard deviation of the amplitude and phase of the analytical signal S(t) = x + i H_x(t)
    :param x: the input signal (shape: batch size x time steps)
    :return: the mean and standard deviation of the amplitude and phase of the analytical signal S(t)
    """
    x_np = x.cpu().detach().numpy()
    xa = signal.hilbert(x_np)
    amps = np.abs(xa)
    phases = np.angle(xa)
    mean_amp = np.mean(amps, axis=-1)
    mean_phase = np.mean(phases, axis=-1)
    std_amp = np.std(amps, axis=-1)
    std_phase = np.std(phases, axis=-1)
    stats = np.stack([mean_amp, mean_phase, std_amp, std_phase], axis=1)
    return torch.tensor(stats, dtype=x.dtype, device=x.device)

def _mean_bicoherence(x: torch.Tensor, dt: float, nperseg: int = 256, step: int = 128, nfft: int = 512) -> torch.Tensor:
    """
    Calculates the bicoherence mean of the input signal x
    :param x: the input signal (shape: batch size x time steps)
    :param dt: the time step
    :return: the bicoherence mean of the input signal x
    """
    x_np = x.cpu().detach().numpy()
    results = []
    for i in range(x.shape[0]):
        row = x_np[i, :]
        freqs, _, coeff = signal.stft(row, fs=(1/dt), nperseg=nperseg, noverlap=step, nfft=nfft)
        coeff = coeff.T
        coeff = np.ascontiguousarray(coeff.reshape(coeff.shape[0], 1, coeff.shape[1]))
        pac = pyi.cfc.PAC(data=coeff, freqs=freqs, sampling_freq=(1/dt))
        pac.compute(indices=((0,),), norm=True)
        vals = pac.results[0]
        mean = np.nanmean(vals)
        results.append(mean)
    return torch.tensor(results, dtype=x.dtype, device=x.device).unsqueeze(1)