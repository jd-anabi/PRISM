from typing import Union
from tqdm import tqdm

import torch
import pybispectra as pyi
import antropy as ap
import numpy as np
from scipy import signal

from ..Helpers import helpers

def get_summary_statistics(x: torch.Tensor, dt: float, n: int = 1) -> torch.Tensor:
    """
    Get the set of summary statistics
    :param x: the input signal (shape: batch size x time steps)
    :param dt: the time step
    :param n: hyperparameter that controls how many lags/bins for acf and psd statistics
    :return: summary statistics
    """
    progress_bar = tqdm(total=10, desc="Getting summary statistics")

    # static stats
    moments = _moments(x, 4)
    progress_bar.update()
    pdf_features = _pdf_features(x, n)
    progress_bar.update()

    # dynamic stats (time-domain)
    acf_at_lags = _acf_at_lags(x, n)
    progress_bar.update()
    zero_crossing_stats = _crossing_stats(x, dt, 0)
    progress_bar.update()
    cramer_crossing_stats = _crossing_stats(x, dt, pdf_features[:, 2])
    progress_bar.update()
    sample_entropy = _sample_entropy(x)
    progress_bar.update()

    # dynamic stats (frequency-domain)
    psd_peak_stats = _psd_peak_features(x, dt)
    progress_bar.update()
    binned_psd_pwr = _binned_psd_pwr(x, n, dt)
    progress_bar.update()

    # dynamic stats (phase-domain)
    analytic_signal_stats = _analytic_signal_stats(x)
    progress_bar.update()
    bicoherence = _mean_bicoherence(x, dt)
    progress_bar.update()

    summary_stats = [moments, pdf_features, acf_at_lags, zero_crossing_stats,
                   cramer_crossing_stats, sample_entropy, psd_peak_stats,
                   binned_psd_pwr, analytic_signal_stats, bicoherence]
    progress_bar.close()
    return torch.cat(summary_stats, dim=1)

# STATIC STATISTICS
def _moments(x: torch.Tensor, order: int) -> torch.Tensor:
    """
    Gets the first n moments of an input signal x
    :param x: input signal (shape: batch size x time steps)
    :param order: number of moments to calculate; first moment = raw, second moment = central, third and higher moments = standardized
    :return: 2D tensor of moments from order 1 to n for each batch
    """
    d = 1
    var = 0
    moments = torch.zeros((x.shape[0], order), dtype=x.dtype, device=x.device)
    if order < 1:
        return torch.tensor(1, dtype=x.dtype, device=x.device)
    mean = torch.mean(x, dim=d, keepdim=True)
    moments[:, 0] = mean.squeeze(1)
    if order > 1:
        var = torch.var(x, dim=d, keepdim=True)
        moments[:, 1] = var.squeeze(1)
    if order > 2:
        z_score = (x - mean) / (torch.sqrt(var) + 1e-6)
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
        # robustness check
        if not np.all(np.isfinite(x_np[i])):
            pdf_features[i, :] = torch.tensor([float('nan')] * 4, dtype=x.dtype, device=x.device)
            continue
        counts, bin_edges = np.histogram(x_np[i], bins=nbins, density=True)
        bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2
        peaks, _ = signal.find_peaks(counts, height=0.01)
        if len(peaks) == 2:
            peak_locs = bin_centers[peaks]
            peak_heights = counts[peaks]
            valley_idx = np.argmin(counts[peaks[0]:peaks[1]]) + peaks[0]
            valley_depth = counts[valley_idx]
            peak_ratio = peak_heights[0] / (peak_heights[1] + 1e-6) # offset added for stability
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
    d = 1
    xf = torch.fft.rfft(x - torch.mean(x, dim=d, keepdim=True), n=2*x.shape[-1], dim=d)
    acf = torch.fft.irfft(torch.abs(xf)**2, dim=d)[:, :x.shape[-1]]
    lag_ids = torch.tensor(helpers.get_even_ids(x.shape[-1], nlags), dtype=torch.long, device=x.device)
    acf_at_lags = torch.index_select(acf, 1, lag_ids)
    acf_at_lags = acf_at_lags / acf_at_lags[:, 0].unsqueeze(1)
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
    if isinstance(boundary, float) or isinstance(boundary, int):
        x = x - boundary
    else:
        x = x - boundary.unsqueeze(1)
    for i in range(x.shape[0]):
        if not torch.all(torch.isfinite(x[i])):
            crossing_stats[i, 0] = float('nan')
            crossing_stats[i, 1] = float('nan')
            continue
        x_curr_batch = x[i, :]
        crossing_ids = (x_curr_batch[:-1] * x_curr_batch[1:] < 0).nonzero().squeeze(-1)
        if crossing_ids.shape[0] < 2:
            crossing_stats[i, 0] = float('nan')
            crossing_stats[i, 1] = float('nan')
            continue
        t = crossing_ids * dt
        t_next = (crossing_ids + 1.0) * dt
        x_abs = torch.abs(x_curr_batch[crossing_ids])
        x_abs_next = torch.abs(x_curr_batch[crossing_ids + 1])
        crossing_time = (t * x_abs_next + t_next * x_abs) / (x_abs + x_abs_next + 1e-6)
        dwell_time = crossing_time[1:] - crossing_time[:-1]
        if dwell_time.shape[0] == 0:
            crossing_stats[i, 0] = float('nan')
            crossing_stats[i, 1] = float('nan')
        elif dwell_time.shape[0] == 1:
            crossing_stats[i, 0] = dwell_time[0]
            crossing_stats[i, 1] = 0
        else:
            crossing_stats[i, 0] = torch.mean(dwell_time)
            crossing_stats[i, 1] = torch.std(dwell_time)
    return crossing_stats

def _sample_entropy(x: torch.Tensor, m: int = 2, r: float = None, downsample: int = 100) -> torch.Tensor:
    """
    Gets the sample entropy of an input signal x
    :param x: the input signal (shape: batch size x time steps)
    :param m: the order
    :param r: the tolerance
    :param downsample: the downsampling scale
    :return: the sample entropy of an input signal x
    """
    x_np = x.cpu().detach().numpy()
    x_np = np.ascontiguousarray(x_np[:, ::downsample])
    sampen = []
    for i in range(x_np.shape[0]):
        if not np.all(np.isfinite(x_np[i])):
            sampen.append(float('nan'))
            continue
        se = ap.sample_entropy(x_np[i], order=m, tolerance=r)
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
    d = 1
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
    d = 1
    psd = _psd(x, dt)
    freqs = torch.fft.rfftfreq(x.shape[-1], d=dt, dtype=x.dtype, device=x.device)
    max_indices = torch.argmax(psd, dim=d)
    peaks = torch.zeros((x.shape[0], 3), dtype=x.dtype, device=x.device)
    for i in range(x.shape[0]):
        if not torch.all(torch.isfinite(x[i])):
            peaks[i] = torch.tensor([float('nan')] * 3, dtype=x.dtype, device=x.device)
            continue
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

def _binned_psd_pwr(x: torch.Tensor, nbins: int, dt: float, norm: bool = False) -> torch.Tensor:
    """
    Calculate the PSD of an input signal x and return the power in n evenly distributed bins
    :param x: the input signal (shape: batch size x time steps)
    :param nbins: number of bins to compute power
    :param dt: time step to compute PSD
    :param norm: if True, normalize the PSD
    :return: 2D tensor of the PSD of n frequency bins
    """
    if nbins > x.shape[1]:
        raise ValueError("n cannot be greater than the length of the time series")
    lag_ids = torch.tensor(helpers.get_even_ids(x.shape[-1], nbins + 1), dtype=torch.long, device=x.device)
    binned_psd_pwr = torch.zeros((x.shape[0], nbins), dtype=x.dtype, device=x.device)
    psd = _psd(x, dt)
    norms_for_bins = [1] * nbins
    if norm:
        norms_for_bins = [(lag_ids[i + 1] - lag_ids[i]) * dt for i in range(nbins)]
    for i in range(nbins):
        start, end = lag_ids[i], lag_ids[i + 1]
        psd_band = psd[:, start:end]
        binned_psd_pwr[:, i] = torch.sum(psd_band / norms_for_bins[i], dim=-1)
    return binned_psd_pwr

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
        if not np.all(np.isfinite(x_np[i])):
            results.append(float('nan'))
            continue
        row = x_np[i, :]
        freqs, _, coeff = signal.stft(row, fs=(1/dt), nperseg=nperseg, noverlap=step, nfft=nfft)
        coeff = coeff.T
        coeff = coeff[:, np.newaxis, :]
        coeff = np.ascontiguousarray(coeff)
        try:
            bispectrum = pyi.Bispectrum(data=coeff, freqs=freqs, sampling_freq=(1/dt))
            bispectrum.compute(indices=((0,), (0,)))
            data = bispectrum.results.get_results()
            abs_data0 = np.abs(data[0])
            results.append(np.mean(abs_data0))
        except Exception:
            results.append(float('nan'))
    return torch.tensor(results, dtype=x.dtype, device=x.device).unsqueeze(1)