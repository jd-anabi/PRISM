from typing import Union

from scipy.linalg import bandwidth
from tqdm import tqdm

import torch
import pybispectra as pyi
import antropy as ap
import numpy as np
from scipy import signal

from ..Helpers import helpers

class SummaryStatistics:
    def __init__(self, x: torch.Tensor, dt: float):
        self.x = x
        self.dt = dt
        self.fs = 1 / dt
        self.batch_size = x.shape[0]
        self.n = x.shape[1]

    # ----------------------- PRIVATE FUNCTIONS -----------------------
    def __compute_stat_dist_features(self) -> torch.Tensor:
        """
        Computes the following statistical distribution features:
            (1) Mean
            (2) Variance
            (3) Skewness
            (4) Kurtosis
            (5) Bimodality coefficient
            (6-10) Quantiles (5%, 25%, 50%, 75%, 100%)
        :return: statistical distribution features with shape: (batch_size, 10)
        """
        # first four moments
        mean = torch.mean(self.x, dim=-1, keepdim=True)
        var = torch.var(self.x, dim=-1, keepdim=True)
        std = torch.sqrt(var)
        z_score = (self.x - mean) / std
        skew = torch.mean(torch.pow(z_score, 3), dim=-1, keepdim=True)
        kurt = torch.mean(torch.pow(z_score, 4), dim=-1, keepdim=True)

    def __compute_spectral_stats(self, n_bands: int) -> torch.Tensor:
        """
        Computes the following spectral statistics:
            (1) Peak frequency
            (2) Quality factor
            (3) Spectral centroid
            (4) Spectral bandwidth
            (5) Spectral entropy
            (6) Power in n_bands frequency bands
        :param n_bands: the number of frequency bands
        :return: the spectral statistics with shape: (batch_size, 5 + k)
        """
        # spectral features
        stats = []

        # preliminary fft calculations
        xf = torch.fft.rfft(self.x - torch.mean(self.x, dim=-1, keepdim=True), dim=-1)
        freqs = torch.fft.rfftfreq(self.n, d=self.dt, device=xf.device)
        psd = torch.abs(xf) ** 2 * self.dt / (xf.shape[-1])

        # peak frequency
        peak_pwr, peak_idx = torch.max(psd, dim=-1)
        peak_freqs = freqs[peak_idx]
        stats.append(peak_freqs.unsqueeze(-1))

        # quality factor
        q_factor = torch.zeros(self.batch_size, device=xf.device, dtype=xf.dtype)
        half_max_pwr = peak_pwr / 2
        half_max_mask = psd > half_max_pwr.unsqueeze(-1)
        for batch_id in range(self.batch_size):
            above_half_max_idx = torch.where(half_max_mask[batch_id])[0]
            width = (above_half_max_idx[-1] - above_half_max_idx[0]) * self.dt
            q_factor[batch_id] = peak_freqs[batch_id] / width if width > 0 else 100.0
        stats.append(q_factor.unsqueeze(-1))

        # spectral centroid
        spectral_centroid = torch.sum(freqs.unsqueeze(0) * psd, dim=-1) / torch.sum(psd, dim=-1)
        stats.append(spectral_centroid.unsqueeze(-1))

        # spectral bandwidth
        freq_diff = freqs.unsqueeze(0) - spectral_centroid.unsqueeze(1) # shape: (batch_size, n) where element (i, j) corresponds to f_j - c_i
        spectral_bandwidth = torch.sqrt(torch.sum(freq_diff ** 2 * psd, dim=-1) / torch.sum(psd, dim=-1))
        stats.append(spectral_bandwidth.unsqueeze(-1))

        # spectral entropy
        psd_sum = torch.sum(psd, dim=-1, keepdim=True)
        psd_prob_density = psd / torch.clamp(psd_sum, min=1e-9)
        spectral_entropy = -1 * torch.sum(psd_prob_density * torch.log(torch.clamp(psd_prob_density, 1e-9)), dim=-1)
        stats.append(spectral_entropy.unsqueeze(-1))

        # power in k frequency bands
        binned_pwr = torch.zeros((self.batch_size, n_bands), device=xf.device, dtype=xf.dtype)
        segs = helpers.get_even_ids(self.n, n_bands + 1)
        for i in range(n_bands):
            start, end = segs[i], segs[i + 1]
            psd_band = psd[:, start:end]
            binned_pwr[:, i] = torch.sum(psd_band, dim=-1) * (freqs[end] - freqs[start])
        stats.append(binned_pwr)

        return torch.cat(stats, dim=-1)