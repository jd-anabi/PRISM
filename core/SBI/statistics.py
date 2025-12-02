from typing import Union
from tqdm import tqdm
import torch
import numpy as np
from scipy import signal, stats
import antropy as ap
from pybispectra.general import Bispectrum
from sklearn.feature_selection import mutual_info_regression
from statsmodels.tsa.stattools import pacf


from ..Helpers import helpers

class SummaryStatistics:
    def __init__(self, x: torch.Tensor, dt: float):
        self.x = x
        self.dt = dt
        self.fs = 1 / dt
        self.batch_size = x.shape[0]
        self.n = x.shape[1]
        self.device = x.device
        self.dtype = x.dtype

    # ----------------------- PRIVATE FUNCTIONS -----------------------
    def __compute_stat_dist_features(self) -> torch.Tensor:
        """
        Computes the following statistical distribution features:
            (1) Mean
            (2) Variance
            (3) Skewness
            (4) Kurtosis
            (5) Bimodality coefficient
            (6-10) Quantiles (5%, 25%, 50%, 75%, 95%)
            (11) Median
            (12) Median Absolute Deviation (MAD)
        :return: statistical distribution features with shape: (batch_size, 10)
        """
        # statistical distribution features
        stats = []

        # first four moments
        mean = torch.mean(self.x, dim=-1, keepdim=True)
        var = torch.var(self.x, dim=-1, keepdim=True)
        stats.append(mean)
        stats.append(var)

        std = torch.sqrt(var)
        z_score = (self.x - mean) / std
        skew = torch.mean(torch.pow(z_score, 3), dim=-1, keepdim=True)
        kurt = torch.mean(torch.pow(z_score, 4), dim=-1, keepdim=True)
        stats.append(skew)
        stats.append(kurt)

        # bimodality
        bimodality = (torch.pow(skew, 2) + 1) / (kurt + 1)
        stats.append(bimodality)

        # quantiles
        q_vals = torch.tensor([0.05, 0.25, 0.5, 0.75, 0.95], device=self.device, dtype=self.dtype)
        quantiles = torch.quantile(self.x, q_vals, dim=-1) # shape: (n_q_vals, batch_size)
        quantiles = torch.transpose(quantiles, 0, -1)
        stats.append(quantiles)

        # median and MAD
        median = quantiles[:, 2].unsqueeze(-1) # 50% quantile
        mad = torch.median(torch.abs(self.x - median), dim=-1, keepdim=True).values
        stats.append(median)
        stats.append(mean)

        return torch.cat(stats, dim=-1)

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
        :return: the spectral statistics with shape: (batch_size, 5 + n_bands)
        """
        # spectral features
        stats = []

        # preliminary fft calculations
        xf = torch.fft.rfft(self.x - torch.mean(self.x, dim=-1, keepdim=True), dim=-1)
        freqs = torch.fft.rfftfreq(self.n, d=self.dt, device=self.device)
        psd = torch.abs(xf) ** 2 * self.dt / self.n

        # peak frequency
        peak_pwr, peak_idx = torch.max(psd, dim=-1)
        peak_freqs = freqs[peak_idx]
        stats.append(peak_freqs.unsqueeze(-1))

        # quality factor
        q_factor = torch.zeros(self.batch_size, device=self.device, dtype=self.dtype)
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
        binned_pwr = torch.zeros((self.batch_size, n_bands), device=self.device, dtype=self.dtype)
        segs = helpers.get_even_ids(self.n, n_bands + 1)
        for i in range(n_bands):
            start, end = segs[i], segs[i + 1]
            psd_band = psd[:, start:end]
            binned_pwr[:, i] = torch.sum(psd_band, dim=-1) * (freqs[end] - freqs[start])
        stats.append(binned_pwr)

        return torch.cat(stats, dim=-1)

    def __compute_temporal_stats(self, n_lags: int, pacf_lags: int, downsample_factor: int = 1000) -> torch.Tensor:
        """
        Computes the following temporal statistics:
            (1) Autocorrelation at n_lags different lags
            (2) Decorrelation Rate
            (3) Partial Autocorrelation (PACF)
            (4) Mutual Information
        :param n_lags: the number of lags for the autocorrelation
        :param pacf_lags: the number of windows for the PACF
        :param downsample_factor: the downsampling factor
        :return: the temporal statistics with shape: (batch_size, 3 + n_lags)
        """
        # temporal features
        stats = []

        if n_lags > self.x.shape[-1] or n_lags <= 1:
            raise ValueError("Number of lags cannot be greater than the length of the time series and must be greater than 1")
        if pacf_lags > self.x.shape[-1] or pacf_lags <= 1:
            raise ValueError("Number of PACF lags cannot be greater than the length of the time series and must be greater than 1")

        # lagged acf
        xf = torch.fft.rfft(self.x - torch.mean(self.x, dim=-1, keepdim=True), n=2*self.n, dim=-1, keepdim=True)
        psd = torch.abs(xf) ** 2 * self.dt / self.n
        acf = torch.fft.irfft(psd, n=2*self.n, dim=-1)[:, :self.n]
        acf = acf / torch.clamp(acf[:, 0].unsqueeze(-1), min=1e-9)
        lag_idx = torch.tensor(helpers.get_even_ids(self.n, n_lags), device=self.device, dtype=torch.long)
        acf_lagged = torch.index_select(acf, dim=1, index=lag_idx)
        stats.append(acf_lagged)

        # decorrelation rate
        negative_mask = acf < 0 # boolean mask for negative values
        negative_mask_int = negative_mask.int() # convert to int mask; < 0 -> 1 and >= 0 -> 0
        first_negative_idx = torch.argmax(negative_mask_int, dim=-1) # find first negative crossing for each batch
        has_negative = negative_mask_int.max(dim=-1).values # want to identify rows that had no negative values
        first_negative_idx[has_negative == 0] = -1 # set index to -1 for rows with no negative crossings
        decorrelation_time = first_negative_idx * self.dt
        decorrelation_time = torch.clamp(decorrelation_time, min=0) # if any values are negative (i.e. no decorrelation) then set the value to 0
        stats.append(decorrelation_time.unsqueeze(-1))

        # PACF and mutual information (move to CPU since we have to implement a for loop)
        x_to_cpu = self.x.detach().cpu().numpy()
        step = max(1, self.n // downsample_factor)
        x_downsampled = x_to_cpu[:, ::step] # need this for pacf and mutual info calculation since they are computationally expensive (i.e. O(n^2))
        x_downsampled = np.ascontiguousarray(x_downsampled)

        pacf_stats = []
        mi_stats = []
        for i in range(self.batch_size):
            row = x_downsampled[i]
            if not np.all(np.isfinite(row)):
                pacf_stats.append(np.zeros(pacf_lags))
                mi_stats.append(0.0)
                continue

            # PACF calculation
            try:
                pacf_vals = pacf(row, nlags=pacf_lags, method='yw')[1:pacf_lags+1] # ignore the zeroth element
                # check if pacf_vals is too short and pad if that is the case
                if pacf_vals.shape[0] < pacf_lags:
                    pacf_vals = np.pad(pacf_vals, (0, pacf_lags - pacf_vals.shape[0]))
                    pacf_stats.append(pacf_vals)
            except Exception:
                pacf_stats.append(np.zeros(pacf_lags))

            # mi calculation
            try:
                mi_vals = mutual_info_regression(row[:-1].reshape(-1, 1), row[1:], discrete_features=False)
                mi_stats.append(mi_vals)
            except Exception:
                mi_stats.append(0.0)
        stats.append(torch.tensor(pacf_stats, device=self.device, dtype=self.dtype))
        stats.append(torch.tensor(mi_stats, device=self.device, dtype=self.dtype).unsqueeze(-1))

        return torch.cat(stats, dim=-1)

    def __compute_analytic_signal_stats(self) -> torch.Tensor:
        """
        Computes the following analytic signal statistics:
            (1) Mean amplitude
            (2) Amplitude variance
            (3) Amplitude coefficient of variation
            (4) Mean frequency
            (5) Frequency variance
            (6) Frequency coefficient of variation
            (7) Amplitude-frequency correlation
        :return: the analytic signal statistics with shape: (batch_size, 8)
        """
        # analytical signal features
        stats = []

        # all of this must be detached from the GPU to use CPU-specific libray
        x_to_cpu = self.x.detach().cpu().numpy()

        try:
            xa = signal.hilbert(x_to_cpu, axis=-1)
            amp = np.abs(xa)
            phase = np.unwrap(np.angle(xa), axis=-1)
            freq = np.diff(phase, axis=-1) / (2 * np.pi * self.dt) # 2 pi f = dphi / dt

            mean_amp = np.mean(amp, axis=-1, keepdims=True)
            stats.append(torch.tensor(mean_amp, device=self.device, dtype=self.dtype))
            amp_var = np.var(amp, axis=-1, keepdims=True)
            stats.append(torch.tensor(amp_var, device=self.device, dtype=self.dtype))
            amp_cv = np.sqrt(amp_var) / np.clip(mean_amp, a_min=1e-9, a_max=None)
            stats.append(torch.tensor(amp_cv, device=self.device, dtype=self.dtype))

            mean_freq = np.mean(freq, axis=-1, keepdims=True)
            stats.append(torch.tensor(mean_freq, device=self.device, dtype=self.dtype))
            freq_var = np.mean(freq, axis=-1, keepdims=True)
            stats.append(torch.tensor(freq_var, device=self.device, dtype=self.dtype))
            freq_cv = np.sqrt(freq_var) / np.clip(mean_freq, a_min=1e-9, a_max=None)
            stats.append(torch.tensor(freq_cv, device=self.device, dtype=self.dtype))

            # amplitude-frequency correlation
            amp_trimmed = amp[:, :-1] # need to trim amplitude array because freq array is shape (batch_size, n - 1)
            amp_centered = amp_trimmed - np.mean(amp_trimmed, axis=-1, keepdims=True)
            freq_centered = freq - np.mean(freq, axis=-1, keepdims=True)

            num = np.sum(amp_centered * freq_centered, axis=-1, keepdims=True)
            den = np.sqrt(np.sum(amp_centered ** 2, axis=-1, keepdims=True) * np.sum(freq_centered ** 2, axis=-1, keepdims=True))
            af_corr = num / np.clip(den, a_min=1e-9, a_max=None)
            stats.append(torch.tensor(af_corr, device=self.device, dtype=self.dtype))
        except Exception:
            nan_tensor = torch.full((self.batch_size, 1), np.nan, device=self.device, dtype=self.dtype)
            for i in range(7):
                stats.append(nan_tensor)

        return torch.cat(stats, dim=-1)

    def __compute_nonlinear_stats(self, downsample_factor: int = 1000, order: int = 2, tolerance: float = None,
                                  nperseg: int = 256, noverlap: int = 128, nfft: int = 512) -> torch.Tensor:
        """
        Computes the following nonlinear statistics:
            (1) Time irreversibility
            (2) Sample entropy
            (3) Correlation dimensions
            (4) Hurst exponent
            (5) Mean bicoherence
        :param downsample_factor: the downsampling factor
        :param order: the order to use for the sample entropy calculation
        :param tolerance: the tolerance to use for the sample entropy calculation
        :param nperseg: the number of segments to use for bicoherence calculation
        :param noverlap: the overlap value to use for bicoherence calculation
        :param nfft: the FFT size to use for bicoherence calculation
        :return: the nonlinear statistics with shape: (batch_size, 4)
        """
        stats = []

        # time irreversibility
        lagged_signal = self.x[:, 1:] - self.x[:, :-1]
        second_moment = torch.mean(torch.pow(lagged_signal, 2), dim=-1)
        third_moment = torch.mean(torch.pow(lagged_signal, 3), dim=-1)
        t_irrev = third_moment / torch.clamp(torch.pow(second_moment, 1.5), min=1e-9)
        stats.append(t_irrev.unsqueeze(-1))

        # for the rest of the statistics, move to cpu for specific libraries
        x_to_cpu = self.x.detach().cpu().numpy()
        step = max(1, self.n // downsample_factor)
        x_downsampled = np.ascontiguousarray(x_to_cpu[:, ::step]) # downsample for high time-complexity calculations

        samp_en_stats = []
        corr_dim_stats = []
        hurst_stats = []
        mean_bicoherence_stats = []
        for i in range(self.batch_size):
            x_curr = x_to_cpu[i]
            x_curr_downsampled = x_downsampled[i]
            if not np.all(np.isfinite(x_curr)):
                samp_en_stats.append(np.nan)
                corr_dim_stats.append(np.nan)
                hurst_stats.append(np.nan)
                mean_bicoherence_stats.append(np.nan)
                continue

            try:
                # sample entropy
                samp_en_stats.append(ap.sample_entropy(x_curr_downsampled, order=order, tolerance=tolerance))

                # correlation dimension
                corr_dim_stats.append(ap.higuchi_fd(x_curr_downsampled))

                # hurst exponent
                lags = range(2, 20)
                vars = [np.std(np.subtract(x_curr_downsampled[lag:], x_curr_downsampled[:-lag])) for lag in lags]
                reg = np.polyfit(np.log(lags), np.log(vars), 1)
                hurst_stats.append(reg[0])

                # bicoherence
                freqs, _, coeff = signal.stft(x_curr, fs=(1 / self.dt), nperseg=nperseg, noverlap=noverlap, nfft=nfft)
                coeff = coeff.T
                coeff = coeff[:, np.newaxis, :]
                coeff = np.ascontiguousarray(coeff)
                try:
                    bispectrum = Bispectrum(data=coeff, freqs=freqs, sampling_freq=(1 / self.dt))
                    bispectrum.compute(indices=((0,), (0,)))
                    data = bispectrum.results.get_results()
                    abs_data = np.abs(data[0])
                    mean_bicoherence_stats.append(np.mean(abs_data))
                except Exception:
                    mean_bicoherence_stats.append(float('nan'))
            except Exception:
                samp_en_stats.append(np.nan)
                corr_dim_stats.append(np.nan)
                hurst_stats.append(np.nan)
                mean_bicoherence_stats.append(np.nan)

        stats.append(torch.tensor(samp_en_stats, dtype=self.dtype, device=self.device).unsqueeze(-1))
        stats.append(torch.tensor(corr_dim_stats, dtype=self.dtype, device=self.device).unsqueeze(-1))
        stats.append(torch.tensor(hurst_stats, dtype=self.dtype, device=self.device).unsqueeze(-1))
        stats.append(torch.tensor(mean_bicoherence_stats, dtype=self.dtype, device=self.device).unsqueeze(-1))

        return torch.cat(stats, dim=-1)

    def __compute_phase_space_stats(self):