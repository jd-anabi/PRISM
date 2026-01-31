import math

from tqdm import tqdm
import torch
import numpy as np
from scipy import signal
import antropy as ap
from pybispectra.general import Bispectrum
from sklearn.feature_selection import mutual_info_regression
from statsmodels.tsa.stattools import pacf

from ..Helpers import helpers

class SummaryStatistics:
    def __init__(self, x: torch.Tensor, dt: float):
        self.x = x.detach()
        self.dt = dt
        self.fs = 1 / dt
        self.batch_size = x.shape[0]
        self.n = x.shape[1]
        self.device = x.device
        self.dtype = x.dtype

    # --- PUBLIC FUNCTIONS --- #
    def compute_statistics(self, n_bands: int, n_lags: int, pacf_lags: int) -> torch.Tensor:
        """
        Compute a comprehensive set of summary statistics from various mathematical and statistical domains.

        This function aggregates statistics related to distribution features, spectral properties, temporal
        dependencies, analytic signal characteristics, nonlinear dynamics, phase space reconstructions,
        information theory, and extreme event analysis. The final set of features is returned as a tensor,
        ensuring that any NaN values are replaced with zeros.

        :param n_bands: The number of frequency bands to consider for computing spectral statistics.
        :param n_lags: The number of lags to analyze for temporal statistics.
        :param pacf_lags: The number of lags to compute for partial autocorrelation statistics.
        :return: A tensor containing the concatenated statistics. NaN values are replaced with zeros.
        :rtype: torch.Tensor
        """
        with torch.no_grad():
            progress_bar = tqdm(total=9, desc="Getting summary statistics", leave=False)
            dist_stats = self.__compute_stat_dist_features()
            progress_bar.update()
            spectral_stats = self.__compute_spectral_stats(n_bands)
            progress_bar.update()
            temporal_stats = self.__compute_temporal_stats(n_lags, pacf_lags)
            progress_bar.update()
            analytic_signal_stats = self.__compute_analytic_signal_stats()
            progress_bar.update()
            nonlinear_stats = self.__compute_nonlinear_stats()
            progress_bar.update()
            phase_space_stats = self.__compute_phase_space_stats()
            progress_bar.update()
            info_theoretic_stats = self.__compute_info_theoretic_stats()
            progress_bar.update()
            extreme_events_stats = self.__compute_extreme_events_stats()
            progress_bar.update()

            all_stats = torch.cat([dist_stats, spectral_stats, temporal_stats, analytic_signal_stats,
                                   phase_space_stats, nonlinear_stats, info_theoretic_stats, extreme_events_stats], dim=-1)
            progress_bar.update()

            # NaN check
            all_stats = torch.nan_to_num(all_stats, nan=0.0)
            progress_bar.close()

        return all_stats

    # --- PRIVATE FUNCTIONS --- #
    def __compute_stat_dist_features(self) -> torch.Tensor:
        """
        Computes statistical distribution features:
            (1) Mean
            (2) Variance
            (3) Skewness
            (4) Kurtosis
            (5) Bimodality coefficient
            (6-10) Quantiles (5%, 25%, 50%, 75%, 95%)
            (11) MAD
        :return: statistical distribution features with shape: (batch_size, 11)
        """
        batch_size = self.x.shape[0]
        features = torch.empty(batch_size, 11, device=self.device, dtype=self.dtype)

        # Mean and variance
        mean = torch.mean(self.x, dim=-1)
        var = torch.var(self.x, dim=-1)
        features[:, 0] = mean
        features[:, 1] = var

        # Standardized moments
        std = torch.sqrt(torch.clamp(var, min=1e-9))
        z = (self.x - mean.unsqueeze(-1)) / std.unsqueeze(-1)

        z2 = z * z
        skew = torch.mean(z2 * z, dim=-1)
        kurt = torch.mean(z2 * z2, dim=-1)

        features[:, 2] = skew
        features[:, 3] = kurt
        features[:, 4] = (skew * skew + 1.0) / torch.clamp(kurt, min=1e-9)

        # Quantiles
        q_vals = torch.tensor([0.05, 0.25, 0.5, 0.75, 0.95], device=self.device, dtype=self.dtype)
        quantiles = torch.quantile(self.x, q_vals, dim=-1)  # Shape: (5, batch_size)
        features[:, 5:10] = quantiles.T

        # MAD (using median = 50th percentile)
        median = quantiles[2, :]
        features[:, 10] = torch.median(torch.abs(self.x - median.unsqueeze(-1)), dim=-1).values

        return features

    def __compute_spectral_stats(self, n_bands: int) -> torch.Tensor:
        """
        Computes spectral statistics (fully GPU-accelerated):
            (1) Peak frequency
            (2) Quality factor
            (3) Spectral centroid
            (4) Spectral bandwidth
            (5) Spectral entropy
            (6 to 5 + n_bands - 1) Power in n_bands frequency bands
        :param n_bands: the number of frequency bands
        :return: spectral statistics with shape: (batch_size, 5 + n_bands)
        """
        batch_size, n = self.x.shape
        n_features = 5 + n_bands
        features = torch.empty(batch_size, n_features, device=self.device, dtype=self.dtype)

        # FFT and PSD
        x_centered = self.x - torch.mean(self.x, dim=-1, keepdim=True)
        xf = torch.fft.rfft(x_centered, dim=-1)
        freqs = torch.fft.rfftfreq(n, d=self.dt, device=self.device)
        n_freqs = freqs.shape[0]
        df = freqs[1] - freqs[0]  # Frequency resolution

        psd = torch.clamp(torch.abs(xf), max=1e15) ** 2 * self.dt / n

        # Peak Frequency
        peak_pwr, peak_idx = torch.max(psd, dim=-1)
        peak_freqs = freqs[peak_idx]
        features[:, 0] = peak_freqs

        # Quality Factor
        # Find FWHM: frequency width where PSD > peak_pwr / 2
        half_max_pwr = peak_pwr / 2
        above_half = psd > half_max_pwr.unsqueeze(-1)

        # Find first and last indices above half-max
        freq_indices = torch.arange(n_freqs, device=self.device)

        # Mask invalid positions with extreme values
        indices_for_min = torch.where(above_half, freq_indices, n_freqs)
        indices_for_max = torch.where(above_half, freq_indices, -1)

        first_idx = torch.min(indices_for_min, dim=-1).values
        last_idx = torch.max(indices_for_max, dim=-1).values

        bandwidth_fwhm = (last_idx - first_idx).to(self.dtype) * df

        # Q = f_peak / bandwidth, with protection against division by zero
        q_factor = torch.where(
            bandwidth_fwhm > 0,
            peak_freqs / bandwidth_fwhm,
            torch.zeros_like(peak_freqs)
        )
        features[:, 1] = q_factor

        # Spectral Centroid
        psd_sum = torch.sum(psd, dim=-1, keepdim=True)
        psd_sum_safe = torch.clamp(psd_sum, min=1e-9)

        spectral_centroid = torch.sum(freqs * psd, dim=-1) / psd_sum_safe.squeeze(-1)
        features[:, 2] = spectral_centroid

        # Spectral Bandwidth
        freq_deviation = freqs - spectral_centroid.unsqueeze(-1)
        spectral_bandwidth = torch.sqrt(
            torch.sum(freq_deviation ** 2 * psd, dim=-1) / psd_sum_safe.squeeze(-1)
        )
        features[:, 3] = spectral_bandwidth

        # Spectral Entropy
        psd_normalized = psd / psd_sum_safe
        log_psd = torch.log(torch.clamp(psd_normalized, min=1e-9))
        spectral_entropy = -torch.sum(psd_normalized * log_psd, dim=-1)
        features[:, 4] = spectral_entropy

        # Band Powers (Vectorized)
        # Compute the cumulative sum for efficient band power calculation
        psd_cumsum = torch.cumsum(psd, dim=-1)
        # Prepend zero for easier indexing: cumsum[end] - cumsum[start]
        psd_cumsum = torch.cat([torch.zeros(batch_size, 1, device=self.device, dtype=self.dtype), psd_cumsum], dim=-1)  # Shape: (batch_size, n_freqs + 1)

        # Band edges as indices
        band_edges = torch.linspace(0, n_freqs, n_bands + 1, device=self.device, dtype=torch.long)

        # Extract cumsum values at band edges: shape (batch_size, n_bands + 1)
        edge_values = psd_cumsum[:, band_edges]

        # Band power = cumsum[end] - cumsum[start], multiplied by frequency width
        band_powers = (edge_values[:, 1:] - edge_values[:, :-1])

        # Multiply by bandwidth in frequency units
        band_widths = (band_edges[1:] - band_edges[:-1]).to(self.dtype) * df
        band_powers = band_powers * band_widths

        features[:, 5:] = band_powers

        return features

    def __compute_temporal_stats(self, n_lags: int, pacf_lags: int) -> torch.Tensor:
        """
        Computes temporal statistics (fully GPU-accelerated):
            (1 to n_lags - 1) Autocorrelation at n_lags evenly spaced lags
            (n_lags) Decorrelation time
            (n_lags + 1 to n_lags + pacf_lags) Partial autocorrelation (PACF)
            (n_lags + pacf_lags + 1) Nonlinear dependence (ACF of x^2 at lag 1)
        :param n_lags: number of ACF lags to sample
        :param pacf_lags: number of PACF lags to compute
        :return: temporal statistics with shape: (batch_size, n_lags + pacf_lags + 2)
        """
        batch_size, n = self.x.shape
        n_features = n_lags + pacf_lags + 2
        features = torch.empty(batch_size, n_features, device=self.device, dtype=self.dtype)

        if n_lags > n or n_lags <= 1:
            raise ValueError("n_lags must be in (1, series_length]")
        if pacf_lags > n or pacf_lags <= 1:
            raise ValueError("pacf_lags must be in (1, series_length]")

        # Autocorrelation via FFT (Wiener-Khinchin)
        x_centered = self.x - torch.mean(self.x, dim=-1, keepdim=True)
        xf = torch.fft.rfft(x_centered, n=2 * n, dim=-1)
        psd = torch.abs(xf) ** 2
        acf_full = torch.fft.irfft(psd, n=2 * n, dim=-1)[:, :n]

        # Normalize so acf[:, 0] = 1
        acf_normalized = acf_full / torch.clamp(acf_full[:, 0:1], min=1e-9)

        # Sample ACF at evenly spaced lags
        lag_idx = torch.linspace(0, n - 1, n_lags, device=self.device).long()
        features[:, :n_lags] = acf_normalized[:, lag_idx]

        # Decorrelation Time
        negative_mask = acf_normalized < 0
        has_negative = negative_mask.any(dim=-1)
        first_negative_idx = torch.argmax(negative_mask.int(), dim=-1)

        decorrelation_time = first_negative_idx.to(self.dtype) * self.dt
        decorrelation_time[~has_negative] = n * self.dt
        features[:, n_lags] = decorrelation_time

        # PACF via Levinson-Durbin (inlined)
        # The PACF φ_{kk} measures direct correlation at lag k, removing intermediate effects
        # Recursion: φ_{kk} = (ρ_k - Σ_{j = 1}^{k-1} φ_{k-1,j} ρ_{k-j}) / (1 - Σ_{j = 1}^{k-1} φ_{k-1,j} ρ_j)

        pacf_vals = torch.zeros(batch_size, pacf_lags, device=self.device, dtype=self.dtype)
        phi = torch.zeros(batch_size, pacf_lags, device=self.device, dtype=self.dtype)

        # Base case: φ_{1,1} = ρ_1
        pacf_vals[:, 0] = acf_normalized[:, 1]
        phi[:, 0] = acf_normalized[:, 1]

        for k in range(1, pacf_lags):
            # φ_{k-1,j} for j = 1, ..., k-1 are in phi[:, :k]
            phi_coeffs = phi[:, :k].clone()  # Need clone since we'll modify phi

            # ρ_{k-j} for j = 1, ..., k-1 → indices k-1, k-2, ..., 1 (reversed)
            rho_reversed = acf_normalized[:, 1:k + 1].flip(dims=[1])

            # ρ_j for j = 1, ..., k-1 → indices 1, 2, ..., k-1
            rho_forward = acf_normalized[:, 1:k + 1]

            # Numerator: ρ_k - Σ φ_{k-1,j} ρ_{k-j}
            numer = acf_normalized[:, k + 1] - torch.sum(phi_coeffs * rho_reversed, dim=1)

            # Denominator: 1 - Σ φ_{k-1,j} ρ_j
            denom = 1.0 - torch.sum(phi_coeffs * rho_forward, dim=1)

            # Safe division
            denom_safe = torch.where(denom.abs() < 1e-9, torch.ones_like(denom), denom)
            pacf_k = numer / denom_safe
            pacf_k = torch.where(denom.abs() < 1e-9, torch.zeros_like(pacf_k), pacf_k)

            pacf_vals[:, k] = pacf_k

            # Update AR coefficients: φ_{k,j} = φ_{k-1,j} - φ_{k,k} * φ_{k-1,k-j}
            # Vectorized update for all j in [0, k-1]
            phi_new = phi_coeffs - pacf_k.unsqueeze(-1) * phi_coeffs.flip(dims=[1])
            phi[:, :k] = phi_new
            phi[:, k] = pacf_k

        features[:, n_lags + 1:n_lags + 1 + pacf_lags] = pacf_vals

        # Nonlinear Dependence: ACF of x² at lag 1
        x2_centered = x_centered ** 2
        x2_centered = x2_centered - torch.mean(x2_centered, dim=-1, keepdim=True)

        x2f = torch.fft.rfft(x2_centered, n=2 * n, dim=-1)
        psd_x2 = torch.abs(x2f) ** 2
        acf_x2 = torch.fft.irfft(psd_x2, n=2 * n, dim=-1)[:, :n]
        acf_x2_normalized = acf_x2 / torch.clamp(acf_x2[:, 0:1], min=1e-9)

        features[:, -1] = acf_x2_normalized[:, 1]

        return features

    def __compute_analytic_signal_stats(self) -> torch.Tensor:
        """
        Computes analytic signal statistics via Hilbert transform (fully GPU-accelerated):
            (1) Mean amplitude
            (2) Amplitude variance
            (3) Mean frequency
            (4) Frequency variance
            (5) Amplitude-frequency correlation
        :return: analytic signal statistics with shape: (batch_size, 5)
        """
        batch_size, n = self.x.shape
        features = torch.empty(batch_size, 5, device=self.device, dtype=self.dtype)

        # Hilbert transform via FFT (GPU-native)
        # The analytic signal z(t) = x(t) + i*H[x(t)] can be computed as:
        # 1. Take FFT of x
        # 2. Create a filter: h[0] = 1, h[1:n//2] = 2, h[n//2] = 1 (if n even), h[n//2+1:] = 0
        # 3. Multiply FFT by filter
        # 4. Inverse FFT gives the analytic signal

        xf = torch.fft.fft(self.x, dim=-1)

        # Construct the Hilbert filter
        h = torch.zeros(n, device=self.device, dtype=self.dtype)
        if n % 2 == 0:
            h[0] = 1.0
            h[1:n // 2] = 2.0
            h[n // 2] = 1.0
            # h[n//2+1:] remains 0
        else:
            h[0] = 1.0
            h[1:(n + 1) // 2] = 2.0
            # h[(n+1)//2:] remains 0

        # Apply filter and inverse FFT to get analytic signal
        analytic = torch.fft.ifft(xf * h.unsqueeze(0), dim=-1)

        # Instantaneous amplitude and phase
        amplitude = torch.abs(analytic)
        phase = torch.angle(analytic)

        # Unwrap phase (vectorized)
        phase_diff = torch.diff(phase, dim=-1)
        # Detect jumps greater than pi and correct
        jumps = torch.zeros_like(phase)
        jumps[:, 1:] = torch.cumsum(
            -2 * torch.pi * torch.round(phase_diff / (2 * torch.pi)),
            dim=-1
        )
        phase_unwrapped = phase + jumps

        # Instantaneous frequency: f = (1/2π) * dφ/dt
        inst_freq = torch.diff(phase_unwrapped, dim=-1) / (2 * torch.pi * self.dt)

        # Amplitude statistics
        mean_amp = torch.mean(amplitude, dim=-1)
        var_amp = torch.var(amplitude, dim=-1)
        features[:, 0] = mean_amp
        features[:, 1] = var_amp

        # Frequency statistics
        mean_freq = torch.mean(inst_freq, dim=-1)
        var_freq = torch.var(inst_freq, dim=-1)
        features[:, 2] = mean_freq
        features[:, 3] = var_freq

        # Amplitude-frequency correlation
        # Trim amplitude to match frequency length (n-1)
        amp_trimmed = amplitude[:, :-1]

        amp_centered = amp_trimmed - torch.mean(amp_trimmed, dim=-1, keepdim=True)
        freq_centered = inst_freq - mean_freq.unsqueeze(-1)

        numerator = torch.mean(amp_centered * freq_centered, dim=-1)
        denominator = torch.std(amp_trimmed, dim=-1) * torch.std(inst_freq, dim=-1)

        af_corr = numerator / torch.clamp(denominator, min=1e-9)
        features[:, 4] = af_corr

        return features

    def __compute_nonlinear_stats(self, hurst_lags: int = 20, m: int = 2,
                                  r_factor: float = 0.2, n_pairs: int = 5000) -> torch.Tensor:
        """
        Computes nonlinear statistics (fully GPU-accelerated):
            (1) Time irreversibility
            (2) Hurst exponent
            (3) Approximate sample entropy
        :param hurst_lags: maximum lag for Hurst exponent estimation
        :param m: embedding dimension for sample entropy
        :param r_factor: tolerance as a fraction of std for sample entropy
        :param n_pairs: number of random pairs to sample for sample entropy
        :return: nonlinear statistics with shape: (batch_size, 3)
        """
        batch_size, n = self.x.shape
        features = torch.empty(batch_size, 3, device=self.device, dtype=self.dtype)

        # Time Irreversibility
        # Skewness of first differences: T = <Δx³> / <Δx²>^(3/2)
        dx = self.x[:, 1:] - self.x[:, :-1]

        dx2 = dx * dx
        dx3 = dx2 * dx

        second_moment = torch.mean(dx2, dim=-1)
        third_moment = torch.mean(dx3, dim=-1)

        t_irrev = third_moment / torch.clamp(second_moment.pow(1.5), min=1e-9)
        features[:, 0] = t_irrev

        # Hurst Exponent (Vectorized)
        # H is estimated from: std(x[t+lag] - x[t]) ∝ lag^H
        # Regress log(std) ~ log(lag)

        lags = torch.arange(2, hurst_lags + 1, device=self.device, dtype=self.dtype)
        n_lags = len(lags)

        # Compute std of lagged differences for all lags
        lagged_stds = torch.empty(batch_size, n_lags, device=self.device, dtype=self.dtype)

        for idx, lag in enumerate(lags.int().tolist()):
            diff = self.x[:, lag:] - self.x[:, :-lag]
            lagged_stds[:, idx] = torch.std(diff, dim=-1)

        # Linear regression: log(std) = H * log(lag) + c
        log_lags = torch.log(lags)
        log_stds = torch.log(torch.clamp(lagged_stds, min=1e-9))

        # Batched linear regression: H = Cov(log_lag, log_std) / Var(log_lag)
        mean_log_lag = torch.mean(log_lags)
        mean_log_std = torch.mean(log_stds, dim=-1)

        log_lags_centered = log_lags - mean_log_lag
        log_stds_centered = log_stds - mean_log_std.unsqueeze(-1)

        cov = torch.mean(log_lags_centered.unsqueeze(0) * log_stds_centered, dim=-1)
        var_log_lag = torch.mean(log_lags_centered ** 2)

        hurst = cov / var_log_lag
        features[:, 1] = hurst

        # Approximate Sample Entropy
        # This measures complexity via template matching
        # SampEn = -log(A/B) where:
        #   B = probability that two templates of length m match within tolerance r
        #   A = probability that two templates of length m+1 match within tolerance r

        # Tolerance based on signal std
        r = r_factor * torch.std(self.x, dim=-1, keepdim=True)  # Shape: (batch_size, 1)

        # Random indices for pairs (same across batch for efficiency)
        max_idx = n - m - 1
        if max_idx < 1:
            # Time series too short for sample entropy
            features[:, 2] = torch.zeros(batch_size, device=self.device, dtype=self.dtype)
            return features

        idx_i = torch.randint(0, max_idx, (n_pairs,), device=self.device)
        idx_j = torch.randint(0, max_idx, (n_pairs,), device=self.device)

        # Build templates of length m and m+1
        # Shape: (batch_size, n_pairs, m) and (batch_size, n_pairs, m+1)
        templates_m_i = torch.stack([self.x[:, idx_i + k] for k in range(m)], dim=-1)
        templates_m_j = torch.stack([self.x[:, idx_j + k] for k in range(m)], dim=-1)
        templates_m1_i = torch.stack([self.x[:, idx_i + k] for k in range(m + 1)], dim=-1)
        templates_m1_j = torch.stack([self.x[:, idx_j + k] for k in range(m + 1)], dim=-1)

        # Chebyshev distance (L-infinity norm)
        dist_m = torch.max(torch.abs(templates_m_i - templates_m_j), dim=-1).values
        dist_m1 = torch.max(torch.abs(templates_m1_i - templates_m1_j), dim=-1).values

        # Count matches (probability of match)
        B = torch.mean((dist_m < r).float(), dim=-1)
        A = torch.mean((dist_m1 < r).float(), dim=-1)

        # Sample entropy: -log(A/B) = log(B) - log(A)
        samp_en = torch.log(torch.clamp(B, min=1e-9)) - torch.log(torch.clamp(A, min=1e-9))
        features[:, 2] = samp_en

        return features

    def __compute_phase_space_stats(self, n_pairs: int = 10000,
                                            epsilon_factor: float = 0.1) -> torch.Tensor:
        """
        Approximate RQA statistics via random pair sampling (GPU-accelerated).
        Only computes recurrence rate (other RQA stats require line structure analysis).

        :param n_pairs: number of random pairs to sample
        :param epsilon_factor: threshold as a fraction of std
        :return: shape (batch_size, 1)
        """
        batch_size, n = self.x.shape

        # Threshold
        epsilon = epsilon_factor * torch.std(self.x, dim=-1, keepdim=True)

        # Random pairs
        idx_i = torch.randint(0, n, (n_pairs,), device=self.device)
        idx_j = torch.randint(0, n, (n_pairs,), device=self.device)

        # Values at random indices
        x_i = self.x[:, idx_i]  # Shape: (batch_size, n_pairs)
        x_j = self.x[:, idx_j]

        # Recurrence: |x_i - x_j| < epsilon
        recurrent = (torch.abs(x_i - x_j) < epsilon).float()
        rr = torch.mean(recurrent, dim=-1, keepdim=True)

        return rr

    def __compute_info_theoretic_stats(self, order: int = 3) -> torch.Tensor:
        """
        Computes information theoretic statistics (fully GPU-accelerated):
            (1) Permutation entropy (normalized)
        :param order: embedding dimension for permutation entropy (default 3)
        :return: information theoretic statistics with shape: (batch_size, 1)
        """
        batch_size, n = self.x.shape
        features = torch.empty(batch_size, 1, device=self.device, dtype=self.dtype)

        n_windows = n - order + 1
        n_patterns = math.factorial(order)

        if n_windows < order:
            features[:, 0] = torch.zeros(batch_size, device=self.device, dtype=self.dtype)
            return features

        # Extract sliding windows: shape (batch_size, n_windows, order)
        windows = self.x.unfold(dimension=-1, size=order, step=1)

        # Get ordinal patterns via argsort
        patterns = torch.argsort(windows, dim=-1)  # Shape: (batch_size, n_windows, order)

        # Lehmer Code Encoding (Correct Permutation Index)
        # The Lehmer code maps each permutation to a unique index in [0, m!-1]
        # For position i, count how many elements to the right are smaller than patterns[i]
        # Then: index = sum_{i=0}^{m-1} (count_i * (m-1-i)!)

        # Compute factorial coefficients: [(m-1)!, (m-2)!, ..., 1!, 0!]
        factorials = torch.tensor(
            [math.factorial(order - 1 - i) for i in range(order)],
            device=self.device, dtype=torch.long
        )

        # For each position, count elements to the right that are smaller
        # patterns shape: (batch_size, n_windows, order)
        # We need to compare patterns[:, :, i] with patterns[:, :, j] for all j > i

        # Expand for pairwise comparison
        # patterns_expanded: (batch_size, n_windows, order, 1)
        # patterns_compare: (batch_size, n_windows, 1, order)
        patterns_expanded = patterns.unsqueeze(-1)
        patterns_compare = patterns.unsqueeze(-2)

        # Create mask for "to the right": only compare j > i
        idx = torch.arange(order, device=self.device)
        right_mask = idx.unsqueeze(0) > idx.unsqueeze(1)  # Shape: (order, order)

        # Count smaller elements to the right
        smaller = (patterns_compare < patterns_expanded) & right_mask
        counts_smaller = smaller.sum(dim=-1)  # Shape: (batch_size, n_windows, order)

        # Compute Lehmer code
        pattern_indices = torch.sum(counts_smaller * factorials, dim=-1)  # Shape: (batch_size, n_windows)

        # Ensure indices are in valid range (should be guaranteed, but safety check)
        pattern_indices = pattern_indices.clamp(0, n_patterns - 1)

        # Count pattern frequencies
        counts = torch.zeros(batch_size, n_patterns, device=self.device, dtype=self.dtype)
        counts.scatter_add_(
            dim=1,
            index=pattern_indices,
            src=torch.ones(batch_size, n_windows, device=self.device, dtype=self.dtype)
        )

        # Compute probabilities and entropy
        probs = counts / n_windows
        log_probs = torch.log(torch.clamp(probs, min=1e-10))
        entropy = -torch.sum(probs * log_probs, dim=-1)

        # Normalize by maximum entropy
        max_entropy = math.log(n_patterns)
        features[:, 0] = entropy / max_entropy

        return features

    def __compute_extreme_events_stats(self, threshold_factor: float = 1.5) -> torch.Tensor:
        """
        Computes extreme events statistics (fully GPU-accelerated):
            (1) Maximum excursion (max - mean)
            (2) Zero crossing rate
            (3) Threshold crossing rate
            (4) Mean burst duration
            (5) Mean return time
        :param threshold_factor: multiplier of std for burst threshold (default 1.5)
        :return: extreme events statistics with shape: (batch_size, 5)
        """
        batch_size, n = self.x.shape
        features = torch.empty(batch_size, 5, device=self.device, dtype=self.dtype)

        # Centered signal
        mean_x = torch.mean(self.x, dim=-1, keepdim=True)
        centered = self.x - mean_x
        std_x = torch.std(self.x, dim=-1, keepdim=True)

        # Maximum Excursion
        max_excursion = torch.max(centered, dim=-1).values
        features[:, 0] = max_excursion

        # Zero Crossing Rate
        # Count sign changes in centered signal
        signs = torch.sign(centered)
        sign_changes = torch.abs(torch.diff(signs, dim=-1)) > 0
        zcr = torch.mean(sign_changes.float(), dim=-1)
        features[:, 1] = zcr

        # Burst Detection (Threshold Crossings)
        threshold = threshold_factor * std_x
        is_burst = (centered > threshold).float()  # Shape: (batch_size, n)

        # Detect burst starts and ends via diff
        # Prepend 0 and append 0 to detect edges at boundaries
        padded = torch.nn.functional.pad(is_burst, (1, 1), value=0)  # Shape: (batch_size, n+2)
        transitions = torch.diff(padded, dim=-1)  # Shape: (batch_size, n+1)

        # Burst starts: transition from 0 to 1 (diff == 1)
        # Burst ends: transition from 1 to 0 (diff == -1)
        burst_starts = (transitions == 1).float()
        burst_ends = (transitions == -1).float()

        # Threshold Crossing Rate
        # Number of burst initiations per unit time
        n_bursts = torch.sum(burst_starts, dim=-1)
        total_time = n * self.dt
        threshold_rate = n_bursts / total_time
        features[:, 2] = threshold_rate

        # Mean Burst Duration
        # Total time in burst state / number of bursts
        total_burst_time = torch.sum(is_burst, dim=-1) * self.dt
        mean_burst_duration = total_burst_time / torch.clamp(n_bursts, min=1.0)
        features[:, 3] = mean_burst_duration

        # Mean Return Time
        # Average interval between burst starts
        # Return time = total time / number of bursts (for a stationary process)
        # This is an approximation; exact computation requires finding intervals
        mean_return_time = total_time / torch.clamp(n_bursts, min=1.0)
        features[:, 4] = mean_return_time

        return features