"""
Hand-crafted summary statistics for SBI on hair-bundle trajectories.

Design (two trajectories per sample):
  * Groups A-F describe the SPONTANEOUS dynamics and are computed on a dedicated
    UNFORCED (zero-drive) trajectory -- no notching needed, the driven line is simply
    absent.
  * Group G describes the FORCED response and is computed by lock-in to the KNOWN drive
    (amp, freq, phase) on the separate forced trajectory.

Conventions (per the spec NOTES):
  * Exactly ONE mean-retaining feature (A1, the absolute mean -> x_offset); every other
    feature is computed on the demeaned signal.
  * Positive / unbounded quantities (frequencies, Q, variance, decay times, gains, ratios)
    are emitted log-transformed, so sbi's per-feature z-scoring standardizes the log.
  * Phase-valued features (G2, G5, the G7 orientation) are emitted as (cos, sin) pairs to
    avoid 2*pi wrap discontinuities.
  * Output dimensionality is FIXED regardless of the input (a missing secondary peak etc.
    falls back to a sentinel), so the conditioning-vector width is constant across
    training and inference.

Units: x is the dimensional trajectory sampled at dt (cell time units); the drive
(amp, freq, phase) is dimensional. f*dt and the local time axis t_j = j*dt are therefore
dimensionless / consistent. Group G locks in against LOCAL time (sample 0 = t=0), so
G2 = arg(R_1) - phase carries t_offset modulo the drive period, as the identifiability
notes require.

Two-run notes:
  * The spontaneous run is fully unforced (zero drive, including any DC force offset), so
    A1-A4 read the natural operating point; the DC offset shifts only the forced run.
  * With both a spontaneous (A2 -> x_scale) and a forced (G1 -> x_scale/f_scale) trajectory,
    x_scale and f_scale are separately identifiable.
"""
import math

import torch

_EPS = 1e-12

# Order matches compute_statistics(); length is asserted against the output width.
FEATURE_LABELS = [
    # Group A -- dimensional anchors
    "A1_mean", "A2_log_var", "A3_log_fpeak", "A4_log_acf_decay",
    # Group B -- spectral shape
    "B1_log_Q", "B2_log_peak_floor", "B3_log_centroid_ratio", "B4_spec_entropy",
    "B5_pow_frac_peakband", "B6_low_freq_frac", "B6_log_rolloff_ratio",
    "B7_log_sec_freq_ratio", "B7_sec_height_ratio",
    # Group C -- amplitude envelope
    "C1_log_norm_env", "C2_log_env_cv", "C3_env_onset", "C4_mode_ratio",
    "C5_env_skew", "C6_log_slowenv_corrtime", "C7_log_slowenv_relvar",
    # Group D -- marginal distribution
    "D1_excess_kurt", "D2_skew", "D3_bimodality",
    # Group E -- multi-timescale / waveform
    "E1_log_tau_fast", "E1_log_tau_slow", "E1_w_fast", "E2_log_h2", "E2_log_h3",
    # Group F -- phase-amplitude coupling
    "F1_noniso_slope", "F2_corr_omega_A2",
    # Group G -- forced response
    "G1_log_gain", "G2_cos", "G2_sin", "G3_log_h2_ratio", "G4_log_h3_ratio",
    "G5_cos", "G5_sin", "G6_plv", "G7_log_amp", "G7_orient_cos", "G7_orient_sin",
]


def _logp(x: torch.Tensor) -> torch.Tensor:
    """Log of a strictly-positive quantity, clamped for safety."""
    return torch.log(torch.clamp(x, min=_EPS))


def _cossin(theta: torch.Tensor) -> torch.Tensor:
    """(B,) angle -> (B, 2) of (cos, sin)."""
    return torch.stack([torch.cos(theta), torch.sin(theta)], dim=-1)


class SummaryStatistics:
    def __init__(self, x_spont: torch.Tensor, x_forced: torch.Tensor, dt: float | torch.Tensor,
                 drive_amp, drive_freq, drive_phase,
                 band_halfwidth: int = 2,
                 bp_lo: float = 0.5, bp_hi: float = 1.5, slow_env_frac: float = 0.15):
        """
        :param x_spont: unforced (spontaneous) trajectory for Groups A-F, shape (B, n).
        :param x_forced: forced (driven) trajectory for Group G, shape (B, n). Same shape as x_spont.
        :param dt: sampling interval (scalar; a per-sample tensor is collapsed to its mean
                   for the shared frequency grid).
        :param drive_amp/drive_freq/drive_phase: dimensional drive params, scalar or (B,).
        :param band_halfwidth: spectral band half-width in FFT bins (B7 / E2 harmonic powers).
        :param bp_lo, bp_hi: envelope band-pass edges as fractions of the centre frequency.
        :param slow_env_frac: slow-envelope low-pass cutoff as a fraction of f_peak.
        """
        x_spont = x_spont.detach()
        x_forced = x_forced.detach()
        assert x_spont.shape == x_forced.shape, "spontaneous and forced trajectories must share shape"
        self.B, self.n = x_spont.shape
        self.device = x_spont.device
        self.dtype = x_spont.dtype
        self.dt = float(dt.float().mean().item()) if torch.is_tensor(dt) else float(dt)

        self.amp = self._as_col(drive_amp)      # (B, 1)
        self.f = self._as_col(drive_freq)       # (B, 1)
        self.phase = self._as_col(drive_phase)  # (B, 1)

        self.hw = band_halfwidth
        self.bp_lo, self.bp_hi = bp_lo, bp_hi
        self.slow_env_frac = slow_env_frac

        self.rfreqs = torch.fft.rfftfreq(self.n, d=self.dt, device=self.device).to(self.dtype)  # (nfr,)
        self.ffreqs = torch.fft.fftfreq(self.n, d=self.dt, device=self.device).to(self.dtype)   # (n,)
        self.df = (self.rfreqs[1] - self.rfreqs[0]).item() if self.rfreqs.numel() > 1 else 1.0

        # A1 (the one mean-retaining feature) comes from the spontaneous run -> x_offset.
        self.mean = x_spont.mean(dim=-1)                                 # (B,)
        self.x_spont = x_spont - self.mean.unsqueeze(-1)                 # demeaned spontaneous (Groups A-F)
        self.x0_forced = x_forced - x_forced.mean(dim=-1, keepdim=True)  # demeaned forced (Group G)
        self._build_spectral()

    # --- helpers ---------------------------------------------------------------
    def _as_col(self, v) -> torch.Tensor:
        """scalar / (B,) -> (B, 1) on the right device/dtype."""
        if not torch.is_tensor(v):
            return torch.full((self.B, 1), float(v), device=self.device, dtype=self.dtype)
        v = v.to(device=self.device, dtype=self.dtype)
        if v.dim() == 0:
            return v.view(1, 1).expand(self.B, 1)
        return v.reshape(self.B, 1)

    def _acf(self, sig: torch.Tensor) -> torch.Tensor:
        """Normalised autocorrelation (acf[:, 0] == 1) via Wiener-Khinchin."""
        s = sig - sig.mean(dim=-1, keepdim=True)
        sf = torch.fft.rfft(s, n=2 * self.n, dim=-1)
        ac = torch.fft.irfft(sf.abs() ** 2, n=2 * self.n, dim=-1)[:, :self.n]
        return ac / torch.clamp(ac[:, :1], min=_EPS)

    def _corr_time(self, sig: torch.Tensor) -> torch.Tensor:
        """1/e decay time (s) of the normalised ACF, falling back to the record length."""
        ac = self._acf(sig)
        below = ac < math.exp(-1.0)
        has = below.any(dim=-1)
        idx = torch.argmax(below.int(), dim=-1)
        tau = idx.to(self.dtype) * self.dt
        return torch.where(has, tau, torch.full_like(tau, self.n * self.dt))

    @staticmethod
    def _unwrap(phase: torch.Tensor) -> torch.Tensor:
        d = torch.diff(phase, dim=-1)
        adj = torch.zeros_like(phase)
        adj[:, 1:] = torch.cumsum(-2 * math.pi * torch.round(d / (2 * math.pi)), dim=-1)
        return phase + adj

    def _analytic_bandpass(self, sig: torch.Tensor, f_lo: torch.Tensor, f_hi: torch.Tensor) -> torch.Tensor:
        """Analytic signal of `sig` band-passed to [f_lo, f_hi] (per-sample, (B,1)). Complex (B, n)."""
        xf = torch.fft.fft(sig, dim=-1)
        ff = self.ffreqs.unsqueeze(0)                                  # (1, n)
        keep = (ff > 0) & (ff >= f_lo) & (ff <= f_hi)                  # (B, n)
        h = keep.to(self.dtype) * 2.0
        return torch.fft.ifft(xf * h, dim=-1)

    def _lowpass(self, sig: torch.Tensor, fc: torch.Tensor) -> torch.Tensor:
        """Zero-phase low-pass below fc (per-sample, (B,1))."""
        xf = torch.fft.rfft(sig, dim=-1)
        keep = self.rfreqs.unsqueeze(0) <= fc                         # (B, nfr)
        return torch.fft.irfft(xf * keep.to(self.dtype), n=self.n, dim=-1)

    @staticmethod
    def _corr(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        am = a - a.mean(dim=-1, keepdim=True)
        bm = b - b.mean(dim=-1, keepdim=True)
        num = (am * bm).mean(dim=-1)
        den = torch.sqrt((am * am).mean(dim=-1) * (bm * bm).mean(dim=-1)).clamp(min=_EPS)
        return num / den

    @staticmethod
    def _slope(xv: torch.Tensor, yv: torch.Tensor) -> torch.Tensor:
        """OLS slope d<y>/dx = cov(x, y) / var(x)."""
        xm = xv - xv.mean(dim=-1, keepdim=True)
        ym = yv - yv.mean(dim=-1, keepdim=True)
        return (xm * ym).mean(dim=-1) / (xm * xm).mean(dim=-1).clamp(min=_EPS)

    def _band_power(self, center: torch.Tensor) -> torch.Tensor:
        """PSD power inside +-hw bins of `center` ((B,1)), summed -> (B,)."""
        mask = (self.rfreqs.unsqueeze(0) - center).abs() <= (self.hw * self.df)
        return (self.psd * mask).sum(dim=-1)

    def _build_spectral(self):
        """Cache PSD / peak frequency from the demeaned SPONTANEOUS trajectory (no notch)."""
        xf = torch.fft.rfft(self.x_spont, dim=-1)                    # (B, nfr)
        self.psd = (xf.abs().clamp(max=1e15) ** 2) * self.dt / self.n
        psd_nodc = self.psd.clone()
        psd_nodc[:, 0] = 0.0
        self.psd_nodc = psd_nodc
        peak_idx = psd_nodc.argmax(dim=-1)                           # (B,)
        self.peak_idx = peak_idx
        self.f_peak = self.rfreqs[peak_idx].clamp(min=self.df)       # (B,)
        self.peak_pwr = psd_nodc.gather(1, peak_idx.unsqueeze(1)).squeeze(1)  # (B,)

    # --- GROUP A: dimensional anchors -----------------------------------------
    def _group_a(self) -> torch.Tensor:
        var = self.x_spont.var(dim=-1)
        a4 = _logp(self._corr_time(self.x_spont))
        return torch.stack([
            self.mean,                       # A1 absolute mean (NOT demeaned) -> x_offset
            _logp(var),                      # A2 log variance -> x_scale
            _logp(self.f_peak),              # A3 log peak frequency -> t_scale
            a4,                              # A4 log ACF decay time -> t_scale
        ], dim=-1)

    # --- GROUP B: spectral shape ----------------------------------------------
    def _group_b(self) -> torch.Tensor:
        psd, fr = self.psd, self.rfreqs
        fpk = self.f_peak.unsqueeze(1)                               # (B, 1)

        # B1 quality factor from FWHM of the spontaneous peak
        half = (self.peak_pwr / 2).unsqueeze(1)
        above = psd > half
        idxs = torch.arange(fr.numel(), device=self.device)
        first = torch.where(above, idxs, torch.full_like(idxs, fr.numel())).min(dim=-1).values
        last = torch.where(above, idxs, torch.full_like(idxs, -1)).max(dim=-1).values
        fwhm = (last - first).clamp(min=0).to(self.dtype) * self.df
        q = torch.where(fwhm > 0, self.f_peak / fwhm, torch.zeros_like(self.f_peak))

        # B2 peak-to-floor ratio (median over positive frequencies as the floor)
        floor = torch.median(psd[:, 1:], dim=-1).values
        b2 = self.peak_pwr / floor.clamp(min=_EPS)

        psd_sum = psd.sum(dim=-1, keepdim=True).clamp(min=_EPS)
        centroid = (fr.unsqueeze(0) * psd).sum(dim=-1) / psd_sum.squeeze(-1)

        pn = psd / psd_sum
        entropy = -(pn * _logp(pn)).sum(dim=-1) / math.log(max(fr.numel(), 2))

        in_peak = (fr.unsqueeze(0) >= self.bp_lo * fpk) & (fr.unsqueeze(0) <= self.bp_hi * fpk)
        b5 = (psd * in_peak).sum(dim=-1) / psd_sum.squeeze(-1)

        low = fr.unsqueeze(0) < 0.5 * fpk
        b6_low = (psd * low).sum(dim=-1) / psd_sum.squeeze(-1)
        # 95%-power spectral rolloff frequency
        cdf = torch.cumsum(psd, dim=-1) / psd_sum
        roll_idx = torch.argmax((cdf >= 0.95).int(), dim=-1)
        rolloff = fr[roll_idx]

        # B7 secondary spectral peak (highest interior local max away from f_peak)
        left = self.psd_nodc[:, 1:-1]
        is_max = (left > self.psd_nodc[:, :-2]) & (left > self.psd_nodc[:, 2:])
        near_peak = (fr[1:-1].unsqueeze(0) - fpk).abs() <= (self.hw * self.df)
        cand = torch.where(is_max & ~near_peak, left, torch.zeros_like(left))
        sec_pwr, sec_off = cand.max(dim=-1)
        sec_freq = fr[sec_off + 1]
        has_sec = sec_pwr > (0.05 * self.peak_pwr)
        sec_freq_ratio = torch.where(has_sec, sec_freq / self.f_peak, torch.ones_like(sec_freq))
        sec_height = torch.where(has_sec, sec_pwr / self.peak_pwr.clamp(min=_EPS), torch.zeros_like(sec_pwr))

        return torch.stack([
            _logp(q),                                   # B1
            _logp(b2),                                  # B2
            _logp(centroid / self.f_peak),              # B3
            entropy,                                    # B4 (already 0-1)
            b5,                                         # B5 (fraction)
            b6_low,                                     # B6 low-frequency fraction
            _logp(rolloff / self.f_peak),               # B6 rolloff ratio
            _logp(sec_freq_ratio),                      # B7 secondary freq ratio
            sec_height,                                 # B7 secondary height ratio
        ], dim=-1)

    # --- GROUP C: amplitude envelope (band-passed Hilbert) --------------------
    def _group_c(self) -> torch.Tensor:
        fpk = self.f_peak.unsqueeze(1)
        z = self._analytic_bandpass(self.x_spont, self.bp_lo * fpk, self.bp_hi * fpk)
        amp = z.abs()                                               # (B, n)
        mean_a = amp.mean(dim=-1)
        mean_a2 = (amp * amp).mean(dim=-1)
        std_a = amp.std(dim=-1)
        sqrt_var_x = torch.sqrt(self.x_spont.var(dim=-1).clamp(min=_EPS))

        za = (amp - mean_a.unsqueeze(-1)) / std_a.clamp(min=_EPS).unsqueeze(-1)
        env_skew = (za ** 3).mean(dim=-1)

        a_slow = self._lowpass(amp, self.slow_env_frac * fpk)
        c6 = self._corr_time(a_slow) * self.f_peak
        c7 = a_slow.var(dim=-1) / mean_a.clamp(min=_EPS) ** 2

        return torch.stack([
            _logp(mean_a / sqrt_var_x),                 # C1 normalised mean envelope
            _logp(std_a / mean_a.clamp(min=_EPS)),      # C2 envelope CV
            mean_a ** 2 / mean_a2.clamp(min=_EPS),      # C3 onset ratio in [pi/4, 1]
            self._mode(amp) / mean_a.clamp(min=_EPS),   # C4 mode / <A>
            env_skew,                                   # C5 envelope skewness
            _logp(c6),                                  # C6 slow-envelope corr time * f_peak
            _logp(c7),                                  # C7 slow-envelope relative variance
        ], dim=-1)

    def _mode(self, a: torch.Tensor, nb: int = 64) -> torch.Tensor:
        """Histogram-peak estimate of the per-sample mode of `a` (B, n) -> (B,)."""
        amin = a.min(dim=-1, keepdim=True).values
        amax = a.max(dim=-1, keepdim=True).values
        rng = (amax - amin).clamp(min=_EPS)
        idx = ((a - amin) / rng * (nb - 1)).long().clamp(0, nb - 1)
        hist = torch.zeros(self.B, nb, device=self.device, dtype=self.dtype)
        hist.scatter_add_(1, idx, torch.ones_like(a))
        mbin = hist.argmax(dim=-1).to(self.dtype)
        return amin.squeeze(-1) + (mbin + 0.5) / nb * rng.squeeze(-1)

    # --- GROUP D: marginal distribution of x ----------------------------------
    def _group_d(self) -> torch.Tensor:
        s = self.x_spont
        mu = s.mean(dim=-1, keepdim=True)
        std = s.std(dim=-1, keepdim=True).clamp(min=_EPS)
        z = (s - mu) / std
        skew = (z ** 3).mean(dim=-1)
        kurt = (z ** 4).mean(dim=-1)
        bimod = (skew ** 2 + 1.0) / kurt.clamp(min=_EPS)            # Sarle's coefficient
        return torch.stack([kurt - 3.0, skew, bimod], dim=-1)      # D1, D2, D3

    # --- GROUP E: multi-timescale / waveform ----------------------------------
    def _group_e(self) -> torch.Tensor:
        ac = self._acf(self.x_spont)                               # (B, n)
        n = self.n

        def first_below(thr):
            below = ac < thr
            idx = torch.argmax(below.int(), dim=-1)
            idx = torch.where(below.any(dim=-1), idx, torch.full_like(idx, n - 1))
            return idx.clamp(min=2)

        l1 = first_below(0.5)
        l2 = torch.maximum(first_below(0.1), l1 + 1).clamp(max=n - 1)
        log_ac = _logp(ac.clamp(min=_EPS))
        a1 = log_ac.gather(1, l1.unsqueeze(1)).squeeze(1)
        a2v = log_ac.gather(1, l2.unsqueeze(1)).squeeze(1)
        # two-window log-ACF slopes -> fast/slow decay times (approximation, not an NLS fit)
        slope_fast = a1 / (l1.to(self.dtype) - 1.0).clamp(min=1.0)         # log_ac[1] ~ 0
        slope_slow = (a2v - a1) / (l2 - l1).to(self.dtype).clamp(min=1.0)
        tau_fast = (-1.0 / slope_fast.clamp(max=-1e-6)) * self.dt
        tau_slow = (-1.0 / slope_slow.clamp(max=-1e-6)) * self.dt
        w_fast = (1.0 - ac.gather(1, l1.unsqueeze(1)).squeeze(1)).clamp(0.0, 1.0)

        p1 = self._band_power(self.f_peak.unsqueeze(1))
        e2_h2 = self._band_power(2 * self.f_peak.unsqueeze(1)) / p1.clamp(min=_EPS)
        e2_h3 = self._band_power(3 * self.f_peak.unsqueeze(1)) / p1.clamp(min=_EPS)

        return torch.stack([
            _logp(tau_fast), _logp(tau_slow), w_fast,  # E1
            _logp(e2_h2), _logp(e2_h3),                # E2
        ], dim=-1)

    # --- GROUP F: phase-amplitude coupling / nonisochronicity -----------------
    def _group_f(self) -> torch.Tensor:
        fpk = self.f_peak.unsqueeze(1)
        z = self._analytic_bandpass(self.x_spont, self.bp_lo * fpk, self.bp_hi * fpk)
        amp2 = (z.abs() ** 2)[:, :-1]                              # align with omega (n-1)
        phi = self._unwrap(torch.angle(z))
        omega = torch.diff(phi, dim=-1) / self.dt                 # rad/s
        # F1: slope of <omega | A^2>, normalised to (A^2/<A^2>) and 2*pi*f_peak
        slope = self._slope(amp2, omega) * amp2.mean(dim=-1)
        f1 = slope / (2 * math.pi * self.f_peak).clamp(min=_EPS)
        f2 = self._corr(omega, amp2)
        return torch.stack([f1, f2], dim=-1)

    # --- GROUP G: forced response (lock-in to the known drive) ----------------
    def _group_g(self) -> torch.Tensor:
        t = torch.arange(self.n, device=self.device, dtype=self.dtype) * self.dt   # (n,)
        t = t.unsqueeze(0)                                          # (1, n)
        two_pi_ft = 2 * math.pi * self.f * t                       # (B, n)

        r = []
        for k in (1, 2, 3):
            cexp = torch.exp(torch.complex(torch.zeros_like(two_pi_ft), -k * two_pi_ft))
            r.append((2.0 / self.n) * (self.x0_forced * cexp).sum(dim=-1))   # (B,) complex
        r1, r2, r3 = r
        mag1, mag2, mag3 = r1.abs(), r2.abs(), r3.abs()
        ang1, ang3 = torch.angle(r1), torch.angle(r3)

        g1 = _logp(mag1 / self.amp.squeeze(-1).clamp(min=_EPS))
        g2 = _cossin(ang1 - self.phase.squeeze(-1))
        g3 = _logp(mag2 / mag1.clamp(min=_EPS))
        g4 = _logp(mag3 / mag1.clamp(min=_EPS))
        g5 = _cossin(ang3 - 3 * ang1)

        # G6 phase-locking value of the response band around the drive to the drive phase
        zr = self._analytic_bandpass(self.x0_forced, self.bp_lo * self.f, self.bp_hi * self.f)
        phi_resp = torch.angle(zr)
        plv = torch.exp(torch.complex(torch.zeros_like(phi_resp), phi_resp - two_pi_ft)).mean(dim=-1).abs()

        # G7 drive-phase-binned residual variance (residual = signal minus locked harmonics)
        locked = torch.zeros_like(self.x0_forced)
        for k, rk in zip((1, 2, 3), (r1, r2, r3)):
            ang = (2 * math.pi * k) * (self.f * t)
            locked = locked + (rk.real.unsqueeze(-1) * torch.cos(ang) - rk.imag.unsqueeze(-1) * torch.sin(ang))
        resid2 = (self.x0_forced - locked) ** 2
        phi_d = two_pi_ft + self.phase                            # drive phase at each sample
        a0 = resid2.mean(dim=-1)
        a2c = 2 * (resid2 * torch.cos(2 * phi_d)).mean(dim=-1)
        b2c = 2 * (resid2 * torch.sin(2 * phi_d)).mean(dim=-1)
        g7_amp = _logp(torch.sqrt(a2c ** 2 + b2c ** 2) / a0.clamp(min=_EPS))
        g7_orient = _cossin(0.5 * torch.atan2(b2c, a2c))

        return torch.cat([
            g1.unsqueeze(-1), g2, g3.unsqueeze(-1), g4.unsqueeze(-1), g5,
            plv.unsqueeze(-1), g7_amp.unsqueeze(-1), g7_orient,
        ], dim=-1)

    # --- PUBLIC ----------------------------------------------------------------
    def compute_statistics(self) -> torch.Tensor:
        """
        Assemble Groups A-G into a fixed-width feature tensor of shape
        (B, len(FEATURE_LABELS)). NaN/Inf are mapped to 0.
        """
        with torch.no_grad():
            out = torch.cat([
                self._group_a(), self._group_b(), self._group_c(), self._group_d(),
                self._group_e(), self._group_f(), self._group_g(),
            ], dim=-1)
            assert out.shape[-1] == len(FEATURE_LABELS), (
                f"feature count {out.shape[-1]} != len(FEATURE_LABELS) {len(FEATURE_LABELS)}"
            )
            return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
