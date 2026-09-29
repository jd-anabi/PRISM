"""Pure helpers the probe checks share: recording geometry, default grids, and the spectrum measures.

The probe checks re-measure the chi probe settings -- the band, the drive strength, the masking -- on
a lab's own cell or prior. They measure and never change a setting, and what they measure only means
something if a probe length describes the same trace a training batch of that length holds. So the
geometry here RESTATES the training batch loop's arithmetic and the Sobol pre-filter's ceiling, float
truncation included; it does not replace them. The production copies stay where they are, and they
are not merged with each other either: the pre-filter works in float32 and the batch loop in Python
floats, and the two disagree on a handful of candidates.

Nothing here logs, prints, simulates or refuses. A mismatch between two ensembles handed to the
own-peak ratio is a programming error and raises ``ValueError``.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from typing import NamedTuple

import torch

from core import config


class Geometry(NamedTuple):
    """One recording's sizes on the training grid: the samples kept at the recording rate, the fine
    integration steps simulated (transient included), and the fine steps per kept sample."""
    n_obs: int
    n_fine: int
    subsample: int


def _subsample(cfg, t_scale: float) -> int:
    """Fine integration steps per recorded sample at one t_scale, by the batch loop's rule."""
    dt_nd = cfg.dt_exp / t_scale
    return max(1, round(dt_nd / cfg.dt_nd_min))


def recording_geometry(cfg, t_obs_s: float, t_scale: float) -> Geometry:
    """The geometry of a ``t_obs_s``-second recording at ``t_scale``, computed exactly as the training
    batch loop computes a batch's: the length in cell time units over t_scale, the recording step over
    t_scale, the sample count TRUNCATED from their ratio. No clipping and no correction of that
    truncation -- at t_scale 3.73 one second holds 999 samples, in training as here."""
    t_cell = t_obs_s * cfg.get_unit_conversion_factor("s")
    t_nd = t_cell / t_scale
    dt_nd = cfg.dt_exp / t_scale
    subsample = _subsample(cfg, t_scale)
    n_obs = int(t_nd / dt_nd)
    return Geometry(n_obs=n_obs, n_fine=cfg.steady_idx + n_obs * subsample, subsample=subsample)


def training_ceiling_s(cfg, t_scale: float) -> float:
    """The longest recording, in seconds, that training can draw at ``t_scale``: the ceiling the
    training pre-filter enforces, whose fine grid (transient plus samples times subsample) must fit
    both the cost cap ``N_ND_MAX`` and the length of the pre-simulated time grid."""
    n_fine_max = min(config.N_ND_MAX, cfg.t.shape[0])
    n_obs_cap = max(1, (n_fine_max - cfg.steady_idx) // _subsample(cfg, t_scale))
    return n_obs_cap * cfg.dt_exp / cfg.get_unit_conversion_factor("s")


def default_lengths(cfg, t_scale: float, n: int = 5) -> list[float]:
    """``max(2, n)`` recording lengths in seconds, log-spaced from the shortest expected recording to
    the shorter of the longest expected recording and this cell's training ceiling at ``t_scale``.
    Each is rounded DOWN to 0.01 s: rounding up would put the top length a few fine steps past the
    ceiling. The shortest expected recording alone when the ceiling is not above it."""
    lo = config.T_MIN_EXP_S
    hi = min(config.T_MAX_EXP_S, training_ceiling_s(cfg, t_scale))
    if not hi > lo:
        return [lo]
    return [math.floor(v * 100) / 100 for v in _log_spaced(lo, hi, max(2, n))]


def default_multipliers(n: int = 4) -> list[float]:
    """``max(2, n)`` probe frequencies, as multiples of the cell's own peak, log-spaced across the
    configured chi band and rounded to four places, plus one control outside the band at twice its top
    edge."""
    lo, hi = config.CHI_FREQ_BOUNDS
    return [round(v, 4) for v in _log_spaced(lo, hi, max(2, n))] + [round(2 * hi, 4)]


def _log_spaced(lo: float, hi: float, m: int) -> list[float]:
    """``m`` (at least 2) points log-spaced from ``lo`` to ``hi``, the two ends exactly as given."""
    step = math.log(hi / lo) / (m - 1)
    return [lo] + [lo * math.exp(k * step) for k in range(1, m - 1)] + [hi]


def _ensemble_power(x: torch.Tensor, dt: float) -> tuple[torch.Tensor, torch.Tensor]:
    """The frequency axis and the ensemble-mean power spectrum of ``x`` (rows are traces, the last
    axis is time), in float64: each row demeaned, transformed, squared in magnitude, then averaged
    over rows."""
    x = x.to(torch.float64).reshape(-1, x.shape[-1])
    x = x - x.mean(dim=-1, keepdim=True)
    power = torch.fft.rfft(x, dim=-1).abs().pow(2).mean(dim=0)
    freqs = torch.fft.rfftfreq(x.shape[-1], d=dt, dtype=torch.float64, device=x.device)
    return freqs, power


def own_peak_window(n: int, omega0: float, window_frac: float, dt: float) -> tuple[int, int]:
    """The own-peak window, as the bins ``[lo, hi)`` of the spectrum of ``n``-sample traces sampled
    every ``dt``: the bin nearest ``omega0`` (a frequency, in cycles per unit of ``dt``) with
    ``window_frac * omega0`` either side, never fewer than two bins either side, never the
    zero-frequency bin. Empty (``lo >= hi``) when the traces have no bin above zero frequency.
    ``omega0`` must be finite and positive."""
    n_bins = n // 2 + 1
    df = 1.0 / (n * dt)
    i = min(round(omega0 / df), n_bins - 1)
    hw = max(2, round(window_frac * omega0 / df))
    return max(1, i - hw), min(n_bins, i + hw + 1)


def own_peak_ratio(forced: torch.Tensor, reference: torch.Tensor, omega0: float, window_frac: float,
                   dt: float) -> float:
    """The power the forced ensemble keeps in the cell's own peak, over the undriven reference's.

    ``omega0`` is the undriven peak FREQUENCY in cell frequency units -- cycles per cell time unit, as
    the chi peak estimator returns it -- not an angular frequency; ``dt`` is the sample step in the
    same time unit. The window is ``own_peak_window``'s: the bin nearest ``omega0`` with
    ``window_frac * omega0`` either side, never fewer than two bins either side, and it never includes
    the zero-frequency bin. Near 1 the cell still runs free under the drive; near 0 the drive has
    captured it.

    NaN when ``omega0`` is not finite and positive (no peak to measure), and when the traces are too
    short to have any bin above zero frequency. The two ensembles must hold traces of one length, so
    that their bins are the same frequencies; anything else is a programming error and raises
    ``ValueError``.
    """
    n = forced.shape[-1]
    if n != reference.shape[-1]:
        raise ValueError(
            f"own_peak_ratio: the forced traces hold {n} samples and the reference traces "
            f"{reference.shape[-1]}; the two spectra must share one frequency axis.")
    if not (math.isfinite(omega0) and omega0 > 0):
        return float("nan")
    _, p_forced = _ensemble_power(forced, dt)
    _, p_ref = _ensemble_power(reference, dt)
    lo, hi = own_peak_window(n, omega0, window_frac, dt)
    if lo >= hi:
        return float("nan")
    kept = float(p_forced[lo:hi].sum())
    base = float(p_ref[lo:hi].sum())
    return kept / max(base, 1e-30)


def circular_spread(z: torch.Tensor) -> float:
    """The circular standard deviation of the phases of complex ``z``, in radians: 0 when every entry
    points one way, growing without bound as the phases spread round the circle. Each entry counts
    by its direction only, and two phases either side of plus-or-minus pi read as close, as they are.
    NaN when ``z`` is empty or holds a non-finite entry."""
    unit = z / torch.clamp(z.abs(), min=1e-30)
    r = float(unit.mean().abs())
    if not math.isfinite(r):
        return float("nan")
    return math.sqrt(max(0.0, -2.0 * math.log(max(r, 1e-30))))


def harmonic_flags(multipliers: Sequence[float], window_frac: float, max_order: int = 5) -> list[bool]:
    """For each probe multiplier ``m``, whether a low harmonic ``k * m`` (``k`` from 1, the probe
    itself, to ``max_order``) lands inside the own-peak window ``[1 - window_frac, 1 + window_frac]``.
    Harmonic power there inflates the not-captured measure and could hide capture, so such a probe is
    flagged. The edges carry a small tolerance, so that 3 x 0.3, which floats to just under 0.9, counts
    as on the edge of a 10 % window."""
    lo, hi = 1.0 - window_frac - 1e-9, 1.0 + window_frac + 1e-9
    return [any(lo <= k * float(m) <= hi for k in range(1, max_order + 1)) for m in multipliers]
