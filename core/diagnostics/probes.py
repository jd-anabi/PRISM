"""The probe checks: re-measure the chi probe settings on a lab's own cell.

The chi observation drives a cell at a handful of frequencies below its own oscillation and locks in
on each. Its band, its drive strength and its lock-in ceiling were chosen by measurements on one
cell; a lab bringing another cell can ask whether they still hold there. These checks answer that and
nothing else: they MEASURE, and never change a setting. Each writes one ordinary diagnostic record.

  band  Do the configured band and drive hold for this cell? At every recording length, probe
        frequency and drive strength it drives an ensemble of the cell's own noisy runs and judges
        four criteria, at full length and under the configured lock-in ceiling: the spread of |chi|
        over the repeats, the phase scatter (reported, and judged only when given a threshold), the
        signal over the same lock-in on the undriven runs, and capture -- how much of the cell's own
        peak power survives the drive. Only capture is physical evidence; the other three thresholds
        are conventions, and the record says so.

  mask  Why are training probes thrown out, over a prior? It runs a few batches of the training
        generator itself over the prior -- its schedule, probe draw, placement, duration draw and
        lock-in -- reads each committed row range through the generator's probe observer, and
        splits every masked probe by cause: too slow for the cycle floor even at the band's top over
        the full recording, shortened below the floor by the duration draw, or a lock-in the packer
        found non-finite or zero. It checks that every batch and row asked for reached it, and its
        count against the generator's own masked-probe warning, row range by row range. No
        simulation cache is written.

  drive How hard can a lab drive this cell? It records the cell's own noisy runs undriven -- the
        peak frequency, its clarity, the cycles in the recording -- then drives them at a range of
        strengths at a frequency detuned from the peak, and judges each strength by how much of the
        cell's own peak power survives: free-running, captured, or in between. It names the
        strongest free-running strength and the weakest captured one, in model units and in the
        cell's force unit, and suggests Forcing lines for a cell file, which it never writes.

MEASURING IS NOT OVERRIDING. A check takes its own recording lengths, frequencies, drive strengths and
lock-in ceilings, and they reach the simulator only as the drive builder's frequency and amplitude,
inside this module. None of them passes through a configuration field, through the training
generator's band and drive keywords, or through an assignment to a configuration constant, so a
training run in the same process uses exactly what it would have used without them. The mask audit
hands the training generator the configuration's own band and drive, after the check that they are
core/config.py's. Every record states the configured band, drive, pad, cycle floor, cycle ceiling and
probe count it judged; a deliberate change to any of them is made in core/config.py.

FAITHFUL TO TRAINING. A check measures what training measures: each run's own peak from the chi peak
estimator; a probe at or above 0.9 x Nyquist, non-finite or not positive is MASKED, never moved to a
frequency the cell can be sampled at; the lock-in ceiling is per run and rounds down, in float64;
spectra are float64; the length offset is applied when the box has one; the drive is a cosine (phase
pi/2) built against a literal forcing index, because a box with no Forcing section has no index to
read; and on a box that declares temperature in place of the force scale, every simulation is driven
at the force scale derived from it, as training is.
"""
from __future__ import annotations

import logging
import math
import re
import warnings
from contextlib import contextmanager
from typing import NamedTuple

import numpy as np
import torch

from core import orchestrator as orch
from core.artifacts import resolve_store
from core.Helpers import file_manager
from core.refusals import (Refusal, describe, refuse, require_at_least, require_between, require_finite,
                           require_positive)
from core.rng import require_seed
from core.runs import public_entry

from . import probe_math
from .rng import seeded

# The report is information records; what to act on -- a length training never draws, a probe masked
# at the sampling limit, a verdict that does not hold -- is a warning, one record however many lines.
log = logging.getLogger(__name__)

#: The forcing index every probe drive is built against. A literal: a box with no Forcing section has
#: no forcing index of its own, and chi mode ignores the cell's drive anyway.
_FORCE_IDX = {"amp": 0, "freq": 1, "phase": 2, "offset": 3}
#: The drive's phase. pi/2 turns the sine builder's drive into a cosine, training's convention.
DRIVE_PHASE = math.pi / 2.0
#: A probe at or above this share of the sampling limit is masked, as training masks it.
_NYQUIST_SHARE = 0.9
#: Relative tolerance on "inside the configured band" and "the configured drive".
_REL_TOL = 1e-9

CAVEATS = (
    "The amplitude-spread, signal-over-floor and phase thresholds are conventions; only capture -- the "
    "drive taking over the cell's own oscillation -- is physical evidence against a probe.",
    "A probe whose low harmonic (up to the fifth) lands within a bin of the own-peak window as measured "
    "at that length is marked harmonic: harmonic power there inflates the not-captured measure and can "
    "hide capture.",
)


def _simulate(cfg, geom, nd, res_sim, sim_idx, inits, *, amp_dim=None, freq=None) -> torch.Tensor:
    """``(B, geom.n_obs)`` traces of the observed variable in the cell's length unit: one run per row
    of ``nd`` (ND parameters), ``res_sim`` (the simulator's rescale block, ``sim_idx`` its index) and
    ``inits``, on the training grid ``geom`` describes.

    Undriven when ``amp_dim`` is None. Otherwise row b is driven by a cosine of dimensional amplitude
    ``amp_dim[b]`` at ``freq[b]`` (cell frequency units), built by the sine builder against the
    literal forcing index and padded to the model's force channels as the chi probe loop pads it. The
    frequency is used as given: a probe past the sampling limit is still driven there, never clamped.

    The one seam every simulation of the checks goes through, so a test can stand a cell of known
    response in for it. Both pipeline functions are looked up through the module at call time."""
    from core import config, forcing
    from core.Helpers import helpers
    from core.SBI import pipeline
    B = nd.shape[0]
    dtype, device = cfg.hw.dtype, cfg.hw.device
    t_fine = cfg.t[:geom.n_fine]
    n_ch = forcing.n_force_channels(cfg.model, _FORCE_IDX, inits.shape[-1])
    if amp_dim is None:
        force = forcing.zero_force(B, n_ch, t_fine.shape[0], dtype, device)
    else:
        fp = torch.zeros((B, 4), dtype=dtype, device=device)
        fp[:, _FORCE_IDX["amp"]] = amp_dim
        fp[:, _FORCE_IDX["freq"]] = freq
        fp[:, _FORCE_IDX["phase"]] = DRIVE_PHASE
        force = pipeline.build_nondim_sin_force_tensor(fp, t_fine, res_sim, _FORCE_IDX, sim_idx)
        if force.shape[1] < n_ch:
            padded = torch.zeros((B, n_ch, force.shape[2]), dtype=force.dtype, device=force.device)
            padded[:, :force.shape[1], :] = force
            force = padded
    x = pipeline.gen_obs(model=cfg.model, params=nd, t=t_fine, inits=inits, force=force,
                         n_segs=max(1, math.ceil(geom.n_fine / config.CHUNK_LEN)),
                         steady_idx=cfg.steady_idx, state_dep_drift=cfg.state_dep_drift, batch_size=B,
                         var_idx=0, dtype=dtype, device=device)[0]
    x = x[:, ::geom.subsample][:, :geom.n_obs]
    x_scale = res_sim[:, sim_idx["x_scale"]].unsqueeze(1)
    x_offset = res_sim[:, sim_idx["x_offset"]].unsqueeze(1) if "x_offset" in sim_idx else 0.0
    return helpers.rescale(x, x_scale, x_offset)


def _force_scale(res_sim, sim_idx) -> torch.Tensor:
    """Per row, in float64, the force scale a drive amplitude is built on, exactly as the chi probe
    loop takes it: the simulator block's f_scale -- on a box that declares temperature in its place,
    the one derived from it -- else the Hopf-style x_scale / t_scale."""
    if "f_scale" in sim_idx:
        return res_sim[:, sim_idx["f_scale"]].double()
    return (res_sim[:, sim_idx["x_scale"]] / res_sim[:, sim_idx["t_scale"]]).double()


class _Criteria(NamedTuple):
    cv_max: float
    phase_max: float | None
    snr_min: float
    sup_min: float
    peak_window: float


def _what(key: str) -> str:
    what = describe(key)
    return what[0].upper() + what[1:]


def _configured(cfg) -> dict:
    """The configured chi settings a check judged, as its record states them: the drive, the band,
    the slot count, the cycle floor and ceiling, and the probe count an observation supplies."""
    from core import config
    return {"chi_f0": float(cfg.chi_f0), "chi_freq_bounds": [float(v) for v in cfg.chi_freq_bounds],
            "chi_k_pad": int(cfg.chi_k_pad), "chi_min_cycles": float(config.CHI_MIN_CYCLES),
            "chi_max_cycles": float(cfg.chi_max_cycles), "probe_count": int(cfg.chi_n_freqs)}


def _positive_list(key: str, values) -> list[float] | None:
    """None as given (the check's own rule applies); otherwise a non-empty list whose every value is
    finite and above 0, each as a float."""
    if values is None:
        return None
    values = list(values)
    if not values:
        refuse(key, f"{_what(key)} must hold at least one value; got none.")
    return [require_positive(key, v) for v in values]


def _inside(m: float, lo: float, hi: float) -> bool:
    return lo * (1.0 - _REL_TOL) <= m <= hi * (1.0 + _REL_TOL)


def _lock_in(xd, x0, freq, amp, cap, n_obs, dt):
    """The lock-in measures over the kept runs, each locked in over at most ``cap`` drive cycles
    (None: the full length), and whether the ceiling shortened any run.

    The ceiling is training's, per run and rounded DOWN in float64: ``floor(cap / freq / dt)`` samples,
    at least one and at most the recording. The undriven runs are locked in on at the same frequency
    over the same samples and divided by the same drive amplitude, so the signal over the floor is a
    pure ratio of |chi|. Every measure is a finite float or None: NaN and infinity are not measured."""
    from core import config
    from core.SBI import chi
    if cap is None:
        n_row = torch.full_like(freq, float(n_obs))
    else:
        n_row = torch.floor(cap / freq / dt).clamp(min=1.0, max=float(n_obs))
    n_row = n_row.long()
    t_row = n_row.double() * dt
    omega = 2.0 * math.pi * freq
    chi_d = chi.lock_in_batched(xd, omega, amp, t_row, dt, n_samples=n_row)
    chi_f = chi.lock_in_batched(x0, omega, amp, t_row, dt, n_samples=n_row)
    mag, floor = chi_d.abs(), chi_f.abs()
    cycles = freq * t_row
    measures = {"cv": orch._num(mag.std() / mag.mean()),
                "phase": orch._num(probe_math.circular_spread(chi_d)),
                "snr": orch._num(mag.mean() / floor.mean()),
                "chi_mag": orch._num(mag.mean()),
                "cycles": orch._num(cycles.mean()),
                "floor_masked": orch._num((cycles < config.CHI_MIN_CYCLES).double().mean())}
    return measures, bool((n_row < n_obs).any())


def _verdict(m: dict, sup, crit: _Criteria, *, capture: bool = True) -> str:
    """"pass", the failing tags joined by ", ", or "not measured" when a judged measure is None.

    Every measure arrives as a float or None, so no comparison below ever meets a NaN -- which
    compares false against every threshold and would read as a pass. The phase is judged only when it
    has a threshold; capture is judged over the whole trace, so a per-cap verdict leaves it out."""
    judged = [m["cv"], m["snr"]]
    if crit.phase_max is not None:
        judged.append(m["phase"])
    if capture:
        judged.append(sup)
    if any(v is None for v in judged):
        return "not measured"
    tags = []
    if m["cv"] > crit.cv_max:
        tags.append("noisy")
    if crit.phase_max is not None and m["phase"] > crit.phase_max:
        tags.append("phase")
    if m["snr"] < crit.snr_min:
        tags.append("low signal")
    if capture and sup < crit.sup_min:
        tags.append("captured")
    return ", ".join(tags) or "pass"


def _masked_point(base: dict, caps: list[float]) -> dict:
    """A point with fewer than two runs below the sampling limit: reported, never driven."""
    empty = {"cv": None, "phase": None, "snr": None, "chi_mag": None, "cycles": None,
             "floor_masked": None, "verdict": "masked"}
    return {**base, "sup": None, "full": dict(empty), "capped": dict(empty),
            "caps": [{"cap": c, "cv": None, "phase": None, "snr": None, "cycles": None, "verdict": "masked"}
                     for c in caps]}


def _measure(cfg, res_sim, sim_idx, t_scale, lengths, multipliers, drives, caps, repeats, crit):
    """Simulate and lock in on every (length, multiplier, drive) point, in that nesting order.

    Returns the per-length rows, the point records and, per point, which caps shortened a run. One
    undriven ensemble per length -- the reference every point at that length is judged against -- and
    one driven ensemble per unmasked (multiplier, drive)."""
    from core import config
    from core.SBI import chi
    dt = float(cfg.dt_exp)
    per_s = cfg.get_unit_conversion_factor("s")          # cell time units per second
    nyq = 0.5 / dt
    configured_cap = float(cfg.chi_max_cycles)
    nd = cfg.params_tensor.expand(repeats, -1).contiguous()
    rs = res_sim.expand(repeats, -1).contiguous()
    inits = cfg.inits_tensor.expand(repeats, -1).contiguous()
    f_eff = _force_scale(rs, sim_idx)
    nominal = probe_math.harmonic_flags(multipliers, crit.peak_window)
    n_grid = cfg.t.shape[0]
    n_max = min(config.N_ND_MAX, n_grid)
    ceiling_s = probe_math.training_ceiling_s(cfg, t_scale)
    length_rows, points, shortened = [], [], []
    for length in lengths:
        geom = probe_math.recording_geometry(cfg, length, t_scale)
        beyond, clipped = geom.n_fine > n_max, geom.n_fine > n_grid
        if clipped:
            n_obs = (n_grid - cfg.steady_idx) // geom.subsample
            geom = probe_math.Geometry(n_obs, cfg.steady_idx + n_obs * geom.subsample, geom.subsample)
        achieved = geom.n_obs * dt / per_s
        if beyond:
            log.warning(f"[band] a {length:g} s recording is longer than training can draw for this "
                        f"cell (its ceiling is {ceiling_s:.3f} s): it is measured and marked, but no "
                        f"training batch holds a recording this long.")
        if clipped:
            log.warning(f"[band] a {length:g} s recording runs past the pre-simulated time grid, so it "
                        f"is measured over the {achieved:.3f} s that fit.")
        x0 = _simulate(cfg, geom, nd, rs, sim_idx, inits).double()
        f_peak = chi.peak_freq(x0, dt).double()
        axis, power = probe_math._ensemble_power(x0, dt)
        omega0 = float(axis[1:][power[1:].argmax()]) if power.shape[0] > 1 else float("nan")
        median_hz = float(torch.quantile(f_peak, 0.5)) * per_s
        spread_hz = float(f_peak.std()) * per_s
        # The window the capture share sums at this length, never narrower than two bins either side:
        # on a slow peak or a short recording it is much wider than the nominal fraction, so a probe
        # tone or harmonic is checked against it, as the drive check checks its drive.
        lo = hi = df = None
        if math.isfinite(omega0) and omega0 > 0:
            lo, hi = probe_math.own_peak_window(geom.n_obs, omega0, crit.peak_window, dt)
            df = 1.0 / (geom.n_obs * dt)
            if lo >= hi:
                lo = hi = df = None
        length_rows.append({
            "length_s": length, "achieved_s": achieved, "n_obs": geom.n_obs, "n_fine": geom.n_fine,
            "beyond_ceiling": beyond, "clipped": clipped, "omega0_cell": orch._num(omega0),
            "omega0_hz": orch._num(omega0 * per_s), "peak_median_hz": orch._num(median_hz),
            "peak_spread_hz": orch._num(spread_hz),
            "own_peak_window_x": (None if lo is None else
                                  [orch._num(lo * df / omega0), orch._num((hi - 1) * df / omega0)])})
        log.info(f"[band] {length:g} s ({geom.n_obs} samples): the ensemble's own peak is at "
                 f"{omega0 * per_s:.3f} Hz; per run, median {median_hz:.3f} Hz, spread {spread_hz:.3f} Hz")
        for j, m in enumerate(multipliers):
            freq = m * f_peak
            ok = torch.isfinite(freq) & (freq > 0) & (freq < _NYQUIST_SHARE * nyq)
            n_valid = int(ok.sum())
            # the probe itself (k = 1) and its harmonics up to the fifth
            harmonic = nominal[j] or (lo is not None and any(
                _reaches(k * m * omega0, lo, hi, df) for k in range(1, 6)))
            for drive in drives:
                base = {"length_s": length, "multiplier": m, "drive": drive, "n_valid": n_valid,
                        "nyquist_masked": repeats - n_valid, "harmonic": harmonic}
                if n_valid < 2:
                    points.append(_masked_point(base, caps))
                    shortened.append({c: False for c in caps})
                    continue
                amp = drive * f_eff
                xd = _simulate(cfg, geom, nd, rs, sim_idx, inits, amp_dim=amp, freq=freq).double()
                fk, ak, xdk, x0k = freq[ok], amp[ok], xd[ok], x0[ok]
                full, _ = _lock_in(xdk, x0k, fk, ak, None, geom.n_obs, dt)
                per_cap = {c: _lock_in(xdk, x0k, fk, ak, c, geom.n_obs, dt) for c in caps}
                # The reference is the WHOLE undriven ensemble, not x0[ok]: the kept runs are chosen from
                # x0's own per-run peak estimates, so x0[ok] would condition the reference on x0's noise
                # and bias the share toward "captured". The driven draws are independent of that choice.
                sup = orch._num(probe_math.own_peak_ratio(xdk, x0, omega0, crit.peak_window, dt))
                capped = per_cap[configured_cap][0]
                points.append({
                    **base, "sup": sup,
                    "full": {**full, "verdict": _verdict(full, sup, crit)},
                    "capped": {**capped, "verdict": _verdict(capped, sup, crit)},
                    "caps": [{"cap": c,
                              **{key: per_cap[c][0][key] for key in ("cv", "phase", "snr", "cycles")},
                              "verdict": _verdict(per_cap[c][0], None, crit, capture=False)} for c in caps]})
                shortened.append({c: per_cap[c][1] for c in caps})
    return length_rows, points, shortened


def _frequencies(points, n_l, n_m, n_d, multipliers, band):
    """Per probe frequency: it passes only if it passes at every length and every drive, full length
    and capped judged apart, with a reason for each point that does not. It is marked harmonic when
    any of its points is, at any length."""
    out = []
    for j, m in enumerate(multipliers):
        own = [points[(i * n_m + j) * n_d + k] for i in range(n_l) for k in range(n_d)]
        reasons = []
        for p in own:
            where = f"at {p['length_s']:.2f} s"
            if p["capped"]["verdict"] == "masked":
                why = f"masked at the sampling limit {where}"
                if why not in reasons:
                    reasons.append(why)
            elif p["capped"]["verdict"] != "pass":
                reasons.append(f"{p['capped']['verdict']} {where}, drive {p['drive']:g}")
            elif p["full"]["verdict"] != "pass":
                reasons.append(f"{p['full']['verdict']} {where} uncapped, drive {p['drive']:g}")
        out.append({"multiplier": m, "in_band": _inside(m, *band), "harmonic": any(p["harmonic"] for p in own),
                    "passes_full": all(p["full"]["verdict"] == "pass" for p in own),
                    "passes_capped": all(p["capped"]["verdict"] == "pass" for p in own),
                    "reasons": reasons})
    return out


def _drive_rows(points, drives, n_d, band, configured_f0, sup_min):
    """Per drive, whether it is the configured one and the in-band points it captured; and the drive
    verdict, judged on the configured drive's in-band points alone: True when every one has a capture
    measure at or above the threshold, False when any has one below it, None otherwise."""
    rows, judged = [], []
    for k, drive in enumerate(drives):
        configured = math.isclose(drive, configured_f0, rel_tol=_REL_TOL)
        mine = [p for idx, p in enumerate(points) if idx % n_d == k and _inside(p["multiplier"], *band)]
        captured = [{"length_s": p["length_s"], "multiplier": p["multiplier"], "sup": p["sup"]}
                    for p in mine if p["sup"] is not None and p["sup"] < sup_min]
        rows.append({"drive": drive, "configured": configured, "captured_in_band": captured})
        if configured:
            judged += [p["sup"] for p in mine]
    if not judged:
        holds = None
    elif any(s is not None and s < sup_min for s in judged):
        holds = False
    elif all(s is not None for s in judged):
        holds = True
    else:
        holds = None
    return rows, holds


def _cap_rescue(points, shortened, configured_cap):
    """The points that fail at full length, those the configured ceiling shortened, and those of them
    that pass under it."""
    failed = [i for i, p in enumerate(points) if p["full"]["verdict"] not in ("pass", "masked")]
    cut = [i for i in failed if shortened[i][configured_cap]]
    rescued = [i for i in cut if points[i]["capped"]["verdict"] == "pass"]
    if not failed:
        reading = "no point fails at full length"
    else:
        reading = (f"{len(rescued)} of the {len(failed)} points that fail at full length pass under the "
                   f"configured {configured_cap:g}-cycle ceiling")
        if len(cut) < len(failed):
            reading += f"; the ceiling does not shorten {len(failed) - len(cut)} of them"
    return {"failed_full": len(failed), "shortened": len(cut), "rescued": len(rescued), "reading": reading}


def _wall(rows) -> dict:
    """Where a longer lock-in stops helping, read off the cap grid (``rows`` in increasing cap, each
    with ``cap``, ``n_truncated`` and ``n_fail``): the first cap that fails among those that bind a
    point, the largest clean one below it, and a sentence."""
    binding = [r for r in rows if r["n_truncated"] > 0]
    failing = [r for r in binding if r["n_fail"] > 0]
    first = failing[0]["cap"] if failing else None
    below = [r["cap"] for r in binding if first is not None and r["cap"] < first and r["n_fail"] == 0]
    clean = below[-1] if below else None
    if not binding:
        reading = "no cap binds any point"
    elif first is None:
        reading = f"no cap in the grid fails: the wall lies above {binding[-1]['cap']:g} cycles"
    elif clean is None:
        reading = (f"the smallest cap that binds ({binding[0]['cap']:g} cycles) already fails: the grid "
                   f"does not bracket the wall from below")
    else:
        reading = f"the wall lies between {clean:g} and {first:g} cycles"
    return {"first_failing": first, "largest_clean_below": clean, "reading": reading}


def _wall_report(points, shortened, caps):
    """Per cap, the points it shortened and how they fare locked in over at most that many cycles."""
    rows = []
    for c in caps:
        hit = [i for i, p in enumerate(points) if shortened[i][c]]
        entries = [(points[i], next(e for e in points[i]["caps"] if e["cap"] == c)) for i in hit]

        def worst(key, pick):
            vals = [e[key] for _, e in entries if e[key] is not None]
            return pick(vals) if vals else None

        rows.append({"cap": c, "n_truncated": len(hit),
                     "truncated": [{"length_s": p["length_s"], "multiplier": p["multiplier"],
                                    "drive": p["drive"], "verdict": e["verdict"]} for p, e in entries],
                     "worst_cv": worst("cv", max), "worst_snr": worst("snr", min),
                     "worst_phase": worst("phase", max),
                     "n_fail": sum(e["verdict"] != "pass" for _, e in entries)})
    return {"caps": rows, **_wall(rows)}


def _grid(points, shape, get) -> np.ndarray:
    """One measure over the points as a float array of ``shape``, NaN where it is None."""
    return np.array([np.nan if get(p) is None else float(get(p)) for p in points], dtype=float).reshape(shape)


def _fmt(v, spec: str) -> str:
    return "--" if v is None else format(v, spec)


def _count(n: int, one: str, many: str) -> str:
    return f"{n} {one if n == 1 else many}"


def _point_line(p: dict) -> str:
    """One point's report line: where it is, then each column's verdict and the measures it judged."""
    where = f"{p['length_s']:6.2f} s  x{p['multiplier']:<7g} drive {p['drive']:<6g}"
    if p["full"]["verdict"] == "masked":
        return (f"{where} masked at the sampling limit: {p['n_valid']} of "
                f"{p['n_valid'] + p['nyquist_masked']} runs below 0.9 x Nyquist")

    def column(name):
        c = p[name]
        return (f"{name} {c['verdict']} (cv {_fmt(c['cv'], '.3f')}, snr {_fmt(c['snr'], '.3g')}, "
                f"phase {_fmt(c['phase'], '.2f')}, {_fmt(c['cycles'], '.3g')} cycles)")

    line = (f"{where} {column('full')}  {column('capped')}  own-peak share {_fmt(p['sup'], '.3f')}"
            + ("  (harmonic)" if p["harmonic"] else ""))
    if p["nyquist_masked"]:
        # measured, but over fewer runs than were simulated: the reader is told how many
        line += (f"  {p['nyquist_masked']} of {p['n_valid'] + p['nyquist_masked']} runs masked at the "
                 f"sampling limit")
    return line


def _figure(sink, drive, lengths, multipliers, panels) -> None:
    """One 2 x 2 heatmap for a drive: the amplitude spread and the signal over the floor (log10), at
    full length and capped, one colour scale per measure."""
    from matplotlib import pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(4 + 1.2 * len(multipliers), 3 + 1.2 * len(lengths)),
                             squeeze=False)
    for row, (label, full, capped) in enumerate(panels):
        finite = np.concatenate([a[np.isfinite(a)] for a in (full, capped)])
        vmin, vmax = (float(finite.min()), float(finite.max())) if finite.size else (0.0, 1.0)
        if not vmax > vmin:
            vmin, vmax = vmin - 0.5, vmax + 0.5
        for col, (column, data) in enumerate((("full length", full), ("capped", capped))):
            ax = axes[row][col]
            im = ax.imshow(np.ma.masked_invalid(data), vmin=vmin, vmax=vmax, cmap="viridis",
                           aspect="auto", origin="lower")
            for (i, j), v in np.ndenumerate(data):
                ax.text(j, i, "--" if not np.isfinite(v) else f"{v:.2g}", ha="center", va="center",
                        fontsize=7, color="white")
            ax.set_xticks(range(len(multipliers)))
            ax.set_xticklabels([f"{m:g}" for m in multipliers])
            ax.set_yticks(range(len(lengths)))
            ax.set_yticklabels([f"{x:g}" for x in lengths])
            ax.set_xlabel("probe frequency / peak frequency")
            ax.set_ylabel("recording length (s)")
            ax.set_title(f"{label}, {column}", fontsize=9)
            fig.colorbar(im, ax=ax, fraction=0.046)
    title = f"Probe band at drive {drive:g}"
    fig.suptitle(title)
    fig.tight_layout()
    sink(title, fig)


@public_entry
def probe_band(cfg, *, lengths=None, multipliers=None, drives=None, repeats: int = 24, cycle_caps=None,
               cv_max: float = 0.20, phase_max=None, snr_min: float = 3.0, sup_min: float = 0.50,
               peak_window: float = 0.10, seed: int = 0, name: str = "", note: str = "", fig_sink=None,
               store=None):
    """Do the configured chi band and drive hold for this cell?

    At every (recording length, probe frequency, drive strength) an ensemble of ``repeats`` noisy runs
    of the cell is simulated undriven and driven, each run at its own peak frequency times the
    multiplier, and locked in on at full length and under the configured cycle ceiling. Four criteria
    per point and column: the amplitude spread (unbiased std of |chi| over its mean, at most
    ``cv_max``), the phase scatter (circular spread, judged only against ``phase_max``), the signal over
    the floor (mean |chi| driven over the same lock-in on the undriven runs, at least ``snr_min``), and
    capture (the driven runs' power within ``peak_window`` of the undriven ensemble's peak, never
    narrower than two frequency bins either side, over the undriven power there, at least
    ``sup_min``). The cycle floor is reported per point, never judged. A probe whose tone or low
    harmonic lands within a bin of that window, as measured at its length, is marked harmonic.

    A frequency passes only if it passes at every length and every drive. The band holds when every
    frequency inside it passes under the ceiling; the drive holds when no in-band point at the
    configured drive is captured. A point with fewer than two runs below 0.9 x Nyquist is masked and
    not driven; a judged measure that is not finite reads "not measured". Measures only: see the
    module docstring.

    :param lengths: recording lengths in seconds; None = five log-spaced from the shortest expected
                    recording to this cell's training ceiling. A length beyond that ceiling is measured
                    and marked, never refused.
    :param multipliers: probe frequencies as multiples of each run's own peak; None = four across the
                        configured band plus one control at twice its top.
    :param drives: non-dimensional drive amplitudes; None = the configured chi drive alone.
    :param repeats: noise repeats per point, at least 2.
    :param cycle_caps: lock-in ceilings in drive cycles to bracket the wall with; the configured
                       ceiling is always added. None = the configured ceiling only, and no wall report.
    :param seed: the whole run is seeded with it, and it is recorded.
    """
    from core.SBI import derived
    store = resolve_store(store)
    store.assert_name_free("diagnostic", name)
    if not cfg.chi_mode:
        raise Refusal(
            f"The band check measures the chi probes, so it needs a configuration in chi observation "
            f"mode; this one is in {cfg.observation_mode} mode.", field=None)
    _ = cfg.ground_truth                      # refuses a cell-free config, before anything is spent
    dtype, device = cfg.hw.dtype, cfg.hw.device
    # The cell's simulation block in the config's own dtype, built as a simulated observation builds
    # its own. On the CPU that is float32, so t_scale reads as training's float32 value and a length's
    # sample count is the one training draws.
    res = torch.tensor([[v for v, _ in cfg.rescale_params.values()]], dtype=dtype, device=device)
    res_sim, sim_idx = derived.for_simulation(cfg, cfg.params_tensor, res)
    t_scale = float(res_sim[0, sim_idx["t_scale"]])

    seed = require_seed(seed, key="probe_seed")
    repeats = require_at_least("band_repeats", repeats, 2)
    lengths = _positive_list("probe_lengths", lengths)
    multipliers = _positive_list("probe_multipliers", multipliers)
    drives = _positive_list("probe_drives", drives)
    cycle_caps = _positive_list("probe_cycle_caps", cycle_caps)
    for length in lengths or ():
        if probe_math.recording_geometry(cfg, length, t_scale).n_obs < 1:
            refuse("probe_lengths", f"{_what('probe_lengths')} must each be long enough for at least one "
                                    f"sample; {length:g} s gives none.")
    cv_max = require_finite("cv_max", cv_max)
    snr_min = require_finite("snr_min", snr_min)
    sup_min = require_between("sup_min", sup_min, 0, 1, open_lo=True)
    phase_max = None if phase_max is None else require_finite("phase_max", phase_max)
    peak_window = require_between("band_peak_window", peak_window, 0, 1, open_lo=True, open_hi=True)

    if lengths is None:
        # rounded down to 0.01 s, a cell whose ceiling sits just above the shortest length would
        # repeat a length; each is measured once
        lengths = list(dict.fromkeys(probe_math.default_lengths(cfg, t_scale)))
    if multipliers is None:
        multipliers = [float(m) for m in probe_math.default_multipliers()]
    if drives is None:
        drives = [float(cfg.chi_f0)]
    configured_cap = float(cfg.chi_max_cycles)
    caps = sorted(set(cycle_caps or ()) | {configured_cap})
    band = tuple(float(v) for v in cfg.chi_freq_bounds)
    configured = _configured(cfg)
    settings = {"lengths": lengths, "multipliers": multipliers, "drives": drives, "repeats": repeats,
                "cycle_caps": caps, "cv_max": cv_max, "phase_max": phase_max, "snr_min": snr_min,
                "sup_min": sup_min, "peak_window": peak_window, "seed": seed, "drive_phase": DRIVE_PHASE,
                "configured": configured}
    crit = _Criteria(cv_max, phase_max, snr_min, sup_min, peak_window)
    n_l, n_m, n_d = len(lengths), len(multipliers), len(drives)

    with store.create("diagnostic", cfg, name=name, note=note) as w:
        log.info(f"[band] judging the configured band ({band[0]:g}, {band[1]:g}) x the peak frequency and "
                 f"drive {configured['chi_f0']:g} ({configured['probe_count']} probes in "
                 f"{configured['chi_k_pad']} slots, cycle floor {configured['chi_min_cycles']:g}, ceiling "
                 f"{configured_cap:g}) at {_count(n_l, 'length', 'lengths')} x "
                 f"{_count(n_m, 'frequency', 'frequencies')} x {_count(n_d, 'drive', 'drives')}, "
                 f"{repeats} runs each")
        with seeded(seed, device):
            length_rows, points, shortened = _measure(cfg, res_sim, sim_idx, t_scale, lengths, multipliers,
                                                      drives, caps, repeats, crit)
        for p in points:
            log.info(f"[band] {_point_line(p)}")
        masked = sorted({(p["length_s"], p["multiplier"]) for p in points
                         if p["full"]["verdict"] == "masked"})
        if masked:
            log.warning("[band] masked at the sampling limit -- fewer than two runs put the probe below "
                        "0.9 x Nyquist, so it was reported and not driven:\n"
                        + "\n".join(f"  {length:g} s, x{m:g}" for length, m in masked))

        frequencies = _frequencies(points, n_l, n_m, n_d, multipliers, band)
        in_band = [f for f in frequencies if f["in_band"]]
        band_holds = None if not in_band else all(f["passes_capped"] for f in in_band)
        drive_rows, drive_holds = _drive_rows(points, drives, n_d, band, configured["chi_f0"], sup_min)
        rescue = _cap_rescue(points, shortened, configured_cap)
        wall = None if cycle_caps is None else _wall_report(points, shortened, caps)

        band_text = f"the configured band ({band[0]:g}, {band[1]:g})"
        if band_holds is None:
            band_line = (f"{band_text} was not judged for this cell: no probe frequency measured lies "
                         f"inside it")
        elif band_holds:
            band_line = (f"{band_text} holds for this cell: every probe frequency inside it passes under the "
                         f"configured ceiling at every length and drive")
        else:
            failing = [f for f in in_band if not f["passes_capped"]]
            band_line = (f"{band_text} does not hold for this cell: {len(failing)} of its {len(in_band)} "
                         f"probe frequencies fail -- "
                         + "; ".join(f"x{f['multiplier']:g}: {', '.join(f['reasons'])}" for f in failing))
        drive_text = f"the configured drive {configured['chi_f0']:g}"
        n_captured = sum(len(r["captured_in_band"]) for r in drive_rows if r["configured"])
        if drive_holds is None:
            why = ("it is not among the drives measured" if not any(r["configured"] for r in drive_rows)
                   else "a capture measure inside the band is missing")
            drive_line = f"{drive_text} was not judged for this cell: {why}"
        elif drive_holds:
            drive_line = f"{drive_text} holds for this cell: no probe inside the band captured the cell"
        else:
            drive_line = (f"{drive_text} does not hold for this cell: it captured the cell at "
                          f"{_count(n_captured, 'in-band point', 'in-band points')}")
        for line, holds, setting in ((band_line, band_holds, "band"), (drive_line, drive_holds, "drive")):
            if holds is False:
                log.warning(f"[band] {line}")
                log.info(f"[band] the chi {setting} is set in core/config.py, where a deliberate change is "
                         f"made; this check changes nothing")
            else:
                log.info(f"[band] {line}")
        log.info(f"[band] cap rescue: {rescue['reading']}")
        if wall is not None:
            log.info(f"[band] wall: {wall['reading']}")
        for caveat in CAVEATS:
            log.info(f"[band] note: {caveat}")

        shape = (n_l, n_m, n_d)
        grids = {
            "cv_full": _grid(points, shape, lambda p: p["full"]["cv"]),
            "cv_capped": _grid(points, shape, lambda p: p["capped"]["cv"]),
            "phase_full": _grid(points, shape, lambda p: p["full"]["phase"]),
            "phase_capped": _grid(points, shape, lambda p: p["capped"]["phase"]),
            "snr_full": _grid(points, shape, lambda p: p["full"]["snr"]),
            "snr_capped": _grid(points, shape, lambda p: p["capped"]["snr"]),
            "chi_mag": _grid(points, shape, lambda p: p["full"]["chi_mag"]),
            "chi_mag_capped": _grid(points, shape, lambda p: p["capped"]["chi_mag"]),
            "sup": _grid(points, shape, lambda p: p["sup"]),
            "cycles_full": _grid(points, shape, lambda p: p["full"]["cycles"]),
            "cycles_capped": _grid(points, shape, lambda p: p["capped"]["cycles"]),
        }
        per_cap = {f"{key}_caps": np.stack([_grid(points, shape, lambda p, c=c, key=key: next(
            e[key] for e in p["caps"] if e["cap"] == c)) for c in caps], axis=-1)
            for key in ("cv", "snr", "phase", "cycles")}
        sink = w.fig_sink(fig_sink)
        with np.errstate(divide="ignore", invalid="ignore"):
            log_snr = {col: np.log10(np.where(grids[f"snr_{col}"] > 0, grids[f"snr_{col}"], np.nan))
                       for col in ("full", "capped")}
        for k, drive in enumerate(drives):
            _figure(sink, drive, lengths, multipliers, (
                ("amplitude spread", grids["cv_full"][:, :, k], grids["cv_capped"][:, :, k]),
                ("log10 signal over floor", log_snr["full"][:, :, k], log_snr["capped"][:, :, k])))
        file_manager.atomic_savez(w.payload("probe_band.npz"), {
            **grids, **per_cap,
            "n_valid": np.array([p["n_valid"] for p in points], dtype=int).reshape(shape),
            "lengths": np.asarray(lengths, dtype=float), "multipliers": np.asarray(multipliers, dtype=float),
            "drives": np.asarray(drives, dtype=float), "caps": np.asarray(caps, dtype=float),
            "achieved_s": np.asarray([r["achieved_s"] for r in length_rows], dtype=float),
            "omega0_cell": np.asarray([np.nan if r["omega0_cell"] is None else r["omega0_cell"]
                                       for r in length_rows], dtype=float)})
        results = {"lengths": length_rows, "points": points, "frequencies": frequencies,
                   "drives": drive_rows, "band_holds": band_holds, "drive_holds": drive_holds,
                   "cap_rescue": rescue, "wall": wall,
                   "training_ceiling_s": orch._num(probe_math.training_ceiling_s(cfg, t_scale)),
                   "caveats": list(CAVEATS),
                   "summary": [band_line, drive_line, rescue["reading"]]
                   + ([wall["reading"]] if wall is not None else [])}
        w.parents = {}
        w.config.update(settings)
        w.body = {"diagnostic": "probes", "variant": "band", "settings": settings, "results": results}
    return store.load_diagnostic(w.id)


# ── mask: why training probes are thrown out ─────────────────────────────────────────────────────────

#: The training generator's masked-probe warning: the batch tag, then one row range's masked and
#: simulated probe counts.
_MASKED_WARNING = re.compile(r"(?P<tag>.+?): chi: (?P<masked>\d+)/(?P<total>\d+) probes masked")

#: What a mask record says about its seed.
PROBE_LAYOUT = ("the probe count, the drawn multipliers and the duration draw come from training's own fixed "
                "probe-generator seed, so they do not change with the seed; the placement each row's own "
                "peak and recording length give them does")

#: The three causes a masked probe is attributed to, in the order a tie between them is broken. The
#: cycle floor is the sum of the first two.
_LEAF_CAUSES = ("too_slow_at_band_top", "shortened_by_duration_draw", "non_finite_lock_in")
_OMEGA0_QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95]
_SPAN_QUANTILES = [0.25, 0.5, 0.75]
_MULTIPLIER_QUANTILES = [0.05, 0.5, 0.95]

#: Each cause in words, and what to change when it dominates. The audit itself changes nothing.
_READINGS = {
    "too_slow_at_band_top": ("too slow even at the band's top", "the lever is the prior's "
                             "peak-frequency range"),
    "shortened_by_duration_draw": ("shortened by the duration draw", "the lever is the duration draw"),
    "non_finite_lock_in": ("a non-finite lock-in", "no setting is its lever: probes that cleared the "
                           "floor came back from the lock-in non-finite or zero, which points at the "
                           "simulated traces"),
}


@contextmanager
def _capture_masked_warnings():
    """Collect ``(batch_tag, masked, total)`` from every masked-probe warning raised inside the block,
    in the order raised, into the list it yields.

    Every warning, matched or not, is still handed to the hook in force when the block began, so it
    reaches the console and the run's log exactly as it would have. The filter is "always" inside:
    the two halves of a batch the row halving splits can warn identical text, and the default
    once-per-message registry would swallow the second."""
    seen = []
    with warnings.catch_warnings():
        warnings.simplefilter("always")
        previous = warnings.showwarning

        def _tee(message, category, filename, lineno, file=None, line=None):
            found = _MASKED_WARNING.match(str(message))
            if found:
                seen.append((found["tag"], int(found["masked"]), int(found["total"])))
            previous(message, category, filename, lineno, file, line)

        warnings.showwarning = _tee          # restored by catch_warnings on the way out
        yield seen


class _Tally(NamedTuple):
    counts: dict              # probe and row counts, summed over the records
    batches: dict             # batch tag -> [masked, total, probe count K], in the order first seen
    ranges: list              # per record: (batch tag, lo, hi, masked, total, K)
    f_peak: np.ndarray        # (rows,) each row's own peak frequency, cell units
    span: np.ndarray          # (rows with a valid probe,) max / min of the frequencies driven
    multipliers: np.ndarray   # (valid probes,) each driven frequency over its row's peak


#: How far above the cycle floor a probe the audit labels "below the floor" may read, as a fraction:
#: the record's log-cycle count is single precision, while the generator judged the floor in double.
_CYCLES_RTOL = 1e-6


def _tally(cfg, records) -> _Tally:
    """Count every probe of every record by what happened to it.

    A probe's driven frequency is ``f_peak * exp(u)``, in float64. Not finite or not positive, or at
    or above 0.9 x Nyquist, it was masked before any lock-in (causes that cannot fire in training).
    Otherwise a probe the generator marked invalid failed the cycle floor: too slow even at the
    band's top over the full recording when its ROW's peak times the full length times the band's top
    falls short of the floor, else shortened below it by the duration draw. The packer then drops
    valid probes outside the band, and those whose lock-in is non-finite or zero (a non-finite lock-in,
    here); ``packed_mask`` is in slot order, so it is compared with ``valid`` only through per-row
    counts of its first ``k`` columns.

    The labels are checked against the record: a probe labelled below the floor must have been locked
    in over fewer cycles than the floor, and one labelled shortened by the duration draw must have been
    handed less than the full recording. A probe that fails either was masked by a rule the audit does
    not know, and is counted as such rather than reported as the floor."""
    from core import config
    from core.SBI import chi
    top = float(cfg.chi_freq_bounds[1])
    u_mid, u_half = chi.band_norm(tuple(float(v) for v in cfg.chi_freq_bounds))
    n = dict.fromkeys(("probes", "live", "masked", "floor", "too_slow", "packer", "out_of_band",
                       "non_finite_frequency", "at_or_above_nyquist", "floor_label_at_or_above_the_floor",
                       "shortened_label_at_full_length", "rows", "zero_live", "one_live", "span_rows",
                       "single_probe_rows"), 0)
    batches, ranges, f_peaks, spans, mults = {}, [], [], [], []
    for rec in records:
        k, lo, hi = int(rec.k), int(rec.lo), int(rec.hi)
        f_peak, u, valid = rec.f_peak.double(), rec.u.double(), rec.valid.bool()
        packed = rec.packed_mask[:, :k].bool()
        freq = f_peak.unsqueeze(1) * torch.exp(u)
        bad = ~torch.isfinite(freq) | (freq <= 0)
        nyq = (freq >= _NYQUIST_SHARE * (0.5 / rec.dt_exp)) & ~bad
        floor = ~valid & ~bad & ~nyq
        too_slow = (f_peak * (rec.n_points * rec.dt_exp) * top < config.CHI_MIN_CYCLES).unsqueeze(1)
        shortened = floor & ~too_slow
        out_of_band = valid & torch.isfinite(u) & (((u - u_mid) / u_half).abs() > config.CHI_UHAT_MAX)
        below = torch.exp(rec.logcyc.double()) < config.CHI_MIN_CYCLES * (1.0 + _CYCLES_RTOL)
        full_length = (torch.ones(k, dtype=torch.bool) if rec.duration_frac is None
                       else rec.duration_frac.double() >= 1.0)
        live, n_valid = packed.sum(1), valid.sum(1)
        masked, total = int((~packed).sum()), (hi - lo) * k
        for key, value in (("probes", total), ("live", live.sum()), ("masked", masked),
                           ("floor", floor.sum()), ("too_slow", (floor & too_slow).sum()),
                           ("packer", (n_valid - live).clamp(min=0).sum()),
                           ("out_of_band", out_of_band.sum()), ("non_finite_frequency", bad.sum()),
                           ("at_or_above_nyquist", nyq.sum()),
                           ("floor_label_at_or_above_the_floor", (floor & ~below).sum()),
                           ("shortened_label_at_full_length", (shortened & full_length.unsqueeze(0)).sum()),
                           ("rows", hi - lo), ("zero_live", (live == 0).sum()),
                           ("one_live", (live == 1).sum()),
                           ("span_rows", (n_valid > 0).sum()), ("single_probe_rows", (n_valid == 1).sum())):
            n[key] += int(value)
        batch = batches.setdefault(rec.batch_tag, [0, 0, k])
        batch[0] += masked
        batch[1] += total
        ranges.append((rec.batch_tag, lo, hi, masked, total, k))
        f_peaks.append(f_peak.numpy())
        # each row's highest over its lowest driven frequency, over its valid probes; one probe spans 1
        highest = torch.where(valid, freq, torch.full_like(freq, -math.inf)).amax(1)
        lowest = torch.where(valid, freq, torch.full_like(freq, math.inf)).amin(1)
        spans.append((highest / lowest)[n_valid > 0].numpy())
        mults.append(torch.exp(u)[valid].numpy())

    def _cat(parts):
        return np.concatenate(parts).astype(float) if parts else np.zeros(0)

    return _Tally(n, batches, ranges, _cat(f_peaks), _cat(spans), _cat(mults))


def _quantiles(values: np.ndarray, qs: list) -> list:
    """The ``qs`` quantiles of the finite values, each a float, or all None when there are none."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return [None] * len(qs)
    return [orch._num(v) for v in np.quantile(finite, qs)]


#: What writing a warning off as an abandoned attempt's costs a pairing, against 1 for a mismatch. So
#: a warning is paired with a committed range that could have raised it before it is set aside, and a
#: range the audit counts as clean is never excused by calling the generator's warning for it stale.
_ABANDON_COST = 2


def _attempts(lo: int, hi: int, leaves, k: int) -> "list | None":
    """The attempts the row halving made over rows [lo, hi) of a batch whose committed row ranges are
    ``leaves``: ``[(lo, hi, total, committed)]`` in the order they ran, or None when the leaves are not
    the halving's split. The split is the batch loop's own: a range that fails is re-run in parts of
    half its rows, the last part whatever is left, one part after another."""
    if (lo, hi) in leaves:
        return [(lo, hi, (hi - lo) * k, True)]
    half = (hi - lo) // 2
    if half < 1:
        return None
    out = [(lo, hi, (hi - lo) * k, False)]
    for s in range(lo, hi, half):
        below = _attempts(s, min(s + half, hi), leaves, k)
        if below is None:
            return None
        out += below
    return out


def _pair(attempts: list, masked: dict, got: list) -> tuple:
    """Pair one batch's warnings ``got`` -- ``(masked, total)``, in the order raised -- with its
    ``attempts``, in order: ``(abandoned, {attempt index: warning index})``.

    A warning belongs to an attempt of its size. A committed range warns exactly when it masked
    anything, so pairing it with a warning whose count differs, or leaving a masking range with none,
    each costs 1 (a mismatch); an attempt the halving abandoned may or may not have warned, for free;
    and a warning raised before the attempts that committed -- a whole-batch retry's -- is written off
    at ``_ABANDON_COST``. The cheapest pairing is taken, the fewest written off on a tie."""
    import functools
    n, m = len(attempts), len(got)

    @functools.lru_cache(maxsize=None)
    def best(i: int, j: int) -> tuple:
        if i == n:
            return (0, ()) if j == m else (math.inf, ())
        _lo, _hi, total, committed = attempts[i]
        want = masked.get(i, 0) if committed else None
        cost, rest = best(i + 1, j)
        options = [(cost + (1 if committed and want else 0), rest)]
        if j < m and got[j][1] == total:
            cost, rest = best(i + 1, j + 1)
            options.append((cost + (1 if committed and got[j][0] != want else 0), ((i, j),) + rest))
        return min(options, key=lambda o: o[0])

    cost, skipped, pairs = min(((best(0, j0)[0] + _ABANDON_COST * j0, j0, best(0, j0)[1])
                                for j0 in range(m + 1)), key=lambda o: (o[0], o[1]))
    return skipped, dict(pairs)


def _cross_check(ranges: list, warned: list, *, num_runs: int, run_size: int) -> dict:
    """The audit's reading of the committed row ranges against the training generator's own record of
    them: every batch asked for reached the audit, each batch's committed ranges cover its rows exactly
    once as the row halving splits them, and each range's masked count is the one the generator warned.

    A warning carries its batch and its size, not its rows, so each batch's warnings are paired with
    the attempts the halving made, in the order they ran (``_pair``): a failed attempt's warning is set
    aside as abandoned, and anything left unexplained is a mismatch. A batch the coverage check finds
    wanting is not paired -- its finding stands for it. Nothing is accepted silently: any finding makes
    ``holds`` False."""
    leaves, spans_of, ks = {}, {}, {}             # per batch tag, in the order first seen
    for tag, lo, hi, masked, _total, k in ranges:
        leaves.setdefault(tag, {})[(lo, hi)] = masked
        spans_of.setdefault(tag, []).append((lo, hi))
        ks.setdefault(tag, k)
    captured = {}
    for tag, masked, total in warned:
        captured.setdefault(tag, []).append((masked, total))
    coverage, mismatches, abandoned = [], [], 0
    for tag in leaves:
        spans = sorted(spans_of[tag])
        tiled = (spans[0][0] == 0 and spans[-1][1] == run_size
                 and all(a[1] == b[0] for a, b in zip(spans, spans[1:])))
        attempts = _attempts(0, run_size, leaves[tag], ks[tag]) if tiled else None
        if attempts is None:
            reason = ("its committed rows are not the row halving's split of the batch" if tiled
                      else "its committed rows do not cover the batch exactly once")
            coverage.append({"batch_tag": tag, "ranges": [[lo, hi] for lo, hi in spans], "rows": run_size,
                             "reason": reason})
            continue
        want = {i: leaves[tag][(a[0], a[1])] for i, a in enumerate(attempts) if a[3]}
        got = captured.get(tag, [])
        skipped, pairs = _pair(attempts, want, got)
        abandoned += skipped + sum(1 for i in pairs if not attempts[i][3])
        for i, (lo, hi, total, committed) in enumerate(attempts):
            seen = got[pairs[i]] if i in pairs else None
            if committed and seen != ((want[i], total) if want[i] else None):
                production = None if seen is None else {"masked": seen[0], "total": seen[1]}
                mismatches.append({"batch_tag": tag, "lo": lo, "hi": hi,
                                   "audit": {"masked": want[i], "total": total}, "production": production})
    batches = {"expected": int(num_runs), "committed": len(leaves)}
    return {"row_ranges": len(ranges), "batches": batches, "coverage": coverage, "mismatches": mismatches,
            "abandoned_attempt_warnings": abandoned,
            "holds": batches["expected"] == batches["committed"] and not coverage and not mismatches}


def _audit(cfg, records, warned, *, num_runs: int, run_size: int) -> dict:
    """The mask record's results, from the committed probe records and the masked-probe warnings
    captured beside them; ``num_runs`` and ``run_size`` are what the generator was asked for, which
    the records must cover. Pure: nothing is logged, written or changed; every float goes through
    ``orchestrator._num``. Each cause's share is taken of every probe simulated."""
    tally = _tally(cfg, records)
    n = tally.counts
    probes = n["probes"]

    def share(count):
        return orch._num(count / probes) if probes else None

    counts = {"cycle_floor": n["floor"], "too_slow_at_band_top": n["too_slow"],
              "shortened_by_duration_draw": n["floor"] - n["too_slow"],
              "non_finite_lock_in": n["packer"] - n["out_of_band"]}
    invariants = {key: n[key] for key in ("non_finite_frequency", "at_or_above_nyquist", "out_of_band")}
    invariants["hold"] = not any(invariants.values())
    attribution = {key: n[key]
                   for key in ("floor_label_at_or_above_the_floor", "shortened_label_at_full_length")}
    attribution["holds"] = not any(attribution.values())
    fractions = [orch._num(m / t) if t else None for m, t, _k in tally.batches.values()]
    finite = [f for f in fractions if f is not None]
    per_s = cfg.get_unit_conversion_factor("s")                  # cell time units per second
    cell = _quantiles(tally.f_peak, _OMEGA0_QUANTILES)
    first = max(_LEAF_CAUSES, key=lambda c: counts[c])          # a tie goes to the earlier cause
    return {
        "probes": probes, "live": n["live"], "masked": n["masked"], "masked_fraction": share(n["masked"]),
        "causes": {c: {"count": v, "share": share(v)} for c, v in counts.items()},
        "invariants": invariants,
        "attribution": attribution,
        "per_batch": {"fractions": fractions, "k": [k for _m, _t, k in tally.batches.values()],
                      "mean": orch._num(np.mean(finite)) if finite else None,
                      "sd": orch._num(np.std(finite, ddof=1)) if len(finite) > 1 else None,
                      "n_batches": len(tally.batches)},
        "omega0": {"quantiles": list(_OMEGA0_QUANTILES), "cell_units": cell,
                   "hz": [None if v is None else orch._num(v * per_s) for v in cell]},
        "rows": {"total": n["rows"], "zero_live": n["zero_live"], "one_live": n["one_live"]},
        "span": {"quantiles": list(_SPAN_QUANTILES), "values": _quantiles(tally.span, _SPAN_QUANTILES),
                 "rows": n["span_rows"], "single_probe_rows": n["single_probe_rows"]},
        "driven_multipliers": {"quantiles": list(_MULTIPLIER_QUANTILES),
                               "values": _quantiles(tally.multipliers, _MULTIPLIER_QUANTILES)},
        "dominant_cause": first if n["masked"] and counts[first] > 0 else None,
        "cross_check": _cross_check(tally.ranges, warned, num_runs=num_runs, run_size=run_size),
    }


def _of(count: int, total: int) -> str:
    return f"{count:,} ({100.0 * count / total:.1f}%)" if total else f"{count:,}"


def _joined(values, spec: str, unit: str = "") -> str:
    names = {0.05: "p5", 0.25: "p25", 0.5: "median", 0.75: "p75", 0.95: "p95"}
    return ", ".join(f"{names[q]} {_fmt(v, spec)}{unit}" for q, v in values)


def _log_mask(res: dict) -> None:
    """The mask report: information records, and one warning record per finding that needs acting on
    -- a cause that cannot fire in training firing, a masked probe the audit's causes do not explain,
    and the audit not matching the training generator's own record."""
    from core import config
    probes, c = res["probes"], res["causes"]
    log.info(f"[mask] {_of(res['masked'], probes)} of {probes:,} probes masked, {res['live']:,} live; "
             f"by cause, each a share of every probe:")
    log.info(f"[mask]   the cycle floor: {_of(c['cycle_floor']['count'], probes)}")
    log.info(f"[mask]     too slow even at the band's top over the full recording: "
             f"{_of(c['too_slow_at_band_top']['count'], probes)}")
    log.info(f"[mask]     shortened below the floor by the duration draw: "
             f"{_of(c['shortened_by_duration_draw']['count'], probes)}")
    log.info(f"[mask]   a non-finite or zero lock-in, dropped by the packer: "
             f"{_of(c['non_finite_lock_in']['count'], probes)}")
    cause = res["dominant_cause"]
    if cause is not None:
        what, lever = _READINGS[cause]
        log.info(f"[mask] reading: {what} dominates -- {lever}; this audit changes nothing")
    elif res["masked"]:
        log.info("[mask] reading: none of training's three causes explains the masked probes; see the "
                 "invariants below")
    else:
        log.info("[mask] reading: no probe was masked")
    om = res["omega0"]
    log.info(f"[mask] each row's own peak frequency: "
             f"{_joined(zip(om['quantiles'], om['hz']), '.3g', ' Hz')}")
    pb = res["per_batch"]
    mean = "--" if pb["mean"] is None else f"{100 * pb['mean']:.1f}%"
    sd = "undefined for one batch" if pb["sd"] is None else f"{100 * pb['sd']:.1f}%"
    log.info(f"[mask] masked per batch: mean {mean}, sd {sd}, over {pb['n_batches']} batches (probes per "
             f"batch: {', '.join(str(k) for k in pb['k'])}) -- the batch count is the effective sample size")
    rows = res["rows"]
    log.info(f"[mask] rows left with no live probe: {_of(rows['zero_live'], rows['total'])} of "
             f"{rows['total']:,}; with exactly one: {_of(rows['one_live'], rows['total'])}")
    span, dm = res["span"], res["driven_multipliers"]
    log.info(f"[mask] span of the frequencies each row was driven at, max over min of its valid probes: "
             f"{_joined(zip(span['quantiles'], span['values']), '.3g', 'x')}, over {span['rows']:,} rows, "
             f"{span['single_probe_rows']:,} of them with a single probe")
    log.info(f"[mask] each valid probe's frequency over its row's peak: "
             f"{_joined(zip(dm['quantiles'], dm['values']), '.3g')}")
    inv = res["invariants"]
    if inv["hold"]:
        log.info("[mask] invariants hold: no probe frequency was non-finite, at or above 0.9 x Nyquist, "
                 "or outside the band")
    else:
        log.warning(f"[mask] a cause that cannot fire in training fired: {inv['non_finite_frequency']:,} "
                    f"probe frequencies non-finite or not positive, {inv['at_or_above_nyquist']:,} at or "
                    f"above 0.9 x Nyquist, {inv['out_of_band']:,} outside the band")
    att = res["attribution"]
    if not att["holds"]:
        log.warning(f"[mask] the audit's causes do not explain every masked probe: "
                    f"{att['floor_label_at_or_above_the_floor']:,} counted below the cycle floor were "
                    f"locked in over at least {config.CHI_MIN_CYCLES:g} cycles, and "
                    f"{att['shortened_label_at_full_length']:,} counted as shortened by the duration draw "
                    f"were handed the full recording -- a masking rule the audit does not know may be at "
                    f"work")
    check = res["cross_check"]
    set_aside = check["abandoned_attempt_warnings"]
    aside = (f"; {set_aside:,} warning{'s' if set_aside != 1 else ''} from an abandoned attempt set aside"
             if set_aside else "")
    if check["holds"]:
        log.info(f"[mask] cross-check: every batch and row asked for reached the audit, and the training "
                 f"generator's own warnings confirm every row range's masked count "
                 f"({_count(check['row_ranges'], 'row range', 'row ranges')}){aside}")
        return
    lines = []
    batches = check["batches"]
    if batches["committed"] != batches["expected"]:
        lines.append(f"  {batches['committed']} of the {batches['expected']} batches asked for reached the "
                     f"audit")
    for gap in check["coverage"]:
        rows_seen = ", ".join(f"{lo}-{hi}" for lo, hi in gap["ranges"])
        lines.append(f"  {gap['batch_tag']}: rows {rows_seen} of {gap['rows']} -- {gap['reason']}")
    for m in check["mismatches"]:
        said = ("warned nothing" if m["production"] is None
                else f"warned {m['production']['masked']} of {m['production']['total']}")
        lines.append(f"  {m['batch_tag']}, rows {m['lo']}-{m['hi']}: the audit counts {m['audit']['masked']} "
                     f"of {m['audit']['total']} probes masked, the generator {said}")
    log.warning(f"[mask] the audit does not match the training generator's own record{aside}:\n"
                + "\n".join(lines))


@public_entry
def probe_mask(cfg, prior, *, num_runs: int = 12, run_size: int = 32, chi_k_fixed=None, seed: int = 0,
               name: str = "", note: str = "", fig_sink=None, store=None):
    """Why are training probes thrown out, over this prior?

    Runs ``num_runs`` batches of ``run_size`` rows through the training generator itself, at the
    configuration's own band and drive, over the physical prior -- the (t_scale, recording length)
    schedule, the per-batch probe count, placement and duration draw, the lock-in and the packer are
    all training's -- with the generator's probe observer attached. Each committed row range's record
    is then audited: every masked probe is attributed to the cycle floor (too slow even at the band's
    top over the full recording, or shortened below the floor by the duration draw) or to a
    non-finite or zero lock-in the packer dropped; the three causes that cannot fire in training are
    checked as invariants, and each label against the record. Every batch and row asked for must reach
    the audit, and the count of each row range is compared with the generator's own masked-probe
    warning for it; a gap or a disagreement is recorded and warned, never accepted silently. Nothing
    is written but the diagnostic record: no simulation cache.

    :param prior: the LoadedPrior audited, its parent in the record; loaded, never built.
    :param num_runs: training batches to audit, at least 1. The batch count is the per-batch masked
                     fraction's effective sample size.
    :param run_size: rows per batch, at least 1.
    :param chi_k_fixed: audit one probe count, from 2 to the number of probe slots; None audits
                        training's own per-batch draw of counts. The record states which, and each
                        batch's count.
    :param seed: the whole run is seeded with it, and it is recorded. The probe layout comes from
                 training's own fixed probe-generator seed and does not change with it.
    :param fig_sink: taken as every diagnostic takes it; the audit draws no figure.
    """
    from core import config
    from core.SBI import pipeline, run_guards
    store = resolve_store(store)
    store.assert_name_free("diagnostic", name)
    if not cfg.chi_mode:
        raise Refusal(
            f"The mask audit reads the chi probes training draws, so it needs a configuration in chi "
            f"observation mode; this one is in {cfg.observation_mode} mode.", field=None)
    run_guards._assert_chi_config_is_deliberate(cfg)
    run_guards._assert_user_model_in_sync(cfg)       # the prior's parameters bind to columns by position
    seed = require_seed(seed, key="probe_seed")
    num_runs = require_at_least("mask_num_runs", num_runs, 1)
    run_size = require_at_least("mask_run_size", run_size, 1)
    if chi_k_fixed is not None:
        # both ends before the spend: the generator's own range check fires only inside the first batch
        chi_k_fixed = require_at_least("chi_k_fixed", chi_k_fixed, config.CHI_K_MIN_TRAIN)
        if chi_k_fixed > cfg.chi_k_pad:
            refuse("chi_k_fixed", f"{_what('chi_k_fixed')} must be at most the number of chi probe slots "
                                  f"({cfg.chi_k_pad}); got {chi_k_fixed}.")

    # The probe count the audit judged, in place of the one an observation supplies: training draws a
    # count per batch unless one is held fixed. Each batch's own count is in the results.
    drawn = [int(config.CHI_K_MIN_TRAIN), int(cfg.chi_k_pad)]
    configured = {**_configured(cfg),
                  "probe_count": chi_k_fixed if chi_k_fixed is not None else {"drawn_per_batch_from": drawn}}
    settings = {"num_runs": num_runs, "run_size": run_size, "chi_k_fixed": chi_k_fixed, "seed": seed,
                "configured": configured, "probe_layout": PROBE_LAYOUT}
    stratum = (f"drawn per batch from {drawn[0]} to {drawn[1]}, as training draws it" if chi_k_fixed is None
               else f"held at {chi_k_fixed}")
    n_vars = orch._observation_inits(cfg).shape[-1]         # training is truth-free: no cell needed
    records = []
    with store.create("diagnostic", cfg, name=name, note=note) as w:
        log.info(f"[mask] auditing {_count(num_runs, 'training batch', 'training batches')} of "
                 f"{run_size} rows over the prior {prior.name or prior.id}, the probe count {stratum}")
        band = configured["chi_freq_bounds"]
        log.info(f"[mask] at the configured band ({band[0]:g}, {band[1]:g}) x the peak frequency and drive "
                 f"{configured['chi_f0']:g} ({configured['chi_k_pad']} slots, cycle floor "
                 f"{configured['chi_min_cycles']:g}, ceiling {configured['chi_max_cycles']:g})")
        log.info(f"[mask] {PROBE_LAYOUT}")
        with seeded(seed, cfg.hw.device), _capture_masked_warnings() as warned:
            pipeline.gen_training_data(
                cfg.model, prior.prior, prior.force_prior, cfg.t, run_size=run_size, n_runs=num_runs,
                steady_idx=cfg.steady_idx, dt_nd_min=cfg.dt_nd_min, nd_dim=len(cfg.params_dict),
                forcing_idx=cfg.forcing_idx, rescale_idx=cfg.rescale_idx, dt_exp=cfg.dt_exp,
                t_min_exp=cfg.t_min_exp, t_max_exp=cfg.t_max_exp, t_scale_bounds=cfg.t_scale_bounds,
                state_dep_drift=cfg.state_dep_drift, chi_mode=True, chi_f0=cfg.chi_f0,
                chi_freq_bounds=cfg.chi_freq_bounds, chi_k_pad=cfg.chi_k_pad, chi_k_fixed=chi_k_fixed,
                chi_max_cycles=cfg.chi_max_cycles, n_vars=n_vars, checkpoint=None,
                nd_idx=cfg.tier1_args[0], k_b_cell=cfg.tier1_args[1], dtype=cfg.hw.dtype,
                device=cfg.hw.device, probe_observer=records.append)
        if not records:
            raise RuntimeError("the training generator committed no probe record for the audit to read: "
                               "its probe observer was never called")
        results = _audit(cfg, records, warned, num_runs=num_runs, run_size=run_size)
        _log_mask(results)
        tally = _tally(cfg, records)
        per_batch = list(tally.batches.values())
        file_manager.atomic_savez(w.payload("probe_mask.npz"), {
            "f_peak": tally.f_peak,
            "batch_masked": np.asarray([b[0] for b in per_batch], dtype=np.int64),
            "batch_total": np.asarray([b[1] for b in per_batch], dtype=np.int64),
            "batch_k": np.asarray([b[2] for b in per_batch], dtype=np.int64),
            "span": tally.span})
        w.parents = {"prior": prior.id}
        w.fingerprints["gmm"] = prior.fingerprint
        w.config.update(settings)
        w.body = {"diagnostic": "probes", "variant": "mask", "settings": settings, "results": results}
    return store.load_diagnostic(w.id)


# ── drive: how hard a lab can drive this cell ────────────────────────────────────────────────────────

#: The drive strengths measured when none are given, in model units: from well below the configured chi
#: drive to far past where a cell is captured. The configured chi drive is always measured as well.
_DRIVE_STRENGTHS = (0.01, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0)

#: What every drive record says about the measurement it made.
DRIVE_NOTES = (
    "Free-running versus captured is the criterion: the share of the undriven own-peak power the driven "
    "runs keep inside the own-peak window. There is no linearity test.",
    "The phase-locking value is reported and never judged.",
    "The drive is detuned from the peak frequency so that the own-peak window reads the cell's own "
    "rhythm and not the drive; every verdict is read at that detune.",
    "Pure noise scores a clarity too, the higher the fewer the runs: noise_clarity states the level for "
    "this run's size -- typical, and the one 1 in 100 noise ensembles reaches -- and a clarity threshold "
    "that does not clear it cannot tell an oscillation from noise.",
)

#: Fewer cycles of the peak than this in the recording, and the drive lies only a few frequency bins
#: from the own-peak window: (detune - 1) times the cycles.
_FEW_CYCLES = 30.0

#: What a record with so few cycles adds to its notes.
FEW_CYCLES_NOTE = ("This recording fits fewer than 30 cycles of the peak, so the drive lies only a few "
                   "frequency bins from the own-peak window: unless it falls on a bin, its leakage into the "
                   "window can make a captured cell read in between or free-running.")

#: The drive's harmonics checked against the own-peak window, as the band check checks its probes'.
_HARMONIC_ORDERS = (2, 3, 4, 5)

#: The configured chi drive's own row carries this note: it is driven here at the detuned frequency,
#: never at a chi probe's.
CHI_F0_NOTE = ("its verdict depends on the detune, so this check cannot certify the configured chi drive; "
               "probes band judges that")

#: The suggested bounds box's phase and offset rows: one whole cycle, and the master box's offset range.
_BOX_PHASE = (0.0, 6.283185307)
_BOX_OFFSET = (-50.0, 50.0)


class _Undriven(NamedTuple):
    freqs: torch.Tensor       # the undriven ensemble spectrum's frequency axis, cell frequency units
    power: torch.Tensor       # its ensemble-mean power, float64
    omega0: float             # the highest bin's frequency, zero frequency excluded; NaN when not measured
    clarity: float            # that bin's power over the median power, zero frequency excluded; NaN likewise
    peak_median: float        # each run's own peak frequency: the median over the runs, cell units
    peak_spread: float        # and its standard deviation
    peak_to_peak: float       # each run's largest minus smallest value: the median over the runs


def _undriven(x0: torch.Tensor, dt: float) -> _Undriven:
    """What the drive check reads off the undriven ensemble ``x0`` (runs by samples, float64).

    The peak is the ensemble spectrum's highest bin above zero frequency. Its clarity is that single
    bin's power over the median power above zero frequency. Measured so, pure noise over 16 runs of 5000
    samples scores about 2; summed over a five-bin window and set against the median bin, it would score
    about 6 and pass a clarity of 3. Fewer runs raise the single bin's noise score as well: about 6 over
    two runs. The peak and its clarity are not measured (NaN) when the spectrum is not finite or has no
    bin above zero frequency, and the clarity also when the median power is not above 0. Each run's own
    peak is the chi peak estimator's, as training reads it."""
    from core.SBI import chi
    freqs, power = probe_math._ensemble_power(x0, dt)
    omega0 = clarity = math.nan
    if power.shape[0] > 1 and bool(torch.isfinite(power).all()):
        above = power[1:]
        k = int(above.argmax())
        omega0 = float(freqs[1 + k])
        median = float(torch.quantile(above, 0.5))
        if median > 0:
            clarity = float(above[k]) / median
    f_peak = chi.peak_freq(x0, dt).double()
    return _Undriven(freqs, power, omega0, clarity, float(torch.quantile(f_peak, 0.5)), float(f_peak.std()),
                     float(torch.quantile(x0.amax(dim=-1) - x0.amin(dim=-1), 0.5)))


def _oscillating(clarity, clarity_min: float) -> bool:
    """Whether the undriven ensemble counts as an oscillation: a clarity AT or above ``clarity_min``.
    The clarity arrives as a float or None (not measured), and None is no oscillation."""
    return clarity is not None and clarity >= clarity_min


def _reaches(freq: float, lo: int, hi: int, df: float) -> bool:
    """Whether ``freq`` lies less than one frequency bin from the own-peak window's bins ``[lo, hi)``:
    closer than that, a tone that does not fall on a bin leaks straight into the window."""
    return lo - 1 < freq / df < hi


def _drive_verdict(ratio, free_min: float, captured_max: float):
    """"free-running" at an own-peak ratio of ``free_min`` or more, "captured" at ``captured_max`` or
    less, "in between" otherwise, and None when the ratio was not measured. The ratio arrives as a
    float or None, so no comparison here ever meets a NaN -- which compares false against both
    thresholds and would read as in between."""
    if ratio is None:
        return None
    if ratio >= free_min:
        return "free-running"
    if ratio <= captured_max:
        return "captured"
    return "in between"


def _drive_picks(rows: list) -> tuple:
    """The strongest free-running row and the weakest captured one, each None when there is none.

    The strongest free-running strength is the last of the unbroken run of free-running strengths from
    the weakest up -- never the strongest free-running strength anywhere in the grid, so a strength
    whose own peak refills at a very large drive can never be picked. The weakest captured strength is
    the first captured one."""
    strongest = None
    for row in rows:
        if row["verdict"] != "free-running":
            break
        strongest = row
    weakest = next((row for row in rows if row["verdict"] == "captured"), None)
    return strongest, weakest


def _forcing_lines(suggested: dict) -> str:
    """The suggested Forcing section as the report's one information record."""
    lines = ["[drive] suggested Forcing section for a cell file, in the cell's own units -- logged and "
             "recorded here, never written to any file:"]
    for label, key in (("free-running", "free_running"), ("captured", "captured")):
        row = suggested[key]
        if row is not None:
            lines.append(f"  {label}: " + ", ".join(f"{k} = {v:.10g}" for k, v in row.items()))
    if suggested["box"] is not None:
        lines.append("  and for its bounds file: " + ", ".join(
            f"{k} in ({lo:.10g}, {hi:.10g})" for k, (lo, hi) in suggested["box"].items()))
    return "\n".join(lines)


def _drive_figure(sink, strengths, ratios, free_min: float, captured_max: float) -> None:
    """The own-peak ratio against the drive strength on a log axis, with the two thresholds drawn; a
    strength not measured leaves a gap."""
    from matplotlib import pyplot as plt
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.set_xscale("log")
    ax.plot(strengths, [np.nan if r is None else r for r in ratios], "o-", color="tab:blue")
    ax.axhline(free_min, ls="--", color="tab:green", label=f"free-running at or above {free_min:g}")
    ax.axhline(captured_max, ls="--", color="tab:red", label=f"captured at or below {captured_max:g}")
    ax.set_xlim(strengths[0] / 1.5, strengths[-1] * 1.5)
    ax.set_xlabel("drive strength (model units)")
    ax.set_ylabel("own-peak power, driven / undriven")
    ax.legend(fontsize=8)
    title = "Probe drive own-peak ratio"
    ax.set_title(title)
    fig.tight_layout()
    sink(title, fig)


@public_entry
def probe_drive(cfg, *, t_obs_s: float = 5.0, repeats: int = 16, detune: float = 1.4, strengths=None,
                free_min: float = 0.70, captured_max: float = 0.10, peak_window: float = 0.02,
                clarity_min: float = 3.0, seed: int = 0, name: str = "", note: str = "", fig_sink=None,
                store=None):
    """How hard can a lab drive this cell?

    An ensemble of ``repeats`` noisy runs of the cell, ``t_obs_s`` seconds long, is simulated undriven:
    its own peak frequency (the ensemble spectrum's highest bin above zero frequency), each run's own
    peak, the peak's clarity (that bin's power over the median power), the peak-to-peak and the cycles
    in the recording. The cell is then driven, a fresh ensemble per strength, by a cosine at ``detune``
    times the peak frequency whose amplitude is the strength times the cell's force scale. Per strength:
    the median |chi| of the lock-in at the drive, the median phase-locking value (reported, never
    judged), and the own-peak ratio -- the power the driven runs keep within ``peak_window`` of the
    peak, over the undriven power there. A strength is free-running at a ratio of ``free_min`` or more,
    captured at ``captured_max`` or less, in between otherwise, and not judged when the ratio is not
    finite.

    The strongest free-running strength is the largest of the unbroken run of free-running strengths
    from the weakest up; the weakest captured strength is the first captured one. Both are reported in
    model units and in the cell's force unit, with suggested Forcing lines and a bounds box for a cell
    file, logged and recorded, never written. Three things are known only after the undriven spend, and
    each is a result -- recorded with every verdict null and one warning, and nothing driven: no clear
    oscillation (a clarity below ``clarity_min``), a drive at or above 0.9 x Nyquist, and a drive less
    than one frequency bin from the own-peak window, which is never narrower than two bins either side
    and so can reach past the detune on a short recording.

    Three more are warned and recorded while the strengths are still judged. The clarity pure noise
    reaches with this many runs and samples is computed before the spend and recorded, and a
    ``clarity_min`` that does not clear it is warned: noise passing it would be judged as a cell. Fewer
    than 30 cycles of the peak in the recording put the drive few bins from the window, where an
    off-bin drive's leakage can hide capture. And a drive whose harmonic, second to fifth, lands within
    a bin of the window -- a sub-harmonic detune -- lets a nonlinear cell's harmonic refill it. Measures
    only: see the module docstring.

    :param t_obs_s: the recording length in seconds, long enough for at least one sample and no longer
                    than the pre-simulated time grid holds.
    :param repeats: noise runs per ensemble, at least 2.
    :param detune: the drive frequency as a multiple of the peak frequency; it must differ from 1 by more
                   than ``peak_window``.
    :param strengths: non-dimensional drive strengths, increasing; None = sixteen from 0.01 to 20 plus
                      the configured chi drive, which is always measured explicitly.
    :param free_min: the own-peak ratio that counts as free-running, in (0, 1].
    :param captured_max: the own-peak ratio that counts as captured, in (0, 1) and below ``free_min``.
    :param peak_window: the own-peak window's half-width as a fraction of the peak frequency, in (0, 1).
    :param clarity_min: the smallest peak clarity that counts as an oscillation, above 0.
    :param seed: the whole run is seeded with it, and it is recorded.
    """
    from core.SBI import chi, derived
    from core.SBI.statistics import FEATURE_LABELS, SummaryStatistics
    store = resolve_store(store)
    store.assert_name_free("diagnostic", name)
    if not cfg.has_forcing:
        raise Refusal(
            f"The drive check states each drive strength as a force in the cell's own unit and suggests a "
            f"Forcing section for the cell file, so it needs a bounds file that declares a Forcing section; "
            f"this config is {cfg.observation_mode}: its bounds file declares none. A cell loaded against a "
            f"box with a Forcing section states Forcing values of its own, which the check never reads -- "
            f"it drives the cell itself.", field=None)
    _ = cfg.ground_truth                      # refuses a cell-free config, before anything is spent
    dtype, device = cfg.hw.dtype, cfg.hw.device
    # the cell's simulation block in the config's own dtype, exactly as the band check builds it
    res = torch.tensor([[v for v, _ in cfg.rescale_params.values()]], dtype=dtype, device=device)
    res_sim, sim_idx = derived.for_simulation(cfg, cfg.params_tensor, res)
    t_scale = float(res_sim[0, sim_idx["t_scale"]])

    seed = require_seed(seed, key="probe_seed")
    t_obs_s = require_positive("drive_t_obs", t_obs_s)
    geom = probe_math.recording_geometry(cfg, t_obs_s, t_scale)
    if geom.n_obs < 1:
        refuse("drive_t_obs", f"{_what('drive_t_obs')} must be long enough for at least one sample; "
                              f"{t_obs_s:g} s gives none.")
    n_grid = cfg.t.shape[0]
    if geom.n_fine > n_grid:
        # refused, never clipped: a clipped run would record one length and measure another
        fit = (n_grid - cfg.steady_idx) // geom.subsample
        longest = math.floor(fit * cfg.dt_exp / cfg.get_unit_conversion_factor("s") * 1000) / 1000
        refuse("drive_t_obs", f"{_what('drive_t_obs')} must fit the pre-simulated time grid, which holds at "
                              f"most {longest:.3f} s for this cell; got {t_obs_s:g} s.")
    repeats = require_at_least("drive_repeats", repeats, 2)
    peak_window = require_between("drive_peak_window", peak_window, 0, 1, open_lo=True, open_hi=True)
    detune = require_positive("drive_detune", detune)
    if not abs(detune - 1.0) > peak_window:
        refuse("drive_detune", f"{_what('drive_detune')} must differ from 1 by more than the own-peak window "
                               f"({peak_window:g}), so that the window measures the cell's own rhythm and "
                               f"not the drive; got {detune:g}.")
    strengths = _positive_list("drive_strengths", strengths)
    if strengths is not None and any(b <= a for a, b in zip(strengths, strengths[1:])):
        refuse("drive_strengths", f"{_what('drive_strengths')} must increase, each above the one before; "
                                  f"got {', '.join(f'{s:g}' for s in strengths)}.")
    free_min = require_between("free_min", free_min, 0, 1, open_lo=True)
    captured_max = require_between("captured_max", captured_max, 0, 1, open_lo=True, open_hi=True)
    if not captured_max < free_min:
        refuse("captured_max", f"{_what('captured_max')} must be below {describe('free_min')} "
                               f"({free_min:g}); got {captured_max:g}.")
    clarity_min = require_positive("clarity_min", clarity_min)

    if strengths is None:
        # the configured chi drive is measured at its own value, never read off a neighbouring strength
        strengths = sorted(set(_DRIVE_STRENGTHS) | {float(cfg.chi_f0)})
    configured = _configured(cfg)
    settings = {"t_obs_s": t_obs_s, "repeats": repeats, "detune": detune, "strengths": strengths,
                "free_min": free_min, "captured_max": captured_max, "peak_window": peak_window,
                "clarity_min": clarity_min, "seed": seed, "drive_phase": DRIVE_PHASE,
                "configured": configured}
    dt = float(cfg.dt_exp)
    per_s = cfg.get_unit_conversion_factor("s")          # cell time units per second
    nd = cfg.params_tensor.expand(repeats, -1).contiguous()
    rs = res_sim.expand(repeats, -1).contiguous()
    inits = cfg.inits_tensor.expand(repeats, -1).contiguous()
    f_eff = _force_scale(rs, sim_idx)
    force_scale = float(f_eff[0])                        # one cell: every run's is the same
    plv_column = FEATURE_LABELS.index("G6_plv")
    n_obs = geom.n_obs
    # what pure noise of this size scores, known before the spend: the spectrum's bins above zero
    # frequency are the samples' half
    noise = {"median": orch._num(probe_math.noise_clarity(repeats, n_obs // 2, 0.5)),
             "p99": orch._num(probe_math.noise_clarity(repeats, n_obs // 2, 0.99))}
    notes = list(DRIVE_NOTES)

    with store.create("diagnostic", cfg, name=name, note=note) as w:
        log.info(f"[drive] driving the cell at {_count(len(strengths), 'strength', 'strengths')} from "
                 f"{strengths[0]:g} to {strengths[-1]:g}, {repeats} runs each over {n_obs * dt / per_s:g} s "
                 f"({n_obs} samples), beside the configured chi drive {configured['chi_f0']:g}")
        rows = [{"strength": a, "cell_force": orch._num(a * force_scale), "chi_median": None,
                 "plv_median": None, "own_peak_ratio": None, "verdict": None} for a in strengths]
        with seeded(seed, device):
            x0 = _simulate(cfg, geom, nd, rs, sim_idx, inits).double()
            u = _undriven(x0, dt)
            omega0, clarity = u.omega0, orch._num(u.clarity)
            f_drive = detune * omega0                    # NaN when the peak was not measured
            oscillating = _oscillating(clarity, clarity_min)
            omega0_hz = orch._num(omega0 * per_s)
            cycles = orch._num(omega0 * n_obs * dt)
            log.info(f"[drive] undriven: the ensemble's own peak is at {_fmt(omega0_hz, '.3f')} Hz; per run, "
                     f"median {u.peak_median * per_s:.3f} Hz, spread {u.peak_spread * per_s:.3f} Hz; clarity "
                     f"{_fmt(clarity, '.3g')} (peak over median power; pure noise of this size scores "
                     f"{_fmt(noise['median'], '.3g')} typically, {_fmt(noise['p99'], '.3g')} in 1 of 100 "
                     f"ensembles); peak-to-peak {u.peak_to_peak:.4g}; {_fmt(cycles, '.4g')} cycles in the "
                     f"recording")
            if noise["p99"] is not None and clarity_min <= noise["p99"]:
                log.warning(f"[drive] the clarity threshold {clarity_min:g} does not clear the clarity pure "
                            f"noise reaches with {repeats} runs of {n_obs} samples -- {noise['median']:.3g} "
                            f"typically, {noise['p99']:.3g} in 1 of 100 noise ensembles -- so passing it "
                            f"does not tell an oscillation from noise; more runs or a higher threshold "
                            f"does. This ensemble's clarity is {_fmt(clarity, '.3g')}.")
            not_judged, harmonic = None, None
            if not oscillating:
                said = ("could not be measured" if clarity is None
                        else f"is {clarity:.3g}, below the {clarity_min:g} that counts as an oscillation")
                not_judged = f"no clear oscillation: the undriven peak's clarity {said}"
            elif f_drive >= _NYQUIST_SHARE * (0.5 / dt):
                not_judged = (f"the drive, {detune:g} x the peak frequency = {f_drive * per_s:.3f} Hz, is at "
                              f"or above 0.9 x Nyquist ({_NYQUIST_SHARE * 0.5 / dt * per_s:.3f} Hz at this "
                              f"sampling rate); a smaller detune brings it below")
            else:
                lo, hi = probe_math.own_peak_window(n_obs, omega0, peak_window, dt)
                df = 1.0 / (n_obs * dt)
                if _reaches(f_drive, lo, hi, df):
                    not_judged = (f"the drive at {f_drive * per_s:.3f} Hz is less than one frequency bin "
                                  f"from the own-peak window ({lo * df * per_s:.3f} to "
                                  f"{(hi - 1) * df * per_s:.3f} Hz over this recording, never narrower than "
                                  f"two bins either side), so the window would read the drive; a longer "
                                  f"recording or a detune further from 1 moves it clear")
                else:
                    harmonic = next((k for k in _HARMONIC_ORDERS if _reaches(k * f_drive, lo, hi, df)), None)
            if not_judged is not None:
                log.warning(f"[drive] {not_judged}: nothing was driven, and no strength is judged")
            else:
                if cycles < _FEW_CYCLES:
                    log.warning(f"[drive] only {cycles:.3g} cycles of the peak fit in the recording, fewer "
                                f"than {_FEW_CYCLES:g}: the drive lies {abs(detune - 1.0) * cycles:.3g} "
                                f"frequency bins from the peak, and unless it falls on a bin its leakage "
                                f"into the own-peak window can make a captured cell read in between or "
                                f"free-running; a longer recording avoids it")
                    notes.append(FEW_CYCLES_NOTE)
                if harmonic is not None:
                    log.warning(f"[drive] the drive's harmonic {harmonic}, at "
                                f"{harmonic * f_drive * per_s:.3f} Hz, lands within a bin of the own-peak "
                                f"window: a nonlinear cell's response "
                                f"there can refill the window and hide capture; a detune away from 1/2, 1/3, "
                                f"1/4 and 1/5 avoids it")
                log.info(f"[drive] driving at {detune:g} x the peak frequency, {f_drive * per_s:.3f} Hz, "
                         f"phase pi/2, each strength times the force scale {force_scale:.4g}")
                for row in rows:
                    amp = row["strength"] * f_eff
                    freq = torch.full((repeats,), f_drive, dtype=torch.float64, device=device)
                    xf = _simulate(cfg, geom, nd, rs, sim_idx, inits, amp_dim=amp, freq=freq).double()
                    lock = chi.lock_in_batched(xf, 2.0 * math.pi * freq, amp, n_obs * dt, dt)
                    stats = SummaryStatistics(x0, xf, dt, amp, freq, DRIVE_PHASE).compute_statistics()
                    # the statistic turns a non-finite value into 0, which would read as measured
                    plv = torch.where(torch.isfinite(xf).all(dim=-1), stats[:, plv_column].double(),
                                      torch.full((repeats,), math.nan, dtype=torch.float64, device=device))
                    ratio = orch._num(probe_math.own_peak_ratio(xf, x0, float(omega0), peak_window, dt))
                    row.update(chi_median=orch._num(torch.quantile(lock.abs(), 0.5)),
                               plv_median=orch._num(torch.quantile(plv, 0.5)), own_peak_ratio=ratio,
                               verdict=_drive_verdict(ratio, free_min, captured_max))
                    log.info(f"[drive] strength {row['strength']:g} ({_fmt(row['cell_force'], '.4g')} in the "
                             f"cell's force unit): median |chi| {_fmt(row['chi_median'], '.3g')}, phase "
                             f"locking {_fmt(row['plv_median'], '.2f')}, own-peak ratio "
                             f"{_fmt(ratio, '.3f')}, {row['verdict'] or 'not measured'}")

        strongest, weakest = _drive_picks(rows)
        if not_judged is None:
            unmeasured = [row["strength"] for row in rows if row["verdict"] is None]
            if unmeasured:
                log.warning(f"[drive] the driven runs gave a non-finite own-peak ratio at "
                            f"{_count(len(unmeasured), 'strength', 'strengths')} of {len(rows)}, not judged: "
                            + ", ".join(f"{a:g}" for a in unmeasured))
            if strongest is None:
                first = rows[0]
                state = ("was not measured" if first["verdict"] is None else
                         f"is already {first['verdict']}; weaker strengths may find where the cell still "
                         f"runs free")
                log.warning(f"[drive] no free-running strength is named: the weakest strength in the grid, "
                            f"{first['strength']:g}, {state}")
            if weakest is None:
                last = rows[-1]
                state = ("was not measured" if last["own_peak_ratio"] is None else
                         f"leaves an own-peak ratio of {last['own_peak_ratio']:.3f}; stronger strengths may "
                         f"find where it is captured")
                log.warning(f"[drive] nothing in the grid captured the cell (captured is an own-peak ratio "
                            f"of {captured_max:g} or less): the strongest strength, {last['strength']:g}, "
                            f"{state}")
        for label, row in (("strongest free-running", strongest), ("weakest captured", weakest)):
            if row is not None:
                a = row["strength"]
                log.info(f"[drive] {label} strength {a:g} ({a * force_scale:.4g} in the cell's force unit)")

        chi_row = next((row for row in rows if math.isclose(row["strength"], configured["chi_f0"],
                                                             rel_tol=_REL_TOL)), None)
        chi_f0_row = None if chi_row is None else {
            "strength": chi_row["strength"], "own_peak_ratio": chi_row["own_peak_ratio"],
            "verdict": chi_row["verdict"], "note": CHI_F0_NOTE}
        if chi_f0_row is None:
            log.info(f"[drive] the configured chi drive {configured['chi_f0']:g} is not among the strengths "
                     f"measured")
        else:
            log.info(f"[drive] the configured chi drive {chi_f0_row['strength']:g}: own-peak ratio "
                     f"{_fmt(chi_f0_row['own_peak_ratio'], '.3f')}, {chi_f0_row['verdict'] or 'not measured'}"
                     f" -- {CHI_F0_NOTE}")

        def forcing(row):
            if row is None:
                return None
            return {"amp": orch._num(row["strength"] * force_scale), "freq": orch._num(f_drive), "phase": 0.0,
                    "offset": 0.0}

        suggested = {"free_running": forcing(strongest), "captured": forcing(weakest), "box": None}
        if weakest is not None:
            suggested["box"] = {"amp": [0.0, orch._num(3.0 * weakest["strength"] * force_scale)],
                                "freq": [orch._num(omega0 / 10.0), orch._num(omega0 * 10.0)],
                                "phase": list(_BOX_PHASE), "offset": list(_BOX_OFFSET)}
        if strongest is None and weakest is None:
            log.info("[drive] no Forcing section is suggested: no strength was judged free-running or "
                     "captured")
        else:
            log.info(_forcing_lines(suggested))
        for text in notes:
            log.info(f"[drive] note: {text}")

        results = {
            "omega0": {"cell_units": orch._num(omega0), "hz": omega0_hz},
            "peak_per_trace_hz": {"median": orch._num(u.peak_median * per_s),
                                  "spread": orch._num(u.peak_spread * per_s)},
            "clarity": clarity, "noise_clarity": noise, "oscillating": oscillating,
            "peak_to_peak": orch._num(u.peak_to_peak), "cycles_in_recording": cycles, "n_obs": n_obs,
            # the lowest harmonic of the drive that lands within a bin of the own-peak window; None when
            # none does, and when the run stopped short of checking (no oscillation, Nyquist, or the
            # drive itself within a bin of the window)
            "drive_frequency": {"cell_units": orch._num(f_drive), "hz": orch._num(f_drive * per_s),
                                "detune": detune, "harmonic": harmonic},
            "force_scale": orch._num(force_scale), "strengths": rows,
            "strongest_free_running": None if strongest is None else {
                "strength": strongest["strength"], "cell_force": strongest["cell_force"]},
            "weakest_captured": None if weakest is None else {
                "strength": weakest["strength"], "cell_force": weakest["cell_force"]},
            "chi_f0_row": chi_f0_row, "suggested_forcing": suggested, "notes": notes}

        def column(key):
            return np.asarray([np.nan if row[key] is None else row[key] for row in rows], dtype=float)

        _drive_figure(w.fig_sink(fig_sink), strengths, [row["own_peak_ratio"] for row in rows], free_min,
                      captured_max)
        file_manager.atomic_savez(w.payload("probe_drive.npz"), {
            "strengths": np.asarray(strengths, dtype=float), "own_peak_ratio": column("own_peak_ratio"),
            "chi_median": column("chi_median"), "plv_median": column("plv_median"),
            "freqs": u.freqs.cpu().numpy(), "power": u.power.cpu().numpy()})
        w.parents = {}
        w.config.update(settings)
        w.body = {"diagnostic": "probes", "variant": "drive", "settings": settings, "results": results}
    return store.load_diagnostic(w.id)
