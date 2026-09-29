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
        found non-finite. It checks its count against the generator's own masked-probe warning, row
        range by row range. No simulation cache is written.

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
    "A probe whose low harmonic (up to the fifth) lands inside the own-peak window is marked harmonic: "
    "harmonic power there inflates the not-captured measure and can hide capture.",
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
    # The force scale the drive amplitude is built on, exactly as the chi probe loop takes it: the
    # simulator block's f_scale, else the Hopf-style x_scale / t_scale.
    if "f_scale" in sim_idx:
        f_eff = rs[:, sim_idx["f_scale"]].double()
    else:
        f_eff = (rs[:, sim_idx["x_scale"]] / rs[:, sim_idx["t_scale"]]).double()
    harmonic = probe_math.harmonic_flags(multipliers, crit.peak_window)
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
        length_rows.append({
            "length_s": length, "achieved_s": achieved, "n_obs": geom.n_obs, "n_fine": geom.n_fine,
            "beyond_ceiling": beyond, "clipped": clipped, "omega0_cell": orch._num(omega0),
            "omega0_hz": orch._num(omega0 * per_s), "peak_median_hz": orch._num(median_hz),
            "peak_spread_hz": orch._num(spread_hz)})
        log.info(f"[band] {length:g} s ({geom.n_obs} samples): the ensemble's own peak is at "
                 f"{omega0 * per_s:.3f} Hz; per run, median {median_hz:.3f} Hz, spread {spread_hz:.3f} Hz")
        for j, m in enumerate(multipliers):
            freq = m * f_peak
            ok = torch.isfinite(freq) & (freq > 0) & (freq < _NYQUIST_SHARE * nyq)
            n_valid = int(ok.sum())
            for drive in drives:
                base = {"length_s": length, "multiplier": m, "drive": drive, "n_valid": n_valid,
                        "nyquist_masked": repeats - n_valid, "harmonic": harmonic[j]}
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


def _frequencies(points, n_l, n_m, n_d, multipliers, harmonic, band):
    """Per probe frequency: it passes only if it passes at every length and every drive, full length
    and capped judged apart, with a reason for each point that does not."""
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
        out.append({"multiplier": m, "in_band": _inside(m, *band), "harmonic": harmonic[j],
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
    capture (the driven runs' power within ``peak_window`` of the undriven ensemble's peak, over the
    undriven power there, at least ``sup_min``). The cycle floor is reported per point, never judged.

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
        harmonic = probe_math.harmonic_flags(multipliers, peak_window)
        for p in points:
            log.info(f"[band] {_point_line(p)}")
        masked = sorted({(p["length_s"], p["multiplier"]) for p in points
                         if p["full"]["verdict"] == "masked"})
        if masked:
            log.warning("[band] masked at the sampling limit -- fewer than two runs put the probe below "
                        "0.9 x Nyquist, so it was reported and not driven:\n"
                        + "\n".join(f"  {length:g} s, x{m:g}" for length, m in masked))

        frequencies = _frequencies(points, n_l, n_m, n_d, multipliers, harmonic, band)
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
PROBE_LAYOUT = ("the probe count, placement and duration draw come from training's own fixed "
                "probe-generator seed, so they do not change with the seed")

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
    batches: dict             # batch tag -> [masked, total], in the order first seen
    ranges: list              # per record: (batch tag, lo, hi, masked, total)
    f_peak: np.ndarray        # (rows,) each row's own peak frequency, cell units
    span: np.ndarray          # (rows with a valid probe,) max / min of the frequencies driven
    multipliers: np.ndarray   # (valid probes,) each driven frequency over its row's peak


def _tally(cfg, records) -> _Tally:
    """Count every probe of every record by what happened to it.

    A probe's driven frequency is ``f_peak * exp(u)``, in float64. Not finite or not positive, or at
    or above 0.9 x Nyquist, it was masked before any lock-in (causes that cannot fire in training).
    Otherwise a probe the generator marked invalid failed the cycle floor: too slow even at the
    band's top over the full recording when its ROW's peak times the full length times the band's top
    falls short of the floor, else shortened below it by the duration draw. The packer then drops
    valid probes outside the band, and those whose lock-in is non-finite or zero (a non-finite lock-in,
    here); ``packed_mask`` is in slot order, so it is compared with ``valid`` only through per-row
    counts of its first ``k`` columns."""
    from core import config
    from core.SBI import chi
    top = float(cfg.chi_freq_bounds[1])
    u_mid, u_half = chi.band_norm(tuple(float(v) for v in cfg.chi_freq_bounds))
    n = dict.fromkeys(("probes", "live", "masked", "floor", "too_slow", "packer", "out_of_band",
                       "non_finite_frequency", "at_or_above_nyquist", "rows", "zero_live", "one_live",
                       "span_rows", "single_probe_rows"), 0)
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
        out_of_band = valid & torch.isfinite(u) & (((u - u_mid) / u_half).abs() > config.CHI_UHAT_MAX)
        live, n_valid = packed.sum(1), valid.sum(1)
        masked, total = int((~packed).sum()), (hi - lo) * k
        for key, value in (("probes", total), ("live", live.sum()), ("masked", masked),
                           ("floor", floor.sum()), ("too_slow", (floor & too_slow).sum()),
                           ("packer", (n_valid - live).clamp(min=0).sum()),
                           ("out_of_band", out_of_band.sum()), ("non_finite_frequency", bad.sum()),
                           ("at_or_above_nyquist", nyq.sum()), ("rows", hi - lo),
                           ("zero_live", (live == 0).sum()), ("one_live", (live == 1).sum()),
                           ("span_rows", (n_valid > 0).sum()), ("single_probe_rows", (n_valid == 1).sum())):
            n[key] += int(value)
        batch = batches.setdefault(rec.batch_tag, [0, 0])
        batch[0] += masked
        batch[1] += total
        ranges.append((rec.batch_tag, lo, hi, masked, total))
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


def _cross_check(ranges: list, warned: list) -> dict:
    """The audit's masked count of each committed row range against the training generator's own
    warning for it, tag by tag.

    The committed ranges of a tag that masked anything, in range order, must equal the LAST that many
    warnings captured under that tag: the whole-batch retry and the row halving both re-run work that
    already warned, so earlier warnings of a tag -- and every warning of a tag with no masked record --
    are an abandoned attempt's, counted and set aside. Any other difference is a mismatch; a range
    left with no warning to match reads production None."""
    expected, captured = {}, {}
    for tag, lo, hi, masked, total in ranges:
        expected.setdefault(tag, [])
        if masked:
            expected[tag].append((lo, hi, masked, total))
    for tag, masked, total in warned:
        captured.setdefault(tag, []).append((masked, total))
    mismatches, abandoned = [], 0
    for tag in [*expected, *(t for t in captured if t not in expected)]:
        want, got = expected.get(tag, []), captured.get(tag, [])
        spare = len(got) - len(want)
        abandoned += max(spare, 0)
        got = got[spare:] if spare >= 0 else [None] * -spare + got
        for (lo, hi, masked, total), seen in zip(want, got):
            if seen != (masked, total):
                mismatches.append({"batch_tag": tag, "lo": lo, "hi": hi, "audit": masked,
                                   "production": None if seen is None else seen[0]})
    return {"row_ranges": len(ranges), "mismatches": mismatches, "abandoned_attempt_warnings": abandoned}


def _audit(cfg, records, warned) -> dict:
    """The mask record's results, from the committed probe records and the masked-probe warnings
    captured beside them. Pure: nothing is logged, written or changed; every float goes through
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
    fractions = [orch._num(m / t) if t else None for m, t in tally.batches.values()]
    finite = [f for f in fractions if f is not None]
    per_s = cfg.get_unit_conversion_factor("s")                  # cell time units per second
    cell = _quantiles(tally.f_peak, _OMEGA0_QUANTILES)
    first = max(_LEAF_CAUSES, key=lambda c: counts[c])          # a tie goes to the earlier cause
    return {
        "probes": probes, "live": n["live"], "masked": n["masked"], "masked_fraction": share(n["masked"]),
        "causes": {c: {"count": v, "share": share(v)} for c, v in counts.items()},
        "invariants": invariants,
        "per_batch": {"fractions": fractions,
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
        "cross_check": _cross_check(tally.ranges, warned),
    }


def _of(count: int, total: int) -> str:
    return f"{count:,} ({100.0 * count / total:.1f}%)" if total else f"{count:,}"


def _joined(values, spec: str, unit: str = "") -> str:
    names = {0.05: "p5", 0.25: "p25", 0.5: "median", 0.75: "p75", 0.95: "p95"}
    return ", ".join(f"{names[q]} {_fmt(v, spec)}{unit}" for q, v in values)


def _log_mask(res: dict) -> None:
    """The mask report: information records, and one warning record per finding that needs acting on
    -- a cause that cannot fire in training firing, a count the generator's own warning disagrees with."""
    probes, c = res["probes"], res["causes"]
    log.info(f"[mask] {_of(res['masked'], probes)} of {probes:,} probes masked, {res['live']:,} live; "
             f"by cause, each a share of every probe:")
    log.info(f"[mask]   the cycle floor: {_of(c['cycle_floor']['count'], probes)}")
    log.info(f"[mask]     too slow even at the band's top over the full recording: "
             f"{_of(c['too_slow_at_band_top']['count'], probes)}")
    log.info(f"[mask]     shortened below the floor by the duration draw: "
             f"{_of(c['shortened_by_duration_draw']['count'], probes)}")
    log.info(f"[mask]   a non-finite lock-in, dropped by the packer: "
             f"{_of(c['non_finite_lock_in']['count'], probes)}")
    cause = res["dominant_cause"]
    if cause is None:
        log.info("[mask] reading: no probe was masked by any of the three causes")
    else:
        what, lever = _READINGS[cause]
        log.info(f"[mask] reading: {what} dominates -- {lever}; this audit changes nothing")
    om = res["omega0"]
    log.info(f"[mask] each row's own peak frequency: "
             f"{_joined(zip(om['quantiles'], om['hz']), '.3g', ' Hz')}")
    pb = res["per_batch"]
    mean = "--" if pb["mean"] is None else f"{100 * pb['mean']:.1f}%"
    sd = "undefined for one batch" if pb["sd"] is None else f"{100 * pb['sd']:.1f}%"
    log.info(f"[mask] masked per batch: mean {mean}, sd {sd}, over {pb['n_batches']} batches -- the "
             f"batch count is the effective sample size")
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
    check = res["cross_check"]
    set_aside = check["abandoned_attempt_warnings"]
    aside = (f"; {set_aside:,} warning{'s' if set_aside != 1 else ''} from an abandoned attempt set aside"
             if set_aside else "")
    if check["mismatches"]:
        log.warning(f"[mask] the audit's masked count does not match the training generator's own "
                    f"warning for {len(check['mismatches'])} of {check['row_ranges']} row ranges{aside}:\n"
                    + "\n".join(f"  {m['batch_tag']}, rows {m['lo']}-{m['hi']}: the audit counts "
                                f"{m['audit']}, the generator "
                                + ("warned nothing" if m["production"] is None else f"{m['production']}")
                                for m in check["mismatches"]))
    else:
        log.info(f"[mask] cross-check: the training generator's own warnings confirm every row range's "
                 f"masked count ({_count(check['row_ranges'], 'row range', 'row ranges')}){aside}")


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
    non-finite lock-in the packer dropped, and the three causes that cannot fire in training are
    checked as invariants. The count of each row range is compared with the generator's own
    masked-probe warning for it, and a disagreement is recorded and warned, never accepted silently.
    Nothing is written but the diagnostic record: no simulation cache.

    :param prior: the LoadedPrior audited, its parent in the record; loaded, never built.
    :param num_runs: training batches to audit, at least 1. The batch count is the per-batch masked
                     fraction's effective sample size.
    :param run_size: rows per batch, at least 1.
    :param chi_k_fixed: audit one probe count, from 2 to the number of probe slots; None audits
                        training's own mixture of counts.
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
    seed = require_seed(seed, key="probe_seed")
    num_runs = require_at_least("mask_num_runs", num_runs, 1)
    run_size = require_at_least("mask_run_size", run_size, 1)
    if chi_k_fixed is not None:
        # both ends before the spend: the generator's own range check fires only inside the first batch
        chi_k_fixed = require_at_least("chi_k_fixed", chi_k_fixed, config.CHI_K_MIN_TRAIN)
        if chi_k_fixed > cfg.chi_k_pad:
            refuse("chi_k_fixed", f"{_what('chi_k_fixed')} must be at most the number of chi probe slots "
                                  f"({cfg.chi_k_pad}); got {chi_k_fixed}.")

    configured = _configured(cfg)
    settings = {"num_runs": num_runs, "run_size": run_size, "chi_k_fixed": chi_k_fixed, "seed": seed,
                "configured": configured, "probe_layout": PROBE_LAYOUT}
    stratum = "pooled over the training mixture" if chi_k_fixed is None else f"fixed at {chi_k_fixed}"
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
        results = _audit(cfg, records, warned)
        _log_mask(results)
        tally = _tally(cfg, records)
        file_manager.atomic_savez(w.payload("probe_mask.npz"), {
            "f_peak": tally.f_peak,
            "batch_masked": np.asarray([m for m, _ in tally.batches.values()], dtype=np.int64),
            "batch_total": np.asarray([t for _, t in tally.batches.values()], dtype=np.int64),
            "span": tally.span})
        w.parents = {"prior": prior.id}
        w.fingerprints["gmm"] = prior.fingerprint
        w.config.update(settings)
        w.body = {"diagnostic": "probes", "variant": "mask", "settings": settings, "results": results}
    return store.load_diagnostic(w.id)
