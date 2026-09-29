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

MEASURING IS NOT OVERRIDING. A check takes its own recording lengths, frequencies, drive strengths and
lock-in ceilings, and they reach the simulator only as the drive builder's frequency and amplitude,
inside this module. None of them passes through a configuration field, through the training
generator's band and drive keywords, or through an assignment to a configuration constant, so a
training run in the same process uses exactly what it would have used without them. Every record
states the configured band, drive, pad, cycle floor, cycle ceiling and probe count it judged; a
deliberate change to any of them is made in core/config.py.

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

    return (f"{where} {column('full')}  {column('capped')}  own-peak share {_fmt(p['sup'], '.3f')}"
            + ("  (harmonic)" if p["harmonic"] else ""))


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
    from core import config
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
    configured = {"chi_f0": float(cfg.chi_f0), "chi_freq_bounds": list(band),
                  "chi_k_pad": int(cfg.chi_k_pad), "chi_min_cycles": float(config.CHI_MIN_CYCLES),
                  "chi_max_cycles": configured_cap, "probe_count": int(cfg.chi_n_freqs)}
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
