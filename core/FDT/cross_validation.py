"""
FDT parameter-sweep study.

Sweeps a single NWK parameter (others held fixed) and runs the FDT campaigns at
each value to map how the effective-temperature ratio T_eff/T responds. Two
canonical sweeps probe FDT restoration:

  - S sweep (T_a/T = 1 fixed): the calcium feedback is the only non-equilibrium
    source. As S -> 0 calcium decouples and FDT is restored (T_eff/T -> 1).
  - T sweep (S = 0 fixed): calcium is decoupled, so the hot motor (T_a > T) is the
    only non-equilibrium source. As T_a/T -> 1, FDT is restored.

Pure FDT measurement -- no reduction map. Output: one ``fdt`` record per sweep
(spec §4.1), whose data.h5 holds one group per swept value. Downstream plotting
renders T_eff/T vs (omega/omega_0, param). The omega/omega_0 grid is identical
across operating points by construction (Campaign 2's grid is omega_0 x fixed
log-ratios), so rows stack with no interpolation.
"""
from __future__ import annotations
import logging
import warnings
import json
import math
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import torch

from ..config import FDTConfig
# core.rng, never core.diagnostics.rng: importing anything under core.diagnostics runs its __init__,
# which loads the orchestrator, sbi and pytensor -- two seconds and two false "g++" lines on every
# sweep, for a context manager that needs none of them (Task 17, fix round 1).
from ..refusals import Refusal, require_at_least
from ..rng import seeded
from ..runs import public_entry
from .campaigns import run_campaign1_psd, run_campaign2_chi, observable_noise_prefactor
from .spectral import eff_temp_ratio
from .sanity import _interp_log
from .fdt_pipeline import (_estimate_omega_0, _resolve_seed, _results_block, _settings_block,
                           thin_notices, warn_thin_settings)
from .cross_validation_plots import plot_fdt_3d_vs_param

# Phase banners and per-point progress are information; an operating point that failed in either
# campaign and was recorded rather than raised is an error record (piece 3, V4).
log = logging.getLogger(__name__)

# The two canonical sweeps' labels, which used to live inline in run_param_study_cli's two plot calls.
_PARAM_SYMBOL = {"s": r"$S$", "temp": r"$T_a/T$"}
_PARAM_TITLE = {"s": r"FDT ratio vs $(\tilde\omega/\Omega_0,\ S)$  ($T_a/T=1$)",
                "temp": r"FDT ratio vs $(\tilde\omega/\Omega_0,\ T_a/T)$  ($S=0$)"}
# Which front-end setting an all-failed sweep names (spec §4.3, E4). The swept parameter IS the grid
# the operator would change, and each grid has its own control and its own flag.
_GRID_FIELD = {"s": "s_grid", "temp": "t_grid"}
# Each sweep's number in its points' stream derivation (_point_seed). FIXED values, never an
# enumeration order: renumbering a sweep would silently change what every recorded seed reproduces.
_SWEEP_NO = {"s": 0, "temp": 1}


def _detect_resonance(omegas, G, omega_0_lin: float) -> tuple[float, bool]:
    """
    Locate the spontaneous-oscillation frequency robustly.

    Picks the highest interior local maximum of the PSD (via scipy.signal.find_peaks
    with a prominence threshold to ignore noise wiggles), skipping the DC bin. If the
    spectrum is monotonic (near-equilibrium / overdamped -- no resonance), falls back
    to the linearized estimate. This avoids the floor-latching that plain argmax
    suffers at low S / T_a/T=1, where the lowest PSD bin dominates.

    :param omegas: (n,) angular frequencies (torch or numpy).
    :param G: (n,) one-sided PSD (torch or numpy).
    :param omega_0_lin: linearized natural-frequency estimate, used as fallback.
    :return: (omega_0, is_resonant). is_resonant is False when the fallback was used.
    """
    from scipy.signal import find_peaks  # lazy import: avoids OpenMP load-order conflicts

    om = omegas.detach().cpu().numpy() if hasattr(omegas, "detach") else np.asarray(omegas)
    g = G.detach().cpu().numpy() if hasattr(G, "detach") else np.asarray(G)
    om, g = om[1:], g[1:]   # skip DC
    if g.size < 3 or not np.any(np.isfinite(g)):
        return float(omega_0_lin), False

    g_range = float(np.nanmax(g) - np.nanmin(g))
    prominence = max(0.05 * g_range, 1e-30)   # 5% of dynamic range
    peaks, _props = find_peaks(g, prominence=prominence)
    if peaks.size == 0:
        return float(omega_0_lin), False
    best = int(peaks[np.argmax(g[peaks])])
    return float(om[best]), True


def _campaign2_ratio(cfg: FDTConfig, omegas: torch.Tensor,
                     freqs_psd: torch.Tensor, G: torch.Tensor):
    """
    Forced response (Campaign 2) on a SUPPLIED frequency grid, plus T_eff/T using a
    precomputed Campaign-1 PSD. Split out so a sweep can run all operating points on
    one common grid.

    :return: (chis, ratio) -- complex susceptibility and T_eff/T, both on `omegas`.
    """
    chis = run_campaign2_chi(cfg, omegas)
    G_at = _interp_log(omegas, freqs_psd, G)
    prefactor = observable_noise_prefactor(cfg)
    ratio = eff_temp_ratio(G_at, chis.imag, omegas.to(torch.float64), prefactor)
    return chis, ratio


def _build_common_grid(cfg: FDTConfig, omega0s: list[float], is_res: list[bool]
                       ) -> tuple[torch.Tensor, float]:
    """
    Build ONE absolute frequency grid covering every operating point's resonance band,
    plus the single reference omega_0 used to normalize the x-axis.

    The grid spans [freq_bounds[0]*min(omega0), freq_bounds[1]*max(omega0)] over the
    *resonant* operating points (so passive fallbacks don't needlessly widen it). If
    no point is resonant (fully near-equilibrium sweep), the linearized estimates are
    used. Point count scales with the log-span to preserve per-decade density.

    :return: (omegas_common, omega_0_ref). omega_0_ref = max resonant omega_0.
    """
    resonant = [w for w, r in zip(omega0s, is_res) if r]
    basis = resonant if resonant else omega0s
    lo_mult, hi_mult = cfg.freq_bounds
    lo = lo_mult * min(basis)
    hi = hi_mult * max(basis)

    ref_decades = math.log10(hi_mult / lo_mult)
    span_decades = math.log10(hi / lo)
    n = max(cfg.n_freqs, int(round(cfg.n_freqs * span_decades / ref_decades)))

    omegas = torch.exp(torch.linspace(math.log(lo), math.log(hi), n,
                                      device=cfg.hw.device, dtype=cfg.hw.dtype))
    omega_0_ref = float(max(basis))
    return omegas, omega_0_ref


def _point_seed(seed: int, sweep_param: str, phase: int, idx: int) -> int:
    """The seed of one operating point's stream in one phase (0 = Phase A's spontaneous campaign,
    1 = Phase B's driven one), derived from the study's single seed (spec §4.1).

    Through numpy's ``SeedSequence``, which hashes its whole entropy tuple, so every stream of one
    study has a derivation of its own -- two coincide only by a 32-bit hash collision -- and a point
    is reproducible from the seed, its sweep, its phase and its index, nothing else. The arithmetic it
    replaces (``seed + idx`` in Phase A, ``seed + n_points + idx`` in Phase B) gave point k of the S
    sweep and point k of the T sweep the SAME Phase-A stream -- point 0 of both was bit-identical --
    let one sweep's Phase B land on the other's Phase A when the grids differ in length, and made
    Phase B depend on how many points the grid held (Task 19's review). Noise shared across the two
    records is exactly what a comparison of sweeps would read as signal.

    The seed must be non-negative -- ``SeedSequence``'s domain, and the rule both FDT builders and
    ``run_fdt_param_sweep`` apply before anything is spent.
    """
    return int(np.random.SeedSequence([int(seed), _SWEEP_NO[sweep_param], int(phase), int(idx)])
               .generate_state(1)[0])


def _refresh_points(writer, *, planned: int, done: int, failed: int) -> None:
    """Keep ``body.points`` current as the sweep runs, and rewrite the manifest and the log.

    ``done`` counts the operating points whose Campaign 2 has LANDED and ``failed`` those that failed
    in either phase; ``planned - done - failed`` are still to run. Deriving ``done`` as
    ``planned - failed`` would report every not-yet-run point as done for the whole run -- the
    browser row this refresh exists for would read "4 done of 4" after the first point.
    """
    writer.body["points"] = {**writer.body["points"], "planned": planned, "done": done,
                             "failed": failed}
    writer.refresh()


def _check_sweep_param(sweep_param: str) -> None:
    """Refuse a parameter this module cannot sweep, before anything is spent.

    A sweep reads four per-parameter tables, each of them only once it is under way: ``_SWEEP_NO``
    derives every point's streams inside the per-point guard, where a KeyError would be counted as a
    failure at EVERY point and end in a false "measured nothing"; ``_GRID_FIELD`` names that
    refusal's setting; ``_PARAM_SYMBOL`` and ``_PARAM_TITLE`` label the figure drawn after the whole
    spend. A programming error -- no front end sweeps anything else -- so a ValueError, not a Refusal.
    """
    missing = [name for name, table in (("_PARAM_SYMBOL", _PARAM_SYMBOL), ("_PARAM_TITLE", _PARAM_TITLE),
                                        ("_GRID_FIELD", _GRID_FIELD), ("_SWEEP_NO", _SWEEP_NO))
               if sweep_param not in table]
    if missing:
        raise ValueError(f"cannot sweep {sweep_param!r}: it has no entry in {', '.join(missing)} "
                         f"(core/FDT/cross_validation.py). The sweeps this build runs are "
                         f"{sorted(_SWEEP_NO)}.")


def _check_study_call(writers, keys) -> None:
    """Refuse a malformed study call before either sweep spends (Task 19's review).

    Each of these used to surface only when the sweep that needed it got there: a missing writer as
    the T sweep's KeyError and an already-entered one as its FileExistsError, both after the whole S
    sweep had been paid for; one writer handed in for both sweeps as the second ``__enter__`` failing
    on the first's folder; a parameter the plot tables cannot label at the END of its own sweep. All
    are programming errors -- no front end can build one -- so a ValueError, never a Refusal, which
    would send the operator looking for a setting to change. "Entered" is read off the disk: the
    writer creates its folder at ``__enter__`` and refuses one that already exists.

    :param keys: the study's sweep parameters, in study order -- the same tuple its loop runs over.
    """
    got, want = set(writers), set(keys)
    if got != want:
        raise ValueError(f"run_param_study_cli takes one writer per sweep, keyed {sorted(want)}: "
                         f"missing {sorted(want - got)}, unexpected {sorted(got - want, key=str)}.")
    if len({writers[k].dir for k in keys}) != len(keys):
        raise ValueError("run_param_study_cli needs a distinct writer per sweep: one writer handed in "
                         "for two sweeps would have the second enter the first's record.")
    entered = [k for k in keys if writers[k].dir.exists()]
    if entered:
        raise ValueError(f"run_param_study_cli takes writers that are open but NOT entered -- each "
                         f"sweep enters its own -- and the {entered} writer's folder already exists: "
                         f"it was entered already.")
    for k in keys:
        _check_sweep_param(k)


def run_fdt_param_sweep(
    cfg: FDTConfig,
    sweep_param: str,
    sweep_grid: np.ndarray,
    fixed_overrides: dict | None = None,
    *,
    writer: "ArtifactWriter",
) -> "LoadedFdt":
    """
    Sweep one NWK parameter, running FDT at each value, into this sweep's own ``fdt`` record.

    Two-phase so all rows share ONE frequency grid that covers every operating
    point's resonance:
      Phase A: Campaign 1 (spontaneous PSD) per row -> robust omega_0(value).
      Phase B: Campaign 2 on the common grid per row -> chi, T_eff/T.

    The HDF5 is written incrementally (PSD in Phase A, Campaign-2 data added in
    Phase B), so an interrupt still leaves a partially-populated, readable file.

    An operating point that fails in EITHER phase is logged, recorded in its group's ``error``
    attribute and COUNTED, and the sweep goes on (spec §4.3, E4); ``body.points`` is refreshed after
    every point, so the listings follow the counts while the sweep runs. Some points failed is a
    completed record carrying the count. All points failed is a ``Refusal`` naming the grid and the
    cell (field ``s_grid`` or ``t_grid``), raised AFTER the record's final refresh and from inside the
    entered writer, so the folder and the spectra it does hold stay on disk (E2). A finished sweep
    fills ``results`` and ``offgrid`` over the points that landed (P78).

    DELIBERATELY NOT an atomic write, unlike the prior/posterior artifacts, and the reason is that
    the two situations are not alike. An atomic write buys exactly one thing: an existing good file is
    never replaced by a partial one. Here the file is ``writer.payload("data.h5")``, a fresh file inside
    a fresh record, so there is no existing file to protect -- staging through a temp sibling would only
    rename "partial file at X" to "partial file at X.tmp", while destroying the incremental readability
    above, which is a used property: ``load_param_sweep`` marks a row ``failed`` precisely so a run
    interrupted after Phase A still plots, and the all-failed refusal below points the reader at this
    record for the PSDs it does have. The record's directory is new -- the writer's ``__enter__``
    refuses one that exists -- so the ``"w"`` below never truncates an earlier run's file, which is
    the one way the old explicit ``output_path`` could lose data. SINCE PIECE 5 the file lives inside
    the record and E2 is what makes the incremental write safe: an interrupted sweep keeps its folder,
    so the partial file is listed and deletable rather than invisible.

    :param cfg: baseline FDTConfig.
    :param sweep_param: NWK param name to vary, a key in params_dict: "s" or "temp", the two sweeps
                        this module labels, names a field for and derives streams for. Any other is
                        refused by ``_check_sweep_param`` before the writer is entered.
    :param sweep_grid: 1D array of values for sweep_param.
    :param fixed_overrides: other params pinned for the whole sweep
                            (e.g. {"temp": 1.0} for the S sweep, {"s": 0.0} for the T sweep).
    :param writer: the OPEN-BUT-NOT-ENTERED ArtifactWriter for this sweep's own record. THIS function
                   enters it, so __enter__, every refresh() and __exit__ run on the thread whose run
                   log becomes log.txt (spec §1.2, §4.1). Its seed comes from ``cfg.seed``; nothing
                   here writes on ``cfg``.
    :returns: the LoadedFdt for the record just written.
    :raises Refusal: every operating point failed; the unfinished record stays on disk.
    """
    fixed_overrides = fixed_overrides or {}
    _check_sweep_param(sweep_param)                  # a malformed call, before the writer is entered
    # Non-negative, the builders' own rule and _point_seed's domain, refused HERE: inside the per-point
    # guard a negative seed would fail every point and end in a false "measured nothing".
    seed = require_at_least("seed", _resolve_seed(None, cfg), 0)
    # The normalisation BEFORE the writer is entered, as run_fdt does (Task 17): a cell that cannot
    # supply it is then refused with no record opened, where resolving it after data.h5 had been
    # handed out would keep an unfinished record around an empty file. The study checks the same cell
    # once at its top; this is what covers a sweep called on its own.
    prefactor = float(observable_noise_prefactor(cfg))
    # E5: the sentences for body.notices, and NO warning. The study raises it once, at its top, for
    # both of its records: raised here it fired once per sweep, the second time only after the first
    # sweep's whole spend (Task 12's review).
    notices = thin_notices(cfg)
    fixed_str = ", ".join(f"{k}={v}" for k, v in fixed_overrides.items())
    n_points = int(len(sweep_grid))
    # The first body is UPDATED IN PLACE, never replaced (P15): the front end may already have set the
    # study, a seed and notices (T26's panel does). The stage owns settings, points and the RESOLVED
    # seed; the swept parameter is recorded once, in points.param (spec §2.3). The grid is recorded as
    # [min, max, N] (F40); every value it held is in data.h5.
    body = writer.body
    body.setdefault("study", "sweep")
    body["settings"] = {**_settings_block(cfg), "preset": cfg.preset_name,
                        "sweep_grid": [float(sweep_grid[0]), float(sweep_grid[-1]), n_points]}
    body["seed"] = seed
    # The config block too, and before the writer is entered so the FIRST manifest carries it:
    # store.create computed that block from the caller's object before any seed was resolved
    # (Task 17, fix round 1).
    writer.config["seed"] = seed
    body["points"] = {"param": sweep_param, "planned": n_points, "done": 0, "failed": 0}
    body["notices"] = [*(body.get("notices") or []), *notices]
    for key in ("grid", "offgrid", "compared", "results"):
        body.setdefault(key, None)
    body["complete"] = False
    # NESTED, not combined: the HDF5 file must be CLOSED before the sweep's own plot reopens it with
    # load_param_sweep at the end, and that plot must still be drawn inside the writer so its PNG
    # lands in the record's figures/.
    with writer:
        with h5py.File(writer.payload("data.h5"), "w") as h5:
            # The contract's root attributes (P6), on every data.h5 so a reader can check the layout
            # before reading a dataset. omega_0 is the common grid's reference, set once it exists.
            h5.attrs["study"] = "sweep"
            h5.attrs["prefactor"] = prefactor
            h5.attrs["omega_0"] = math.nan
            h5.attrs["timestamp"] = datetime.now().isoformat()
            h5.attrs["model"] = cfg.model
            h5.attrs["sweep_param"] = sweep_param
            h5.attrs["fixed_overrides"] = json.dumps(fixed_overrides)
            h5.attrs["n_freqs"] = cfg.n_freqs
            h5.attrs["ensemble_M"] = cfg.ensemble_M
            h5.attrs["F0"] = cfg.F0
            h5.attrs["psd_T_obs_nd"] = cfg.psd_T_obs_nd
            h5.attrs["n_operating_points"] = int(len(sweep_grid))
            h5.attrs["sweep_grid"] = np.asarray(sweep_grid, dtype=np.float64)
            ops = h5.create_group("operating_points")

            # --- Phase A: spontaneous PSD + robust omega_0 per operating point ---
            log.info(f"--- Phase A ({sweep_param} sweep): spontaneous PSD + omega_0 detection ---")
            cfg_ops, psds, omega0s, is_res, omega0_lins, indices = [], [], [], [], [], []
            n_failed, n_done = 0, 0
            for idx, value in enumerate(sweep_grid):
                log.info(f"  [A {idx+1}/{len(sweep_grid)}] {sweep_param}={value:.6g} ({fixed_str})")
                cfg_op = cfg.with_overrides(**{sweep_param: float(value), **fixed_overrides})
                grp = ops.create_group(f"{idx:03d}")
                grp.attrs["param_value"] = float(value)
                grp.attrs["sweep_param"] = sweep_param
                grp.attrs["failed"] = True     # flipped to False once Campaign 2 lands in Phase B
                try:
                    omega_0_lin, _ = _estimate_omega_0(cfg_op)
                    # One stream per operating point, derived from the study's single seed (spec §4.1): a
                    # point is reproducible from the seed and its index, and seeding once for the whole sweep
                    # would make point k's draw depend on how many points preceded it.
                    with seeded(_point_seed(seed, sweep_param, 0, idx), cfg.hw.device):
                        freqs_psd, G = run_campaign1_psd(cfg_op)
                    w0, res = _detect_resonance(freqs_psd, G, omega_0_lin)
                except Exception as e:         # noqa: BLE001 -- counted, recorded, and the sweep goes on
                    # Phase A caught NOTHING before piece 5: one bad operating point ended the study with
                    # a traceback and lost every point that had already worked. E4 makes it a count.
                    log.error(f"      Campaign 1 FAILED: {e}")
                    grp.attrs["error"] = str(e)
                    n_failed += 1
                    # The file FIRST, then the manifest: a reader the refresh sends to the record finds
                    # the failed point's attributes already there.
                    h5.flush()
                    _refresh_points(writer, planned=n_points, done=n_done, failed=n_failed)
                    continue

                cfg_ops.append(cfg_op); psds.append((freqs_psd, G)); indices.append(idx)
                omega0s.append(w0); is_res.append(res); omega0_lins.append(float(omega_0_lin))

                for k, v in fixed_overrides.items():
                    grp.attrs[f"fixed_{k}"] = float(v)
                grp.attrs["omega_0_resonance"] = float(w0)
                grp.attrs["omega_0_linearized"] = float(omega_0_lin)
                grp.attrs["is_resonant"] = bool(res)
                grp.create_dataset("PSD_omegas", data=freqs_psd.cpu().numpy().astype(np.float64),
                                   compression="gzip")
                grp.create_dataset("PSD_G", data=G.cpu().numpy().astype(np.float64),
                                   compression="gzip")
                tag = "" if res else " [no resonance -> linearized fallback]"
                log.info(f"      omega_0 = {w0:.4f}{tag}")
                h5.flush()

            # --- Common grid covering every row's resonance band ---
            ok_ratios, blanks_total = [], 0      # the landed points' ratios and blanks (P78)
            if not omega0s:
                # Every point failed in Phase A, each one counted there: with no resonance to build it
                # around, _build_common_grid would raise ValueError from min() on an empty sequence.
                # Phase B is skipped and the sweep falls through to the refusal below, which names the
                # setting to change and leaves the record behind.
                omegas_common = omega_0_ref = None
            else:
                omegas_common, omega_0_ref = _build_common_grid(cfg, omega0s, is_res)
                omega_grid_np = omegas_common.cpu().numpy().astype(np.float64)
                omega_norm_np = (omega_grid_np / omega_0_ref).astype(np.float64)
                h5.attrs["omega_0_ref"] = omega_0_ref
                h5.attrs["omega_0"] = omega_0_ref
                h5.attrs["common_grid_n"] = int(omegas_common.shape[0])
                h5.attrs["common_grid_span"] = np.array([omega_grid_np[0], omega_grid_np[-1]],
                                                        dtype=np.float64)
                log.info(f"Common grid: {omegas_common.shape[0]} pts spanning "
                         f"[{omega_grid_np[0]:.4f}, {omega_grid_np[-1]:.4f}] (omega_0_ref={omega_0_ref:.4f})")

            # --- Phase B: forced response on the common grid per operating point ---
            if omegas_common is not None:
                log.info(f"--- Phase B ({sweep_param} sweep): forced response on common grid ---")
                # ``indices`` keeps Phase B's group keys aligned with Phase A's after a Phase-A point
                # dropped out: enumerating the survivors would write point 2's response into point 1's
                # group.
                for cfg_op, (freqs_psd, G), idx in zip(cfg_ops, psds, indices):
                    log.info(f"  [B {idx+1}/{n_points}] {sweep_param}={sweep_grid[idx]:.6g}")
                    grp = ops[f"{idx:03d}"]
                    try:
                        with seeded(_point_seed(seed, sweep_param, 1, idx), cfg.hw.device):
                            chis, ratio = _campaign2_ratio(cfg_op, omegas_common, freqs_psd, G)
                        # The peak line FIRST, before anything about this point is called a success: an
                        # EMPTY ratio raises here (numpy's nanmax has no identity for a zero-size array),
                        # and a group already flipped to failed=False would then be counted a failure
                        # and stored as a success at the same time. An all-blank ratio does not raise; it
                        # lands, and its blanks are counted below.
                        log.info(f"      T_eff/T peak = {np.nanmax(ratio.cpu().numpy()):.3g}")
                        grp.attrs["omega_0_ref"] = omega_0_ref
                        grp.create_dataset("omega_grid", data=omega_grid_np, compression="gzip")
                        grp.create_dataset("omega_norm", data=omega_norm_np, compression="gzip")
                        grp.create_dataset("T_eff_over_T", data=ratio.cpu().numpy().astype(np.float64),
                                           compression="gzip")
                        grp.create_dataset("chi_prime", data=chis.real.cpu().numpy().astype(np.float64),
                                           compression="gzip")
                        grp.create_dataset("chi_double_prime",
                                           data=chis.imag.cpu().numpy().astype(np.float64),
                                           compression="gzip")
                        grp.attrs["failed"] = False
                    except Exception as e:     # noqa: BLE001 -- counted, recorded, and the sweep goes on
                        # Recorded per point, and COUNTED. A systematic failure (a CUDA OOM, say) fails every
                        # point identically, so an overnight sweep could "complete" with nothing in it -- the
                        # per-point note scrolled past hours ago and the summary said nothing.
                        log.error(f"      Campaign 2 FAILED: {e}")
                        grp.attrs["error"] = str(e)
                        n_failed += 1
                        h5.flush()
                        _refresh_points(writer, planned=n_points, done=n_done, failed=n_failed)
                        continue
                    n_done += 1
                    ok_ratios.append(ratio.detach().cpu().to(torch.float64).reshape(-1))
                    blanks_total += int(torch.isnan(ratio).sum())   # its blanks on the common grid (P78)
                    h5.flush()
                    _refresh_points(writer, planned=n_points, done=n_done, failed=n_failed)

            if n_failed:
                warnings.warn(
                    f"{sweep_param} sweep: {n_failed}/{n_points} operating points failed (in either "
                    f"campaign) and carry no response data. See their 'error' attrs in {writer.dir}.",
                    stacklevel=2)
            if n_failed == n_points:
                # AFTER the final refresh, so the spectra this message points at are on disk before the
                # refusal unwinds (spec §4.3) -- and it puts the warning above into the record's log. A
                # Refusal, not a RuntimeError: the command line reports one as an operator line naming
                # the flag, the window as the yellow box naming the grid. Raised inside the entered
                # writer after data.h5 was handed out, so E2 keeps the folder. Every clause is true of
                # what the record holds, so the spectra clause depends on how many first campaigns
                # landed (F41: the cell is named by its file name).
                _refresh_points(writer, planned=n_points, done=n_done, failed=n_failed)
                cell = (getattr(cfg, "sources", None) or {}).get("cell")
                held = (f"The record holds the spontaneous spectra of the {len(omega0s)} points whose "
                        f"first campaign finished" if omega0s else
                        "No point's spontaneous campaign finished either, so the record holds no spectra")
                raise Refusal(
                    f"The {sweep_param} sweep of {Path(cell).name if cell else 'this cell'} measured "
                    f"nothing: all {n_points} operating points failed (each point's reason is in its "
                    f"'error' attribute in the record's data.h5). This is almost always one systematic "
                    f"cause repeating identically at every point, not {n_points} independent failures. "
                    f"{held}; widen or move the grid, or choose a cell whose operating points are "
                    f"reachable.",
                    field=_GRID_FIELD[sweep_param])
            # A FINISHED sweep fills results and offgrid (P78; spec §2.3 "null only until the run
            # finishes"). results: T17's _results_block over every usable point's ratio on the common
            # grid; offgrid: the blanks those points left on it, of every planned point's slot. Reached
            # only when a point landed, so the common grid exists and ok_ratios is not empty.
            writer.body["results"] = _results_block(
                omegas_common.detach().cpu().to(torch.float64).repeat(len(ok_ratios)),
                torch.cat(ok_ratios), omega_0_ref, blanks_total)
            writer.body["offgrid"] = {"blanks": int(blanks_total),
                                      "of": int(n_points * omegas_common.numel())}
            log.info(f"{sweep_param} sweep complete ({n_done}/{n_points} points). "
                     f"Saved to: {writer.dir}")
        # The file is closed; the sweep's own figure is drawn from it, into this record, while the
        # writer is still entered -- each sweep plots itself when it finishes (spec §4.1).
        plot_fdt_3d_vs_param(load_param_sweep(writer.payload("data.h5")),
                             param_symbol=_PARAM_SYMBOL[sweep_param], title=_PARAM_TITLE[sweep_param],
                             save_path=writer.figure_path(f"FDT ratio vs {sweep_param}"))
        writer.body["complete"] = True
        writer.refresh()
    return writer.store.load_fdt(writer.id)


@public_entry
def run_param_study_cli(cfg: FDTConfig, *, s_grid: np.ndarray, t_grid: np.ndarray,
                        writers: dict, seed=None) -> "list[LoadedFdt]":
    """The two canonical sweeps, as TWO records (spec §4.1).

    One record per swept parameter, each complete on its own, because an all-failed S sweep used to
    raise before the T sweep had started -- two measurements held hostage to one. It no longer costs
    the other (P77, spec §4.3): a sweep that measured nothing refuses, its unfinished record stays on
    disk (E2), the study logs that at error and runs the other sweep. One SEED, drawn once when none
    is given and recorded on both, because the study is one experiment: repeating half of it at a
    fresh seed answers a different question.

    Both sweeps run inside ONE captured run -- this public entry's -- and a record's log.txt is that
    run's log as it stood at the record's last write. So the S record's log ends with the S sweep,
    and the T record's holds the WHOLE study, the S sweep's lines included.

    :param writers: {"s": ArtifactWriter, "temp": ArtifactWriter} -- two distinct writers, open and not
                    entered; each sweep enters its own. Anything else is a malformed call, refused
                    with a ValueError before either sweep spends.
    :param seed: overrides ``cfg.seed``; None falls back to it, and None in both draws one (P12).
    :returns: the LoadedFdt of every sweep that finished, S first; a sweep that measured nothing is
              logged and left on disk unfinished, and the study refuses only when both did -- with
              the activity sweep's refusal (field ``s_grid``), after logging both at error (F14/F45).
    """
    sweeps = (("s", s_grid, {"temp": 1.0},
               "# S sweep:  vary S, hold T_a/T = 1   (FDT restored as S -> 0)"),
              ("temp", t_grid, {"s": 0.0},
               "# T sweep:  vary T_a/T, hold S = 0   (FDT restored as T_a/T -> 1)"))
    # FIRST, before the cell is even judged: a programming error must not be reported as, or after,
    # something about the operator's inputs (Task 19's review).
    _check_study_call(writers, [key for key, *_ in sweeps])
    # §3.4's check, applied to the sweep for the same reason: a cell that cannot supply the
    # normalisation constant would otherwise cost the whole first phase before anything noticed.
    observable_noise_prefactor(cfg)
    # E5, ONCE for the whole study and before either sweep spends anything -- and HERE, in the public
    # entry, so stacklevel=3 names the front end's call rather than a line of this module. After the
    # prefactor (ruling F10): a refused study must not first warn about how far to trust its result.
    # Each sweep keeps the same sentences in its own body.notices without warning again.
    warn_thin_settings(cfg)
    cfg.seed = _resolve_seed(seed, cfg)              # on the PRIVATE copy; both sweeps read it

    recs, refused = [], []
    for key, grid, fixed, banner in sweeps:
        log.info("#" * 64)
        log.info(banner)
        log.info("#" * 64)
        try:
            recs.append(run_fdt_param_sweep(cfg, sweep_param=key, sweep_grid=grid,
                                            fixed_overrides=fixed, writer=writers[key]))
        except Refusal as e:
            if e.field != _GRID_FIELD[key]:
                raise                   # not "measured nothing": any other refusal ends the study
            # P77, spec §4.3: a sweep that measured nothing costs ITSELF, never the other one. Its
            # record stays on disk, unfinished (E2). Its own log.txt was written as its writer exited,
            # so this line reaches only a record written after it: the T record's, when the S sweep
            # refused.
            log.error(f"The {key} sweep measured nothing; its unfinished record is kept. {e}")
            refused.append(e)
    if len(refused) == len(sweeps):
        raise refused[0]            # the study measured nothing at all: the first grid's refusal
    return recs                     # the sweeps that FINISHED, in study order


def load_param_sweep(path: Path) -> list[dict]:
    """
    Read a param-sweep HDF5 back into a list of per-operating-point dicts, sorted by
    param_value. Each dict has: param_value, sweep_param, omega_0_resonance,
    omega_0_ref, omega_0_linearized, is_resonant, failed, and (when Campaign 2
    completed) the FDT arrays. A row is `failed` if Campaign 2 didn't finish (e.g.
    interrupted after Phase A) -- detected by a missing T_eff_over_T dataset.
    """
    records: list[dict] = []
    with h5py.File(path, "r") as h5:
        omega_0_ref = float(h5.attrs.get("omega_0_ref", math.nan))
        ops = h5["operating_points"]
        for key in sorted(ops.keys()):
            grp = ops[key]
            complete = ("T_eff_over_T" in grp) and not bool(grp.attrs.get("failed", True))
            rec = {
                "param_value": float(grp.attrs["param_value"]),
                "sweep_param": grp.attrs.get("sweep_param", "?"),
                "failed": not complete,
                "omega_0_resonance": float(grp.attrs.get("omega_0_resonance", math.nan)),
                "omega_0_ref": float(grp.attrs.get("omega_0_ref", omega_0_ref)),
                "omega_0_linearized": float(grp.attrs.get("omega_0_linearized", math.nan)),
                "is_resonant": bool(grp.attrs.get("is_resonant", False)),
            }
            for k in ("omega_grid", "omega_norm", "T_eff_over_T", "chi_prime",
                      "chi_double_prime", "PSD_omegas", "PSD_G"):
                if k in grp:
                    rec[k] = grp[k][...]
            records.append(rec)
    records.sort(key=lambda r: r["param_value"])
    return records
