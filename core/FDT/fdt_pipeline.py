"""
Top-level FDT analysis pipeline.

Two campaigns:
  Campaign 1: spontaneous fluctuations -> Welch PSD G(omega)
  Campaign 2: forced response at each driving frequency -> chi(omega) via lock-in

Computes T_eff(omega)/T = N * beta * omega * G(omega) / (4 * chi''(omega))
(one-sided PSD convention). At equilibrium this ratio is 1; deviations near
resonance quantify FDT violation (activity of the hair bundle).
"""
import logging
import math
import random
import warnings

import h5py
import numpy as np
import torch

from core import config
from core.config import FDTConfig
from core.FDT.campaigns import run_campaign1_psd, run_campaign2_chi, observable_noise_prefactor
from core.FDT.spectral import gen_freqs_log, eff_temp_ratio, find_spectral_peak
from core.FDT.sanity import (run_all_sanity, _interp_log, _resolved_span, _low_end_advice,
                             _runs_nadrowski_only_checks)
from core.FDT.plots import (
    plot_eff_temp_ratio, plot_chi_components, plot_psd,
    plot_spontaneous_trajectory,
)
# All three stdlib-only at import: the judgement channel, the seeding context and the run boundary
# cost this module nothing, where core.orchestrator (which re-exports the same PreflightWarning) or
# core.diagnostics (the seeding context's first home) would load the SBI stack.
from core.refusals import PreflightWarning, Refusal
from core.rng import seeded
from core.runs import RUN_BOUNDARY_FILES, public_entry

# Banners and saved-plot paths are information; a failed sanity verdict is a warning (piece 3, V4).
log = logging.getLogger(__name__)


def _resolve_seed(seed, cfg) -> int:
    """The seed this run used (E7): the one given, else the one the config was built with, else one
    DRAWN and recorded. Drawn from Python's own ``random``, which is the one global stream neither
    the solver nor the spectrum reads -- ``torch.seed()`` would reseed every CUDA device as a side
    effect of being asked a question (the hazard core/SBI/decorrelate.py documents)."""
    if seed is not None:
        return int(seed)
    if getattr(cfg, "seed", None) is not None:
        return int(cfg.seed)
    return random.randrange(2 ** 31)


def _settings_block(cfg, *, skip_sanity=None, confirm_production=None) -> dict:
    """Every knob the run resolved, for ``body.settings`` (spec §2.3). The five that no front end
    exposes -- freq_bounds, burn_in_nd, T_obs_periods, dt_nd, psd_T_obs_nd -- are recorded here even
    though no control and no flag names them, because without them a number is not reproducible.
    ``skip_sanity`` and ``confirm_production`` are run_fdt's ARGUMENTS, not FDTConfig fields, so the
    caller hands them in; a sweep passes neither and records both null."""
    return {"n_freqs": int(cfg.n_freqs), "ensemble_M": int(cfg.ensemble_M),
            "freqs_per_batch": int(cfg.freqs_per_batch), "F0": float(cfg.F0),
            "freq_bounds": [float(v) for v in cfg.freq_bounds],
            "burn_in_nd": float(cfg.burn_in_nd), "T_obs_periods": int(cfg.T_obs_periods),
            "dt_nd": float(cfg.dt_nd), "psd_T_obs_nd": float(cfg.psd_T_obs_nd),
            "skip_sanity": None if skip_sanity is None else bool(skip_sanity),
            "confirm_production": None if confirm_production is None else bool(confirm_production)}


def _results_block(omegas, ratio, omega_natural: float, blanks: int) -> dict:
    """The short summary the listing and the detail pane show. FINITE NUMBERS ONLY: manifest.validate
    refuses a non-finite float anywhere in the body, and T_eff/T legitimately carries NaN wherever the
    spectrum came back blank or chi'' crossed zero -- so a summary that would be NaN is recorded as
    null (spec §2.3)."""
    r = ratio.detach().cpu().to(torch.float64)
    w = omegas.detach().cpu().to(torch.float64)
    usable = torch.isfinite(r)

    def _f(v):
        v = float(v)
        return v if math.isfinite(v) else None

    if not bool(usable.any()):
        return {"peak_ratio": None, "peak_omega": None, "ratio_at_resonance": None,
                "usable_fraction": 0.0, "offgrid_blanks": int(blanks)}
    idx = int(torch.argmax(torch.where(usable, r, torch.full_like(r, -math.inf))))
    at_res = int(torch.argmin(torch.abs(w - float(omega_natural))))
    return {"peak_ratio": _f(r[idx]), "peak_omega": _f(w[idx]),
            "ratio_at_resonance": _f(r[at_res]),
            "usable_fraction": float(usable.sum()) / float(r.numel()),
            "offgrid_blanks": int(blanks)}


def _write_single_h5(path, cfg, omegas, ratio, chis, freqs_psd, G, omega_natural, prefactor) -> None:
    """The single-cell layout of ``data.h5`` (spec §2.3, §3.8).

    ``study`` sits at the ROOT so a reader can check the layout before reading a dataset: a sweep's
    file and a comparison's file carry the same name inside their own records. The dataset names are
    ``cross_validation._fdt_measure``'s and the sweep file's own vocabulary -- ``omega_grid``,
    ``T_eff_over_T``, ``chi_prime``, ``chi_double_prime``, ``PSD_omegas``, ``PSD_G``, all float64 --
    so the comparison (T36, spec §7) reads one set of names whichever study wrote them (P5, P71). The
    spontaneous spectrum keeps its OWN frequency axis -- it is a Welch grid, not the log-spaced chi
    grid, and interpolating one onto the other is exactly the step §3.5's off-grid fix made honest.
    """
    def _f64(t):
        return t.detach().cpu().numpy().astype(np.float64)

    with h5py.File(path, "w") as h5:
        h5.attrs["study"] = "single"
        h5.attrs["model"] = cfg.model
        h5.attrs["omega_0"] = float(omega_natural)
        h5.attrs["omega_0_source"] = "spectrum peak"
        h5.attrs["prefactor"] = float(prefactor)
        h5.attrs["n_freqs"] = int(cfg.n_freqs)
        h5.attrs["freq_bounds"] = np.asarray(cfg.freq_bounds, dtype=np.float64)
        h5.create_dataset("omega_grid", data=_f64(omegas), compression="gzip")
        h5.create_dataset("T_eff_over_T", data=_f64(ratio), compression="gzip")
        h5.create_dataset("chi_prime", data=_f64(chis.real), compression="gzip")
        h5.create_dataset("chi_double_prime", data=_f64(chis.imag), compression="gzip")
        h5.create_dataset("PSD_omegas", data=_f64(freqs_psd), compression="gzip")
        h5.create_dataset("PSD_G", data=_f64(G), compression="gzip")


def _estimate_omega_0(cfg: FDTConfig) -> tuple[float, str]:
    """
    Model-specific starting estimate for the natural angular frequency. Used to
    set up the production frequency grid and bracket the PSD peak search; the
    actual peak comes from Campaign 1 (find_spectral_peak), so this only needs
    to be in the right ballpark.

    :return: (omega_0_estimate, description)
    """
    model = cfg.model.lower()
    if model == "nadrowski":
        k = cfg.params_dict["k"][0]
        return math.sqrt(1.0 + k), f"sqrt(1 + k) = {math.sqrt(1.0 + k):.4f} (linearized bundle stiffness)"
    if model == "hopf":
        # ND Hopf normal form has unit natural frequency by construction
        return 1.0, "1.0 (ND Hopf natural frequency)"
    if model == "bp":
        # BP model: no simple analytical form; use 1.0 as a generic ND default
        return 1.0, "1.0 (generic ND default; refine from PSD peak)"
    return 1.0, "1.0 (fallback default)"


def thin_notices(cfg: FDTConfig) -> list:
    """The "too thin to trust" sentences for this run's settings; ``[]`` when there are none (E5).

    Pure, and shared by the single-cell run and the sweep, so both mark a quick look the same way
    and ``body.notices`` carries the same words the operator was shown. The thresholds are
    ``config.FDT_THIN_*`` and are read LIVE, never from-imported, so a session can move one.
    """
    notices = []
    if int(cfg.n_freqs) < config.FDT_THIN_N_FREQS:
        notices.append(
            f"The frequency grid has {int(cfg.n_freqs)} point"
            f"{'' if int(cfg.n_freqs) == 1 else 's'}, below the {config.FDT_THIN_N_FREQS} this "
            f"measurement is trusted at: read the result as a quick look, not as a measurement.")
    if int(cfg.ensemble_M) < config.FDT_THIN_ENSEMBLE_M:
        notices.append(
            f"The ensemble is {int(cfg.ensemble_M)} trajector"
            f"{'y' if int(cfg.ensemble_M) == 1 else 'ies'}, below the "
            f"{config.FDT_THIN_ENSEMBLE_M} this measurement is trusted at: read the result as a "
            f"quick look, not as a measurement.")
    return notices


def warn_thin_settings(cfg: FDTConfig) -> list:
    """Raise one ``PreflightWarning`` per thin setting and return the sentences for ``body.notices``.

    The warning is what the operator sees while the run is going (the window's pane at warning
    severity, the tool's stderr) and what the run buffer copies into ``log.txt``; the returned list
    is what the record keeps, which a warning alone cannot do. ``stacklevel=3`` names whoever called
    the STAGE, as the orchestrator's ``_preflight_warn`` does: frame 1 is this helper, frame 2 the
    stage (``run_fdt`` or ``run_fdt_param_sweep``), frame 3 its caller, with a ``@public_entry``
    wrapper skipped when counting (``RUN_BOUNDARY_FILES``). The stage's own call line would tell the
    operator nothing.
    """
    notices = thin_notices(cfg)
    for sentence in notices:
        warnings.warn(sentence, PreflightWarning, stacklevel=3,
                      skip_file_prefixes=RUN_BOUNDARY_FILES)
    return notices


def _nothing_measurable(cfg, n_probes: int, omega_lo: float, lo_res: float, hi_res: float) -> str:
    """The refusal's words when every probe frequency came back blank (spec §3.6, E4).

    Every clause must be TRUE of the spectrum the run measured (the ruling after Task 14's review,
    carried to every sentence about the resolution). By the time this is asked, the band check has
    refused any grid reaching below the spectrum's first real bin, so a probe that blanks lies ABOVE
    the spectrum's top -- and since the probes ascend, every probe blanks exactly when the lowest one
    does. The resonance the grid is built around is itself a bin of the spectrum, so this happens only
    when freq_bounds' lower multiplier exceeds top/omega_0, which is at least 1. Lowering the UPPER
    multiplier alone would then bring no probe back, so the advice names the lower one first, with the
    floor the band check enforces. The top is the Nyquist frequency pi/dt_nd, which the recording
    length does not move, so a longer recording is named only to say it would not help: the one thing
    a longer recording can lower is the BOTTOM, which is not what failed here (``_low_end_advice``
    words that case).

    The explanation is conditional on the lowest probe really lying above the top. A spectrum with no
    finite value at all never gets here -- run_fdt refuses it earlier, as a diverged simulation -- but
    one non-finite in the bins around every probe does, with its probes INSIDE the resolved band,
    and there the Nyquist sentence would be false; the first sentence alone is true in both cases.
    """
    what = "the one probe frequency" if n_probes == 1 else f"any of the {n_probes} probe frequencies"
    msg = (f"The spontaneous spectrum has no value at {what}, so the effective-temperature ratio is "
           f"unmeasurable here; the band it resolves is {lo_res:g}..{hi_res:g} (ND).")
    if omega_lo > hi_res:
        top = hi_res / cfg.omega_0
        msg += (f" The lowest probe frequency, {omega_lo:g} (ND), is above that band's top, which is "
                f"the spectrum's Nyquist frequency pi/dt_nd (dt_nd = {cfg.dt_nd:g}): only a shorter "
                f"integration step raises it. Lower freq_bounds' lower multiplier into "
                f"{lo_res / cfg.omega_0:g}..{top:g} for any probe to be measured, and its upper "
                f"multiplier below {top:g} for all of them. Lengthening the spontaneous recording "
                f"(psd_T_obs_nd = {cfg.psd_T_obs_nd:g}) would not help.")
    return msg


@public_entry
def run_fdt(cfg: FDTConfig, *, skip_sanity: bool, confirm_production: bool,
            writer: "ArtifactWriter", seed: "int | None" = None) -> "LoadedFdt":
    """End-to-end FDT analysis, written into one ``fdt`` record. Runs sanity checks first; gates on
    the caller's answer before the production sweep.

    :param skip_sanity: skip the sanity checks. REQUIRED: there is no prompt to fall back to (D1
                        retired the CLI), and the old None default meant an input() that a GUI worker
                        thread could never answer.
    :param confirm_production: proceed to the production sweep after sanity. REQUIRED, for the same
                        reason; only consulted when the sanity checks run.
    :param writer: the OPEN-BUT-NOT-ENTERED ArtifactWriter the front end minted with
                        ``store.create("fdt", cfg, ...)``. THIS function enters it, and that is not a
                        detail: ``log.txt`` is written from ``runs.current_run_log()``, which is
                        thread-local and only populated inside ``capture_run()`` on the worker thread
                        -- a writer entered on the window's own thread would write no log at all
                        (spec §1.2). The front end needs ``writer.dir`` before it dispatches, to point
                        the figure watcher at the record's ``figures/``; hence the split.
    :param seed: the seed, or None to draw one and record it (E7).
    :returns: the LoadedFdt for the record just written.

    The decorator hands the body ``cfg.copy_for_run()``, so the two ``cfg.omega_0 = ...`` writes below
    land on a private copy and the caller's settings object is exactly what it was, whether this
    returned, refused or crashed (V1).
    """
    # The per-model normalisation prefactor FIRST, before anything is simulated -- and before the
    # writer is entered, so a cell FDT cannot normalise opens no record at all. It reads the cell's
    # parameters and nothing else, and it used to sit at step 8 -- so a cell missing `n` or `beta`
    # was refused only after BOTH campaigns had been spent (spec §3.4). Carried to step 8 of
    # _measure. It also precedes the thin-setting notices (ruling F10): a run that is refused must
    # not first print a warning about how far to trust its result.
    prefactor = observable_noise_prefactor(cfg)

    seed = _resolve_seed(seed, cfg)
    cfg.seed = seed                                  # on the PRIVATE copy; recorded in the body below
    # 0. The settings too thin to trust (T12, E5): warned now and KEPT in body.notices (P51). HERE, in
    #    the decorated function itself: T12's pin reads inspect.getsource(fdt_pipeline.run_fdt), which
    #    is this function's source (public_entry uses functools.wraps), not _measure's.
    notices = warn_thin_settings(cfg)
    # The first body is UPDATED IN PLACE, never replaced (P15): a front end may already have set the
    # study and the notices (T25's panel does). The stage owns `settings` and the RESOLVED seed.
    body = writer.body
    body.setdefault("study", "single")
    body["settings"] = _settings_block(cfg, skip_sanity=skip_sanity,
                                       confirm_production=confirm_production)
    body["seed"] = seed
    # The config block too, and before the writer is entered so the FIRST manifest carries it:
    # store.create computed that block from the caller's object before any seed was resolved, so it
    # holds the builder's seed (or none), and `artifacts show` would print that beside a body that
    # names the one this run used (Task 17, fix round 1).
    writer.config.update({"seed": seed})
    body["notices"] = [*(body.get("notices") or []), *notices]
    for key in ("grid", "points", "offgrid", "compared", "results"):
        body.setdefault(key, None)
    body["complete"] = False
    with writer:
        with seeded(seed, cfg.hw.device):
            _measure(cfg, skip_sanity=skip_sanity, confirm_production=confirm_production,
                     writer=writer, prefactor=prefactor)
        writer.body["complete"] = True
    return writer.store.load_fdt(writer.id)


def _measure(cfg: FDTConfig, *, skip_sanity: bool, confirm_production: bool, writer,
             prefactor: float) -> None:
    """The measurement itself, inside the entered writer and inside one seeded context.

    ``prefactor`` is the normalisation run_fdt resolved before entering the writer, first read at
    step 8. Each figure path, and ``data.h5``'s, is asked of the writer at the moment its file is
    written, never up front (P49): once one has been handed out the writer keeps a refused record
    (spec §2.2 step 3), so a path asked for early would turn a pre-spend refusal into a kept, empty
    record."""
    # 1. Model-specific natural-frequency starting estimate; the production omega_0
    #    is refined from the Campaign 1 PSD peak below.
    cfg.omega_0, omega_0_desc = _estimate_omega_0(cfg)
    log.info(f"Cell file natural-frequency estimate: omega_0 ~= {omega_0_desc}")

    # 2. Sanity checks (optional skip). Both booleans are supplied by the caller -- the FDT panel's two
    #    checkboxes, or the `fdt` subcommand's flags. Nothing here prompts.
    if skip_sanity:
        log.info("Skipping sanity checks.")
    else:
        # The passive-baseline figure is drawn only where that check runs -- Nadrowski alone, a rule
        # sanity owns -- so its path is asked for only then: a path handed out is a figure the record
        # lists, and a listed figure that is never drawn is a phantom (Task 17, fix round 1).
        passive_plot_path = (writer.figure_path("Passive baseline ratio")
                             if _runs_nadrowski_only_checks(cfg) else None)
        results = run_all_sanity(cfg, passive_plot_path=passive_plot_path)
        if not all(passed for passed, _ in results.values()):
            # The level carries the severity: the hand-typed "WARNING: " word went with the print (on
            # the tool it would have read "warning: WARNING: ...").
            log.warning("One or more sanity checks failed (see metrics above).")
        if not confirm_production:
            log.info("Aborted by user.")
            return

    # 3. Campaign 1 first: spontaneous PSD gives us a data-driven estimate of the
    #    natural oscillation frequency, which is then used to center the Campaign 2
    #    drive-frequency grid. Without this, the grid would sit on the linearized
    #    analytical estimate -- which can be orders of magnitude away from the actual
    #    Hopf-shifted Omega_0 in the active regime.
    log.info("Campaign 1: spontaneous fluctuations -> PSD")
    freqs_psd, G, t_traj, x_mean_traj = run_campaign1_psd(cfg, return_trajectory=True)

    # Save the ensemble-mean unforced trajectory as a diagnostic before moving on.
    traj_path = writer.figure_path("Spontaneous trajectory")
    plot_spontaneous_trajectory(
        t_traj.cpu().numpy(), x_mean_traj.cpu().numpy(),
        save_path=traj_path,
        title=f"Spontaneous trajectory (Campaign 1): ND {cfg.model}",
        burn_in=cfg.burn_in_nd,
    )
    log.info(f"Saved spontaneous trajectory plot to: {traj_path}")

    # A spontaneous simulation that diverged leaves a spectrum with no finite value at any positive
    # frequency, and there is nothing to measure against it. Refused HERE, before the peak search:
    # argmax takes NaN for the largest value, so the search would return the first bin as the
    # resonance, and the band check and the nothing-measurable refusal below would then describe a
    # band that does not exist -- advice that is false and a diagnosis never said (the review of Task
    # 16). After the trajectory figure, which is the picture that shows the divergence, and which the
    # folder keeps (E2). field="cell": neither the band nor the recording length can help; what
    # diverged is the cell's dynamics. A spectrum non-finite in only SOME bins is not this refusal --
    # its blanks are counted below like any other. A grid with no positive bin at all (a one-sample
    # recording) is not a divergence either, and is not called one.
    positive = freqs_psd > 0
    if bool(positive.any()) and not bool(torch.isfinite(G[positive]).any()):
        raise Refusal(
            f"The spontaneous simulation diverged: its spectrum holds no finite value at any positive "
            f"frequency, so there is nothing to measure. Neither the frequency band (freq_bounds) nor "
            f"the recording length (psd_T_obs_nd = {cfg.psd_T_obs_nd:g}) can help; what diverged is "
            f"the cell's own dynamics, integrated at dt_nd = {cfg.dt_nd:g}.",
            field="cell")

    # 4. Find natural frequency from the PSD peak directly (no search band).
    #    The PSD's argmax (skipping the DC bin) is robust because the peak is
    #    orders of magnitude above the noise floor for any active oscillator.
    omega_natural = find_spectral_peak(freqs_psd, G)
    cfg.omega_0 = omega_natural   # use data-driven value for the Campaign 2 grid
    log.info(f"Spontaneous-oscillation frequency from PSD peak: {omega_natural:.4f} (ND)")
    writer.body["grid"] = {"omega_0": float(omega_natural), "omega_0_source": "spectrum peak",
                           "n_freqs": int(cfg.n_freqs),
                           "bounds": [float(v) for v in cfg.freq_bounds],
                           "omega_min": float(cfg.freq_bounds[0] * omega_natural),
                           "omega_max": float(cfg.freq_bounds[1] * omega_natural)}
    writer.refresh()          # the spontaneous half is on disk before the driven campaign is spent

    # 5. Build production grid centered on the data-driven Omega_0.
    #    freq_bounds default (0.1, 30) gives 1 decade below + 1.5 decades above,
    #    so ~50% more drive frequencies above Omega_0 than below.
    omegas = gen_freqs_log(cfg.omega_0, cfg.n_freqs, cfg.freq_bounds,
                            cfg.hw.device, cfg.hw.dtype)

    # The spectrum's own picture goes to disk BEFORE the band check and the driven campaign, because
    # it is what diagnoses both refusals that can follow -- a band reaching below what the spectrum
    # resolves (just below) and nothing measurable (below that, still before the drive) -- and E2
    # keeps the folder it is written into (spec §3.6, P22). It used to be written at the very end,
    # where neither refusal could ever reach it. plot_psd draws the whole spectrum with the band
    # shaded when the band holds none of it, which is exactly when those refusals fire.
    psd_path = writer.figure_path("Spontaneous PSD")
    plot_psd(freqs_psd.cpu().numpy(), G.cpu().numpy(),
              save_path=psd_path,
              title=f"Spontaneous PSD (Campaign 1): ND {cfg.model}",
              omega_natural=omega_natural,
              plot_band=(cfg.omega_0 * cfg.freq_bounds[0],
                          cfg.omega_0 * cfg.freq_bounds[1]))
    log.info(f"Saved spontaneous PSD plot to: {psd_path}")

    # The band, checked the first moment it is knowable and BEFORE the driven campaign -- the
    # expensive half of the run (spec §3.4). The grid's lowest frequency is known only now, because
    # it is built around the resonance Campaign 1 found; the spectrum's lowest RESOLVED frequency is
    # its first non-zero bin, which the spectrum's Welch segment sets -- not the recording, which
    # stops lengthening the segment at campaigns.WELCH_NPERSEG_CAP samples (the ruling after Task
    # 14's review; _low_end_advice words it). A grid reaching below it comes back blank there (spec
    # §3.5), and used to come back with a fabricated tail instead.
    # field="freq_bounds" (P2, P75): the key is registered and BOTH front-end tables map it to None,
    # because neither the band nor the spontaneous duration is exposed by a front end -- so no table
    # offers a fix sentence and none pretends to.
    lo_res, hi_res = _resolved_span(freqs_psd)
    omega_lo = float(omegas[0])
    if omega_lo < lo_res:
        raise Refusal(
            f"The frequency band reaches below what the spontaneous spectrum resolves: the lowest "
            f"probe frequency is {omega_lo:g} (ND) but the lowest frequency the spectrum resolves "
            f"is {lo_res:g} (ND). Raise freq_bounds' lower multiplier above "
            f"{lo_res / cfg.omega_0:g}{_low_end_advice(cfg, lo_res, 'the spontaneous recording')}",
            field="freq_bounds")

    # 6. Interpolate Welch G onto the chi frequency grid (log-omega, linear-y). BEFORE the drive: it
    #    reads only the grid and Campaign 1's spectrum, so whether anything is measurable is known
    #    now, and Campaign 2 changes neither (the review of Task 16).
    G_at_omegas = _interp_log(omegas, freqs_psd, G)

    # Nothing measurable is a refusal, not an empty picture (spec §3.6, E4). A probe the spontaneous
    # spectrum does not resolve comes back blank; when EVERY probe is blank there is no ratio, and
    # this used to divide blanks by blanks, save a figure with no points on it and report success --
    # after spending the driven campaign, which it is refused before now. The low end is gated
    # above, so what stays reachable here is the UPPER end: the grid tops out at
    # freq_bounds[1]*omega_0 against a spectrum Nyquist of pi/dt_nd, which an active cell can exceed
    # (spec §3.5). `blanks` and `of` are recorded as body.offgrid, which is how the record says what
    # the blanks cost (_interp_log's docstring leaves that to its callers). Keyed "freq_bounds" like
    # the band refusal above (F37): the same subject, and no front end offers a fix sentence for it.
    # The warning says "no value", not "outside the band": a spectrum non-finite in some bins blanks
    # probes INSIDE the band too, and the sentence has to be true of those.
    blanks, of = int(torch.isnan(G_at_omegas).sum()), G_at_omegas.numel()
    writer.body["offgrid"] = {"blanks": blanks, "of": int(of)}   # the Campaign-2 probe grid (P57)
    if blanks == of:
        raise Refusal(_nothing_measurable(cfg, of, omega_lo, lo_res, hi_res), field="freq_bounds")
    if blanks:
        log.warning(f"{blanks}/{of} probe frequencies have no value in the spontaneous spectrum, whose "
                    f"resolved band is {lo_res:g}..{hi_res:g} (ND), and are blank in the ratio.")

    # 7. Campaign 2: forced chi via lock-in
    log.info("Campaign 2: forced response -> chi via lock-in")
    chis = run_campaign2_chi(cfg, omegas)

    # 8. T_eff/T -- the per-model normalization prefactor (Nadrowski n*beta, else 1/D_x) was
    #    resolved by run_fdt before the writer was entered, before anything was spent.
    ratio = eff_temp_ratio(G_at_omegas, chis.imag, omegas.to(torch.float64), prefactor)

    # The numbers (E1, spec §3.8), the moment they all exist and BEFORE the two figures below: they are
    # hours of simulation and the figures minutes of matplotlib, so a figure that fails to draw must
    # not cost them -- E2 keeps the folder, and this puts the numbers in it (Task 18's ruling). Still
    # after both campaigns, never at the top of the run (P49): a path handed out is a payload the
    # record lists, so a run that stops before this line leaves an unfinished record that lists no
    # data.h5.
    _write_single_h5(writer.payload("data.h5"), cfg, omegas, ratio, chis, freqs_psd, G,
                     omega_natural, prefactor)

    # 9. Plot + save (the PSD went to disk before Campaign 2)
    ratio_path = writer.figure_path("Effective temperature ratio")
    chi_path = writer.figure_path("Chi components")

    plot_eff_temp_ratio(omegas.cpu().numpy(), ratio.cpu().numpy(),
                        save_path=ratio_path,
                        title=f"FDT violation: ND {cfg.model} (cell file defaults)",
                        omega_natural=omega_natural)
    plot_chi_components(omegas.cpu().numpy(), chis.cpu().numpy(),
                        save_path=chi_path,
                        title=fr"Susceptibility components: ND {cfg.model}",
                        omega_natural=omega_natural)
    writer.body["results"] = _results_block(omegas, ratio, omega_natural, blanks)
    writer.refresh()
    log.info(f"Saved plots to:\n  {ratio_path}\n  {chi_path}")
