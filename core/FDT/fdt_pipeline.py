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
import warnings
from datetime import datetime

import torch

from core import config
from core.config import FDTConfig
from core.FDT.campaigns import run_campaign1_psd, run_campaign2_chi, observable_noise_prefactor
from core.FDT.spectral import gen_freqs_log, eff_temp_ratio, find_spectral_peak
from core.FDT.sanity import run_all_sanity, _interp_log, _resolved_span, _low_end_advice
from core.FDT.plots import (
    plot_eff_temp_ratio, plot_chi_components, plot_psd,
    plot_spontaneous_trajectory,
)
# Both torch-free and stdlib-only: the judgement channel and the run boundary cost this module nothing,
# where core.orchestrator (which re-exports the same PreflightWarning) would load the SBI stack.
from core.refusals import PreflightWarning, Refusal
from core.runs import RUN_BOUNDARY_FILES

# Banners and saved-plot paths are information; a failed sanity verdict is a warning (piece 3, V4).
log = logging.getLogger(__name__)


def _out_dir():
    """Where FDT saves its plots: <artifacts root>/fdt, created on demand (piece 5 wraps FDT in the
    store; until then this is a plain directory)."""
    d = config.artifacts_root() / "fdt"
    d.mkdir(parents=True, exist_ok=True)
    return d


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


def run_fdt(cfg: FDTConfig, *, skip_sanity: bool, confirm_production: bool) -> None:
    """End-to-end FDT analysis. Runs sanity checks first; gates on the caller's answer before the
    production sweep.

    :param skip_sanity: skip the sanity checks. REQUIRED: there is no prompt to fall back to (D1
                        retired the CLI), and the old None default meant an input() that a GUI worker
                        thread could never answer.
    :param confirm_production: proceed to the production sweep after sanity. REQUIRED, for the same
                        reason; only consulted when the sanity checks run."""
    # The per-model normalisation prefactor FIRST, before anything is simulated. It reads the cell's
    # parameters and nothing else, and it used to sit at step 8 -- so a cell missing `n` or `beta`
    # was refused only after BOTH campaigns had been spent (spec §3.4). Carried to step 8 below. It
    # also precedes the thin-setting notices: a run that is refused must not first print a warning
    # about how far to trust its result.
    prefactor = observable_noise_prefactor(cfg)

    # 0. The settings too thin to trust: not a refusal (E5 keeps the quick look possible), a
    #    judgement the operator sees now and the record keeps afterwards (Task 17 stores it).
    notices = warn_thin_settings(cfg)

    # 1. Model-specific natural-frequency starting estimate; the production omega_0
    #    is refined from the Campaign 1 PSD peak below.
    cfg.omega_0, omega_0_desc = _estimate_omega_0(cfg)
    log.info(f"Cell file natural-frequency estimate: omega_0 ~= {omega_0_desc}")

    # Single plot dir + timestamp for all outputs from this run (incl. sanity plots).
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 2. Sanity checks (optional skip). Both booleans are supplied by the caller -- the FDT panel's two
    #    checkboxes, or the `fdt` subcommand's flags. Nothing here prompts.
    if skip_sanity:
        log.info("Skipping sanity checks.")
    else:
        passive_plot_path = _out_dir() / f"fdt_ratio_passive_{timestamp}.png"
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
    # (timestamp set at the top of run_fdt.)
    traj_path = _out_dir() / f"spontaneous_trajectory_{timestamp}.png"
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
    psd_path = _out_dir() / f"psd_{timestamp}.png"
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
    # (spec §3.5). `blanks` and `of` are what T17 records as body.offgrid. Keyed "freq_bounds" like
    # the band refusal above (F37): the same subject, and no front end offers a fix sentence for it.
    # The warning says "no value", not "outside the band": a spectrum non-finite in some bins blanks
    # probes INSIDE the band too, and the sentence has to be true of those.
    blanks, of = int(torch.isnan(G_at_omegas).sum()), G_at_omegas.numel()
    if blanks == of:
        raise Refusal(_nothing_measurable(cfg, of, omega_lo, lo_res, hi_res), field="freq_bounds")
    if blanks:
        log.warning(f"{blanks}/{of} probe frequencies have no value in the spontaneous spectrum, whose "
                    f"resolved band is {lo_res:g}..{hi_res:g} (ND), and are blank in the ratio.")

    # 7. Campaign 2: forced chi via lock-in
    log.info("Campaign 2: forced response -> chi via lock-in")
    chis = run_campaign2_chi(cfg, omegas)

    # 8. T_eff/T -- the per-model normalization prefactor (Nadrowski n*beta, else 1/D_x) was
    #    resolved at the top of this function, before anything was spent.
    ratio = eff_temp_ratio(G_at_omegas, chis.imag, omegas.to(torch.float64), prefactor)

    # 9. Plot + save (timestamp set at the top of run_fdt; the PSD went to disk before Campaign 2)
    ratio_path = _out_dir() / f"fdt_ratio_{timestamp}.png"
    chi_path = _out_dir() / f"chi_components_{timestamp}.png"

    plot_eff_temp_ratio(omegas.cpu().numpy(), ratio.cpu().numpy(),
                        save_path=ratio_path,
                        title=f"FDT violation: ND {cfg.model} (cell file defaults)",
                        omega_natural=omega_natural)
    plot_chi_components(omegas.cpu().numpy(), chis.cpu().numpy(),
                        save_path=chi_path,
                        title=fr"Susceptibility components: ND {cfg.model}",
                        omega_natural=omega_natural)
    log.info(f"Saved plots to:\n  {ratio_path}\n  {chi_path}")
