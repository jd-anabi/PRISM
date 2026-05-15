"""
Top-level FDT analysis pipeline.

Two campaigns:
  Campaign 1: spontaneous fluctuations -> Welch PSD G(omega)
  Campaign 2: forced response at each driving frequency -> chi(omega) via lock-in

Computes T_eff(omega)/T = N * beta * omega * G(omega) / (4 * chi''(omega))
(one-sided PSD convention). At equilibrium this ratio is 1; deviations near
resonance quantify FDT violation (activity of the hair bundle).
"""
import math
from datetime import datetime

import torch

from core.config import FDTConfig, PLOT_PATH
from core.FDT.campaigns import run_campaign1_psd, run_campaign2_chi
from core.FDT.spectral import gen_freqs_log, eff_temp_ratio, find_spectral_peak
from core.FDT.sanity import run_all_sanity, _interp_log
from core.FDT.plots import plot_eff_temp_ratio, plot_chi_components, plot_psd


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


def run_fdt(cfg: FDTConfig) -> None:
    """End-to-end FDT analysis. Runs sanity checks first; gates on user
    confirmation before the production sweep."""
    # 1. Model-specific natural-frequency starting estimate; the production omega_0
    #    is refined from the Campaign 1 PSD peak below.
    cfg.omega_0, omega_0_desc = _estimate_omega_0(cfg)
    print(f"Cell file natural-frequency estimate: omega_0 ~= {omega_0_desc}")

    # 2. Sanity checks (optional skip)
    skip_ans = input("Skip sanity checks? (y/N): ").strip().lower()
    if skip_ans in ("y", "yes"):
        print("Skipping sanity checks.")
    else:
        results = run_all_sanity(cfg)
        if not all(passed for passed, _ in results.values()):
            print("WARNING: one or more sanity checks failed (see metrics above).")
        ans = input("Proceed to production sweep? (y/N): ").strip().lower()
        if ans not in ("y", "yes"):
            print("Aborted by user.")
            return

    # 3. Frequency grid
    omegas = gen_freqs_log(cfg.omega_0, cfg.n_freqs, cfg.freq_bounds,
                            cfg.hw.device, cfg.hw.dtype)

    # 4. Campaign 1: spontaneous PSD
    print("\nCampaign 1: spontaneous fluctuations -> PSD")
    freqs_psd, G = run_campaign1_psd(cfg)

    # Spontaneous-oscillation frequency from the PSD peak (model-agnostic).
    # Search across the full production grid range, since active feedback can shift
    # the actual resonance far from the linearized estimate (a tighter bracket like
    # (omega_0/3, omega_0*3) silently clips peaks that lie below the linear estimate).
    omega_natural = find_spectral_peak(
        freqs_psd, G,
        search_band=(cfg.omega_0 * cfg.freq_bounds[0],
                      cfg.omega_0 * cfg.freq_bounds[1]),
    )
    print(f"Spontaneous-oscillation frequency from PSD peak: {omega_natural:.4f} (ND)")

    # 5. Campaign 2: forced chi via lock-in
    print("\nCampaign 2: forced response -> chi via lock-in")
    chis = run_campaign2_chi(cfg, omegas)

    # 6. Interpolate Welch G onto the chi frequency grid (log-omega, linear-y)
    G_at_omegas = _interp_log(omegas, freqs_psd, G)

    # 7. T_eff/T
    n = cfg.params_dict["n"][0]
    beta = cfg.params_dict["beta"][0]
    ratio = eff_temp_ratio(G_at_omegas, chis.imag, omegas.to(torch.float64), n, beta)

    # 8. Plot + save
    PLOT_PATH.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ratio_path = PLOT_PATH / f"fdt_ratio_{timestamp}.png"
    chi_path = PLOT_PATH / f"chi_components_{timestamp}.png"
    psd_path = PLOT_PATH / f"psd_{timestamp}.png"

    plot_psd(freqs_psd.cpu().numpy(), G.cpu().numpy(),
              save_path=psd_path,
              title=f"Spontaneous PSD (Campaign 1): ND {cfg.model}",
              omega_natural=omega_natural,
              plot_band=(cfg.omega_0 * cfg.freq_bounds[0],
                          cfg.omega_0 * cfg.freq_bounds[1]))
    plot_eff_temp_ratio(omegas.cpu().numpy(), ratio.cpu().numpy(),
                        save_path=ratio_path,
                        title=f"FDT violation: ND {cfg.model} (cell file defaults)",
                        omega_natural=omega_natural)
    plot_chi_components(omegas.cpu().numpy(), chis.cpu().numpy(),
                        save_path=chi_path,
                        title=fr"Susceptibility components: ND {cfg.model}",
                        omega_natural=omega_natural)
    print(f"\nSaved plots to:\n  {psd_path}\n  {ratio_path}\n  {chi_path}")
