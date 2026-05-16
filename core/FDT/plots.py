"""Plotting helpers for FDT analysis output."""
from os import PathLike

import numpy as np
from matplotlib import pyplot as plt


def plot_eff_temp_ratio(omegas: np.ndarray, ratio: np.ndarray,
                        save_path: str | PathLike[str] = None,
                        title: str = "FDT violation diagnostic",
                        omega_natural: float = None,
                        linthresh: float = 1.0) -> None:
    """
    Plot T_eff/T vs angular frequency with symlog y-axis so the divergence is
    visible on BOTH sides of zero (chi'' sign change manifests as a +inf -> -inf
    crossing through y=0). The linear region |y| < linthresh keeps the equilibrium
    reference y=1 readable.

    Non-finite values (NaN/inf) are dropped with a warning; negative ratios are
    kept and displayed in the lower half of the symlog axis.

    :param omegas: (n_freqs,) angular frequencies (ND).
    :param ratio:  (n_freqs,) T_eff(omega) / T values.
    :param save_path: optional path to save the figure.
    :param title: plot title.
    :param omega_natural: optional angular frequency at which to draw a vertical
                          marker (e.g. spontaneous-oscillation peak from Campaign 1).
    :param linthresh: half-width of the linear region around y=0 in the symlog y-axis.
                      Defaults to 1.0 (the equilibrium value).
    """
    omegas = np.asarray(omegas)
    ratio = np.asarray(ratio)

    n_bad = int((~np.isfinite(ratio)).sum())
    if n_bad > 0:
        print(f"plot_eff_temp_ratio: dropping {n_bad}/{len(ratio)} non-finite (NaN/inf) points.")
        mask = np.isfinite(ratio)
        omegas = omegas[mask]
        ratio = ratio[mask]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(omegas, ratio, marker='o', linestyle='none',
            markersize=4, color='steelblue')
    ax.set_xscale('log')
    ax.set_yscale('symlog', linthresh=linthresh)
    ax.axhline(1.0, color='k', linestyle='--', linewidth=0.8,
               label=r'$T_{\rm eff}/T = 1$ (equilibrium)')
    ax.axhline(0.0, color='gray', linestyle=':', linewidth=0.6,
                label=r'$T_{\rm eff}/T = 0$ ($\chi''=0$ crossing)')
    if omega_natural is not None:
        ax.axvline(omega_natural, color='darkorange', linestyle=':', linewidth=1.2,
                    label=fr'$\Omega_0 \approx {omega_natural:.3f}$ (PSD peak)')
    ax.set_xlabel(r'$\tilde\omega$ (ND), log scale')
    ax.set_ylabel(r'$T_{\rm eff}(\tilde\omega) / T$, symlog scale')
    ax.set_title(title)
    ax.grid(False)
    ax.legend()
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_spontaneous_trajectory(t: np.ndarray, x_mean: np.ndarray,
                                  save_path: str | PathLike[str] = None,
                                  title: str = "Spontaneous trajectory (Campaign 1)",
                                  burn_in: float = None) -> None:
    """
    Plot the ensemble-mean unforced trajectory <X>(t) from Campaign 1.

    For a stationary equilibrium system this should: (a) follow the deterministic
    transient during burn-in, (b) decorrelate as independent noise averages out,
    (c) settle to fluctuations of amplitude ~ sigma_x / sqrt(M) around the
    equilibrium mean. Useful as a visual stationarity / burn-in check.

    :param t: (T,) time axis in ND units.
    :param x_mean: (T,) ensemble-mean bundle position trajectory.
    :param save_path: optional path to save the figure.
    :param title: plot title.
    :param burn_in: optional ND time at which to draw a vertical line marking the
                    transient/steady-state boundary (where PSD computation starts).
    """
    t = np.asarray(t)
    x_mean = np.asarray(x_mean)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, x_mean, linewidth=0.8, color='steelblue')
    ax.axhline(0.0, color='k', linestyle='--', linewidth=0.8)
    if burn_in is not None:
        ax.axvline(burn_in, color='darkorange', linestyle=':', linewidth=1.2,
                    label=fr'burn-in ends: $\tilde t = {burn_in:.0f}$')
        ax.legend()
    ax.set_xlabel(r'$\tilde t$ (ND)')
    ax.set_ylabel(r'$\langle \tilde X \rangle(\tilde t)$ (ND)')
    ax.set_title(title)
    ax.grid(False)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_psd(omegas: np.ndarray, G: np.ndarray,
             save_path: str | PathLike[str] = None,
             title: str = "Spontaneous PSD (Campaign 1)",
             omega_natural: float = None,
             plot_band: tuple = None) -> None:
    """
    Log-log plot of one-sided angular-frequency PSD G(omega) vs omega.

    Used to visually verify the spontaneous-oscillation frequency that
    find_spectral_peak picks out. Negative or non-finite PSD values (numerical
    artifacts) are dropped with a warning.

    :param omegas: (n_freqs,) angular frequencies (ND) from psd_welch.
    :param G: (n_freqs,) one-sided PSD values from psd_welch.
    :param save_path: optional path to save the figure.
    :param title: plot title.
    :param omega_natural: optional vertical marker for the detected PSD peak.
    :param plot_band: optional (lo, hi) to restrict the x-range, e.g. matching the
                      production grid. Defaults to the full PSD range (minus DC).
    """
    omegas = np.asarray(omegas)
    G = np.asarray(G)

    # Skip DC bin (omega = 0 plots at -inf on log scale)
    mask = omegas > 0
    omegas, G = omegas[mask], G[mask]

    n_bad = int((G <= 0).sum() + (~np.isfinite(G)).sum())
    if n_bad > 0:
        print(f"plot_psd: dropping {n_bad}/{len(G)} non-positive or non-finite PSD points.")
        good = np.isfinite(G) & (G > 0)
        omegas, G = omegas[good], G[good]

    if plot_band is not None:
        in_band = (omegas >= plot_band[0]) & (omegas <= plot_band[1])
        omegas, G = omegas[in_band], G[in_band]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(omegas, G, marker='.', linestyle='none',
              markersize=3, color='steelblue')
    if omega_natural is not None:
        ax.axvline(omega_natural, color='darkorange', linestyle=':', linewidth=1.2,
                    label=fr'$\Omega_0 \approx {omega_natural:.3f}$ (peak)')
        ax.legend()
    ax.set_xlabel(r'$\tilde\omega$ (ND), log scale')
    ax.set_ylabel(r'$G(\tilde\omega)$, log scale')
    ax.set_title(title)
    ax.grid(False)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_chi_components(omegas: np.ndarray, chis: np.ndarray,
                        save_path: str | PathLike[str] = None,
                        title: str = "Susceptibility components",
                        omega_natural: float = None) -> None:
    """
    Two-panel plot of chi'(omega) and chi''(omega) vs angular frequency.

    Directly comparable to Martin et al. 2001 Fig 2: log-spaced x-axis, linear y,
    horizontal line at zero in each panel. Negative chi'' below the natural
    frequency indicates active force generation (energy gain from the drive);
    positive chi'' indicates dissipation. chi' (in-phase) changes sign at the
    effective resonance for a damped oscillator.

    :param omegas: (n_freqs,) angular frequencies (ND).
    :param chis: (n_freqs,) complex chi values.
    :param save_path: optional path to save the figure.
    :param title: plot title (applied to the top panel).
    :param omega_natural: optional vertical marker at the spontaneous-oscillation freq.
    """
    omegas = np.asarray(omegas)
    chis = np.asarray(chis)
    chi_real = chis.real
    chi_imag = chis.imag

    fig, (ax_real, ax_imag) = plt.subplots(2, 1, figsize=(7, 8), sharex=True)

    # chi' (real / in-phase)
    ax_real.semilogx(omegas, chi_real, marker='o', linestyle='none',
                     markersize=4, color='steelblue')
    ax_real.axhline(0.0, color='k', linestyle='--', linewidth=0.8)
    if omega_natural is not None:
        ax_real.axvline(omega_natural, color='darkorange', linestyle=':', linewidth=1.2,
                         label=fr'$\Omega_0 \approx {omega_natural:.3f}$ (PSD peak)')
        ax_real.legend()
    ax_real.set_ylabel(r"$\chi'(\tilde\omega)$ (in-phase)")
    ax_real.set_title(title)
    ax_real.grid(False)

    # chi'' (imaginary / dissipative)
    ax_imag.semilogx(omegas, chi_imag, marker='o', linestyle='none',
                     markersize=4, color='steelblue')
    ax_imag.axhline(0.0, color='k', linestyle='--', linewidth=0.8)
    if omega_natural is not None:
        ax_imag.axvline(omega_natural, color='darkorange', linestyle=':', linewidth=1.2)
    ax_imag.set_xlabel(r"$\tilde\omega$ (ND), log scale")
    ax_imag.set_ylabel(r"$\chi''(\tilde\omega)$ (dissipative)")
    ax_imag.grid(False)

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=150)
    plt.show()
