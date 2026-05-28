"""
3D plotting for the FDT parameter-sweep study.

One surface per sweep: T_eff/T  vs  (omega/omega_0, swept_param).

All axes linear. The omega/omega_0 grid is shared across operating points (each
Campaign-2 grid is omega_0 x fixed log-ratios), so rows stack directly. A faint
reference line at omega/omega_0 = 1 marks the resonance; the FDT-satisfied level
is T_eff/T = 1.
"""
from __future__ import annotations
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


_PLOT_DIR = Path("Resources/Plots")

# Default linear x-window. omega/omega_0 spans [0.1, 30] but the structure
# (deviation from FDT) lives near resonance; a linear axis out to 30 squashes it.
# Crop to focus on the active band; full data is saved in the HDF5 for re-plotting.
_OMEGA_NORM_MAX = 5.0


def _stack(records: list[dict], omega_norm_max: float):
    """
    Filter to usable rows, sort by param, and stack T_eff/T into a (n_param, n_omega)
    matrix on the shared omega/omega_0 grid (cropped to [min, omega_norm_max]).

    :returns: (omega_norm, param_values, T_matrix)
    """
    usable = [r for r in records if not r["failed"] and "omega_norm" in r]
    if not usable:
        raise ValueError("No usable (non-failed) operating points to plot.")
    usable.sort(key=lambda r: r["param_value"])

    # All rows share the same omega/omega_0 grid by construction; take the first
    # and verify the rest match within tolerance.
    omega_norm = np.asarray(usable[0]["omega_norm"], dtype=np.float64)
    for r in usable[1:]:
        if not np.allclose(r["omega_norm"], omega_norm, rtol=1e-6, atol=1e-9):
            # Fall back to interpolation if grids unexpectedly differ.
            return _stack_interp(usable, omega_norm_max)

    crop = omega_norm <= omega_norm_max
    omega_norm = omega_norm[crop]
    param_values = np.array([r["param_value"] for r in usable])
    T_matrix = np.stack([np.asarray(r["T_eff_over_T"])[crop] for r in usable], axis=0)
    return omega_norm, param_values, T_matrix


def _stack_interp(usable: list[dict], omega_norm_max: float):
    """Fallback: interpolate every row onto a common cropped omega/omega_0 grid."""
    lo = max(r["omega_norm"].min() for r in usable)
    hi = min(min(r["omega_norm"].max() for r in usable), omega_norm_max)
    omega_norm = np.linspace(lo, hi, 200)
    param_values = np.array([r["param_value"] for r in usable])
    rows = []
    for r in usable:
        on = np.asarray(r["omega_norm"]); te = np.asarray(r["T_eff_over_T"])
        order = np.argsort(on)
        rows.append(np.interp(omega_norm, on[order], te[order], left=np.nan, right=np.nan))
    return omega_norm, param_values, np.stack(rows, axis=0)


def plot_fdt_3d_vs_param(
    records: list[dict],
    param_symbol: str,
    title: str,
    filename_tag: str,
    omega_norm_max: float = _OMEGA_NORM_MAX,
    z_clip: tuple[float, float] = (0.0, 2.0),
    save: bool = True,
    show: bool = False,
) -> Path | None:
    """
    3D surface + 2D heatmap of T_eff/T vs (omega/omega_0, swept param).

    :param records: output of load_param_sweep.
    :param param_symbol: y-axis label, e.g. r"$S$" or r"$T_a/T$".
    :param title: figure title.
    :param filename_tag: prefix for the saved PNG.
    :param omega_norm_max: linear x-axis upper limit (crop). Full data is in the HDF5.
    :param z_clip: (lo, hi) display range for T_eff/T. T_eff/T has a genuine pole
                   where chi'' -> 0 (it can spike to ~1000s), which on a linear scale
                   crushes the structure near the FDT-satisfied level. Values are
                   clipped to this range so the pole saturates and the restoration
                   trend (-> 1) stays readable. Pass None to disable clipping.
    """
    omega_norm, param_values, T_matrix = _stack(records, omega_norm_max)

    # Clip for display so the chi''=0 pole saturates instead of dominating the scale.
    if z_clip is not None:
        vmin, vmax = z_clip
        T_disp = np.clip(T_matrix, vmin, vmax)
        clip_note = f"  (clipped to [{vmin:g}, {vmax:g}])"
    else:
        T_disp = T_matrix
        vmin = float(np.nanmin(T_matrix)); vmax = float(np.nanmax(T_matrix))
        clip_note = ""

    fig = plt.figure(figsize=(13, 5.5))
    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    ax2d = fig.add_subplot(1, 2, 2)

    X, Y = np.meshgrid(omega_norm, param_values)   # X=omega/omega_0, Y=param
    ax3d.plot_surface(X, Y, T_disp, cmap="viridis", edgecolor="none", alpha=0.9,
                      vmin=vmin, vmax=vmax)
    ax3d.set_zlim(vmin, vmax)
    ax3d.set_xlabel(r"$\tilde\omega / \Omega_0$")
    ax3d.set_ylabel(param_symbol)
    ax3d.set_zlabel(r"$T_{\rm eff}/T$")
    ax3d.set_title(title + clip_note)

    im = ax2d.pcolormesh(X, Y, T_disp, cmap="viridis", shading="auto",
                         vmin=vmin, vmax=vmax)
    ax2d.axvline(1.0, color="darkorange", ls=":", lw=1.2, label=r"$\tilde\omega/\Omega_0 = 1$")
    ax2d.set_xlabel(r"$\tilde\omega / \Omega_0$")
    ax2d.set_ylabel(param_symbol)
    ax2d.set_title("2D projection")
    ax2d.legend(loc="upper right", fontsize=8)
    fig.colorbar(im, ax=ax2d, label=r"$T_{\rm eff}/T$")

    fig.tight_layout()

    if save:
        _PLOT_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = _PLOT_DIR / f"{filename_tag}_{stamp}.png"
        fig.savefig(out, dpi=160, bbox_inches="tight")
        if not show:
            plt.close(fig)
        return out
    if show:
        plt.show()
    return None
