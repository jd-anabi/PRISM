"""Drive matplotlib's global rcParams from the app's design tokens so figures follow the dark/light
theme (B-c). Installed once by ``app.build_app`` and re-applied on ``Appearance.theme_changed`` -- the
same "apply-now + subscribe" shape the pyqtgraph live view uses (widgets/live_hair_bundle._install_theme).

Figures are one-shot in this app (rasterized to a PNG at build time, shown as a QPixmap), so colours only
need to be right AT CONSTRUCTION -- global rcParams delivers that with no per-figure code, and the core
plot modules (visualizers.py, FDT/plots.py, Reduction/plots.py) stay CLI-safe by reading plt.rcParams
(matplotlib's own defaults under the CLI/tests) rather than these GUI tokens. Known limitation: a theme
flip does NOT recolour figures already displayed -- they adopt the new theme on the next run/build.
"""
import matplotlib

from . import design


def _rc(dark: bool) -> dict:
    t = design.tokens(dark)                                   # no accent: figure chrome is neutral
    return {
        "figure.facecolor":  t["window"], "figure.edgecolor":  t["window"],
        "savefig.facecolor": t["window"], "savefig.edgecolor": t["window"],
        "axes.facecolor":    t["base"],   "axes.edgecolor":    t["text_2nd"],
        "axes.labelcolor":   t["text"],   "axes.titlecolor":   t["text"],
        "text.color":        t["text"],
        "xtick.color":       t["text_2nd"], "ytick.color":     t["text_2nd"],
        "xtick.labelcolor":  t["text"],   "ytick.labelcolor":  t["text"],
        "grid.color":        t["mid"],
        "legend.facecolor":  t["base"],   "legend.edgecolor":  t["mid"],
    }


def apply_mpl_theme(dark: bool) -> None:
    """Set the theme-driven rcParams. Affects only figures built AFTER this call (rcParams is global)."""
    matplotlib.rcParams.update(_rc(dark))


def install(appearance) -> None:
    """Apply the current theme now and re-apply on every theme change. No-op without an Appearance
    (the test path builds MainWindow directly, so figures keep matplotlib's default white there)."""
    if appearance is None:                                    # mirrors live_hair_bundle._install_theme
        return
    apply_mpl_theme(appearance.is_dark())
    appearance.theme_changed.connect(apply_mpl_theme)
