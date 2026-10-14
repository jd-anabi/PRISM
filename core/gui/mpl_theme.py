"""matplotlib's process-wide settings for every figure the engine builds.

They are the LIGHT side of the look's table in both appearances, written once, when the application
is built; a change of appearance does not touch them. So a stored figure never carries dark colours,
and a run that straddles a change of appearance cannot build a figure under one set of settings and
save it under another. The lengths do not grow with the text size: a stored figure must not depend
on the computer that drew it.

The engine's plotting code imports nothing of the window. It reads matplotlib's settings, which are
these in the window and matplotlib's own on the command line. A figure shown in a page is a picture
and is not drawn again. Popped out, a figure that came with its stored form is built again from it
and drawn by matplotlib in a window of its own; any other is shown there as its picture.
"""
import matplotlib

from . import design, fonts

_PT = 0.72                        # points in one pixel at 100 dots per inch
_FALLBACK_FACE = "DejaVu Sans"    # matplotlib's own face: it draws the characters Inter lacks


def _rc() -> dict:
    """The figure settings, built anew at each call: the light side of the look's table whatever the
    theme in force, the three lengths as the table holds them, the lettering face only when matplotlib
    has both of its files."""
    light = design.tokens(False)
    tick = design.FIGURE["tick"] * _PT
    axis_w = design.FIGURE["axis_w"] * _PT
    rc = {
        "figure.facecolor":  light["paper"],  "figure.edgecolor":  light["paper"],
        "savefig.facecolor": light["paper"],  "savefig.edgecolor": light["paper"],
        "axes.facecolor":    light["paper"],  "axes.edgecolor":    light["rule"],
        "axes.labelcolor":   light["ink"],    "axes.titlecolor":   light["ink"],
        "text.color":        light["ink"],
        "xtick.color":       light["rule"],   "ytick.color":       light["rule"],
        "xtick.labelcolor":  light["ink_2"],  "ytick.labelcolor":  light["ink_2"],
        "grid.color":        light["rule_light"],
        "legend.facecolor":  light["paper"],  "legend.edgecolor":  light["paper"],
        "legend.framealpha": 1.0,
        "axes.spines.top":   False,           "axes.spines.right": False,
        "xtick.direction":   "out",           "ytick.direction":   "out",
        "xtick.major.size":  tick,            "ytick.major.size":  tick,
        "axes.linewidth":    axis_w,
        "xtick.major.width": axis_w,          "ytick.major.width": axis_w,
        "lines.linewidth":   design.FIGURE["line_w"] * _PT,
        "lines.solid_joinstyle": "round",     "lines.dash_joinstyle": "round",
        "axes.unicode_minus": True,
        # the seven colours in their order, and a colour only: the set carries no line style
        "axes.prop_cycle":   matplotlib.cycler(color=[light[name] for name in design.SERIES]),
    }
    if fonts.register_with_matplotlib():
        rc["font.family"] = [design.LETTERING, _FALLBACK_FACE]
    return rc


def apply_mpl_theme() -> None:
    """Write the light figure settings into matplotlib's process-wide settings. Only a figure built
    after the call takes them."""
    matplotlib.rcParams.update(_rc())


def install(appearance) -> None:
    """Write the light figure settings now. Nothing without an appearance; and it listens to nothing,
    so a later change of appearance leaves the settings as they are."""
    if appearance is None:
        return
    apply_mpl_theme()
