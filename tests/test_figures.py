"""Figure sinks, pop-out windows, the figure settings and label rendering. Split from test_gui_progress.py; run directly: pytest tests/test_figures.py"""
"""Progress-rendering regression tests for the GUI.

THE BUG THESE LOCK DOWN
    tqdm redraws a bar at pos>0 as three writes -- '\\n'*pos, then '\\r'+frame, then '\\x1b[A'*pos
    (tqdm/std.py:1493-1497). The old stream reader split on terminators, so the frame (which is never
    terminated) stranded in its buffer and was flushed by the NEXT redraw's leading '\\n' -- i.e. as a
    LOG LINE. Every nested-bar redraw appended one row, so a training run buried the log pane under
    hundreds of bar snapshots.

    The pipeline nests bars four deep (core/SBI/pipeline.py:517 -> :371 ->
    core/Simulator/simulator.py:50 -> core/Solvers/sdeint.py:15), so this fired constantly.

Run:  pytest tests/test_figures.py
"""
import os
import tempfile
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")   # must precede any PySide6 import
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib                                                 # noqa: E402
matplotlib.use("Agg")                                            # match the app (core/gui/__main__.py forces it)

import torch                                                      # noqa: E402
from tqdm import tqdm                                             # noqa: E402

from core.gui.worker import WorkerSignals                         # noqa: E402
from tests._fixtures import build_part, qt_app, pump              # noqa: E402
import contextlib                                                  # noqa: E402

# ── interactive "Pop out" for figures ────────────────────────────────────────────────────────────
def _tiny_fig():
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot([0, 1, 2], [0, 1, 4])
    return fig
def _png_bytes(fig):
    import io
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    return buf.getvalue()
# ── Simulate section: "Save video…" export ───────────────────────────────────────────────────────
def _tiny_series():
    import numpy as np
    t = np.arange(600) * 1e-3                                   # 0.6 s -> several video frames at 30 fps
    x = np.sin(2 * np.pi * 8 * t) * 1.5 + 0.2
    return np.column_stack((t, x))
def _export_kwargs():
    import numpy as np
    return dict(window_pts=2000, grid_x=np.linspace(0, 2.6, 60), grid_y=np.linspace(0, 1, 24),
                sigma_x=0.10, sigma_y=0.20, aspect=2.6, margin=0.35, video_fps=30)
@contextlib.contextmanager
def _rcparams_guard():
    """Snapshot + restore matplotlib rcParams (process-global), so a test that writes figure settings
    leaves matplotlib's own for the tests after it."""
    import matplotlib
    saved = dict(matplotlib.rcParams)
    try:
        yield
    finally:
        matplotlib.rcParams.update(saved)
# Dark figure settings, written out by hand: nothing in the window writes dark ones, and three tests
# still need them to show what the engine's helpers and the video export do under them.
_DARK_BY_HAND = {
    "figure.facecolor": "#1C1C1C", "figure.edgecolor": "#1C1C1C",
    "savefig.facecolor": "#1C1C1C", "savefig.edgecolor": "#1C1C1C",
    "axes.facecolor": "#1C1C1C", "axes.edgecolor": "#C8C8C8",
    "axes.labelcolor": "#EDEDED", "axes.titlecolor": "#EDEDED", "text.color": "#EDEDED",
    "xtick.color": "#C8C8C8", "ytick.color": "#C8C8C8",
    "xtick.labelcolor": "#B8B8B8", "ytick.labelcolor": "#B8B8B8",
}
def _fake_appearance(dark):
    """A minimal Appearance stand-in (is_dark + a real theme_changed Signal) so the test of the figure
    settings never touches theming._ACTIVE / the app styleHints wiring a real Appearance installs."""
    from PySide6.QtCore import QObject, Signal

    class _FakeAppearance(QObject):
        theme_changed = Signal(bool)

        def __init__(self, d):
            super().__init__()
            self._dark = d

        def is_dark(self):
            return self._dark

        def flip(self, d):
            self._dark = d
            self.theme_changed.emit(d)

    return _FakeAppearance(dark)
def _settings_that_changed(before) -> list:
    """The names of matplotlib's settings whose value is no longer the one in `before`, a plain copy of
    all of them taken earlier. Empty when nothing was written over; a failure names what was."""
    import matplotlib
    return sorted(name for name, value in dict(matplotlib.rcParams).items() if before[name] != value)


def test_fig_sink_emits_png_and_a_reloadable_pickle():
    """The worker sink must emit BOTH the PNG thumbnail and a pickle that reloads to a real Figure on
    the GUI thread -- that pickle is what "Pop out" rebuilds into an interactive window."""
    import pickle
    import matplotlib.pyplot as plt
    from core.gui.panels.base_panel import _png_fig_sink

    qt_app()
    sig = WorkerSignals()
    events = []
    sig.figure.connect(lambda title, png, fp: events.append((title, png, fp)))

    _png_fig_sink(sig.figure)("Corner", _tiny_fig())

    assert len(events) == 1, events
    title, png, fp = events[0]
    assert title == "Corner"
    assert png[:4] == b"\x89PNG", "the PNG thumbnail is missing / not a PNG"
    assert fp is not None, "no pickle was shipped for the pop-out"
    assert len(pickle.loads(fp).axes) == 1, "the pickle did not reload to the figure"
    plt.close("all")

def test_fig_sink_pickle_failure_still_emits_the_png():
    """If a figure will not pickle, the run must not break: emit fig_pickle=None and keep the PNG."""
    import pickle
    import matplotlib.pyplot as plt
    from core.gui.panels.base_panel import _png_fig_sink

    qt_app()
    sig = WorkerSignals()
    events = []
    sig.figure.connect(lambda title, png, fp: events.append((title, png, fp)))

    real = pickle.dumps
    pickle.dumps = lambda *a, **k: (_ for _ in ()).throw(TypeError("cannot pickle this"))
    try:
        _png_fig_sink(sig.figure)("X", _tiny_fig())
    finally:
        pickle.dumps = real

    assert len(events) == 1
    _title, png, fp = events[0]
    assert fp is None, "a pickle failure should degrade to None, not raise or drop the event"
    assert png[:4] == b"\x89PNG", "the PNG must still be emitted when pickling fails"
    plt.close("all")

def test_add_figure_creates_an_interactive_capable_tab():
    import pickle
    import matplotlib.pyplot as plt
    from PySide6.QtWidgets import QPushButton
    from core.gui.widgets.figure_stack import FigureStack

    qt_app()
    fs = FigureStack()
    fig = _tiny_fig()
    fs.add_figure("Corner", _png_bytes(fig), fig_pickle=pickle.dumps(fig))
    plt.close("all")

    assert fs.count() == 1
    container = fs.widget(0)
    assert getattr(container, "_fig_pickle", None) is not None, "the pickle was not stored on the tab"
    assert any(b.text() == "Pop out" for b in container.findChildren(QPushButton)), "no Pop out button"

def test_pop_out_of_a_pickle_builds_a_qtagg_canvas_and_keeps_pyplot_clean():
    """Popping a pickled figure builds a live FigureCanvasQTAgg on the GUI thread, and -- the linchpin
    -- must NOT leave the reconstructed figure registered in pyplot's Gcf (else the worker's
    plt.close("all") would later close it out from under the user)."""
    import pickle
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
    from core.gui.widgets.figure_stack import FigureStack

    app = qt_app()
    fs = FigureStack()
    fig = _tiny_fig()
    fs.add_figure("Corner", _png_bytes(fig), fig_pickle=pickle.dumps(fig))
    plt.close("all")

    before = plt.get_fignums()
    fs._pop_out(fs.widget(0))
    assert len(fs._windows) == 1
    win = next(iter(fs._windows))
    assert isinstance(win.canvas, FigureCanvasQTAgg)
    assert plt.get_fignums() == before, "the unpickled figure leaked into pyplot's Gcf"

    win.close()
    pump(app, 0.1)
    assert len(fs._windows) == 0, "the window ref was not dropped on close"
    plt.close("all")

def test_pop_out_without_a_pickle_uses_the_image_viewer():
    import matplotlib.pyplot as plt
    from PySide6.QtWidgets import QGraphicsView
    from core.gui.widgets.figure_stack import FigureStack
    from core.gui.widgets.figure_window import ImageZoomWindow

    app = qt_app()
    fs = FigureStack()
    fig = _tiny_fig()
    fs.add_figure("NoPickle", _png_bytes(fig), fig_pickle=None)
    plt.close("all")

    fs._pop_out(fs.widget(0))
    win = next(iter(fs._windows))
    assert isinstance(win, ImageZoomWindow), "a pickle-less figure should open the image viewer"
    assert isinstance(win.view, QGraphicsView)
    win.close()
    pump(app, 0.1)

def test_disk_png_pop_out_is_an_image_viewer():
    import tempfile
    import matplotlib.pyplot as plt
    from PySide6.QtWidgets import QGraphicsView
    from core.gui.widgets.figure_stack import FigureStack
    from core.gui.widgets.figure_window import ImageZoomWindow

    app = qt_app()
    png = Path(tempfile.mkdtemp()) / "sweep.png"
    fig = _tiny_fig()
    fig.savefig(str(png), format="png")
    plt.close("all")

    fs = FigureStack()
    fs.add_png("Sweep", str(png))
    fs._pop_out(fs.widget(0))
    win = next(iter(fs._windows))
    assert isinstance(win, ImageZoomWindow)
    assert isinstance(win.view, QGraphicsView), "the disk-PNG pop-out has no zoom/pan view"
    win.close()
    pump(app, 0.1)

def test_a_popped_out_figure_survives_a_worker_plt_close_all():
    """Worker.run runs plt.close("all") after every run and every cancel. A figure the user has popped
    out must NOT be torn down by it -- the Gcf detach in build_interactive_window is the guarantee."""
    import pickle
    import matplotlib.pyplot as plt
    from core.gui.widgets.figure_stack import FigureStack

    app = qt_app()
    fs = FigureStack()
    fig = _tiny_fig()
    fs.add_figure("Corner", _png_bytes(fig), fig_pickle=pickle.dumps(fig))
    plt.close("all")

    fs._pop_out(fs.widget(0))
    win = next(iter(fs._windows))
    assert len(win._fig.axes) == 1

    plt.close("all")     # the worker's teardown, fired process-wide
    assert len(win._fig.axes) == 1, "plt.close('all') destroyed a figure being viewed in a pop-out"

    win.close()
    pump(app, 0.1)
    plt.close("all")

# ── Labels + units ───────────────────────────────────────────────────────────────────────────────
def test_labels_axis_and_rescale_render_latex_with_units():
    from core.Helpers import labels as L
    assert L.axis_label("x", "nm") == "$x$ (nm)"
    assert L.axis_label(r"\tilde\omega") == r"$\tilde\omega$ (ND)"          # unit=None -> ND
    assert "ms/ND" in L.rescale_axis_label("t_scale", time_unit="ms")
    assert "nm/ND" in L.rescale_axis_label("x_scale", length_unit="nm")
    assert "pN/ND" in L.rescale_axis_label("f_scale", force_unit="pN")
    assert L.rescale_axis_label("x_offset", length_unit="nm") == r"$x_{\mathrm{off}}$ (nm)"
    assert L.rescale_axis_label("t_scale") == r"$t_{\mathrm{scale}}$"       # missing token -> bare symbol

def test_labels_pretty_gui_and_forcing():
    from core.Helpers import labels as L
    assert L.pretty_gui("F0 (ND forcing amplitude)") == "F<sub>0</sub> (ND forcing amplitude)"
    assert "<sub>obs</sub>" in L.pretty_gui("T_obs (s)")
    assert "<sub>a</sub>/T" in L.pretty_gui("T_a/T grid  (S = 0)")
    assert L.pretty_gui("Model") == "Model"                                # passthrough for non-math
    assert L.gui_forcing_label("phase", "rad") == "φ (rad)"
    assert L.gui_forcing_label("amp") == "A"

def test_simconfig_units_and_inferred_labels_are_latex():
    from pathlib import Path
    from core import cli
    from core.config import BOUNDS_PATH, CELL_PATH, VALID_LABELS, VALID_MODELS

    cell = CELL_PATH / "nadrowski" / "master_weak.txt"
    bounds = BOUNDS_PATH / "nadrowski" / "master.txt"
    if not (cell.exists() and bounds.exists()):
        return                                                             # environment without Resources: skip
    cfg = cli.make_sim_config("NADROWSKI", VALID_LABELS[VALID_MODELS.index("NADROWSKI")], True, str(bounds))
    cli.load_and_validate_gt(cfg, str(cell))
    # freq is declared kHz, NOT Hz: drive frequency is consumed as INVERSE CELL TIME (1/ms = kHz), so a
    # declared "Hz" against an `ms` cell is a 1000x lie. See SimConfig.freq_si_to_cell.
    assert (cfg.length_unit, cfg.time_unit, cfg.force_unit, cfg.freq_unit) == ("nm", "ms", "pN", "kHz")
    assert cfg.check_unit_consistency() == [], "nadrowski units must be self-consistent"
    assert abs(30.0 * cfg.freq_si_to_cell - 0.03) < 1e-12, "30 Hz must be 0.03 cycles/ms in an ms cell"
    labels = cfg.inferred_labels
    assert all(l.startswith("$") for l in labels), labels
    assert any("nm/ND" in l for l in labels), "x_scale should carry nm/ND"

    bp_bounds = BOUNDS_PATH / "bp" / "cell_1.txt"
    if bp_bounds.exists():                                                 # BP declares no force/freq unit
        bp = cli.make_sim_config("BP", VALID_LABELS[VALID_MODELS.index("BP")], False, str(bp_bounds))
        assert bp.force_unit is None and bp.length_unit == "nm"

def test_plot_posterior_vs_truth_default_labels_have_units():
    import numpy as np
    from core.Helpers import visualizers
    from matplotlib import pyplot as plt
    fig = visualizers.plot_posterior_vs_truth(np.arange(5) * 1.0, np.zeros(5))
    ax = fig.axes[0]
    labels_ = (ax.get_xlabel(), ax.get_ylabel())
    plt.close(fig)                                   # the gate's pyplot is shared: close what is drawn
    assert labels_ == ("$t$ (s)", "$x$ (nm)")
    fig2 = visualizers.plot_posterior_vs_truth(np.arange(5) * 1.0, np.zeros(5),
                                               xlabel="$t$ (ms)", ylabel="$x$ (µm)")
    xlabel2 = fig2.axes[0].get_xlabel()
    plt.close(fig2)
    assert xlabel2 == "$t$ (ms)"

def test_gui_form_labels_are_prettified():
    from PySide6.QtWidgets import QLabel
    from core.gui.widgets.help_badge import help_label

    qt_app()
    holder = help_label("F0 (ND forcing amplitude)", "help text")
    lbl = holder.findChild(QLabel)
    assert lbl is not None and "F<sub>0</sub>" in lbl.text()
    # panels still build with the prettify hook in place
    from core.gui.panels.crossval_panel import CrossValPanel
    from core.gui.panels.fdt_panel import FdtPanel
    from core.gui.panels.simulate_panel import SimulatePanel
    build_part(FdtPanel); build_part(CrossValPanel); build_part(SimulatePanel)

def test_a_theme_flip_between_build_and_save_cannot_split_a_figure():
    """THE 2026-08-25 RENDERING FAILURE, pinned, with its own counterfactual.

    matplotlib bakes artist colours at BUILD time but reads `savefig.facecolor` at SAVE time. The
    window used to rewrite the process-wide settings at every change of appearance, so a multi-day
    run straddling one saved dark axes onto a light page with `text.color` still light -- every tick
    label, axis label, title and table cell at ~1.06:1 contrast, i.e. invisible. 14 of 15 figures.
    The window writes its figure settings once now; `save_figure` still guards a figure against
    anything that changes them between its build and its save, and this test is that guard's.
    """
    import io as _io
    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image
    from core.gui import mpl_theme
    from core.Helpers import visualizers

    qt_app()
    with _rcparams_guard():
        matplotlib.rcParams.update(_DARK_BY_HAND)   # build under dark settings
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])
        ax.set_xlabel("x")
        try:
            mpl_theme.apply_mpl_theme()   # ...then the settings turn light before the save

            def corner_luminance(save):
                buf = _io.BytesIO()
                save(buf)
                buf.seek(0)
                return float(np.array(Image.open(buf).convert("RGB")).astype(int)[:6, :6].mean())

            pinned = corner_luminance(
                lambda b: visualizers.save_figure(fig, b, format="png", dpi=50))
            unpinned = corner_luminance(
                lambda b: fig.savefig(b, format="png", dpi=50))
        finally:
            plt.close(fig)

    assert pinned < 120, \
        f"save_figure put dark artists on a light page after a theme flip (corner {pinned:.0f})"
    assert unpinned > 200, \
        f"the counterfactual did not reproduce the bug (corner {unpinned:.0f}); this test proves nothing"

def test_the_parameter_table_is_readable_against_its_own_cells():
    """The best-fit parameter table came back completely blank. Transparent cells show the FIGURE
    background while the text is `text.color`, and those two only contrast while the theme is
    self-consistent -- so the 13 numbers you most want to read are the first thing to vanish."""
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import to_rgb
    from core.gui import mpl_theme
    from core.Helpers import visualizers

    def relative_luminance(c):
        r, g, b = (v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4 for v in to_rgb(c))
        return 0.2126 * r + 0.7152 * g + 0.0722 * b

    qt_app()
    with _rcparams_guard():
        mpl_theme.apply_mpl_theme()
        t = np.linspace(0, 1, 32)
        fig = visualizers.plot_best_fit_overlay(
            t, np.sin(t), np.sin(t) * 1.02,
            param_labels=[f"p{i}" for i in range(13)], param_values=list(range(13)))
        try:
            tables = [c for a in fig.axes for c in a.tables]
            assert tables, "the parameter table is gone"
            cells = list(tables[0].get_celld().values())
            for cell in cells:
                bg = cell.get_facecolor()
                assert bg[3] > 0.5, "a transparent cell: the text falls back to the FIGURE colour"
                l1 = relative_luminance(bg[:3])
                l2 = relative_luminance(cell.get_text().get_color())
                ratio = (max(l1, l2) + 0.05) / (min(l1, l2) + 0.05)
                assert ratio >= 3.0, \
                    f"table text at {ratio:.2f}:1 against its cell (was 1.06:1)"
        finally:
            plt.close(fig)

def test_apply_mpl_theme_writes_the_light_settings():
    """apply_mpl_theme() writes the light side of the look's table into matplotlib's settings."""
    import matplotlib
    from core.gui import design, mpl_theme
    with _rcparams_guard():
        mpl_theme.apply_mpl_theme()
        light = design.tokens(False)
        assert matplotlib.rcParams["figure.facecolor"] == light["paper"] == "#FFFFFF"
        assert matplotlib.rcParams["axes.facecolor"] == light["paper"]
        assert matplotlib.rcParams["savefig.facecolor"] == light["paper"]
        assert matplotlib.rcParams["text.color"] == light["ink"] == "#1A1A1A"
        assert all(matplotlib.rcParams[name] == value for name, value in mpl_theme._rc().items())

def test_mpl_theme_install_none_is_a_noop():
    """No Appearance (the test path) -> rcParams untouched, so figures keep matplotlib's default white."""
    import matplotlib
    from core.gui import mpl_theme
    with _rcparams_guard():
        matplotlib.rcdefaults()     # matplotlib's own in force, so the window's would show if they were written
        before = dict(matplotlib.rcParams)
        mpl_theme.install(None)
        written = _settings_that_changed(before)
        assert not written, f"install(None) wrote these figure settings: {written}"

def test_a_change_of_theme_changes_no_figure_setting():
    """install(appearance) writes the light settings once, under a dark appearance too, and listens
    to nothing afterwards: a later change of theme leaves every figure setting as it finds it."""
    import matplotlib
    from core.gui import mpl_theme
    qt_app()
    with _rcparams_guard():
        ap = _fake_appearance(True)
        mpl_theme.install(ap)
        assert matplotlib.rcParams["figure.facecolor"] == "#FFFFFF"
        assert matplotlib.rcParams["text.color"] == "#1A1A1A"
        matplotlib.rcdefaults()     # matplotlib's own back, so a setting install wrote would show if written again
        matplotlib.rcParams["figure.facecolor"] = "#123456"        # a mark no setting of the look has
        before = dict(matplotlib.rcParams)
        ap.flip(False)
        ap.flip(True)
        rewritten = _settings_that_changed(before)
        assert not rewritten, f"a change of theme rewrote these figure settings: {rewritten}"

def test_plot_ppc_summary_box_takes_its_ground_from_the_figure_settings():
    """The two PPC summary boxes read plt.rcParams, so under dark figure settings they are NOT the old
    hardcoded white -- a regression back to that hardcoded white fails here."""
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt
    from core.Helpers.visualizers import plot_ppc
    with _rcparams_guard():
        matplotlib.rcParams.update(_DARK_BY_HAND)
        ppc = {"z_scores": np.array([0.1, 0.5, 2.5, np.nan]),
               "coverage_90": 0.9, "num_outside": 1, "num_invalid": 1}
        fig = plot_ppc(ppc)
        try:
            fig.canvas.draw()
            boxes = [t.get_bbox_patch() for t in fig.axes[0].texts if t.get_bbox_patch() is not None]
            assert boxes, "expected summary boxes with a bbox patch"
            assert all(max(p.get_facecolor()[:3]) < 0.5 for p in boxes), \
                [p.get_facecolor() for p in boxes]
        finally:
            plt.close(fig)

def test_export_animation_page_stays_white_under_dark_figure_settings():
    """The Simulate video export pins its own page: it stays white even when matplotlib's process-wide
    settings are dark, so the page of an exported video does not depend on the settings in force."""
    import os
    import tempfile
    import matplotlib
    from core.gui.panels.simulate_export import export_animation
    with _rcparams_guard():
        matplotlib.rcParams.update(_DARK_BY_HAND)                  # dark settings in force
        path = os.path.join(tempfile.mkdtemp(), "dark.gif")
        export_animation(_tiny_series(), path, **_export_kwargs())
        import imageio
        frame0 = imageio.mimread(path)[0]
        corner = frame0[0, 0][:3]                                  # top-left = figure background margin
        assert min(int(c) for c in corner) > 230, corner          # near-white, not the dark page

# ── what a figure draws, its colours left out ────────────────────────────────────────────────────
def _plain(value):
    """Anything an artist hands back, as text and rounded numbers that compare equal run after run.
    Ten significant digits, so the last bits of a float never decide a comparison."""
    import numpy as np
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, (int, float, np.integer, np.floating)):
        v = float(value)
        return repr(v) if (v != v or v in (float("inf"), float("-inf"))) else float(f"{v:.10g}")
    if isinstance(value, np.ndarray):
        flat = np.ma.filled(np.ma.asarray(value, dtype=float), np.nan)
        return {"shape": list(flat.shape), "values": [_plain(v) for v in flat.ravel()]}
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return str(value)

def _drawn(fig) -> dict:
    """What a figure draws, as plain data, as far as the entries below go: for every line, collection,
    patch, text, image and table on every axes its numbers, widths, dashes, markers, transparency,
    label, drawing order and colour-map name, the frame a text is placed in and whether a marker has an
    edge; each axes' title, labels, scales, limits and legend words; and NO colour. Read off the
    artists, never off pixels, so the page and the lettering do not enter it. It is not the whole
    picture: beyond colours it leaves out the ticks and their labels, the grid, a legend's place,
    columns and letter size, a table's letter size, the width of an outline effect, the linear range of
    a symmetric log axis, a line's dash given as numbers, the edge width of a line's markers, a text's
    rotation and weight, an image's smoothing, an axes switched off, where the axes sit on the page, a
    3-D surface's corner points, and a 3-D artist's drawing order and projected paths, which are set
    only when it is drawn."""
    import numpy as np
    from matplotlib.collections import PathCollection

    def common(a):
        solid = hasattr(a, "do_3d_projection")      # a 3-D artist's drawing order is set when it is drawn
        return {"kind": type(a).__name__, "label": a.get_label(), "alpha": a.get_alpha(),
                "order": None if solid else a.get_zorder(),
                "effects": [type(e).__name__ for e in (a.get_path_effects() or [])]}

    def line(ln):
        data = ln.get_data_3d() if hasattr(ln, "get_data_3d") else ln.get_data(orig=False)
        return dict(common(ln), data=[np.asarray(d, dtype=float) for d in data], style=ln.get_linestyle(),
                    width=ln.get_linewidth(), marker=str(ln.get_marker()), marker_size=ln.get_markersize(),
                    steps=ln.get_drawstyle())

    def collection(c):
        out = dict(common(c), widths=np.asarray(c.get_linewidths(), dtype=float), styles=c.get_linestyles())
        if hasattr(c, "get_coordinates"):                       # a colour mesh: its corners
            out["mesh"] = np.asarray(c.get_coordinates())
        elif not hasattr(c, "do_3d_projection"):                # a 3-D artist's paths are its projection
            out["offsets"] = np.asarray(c.get_offsets())
            out["paths"] = [np.asarray(p.vertices) for p in c.get_paths()]
            if isinstance(c, PathCollection):                   # markers: is an edge drawn round each?
                out["edged"] = len(c.get_edgecolor()) > 0       # whether an edge colour is set, never which
        if hasattr(c, "get_sizes"):
            out["sizes"] = np.asarray(c.get_sizes(), dtype=float)
        if c.get_array() is not None:
            out.update(values=np.asarray(c.get_array()), limits=c.get_clim(), colour_map=c.get_cmap().name)
        return out

    def patch(p):
        where = p.get_bbox().bounds if hasattr(p, "get_bbox") else np.asarray(p.get_path().vertices)
        return dict(common(p), where=where, width=p.get_linewidth(), style=p.get_linestyle(), hatch=p.get_hatch())

    def text(t):
        box, frame = t.get_bbox_patch(), t.get_transform()      # `at` is two numbers in that frame
        return dict(common(t), text=t.get_text(), at=t.get_position(), size=t.get_fontsize(),
                    face=t.get_fontfamily(), align=[t.get_ha(), t.get_va()],
                    within="axes" if frame is t.axes.transAxes else "data" if frame is t.axes.transData else "other",
                    box=None if box is None else [type(box.get_boxstyle()).__name__, box.get_alpha()])

    def image(im):
        return dict(common(im), values=np.asarray(im.get_array()), limits=im.get_clim(),
                    colour_map=im.get_cmap().name, extent=im.get_extent())

    def axes(ax):
        legend = ax.get_legend()
        out = {"title": ax.get_title(), "x": [ax.get_xlabel(), ax.get_xscale(), ax.get_xlim()],
               "y": [ax.get_ylabel(), ax.get_yscale(), ax.get_ylim()],
               "lines": [line(a) for a in ax.lines], "collections": [collection(a) for a in ax.collections],
               "patches": [patch(a) for a in ax.patches], "texts": [text(a) for a in ax.texts],
               "images": [image(a) for a in ax.images],
               "tables": [{f"{r},{c}": cell.get_text().get_text() for (r, c), cell in sorted(t.get_celld().items())}
                          for t in ax.tables],
               "legend": None if legend is None else [t.get_text() for t in legend.get_texts()]}
        if hasattr(ax, "get_zlim"):
            out["z"] = [ax.get_zlabel(), ax.get_zlim()]
        return out

    return _plain({"size": fig.get_size_inches(), "heading": fig.get_suptitle(),
                   "axes": [axes(ax) for ax in fig.axes]})

def _digest(record) -> str:
    import hashlib
    import json
    return hashlib.sha256(json.dumps(record, sort_keys=True).encode("utf-8")).hexdigest()[:16]

def test_the_record_of_a_figure_holds_everything_but_its_colours():
    """The record the next test compares must see a changed number, width, dash, marker, transparency
    or label, the frame a text is placed in and whether a marker has an edge, and must not see a
    changed colour, the colour of a marker's edge included; and two builds of one figure must give one
    record."""
    import numpy as np
    from matplotlib.figure import Figure

    def build(colour="steelblue", width=1.0, dash="--", alpha=0.25, label="band", lift=0.0, marker="o",
              frame="data", edge=None):
        t = np.linspace(0.0, 1.0, 8)
        ax = Figure().add_subplot()
        ax.fill_between(t, t - 0.1, t + 0.1, alpha=alpha, color=colour, label=label)
        ax.plot(t, t + lift, color=colour, linewidth=width, linestyle=dash)
        ax.plot(t, 1.0 - t, marker=marker, linestyle="none", color=colour)
        ax.scatter(t[:2], t[:2], c=colour, s=40, edgecolors=edge)
        ax.axvspan(0.2, 0.4, color=colour, alpha=0.1)
        ax.text(0.5, 0.5, "note", color=colour, transform=ax.transAxes if frame == "axes" else ax.transData)
        return _digest(_drawn(ax.figure))

    same = build()
    assert build() == same, "two builds of one figure differ: the record is not repeatable"
    assert build(colour="#0072B2") == same, "a colour reached the record"
    assert build(edge="black") == same, "the colour of a marker's edge reached the record"
    for change in (dict(width=1.1), dict(dash="-."), dict(alpha=0.3), dict(label="Band"),
                   dict(lift=1e-6), dict(marker="."), dict(frame="axes"), dict(edge="none")):
        assert build(**change) != same, f"the record does not see {change}"

# What each plotting function drew on the tree as it stood before any of them took its colours from
# one shared module: one digest a figure, of `_drawn`. Tied to the pinned matplotlib: an upgrade of it
# is recorded again, on a tree whose plotting files are untouched.
_DRAWN_FIGURES: dict = {
    "plot": "6408809307b6b286",
    "plot_ppc": "11f120327b1ec853",
    "plot_posterior_vs_truth": "d56bddd460aafd8b",
    "plot_best_fit_overlay": "29e93c12a16f8952",
    "plot_overlay_band": "20d7569f8824084b",
    "plot_psd_overlay": "e8acf81a0b10a512",
    "plot_psd_overlay, nothing finite": "0f3259c78c4187f6",
    "plot_cycle_average": "c549495d1076b0e3",
    "plot_training_loss": "2695bad252b76e82",
    "plot_eff_temp_ratio": "12c9e81bbcd22e5d",
    "plot_spontaneous_trajectory": "b8f3c318519c9754",
    "plot_psd": "0780c351b338e84c",
    "plot_chi_components": "56ad3b53ada40e13",
    "FDT ratio across repeats": "c76a4a21722c902a",
    "Sweep surfaces": "efcb50c9d30b7a0e",
    "Sweep slice": "911994075a7ab1a7",
    "plot_fdt_3d_vs_param": "753e7a3f5c7b0dbd",
    "plot_sweep_summary": "c020e21966f75c9b",
    "plot_cross_validation_3d": "8a10155901ca25c3",
    "Jacobian cosine matrix": "a556abf4f111a624",
    "Jacobian singular spectrum": "4eb4bc606a9a42d3",
    "Probe drive own-peak ratio": "97633d5b965d5090"
}

# Words each figure shows only when the call behind them ran: the proof that the inputs below open
# every branch that names a colour.
_MUST_SHOW = {
    "plot": ("LineCollection",),
    "plot_ppc": ("Invalid (zero variance)",),
    "plot_posterior_vs_truth": ("Posterior 95% band", "Posterior samples", "Posterior median", "Posterior mean"),
    "plot_psd_overlay, nothing finite": ("no finite posterior-predictive PSD",),
    "plot_training_loss": ("training loss", "early-stop window"),
    "plot_eff_temp_ratio": ("approx",),
    "plot_spontaneous_trajectory": ("burn-in ends",),
    "plot_psd": ("probe band", "approx"),
    "plot_chi_components": ("approx",),
    "FDT ratio across repeats": ("spread across the repeats", "mean of the repeats"),
    "plot_fdt_3d_vs_param": ("Omega_0 = 1",),
    "plot_cross_validation_3d": ("Hopf manifold",),
    "Probe drive own-peak ratio": ("free-running at or above", "captured at or below"),
}

def test_what_each_plotting_function_draws_is_unchanged(tmp_path, monkeypatch):
    """Every plotting function of the engine that names a colour, called with fixed inputs under
    matplotlib's own settings, draws what it drew when `_DRAWN_FIGURES` was written, as far as `_drawn`
    records a figure: the same numbers, widths, dashes, markers, transparencies, labels, text frames
    and marker edges. Colours are not compared, so a change of a line's colour alone passes, and a
    change of anything else `_drawn` records fails; `_drawn` says what it leaves out."""
    import json
    from types import SimpleNamespace

    import matplotlib
    import numpy as np
    import pandas as pd
    from matplotlib import pyplot as plt

    from core.artifacts import ArtifactStore
    from core.diagnostics import identifiability, probes
    from core.FDT import compare, cross_validation_plots, plots
    from core.Helpers import visualizers
    from core.Reduction import plots as reduction_plots
    from tests._fixtures import build_fdt_record

    drawn = {}

    def keep(name, fig):
        assert name not in drawn, name
        drawn[name] = _drawn(fig)
        plt.close(fig)

    with _rcparams_guard():
        matplotlib.rcdefaults()              # matplotlib's own settings, whatever an earlier test left
        t = np.linspace(0.0, 1.0, 64)
        wave = np.sin(6.0 * t)

        # core/Helpers/visualizers.py
        visualizers.plot(t, wave, hlines=([0.5], 0.0, 1.0), sink=lambda _title, fig: keep("plot", fig))
        keep("plot_ppc", visualizers.plot_ppc({"z_scores": np.array([0.1, 1.5, 2.5, np.nan]),
                                               "coverage_90": 0.9, "num_outside": 1, "num_invalid": 1}))
        samples = np.stack([np.sin(6.0 * t + 0.1 * k) for k in range(12)])
        keep("plot_posterior_vs_truth", visualizers.plot_posterior_vs_truth(
            t, wave, x_mean=samples.mean(axis=0), x_median=np.median(samples, axis=0), x_samples=samples))
        keep("plot_best_fit_overlay", visualizers.plot_best_fit_overlay(
            t, wave, 1.02 * wave, param_labels=["a", "b"], param_values=[1.0, 2.0], criterion="closest"))
        keep("plot_overlay_band", visualizers.plot_overlay_band(t, wave, wave - 0.2, wave, wave + 0.2, n_used=5))
        freqs = np.linspace(0.0, 10.0, 16)
        power, blank = 1.0 / (1.0 + freqs ** 2), np.full(16, np.nan)
        keep("plot_psd_overlay", visualizers.plot_psd_overlay(freqs, power, 0.5 * power, 0.9 * power, 2.0 * power))
        keep("plot_psd_overlay, nothing finite",
             visualizers.plot_psd_overlay(freqs, power, blank, blank, blank, n_dropped=3))
        phase = np.linspace(0.0, 2.0 * np.pi, 24)
        cycle = 0.9 * np.sin(phase)
        keep("plot_cycle_average", visualizers.plot_cycle_average(phase, np.sin(phase), cycle, cycle - 0.1, cycle + 0.1))
        keep("plot_training_loss", visualizers.plot_training_loss(
            {"training_loss": [1.0, 0.7, 0.5, 0.45, 0.44], "validation_loss": [1.1, 0.8, 0.6, 0.62, 0.63],
             "stop_after_epochs": 2, "epochs_trained": 5, "best_validation_loss": 0.6}))

        # core/FDT/plots.py: each saves through pyplot and then closes its figure, so the figure is
        # taken at the save, and nothing is written; the path they are handed is in this test's own
        # folder all the same, so a save by another way could not land in the working directory
        caught = []
        monkeypatch.setattr(plots.plt, "savefig", lambda *a, **k: caught.append(plots.plt.gcf()))
        unwritten = tmp_path / "unwritten.png"
        om = np.array([0.5, 1.0, 2.0, 4.0])
        plots.plot_eff_temp_ratio(om, np.array([0.5, 1.0, -2.0, 3.0]), save_path=unwritten, omega_natural=1.0)
        keep("plot_eff_temp_ratio", caught.pop())
        plots.plot_spontaneous_trajectory(t, wave, save_path=unwritten, burn_in=0.25)
        keep("plot_spontaneous_trajectory", caught.pop())
        welch = np.linspace(0.0, 2.0 * np.pi, 9)
        plots.plot_psd(welch, 1.0 / (1.0 + (welch - np.pi) ** 2), save_path=unwritten,
                       omega_natural=np.pi, plot_band=(8.0, 30.0))
        keep("plot_psd", caught.pop())
        plots.plot_chi_components(om, np.array([1.0 + 0.5j, 0.5 - 0.2j, -0.3 + 0.1j, 0.1 + 0.4j]),
                                  save_path=unwritten, omega_natural=1.0)
        keep("plot_chi_components", caught.pop())

        # core/FDT/compare.py, through the comparison's own entry, on records written in seconds
        store = ArtifactStore(tmp_path / "artifacts")
        repeats = [build_fdt_record(store, name=f"rep{i}", omegas=(1.0, 2.0, 4.0), ratio=r, seed=i)
                   for i, r in enumerate([(1.0, 3.0, 1.0), (1.2, 5.0, 1.1), (0.8, 4.0, 0.9)])]
        compare.compare("repeats", repeats, fig_sink=keep, store=store)
        sweeps = [build_fdt_record(store, name=name, study="sweep", omegas=(0.5, 1.0, 2.0), sweep_param="s",
                                   points=points)
                  for name, points in (("s_low", [(0.0, (1.0, 1.0, 1.0)), (0.5, (1.0, 3.0, 1.1))]),
                                       ("s_high", [(0.25, (1.0, 2.0, 1.0)), (0.75, (1.0, 6.0, 1.2))]))]
        compare.compare("sweeps", sweeps, fig_sink=keep, store=store)

        # the three that return nothing and leave their figure open: it is pyplot's current one
        cross_validation_plots.plot_fdt_3d_vs_param(
            [{"failed": False, "omega_norm": om, "param_value": v, "T_eff_over_T": v + 1.0 / om}
             for v in (0.0, 0.5, 1.0)], r"$S$", "Sweep", None)
        keep("plot_fdt_3d_vs_param", plt.gcf())
        table = pd.DataFrame({"failed": [False] * 4, "f_max": [1.0, 2.0, 3.0, 4.0], "mu_N": [-0.2, -0.1, 0.1, 0.2],
                              "Omega0_N": [1.0, 1.5, 2.0, 2.5], "alpha_H_N": [0.3, 0.2, -0.1, -0.2],
                              "beta_H_N": [0.1, 0.0, -0.1, -0.2],
                              "regime": ["subcritical", "subcritical", "supercritical", "supercritical"]})
        reduction_plots.plot_sweep_summary(table, save=False)
        keep("plot_sweep_summary", plt.gcf())
        reduction_plots.plot_cross_validation_3d(
            table, [{"f_max": f, "omega_grid": om, "T_eff_over_T": f + om} for f in (1.0, 2.0, 3.0, 4.0)], save=False)
        keep("plot_cross_validation_3d", plt.gcf())

        # core/diagnostics: the two figure builders, called as their diagnostics call them
        J = np.array([[1.0, 0.2, 0.0], [0.5, 1.0, 0.1], [0.1, 0.3, 1.0], [0.9, 0.1, 0.4], [0.2, 0.8, 0.6], [0.3, 0.5, 0.7]])
        identifiability._summaries(SimpleNamespace(feat_labels=[f"feature {i}" for i in range(6)]), J, np.ones(6),
                                   np.zeros(6, dtype=bool), ["alpha", "beta", "gamma"], ["central"] * 3,
                                   [1.0] * 3, 0.05, keep)
        probes._drive_figure(keep, [0.01, 0.1, 1.0], [0.9, None, 0.05], 0.70, 0.10)

    assert len(drawn) == 22, sorted(drawn)
    for name, words in _MUST_SHOW.items():
        shown = json.dumps(drawn[name])
        for word in words:
            assert word in shown, f"{name}: the call that draws {word!r} was not reached"
    assert [len(a["patches"]) for a in drawn["plot_sweep_summary"]["axes"]] == [2, 2, 2, 2], \
        "the reduction map's two shaded ranges were not both drawn"
    got = {name: _digest(record) for name, record in drawn.items()}
    (tmp_path / "drawn.json").write_text(json.dumps(drawn, indent=1, sort_keys=True), encoding="utf-8")
    assert got == _DRAWN_FIGURES, (
        "what the plotting functions draw has changed (colours are not compared). Every record is in "
        f"{tmp_path / 'drawn.json'}. The table as this run found it:\n{json.dumps(got, indent=4)}")

def test_every_plotting_function_draws_under_the_windows_figure_settings(tmp_path, monkeypatch):
    """The window draws the engine's figures under its own settings, which hold seven line colours
    where matplotlib's own hold ten, another lettering face and no top or right spine, and the test
    above draws them under matplotlib's own alone. Here each of its figures is built and then rendered
    under the window's settings, and must do so without raising. What is drawn is not compared.

    The cases are the body of the test above and are not written a second time: that test is run
    again with four of the names it reads standing for something else. Its return to matplotlib's own
    settings goes on to write the window's; its record of a figure also renders the figure; and its
    digest of a record is one fixed word against a table of that word, so that its last comparison
    says only that every figure the table names was drawn."""
    import io
    import matplotlib
    from matplotlib import pyplot as plt
    from core.gui import mpl_theme

    names, record, own_settings = list(_DRAWN_FIGURES), _drawn, matplotlib.rcdefaults
    windows, rendered, open_before = mpl_theme._rc(), [], plt.get_fignums()

    def the_windows_settings():
        own_settings()
        mpl_theme.apply_mpl_theme()

    def record_and_render(fig):
        elsewhere = [name for name, value in windows.items() if matplotlib.rcParams[name] != value]
        assert not elsewhere, f"not drawn under the window's figure settings: {elsewhere}"
        what = record(fig)
        picture = io.BytesIO()
        fig.savefig(picture, format="png")
        rendered.append(picture.tell())
        return what

    monkeypatch.setattr(matplotlib, "rcdefaults", the_windows_settings)
    monkeypatch.setitem(globals(), "_drawn", record_and_render)
    monkeypatch.setitem(globals(), "_digest", lambda _record: "drawn")
    monkeypatch.setitem(globals(), "_DRAWN_FIGURES", dict.fromkeys(names, "drawn"))
    with _rcparams_guard():
        test_what_each_plotting_function_draws_is_unchanged(tmp_path, monkeypatch)
    assert len(rendered) == len(names) and all(rendered), \
        f"{len(names)} figures named, and the pictures rendered hold {rendered} bytes"
    assert plt.get_fignums() == open_before, "a figure was left open"
