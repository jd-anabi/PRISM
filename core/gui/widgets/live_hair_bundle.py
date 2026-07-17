"""The live view for the Simulate section: a scrolling hair-bundle-displacement trace over a
"top-down hair bundle" 2D intensity-field heatmap, both rendered with pyqtgraph.

WHY pyqtgraph AND WHY THE GUI THREAD. GOTCHA #2 (a matplotlib Figure built on a worker thread must
never be painted by a live canvas -- it deadlocks on matplotlib's global lock) is matplotlib-specific.
pyqtgraph items are ordinary GUI-thread-owned Qt widgets, so the worker emits raw numpy frames over the
`chunk` signal and THIS widget updates on the GUI thread -- no worker-built drawable is ever painted.

THE HEATMAP is an anisotropic Gaussian -- an ellipse -- whose center tracks the instantaneous
displacement x0 (normalized against the visible trace window). The bundle deflects along one axis and is
wide across it, so the ellipse's MAJOR axis is perpendicular to the oscillation: displacement maps to a
horizontal slide, so the ellipse is elongated VERTICALLY (sigma_perp > sigma_par). Two things keep it
from clipping as it oscillates: the field is a WIDE rectangle (aspect > 1), and the center is mapped into
a horizontal MARGIN so it never reaches the field edge at the oscillation peaks. The field is recomputed
once per pushed frame (not per sample) over a precomputed grid, with fixed levels so there is no
per-frame autoscale flicker.
"""
import numpy as np
import pyqtgraph as pg


def gaussian_field(cx: float, cy: float, grid_x: np.ndarray, grid_y: np.ndarray,
                   sigma_x: float, sigma_y: float) -> np.ndarray:
    """An anisotropic 2D Gaussian (ellipse) peaked at ``(cx, cy)`` over the ``grid_x``×``grid_y`` grid.

    Pure and unit-testable. ``sigma_x``/``sigma_y`` are the along-axis widths (in grid-coordinate units),
    so ``sigma_x != sigma_y`` makes an ellipse whose major axis is the one with the LARGER sigma. Returns
    a ``(len(grid_x), len(grid_y))`` float32 field in [0, 1] (peak 1.0 at the center).
    """
    x = np.asarray(grid_x, dtype=np.float32).reshape(-1, 1)
    y = np.asarray(grid_y, dtype=np.float32).reshape(1, -1)
    d2 = ((x - cx) / sigma_x) ** 2 + ((y - cy) / sigma_y) ** 2
    return np.exp(-0.5 * d2).astype(np.float32)


def hb_center(x0_norm: float, aspect: float, margin: float) -> float:
    """Map a normalized displacement in [0, 1] to a horizontal blob center inside the safe margin.

    Shared by the live view and the video exporter so the two can't drift: the center stays in
    ``[margin, aspect - margin]`` (see ``LiveHairBundleView.__init__`` for how ``margin`` is sized to
    keep the whole blob on-field at the oscillation peaks)."""
    return margin + x0_norm * (aspect - 2.0 * margin)


class LiveHairBundleView(pg.GraphicsLayoutWidget):
    """A scrolling trace (top) + a top-down elliptical-blob heatmap (bottom).

    ``push(chunk)`` takes a ``(k, 2)`` array of ``[t_seconds, x_displacement]`` samples (as emitted by
    ``simulate_runner.run_simulation_stream``) and updates both plots; ``reset()`` clears them before a
    new run.

    ``aspect`` is the heatmap width:height (a wide rectangle gives the blob room to oscillate without
    clipping). ``sigma_par`` is the ellipse half-width ALONG the motion (horizontal) axis and
    ``sigma_perp`` the half-width PERPENDICULAR to it (vertical) -- keep ``sigma_perp > sigma_par`` so the
    major axis is perpendicular to the oscillation.
    """

    def __init__(self, window_pts: int = 2000, grid_n: int = 96, aspect: float = 2.6,
                 sigma_par: float = 0.10, sigma_perp: float = 0.20, x_unit: str = "nm", parent=None):
        super().__init__(parent)

        self._x_unit = x_unit
        self._trace_plot = self.addPlot(row=0, col=0, title="Hair-bundle displacement")
        self._trace_plot.setLabel("bottom", "t", units="s")            # pyqtgraph SI-prefixes (ms, µs)
        # Displacement unit goes in the label TEXT, not units= -- pyqtgraph would SI-prefix "nm" wrongly.
        self._trace_plot.setLabel("left", f"x ({x_unit})")
        self._trace_plot.showGrid(x=True, y=True, alpha=0.2)
        self._curve = self._trace_plot.plot(pen=pg.mkPen(width=1))

        self._hm_plot = self.addPlot(row=1, col=0, title="Top-down hair bundle")
        self._hm_plot.setAspectLocked(True)                  # square pixels -> renders as an aspect:1 rect
        self._hm_plot.hideAxis("bottom")
        self._hm_plot.hideAxis("left")
        self._img = pg.ImageItem()
        self._hm_plot.addItem(self._img)
        try:                                                 # a colormap is cosmetic; never fail on it
            lut = pg.colormap.get("inferno").getLookupTable(0.0, 1.0, 256)
            self._img.setLookupTable(lut)
        except Exception:                                    # noqa: BLE001
            pass

        # Precomputed heatmap grid: a WIDE field (x spans [0, aspect]) sampled with square pixels
        # (nx = ny*aspect), so the ellipse's sigma ratio renders true under the aspect lock.
        self._aspect = float(aspect)
        self._sig_x = float(sigma_par)                       # along motion (horizontal)
        self._sig_y = float(sigma_perp)                      # perpendicular (vertical) -> major axis
        nx = max(2, round(grid_n * self._aspect))
        self._grid_x = np.linspace(0.0, self._aspect, nx)
        self._grid_y = np.linspace(0.0, 1.0, grid_n)
        # Keep the whole blob (~3 sigma) inside the field horizontally, so it never clips at the peaks.
        self._margin = min(self._aspect / 2.0 - 1e-3, 3.5 * self._sig_x)

        self._w = int(window_pts)
        self._buf_t = np.zeros(self._w, dtype=np.float64)
        self._buf_x = np.zeros(self._w, dtype=np.float64)
        self._count = 0
        self.reset()

    def _cx(self, x0_norm: float) -> float:
        """Map a normalized displacement in [0, 1] to a horizontal center inside the safe margin."""
        return hb_center(x0_norm, self._aspect, self._margin)

    def _paint(self, x0_norm: float) -> None:
        self._img.setImage(
            gaussian_field(self._cx(x0_norm), 0.5, self._grid_x, self._grid_y, self._sig_x, self._sig_y),
            autoLevels=False, levels=(0.0, 1.0))

    def set_displacement_unit(self, unit: str) -> None:
        """Relabel the trace's y-axis with the cell's length unit (e.g. from cfg.length_unit)."""
        self._x_unit = unit or "nm"
        self._trace_plot.setLabel("left", f"x ({self._x_unit})")

    def reset(self) -> None:
        """Clear the trace and re-center the blob (called before each new stream)."""
        self._count = 0
        self._curve.setData(np.empty(0), np.empty(0))
        self._paint(0.5)

    def push(self, chunk) -> None:
        """Append one streamed frame and repaint. ``chunk`` is a ``(k, 2)`` [t, x] array."""
        if chunk is None:
            return
        arr = np.asarray(chunk)
        if arr.ndim != 2 or arr.shape[0] == 0:
            return
        t, x = arr[:, 0], arr[:, 1]
        self._append(t, x)

        n = self._count
        self._curve.setData(self._buf_t[:n], self._buf_x[:n])

        lo = float(self._buf_x[:n].min())
        hi = float(self._buf_x[:n].max())
        x0 = float(x[-1])
        x0_norm = 0.5 if hi <= lo else (x0 - lo) / (hi - lo)
        self._paint(min(1.0, max(0.0, x0_norm)))

    def _append(self, t: np.ndarray, x: np.ndarray) -> None:
        """Push (t, x) into the ring buffer, keeping the newest ``window_pts`` samples."""
        k = len(t)
        w = self._w
        if k >= w:
            self._buf_t[:] = t[-w:]
            self._buf_x[:] = x[-w:]
            self._count = w
            return
        keep = self._count
        if keep + k > w:                                 # drop the oldest to make room
            shift = keep + k - w
            self._buf_t[:keep - shift] = self._buf_t[shift:keep]
            self._buf_x[:keep - shift] = self._buf_x[shift:keep]
            keep -= shift
        self._buf_t[keep:keep + k] = t
        self._buf_x[keep:keep + k] = x
        self._count = keep + k
