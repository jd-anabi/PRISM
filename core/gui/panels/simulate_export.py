"""Worker-thread side of the Simulate section's "Save video…" export.

Re-renders a recorded run's ``[t_seconds, x_displacement]`` series into a smooth animation (a scrolling
trace over the top-down ellipse heatmap) with matplotlib (Agg -- worker-safe, no pyplot/Gcf) and streams
frames to imageio: ``.gif`` via Pillow, ``.mp4`` via imageio-ffmpeg's ffmpeg.

It is a RE-RENDER, not a screen-grab: the live stream only emits a handful of coarse chunks, so grabbing
the widget per chunk would give a ~10-frame choppy video. Sweeping the fine display series at the chosen
fps (``export_stride``) yields a smooth, real-time animation. The render is matplotlib-styled, so it is
not pixel-identical to the pyqtgraph live view (and animates smoother) -- intended.

Agg rendering on a worker thread is safe: GOTCHA #2 only forbids painting a worker-built figure in a
LIVE canvas; here we render to an Agg buffer and hand bytes to imageio.
"""
from __future__ import annotations

import os

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from tqdm import tqdm

from core.config import DT_EXP_S

from ..widgets.live_hair_bundle import gaussian_field, hb_center


def export_stride(sample_rate_hz: float, video_fps: float) -> int:
    """Fine samples to advance per video frame for real-time playback at ``video_fps``.

    The display series is at ``sample_rate_hz`` (= 1/DT_EXP_S = 1000 Hz), so at 30 fps this is ~33."""
    if video_fps is None or video_fps <= 0:
        return 1
    return max(1, round(sample_rate_hz / video_fps))


def estimate_frame_count(n_samples: int, stride: int) -> int:
    """How many video frames ``export_animation`` will write for a series of ``n_samples``."""
    return len(range(stride - 1, n_samples, stride))


def ffmpeg_available() -> bool:
    """True if an ffmpeg binary is resolvable (needed for .mp4; .gif never needs it). Lets the panel
    pre-check on the GUI thread and show a friendly message instead of dispatching a doomed export."""
    try:
        import imageio_ffmpeg
        imageio_ffmpeg.get_ffmpeg_exe()
        return True
    except Exception:                                # noqa: BLE001
        return False


def _open_writer(path: str, video_fps: float):
    """A streaming imageio writer chosen by extension. Raises a friendly error for .mp4 without ffmpeg."""
    import imageio

    ext = os.path.splitext(path)[1].lower()
    if ext == ".mp4":
        try:
            import imageio_ffmpeg
            imageio_ffmpeg.get_ffmpeg_exe()          # raises if no ffmpeg binary is resolvable
        except Exception as e:                       # noqa: BLE001 -- surface a clear, actionable message
            raise RuntimeError(
                "Saving MP4 needs an ffmpeg binary, which isn't available here. Save as .gif instead, "
                "or install ffmpeg (or set the IMAGEIO_FFMPEG_EXE environment variable).") from e
        return imageio.get_writer(path, fps=float(video_fps), codec="libx264", macro_block_size=16)
    # GIF (Pillow): per-frame duration in MILLISECONDS (Pillow's native unit, imageio >= 2.28).
    return imageio.get_writer(path, mode="I", duration=1000.0 / float(video_fps), loop=0)


def export_animation(series, path, *, window_pts, grid_x, grid_y, sigma_x, sigma_y, aspect, margin,
                     video_fps, x_unit="nm", sample_rate_hz=1.0 / DT_EXP_S,
                     figsize=(6.4, 4.8), dpi=100) -> str:
    """Render ``series`` (an ``(M, 2)`` ``[t_seconds, x]`` array) to a gif/mp4 at ``path``.

    Reproduces ``LiveHairBundleView``: the trace is the last ``window_pts`` samples up to the sweep index
    (y-range fixed to the whole recording); the heatmap is ``gaussian_field`` at the mapped center, with
    x0 normalized against the SAME sliding window as the live view. Geometry params come straight off the
    live view so the two match. Cancellation rides the per-frame tqdm redraw (the app's write() cancel
    checkpoint); a cancel/error removes the partial file.
    """
    series = np.asarray(series)
    if series.ndim != 2 or series.shape[0] == 0:
        raise ValueError("Nothing to export: the recording is empty.")
    t_all, x_all = series[:, 0], series[:, 1]
    n = series.shape[0]
    stride = export_stride(sample_rate_hz, video_fps)
    grid_x, grid_y = np.asarray(grid_x), np.asarray(grid_y)

    fig = Figure(figsize=figsize, dpi=dpi)                    # 640x480 @ dpi 100: even + a multiple of 16
    canvas = FigureCanvasAgg(fig)
    ax_tr = fig.add_subplot(2, 1, 1)
    ax_tr.set_title("Hair-bundle displacement")
    ax_tr.set_xlabel("t (s)")
    ax_tr.set_ylabel(f"x ({x_unit})")
    (line,) = ax_tr.plot([], [], lw=1.0)

    ax_hm = fig.add_subplot(2, 1, 2)
    ax_hm.set_title("Top-down hair bundle")
    ax_hm.set_xticks([])
    ax_hm.set_yticks([])
    field0 = gaussian_field(hb_center(0.5, aspect, margin), 0.5, grid_x, grid_y, sigma_x, sigma_y)
    im = ax_hm.imshow(field0.T, origin="lower", extent=[0.0, aspect, 0.0, 1.0], aspect="equal",
                      cmap="inferno", vmin=0.0, vmax=1.0)    # .T + origin='lower' matches the ImageItem
    fig.tight_layout()

    # Fixed y-range for the trace (stable axis across the whole video), with a small pad.
    xlo, xhi = float(np.min(x_all)), float(np.max(x_all))
    if xhi <= xlo:
        xhi = xlo + 1.0
    pad = 0.05 * (xhi - xlo)
    ax_tr.set_ylim(xlo - pad, xhi + pad)

    writer = _open_writer(path, video_fps)
    ok = False
    try:
        for i in tqdm(range(stride - 1, n, stride), desc="Exporting animation"):
            lo_i = max(0, i - window_pts + 1)
            wt, wx = t_all[lo_i:i + 1], x_all[lo_i:i + 1]
            line.set_data(wt, wx)
            t0, t1 = float(wt[0]), float(wt[-1])
            ax_tr.set_xlim(t0, t1 if t1 > t0 else t0 + 1e-9)

            lo, hi = float(wx.min()), float(wx.max())
            x0 = float(wx[-1])
            x0n = 0.5 if hi <= lo else (x0 - lo) / (hi - lo)
            x0n = min(1.0, max(0.0, x0n))
            im.set_data(gaussian_field(hb_center(x0n, aspect, margin), 0.5,
                                       grid_x, grid_y, sigma_x, sigma_y).T)

            canvas.draw()
            rgba = np.asarray(canvas.buffer_rgba())
            h, w = rgba.shape[0], rgba.shape[1]
            # [..., :3] is a non-contiguous view; imageio pipes raw bytes, so make it contiguous. Crop to
            # even dims defensively (H.264 needs even width/height).
            rgb = np.ascontiguousarray(rgba[:h - h % 2, :w - w % 2, :3])
            writer.append_data(rgb)
        ok = True
    finally:
        try:
            writer.close()                                   # closing a 0-frame writer can itself raise
        except Exception:                                    # noqa: BLE001 -- never mask the real outcome
            pass
        if not ok:                                           # cancel/error -> don't leave a partial file
            try:
                os.remove(path)
            except OSError:
                pass
    return path
