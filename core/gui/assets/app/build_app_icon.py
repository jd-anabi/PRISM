"""Render ``prism.svg`` to the committed PNG set used as the application/window icon.
Run:  python core/gui/assets/app/build_app_icon.py

WHY A PNG SET AND NOT THE SVG. Qt's `svg` IMAGE-FORMAT PLUGIN is absent in this environment --
``QImageReader.supportedImageFormats()`` lists no "svg" -- so ``QIcon("prism.svg")`` returns a null
icon and the window silently keeps Qt's default mark, with nothing logged. The QtSvg *module*
(QSvgRenderer) is a different thing and is available, which is what lets this script rasterise
offline. So: SVG is the editable source, PNGs are what ships, exactly the split the icon FONT uses
(assets/icons/build_prism_icons.py authors glyphs, the .ttf is committed).

Several sizes rather than one: Qt picks the nearest and scales, and a 256 -> 16 downscale of a
stroked mark turns to mush. Regenerate after editing the SVG; commit the PNGs.

AND A .ICO, assembled from those PNGs (``build_ico``; ``--ico-only`` skips the rasterising). The
Windows taskbar draws the window CLASS icon whenever its icon query is not answered in time, and a
class icon is a Win32 HICON, which LoadImage can only read from an .ico or an exe resource --
never from a PNG. app_icon.set_windows_class_icon loads this file. Same frames as the PNG set, so
the title bar and the taskbar cannot disagree (a test pins the 256 frame to prism-256.png).
"""
import sys
from pathlib import Path

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QImage, QPainter, QColor
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtWidgets import QApplication

SIZES = (16, 24, 32, 48, 64, 128, 256)
HERE = Path(__file__).resolve().parent


def build_ico(out_dir: Path = None) -> Path:
    """Assemble ``prism.ico`` from the committed ``prism-<n>.png`` frames (Pillow, transitive via
    matplotlib). Every size is a provided frame, none is a resize."""
    from PIL import Image
    out_dir = out_dir or HERE
    frames = [Image.open(out_dir / f"prism-{n}.png").convert("RGBA") for n in SIZES]
    out = out_dir / "prism.ico"
    # Pillow writes the first image and takes each requested size from append_images when a frame of
    # that size is provided (resizing only what is missing, which here is nothing).
    frames[-1].save(out, format="ICO", sizes=[(n, n) for n in SIZES], append_images=frames[:-1])
    return out


def build(svg_path: Path = None, out_dir: Path = None) -> list:
    svg_path = svg_path or (HERE / "prism.svg")
    out_dir = out_dir or HERE
    renderer = QSvgRenderer(str(svg_path))
    if not renderer.isValid():
        raise SystemExit(f"QSvgRenderer could not parse {svg_path}")
    written = []
    for n in SIZES:
        img = QImage(n, n, QImage.Format_ARGB32)
        img.fill(QColor(0, 0, 0, 0))                      # transparent: the tile's rx corners show
        p = QPainter(img)
        p.setRenderHint(QPainter.Antialiasing, True)
        p.setRenderHint(QPainter.SmoothPixmapTransform, True)
        renderer.render(p, QRectF(0, 0, n, n))
        p.end()
        out = out_dir / f"prism-{n}.png"
        if not img.save(str(out), "PNG"):
            raise SystemExit(f"failed to write {out}")
        written.append(out)
    return written


if __name__ == "__main__":
    if "--ico-only" not in sys.argv:
        app = QApplication(sys.argv)                      # QImage/QPainter need a QGuiApplication
        for p in build():
            print(f"wrote {p}")
    print(f"wrote {build_ico()}")
