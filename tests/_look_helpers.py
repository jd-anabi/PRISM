"""The look suite's helpers for setting the theme and the text size, for drawing into a picture and reading
it back, for moving the focus and for letting a widget go: kept here, where a suite of the window can use
them too, and imported back by tests/test_look.py under the same names.

This file imports the toolkit, the look's table and the standard library, and nothing else -- no torch, no
part of the program's engine, never tests/_fixtures.py -- so a suite that may not load torch can use it; the
look's fresh-interpreter test reads every import in it. Importing it loads the toolkit. One application
object serves every suite of a run, so each helper that changes something of it puts back what it found.
"""
import contextlib
import math

from PySide6.QtCore import QCoreApplication, QEvent, QObject, QPoint, QPointF, Qt
from PySide6.QtGui import QColor, QFocusEvent, QImage, QKeyEvent, QMouseEvent, QPainter, QPalette, QRegion
from PySide6.QtWidgets import QApplication, QWidget

from core.gui import design


@contextlib.contextmanager
def theme(dark):
    """The theme in force inside the block; what was in force before is put back."""
    was = design.is_dark()
    design.set_dark(dark)
    try:
        yield
    finally:
        design.set_dark(was)


@contextlib.contextmanager
def text_size(factor):
    """The text factor in force inside the block; what was in force before is put back."""
    was = design.text_scale()
    design.set_text_scale(factor)
    try:
        yield
    finally:
        design.set_text_scale(was)


def _image(width, height, ratio):
    """A clear picture of that many logical pixels at a display scaling, its size rounded up."""
    picture = QImage(math.ceil(width * ratio), math.ceil(height * ratio), QImage.Format_ARGB32_Premultiplied)
    picture.setDevicePixelRatio(ratio)
    picture.fill(Qt.transparent)
    return picture


def _at(picture, x, y):
    """The colour, as the look's table writes it, of the device pixel under a logical point."""
    ratio = picture.devicePixelRatio()
    return picture.pixelColor(int(x * ratio), int(y * ratio)).name().upper()


def _moved_on(widget, reason):
    """The focus arrives with no press of its own, as when the toolkit moves it on from a greyed control."""
    QApplication.sendEvent(widget, QFocusEvent(QEvent.FocusIn, reason))


def _arrive(widget, reason):
    """The focus arrives as a person moves it: first the press, which the application alone hears (a key for
    the keyboard's three reasons, a mouse button for a click), then the focus itself."""
    if reason in (Qt.TabFocusReason, Qt.BacktabFocusReason, Qt.ShortcutFocusReason):
        QApplication.sendEvent(QObject(), QKeyEvent(QEvent.KeyPress, Qt.Key_Tab, Qt.NoModifier))
    elif reason == Qt.MouseFocusReason:
        QApplication.sendEvent(QObject(), QMouseEvent(QEvent.MouseButtonPress, QPointF(1, 1), QPointF(1, 1),
                                                      Qt.LeftButton, Qt.LeftButton, Qt.NoModifier))
    _moved_on(widget, reason)


_BLANK = QColor(255, 0, 255)                         # a ground no colour of the look can be taken for


def _screen(width, height, ratio, ground=_BLANK, draw=None):
    """A picture standing in for a screen at one scaling, on a ground of one colour, with
    ``draw(painter)`` painted into it. The painter is ended whatever ``draw`` does: a picture that
    goes while a painter still paints on it ends the whole run, where the test should only fail."""
    image = _image(width, height, ratio)             # its size is rounded UP there: rounded down, a last row is cut
    image.fill(ground)
    if draw is not None:
        painter = QPainter(image)
        try:
            draw(painter)
        finally:
            painter.end()
    return image


def _screen_of(widget, ratio, ground=_BLANK):
    """A widget painted into such a picture, with no ground of its own under it."""
    image = _screen(widget.width(), widget.height(), ratio, ground)
    widget.render(image, QPoint(0, 0), QRegion(), QWidget.RenderFlag.DrawChildren)
    return image


def _settle():
    for _ in range(4):
        QApplication.sendPostedEvents()
        QApplication.processEvents()


@contextlib.contextmanager
def _the_look(dark, sheet=True):
    """One theme in force, its palette on the application and its sheet too unless that is left out; all three
    put back, whatever the block does."""
    app = QApplication.instance() or QApplication([])
    was_sheet, was_palette = app.styleSheet(), QPalette(app.palette())
    with theme(dark):
        try:
            app.setPalette(design.build_palette(dark))
            if sheet:
                app.setStyleSheet(design.build_qss(dark))
            yield
        finally:
            app.setStyleSheet(was_sheet)
            app.setPalette(was_palette)


def _gone(widget):
    """Delete a widget now. Asked only to delete itself later it would stay to the end of the run: no event
    loop turns between two tests. This widget's own request is delivered, and nobody else's."""
    widget.deleteLater()
    QCoreApplication.sendPostedEvents(widget, QEvent.DeferredDelete)
