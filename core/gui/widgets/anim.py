"""Snapshot-based screen / tab transitions.

A transition grabs the outgoing and incoming pages to static QPixmaps ONCE and animates those images in a
short-lived overlay, then deletes the overlay to reveal the real (already-current) widget underneath.

WHY SNAPSHOTS, NOT a QGraphicsOpacityEffect on the live widget: an opacity effect re-renders the whole
subtree to an offscreen pixmap EVERY frame, which stutters on the content-heavy pyqtgraph / matplotlib /
big-form pages here. Static pixmaps paint once per frame and never touch the live widget's graphicsEffect.

Callers change the logical state (setCurrentIndex + back-arrow sync) SYNCHRONOUSLY first, then start the
transition -- so index / gating / back-arrow state is correct the instant the call returns (the offscreen
test suite reads it with no event-loop pump). Everything here is SKIPPED under the offscreen platform and
when the container is not yet visible (construction-time nav), so the suite never animates.
"""
from PySide6.QtCore import Property, QEasingCurve, QPoint, QPointF, QPropertyAnimation, QRect, Qt
from PySide6.QtGui import QPainter, QPixmap
from PySide6.QtWidgets import QApplication, QWidget

_SCREEN_MS = 280       # slide + fade for top-level screen navigation
_TAB_MS = 180          # quick cross-fade for tab switches


def _animations_enabled(container) -> bool:
    """Animate only in a real, visible app -- never under the offscreen test platform, and not before the
    window is shown (MainWindow.__init__ navigates Home during construction)."""
    app = QApplication.instance()
    return app is not None and app.platformName() != "offscreen" and container.isVisible()


def snapshot(widget) -> QPixmap | None:
    """A DPR-aware pixmap of ``widget``, or None if it can't be grabbed (=> caller does an instant switch).

    Callers grab the OUTGOING page with this BEFORE switching (grab() renders even a just-hidden, sized
    widget). Every rendering surface in this app is raster (pyqtgraph default viewport; matplotlib shown as
    static QPixmaps), so grab() is reliable; a null/zero-size result still degrades gracefully to no anim.
    """
    if widget is None:
        return None
    pm = widget.grab()
    if pm.isNull() or pm.width() == 0 or pm.height() == 0:
        return None
    return pm


class _TransitionOverlay(QWidget):
    """Paints two page snapshots over the container while a ``progress`` 0->1 animation runs, then removes
    itself. ``slide`` picks a pure slide (screens) vs a cross-fade (tabs)."""

    def __init__(self, container, old_pm, new_pm, *, slide, forward=True):
        super().__init__(container)
        self._old = old_pm
        self._new = new_pm
        self._slide = slide
        self._dir = 1 if forward else -1
        self._p = 0.0
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)   # brief; don't eat clicks

    def _get_p(self):
        return self._p

    def _set_p(self, value):
        self._p = value
        self.update()

    progress = Property(float, _get_p, _set_p)

    def paintEvent(self, _event):
        p = self._p
        w = self.width()               # LOGICAL width -- never pm.width() (DPR would slide twice as far)
        painter = QPainter(self)
        if self._slide:
            # Pure slide: both pages move together at full opacity (they tile the viewport with no gap or
            # overlap), old exiting one side as new enters the other. No cross-fade.
            painter.drawPixmap(QPointF(-self._dir * p * w, 0.0), self._old)         # old slides out
            painter.drawPixmap(QPointF(self._dir * (1.0 - p) * w, 0.0), self._new)  # new slides in
        else:
            painter.setOpacity(1.0)
            painter.drawPixmap(QPointF(0.0, 0.0), self._old)                     # old static underneath
            painter.setOpacity(min(1.0, p))
            painter.drawPixmap(QPointF(0.0, 0.0), self._new)                     # new dissolves in on top
        painter.end()

    def run(self, ms, easing):
        anim = QPropertyAnimation(self, b"progress", self)     # parented to overlay => strong ref
        anim.setDuration(ms)
        anim.setStartValue(0.0)
        anim.setEndValue(1.0)
        anim.setEasingCurve(easing)
        anim.finished.connect(self.deleteLater)                # connect BEFORE start
        self.show()
        self.raise_()
        anim.start()


def _replace_active(container, overlay) -> None:
    """Keep only one live overlay per container: a fast second navigation cancels the first (no stacking)."""
    prev = getattr(container, "_transition", None)
    if prev is not None:
        try:
            prev.deleteLater()
        except RuntimeError:
            pass                                               # already destroyed
    container._transition = overlay


def slide_screens(stack, old_pm, forward, ms=_SCREEN_MS) -> None:
    """Slide the stack's now-current page into view (from the right if ``forward``, else the left)."""
    if old_pm is None or not _animations_enabled(stack):
        return
    new_pm = snapshot(stack.currentWidget())
    if new_pm is None:
        return
    overlay = _TransitionOverlay(stack, old_pm, new_pm, slide=True, forward=forward)
    overlay.setGeometry(stack.rect())
    _replace_active(stack, overlay)
    overlay.run(ms, QEasingCurve.OutCubic)


def crossfade_tab(tabwidget, old_widget, ms=_TAB_MS) -> None:
    """Cross-fade the tab CONTENT area from ``old_widget`` to the now-current page (the tab bar stays put)."""
    page = tabwidget.currentWidget()
    if old_widget is None or old_widget is page or not _animations_enabled(tabwidget):
        return
    old_pm = snapshot(old_widget)
    new_pm = snapshot(page)
    if old_pm is None or new_pm is None:
        return
    overlay = _TransitionOverlay(tabwidget, old_pm, new_pm, slide=False)
    # The page lives in QTabWidget's PRIVATE internal stack (below the tab bar), so page.geometry() is in
    # that stack's coords -- map to tabwidget coords to land on the real content area, not over the tab bar.
    overlay.setGeometry(QRect(page.mapTo(tabwidget, QPoint(0, 0)), page.size()))
    _replace_active(tabwidget, overlay)
    overlay.run(ms, QEasingCurve.InOutQuad)
