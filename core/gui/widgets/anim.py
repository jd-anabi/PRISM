"""A subtle opacity fade for screen / tab transitions.

Purely VISUAL. Callers change the logical state (``setCurrentIndex``) SYNCHRONOUSLY first, then call
``fade_in`` on the newly-shown widget -- so the current index / gating / back-arrow state is already
correct the instant the call returns (the offscreen test suite reads it with no event-loop pump).

Two deliberate guards:
  * Skipped entirely under the ``offscreen`` QPA platform (the tests) -- keeps them deterministic and
    never composites an opacity effect over a headless pyqtgraph/matplotlib page.
  * The graphics effect is REMOVED when the animation finishes, so steady state is native/opaque. A
    QGraphicsOpacityEffect left permanently on a live pyqtgraph view repaints the whole subtree to an
    offscreen pixmap every frame; the effect must exist only for the ~150 ms transition.
"""
from PySide6.QtCore import QEasingCurve, QPropertyAnimation
from PySide6.QtWidgets import QApplication, QGraphicsOpacityEffect


def fade_in(widget, ms: int = 150):
    """Fade ``widget`` from transparent to opaque over ``ms`` ms. No-op under offscreen / no app.

    Returns the running QPropertyAnimation (parented to ``widget`` so it is not GC'd mid-flight), or
    ``None`` when skipped.
    """
    if widget is None:
        return None
    app = QApplication.instance()
    if app is None or app.platformName() == "offscreen":
        return None

    effect = QGraphicsOpacityEffect(widget)
    widget.setGraphicsEffect(effect)

    anim = QPropertyAnimation(effect, b"opacity", widget)   # parented to widget => strong ref
    anim.setDuration(ms)
    anim.setStartValue(0.0)
    anim.setEndValue(1.0)
    anim.setEasingCurve(QEasingCurve.InOutQuad)
    anim.finished.connect(lambda: widget.setGraphicsEffect(None))
    anim.start(QPropertyAnimation.DeleteWhenStopped)
    return anim
