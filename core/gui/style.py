"""The small style, the focus ring, and how a widget is given a class.

``PrismStyle`` is Fusion with the few things a style sheet cannot do: it paints the tick box and the radio
dot from the look's table, leaves the toolkit's dotted focus mark out, tells the focus frame how far outside
its control it stands, and lets the wheel pass over a closed drop-down. ``FocusRing`` is that frame, painted
as the one sign of keyboard focus. ``FocusFollower`` decides where it shows, from the reason the focus
arrived with and from whether a key or a mouse button was pressed last. ``set_class`` names a widget's
class for the style sheet.

What a widget may ask of the follower: the property ``ring`` with the value ``"inset"`` puts the ring just
inside the widget's own edge; the property ``ring_left`` is a number of pixels an inset ring leaves free at
the left; and a widget with a method ``set_focus_shown(on)`` shows keyboard focus itself, is told when to,
and gets no ring.
"""
import shiboken6
from PySide6.QtCore import QEvent, QObject, QRectF, Qt
from PySide6.QtGui import QPainter, QPen
from PySide6.QtWidgets import (QAbstractScrollArea, QAbstractSpinBox, QFocusFrame, QLineEdit, QMessageBox,
                               QProxyStyle, QStyle)

from . import design, paint


class PrismStyle(QProxyStyle):
    """Fusion, with the few things a style sheet cannot do taken from the look's table: the tick box and
    the radio dot are painted (no image files), the toolkit's own dotted focus mark is left out (the ring is
    the one sign of focus), the focus frame is told how far outside its control it stands and to lie above
    it, and the wheel does not turn a closed drop-down."""
    _INDICATORS = (QStyle.PM_IndicatorWidth, QStyle.PM_IndicatorHeight,
                   QStyle.PM_ExclusiveIndicatorWidth, QStyle.PM_ExclusiveIndicatorHeight)

    def __init__(self):
        super().__init__("Fusion")

    def pixelMetric(self, metric, option=None, widget=None):
        if metric in self._INDICATORS:
            return design.scaled(design.SIZE["indicator"])           # the box grows with the lettering
        if metric in (QStyle.PM_FocusFrameHMargin, QStyle.PM_FocusFrameVMargin) and isinstance(widget, FocusRing):
            # Only when the frame asks for itself. Lists and tables take the margin of their cells from the
            # same two metrics, so any other widget, and none, gets the toolkit's own answer below.
            if widget.inset:
                return 0
            return design.SIZE["focus_ring"] + design.SIZE["focus_gap"]      # a line and its gap: never scaled
        return super().pixelMetric(metric, option, widget)

    def styleHint(self, hint, option=None, widget=None, returnData=None):
        if hint == QStyle.SH_FocusFrame_AboveWidget:
            # The frame is a child of the window, or of a scroll area's viewport, and lies above its control.
            return 1
        if hint == QStyle.SH_ComboBox_AllowWheelScrolling:
            # A setting must not change while a form is scrolled past it: the wheel is left for the form.
            return 0
        return super().styleHint(hint, option, widget, returnData)

    def drawPrimitive(self, element, option, painter, widget=None):
        if element in (QStyle.PE_IndicatorCheckBox, QStyle.PE_IndicatorRadioButton):
            paint.paint_indicator(painter, "radio" if element == QStyle.PE_IndicatorRadioButton else "check",
                                  QRectF(option.rect), on=bool(option.state & QStyle.State_On),
                                  enabled=bool(option.state & QStyle.State_Enabled))
            return
        if element == QStyle.PE_FrameFocusRect:
            return                                   # the dotted mark: never drawn
        super().drawPrimitive(element, option, painter, widget)


class FocusRing(QFocusFrame):
    """The one ring: a line of the focus colour, clear of its control by the look's gap. The toolkit's focus
    frame keeps it on the control through every move, resize, scroll and new layout of any ancestor, and
    inside a scroll area it is a child of the viewport, so it scrolls with its control and is cut where the
    control is. A control that fills its place (a row, a page that scrolls) has the ring just inside its own
    edge instead."""
    inset = False                                    # read by PrismStyle: no margin round an inset ring

    def paintEvent(self, _event):
        target = self.widget()
        if target is None:
            return
        painter = QPainter(self)
        try:                                         # a raise below must not leave a live painter on the frame
            painter.setRenderHint(QPainter.Antialiasing, True)
            ring = design.SIZE["focus_ring"]
            pen = QPen(design.colour("focus"), ring)
            pen.setJoinStyle(Qt.MiterJoin)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            edge = QRectF(self.rect()).adjusted(ring / 2, ring / 2, -ring / 2, -ring / 2)    # half a pen inside
            if self.inset:
                left = int(target.property("ring_left") or 0)
                painter.drawRect(edge.adjusted(left, 0, 0, 0))
            else:
                corner = design.SIZE["radius"] + design.SIZE["focus_gap"] + ring / 2
                painter.drawRoundedRect(edge, corner, corner)
        finally:
            painter.end()


# The kinds of event the follower hears, held in module names: looked up on QEvent at every event, they
# cost more than the rest of the filter.
_FOCUS_IN, _FOCUS_OUT, _HIDE = QEvent.FocusIn, QEvent.FocusOut, QEvent.Hide
_KEY_PRESS, _MOUSE_PRESS = QEvent.KeyPress, QEvent.MouseButtonPress


class FocusFollower(QObject):
    """Decides where the ring shows. After Tab, Shift+Tab, an arrow key within a group or a shortcut it
    shows wherever the focus goes; after a mouse click only in a field one types in. A window coming back
    to the front keeps what it had. A widget with ``set_focus_shown`` draws its own sign and gets no ring,
    and an opened list marks its own row. One event filter on the application, which reads the kind of an
    event and nothing more for all but five kinds."""
    _TYPED_IN = (QLineEdit, QAbstractSpinBox)

    def __init__(self, app, worn=None, *, listen: bool = True):
        super().__init__(app)
        self._app = app
        self._worn = worn                # a new ring's style, kept alive here: a widget does not own its style
        self._ring, self._target = None, None
        self._keyboard = False           # the keyboard's turn: the ring shows wherever the focus goes
        self._key_last = False           # a key, and not a mouse button, was the last thing pressed
        self._listening = False
        if listen:
            self.listen()

    def listen(self) -> None:
        """Start to hear the application's events; a second call does nothing."""
        if not self._listening:
            self._app.installEventFilter(self)
            self._listening = True

    def listening(self) -> bool:
        return self._listening

    def stop(self) -> None:
        """Stop hearing, take the ring off its control and let it go, and leave the application, so that the
        follower goes with its last name. A second call does nothing more."""
        if self._listening:
            self._app.removeEventFilter(self)
            self._listening = False
        self.show_on(None)
        ring = self.ring()
        if ring is not None:
            ring.hide()
            ring.setParent(None)
        self._ring = None
        self.setParent(None)

    def eventFilter(self, obj, event):
        kind = event.type()                  # first, and nothing else for almost every event there is
        if kind == _FOCUS_IN:
            if obj.isWidgetType():
                self.arrived(obj, event.reason())
        elif kind == _KEY_PRESS:
            self._key_last = True            # a key was the last thing pressed
        elif kind == _MOUSE_PRESS:
            self._key_last = False           # a mouse button was
        elif (kind == _FOCUS_OUT or kind == _HIDE) and obj is self._target:
            self.show_on(None)               # also when its page goes: before the control can be deleted
        return False

    def arrived(self, widget, reason) -> None:
        """The focus arrived on ``widget`` for ``reason``: note whose turn it is, and show the ring or not."""
        if widget.window().windowType() in (Qt.Popup, Qt.ToolTip):
            return                           # an opened list or menu marks its own row
        if reason == Qt.ShortcutFocusReason:
            self._keyboard = True
        elif reason in (Qt.TabFocusReason, Qt.BacktabFocusReason):
            # The toolkit gives these two reasons also to a move no key made: when the control that holds
            # the focus is greyed or hidden, the focus goes on to the next one. Only a key press counts.
            if self._key_last:
                self._keyboard = True
        elif reason == Qt.MouseFocusReason:
            self._keyboard = False
        # An old message box marks the answer the Enter key gives by nothing else than the focus, and it
        # opens with the focus already on that answer. So in one the focus is always shown; these lines go
        # when the last such box is replaced.
        if isinstance(widget.window(), QMessageBox):
            self.show_on(widget)
            return
        self.show_on(widget if (self._keyboard or isinstance(widget, self._TYPED_IN)) else None)

    def target(self):
        """The widget the focus is shown on, or None; one the toolkit has deleted is forgotten."""
        if self._target is not None and not shiboken6.isValid(self._target):
            self._target = None
        return self._target

    def ring(self):
        """The ring, or None before the first one is needed; one the toolkit has deleted is forgotten."""
        if self._ring is not None and not shiboken6.isValid(self._ring):
            self._ring = None
        return self._ring

    def show_on(self, widget) -> None:
        """Show keyboard focus on ``widget``, or on nothing. A widget that shows it itself is told, and the
        ring is taken off; any other widget gets the ring, outside it or just inside its edge."""
        old, self._target = self.target(), widget
        if old is not None and old is not widget and hasattr(old, "set_focus_shown"):
            old.set_focus_shown(False)
        ring = self.ring()
        own = widget is not None and hasattr(widget, "set_focus_shown")
        if own:
            widget.set_focus_shown(True)
        if widget is None or own:
            if ring is not None:
                ring.setWidget(None)
            return
        if ring is None:                     # the frame moves itself into whichever window holds its control;
            ring = self._ring = FocusRing(widget.window())      # if that window was deleted, so was the frame
            if self._worn is not None:
                ring.setStyle(self._worn)
        ring.setWidget(None)
        # Between the two calls: the frame's margin and its mask are asked of its style inside setWidget.
        ring.inset = widget.property("ring") == "inset" or isinstance(widget, QAbstractScrollArea)
        ring.setWidget(widget)
        ring.update()

    def refresh(self) -> None:
        """Put the ring back on its control: a change of style at the control hides the frame."""
        if self.target() is not None:
            self.show_on(self._target)


def set_class(widget, **props):
    """Name a widget's class (or any other property a rule of the style sheet tests) and make the sheet look
    again. A sheet matches its rules when a widget is first polished; a property set after that changes nothing
    by itself, so every class is set through here."""
    for name, value in props.items():
        widget.setProperty(name, value)
    if widget.testAttribute(Qt.WA_WState_Polished):
        style = widget.style()
        style.unpolish(widget)
        style.polish(widget)
        widget.update()
    return widget


_STYLE = None          # the application's style: the only handle on it once a sheet is set
_FOLLOWER = None
_WAITING_ON = None     # the application whose first change of focus is awaited


def install(app) -> None:
    """Give the application the small style and one follower. The follower starts to listen at the
    application's first change of focus: a filter on every event is not paid for while the window
    is built and first shown, and no control can hold the keyboard before that change."""
    global _STYLE, _FOLLOWER, _WAITING_ON
    uninstall(app)
    _STYLE = PrismStyle()
    app.setStyle(_STYLE)
    _FOLLOWER = FocusFollower(app, listen=False)
    app.focusChanged.connect(_first_focus)
    _WAITING_ON = app


def _first_focus(_old, now) -> None:
    global _WAITING_ON
    if _WAITING_ON is not None:
        _WAITING_ON.focusChanged.disconnect(_first_focus)
        _WAITING_ON = None
    if _FOLLOWER is not None:
        _FOLLOWER.listen()
        if now is not None:                  # its arrival was not heard: the window came to the front
            _FOLLOWER.arrived(now, Qt.ActiveWindowFocusReason)


def uninstall(app) -> None:
    """Take the follower off the application and forget it and the style. The style object stays the
    application's until the caller sets another; twice is once."""
    global _STYLE, _FOLLOWER, _WAITING_ON
    if _WAITING_ON is not None:
        _WAITING_ON.focusChanged.disconnect(_first_focus)
        _WAITING_ON = None
    if _FOLLOWER is not None:
        _FOLLOWER.stop()
    _STYLE = _FOLLOWER = None


def current_style():
    """The style ``install`` gave the application, or None."""
    return _STYLE


def current_follower():
    """The follower ``install`` built, or None."""
    return _FOLLOWER


def refresh_focus() -> None:
    """Put the application's ring back on its control; nothing when no follower is installed."""
    if _FOLLOWER is not None:
        _FOLLOWER.refresh()
