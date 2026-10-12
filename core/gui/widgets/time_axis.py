"""The time axis of a run: where everything on it goes, as plain numbers; a widget that paints only
those numbers; and the one line of words under it."""
import datetime
import math
from collections.abc import Callable

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QFontMetricsF, QPainter
from PySide6.QtWidgets import QHBoxLayout, QLabel, QSizePolicy, QWidget

from .. import design, fonts, paint
from ..reported import reported
from .marker import Marker


def _tick_day(when) -> str:
    """The day as a tick writes it, "2 Oct": no weekday and no year. The month's letters are the look's own,
    read from its table at every call, so a tick has the letters every date of the program has."""
    return f"{when.day} {design.MONTHS[when.month - 1]}"


def axis_layout(start, now, end_from, end_by, width: int, text_width: Callable[[str], float]) -> dict:
    """Where everything on the time axis goes, as plain numbers: no painter and no widget, so it is
    tested without a screen. `end_by` None: no end is known. `end_from` None, or equal to `end_by`:
    the end is firm. Given moments of one kind (all with a time zone, or all without) it never raises,
    whatever their order, and the axis is never shorter than 30 s.
    Keys: late, inside, axis_end, x_done, x_now, x_from, x_by, step, ticks, labels (pairs of x and text)."""
    day = datetime.timedelta(days=1)
    if end_by is None:                                 # no end is known: nothing is late and nothing is hatched
        late = inside = False
        axis_end = end_from if end_from is not None and now <= end_from else now
    else:
        if end_from is None:                           # an end with no earliest end is a firm one
            end_from = end_by
        late, inside = now > end_by, end_from <= now <= end_by
        # a run that has passed its range gets an axis as many whole days longer as it needs
        axis_end = end_by + day * max(1, math.ceil((now - end_by) / day)) if late else end_by
    axis_end = max(axis_end, start + datetime.timedelta(seconds=30))       # before the span is taken: it is never zero
    span = (axis_end - start).total_seconds()

    def at(moment):
        return (moment - start).total_seconds() / span * width

    def rnd(value):
        return int(math.floor(value + 0.5))

    half = design.AXIS["mark_w"] / 2 + design.AXIS["mark_halo"]            # the mark never leaves the axis
    x_done = min(max(rnd(at(now)), 0), width)
    if end_by is None:
        x_from = x_by = width                          # a range of no width
    else:
        x_by = min(max(rnd(at(end_by)), 0), width)
        x_from = min(max(rnd(at(end_from)), 0), x_by)
    out = {"late": late, "inside": inside, "axis_end": axis_end, "x_done": x_done,
           "x_now": min(max(x_done, half), width - half), "x_from": x_from, "x_by": x_by}
    # a tick every 30 s on an axis under 5 minutes, every 15 minutes under 2 hours, every 6 hours under 2 days,
    # and beyond that at each midnight
    step = next(seconds for limit, seconds in ((300, 30), (7200, 900), (172800, 21600), (math.inf, 86400))
                if span < limit)
    midnight = start.replace(hour=0, minute=0, second=0, microsecond=0)    # the ticks are counted on the clock
    k = math.floor((start - midnight).total_seconds() / step) + 1          # the first tick after the start
    times = []
    while midnight + datetime.timedelta(seconds=k * step) < axis_end:
        times.append(midnight + datetime.timedelta(seconds=k * step))
        k += 1
    out["step"], out["ticks"] = step, [rnd(at(t)) for t in times]
    labels, right, gap = [], -1e9, design.scaled(design.AXIS["label_gap"])

    def put(text, x):                                  # a label is left out sooner than printed over its neighbour
        nonlocal right
        w = text_width(text)
        x = min(max(x(w), 0.0), width - w)
        if x >= right + gap or not labels:
            labels.append((x, text))
            right = x + w

    if step < 86400:                                   # the clock under each tick; the day before it at a midnight
        for t in times:
            clock = f"{t.hour:02d}:{t.minute:02d}" + (f":{t.second:02d}" if step == 30 else "")
            put(f"{_tick_day(t)} {clock}" if t.time() == datetime.time() else clock, lambda w, t=t: at(t) - w / 2)
    else:                                              # the day between two midnights
        edges = [start] + times + [axis_end]
        for i, (a, b) in enumerate(zip(edges, edges[1:])):
            if i == 0 and a.time() != datetime.time():     # a run begun inside a day: that day at the origin
                put(_tick_day(a), lambda w: 0.0)
            else:
                put(_tick_day(a), lambda w, a=a, b=b: (at(a) + at(b)) / 2 - w / 2)
    out["labels"] = labels
    return out


def _tick_font():
    """The font of the tick labels at the text size in force: the small lettering with its digits of equal
    width, so that a clock keeps its width while its digits change. It is asked where the labels are measured
    and where they are painted, so the two agree; no font is kept."""
    return fonts.inter("small", tabular=True)


def _words(start, now, end_from, end_by, unknown) -> tuple:
    """The line under the axis for these moments: the form it takes, its left-hand words and its right-hand
    ones. The forms are tried in this order and the first that holds is taken: "no end" (the caller's own
    sentence, as given), "late", "firm" (an end with no earliest end, or the two equal), "inside" (the clock
    has reached the range) and "ahead". Every time is written as the window writes a date, without seconds."""
    left = f"Started {design.date_text(start, now)}"
    if end_by is None:
        return "no end", left, unknown
    due = design.date_text(end_by, now)
    if now > end_by:
        return "late", left, f"Later than expected: it was due by {due}"
    if end_from is None or end_from == end_by:
        return "firm", left, f"Expected to end by {due}"
    if end_from <= now:
        return "inside", left, f"Expected to end by {due}"
    return "ahead", left, f"Expected to end between {design.date_text(end_from, now)} and {due}"


def _tried(start, now, end_from, end_by, unknown) -> tuple:
    """The words for these moments, with one layout worked out and thrown away. Both devices call this before
    they build or keep anything: moments that cannot be compared with each other (some with a time zone, some
    without) raise here, in the call that handed them in, and never later, when the axis paints."""
    axis_layout(start, now, end_from, end_by, 1, len)
    return _words(start, now, end_from, end_by, unknown)


class TimeAxis(QWidget):
    """A run drawn on a labelled time axis: what has passed, where it stands now and the range in which
    it should end. It paints what axis_layout returns, reads every colour when it paints, and moves
    only when its page gives it new times."""

    def __init__(self, start, now, end_from=None, end_by=None, *, unknown: str = "") -> None:
        _tried(start, now, end_from, end_by, unknown)        # before anything is built
        super().__init__()
        self.setFixedHeight(design.scaled(design.AXIS["h"]))
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.set_times(start, now, end_from, end_by, unknown=unknown)

    def set_times(self, start, now, end_from=None, end_by=None, *, unknown: str = "") -> None:
        """Draw these moments from the next repaint, which this asks for, and say them in the accessible
        name as the line under the axis words them; `unknown` is the caller's sentence for when no end is
        known. Moments that cannot be laid out raise here and leave the axis as it was."""
        _form, left, right = _tried(start, now, end_from, end_by, unknown)       # before anything is kept
        self._times = (start, now, end_from, end_by)
        self.setAccessibleName(f"{left}." + (f" {right.rstrip('.')}." if right else ""))
        self.update()

    @reported()
    def paintEvent(self, _event):
        painter = QPainter(self)
        try:                                 # a raise below must not leave a live painter on the widget
            font, width = _tick_font(), self.width()
            lay = axis_layout(*self._times, width, QFontMetricsF(font).horizontalAdvance)
            axis, hair = design.AXIS, design.SIZE["hair"]
            top, h = design.scaled(axis["track_y"]), design.scaled(axis["track_h"])
            x_from, x_by, x_done = lay["x_from"], lay["x_by"], lay["x_done"]
            rule, paper = design.colour("rule"), design.colour("paper")
            painter.fillRect(QRectF(0, top, width, h), paper)                       # the track's ground
            if not lay["late"] and x_from < width - hair:                          # the range still ahead, hatched
                ahead = QRectF(x_from, top + hair, width - hair - x_from, h - 2 * hair)
                paint.paint_hatch(painter, ahead, design.colour("hatch"))
                paint.fill_px(painter, QRectF(x_from, top, hair, h), rule, left=hair)
            painter.fillRect(QRectF(0, top, x_done, h), design.colour("elapsed"))   # what has passed
            if lay["late"] or lay["inside"]:                                       # the range shows through the bar
                under = QRectF(x_from, top + hair, min(x_done, x_by) - x_from, h - 2 * hair)
                if under.width() > 0:
                    paint.paint_hatch(painter, under, paper)
                paint.fill_px(painter, QRectF(x_from, top, hair, h), paper, left=hair)
                if lay["late"]:                                                    # and so does the moment it was due
                    paint.fill_px(painter, QRectF(x_by - hair, top, hair, h), paper, left=hair)
            paint.fill_px(painter, QRectF(0, top, width, h), rule, top=hair, bottom=hair, left=hair, right=hair)
            tick_y, tick_h = design.scaled(axis["tick_y"]), design.scaled(axis["tick_h"])
            for x in lay["ticks"]:           # before the mark: a tick under the mark must not be drawn through it
                paint.fill_px(painter, QRectF(x, tick_y, hair, tick_h), rule, left=hair)
            x, mark_w, halo = lay["x_now"], axis["mark_w"], axis["mark_halo"]
            mark_h = design.scaled(axis["mark_h"])       # grown with the text: it must still stand below the track
            painter.fillRect(QRectF(x - mark_w / 2 - halo, 0, mark_w + 2 * halo, mark_h), paper)
            painter.fillRect(QRectF(x - mark_w / 2, 0, mark_w, mark_h), design.colour("accent"))
            painter.setFont(font)            # the font the layout measured with
            painter.setPen(design.colour("ink_2"))
            label_y, label_h = design.scaled(axis["label_y"]), design.scaled(axis["label_h"])
            for x, text in lay["labels"]:
                painter.drawText(QRectF(x, label_y, width, label_h), Qt.AlignLeft | Qt.AlignVCenter, text)
        finally:
            painter.end()


class _HatchKey(QWidget):
    """The hatched range in small, to stand before the words that name it."""

    def __init__(self) -> None:
        super().__init__()
        wide, high = design.SIZE["hatch_key"]
        self.setFixedSize(design.scaled(wide), design.scaled(high))

    @reported()
    def paintEvent(self, _event):
        painter = QPainter(self)
        try:                                 # a raise below must not leave a live painter on the widget
            box, hair = QRectF(self.rect()), design.SIZE["hair"]
            painter.fillRect(box, design.colour("paper"))
            paint.paint_hatch(painter, box, design.colour("hatch"))
            paint.fill_px(painter, box, design.colour("rule"), top=hair, bottom=hair, left=hair, right=hair)
        finally:
            painter.end()


class AxisCaption(QWidget):
    """The one line under a time axis: when the run started and, at the right, when it should end."""
    started: QLabel
    words: QLabel
    key: QWidget             # the hatched range in small
    marker: Marker

    def __init__(self, start, now, end_from=None, end_by=None, *, unknown: str = "") -> None:
        _tried(start, now, end_from, end_by, unknown)        # before anything is built
        super().__init__()
        self.started, self.words = QLabel(), QLabel()
        for label in (self.started, self.words):             # the application's lettering: no font is set here
            label.setTextFormat(Qt.PlainText)                # a sentence may hold "<" or "&": it is shown as written
            label.setFixedHeight(design.scaled(design.SIZE["line_h"]))
        self.key, self.marker = _HatchKey(), Marker("warn")
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(design.scaled(design.GAP["axis_key"]))
        row.addWidget(self.started)
        row.addStretch(1)
        row.addWidget(self.key, 0, Qt.AlignVCenter)
        row.addWidget(self.marker, 0, Qt.AlignVCenter)
        row.addWidget(self.words)
        self.set_times(start, now, end_from, end_by, unknown=unknown)

    def set_times(self, start, now, end_from=None, end_by=None, *, unknown: str = "") -> None:
        """Word the line for these moments. `unknown` is the caller's sentence for the right-hand side when
        no end is known, shown as written. The key stands before the words while a range is hatched on the
        axis, the attention marker when the run is late. Moments that cannot be laid out raise here and
        leave the line as it was."""
        form, left, right = _tried(start, now, end_from, end_by, unknown)        # before anything is changed
        self.started.setText(left)
        self.words.setText(right)
        self.key.setVisible(form in ("inside", "ahead"))
        self.marker.setVisible(form == "late")
