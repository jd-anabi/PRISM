"""The fit's loss by epoch, drawn by the window itself from the numbers of the fit's epoch records.

It is drawn as a journal draws a curve: a left and a bottom axis, their ticks outward, no grid and no box round the
names of the lines; the validation loss in the ink and the training loss dashed in a figure's second colour, so the
two differ without their colours; each line named where it ends, in semibold; the best epoch an open ring named
"best so far: epoch 58", with a dotted drop to the epoch axis; the newest epoch's tick in the ink and in semibold.
Two choices sit in the empty top-right corner: "Recent", which draws the epochs from the tenth on, on a scale
fitted to them, and "All epochs", set at first. A loss that is not a finite number leaves a gap in its line. The
caption under it, in the book face, says what the loss is and when the fit ends.

``loss_layout`` works out where everything goes, as plain numbers, so it is tested without a painter; the curve
paints only what it returns. Every colour, length and face is read from the look's table when the curve paints, so
it follows the theme; it is drawn again when a record arrives, when a choice is made and when the theme changes,
and never on a timer.

Nothing here uses the figure library: that library lays out its text with one parser for the whole process and no
lock, and a run builds its own figures on its own thread while the window draws this one.
"""
import math
from collections.abc import Callable, Sequence

from PySide6.QtCore import QLineF, QPointF, QRectF, Qt
from PySide6.QtGui import QFontMetricsF, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QButtonGroup, QHBoxLayout, QRadioButton, QSizePolicy, QVBoxLayout, QWidget

from .. import design, fonts, paint
from ..reported import reported
from ..widgets.page import Sentence
from .model import EpochRecord

#: "Recent" draws the epochs from this one on.
RECENT_FROM = 10
#: "Recent" is greyed until the newest epoch is this one.
RECENT_AVAILABLE_AT = 12
#: The most intervals between two round ticks on either axis: an axis takes the least round step that keeps to it.
TICK_INTERVALS_MOST = 7
RECENT_TEXT = "Recent"
ALL_TEXT = "All epochs"
EPOCH_NAME = "Epoch"
LOSS_NAME = "Loss"
VALIDATION_NAME = "validation"
TRAINING_NAME = "training"
CAPTION_LEAD = "Loss by epoch."
CAPTION_TEXT = "Loss is the negative log-probability; lower is better."
_MARGIN = 1 / 20                     # a twentieth of the losses' range is added at each end of the loss axis
_MINUS = chr(0x2212)                 # a tick's minus: the minus sign, as figures write it


def caption_words(patience: int | None) -> str:
    """The caption's words after its bold lead: what the loss is and, when the patience is known, when the fit
    ends."""
    if patience is None:
        return CAPTION_TEXT
    epochs = "epoch" if patience == 1 else "epochs"
    return (f"{CAPTION_TEXT} Fitting ends when validation has not improved for {design.number_text(patience)} "
            f"{epochs}.")


def best_words(epoch: int) -> str:
    """The words beside the best epoch's ring."""
    return f"best so far: epoch {design.number_text(epoch)}"


def left_out_words(first: int) -> str:
    """The note "Recent" shows: the epochs held before the tenth, which it does not draw."""
    last = RECENT_FROM - 1
    if first >= last:
        return f"epoch {design.number_text(last)} is left out"
    return f"epochs {design.number_text(first)} to {design.number_text(last)} are left out"


def recent_offered(epochs: Sequence[int]) -> bool:
    """Whether "Recent" can be chosen for these epochs: the newest is the twelfth or later, and an epoch before the
    tenth is held, which "Recent" would leave out. A fit resumed after its ninth pass or later holds none."""
    return bool(epochs) and max(epochs) >= RECENT_AVAILABLE_AT and min(epochs) < RECENT_FROM


def _finite(value) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value)


def _round_step(span: float, whole: bool) -> float:
    """The least round step (1, 2 or 5 times a power of ten; a whole number when ``whole``) that cuts ``span``
    into no more intervals than an axis is given."""
    raw = span / TICK_INTERVALS_MOST
    if raw <= 0:
        return 1.0
    power = 10.0 ** math.floor(math.log10(raw))
    step = next(lead * power for lead in (1, 2, 5, 10) if lead * power >= raw * (1 - 1e-9))
    return max(1.0, step) if whole else step


def _next_step(step: float) -> float:
    """The round step after ``step``."""
    power = 10.0 ** math.floor(math.log10(step) + 1e-9)
    lead = step / power
    return (2 if lead < 1.5 else 5 if lead < 3.5 else 10) * power


def _loss_words(value: float, step: float) -> str:
    """A tick of the loss axis, with as many decimals as its step needs and the minus sign for its minus."""
    decimals = max(0, -math.floor(math.log10(step) + 1e-9))
    words = f"{value:,.{decimals}f}"
    if float(words.replace(",", "")) == 0:
        words = words.lstrip("-")                   # no minus before a nought
    return words.replace("-", _MINUS)


def _meets(box: QRectF, points: Sequence) -> bool:
    """Whether a line through ``points``, each an x and a y, enters ``box``."""
    if any(box.contains(QPointF(*point)) for point in points):
        return True
    edges = (QLineF(box.topLeft(), box.topRight()), QLineF(box.topRight(), box.bottomRight()),
             QLineF(box.bottomRight(), box.bottomLeft()), QLineF(box.bottomLeft(), box.topLeft()))
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        segment = QLineF(x0, y0, x1, y1)
        if any(segment.intersects(edge)[0] == QLineF.IntersectionType.BoundedIntersection for edge in edges):
            return True
    return False


def loss_layout(records: Sequence[EpochRecord], best: int | None, recent: bool, width: float, height: float,
                measure: Callable[[str, bool], tuple[float, float, float]]) -> dict:
    """Where everything on the loss curve goes, as plain numbers: no painter and no widget, so it is tested without
    a screen. ``records`` are the fit's epoch records in order, ``best`` the best epoch so far, ``recent`` whether
    "Recent" is chosen (taken only while ``recent_offered``), ``width`` and ``height`` the plot's size in pixels,
    and ``measure(words, semibold)`` the width of words in the curve's lettering, and its ascent and descent.

    Keys: ``plot`` (the left, top, right and bottom of the axes' box); ``epochs`` (the epochs drawn); ``x_ticks``
    (each an x, its words and whether it is the newest epoch's) and ``y_ticks`` (each a y and its words);
    ``validation`` and ``training`` (each a list of runs of points: a loss that is not finite ends a run); ``best``
    (the centre of the best epoch's ring, or None); ``note`` (the words "Recent" shows, or ""); ``words`` (every
    text drawn, in the order drawn: its words, the left and the top of its box, its width and its height, whether
    it is semibold and the name of its colour in the look's table). A text's baseline is its box's top and its
    ascent."""
    spec, shared = design.FIGURE["loss"], design.FIGURE
    left, top = design.scaled(spec["left"]), design.scaled(spec["top"])
    right, bottom = width - design.scaled(spec["right"]), height - design.scaled(spec["bottom"])
    gap = design.scaled(shared["line_label_gap"])
    axis, tick, pad = shared["axis_w"], shared["tick"], design.scaled(spec["tick_pad"])
    chosen = recent and recent_offered([record.epoch for record in records])
    shown = [record for record in records if record.epoch >= RECENT_FROM] if chosen else list(records)
    out = {"plot": (left, top, right, bottom), "epochs": [record.epoch for record in shown], "x_ticks": [],
           "y_ticks": [], "validation": [], "training": [], "best": None,
           "note": left_out_words(records[0].epoch) if chosen else "", "words": []}

    def say(words, x, baseline, colour, semibold=False):
        """Words whose box has its left at ``x`` and whose letters stand on ``baseline``."""
        w, ascent, descent = measure(words, semibold)
        out["words"].append((words, x, baseline - ascent, w, ascent + descent, semibold, colour))

    def middle_line(words, semibold, y):
        """The baseline that sets these words' box in the middle of ``y``."""
        _w, ascent, descent = measure(words, semibold)
        return y + (ascent - descent) / 2

    # the axes' names: the epoch's centred under its axis; the loss's at the head of its axis, to its left, and the
    # note of "Recent" beside the head, inside the axes
    head = top + design.scaled(shared["y_name_sink"])
    say(EPOCH_NAME, (left + right - measure(EPOCH_NAME, False)[0]) / 2, bottom + design.scaled(spec["x_name_drop"]),
        "ink")
    say(LOSS_NAME, left - design.scaled(shared["y_name_gap"]) - measure(LOSS_NAME, False)[0], head, "ink")
    if out["note"]:
        say(out["note"], left + design.scaled(shared["note_x"]), head, "ink_2")
    if not shown:
        return out

    first, last = shown[0].epoch, shown[-1].epoch

    def at_x(epoch):
        return left + (epoch - first) / max(1, last - first) * (right - left)

    # the epoch ticks: the first epoch drawn and the newest, and round ones between that crowd neither of them
    # nor each other
    def placed(epoch):
        return at_x(epoch), measure(design.number_text(epoch), epoch == last)[0]

    def apart(a, b):
        (xa, wa), (xb, wb) = placed(a), placed(b)
        return abs(xa - xb) >= (wa + wb) / 2 + gap

    ends = [first] if last == first else [first, last]
    step = int(_round_step(last - first, whole=True))
    while True:
        middle = [epoch for epoch in range((first // step + 1) * step, last, step)
                  if all(apart(epoch, end) for end in ends)]
        if all(apart(a, b) for a, b in zip(middle, middle[1:])):
            break
        step = int(_next_step(step))
    under = bottom + axis + tick + pad                  # the top of the tick labels' boxes
    for epoch in sorted(ends + middle):
        x, w = placed(epoch)
        words, newest = design.number_text(epoch), epoch == last
        out["x_ticks"].append((x, words, newest))
        say(words, x - w / 2, under + measure(words, newest)[1], "ink" if newest else "ink_2", newest)

    # the loss axis: the finite losses drawn, with a twentieth of their range more at each end
    losses = [value for record in shown for value in (record.validation_loss, record.training_loss)
              if _finite(value)]
    if not losses:
        return out
    low, high = min(losses), max(losses)
    low, high = (low - 1, high + 1) if high == low else (low - (high - low) * _MARGIN, high + (high - low) * _MARGIN)

    def at_y(value):
        return bottom - (value - low) / (high - low) * (bottom - top)

    _w, ascent, descent = measure("0", False)
    tall = ascent + descent
    step = _round_step(high - low, whole=False)
    while True:
        values = [k * step for k in range(math.ceil(low / step - 1e-9), math.floor(high / step + 1e-9) + 1)]
        if len(values) < 2 or abs(at_y(values[0]) - at_y(values[1])) >= tall + gap:
            break
        step = _next_step(step)
    for value in values:
        words = _loss_words(value, step)
        out["y_ticks"].append((at_y(value), words))
        say(words, left - axis - tick - pad - measure(words, False)[0], middle_line(words, False, at_y(value)),
            "ink_2")

    # the lines, a loss that is not finite ending a run; each line's name where it ends, the two never one over
    # the other
    named = []
    for key, loss, words in (("validation", "validation_loss", VALIDATION_NAME),
                             ("training", "training_loss", TRAINING_NAME)):
        runs, run = [], []
        for record in shown:
            value = getattr(record, loss)
            if _finite(value):
                run.append((at_x(record.epoch), at_y(value)))
            elif run:
                runs.append(run)
                run = []
        if run:
            runs.append(run)
        out[key] = runs
        if runs:
            named.append([words, runs[-1][-1][0] + gap, runs[-1][-1][1]])
    if len(named) == 2 and abs(named[0][2] - named[1][2]) < tall:
        centre = (named[0][2] + named[1][2]) / 2
        upper, lower = sorted(named, key=lambda one: one[2])
        upper[2], lower[2] = centre - tall / 2, centre + tall / 2
    for words, x, y in named:
        say(words, x, middle_line(words, True, y), "ink", True)

    # the best epoch: its ring where its validation loss was; its words above the ring, or under it where a line
    # runs through the place above
    marked = next((record for record in shown if record.epoch == best and _finite(record.validation_loss)), None)
    if marked is not None:
        cx, cy = at_x(marked.epoch), at_y(marked.validation_loss)
        out["best"] = (cx, cy)
        words = best_words(marked.epoch)
        w, ascent, descent = measure(words, True)
        h, mark = ascent + descent, design.scaled(shared["best_mark"])
        lines = out["validation"] + out["training"]

        def clear(place):
            return place.top() >= 0 and place.bottom() <= bottom and not any(_meets(place, run) for run in lines)

        above, below = QRectF(cx + mark / 2 - w, cy - mark - h, w, h), QRectF(cx - gap - w, cy + mark, w, h)
        place = above if clear(above) or not clear(below) else below
        say(words, max(place.left(), left), place.top() + ascent, "ink", True)
    return out


def _font(semibold: bool = False):
    """The curve's lettering: the small Inter with its digits of equal width, at the ordinary weight or semibold.
    It is asked where the words are measured and where they are painted, so the two agree; no font is kept."""
    return fonts.inter("small", 600 if semibold else 400, tabular=True)


def _measure(words: str, semibold: bool) -> tuple[float, float, float]:
    """The width of words in the curve's lettering, and its ascent and descent."""
    metrics = QFontMetricsF(_font(semibold))
    return metrics.horizontalAdvance(words), metrics.ascent(), metrics.descent()


class _Plot(QWidget):
    """The axes and the lines, painted from what ``loss_layout`` gives for the plot's size. The curve lays its two
    choices in the plot's top-right corner."""

    def __init__(self) -> None:
        super().__init__()
        self.records: tuple = ()
        self.best: int | None = None
        self.recent = False
        spec = design.FIGURE["loss"]
        self.setFixedHeight(design.scaled(spec["h"]))
        self.setMinimumWidth(design.scaled(spec["min_w"]))
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    def laid_out(self) -> dict:
        """Where everything goes at the plot's size now, measured in the lettering it is painted in."""
        return loss_layout(self.records, self.best, self.recent, self.width(), self.height(), _measure)

    @reported()
    def paintEvent(self, _event):
        painter = QPainter(self)
        try:                                 # a raise below must not leave a live painter on the widget
            lay = self.laid_out()
            left, top, right, bottom = lay["plot"]
            axis, line, tick = design.FIGURE["axis_w"], design.FIGURE["line_w"], design.FIGURE["tick"]
            rule = design.colour("rule")
            painter.fillRect(QRectF(self.rect()), design.colour("paper"))
            painter.setRenderHint(QPainter.Antialiasing, True)
            for key, colour in (("training", "series_2"), ("validation", "ink")):
                pen = QPen(design.colour(colour), line)
                pen.setJoinStyle(Qt.RoundJoin)
                if key == "training":
                    pen.setCapStyle(Qt.FlatCap)
                    pen.setDashPattern([3, 2])              # dashed, in widths of the line: 3 on, 2 off
                else:
                    pen.setCapStyle(Qt.RoundCap)
                painter.setPen(pen)
                painter.setBrush(Qt.NoBrush)
                for run in lay[key]:
                    path = QPainterPath(QPointF(*run[0]))
                    for point in run[1:]:
                        path.lineTo(QPointF(*point))
                    painter.drawPath(path)
            if lay["best"] is not None:
                cx, cy = lay["best"]
                drop, ink_2 = cy, design.colour("ink_2")
                while drop < bottom:                        # dotted, on whole device pixels: 2 on, 3 off
                    paint.fill_px(painter, QRectF(cx - axis / 2, drop, axis, min(2 * axis, bottom - drop)), ink_2,
                                  left=axis)
                    drop += 5 * axis
                radius = design.scaled(design.FIGURE["best_mark"]) / 2
                painter.setPen(QPen(design.colour("ink"), line))
                painter.setBrush(design.colour("paper"))
                painter.drawEllipse(QPointF(cx, cy), radius, radius)
            # the two axes and their ticks, outward from them, each on whole device pixels
            paint.fill_px(painter, QRectF(left - axis, top, axis, bottom - top + axis), rule, left=axis)
            paint.fill_px(painter, QRectF(left - axis, bottom, right - left + axis, axis), rule, top=axis)
            for x, _words, _newest in lay["x_ticks"]:
                paint.fill_px(painter, QRectF(x - axis / 2, bottom, axis, axis + tick), rule, left=axis)
            for y, _words in lay["y_ticks"]:
                paint.fill_px(painter, QRectF(left - axis - tick, y - axis / 2, tick + axis, axis), rule, top=axis)
            for words, x, y, _w, _h, semibold, colour in lay["words"]:
                font = _font(semibold)
                painter.setFont(font)
                painter.setPen(design.colour(colour))
                painter.drawText(QPointF(x, y + QFontMetricsF(font).ascent()), words)
        finally:
            painter.end()


class LossCurve(QWidget):
    """The fit's loss by epoch: the plot, its two choices in the plot's top-right corner, and its caption under it.
    ``set_epochs`` hands it the fit's records; it is drawn again only when they, the best epoch or the choice
    change, and on ``retheme``. Its caption is a sentence of the look, which a change of theme reaches by itself."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._patience: int | None = None
        self.plot = _Plot()
        self.recent_button = QRadioButton(RECENT_TEXT)
        self.all_button = QRadioButton(ALL_TEXT)
        self._choice = QButtonGroup(self)
        corner = QHBoxLayout(self.plot)
        corner.setContentsMargins(0, 0, 0, 0)
        corner.setSpacing(design.scaled(design.GAP["switch"]))
        corner.addStretch(1)
        for button in (self.recent_button, self.all_button):
            self._choice.addButton(button)
            corner.addWidget(button, 0, Qt.AlignTop)
        self.all_button.setChecked(True)
        self.recent_button.setEnabled(False)
        self.recent_button.toggled.connect(self._chosen)
        self.caption = Sentence(quiet=True)
        self.caption.set_text(caption_words(None), lead=CAPTION_LEAD)
        column = QVBoxLayout(self)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(0)
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, design.scaled(design.GAP["figure_inset"]), 0)
        row.addWidget(self.plot)
        column.addLayout(row)
        column.addSpacing(design.scaled(design.GAP["figure_caption"]))
        under = QHBoxLayout()
        under.setContentsMargins(0, 0, design.scaled(design.GAP["caption_inset"]), 0)
        under.addWidget(self.caption)
        column.addLayout(under)
        self.setAccessibleName(CAPTION_LEAD.rstrip("."))
        self._describe()

    @property
    def recent(self) -> bool:
        """Whether "Recent" is chosen, and so drawn."""
        return self.recent_button.isChecked()

    def set_recent(self, on: bool) -> None:
        """Choose "Recent", when it can be chosen, or "All epochs"."""
        (self.recent_button if on and self.recent_button.isEnabled() else self.all_button).setChecked(True)

    def set_epochs(self, epochs: Sequence[EpochRecord], *, best: int | None = None,
                   patience: int | None = None) -> None:
        """Draw these records, the fit's ``best`` epoch marked, from the next repaint, which this asks for only
        when they or the best epoch have changed; ``patience`` is written in the caption (None leaves out when the
        fit ends). "Recent" is greyed while it cannot be chosen, and "All epochs" set again when it is greyed."""
        records = tuple(epochs)
        if (records, best) != (self.plot.records, self.plot.best):
            self.plot.records, self.plot.best = records, best
            offered = recent_offered([record.epoch for record in records])
            self.recent_button.setEnabled(offered)
            if not offered and self.recent_button.isChecked():
                self.all_button.setChecked(True)
            self.plot.update()
            self._describe()
        if patience != self._patience:
            self._patience = patience
            self.caption.set_text(caption_words(patience), lead=CAPTION_LEAD)

    def retheme(self) -> None:
        """Draw the plot again, in the colours now in force."""
        self.plot.update()

    def drawn(self) -> dict:
        """What the plot draws at its size now: ``loss_layout``'s answer."""
        return self.plot.laid_out()

    def _chosen(self, recent: bool) -> None:
        self.plot.recent = recent
        self.plot.update()
        self._describe()

    def _describe(self) -> None:
        """Say what is drawn in the accessible description: the epochs, the best epoch's words and the note."""
        lay = self.plot.laid_out()
        epochs, said = lay["epochs"], []
        if epochs:
            said.append(f"epochs {design.number_text(epochs[0])} to {design.number_text(epochs[-1])}"
                        if len(epochs) > 1 else f"epoch {design.number_text(epochs[0])}")
        if lay["best"] is not None:
            said.append(best_words(self.plot.best))
        if lay["note"]:
            said.append(lay["note"])
        self.setAccessibleDescription("; ".join(said))
