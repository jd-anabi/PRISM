"""Editable grids for entering parameter BOUNDS and cell VALUES by hand instead of picking a file.

Both grids are LOADED FROM a parsed file first and then edited: the parameter NAMES and their ORDER are
not the user's to invent -- they are fixed by the model, and the order is load-bearing (simulators bind
parameter columns positionally via ``torch.unbind``, and build_prior asserts the ND order matches a user
model's compiled signature). So these widgets only ever let you change NUMBERS, never the schema, which
is what makes hand-entry safe.

Validation is explicit because ``FloatField.value()`` silently returns 0.0 on unparseable text -- a blank
"min" box would otherwise become a real bound of 0. ``problems()`` returns human-readable strings and the
callers refuse to build anything while it is non-empty.
"""
import math
from collections import OrderedDict

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QFormLayout, QHBoxLayout, QLabel, QScrollArea, QVBoxLayout, QWidget)

from .. import design, fonts
from .field_row import LabeledFieldRow
from .labeled_inputs import FloatField
from .forms import make_form

_MAX_VISIBLE_HEIGHT = 320          # keep a 13-parameter grid from swallowing the controls column
NOT_LOADED_TEXT = "No limits are loaded: choose a bounds file to start them from."
NOT_NUMBERS_TEXT = "The limits of {} must both be numbers."
CROSSED_TEXT = "The lower limit of {}, {:g}, must be less than its upper limit, {:g}."
#: A limit typed that the field itself refuses, a comma for a point: its words, then the field's own.
FIELD_SAYS_TEXT = "The {} limit of {} reads {}. {}"


def limit_problems(limits) -> list:
    """What is wrong with ``limits``, each ``(name, lower, upper)`` in their order, a limit None where it reads as no
    number: one sentence for each parameter whose limits are not two numbers, the lower below the upper. The one rule
    for the limits a bounds file gives and those typed by hand."""
    out = []
    for name, lo, hi in limits:
        if not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in (lo, hi)):
            out.append(NOT_NUMBERS_TEXT.format(name))
        elif not lo < hi:
            out.append(CROSSED_TEXT.format(name, lo, hi))
    return out


def two_limits(bounds) -> tuple:
    """A parameter's limits as a file gives them, as two values; (None, None) for limits it does not give as two."""
    return tuple(bounds) if isinstance(bounds, (tuple, list)) and len(bounds) == 2 else (None, None)


def _num(field) -> "float | None":
    """The field's value, or None when it does not parse (FloatField.value() would return 0.0).

    Thin alias for ``FloatField.value_or_none``, which is where the rule now lives so the model
    builder's parameter rows share it instead of re-deriving (or, as they did, omitting) it.
    """
    return field.value_or_none()


class _PassThroughScrollArea(QScrollArea):
    """A scroll area that hands the wheel back to its parent once it cannot scroll any further.

    This is the ONLY nested scroll area in the app -- a bounds grid sits inside a SourceToggle inside
    BasePanel's controls scroll area. Qt's default is for the inner area to swallow every wheel event
    it receives, so scrolling over the grid moved the grid and then simply stopped: you could not
    reach the "Build / Load prior" button below it by scrolling from there, which reads as the page
    being stuck.

    Ignoring the event at the limit lets it propagate to the outer area, which is the behaviour
    people expect from nested scrollers everywhere else.
    """

    def wheelEvent(self, event):
        bar = self.verticalScrollBar()
        going_up = event.angleDelta().y() > 0
        at_limit = (bar.value() == bar.minimum()) if going_up else (bar.value() == bar.maximum())
        if at_limit or bar.maximum() == 0:
            event.ignore()               # -> outer scroll area handles it
            return
        super().wheelEvent(event)


def _scrollable(inner: QWidget) -> QScrollArea:
    area = _PassThroughScrollArea()
    area.setWidgetResizable(True)
    area.setWidget(inner)
    # Sized to CONTENT up to a ceiling, rather than a flat 320px for everything: a 3-parameter grid
    # and a 13-parameter grid used to reserve exactly the same box.
    area.setMaximumHeight(_MAX_VISIBLE_HEIGHT)
    inner.adjustSize()
    area.setMinimumHeight(min(inner.sizeHint().height() + 4, _MAX_VISIBLE_HEIGHT))
    return area


class _BoundsRow(LabeledFieldRow):
    """min / max for one parameter."""

    def __init__(self, lo: "float | None", hi: "float | None", parent=None):
        self.lo, self.hi = FloatField(lo), FloatField(hi)
        super().__init__((("min", self.lo), ("max", self.hi)), parent=parent)


class BoundsGrid(QWidget):
    """A (min, max) row per parameter, across the ND / Dimensional / Forcing sections.

    Produces exactly what ``file_manager.parse_bounds_file`` produces -- ``{name: (None, (lo, hi))}``
    per section, in the loaded order -- so downstream code cannot tell the difference.

    ``scrolls`` puts the rows in a scroll area of their own, as a form that does not scroll needs. A page that scrolls
    as a whole asks for none: its grid stands at its full height, and every field is reached by the page's own scroll,
    which keeps the room a field's ring needs round what it holds. Its fields then stand as a rebuilt page's number
    fields stand: right aligned, their digits of equal width, the look's width."""

    changed = Signal()                             # a limit was typed or refused, or the rows were made again

    def __init__(self, parent=None, *, scrolls: bool = True):
        super().__init__(parent)
        self._rows = OrderedDict()                 # (section, name) -> _BoundsRow
        self._sections = OrderedDict()             # section -> [names] (preserves file order)
        inner = QWidget()
        self._form = make_form(inner)
        self._form.setContentsMargins(0, 0, 0, 0)
        outer = QVBoxLayout(self)
        self._dressed = not scrolls
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(_scrollable(inner) if scrolls else inner)
        self._placeholder = QLabel("(load a bounds file to edit its values)")
        self._placeholder.setProperty("type", "caption")
        self._form.addRow(self._placeholder)

    def load(self, params: OrderedDict, rescale: OrderedDict, forcing: OrderedDict) -> None:
        """Populate from parsed bounds dicts. Rebuilds the whole grid, preserving section + name order. A parameter
        whose limits the file does not give as two values stands with its two fields blank."""
        while self._form.rowCount():
            self._form.removeRow(0)
        self._rows, self._sections = OrderedDict(), OrderedDict()
        for section, d, title in (("PARAM", params, "Non-dimensional"),
                                  ("RESCALE", rescale, "Dimensional"),
                                  ("FORCING", forcing, "Forcing")):
            if not d:
                continue
            header = QLabel(title)
            header.setProperty("type", "caption")
            self._form.addRow(header)
            self._sections[section] = list(d)
            for name, (_v, bounds) in d.items():
                row = _BoundsRow(*two_limits(bounds))
                self._rows[(section, name)] = row
                self._form.addRow(name, row)
                for field in (row.lo, row.hi):
                    if self._dressed:
                        field.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
                        field.setFont(fonts.inter("body", tabular=True))
                        field.setFixedWidth(design.scaled(design.SIZE["number_w"]))
                    field.textChanged.connect(self.changed)
                    field.error_changed.connect(self.changed)
        self.changed.emit()

    def is_loaded(self) -> bool:
        return bool(self._rows)

    def problems(self) -> list:
        """Each thing wrong with the limits typed, in a sentence, in the rows' order; none when they can be used."""
        if not self.is_loaded():
            return [NOT_LOADED_TEXT]
        out = []
        for (_section, name), row in self._rows.items():
            said = [FIELD_SAYS_TEXT.format(which, name, field.text().strip(), field.error())
                    for which, field in (("lower", row.lo), ("upper", row.hi)) if field.error()]
            out += said[:1] or limit_problems([(name, _num(row.lo), _num(row.hi))])
        return out

    def to_dicts(self) -> tuple:
        """(params, rescale, forcing) in parse_bounds_file's shape. Call only when problems() is empty."""
        out = {s: OrderedDict() for s in ("PARAM", "RESCALE", "FORCING")}
        for section, names in self._sections.items():
            for name in names:
                row = self._rows[(section, name)]
                out[section][name] = (None, (_num(row.lo), _num(row.hi)))
        return out["PARAM"], out["RESCALE"], out["FORCING"]


class ValuesGrid(QWidget):
    """One value field per initial condition and per parameter -- a hand-editable cell file.

    Produces ``file_manager.parse_values_file``'s shape: (inits, params, rescale, forcing) plain
    ``{name: value}`` dicts, ready for ``SimConfig.inject_ground_truth``."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._fields = OrderedDict()               # (section, name) -> FloatField
        self._sections = OrderedDict()
        inner = QWidget()
        self._form = make_form(inner)
        self._form.setContentsMargins(0, 0, 0, 0)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(_scrollable(inner))
        self._placeholder = QLabel("(load a cell file to edit its values)")
        self._placeholder.setProperty("type", "caption")
        self._form.addRow(self._placeholder)

    def load(self, inits: dict, params: dict, rescale: dict, forcing: dict) -> None:
        while self._form.rowCount():
            self._form.removeRow(0)
        self._fields, self._sections = OrderedDict(), OrderedDict()
        for section, d, title in (("INITS", inits, "Initial conditions"),
                                  ("PARAM", params, "Non-dimensional"),
                                  ("RESCALE", rescale, "Dimensional"),
                                  ("FORCING", forcing, "Forcing")):
            if not d:
                continue
            header = QLabel(title)
            header.setProperty("type", "caption")
            self._form.addRow(header)
            self._sections[section] = list(d)
            for name, v in d.items():
                fld = FloatField(float(v))
                self._fields[(section, name)] = fld
                self._form.addRow(name, fld)

    def is_loaded(self) -> bool:
        return bool(self._fields)

    def problems(self) -> list:
        if not self.is_loaded():
            return ["no values loaded — pick a cell file to start from, then edit it"]
        return [f"{name}: must be a number" for (_s, name), fld in self._fields.items()
                if _num(fld) is None]

    def to_dicts(self) -> tuple:
        """(inits, params, rescale, forcing). Call only when problems() is empty."""
        out = {s: OrderedDict() for s in ("INITS", "PARAM", "RESCALE", "FORCING")}
        for section, names in self._sections.items():
            for name in names:
                out[section][name] = _num(self._fields[(section, name)])
        return out["INITS"], out["PARAM"], out["RESCALE"], out["FORCING"]
