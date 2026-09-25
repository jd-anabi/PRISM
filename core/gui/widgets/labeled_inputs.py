"""Small typed input widgets: the numeric fields and file pickers the GUI's forms are built from."""
from PySide6.QtGui import QDoubleValidator, QIntValidator
from PySide6.QtWidgets import QFileDialog, QHBoxLayout, QLineEdit, QPushButton, QWidget

from core.refusals import describe, refuse

from ..design import FIELD_MIN_W, PATH_FIELD_MIN_W

# These classes set a validator and, now, a MINIMUM WIDTH. Qt's default QFormLayout policy on Windows
# is FieldsStayAtSizeHint, so without a floor a numeric field rendered 3-6 characters wide: typing
# "0.033333" scrolled inside the box and the value could not be read back without clicking into it.
# The matching half of the fix is the field-growth policy (see widgets/forms.make_form).


class FloatField(QLineEdit):
    def __init__(self, default: "float | None" = 0.0, parent=None):
        # None is an EMPTY box, not the text "None": a seeded value nobody typed is a different state
        # from a blank one, and value_or_none() is what tells them apart (piece 4, B16). Several
        # callers hand this a pre-formatted string instead of a float, which str() leaves alone.
        super().__init__("" if default is None else str(default), parent)
        self.setValidator(QDoubleValidator())
        self.setMinimumWidth(FIELD_MIN_W)

    def value(self) -> float:
        try:
            return float(self.text())
        except ValueError:
            return 0.0

    def value_or_none(self) -> "float | None":
        """The field's value, or None when it does not parse.

        Use this ANYWHERE 0.0 is a legal value the user could also have meant, because ``value()``
        cannot tell "0" from "" or from "-" mid-typing -- a blank "min" box silently becomes a real
        bound of 0. That is not hypothetical: it is why widgets/param_grid validates explicitly (see
        its module docstring), and the model builder's own parameter rows had the bug it warns about.
        """
        try:
            return float(self.text().strip())
        except (TypeError, ValueError):
            return None


def number_or_blank(field, key: str):
    """``field.value_or_none()`` -- except that text which is NOT blank and does not parse is refused
    under ``key`` rather than read as blank.

    A numeric validator lets '-', '1e', '.' and '+' stand in a box as text still being typed, and
    value_or_none() reads each as None. For the boxes this serves, None MEANS something -- the middle
    of the range two sweeps share, "draw a seed", "the constant is blank" -- so a half-typed number
    silently became that (the whole-piece review's N22). Only a truly empty box is blank."""
    value = field.value_or_none()
    text = field.text().strip()
    if value is None and text:
        what = describe(key)
        refuse(key, f"{what[0].upper()}{what[1:]} must be a number; got {text!r}.")
    return value


class IntField(QLineEdit):
    def __init__(self, default: int = 0, parent=None):
        super().__init__(str(default), parent)
        self.setValidator(QIntValidator())
        self.setMinimumWidth(FIELD_MIN_W)

    def value(self) -> int:
        try:
            return int(self.text())
        except ValueError:
            return 0

    def value_or_none(self) -> "int | None":
        """The field's value, or None when it does not parse -- the IntField twin of FloatField's.

        ``value()`` returns 0 for "" and for "-" mid-typing, and 0 is a legal value for the two
        "0 = automatic" boxes (the rows-per-batch cap, the candidates per sweep round), so a tab
        that wants to refuse a blank instead of reading it as zero has to ask this. The inference
        tabs read every integer box through it at the click (piece 3, V2).
        """
        try:
            return int(self.text().strip())
        except (TypeError, ValueError):
            return None


class PathField(QWidget):
    def __init__(self, file_filter: str = "Data (*.csv *.npy);;All files (*)", parent=None):
        super().__init__(parent)
        self.edit = QLineEdit()
        self.edit.setMinimumWidth(PATH_FIELD_MIN_W)
        # Absolute paths elide at the FRONT, so the filename stays visible; the tooltip carries the
        # whole thing (a per-frequency chi row leaves only ~120px of line edit).
        self.edit.textChanged.connect(lambda t: self.edit.setToolTip(t))
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._browse)
        self._filter = file_filter
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.edit, 1)
        layout.addWidget(browse)

    def _browse(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select file", "", self._filter)
        if path:
            self.edit.setText(path)

    def value(self) -> str:
        return self.edit.text().strip()
