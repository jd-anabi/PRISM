"""Small typed input widgets: the numeric fields and file pickers the GUI's forms are built from."""
import re

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QValidator
from PySide6.QtWidgets import QHBoxLayout, QLineEdit, QPushButton, QWidget

from core.refusals import describe, refuse

from ..design import FIELD_MIN_W, PATH_FIELD_MIN_W
from ..file_chooser import choose_file
from ..style import set_class

COMMA_SENTENCE = "Use a point as the decimal sign."
_WHOLE = re.compile(r"-?(?:[0-9]+|[0-9]{1,3}(?:,[0-9]{3})+)")     # read: digits, or commas where grouping puts them
_WHOLE_KEYS = re.compile(r"-?[0-9]*")                             # typed, once its commas are taken out
_DECIMAL_KEYS = re.compile(r"[+-]?[0-9,]*\.?[0-9,]*(?:[eE][+-]?[0-9]*)?")   # typed: a comma shows, a second point does not

# These classes decide for themselves which keys they take: the toolkit's own number checks are not used, because
# they rewrote a typed text into another number when editing ended. They also set a MINIMUM WIDTH. Qt's default
# QFormLayout policy on Windows is FieldsStayAtSizeHint, so without a floor a numeric field rendered 3-6 characters
# wide: typing "0.033333" scrolled inside the box and the value could not be read back without clicking into it.
# The matching half of the fix is the field-growth policy (see widgets/forms.make_form).


def _whole(text) -> "int | None":
    """The whole number a text reads as, or None when it is not one: digits under an optional leading minus,
    with a comma only where grouping puts it ("20,480,000" is a number, "1,5" is not), and never a plus. More
    digits than Python reads as one number are no number either: no text makes this raise."""
    text = text.strip()
    if _WHOLE.fullmatch(text):
        try:
            return int(text.replace(",", ""))
        except ValueError:                           # more digits than Python reads as one number
            return None
    return None


def _commas(digits) -> str:
    """Digits with a comma before every third one from the right. Text in and text out: no digit is read as a
    number on the way, so none is dropped, a zero typed first among them, and no length fails."""
    head = len(digits) % 3 or 3
    return ",".join([digits[:head]] + [digits[at:at + 3] for at in range(head, len(digits), 3)])


class _WholeKeys(QValidator):
    """Which texts a whole-number field lets stand while it is typed in: digits under an optional leading
    minus, once the commas are taken out. The answer is that a text can stand or that it cannot, never that
    it is half-way, and the text and the caret go back as they came: on a half-way answer the toolkit rewrites
    the text when editing ends, and the field never says that editing ended."""

    def validate(self, text, pos):
        fits = _WHOLE_KEYS.fullmatch(text.replace(",", "")) is not None
        return (QValidator.Acceptable if fits else QValidator.Invalid), text, pos


class _DecimalKeys(QValidator):
    """The same for a decimal field: a sign, digits, one point and an exponent. A comma is let stand, so that
    the key shows; it is never read as part of a number, and the field says so when editing ends."""

    def validate(self, text, pos):
        fits = _DECIMAL_KEYS.fullmatch(text) is not None
        return (QValidator.Acceptable if fits else QValidator.Invalid), text, pos


class _NumberField(QLineEdit):
    """What the two number fields share: the floor under the width, and an error that a page, or the field
    itself, puts on the box and takes back. In error the box carries the state the style sheet outlines, and
    the sentence as its tool tip and as its accessible description."""

    error_changed = Signal(str)              # the sentence now in force; "" when the error is taken back

    def __init__(self, text="", parent=None):
        super().__init__(text, parent)
        self._error = ""
        self._tip = ""                       # the tool tip the field had before it was in error
        self.setMinimumWidth(FIELD_MIN_W)

    def error(self) -> str:
        """The error sentence in force, "" for none."""
        return self._error

    def set_error(self, sentence: str = "") -> None:
        """Put the field in error under ``sentence``, or take the error back with an empty one. The tool tip
        the field had comes back with it. Setting the sentence already in force does nothing."""
        if sentence == self._error:
            return
        if not self._error:
            self._tip = self.toolTip()
        self._error = sentence
        set_class(self, state="error" if sentence else "")
        self.setToolTip(sentence if sentence else self._tip)
        self.setAccessibleDescription(sentence)
        self.error_changed.emit(sentence)

    def plain_text(self) -> str:
        """The text to remember for this field: the text as it stands."""
        return self.text()


class FloatField(_NumberField):
    def __init__(self, default: "float | None" = 0.0, parent=None):
        # None is an EMPTY box, not the text "None": a seeded value nobody typed is a different
        # state from a blank one, and value_or_none() is what tells them apart (a new probe row's
        # frequency box starts blank this way). Several callers hand this a pre-formatted string
        # instead of a float, which str() leaves alone.
        super().__init__("" if default is None else str(default), parent)
        self.setValidator(_DecimalKeys(self))
        self._comma_told = False             # True from the moment the field itself refuses a comma
        self.editingFinished.connect(self._editing_ended)
        self.textChanged.connect(self._text_changed)

    def _editing_ended(self):
        if "," in self.text():               # the comma showed while it was typed; now it is refused in words
            self.set_error(COMMA_SENTENCE)
            self._comma_told = True

    def _text_changed(self, text):
        if self._comma_told and "," not in text:
            self._comma_told = False
            if self.error() == COMMA_SENTENCE:       # an error another caller has set since is not this field's
                self.set_error("")

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

    A number field lets '-', '1e', '.' and '+' stand in a box as text still being typed, and
    value_or_none() reads each as None. For the boxes this serves, None MEANS something -- the middle
    of the range two sweeps share, "draw a seed", "the constant is blank" -- so a half-typed number
    silently became that. Only a truly empty box is blank."""
    value = field.value_or_none()
    text = field.text().strip()
    if value is None and text:
        what = describe(key)
        refuse(key, f"{what[0].upper()}{what[1:]} must be a number; got {text!r}.")
    return value


class IntField(_NumberField):
    """A whole number: digits under an optional leading minus. With ``grouped`` the thousands are set off by
    commas, as the digits are typed and when a number is set from code; the caret keeps its place among the
    digits, and what is read and what is remembered is the number without them. While it is typed the digits
    are grouped as they stand: none goes that the person did not take out, so a zero in front stays until it
    is deleted, and is not read."""

    def __init__(self, value=None, parent=None, *, grouped: bool = False):
        # None is an EMPTY box, not the text "None". The line edit is built empty and the text is set
        # afterwards: the toolkit's own constructor would not come through setText below, where a number
        # is grouped.
        super().__init__("", parent)
        self._grouped = grouped
        self._forward = False                # True while a Delete key is handled: it takes what is right of the caret
        self.setValidator(_WholeKeys(self))
        self.textEdited.connect(self._edited)
        if value is not None:
            self.setText(str(value))

    def setText(self, text) -> None:
        """Text set from code, a restored one among it: a whole number is grouped when the field groups,
        and any other text is shown as given."""
        n = _whole(text) if self._grouped else None
        QLineEdit.setText(self, text if n is None else f"{n:,}")

    def keyPressEvent(self, event):
        self._forward = event.key() == Qt.Key_Delete
        try:
            super().keyPressEvent(event)
        finally:
            self._forward = False

    def _edited(self, text):                         # IntField: connected to textEdited (the user's edits only)
        caret = self.cursorPosition()
        kept = text.replace(",", "")
        before = sum(ch != "," for ch in text[:caret])                # the minus counts; a comma does not
        if self._grouped:
            if not _WHOLE_KEYS.fullmatch(kept):      # no number: a text set from code that the keys have not mended
                return                               # yet is left as it stands
            sign = "-" if kept.startswith("-") else ""
            shown = sign + _commas(kept[len(sign):])                  # as the digits stand: a zero typed first stays
            if len(shown) > self.maxLength():        # more than the box holds with its commas: the toolkit would cut
                return                               # the end off, so this too is left as it stands
        else:
            shown = kept
        pos = 0
        while pos < len(shown) and before > 0:
            before -= shown[pos] != ","
            pos += 1
        if self._forward and shown[caret:caret + 1] == "," and shown[:caret] + shown[caret + 1:] == text:
            pos = caret + 1                          # Delete took a comma only: step over it, so the next takes a digit
        if shown != text:
            QLineEdit.setText(self, shown)           # not self.setText: that one is for text set from code
        self.setCursorPosition(pos)

    def value(self) -> int:
        return _whole(self.text()) or 0

    def value_or_none(self) -> "int | None":
        """The field's value, or None when it does not parse -- the IntField twin of FloatField's.

        ``value()`` returns 0 for "" and for "-" mid-typing, and 0 is a legal value for the two
        "0 = automatic" boxes (the rows-per-batch cap, the candidates per sweep round), so a tab
        that wants to refuse a blank instead of reading it as zero has to ask this. The inference
        tabs read every integer box through it at the click, where a blank box is a refusal, never a
        zero.
        """
        return _whole(self.text())

    def plain_text(self) -> str:
        """The text to remember: the number without its commas when the text is a whole number that carries
        them, and otherwise the text as it stands."""
        text = self.text()
        n = _whole(text) if "," in text else None
        return text if n is None else str(n)


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
        path = choose_file(self, "Select file", "", self._filter, save=False)
        if path:
            self.edit.setText(path)

    def value(self) -> str:
        return self.edit.text().strip()
