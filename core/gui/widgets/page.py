"""The devices every page is built from: the sentence set in the book face, the heading, the page with a middle that scrolls, and two columns that become one."""
import html
import math
import re

from PySide6.QtCore import Qt
from PySide6.QtGui import QFontMetricsF
from PySide6.QtWidgets import (QAbstractButton, QBoxLayout, QFrame, QHBoxLayout, QLabel, QLayout, QScrollArea,
                               QVBoxLayout, QWidget)

from .. import design, fonts
from ..reported import reported

# A word this long, a path most often, cannot wrap, and a label cuts it at its edge with no sign: it may break
# after a backslash, a slash, an underscore or a full stop that has more of the word after it.
_LONG_WORD = re.compile(r"\S{40,}")
_BREAK_AFTER = re.compile(r"(?<=[\\/_.])(?=\S)")
_BREAK = chr(0x200B)                 # a place where a line may break, of no width: the zero-width space


def _breakable(words: str) -> str:
    """``words`` with a place to break after each backslash, slash, underscore and full stop inside a word of
    40 characters or more that has more of the word after it. A shorter word is left as it is."""
    return _LONG_WORD.sub(lambda word: _BREAK_AFTER.sub(_BREAK, word.group()), words)


def _ring_room() -> int:
    """The room a focus ring needs outside a control: the ring's weight and the gap it keeps. Both are line
    weights, so the room does not grow with the text."""
    return design.SIZE["focus_ring"] + design.SIZE["focus_gap"]


def bold_weight(family: str) -> int:
    """The weight to ask for bold in a face: 600 in the lettering face, whose two files end there, 700 in a book face."""
    return 600 if family == design.LETTERING else 700


def _tell(part, small: bool) -> None:
    """Tell ``part`` the page's form when it offers ``set_small``. A layout is looked into when it is told, so that
    every widget in it, or in a layout inside it, that offers ``set_small`` is told, one put there after the
    layout was handed in among them."""
    if isinstance(part, QLayout):
        for index in range(part.count()):
            item = part.itemAt(index)
            widget, inner = item.widget(), item.layout()
            if widget is not None:
                _tell(widget, small)
            elif inner is not None:
                _tell(inner, small)
        return
    tell = getattr(part, "set_small", None)
    if tell is not None:
        tell(small)


class Sentence(QLabel):
    """One sentence in the book face, set at a fixed line height. Its text is shown as written: nothing in it
    is read as markup, and a word of 40 characters or more may break after a backslash, a slash, an underscore
    or a full stop. The opening words may be bold; a quiet sentence is set in the second ink. Its words are
    changed by ``set_text`` and never by the toolkit's own ``setText``, which reads them as markup and whose
    words the next change of theme writes over with the old ones."""

    def __init__(self, text: str = "", *, quiet: bool = False) -> None:
        super().__init__()
        self._quiet = quiet
        self._text, self._lead = "", None
        self.setTextFormat(Qt.RichText)
        self.setWordWrap(True)
        self.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        # The font is given here, when the label is built, and nowhere later: a font handed to a widget after
        # a sheet has dressed it is dropped at the next change of sheet.
        face = fonts.book()
        self.setFont(face)
        self._line = design.scaled(design.TYPE["book"][1])   # the one line height, whichever face was found
        # Rich text puts all the extra of a fixed line height above the line. Half of it is taken off the top
        # and given back at the foot, so the letters stand in the middle of their line.
        lift = max(0, math.ceil((self._line - QFontMetricsF(face).lineSpacing()) / 2))
        self.setContentsMargins(0, -lift, 0, lift)
        # A bare bold asks for 700, which the lettering face's two files do not hold. The weight is read off the
        # font just built, whose face is the one found, so the toolkit's list of every family is not asked for.
        self._bold = bold_weight(face.family())
        self.set_text(text)

    def set_text(self, text: str, *, lead: str | None = None) -> None:
        """Show ``text`` after ``lead``, its bold opening words; no lead, or an empty one, drops them. The words
        as handed in are the label's accessible name: what it shows may hold places to break."""
        self._text, self._lead = text, lead or None
        self.setAccessibleName(self.plain_text())
        self.retheme()

    def plain_text(self) -> str:
        """What is shown, as it was handed in: the lead, a space, the text."""
        return " ".join(words for words in (self._lead, self._text) if words)

    def retheme(self) -> None:
        """Write the text again in the colours now in force: a quiet sentence names its colours in its text."""
        self.setText(self._markup())

    def _markup(self) -> str:
        """The one place where a caller's words meet markup: both strings are given their places to break, then
        escaped, and the paragraph keeps its spaces and its line breaks. Nothing is written for no words: an
        empty paragraph has a height."""
        if not self._text and not self._lead:
            return ""
        style = f"line-height:{self._line}px; white-space:pre-wrap"
        if self._quiet:
            style += f"; color:{design.hex_colour('ink_2')}"
        lead = ""
        if self._lead:
            lead_style = f"font-weight:{self._bold}"
            if self._quiet:
                lead_style += f"; color:{design.hex_colour('ink')}"
            lead = f"<span style='{lead_style}'>{html.escape(_breakable(self._lead), quote=False)}</span>"
        body = html.escape(_breakable(self._text), quote=False)
        return f"<p style='{style}'>{lead}{' ' if lead and body else ''}{body}</p>"


class PageHead(QWidget):
    """A page's heading: its title, one sentence under it in the book face, its actions at the right. The
    sentence is changed by ``set_sentence``, never by the toolkit's own ``setText`` nor by the sentence's own
    ``set_text``: only ``set_sentence`` hides a sentence left with no words, and shows it again."""

    def __init__(self, title: str, sentence: str = "") -> None:
        super().__init__()
        self._small = False
        self._row = QHBoxLayout(self)
        self._row.setContentsMargins(0, 0, 0, 0)
        self._row.setSpacing(design.scaled(design.GAP["head_buttons"]))
        words = QVBoxLayout()
        words.setContentsMargins(0, 0, 0, 0)
        words.setSpacing(design.scaled(design.GAP["head_lead"]))
        self.title = QLabel(title)
        self.title.setTextFormat(Qt.PlainText)               # a name may hold "<" or "&"
        # Its font is given at build and it takes nothing from the sheet: a font handed to a widget after a
        # sheet has dressed it is dropped at the next change of sheet.
        self.title.setFont(fonts.inter("title"))
        self.title.setFixedHeight(design.scaled(design.TYPE["title"][1]))
        words.addWidget(self.title)                          # as wide as its column: the actions begin where it ends
        self.sentence = Sentence(sentence, quiet=True)
        words.addWidget(self.sentence)
        words.addStretch(1)
        self._row.addLayout(words, 1)
        self._show_sentence()                                # now that it is this heading's child, no window of its own

    def add_action(self, button: QAbstractButton) -> None:
        """Put an action at the right of the heading, level with its top. The actions stand in the order
        added, left to right."""
        self._row.addWidget(button, 0, Qt.AlignTop)

    def set_sentence(self, text: str) -> None:
        """Change the sentence under the title; an empty one takes no room."""
        self.sentence.set_text(text)
        self._show_sentence()

    def set_small(self, small: bool) -> None:
        """The low window: the sentence goes, and comes back when the window is high enough again."""
        self._small = bool(small)
        self._show_sentence()

    def _show_sentence(self) -> None:
        """The sentence is in sight only while it has words and the page is not small."""
        self.sentence.setVisible(bool(self.sentence.plain_text()) and not self._small)


class Scroller(QScrollArea):
    """The middle of a page. It scrolls when the window is too low for what it holds, keeps the room a focus
    ring needs round it, and tells it how wide the page is, so that two columns can become one."""

    def __init__(self, body: QWidget) -> None:
        super().__init__()
        self.body = body
        # The frame is switched off here, at build: the sheet gives a line to a scrolling area that still has
        # the frame it is born with, and reads that when the area is dressed.
        self.setFrameShape(QFrame.NoFrame)
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setFocusPolicy(Qt.NoFocus)                      # a stop for the Tab key only while it scrolls
        self.setProperty("ring", "inset")                    # its focus ring is drawn just inside its edge
        holder = QWidget()
        self._holder = QVBoxLayout(holder)
        self._holder.setSpacing(0)
        self._holder.addWidget(body)
        self._holder.addStretch(1)
        room = _ring_room()
        self._holder.setContentsMargins(room, 0, room, room)
        self.setWidget(holder)
        # A bound method, never a function that holds the scroller: connected to its own bar, such a function
        # would keep a scroller with no parent alive after its last name is gone.
        self.verticalScrollBar().rangeChanged.connect(self._range_changed)

    @reported()
    def _range_changed(self, low, high):
        """Hears the scroll bar's range. While there is something to scroll, the bar gets a gap beside it
        and the middle is a stop for the Tab key, so that the keyboard can scroll it."""
        scrolls = high > low
        room = _ring_room()
        self._holder.setContentsMargins(room, 0, room + (design.SIZE["scroll_gap"] if scrolls else 0), room)
        self.setFocusPolicy(Qt.TabFocus if scrolls else Qt.NoFocus)

    @reported(QScrollArea.resizeEvent)
    def resizeEvent(self, event):
        fit = getattr(self.body, "fit", None)                # asked at every resize: a body need not offer one
        if fit is not None:                                  # the width it has while the bar and its gap are there
            fit(self.width() - 2 * _ring_room() - design.SIZE["scroll_bar"] - design.SIZE["scroll_gap"])
        super().resizeEvent(event)                           # last: the body is laid out in the form it just took


class Columns(QWidget):
    """Two blocks side by side while the page is wide enough for both, one above the other when it is not."""

    def __init__(self, left: QWidget, right: QWidget, *, stretch: tuple[int, int] = (1, 1),
                 gap: int | None = None, gap_stacked: int | None = None) -> None:
        super().__init__()
        self._blocks = (left, right)
        # a gap handed in is used as given, so a caller scales it; the look's own grow with the text
        self._gaps = (design.scaled(design.SIZE["col_gap"]) if gap is None else gap,
                      design.scaled(design.GAP["columns_stacked"]) if gap_stacked is None else gap_stacked)
        self.box = QBoxLayout(QBoxLayout.LeftToRight, self)
        self.box.setContentsMargins(0, 0, 0, 0)
        self.box.setSpacing(self._gaps[0])
        self.box.addWidget(left, stretch[0])
        self.box.addWidget(right, stretch[1])

    def fit(self, width: int) -> None:
        """Set the two blocks side by side when ``width`` has room for both, else one above the other. Both
        need the width the look names, which grows with the text as far as the screen offers, and their own
        least widths with the gap between them."""
        # a widget with no layout answers no least size of its own, whatever least width it was given
        least = (max(block.minimumSizeHint().width(), block.minimumWidth(), 0) for block in self._blocks)
        need = sum(least) + self._gaps[0]
        wide = width >= max(design.two_columns_from(), need)
        want = QBoxLayout.LeftToRight if wide else QBoxLayout.TopToBottom
        if self.box.direction() != want:
            self.box.setDirection(want)
            self.box.setSpacing(self._gaps[0 if wide else 1])

    def stacked(self) -> bool:
        """Whether the two blocks stand one above the other."""
        return self.box.direction() == QBoxLayout.TopToBottom


class Page(QWidget):
    """A page of the program: parts that stay where they are and a middle that scrolls."""

    def __init__(self) -> None:
        super().__init__()
        self._small = False
        self._told: list = []                # what was handed in: each is told when the window is low
        self._gaps: list = []                # (the space, its usual height, its height in a low window)
        self.box = QVBoxLayout(self)
        self.box.setSpacing(0)
        self._pad()

    def _pad(self) -> None:
        """The page's padding, which grows with the text, less the ring's room at each side: every part
        brings that room with it, the middle for the focus ring and a fixed part to line up with it."""
        pad = design.SIZE["page_pad_small" if self._small else "page_pad"]
        left, top, right, bottom = (design.scaled(n) for n in pad)
        room = _ring_room()
        self.box.setContentsMargins(left - room, top, right - room, bottom)

    def add_fixed(self, part: QWidget | QLayout) -> None:
        """Add a part that stays where it is. It is set in from the page's sides by the room the middle
        keeps for a focus ring, so that the two line up. It is told the form the page is in now, and every
        later change of form."""
        row = QHBoxLayout()
        room = _ring_room()
        row.setContentsMargins(room, 0, room, 0)
        if isinstance(part, QLayout):
            row.addLayout(part)
        else:
            row.addWidget(part)
        self.box.addLayout(row)
        self._told.append(part)
        _tell(part, self._small)

    def add_gap(self, usual: int, small: int | None = None) -> None:
        """Add a space ``usual`` pixels high, which the low window makes ``small`` pixels high when that is
        handed in; it is added at the height of the form the page is in now. The heights are used as given,
        so a caller scales them."""
        gap = QWidget()
        low = usual if small is None else small
        gap.setFixedHeight(low if self._small else usual)
        self.box.addWidget(gap)
        self._gaps.append((gap, usual, low))

    def add_scrolling(self, body: QWidget) -> Scroller:
        """Add the middle: ``body`` in a scroller that takes the room the other parts leave, told the form the
        page is in now. Returns the scroller."""
        scroller = Scroller(body)
        self.box.addWidget(scroller, 1)
        self._told.append(body)
        _tell(body, self._small)
        return scroller

    def set_small(self, small: bool) -> None:
        """The low window, or the ordinary one again: the padding and the spaces take their other size, and
        every part handed in that offers ``set_small`` is told, and every such widget inside a layout handed
        in."""
        self._small = bool(small)
        self._pad()
        for gap, usual, low in self._gaps:
            gap.setFixedHeight(low if self._small else usual)
        for part in self._told:
            _tell(part, self._small)

    def is_small(self) -> bool:
        """Whether the page is in the form of the low window."""
        return self._small

    def first_focus(self) -> QWidget | None:
        """The control the keyboard starts at when the page comes up. None here; a page with settings
        answers its first."""
        return None
