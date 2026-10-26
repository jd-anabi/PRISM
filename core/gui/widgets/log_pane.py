"""A read-only pane of text lines: a page's own lines under its figures, and a run's full text on its page.

It holds the newest ``LINES_KEPT`` lines. Given a head, as the run's full text is given "Earlier lines are in the
file.", it shows the head as its first line once older lines have gone. It follows the end only while it is at
the end: scrolled up, a line that arrives moves nothing, and a line that goes from the top leaves what is in sight
where it was. A warning and an error carry a mark before their words; a line of the error stream and the window's
own line carry none. Progress bars never come here: their frames are read as rows, not lines.
"""
from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import QPlainTextEdit

#: The lines a pane keeps, the newest.
LINES_KEPT = 5000
_PREFIX = {"warning": "⚠ ", "error": "✖ "}


class LogPane(QPlainTextEdit):
    """``head``: the line shown first once older lines have gone; none when empty."""

    def __init__(self, parent=None, *, head: str = ""):
        super().__init__(parent)
        self.setReadOnly(True)
        # The pane keeps its own count of lines: the toolkit's own limit would cut more than one line at a time
        # with no regard for what is in sight, and would take the head with them. Nothing is kept to undo either:
        # every line that went would be held a second time.
        self.setUndoRedoEnabled(False)
        # WRAP. Panels write long single-line diagnostics (the f_scale warning is ~230 chars), and
        # with NoWrap the user saw the first ~60 characters and had to scroll horizontally to reach
        # the part that says what to do about it.
        self.setLineWrapMode(QPlainTextEdit.WidgetWidth)
        self._head = head
        self._cut = False
        self._following = True
        self.verticalScrollBar().valueChanged.connect(self._scrolled)

    def append_line(self, text: str, level: str = "info") -> None:
        self.append_lines([(text, level)])

    def append_lines(self, batch) -> None:
        """Append one batch of lines, each ``(text, level)``."""
        if not batch:
            return
        bar = self.verticalScrollBar()
        top = self.firstVisibleBlock()
        top_number, within = top.blockNumber(), bar.value() - top.firstLineNumber()
        # a text written whole (a traceback) may end in a line end, which would leave an empty line after it
        self.appendPlainText("\n".join(_PREFIX.get(level, "") + text.rstrip("\n") for text, level in batch))
        start, gone, added = self._trim()
        if self._following:
            bar.setValue(bar.maximum())
        elif gone or added:
            # what was in sight stays in sight: the block that stood at the top, moved up by what went above it
            # and down by the head put above it; a block that went itself leaves the oldest line kept at the top
            number = top_number if top_number < start else max(top_number - gone, start) + added
            block = self.document().findBlockByNumber(number)
            bar.setValue(block.firstLineNumber() + max(0, within))

    def is_cut(self) -> bool:
        """Whether older lines have gone."""
        return self._cut

    def lines(self) -> list[str]:
        """The lines held, as shown, the head left out."""
        lines = self.toPlainText().split("\n")
        return lines[1:] if self._cut and self._head else lines

    def clear_lines(self) -> None:
        """Hold no line, and follow the end again."""
        self.clear()
        self._cut, self._following = False, True

    def find_next(self, words: str) -> bool:
        """Select the next place that holds ``words`` after the one selected, from the top again after the
        end, and show it; False, with nothing changed, when no line holds them."""
        if not words:
            return False
        if self.find(words):
            return True
        kept = self.textCursor()
        self.moveCursor(QTextCursor.Start)
        if self.find(words):
            return True
        self.setTextCursor(kept)
        return False

    def _trim(self) -> tuple[int, int, int]:
        """Let the lines beyond ``LINES_KEPT`` go, the oldest first, below the head once it stands; the head is
        put first at the first cut. Returns the first block number that went, how many went and how many blocks
        were put above them (the head, at the first cut)."""
        document = self.document()
        heads = 1 if self._cut and self._head else 0
        extra = document.blockCount() - heads - LINES_KEPT
        if extra <= 0:
            return 0, 0, 0
        cursor = QTextCursor(document)
        cursor.setPosition(document.findBlockByNumber(heads).position())
        cursor.setPosition(document.findBlockByNumber(heads + extra).position(), QTextCursor.KeepAnchor)
        cursor.removeSelectedText()
        added = 0
        if self._head and not self._cut:
            cursor.setPosition(0)
            cursor.insertText(self._head + "\n")
            added = 1
        self._cut = True
        return heads, extra, added

    def _scrolled(self, value: int) -> None:
        """Hears the scroll bar: the pane follows the end from now on only while it stands at the end."""
        self._following = value >= self.verticalScrollBar().maximum()
