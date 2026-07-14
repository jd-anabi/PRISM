"""A read-only text pane for the pipeline's log output.

Progress does NOT come here -- tqdm bars render in ProgressPane. This pane only ever appends
completed lines, so it can never be polluted by a redrawing bar.
"""
from PySide6.QtWidgets import QPlainTextEdit

_PREFIX = {"warning": "⚠ ", "error": "✖ "}


class LogPane(QPlainTextEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setMaximumBlockCount(5000)          # cap memory on very long runs
        self.setLineWrapMode(QPlainTextEdit.NoWrap)

    def append_line(self, text: str, level: str = "info"):
        self.appendPlainText(_PREFIX.get(level, "") + text)
        self._scroll_to_end()

    def append_lines(self, batch):
        """Append one pump tick's worth of lines: `batch` is a list of (text, level)."""
        if not batch:
            return
        self.appendPlainText("\n".join(_PREFIX.get(level, "") + text for text, level in batch))
        self._scroll_to_end()

    def _scroll_to_end(self):
        sb = self.verticalScrollBar()
        sb.setValue(sb.maximum())
