"""A tabbed container of figure images. Figures are rendered to PNG on the worker thread (see
BasePanel's fig_sink) and shown here as QPixmaps -- deliberately NOT live FigureCanvasQTAgg widgets:
painting a matplotlib figure that was created on a worker thread deadlocks on matplotlib's global
lock. Static PNGs sidestep that entirely and unify with the FDT/Reduction/CrossVal runners, which
already save PNGs to disk."""
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QLabel, QScrollArea, QTabWidget


class FigureStack(QTabWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTabsClosable(True)
        self.setMovable(True)
        self.tabCloseRequested.connect(self._close_tab)

    def add_figure(self, title: str, png_bytes: bytes):
        """Show a figure rendered to PNG bytes on a worker thread."""
        pix = QPixmap()
        if png_bytes:
            pix.loadFromData(png_bytes, "PNG")
        self._add_pixmap(title, pix)

    def add_png(self, title: str, path):
        """Show a PNG that a runner saved to disk (FDT/Reduction/CrossVal modes)."""
        self._add_pixmap(title, QPixmap(str(path)))

    def _add_pixmap(self, title: str, pix: QPixmap):
        label = QLabel()
        label.setAlignment(Qt.AlignCenter)
        if pix.isNull():
            label.setText(f"(could not render figure: {title})")
        else:
            label.setPixmap(pix)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(label)
        idx = self.addTab(scroll, title)
        self.setCurrentIndex(idx)

    def _close_tab(self, index: int):
        widget = self.widget(index)
        self.removeTab(index)
        widget.deleteLater()

    def clear_all(self):
        while self.count():
            self._close_tab(0)
