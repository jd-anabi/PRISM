"""The Settings screen, reached from the rail's row Settings.

An Appearance chooser (Follow Windows / Light / Dark) that calls back to MainWindow to apply and persist the
mode. The list of the user-defined models stands under Your own models, and the help text under Help.
"""
from PySide6.QtWidgets import QButtonGroup, QGroupBox, QLabel, QRadioButton, QScrollArea, QVBoxLayout, QWidget

from ..theming import MODE_LABELS


class SettingsScreen(QWidget):
    """Settings: the appearance mode.

    Drives the one appearance callback it is constructed with; persistence of the chosen mode is that
    callback's job (app-level QSettings), not this widget's.
    """

    def __init__(self, on_mode, current_mode="system", parent=None):
        """``on_mode(mode)`` is called when the user picks an appearance; ``current_mode`` pre-selects."""
        super().__init__(parent)
        self._on_mode = on_mode

        heading = QLabel("Settings")
        heading.setProperty("type", "heading")     # Fluent type ramp (global QSS)

        appearance = QGroupBox("Appearance")
        av = QVBoxLayout(appearance)
        self._group = QButtonGroup(self)
        self._radios = {}
        for mode, label in MODE_LABELS:
            rb = QRadioButton(label)
            self._group.addButton(rb)
            self._radios[mode] = rb
            av.addWidget(rb)
        # Select the current mode BEFORE connecting, so construction never fires on_mode (which would
        # re-enter MainWindow before it has finished wiring this screen).
        self._radios.get(current_mode, self._radios["system"]).setChecked(True)
        for mode, rb in self._radios.items():
            rb.toggled.connect(lambda checked, m=mode: checked and self._on_mode(m))

        inner = QWidget()
        iv = QVBoxLayout(inner)
        iv.addWidget(heading)
        iv.addWidget(appearance)
        iv.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(inner)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(scroll)

    def set_mode(self, mode: str) -> None:
        """Reflect a mode change made elsewhere (without re-firing ``on_mode``)."""
        rb = self._radios.get(mode)
        if rb is not None and not rb.isChecked():
            rb.blockSignals(True)
            rb.setChecked(True)
            rb.blockSignals(False)
