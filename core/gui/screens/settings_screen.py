"""The Settings + Help screen (reached from the bottom-left gear's "Full settings…").

Two parts: an Appearance chooser (Follow-system / Light / Dark / Auto) that calls back to MainWindow to
apply + persist the mode, and a per-section Help reference assembled from the user-facing panel module
docstrings and their per-control ``HELP`` dicts -- filling the gap where the app had no section-level
"what is this for" text. Reached only from the gear; the back arrow still returns Home.
"""
import html
import importlib

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QButtonGroup, QGroupBox, QLabel, QRadioButton, QScrollArea, QVBoxLayout,
                               QWidget)

from ..theming import MODE_LABELS

# (module path, human title) for the five user-facing sections, in display order. Each module opens with
# a "<MODE> mode: ..." docstring line and (most) carry a per-control HELP dict.
_HELP_SECTIONS = (
    ("core.gui.panels.reduction_panel", "Reduction Map"),
    ("core.gui.panels.fdt_panel", "FDT Analysis"),
    ("core.gui.panels.crossval_panel", "Cross-validation (sweep study)"),
    ("core.gui.panels.inference_tabs", "Parameter Inference"),
    ("core.gui.panels.simulate_panel", "Simulate"),
)


def _first_paragraph(doc: str) -> str:
    """The first paragraph of a docstring (up to the first blank line) -- the user-facing summary."""
    out = []
    for line in doc.strip().splitlines():
        if not line.strip():
            break
        out.append(line.strip())
    return " ".join(out)


class SettingsScreen(QWidget):
    def __init__(self, on_mode, current_mode="system", parent=None):
        """``on_mode(mode)`` is called when the user picks an appearance; ``current_mode`` pre-selects."""
        super().__init__(parent)
        self._on_mode = on_mode

        heading = QLabel("Settings")
        heading.setStyleSheet("font-size: 16px; font-weight: bold;")

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
        # re-enter MainWindow before it has finished wiring the gear menu + this screen).
        self._radios.get(current_mode, self._radios["system"]).setChecked(True)
        for mode, rb in self._radios.items():
            rb.toggled.connect(lambda checked, m=mode: checked and self._on_mode(m))

        help_box = QGroupBox("Help — what each section does")
        hv = QVBoxLayout(help_box)
        hv.addWidget(self._build_help())

        inner = QWidget()
        iv = QVBoxLayout(inner)
        iv.addWidget(heading)
        iv.addWidget(appearance)
        iv.addWidget(help_box)
        iv.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(inner)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(scroll)

    def set_mode(self, mode: str) -> None:
        """Reflect a mode change that came from the gear popover (without re-firing ``on_mode``)."""
        rb = self._radios.get(mode)
        if rb is not None and not rb.isChecked():
            rb.blockSignals(True)
            rb.setChecked(True)
            rb.blockSignals(False)

    def _build_help(self) -> QLabel:
        parts = []
        for modpath, title in _HELP_SECTIONS:
            try:
                mod = importlib.import_module(modpath)     # already imported by MainWindow -> cache hit
            except Exception:                              # noqa: BLE001 -- Help is best-effort
                continue
            parts.append(f"<h3>{html.escape(title)}</h3>")
            summary = _first_paragraph(mod.__doc__ or "")
            if summary:
                parts.append(f"<p>{html.escape(summary)}</p>")
            help_dict = getattr(mod, "HELP", None)
            if isinstance(help_dict, dict) and help_dict:
                items = "".join(
                    f"<li><b>{html.escape(str(k))}</b>: {html.escape(str(v))}</li>"
                    for k, v in help_dict.items())
                parts.append(f"<ul>{items}</ul>")
        label = QLabel("".join(parts))
        label.setTextFormat(Qt.RichText)
        label.setWordWrap(True)
        label.setAlignment(Qt.AlignTop)
        return label
