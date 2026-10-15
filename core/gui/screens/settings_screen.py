"""The Settings screen, reached from the rail's row Settings.

Three parts: an Appearance chooser (Follow Windows / Light / Dark) that calls back to MainWindow to
apply + persist the mode, the list of the user-defined models, and a per-section Help reference assembled
from the user-facing panel module docstrings and their per-control ``HELP`` dicts -- filling the gap where
the app had no section-level "what is this for" text. ``help_text`` builds that reference, for this screen
and for the Help screen alike.
"""
import html
import importlib

from PySide6.QtWidgets import (QButtonGroup, QGroupBox, QHBoxLayout, QLabel, QPushButton,
                               QRadioButton, QScrollArea, QTextBrowser, QVBoxLayout, QWidget)

from core import registry

from ..design import HELP_VIEWER_MIN_H
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


def help_text() -> str:
    """The help reference as rich text: for each user-facing section, its title, the first paragraph of its
    module's docstring and its per-control ``HELP`` entries. A module that cannot be imported is left out."""
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
    return "".join(parts)


class SettingsScreen(QWidget):
    """Settings + Help: appearance mode, user-model management, per-section help.

    Drives the one appearance callback it is constructed with; persistence of the chosen mode is that
    callback's job (app-level QSettings), not this widget's. The help viewer is assembled from each
    panel module's docstring and HELP dict at build time.
    """

    def __init__(self, on_mode, current_mode="system", parent=None, *,
                 on_open_builder=None, on_edit_model=None, on_delete_model=None):
        """``on_mode(mode)`` is called when the user picks an appearance; ``current_mode`` pre-selects.

        The keyword callbacks wire the "User-defined models" group to MainWindow. All default to None so
        a test can construct the screen with two arguments; the group's buttons are then inert.
        """
        super().__init__(parent)
        self._on_mode = on_mode
        self._on_open_builder = on_open_builder
        self._on_edit_model = on_edit_model
        self._on_delete_model = on_delete_model

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

        models_box = QGroupBox("User-defined models")
        mv = QVBoxLayout(models_box)
        self._models_list = QVBoxLayout()          # rebuilt by refresh_models()
        mv.addLayout(self._models_list)
        btn_new = QPushButton("Open model builder")
        btn_new.setProperty("accent", True)
        btn_new.clicked.connect(lambda: self._on_open_builder and self._on_open_builder())
        mv.addWidget(btn_new)
        self.refresh_models()

        help_box = QGroupBox("Help — what each section does")
        hv = QVBoxLayout(help_box)
        # A word-wrapped QLabel does not propagate heightForWidth through the intervening layouts of a
        # setWidgetResizable QScrollArea, so this blob (five module docstrings plus every HELP dict,
        # several thousand characters) got a height computed for the wrong width and the BOTTOM WAS
        # UNREACHABLE. A QTextBrowser lays out its own rich text and scrolls itself, which sidesteps
        # the propagation problem entirely -- and it makes the help selectable, which a QLabel is not.
        viewer = QTextBrowser()
        viewer.setHtml(help_text())
        viewer.setOpenExternalLinks(True)
        viewer.setMinimumHeight(HELP_VIEWER_MIN_H)
        hv.addWidget(viewer)

        inner = QWidget()
        iv = QVBoxLayout(inner)
        iv.addWidget(heading)
        iv.addWidget(appearance)
        iv.addWidget(models_box)
        iv.addWidget(help_box)
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

    def refresh_models(self) -> None:
        """Rebuild the user-model list (one row per registered model + any startup load failures)."""
        while self._models_list.count():
            item = self._models_list.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)
                w.deleteLater()
        names = registry.user_model_names()
        if not names and not registry.load_errors:
            empty = QLabel("No user-defined models yet. Build one to add it to every model list.")
            empty.setProperty("type", "caption")
            self._models_list.addWidget(empty)
        for name in names:
            row = QWidget()
            h = QHBoxLayout(row)
            h.setContentsMargins(0, 0, 0, 0)
            h.addWidget(QLabel(name))
            h.addStretch(1)
            btn_edit = QPushButton("Edit")
            btn_edit.clicked.connect(lambda _=False, n=name: self._on_edit_model and self._on_edit_model(n))
            btn_del = QPushButton("Delete")
            btn_del.clicked.connect(lambda _=False, n=name: self._on_delete_model and self._on_delete_model(n))
            h.addWidget(btn_edit)
            h.addWidget(btn_del)
            self._models_list.addWidget(row)
        for path, msg in registry.load_errors:
            err = QLabel(f"Failed to load {path.name}: {msg}")
            err.setProperty("type", "caption")
            err.setWordWrap(True)
            self._models_list.addWidget(err)

