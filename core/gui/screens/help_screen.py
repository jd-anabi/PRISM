"""Help: what each part's settings mean, in a reference assembled from the user-facing panel module docstrings and
their per-control ``HELP`` dicts -- filling the gap where the app had no section-level "what is this for" text.
``help_text`` builds that reference."""
import html
import importlib

from PySide6.QtWidgets import QLabel, QTextBrowser, QVBoxLayout, QWidget

from ..design import HELP_VIEWER_MIN_H

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


class HelpScreen(QWidget):
    """Help: a heading over the help text, which lays itself out, scrolls itself and can be selected."""

    def __init__(self, parent=None):
        super().__init__(parent)
        heading = QLabel("Help")
        heading.setProperty("type", "heading")
        self.viewer = QTextBrowser()
        self.viewer.setHtml(help_text())
        self.viewer.setOpenExternalLinks(True)
        self.viewer.setMinimumHeight(HELP_VIEWER_MIN_H)
        layout = QVBoxLayout(self)
        layout.addWidget(heading)
        layout.addWidget(self.viewer, 1)

    def first_focus(self) -> QWidget:
        """The control the keyboard starts at: the help text, which the keys scroll."""
        return self.viewer
