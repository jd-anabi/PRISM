"""FDT mode: the fluctuation-dissipation-theorem analysis.

Mirrors cli.build_fdt_config (core/cli.py:431) with widgets, then runs FDT.fdt_pipeline.run_fdt on a
worker.

The two checkboxes are load-bearing. run_fdt's `skip_sanity` / `confirm_production` default to None,
which means "ask via input()" -- that is the CLI path (core/FDT/fdt_pipeline.py:69-80). A GUI MUST pass
explicit booleans; leaving them None would block the worker forever on an input() nobody can answer.
"""
from PySide6.QtWidgets import QCheckBox, QComboBox, QFormLayout, QGroupBox, QPushButton

from core import cli
from core.config import CELL_PATH, PLOT_PATH, VALID_MODELS
from core.FDT.fdt_pipeline import run_fdt

from .base_panel import BasePanel
from .. import settings
from ..widgets.artifact_picker import ArtifactPicker
from ..widgets.labeled_inputs import FloatField, IntField


def _run_fdt_guarded(cfg, *, skip_sanity, confirm_production):
    """run_fdt is Nadrowski-coupled: it reads params_dict["n"] / ["beta"], so it KeyErrors on the
    hopf and bp cells. That is a pre-existing pipeline limitation, not a GUI one -- catch it and say
    so plainly rather than dumping a bare KeyError traceback on the user."""
    try:
        return run_fdt(cfg, skip_sanity=skip_sanity, confirm_production=confirm_production)
    except KeyError as e:
        raise RuntimeError(
            f"The FDT pipeline needs the Nadrowski parameter {e} and the selected "
            f"{cfg.model} cell does not define it. FDT analysis currently only supports "
            f"NADROWSKI cells."
        ) from e


class FdtPanel(BasePanel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_controls()
        self.restore_settings(settings.settings())

    def _build_controls(self):
        box = QGroupBox("FDT analysis")
        form = QFormLayout(box)

        self.model_combo = QComboBox()
        self.model_combo.addItems(VALID_MODELS)
        self.model_combo.setCurrentText("NADROWSKI")
        self.model_combo.currentTextChanged.connect(self._on_model_changed)

        self.cell_picker = ArtifactPicker(CELL_PATH / "nadrowski")
        self.n_freqs = IntField(60)
        self.ensemble_m = IntField(256)
        self.freqs_per_batch = IntField(1)
        self.f0 = FloatField(0.05)

        self.skip_sanity = QCheckBox("Skip sanity checks")
        self.confirm_production = QCheckBox("Proceed to the production sweep after sanity")
        self.confirm_production.setChecked(True)
        self.skip_sanity.toggled.connect(
            lambda on: self.confirm_production.setEnabled(not on))   # only consulted when sanity runs

        self.btn_run = QPushButton("Run FDT analysis")
        self.btn_run.clicked.connect(self._run)

        form.addRow("Model", self.model_combo)
        form.addRow("Cell", self.cell_picker)
        form.addRow("n_freqs", self.n_freqs)
        form.addRow("ensemble_M", self.ensemble_m)
        form.addRow("freqs_per_batch", self.freqs_per_batch)
        form.addRow("F0 (ND forcing amplitude)", self.f0)
        form.addRow(self.skip_sanity)
        form.addRow(self.confirm_production)
        form.addRow(self.btn_run)

        self.controls_layout.addWidget(box)

    def _on_model_changed(self, model: str):
        self.cell_picker.base_path = CELL_PATH / model.lower()
        self.cell_picker.refresh()

    def _run(self):
        cell = self.cell_picker.selected_path()
        if not cell:
            self.log_pane.append_line("Select a cell file first.", "warning")
            return
        model = self.model_combo.currentText()
        try:
            cfg = cli.make_fdt_config(
                model, "nadrowski" in model.lower(), cell,
                n_freqs=self.n_freqs.value(), ensemble_M=self.ensemble_m.value(),
                freqs_per_batch=self.freqs_per_batch.value(), F0=self.f0.value())
        except Exception as e:                       # noqa: BLE001 -- see BasePanel._config_error
            self._config_error(e)
            return

        # Explicit bools, never None -- see the module docstring.
        self.dispatch(_run_fdt_guarded, cfg, watch_dir=PLOT_PATH,
                      skip_sanity=self.skip_sanity.isChecked(),
                      confirm_production=self.confirm_production.isChecked(),
                      on_finished=lambda: self.log_pane.append_line("FDT run finished."))

    def save_settings(self, qs):
        qs.beginGroup("fdt")
        qs.setValue("model", self.model_combo.currentText())
        qs.setValue("cell", self.cell_picker.key())
        for name, fld in (("n_freqs", self.n_freqs), ("ensemble_m", self.ensemble_m),
                          ("freqs_per_batch", self.freqs_per_batch), ("f0", self.f0)):
            settings.save_field(qs, name, fld)
        settings.set_bool(qs, "skip_sanity", self.skip_sanity.isChecked())
        settings.set_bool(qs, "confirm_production", self.confirm_production.isChecked())
        qs.endGroup()

    def restore_settings(self, qs):
        qs.beginGroup("fdt")
        # Model FIRST, and drive _on_model_changed explicitly: currentTextChanged won't fire if the
        # saved value equals the current text, and the picker's base_path must be repointed before the
        # cell key is restored (otherwise refresh() would clear the just-set selection). Feed it the
        # combo's ACCEPTED text, not the raw saved string -- a corrupt saved model a non-editable combo
        # rejects would otherwise point the cell picker at a nonexistent folder.
        self.model_combo.setCurrentText(settings.get_str(qs, "model", self.model_combo.currentText()))
        self._on_model_changed(self.model_combo.currentText())
        self.cell_picker.restore_key(settings.get_str(qs, "cell"))
        for name, fld in (("n_freqs", self.n_freqs), ("ensemble_m", self.ensemble_m),
                          ("freqs_per_batch", self.freqs_per_batch), ("f0", self.f0)):
            settings.restore_field(qs, name, fld)
        self.skip_sanity.setChecked(settings.get_bool(qs, "skip_sanity", False))
        self.confirm_production.setChecked(settings.get_bool(qs, "confirm_production", True))
        qs.endGroup()
