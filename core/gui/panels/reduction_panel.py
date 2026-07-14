"""REDUCTION mode: the NWK -> Hopf reduction map.

Mirrors cli.build_reduction_config (core/cli.py:451) but with widgets instead of prompts, then runs
the already prompt-free Reduction.sweep.run_reduction_map on a worker. The model is fixed to NADROWSKI
-- the reduction is Nadrowski-specific (core/cli.py:470-472).

This is the only mode that is purely analytical (no SDE simulation), so it is also the quickest way to
sanity-check the whole panel -> worker -> figure path.
"""
from PySide6.QtWidgets import QFormLayout, QGroupBox, QLabel, QPushButton

from core import cli
from core.config import CELL_PATH, PLOT_PATH
from core.Reduction.sweep import run_reduction_map

from .base_panel import BasePanel
from .. import settings
from ..widgets.artifact_picker import ArtifactPicker
from ..widgets.labeled_inputs import FloatField

_MODEL = "NADROWSKI"


class ReductionPanel(BasePanel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.record = None
        self._build_controls()
        self.restore_settings(settings.settings())

    def _build_controls(self):
        box = QGroupBox("NWK → Hopf reduction map")
        form = QFormLayout(box)

        self.cell_picker = ArtifactPicker(CELL_PATH / _MODEL.lower())
        self.f0 = FloatField(0.05)
        self.btn_run = QPushButton("Run reduction map")
        self.btn_run.clicked.connect(self._run)

        form.addRow(QLabel(f"Model is fixed to {_MODEL} (the reduction is Nadrowski-specific)."))
        form.addRow("Cell", self.cell_picker)
        form.addRow("F0 (ND forcing amplitude)", self.f0)
        form.addRow(self.btn_run)

        self.controls_layout.addWidget(box)

    def _run(self):
        cell = self.cell_picker.selected_path()
        if not cell:
            self.log_pane.append_line("Select a cell file first.", "warning")
            return
        try:
            cfg = cli.make_reduction_config(cell, F0=self.f0.value())
        except Exception as e:                       # noqa: BLE001 -- see BasePanel._config_error
            self._config_error(e)
            return

        # run_reduction_map saves its sweep table and its diagnostic PNG itself and returns only the
        # Part-A record, so the figure comes back via the plot watcher rather than the return value.
        self.dispatch(run_reduction_map, cfg, watch_dir=PLOT_PATH, on_result=self._on_result)

    def _on_result(self, record):
        self.record = record
        self.log_pane.append_line("Reduction map complete.")

    def save_settings(self, qs):
        qs.beginGroup("reduction")
        qs.setValue("cell", self.cell_picker.key())
        settings.save_field(qs, "f0", self.f0)
        qs.endGroup()

    def restore_settings(self, qs):
        qs.beginGroup("reduction")
        self.cell_picker.restore_key(settings.get_str(qs, "cell"))
        settings.restore_field(qs, "f0", self.f0)
        qs.endGroup()
