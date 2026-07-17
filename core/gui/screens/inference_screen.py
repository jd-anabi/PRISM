"""The Parameter Inference section: six tabs over ONE shared SbiSession, with cross-tab gating.

The screen owns the session (the single source of truth); the tabs read it through their ``_screen``
back-reference and never cache it, so Config replacing the session on each build cannot desync them.
``refresh_gates`` is the truth table that greys tabs via setTabEnabled after every stage.
"""
from PySide6.QtWidgets import QLabel, QTabWidget, QVBoxLayout, QWidget

from ..panels.inference_tabs import (ConfigPanel, InferPanel, PosteriorPanel, PriorPanel,
                                     SimulatePanel, ValidatePanel)
from ..session import SbiSession
from ..widgets.anim import fade_in


class InferenceScreen(QWidget):
    def __init__(self, title="Parameter Inference", parent=None):
        super().__init__(parent)
        self.session = SbiSession()

        heading = QLabel(title)
        heading.setStyleSheet("font-size: 16px; font-weight: bold;")

        self.tabs = QTabWidget()
        self.config_panel = ConfigPanel(self)
        self.simulate_panel = SimulatePanel(self)
        self.prior_panel = PriorPanel(self)
        self.posterior_panel = PosteriorPanel(self)
        self.validate_panel = ValidatePanel(self)
        self.infer_panel = InferPanel(self)
        for label, panel in (("Config", self.config_panel), ("Simulate", self.simulate_panel),
                             ("Prior", self.prior_panel), ("Posterior", self.posterior_panel),
                             ("Validate", self.validate_panel), ("Infer", self.infer_panel)):
            self.tabs.addTab(panel, label)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(heading)
        layout.addWidget(self.tabs, 1)

        self.refresh_gates()
        # Connect AFTER refresh_gates so the initial programmatic gating doesn't trigger a fade during
        # construction; a later fall-back-to-Config re-gate will fade, which is fine (no-op under offscreen).
        self.tabs.currentChanged.connect(lambda _i: fade_in(self.tabs.currentWidget()))

    def panels(self):
        return [self.config_panel, self.simulate_panel, self.prior_panel,
                self.posterior_panel, self.validate_panel, self.infer_panel]

    def new_session(self, cfg):
        """Config built (or rebuilt): replace the shared session, repoint the cell-picker tabs to the
        new model, and re-gate. ONE assignment on the owner, so every tab that reads ``self.session``
        next sees the new object."""
        self.session = SbiSession(cfg=cfg)
        self.simulate_panel.on_config_built(cfg)
        self.infer_panel.on_config_built(cfg)
        self.refresh_gates()

    def refresh_gates(self):
        s = self.session
        has_cfg = s.cfg is not None
        has_priors = s.inf_prior is not None and s.force_prior is not None
        can_validate = s.posterior is not None and has_priors      # validate_calibration needs the priors
        can_infer = s.posterior is not None                        # infer_and_visualize does not

        self.tabs.setTabEnabled(1, has_cfg)          # Simulate
        self.tabs.setTabEnabled(2, has_cfg)          # Prior
        self.tabs.setTabEnabled(3, has_cfg)          # Posterior
        self.tabs.setTabEnabled(4, can_validate)     # Validate
        self.tabs.setTabEnabled(5, can_infer)        # Infer

        need_cfg = "" if has_cfg else "Build a config first."
        for i in (1, 2, 3):
            self.tabs.setTabToolTip(i, need_cfg)
        self.tabs.setTabToolTip(4, "" if can_validate else
                                "Needs a posterior AND its prior — build/load a prior, then a posterior.")
        self.tabs.setTabToolTip(5, "" if can_infer else "Train or load a posterior first.")

        for panel in self.panels():
            panel.refresh_local_gates()

        # If the visible tab just got disabled (e.g. re-running the prior greys Validate/Infer), Qt would
        # jump to an arbitrary neighbour; make it deterministic -- fall back to Config.
        if not self.tabs.isTabEnabled(self.tabs.currentIndex()):
            self.tabs.setCurrentIndex(0)
