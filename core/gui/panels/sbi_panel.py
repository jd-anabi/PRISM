"""SBI mode panel: drive the decomposed pipeline stage by stage (Config -> Prior -> Posterior ->
Validate -> Infer) over an SbiSession, running heavy stages on a background worker and embedding the
figures each stage produces. Reuses cli.make_sim_config + the orchestrator stage functions (with the
fig_sink / save=False hooks added in Phase 0), so the CLI path is untouched."""
from PySide6.QtWidgets import (QComboBox, QFormLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
                               QPushButton, QStackedWidget, QVBoxLayout, QWidget)

from core import cli, orchestrator
from core.Helpers import file_manager, visualizers
from core.config import (VALID_MODELS, VALID_LABELS, BOUNDS_PATH, CELL_PATH, PRIOR_PATH,
                         POSTERIOR_PATH, T_MIN_EXP_S)

from .base_panel import BasePanel
from .. import settings
from ..session import SbiSession
from ..widgets.artifact_picker import ArtifactPicker
from ..widgets.labeled_inputs import FloatField, PathField


# ── inference runners (module-level so a Worker can call them with an injected fig_sink) ──────────
def _run_simulated_inference(cfg, posterior, cell_path, T_obs_s, *, fig_sink=None):
    """Mirror orchestrator.run's simulated branch: inject GT + T_obs, simulate, show GT trace + infer."""
    cli.load_and_validate_gt(cfg, cell_path)
    cfg.T_obs = T_obs_s * cfg.get_unit_conversion_factor("s")
    x_dim, obs_stats, t_dim = orchestrator.generate_observations(cfg)
    visualizers.plot(t_dim.squeeze(0).cpu().detach().numpy(), x_dim[0, :].cpu().detach().numpy(),
                     title="Ground-truth trace", labels=("t", "x"), sink=fig_sink)
    orchestrator.infer_and_visualize(cfg, posterior, obs_stats, x_dim, t_dim, show_truth=True, fig_sink=fig_sink)


def _run_experimental_inference(cfg, posterior, spont_path, forced_path, T_obs_s, forcing_si, *, fig_sink=None):
    """Mirror orchestrator.run's experimental branch."""
    x_spont = file_manager.load_experimental_data(spont_path, dtype=cfg.hw.dtype)
    x_forced = file_manager.load_experimental_data(forced_path, dtype=cfg.hw.dtype)
    obs_stats, obs_data, t_dim = orchestrator.build_experiment_obs(cfg, x_spont, x_forced, T_obs_s, forcing_si)
    orchestrator.infer_and_visualize(cfg, posterior, obs_stats, obs_data, t_dim, show_truth=False, fig_sink=fig_sink)


class SbiPanel(BasePanel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.session = SbiSession()
        self._forcing_fields = {}          # name -> FloatField (experimental drive)
        self._build_controls()
        self.restore_settings(settings.settings())
        self._refresh_gates()

    # ── UI construction ──────────────────────────────────────────────────────
    def _build_controls(self):
        cl = self.controls_layout

        # 1. Config
        g_cfg = QGroupBox("1 · Config")
        f = QFormLayout(g_cfg)
        self.model_combo = QComboBox()
        self.model_combo.addItems(VALID_MODELS)
        self.model_combo.setCurrentText("NADROWSKI")
        self.bounds_picker = ArtifactPicker(BOUNDS_PATH / "nadrowski")
        self.model_combo.currentTextChanged.connect(self._on_model_changed)
        self.btn_config = QPushButton("Build config")
        self.btn_config.clicked.connect(self._build_config)
        f.addRow("Model", self.model_combo)
        f.addRow("Bounds", self.bounds_picker)
        f.addRow(self.btn_config)
        cl.addWidget(g_cfg)

        # 2. Prior
        g_prior = QGroupBox("2 · Prior")
        v = QVBoxLayout(g_prior)
        self.prior_picker = ArtifactPicker(PRIOR_PATH, keep=lambda fn: fn.endswith(".pt"), allow_new=True)
        self.btn_prior = QPushButton("Build / Load prior")
        self.btn_prior.clicked.connect(self._build_prior)
        v.addWidget(self.prior_picker)
        v.addWidget(self.btn_prior)
        v.addLayout(self._save_row("prior"))
        cl.addWidget(g_prior)

        # 3. Posterior
        g_post = QGroupBox("3 · Posterior")
        v = QVBoxLayout(g_post)
        self.post_picker = ArtifactPicker(
            POSTERIOR_PATH, keep=lambda fn: fn.endswith(".pt") and not fn.endswith(".rot.pt"), allow_new=True)
        self.btn_post = QPushButton("Train / Load posterior")
        self.btn_post.clicked.connect(self._build_posterior)
        v.addWidget(self.post_picker)
        v.addWidget(self.btn_post)
        v.addLayout(self._save_row("posterior"))
        cl.addWidget(g_post)

        # 4. Validate
        g_val = QGroupBox("4 · Validate (SBC + TARP)")
        v = QVBoxLayout(g_val)
        self.btn_validate = QPushButton("Run calibration")
        self.btn_validate.clicked.connect(self._validate)
        v.addWidget(self.btn_validate)
        cl.addWidget(g_val)

        # 5. Infer
        g_inf = QGroupBox("5 · Infer")
        v = QVBoxLayout(g_inf)
        self.infer_mode = QComboBox()
        self.infer_mode.addItems(["Simulated (cell ground truth)", "Experimental data"])
        self.infer_mode.currentIndexChanged.connect(lambda i: self.infer_stack.setCurrentIndex(i))
        v.addWidget(self.infer_mode)

        self.infer_stack = QStackedWidget()
        # simulated inputs
        sim_w = QWidget(); sim_f = QFormLayout(sim_w)
        self.cell_picker = ArtifactPicker(CELL_PATH / "nadrowski")
        self.sim_tobs = FloatField(T_MIN_EXP_S)
        sim_f.addRow("Cell", self.cell_picker)
        sim_f.addRow("T_obs (s)", self.sim_tobs)
        self.infer_stack.addWidget(sim_w)
        # experimental inputs
        exp_w = QWidget(); self.exp_form = QFormLayout(exp_w)
        self.exp_spont = PathField()
        self.exp_forced = PathField()
        self.exp_tobs = FloatField(T_MIN_EXP_S)
        self.exp_form.addRow("Spontaneous", self.exp_spont)
        self.exp_form.addRow("Forced", self.exp_forced)
        self.exp_form.addRow("T_obs (s)", self.exp_tobs)
        self._forcing_anchor = QLabel("(build config to list drive params)")
        self.exp_form.addRow(self._forcing_anchor)
        self.infer_stack.addWidget(exp_w)
        v.addWidget(self.infer_stack)

        self.btn_infer = QPushButton("Run inference")
        self.btn_infer.clicked.connect(self._infer)
        v.addWidget(self.btn_infer)
        cl.addWidget(g_inf)

    def _save_row(self, kind: str) -> QHBoxLayout:
        row = QHBoxLayout()
        name = QLineEdit(); name.setPlaceholderText(f"name to save {kind} as…")
        btn = QPushButton("Save")
        if kind == "prior":
            self.prior_name, self.btn_save_prior = name, btn
            btn.clicked.connect(self._save_prior)
        else:
            self.post_name, self.btn_save_post = name, btn
            btn.clicked.connect(self._save_posterior)
        row.addWidget(name, 1); row.addWidget(btn)
        return row

    # ── stage handlers ───────────────────────────────────────────────────────
    def _on_model_changed(self, model: str):
        # repoint the bounds + cell pickers to the selected model's subfolder
        self.bounds_picker.base_path = BOUNDS_PATH / model.lower(); self.bounds_picker.refresh()
        self.cell_picker.base_path = CELL_PATH / model.lower(); self.cell_picker.refresh()

    def _build_config(self):
        model = self.model_combo.currentText()
        labels = VALID_LABELS[VALID_MODELS.index(model)]
        state_dep_drift = "nadrowski" in model.lower()
        bounds_path = self.bounds_picker.selected_path()
        if not bounds_path:
            self.log_pane.append_line("Select a bounds file first.", "warning")
            return
        try:
            cfg = cli.make_sim_config(model, labels, state_dep_drift, bounds_path)
        except Exception as e:                       # noqa: BLE001 -- see BasePanel._config_error
            self._config_error(e)
            return
        self.session = SbiSession(cfg=cfg)
        self._rebuild_forcing_fields(cfg)
        self.log_pane.append_line(
            f"Config built: {model} — {len(cfg.params_dict)} ND + {len(cfg.rescale_params)} rescale params.")
        self._refresh_gates()

    def _build_prior(self):
        if self.session.cfg is None:
            return
        entry, is_new = self.prior_picker.selected()
        self.session.reset_downstream("prior")
        self.dispatch(orchestrator.build_prior, self.session.cfg, entry, is_new, save=False,
                      provide_fig_sink=True, on_result=self._on_prior)

    def _on_prior(self, payload):
        self.session.inf_prior, self.session.force_prior = payload
        self.log_pane.append_line("Prior ready.")
        self._refresh_gates()

    def _save_prior(self):
        name = self.prior_name.text().strip()
        if not name or self.session.inf_prior is None:
            self.log_pane.append_line("Build a prior and enter a name first.", "warning")
            return
        nd_prior = self.session.inf_prior.distributions[0]
        self.dispatch(orchestrator.save_prior_artifacts, name, nd_prior, self.session.cfg,
                      on_finished=lambda: (self.prior_picker.refresh(),
                                           self.log_pane.append_line(f"Saved prior '{name}'.")))

    def _build_posterior(self):
        if self.session.inf_prior is None:
            return
        entry, is_new = self.post_picker.selected()
        self.session.reset_downstream("posterior")
        self.dispatch(orchestrator.build_posterior, self.session.cfg, self.session.inf_prior,
                      self.session.force_prior, entry, is_new, save=False,
                      provide_fig_sink=True, on_result=self._on_posterior)

    def _on_posterior(self, payload):
        self.session.posterior, self.session.diagnostics = payload
        self.session.posterior_latent = getattr(self.session.posterior, "latent", None)
        self.session.V = self._extract_rotation(self.session.posterior)
        self.log_pane.append_line("Posterior ready.")
        self._refresh_gates()

    def _save_posterior(self):
        name = self.post_name.text().strip()
        if not name or self.session.posterior_latent is None:
            self.log_pane.append_line("Train a posterior and enter a name first.", "warning")
            return
        self.dispatch(orchestrator.save_posterior_artifacts, name, self.session.posterior_latent,
                      self.session.V, self.session.diagnostics, self.session.cfg,
                      on_finished=lambda: (self.post_picker.refresh(),
                                           self.log_pane.append_line(f"Saved posterior '{name}'.")))

    def _validate(self):
        if self.session.posterior is None:
            return
        self.dispatch(orchestrator.validate_calibration, self.session.cfg, self.session.posterior,
                      self.session.inf_prior, self.session.force_prior, provide_fig_sink=True)

    def _infer(self):
        if self.session.posterior is None:
            return
        cfg, post = self.session.cfg, self.session.posterior
        if self.infer_mode.currentIndex() == 0:      # simulated
            cell_path = self.cell_picker.selected_path()
            if not cell_path:
                self.log_pane.append_line("Select a cell file first.", "warning")
                return
            self.dispatch(_run_simulated_inference, cfg, post, cell_path, self.sim_tobs.value(),
                          provide_fig_sink=True)
        else:                                        # experimental
            forcing_si = {name: fld.value() for name, fld in self._forcing_fields.items()}
            self.dispatch(_run_experimental_inference, cfg, post, self.exp_spont.value(),
                          self.exp_forced.value(), self.exp_tobs.value(), forcing_si, provide_fig_sink=True)

    # ── helpers ──────────────────────────────────────────────────────────────
    def _rebuild_forcing_fields(self, cfg):
        for fld in self._forcing_fields.values():
            self.exp_form.removeRow(fld)
        self._forcing_fields = {}
        if self._forcing_anchor is not None:
            self.exp_form.removeRow(self._forcing_anchor)
            self._forcing_anchor = None
        for name in cfg.force_params_dict:
            unit = cli._INFERENCE_PROMPT_UNITS.get(name, "")
            fld = FloatField(0.0)
            self._forcing_fields[name] = fld
            self.exp_form.addRow(f"{name}{f' ({unit})' if unit else ''}", fld)

    @staticmethod
    def _extract_rotation(posterior):
        """Recover the decorrelating rotation V from the posterior's transform (for a deferred save)."""
        try:
            from core.SBI.reparam import OrthogonalTransform
            parts = getattr(getattr(posterior, "T", None), "parts", [])
            if parts and isinstance(parts[0], OrthogonalTransform):
                return parts[0].M
        except Exception:
            pass
        return None

    # ── persistence ──────────────────────────────────────────────────────────
    def save_settings(self, qs):
        qs.beginGroup("sbi")
        qs.setValue("model", self.model_combo.currentText())
        qs.setValue("bounds", self.bounds_picker.key())
        qs.setValue("prior", self.prior_picker.key())
        qs.setValue("posterior", self.post_picker.key())
        qs.setValue("cell", self.cell_picker.key())
        qs.setValue("infer_mode", self.infer_mode.currentIndex())
        settings.save_field(qs, "sim_tobs", self.sim_tobs)
        settings.save_field(qs, "exp_tobs", self.exp_tobs)
        settings.save_field(qs, "exp_spont", self.exp_spont)
        settings.save_field(qs, "exp_forced", self.exp_forced)
        qs.endGroup()
        # The experimental forcing fields do not exist until "Build config" runs (they are built from
        # cfg.force_params_dict), so they are not persisted -- restoring a deferred widget is out of scope.

    def restore_settings(self, qs):
        qs.beginGroup("sbi")
        # Model FIRST + explicit _on_model_changed: it repoints the bounds/cell pickers' base_path and
        # refresh()es them, which would wipe a picker selection restored before it. Feed it the combo's
        # ACCEPTED text (a non-editable combo silently ignores a setCurrentText it can't honour), or a
        # stale/corrupt saved model would point the pickers at a nonexistent folder while the combo
        # still shows the default.
        self.model_combo.setCurrentText(settings.get_str(qs, "model", self.model_combo.currentText()))
        self._on_model_changed(self.model_combo.currentText())
        self.bounds_picker.restore_key(settings.get_str(qs, "bounds"))
        self.cell_picker.restore_key(settings.get_str(qs, "cell"))
        self.prior_picker.restore_key(settings.get_str(qs, "prior"))    # own base_path, order-independent
        self.post_picker.restore_key(settings.get_str(qs, "posterior"))
        try:
            self.infer_mode.setCurrentIndex(int(settings.get_str(qs, "infer_mode", "0")))
        except ValueError:
            pass
        settings.restore_field(qs, "sim_tobs", self.sim_tobs)
        settings.restore_field(qs, "exp_tobs", self.exp_tobs)
        settings.restore_field(qs, "exp_spont", self.exp_spont)
        settings.restore_field(qs, "exp_forced", self.exp_forced)
        qs.endGroup()

    def _refresh_gates(self):
        s = self.session
        self.btn_prior.setEnabled(s.cfg is not None)
        self.btn_save_prior.setEnabled(s.inf_prior is not None)
        self.btn_post.setEnabled(s.inf_prior is not None)
        self.btn_save_post.setEnabled(s.posterior_latent is not None)
        self.btn_validate.setEnabled(s.posterior is not None)
        self.btn_infer.setEnabled(s.posterior is not None)

    def set_controls_enabled(self, enabled: bool):
        super().set_controls_enabled(enabled)   # locks the whole column, pickers included
        if enabled:
            self._refresh_gates()   # re-apply stage gating after a task frees the controls
