"""Parameter Inference, split into six tabs over ONE shared SbiSession owned by the InferenceScreen.

    Config -> Simulate -> Prior -> Posterior -> Validate -> Infer

Each tab is its own BasePanel (its own FigureStack/ProgressPane/LogPane), so dispatch()/cancel/the
figure-sink/progress plumbing are reused verbatim; BasePanel._running is class-level, so the six tabs
can never run concurrently. The screen owns the SbiSession and greys tabs via setTabEnabled; every tab
reads/writes the session through ``self._screen`` and calls ``self._screen.refresh_gates()`` after a
stage completes. This is the old single-column SbiPanel decomposed -- the stage logic (cli.make_sim_config
+ the orchestrator stage fns with save=False/fig_sink) is unchanged.
"""
from PySide6.QtWidgets import (QComboBox, QFormLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
                               QPushButton, QStackedWidget, QVBoxLayout, QWidget)

from core import cli, orchestrator, registry
from core.Helpers import file_manager, labels, visualizers
from core.config import (VALID_MODELS, VALID_LABELS, BOUNDS_PATH, CELL_PATH, PRIOR_PATH,
                         POSTERIOR_PATH, T_MIN_EXP_S)

from .base_panel import BasePanel
from .. import settings
from ..widgets.artifact_picker import ArtifactPicker
from ..widgets.help_badge import add_help_row
from ..widgets.labeled_inputs import FloatField, PathField

# Help text shown by the "?" badge next to each option. Drafted from the code/science; user reviews.
HELP = {
    "model": "Which hair-cell model to fit. NADROWSKI is the state-dependent-drift model the pipeline "
             "is built around; HOPF and BP are alternatives.",
    "bounds": "Parameter-bounds file (Resources/Bounds/<model>) defining the inference box: which "
              "parameters are inferred and the prior range of each.",
    "cell": "A cell file (Resources/Cells/<model>) whose parameter values are the ground truth — the "
            "simulator uses them to generate a synthetic observation.",
    "tobs": "Observation duration in seconds. Longer traces carry more information but cost more to "
            "simulate.",
    "prior": "Load a saved prior (.pt), or choose “(from scratch)” to construct a new "
             "stability-screened parameter prior.",
    "posterior": "Load a trained posterior (.pt), or “(from scratch)” to train a new one. Training "
                 "from scratch needs a prior; loading an existing posterior does not.",
    "infer_mode": "Simulated: infer on a synthetic observation from a cell’s ground truth. "
                  "Experimental: infer on your own recorded spontaneous + forced traces.",
    "spont": "Path to the recorded spontaneous (undriven) hair-bundle trace (.csv or .npy; last column "
             "= values).",
    "forced": "Path to the recorded forced (driven) hair-bundle trace (.csv or .npy; last column = "
              "values).",
    "forcing": "The value of this sinusoidal-drive parameter used in the forced recording, in the "
               "shown units.",
}


# ── worker-callable runners (module-level so a Worker can call them with an injected fig_sink) ─────
def _run_simulated_preview(cfg, cell_path, T_obs_s, *, fig_sink=None):
    """Simulate a ground-truth trace from a cell and show it -- the observation-generation half of the
    simulated-inference path, WITHOUT the inference. A quick check that the config + cell simulate
    sanely before spending time on priors/posteriors."""
    cli.load_and_validate_gt(cfg, cell_path)
    cfg.T_obs = T_obs_s * cfg.get_unit_conversion_factor("s")
    x_dim, _obs_stats, t_dim = orchestrator.generate_observations(cfg)
    visualizers.plot(t_dim.squeeze(0).cpu().detach().numpy(), x_dim[0, :].cpu().detach().numpy(),
                     title="Ground-truth trace",
                     labels=(labels.axis_label("t", "s"), labels.axis_label("x", cfg.length_unit)),
                     sink=fig_sink)


def _run_simulated_inference(cfg, posterior, cell_path, T_obs_s, *, fig_sink=None):
    """Mirror orchestrator.run's simulated branch: inject GT + T_obs, simulate, show GT trace + infer."""
    cli.load_and_validate_gt(cfg, cell_path)
    cfg.T_obs = T_obs_s * cfg.get_unit_conversion_factor("s")
    x_dim, obs_stats, t_dim = orchestrator.generate_observations(cfg)
    visualizers.plot(t_dim.squeeze(0).cpu().detach().numpy(), x_dim[0, :].cpu().detach().numpy(),
                     title="Ground-truth trace",
                     labels=(labels.axis_label("t", "s"), labels.axis_label("x", cfg.length_unit)),
                     sink=fig_sink)
    orchestrator.infer_and_visualize(cfg, posterior, obs_stats, x_dim, t_dim, show_truth=True, fig_sink=fig_sink)


def _run_experimental_inference(cfg, posterior, spont_path, forced_path, T_obs_s, forcing_si, *, fig_sink=None):
    """Mirror orchestrator.run's experimental branch."""
    x_spont = file_manager.load_experimental_data(spont_path, dtype=cfg.hw.dtype)
    x_forced = file_manager.load_experimental_data(forced_path, dtype=cfg.hw.dtype)
    obs_stats, obs_data, t_dim = orchestrator.build_experiment_obs(cfg, x_spont, x_forced, T_obs_s, forcing_si)
    orchestrator.infer_and_visualize(cfg, posterior, obs_stats, obs_data, t_dim, show_truth=False, fig_sink=fig_sink)


class _StagePanel(BasePanel):
    """Common base for the six inference tabs: holds a back-reference to the owning InferenceScreen and
    reads/writes the shared session through it (never caching the session object, which Config replaces
    wholesale on each build)."""

    def __init__(self, screen, parent=None):
        super().__init__(parent)
        self._screen = screen

    @property
    def session(self):
        return self._screen.session


# ── 1. Config ─────────────────────────────────────────────────────────────────
class ConfigPanel(_StagePanel):
    def __init__(self, screen, parent=None):
        super().__init__(screen, parent)
        box = QGroupBox("Config")
        form = QFormLayout(box)
        self.model_combo = QComboBox()
        self.model_combo.addItems(VALID_MODELS)
        self.model_combo.setCurrentText("NADROWSKI")
        self.bounds_picker = ArtifactPicker(BOUNDS_PATH / "nadrowski")
        self.model_combo.currentTextChanged.connect(self._on_model_changed)
        self.btn_config = QPushButton("Build config")
        self.btn_config.setProperty("accent", True)       # primary CTA (Fluent accent)
        self.btn_config.clicked.connect(self._build_config)
        add_help_row(form, "Model", self.model_combo, HELP["model"])
        add_help_row(form, "Bounds", self.bounds_picker, HELP["bounds"])
        form.addRow(self.btn_config)
        self.controls_layout.addWidget(box)
        self.restore_settings(settings.settings())

    def _on_model_changed(self, model: str):
        self.bounds_picker.base_path = BOUNDS_PATH / model.lower()
        self.bounds_picker.refresh()
        # User-defined models are Simulate-only (v1): the SBI stack has no Prior/INIT_SHAPES for them.
        is_user = registry.is_user_model(model)
        self.btn_config.setEnabled(not is_user)
        if is_user:
            self.log_pane.append_line(
                f"'{model}' is a user-defined model. Parameter inference does not support "
                "user-defined models (v1); use the Simulate section.", "warning")

    def _build_config(self):
        model = self.model_combo.currentText()
        if registry.is_user_model(model):                 # backstop; the CTA is already disabled
            self.log_pane.append_line(
                "Parameter inference does not support user-defined models (v1); use Simulate.",
                "warning")
            return
        labels = VALID_LABELS[VALID_MODELS.index(model)]
        state_dep_drift = registry.state_dep_drift(model)
        bounds_path = self.bounds_picker.selected_path()
        if not bounds_path:
            self.log_pane.append_line("Select a bounds file first.", "warning")
            return
        try:
            cfg = cli.make_sim_config(model, labels, state_dep_drift, bounds_path)
        except Exception as e:                       # noqa: BLE001 -- see BasePanel._config_error
            self._config_error(e)
            return
        self._screen.new_session(cfg)                # replaces the shared session + repoints + re-gates
        self.log_pane.append_line(
            f"Config built: {model} — {len(cfg.params_dict)} ND + {len(cfg.rescale_params)} rescale params.")

    def save_settings(self, qs):
        qs.beginGroup("inference_config")
        qs.setValue("model", self.model_combo.currentText())
        qs.setValue("bounds", self.bounds_picker.key())
        qs.endGroup()

    def restore_settings(self, qs):
        qs.beginGroup("inference_config")
        # Model FIRST + explicit _on_model_changed (currentTextChanged won't fire if the value already
        # equals the default), THEN the bounds picker -- a picker restored first gets wiped by refresh().
        self.model_combo.setCurrentText(settings.get_str(qs, "model", self.model_combo.currentText()))
        self._on_model_changed(self.model_combo.currentText())
        self.bounds_picker.restore_key(settings.get_str(qs, "bounds"))
        qs.endGroup()


class _CellPreviewMixin:
    """Shared cell-picker handling for Simulate + Infer: the cell folder follows the BUILT config's
    model (there is no live model combo in these tabs), so the picker is repointed in on_config_built
    and the saved key is re-applied there (it could not resolve at __init__, before any config)."""

    def _init_cell_picker(self):
        self.cell_picker = ArtifactPicker(CELL_PATH / "nadrowski")
        self._saved_cell_key = ""

    def on_config_built(self, cfg):
        self.cell_picker.base_path = CELL_PATH / cfg.model.lower()
        self.cell_picker.refresh()
        self.cell_picker.restore_key(self._saved_cell_key)   # -1 guard leaves default if not in folder


# ── 2. Simulate (preview a ground-truth trace) ────────────────────────────────
class SimulatePanel(_StagePanel, _CellPreviewMixin):
    def __init__(self, screen, parent=None):
        super().__init__(screen, parent)
        self._init_cell_picker()
        box = QGroupBox("Simulate a ground-truth trace")
        form = QFormLayout(box)
        self.sim_tobs = FloatField(T_MIN_EXP_S)
        self.btn_run = QPushButton("Simulate trace")
        self.btn_run.setProperty("accent", True)          # primary CTA (Fluent accent)
        self.btn_run.clicked.connect(self._run)
        form.addRow(QLabel("Preview a synthetic observation from a cell's ground-truth parameters."))
        add_help_row(form, "Cell", self.cell_picker, HELP["cell"])
        add_help_row(form, "T_obs (s)", self.sim_tobs, HELP["tobs"])
        form.addRow(self.btn_run)
        self.controls_layout.addWidget(box)
        self.restore_settings(settings.settings())

    def _run(self):
        cfg = self.session.cfg
        if cfg is None:
            return
        cell = self.cell_picker.selected_path()
        if not cell:
            self.log_pane.append_line("Select a cell file first.", "warning")
            return
        self.dispatch(_run_simulated_preview, cfg, cell, self.sim_tobs.value(), provide_fig_sink=True)

    def refresh_local_gates(self):
        self.btn_run.setEnabled(self.session.cfg is not None)

    def save_settings(self, qs):
        qs.beginGroup("inference_simulate")
        qs.setValue("cell", self.cell_picker.key())
        settings.save_field(qs, "sim_tobs", self.sim_tobs)
        qs.endGroup()

    def restore_settings(self, qs):
        qs.beginGroup("inference_simulate")
        self._saved_cell_key = settings.get_str(qs, "cell")     # re-applied in on_config_built
        settings.restore_field(qs, "sim_tobs", self.sim_tobs)
        qs.endGroup()


# ── 3. Prior ──────────────────────────────────────────────────────────────────
class PriorPanel(_StagePanel):
    def __init__(self, screen, parent=None):
        super().__init__(screen, parent)
        box = QGroupBox("Prior")
        v = QVBoxLayout(box)
        form = QFormLayout()
        self.prior_picker = ArtifactPicker(PRIOR_PATH, keep=lambda fn: fn.endswith(".pt"), allow_new=True)
        add_help_row(form, "Prior", self.prior_picker, HELP["prior"])
        v.addLayout(form)
        self.btn_prior = QPushButton("Build / Load prior")
        self.btn_prior.setProperty("accent", True)        # primary CTA (Fluent accent)
        self.btn_prior.clicked.connect(self._build_prior)
        v.addWidget(self.btn_prior)
        self.prior_name = QLineEdit()
        self.prior_name.setPlaceholderText("name to save prior as…")
        self.btn_save_prior = QPushButton("Save")
        self.btn_save_prior.clicked.connect(self._save_prior)
        row = QHBoxLayout()
        row.addWidget(self.prior_name, 1)
        row.addWidget(self.btn_save_prior)
        v.addLayout(row)
        self.controls_layout.addWidget(box)
        self.restore_settings(settings.settings())

    def _build_prior(self):
        cfg = self.session.cfg
        if cfg is None:
            return
        entry, is_new = self.prior_picker.selected()
        self.session.reset_downstream("prior")
        self._screen.refresh_gates()                 # grey Posterior-from-scratch/Validate/Infer while building
        self.dispatch(orchestrator.build_prior, cfg, entry, is_new, save=False,
                      provide_fig_sink=True, on_result=self._on_prior)

    def _on_prior(self, payload):
        self.session.inf_prior, self.session.force_prior = payload
        self.log_pane.append_line("Prior ready.")
        self._screen.refresh_gates()

    def _save_prior(self):
        name = self.prior_name.text().strip()
        if not name or self.session.inf_prior is None:
            self.log_pane.append_line("Build a prior and enter a name first.", "warning")
            return
        nd_prior = self.session.inf_prior.distributions[0]
        self.dispatch(orchestrator.save_prior_artifacts, name, nd_prior, self.session.cfg,
                      on_finished=lambda: (self.prior_picker.refresh(),
                                           self.log_pane.append_line(f"Saved prior '{name}'.")))

    def refresh_local_gates(self):
        self.btn_prior.setEnabled(self.session.cfg is not None)
        self.btn_save_prior.setEnabled(self.session.inf_prior is not None)

    def save_settings(self, qs):
        qs.beginGroup("inference_prior")
        qs.setValue("prior", self.prior_picker.key())
        qs.endGroup()

    def restore_settings(self, qs):
        qs.beginGroup("inference_prior")
        self.prior_picker.restore_key(settings.get_str(qs, "prior"))
        qs.endGroup()


# ── 4. Posterior ──────────────────────────────────────────────────────────────
class PosteriorPanel(_StagePanel):
    def __init__(self, screen, parent=None):
        super().__init__(screen, parent)
        box = QGroupBox("Posterior")
        v = QVBoxLayout(box)
        form = QFormLayout()
        self.post_picker = ArtifactPicker(
            POSTERIOR_PATH, keep=lambda fn: fn.endswith(".pt") and not fn.endswith(".rot.pt"), allow_new=True)
        self.post_picker.combo.currentIndexChanged.connect(lambda _i: self._sync_train_button())
        add_help_row(form, "Posterior", self.post_picker, HELP["posterior"])
        v.addLayout(form)
        self.btn_post = QPushButton("Train / Load posterior")
        self.btn_post.setProperty("accent", True)         # primary CTA (Fluent accent)
        self.btn_post.clicked.connect(self._build_posterior)
        v.addWidget(self.btn_post)
        self.post_name = QLineEdit()
        self.post_name.setPlaceholderText("name to save posterior as…")
        self.btn_save_post = QPushButton("Save")
        self.btn_save_post.clicked.connect(self._save_posterior)
        row = QHBoxLayout()
        row.addWidget(self.post_name, 1)
        row.addWidget(self.btn_save_post)
        v.addLayout(row)
        self.controls_layout.addWidget(box)
        self.restore_settings(settings.settings())

    def _build_posterior(self):
        cfg = self.session.cfg
        if cfg is None:
            return
        entry, is_new = self.post_picker.selected()
        if is_new and self.session.inf_prior is None:
            self.log_pane.append_line("Build or load a prior first to train a new posterior.", "warning")
            return
        self.session.reset_downstream("posterior")
        self._screen.refresh_gates()
        self.dispatch(orchestrator.build_posterior, cfg, self.session.inf_prior,
                      self.session.force_prior, entry, is_new, save=False,
                      provide_fig_sink=True, on_result=self._on_posterior)

    def _on_posterior(self, payload):
        self.session.posterior, self.session.diagnostics = payload
        self.session.posterior_latent = getattr(self.session.posterior, "latent", None)
        self.session.V = self._extract_rotation(self.session.posterior)
        self.log_pane.append_line("Posterior ready.")
        self._screen.refresh_gates()

    def _save_posterior(self):
        name = self.post_name.text().strip()
        if not name or self.session.posterior_latent is None:
            self.log_pane.append_line("Train a posterior and enter a name first.", "warning")
            return
        self.dispatch(orchestrator.save_posterior_artifacts, name, self.session.posterior_latent,
                      self.session.V, self.session.diagnostics, self.session.cfg,
                      on_finished=lambda: (self.post_picker.refresh(),
                                           self.log_pane.append_line(f"Saved posterior '{name}'.")))

    def _sync_train_button(self):
        """Disable the Train button when the "(from scratch)" option is selected but no prior exists --
        loading an existing posterior is always allowed; training a new one needs a prior."""
        _entry, is_new = self.post_picker.selected()
        blocked = is_new and self.session.inf_prior is None
        self.btn_post.setEnabled(self.session.cfg is not None and not blocked)
        self.btn_post.setToolTip("Build or load a prior first to train a new posterior." if blocked else "")

    def refresh_local_gates(self):
        self._sync_train_button()
        self.btn_save_post.setEnabled(self.session.posterior_latent is not None)

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

    def save_settings(self, qs):
        qs.beginGroup("inference_posterior")
        qs.setValue("posterior", self.post_picker.key())
        qs.endGroup()

    def restore_settings(self, qs):
        qs.beginGroup("inference_posterior")
        self.post_picker.restore_key(settings.get_str(qs, "posterior"))
        qs.endGroup()


# ── 5. Validate ───────────────────────────────────────────────────────────────
class ValidatePanel(_StagePanel):
    def __init__(self, screen, parent=None):
        super().__init__(screen, parent)
        box = QGroupBox("Validate (SBC + TARP)")
        v = QVBoxLayout(box)
        v.addWidget(QLabel("Data-free calibration. Needs a posterior and the prior it was trained against."))
        self.btn_validate = QPushButton("Run calibration")
        self.btn_validate.setProperty("accent", True)     # primary CTA (Fluent accent)
        self.btn_validate.clicked.connect(self._validate)
        v.addWidget(self.btn_validate)
        self.controls_layout.addWidget(box)

    def _validate(self):
        s = self.session
        if s.posterior is None or s.inf_prior is None or s.force_prior is None:
            return
        self.dispatch(orchestrator.validate_calibration, s.cfg, s.posterior,
                      s.inf_prior, s.force_prior, provide_fig_sink=True)

    def refresh_local_gates(self):
        s = self.session
        self.btn_validate.setEnabled(
            s.posterior is not None and s.inf_prior is not None and s.force_prior is not None)


# ── 6. Infer ──────────────────────────────────────────────────────────────────
class InferPanel(_StagePanel, _CellPreviewMixin):
    def __init__(self, screen, parent=None):
        super().__init__(screen, parent)
        self._init_cell_picker()
        self._forcing_fields = {}            # name -> FloatField (experimental drive)
        box = QGroupBox("Infer")
        v = QVBoxLayout(box)

        self.infer_mode = QComboBox()
        self.infer_mode.addItems(["Simulated (cell ground truth)", "Experimental data"])
        self.infer_mode.currentIndexChanged.connect(lambda i: self.infer_stack.setCurrentIndex(i))
        mode_form = QFormLayout()
        add_help_row(mode_form, "Mode", self.infer_mode, HELP["infer_mode"])
        v.addLayout(mode_form)

        self.infer_stack = QStackedWidget()
        # simulated inputs
        sim_w = QWidget(); sim_f = QFormLayout(sim_w)
        self.sim_tobs = FloatField(T_MIN_EXP_S)
        add_help_row(sim_f, "Cell", self.cell_picker, HELP["cell"])
        add_help_row(sim_f, "T_obs (s)", self.sim_tobs, HELP["tobs"])
        self.infer_stack.addWidget(sim_w)
        # experimental inputs
        exp_w = QWidget(); self.exp_form = QFormLayout(exp_w)
        self.exp_spont = PathField()
        self.exp_forced = PathField()
        self.exp_tobs = FloatField(T_MIN_EXP_S)
        add_help_row(self.exp_form, "Spontaneous", self.exp_spont, HELP["spont"])
        add_help_row(self.exp_form, "Forced", self.exp_forced, HELP["forced"])
        add_help_row(self.exp_form, "T_obs (s)", self.exp_tobs, HELP["tobs"])
        self._forcing_anchor = QLabel("(build config to list drive params)")
        self.exp_form.addRow(self._forcing_anchor)
        self.infer_stack.addWidget(exp_w)
        v.addWidget(self.infer_stack)

        self.btn_infer = QPushButton("Run inference")
        self.btn_infer.setProperty("accent", True)        # primary CTA (Fluent accent)
        self.btn_infer.clicked.connect(self._infer)
        v.addWidget(self.btn_infer)
        self.controls_layout.addWidget(box)
        self.restore_settings(settings.settings())

    def on_config_built(self, cfg):
        _CellPreviewMixin.on_config_built(self, cfg)
        self._rebuild_forcing_fields(cfg)

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
            add_help_row(self.exp_form, labels.gui_forcing_label(name, unit), fld, HELP["forcing"])

    def _infer(self):
        cfg, post = self.session.cfg, self.session.posterior
        if post is None:
            return
        if self.infer_mode.currentIndex() == 0:      # simulated
            cell = self.cell_picker.selected_path()
            if not cell:
                self.log_pane.append_line("Select a cell file first.", "warning")
                return
            self.dispatch(_run_simulated_inference, cfg, post, cell, self.sim_tobs.value(),
                          provide_fig_sink=True)
        else:                                        # experimental
            forcing_si = {name: fld.value() for name, fld in self._forcing_fields.items()}
            self.dispatch(_run_experimental_inference, cfg, post, self.exp_spont.value(),
                          self.exp_forced.value(), self.exp_tobs.value(), forcing_si, provide_fig_sink=True)

    def refresh_local_gates(self):
        self.btn_infer.setEnabled(self.session.posterior is not None)

    def save_settings(self, qs):
        qs.beginGroup("inference_infer")
        qs.setValue("cell", self.cell_picker.key())
        qs.setValue("infer_mode", self.infer_mode.currentIndex())
        settings.save_field(qs, "sim_tobs", self.sim_tobs)
        settings.save_field(qs, "exp_tobs", self.exp_tobs)
        settings.save_field(qs, "exp_spont", self.exp_spont)
        settings.save_field(qs, "exp_forced", self.exp_forced)
        qs.endGroup()
        # The forcing fields don't exist until "Build config" runs, so they are not persisted.

    def restore_settings(self, qs):
        qs.beginGroup("inference_infer")
        self._saved_cell_key = settings.get_str(qs, "cell")     # re-applied in on_config_built
        try:
            self.infer_mode.setCurrentIndex(int(settings.get_str(qs, "infer_mode", "0")))
        except ValueError:
            pass
        settings.restore_field(qs, "sim_tobs", self.sim_tobs)
        settings.restore_field(qs, "exp_tobs", self.exp_tobs)
        settings.restore_field(qs, "exp_spont", self.exp_spont)
        settings.restore_field(qs, "exp_forced", self.exp_forced)
        qs.endGroup()
