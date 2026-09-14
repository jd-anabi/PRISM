from PySide6.QtWidgets import (QGroupBox, QHBoxLayout, QLineEdit, QMessageBox, QPushButton, QVBoxLayout)

from core import config, orchestrator
from core.artifacts import Accept, default_store

from ... import icons, settings
from ...widgets.artifact_picker import StorePicker
from ...widgets.forms import make_form
from ...widgets.help_badge import add_help_row, with_badge
from ...widgets.labeled_inputs import FloatField, IntField, PathField
from .base import _StagePanel, _TrainingBudgetMixin
from .help_text import HELP


# ── 3. Posterior ──────────────────────────────────────────────────────────────
class PosteriorPanel(_TrainingBudgetMixin, _StagePanel):
    """Tab 3. Trains a new neural posterior, or loads a saved one.

    A loaded posterior is checked against the config's observation mode before anything else runs --
    the three conditioning widths cannot collide, so a cross-mode load is caught immediately rather
    than as a matrix-shape error deep inside the embedding net.

    Also owns the TRAINING BUDGET (batches x rows-per-batch), which was previously reachable only by
    editing config.py. Both fields are passed to build_posterior as arguments rather than written to
    config: orchestrator snapshots those constants at import, so assigning to them would be a silent
    no-op. See _sync_budget for the three derived lines and why the checkpoint one is not a tooltip.

    Persists (group "inference_posterior"): the posterior picker selection, the batch count and the
    rows-per-batch cap.
    """
    def __init__(self, screen, parent=None):
        super().__init__(screen, parent)
        box = QGroupBox("Posterior")
        v = QVBoxLayout(box)
        form = make_form()
        self.post_picker = StorePicker("posterior", allow_new=True)
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

        # -- training budget: what a new posterior will actually simulate, and what it costs -------
        budget = QGroupBox("Training budget")
        bv = QVBoxLayout(budget)
        bform = make_form()
        self.num_runs = IntField(config.TRAINING_NUM_RUNS)
        self.run_size_cap = IntField(config.TRAINING_RUN_SIZE)
        add_help_row(bform, "Batches", self.num_runs, HELP["num_runs"])
        add_help_row(bform, "Max rows per batch (0 = auto)", self.run_size_cap, HELP["run_size"])
        bv.addLayout(bform)
        self.budget_total = self._derived_label()
        self.budget_mem = self._derived_label()
        self.budget_ckpt = self._derived_label()
        for lab in (self.budget_total, self.budget_mem, self.budget_ckpt):
            bv.addWidget(lab)
        # After the labels exist: textChanged fires during restore_settings below.
        for fld in (self.num_runs, self.run_size_cap):
            fld.textChanged.connect(lambda _t: self._sync_budget())
        self.controls_layout.addWidget(budget)

        # -- flow capacity: re-tryable against a COMPLETE checkpoint without re-simulating --------
        flow = QGroupBox("Density estimator")
        fv = QVBoxLayout(flow)
        fform = make_form()
        self.flow_hidden = IntField(str(config.NSF_HIDDEN_FEATURES))
        self.flow_transforms = IntField(str(config.NSF_NUM_TRANSFORMS))
        self.flow_lr = FloatField(str(config.TRAINING_LEARNING_RATE))
        self.flow_patience = IntField(str(config.TRAINING_STOP_AFTER_EPOCHS))
        add_help_row(fform, "Hidden features", self.flow_hidden, HELP["flow_hidden"])
        add_help_row(fform, "Transforms", self.flow_transforms, HELP["flow_transforms"])
        add_help_row(fform, "Learning rate", self.flow_lr, HELP["flow_lr"])
        add_help_row(fform, "Early-stop patience", self.flow_patience, HELP["flow_patience"])
        fv.addLayout(fform)
        self.controls_layout.addWidget(flow)

        fisher = QGroupBox("Fisher rotation")
        rv = QVBoxLayout(fisher)
        rform = make_form()
        self.fisher_m = IntField(str(config.REPARAM_FISHER_M))
        self.fisher_dz = FloatField(str(config.REPARAM_FISHER_DZ))
        self.fisher_points = IntField(str(config.REPARAM_FISHER_POINTS))
        add_help_row(rform, "Ensemble per perturbation", self.fisher_m, HELP["fisher_m"])
        add_help_row(rform, "Central-difference step", self.fisher_dz, HELP["fisher_dz"])
        add_help_row(rform, "Operating points", self.fisher_points, HELP["fisher_points"])
        rv.addLayout(rform)
        self.controls_layout.addWidget(fisher)

        self.restore_settings(settings.settings())
        self._sync_budget()

    def _build_posterior(self):
        cfg = self.session.cfg
        if cfg is None:
            return
        entry, is_new = self.post_picker.selected()
        if is_new and self.session.inf_prior is None:
            self.log_pane.append_line("Build or load a prior first to train a new posterior.", "warning")
            return
        n_runs, cap = self._budget_values()
        if n_runs < 1:
            self.log_pane.append_line("Batches must be at least 1.", "warning")
            return
        if cap < 0:
            self.log_pane.append_line("Max rows per batch cannot be negative (0 = auto).", "warning")
            return
        new_run, accept = False, Accept()
        if is_new:
            go, new_run = self._confirm_fresh_run(cfg, n_runs, cap)
            if not go:
                return
        else:
            accept = self._accept_for_load(entry)
            if accept is None:                 # the user cancelled the load
                return
        # AFTER both questions, never before: Cancel must leave the session exactly as it was, and
        # reset_downstream drops the posterior the user is still working with. The load branch above
        # asks through _accept_for_load before loading a non-amortized posterior.
        self.session.reset_downstream("posterior")
        self._screen.refresh_gates()
        # Passed, never written to config: orchestrator does `from .config import TRAINING_NUM_RUNS`,
        # so setting the constant here would be a silent no-op and the run would use the default.
        self.dispatch(orchestrator.build_posterior, cfg, self.session.inf_prior,
                      entry, is_new,
                      num_runs=n_runs, run_size_cap=cap, accept=accept, new_run=new_run,
                      hidden_features=max(1, self.flow_hidden.value()),
                      num_transforms=max(1, self.flow_transforms.value()),
                      learning_rate=self.flow_lr.value() or config.TRAINING_LEARNING_RATE,
                      stop_after_epochs=max(1, self.flow_patience.value()),
                      fisher_m=max(1, self.fisher_m.value()),
                      fisher_dz=self.fisher_dz.value() or config.REPARAM_FISHER_DZ,
                      fisher_points=max(1, self.fisher_points.value()),
                      provide_fig_sink=True, on_result=self._on_posterior)

    def _confirm_fresh_run(self, cfg, n_runs: int, cap: int) -> tuple:
        """Ask before starting from zero when a committed cache is ONE FIELD away.

        Returns ``(go, new_run)``: whether to dispatch at all, and the consent to hand
        build_posterior, which refuses the same near miss again inside the worker. The stage is the
        one that must be sure; this dialog is the early, cheap version of that refusal, and the two
        share ONE detector so they cannot disagree about which directory the run will touch.

        THE STATUS LINE WAS NOT ENOUGH, and this is the evidence. `_budget_checkpoint` already says
        "these settings match no checkpoint, so this starts a NEW run" and names the differing field
        -- but it is a passive label, on a tab the user has usually scrolled past by the time they
        press Train, and it has now failed to prevent three restarts: 884 batches lost outright on
        2026-08-27 (a prior rebuilt rather than loaded, and never saved, so unrecoverable), and a
        3989-batch checkpoint nearly abandoned twice on 2026-08-28.

        DELIBERATELY NARROW. The detector reports only a committed sibling that differs in EXACTLY
        ONE field -- the signature of an accident rather than of a different experiment.

        FAILS OPEN. No prior, a detector that raises, and no near miss all return ``(True, False)``
        and the run proceeds, exactly as before.
        """
        if self.session.inf_prior is None:
            return True, False
        try:
            near = orchestrator.fresh_run_near_misses(cfg, self.session.inf_prior,
                                                      num_runs=n_runs, run_size_cap=cap)
        except Exception as e:               # noqa: BLE001 -- never block a run over a warning
            self.log_pane.append_line(
                f"Could not check for resumable checkpoints ({type(e).__name__}: {e}); "
                f"continuing -- the training stage checks again before it simulates.", "warning")
            return True, False
        if not near:
            return True, False
        return (True, True) if self._ask_new_run(near, n_runs) else (False, False)

    def _ask_new_run(self, near, n_runs: int) -> bool:
        """The modal itself, split out so a test can answer it without a click. True = start anyway."""
        lines = [f"  • {r['name']}: {r['batches']:,} batches — differs only in {r['field']}\n"
                 f"      this run: {str(r['mine'])[:60]}\n"
                 f"      that one: {str(r['theirs'])[:60]}" for r in near[:3]]
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Warning)
        box.setWindowTitle("This starts a NEW run")
        box.setText(f"This will simulate {n_runs:,} batches from zero.")
        box.setInformativeText(
            "A checkpoint that is ONE setting away already exists:\n\n" + "\n".join(lines) +
            "\n\nIf you meant to continue that run, cancel and change the setting named above "
            "— for a prior, remember that choosing it in the picker does nothing until you press "
            "\"Build / Load prior\".")
        go = box.addButton("Start a new run anyway", QMessageBox.DestructiveRole)
        box.addButton("Cancel", QMessageBox.RejectRole)
        box.setDefaultButton(box.buttons()[-1])
        box.exec()
        return box.clickedButton() is go

    def _accept_for_load(self, entry) -> "Accept | None":
        """What to load a STORED posterior with: ``Accept()``, ``Accept(truncated=True)``, or None for
        "the user cancelled".

        The tab used to pass ``Accept(truncated=True)`` unconditionally, which made the store's refusal
        unreachable from the GUI: a non-amortized artifact -- valid only near ONE observation -- loaded
        as silently as an amortized one, and the only sign was a log line after the fact. Asking costs a
        click on the rare occasion a TSNPE posterior is picked, and the question is the only place the
        three consequences (Validate restricts its prior, Infer refuses another observation, a further
        round can only be drawn around this posterior's own observation) are stated before the load.

        FAILS OPEN on any error: the load itself then names the problem, which it does far better than a
        dialog here could.
        """
        try:
            m = default_store().get("posterior", entry)       # the picker's userData is the id
        except Exception:                                     # noqa: BLE001 -- let the load report it
            return Accept()
        if m.body.get("amortized", True):
            return Accept()
        return Accept(truncated=True) if self._ask_load_non_amortized(m) else None

    def _ask_load_non_amortized(self, m) -> bool:
        """The dialog. A method of its own so tests can replace it without a live event loop."""
        trd = m.body.get("truncation") or {}
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Warning)
        box.setWindowTitle("Load a NON-AMORTIZED posterior?")
        box.setText(f"'{m.name or m.id}' was trained by a TSNPE round.")
        box.setInformativeText(
            f"It was trained on a prior truncated to a {trd.get('level', '?')}-HPD region along Fisher "
            f"direction(s) {trd.get('dims')}, drawn around the observation with digest "
            f"{trd.get('x_obs_digest')}.\n\n"
            "It is valid only near that observation; anywhere else the flow has never seen a training "
            "row and extrapolates confidently rather than returning the prior.\n\n"
            "If you load it:\n"
            "  • Validate restricts its prior to that region.\n"
            "  • Infer refuses any other observation unless \"Run on a different observation\" is "
            "ticked on the Infer tab.\n"
            "  • A further TSNPE round can only be drawn around this posterior's own observation — "
            "there is no override for that one.")
        load = box.addButton("Load it", QMessageBox.DestructiveRole)
        box.addButton("Cancel", QMessageBox.RejectRole)
        box.setDefaultButton(box.buttons()[-1])
        box.exec()
        return box.clickedButton() is load

    def _on_posterior(self, payload):
        self.session.posterior = payload                 # a LoadedPosterior
        region = payload.posterior.truncation
        if region is not None:
            self.log_pane.append_line(
                f"This posterior is NON-AMORTIZED (observation digest {payload.posterior.x_obs_digest}): "
                f"Validate restricts its prior to the region it was trained on; Infer refuses any other "
                f"observation unless 'Run on a different observation' is ticked on the Infer tab.",
                "warning")
        self.log_pane.append_line(f"Posterior ready: {payload.name or '(unnamed, id ' + payload.id + ')'}. "
                                  f"Name it below to keep it.")
        self._screen.refresh_gates()

    def _save_posterior(self):
        name = self.post_name.text().strip()
        lp = self.session.posterior
        if not name or lp is None:
            self.log_pane.append_line("Train a posterior and enter a name first.", "warning")
            return
        try:
            lp.manifest = default_store().rename("posterior", lp.id, name)
        except Exception as e:                       # noqa: BLE001 -- a bad or duplicate name is user input
            self._config_error(e)
            return
        lp.name = name
        self.post_picker.refresh()
        self.post_picker.restore_key(lp.id)
        self.log_pane.append_line(f"Posterior named '{name}'.")

    def _sync_train_button(self):
        """Disable the Train button when the "(from scratch)" option is selected but no prior exists --
        loading an existing posterior is always allowed; training a new one needs a prior."""
        _entry, is_new = self.post_picker.selected()
        blocked = is_new and self.session.inf_prior is None
        self.btn_post.setEnabled(self.session.cfg is not None and not blocked)
        self.btn_post.setToolTip("Build or load a prior first to train a new posterior." if blocked else "")

    def refresh_local_gates(self):
        self._sync_train_button()
        self.btn_save_post.setEnabled(self.session.posterior is not None)
        # A config or a prior arriving changes every derived line, and the checkpoint line cannot be
        # computed without both.
        self._sync_budget()

    def save_settings(self, qs):
        qs.beginGroup("inference_posterior")
        qs.setValue("posterior", self.post_picker.key())
        qs.setValue("num_runs", self.num_runs.value())
        qs.setValue("run_size_cap", self.run_size_cap.value())
        for name in ("flow_hidden", "flow_transforms", "flow_lr", "flow_patience",
                     "fisher_m", "fisher_dz", "fisher_points"):
            qs.setValue(name, str(getattr(self, name).value()))
        qs.endGroup()

    def restore_settings(self, qs):
        qs.beginGroup("inference_posterior")
        self.post_picker.restore_key(settings.get_str(qs, "posterior"))
        for name, default, cast in (("flow_hidden", config.NSF_HIDDEN_FEATURES, int),
                                    ("flow_transforms", config.NSF_NUM_TRANSFORMS, int),
                                    ("flow_lr", config.TRAINING_LEARNING_RATE, float),
                                    ("flow_patience", config.TRAINING_STOP_AFTER_EPOCHS, int),
                                    ("fisher_m", config.REPARAM_FISHER_M, int),
                                    ("fisher_dz", config.REPARAM_FISHER_DZ, float),
                                    ("fisher_points", config.REPARAM_FISHER_POINTS, int)):
            try:
                getattr(self, name).setText(str(cast(settings.get_str(qs, name, str(default)))))
            except (TypeError, ValueError):
                getattr(self, name).setText(str(default))
        # Defaults are the config constants, so a fresh install and a wiped QSettings both land on
        # exactly the stage defaults.
        self.num_runs.setText(str(settings.get_int(qs, "num_runs", config.TRAINING_NUM_RUNS)))
        self.run_size_cap.setText(str(settings.get_int(qs, "run_size_cap", config.TRAINING_RUN_SIZE)))
        qs.endGroup()
