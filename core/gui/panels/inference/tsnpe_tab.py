import traceback

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QCheckBox, QGroupBox, QLabel, QPushButton, QVBoxLayout)

from core import config, orchestrator
from core.artifacts import default_store

from ... import icons, settings
from ...widgets.artifact_picker import StorePicker
from ...widgets.forms import make_form
from ...widgets.help_badge import add_help_row, with_badge
from ...widgets.labeled_inputs import FloatField, IntField, PathField
from .base import _StagePanel, _TrainingBudgetMixin
from .help_text import HELP


# ── 6. TSNPE (truncated sequential NPE) ───────────────────────────────────────
class TSNPEPanel(_TrainingBudgetMixin, _StagePanel):
    """Tab 6. Refines the current posterior on ONE observation, without giving up the amortized one.

    ⚠⚠ WHAT THIS TAB MUST NOT BECOME. TSNPE proposes from the PRIOR RESTRICTED to an HPD region. It
    does NOT propose from the posterior. Fitting a density to the posterior and proposing from that
    gives ``p_L ∝ L^(L+1) q`` -- tempering -- and credible intervals then contract as
    ``(L+1)^(-1/2)`` with NO new information entering. SBC comes out flat anyway, because it validates
    the flow against the proposal it was trained on, so nothing on the Validate tab would catch it.
    The rule lives in ``core/SBI/truncate.py`` and is pinned by ``tests/test_conditioning_repair.py``.

    Gated on a posterior, the prior it was trained against, AND a persisted observation.
    An amortized posterior has no observation at SAVE time (``default_x`` is None on
    posterior_08232026), so the Infer tab records one at INFERENCE time and this tab keys on that.

    The budget group is the Posterior tab's, through ``_TrainingBudgetMixin``: a round is a simulation
    campaign, not a click, and the number belongs on screen before the button.

    Persists (group "inference_tsnpe"): the observation, the HPD level, the direction count and the
    two budget fields.
    """

    def __init__(self, screen, parent=None):
        super().__init__(screen, parent)
        from core.SBI import truncate as _tr

        box = QGroupBox("TSNPE round")
        v = QVBoxLayout(box)
        warn = QLabel("Restricts the PRIOR to the posterior's credible region and retrains there. It "
                      "never proposes from the posterior itself. The result is NON-AMORTIZED and is "
                      "marked as such in its manifest: loading it later asks first, and Infer refuses "
                      "any other observation unless 'Run on a different observation' is ticked.")
        warn.setWordWrap(True)
        warn.setTextFormat(Qt.PlainText)
        v.addWidget(warn)

        form = make_form()
        self.obs_picker = StorePicker("observation")
        add_help_row(form, "Observation", self.obs_picker, HELP["tsnpe_obs"])
        self.hpd = FloatField(str(_tr.DEFAULT_HPD))
        self.n_dirs = IntField(str(_tr.DEFAULT_N_DIRECTIONS))
        add_help_row(form, "HPD level", self.hpd, HELP["tsnpe_hpd"])
        add_help_row(form, "Directions truncated", self.n_dirs, HELP["tsnpe_dirs"])
        v.addLayout(form)

        self.btn_round = QPushButton("Run TSNPE round")
        self.btn_round.setProperty("accent", True)         # primary CTA (Fluent accent)
        self.btn_round.clicked.connect(self._round)
        v.addWidget(self.btn_round)
        self.controls_layout.addWidget(box)

        # The SAME budget group the Posterior tab shows, and the same code behind it.
        budget = QGroupBox("Simulation budget for this round")
        bv = QVBoxLayout(budget)
        bform = make_form()
        self.num_runs = IntField(str(config.TRAINING_NUM_RUNS))
        self.run_size_cap = IntField(str(config.TRAINING_RUN_SIZE))
        add_help_row(bform, "Batches", self.num_runs, HELP["num_runs"])
        add_help_row(bform, "Max rows per batch (0 = auto)", self.run_size_cap, HELP["run_size"])
        bv.addLayout(bform)
        self.budget_total = self._derived_label()
        self.budget_mem = self._derived_label()
        self.budget_ckpt = self._derived_label()
        for lab in (self.budget_total, self.budget_mem, self.budget_ckpt):
            bv.addWidget(lab)
        for fld in (self.num_runs, self.run_size_cap):
            fld.textChanged.connect(lambda _t: self._sync_budget())
        # D7's consent, per run and never persisted. It is needed on THIS tab because a near miss fires
        # for rounds too -- a 2-batch test round and a 5000-batch round on the same parent differ only in
        # the batch count -- and the region is drawn inside the worker, so an up-front dialog like the
        # Posterior tab's is impossible: the refusal arrives from the stage, seconds in, and this is how
        # the user answers it.
        self.new_run = QCheckBox("Start a new simulation even if a cache one setting away exists")
        bv.addWidget(with_badge(self.new_run, HELP["tsnpe_new_run"]))
        self.controls_layout.addWidget(budget)
        self._sync_budget()

    def _round(self):
        s = self.session
        if s.posterior is None or s.inf_prior is None or not self.obs_picker.key():
            return
        # Loaded HERE, on the GUI thread: the load re-hashes the file and checks its mode and
        # conditioning width against this config, so a mismatch is a dialog now rather than an
        # exception hours into the round. The stage takes the wrapper; nothing loads by reference.
        try:
            obs = default_store().load_observation(s.cfg, self.obs_picker.key())
        except Exception as e:                                  # noqa: BLE001 -- _on_error sorts refusal from bug
            # The exception itself, unwrapped. A store refusal is a Refusal that already names the
            # observation (store.py's load_observation labels every sentence), so it reaches the
            # yellow box through _on_error's isinstance; a bug reaches the red one with its
            # traceback. The old "Could not load observation '<key>':" prefix re-typed both into
            # one string, and the yellow box could never be opened for it.
            self._on_error(e, traceback.format_exc())
            return
        # The HPD / direction-count checks are the STAGE's now (one copy, shared with the command-line
        # tool), and it refuses before it simulates -- so the error dialog still arrives within seconds.
        n_runs, cap = self._budget_values()
        self.dispatch(orchestrator.tsnpe_round, s.cfg, s.posterior, s.inf_prior, obs,
                      n_directions=self.n_dirs.value(), level=self.hpd.value(),
                      num_runs=max(1, n_runs), run_size_cap=max(0, cap),
                      new_run=self.new_run.isChecked(),
                      provide_fig_sink=True, on_result=self._on_round)
        self.new_run.setChecked(False)

    def _on_round(self, payload):
        """Install the round's LoadedPosterior -- it carries its own region and observation digest
        (None for an amortized one), so Validate restricts its prior and Infer can warn about any
        other observation. Without an on_result the round trains for hours and the result is
        discarded; without the region riding on the returned wrapper, a later Save (a rename) would
        write it marked amortized -- but the region is already on disk from build_posterior's own
        write, not something this handler has to remember to carry.
        """
        s = self.session
        s.posterior = payload
        self.log_pane.append_line(
            f"TSNPE round complete. This posterior is NON-AMORTIZED: it is valid near the "
            f"observation {payload.posterior.x_obs_digest}, and its manifest records so.", "warning")
        self._screen.refresh_gates()

    def _budget_checkpoint(self, cfg, width: int, n_runs: int) -> str:
        """The mixin's line is computed from the AMORTIZED identity, and on this tab it used to say
        "Resumes a COMPLETE checkpoint ... simulation will be skipped entirely" whenever the budget
        matched the parent's -- and that is what a round at that budget did before the region became
        part of the identity (D3). The region is drawn when the round starts, so nothing here can be
        resolved in advance; the honest line is the rule."""
        if not config.TRAINING_CHECKPOINT_EVERY:
            return super()._budget_checkpoint(cfg, width, n_runs)
        return ("A TSNPE round is checkpointed under its OWN identity: the truncation region is part "
                "of it and is drawn when the round starts, so it never resumes the amortized checkpoint "
                "at these settings. Re-running with the SAME posterior, observation, HPD level, "
                "direction count and budget redraws the same region and resumes that round's own "
                "checkpoint; a run exactly one setting away from a committed cache is refused unless "
                "'Start a new simulation even if a cache one setting away exists' is ticked; anything "
                "else simulates the full budget from zero.")

    def refresh_local_gates(self):
        s = self.session
        self.obs_picker.refresh()
        self.btn_round.setEnabled(s.posterior is not None and s.inf_prior is not None
                                  and bool(self.obs_picker.key()))
        self._sync_budget()

    def save_settings(self, qs):
        qs.beginGroup("inference_tsnpe")
        qs.setValue("observation", self.obs_picker.key())
        qs.setValue("hpd", str(self.hpd.value()))
        qs.setValue("n_dirs", self.n_dirs.value())
        qs.setValue("num_runs", self.num_runs.value())
        qs.setValue("run_size_cap", self.run_size_cap.value())
        qs.endGroup()

    def restore_settings(self, qs):
        from core.SBI import truncate as _tr
        qs.beginGroup("inference_tsnpe")
        self.obs_picker.restore_key(settings.get_str(qs, "observation"))
        # get_str + float, because settings has no get_float and inventing one for a single caller
        # would be a wider change than this needs. A blank or unparseable value falls back to the
        # module default rather than to 0.0, which FloatField.value() would otherwise hand back --
        # and an HPD of 0 would truncate the prior to a point.
        try:
            self.hpd.setText(str(float(settings.get_str(qs, "hpd", str(_tr.DEFAULT_HPD)))))
        except ValueError:
            self.hpd.setText(str(_tr.DEFAULT_HPD))
        self.n_dirs.setText(str(settings.get_int(qs, "n_dirs", _tr.DEFAULT_N_DIRECTIONS)))
        self.num_runs.setText(str(settings.get_int(qs, "num_runs", config.TRAINING_NUM_RUNS)))
        self.run_size_cap.setText(str(settings.get_int(qs, "run_size_cap", config.TRAINING_RUN_SIZE)))
        qs.endGroup()
