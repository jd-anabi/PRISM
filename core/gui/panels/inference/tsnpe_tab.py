import traceback

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QCheckBox, QGroupBox, QLabel, QPushButton, QVBoxLayout)

from core import config, orchestrator
from core.artifacts import default_store
from core.refusals import Refusal, require_at_least, require_between

from ... import icons, settings
from ...fields import label
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

    Persists (group "inference_tsnpe"): the observation and the two budget fields. The HPD level and
    the direction count open at ``truncate.DEFAULT_HPD`` and ``DEFAULT_N_DIRECTIONS`` on every launch
    (V5), and the new-simulation box is a per-round consent that is never persisted.
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
        add_help_row(form, label("observation"), self.obs_picker, HELP["tsnpe_obs"])
        self.hpd = FloatField(str(_tr.DEFAULT_HPD))
        self.n_dirs = IntField(str(_tr.DEFAULT_N_DIRECTIONS))
        add_help_row(form, label("hpd_level"), self.hpd, HELP["tsnpe_hpd"])
        add_help_row(form, label("n_directions"), self.n_dirs, HELP["tsnpe_dirs"])
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
        add_help_row(bform, label("num_runs"), self.num_runs, HELP["num_runs"])
        add_help_row(bform, label("run_size_cap"), self.run_size_cap, HELP["run_size"])
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
        # LAST, like every other panel's __init__ (BasePanel.restore_settings): the budget boxes'
        # textChanged is wired above, so a restored budget redraws the three lines, and the picker
        # listed the store at construction, so a saved observation id resolves. This line was missing
        # from the tab's first commit until piece 3 (spec §5.3): every launch showed config.py's
        # budget and the first observation while PRISM.ini held the last session's.
        self.restore_settings(settings.settings())

    def _read_inputs(self) -> dict:
        """Every box a round reads, through the shared rules, or the first Refusal.

        A blank box is a refusal, never a zero: FloatField.value() handed a blank HPD to the stage as
        0.0 and IntField.value() a blank direction box as 0, and the stage refused each with a
        sentence naming neither the box nor the default. The batch count was clamped to 1 and a blank
        rows-per-batch became a 0 nobody typed; rows-per-batch accepts a TYPED 0 (= automatic) and
        nothing blank. The latent-width ceiling on the direction count is the stage's: it needs the
        posterior, and the click does not have its width.
        """
        n_runs, cap = self._budget_values()
        return {
            "n_directions": require_at_least("n_directions", self.n_dirs.value_or_none(), 1),
            "hpd_level": require_between("hpd_level", self.hpd.value_or_none(), 0.0, 1.0,
                                         open_lo=True, open_hi=True),
            "num_runs": require_at_least("num_runs", n_runs, 1),
            "run_size_cap": require_at_least("run_size_cap", cap, 0),
        }

    def _round(self):
        s = self.session
        if s.posterior is None or s.inf_prior is None or not self.obs_picker.key():
            return
        # The boxes first, before the observation is re-hashed: a refused box costs nothing. The stage
        # runs the same rules again at entry and adds the one check only it can make, the direction
        # count against the latent width -- this tab has no posterior width to measure it against.
        try:
            v = self._read_inputs()
        except Refusal as e:
            self._refusal(e)
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
        self.dispatch(orchestrator.tsnpe_round, s.cfg, s.posterior, s.inf_prior, obs,
                      n_directions=v["n_directions"], level=v["hpd_level"],
                      num_runs=v["num_runs"], run_size_cap=v["run_size_cap"],
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
        """The observation and the budget only (V5, spec §5.3), the budget as the boxes' TEXT so a box
        left blank at close opens at config.py's default. The HPD level and the direction count are
        science knobs and open at the truncate module's defaults on every launch; the consent box
        under the budget is answered per round and is never written."""
        qs.beginGroup("inference_tsnpe")
        qs.setValue("observation", self.obs_picker.key())
        settings.save_field(qs, "num_runs", self.num_runs)
        settings.save_field(qs, "run_size_cap", self.run_size_cap)
        qs.endGroup()

    def restore_settings(self, qs):
        qs.beginGroup("inference_tsnpe")
        self.obs_picker.restore_key(settings.get_str(qs, "observation"))
        # Defaults are the config constants: a missing key, a wiped file and a box saved blank all
        # open at the stage defaults (get_int falls back on "" as on a missing key).
        self.num_runs.setText(str(settings.get_int(qs, "num_runs", config.TRAINING_NUM_RUNS)))
        self.run_size_cap.setText(str(settings.get_int(qs, "run_size_cap", config.TRAINING_RUN_SIZE)))
        qs.endGroup()
