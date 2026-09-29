import random

from PySide6.QtWidgets import (QGroupBox, QLabel, QPushButton, QVBoxLayout)

from core import config, orchestrator
from core.refusals import Refusal, require_at_least

from ... import icons
from ...fields import label
from ...widgets.forms import make_form
from ...widgets.help_badge import add_help_row, with_badge
from ...widgets.labeled_inputs import FloatField, IntField, PathField
from .base import _StagePanel
from .help_text import HELP


# ── 4. Validate ───────────────────────────────────────────────────────────────
class ValidatePanel(_StagePanel):
    """Tab 4. Runs the calibration battery (SBC / TARP / PPC) on the current posterior.

    Gated on a posterior AND ``inf_prior`` -- deliberately not on ``force_prior``, which is None for
    every no-forcing model and once made this tab permanently unreachable for exactly those.

    Persists: nothing. Its two boxes are science knobs: the calibration's dataset count and its
    (t_scale, T_obs) operating points open at config.py's SBC_N_CAL and CAL_N_SCALES on every launch,
    and a `cal_n` / `cal_scales` key an older build left in PRISM.ini is ignored. Each run draws its
    seed afresh and the calibration record keeps it, so any calibration can be repeated from the
    command line; there is no Seed box, and nothing about the seed is remembered. With nothing to
    restore, the tab has no save_settings / restore_settings of its own (BasePanel's are no-ops).
    """
    def __init__(self, screen, parent=None):
        super().__init__(screen, parent)
        box = QGroupBox("Validate (SBC + TARP)")
        v = QVBoxLayout(box)
        v.addWidget(QLabel("Data-free calibration. Needs a posterior and the prior it was trained against."))
        cform = make_form()
        self.cal_n = IntField(str(config.SBC_N_CAL))
        self.cal_scales = IntField(str(config.CAL_N_SCALES))
        add_help_row(cform, label("n_cal"), self.cal_n, HELP["cal_n"])
        add_help_row(cform, label("cal_n_scales"), self.cal_scales, HELP["cal_scales"])
        v.addLayout(cform)
        self.btn_validate = QPushButton("Run calibration")
        self.btn_validate.setProperty("accent", True)     # primary CTA (Fluent accent)
        self.btn_validate.clicked.connect(self._validate)
        v.addWidget(self.btn_validate)
        self.controls_layout.addWidget(box)

    def _read_inputs(self) -> dict:
        """The two boxes, through the shared rules, or the first Refusal.

        Both were clamped with max(1, ...), which turned a blank or a 0 into a 1 nobody typed -- and
        for the operating points a 1 is a DIFFERENT measurement, not a smaller one: cal_n_scales is
        t_scale's effective sample size. The stage runs the same rules again at entry; this copy is
        what makes the refusal a click-time dialog naming the box instead of a worker error.
        """
        return {
            "n_cal": require_at_least("n_cal", self.cal_n.value_or_none(), 1),
            "cal_n_scales": require_at_least("cal_n_scales", self.cal_scales.value_or_none(), 1),
        }

    def _validate(self):
        s = self.session
        if s.posterior is None or s.inf_prior is None:   # force_prior is legitimately None (no drive)
            return
        try:
            v = self._read_inputs()
        except Refusal as e:
            self._refusal(e)
            return
        # the region (a TSNPE posterior calibrates on the prior RESTRICTED to it -- the
        # calibrate-on-the-region rule) comes off s.posterior.posterior.truncation inside
        # validate_calibration; None for an amortized one leaves the battery exactly as it was.
        # A seed drawn afresh on every run, from Python's own stream: the record keeps it, so the
        # calibration can be repeated, and nothing here remembers it.
        self.dispatch(orchestrator.validate_calibration, s.cfg, s.posterior, s.inf_prior, provide_fig_sink=True,
                      n_cal=v["n_cal"], cal_n_scales=v["cal_n_scales"], seed=random.randrange(2 ** 31),
                      on_result=self._on_calibration)

    def _on_calibration(self, payload):
        res = payload.results
        info = res.get("informativeness") or {}
        passed = (res.get("verdict") or {}).get("passed")
        self.log_pane.append_line(
            f"Calibration verdict {'PASS' if passed else 'FAIL'} (seed {res.get('seed')}), recorded as "
            f"{payload.name or '(unnamed, id ' + payload.id + ')'}: TARP ATC="
            f"{res['tarp']['atc']}, KS p={res['tarp']['ks_p']}"
            + (f"; informativeness {info['total_nats']:.2f} nats" if info.get("total_nats") is not None else ""))

    def refresh_local_gates(self):
        s = self.session
        self.btn_validate.setEnabled(s.posterior is not None and s.inf_prior is not None)
