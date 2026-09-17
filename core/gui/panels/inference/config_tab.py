import os

from PySide6.QtWidgets import (QCheckBox, QComboBox, QGroupBox, QLabel, QLineEdit, QPushButton, QVBoxLayout)

from core import cli, config, registry
from core.Helpers import file_manager
from core.refusals import Refusal, require_at_least, require_finite
from core.SBI import pipeline
from core.config import CHI_K_MAX, VALID_LABELS, VALID_MODELS

from ... import icons, settings
from ...fields import label
from ...session import ConfigDraft
from ...widgets.forms import make_form
from ...widgets.help_badge import add_help_row, with_badge
from ...widgets.labeled_inputs import FloatField, IntField, PathField
from ...widgets.source_toggle import SourceToggle
from .rows import _ChiRangeRow
from .base import _StagePanel, _TrainingBudgetMixin, _nvidia_smi_free_gib
from .help_text import HELP


# ── 1. Config ─────────────────────────────────────────────────────────────────
class ConfigPanel(_StagePanel):
    """Tab 1. Records the MODEL-level choices as a ``ConfigDraft`` -- it does NOT build the SimConfig.

    A SimConfig cannot exist without a bounds file, because the bounds file declares which parameters
    are inferred and hence the observation mode; the Prior tab owns that. Reads and validates the
    boxes at Apply through ``_read_inputs`` (2 <= K <= pad <= CHI_K_MAX, ceiling > CHI_MIN_CYCLES,
    typed units resolvable), since this is where they are entered; a bad box is a ``Refusal`` shown
    as the yellow "Check your inputs" box, and the session is left untouched. The chi drive
    amplitude and band are READ-ONLY displays of config.py (piece 3, V5): the draft carries None for
    both, so every config built from it carries config.py's values, as the command-line tool's does.

    Persists (group "inference_config"): model, units source/text, the chi-mode tick and the
    rotation tick -- the selections. The chi probe count, slots and lock-in ceiling open at
    config.py's values on every launch, and the amplitude and band are never written or read.
    """
    def __init__(self, screen, parent=None):
        super().__init__(screen, parent)
        box = QGroupBox("Config")
        form = make_form(box)
        self.model_combo = QComboBox()
        self.model_combo.addItems(VALID_MODELS)
        self.model_combo.setCurrentText("NADROWSKI")
        self.model_combo.currentTextChanged.connect(self._on_model_changed)
        self.btn_config = QPushButton("Apply model & options")
        self.btn_config.setProperty("accent", True)       # primary CTA (Fluent accent)
        self.btn_config.clicked.connect(self._build_config)
        add_help_row(form, label("model"), self.model_combo, HELP["model"])

        # Units DECLARE what the numbers in the bounds/cell files mean; they never convert them.
        self.units_default = QLabel("—")
        self.units_default.setProperty("type", "caption")
        self.units_text = QLineEdit()
        self.units_text.setPlaceholderText("e.g. nm ms pN kHz")
        self.units_toggle = SourceToggle(self.units_default, self.units_text,
                                         file_label="Model's units file", direct_label="Type units")
        add_help_row(form, label("units"), self.units_toggle, HELP["units"])

        # chi(omega) mode. Captured onto the config at build time (SimConfig carries K/F0/range), so a
        # posterior is self-describing and toggling this later cannot reinterpret an existing run.
        self.chi_check = QCheckBox("Multi-frequency χ(ω) conditioning")
        self.chi_k = IntField(config.CHI_N_FREQS)
        # The drive amplitude and band are MEASUREMENTS (config.py's CHI_F0 record), not per-run
        # choices: since D11 build_prior refuses any other value, so a box that accepted one only
        # manufactured that refusal seconds after Apply. Read-only displays; the draft never reads
        # them (see _build_config), so every config built here carries config.py's values.
        self.chi_f0 = FloatField(config.CHI_F0)
        self.chi_f0.setReadOnly(True)
        self.chi_range = _ChiRangeRow(*config.CHI_FREQ_BOUNDS)
        self.chi_range.lo.setReadOnly(True)
        self.chi_range.hi.setReadOnly(True)
        self.chi_pad = IntField(config.CHI_K_PAD)
        self.chi_cycles = FloatField(config.CHI_MAX_CYCLES)
        self.chi_check.toggled.connect(lambda _on: self._sync_chi_enabled())
        form.addRow(with_badge(self.chi_check, HELP["chi_mode"]))
        add_help_row(form, label("chi_n_freqs"), self.chi_k, HELP["chi_k"])
        add_help_row(form, label("chi_k_pad"), self.chi_pad, HELP["chi_k_pad"])
        add_help_row(form, "χ drive F₀ (ND)", self.chi_f0, HELP["chi_f0"])
        add_help_row(form, "χ frequency range", self.chi_range, HELP["chi_range"])
        self.chi_fixed_note = QLabel("fixed by measurement; change it in config.py")
        self.chi_fixed_note.setProperty("type", "caption")
        form.addRow("", self.chi_fixed_note)
        add_help_row(form, label("chi_max_cycles"), self.chi_cycles, HELP["chi_max_cycles"])

        self.rot_check = QCheckBox("Decorrelating Fisher rotation")
        form.addRow(with_badge(self.rot_check, HELP["reparam_rotate"]))

        form.addRow(self.btn_config)
        self.controls_layout.addWidget(box)

        # -- hardware: not a science knob. It changes how a batch is PLANNED, never what is trained,
        # which is exactly why it is kept out of the checkpoint identity (a memory knob in the
        # digest would rename the checkpoint directory and silently restart a resumable multi-day
        # run). It lives on Config rather than Prior/Posterior because it applies to every stage
        # that simulates.
        hw = QGroupBox("Hardware")
        hv = QVBoxLayout(hw)
        hform = make_form()
        self.vram_ceiling = FloatField(str(pipeline.vram_ceiling_gib()))
        add_help_row(hform, "VRAM ceiling per batch (GiB, 0 = off)", self.vram_ceiling,
                     HELP["vram_ceiling"])
        hv.addLayout(hform)
        self.vram_note = _TrainingBudgetMixin._derived_label()
        hv.addWidget(self.vram_note)
        self.vram_ceiling.textChanged.connect(lambda _t: self._apply_vram_ceiling())
        self.controls_layout.addWidget(hw)
        self._apply_vram_ceiling()

        self.restore_settings(settings.settings())
        self._sync_chi_enabled()

    def _apply_vram_ceiling(self):
        """Push the field into ``config.SIM_VRAM_CEILING_GIB`` and say what will ACTUALLY take effect.

        ASSIGNING THE CONSTANT IS ENOUGH HERE, and that is not true of the sweep and flow knobs
        beside it. Those had to become ARGUMENTS because `orchestrator` does `from .config import ...`
        and binds them at import, so assigning to the constant is a silent no-op. This one
        is read LIVE -- `pipeline.vram_ceiling_gib()` does a `getattr` on the module every time the
        planner asks -- so a plain assignment reaches every stage that simulates, with no plumbing.

        NOT PERSISTED, ON PURPOSE, and it is the only field on this tab that is not. Stale QSettings
        have already cost this project a ~5-day run (the 2026-08-19 retrain trained on the retired
        band because a saved value silently won over config.py). A ceiling fails the same way but
        more quietly: a forgotten 2 GiB would not error, it would just make every future run split
        from batch 0 and take several times longer, with nothing in the log to explain it. Starting
        each session from config.py's 0.0 means the throttle is always a decision someone just made.

        The env override still wins if it is set -- said out loud here rather than left to puzzle
        over, because a field that silently does nothing is worse than no field.
        """
        config.SIM_VRAM_CEILING_GIB = max(0.0, self.vram_ceiling.value())
        effective = pipeline.vram_ceiling_gib()
        env = os.environ.get(pipeline.VRAM_CEILING_ENV)
        free = _nvidia_smi_free_gib()
        free_txt = (f"nvidia-smi reports {free:.2f} GiB free"
                    if free is not None else "free VRAM unreadable (no nvidia-smi)")
        if env is not None and env.strip():
            self.vram_note.setText(
                f"⚠ {pipeline.VRAM_CEILING_ENV}={env} is set and OVERRIDES this field — "
                f"planning to {effective:.2f} GiB. {free_txt}.")
        elif effective <= 0:
            self.vram_note.setText(
                f"Off: batches are planned from the free-memory reading and the learned cap alone. "
                f"{free_txt} — set a ceiling near that, minus ~1 GiB, only if you need the desktop.")
        else:
            self.vram_note.setText(
                f"Batches will be planned to fit {effective:.2f} GiB. {free_txt}. Above the real "
                f"free figure this does nothing; below it, expect more splitting and more wall-clock.")

    def _sync_chi_enabled(self):
        """The three χ knobs are meaningless unless χ-mode is on. The rotation is available in ALL
        THREE observation modes -- it used to be greyed out under χ, on the assumption that χ already
        decorrelated what the rotation targets; measured on the master cell, χ leaves k~x_scale at
        0.95 (vs 0.98 forced), so that assumption was wrong and the exclusion is gone."""
        on = self.chi_check.isChecked()
        for w in (self.chi_k, self.chi_pad, self.chi_f0, self.chi_range, self.chi_cycles):
            w.setEnabled(on)
        self.rot_check.setEnabled(True)
        self.rot_check.setToolTip(
            "In χ(ω) mode the Fisher is built over the χ feature set, which costs (K+1)/2× what a "
            "forced-mode rotation does." if on else "")

    def _show_model_units(self, model: str):
        """Reflect the model's declared units, and seed the direct-entry box from them so switching to
        'Type units' starts from the truth rather than an empty field."""
        try:
            tokens = file_manager.parse_units_file(cli.resolve_units_file(model))
        except Exception as e:                         # noqa: BLE001 -- a missing units file is the
            self.units_default.setText(f"(no units file for {model}: {e})")   # model's problem, not a crash
            return
        self.units_default.setText(" ".join(tokens) + f"   —  Resources/Units/{model.lower()}/units.txt")
        if not self.units_text.text().strip():
            self.units_text.setText(" ".join(tokens))

    def _units_override(self):
        """None => use the model's units file; a token tuple => the user typed them."""
        if not self.units_toggle.is_direct():
            return None
        tokens = tuple(self.units_text.text().split())
        return tokens or None

    def _on_model_changed(self, model: str):
        self._show_model_units(model)
        # The bounds picker lives on the PRIOR tab now (bounds are what build the config), and it is
        # repointed via InferenceScreen.on_draft_set when the model is APPLIED -- not on every combo
        # change, so a half-changed selection cannot leave the two tabs disagreeing.
        # User models are inferable only when SBI-eligible: no forcing (spontaneous dynamics) AND at
        # least one ND parameter. Forced / zero-parameter user models stay Simulate-only.
        ineligible_user = registry.is_user_model(model) and not registry.is_sbi_user_model(model)
        self.btn_config.setEnabled(not ineligible_user)
        if ineligible_user:
            reason = ("has external forcing" if registry.user_model_has_forcing(model)
                      else "has no free parameters to infer")
            self.log_pane.append_line(
                f"'{model}' {reason}, so it is Simulate-only. Parameter inference supports "
                "user-defined models with no forcing and at least one parameter.", "warning")

    def _read_inputs(self) -> dict:
        """Every box Apply will read, through ``value_or_none()`` and the shared rules; the first bad
        one raises ``Refusal`` (spec §3.4). The keys are the field keys of ``core.refusals.FIELDS``.

        The χ boxes are read only while χ mode is on: off, they are disabled and the draft carries
        None for them, so ``make_sim_config`` takes config.py's values -- the tool's behaviour. A
        blank box is a refusal, never a zero: ``IntField.value()`` returned 0 for "" and the old
        chain refused it by accident (0 < 2), while a blank pad was not checked here at all and
        surfaced one tab later as SimConfig.__post_init__'s traceback dialog.
        """
        out = {"units": None, "chi_n_freqs": None, "chi_k_pad": None, "chi_max_cycles": None}
        if self.chi_check.isChecked():
            # K has an UPPER bound too. Cost is linear in K (each probe is a whole extra simulation
            # per observation, so training and calibration both scale as K+1), and the Infer tab
            # grows one file-picker row per probe frequency -- K=500 would mean 500 rows.
            k = require_at_least("chi_n_freqs", self.chi_k.value_or_none(), 2)
            if k > CHI_K_MAX:
                raise Refusal(f"The number of chi probe frequencies must be at most {CHI_K_MAX}; "
                              f"got {k} (default {config.CHI_N_FREQS}).", field="chi_n_freqs")
            pad = require_at_least("chi_k_pad", self.chi_pad.value_or_none(), 2)
            if pad > CHI_K_MAX:
                raise Refusal(f"The number of chi probe slots must be at most {CHI_K_MAX}; got {pad} "
                              f"(default {config.CHI_K_PAD}).", field="chi_k_pad")
            if k > pad:
                raise Refusal(f"The number of chi probe frequencies must be at most the number of "
                              f"chi probe slots, {pad}; got {k} (default {config.CHI_N_FREQS}).",
                              field="chi_n_freqs")
            # Caught here rather than by SimConfig.__post_init__ so it reads as a form error next to
            # the box, not as a dialog on the Prior tab's "Build / Load prior".
            cycles = require_finite("chi_max_cycles", self.chi_cycles.value_or_none())
            if cycles <= config.CHI_MIN_CYCLES:
                raise Refusal(f"The chi lock-in ceiling, in cycles must be greater than "
                              f"{config.CHI_MIN_CYCLES:g}, the floor below which a probe is masked; "
                              f"got {cycles:g} (default {config.CHI_MAX_CYCLES:g}).",
                              field="chi_max_cycles")
            out.update(chi_n_freqs=k, chi_k_pad=pad, chi_max_cycles=cycles)
        if self.units_toggle.is_direct():
            units = self._units_override()
            if units is None:
                raise Refusal("The units are blank: type at least one unit token, or use the "
                              "model's units file.", field="units")
            try:                                       # reject unresolvable tokens HERE, not mid-run
                cli.units_to_factors(units)
            except Exception as e:                     # noqa: BLE001 -- UnitParseError, or pint's own
                raise Refusal(f"The units are not usable: {e}", field="units") from e
            out["units"] = units
        return out

    def _build_config(self):
        model = self.model_combo.currentText()
        if registry.is_user_model(model) and not registry.is_sbi_user_model(model):   # backstop
            self.log_pane.append_line(
                "This user-defined model is Simulate-only (needs no forcing + ≥1 parameter for "
                "inference).", "warning")
            return
        try:
            v = self._read_inputs()
        except Refusal as e:
            self._refusal(e)
            return
        # model_labels, NOT `labels`: this module imports the core.Helpers.labels MODULE at the top
        # and calls labels.axis_label(...) / labels.gui_forcing_label(...) elsewhere in this same
        # file. Binding a local of that name shadowed it for the whole function, so any future line
        # added here that touched the module would raise AttributeError on a list -- a crash sitting
        # one edit away, and invisible until someone made that edit.
        model_labels = VALID_LABELS[VALID_MODELS.index(model)]
        state_dep_drift = registry.state_dep_drift(model)
        chi_on = self.chi_check.isChecked()
        # chi_f0 / chi_freq_bounds are NEVER read from their boxes: they are read-only displays of
        # config.py, and None makes make_sim_config take config.py's values (cli.py), exactly as the
        # command-line tool's config does. _assert_chi_config_is_deliberate therefore cannot fire
        # on a config built from this tab; it stays as the last line of defence for every other caller.
        draft = ConfigDraft(
            model=model, labels=model_labels, state_dep_drift=state_dep_drift,
            units_override=v["units"],
            chi_mode=chi_on, chi_n_freqs=v["chi_n_freqs"], chi_f0=None, chi_freq_bounds=None,
            chi_k_pad=v["chi_k_pad"], chi_max_cycles=v["chi_max_cycles"],
            reparam_rotate=self.rot_check.isChecked())
        self._screen.new_draft(draft)                # replaces the session + repoints Prior + re-gates
        extras = []
        if chi_on:
            lo, hi = config.CHI_FREQ_BOUNDS          # config.py's, not the boxes': the values the run gets
            extras.append(f"χ(ω) on — {v['chi_n_freqs']} frequencies over {lo:g}–{hi:g}×Ω₀ at ND "
                          f"amplitude {config.CHI_F0:g}, ≤{v['chi_max_cycles']:g} cycles per probe, "
                          f"so expect ~{(v['chi_n_freqs'] + 1) / 2:.1f}× the "
                          f"usual training time and train a NEW posterior")
        elif draft.reparam_rotate:
            extras.append("decorrelating Fisher rotation on")
        self.log_pane.append_line(
            f"Model applied: {model}" + (f" ({'; '.join(extras)})" if extras else "")
            + ". Now pick a bounds file on the Prior tab — that builds the config and selects the "
              "observation mode.")
        if registry.is_user_model(model):
            self.log_pane.append_line(
                "Note: user-model inference runs the full pipeline (spontaneous dynamics only), but "
                "calibration is NOT pre-tuned — check the Validate tab's SBC/TARP results for this model.",
                "warning")

    def save_settings(self, qs):
        """The SELECTIONS only (V5). The chi probe count, slots and lock-in ceiling are science knobs
        that open at config.py on every launch, and the amplitude and band are read-only displays:
        none of the six is written, and a stale `chi_k` / `chi_k_pad` / `chi_f0` / `chi_lo` /
        `chi_hi` / `chi_max_cycles` key in an old PRISM.ini is ignored by restore_settings. Seeding
        those boxes from config.py and then restoring them is what trained the 2026-08-19 retrain
        on the retired band."""
        qs.beginGroup("inference_config")
        qs.setValue("model", self.model_combo.currentText())
        qs.setValue("units_mode", self.units_toggle.key())
        settings.save_field(qs, "units_text", self.units_text)
        settings.set_bool(qs, "chi_mode", self.chi_check.isChecked())
        settings.set_bool(qs, "reparam_rotate", self.rot_check.isChecked())
        qs.endGroup()

    def restore_settings(self, qs):
        qs.beginGroup("inference_config")
        # Explicit _on_model_changed: currentTextChanged won't fire if the value already equals the
        # default. (The bounds picker moved to the Prior tab, which restores its own key in on_draft_set,
        # so the old restore-order trap no longer applies here.)
        self.model_combo.setCurrentText(settings.get_str(qs, "model", self.model_combo.currentText()))
        # units_text BEFORE _on_model_changed: the latter seeds the box from the model's file only when
        # it is empty, so a restored custom value must already be in place to survive.
        settings.restore_field(qs, "units_text", self.units_text)
        self._on_model_changed(self.model_combo.currentText())
        self.units_toggle.restore_key(settings.get_str(qs, "units_mode", "file"))
        self.chi_check.setChecked(settings.get_bool(qs, "chi_mode", False))
        self.rot_check.setChecked(settings.get_bool(qs, "reparam_rotate", config.REPARAM_ROTATE))
        qs.endGroup()
