import math
import traceback

from PySide6.QtWidgets import (QCheckBox, QComboBox, QGroupBox, QHBoxLayout, QLabel, QPushButton,
                               QVBoxLayout, QWidget)

from core import cli, config, forcing, orchestrator
from core.artifacts import Accept
from core.Helpers import file_manager, labels
from core.config import T_MIN_EXP_S
from core.refusals import Refusal, require_file, require_finite, require_positive

from ... import icons, settings
from ...fields import label
from ...widgets.adaptive_stack import AdaptiveStack
from ...widgets.forms import make_form
from ...widgets.help_badge import add_help_row, with_badge
from ...widgets.labeled_inputs import FloatField, IntField, PathField
from ...widgets.param_grid import BoundsGrid, ValuesGrid
from ...widgets.source_toggle import SourceToggle
from core.SBI.observations import RecordingSet

from .rows import _ChiProbeRow
from .base import _CellPreviewMixin, _StagePanel
from .help_text import HELP

# The driven page's drive boxes, by forcing name: the registered field key each answers to. Those
# three rows take their label from the control table (fields.CONTROL, through label(key)), so a
# rename there renames the row and the refusal's "Set it in the '…' box" sentence together; a forcing
# name outside the registry (offset, amp_y) keeps its derived pretty label and answers to no key.
_DRIVE_FIELD = {"amp": "drive_amplitude", "freq": "drive_frequency", "phase": "drive_phase"}


def _drive_label(name: str) -> str:
    key = _DRIVE_FIELD.get(name)
    if key is not None:
        return label(key)
    return labels.gui_forcing_label(name, config.FORCING_DISPLAY_UNITS.get(name, ""))


def _probe_frequency(row) -> "float | None":
    """A probe row's drive frequency when one is given, finite and positive; None for a blank, a
    half-typed or a non-positive box (0 Hz is a DC probe the lock-in would attempt)."""
    f = row.freq.value_or_none()
    return f if f is not None and math.isfinite(f) and f > 0 else None


# ── 5. Infer ──────────────────────────────────────────────────────────────────
class InferPanel(_StagePanel, _CellPreviewMixin):
    """Tab 5. Infers on a simulated observation (from a cell's ground truth) or on real recordings.

    Its cell picker follows the BUILT config's model rather than a live combo -- there isn't one in
    this tab. In chi mode the experimental page grows one file-picker row per probe frequency, which
    is why K is bounded at ``config.CHI_K_MAX``.

    Persists (group "inference_infer"): the mode, the cell picker, and the experimental file paths.
    """
    def __init__(self, screen, parent=None):
        super().__init__(screen, parent)
        self._init_cell_picker()
        self._cell_problems = []             # why the picked cell can't be used (empty = usable)
        self.cell_picker.combo.currentIndexChanged.connect(lambda _i: self._on_cell_changed())
        self._forcing_fields = {}            # name -> FloatField (experimental drive)
        self._gated_posterior = None         # which posterior the other-observation box was gated on
        box = QGroupBox("Infer")
        v = QVBoxLayout(box)

        self.infer_mode = QComboBox()
        self.infer_mode.addItems(["Simulated (cell ground truth)", "Experimental data"])
        self.infer_mode.currentIndexChanged.connect(
            lambda _i: (self._sync_infer_page(), self.refresh_local_gates()))
        mode_form = make_form()
        add_help_row(mode_form, "Mode", self.infer_mode, HELP["infer_mode"])
        v.addLayout(mode_form)

        # AdaptiveStack: the simulated page is two rows, the chi page is K+3, so a plain stack left a
        # large dead gap under the short pages.
        self.infer_stack = AdaptiveStack()
        # simulated inputs
        sim_w = QWidget(); sim_f = make_form(sim_w)
        self.sim_tobs = FloatField(T_MIN_EXP_S)
        self.values_grid = ValuesGrid()
        self.cell_source = SourceToggle(self.cell_picker, self.values_grid,
                                        file_label="Use file", direct_label="Edit values")
        self.cell_source.changed.connect(self._on_cell_source_changed)
        add_help_row(sim_f, label("cell"), self.cell_source, HELP["cell_source"])
        add_help_row(sim_f, label("t_obs"), self.sim_tobs, HELP["tobs"])
        self.infer_stack.addWidget(sim_w)
        # experimental inputs
        exp_w = QWidget(); self.exp_form = make_form(exp_w)
        self.exp_spont = PathField()
        self.exp_forced = PathField()
        self.exp_tobs = FloatField(T_MIN_EXP_S)
        # "Spontaneous" and the χ page's "Passive" are one field key (recording_spont) with two boxes,
        # so its CONTROL entry is a sentence naming both, and the two literals stay here.
        add_help_row(self.exp_form, "Spontaneous", self.exp_spont, HELP["spont"])
        add_help_row(self.exp_form, label("recording_forced"), self.exp_forced, HELP["forced"])
        add_help_row(self.exp_form, label("t_obs"), self.exp_tobs, HELP["tobs"])
        self._forcing_anchor = QLabel("(build config to list drive params)")
        self.exp_form.addRow(self._forcing_anchor)
        self.infer_stack.addWidget(exp_w)
        # page 2: experimental, chi(omega) -- one passive recording + K single-tone forced recordings
        chi_w = QWidget(); self.chi_form = make_form(chi_w)
        self.chi_spont = PathField()
        self.chi_tobs = FloatField(T_MIN_EXP_S)
        self.chi_f0_si = FloatField(1.0)
        self._chi_forced_fields = []
        add_help_row(self.chi_form, "Passive", self.chi_spont, HELP["chi_passive"])
        add_help_row(self.chi_form, label("t_obs"), self.chi_tobs, HELP["tobs"])
        add_help_row(self.chi_form, label("chi_f0_si"), self.chi_f0_si, HELP["chi_f0_si"])
        # The probe table. Rows live in their OWN container rather than as form rows, so
        # adding and removing one is a local layout edit that cannot disturb the fields above it.
        self._chi_probe_host = QWidget()
        self._chi_probe_layout = QVBoxLayout(self._chi_probe_host)
        self._chi_probe_layout.setContentsMargins(0, 0, 0, 0)
        add_help_row(self.chi_form, "Forced probes", self._chi_probe_host, HELP["chi_forced"])
        self._chi_buttons = chi_btns = QWidget(); chi_btns_l = QHBoxLayout(chi_btns)
        chi_btns_l.setContentsMargins(0, 0, 0, 0)
        self.btn_chi_add = QPushButton("+ Add probe")
        self.btn_chi_add.clicked.connect(lambda: self._add_chi_probe())
        self.btn_chi_plan = QPushButton("Plan probes…")
        self.btn_chi_plan.setToolTip("Measure Ω₀ from the passive recording and report what is in "
                                     "band, and how long each probe must be recorded.")
        self.btn_chi_plan.clicked.connect(self._plan_chi_probes)
        chi_btns_l.addWidget(self.btn_chi_add)
        chi_btns_l.addWidget(self.btn_chi_plan)
        chi_btns_l.addStretch(1)
        self.chi_form.addRow(chi_btns)
        self._chi_anchor = QLabel("(build a χ config to enable the probe table)")
        self.chi_form.addRow(self._chi_anchor)
        self.infer_stack.addWidget(chi_w)
        v.addWidget(self.infer_stack)

        # D8: the GUI's only way to run a TSNPE posterior on an observation other than its region's.
        # Enabled only when the session's posterior IS non-amortized -- on an amortized one it would
        # mean nothing -- and cleared whenever the posterior changes.
        self.other_obs = QCheckBox("Run on a different observation")
        v.addWidget(with_badge(self.other_obs, HELP["infer_other_obs"]))

        self.btn_infer = QPushButton("Run inference")
        self.btn_infer.setProperty("accent", True)        # primary CTA (Fluent accent)
        self.btn_infer.clicked.connect(self._infer)
        v.addWidget(self.btn_infer)
        self.controls_layout.addWidget(box)
        self.restore_settings(settings.settings())

    def _on_cell_source_changed(self):
        """Entering direct-entry seeds the grid from the picked cell -- same rule as the bounds grid:
        the parameter schema belongs to the model, only the numbers are the user's."""
        if not self.cell_source.is_direct():
            self._on_cell_changed()                  # back to file mode: re-validate the picked file
            return
        path = self.cell_picker.selected_path()
        if not path:
            self.log_pane.append_line("Select a cell file first — direct entry starts from it.",
                                      "warning")
            self.cell_source.set_direct(False)
            return
        try:
            inits, params, rescale, forcing = file_manager.parse_values_file(path)
        except Refusal as e:                         # the file's problem: the yellow box
            self._refusal(e)
            self.cell_source.set_direct(False)
            return
        except Exception as e:                       # noqa: BLE001 -- a bug in the parser: the red box
            self._on_error(e, traceback.format_exc())
            self.cell_source.set_direct(False)
            return
        self.values_grid.load(inits, params, rescale, forcing)
        self._cell_problems = []                     # hand-entered values are validated at Run instead
        self.refresh_local_gates()

    def _on_cell_changed(self):
        """Validate the picked cell against the bounds file ON THE GUI THREAD, the moment it is chosen.

        The check already existed inside inject_ground_truth, but it only fired inside the worker -- so a
        mismatched cell surfaced as a mid-run error dialog after the user had already committed. Here it
        is immediate, and the Run button stays disabled until a usable cell is selected."""
        self._cell_problems = []
        cfg, path = self.session.cfg, self.cell_picker.selected_path()
        if cfg is not None and path:
            try:
                self._cell_problems = cli.validate_gt_file(cfg, path)
            except Exception:                          # noqa: BLE001 -- a pre-flight check must never
                self._cell_problems = []               # break the panel; the worker still validates
            if self._cell_problems:
                self.log_pane.append_line(
                    "This cell does not fit the bounds file used to build the config: "
                    + "; ".join(self._cell_problems) + ". Choose another cell (or rebuild the config "
                    "against matching bounds).", "warning")
        self.refresh_local_gates()

    def on_config_built(self, cfg):
        _CellPreviewMixin.on_config_built(self, cfg)
        self._on_cell_changed()                       # re-validate against the newly built config
        self._rebuild_forcing_fields(cfg)
        # A no-forcing (passive) model has no forced recording and no drive params: hide the forced row.
        # _rebuild_forcing_fields already produces no forcing fields for an empty force_params_dict.
        self.exp_form.setRowVisible(self.exp_forced, cfg.has_forcing)
        self._rebuild_chi_fields(cfg)
        self._sync_infer_page()

    def _sync_infer_page(self):
        """Experimental mode shows the χ page when the config is χ-mode. Load-bearing: a χ observation
        needs K forced recordings, and falling through to the ordinary experimental branch would build a
        silently wrong-width conditioning vector rather than failing."""
        if self.infer_mode.currentIndex() == 0:
            self.infer_stack.setCurrentIndex(0)
            return
        cfg = self.session.cfg
        mode = cfg.observation_mode if cfg is not None else "forced"
        self.infer_stack.setCurrentIndex(2 if mode == "chi" else 1)

    def _add_chi_probe(self, freq_hz: float = 0.0):
        """Append one probe row, up to the posterior's slot capacity."""
        cfg = self.session.cfg
        cap = cfg.chi_k_pad if cfg is not None and cfg.chi_mode else config.CHI_K_PAD
        if len(self._chi_forced_fields) >= cap:
            self.log_pane.append_line(
                f"This posterior reserves {cap} probe slots (CHI_K_PAD), which is frozen into the "
                f"trained artifact — it cannot take more probes than that.", "warning")
            return None
        row = _ChiProbeRow(self._remove_chi_probe, freq_hz)
        self._chi_forced_fields.append(row)
        self._chi_probe_layout.addWidget(row)
        return row

    def _remove_chi_probe(self, row):
        if row not in self._chi_forced_fields:
            return
        self._chi_forced_fields.remove(row)
        self._chi_probe_layout.removeWidget(row)
        row.setParent(None)
        row.deleteLater()

    def _rebuild_chi_fields(self, cfg):
        """Enable/disable the probe table for the built config, PRESERVING every existing row.

        Rows are never destroyed on a rebuild, and that is deliberate. They carry hand-typed drive
        frequencies and browsed recording paths -- neither of which this method could regenerate, and
        both of which represent a bench session that already happened. Rebuilding the config (to fix
        a bounds file, say) must not silently discard them. Contrast _rebuild_forcing_fields, whose
        rows ARE derivable from the config's forcing schema and so are rebuilt freely.

        Count and placement are both free: the encoder is permutation-invariant and
        carries each probe's frequency explicitly, so the table seeds a suggested number of rows and
        then gets out of the way. `cfg.chi_n_freqs` is a suggestion, NOT a requirement -- the core
        accepts 1..chi_k_pad probes at whatever frequencies the experiment achieved.
        """
        if self._chi_anchor is not None:
            self.chi_form.removeRow(self._chi_anchor)
            self._chi_anchor = None
        on = bool(cfg.chi_mode)
        self.btn_chi_add.setEnabled(on)
        self.btn_chi_plan.setEnabled(on)
        # setRowVisible, not setVisible: hiding the widget alone strands its form LABEL, so a
        # non-chi config would show a "Forced probes" caption with nothing under it.
        self.chi_form.setRowVisible(self._chi_probe_host, on)
        self.chi_form.setRowVisible(self._chi_buttons, on)
        if not on:
            self._chi_anchor = QLabel("(build a χ config to enable the probe table)")
            self.chi_form.addRow(self._chi_anchor)
            return
        # Seed only when EMPTY -- never top up, never trim. A user who deleted rows meant it.
        if not self._chi_forced_fields:
            for _ in range(max(1, min(int(cfg.chi_n_freqs), cfg.chi_k_pad))):
                self._add_chi_probe()

    def _plan_chi_probes(self):
        """Backlog C-3: say what is in band for THIS cell, and how long each probe must be recorded.

        Every predicate comes from chi.probe_verdict, the same function build_experiment_obs_chi
        refuses and masks on -- so this cannot tell the user one thing and the run another. That is
        the point of the exercise: these answers were previously only discoverable by running the
        inference, i.e. after the bench session rather than before it.

        The band is RELATIVE to the cell's own Ω₀, so nothing useful can be said until a passive
        recording exists. Measuring it needs one load and one FFT, which is why this is a button
        rather than something recomputed on every keystroke. A button is a click, and a click is
        refused like one (V2): a blank or non-positive T_obs and a blank or missing passive recording
        go to the yellow box through _refusal before anything is loaded.
        """
        cfg = self.session.cfg
        if cfg is None or not cfg.chi_mode:
            return
        try:
            t_obs = require_positive("t_obs", self.chi_tobs.value_or_none())
            path = require_file("recording_spont", self.chi_spont.value(), "recording")
        except Refusal as e:
            self._refusal(e)
            return
        try:
            from core.SBI import chi as _chi
            x = file_manager.load_experimental_data(path, dtype=cfg.hw.dtype)
            f_peak = float(_chi.peak_freq(x.unsqueeze(0), cfg.dt_exp))
        except Exception as e:                                  # noqa: BLE001 -- a planner must never
            self.log_pane.append_line(f"Could not measure Ω₀ from {path}: {e}", "error")   # break the panel
            return
        hz = cfg.get_unit_conversion_factor("s")
        lo_hz, hi_hz = _chi.band_hz(cfg, f_peak)
        n_samp = max(1, int(round(t_obs * hz / cfg.dt_exp)))
        self.log_pane.append_line(
            f"Ω₀ = {f_peak * hz:.4g} Hz for this recording. In band for this cell: "
            f"{lo_hz:.3g}–{hi_hz:.3g} Hz "
            f"({cfg.chi_freq_bounds[0]:g}–{cfg.chi_freq_bounds[1]:g}×Ω₀).")
        self.log_pane.append_line(
            f"At the band's low edge a probe needs ≥ {config.CHI_MIN_CYCLES / lo_hz:.3g} s to clear "
            f"the {config.CHI_MIN_CYCLES:g}-cycle floor; above "
            f"{cfg.chi_max_cycles / hi_hz:.3g} s the high edge is truncated to the "
            f"{cfg.chi_max_cycles:g}-cycle ceiling (which is fine — only the tail is dropped).")
        # Fill blank frequency boxes with the nominal in-band grid so the table is usable immediately.
        # Only BLANK ones: a typed frequency is a record of what the bench actually did.
        blanks = [r for r in self._chi_forced_fields if _probe_frequency(r) is None]
        if blanks:
            grid = _chi.chi_multipliers(n_freqs=len(blanks), bounds=cfg.chi_freq_bounds).tolist()
            for row, mult in zip(blanks, grid):
                row.freq.setText(f"{mult * f_peak * hz:.4g}")
            self.log_pane.append_line(
                f"Filled {len(blanks)} blank frequency box(es) with a nominal log-spaced in-band "
                f"grid. These are SUGGESTIONS — replace each with the frequency you actually drove "
                f"at, because a lock-in decays like a sinc and a small mismatch destroys it.")
        # Now report each row's verdict against the T_obs entered.
        for i, row in enumerate(self._chi_forced_fields):
            f = _probe_frequency(row)
            if f is None:
                self.log_pane.append_line(f"  probe {i + 1}: no frequency entered.", "warning")
                continue
            v = _chi.probe_verdict(cfg, f_peak, f, n_samp)
            if v.action == "use":
                self.log_pane.append_line(
                    f"  probe {i + 1}: {f:g} Hz — OK, {v.cycles:.1f} drive cycles at "
                    f"T_obs = {t_obs:g} s.")
            else:
                self.log_pane.append_line(f"  probe {i + 1}: {f:g} Hz — {v.action.upper()}: "
                                          f"{v.reason}.", "warning" if v.action != "refuse" else "error")

    def _rebuild_forcing_fields(self, cfg):
        """One drive box per forcing name of the built config. These rows ARE derivable from the
        config, so they are rebuilt freely (contrast _rebuild_chi_fields); their labels come from
        _drive_label, i.e. from the control table for the three registered names."""
        for fld in self._forcing_fields.values():
            self.exp_form.removeRow(fld)
        self._forcing_fields = {}
        if self._forcing_anchor is not None:
            self.exp_form.removeRow(self._forcing_anchor)
            self._forcing_anchor = None
        for name in cfg.force_params_dict:
            fld = FloatField(0.0)
            self._forcing_fields[name] = fld
            add_help_row(self.exp_form, _drive_label(name), fld, HELP["forcing"])

    def _branch(self) -> str:
        """Which of the four Run paths the mode combo and the built config select: "simulated", "chi",
        "passive" or "driven". ONE decision, read by _read_inputs and _infer, so the boxes validated
        at the click are the boxes the dispatch reads."""
        if self.infer_mode.currentIndex() == 0:
            return "simulated"
        cfg = self.session.cfg
        if cfg.observation_mode == "chi":
            return "chi"
        return "driven" if cfg.has_forcing else "passive"

    def _read_inputs(self) -> dict:
        """Every box the chosen branch will read, through core.refusals' rules; the first bad one
        raises Refusal and the click dispatches nothing (V2). Numbers first, then files, in the
        stage's own order.

        Keys are the field keys. ``cell`` is the picked file's path or, in direct entry, the
        hand-entered value dicts (the two sides of the one Cell row); ``recording_probe`` is the χ
        table's (path, Hz) pairs; ``drive`` is the name-keyed SI dict the recording set takes, whose
        amp/freq/phase entries answer to the three registered drive keys (the tool's one --drive flag
        carries them the same way).
        """
        branch = self._branch()
        if branch == "simulated":
            v = {"t_obs": require_positive("t_obs", self.sim_tobs.value_or_none())}
            if self.cell_source.is_direct():
                problems = self.values_grid.problems()
                if problems:
                    raise Refusal("Fix the values first: " + "; ".join(problems), field="cell")
                v["cell"] = self.values_grid.to_dicts()
                return v
            v["cell"] = require_file("cell", self.cell_picker.selected_path(), "cell")
            if self._cell_problems:
                raise Refusal("This cell does not fit the bounds file used to build the config: "
                              + "; ".join(self._cell_problems) + ". Choose another cell (or rebuild the "
                              "config against matching bounds).", field="cell")
            return v
        if branch == "chi":                          # 1 passive + K single-tone forced
            v = {"t_obs": require_positive("t_obs", self.chi_tobs.value_or_none()),
                 "chi_f0_si": require_positive("chi_f0_si", self.chi_f0_si.value_or_none()),
                 "recording_spont": require_file("recording_spont", self.chi_spont.value(), "recording")}
            if not self._chi_forced_fields:
                raise Refusal("Add at least one forced probe. χ mode conditions on a passive recording "
                              "plus any number of single-tone forced ones, but zero probes is a "
                              "spontaneous observation wearing a χ conditioning vector.",
                              field="recording_probe")
            problems = [p for i, r in enumerate(self._chi_forced_fields) for p in r.problems(i)]
            if problems:
                raise Refusal("Fix the probe table first: " + "; ".join(problems), field="recording_probe")
            # (recording, drive frequency in Hz) PAIRS, never a bare path list. The core locks in at
            # the frequency it is TOLD, rather than assuming mult_k * Omega_0 -- the frequencies a
            # bench achieves are not exactly that, and a lock-in at the wrong frequency decays like a
            # sinc. Pairs come straight off each row widget, so they cannot be mismatched by an
            # add/remove in the middle of the table.
            v["recording_probe"] = tuple((require_file("recording_probe", p, "recording"), f)
                                         for p, f in (r.pair() for r in self._chi_forced_fields))
            return v
        v = {"t_obs": require_positive("t_obs", self.exp_tobs.value_or_none())}
        if branch == "driven":
            v["drive"] = self._read_drive()
        v["recording_spont"] = require_file("recording_spont", self.exp_spont.value(), "recording")
        if branch == "driven":
            v["recording_forced"] = require_file("recording_forced", self.exp_forced.value(), "recording")
        return v

    def _read_drive(self) -> dict:
        """The driven page's drive boxes as {forcing name: SI value}. Amplitude and frequency must be
        > 0 (a zero drive is the passive branch's job, and 0 Hz would reach the lock-in), the phase
        finite. A forcing name outside the registry (offset, amp_y) is a finite number of either
        sign, refused under its own row label with no field key to point at."""
        out = {}
        for name, fld in self._forcing_fields.items():
            key, v = _DRIVE_FIELD.get(name), fld.value_or_none()
            if key == "drive_phase":
                out[name] = require_finite(key, v)
            elif key is not None:
                out[name] = require_positive(key, v)
            elif v is None:
                raise Refusal(f"{_drive_label(name)} is blank.")
            elif not math.isfinite(v):
                raise Refusal(f"{_drive_label(name)} must be a finite number; got {v!r}.")
            else:
                out[name] = v
        return out

    def _infer(self):
        cfg, post = self.session.cfg, self.session.posterior
        if post is None:
            return
        try:
            v = self._read_inputs()
        except Refusal as e:
            self._refusal(e)
            return
        branch = self._branch()
        common = dict(accept=self._accept(), provide_fig_sink=True, on_result=self._on_observation)
        if branch == "simulated":
            direct = self.cell_source.is_direct()
            # ONE flow: the composition in orchestrator carries the ignored-cell note, the T_obs range
            # check, the out-of-distribution check and the up-front non-amortized refusal that used to
            # live in three copies (the GUI runner, orchestrator.run and scripts/_common).
            self.dispatch(orchestrator.simulated_inference, cfg, post, v["t_obs"],
                          cell=None if direct else v["cell"], gt_values=v["cell"] if direct else None,
                          prior=self.session.inf_prior, **common)
            return
        if branch == "chi":                          # experimental, χ(ω): 1 passive + K forced
            rec = RecordingSet(spont=v["recording_spont"], forced=v["recording_probe"],
                               T_obs_s=v["t_obs"], F0_si=v["chi_f0_si"])
        elif branch == "passive":                    # experimental, passive (no drive)
            rec = RecordingSet(spont=v["recording_spont"], T_obs_s=v["t_obs"])
        else:                                        # experimental, driven
            rec = RecordingSet(spont=v["recording_spont"], forced=((v["recording_forced"], None),),
                               T_obs_s=v["t_obs"], forcing_params_si=v["drive"])
        self.dispatch(orchestrator.experimental_inference, cfg, post, rec, **common)

    def _accept(self):
        """The Accept this run opts in with, or None for the default (refuse everything).

        Reads the BOX, and only while it is enabled: a tick left behind by a posterior that has since
        been replaced must never silence guardrail 2 for the new one.
        """
        if self.other_obs.isEnabled() and self.other_obs.isChecked():
            return Accept(other_observation=True)
        return None

    def _on_observation(self, payload):
        obs, inf = payload
        self.session.observation = obs
        self.log_pane.append_line(f"Observation recorded as {obs.name or '(unnamed, id ' + obs.id + ')'}; "
                                  f"the TSNPE tab can build a region around it.")
        cov = inf.results["ppc"]["coverage_90"]
        self.log_pane.append_line(f"Inference {inf.name or '(unnamed, id ' + inf.id + ')'} written; "
                                  f"90% PPC coverage {'n/a' if cov is None else f'{cov:.3f}'}.")
        self._screen.refresh_gates()

    def refresh_local_gates(self):
        simulated = self.infer_mode.currentIndex() == 0
        # _cell_problems only describes the PICKED FILE; hand-entered values are validated at Run.
        blocked = (simulated and not self.cell_source.is_direct()
                   and bool(getattr(self, "_cell_problems", [])))
        self.btn_infer.setEnabled(self.session.posterior is not None and not blocked)
        self.btn_infer.setToolTip(
            "The selected cell does not fit the bounds file used to build the config." if blocked else "")
        # The other-observation box. getattr all the way down: the gate tests put a bare object() on
        # session.posterior, and a status gate must never raise into refresh_gates().
        post = self.session.posterior
        truncated = getattr(getattr(post, "posterior", None), "truncation", None) is not None
        if post is not self._gated_posterior or not truncated:
            self.other_obs.setChecked(False)
        self._gated_posterior = post
        self.other_obs.setEnabled(truncated)

    def save_settings(self, qs):
        qs.beginGroup("inference_infer")
        qs.setValue("cell", self.cell_picker.key())
        qs.setValue("infer_mode", self.infer_mode.currentIndex())
        settings.save_field(qs, "sim_tobs", self.sim_tobs)
        settings.save_field(qs, "exp_tobs", self.exp_tobs)
        settings.save_field(qs, "exp_spont", self.exp_spont)
        settings.save_field(qs, "exp_forced", self.exp_forced)
        settings.save_field(qs, "chi_spont", self.chi_spont)
        settings.save_field(qs, "chi_tobs", self.chi_tobs)
        settings.save_field(qs, "chi_f0_si", self.chi_f0_si)
        qs.endGroup()
        # The forcing fields and the per-frequency χ forced-recording fields don't exist until
        # "Build config" runs (and their COUNT depends on K), so they are not persisted.

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
        settings.restore_field(qs, "chi_spont", self.chi_spont)
        settings.restore_field(qs, "chi_tobs", self.chi_tobs)
        settings.restore_field(qs, "chi_f0_si", self.chi_f0_si)
        qs.endGroup()
