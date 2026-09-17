import traceback

from PySide6.QtWidgets import (QGroupBox, QHBoxLayout, QLineEdit, QPushButton, QVBoxLayout)

from core import config, forcing, orchestrator
from core.artifacts import default_store
from core.Helpers import file_manager
from core.config import BOUNDS_PATH
from core.refusals import Refusal, require_at_least, require_positive

from ... import icons, settings
from ...fields import label
from ...widgets.artifact_picker import ArtifactPicker, StorePicker
from ...widgets.forms import make_form
from ...widgets.help_badge import add_help_row, with_badge
from ...widgets.labeled_inputs import FloatField, IntField, PathField
from ...widgets.param_grid import BoundsGrid, ValuesGrid
from ...widgets.source_toggle import SourceToggle
from .base import _StagePanel, _TrainingBudgetMixin
from .help_text import HELP


# THE SEVEN KNOB BOXES, in the order the stage checks them (build_prior's resolution block, spec
# §3.4): the field key (which is also build_prior's keyword name), the attribute holding the box, and
# the rule's floor -- an int for require_at_least, None for require_positive. ONE table, read by
# _read_inputs at the click and by _sync_sweep on every keystroke, so the note under the boxes and
# the refusal at the button can never name different limits.
_KNOB_RULES = (
    ("num_iterations", "sweep_iters", 1),
    ("sweep_batch", "sweep_batch", 0),            # a TYPED 0 = follow the hardware batch
    ("max_sets", "sweep_max_sets", 1),
    ("walk_step", "sweep_step", None),
    ("stability_units", "sweep_units", None),
    ("min_cluster_size", "cluster_size", 2),
    ("min_samples", "cluster_samples", 1),
)
# The boxes the live sweep note reads: walk_step and the two clustering boxes are not in the census line.
_NOTE_KEYS = ("num_iterations", "sweep_batch", "max_sets", "stability_units")


def _check_knob(key: str, value, minimum):
    """Run one knob's rule: require_at_least with its floor, or require_positive when the floor is
    None. Returns the coerced value; raises Refusal(field=key) on a blank or an out-of-rule value."""
    return require_positive(key, value) if minimum is None else require_at_least(key, value, minimum)


def _rule_words(minimum) -> str:
    """The rule as the live note says it: the words the refusal's own message uses after "must be"."""
    return "greater than 0" if minimum is None else f"at least {minimum}"


# ── 2. Prior (also picks the BOUNDS file, which is what builds the config) ────
class PriorPanel(_StagePanel):
    """Tab 2. Picks the BOUNDS file -- which is what turns the draft into a real SimConfig -- then
    builds or loads the parameter prior.

    Installs the config IN PLACE (``install_config``), not as a new session: building the prior is
    the first step of the existing session, so replacing it would discard the draft.

    Persists (group "inference_prior"): the bounds and prior picker selections. NOT the bounds grid --
    parameter names and order belong to the model, so hand-entry only ever edits numbers.
    """
    def __init__(self, screen, parent=None):
        super().__init__(screen, parent)
        self._saved_bounds_key = ""              # resolved in on_draft_set (needs the chosen model)
        box = QGroupBox("Prior")
        v = QVBoxLayout(box)
        form = make_form()
        self.bounds_picker = ArtifactPicker(BOUNDS_PATH / "nadrowski")
        self.bounds_grid = BoundsGrid()
        self.bounds_source = SourceToggle(self.bounds_picker, self.bounds_grid,
                                          file_label="Use file", direct_label="Edit values")
        self.bounds_source.changed.connect(self._on_bounds_source_changed)
        add_help_row(form, label("bounds"), self.bounds_source, HELP["bounds_source"])
        self.prior_picker = StorePicker("prior", allow_new=True)
        add_help_row(form, label("prior"), self.prior_picker, HELP["prior"])
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

        # -- the stability sweep: what actually builds the prior ---------------------------------
        sweep = QGroupBox("Stability sweep")
        sv = QVBoxLayout(sweep)
        sform = make_form()
        self.sweep_iters = IntField(str(config.PRIOR_SWEEP_ITERATIONS))
        self.sweep_batch = IntField(str(config.PRIOR_SWEEP_BATCH))
        self.sweep_max_sets = IntField(str(config.PRIOR_SWEEP_MAX_SETS))
        self.sweep_step = FloatField(str(config.PRIOR_SWEEP_STEP))
        self.sweep_units = FloatField(str(config.STABILITY_SWEEP_ND_UNITS))
        add_help_row(sform, label("num_iterations"), self.sweep_iters, HELP["sweep_iters"])
        add_help_row(sform, label("sweep_batch"), self.sweep_batch, HELP["sweep_batch"])
        add_help_row(sform, label("max_sets"), self.sweep_max_sets, HELP["sweep_max_sets"])
        add_help_row(sform, label("walk_step"), self.sweep_step, HELP["sweep_step"])
        add_help_row(sform, label("stability_units"), self.sweep_units, HELP["sweep_units"])
        sv.addLayout(sform)
        self.sweep_note = _TrainingBudgetMixin._derived_label()
        sv.addWidget(self.sweep_note)
        for fld in (self.sweep_iters, self.sweep_batch, self.sweep_max_sets, self.sweep_units):
            fld.textChanged.connect(lambda _t: self._sync_sweep())
        self.controls_layout.addWidget(sweep)
        self._sync_sweep()

        # -- clustering: a different STAGE from the sweep. The sweep maps the stable manifold;
        # this decides how many MODES the prior has, because HDBSCAN's label count becomes the
        # GMM's n_components.
        clust = QGroupBox("Clustering / GMM")
        cv = QVBoxLayout(clust)
        cform = make_form()
        self.cluster_size = IntField(str(config.PRIOR_CLUSTER_MIN_SIZE))
        self.cluster_samples = IntField(str(config.PRIOR_CLUSTER_MIN_SAMPLES))
        add_help_row(cform, label("min_cluster_size"), self.cluster_size, HELP["cluster_size"])
        add_help_row(cform, label("min_samples"), self.cluster_samples, HELP["cluster_samples"])
        cv.addLayout(cform)
        self.controls_layout.addWidget(clust)

        self.restore_settings(settings.settings())

    def on_draft_set(self, draft):
        """Config applied: repoint the bounds picker at the new model's folder and re-apply the saved
        key (it could not resolve at __init__, before any model was chosen)."""
        self.bounds_picker.repoint(BOUNDS_PATH / draft.model.lower(), self._saved_bounds_key)
        if self.bounds_source.is_direct():        # a different model means a different parameter set
            self._on_bounds_source_changed()

    def _on_bounds_source_changed(self):
        """Entering direct-entry mode seeds the grid FROM the selected file: the parameter names and
        their order belong to the model, so hand-entry edits numbers rather than inventing a schema."""
        if not self.bounds_source.is_direct():
            return
        path = self.bounds_picker.selected_path()
        if not path:
            self.log_pane.append_line("Select a bounds file first — direct entry starts from it.",
                                      "warning")
            self.bounds_source.set_direct(False)
            return
        try:
            params, rescale, forcing, _ = file_manager.parse_bounds_file(path)
        except Refusal as e:                         # the file's problem: the yellow box
            self._refusal(e)
            self.bounds_source.set_direct(False)
            return
        except Exception as e:                       # noqa: BLE001 -- a bug in the parser: the red box
            self._on_error(e, traceback.format_exc())
            self.bounds_source.set_direct(False)
            return
        self.bounds_grid.load(params, rescale, forcing)

    def _build_prior(self):
        """Build the SimConfig from (Config draft + this tab's bounds file), then build/load the prior.

        The config is built HERE because the bounds file is what defines the inferred parameter set --
        and therefore the observation mode -- so it cannot exist until this tab has been used."""
        draft = self.session.draft
        if draft is None:
            return
        entry, is_new = self.prior_picker.selected()
        # THE CLICK-TIME HALF OF V2, before the config is built and before the session's downstream
        # is reset: a refused click leaves the session exactly as it found it. What is read depends
        # on the branch the stage will take (_read_inputs). The stage runs the same rules again at
        # its entry; this is the early, cheap copy that names the box.
        try:
            knobs = self._read_inputs(is_new)
        except Refusal as e:
            self._refusal(e)
            return
        if self.bounds_source.is_direct():
            problems = self.bounds_grid.problems()
            if problems:
                self.log_pane.append_line("Fix the bounds first: " + "; ".join(problems), "warning")
                return
            source = dict(bounds_dicts=self.bounds_grid.to_dicts())
        else:
            bounds_path = self.bounds_picker.selected_path()
            if not bounds_path:
                self.log_pane.append_line("Select a bounds file first.", "warning")
                return
            source = dict(bounds_path=bounds_path)
        try:
            cfg = draft.make_config(**source)
        except Refusal as e:                         # a bounds or units problem: the yellow box
            self._refusal(e)
            return
        except Exception as e:                       # noqa: BLE001 -- a bug in the builder: the red box
            self._on_error(e, traceback.format_exc())
            return
        for msg in cfg.check_unit_consistency():      # a units declaration that contradicts the pipeline
            self.log_pane.append_line(msg, "warning")
        self.session.reset_downstream("prior")
        self._screen.install_config(cfg)             # sets session.cfg + repoints the Infer tab + re-gates
        _MODE_BLURB = {
            "spontaneous": "one passive trace, no drive anywhere",
            "forced": "passive + one forced trace at the cell's drive",
            "chi": "passive + K single-tone forced traces (the cell's own drive is ignored)",
        }
        self.log_pane.append_line(
            f"Config built: {cfg.model} — {len(cfg.params_dict)} ND + {len(cfg.rescale_params)} rescale "
            f"params. Observation mode: {cfg.observation_mode.upper()} "
            f"({_MODE_BLURB[cfg.observation_mode]}).")
        if cfg.observation_mode == "spontaneous" and "f_scale" in cfg.rescale_params:
            self.log_pane.append_line(
                "This config infers f_scale but has no drive anywhere, so f_scale cannot affect the "
                "observable — its marginal will just return the prior. Use a bounds file without a "
                "Forcing section AND without f_scale for spontaneous inference.", "warning")
        if cfg.chi_mode:
            lo, hi = cfg.chi_freq_bounds
            # Width from the SHARED rule, not 3*K: that was layout 1, where a probe's frequency was
            # implied by its slot. Under the padded probe set it is CHI_ELEM_W * chi_k_pad and does
            # not depend on K at all -- which is the entire point of the layout, so reporting the old
            # formula here told the user the opposite of what the mode now does.
            self.log_pane.append_line(
                f"χ(ω) mode: {cfg.chi_n_freqs} drive frequencies over {lo:g}–{hi:g}×Ω₀ at ND amplitude "
                f"{cfg.chi_f0:g}, each locked in over at most {cfg.chi_max_cycles:g} drive cycles; "
                f"conditioning is [S(41) | log T | χ({orchestrator.expected_forcing_dim(cfg)})] over "
                f"{cfg.chi_k_pad} probe slots. Train a NEW posterior (the width differs from a non-χ one).")

        # Passed, never written to config: orchestrator does `from .config import
        # PRIOR_SWEEP_ITERATIONS, ...`, so assigning to the constants here would be a silent no-op.
        # Unclamped and undefaulted: _read_inputs already refused a blank or out-of-rule box. On a
        # load `knobs` is empty and the stage resolves every None to its config.py default -- its
        # load branch (`if not build_new and ref is not None:`) never reads them.
        self.dispatch(orchestrator.build_prior, cfg, entry, is_new,
                      provide_fig_sink=True, on_result=self._on_prior, **knobs)

    def _read_inputs(self, is_new: bool) -> dict:
        """The click-time half of V2 for this tab: every knob box the chosen branch will read, read
        through value_or_none() and the shared rules (core.refusals), as a dict of build_prior's
        keyword arguments. Raises the FIRST Refusal, before any config is built.

        The branch is decided as the stage decides it: build_prior takes its load branch on
        ``not build_new and ref is not None`` and never reads a sweep or clustering knob there, so a
        LOAD click reads none of the seven boxes and forwards none (the stage resolves each None to
        its config.py default). Refusing a blank "Min cluster size" on a load would refuse a value
        the run never uses. On a load the picker is the only input, and StorePicker.selected()
        returns is_new=False only with an id in hand, so there is nothing left to check here.
        """
        if not is_new:
            return {}
        return {key: _check_knob(key, getattr(self, attr).value_or_none(), minimum)
                for key, attr, minimum in _KNOB_RULES}

    def _sweep_problem(self) -> "str | None":
        """The first of the four boxes the note reads that is blank or out of rule, as the line the
        note shows; None when all four pass. The click's own rule on the same value, rendered short
        ("<label> is blank." / "<label> must be <rule>."): a line under a box the user is still
        typing in does not shout the default at them, and it never pops a dialog."""
        for key, attr, minimum in _KNOB_RULES:
            if key not in _NOTE_KEYS:
                continue
            value = getattr(self, attr).value_or_none()
            if value is None:
                return f"{label(key)} is blank."
            try:
                _check_knob(key, value, minimum)
            except Refusal:
                return f"{label(key)} must be {_rule_words(minimum)}."
        return None

    def _sync_sweep(self) -> None:
        """The one derived line: how many candidates the GLOBAL census screens, and where the time
        goes. Pure and cheap, so it is safe on every keystroke; wrapped because a status line must
        never be able to raise into refresh_gates and take the tab down.

        A blank, half-typed or out-of-rule box renders as "<label> is blank." or "<label> must be
        <rule>." and nothing else (spec §1.2, the live lines): no clamp, no default, no dialog. A
        number computed from a value the user did not type is a lie about the run, and a live line
        cannot open a dialog mid-typing -- "blank is a refusal" is the click's rule, not the note's.
        """
        try:
            problem = self._sweep_problem()
            if problem is not None:
                self.sweep_note.setText(problem)
                return
            cfg = self.session.cfg
            hw_batch = getattr(getattr(cfg, "hw", None), "batch_size", None) or config.detect_device().batch_size
            rounds = self.sweep_iters.value_or_none()
            batch = self.sweep_batch.value_or_none()
            per_round = batch if batch else hw_batch      # a TYPED 0 = follow the hardware batch
            max_sets = self.sweep_max_sets.value_or_none()
            units = self.sweep_units.value_or_none()
            dt = getattr(cfg, "dt_nd_min", None)
            steps = f"{int(units / dt):,}" if dt else "?"
            self.sweep_note.setText(
                f"Global census screens {rounds * per_round:,} candidates ({rounds:,} rounds x "
                f"{per_round:,}), each integrated over {steps} steps.\n"
                f"The LOCAL flood-fill then runs until {max_sets:,} sets "
                f"are accepted — that is the dominant cost of a prior build, and it now runs on the "
                f"same device as the global sweep (falling back to the CPU when there is no "
                f"accelerator).")
        except Exception as e:                    # noqa: BLE001 -- never break the tab over a label
            self.sweep_note.setText(f"Sweep summary unavailable: {type(e).__name__}: {e}")

    def _on_prior(self, payload):
        self.session.inf_prior = payload                   # a LoadedPrior
        self.log_pane.append_line(f"Prior ready: {payload.name or '(unnamed, id ' + payload.id + ')'}. "
                                  f"Name it below to keep it.")
        self._screen.refresh_gates()

    def _save_prior(self):
        name = self.prior_name.text().strip()
        lp = self.session.inf_prior
        if not name or lp is None:
            self.log_pane.append_line("Build a prior and enter a name first.", "warning")
            return
        try:
            lp.manifest = default_store().rename("prior", lp.id, name)
        except Refusal as e:                         # a bad or taken name is a StoreError(field="name")
            self._refusal(e)
            return
        except Exception as e:                       # noqa: BLE001 -- anything else is a bug
            self._on_error(e, traceback.format_exc())
            return
        lp.name = name
        self.prior_picker.refresh()
        self.prior_picker.restore_key(lp.id)
        self.log_pane.append_line(f"Prior named '{name}'.")

    def refresh_local_gates(self):
        self.btn_prior.setEnabled(self.session.draft is not None)
        self.btn_save_prior.setEnabled(self.session.inf_prior is not None)

    def save_settings(self, qs):
        qs.beginGroup("inference_prior")
        qs.setValue("prior", self.prior_picker.key())
        qs.setValue("bounds", self.bounds_picker.key())
        qs.setValue("bounds_source", self.bounds_source.key())
        for name in ("sweep_iters", "sweep_batch", "sweep_max_sets", "sweep_step", "sweep_units",
                     "cluster_size", "cluster_samples"):
            qs.setValue(name, str(getattr(self, name).value()))
        qs.endGroup()
        # The bounds GRID is not persisted: it is seeded from whichever file is selected, so restoring a
        # stale hand-edited grid against a different model/bounds would silently mis-bind parameters.

    def restore_settings(self, qs):
        qs.beginGroup("inference_prior")
        self.prior_picker.restore_key(settings.get_str(qs, "prior"))
        # The bounds picker points at CONFIG's model, which is not known at __init__ -- stash the key and
        # re-apply it in on_draft_set (the same deferred-restore trap the cell pickers have).
        self._saved_bounds_key = settings.get_str(qs, "bounds")
        # str + cast, because settings has no get_float; a blank or unparseable value falls back to
        # the config constant rather than to FloatField.value()'s 0.0 -- and a 0 here would mean a
        # sweep with no rounds, or a flood-fill that stops at zero accepted sets.
        for name, default, cast in (("sweep_iters", config.PRIOR_SWEEP_ITERATIONS, int),
                                    ("sweep_batch", config.PRIOR_SWEEP_BATCH, int),
                                    ("sweep_max_sets", config.PRIOR_SWEEP_MAX_SETS, int),
                                    ("sweep_step", config.PRIOR_SWEEP_STEP, float),
                                    ("sweep_units", config.STABILITY_SWEEP_ND_UNITS, float),
                                    ("cluster_size", config.PRIOR_CLUSTER_MIN_SIZE, int),
                                    ("cluster_samples", config.PRIOR_CLUSTER_MIN_SAMPLES, int)):
            try:
                getattr(self, name).setText(str(cast(settings.get_str(qs, name, str(default)))))
            except (TypeError, ValueError):
                getattr(self, name).setText(str(default))
        # Always start in FILE mode: direct entry has to be seeded from a file, and no file is selected
        # until on_draft_set runs. The saved mode is deliberately not restored for that reason.
        self.bounds_source.set_direct(False)
        qs.endGroup()


# ── the training budget, shared by the Posterior and TSNPE tabs ───────────────────────────────
