"""FDT mode: the fluctuation-dissipation-theorem analysis.

Mirrors cli.make_fdt_config with widgets, then runs FDT.fdt_pipeline.run_fdt on a
worker.

The two checkboxes are load-bearing. run_fdt's `skip_sanity` / `confirm_production` are REQUIRED
keyword booleans (D1 deleted the prompts they used to fall back to), and these boxes are where a user
answers them -- a worker thread has no terminal to be asked at.
"""
import traceback

from PySide6.QtWidgets import (QCheckBox, QComboBox, QFormLayout, QGroupBox, QLineEdit,
                               QPushButton)

from core import cli, registry
from core.config import CELL_PATH, VALID_MODELS
from core.refusals import Refusal, require_note
from core.FDT.fdt_pipeline import run_fdt
from core.FDT.compare import compare
from core.artifacts import default_store

from . import record_view
from .base_panel import BasePanel
from .. import settings
from ..widgets.artifact_picker import ArtifactPicker, StorePicker
from ..widgets.compare_list import CompareList
from ..widgets.help_badge import add_help_row, with_badge
from ..widgets.labeled_inputs import FloatField, IntField
from ..widgets.forms import make_form
from .. import fields as gui_fields

HELP = {
    "model": "Which model to analyse. FDT supports NADROWSKI, HOPF, BP (experimental), and "
             "additive-noise user-defined models. The frequency grid and burn-in are Nadrowski-tuned, "
             "so treat other models' calibration as your responsibility.",
    "cell": "A cell file whose parameters define the system whose fluctuation-dissipation relation is tested.",
    "n_freqs": "Number of (log-spaced) frequencies at which G(ω) and χ(ω) are evaluated.",
    "ensemble_M": "Ensemble size: independent trajectories averaged per frequency. More reduces noise at higher cost.",
    "freqs_per_batch": "How many frequencies to simulate per batch (a memory/speed trade-off; results are identical).",
    "f0": "Non-dimensional forcing amplitude used to probe the susceptibility χ(ω).",
    "skip_sanity": "Skip the passive-baseline sanity checks and go straight to the production sweep.",
    "confirm_production": "After the sanity checks, proceed to the (long) production sweep automatically.",
    "seed": "The seed this run uses. Leave it blank to draw one — whichever is used is recorded, so a "
            "run can be repeated by typing its seed back here. Repeats of one cell with DIFFERENT "
            "seeds are what measure the spread.",
    "record_name": "A name for the record this run writes. The name is claimed before anything is "
                   "computed, so a name already taken is refused at the click rather than hours in. "
                   "Leave it blank for an unnamed record.",
    "record_note": "Kept with the record and shown in the Artifacts browser.",
    "record": "An earlier single-cell run from the artifact store. Selecting one shows its cell, its "
              "settings, its seed and its notices, and re-opens its figures.",
    "compare_list": "The saved runs a comparison draws. Pick one in the record picker above and press "
                    "Add selected; the picker holds one at a time, so the list is how several are chosen.",
    "compare_mode": "cells: each run's ratio curve on one axis, labelled by cell. repeats: several "
                    "runs of one cell, with the spread across them as a band. renormalise: one run's "
                    "ratio recomputed with the constant below, drawn against the original.",
    "prefactor": "The normalisation constant to recompute T_eff/T with. The ratio is linear in it, so "
                 "nothing is re-simulated.",
}


def _run_fdt_guarded(cfg, *, skip_sanity, confirm_production, writer, seed=None):
    """Translate a malformed cell into a readable message. An FDTModelError (a missing FDT parameter,
    or a user model with multiplicative/zero observable noise) is a Refusal and passes through
    UNWRAPPED: the worker hands it to BasePanel._on_error, which opens the yellow "Check your inputs"
    box for it -- wrapping it in a RuntimeError, as this used to, re-typed a refusal into a bug and
    bought it a traceback. The KeyError net is a defensive backstop for a malformed cell (it should
    no longer fire for HOPF/BP); that one is a bare KeyError nobody raised as a refusal, so it is
    still translated."""
    try:
        return run_fdt(cfg, skip_sanity=skip_sanity, confirm_production=confirm_production,
                       writer=writer, seed=seed)
    except KeyError as e:
        raise RuntimeError(
            f"The FDT pipeline needs the parameter {e}, which the selected {cfg.model} cell does not "
            f"define."
        ) from e


class FdtPanel(BasePanel):
    """Drives a single-cell FDT analysis (``FDT.fdt_pipeline.run_fdt``).

    Passes run_fdt's required ``skip_sanity``/``confirm_production`` booleans from the two checkboxes:
    the stage no longer prompts, so a front end -- this panel, or ``python -m core fdt`` -- must
    supply both.

    Persists (group "fdt"): model, cell picker, saved-run picker, and the campaign knobs. Restore
    order matters -- model FIRST, then the pickers, or the model's refresh() wipes the restored
    picker.

    The Seed box, the record name and the note are NOT persisted (E7, spec §5.5). The seed is how a
    run is made a deliberate repeat of an earlier one, so a remembered value would turn every later
    run into that repeat in silence and collapse the spread E8 measures; a remembered NAME would be
    refused by assert_name_free at the next launch's first click, for a name nobody typed; and a note
    describes one run. Nor are the comparison controls (spec §7.1): a remembered list would name
    records a later session may have deleted, and a remembered constant would renormalise by a number
    nobody typed this time.

    The two checkboxes are CONSENTS and are never persisted (V5): every launch opens at the
    construction defaults, sanity checks on and the production sweep after them -- the run
    ``python -m core fdt`` makes without --skip-sanity or --no-production. A remembered "skip" would
    silently drop the checks from every later session.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_controls()
        self.restore_settings(settings.settings())
        # The saved-run viewer, built AFTER the restore: it fills the line for the restored selection
        # and opens none of its figures (ruling F47) -- re-opening an old run's pictures on every start
        # is behaviour nobody asked for. Figures open only on a user's pick (record_view.RecordViewer).
        self._viewer = record_view.RecordViewer(self.record_picker, self.record_line,
                                                self.figure_stack, busy=lambda: self._busy)

    def _build_controls(self):
        box = QGroupBox("FDT analysis")
        form = make_form(box)

        self.model_combo = QComboBox()
        self.model_combo.addItems(VALID_MODELS)
        self.model_combo.setCurrentText("NADROWSKI")
        self.model_combo.currentTextChanged.connect(self._on_model_changed)

        self.cell_picker = ArtifactPicker(CELL_PATH / "nadrowski")
        self.n_freqs = IntField(60)
        self.ensemble_m = IntField(256)
        self.freqs_per_batch = IntField(1)
        self.f0 = FloatField(0.05)
        # Blank on purpose (IntField(None) is not a thing, so the text is cleared): blank means "draw
        # one and record it" (E7). Never restored -- see the class docstring.
        self.seed = IntField(0)
        self.seed.clear()
        self.record_name = QLineEdit()
        self.record_name.setPlaceholderText("name for this run's record (optional)…")
        self.record_note = QLineEdit()
        self.record_note.setPlaceholderText("a note to keep with it (optional)…")
        # Earlier runs of THIS analysis: filtered to its own study by the row predicate -- the kind
        # also holds sweeps and comparisons (spec §5.4) -- and to finished records by StorePicker
        # itself (T24).
        self.record_picker = StorePicker("fdt", row_filter=lambda s: s.study == "single")

        self.skip_sanity = QCheckBox("Skip sanity checks")
        self.confirm_production = QCheckBox("Proceed to the production sweep after sanity")
        self.confirm_production.setChecked(True)
        self.skip_sanity.toggled.connect(
            lambda on: self.confirm_production.setEnabled(not on))   # only consulted when sanity runs

        self.btn_run = QPushButton("Run FDT analysis")
        self.btn_run.setProperty("accent", True)          # primary CTA (Fluent accent)
        self.btn_run.clicked.connect(self._run)

        # Row labels come from core/gui/fields.py (P34), so a row and the fix sentence a refusal prints
        # for it cannot drift apart. "Record name", "Note" and "Saved run" stay literal: `name` and
        # `note` are sentence entries (label() raises KeyError for them) and the saved-run picker is
        # not a refusal field.
        add_help_row(form, gui_fields.label("model"), self.model_combo, HELP["model"])
        add_help_row(form, gui_fields.label("cell"), self.cell_picker, HELP["cell"])
        add_help_row(form, gui_fields.label("n_freqs"), self.n_freqs, HELP["n_freqs"])
        add_help_row(form, gui_fields.label("ensemble_m"), self.ensemble_m, HELP["ensemble_M"])
        add_help_row(form, gui_fields.label("freqs_per_batch"), self.freqs_per_batch,
                     HELP["freqs_per_batch"])
        add_help_row(form, gui_fields.label("f0"), self.f0, HELP["f0"])
        add_help_row(form, gui_fields.label("seed"), self.seed, HELP["seed"])
        add_help_row(form, "Record name", self.record_name, HELP["record_name"])
        add_help_row(form, "Note", self.record_note, HELP["record_note"])
        add_help_row(form, "Saved run", self.record_picker, HELP["record"])
        # What the selected run says, under its picker (spec §5.4). The viewer that fills it is built
        # in __init__, after restore_settings (F47).
        self.record_line = record_view.details_label()
        form.addRow("", self.record_line)
        form.addRow(with_badge(self.skip_sanity, HELP["skip_sanity"]))
        form.addRow(with_badge(self.confirm_production, HELP["confirm_production"]))
        form.addRow(self.btn_run)

        self.controls_layout.addWidget(box)
        self.controls_layout.addWidget(self._build_compare())

    def _build_compare(self):
        """The comparison controls (spec §7.1, E8): the first three modes, over this screen's own
        record picker. The list is what the single-selection picker appends to; the normalisation
        constant is read only by the renormalise mode, so its box is enabled only there."""
        box = QGroupBox("Compare saved runs")
        form = make_form(box)
        self.compare_list = CompareList(self.record_picker)
        # Blank, and live in the renormalise mode alone. Built before the mode combo that toggles it.
        self.renorm_prefactor = FloatField(None)
        self.renorm_prefactor.setEnabled(False)
        self.compare_mode = QComboBox()
        self.compare_mode.addItems(["cells", "repeats", "renormalise"])
        self.compare_mode.currentTextChanged.connect(
            lambda mode: self.renorm_prefactor.setEnabled(mode == "renormalise"))
        self.btn_compare = QPushButton("Compare saved runs")
        self.btn_compare.clicked.connect(self._compare)
        # The constant's row is labelled from core/gui/fields.py (P34), like every registered box on
        # this screen. "Runs to compare" and "Comparison" stay literal: neither answers a registered
        # key -- a refusal about the list is `compare_records`, a sentence entry naming it.
        add_help_row(form, "Runs to compare", self.compare_list, HELP["compare_list"])
        add_help_row(form, "Comparison", self.compare_mode, HELP["compare_mode"])
        add_help_row(form, gui_fields.label("prefactor"), self.renorm_prefactor, HELP["prefactor"])
        form.addRow(self.btn_compare)
        return box

    def _compare(self):
        """Dispatch one comparison. The panel checks nothing itself: the arity, the study and the
        unfinished-record refusals belong to the stage (one wording for both front ends), and a
        Refusal from the worker reaches the yellow box through BasePanel._on_error."""
        mode = self.compare_mode.currentText()
        options = {}
        if mode == "renormalise":
            # value_or_none, never value(): a blank box returns 0.0 from value(), and 0 is refused
            # with a sentence about a value nobody typed instead of "it is blank".
            options["prefactor"] = self.renorm_prefactor.value_or_none()
        self.dispatch(compare, mode, self.compare_list.ids(), provide_fig_sink=True,
                      on_result=self._on_comparison, **options)

    def _on_comparison(self, record) -> None:
        """The comparison record ``compare`` wrote, named on the pane as a run's record is
        (``_on_record``). The picker is not moved onto it: the picker lists single-cell runs, and a
        comparison is a record of its own study, which the Artifacts screen lists."""
        self.log_pane.append_line(
            f"Comparison record written: {record.name or '(unnamed)'} [{record.id}].")

    def _show_record(self, *, figures: bool = True) -> None:
        """Spec §5.4: the selected run's line and, with ``figures``, its figures (RecordViewer)."""
        self._viewer.show_record(figures=figures)

    def _on_run_finished(self) -> None:
        """A run has ended, whatever its outcome. The viewer closes the tabs it opened for an earlier
        record, so the stack holds this run's own figures alone -- every run of this analysis draws
        the same titles, and a mix of two runs' tabs cannot be read (fix round 1 of T27)."""
        self.log_pane.append_line("FDT run finished.")
        self._viewer.close_figures()

    def _on_model_changed(self, model: str):
        self.cell_picker.repoint(CELL_PATH / model.lower())
        # FDT supports the built-ins + additive-noise, no-forcing user models. Gate the CTA on
        # registry.fdt_support and show the reason (multiplicative/zero-noise or forced user models).
        ok, reason = registry.fdt_support(model)
        self.btn_run.setEnabled(ok)
        if not ok:
            self.log_pane.append_line(reason, "warning")

    def _run(self):
        cell = self.cell_picker.selected_path()
        if not cell:
            self.log_pane.append_line("Select a cell file first.", "warning")
            return
        model = self.model_combo.currentText()
        # No model check of its own. The Run button is gated on registry.fdt_support
        # (_on_model_changed), and a model that stops qualifying under a live selection -- a user
        # model re-saved with multiplicative noise -- is refused by cli.make_fdt_config with
        # fdt_support's own sentence as field "model", which the Refusal arm shows in the yellow box.
        # The "backstop" that stood here only logged that sentence and returned: a second, boxless path.

        # The other boxes use value(), not value_or_none(): a blank box reads as 0 and the BUILDER
        # refuses 0 by name (T10, spec §3.3), which is the one wording both front ends inherit. The
        # seed is the exception -- 0 is a legal seed, so a blank must stay blank and mean "draw one"
        # (E7), which only value_or_none() can tell apart from a typed 0.
        seed = self.seed.value_or_none()
        try:
            cfg = cli.make_fdt_config(
                model, registry.state_dep_drift(model), cell,
                n_freqs=self.n_freqs.value(), ensemble_M=self.ensemble_m.value(),
                freqs_per_batch=self.freqs_per_batch.value(), F0=self.f0.value(), seed=seed)
            # THE FRONT END CREATES, THE STAGE ENTERS (spec §1.2). create() mints the id and claims
            # the name -- assert_name_free runs here, before a single trajectory is integrated -- and
            # fills writer.dir WITHOUT creating it; run_fdt does `with writer:` on the worker thread,
            # where runs.current_run_log() is populated and log.txt can therefore be written. It is
            # inside this try because a taken name is a Refusal about an input on this screen, and
            # belongs in the same yellow box as a bad n_freqs.
            # The NOTE is judged first, because the store does not judge it (ArtifactStore.set_note):
            # every front end runs require_note -- one line, at most NOTE_MAX_CHARS -- before it
            # writes one, so this box cannot store a note the Artifacts screen would refuse to write
            # back. require_note trims it itself, and a blank box comes back "" (no note).
            note = require_note("note", self.record_note.text())
            writer = default_store().create("fdt", cfg, name=self.record_name.text().strip(),
                                            note=note)
        except Refusal as e:                         # a setting the user can change: the yellow box
            self._refusal(e)
            return
        except Exception as e:                       # noqa: BLE001 -- a bug in the builder: the red box
            self._on_error(e, traceback.format_exc())
            return

        # The facts the panel knows. The STAGE fills `settings` from its own private copy of cfg
        # before __enter__ writes the first manifest, and updates this dict in place -- replacing it
        # would drop the study and the seed (spec §2.2, §2.3).
        writer.body = {"study": "single", "settings": None, "seed": seed, "grid": None,
                       "points": None, "offgrid": None, "notices": [], "compared": None,
                       "complete": False, "results": None}
        # Explicit bools, never None -- see the module docstring. The watcher is pointed at the
        # record's own figures/ (spec §5.4); it globs ONE directory and does not recurse, which is
        # exactly that shape, and a directory that does not exist yet lists nothing.
        self.dispatch(_run_fdt_guarded, cfg, writer=writer, seed=seed,
                      watch_dir=writer.dir / "figures",
                      skip_sanity=self.skip_sanity.isChecked(),
                      confirm_production=self.confirm_production.isChecked(),
                      on_result=self._on_record,
                      on_finished=self._on_run_finished)

    def _on_record(self, record):
        """The ``LoadedFdt`` run_fdt returned (E1: the run now says what it wrote, where it used to
        return None). The picker is re-listed and moved onto it, so the run that just finished is the
        panel's current selection. The move is programmatic, so the viewer describes the record on
        the line and opens nothing: the run's watcher already put its figures up."""
        if record is None:
            return
        self.log_pane.append_line(
            f"FDT record written: {record.name or '(unnamed)'} [{record.id}].")
        self.record_picker.refresh()
        self.record_picker.restore_key(record.id)

    def save_settings(self, qs):
        qs.beginGroup("fdt")
        qs.setValue("model", self.model_combo.currentText())
        qs.setValue("cell", self.cell_picker.key())
        qs.setValue("record", self.record_picker.key())
        for name, fld in (("n_freqs", self.n_freqs), ("ensemble_m", self.ensemble_m),
                          ("freqs_per_batch", self.freqs_per_batch), ("f0", self.f0)):
            settings.save_field(qs, name, fld)
        # The two checkboxes are not written: they are consents (see the class docstring). Neither
        # are the seed, the record name and the note -- all three belong to ONE run (E7, §5.5).
        qs.endGroup()

    def restore_settings(self, qs):
        qs.beginGroup("fdt")
        # Model FIRST, explicit _on_model_changed, THEN the picker key -- ArtifactPicker.repoint owns
        # the rationale. Feed the combo's ACCEPTED text, not the raw saved string: a corrupt saved
        # model a non-editable combo rejects would otherwise point the picker at a nonexistent folder.
        self.model_combo.setCurrentText(settings.get_str(qs, "model", self.model_combo.currentText()))
        self._on_model_changed(self.model_combo.currentText())
        self.cell_picker.restore_key(settings.get_str(qs, "cell"))
        self.record_picker.restore_key(settings.get_str(qs, "record"))
        for name, fld in (("n_freqs", self.n_freqs), ("ensemble_m", self.ensemble_m),
                          ("freqs_per_batch", self.freqs_per_batch), ("f0", self.f0)):
            settings.restore_field(qs, name, fld)
        # The two checkboxes keep their construction defaults (_build_controls), and a key an older
        # build left in PRISM.ini is ignored: a consent is answered by the session that runs.
        qs.endGroup()
