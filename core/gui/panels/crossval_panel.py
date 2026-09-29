"""CROSSVAL mode: the FDT parameter-sweep study.

Two sweeps probe FDT restoration on the Nadrowski model (see cli.make_param_sweep_config):
  * S sweep  (T_a/T held at 1): vary S;      FDT is restored as S -> 0.
  * T sweep  (S held at 0):     vary T_a/T;  FDT is restored as T_a/T -> 1.

Mirrors cli.make_param_sweep_config with widgets, then runs the prompt-free
FDT.cross_validation.run_param_study_cli on a worker. Model is fixed to NADROWSKI.
"""
import traceback

from PySide6.QtWidgets import (QComboBox, QFormLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
                               QPushButton, QWidget)

from core import cli
from core.config import CELL_PATH
from core.refusals import Refusal, require_given, require_note
from core.FDT.cross_validation import run_param_study_cli
from core.FDT.compare import compare
from core.artifacts import default_store

from . import record_view
from .base_panel import BasePanel
from .. import settings
from ..widgets.artifact_picker import ArtifactPicker, StorePicker
from ..widgets.compare_list import CompareList
from ..widgets.help_badge import add_help_row
from ..widgets.labeled_inputs import FloatField, IntField, number_or_blank
from ..widgets.field_row import LabeledFieldRow
from ..widgets.forms import make_form
from .. import fields as gui_fields

_MODEL = "NADROWSKI"

HELP = {
    "cell": "A cell whose S and T_a/T set the far end of each sweep (the FDT-violating extreme).",
    "preset": "exploratory = quick/coarse; production = finer grids and more frequencies.",
    "s_grid": "S sweep (T_a/T fixed at 1): FDT is restored as S → 0. min / max / number of points.",
    "t_grid": "T_a/T sweep (S fixed at 0): FDT is restored as T_a/T → 1. min / max / number of points.",
    "n_freqs": "Number of (log-spaced) frequencies evaluated per sweep point.",
    "ensemble_M": "Ensemble size: independent trajectories averaged per frequency.",
    "freqs_per_batch": "How many frequencies to simulate per batch (memory/speed; results identical).",
    "f0": "Non-dimensional forcing amplitude used to probe χ(ω) at each sweep point.",
    "seed": "The seed this study uses. Leave it blank to draw one — whichever is used is recorded on "
            "BOTH of the study's records, and each operating point derives its own stream from it, so "
            "a point is reproducible from the seed and its index.",
    "record_name": "A base name for the two records this study writes: '-s' and '-temp' are appended, "
                   "one per swept parameter. Both names are claimed before anything is computed. "
                   "Leave it blank for two unnamed records.",
    "record_note": "Kept with both records and shown in the Artifacts browser.",
    "record": "An earlier sweep from the artifact store. Selecting one shows its cell, its settings, "
              "its seed and its notices, and re-opens its figures.",
    "compare_list": "The saved sweep records a comparison draws. Pick one in the 'Saved sweep' picker "
                    "above and press Add selected; the picker holds one at a time.",
    "slice_at": "The operating point to slice both sweeps at. Blank means the middle of the range the "
                "two of them share.",
    "compare_name": "A name for the record the comparison writes. A name already taken is refused. "
                    "Leave it blank for an unnamed record.",
    "compare_note": "Kept with the comparison's record and shown in the Artifacts browser.",
}


class _GridRow(LabeledFieldRow):
    """min / max / n_points for one sweep axis."""

    def __init__(self, lo: float, hi: float, points: int, parent=None):
        self.lo, self.hi, self.points = FloatField(lo), FloatField(hi), IntField(points)
        super().__init__((("min", self.lo), ("max", self.hi), ("n", self.points)), parent=parent)

    def spec_or_none(self) -> tuple:
        """``(min, max, n)``, each None when its box is blank.

        Every box through value_or_none(), never value(): 0 is a legal END of a sweep (the activity
        sweep starts at S = 0), so a blank read as 0 is not refused by any rule the builder could
        write -- a blank min under a positive max arrived as (0.0, 1.5, n) and passed "min below
        max", running a sweep nobody typed, and for the temperature grid one that starts below the
        bounds' own floor. A None end is refused by cli._check_grid as blank, naming the grid; a None
        count reaches it as "fewer than 2 points"."""
        return self.lo.value_or_none(), self.hi.value_or_none(), self.points.value_or_none()


class CrossValPanel(BasePanel):
    """Drives the FDT parameter-sweep study (``FDT.cross_validation.run_param_study_cli``).

    Model is fixed to NADROWSKI. Two sweeps -- S with T_a/T held at 1, and T_a/T with S held at 0 --
    each running from the FDT-restoring limit to the selected cell's own value.

    Persists (group "crossval"): the cell picker, the saved-sweep picker, the preset, the free knobs
    (freqs_per_batch, F0) and each grid's point COUNT. Deliberately NOT n_freqs and ensemble_M: the
    preset re-derives them. Deliberately NOT the grids' lo/hi: those are re-derived from the cell,
    so a value saved against a different cell would be a stale bound. Deliberately NOT the seed, the
    record name or the note either: one seed is recorded on both of the study's records, so a
    remembered one would make every later study a repeat of the last at every operating point, and a
    remembered name would be refused by assert_name_free at the next launch's first click. Nor are
    the comparison controls: a remembered list would name records a later session may have deleted,
    and the comparison's own name and note are the study's twins.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_controls()
        self._on_preset_changed(self.preset_combo.currentText())
        self._on_cell_changed()
        self.restore_settings(settings.settings())
        # The saved-sweep viewer, built AFTER the restore so a launch fills the line and opens no
        # figures -- FdtPanel.__init__ gives the reason.
        self._viewer = record_view.RecordViewer(self.record_picker, self.record_line,
                                                self.figure_stack, busy=lambda: self._busy)

    def _build_controls(self):
        box = QGroupBox("FDT parameter-sweep study")
        form = make_form(box)

        self.cell_picker = ArtifactPicker(CELL_PATH / _MODEL.lower())
        self.cell_picker.combo.currentIndexChanged.connect(self._on_cell_changed)

        self.preset_combo = QComboBox()
        self.preset_combo.addItems(list(cli.SWEEP_PRESETS))          # exploratory | production
        self.preset_combo.currentTextChanged.connect(self._on_preset_changed)

        self.cell_values = QLabel("—")
        # This label receives an UNBOUNDED str(e) from _on_cell_changed's broad except. Without wrap
        # it forced the whole controls column wider than any sensible width, which used to trigger
        # the permanent horizontal scrollbar on a panel that otherwise fit.
        self.cell_values.setWordWrap(True)
        self.s_grid = _GridRow(0.0, 1.0, 8)
        self.t_grid = _GridRow(1.0, 1.0, 8)
        self.n_freqs = IntField(30)
        self.ensemble_m = IntField(256)
        self.freqs_per_batch = IntField(1)
        self.f0 = FloatField(0.05)
        # ONE seed for the study, recorded on BOTH records; blank draws one. Never restored -- see
        # the class docstring.
        self.seed = IntField(0)
        self.seed.clear()
        self.record_name = QLineEdit()
        self.record_name.setPlaceholderText("base name for this study's two records (optional)…")
        self.record_note = QLineEdit()
        self.record_note.setPlaceholderText("a note to keep with both (optional)…")
        # Earlier sweeps: filtered to this screen's own study by the row predicate -- the kind also
        # holds single-cell runs and comparisons -- and to finished records by StorePicker itself.
        self.record_picker = StorePicker("fdt", row_filter=lambda s: s.study == "sweep")

        self.btn_run = QPushButton("Run sweep study")
        self.btn_run.setProperty("accent", True)          # primary CTA (Fluent accent)
        self.btn_run.clicked.connect(self._run)

        # Row labels come from core/gui/fields.py, so a row and the fix sentence a refusal prints
        # for it cannot drift apart. "Cell values", "Record name", "Note" and "Saved sweep" stay
        # literal: the first is an output, `name` and `note` are sentence entries (label() raises
        # KeyError for them) and the saved-sweep picker is not a refusal field.
        form.addRow(QLabel(f"Model is fixed to {_MODEL}."))
        add_help_row(form, gui_fields.label("cell"), self.cell_picker, HELP["cell"])
        form.addRow("Cell values", self.cell_values)          # an output display, not a configurable input
        add_help_row(form, gui_fields.label("preset"), self.preset_combo, HELP["preset"])
        add_help_row(form, gui_fields.label("s_grid"), self.s_grid, HELP["s_grid"])
        add_help_row(form, gui_fields.label("t_grid"), self.t_grid, HELP["t_grid"])
        add_help_row(form, gui_fields.label("n_freqs"), self.n_freqs, HELP["n_freqs"])
        add_help_row(form, gui_fields.label("ensemble_m"), self.ensemble_m, HELP["ensemble_M"])
        add_help_row(form, gui_fields.label("freqs_per_batch"), self.freqs_per_batch,
                     HELP["freqs_per_batch"])
        add_help_row(form, gui_fields.label("f0"), self.f0, HELP["f0"])
        add_help_row(form, gui_fields.label("seed"), self.seed, HELP["seed"])
        add_help_row(form, "Record name", self.record_name, HELP["record_name"])
        add_help_row(form, "Note", self.record_note, HELP["record_note"])
        add_help_row(form, "Saved sweep", self.record_picker, HELP["record"])
        # What the selected sweep says, under its picker. The viewer that fills it is built in
        # __init__, after restore_settings.
        self.record_line = record_view.details_label()
        form.addRow("", self.record_line)
        form.addRow(self.btn_run)

        self.controls_layout.addWidget(box)
        self.controls_layout.addWidget(self._build_compare())

    def _build_compare(self):
        """The fourth comparison mode: two sweep records together, and a slice of both at one
        operating point. A blank slice point means the middle of the range the two sweeps share,
        which is what the stage does with no `at` at all."""
        box = QGroupBox("Compare saved sweeps")
        form = make_form(box)
        self.compare_list = CompareList(self.record_picker)
        self.slice_at = FloatField(None)
        # The comparison's OWN name and note, never the study's Record name box. Never persisted
        # (the class docstring).
        self.compare_name = QLineEdit()
        self.compare_name.setPlaceholderText("name for the comparison's record (optional)…")
        self.compare_note = QLineEdit()
        self.compare_note.setPlaceholderText("a note to keep with it (optional)…")
        self.btn_compare = QPushButton("Compare saved sweeps")
        self.btn_compare.clicked.connect(self._compare)
        # The slice point's row is labelled from core/gui/fields.py; "Sweeps to compare" stays
        # literal, because a refusal about the list is `compare_records`, a sentence entry naming it,
        # and so do "Comparison name" and "Comparison note" (`name` and `note` are sentence entries).
        add_help_row(form, "Sweeps to compare", self.compare_list, HELP["compare_list"])
        add_help_row(form, gui_fields.label("slice_at"), self.slice_at, HELP["slice_at"])
        add_help_row(form, "Comparison name", self.compare_name, HELP["compare_name"])
        add_help_row(form, "Comparison note", self.compare_note, HELP["compare_note"])
        form.addRow(self.btn_compare)
        return box

    def _compare(self):
        """Dispatch the sweeps comparison, under the name and the note typed for it. The NOTE is
        judged at the click (every front end runs require_note; the store does not), and so is a
        half-typed slice point. Everything else is the stage's: a slice point outside the range
        the sweeps share, two sweeps of different parameters, an unfinished record and a taken name
        are its refusals, and reach the yellow box through BasePanel._on_error."""
        options = {}
        try:
            note = require_note("note", self.compare_note.text())
            # A blank slice point means the middle of the shared range; half-typed text ('-') is not
            # blank, and is refused rather than sliced at the middle in silence.
            at = number_or_blank(self.slice_at, "slice_at")
        except Refusal as e:
            self._refusal(e)
            return
        if at is not None:
            options["at"] = at
        self.dispatch(compare, "sweeps", self.compare_list.ids(), provide_fig_sink=True,
                      on_result=self._on_comparison, name=self.compare_name.text().strip(),
                      note=note, **options)

    def _on_comparison(self, record) -> None:
        """The comparison record ``compare`` wrote, named on the pane as the study's records are
        (``_on_result``). The picker is not moved onto it: it lists sweeps, and a comparison is a
        record of its own study, which the Artifacts screen lists."""
        self.log_pane.append_line(
            f"Comparison record written: {record.name or '(unnamed)'} [{record.id}].")

    def _show_record(self, *, figures: bool = True) -> None:
        """The selected sweep's line and, with ``figures``, its figures (RecordViewer)."""
        self._viewer.show_record(figures=figures)

    # ── prefill from the cell file: the values cli.make_param_sweep_config then consumes ─────────
    def _on_cell_changed(self):
        cell = self.cell_picker.selected_path()
        if not cell:
            return
        try:
            _i, params, _r, _f, _u, _si, _s = cli.parse_cell(cell, model=_MODEL)
        except Exception as e:                       # noqa: BLE001
            # Deliberately broad. __init__ calls this, so ANY exception here escapes CrossValPanel()
            # -> MainWindow() -> build_app() and the whole GUI fails to launch -- and app.py installs
            # its excepthook only AFTER MainWindow() is built, so nothing would even show it. A cell
            # with no sibling Bounds/<model>/<name>.txt is enough to trigger it: parse_cell then takes
            # the legacy branch and raises a plain ValueError, not UnitParseError. A bad cell file must
            # degrade this one label, never brick the app.
            self.cell_values.setText(f"(could not read cell: {e})")
            return
        cell_s = float(params["s"][0])
        cell_temp = float(params["temp"][0])
        self.cell_values.setText(f"S = {cell_s:.4f},  T_a/T = {cell_temp:.4f}")
        # Each sweep spans the FDT-restoring limit (S = 0, T_a/T = 1) and the cell's own value, filled
        # in ASCENDING order: the bounds allow T_a/T anywhere in (0.05, 10), so "limit first" made the
        # untouched default descending for any cell below 1, and the builder refuses a grid whose
        # min is not below its max. The sweep covers the same points either way.
        for row, limit, value in ((self.s_grid, 0.0, cell_s), (self.t_grid, 1.0, cell_temp)):
            row.lo.setText(f"{min(limit, value):g}")
            row.hi.setText(f"{max(limit, value):g}")

    def _on_preset_changed(self, name: str):
        preset = cli.SWEEP_PRESETS[name]
        self.n_freqs.setText(str(preset["n_freqs"]))
        self.ensemble_m.setText(str(preset["ensemble_M"]))
        self.s_grid.points.setText(str(preset["points"]))
        self.t_grid.points.setText(str(preset["points"]))

    def _run(self):
        cell = self.cell_picker.selected_path()
        if not cell:
            self.log_pane.append_line("Select a cell file first.", "warning")
            return
        preset_name = self.preset_combo.currentText()
        preset = dict(cli.SWEEP_PRESETS[preset_name])
        base = self.record_name.text().strip()
        try:
            # The four knob boxes are refused HERE when blank: the builder reads None as "use the
            # preset", so a blank passed on would run the preset's value in silence, and one read
            # through value() was refused as "got 0", a value nobody typed. The grid rows and the
            # seed arrive as None when blank too -- 0 is a legal sweep END and a legal seed -- a
            # grid end then refused as blank (spec_or_none), and a blank seed meaning "draw one",
            # which is why half-typed seed text is refused rather than read as blank
            # (number_or_blank). The seed is read ONCE and the same value goes to the builder, the
            # run and both first bodies.
            knobs = {kw: require_given(key, box.value_or_none())
                     for kw, key, box in (("n_freqs", "n_freqs", self.n_freqs),
                                          ("ensemble_M", "ensemble_m", self.ensemble_m),
                                          ("freqs_per_batch", "freqs_per_batch", self.freqs_per_batch),
                                          ("F0", "f0", self.f0))}
            seed = number_or_blank(self.seed, "seed")
            cfg, s_grid, temp_grid = cli.make_param_sweep_config(
                cell, preset=preset, preset_name=preset_name,
                s_spec=self.s_grid.spec_or_none(), t_spec=self.t_grid.spec_or_none(),
                seed=seed, **knobs)
            # The NOTE is judged before either create, because the store does not judge it
            # (ArtifactStore.set_note): every front end runs require_note -- one line, at most
            # NOTE_MAX_CHARS -- so this box cannot put a note on both records that the Artifacts screen
            # would refuse to write back. It trims the text itself; a blank box comes back "".
            note = require_note("note", self.record_note.text())
            # ONE RECORD PER SWEPT PARAMETER, keyed by the names run_fdt_param_sweep already uses as
            # sweep_param. Two records cannot share one name -- and assert_name_free reads only the
            # disk, so two creates of one name would both pass -- hence the distinct suffixes. Both
            # names are claimed here, before anything is spent, inside this try so a taken one
            # reaches the yellow box; ONE store object mints both ids, so they differ; and neither
            # directory exists until the stage's __enter__ on the worker thread.
            store = default_store()
            writers = {key: store.create("fdt", cfg, name=f"{base}-{key}" if base else "", note=note)
                       for key in ("s", "temp")}
        except Refusal as e:                         # a setting the user can change: the yellow box
            self._refusal(e)
            return
        except Exception as e:                       # noqa: BLE001 -- a bug in the builder: the red box
            self._on_error(e, traceback.format_exc())
            return

        for w in writers.values():
            # The facts the panel knows. The STAGE fills `settings`, `points`, `offgrid` and
            # `results` and updates this dict in place -- replacing it would drop the study and the
            # seed before the first manifest is written. `points.param` is the stage's too: the
            # swept parameter is recorded once, there.
            w.body = {"study": "sweep", "settings": None, "seed": seed, "grid": None,
                      "points": None, "offgrid": None, "notices": [], "compared": None,
                      "complete": False, "results": None}
        # run_param_study_cli returns the records it FINISHED, not the figures -- each sweep plots
        # itself into its own record's figures/ when it ends (the S sweep's at the study's midpoint),
        # and the plots arrive through ONE WATCHER PER RECORD: the watcher globs one directory
        # and does not recurse, so both folders are handed over. A folder the T sweep never reached
        # simply lists nothing.
        # on_finished: whatever the outcome, the viewer closes the tabs it opened for an earlier sweep,
        # so the stack holds this study's own figures alone -- FdtPanel._on_run_finished.
        self.dispatch(run_param_study_cli, cfg, s_grid=s_grid, t_grid=temp_grid,
                      writers=writers, seed=seed,
                      watch_dir=[writers["s"].dir / "figures", writers["temp"].dir / "figures"],
                      on_result=self._on_result, on_finished=self._viewer.close_figures)

    def _on_result(self, records):
        """The ``LoadedFdt`` records the study FINISHED, S first (it used to return two loose HDF5
        paths). A sweep that measured nothing is left out of the list -- its unfinished record stays
        on disk and the study has already logged it at error -- and a None, were one ever handed
        back, is dropped rather than named. The picker is re-listed and moved onto the
        LAST record, so the panel's selection is the one whose figures finished the run; the move is
        programmatic, so the viewer describes it and opens nothing (FdtPanel._on_record)."""
        records = [r for r in (records or []) if r is not None]   # one sweep may have refused
        if not records:
            return
        for record in records:
            self.log_pane.append_line(
                f"Sweep record written: {record.name or '(unnamed)'} [{record.id}].")
        self.record_picker.refresh()
        self.record_picker.restore_key(records[-1].id)

    def save_settings(self, qs):
        qs.beginGroup("crossval")
        qs.setValue("preset", self.preset_combo.currentText())
        qs.setValue("cell", self.cell_picker.key())
        qs.setValue("record", self.record_picker.key())
        # The seed, the record name and the note belong to ONE study and are never written.
        settings.save_field(qs, "f0", self.f0)
        settings.save_field(qs, "freqs_per_batch", self.freqs_per_batch)
        settings.save_field(qs, "s_points", self.s_grid.points)
        settings.save_field(qs, "t_points", self.t_grid.points)
        qs.endGroup()

    def restore_settings(self, qs):
        qs.beginGroup("crossval")
        # Order: preset first (it overwrites n_freqs/ensemble_M and the grid `points`), then cell (its
        # currentIndexChanged fires _on_cell_changed, which sets the grid lo/hi from the cell file).
        self.preset_combo.setCurrentText(settings.get_str(qs, "preset", self.preset_combo.currentText()))
        self.cell_picker.restore_key(settings.get_str(qs, "cell"))
        self.record_picker.restore_key(settings.get_str(qs, "record"))
        # Restore ONLY the freely-set knobs -- and after the cell, so a saved `points` survives. Do NOT
        # restore the grid lo/hi: those are re-derived from the cell (cli.make_param_sweep_config's inputs), and a saved
        # value from a DIFFERENT cell would be a stale, wrong bound.
        settings.restore_field(qs, "f0", self.f0)
        settings.restore_field(qs, "freqs_per_batch", self.freqs_per_batch)
        settings.restore_field(qs, "s_points", self.s_grid.points)
        settings.restore_field(qs, "t_points", self.t_grid.points)
        qs.endGroup()
