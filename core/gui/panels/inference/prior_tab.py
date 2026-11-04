"""Step 2, Prior: the prior a training draws its parameters from, loaded from a saved record or built anew, with the
bounds file whose limits it is cut from.

The step holds a configuration and its prior, always together: a configuration is built from the draft step 1 holds and
a bounds file, and a prior belongs to the box that file declares. Neither is held until a run returns its prior. The
page builds the configuration a press would hand the run, the candidate, as it reads its facts; a load or a build hands
that very object to the run, and only the run's prior, when it comes back, puts both in use, with what depends on the
prior in use let go then and not before. A run refused, failed, stopped or withdrawn changes nothing that is held.

Two views, each with its one main button: "Use a saved prior" lists the saved priors and loads one; "Build a new
prior" names the record and holds the seven settings the build reads, and the limits typed by hand. The bounds file
stands in both. The two lists keep what is selected while they are read again, and the bounds file and the prior that
came into use last are put back as choices at the next launch.

Nothing is read while a run is live: a prior that comes back while its run is live is put in use, and the lists, the
record's facts and the candidate are read again once the run has ended.
"""
import math
from datetime import datetime
from pathlib import Path, PurePath, PurePosixPath

from PySide6.QtWidgets import QCheckBox, QPushButton, QWidget

from core import config, orchestrator
from core.artifacts import default_store
from core.artifacts.provenance import file_ref
from core.config import BOUNDS_PATH
from core.Helpers import file_manager
from core.refusals import Refusal, default_clause, require_at_least, require_positive
from core.SBI import training_checkpoint

from ... import record_words, settings, style, texts
from ...reported import reported
from ...remembered import BOUNDS_KEY, BOUNDS_MODEL_KEY, MODEL_KEY, PRIOR_KEY
from ...run import history
from ...run.controller import controller
from ...run.model import RunSpec
from ...step_facts import (LOAD_WORDS, NO_BOUNDS_BLOCKS, NO_BOUNDS_TEXT, NO_PRIOR_BLOCKS, NO_PRIOR_TEXT,
                           NONE_SAVED_BLOCKS, NONE_SAVED_TEXT, PRIOR_BLOCKS,
                           NOT_HELD, NOTHING_HELD_TEXT, TYPED_BY_HAND, Fact, SelectionRow, bounds_words, build_words,
                           held_value, prior_facts, prior_in_use, saved_prior_words)
from ...widgets.artifact_picker import ArtifactPicker, StorePicker
from ...widgets.ask import ask
from ...widgets.form import NameAndNote, TickedField
from ...widgets.labeled_inputs import FloatField, IntField
from ...widgets.param_grid import BoundsGrid, two_limits, limit_problems
from .help_text import HELP
from .step_page import StepPage

TITLE = "Prior"
SENTENCE = ("The prior is the range of parameter values training draws from: the bounds file's limits, cut down to the "
            "values for which the model is stable.")
TO_BUILD = "Build a new prior instead…"
TO_SAVED = "Use a saved prior instead…"
CHOOSE_BOUNDS = "Choose a bounds file"
CHOOSE_PRIOR = "Choose a saved prior"
TYPED_LABEL = "Edit the limits by hand"
TYPED_NOTE = "Typed limits are not saved as a file. Records made from them name no bounds file."
TYPED_BLOCKS = "the limits typed cannot be used"
FILE_UNUSABLE = "The bounds file cannot be used. {}"
NO_PARAMETERS = "It names no parameter of the model."
SETTING_BLOCKS = "a setting cannot be used"
NEW_PRIOR = "The new prior"
GONE_TEXT = "The bounds file used last time, {}, is no longer in the folder."
#: How the bounds file in use is named when the file has been written since its configuration was built on it.
EDITED_IN_USE = "{}, as it was before it was edited"
GO_TO_TRAINING = "Go to Training"
KEPT = "kept from last time"
UNITS = "model time units"
SAVED_WITHOUT_NAME = "It was saved without its name: '{}' was taken while the run was going."
FORCE_SCALE_TEXT = ("This bounds file lists f_scale, but a posterior trained on it reads a spontaneous recording only, "
                    "where no drive acts: f_scale cannot change what is observed, so its estimate will be the prior's. "
                    "Choose a bounds file with neither a forcing section nor f_scale for a spontaneous recording.")
KEEP_POSTERIOR = "Keep the posterior"
LOAD_TITLE, LOAD_RELEASE = "Load this prior?", "Load and release the posterior"
BUILD_TITLE, BUILD_RELEASE = "Build a new prior?", "Build and release the posterior"
RELEASE_TEXT = ("This session holds {}. {} to the prior in use, so another prior releases {} once it {}. {} in Records "
                "and can be loaded again.")

# THE SEVEN SETTINGS OF A BUILD, in the order the stage checks them: the field key, which is also build_prior's
# keyword, and the rule's floor, an int for require_at_least, None for require_positive. ONE table, read by the line
# under each field as it is typed and at the press, so the two can never name different limits.
_KNOB_RULES = (
    ("num_iterations", 1),
    ("sweep_batch", 0),                      # a TYPED 0 = follow the hardware batch, as "Automatic" does
    ("max_sets", 1),
    ("walk_step", None),
    ("stability_units", None),
    ("min_cluster_size", 2),
    ("min_samples", 1),
)
#: The rows only a build reads: the seven settings and the limits typed by hand.
_BUILD_ROWS = ("bounds_typed", *(key for key, _ in _KNOB_RULES))

# Where the window shows this step's runs: Parameter inference, its step Prior.
_PLACE = ("inference", "prior")


def _check_knob(key: str, value, minimum):
    """Run one setting's rule: require_at_least with its floor, or require_positive when the floor is None. Returns the
    value; raises Refusal(field=key) on a blank or an out-of-rule value."""
    return require_positive(key, value) if minimum is None else require_at_least(key, value, minimum)


def _rule_words(minimum) -> str:
    """The rule as the line under a field says it: the words the refusal's own message uses after "must be"."""
    return "greater than 0" if minimum is None else f"at least {minimum}"


def bounds_in_use(cfg) -> "str | None":
    """The bounds file a configuration was built on, as the list names it: its path under the model's own folder of
    bounds files; "" for limits typed by hand; None for no configuration."""
    if cfg is None:
        return None
    path = (getattr(cfg, "sources", None) or {}).get("bounds")
    if not path:
        return ""
    folder = Path(BOUNDS_PATH) / str(getattr(cfg, "model", "")).lower()
    try:
        return str(PurePath(Path(path).resolve().relative_to(folder.resolve())))
    except (ValueError, OSError):
        return str(path)


def _named(key: str) -> str:
    """A bounds file's key as a person reads it: with forward slashes."""
    return key.replace("\\", "/")


def recorded_file(bounds) -> "str | None":
    """The bounds file a record's inputs name, as the list names it: its path under its model's folder of bounds files;
    None for limits typed by hand."""
    path = bounds.get("path") if isinstance(bounds, dict) else None
    if not path:
        return None
    parts = PurePosixPath(str(path).replace("\\", "/")).parts
    return "/".join(parts[2:]) if len(parts) > 2 and parts[0] == config.BOUNDS_PATH.name else str(path)


def _limits(cfg) -> tuple:
    """The parameters of a configuration and their limits, in its order, as plain values to compare."""
    out = []
    for section in ("params_dict", "rescale_params", "force_params_dict"):
        for name, (_value, (lo, hi)) in (getattr(cfg, section, None) or {}).items():
            out.append((section, name, float(lo), float(hi)))
    return tuple(out)


def _close(a: float, b: float) -> bool:
    """Equal as the engine's own check of a prior's box judges two limits."""
    return abs(a - b) <= 1e-8 + 1e-5 * abs(b)


def _same_box(gmm, cfg) -> bool:
    """Whether a saved prior's parameters and their limits are a configuration's: the model's parameters, in their
    order, and the box over them."""
    try:
        keys = list(gmm["param_keys"])
        lows, highs = gmm["box"]["nd_lows"], gmm["box"]["nd_highs"]
        wanted = list(cfg.params_dict.values())
        return (keys == list(cfg.params_dict) and len(lows) == len(highs) == len(wanted)
                and all(_close(float(lo), float(w[1][0])) and _close(float(hi), float(w[1][1]))
                        for lo, hi, w in zip(lows, highs, wanted)))
    except Exception:                        # noqa: BLE001 -- a record no comparison can be made with
        return False


def _shown(value) -> str:
    return f"{value:g}" if isinstance(value, float) else str(value)


def _made(created) -> str:
    """When a record was made, as the window writes a moment: in local time; the record's own words when they are no
    moment."""
    try:
        return texts.date_text(datetime.fromisoformat(created).astimezone())
    except (TypeError, ValueError, OverflowError, OSError):
        return str(created)


class PriorPanel(StepPage):
    """Step 2. ``bounds_picker`` and ``prior_picker`` list the bounds files of the model applied and the saved priors;
    ``typed_tick`` and ``bounds_grid`` are the limits typed by hand; ``view`` is "saved" or "build"."""

    def __init__(self, screen, parent=None):
        super().__init__(screen, TITLE, SENTENCE, step="prior", parent=parent)
        self.view = "build"
        self._view_chosen = False                # the view is chosen as the page first comes into sight
        self._candidate, self._candidate_key, self._candidate_error = None, None, ""
        self._read: dict = {}                    # what the last reading found
        self._took = None                        # the run history's last build, as the last reading found it
        self._dispatched = None                  # the candidate handed to the run in hand
        self._dispatched_key = None              # what that candidate was built from: a file by when it was written
        self._held_at = (None, None)             # the configuration in use and when its bounds file was written then
        self._made = None                        # the prior a run of this page returned, until the run has ended
        self._bounds_pick = ("", "")             # the model and the bounds file that came into use last
        self._kept_pick = None                   # the same, as the settings file gave it, until a model is applied
        self._grid_from = None                   # the bounds file the limits typed by hand were loaded from
        self._lists_wait = False                 # the reading after the typing stops reads neither list
        self.switch_button = QPushButton()
        self.switch_button.clicked.connect(self._switch)
        self.head.add_action(self.switch_button)
        self.bounds_picker = ArtifactPicker(BOUNDS_PATH / "nadrowski", parent=self, keeps_selection=True,
                                            hint=CHOOSE_BOUNDS)
        self.prior_picker = StorePicker("prior", parent=self, keeps_selection=True, hint=CHOOSE_PRIOR)
        self.typed_tick = QCheckBox()
        self.bounds_grid = BoundsGrid(scrolls=False)
        form = self.form
        # Heard before their rows hear them: the limits typed by hand stand, from the file selected, before the page
        # judges them.
        self.bounds_picker.combo.currentIndexChanged.connect(self._bounds_chosen)
        self.typed_tick.toggled.connect(self._typed_toggled)
        form.add("bounds", self.bounds_picker.combo, help_text=HELP["bounds_source"])
        self.gone_line = form.add_sentence("", under="bounds")
        self.gone_line.setVisible(False)
        self.file_line = form.add_sentence("", under="bounds")
        self.file_line.setVisible(False)
        form.add("prior", self.prior_picker.combo, help_text=HELP["prior"])
        self.prior_line = form.add_sentence("", under="prior")
        self.prior_line.setVisible(False)
        typed = form.add("bounds_typed", self.typed_tick, usual=False, folded=True, label=TYPED_LABEL)
        typed.add_under(self.bounds_grid)
        typed.hear(self.bounds_grid)
        self.typed_note = form.add_sentence(TYPED_NOTE, under="bounds_typed")
        form.add("num_iterations", IntField(config.PRIOR_SWEEP_ITERATIONS, grouped=True),
                 usual=config.PRIOR_SWEEP_ITERATIONS, folded=True, help_text=HELP["sweep_iters"])
        form.add("sweep_batch", TickedField(IntField(config.PRIOR_SWEEP_BATCH, grouped=True), "Automatic", 0),
                 usual=0, folded=True, help_text=HELP["sweep_batch"])
        form.add("max_sets", IntField(config.PRIOR_SWEEP_MAX_SETS, grouped=True), usual=config.PRIOR_SWEEP_MAX_SETS,
                 folded=True, help_text=HELP["sweep_max_sets"])
        form.add("walk_step", FloatField(config.PRIOR_SWEEP_STEP), usual=config.PRIOR_SWEEP_STEP, folded=True,
                 help_text=HELP["sweep_step"])
        form.add("stability_units", FloatField(config.STABILITY_SWEEP_ND_UNITS), UNITS,
                 usual=float(config.STABILITY_SWEEP_ND_UNITS), folded=True, help_text=HELP["sweep_units"])
        form.add("min_cluster_size", IntField(config.PRIOR_CLUSTER_MIN_SIZE, grouped=True),
                 usual=config.PRIOR_CLUSTER_MIN_SIZE, folded=True, help_text=HELP["cluster_size"])
        form.add("min_samples", IntField(config.PRIOR_CLUSTER_MIN_SAMPLES, grouped=True),
                 usual=config.PRIOR_CLUSTER_MIN_SAMPLES, folded=True, help_text=HELP["cluster_samples"])
        form.set_side_heading(NEW_PRIOR)
        self._names = NameAndNote("prior", required=True)
        self.use_names(self._names)
        self.restore_settings(settings.settings())
        self._apply_view("build")
        self.refresh_local_gates()

    # ── what the step holds ──
    def _draft(self):
        return getattr(getattr(self._screen, "session", None), "draft", None)

    def _held(self) -> tuple:
        """The configuration and the prior the step holds, or ``(None, None)``: the one is held only with the other."""
        session = getattr(self._screen, "session", None)
        prior = getattr(session, "inf_prior", None)
        return (getattr(session, "cfg", None), prior) if prior is not None else (None, None)

    def _typed(self) -> bool:
        """Whether the build reads the limits typed by hand: a choice of the build alone."""
        return self.view == "build" and self.typed_tick.isChecked()

    def _bounds_chosen_value(self) -> tuple:
        """What the bounds row shows now: the file selected, with when it was written as the last reading found it, or
        the limits typed by hand. The moment is compared only with a moment the step holds."""
        if self._typed():
            return "typed", self._grid_limits()
        key, held = self.bounds_picker.key(), self._bounds_held_value()
        if held is NOT_HELD or held[0] != "file" or held[2] is None:
            return "file", key, None
        built = self._candidate_key
        written = built[3] if built and built[0] == "file" and built[2] == self.bounds_picker.selected_path() else None
        return "file", key, written if written is not None else held[2]

    def _bounds_held_value(self):
        """What the step holds of the bounds row: the file its configuration was built on, with when the file was
        written as that configuration was built when the page built it, else None; or the limits typed by hand;
        ``NOT_HELD`` while it holds no prior."""
        cfg, prior = self._held()
        if prior is None or held_value(cfg, "sources") is NOT_HELD:
            return NOT_HELD
        try:
            key = bounds_in_use(cfg)
            if key:
                return "file", key, self._held_at[1] if self._held_at[0] is cfg else None
            return NOT_HELD if held_value(cfg, "params_dict") is NOT_HELD else ("typed", _limits(cfg))
        except Exception:                    # noqa: BLE001 -- what a session holds may be anything
            return NOT_HELD

    def _grid_limits(self) -> tuple:
        """The limits typed by hand, as plain values to compare; None where a field reads as no number."""
        try:
            params, rescale, forcing = self.bounds_grid.to_dicts()
        except Exception:                    # noqa: BLE001 -- a grid with a field that reads as nothing
            return ()
        return tuple((section, name, lo, hi) for section, held in (("params_dict", params), ("rescale_params", rescale),
                                                                    ("force_params_dict", forcing))
                     for name, (_value, (lo, hi)) in held.items())

    def _bounds_words(self, value) -> str:
        """The bounds held, as the line under the row names them: the file, said as it was before it was edited when
        the file selected is the same and was written since."""
        if value[0] != "file":
            return TYPED_BY_HAND
        chosen = self._bounds_chosen_value()
        edited = chosen[0] == "file" and chosen[1] == value[1] and chosen[2] != value[2]
        return EDITED_IN_USE.format(_named(value[1])) if edited else _named(value[1])

    def _prior_words(self, value) -> str:
        found = self.prior_picker.combo.findData(value)
        if found >= 0:
            return self.prior_picker.combo.itemText(found)
        return record_words.label_of(self._held()[1]) or str(value)

    # ── what a page answers ──
    def main_words(self) -> str:
        return LOAD_WORDS if self.view == "saved" else build_words(self._took)

    def nothing_to_apply(self) -> bool:
        """The saved view holds the prior selected, built on the file selected: a load would change nothing."""
        return (self.view == "saved" and self._held()[1] is not None
                and not any(row.changed() for row in self.selection_rows()))

    def selection_rows(self) -> tuple:
        """The bounds file, or the limits typed by hand, said under the row that chooses them, the list or the tick; and
        in the saved view the prior selected."""
        rows = [SelectionRow(self._candidate_row(), self._bounds_held_value, self._bounds_chosen_value,
                             self._bounds_words)]
        if self.view == "saved":
            prior = self._held()[1]
            rows.append(SelectionRow("prior", lambda: held_value(prior, "id"), self.prior_picker.key,
                                     self._prior_words))
        return tuple(rows)

    def plan_fields(self) -> dict:
        return {key: self.form.rows[key] for key in ("bounds", *(key for key, _ in _KNOB_RULES))}

    def in_use_words(self) -> tuple:
        cfg, prior = self._held()
        if prior is None:
            return "", NOTHING_HELD_TEXT
        manifest = getattr(prior, "manifest", None)
        inputs, made_for = getattr(manifest, "inputs", None), getattr(manifest, "config", None)
        built_on = recorded_file(inputs.get("bounds") if isinstance(inputs, dict) else None)
        keys = made_for.get("param_keys") if isinstance(made_for, dict) else None
        if not isinstance(keys, (list, tuple)):
            keys = [*(getattr(cfg, "params_dict", None) or {}), *(getattr(cfg, "rescale_params", None) or {})]
        return prior_in_use(record_words.label_of(prior), built_on, len(keys))

    def problems(self) -> dict:
        """Every rule of the build the form breaks now, by field key in the order a press reads them: the line under the
        field, built from its row's label, and what the press says, the same with the value it found and the usual one.
        A load reads none of the seven settings, so the saved view breaks none of them."""
        found = {}
        if self.view != "build":
            return found
        rows = self.form.rows
        for key, minimum in _KNOB_RULES:
            row = rows[key]
            said = row.reading()
            if said is not None:
                found[key] = (said, f"{said[:-1]}{default_clause(key)}." if said.endswith(" is blank.") else said)
                continue
            value = row.value()
            if value is None:
                continue
            if isinstance(value, float) and not math.isfinite(value):
                line = f"{row.label_text()} must be a finite number."
            else:
                try:
                    _check_knob(key, value, minimum)
                    continue
                except Refusal:
                    line = f"{row.label_text()} must be {_rule_words(minimum)}."
            found[key] = (line, f"{line[:-1]}; got {_shown(value)}{default_clause(key)}.")
        if self._typed():
            trouble = self.bounds_grid.problems()
            if trouble:
                found["bounds_typed"] = (trouble[0], trouble[0])
        return found

    def facts_from_form(self) -> tuple:
        """The facts of the step as the form, the session and the last reading stand: the rules under their fields; the
        candidate's trouble beside the list; the bounds file and the saved prior, as the last reading read them; what a
        build costs and takes; the configuration in use's attention lines; the name and the note."""
        found = self.problems()
        for key, row in self.form.rows.items():
            if row.widget is None or key in ("name", "note"):
                continue
            if key in found:
                row.set_line("err", found[key][0])
            elif key == self._candidate_row():
                row.set_line("err" if self._candidate_error else "", self._candidate_error)
            else:
                row.set_line("", "")
            row.set_origin(KEPT if row.kept else "")
        read, build = self._read, self.view == "build"
        field_facts = [Fact("err", line, TYPED_BLOCKS if key == "bounds_typed" else SETTING_BLOCKS)
                       for key, (line, _said) in found.items()]
        names = [] if not build or self.names is None else [fact for fact in (self.names.name_fact(),
                                                                                self.names.note_fact()) if fact]
        bounds_error, bounds_blocks = self._bounds_trouble()
        prior_error, prior_blocks = "", PRIOR_BLOCKS
        if not build and self.prior_picker.key():
            prior_error = read.get("prior_error", "")
        elif not build:
            prior_error, prior_blocks = ((NO_PRIOR_TEXT, NO_PRIOR_BLOCKS) if self.prior_picker.has_entries()
                                         else (NONE_SAVED_TEXT, NONE_SAVED_BLOCKS))
        candidate = self._candidate
        batch = self.form.rows["sweep_batch"].value() or getattr(getattr(candidate, "hw", None), "batch_size", None)
        values = {key: self.form.rows[key].value() for key in ("num_iterations", "max_sets", "stability_units")}
        steps = None
        if candidate is not None and build and isinstance(values["stability_units"], (int, float)):
            try:
                steps = int(values["stability_units"] / candidate.dt_nd_min)
            except Exception:                # noqa: BLE001 -- a configuration that cannot say its step
                steps = None
        cost_ok = build and not found.keys() & {"num_iterations", "sweep_batch", "max_sets", "stability_units"}
        return prior_facts(
            view=self.view, needs=self._draft() is None, bounds_error=bounds_error, bounds_blocks=bounds_blocks,
            prior_error=prior_error, prior_blocks=prior_blocks, record_model=read.get("record_model"),
            model=getattr(self._draft(), "model", None), built_on=read.get("built_on"),
            selected=_named(self.bounds_picker.key()), edited=read.get("edited", False),
            same_box=read.get("same_box", True), sets_of_prior=read.get("sets_of_prior"),
            rounds=values["num_iterations"] if cost_ok else None, batch=batch if cost_ok else None,
            stable=values["max_sets"] if cost_ok else None, steps=steps, took=self._took,
            sets_for_model=read.get("sets_for_model", 0), attention=self._attention(), name_facts=names,
            field_facts=field_facts)

    def _candidate_row(self) -> str:
        return "bounds_typed" if self._typed() else "bounds"

    def _bounds_trouble(self) -> tuple:
        """Why the bounds chosen cannot be used, and the words beside the button; ``("", "")`` when they can."""
        if self._draft() is None:
            return "", ""
        if not self._typed() and not self.bounds_picker.key():
            return NO_BOUNDS_TEXT, NO_BOUNDS_BLOCKS
        if self._candidate_error:
            return self._candidate_error, TYPED_BLOCKS if self._typed() else "the bounds file cannot be read"
        return "", ""

    def _attention(self) -> list:
        """What the configuration in use asks a person to know: its units declared against the pipeline's own, and a
        scale of the force no drive can move. Never raises."""
        cfg, _prior = self._held()
        if cfg is None:
            return []
        try:
            said = [str(line) for line in cfg.check_unit_consistency()]
        except Exception:                    # noqa: BLE001 -- what a session holds may be anything
            said = []
        try:
            if cfg.observation_mode == "spontaneous" and "f_scale" in cfg.rescale_params:
                said.append(FORCE_SCALE_TEXT)
        except Exception:                    # noqa: BLE001
            pass
        return said

    # ── the reading of the disk and the card ──
    def read_disk(self) -> None:
        """The two lists read again, but at the reading after the typing stops; the candidate the press would hand the
        run, built once for each bounds file as it stands and each draft; what it is; and, for the view shown, the saved
        prior's record and the simulation sets made with it, or how long the last build took and the sets of the
        model."""
        self._read = {}
        if not self._lists_wait:
            self._read_the_lists()
        self._build_candidate()
        self._say_the_file()
        if self.view == "saved":
            self._read_the_record()
        else:
            self._took = history.duration(controller().folder(), "Prior build")
            self._read["sets_for_model"] = self._sets_of_model()

    def _read_the_lists(self) -> None:
        """The bounds files of the model's folder and the saved priors, listed again: a file or a record written by
        another program comes in. Each list keeps its selection and says it only when it moved."""
        self.bounds_picker.refresh()
        self.prior_picker.refresh()

    def _timer_fired(self) -> None:
        """The reading after the typing stops reads what the typing changed and neither list: a person typing adds no
        file, and the saved list takes longer to read the more priors are saved. Every other reading reads both."""
        self._lists_wait = True
        try:
            super()._timer_fired()
        finally:
            self._lists_wait = False

    def _build_candidate(self) -> None:
        """The configuration of the draft held and the bounds chosen, built again only when either changed, a file by
        its path and the moment it was written; its trouble in a sentence when it cannot be built."""
        draft = self._draft()
        if draft is None:
            self._candidate, self._candidate_key, self._candidate_error = None, None, ""
            return
        if self._typed():
            if self.bounds_grid.problems():
                self._candidate, self._candidate_key, self._candidate_error = None, None, ""
                return
            source = {"bounds_dicts": self.bounds_grid.to_dicts()}
            key = ("typed", draft, repr(source["bounds_dicts"]))
        else:
            path = self.bounds_picker.selected_path()
            if not path:
                self._candidate, self._candidate_key, self._candidate_error = None, None, ""
                return
            try:
                written = Path(path).stat().st_mtime_ns
            except OSError:
                written = None
            source, key = {"bounds_path": path}, ("file", draft, path, written)
        if key == self._candidate_key:
            return
        self._candidate, self._candidate_key, self._candidate_error = None, key, ""
        try:
            candidate = draft.make_config(**source)
        except Refusal as refused:
            self._candidate_error = refused.message
            return
        except Exception as error:           # noqa: BLE001 -- a file the builder cannot read is its trouble, said
            self._candidate_error = f"The bounds file cannot be read: {type(error).__name__}: {error}"
            return
        trouble = self._limits_trouble(candidate)
        if trouble:
            self._candidate_error = FILE_UNUSABLE.format(trouble)
        else:
            self._candidate = candidate

    @staticmethod
    def _limits_trouble(cfg) -> str:
        """What is wrong with the limits a bounds file gave a configuration, by the rule the limits typed by hand are
        held to: the builder takes a line it cannot read as a parameter with no limits. "" for none."""
        try:
            limits = [(name, *two_limits(bounds)) for section in ("params_dict", "rescale_params", "force_params_dict")
                      for name, (_value, bounds) in (getattr(cfg, section, None) or {}).items()]
            if not getattr(cfg, "params_dict", None):
                return NO_PARAMETERS
            return next(iter(limit_problems(limits)), "")
        except Exception as error:           # noqa: BLE001 -- a configuration no rule can be read on
            return f"{type(error).__name__}: {error}"

    def _say_the_file(self) -> None:
        """What the bounds file is, under its list, from the candidate built on it."""
        cfg = self._candidate
        words = ""
        if cfg is not None and not self._typed():
            try:
                words = bounds_words(list(cfg.params_dict), list(cfg.rescale_params), bool(cfg.has_forcing),
                                     cfg.observation_mode)
            except Exception:                # noqa: BLE001 -- a configuration that cannot say what it is
                words = ""
        self.file_line.set_text(words)
        self.file_line.setVisible(bool(words))

    def _read_the_record(self) -> None:
        """The saved prior selected, as its record says it: its model, the bounds file it was built on against the one
        selected, and the simulation sets made with it."""
        key = self.prior_picker.key()
        self.prior_line.set_text("")
        self.prior_line.setVisible(False)
        if not key:
            return
        try:
            manifest = default_store().get("prior", key)
            gmm = manifest.body["gmm"]
        except Exception as error:           # noqa: BLE001 -- a record that cannot be read is said, not raised
            self._read["prior_error"] = f"The prior cannot be read: {type(error).__name__}: {error}"
            return
        recorded = (manifest.inputs or {}).get("bounds")
        self._read["record_model"] = (manifest.config or {}).get("model")
        self._read["built_on"] = recorded_file(recorded)
        path = self.bounds_picker.selected_path()
        if path and isinstance(recorded, dict) and self._read["built_on"] is not None:
            try:
                selected = file_ref(path, relative_to=config.RESOURCES_ROOT)
                self._read["edited"] = (selected["path"] == recorded.get("path")
                                        and selected["sha256"] != recorded.get("sha256"))
            except Exception:                # noqa: BLE001 -- a file gone since it was listed
                self._read["edited"] = False
        self._read["same_box"] = self._candidate is not None and _same_box(gmm, self._candidate)
        try:
            self._read["sets_of_prior"] = [batches for _name, batches in
                                           training_checkpoint.checkpoints_using_prior(manifest.fingerprints.get("gmm"))]
        except Exception:                    # noqa: BLE001 -- sets that cannot be read say nothing of the prior
            self._read["sets_of_prior"] = None
        keys = (manifest.config or {}).get("param_keys") or []
        self.prior_line.set_text(saved_prior_words(_made(manifest.created), self._read["record_model"] or "", len(keys),
                                                   self._read["built_on"]))
        self.prior_line.setVisible(True)

    def _sets_of_model(self) -> int:
        """How many saved simulation sets hold batches for the model applied: a new prior continues none of them."""
        model = getattr(self._draft(), "model", None)
        if model is None:
            return 0
        store, count = default_store(), 0
        try:
            rows = store.list("simulation")
        except Exception:                    # noqa: BLE001 -- a store that cannot be read holds no set to speak of
            return 0
        for row in rows:
            if not row.complete or not row.batches_done:
                continue
            try:
                identity = store.get("simulation", row.id).body.get("identity") or {}
            except Exception:                # noqa: BLE001
                continue
            count += identity.get("model") == model
        return count

    # ── the views ──
    def show_view(self, view: str) -> None:
        """Show "saved", a saved prior to load, or "build", a new prior to build: each its rows and its main button, its
        facts read once the typing has stopped."""
        self._view_chosen = True
        self._apply_view(view)
        self._edited()

    def _apply_view(self, view: str) -> None:
        if view not in ("saved", "build"):
            raise ValueError(f"no such view of the Prior step: {view!r}")
        self.view = view
        build = view == "build"
        self.switch_button.setText(style.access_key(TO_SAVED if build else TO_BUILD))
        rows = self.form.rows
        rows["prior"].set_shown(not build)
        for key in _BUILD_ROWS:
            rows[key].set_shown(build)
        for row in self._names.rows.values():
            row.set_shown(build)
        self.form.side.setVisible(build)
        self.names = self._names if build else None
        self._typed_shown()

    def _switch(self) -> None:
        self.show_view("saved" if self.view == "build" else "build")

    @reported(QWidget.showEvent)
    def showEvent(self, event):
        """The first time the page comes into sight it opens on a saved prior to load when one is saved, else on a new
        one to build: the list of saved priors is read for it, which a run does not forbid. Each time it comes into
        sight with no run live, the two lists are read again, by the reading when the page has one to make, else on
        their own and drawn: a list that gained its first record moves no selection, and says nothing itself."""
        if not self._view_chosen:
            self._view_chosen = True
            self.prior_picker.refresh()
            self._apply_view("saved" if self.prior_picker.has_entries() else "build")
            self.redraw()
            self._tell_the_rail()
        reads = self.stale
        super().showEvent(event)
        if not reads and not controller().is_live():
            self._read_the_lists()
            self.redraw()

    # ── the choices ──
    def _bounds_chosen(self, _index: int = 0) -> None:
        """Another bounds file selected: what was said of the last one goes, and the limits typed by hand follow it."""
        self.gone_line.setVisible(False)
        self.file_line.setVisible(False)
        if self.typed_tick.isChecked():
            self._load_the_grid()

    def _typed_toggled(self, _on: bool = False) -> None:
        """The limits typed by hand, under the tick: loaded from the file selected the first time, and kept after."""
        if self.typed_tick.isChecked() and self._grid_from != self.bounds_picker.selected_path():
            self._load_the_grid()
        self._typed_shown()

    def _typed_shown(self) -> None:
        on = self.typed_tick.isChecked()
        self.bounds_grid.setVisible(on)
        self.typed_note.setVisible(on)

    def _load_the_grid(self) -> None:
        """Fill the limits typed by hand from the file selected; a file that cannot be read leaves them as they were."""
        path = self.bounds_picker.selected_path()
        if not path:
            return
        try:
            params, rescale, forcing, _ = file_manager.parse_bounds_file(path)
            self.bounds_grid.load(params, rescale, forcing)
        except Exception:                    # noqa: BLE001 -- the file's trouble is said beside the list
            return
        self._grid_from = path
        self._a_stop_joined()

    def on_draft_set(self, draft) -> None:
        """Step 1 applied: the list of bounds files points at the model's own folder, its selection kept where that
        folder holds it, else the file that came into use last for this model, which is said when it has gone; the
        limits typed by hand follow."""
        model = getattr(draft, "model", None)
        folder = BOUNDS_PATH / str(model or "nadrowski").lower()
        key = self.bounds_picker.key() if Path(folder) == Path(self.bounds_picker.base_path) else ""
        if not key and self._bounds_pick[0] == model:
            key = self._bounds_pick[1]
        self.bounds_picker.repoint(folder, key)
        gone = bool(key) and self.bounds_picker.key() != key
        self.gone_line.set_text(GONE_TEXT.format(_named(key)) if gone else "")
        self.gone_line.setVisible(gone)
        self.form.rows["bounds"].kept = not gone and bool(key) and self._kept_pick == (model, key)
        if self._kept_pick is not None and self._kept_pick[0] == model:
            self._kept_pick = None               # given back to its model: kept from last time until touched
        self._grid_from = None
        if self.typed_tick.isChecked():
            self._load_the_grid()
        self._edited()

    # ── the press ──
    def _read_knobs(self) -> dict:
        """The seven settings, read through the rules: the first broken is refused, and nothing changes."""
        for key, (_line, said) in self.problems().items():
            raise Refusal(said, field=key)
        return {key: self.form.rows[key].value() for key, _ in _KNOB_RULES}

    def _release_asked(self, build: bool) -> bool:
        """With a posterior held, ask before another prior lets it go, and the observation with it; True to go on."""
        session = self.session
        posterior, observation = getattr(session, "posterior", None), getattr(session, "observation", None)
        if posterior is None:
            return True
        held = [record_words.phrase_of("posterior", posterior)]
        if observation is not None:
            held.append(record_words.phrase_of("observation", observation))
        several = len(held) > 1
        sentence = RELEASE_TEXT.format(" and ".join(held), "They belong" if several else "It belongs",
                                       "them" if several else "it", "is built" if build else "has loaded",
                                       "Each stays" if several else "It stays")
        return ask(self, BUILD_TITLE if build else LOAD_TITLE, sentence, KEEP_POSTERIOR,
                   BUILD_RELEASE if build else LOAD_RELEASE)

    def main_pressed(self) -> None:
        """Load the saved prior selected, or build a new one, on the candidate the press's reading holds: asked first
        when a posterior is held. Nothing is held until the run returns its prior."""
        candidate = self._candidate
        if candidate is None or self._draft() is None:
            return                               # the facts say why, and the button is greyed with them
        build = self.view == "build"
        if build:
            try:
                knobs = self._read_knobs()
            except Refusal as refused:
                self._refusal(refused)
                return
            args, kwargs = (candidate, None, True), dict(name=self._names.name_value(), note=self._names.note_value(),
                                                         if_name_taken="unnamed", **knobs)
        else:
            key = self.prior_picker.key()
            if not key:
                return
            args, kwargs = (candidate, key, False), {}
        if not self._release_asked(build):
            return
        self._dispatched, self._dispatched_key = candidate, self._candidate_key
        self.dispatch(orchestrator.build_prior, *args, provide_fig_sink=True, on_result=self._on_prior, **kwargs)

    def load_record(self, ref: str) -> bool:
        """Load the saved prior ``ref`` as "Load this prior" does, with the bounds file selected, for a caller that asked
        its own question first: the saved view shown and read, so that the list holds a record saved since by another
        program, ``ref`` selected, the facts read, and the load started unless they block it. True when its run was
        started."""
        self._by_keyboard = False                # no key pressed this load
        self.show_view("saved")
        self.read_facts()
        self.prior_picker.restore_key(ref)
        self.read_facts()
        if self.prior_picker.key() != ref or not self.foot.main_button.isEnabled() or self._candidate is None:
            return False
        self._dispatched, self._dispatched_key = self._candidate, self._candidate_key
        return self.dispatch(orchestrator.build_prior, self._candidate, ref, False, provide_fig_sink=True,
                             on_result=self._on_prior)

    def run_spec(self, fn, *args, **kwargs) -> RunSpec:
        """The prior's build, which writes a prior; or its load, which writes nothing, keeps no file and stops
        without a question (its third argument says which, as the stage reads it)."""
        if fn is not orchestrator.build_prior:
            return super().run_spec(fn, *args, **kwargs)
        if args[2]:
            return RunSpec("Prior build", _PLACE, name=kwargs.get("name", ""), stop_button="Stop the prior build…",
                           kinds=("prior",))
        return RunSpec("Prior load", _PLACE, stop_button="Stop loading", loads_only=True, keeps_files=False,
                       stop_asks=False)

    def _on_prior(self, payload) -> None:
        """The run returned its prior, while it is still live: the prior and the candidate it was made on come into
        use, and nothing is read until the run has ended."""
        self._made = payload
        self._screen.take_prior(self._dispatched, payload)

    def _run_ended(self, status) -> None:
        """A run has ended. One of this page's that returned its prior: the saved list read again and moved to it, its
        bounds file remembered as the last in use, and after a build the name and the note emptied, with a word when
        the record came back under another name. Then the reading."""
        made, self._made = self._made, None
        if made is not None:
            cfg, built = self._dispatched, self._dispatched_key
            self._held_at = (cfg, built[3] if built and built[0] == "file" else None)
            key = bounds_in_use(cfg)
            if key:
                self._bounds_pick, self._kept_pick = (getattr(cfg, "model", ""), key), None
            self.prior_picker.refresh()
            self.prior_picker.restore_key(getattr(made, "id", "") or "")
            if self.view == "build":
                typed = self._names.name_value()
                self._names.clear()
                got = getattr(made, "name", "")
                if typed and got != typed:
                    self.in_use.say(Fact("warn", SAVED_WITHOUT_NAME.format(typed)))
        self._dispatched = self._dispatched_key = None
        super()._run_ended(status)

    def refresh_local_gates(self) -> None:
        """What the step holds and what follows from it, and the way on to Training once a prior is in use."""
        super().refresh_local_gates()
        held = self._held()[1] is not None
        self.set_next(GO_TO_TRAINING if held else "", "training" if held else None)

    def set_controls_enabled(self, enabled: bool) -> None:
        """A run locks the form, the foot's two buttons, and the way between the two views."""
        self.switch_button.setEnabled(enabled)
        super().set_controls_enabled(enabled)

    # ── what is remembered ──
    def restore_settings(self, qs) -> None:
        """The bounds file and the prior that came into use last, put back as choices, never applied, and marked kept
        from last time until they are touched. The bounds file waits for step 1 to apply the model it came into use
        with, whose folder holds it; a file no longer there is said. A file remembered with no model, as an older PRISM
        wrote it, waits for the model step 1 applied last."""
        model = settings.get_str(qs, BOUNDS_MODEL_KEY, "") or settings.get_str(qs, MODEL_KEY, "")
        self._bounds_pick = self._kept_pick = (model, settings.get_str(qs, BOUNDS_KEY, ""))
        prior = settings.get_str(qs, PRIOR_KEY, "")
        self.prior_picker.restore_key(prior)
        self.form.rows["prior"].kept = bool(prior) and self.prior_picker.key() == prior
