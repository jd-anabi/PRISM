"""Step 4, Checks: every check there is, in one ruled table under its four groups, each with its role, its state and when
it last ran, under the headline over the six deciding checks; and one page for each check, beside the table on a wide
page or in its place on a narrow one, with "All checks" to go back.

What the page shows is read when it is shown, when a run ends and when Records changes the store, by the screen's one
reading of what the checks need; never while a run is live, when the page shows what it read last. A check's result,
its figures among it, is made at a reading, from the record that counts.

A row opens its check's page by a click, Space or Enter. The foot's main button runs the check open; with none open it
opens the first deciding check a run can still pass, or goes to Inference when none is left.

A check that simulates from the cell or in a fixed kind of observation runs on a configuration of its own, made at the
press from the step 1 settings applied and the bounds in use, as the session's was made, its cell loaded on it alone:
the session's configuration is never written. A check's form is read at every edit with what the page's last reading
holds, this computer's finished checks among it, and reads nothing.
"""
import dataclasses
import functools
import random
from collections import OrderedDict
from pathlib import Path

from PySide6.QtCore import QEvent, QItemSelectionModel, QPoint, Qt, Signal
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from core import cli, orchestrator
from core.config import CELL_PATH
from core.diagnostics import ablation, identifiability, probes
from core.Helpers import labels
from core.refusals import Refusal

from ... import check_lineage, check_readings, design, fonts, record_words, style
from ...check_catalogue import AGAIN_WORDS, CHECKS, DECIDING, GROUPS, entry
from ...memory_limit import computer, limit_words, memory_limit_in_force
from ...reported import reported
from ...run import ends, history
from ...run.controller import controller
from ...run.model import RunSpec
from ...run.questions import TRAINING_WORDS
from ...step_facts import NEEDS_A_PRIOR, NEEDS_A_PRIOR_BLOCKS, NOT_BUILT_BLOCKS, Fact, parameter_units
from ...texts import date_text
from ...widgets.list_detail import ListAndDetail
from ...widgets.page import Sentence, _ring_room
from ...widgets.rule_table import Column, RuleTable, StateCell, TextCell
from . import check_forms
from .check_page import CheckPage
from .check_results import comparison_block, history_block, result_block
from .step_page import StepPage

TITLE = "Checks"
TITLE_OF_POSTERIOR = "Checks of {}"
TITLE_BEFORE_TRAINING = "Checks before training on {}"
SENTENCE = "A posterior passes when all six deciding checks have passed."
SENTENCE_OF_THREE = "A posterior passes when the three deciding checks that apply to it have passed."
#: What "In use" names: the posterior and the prior it was trained on, or the two when the prior in use is another,
#: or the prior alone.
IN_USE_BOTH = "posterior {}, trained on prior {}"
IN_USE_PAIR = "posterior {} and prior {}"
IN_USE_PRIOR = "prior {}"
#: The table's columns, whole and beside a check's page.
COLUMNS = ("Check", "Role", "State", "Last run")
NARROW_COLUMNS = ("Check", "State")
#: From how wide a page's body a check's page stands beside the list, in the look's pixels at the ordinary text size:
#: the list's two columns beside it, each check's name on two lines; and the width the list asks for beside it.
CHECK_BESIDE_WIDTH = 890
LIST_NARROW_WIDTH = 436
#: The foot's main button with no check open.
OPEN_WORDS = "Open {}"
GO_TO_INFERENCE = "Go to Inference"
#: What the comparison after simulating says while a training or a narrowing round is live: it is worked out once the
#: run's simulation ends.
COUNTED_LATER_TEXT = "Counted when simulation ends."
#: What the list says of a record whose manifest cannot be read.
UNREADABLE_RECORD_TEXT = "The record {} cannot be read, so it counts for no check."
#: The words beside the main button for a setting of a check's form that breaks its rule.
SETTING_BLOCKS = "a setting cannot be used"
#: What a check's own configuration says when its bounds or its cell file cannot be read.
BOUNDS_UNREAD_TEXT = "The bounds file cannot be read: {}"
CELL_UNREAD_TEXT = "The cell file cannot be read: {}"
#: A calibration's seed left blank is drawn at the press, below this, and its record names it.
DRAWN_SEED_BELOW = 2 ** 31
#: Each check's stage by the module that holds it and its name there, read at each look, so that what a test puts in
#: its place is the function looked for; and the method of this page that starts it.
_ENGINE = {"band": (probes, "probe_band"), "audit": (probes, "probe_mask"),
           "calibration": (orchestrator, "validate_calibration"), "input_use": (ablation, "channel_ablation"),
           "identifiability": (identifiability, "identifiability_rotation")}
_STARTS = {"band": "_start_band", "audit": "_start_audit", "calibration": "_start_calibration",
           "input_use": "_start_input_use", "identifiability": "_start_identifiability"}
#: The keys that open the check of the row the keyboard is on.
_OPENING_KEYS = (Qt.Key_Return, Qt.Key_Enter, Qt.Key_Space)


class _Listing(QWidget):
    """The twelve checks under their groups' headings: the whole table, or, told narrow, the name and the state alone.
    ``chosen`` carries the key of the check whose row was clicked, or that Space or Enter was pressed on."""

    chosen = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self.narrowed = False
        self.drawing = False                 # the rows are being drawn again: the keyboard's row moves no page
        self.headings: list = []
        self.row_of: dict = {}               # each check's row, by its key
        self.key_at: dict = {}               # the key of each row of a check, by the row
        self.whole = RuleTable([Column(COLUMNS[0]), Column(COLUMNS[1], "fit"), Column(COLUMNS[2], "fit", wrap=True),
                                Column(COLUMNS[3], "fit")])
        self.narrow = RuleTable([Column(NARROW_COLUMNS[0]), Column(NARROW_COLUMNS[1], "fit", wrap=True)])
        box = QVBoxLayout(self)
        box.setContentsMargins(0, 0, 0, 0)
        for table in (self.whole, self.narrow):
            table.view.clicked.connect(self._clicked)
            table.view.installEventFilter(self)
            box.addWidget(table)
        self.set_rows(None)

    def set_rows(self, found) -> None:
        """Show each check's row of ``found`` (the checks' rows in the list's order; None before the first reading,
        the names alone). The row the keyboard is on in each table stays the current one, by its check, and no drawing
        moves the keyboard: ``drawing`` says so while it lasts. A group's heading is no row a click selects or the keyboard
        stops on."""
        self.drawing = True
        try:
            self._set_rows(found)
        finally:
            self.drawing = False

    def _set_rows(self, found) -> None:
        kept = {table: self.key_at.get(table.view.currentIndex().row()) for table in (self.whole, self.narrow)
                if table.view.currentIndex().isValid()}
        by_key = {row.key: row for row in found or ()}
        whole, narrow = [], []
        self.headings, self.row_of, self.key_at = [], {}, {}
        for group in GROUPS:
            self.headings.append(len(whole))
            whole.append([TextCell(group, wrap=True), TextCell(""), TextCell(""), TextCell("")])
            narrow.append([TextCell(group, wrap=True), TextCell("")])
            for one in (one for one in CHECKS if one.group == group):
                row = by_key.get(one.key)
                state = (StateCell(check_readings.STATE_MARKERS.get(row.state, "later"), row.state) if row is not None
                         else TextCell(""))
                when = TextCell(date_text(row.when) if row is not None and row.when is not None else "")
                self.row_of[one.key], self.key_at[len(whole)] = len(whole), one.key
                whole.append([TextCell(one.name, wrap=True), TextCell(one.role), state, when])
                narrow.append([TextCell(one.name, wrap=True), state])
        for table, rows in ((self.whole, whole), (self.narrow, narrow)):
            table.set_rows(rows)
            model = table.view.model()
            for at in self.headings:
                table.view.setSpan(at, 0, 1, model.columnCount())
                for column in range(model.columnCount()):
                    model.item(at, column).setFlags(Qt.NoItemFlags)
            if kept.get(table) in self.row_of:
                table.view.selectionModel().setCurrentIndex(model.index(self.row_of[kept[table]], 0),
                                                            QItemSelectionModel.NoUpdate)

    def view(self):
        """The table in sight's view."""
        return (self.narrow if self.narrowed else self.whole).view

    def select(self, key: "str | None", *, current: bool = True) -> None:
        """Select the row of the check ``key`` in both tables, its row the current one unless ``current`` is False, when
        the keyboard's row stays where it is; None selects none."""
        for table in (self.whole, self.narrow):
            row = self.row_of.get(key)
            if row is None:
                table.select_row(None)
            elif current:
                table.select_row(row)
                table.view.setCurrentIndex(table.view.model().index(row, 0))
            else:
                table.view.selectionModel().select(table.view.model().index(row, 0),
                                                   QItemSelectionModel.ClearAndSelect | QItemSelectionModel.Rows)

    def set_narrow(self, on: bool) -> None:
        """Beside a check's page, the name and the state alone; else the whole table."""
        self.narrowed = bool(on)
        self.whole.setVisible(not self.narrowed)
        self.narrow.setVisible(self.narrowed)

    @reported(QWidget.sizeHint)
    def sizeHint(self):
        hint = super().sizeHint()
        if self.narrowed:
            hint.setWidth(design.scaled(LIST_NARROW_WIDTH))
        return hint

    def stops(self) -> list:
        return [self.view()]

    def take_the_keyboard(self, detail) -> bool:
        """The keyboard on the row of the check ``detail`` shows, in the table in sight."""
        row = self.row_of.get(getattr(detail, "key", None))
        if row is None:
            return False
        view = self.view()
        view.setCurrentIndex(view.model().index(row, 0))
        view.setFocus(Qt.TabFocusReason)
        return True

    def _clicked(self, index) -> None:
        key = self.key_at.get(index.row())
        if key is not None:
            self.chosen.emit(key)

    @reported(False)
    def eventFilter(self, watched, event):
        """Space and Enter open the check of the row the keyboard is on."""
        if event.type() == QEvent.KeyPress and event.key() in _OPENING_KEYS and watched in (self.whole.view,
                                                                                           self.narrow.view):
            key = self.key_at.get(watched.currentIndex().row()) if watched.currentIndex().isValid() else None
            if key is not None:
                self.chosen.emit(key)
            return True
        return False


class ValidatePanel(StepPage):
    """Step 4. ``listing`` is the table of the twelve checks, ``list_detail`` the list and the check open beside it or
    in its place; ``check_pages`` holds each check's page by its key, ``open_key`` the check open (None for none),
    ``headline`` the line over the deciding checks and ``headline_sentence`` the one under it. The page reads what the
    checks need at its own readings, so the screen does not read them again while the page is in sight."""

    #: Whether the page's reading reads what the checks need, which the screen then leaves to it while it is in sight.
    reads_the_checks = True

    def __init__(self, screen, parent=None):
        super().__init__(screen, TITLE, SENTENCE, step="checks", parent=parent)
        self.open_key: "str | None" = None
        self.headline = style.set_class(QLabel(""), cls="head")
        self.headline.setTextFormat(Qt.PlainText)
        self.headline.setFont(fonts.inter("head"))
        self.headline.setWordWrap(True)
        self.headline_sentence = Sentence("")
        head = QWidget()
        lines = QVBoxLayout(head)
        lines.setContentsMargins(0, 0, 0, 0)
        lines.setSpacing(design.scaled(design.GAP["head_lead"]))
        lines.addWidget(self.headline)
        lines.addWidget(self.headline_sentence)
        self.headline_block = head
        self.listing = _Listing()
        self.listing.chosen.connect(self.open_check)
        # Each check's page is this page's child from the start, out of sight until it is opened: a widget kept with no
        # parent until then, built under the application's style sheet and let go after its window, can corrupt the
        # process's memory.
        self.check_pages = {one.key: CheckPage(one, check_forms.build_form(one.key), self) for one in CHECKS}
        for page in self.check_pages.values():
            page.hide()
            if page.form.names is not None:
                page.form.names.changed.connect(self.redraw)
        # What the last reading found of this computer's runs: its finished checks, its last run and the last that
        # measured its batches; a check's "Before you start" says them at every edit, reading nothing.
        self._ran_here: list = []
        self._last = self._measured = None
        self._history_read = False
        self.list_detail = ListAndDetail(self.listing, beside_width=design.scaled(CHECK_BESIDE_WIDTH))
        self.list_detail.back_pressed.connect(self._all_checks)
        self.list_detail.switched.connect(self._switched)
        at = self._body.indexOf(self.in_use) + 1
        self._body.insertWidget(at, self.headline_block)
        self._body.insertWidget(at + 1, self.list_detail)
        self.form.setVisible(False)          # the list's own form holds nothing
        for table in (self.listing.whole, self.listing.narrow):
            table.view.selectionModel().currentChanged.connect(self._row_moved)
        self.refresh_local_gates()

    # ── what is in use ──
    def _held_records(self) -> tuple:
        session = getattr(self._screen, "session", None)
        return getattr(session, "inf_prior", None), getattr(session, "posterior", None)

    def in_use_words(self) -> tuple:
        """The posterior in use and the prior it was trained on, or the prior alone; nothing with no prior."""
        prior, posterior = self._held_records()
        if prior is None:
            return "", ""
        prior_label = record_words.label_of(prior)
        if posterior is None:
            return IN_USE_PRIOR.format(prior_label), ""
        parents = getattr(getattr(posterior, "manifest", None), "parents", None)
        trained = parents.get("prior") if isinstance(parents, dict) else None
        words = IN_USE_BOTH if trained is not None and trained == getattr(prior, "id", None) else IN_USE_PAIR
        return words.format(record_words.label_of(posterior), prior_label), ""

    def _title(self) -> tuple:
        """The page's heading and the sentence under it, by what is in use."""
        prior, posterior = self._held_records()
        if posterior is not None:
            title = TITLE_OF_POSTERIOR.format(record_words.label_of(posterior))
        elif prior is not None:
            title = TITLE_BEFORE_TRAINING.format(record_words.label_of(prior))
        else:
            title = TITLE
        cfg = getattr(getattr(self._screen, "session", None), "cfg", None)
        three = posterior is not None and getattr(cfg, "chi_mode", None) is False
        return title, SENTENCE_OF_THREE if three else SENTENCE

    # ── the rows, as the screen last read them ──
    def _rows(self) -> "tuple | None":
        rows = getattr(self._screen, "check_rows", None)
        return rows() if callable(rows) else None

    def _by_key(self) -> dict:
        return {row.key: row for row in self._rows() or ()}

    def _held(self):
        held = getattr(self._screen, "checks_in_use", None)
        return held() if callable(held) else None

    def _next_check(self) -> "str | None":
        """The first deciding check that applies, has not passed and that a run can still pass, the check open passed
        over, since its own page stands already; with nothing read, the first deciding check that is not open; None when
        none is left."""
        found, held = self._rows(), self._held()
        if found is None or held is None:
            return next(key for key in DECIDING if key != self.open_key)
        for row in found:
            if (row.key in DECIDING and row.key != self.open_key and row.state not in ("Passed", "Does not apply")
                    and not check_lineage.no_run_passes(row, held=held)):
                return row.key
        return None

    def _draw(self, *, read: bool) -> None:
        """The heading, the headline, the table and each check's page, from the screen's last reading of what the
        checks need and what is in use now. A check's result from a record, figures and all, is made only at a reading
        (``read``); between readings a result whose record no longer decides goes."""
        title, sentence = self._title()
        self.head.title.setText(title)
        self.head.set_sentence(sentence)
        summary = getattr(self._screen, "checks_summary", lambda: None)()
        self.headline.setText(summary.headline if summary is not None else "")
        self.headline_sentence.set_text(summary.sentence if summary is not None else "")
        self.headline_block.setVisible(summary is not None)
        found = self._rows()
        self.listing.set_rows(found)
        self.listing.select(self.open_key, current=False)
        by_key = {row.key: row for row in found or ()}
        kept = getattr(self._screen, "checks_held", None)
        records = kept.records if kept is not None else ()
        for key, page in self.check_pages.items():
            row = by_key.get(key)
            result, source = self._result(key, row, found, kept, read=read, page=page)
            page.show_row(row, result, earlier=check_lineage.earlier_lines(row, records) if row is not None else (),
                          source=source)

    def _result(self, key, row, found, kept, *, read: bool, page) -> tuple:
        """The result the page of the check ``key`` shows for its ``row``, and what it was made from: kept when it would
        say the same; a record's made only at a reading, from its folder with the count of the other checks still open,
        the audit's prior and the units in use, which its words say, and kept between readings while its folder
        decides."""
        if row is None or kept is None:
            return None, None
        if key == "comparison":
            return self._comparison_result(row, found, page)
        if key in ("timing", "card_test"):
            ran = [one for one in kept.histories
                   if isinstance(one.get("check"), dict) and one["check"].get("key") == key]
            if not ran:
                return None, None
            source = ran[-1].get("id")
            return ((page.result, source) if page.result_source == source
                    else (history_block(key, ran[-1], parent=page), source))
        if row.record is None:
            return None, None
        # What the block's words say beside the record: the count of the others still open, which the calibration's
        # says, the prior an audit was made of, and the units in use, each parameter's and the cell's of force.
        others = self._others_open(key, found) if key == "calibration" else 0
        prior_words = self._audit_prior(row)
        units, force_unit = self._units()
        source = (row.record.folder, others, prior_words, tuple(sorted(units.items())), force_unit)
        if page.result_source == source:
            return page.result, source
        if not read:
            same_folder = isinstance(page.result_source, tuple) and page.result_source[0] == row.record.folder
            return (page.result, page.result_source) if same_folder else (None, None)
        return self._record_result(key, row, kept, others=others, prior_words=prior_words, units=units,
                                   force_unit=force_unit, page=page), source

    def _others_open(self, key, found) -> int:
        """How many deciding checks other than ``key`` apply, have not passed and a run can still pass."""
        held = self._held()
        return sum(1 for one in found if one.key in DECIDING and one.key != key
                   and one.state not in ("Passed", "Does not apply")
                   and not (held is not None and check_lineage.no_run_passes(one, held=held)))

    def _record_result(self, key, row, kept, *, others: int, prior_words: str, units, force_unit: str, page):
        """The result of the record that decides the row of the check ``key``, made inside its page."""
        return result_block(key, row.record.manifest, row.record.folder, parent=page, units=units,
                            force_unit=force_unit, others_open=others, prior_words=prior_words,
                            text=kept.texts.get(row.record.folder))

    def _comparison_result(self, row, found, page) -> tuple:
        """The comparison's own result, from the counts the reading found: made, or waiting for the audit; none when
        it is not compared, since its state line says why, or when it waits for a posterior or the simulations. While a
        training or a narrowing round is live, that it is counted when the simulation ends, where it applies. The one
        shown is kept while it would say the same."""
        held = self._held()
        status = controller().status
        if status.live and status.word in TRAINING_WORDS and row.state != "Does not apply":
            made = (COUNTED_LATER_TEXT, "Running", False)
        elif held is None or row.state not in ("Passed", "Look closer", "Failed", "Waiting for the audit"):
            return None, None
        else:
            compared = check_lineage.comparison_of(held, found)
            if compared is None:
                return None, None
            unusable, total = held.set_counts[0], held.set_counts[1]
            made = (check_readings.compared_sentence(compared, unusable=unusable, total=total,
                                                     audit_fraction=check_lineage.audit_fraction(found),
                                                     prior_words=held.prior_words or "the prior in use"),
                    row.state, held.narrowed)
        if page.result_source == made:
            return page.result, made
        sentence, state, narrowed = made
        return comparison_block(sentence, state, narrowed=narrowed, parent=page), made

    def _audit_prior(self, row) -> str:
        """The prior an audit was made of, as its sentence names it."""
        if row.key != "audit":
            return ""
        parents = row.record.manifest.get("parents") if row.record is not None else None
        made_of = parents.get("prior") if isinstance(parents, dict) else None
        prior = self._held_records()[0]
        if prior is not None and made_of == getattr(prior, "id", None):
            return record_words.phrase_of("prior", prior)
        held = self._held()
        named = getattr(held, "posterior_prior_words", "")
        return f"the prior {named}" if named else ""

    def _units(self) -> tuple:
        """Each parameter's unit by its name, and the cell's unit of force, from the configuration in use."""
        cfg = getattr(getattr(self._screen, "session", None), "cfg", None)
        if cfg is None:
            return {}, ""
        try:
            kinds = {name: labels._RESCALE_KIND.get(name) for name in cfg.rescale_idx}
            units = {"length": cfg.length_unit, "time": cfg.time_unit, "force": cfg.force_unit}
            return parameter_units(kinds, units, cfg.assumed_params), cfg.force_unit or ""
        except Exception:                    # noqa: BLE001 -- a configuration that cannot say its units shows none
            return {}, ""

    def _unreadable(self) -> list:
        """The folders of the check records whose manifest the last reading could not read."""
        kept = getattr(self._screen, "checks_held", None)
        return [Path(record.folder).name for record in (kept.records if kept is not None else ()) if not record.manifest]

    # ── what a page answers ──
    def main_words(self) -> str:
        """The open check's button, "Run again" once a record of it exists; with none open, or with one open that has
        no button, "Open" and the next deciding check, or "Go to Inference"."""
        key = self.open_key
        if key is not None and entry(key).button:
            row = self._by_key().get(key)
            ran = row is not None and (row.record is not None or bool(row.earlier) or row.state == "Done")
            return AGAIN_WORDS if ran else entry(key).button
        following = self._next_check()
        return OPEN_WORDS.format(entry(following).name) if following is not None else GO_TO_INFERENCE

    def main_pressed(self) -> None:
        """Run the open check; with none open, open the next deciding check, or go to Inference."""
        key = self.open_key
        if key in _STARTS:
            getattr(self, _STARTS[key])()
            return
        if key is not None and entry(key).button:
            return                           # its facts grey it: the window cannot run it yet
        following = self._next_check()
        if following is None:
            self._screen.navigate.emit("inference", "infer")
        else:
            self.open_check(following)

    def facts_from_form(self) -> tuple:
        """With no check open: that nothing can be checked with no prior, the changes of the earlier steps not applied,
        and each record that cannot be read. With a check open: what its run simulates or reads, how long it took here
        last at these sizes, the memory limit and the last run for a check on the graphics card, that nothing else can
        run meanwhile and what it writes; the earlier steps' changes; and what stops its button: its row's reason, a cell
        file not chosen, the rules its form breaks, its name and its note."""
        others = tuple(Fact("warn", said) for said in self._others_pending() if said)
        key = self.open_key
        if key is None or not entry(key).button:
            blocks = () if self._held_records()[0] is not None else (Fact("lock", NEEDS_A_PRIOR, NEEDS_A_PRIOR_BLOCKS),)
            unread = tuple(Fact("warn", UNREADABLE_RECORD_TEXT.format(name)) for name in self._unreadable())
            return (*blocks, *others, *unread)
        row = self._by_key().get(key)
        not_built = "" if key in check_forms.FORMS else check_forms.NOT_BUILT_TEXT
        if row is not None:
            blocks = check_lineage.button_facts(row, not_built=not_built)
        else:
            blocks = (Fact("lock", not_built, NOT_BUILT_BLOCKS),) if not_built else ()
        if not_built:
            return (*blocks, *others)
        form = self.check_pages[key].form
        found = check_forms.problems(key, form, slots=self._slots())
        for setting, row_ in form.rows.items():
            if row_.widget is None or setting in ("name", "note", "cell"):
                continue
            row_.set_line("err" if setting in found else "", found[setting][0] if setting in found else "")
        fields = tuple(Fact("err", line, SETTING_BLOCKS) for line, _said in found.values())
        picker = check_forms.cell_picker(form)
        cell = (Fact("err", check_forms.NO_CELL_TEXT, check_forms.NO_CELL_BLOCKS),) if (
            picker is not None and not picker.key()) else ()
        names = tuple(fact for fact in (form.names.name_fact(), form.names.note_fact()) if fact) if form.names else ()
        sizes = check_forms.sizes_of(key, form)
        said = check_forms.before_you_start(key, sizes=sizes, last=history.check_at(self._ran_here, check=key,
                                                                                      sizes=sizes),
                                            memory=self._memory())
        return (*said, *others, *blocks, *cell, *fields, *names)

    def _memory(self) -> tuple:
        """The memory limit in force and, once the history has been read, the last run on this computer."""
        limit = limit_words(memory_limit_in_force())
        return (limit, ends.last_run_sentence(self._last, self._measured)) if self._history_read else (limit,)

    def _slots(self) -> "int | None":
        """The driven recordings an observation holds at most, by the configuration in use; None without one."""
        slots = getattr(getattr(self.session, "cfg", None), "chi_k_pad", None) if self._screen is not None else None
        return slots if isinstance(slots, int) and not isinstance(slots, bool) else None

    def region_stops(self) -> list:
        return self.list_detail.stops()

    def first_focus(self):
        """The first setting of the check open, while its form can be edited; else, with a check open in the list's
        place, the way back to the list; else the table."""
        if self.open_key is not None:
            target = self.form.first_control() if self.form.isEnabled() else None
            if target is not None:
                return target
            if not self.list_detail.beside():
                return self.list_detail.back
        return self.listing.view()

    # ── the check open ──
    def open_check(self, key: "str | None") -> None:
        """Open the page of the check ``key`` beside the list or in its place, its form handed the foot, the keyboard
        on its first setting, or else on the way back; None for the whole list again. KeyError for a key that is no
        check's, before anything moves."""
        page = None if key is None else self.check_pages[key]
        self.open_key = key
        self.names = None if page is None else page.form.names
        self.list_detail.show_detail(page)
        self.use_form(self._own_form if page is None else page.form)
        self.listing.select(key)
        self.redraw()
        if page is None or not self.isVisible():
            return
        target = page.form.first_control() if page.form.isVisible() and page.form.isEnabled() else None
        if target is None and not self.list_detail.beside():
            target = self.list_detail.back
        if target is not None:
            target.setFocus(Qt.OtherFocusReason)

    def _switched(self) -> None:
        """The check's page moved between beside the list and its place: the Tab key's order again, as it now stands."""
        if self._keys_ordered:
            self.order_the_keys()

    def _all_checks(self) -> None:
        """"All checks": the whole list again, its own form handed back."""
        self.open_key = self.names = None
        self.use_form(self._own_form)
        self.listing.select(None)
        self.redraw()

    def set_controls_enabled(self, enabled: bool) -> None:
        """A run locks every check's form, whichever is in sight, and the foot's two buttons."""
        for page in self.check_pages.values():
            page.form.setEnabled(enabled)
        super().set_controls_enabled(enabled)

    # ── the readings ──
    def read_disk(self) -> None:
        """What the checks need, read by the screen; this computer's runs from the run history: its finished checks, its
        last run and the last that measured its batches; each check's list of cell files listed again in the folder of
        the model in use, one that holds none taking the cell step 5 holds; then every check's row and its result drawn
        from it."""
        read = getattr(self._screen, "read_checks", None)
        if callable(read):
            read()
        folder, here = controller().folder(), computer()
        self._ran_here = history.checks_on(folder, computer=here)
        self._last, self._measured = history.last_on(folder, here), history.last_measured_on(folder, here)
        self._history_read = True
        self._point_the_cells()
        self._draw(read=True)

    def _point_the_cells(self) -> None:
        """Each check's list of cell files, listed in the folder of the model in use, its choice kept while the file is
        there; one that holds none takes the cell the Inference step's list holds, or stays on none."""
        session = getattr(self._screen, "session", None)
        model = (getattr(getattr(session, "cfg", None), "model", None)
                 or getattr(getattr(session, "draft", None), "model", None))
        if not isinstance(model, str) or not model:
            return
        infer = getattr(getattr(self._screen, "infer_panel", None), "cell_picker", None)
        theirs = infer.key() if infer is not None else ""
        folder = CELL_PATH / model.lower()
        for page in self.check_pages.values():
            picker = check_forms.cell_picker(page.form)
            if picker is None:
                continue
            if Path(picker.base_path) != folder:
                picker.repoint(folder)
            else:
                picker.refresh()
            if not picker.key() and theirs:
                picker.restore_key(theirs)

    def _edited(self) -> None:
        """A check's form was edited: drawn again from what the last reading holds. Nothing is read, for nothing the page
        reads hangs on its forms; the name's own check is read after its own typing."""
        self.redraw()

    def refresh_local_gates(self) -> None:
        """The heading, what is in use, the table and the pages, from the last reading and what is in use now; no
        reading."""
        self._draw(read=False)
        super().refresh_local_gates()

    def _fit_figures(self) -> None:
        """The figures a check's run draws stand on its run's page alone: once it is over, the record's own stand in its
        result."""
        if self._body.indexOf(self.figure_stack) >= 0:
            self.figure_stack.setVisible(False)

    # ── the keyboard ──
    def _row_moved(self, current, _previous) -> None:
        """The keyboard moved to another row: that row whole in view. A drawing that keeps the row the keyboard is on
        moves no keyboard, and the page stays where it was scrolled."""
        view = self.listing.view()
        if (current.isValid() and view.hasFocus() and current.model() is view.model()
                and not self.listing.drawing):
            self._row_into_view(view, current)

    def _row_into_view(self, view, index) -> None:
        """The row ``index`` of ``view`` whole in the viewport, with its ring's room."""
        rect = view.visualRect(index)
        body, bar = self.scroller.body, self.scroller.verticalScrollBar()
        room = _ring_room()
        top = view.viewport().mapTo(body, QPoint(0, rect.top())).y() + body.y() - room
        bottom = top + rect.height() + 2 * room
        if bottom > bar.value() + self.scroller.viewport().height():
            bar.setValue(bottom - self.scroller.viewport().height())
        if top < bar.value():
            bar.setValue(top)

    def _keyboard_into_view(self) -> None:
        """As every page brings the control that holds the keyboard into view, and on the table the row it is on."""
        super()._keyboard_into_view()
        view = self.listing.view()
        held = self.window().focusWidget()
        if held is view and self.isVisible() and view.currentIndex().isValid():
            self._row_into_view(view, view.currentIndex())

    # ── the checks' runs ──
    def _start_band(self) -> None:
        """The frequency band, on a configuration of its own with the frequency response, the cell chosen loaded on it."""
        made = self._prepared("band", own=True, chi_mode=True, cell=True)
        if made is not None:
            cfg, arguments = made
            self.dispatch(probes.probe_band, cfg, provide_fig_sink=True, on_result=self._made_by("band"),
                          **arguments)

    def _start_audit(self) -> None:
        """The audit of the prior in use, on a configuration of its own with the frequency response; it reads no cell."""
        prior = self.session.inf_prior
        made = self._prepared("audit", own=True, chi_mode=True) if prior is not None else None
        if made is not None:
            cfg, arguments = made
            self.dispatch(probes.probe_mask, cfg, prior, provide_fig_sink=True, on_result=self._made_by("audit"),
                          **arguments)

    def _start_calibration(self) -> None:
        """The calibration of the posterior in use with the prior in use, at the form's sizes, its seed the form's or,
        left blank, one drawn afresh: the record keeps it, so the calibration can be repeated, and nothing here
        remembers it."""
        session = self.session
        if session.posterior is None or session.inf_prior is None:
            return                           # the facts say why, and the button is greyed with them
        made = self._prepared("calibration")
        if made is None:
            return
        cfg, arguments = made
        if arguments.get("seed") is None:
            arguments["seed"] = random.randrange(DRAWN_SEED_BELOW)
        # The region a narrowed posterior was narrowed to comes off its own network inside the stage.
        self.dispatch(orchestrator.validate_calibration, cfg, session.posterior, session.inf_prior,
                      provide_fig_sink=True, on_result=self._made_by("calibration"), **arguments)

    def _start_input_use(self) -> None:
        """The use of each part of the input by the posterior in use, from its own simulation set."""
        posterior = self.session.posterior
        made = self._prepared("input_use") if posterior is not None else None
        if made is not None:
            cfg, arguments = made
            self.dispatch(ablation.channel_ablation, cfg, posterior, provide_fig_sink=True,
                          on_result=self._made_by("input_use"), **arguments)

    def _start_identifiability(self) -> None:
        """How strongly the posterior in use pins each direction down, read from its record."""
        posterior = self.session.posterior
        made = self._prepared("identifiability") if posterior is not None else None
        if made is not None:
            cfg, arguments = made
            self.dispatch(identifiability.identifiability_rotation, cfg, posterior, provide_fig_sink=True,
                          on_result=self._made_by("identifiability"), **arguments)

    def _prepared(self, key: str, *, own: bool = False, chi_mode=None, cell: bool = False) -> "tuple | None":
        """What the press of the check ``key`` hands its stage: the configuration (the session's, or with ``own`` one of
        the check's own in the kind of observation ``chi_mode`` holds, None for the session's, with the cell chosen
        loaded on it when ``cell``) and the form's arguments with the record's name and note. A rule broken, a bounds
        file or a cell that cannot be read, is shown as every refusal is, and nothing is handed (None)."""
        form = self.check_pages[key].form
        try:
            arguments = check_forms.call_arguments(key, form, slots=self._slots())
        except Refusal as refused:
            self._refusal(refused)
            return None
        if form.names is not None:
            arguments.update(name=form.names.name_value(), note=form.names.note_value())
        if not own:
            return self.session.cfg, arguments
        picker = check_forms.cell_picker(form)
        cfg = self._own_config(chi_mode=chi_mode, cell=(picker.selected_path() or "") if cell else None)
        return None if cfg is None else (cfg, arguments)

    def _own_config(self, *, chi_mode, cell: "str | None"):
        """A configuration of the check's own, made as the session's was, from the step 1 settings applied and the bounds
        file in use or the limits typed by hand, on the session configuration's device: with the frequency response
        ``chi_mode`` (None: as the session's holds it), and the cell file ``cell`` loaded on it (None for none). A
        refusal is shown as every refusal is, a file that cannot be read as a refusal of its field; None then."""
        session = self.session
        held = session.cfg
        mode = bool(getattr(held, "chi_mode", False)) if chi_mode is None else bool(chi_mode)
        bounds = (getattr(held, "sources", None) or {}).get("bounds")
        try:
            draft = dataclasses.replace(session.draft, chi_mode=mode)
            made = (draft.make_config(str(bounds), hw=held.hw) if bounds
                    else draft.make_config(bounds_dicts=_limits_of(held), hw=held.hw))
        except Refusal as refused:
            self._refusal(refused)
            return None
        except Exception as error:           # noqa: BLE001 -- a file the builder cannot read is said, as a refusal
            self._refusal(Refusal(BOUNDS_UNREAD_TEXT.format(f"{type(error).__name__}: {error}"), field="bounds"))
            return None
        if cell is not None:
            try:
                cli.load_and_validate_gt(made, cell)
            except Refusal as refused:
                self._refusal(refused)
                return None
            except Exception as error:       # noqa: BLE001 -- a cell file that does not parse is said, as a refusal
                self._refusal(Refusal(CELL_UNREAD_TEXT.format(f"{type(error).__name__}: {error}"), field="cell"))
                return None
        return made

    def _made_by(self, key: str):
        """What a check's run that returned its record does: the name and the note of its form emptied, the name being
        taken now."""
        return functools.partial(self._record_made, key)

    def _record_made(self, key: str, _record) -> None:
        names = self.check_pages[key].form.names
        if names is not None:
            names.clear()

    def _check_of(self, fn) -> "str | None":
        """The check whose stage ``fn`` is, looked up where each stage lives now; None for none."""
        return next((key for key, (module, name) in _ENGINE.items() if fn is getattr(module, name, None)), None)

    def run_spec(self, fn, *args, **kwargs) -> RunSpec:
        """A check's run: named for its check, writing its one record, in its stages; it keeps the plan in use and the
        sizes it was made at, and no settings, which its record holds."""
        key = self._check_of(fn)
        if key is None:
            return super().run_spec(fn, *args, **kwargs)
        return check_forms.check_spec(key, sizes=check_forms.sizes_from_call(key, kwargs),
                                      plan=self._screen.plan_entry() if self._screen is not None else None)


def _limits_of(cfg) -> tuple:
    """The limits a configuration was made on, as limits typed by hand are handed to make one: each section's names in
    their order, each with its two ends and no value."""
    return tuple(OrderedDict((name, (None, tuple(ends))) for name, (_value, ends) in section.items())
                 for section in (cfg.params_dict, cfg.rescale_params, cfg.force_params_dict))
