"""Plans in the window: a model's plans listed, one read with the two checks its reader cannot make, one saved, the
forms of steps 1 to 3 filled from one, where each field's value came from, and what "Before you start" says of the plan
in use.

The engine reads a plan and writes its text (``core/plans.py``); the window lists the folder, checks a plan against the
model's own folders, writes the file, fills the forms and says what is in use. Every function here that lists, reads or
writes takes its folder as an argument: the window hands the folder of the model shown on step 1 under ``PLAN_FOLDER``,
and a test hands a folder of its own, so that no test lists or writes the real one.

Filling is the only thing a plan does. It never starts a run, never applies a step, never changes which record is
picked, and never replaces the session: the forms change, and what the steps hold reads "Changed, not applied" where
they now differ from it. The source of what was filled, a plan's name, path and hash and the values as filled, is kept
by the inference screen, which outlives the session a step 1 applied replaces.

A page hands the plan code its rows through ``plan_fields``, each with ``value``, ``set_value``, ``usual``,
``set_origin``, ``label_text`` and ``kept``, and says through ``plan_read`` what its run reads: the values of its own
form, or, for a step the page's run follows, what that step holds. Nothing here reads the disk but ``list_plans``,
``read_checked``, ``save_plan``, ``taken`` and ``file_sha``, and ``plan_path`` and ``bounds_key``, which only those and
the page's readings call; a page calls them only at its readings and its presses, never while a run is live.
"""
import hashlib
import os
import re
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import QComboBox, QLineEdit

from core import cli, config
from core import plans as engine_plans
from core.artifacts.provenance import file_ref
from core.Helpers import file_manager
from core.refusals import Refusal

from . import texts
from .run.controller import controller
from .step_facts import Fact, origin_words
from .widgets.form import TickedField

#: The folder whose sub-folders, one for each model, hold the plans. A test hands the screen a folder of its own, and
#: the suites' session binds this one to a folder of their own.
PLAN_FOLDER: Path = config.PLANS_PATH
#: The heading of a plan each step's form takes its values from.
HEADING_OF = {"model": "Model and options", "prior": "Prior", "training": "Training"}
#: The headings a fill reaches, in their order: the steps whose pages are of the look. The later headings are read and
#: shown, and not filled.
FILLED = tuple(HEADING_OF.values())
#: A plan's name: what the file's name may be when the window saves one.
NAME_RULE = re.compile(r"[A-Za-z0-9_-]{1,40}")
NAME_TEXT = "A plan's name is letters, digits, _ and -, at most 40 characters."
NOT_IN_FOLDER_TEXT = "The bounds file {} is not in this model's bounds folder."
OTHER_CASE_TEXT = ("The bounds file {} is not in this model's bounds folder; {} is. A plan names its file with the "
                   "same capitals.")
UNITS_TEXT = "The units are not usable: {}"
#: What the plan line under each heading says.
NO_PLAN = "No plan"
PLAN_LINE = "Plan: {}"
SOURCE_LINE = "Settings from {} {}"
ONE_CHANGE = "{}, 1 change"
CHANGES = "{}, {} changes"
#: What "Before you start" says of the plan in use, the source's head first.
PLAN_HEAD = "Plan {}"
SOURCE_HEAD = "Settings from {} {}"
NO_PLAN_USUAL_TEXT = "No plan. Every setting is at its usual value."
NO_PLAN_ONE_TEXT = "No plan. 1 setting differs from the usual: {}."
NO_PLAN_DIFFER_TEXT = "No plan. {} settings differ from the usual: {}."
AS_WRITTEN_ONE_TEXT = "{}. Its 1 setting is in use as written."
AS_WRITTEN_TEXT = "{}. Its {} settings are in use as written."
NONE_READ_TEXT = "{}. This step reads none of its settings."
ONE_CHANGE_TEXT = "{}, with 1 change: {}."
CHANGES_TEXT = "{}, with {} changes: {}."
DEPARTURE = "{} {} ({} {})"
CHANGED_FILE_TEXT = ("{} has been changed since its settings were filled in on {}. The forms hold the earlier values; "
                     "fill again to take the changes.")
GONE_FILE_TEXT = "{} is no longer in the plans folder. The forms hold the values filled from it on {}."
#: The two, for a source that keeps no moment of its fill.
CHANGED_UNDATED_TEXT = ("{} has been changed since its settings were filled in. The forms hold the earlier values; "
                        "fill again to take the changes.")
GONE_UNDATED_TEXT = "{} is no longer in the plans folder. The forms hold the values filled from it."
#: How a value that has none of its own is said.
BLANK = "blank"
NONE_WORDS = "none"
TYPED_WORDS = "limits typed by hand"


class _Typed:
    """The bounds a form or a configuration holds when its limits were typed by hand: no file, and no file's name."""

    def __repr__(self) -> str:
        return "TYPED"


#: The value of the bounds row while its limits are typed by hand.
TYPED = _Typed()


class _Unread:
    """What the screen knows of the plan file in use before a reading has looked at it."""

    def __repr__(self) -> str:
        return "UNREAD"


#: The state of the plan file in use while no reading has looked at it.
UNREAD = _Unread()
#: What a plan's absence of a value is told apart from: a plan never holds None.
_MISSING = object()
#: The rows a fill leaves as they are when the plan holds nothing for them: a bounds file has no usual value.
LEFT_ALONE = ("bounds",)


@dataclass(frozen=True)
class PlanSource:
    """Where what the forms were filled with came from: ``kind`` "plan", "record", "run", or "usual" for the press that
    returns the forms to their usual values, which leaves no source in use; ``name``; for a plan its file and that
    file's ``sha256`` as it was filled; ``at``, when; ``values``, keyed as a plan's are, by heading and name, as filled;
    ``ident``, a record's or a run's own id."""
    kind: str
    name: str
    path: "Path | None"
    sha256: str
    at: "datetime | None"
    values: dict
    ident: str = ""


# ── the folder and the file ──
def model_folder(root, model: str) -> Path:
    """The folder of ``model``'s plans under ``root``: one folder for each model, named in lower case."""
    return Path(root) / str(model).lower()


def _plan_files(folder: Path) -> dict:
    """Every plan of ``folder``, by its name, the file's own: each ``.txt`` file, its suffix read without case."""
    try:
        files = [entry for entry in Path(folder).iterdir() if entry.suffix.lower() == ".txt" and entry.is_file()]
    except OSError:
        return {}
    return {entry.stem: entry for entry in files}


def list_plans(folder) -> tuple[str, ...]:
    """The names of the plans in ``folder``, in the order of the alphabet without regard to case: every ``.txt`` file,
    a hand-named one too, whatever it holds; none while the folder does not exist."""
    return tuple(sorted(_plan_files(Path(folder)), key=lambda name: (name.casefold(), name)))


def plan_path(folder, name: str) -> Path:
    """The file of the plan ``name`` in ``folder``, as the folder names it; the name with ``.txt`` when none is
    there."""
    return _plan_files(Path(folder)).get(name, Path(folder) / f"{name}.txt")


def _refused(path: Path, problems: list) -> engine_plans.PlanRefusal:
    count = f"{len(problems)} problem{'' if len(problems) == 1 else 's'}"
    return engine_plans.PlanRefusal(f"This plan cannot be used: {count} in {path.name}", tuple(problems))


def bounds_key(name: str, folder) -> "str | None":
    """The entry of the bounds list that a plan's bounds name stands for, as the list keeps it, with the system's own
    separators; None when the folder's listing holds no file of exactly that name. Judged against the listing, never by
    asking the file system for the name: Windows finds MASTER.TXT for master.txt, and the list would select neither."""
    folder = Path(folder)
    if not folder.is_dir():
        return None
    return next((entry for entry in file_manager.list_dir(str(folder)) if entry.replace("\\", "/") == name), None)


def _in_another_case(name: str, folder) -> "str | None":
    """The one file of the folder's listing whose name differs from ``name`` only in case, with forward slashes; None
    when there is none, or more than one."""
    folder = Path(folder)
    if not folder.is_dir():
        return None
    found = [entry.replace("\\", "/") for entry in file_manager.list_dir(str(folder))
             if entry.replace("\\", "/").casefold() == name.casefold()]
    return found[0] if len(found) == 1 else None


def window_problems(values: dict, *, bounds_folder) -> list:
    """The two checks that take more than a plan's text, on its ``values``: its bounds file stands in
    ``bounds_folder``, the model's own, and its units resolve. Each problem in the reader's form, in that order."""
    problems = []
    bounds = values.get(("Prior", "bounds"))
    if bounds is not None and bounds_key(bounds, bounds_folder) is None:
        other = _in_another_case(bounds, bounds_folder)
        said = NOT_IN_FOLDER_TEXT.format(bounds) if other is None else OTHER_CASE_TEXT.format(bounds, other)
        problems.append(engine_plans.PlanProblem(None, "Prior", "bounds", said))
    units = values.get(("Model and options", "units"))
    if units is not None:
        try:
            cli.units_to_factors(tuple(units.split()))
        except Refusal as refused:
            problems.append(engine_plans.PlanProblem(None, "Model and options", "units",
                                                     UNITS_TEXT.format(refused.message)))
        except Exception as error:           # noqa: BLE001 -- what a units library raises on a word it cannot read
            problems.append(engine_plans.PlanProblem(None, "Model and options", "units", UNITS_TEXT.format(error)))
    return problems


def read_checked(path, *, bounds_folder) -> engine_plans.Plan:
    """The plan at ``path``, read whole by the engine's reader, then held to the window's two checks
    (``window_problems``). A ``PlanRefusal``, its field ``plan``, for every problem at once: the reader's, or else the
    window's, in one list in the reader's form."""
    path = Path(path)
    plan = engine_plans.read_plan(path)
    problems = window_problems(plan.values, bounds_folder=bounds_folder)
    if problems:
        raise _refused(path, problems)
    return plan


def taken(folder, name: str) -> "Path | None":
    """The plan a save under ``name`` would replace: one of the folder's, its name matched without regard to case, for
    Windows keeps one file for both; None when there is none."""
    wanted = name.casefold()
    return next((path for stem, path in _plan_files(Path(folder)).items() if stem.casefold() == wanted), None)


def save_plan(folder, name: str, text: str) -> Path:
    """Write ``text`` as the plan ``name`` in ``folder``, made when it is not there: into a file beside it first,
    which is then moved into place, so that a plan is never left half written. The plan's file, as the folder now names
    it."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / f"{name}.txt"
    handle, beside = tempfile.mkstemp(prefix=f".{name}.", suffix=".part", dir=folder)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as out:
            out.write(text)
        os.replace(beside, target)
    except BaseException:
        Path(beside).unlink(missing_ok=True)
        raise
    return target


def _hash_of(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def file_sha(path, cache: dict) -> "str | None":
    """The sha256 of the plan file at ``path``, None when it is no longer there. Read once and kept in ``cache`` until
    the file's moment of writing or its size changes: a page reads the plan in use at each of its readings."""
    path = Path(path)
    try:
        held = os.stat(path)
    except OSError:
        cache.pop(path, None)
        return None
    key = (held.st_mtime_ns, held.st_size)
    kept = cache.get(path)
    if kept is not None and kept[0] == key:
        return kept[1]
    try:
        sha = _hash_of(path)
    except OSError:
        cache.pop(path, None)
        return None
    cache[path] = (key, sha)
    return sha


def open_folder(folder) -> None:
    """Show ``folder`` in Windows' own file window."""
    QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))


# ── values and their words ──
def plan_value(source: "PlanSource | None", heading: str, name: str):
    """The value the source in use holds for ``name`` under ``heading``, or None when it holds none."""
    return None if source is None else source.values.get((heading, name))


def same(one, other) -> bool:
    """Whether two values of a setting are the same value: units by their words, a whole number and a number by what
    they count, and nothing else by anything but equality. A truth value is never a number here."""
    if isinstance(one, str) and isinstance(other, str):
        return one.split() == other.split()
    if isinstance(one, bool) or isinstance(other, bool):
        return one is other
    if one is TYPED or other is TYPED:
        return one is other
    try:
        return bool(one == other)
    except Exception:                        # noqa: BLE001 -- a value no comparison can be made with is another one
        return False


def _lowered(words: str) -> str:
    return words[:1].lower() + words[1:]


def _number(value) -> str:
    """A number as a sentence says it, whole: grouped, or in the shortest form that is exactly it."""
    if isinstance(value, int):
        return f"{value:,}"
    said = repr(value)
    return said[:-2] if said.endswith(".0") else said


def typed_words(row) -> str:
    """What stands in a row's field as a person typed it, "" for a list, a tick, an empty field or a row with none."""
    widget = getattr(row, "widget", None)
    if isinstance(widget, TickedField):
        return "" if widget.is_ticked() else widget.field.text().strip()
    return widget.text().strip() if isinstance(widget, QLineEdit) else ""


def value_words(row, value) -> str:
    """``value`` of ``row`` as a sentence says it: a row's own words when it has a way of saying its values; a tick's
    words for the value it stands for; a list's item for its value, the first letter small; a number grouped, or
    exactly; nothing as "blank"."""
    said = getattr(row, "words_of", None)
    if callable(said):
        return said(value)
    widget = getattr(row, "widget", None)
    if isinstance(widget, TickedField) and value == widget.stands_for:
        return _lowered(widget.ticked_box.text())
    if isinstance(widget, QComboBox):
        found = widget.findData(value)
        if found >= 0:
            return _lowered(widget.itemText(found))
    if value is None:
        return BLANK
    if isinstance(value, bool):
        return "on" if value else "off"
    if isinstance(value, (int, float)):
        return _number(value)
    return str(value)


def shown_words(row, value) -> "str | None":
    """The words the form shows for ``value`` where they differ from the file's: a list's item, a tick's words for the
    value it stands for; None where the file writes what the form shows. A saved plan's remark carries them."""
    widget = getattr(row, "widget", None)
    if isinstance(widget, TickedField) and value == widget.stands_for:
        return widget.ticked_box.text()
    if isinstance(widget, QComboBox):
        found = widget.findData(value)
        written = "on" if value is True else "off" if value is False else str(value)
        if found >= 0 and widget.itemText(found).lower() != written.lower():
            return widget.itemText(found)
    return None


def origin(row, source: "PlanSource | None", *, kept: bool, heading: str) -> str:
    """Where ``row``'s value came from, as the words beside it say: the source in use holds the setting and the form has
    its value, or has another; else put back from the settings file and not touched since; else the form has the usual
    value, or another. A bounds file has no usual value: one the source does not hold and nothing put back says
    nothing."""
    held = plan_value(source, heading, row.key)
    if held is None and row.key in LEFT_ALONE and not kept:
        return ""
    return origin_words(held=held is not None, at_source=held is not None and same(row.value(), held),
                        at_usual=same(row.value(), row.usual()), kept=kept,
                        source=getattr(source, "kind", "plan"), name=getattr(source, "name", ""))


# ── filling ──
def fill_forms(screen, values: dict, source: PlanSource, keep_remembered: bool = False) -> bool:
    """Fill the forms of steps 1 to 3: every row each page hands through ``plan_fields`` takes the value ``values``
    holds for it under the page's heading, else its usual value; a bounds file the values do not name is left as it is.
    With ``keep_remembered`` a row put back from the settings file and not touched since is left as it is. Nothing is
    asked, nothing applied, nothing started, no record's pick changed, and no session replaced: the source is kept by
    the screen, and the pages are drawn again. While a run is live nothing changes: False."""
    if controller().is_live():
        return False
    for page in screen.step_pages():
        heading = HEADING_OF.get(page.step)
        for name, row in page.plan_fields().items():
            if keep_remembered and row.kept:
                continue
            value = values.get((heading, name), _MISSING)
            if value is _MISSING:
                if name in LEFT_ALONE:
                    continue
                value = row.usual()
            row.set_value(value)
            row.kept = False
    screen.take_plan(None if source.kind == "usual" else source)
    return True


def would_replace(screen, values: dict) -> tuple[str, ...]:
    """What a fill with ``values`` would change that a person typed or that was put back: each row of steps 1 to 3 whose
    value is neither its usual one nor the source's in use, and that the fill would set to another, as "Batches: 9,500
    becomes 10,000", in the steps' order."""
    source, changed = screen.plan_source, []
    for page in screen.step_pages():
        heading = HEADING_OF.get(page.step)
        for name, row in page.plan_fields().items():
            new = values.get((heading, name), _MISSING)
            if new is _MISSING:
                if name in LEFT_ALONE:
                    continue
                new = row.usual()
            now = row.value()
            held = plan_value(source, heading, name)
            if same(now, new) or same(now, row.usual()) or (held is not None and same(now, held)):
                continue
            said = typed_words(row) if now is None else ""
            changed.append(f"{row.label_text()}: {said or value_words(row, now)} becomes {value_words(row, new)}")
    return tuple(changed)


# ── what is said ──
def headings_to(step: str) -> tuple[str, ...]:
    """The headings a step's run reads: its own and every earlier step's."""
    order = list(HEADING_OF)
    return tuple(HEADING_OF[key] for key in order[:order.index(step) + 1]) if step in HEADING_OF else ()


def _departures(screen, headings: Sequence[str]) -> tuple[int, list]:
    """How many of the settings the run reads the source in use holds, and every departure among them, in the steps'
    order: against the source's value where it holds the setting, else against the usual one. The last heading is the
    page's own, read from its form, where a row in error is said by its own line and is no departure; the others are
    what their steps hold."""
    source = screen.plan_source
    pages = {HEADING_OF.get(page.step): page for page in screen.step_pages()}
    held_count, departures = 0, []
    for heading in headings:
        page = pages.get(heading)
        if page is None:
            continue
        fields = page.plan_fields()
        for name, value in page.plan_read(applied=heading != headings[-1]).items():
            row = fields.get(name)
            if row is None or (heading == headings[-1] and _in_error(row)):
                continue
            held = plan_value(source, heading, name)
            if held is None and name in LEFT_ALONE:
                continue
            against = held if held is not None else row.usual()
            held_count += held is not None
            if not same(value, against):
                departures.append(DEPARTURE.format(row.label_text(), value_words(row, value),
                                                   "usual" if held is None else source.kind, value_words(row, against)))
    return held_count, departures


def _in_error(row) -> bool:
    """Whether the form's row breaks its rule: its own line says so, and it is no departure."""
    ask = getattr(row, "in_error", None)
    return bool(ask()) if callable(ask) else False


def _head(source: PlanSource) -> str:
    return PLAN_HEAD.format(source.name) if source.kind == "plan" else SOURCE_HEAD.format(source.kind, source.name)


def plan_facts(screen, headings: Sequence[str]) -> tuple[Fact, ...]:
    """What "Before you start" says of the plan in use for a run that reads ``headings``: the source, and every
    departure among the settings the run reads, each named; and, for a plan, that its file has been changed or removed
    since the fill, as the last reading found it. Reads nothing."""
    source = screen.plan_source
    held, departures = _departures(screen, headings)
    listed = "; ".join(departures)
    if source is None:
        sentence = (NO_PLAN_USUAL_TEXT if not departures else NO_PLAN_ONE_TEXT.format(listed) if len(departures) == 1
                    else NO_PLAN_DIFFER_TEXT.format(len(departures), listed))
        return (Fact("info", sentence),)
    head = _head(source)
    if departures:
        sentence = (ONE_CHANGE_TEXT.format(head, listed) if len(departures) == 1
                    else CHANGES_TEXT.format(head, len(departures), listed))
    else:
        sentence = (NONE_READ_TEXT.format(head) if held == 0 else AS_WRITTEN_ONE_TEXT.format(head) if held == 1
                    else AS_WRITTEN_TEXT.format(head, held))
    facts = [Fact("info", sentence)]
    state = getattr(screen, "plan_file", UNREAD)
    if source.kind == "plan" and state is not UNREAD and source.path is not None:
        when, name = (texts.date_text(source.at) if source.at is not None else None), source.path.name
        if state is None:
            facts.append(Fact("warn", GONE_UNDATED_TEXT.format(name) if when is None
                              else GONE_FILE_TEXT.format(name, when)))
        elif state != source.sha256:
            facts.append(Fact("warn", CHANGED_UNDATED_TEXT.format(name) if when is None
                              else CHANGED_FILE_TEXT.format(name, when)))
    return tuple(facts)


def plan_line(screen, headings: Sequence[str]) -> str:
    """The quiet line under a step's heading: "No plan", "Plan: retrain", or with the departures its "Before you start"
    names, "Plan: retrain, 2 changes"."""
    source = screen.plan_source
    if source is None:
        return NO_PLAN
    head = PLAN_LINE.format(source.name) if source.kind == "plan" else SOURCE_LINE.format(source.kind, source.name)
    count = len(_departures(screen, headings)[1])
    return head if count == 0 else ONE_CHANGE.format(head) if count == 1 else CHANGES.format(head, count)
