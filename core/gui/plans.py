"""Plans in the window: a model's plans listed, one read with the two checks its reader cannot make, and one
saved.

The engine reads a plan and writes its text (``core/plans.py``); the window lists the folder, checks a plan against the
model's own folders and writes the file. Every function here that lists, reads or
writes takes its folder as an argument: the window hands the folder of the model shown on step 1 under ``PLAN_FOLDER``,
and a test hands a folder of its own, so that no test lists or writes the real one.

Nothing here reads the disk but ``list_plans``, ``read_checked``, ``save_plan``, ``taken`` and ``file_sha``, and a page
calls them only at its readings and its presses, never while a run is live.
"""
import hashlib
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QUrl
from PySide6.QtGui import QDesktopServices

from core import cli, config
from core import plans as engine_plans
from core.Helpers import file_manager
from core.refusals import Refusal


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
UNITS_TEXT = "The units are not usable: {}"
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


def read_checked(path, *, bounds_folder) -> engine_plans.Plan:
    """The plan at ``path``, read whole by the engine's reader, then held to the two checks that take more than the
    file: its bounds file stands in ``bounds_folder``, the model's own, and its units resolve. A ``PlanRefusal``, its
    field ``plan``, for every problem at once: the reader's, or else the window's, in one list in the reader's form."""
    path = Path(path)
    plan = engine_plans.read_plan(path)
    problems = []
    bounds = plan.values.get(("Prior", "bounds"))
    if bounds is not None and bounds_key(bounds, bounds_folder) is None:
        problems.append(engine_plans.PlanProblem(None, "Prior", "bounds", NOT_IN_FOLDER_TEXT.format(bounds)))
    units = plan.values.get(("Model and options", "units"))
    if units is not None:
        try:
            cli.units_to_factors(tuple(units.split()))
        except Refusal as refused:
            problems.append(engine_plans.PlanProblem(None, "Model and options", "units",
                                                     UNITS_TEXT.format(refused.message)))
        except Exception as error:           # noqa: BLE001 -- what a units library raises on a word it cannot read
            problems.append(engine_plans.PlanProblem(None, "Model and options", "units", UNITS_TEXT.format(error)))
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
