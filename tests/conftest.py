"""Fixtures for tests/ (the root conftest.py sets the environment and the marker skips).

`qapp` is the one QApplication a process may hold; ``tests/_fixtures.qt_app()`` is the same
expression as a plain call, which is what the GUI suites use. The settings, dialog and drawing-code
fixtures at the end, the two that stand in for the entry point's plain Windows message and put back
the streams it binds, the one that lets go what the builders made, the one that stands in for
bringing a window to the front and the one that stands in for the requests made of Windows, are
autouse at session and function scope, and the check that no panel outlives its module at module
scope; a test requests the last two by name, ``brought_forward`` and ``windows_asked``, to read what
each kept, and no other by name.
"""
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def qapp():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


@pytest.fixture(scope="session", autouse=True)
def _sandbox_default_store(tmp_path_factory):
    """Every stage now WRITES its artifact, so the whole session runs against a temp root; the real
    Artifacts/ must gain nothing. Replaces the per-suite PERSIST_OBSERVATIONS / _CKPT_DIRS_AT_IMPORT
    sandboxing the runners used to carry."""
    from core import config
    from core.artifacts import ArtifactStore, set_default_store
    real = config.artifacts_root()
    before = {p for p in real.rglob("*")} if real.exists() else set()
    set_default_store(ArtifactStore(tmp_path_factory.mktemp("artifacts")))
    yield
    after = {p for p in real.rglob("*")} if real.exists() else set()
    assert after == before, f"the suite wrote into {real}: {sorted(map(str, after - before))[:5]}"


@pytest.fixture(scope="session", autouse=True)
def _no_sbi_logs():
    """<repo>/sbi-logs must not exist before or after the session. This sits beside
    _sandbox_default_store for the same reason: the suites leave the repository as they found it.

    sbi's default TensorBoard writer made <cwd>/sbi-logs/NPE_C/<timestamp>/ on every training, and
    every gate trains at the repo root (tiny_run's build_posterior), which is how 1360 directories
    piled up there. core/SBI/train.py now hands sbi _NoTracker, the tree was deleted, and its
    .gitignore line went. So the directory coming back means a training reached sbi's default writer
    again, and with no ignore line it would also sit in git status.

    It is checked at SETUP as well: a tree left by some earlier run would otherwise make the teardown
    blame this session for it. A training started from another working directory is covered by
    tests/test_user_sbi.py::test_training_creates_no_sbi_logs_directory."""
    from core import config
    logs = config.REPO_ROOT / "sbi-logs"
    assert not logs.exists(), (
        f"{logs} exists before the session: delete it (nothing in PRISM writes it any more), "
        f"or this session's teardown check would blame the tests for it")
    yield
    assert not logs.exists(), (
        f"the suite created {logs}: a training reached sbi's default TensorBoard writer instead of "
        f"core.SBI.train._NoTracker (train_nn's tracker=None)")


def check_plans_kept(folder: Path, held: "list[str] | None", kept: "dict[str, bytes]") -> None:
    """Fail, by raising, when ``folder``, where the plans a person saves are kept, does not hold what it held at the
    start of the run: ``held`` is its listing then (tests/_settings_folder.listing) and ``kept`` its files' bytes then.
    The message names the first five of what changed: the folder itself comes first when it was made or has gone, so
    that a folder empty before or after is not read as nothing; then a plan rewritten under its own name, marked so,
    since it is the one change that loses work; then the paths that came or went. A plan the person saved before the
    run is in both and does no harm; one saved by hand while the run was going fails the same way, and the message says
    so.
    Raised rather than asserted, so the check survives a run with assertions switched off."""
    from tests import _settings_folder
    holds = _settings_folder.listing(folder)
    changed = [f"{path} (rewritten)" for path, was in kept.items()
               if path in (holds or ()) and Path(path).read_bytes() != was]
    changed += sorted(set(holds or ()) ^ set(held or ()))
    if (holds is None) != (held is None):
        changed.insert(0, f"the folder itself ({'gone' if holds is None else 'made'})")
    if changed:
        raise AssertionError(f"the suite changed what the real plans folder {folder} holds, or a plan was saved by "
                             f"hand while the run was going, so look at the paths before blaming the code: "
                             f"{changed[:5]}")


@pytest.fixture(scope="session", autouse=True)
def _no_plans_written(tmp_path_factory):
    """The folder of saved plans, ``Resources/Plans``, holds at the end of the session exactly what it held at its
    start: the same paths, and each file the same bytes. It sits beside _no_sbi_logs for the same reason: the suites
    leave the repository as they found it. A plan a test wrote there would be read by the next launch as one of the
    person's own, and the bytes are compared, as _settings_home compares the real settings file's, because a plan
    written over leaves the listing as it was and cannot be brought back. The folder may be absent (it is until a plan
    is saved), and a plan the person saved before the run is kept here as it stands and does no harm; a folder that
    appears fails the run, and so does a plan saved by hand while the run is going. The listing is
    tests/_settings_folder.py's and the check is ``check_plans_kept`` above. For the session the window's folder of
    plans is one of the session's own, empty, so that a window built by any test lists there, and the real one is never
    read; it is put back at the end, and it must hold nothing then: a test that writes a plan binds a folder of its own
    with ``monkeypatch``, which puts this one back as the test ends, and ``_streams_put_back`` fails a test that left
    another bound."""
    from core import config
    from core.gui import plans
    from tests import _settings_folder
    folder = config.RESOURCES_ROOT / "Plans"
    held = _settings_folder.listing(folder)
    kept = {path: Path(path).read_bytes() for path in held or () if Path(path).is_file()}
    real, plans.PLAN_FOLDER = plans.PLAN_FOLDER, tmp_path_factory.mktemp("plans")
    session = plans.PLAN_FOLDER
    try:
        yield
    finally:
        plans.PLAN_FOLDER = real
    check_plans_kept(folder, held, kept)
    written = sorted(str(path.relative_to(session)) for path in session.rglob("*"))
    assert not written, f"the suites' own folder of plans was written to: {written}"


@pytest.fixture
def store(tmp_path):
    """A fresh store that is ALSO the process default for the duration of the test."""
    from core.artifacts import ArtifactStore, use_store
    with use_store(ArtifactStore(tmp_path / "Artifacts")) as s:
        yield s


@pytest.fixture(scope="module")
def tiny_run(tmp_path_factory):
    """A real prior + posterior for the SBITEST user model at tiny size, in a store of its own that is
    the process default for the module. Minutes on CPU, built once per module."""
    from core.artifacts import ArtifactStore, use_store
    from tests._fixtures import build_tiny_run
    with use_store(ArtifactStore(tmp_path_factory.mktemp("tiny"))) as s:
        run = build_tiny_run(s)
        try:
            yield run
        finally:
            run.teardown()


@pytest.fixture(scope="module")
def screen_run(tiny_run):
    """An InferenceScreen whose session holds the tiny run's prior and posterior and a TRUTH-FREE
    copy of its config -- the window's own shape: a session config is bounds-built at Build/Load prior
    and never carries a cell. A COPY, because tiny_run is module-scoped and its siblings in the same
    file need the truth it injected; a truth-carrying session would also make the pin vacuous, since
    the composition re-injects the very same cell on its copy.

    Module-scoped like tiny_run, so a file pays for one screen. Its panels restore from the session's
    temporary settings file (_settings_home), never from the real PRISM.ini. The screen is let go when
    its module ends, by its own deferred delete: left alive, its panels would stay among the panels
    every later run locks and frees, and a later module's run would refresh them at its end, reading
    whatever store that module has in use."""
    from types import SimpleNamespace
    from PySide6.QtCore import QCoreApplication, QEvent
    from tests._fixtures import build_inference, qt_app
    qt_app()
    cfg = tiny_run.cfg.copy_for_run()
    cfg.clear_ground_truth()
    inf = build_inference(cfg=cfg)
    inf.session.inf_prior = tiny_run.prior
    inf.session.posterior = tiny_run.posterior
    inf.refresh_gates()
    yield SimpleNamespace(screen=inf, run=tiny_run, cell=tiny_run.cfg.sources["cell"])
    inf.deleteLater()
    QCoreApplication.sendPostedEvents(inf, QEvent.DeferredDelete)


@pytest.fixture(scope="session", autouse=True)
def _checkpointing_off_unless_asked():
    """Training-data checkpointing OFF for the whole session. A test that wants a simulation cache
    passes ``checkpoint_every=`` to build_posterior explicitly.

    A TEST-INTEGRITY guard, not housekeeping. Left on, the full-pipeline tests write real caches keyed
    on a digest of their config -- and a COMPLETE cache short-circuits generation and returns its
    stored rows. So the FIRST run would create them and every run afterwards would silently skip
    gen_training_data entirely while the suite stayed green. Under the rule that refuses a cache one
    setting away before any simulation, a stray cache is worse still: a committed sibling one identity
    field away REFUSES a later run instead of quietly restarting it, so one test's leftovers would fail
    another's.

    Rebound on ORCHESTRATOR, not on config: orchestrator does ``from .config import
    TRAINING_CHECKPOINT_EVERY`` at import and would otherwise keep its snapshot. SESSION-scoped, which
    is the point of moving it here: it used to be an import-time assignment in
    tests/test_user_sbi.py, so it applied only when that file was collected and a single-file run
    behaved differently from the gate.
    """
    from core import orchestrator
    mp = pytest.MonkeyPatch()
    mp.setattr(orchestrator, "TRAINING_CHECKPOINT_EVERY", 0)
    yield
    mp.undo()


@pytest.fixture(scope="session", autouse=True)
def _settings_home(tmp_path_factory):
    """The settings location is NEVER the real PRISM.ini during a test process, at any scope. This
    is the session half: a path under the session temp that does not exist until Qt writes it,
    installed before any module- or session-scoped fixture can build a panel (pytest sets higher
    scopes up first), so a panel built by tiny_run's siblings or by screen_run restores from an empty
    file too. The function half is _isolated_settings, below.

    Teardown asserts the developer's real user-scope file is byte-identical, mirroring
    _sandbox_default_store's assertion on the real Artifacts/: a saved value winning over config.py is
    the mechanism that cost a ~5-day run on 2026-08-19 (the retired chi band), and a test that wrote
    one would arm it for the next real launch. The real path is resolved the way settings() resolves
    it with no override; constructing that QSettings neither creates nor rewrites the file.

    The folder that holds the real file must hold at the end exactly what it held at the start, read
    as a listing of every path under it: whatever the window writes beside its settings file goes
    beside the test's own, and a file it left beside the real one would be read by the next launch. The
    listing and its check are tests/_settings_folder.py's, which a test tries on a folder of its own.
    """
    from PySide6.QtCore import QSettings
    from core.gui import settings
    from tests import _settings_folder
    real = Path(QSettings(QSettings.IniFormat, QSettings.UserScope, settings.ORG, settings.APP).fileName())
    folder = real.parent
    before = real.read_bytes() if real.exists() else None
    held = _settings_folder.listing(folder)
    settings.use_ini_file(str(tmp_path_factory.mktemp("settings") / "prism.ini"))
    yield
    settings.use_ini_file(None)
    after = real.read_bytes() if real.exists() else None
    assert after == before, f"the suite wrote into the real settings file {real}"
    _settings_folder.check_kept(folder, held)


@pytest.fixture(autouse=True)
def _isolated_settings(tmp_path):
    """A FRESH .ini per test -- the primitive; a shared file cleared per test would fight QSettings'
    file cache. Teardown restores the SESSION path _settings_home installed, never None: None is the
    real file, and a module fixture set up after this test would otherwise read it. A test that
    asserts a default therefore asserts config.py, never the developer's last session, and the
    _temp_settings helper the suites used to carry per file is gone."""
    from core.gui import settings
    session_path = settings._override_path
    settings.use_ini_file(str(tmp_path / "prism.ini"))
    yield
    settings.use_ini_file(session_path)


@pytest.fixture(scope="session", autouse=True)
def _no_modal_dialogs():
    """Offscreen, the exec() of a message box or of the shared question spins a nested event loop that
    nothing ever closes, so a dialog a
    test did not expect is a STALL past the ten-minute tool-call limit (no timeout plugin is
    installed), not a failure. For the whole session the dialog is handed to tests/_dialogs.record, which
    keeps it in SHOWN with its words, and exec returns 0 -- no clicked button, which the consent
    dialogs read as Cancel and ModelsScreen._delete_user_model's confirmation as the safe branch. A test
    that wants an answer calls tests/_dialogs.choose, which layers a fake that clicks on top with the
    test's own monkeypatch; the undo puts this guard back, because a MonkeyPatch on a class records the
    __dict__ entry, which here is one guard function set on both classes, so that a test can tell the
    guard stands on each. exec belongs to neither class but to their base, QDialog, so each class has to
    be given the attribute; a dialog class added later is not covered until it is named here. Same
    MonkeyPatch shape as _checkpointing_off_unless_asked. Nothing
    under core/ calls the statics any more: the last three, now Your own models', go through
    ModelsScreen._tell, so this guard covers every box the GUI shows and a test reads one with
    tests/_dialogs.asked() -- and that is a FACT WITH A PIN, not a claim: tests/test_refusals.py::
    test_no_static_message_box_is_called_anywhere_under_core scans CODE_ROOTS + CODE_FILES for one,
    because a static added here would HANG this suite rather than fail it."""
    from PySide6.QtWidgets import QMessageBox
    from core.gui.widgets.ask import Ask
    from tests._dialogs import record
    guard = lambda self: record(self) or 0
    mp = pytest.MonkeyPatch()
    mp.setattr(QMessageBox, "exec", guard)
    mp.setattr(Ask, "exec", guard)
    mp.setattr(QMessageBox, "exec_", guard)      # the binding's older spelling of the same call
    mp.setattr(Ask, "exec_", guard)
    yield
    mp.undo()


@pytest.fixture(autouse=True)
def _clear_shown():
    """The dialog record holds this test's boxes only, so asked() is never a leftover from an earlier
    test."""
    from tests._dialogs import forget_asked
    forget_asked()
    yield


@pytest.fixture(scope="session", autouse=True)
def _no_file_choosers():
    """Offscreen a file chooser stalls the run exactly as a message box does, and QFileDialog's
    statics are not reached by a patched ``exec``. For the whole session the two the window calls,
    getSaveFileName and getOpenFileName, are tests/_dialogs.chooser_guard: the call is recorded in
    CHOOSER_CALLS and returns as cancelled unless the test answered it with answer_chooser. A test's
    own fake layered with monkeypatch wins while installed, and its undo puts this guard back.
    The save button of matplotlib's toolbar in a popped-out figure calls the same getSaveFileName,
    so that chooser is covered too. Not covered: the directory and several-files choosers, which
    nothing under core/ calls, and that toolbar's other dialogs, which matplotlib opens through
    static calls of its own -- a test must not press the buttons that lead to them."""
    from PySide6.QtWidgets import QFileDialog
    from tests._dialogs import chooser_guard
    mp = pytest.MonkeyPatch()
    mp.setattr(QFileDialog, "getSaveFileName", chooser_guard("save"))
    mp.setattr(QFileDialog, "getOpenFileName", chooser_guard("open"))
    yield
    mp.undo()


@pytest.fixture(autouse=True)
def _every_file_chooser_is_answered():
    """CHOOSER_CALLS holds this test's choosers only, and a chooser the test did not answer FAILS it
    here, at teardown. The guard itself cannot: it runs inside a clicked slot, where a raise never
    reaches the test."""
    from tests._dialogs import CHOOSER_CALLS, check_choosers_answered
    CHOOSER_CALLS.clear()
    yield
    try:
        check_choosers_answered()
    finally:
        CHOOSER_CALLS.clear()


@pytest.fixture(scope="session", autouse=True)
def _no_native_boxes():
    """A plain Windows message waits for a person's OK, and the entry point shows one when PRISM cannot start before
    its window is shown: one reached in a test would hold the run past the tool call's limit, on the real screen as
    much as off it. For the whole session ``core.gui.__main__._native_box`` is tests/_dialogs.native_box, which keeps
    the words in NATIVE_BOXES and shows nothing, whatever the test is marked for; a test's own stand-in layered with
    monkeypatch wins while installed, and its undo puts this one back. The function must be there to be stood in
    for: renamed, every test stops here, at its setup, where a stand-in set under the old name would be passed by
    and the message shown. The entry point's module loads nothing but ``os`` and ``sys`` as it is imported, so
    importing it here changes nothing about what a suite loads first."""
    from core.gui import __main__ as entry
    from tests._dialogs import native_box
    mp = pytest.MonkeyPatch()
    mp.setattr(entry, "_native_box", native_box)
    yield
    mp.undo()


@pytest.fixture(autouse=True)
def _every_native_box_is_read():
    """NATIVE_BOXES holds this test's plain Windows messages only, and one the test did not read -- with
    tests/_dialogs.native_boxes() -- FAILS it here, at teardown: a start that failed where none was expected."""
    from tests._dialogs import NATIVE_BOXES, check_native_boxes_read
    NATIVE_BOXES.clear()
    yield
    try:
        check_native_boxes_read()
    finally:
        NATIVE_BOXES.clear()


@pytest.fixture(autouse=True)
def _streams_put_back():
    """A test that binds the standard streams to a window's log, as the entry point does when PRISM starts with no
    console, and leaves them bound FAILS here, once they are put back and the streams the test began with are in
    force again: left bound, every later test's output, the toolkit's messages among it, would go to that test's
    file. So does a test that leaves the window's folder of plans bound to another than the one it began with, the
    session's, once that one is put back: left bound, every later window would list and write the plans there."""
    import sys
    from core.gui import __main__ as entry
    began = (sys.stdout, sys.stderr)
    plans = sys.modules.get("core.gui.plans")
    folder = None if plans is None else plans.PLAN_FOLDER
    yield
    if getattr(entry, "_bound", None) is not None:
        entry.unbind_streams()
        sys.stdout, sys.stderr = began
        pytest.fail("the test ended with the standard streams bound to a window's log, now put back")
    if plans is not None and plans.PLAN_FOLDER != folder:
        plans.PLAN_FOLDER = folder
        pytest.fail("the test ended with the window's folder of plans bound to another, now put back")


@pytest.fixture(autouse=True)
def _application_font_put_back():
    """A test that sets the application's font and leaves it set FAILS here, once the font it began with is in force
    again: left set, every later test of the process would draw in it, and the program's fonts registered again by a
    later test would take its family for the toolkit's own. A test that began with no application is held to the font
    the toolkit gives a new one. The toolkit is read where its module is kept, so that this check loads nothing."""
    import sys

    def held():
        widgets = sys.modules.get("PySide6.QtWidgets")
        return None if widgets is None else widgets.QApplication.instance()

    app = held()
    began = None if app is None else app.font().toString()
    yield
    app = held()
    if app is None:
        return
    from PySide6.QtGui import QFont, QFontDatabase
    if began is None:
        began = QFontDatabase.systemFont(QFontDatabase.SystemFont.GeneralFont).toString()
    if app.font().toString() != began:
        kept = QFont()
        kept.fromString(began)
        app.setFont(kept)
        pytest.fail("the test ended with the application's font set, now put back")


def check_drawing_code_raised_nothing(errors, counts) -> None:
    """Fail when ``errors`` -- core.gui.reported.ERRORS as a test left it -- holds anything, naming
    each method, what it raised the first time and, from ``counts`` (that module's COUNTS), how
    often it failed. Raised rather than asserted, so the check survives a run with assertions
    switched off."""
    if errors:
        raise AssertionError(
            "drawing code raised during this test (each error was caught and recorded, and the "
            "method answered its fallback, as in the running window): "
            + "; ".join(f"{name}: {words} ({counts.get(name, 1)} time(s))" for name, words in errors)
            + ". A test that makes one raise on purpose does it inside "
              "`with reported.expected():` (core/gui/reported.py).")


@pytest.fixture(autouse=True)
def _drawing_code_raises_nothing():
    """A method the toolkit calls while a window is drawn or laid out catches its own error
    (core/gui/reported.py): the error is recorded, the method answers a fallback, and nothing
    reaches the test that made the window draw. So the record is emptied before every test and read
    after it, and a test that leaves anything in it FAILS here, at teardown. That module imports
    the standard library and nothing else, so this costs a test two emptied containers and the
    copies of a list and a dictionary that are empty."""
    from core.gui import reported
    reported.forget()
    yield
    try:
        check_drawing_code_raised_nothing(list(reported.ERRORS), dict(reported.COUNTS))
    finally:
        reported.forget()


@pytest.fixture(autouse=True)
def _layout_room_put_back():
    """The room the screen offers the window, which the look's two layout thresholds are held to, is put back to
    none after every test. Every window a test builds hands the look its room as it sizes itself, and a room left
    in force would hold the thresholds of every later test. The look is read where its module is kept."""
    import sys
    yield
    design = sys.modules.get("core.gui.design")
    if design is not None:
        design.set_room(None)


@pytest.fixture(autouse=True)
def _memory_limit_put_back():
    """After every test the memory limit is put back as this test found it, and the keep-awake switch on, as every test
    finds it: the first is the one setting the window assigns on the configuration, and every later run would plan with
    it; the second is the process's run controller's, and every later run would ask Windows nothing. Each is read where
    its module is kept."""
    import sys
    config = sys.modules.get("core.config")
    found = None if config is None else config.SIM_VRAM_CEILING_GIB
    yield
    config = sys.modules.get("core.config")
    if config is not None:
        config.SIM_VRAM_CEILING_GIB = 0.0 if found is None else found
    runs = sys.modules.get("core.gui.run.controller")
    if runs is not None:
        runs.controller()._wish = True


@pytest.fixture(autouse=True)
def _built_widgets_let_go(_isolated_settings, _drawing_code_raises_nothing, _layout_room_put_back,
                          _memory_limit_put_back, windows_asked):
    """Every window, part, inference screen and bare run host the builders of tests/_fixtures.py made during
    a test is deleted when the test ends (tests/_built.py), so that none outlives it. Left alive, each would
    be dressed again by every later change of sheet, style or font on the one application, and an error of
    one of its guarded methods would be charged to whichever later test made that change. Each goes by its
    own deferred delete, the newest first, and is never closed, because closing a window writes settings.
    The five fixtures it names are set up before it and torn down after it, so the widgets go while this
    test's own settings file and its room are in force and before the drawing record is read, and a run this
    ends lets go of what it asked of Windows before the guard of those requests reads its record. What a
    fixture of wider scope built was made before this test was set up, so it is left alone. A test that ends
    with a run still live fails here, and its run is asked to stop and waited for before anything goes, so
    that no panel is deleted under its run (tests/_built.py's end_of_test says how long, and why). The
    process's one run controller is put back as the next test must find it: a run that ended is acknowledged,
    in this test's own folder of runs, and the clock and the folder the test handed it are let go."""
    from tests import _built
    at_start = _built.noted()
    yield
    _built.end_of_test(at_start)


@pytest.fixture(scope="module", autouse=True)
def _no_panel_outlives_its_module():
    """Every panel enrols in the set that the run's controller locks at each run's start and frees at its end, and
    what a fixture of a module's scope built is let go by that fixture, not by the end of a test. So once a module's
    tests and fixtures have all ended, the set holds no panel it did not hold when the module began: a panel left there
    would be locked and freed by every later module's runs, and refreshed at their end through whatever records that
    module has in use. Deferred deletions are carried out and the collector run before the set is read; nothing is
    read when no panel's module was ever loaded."""
    import gc
    import sys
    import weakref
    base = sys.modules.get("core.gui.panels.base_panel")
    held = weakref.WeakSet() if base is None else weakref.WeakSet(base.BasePanel._instances)
    yield
    base = sys.modules.get("core.gui.panels.base_panel")
    if base is None:
        return
    from PySide6.QtCore import QCoreApplication, QEvent
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    gc.collect()
    left = sorted(type(panel).__name__ for panel in list(base.BasePanel._instances) if panel not in held)
    assert not left, f"panels built in this module outlived it, still among the panels every run locks: {left}"


@pytest.fixture(autouse=True)
def brought_forward(request):
    """No window is brought to the front, and no process is let take the front, in a test that is not marked for the
    real screen: the window's ``come_to_front`` and ``one_instance.allow_foreground`` are stood in for, and each call
    is kept, as its name and what it was handed (the window, the process number), in the list this fixture gives a
    test that asks for it by name. A second start's knock that reached a window while a test pressed keys would
    otherwise take the front from the window the test brought there, and the keys would go to the other; and the
    second call reaches the system itself. A test marked for the real screen makes both calls for real. The window's
    module is loaded here when nothing has loaded it yet, once in a process, so that the stand-in is in every test.
    A monkeypatch of its own, so that a test that undoes its own does not undo this one. A test that ends holding
    the claim of a start of PRISM -- its lock and its listener -- fails here, once the claim is let go: kept, it
    would turn a later test's start away, or hear its knock."""
    from core.gui import one_instance
    calls = []
    mp = pytest.MonkeyPatch()
    if request.node.get_closest_marker("display") is None:
        from core.gui.main_window import MainWindow
        mp.setattr(MainWindow, "come_to_front", lambda window: calls.append(("come_to_front", window)))
        mp.setattr(one_instance, "allow_foreground", lambda pid: calls.append(("allow_foreground", pid)))
    yield calls
    mp.undo()
    held = one_instance.current()
    if held is not None:
        held.release()
        pytest.fail("the test ended holding the lock and the listener of a start of PRISM, now let go")


class _StandInWatch:
    """The watch on Windows' end of a session, as the guard of the requests made of Windows hands it back: nothing is
    installed; its removal is kept, and the process holds it no more."""

    def __init__(self, kept):
        self._kept = kept

    def remove(self):
        from core.gui import this_computer
        self._kept("remove the session watch")
        if this_computer._watch is self:
            this_computer._watch = None


def windows_requests_left(calls: list) -> list[str]:
    """What a test's record of the requests made of Windows -- ``windows_asked``'s, each call as its name, what it
    was handed and the name of the thread it came from -- leaves standing at the test's end, in words: the computer
    still asked to stay awake, a reason against ending the session still registered, and each time Windows itself
    was reached by a test not marked for the real screen. Nothing for a record that leaves nothing."""
    left = []
    awake = [args[0] for name, args, _thread in calls if name == "keep_awake"]
    if awake and awake[-1]:
        left.append("the computer is still kept awake")
    reasons = [args[1] for name, args, _thread in calls if name == "block_shutdown"]
    if reasons and reasons[-1] is not None:
        left.append(f"a reason against shutting down is still registered: {reasons[-1]!r}")
    left += [f"Windows was reached by a test not marked for the real screen: {args[0]}"
             for name, args, _thread in calls if name == "windows"]
    return left


@pytest.fixture(autouse=True)
def windows_asked(request):
    """Nothing a test does reaches Windows' power or session state, unless the test is marked for the real screen.
    The requests core/gui/this_computer.py makes of Windows -- keeping the computer awake, the reason against ending
    the session, the taskbar's flash, the watch on Windows' end of a session (handed back as one that installs
    nothing) -- are stood in for, each answering that Windows took it, and each call is kept,
    as its name, what it was handed and the name of the thread it came from, in the list this fixture gives a test
    that asks for it by name. Each stand-in keeps the module's record of the last request as the request would. The
    one place that module reaches Windows is stood in for too: a test that reaches it all the same is noted, and it
    is answered with an error. A test marked for the real screen makes every call for real, and releases each.

    At the test's end, after the end of the test has stopped a run left live (a run's end lets go of what it asked),
    a test whose record leaves the computer kept awake, a reason registered or Windows reached fails here, as
    ``windows_requests_left`` says it, and the module's record is put back to nothing asked and no watch. A
    monkeypatch of its own, so that a test that undoes its own does not undo this one."""
    import threading
    from core.gui import this_computer
    calls = []
    mp = pytest.MonkeyPatch()

    def kept(name, *args):
        calls.append((name, args, threading.current_thread().name))

    if request.node.get_closest_marker("display") is None:
        def keep_awake(on):
            kept("keep_awake", on)
            this_computer._requested["awake"] = bool(on)
            return True

        def block_shutdown(window, reason):
            kept("block_shutdown", window, reason)
            this_computer._requested["reason"] = None if reason is None else this_computer.fitted(reason)
            return True

        def flash(window):
            kept("flash", window)
            return True

        def watch_session_end(app, *, asked, ending):
            kept("watch_session_end", app, asked, ending)
            this_computer._watch = _StandInWatch(kept)
            return this_computer._watch

        def reached(library, name):
            kept("windows", f"{library}.{name}")
            raise RuntimeError("a test not marked for the real screen reached Windows")

        for name, stand_in in (("keep_awake", keep_awake), ("block_shutdown", block_shutdown), ("flash", flash),
                               ("watch_session_end", watch_session_end), ("_windows", reached)):
            mp.setattr(this_computer, name, stand_in)
    yield calls
    mp.undo()
    left = windows_requests_left(calls)
    this_computer._requested.update(awake=False, reason=None)
    if isinstance(this_computer._watch, _StandInWatch):
        this_computer._watch = None
    if left:
        raise AssertionError("the test ended with what it asked of Windows standing: " + "; ".join(left))
