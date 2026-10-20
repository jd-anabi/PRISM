"""Fixtures for tests/ (the root conftest.py sets the environment and the marker skips).

`qapp` is the one QApplication a process may hold; ``tests/_fixtures.qt_app()`` is the same
expression as a plain call, which is what the GUI suites use. The settings, dialog and drawing-code
fixtures at the end, the one that lets go what the builders made and the one that stands in for
bringing a window to the front, are autouse at session and function scope; a test requests the last
by name, ``brought_forward``, to read what it kept, and no other by name.
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
    temporary settings file (_settings_home), never from the real PRISM.ini."""
    from types import SimpleNamespace
    from tests._fixtures import build_inference, qt_app
    qt_app()
    cfg = tiny_run.cfg.copy_for_run()
    cfg.clear_ground_truth()
    inf = build_inference(cfg=cfg)
    inf.session.inf_prior = tiny_run.prior
    inf.session.posterior = tiny_run.posterior
    inf.refresh_gates()
    return SimpleNamespace(screen=inf, run=tiny_run, cell=tiny_run.cfg.sources["cell"])


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
    in force would hold the thresholds of every later test. The look is read where its module is kept, and loaded
    by nothing here: a test that never loads it pays nothing."""
    import sys
    yield
    design = sys.modules.get("core.gui.design")
    if design is not None:
        design.set_room(None)


@pytest.fixture(autouse=True)
def _built_widgets_let_go(_isolated_settings, _drawing_code_raises_nothing, _layout_room_put_back):
    """Every window, part, inference screen and bare run host the builders of tests/_fixtures.py made during
    a test is deleted when the test ends (tests/_built.py), so that none outlives it. Left alive, each would
    be dressed again by every later change of sheet, style or font on the one application, and an error of
    one of its guarded methods would be charged to whichever later test made that change. Each goes by its
    own deferred delete, the newest first, and is never closed, because closing a window writes settings.
    The three fixtures it names are set up before it and torn down after it, so the widgets go while this
    test's own settings file and its room are in force and before the drawing record is read. What a
    fixture of wider scope built was made before this test was set up, so it is left alone. A test that ends
    with a run still live fails here, and its run is asked to stop and waited for before anything goes, so
    that no panel is deleted under its run (tests/_built.py's end_of_test says how long, and why)."""
    from tests import _built
    at_start = _built.noted()
    yield
    _built.end_of_test(at_start)


@pytest.fixture(autouse=True)
def brought_forward(request):
    """No window is brought to the front, and no process is let take the front, in a test that is not marked for the
    real screen: the window's ``come_to_front`` and ``one_instance.allow_foreground`` are stood in for, and each call
    is kept, as its name and what it was handed (the window, the process number), in the list this fixture gives a
    test that asks for it by name. A second start's knock that reached a window while a test pressed keys would
    otherwise take the front from the window the test brought there, and the keys would go to the other; and the
    second call reaches the system itself. A test marked for the real screen makes both calls for real. The window's
    module is loaded here when nothing has loaded it yet, once in a process, so that the stand-in is in every test.
    A monkeypatch of its own, so that a test that undoes its own does not undo this one."""
    calls = []
    mp = pytest.MonkeyPatch()
    if request.node.get_closest_marker("display") is None:
        from core.gui import one_instance
        from core.gui.main_window import MainWindow
        mp.setattr(MainWindow, "come_to_front", lambda window: calls.append(("come_to_front", window)))
        mp.setattr(one_instance, "allow_foreground", lambda pid: calls.append(("allow_foreground", pid)))
    yield calls
    mp.undo()
