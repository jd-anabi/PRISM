"""Fixtures for tests/ (the root conftest.py sets the environment and the marker skips).

`qapp` is the one QApplication a process may hold; ``tests/_fixtures.qt_app()`` is the same
expression as a plain call, which is what the GUI suites use. The settings and dialog fixtures at
the end are autouse at session and function scope and are never requested by name.
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
    """<repo>/sbi-logs must not exist before or after the session (V9, spec §6.4). This sits beside
    _sandbox_default_store for the same reason: the suites leave the repository as they found it.

    sbi's default TensorBoard writer made <cwd>/sbi-logs/NPE_C/<timestamp>/ on every training, and
    every gate trains at the repo root (tiny_run's build_posterior), which is how 881 directories
    piled up there. core/SBI/train.py now hands sbi _NoSummary, the tree was deleted, and its
    .gitignore line went. So the directory coming back means a training reached sbi's default writer
    again, and with no ignore line it would also sit in git status.

    It is checked at SETUP as well: a tree left by some earlier run would otherwise make the teardown
    blame this session for it. A training started from another working directory is covered by
    tests/test_user_sbi.py::test_training_creates_no_sbi_logs_directory."""
    from core import config
    logs = config.REPO_ROOT / "sbi-logs"
    assert not logs.exists(), (
        f"{logs} exists before the session: delete it (nothing in PRISM writes it since piece 3), "
        f"or this session's teardown check would blame the tests for it")
    yield
    assert not logs.exists(), (
        f"the suite created {logs}: a training reached sbi's default TensorBoard writer instead of "
        f"core.SBI.train._NoSummary (train_nn's summary_writer=None)")


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
    and never carries a cell (spec §2.4). A COPY, because tiny_run is module-scoped and its siblings in
    the same file need the truth it injected; a truth-carrying session would also make the pin
    vacuous, since the composition re-injects the very same cell on its copy.

    Module-scoped like tiny_run, so a file pays for one screen. Its panels restore from the session's
    temporary settings file (_settings_home), never from the real PRISM.ini."""
    from types import SimpleNamespace
    from core.gui.screens.inference_screen import InferenceScreen
    from tests._fixtures import qt_app
    qt_app()
    cfg = tiny_run.cfg.copy_for_run()
    cfg.clear_ground_truth()
    inf = InferenceScreen()
    inf.install_config(cfg)
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
    gen_training_data entirely while the suite stayed green. Under D7 a stray cache is worse still: a
    committed sibling one identity field away now REFUSES a later run instead of quietly restarting it,
    so one test's leftovers would fail another's.

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
    """The settings location is NEVER the real PRISM.ini during a test process, at any scope (spec
    §5.4). This is the session half: a path under the session temp that does not exist until Qt
    writes it, installed before any module- or session-scoped fixture can build a panel (pytest sets
    higher scopes up first), so a panel built by tiny_run's siblings or by screen_run restores from an
    empty file too. The function half is _isolated_settings, below.

    Teardown asserts the developer's real user-scope file is byte-identical, mirroring
    _sandbox_default_store's assertion on the real Artifacts/: a saved value winning over config.py is
    the mechanism that cost a ~5-day run on 2026-08-19 (the retired chi band), and a test that wrote
    one would arm it for the next real launch. The real path is resolved the way settings() resolves
    it with no override; constructing that QSettings neither creates nor rewrites the file.
    """
    from PySide6.QtCore import QSettings
    from core.gui import settings
    real = Path(QSettings(QSettings.IniFormat, QSettings.UserScope, settings.ORG, settings.APP).fileName())
    before = real.read_bytes() if real.exists() else None
    settings.use_ini_file(str(tmp_path_factory.mktemp("settings") / "prism.ini"))
    yield
    settings.use_ini_file(None)
    after = real.read_bytes() if real.exists() else None
    assert after == before, f"the suite wrote into the real settings file {real}"


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
    """Offscreen, QMessageBox.exec() spins a nested event loop that nothing ever closes, so a dialog a
    test did not expect is a STALL past the ten-minute tool-call limit (no timeout plugin is
    installed), not a failure. For the whole session the box is appended to tests/_fixtures.SHOWN and
    exec returns 0 -- no clicked button, which the consent dialogs read as Cancel and main_window.py:231
    as the safe branch. A test that wants its own fake layers it on top with monkeypatch, as
    test_the_two_dialogs_default_to_cancel does; the undo puts this guard back, because on a class
    MonkeyPatch records the class __dict__ entry, which is this lambda. Same MonkeyPatch shape as
    _checkpointing_off_unless_asked. The three static QMessageBox.warning/information calls in
    main_window.py are C++ statics and outside this guard; no test reaches them offscreen."""
    from PySide6.QtWidgets import QMessageBox
    from tests._fixtures import SHOWN
    mp = pytest.MonkeyPatch()
    mp.setattr(QMessageBox, "exec", lambda self: SHOWN.append(self) or 0)
    yield
    mp.undo()


@pytest.fixture(autouse=True)
def _clear_shown():
    """SHOWN holds this test's boxes only, so SHOWN[-1] is never a leftover from an earlier test."""
    from tests._fixtures import SHOWN
    SHOWN.clear()
    yield
