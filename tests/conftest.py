"""Fixtures for tests/ (the root conftest.py sets the environment and the marker skips).

`qapp` is the one QApplication a process may hold. The suites historically built it themselves
with ``QApplication.instance() or QApplication([])`` inside a helper; that stays valid, and this
fixture is the same expression for tests that prefer to declare the dependency.
"""
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
