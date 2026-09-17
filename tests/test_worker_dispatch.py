"""Worker dispatch, cooperative cancellation, error dialogs and payload lifetime. Split from test_gui_progress.py; run directly: pytest tests/test_worker_dispatch.py"""
"""Progress-rendering regression tests for the GUI.

THE BUG THESE LOCK DOWN
    tqdm redraws a bar at pos>0 as three writes -- '\\n'*pos, then '\\r'+frame, then '\\x1b[A'*pos
    (tqdm/std.py:1493-1497). The old stream reader split on terminators, so the frame (which is never
    terminated) stranded in its buffer and was flushed by the NEXT redraw's leading '\\n' -- i.e. as a
    LOG LINE. Every nested-bar redraw appended one row, so a training run buried the log pane under
    hundreds of bar snapshots.

    The pipeline nests bars four deep (core/SBI/pipeline.py:517 -> :371 ->
    core/Simulator/simulator.py:50 -> core/Solvers/sdeint.py:15), so this fired constantly.

Run:  pytest tests/test_worker_dispatch.py
      (or just: pytest tests/test_gui_progress.py)
"""
import ast
import inspect
import textwrap
import os
import tempfile
import sys
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")   # must precede any PySide6 import
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib                                                 # noqa: E402
matplotlib.use("Agg")                                            # match the app (core/gui/__main__.py forces it)

import torch                                                      # noqa: E402
from PySide6.QtWidgets import QApplication                        # noqa: E402
from tqdm import tqdm                                             # noqa: E402

from core.gui.panels.base_panel import BasePanel                  # noqa: E402
from core.gui.streams import redirect_streams                     # noqa: E402
from core.gui.vt import StreamRouter, parse_bar                   # noqa: E402
from core.gui.widgets.log_pane import LogPane                     # noqa: E402
from core.gui.widgets.progress_pane import ProgressPane           # noqa: E402
from core.gui.worker import WorkerSignals                         # noqa: E402
from tests._fixtures import qt_app, pump                          # noqa: E402
import contextlib                                                  # noqa: E402


def test_worker_payload_is_released_after_the_run():
    """setAutoDelete(False) + the _finished closure keep the Worker shell alive forever. That is fine
    for the shell, but NOT for what it points at -- without an explicit release every dispatch pins its
    cfg / prior / posterior / CUDA tensors for the life of the process."""
    import gc
    import weakref

    app = qt_app()

    class Big:
        pass

    class P(BasePanel):
        pass

    panel = P()
    big = Big()
    ref = weakref.ref(big)

    panel.dispatch(lambda payload: None, big)
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and panel._busy:
        app.processEvents()
        time.sleep(0.01)
    pump(app, 0.2)

    del big
    gc.collect()
    assert ref() is None, "the worker still pins its argument after the run finished"

# ── Phase 3: cancellation ────────────────────────────────────────────────────────────────────────
def test_worker_cancelled_passes_through_except_exception():
    """The cancel exception must be a BaseException so the pipeline's many `except Exception` handlers
    (sbi, cross_validation, Worker.run itself) do not swallow it."""
    from core.gui.streams import WorkerCancelled

    try:
        try:
            raise WorkerCancelled()
        except Exception:  # noqa: BLE001
            raise AssertionError("`except Exception` caught the cancel -- it is not BaseException-derived")
    except WorkerCancelled:
        pass

def test_cancel_token_latches_and_leaves_tqdm_usable():
    """The token raises exactly ONCE (so tqdm's teardown write can finish), and after the cancel a NEW
    tqdm bar must be creatable on a fresh thread.

    That last part is the sharp edge: tqdm's refresh() manual-acquires its global write lock and
    manual-releases it (tqdm/std.py:1346-1349), so a raise from inside a redraw's write() skips the
    release and leaks the lock -- after which the next tqdm.__new__ deadlocks. redirect_streams' cancel
    teardown resets the lock; without that reset THIS TEST HANGS (which is exactly the production bug:
    cancel a run, start another, hang)."""
    import threading

    from core.gui.streams import CancelToken, WorkerCancelled, redirect_streams

    signals = WorkerSignals()
    token = CancelToken()
    token.requested.set()                      # cancel already requested when the run starts

    with redirect_streams(signals, token):
        try:
            for _ in tqdm(range(5), desc="Generating training data", leave=False):
                for _ in tqdm(range(5), desc="step (batch=8)", leave=False):
                    pass
        except WorkerCancelled:
            pass
    assert token.fired, "the token never fired"
    assert len(tqdm._instances) == 0, f"tqdm left {len(tqdm._instances)} stale bar(s)"

    # tqdm must not be wedged: make a bar on a fresh thread with a bounded join.
    made = []

    def _probe():
        b = tqdm(range(2), desc="probe", file=open(os.devnull, "w"))
        b.close()
        made.append(True)

    th = threading.Thread(target=_probe, daemon=True)
    th.start()
    th.join(3.0)
    assert made == [True], "tqdm deadlocked after a cancel -- the write lock was not recovered"

def test_dispatched_run_cancels_cleanly_and_a_later_run_still_works():
    """End to end: a run cancelled mid-flight emits `cancelled` (not `error`), ends not-busy with rows
    dropped and stray figures closed, clears the active token, and a fresh run afterwards completes."""
    import matplotlib.pyplot as plt

    app = qt_app()

    class P(BasePanel):
        pass

    panel = P()
    started = {"v": False}
    outcome = {"cancelled": 0, "error": 0, "result": []}

    def heavy(fig_sink=None):
        plt.figure()                          # a stray figure the cancel path must close
        for i in range(2000):
            for _ in tqdm(range(20), desc="step (batch=8)", leave=False):
                pass
            started["v"] = True
            print(f"epoch {i}")               # a write() checkpoint
            time.sleep(0.01)
        return "COMPLETED"

    panel.dispatch(heavy, on_result=lambda r: outcome["result"].append(r))
    for w in panel._workers:
        w.signals.cancelled.connect(lambda: outcome.__setitem__("cancelled", outcome["cancelled"] + 1))
        w.signals.error.connect(lambda *_a: outcome.__setitem__("error", outcome["error"] + 1))

    t0 = time.monotonic()
    while time.monotonic() - t0 < 5 and not started["v"]:
        app.processEvents()
        time.sleep(0.005)
    assert panel._busy and BasePanel._active_cancel is not None

    panel._request_cancel()
    assert panel._cancel.requested.is_set()
    assert panel.btn_cancel.text() == "Cancelling…" and not panel.btn_cancel.isEnabled()

    t0 = time.monotonic()
    while time.monotonic() - t0 < 10 and panel._busy:
        app.processEvents()
        time.sleep(0.005)
    pump(app, 0.3)

    assert not panel._busy, "panel stuck busy after cancel"
    assert outcome["result"] == [], "the run COMPLETED instead of cancelling"
    assert outcome["cancelled"] == 1 and outcome["error"] == 0, outcome
    assert not panel.progress_pane._rows, "rows leaked after cancel"
    assert plt.get_fignums() == [], "stray figures not closed on cancel"
    assert BasePanel._active_cancel is None and not BasePanel._running
    assert "Run cancelled." in panel.log_pane.toPlainText()

    later = []
    panel.dispatch(lambda: "SECOND OK", on_result=later.append)
    t0 = time.monotonic()
    while time.monotonic() - t0 < 5 and (panel._busy or not later):
        app.processEvents()
        time.sleep(0.005)
    assert later == ["SECOND OK"], f"a run after a cancel did not complete: {later}"

def test_cancel_is_not_consumed_by_a_non_worker_thread():
    """tqdm's TMonitor daemon force-refreshes a quiet bar, writing to our stream from ITS thread. If
    that write consumed the cancel latch, it would raise where nobody catches it and leave the worker
    to sail past a fired latch -- silently losing the cancel. Only the armed (worker) thread may raise."""
    import threading
    from core.gui.streams import CancelToken, WorkerCancelled

    token = CancelToken()
    token.requested.set()
    out = {}

    def worker():
        token.arm()                            # redirect_streams arms on the worker thread
        # a non-worker ("monitor") write happens first and must NOT consume the latch
        other = threading.Thread(target=token.check)
        other.start()
        other.join()
        try:
            token.check()                      # the worker's own next write MUST still raise
        except WorkerCancelled:
            out["worker_raised"] = True

    th = threading.Thread(target=worker)
    th.start()
    th.join(3.0)
    assert out.get("worker_raised") is True, "the cancel was consumed by a non-worker thread and lost"

def test_inference_config_restore_with_a_stale_model_does_not_desync_the_bounds_picker():
    """A corrupt/version-skewed .ini with an unknown model must not leave the (Prior-tab) bounds picker
    pointing at a nonexistent folder while the Config combo shows a real default. The picker is repointed
    when the model is APPLIED, so drive that path."""
    from core.gui import settings as st
    from core.gui.screens.inference_screen import InferenceScreen

    qt_app()
    qs = st.settings()
    qs.beginGroup("inference_config")
    qs.setValue("model", "NOT_A_REAL_MODEL")
    qs.endGroup()
    qs.sync()

    screen = InferenceScreen()
    model = screen.config_panel.model_combo.currentText()
    assert model in ("BP", "NADROWSKI", "HOPF"), model
    screen.config_panel._build_config()                 # apply -> new_draft -> repoint the picker
    picker = screen.prior_panel.bounds_picker
    assert picker.base_path.name == model.lower()
    assert picker.combo.count() > 0, "the bounds picker was left empty by a stale model"

# ── Phase 3: error dialogs ───────────────────────────────────────────────────────────────────────
def test_on_error_puts_the_traceback_in_details_not_the_body():
    """A run failure's traceback belongs in a collapsible Details panel, not pasted whole into the
    dialog body."""
    from PySide6.QtWidgets import QMessageBox

    qt_app()

    class P(BasePanel):
        pass

    panel = P()
    captured = {}
    orig_exec = QMessageBox.exec

    def fake_exec(self):
        captured["text"] = self.text()
        captured["detail"] = self.detailedText()
        return 0

    QMessageBox.exec = fake_exec
    try:
        panel._on_error("Something failed", "Traceback (most recent call last):\n  ...\nValueError: x")
    finally:
        QMessageBox.exec = orig_exec

    assert captured["text"] == "Something failed"
    assert "Traceback" in captured["detail"], "the traceback was not routed to Details"
    assert "Traceback" not in captured["text"], "the traceback leaked into the body"


# ── piece 3: the session dialog guard and the pane recorder ──────────────────────────────────────
def test_an_unexpected_modal_is_recorded_in_shown_and_returns_zero(monkeypatch):
    """Offscreen, QMessageBox.exec() spins a nested event loop that nothing ever closes: a box a test
    did not fake itself was a STALL past the ten-minute tool-call limit (no timeout plugin is
    installed), not a failure anyone could read. The session guard (tests/conftest.py::
    _no_modal_dialogs) turns it into a record -- the box lands in tests/_fixtures.SHOWN and exec
    returns 0 with no button clicked, which the consent dialogs read as Cancel and main_window.py:231
    as the safe branch -- and _clear_shown empties the list before every test, so SHOWN[-1] is always
    the box the code under test just tried to show. A test's own fake, layered with monkeypatch the
    way test_the_two_dialogs_default_to_cancel does, wins while it is installed, and the guard is
    back the moment it is undone."""
    from PySide6.QtWidgets import QMessageBox
    from tests._fixtures import SHOWN, qt_app

    qt_app()
    assert SHOWN == [], "SHOWN must be empty at the start of every test"
    box = QMessageBox()
    box.setText("an unexpected modal")
    assert box.exec() == 0
    assert SHOWN == [box] and box.clickedButton() is None

    monkeypatch.setattr(QMessageBox, "exec", lambda self: 7)
    assert QMessageBox().exec() == 7 and SHOWN == [box], "a test's own fake must win while installed"
    monkeypatch.undo()
    assert QMessageBox().exec() == 0 and len(SHOWN) == 2, "the session guard must be back after undo"


def test_pane_capture_records_level_and_text_and_the_pane_stays_blank():
    """The window's refusals and warnings are asserted OFF THE PANE. PaneCapture stands in for
    ``log_pane.append_line`` on one panel and keeps ``(level, text)`` in order, with LogPane's own
    default level, so the ``lambda text, kind="": lines.append((kind, text))`` stub that five tests
    each wrote by hand becomes one helper that cannot drift from the real signature."""
    from tests._fixtures import PaneCapture, qt_app

    qt_app()

    class P(BasePanel):
        pass

    panel = P()
    before = panel.log_pane.toPlainText()
    cap = PaneCapture(panel)
    panel.log_pane.append_line("plain")
    panel.log_pane.append_line("watch out", "warning")
    assert cap.lines == [("info", "plain"), ("warning", "watch out")]
    assert panel.log_pane.toPlainText() == before, "a captured line must not also reach the widget"
