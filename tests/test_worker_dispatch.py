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


def test_a_failed_run_does_not_pin_its_frames():
    """The worker carries the EXCEPTION across the thread (so the panel can route a Refusal to the
    yellow box), and an exception's traceback owns every frame of the failed run -- the stage's host
    buffers, the prior, the CUDA tensors. Held in a local of Worker.run, whose own frame heads that
    traceback, it made a reference cycle only a full garbage collection frees, which in a process that
    has loaded torch, sbi and Qt can be hours away. Worker.run drops the tracebacks once the text is
    formatted, so reference counting frees the frames when the box is dismissed.

    NO gc.collect() here, and the collector is off for the whole run: a collection is exactly what
    hides the cycle (test_worker_payload_is_released_after_the_run calls one). Two legs: a bug (the
    red box) and a Refusal raised from a cause its frame keeps bound (the yellow box; the cause's own
    traceback is dropped too, which the leg fails without)."""
    import gc
    import weakref
    from PySide6.QtWidgets import QMessageBox
    from core.refusals import Refusal
    from tests._fixtures import SHOWN

    app = qt_app()

    class Big:
        pass

    class P(BasePanel):
        pass

    def crash(payload):
        raise RuntimeError("the batch failed")

    def refuse(payload):
        held = None
        try:
            raise OSError("the underlying cause")
        except OSError as cause:
            # kept bound, as a stage's retry ladder keeps its `err`: this frame -> cause -> its
            # traceback -> this frame is a cycle of its own, which only the chain walk breaks
            held = cause
        raise Refusal("The run was refused.", field="name") from held

    for fn, title, icon in ((crash, "Error", QMessageBox.Critical),
                            (refuse, "Check your inputs", QMessageBox.Warning)):
        panel = P()
        big = Big()
        ref = weakref.ref(big)
        was_enabled = gc.isenabled()
        gc.disable()
        try:
            SHOWN.clear()
            panel.dispatch(fn, big)
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline and panel._busy:
                app.processEvents()
                time.sleep(0.01)
            pump(app, 0.2)
            assert SHOWN and SHOWN[-1].windowTitle() == title and SHOWN[-1].icon() == icon, \
                (fn.__name__, [(b.windowTitle(), b.text()) for b in SHOWN])
            del big
            assert ref() is None, f"{fn.__name__}: a failed run still pins its frames after the box"
        finally:
            if was_enabled:
                gc.enable()

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
    dialog body. The worker hands over the EXCEPTION now, not its text: a bug is anything that is not
    a Refusal, and it keeps this red box."""
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
        panel._on_error(RuntimeError("Something failed"),
                        "Traceback (most recent call last):\n  ...\nRuntimeError: Something failed")
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


def test_on_error_routes_a_refusal_to_the_yellow_box():
    """The one place the window tells a refusal from a bug. A Refusal reaching _on_error -- from the
    worker's error signal, or from a click handler that passes what it caught -- opens the YELLOW
    "Check your inputs" box: the core's neutral sentence as the text, this front end's "where to fix
    it" (core/gui/fields.py, looked up by the field key) as the informative line, one OK button which
    is the default, and NO Details -- a refusal is not a crash and a traceback would only say so
    louder. A refusal with no field has no fix sentence and no informative line. The same sentence
    goes to the log pane at warning, so it outlives the click that dismisses the box. And the plain
    string the tabs used to pass still works, and still means "bug": the red box."""
    from PySide6.QtWidgets import QMessageBox
    from core.gui import fields as gui_fields
    from core.refusals import Refusal
    from tests._fixtures import SHOWN, PaneCapture, qt_app

    qt_app()

    class P(BasePanel):
        pass

    panel = P()
    pane = PaneCapture(panel)
    exc = Refusal("The observation length, in seconds, is blank (default none: it must be given).",
                  field="t_obs")
    panel._on_error(exc, "Traceback (most recent call last):\n  ...\nRefusal: blank")

    box = SHOWN[-1]
    assert isinstance(box, QMessageBox)
    assert box.windowTitle() == "Check your inputs"
    assert box.icon() == QMessageBox.Warning
    assert box.text() == exc.message
    assert box.informativeText() == "Set it in the 'T_obs (s)' box on the Infer tab."
    assert box.informativeText() == gui_fields.fix_sentence("t_obs")
    assert box.detailedText() == "", "a refusal carries no traceback"
    assert box.standardButtons() == QMessageBox.Ok
    assert box.buttonRole(box.defaultButton()) == QMessageBox.AcceptRole, "OK is the default"
    assert pane.lines[-1] == ("warning", f"{exc.message} {box.informativeText()}")

    # field=None: no fix sentence, so no informative line, and the log line is the message alone
    SHOWN.clear()
    panel._on_error(Refusal("resume='require' but there is no resumable cache at x."), "")
    assert SHOWN[-1].windowTitle() == "Check your inputs"
    assert SHOWN[-1].informativeText() == ""
    assert pane.lines[-1] == ("warning", "resume='require' but there is no resumable cache at x.")

    # a plain string is still accepted, and is still a bug: the red box
    SHOWN.clear()
    panel._on_error("plain text", "")
    assert SHOWN[-1].windowTitle() == "Error" and SHOWN[-1].icon() == QMessageBox.Critical
    assert SHOWN[-1].text() == "plain text"
    assert pane.lines[-1] == ("error", "plain text")


def test_a_refusal_opens_the_yellow_box_without_a_traceback_and_a_bug_the_red_one():
    """End to end through dispatch(): the exception OBJECT crosses the worker thread on the error
    signal, so the panel can route by type. Worker.run used to flatten every failure to
    (str(e), traceback) and the panel had nothing left to route on -- a refusal raised inside a
    stage (the direction count, a near miss, D12, a load mismatch) opened the same red Critical box,
    traceback and all, as a genuine crash. Read off the conftest's SHOWN record, which is what the
    class-level QMessageBox.exec guard leaves behind instead of a modal that would stall offscreen."""
    from PySide6.QtWidgets import QMessageBox
    from core.refusals import Refusal
    from tests._fixtures import SHOWN, PaneCapture, pump, qt_app

    app = qt_app()

    class P(BasePanel):
        pass

    panel = P()
    pane = PaneCapture(panel)

    def _run_to_dialog(fn):
        SHOWN.clear()
        panel.dispatch(fn)
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and (panel._busy or not SHOWN):
            app.processEvents()
            time.sleep(0.01)
        pump(app, 0.2)
        assert not panel._busy, "the panel stayed busy after the failure"
        assert SHOWN, "no dialog opened"
        return SHOWN[-1]

    def refuse():
        raise Refusal("The observation length, in seconds, must be greater than 0; got 0 "
                      "(default none: it must be given).", field="t_obs")

    box = _run_to_dialog(refuse)
    assert box.windowTitle() == "Check your inputs" and box.icon() == QMessageBox.Warning
    assert box.text().startswith("The observation length, in seconds, must be greater than 0")
    assert box.informativeText() == "Set it in the 'T_obs (s)' box on the Infer tab."
    assert box.detailedText() == "", "a refusal must not carry a traceback"
    assert pane.lines[-1][0] == "warning" and "'T_obs (s)'" in pane.lines[-1][1], pane.lines

    def bug():
        raise RuntimeError("Something failed")

    box = _run_to_dialog(bug)
    assert box.windowTitle() == "Error" and box.icon() == QMessageBox.Critical
    assert box.text() == "Something failed"
    assert "Traceback" in box.detailedText() and "RuntimeError: Something failed" in box.detailedText()
    assert pane.lines[-1] == ("error", "Something failed")


def test_list_dir_returns_and_the_picker_no_longer_swaps_stdout(tmp_path, capsys):
    """file_manager.list_dir used to PRINT a numbered tree and return the list as a side line, and the
    GUI's ArtifactPicker silenced the tree with contextlib.redirect_stdout -- which reassigns the
    PROCESS-WIDE sys.stdout, the very stream redirect_streams installs for a running worker. A picker
    refreshed mid-run swallowed the worker's output, and a worker teardown inside that window left
    the dead _SignalStream as the process's stdout for good (the hazard base_panel documented and
    locked the whole column against). The listing is now returned and nothing is printed, so the
    picker has nothing to silence: pinned on the function's stdout, on what sys.stdout IS while the
    picker calls it, and on the refresh's source."""
    import io

    import pytest

    from core.Helpers import file_manager
    from core.gui.widgets import artifact_picker
    from core.gui.widgets.artifact_picker import ArtifactPicker
    from tests._fixtures import code_only, qt_app

    (tmp_path / "a.txt").write_text("a")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.txt").write_text("b")
    (tmp_path / "sub" / "c.rot.pt").write_bytes(b"")
    keep = lambda rel: rel.endswith(".txt")                                  # noqa: E731
    capsys.readouterr()
    got = file_manager.list_dir(str(tmp_path), keep=keep)
    assert sorted(got) == ["a.txt", os.path.join("sub", "b.txt")]
    assert capsys.readouterr().out == "", "list_dir printed its tree"

    qt_app()
    seen, real = [], file_manager.list_dir

    def _spy(path, keep=None):
        seen.append(sys.stdout)
        return real(path, keep=keep)

    marker = io.StringIO()
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(artifact_picker.file_manager, "list_dir", _spy)
        mp.setattr(sys, "stdout", marker)
        ap = ArtifactPicker(tmp_path, keep=keep)                             # __init__ refreshes once
        ap.refresh()
    assert len(seen) == 2 and all(s is marker for s in seen), \
        "refresh swapped sys.stdout around list_dir"
    assert sorted(ap.combo.itemData(i) for i in range(ap.combo.count())) == \
        ["a.txt", os.path.join("sub", "b.txt")]
    assert marker.getvalue() == ""
    src = code_only(ArtifactPicker.refresh)
    assert "redirect_stdout" not in src and "StringIO" not in src
