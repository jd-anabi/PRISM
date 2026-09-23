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
from tqdm import tqdm                                             # noqa: E402

from core.artifacts.store import KIND_DIRS, Summary                # noqa: E402
from core.gui.panels.base_panel import BasePanel                  # noqa: E402
from core.gui.streams import redirect_streams                     # noqa: E402
from core.gui.vt import StreamRouter, parse_bar                   # noqa: E402
from core.gui.widgets import artifact_table as at                 # noqa: E402
from core.gui.widgets.artifact_table import ArtifactTable, cells_for, columns_for   # noqa: E402
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
    returns 0 with no button clicked, which the consent dialogs read as Cancel and MainWindow.closeEvent
    reads as its ignore branch -- and _clear_shown empties the list before every test, so SHOWN[-1] is
    always the box the code under test just tried to show. A test's own fake, layered with monkeypatch
    the way test_the_d7_and_d8_dialogs_default_to_cancel does, wins while it is installed, and the
    guard is back the moment it is undone."""
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
    """The window's refusals and warnings are asserted OFF THE PANE. PaneCapture stands in for BOTH of
    ``log_pane``'s channels on one panel -- ``append_line``, the panel's own messages, and
    ``append_lines``, one pump tick of the run's output -- and keeps ``(level, text)`` in order, with
    LogPane's own default level, so the ``lambda text, kind="": lines.append((kind, text))`` stub that
    five tests each wrote by hand becomes one helper that cannot drift from the real signatures.

    ONE list, in the order the lines happened, because that is the order the user reads them in: a
    batch's pairs arrive as ``(text, level)`` (WorkerSignals.log_batch, worker.py:27) and are flipped
    into the helper's ``(level, text)`` as they are appended. An empty batch appends nothing, exactly
    as LogPane.append_lines renders nothing for one."""
    from tests._fixtures import PaneCapture, qt_app

    qt_app()

    class P(BasePanel):
        pass

    panel = P()
    before = panel.log_pane.toPlainText()
    cap = PaneCapture(panel)
    panel.log_pane.append_line("plain")
    panel.log_pane.append_lines([("Config built: NADROWSKI", "info"),
                                 ("[tsnpe] loaded a NON-AMORTIZED posterior", "warning")])
    panel.log_pane.append_line("watch out", "warning")
    panel.log_pane.append_lines([])
    panel.log_pane.append_lines(None)
    assert cap.lines == [("info", "plain"),
                         ("info", "Config built: NADROWSKI"),
                         ("warning", "[tsnpe] loaded a NON-AMORTIZED posterior"),
                         ("warning", "watch out")]
    assert panel.log_pane.toPlainText() == before, "a captured line must not also reach the widget"


def test_pane_capture_sees_the_pumps_batch_channel_through_a_dispatch():
    """B17 end to end, through the machinery the helper exists for. ``BasePanel.dispatch`` resolves
    ``self.log_pane.append_line`` AND ``self.log_pane.append_lines`` at CONNECT time
    (base_panel.py:219-220), so a capture installed before the dispatch sees both; the batch channel is
    the only way a print, a retired tqdm bar, a ``core`` record or a Python warning from the WORKER
    reaches the pane, and before this task it went to the widget and nowhere a test could read it.

    A batch is published by the pump's daemon thread at 15 Hz as a QUEUED cross-thread signal, so the
    Qt loop has to be driven before asserting -- ``pump(app)``, as every other worker test here does.
    The two lines must also keep their order relative to each other: records and prints share one
    ordered sink (streams._Pump)."""
    import logging

    from tests._fixtures import PaneCapture, qt_app

    app = qt_app()

    class P(BasePanel):
        pass

    panel = P()
    cap = PaneCapture(panel)                     # BEFORE the dispatch, or neither channel is captured
    before = panel.log_pane.toPlainText()

    def work():
        print("Config built: NADROWSKI")
        logging.getLogger("core.tests.pane_capture").warning("[tsnpe] loaded a NON-AMORTIZED posterior")
        return "done"

    got = []
    panel.dispatch(work, on_result=got.append)
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and (panel._busy or not got):
        app.processEvents()
        time.sleep(0.01)
    pump(app, 0.3)

    assert got == ["done"], got
    assert ("info", "Config built: NADROWSKI") in cap.lines, cap.lines
    assert ("warning", "[tsnpe] loaded a NON-AMORTIZED posterior") in cap.lines, cap.lines
    texts = [t for _level, t in cap.lines]
    assert texts.index("Config built: NADROWSKI") < \
        texts.index("[tsnpe] loaded a NON-AMORTIZED posterior"), cap.lines
    assert panel.log_pane.toPlainText() == before, \
        "a captured batch must not also reach the widget"


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


def test_the_panel_and_a_plain_widget_show_one_shared_refusal_box(monkeypatch):
    """One yellow box, two callers. BasePanel._refusal keeps its signature and its log-pane line, but
    the box itself is core/gui/widgets/refusal_box.show_refusal(parent, exc) -- so the Artifacts
    screen, which is a plain QWidget and NOT a BasePanel on purpose (piece 4, B1: a BasePanel enrols
    in _instances, and a run in any panel would then grey out the browser while you were reading a
    log), shows the same title, icon, text, fix sentence and single OK button instead of a lookalike
    that drifts from this one.

    Two legs, because "the same box" and "the same code" are different claims. The first compares the
    two boxes field by field off the conftest's SHOWN record. The second pins the DELEGATION: a
    copy-pasted twin in base_panel.py would pass the first leg today and rot the first time either
    copy is edited."""
    from PySide6.QtWidgets import QMessageBox, QWidget
    from core.gui.panels import base_panel as bp
    from core.gui.widgets.refusal_box import show_refusal
    from core.refusals import Refusal
    from tests._fixtures import SHOWN, PaneCapture, qt_app

    qt_app()

    class P(BasePanel):
        pass

    panel = P()
    pane = PaneCapture(panel)
    holder = QWidget()                      # a parent that is NOT a panel; kept bound for its lifetime
    fix = "Set it in the 'T_obs (s)' box on the Infer tab."
    exc = Refusal("The observation length, in seconds, is blank (default none: it must be given).",
                  field="t_obs")

    SHOWN.clear()
    panel._refusal(exc)
    show_refusal(holder, exc)
    assert len(SHOWN) == 2, [b.text() for b in SHOWN]
    for who, box in (("panel", SHOWN[0]), ("widget", SHOWN[1])):
        assert isinstance(box, QMessageBox), who
        assert box.windowTitle() == "Check your inputs", who
        assert box.icon() == QMessageBox.Warning, who
        assert box.text() == exc.message, who
        assert box.informativeText() == fix, who
        assert box.detailedText() == "", f"{who}: a refusal carries no traceback"
        assert box.standardButtons() == QMessageBox.Ok, who
        assert box.buttonRole(box.defaultButton()) == QMessageBox.AcceptRole, f"{who}: OK is default"
    assert pane.lines[-1] == ("warning", f"{exc.message} {fix}")

    # field=None through both callers: no fix sentence, so no informative line at all
    SHOWN.clear()
    bare = Refusal("resume='require' but there is no resumable cache at x.")
    panel._refusal(bare)
    show_refusal(holder, bare)
    assert [b.informativeText() for b in SHOWN] == ["", ""]
    assert [b.text() for b in SHOWN] == [bare.message, bare.message]

    # the delegation itself: the panel must CALL the shared helper and keep no box of its own
    calls = []
    monkeypatch.setattr(bp, "show_refusal", lambda parent, e: calls.append((parent, e)))
    SHOWN.clear()
    panel._refusal(exc)
    assert calls == [(panel, exc)], "BasePanel._refusal no longer routes through refusal_box"
    assert SHOWN == [], "the panel built a box of its own instead of delegating"
    assert pane.lines[-1] == ("warning", f"{exc.message} {fix}"), \
        "the log-pane line is the panel's own (a plain QWidget has no pane) and must stay"


# ── the artifact browser's table (piece 4, design §3.2) ──────────────────────────────────────────
# NOTHING BELOW READS A STORE. columns_for/cells_for are pure over a Summary (B2: a row needs no
# second manifest read), so every formatting leg builds by hand the Summary it wants, including the
# shapes a real store is slow or awkward to produce -- a cache mid-run, a posterior whose manifest
# records no amortization, a directory with no manifest at all.


def _row(kind, **over):
    """A COMPLETE ``Summary`` of ``kind``, with every field every kind shows set to a recognisable
    value. The cache-only ones (``batches_done``, ``batches_planned``, ``rows``) and the per-kind ones
    (``mode``, ``width``, ``amortized``, ``variant``) keep their dataclass defaults, so a leg that
    wants them passes them: ``over`` replaces any field, and that is how the awkward shapes -- a cache
    mid-run, a posterior with no recorded amortization -- are reached."""
    base = dict(kind=kind, id="20260914T102231", name="run-a", created="2026-09-14T10:22:31",
                note="a note", path=Path("nowhere"), complete=True, reason=None,
                dir_name="run-a__20260914T102231", finished=True)
    base.update(over)
    return Summary(**base)


def _bad_row(kind, **over):
    """An INCOMPLETE ``Summary``, laid out exactly as ``ArtifactStore.list`` builds one
    (core/artifacts/store.py:342): the directory name in both ``id`` and ``dir_name``, no name, no
    created, no note, and a reason."""
    base = dict(kind=kind, id="halfwritten", name="", created="", note="", path=Path("nowhere"),
                complete=False, reason="no manifest.json (incomplete or interrupted)",
                dir_name="halfwritten", finished=False)
    base.update(over)
    return Summary(**base)


def test_columns_for_names_all_seven_kinds_after_name_and_created():
    """Every kind's header row, verbatim (design §3.2's table), and the set of kinds is CLOSED against
    KIND_DIRS: a new kind added to the store must gain a column list here or fail this, rather
    than reaching the browser as a KeyError at the click. "Name" and "Created" lead every kind and
    "Note" ends every kind; what differs in between is what only that kind has."""
    import pytest

    assert set(at._EXTRA_COLUMNS) == set(KIND_DIRS), "a kind has no column list (or has a stale one)"
    assert columns_for("prior") == ("Name", "Created", "Note")
    assert columns_for("simulation") == ("Name", "Created", "Progress", "Finished", "Note")
    assert columns_for("posterior") == ("Name", "Created", "Mode", "Width", "Amortized", "Note")
    assert columns_for("observation") == ("Name", "Created", "Mode", "Width", "Note")
    assert columns_for("calibration") == ("Name", "Created", "Note")
    assert columns_for("inference") == ("Name", "Created", "Note")
    assert columns_for("diagnostic") == ("Name", "Created", "Variant", "Note")
    assert columns_for("fdt") == ("Name", "Created", "Study", "Points", "Finished", "Note")
    for kind in KIND_DIRS:
        cols = columns_for(kind)
        assert cols[:2] == ("Name", "Created") and cols[-1] == "Note", kind
    with pytest.raises(KeyError):
        columns_for("plot")             # loud: the kind comes from the screen's own KIND_DIRS selector


def test_cells_for_renders_each_kind_from_the_summary_alone():
    """One cell per column, every one a string, and the same length as the header. The values come off
    the Summary and nothing else -- no store, no manifest, no disk (B2)."""
    for kind in KIND_DIRS:
        s = _row(kind)
        cells = cells_for(kind, s)
        assert len(cells) == len(columns_for(kind)), kind
        assert all(isinstance(c, str) for c in cells), (kind, cells)
        assert cells[0] == "run-a" and cells[1] == "2026-09-14T10:22:31", kind
        assert cells[-1] == "a note", kind
    assert cells_for("prior", _row("prior")) == ("run-a", "2026-09-14T10:22:31", "a note")
    assert cells_for("calibration", _row("calibration", note="")) == (
        "run-a", "2026-09-14T10:22:31", "")
    assert cells_for("observation", _row("observation", mode="chi", width=18)) == (
        "run-a", "2026-09-14T10:22:31", "chi", "18", "a note")
    assert cells_for("diagnostic", _row("diagnostic", variant="laplace")) == (
        "run-a", "2026-09-14T10:22:31", "laplace", "a note")
    # ablation records variant=None deliberately (core/diagnostics/ablation.py:206): blank, not "None"
    assert cells_for("diagnostic", _row("diagnostic", variant=None))[2] == ""
    # an observation carries no amortization, so it gets no such column at all
    assert "Amortized" not in columns_for("observation")


def test_a_caches_progress_and_whether_it_finished_come_apart():
    """B3. ``complete`` means "has a valid manifest" and a cache is manifested from its first batch on,
    so a row must say BOTH: the progress, and whether the run finished. Both halves of the fraction
    are on the Summary -- ``batches_done`` and ``batches_planned``, the latter lifted from
    ``body["identity"]["n_runs"]`` by the store task -- so the cell reads "3/4 batches" and an
    operator can see how far a running cache has to go. The rows-per-batch list is different: it is
    written only by ``mark_complete`` (core/SBI/training_checkpoint.py:346), so a running cache has no
    row count and the cell adds one only once it exists."""
    running = _row("simulation", name="", batches_done=3, batches_planned=4, rows=None, finished=False)
    done = _row("simulation", name="", batches_done=4, batches_planned=4, rows=(24, 24, 24, 24),
                finished=True)
    assert cells_for("simulation", running)[2:4] == ("3/4 batches", "no")
    assert cells_for("simulation", done)[2:4] == ("4/4 batches · 96 rows", "yes")
    assert cells_for("simulation", _row("simulation", batches_done=1, batches_planned=4))[2] == \
        "1/4 batches"
    # a manifest that records no planned count (nothing writes one today, but a hand-edited or an
    # older manifest can) drops to the count alone rather than showing "1/None"
    assert cells_for("simulation", _row("simulation", batches_done=1, batches_planned=None))[2] == \
        "1 batch"
    assert cells_for("simulation", _row("simulation", batches_done=3, batches_planned=None))[2] == \
        "3 batches"
    # nothing to say at all: no batch count, so no cell -- not "0 batches", which would claim a fact
    assert cells_for("simulation", _row("simulation", batches_done=None))[2] == ""
    # both are complete-with-a-manifest, so `complete` cannot be what the cell is wired to: flip
    # ONLY `finished` and the answer flips with it
    assert columns_for("simulation")[3] == "Finished"
    assert cells_for("simulation", _row("simulation", finished=False))[3] == "no"
    assert cells_for("simulation", _row("simulation", finished=True))[3] == "yes"
    # a kind with no progress to report leaves the cell out entirely
    assert "Progress" not in columns_for("prior")


def test_an_fdt_rows_points_are_a_sweeps_fraction_and_blank_for_anything_else():
    """Piece 5 (E4). The Points cell is a SWEEP's operating points as a fraction, with the failures
    named when there are any. A single run and a comparison carry no points (``body["points"]`` is
    null, so ``points_done`` is None), and their cell is blank rather than "0/0", which would claim
    a fact. Finished is ``finished``, never ``complete``, exactly as for a cache (E2)."""
    sweep = _row("fdt", study="sweep", points_done=10, points_planned=12, points_failed=0)
    assert cells_for("fdt", sweep)[2:5] == ("sweep", "10/12", "yes")
    assert cells_for("fdt", _row("fdt", study="sweep", points_done=10, points_planned=12,
                                 points_failed=2))[3] == "10/12 · 2 failed"
    # a planned total the body does not carry: the count alone, not "10/None"
    assert cells_for("fdt", _row("fdt", points_done=10, points_planned=None))[3] == "10"
    assert cells_for("fdt", _row("fdt", study="single", finished=False))[2:5] == ("single", "", "no")


def test_a_posteriors_amortization_is_spelled_out_and_the_norm_is_named_too():
    """The column, unlike the picker's item text (§6.3, which suffixes the exception only), names both
    states: a table column that is blank for the common case reads as missing data."""
    assert cells_for("posterior", _row("posterior", mode="chi", width=18, amortized=True)) == (
        "run-a", "2026-09-14T10:22:31", "chi", "18", "amortized", "a note")
    assert cells_for("posterior", _row("posterior", amortized=False))[4] == "narrowed (TSNPE)"
    assert cells_for("posterior", _row("posterior", amortized=None))[4] == ""
    assert cells_for("posterior", _row("posterior", mode=None, width=None))[2:4] == ("", "")


def test_an_unnamed_artifact_shows_its_label():
    """``Summary.label`` is ``(unnamed <id>)``, which is what the pickers show, so the table and the
    dropdowns name the same artifact the same way. Every simulation cache is unnamed by construction
    (``write_simulation_manifest`` always writes ``name=""``)."""
    s = _row("prior", name="")
    assert s.label == "(unnamed 20260914T102231)"
    assert cells_for("prior", s)[0] == "(unnamed 20260914T102231)"


def test_an_incomplete_row_shows_its_directory_name_and_its_reason():
    """A directory with no usable manifest has no id, no name, no created and no note -- six blank
    cells would read as an artifact with nothing in it. It is identified by its ``dir_name`` (the
    handle ``remove_incomplete`` takes) and carries its reason in the third column, in place of the
    kind's own columns."""
    for kind in KIND_DIRS:
        cells = cells_for(kind, _bad_row(kind))
        assert len(cells) == len(columns_for(kind)), kind
        assert cells[0] == "halfwritten", kind
        assert cells[1] == "", kind
        assert cells[2] == "no manifest.json (incomplete or interrupted)", kind
        assert all(c == "" for c in cells[3:]), (kind, cells)
    assert cells_for("simulation", _bad_row("simulation", reason=None))[2] == ""


def test_the_table_is_flat_read_only_and_selects_whole_rows():
    """A QTreeWidget used FLAT, not a hand-built grid of QLabels: the chi probe table's and
    SettingsScreen.refresh_models' row-building idiom neither sorts nor scales, and the store can hold
    hundreds of rows. No expander column in front of "Name", no in-place editing (a note is edited in
    its own box, so a double-click must not turn a cell into a line edit), one row at a time because
    delete is one artifact at a time (B6)."""
    from PySide6.QtWidgets import QAbstractItemView

    qt_app()
    table = ArtifactTable()
    assert table.rootIsDecorated() is False
    assert table.isSortingEnabled() is True
    assert table.selectionBehavior() == QAbstractItemView.SelectRows
    assert table.selectionMode() == QAbstractItemView.SingleSelection
    assert table.editTriggers() == QAbstractItemView.NoEditTriggers
    assert table.current_summary() is None, "an empty table has no selection"


def test_set_rows_fills_the_header_and_hands_back_the_selected_summary():
    """The table holds the Summary objects it was given and returns the selected one, so a caller
    never parses a cell back into a fact -- the detail pane, Note, Delete and Sweep all need the id,
    the kind and the dir_name, none of which is on screen in full."""
    qt_app()
    table = ArtifactTable()
    rows = [_row("posterior", name="beta", created="2026-09-14T10:22:31", amortized=False, width=18,
                 mode="chi"),
            _row("posterior", name="alpha", created="2026-09-10T08:00:00", id="20260910T080000")]
    table.set_rows("posterior", rows)

    assert table.topLevelItemCount() == 2
    header = table.headerItem()
    assert [header.text(c) for c in range(table.columnCount())] == list(columns_for("posterior"))
    first = table.topLevelItem(0)
    assert [first.text(c) for c in range(table.columnCount())] == list(
        cells_for("posterior", rows[0])), "the default sort must put the newest first"

    table.setCurrentItem(first)
    got = table.current_summary()
    assert got is rows[0] and got.id == "20260914T102231"
    table.clearSelection()
    assert table.current_summary() is None


def test_the_default_sort_is_newest_first_with_the_incomplete_rows_last():
    """ArtifactStore.list already returns that order (newest first, incomplete last) and the table
    must not undo it. A QTreeWidget with sorting enabled sorts by column 0 ascending unless told
    otherwise, which would put an alphabetical Name order in front of an operator looking for the run
    they just made -- so the default is "Created" descending. That the incomplete row also lands last
    is _Row's doing and not this constant's; the next test is the one that pins it."""
    qt_app()
    table = ArtifactTable()
    rows = [_row("prior", name="newer", created="2026-09-14T10:22:31"),
            _row("prior", name="older", created="2026-09-10T08:00:00", id="20260910T080000"),
            _bad_row("prior")]
    table.set_rows("prior", rows)
    assert at.DEFAULT_SORT == (1, 1)
    assert table.sort_state() == (1, 1)
    assert [table.topLevelItem(i).text(0) for i in range(3)] == ["newer", "older", "halfwritten"]


def test_an_incomplete_row_sorts_last_under_every_column_and_either_order():
    """Incomplete rows sort last is a PROPERTY of the table (design §3.2), not a side effect of an
    incomplete row's blank `created`. Three of the four sorts below break the accident: sorted by
    Name, "halfwritten" lands BETWEEN "alpha" and "newer" in either direction, and sorted by Created
    ASCENDING its blank cell floats to the very top -- only the default, Created descending, is right
    on its own. So _Row.__lt__ ranks on completeness before the column's own value, and flips that
    rank for a descending sort because Qt inverts the whole comparison; the leftover directories then
    stay at the bottom under every column and in both directions, out of the way of the newest real
    artifact, which is where a selection-driven Delete wants them."""
    qt_app()
    table = ArtifactTable()
    rows = [_row("prior", name="newer", created="2026-09-14T10:22:31"),
            _row("prior", name="alpha", created="2026-09-10T08:00:00", id="20260910T080000"),
            _bad_row("prior")]
    table.set_rows("prior", rows)

    def names():
        return [table.topLevelItem(i).text(0) for i in range(table.topLevelItemCount())]

    table.apply_sort_state(0, 0)                    # Name, ascending: "alpha" < "halfwritten"
    assert names() == ["alpha", "newer", "halfwritten"]
    table.apply_sort_state(0, 1)                    # Name, descending
    assert names() == ["newer", "alpha", "halfwritten"]
    table.apply_sort_state(1, 0)                    # Created, ascending: the blank one does NOT lead
    assert names() == ["alpha", "newer", "halfwritten"]
    table.apply_sort_state(1, 1)                    # Created, descending (the default)
    assert names() == ["newer", "alpha", "halfwritten"]

    # the rows keep their Summaries through every reorder: the item carries its INDEX, not its cells
    table.setCurrentItem(table.topLevelItem(2))
    assert table.current_summary() is rows[2] and not table.current_summary().complete


def test_the_sort_state_round_trips_and_a_narrower_kind_falls_back():
    """What the screen remembers is the kind and the sort (§3.5) -- two PLAIN ints, because QSettings
    stores ints and a Qt enum does not survive the round trip. Plain both ways, and not as a
    convenience: `int(Qt.SortOrder)` raises `TypeError` in PySide6 6.9.3, so the order is read as
    `header().sortIndicatorOrder().value` inside the widget and no caller ever holds a Qt enum. The
    kinds have different widths, so a sort on the posterior's "Amortized" column cannot survive a
    switch to the prior kind: it falls back to the default rather than to column 0, which would
    quietly re-sort by Name -- neither the order the store hands back nor one the user asked for."""
    qt_app()
    table = ArtifactTable()
    posteriors = [_row("posterior", name="beta", amortized=True),
                  _row("posterior", name="alpha", amortized=False, id="20260910T080000",
                       created="2026-09-10T08:00:00")]
    table.set_rows("posterior", posteriors)
    assert [type(v) for v in table.sort_state()] == [int, int], \
        "a Qt enum crossed the seam; int(Qt.SortOrder) raises TypeError on this PySide6"

    table.apply_sort_state(0, 0)                       # by Name, ascending
    assert table.sort_state() == (0, 0)
    assert [table.topLevelItem(i).text(0) for i in range(2)] == ["alpha", "beta"]
    table.apply_sort_state(0, 1)
    assert table.sort_state() == (0, 1)
    assert [table.topLevelItem(i).text(0) for i in range(2)] == ["beta", "alpha"]

    table.apply_sort_state("1", "0")                   # what a QSettings round trip hands back
    assert table.sort_state() == (1, 0)

    table.apply_sort_state(4, 0)                       # by Amortized: posterior-only, column 4
    assert table.sort_state() == (4, 0)
    table.set_rows("posterior", posteriors)
    assert table.sort_state() == (4, 0), "a refill must keep the sort the user chose"

    table.set_rows("prior", [_row("prior", name="p")])  # three columns: 4 is gone
    assert table.columnCount() == 3
    assert table.sort_state() == at.DEFAULT_SORT

    table.apply_sort_state(9, 0)                       # out of range: ignored, never clamped to 0
    assert table.sort_state() == at.DEFAULT_SORT


def test_selection_changed_fires_on_a_selection_and_a_refresh_drops_it():
    """One signal, so the detail pane, the Note box and the two buttons are driven from one place. A
    refresh clears the table, so the selection goes with it and the pane must tolerate a None -- the
    alternative, remembering an id across a refresh, is exactly the dangling state §3.4 protects the
    pickers from."""
    qt_app()
    table = ArtifactTable()
    fired = []
    table.selection_changed.connect(lambda: fired.append(1))
    rows = [_row("prior", name="p"), _row("prior", name="q", id="20260910T080000",
                                          created="2026-09-10T08:00:00")]
    table.set_rows("prior", rows)
    assert fired == [], "filling an empty table selects nothing"

    table.setCurrentItem(table.topLevelItem(0))
    assert fired, "selecting a row must emit selection_changed"
    assert table.current_summary() is not None

    fired.clear()
    table.set_rows("prior", rows)
    assert table.current_summary() is None, "a refresh must drop the stale selection"
    assert fired, "clearing the selection must emit selection_changed too"


def test_the_stylesheet_paints_the_item_view_in_both_themes():
    """core/gui/design.py styled NO item view before this piece: its one QAbstractItemView rule is the
    combo popup's (design.py:237). So a QTreeWidget would have rendered in the Fusion default -- a
    white grid in dark mode under a header matching nothing else. The rules go in design.py, in the
    same token vocabulary as the rest, so a theme flip recolours them for free (theming.Appearance
    re-applies build_qss on every change).

    The substitution is the real regression risk: _QSS is a string.Template, so one $name that is not
    a key of _qss_vars raises KeyError inside substitute() and takes the WHOLE stylesheet down, not
    just the new block. Both themes and an accent override are built here for that reason."""
    from core.gui import design

    for dark in (False, True):
        qss = design.build_qss(dark)
        for selector in ("QTreeView {", "QTreeView::item", "QHeaderView::section"):
            assert selector in qss, (dark, selector)
        t = design.tokens(dark)
        assert f"alternate-background-color: {t['alt_base']}" in qss, dark
        assert f"background: {t['alt_base']}; color: {t['text_2nd']}" in qss, (dark, "header section")
        assert "$" not in qss, "an unsubstituted token escaped"
    # an accent override is the substitution most likely to break, and "QTreeView" is a literal of
    # the template, so its presence proves nothing: what is pinned is that the table's own rule
    # RESOLVED -- the override reached its selection colour, and no placeholder survived anywhere.
    accented = design.build_qss(True, "#AA3366")
    assert "$" not in accented, "an unsubstituted token escaped under an accent override"
    block = accented.split("QTreeView {", 1)[1].split("}", 1)[0]
    assert "selection-background-color: #AA3366" in block, block


def test_a_sort_applied_before_the_first_fill_is_held_and_not_lost():
    """The screen restores its remembered sort from settings (§3.5), and nothing says it must do so
    AFTER its first refresh. Before a fill there is no header, so the sort cannot be applied then and
    there -- it is held and spent by the next set_rows, which makes "the sort survives a relaunch"
    true whichever order the screen calls the two in, rather than a sequencing rule the next task has
    to read a docstring to obey. Held, not stored: once spent it cannot come back over a sort the
    user has since chosen."""
    qt_app()
    rows = [_row("prior", name="newer", created="2026-09-14T10:22:31"),
            _row("prior", name="alpha", created="2026-09-10T08:00:00", id="20260910T080000")]

    # (a) restored BEFORE the first fill: the fill must honour it, not DEFAULT_SORT
    table = ArtifactTable()
    table.apply_sort_state(0, 0)                        # by Name, ascending
    assert table.sort_state() == (0, 0), "what was applied must read back, even before a header"
    table.set_rows("prior", rows)
    assert table.sort_state() == (0, 0) != at.DEFAULT_SORT
    assert [table.topLevelItem(i).text(0) for i in range(2)] == ["alpha", "newer"]

    # (b) a SECOND fill keeps the sort chosen since; the spent value does not resurrect
    table.apply_sort_state(1, 1)
    table.set_rows("prior", rows)
    assert table.sort_state() == (1, 1)
    assert [table.topLevelItem(i).text(0) for i in range(2)] == ["newer", "alpha"]

    # (c) after a fill, apply_sort_state is exactly what it was: immediate, out of range ignored
    table.apply_sort_state(0, 1)
    assert table.sort_state() == (0, 1)
    assert [table.topLevelItem(i).text(0) for i in range(2)] == ["newer", "alpha"]
    table.apply_sort_state(9, 0)
    assert table.sort_state() == (0, 1), "an out-of-range column is ignored, not held for later"

    # a held sort naming a column the first kind lacks falls back -- and is spent, not carried to a
    # later kind that happens to be wide enough for it
    other = ArtifactTable()
    other.apply_sort_state(4, 0)                        # the posterior's "Amortized": not a prior's
    other.set_rows("prior", rows)
    assert other.sort_state() == at.DEFAULT_SORT
    other.set_rows("posterior", [_row("posterior", name="p")])
    assert other.sort_state() == at.DEFAULT_SORT, "a spent sort came back on a wider kind"


# ── piece 4, B14: the window's root sink ─────────────────────────────────────────────────────────
def test_the_windows_root_sink_prefixes_a_library_record_and_resolves_its_stream_late(capsys):
    """Below WARNING to ``sys.stdout``, at WARNING and above to ``sys.stderr``, BOTH RESOLVED AT EMIT
    TIME, as ``library: <logger name>: <message>``.

    Resolving late is the FIX, not the bug: what broke was basicConfig's handler binding the stream at
    CONSTRUCTION and then writing into a stopped pump for the rest of the process, and
    core/tool/logging_console.py already states the resolve-late rule for exactly this reason.
    Resolving late also solves three problems at once with no plumbing -- DURING A RUN those two names
    ARE the run's ``_SignalStream``s, which are thread-safe by construction and feed the pump, so a
    library record arriving on the WORKER thread never touches a widget (a ``log_pane.append_line``
    from the handler would have), and outside a run they are the real console.

    The level split is not cosmetic: ``_SignalStream`` on err is hard-wired to the pane's ``warning``
    level, so routing an information record through stderr would put a warning triangle on it.

    ``logging.warning`` (the module-level function, on the ROOT logger) is sbi's own shape -- its
    leakage warnings at sbi/samplers/rejection/rejection.py:336,359 are that call -- and its logger
    name is ``root``, which names nothing. So for that ONE name the prefix falls back to
    ``record.module``, the basename of the file that logged: ``library: rejection:`` under sbi, and
    this test file's own stem here. Walkthrough row D15 expects that module name. A library's
    ``log.exception`` keeps its traceback, with the prefix on the first line only.

    Fix round 1 -- PRISM's own records. BETWEEN RUNS no handler sits on ``core``, so a ``core``
    warning reaches this sink rather than vanishing, and it is written in the tool's own shape
    (``warning: ...``), never ``library:``. DURING A RUN the pump's handler on ``core`` has already
    emitted it, so it lands in the pane exactly once, at its own level, and the sink stays silent.
    """
    import logging

    from core import logging_root, runs
    from core.gui.app import _library_record_sink
    from tests._fixtures import pump, qt_app

    app = qt_app()
    lib = logging.getLogger("a_library")
    lib.setLevel(logging.INFO)          # a library that lowers its own level; the root sits at WARNING
    logging_root.install(_library_record_sink)
    try:
        capsys.readouterr()
        lib.info("an information record")
        logging.warning("Only 0.5% proposal samples are accepted.")
        cap = capsys.readouterr()
        here = Path(__file__).stem          # record.module: the file that called logging.warning
        assert cap.out == "library: a_library: an information record\n", cap.out
        assert cap.err == f"library: {here}: Only 0.5% proposal samples are accepted.\n", cap.err

        try:
            raise ValueError("the library's own failure")
        except ValueError:
            lib.exception("it failed")
        err = capsys.readouterr().err
        assert err.startswith("library: a_library: it failed\nTraceback (most recent call last):\n"), err
        assert err.endswith("ValueError: the library's own failure\n"), err
        assert sum(ln.startswith("library:") for ln in err.splitlines()) == 1, err

        # between runs: no core handler, so the record reaches the sink -- as PRISM's own voice
        assert runs.LOGGER.handlers == [], runs.LOGGER.handlers
        capsys.readouterr()
        logging.getLogger("core.probe").warning("the pipeline's own voice, between runs")
        logging.getLogger("core.probe").info("and its information record")
        cap = capsys.readouterr()
        assert cap.err == "warning: the pipeline's own voice, between runs\n", cap.err
        assert cap.out == "and its information record\n", cap.out

        signals = WorkerSignals()
        lines = []
        signals.log_batch.connect(lambda batch: lines.extend(batch))
        signals.rows.connect(lambda _s: None)
        with redirect_streams(signals):
            lib.warning("a leaky posterior")
            lib.info("and a quiet note")
            logging.getLogger("core.probe").warning("the pipeline's own voice, during a run")
        pump(app)
        assert ("library: a_library: a leaky posterior", "warning") in lines, lines
        assert ("library: a_library: and a quiet note", "info") in lines, lines
        # during a run: the pump's own handler emitted it -- once, at its level, and nothing else did
        mine = [ln for ln in lines if "the pipeline's own voice, during a run" in ln[0]]
        assert mine == [("the pipeline's own voice, during a run", "warning")], lines
    finally:
        logging_root.remove()
        lib.setLevel(logging.NOTSET)


# ── piece 4, B15: the deferred-cancel critical section ───────────────────────────────────────────
def test_cancel_deferred_defers_a_cancel_and_never_discards_it():
    """Inside the section neither cancel checkpoint fires -- ``_SignalStream.write`` (every print and
    every tqdm redraw) and ``_PumpLogHandler.emit`` (every ``core`` record) -- and the records still
    flow. What must NOT happen is the cancel being lost: the token stays REQUESTED, its latch stays
    unfired, and the very next check OUTSIDE the block raises exactly as it would have.

    Re-entrant, because the section nests: Task 22 puts the pipeline's rescue block inside one and the
    ``training_checkpoint.save`` it calls opens another, so a plain boolean would be cleared by the
    inner block's exit and leave the rest of the outer block unprotected."""
    import logging

    import pytest

    from core import runs
    from core.gui.streams import CancelToken, WorkerCancelled, _PumpLogHandler, _SignalStream

    class _FakePump:
        def __init__(self):
            self.logs = []

        def sink(self, kind, payload):
            self.logs.append((kind, payload))

    token = CancelToken()
    token.arm()                                # redirect_streams does this on the worker thread
    token.requested.set()                      # Cancel pressed, latch not yet fired
    fake = _FakePump()
    stream = _SignalStream(fake, "out", "info", token)
    handler = _PumpLogHandler(fake, token)
    record = logging.LogRecord("core.probe", logging.INFO, __file__, 1, "inside the commit", (), None)

    assert runs.cancel_is_deferred() is False, "no section is active outside one"
    with runs.cancel_deferred():
        assert runs.cancel_is_deferred() is True
        with runs.cancel_deferred():           # nested: the section must count, not toggle
            stream.write("a line written mid-commit\n")
            handler.emit(record)
        assert runs.cancel_is_deferred() is True, "the inner block's exit ended the outer section"
        stream.write("and another, still inside the outer section\n")
        assert token.fired is False and token.requested.is_set(), \
            "the cancel fired inside the section instead of being deferred"
    assert runs.cancel_is_deferred() is False

    assert ("log", ("inside the commit", "info")) in fake.logs, fake.logs
    assert any(kind == "log" and "a line written mid-commit" in payload[0]
               for kind, payload in fake.logs), fake.logs

    with pytest.raises(WorkerCancelled):
        stream.write("the first write after the section\n")
    assert token.fired is True, "the deferred cancel was discarded instead of deferred"


def test_the_checkpoint_commit_runs_inside_the_deferred_cancel_section(tmp_path, monkeypatch):
    """Steps 1-3 of a checkpoint save -- the shard writes, the state.prev copy and the atomic replace
    of state.pt -- are the two-writes-that-must-both-happen case the section exists for: a record
    emitted between the shard fsync and the state replace would raise mid-commit and leave a
    checkpoint pointing at data still in the page cache. The rule in CLAUDE.md and in
    training_checkpoint's module docstring stays, and so does the source-reading test that polices it
    (tests/test_user_sbi.py::test_nothing_prints_or_logs_inside_a_checkpoint_commit): the section is
    the guard, that test is the proof the guard is where it is claimed to be.

    The WHOLE save is ONE section, the manifest refresh included (the test below says why that one is
    inside): every step is recorded with the entry number of the section open around it, so a commit
    split into two sections with a gap between them fails here as surely as a step left outside. Two
    saves, because only the second finds a state.pt to copy to state.prev.pt."""
    import contextlib
    from pathlib import Path

    from core import runs
    from core.SBI import training_checkpoint as tc

    entries, open_now, steps = [], [], []
    real_section = tc.cancel_deferred

    @contextlib.contextmanager
    def counted_section():
        entries.append(len(entries))
        open_now.append(entries[-1])
        try:
            with real_section():
                yield
        finally:
            open_now.pop()

    def recorded(fn, name):
        def wrapped(*args, **kwargs):
            steps.append((name(*args), open_now[0] if open_now else None, runs.cancel_is_deferred()))
            return fn(*args, **kwargs)
        return wrapped

    monkeypatch.setattr(tc, "cancel_deferred", counted_section)
    monkeypatch.setattr(tc, "atomic_torch_save",
                        recorded(tc.atomic_torch_save, lambda obj, dest: f"write {Path(dest).name}"))
    monkeypatch.setattr(tc.shutil, "copyfile",
                        recorded(tc.shutil.copyfile, lambda src, dst: f"copy {Path(src).name}"))
    monkeypatch.setattr(tc, "_refresh_manifest", recorded(tc._refresh_manifest, lambda path: "manifest"))

    d = tmp_path / "commit"
    x, th = torch.arange(12.).reshape(4, 3), torch.arange(8.).reshape(4, 2)
    expected = {0: ["write x_000000_000001.pt", "write th_000000_000001.pt", "write state.pt",
                    "manifest"],
                1: ["write x_000001_000002.pt", "write th_000001_000002.pt", "copy state.pt",
                    "write state.pt", "manifest"]}
    for from_batch in (0, 1):
        del steps[:]
        before = len(entries)
        tc.save(d, from_batch=from_batch, batch_k=from_batch + 1, rng={},
                x_buf=x, th_buf=th, run_size=2)
        assert len(entries) == before + 1, \
            f"save {from_batch}: the commit must be ONE section, not {len(entries) - before}"
        assert [name for name, _, _ in steps] == expected[from_batch], steps
        assert all(entry == before and deferred for _, entry, deferred in steps), \
            f"save {from_batch}: every step must run inside that one section: {steps}"
        assert runs.cancel_is_deferred() is False, "the section leaked past the commit"


def test_a_cancel_raised_by_anything_the_manifest_refresh_reaches_is_carried_through_the_save(
        tmp_path, monkeypatch):
    """The manifest refresh is inside the section although it is not the commit point, because a
    cancel raised there is NOT harmless. WorkerCancelled is a BaseException, so the refresh's
    ``except Exception`` does not catch it: it escapes save() AFTER state.pt says batches_done = k but
    BEFORE the pipeline advances its own counter, so the next save re-commits a range overlapping the
    shard already on disk -- and load_rows then refuses the whole cache ("commits N batches but only
    M are present on disk").

    Nothing the refresh reaches logs or prints today, and the source-reading silence test only reads
    ``_refresh_manifest``'s OWN body, so this pins it the way the fault would arrive: the store's
    manifest writer, which the refresh calls, learns to log. A Cancel is pressed before the second
    save; both saves must complete, batches_done must advance, the cache must still load -- and the
    cancel must still be taken at the first record OUTSIDE the section."""
    import logging

    import pytest

    from core.artifacts import store
    from core.gui.streams import CancelToken, WorkerCancelled, _PumpLogHandler
    from core.SBI import training_checkpoint as tc

    class _FakePump:
        def __init__(self):
            self.logs = []

        def sink(self, kind, payload):
            self.logs.append((kind, payload))

    real_writer = store.write_simulation_manifest

    def a_writer_that_speaks(path, identity, **kwargs):
        logging.getLogger("core.artifacts.store").info("refreshing the simulation manifest")
        return real_writer(path, identity, **kwargs)

    monkeypatch.setattr(store, "write_simulation_manifest", a_writer_that_speaks)

    run_size, n_runs = 2, 4
    d = tmp_path / "ck"
    x = torch.arange(float(n_runs * run_size * 3)).reshape(n_runs * run_size, 3)
    th = torch.arange(float(n_runs * run_size * 2)).reshape(n_runs * run_size, 2)
    token = CancelToken()
    token.arm()                                # this thread plays the worker
    fake = _FakePump()
    handler = _PumpLogHandler(fake, token)
    core_logger = logging.getLogger("core")
    core_logger.addHandler(handler)
    try:
        tc.create(d, {"model": "task21-probe", "run_size": run_size, "n_runs": n_runs},
                  schedule_t_scales=torch.ones(n_runs), schedule_Ts=torch.ones(n_runs),
                  inits=torch.zeros(run_size, 3), V=None, probe=torch.zeros(0, dtype=torch.float64),
                  run_size=run_size, n_runs=n_runs)
        tc.save(d, from_batch=0, batch_k=1, rng={}, x_buf=x, th_buf=th, run_size=run_size)

        token.requested.set()                  # Cancel pressed, latch not yet fired
        try:
            tc.save(d, from_batch=1, batch_k=2, rng={}, x_buf=x, th_buf=th, run_size=run_size)
        except WorkerCancelled:
            pytest.fail("a record from inside the manifest refresh raised the cancel out of save(): "
                        "the refresh is outside the deferred-cancel section")
        assert token.fired is False and token.requested.is_set(), "the cancel was taken mid-save"
        assert tc.peek(d)["batches_done"] == 2, tc.peek(d)
        xr, thr = tc.load_rows(d, tc.peek(d)["batches_done"], run_size)
        assert torch.equal(xr, x[:4]) and torch.equal(thr, th[:4]), "the cache no longer reads back"
        assert sum(p == ("refreshing the simulation manifest", "info")
                   for k, p in fake.logs if k == "log") == 3, fake.logs   # create + both saves

        with pytest.raises(WorkerCancelled):   # deferred, never discarded
            logging.getLogger("core.probe").info("the first record after the save")
        assert token.fired is True
    finally:
        core_logger.removeHandler(handler)


def test_the_completion_write_runs_inside_the_deferred_cancel_section(tmp_path, monkeypatch):
    """``mark_complete`` is two writes that must both happen: state.pt flips to ``complete`` and the
    manifest follows it. A cancel between them -- or out of anything the manifest refresh reaches,
    whose ``except Exception`` cannot stop a BaseException -- leaves every row committed and the
    manifest still saying the cache is unfinished, so the Artifacts browser labels a finished
    multi-day cache "unfinished" until the next resume happens to rewrite it. So both writes sit in
    one ``runs.cancel_deferred()`` section, as ``save``'s commit does.

    Pinned the way the fault would arrive, as the test above pins ``save``: BOTH writes learn to
    speak -- the state write through ``atomic_torch_save`` and the store's manifest writer -- and a
    Cancel is pressed before the call. Move either write back out of the section and its own record
    raises the cancel before that write happens: state.pt or the manifest is left unfinished and this
    fails by name. The cancel must still be taken at the first record OUTSIDE the section."""
    import logging

    import pytest

    from core.artifacts import manifest as mf
    from core.artifacts import store
    from core.gui.streams import CancelToken, WorkerCancelled, _PumpLogHandler
    from core.SBI import training_checkpoint as tc

    class _FakePump:
        def __init__(self):
            self.logs = []

        def sink(self, kind, payload):
            self.logs.append((kind, payload))

    run_size, n_runs = 2, 2
    d = tmp_path / "ck"
    x = torch.arange(float(n_runs * run_size * 3)).reshape(n_runs * run_size, 3)
    th = torch.arange(float(n_runs * run_size * 2)).reshape(n_runs * run_size, 2)
    tc.create(d, {"model": "task22-probe", "run_size": run_size, "n_runs": n_runs},
              schedule_t_scales=torch.ones(n_runs), schedule_Ts=torch.ones(n_runs),
              inits=torch.zeros(run_size, 3), V=None, probe=torch.zeros(0, dtype=torch.float64),
              run_size=run_size, n_runs=n_runs)
    tc.save(d, from_batch=0, batch_k=n_runs, rng={}, x_buf=x, th_buf=th, run_size=run_size)

    real_state_write, real_writer = tc.atomic_torch_save, store.write_simulation_manifest

    def a_state_write_that_speaks(obj, dest):
        logging.getLogger("core.Helpers.file_manager").info(f"writing {Path(dest).name}")
        return real_state_write(obj, dest)

    def a_writer_that_speaks(path, identity, **kwargs):
        logging.getLogger("core.artifacts.store").info("refreshing the simulation manifest")
        return real_writer(path, identity, **kwargs)

    monkeypatch.setattr(tc, "atomic_torch_save", a_state_write_that_speaks)
    monkeypatch.setattr(store, "write_simulation_manifest", a_writer_that_speaks)

    token = CancelToken()
    token.arm()                                # this thread plays the worker
    fake = _FakePump()
    handler = _PumpLogHandler(fake, token)
    core_logger = logging.getLogger("core")
    core_logger.addHandler(handler)
    try:
        token.requested.set()                  # Cancel pressed, latch not yet fired
        try:
            tc.mark_complete(d, n_runs, rows=(n_runs * run_size, 3))
        except WorkerCancelled:
            pytest.fail("a record from inside the completion write raised the cancel out of "
                        "mark_complete(): a write is outside the deferred-cancel section")
        assert token.fired is False and token.requested.is_set(), "the cancel was taken mid-write"
        said = [p for k, p in fake.logs if k == "log"]
        assert said == [("writing state.pt", "info"), ("refreshing the simulation manifest", "info")], \
            ("both writes must have spoken, or this test measures nothing", fake.logs)
        st = tc.peek(d)
        assert st["complete"] is True and st["batches_done"] == n_runs, st
        m = mf.from_json_text((d / store.MANIFEST).read_text(encoding="utf-8"))
        assert m.body["complete"] is True and m.body["batches_done"] == n_runs, m.body
        assert m.body["rows"] == [n_runs * run_size, 3], m.body

        with pytest.raises(WorkerCancelled):   # deferred, never discarded
            logging.getLogger("core.probe").info("the first record after the completion write")
        assert token.fired is True
    finally:
        core_logger.removeHandler(handler)


def test_cancel_deferred_applies_only_to_the_thread_that_entered_it():
    """PER THREAD: the window runs its task on a worker thread while tqdm's monitor and the GUI thread
    write too, so a section held on ANOTHER thread must not defer the armed thread's cancel. A
    process-wide flag would let any thread's section swallow the worker's checkpoint for as long as
    that section stayed open."""
    import threading

    import pytest

    from core import runs
    from core.gui.streams import CancelToken, WorkerCancelled, _SignalStream

    class _FakePump:
        def sink(self, kind, payload):
            pass

    token = CancelToken()
    token.arm()                                # this thread plays the worker
    token.requested.set()
    stream = _SignalStream(_FakePump(), "out", "info", token)
    inside, release, other = threading.Event(), threading.Event(), {}

    def hold_a_section():
        with runs.cancel_deferred():
            other["deferred"] = runs.cancel_is_deferred()
            inside.set()
            release.wait(10)

    t = threading.Thread(target=hold_a_section, daemon=True)
    t.start()
    try:
        assert inside.wait(10), "the other thread never entered its section"
        assert other["deferred"] is True
        assert runs.cancel_is_deferred() is False, "another thread's section leaked onto this one"
        with pytest.raises(WorkerCancelled):
            stream.write("the worker's write while another thread holds a section\n")
    finally:
        release.set()
        t.join(10)
    assert token.fired is True


# ── piece 4, B15 fix round 1: a cancel waits out an exception handler; a crash stays a crash ──────
def test_a_cancel_waits_out_an_exception_handler_and_never_waits_in_normal_flow():
    """The raise-time deferral at the unit level (``core.runs.cancel_is_deferred``), on both cancel
    checkpoints -- ``_SignalStream.write`` (every print and tqdm redraw) and ``_PumpLogHandler.emit``
    (every ``core`` record).

    (c) THE KEY REGRESSION: in NORMAL flow a requested cancel is taken at the very next check, and it
    raises BEFORE the line is sunk, as it always has. Checked plain and right after the three things a
    sticky or over-broad rule would leak out of: an ``except`` that recovered, a ``finally`` reached
    normally, a ``with`` that exited normally.

    (d) DELAYED, NEVER LOST: inside each of the four ways a thread can be handling an exception -- an
    ``except``, a ``finally`` reached by one, an ``__exit__`` leaving by one, a generator's teardown
    (tqdm's ``finally: self.close()``) -- both checkpoints carry their line instead of raising, the
    token stays REQUESTED with its latch unfired, and the first check after the handler has exited
    raises."""
    import logging

    import pytest

    from core import runs
    from core.gui.streams import CancelToken, WorkerCancelled, _PumpLogHandler, _SignalStream

    class _FakePump:
        def __init__(self):
            self.logs = []

        def sink(self, kind, payload):
            self.logs.append((kind, payload))

    def rigged():
        token, fake = CancelToken(), _FakePump()
        token.arm()                            # this thread plays the worker
        token.requested.set()                  # Cancel pressed, latch not yet fired
        return token, fake, _SignalStream(fake, "out", "info", token), _PumpLogHandler(fake, token)

    def record(text):
        return logging.LogRecord("core.probe", logging.INFO, __file__, 1, text, (), None)

    # (c) normal flow
    def nothing_before():
        pass

    def after_a_recovered_except():
        try:
            raise RuntimeError("recovered")
        except RuntimeError:
            pass

    def after_a_finally_reached_normally():
        try:
            pass
        finally:
            pass

    def after_a_with_that_exited_normally():
        with contextlib.nullcontext():
            pass

    for before in (nothing_before, after_a_recovered_except, after_a_finally_reached_normally,
                   after_a_with_that_exited_normally):
        for channel in ("record", "print"):
            token, fake, stream, handler = rigged()
            before()
            assert runs.cancel_is_deferred() is False, (before.__name__, channel)
            with pytest.raises(WorkerCancelled):
                if channel == "record":
                    handler.emit(record("the very next record"))
                else:
                    stream.write("the very next print\n")
            assert token.fired is True and fake.logs == [], (before.__name__, channel, fake.logs)

    # (d) inside a handler: carried; after it: taken
    class _Swallow:
        def __init__(self, speak):
            self.speak = speak

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            self.speak()
            return True

    def in_an_except(speak):
        try:
            raise RuntimeError("x")
        except RuntimeError:
            speak()

    def in_a_finally_reached_by_one(speak):
        try:
            try:
                raise RuntimeError("x")
            finally:
                speak()
        except RuntimeError:
            pass

    def in_an_exit_leaving_by_one(speak):
        with _Swallow(speak):
            raise RuntimeError("x")

    def in_a_generator_teardown(speak):
        def bar():
            try:
                yield 1
            finally:                           # tqdm.__iter__'s `finally: self.close()`
                speak()
        it = bar()
        next(it)
        it.close()

    for shape in (in_an_except, in_a_finally_reached_by_one, in_an_exit_leaving_by_one,
                  in_a_generator_teardown):
        token, fake, stream, handler = rigged()
        inside = {}

        def speak():
            inside["deferred"] = runs.cancel_is_deferred()
            handler.emit(record("a record inside the handler"))
            stream.write("a print inside the handler\n")

        shape(speak)
        assert inside["deferred"] is True, shape.__name__
        assert token.fired is False and token.requested.is_set(), \
            f"{shape.__name__}: the cancel fired inside the handler"
        said = [payload[0] for kind, payload in fake.logs if kind == "log"]
        assert "a record inside the handler" in said, (shape.__name__, fake.logs)
        assert any("a print inside the handler" in t for t in said), (shape.__name__, fake.logs)
        assert runs.cancel_is_deferred() is False, f"{shape.__name__}: the deferral outlived the handler"
        with pytest.raises(WorkerCancelled):
            handler.emit(record("the first record after the handler"))
        assert token.fired is True, f"{shape.__name__}: the deferred cancel was lost"


_STOP = ("warning", "Run cancelled.")
_NOTED = "A cancel was requested before the run failed"


def _run_leg(app, panel, pane, fn, wait_for_dialog):
    """Dispatch ``fn`` on ``panel`` and drive the event loop until the run has settled: the panel is
    idle and, when a dialog is expected, the box has been shown. Each leg reads only its own pane
    lines and its own dialogs."""
    from tests._fixtures import SHOWN

    SHOWN.clear()
    del pane.lines[:]
    panel.dispatch(fn)
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and (panel._busy or (wait_for_dialog and not SHOWN)):
        app.processEvents()
        time.sleep(0.01)
    pump(app, 0.2)
    assert not panel._busy, "the panel stayed busy"


def test_a_cancel_never_reports_a_failure_as_a_clean_stop():
    """Every leg runs through the WINDOW'S OWN channels -- a real ``print`` and a real ``core`` record
    under the ``redirect_streams`` the dispatch installs -- and requests the PANEL'S OWN token
    (``panel._cancel``, which ``Worker.run`` holds as ``self.cancel``), so a Worker.run that decides
    anything from its token is exercised for real.

    The rule under test is the raise-time deferral (``core.runs.cancel_is_deferred``): no cancel
    checkpoint fires while THIS thread is handling an exception -- inside an ``except``, a ``finally``
    reached by one, an ``__exit__`` leaving by one, a generator's teardown -- or inside an explicit
    ``cancel_deferred()`` section. Normal flow is untouched (leg (c), and the unit test above).

    (a) THE COLLISION -- Cancel pressed in the moment before a crash, in the rescue block's shape: a
    record and a print inside the section, then a bare ``raise``. The ORIGINAL exception reaches
    Worker.run's generic handler: the red box, its traceback in Details, the message in the pane --
    and, because the token had been requested, the cancel NOTED in the pane and in the box.

    (a2) THE RACE THE FIRST DESIGN CALLED RESIDUAL, and it is not rare: ``sdeint.euler_compiled``'s
    ``try ... finally: bar.close()`` sits on the graphed CUDA solver, the likeliest GPU crash site,
    and the bar's closing write is a checkpoint reached while the crash unwinds. It used to raise
    WorkerCancelled chained to the crash, and the crash was told as "Run cancelled.". Now it is a
    crash, reported as one, with the cancel noted.

    (b) A RECOVERED GUARD. The three best-effort guards in core/SBI/pipeline.py --
    ``_release_device_memory``'s ``_try``, ``_try_rng_snapshot`` and ``_try_rng_restore`` -- log from
    INSIDE their ``except`` and carry on. (The OOM ladders do not: they bind a note inside the
    handler and log after it has closed.) A cancel taken inside such a guard would carry the handled
    exception as its ``__context__``. Deferred, it is taken at the first check AFTER the guard, in
    normal flow, with nothing chained: a plain cancel, no dialog, no in-flight line -- and the guard's
    own warning still reaches the pane.

    (c) A CANCEL IN NORMAL FLOW is taken at the very next check: the print it lands on never reaches
    the pane and the line after it never runs.

    (f) THE FALLBACK. A WorkerCancelled raised EXPLICITLY inside a handler -- no checkpoint does that
    any more -- still carries a context. It stays a cancel, and what it carried is kept in the pane at
    ERROR under a HEDGED line, because the chain cannot say whether that exception was still in flight
    or had already been recovered from.

    The live-bar collision, leg (e), is the next test.
    """
    import logging

    from PySide6.QtWidgets import QMessageBox

    from core import runs
    from core.gui.streams import WorkerCancelled
    from tests._fixtures import SHOWN, PaneCapture

    app = qt_app()

    class P(BasePanel):
        pass

    panel = P()
    pane = PaneCapture(panel)
    log = logging.getLogger("core.SBI.pipeline")
    reached = []

    def crash_behind_the_section():
        panel._cancel.requested.set()          # Cancel pressed just before the failure
        try:
            raise RuntimeError("the batch failed")
        except RuntimeError:
            with runs.cancel_deferred():       # what pipeline.py's rescue block opens
                log.info("[checkpoint] stopping: saving 1 completed batches")
                print("a print beside the rescue write")
            raise                              # the ORIGINAL exception, unconditionally

    def crash_through_a_finally_that_speaks():
        panel._cancel.requested.set()
        try:
            raise RuntimeError("the graphed solver failed")
        finally:                               # sdeint.euler_compiled: `finally: bar.close()`
            print("the bar's last frame")
            logging.getLogger("core.Solvers.sdeint").info("closing the solver's bar")

    def recovered_guard_then_normal_flow():
        panel._cancel.requested.set()
        try:
            raise RuntimeError("empty_cache() on a starved card")
        except RuntimeError:
            log.warning("empty_cache() failed during recovery and was ignored")
        reached.append("after the guard")
        log.info("the first record in normal flow")    # the deferred cancel is taken HERE
        reached.append("past the first check")

    def cancel_in_normal_flow():
        panel._cancel.requested.set()
        print("the first print after Cancel")          # taken at this very write
        reached.append("past the first check")

    def cancel_raised_inside_a_handler():
        try:
            raise RuntimeError("an error the run may already have dealt with")
        except RuntimeError:
            raise WorkerCancelled()

    def reported_as_a_crash(message):
        assert _STOP not in pane.lines, ("a run that CRASHED was reported as a cancellation", pane.lines)
        box = SHOWN[-1]
        assert box.windowTitle() == "Error" and box.icon() == QMessageBox.Critical
        assert box.text() == message, box.text()
        assert f"RuntimeError: {message}" in box.detailedText(), box.detailedText()
        assert _NOTED in box.informativeText(), ("the box does not note the cancel", box.informativeText())
        assert pane.lines[-2] == ("error", message), pane.lines
        assert pane.lines[-1][0] == "warning" and _NOTED in pane.lines[-1][1], \
            ("the pane does not note the cancel", pane.lines)

    # (a) the collision: a crash, reported as a crash, with the cancel noted
    _run_leg(app, panel, pane, crash_behind_the_section, True)
    reported_as_a_crash("the batch failed")
    assert ("info", "[checkpoint] stopping: saving 1 completed batches") in pane.lines, pane.lines
    assert ("info", "a print beside the rescue write") in pane.lines, pane.lines

    # (a2) the crash unwinding through a finally that speaks: a crash too
    _run_leg(app, panel, pane, crash_through_a_finally_that_speaks, True)
    reported_as_a_crash("the graphed solver failed")
    assert ("info", "the bar's last frame") in pane.lines, pane.lines
    assert ("info", "closing the solver's bar") in pane.lines, pane.lines

    # (b) a recovered guard, then normal flow: a plain cancel, taken at the first check after it
    del reached[:]
    _run_leg(app, panel, pane, recovered_guard_then_normal_flow, False)
    assert SHOWN == [], "a recovered exception behind a cancel opened a dialog"
    assert pane.lines == [("warning", "empty_cache() failed during recovery and was ignored"), _STOP], \
        pane.lines
    assert reached == ["after the guard"], reached

    # (c) normal flow: the very next check
    del reached[:]
    _run_leg(app, panel, pane, cancel_in_normal_flow, False)
    assert pane.lines == [_STOP] and SHOWN == [] and reached == [], (pane.lines, SHOWN, reached)

    # (f) the fallback: a cancel that still carries a context stays a cancel; the context is kept
    _run_leg(app, panel, pane, cancel_raised_inside_a_handler, False)
    assert SHOWN == [], "an exception behind a cancel opened a dialog"
    assert pane.lines.count(_STOP) == 1 and pane.lines[-1] == _STOP, pane.lines
    assert any(lv == "warning" and "may be a failure that was still in flight" in t
               for lv, t in pane.lines), pane.lines
    assert any(lv == "error" and "RuntimeError: an error the run may already have dealt with" in t
               for lv, t in pane.lines), pane.lines


def test_a_cancel_that_arrived_too_late_is_said_and_the_run_still_reports_success():
    """R6 (whole-piece review). ``cancel_is_deferred()`` defers a checkpoint while this thread is
    handling an exception, and the token stays REQUESTED so "the next check outside raises" -- which
    is true only if there IS a next check. ``Worker.run`` asked the token nothing on the SUCCESS
    path, so a Cancel that arrived inside a deferral window nothing re-checked (a stage whose last
    write happens in a ``finally`` an exception entered -- ``core/Solvers/sdeint.py``'s bar teardown
    is on the graphed solver path) left the run reported as a clean, completed success. The person
    who pressed Cancel was told nothing at all.

    The run really DID finish, so it must not be reported as cancelled: nothing on disk is wrong and
    the payload is correct. It is reported as a SUCCESS with the fact said out loud -- the missing
    counterpart of the failure path's "A cancel was requested before the run failed." (B15's fix
    round 1), which the window has had since.

    A run that finishes with NO cancel pending says nothing extra: the sentence has to be the answer
    to a question somebody asked."""
    from core.gui.worker import CANCEL_TOO_LATE_TEXT
    from tests._fixtures import SHOWN, PaneCapture

    app = qt_app()

    class P(BasePanel):
        pass

    panel = P()
    pane = PaneCapture(panel)
    results = []

    def _drive(fn):
        SHOWN.clear()
        del pane.lines[:]
        del results[:]
        panel.dispatch(fn, on_result=results.append)
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and panel._busy:
            app.processEvents()
            time.sleep(0.01)
        pump(app, 0.2)
        assert not panel._busy, "the panel stayed busy"

    def finishes_with_a_cancel_pending():
        try:
            raise RuntimeError("something the run recovered from")
        except RuntimeError:
            panel._cancel.requested.set()          # Cancel, inside a deferral window
            print("the last thing the run writes")  # deferred: no checkpoint fires here
        return "the payload"                        # ... and nothing checks the token again

    _drive(finishes_with_a_cancel_pending)
    assert results == ["the payload"], "the run's result was not delivered"
    assert SHOWN == [], "a completed run opened a dialog"
    assert ("warning", "Run cancelled.") not in pane.lines, \
        ("a run that FINISHED was reported as a cancellation", pane.lines)
    assert any(lv == "warning" and CANCEL_TOO_LATE_TEXT in text for lv, text in pane.lines), \
        ("the person who pressed Cancel was told nothing", pane.lines)
    assert ("info", "the last thing the run writes") in pane.lines, pane.lines

    _drive(lambda: "a quiet payload")
    assert results == ["a quiet payload"], results
    assert not any("cancel" in text.lower() for _lv, text in pane.lines), \
        ("a run nobody cancelled was told about a cancel", pane.lines)


def test_a_crash_under_a_live_bar_is_a_crash_and_leaves_no_ignored_exception(monkeypatch):
    """(e) THE COLLISION WITH A LIVE BAR. A crash inside ``for ... in tqdm(...)`` finalizes the bar's
    generator while the crash unwinds, and tqdm's ``finally: self.close()`` writes the bar's last frame
    -- a cancel checkpoint. Before the raise-time deferral that write raised WorkerCancelled INSIDE a
    generator finalizer, where Python cannot propagate it: a ~20-line "Exception ignored in:
    <generator object tqdm.__iter__>" block landed in the pane, and the latch was consumed by a raise
    nobody could see. Now the teardown runs while GeneratorExit is being handled, so the write is
    carried: the crash is reported as a crash with the cancel noted, the latch is never consumed, and
    nothing is ignored.

    ``sys.unraisablehook`` is put back to Python's own for the run. pytest installs a collecting hook,
    which would keep the "Exception ignored" block out of the pane and make the last assertion
    vacuous; Python's own writes it to ``sys.stderr``, which under the run is the window's stream."""
    from tests._fixtures import SHOWN, PaneCapture

    monkeypatch.setattr(sys, "unraisablehook", sys.__unraisablehook__)
    app = qt_app()

    class P(BasePanel):
        pass

    panel = P()
    pane = PaneCapture(panel)
    tokens = []

    def crash_under_a_live_bar():
        tokens.append(panel._cancel)           # the panel drops its token when the run finishes
        for i in tqdm(range(1000), desc="a live bar"):
            if i == 3:
                panel._cancel.requested.set()
                raise RuntimeError("the batch failed under a live bar")

    _run_leg(app, panel, pane, crash_under_a_live_bar, True)
    assert _STOP not in pane.lines, ("a run that CRASHED was reported as a cancellation", pane.lines)
    box = SHOWN[-1]
    assert box.text() == "the batch failed under a live bar", box.text()
    assert _NOTED in box.informativeText(), box.informativeText()
    assert tokens and tokens[0].requested.is_set() and tokens[0].fired is False, \
        "the bar's teardown consumed the cancel latch"
    assert not any("Exception ignored" in t for _lv, t in pane.lines), pane.lines
