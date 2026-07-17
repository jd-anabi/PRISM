"""Shared panel scaffolding: a left controls column and a right results area (a figure stack over a
progress pane + log pane), plus ``dispatch()`` to run a callable on a background worker with its
output wired to those widgets."""
import weakref

from PySide6.QtCore import QThreadPool
from PySide6.QtWidgets import (QHBoxLayout, QMessageBox, QPushButton, QScrollArea, QSplitter,
                               QVBoxLayout, QWidget)
from PySide6.QtCore import Qt

from ..plot_watcher import NewPngWatcher
from ..streams import CancelToken
from ..widgets.figure_stack import FigureStack
from ..widgets.log_pane import LogPane
from ..widgets.progress_pane import ProgressPane
from ..worker import Worker


def _png_fig_sink(figure_signal):
    """Return a ``(title, fig) -> None`` sink that renders a matplotlib Figure to PNG bytes ON THE
    WORKER THREAD and emits them. The UI thread then shows a QPixmap -- it never paints a live canvas
    created on the worker thread, which deadlocks on matplotlib's global lock. The figure is closed
    after rendering to free memory.

    It ALSO pickles the figure (best-effort) and ships the bytes alongside the PNG, so the panel can
    rebuild an interactive copy on the GUI thread for the "Pop out" button (FigureStack). Pickling is
    done here on the worker because the Figure is closed right after -- but pickling never renders
    (it does not touch Agg's renderer lock), so it is safe, unlike painting a live canvas. A pickle
    failure must never break the run, so it degrades to None and the pop-out falls back to the image
    viewer; the PNG thumbnail is emitted unconditionally either way.
    """
    def _sink(title, fig):
        import io
        import pickle
        import matplotlib.pyplot as plt
        try:
            fig_pickle = pickle.dumps(fig)      # before savefig -> the pristine figure the stage built
        except Exception:                       # noqa: BLE001 -- pop-out is optional; keep the run alive
            fig_pickle = None
        buf = io.BytesIO()
        try:
            fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
        finally:
            plt.close(fig)
        figure_signal.emit(title, buf.getvalue(), fig_pickle)
    return _sink


class BasePanel(QWidget):
    # Class-level, deliberately: redirect_streams swaps sys.stdout/stderr PROCESS-WIDE (see
    # core/gui/streams.py), so only ONE panel may run at a time -- a per-panel guard would let the FDT
    # tab start a run while the SBI tab is training, and the two would fight over the console. The
    # _REDIRECT lock in streams.py is the backstop; this is the thing that keeps us away from it.
    _running = False
    # The one live run's cancel token (there is only ever one, per _running). MainWindow.closeEvent
    # reaches it through request_cancel_all() to stop a run before quitting.
    _active_cancel: "CancelToken | None" = None
    # Every live panel, so a run in ANY panel can lock the controls of ALL of them (see _set_busy).
    # A WeakSet so panels torn down in tests (or a future dynamic UI) drop out without bookkeeping.
    _instances: "weakref.WeakSet" = weakref.WeakSet()

    @classmethod
    def request_cancel_all(cls) -> None:
        """Ask the currently-running task (if any) to stop at its next checkpoint."""
        if cls._active_cancel is not None:
            cls._active_cancel.requested.set()

    def __init__(self, parent=None):
        super().__init__(parent)
        BasePanel._instances.add(self)
        self._busy = False
        self._cancel: "CancelToken | None" = None
        self._workers = set()   # keep workers alive until 'finished' (else Qt purges its queued signals)

        # Left: subclasses fill self.controls (inside a scroll area so long forms stay usable).
        self.controls = QWidget()
        self.controls_layout = QVBoxLayout(self.controls)
        self.controls_layout.setAlignment(Qt.AlignTop)
        controls_scroll = QScrollArea()
        controls_scroll.setWidgetResizable(True)
        controls_scroll.setWidget(self.controls)
        controls_scroll.setMinimumWidth(340)
        controls_scroll.setMaximumWidth(460)

        # Right: figures over a progress pane over a log. Progress lives in its own widget -- one row
        # per live tqdm bar -- and never touches the log, which only ever appends completed lines.
        self.figure_stack = FigureStack()
        self.progress_pane = ProgressPane()
        self.log_pane = LogPane()

        # Cancel sits on the progress row, hidden until a run starts. It is a "please stop": it sets a
        # flag the pipeline's next print/redraw checks, so the run unwinds at its next checkpoint.
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setVisible(False)
        self.btn_cancel.clicked.connect(self._request_cancel)
        progress_row = QWidget()
        progress_layout = QHBoxLayout(progress_row)
        progress_layout.setContentsMargins(0, 0, 0, 0)
        progress_layout.addWidget(self.progress_pane, 1)
        progress_layout.addWidget(self.btn_cancel)

        # Stored as attributes so a subclass can insert its own primary view above the figure stack
        # (e.g. SimulatePanel mounts a live pyqtgraph view here and hides the static figure stack).
        self.right = QWidget()
        self.right_layout = QVBoxLayout(self.right)
        self.right_layout.setContentsMargins(0, 0, 0, 0)
        self.right_layout.addWidget(self.figure_stack, 3)
        self.right_layout.addWidget(progress_row)
        self.right_layout.addWidget(self.log_pane, 1)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(controls_scroll)
        splitter.addWidget(self.right)
        splitter.setStretchFactor(1, 1)

        outer = QHBoxLayout(self)
        outer.setContentsMargins(6, 6, 6, 6)
        outer.addWidget(splitter)

    # ── background dispatch ──────────────────────────────────────────────────
    def dispatch(self, fn, *args, provide_fig_sink: bool = False, provide_stream: bool = False,
                 on_chunk=None, watch_dir=None, on_result=None, on_finished=None, **kwargs):
        """Run ``fn`` on a worker thread. Its print()s and warnings stream to the log pane and its tqdm
        bars to the progress pane; figures (when ``provide_fig_sink``) embed in the figure stack; the
        return value goes to ``on_result``.

        ONE TASK AT A TIME APP-WIDE, not merely per panel -- redirect_streams swaps sys.stdout/stderr
        process-wide, so two concurrent runs would fight over the console (see GOTCHA #4).

        ``watch_dir`` is for the FDT / Reduction / CrossVal runners, which save their figures to disk
        instead of handing them back: any PNG appearing there during the run is picked up and shown
        (see core/gui/plot_watcher.py).
        """
        if BasePanel._running:
            where = "in this tab" if self._busy else "in another tab"
            self.log_pane.append_line(
                f"A task is already running ({where}); please wait for it to finish.", "warning")
            return

        watcher = None
        if watch_dir is not None:
            watcher = NewPngWatcher(watch_dir, self)
            watcher.png_ready.connect(self.figure_stack.add_png)
            watcher.start()

        self._cancel = CancelToken()
        BasePanel._active_cancel = self._cancel
        worker = Worker(fn, *args, cancel=self._cancel, **kwargs)
        # Retain the worker (and thus its WorkerSignals sender) until it reports finished, and stop the
        # thread pool from auto-deleting the C++ QRunnable underneath it -- otherwise the sender is
        # destroyed as soon as run() returns and Qt discards its still-queued result/finished events.
        worker.setAutoDelete(False)
        self._workers.add(worker)
        if provide_fig_sink:
            worker.kwargs["fig_sink"] = _png_fig_sink(worker.signals.figure)
        if provide_stream:
            # A long-lived streaming runner (e.g. SimulatePanel) emits numpy frames through the `chunk`
            # signal and polls should_stop() to unwind cooperatively -- the same injection trick as
            # provide_fig_sink, but for a continuous stream rather than one-shot figures. should_stop is
            # the cancel token's flag: the runner raises WorkerCancelled between frames when it flips.
            worker.kwargs["emit_chunk"] = worker.signals.chunk.emit
            worker.kwargs["should_stop"] = self._cancel.requested.is_set
            if on_chunk is not None:
                worker.signals.chunk.connect(on_chunk)
        worker.signals.log.connect(self.log_pane.append_line)
        worker.signals.log_batch.connect(self.log_pane.append_lines)
        worker.signals.log_batch.connect(lambda _b: self.progress_pane.heartbeat())
        worker.signals.rows.connect(self.progress_pane.set_rows)
        worker.signals.figure.connect(self.figure_stack.add_figure)
        worker.signals.error.connect(self._on_error)
        worker.signals.cancelled.connect(lambda: self.log_pane.append_line("Run cancelled.", "warning"))
        if on_result is not None:
            worker.signals.result.connect(on_result)

        def _finished():
            if watcher is not None:
                watcher.stop()          # one last scan: the final figure often lands right at the end
                watcher.deleteLater()
            self._set_busy(False)
            self._workers.discard(worker)
            # Release the PAYLOAD explicitly. discard() is not a lifetime bound: QThreadPool.start()
            # hands the QRunnable to C++, and setAutoDelete(False) (above) means C++ never frees it, so
            # the Worker shell outlives the run -- as does the _finished closure, which captures it and
            # is never disconnected. Both are tiny; what is NOT tiny is what the Worker still points at
            # (cfg, prior, posterior, CUDA tensors). Without this, every posterior you build stays
            # pinned for the life of the process even after SbiSession.reset_downstream drops it.
            worker.fn = None
            worker.args = ()
            worker.kwargs = {}
            self._cancel = None
            BasePanel._active_cancel = None
            if on_finished is not None:
                on_finished()

        worker.signals.finished.connect(_finished)
        self._set_busy(True)
        QThreadPool.globalInstance().start(worker)

    def _request_cancel(self):
        if self._cancel is None:
            return
        self._cancel.requested.set()
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.setText("Cancelling…")
        # A cancel is cooperative: it lands at the next print/tqdm redraw. That is ~1s almost
        # everywhere, but a neural-network training epoch and the SBC C2ST block are silent for longer.
        self.log_pane.append_line(
            "Cancelling — will stop at the next checkpoint (up to ~1 min during training).", "warning")

    def _set_busy(self, busy: bool):
        self._busy = busy
        BasePanel._running = busy
        self.btn_cancel.setVisible(busy)
        self.btn_cancel.setEnabled(busy)
        self.btn_cancel.setText("Cancel")
        if busy:
            self.progress_pane.begin()
        else:
            self.progress_pane.end()   # authoritative: drops any row a crashed worker left behind
        # Lock EVERY panel's controls while a run is live, not just this one: redirect_streams swaps
        # sys.stdout/stderr process-wide, so another panel's ArtifactPicker refresh (which wraps
        # list_dir in redirect_stdout) or a model combo could swallow / corrupt the running worker's
        # stream -- the hazard set_controls_enabled documents, now spread across sibling inference tabs.
        # Only this panel keeps its Cancel button live (it lives outside `controls`).
        for panel in list(BasePanel._instances):
            panel.set_controls_enabled(not busy)

    def set_controls_enabled(self, enabled: bool):
        """Lock the whole left-hand column while a task runs.

        The WHOLE column, not just the run button: ArtifactPicker.refresh() (its ⟳ button, and the
        model combos that call it) wraps file_manager.list_dir in contextlib.redirect_stdout, which
        reassigns the PROCESS-WIDE sys.stdout -- i.e. the very stream redirect_streams installed for
        the running worker. Leaving a picker live mid-run lets a click swallow the worker's output,
        and if the worker's teardown restores sys.stdout inside that window, redirect_stdout.__exit__
        then reinstates the dead _SignalStream as the process's stdout permanently.
        """
        self.controls.setEnabled(enabled)
        if enabled:
            self.refresh_local_gates()   # re-apply this panel's own widget gating after a run frees it

    def refresh_local_gates(self) -> None:
        """Re-apply widget-level gating within this panel (which buttons/options are enabled). Base is a
        no-op; the inference sub-panels override it, and an owning screen may call it after a stage
        completes. Distinct from the tab-level greying an InferenceScreen does via setTabEnabled."""

    # ── persistence (subclasses override; keys are namespaced under group() by MainWindow) ──────────
    def save_settings(self, qs) -> None:
        """Persist this panel's user selections. Base is a no-op; subclasses override."""

    def restore_settings(self, qs) -> None:
        """Restore what save_settings wrote. Called at the END of a subclass __init__, after signals
        are connected -- restore order matters (a picker restored before its model gets wiped by the
        model's refresh())."""

    def _config_error(self, exc: Exception):
        """Report a failed config build as user-input trouble, not a crash.

        Deliberately catches broadly at the call sites: cli's builders raise a bare ValueError (NOT
        UnitParseError) for the two most plausible user mistakes -- a cell with no sibling bounds file
        (core/cli.py:331) and a cell missing a param the bounds file requires (core/cli.py:268). A
        narrow `except cli.UnitParseError` lets those escape the clicked slot and surface as a raw
        traceback in app.py's last-resort excepthook, with nothing in the panel's own log.
        """
        msg = str(exc)
        self.log_pane.append_line(f"Could not build the config: {msg}", "error")
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Warning)                  # user input, not a crash
        box.setWindowTitle("Check your inputs")
        box.setText("The configuration could not be built.")
        box.setInformativeText(msg)
        box.exec()

    def _on_error(self, message: str, tb: str):
        """Show a run failure. The traceback goes in a collapsible Details panel, not pasted whole into
        the body (which produced an unscrollable, un-copyable wall of text stretched to the widest stack
        frame)."""
        self.log_pane.append_line(message, "error")
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Critical)
        box.setWindowTitle("Error")
        box.setText(message)
        if tb:
            box.setDetailedText(tb)
        box.exec()
