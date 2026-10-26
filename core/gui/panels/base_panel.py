"""Shared panel scaffolding: a left controls column and a right results column (a figure stack over the page's own
lines, or, while a run of the page shows, the run's page in their place), plus ``dispatch()``, which asks the
process's run controller to run a callable on a background worker, described by the panel's own ``run_spec``."""
import weakref

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (QApplication, QHBoxLayout, QMessageBox, QScrollArea, QSplitter, QStackedWidget,
                               QVBoxLayout, QWidget)
from PySide6.QtCore import Qt

from core.refusals import Refusal

from .. import fields as gui_fields
from ..design import CONTROLS_MIN_W, DEFAULT_RESULTS_SPLIT, DEFAULT_SPLIT
from ..reported import reported
from ..run.controller import LIVE_TEXT, controller
from ..run.view import RunView
from ..widgets.figure_stack import FigureStack
from ..widgets.log_pane import LogPane
from ..widgets.refusal_box import show_refusal

# "The cancel noted": shown alongside an ordinary crash report when the cancel token had already
# been REQUESTED at the moment the failure was caught. It never claims the cancel caused the
# failure (the two are independent: the run crashed, and separately someone had clicked Cancel) --
# only that both are true, which the person who clicked Cancel is owed.
CANCEL_NOTED_TEXT = "A cancel was requested before the run failed."


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
            # save_figure, not fig.savefig: it saves the page in the colours the figure was BUILT
            # with, whatever matplotlib's settings are at the moment of saving. The window writes its
            # figure settings once, when the application is built, so the two agree here; the guard
            # stays for anything that changes them between build and save. See visualizers.save_figure.
            from core.Helpers.visualizers import save_figure
            save_figure(fig, buf, format="png", dpi=110, bbox_inches="tight")
        finally:
            plt.close(fig)
        figure_signal.emit(title, buf.getvalue(), fig_pickle)
    return _sink


class _Column(QVBoxLayout):
    """The column of a panel's controls, laid from its top.

    A layout held to its top is laid out no taller than its size hint, and the hint of a line that wraps is
    worked out for a width other than the column's: a column holding such a line was given less height than
    its words take at its width, and its lines were drawn over one another. The hint asked for here is never
    less than the column's height at its narrowest, the most its lines can take, so the column is laid at its
    height for the width it has. A column holding something that grows is laid over all its room, as before.
    """

    @reported(QVBoxLayout.sizeHint)
    def sizeHint(self):
        hint = super().sizeHint()
        if self.hasHeightForWidth():
            hint.setHeight(max(hint.height(), self.heightForWidth(self.minimumSize().width())))
        return hint


class BasePanel(QWidget):
    """Shared scaffolding for every run-a-thing panel: a controls column, a results column, and
    ``dispatch()`` to run work on a background thread with its output wired to both.

    Ten of these exist (Reduction, FDT, CrossVal, Simulate + the six inference tabs: Config, Prior,
    Posterior, Validate, Infer, TSNPE). The Artifacts screen is NOT one -- it is a plain QWidget on
    purpose, because a BasePanel enrols in _instances and every run anywhere would grey it out.
    ``_instances`` is class-level ON PURPOSE (see its comment): the run's controller locks every panel
    in it while a run is live. What runs, and whether a run is live, is the controller's to say
    (``core/gui/run/controller.py``): one run goes at a time in the whole process, because stream
    redirection is process-wide. Each panel describes the runs it starts in ``run_spec``.

    Persists: nothing here. Subclasses override ``save_settings``/``restore_settings`` for their own
    selections; the splitter geometry goes through the separate ``save_layout``/``restore_layout``
    pair, because 9 of the 10 subclasses override save_settings without calling super().
    """
    # Every live panel, so a run in ANY panel can lock the controls of ALL of them: the run's controller walks
    # this set at a run's start and at its end. Class-level, deliberately: one run goes at a time in the whole
    # process. Weak, and a panel leaves it as its object goes, though Python may hold the panel longer (see
    # __init__).
    _instances: "weakref.WeakSet" = weakref.WeakSet()

    def __init__(self, parent=None):
        super().__init__(parent)
        BasePanel._instances.add(self)
        # It leaves the set as its object goes. A panel that ran something is still held in Python once it is
        # deleted, by its run's own signals, and the next run locks every panel of the set: handed an object
        # that is gone, that lock would raise, and no run could start again.
        held, enrolled = weakref.ref(self), BasePanel._instances
        self.destroyed.connect(lambda *_: enrolled.discard(held()) if held() is not None else None)

        # Left: subclasses fill self.controls (inside a scroll area so long forms stay usable).
        self.controls = QWidget()
        self.controls_layout = _Column(self.controls)
        self.controls_layout.setAlignment(Qt.AlignTop)
        # Kept on self so save_layout/restore_layout can reach them (both used to be locals, which is
        # why nothing could persist or even inspect the splitter).
        self.controls_scroll = QScrollArea()
        self.controls_scroll.setWidgetResizable(True)
        self.controls_scroll.setWidget(self.controls)
        self.controls_scroll.setMinimumWidth(CONTROLS_MIN_W)
        # No maximum. The old hard 460px cap could not be escaped by widening the window, so ANY form
        # wider than that got a permanent horizontal scrollbar in the left column -- one of the two
        # halves of the "input boxes are cut off" report. Width is now governed by the splitter, which
        # the user can drag and which now remembers where they put it.

        # Right: the page's figures over its own lines (a refusal's sentence, "Model applied", a record
        # written). A run's lines never come here: they are the run's full text, on the run's page.
        self.figure_stack = FigureStack()
        self.log_pane = LogPane()

        # Stored as attributes so a subclass can insert its own primary view above the figure stack
        # (e.g. SimulatePanel mounts a live pyqtgraph view here and hides the static figure stack).
        # The figures and the lines are a VERTICAL SPLITTER, not a fixed 3:1 stretch. The log pane used to
        # hold ~25% of the height whether it had one line in it or a thousand, with no handle to drag
        # -- that is the other half of "panes have to be dragged to show everything": here there was
        # nothing to drag at all.
        self.results_split = QSplitter(Qt.Vertical)
        self.results_split.addWidget(self.figure_stack)
        self.results_split.addWidget(self.log_pane)
        self.results_split.setStretchFactor(0, 3)
        self.results_split.setStretchFactor(1, 1)
        self.results_split.setChildrenCollapsible(False)
        self.results_split.setSizes(list(DEFAULT_RESULTS_SPLIT))

        # The right-hand column is a stack of two: the figures, with "How the run went" above them once a run
        # of the page has ended, and the run's page, which takes their place while it shows and holds the
        # figures under its messages. The form stays at the left, locked while a run is live.
        self.run_view = RunView()
        self.run_view.setObjectName("run")
        self.run_view.back.connect(self._back_to_the_form)
        self.run_view.how_link.clicked.connect(self.show_run)
        self.figures = QWidget()
        self.figures.setObjectName("figures")
        self.figures_layout = QVBoxLayout(self.figures)
        self.figures_layout.setContentsMargins(0, 0, 0, 0)
        self.figures_layout.addWidget(self.run_view.how_link, 0, Qt.AlignLeft)
        self.figures_layout.addWidget(self.results_split, 1)
        self.results = QStackedWidget()
        self.results.addWidget(self.figures)
        self.results.addWidget(self.run_view)
        self._split_sizes = None             # the figures' and the lines' heights, kept while the run's page holds them

        self.splitter = QSplitter(Qt.Horizontal)
        self.splitter.addWidget(self.controls_scroll)
        self.splitter.addWidget(self.results)
        self.splitter.setStretchFactor(1, 1)
        # A drag past the left edge used to collapse the controls column to zero width, recoverable
        # only by finding a 5px handle at x=0.
        self.splitter.setChildrenCollapsible(False)
        self.splitter.setSizes(list(DEFAULT_SPLIT))

        outer = QHBoxLayout(self)
        outer.setContentsMargins(6, 6, 6, 6)
        outer.addWidget(self.splitter)

        # Persist the split as the user drags it, not only on a clean quit: save_settings runs from
        # MainWindow.closeEvent alone, so a crash -- or answering "don't quit" -- lost everything, and
        # "I have to re-drag it every launch" is the actual complaint.
        self._layout_save_timer = QTimer(self)
        self._layout_save_timer.setSingleShot(True)
        self._layout_save_timer.setInterval(1500)
        self._layout_save_timer.timeout.connect(self._persist_layout)
        self.splitter.splitterMoved.connect(lambda *_: self._layout_save_timer.start())
        self.results_split.splitterMoved.connect(lambda *_: self._layout_save_timer.start())

        self.restore_layout()

    # ── background dispatch ──────────────────────────────────────────────────
    def dispatch(self, fn, *args, provide_fig_sink: bool = False, provide_stream: bool = False,
                 on_chunk=None, watch_dir=None, on_result=None, **kwargs) -> bool:
        """Ask the process's run controller to run ``fn(*args, **kwargs)`` on a worker thread, described by this
        panel's ``run_spec`` for it, this panel its host: figures (when ``provide_fig_sink``) and the pictures that
        land in ``watch_dir`` (one folder, or several) go to the figure stack, and the return value to
        ``on_result``. The run's lines go to its text, which the controller keeps.

        ONE RUN AT A TIME IN THE WHOLE PROCESS, not merely per panel: redirect_streams swaps sys.stdout/stderr
        process-wide, so two concurrent runs would fight over the console. While a run is live this says so
        among the page's own lines, starts nothing and returns False; True when the run was started.

        A run started from this page's form leaves the keyboard on the run page's stop button, where the
        form's lock would leave it on nothing. Given for no key of its own, the focus shows its ring only
        while the keyboard has the turn: after a start made with a key."""
        held = controller()
        if held.is_live():
            self.log_pane.append_line(LIVE_TEXT, "warning")
            return False
        focus = QApplication.focusWidget()
        from_here = focus is not None and self.isAncestorOf(focus)
        started = held.dispatch(fn, *args, spec=self.run_spec(fn, *args, **kwargs), host=self, on_result=on_result,
                                provide_fig_sink=provide_fig_sink, provide_stream=provide_stream, on_chunk=on_chunk,
                                watch_dir=watch_dir, **kwargs)
        stop = self.run_view.stop_button
        if started and from_here and stop.isVisibleTo(self) and stop.isEnabled():
            stop.setFocus(Qt.OtherFocusReason)
        return started

    def run_spec(self, fn, *args, **kwargs):
        """What this panel knows of the run ``fn(*args, **kwargs)`` would be, and the run does not say: its word,
        its place, its stop button, its stages, the records it writes, whether it keeps files. Each panel that starts
        runs answers for its own; a call this panel does not start fails loudly here."""
        raise NotImplementedError(f"{type(self).__name__} has no run spec for {fn!r}")

    def is_running(self) -> bool:
        """Whether the live run is this panel's own."""
        held = controller()
        return held.is_live() and held.host() is self

    def show_run(self) -> None:
        """The run's page in the right-hand column, holding this page's figures under its messages: while this
        page's run is live, after a stop or a failure until "Back to the form", and again from "How the run
        went"."""
        if self.results_split.indexOf(self.figure_stack) >= 0:
            self._split_sizes = self.results_split.sizes()
        self.run_view.hold_figures(self.figure_stack)
        self.run_view.how_link.setVisible(False)
        self.results.setCurrentWidget(self.run_view)

    def show_figures(self) -> None:
        """The right-hand column back to this page's figures over its own lines, the figures the run's page held
        handed back to their place, at the heights they had."""
        stack = self.run_view.release_figures()
        if stack is not None:
            self.results_split.insertWidget(self.results_split.count() - 1, stack)
            if self._split_sizes is not None:
                self.results_split.setSizes(self._split_sizes)
        self.results.setCurrentWidget(self.figures)

    def _run_began(self, spec) -> None:
        """This page's run has begun: its page takes the right-hand column, unless the run only loads a record.
        Called by the run's controller, once the run's page has been handed the run."""
        if not spec.loads_only:
            self.show_run()

    def _run_over(self, status) -> None:
        """This page's run has ended, as ``status`` says. A run that finished gives the column back to the
        figures, "How the run went" above them; one that ended into idle (a refusal before anything was spent, a
        run that keeps no file) gives it back with no link; a stopped or a failed run keeps its page until "Back
        to the form". Called by the run's controller, once every panel is free again."""
        if status.state == "idle":
            self.show_figures()
        elif status.outcome == "finished":
            self.show_figures()
            self.run_view.how_link.setVisible(True)

    def _back_to_the_form(self) -> None:
        """"Back to the form": the figures again, "How the run went" above them."""
        self.show_figures()
        self.run_view.how_link.setVisible(True)

    def set_controls_enabled(self, enabled: bool):
        """Lock the whole left-hand column while a task runs.

        The WHOLE column, not just the run button: a model combo changed mid-run repoints the cell
        picker and re-seeds the panel's inputs (simulate_panel / fdt_panel / config_tab
        ._on_model_changed) while the worker is still reporting against the selection it started
        from. The stdout hazard this used to name is gone: file_manager.list_dir returns its listing
        and prints nothing, so ArtifactPicker.refresh() no longer swaps the process-wide sys.stdout
        under redirect_streams.
        """
        self.controls.setEnabled(enabled)
        if enabled:
            self.refresh_local_gates()   # re-apply this panel's own widget gating after a run frees it

    def refresh_local_gates(self) -> None:
        """Re-apply widget-level gating within this panel (which buttons/options are enabled). Base is a
        no-op; the inference sub-panels override it, and an owning screen may call it after a stage
        completes. It is the only greying a step of parameter inference has: no step is disabled as a whole."""

    # ── persistence (subclasses override; keys are namespaced under group() by MainWindow) ──────────
    def insert_result_widget(self, index: int, widget, stretch: int = 1) -> None:
        """Mount a panel-specific primary view in the results column, above the figure stack.

        SimulatePanel puts a live pyqtgraph view here and hides the static figure stack. The figures
        and the lines are a QSplitter, so a view inserted beside them by hand would land in the wrong
        parent; this is the supported seam. Going through it also means the live view gets a drag
        handle like everything else in that column.
        """
        self.results_split.insertWidget(index, widget)
        self.results_split.setStretchFactor(index, stretch)

    # ── layout persistence ───────────────────────────────────────────────────
    # DELIBERATELY SEPARATE from save_settings/restore_settings. Two reasons, and the first is fatal:
    #   1. save_settings is overridden by 9 of the 10 panels WITHOUT calling super(), so anything
    #      hooked there would silently never run for them. This pair is driven from BasePanel itself
    #      (restore) and MainWindow._save_state (save), so a subclass cannot break it by forgetting.
    #   2. Layout is not a user *selection*. A panel that resets its pickers should not lose the
    #      column widths the user set.
    def layout_key(self) -> str:
        """Settings key for this panel's layout. Distinct per panel -- there are ten independent
        splitters. Subclasses that share a class name would override this; today all ten differ."""
        return type(self).__name__

    def save_layout(self, qs) -> None:
        """Persist the splitter geometry. Called by MainWindow._save_state and the debounce timer."""
        qs.setValue(f"layout/{self.layout_key()}/splitter", self.splitter.saveState())
        if self.results_split.indexOf(self.figure_stack) >= 0:     # not while the run's page holds the figures
            qs.setValue(f"layout/{self.layout_key()}/results", self.results_split.saveState())

    def restore_layout(self, qs=None) -> None:
        """Restore the splitter geometry, if any was stored.

        saveState/restoreState rather than sizes(): restoreState works before the widget is shown or
        polished, which setSizes does not do reliably. The return value MUST be checked -- it is
        False when the stored state does not match the current child count, and silently ignoring
        that is the same class of bug as an unchecked restoreGeometry restoring a window off-screen.
        """
        from .. import settings as _settings
        qs = qs or _settings.settings()
        blob = qs.value(f"layout/{self.layout_key()}/splitter")
        if blob is not None and not self.splitter.restoreState(blob):
            self.splitter.setSizes(list(DEFAULT_SPLIT))     # stale/incompatible -> sane default
        # The results split has a variable child count (SimulatePanel inserts a live view), so
        # restoreState legitimately fails after such a change -- fall back rather than ignore.
        rblob = qs.value(f"layout/{self.layout_key()}/results")
        if rblob is not None and not self.results_split.restoreState(rblob):
            self.results_split.setSizes(list(DEFAULT_RESULTS_SPLIT))

    def _persist_layout(self) -> None:
        """Debounced write from splitterMoved. Touches ONLY this panel's layout key."""
        from .. import settings as _settings
        qs = _settings.settings()
        self.save_layout(qs)
        qs.sync()

    def save_settings(self, qs) -> None:
        """Persist this panel's user selections. Base is a no-op; subclasses override."""

    def restore_settings(self, qs) -> None:
        """Restore what save_settings wrote. Called at the END of a subclass __init__, after signals
        are connected -- restore order matters (a picker restored before its model gets wiped by the
        model's refresh())."""

    def _refusal(self, exc: Refusal) -> None:
        """Show a refusal: the yellow "Check your inputs" box, and no traceback.

        A refusal is something the program will not do with what it was given -- a blank box, a taken
        name, a training run one setting away from its cache -- and the fix is on this screen, so the
        box names it. The text is the core's neutral sentence (it names the setting, the rule, what
        was given and the default; never a box, tab or flag) and the informative line is this front
        end's own "where to fix it", looked up by the refusal's field key in core/gui/fields.py --
        the one place a control is named, so a renamed control is renamed once. No Details: nothing
        here is a crash, and a traceback would only say so louder. A bug keeps the red box
        (_on_error). The same sentence goes to the log pane at warning, so it outlives the click that
        dismisses the box.

        THE BOX ITSELF lives in ../widgets/refusal_box.py, shared with the Artifacts screen and the
        model builder, which show the same box and are deliberately not BasePanels. What stays here
        is the log-pane line, because only a panel has a pane -- those two screens record the
        sentence on their own status lines instead.
        """
        fix = gui_fields.fix_sentence(exc.field)
        self.log_pane.append_line(f"{exc.message} {fix}".rstrip(), "warning")
        show_refusal(self, exc)

    def _on_error(self, exc_or_message, tb: str, cancel_noted: bool = False, kept_fit=None) -> None:
        """Show a failure. A Refusal goes to the yellow box (_refusal); anything else is a bug and
        gets the red one, with the traceback in a collapsible Details panel rather than pasted whole
        into the body (which produced an unscrollable, un-copyable wall of text stretched to the
        widest stack frame). Called by the run's controller once a run of this panel that failed has
        ended, with the exception object WorkerSignals.error carried and what came with it;
        a click handler that catches on the GUI thread passes what it caught the same way (omitting
        ``cancel_noted``, which defaults to False -- a GUI-thread click has no worker cancel token to
        ask). A plain string is still accepted and is still a bug.

        ``cancel_noted``: true when the cancel token had already been REQUESTED at the moment
        ``Worker.run`` caught this failure -- a coincidence, never a cause (the crash is reported as
        a crash regardless; see core/gui/worker.py). Shown as an extra line in both places a crash
        is reported, appended AFTER the crash itself: the box's informative text (below the one-line
        summary, above Details) and the pane, at warning, after the error line. Never shown for a
        Refusal -- that box is about fixing an input, not about a run.

        ``kept_fit``: the training a failure in the fit kept, as the worker hands it on (the training
        stage's note, or None). This box does not show it.
        """
        if isinstance(exc_or_message, Refusal):
            return self._refusal(exc_or_message)
        message = str(exc_or_message)
        self.log_pane.append_line(message, "error")
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Critical)
        box.setWindowTitle("Error")
        box.setText(message)
        if tb:
            box.setDetailedText(tb)
        if cancel_noted:
            box.setInformativeText(CANCEL_NOTED_TEXT)
            self.log_pane.append_line(CANCEL_NOTED_TEXT, "warning")
        box.exec()
