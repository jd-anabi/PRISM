"""Shared panel scaffolding: a left controls column and a right results column (a figure stack over the page's own
lines, or, while a run of the page shows, the run's page in their place). What makes a panel the host of a run,
``dispatch()`` among it, is ``RunHost``'s, in ``run_host.py``."""
import warnings

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QHBoxLayout, QScrollArea, QSplitter, QStackedWidget, QVBoxLayout, QWidget
from PySide6.QtCore import Qt

from .. import design
from ..design import CONTROLS_MIN_W, DEFAULT_RESULTS_SPLIT, DEFAULT_SPLIT
from ..reported import reported
from ..widgets.figure_stack import FigureStack
from ..widgets.log_pane import LogPane
from .run_host import CANCEL_NOTED_TEXT, RunHost                  # noqa: F401 -- the words are said from here too

# Said, as one warning of the run, when a figure the run handed over could not be shown, filled with the figure's
# title and the error's kind and message.
FIGURE_NOT_SHOWN_TEXT = "The figure '{}' could not be shown ({}). The record is not affected."
# How narrow the form may stand while a run's page, and not the figures, shares the width with it: the run's page keeps
# its own least width, and the form takes what is left. Nought, so that this number has nothing to keep in step with.
FORM_LEAST_BESIDE_A_RUN = 0


class FigureNotShown(UserWarning):
    """A figure a run handed over that the window could not show. Said every time, never dropped as a repeat of
    one said before: ``_said_every_time`` puts the process's filter for this kind alone in force."""


def _said_every_time() -> None:
    """Put in force the filter that says every ``FigureNotShown``, when it is not in force already: at each sink made,
    not once at import, because what puts the warnings filters back as they were (a test runner does, each time it
    has collected or run a test) also takes away a filter added in between."""
    if not any(entry[0] == "always" and entry[2] is FigureNotShown for entry in warnings.filters):
        warnings.filterwarnings("always", category=FigureNotShown)


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

    The store calls this sink inside the writer of a record it has saved the figure into, and that writer
    removes its record's folder on any exception. So NOTHING the sink does may raise into it: a render that
    fails, an emit that fails, a signal whose source is gone (the window closed while the run went on) each
    become one warning of the run, itself guarded, and the record is written as if the figure had been shown.
    A stop (``WorkerCancelled``, which is no ``Exception``) still passes, as it passes every other point
    where a stop is taken. The figure is closed whatever happens.
    """
    _said_every_time()

    def _sink(title, fig):
        try:
            import io
            import pickle
            try:
                fig_pickle = pickle.dumps(fig)      # before savefig -> the pristine figure the stage built
            except Exception:                       # noqa: BLE001 -- pop-out is optional; keep the run alive
                fig_pickle = None
            buf = io.BytesIO()
            # save_figure, not fig.savefig: it saves the page in the colours the figure was BUILT
            # with, whatever matplotlib's settings are at the moment of saving. The window writes its
            # figure settings once, when the application is built, so the two agree here; the guard
            # stays for anything that changes them between build and save. See visualizers.save_figure.
            from core.Helpers.visualizers import save_figure
            save_figure(fig, buf, format="png", dpi=110, bbox_inches="tight")
            figure_signal.emit(title, buf.getvalue(), fig_pickle)
        except Exception as error:                  # noqa: BLE001 -- the figure, never the record
            _not_shown(title, error)
        finally:
            _close(fig)
    return _sink


def _not_shown(title, error) -> None:
    """Say, as one warning of the run, that the figure ``title`` could not be shown and why. Never raises: a warning
    the filters turn into an error, or an error whose message cannot be had, is dropped or said by its kind."""
    try:
        try:
            why = f"{type(error).__name__}: {error}"
        except Exception:                           # noqa: BLE001 -- said by its kind alone
            why = type(error).__name__
        warnings.warn(FIGURE_NOT_SHOWN_TEXT.format(title, why), FigureNotShown, stacklevel=2)
    except Exception:                               # noqa: BLE001 -- a warning that cannot be said is dropped
        pass


def _close(fig) -> None:
    """Close ``fig``, letting its memory go. Never raises."""
    try:
        import matplotlib.pyplot as plt
        plt.close(fig)
    except Exception:                               # noqa: BLE001 -- a figure that cannot be closed stays open
        pass


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


class BasePanel(RunHost, QWidget):
    """Shared scaffolding for a page that has not been rebuilt yet: a controls column and a results column, a run
    hosted by ``RunHost``. The Artifacts screen is NOT one -- it is a plain QWidget on purpose, because a run host
    enrols in the set of pages every run locks, and every run anywhere would grey it out.

    Persists: nothing here. Subclasses override ``save_settings``/``restore_settings`` for their own
    selections; the splitter geometry goes through the separate ``save_layout``/``restore_layout``
    pair, because most subclasses override save_settings without calling super().
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._enrol()

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
        # figures under its messages; a third, the past view, joins them the first time a run that is over is
        # drawn here (``past_view``). The form stays at the left, locked while a run is live.
        self.run_view = self._new_run_page()
        self.run_view.setObjectName("run")
        self.run_view.back.connect(self._back_to_the_form)
        self.run_view.how_link.clicked.connect(self.show_run)
        self.figures = QWidget()
        self.figures.setObjectName("figures")
        self.figures_layout = QVBoxLayout(self.figures)
        self.figures_layout.setContentsMargins(0, 0, 0, 0)
        # the link over the figures as a heading over its block
        self.figures_layout.setSpacing(design.scaled(design.GAP["head_table"]))
        self.figures_layout.addWidget(self.run_view.how_link, 0, Qt.AlignLeft)
        self.figures_layout.addWidget(self.results_split, 1)
        self.results = QStackedWidget()
        self.results.addWidget(self.figures)
        self.results.addWidget(self.run_view)
        self.results.currentChanged.connect(self._fit_the_form)
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

    def _fit_the_form(self, _index: int = 0) -> None:
        """The form gives up its least width while a run's page, its own or a past one, stands beside it: a splitter
        cannot give two neighbours less than their least widths add up to, and what it cannot give, the second is
        drawn under the first. So the page, which holds a run, keeps its least width, and the form scrolls sideways
        in what is left. With the page gone the form has its least width again, and a divider left by hand stands
        where it stood, for the splitter keeps it. Heard at every change of what the column shows; a panel that stands
        the run's page anywhere else calls this as the page comes and goes."""
        self.controls_scroll.setMinimumWidth(FORM_LEAST_BESIDE_A_RUN if self._run_page_beside() else CONTROLS_MIN_W)

    def _run_page_beside(self) -> bool:
        """Whether a run's page stands beside the form: here, whether the column shows anything but the figures. A
        panel that stands the run's page anywhere else says so."""
        return self.results.currentWidget() is not self.figures

    def own_line(self, text: str, level: str = "info") -> None:
        """A line of the panel's own goes among its lines, under the figures."""
        self.log_pane.append_line(text, level)

    def _figures_leaving(self) -> None:
        """The heights of the figures and the lines are kept, while the figures are in their place, for their return."""
        if self.results_split.indexOf(self.figure_stack) >= 0:
            self._split_sizes = self.results_split.sizes()

    def _figures_back(self, stack) -> None:
        """The figures back above the lines, at the heights they had."""
        self.results_split.insertWidget(self.results_split.count() - 1, stack)
        if self._split_sizes is not None:
            self.results_split.setSizes(self._split_sizes)

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
    #   1. save_settings is overridden by most panels WITHOUT calling super(), so anything
    #      hooked there would silently never run for them. This pair is driven from BasePanel itself
    #      (restore) and MainWindow._save_state (save), so a subclass cannot break it by forgetting.
    #   2. Layout is not a user *selection*. A panel that resets its pickers should not lose the
    #      column widths the user set.
    def layout_key(self) -> str:
        """Settings key for this panel's layout. Distinct per panel: each has its own splitters.
        Subclasses that share a class name would override this; today none does."""
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
