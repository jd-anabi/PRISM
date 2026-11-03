"""What makes a page the host of a run: ``dispatch``, which asks the process's run controller to run a callable on a
background worker, described by the page's own ``run_spec``; the set of pages every run locks; and the moves of the run's
page in and out of the page's column, its figures with it.

``RunHost`` is a mixin, and stands first in a page's bases, before its toolkit class. It declares no signal and has no
``__init__``: a page calls ``_enrol`` right after its toolkit class is built. A page holds ``results``, a stack that
shows one thing at a time; ``figures``, what that stack shows when no run does; ``run_view``, the run's page, in that
stack, whose ``back`` it hands to ``_back_to_the_form`` and whose ``how_link`` to ``show_run``; ``figure_stack``; and
``controls``, what a run locks. It answers ``refresh_local_gates``, ``own_line``, ``_figures_leaving`` and
``_figures_back``."""
import weakref

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QMessageBox

from core.refusals import Refusal

from .. import fields as gui_fields
from ..run.controller import LIVE_TEXT, controller
from ..run.retry import NOTED_TEXT
from ..run.view import RunView
from ..widgets.refusal_box import show_refusal

# "The stop noted": shown alongside an ordinary crash report when the run's stop had already been
# ASKED FOR at the moment the failure was caught. It never claims the stop caused the failure (the two
# are independent: the run crashed, and separately someone had asked it to stop) -- only that both are
# true, which the person who asked for the stop is owed. The run's page says the same words; this box
# says them for the one run whose failure still reaches it, a load's.
CANCEL_NOTED_TEXT = NOTED_TEXT


class RunHost:
    """The run-hosting half of a page, shared by every page that starts runs. See the module's own words for what a
    page must hold. What runs, and whether a run is live, is the controller's to say (``core/gui/run/controller.py``):
    one run goes at a time in the whole process, because stream redirection is process-wide. Each page describes the
    runs it starts in ``run_spec``."""
    # Every live page, so a run in ANY page can lock the controls of ALL of them: the run's controller walks this set
    # at a run's start and at its end. Class-level, deliberately: one run goes at a time in the whole process. Weak,
    # and a page leaves it as its object goes, though Python may hold the page longer (see ``_enrol``).
    _instances: "weakref.WeakSet" = weakref.WeakSet()
    _past_view: RunView | None = None        # built the first time a run that is over is drawn here
    _before_past = None                      # what the column showed before the past view took it

    def _enrol(self) -> None:
        """Join the set of pages every run locks. A page calls this right after its toolkit class is built: it needs
        the object's ``destroyed``."""
        RunHost._instances.add(self)
        # It leaves the set as its object goes. A panel that ran something is still held in Python once it is
        # deleted, by its run's own signals, and the next run locks every panel of the set: handed an object
        # that is gone, that lock would raise, and no run could start again.
        held, enrolled = weakref.ref(self), RunHost._instances
        self.destroyed.connect(lambda *_: enrolled.discard(held()) if held() is not None else None)

    def _new_run_page(self) -> RunView:
        """A run's page for this page's stack, the run's own and the past view alike: made here, so that what a page does
        to one it does to both."""
        return RunView()

    def own_line(self, text: str, level: str = "info") -> None:
        """Say a line of the page's own, outside a run: where a page says one is the page's to answer."""
        raise NotImplementedError(f"{type(self).__name__} has no place for a line of its own")

    def _figures_leaving(self) -> None:
        """The run's page is about to take the figures: the page keeps what it must to put them back."""
        raise NotImplementedError(f"{type(self).__name__} does not say what it keeps of its figures")

    def _figures_back(self, stack) -> None:
        """The figures ``stack``, given back by the run's page: the page puts them in their place."""
        raise NotImplementedError(f"{type(self).__name__} does not say where its figures go back")

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
            self.own_line(LIVE_TEXT, "warning")
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
        """The run's page in the right-hand column, holding this page's figures under its messages, of them only
        those its run made in sight (the others stay, out of sight, until the figures are given back): while this
        page's run is live, after a stop or a failure until "Back to the form", and again from "How the run
        went"."""
        self._figures_leaving()
        self.figure_stack.only_the_run(True)
        self.run_view.hold_figures(self.figure_stack)
        self.run_view.how_link.setVisible(False)
        self.results.setCurrentWidget(self.run_view)

    @property
    def past_view(self) -> RunView:
        """The page a run that is over is drawn on from its two files, by the run's controller: built the first time
        it is asked for, beside the run's page and never in its place, so that the run's page keeps its own run, its
        offer, its question and its failure; and holding no figures, since this page's are no past run's."""
        if self._past_view is None:
            self._past_view = self._new_run_page()
            self._past_view.setObjectName("past")
            self._past_view.back.connect(self._back_from_the_past)
            self.results.addWidget(self._past_view)
        return self._past_view

    def show_past_view(self) -> None:
        """The past view in the right-hand column, in the place of what the column showed."""
        shown = self.results.currentWidget()
        if shown is not self._past_view:
            self._before_past = shown
        self.results.setCurrentWidget(self.past_view)

    def _back_from_the_past(self) -> None:
        """"Back to the form" on the past view: the figures again, with "How the run went" above them when this page
        has a run of its own to show, as before the past view was shown."""
        own = self._before_past is self.run_view or self.run_view.how_link.isVisibleTo(self.figures)
        self.show_figures()
        self.run_view.how_link.setVisible(own)

    def show_figures(self) -> None:
        """The right-hand column back to this page's figures over its own lines, the figures the run's page held
        handed back to their place, at the heights they had, every one of them in sight again. A run of this page
        that is not live has ended: no figure that comes from now on is one of its."""
        if not self.is_running():
            self.figure_stack.run_ends()
        stack = self.run_view.release_figures()
        if stack is not None:
            self._figures_back(stack)
        self.figure_stack.only_the_run(False)
        self.results.setCurrentWidget(self.figures)

    def _run_began(self, spec) -> None:
        """This page's run has begun: its page takes the right-hand column, its figures those that come from now on,
        unless the run only loads a record. Called by the run's controller, once the run's page has been handed the
        run."""
        if not spec.loads_only:
            self.figure_stack.run_begins()
            self.show_run()

    def _run_over(self, status) -> None:
        """This page's run has ended, as ``status`` says. A run that finished gives the column back to the
        figures, "How the run went" above them; one that ended into idle (a refusal before anything was spent, a
        run that keeps no file and did not fail) gives it back with no link; a stopped or a failed run keeps its
        page until "Back to the form", a failed one whatever it keeps, since its page is where its failure is said;
        a run that only loads a record leaves the column as it is. When the keyboard was on the run's page, a
        finish gives it to "How the run went", for no key of its own. Called by the run's controller, once every
        panel is free again and the run's page has been drawn ended."""
        if status.loading:
            return
        self.figure_stack.run_ends()
        focus = QApplication.focusWidget()
        keys = focus is not None and self.run_view.isAncestorOf(focus)
        if status.state == "idle" and status.outcome != "failed":
            self.show_figures()
        elif status.outcome == "finished":
            self.show_figures()
            self.run_view.how_link.setVisible(True)
            if keys:
                self.run_view.how_link.setFocus(Qt.OtherFocusReason)

    def _back_to_the_form(self) -> None:
        """"Back to the form": the figures again, "How the run went" above them."""
        self.show_figures()
        self.run_view.how_link.setVisible(True)

    def _to_the_figures(self) -> None:
        """The figures in sight before a saved record's are opened among them: whatever shows in their place gives
        the column back by its own "Back to the form" -- the past view as it leaves, so that "How the run went" stands
        above the figures only when this page has a run of its own to show, and a run's page as it does."""
        shown = self.results.currentWidget()
        if shown is self.figures:
            return
        if shown is self._past_view:
            self._back_from_the_past()
        else:
            self._back_to_the_form()

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

    def _refusal(self, exc: Refusal) -> None:
        """Show a refusal: the yellow "Check your inputs" box, and no traceback.

        A refusal is something the program will not do with what it was given -- a blank box, a taken
        name, a training run one setting away from its cache -- and the fix is on this screen, so the
        box names it. The text is the core's neutral sentence (it names the setting, the rule, what
        was given and the default; never a box, tab or flag) and the informative line is this front
        end's own "where to fix it", looked up by the refusal's field key in core/gui/fields.py --
        the one place a control is named, so a renamed control is renamed once. No Details: nothing
        here is a crash, and a traceback would only say so louder. A bug keeps the red box
        (_on_error). The same sentence goes to the page's own lines at warning, so it outlives the click
        that dismisses the box.

        THE BOX ITSELF lives in ../widgets/refusal_box.py, shared with the Artifacts screen and the
        model builder, which show the same box and are deliberately not run hosts. What stays here
        is the page's own line, because only a page has those -- those two screens record the
        sentence on their own status lines instead.
        """
        fix = gui_fields.fix_sentence(exc.field)
        self.own_line(f"{exc.message} {fix}".rstrip(), "warning")
        show_refusal(self, exc)

    def _on_error(self, exc_or_message, tb: str, cancel_noted: bool = False, kept_fit=None) -> None:
        """Show a failure at a click. A Refusal goes to the yellow box (_refusal); anything else is a bug and
        gets the red one, with the traceback in a collapsible Details panel rather than pasted whole
        into the body (which produced an unscrollable, un-copyable wall of text stretched to the
        widest stack frame). A click handler that catches on the GUI thread passes what it caught
        (omitting ``cancel_noted``, which defaults to False -- a GUI-thread click has no worker cancel
        token to ask). A run's failure no longer comes here: it is a state of the run's page. The one
        exception is a run that only loads a record, which is never handed the run's page: the run's
        controller hands its failure here once it has ended, with the exception object
        WorkerSignals.error carried and what came with it. A plain string is still accepted and is
        still a bug.

        ``cancel_noted``: true when the cancel token had already been REQUESTED at the moment
        ``Worker.run`` caught this failure -- a coincidence, never a cause (the crash is reported as
        a crash regardless; see core/gui/worker.py). Shown as an extra line in both places a crash
        is reported, appended AFTER the crash itself: the box's informative text (below the one-line
        summary, above Details) and the page's own lines, at warning, after the error line. Never
        shown for a Refusal -- that box is about fixing an input, not about a run.

        ``kept_fit``: the training a failure in the fit kept, as the worker hands it on (the training
        stage's note, or None). This box does not show it.
        """
        if isinstance(exc_or_message, Refusal):
            return self._refusal(exc_or_message)
        message = str(exc_or_message)
        self.own_line(message, "error")
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Critical)
        box.setWindowTitle("Error")
        box.setText(message)
        if tb:
            box.setDetailedText(tb)
        if cancel_noted:
            box.setInformativeText(CANCEL_NOTED_TEXT)
            self.own_line(CANCEL_NOTED_TEXT, "warning")
        box.exec()
