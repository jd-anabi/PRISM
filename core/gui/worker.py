"""Generic background worker: run a callable off the UI thread on the global QThreadPool, with its
log records, stdout/stderr/warnings routed to signals, and its return value / any figures / errors
emitted back. Given the run's text file, it hands that file every line of the run as it is written and
writes there how the run ended before it says anything to the window."""
import traceback

from PySide6.QtCore import QObject, QRunnable, Signal, Slot

from .streams import WorkerCancelled, redirect_streams

# Said, at warning, when a run RETURNS with the cancel token requested but never fired. The
# counterpart of ``base_panel.CANCEL_NOTED_TEXT``, which the failure path carries: there the run
# crashed, here it finished. Kept here rather than in base_panel because this module must not import
# a panel, and it is the worker that knows the token's final state.
CANCEL_TOO_LATE_TEXT = ("The run finished before the cancel could take effect; nothing was stopped, "
                        "and its result is complete.")

# Said, at warning, before the traceback of a failure a stop may have replaced (the WorkerCancelled
# branch of Worker.run says why it is hedged).
IN_FLIGHT_TEXT = ("Cancelled. There may be a failure that was still in flight -- it cannot be told apart "
                  "here from one the run had already recovered from. Its traceback follows, in case it helps.")

# How a run ended, in the run's text file: the window's own words, written before the worker says anything.
FINISHED_TEXT = "The run finished."
STOPPED_TEXT = "The run stopped: a stop was asked for."
# A failure: its kind always, then its message; its traceback goes under it.
FAILED_TEXT = "The run failed: {}"
# The failure of a run whose output could not be captured; filled with the reason the capture gave.
NO_CAPTURE_TEXT = "The run could not start: PRISM could not capture its output ({})."
# A run that ended with none of the three outcomes (a library's SystemExit passed every handler, or the capture's
# own exit failed): written before ``finished`` is said, so the run's text file never stops without an ending.
UNSAID_TEXT = "The run ended without saying how."


def _drop_tracebacks(exc: BaseException) -> None:
    """Set ``__traceback__`` to None on ``exc`` and on every exception along its ``__cause__`` /
    ``__context__`` chain, so none of them keeps the failed run's frames alive. A chain can loop (an
    exception re-raised as its own context's context), so each is visited once."""
    seen, stack = set(), [exc]
    while stack:
        e = stack.pop()
        if e is None or id(e) in seen:
            continue
        seen.add(id(e))
        e.__traceback__ = None
        stack += [e.__cause__, e.__context__]


def _pending_failure(cancel: BaseException) -> "BaseException | None":
    """The first non-cancel exception on ``cancel``'s ``__cause__``/``__context__`` chain, or None.

    A cancel checkpoint (a print, a tqdm redraw, a ``core`` record) that fires while something is
    already unwinding raises WorkerCancelled CHAINED to it, so the chain is the only place that
    exception still exists once the cancel has replaced it.

    It is looked up to be LOGGED, never to decide the run's verdict. The chain cannot tell a failure
    that was still propagating from one the code had already dealt with: a cancel raised
    (explicitly, not through a checkpoint -- see below) inside a RECOVERED ``except`` block carries
    the handled exception as its context, so treating a chained exception as THE failure would open
    a red box for something the run had already survived. Since ``core.runs.cancel_is_deferred``
    reads ``sys.exc_info()``, a checkpoint reached while this thread is handling an exception is
    DEFERRED rather than fired, so this function no longer covers the OOM ladders or the collision:
    those now reach here with nothing chained (see the ``WorkerCancelled`` branch below). What is
    left is explicit: a bare ``raise WorkerCancelled()`` written inside a handler, which no
    checkpoint in this codebase does any more, but which the fallback still must not misreport. THE
    OOM LADDERS DO NOT LOG FROM INSIDE THEIR RECOVERED ``except`` -- an earlier version of this
    docstring claimed they did. They log AFTER it has closed (core/SBI/pipeline.py's
    ``_gen_obs_retry``, ``_rows_with_oom_retry``, ``retry_on_oom`` and the per-batch retry in
    ``gen_training_data``); the guards that genuinely log from inside a recovered handler, and so
    are the real false-positive risk for the fallback, are the three best-effort ones:
    ``_release_device_memory``'s ``_try``, ``_try_rng_snapshot`` and ``_try_rng_restore``.

    ``__cause__`` first (an explicit ``raise ... from``), then ``__context__``. A chain can loop, so
    each link is visited once.
    """
    seen, e = set(), cancel
    while e is not None and id(e) not in seen:
        seen.add(id(e))
        e = e.__cause__ if e.__cause__ is not None else e.__context__
        if e is not None and not isinstance(e, WorkerCancelled):
            return e
    return None


def _failed_text(exc: BaseException) -> str:
    """A failure as the run's text file says it: its kind always, then its message when it has one. A message that
    cannot be had (the exception's ``str`` raises) is left out, so the failure is still written and said."""
    try:
        message = str(exc)
    except Exception:                         # noqa: BLE001 -- the failure is said by its kind alone
        message = ""
    return FAILED_TEXT.format(f"{type(exc).__name__}: {message}" if message else type(exc).__name__)


class WorkerSignals(QObject):
    log = Signal(str, str)          # (text, level in {"info","warning","error"}) -- panel-side messages
    log_batch = Signal(object)      # list[(text, level)]: one pump tick of pipeline output
    rows = Signal(object)           # tuple[vt.RowState]: ALL live progress rows (a full snapshot)
    figure = Signal(str, object, object)  # (title, png_bytes, fig_pickle | None) -- see base_panel._png_fig_sink
    chunk = Signal(object)          # one streamed numpy chunk (worker thread -> GUI) -- see base_panel.dispatch(provide_stream=)
    result = Signal(object)         # the callable's return value
    error = Signal(object, str, bool, object)  # (exception, traceback, cancel_noted, kept training) --
                                        # base_panel._on_error routes by type; cancel_noted says whether the
                                        # cancel token had been requested when this failure was caught -- a
                                        # coincidence to mention, never a cause to claim; the kept training is
                                        # the note the training stage puts on a failure in its fit, or None.
    cancelled = Signal(object)      # the user cancelled: a stop, not a failure -- no error dialog. Carries the
                                    # note naming the training the stop kept, or None.
    finished = Signal()


class Worker(QRunnable):
    """Run ``fn(*args, **kwargs)`` on a worker thread. A panel connects ``signals`` to its widgets.

    If the callable takes a ``fig_sink`` (a ``(title, fig) -> None`` display hook), the panel injects
    one that renders the Figure to PNG bytes here on the worker thread -- a figure created off-thread
    must never be painted by a live canvas (it deadlocks on matplotlib's global lock). It also pickles
    the Figure (best-effort) so the panel can rebuild an interactive copy on the GUI thread (the
    "Pop out" button); pickling never renders, so it is safe here.

    ``cancel`` (a streams.CancelToken) makes the pipeline's next print/redraw raise WorkerCancelled, so
    the run unwinds cooperatively.

    ``run_text`` (the run's ``core.gui.run.textfile.RunTextFile``), when given, is the run's line sink:
    every finished line of the run reaches it as it is written; the lines this worker says itself
    reach it too, and how the run ended, each written before any signal says it (a run that ended
    with none of the three outcomes is written as one that ended without saying how, before
    ``finished``). ``cancelled`` and ``error`` carry the kept training a stop or a failure in the
    fit names (``kept_fit`` on the exception the training stage raises), or None.
    """

    def __init__(self, fn, *args, cancel=None, run_text=None, **kwargs):
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.cancel = cancel
        self.run_text = run_text
        self.signals = WorkerSignals()

    @Slot()
    def run(self):
        payload, failure, cancelled, in_flight, cancel_noted = None, None, False, None, False
        kept_fit, entered, text, ended = None, False, self.run_text, False
        try:
            try:
                with redirect_streams(self.signals, self.cancel, line_sink=text.line if text is not None else None):
                    entered = True
                    try:
                        payload = self.fn(*self.args, **self.kwargs)
                    except WorkerCancelled as cancel:
                        # A cooperative cancel -- caught by name (BaseException, so it skipped the generic
                        # handler below). ALWAYS reported as a cancel: no traceback, no error dialog.
                        #
                        # THE COLLISION AND THE RESIDUAL RACE ARE BOTH HANDLED BEFORE THIS BRANCH RUNS,
                        # by runs.cancel_is_deferred(): a checkpoint reached while this thread is
                        # handling an exception -- any `except`, any `finally` or `__exit__` an
                        # exception entered, any generator teardown (a `leave=True` tqdm bar's closing
                        # write included) -- is deferred rather than fired. So a crash unwinding through
                        # the pipeline's rescue save (inside its own explicit section) OR through an
                        # ordinary `finally` that logs (core/Solvers/sdeint.py's bar teardown, the
                        # likeliest GPU crash site) keeps propagating as itself and reaches the generic
                        # handler below, which reports it like any other crash -- with the cancel NOTED
                        # rather than substituted (the `except Exception` branch below reads the same
                        # token for that; nothing here needs it, because a run THIS branch reports is,
                        # by definition, a cancel and never reaches that branch).
                        #
                        # What reaches THIS branch now is only a WorkerCancelled that a checkpoint raised in
                        # NORMAL flow (a plain cancel, __context__ empty) or one written EXPLICITLY inside a
                        # handler -- no checkpoint in this codebase does that. The fallback below still
                        # covers that last, rare shape: __context__ may hold an exception the code had
                        # already RECOVERED from (a cancel raised inside a recovered `except`, such as the
                        # three best-effort guards in core/SBI/pipeline.py -- see _pending_failure), and the
                        # chain cannot tell that apart from one still propagating, so it is never reported
                        # as THE failure -- only kept, hedged, for the log.
                        cancelled = True
                        # The training the stop kept, when the training stage named one: read HERE, off the
                        # exception this clause caught, whose name is unbound once the clause ends.
                        kept_fit = getattr(cancel, "kept_fit", None)
                        original = _pending_failure(cancel)
                        if original is not None:
                            # rstrip: format_exception ends on a newline, which the pane would render as
                            # a blank line after the traceback.
                            in_flight = "".join(traceback.format_exception(
                                type(original), original, original.__traceback__)).rstrip()
                        # ...and then the frames go, for the reason the Exception branch gives below.
                        # _drop_tracebacks walks the chain, so `original` is covered by this one call.
                        _drop_tracebacks(cancel)
                    except Exception as e:               # noqa: BLE001 -- surface any failure to the UI
                        # The EXCEPTION, not its text. The panel opens the yellow "Check your inputs"
                        # box for a Refusal and the red one with the traceback for anything else, and
                        # it can only tell the two apart if the object itself crosses the thread.
                        failure = (e, traceback.format_exc())
                        # "The cancel noted": the cancel checkpoint that would have raised here was
                        # deferred instead (see the WorkerCancelled branch above), so a
                        # REQUESTED-but-never-fired token is exactly what a crash mid-unwind of a
                        # pending Cancel leaves behind. Read here, before anything below clears it, and
                        # carried to the panel alongside the failure -- never as a cause, only as a fact
                        # worth telling the person who clicked Cancel.
                        cancel_noted = self.cancel is not None and self.cancel.requested.is_set()
                        # The training a failure in the fit kept, read off the exception as the stop's is.
                        kept_fit = getattr(e, "kept_fit", None)
                        # ...but not its traceback. The traceback owns every frame of the failed run
                        # (the stage's host buffers, the prior, CUDA tensors), and this frame heads it
                        # while `failure` holds the exception: a cycle only a full collection frees.
                        # The panel needs the type, message, field and the text formatted above.
                        _drop_tracebacks(e)
                    finally:
                        # Stray figures a stage built but never handed to the sink (e.g. it unwound on a
                        # cancel before _emit): harmless under Agg, but they pile up across cancelled runs.
                        try:
                            import matplotlib.pyplot as plt
                            plt.close("all")
                        except Exception:                # noqa: BLE001 -- cleanup must not mask the outcome
                            pass
            except RuntimeError as e:
                if entered:
                    raise
                # The capture could not start (the thread its pump needs could not be made, say): the run
                # never ran, and that is a failure said as one, never swallowed. redirect_streams has let
                # go of everything it took.
                failure = (RuntimeError(NO_CAPTURE_TEXT.format(e)), traceback.format_exc())
                _drop_tracebacks(e)

            # Everything below runs with sys.stdout/stderr already restored and the pump drained and
            # stopped, so (a) every line the pipeline produced -- including a leave=True bar's final
            # frame, which is only flushed on teardown -- is queued AHEAD of the result, and (b) the
            # modal dialog that _on_error opens cannot spin a nested event loop while the process's
            # streams are still swapped out from under it. The in-flight note and its traceback are
            # emitted HERE, not inside the `with`, for the same ordering reason -- and BEFORE
            # cancelled.emit(), so the panel's "Run cancelled." stays the last line of the run. The
            # run's text file gets each of them, and how the run ended, before any of it is said.
            if cancelled:
                # HEDGED, deliberately, the in-flight note: what is chained under a WorkerCancelled
                # reaching this branch can only be an exception the code had already RECOVERED from (see
                # the WorkerCancelled branch's comment), and the chain cannot prove that -- so this never
                # claims the exception caused anything, only that it MAY still have been propagating.
                if text is not None:
                    if in_flight is not None:
                        text.line(IN_FLIGHT_TEXT, "warning")
                        text.line(in_flight, "error")
                    text.outcome(STOPPED_TEXT, "window")
                    ended = True
                if in_flight is not None:
                    self.signals.log.emit(IN_FLIGHT_TEXT, "warning")
                    self.signals.log.emit(in_flight, "error")
                self.signals.cancelled.emit(kept_fit)
            elif failure is None:
                # THE CANCEL THAT ARRIVED TOO LATE. A cancel taken inside a deferral window is
                # deferred, not discarded -- "the next check outside raises" -- and that is true
                # only if there IS a next check. A stage whose last write happens while an exception
                # is unwinding (core/Solvers/sdeint.py's bar teardown is on the graphed solver path)
                # can finish with the token requested and never fired. The run really did finish, so
                # it is NOT reported as cancelled: the payload is correct and nothing on disk is
                # wrong. It is reported as the success it is, with the fact said -- the counterpart
                # of cancel_noted on the failure path above.
                too_late = self.cancel is not None and self.cancel.requested.is_set()
                if text is not None:
                    if too_late:
                        text.line(CANCEL_TOO_LATE_TEXT, "warning")
                    text.outcome(FINISHED_TEXT, "window")
                    ended = True
                if too_late:
                    self.signals.log.emit(CANCEL_TOO_LATE_TEXT, "warning")
                self.signals.result.emit(payload)
            else:
                if text is not None:
                    text.outcome(_failed_text(failure[0]), "error", failure[1])
                    ended = True
                self.signals.error.emit(failure[0], failure[1], cancel_noted, kept_fit)
                failure = None                       # the queued signal holds its own reference
        except RuntimeError:
            # "Signal source has been deleted": the window was closed while this run was still going,
            # so the QApplication and our WorkerSignals are already gone. Nothing to report to.
            return
        finally:
            # None of the three outcomes was written: a BaseException no clause above takes (a library's
            # SystemExit) left the callable, or the capture's exit raised. The file still says the run ended,
            # before `finished` says it; whatever is propagating goes on as it would have.
            if text is not None and not ended:
                text.outcome(UNSAID_TEXT, "window")
            try:
                self.signals.finished.emit()
            except RuntimeError:
                pass
