"""Generic background worker: run a callable off the UI thread on the global QThreadPool, with its
log records, stdout/stderr/warnings routed to signals, and its return value / any figures / errors
emitted back."""
import traceback

from PySide6.QtCore import QObject, QRunnable, Signal, Slot

from .streams import WorkerCancelled, redirect_streams


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
    that was still propagating from one the code had already dealt with: a cancel raised inside a
    RECOVERED ``except`` block -- where the OOM ladders log -- carries the handled exception as its
    context, so treating a chained exception as THE failure would open a red box for an OOM the run
    survived. THE collision (a Cancel in the moment before a crash) is not fixed here at all: the
    pipeline's rescue write runs inside ``runs.cancel_deferred()``, so the original exception
    propagates by itself and lands in Worker.run's generic handler.

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


class WorkerSignals(QObject):
    log = Signal(str, str)          # (text, level in {"info","warning","error"}) -- panel-side messages
    log_batch = Signal(object)      # list[(text, level)]: one pump tick of pipeline output
    rows = Signal(object)           # tuple[vt.RowState]: ALL live progress rows (a full snapshot)
    figure = Signal(str, object, object)  # (title, png_bytes, fig_pickle | None) -- see base_panel._png_fig_sink
    chunk = Signal(object)          # one streamed numpy chunk (worker thread -> GUI) -- see base_panel.dispatch(provide_stream=)
    result = Signal(object)         # the callable's return value
    error = Signal(object, str)     # (exception, traceback) -- base_panel._on_error routes by type
    cancelled = Signal()            # the user cancelled: a stop, not a failure -- no error dialog
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
    """

    def __init__(self, fn, *args, cancel=None, **kwargs):
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.cancel = cancel
        self.signals = WorkerSignals()

    @Slot()
    def run(self):
        payload, failure, cancelled, in_flight = None, None, False, None
        try:
            with redirect_streams(self.signals, self.cancel):
                try:
                    payload = self.fn(*self.args, **self.kwargs)
                except WorkerCancelled as cancel:
                    # A cooperative cancel -- caught by name (BaseException, so it skipped the generic
                    # handler below). ALWAYS reported as a cancel: no traceback, no error dialog.
                    #
                    # THE COLLISION IS NOT HANDLED HERE. A Cancel pressed in the moment before a crash
                    # is handled where the work is: the pipeline's rescue save runs inside
                    # runs.cancel_deferred() (piece 4, B15), so the checkpoint it crosses does not
                    # fire, the rows are committed, and the ORIGINAL exception keeps propagating --
                    # into the generic handler below, which reports it like any other crash.
                    #
                    # What is left is the residual race: a checkpoint OUTSIDE any section firing while
                    # something is already unwinding. Python chains it, so __context__ still holds
                    # what was in flight, and that is NOT discarded -- its whole traceback goes to the
                    # log pane at ERROR below. It is deliberately not reported as THE failure: a
                    # cancel raised inside a RECOVERED `except` block (the OOM ladders log from
                    # exactly there) carries the handled exception as its context, so reporting the
                    # chain would open a red box for an OOM the run survived.
                    cancelled = True
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

            # Everything below runs with sys.stdout/stderr already restored and the pump drained and
            # stopped, so (a) every line the pipeline produced -- including a leave=True bar's final
            # frame, which is only flushed on teardown -- is queued AHEAD of the result, and (b) the
            # modal dialog that _on_error opens cannot spin a nested event loop while the process's
            # streams are still swapped out from under it. The in-flight note and its traceback are
            # emitted HERE, not inside the `with`, for the same ordering reason -- and BEFORE
            # cancelled.emit(), so the panel's "Run cancelled." stays the last line of the run.
            if cancelled:
                if in_flight is not None:
                    self.signals.log.emit("Cancelled while a failure was in flight. The cancel did "
                                          "not cause it; its traceback follows.", "warning")
                    self.signals.log.emit(in_flight, "error")
                self.signals.cancelled.emit()
            elif failure is None:
                self.signals.result.emit(payload)
            else:
                self.signals.error.emit(*failure)
                failure = None                       # the queued signal holds its own reference
        except RuntimeError:
            # "Signal source has been deleted": the window was closed while this run was still going,
            # so the QApplication and our WorkerSignals are already gone. Nothing to report to.
            return
        finally:
            try:
                self.signals.finished.emit()
            except RuntimeError:
                pass
