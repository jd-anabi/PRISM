"""Generic background worker: run a callable off the UI thread on the global QThreadPool, with its
stdout/stderr/warnings routed to signals, and its return value / any figures / errors emitted back."""
import traceback

from PySide6.QtCore import QObject, QRunnable, Signal, Slot

from .streams import WorkerCancelled, redirect_streams


class WorkerSignals(QObject):
    log = Signal(str, str)          # (text, level in {"info","warning","error"}) -- panel-side messages
    log_batch = Signal(object)      # list[(text, level)]: one pump tick of pipeline output
    rows = Signal(object)           # tuple[vt.RowState]: ALL live progress rows (a full snapshot)
    figure = Signal(str, object, object)  # (title, png_bytes, fig_pickle | None) -- see base_panel._png_fig_sink
    result = Signal(object)         # the callable's return value
    error = Signal(str, str)        # (message, traceback)
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
        payload, failure, cancelled = None, None, False
        try:
            with redirect_streams(self.signals, self.cancel):
                try:
                    payload = self.fn(*self.args, **self.kwargs)
                except WorkerCancelled:
                    # A cooperative cancel -- caught by name (BaseException, so it skipped the generic
                    # handler below). Not a failure: report it as such, no traceback, no error dialog.
                    cancelled = True
                except Exception as e:               # noqa: BLE001 -- surface any failure to the UI
                    failure = (str(e), traceback.format_exc())
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
            # streams are still swapped out from under it.
            if cancelled:
                self.signals.cancelled.emit()
            elif failure is None:
                self.signals.result.emit(payload)
            else:
                self.signals.error.emit(*failure)
        except RuntimeError:
            # "Signal source has been deleted": the window was closed while this run was still going,
            # so the QApplication and our WorkerSignals are already gone. Nothing to report to.
            return
        finally:
            try:
                self.signals.finished.emit()
            except RuntimeError:
                pass
