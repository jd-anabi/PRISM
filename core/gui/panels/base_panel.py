"""The sink a run's figures go through on their way to the page that hosts the run: each rendered to PNG on the
worker's thread (``_png_fig_sink``), and said as one warning of the run when it cannot be shown. What makes a page
the host of a run is ``RunHost``'s, in ``run_host.py``."""
import warnings

# Said, as one warning of the run, when a figure the run handed over could not be shown, filled with the figure's
# title and the error's kind and message.
FIGURE_NOT_SHOWN_TEXT = "The figure '{}' could not be shown ({}). The record is not affected."


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
