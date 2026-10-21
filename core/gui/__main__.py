"""Entry point: ``python -m core.gui``, from any working directory (``core/config.py`` resolves
``Resources/`` and the artifact store from its own location; ``PRISM_RESOURCES`` / ``PRISM_ARTIFACTS``
override them).

The FIRST thing we do is force the Agg matplotlib backend, BEFORE importing any core.* module that
imports pyplot. Under Agg, figures are still created (and we embed them via FigureCanvasQTAgg), while
every stray ``plt.show()`` in the runner plotters becomes a harmless no-op — which is what makes it
safe to run the pipeline on a background thread."""
import matplotlib

matplotlib.use("Agg")

from core.gui import one_instance    # noqa: E402
from core.gui.app import build_app   # noqa: E402 -- must import after matplotlib.use


def main() -> int:
    app, window = build_app()
    if window is None:          # PRISM is open already: this start handed over to it, and ends with nothing shown
        return 0
    window.show()
    try:
        return app.exec()
    finally:
        # The window's close let go of the lock and the listener as it was accepted, though a figure window left
        # open keeps the loop running and a run on the thread pool may go on to its end unseen; a loop that ended
        # otherwise lets go of them here. Letting go twice does nothing.
        one_instance.release()


if __name__ == "__main__":
    raise SystemExit(main())
