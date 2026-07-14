"""Entry point: ``python -m core.gui`` (run from the repo root so Resources/ resolves).

The FIRST thing we do is force the Agg matplotlib backend, BEFORE importing any core.* module that
imports pyplot. Under Agg, figures are still created (and we embed them via FigureCanvasQTAgg), while
every stray ``plt.show()`` in the runner plotters becomes a harmless no-op — which is what makes it
safe to run the pipeline on a background thread."""
import matplotlib

matplotlib.use("Agg")

from core.gui.app import build_app   # noqa: E402 -- must import after matplotlib.use


def main() -> int:
    app, window = build_app()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
