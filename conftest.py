"""Root pytest configuration (piece 0 of the 2026-09-10 hardening programme).

Lives at the repo root so it covers BOTH test trees named in pytest.ini (tests/ and
core/Reduction/tests/). It runs before any test module is imported, which is the whole point:

* QT_QPA_PLATFORM=offscreen  -- headless Qt; a test module that imports PySide6 first would
  otherwise try to open a display.
* KMP_DUPLICATE_LIB_OK=TRUE  -- torch + MKL ship two OpenMP runtimes on Windows/conda; without
  this, the first simulation aborts the interpreter with OMP Error #15 (no traceback).
* matplotlib.use("Agg")      -- no figure windows.

Markers (registered in pytest.ini) are turned into skips here: `gpu` when CUDA is absent,
`display` when Qt is offscreen. Fixtures live in tests/conftest.py.
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import pytest  # noqa: E402


def pytest_collection_modifyitems(config, items):
    import torch

    if not torch.cuda.is_available():
        skip_gpu = pytest.mark.skip(reason="needs CUDA (torch.cuda.is_available() is False)")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)
    if os.environ.get("QT_QPA_PLATFORM") == "offscreen":
        skip_display = pytest.mark.skip(reason="needs a real display (QT_QPA_PLATFORM is offscreen)")
        for item in items:
            if "display" in item.keywords:
                item.add_marker(skip_display)
