"""The seeding context's old address, kept as a re-export.

``seeded`` lives in ``core/rng.py`` since Task 17's fix round 1: the FDT measurement needs it, and
importing it from here runs ``core/diagnostics/__init__.py``, which loads the whole inference stack
(core/rng.py's docstring says why that matters). This name stays so ``from .rng import seeded`` in
the diagnostics and ``core.diagnostics.rng.seeded`` in ``smoke`` (which a tool test patches there)
keep working -- it is the SAME object, not a copy.
"""
from core.rng import seeded  # noqa: F401
