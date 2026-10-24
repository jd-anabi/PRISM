"""The seeding context's old address, kept as a re-export.

``seeded`` lives in ``core/rng.py`` because the FDT measurement needs it too, and importing it from
here runs ``core/diagnostics/__init__.py``, which loads the whole inference stack
(core/rng.py's docstring says why that matters). This name stays so that ``from .rng import seeded``
in the diagnostics keeps working, and so that the card test step (``card.py``, which the ``smoke``
command calls) can read ``rng.seeded`` here each time it runs, where the tool tests patch it -- it
is the SAME object, not a copy.
"""
from core.rng import seeded  # noqa: F401
