"""
PySide6 desktop GUI for the SBI research app.

An ADDITIONAL front-end (the CLI ``python -m core`` and the diagnostic scripts remain the source of
truth for the pipeline logic). The GUI drives the already-decomposed orchestrator stage functions
directly, reusing the pure ``cli.make_*`` config cores and the ``orchestrator`` stages, with heavy
work on a background thread and matplotlib figures embedded inline.

Launch: ``python -m core.gui`` (or ``run.bat``), from any working directory -- ``core/config.py``
resolves ``Resources/`` and the artifact store from its own location, and ``PRISM_RESOURCES`` /
``PRISM_ARTIFACTS`` override either.
"""
