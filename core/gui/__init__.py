"""
PySide6 desktop GUI for the SBI research app.

The GUI and the command-line tool (``python -m core``, ``core/tool/``) are two front ends over the
same orchestrator stages and compositions. The GUI drives those stage functions
directly, reusing the pure ``cli.make_*`` config cores and the ``orchestrator`` stages, with heavy
work on a background thread and matplotlib figures embedded inline.

Launch: ``python -m core.gui`` (or ``run.bat``), from any working directory -- ``core/config.py``
resolves ``Resources/`` and the artifact store from its own location, and ``PRISM_RESOURCES`` /
``PRISM_ARTIFACTS`` override either.
"""
