"""Diagnostics: measurements ABOUT a trained posterior, a cell or a prior, each writing one
``diagnostic`` artifact.

Moved out of the retired ``scripts/`` directory, where they were driven by environment variables,
could not be tested, and wrote nothing the store could describe. The five functions --
``sbc_repeats``, ``identifiability_rotation``, ``identifiability_laplace``,
``identifiability_jacobian`` and ``channel_ablation`` -- are re-exported from here, beside the two
submodules all of them share. So are the probe checks (``probes``), which re-measure the chi probe
settings on a lab's own cell and change none of them: ``probe_band`` asks whether the configured band
and drive hold for a cell. Their shared geometry and spectrum helpers are ``probe_math``.

``feature_sets`` answers "which features does THIS config's posterior actually condition on", which
is what makes a Jacobian diagnostic a statement about the experiment that was run rather than about
a 41-feature single-frequency one that was not. ``rng`` is the one seeding context: a diagnostic
seeds once, lets the stream run on, and hands the caller's RNG back.

The function names differ from the module names on purpose, so no function shadows its submodule.
"""
from .identifiability import (identifiability_jacobian, identifiability_laplace,  # noqa: F401
                              identifiability_rotation)
from .sbc import sbc_repeats  # noqa: F401
from .ablation import channel_ablation  # noqa: F401
from .probes import probe_band  # noqa: F401
