"""Diagnostics: measurements ABOUT a trained posterior, each writing one ``diagnostic`` artifact.

Moved out of ``scripts/`` in piece 2 of the 2026-09-11 one-flow design, where they were driven by
environment variables, could not be tested, and wrote nothing the store could describe. The five
functions -- ``sbc_repeats``, ``identifiability_rotation``, ``identifiability_laplace``,
``identifiability_jacobian`` and ``channel_ablation`` -- arrive with T16-T18 and are re-exported from
here then. This module lands the two pieces all of them share.

``feature_sets`` answers "which features does THIS config's posterior actually condition on", which
is what makes a Jacobian diagnostic a statement about the experiment that was run rather than about
a 41-feature single-frequency one that was not. ``rng`` is the one seeding context: a diagnostic
seeds once, lets the stream run on, and hands the caller's RNG back.

The function names differ from the module names on purpose, so no function shadows its submodule.
"""
from .sbc import sbc_repeats  # noqa: F401
