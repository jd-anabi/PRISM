"""
Backward-compatibility shim. All logic has moved to:
  - core.config   (DeviceConfig, SimConfig, constants)
  - core.cli      (interactive prompts)
  - core.orchestrator (pipeline flow)

This file re-exports setup() and run() so that any code doing
``import core.app as app; app.setup(); app.run(...)`` still works,
but new code should use core.cli.build_sim_config() + core.orchestrator.run().
"""
import warnings

from .cli import build_sim_config
from .orchestrator import run as _run
from .config import *  # noqa: F401,F403  -- re-export constants


def setup():
    """Legacy entry point. Returns a SimConfig instead of the old 9-element tuple."""
    warnings.warn(
        "app.setup() is deprecated. Use core.cli.build_sim_config() instead.",
        DeprecationWarning, stacklevel=2,
    )
    cfg = build_sim_config()
    # Return the config object -- callers that unpacked a tuple will need updating
    return cfg


def run(cfg_or_inits=None, params_dict=None, rescale_params=None,
        force_params_dict=None, units_dict=None, si_factors=None,
        model=None, labels=None, state_dep_drift=None):
    """Legacy entry point. Accepts either a SimConfig or the old positional args."""
    warnings.warn(
        "app.run() is deprecated. Use core.orchestrator.run(cfg) instead.",
        DeprecationWarning, stacklevel=2,
    )
    from .config import SimConfig
    if isinstance(cfg_or_inits, SimConfig):
        _run(cfg_or_inits)
    else:
        raise TypeError(
            "app.run() now requires a SimConfig object. "
            "Use cfg = core.cli.build_sim_config(); core.orchestrator.run(cfg)"
        )
