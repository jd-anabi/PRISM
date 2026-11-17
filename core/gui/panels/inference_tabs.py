"""Parameter inference, in six steps over ONE shared SbiSession owned by the InferenceScreen.

    Model and options -> Prior -> Training -> Checks -> Inference -> Narrowing round

The first step records the model-level choices as a ConfigDraft; the Prior step picks the bounds file and turns that
draft into the SimConfig, because the bounds file declares the inferred parameter set (and hence the observation mode),
so a configuration cannot exist before it.

Each step is a page of the look (``StepPage``), so a run's dispatch, its stop, its figures and its page are the shared
run machinery; the run controller lets one run go at a time in the process. The screen owns the SbiSession and holds
the steps as pages, none disabled; every step reads and writes the session through its screen.
"""
# THE IMPORT SURFACE, kept deliberately: help_screen reads this module's HELP and docstring BY
# STRING PATH ("core.gui.panels.inference_tabs"), inference_screen imports the six pages from here,
# and the test suites reach _nvidia_smi_free_gib through it. Each step dispatches an orchestrator
# composition directly. The pages live in the inference/ package, one module per step plus the shared matter.
from .inference.help_text import HELP  # noqa: F401
from .inference.base import _nvidia_smi_free_gib  # noqa: F401
from .inference.config_tab import ConfigPanel  # noqa: F401
from .inference.prior_tab import PriorPanel  # noqa: F401
from .inference.posterior_tab import PosteriorPanel  # noqa: F401
from .inference.validate_tab import ValidatePanel  # noqa: F401
from .inference.infer_tab import InferPanel  # noqa: F401
from .inference.tsnpe_tab import TSNPEPanel  # noqa: F401
