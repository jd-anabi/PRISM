"""Session state for the SBI panel: the config + the artifacts produced by each stage, held so the
GUI can drive the pipeline one stage at a time (build prior -> train posterior -> validate -> infer)."""
from dataclasses import dataclass
from typing import Any


@dataclass
class SbiSession:
    cfg: Any = None                 # SimConfig (bounds-only until a cell injects ground truth)
    inf_prior: Any = None           # physical inferred product prior (from build_prior)
    force_prior: Any = None         # forcing prior (from build_prior)
    posterior: Any = None           # TransformedPosterior (from build_posterior)
    diagnostics: Any = None         # training diagnostics dict (loss curve etc.)
    posterior_latent: Any = None    # raw latent DirectPosterior, for deferred save
    V: Any = None                   # decorrelating rotation, for the deferred .rot.pt sidecar

    def reset_downstream(self, from_stage: str) -> None:
        """Invalidate artifacts that depend on an earlier stage when it is re-run."""
        order = ["config", "prior", "posterior", "validate"]
        i = order.index(from_stage)
        if i <= order.index("prior"):
            self.inf_prior = self.force_prior = None
        if i <= order.index("posterior"):
            self.posterior = self.diagnostics = self.posterior_latent = self.V = None
