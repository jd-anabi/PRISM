"""Session state for Parameter inference: the config + the artifacts produced by each stage, held so the
GUI can drive the pipeline one stage at a time (build prior -> train posterior -> validate -> infer)."""
from dataclasses import dataclass
from typing import Any


@dataclass
class ConfigDraft:
    """The MODEL-level choices made in step 1, Model and options, before a bounds file exists.

    A SimConfig cannot be built without bounds -- the bounds file is what declares which parameters are
    inferred, in what order, and over what range (and hence the observation mode). So step 1 records
    intent here, and the Prior step turns it into a SimConfig once a bounds file is chosen.
    """
    model: str
    labels: Any
    state_dep_drift: bool
    units_override: Any = None          # units file path / token tuple; None => the per-model default
    chi_mode: bool = False
    chi_n_freqs: Any = None
    chi_f0: Any = None
    chi_freq_bounds: Any = None
    chi_k_pad: Any = None               # probe-slot capacity; frozen into any posterior trained with it
    chi_max_cycles: Any = None          # lock-in duration ceiling; also frozen into the artifact
    reparam_rotate: Any = None

    def make_config(self, bounds_path: str = None, *, bounds_dicts=None, hw=None):
        """Turn this draft + a bounds source into a SimConfig (the Prior step's first act).

        Pass either a bounds FILE path or hand-entered ``bounds_dicts`` -- exactly one. ``hw``, the device the
        configuration is made for, None for the one this computer offers, as with no ``hw`` at all."""
        from core import cli
        return cli.make_sim_config(
            self.model, self.labels, self.state_dep_drift, bounds_path, bounds_dicts=bounds_dicts,
            units_override=self.units_override, chi_mode=self.chi_mode,
            chi_n_freqs=self.chi_n_freqs, chi_f0=self.chi_f0,
            chi_freq_bounds=self.chi_freq_bounds, chi_k_pad=self.chi_k_pad,
            chi_max_cycles=self.chi_max_cycles, reparam_rotate=self.reparam_rotate, hw=hw)


@dataclass
class SbiSession:
    draft: Any = None               # ConfigDraft from step 1, Model and options (model + units + knobs)
    cfg: Any = None                 # SimConfig (built at the Prior stage, once bounds are chosen)
    inf_prior: Any = None           # artifacts.LoadedPrior -- .prior (ProductPrior), .force_prior, .id, .manifest
    posterior: Any = None           # artifacts.LoadedPosterior -- .posterior (TransformedPosterior, carrying
                                    # .T, .truncation, .x_obs_digest), .latent, .id, .manifest, .diagnostics
    observation: Any = None         # artifacts.LoadedObservation -- the Inference step's last product; the
                                    # Narrowing round's input

    def reset_downstream(self, from_stage: str) -> None:
        """Invalidate artifacts that depend on an earlier stage when it is re-run. Everything here is
        already on disk (every stage writes at completion), so this only drops the session's handles."""
        order = ["config", "prior", "posterior", "validate"]
        i = order.index(from_stage)
        if i <= order.index("prior"):
            self.inf_prior = self.observation = None     # an observation is tied to a config's conditioning
        if i <= order.index("posterior"):
            self.posterior = None
