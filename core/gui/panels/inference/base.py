"""What the pages of a training and of a narrowing round share: the stages of their runs, and the run itself as they
hand it to the run's controller."""
from core import config, orchestrator
from core.artifacts.provenance import _nvidia_smi_free_gib  # noqa: F401 -- re-exported for inference_tabs

from ...run.model import FIT_ENDED, SIMULATION_ROW, RunSpec, Stage

#: A training's stages, as its page hands them to its run: what begins each, and what counts it. The fit begins
#: when the simulation set is complete, or, for a training whose set is not kept, at the counts said when the
#: simulation ends; the saving of the posterior begins when the fit has ended.
TRAINING_STAGES = (Stage("Rotation", ("rotation_begins",)),
                   Stage("Simulation", ("simulation_writing", "simulation_resuming"), SIMULATION_ROW),
                   Stage("Fitting", ("simulation_complete", "patho_total", "chi_total", "chi_none"), "fit_epoch"),
                   Stage("Saving", (FIT_ENDED, "sbi_converged")))


def training_spec(word: str, place: tuple, stop_button: str, cfg, kwargs: dict, *, narrowed: bool,
                  expected_epochs: int | None = None, plan: dict | None = None,
                  settings: dict | None = None, audit: dict | None = None, prior: str | None = None,
                  prior_words: str = "") -> RunSpec:
    """The run of a training or of a narrowing round, as its page knows it: its stages; the patience and the epoch
    limit it trains with, the call's own or else the settings' (read, never written); and the facts its pace and
    its expected epochs are looked up by (the model, the kind of observation, whether it is narrowed, the network)
    with what the page knows before the run says it (whether it rotates, and the batches between two saves of
    its simulation set: the call's own, else what the stage reads), and what its first total of unusable driven
    recordings is compared with: the id of the prior in use (``prior``) and the audit of it that counted as the run
    started (``audit``, ``{"id", "fraction", "batches"}``; None for none), with the prior as a sentence names it."""
    def given(name, default):
        return default if kwargs.get(name) is None else kwargs[name]

    return RunSpec(word, place, name=kwargs.get("name", ""), stop_button=stop_button, stages=TRAINING_STAGES,
                   kinds=("posterior", "simulation", "fit"), simulation_set=True, retries=True,
                   patience=given("stop_after_epochs", config.TRAINING_STOP_AFTER_EPOCHS),
                   epoch_limit=given("max_num_epochs", config.TRAINING_MAX_NUM_EPOCHS),
                   expected_epochs=expected_epochs, plan=plan, settings=settings,
                   facts={"model": cfg.model, "observation": cfg.observation_mode, "narrowed": narrowed,
                          "network": (given("hidden_features", config.NSF_HIDDEN_FEATURES),
                                      given("num_transforms", config.NSF_NUM_TRANSFORMS)),
                          "rotated": bool(cfg.reparam_rotate),
                          "cadence": given("checkpoint_every", orchestrator.TRAINING_CHECKPOINT_EVERY),
                          "prior": prior, "audit": None if audit is None else dict(audit),
                          "prior_words": prior_words})
