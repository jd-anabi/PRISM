"""The card test: the smoke command's four stages -- the prior, the training, the calibration and a
simulated inference -- run end to end at tiny sizes, as one step.

It is run on the graphics card after a change to code that moves tensors: every suite runs on the
processor, and a tensor on the wrong device is invisible until the card sees it. It shows that the
chain runs, not that anything is calibrated: at these sizes the calibration has no power, so only a
failure means anything.

The step prints nothing and says nothing of its own; the stages say what they say. Its caller is told
each stage's start and end through ``on_stage`` and reads the rest from what the step returns: the
command line's ``smoke`` frames its console from these, and the window's screen will read the same.
The stages are looked up on ``core.orchestrator`` when they are called, and the seeding context on
``core.diagnostics.rng``, so a stage or the context replaced there is the one the step calls.
"""
import time
from dataclasses import dataclass, field

from core import orchestrator
from core.refusals import Refusal
from core.runs import public_entry

from . import rng

STAGES = ("prior", "posterior", "validate", "infer")


@dataclass(frozen=True)
class CardTestResult:
    """What one card test ran and what each stage returned: ``stages_run``, the stages that ran, in order;
    ``seconds``, each one's wall seconds; ``prior``, the prior built or loaded; ``posterior``, the one
    trained; ``calibration``, ``observation`` and ``inference``, the calibration's record and the simulated
    inference's two; each None when its stage did not run. ``conditioning_width`` is the width the network
    conditions on in the configuration's mode."""
    stages_run: tuple = ()
    seconds: dict = field(default_factory=dict)
    prior: object = None
    posterior: object = None
    calibration: object = None
    observation: object = None
    inference: object = None
    conditioning_width: int = 0


@public_entry
def card_test(cfg, *, cell, t_obs_s=None, seed=0, stages=STAGES, num_runs=4, run_size_cap=32, n_cal=40,
              max_num_epochs=5, hidden_features=None, num_transforms=None, checkpoint=False, save=False,
              prior_ref=None, resume=None, new_run=False, on_stage=None, fig_sink=None,
              store=None) -> CardTestResult:
    """The smoke command's four stages on ``cfg``, in order, inside one seeding context: one seed, the
    streams running on, since a seed per stage would replay the training's draws in the calibration set.

    ``stages`` is a subset of ``STAGES``; a stage not asked for is skipped. With no prior nothing further
    runs, and with no posterior neither the calibration nor the inference: the step returns there.
    ``prior_ref`` names a prior to load instead of building one; ``save`` names what this run builds,
    ``smoke_prior`` and ``smoke_posterior``. ``checkpoint`` keeps the training's simulation set, committed
    every ``max(1, num_runs // 2)`` batches; ``resume`` and ``new_run`` are the training's own.
    ``resume``, ``hidden_features`` and ``num_transforms`` reach the training only when given, so that
    each default stays the stage's own. ``t_obs_s`` is the length, in seconds, of the observation the
    inference simulates from ``cell`` (``config.T_MIN_EXP_S`` when None).

    Refused before the prior: a list naming a stage the card test does not have (no setting of its own
    answers that, so the refusal names no field); a ``cell`` that names no file or does not fit the
    bounds, when the inference is asked for (checked on a copy of the configuration, so the stages are
    handed none of its truth); and, when the training is asked for, a ``resume`` of "require" or "never"
    without ``checkpoint`` (no set is kept for the policy to act on; field ``resume``) and a ``resume`` of
    "require" without ``prior_ref`` (a prior built afresh is a new fit, which no kept set can match;
    field ``prior``).

    ``on_stage(stage, event, seconds)`` is called on the calling thread: "skipped" or "started" with
    None, then "finished" with the stage's wall seconds, or "failed" with them before the failure passes
    on. The inference's observation is checked after its stage: its width must be the mode's and what it
    conditions on finite.
    """
    from core import cli, config
    from core.SBI.statistics import SUMMARY_WIDTH
    stages = tuple(stages)
    unknown = [s for s in stages if s not in STAGES]
    if unknown:
        raise Refusal(f"The card test has no stage {unknown}: its stages are {list(STAGES)}.")
    if "posterior" in stages:
        if resume in ("require", "never") and not checkpoint:
            raise Refusal(f"resume={resume!r} needs checkpoint=True: without it no simulation set is read or "
                          f"written, so the policy has nothing to act on.", field="resume")
        if resume == "require" and prior_ref is None:
            raise Refusal("resume='require' needs prior_ref, the prior an earlier run built and saved: a prior "
                          "built afresh is a new fit, and the fit is part of a simulation set's identity, so "
                          "no kept set can match.", field="prior")
    if "infer" in stages:
        # The cell is first read by the inference, after the prior, the training and the calibration.
        # Checked here, on a copy, so that a mistyped one costs nothing and the stages are handed no truth.
        cli.load_and_validate_gt(cfg.copy_for_run(), cell)
    t_obs = config.T_MIN_EXP_S if t_obs_s is None else float(t_obs_s)
    want = SUMMARY_WIDTH + 1 + orchestrator.expected_forcing_dim(cfg)
    every = max(1, num_runs // 2) if checkpoint else 0
    given = {name: value for name, value in (("resume", resume), ("hidden_features", hidden_features),
                                              ("num_transforms", num_transforms)) if value is not None}
    seconds, ran = {}, []

    def tell(stage, event, took):
        if on_stage is not None:
            on_stage(stage, event, took)

    def run(stage, call):
        if stage not in stages:
            tell(stage, "skipped", None)
            return None
        began = time.time()
        tell(stage, "started", None)
        try:
            out = call()
        except BaseException:
            tell(stage, "failed", time.time() - began)
            raise
        seconds[stage] = time.time() - began
        ran.append(stage)
        tell(stage, "finished", seconds[stage])
        return out

    def result(**made) -> CardTestResult:
        return CardTestResult(stages_run=tuple(ran), seconds=dict(seconds), conditioning_width=want, **made)

    with rng.seeded(seed, cfg.hw.device):
        prior = run("prior", lambda: orchestrator.build_prior(
            cfg, prior_ref, prior_ref is None, fig_sink=fig_sink, store=store,
            name="smoke_prior" if (save and prior_ref is None) else ""))
        if prior is None:
            return result()
        posterior = run("posterior", lambda: orchestrator.build_posterior(
            cfg, prior, None, True, fig_sink=fig_sink, store=store, name="smoke_posterior" if save else "",
            num_runs=num_runs, run_size_cap=run_size_cap, max_num_epochs=max_num_epochs,
            checkpoint_every=every, new_run=new_run, **given))
        if posterior is None:
            return result(prior=prior)
        calibration = run("validate", lambda: orchestrator.validate_calibration(
            cfg, posterior, prior, fig_sink=fig_sink, store=store, n_cal=n_cal))
        inferred = run("infer", lambda: _infer(cfg, posterior, prior, t_obs, cell, want, fig_sink, store))
    observation, inference = inferred if inferred is not None else (None, None)
    return result(prior=prior, posterior=posterior, calibration=calibration, observation=observation,
                  inference=inference)


def _infer(cfg, posterior, prior, t_obs_s, cell, want, fig_sink, store):
    """The inference stage: the composition, then the two checks of what it observed. They run after the
    composition, which is one call, on a record already on disk, so nothing is lost by asking a second
    later."""
    import torch
    obs, inf = orchestrator.simulated_inference(cfg, posterior, t_obs_s, cell=cell, prior=prior,
                                                fig_sink=fig_sink, store=store)
    if obs.width != want:
        raise RuntimeError(f"observation width {obs.width} != the mode's {want}")
    if not torch.isfinite(obs.x_obs).all():
        raise RuntimeError("the observation carries non-finite conditioning")
    return obs, inf
