"""The stage subcommands of ``python -m core``: one thin handler per orchestrator stage.

A handler resolves nothing of its own. Every value flag defaults to None and is forwarded only when it
was given, so the stage's own default -- read from config.py, in one place -- is the default. The heavy
imports live inside the handlers, so building the parser costs no torch.
"""
from .config_args import (UsageError, accept_from, add_accept_flags, add_config_flags, add_name_flags,
                          build_cfg, close_sink, knobs, load_posterior_and_prior, recording_set, report)

PRIOR_KNOBS = ("num_iterations", "sweep_batch", "max_sets", "walk_step", "stability_units",
               "min_cluster_size", "min_samples")
TRAIN_KNOBS = ("num_runs", "run_size_cap", "hidden_features", "num_transforms", "learning_rate",
               "stop_after_epochs", "max_num_epochs", "fisher_m", "fisher_dz", "fisher_points",
               "checkpoint_every", "resume")
VALIDATE_KNOBS = ("n_cal", "cal_n_scales", "num_posterior_samples")
# No Fisher knobs, by design: with a region set, build_posterior reuses the parent's basis and never
# reaches the Fisher, so a --fisher-* flag could only ever be silently ignored.
TSNPE_KNOBS = ("n_directions", "level", "num_runs", "run_size_cap", "hidden_features",
               "num_transforms", "learning_rate", "stop_after_epochs", "max_num_epochs",
               "checkpoint_every", "resume")


def register(subparsers) -> dict:
    """Add this module's subcommands; ``{name: subparser}`` for ``build_parser``."""
    out = {}

    p = subparsers.add_parser("prior", help="build a stability-screened GMM prior from a bounds file")
    add_config_flags(p)
    add_name_flags(p)
    p.add_argument("--num-iterations", type=int, default=None, metavar="N",
                   help="stability sweep iterations")
    p.add_argument("--sweep-batch", type=int, default=None, metavar="N",
                   help="candidate parameter sets per sweep iteration")
    p.add_argument("--max-sets", type=int, default=None, metavar="N",
                   help="accepted sets the sweep stops at")
    p.add_argument("--walk-step", type=float, default=None, metavar="X",
                   help="random-walk step, as a fraction of each box side")
    p.add_argument("--stability-units", type=float, default=None, metavar="X",
                   help="ND time units a candidate must stay bounded for")
    p.add_argument("--min-cluster-size", type=int, default=None, metavar="N",
                   help="HDBSCAN minimum cluster size")
    p.add_argument("--min-samples", type=int, default=None, metavar="N",
                   help="HDBSCAN min_samples")
    p.set_defaults(handler=_prior)
    out["prior"] = p

    p = subparsers.add_parser("train", help="train an amortized posterior (NPE) on a prior")
    add_config_flags(p)
    add_name_flags(p)
    p.add_argument("--prior", required=True, metavar="REF",
                   help="prior artifact to train on, by name or id")
    p.add_argument("--num-runs", type=int, default=None, metavar="N",
                   help="training batches to simulate")
    p.add_argument("--run-size", dest="run_size_cap", type=int, default=None, metavar="N",
                   help="ceiling on simulations per batch (0 = the hardware default)")
    p.add_argument("--hidden-features", type=int, default=None, metavar="N",
                   help="flow width per transform")
    p.add_argument("--num-transforms", type=int, default=None, metavar="N", help="flow depth")
    p.add_argument("--learning-rate", type=float, default=None, metavar="X", help="Adam LR")
    p.add_argument("--stop-after-epochs", type=int, default=None, metavar="N",
                   help="early-stopping patience")
    p.add_argument("--max-epochs", dest="max_num_epochs", type=int, default=None, metavar="N",
                   help="hard epoch ceiling")
    p.add_argument("--fisher-m", type=int, default=None, metavar="N",
                   help="ensemble per latent perturbation for the Fisher rotation")
    p.add_argument("--fisher-dz", type=float, default=None, metavar="X",
                   help="latent central-difference step")
    p.add_argument("--fisher-points", type=int, default=None, metavar="N",
                   help="operating points the Fisher is averaged over")
    p.add_argument("--checkpoint-every", type=int, default=None, metavar="N",
                   help="batches between checkpoint commits (0 = no cache, nothing resumable)")
    p.add_argument("--resume", choices=("auto", "require", "never"), default=None,
                   help="auto resumes this run's own cache; require refuses when there is none; "
                        "never refuses to resume one (default: auto)")
    p.add_argument("--new-run", action="store_true",
                   help="start a new simulation cache even though a committed one ONE setting away "
                        "exists; it silences that refusal and nothing else")
    p.set_defaults(handler=_train)
    out["train"] = p

    p = subparsers.add_parser("validate", help="SBC + TARP calibration of a posterior (no observation)")
    add_config_flags(p)
    add_name_flags(p)
    p.add_argument("--posterior", required=True, metavar="REF",
                   help="posterior artifact to calibrate, by name or id")
    p.add_argument("--n-cal", type=int, default=None, metavar="N",
                   help="calibration datasets to simulate")
    p.add_argument("--cal-n-scales", type=int, default=None, metavar="N",
                   help="distinct (t_scale, T) strata across the calibration set")
    p.add_argument("--posterior-samples", dest="num_posterior_samples", type=int, default=None,
                   metavar="N", help="posterior draws per calibration dataset")
    add_accept_flags(p)
    p.set_defaults(handler=_validate)
    out["validate"] = p

    p = subparsers.add_parser("infer", help="run a posterior on one observation, simulated or measured")
    add_config_flags(p)
    add_name_flags(p)
    p.add_argument("--posterior", required=True, metavar="REF")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--cell", default=None, metavar="PATH",
                     help="simulated: the cell file whose ground truth is re-simulated")
    src.add_argument("--spont", default=None, metavar="PATH",
                     help="experimental: the passive (unforced) recording")
    p.add_argument("--t-obs", dest="t_obs_s", type=float, required=True, metavar="S",
                   help="observation duration, in SECONDS")
    p.add_argument("--forced", action="append", default=None, metavar="PATH[@HZ]",
                   help="experimental: a driven recording. chi mode needs @HZ on every one of them; "
                        "forced mode takes exactly one, without @HZ. Repeatable.")
    p.add_argument("--drive", action="append", default=None, metavar="NAME=VALUE",
                   help="forced mode: the drive the recording was made at, in SI units. One per "
                        "forcing parameter the bounds file declares.")
    p.add_argument("--f0-si", dest="f0_si", type=float, default=None, metavar="N",
                   help="chi mode: the physical drive amplitude every probe was driven at")
    p.add_argument("--n-samples", type=int, default=None, metavar="N",
                   help="posterior draws for the corner plot, the PPC and the summary")
    add_accept_flags(p, other_observation=True)
    p.set_defaults(handler=_infer)
    out["infer"] = p

    p = subparsers.add_parser("tsnpe", help="one TSNPE round: a region around an observation, then "
                                            "training on the prior restricted to it")
    add_config_flags(p)
    add_name_flags(p)
    p.add_argument("--posterior", required=True, metavar="REF",
                   help="the parent posterior the region is measured in")
    p.add_argument("--observation", required=True, metavar="REF",
                   help="the observation the region is drawn around, by name or id")
    p.add_argument("--directions", dest="n_directions", type=int, default=None, metavar="N",
                   help="Fisher directions to truncate (the flat ones are left full width)")
    p.add_argument("--level", type=float, default=None, metavar="Q",
                   help="HPD level of the region; below 0.99 warns, because truncation permanently "
                        "deletes prior support")
    p.add_argument("--num-runs", type=int, default=None, metavar="N")
    p.add_argument("--run-size", dest="run_size_cap", type=int, default=None, metavar="N")
    p.add_argument("--hidden-features", type=int, default=None, metavar="N")
    p.add_argument("--num-transforms", type=int, default=None, metavar="N")
    p.add_argument("--learning-rate", type=float, default=None, metavar="X")
    p.add_argument("--stop-after-epochs", type=int, default=None, metavar="N")
    p.add_argument("--max-epochs", dest="max_num_epochs", type=int, default=None, metavar="N")
    p.add_argument("--checkpoint-every", type=int, default=None, metavar="N",
                   help="batches between checkpoint commits (0 = no cache, nothing resumable)")
    p.add_argument("--resume", choices=("auto", "require", "never"), default=None,
                   help="auto resumes this run's own cache; require refuses when there is none; "
                        "never refuses to resume one (default: auto)")
    p.add_argument("--new-run", action="store_true",
                   help="start a new simulation cache even though a committed one ONE setting away "
                        "exists (a test round and a real one differ only in --num-runs)")
    add_accept_flags(p)
    p.set_defaults(handler=_tsnpe)
    out["tsnpe"] = p
    return out


def _prior(args, store) -> int:
    from core import orchestrator
    cfg, _ = build_cfg(args)
    lp = orchestrator.build_prior(cfg, None, True, name=args.name, note=args.note,
                                  fig_sink=close_sink, store=store, **knobs(args, *PRIOR_KNOBS))
    report(lp)
    return 0


def _train(args, store) -> int:
    from core import orchestrator
    cfg, _ = build_cfg(args)
    prior = orchestrator.build_prior(cfg, args.prior, False, fig_sink=close_sink, store=store)
    lp = orchestrator.build_posterior(cfg, prior, None, True, name=args.name, note=args.note,
                                      fig_sink=close_sink, store=store, new_run=args.new_run,
                                      **knobs(args, *TRAIN_KNOBS))
    report(lp)
    return 0


def _validate(args, store) -> int:
    from core import orchestrator
    cfg, _ = build_cfg(args)
    posterior, prior = load_posterior_and_prior(cfg, args.posterior, accept_from(args), store)
    cal = orchestrator.validate_calibration(cfg, posterior, prior, name=args.name, note=args.note,
                                            fig_sink=close_sink, store=store,
                                            **knobs(args, *VALIDATE_KNOBS))
    report(cal)
    return 0


def _infer(args, store) -> int:
    from core import orchestrator
    # load_gt stays off: simulated_inference injects the truth, so the note about ignored cell values
    # is printed once, by the composition, rather than here and there.
    cfg, _ = build_cfg(args)
    accept = accept_from(args)
    if args.cell and (args.forced or args.drive or args.f0_si is not None):
        # Spec 3.5: --forced/--drive/--f0-si describe a MEASURED recording. --cell re-simulates the
        # cell's own drive, so on that branch they name nothing the command reads -- exactly the
        # silently-ignored flag D6 forbids.
        raise UsageError("--forced, --drive and --f0-si describe measured recordings; --cell "
                         "re-simulates the cell's own drive, so drop them.")
    # A usage error costs no posterior load: recording_set is checked here, before
    # load_posterior_and_prior, so a bad --forced/--drive/--f0-si combination is refused before the
    # load is spent.
    rec = recording_set(cfg, args) if args.spont else None
    posterior, prior = load_posterior_and_prior(cfg, args.posterior, accept, store)
    if args.cell:
        obs, inf = orchestrator.simulated_inference(
            cfg, posterior, args.t_obs_s, cell=args.cell, prior=prior, accept=accept,
            name=args.name, note=args.note, fig_sink=close_sink, store=store,
            **knobs(args, "n_samples"))
    else:
        obs, inf = orchestrator.experimental_inference(
            cfg, posterior, rec, accept=accept, name=args.name, note=args.note,
            fig_sink=close_sink, store=store, **knobs(args, "n_samples"))
    report(obs, inf)
    return 0


def _tsnpe(args, store) -> int:
    from core import orchestrator
    cfg, _ = build_cfg(args)
    posterior, prior = load_posterior_and_prior(cfg, args.posterior, accept_from(args), store)
    observation = store.load_observation(cfg, args.observation)      # re-hashed; mode and width checked
    child = orchestrator.tsnpe_round(cfg, posterior, prior, observation, name=args.name,
                                     note=args.note, fig_sink=close_sink, store=store,
                                     new_run=args.new_run, **knobs(args, *TSNPE_KNOBS))
    report(child)
    return 0
