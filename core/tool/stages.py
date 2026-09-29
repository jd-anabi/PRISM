"""The stage subcommands of ``python -m core``: one thin handler per orchestrator stage.

A handler resolves nothing of its own. Every value flag defaults to None and is forwarded only when it
was given, so the stage's own default -- read from config.py, in one place -- is the default. The one
exception is ``validate``'s ``--seed``: without it the handler draws a seed and passes it, because the
calibration record must name the seed it ran with, so that any calibration the tool writes can be
repeated. The heavy imports live inside the handlers, so building the parser costs no torch.
"""
import random

from core.refusals import default_clause

from .config_args import (UsageError, accept_from, add_accept_flags, add_config_flags, add_name_flags,
                          add_resume_flags, add_training_flags, build_cfg, close_sink, knobs,
                          load_posterior_and_prior, recording_set, report)

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

    text = "build a stability-screened GMM prior from a bounds file"
    p = subparsers.add_parser("prior", help=text, description=text)
    add_config_flags(p)
    add_name_flags(p)
    p.add_argument("--num-iterations", type=int, default=None, metavar="N",
                   help="stability sweep iterations" + default_clause("num_iterations"))
    p.add_argument("--sweep-batch", type=int, default=None, metavar="N",
                   help="candidate parameter sets per sweep iteration; 0 = automatic"
                        + default_clause("sweep_batch"))
    p.add_argument("--max-sets", type=int, default=None, metavar="N",
                   help="accepted sets the sweep stops at" + default_clause("max_sets"))
    p.add_argument("--walk-step", type=float, default=None, metavar="X",
                   help="random-walk step, as a fraction of each box side"
                        + default_clause("walk_step"))
    p.add_argument("--stability-units", type=float, default=None, metavar="X",
                   help="ND time units a candidate must stay bounded for"
                        + default_clause("stability_units"))
    p.add_argument("--min-cluster-size", type=int, default=None, metavar="N",
                   help="HDBSCAN minimum cluster size" + default_clause("min_cluster_size"))
    p.add_argument("--min-samples", type=int, default=None, metavar="N",
                   help="HDBSCAN's min_samples: neighbours a point needs to count as a cluster "
                        "core; larger declares more points noise" + default_clause("min_samples"))
    p.set_defaults(handler=_prior)
    out["prior"] = p

    text = "train an amortized posterior (NPE) on a prior"
    p = subparsers.add_parser("train", help=text, description=text)
    add_config_flags(p)
    add_name_flags(p)
    p.add_argument("--prior", required=True, metavar="REF",
                   help="prior artifact to train on, by name or id")
    add_training_flags(p, fisher=True)
    add_resume_flags(p)
    p.set_defaults(handler=_train)
    out["train"] = p

    text = "SBC + TARP calibration of a posterior (no observation)"
    p = subparsers.add_parser("validate", help=text, description=text)
    add_config_flags(p)
    add_name_flags(p)
    p.add_argument("--posterior", required=True, metavar="REF",
                   help="posterior artifact to calibrate, by name or id")
    p.add_argument("--n-cal", type=int, default=None, metavar="N",
                   help="calibration datasets to simulate" + default_clause("n_cal"))
    p.add_argument("--cal-n-scales", type=int, default=None, metavar="N",
                   help="(t_scale, T_obs) operating points the calibration set is spread over"
                        + default_clause("cal_n_scales"))
    p.add_argument("--posterior-samples", dest="num_posterior_samples", type=int, default=None,
                   metavar="N",
                   help="posterior draws per calibration dataset"
                        + default_clause("num_posterior_samples"))
    p.add_argument("--seed", type=int, default=None, metavar="N",
                   help="the calibration set's random seed: on one device the same seed repeats the "
                        "calibration -- on the CPU bit for bit, on a CUDA card not bitwise -- and never "
                        "replays the stream a training run with that seed used" + default_clause("seed"))
    add_accept_flags(p)
    p.set_defaults(handler=_validate)
    out["validate"] = p

    text = "run a posterior on one observation, simulated or measured"
    p = subparsers.add_parser("infer", help=text, description=text)
    add_config_flags(p)
    add_name_flags(p)
    p.add_argument("--posterior", required=True, metavar="REF",
                   help="posterior artifact to run, by name or id; its own training prior is loaded "
                        "with it")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--cell", default=None, metavar="PATH",
                     help="simulated: the cell file whose ground truth is re-simulated")
    src.add_argument("--spont", default=None, metavar="PATH",
                     help="experimental: the passive (unforced) recording")
    p.add_argument("--t-obs", dest="t_obs_s", type=float, required=True, metavar="S",
                   help="observation length, in seconds: the recordings' length, or the length the "
                        "cell is re-simulated at")
    p.add_argument("--forced", action="append", default=None, metavar="PATH[@HZ]",
                   help="experimental: a driven recording. chi mode needs @HZ on every one of them; "
                        "forced mode takes exactly one, without @HZ. Repeatable.")
    p.add_argument("--drive", action="append", default=None, metavar="NAME=VALUE",
                   help="forced mode: the drive the recording was made at, in SI units. One per "
                        "forcing parameter the bounds file declares.")
    p.add_argument("--f0-si", dest="f0_si", type=float, default=None, metavar="X",
                   help="chi mode: the physical drive amplitude every probe was driven at, in "
                        "newtons")
    p.add_argument("--n-samples", type=int, default=None, metavar="N",
                   help="posterior draws for the corner plot, the posterior predictive check and "
                        "the summary" + default_clause("n_samples"))
    add_accept_flags(p, other_observation=True)
    p.set_defaults(handler=_infer)
    out["infer"] = p

    text = ("one TSNPE round: a region around an observation, then training on the prior "
            "restricted to it")
    p = subparsers.add_parser("tsnpe", help=text, description=text)
    add_config_flags(p)
    add_name_flags(p)
    p.add_argument("--posterior", required=True, metavar="REF",
                   help="parent posterior artifact the region is measured in, by name or id")
    p.add_argument("--observation", required=True, metavar="REF",
                   help="the observation the region is drawn around, by name or id")
    p.add_argument("--directions", dest="n_directions", type=int, default=None, metavar="N",
                   help="Fisher directions to truncate (the flat ones are left full width)"
                        + default_clause("n_directions"))
    p.add_argument("--level", type=float, default=None, metavar="X",
                   help="HPD level of the region; below 0.99 warns, because truncation permanently "
                        "deletes prior support" + default_clause("hpd_level"))
    add_training_flags(p, fisher=False)
    add_resume_flags(p)
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
    # Always a seed, drawn from Python's own stream when none was given: the record names the seed
    # the calibration ran with, so every calibration the tool writes can be repeated.
    seed = args.seed if args.seed is not None else random.randrange(2 ** 31)
    cal = orchestrator.validate_calibration(cfg, posterior, prior, name=args.name, note=args.note,
                                            fig_sink=close_sink, store=store, seed=seed,
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
        # --forced/--drive/--f0-si describe a MEASURED recording. --cell re-simulates the cell's own
        # drive, so on that branch they name nothing the command reads -- a flag that would be
        # silently ignored, which the tool refuses instead.
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
