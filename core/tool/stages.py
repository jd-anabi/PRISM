"""The stage subcommands of ``python -m core``: one thin handler per orchestrator stage.

A handler resolves nothing of its own. Every value flag defaults to None and is forwarded only when it
was given, so the stage's own default -- read from config.py, in one place -- is the default. The heavy
imports live inside the handlers, so building the parser costs no torch.
"""
from .config_args import add_config_flags, add_name_flags, build_cfg, close_sink, knobs, report

PRIOR_KNOBS = ("num_iterations", "sweep_batch", "max_sets", "walk_step", "stability_units",
               "min_cluster_size", "min_samples")
TRAIN_KNOBS = ("num_runs", "run_size_cap", "hidden_features", "num_transforms", "learning_rate",
               "stop_after_epochs", "max_num_epochs", "fisher_m", "fisher_dz", "fisher_points",
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
