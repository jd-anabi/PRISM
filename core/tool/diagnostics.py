"""``python -m core {sbc,identifiability,ablation}``: the diagnostics.

Every handler here loads its posterior THROUGH the store's own loader (so a mode, box, order or
width mismatch is refused before anything is spent) and takes the prior from that posterior's
manifest. There is no ``--prior`` on a diagnostic: an explicitly supplied one could only ever be the
posterior's own or a refusal, and the pairing is recorded in the artifact already.
"""
from __future__ import annotations

from . import config_args
# The four helpers below are Task 11/12's, imported by name so a test can monkeypatch them on THIS
# module (tests patch ``tool_diag.load_posterior_and_prior``). Do not re-implement them here: the
# pairing rule (the prior is always the posterior's own) must exist once, in config_args. The accept
# flag and the Accept it builds are T12's D8 pair too -- add_accept_flags/accept_from -- so this
# module carries no second definition of them to drift from config_args's.
from .config_args import (add_accept_flags, add_name_flags, accept_from, knobs,
                          load_posterior_and_prior, report)


def _sbc(args, store) -> None:
    import core.diagnostics as diag
    cfg, _ = config_args.build_cfg(args, load_gt=False)
    posterior, prior = load_posterior_and_prior(cfg, args.posterior, accept_from(args), store)
    return report(diag.sbc_repeats(
        cfg, posterior, prior, name=args.name, note=args.note,
        fig_sink=config_args.close_sink, store=store,
        **knobs(args, "repeats", "n_cal", "num_posterior_samples", "cal_n_scales", "chi_k_fixed", "seed")))


def _identifiability(args, store) -> None:
    import core.diagnostics as diag
    needs_gt = args.variant in ("laplace", "jacobian")
    cfg, _ = config_args.build_cfg(args, load_gt=needs_gt)
    common = dict(name=args.name, note=args.note, fig_sink=config_args.close_sink, store=store)
    if args.variant == "rotation":
        posterior, _prior = load_posterior_and_prior(cfg, args.posterior, accept_from(args), store)
        return report(diag.identifiability_rotation(
            cfg, posterior, **common, **knobs(args, "n_worst", "top_n")))
    if args.variant == "laplace":
        posterior, _prior = load_posterior_and_prior(cfg, args.posterior, accept_from(args), store)
        return report(diag.identifiability_laplace(
            cfg, posterior, **common,
            **knobs(args, "n_points", "m", "m_noise", "rel", "min_valid", "sd_identified",
                  "t_obs_s", "seed")))
    return report(diag.identifiability_jacobian(
        cfg, **common,
        **knobs(args, "m", "m_noise", "rel", "min_valid", "zero_tol", "noise_eps", "t_obs_s", "seed")))


def _add_probe_flags(p) -> None:
    p.add_argument("--m", type=int, help="ensemble per perturbation arm (stage default 32)")
    p.add_argument("--m-noise", type=int, dest="m_noise",
                   help="ensemble for the single-trajectory feature-noise floor (stage default 128)")
    p.add_argument("--rel", type=float, help="perturbation as a fraction of the prior range")
    p.add_argument("--min-valid", type=float, dest="min_valid",
                   help="valid-member fraction an arm needs before it is used")
    p.add_argument("--seed", type=int, help="seed for the whole measurement (stage default 0)")


def _register_identifiability(sub) -> dict:
    p = sub.add_parser("identifiability",
                       help="what a posterior measured (rotation), or whether the information is "
                            "there at all (laplace, jacobian)")
    modes = p.add_subparsers(dest="variant", required=True, metavar="{rotation,laplace,jacobian}")

    rot = modes.add_parser("rotation", help="decompose a trained posterior's Fisher eigenbasis "
                                            "(reads the artifact; simulates nothing)")
    config_args.add_config_flags(rot)
    rot.add_argument("--posterior", required=True, metavar="REF")
    rot.add_argument("--n-worst", type=int, dest="n_worst",
                     help="worst directions totalled per parameter (stage default 3)")
    rot.add_argument("--top-n", type=int, dest="top_n",
                     help="loadings shown per direction (stage default 4)")
    add_accept_flags(rot)
    add_name_flags(rot)

    lap = modes.add_parser("laplace", help="Laplace marginal SD at the ground truth and at draws "
                                           "from the posterior's own training prior")
    config_args.add_config_flags(lap)
    lap.add_argument("--posterior", required=True, metavar="REF")
    lap.add_argument("--cell", required=True, metavar="PATH")
    lap.add_argument("--t-obs", type=float, dest="t_obs_s", required=True, metavar="S")
    lap.add_argument("--n-points", type=int, dest="n_points",
                     help="evaluation points including the ground truth (stage default 6)")
    _add_probe_flags(lap)
    lap.add_argument("--sd-identified", type=float, dest="sd_identified",
                     help="SD below this counts as identified (stage default 0.3)")
    add_accept_flags(lap)
    add_name_flags(lap)

    jac = modes.add_parser("jacobian", help="degeneracy / sloppiness map over the mode's own feature "
                                            "set at the cell's ground truth (no posterior)")
    config_args.add_config_flags(jac)
    jac.add_argument("--cell", required=True, metavar="PATH")
    jac.add_argument("--t-obs", type=float, dest="t_obs_s", required=True, metavar="S")
    _add_probe_flags(jac)
    jac.add_argument("--zero-tol", type=float, dest="zero_tol",
                     help="||g||_std below this is 'no local info' (stage default 0.05)")
    jac.add_argument("--noise-eps", type=float, dest="noise_eps",
                     help="fnoise/|feature| below this is a dead channel (stage default 1e-6)")
    add_name_flags(jac)

    for mode in (rot, lap, jac):
        mode.set_defaults(handler=_identifiability)
    return {"identifiability": p}


def _ablation(args, store) -> None:
    import core.diagnostics as diag
    cfg, _ = config_args.build_cfg(args, load_gt=False)
    posterior, _prior = load_posterior_and_prior(cfg, args.posterior, accept_from(args), store)
    return report(diag.channel_ablation(
        cfg, posterior, name=args.name, note=args.note, fig_sink=config_args.close_sink,
        store=store, **knobs(args, "rows", "n_sweep")))


def _register_ablation(sub) -> dict:
    p = sub.add_parser("ablation", help="which conditioning channels the trained flow can SEE "
                                        "(reads the artifacts; simulates nothing)")
    config_args.add_config_flags(p)
    p.add_argument("--posterior", required=True, metavar="REF")
    p.add_argument("--rows", type=int,
                   help="rows read from the posterior's own simulation cache (stage default 200000)")
    p.add_argument("--n-sweep", type=int, dest="n_sweep",
                   help="points per channel sweep (stage default 33)")
    add_accept_flags(p)
    add_name_flags(p)
    p.set_defaults(handler=_ablation)
    return {"ablation": p}


def register(sub) -> dict:
    """``{name: subparser}`` -- the contract ``build_parser``'s ``p.subcommands.update(...)`` loop
    needs. T17 and T18 append their entries to the dict this returns."""
    p = sub.add_parser("sbc", help="SBC repeated K times on one posterior (the run-to-run KS spread)")
    config_args.add_config_flags(p)
    p.add_argument("--posterior", required=True, metavar="REF", help="posterior name or id")
    p.add_argument("--repeats", type=int, help="independent SBC runs (stage default 10)")
    p.add_argument("--n-cal", type=int, dest="n_cal",
                   help="calibration datasets PER REPEAT (stage default 2000)")
    p.add_argument("--posterior-samples", type=int, dest="num_posterior_samples",
                   help="posterior draws per calibration point (stage default 1000)")
    p.add_argument("--cal-n-scales", type=int, dest="cal_n_scales",
                   help="(t_scale, T) operating points the calibration set is spread over")
    p.add_argument("--chi-k-fixed", type=int, dest="chi_k_fixed",
                   help="hold the chi probe COUNT here instead of pooling over the training mixture; "
                        "chi mode only. Run once per stratum AND once pooled, then compare")
    p.add_argument("--seed", type=int, help="base seed; repeat r runs at seed+r (stage default 0)")
    add_accept_flags(p)
    add_name_flags(p)
    p.set_defaults(handler=_sbc)
    return {"sbc": p, **_register_identifiability(sub), **_register_ablation(sub)}
