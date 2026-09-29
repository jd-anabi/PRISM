"""``python -m core {sbc,identifiability,ablation}``: the diagnostics.

Every handler here loads its posterior THROUGH the store's own loader (so a mode, box, order or
width mismatch is refused before anything is spent) and takes the prior from that posterior's
manifest. No diagnostic that reads a posterior takes ``--prior``: an explicitly supplied one could
only ever be the posterior's own or a refusal, and the pairing is recorded in the artifact already.
The one diagnostic with no posterior, ``probes mask`` (core/tool/probes.py), takes the prior it
audits as its input and records it as its parent.
"""
from __future__ import annotations

from core.refusals import default_clause, default_text

from . import config_args
# The helpers below come from config_args, imported by name so a test can monkeypatch them on THIS
# module (tests patch ``tool_diag.load_posterior_and_prior``). Do not re-implement them here: the
# pairing rule (the prior is always the posterior's own) must exist once, in config_args. The accept
# flags and the Accept they build -- add_accept_flags/accept_from, the opt-ins to a non-amortized
# load the tool otherwise refuses -- come from there too, so this module carries no second
# definition of them to drift from config_args's.
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
    p.add_argument("--m", type=int, metavar="N",
                   help="ensemble per perturbation arm" + default_clause("m"))
    p.add_argument("--m-noise", type=int, dest="m_noise", metavar="N",
                   help="ensemble for the single-trajectory feature-noise floor"
                        + default_clause("m_noise"))
    p.add_argument("--rel", type=float, metavar="X",
                   help="perturbation as a fraction of the prior range" + default_clause("rel"))
    p.add_argument("--min-valid", type=float, dest="min_valid", metavar="X",
                   help="valid-member fraction an arm needs before it is used" + default_text("0.5"))
    p.add_argument("--seed", type=int, metavar="N",
                   help="the random seed for the whole run" + default_text("0"))


def _register_identifiability(sub) -> dict:
    text = ("what a posterior measured (rotation), or whether the information is there at all "
            "(laplace, jacobian)")
    p = sub.add_parser("identifiability", help=text, description=text)
    modes = p.add_subparsers(dest="variant", required=True, metavar="{rotation,laplace,jacobian}")

    text = "decompose a trained posterior's Fisher eigenbasis (reads the artifact; simulates nothing)"
    rot = modes.add_parser("rotation", help=text, description=text)
    config_args.add_config_flags(rot)
    add_name_flags(rot)
    rot.add_argument("--posterior", required=True, metavar="REF",
                     help="posterior artifact whose Fisher eigenbasis is decomposed, by name or id")
    rot.add_argument("--n-worst", type=int, dest="n_worst", metavar="N",
                     help="worst directions totalled per parameter" + default_clause("n_worst"))
    rot.add_argument("--top-n", type=int, dest="top_n", metavar="N",
                     help="loadings shown per direction" + default_clause("top_n"))
    add_accept_flags(rot)

    text = ("Laplace marginal SD at the ground truth and at draws from the posterior's own training "
            "prior")
    lap = modes.add_parser("laplace", help=text, description=text)
    config_args.add_config_flags(lap)
    add_name_flags(lap)
    lap.add_argument("--posterior", required=True, metavar="REF",
                     help="posterior artifact whose training prior supplies the evaluation points "
                          "beyond the ground truth, by name or id")
    lap.add_argument("--cell", required=True, metavar="PATH",
                     help="the cell file whose ground truth is the first evaluation point")
    lap.add_argument("--t-obs", type=float, dest="t_obs_s", required=True, metavar="S",
                     help="observation length, in seconds, the metric is measured at")
    lap.add_argument("--n-points", type=int, dest="n_points", metavar="N",
                     help="evaluation points including the ground truth" + default_clause("n_points"))
    _add_probe_flags(lap)
    lap.add_argument("--sd-identified", type=float, dest="sd_identified", metavar="X",
                     help="SD below this counts as identified" + default_text("0.3"))
    add_accept_flags(lap)

    text = ("degeneracy / sloppiness map over the mode's own feature set at the cell's ground truth "
            "(no posterior)")
    jac = modes.add_parser("jacobian", help=text, description=text)
    config_args.add_config_flags(jac)
    add_name_flags(jac)
    jac.add_argument("--cell", required=True, metavar="PATH",
                     help="the cell file whose ground truth the map is measured at")
    jac.add_argument("--t-obs", type=float, dest="t_obs_s", required=True, metavar="S",
                     help="observation length, in seconds, the map is measured at; compare only maps "
                          "made at the same length")
    _add_probe_flags(jac)
    jac.add_argument("--zero-tol", type=float, dest="zero_tol", metavar="X",
                     help="a parameter whose sensitivity, in units of the feature noise, has a norm "
                          "below this carries no local information" + default_text("0.05"))
    jac.add_argument("--noise-eps", type=float, dest="noise_eps", metavar="X",
                     help="a feature channel whose spread across the ensemble is below this fraction "
                          "of its size is treated as dead and zeroed in the map"
                          + default_text("1e-06"))

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
    text = ("which conditioning channels the trained flow can SEE (reads the artifacts; simulates "
            "nothing)")
    p = sub.add_parser("ablation", help=text, description=text)
    config_args.add_config_flags(p)
    add_name_flags(p)
    p.add_argument("--posterior", required=True, metavar="REF",
                   help="posterior artifact whose flow is probed, by name or id; the first rows of "
                        "its own simulation cache set each channel's range")
    p.add_argument("--rows", type=int, metavar="N",
                   help="rows read from the posterior's own simulation cache"
                        + default_clause("rows"))
    p.add_argument("--n-sweep", type=int, dest="n_sweep", metavar="N",
                   help="points per channel sweep" + default_clause("n_sweep"))
    add_accept_flags(p)
    p.set_defaults(handler=_ablation)
    return {"ablation": p}


def register(sub) -> dict:
    """``{name: subparser}`` -- the contract ``build_parser``'s ``p.subcommands.update(...)`` loop
    needs. The identifiability and ablation subcommands add their entries to the dict this returns."""
    text = "SBC repeated N times (--repeats) on one posterior (the run-to-run KS spread)"
    p = sub.add_parser("sbc", help=text, description=text)
    config_args.add_config_flags(p)
    add_name_flags(p)
    p.add_argument("--posterior", required=True, metavar="REF",
                   help="posterior artifact to calibrate repeatedly, by name or id")
    p.add_argument("--repeats", type=int, metavar="N",
                   help="independent SBC runs" + default_clause("repeats"))
    p.add_argument("--n-cal", type=int, dest="n_cal", metavar="N",
                   help="calibration datasets to simulate per repeat" + default_text("2000"))
    p.add_argument("--posterior-samples", type=int, dest="num_posterior_samples", metavar="N",
                   help="posterior draws per calibration dataset" + default_text("1000"))
    p.add_argument("--cal-n-scales", type=int, dest="cal_n_scales", metavar="N",
                   help="(t_scale, T_obs) operating points the calibration set is spread over"
                        + default_clause("cal_n_scales"))
    p.add_argument("--chi-k-fixed", type=int, dest="chi_k_fixed", metavar="K",
                   help="hold the chi probe count at K instead of pooling over the training mixture; "
                        "chi mode only. Run once per count and once pooled, then compare"
                        + default_text("pooled over the training mixture"))
    p.add_argument("--seed", type=int, metavar="N",
                   help="the random seed for the whole run; repeat r runs at seed + r"
                        + default_text("0"))
    add_accept_flags(p)
    p.set_defaults(handler=_sbc)
    return {"sbc": p, **_register_identifiability(sub), **_register_ablation(sub)}
