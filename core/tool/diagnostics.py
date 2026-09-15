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
    return {"sbc": p}
