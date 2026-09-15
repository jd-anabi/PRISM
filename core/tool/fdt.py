"""``python -m core fdt`` and ``python -m core crossval``: the prompt CLI's two FDT modes as flags.

No new science, and no hardening: each subcommand builds the config ``core/cli.py`` already builds
prompt-free (``make_fdt_config`` / ``make_param_sweep_config``, the same functions the GUI panels
call) and hands it to the same pipeline. FDT/CrossVal hardening, and wrapping their outputs in the
artifact store, is piece 5's work -- until then FDT saves under ``<artifacts root>/fdt`` and the
sweep study under ``<artifacts root>/crossval``, both moved by PRISM_ARTIFACTS and neither touching
the store the tool opens.

Neither takes the SBI config flags: no prior is built and no training runs, so there is no bounds
file and no observation mode. There is no ``--device`` either -- both config builders force the CPU,
where the sequential SDE loop at M ~ 256 is about 3.4x faster than on the card.
"""
import argparse
from pathlib import Path

from .config_args import knobs

FDT_EPILOG = """\
The model comes from the cell's parent folder (Resources/Cells/<model>/), or from --model. A model
FDT cannot run is refused with the reason: FDT drives the observable itself, so a user model needs
additive, non-zero observable noise and no intrinsic forcing.

Plots (PSD, chi components, T_eff/T, the spontaneous trajectory) are written to
<artifacts root>/fdt, timestamped. Sanity checks run first unless --skip-sanity; --no-production
stops after them.
"""

CROSSVAL_EPILOG = """\
Two sweeps probe FDT restoration on the Nadrowski model (the model is fixed): the S sweep holds
T_a/T = 1 and varies S (FDT restored as S -> 0), the T sweep holds S = 0 and varies T_a/T (restored
as T_a/T -> 1). Each grid is MIN MAX N, and N must be a whole number of at least 2.

--preset drives the resolution levers the flags do not expose (freq_bounds, T_obs_periods,
psd_T_obs_nd) and supplies the defaults for --n-freqs and --ensemble-m. One HDF5 per sweep plus a
3-D plot each go to <artifacts root>/crossval; the S-sweep plot is saved at the study's midpoint,
so a long run gives you half its answer early.
"""


def model_for_cell(args) -> str:
    """``--model``, else the cell's parent folder upper-cased (the ``Cells/<model>/`` layout). The
    SBI subcommands take the model from the BOUNDS folder; these two have no bounds file."""
    return str(args.model or Path(args.cell).parent.name).upper()


def _grid(flag: str, triple) -> tuple:
    """``MIN MAX N`` -> ``(float, float, int)`` for ``np.linspace``, which refuses a float ``num``."""
    from core.tool import UsageError
    lo, hi, n = triple
    if float(n) != int(n):
        raise UsageError(f"{flag}: the point count must be a whole number, got {n!r}")
    if int(n) < 2:
        raise UsageError(f"{flag}: a sweep needs at least 2 points, got {int(n)}")
    return float(lo), float(hi), int(n)


def _add_fdt_knobs(p) -> None:
    """The four resolution knobs both subcommands share. Each defaults to None and travels only when
    set; the dest is the builder's keyword, capital M and F0 included."""
    p.add_argument("--n-freqs", dest="n_freqs", type=int, default=None,
                   help="drive frequencies in Campaign 2")
    p.add_argument("--ensemble-m", dest="ensemble_M", type=int, default=None,
                   help="trajectories per frequency")
    p.add_argument("--freqs-per-batch", dest="freqs_per_batch", type=int, default=None,
                   help="frequencies packed into one simulator call")
    p.add_argument("--f0", dest="F0", type=float, default=None,
                   help="ND forcing amplitude (keep it inside the linear regime)")


def register(subparsers):
    fdt = subparsers.add_parser(
        "fdt", help="effective-temperature (FDT) analysis for one cell",
        epilog=FDT_EPILOG, formatter_class=argparse.RawDescriptionHelpFormatter)
    fdt.add_argument("--cell", required=True, help="the cell file to analyse")
    fdt.add_argument("--model", default=None,
                     help="model name (default: the cell's parent folder)")
    _add_fdt_knobs(fdt)
    fdt.add_argument("--skip-sanity", dest="skip_sanity", action="store_true",
                     help="skip the sanity checks and go straight to the production sweep")
    fdt.add_argument("--no-production", dest="no_production", action="store_true",
                     help="stop after the sanity checks (ignored with --skip-sanity)")
    fdt.set_defaults(handler=run_fdt_cmd)

    cv = subparsers.add_parser(
        "crossval", help="the FDT parameter-sweep study (S and T_a/T), NADROWSKI only",
        epilog=CROSSVAL_EPILOG, formatter_class=argparse.RawDescriptionHelpFormatter)
    cv.add_argument("--cell", required=True, help="the NWK cell file the sweeps start from")
    cv.add_argument("--preset", choices=("exploratory", "production"), default="exploratory",
                    help="resolution preset (default: exploratory)")
    cv.add_argument("--s-grid", dest="s_grid", nargs=3, type=float, required=True,
                    metavar=("MIN", "MAX", "N"), help="the S sweep grid")
    cv.add_argument("--t-grid", dest="t_grid", nargs=3, type=float, required=True,
                    metavar=("MIN", "MAX", "N"), help="the T_a/T sweep grid")
    _add_fdt_knobs(cv)
    cv.set_defaults(handler=run_crossval)
    return {"fdt": fdt, "crossval": cv}


def run_fdt_cmd(args, store):
    """``store`` is unused: FDT writes plots, not artifacts (piece 5 wraps it)."""
    from core import cli, config, registry
    from core.FDT import fdt_pipeline
    model = model_for_cell(args)
    ok, reason = registry.fdt_support(model)
    if not ok:
        raise ValueError(reason)
    cfg = cli.make_fdt_config(model, registry.state_dep_drift(model), args.cell,
                              **knobs(args, "n_freqs", "ensemble_M", "freqs_per_batch", "F0"))
    fdt_pipeline.run_fdt(cfg, skip_sanity=args.skip_sanity,
                         confirm_production=not args.no_production)
    print(f"[prism fdt] plots under {config.artifacts_root() / 'fdt'}")


def run_crossval(args, store):
    """``store`` is unused: the sweep study writes HDF5 and plots, not artifacts (piece 5)."""
    from core import cli, config
    from core.FDT import cross_validation
    preset = dict(cli.SWEEP_PRESETS[args.preset])
    cfg, s_grid, temp_grid = cli.make_param_sweep_config(
        args.cell, preset=preset,
        s_spec=_grid("--s-grid", args.s_grid), t_spec=_grid("--t-grid", args.t_grid),
        n_freqs=preset["n_freqs"] if args.n_freqs is None else args.n_freqs,
        ensemble_M=preset["ensemble_M"] if args.ensemble_M is None else args.ensemble_M,
        **knobs(args, "freqs_per_batch", "F0"))
    s_path, t_path = cross_validation.run_param_study_cli(cfg, s_grid, temp_grid)
    print(f"[prism crossval] S sweep: {s_path}")
    print(f"[prism crossval] T sweep: {t_path}")
    print(f"[prism crossval] plots under {config.artifacts_root() / 'crossval'}")
