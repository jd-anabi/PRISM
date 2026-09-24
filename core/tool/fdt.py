"""``python -m core fdt`` and ``python -m core crossval``: the two FDT analyses as flags.

Each subcommand builds the config ``core/cli.py`` builds prompt-free (``make_fdt_config`` /
``make_param_sweep_config``, the same functions the GUI panels call) and hands it to the same
pipeline. Since piece 5 both write an ``fdt`` artifact RECORD -- progressively, so an interrupted
run keeps its folder marked unfinished -- into ``--store-root``, or into the PRISM_ARTIFACTS root
when that flag is not given.

Neither takes the SBI config flags: no prior is built and no training runs, so there is no
observation mode and no ``--bounds``. A bounds file is not ABSENT, though: ``cli.parse_cell``
resolves one for the cell (``cli.resolve_bounds_for_cell`` -- the same-named sibling, else the
model folder's master), it defines the parameter set and its order on the decoupled path, and the
record names it by path and SHA-256 like every other kind. There is no ``--device`` either -- both
config builders force the CPU, where the sequential SDE loop at M ~ 256 is about 3.4x faster than
on the card.
"""
import argparse
import math

from core.refusals import Refusal

from .config_args import UsageError, knobs, model_from_path

FDT_EPILOG = """\
The model comes from the cell's parent folder (Resources/Cells/<model>/), or from --model. A model
FDT cannot run is refused with the reason: FDT drives the observable itself, so a user model needs
additive, non-zero observable noise and no intrinsic forcing.

The run writes one `fdt` record: its figures (PSD, chi components, T_eff/T, the spontaneous
trajectory), its numbers in data.h5, its settings and seed, and its log. Sanity checks run first
unless --skip-sanity; --no-production stops after them.
"""

CROSSVAL_EPILOG = """\
Two sweeps probe FDT restoration on the Nadrowski model (the model is fixed): the S sweep holds
T_a/T = 1 and varies S (FDT restored as S -> 0), the T sweep holds S = 0 and varies T_a/T (restored
as T_a/T -> 1). Each grid is MIN MAX N, and N must be a whole number of at least 2.

--preset drives the resolution levers the flags do not expose (freq_bounds, T_obs_periods,
psd_T_obs_nd) and supplies the defaults for --n-freqs and --ensemble-m. Each sweep writes its OWN
`fdt` record -- its data.h5, its 3-D plot and its point counts -- so the S sweep is a finished,
readable answer before the T sweep starts.
"""

# I2, fix round 1, rewritten for piece 5 (E2): fdt/crossval keep no cache and take no --resume, so
# main's generic Ctrl-C advice -- "if a [checkpoint] line above says batches were saved, ...
# --resume require" -- is simply wrong for them. What changed is WHAT SURVIVES. These runs write
# their record PROGRESSIVELY, so an interrupt leaves a real artifact behind -- its directory, its
# manifest marked unfinished, its data file and its figures, and the run's own log.txt -- listed
# like any other and removable by id. Each note therefore says where to find that record and how
# to clear it, and repeats that re-running starts over: an unfinished record is evidence, not a
# resume point (spec §1.3, "Resuming an interrupted sweep": not asked for, and none is added).
#
# A note is fixed text, set once through set_defaults, so it cannot carry the record's id. Ruling
# F20: the handler prints each record's id and directory -- the `writing record` line -- the moment
# store.create mints it, which is before anything can be interrupted, and the note points back at
# that line. And because the `artifacts` family reads only PRISM_ARTIFACTS (it has no --store-root,
# core/tool/browse.py), a run given --store-root must be followed by pointing the variable there, or
# the listing looks in the wrong root and finds nothing.
_STORE_ROOT_ADVICE = ("If this run was given --store-root, set PRISM_ARTIFACTS to that root first: "
                      "the `artifacts` commands read only the environment.")
FDT_INTERRUPT_NOTE = (
    "the record named by the `writing record` line above is KEPT and marked unfinished, holding "
    "everything measured so far; `python -m core artifacts list fdt` lists it and `python -m core "
    f"artifacts rm fdt <id>` removes it. {_STORE_ROOT_ADVICE} fdt keeps no cache and nothing "
    "resumes, so re-running the same command starts the analysis from scratch.")
# Two records, three possible states (spec §4.1): the sweeps run S first, then T, each entering its
# own record only when it starts. So an interrupt during S leaves S's record unfinished and T's never
# written, and one during T leaves S's finished (or, had S measured nothing, unfinished) beside T's
# unfinished one. The note says exactly that rather than promising two records on disk.
CROSSVAL_INTERRUPT_NOTE = (
    "each sweep writes a record of its own, named by the two `writing record` lines above: a sweep "
    "that had finished keeps its finished record, the one this run was in the middle of is KEPT and "
    "marked unfinished, holding its data file and every operating point measured so far, and a "
    "sweep that had not started left nothing on disk. `python -m core artifacts list fdt` lists "
    f"them and `python -m core artifacts rm fdt <id>` removes one. {_STORE_ROOT_ADVICE} crossval "
    "keeps no cache and nothing resumes, so re-running the same command starts the study from "
    "scratch.")


def model_for_cell(args) -> str:
    """``--model``, else the cell's parent folder upper-cased (the ``Cells/<model>/`` layout). The
    SBI subcommands take the model from the BOUNDS folder, which they are given; these two are given
    a CELL and resolve its bounds file from it (``cli.resolve_bounds_for_cell``), so the folder is
    what names the model here. A thin wrapper over ``config_args.model_from_path`` -- kept as its
    own function because the interface and the test suite name it ``fdt.model_for_cell``."""
    return model_from_path(args.model, args.cell)


def _grid(flag: str, triple) -> tuple:
    """``MIN MAX N`` -> ``(float, float, int)`` for ``np.linspace``, which refuses a float ``num``.

    ``N`` must be a FINITE whole number, checked BEFORE any ``int(n)``: ``int(float("inf"))`` raises
    ``OverflowError`` and ``int(float("nan"))`` raises ``ValueError``, neither caught by ``main``'s
    ``(ValueError, FileNotFoundError)`` usage-error net for an ``OverflowError``, and even the
    ``ValueError`` from ``nan`` would print as a bug's traceback rather than naming ``--s-grid``/
    ``--t-grid``. A malformed grid is a usage error (exit 2) regardless of which of the three ways
    it is malformed.
    """
    lo, hi, n = triple
    if not math.isfinite(n) or not float(n).is_integer():
        raise UsageError(f"{flag}: the point count must be a finite whole number, got {n!r}")
    if int(n) < 2:
        raise UsageError(f"{flag}: a sweep needs at least 2 points, got {int(n)}")
    return float(lo), float(hi), int(n)


def _add_store_root(p) -> None:
    """``--store-root`` for an analysis that writes an ``fdt`` record (E11, spec §6.1).

    DECLARING it changes nothing else by itself. ``main`` used to key three behaviours on whether a
    subcommand declared this flag -- a throwaway ``mkdtemp`` root, the removal of an empty
    auto-created root, and which Ctrl-C advice is printed -- which was ``smoke`` alone. All three are
    wrong here: an unnamed root must be the operator's own PRISM_ARTIFACTS, a failed run must never
    ``rmdir`` that root, and ``_smoke_interrupt_advice`` reads ``args.prior``/``args.save``, which
    these two do not have. So ``main`` now keys them on ``args.temp_store_root``, which only
    ``smoke`` sets.
    """
    p.add_argument("--store-root", dest="store_root", default=None,
                   help="the artifact store this run writes its record into (default: the "
                        "PRISM_ARTIFACTS root, like every subcommand but `smoke`)")


def _add_fdt_knobs(p) -> None:
    """The four resolution knobs both subcommands share, and the seed. Each defaults to None and
    travels only when set; the dest is the builder's keyword, capital M and F0 included. ``--seed``
    is E7's command-line half (P79): a seed a record carries must be one the operator can supply
    back. Unset, the run draws one from [0, 2**31) and records it (P12, P13)."""
    p.add_argument("--n-freqs", dest="n_freqs", type=int, default=None,
                   help="drive frequencies in Campaign 2")
    p.add_argument("--ensemble-m", dest="ensemble_M", type=int, default=None,
                   help="trajectories per frequency")
    p.add_argument("--freqs-per-batch", dest="freqs_per_batch", type=int, default=None,
                   help="frequencies packed into one simulator call")
    p.add_argument("--f0", dest="F0", type=float, default=None,
                   help="ND forcing amplitude (keep it inside the linear regime)")
    p.add_argument("--seed", type=int, default=None,
                   help="the run's random seed, a whole number from 0 (default: draw one; either "
                        "way the record carries it, so the run can be repeated)")


def register(subparsers):
    fdt = subparsers.add_parser(
        "fdt", help="effective-temperature (FDT) analysis for one cell",
        epilog=FDT_EPILOG, formatter_class=argparse.RawDescriptionHelpFormatter)
    fdt.add_argument("--cell", required=True, help="the cell file to analyse")
    fdt.add_argument("--model", default=None,
                     help="model name (default: the cell's parent folder)")
    _add_store_root(fdt)
    _add_fdt_knobs(fdt)
    fdt.add_argument("--skip-sanity", dest="skip_sanity", action="store_true",
                     help="skip the sanity checks and go straight to the production sweep")
    fdt.add_argument("--no-production", dest="no_production", action="store_true",
                     help="stop after the sanity checks")
    fdt.set_defaults(handler=run_fdt_cmd, interrupt_note=FDT_INTERRUPT_NOTE)

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
    _add_store_root(cv)
    _add_fdt_knobs(cv)
    cv.set_defaults(handler=run_crossval, interrupt_note=CROSSVAL_INTERRUPT_NOTE)
    return {"fdt": fdt, "crossval": cv}


def run_fdt_cmd(args, store):
    """One ``fdt`` record per run. The record is created HERE -- ``store.create`` mints the id and runs
    ``assert_name_free`` before anything is spent -- and ENTERED by ``run_fdt`` (spec §1.2)."""
    from core import cli, registry
    from core.FDT import fdt_pipeline
    if args.skip_sanity and args.no_production:
        # run_fdt reads confirm_production only on the sanity branch, so this pair used to run the
        # full production sweep without a word.
        raise UsageError("--skip-sanity and --no-production together run nothing: --skip-sanity goes "
                         "straight to the production sweep, and --no-production stops after the "
                         "sanity checks. Drop one of them.")
    model = model_for_cell(args)
    ok, reason = registry.fdt_support(model)
    if not ok:
        # M1, fix round 1, now as V3's one line (spec §6.2): a Refusal with a field key, so the
        # ladder prints `refused: <reason> (<where the name came from>.) (--model)` instead of
        # `refused: ValueError: ... [raised at fdt.py:139]`. The FLAG is fix_sentence's to add, from
        # core/tool/fields.py; what the hint still carries is what the flag cannot -- WHERE the name
        # came from, because --model and a cell's parent folder are two different mistakes.
        # cli.make_fdt_config refuses the same model since Task 10, with fdt_support's reason alone;
        # this gate stays first because only a front end knows where the name came from.
        hint = ("--model named it" if args.model else
                "the cell's parent folder named it; pass --model to override that")
        raise Refusal(f"{reason} ({hint}.)", field="model")
    cfg = cli.make_fdt_config(model, registry.state_dep_drift(model), args.cell,
                              **knobs(args, "n_freqs", "ensemble_M", "freqs_per_batch", "F0", "seed"))
    writer = store.create("fdt", cfg)
    # F20: on screen BEFORE anything is spent, so a Ctrl-C finds the record's id already printed --
    # FDT_INTERRUPT_NOTE points back at this line, being fixed text that cannot carry the id itself.
    print(f"[prism fdt] writing record {writer.id} at {writer.dir}", flush=True)
    rec = fdt_pipeline.run_fdt(cfg, skip_sanity=args.skip_sanity,
                               confirm_production=not args.no_production, writer=writer,
                               seed=args.seed)
    print(f"[prism fdt] record {rec.id} at {rec.path}")


def run_crossval(args, store):
    """Two ``fdt`` records per study, one per swept parameter (spec §4.1). Both are created HERE --
    ``store.create`` mints each id and runs ``assert_name_free`` before anything is spent -- and each
    is ENTERED by its own sweep (spec §1.2)."""
    from core import cli
    from core.FDT import cross_validation
    preset = dict(cli.SWEEP_PRESETS[args.preset])
    cfg, s_grid, temp_grid = cli.make_param_sweep_config(
        args.cell, preset=preset, preset_name=args.preset,
        s_spec=_grid("--s-grid", args.s_grid), t_spec=_grid("--t-grid", args.t_grid),
        n_freqs=preset["n_freqs"] if args.n_freqs is None else args.n_freqs,
        ensemble_M=preset["ensemble_M"] if args.ensemble_M is None else args.ensemble_M,
        **knobs(args, "freqs_per_batch", "F0", "seed"))
    writers = {"s": store.create("fdt", cfg), "temp": store.create("fdt", cfg)}
    # F20: one line per record, before either sweep spends anything -- CROSSVAL_INTERRUPT_NOTE points
    # back at these two lines, being fixed text that cannot carry the ids itself.
    for key, label in (("s", "S"), ("temp", "T_a/T")):
        print(f"[prism crossval] writing record {writers[key].id} at {writers[key].dir} "
              f"(the {label} sweep)", flush=True)
    recs = cross_validation.run_param_study_cli(cfg, s_grid=s_grid, t_grid=temp_grid,
                                                writers=writers, seed=args.seed)
    for rec in recs:
        print(f"[prism crossval] sweep record {rec.id} at {rec.path}")
