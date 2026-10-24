"""``python -m core smoke``: every pipeline stage end to end, at tiny sizes, non-interactively.

WHY THIS EXISTS. The suites pin contracts on stubs and tiny synthetic inputs. This runs the REAL
chain on the REAL bounds/cell files -- stability-screened prior build, NPE training, SBC/TARP
calibration, PPC and the eye test -- so plumbing that only appears when the stages are wired
together (mode widths, the rotation's round trip through the manifest, the chi block's shape
agreeing with the network's input layer) fails here, in minutes, instead of hours into a multi-day
run. It is the GPU gate: every suite runs on the CPU, and a tensor on the wrong device is invisible
until the card sees it.

WHAT TO WATCH, beyond "it finished":

  * ``[chi] masked probes: M of N (P%) over n batches, <scope>`` -- the masked-probe number to read:
    one information line when the training rows are done, with the per-batch spread beside it. With
    --checkpoint it is the run total over every committed batch of the simulation cache, resumed
    batches included, and the calibration draw's own line, scoped "this process only (no simulation
    cache)", is not the training figure. Without --checkpoint, smoke's default, there is no cache:
    both lines are scoped "this process only (no simulation cache)", and the training line is the
    first of them. The COUNT matters, not the presence. Some masking is by design (a probe under
    CHI_MIN_CYCLES on a short recording). The reference is ~37 % of TRAINING probes. The per-call
    ``chi: N/M probes masked`` warnings remain, one per probe block, but the run total replaces
    summing them by hand.
    THE RUN-LEVEL FIGURE IS NOISY, AND BY MUCH MORE THAN IT LOOKS: all rows in a batch share one
    (t_scale, T) stratum AND one probe set, so the effective n is the BATCH COUNT, not the probe
    count. Measured per-batch fractions span 13.5-57.3 % (SD 12.2 pp over 12 batches). Compare the
    MEAN of a few runs against ~37 %, and treat a single run within roughly +/-12 pp as
    uninformative. The PPC's fraction is much higher by design -- it reads the POSTERIOR, not the
    probe machinery.
  * the out-of-distribution and T_obs warnings, which ``simulated_inference`` emits as
    PreflightWarnings (the GUI shows them at warning severity; here they go to stderr).
  * the mode banner. A width mismatch between the config and the trained net is exactly what this
    run exists to catch before the long one.
  * ``bounds=`` AND ``rescale order:`` in the config banner, TOGETHER. ``--bounds`` is required on
    every SBI subcommand precisely because of this pairing: bounds resolution prefers a same-named
    SIBLING over the shared ``master.txt``, so a cell named ``master_spont`` used to resolve the
    12-dim SPONTANEOUS box while the retrain uses the 13-dim one -- differing only in ``f_scale``,
    the retrain's headline hypothesis. Under chi that is silent: both boxes report ``mode=chi`` and
    both build the same conditioning width, because the chi block is a function of CHI_K_PAD, not
    of the parameter set.

THIS IS NOT A CALIBRATION MEASUREMENT. SBC at these sizes has no power -- cal_n_scales is t_scale's
effective sample size and it is tiny here. A flat rank histogram from this run means nothing at all;
only a CRASH means anything.

Every knob is an ARGUMENT: this module rebinds no module constant (``scripts/smoke_train.py``
rebound four on ``orchestrator`` and leaked a default store; both are gone). The whole stage
sequence runs inside one ``seeded(seed, device)`` context, which the card test step this command
calls (``core.diagnostics.card_test``) opens -- one seed, streams running on, as the script did:
per-stage seeding would replay the training strata in the calibration set.
Seeded runs are NOT bitwise-reproducible on CUDA or across devices.
"""
import argparse
import time

STAGES = ("prior", "posterior", "validate", "infer")

EPILOG = """\
What to watch in its output, and why it shows that the chain runs but not that
it is calibrated: docs/guide/command-line.md#smoke
A seeded run is not bitwise-reproducible on CUDA or across devices.
"""


def _stages_type(value: str) -> str:
    """argparse ``type=`` for ``--stages``: validated at PARSE TIME, so an unknown stage is an
    argparse error before ``main`` ever resolves a store root or creates a temp directory -- and
    before ``registry.load_user_models()``, so it costs no torch import either.
    Returns the original string unchanged; ``run_smoke`` still does its own split (kept as a second,
    defensive check for any caller that reaches it without going through argparse)."""
    stages = [s.strip() for s in value.split(",") if s.strip()]
    unknown = [s for s in stages if s not in STAGES]
    if unknown:
        raise argparse.ArgumentTypeError(
            f"--stages: unknown stage(s) {unknown}; choose from {list(STAGES)}")
    return value


def register(subparsers):
    """The ``smoke`` subcommand. Its defaults are the smoke gate's small sizes, and they are the ONE
    place in the tool that restates a literal (every other subcommand leaves defaults to its stage)."""
    from core.refusals import default_clause, default_text
    from core.tool.config_args import add_config_flags, add_resume_flags
    text = ("every stage end to end at tiny sizes; run it on the card after changing code\n"
            "that moves tensors")
    p = subparsers.add_parser(
        "smoke", help=text, description=text,
        epilog=EPILOG, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_flags(p)
    p.add_argument("--cell", required=True, metavar="PATH",
                   help="the cell file whose ground truth the infer stage simulates")
    p.add_argument("--t-obs", dest="t_obs_s", type=float, default=None, metavar="S",
                   help="observation length, in seconds, of the observation the infer stage "
                        "simulates" + default_text("1.0"))
    p.add_argument("--seed", type=int, default=0, metavar="N",
                   help="the random seed for the whole run" + default_text("%(default)s"))
    p.add_argument("--stages", default=",".join(STAGES), type=_stages_type,
                   metavar="STAGE[,STAGE...]",
                   help=f"the stages to run, a comma-separated subset of {','.join(STAGES)}"
                        + default_text("%(default)s"))
    p.add_argument("--num-runs", dest="num_runs", type=int, default=4, metavar="N",
                   help="training batches to simulate" + default_text("%(default)s"))
    p.add_argument("--run-size", dest="run_size_cap", type=int, default=32, metavar="N",
                   help="ceiling on simulations per training batch; the prior sweep keeps the "
                        "hardware batch" + default_text("%(default)s"))
    p.add_argument("--n-cal", dest="n_cal", type=int, default=40, metavar="N",
                   help="calibration datasets to simulate" + default_text("%(default)s"))
    p.add_argument("--max-epochs", dest="max_num_epochs", type=int, default=5, metavar="N",
                   help="epoch ceiling handed to sbi, whose loop counts epochs from 0 and so trains at "
                        "most N+1; a run the ceiling stops keeps the last epoch's network, not the best, "
                        "unless the patience ran out on that epoch too" + default_text("%(default)s"))
    # No default here: an absent flag forwards nothing, so the stage keeps its own network size.
    p.add_argument("--hidden-features", dest="hidden_features", type=int, default=None, metavar="N",
                   help="flow width per transform" + default_clause("hidden_features"))
    p.add_argument("--num-transforms", dest="num_transforms", type=int, default=None, metavar="N",
                   help="flow depth" + default_clause("num_transforms"))
    p.add_argument("--checkpoint", action="store_true",
                   help="checkpoint the training rows every max(1, num_runs // 2) batches; OFF by "
                        "default so a second run of one config re-runs the simulation path this "
                        "subcommand exists to exercise")
    p.add_argument("--save", action="store_true",
                   help="name the artifacts this run BUILDS: smoke_prior / smoke_posterior. A second "
                        "--save run against the same store is refused by name, at stage entry")
    p.add_argument("--store-root", dest="store_root", default=None, metavar="PATH",
                   help="the artifact store this run writes (prior, cache, posterior, observation, "
                        "calibration, inference); reuse one, with --prior and --checkpoint, to "
                        "resume" + default_text("a fresh temporary directory, left on disk"))
    p.add_argument("--prior", dest="prior", default=None, metavar="REF",
                   help="prior artifact in --store-root to load instead of building, by name or id; "
                        "a resume needs it, because the cache is keyed on the prior's fit and two "
                        "fits of one box differ")
    add_resume_flags(p)
    # THE ONE subcommand that writes to a root of its own rather than to PRISM_ARTIFACTS: `main`
    # reads this property (never `hasattr(args, "store_root")`, which `fdt` and `crossval` now
    # satisfy too) to decide whether an unnamed --store-root means a fresh mkdtemp root, whether an
    # empty auto-created root is removed after a failure, and which Ctrl-C advice is printed.
    p.set_defaults(handler=run_smoke, temp_store_root=True)
    return {"smoke": p}


def run_smoke(args, store):
    """The stage sequence: the card test (``core.diagnostics.card_test``), one step over one process, one
    store and one cfg -- the resume run (same store, same --num-runs, the prior loaded by name) loads its
    prior into that cfg, and --stages can stop early. What stays here is the command line's own: the
    flags' checks, the configuration and its banner, and every line of the console, framed from what the
    step tells of each stage and from what it returns."""
    from core import cli, config, orchestrator
    from core.diagnostics import card_test
    from core.SBI.statistics import FEATURE_LABELS, SUMMARY_WIDTH, VALID_FLAG_LABELS
    from core.tool import UsageError
    from core.tool.config_args import build_cfg, close_sink, make_cfg

    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    unknown = [s for s in stages if s not in STAGES]
    if unknown:
        raise UsageError(f"--stages: unknown stage(s) {unknown}; choose from {list(STAGES)}")
    if "posterior" in stages:
        # Two --resume settings that can never work, refused before the prior build in this command's
        # own words, naming its own flags; the step refuses the same two after these, naming its keywords.
        if args.resume in ("require", "never") and not args.checkpoint:
            raise UsageError(f"--resume {args.resume} needs --checkpoint: without it no simulation "
                             f"cache is read or written, so the policy has nothing to act on.")
        if args.resume == "require" and args.prior is None:
            raise UsageError("--resume require needs --prior <the prior an earlier --save run built, by "
                             "name or id>: without it this run fits a new prior, whose fingerprint is "
                             "part of the cache identity, so no cache can match.")

    # load_gt=False: the truth is injected by simulated_inference, so the note about ignored cell
    # values prints once, at the stage that uses them.
    cfg, _ignored = build_cfg(args, load_gt=False)
    if "infer" in stages:
        # The cell is first read by the infer stage, after the prior, the training and the calibration.
        # Checked here against a THROWAWAY config, so a mistyped --cell (or one outside --bounds) costs
        # nothing, while the training cfg stays truth-free. Before the banner, so the console reads as
        # it always has; the step checks it again, for a caller that has not.
        cli.load_and_validate_gt(make_cfg(args), args.cell)
    fdim = orchestrator.expected_forcing_dim(cfg)
    want = SUMMARY_WIDTH + 1 + fdim

    print(f"[smoke] stages={','.join(stages)} num_runs={args.num_runs} "
          f"run_size={args.run_size_cap} n_cal={args.n_cal} max_epochs={args.max_num_epochs} "
          f"seed={args.seed} save={args.save} checkpoint={args.checkpoint} "
          f"prior={args.prior or '(build new)'} store={store.root}")
    if cfg.chi_mode:
        print(f"[smoke] chi ceiling: <= {cfg.chi_max_cycles:g} drive cycles per probe "
              f"(floor {config.CHI_MIN_CYCLES:g})")
    print(f"[smoke] conditioning width = {len(FEATURE_LABELS)}+{len(VALID_FLAG_LABELS)} + 1 + "
          f"{fdim} = {want}", flush=True)
    if args.store_root and not args.checkpoint:
        print("[smoke] --store-root given without --checkpoint: nothing will be resumable",
              flush=True)
    print(f"[smoke] --run-size caps the training batch; the prior sweep keeps the hardware batch "
          f"({cfg.hw.batch_size}).", flush=True)

    t0 = time.time()

    def _framed(name, event, seconds):
        if event == "skipped":
            print(f"[skip] {name}", flush=True)
        elif event == "started":
            print(f"\n=== {name} ===", flush=True)
        elif event == "finished":
            print(f"[ok] {name} in {seconds:.1f}s", flush=True)
        elif event == "failed":
            # The stage banner, before the traceback: a failure an hour into a simulation log must
            # not be findable only by scrolling.
            print(f"\n[smoke] *** FAILED in stage {name} ***", flush=True)

    done = card_test(cfg, cell=args.cell, t_obs_s=args.t_obs_s, seed=args.seed, stages=tuple(stages),
                     num_runs=args.num_runs, run_size_cap=args.run_size_cap, n_cal=args.n_cal,
                     max_num_epochs=args.max_num_epochs, hidden_features=args.hidden_features,
                     num_transforms=args.num_transforms, checkpoint=args.checkpoint, save=args.save,
                     prior_ref=args.prior, resume=args.resume, new_run=args.new_run, on_stage=_framed,
                     fig_sink=close_sink, store=store)
    if done.prior is None:
        print("\n[smoke] nothing further to run without a prior.")
        return 0
    if done.posterior is None:
        print("\n[smoke] prior only; stopping before training.")
        return 0

    print(f"\n[smoke] ALL STAGES COMPLETED in {time.time() - t0:.1f}s "
          f"({', '.join(f'{k} {v:.0f}s' for k, v in done.seconds.items())})")
    print("[smoke] This says the chain RUNS. It says nothing about calibration -- SBC at these "
          "sizes has no power (t_scale's effective sample size is the batch count). Read the "
          "masked-probe and OOD warnings above.")
    return 0
