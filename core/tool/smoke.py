"""``python -m core smoke``: every pipeline stage end to end, at tiny sizes, non-interactively.

WHY THIS EXISTS. The suites pin contracts on stubs and tiny synthetic inputs. This runs the REAL
chain on the REAL bounds/cell files -- stability-screened prior build, NPE training, SBC/TARP
calibration, PPC and the eye test -- so plumbing that only appears when the stages are wired
together (mode widths, sidecar round-trip, the chi block's shape agreeing with the network's input
layer) fails here, in minutes, instead of hours into a multi-day run. It is the GPU gate: every
suite runs on the CPU, and a tensor on the wrong device is invisible until the card sees it.

WHAT TO WATCH, beyond "it finished":

  * ``chi: N/M probes masked`` -- the COUNT, not the presence. Some masking is by design (a probe
    under CHI_MIN_CYCLES on a short recording). The reference is ~37 % of TRAINING probes.
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
  * ``bounds=`` AND ``rescale order=`` in the config banner, TOGETHER. ``--bounds`` is required on
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
sequence runs inside one ``seeded(seed, device)`` context -- one seed, streams running on, as the
script did: per-stage seeding would replay the training strata in the calibration set (trap X5).
Seeded runs are NOT bitwise-reproducible on CUDA or across devices.
"""
import argparse
import time

STAGES = ("prior", "posterior", "validate", "infer")

EPILOG = """\
Writes to --store-root, or a fresh temp directory when it is not given -- NEVER to PRISM_ARTIFACTS,
which every OTHER subcommand follows but smoke does not. Reads PRISM_RESOURCES (the inputs root)
like every subcommand, and two core-level settings read live by core/SBI/pipeline.py, never by this
tool: PRISM_VRAM_CEILING_GIB (GiB one simulation batch may plan to occupy, 0 = auto) and
PRISM_MEM_LOG_EVERY (batches between memory log lines).

What to watch: the masked-probe count (~37 % of TRAINING probes, and a single run within +/-12 pp
is uninformative -- the effective sample size is the BATCH count, not the probe count); the mode
banner's width; bounds= and rescale order= together. SBC at these sizes has no power: only a crash
means anything.

--run-size caps the TRAINING batch only; the prior sweep keeps the hardware batch, because that
sweep is iteration-bounded and shrinking its batch makes the prior worse for the same wall clock.

A seeded run is not bitwise-reproducible on CUDA or across devices.
"""


def _stages_type(value: str) -> str:
    """argparse ``type=`` for ``--stages``: validated at PARSE TIME (fix round 1, K2), so an unknown
    stage is an argparse error before ``main`` ever resolves a store root or creates a temp
    directory -- and before ``registry.load_user_models()``, so it costs no torch import either.
    Returns the original string unchanged; ``run_smoke`` still does its own split (kept as a second,
    defensive check for any caller that reaches it without going through argparse)."""
    stages = [s.strip() for s in value.split(",") if s.strip()]
    unknown = [s for s in stages if s not in STAGES]
    if unknown:
        raise argparse.ArgumentTypeError(
            f"--stages: unknown stage(s) {unknown}; choose from {list(STAGES)}")
    return value


def register(subparsers):
    """The ``smoke`` subcommand. Its defaults are the drill's sizes, and they are the ONE place in
    the tool that restates a literal (every other subcommand leaves defaults to its stage)."""
    from core.tool.config_args import add_config_flags, add_resume_flags
    p = subparsers.add_parser(
        "smoke", help="every stage end to end at tiny sizes (the GPU gate)",
        epilog=EPILOG, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_flags(p)
    p.add_argument("--cell", required=True,
                   help="the cell file whose truth the infer stage simulates")
    p.add_argument("--t-obs", dest="t_obs_s", type=float, default=None,
                   help="observation duration in seconds (default: config.T_MIN_EXP_S)")
    p.add_argument("--seed", type=int, default=0, help="RNG seed for the whole run (default 0)")
    p.add_argument("--stages", default=",".join(STAGES), type=_stages_type,
                   help=f"comma list, a subset of {','.join(STAGES)} (default: all four)")
    p.add_argument("--num-runs", dest="num_runs", type=int, default=4,
                   help="training batches (default 4)")
    p.add_argument("--run-size", dest="run_size_cap", type=int, default=32,
                   help="simulations per training batch (default 32); the prior sweep is unaffected")
    p.add_argument("--n-cal", dest="n_cal", type=int, default=40,
                   help="calibration datasets for SBC/TARP (default 40)")
    p.add_argument("--max-epochs", dest="max_num_epochs", type=int, default=5,
                   help="training epochs ceiling (default 5)")
    p.add_argument("--checkpoint", action="store_true",
                   help="checkpoint the training rows every max(1, num_runs // 2) batches; OFF by "
                        "default so a second run of one config re-runs the simulation path this "
                        "subcommand exists to exercise")
    p.add_argument("--save", action="store_true",
                   help="name the artifacts this run BUILDS: smoke_prior / smoke_posterior. A second "
                        "--save run against the same store is refused by name, at stage entry")
    p.add_argument("--store-root", dest="store_root", default=None,
                   help="the artifact store this run writes (prior, cache, posterior, observation, "
                        "calibration, inference). Default: a fresh temp directory, left on disk. "
                        "Reuse one, with --prior and --checkpoint, to resume")
    p.add_argument("--prior", dest="prior", default=None,
                   help="a prior artifact (name or id) in --store-root to LOAD instead of building. "
                        "Required for a resume: the cache identity includes prior_fingerprint, and "
                        "two fits of one box differ")
    add_resume_flags(p)
    p.set_defaults(handler=run_smoke)
    return {"smoke": p}


def run_smoke(args, store):
    """The stage sequence. A composition of its own, not a chain of ``main`` calls: one process, one
    store, one cfg -- run 2 loads its prior into that cfg and --stages can stop early."""
    from core import config, orchestrator
    from core.diagnostics.rng import seeded
    from core.SBI.statistics import FEATURE_LABELS, SUMMARY_WIDTH, VALID_FLAG_LABELS
    from core.tool import UsageError
    from core.tool.config_args import build_cfg, close_sink, knobs

    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    unknown = [s for s in stages if s not in STAGES]
    if unknown:
        raise UsageError(f"--stages: unknown stage(s) {unknown}; choose from {list(STAGES)}")

    # load_gt=False: the truth is injected by simulated_inference, so the note about ignored cell
    # values prints once, at the stage that uses them.
    cfg, _ignored = build_cfg(args, load_gt=False)
    t_obs_s = config.T_MIN_EXP_S if args.t_obs_s is None else float(args.t_obs_s)
    fdim = orchestrator.expected_forcing_dim(cfg)
    want = SUMMARY_WIDTH + 1 + fdim

    print(f"[smoke] stages={','.join(stages)} num_runs={args.num_runs} "
          f"run_size={args.run_size_cap} n_cal={args.n_cal} epochs={args.max_num_epochs} "
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

    ck_every = max(1, args.num_runs // 2) if args.checkpoint else 0
    t0, done = time.time(), {}

    def _stage(name, fn):
        if name not in stages:
            print(f"[skip] {name}", flush=True)
            return None
        t = time.time()
        print(f"\n=== {name} ===", flush=True)
        try:
            out = fn()
        except BaseException:
            # The stage banner, before the traceback: a failure an hour into a simulation log must
            # not be findable only by scrolling.
            print(f"\n[smoke] *** FAILED in stage {name} ***", flush=True)
            raise
        done[name] = time.time() - t
        print(f"[ok] {name} in {done[name]:.1f}s", flush=True)
        return out

    with seeded(args.seed, cfg.hw.device):
        prior = _stage("prior", lambda: orchestrator.build_prior(
            cfg, args.prior, args.prior is None, fig_sink=close_sink, store=store,
            name="smoke_prior" if (args.save and args.prior is None) else ""))
        if prior is None:
            print("\n[smoke] nothing further to run without a prior.")
            return 0

        post = _stage("posterior", lambda: orchestrator.build_posterior(
            cfg, prior, None, True, fig_sink=close_sink, store=store,
            name="smoke_posterior" if args.save else "",
            num_runs=args.num_runs, run_size_cap=args.run_size_cap,
            max_num_epochs=args.max_num_epochs, checkpoint_every=ck_every,
            new_run=args.new_run, **knobs(args, "resume")))
        if post is None:
            print("\n[smoke] prior only; stopping before training.")
            return 0

        _stage("validate", lambda: orchestrator.validate_calibration(
            cfg, post, prior, fig_sink=close_sink, store=store, n_cal=args.n_cal))
        _stage("infer", lambda: _infer(cfg, post, prior, t_obs_s, args.cell, store, want))

    print(f"\n[smoke] ALL STAGES COMPLETED in {time.time() - t0:.1f}s "
          f"({', '.join(f'{k} {v:.0f}s' for k, v in done.items())})")
    print("[smoke] This says the chain RUNS. It says nothing about calibration -- SBC at these "
          "sizes has no power (t_scale's effective sample size is the batch count). Read the "
          "masked-probe and OOD warnings above.")
    return 0


def _infer(cfg, posterior, prior, t_obs_s, cell, store, want):
    """The infer stage: the composition, then the two shape checks the script carried.

    The checks run AFTER the inference rather than between the two stages, because the composition
    is one call -- and they are cheap assertions on an artifact that is already on disk, so nothing
    is lost by asking a second later.
    """
    import torch
    from core import orchestrator
    from core.tool.config_args import close_sink
    obs, inf = orchestrator.simulated_inference(cfg, posterior, t_obs_s, cell=cell, prior=prior,
                                                fig_sink=close_sink, store=store)
    if obs.width != want:
        raise RuntimeError(f"observation width {obs.width} != the mode's {want}")
    if not torch.isfinite(obs.x_obs).all():
        raise RuntimeError("the observation carries non-finite conditioning")
    return obs, inf
