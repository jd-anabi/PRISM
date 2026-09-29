"""Shared flags, the config builder, the figure sink and the success line for ``python -m core``.

WHAT THIS REPLACES. ``scripts/_common.script_cfg`` built a config out of the environment --
CELL/BOUNDS/MODEL/CHI/CHI_K/TOBS_S -- which meant a run's inputs lived in a shell nobody recorded and
a forgotten ``BOUNDS`` silently smoke-tested a different box (the warning at the top of
scripts/smoke_train.py). Here every one of them is a flag, ``--bounds`` is required, and every flag
reaches its stage as a KEYWORD ARGUMENT: assigning a module constant would be a no-op, because
orchestrator binds config constants at import.

NOTHING IN THIS PACKAGE READS THE ENVIRONMENT (tests/test_tool.py pins it). The two roots and the two
core-level settings PRISM does read are named in the --help epilog.

Every ``core`` import that costs torch is inside a function: the parser is built before
``registry.load_user_models`` runs, and ``--help`` must not cost a torch import. ``core.refusals`` is
torch-free and is the one top-level exception, for ``UsageError``'s base, the note's length limit
and the default clause every help ends with.
"""
import argparse
from pathlib import Path

from core.refusals import NOTE_MAX_CHARS, Refusal, default_clause, default_text


class UsageError(Refusal):
    """A bad flag COMBINATION -- one argparse cannot express because it depends on the built config
    (which recordings a mode takes, which drive keys exist). Raised after parsing; exit code 2."""


def add_config_flags(p) -> None:
    """The config flags every SBI subcommand takes.

    ``--bounds`` is required on all of them, and it is one rule rather than a default: the bounds file
    declares WHICH parameters are inferred and in what order, and therefore the observation mode, so an
    unset one used to resolve the 12-dimensional spontaneous box behind the operator's back.
    """
    p.add_argument("--bounds", required=True, metavar="PATH",
                   help="bounds file: which parameters are inferred, in what order, and the box. It "
                        "must be the one any posterior this command loads was trained with -- the "
                        "store refuses a mismatch in model, order, box or mode.")
    p.add_argument("--model", default=None, metavar="NAME",
                   help="model name" + default_text("the bounds file's parent folder, upper-cased"))
    p.add_argument("--chi", dest="chi_mode", default=None, action=argparse.BooleanOptionalAction,
                   help="chi(omega) observation mode" + default_text("--no-chi"))
    p.add_argument("--chi-k", dest="chi_n_freqs", type=int, default=None, metavar="K",
                   help="probe frequencies per observation" + default_clause("chi_n_freqs"))
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto",
                   help="auto picks the card when CUDA is present with compute capability 8.0 or "
                        "above, else Apple's MPS when present, else the CPU; cpu forces the CPU; "
                        "cuda requires the card and is refused when it is absent or below 8.0"
                        + default_text("%(default)s"))


def add_name_flags(p, *, name_help: "str | None" = None) -> None:
    """``--name`` and ``--note``. ``name_help`` replaces the name's help for a subcommand whose name
    is not used as given -- ``crossval`` names its two records from it (the stem rule)."""
    p.add_argument("--name", default="", metavar="NAME",
                   help=name_help or ("name the artifact this command writes ('' = unnamed). A taken "
                                      "name is refused at the stage's entry, before anything is spent."))
    p.add_argument("--note", default="", metavar="TEXT",
                   help=f"free text recorded in the artifact's manifest: one line, at most "
                        f"{NOTE_MAX_CHARS} characters ('' = no note)")


def add_training_flags(p, *, fisher: bool) -> None:
    """The training knobs ``train`` and ``tsnpe`` share, each defaulting to None so that only a given
    one reaches the stage and the stage's own default applies otherwise.

    Defined once here because the two subcommands used to define these flags twice, and ``tsnpe``'s
    copy carried no help at all. ``tsnpe`` passes ``fisher=False``: a narrowing round reuses its
    parent's rotation and never computes a Fisher of its own, so a Fisher flag there would be
    silently ignored. ``smoke`` does not use this helper: its defaults are its own literals, the
    drill's sizes, and its ``--checkpoint`` is an on/off switch rather than a count of batches.
    """
    p.add_argument("--num-runs", type=int, default=None, metavar="N",
                   help="training batches to simulate" + default_clause("num_runs"))
    p.add_argument("--run-size", dest="run_size_cap", type=int, default=None, metavar="N",
                   help="ceiling on simulations per training batch; 0 = the hardware batch"
                        + default_clause("run_size_cap"))
    p.add_argument("--hidden-features", type=int, default=None, metavar="N",
                   help="flow width per transform" + default_clause("hidden_features"))
    p.add_argument("--num-transforms", type=int, default=None, metavar="N",
                   help="flow depth" + default_clause("num_transforms"))
    p.add_argument("--learning-rate", type=float, default=None, metavar="X",
                   help="Adam learning rate" + default_clause("learning_rate"))
    p.add_argument("--stop-after-epochs", type=int, default=None, metavar="N",
                   help="early-stopping patience, in epochs" + default_clause("stop_after_epochs"))
    p.add_argument("--max-epochs", dest="max_num_epochs", type=int, default=None, metavar="N",
                   help="hard ceiling on training epochs" + default_clause("max_num_epochs"))
    if fisher:
        p.add_argument("--fisher-m", type=int, default=None, metavar="N",
                       help="ensemble per latent perturbation for the Fisher rotation"
                            + default_clause("fisher_m"))
        p.add_argument("--fisher-dz", type=float, default=None, metavar="X",
                       help="latent central-difference step" + default_clause("fisher_dz"))
        p.add_argument("--fisher-points", type=int, default=None, metavar="N",
                       help="operating points the Fisher rotation is averaged over"
                            + default_clause("fisher_points"))
    p.add_argument("--checkpoint-every", type=int, default=None, metavar="N",
                   help="batches between checkpoint commits; 0 = no cache, nothing resumable"
                        + default_clause("checkpoint_every"))


def add_resume_flags(p) -> None:
    """``--resume`` and ``--new-run``: the resume policy and the near-miss consent, where a cache one
    setting away is refused before any simulation unless ``--new-run`` is given. Shared verbatim by
    every subcommand that trains a cache: ``train``, ``tsnpe`` and ``smoke``. Before this the three
    defined the same two flags three times, with three slightly different help strings.

    ``--checkpoint-every`` is deliberately NOT here even though ``train`` and ``tsnpe`` share it
    identically: ``smoke`` has no flag of that shape at all -- its own ``--checkpoint`` is a bool
    (on/off; the cadence is computed from ``--num-runs``), not the batches-between-commits int
    ``train``/``tsnpe`` take. Bundling it would restate one subcommand's knob as another's.
    """
    p.add_argument("--resume", choices=("auto", "require", "never"), default=None,
                   help="resume policy for the training cache: auto resumes this run's own cache; "
                        "require refuses when there is none; never refuses to resume one"
                        + default_clause("resume"))
    p.add_argument("--new-run", action="store_true",
                   help="start a new simulation cache even though a committed one ONE setting away "
                        "exists; it silences that near-miss refusal and nothing else")


def model_from_path(model, path) -> str:
    """``model``, else ``path``'s parent folder upper-cased (the ``<kind>/<model>/`` layout every
    input folder shares -- ``Bounds/<model>/``, ``Cells/<model>/``). The one rule both ``model_for``
    (over ``--bounds``) and ``core.tool.fdt.model_for_cell`` (over ``--cell``) apply to their own
    path flag, written once so the two do not restate it."""
    return str(model or Path(path).parent.name).upper()


def model_for(args) -> str:
    """``--model``, else the ``--bounds`` parent folder upper-cased (the ``Bounds/<model>/`` layout --
    the same rule cells resolve by, so a command never has to hard-code a model name)."""
    return model_from_path(args.model, args.bounds)


def build_cfg(args, *, load_gt: bool = False):
    """``(cfg, ignored)`` from the shared flags and nothing else: no environment variable reaches the
    config.

    THE BUILDER NEVER SETS ``cfg.T_obs``. ``--t-obs`` travels to the one function that uses it, so a
    stage that has no business with a duration cannot inherit one nobody asked for.

    ``load_gt`` injects the cell's ground-truth values here; only the identifiability modes that need
    the truth before the stage runs ask for it. ``infer`` and ``smoke`` leave it to
    ``simulated_inference``, so the note about ignored cell values prints exactly once.
    """
    from core import cli
    cfg = make_cfg(args)
    cell = getattr(args, "cell", None)
    ignored = cli.load_and_validate_gt(cfg, cell) if load_gt else []
    describe(cfg, cell=cell, bounds=args.bounds, ignored=ignored)
    return cfg, ignored


def make_cfg(args):
    """The SimConfig the shared flags describe, and nothing else: no truth loaded, nothing printed.
    ``build_cfg`` is this plus the cell and the banner; ``smoke`` also calls it alone, for a throwaway
    config to check --cell against before the prior build."""
    from core import cli, config, registry
    from core.config import VALID_LABELS, VALID_MODELS
    from core.refusals import refuse
    model = model_for(args)
    spec = registry.get(model)
    if spec is None or (spec.is_user_model and not registry.is_sbi_user_model(model)):
        raise ValueError(
            f"'{model}' is not a model this command can run: it is neither a built-in "
            f"{list(VALID_MODELS)[:3]} nor a registered user model eligible for inference (one saved "
            f"in Resources/Models/, with spontaneous dynamics and at least one ND parameter). Pass "
            f"--model, or point --bounds at Bounds/<model>/.")
    labels = (VALID_LABELS[VALID_MODELS.index(model)] if model in VALID_MODELS
              else registry.get(model).labels)
    if args.device == "cuda":
        # detect_device()'s OWN DeviceConfig, never a hand-built one: its batch size and dtype enter
        # the simulation identity, so `cuda` and `auto` on a qualifying card share one cache.
        hw = config.detect_device()
        if hw.device.type != "cuda":
            refuse("device", f"The compute device 'cuda' is not available on this machine: CUDA is "
                             f"absent or the card's compute capability is below 8.0 "
                             f"(detected {hw.device.type}).")
    else:
        hw = config.cpu_device() if args.device == "cpu" else None
    return cli.make_sim_config(model, labels, registry.state_dep_drift(model), args.bounds,
                               chi_mode=args.chi_mode, chi_n_freqs=args.chi_n_freqs, hw=hw)


def describe(cfg, *, cell=None, bounds=None, ignored=()) -> None:
    """Print the resolved configuration, including the PARAMETER ORDER.

    The order is printed on every run on purpose. Simulators bind parameter columns POSITIONALLY
    (``Model(*torch.unbind(params, dim=1), ...)``), so a bounds file whose order differs from the one a
    command assumes mis-binds values with no error and no crash -- just wrong physics. The bounds path
    printed beside the ND and rescale order is the only place that difference shows.
    """
    from core import orchestrator
    t_obs = "(unset)" if cfg.T_obs is None else f"{cfg.T_obs:g} (cell time units)"
    print(f"[cfg] model={cfg.model} mode={cfg.observation_mode.upper()} "
          f"device={cfg.hw.device} T_obs={t_obs}", flush=True)
    if cell:
        print(f"[cfg] cell={cell}", flush=True)
    if bounds:
        print(f"[cfg] bounds={bounds}", flush=True)
    if cfg.chi_mode:
        print(f"[cfg] chi: K={cfg.chi_n_freqs} F0={cfg.chi_f0} range={cfg.chi_freq_bounds} "
              f"x Omega_0  -> {cfg.chi_k_pad} probe slots, conditioning block = "
              f"{orchestrator.expected_forcing_dim(cfg)} features", flush=True)
    print(f"[cfg] ND order:      {list(cfg.params_dict.keys())}", flush=True)
    print(f"[cfg] rescale order: {list(cfg.rescale_params.keys())}", flush=True)
    if cfg.force_params_dict:
        note = " (IGNORED in chi mode -- chi probes at multiples of the measured Omega_0)" \
               if cfg.chi_mode else ""
        print(f"[cfg] forcing order: {list(cfg.force_params_dict.keys())}{note}", flush=True)
    if ignored:
        print(f"[cfg] cell values the bounds file does not declare, so IGNORED: {sorted(ignored)}",
              flush=True)


def close_sink(title, fig) -> None:
    """The one figure sink every call is given. On a write path ``ArtifactWriter.fig_sink`` saves the
    PNG into the artifact and THEN forwards here, so the tool never has to know which stage draws."""
    from matplotlib import pyplot as plt
    plt.close(fig)


def knobs(args, *names) -> dict:
    """``{name: value}`` for every knob that was actually given.

    Forwarding an unset knob as None would work, but forwarding NOTHING is what keeps each default in
    the stage that owns it -- one place, read from config.py, rather than restated here.
    """
    return {n: getattr(args, n) for n in names if getattr(args, n, None) is not None}


def report(*artifacts) -> None:
    """One line per artifact this run wrote: kind, ``<name>__<id>``, and the path to it."""
    for a in artifacts:
        print(f"[prism] {a.kind} {a.path.name}  {a.path}", flush=True)


def add_accept_flags(p, *, other_observation: bool = False) -> None:
    """The tool REFUSES a non-amortized load by default, and these two map 1:1 onto
    ``artifacts.Accept``. An INFERENCE records the flags used in its own artifact's
    ``results.accepted``, so a number produced under one carries that fact with it; a calibration and
    a TSNPE round do NOT carry that record themselves -- for those, the flags only ever unlock a load,
    and what is on record instead is the non-amortized posterior's own manifest, which already says
    ``amortized: false``."""
    p.add_argument("--accept-truncated", action="store_true",
                   help="load a NON-AMORTIZED (TSNPE) posterior. It is valid only near the "
                        "observation its region was drawn around; elsewhere the flow extrapolates.")
    if other_observation:
        p.add_argument("--accept-other-observation", action="store_true",
                       help="run a NON-AMORTIZED posterior on an observation other than its "
                            "region's. A re-simulated cell draws new noise, so simulated inference "
                            "needs this for any TSNPE posterior.")


def accept_from(args):
    from core.artifacts import Accept
    return Accept(truncated=args.accept_truncated,
                  other_observation=getattr(args, "accept_other_observation", False))


def parse_forced(spec: str):
    """``"rec.npy@12.5"`` -> ``("rec.npy", 12.5)``; a bare path -> ``(path, None)``.

    Split at the LAST ``@`` so a path that contains one still parses. The frequency is the one the
    bench actually drove at: the lock-in locks in where it is TOLD, and a lock-in at the wrong
    frequency decays like a sinc rather than failing.
    """
    path, sep, freq = str(spec).rpartition("@")
    if not sep:
        return str(spec), None
    try:
        return path, float(freq)
    except ValueError:
        raise UsageError(f"--forced {spec!r}: {freq!r} is not a drive frequency in Hz. Write "
                         f"PATH@HZ, e.g. recording.npy@12.5.") from None


def recording_set(cfg, args):
    """The ``RecordingSet`` ``infer --spont`` describes, checked against the mode the bounds file
    declares. A pure function of ``cfg.observation_mode`` and ``cfg.force_params_dict``, so every
    refusal here costs nothing and names the flag to fix."""
    from core.SBI.observations import RecordingSet
    forced = [parse_forced(s) for s in (args.forced or [])]

    if cfg.observation_mode == "chi":
        if not forced:
            raise UsageError("chi mode conditions on a passive recording plus at least one "
                             "single-tone forced one: pass --forced PATH@HZ.")
        bare = [p for p, f in forced if f is None]
        if bare:
            raise UsageError(f"chi mode: every driven recording must state the frequency (Hz) it was "
                             f"driven at -- write --forced PATH@HZ. Missing for: {bare}.")
        if args.f0_si is None:
            raise UsageError("chi mode needs --f0-si: chi is response/drive, so the lock-in divides "
                             "by the physical drive amplitude the recordings were made at.")
        if args.drive:
            raise UsageError("--drive is a single-drive (forced) setting. chi probes at the "
                             "frequencies --forced names, so drop it.")
        return RecordingSet(spont=args.spont, forced=tuple(forced), T_obs_s=args.t_obs_s,
                            F0_si=args.f0_si)

    if cfg.observation_mode == "spontaneous":
        if forced or args.drive or args.f0_si is not None:
            raise UsageError("this bounds file declares no Forcing section, so the observation is a "
                             "single passive recording: drop --forced, --drive and --f0-si.")
        return RecordingSet(spont=args.spont, T_obs_s=args.t_obs_s)

    if len(forced) != 1 or forced[0][1] is not None:
        raise UsageError("forced mode takes exactly one --forced PATH, without @HZ: its drive is the "
                         "one --drive describes.")
    if args.f0_si is not None:
        raise UsageError("--f0-si is a chi setting; in forced mode the amplitude is --drive amp=<N>.")
    drive = {}
    for item in (args.drive or []):
        key, sep, value = str(item).partition("=")
        if not sep:
            raise UsageError(f"--drive {item!r}: write NAME=VALUE in SI units, e.g. amp=1e-12.")
        try:
            drive[key.strip()] = float(value)
        except ValueError:
            raise UsageError(f"--drive {item!r}: {value!r} is not a number.") from None
    want = set(cfg.force_params_dict)
    if set(drive) != want:
        raise UsageError(f"--drive must name exactly the drive this bounds file declares: "
                         f"{sorted(want)}; got {sorted(drive)}.")
    return RecordingSet(spont=args.spont, forced=((forced[0][0], None),), T_obs_s=args.t_obs_s,
                        forcing_params_si=drive)


def load_posterior_and_prior(cfg, ref, accept, store):
    """``(posterior, prior)`` for a subcommand that loads one: THE PRIOR IS ALWAYS THE POSTERIOR'S OWN.

    Never a prior the operator names. build_posterior refuses a prior that is not the one the
    posterior's manifest records, so an explicit flag could only ever be refused -- and a TSNPE child
    records its base prior, so this resolves for a round's posterior too.
    """
    from core import orchestrator
    p_ref = store.get("posterior", ref).parents["prior"]
    print(f"[prism] prior: {p_ref} (the posterior's training prior)", flush=True)
    prior = orchestrator.build_prior(cfg, p_ref, False, fig_sink=close_sink, store=store)
    posterior = orchestrator.build_posterior(cfg, prior, ref, False, accept=accept,
                                             fig_sink=close_sink, store=store)
    return posterior, prior
