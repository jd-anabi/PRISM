"""Shared flags, the config builder, the figure sink and the success line for ``python -m core``.

WHAT THIS REPLACES. ``scripts/_common.script_cfg`` built a config out of the environment --
CELL/BOUNDS/MODEL/CHI/CHI_K/TOBS_S -- which meant a run's inputs lived in a shell nobody recorded and
a forgotten ``BOUNDS`` silently smoke-tested a different box (the warning at the top of
scripts/smoke_train.py). Here every one of them is a flag, ``--bounds`` is required, and every flag
reaches its stage as a KEYWORD ARGUMENT: assigning a module constant would be a no-op, because
orchestrator binds config constants at import (CLAUDE.md).

NOTHING IN THIS PACKAGE READS THE ENVIRONMENT (tests/test_tool.py pins it). The two roots and the two
core-level settings PRISM does read are named in the --help epilog.

Every ``core`` import is inside a function: the parser is built before ``registry.load_user_models``
runs, and ``--help`` must not cost a torch import.
"""
import argparse
from pathlib import Path


class UsageError(ValueError):
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
                   help="model name (default: the --bounds parent folder, upper-cased)")
    p.add_argument("--chi", dest="chi_mode", default=None, action=argparse.BooleanOptionalAction,
                   help="chi(omega) observation mode (default: config.CHI_MODE)")
    p.add_argument("--chi-k", dest="chi_n_freqs", type=int, default=None, metavar="K",
                   help="probe frequencies per observation (default: config.CHI_N_FREQS)")
    p.add_argument("--device", choices=("auto", "cpu"), default="auto",
                   help="auto detects CUDA; cpu forces config.cpu_device() (default: auto)")


def add_name_flags(p) -> None:
    p.add_argument("--name", default="", metavar="NAME",
                   help="name the artifact this command writes ('' = unnamed). A taken name is "
                        "refused at the stage's entry, before anything is spent.")
    p.add_argument("--note", default="", metavar="TEXT",
                   help="free text recorded in the artifact's manifest")


def model_for(args) -> str:
    """``--model``, else the ``--bounds`` parent folder upper-cased (the ``Bounds/<model>/`` layout --
    the same rule cells resolve by, so a command never has to hard-code a model name)."""
    if args.model:
        return str(args.model).upper()
    return Path(args.bounds).parent.name.upper()


def build_cfg(args, *, load_gt: bool = False):
    """``(cfg, ignored)`` from the shared flags: the environment-free ``_common.script_cfg``.

    THE BUILDER NEVER SETS ``cfg.T_obs``. ``--t-obs`` travels to the one function that uses it, so a
    stage that has no business with a duration cannot inherit one nobody asked for.

    ``load_gt`` injects the cell's ground-truth values here; only the identifiability modes that need
    the truth before the stage runs ask for it. ``infer`` and ``smoke`` leave it to
    ``simulated_inference``, so the note about ignored cell values prints exactly once.
    """
    from core import cli, config, registry
    from core.config import VALID_LABELS, VALID_MODELS
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
    hw = config.cpu_device() if args.device == "cpu" else None
    cfg = cli.make_sim_config(model, labels, registry.state_dep_drift(model), args.bounds,
                              chi_mode=args.chi_mode, chi_n_freqs=args.chi_n_freqs, hw=hw)
    cell = getattr(args, "cell", None)
    ignored = cli.load_and_validate_gt(cfg, cell) if load_gt else []
    describe(cfg, cell=cell, bounds=args.bounds, ignored=ignored)
    return cfg, ignored


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
