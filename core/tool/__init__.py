"""``python -m core <subcommand>``: the flags-only command-line tool.

One of the two front ends over the orchestrator stages and compositions; the other is the GUI. Every
flag reaches its stage as a keyword argument, and this package reads no environment variable at all --
the four PRISM does read are named in the epilog below and nowhere else.

``main(argv) -> int`` never calls ``sys.exit``: the suite drives it in process, which is the only way
to catch a handler that swaps the process default store and never puts it back.
"""
import argparse
import sys
import tempfile
import traceback
from pathlib import Path

from core.refusals import Refusal

from . import config_args, diagnostics, fdt, smoke, stages
from .config_args import UsageError  # noqa: F401 -- part of this package's public surface
from .logging_console import console_handlers
from .fields import fix_sentence

EPILOG = """\
environment -- the only variables PRISM reads, and none of them is a substitute for a flag:
  PRISM_RESOURCES         inputs root: Bounds/ Cells/ Units/ Models/  (default <repo>/Resources)
  PRISM_ARTIFACTS         artifacts root (default <repo>/Artifacts); every subcommand but `smoke`
                          writes here, and `smoke` takes --store-root instead
  PRISM_VRAM_CEILING_GIB  core-level: GiB one simulation batch may plan to occupy (0 = auto)
  PRISM_MEM_LOG_EVERY     core-level: batches between memory log lines
The last two are read by core/SBI/pipeline.py, never by this tool: PRISM_VRAM_CEILING_GIB live, on
every batch plan; PRISM_MEM_LOG_EVERY once, when core.SBI.pipeline is imported. They are
deliberately not flags: they change the memory PLAN for a batch, not the rows it produces.
"""


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m core", epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="The PRISM command-line tool. The GUI is `python -m core.gui`.")
    sub = p.add_subparsers(dest="cmd", required=True, metavar="<subcommand>")
    p.subcommands = {}
    for module in (stages, diagnostics, smoke, fdt):
        p.subcommands.update(module.register(sub))
    return p


def _smoke_interrupt_advice(args, root) -> str:
    """K5, fix round 1: the Ctrl-C hint for a resumable cache. Every OTHER subcommand's own command
    line, re-issued with --resume require, IS the resumable one -- that is what the generic message
    says. smoke keys its store on --store-root rather than on PRISM_ARTIFACTS, so re-issuing run 1's
    OWN command line instead re-BUILDS the prior (refused by name under --save, or a fresh, different
    fit otherwise); the resumable form names --prior and --stages explicitly, as the drill does.

    The prior to name is the one this run used: the one it loaded (--prior), else smoke_prior when
    --save named what it built, else the unnamed one it built, whose id only its directory carries.
    The store is --store-root, else the temp ``root`` main created -- main keeps that root whenever it
    holds anything, committed batches included, and the [smoke] banner printed its path."""
    prior = args.prior or ("smoke_prior" if args.save else
                           "<the id of the prior this run built: its directory is "
                           "priors/_unnamed__<id> under the store root>")
    return (f"re-run with --store-root {args.store_root or root} --prior {prior} --checkpoint "
            f"--stages prior,posterior --resume require, and the same --num-runs and --run-size, "
            f"to continue them.")


def _remove_if_still_empty(root: Path) -> None:
    """K3, fix round 1: ``main`` calls this only for a root IT created via ``mkdtemp`` (never a
    user-named ``--store-root``, never PRISM_ARTIFACTS), and only when the run did not succeed. A bad
    ``--bounds`` used to leave an empty ``%TEMP%\\prism_smoke_*`` behind forever, its path never
    printed anywhere an operator would think to look. ``rmdir`` is the whole safety property: it
    raises (caught and ignored) the moment the directory holds anything at all, so a root that
    reached even one write -- a partial prior, a committed checkpoint batch -- is never touched."""
    try:
        root.rmdir()
    except OSError:
        pass


def main(argv=None) -> int:
    """Parse, build the store, run one handler. Exit codes: 0 success or --help; 2 a usage error;
    1 a refusal or a bug; 130 Ctrl-C."""
    parser = build_parser()
    try:
        args = parser.parse_args(sys.argv[1:] if argv is None else list(argv))
    except SystemExit as e:                    # argparse's own exit: --help is 0, bad usage is 2
        return int(e.code or 0)

    from core import config, registry
    from core.artifacts import ArtifactStore, use_store
    registry.load_user_models()                # idempotent; AFTER parsing, so --help stays torch-free
    # smoke is the one subcommand with its own root: a fresh store per run unless one is named, so
    # two runs never share a cache by accident and a named one can be resumed. Keyed on the FLAG
    # (only smoke defines --store-root), never on the subcommand name.
    has_store_root = hasattr(args, "store_root")
    auto_root = has_store_root and not args.store_root         # True only for smoke's own mkdtemp
    if has_store_root:
        root = Path(args.store_root or tempfile.mkdtemp(prefix="prism_smoke_"))
    else:
        root = config.artifacts_root()
    rc = 1
    try:
        root.mkdir(parents=True, exist_ok=True)
        # use_store AND store= at every call: the context makes the default right for anything that
        # reaches for it, the keyword makes each stage independent of the default. set_default_store
        # is never called -- that was scripts/smoke_train.py's leak.
        # The console handlers live exactly as long as the handler call: information to stdout,
        # warning/error to stderr with a prefix (spec §4.3). The tool's own framing prints ([prism],
        # [cfg], [smoke], this ladder) stay prints and never pass through them.
        with use_store(ArtifactStore(root)) as store, console_handlers():
            rc = int(args.handler(args, store) or 0)
    except KeyboardInterrupt:
        # I2, fix round 1: fdt/crossval set their own `interrupt_note` (core/tool/fdt.py) through
        # set_defaults -- they keep no cache and take no --resume, so the generic advice below
        # (written for a checkpointed training cache) would be flatly wrong for them. Every other
        # subcommand leaves interrupt_note unset, so getattr's default keeps their message as is.
        note = getattr(args, "interrupt_note", None)
        if note is not None:
            print(f"prism {args.cmd}: interrupted: {note}", file=sys.stderr)
        else:
            advice = _smoke_interrupt_advice(args, root) if has_store_root else \
                "the same command with --resume require continues them."
            print(f"prism {args.cmd}: interrupted: the artifact being written was removed. If a "
                  f"[checkpoint] line above says batches were saved, {advice}", file=sys.stderr)
        rc = 130
    except UsageError as e:
        # A Refusal too since piece 3, caught FIRST so a bad flag combination keeps exit 2.
        print(f"prism {args.cmd}: usage: {e}", file=sys.stderr)
        rc = 2
    except Refusal as e:
        # V3: one operator line -- the message, then the flag that answers its field, from
        # core/tool/fields.py. fix_sentence owns the parentheses and is empty for field=None or a
        # key with no flag, so the line then ends at the message. No class name and no
        # [raised at ...]: those were hedges for a bug disguised as a ValueError, which a dedicated
        # class no longer needs.
        fix = fix_sentence(e.field)
        sfx = (" " + fix) if fix else ""
        print(f"prism {args.cmd}: refused: {e.message}{sfx}", file=sys.stderr)
        rc = 1
    except (ValueError, FileNotFoundError) as e:
        # An UNCONVERTED refusal -- a bare ValueError from a site piece 3 has not reached yet -- or a
        # missing file. No traceback -- the message is written for an operator -- but the class and
        # the innermost frame are named, so a genuine bug that happens to raise ValueError still says
        # where it came from.
        tb = e.__traceback__
        while tb is not None and tb.tb_next is not None:
            tb = tb.tb_next
        where = "" if tb is None else \
            f" [raised at {Path(tb.tb_frame.f_code.co_filename).name}:{tb.tb_lineno}]"
        print(f"prism {args.cmd}: refused: {type(e).__name__}: {e}{where}", file=sys.stderr)
        rc = 1
    except Exception:                          # noqa: BLE001 -- a bug: show the whole thing
        traceback.print_exc()
        print(f"prism {args.cmd}: *** FAILED ***", file=sys.stderr)
        rc = 1
    if auto_root and rc != 0:
        _remove_if_still_empty(root)
    return rc
