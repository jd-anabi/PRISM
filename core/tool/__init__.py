"""``python -m core <subcommand>``: the flags-only command-line tool.

One of the two front ends over the orchestrator stages and compositions; the other is the GUI. Every
flag reaches its stage as a keyword argument, and this package reads no environment variable at all --
the four PRISM does read are named in the epilog below and nowhere else.

``main(argv) -> int`` never calls ``sys.exit``: the suite drives it in process, which is the only way
to catch a handler that swaps the process default store and never puts it back.
"""
import argparse
import sys
import traceback
from pathlib import Path

from . import config_args, stages
from .config_args import UsageError  # noqa: F401 -- part of this package's public surface

EPILOG = """\
environment -- the only variables PRISM reads, and none of them is a substitute for a flag:
  PRISM_RESOURCES         inputs root: Bounds/ Cells/ Units/ Models/  (default <repo>/Resources)
  PRISM_ARTIFACTS         artifacts root (default <repo>/Artifacts); every subcommand but `smoke`
                          writes here, and `smoke` takes --store-root instead
  PRISM_VRAM_CEILING_GIB  core-level: GiB one simulation batch may plan to occupy (0 = auto)
  PRISM_MEM_LOG_EVERY     core-level: batches between memory log lines
The last two are read live by core/SBI/pipeline.py, never by this tool. They are deliberately not
flags: they change the memory PLAN for a batch, not the rows it produces.
"""


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m core", epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="The PRISM command-line tool. The GUI is `python -m core.gui`.")
    sub = p.add_subparsers(dest="cmd", required=True, metavar="<subcommand>")
    p.subcommands = {}
    for module in (stages,):
        p.subcommands.update(module.register(sub))
    return p


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
    try:
        # smoke names its own root; everything else follows PRISM_ARTIFACTS, read here at call time.
        # Created up front so an unusable root fails in the first second, not on day three.
        root = Path(getattr(args, "store_root", None) or config.artifacts_root())
        root.mkdir(parents=True, exist_ok=True)
        # use_store AND store= at every call: the context makes the default right for anything that
        # reaches for it, the keyword makes each stage independent of the default. set_default_store
        # is never called -- that was scripts/smoke_train.py's leak.
        with use_store(ArtifactStore(root)) as store:
            return int(args.handler(args, store) or 0)
    except KeyboardInterrupt:
        print(f"prism {args.cmd}: interrupted: the artifact being written was removed. If a "
              f"[checkpoint] line above says batches were saved, the same command with "
              f"--resume require continues them.", file=sys.stderr)
        return 130
    except UsageError as e:
        print(f"prism {args.cmd}: usage: {e}", file=sys.stderr)
        return 2
    except (ValueError, FileNotFoundError) as e:
        # Every stage refusal, StoreError, cli.UnitParseError and FDTModelError land here. No
        # traceback -- the message is written for an operator -- but the innermost frame is named, so
        # a genuine bug that happens to raise ValueError still says where it came from.
        tb = e.__traceback__
        while tb is not None and tb.tb_next is not None:
            tb = tb.tb_next
        where = "" if tb is None else \
            f" [raised at {Path(tb.tb_frame.f_code.co_filename).name}:{tb.tb_lineno}]"
        print(f"prism {args.cmd}: refused: {type(e).__name__}: {e}{where}", file=sys.stderr)
        return 1
    except Exception:                          # noqa: BLE001 -- a bug: show the whole thing
        traceback.print_exc()
        print(f"prism {args.cmd}: *** FAILED ***", file=sys.stderr)
        return 1
