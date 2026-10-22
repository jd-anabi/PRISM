"""Run ``python -m core`` and end it like a power cut right after a chosen pass of a training is saved.

It exists for the resume run of the smoke gate on the graphics card. A fit of seconds cannot be stopped
at an exact pass from outside, so this script runs the command-line tool with the arguments it is given
and, the moment the training's chosen pass is saved whole (its state, its record and its line), ends the
process at once, with exit code 9 and nothing cleaned up: no stop is recorded, so the kept training reads
as stopped unexpectedly, and its lock is freed by the system as the process dies. The tool then resumes
it in a new process.

    python tests/_fit_drill.py --end-after-pass N <the arguments of python -m core>

Its name does not start with ``test_``, so the suites never collect it.
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # the repository root, from this file's place
import core.__main__  # noqa: E402,F401 -- the OpenMP setting and the Agg backend, before torch; runs nothing


def main(argv: list[str] | None = None) -> int:
    """``--end-after-pass N`` and then ``python -m core``'s own arguments; the tool's exit code, unless the
    training saves pass ``N``, when the process ends there with exit code 9."""
    parser = argparse.ArgumentParser(prog="_fit_drill", add_help=False, allow_abbrev=False)
    parser.add_argument("--end-after-pass", dest="end_after_pass", type=int, required=True, metavar="N")
    args, rest = parser.parse_known_args(sys.argv[1:] if argv is None else list(argv))

    from core.SBI import fit_state
    from core.tool import main as tool_main
    real = fit_state.FitSession.save_pass

    def save_pass(self, take):
        real(self, take)
        if self.passes == args.end_after_pass:      # the last pass whose save went through whole
            # os._exit skips Python's own flush, and the last saved pass's line must reach the log.
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(9)

    fit_state.FitSession.save_pass = save_pass
    try:
        return tool_main(rest)
    finally:
        fit_state.FitSession.save_pass = real


if __name__ == "__main__":
    raise SystemExit(main())
