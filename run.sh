#!/bin/bash
# Launch the PRISM desktop GUI (PySide6) on macOS / Linux. Windows users: run.bat.
#
# The working directory does not matter to the code (core/config.py resolves both roots from its own
# location); the cd below only keeps typed paths relative to the repo.

# The directory this script lives in = the project root.
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
cd "$SCRIPT_DIR" || exit 1

# Prefer python3 (a bare `python` may be Python 2 or absent on macOS/Linux). `python3 -m core
# <subcommand>` is the command-line tool. exec replaces this shell so signals reach Python directly.
# torch+MKL ship two OpenMP runtimes under conda; without this the first simulation aborts with OMP Error #15.
export KMP_DUPLICATE_LIB_OK=TRUE
echo "Starting GUI..."
exec python3 -m core.gui
