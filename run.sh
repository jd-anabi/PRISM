#!/bin/bash
# Launch the PRISM desktop GUI (PySide6) on macOS / Linux. Windows users: run.bat.
#
# The working directory does not matter to the code (core/config.py resolves both roots from its own
# location); the cd below only keeps typed paths relative to the repo.

# The directory this script lives in = the project root.
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
cd "$SCRIPT_DIR" || exit 1

# Prefer python3 (a bare `python` may be Python 2 or absent on macOS/Linux). `python3 -m core
# <subcommand>` is the command-line tool.
# The one line this script prints: the window says nothing here, even when it cannot start.
echo "PRISM starts in the background; if no window opens, see window-log.txt beside its settings file or in the temporary folder."
# torch+MKL ship two OpenMP runtimes under conda; without this the first simulation aborts with OMP Error #15.
export KMP_DUPLICATE_LIB_OK=TRUE
# The window runs detached from this terminal, which may then be closed: what it prints outside a run goes to
# window-log.txt beside its settings file, and a run's full text to the run's own dated file there.
export PRISM_DETACHED=1
nohup python3 -m core.gui >/dev/null 2>&1 &
