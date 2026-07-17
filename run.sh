#!/bin/bash
# Launch the PRISM desktop GUI (PySide6) on macOS / Linux. Windows users: run.bat.
#
# Must be launched such that the working directory is the repo root -- core/config.py builds all
# Resources/ paths from the current working directory. The cd below guarantees that regardless of where
# this script is invoked from.

# The directory this script lives in = the project root.
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
cd "$SCRIPT_DIR" || exit 1

# Prefer python3 (a bare `python` may be Python 2 or absent on macOS/Linux). Use `python3 -m core` for
# the interactive CLI instead of the GUI. exec replaces this shell so signals reach Python directly.
echo "Starting GUI..."
exec python3 -m core.gui
