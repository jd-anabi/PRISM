#!/bin/bash

# Get the directory where this script is located
SCRIPT_DIR=$(cd $(dirname "$0") && pwd)

# Change to that directory (this is your project root)
cd "$SCRIPT_DIR"

# Run the desktop GUI (PySide6). Use `python -m core` instead for the interactive CLI.
echo Starting GUI...
python -m core.gui