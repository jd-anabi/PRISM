#!/bin/bash

# Get the directory where this script is located
SCRIPT_DIR=$(cd $(dirname "$0") && pwd)

# Change to that directory (this is your project root)
cd "$SCRIPT_DIR"

# Run the python module
echo Starting application...
python -m core