@echo off
REM Change the directory to the one where this .bat file is located
pushd "%~dp0"

echo Starting GUI...
REM Run the desktop GUI (PySide6). Use `python -m core` instead for the interactive CLI.
python -m core.gui

REM Return to the original directory
popd

REM THIS IS THE NEW, IMPORTANT LINE
REM It will keep the window open so you can see any output or errors
pause