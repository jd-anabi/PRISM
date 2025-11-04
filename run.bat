@echo off
REM Change the directory to the one where this .bat file is located
pushd "%~dp0"

echo Starting application...
REM Run the python module
python -m core

REM Return to the original directory
popd

REM THIS IS THE NEW, IMPORTANT LINE
REM It will keep the window open so you can see any output or errors
pause