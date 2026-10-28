@echo off
setlocal
REM PRISM's launcher for Windows. Double-click it, or type run.bat in a terminal: it finds the Python of PRISM's
REM environment itself, starts the window with no console window beside it, and returns at once. The first time,
REM it also puts a PRISM shortcut on the desktop, and never again, even when that shortcut was deleted.
REM   run.bat --where          says which Python it would use and whether it would make the desktop shortcut,
REM                            and starts and makes nothing.
REM   run.bat --make-shortcut  makes the desktop shortcut, when it was never made, and starts nothing; to have it
REM                            made again, delete its mark first (--where names it).
REM The working directory does not matter to PRISM; the window is started from the repository's folder.
pushd "%~dp0"
set "LAUNCH_REPO=%CD%"
REM torch+MKL ship two OpenMP runtimes on Windows/conda; without this the first simulation aborts with OMP Error #15.
set KMP_DUPLICATE_LIB_OK=TRUE

set "LAUNCH_MODE=start"
if /i "%~1"=="--where" set "LAUNCH_MODE=where"
if /i "%~1"=="--make-shortcut" set "LAUNCH_MODE=shortcut"
if not "%~1"=="" if "%LAUNCH_MODE%"=="start" (
    echo run.bat takes no argument but --where or --make-shortcut, and started nothing.
    popd
    exit /b 2
)

REM PRISM's Python: the active environment's, when its folder holds PRISM's libraries (a prompt left in conda's
REM base environment holds neither, and is passed over); else the environment biophys-env in the usual places.
REM Folders are looked at; nothing is imported or run.
set "LAUNCH_PYTHONW="
if defined CONDA_PREFIX if exist "%CONDA_PREFIX%\Lib\site-packages\PySide6\" if exist "%CONDA_PREFIX%\Lib\site-packages\torch\" if exist "%CONDA_PREFIX%\pythonw.exe" set "LAUNCH_PYTHONW=%CONDA_PREFIX%\pythonw.exe"
for %%P in ("%USERPROFILE%\anaconda3" "%USERPROFILE%\miniconda3" "%LOCALAPPDATA%\anaconda3" "%ProgramData%\anaconda3" "%ProgramData%\miniconda3") do if not defined LAUNCH_PYTHONW if exist "%%~P\envs\biophys-env\pythonw.exe" set "LAUNCH_PYTHONW=%%~P\envs\biophys-env\pythonw.exe"

set "LAUNCH_MISSING=PRISM's Python was not found. Open the Anaconda Prompt, type: conda activate biophys-env, then run run.bat from there."
if not defined LAUNCH_PYTHONW (
    if "%LAUNCH_MODE%"=="start" (
        powershell -NoProfile -NonInteractive -Command "[void](New-Object -ComObject WScript.Shell).Popup($env:LAUNCH_MISSING, 0, 'PRISM', 48)"
    ) else (
        echo %LAUNCH_MISSING%
    )
    popd
    exit /b 1
)

REM The shortcut is made once: a mark beside PRISM's settings file says it was. PRISM_DESKTOP names the folder it
REM goes in, for the launcher's own tests; a person never sets it, and the shortcut goes on Windows' own desktop.
set "LAUNCH_MARK=%APPDATA%\PRISM\shortcut-made"
set "LAUNCH_ICON=%LAUNCH_REPO%\core\gui\assets\app\prism.ico"
set "LAUNCH_LINK=PRISM.lnk on Windows' own desktop"
if defined PRISM_DESKTOP set "LAUNCH_LINK=%PRISM_DESKTOP%\PRISM.lnk"
if "%LAUNCH_MODE%"=="where" (
    echo PRISM's Python: "%LAUNCH_PYTHONW%"
    if exist "%LAUNCH_MARK%" (
        echo The desktop shortcut was made once already, and is not made again: its mark is "%LAUNCH_MARK%".
    ) else (
        echo The desktop shortcut would be made now: "%LAUNCH_LINK%", its mark "%LAUNCH_MARK%".
    )
    popd
    exit /b 0
)
set "LAUNCH_FAILED="
if not exist "%LAUNCH_MARK%" (
    powershell -NoProfile -NonInteractive -Command "$ErrorActionPreference = 'Stop'; $folder = $env:PRISM_DESKTOP; if (-not $folder) { $folder = [Environment]::GetFolderPath('Desktop') }; $path = Join-Path $folder 'PRISM.lnk'; $link = (New-Object -ComObject WScript.Shell).CreateShortcut($path); $link.TargetPath = $env:LAUNCH_PYTHONW; $link.Arguments = '-m core.gui'; $link.WorkingDirectory = $env:LAUNCH_REPO; $link.IconLocation = $env:LAUNCH_ICON; $link.Description = 'PRISM'; $link.Save(); New-Item -ItemType Directory -Force -Path (Split-Path $env:LAUNCH_MARK) | Out-Null; Set-Content -Path $env:LAUNCH_MARK -Value 'run.bat made the desktop shortcut PRISM.lnk once, and does not make it again.'; Write-Output ('The desktop shortcut was made: ' + $path)"
    if errorlevel 1 set "LAUNCH_FAILED=1"
) else (
    if "%LAUNCH_MODE%"=="shortcut" echo The desktop shortcut was made once already, and is not made again: delete its mark "%LAUNCH_MARK%" to make it again.
)
if defined LAUNCH_FAILED echo The desktop shortcut could not be made.

REM PRISM writes what it prints to its window's log whatever streams it is handed: a file or a pipe, when the output
REM of run.bat itself is sent to one, as much as none at all.
set "PRISM_DETACHED=1"
if "%LAUNCH_MODE%"=="start" start "PRISM" "%LAUNCH_PYTHONW%" -m core.gui
popd
if defined LAUNCH_FAILED if "%LAUNCH_MODE%"=="shortcut" exit /b 1
exit /b 0
