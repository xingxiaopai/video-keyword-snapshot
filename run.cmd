@echo off
rem ============================================================
rem  Video keyword snapshot - launcher
rem
rem  Double-click this file to start the interactive wizard.
rem  Or pass arguments to use CLI mode directly:
rem     run.cmd -i "D://videos" -k "keyword1,keyword2" -o "D://out"
rem
rem  Interpreter lookup order:
rem     1. VKWS_PYTHON environment variable
rem     2. .python-path file in this folder (one line, gitignored)
rem     3. .venv\Scripts\python.exe in this folder
rem     4. python on PATH
rem ============================================================
setlocal
cd /d "%~dp0"

set "PY="
if defined VKWS_PYTHON if exist "%VKWS_PYTHON%" set "PY=%VKWS_PYTHON%"
if not defined PY if exist ".python-path" set /p PY=<".python-path"
if defined PY if not exist "%PY%" set "PY="
if not defined PY if exist ".venv\Scripts\python.exe" set "PY=%CD%\.venv\Scripts\python.exe"
if not defined PY set "PY=python"

if not "%~1"=="" goto cli
"%PY%" run_interactive.py
goto done
:cli
"%PY%" video_keyword_snapshot.py %*
:done
echo.
pause
