@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo No .venv found in %CD% - run the install steps in INSTALL_NOTES.md first.
  exit /b 1
)

set "PORT=%~1"
if "%PORT%"=="" set "PORT=7860"

set "GRADIO_ANALYTICS_ENABLED=False"
echo Starting Budget3D web UI on http://127.0.0.1:%PORT%  (Ctrl+C to stop)
.venv\Scripts\python.exe webui.py --port %PORT% --inbrowser
