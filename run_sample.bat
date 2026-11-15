@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

if "%~1"=="" goto usage

set "IMAGE=%~1"
set "BUDGET=%~2"
if "%BUDGET%"=="" set "BUDGET=0.5"
set "OUT=%~3"
if "%OUT%"=="" set "OUT=outputs\%~n1.ply"
if not exist "outputs" mkdir "outputs"

echo [1/2] sampling at budget %BUDGET% ...
.venv\Scripts\python.exe scripts\sample.py --ckpt weights\budget3d_idol.safetensors --image "%IMAGE%" --budget %BUDGET% --out "%OUT%" --mode synthetic
if errorlevel 1 (
  echo SAMPLE FAILED
  exit /b 1
)

set "PREV=%OUT:~0,-4%_preview.png"
echo [2/2] rendering turntable preview ...
.venv\Scripts\python.exe scripts\preview_ply.py --ply "%OUT%" --out "%PREV%"

echo.
echo   Gaussians : %OUT%
echo   Preview   : %PREV%
exit /b 0

:usage
echo Usage: run_sample.bat ^<image^> [budget 0.0625-1.0] [out.ply]
echo Example: run_sample.bat inputs\character_male.png 0.5
exit /b 1
