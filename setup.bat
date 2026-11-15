@echo off
REM Budget3D Web UI - one-command Windows setup.  Idempotent: re-running skips finished steps.
REM Requires: git, and either uv (https://docs.astral.sh/uv/) or Python 3.12 on PATH via py launcher.
setlocal
cd /d "%~dp0"

echo ============================================
echo  Budget3D Web UI - setup
echo ============================================
echo.

where git >nul 2>&1
if errorlevel 1 ( echo [X] git is required but was not found on PATH. & goto fail )

set "USE_UV=0"
where uv >nul 2>&1
if not errorlevel 1 set "USE_UV=1"
if "%USE_UV%"=="0" (
  py -3.12 --version >nul 2>&1
  if errorlevel 1 (
    echo [X] Need either uv ^(https://docs.astral.sh/uv/^) or Python 3.12 ^(py -3.12^).
    goto fail
  )
  echo [i] uv not found - falling back to py -3.12 + pip
)

REM ---------------------------------------------------------------- 1. Idol's model code
if exist "third_party\idol-sdk\.git" (
  echo [1/6] idol-sdk present
) else (
  echo [1/6] cloning idol-sdk ...
  if not exist "third_party" mkdir "third_party"
  git clone --depth 1 https://github.com/speridlabs/idol-sdk third_party/idol-sdk
  if errorlevel 1 goto fail
)

REM ---------------------------------------------------------------- 2. diffusers dtype patch
findstr /C:"_dtype_kwarg" "third_party\idol-sdk\idol\conditioning.py" >nul 2>&1
if not errorlevel 1 (
  echo [2/6] diffusers dtype patch already applied
) else (
  echo [2/6] applying diffusers dtype patch ...
  git apply patches\idol-sdk-diffusers-dtype.patch --directory=third_party/idol-sdk
  if errorlevel 1 (
    echo     Could not apply the patch - idol-sdk may have changed upstream.
    echo     See "Why the patch exists" in FORK_NOTES.md.
    goto fail
  )
)

REM ---------------------------------------------------------------- 3. venv
if exist ".venv\Scripts\python.exe" (
  echo [3/6] venv present
) else (
  echo [3/6] creating Python 3.12 venv ...
  if "%USE_UV%"=="1" (
    uv venv --python 3.12 .venv
  ) else (
    py -3.12 -m venv .venv
  )
  if errorlevel 1 goto fail
)

set "PY=.venv\Scripts\python.exe"
if not exist "%PY%" ( echo [X] venv python missing & goto fail )

REM ---------------------------------------------------------------- 4. dependencies
"%PY%" -c "import torch, gradio, diffusers, tensordict, PIL.Image" >nul 2>&1
if not errorlevel 1 (
  echo [4/6] dependencies already installed
  goto dinov3
)

echo [4/6] installing dependencies ^(torch cu128 is a ~2.5 GB download; first run only^) ...
if "%USE_UV%"=="1" goto deps_uv

"%PY%" -m pip install --upgrade pip
"%PY%" -m pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128
if errorlevel 1 goto fail
"%PY%" -m pip install -r requirements.txt plyfile
if errorlevel 1 goto fail
REM gradio must go AFTER requirements: it caps pillow<12 while requirements.txt pins 12.1.1, so
REM resolving them together is unsatisfiable. Installing it separately downgrades pillow to 11.x,
REM which sampling does not care about (see FORK_NOTES.md).
"%PY%" -m pip install gradio==5.49.1
if errorlevel 1 goto fail
"%PY%" -m pip install --no-deps -e third_party/idol-sdk
if errorlevel 1 goto fail
goto dinov3

:deps_uv
uv pip install --python "%PY%" torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128
if errorlevel 1 goto fail
uv pip install --python "%PY%" -r requirements.txt plyfile
if errorlevel 1 goto fail
REM separate, for the pillow cap reason above
uv pip install --python "%PY%" gradio==5.49.1
if errorlevel 1 goto fail
uv pip install --python "%PY%" --no-deps -e third_party/idol-sdk
if errorlevel 1 goto fail

:dinov3
REM -------------------------------------------------------------- 5. DINOv3 image encoder
if exist "weights\dinov3\vith16plus_lvd1689m\model.safetensors" (
  echo [5/6] DINOv3 already converted
) else (
  echo [5/6] converting DINOv3 ViT-H+/16 to HF format ^(downloads ~1.2 GB^) ...
  "%PY%" scripts\convert_dinov3_to_hf.py --model-name vith16plus_lvd1689m --save-dir weights/dinov3
  if errorlevel 1 goto fail
)

REM -------------------------------------------------------------- 6. Budget3D weights
if exist "weights\budget3d_idol.safetensors" (
  echo [6/6] Budget3D-Idol weights present
) else (
  echo [6/6] downloading Budget3D-Idol weights ^(~1.5 GB^) ...
  "%PY%" -c "from huggingface_hub import snapshot_download; snapshot_download('BudgetAny/Budget3D-Idol', allow_patterns=['config.json','*.safetensors'], local_dir='weights')"
  if errorlevel 1 goto fail
)

echo.
echo ============================================
echo  Setup complete.
echo.
echo    webui.bat                 start the web UI on http://127.0.0.1:7860
echo    webui.bat 7000            ... on another port
echo    run_sample.bat img.png    CLI: image -^> .ply + turntable
echo ============================================
exit /b 0

:fail
echo.
echo Setup stopped. Fix the error above and re-run setup.bat - finished steps are skipped.
exit /b 1
