@echo off
setlocal
cd /d "%~dp0"

IF EXIST "disclaimer.md" (
   TYPE "disclaimer.md"
   pause
)

title Install PiD Decoder Studio
set "HF_HOME=%~dp0.hf_home"
set "HF_HUB_DISABLE_SYMLINKS_WARNING=1"
set "HF_HUB_DISABLE_XET=1"

echo ============================================
echo         Install PiD Decoder Studio
echo ============================================
echo.

if exist "%~dp0.git" (
  echo [SETUP] Git repo detected. Skipping clone and using local folder.
)

if not exist "%~dp0.venv\Scripts\python.exe" (
  echo [SETUP] Creating Python venv...
  py -3.10 -m venv "%~dp0.venv"
  if errorlevel 1 (
    py -3.11 -m venv "%~dp0.venv"
    if errorlevel 1 (
      echo [ERROR] Could not create venv with py -3.10 or py -3.11.
      pause
      exit /b 1
    )
  )
)

call "%~dp0.venv\Scripts\activate.bat"

echo [SETUP] Upgrading pip...
"%~dp0.venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto :fail

echo [SETUP] Installing CUDA PyTorch for NVIDIA...
"%~dp0.venv\Scripts\python.exe" -m pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision torchaudio
if errorlevel 1 goto :fail

echo [SETUP] Installing PiD dependencies...
"%~dp0.venv\Scripts\python.exe" -m pip install "diffusers>=0.37.0" "transformers>=4.57.0" accelerate gradio hydra-core omegaconf pyyaml attrs einops loguru termcolor fvcore iopath wandb imageio opencv-python pandas safetensors sentencepiece boto3 botocore huggingface_hub pillow packaging
if errorlevel 1 goto :fail

echo [SETUP] Installing local package...
"%~dp0.venv\Scripts\python.exe" -m pip install -e .
if errorlevel 1 goto :fail

echo [MODELS] Downloading official PiD Flux-compatible assets...
"%~dp0.venv\Scripts\python.exe" "%~dp0app\download_models.py"
if errorlevel 1 goto :fail

IF EXIST "about.nfo" TYPE "about.nfo"
echo.
echo [DONE] Install complete.
echo Run %~dp0run.bat
pause
exit /b 0

:fail
echo.
echo [ERROR] Install failed.
pause
exit /b 1
