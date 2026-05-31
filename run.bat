@echo off
setlocal
cd /d "%~dp0"

title GGF PiD Decoder Studio

set "PYTHONIOENCODING=utf-8"
set "PYTHONUTF8=1"
set "CUDA_VISIBLE_DEVICES=0"
set "HF_HOME=%~dp0.hf_home"
set "HF_HUB_DISABLE_SYMLINKS_WARNING=1"
set "HF_HUB_OFFLINE=1"
set "TRANSFORMERS_OFFLINE=1"

if not exist "%~dp0.venv\Scripts\python.exe" (
  echo [ERROR] Local venv not found.
  echo Run Install-PiD.bat first.
  pause
  exit /b 1
)

call "%~dp0.venv\Scripts\activate.bat"
"%~dp0.venv\Scripts\python.exe" "%~dp0app\app.py"
pause
