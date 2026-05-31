# GGF PiD Agent Notes

This repository is the public source for the GGF Windows/NVIDIA standalone PiD app.

## What This App Does

- It is an image-to-image PiD decoder/upscaler workflow.
- It uses the official PiD Flux-compatible clean decode path.
- It does not run full text-to-image generation.
- It installs official PiD checkpoint files from `nvidia/PiD` on Hugging Face.
- It primes the local Gemma caption model cache used by PiD's text conditioning.

## Packaging Contract

- Do not commit model files, checkpoints, `.venv`, `.hf_home`, outputs, or zip files.
- The distribution installer should live outside the cloned app folder and clone this repo from GitHub.
- The expected install flow is:
  1. `git clone https://github.com/gjnave/GGF-PiD.git`
  2. `cd GGF-PiD`
  3. run `Install-PiD.bat`
  4. launch with `run.bat`
- Runtime should use the app-local Hugging Face cache at `.hf_home`.
- Installer and runtime set `HF_HUB_DISABLE_XET=1` to avoid Windows Xet transfer hangs.
- `run.bat` intentionally sets `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` after installation.

## Maintenance Notes

- Keep the UI in `app/app.py`.
- Keep runtime logic in `app/pid_runtime.py`.
- Keep model download logic in `app/download_models.py`.
- Keep the app branded in the GGF style.
- Validate with a real image decode smoke test before publishing changes.
