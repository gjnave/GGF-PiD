from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from huggingface_hub import snapshot_download


ROOT = Path(__file__).resolve().parents[1]
CACHE_ROOT = ROOT / ".hf_home"

CHECKPOINT_FILES = {
    Path("checkpoints/ae.safetensors"): 335_304_388,
    Path("checkpoints/PiD_res2k_sr4x_official_flux_distill_4step/model_ema_bf16.pth"): 2_724_842_961,
    Path("checkpoints/PiD_res2kto4k_sr4x_official_flux_distill_4step/model_ema_bf16.pth"): 2_724_842_961,
}

GEMMA_PATTERNS = [
    "config.json",
    "generation_config.json",
    "model-00001-of-00002.safetensors",
    "model-00002-of-00002.safetensors",
    "model.safetensors.index.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
]


def _seed_local_hf_cache(repo_id: str) -> None:
    repo_key = f"models--{repo_id.replace('/', '--')}"
    local_repo_dir = CACHE_ROOT / "hub" / repo_key
    default_repo_dir = Path.home() / ".cache" / "huggingface" / "hub" / repo_key

    if not default_repo_dir.exists():
        return

    local_repo_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(default_repo_dir, local_repo_dir, dirs_exist_ok=True)


def _seed_checkpoint_files() -> None:
    for relative_path, expected_size in CHECKPOINT_FILES.items():
        target_path = ROOT / relative_path
        if target_path.exists() and target_path.stat().st_size == expected_size:
            continue

        for candidate in ROOT.parent.glob(f"*/{relative_path.as_posix()}"):
            if ROOT in candidate.parents:
                continue
            if candidate.is_file():
                target_path.parent.mkdir(parents=True, exist_ok=True)
                print(f"[MODELS] Reusing local checkpoint: {relative_path}")
                shutil.copy2(candidate, target_path)
                break


def _download_checkpoint_files() -> None:
    curl_exe = shutil.which("curl.exe") or shutil.which("curl")
    if curl_exe is None:
        raise RuntimeError("curl.exe is required to download PiD checkpoint files on Windows.")

    for relative_path, expected_size in CHECKPOINT_FILES.items():
        target_path = ROOT / relative_path
        if target_path.exists() and target_path.stat().st_size == expected_size:
            print(f"[SKIP] {relative_path}")
            continue
        if target_path.exists() and target_path.stat().st_size > expected_size:
            target_path.unlink()

        target_path.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://huggingface.co/nvidia/PiD/resolve/main/{relative_path.as_posix()}"
        print(f"[DOWNLOADING] {relative_path}")
        subprocess.run(
            [
                curl_exe,
                "-L",
                "--fail",
                "--retry",
                "12",
                "--retry-delay",
                "5",
                "--retry-all-errors",
                "--continue-at",
                "-",
                "--output",
                str(target_path),
                url,
            ],
            check=True,
        )


def main() -> None:
    os.environ.setdefault("HF_HOME", str(CACHE_ROOT))
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

    print("[MODELS] Downloading official PiD Flux-compatible assets from nvidia/PiD")
    _seed_checkpoint_files()
    _download_checkpoint_files()

    print("[MODELS] Priming Gemma caption model cache for offline first run")
    _seed_local_hf_cache("Efficient-Large-Model/gemma-2-2b-it")
    snapshot_download(
        repo_id="Efficient-Large-Model/gemma-2-2b-it",
        allow_patterns=GEMMA_PATTERNS,
    )

    print("[DONE] PiD model download complete.")


if __name__ == "__main__":
    main()
