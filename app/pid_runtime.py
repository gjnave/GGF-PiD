from __future__ import annotations

import gc
import os
import time
from pathlib import Path

import torch
from PIL import Image

from pid._src.inference._demo_from_clean_common import _add_noise, _load_input_image, _vae_decode
from pid._src.inference.checkpoint_registry import get_pid_checkpoint
from pid._src.utils.model_loader import load_model_from_checkpoint


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs"
CHECKPOINT_ROOT = ROOT / "checkpoints"
CACHE_ROOT = ROOT / ".hf_home"

_LOADED_MODEL = None
_LOADED_KEY: str | None = None

CHECKPOINT_FILES = {
    "ae": CHECKPOINT_ROOT / "ae.safetensors",
    "2k": CHECKPOINT_ROOT / "PiD_res2k_sr4x_official_flux_distill_4step" / "model_ema_bf16.pth",
    "2kto4k": CHECKPOINT_ROOT / "PiD_res2kto4k_sr4x_official_flux_distill_4step" / "model_ema_bf16.pth",
}

PRESET_CONFIG = {
    "2K Decode": {"ckpt_type": "2k", "input_resolution": 512, "target_resolution": 2048},
    "4K Decode": {"ckpt_type": "2kto4k", "input_resolution": 1024, "target_resolution": 4096},
}

os.environ.setdefault("HF_HOME", str(CACHE_ROOT))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


def _tensor_to_pil(sample: torch.Tensor) -> Image.Image:
    if sample.dim() == 4:
        sample = sample.squeeze(1)
    sample = (sample.float().clamp(-1, 1) + 1.0) * 127.5
    array = sample.permute(1, 2, 0).cpu().numpy().astype("uint8")
    return Image.fromarray(array)


def _save_image(sample: torch.Tensor, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    _tensor_to_pil(sample).save(path, quality=95)
    return str(path)


def _make_compare_strip(input_tensor: torch.Tensor, vae_tensor: torch.Tensor, pid_tensor: torch.Tensor, path: Path) -> str:
    input_img = _tensor_to_pil(input_tensor)
    vae_img = _tensor_to_pil(vae_tensor)
    pid_img = _tensor_to_pil(pid_tensor)

    target_height = pid_img.height

    def fit_height(img: Image.Image) -> Image.Image:
        width = max(1, int(round(img.width * (target_height / img.height))))
        return img.resize((width, target_height), Image.Resampling.LANCZOS)

    tiles = [fit_height(input_img), fit_height(vae_img), pid_img]
    total_width = sum(tile.width for tile in tiles)
    strip = Image.new("RGB", (total_width, target_height), color=(255, 249, 237))

    x = 0
    for tile in tiles:
        strip.paste(tile, (x, 0))
        x += tile.width

    path.parent.mkdir(parents=True, exist_ok=True)
    strip.save(path, quality=95)
    return str(path)


def _checkpoint_status() -> str:
    lines = []
    for name, file_path in CHECKPOINT_FILES.items():
        state = "present" if file_path.exists() else "missing"
        lines.append(f"{name}: {state}")
    return "\n".join(lines)


def _gemma_status() -> str:
    gemma_root = CACHE_ROOT / "hub" / "models--Efficient-Large-Model--gemma-2-2b-it" / "snapshots"
    model_files = list(gemma_root.glob("*/model-00001-of-00002.safetensors"))
    token_files = list(gemma_root.glob("*/tokenizer.json"))
    if model_files and token_files:
        return "present"
    return "missing"


def model_status() -> str:
    cuda_status = "unavailable"
    if torch.cuda.is_available():
        cuda_status = f"{torch.cuda.get_device_name(0)} / CUDA {torch.version.cuda}"

    loaded = _LOADED_KEY if _LOADED_KEY is not None else "none"
    return (
        f"PiD root: {ROOT}\n"
        f"Mode: Flux / Z-Image compatible clean decode\n"
        f"Loaded checkpoint: {loaded}\n"
        f"GPU: {cuda_status}\n"
        f"Checkpoint assets:\n{_checkpoint_status()}\n"
        f"Gemma caption cache: {_gemma_status()}\n"
        f"HF cache: {CACHE_ROOT}\n"
        f"Outputs: {OUTPUT_DIR}\n"
        f"Model license: NVIDIA NSCLv1, non-commercial use only"
    )


def unload_model() -> str:
    global _LOADED_MODEL, _LOADED_KEY

    _LOADED_MODEL = None
    _LOADED_KEY = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return model_status()


def _load_flux_model(ckpt_type: str):
    global _LOADED_MODEL, _LOADED_KEY

    if _LOADED_MODEL is not None and _LOADED_KEY == ckpt_type:
        return _LOADED_MODEL

    unload_model()
    ckpt = get_pid_checkpoint("flux", ckpt_type)
    model, _config = load_model_from_checkpoint(
        experiment_name=ckpt.experiment,
        checkpoint_path=ckpt.checkpoint_path,
        config_file="pid/_src/configs/pid/config.py",
        enable_fsdp=False,
        experiment_opts=[],
        strict=False,
        load_ema_to_reg=False,
    )
    model.eval()
    _LOADED_MODEL = model
    _LOADED_KEY = ckpt_type
    return model


def _resolve_caption(model, prompt: str) -> str:
    prompt = (prompt or "").strip()
    if prompt:
        return prompt

    fixed = None
    if getattr(model.config, "use_fixed_prompt", False):
        fixed = getattr(model.config, "fixed_positive_prompt", None)
    return fixed or "A detailed high quality image."


def decode_image(
    input_image: str,
    prompt: str,
    preset: str,
    degrade_sigma: float,
    keep_input_size: bool,
    cfg_scale: float,
    pid_inference_steps: int,
    seed: int,
) -> tuple[str | None, str | None, str | None, str]:
    if not input_image:
        return None, None, None, "Choose an input image first."

    if not torch.cuda.is_available():
        return None, None, None, "CUDA GPU not available. PiD needs an NVIDIA CUDA setup."

    if preset not in PRESET_CONFIG:
        return None, None, None, f"Unknown preset: {preset}"

    for name, required_path in CHECKPOINT_FILES.items():
        if name in ("ae", PRESET_CONFIG[preset]["ckpt_type"]) and not required_path.exists():
            return None, None, None, f"Missing checkpoint asset: {required_path}"

    preset_config = PRESET_CONFIG[preset]
    model = _load_flux_model(preset_config["ckpt_type"])
    caption = _resolve_caption(model, prompt)

    input_tensor = _load_input_image(
        input_image,
        preset_config["input_resolution"],
        keep_input_size=keep_input_size,
    ).to(dtype=torch.bfloat16, device="cuda")
    clean_latent = model.encode_lq_latent(input_tensor)

    vae_compression = int(model.vae_encoder.spatial_compression_factor)
    vae_h = int(clean_latent.shape[-2]) * vae_compression
    vae_w = int(clean_latent.shape[-1]) * vae_compression
    target_hw = (vae_h * 4, vae_w * 4)

    generator = torch.Generator(device="cuda").manual_seed(int(seed))
    latent = _add_noise(clean_latent.float(), float(degrade_sigma), generator).to(dtype=torch.bfloat16)

    with torch.no_grad():
        vae_img = _vae_decode(model, latent)

    data_batch = {
        model.config.input_caption_key: [caption],
        "LQ_video_or_image": torch.zeros_like(vae_img, dtype=torch.bfloat16, device="cuda"),
        "LQ_latent": latent.to(dtype=torch.bfloat16, device="cuda"),
        "degrade_sigma": torch.tensor([float(degrade_sigma)], device="cuda", dtype=torch.float32),
    }
    samples = model.generate_samples_from_batch(
        data_batch,
        cfg_scale=float(cfg_scale),
        num_steps=int(pid_inference_steps),
        seed=int(seed),
        shift=None,
        image_size=target_hw,
    )

    input_cpu = input_tensor.float().cpu().squeeze(0).clamp(-1, 1)
    vae_cpu = vae_img.float().cpu().squeeze(0).clamp(-1, 1)
    pid_cpu = samples[0].float().cpu().clamp(-1, 1)

    run_dir = OUTPUT_DIR / time.strftime("run_%Y%m%d_%H%M%S")
    pid_path = _save_image(pid_cpu, run_dir / "pid_output.jpg")
    vae_path = _save_image(vae_cpu, run_dir / "vae_baseline.jpg")
    compare_path = _make_compare_strip(input_cpu, vae_cpu, pid_cpu, run_dir / "compare_strip.jpg")

    log = (
        f"Preset: {preset}\n"
        f"Checkpoint: {preset_config['ckpt_type']}\n"
        f"Prompt: {caption}\n"
        f"Sigma: {float(degrade_sigma):.3f}\n"
        f"Keep input size: {keep_input_size}\n"
        f"VAE native size: {vae_w}x{vae_h}\n"
        f"PiD output size: {target_hw[1]}x{target_hw[0]}\n"
        f"Output folder: {run_dir}"
    )
    return pid_path, compare_path, vae_path, log
