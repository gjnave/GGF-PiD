from __future__ import annotations

import argparse
import os
import socket
from pathlib import Path

import gradio as gr

from pid_runtime import decode_image, model_status, unload_model


ROOT = Path(__file__).resolve().parents[1]
PORT_TRIES = 20

CSS = """
:root {
    --ggf-ink: #131517;
    --ggf-paper: #fbf7ee;
    --ggf-sand: #efe4d0;
    --ggf-red: #d2462f;
    --ggf-teal: #145d69;
    --ggf-gold: #d8a33d;
    --ggf-line: #d7ccbb;
    --ggf-muted: #5d625f;
}
body, .gradio-container {
    background:
        radial-gradient(circle at top left, rgba(216, 163, 61, 0.10), transparent 28%),
        linear-gradient(135deg, rgba(210, 70, 47, 0.08), transparent 34%),
        linear-gradient(315deg, rgba(20, 93, 105, 0.12), transparent 44%),
        #fbf7ee !important;
    color: var(--ggf-ink);
    font-family: "Segoe UI", Arial, sans-serif;
}
.gradio-container {
    max-width: none !important;
    width: 100% !important;
    padding: 18px 24px !important;
}
.brand-hero {
    border: 1px solid rgba(19, 21, 23, 0.12);
    border-radius: 8px;
    padding: 20px 24px;
    margin-bottom: 12px;
    background:
        linear-gradient(120deg, rgba(19, 21, 23, 0.96), rgba(20, 93, 105, 0.90)),
        repeating-linear-gradient(90deg, rgba(255,255,255,0.08) 0, rgba(255,255,255,0.08) 1px, transparent 1px, transparent 18px);
    color: #fff8ec;
    box-shadow: 0 18px 44px rgba(19, 21, 23, 0.16);
}
.brand-lockup {
    display: inline-flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 8px;
}
.brand-badge {
    display: inline-flex;
    width: 44px;
    height: 44px;
    align-items: center;
    justify-content: center;
    border-radius: 8px;
    background: var(--ggf-red);
    color: #fff8ec;
    font-weight: 900;
    box-shadow: inset 0 -4px 0 rgba(0, 0, 0, 0.18);
}
.brand-kicker {
    color: #f4ca6b;
    font-weight: 800;
    margin: 0;
}
.brand-hero h1 {
    margin: 0;
    font-size: 34px;
    line-height: 1.02;
    color: #fff8ec !important;
}
.brand-copy {
    max-width: 980px;
    color: #f1e8d9;
    font-size: 15px;
    margin: 10px 0 8px;
}
.brand-chips {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-top: 14px;
}
.brand-chip {
    border: 1px solid rgba(255, 248, 236, 0.22);
    border-radius: 999px;
    padding: 5px 10px;
    color: #fff8ec;
    background: rgba(255, 248, 236, 0.08);
    font-size: 13px;
}
.panel {
    border: 1px solid var(--ggf-line);
    border-radius: 8px;
    padding: 12px;
    background: rgba(255, 252, 245, 0.92);
}
.main-grid {
    gap: 14px !important;
    align-items: flex-start !important;
}
.control-panel {
    max-width: 560px;
}
.output-panel {
    min-width: 720px;
}
.output-grid {
    gap: 14px !important;
}
.primary-output img,
.secondary-output img {
    object-fit: contain !important;
}
.status-box textarea {
    min-height: 180px !important;
}
.run-button button, .run-button, button.primary, .gradio-button.primary {
    background: linear-gradient(180deg, #e4553d, var(--ggf-red)) !important;
    border-color: #b33824 !important;
    color: #fff8ec !important;
    font-weight: 900 !important;
    box-shadow: 0 12px 22px rgba(210, 70, 47, 0.22) !important;
}
.utility-button button, .utility-button {
    border: 1px solid #cfc3af !important;
    background: #fffaf0 !important;
    color: var(--ggf-ink) !important;
    font-weight: 800 !important;
}
.status-box textarea {
    font-family: Consolas, monospace !important;
    font-size: 12px !important;
}
"""


def build_app() -> gr.Blocks:
    with gr.Blocks(title="GGF PiD", analytics_enabled=False) as app:
        gr.HTML(
            "<div class='brand-hero'>"
            "<div class='brand-lockup'><div class='brand-badge'>GGF</div>"
            "<p class='brand-kicker'>GET GOING FAST</p></div>"
            "<h1>PiD Decoder Studio</h1>"
            "<p class='brand-copy'>Standalone local PiD decoding for Flux and Z-Image compatible latents. "
            "Feed a source image, condition it with a caption, and compare native VAE decode against PiD super-res output.</p>"
            "<div class='brand-chips'>"
            "<span class='brand-chip'>Official nvidia/PiD weights</span>"
            "<span class='brand-chip'>Flux / Z-Image VAE path</span>"
            "<span class='brand-chip'>2K or 4K decode presets</span>"
            "<span class='brand-chip'>Local browser UI</span>"
            "</div>"
            "</div>"
        )

        with gr.Row(elem_classes=["main-grid"]):
            with gr.Column(scale=5, min_width=420, elem_classes=["control-panel"]):
                with gr.Group(elem_classes=["panel"]):
                    input_image = gr.Image(label="Input image", type="filepath", height=470)
                    prompt = gr.Textbox(label="Prompt", lines=3, placeholder="Describe the image for PiD caption conditioning.")
                    preset = gr.Radio(
                        choices=["2K Decode", "4K Decode"],
                        value="2K Decode",
                        label="Decode preset",
                    )
                    with gr.Row():
                        degrade_sigma = gr.Slider(0.0, 1.0, value=0.0, step=0.05, label="Latent noise sigma")
                        keep_input_size = gr.Checkbox(value=False, label="Keep input size")
                    with gr.Row():
                        cfg_scale = gr.Slider(0.5, 3.0, value=1.0, step=0.1, label="CFG scale")
                        pid_steps = gr.Slider(1, 8, value=4, step=1, label="PiD steps")
                        seed = gr.Number(value=5, precision=0, label="Seed")
                    with gr.Row():
                        run_button = gr.Button("Decode with PiD", variant="primary", size="lg", elem_classes=["run-button"])
                        unload_button = gr.Button("Unload model", elem_classes=["utility-button"])
                        refresh_button = gr.Button("Refresh status", elem_classes=["utility-button"])

            with gr.Column(scale=14, min_width=720, elem_classes=["output-panel"]):
                with gr.Row(elem_classes=["output-grid"]):
                    with gr.Column(scale=8, min_width=520):
                        pid_output = gr.Image(
                            label="PiD output",
                            type="filepath",
                            height=710,
                            elem_classes=["primary-output"],
                        )
                    with gr.Column(scale=6, min_width=430):
                        compare_strip = gr.Image(
                            label="Comparison strip",
                            type="filepath",
                            height=255,
                            elem_classes=["secondary-output"],
                        )
                        vae_output = gr.Image(
                            label="VAE baseline",
                            type="filepath",
                            height=255,
                            elem_classes=["secondary-output"],
                        )
                        status_box = gr.Textbox(
                            label="Status / log",
                            value=model_status(),
                            lines=9,
                            elem_classes=["status-box"],
                        )

        run_button.click(
            fn=decode_image,
            inputs=[input_image, prompt, preset, degrade_sigma, keep_input_size, cfg_scale, pid_steps, seed],
            outputs=[pid_output, compare_strip, vae_output, status_box],
        )
        refresh_button.click(fn=model_status, inputs=[], outputs=[status_box])
        unload_button.click(fn=unload_model, inputs=[], outputs=[status_box])

    return app


def _find_open_port(host: str, preferred_port: int) -> int:
    bind_host = host
    if host in ("0.0.0.0", "::"):
        bind_host = "127.0.0.1"

    for port in range(preferred_port, preferred_port + PORT_TRIES):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind((bind_host, port))
            except OSError:
                continue
            return port

    raise RuntimeError(f"No open port found from {preferred_port} to {preferred_port + PORT_TRIES - 1}.")


def main() -> None:
    parser = argparse.ArgumentParser(description="GGF PiD standalone app")
    parser.add_argument("--host", default=os.environ.get("GGF_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("GGF_PORT", "7868")))
    args = parser.parse_args()

    ROOT.joinpath("outputs").mkdir(exist_ok=True)
    port = _find_open_port(args.host, args.port)
    if port != args.port:
        print(f"[PORT] {args.port} is busy. Using {port} instead.")

    build_app().queue(default_concurrency_limit=1).launch(
        server_name=args.host,
        server_port=port,
        inbrowser=True,
        theme=gr.themes.Soft(primary_hue="red", neutral_hue="stone"),
        css=CSS,
        footer_links=[],
    )


if __name__ == "__main__":
    main()
