"""Budget3D web UI -- local helper, not part of the upstream repo.

Every sampling knob of ``scripts/sample.py`` (plus a few the CLI does not expose), the preview
rasteriser, and downloads, in one Gradio app.  Models, the image preprocessor and the decoders are
cached, so repeated runs skip the ~15 s of loading.

    .venv\\Scripts\\python.exe webui.py            # http://127.0.0.1:7860
    .venv\\Scripts\\python.exe webui.py --port 7000 --share
"""
import argparse
import math
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import gradio as gr
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from budget3d.models.load import decode_gaussians, idol_weights_dir, load_budget3d, load_decoders  # noqa: E402
from budget3d.render.splat import Splats, orbit_positions, render  # noqa: E402
from budget3d.sampling.sampler import ImagePreprocessor, sample  # noqa: E402

OUT_ROOT = ROOT / "outputs" / "webui"
DEV = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------- caches
_C = {"model": None, "model_key": None, "step": None, "fast": set(), "decoders": None, "pre": {}}


def checkpoints():
    """Every .safetensors in weights/, newest first (the dropdown's choices)."""
    found = sorted((ROOT / "weights").glob("*.safetensors"), key=lambda p: p.stat().st_mtime, reverse=True)
    return [str(p.relative_to(ROOT)) for p in found] or ["weights/budget3d_idol.safetensors"]


def get_model(ckpt, turbo, lora, raw, fast, log):
    key = (ckpt, turbo, lora or None, raw)
    if _C["model_key"] != key:
        if _C["model"] is not None:                      # free the previous checkpoint before loading the next
            _C["model"] = None
            torch.cuda.empty_cache()
        t = time.time()
        log(f"loading `{ckpt}` (turbo={turbo}, {'raw' if raw else 'EMA'} weights) ...")
        model, step = load_budget3d(ckpt, turbo=turbo, lora=lora or None, device=DEV,
                                    ema=not raw, weights_dir=idol_weights_dir())
        _C.update(model=model, model_key=key, step=step)
        log(f"loaded in {time.time() - t:.1f}s (step {step})")
    model = _C["model"]
    if fast and key not in _C["fast"]:
        log("optimising for inference (bf16 blocks, fused attention, torch.compile -- first call compiles)")
        from budget3d.sampling.fast import optimize_for_inference
        optimize_for_inference(model)
        _C["fast"].add(key)
    return model


def get_pre(mode):
    if mode not in _C["pre"]:
        _C["pre"][mode] = ImagePreprocessor(idol_weights_dir(), DEV, mode)
    return _C["pre"][mode]


def get_decoders():
    if _C["decoders"] is None:
        _C["decoders"] = load_decoders(idol_weights_dir(), DEV)
    return _C["decoders"]


# ---------------------------------------------------------------- media helpers
def write_mp4(frames, path, fps):
    """Raw RGB frames -> H.264 MP4 through imageio-ffmpeg's bundled binary (no torchvision needed)."""
    try:
        import imageio_ffmpeg
        exe = os.environ.get("FFMPEG") or imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        exe = "ffmpeg"
    h, w = frames[0].shape[:2]
    cmd = [exe, "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-movflags", "+faststart", str(path)]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for f in frames:
            p.stdin.write(f.tobytes())
        p.stdin.close()
        return p.wait() == 0
    except Exception:
        p.kill()
        return False


def grid_image(frames, captions, cols, header):
    """Views laid out in a labelled grid on white."""
    tile = frames[0].shape[0]
    rows = int(math.ceil(len(frames) / cols))
    bar, pad = 26, 8
    W = cols * tile + pad * (cols + 1)
    H = header_h = 34 if header else 0
    H += rows * (tile + bar) + pad * (rows + 1)
    sheet = Image.new("RGB", (W, H), (250, 250, 250))
    d = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("segoeui.ttf", 15)
    except Exception:
        font = ImageFont.load_default()
    if header:
        d.text((pad, 9), header, fill=(20, 20, 20), font=font)
    y = header_h
    for i, fr in enumerate(frames):
        x = pad + (i % cols) * (tile + pad)
        yy = y + (i // cols) * (tile + bar + pad) + pad
        sheet.paste(Image.fromarray(fr), (x, yy))
        d.text((x + 2, yy + tile + 4), captions[i], fill=(60, 60, 60), font=font)
    return sheet


# ---------------------------------------------------------------- the pipeline
def generate(image, ckpt, budget, steps, cfg, rescale_t, mode, num_gaussians, seed, randomize,
             warmup, relayout_every, relayout_until, raw, turbo, lora, fast,
             pv_size, pv_views, pv_pitch, pv_yaw, pv_radius, pv_fov, pv_bg,
             make_orbit, orbit_frames, orbit_fps, progress=gr.Progress()):
    """One image -> PLY + rendered views (+ orbit MP4). Returns the components' new values."""
    lines = []

    def log(msg):
        lines.append(msg)
        print("[webui]", msg, flush=True)

    if image is None:
        raise gr.Error("Upload an image first (or pick one of the examples).")

    # Turbo injects Idol-turbo's phase-LoRA experts and expects turbo-tuned weights on top; loading a
    # plain Idol checkpoint that way is silently meaningless, so refuse it instead.
    if not Path(ckpt).exists():
        raise gr.Error(f"Checkpoint not found: {ckpt}")
    if turbo and "turbo" not in Path(ckpt).name.lower():
        raise gr.Error("Turbo needs a Budget3D-Idol-Turbo checkpoint (e.g. weights/budget3d_idol_turbo.safetensors), "
                       "not a plain Idol one. Download it with:\n\n"
                       "hf download BudgetAny/Budget3D-Idol-Turbo budget3d_idol_turbo.safetensors --local-dir weights")
    if fast and DEV != "cuda":
        raise gr.Error("Fast inference needs CUDA.")

    t0 = time.time()
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = OUT_ROOT / stamp
    run_dir.mkdir(parents=True, exist_ok=True)
    stem = "result"

    progress(0.02, desc="preparing image")
    pil = image if isinstance(image, Image.Image) else Image.open(image)
    pil = pil.convert("RGBA")
    src_path = run_dir / "input.png"
    pil.save(src_path)

    progress(0.10, desc="loading model")
    model = get_model(ckpt, turbo, lora, raw, fast, log)

    progress(0.20, desc="encoding image (BiRefNet + DINOv3 + FLUX.2 VAE)")
    t = time.time()
    prepared, cond = get_pre(mode)(str(src_path))
    prepared.save(run_dir / "prepared.png")
    log(f"encoded in {time.time() - t:.1f}s (Idol framing + frozen encoders)")

    if randomize:
        seed = int(torch.randint(0, 2 ** 31 - 1, (1,)).item())
    steps = int(steps) or None                     # 0 = auto: 50, or 5 with turbo
    log(f"sampling: budget {budget:.1%}, steps {steps or ('5' if turbo else '50')}, "
        f"cfg {cfg}, time shift {rescale_t}, seed {seed}")

    progress(0.25, desc="sampling (this is the slow part)")
    t = time.time()
    with torch.no_grad():
        if turbo:
            from budget3d.sampling.turbo import sample_turbo
            latent, camera, layout = sample_turbo(model, cond, float(budget), steps or 5,
                                                  float(rescale_t), seed=int(seed), device=DEV)
        else:
            latent, camera, layout = sample(model, cond, float(budget), steps or 50, float(cfg),
                                            float(rescale_t), seed=int(seed), device=DEV,
                                            warmup_steps=int(warmup),
                                            relayout_every=int(relayout_every),
                                            relayout_until_step=int(relayout_until))
    log(f"sampled in {time.time() - t:.1f}s -> {layout!r}")

    progress(0.70, desc="decoding Gaussians")
    t = time.time()
    with torch.no_grad():
        dec, dec_gs, cfg_json = get_decoders()
        gauss = decode_gaussians(latent, dec, dec_gs, cfg_json, int(num_gaussians))
    ply_path = run_dir / f"{stem}_b{int(round(budget * 100))}.ply"
    gauss.save_ply(str(ply_path))
    log(f"decoded + wrote {ply_path.name} in {time.time() - t:.1f}s "
        f"({ply_path.stat().st_size / 1e6:.1f} MB)")

    progress(0.82, desc="rendering preview")
    t = time.time()
    sp = Splats(gauss)
    log(f"rendering {int(pv_views)} views of {sp.n:,} visible Gaussians on {DEV}")
    frames, caps = [], []
    for i, c in enumerate(orbit_positions(float(pv_yaw), float(pv_pitch), float(pv_radius), int(pv_views))):
        img = render(sp, c, float(pv_fov), size=int(pv_size), bg=1.0 if pv_bg == "white" else 0.0)
        frames.append((img.cpu().numpy() * 255).round().astype(np.uint8))
        caps.append(f"view {i + 1}/{int(pv_views)}  yaw {math.degrees(float(pv_yaw) + 2 * math.pi * i / int(pv_views)):.0f} deg")
    header = (f"Budget3D  |  budget {budget:.0%}  |  {layout.n_tokens} tokens  |  "
              f"{int(num_gaussians):,} Gaussians  |  seed {seed}  |  {Path(ckpt).name}")
    grid = grid_image(frames, caps, 2 if len(frames) > 1 else 1, header)
    grid_path = run_dir / f"{stem}_views.png"
    grid.save(grid_path)
    log(f"rendered in {time.time() - t:.1f}s")

    video = None
    if make_orbit:
        progress(0.92, desc="rendering orbit")
        t = time.time()
        n = int(orbit_frames)
        orb = [(render(sp, c, float(pv_fov), size=int(pv_size)).cpu().numpy() * 255).round().astype(np.uint8)
               for c in orbit_positions(float(pv_yaw), float(pv_pitch), float(pv_radius), n)]
        video = run_dir / f"{stem}_orbit.mp4"
        if write_mp4(orb, video, int(orbit_fps)):
            log(f"orbit: {n} frames -> {video.name} in {time.time() - t:.1f}s")
        else:
            log("orbit: ffmpeg failed, skipped")
            video = None

    log(f"total {time.time() - t0:.1f}s")
    progress(1.0, desc="done")
    status = "```\n" + "\n".join(lines) + "\n```"
    files = [str(p) for p in (ply_path, grid_path, run_dir / "prepared.png", src_path, video) if p and Path(p).exists()]
    return str(ply_path), [(str(grid_path), f"budget {budget:.0%}"), (str(run_dir / 'prepared.png'), "prepared input")], \
        (str(video) if video else None), files, status


# ---------------------------------------------------------------- UI
CSS = """
#gen { min-height: 46px; font-weight: 600; }
#status code, #status pre { font-size: 12.5px; line-height: 1.45; }
footer { display: none !important; }
"""

with gr.Blocks(title="Budget3D", theme=gr.themes.Soft(primary_hue="red"), css=CSS) as demo:
    gr.Markdown(
        "# Budget3D — image to 3D, at any token budget\n"
        "One image in, Gaussian splats out. Every setting from `scripts/sample.py` is exposed below; "
        f"running on **{DEV}**" + (f" ({torch.cuda.get_device_name(0)})" if DEV == "cuda" else "") + "."
    )

    with gr.Row(equal_height=False):
        with gr.Column(scale=5):
            image = gr.Image(type="pil", image_mode="RGBA", label="Input image", height=320,
                             sources=["upload", "clipboard"])
            with gr.Row():
                ckpt = gr.Dropdown(checkpoints(), value=checkpoints()[0], label="Checkpoint",
                                   allow_custom_value=True, scale=4)
                turbo = gr.Checkbox(value=False, label="Turbo", scale=1,
                                    info="Idol-Turbo checkpoint: 5 guidance-free steps")
            run = gr.Button("Generate", variant="primary", elem_id="gen")

            with gr.Accordion("Sampling settings", open=True):
                budget = gr.Slider(0.0625, 1.0, value=0.5, step=0.0625, label="Token budget",
                                   info="Fraction of the 8,192 dense tokens (0.0625-1.0)")
                with gr.Row():
                    steps = gr.Slider(0, 100, value=0, step=1, label="Steps",
                                      info="0 = auto (50, or 5 with Turbo)")
                    num_gaussians = gr.Dropdown([65536, 131072, 262144, 524288], value=262144,
                                                label="Gaussians")
                with gr.Row():
                    cfg = gr.Slider(0.0, 10.0, value=3.0, step=0.1, label="CFG scale",
                                    info="ignored with Turbo")
                    rescale_t = gr.Slider(1.0, 6.0, value=3.0, step=0.1, label="Time shift")
                with gr.Row():
                    mode = gr.Radio(["synthetic", "real"], value="synthetic", label="Input type",
                                    info="synthetic = renders / generated images, real = photographs")
                    seed = gr.Number(value=42, precision=0, label="Seed")
                    randomize = gr.Checkbox(value=False, label="Randomise")

            with gr.Accordion("Advanced sampling", open=False):
                with gr.Row():
                    warmup = gr.Slider(0, 20, value=0, step=1, label="Dense warm-up steps",
                                       info="Euler steps at the full 8,192 layout before the budget applies")
                    relayout_every = gr.Slider(1, 20, value=5, step=1, label="Re-layout every N steps")
                    relayout_until = gr.Slider(-1, 40, value=5, step=1,
                                               label="Re-layout until step (-1 = always)")
                with gr.Row():
                    raw = gr.Checkbox(value=False, label="Raw weights (not EMA)")
                    fast = gr.Checkbox(value=False, label="Fast inference (bf16 + torch.compile)",
                                       info="first run pays a compile")
                    lora = gr.Textbox(value="", label="Turbo LoRA path (optional)", scale=2)

            with gr.Accordion("Preview render", open=False):
                with gr.Row():
                    pv_views = gr.Slider(1, 12, value=4, step=1, label="Views in grid")
                    pv_size = gr.Dropdown([256, 384, 448, 512, 640, 768], value=512, label="Resolution")
                    pv_bg = gr.Radio(["white", "black"], value="white", label="Background")
                with gr.Row():
                    pv_yaw = gr.Slider(-3.2, 3.2, value=0.5, step=0.05, label="Start yaw (rad)")
                    pv_pitch = gr.Slider(-1.4, 1.4, value=0.20, step=0.05, label="Pitch (rad)")
                    pv_radius = gr.Slider(0.8, 4.0, value=1.55, step=0.05, label="Camera radius")
                    pv_fov = gr.Slider(10, 90, value=42, step=1, label="FOV (deg)")
                with gr.Row():
                    make_orbit = gr.Checkbox(value=True, label="Render orbit video (MP4)")
                    orbit_frames = gr.Slider(16, 240, value=72, step=8, label="Orbit frames")
                    orbit_fps = gr.Slider(8, 60, value=24, step=1, label="FPS")

        with gr.Column(scale=7):
            model3d = gr.Model3D(label="3D Gaussians (.ply) — drag to orbit", height=420,
                                 clear_color=[1.0, 1.0, 1.0, 1.0])
            with gr.Tabs():
                with gr.Tab("Rendered views"):
                    gallery = gr.Gallery(label="Rendered from the generated splats", columns=2,
                                         height=380, object_fit="contain")
                with gr.Tab("Orbit"):
                    video = gr.Video(label="Turntable", height=380, autoplay=True, loop=True)
                with gr.Tab("Downloads"):
                    files = gr.File(label="Output files", file_count="multiple")
            status = gr.Markdown("Ready.", elem_id="status")

    run.click(
        generate,
        inputs=[image, ckpt, budget, steps, cfg, rescale_t, mode, num_gaussians, seed, randomize,
                warmup, relayout_every, relayout_until, raw, turbo, lora, fast,
                pv_size, pv_views, pv_pitch, pv_yaw, pv_radius, pv_fov, pv_bg,
                make_orbit, orbit_frames, orbit_fps],
        outputs=[model3d, gallery, video, files, status],
    )

    gr.Examples(
        examples=[[p] for p in
                  sorted((ROOT / "third_party/idol-sdk/demo/examples").glob("*.webp"))[:10]
                  + sorted((ROOT / "inputs").glob("*.png"))],
        inputs=[image], label="Examples (Idol's demo set, plus the local test images)",
    )
    gr.Markdown("Local helper app — upstream Budget3D ships only `scripts/sample.py` and `scripts/gallery.py`.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--share", action="store_true")
    ap.add_argument("--inbrowser", action="store_true")
    a = ap.parse_args()
    demo.queue(default_concurrency_limit=1).launch(server_name=a.host, server_port=a.port,
                                                   share=a.share, inbrowser=a.inbrowser, show_error=True)
