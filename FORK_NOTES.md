# Fork notes — what is different from upstream

**Upstream:** [BudgetAny/Budget3D](https://github.com/BudgetAny/Budget3D) @
[`31bc02c`](https://github.com/BudgetAny/Budget3D/commit/31bc02c0a30b245fbccc74dc7f550e1945df0b84)
(*"README: download each repo's config.json with its weights"*, 2026-11-14)

**This fork:** adds a web interface and the packaging needed to get the pinned environment running.
The model, the method, the training code and the weights are untouched and remain the work of the
BudgetAny authors — see the Credits section of the README.

---

## Added

| File | What it is |
|---|---|
| `webui.py` | Gradio web UI. Wraps `load_budget3d` / `ImagePreprocessor` / `sample` / `decode_gaussians` (the same calls `scripts/sample.py` makes) and adds caching, the preview renderer, orbit video and downloads. |
| `webui.bat` | Launcher (`webui.bat [port]`), starts the UI and opens the browser. |
| `run_sample.bat` | CLI wrapper: sample → `.ply` → turntable PNG in one command. |
| `scripts/preview_ply.py` | Renders a `.ply` to a multi-view PNG using the repo's own `budget3d.render.splat` rasteriser. Inverts the PLY's stored activations first (opacity logit, log scale, quaternion normalisation, z-up→y-up swap), because the in-memory `Gaussian` those values came from is what `Splats` expects. |
| `setup.bat` | One-command Windows bootstrap: idol-sdk clone, patch, venv, CUDA torch, requirements, gradio, DINOv3 conversion, weight download. Idempotent. |
| `patches/idol-sdk-diffusers-dtype.patch` | Fixes Idol-SDK for the pinned diffusers (see below). |
| `docs/webui.png` | Screenshot used by the README. |

## Modified

| File | Change |
|---|---|
| `README.md` | Fork banner, credits, web-UI usage, change table, manual install, patch rationale. **Upstream README text is preserved verbatim** at the bottom, unedited. |
| `.gitignore` | Added `.venv/`, `outputs/`, `inputs/` and `*.mp4` so the 10 GB of local state cannot be committed. |

## Not modified

Every upstream source file is byte-identical to the base commit — `scripts/sample.py`,
`scripts/gallery.py`, `scripts/convert_dinov3_to_hf.py`, `budget3d/**`, `requirements.txt`,
`pyproject.toml`, `LICENSE`. Verified with `git diff 31bc02c --stat -- . ':!README.md' ':!.gitignore'`
returning nothing.

## The one fix outside the repo

`third_party/idol-sdk` is a **vendored clone, not a submodule** (it is in upstream's `.gitignore`), so a
fix inside it cannot be committed here. It is shipped as a patch instead.

Upstream Idol-SDK does:

```python
AutoencoderKLFlux2.from_pretrained(flux2_vae, subfolder='vae', dtype=torch.bfloat16)
```

`dtype=` was introduced in **diffusers 0.40**, but diffusers 0.40 requires
`huggingface_hub>=1.23` while Budget3D pins `transformers==4.57.6`, which requires
`huggingface_hub<1.0`. The constraints are mutually exclusive, so `diffusers==0.36.0` is the only
consistent choice — and it needs the pre-rename spelling `torch_dtype=`.

Symptoms without the patch, at the encoder stage of every run:

```
TypeError: AutoencoderKLFlux2.__init__() got an unexpected keyword argument 'dtype'
```

The patch selects the keyword from the installed diffusers version, so it stays correct if the pins
ever move. Apply with:

```bat
git apply patches/idol-sdk-diffusers-dtype.patch --directory=third_party/idol-sdk
```

## Environment deviations worth knowing

- **Python 3.12**, `torch==2.9.1+cu128`, `torchvision==0.24.1+cu128` — upstream pins torch 2.9.1/CUDA
  12.8, so the cu128 wheel index is the matching one. Python 3.14 does **not** work (no torch 2.9.x
  wheels); 3.10–3.12 do.
- **`gradio==5.49.1`** is an addition (upstream has no UI). It requires `pillow<12`, which pulls
  pillow from the pinned 12.1.1 down to 11.3.0. Re-verified end to end afterwards: sampling, decoding
  and PLY writing are unaffected.
- Idol's own weights, the FLUX.2 VAE and DINOv3 are fetched at runtime into the Hugging Face cache,
  not committed. DINOv3 ViT-H+/16 is converted from the ungated mirror the upstream script already
  selects (`Fanqi-Lin-IR/dinov3_vith16plus`), so no gated-repo access or `HF_TOKEN` is needed.

## What was verified

On an RTX 3090 (24 GB), Windows 11, Python 3.12.9:

- CLI: `scripts/sample.py` produced a 262,144-Gaussian `.ply` (17.8 MB) in 45 s at 50 steps / 50 %
  budget; rendered to a 4-view turntable to confirm the geometry is real and consistent from all sides.
- UI: driven both programmatically (Gradio client) and by real browser clicks — example → Generate →
  3D viewer, view grid, MP4 and downloads all populate; 26 s at 12 steps including a 72-frame orbit.
- Second budget (40 %) and a second subject to confirm the budget knob behaves as described.

Known limitations: the UI is single-concurrency (one GPU job at a time, by design), the Turbo path was
not exercised end to end (no Turbo checkpoint was downloaded during development — the guard for
mismatched checkpoints is the only Turbo-related code that ran), and `--fast` / `torch.compile` was
not benchmarked.
