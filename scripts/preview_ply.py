"""Render a turntable preview PNG from a Budget3D .ply (local helper, not part of the upstream repo).

The PLY writes *inverse*-activated values -- opacity as a sigmoid logit, scaling as a log, unit
quaternions, and a z-up -> y-up axis swap -- so they are undone here before the repo's own
rasterizer (budget3d.render.splat) sees them, exactly as gallery.py feeds it in-memory Gaussians.

    python scripts/preview_ply.py --ply outputs/character_male.ply --out outputs/character_male_view.png
"""
import argparse
import math
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from plyfile import PlyData

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from budget3d.render.splat import Splats, orbit_positions, render  # noqa: E402


def load_ply(path):
    """PLY vertex rows -> the activated Gaussian fields Splats expects, back in the z-up world."""
    v = PlyData.read(path)["vertex"].data
    col = lambda names: np.stack([v[n] for n in names], axis=1).astype(np.float32)  # noqa: E731

    xyz_up = col(["x", "y", "z"])                       # y-up: (X, Y, Z) = (x, -z, y)
    xyz = np.stack([xyz_up[:, 0], xyz_up[:, 2], -xyz_up[:, 1]], axis=1)   # -> z-up

    opacity = 1.0 / (1.0 + np.exp(-col(["opacity"])))    # undo inverse_sigmoid
    scaling = np.exp(col(["scale_0", "scale_1", "scale_2"]))             # undo log
    rot = col(["rot_0", "rot_1", "rot_2", "rot_3"])
    rot /= np.linalg.norm(rot, axis=1, keepdims=True)                    # keep unit quaternions

    # undo the +90 deg x rotation the writer pre-multiplied: q_ply = q90 * q_world
    q90 = np.array([math.sqrt(0.5), math.sqrt(0.5), 0.0, 0.0], dtype=np.float32)
    q90c = np.array([q90[0], -q90[1], -q90[2], -q90[3]], dtype=np.float32)
    aw, ax, ay, az = q90c
    bw, bx, by, bz = rot[:, 0], rot[:, 1], rot[:, 2], rot[:, 3]
    rot = np.stack([aw * bw - ax * bx - ay * by - az * bz,
                    aw * bx + ax * bw + ay * bz - az * by,
                    aw * by - ax * bz + ay * bw + az * bx,
                    aw * bz + ax * by - ay * bx + az * bw], axis=1)
    rot /= np.linalg.norm(rot, axis=1, keepdims=True)

    t = torch.from_numpy
    return SimpleNamespace(xyz=t(xyz), features_dc=t(col(["f_dc_0", "f_dc_1", "f_dc_2"]))[:, None, :],
                           opacity=t(opacity), scaling=t(scaling), rotation=t(rot))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ply", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--views", type=int, default=4)
    ap.add_argument("--radius", type=float, default=1.55)
    ap.add_argument("--fov", type=float, default=42.0)
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    g = load_ply(a.ply)
    g = SimpleNamespace(**{k: v.to(dev) for k, v in vars(g).items()})
    splats = Splats(g)
    print(f"[preview] {splats.n} visible gaussians of {len(g.opacity)} on {dev}")

    frames = []
    for i, c in enumerate(orbit_positions(0.5, 0.20, a.radius, a.views)):
        img = render(splats, c, a.fov, size=a.size)
        frames.append((img.cpu().numpy() * 255).round().astype(np.uint8))
        print(f"[preview] view {i + 1}/{a.views} done")

    cols = 2 if a.views == 4 else a.views
    rows = int(math.ceil(a.views / cols))
    tile = a.size
    sheet = Image.new("RGB", (cols * tile, rows * tile), (255, 255, 255))
    for i, f in enumerate(frames):
        sheet.paste(Image.fromarray(f), ((i % cols) * tile, (i // cols) * tile))
    sheet.save(a.out)
    print("[preview] wrote", a.out)


if __name__ == "__main__":
    main()
