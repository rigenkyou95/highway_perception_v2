# -*- coding: utf-8 -*-
"""
run_depth_anything_kitti.py

Run Depth Anything V2 Tiny on KITTI image_02 frames and save relative depth maps.

Output:
    outputs/kitti/depth_anything_v2/2011_09_26_drive_0005_sync/
        npy/*.npy
        vis/*.png
"""

import os
import sys
import argparse
from pathlib import Path

import cv2
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models.depth import DepthAnythingV2TinyWrapper


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def colorize_depth(depth: np.ndarray) -> np.ndarray:
    d = depth.astype(np.float32)
    valid = np.isfinite(d)

    vis = np.zeros((d.shape[0], d.shape[1], 3), dtype=np.uint8)
    if valid.sum() < 10:
        return vis

    vals = d[valid]
    vmin = np.percentile(vals, 2)
    vmax = np.percentile(vals, 98)
    vmax = max(vmax, vmin + 1e-6)

    norm = np.clip((d - vmin) / (vmax - vmin), 0, 1)
    gray = (norm * 255).astype(np.uint8)
    vis = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
    vis[~valid] = 0
    return vis


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--depth-ckpt", type=str, default=os.path.join(
        ROOT, "models", "ckpts", "depth_anything_v2", "depth_anything_v2_vits.pth"
    ))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-frames", type=int, default=-1)
    args = parser.parse_args()

    if not os.path.isdir(args.image_dir):
        raise FileNotFoundError(args.image_dir)

    if not os.path.isfile(args.depth_ckpt):
        raise FileNotFoundError(args.depth_ckpt)

    out_npy = os.path.join(args.out_dir, "npy")
    out_vis = os.path.join(args.out_dir, "vis")
    ensure_dir(out_npy)
    ensure_dir(out_vis)

    image_files = sorted(Path(args.image_dir).glob("*.png"))

    if args.max_frames > 0:
        image_files = image_files[:args.max_frames]

    print(f"[DA-V2] image files: {len(image_files)}")
    print(f"[DA-V2] ckpt       : {args.depth_ckpt}")
    print(f"[DA-V2] device     : {args.device}")

    model = DepthAnythingV2TinyWrapper(
        ckpt_path=args.depth_ckpt,
        device=args.device
    )

    for i, img_path in enumerate(image_files):
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            print(f"[Skip] Cannot read image: {img_path}")
            continue

        h, w = img.shape[:2]

        depth_rel = model(img)
        if not isinstance(depth_rel, np.ndarray):
            depth_rel = np.array(depth_rel, dtype=np.float32)

        depth_rel = depth_rel.astype(np.float32)

        if depth_rel.shape[:2] != (h, w):
            depth_rel = cv2.resize(depth_rel, (w, h), interpolation=cv2.INTER_LINEAR)

        stem = img_path.stem

        np.save(os.path.join(out_npy, f"{stem}.npy"), depth_rel)

        vis = colorize_depth(depth_rel)
        cv2.imwrite(os.path.join(out_vis, f"{stem}.png"), vis)

        if i % 10 == 0:
            print(
                f"[{i:04d}/{len(image_files)}] {stem} "
                f"rel_min={float(np.min(depth_rel)):.4f} "
                f"rel_med={float(np.median(depth_rel)):.4f} "
                f"rel_max={float(np.max(depth_rel)):.4f}"
            )

    print("=" * 60)
    print("Done.")
    print("Output:", args.out_dir)
    print("=" * 60)


if __name__ == "__main__":
    main()
