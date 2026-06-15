# -*- coding: utf-8 -*-
"""
kitti_lidar_to_depth.py

Convert KITTI raw Velodyne point clouds into sparse depth maps on image_02.

Important:
    This version matches image and Velodyne files by filename stem.
    It avoids frame misalignment when some Velodyne .bin files are missing.

Example:
    image_02/data/0000000181.png
    velodyne_points/data/0000000181.bin

Input:
    KITTI raw date directory:
        E:/detected sys/dataset/KITTI/raw/2011_09_26

    KITTI raw drive directory:
        E:/detected sys/dataset/KITTI/raw/2011_09_26/2011_09_26_drive_0005_sync

Output:
    outputs/kitti/gt_depth/2011_09_26_drive_xxxx_sync/
        npy/*.npy       float32 sparse depth, meter
        png/*.png       uint16 sparse depth, depth*256
        vis/*.png       colored quick visualization
"""

import argparse
import os
from pathlib import Path

import cv2
import numpy as np


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def read_calib_file(path: str) -> dict:
    data = {}

    with open(path, "r", encoding="utf-8") as f:
        for line in f.readlines():
            if ":" not in line:
                continue

            key, value = line.split(":", 1)
            value = value.strip()

            if len(value) == 0:
                continue

            try:
                data[key] = np.array([float(x) for x in value.split()], dtype=np.float64)
            except ValueError:
                data[key] = value

    return data


def load_kitti_calib(date_dir: str):
    cam_path = os.path.join(date_dir, "calib_cam_to_cam.txt")
    velo_path = os.path.join(date_dir, "calib_velo_to_cam.txt")

    if not os.path.isfile(cam_path):
        raise FileNotFoundError(cam_path)
    if not os.path.isfile(velo_path):
        raise FileNotFoundError(velo_path)

    cam = read_calib_file(cam_path)
    velo = read_calib_file(velo_path)

    if "P_rect_02" not in cam:
        raise KeyError("P_rect_02 not found in calib_cam_to_cam.txt")
    if "R_rect_00" not in cam:
        raise KeyError("R_rect_00 not found in calib_cam_to_cam.txt")
    if "R" not in velo or "T" not in velo:
        raise KeyError("R or T not found in calib_velo_to_cam.txt")

    # Projection matrix for left color camera image_02
    P2 = cam["P_rect_02"].reshape(3, 4)

    # Rectification matrix
    R_rect = np.eye(4, dtype=np.float64)
    R_rect[:3, :3] = cam["R_rect_00"].reshape(3, 3)

    # Velodyne to camera transform
    Tr = np.eye(4, dtype=np.float64)
    Tr[:3, :3] = velo["R"].reshape(3, 3)
    Tr[:3, 3] = velo["T"].reshape(3)

    return P2, R_rect, Tr


def load_velodyne_bin(path: str) -> np.ndarray:
    points = np.fromfile(path, dtype=np.float32)

    if points.size % 4 != 0:
        raise ValueError(f"Invalid Velodyne file size: {path}")

    points = points.reshape(-1, 4)
    return points


def project_velodyne_to_image(
    points_velo: np.ndarray,
    P2: np.ndarray,
    R_rect: np.ndarray,
    Tr_velo_to_cam: np.ndarray,
    img_h: int,
    img_w: int,
) -> np.ndarray:
    """
    Project Velodyne points to image_02 plane.

    Returns:
        depth: sparse depth map.
        Depth value is rectified camera Z in meters.
    """
    if points_velo.shape[0] == 0:
        return np.zeros((img_h, img_w), dtype=np.float32)

    # Keep points in front of Velodyne sensor.
    points_velo = points_velo[points_velo[:, 0] > 0]

    if points_velo.shape[0] == 0:
        return np.zeros((img_h, img_w), dtype=np.float32)

    pts_h = np.ones((points_velo.shape[0], 4), dtype=np.float64)
    pts_h[:, :3] = points_velo[:, :3].astype(np.float64)

    # Velodyne -> camera -> rectified camera
    pts_cam = (R_rect @ Tr_velo_to_cam @ pts_h.T).T

    # Valid camera depth
    z = pts_cam[:, 2]
    valid_z = z > 0

    pts_cam = pts_cam[valid_z]
    z = z[valid_z]

    if pts_cam.shape[0] == 0:
        return np.zeros((img_h, img_w), dtype=np.float32)

    # Project to image
    pts_img_h = (P2 @ pts_cam.T).T

    u = pts_img_h[:, 0] / pts_img_h[:, 2]
    v = pts_img_h[:, 1] / pts_img_h[:, 2]

    u_round = np.round(u).astype(np.int32)
    v_round = np.round(v).astype(np.int32)

    valid = (
        (u_round >= 0) & (u_round < img_w) &
        (v_round >= 0) & (v_round < img_h) &
        np.isfinite(z)
    )

    u_round = u_round[valid]
    v_round = v_round[valid]
    z = z[valid]

    depth = np.zeros((img_h, img_w), dtype=np.float32)

    if len(z) == 0:
        return depth

    # If multiple points hit the same pixel, keep the nearest one.
    # Write far points first, then near points overwrite them.
    order = np.argsort(z)[::-1]

    u_round = u_round[order]
    v_round = v_round[order]
    z = z[order]

    depth[v_round, u_round] = z.astype(np.float32)

    return depth


def colorize_sparse_depth(depth: np.ndarray) -> np.ndarray:
    valid = depth > 0

    vis = np.zeros((depth.shape[0], depth.shape[1], 3), dtype=np.uint8)

    if valid.sum() == 0:
        return vis

    vals = depth[valid]

    vmin = np.percentile(vals, 2)
    vmax = np.percentile(vals, 98)
    vmax = max(vmax, vmin + 1e-6)

    norm = np.zeros_like(depth, dtype=np.float32)
    norm[valid] = np.clip((depth[valid] - vmin) / (vmax - vmin), 0, 1)

    gray = (norm * 255).astype(np.uint8)
    vis = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
    vis[~valid] = 0

    return vis


def collect_matched_files(image_dir: str, velo_dir: str, max_frames: int = -1):
    image_dict = {p.stem: p for p in Path(image_dir).glob("*.png")}
    velo_dict = {p.stem: p for p in Path(velo_dir).glob("*.bin")}

    image_stems = set(image_dict.keys())
    velo_stems = set(velo_dict.keys())

    matched_stems = sorted(image_stems & velo_stems)
    missing_velo = sorted(image_stems - velo_stems)
    missing_image = sorted(velo_stems - image_stems)

    print(f"[KITTI] image files: {len(image_stems)}")
    print(f"[KITTI] velo files : {len(velo_stems)}")
    print(f"[KITTI] matched    : {len(matched_stems)}")
    print(f"[KITTI] image without velo: {len(missing_velo)}")
    print(f"[KITTI] velo without image: {len(missing_image)}")

    if len(missing_velo) > 0:
        print("[KITTI] first missing velo stems:", missing_velo[:20])

    if len(missing_image) > 0:
        print("[KITTI] first missing image stems:", missing_image[:20])

    if max_frames > 0:
        matched_stems = matched_stems[:max_frames]
        print(f"[KITTI] max_frames applied: {len(matched_stems)}")

    if len(matched_stems) <= 0:
        raise RuntimeError("No matched image/velodyne files found.")

    return image_dict, velo_dict, matched_stems, missing_velo, missing_image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--date-dir",
        required=True,
        help="KITTI date directory, e.g. .../2011_09_26",
    )
    parser.add_argument(
        "--drive-dir",
        required=True,
        help="KITTI sync drive directory",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        help="Output directory for sparse depth maps.",
    )
    parser.add_argument(
        "--camera",
        default="image_02",
        help="Camera folder name. Default: image_02",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=-1,
        help="Process only the first N matched frames. Default: all matched frames.",
    )

    args = parser.parse_args()

    image_dir = os.path.join(args.drive_dir, args.camera, "data")
    velo_dir = os.path.join(args.drive_dir, "velodyne_points", "data")

    if not os.path.isdir(image_dir):
        raise FileNotFoundError(image_dir)

    if not os.path.isdir(velo_dir):
        raise FileNotFoundError(velo_dir)

    out_npy = os.path.join(args.out_dir, "npy")
    out_png = os.path.join(args.out_dir, "png")
    out_vis = os.path.join(args.out_dir, "vis")

    ensure_dir(out_npy)
    ensure_dir(out_png)
    ensure_dir(out_vis)

    P2, R_rect, Tr_velo_to_cam = load_kitti_calib(args.date_dir)

    image_dict, velo_dict, matched_stems, missing_velo, missing_image = collect_matched_files(
        image_dir=image_dir,
        velo_dir=velo_dir,
        max_frames=args.max_frames,
    )

    for i, stem in enumerate(matched_stems):
        img_path = str(image_dict[stem])
        velo_path = str(velo_dict[stem])

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)

        if img is None:
            print(f"[Skip] Cannot read image: {img_path}")
            continue

        img_h, img_w = img.shape[:2]

        try:
            points = load_velodyne_bin(velo_path)
        except Exception as e:
            print(f"[Skip] Cannot read velodyne: {velo_path}, error={e}")
            continue

        depth = project_velodyne_to_image(
            points_velo=points,
            P2=P2,
            R_rect=R_rect,
            Tr_velo_to_cam=Tr_velo_to_cam,
            img_h=img_h,
            img_w=img_w,
        )

        np.save(os.path.join(out_npy, f"{stem}.npy"), depth)

        # KITTI-style uint16 png, depth in meters * 256
        depth_png = np.clip(depth * 256.0, 0, 65535).astype(np.uint16)
        cv2.imwrite(os.path.join(out_png, f"{stem}.png"), depth_png)

        vis = colorize_sparse_depth(depth)
        overlay = cv2.addWeighted(img, 0.65, vis, 0.35, 0)
        cv2.imwrite(os.path.join(out_vis, f"{stem}.png"), overlay)

        if i % 10 == 0:
            valid = int((depth > 0).sum())
            med = float(np.median(depth[depth > 0])) if valid > 0 else 0.0

            print(
                f"[{i:04d}/{len(matched_stems)}] "
                f"{stem} valid={valid} med_depth={med:.2f}m"
            )

    print("=" * 60)
    print("Done.")
    print("Output:", args.out_dir)
    print(f"Processed matched frames: {len(matched_stems)}")
    print(f"Skipped missing velodyne frames: {len(missing_velo)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
