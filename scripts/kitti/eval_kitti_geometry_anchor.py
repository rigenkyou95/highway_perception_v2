# -*- coding: utf-8 -*-
"""
eval_kitti_geometry_anchor.py

KITTI geometry-anchor scale candidate evaluation.

Pipeline:
1. Load KITTI image_02 frames.
2. Run TwinLiteNet++ lane/drivable segmentation.
3. Sample road support points.
4. Convert support points to sparse geometric metric anchors.
5. Align Z_geo with Depth Anything V2 relative depth to get s_candidate.
6. Apply temporal state machine.
7. Evaluate metric depth against KITTI sparse LiDAR GT.

This is the first runnable version for KITTI geometry-anchor validation.
"""

import os
import sys
import argparse
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.infer.test_full_system_video_scale_stable import build_segmentation_runner


# =========================================================
# Basic utilities
# =========================================================

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def now_ms() -> float:
    """High-resolution wall-clock timer in milliseconds."""
    return time.perf_counter() * 1000.0


def safe_fps(time_ms: float) -> float:
    """Convert milliseconds per frame to FPS safely."""
    if time_ms is None or not np.isfinite(time_ms) or time_ms <= 0:
        return np.nan
    return 1000.0 / float(time_ms)


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


def load_kitti_intrinsics(date_dir: str) -> Dict[str, float]:
    cam_path = os.path.join(date_dir, "calib_cam_to_cam.txt")
    if not os.path.isfile(cam_path):
        raise FileNotFoundError(cam_path)

    cam = read_calib_file(cam_path)
    P2 = cam["P_rect_02"].reshape(3, 4)

    return {
        "fx": float(P2[0, 0]),
        "fy": float(P2[1, 1]),
        "cx": float(P2[0, 2]),
        "cy": float(P2[1, 2]),
    }


def normalize_mask(mask: np.ndarray) -> np.ndarray:
    if mask is None:
        return None
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    if mask.max() <= 1:
        mask = mask * 255
    return mask


def to_float_depth(path: str) -> np.ndarray:
    return np.load(path).astype(np.float32)


def eigen_crop_mask(h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    y1 = int(0.40810811 * h)
    y2 = int(0.99189189 * h)
    x1 = int(0.03594771 * w)
    x2 = int(0.96405229 * w)
    mask[y1:y2, x1:x2] = True
    return mask


def make_valid_mask(gt: np.ndarray,
                    pred: np.ndarray,
                    min_depth: float,
                    max_depth: float,
                    use_eigen_crop: bool) -> np.ndarray:
    valid = (
        np.isfinite(gt) &
        np.isfinite(pred) &
        (gt > min_depth) &
        (gt < max_depth) &
        (pred > 1e-8)
    )

    if use_eigen_crop:
        h, w = gt.shape[:2]
        valid &= eigen_crop_mask(h, w)

    return valid


def compute_metrics(gt: np.ndarray,
                    pred: np.ndarray,
                    min_depth: float,
                    max_depth: float) -> Dict[str, float]:
    gt = gt.astype(np.float64)
    pred = pred.astype(np.float64)

    gt = np.clip(gt, min_depth, max_depth)
    pred = np.clip(pred, min_depth, max_depth)

    thresh = np.maximum(gt / pred, pred / gt)

    return {
        "AbsRel": float(np.mean(np.abs(gt - pred) / gt)),
        "SqRel": float(np.mean(((gt - pred) ** 2) / gt)),
        "RMSE": float(np.sqrt(np.mean((gt - pred) ** 2))),
        "RMSElog": float(np.sqrt(np.mean((np.log(gt) - np.log(pred)) ** 2))),
        "delta1": float(np.mean(thresh < 1.25)),
        "delta2": float(np.mean(thresh < 1.25 ** 2)),
        "delta3": float(np.mean(thresh < 1.25 ** 3)),
    }


def median_scale(gt_valid: np.ndarray, pred_valid: np.ndarray) -> float:
    med_pred = np.median(pred_valid)
    med_gt = np.median(gt_valid)

    if not np.isfinite(med_pred) or med_pred <= 1e-8:
        return np.nan

    return float(med_gt / med_pred)


def transform_rel_depth(rel: np.ndarray, mode: str) -> np.ndarray:
    rel = rel.astype(np.float32)

    if mode == "depth":
        out = rel.copy()
    elif mode == "inverse":
        out = 1.0 / np.clip(rel, 1e-6, None)
    else:
        raise ValueError(f"Unsupported rel mode: {mode}")

    out[~np.isfinite(out)] = 0.0
    return out.astype(np.float32)


# =========================================================
# Support-point sampling
# =========================================================

def sample_support_points(road_mask: np.ndarray,
                          lane_mask: np.ndarray,
                          y_ratio0: float = 0.45,
                          center_width_ratio: float = 0.50,
                          sample_step_y: int = 8,
                          sample_step_x: int = 12,
                          min_row_width: int = 20) -> Tuple[np.ndarray, Dict]:
    road_mask = normalize_mask(road_mask)
    lane_mask = normalize_mask(lane_mask)

    h, w = road_mask.shape[:2]
    y0 = int(h * y_ratio0)

    points = []
    row_counts = []

    for y in range(h - 1, y0 - 1, -sample_step_y):
        road_xs = np.where(road_mask[y] > 0)[0]
        if len(road_xs) < min_row_width:
            row_counts.append(0)
            continue

        x_left = int(road_xs.min())
        x_right = int(road_xs.max())
        row_width = x_right - x_left + 1

        if row_width < min_row_width:
            row_counts.append(0)
            continue

        lane_xs = np.where(lane_mask[y] > 0)[0]
        if len(lane_xs) >= 2:
            row_center = float(np.median(lane_xs))
        else:
            row_center = 0.5 * (x_left + x_right)

        half_keep = 0.5 * row_width * center_width_ratio
        keep_left = max(0, int(row_center - half_keep))
        keep_right = min(w - 1, int(row_center + half_keep))

        keep_num = 0
        for x in range(keep_left, keep_right + 1, sample_step_x):
            if road_mask[y, x] > 0:
                points.append([x, y])
                keep_num += 1

        row_counts.append(keep_num)

    if len(points) == 0:
        return np.zeros((0, 2), dtype=np.int32), {
            "num_points": 0,
            "row_counts": row_counts,
        }

    return np.array(points, dtype=np.int32), {
        "num_points": len(points),
        "row_counts": row_counts,
    }


def estimate_geometry_anchors(points_xy: np.ndarray,
                              fx: float,
                              fy: float,
                              cx: float,
                              cy: float,
                              cam_height: float,
                              pitch_deg: float,
                              img_h: int,
                              img_w: int,
                              min_anchor_z: float,
                              max_anchor_z: float,
                              min_y_margin: int = 4) -> Tuple[np.ndarray, np.ndarray, Dict]:
    if points_xy is None or len(points_xy) == 0:
        return (
            np.zeros((0, 2), dtype=np.int32),
            np.zeros((0,), dtype=np.float32),
            {"num_points": 0, "reason": "no_support_points"}
        )

    H = float(cam_height)
    pitch = np.deg2rad(float(pitch_deg))

    valid_points = []
    z_list = []

    for x, y in points_xy:
        x = int(x)
        y = int(y)

        if x < 0 or x >= img_w or y < 0 or y >= img_h:
            continue

        if y < cy + min_y_margin:
            continue

        alpha = np.arctan((float(y) - cy) / max(fy, 1e-6))
        theta = pitch + alpha

        if theta <= np.deg2rad(0.5):
            continue

        z = H / np.tan(theta)

        if not np.isfinite(z):
            continue

        if z <= 0.5 or z > 200.0:
            continue

        if z < min_anchor_z or z > max_anchor_z:
            continue

        valid_points.append([x, y])
        z_list.append(float(z))

    if len(valid_points) == 0:
        return (
            np.zeros((0, 2), dtype=np.int32),
            np.zeros((0,), dtype=np.float32),
            {
                "num_points": 0,
                "reason": "no_valid_geometry_anchor",
                "min_anchor_z": min_anchor_z,
                "max_anchor_z": max_anchor_z,
            }
        )

    valid_points = np.array(valid_points, dtype=np.int32)
    z_geo = np.array(z_list, dtype=np.float32)

    return valid_points, z_geo, {
        "num_points": int(len(valid_points)),
        "z_min": float(np.min(z_geo)),
        "z_med": float(np.median(z_geo)),
        "z_max": float(np.max(z_geo)),
        "min_anchor_z": float(min_anchor_z),
        "max_anchor_z": float(max_anchor_z),
    }


def iqr_keep_mask(values: np.ndarray, k: float = 1.5) -> np.ndarray:
    if values is None or len(values) == 0:
        return np.zeros((0,), dtype=bool)

    q1 = np.percentile(values, 25)
    q3 = np.percentile(values, 75)
    iqr = q3 - q1

    if iqr < 1e-12:
        return np.ones_like(values, dtype=bool)

    lo = q1 - k * iqr
    hi = q3 + k * iqr

    return (values >= lo) & (values <= hi)


def estimate_scale_candidate(points_xy: np.ndarray,
                             z_geo: np.ndarray,
                             base_depth: np.ndarray,
                             min_points: int = 8) -> Tuple[Optional[float], Dict]:
    if len(points_xy) == 0 or len(z_geo) == 0:
        return None, {"reason": "no_geo_points", "n_raw": 0, "n_filt": 0}

    h, w = base_depth.shape[:2]
    ratios = []

    for (x, y), zg in zip(points_xy, z_geo):
        x = int(x)
        y = int(y)

        if x < 0 or x >= w or y < 0 or y >= h:
            continue

        d = float(base_depth[y, x])

        if not np.isfinite(d) or d <= 1e-8:
            continue

        ratios.append(float(zg) / (d + 1e-8))

    if len(ratios) < min_points:
        return None, {
            "reason": "too_few_ratio_points",
            "n_raw": int(len(ratios)),
            "n_filt": 0,
        }

    ratios = np.array(ratios, dtype=np.float32)
    keep = iqr_keep_mask(ratios, k=1.5)
    ratios_f = ratios[keep]

    if len(ratios_f) < min_points:
        return None, {
            "reason": "too_few_after_iqr",
            "n_raw": int(len(ratios)),
            "n_filt": int(len(ratios_f)),
        }

    return float(np.median(ratios_f)), {
        "reason": "ok",
        "n_raw": int(len(ratios)),
        "n_filt": int(len(ratios_f)),
        "ratio_med": float(np.median(ratios_f)),
        "ratio_mean": float(np.mean(ratios_f)),
        "ratio_std": float(np.std(ratios_f)),
    }


# =========================================================
# Temporal filters
# =========================================================

def ema_hold(values: np.ndarray, alpha: float = 0.08) -> np.ndarray:
    out = np.full_like(values, np.nan, dtype=np.float64)
    prev = np.nan

    for i, v in enumerate(values):
        if np.isfinite(v):
            if not np.isfinite(prev):
                prev = float(v)
            else:
                prev = alpha * float(v) + (1.0 - alpha) * prev

        out[i] = prev

    return out


def state_machine_scale(values: np.ndarray,
                        alpha_fast: float = 0.35,
                        alpha_slow: float = 0.12,
                        alpha_jump: float = 0.06,
                        jump_slow_ratio: float = 0.09,
                        jump_reject_ratio: float = 0.22,
                        jump_freeze_ratio: float = 0.55) -> np.ndarray:
    out = np.full_like(values, np.nan, dtype=np.float64)
    s_final = np.nan

    for i, s_candidate in enumerate(values):
        if not np.isfinite(s_candidate):
            out[i] = s_final
            continue

        s_candidate = float(s_candidate)

        if not np.isfinite(s_final):
            s_final = s_candidate
            out[i] = s_final
            continue

        jump_ratio = abs(s_candidate - s_final) / max(abs(s_final), 1e-6)

        if jump_ratio > jump_freeze_ratio:
            out[i] = s_final
            continue
        elif jump_ratio > jump_reject_ratio:
            alpha = alpha_jump
        elif jump_ratio > jump_slow_ratio:
            alpha = alpha_slow
        else:
            alpha = alpha_fast

        s_final = alpha * s_candidate + (1.0 - alpha) * s_final
        out[i] = s_final

    return out


# =========================================================
# Evaluation
# =========================================================

def evaluate_scale_series(method: str,
                          stems: List[str],
                          gt_list: List[np.ndarray],
                          base_list: List[np.ndarray],
                          scales: np.ndarray,
                          min_depth: float,
                          max_depth: float,
                          use_eigen_crop: bool) -> Tuple[Dict, List[Dict]]:
    frame_rows = []

    for idx, stem in enumerate(stems):
        s = scales[idx]
        if not np.isfinite(s):
            continue

        gt = gt_list[idx]
        base = base_list[idx]

        pred = base * float(s)

        valid = make_valid_mask(gt, pred, min_depth, max_depth, use_eigen_crop)
        if valid.sum() < 50:
            continue

        m = compute_metrics(gt[valid], pred[valid], min_depth, max_depth)
        m["method"] = method
        m["frame"] = stem
        m["scale"] = float(s)
        m["valid_pixels"] = int(valid.sum())
        frame_rows.append(m)

    if len(frame_rows) == 0:
        return {
            "method": method,
            "frames": 0,
        }, []

    df = pd.DataFrame(frame_rows)

    scale_s = pd.Series(scales, dtype="float64")
    delta_s = scale_s.diff().abs()

    row = {
        "method": method,
        "frames": int(len(df)),
        "scale_mean": float(scale_s.mean(skipna=True)),
        "scale_std": float(scale_s.std(skipna=True)),
        "scale_delta_mean": float(delta_s.mean(skipna=True)),
        "scale_delta_p95": float(delta_s.quantile(0.95)),
        "valid_pixels_mean": float(df["valid_pixels"].mean()),
    }

    for k in ["AbsRel", "SqRel", "RMSE", "RMSElog", "delta1", "delta2", "delta3"]:
        row[k] = float(df[k].mean())

    return row, frame_rows


def auto_select_rel_mode(stems: List[str],
                         gt_dir: str,
                         pred_dir: str,
                         min_depth: float,
                         max_depth: float,
                         use_eigen_crop: bool,
                         max_frames: int = 30) -> str:
    scores = {}

    for mode in ["depth", "inverse"]:
        absrels = []

        for stem in stems[:max_frames]:
            gt = to_float_depth(os.path.join(gt_dir, f"{stem}.npy"))
            rel = to_float_depth(os.path.join(pred_dir, f"{stem}.npy"))

            if rel.shape[:2] != gt.shape[:2]:
                rel = cv2.resize(rel, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LINEAR)

            base = transform_rel_depth(rel, mode)
            valid = make_valid_mask(gt, base, min_depth, max_depth, use_eigen_crop)

            if valid.sum() < 50:
                continue

            s = median_scale(gt[valid], base[valid])
            if not np.isfinite(s):
                continue

            pred = base * s
            valid2 = make_valid_mask(gt, pred, min_depth, max_depth, use_eigen_crop)
            if valid2.sum() < 50:
                continue

            m = compute_metrics(gt[valid2], pred[valid2], min_depth, max_depth)
            absrels.append(m["AbsRel"])

        scores[mode] = float(np.mean(absrels)) if len(absrels) > 0 else np.inf

    print(f"[Auto rel-mode] depth   AbsRel={scores['depth']:.6f}")
    print(f"[Auto rel-mode] inverse AbsRel={scores['inverse']:.6f}")

    return "depth" if scores["depth"] <= scores["inverse"] else "inverse"


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--date-dir", required=True)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--gt-dir", required=True)
    parser.add_argument("--pred-dir", required=True)
    parser.add_argument("--out-dir", required=True)

    parser.add_argument("--rel-mode", choices=["auto", "depth", "inverse"], default="auto")
    parser.add_argument("--device", default="cuda")

    parser.add_argument("--cam-height", type=float, default=1.65)
    parser.add_argument("--pitch-deg", type=float, default=0.0)

    parser.add_argument("--scale-min-anchor-z", type=float, default=10.0)
    parser.add_argument("--scale-max-anchor-z", type=float, default=40.0)
    parser.add_argument("--min-scale-points", type=int, default=8)

    parser.add_argument("--support-y-ratio0", type=float, default=0.45)
    parser.add_argument("--support-center-width-ratio", type=float, default=0.50)
    parser.add_argument("--sample-step-y", type=int, default=8)
    parser.add_argument("--support-sample-step-x", type=int, default=12)
    parser.add_argument("--support-min-row-width", type=int, default=20)

    parser.add_argument("--seg-da-weight", type=str, default=os.path.join(
        ROOT, "models", "ckpts", "twinlitenetpp_geodepth", "twinlitenetpp_da_best.pth"
    ))
    parser.add_argument("--seg-ll-weight", type=str, default=os.path.join(
        ROOT, "models", "ckpts", "twinlitenetpp_geodepth", "twinlitenetpp_lanegeo_best.pth"
    ))
    parser.add_argument("--seg-input-w", type=int, default=640)
    parser.add_argument("--seg-input-h", type=int, default=360)
    parser.add_argument("--seg-da-thresh", type=float, default=0.5)
    parser.add_argument("--seg-lane-thresh", type=float, default=0.5)

    parser.add_argument("--min-depth", type=float, default=1.0)
    parser.add_argument("--max-depth", type=float, default=80.0)
    parser.add_argument("--use-eigen-crop", action="store_true")
    parser.add_argument("--max-frames", type=int, default=-1)

    parser.add_argument("--save-vis-every", type=int, default=20)

    args = parser.parse_args()

    ensure_dir(args.out_dir)
    vis_dir = os.path.join(args.out_dir, "vis")
    ensure_dir(vis_dir)

    intr = load_kitti_intrinsics(args.date_dir)
    fx, fy, cx, cy = intr["fx"], intr["fy"], intr["cx"], intr["cy"]

    print("[KITTI Camera]")
    print(f"fx={fx:.3f}, fy={fy:.3f}, cx={cx:.3f}, cy={cy:.3f}")
    print(f"H={args.cam_height:.3f}, pitch={args.pitch_deg:.3f}")

    image_files = sorted(Path(args.image_dir).glob("*.png"))
    gt_files = {p.stem: str(p) for p in Path(args.gt_dir).glob("*.npy")}
    pred_files = {p.stem: str(p) for p in Path(args.pred_dir).glob("*.npy")}

    stems = []
    for p in image_files:
        if p.stem in gt_files and p.stem in pred_files:
            stems.append(p.stem)

    if args.max_frames > 0:
        stems = stems[:args.max_frames]

    if len(stems) == 0:
        raise RuntimeError("No matched KITTI frames found.")

    print(f"[Frames] matched={len(stems)}")

    if args.rel_mode == "auto":
        rel_mode = auto_select_rel_mode(
            stems=stems,
            gt_dir=args.gt_dir,
            pred_dir=args.pred_dir,
            min_depth=args.min_depth,
            max_depth=args.max_depth,
            use_eigen_crop=args.use_eigen_crop,
        )
    else:
        rel_mode = args.rel_mode

    print(f"[Rel mode] {rel_mode}")

    seg_args = SimpleNamespace(
        device=args.device,
        seg_da_weight=args.seg_da_weight,
        seg_ll_weight=args.seg_ll_weight,
        seg_input_w=args.seg_input_w,
        seg_input_h=args.seg_input_h,
        seg_da_thresh=args.seg_da_thresh,
        seg_lane_thresh=args.seg_lane_thresh,
        road_open_ksize=3,
        road_close_ksize=7,
        road_min_area=200,
        lane_open_ksize=3,
        lane_close_ksize=5,
        lane_min_area=50,
    )

    seg_runner = build_segmentation_runner(seg_args)

    gt_list = []
    base_list = []
    s_oracle_list = []
    s_geo_list = []

    debug_rows = []
    runtime_rows = []

    for idx, stem in enumerate(stems):
        t_frame_start = now_ms()

        img_path = os.path.join(args.image_dir, f"{stem}.png")
        gt_path = os.path.join(args.gt_dir, f"{stem}.npy")
        pred_path = os.path.join(args.pred_dir, f"{stem}.npy")

        # -------------------------------------------------
        # 1. Data loading + relative-depth transformation
        # -------------------------------------------------
        t0 = now_ms()

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            print(f"[Skip] Cannot read image: {img_path}")
            continue

        h, w = img.shape[:2]

        gt = to_float_depth(gt_path)
        rel = to_float_depth(pred_path)

        if rel.shape[:2] != gt.shape[:2]:
            rel = cv2.resize(rel, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LINEAR)

        base = transform_rel_depth(rel, rel_mode)

        load_time_ms = now_ms() - t0

        # -------------------------------------------------
        # 2. TwinLiteNet++ lane/drivable segmentation
        # -------------------------------------------------
        t0 = now_ms()

        lane_mask, road_mask, seg_debug = seg_runner(img)
        lane_mask = normalize_mask(lane_mask)
        road_mask = normalize_mask(road_mask)

        segmentation_time_ms = now_ms() - t0

        # -------------------------------------------------
        # 3. Road support-point sampling
        # -------------------------------------------------
        t0 = now_ms()

        sample_pts, support_debug = sample_support_points(
            road_mask=road_mask,
            lane_mask=lane_mask,
            y_ratio0=args.support_y_ratio0,
            center_width_ratio=args.support_center_width_ratio,
            sample_step_y=args.sample_step_y,
            sample_step_x=args.support_sample_step_x,
            min_row_width=args.support_min_row_width,
        )

        support_sampling_time_ms = now_ms() - t0

        # -------------------------------------------------
        # 4. Sparse geometric metric-anchor estimation
        # -------------------------------------------------
        t0 = now_ms()

        geo_pts, z_geo, geo_debug = estimate_geometry_anchors(
            points_xy=sample_pts,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            cam_height=args.cam_height,
            pitch_deg=args.pitch_deg,
            img_h=h,
            img_w=w,
            min_anchor_z=args.scale_min_anchor_z,
            max_anchor_z=args.scale_max_anchor_z,
        )

        geometry_anchor_time_ms = now_ms() - t0

        # -------------------------------------------------
        # 5. Scale candidate fitting: Z_geo / D_rel
        # -------------------------------------------------
        t0 = now_ms()

        s_geo, fit_debug = estimate_scale_candidate(
            points_xy=geo_pts,
            z_geo=z_geo,
            base_depth=base,
            min_points=args.min_scale_points,
        )

        scale_candidate_time_ms = now_ms() - t0

        # -------------------------------------------------
        # 6. Validation-only oracle scale
        #    This is not part of deployable inference.
        # -------------------------------------------------
        t0 = now_ms()

        valid = make_valid_mask(gt, base, args.min_depth, args.max_depth, args.use_eigen_crop)
        s_oracle = median_scale(gt[valid], base[valid]) if valid.sum() >= 50 else np.nan

        oracle_scale_time_ms = now_ms() - t0

        gt_list.append(gt)
        base_list.append(base)
        s_oracle_list.append(s_oracle)
        s_geo_list.append(np.nan if s_geo is None else float(s_geo))

        debug_rows.append({
            "frame": stem,
            "cam_height": float(args.cam_height),
            "pitch_deg": float(args.pitch_deg),
            "s_oracle": float(s_oracle) if np.isfinite(s_oracle) else np.nan,
            "s_geo": np.nan if s_geo is None else float(s_geo),

            "geo_fit_reason": fit_debug.get("reason", ""),
            "sample_points": int(len(sample_pts)),
            "geo_points": int(len(geo_pts)),
            "ratio_raw": fit_debug.get("n_raw", 0),
            "ratio_filt": fit_debug.get("n_filt", 0),
            "z_min": geo_debug.get("z_min", np.nan),
            "z_med": geo_debug.get("z_med", np.nan),
            "z_max": geo_debug.get("z_max", np.nan),
            "lane_pixels": int((lane_mask > 0).sum()),
            "road_pixels": int((road_mask > 0).sum()),
        })

        # -------------------------------------------------
        # 7. Visualization.
        #    This is debug overhead and should be reported separately.
        # -------------------------------------------------
        t0 = now_ms()

        if args.save_vis_every > 0 and idx % args.save_vis_every == 0:
            overlay = img.copy()

            road_bool = road_mask > 0
            lane_bool = lane_mask > 0

            green = np.zeros_like(overlay)
            green[..., 1] = 255
            overlay = np.where(
                road_bool[..., None],
                (overlay * 0.6 + green * 0.4).astype(np.uint8),
                overlay
            )

            red = np.zeros_like(overlay)
            red[..., 2] = 255
            overlay = np.where(
                lane_bool[..., None],
                (overlay * 0.35 + red * 0.65).astype(np.uint8),
                overlay
            )

            for x, y in geo_pts:
                cv2.circle(overlay, (int(x), int(y)), 2, (255, 255, 0), -1)

            cv2.putText(
                overlay,
                f"frame={stem} s_geo={s_geo if s_geo is not None else None}",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

            cv2.imwrite(os.path.join(vis_dir, f"{stem}.png"), overlay)

        visualization_time_ms = now_ms() - t0

        total_time_ms = now_ms() - t_frame_start

        # Candidate generation part in this KITTI script.
        # Note: Depth Anything V2 inference is not included here because this script loads precomputed depth.
        deploy_core_time_ms = (
            segmentation_time_ms +
            support_sampling_time_ms +
            geometry_anchor_time_ms +
            scale_candidate_time_ms
        )

        runtime_rows.append({
            "frame_index": int(idx),
            "frame": stem,
            "image_h": int(h),
            "image_w": int(w),

            "load_time_ms": float(load_time_ms),
            "segmentation_time_ms": float(segmentation_time_ms),
            "support_sampling_time_ms": float(support_sampling_time_ms),
            "geometry_anchor_time_ms": float(geometry_anchor_time_ms),
            "scale_candidate_time_ms": float(scale_candidate_time_ms),
            "oracle_scale_time_ms": float(oracle_scale_time_ms),
            "visualization_time_ms": float(visualization_time_ms),
            "deploy_core_time_ms": float(deploy_core_time_ms),
            "total_time_ms": float(total_time_ms),

            "deploy_core_fps": float(safe_fps(deploy_core_time_ms)),
            "total_fps": float(safe_fps(total_time_ms)),

            "sample_points": int(len(sample_pts)),
            "geo_points": int(len(geo_pts)),
            "ratio_raw": int(fit_debug.get("n_raw", 0)),
            "ratio_filt": int(fit_debug.get("n_filt", 0)),
            "candidate_valid": int(s_geo is not None and np.isfinite(float(s_geo))),
            "geo_fit_reason": fit_debug.get("reason", ""),
        })

        if idx % 10 == 0:
            print(
                f"[{idx:04d}/{len(stems)}] {stem} "
                f"s_geo={s_geo} "
                f"geo_pts={len(geo_pts)} "
                f"z_med={geo_debug.get('z_med', np.nan)} "
                f"deploy_core={deploy_core_time_ms:.2f}ms "
                f"total={total_time_ms:.2f}ms"
            )

    s_oracle_arr = np.array(s_oracle_list, dtype=np.float64)
    s_geo_arr = np.array(s_geo_list, dtype=np.float64)

    # Sequence oracle baseline
    all_gt = []
    all_base = []

    for gt, base in zip(gt_list, base_list):
        valid = make_valid_mask(gt, base, args.min_depth, args.max_depth, args.use_eigen_crop)
        if valid.sum() > 0:
            all_gt.append(gt[valid])
            all_base.append(base[valid])

    all_gt = np.concatenate(all_gt)
    all_base = np.concatenate(all_base)
    sequence_scale = median_scale(all_gt, all_base)

    t_scale_series_start = now_ms()

    scale_series = {
        "raw_unscaled": np.ones_like(s_oracle_arr, dtype=np.float64),
        "per_frame_median_scale_oracle": s_oracle_arr,
        "sequence_median_scale_oracle": np.full_like(s_oracle_arr, sequence_scale),
        "geometry_candidate": s_geo_arr,
        "geometry_ema": ema_hold(s_geo_arr, alpha=0.08),
        "geometry_state_machine": state_machine_scale(s_geo_arr),
    }

    scale_series_time_ms = now_ms() - t_scale_series_start

    summary_rows = []
    per_frame_rows = []

    t_evaluation_start = now_ms()

    for method, scales in scale_series.items():
        row, rows = evaluate_scale_series(
            method=method,
            stems=stems,
            gt_list=gt_list,
            base_list=base_list,
            scales=scales,
            min_depth=args.min_depth,
            max_depth=args.max_depth,
            use_eigen_crop=args.use_eigen_crop,
        )
        summary_rows.append(row)
        per_frame_rows.extend(rows)

    evaluation_all_methods_time_ms = now_ms() - t_evaluation_start

    summary = pd.DataFrame(summary_rows)
    per_frame = pd.DataFrame(per_frame_rows)
    debug_df = pd.DataFrame(debug_rows)

    out_summary = os.path.join(args.out_dir, "kitti_geometry_eval_summary.csv")
    out_per_frame = os.path.join(args.out_dir, "kitti_geometry_eval_per_frame.csv")
    out_debug = os.path.join(args.out_dir, "kitti_geometry_scale_debug.csv")

    summary.to_csv(out_summary, index=False, encoding="utf-8-sig")
    per_frame.to_csv(out_per_frame, index=False, encoding="utf-8-sig")
    debug_df.to_csv(out_debug, index=False, encoding="utf-8-sig")

    # Runtime profiling output.
    runtime_df = pd.DataFrame(runtime_rows)
    out_runtime_per_frame = os.path.join(args.out_dir, "runtime_per_frame.csv")
    out_runtime_summary = os.path.join(args.out_dir, "runtime_summary.csv")

    if len(runtime_df) > 0:
        runtime_df.to_csv(out_runtime_per_frame, index=False, encoding="utf-8-sig")

        runtime_time_cols = [
            "load_time_ms",
            "segmentation_time_ms",
            "support_sampling_time_ms",
            "geometry_anchor_time_ms",
            "scale_candidate_time_ms",
            "oracle_scale_time_ms",
            "visualization_time_ms",
            "deploy_core_time_ms",
            "total_time_ms",
        ]

        runtime_summary_row = {
            "num_frames": int(len(runtime_df)),
            "scale_series_time_ms": float(scale_series_time_ms),
            "evaluation_all_methods_time_ms": float(evaluation_all_methods_time_ms),
            "candidate_valid_ratio": float(runtime_df["candidate_valid"].mean()),
            "geo_points_mean": float(runtime_df["geo_points"].mean()),
            "geo_points_median": float(runtime_df["geo_points"].median()),
        }

        for col in runtime_time_cols:
            vals = runtime_df[col].astype(float)
            runtime_summary_row[f"{col}_mean"] = float(vals.mean())
            runtime_summary_row[f"{col}_median"] = float(vals.median())
            runtime_summary_row[f"{col}_p95"] = float(vals.quantile(0.95))

        runtime_summary_row["deploy_core_fps_from_mean_ms"] = float(
            safe_fps(runtime_summary_row["deploy_core_time_ms_mean"])
        )
        runtime_summary_row["total_fps_from_mean_ms"] = float(
            safe_fps(runtime_summary_row["total_time_ms_mean"])
        )

        runtime_summary_df = pd.DataFrame([runtime_summary_row])
        runtime_summary_df.to_csv(out_runtime_summary, index=False, encoding="utf-8-sig")
    else:
        pd.DataFrame().to_csv(out_runtime_per_frame, index=False, encoding="utf-8-sig")
        pd.DataFrame().to_csv(out_runtime_summary, index=False, encoding="utf-8-sig")

    # Scale plot
    plt.figure(figsize=(14, 5))
    x = np.arange(len(stems))

    for method, scales in scale_series.items():
        if method == "raw_unscaled":
            continue
        if "oracle" in method:
            plt.plot(x, scales, linewidth=0.8, alpha=0.45, label=method)
        else:
            plt.plot(x, scales, linewidth=1.4, label=method)

    plt.xlabel("Frame")
    plt.ylabel("Scale")
    plt.title("KITTI Geometry-Anchor Scale Sequence")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out_scale_fig = os.path.join(args.out_dir, "kitti_geometry_scale_sequence.png")
    plt.savefig(out_scale_fig, dpi=300)
    plt.close()

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 300)

    print("\n=== KITTI Geometry Anchor Evaluation Summary ===")
    print(summary.to_string(index=False))

    print("\nSaved:")
    print(out_summary)
    print(out_per_frame)
    print(out_debug)
    print(out_runtime_per_frame)
    print(out_runtime_summary)
    print(out_scale_fig)


if __name__ == "__main__":
    main()
