# -*- coding: utf-8 -*-
"""
eval_kitti_depth_baselines.py

Evaluate Depth Anything V2 relative depth on KITTI sparse LiDAR GT.

This script evaluates several scale baselines:

1. raw_unscaled
2. per_frame_median_scale
3. sequence_median_scale
4. moving_average_scale
5. ema_scale
6. median_filter_scale

Important:
- Depth Anything V2 output may be relative depth or inverse-depth-like.
- Use --rel-mode auto to test both rel and 1/rel and choose the better direction
  under per-frame median scaling.
"""

import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def load_npy(path: str) -> np.ndarray:
    return np.load(path).astype(np.float32)


def eigen_crop_mask(h: int, w: int) -> np.ndarray:
    """
    Common Eigen crop for KITTI monocular depth evaluation.
    """
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
                    min_depth: float = 1.0,
                    max_depth: float = 80.0) -> Dict[str, float]:
    gt = gt.astype(np.float64)
    pred = pred.astype(np.float64)

    # KITTI-style depth clipping
    pred = np.clip(pred, min_depth, max_depth)
    gt = np.clip(gt, min_depth, max_depth)

    thresh = np.maximum(gt / pred, pred / gt)

    delta1 = np.mean(thresh < 1.25)
    delta2 = np.mean(thresh < 1.25 ** 2)
    delta3 = np.mean(thresh < 1.25 ** 3)

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)
    rmse = np.sqrt(np.mean((gt - pred) ** 2))
    rmse_log = np.sqrt(np.mean((np.log(gt) - np.log(pred)) ** 2))

    return {
        "AbsRel": float(abs_rel),
        "SqRel": float(sq_rel),
        "RMSE": float(rmse),
        "RMSElog": float(rmse_log),
        "delta1": float(delta1),
        "delta2": float(delta2),
        "delta3": float(delta3),
    }


def median_scale(gt_valid: np.ndarray, pred_valid: np.ndarray) -> float:
    med_pred = np.median(pred_valid)
    med_gt = np.median(gt_valid)

    if not np.isfinite(med_pred) or med_pred <= 1e-8:
        return np.nan

    return float(med_gt / med_pred)


def moving_average_hold(values: np.ndarray, window: int = 15) -> np.ndarray:
    out = np.full_like(values, np.nan, dtype=np.float64)
    prev = np.nan

    for i in range(len(values)):
        start = max(0, i - window + 1)
        vals = values[start:i + 1]
        vals = vals[np.isfinite(vals)]

        if len(vals) > 0:
            prev = float(np.mean(vals))

        out[i] = prev

    return out


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


def median_filter_hold(values: np.ndarray, window: int = 15) -> np.ndarray:
    out = np.full_like(values, np.nan, dtype=np.float64)
    prev = np.nan

    for i in range(len(values)):
        start = max(0, i - window + 1)
        vals = values[start:i + 1]
        vals = vals[np.isfinite(vals)]

        if len(vals) > 0:
            prev = float(np.median(vals))

        out[i] = prev

    return out

def state_machine_scale(
    values: np.ndarray,
    alpha_fast: float = 0.08,
    alpha_slow: float = 0.025,
    alpha_jump: float = 0.015,
    jump_slow_ratio: float = 0.04,
    jump_reject_ratio: float = 0.10,
    jump_freeze_ratio: float = 0.30,
) -> np.ndarray:
    """
    Simplified proposed temporal state machine for KITTI scale sequence.

    values:
        per-frame scale candidates.

    returns:
        stabilized scale sequence.
    """
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
            # Freeze: keep previous stabilized scale
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

def collect_files(gt_dir: str, pred_dir: str) -> List[Tuple[str, str, str]]:
    gt_files = {p.stem: str(p) for p in Path(gt_dir).glob("*.npy")}
    pred_files = {p.stem: str(p) for p in Path(pred_dir).glob("*.npy")}

    stems = sorted(set(gt_files.keys()) & set(pred_files.keys()))

    pairs = []
    for stem in stems:
        pairs.append((stem, gt_files[stem], pred_files[stem]))

    return pairs


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


def evaluate_per_frame_median_for_mode(pairs,
                                       mode: str,
                                       min_depth: float,
                                       max_depth: float,
                                       use_eigen_crop: bool,
                                       max_frames_for_auto: int = -1) -> float:
    metrics = []

    if max_frames_for_auto > 0:
        pairs_eval = pairs[:max_frames_for_auto]
    else:
        pairs_eval = pairs

    for stem, gt_path, pred_path in pairs_eval:
        gt = load_npy(gt_path)
        rel = load_npy(pred_path)

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

        m = compute_metrics(
            gt[valid2],
            pred[valid2],
            min_depth=min_depth,
            max_depth=max_depth
        )
        metrics.append(m["AbsRel"])

    if len(metrics) == 0:
        return np.inf

    return float(np.mean(metrics))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt-dir", required=True)
    parser.add_argument("--pred-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--rel-mode", choices=["auto", "depth", "inverse"], default="auto")
    parser.add_argument("--min-depth", type=float, default=1.0)
    parser.add_argument("--max-depth", type=float, default=80.0)
    parser.add_argument("--use-eigen-crop", action="store_true")
    parser.add_argument("--ma-window", type=int, default=15)
    parser.add_argument("--median-window", type=int, default=15)
    parser.add_argument("--ema-alpha", type=float, default=0.08)
    parser.add_argument("--max-frames", type=int, default=-1)
    parser.add_argument("--state-alpha-fast", type=float, default=0.08)
    parser.add_argument("--state-alpha-slow", type=float, default=0.025)
    parser.add_argument("--state-alpha-jump", type=float, default=0.015)
    parser.add_argument("--state-jump-slow-ratio", type=float, default=0.04)
    parser.add_argument("--state-jump-reject-ratio", type=float, default=0.10)
    parser.add_argument("--state-jump-freeze-ratio", type=float, default=0.30)
    args = parser.parse_args()

    ensure_dir(args.out_dir)

    pairs = collect_files(args.gt_dir, args.pred_dir)

    if args.max_frames > 0:
        pairs = pairs[:args.max_frames]

    if len(pairs) == 0:
        raise RuntimeError("No matched GT / prediction npy files found.")

    print(f"[Eval] matched frames: {len(pairs)}")
    print(f"[Eval] GT dir        : {args.gt_dir}")
    print(f"[Eval] Pred dir      : {args.pred_dir}")
    print(f"[Eval] Eigen crop    : {args.use_eigen_crop}")

    # Decide relative-depth direction
    if args.rel_mode == "auto":
        absrel_depth = evaluate_per_frame_median_for_mode(
            pairs, "depth", args.min_depth, args.max_depth, args.use_eigen_crop
        )
        absrel_inverse = evaluate_per_frame_median_for_mode(
            pairs, "inverse", args.min_depth, args.max_depth, args.use_eigen_crop
        )

        if absrel_depth <= absrel_inverse:
            selected_mode = "depth"
        else:
            selected_mode = "inverse"

        print(f"[Auto rel-mode] depth   AbsRel={absrel_depth:.6f}")
        print(f"[Auto rel-mode] inverse AbsRel={absrel_inverse:.6f}")
        print(f"[Auto rel-mode] selected: {selected_mode}")
    else:
        selected_mode = args.rel_mode
        print(f"[Rel-mode] selected: {selected_mode}")

    stems = []
    base_depths = []
    gt_depths = []
    valid_masks = []
    per_frame_scales = []

    # Load all frames and compute per-frame median scales
    for stem, gt_path, pred_path in pairs:
        gt = load_npy(gt_path)
        rel = load_npy(pred_path)

        if rel.shape[:2] != gt.shape[:2]:
            rel = cv2.resize(rel, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LINEAR)

        base = transform_rel_depth(rel, selected_mode)

        valid = make_valid_mask(gt, base, args.min_depth, args.max_depth, args.use_eigen_crop)

        if valid.sum() < 50:
            s = np.nan
        else:
            s = median_scale(gt[valid], base[valid])

        stems.append(stem)
        gt_depths.append(gt)
        base_depths.append(base)
        valid_masks.append(valid)
        per_frame_scales.append(s)

    per_frame_scales = np.array(per_frame_scales, dtype=np.float64)

    # Sequence-level median scale
    all_gt_vals = []
    all_base_vals = []

    for gt, base, valid in zip(gt_depths, base_depths, valid_masks):
        if valid.sum() > 0:
            all_gt_vals.append(gt[valid])
            all_base_vals.append(base[valid])

    all_gt_vals = np.concatenate(all_gt_vals, axis=0)
    all_base_vals = np.concatenate(all_base_vals, axis=0)
    sequence_scale = median_scale(all_gt_vals, all_base_vals)

    scale_series = {
        "raw_unscaled": np.ones_like(per_frame_scales, dtype=np.float64),
        "per_frame_median_scale": per_frame_scales,
        "sequence_median_scale": np.full_like(per_frame_scales, sequence_scale, dtype=np.float64),
        f"moving_average_w{args.ma_window}": moving_average_hold(per_frame_scales, window=args.ma_window),
        f"ema_a{args.ema_alpha}": ema_hold(per_frame_scales, alpha=args.ema_alpha),
        f"median_filter_w{args.median_window}": median_filter_hold(per_frame_scales, window=args.median_window),
        "proposed_state_machine": state_machine_scale(
            per_frame_scales,
            alpha_fast=args.state_alpha_fast,
            alpha_slow=args.state_alpha_slow,
            alpha_jump=args.state_alpha_jump,
            jump_slow_ratio=args.state_jump_slow_ratio,
            jump_reject_ratio=args.state_jump_reject_ratio,
            jump_freeze_ratio=args.state_jump_freeze_ratio,
        ),
    }

    summary_rows = []
    per_frame_rows = []

    for method, scales in scale_series.items():
        frame_metrics = []

        for idx, stem in enumerate(stems):
            gt = gt_depths[idx]
            base = base_depths[idx]
            s = scales[idx]

            if not np.isfinite(s):
                continue

            pred = base * float(s)
            valid = make_valid_mask(gt, pred, args.min_depth, args.max_depth, args.use_eigen_crop)

            if valid.sum() < 50:
                continue

            m = compute_metrics(
                gt[valid],
                pred[valid],
                min_depth=args.min_depth,
                max_depth=args.max_depth
            )
            m["frame"] = stem
            m["method"] = method
            m["scale"] = float(s)
            m["valid_pixels"] = int(valid.sum())
            frame_metrics.append(m)
            per_frame_rows.append(m)

        if len(frame_metrics) == 0:
            print(f"[Warn] No valid frames for method: {method}")
            continue

        df_m = pd.DataFrame(frame_metrics)

        row = {
            "method": method,
            "frames": int(len(df_m)),
            "scale_mean": float(df_m["scale"].mean()),
            "scale_std": float(df_m["scale"].std()),
            "scale_delta_mean": float(df_m["scale"].diff().abs().mean()),
            "scale_delta_p95": float(df_m["scale"].diff().abs().quantile(0.95)),
            "valid_pixels_mean": float(df_m["valid_pixels"].mean()),
        }

        for key in ["AbsRel", "SqRel", "RMSE", "RMSElog", "delta1", "delta2", "delta3"]:
            row[key] = float(df_m[key].mean())

        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    per_frame = pd.DataFrame(per_frame_rows)

    out_summary = os.path.join(args.out_dir, "kitti_depth_eval_summary.csv")
    out_per_frame = os.path.join(args.out_dir, "kitti_depth_eval_per_frame.csv")
    out_scales = os.path.join(args.out_dir, "kitti_scale_sequences.csv")

    scale_df = pd.DataFrame({"frame": stems, "s_per_frame": per_frame_scales})
    for method, scales in scale_series.items():
        scale_df[method] = scales

    summary.to_csv(out_summary, index=False, encoding="utf-8-sig")
    per_frame.to_csv(out_per_frame, index=False, encoding="utf-8-sig")
    scale_df.to_csv(out_scales, index=False, encoding="utf-8-sig")

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 300)

    print("\n=== KITTI Depth Evaluation Summary ===")
    print(summary.to_string(index=False))

    # Plot scale sequences
    plt.figure(figsize=(14, 5))
    x = np.arange(len(stems))

    for method, scales in scale_series.items():
        if method == "raw_unscaled":
            continue
        if method == "per_frame_median_scale":
            plt.plot(x, scales, linewidth=0.8, alpha=0.45, label=method)
        else:
            plt.plot(x, scales, linewidth=1.2, label=method)

    plt.xlabel("Frame")
    plt.ylabel("Scale")
    plt.title("KITTI Scale Sequences")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out_scale_fig = os.path.join(args.out_dir, "kitti_scale_sequences.png")
    plt.savefig(out_scale_fig, dpi=300)
    plt.close()

    # Plot AbsRel over frames
    if len(per_frame) > 0:
        plt.figure(figsize=(14, 5))
        for method in per_frame["method"].unique():
            df_m = per_frame[per_frame["method"] == method].copy()
            df_m = df_m.sort_values("frame")
            plt.plot(np.arange(len(df_m)), df_m["AbsRel"].to_numpy(), linewidth=1.0, label=method)

        plt.xlabel("Frame")
        plt.ylabel("AbsRel")
        plt.title("KITTI AbsRel over Frames")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        out_absrel_fig = os.path.join(args.out_dir, "kitti_absrel_curves.png")
        plt.savefig(out_absrel_fig, dpi=300)
        plt.close()

    print("\nSaved:")
    print(out_summary)
    print(out_per_frame)
    print(out_scales)
    print(out_scale_fig)


if __name__ == "__main__":
    main()
