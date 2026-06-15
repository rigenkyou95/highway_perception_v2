# -*- coding: utf-8 -*-
"""
extract_failure_segment_visuals.py

Purpose:
  Sequentially process the video with the existing full pipeline,
  but only save tri-panel visualizations for selected failure frames.

Why sequential:
  - ScaleState depends on previous frames.
  - Static hood lock depends on the first N frames.
  - Direct random access to failure frames would make s_final / hood prior inconsistent.

Inputs:
  - representative_failure_segments.csv
    or representative_failure_frames.csv
    or manual frame list.

Outputs:
  - selected tri-panel JPG images only
  - saved_visuals.csv
"""

import os
import sys
import csv
import argparse
import importlib.util
from pathlib import Path
from typing import Dict, Any, List, Tuple

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
INFER_SCRIPT = ROOT / "scripts" / "infer" / "test_full_system_video_scale_stable.py"


def import_video_pipeline():
    if not INFER_SCRIPT.is_file():
        raise FileNotFoundError(f"Cannot find inference script: {INFER_SCRIPT}")

    spec = importlib.util.spec_from_file_location("video_pipeline", str(INFER_SCRIPT))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


vp = import_video_pipeline()


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def parse_int(x, default=-1):
    try:
        if x is None:
            return default
        s = str(x).strip()
        if s == "":
            return default
        return int(float(s))
    except Exception:
        return default


def read_csv_dict(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def write_csv_dict(path: str, rows: List[Dict[str, Any]], fieldnames: List[str]):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def sanitize_name(s: str) -> str:
    s = str(s)
    for ch in ['\\', '/', ':', '*', '?', '"', '<', '>', '|', ' ', ';']:
        s = s.replace(ch, "_")
    while "__" in s:
        s = s.replace("__", "_")
    return s.strip("_")


def add_target(targets: Dict[int, List[Dict[str, Any]]],
               frame_id: int,
               meta: Dict[str, Any]):
    if frame_id < 0:
        return
    if frame_id not in targets:
        targets[frame_id] = []
    targets[frame_id].append(meta)


def load_targets_from_segments(path: str,
                               use_start: bool = True,
                               use_center: bool = True,
                               use_end: bool = True) -> Dict[int, List[Dict[str, Any]]]:
    rows = read_csv_dict(path)
    targets: Dict[int, List[Dict[str, Any]]] = {}

    for r in rows:
        sid = parse_int(r.get("segment_id"), default=-1)
        ftype = r.get("dominant_failure_type", "")
        reason = r.get("representative_reason", "")

        if use_start:
            fid = parse_int(r.get("start_frame"), default=-1)
            add_target(targets, fid, {
                "source": "segment",
                "segment_id": sid,
                "tag": "start",
                "failure_type": ftype,
                "representative_reason": reason,
                "start_frame": r.get("start_frame", ""),
                "end_frame": r.get("end_frame", ""),
                "length_frames": r.get("length_frames", ""),
                "max_jump_ratio": r.get("max_jump_ratio", ""),
                "mean_z_med": r.get("mean_z_med", ""),
            })

        if use_center:
            fid = parse_int(r.get("center_frame"), default=-1)
            add_target(targets, fid, {
                "source": "segment",
                "segment_id": sid,
                "tag": "center",
                "failure_type": ftype,
                "representative_reason": reason,
                "start_frame": r.get("start_frame", ""),
                "end_frame": r.get("end_frame", ""),
                "length_frames": r.get("length_frames", ""),
                "max_jump_ratio": r.get("max_jump_ratio", ""),
                "mean_z_med": r.get("mean_z_med", ""),
            })

        if use_end:
            fid = parse_int(r.get("end_frame"), default=-1)
            add_target(targets, fid, {
                "source": "segment",
                "segment_id": sid,
                "tag": "end",
                "failure_type": ftype,
                "representative_reason": reason,
                "start_frame": r.get("start_frame", ""),
                "end_frame": r.get("end_frame", ""),
                "length_frames": r.get("length_frames", ""),
                "max_jump_ratio": r.get("max_jump_ratio", ""),
                "mean_z_med": r.get("mean_z_med", ""),
            })

    return targets


def load_targets_from_failure_frames(path: str) -> Dict[int, List[Dict[str, Any]]]:
    rows = read_csv_dict(path)
    targets: Dict[int, List[Dict[str, Any]]] = {}

    for r in rows:
        fid = parse_int(r.get("frame_id"), default=-1)
        add_target(targets, fid, {
            "source": "failure_frame",
            "segment_id": "",
            "tag": "frame",
            "failure_type": r.get("representative_type", r.get("failure_types", "")),
            "representative_reason": "",
            "start_frame": "",
            "end_frame": "",
            "length_frames": "",
            "max_jump_ratio": r.get("jump_ratio", ""),
            "mean_z_med": r.get("z_med", ""),
        })

    return targets


def load_targets_from_manual(frames: str) -> Dict[int, List[Dict[str, Any]]]:
    targets: Dict[int, List[Dict[str, Any]]] = {}

    if frames is None or str(frames).strip() == "":
        return targets

    for item in str(frames).split(","):
        fid = parse_int(item, default=-1)
        add_target(targets, fid, {
            "source": "manual",
            "segment_id": "",
            "tag": "manual",
            "failure_type": "manual",
            "representative_reason": "",
            "start_frame": "",
            "end_frame": "",
            "length_frames": "",
            "max_jump_ratio": "",
            "mean_z_med": "",
        })

    return targets


def merge_targets(*target_dicts) -> Dict[int, List[Dict[str, Any]]]:
    merged: Dict[int, List[Dict[str, Any]]] = {}
    for td in target_dicts:
        for fid, metas in td.items():
            if fid not in merged:
                merged[fid] = []
            merged[fid].extend(metas)
    return merged


def build_pipeline_args(args):
    """
    Use the original inference script's parser, then override with our settings.
    This ensures all default model paths and parameters remain consistent.
    """
    argv = [
        "--video", args.video,
        "--cam-yaml", args.cam_yaml,
        "--out-dir", args.out_dir,
        "--resize-w", str(args.resize_w),
        "--device", args.device,
        "--hood-lock-after-frames", str(args.hood_lock_after_frames),
        "--hood-lock-smooth-ksize", str(args.hood_lock_smooth_ksize),
        "--hood-lock-mode", args.hood_lock_mode,
    ]

    if args.undistort:
        argv.append("--undistort")

    # For visual extraction, do not use --no-vis.
    # We need process_one_frame() to return the tri-panel.
    if args.profile_sync_cuda:
        argv.append("--profile-sync-cuda")

    vp_args = vp.build_parser().parse_args(argv)

    # Guard in case current pipeline supports no_vis.
    if hasattr(vp_args, "no_vis"):
        vp_args.no_vis = False

    vp_args.show = False
    return vp_args


def save_panel(panel: np.ndarray,
               out_dir: str,
               frame_id: int,
               meta: Dict[str, Any]) -> Tuple[str, str]:
    source = sanitize_name(meta.get("source", "unknown"))
    tag = sanitize_name(meta.get("tag", "frame"))
    ftype = sanitize_name(meta.get("failure_type", "failure"))
    reason = sanitize_name(meta.get("representative_reason", ""))

    sid = meta.get("segment_id", "")
    if sid == "" or sid is None:
        sid_part = "segNA"
    else:
        sid_part = f"seg{int(sid):03d}"

    if reason:
        name = f"{sid_part}_{tag}_frame_{frame_id:06d}_{ftype}_{reason}.jpg"
    else:
        name = f"{sid_part}_{tag}_frame_{frame_id:06d}_{ftype}.jpg"

    out_path = os.path.join(out_dir, name)
    cv2.imwrite(out_path, panel)
    return out_path, name


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--video", required=True)
    parser.add_argument("--cam-yaml", required=True)
    parser.add_argument("--out-dir", required=True)

    parser.add_argument("--segments-csv", default="",
                        help="Path to representative_failure_segments.csv or failure_segments.csv.")
    parser.add_argument("--failure-frames-csv", default="",
                        help="Path to representative_failure_frames.csv.")
    parser.add_argument("--frames", default="",
                        help="Manual comma-separated frame ids, e.g. 1257,1311,1365.")

    parser.add_argument("--resize-w", type=int, default=1280)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--undistort", action="store_true")
    parser.add_argument("--profile-sync-cuda", action="store_true")

    parser.add_argument("--hood-lock-after-frames", type=int, default=10)
    parser.add_argument("--hood-lock-smooth-ksize", type=int, default=101)
    parser.add_argument("--hood-lock-mode", type=str, default="median", choices=["median", "best"])

    parser.add_argument("--segment-use-start", action="store_true")
    parser.add_argument("--segment-use-center", action="store_true")
    parser.add_argument("--segment-use-end", action="store_true")

    parser.add_argument("--max-target-frame", type=int, default=-1,
                        help="Optional hard limit. Default: process until the largest selected frame.")

    args = parser.parse_args()

    ensure_dir(args.out_dir)

    # If no explicit start/center/end flag is given, use all three.
    if not (args.segment_use_start or args.segment_use_center or args.segment_use_end):
        args.segment_use_start = True
        args.segment_use_center = True
        args.segment_use_end = True

    target_dicts = []

    if args.segments_csv:
        if not os.path.isfile(args.segments_csv):
            raise FileNotFoundError(args.segments_csv)
        target_dicts.append(
            load_targets_from_segments(
                args.segments_csv,
                use_start=args.segment_use_start,
                use_center=args.segment_use_center,
                use_end=args.segment_use_end,
            )
        )

    if args.failure_frames_csv:
        if not os.path.isfile(args.failure_frames_csv):
            raise FileNotFoundError(args.failure_frames_csv)
        target_dicts.append(load_targets_from_failure_frames(args.failure_frames_csv))

    if args.frames:
        target_dicts.append(load_targets_from_manual(args.frames))

    targets = merge_targets(*target_dicts)

    if len(targets) == 0:
        raise RuntimeError(
            "No target frames found. Provide --segments-csv, --failure-frames-csv, or --frames."
        )

    target_frame_ids = sorted(targets.keys())
    max_target = max(target_frame_ids)
    if args.max_target_frame >= 0:
        max_target = min(max_target, int(args.max_target_frame))

    print("=" * 60)
    print("Failure segment visual extraction")
    print(f"Video          : {args.video}")
    print(f"Camera         : {args.cam_yaml}")
    print(f"Out dir        : {args.out_dir}")
    print(f"Target frames  : {len(target_frame_ids)}")
    print(f"First target   : {target_frame_ids[0]}")
    print(f"Last target    : {target_frame_ids[-1]}")
    print(f"Process until  : {max_target}")
    print("=" * 60)

    vp_args = build_pipeline_args(args)

    # Load camera and video. This mirrors run_video() behavior.
    cam = vp.load_camera_params(vp_args.cam_yaml)

    cap = cv2.VideoCapture(vp_args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {vp_args.video}")

    ok, frame0 = cap.read()
    if not ok:
        raise RuntimeError("Cannot read first frame from video.")

    orig_h, orig_w = frame0.shape[:2]

    if vp_args.resize_w > 0:
        frame0 = vp.resize_if_needed(frame0, width=vp_args.resize_w)

    new_h, new_w = frame0.shape[:2]

    if (new_w != orig_w) or (new_h != orig_h):
        cam = vp.scale_camera_params(cam, orig_w, orig_h, new_w, new_h)

    print(f"[Frame] original size   = {orig_w}x{orig_h}")
    print(f"[Frame] processing size = {new_w}x{new_h}")
    print(f"[Camera] fx={cam.fx:.2f}, fy={cam.fy:.2f}, cx={cam.cx:.2f}, cy={cam.cy:.2f}")
    print(f"[Camera] H={cam.cam_height:.3f}, pitch={cam.pitch_deg:.3f}")

    seg_runner = vp.build_segmentation_runner(vp_args)
    depth_runner = vp.build_depth_runner(vp_args)
    det_runner = vp.build_detection_runner(vp_args)

    scale_state = vp.ScaleState()
    hood_state = vp.HoodState()

    saved_rows = []

    def process_and_maybe_save(frame_id: int, frame):
        nonlocal scale_state, hood_state, saved_rows

        panel, ev, scale_state, hood_state, runtime = vp.process_one_frame(
            frame_id=frame_id,
            bgr=frame,
            cam=cam,
            seg_runner=seg_runner,
            depth_runner=depth_runner,
            det_runner=det_runner,
            scale_state=scale_state,
            hood_state=hood_state,
            args=vp_args,
        )

        if frame_id not in targets:
            return

        if panel is None:
            raise RuntimeError(
                "process_one_frame returned panel=None. "
                "Make sure the imported inference script is not running with no_vis=True."
            )

        for meta in targets[frame_id]:
            out_path, name = save_panel(panel, args.out_dir, frame_id, meta)

            saved_rows.append({
                "frame_id": frame_id,
                "saved_name": name,
                "saved_path": out_path,
                "source": meta.get("source", ""),
                "segment_id": meta.get("segment_id", ""),
                "tag": meta.get("tag", ""),
                "failure_type": meta.get("failure_type", ""),
                "representative_reason": meta.get("representative_reason", ""),
                "start_frame": meta.get("start_frame", ""),
                "end_frame": meta.get("end_frame", ""),
                "length_frames": meta.get("length_frames", ""),
                "max_jump_ratio": meta.get("max_jump_ratio", ""),
                "mean_z_med": meta.get("mean_z_med", ""),
                "s_candidate": "" if ev.s_candidate is None else f"{ev.s_candidate:.6f}",
                "s_final": "" if scale_state.s_final is None else f"{scale_state.s_final:.6f}",
                "confidence": f"{ev.confidence:.6f}",
                "update_action": ev.update_action,
                "hood_conf": f"{ev.debug.get('hood_conf', 0.0):.6f}",
                "geo_points_num": ev.debug.get("geo_debug", {}).get("num_points", 0),
                "sample_points_num": ev.debug.get("sample_points_num", 0),
            })

        print(f"[SAVE] frame={frame_id} count={len(targets[frame_id])}")

    # Frame 0
    frame_id = 0
    process_and_maybe_save(frame_id, frame0)

    frame_id = 1

    while frame_id <= max_target:
        ok, frame = cap.read()
        if not ok:
            break

        if vp_args.resize_w > 0:
            frame = vp.resize_if_needed(frame, width=vp_args.resize_w)

        process_and_maybe_save(frame_id, frame)

        if frame_id % 50 == 0:
            print(f"[{frame_id}] processed, saved={len(saved_rows)}")

        frame_id += 1

    cap.release()

    saved_csv = os.path.join(args.out_dir, "saved_visuals.csv")
    fieldnames = [
        "frame_id",
        "saved_name",
        "saved_path",
        "source",
        "segment_id",
        "tag",
        "failure_type",
        "representative_reason",
        "start_frame",
        "end_frame",
        "length_frames",
        "max_jump_ratio",
        "mean_z_med",
        "s_candidate",
        "s_final",
        "confidence",
        "update_action",
        "hood_conf",
        "geo_points_num",
        "sample_points_num",
    ]
    write_csv_dict(saved_csv, saved_rows, fieldnames)

    print("=" * 60)
    print("Done.")
    print(f"Processed frames : {frame_id}")
    print(f"Saved panels     : {len(saved_rows)}")
    print(f"Output dir       : {args.out_dir}")
    print(f"Saved CSV        : {saved_csv}")
    print("=" * 60)


if __name__ == "__main__":
    main()
