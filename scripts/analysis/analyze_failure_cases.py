# -*- coding: utf-8 -*-
"""
analyze_failure_cases.py

Analyze failure cases from:
  - scale_debug.csv
  - runtime_per_frame.csv optional

Outputs:
  - failure_cases.csv
  - failure_summary.csv
  - representative_failure_frames.csv
  - optional raw frame images for representative failure cases
"""

import os
import csv
import argparse
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np


def parse_float(x, default=np.nan):
    if x is None:
        return default
    x = str(x).strip()
    if x == "":
        return default
    try:
        return float(x)
    except Exception:
        return default


def parse_int(x, default=0):
    if x is None:
        return default
    x = str(x).strip()
    if x == "":
        return default
    try:
        return int(float(x))
    except Exception:
        return default


def read_csv_dict(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def write_csv_dict(path, rows, fieldnames):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def load_runtime_by_frame(runtime_csv):
    if runtime_csv is None or not os.path.isfile(runtime_csv):
        return {}

    runtime_rows = read_csv_dict(runtime_csv)
    out = {}
    for r in runtime_rows:
        fid = parse_int(r.get("frame_id"), default=-1)
        if fid >= 0:
            out[fid] = r
    return out


def classify_failure(row,
                     low_conf_thr=0.50,
                     min_geo_points=5,
                     min_ratio_points=5,
                     jump_thr=0.30):
    """
    Return:
      failure_types: list[str]
      failure_score: higher means more severe
      jump_ratio: float or nan
    """
    failure_types = []

    s_candidate = parse_float(row.get("s_candidate"))
    s_final = parse_float(row.get("s_final"))
    confidence = parse_float(row.get("confidence"), default=0.0)

    update_action = str(row.get("update_action", "")).strip()
    mode = str(row.get("mode", "")).strip()

    geo_points = parse_int(row.get("geo_points_num"), default=0)
    ratio_raw = parse_int(row.get("ratio_points_raw"), default=0)
    ratio_filt = parse_int(row.get("ratio_points_filt"), default=0)

    fit_reason = str(row.get("fit_reason", "")).strip()

    jump_ratio = np.nan
    if np.isfinite(s_candidate) and np.isfinite(s_final) and abs(s_final) > 1e-9:
        jump_ratio = abs(s_candidate - s_final) / abs(s_final)

    if not np.isfinite(s_candidate):
        failure_types.append("NO_SCALE_CANDIDATE")

    if confidence < low_conf_thr:
        failure_types.append("LOW_CONFIDENCE")

    if update_action in ["FREEZE", "HOLD", "JUMP_REJECT"]:
        failure_types.append("FREEZE_OR_HOLD")

    if geo_points < min_geo_points:
        failure_types.append("FEW_GEO_POINTS")

    if ratio_raw < min_ratio_points:
        failure_types.append("FEW_RATIO_POINTS")

    if np.isfinite(jump_ratio) and jump_ratio > jump_thr:
        failure_types.append("SCALE_JUMP")

    if fit_reason not in ["", "ok"] and fit_reason != "None":
        if fit_reason in [
            "no_geo_points",
            "too_few_ratio_points",
            "too_few_after_iqr",
            "too_few_valid_bands",
            "z_range_too_small",
        ]:
            failure_types.append("FIT_" + fit_reason.upper())

    # severity score
    score = 0.0
    score += 2.0 if "NO_SCALE_CANDIDATE" in failure_types else 0.0
    score += 1.5 if "FREEZE_OR_HOLD" in failure_types else 0.0
    score += 1.2 if "SCALE_JUMP" in failure_types else 0.0
    score += 1.0 if "FEW_GEO_POINTS" in failure_types else 0.0
    score += 1.0 if "FEW_RATIO_POINTS" in failure_types else 0.0
    score += max(0.0, low_conf_thr - confidence)

    return failure_types, score, jump_ratio


def extract_representatives(failure_rows, max_per_type=8):
    """
    Pick representative frames for each failure type.
    Higher failure_score first, with temporal spacing to avoid near-duplicates.
    """
    by_type = defaultdict(list)
    for r in failure_rows:
        types = str(r["failure_types"]).split(";")
        for t in types:
            if t:
                by_type[t].append(r)

    reps = []
    used_frames = set()

    for t, rows in by_type.items():
        rows_sorted = sorted(
            rows,
            key=lambda x: parse_float(x.get("failure_score"), default=0.0),
            reverse=True
        )

        selected = []
        for r in rows_sorted:
            fid = parse_int(r["frame_id"], default=-1)
            if fid < 0:
                continue

            # avoid selecting too many adjacent frames
            too_close = any(abs(fid - old) < 15 for old in used_frames)
            if too_close:
                continue

            rr = dict(r)
            rr["representative_type"] = t
            selected.append(rr)
            used_frames.add(fid)

            if len(selected) >= max_per_type:
                break

        reps.extend(selected)

    reps = sorted(reps, key=lambda x: (x["representative_type"], parse_int(x["frame_id"])))
    return reps


def save_raw_frames(video_path, reps, out_dir, resize_w=0):
    if video_path is None or video_path == "":
        return

    if not os.path.isfile(video_path):
        print(f"[Warning] video not found: {video_path}")
        return

    frame_dir = os.path.join(out_dir, "failure_frames_raw")
    os.makedirs(frame_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[Warning] cannot open video: {video_path}")
        return

    wanted = {}
    for r in reps:
        fid = parse_int(r.get("frame_id"), default=-1)
        if fid >= 0:
            wanted[fid] = r

    for fid, r in wanted.items():
        cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
        ok, frame = cap.read()
        if not ok:
            continue

        if resize_w and resize_w > 0:
            h, w = frame.shape[:2]
            scale = float(resize_w) / float(w)
            frame = cv2.resize(frame, (resize_w, int(h * scale)), interpolation=cv2.INTER_LINEAR)

        ftype = str(r.get("representative_type", "FAIL"))
        score = parse_float(r.get("failure_score"), default=0.0)
        conf = parse_float(r.get("confidence"), default=0.0)

        text_lines = [
            f"frame={fid}",
            f"type={ftype}",
            f"score={score:.2f}",
            f"conf={conf:.3f}",
            f"action={r.get('update_action', '')}",
            f"geo={r.get('geo_points_num', '')}",
            f"ratio={r.get('ratio_points_raw', '')}/{r.get('ratio_points_filt', '')}",
        ]

        x, y = 20, 35
        for line in text_lines:
            cv2.putText(frame, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 255), 2, cv2.LINE_AA)
            y += 32

        out_name = f"{ftype}_frame_{fid:06d}.jpg"
        out_path = os.path.join(frame_dir, out_name)
        cv2.imwrite(out_path, frame)

    cap.release()
    print(f"[Frames] saved representative raw frames to: {frame_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale-csv", required=True, help="Path to scale_debug.csv")
    parser.add_argument("--runtime-csv", default="", help="Optional path to runtime_per_frame.csv")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--video", default="", help="Optional source video path for raw frame extraction")
    parser.add_argument("--resize-w", type=int, default=1280)
    parser.add_argument("--low-conf-thr", type=float, default=0.50)
    parser.add_argument("--min-geo-points", type=int, default=5)
    parser.add_argument("--min-ratio-points", type=int, default=5)
    parser.add_argument("--jump-thr", type=float, default=0.30)
    parser.add_argument("--max-per-type", type=int, default=8)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    scale_rows = read_csv_dict(args.scale_csv)
    runtime_by_frame = load_runtime_by_frame(args.runtime_csv)

    failure_rows = []
    summary_count = defaultdict(int)
    total_rows = len(scale_rows)

    for row in scale_rows:
        fid = parse_int(row.get("frame_id"), default=-1)
        failure_types, failure_score, jump_ratio = classify_failure(
            row,
            low_conf_thr=args.low_conf_thr,
            min_geo_points=args.min_geo_points,
            min_ratio_points=args.min_ratio_points,
            jump_thr=args.jump_thr,
        )

        if len(failure_types) == 0:
            continue

        out = dict(row)
        out["failure_types"] = ";".join(sorted(set(failure_types)))
        out["failure_score"] = f"{failure_score:.6f}"
        out["jump_ratio"] = "" if not np.isfinite(jump_ratio) else f"{jump_ratio:.6f}"

        rt = runtime_by_frame.get(fid, {})
        if rt:
            out["hood_time_ms"] = rt.get("hood_time_ms", "")
            out["depth_time_ms"] = rt.get("depth_time_ms", "")
            out["segmentation_time_ms"] = rt.get("segmentation_time_ms", "")
            out["inference_core_time_ms"] = rt.get("inference_core_time_ms", "")

        for t in sorted(set(failure_types)):
            summary_count[t] += 1

        failure_rows.append(out)

    failure_rows = sorted(
        failure_rows,
        key=lambda x: parse_float(x.get("failure_score"), default=0.0),
        reverse=True
    )

    # fieldnames
    base_fields = list(scale_rows[0].keys()) if scale_rows else []
    extra_fields = [
        "failure_types",
        "failure_score",
        "jump_ratio",
        "hood_time_ms",
        "depth_time_ms",
        "segmentation_time_ms",
        "inference_core_time_ms",
    ]
    fieldnames = base_fields + [x for x in extra_fields if x not in base_fields]

    failure_csv = os.path.join(args.out_dir, "failure_cases.csv")
    write_csv_dict(failure_csv, failure_rows, fieldnames)

    summary_rows = []
    for k, v in sorted(summary_count.items(), key=lambda x: x[0]):
        summary_rows.append({
            "failure_type": k,
            "count": v,
            "ratio": f"{v / max(total_rows, 1):.6f}",
            "total_frames": total_rows,
        })

    summary_csv = os.path.join(args.out_dir, "failure_summary.csv")
    write_csv_dict(summary_csv, summary_rows, ["failure_type", "count", "ratio", "total_frames"])

    reps = extract_representatives(failure_rows, max_per_type=args.max_per_type)
    reps_csv = os.path.join(args.out_dir, "representative_failure_frames.csv")
    rep_fields = fieldnames + ["representative_type"]
    write_csv_dict(reps_csv, reps, rep_fields)

    save_raw_frames(args.video, reps, args.out_dir, resize_w=args.resize_w)

    print("=" * 60)
    print("Failure analysis done.")
    print(f"Total frames        : {total_rows}")
    print(f"Failure frames      : {len(failure_rows)}")
    print(f"Failure frame ratio : {len(failure_rows) / max(total_rows, 1):.6f}")
    print(f"Failure cases CSV   : {failure_csv}")
    print(f"Failure summary CSV : {summary_csv}")
    print(f"Representative CSV  : {reps_csv}")
    print("=" * 60)

    for r in summary_rows:
        print(f"{r['failure_type']}: {r['count']} / {r['total_frames']} = {r['ratio']}")


if __name__ == "__main__":
    main()
