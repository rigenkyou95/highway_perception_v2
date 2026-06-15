# -*- coding: utf-8 -*-
"""
analyze_failure_segments.py

Input:
  - scale_debug.csv
  - failure_cases.csv
  - runtime_per_frame.csv optional
  - source video optional

Output:
  - failure_segments.csv
  - failure_segment_summary.csv
  - representative_failure_segments.csv
  - segment_frames_raw/*.jpg optional

Purpose:
  Convert frame-level failure cases into segment-level failure cases.
"""

import os
import csv
import argparse
from pathlib import Path
from collections import Counter, defaultdict

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


def safe_mean(values):
    vals = [v for v in values if np.isfinite(v)]
    if len(vals) == 0:
        return np.nan
    return float(np.mean(vals))


def safe_median(values):
    vals = [v for v in values if np.isfinite(v)]
    if len(vals) == 0:
        return np.nan
    return float(np.median(vals))


def safe_max(values):
    vals = [v for v in values if np.isfinite(v)]
    if len(vals) == 0:
        return np.nan
    return float(np.max(vals))


def safe_min(values):
    vals = [v for v in values if np.isfinite(v)]
    if len(vals) == 0:
        return np.nan
    return float(np.min(vals))


def fmt_float(x, ndigits=6):
    if x is None or not np.isfinite(x):
        return ""
    return f"{float(x):.{ndigits}f}"


def count_to_str(counter):
    if not counter:
        return ""
    return ";".join([f"{k}:{v}" for k, v in counter.most_common()])


def split_failure_types(s):
    if s is None:
        return []
    s = str(s).strip()
    if s == "":
        return []
    return [x for x in s.split(";") if x]


def build_lookup(rows, key="frame_id"):
    out = {}
    for r in rows:
        fid = parse_int(r.get(key), default=-1)
        if fid >= 0:
            out[fid] = r
    return out


def make_segments(failure_rows, max_gap=2):
    """
    Merge nearby failure frames into one segment.

    max_gap:
      1 -> only strictly consecutive frames
      2 -> allow one normal frame between failures
      5 -> looser grouping
    """
    if len(failure_rows) == 0:
        return []

    failure_rows = sorted(failure_rows, key=lambda r: parse_int(r.get("frame_id"), -1))

    segments = []
    current = [failure_rows[0]]

    prev_fid = parse_int(failure_rows[0].get("frame_id"), -1)

    for r in failure_rows[1:]:
        fid = parse_int(r.get("frame_id"), -1)
        if fid < 0:
            continue

        if fid - prev_fid <= max_gap:
            current.append(r)
        else:
            segments.append(current)
            current = [r]

        prev_fid = fid

    if current:
        segments.append(current)

    return segments


def summarize_segment(segment_rows, scale_by_frame, runtime_by_frame, segment_id):
    frame_ids = [parse_int(r.get("frame_id"), -1) for r in segment_rows]
    frame_ids = [x for x in frame_ids if x >= 0]

    start_frame = int(min(frame_ids))
    end_frame = int(max(frame_ids))
    length = int(end_frame - start_frame + 1)
    failure_frame_count = int(len(frame_ids))
    center_frame = int(round(0.5 * (start_frame + end_frame)))

    failure_type_counter = Counter()
    update_action_counter = Counter()
    mode_counter = Counter()

    s_candidates = []
    s_finals = []
    confidences = []
    jump_ratios = []
    failure_scores = []
    geo_points = []
    ratio_raw = []
    ratio_filt = []
    z_min = []
    z_med = []
    z_max = []
    lane_pixels = []
    road_pixels = []
    hood_conf = []
    core_time = []
    hood_time = []
    depth_time = []
    seg_time = []

    for fr in segment_rows:
        fid = parse_int(fr.get("frame_id"), -1)

        for t in split_failure_types(fr.get("failure_types", "")):
            failure_type_counter[t] += 1

        row = scale_by_frame.get(fid, {})
        rt = runtime_by_frame.get(fid, {})

        update_action_counter[str(row.get("update_action", fr.get("update_action", "")))] += 1
        mode_counter[str(row.get("mode", fr.get("mode", "")))] += 1

        s_candidates.append(parse_float(row.get("s_candidate", fr.get("s_candidate"))))
        s_finals.append(parse_float(row.get("s_final", fr.get("s_final"))))
        confidences.append(parse_float(row.get("confidence", fr.get("confidence"))))
        jump_ratios.append(parse_float(fr.get("jump_ratio")))
        failure_scores.append(parse_float(fr.get("failure_score")))

        geo_points.append(parse_float(row.get("geo_points_num", fr.get("geo_points_num"))))
        ratio_raw.append(parse_float(row.get("ratio_points_raw", fr.get("ratio_points_raw"))))
        ratio_filt.append(parse_float(row.get("ratio_points_filt", fr.get("ratio_points_filt"))))

        z_min.append(parse_float(row.get("z_min")))
        z_med.append(parse_float(row.get("z_med")))
        z_max.append(parse_float(row.get("z_max")))

        lane_pixels.append(parse_float(row.get("lane_pixels")))
        road_pixels.append(parse_float(row.get("road_pixels")))
        hood_conf.append(parse_float(row.get("hood_conf")))

        core_time.append(parse_float(rt.get("inference_core_time_ms")))
        hood_time.append(parse_float(rt.get("hood_time_ms")))
        depth_time.append(parse_float(rt.get("depth_time_ms")))
        seg_time.append(parse_float(rt.get("segmentation_time_ms")))

    dominant_failure_type = ""
    if failure_type_counter:
        dominant_failure_type = failure_type_counter.most_common(1)[0][0]

    row_out = {
        "segment_id": segment_id,
        "start_frame": start_frame,
        "end_frame": end_frame,
        "center_frame": center_frame,
        "length_frames": length,
        "failure_frame_count": failure_frame_count,
        "failure_density": fmt_float(failure_frame_count / max(length, 1)),
        "dominant_failure_type": dominant_failure_type,
        "failure_type_counts": count_to_str(failure_type_counter),
        "update_action_counts": count_to_str(update_action_counter),
        "mode_counts": count_to_str(mode_counter),

        "max_failure_score": fmt_float(safe_max(failure_scores)),
        "mean_failure_score": fmt_float(safe_mean(failure_scores)),

        "mean_confidence": fmt_float(safe_mean(confidences)),
        "min_confidence": fmt_float(safe_min(confidences)),

        "mean_s_candidate": fmt_float(safe_mean(s_candidates)),
        "median_s_candidate": fmt_float(safe_median(s_candidates)),
        "mean_s_final": fmt_float(safe_mean(s_finals)),
        "median_s_final": fmt_float(safe_median(s_finals)),

        "max_jump_ratio": fmt_float(safe_max(jump_ratios)),
        "mean_jump_ratio": fmt_float(safe_mean(jump_ratios)),
        "median_jump_ratio": fmt_float(safe_median(jump_ratios)),

        "mean_geo_points": fmt_float(safe_mean(geo_points)),
        "min_geo_points": fmt_float(safe_min(geo_points)),
        "mean_ratio_raw": fmt_float(safe_mean(ratio_raw)),
        "mean_ratio_filt": fmt_float(safe_mean(ratio_filt)),

        "mean_z_min": fmt_float(safe_mean(z_min)),
        "mean_z_med": fmt_float(safe_mean(z_med)),
        "mean_z_max": fmt_float(safe_mean(z_max)),

        "mean_lane_pixels": fmt_float(safe_mean(lane_pixels)),
        "mean_road_pixels": fmt_float(safe_mean(road_pixels)),
        "mean_hood_conf": fmt_float(safe_mean(hood_conf)),

        "mean_core_time_ms": fmt_float(safe_mean(core_time)),
        "mean_hood_time_ms": fmt_float(safe_mean(hood_time)),
        "mean_depth_time_ms": fmt_float(safe_mean(depth_time)),
        "mean_seg_time_ms": fmt_float(safe_mean(seg_time)),
    }

    return row_out


def select_representative_segments(segment_rows, top_k_longest=10, top_k_jump=10, top_k_score=10):
    selected = {}

    def add_rows(rows, reason):
        for r in rows:
            sid = r["segment_id"]
            rr = dict(r)
            rr["representative_reason"] = reason
            if sid not in selected:
                selected[sid] = rr

    longest = sorted(segment_rows, key=lambda r: parse_int(r["length_frames"], 0), reverse=True)[:top_k_longest]
    add_rows(longest, "longest_segment")

    by_jump = sorted(segment_rows, key=lambda r: parse_float(r["max_jump_ratio"], 0.0), reverse=True)[:top_k_jump]
    add_rows(by_jump, "largest_jump")

    by_score = sorted(segment_rows, key=lambda r: parse_float(r["max_failure_score"], 0.0), reverse=True)[:top_k_score]
    add_rows(by_score, "highest_failure_score")

    reps = list(selected.values())
    reps = sorted(reps, key=lambda r: parse_int(r["start_frame"], 0))
    return reps


def save_segment_frames(video_path, representative_rows, out_dir, resize_w=1280):
    if video_path is None or str(video_path).strip() == "":
        return

    if not os.path.isfile(video_path):
        print(f"[Warning] video not found: {video_path}")
        return

    frame_out_dir = os.path.join(out_dir, "segment_frames_raw")
    os.makedirs(frame_out_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[Warning] cannot open video: {video_path}")
        return

    for r in representative_rows:
        sid = parse_int(r.get("segment_id"), -1)
        start = parse_int(r.get("start_frame"), -1)
        end = parse_int(r.get("end_frame"), -1)
        center = parse_int(r.get("center_frame"), -1)

        frames_to_save = [
            ("start", start),
            ("center", center),
            ("end", end),
        ]

        for tag, fid in frames_to_save:
            if fid < 0:
                continue

            cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
            ok, frame = cap.read()
            if not ok:
                continue

            if resize_w and resize_w > 0:
                h, w = frame.shape[:2]
                scale = float(resize_w) / float(w)
                frame = cv2.resize(frame, (resize_w, int(h * scale)), interpolation=cv2.INTER_LINEAR)

            lines = [
                f"seg={sid} {tag}",
                f"frame={fid}",
                f"type={r.get('dominant_failure_type', '')}",
                f"len={r.get('length_frames', '')}",
                f"max_jump={r.get('max_jump_ratio', '')}",
                f"mean_z_med={r.get('mean_z_med', '')}",
                f"action={r.get('update_action_counts', '')}",
            ]

            x, y = 20, 35
            for line in lines:
                cv2.putText(frame, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                            0.75, (0, 255, 255), 2, cv2.LINE_AA)
                y += 30

            out_name = f"segment_{sid:03d}_{tag}_frame_{fid:06d}.jpg"
            out_path = os.path.join(frame_out_dir, out_name)
            cv2.imwrite(out_path, frame)

    cap.release()
    print(f"[Frames] saved segment raw frames to: {frame_out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale-csv", required=True)
    parser.add_argument("--failure-csv", required=True)
    parser.add_argument("--runtime-csv", default="")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--video", default="")
    parser.add_argument("--resize-w", type=int, default=1280)

    parser.add_argument("--max-gap", type=int, default=2,
                        help="Merge failure frames into one segment if the frame gap is <= max_gap.")
    parser.add_argument("--min-segment-len", type=int, default=3,
                        help="Keep only segments whose total length is >= this value.")

    parser.add_argument("--top-k-longest", type=int, default=10)
    parser.add_argument("--top-k-jump", type=int, default=10)
    parser.add_argument("--top-k-score", type=int, default=10)

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    scale_rows = read_csv_dict(args.scale_csv)
    failure_rows = read_csv_dict(args.failure_csv)

    runtime_rows = []
    if args.runtime_csv and os.path.isfile(args.runtime_csv):
        runtime_rows = read_csv_dict(args.runtime_csv)

    scale_by_frame = build_lookup(scale_rows, key="frame_id")
    runtime_by_frame = build_lookup(runtime_rows, key="frame_id")

    raw_segments = make_segments(failure_rows, max_gap=args.max_gap)

    segment_rows = []
    sid = 0
    for seg in raw_segments:
        frame_ids = [parse_int(r.get("frame_id"), -1) for r in seg]
        frame_ids = [x for x in frame_ids if x >= 0]
        if len(frame_ids) == 0:
            continue

        seg_len = max(frame_ids) - min(frame_ids) + 1
        if seg_len < args.min_segment_len:
            continue

        sid += 1
        segment_rows.append(
            summarize_segment(
                segment_rows=seg,
                scale_by_frame=scale_by_frame,
                runtime_by_frame=runtime_by_frame,
                segment_id=sid,
            )
        )

    fieldnames = [
        "segment_id",
        "start_frame",
        "end_frame",
        "center_frame",
        "length_frames",
        "failure_frame_count",
        "failure_density",
        "dominant_failure_type",
        "failure_type_counts",
        "update_action_counts",
        "mode_counts",
        "max_failure_score",
        "mean_failure_score",
        "mean_confidence",
        "min_confidence",
        "mean_s_candidate",
        "median_s_candidate",
        "mean_s_final",
        "median_s_final",
        "max_jump_ratio",
        "mean_jump_ratio",
        "median_jump_ratio",
        "mean_geo_points",
        "min_geo_points",
        "mean_ratio_raw",
        "mean_ratio_filt",
        "mean_z_min",
        "mean_z_med",
        "mean_z_max",
        "mean_lane_pixels",
        "mean_road_pixels",
        "mean_hood_conf",
        "mean_core_time_ms",
        "mean_hood_time_ms",
        "mean_depth_time_ms",
        "mean_seg_time_ms",
    ]

    segments_csv = os.path.join(args.out_dir, "failure_segments.csv")
    write_csv_dict(segments_csv, segment_rows, fieldnames)

    # Summary by dominant type
    summary = defaultdict(lambda: {
        "segment_count": 0,
        "total_length_frames": 0,
        "total_failure_frames": 0,
        "max_segment_length": 0,
        "max_jump_ratio": 0.0,
    })

    for r in segment_rows:
        t = r["dominant_failure_type"]
        summary[t]["segment_count"] += 1
        summary[t]["total_length_frames"] += parse_int(r["length_frames"], 0)
        summary[t]["total_failure_frames"] += parse_int(r["failure_frame_count"], 0)
        summary[t]["max_segment_length"] = max(
            summary[t]["max_segment_length"],
            parse_int(r["length_frames"], 0)
        )
        summary[t]["max_jump_ratio"] = max(
            summary[t]["max_jump_ratio"],
            parse_float(r["max_jump_ratio"], 0.0)
        )

    summary_rows = []
    for t, v in sorted(summary.items(), key=lambda x: x[0]):
        summary_rows.append({
            "dominant_failure_type": t,
            "segment_count": v["segment_count"],
            "total_length_frames": v["total_length_frames"],
            "total_failure_frames": v["total_failure_frames"],
            "mean_segment_length": fmt_float(v["total_length_frames"] / max(v["segment_count"], 1)),
            "max_segment_length": v["max_segment_length"],
            "max_jump_ratio": fmt_float(v["max_jump_ratio"]),
        })

    summary_csv = os.path.join(args.out_dir, "failure_segment_summary.csv")
    write_csv_dict(
        summary_csv,
        summary_rows,
        [
            "dominant_failure_type",
            "segment_count",
            "total_length_frames",
            "total_failure_frames",
            "mean_segment_length",
            "max_segment_length",
            "max_jump_ratio",
        ],
    )

    reps = select_representative_segments(
        segment_rows,
        top_k_longest=args.top_k_longest,
        top_k_jump=args.top_k_jump,
        top_k_score=args.top_k_score,
    )

    rep_fields = fieldnames + ["representative_reason"]
    reps_csv = os.path.join(args.out_dir, "representative_failure_segments.csv")
    write_csv_dict(reps_csv, reps, rep_fields)

    save_segment_frames(args.video, reps, args.out_dir, resize_w=args.resize_w)

    print("=" * 60)
    print("Failure segment analysis done.")
    print(f"Frame-level failures : {len(failure_rows)}")
    print(f"Raw segments         : {len(raw_segments)}")
    print(f"Kept segments        : {len(segment_rows)}")
    print(f"Segments CSV         : {segments_csv}")
    print(f"Summary CSV          : {summary_csv}")
    print(f"Representative CSV   : {reps_csv}")
    print("=" * 60)

    for r in summary_rows:
        print(
            f"{r['dominant_failure_type']}: "
            f"segments={r['segment_count']}, "
            f"total_len={r['total_length_frames']}, "
            f"max_len={r['max_segment_length']}, "
            f"max_jump={r['max_jump_ratio']}"
        )


if __name__ == "__main__":
    main()
