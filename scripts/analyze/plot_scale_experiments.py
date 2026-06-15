# -*- coding: utf-8 -*-
import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def read_csv_safe(path):
    df = pd.read_csv(path)

    numeric_cols = [
        "s_candidate", "s_final", "confidence",
        "score_lane_visibility", "score_temporal_scale_stability",
        "score_mask_visibility", "hood_conf",
        "geo_points_num", "sample_points_num",
        "z_min", "z_med", "z_max",
        "mean_point_weight",
        "ratio_points_raw", "ratio_points_filt"
    ]

    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "mode" in df.columns:
        mode_map = {"INIT": 0, "TRACKING": 1, "HOLD": 2}
        df["mode_id"] = df["mode"].map(mode_map).fillna(-1)

    return df


def plot_single_run(df, out_dir, name):
    ensure_dir(out_dir)

    # 1. s_candidate vs s_final
    plt.figure(figsize=(12, 5))
    plt.plot(df["frame_id"], df["s_candidate"], label="s_candidate", linewidth=1)
    plt.plot(df["frame_id"], df["s_final"], label="s_final", linewidth=2)
    plt.xlabel("Frame")
    plt.ylabel("Scale")
    plt.title(f"Scale Candidate and Stabilized Scale - {name}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{name}_scale_curve.png"), dpi=300)
    plt.close()

    # 2. confidence
    plt.figure(figsize=(12, 4))
    plt.plot(df["frame_id"], df["confidence"], label="confidence", linewidth=1.5)
    if "hood_conf" in df.columns:
        plt.plot(df["frame_id"], df["hood_conf"], label="hood_conf", linewidth=1)
    plt.xlabel("Frame")
    plt.ylabel("Confidence")
    plt.title(f"Confidence Curve - {name}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{name}_confidence_curve.png"), dpi=300)
    plt.close()

    # 3. anchor points
    plt.figure(figsize=(12, 4))
    plt.plot(df["frame_id"], df["geo_points_num"], label="valid geo points", linewidth=1.5)
    if "sample_points_num" in df.columns:
        plt.plot(df["frame_id"], df["sample_points_num"], label="sample points", linewidth=1)
    plt.xlabel("Frame")
    plt.ylabel("Number of Points")
    plt.title(f"Anchor Point Count - {name}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{name}_anchor_points.png"), dpi=300)
    plt.close()

    # 4. z anchor distribution
    if "z_min" in df.columns and "z_med" in df.columns and "z_max" in df.columns:
        plt.figure(figsize=(12, 4))
        plt.plot(df["frame_id"], df["z_min"], label="z_min", linewidth=1)
        plt.plot(df["frame_id"], df["z_med"], label="z_med", linewidth=1.5)
        plt.plot(df["frame_id"], df["z_max"], label="z_max", linewidth=1)
        plt.xlabel("Frame")
        plt.ylabel("Metric Depth Z (m)")
        plt.title(f"Anchor Depth Distribution - {name}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{name}_anchor_z_distribution.png"), dpi=300)
        plt.close()

    # 5. mode timeline
    if "mode_id" in df.columns:
        plt.figure(figsize=(12, 3))
        plt.plot(df["frame_id"], df["mode_id"], linewidth=1.5)
        plt.yticks([0, 1, 2], ["INIT", "TRACKING", "HOLD"])
        plt.xlabel("Frame")
        plt.ylabel("Mode")
        plt.title(f"State Machine Mode - {name}")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{name}_mode_timeline.png"), dpi=300)
        plt.close()


def compute_metrics(df, name):
    total = len(df)
    valid_s = df["s_final"].dropna()
    valid_cand = df["s_candidate"].dropna()

    if len(valid_s) >= 2:
        ds = valid_s.diff().abs().dropna()
        pct = valid_s.pct_change().abs().replace([np.inf, -np.inf], np.nan).dropna()
    else:
        ds = pd.Series(dtype=float)
        pct = pd.Series(dtype=float)

    tracking_ratio = float((df["mode"] == "TRACKING").sum()) / max(total, 1) if "mode" in df.columns else np.nan
    hold_ratio = float((df["mode"] == "HOLD").sum()) / max(total, 1) if "mode" in df.columns else np.nan
    candidate_valid_ratio = float(df["s_candidate"].notna().sum()) / max(total, 1)

    return {
        "run": name,
        "frames": total,
        "candidate_valid_ratio": candidate_valid_ratio,
        "tracking_ratio": tracking_ratio,
        "hold_ratio": hold_ratio,
        "s_final_mean": float(valid_s.mean()) if len(valid_s) > 0 else np.nan,
        "s_final_std": float(valid_s.std()) if len(valid_s) > 1 else np.nan,
        "s_final_cv": float(valid_s.std() / valid_s.mean()) if len(valid_s) > 1 and valid_s.mean() != 0 else np.nan,
        "mean_abs_scale_change": float(ds.mean()) if len(ds) > 0 else np.nan,
        "max_abs_scale_change": float(ds.max()) if len(ds) > 0 else np.nan,
        "mean_abs_pct_change": float(pct.mean()) if len(pct) > 0 else np.nan,
        "large_jump_ratio_5pct": float((pct > 0.05).sum()) / max(len(pct), 1) if len(pct) > 0 else np.nan,
        "geo_points_mean": float(df["geo_points_num"].mean()) if "geo_points_num" in df.columns else np.nan,
        "geo_points_min": float(df["geo_points_num"].min()) if "geo_points_num" in df.columns else np.nan,
        "z_med_mean": float(df["z_med"].mean()) if "z_med" in df.columns else np.nan,
        "confidence_mean": float(df["confidence"].mean()) if "confidence" in df.columns else np.nan,
    }


def plot_compare_runs(run_items, out_dir):
    ensure_dir(out_dir)

    plt.figure(figsize=(12, 5))
    for name, df in run_items:
        plt.plot(df["frame_id"], df["s_final"], label=name, linewidth=1.5)
    plt.xlabel("Frame")
    plt.ylabel("s_final")
    plt.title("Comparison of Stabilized Scale")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "compare_s_final.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(12, 5))
    for name, df in run_items:
        if "z_med" in df.columns:
            plt.plot(df["frame_id"], df["z_med"], label=name, linewidth=1.5)
    plt.xlabel("Frame")
    plt.ylabel("Median Anchor Z (m)")
    plt.title("Comparison of Anchor Median Depth")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "compare_anchor_z_med.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(12, 5))
    for name, df in run_items:
        if "geo_points_num" in df.columns:
            plt.plot(df["frame_id"], df["geo_points_num"], label=name, linewidth=1.2)
    plt.xlabel("Frame")
    plt.ylabel("Valid Anchor Points")
    plt.title("Comparison of Valid Anchor Points")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "compare_anchor_points.png"), dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True,
                        help="Format: name=path/to/scale_debug.csv")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    ensure_dir(args.out_dir)

    run_items = []
    metrics = []

    for item in args.runs:
        name, path = item.split("=", 1)
        df = read_csv_safe(path)
        run_items.append((name, df))

        single_out = os.path.join(args.out_dir, name)
        plot_single_run(df, single_out, name)

        metrics.append(compute_metrics(df, name))

    plot_compare_runs(run_items, args.out_dir)

    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(os.path.join(args.out_dir, "metrics_summary.csv"), index=False, encoding="utf-8-sig")
    print(metrics_df)


if __name__ == "__main__":
    main()
