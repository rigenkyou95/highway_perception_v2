# -*- coding: utf-8 -*-
"""
collect_kitti_sequence_summary.py

Collect KITTI two-sequence evaluation results.

This script summarizes:
1. Baseline evaluation results
2. Geometry-anchor evaluation results
3. Geometry scale debug analysis

Expected current outputs:

0005:
    outputs/kitti/eval_depth/2011_09_26_drive_0005_sync_eigen_crop_state/kitti_depth_eval_summary.csv
    outputs/kitti/eval_depth/kitti_geometry_anchor_H165_P0_Z10_40/kitti_geometry_eval_summary.csv
    outputs/kitti/eval_depth/kitti_geometry_anchor_H165_P0_Z10_40/debug_analysis/geometry_scale_debug_summary.csv

0009:
    outputs/kitti/eval_depth/2011_09_26_drive_0009_sync_baselines/kitti_depth_eval_summary.csv
    outputs/kitti/eval_depth/kitti_geometry_anchor_0009_H165_P0_Z10_40/kitti_geometry_eval_summary.csv
    outputs/kitti/eval_depth/kitti_geometry_anchor_0009_H165_P0_Z10_40/debug_analysis/geometry_scale_debug_summary.csv

Output:
    outputs/kitti/eval_depth/kitti_two_sequence_summary.csv
    outputs/kitti/eval_depth/kitti_two_sequence_summary.md
"""

import os
import argparse
import pandas as pd


SEQUENCES = {
    "0005": {
        "name": "2011_09_26_drive_0005_sync",
        "baseline_csv": r"outputs\kitti\eval_depth\kitti_state_cd_balance\kitti_depth_eval_summary.csv",
        "geometry_eval_csv": r"outputs\kitti\eval_depth\kitti_geometry_anchor_H165_P0_Z10_40\kitti_geometry_eval_summary.csv",
        "geometry_debug_csv": r"outputs\kitti\eval_depth\kitti_geometry_anchor_H165_P0_Z10_40\debug_analysis\geometry_scale_debug_summary.csv",
    },
    "0009": {
        "name": "2011_09_26_drive_0009_sync",
        "baseline_csv": r"outputs\kitti\eval_depth\2011_09_26_drive_0009_sync_baselines\kitti_depth_eval_summary.csv",
        "geometry_eval_csv": r"outputs\kitti\eval_depth\kitti_geometry_anchor_0009_H165_P0_Z10_40\kitti_geometry_eval_summary.csv",
        "geometry_debug_csv": r"outputs\kitti\eval_depth\kitti_geometry_anchor_0009_H165_P0_Z10_40\debug_analysis\geometry_scale_debug_summary.csv",
    },
}


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def read_csv_safe(path: str):
    if not os.path.isfile(path):
        print(f"[Missing] {path}")
        return None
    return pd.read_csv(path)


def get_method_row(df: pd.DataFrame, method: str):
    if df is None or "method" not in df.columns:
        return None

    hit = df[df["method"].astype(str) == method]
    if len(hit) == 0:
        return None

    return hit.iloc[0].to_dict()


def get_first_row(df: pd.DataFrame):
    if df is None or len(df) == 0:
        return {}
    return df.iloc[0].to_dict()


def safe_get(row, key, default=None):
    if row is None:
        return default
    return row.get(key, default)


def add_method_metrics(out: dict, prefix: str, row: dict):
    out[f"{prefix}_frames"] = safe_get(row, "frames")
    out[f"{prefix}_scale_mean"] = safe_get(row, "scale_mean")
    out[f"{prefix}_scale_std"] = safe_get(row, "scale_std")
    out[f"{prefix}_scale_delta_mean"] = safe_get(row, "scale_delta_mean")
    out[f"{prefix}_scale_delta_p95"] = safe_get(row, "scale_delta_p95")
    out[f"{prefix}_AbsRel"] = safe_get(row, "AbsRel")
    out[f"{prefix}_SqRel"] = safe_get(row, "SqRel")
    out[f"{prefix}_RMSE"] = safe_get(row, "RMSE")
    out[f"{prefix}_RMSElog"] = safe_get(row, "RMSElog")
    out[f"{prefix}_delta1"] = safe_get(row, "delta1")
    out[f"{prefix}_delta2"] = safe_get(row, "delta2")
    out[f"{prefix}_delta3"] = safe_get(row, "delta3")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-csv",
        default=r"outputs\kitti\eval_depth\kitti_two_sequence_summary.csv"
    )
    parser.add_argument(
        "--out-md",
        default=r"outputs\kitti\eval_depth\kitti_two_sequence_summary.md"
    )
    args = parser.parse_args()

    rows = []

    for seq_id, paths in SEQUENCES.items():
        print(f"\n[Sequence] {seq_id}: {paths['name']}")

        baseline_df = read_csv_safe(paths["baseline_csv"])
        geometry_eval_df = read_csv_safe(paths["geometry_eval_csv"])
        geometry_debug_df = read_csv_safe(paths["geometry_debug_csv"])

        # Baseline rows
        raw_row = get_method_row(baseline_df, "raw_unscaled")
        per_frame_row = get_method_row(baseline_df, "per_frame_median_scale")
        sequence_row = get_method_row(baseline_df, "sequence_median_scale")
        moving_average_row = get_method_row(baseline_df, "moving_average_w15")
        ema_row = get_method_row(baseline_df, "ema_a0.08")
        median_filter_row = get_method_row(baseline_df, "median_filter_w15")
        gt_state_row = get_method_row(baseline_df, "proposed_state_machine")

        # Geometry rows
        geom_candidate_row = get_method_row(geometry_eval_df, "geometry_candidate")
        geom_ema_row = get_method_row(geometry_eval_df, "geometry_ema")
        geom_state_row = get_method_row(geometry_eval_df, "geometry_state_machine")

        # Some 0005 geometry eval uses oracle method names with suffix.
        geom_per_frame_oracle_row = get_method_row(
            geometry_eval_df,
            "per_frame_median_scale_oracle"
        )
        geom_sequence_oracle_row = get_method_row(
            geometry_eval_df,
            "sequence_median_scale_oracle"
        )

        debug_row = get_first_row(geometry_debug_df)

        out = {
            "sequence_id": seq_id,
            "sequence_name": paths["name"],

            # Geometry debug summary
            "geo_frames_total": debug_row.get("frames_total"),
            "geo_frames_valid": debug_row.get("frames_geo_valid"),
            "geo_valid_ratio": debug_row.get("geo_valid_ratio"),
            "cam_height": debug_row.get("cam_height"),
            "pitch_deg": debug_row.get("pitch_deg"),
            "s_oracle_mean": debug_row.get("s_oracle_mean"),
            "s_geo_mean": debug_row.get("s_geo_mean"),
            "scale_rel_err_mean": debug_row.get("scale_rel_err_mean"),
            "scale_rel_err_median": debug_row.get("scale_rel_err_median"),
            "scale_rel_err_p95": debug_row.get("scale_rel_err_p95"),
            "corr_s_geo_oracle": debug_row.get("corr_s_geo_oracle"),
            "geo_points_mean": debug_row.get("geo_points_mean"),
            "geo_points_min": debug_row.get("geo_points_min"),
            "z_med_mean": debug_row.get("z_med_mean"),
        }

        # Baseline method metrics
        add_method_metrics(out, "raw_unscaled", raw_row)
        add_method_metrics(out, "baseline_per_frame_oracle", per_frame_row)
        add_method_metrics(out, "baseline_sequence_oracle", sequence_row)
        add_method_metrics(out, "baseline_moving_average", moving_average_row)
        add_method_metrics(out, "baseline_ema", ema_row)
        add_method_metrics(out, "baseline_median_filter", median_filter_row)
        add_method_metrics(out, "baseline_gt_state_machine", gt_state_row)

        # Geometry method metrics
        add_method_metrics(out, "geometry_per_frame_oracle", geom_per_frame_oracle_row)
        add_method_metrics(out, "geometry_sequence_oracle", geom_sequence_oracle_row)
        add_method_metrics(out, "geometry_candidate", geom_candidate_row)
        add_method_metrics(out, "geometry_ema", geom_ema_row)
        add_method_metrics(out, "geometry_state_machine", geom_state_row)

        rows.append(out)

    summary = pd.DataFrame(rows)

    out_dir = os.path.dirname(args.out_csv)
    ensure_dir(out_dir)

    summary.to_csv(args.out_csv, index=False, encoding="utf-8-sig")

    # Compact markdown table for reporting
    compact_cols = [
        "sequence_id",
        "geo_frames_total",
        "geo_frames_valid",
        "geo_valid_ratio",
        "scale_rel_err_mean",
        "scale_rel_err_p95",
        "corr_s_geo_oracle",
        "geo_points_mean",
        "z_med_mean",
        "baseline_per_frame_oracle_AbsRel",
        "baseline_sequence_oracle_AbsRel",
        "baseline_moving_average_AbsRel",
        "baseline_ema_AbsRel",
        "baseline_median_filter_AbsRel",
        "baseline_gt_state_machine_AbsRel",
        "geometry_candidate_frames",
        "geometry_candidate_AbsRel",
        "geometry_candidate_RMSE",
        "geometry_state_machine_frames",
        "geometry_state_machine_AbsRel",
        "geometry_state_machine_RMSE",
    ]

    compact_cols = [c for c in compact_cols if c in summary.columns]
    compact = summary[compact_cols].copy()

    md_text = compact.to_markdown(index=False)

    with open(args.out_md, "w", encoding="utf-8") as f:
        f.write("# KITTI Two-Sequence Summary\n\n")
        f.write(md_text)
        f.write("\n")

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 500)

    print("\n=== KITTI Two-Sequence Summary ===")
    print(summary.to_string(index=False))

    print("\n=== Compact Summary ===")
    print(compact.to_string(index=False))

    print("\nSaved:")
    print(args.out_csv)
    print(args.out_md)


if __name__ == "__main__":
    main()
