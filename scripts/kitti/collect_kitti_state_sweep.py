# -*- coding: utf-8 -*-
"""
collect_kitti_state_sweep.py

Collect KITTI state-machine parameter sweep results.

This script reads multiple kitti_depth_eval_summary.csv files and builds one
summary table for:
- common temporal baselines
- proposed state-machine variants

Expected input directories:
    outputs/kitti/eval_depth/kitti_state_mid
    outputs/kitti/eval_depth/kitti_state_fast
    outputs/kitti/eval_depth/kitti_state_cd_balance
    outputs/kitti/eval_depth/kitti_state_ema_like

Output:
    outputs/kitti/eval_depth/state_sweep_summary.csv
"""

import os
import argparse
import pandas as pd


DEFAULT_RUNS = {
    "state_mid": r"outputs\kitti\eval_depth\kitti_state_mid\kitti_depth_eval_summary.csv",
    "state_fast": r"outputs\kitti\eval_depth\kitti_state_fast\kitti_depth_eval_summary.csv",
    "state_cd_balance": r"outputs\kitti\eval_depth\kitti_state_cd_balance\kitti_depth_eval_summary.csv",
    "state_ema_like": r"outputs\kitti\eval_depth\kitti_state_ema_like\kitti_depth_eval_summary.csv",
}


COMMON_BASELINES = [
    "raw_unscaled",
    "per_frame_median_scale",
    "sequence_median_scale",
    "moving_average_w15",
    "ema_a0.08",
    "median_filter_w15",
]


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def read_summary(path: str) -> pd.DataFrame:
    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    df = pd.read_csv(path)

    if "method" not in df.columns:
        raise ValueError(f"Missing 'method' column in {path}")

    return df


def extract_row(df: pd.DataFrame, method_name: str):
    hit = df[df["method"].astype(str) == method_name]
    if len(hit) == 0:
        return None
    return hit.iloc[0].to_dict()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-csv",
        type=str,
        default=r"outputs\kitti\eval_depth\state_sweep_summary.csv",
        help="Output CSV path."
    )
    parser.add_argument(
        "--include-baselines-from",
        type=str,
        default="state_cd_balance",
        help="Which run to use for common baselines."
    )
    args = parser.parse_args()

    rows = []

    # -----------------------------------------------------
    # 1. Load all available sweep runs
    # -----------------------------------------------------
    loaded = {}

    for run_name, csv_path in DEFAULT_RUNS.items():
        if not os.path.isfile(csv_path):
            print(f"[Skip] Missing: {run_name} -> {csv_path}")
            continue

        df = read_summary(csv_path)
        loaded[run_name] = df
        print(f"[Load] {run_name}: {csv_path}")

    if len(loaded) == 0:
        raise RuntimeError("No KITTI state sweep summary files found.")

    # -----------------------------------------------------
    # 2. Add common baselines from one selected run
    # -----------------------------------------------------
    if args.include_baselines_from not in loaded:
        fallback_name = list(loaded.keys())[0]
        print(
            f"[Warn] Baseline source '{args.include_baselines_from}' not found. "
            f"Use '{fallback_name}' instead."
        )
        baseline_df = loaded[fallback_name]
        baseline_source = fallback_name
    else:
        baseline_df = loaded[args.include_baselines_from]
        baseline_source = args.include_baselines_from

    for method in COMMON_BASELINES:
        row = extract_row(baseline_df, method)
        if row is None:
            print(f"[Skip] Baseline not found: {method}")
            continue

        row_out = {
            "setting": method,
            "source_run": baseline_source,
            "type": "baseline",
        }
        row_out.update(row)
        rows.append(row_out)

    # -----------------------------------------------------
    # 3. Add proposed_state_machine from each sweep run
    # -----------------------------------------------------
    for run_name, df in loaded.items():
        row = extract_row(df, "proposed_state_machine")
        if row is None:
            print(f"[Skip] proposed_state_machine not found in {run_name}")
            continue

        row_out = {
            "setting": run_name,
            "source_run": run_name,
            "type": "proposed_variant",
        }
        row_out.update(row)
        rows.append(row_out)

    summary = pd.DataFrame(rows)

    # -----------------------------------------------------
    # 4. Reorder columns
    # -----------------------------------------------------
    preferred_cols = [
        "setting",
        "type",
        "source_run",
        "method",
        "frames",
        "scale_mean",
        "scale_std",
        "scale_delta_mean",
        "scale_delta_p95",
        "valid_pixels_mean",
        "AbsRel",
        "SqRel",
        "RMSE",
        "RMSElog",
        "delta1",
        "delta2",
        "delta3",
    ]

    existing_cols = [c for c in preferred_cols if c in summary.columns]
    other_cols = [c for c in summary.columns if c not in existing_cols]
    summary = summary[existing_cols + other_cols]

    # -----------------------------------------------------
    # 5. Save and print
    # -----------------------------------------------------
    out_dir = os.path.dirname(args.out_csv)
    ensure_dir(out_dir)

    summary.to_csv(args.out_csv, index=False, encoding="utf-8-sig")

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 300)

    print("\n=== KITTI State Sweep Summary ===")
    print(summary.to_string(index=False))

    print("\nSaved:")
    print(args.out_csv)


if __name__ == "__main__":
    main()
