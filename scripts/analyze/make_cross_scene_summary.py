import argparse
from pathlib import Path

import numpy as np
import pandas as pd


SCENE_FILES = {
    "Straight arterial road": "straight_arterial_road_runtime_per_frame.csv",
    "Urban road": "urban_road_runtime_per_frame.csv",
    "Curved road": "curved_road_runtime_per_frame.csv",
    "Heavy traffic / merging": "heavy_traffic_merging_runtime_per_frame.csv",
    "Downhill road": "downhill_runtime_per_frame.csv",
    "Bridge / uphill road": "bridge_uphill_runtime_per_frame.csv",
    "Rainy-night road": "rainy_night_road_runtime_per_frame.csv",
}


def find_column(df, candidates):
    """Return the first existing column name from candidates."""
    for col in candidates:
        if col in df.columns:
            return col
    return None


def to_numeric_safe(series):
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


def summarize_one(csv_path: Path, scene_name: str):
    df = pd.read_csv(csv_path)

    frames = len(df)

    mode_col = find_column(df, ["mode", "state", "state_mode"])
    action_col = find_column(df, ["update_action", "action"])
    s_candidate_col = find_column(df, ["s_candidate", "s_cand", "scale_candidate"])
    s_final_col = find_column(df, ["s_final", "scale_final", "stabilized_scale"])
    anchor_col = find_column(df, ["geo_points_num", "num_anchors", "anchor_count", "ratio_points_filt"])

    if s_final_col is None:
        raise ValueError(f"[ERROR] No s_final column found in: {csv_path}")

    s_final = to_numeric_safe(df[s_final_col])

    # Candidate Valid Ratio
    if s_candidate_col is not None:
        s_candidate = to_numeric_safe(df[s_candidate_col])
        candidate_valid_ratio = s_candidate.notna().mean()
    else:
        candidate_valid_ratio = np.nan

    # Tracking / Hold Ratio
    if mode_col is not None:
        mode = df[mode_col].astype(str).str.upper()
        tracking_ratio = (mode == "TRACKING").mean()
        hold_ratio = (mode == "HOLD").mean()
        init_ratio = (mode == "INIT").mean()
    else:
        tracking_ratio = np.nan
        hold_ratio = np.nan
        init_ratio = np.nan

    # Mean anchors
    if anchor_col is not None:
        anchors = to_numeric_safe(df[anchor_col])
        mean_anchors = anchors.mean()
    else:
        mean_anchors = np.nan

    # Frame-to-frame scale change
    delta_s = s_final.diff().abs().dropna()

    mean_delta_s = delta_s.mean() if len(delta_s) > 0 else np.nan
    p95_delta_s = delta_s.quantile(0.95) if len(delta_s) > 0 else np.nan
    max_delta_s = delta_s.max() if len(delta_s) > 0 else np.nan

    # Rejected / frozen updates
    jump_reject_count = np.nan
    freeze_count = np.nan
    rejected_or_frozen = np.nan

    if action_col is not None:
        action = df[action_col].astype(str).str.upper()
        jump_reject_count = action.str.contains("JUMP_REJECT", regex=False).sum()
        freeze_count = action.str.contains("FREEZE", regex=False).sum()
        rejected_or_frozen = action.str.contains("REJECT|FREEZE", regex=True).sum()

    return {
        "scene": scene_name,
        "frames": frames,
        "candidate_valid_ratio": candidate_valid_ratio,
        "tracking_ratio": tracking_ratio,
        "hold_ratio": hold_ratio,
        "init_ratio": init_ratio,
        "mean_anchors": mean_anchors,
        "mean_s_final": s_final.mean(),
        "std_s_final": s_final.std(),
        "mean_delta_s": mean_delta_s,
        "p95_delta_s": p95_delta_s,
        "max_delta_s": max_delta_s,
        "jump_reject_count": jump_reject_count,
        "freeze_count": freeze_count,
        "rejected_or_frozen": rejected_or_frozen,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv-dir",
        required=True,
        help="Directory containing the seven per-video runtime CSV files.",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output path for cross_scene_summary.csv.",
    )
    args = parser.parse_args()

    csv_dir = Path(args.csv_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []

    for scene_name, filename in SCENE_FILES.items():
        csv_path = csv_dir / filename

        if not csv_path.exists():
            raise FileNotFoundError(f"[ERROR] Missing CSV for {scene_name}: {csv_path}")

        print(f"[INFO] Processing {scene_name}: {csv_path}")
        rows.append(summarize_one(csv_path, scene_name))

    summary = pd.DataFrame(rows)

    # Keep a stable column order
    columns = [
        "scene",
        "frames",
        "candidate_valid_ratio",
        "tracking_ratio",
        "hold_ratio",
        "mean_anchors",
        "mean_s_final",
        "mean_delta_s",
        "p95_delta_s",
        "std_s_final",
        "max_delta_s",
        "jump_reject_count",
        "freeze_count",
        "rejected_or_frozen",
        "init_ratio",
    ]

    summary = summary[columns]

    # Save full precision CSV
    summary.to_csv(out_path, index=False, encoding="utf-8-sig")

    # Print rounded version for quick checking
    print("\n=== Cross-Scene Summary ===")
    print(summary.round(4).to_string(index=False))

    print(f"\n[OK] Saved summary to: {out_path}")


if __name__ == "__main__":
    main()
