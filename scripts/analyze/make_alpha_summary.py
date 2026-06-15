import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ALPHA_DIRS = {
    "0.1": "alpha_01",
    "0.3": "alpha_03",
    "0.5": "alpha_05",
    "0.7": "alpha_07",
}


def find_column(df, candidates):
    for col in candidates:
        if col in df.columns:
            return col
    return None


def to_numeric_safe(series):
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


def summarize_one(csv_path: Path, alpha_name: str):
    df = pd.read_csv(csv_path)

    mode_col = find_column(df, ["mode", "state", "state_mode"])
    action_col = find_column(df, ["update_action", "action"])
    s_candidate_col = find_column(df, ["s_candidate", "s_cand", "scale_candidate"])
    s_final_col = find_column(df, ["s_final", "scale_final", "stabilized_scale"])
    anchor_col = find_column(df, ["geo_points_num", "num_anchors", "anchor_count", "ratio_points_filt"])

    if s_final_col is None:
        raise ValueError(f"No s_final column found in: {csv_path}")

    s_final = to_numeric_safe(df[s_final_col]).dropna()
    delta_s = s_final.diff().abs().dropna()

    if s_candidate_col is not None:
        s_candidate = to_numeric_safe(df[s_candidate_col])
        candidate_valid_ratio = s_candidate.notna().mean()
    else:
        candidate_valid_ratio = np.nan

    if mode_col is not None:
        mode = df[mode_col].astype(str).str.upper()
        tracking_ratio = (mode == "TRACKING").mean()
        hold_ratio = (mode == "HOLD").mean()
        init_ratio = (mode == "INIT").mean()
    else:
        tracking_ratio = np.nan
        hold_ratio = np.nan
        init_ratio = np.nan

    if anchor_col is not None:
        anchors = to_numeric_safe(df[anchor_col])
        mean_anchors = anchors.mean()
    else:
        mean_anchors = np.nan

    if action_col is not None:
        action = df[action_col].astype(str).str.upper()
        jump_reject_count = action.str.contains("JUMP_REJECT", regex=False).sum()
        freeze_count = action.str.contains("FREEZE", regex=False).sum()
        rejected_or_frozen = action.str.contains("REJECT|FREEZE", regex=True).sum()
    else:
        jump_reject_count = np.nan
        freeze_count = np.nan
        rejected_or_frozen = np.nan

    return {
        "alpha": alpha_name,
        "frames": len(df),
        "candidate_valid_ratio": candidate_valid_ratio,
        "tracking_ratio": tracking_ratio,
        "hold_ratio": hold_ratio,
        "mean_anchors": mean_anchors,
        "mean_s_final": s_final.mean(),
        "std_s_final": s_final.std(),
        "mean_delta_s": delta_s.mean() if len(delta_s) > 0 else np.nan,
        "p95_delta_s": delta_s.quantile(0.95) if len(delta_s) > 0 else np.nan,
        "max_delta_s": delta_s.max() if len(delta_s) > 0 else np.nan,
        "jump_reject_count": jump_reject_count,
        "freeze_count": freeze_count,
        "rejected_or_frozen": rejected_or_frozen,
        "init_ratio": init_ratio,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    root_dir = Path(args.root_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []

    for alpha_name, folder in ALPHA_DIRS.items():
        csv_path = root_dir / folder / "runtime_per_frame.csv"

        if not csv_path.exists():
            raise FileNotFoundError(f"Missing CSV for alpha={alpha_name}: {csv_path}")

        print(f"[INFO] Processing alpha={alpha_name}: {csv_path}")
        rows.append(summarize_one(csv_path, alpha_name))

    summary = pd.DataFrame(rows)

    columns = [
        "alpha",
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
    summary.to_csv(out_path, index=False, encoding="utf-8-sig")

    print("\n=== Alpha Sensitivity Summary ===")
    print(summary.round(4).to_string(index=False))
    print(f"\n[OK] Saved summary to: {out_path}")


if __name__ == "__main__":
    main()
