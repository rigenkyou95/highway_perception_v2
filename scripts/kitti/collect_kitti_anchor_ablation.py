# -*- coding: utf-8 -*-
import os
import argparse
import pandas as pd


RUNS = {
    "Z8_40": {
        "eval": r"outputs\kitti\eval_depth\kitti_geometry_anchor_H165_P0_Z8_40\kitti_geometry_eval_summary.csv",
        "debug": r"outputs\kitti\eval_depth\kitti_geometry_anchor_H165_P0_Z8_40\debug_analysis\geometry_scale_debug_summary.csv",
    },
    "Z10_40": {
        "eval": r"outputs\kitti\eval_depth\kitti_geometry_anchor_H165_P0_Z10_40\kitti_geometry_eval_summary.csv",
        "debug": r"outputs\kitti\eval_depth\kitti_geometry_anchor_H165_P0_Z10_40\debug_analysis\geometry_scale_debug_summary.csv",
    },
    "Z12_40": {
        "eval": r"outputs\kitti\eval_depth\kitti_geometry_anchor_H165_P0_Z12_40\kitti_geometry_eval_summary.csv",
        "debug": r"outputs\kitti\eval_depth\kitti_geometry_anchor_H165_P0_Z12_40\debug_analysis\geometry_scale_debug_summary.csv",
    },
}


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def extract_eval_row(eval_csv, method):
    df = pd.read_csv(eval_csv)
    row = df[df["method"].astype(str) == method]
    if len(row) == 0:
        return None
    return row.iloc[0].to_dict()


def extract_debug_row(debug_csv):
    if not os.path.isfile(debug_csv):
        return {}
    df = pd.read_csv(debug_csv)
    if len(df) == 0:
        return {}
    return df.iloc[0].to_dict()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-csv",
        default=r"outputs\kitti\eval_depth\kitti_anchor_ablation_summary.csv"
    )
    args = parser.parse_args()

    rows = []

    for name, paths in RUNS.items():
        eval_csv = paths["eval"]
        debug_csv = paths["debug"]

        if not os.path.isfile(eval_csv):
            print(f"[Skip] missing eval: {eval_csv}")
            continue

        dbg = extract_debug_row(debug_csv)

        cand = extract_eval_row(eval_csv, "geometry_candidate")
        ema = extract_eval_row(eval_csv, "geometry_ema")
        state = extract_eval_row(eval_csv, "geometry_state_machine")

        row = {
            "setting": name,
            "geo_valid_ratio": dbg.get("geo_valid_ratio", None),
            "scale_rel_err_mean": dbg.get("scale_rel_err_mean", None),
            "scale_rel_err_median": dbg.get("scale_rel_err_median", None),
            "scale_rel_err_p95": dbg.get("scale_rel_err_p95", None),
            "corr_s_geo_oracle": dbg.get("corr_s_geo_oracle", None),
            "geo_points_mean": dbg.get("geo_points_mean", None),
            "z_med_mean": dbg.get("z_med_mean", None),
        }

        if cand is not None:
            row.update({
                "candidate_frames": cand.get("frames"),
                "candidate_AbsRel": cand.get("AbsRel"),
                "candidate_RMSE": cand.get("RMSE"),
                "candidate_RMSElog": cand.get("RMSElog"),
                "candidate_delta1": cand.get("delta1"),
            })

        if ema is not None:
            row.update({
                "ema_frames": ema.get("frames"),
                "ema_AbsRel": ema.get("AbsRel"),
                "ema_RMSE": ema.get("RMSE"),
                "ema_scale_delta_p95": ema.get("scale_delta_p95"),
            })

        if state is not None:
            row.update({
                "state_frames": state.get("frames"),
                "state_AbsRel": state.get("AbsRel"),
                "state_RMSE": state.get("RMSE"),
                "state_RMSElog": state.get("RMSElog"),
                "state_delta1": state.get("delta1"),
                "state_scale_delta_p95": state.get("scale_delta_p95"),
            })

        rows.append(row)

    summary = pd.DataFrame(rows)

    out_dir = os.path.dirname(args.out_csv)
    ensure_dir(out_dir)

    summary.to_csv(args.out_csv, index=False, encoding="utf-8-sig")

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 300)

    print("\n=== KITTI Anchor Ablation Summary ===")
    print(summary.to_string(index=False))

    print("\nSaved:")
    print(args.out_csv)


if __name__ == "__main__":
    main()
