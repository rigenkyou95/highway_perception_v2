# -*- coding: utf-8 -*-
import os
import pandas as pd


RUNS = {
    "anchor6":  r"outputs\fusion\experiments\exp_anchor_6_40\scale_debug.csv",
    "anchor8":  r"outputs\fusion\experiments\exp_near_anchor_8m\scale_debug.csv",
    "anchor10": r"outputs\fusion\exp_full\scale_debug.csv",
    "anchor12": r"outputs\fusion\experiments\exp_strict_anchor_12m\scale_debug.csv",
    "anchor15": r"outputs\fusion\experiments\exp_anchor_15_40\scale_debug.csv",
}

OUT_DIR = r"outputs\fusion\experiments\analysis_anchor_range_6_15"
os.makedirs(OUT_DIR, exist_ok=True)


def to_num(df, col):
    if col not in df.columns:
        return pd.Series([float("nan")] * len(df))
    return pd.to_numeric(df[col], errors="coerce")


rows = []

for name, path in RUNS.items():
    if not os.path.isfile(path):
        print(f"[Missing] {name}: {path}")
        continue

    df = pd.read_csv(path)

    s_candidate = to_num(df, "s_candidate")
    s_final = to_num(df, "s_final")
    confidence = to_num(df, "confidence")
    geo_points = to_num(df, "geo_points_num")
    z_med = to_num(df, "z_med")

    mode = df["mode"].astype(str) if "mode" in df.columns else pd.Series([""] * len(df))
    update_action = df["update_action"].astype(str) if "update_action" in df.columns else pd.Series([""] * len(df))

    ds_final = s_final.diff().abs()

    rows.append({
        "run": name,
        "frames": len(df),
        "candidate_valid_ratio": float(s_candidate.notna().mean()),
        "tracking_ratio": float((mode == "TRACKING").mean()),
        "hold_ratio": float((mode == "HOLD").mean()),
        "init_ratio": float((mode == "INIT").mean()),
        "freeze_ratio": float((update_action == "FREEZE").mean()),
        "jump_reject_ratio": float((update_action == "JUMP_REJECT").mean()),
        "force_slow_ratio": float((update_action == "FORCE_SLOW").mean()),
        "s_final_mean": float(s_final.mean(skipna=True)),
        "s_final_std": float(s_final.std(skipna=True)),
        "s_final_delta_mean": float(ds_final.mean(skipna=True)),
        "s_final_delta_p95": float(ds_final.quantile(0.95)),
        "s_final_delta_max": float(ds_final.max(skipna=True)),
        "geo_points_mean": float(geo_points.mean(skipna=True)),
        "geo_points_min": float(geo_points.min(skipna=True)),
        "geo_points_max": float(geo_points.max(skipna=True)),
        "z_med_mean": float(z_med.mean(skipna=True)),
        "confidence_mean": float(confidence.mean(skipna=True)),
    })

summary = pd.DataFrame(rows)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 300)

print(summary.to_string(index=False))

out_csv = os.path.join(OUT_DIR, "anchor_range_summary_full.csv")
summary.to_csv(out_csv, index=False, encoding="utf-8-sig")

print("\nSaved:")
print(out_csv)
