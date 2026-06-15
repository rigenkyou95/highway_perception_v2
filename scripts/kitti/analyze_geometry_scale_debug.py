# -*- coding: utf-8 -*-
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug-csv", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    ensure_dir(args.out_dir)

    df = pd.read_csv(args.debug_csv)

    df["s_oracle"] = pd.to_numeric(df["s_oracle"], errors="coerce")
    df["s_geo"] = pd.to_numeric(df["s_geo"], errors="coerce")
    df["geo_points"] = pd.to_numeric(df["geo_points"], errors="coerce")
    df["z_med"] = pd.to_numeric(df["z_med"], errors="coerce")

    valid = df["s_oracle"].notna() & df["s_geo"].notna()

    df_valid = df[valid].copy()
    df_valid["scale_abs_err"] = (df_valid["s_geo"] - df_valid["s_oracle"]).abs()
    df_valid["scale_rel_err"] = df_valid["scale_abs_err"] / df_valid["s_oracle"].abs().clip(lower=1e-6)
    df_valid["scale_ratio"] = df_valid["s_geo"] / df_valid["s_oracle"].clip(lower=1e-6)

    summary = {
        "frames_total": len(df),
        "frames_geo_valid": int(valid.sum()),
        "geo_valid_ratio": float(valid.mean()),
        "cam_height": float(df_valid["cam_height"].iloc[0]) if "cam_height" in df_valid.columns else np.nan,
        "pitch_deg": float(df_valid["pitch_deg"].iloc[0]) if "pitch_deg" in df_valid.columns else np.nan,
        "s_oracle_mean": float(df_valid["s_oracle"].mean()),
        "s_geo_mean": float(df_valid["s_geo"].mean()),
        "s_oracle_std": float(df_valid["s_oracle"].std()),
        "s_geo_std": float(df_valid["s_geo"].std()),
        "scale_abs_err_mean": float(df_valid["scale_abs_err"].mean()),
        "scale_abs_err_median": float(df_valid["scale_abs_err"].median()),
        "scale_rel_err_mean": float(df_valid["scale_rel_err"].mean()),
        "scale_rel_err_median": float(df_valid["scale_rel_err"].median()),
        "scale_rel_err_p95": float(df_valid["scale_rel_err"].quantile(0.95)),
        "corr_s_geo_oracle": float(df_valid[["s_geo", "s_oracle"]].corr().iloc[0, 1]),
        "geo_points_mean": float(df_valid["geo_points"].mean()),
        "geo_points_min": float(df_valid["geo_points"].min()),
        "z_med_mean": float(df_valid["z_med"].mean()),
    }

    summary_df = pd.DataFrame([summary])
    out_summary = os.path.join(args.out_dir, "geometry_scale_debug_summary.csv")
    summary_df.to_csv(out_summary, index=False, encoding="utf-8-sig")

    print("=== Geometry Scale Debug Summary ===")
    print(summary_df.to_string(index=False))

    # Plot s_geo vs s_oracle
    plt.figure(figsize=(12, 5))
    plt.plot(df["s_oracle"].to_numpy(), linewidth=1.2, label="s_oracle")
    plt.plot(df["s_geo"].to_numpy(), linewidth=1.2, label="s_geo")
    plt.xlabel("Frame")
    plt.ylabel("Scale")
    plt.title("Geometry Scale Candidate vs Oracle Scale")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out_fig1 = os.path.join(args.out_dir, "s_geo_vs_oracle_curve.png")
    plt.savefig(out_fig1, dpi=300)
    plt.close()

    # Scatter
    plt.figure(figsize=(6, 6))
    plt.scatter(df_valid["s_oracle"], df_valid["s_geo"], s=12, alpha=0.7)
    lo = min(df_valid["s_oracle"].min(), df_valid["s_geo"].min())
    hi = max(df_valid["s_oracle"].max(), df_valid["s_geo"].max())
    plt.plot([lo, hi], [lo, hi], linewidth=1.2)
    plt.xlabel("Oracle scale")
    plt.ylabel("Geometry scale")
    plt.title("Geometry Scale vs Oracle Scale")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_fig2 = os.path.join(args.out_dir, "s_geo_vs_oracle_scatter.png")
    plt.savefig(out_fig2, dpi=300)
    plt.close()

    # Geometry points
    plt.figure(figsize=(12, 5))
    plt.plot(df["geo_points"].to_numpy(), linewidth=1.2)
    plt.xlabel("Frame")
    plt.ylabel("Geometry anchor points")
    plt.title("Valid Geometry Anchor Points")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_fig3 = os.path.join(args.out_dir, "geo_points_curve.png")
    plt.savefig(out_fig3, dpi=300)
    plt.close()

    print("\nSaved:")
    print(out_summary)
    print(out_fig1)
    print(out_fig2)
    print(out_fig3)


if __name__ == "__main__":
    main()
