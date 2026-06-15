# -*- coding: utf-8 -*-
"""
plot_temporal_ablation.py

Compare temporal scale stabilization baselines:

1. raw_candidate
2. moving_average
3. ema_only
4. median_filter
5. no_jump_gate
6. full_state_machine

Inputs:
- full CSV: contains s_candidate and full-state-machine s_final
- no-jump CSV: contains s_final from no-jump-gate experiment

Outputs:
- temporal_ablation_metrics.csv
- temporal_ablation_curves.png
- temporal_ablation_delta.png
"""

import argparse
import os
from typing import Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def to_num(series: pd.Series) -> np.ndarray:
    return pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float64)


def ema_hold(values: np.ndarray, alpha: float = 0.08) -> np.ndarray:
    """
    Exponential moving average baseline.
    If current candidate is NaN, keep previous output.
    """
    out = np.full_like(values, np.nan, dtype=np.float64)
    prev = np.nan

    for i, v in enumerate(values):
        if np.isfinite(v):
            if not np.isfinite(prev):
                prev = v
            else:
                prev = alpha * v + (1.0 - alpha) * prev

        out[i] = prev

    return out


def moving_average_hold(values: np.ndarray, window: int = 15) -> np.ndarray:
    """
    Causal moving average baseline.
    Uses current and previous valid candidates in a fixed window.
    If no valid value exists in the window, keep previous output.
    """
    out = np.full_like(values, np.nan, dtype=np.float64)
    prev = np.nan

    for i in range(len(values)):
        start = max(0, i - window + 1)
        window_vals = values[start:i + 1]
        valid = window_vals[np.isfinite(window_vals)]

        if len(valid) > 0:
            prev = float(np.mean(valid))

        out[i] = prev

    return out


def median_filter_hold(values: np.ndarray, window: int = 15) -> np.ndarray:
    """
    Causal median filter baseline.
    Uses current and previous valid candidates in a fixed window.
    If no valid value exists in the window, keep previous output.
    """
    out = np.full_like(values, np.nan, dtype=np.float64)
    prev = np.nan

    for i in range(len(values)):
        start = max(0, i - window + 1)
        window_vals = values[start:i + 1]
        valid = window_vals[np.isfinite(window_vals)]

        if len(valid) > 0:
            prev = float(np.median(valid))

        out[i] = prev

    return out


def calc_metrics(name: str, values: np.ndarray) -> Dict[str, float]:
    s = pd.Series(values, dtype="float64")
    ds = s.diff().abs()

    return {
        "setting": name,
        "std": float(s.std(skipna=True)),
        "delta_mean": float(ds.mean(skipna=True)),
        "delta_p95": float(ds.quantile(0.95)),
        "delta_max": float(ds.max(skipna=True)),
        "valid_ratio": float(s.notna().mean()),
    }


def plot_curves(frame: np.ndarray,
                series_dict: Dict[str, np.ndarray],
                out_path: str,
                title: str,
                ylabel: str):
    plt.figure(figsize=(14, 5))

    for name, values in series_dict.items():
        if name == "raw_candidate":
            plt.plot(frame, values, linewidth=0.8, alpha=0.45, label=name)
        elif name == "full_state_machine":
            plt.plot(frame, values, linewidth=1.8, label=name)
        else:
            plt.plot(frame, values, linewidth=1.2, label=name)

    plt.xlabel("Frame")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-csv", required=True)
    parser.add_argument("--no-jump-csv", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--ema-alpha", type=float, default=0.08)
    parser.add_argument("--ma-window", type=int, default=15)
    parser.add_argument("--median-window", type=int, default=15)
    args = parser.parse_args()

    ensure_dir(args.out_dir)

    df_full = pd.read_csv(args.full_csv)
    df_nojump = pd.read_csv(args.no_jump_csv)

    required_full_cols = ["frame_id", "s_candidate", "s_final"]
    required_nojump_cols = ["s_final"]

    for col in required_full_cols:
        if col not in df_full.columns:
            raise ValueError(f"Missing column in full CSV: {col}")

    for col in required_nojump_cols:
        if col not in df_nojump.columns:
            raise ValueError(f"Missing column in no-jump CSV: {col}")

    frame = to_num(df_full["frame_id"])
    s_candidate = to_num(df_full["s_candidate"])
    s_full = to_num(df_full["s_final"])
    s_nojump = to_num(df_nojump["s_final"])

    # Baselines
    s_moving_average = moving_average_hold(s_candidate, window=args.ma_window)
    s_ema = ema_hold(s_candidate, alpha=args.ema_alpha)
    s_median = median_filter_hold(s_candidate, window=args.median_window)

    series_dict = {
        "raw_candidate": s_candidate,
        f"moving_average_w{args.ma_window}": s_moving_average,
        f"ema_only_a{args.ema_alpha}": s_ema,
        f"median_filter_w{args.median_window}": s_median,
        "no_jump_gate": s_nojump,
        "full_state_machine": s_full,
    }

    # Metrics
    metrics = [calc_metrics(name, values) for name, values in series_dict.items()]
    df_metrics = pd.DataFrame(metrics)

    out_csv = os.path.join(args.out_dir, "temporal_ablation_metrics.csv")
    df_metrics.to_csv(out_csv, index=False, encoding="utf-8-sig")

    print(df_metrics.to_string(index=False))

    # Scale curves
    out_curve = os.path.join(args.out_dir, "temporal_ablation_curves.png")
    plot_curves(
        frame=frame,
        series_dict=series_dict,
        out_path=out_curve,
        title="Temporal Scale Stabilization Baselines",
        ylabel="Scale",
    )

    # Delta curves
    delta_dict = {}
    for name, values in series_dict.items():
        delta_dict[name] = pd.Series(values, dtype="float64").diff().abs().to_numpy()

    out_delta = os.path.join(args.out_dir, "temporal_ablation_delta.png")
    plot_curves(
        frame=frame,
        series_dict=delta_dict,
        out_path=out_delta,
        title="Frame-to-Frame Scale Change",
        ylabel="Absolute frame-to-frame scale change",
    )

    print("Saved:")
    print(out_csv)
    print(out_curve)
    print(out_delta)


if __name__ == "__main__":
    main()
