import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def find_col(df, candidates, required=True):
    lower_map = {c.lower(): c for c in df.columns}
    for name in candidates:
        if name.lower() in lower_map:
            return lower_map[name.lower()]
    if required:
        raise KeyError(f"Missing required column. Tried: {candidates}. Available: {list(df.columns)}")
    return None


def to_numeric_series(df, col):
    return pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)


def is_valid_candidate(x):
    return np.isfinite(x) & (x > 0)


def forward_fill_from_first_valid(arr):
    arr = arr.astype(float).copy()
    valid = np.isfinite(arr)

    if not valid.any():
        return arr

    first = np.argmax(valid)
    last = arr[first]

    for i in range(first):
        arr[i] = np.nan

    for i in range(first, len(arr)):
        if np.isfinite(arr[i]):
            last = arr[i]
        else:
            arr[i] = last

    return arr


def framewise_candidate(cand):
    """
    Directly use frame-level scale candidate.
    If a candidate is missing, keep the previous valid candidate.
    """
    valid = is_valid_candidate(cand)
    out = np.where(valid, cand, np.nan)
    out = forward_fill_from_first_valid(out)
    return out, valid


def ema_only(cand, alpha=0.3):
    """
    Simple EMA over valid frame-level scale candidates.
    Missing candidates hold the previous EMA value.
    """
    valid = is_valid_candidate(cand)
    out = np.full_like(cand, np.nan, dtype=float)

    started = False
    prev = np.nan

    for i, c in enumerate(cand):
        if valid[i]:
            if not started:
                prev = c
                started = True
            else:
                prev = alpha * c + (1.0 - alpha) * prev
            out[i] = prev
        else:
            if started:
                out[i] = prev

    return out, valid


def compute_metrics(name, scale, valid_candidate=None, mode=None, update_action=None):
    scale = np.asarray(scale, dtype=float)
    finite = np.isfinite(scale)

    if finite.sum() < 2:
        raise ValueError(f"Not enough valid scale values for setting: {name}")

    s = scale[finite]
    delta = np.abs(np.diff(s))

    result = {
        "setting": name,
        "frames": len(scale),
        "valid_scale_frames": int(finite.sum()),
        "mean_s": float(np.mean(s)),
        "std_s": float(np.std(s)),
        "mean_delta_s": float(np.mean(delta)) if len(delta) else np.nan,
        "p95_delta_s": float(np.percentile(delta, 95)) if len(delta) else np.nan,
        "max_delta_s": float(np.max(delta)) if len(delta) else np.nan,
    }

    if valid_candidate is not None:
        result["candidate_valid_ratio"] = float(np.mean(valid_candidate))
        result["missing_candidate_ratio"] = float(1.0 - np.mean(valid_candidate))
    else:
        result["candidate_valid_ratio"] = np.nan
        result["missing_candidate_ratio"] = np.nan

    if mode is not None:
        mode_str = pd.Series(mode).astype(str).str.upper()
        result["tracking_ratio"] = float((mode_str == "TRACKING").mean())
        result["hold_ratio"] = float((mode_str == "HOLD").mean())
        result["init_ratio"] = float((mode_str == "INIT").mean())
    else:
        result["tracking_ratio"] = np.nan
        result["hold_ratio"] = np.nan
        result["init_ratio"] = np.nan

    if update_action is not None:
        action_str = pd.Series(update_action).astype(str).str.upper()
        reject_or_freeze = (
            action_str.str.contains("REJECT", regex=False)
            | action_str.str.contains("FREEZE", regex=False)
            | action_str.str.contains("HOLD", regex=False)
        )
        result["rejected_or_frozen"] = int(reject_or_freeze.sum())
    else:
        result["rejected_or_frozen"] = np.nan

    return result


def write_latex_table(summary, out_tex, alpha):
    rows = []
    for _, r in summary.iterrows():
        rows.append(
            f"{r['setting']} & "
            f"{r['mean_delta_s']:.4f} & "
            f"{r['p95_delta_s']:.4f} & "
            f"{r['max_delta_s']:.4f} & "
            f"{r['std_s']:.3f} \\\\"
        )

    latex = r"""\begin{table}[!t]
\centering
\caption{Component ablation of temporal scale stabilization on the straight arterial road sequence.}
\label{tab:component_ablation}
\footnotesize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{lcccc}
\toprule
Setting & Mean $\Delta s$ & P95 $\Delta s$ & Max $\Delta s$ & Std. $s_{final}$ \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    latex = latex.replace("EMA only", f"EMA only ($\\alpha={alpha}$)")
    Path(out_tex).write_text(latex, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to runtime_per_frame.csv")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    parser.add_argument("--alpha", type=float, default=0.3, help="EMA alpha for EMA-only baseline")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)

    cand_col = find_col(df, ["s_candidate", "scale_candidate", "s_cand"])
    final_col = find_col(df, ["s_final", "scale_final", "s_stable", "s_stabilized"])
    mode_col = find_col(df, ["mode", "state"], required=False)
    action_col = find_col(df, ["update_action", "action"], required=False)

    cand = to_numeric_series(df, cand_col)
    full = to_numeric_series(df, final_col)

    mode = df[mode_col].to_numpy() if mode_col else None
    action = df[action_col].to_numpy() if action_col else None

    fw_scale, fw_valid = framewise_candidate(cand)
    ema_scale, ema_valid = ema_only(cand, alpha=args.alpha)
    full_valid = is_valid_candidate(cand)

    results = []
    results.append(compute_metrics("Frame-wise candidate", fw_scale, fw_valid))
    results.append(compute_metrics(f"EMA only", ema_scale, ema_valid))
    results.append(compute_metrics("Full proposed", full, full_valid, mode=mode, update_action=action))

    summary = pd.DataFrame(results)

    out_csv = out_dir / "component_ablation_summary.csv"
    out_tex = out_dir / "component_ablation_table.tex"

    summary.to_csv(out_csv, index=False)
    write_latex_table(summary, out_tex, args.alpha)

    print("[DONE] Component ablation summary saved:")
    print(" ", out_csv)
    print(" ", out_tex)
    print()
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
