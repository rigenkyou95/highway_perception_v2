from pathlib import Path
import pandas as pd


ROOT = Path("outputs/fusion/final_experiments")

REQUIRED_FILES = [
    "task1_cross_scene/cross_scene_summary.csv",
    "task2a_anchor_range/anchor_range_summary.csv",
    "task2b_alpha/alpha_summary.csv",
    "task2c_jump_threshold/jump_threshold_summary.csv",
    "experiment_manifest.csv",
]


def main():
    print("=== Final Experiment File Check ===")

    ok = True
    for rel in REQUIRED_FILES:
        path = ROOT / rel
        if path.exists():
            print(f"[OK] {path}")
        else:
            print(f"[MISSING] {path}")
            ok = False

    if not ok:
        print("\nSome required files are missing.")
        return

    print("\n=== Summary Table Shapes ===")
    for rel in REQUIRED_FILES:
        if not rel.endswith(".csv"):
            continue

        path = ROOT / rel
        df = pd.read_csv(path)
        print(f"{rel}: rows={len(df)}, cols={len(df.columns)}")
        print("columns:", ", ".join(df.columns))
        print()

    print("[DONE] Final experiment files are ready.")


if __name__ == "__main__":
    main()
