import os
import sys
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# ---------------------------------------------------------
# 1. 找到项目根目录和 log 文件
# ---------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
SAVE_DIR = os.path.join(ROOT_DIR, "checkpoints", "twinlitenetpp_geodepth")

LOG_PATH_RAW = os.path.join(SAVE_DIR, "train_log.txt")
LOG_PATH_CLEAN = os.path.join(SAVE_DIR, "train_log_clean.csv")

print("[INFO] ROOT_DIR:", ROOT_DIR)
print("[INFO] SAVE_DIR:", SAVE_DIR)
print("[INFO] RAW LOG :", LOG_PATH_RAW)


# ---------------------------------------------------------
# 2. 只保留“8 列”的新日志行，写到一个干净文件里
#    epoch, lr, train_loss, train_da, train_ll, val_loss, val_da, val_ll
# ---------------------------------------------------------
def clean_log(raw_path: str, out_path: str):
    if not os.path.isfile(raw_path):
        raise FileNotFoundError(raw_path)

    kept = 0
    total = 0
    with open(raw_path, "r", encoding="utf-8") as f_in, \
         open(out_path, "w", encoding="utf-8", newline="") as f_out:
        writer = csv.writer(f_out)
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            total += 1
            # 只保留 8 列的行（就是这次训练脚本写的部分）
            if len(parts) == 8:
                writer.writerow(parts)
                kept += 1

    print(f"[CLEAN] total lines = {total}, kept 8-col lines = {kept}")
    if kept == 0:
        raise RuntimeError("没有找到 8 列格式的行，请确认 train_log.txt 是否是这次训练生成的。")


# ---------------------------------------------------------
# 3. 读干净后的 CSV，用 pandas 画曲线
# ---------------------------------------------------------
def plot_curves(clean_path: str, out_dir: str):
    # 没有表头，所以 header=None 自己指定列名
    cols = [
        "epoch",
        "lr",
        "train_loss",
        "train_da_loss",
        "train_ll_loss",
        "val_loss",
        "val_da_loss",
        "val_ll_loss",
    ]
    df = pd.read_csv(clean_path, header=None, names=cols)

    # 转成 float / int
    df["epoch"] = df["epoch"].astype(int)

    # -------- 图 1：总 loss（train vs val）--------
    plt.figure(figsize=(6, 4))
    plt.plot(df["epoch"], df["train_loss"], label="train_loss")
    plt.plot(df["epoch"], df["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Total Loss (Train vs Val)")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    out1 = os.path.join(out_dir, "loss_curves_total.png")
    plt.tight_layout()
    plt.savefig(out1, dpi=200)
    plt.close()
    print("[PLOT] Saved:", out1)

    # -------- 图 2：Drivable / Lane 分支的 loss --------
    plt.figure(figsize=(6, 4))
    plt.plot(df["epoch"], df["train_da_loss"], label="train_da_loss")
    plt.plot(df["epoch"], df["val_da_loss"], label="val_da_loss")
    plt.plot(df["epoch"], df["train_ll_loss"], label="train_ll_loss")
    plt.plot(df["epoch"], df["val_ll_loss"], label="val_ll_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Branch Losses (DA & Lane)")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    out2 = os.path.join(out_dir, "loss_curves_da_lane.png")
    plt.tight_layout()
    plt.savefig(out2, dpi=200)
    plt.close()
    print("[PLOT] Saved:", out2)


def main():
    print("====================================")
    print("[STEP 1] Clean raw train_log.txt ...")
    clean_log(LOG_PATH_RAW, LOG_PATH_CLEAN)

    print("[STEP 2] Plot curves from clean log ...")
    plot_curves(LOG_PATH_CLEAN, SAVE_DIR)

    print("Done.")


if __name__ == "__main__":
    main()
