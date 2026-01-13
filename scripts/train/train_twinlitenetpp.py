import os
import sys
import time
import glob
from typing import Tuple

import cv2
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ==== 用于自动画 loss 曲线 ====
import matplotlib
matplotlib.use("Agg")   # 服务器/无界面环境也可以画图
import matplotlib.pyplot as plt


# ============================================================
# 设置项目根目录（scripts/train/ → 上两层）
# ============================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))

if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

ROOT = ROOT_DIR

# ============================================================
# 正确导入 TwinLiteNetPP（纯模型版本）
# ============================================================
from models.seg.twinlitenet_pp.TwinLitePP import TwinLiteNetPP


# ============================================================
# 路径 & 训练配置
# ============================================================

class Config:
    # === 数据路径（保持与你第一次训练一致） ===
    img_train_dir = r"E:\detected sys\lane_detect\data\bdd100k\images\train"
    img_val_dir   = r"E:\detected sys\lane_detect\data\bdd100k\images\val"

    da_train_dir  = r"E:\detected sys\lane_detect\data\bdd100k\segments\train"
    da_val_dir    = r"E:\detected sys\lane_detect\data\bdd100k\segments\val"

    ll_train_dir  = r"E:\detected sys\lane_detect\data\bdd100k\lane\train"
    ll_val_dir    = r"E:\detected sys\lane_detect\data\bdd100k\lane\val"

    # 预训练权重（TwinLiteNet 原版 best.pth，可选）
    pretrained_path = r"E:\detected sys\highway_perception_v2\third_party\TwinLiteNet\pretrained\best.pth"

    # 输出模型保存目录（统一到 models/ckpts 下）
    # 结构：
    #   models/ckpts/twinlitenetpp_geodepth/
    #       train_log.txt, loss_curves.png,
    #       twinlitenetpp_da_best.pth, twinlitenetpp_lanegeo_best.pth
    #       checkpoints/  (保存各个 epoch 的权重)
    save_dir = os.path.join(ROOT, "models", "ckpts", "twinlitenetpp_geodepth")
    ckpt_dir = os.path.join(save_dir, "checkpoints")   # 子目录专门放 epochX 权重

    # 训练参数
    num_epochs = 120
    batch_size = 4
    num_workers = 4

    lr = 5e-4
    weight_decay = 1e-4

    img_size = (640, 360)  # 宽,高
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # === 损失超参数 ===
    # Tversky（Lane 分支用）
    alpha_tversky = 0.5
    beta_tversky = 0.5

    # ---- Lane 连续性权重（弱化，避免压制 DA）----
    # 最终 lane_branch_loss = ll_tversky + ll_focal + lambda_lane_cont * ll_cont_raw
    lambda_lane_cont = 0.05

    # Lane 分支内部 Tversky / Focal 的权重
    lambda_ll_tversky = 1.0
    lambda_ll_focal = 1.0

    # ---- Drivable 分支 Dice + BCE 比例 ----
    # da_branch_loss = 0.6 * Dice + 0.4 * BCE
    lambda_da_dice = 0.6
    lambda_da_bce = 0.4

    # ---- 两个分支在总 loss 里的权重（关键）----
    # total_loss = lambda_da_total * da_branch_loss + lambda_ll_total * ll_branch_loss
    lambda_da_total = 1.5   # 强化 DA 分支
    lambda_ll_total = 1.0   # Lane 保持 1.0

    # 混合精度
    use_amp = True


cfg = Config()


# ============================================================
# 数据集定义
# ============================================================

class BDDLaneDrivableDataset(Dataset):
    """
    同时加载图像 + drivable 标签 + lane 标签
    """

    def __init__(self,
                 img_dir: str,
                 da_dir: str,
                 ll_dir: str,
                 img_size: Tuple[int, int] = (640, 360)):
        super().__init__()
        self.img_dir = img_dir
        self.da_dir = da_dir
        self.ll_dir = ll_dir
        self.img_size = img_size

        exts = ["*.jpg", "*.png", "*.jpeg"]
        files = []
        for ext in exts:
            files.extend(glob.glob(os.path.join(img_dir, ext)))
        self.img_paths = sorted(files)

        assert len(self.img_paths) > 0, f"No images found in {img_dir}"

        print(f"[Dataset] {img_dir} -> {len(self.img_paths)} images")

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        name = os.path.splitext(os.path.basename(img_path))[0]

        # 图像
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(img_path)

        img = cv2.resize(img, cfg.img_size)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img_chw = img_rgb.transpose(2, 0, 1)  # 3xH xW

        # drivable 区域标签
        da_path = self._find_label(self.da_dir, name)
        da = cv2.imread(da_path, cv2.IMREAD_GRAYSCALE)
        if da is None:
            raise FileNotFoundError(da_path)
        da = cv2.resize(da, cfg.img_size, interpolation=cv2.INTER_NEAREST)
        # 🔧 关键修复：BDD 的 drivable mask 为 0/1/2，使用 >0 将 1/2 都视为前景
        da = (da > 0).astype(np.int64)  # 0/1

        # lane 标签（0/255，保持 >128）
        ll_path = self._find_label(self.ll_dir, name)
        ll = cv2.imread(ll_path, cv2.IMREAD_GRAYSCALE)
        if ll is None:
            raise FileNotFoundError(ll_path)
        ll = cv2.resize(ll, cfg.img_size, interpolation=cv2.INTER_NEAREST)
        ll = (ll > 128).astype(np.int64)  # 0/1

        img_tensor = torch.from_numpy(img_chw).float()
        da_tensor = torch.from_numpy(da).long()
        ll_tensor = torch.from_numpy(ll).long()

        return img_tensor, da_tensor, ll_tensor, name

    def _find_label(self, label_dir: str, name: str) -> str:
        for ext in [".png", ".jpg", ".jpeg"]:
            p = os.path.join(label_dir, name + ext)
            if os.path.exists(p):
                return p
        raise FileNotFoundError(f"Label for {name} not found in {label_dir}")


# ============================================================
# 可选：数据集修复验证函数
# ============================================================

def verify_dataset_fix():
    """
    简单抽几个样本，检查：
      - Drivable Area 是否是“填充的白色区域”（而不是细线）
      - Lane 是否是线条
    运行方式：
      在命令行单独执行本文件时，取消 main() 下面的注释：
          # verify_dataset_fix(); return
    """
    from torch.utils.data import DataLoader

    dataset = BDDLaneDrivableDataset(
        cfg.img_train_dir, cfg.da_train_dir, cfg.ll_train_dir, img_size=cfg.img_size
    )

    indices = [0, min(100, len(dataset)-1), min(500, len(dataset)-1)]
    n = len(indices)

    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))

    if n == 1:
        axes = np.expand_dims(axes, 0)

    for row, idx in enumerate(indices):
        img, da, ll, name = dataset[idx]

        img_vis = img.permute(1, 2, 0).numpy()
        da_vis = da.numpy()
        ll_vis = ll.numpy()

        axes[row, 0].imshow(img_vis)
        axes[row, 0].set_title(f"{name} - 原图")
        axes[row, 0].axis("off")

        axes[row, 1].imshow(da_vis, cmap="gray", vmin=0, vmax=1)
        axes[row, 1].set_title("Drivable Area (应为填充区域)")
        axes[row, 1].axis("off")

        axes[row, 2].imshow(ll_vis, cmap="gray", vmin=0, vmax=1)
        axes[row, 2].set_title("Lane Lines")
        axes[row, 2].axis("off")

    plt.tight_layout()
    out_path = os.path.join(cfg.save_dir, "dataset_verification.png")
    os.makedirs(cfg.save_dir, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"✅ 数据集验证图已保存：{out_path}")
    print("   请目视确认：DA 为填充面，Lane 为线条。")


# ============================================================
# 损失函数：Dice / BCE / Tversky / Focal / Lane Continuity
# ============================================================

def tversky_loss(logits, targets, alpha=0.5, beta=0.5, eps=1e-6):
    """
    logits: [N, C, H, W]
    targets: [N, H, W]
    针对前景类(1)计算 Tversky
    """
    num_classes = logits.shape[1]
    probs = torch.softmax(logits, dim=1)

    targets_1h = F.one_hot(targets, num_classes=num_classes)  # [N,H,W,C]
    targets_1h = targets_1h.permute(0, 3, 1, 2).float()      # [N,C,H,W]

    p1 = probs[:, 1:2]        # [N,1,H,W]
    g1 = targets_1h[:, 1:2]   # [N,1,H,W]

    tp = torch.sum(p1 * g1)
    fp = torch.sum(p1 * (1 - g1))
    fn = torch.sum((1 - p1) * g1)

    tversky = (tp + eps) / (tp + alpha * fp + beta * fn + eps)
    return 1.0 - tversky


def focal_loss(logits, targets, gamma=2.0, alpha=0.25):
    """
    简化版 Focal Loss，针对前景类
    """
    ce_loss = F.cross_entropy(logits, targets, reduction="none")  # [N,H,W]
    pt = torch.exp(-ce_loss)
    focal = alpha * (1 - pt) ** gamma * ce_loss
    return focal.mean()


def dice_loss_binary_from_logits(logits, targets, eps=1e-6):
    """
    针对二类分割的 Dice：
    - logits: [N,2,H,W]（TwinLiteNetPP 的输出）
    - targets: [N,H,W]，0/1
    做法：
    - 取 softmax 后前景概率 p_fg
    - 和 targets(0/1) 做 Dice
    """
    probs = torch.softmax(logits, dim=1)[:, 1]  # [N,H,W]
    targets_f = targets.float()

    probs = probs.view(probs.size(0), -1)
    targets_f = targets_f.view(targets_f.size(0), -1)

    intersection = (probs * targets_f).sum(dim=1)
    union = probs.sum(dim=1) + targets_f.sum(dim=1)

    dice = (2 * intersection + eps) / (union + eps)
    loss = 1.0 - dice.mean()
    return loss


def bce_loss_from_two_class_logits(logits, targets):
    """
    logits: [N,2,H,W]
    targets: [N,H,W], 0/1

    思路：
    - 对二分类来说：
        logit_fg = logit_1 - logit_0
      则 sigmoid(logit_fg) = P(class=1 | x)
    - 然后对 logit_fg 用 BCEWithLogits，与 0/1 的 targets 对齐。
    """
    logit_fg = logits[:, 1] - logits[:, 0]     # [N,H,W]
    targets_f = targets.float()
    return F.binary_cross_entropy_with_logits(logit_fg, targets_f)


def lane_continuity_loss_second_order(logits):
    """
    更“几何流”导向的 lane 连续性约束：二阶差分（只在垂直方向）
    参考：
        diff1 = p[:,1:,:] - p[:,:-1,:]
        diff2 = diff1[:,1:,:] - diff1[:,:-1,:]
        loss = mean(|diff2|)
    即惩罚概率沿车道方向的“弯折”和“剧烈变化”，让车道更直、更平滑。
    """
    probs = torch.softmax(logits, dim=1)[:, 1]  # [N,H,W]

    # 一阶差分（垂直方向）
    diff1 = probs[:, 1:, :] - probs[:, :-1, :]       # [N,H-1,W]
    # 二阶差分
    diff2 = diff1[:, 1:, :] - diff1[:, :-1, :]       # [N,H-2,W]

    loss = diff2.abs().mean()
    return loss


# ============================================================
# 训练 & 验证逻辑
# ============================================================

def train_one_epoch(model, loader, optimizer, scaler, epoch):
    """
    返回：该 epoch 的平均
      - total_loss
      - da_loss（drivable 分支总 loss，未乘 lambda_da_total）
      - ll_loss（lane 分支总 loss，未乘 lambda_ll_total）
    """
    model.train()
    device = cfg.device

    pbar = tqdm(loader, desc=f"Epoch {epoch} [train]", ncols=120)

    sum_loss = 0.0
    sum_da = 0.0
    sum_ll = 0.0
    num = 0

    avg_loss_for_bar = 0.0

    for step, (images, da_targets, ll_targets, names) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        da_targets = da_targets.to(device, non_blocking=True)
        ll_targets = ll_targets.to(device, non_blocking=True)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=cfg.use_amp):
            da_logits, ll_logits = model(images)

            # ===== Drivable 分支：Dice + BCE（分支内部 loss） =====
            da_dice = dice_loss_binary_from_logits(da_logits, da_targets)
            da_bce = bce_loss_from_two_class_logits(da_logits, da_targets)
            loss_da_branch = (
                cfg.lambda_da_dice * da_dice +
                cfg.lambda_da_bce * da_bce
            )

            # ===== Lane 分支：Tversky + Focal + Continuity（二阶差分） =====
            ll_tversky = tversky_loss(
                ll_logits, ll_targets,
                alpha=cfg.alpha_tversky,
                beta=cfg.beta_tversky
            )
            ll_focal = focal_loss(ll_logits, ll_targets)
            ll_cont_raw = lane_continuity_loss_second_order(ll_logits)

            loss_ll_branch = (
                cfg.lambda_ll_tversky * ll_tversky +
                cfg.lambda_ll_focal * ll_focal +
                cfg.lambda_lane_cont * ll_cont_raw
            )

            # ===== 总 loss：DA 分支整体权重 1.5，Lane 分支 1.0 =====
            loss = (
                cfg.lambda_da_total * loss_da_branch +
                cfg.lambda_ll_total * loss_ll_branch
            )

        if cfg.use_amp and scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        sum_loss += loss.item()
        sum_da += loss_da_branch.item()
        sum_ll += loss_ll_branch.item()
        num += 1

        avg_loss_for_bar = (
            avg_loss_for_bar * 0.9 + loss.item() * 0.1
            if step > 0 else loss.item()
        )
        pbar.set_postfix(
            loss=f"{avg_loss_for_bar:.4f}",
            da=f"{loss_da_branch.item():.4f}",
            ll=f"{loss_ll_branch.item():.4f}"
        )

    mean_loss = sum_loss / max(num, 1)
    mean_da = sum_da / max(num, 1)
    mean_ll = sum_ll / max(num, 1)
    return mean_loss, mean_da, mean_ll


def validate(model, loader, epoch):
    """
    验证阶段同样分别统计：
      - total_val_loss（包含 da_total / ll_total 权重）
      - val_da_loss（分支内部 loss）
      - val_ll_loss（分支内部 loss）
    """
    model.eval()
    device = cfg.device

    total_loss = 0.0
    total_da = 0.0
    total_ll = 0.0
    count = 0

    with torch.no_grad():
        pbar = tqdm(loader, desc=f"Epoch {epoch} [val]", ncols=120)
        for step, (images, da_targets, ll_targets, names) in enumerate(pbar):
            images = images.to(device, non_blocking=True)
            da_targets = da_targets.to(device, non_blocking=True)
            ll_targets = ll_targets.to(device, non_blocking=True)

            da_logits, ll_logits = model(images)

            # Drivable：Dice + BCE（分支内部 loss）
            da_dice = dice_loss_binary_from_logits(da_logits, da_targets)
            da_bce = bce_loss_from_two_class_logits(da_logits, da_targets)
            loss_da_branch = (
                cfg.lambda_da_dice * da_dice +
                cfg.lambda_da_bce * da_bce
            )

            # Lane：Tversky + Focal + Continuity（二阶差分）
            ll_tversky = tversky_loss(
                ll_logits, ll_targets,
                alpha=cfg.alpha_tversky,
                beta=cfg.beta_tversky
            )
            ll_focal = focal_loss(ll_logits, ll_targets)
            ll_cont_raw = lane_continuity_loss_second_order(ll_logits)

            loss_ll_branch = (
                cfg.lambda_ll_tversky * ll_tversky +
                cfg.lambda_ll_focal * ll_focal +
                cfg.lambda_lane_cont * ll_cont_raw
            )

            loss = (
                cfg.lambda_da_total * loss_da_branch +
                cfg.lambda_ll_total * loss_ll_branch
            )

            total_loss += loss.item()
            total_da += loss_da_branch.item()
            total_ll += loss_ll_branch.item()
            count += 1

            avg_total = total_loss / count
            avg_da = total_da / count
            avg_ll = total_ll / count

            pbar.set_postfix(
                val_loss=f"{avg_total:.4f}",
                val_da=f"{avg_da:.4f}",
                val_ll=f"{avg_ll:.4f}"
            )

    mean_total = total_loss / max(count, 1)
    mean_da = total_da / max(count, 1)
    mean_ll = total_ll / max(count, 1)
    return mean_total, mean_da, mean_ll


# ============================================================
# 训练日志可视化：自动画 loss 曲线
# ============================================================

def plot_training_curves(log_path: str, out_dir: str):
    """
    读取 train_log.txt，并输出 loss_curves.png：
      - subplot(1): total train / val
      - subplot(2): drivable train / val
      - subplot(3): lane train / val
    """
    if not os.path.isfile(log_path):
        print(f"[Plot] Log file not found: {log_path}, 跳过画图。")
        return

    try:
        data = np.loadtxt(log_path, delimiter=",")
    except Exception as e:
        print("[Plot] 读取日志失败:", e)
        return

    if data.ndim == 1:
        data = data[None, :]

    # 日志格式：epoch, lr,
    #           train_loss, train_da_loss, train_ll_loss,
    #           val_loss, val_da_loss, val_ll_loss
    epoch = data[:, 0]
    train_loss = data[:, 2]
    train_da = data[:, 3]
    train_ll = data[:, 4]
    val_loss = data[:, 5]
    val_da = data[:, 6]
    val_ll = data[:, 7]

    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

    # 总 loss
    axes[0].plot(epoch, train_loss, label="train_total")
    axes[0].plot(epoch, val_loss, label="val_total")
    axes[0].set_ylabel("Total Loss")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.3)

    # Drivable loss
    axes[1].plot(epoch, train_da, label="train_da")
    axes[1].plot(epoch, val_da, label="val_da")
    axes[1].set_ylabel("DA Loss")
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.3)

    # Lane loss
    axes[2].plot(epoch, train_ll, label="train_lane")
    axes[2].plot(epoch, val_ll, label="val_lane")
    axes[2].set_ylabel("Lane Loss")
    axes[2].set_xlabel("Epoch")
    axes[2].legend()
    axes[2].grid(True, linestyle="--", alpha=0.3)

    fig.tight_layout()
    out_path = os.path.join(out_dir, "loss_curves.png")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    print("[Plot] Saved training curves to:", out_path)


# ============================================================
# 主入口
# ============================================================

def main():
    print(f"[INFO] 项目根目录 ROOT_DIR: {ROOT_DIR}")
    print("[INFO] sys.path 前 5 项:")
    for p in sys.path[:5]:
        print("   ", p)
    print("============================================================")
    print(f"[INFO] TwinLiteNetPP 导入成功：{TwinLiteNetPP}")

    os.makedirs(cfg.save_dir, exist_ok=True)
    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    print("Save dir :", cfg.save_dir)
    print("Ckpt dir :", cfg.ckpt_dir)
    print("Device :", cfg.device)

    # 如需先检查标签是否正确，可先运行：
    # verify_dataset_fix()
    # return

    # 1. 构建数据集 & DataLoader
    train_dataset = BDDLaneDrivableDataset(
        cfg.img_train_dir, cfg.da_train_dir, cfg.ll_train_dir, img_size=cfg.img_size
    )
    val_dataset = BDDLaneDrivableDataset(
        cfg.img_val_dir, cfg.da_val_dir, cfg.ll_val_dir, img_size=cfg.img_size
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False
    )

    # 2. 构建模型
    device = cfg.device
    model = TwinLiteNetPP(use_refine=True)

    # 加载预训练权重（可选）
    if os.path.isfile(cfg.pretrained_path):
        print("Loading pretrained weights from:", cfg.pretrained_path)
        state = torch.load(cfg.pretrained_path, map_location="cpu")
        missing, unexpected = model.load_state_dict(state, strict=False)
        print("missing keys:", missing)
        print("unexpected keys:", unexpected)

    if device == "cuda":
        model = torch.nn.DataParallel(model).to(device)
    else:
        model = model.to(device)

    # 3. 优化器 & 混合精度
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay
    )
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.use_amp)

    # 只保留两种“best”：Drivable & Lane 几何流
    best_da_val = float("inf")     # Drivable 专用（看 val_da_loss）
    best_ll_val = float("inf")     # Lane 几何流专用（看 val_ll_loss）

    log_path = os.path.join(cfg.save_dir, "train_log.txt")

    # 4. 训练循环
    for epoch in range(1, cfg.num_epochs + 1):
        print(f"\n========== Epoch {epoch}/{cfg.num_epochs} ==========")

        train_loss, train_da_loss, train_ll_loss = train_one_epoch(
            model, train_loader, optimizer, scaler, epoch
        )
        val_loss, val_da_loss, val_ll_loss = validate(model, val_loader, epoch)

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"[Epoch {epoch}/{cfg.num_epochs}] "
            f"LR={current_lr:.6e} | "
            f"train_loss={train_loss:.4f} "
            f"(da={train_da_loss:.4f}, ll={train_ll_loss:.4f}) | "
            f"val_loss={val_loss:.4f} "
            f"(da={val_da_loss:.4f}, ll={val_ll_loss:.4f})"
        )

        # 记录日志（供画图使用）
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(
                f"{epoch},{current_lr:.6e},"
                f"{train_loss:.6f},{train_da_loss:.6f},{train_ll_loss:.6f},"
                f"{val_loss:.6f},{val_da_loss:.6f},{val_ll_loss:.6f}\n"
            )

        # Drivable 专用 best（看 val_da_loss）
        if val_da_loss < best_da_val:
            best_da_val = val_da_loss
            save_path = os.path.join(cfg.save_dir, "twinlitenetpp_da_best.pth")
            torch.save(model.state_dict(), save_path)
            print(f"[BEST-DA] val_da_loss improved to {best_da_val:.4f}. Saved to {save_path}")

        # Lane 几何流专用 best（看 val_ll_loss）
        if val_ll_loss < best_ll_val:
            best_ll_val = val_ll_loss
            save_path = os.path.join(cfg.save_dir, "twinlitenetpp_lanegeo_best.pth")
            torch.save(model.state_dict(), save_path)
            print(f"[BEST-LANE] val_ll_loss improved to {best_ll_val:.4f}. Saved to {save_path}")

        # 每隔若干 epoch 保存一个普通 ckpt（方便之后对比）
        if epoch % 10 == 0:
            ckpt_path = os.path.join(cfg.ckpt_dir, f"twinlitenetpp_epoch{epoch}.pth")
            torch.save(model.state_dict(), ckpt_path)
            print(f"[CKPT] Saved checkpoint to {ckpt_path}")

    # 5. 训练结束后画 loss 曲线
    plot_training_curves(log_path, cfg.save_dir)


if __name__ == "__main__":
    main()
