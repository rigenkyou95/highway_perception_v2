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


# ============================================================
# 设置项目根目录（scripts/train/ → 上两层）
# ============================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))

if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# ✅ 兼容旧代码中的 ROOT 用法
ROOT = ROOT_DIR

# ============================================================
# 正确导入 TwinLiteNetPP（纯模型版本）
# 注意：这里不要 print，否则 DataLoader 多进程会重复打印
# ============================================================
from models.seg.twinlitenet_pp.TwinLitePP import TwinLiteNetPP


# ============================================================
# 路径 & 训练配置（按需修改）
# ============================================================

class Config:
    # 数据路径（请根据你实际数据位置修改）
    img_train_dir = r"E:\detected sys\lane_detect\data\bdd100k\images\train"
    img_val_dir   = r"E:\detected sys\lane_detect\data\bdd100k\images\val"

    da_train_dir  = r"E:\detected sys\lane_detect\data\bdd100k\segments\train"
    da_val_dir    = r"E:\detected sys\lane_detect\data\bdd100k\segments\val"

    ll_train_dir  = r"E:\detected sys\lane_detect\data\bdd100k\lane\train"
    ll_val_dir    = r"E:\detected sys\lane_detect\data\bdd100k\lane\val"

    # 使用第一次大训练的 epoch50 权重作为新的预训练
    pretrained_path = r"E:\detected sys\highway_perception_v2\checkpoints\twinlitenetpp_geodepth\twinlitenetpp_epoch50.pth"

    # 输出模型保存目录（仍然写回同一目录）
    save_dir = os.path.join(ROOT, "checkpoints", "twinlitenetpp_geodepth")

    # 微调 epoch 数不用太多
    num_epochs = 30
    batch_size = 4
    num_workers = 4

    # 微调用小学习率
    lr = 5e-5          # 原来是 5e-4，这里缩小 10 倍
    weight_decay = 1e-4

    img_size = (640, 360)  # 宽,高
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 损失权重
    alpha_tversky = 0.5
    beta_tversky = 0.5
    lambda_focal = 1.0
    lambda_lane_cont = 0.1  # 车道连续性loss权重

    # 混合精度
    use_amp = True


cfg = Config()


# ============================================================
# 数据集定义
# ============================================================

class BDDLaneDrivableDataset(Dataset):
    """
    同时加载图像 + drivable 标签 + lane 标签
    假定：
    - 图像为 .jpg
    - 标签为单通道 .png 或 .jpg（像素值为 {0,1}）
    - 以图像文件名为基准，标签同名
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

        # 收集所有图像文件
        exts = ["*.jpg", "*.png", "*.jpeg"]
        files = []
        for ext in exts:
            files.extend(glob.glob(os.path.join(img_dir, ext)))
        self.img_paths = sorted(files)

        assert len(self.img_paths) > 0, f"No images found in {img_dir}"

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        name = os.path.splitext(os.path.basename(img_path))[0]

        # 读取图像
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(img_path)

        img = cv2.resize(img, cfg.img_size)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img_chw = img_rgb.transpose(2, 0, 1)  # 3xH xW

        # 读取 drivable 区域标签
        da_path = self._find_label(self.da_dir, name)
        da = cv2.imread(da_path, cv2.IMREAD_GRAYSCALE)
        if da is None:
            raise FileNotFoundError(da_path)
        da = cv2.resize(da, cfg.img_size, interpolation=cv2.INTER_NEAREST)
        da = (da > 128).astype(np.int64)  # 0/1

        # 读取 lane 标签
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
        # 尝试常见扩展名
        for ext in [".png", ".jpg", ".jpeg"]:
            p = os.path.join(label_dir, name + ext)
            if os.path.exists(p):
                return p
        raise FileNotFoundError(f"Label for {name} not found in {label_dir}")


# ============================================================
# 损失函数：Tversky + Focal + Lane Continuity
# ============================================================

def tversky_loss(logits, targets, alpha=0.5, beta=0.5, eps=1e-6):
    """
    logits: [N, C, H, W]
    targets: [N, H, W] (0~C-1)
    针对前景类(1)计算 Tversky
    """
    num_classes = logits.shape[1]
    probs = torch.softmax(logits, dim=1)

    # one-hot
    targets_1h = F.one_hot(targets, num_classes=num_classes)  # [N,H,W,C]
    targets_1h = targets_1h.permute(0, 3, 1, 2).float()      # [N,C,H,W]

    # 只关心前景类（假定为1）
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
    num_classes = logits.shape[1]
    ce_loss = F.cross_entropy(logits, targets, reduction="none")  # [N,H,W]
    pt = torch.exp(-ce_loss)

    focal = alpha * (1 - pt) ** gamma * ce_loss
    return focal.mean()


def lane_continuity_loss(logits, weight=1.0):
    """
    针对 lane 分支的连续性约束：
    - 取 lane 的前景概率 p (softmax 后)
    - 对垂直方向做差分，鼓励 p[y] 与 p[y-1] 相近
    """

    probs = torch.softmax(logits, dim=1)[:, 1]  # [N,H,W] 前景概率
    # 垂直方向差分
    diff = probs[:, 1:, :] - probs[:, :-1, :]   # [N,H-1,W]
    loss = torch.mean(diff.abs())
    return weight * loss


# ============================================================
# 训练 & 验证逻辑
# ============================================================

def train_one_epoch(model, loader, optimizer, scaler, epoch):
    """
    返回：该 epoch 的平均 loss、平均 da_loss、平均 ll_loss
    用于在 main() 中打印和记录
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
            # TwinLiteNetPP(use_refine=True) → 输出 refine 后的 logits
            da_logits, ll_logits = model(images)

            # drivable loss
            da_tversky = tversky_loss(
                da_logits, da_targets,
                alpha=cfg.alpha_tversky,
                beta=cfg.beta_tversky
            )
            da_focal = focal_loss(da_logits, da_targets)

            # lane loss
            ll_tversky = tversky_loss(
                ll_logits, ll_targets,
                alpha=cfg.alpha_tversky,
                beta=cfg.beta_tversky
            )
            ll_focal = focal_loss(ll_logits, ll_targets)

            # lane continuity loss
            ll_cont = lane_continuity_loss(ll_logits, weight=cfg.lambda_lane_cont)

            loss_da = da_tversky + cfg.lambda_focal * da_focal
            loss_ll = ll_tversky + cfg.lambda_focal * ll_focal + ll_cont

            loss = loss_da + loss_ll

        if cfg.use_amp and scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        sum_loss += loss.item()
        sum_da += loss_da.item()
        sum_ll += loss_ll.item()
        num += 1

        # 平滑显示到 tqdm 上（不影响最终统计）
        avg_loss_for_bar = (
            avg_loss_for_bar * 0.9 + loss.item() * 0.1
            if step > 0 else loss.item()
        )
        pbar.set_postfix(
            loss=f"{avg_loss_for_bar:.4f}",
            da=f"{loss_da.item():.4f}",
            ll=f"{loss_ll.item():.4f}"
        )

    mean_loss = sum_loss / max(num, 1)
    mean_da = sum_da / max(num, 1)
    mean_ll = sum_ll / max(num, 1)
    return mean_loss, mean_da, mean_ll


def validate(model, loader, epoch):
    model.eval()
    device = cfg.device

    total_loss = 0.0
    count = 0

    with torch.no_grad():
        pbar = tqdm(loader, desc=f"Epoch {epoch} [val]", ncols=120)
        for step, (images, da_targets, ll_targets, names) in enumerate(pbar):
            images = images.to(device, non_blocking=True)
            da_targets = da_targets.to(device, non_blocking=True)
            ll_targets = ll_targets.to(device, non_blocking=True)

            da_logits, ll_logits = model(images)

            da_tversky = tversky_loss(
                da_logits, da_targets,
                alpha=cfg.alpha_tversky,
                beta=cfg.beta_tversky
            )
            da_focal = focal_loss(da_logits, da_targets)

            ll_tversky = tversky_loss(
                ll_logits, ll_targets,
                alpha=cfg.alpha_tversky,
                beta=cfg.beta_tversky
            )
            ll_focal = focal_loss(ll_logits, ll_targets)

            ll_cont = lane_continuity_loss(ll_logits, weight=cfg.lambda_lane_cont)

            loss_da = da_tversky + cfg.lambda_focal * da_focal
            loss_ll = ll_tversky + cfg.lambda_focal * ll_focal + ll_cont
            loss = loss_da + loss_ll

            total_loss += loss.item()
            count += 1
            avg = total_loss / count

            pbar.set_postfix(val_loss=f"{avg:.4f}")

    return total_loss / max(count, 1)


# ============================================================
# 预训练权重加载（兼容 DataParallel 保存的 ckpt）
# ============================================================

def load_pretrained(model: nn.Module, ckpt_path: str):
    """加载预训练权重，自动去掉 DataParallel 的 'module.' 前缀。"""
    print("Loading pretrained weights from:", ckpt_path)
    state = torch.load(ckpt_path, map_location="cpu")

    # 兼容 {"state_dict": ..., ...} 或 {"model": ..., ...} 这类包装
    if isinstance(state, dict):
        if "state_dict" in state and isinstance(state["state_dict"], dict):
            state = state["state_dict"]
        elif "model" in state and isinstance(state["model"], dict):
            state = state["model"]

    new_state = {}
    for k, v in state.items():
        if k.startswith("module."):
            new_k = k[7:]  # 去掉 "module."
        else:
            new_k = k
        new_state[new_k] = v

    missing, unexpected = model.load_state_dict(new_state, strict=False)
    print("missing keys:", missing)
    print("unexpected keys:", unexpected)


# ============================================================
# 主入口
# ============================================================

def main():
    # ✅ 这些打印只在主进程打印一次，不会在 DataLoader worker 里重复
    print(f"[INFO] 项目根目录 ROOT_DIR: {ROOT_DIR}")
    print("[INFO] sys.path 前 5 项:")
    for p in sys.path[:5]:
        print("   ", p)
    print("============================================================")
    print(f"[INFO] TwinLiteNetPP 导入成功：{TwinLiteNetPP}")

    os.makedirs(cfg.save_dir, exist_ok=True)
    print("Save dir:", cfg.save_dir)
    print("Device :", cfg.device)

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

    # ✅ 正确加载预训练权重（兼容 DataParallel 保存的 ckpt）
    if os.path.isfile(cfg.pretrained_path):
        load_pretrained(model, cfg.pretrained_path)
    else:
        print(f"[WARN] pretrained_path not found: {cfg.pretrained_path}")

    # 再包 DataParallel
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

    best_val = float("inf")

    # 4. 训练循环
    log_path = os.path.join(cfg.save_dir, "train_log_finetune.txt")
    for epoch in range(1, cfg.num_epochs + 1):
        print(f"\n========== Epoch {epoch}/{cfg.num_epochs} ==========")

        train_loss, train_da_loss, train_ll_loss = train_one_epoch(
            model, train_loader, optimizer, scaler, epoch
        )
        val_loss = validate(model, val_loader, epoch)

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"[Epoch {epoch}/{cfg.num_epochs}] "
            f"LR={current_lr:.6e} | "
            f"train_loss={train_loss:.4f} "
            f"(da={train_da_loss:.4f}, ll={train_ll_loss:.4f}) | "
            f"val_loss={val_loss:.4f}"
        )

        # 记录到独立的 finetune 日志文件，方便和第一次大训练分开
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(
                f"{epoch},{current_lr:.6e},"
                f"{train_loss:.6f},{train_da_loss:.6f},{train_ll_loss:.6f},"
                f"{val_loss:.6f}\n"
            )

        # 保存最优模型
        if val_loss < best_val:
            best_val = val_loss
            save_path = os.path.join(cfg.save_dir, f"twinlitenetpp_best_finetune.pth")
            torch.save(model.state_dict(), save_path)
            print(f"[BEST] Val loss improved to {best_val:.4f}. Saved to {save_path}")

        # 每隔若干 epoch 也可另存一份（区分 finetune）
        if epoch % 10 == 0:
            ckpt_path = os.path.join(cfg.save_dir, f"twinlitenetpp_finetune_epoch{epoch}.pth")
            torch.save(model.state_dict(), ckpt_path)
            print(f"[CKPT] Saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()
