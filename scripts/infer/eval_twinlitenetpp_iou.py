import os
import sys
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm  # ✅ 新增

# ============================================================
# 把项目根目录加入 sys.path
# ============================================================
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from models.seg.twinlitenet_pp import TwinLiteNetPP
from scripts.train.train_twinlitenetpp import BDDLaneDrivableDataset, cfg


def compute_iou(pred, target, num_classes=2):
    """
    pred, target: [N, H, W]，数值范围 [0, C-1]
    返回每个类别 IoU 和 mIoU
    """
    ious = []
    pred = pred.view(-1)
    target = target.view(-1)

    for cls in range(num_classes):
        pred_inds = pred == cls
        target_inds = target == cls

        inter = (pred_inds & target_inds).sum().item()
        union = (pred_inds | target_inds).sum().item()

        if union == 0:
            ious.append(float("nan"))  # 该类在这一批里可能不存在
        else:
            ious.append(inter / union)

    valid_ious = [x for x in ious if not (x != x)]  # 去掉 nan
    miou = sum(valid_ious) / len(valid_ious) if valid_ious else 0.0
    return ious, miou


def main():
    device = cfg.device

    # ===== 1. 加载 val 数据集 =====
    val_dataset = BDDLaneDrivableDataset(
        cfg.img_val_dir, cfg.da_val_dir, cfg.ll_val_dir, img_size=cfg.img_size
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,          # 可以改大一点比如 4，看显存
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    print(f"[INFO] Val samples: {len(val_dataset)}")

    # ===== 2. 构建模型 & 加载权重 =====
    model = TwinLiteNetPP(use_refine=True)

    # 训练时是 DataParallel 存的，就保持一致
    if device == "cuda":
        model = torch.nn.DataParallel(model).to(device)
    else:
        model = model.to(device)

    weight_path = os.path.join(cfg.save_dir, "twinlitenetpp_best_finetune.pth")
    print("Loading:", weight_path)

    state = torch.load(weight_path, map_location=device)
    model.load_state_dict(state, strict=False)
    model.eval()

    # ===== 3. 逐张计算 mIoU（带进度条） =====
    da_iou_sum = 0.0
    ll_iou_sum = 0.0
    da_cnt = 0
    ll_cnt = 0

    # 如果你只想先测试前 N 张是否正常，可以设置：
    # ===== 3. 逐张计算 mIoU（带进度条） =====
    da_iou_sum = 0.0
    ll_iou_sum = 0.0
    da_cnt = 0
    ll_cnt = 0

    # 只评估前 N 个 batch（比如 500）；如果想全量评估就改为 None
    MAX_ITERS = None

    total_iters = len(val_loader) if MAX_ITERS is None else min(len(val_loader), MAX_ITERS)

    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Evaluating", ncols=120, total=total_iters)
        for it, (img, da_gt, ll_gt, name) in enumerate(pbar):
            img = img.to(device)
            da_gt = da_gt.to(device)
            ll_gt = ll_gt.to(device)

            da_logits, ll_logits = model(img)

            da_pred = torch.argmax(F.softmax(da_logits, dim=1), dim=1)
            ll_pred = torch.argmax(F.softmax(ll_logits, dim=1), dim=1)

            da_ious, da_m = compute_iou(da_pred.cpu(), da_gt.cpu(), num_classes=2)
            ll_ious, ll_m = compute_iou(ll_pred.cpu(), ll_gt.cpu(), num_classes=2)

            if da_m == da_m:
                da_iou_sum += da_m
                da_cnt += 1
            if ll_m == ll_m:
                ll_iou_sum += ll_m
                ll_cnt += 1

            cur_da_miou = da_iou_sum / max(da_cnt, 1)
            cur_ll_miou = ll_iou_sum / max(ll_cnt, 1)
            pbar.set_postfix(
                da_mIoU=f"{cur_da_miou:.4f}",
                ll_mIoU=f"{cur_ll_miou:.4f}"
            )

            # ✅ 正确的中断条件
            if MAX_ITERS is not None and (it + 1) >= MAX_ITERS:
                break

    da_miou = da_iou_sum / max(da_cnt, 1)
    ll_miou = ll_iou_sum / max(ll_cnt, 1)

    print("======================================")
    print(f"Drivable mIoU: {da_miou:.4f}")
    print(f"Lane     mIoU: {ll_miou:.4f}")
    print("======================================")


if __name__ == "__main__":
    main()
