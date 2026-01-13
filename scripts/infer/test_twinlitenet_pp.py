import os
import sys
import cv2
import torch
import numpy as np

# ============================================================
# 自动定位：项目根目录 ROOT_DIR
# 当前文件：E:\detected sys\highway_perception_v2\scripts\infer\test_twinlitenetpp.py
# 上两层就是根目录：E:\detected sys\highway_perception_v2
# ============================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))              # ...\scripts\infer
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))     # ...\highway_perception_v2

if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)
    print(f"[DEBUG] add ROOT_DIR to sys.path: {ROOT_DIR}")

ROOT = ROOT_DIR  # 如果你在别处也用 ROOT，这里兼容一下

# ============================================================
# 导入模型（纯模型版本 TwinLiteNetPP）
# ============================================================
from models.seg.twinlitenet_pp import TwinLiteNetPP


def load_model(weight_path: str, device: str = "cuda"):
    """
    加载 TwinLiteNetPP 并加载训练好的权重.
    兼容 DataParallel 保存的 state_dict.
    """
    model = TwinLiteNetPP(use_refine=True)

    if device == "cuda" and torch.cuda.is_available():
        model = torch.nn.DataParallel(model).to(device)
    else:
        model = model.to(device)

    state = torch.load(weight_path, map_location=device)

    missing, unexpected = model.load_state_dict(state, strict=False)
    print("[INFO] missing keys:", missing)
    print("[INFO] unexpected keys:", unexpected)

    model.eval()
    return model


# ============================================================
# 预处理（与训练保持一致：resize + BGR→RGB + /255）
# ============================================================
def preprocess(img_bgr, img_size=(640, 360), device="cuda"):
    """
    img_bgr: 原始 BGR 图像 (H, W, 3)
    返回: [1, 3, H, W] Tensor
    """
    img = cv2.resize(img_bgr, img_size)
    img = img[:, :, ::-1]  # BGR -> RGB
    img = img.transpose(2, 0, 1)  # HWC -> CHW
    img = np.ascontiguousarray(img, dtype=np.float32) / 255.0

    tensor = torch.from_numpy(img).unsqueeze(0)
    return tensor.to(device)


# ============================================================
# mask 可视化
# ============================================================
def overlay_masks(img_bgr, da_mask, ll_mask):
    """
    img_bgr: 原始 BGR（已 resize 到 img_size）
    da_mask, ll_mask: uint8 [H, W], 值为 0 或 255
    """

    vis = img_bgr.copy()

    # 可行驶区域：绿
    vis[da_mask > 100] = [0, 255, 0]

    # 车道线：红
    vis[ll_mask > 100] = [0, 0, 255]

    return vis


# ============================================================
# 推理
# ============================================================
def run_inference(model, img_bgr, img_size=(640, 360), device="cuda"):

    inp = preprocess(img_bgr, img_size=img_size, device=device)

    with torch.no_grad():
        da_logits, ll_logits = model(inp)

        da_prob = torch.softmax(da_logits, dim=1)
        ll_prob = torch.softmax(ll_logits, dim=1)

        _, da_pred = torch.max(da_prob, dim=1)
        _, ll_pred = torch.max(ll_prob, dim=1)


    # 转为 0/255 的 mask，便于可视化
    da_mask = (da_pred.cpu().numpy()[0] * 255).astype(np.uint8)
    ll_mask = (ll_pred.cpu().numpy()[0] * 255).astype(np.uint8)

    # ======【新增】保存二值 mask 用于检查 ======
    cv2.imwrite("da_mask_debug.png", da_mask)
    cv2.imwrite("ll_mask_debug.png", ll_mask)
    print("[DEBUG] Saved da_mask_debug.png & ll_mask_debug.png")

    img_resized = cv2.resize(img_bgr, img_size)
    vis = overlay_masks(img_resized, da_mask, ll_mask)


    return vis, da_mask, ll_mask


# ============================================================
# 主函数
# ============================================================
def main():

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ✅ 关键修改：权重路径与当前训练脚本保持一致
    weight_path = os.path.join(
        ROOT, "checkpoints", "twinlitenetpp_geodepth", "twinlitenetpp_best_finetune.pth"
    )

    # 输入图片（你可以随时改）
    image_path = os.path.join(ROOT, "assets", "images", "test.jpg")

    # 输出目录
    output_dir = os.path.join(ROOT, "outputs", "twinlitenetpp")
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "test_twinlitenetpp_vis.jpg")

    print("[INFO] Project ROOT_DIR:", ROOT_DIR)
    print("[INFO] Loading model:", weight_path)
    model = load_model(weight_path, device)

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {image_path}")

    vis, _, _ = run_inference(model, img, img_size=(640, 360), device=device)

    cv2.imwrite(save_path, vis)
    print(f"[INFO] Result saved → {save_path}")


if __name__ == "__main__":
    main()
