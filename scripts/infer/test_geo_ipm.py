# test_geo_ipm.py
# ------------------------------------------------------------
# TwinLiteNetPP 分割 + IPM(鸟瞰图) + 几何距离可视化
#
# 使用:
#   (dl) E:\detected sys\highway_perception_v2> cd scripts\infer
#   (dl) E:\detected sys\highway_perception_v2\scripts\infer> python test_geo_ipm.py
#
# 输入:
#   - ROOT/assets/images/test.jpg
# 模型权重:
#   - ROOT/models/ckpts/twinlitenetpp_geodepth/twinlitenetpp_best.pth
# 输出:
#   - ROOT/outputs/twinlitenetpp_geo/test_ipm_bev.png      (BEV 彩色图)
#   - ROOT/outputs/twinlitenetpp_geo/test_ipm_bev_dist.png (BEV 上带距离标注)
# ------------------------------------------------------------

import os
import sys
import math
import argparse

import cv2
import numpy as np
import torch
import torch.nn.functional as F

# ---------- 让 Python 能找到 highway_perception_v2 这个包 ----------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, ROOT)
print("Project ROOT:", ROOT)

from models.seg.twinlitenet_pp import TwinLiteNetPP


# ---------------- 相机 / 车辆参数配置 ----------------
# 网络输入尺寸 (请与 TwinLiteNetPP 训练时保持一致)
IMG_W, IMG_H = 640, 360

# 摄像头安装信息 (C63 + XT30, 先用估计值)
CAMERA_HEIGHT = 1.35  # m, 从路面到摄像头中心
PITCH_DEG     = 5.0   # 向下俯仰角, 之后可微调 3~7°

# XT30 原始视频参数
NATIVE_W, NATIVE_H = 2304, 1296   # XT30 原始分辨率
HFOV_DEG = 140.0                  # 水平视场角 140°

# 在原始分辨率下计算 fx, 再按比例缩放到网络输入尺寸
fx_native = (NATIVE_W / 2.0) / math.tan(math.radians(HFOV_DEG / 2.0))
scale_w   = IMG_W / NATIVE_W

FX = fx_native * scale_w
FY = FX
CX = IMG_W / 2.0
CY = IMG_H / 2.0

print(f"[Camera] FX={FX:.2f}, FY={FY:.2f}, CX={CX:.1f}, CY={CY:.1f}, H={CAMERA_HEIGHT}m, pitch={PITCH_DEG}°")

# 预先算好俯仰旋转矩阵
PITCH_RAD = math.radians(PITCH_DEG)
R_PITCH = np.array([
    [1,                 0,                  0],
    [0, math.cos(PITCH_RAD), -math.sin(PITCH_RAD)],
    [0, math.sin(PITCH_RAD),  math.cos(PITCH_RAD)],
], dtype=np.float64)
R_PITCH_T = R_PITCH.T


# ---------------- BEV(鸟瞰图) 配置 ----------------
# 车体坐标系: X 左右, Z 前方
# BEV 范围(单位: m)
X_MIN, X_MAX = -10.0, 10.0   # 左右各 10m
Z_MIN, Z_MAX = 0.0, 60.0     # 前方 0~60m

BEV_RES = 0.20  # 每个 BEV 像素对应的实际长度 (m)

BEV_W = int((X_MAX - X_MIN) / BEV_RES)  # 列数
BEV_H = int((Z_MAX - Z_MIN) / BEV_RES)  # 行数 (远处在上, 近处在下)

print(f"[BEV] size = {BEV_W} x {BEV_H} pixels, res = {BEV_RES} m/px")


# ---------------- 模型加载 & 预处理 ----------------
def load_model(weight_path: str, device: str = "cuda"):
    print(f"[Model] Loading TwinLiteNetPP from: {weight_path}")
    model = TwinLiteNetPP()

    state = torch.load(weight_path, map_location="cpu")
    if "state_dict" in state:
        state = state["state_dict"]

    new_state = {}
    for k, v in state.items():
        if k.startswith("module."):
            k = k[len("module."):]
        new_state[k] = v

    model.load_state_dict(new_state, strict=False)
    model.eval()

    if device == "cuda" and torch.cuda.is_available():
        model = model.cuda()
        print("[Model] Using CUDA")
    else:
        device = "cpu"
        print("[Model] Using CPU")

    return model, device


def preprocess_image(img_bgr: np.ndarray):
    """resize + BGR2RGB + [0,1] + CHW tensor"""
    img_resized = cv2.resize(img_bgr, (IMG_W, IMG_H), interpolation=cv2.INTER_LINEAR)
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img = img_rgb.astype(np.float32) / 255.0

    # 若训练时使用了 mean/std, 在此处补上
    # mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    # std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    # img = (img - mean) / std

    img = np.transpose(img, (2, 0, 1))  # CHW
    img_tensor = torch.from_numpy(img).unsqueeze(0)

    return img_tensor, img_resized


def infer_segmentation(model, device, img_bgr: np.ndarray):
    """
    正确版 TwinLiteNetPP 推理
    输出：
      seg_vis: 0=背景,1=drivable(蓝色),2=lane(绿色)
      drivable_mask: bool mask (True=可行驶区域)
    """

    # ----------- 预处理（必须与训练保持一致） -----------
    img_resized = cv2.resize(img_bgr, (IMG_W, IMG_H))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = np.transpose(img_rgb, (2,0,1))  # CHW
    img_tensor = torch.from_numpy(img).unsqueeze(0).to(device)

    # ----------- 推理 -----------
    with torch.no_grad():
        da_logits, ll_logits = model(img_tensor)

        # 确保尺寸一致
        da_logits = F.interpolate(da_logits, size=(IMG_H, IMG_W),
                                  mode="bilinear", align_corners=False)
        ll_logits = F.interpolate(ll_logits, size=(IMG_H, IMG_W),
                                  mode="bilinear", align_corners=False)

        # ----------- Drivable 区域（蓝色）-----------
        da_prob = torch.softmax(da_logits, dim=1)[0,1].cpu().numpy()
        drivable_mask = da_prob > 0.5

        # ----------- Lane 车道线（绿色）-----------
        ll_prob = torch.softmax(ll_logits, dim=1)[0,1].cpu().numpy()
        lane_mask = ll_prob > 0.5

    # ----------- 组合可视化 seg_vis -----------
    seg_vis = np.zeros((IMG_H, IMG_W), dtype=np.uint8)
    seg_vis[drivable_mask] = 1   # 蓝色
    seg_vis[lane_mask] = 2       # 绿色

    return seg_vis, drivable_mask, img_resized


# ---------------- World -> Image 投影 (用于 IPM) ----------------
def project_ground_to_image(X: float, Z: float,
                            fx: float, fy: float, cx: float, cy: float,
                            H: float):
    """
    输入:
      X,Z: 地面坐标 (单位: m)
      H: 相机高度 (m)
    坐标系:
      - 世界坐标: 相机在原点, 地面平面 Y=-H
      - R_PITCH: 将相机坐标旋转到世界坐标 (即世界 = R_PITCH * cam)
    这里反向使用: cam = R_PITCH^T * world
    """
    world = np.array([X, -H, Z], dtype=np.float64)
    cam = R_PITCH_T @ world

    Zc = cam[2]
    if Zc <= 0:
        return None

    x_norm = cam[0] / Zc
    y_norm = cam[1] / Zc

    u = fx * x_norm + cx
    v = fy * y_norm + cy

    return u, v


# ---------------- IPM: 把 drivable_mask 映射到 BEV ----------------
def ipm_drivable_to_bev(drivable_mask: np.ndarray):
    """
    输入:
      drivable_mask: HxW, bool
    输出:
      bev_mask: BEV_H x BEV_W, uint8, 1 表示可行驶区域
    """
    bev_mask = np.zeros((BEV_H, BEV_W), dtype=np.uint8)

    H_img, W_img = drivable_mask.shape
    assert H_img == IMG_H and W_img == IMG_W

    for zi in range(BEV_H):
        # 将 BEV 图像的行 index 转换为实际距离 Z (远处在上方, 近处在下方)
        Z = Z_MIN + (Z_MAX - Z_MIN) * (1.0 - (zi + 0.5) / BEV_H)

        for xi in range(BEV_W):
            X = X_MIN + (xi + 0.5) * BEV_RES

            proj = project_ground_to_image(X, Z, FX, FY, CX, CY, CAMERA_HEIGHT)
            if proj is None:
                continue

            u, v = proj
            u_i = int(round(u))
            v_i = int(round(v))

            if 0 <= u_i < W_img and 0 <= v_i < H_img:
                if drivable_mask[v_i, u_i]:
                    bev_mask[zi, xi] = 1

    return bev_mask


def colorize_bev(bev_mask: np.ndarray):
    """
    BEV 单通道 mask -> 彩色图: 可行驶区域为绿色
    """
    H_bev, W_bev = bev_mask.shape
    bev_color = np.zeros((H_bev, W_bev, 3), dtype=np.uint8)
    bev_color[bev_mask == 1] = (0, 255, 0)
    return bev_color


def draw_bev_centerline_distances(bev_mask: np.ndarray,
                                  bev_color: np.ndarray,
                                  z_ticks=None):
    """
    在 BEV 中央 X=0 的位置, 标注若干距离点 (比如 5m,10m,...)
    z_ticks: 距离列表, 单位 m
    """
    if z_ticks is None:
        z_ticks = [5, 10, 15, 20, 30, 40, 50]

    H_bev, W_bev = bev_mask.shape
    bev_vis = bev_color.copy()

    # 计算 x=0 对应的列 index
    center_x_idx = int((-X_MIN) / BEV_RES)

    for Z in z_ticks:
        if Z < Z_MIN or Z > Z_MAX:
            continue

        # Z -> 行 index (注意 BEV 是远处在上, 近处在下)
        rel = (Z - Z_MIN) / (Z_MAX - Z_MIN)
        zi = int((1.0 - rel) * BEV_H)

        if zi < 0 or zi >= H_bev:
            continue

        # 标记一个小圆点
        cv2.circle(bev_vis, (center_x_idx, zi), 3, (0, 0, 255), -1)

        text = f"{Z:.0f}m"
        cv2.putText(bev_vis, text, (center_x_idx + 5, zi - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (0, 255, 0), 1, cv2.LINE_AA)

    return bev_vis


# ---------------- 主流程 ----------------
def main():
    parser = argparse.ArgumentParser(description="TwinLiteNetPP IPM Bird's-eye View Test")
    parser.add_argument(
        "--img",
        type=str,
        default=os.path.join(ROOT, "assets", "images", "test.jpg"),
        help="input image path"
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=os.path.join(
            ROOT,
            "checkpoints",
            "twinlitenetpp_geodepth",
            "twinlitenetpp_best_finetune.pth"
        ),
        help="TwinLiteNetPP weight path"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=os.path.join(ROOT, "outputs", "twinlitenetpp_geo"),
        help="output directory for BEV images"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="cuda or cpu"
    )

    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    img_bgr = cv2.imread(args.img)
    if img_bgr is None:
        print(f"[Error] failed to read image: {args.img}")
        return

    # 加载模型 & 分割
    model, device = load_model(args.weights, args.device)
    seg_vis, drivable_mask, img_resized = infer_segmentation(model, device, img_bgr)

    # 做 IPM
    print("[IPM] projecting drivable area to BEV...")
    bev_mask = ipm_drivable_to_bev(drivable_mask)
    bev_color = colorize_bev(bev_mask)
    bev_vis_dist = draw_bev_centerline_distances(bev_mask, bev_color)

    # 保存
    base_name = os.path.splitext(os.path.basename(args.img))[0]
    out_bev = os.path.join(args.outdir, f"{base_name}_ipm_bev.png")
    out_bev_dist = os.path.join(args.outdir, f"{base_name}_ipm_bev_dist.png")

    cv2.imwrite(out_bev, bev_color)
    cv2.imwrite(out_bev_dist, bev_vis_dist)

    print(f"[Output] BEV mask saved to: {out_bev}")
    print(f"[Output] BEV with distances saved to: {out_bev_dist}")


if __name__ == "__main__":
    main()
