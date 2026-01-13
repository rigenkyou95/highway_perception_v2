# test_geo_depth.py
# ------------------------------------------------------------
# TwinLiteNetPP 车道线/可行驶区域分割 + 几何测距可视化
#
# 使用说明:
#   (dl) E:\detected sys\highway_perception_v2> cd scripts\infer
#   (dl) E:\detected sys\highway_perception_v2\scripts\infer> python test_geo_depth.py
#
# 默认:
#   - 输入图片: ROOT/assets/images/test.jpg
#   - 模型权重: ROOT/models/ckpts/twinlitenetpp/twinlitenetpp_best.pth
#   - 输出目录: ROOT/outputs/twinlitenetpp_geo/
#
# 注意:
#   1. 请确保 TwinLiteNetPP 的输入尺寸与 IMG_W, IMG_H 一致
#   2. 预处理部分请尽量与训练时保持一致(归一化/mean/std等)
#   3. 若通道顺序与此处假设不一致, 需在 infer_segmentation 中调整
# ------------------------------------------------------------

import os
import sys
import math
import argparse
import cv2
import numpy as np
import torch
import torch.nn.functional as F


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, ROOT)

print("Project ROOT:", ROOT)


from models.seg.twinlitenet_pp import TwinLiteNetPP  # 确保已正确实现并能导入

# ---------------- 相机 / 车辆参数配置 ----------------
# 网络输入尺寸 (请与训练时一致)
IMG_W, IMG_H = 640, 360

# 摄像头安装信息 (根据你的 C63 + XT30 + 后视镜位置估计)
CAMERA_HEIGHT = 1.35  # m, 从路面到摄像头中心
PITCH_DEG     = 5.0   # 向下俯仰角, 之后可微调 3~7°

# XT30 原始视频参数
NATIVE_W, NATIVE_H = 2304, 1296   # XT30 原始分辨率
HFOV_DEG = 140.0                  # XT30 水平视场角 140°

# 在原始分辨率下计算 fx, 再按比例缩放到网络输入尺寸
fx_native = (NATIVE_W / 2.0) / math.tan(math.radians(HFOV_DEG / 2.0))
scale_w   = IMG_W / NATIVE_W

FX = fx_native * scale_w
FY = FX  # 先假设像素为方形, fx ≈ fy
CX = IMG_W / 2.0
CY = IMG_H / 2.0

print(f"[Camera] FX={FX:.2f}, FY={FY:.2f}, CX={CX:.1f}, CY={CY:.1f}, H={CAMERA_HEIGHT}m, pitch={PITCH_DEG}°")


# ---------------- 几何函数: 像素坐标 -> 地平面坐标 ----------------
def pixel_to_world(u, v, fx, fy, cx, cy, H, pitch_deg):
    """
    将像素坐标 (u, v) 映射到地面平面上的坐标 (X, Z)

    坐标系假设:
      - 相机坐标系: Z 轴指向前方, X 轴向右, Y 轴向下(或向上只要保持一致)
      - 这里采用: 相机初始指向前方, 再绕 X 轴做俯仰旋转
      - 地面平面: Y = -H (假定相机位于原点, 地面在其下方)

    输入:
      u, v: 像素坐标
      fx, fy, cx, cy: 相机内参
      H: 相机高度
      pitch_deg: 俯仰角(度), 向下为正

    输出:
      (X, Z): 世界坐标, 单位 m
      若射线无法与地面相交(如 dy>=0), 返回 None
    """
    pitch = math.radians(pitch_deg)

    # 像平面归一化坐标
    x = (u - cx) / fx
    y = (v - cy) / fy

    # 相机坐标系下的射线方向 (未归一化也可以)
    ray_cam = np.array([x, y, 1.0], dtype=np.float64)

    # 绕 X 轴旋转 (俯仰)
    R_pitch = np.array([
        [1,              0,               0],
        [0,  math.cos(pitch), -math.sin(pitch)],
        [0,  math.sin(pitch),  math.cos(pitch)]
    ], dtype=np.float64)

    ray_world = R_pitch @ ray_cam  # 3x1

    dy = ray_world[1]
    if dy >= 0:
        # 射线朝上或平行, 不会与地面相交
        return None

    # 地面平面: Y = -H, 求 t 使得 ray_world_y * t = -H
    t = -H / dy
    X = ray_world[0] * t
    Z = ray_world[2] * t

    return float(X), float(Z)


# ---------------- 模型加载与推理相关 ----------------
def load_model(weight_path: str, device: str = "cuda"):
    print(f"[Model] Loading TwinLiteNetPP from: {weight_path}")
    model = TwinLiteNetPP()

    # 兼容 DataParallel/单卡权重
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
    """
    基础预处理:
      - resize 到 (IMG_W, IMG_H)
      - BGR -> RGB
      - 归一化到 [0,1]
      - HWC -> CHW, float32
    若你训练时使用了 mean/std 归一化, 可在此处补上.
    """
    img_resized = cv2.resize(img_bgr, (IMG_W, IMG_H), interpolation=cv2.INTER_LINEAR)
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img = img_rgb.astype(np.float32) / 255.0

    # TODO: 如果训练时有 mean/std, 在此处减/除
    # mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    # std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    # img = (img - mean) / std

    img = np.transpose(img, (2, 0, 1))  # CHW
    img_tensor = torch.from_numpy(img).unsqueeze(0)  # 1x3xH xW
    return img_tensor, img_resized  # 返回缩放后的图像用于可视化


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


# ---------------- 距离采样与可视化 ----------------
def compute_centerline_distances(drivable_mask: np.ndarray,
                                 fx: float, fy: float, cx: float, cy: float,
                                 H: float, pitch_deg: float,
                                 num_samples: int = 15):
    """
    在图像下方向上采样若干水平行, 在可行驶区域中取“中心像素”, 计算其几何距离.

    返回:
      distances: List[(u, v, X, Z)]
    """
    H_img, W_img = drivable_mask.shape
    distances = []

    for i in range(num_samples):
        # 从底部向上均匀采样
        v = H_img - 1 - int(i * H_img / num_samples)
        row = drivable_mask[v, :]
        xs = np.where(row)[0]
        if len(xs) == 0:
            continue

        u_center = int(xs.mean())

        res = pixel_to_world(u_center, v, fx, fy, cx, cy, H, pitch_deg)
        if res is None:
            continue

        X, Z = res
        distances.append((u_center, v, X, Z))

    return distances


def colorize_segmentation(seg_vis: np.ndarray):
    """
    将 seg_vis (0,1,2) 转为彩色图:
      - 0: 背景 -> 黑
      - 1: drivable -> 蓝/青
      - 2: lane -> 绿
    """
    H_img, W_img = seg_vis.shape
    color = np.zeros((H_img, W_img, 3), dtype=np.uint8)

    # drivable: 蓝色 (你也可以改成青色)
    color[seg_vis == 1] = (255, 0, 0)  # BGR
    # lane: 绿色
    color[seg_vis == 2] = (0, 255, 0)

    return color


def draw_distances_on_image(img_bgr: np.ndarray, distances):
    """
    在图像上绘制采样点与前向距离 Z, 单位 m
    """
    vis = img_bgr.copy()
    for (u, v, X, Z) in distances:
        cv2.circle(vis, (u, v), 4, (0, 0, 255), -1)
        text = f"{Z:.1f}m"
        cv2.putText(vis, text, (u + 5, v - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 1, cv2.LINE_AA)
    return vis


# ---------------- 主流程 ----------------
def main():
    parser = argparse.ArgumentParser(description="TwinLiteNetPP Geometry-based Depth Estimation (Single Image)")
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
        help="output directory"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="cuda or cpu"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=15,
        help="number of horizontal lines to sample distances"
    )

    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # 读取图像
    img_bgr = cv2.imread(args.img)
    if img_bgr is None:
        print(f"[Error] Failed to read image: {args.img}")
        return

    # 加载模型
    model, device = load_model(args.weights, args.device)

    # 分割推理
    seg_vis, drivable_mask, img_resized = infer_segmentation(model, device, img_bgr)

    # 计算几何距离
    distances = compute_centerline_distances(
        drivable_mask,
        FX, FY, CX, CY,
        CAMERA_HEIGHT,
        PITCH_DEG,
        num_samples=args.num_samples
    )

    print("[Distances] Sampled points (u, v, X, Z):")
    for (u, v, X, Z) in distances:
        print(f"  pixel=({u:3d},{v:3d}) -> X={X:6.2f} m, Z={Z:6.2f} m")

    # 可视化: 分割颜色图
    seg_color = colorize_segmentation(seg_vis)
    seg_overlay = cv2.addWeighted(img_resized, 0.6, seg_color, 0.4, 0)

    # 可视化: 在图像上画距离
    depth_vis = draw_distances_on_image(seg_overlay, distances)

    # 保存结果
    base_name = os.path.splitext(os.path.basename(args.img))[0]
    out_seg = os.path.join(args.outdir, f"{base_name}_seg_overlay.png")
    out_depth = os.path.join(args.outdir, f"{base_name}_geo_depth.png")

    cv2.imwrite(out_seg, seg_overlay)
    cv2.imwrite(out_depth, depth_vis)

    print(f"[Output] Segmentation overlay saved to: {out_seg}")
    print(f"[Output] Geo depth visualization saved to: {out_depth}")


if __name__ == "__main__":
    main()
