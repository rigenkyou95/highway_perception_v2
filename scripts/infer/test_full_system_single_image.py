import os
import sys
import argparse
from typing import Tuple, Optional, List
from dataclasses import dataclass
import math

import cv2
import numpy as np
import torch
from ultralytics import YOLO

import matplotlib
matplotlib.use("Agg")  # 方便无界面环境保存图像
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


# ============================================================
# 全局开关
# ============================================================
# 是否启用“利用车道宽 3.6m 标定相机高度”的几何标定
USE_LANE_WIDTH_CALIB = False  # 先关掉，想试再改 True


# ============================================================
# 路径与环境设置：scripts/infer/*.py → 项目根目录在上两层
# ============================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))

if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

ROOT = ROOT_DIR

# ============================================================
# 导入自定义模型
# ============================================================
from models.seg.twinlitenet_pp.TwinLitePP import TwinLiteNetPP
from models.depth import DepthAnythingV2TinyWrapper


# ============================================================
# 相机配置（几何流 / BEV / 3D 可视化）
# ============================================================

@dataclass
class CameraConfig:
    fx: float = 640.0
    fy: float = 640.0
    cx: float = 640.0
    cy: float = 360.0
    cam_height: float = 1.4      # 相机离地高度（米，初始值）
    fov_deg: float = 90.0        # 水平视场角（度）
    # 自动拟合到的俯仰角（向下为正，单位：度）
    pitch_deg: float = 0.0


@dataclass
class BEVConfig:
    width: int = 800             # BEV 图宽 (px)
    height: int = 1200           # BEV 图高 (px)
    scale: float = 4.0           # 每米多少像素 (px/m)
    grid_step_small: float = 5.0 # 小刻度（米）
    grid_step_big: float = 10.0  # 大刻度（米）
    max_range_m: float = 200.0   # 可视化最远距离（米）



# ============================================================
# 全局配置
# ============================================================

class Config:
    # 输入图片（默认 assets/images/test.jpg）
    default_img = os.path.join(ROOT, "assets", "images", "test.jpg")

    # TwinLiteNetPP 权重（分开使用 DA 和 LaneGeo）
    seg_weight_da = os.path.join(
        ROOT, "models", "ckpts", "twinlitenetpp_geodepth", "twinlitenetpp_da_best.pth"
    )
    seg_weight_lane = os.path.join(
        ROOT, "models", "ckpts", "twinlitenetpp_geodepth", "twinlitenetpp_lanegeo_best.pth"
    )

    # Depth Anything V2 Tiny 权重
    depth_ckpt = os.path.join(
        ROOT, "models", "ckpts", "depth_anything_v2", "depth_anything_v2_vits.pth"
    )

    # YOLO11n 权重
    yolo_weight = os.path.join(
        ROOT, "third_party", "yolov11n", "yolo11n.pt"
    )

    # 输出路径规范
    out_seg = os.path.join(ROOT, "outputs", "twinlitenetpp")
    out_depth = os.path.join(ROOT, "outputs", "depth_anything_v2")
    out_det = os.path.join(ROOT, "outputs", "yolov11n")
    out_fusion = os.path.join(ROOT, "outputs", "fusion")

    # TwinLiteNetPP 训练输入尺寸
    img_size = (640, 360)  # (W, H)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 相机 & BEV 配置
    camera = CameraConfig()
    bev = BEVConfig()


cfg = Config()


# ============================================================
# 通用工具函数
# ============================================================

def ensure_dirs():
    for d in [cfg.out_seg, cfg.out_depth, cfg.out_det, cfg.out_fusion]:
        os.makedirs(d, exist_ok=True)


# ============================================================
# TwinLiteNetPP 分割相关
# ============================================================

def load_seg_model(weight_path: str, device: str = "cuda") -> torch.nn.Module:
    """
    加载 TwinLiteNetPP 并加载训练好的权重.
    兼容 DataParallel 保存的 state_dict。
    """
    if not os.path.isfile(weight_path):
        raise FileNotFoundError(f"[Seg] 权重不存在: {weight_path}")

    model = TwinLiteNetPP(use_refine=True)

    if device == "cuda" and torch.cuda.is_available():
        model = torch.nn.DataParallel(model).to(device)
    else:
        model = model.to(device)

    state = torch.load(weight_path, map_location=device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[Seg] Load {os.path.basename(weight_path)}")
    print("[Seg]   missing keys:", missing)
    print("[Seg]   unexpected keys:", unexpected)

    model.eval()
    return model


def preprocess_for_seg(img_bgr: np.ndarray, size: Tuple[int, int]) -> Tuple[torch.Tensor, np.ndarray]:
    """
    将 BGR 图像 resize → RGB → CHW → tensor
    """
    img_resized = cv2.resize(img_bgr, size)  # (W,H)
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img_chw = img_rgb.transpose(2, 0, 1)  # 3,H,W
    tensor = torch.from_numpy(img_chw).unsqueeze(0)  # 1,3,H,W
    return tensor, img_resized


def infer_da_ll_masks(
    model: torch.nn.Module,
    img_bgr: np.ndarray,
    device: str = "cuda",
    da_thr: float = 0.35,
    ll_thr: float = 0.5,
    postproc_da: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    前向推理:
      - da_thr: DA 使用的概率阈值（默认 0.35，比 0.5 低）
      - ll_thr: Lane 使用的概率阈值（默认 0.5）
      - postproc_da: 是否对 DA 做形态学填充
    返回：
      - da_mask_full: 0/1 可行驶区域 mask（原图大小）
      - ll_mask_full: 0/1 车道线 mask（原图大小）
    """
    h0, w0 = img_bgr.shape[:2]
    tensor, _ = preprocess_for_seg(img_bgr, cfg.img_size)
    tensor = tensor.to(device)

    with torch.no_grad():
        da_logits, ll_logits = model(tensor)

    da_probs = torch.softmax(da_logits, dim=1)[0, 1].cpu().numpy()
    ll_probs = torch.softmax(ll_logits, dim=1)[0, 1].cpu().numpy()

    # --- 1) DA: 降低阈值 + 形态学闭运算填充 ---
    da_mask = (da_probs > da_thr).astype(np.uint8)

    if postproc_da:
        k = np.ones((5, 5), np.uint8)
        # 先闭运算填充小洞，再轻微膨胀一下让边界更连贯
        da_mask = cv2.morphologyEx(da_mask, cv2.MORPH_CLOSE, k, iterations=2)
        da_mask = cv2.dilate(da_mask, k, iterations=1)

    # --- 2) Lane: 维持原先 0.5 阈值，不做膨胀，保持细线 ---
    ll_mask = (ll_probs > ll_thr).astype(np.uint8)

    da_mask_full = cv2.resize(da_mask, (w0, h0), interpolation=cv2.INTER_NEAREST)
    ll_mask_full = cv2.resize(ll_mask, (w0, h0), interpolation=cv2.INTER_NEAREST)

    return da_mask_full, ll_mask_full



def build_seg_overlay(
    img_bgr: np.ndarray,
    da_mask_full: np.ndarray,
    ll_mask_full: np.ndarray
) -> np.ndarray:
    """
    根据给定的 da_mask / ll_mask 在原图上生成彩色叠加图。
    """
    overlay = img_bgr.copy().astype(np.float32)

    # 可行驶区域：绿色
    green = np.array([0, 255, 0], dtype=np.float32)
    overlay[da_mask_full == 1] = overlay[da_mask_full == 1] * 0.4 + green * 0.6

    # 车道线：红色
    red = np.array([0, 0, 255], dtype=np.float32)
    overlay[ll_mask_full == 1] = overlay[ll_mask_full == 1] * 0.4 + red * 0.6

    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    return overlay


# ============================================================
# DepthAnything V2 相对深度（视觉流）
# ============================================================

def load_depth_model(device: str = "cuda") -> DepthAnythingV2TinyWrapper:
    ckpt_path = cfg.depth_ckpt
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"DepthAnythingV2 权重不存在: {ckpt_path}")
    model = DepthAnythingV2TinyWrapper(ckpt_path=ckpt_path, device=device)
    return model


def vis_depth(depth: np.ndarray) -> np.ndarray:
    depth = depth.astype(np.float32)
    d_min, d_max = depth.min(), depth.max()
    if d_max - d_min < 1e-6:
        norm = np.zeros_like(depth, dtype=np.uint8)
    else:
        norm = (depth - d_min) / (d_max - d_min + 1e-8)
        norm = (norm * 255.0).clip(0, 255).astype(np.uint8)

    color = cv2.applyColorMap(255 - norm, cv2.COLORMAP_MAGMA)
    return color


def infer_depth(depth_model: DepthAnythingV2TinyWrapper, img_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    depth_rel = depth_model(img_bgr)
    depth_vis = vis_depth(depth_rel)
    return depth_rel, depth_vis


# ============================================================
# YOLOv11n 检测
# ============================================================

def load_yolo_model(weight_path: str) -> YOLO:
    if not os.path.isfile(weight_path):
        raise FileNotFoundError(f"YOLOv11n 权重不存在: {weight_path}")
    model = YOLO(weight_path)
    return model


# ============================================================
# 几何流：由可行驶区域 mask + 相机模型估计稀疏 metric depth D_geo
# ============================================================

def compute_geo_depth_from_seg(
    da_mask: np.ndarray,
    ll_mask: np.ndarray,
    cam_cfg: CameraConfig
) -> np.ndarray:
    h, w = da_mask.shape
    depth_geo = np.full((h, w), np.nan, dtype=np.float32)

    ys, xs = np.where(da_mask > 0)

    fx = cam_cfg.fx
    fy = cam_cfg.fy
    cx = cam_cfg.cx
    cy = cam_cfg.cy
    cam_h = cam_cfg.cam_height

    eps = 1e-6

    for y, x in zip(ys, xs):
        Yc = (y - cy) / (fy + eps)
        if Yc <= 0:
            continue
        t = cam_h / (Yc + eps)
        if t <= 0:
            continue
        Z = t
        depth_geo[y, x] = Z

    return depth_geo


# ============================================================
# 几何流 + 视觉流 融合：由 D_geo 和 D_rel 得到 D_abs
# ============================================================

def fuse_geo_and_visual_depth(
    depth_geo: np.ndarray,
    depth_rel: np.ndarray,
    min_valid_points: int = 500
) -> Tuple[np.ndarray, float]:
    assert depth_geo.shape == depth_rel.shape, "几何深度与视觉深度尺寸不一致"

    mask_geo = np.isfinite(depth_geo) & (depth_geo > 0)
    mask_rel = depth_rel > 1e-6
    mask = mask_geo & mask_rel

    num_valid = int(mask.sum())
    print(f"[Fusion] 有效几何点数: {num_valid}")

    if num_valid < min_valid_points:
        print("[Fusion] 有效几何点过少，退回使用纯视觉流（未对齐）")
        depth_abs = depth_rel.astype(np.float32).copy()
        return depth_abs, 1.0

    geo_vals = depth_geo[mask]
    rel_vals = depth_rel[mask]

    ratios = geo_vals / (rel_vals + 1e-6)
    ratios = ratios[np.isfinite(ratios) & (ratios > 0)]
    if ratios.size == 0:
        print("[Fusion] 比例全无效，退回使用纯视觉流")
        depth_abs = depth_rel.astype(np.float32).copy()
        return depth_abs, 1.0

    low_q = np.percentile(ratios, 5)
    high_q = np.percentile(ratios, 95)
    ratios_clipped = ratios[(ratios >= low_q) & (ratios <= high_q)]
    if ratios_clipped.size == 0:
        ratios_clipped = ratios

    s = float(np.median(ratios_clipped))
    print(f"[Fusion] 全局尺度因子 s = {s:.4f}")

    depth_abs = depth_rel.astype(np.float32) * s
    return depth_abs, s


# ============================================================
# 简单 1D KMeans（供标定、聚类车道线用）
# ============================================================

def simple_kmeans_1d(xs: np.ndarray, K: int, iters: int = 15) -> np.ndarray:
    """
    1D KMeans（仅用于 X 方向聚类车道线）
    xs: [N]
    返回 labels: [N] in {0..K-1}
    """
    N = xs.shape[0]
    if N == 0:
        return np.zeros(0, dtype=np.int32)
    K = max(1, min(K, N))

    percentiles = np.linspace(0, 100, K + 2)[1:-1]
    centers = np.percentile(xs, percentiles)

    labels = np.zeros(N, dtype=np.int32)
    for _ in range(iters):
        dists = np.abs(xs[:, None] - centers[None, :])
        labels = np.argmin(dists, axis=1)

        new_centers = centers.copy()
        for k in range(K):
            mask = labels == k
            if np.any(mask):
                new_centers[k] = xs[mask].mean()
        if np.allclose(new_centers, centers):
            break
        centers = new_centers

    return labels


# ============================================================
# ==== 车道线中心线提取 + 护栏过滤（用于自动 pitch）====
# ============================================================

def skeletonize_mask(mask: np.ndarray) -> np.ndarray:
    """
    将二值 mask 细化为骨架（单像素宽中心线）。
    优先使用 ximgproc.thinning，不可用时退回形态学 skeleton。
    输入 mask: 0/255
    """
    img = (mask > 0).astype(np.uint8) * 255

    try:
        import cv2.ximgproc as xip
        thin = xip.thinning(img)
        return thin
    except Exception:
        pass

    size = np.size(img)
    skel = np.zeros(img.shape, np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    done = False

    while not done:
        eroded = cv2.erode(img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()
        zeros = size - cv2.countNonZero(img)
        if zeros == size:
            done = True

    return skel


def filter_lane_components(ll_mask: np.ndarray) -> np.ndarray:
    """
    过滤掉“类似护栏的大块区域”，只保留较细且在下半部的线状连通域。
    如果过滤后几乎什么都不剩，则退回原始 mask。
    """
    h, w = ll_mask.shape
    binary = (ll_mask > 0).astype(np.uint8) * 255

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    filtered = np.zeros_like(binary, dtype=np.uint8)

    for i in range(1, num_labels):
        x, y, bw, bh, area = stats[i]

        if area < 30:
            continue

        aspect = bh / (bw + 1e-6)   # 高 / 宽
        bottom = y + bh

        if aspect < 0.4:        # 非常扁的，基本就是护栏
            continue
        if bh < h * 0.10:       # 垂直高度太短
            continue
        if bottom < h * 0.5:    # 不在下半部分
            continue

        filtered[labels == i] = 255

    nz_orig = int(np.count_nonzero(binary))
    nz_filt = int(np.count_nonzero(filtered))
    if nz_filt < 0.05 * nz_orig:
        print(f"[Pitch] filter_lane_components 过滤过多 "
              f"(orig={nz_orig}, filt={nz_filt})，退回原始 mask。")
        return binary

    return filtered


def prepare_lane_mask_for_pitch(ll_mask: np.ndarray,
                                save_debug: bool = True) -> np.ndarray:
    """
    综合步骤：
      1) 连通域分析，过滤掉护栏
      2) 对剩余 mask 做 thinning 得到车道中心线
    返回：thin_mask (0/255)，后续 pitch 标定使用这个
    """
    filtered = filter_lane_components(ll_mask)
    thin = skeletonize_mask(filtered)

    if save_debug:
        out_dir = cfg.out_fusion
        os.makedirs(out_dir, exist_ok=True)

        filtered_path = os.path.join(out_dir, "ll_pitch_filtered.png")
        thin_path = os.path.join(out_dir, "ll_pitch_thin.png")

        cv2.imwrite(filtered_path, filtered)
        cv2.imwrite(thin_path, thin)

        print(f"[Pitch] Saved ll_pitch_filtered.png ({np.count_nonzero(filtered)} pixels)")
        print(f"[Pitch] Saved ll_pitch_thin.png ({np.count_nonzero(thin)} pixels)")

    return thin


# ============================================================
# 利用车道宽 3.6m 自动拟合 pitch（俯仰角）
# ============================================================

def collect_lane_width_samples(
    ll_mask: np.ndarray,
    cam_cfg: CameraConfig,
    row_ratio: float = 0.7,
    window: int = 60,
    max_K: int = 4
) -> Tuple[np.ndarray, np.ndarray]:
    """
    从车道线 mask 中收集一批 (y, dx_pix) 样本：
      - y: 图像行号
      - dx_pix: 该行中“相邻两条车道线之间的最小像素宽度”
    ll_mask 应为“中心线版”（单像素宽）。
    """
    h, w = ll_mask.shape
    y_center = int(h * row_ratio)
    ys = range(max(0, y_center - window), min(h, y_center + window + 1))

    ys_list = []
    dx_list = []

    for y in ys:
        xs = np.where(ll_mask[y] > 0)[0]
        if len(xs) < 2:
            continue

        K = min(max_K, len(xs))
        labels = simple_kmeans_1d(xs.astype(np.float32), K=K)

        centers = []
        for k in range(K):
            mask = labels == k
            if np.any(mask):
                centers.append(xs[mask].mean())
        if len(centers) < 2:
            continue

        centers = np.sort(np.array(centers, dtype=np.float32))

        band_min = w * 0.20
        band_max = w * 0.80
        centers = centers[(centers > band_min) & (centers < band_max)]
        if len(centers) < 2:
            continue

        diffs = np.diff(centers)
        diffs = diffs[(diffs > 40) & (diffs < 180)]
        if diffs.size == 0:
            continue

        dx_pix = float(np.median(diffs))
        ys_list.append(float(y))
        dx_list.append(dx_pix)

    if not ys_list:
        return np.array([]), np.array([])

    ys_arr = np.array(ys_list, dtype=np.float32)
    dx_arr = np.array(dx_list, dtype=np.float32)
    return ys_arr, dx_arr


def estimate_pitch_from_lane_width(
    ll_mask: np.ndarray,
    cam_cfg: CameraConfig,
    lane_width_m: float = 3.6,
    theta_min_deg: float = 0.0,
    theta_max_deg: float = 15.0,
    theta_step_deg: float = 0.25,
) -> float:
    """
    利用“车道宽 ≈ lane_width_m”自动拟合一个俯仰角 pitch_deg（向下为正）。
    ll_mask 应为“中心线（thin）+ 已过滤护栏”的版本。
    """
    ys_arr, dx_obs = collect_lane_width_samples(ll_mask, cam_cfg)
    if ys_arr.size == 0:
        print("[Pitch] 没有足够的车道宽样本，保持原始 pitch。")
        return cam_cfg.pitch_deg

    fx = cam_cfg.fx
    fy = cam_cfg.fy
    cy = cam_cfg.cy
    H = cam_cfg.cam_height

    best_theta = cam_cfg.pitch_deg
    best_cost = float("inf")

    thetas_deg = np.arange(theta_min_deg, theta_max_deg + 1e-6, theta_step_deg)
    for theta_deg in thetas_deg:
        theta_rad = math.radians(theta_deg)

        Yc = (ys_arr - cy) / fy + math.tan(theta_rad)
        if np.any(Yc <= 1e-4):
            continue

        Z = H / Yc
        dx_pred = fx * lane_width_m / Z

        diff = np.abs(dx_pred - dx_obs)
        cost = np.median(diff)

        if cost < best_cost:
            best_cost = cost
            best_theta = theta_deg

    print(
        f"[Pitch] 自动拟合 pitch: {best_theta:.2f} deg "
        f"(median |dx_pred-dx_obs| = {best_cost:.2f} px, "
        f"samples={ys_arr.size})"
    )
    if best_cost > 80.0:
        print(f"[Pitch] 最佳残差仍然很大 ({best_cost:.1f} px)，认为拟合不可靠，保持原始 pitch={cam_cfg.pitch_deg:.2f}°。")
        return cam_cfg.pitch_deg

    return float(best_theta)


def estimate_effective_height_from_lane_width(
    ll_mask: np.ndarray,
    cam_cfg: CameraConfig,
    lane_width_m: float = 3.6,
    row_ratio: float = 0.7,
    window: int = 5
) -> float:
    """
    利用车道宽反推 cam_height（可选）。
    """
    h, w = ll_mask.shape
    y0 = int(h * row_ratio)
    ys = range(max(0, y0 - window), min(h, y0 + window + 1))

    fx = cam_cfg.fx
    fy = cam_cfg.fy
    cx = cam_cfg.cx
    cy = cam_cfg.cy

    lanes_dx = []

    for y in ys:
        xs = np.where(ll_mask[y] > 0)[0]
        if len(xs) < 2:
            continue

        K = min(4, len(xs))
        labels = simple_kmeans_1d(xs.astype(np.float32), K=K)
        centers = []
        for k in range(K):
            mask = labels == k
            if np.any(mask):
                centers.append(xs[mask].mean())
        if len(centers) < 2:
            continue

        centers = np.sort(np.array(centers, dtype=np.float32))
        diffs = np.diff(centers)
        diffs = diffs[(diffs > 5) & (diffs < w * 0.8)]
        if diffs.size == 0:
            continue

        dx_pix = float(np.min(diffs))
        lanes_dx.append((y, dx_pix))

    if not lanes_dx:
        print("[Calib] 未找到可靠的车道宽信息，保留原始高度。")
        return cam_cfg.cam_height

    ys_valid = np.array([item[0] for item in lanes_dx], dtype=np.float32)
    dxs_pix = np.array([item[1] for item in lanes_dx], dtype=np.float32)

    y_med = float(np.median(ys_valid))
    dx_med = float(np.median(dxs_pix))

    Yc = (y_med - cy) / fy
    if Yc <= 0:
        print("[Calib] 估计行在地平线以上，无法标定，保留原始高度。")
        return cam_cfg.cam_height

    H_eff = lane_width_m * Yc * fx / dx_med
    print(f"[Calib] 使用车道宽 {lane_width_m}m 标定得到有效高度 H_eff = {H_eff:.3f} m "
          f"(y_med={y_med:.1f}, dx_med={dx_med:.1f} px)")
    return H_eff


# ============================================================
# 检测 + 距离估计
# ============================================================

def estimate_distance_from_depth(
    depth_map: np.ndarray,
    box_xyxy: np.ndarray
) -> Optional[float]:
    h, w = depth_map.shape[:2]
    x1, y1, x2, y2 = box_xyxy
    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

    x1 = max(0, min(x1, w - 1))
    x2 = max(0, min(x2, w - 1))
    y1 = max(0, min(y1, h - 1))
    y2 = max(0, min(y2, h - 1))

    if x2 <= x1 or y2 <= y1:
        return None

    band_top = max(y1, y2 - 10)
    roi = depth_map[band_top:y2, x1:x2]
    if roi.size == 0:
        return None

    d = float(np.median(roi))
    if not np.isfinite(d) or d <= 0:
        return None

    return d


def infer_detection_with_distance(
    yolo_model: YOLO,
    img_bgr: np.ndarray,
    depth_map: Optional[np.ndarray] = None,
    device: str = "cuda"
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    res = yolo_model(img_bgr, imgsz=640, conf=0.25, iou=0.45)[0]

    det_img = img_bgr.copy()
    names = res.names

    if res.boxes is None or len(res.boxes) == 0:
        return det_img, None, None

    boxes = res.boxes.xyxy.cpu().numpy()
    cls_ids = res.boxes.cls.cpu().numpy().astype(int)
    confs = res.boxes.conf.cpu().numpy()

    for box, cls_id, conf in zip(boxes, cls_ids, confs):
        x1, y1, x2, y2 = box
        x1i, y1i, x2i, y2i = map(int, [x1, y1, x2, y2])

        cv2.rectangle(det_img, (x1i, y1i), (x2i, y2i), (0, 255, 255), 2)

        label = f"{names.get(cls_id, str(cls_id))} {conf:.2f}"

        if depth_map is not None:
            dist = estimate_distance_from_depth(depth_map, box)
            if dist is not None:
                label += f" | {dist:.1f}m"

        txt_x = x1i
        txt_y = max(0, y1i - 5)
        cv2.putText(det_img, label, (txt_x, txt_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(det_img, label, (txt_x, txt_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 255), 1, cv2.LINE_AA)

    return det_img, boxes, cls_ids


# ============================================================
# BEV 相关：车道线投影 + 网格 + 检测点
# ============================================================

def create_bev_canvas(bev_cfg: BEVConfig) -> np.ndarray:
    bev = np.zeros((bev_cfg.height, bev_cfg.width, 3), dtype=np.uint8)
    return bev


def draw_bev_grid(bev: np.ndarray, bev_cfg: BEVConfig):
    h, w = bev.shape[:2]
    origin_x = w // 2
    origin_y = h - 1

    cv2.line(bev, (origin_x, 0), (origin_x, h - 1), (80, 80, 80), 1)

    z = 0.0
    while z <= bev_cfg.max_range_m + 1e-6:
        v = int(origin_y - z * bev_cfg.scale)
        if v < 0:
            break
        color = (60, 60, 60)
        thickness = 1
        if abs(z % bev_cfg.grid_step_big) < 1e-3:
            color = (100, 100, 100)
            thickness = 2
            cv2.putText(bev, f"{int(z)}m",
                        (5, max(0, v - 2)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (150, 150, 150), 1, cv2.LINE_AA)
        cv2.line(bev, (0, v), (w - 1, v), color, thickness)
        z += bev_cfg.grid_step_small


def project_point_to_ground(
    x: float,
    y: float,
    cam_cfg: CameraConfig
) -> Optional[Tuple[float, float]]:
    """
    像素点 (x, y) → 地面平面上的 (X, Z_forward)。
    """
    fx = cam_cfg.fx
    fy = cam_cfg.fy
    cx = cam_cfg.cx
    cy = cam_cfg.cy
    H = cam_cfg.cam_height
    pitch_rad = math.radians(cam_cfg.pitch_deg)

    Yc = (y - cy) / fy + math.tan(pitch_rad)

    if Yc <= 0.0:
        return None

    Z = H / Yc
    if Z <= 0:
        return None
    if Z > 250.0:
        Z = 250.0

    X = Z * (x - cx) / fx
    return X, Z


def draw_lanes_on_bev(
    bev: np.ndarray,
    ll_mask: np.ndarray,
    cam_cfg: CameraConfig,
    bev_cfg: BEVConfig
):
    bev_h, bev_w = bev.shape[:2]
    origin_x = bev_w // 2
    origin_y = bev_h - 1

    ys, xs = np.where(ll_mask > 0)
    if len(xs) == 0:
        return

    for x, y in zip(xs[::2], ys[::2]):  # 抽样
        res = project_point_to_ground(x, y, cam_cfg)
        if res is None:
            continue
        X, Z = res
        u = int(origin_x + X * bev_cfg.scale)
        v = int(origin_y - Z * bev_cfg.scale)
        if 0 <= u < bev_w and 0 <= v < bev_h:
            bev[v, u] = (0, 255, 0)


def draw_detections_on_bev(
    bev: np.ndarray,
    boxes: Optional[np.ndarray],
    cls_ids: Optional[np.ndarray],
    cam_cfg: CameraConfig,
    bev_cfg: BEVConfig
):
    if boxes is None or cls_ids is None:
        return

    bev_h, bev_w = bev.shape[:2]
    origin_x = bev_w // 2
    origin_y = bev_h - 1

    for box, cls_id in zip(boxes, cls_ids):
        x1, y1, x2, y2 = box
        xc = 0.5 * (x1 + x2)
        yb = y2

        res = project_point_to_ground(xc, yb, cam_cfg)
        if res is None:
            continue
        X, Z = res
        u = int(origin_x + X * bev_cfg.scale)
        v = int(origin_y - Z * bev_cfg.scale)
        if 0 <= u < bev_w and 0 <= v < bev_h:
            cv2.circle(bev, (u, v), 4, (0, 255, 255), -1)


def build_bev_image(
    ll_mask: np.ndarray,
    boxes: Optional[np.ndarray],
    cls_ids: Optional[np.ndarray],
    cam_cfg: CameraConfig,
    bev_cfg: BEVConfig
) -> np.ndarray:
    bev = create_bev_canvas(bev_cfg)
    draw_bev_grid(bev, bev_cfg)
    draw_lanes_on_bev(bev, ll_mask, cam_cfg, bev_cfg)
    draw_detections_on_bev(bev, boxes, cls_ids, cam_cfg, bev_cfg)
    return bev


# ============================================================
# 3D 车道线可视化
# ============================================================

def lane_mask_to_3d_points(
    ll_mask: np.ndarray,
    cam_cfg: CameraConfig,
    step: int = 2
) -> np.ndarray:
    """
    将车道线 mask 投影到地面 (X, Z_forward) 上，返回 Nx2
    """
    h, w = ll_mask.shape
    ys, xs = np.where(ll_mask > 0)
    if len(xs) == 0:
        return np.zeros((0, 2), dtype=np.float32)

    Xs, Zs = [], []
    for x, y in zip(xs[::step], ys[::step]):
        res = project_point_to_ground(float(x), float(y), cam_cfg)
        if res is None:
            continue
        X, Z = res
        Xs.append(X)
        Zs.append(Z)

    if len(Xs) == 0:
        return np.zeros((0, 2), dtype=np.float32)

    pts = np.stack([np.array(Xs, dtype=np.float32),
                    np.array(Zs, dtype=np.float32)], axis=1)
    return pts


def split_into_lane_curves_centerline(
    pts: np.ndarray,
    K: int = 3,
    min_points_per_lane: int = 30,
    dz: float = 1.0
) -> List[np.ndarray]:
    """
    使用“中心线”方式生成更细的 3D 车道线：
      - 先按 X 方向聚成 K 类（左中右）
      - 每一类在 Z_forward 上按 dz 分 bin，每个 bin 只取一个中心点
    返回 list，每条曲线为 [Ni, 3]，列为 (X, Y_forward, Z=0)
    """
    if pts.shape[0] == 0:
        return []

    xs = pts[:, 0]
    zs = pts[:, 1]

    labels = simple_kmeans_1d(xs, K=K)
    curves = []
    for k in range(K):
        mask = labels == k
        if np.sum(mask) < min_points_per_lane:
            continue
        lane_pts = pts[mask]
        X_all = lane_pts[:, 0]
        Z_all = lane_pts[:, 1]

        z_min, z_max = Z_all.min(), Z_all.max()
        bins = np.arange(z_min, z_max + dz, dz)
        centers_X = []
        centers_Z = []
        for i in range(len(bins) - 1):
            z1, z2 = bins[i], bins[i + 1]
            mask_bin = (Z_all >= z1) & (Z_all < z2)
            if not np.any(mask_bin):
                continue
            centers_X.append(X_all[mask_bin].mean())
            centers_Z.append(0.5 * (z1 + z2))

        if len(centers_Z) < 5:
            continue

        centers_X = np.array(centers_X, dtype=np.float32)
        centers_Z = np.array(centers_Z, dtype=np.float32)
        Y_forward = centers_Z
        Z0 = np.zeros_like(centers_X)
        curve_3d = np.stack([centers_X, Y_forward, Z0], axis=1)
        curves.append(curve_3d)

    return curves


def plot_3d_lanes(
    curves: List[np.ndarray],
    save_path: str,
    x_lim: Tuple[float, float] = (-10, 10),
    y_lim: Tuple[float, float] = (0, 200),
    z_lim: Tuple[float, float] = (-2, 2)
):
    if len(curves) == 0:
        print("[3D] 没有足够的车道点，跳过 3D 可视化图的保存。")
        return

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")

    colors = ["green", "blue", "orange", "purple", "cyan"]
    for i, curve in enumerate(curves):
        X = curve[:, 0]
        Y = curve[:, 1]
        Z = curve[:, 2]
        c = colors[i % len(colors)]
        ax.plot(X, Y, Z, color=c, linewidth=2)

    ax.set_xlabel("x-axis")
    ax.set_ylabel("y-axis")
    ax.set_zlabel("z-axis")

    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.set_zlim(z_lim)

    ax.view_init(elev=25, azim=-60)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close(fig)
    print("[3D] 3D lane figure saved:", save_path)


# ============================================================
# 主流程
# ============================================================

def run_full_pipeline(img_path: str):
    ensure_dirs()
    device = cfg.device
    print("ROOT   :", ROOT)
    print("Device :", device)
    print("Image  :", img_path)

    if not os.path.isfile(img_path):
        raise FileNotFoundError(img_path)

    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        raise RuntimeError(f"无法读取图像: {img_path}")
    h0, w0 = img_bgr.shape[:2]
    print(f"Input size: {w0}x{h0}")

    # 根据图像宽度与给定 FOV 更新 fx, fy, cx, cy
    cfg.camera.cx = w0 / 2.0
    cfg.camera.cy = h0 / 2.0
    fx_fy = w0 / (2.0 * math.tan(math.radians(cfg.camera.fov_deg) / 2.0))
    cfg.camera.fx = fx_fy
    cfg.camera.fy = fx_fy
    print(f"[Camera] fx=fy={fx_fy:.2f}, cx={cfg.camera.cx:.1f}, cy={cfg.camera.cy:.1f}, H0={cfg.camera.cam_height}")

    # 1) 分割：分别加载 DA-best 与 LaneGeo-best
    print("[Step 1] TwinLiteNetPP 分割 ...")
    seg_model_da = load_seg_model(cfg.seg_weight_da, device=device)
    seg_model_lane = load_seg_model(cfg.seg_weight_lane, device=device)

    # 已有：
    da_mask, _ = infer_da_ll_masks(seg_model_da, img_bgr, device=device)
    _, ll_mask_lane = infer_da_ll_masks(seg_model_lane, img_bgr, device=device)

    # === Debug: 把四种组合都各自保存一张 ===
    da_da, ll_da = infer_da_ll_masks(seg_model_da, img_bgr, device=device)
    da_lane, ll_lane = infer_da_ll_masks(seg_model_lane, img_bgr, device=device)

    debug_dir = cfg.out_seg
    os.makedirs(debug_dir, exist_ok=True)
    cv2.imwrite(os.path.join(debug_dir, "da_from_da.png"), da_da * 255)
    cv2.imwrite(os.path.join(debug_dir, "ll_from_da.png"), ll_da * 255)
    cv2.imwrite(os.path.join(debug_dir, "da_from_lanegeo.png"), da_lane * 255)
    cv2.imwrite(os.path.join(debug_dir, "ll_from_lanegeo.png"), ll_lane * 255)


    # 1.1 只用 DA-best 模型获得可行驶区域 mask
    da_mask, _ = infer_da_ll_masks(
    seg_model_da, img_bgr, device=device,
    da_thr=0.35, ll_thr=0.5, postproc_da=True
    )


    # 1.2 用 LaneGeo-best 模型获得车道线 mask
    _, ll_mask_lane = infer_da_ll_masks(
    seg_model_lane, img_bgr, device=device,
    da_thr=0.35, ll_thr=0.5, postproc_da=False
    )

    # 1.3 叠加可视化：DA 用 da_mask，Lane 用 lanegeo 的 ll_mask_lane
    seg_overlay = build_seg_overlay(img_bgr, da_mask, ll_mask_lane)

    # 保存原始车道线 mask（lanegeo）
    raw_ll_path = os.path.join(cfg.out_fusion, "ll_mask_raw.png")
    cv2.imwrite(raw_ll_path, (ll_mask_lane * 255).astype(np.uint8))
    print(f"[Debug] 原始 ll_mask_lane 已保存: {raw_ll_path}")

    # ==== 1.x 车道线中心线提取 + 护栏过滤 → ll_mask_pitch ====
    ll_mask_pitch = prepare_lane_mask_for_pitch(ll_mask_lane, save_debug=True)

    if np.count_nonzero(ll_mask_pitch) < 50:
        print("[Pitch] ll_mask_pitch 几乎为空，改用 lanegeo 原始 ll_mask_lane 进行 pitch 拟合。")
        ll_mask_pitch = (ll_mask_lane > 0).astype(np.uint8) * 255
        cv2.imwrite(os.path.join(cfg.out_fusion, "ll_pitch_fallback.png"), ll_mask_pitch)

    cfg.camera.pitch_deg = 5.0  # 或 4.0, 5.0 试试看
    print(f"[Camera] 手动设置 pitch_deg = {cfg.camera.pitch_deg:.2f}°")
    # 1.x 利用 3.6m 车道宽自动拟合俯仰角
#    cfg.camera.pitch_deg = estimate_pitch_from_lane_width(
#        ll_mask_pitch, cfg.camera, lane_width_m=3.6,
#        theta_min_deg=0.0, theta_max_deg=15.0, theta_step_deg=0.25
#    )
#    print(f"[Camera] 拟合得到 pitch_deg = {cfg.camera.pitch_deg:.2f}°")

    if USE_LANE_WIDTH_CALIB:
        cfg.camera.cam_height = estimate_effective_height_from_lane_width(
            ll_mask_pitch, cfg.camera, lane_width_m=3.6
        )
        print(f"[Camera] After lane-width calib, H={cfg.camera.cam_height:.3f} m")

    # 2) 几何深度（仍然使用 DA mask）
    print("[Step 2] 几何深度 D_geo ...")
    depth_geo = compute_geo_depth_from_seg(da_mask, ll_mask_pitch, cfg.camera)

    # 3) 视觉深度
    print("[Step 3] DepthAnything 相对深度 D_rel ...")
    depth_model = load_depth_model(device=device)
    depth_rel, depth_vis_rel = infer_depth(depth_model, img_bgr)

    # 4) 融合
    print("[Step 4] 几何流 + 视觉流 融合 → D_abs ...")
    depth_abs, scale_s = fuse_geo_and_visual_depth(depth_geo, depth_rel)
    depth_vis_abs = vis_depth(depth_abs)

    # 5) YOLO 检测 + 距离估计
    print("[Step 5] YOLOv11n 检测 + 距离估计 ...")
    yolo_model = load_yolo_model(cfg.yolo_weight)
    det_vis, boxes, cls_ids = infer_detection_with_distance(
        yolo_model, img_bgr, depth_map=depth_abs, device=device
    )

    # 6) BEV：使用 lanegeo 的 ll_mask_lane 投影
    print("[Step 6] 生成 BEV 俯视图 ...")
    bev_img = build_bev_image(ll_mask_lane, boxes, cls_ids, cfg.camera, cfg.bev)

    # 7) 3D 车道线：同样使用 lanegeo 的 ll_mask_lane
    print("[Step 7] 生成 3D 车道线可视化 ...")
    lane_pts_2d = lane_mask_to_3d_points(ll_mask_lane, cfg.camera, step=2)
    lane_curves_3d = split_into_lane_curves_centerline(
        lane_pts_2d, K=3, min_points_per_lane=30, dz=1.0
    )

    base_name = os.path.splitext(os.path.basename(img_path))[0]
    lane3d_out_path = os.path.join(cfg.out_fusion, f"{base_name}_3d_lanes.png")
    plot_3d_lanes(lane_curves_3d, lane3d_out_path,
                  x_lim=(-10, 10), y_lim=(0, 100), z_lim=(-2, 2))

    # 8) 6 视图拼接（3x2，不拉伸，黑底）
    fusion_out_path = os.path.join(cfg.out_fusion, f"{base_name}_full_6view.png")

    lane3d_img = cv2.imread(lane3d_out_path)
    if lane3d_img is None:
        lane3d_img = np.zeros_like(img_bgr)

    img_list = [
        img_bgr,         # 原图
        depth_vis_abs,   # 深度图（融合）
        seg_overlay,     # 分割叠加（DA+LaneGeo）
        det_vis,         # 检测+距离
        bev_img,         # BEV
        lane3d_img       # 3D 车道线
    ]

    H, W = img_bgr.shape[:2]
    cell_w, cell_h = W, H
    canvas = np.zeros((cell_h * 2, cell_w * 3, 3), dtype=np.uint8)

    positions = [
        (0, 0),
        (0, cell_w),
        (0, cell_w * 2),
        (cell_h, 0),
        (cell_h, cell_w),
        (cell_h, cell_w * 2)
    ]

    for (y, x), im in zip(positions, img_list):
        h, w = im.shape[:2]

        scale = min(cell_w / w, cell_h / h, 1.0)
        if scale < 1.0:
            new_w = int(w * scale)
            new_h = int(h * scale)
            im_resized = cv2.resize(im, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            im_resized = im

        h2, w2 = im_resized.shape[:2]
        offset_y = y + (cell_h - h2) // 2
        offset_x = x + (cell_w - w2) // 2

        canvas[offset_y:offset_y + h2, offset_x:offset_x + w2] = im_resized

    cv2.imwrite(fusion_out_path, canvas)

    print("============================================")
    print("Full 6-view image saved:", fusion_out_path)
    print("3D lane image saved:", lane3d_out_path)
    print("使用的尺度因子 s =", scale_s)
    print("============================================")


# ============================================================
# 命令行入口
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="完整全系统（分割 + 几何流 + 视觉流 + 融合深度 + 检测 + 距离 + BEV + 3D车道线 + 6视图拼接）单图推理脚本（DA/Lane 分权重版）"
    )
    parser.add_argument(
        "--img",
        type=str,
        default=cfg.default_img,
        help="输入图像路径（默认: assets/images/test.jpg）",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_full_pipeline(args.img)
