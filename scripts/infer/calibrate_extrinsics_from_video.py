import os
import sys
import cv2
import math
import argparse
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F

# ============================================================
# 0) 自动定位项目根目录（修复 No module named 'models'）
# ============================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
ROOT = ROOT_DIR
print("[INFO] ROOT =", ROOT)

# ============================================================
# 1) 模型
# ============================================================
from models.seg.twinlitenet_pp.TwinLitePP import TwinLiteNetPP

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 你最新训练的权重位置（按你实际文件名改）
WEIGHT_PATH = r"E:\detected sys\highway_perception_v2\models\ckpts\twinlitenetpp_geodepth\twinlitenetpp_da_best.pth"
assert os.path.exists(WEIGHT_PATH), f"Weight not found: {WEIGHT_PATH}"

model = TwinLiteNetPP()
state = torch.load(WEIGHT_PATH, map_location=device)

# 兼容 checkpoint 结构
if isinstance(state, dict) and "state_dict" in state:
    state = state["state_dict"]

# 兼容 DataParallel 的 module. 前缀
new_state = {}
if isinstance(state, dict):
    for k, v in state.items():
        new_state[k.replace("module.", "")] = v
else:
    new_state = state

model.load_state_dict(new_state, strict=False)
model.to(device).eval()
print("[INFO] TwinLiteNetPP loaded")

# ============================================================
# 2) 相机内参 K
# ============================================================
K_PATH = r"E:\detected sys\highway_perception_v2\calib_videos\camera_K.npy"
K = None

def build_K_from_fov(w, h, fov_deg=140.0):
    cx = w / 2.0
    cy = h / 2.0
    fx = w / (2.0 * math.tan(math.radians(fov_deg) / 2.0))
    fy = fx
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)

def ensure_K(frame):
    global K
    H, W = frame.shape[:2]
    if K is None:
        if os.path.exists(K_PATH):
            K = np.load(K_PATH).astype(np.float64)
            print("[INFO] Loaded K from:", K_PATH)
        else:
            K = build_K_from_fov(W, H, 140.0)
            print("[WARN] camera_K.npy not found, using FOV fallback K (rough).")
    return K

def pitch_from_vp(vp, K):
    cy = K[1, 2]
    fy = K[1, 1]
    return math.atan((vp[1] - cy) / fy)

# ============================================================
# 3) DA 推理（这里只用于 fill 信息，不参与 VP 求解）
# ============================================================
def infer_da_mask(frame_bgr, thr=0.35):
    H0, W0 = frame_bgr.shape[:2]
    img = cv2.resize(frame_bgr, (640, 360))
    img = img[:, :, ::-1].astype(np.float32) / 255.0
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(img)
        if isinstance(out, (tuple, list)) and len(out) == 2:
            da_logits, _ = out
        else:
            da_logits = out

    da_soft = torch.softmax(da_logits, dim=1)[0].detach().cpu().numpy()  # [2,H,W]
    p0, p1 = da_soft[0], da_soft[1]

    y0 = int(0.75 * p0.shape[0])
    m0 = float(p0[y0:, :].mean())
    m1 = float(p1[y0:, :].mean())
    drivable_idx = 0 if m0 > m1 else 1
    prob = da_soft[drivable_idx]

    da = (prob > thr).astype(np.uint8)

    k = np.ones((5, 5), np.uint8)
    da = cv2.morphologyEx(da, cv2.MORPH_CLOSE, k, iterations=2)
    da = cv2.dilate(da, k, iterations=1)

    da = cv2.resize(da, (W0, H0), interpolation=cv2.INTER_NEAREST)
    return da

# ============================================================
# 4) Edge-based VP：Canny + Hough + RANSAC 交点
#    (改动 2：线段长度过滤)
# ============================================================
def extract_lines_from_edges(frame_bgr, min_len=120):
    H, W = frame_bgr.shape[:2]

    # ROI：只看路面区域（下半部分）
    y1, y2 = int(0.45 * H), int(0.92 * H)
    roi = frame_bgr[y1:y2].copy()

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # 去掉畸变黑边/极暗区域
    gray[gray < 8] = 0

    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # 你如果想更敏感：可以把 60/180 改成 40/140
    edges = cv2.Canny(gray, 60, 180)

    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180,
        threshold=60,
        minLineLength=80,
        maxLineGap=25
    )
    if lines is None:
        return []

    segs = []
    for l in lines[:, 0]:
        x1, yy1, x2, yy2 = map(int, l)
        yy1 += y1
        yy2 += y1

        dx = x2 - x1
        dy = yy2 - yy1
        if abs(dx) < 5:
            continue

        slope = dy / dx

        # 过滤近水平/近垂直
        if abs(slope) < 0.25:
            continue
        if abs(slope) > 8.0:
            continue

        # ★ 改动 2：只保留足够长的线段（去掉草丛纹理短线）
        length = math.hypot(dx, dy)
        if length < min_len:
            continue

        segs.append((x1, yy1, x2, yy2, float(slope), float(length)))

    return segs

def intersect_line(l1, l2):
    x1, y1, x2, y2 = l1[0], l1[1], l1[2], l1[3]
    x3, y3, x4, y4 = l2[0], l2[1], l2[2], l2[3]

    A = np.array([
        [y2 - y1, x1 - x2],
        [y4 - y3, x3 - x4]
    ], dtype=np.float64)
    B = np.array([
        x1 * (y2 - y1) + y1 * (x1 - x2),
        x3 * (y4 - y3) + y3 * (x3 - x4)
    ], dtype=np.float64)

    det = np.linalg.det(A)
    if abs(det) < 1e-9:
        return None
    vp = np.linalg.solve(A, B)
    return float(vp[0]), float(vp[1])

def estimate_vp_ransac(segs, W, H, iters=300, seed=123):
    # 分左右（根据 slope 正负）
    left = [s for s in segs if s[4] < -0.25]
    right = [s for s in segs if s[4] > 0.25]
    if len(left) < 2 or len(right) < 2:
        return None, 0

    rng = np.random.default_rng(seed)
    best_vp = None
    best_score = -1

    for _ in range(iters):
        l = left[rng.integers(0, len(left))]
        r = right[rng.integers(0, len(right))]
        vp = intersect_line(l, r)
        if vp is None:
            continue

        x, y = vp

        # ★ 改动 1：更强的物理约束（强烈抑制跳点）
        # VP y 一般在图像上部/接近地平线：限制在 [0.05H, 0.55H]
        if not (0.05 * H <= y <= 0.55 * H):
            continue
        # VP x 不要飞太离谱：限制在 [-0.2W, 1.2W]
        if not (-0.2 * W <= x <= 1.2 * W):
            continue

        score = 0
        # 统计多少线段“指向”该 vp
        for s in segs:
            x1, y1, x2, y2 = s[0], s[1], s[2], s[3]
            v1 = np.array([x2 - x1, y2 - y1], dtype=np.float64)
            v2 = np.array([x - x1, y - y1], dtype=np.float64)
            n1 = np.linalg.norm(v1) + 1e-9
            n2 = np.linalg.norm(v2) + 1e-9
            cos = float(np.dot(v1, v2) / (n1 * n2))
            if cos > 0.985:
                score += 1

        if score > best_score:
            best_score = score
            best_vp = vp

    return best_vp, int(best_score)

# ============================================================
# 5) 主流程 + Robust 统计
# ============================================================
def robust_pitch_stats(pitches_rad):
    deg = np.rad2deg(np.array(pitches_rad, dtype=np.float64))
    if deg.size == 0:
        return None

    med = float(np.median(deg))
    mad = float(np.median(np.abs(deg - med)))  # MAD
    robust_sigma = 1.4826 * mad  # ~sigma

    # 3σ robust 去离群
    thr = 3.0 * (robust_sigma + 1e-6)
    mask = np.abs(deg - med) < thr
    deg_f = deg[mask]

    out = {
        "deg_all": deg,
        "deg_f": deg_f,
        "median": med,
        "mad_sigma": robust_sigma,
        "kept": int(deg_f.size),
        "total": int(deg.size),
        "mean_f": float(deg_f.mean()) if deg_f.size else float("nan"),
        "std_f": float(deg_f.std()) if deg_f.size else float("nan"),
    }
    return out

def process_video(video_path, stride=2, debug_stride=30, min_len=120, score_thr=8):
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), video_path
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_dir = os.path.splitext(video_path)[0] + "_debug"
    os.makedirs(out_dir, exist_ok=True)

    pitches = []
    valid_cnt = 0

    pbar = tqdm(total=total, desc=os.path.basename(video_path))
    idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        pbar.update(1)

        if idx % stride != 0:
            idx += 1
            continue

        K = ensure_K(frame)
        H, W = frame.shape[:2]

        # DA 仅用于 fill 监控
        da = infer_da_mask(frame, thr=0.35)
        fill = float(da.mean())

        segs = extract_lines_from_edges(frame, min_len=min_len)
        vp, score = estimate_vp_ransac(segs, W=W, H=H, iters=300)

        valid = 0
        pitch_deg = float("nan")
        if vp is not None and score >= score_thr:
            pitch = pitch_from_vp(vp, K)
            pitches.append(pitch)
            valid_cnt += 1
            valid = 1
            pitch_deg = float(np.rad2deg(pitch))

        # Debug
        if idx % debug_stride == 0:
            vis = frame.copy()

            # 画线段（最多 250 条，避免太乱）
            for s in segs[:250]:
                x1, y1, x2, y2 = int(s[0]), int(s[1]), int(s[2]), int(s[3])
                cv2.line(vis, (x1, y1), (x2, y2), (0, 255, 255), 2)

            if vp is not None:
                cv2.circle(vis, (int(vp[0]), int(vp[1])), 8, (0, 0, 255), -1)

            cv2.putText(
                vis,
                f"frame={idx} valid={valid} score={score} fill={fill:.3f} pitch={pitch_deg:.2f}",
                (25, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2
            )
            if vp is not None:
                cv2.putText(
                    vis,
                    f"vp=({vp[0]:.1f},{vp[1]:.1f}) min_len={min_len}",
                    (25, 85),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (255, 255, 255),
                    2
                )

            cv2.imwrite(os.path.join(out_dir, f"dbg_{idx:06d}.jpg"), vis)

        idx += 1

    pbar.close()
    cap.release()

    print("========== SUMMARY ==========")
    print(f"Video: {video_path}")
    print(f"Valid frames(raw): {valid_cnt}/{total}")

    stats = robust_pitch_stats(pitches)
    if stats is None:
        print("No valid pitch")
    else:
        print(f"Pitch median: {stats['median']:.3f} deg")
        print(f"Pitch robust-sigma(MAD): {stats['mad_sigma']:.3f} deg")
        print(f"Kept frames(after robust filter): {stats['kept']}/{stats['total']}")
        print(f"Pitch mean(filtered): {stats['mean_f']:.3f} deg")
        print(f"Pitch std (filtered): {stats['std_f']:.3f} deg")

    print("[DBG] saved in:", out_dir)

# ============================================================
# 6) CLI
# ============================================================
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--debug_stride", type=int, default=30)
    ap.add_argument("--min_len", type=int, default=120, help="min Hough segment length (px)")
    ap.add_argument("--score_thr", type=int, default=8, help="RANSAC vp score threshold")
    args = ap.parse_args()

    process_video(
        args.video,
        stride=args.stride,
        debug_stride=args.debug_stride,
        min_len=args.min_len,
        score_thr=args.score_thr
    )
