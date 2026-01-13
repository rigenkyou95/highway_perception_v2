import os
import json
import math
import argparse
from dataclasses import dataclass
from typing import Optional, Tuple, List

import cv2
import numpy as np


# =========================
# Camera params (from your calibration log)
# =========================
K_FALLBACK = np.array([
    [1450.90386, 0.0, 944.660261],
    [0.0, 1453.98959, 529.630984],
    [0.0, 0.0, 1.0],
], dtype=np.float64)

D_FALLBACK = np.array([-0.56737988, 0.44540633, 0.01506728, 0.02283647, -0.15235153], dtype=np.float64)


@dataclass
class Estimate:
    roll_deg: float
    yaw_deg: float
    q: float
    x_vp: float
    y_vp: float
    rmseL: float
    rmseR: float
    inL: int
    inR: int
    segs: int


# -------------------------
# Utils
# -------------------------
def mad_based_filter(x: np.ndarray, z_thresh: float = 3.5) -> np.ndarray:
    """Inlier mask using MAD (robust)."""
    if x.size < 10:
        return np.ones_like(x, dtype=bool)
    med = np.median(x)
    mad = np.median(np.abs(x - med)) + 1e-9
    z = 0.6745 * (x - med) / mad
    return np.abs(z) < z_thresh


def weighted_stats(x: np.ndarray, w: np.ndarray):
    """Weighted mean & weighted median."""
    if x.size == 0:
        return None, None
    w = np.clip(w, 1e-9, None)
    mean = float(np.sum(w * x) / np.sum(w))

    order = np.argsort(x)
    xs = x[order]
    ws = w[order]
    cdf = np.cumsum(ws) / np.sum(ws)
    median = float(xs[np.searchsorted(cdf, 0.5)])
    return mean, median


def undistort_prepare(frame_w: int, frame_h: int, K: np.ndarray, D: np.ndarray, alpha: float = 0.0):
    newK, _ = cv2.getOptimalNewCameraMatrix(K, D, (frame_w, frame_h), alpha, (frame_w, frame_h))
    map1, map2 = cv2.initUndistortRectifyMap(K, D, None, newK, (frame_w, frame_h), cv2.CV_16SC2)
    return newK, map1, map2


def apply_undistort(frame: np.ndarray, map1, map2) -> np.ndarray:
    return cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR)


def hough_segments(edges: np.ndarray,
                   rho: float = 1,
                   theta: float = np.pi / 180,
                   threshold: int = 40,
                   min_line_len: int = 60,
                   max_line_gap: int = 40) -> List[Tuple[int, int, int, int]]:
    lines = cv2.HoughLinesP(edges, rho, theta, threshold,
                            minLineLength=min_line_len, maxLineGap=max_line_gap)
    if lines is None:
        return []
    return [tuple(l[0].tolist()) for l in lines]


def put_text(img, text, y=30, color=(255, 255, 255)):
    cv2.putText(img, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    return img


def draw_segs(img, segs, color=(0, 255, 255), thickness=2, max_draw=120):
    out = img.copy()
    for i, (x1, y1, x2, y2) in enumerate(segs[:max_draw]):
        cv2.line(out, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
    return out


# -------------------------
# RANSAC fit: x = a*y + b
# -------------------------
def fit_x_ay_b_ransac(points_xy: np.ndarray,
                      iters: int = 250,
                      thresh_px: float = 6.0,
                      min_inliers: int = 80,
                      seed: int = 0):
    """
    RANSAC fit for x = a*y + b
    Returns (a, b, rmse, inlier_mask) or None
    """
    if points_xy.shape[0] < 30:
        return None

    x = points_xy[:, 0].astype(np.float64)
    y = points_xy[:, 1].astype(np.float64)
    n = points_xy.shape[0]

    rng = np.random.default_rng(seed)

    best_count = 0
    best_inliers = None
    best_a = None
    best_b = None

    for _ in range(iters):
        i1, i2 = rng.integers(0, n, size=2)
        if i1 == i2:
            continue
        y1, y2 = y[i1], y[i2]
        x1, x2 = x[i1], x[i2]
        if abs(y2 - y1) < 1e-6:
            continue

        a = (x2 - x1) / (y2 - y1)
        b = x1 - a * y1

        resid = np.abs(x - (a * y + b))
        inliers = resid < thresh_px
        cnt = int(np.sum(inliers))

        if cnt > best_count:
            best_count = cnt
            best_inliers = inliers
            best_a, best_b = a, b

    if best_inliers is None or best_count < min_inliers:
        return None

    # refine with LS on inliers
    xi = x[best_inliers]
    yi = y[best_inliers]
    A = np.stack([yi, np.ones_like(yi)], axis=1)
    sol, _, _, _ = np.linalg.lstsq(A, xi, rcond=None)
    a_ref, b_ref = sol[0], sol[1]

    x_pred = a_ref * yi + b_ref
    rmse = float(np.sqrt(np.mean((xi - x_pred) ** 2)))

    return float(a_ref), float(b_ref), rmse, best_inliers


# -------------------------
# Frame estimation
# -------------------------
def estimate_from_frame(frame_bgr: np.ndarray,
                        newK: np.ndarray,
                        *,
                        roi_y0_ratio: float = 0.40,
                        canny1: int = 30,
                        canny2: int = 110,
                        # 改动2：靠近路面过滤阈值（端点必须足够低）
                        ground_y_ratio: float = 0.65,
                        # 线段角度过滤：像车道线
                        ang_min1: float = 25.0,
                        ang_max1: float = 80.0,
                        # RANSAC
                        ransac_iters: int = 250,
                        ransac_thresh_px: float = 6.0,
                        ransac_min_inliers: int = 80,
                        # 每条线段采样点数（夜间很关键）
                        sample_pts_per_seg: int = 15,
                        # Debug
                        return_debug: bool = False):
    H, W = frame_bgr.shape[:2]
    fx = float(newK[0, 0])
    cx = float(newK[0, 2])

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, canny1, canny2)

    roi_y0 = int(roi_y0_ratio * H)
    mask = np.zeros_like(edges)
    mask[roi_y0:H, :] = 255
    edges_roi = cv2.bitwise_and(edges, mask)

    segs = hough_segments(edges_roi, threshold=40, min_line_len=60, max_line_gap=40)
    seg_count = len(segs)

    debug_pack = None
    if return_debug:
        debug_pack = {"edges": edges, "edges_roi": edges_roi, "segs": segs, "roi_y0": roi_y0}

    if seg_count < 5:
        return None, debug_pack

    left_pts_list = []
    right_pts_list = []

    # ----------- collect lane-like points -----------
    for (x1, y1, x2, y2) in segs:
        # 改动2：只保留靠近路面的线段（至少一个端点在下方）
        if max(y1, y2) < ground_y_ratio * H:
            continue

        dx = x2 - x1
        dy = y2 - y1

        # angle filter: reject near-horizontal
        ang = abs(math.degrees(math.atan2(dy, dx)))  # 0..180
        # keep steep-ish segments (two symmetric ranges)
        if not (ang_min1 < ang < ang_max1 or (180 - ang_max1) < ang < (180 - ang_min1)):
            continue

        xm = 0.5 * (x1 + x2)
        # optional: reject extreme borders (car hood/window)
        if xm < 0.05 * W or xm > 0.95 * W:
            continue

        # sample points along segment (IMPORTANT)
        N = max(2, int(sample_pts_per_seg))
        xs = np.linspace(x1, x2, N)
        ys = np.linspace(y1, y2, N)
        pts = np.stack([xs, ys], axis=1).astype(np.float32)

        if xm < cx:
            left_pts_list.append(pts)
        else:
            right_pts_list.append(pts)

    if len(left_pts_list) == 0 or len(right_pts_list) == 0:
        return None, debug_pack

    left_pts = np.concatenate(left_pts_list, axis=0)
    right_pts = np.concatenate(right_pts_list, axis=0)

    # ----------- RANSAC fit (改动4) -----------
    fitL = fit_x_ay_b_ransac(left_pts, iters=ransac_iters, thresh_px=ransac_thresh_px,
                             min_inliers=ransac_min_inliers, seed=0)
    fitR = fit_x_ay_b_ransac(right_pts, iters=ransac_iters, thresh_px=ransac_thresh_px,
                             min_inliers=ransac_min_inliers, seed=1)

    if fitL is None or fitR is None:
        return None, debug_pack

    aL, bL, rmseL, inL = fitL
    aR, bR, rmseR, inR = fitR

    denom = (aL - aR)
    if abs(denom) < 1e-6:
        return None, debug_pack

    # Vanishing point
    y_vp = (bR - bL) / denom
    x_vp = aL * y_vp + bL

    # Yaw from VP x offset
    yaw_deg = math.degrees(math.atan((x_vp - cx) / (fx + 1e-9)))

    # Roll: keep simple (optional). If no reliable horizontal structure -> 0 with lower q.
    # Here we use near-horizontal in top 65% of image.
    edges_top = edges.copy()
    edges_top[int(0.65 * H):, :] = 0
    segs_top = hough_segments(edges_top, threshold=35, min_line_len=60, max_line_gap=30)
    angles = []
    for (x1, y1, x2, y2) in segs_top:
        dx = x2 - x1
        dy = y2 - y1
        if abs(dx) < 1e-6:
            continue
        slope = dy / (dx + 1e-9)
        if abs(slope) > 0.25:
            continue
        ang2 = math.degrees(math.atan2(dy, dx))
        angles.append(ang2)

    if len(angles) >= 5:
        roll_deg = float(np.median(np.array(angles, dtype=np.float64)))
        roll_valid = True
    else:
        roll_deg = 0.0
        roll_valid = False

    # Quality q
    q = 1.0
    # rmse penalty
    q *= float(np.clip(1.0 - (rmseL + rmseR) / 60.0, 0.0, 1.0))
    # VP plausibility: keep if not totally crazy
    if not (-0.5 * W <= x_vp <= 1.5 * W):
        q *= 0.2
    # roll evidence
    if not roll_valid:
        q *= 0.7
    # inlier ratio preference
    inlier_strength = min(inL.sum() / max(1, left_pts.shape[0]), inR.sum() / max(1, right_pts.shape[0]))
    q *= float(np.clip(inlier_strength * 3.0, 0.0, 1.0))

    est = Estimate(
        roll_deg=float(roll_deg),
        yaw_deg=float(yaw_deg),
        q=float(q),
        x_vp=float(x_vp),
        y_vp=float(y_vp),
        rmseL=float(rmseL),
        rmseR=float(rmseR),
        inL=int(inL.sum()),
        inR=int(inR.sum()),
        segs=int(seg_count),
    )
    return est, debug_pack


# -------------------------
# Video processing
# -------------------------
def save_debug_collage(out_path: str,
                       und: np.ndarray,
                       debug_pack,
                       est: Optional[Estimate],
                       reason: str):
    H, W = und.shape[:2]
    edges = debug_pack["edges"]
    edges_roi = debug_pack["edges_roi"]
    segs = debug_pack["segs"]
    roi_y0 = debug_pack["roi_y0"]

    vis = und.copy()
    cv2.line(vis, (0, roi_y0), (W - 1, roi_y0), (255, 0, 0), 2)
    vis = draw_segs(vis, segs, color=(0, 255, 255), thickness=2, max_draw=150)

    if est is not None:
        cv2.circle(vis, (int(est.x_vp), int(est.y_vp)), 8, (0, 0, 255), -1)
        put_text(vis, f"VP=({est.x_vp:.1f},{est.y_vp:.1f})", y=60)
        put_text(vis, f"yaw_deg={est.yaw_deg:.3f}", y=90, color=(0, 255, 0))
        put_text(vis, f"roll_deg={est.roll_deg:.3f}", y=120, color=(0, 255, 0))
        put_text(vis, f"rmseL={est.rmseL:.1f}, rmseR={est.rmseR:.1f}, inL={est.inL}, inR={est.inR}, segs={est.segs}", y=150, color=(255, 255, 0))
    else:
        put_text(vis, "yaw_deg=NA", y=90, color=(0, 0, 255))
        put_text(vis, "roll_deg=NA", y=120, color=(0, 0, 255))

    put_text(vis, reason, y=180, color=(255, 255, 0))

    edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    edges_roi_bgr = cv2.cvtColor(edges_roi, cv2.COLOR_GRAY2BGR)

    vis_small = cv2.resize(vis, (W // 2, H // 2))
    edges_small = cv2.resize(edges_bgr, (W // 2, H // 2))
    roi_small = cv2.resize(edges_roi_bgr, (W // 2, H // 2))
    blank = np.zeros_like(roi_small)

    top = np.hstack([vis_small, edges_small])
    bot = np.hstack([roi_small, blank])
    collage = np.vstack([top, bot])

    cv2.imwrite(out_path, collage)


def process_video(video_path: str,
                  K: np.ndarray,
                  D: np.ndarray,
                  out_dir: str,
                  *,
                  sample_every: int = 5,
                  max_frames: int = 0,
                  alpha: float = 0.0,
                  save_debug: bool = True,
                  debug_every: int = 20,
                  debug_max: int = 200):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    newK, map1, map2 = undistort_prepare(W, H, K, D, alpha=alpha)

    os.makedirs(out_dir, exist_ok=True)
    debug_dir = os.path.join(out_dir, "debug")
    if save_debug:
        os.makedirs(debug_dir, exist_ok=True)

    rolls, yaws, qs = [], [], []
    debug_saved = 0

    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if idx % sample_every != 0:
            idx += 1
            continue

        und = apply_undistort(frame, map1, map2)

        est, dbg = estimate_from_frame(
            und, newK,
            roi_y0_ratio=0.40,
            canny1=30, canny2=110,
            ground_y_ratio=0.65,      # 改动2：靠近路面
            ransac_iters=250,         # 改动4：RANSAC
            ransac_thresh_px=6.0,
            ransac_min_inliers=80,
            sample_pts_per_seg=15,
            return_debug=save_debug
        )

        reason = ""
        if est is None:
            reason = "FAIL: insufficient/dirty lane evidence (after ground+angle filters or RANSAC fail)"
        else:
            # basic gate
            if est.q >= 0.25:
                rolls.append(est.roll_deg)
                yaws.append(est.yaw_deg)
                qs.append(est.q)
                reason = f"OK q={est.q:.2f}"
            else:
                reason = f"LOWQ q={est.q:.2f}"

        # save debug
        if save_debug and dbg is not None and (idx // sample_every) % debug_every == 0 and debug_saved < debug_max:
            out_path = os.path.join(debug_dir, f"debug_{debug_saved:05d}_frame_{idx:07d}.jpg")
            save_debug_collage(out_path, und, dbg, est, reason)
            debug_saved += 1

        idx += 1
        if max_frames > 0 and idx >= max_frames:
            break

    cap.release()

    rolls = np.array(rolls, dtype=np.float64)
    yaws = np.array(yaws, dtype=np.float64)
    qs = np.array(qs, dtype=np.float64)

    return rolls, yaws, qs, {"W": W, "H": H, "fps": fps, "debug_saved": debug_saved}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos", nargs="+", default=[
        r"E:\detected sys\highway_perception_v2\calib_videos\MP4\D1.MP4",
        r"E:\detected sys\highway_perception_v2\calib_videos\MP4\251215115034.mp4",
    ])
    parser.add_argument("--out_dir", type=str,
                        default=r"E:\detected sys\highway_perception_v2\outputs\camera_extrinsic_calib\roll_yaw_video")
    parser.add_argument("--sample_every", type=int, default=5)
    parser.add_argument("--max_frames", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=0.0)

    parser.add_argument("--save_debug", action="store_true", help="save debug collages")
    parser.add_argument("--debug_every", type=int, default=20, help="save one debug every N sampled frames")
    parser.add_argument("--debug_max", type=int, default=200, help="max debug images per video")

    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    K = K_FALLBACK.copy()
    D = D_FALLBACK.copy()

    all_roll, all_yaw, all_q = [], [], []
    per_video_summary = []

    for vp in args.videos:
        name = os.path.splitext(os.path.basename(vp))[0]
        out_dir = os.path.join(args.out_dir, name)
        print(f"\n[Video] {vp}")

        rolls, yaws, qs, info = process_video(
            vp, K, D, out_dir,
            sample_every=args.sample_every,
            max_frames=args.max_frames,
            alpha=args.alpha,
            save_debug=args.save_debug,
            debug_every=args.debug_every,
            debug_max=args.debug_max
        )

        if rolls.size == 0:
            print(f"  -> No valid samples. (debug_saved={info['debug_saved']})")
            per_video_summary.append({"video": vp, "num": 0, "debug_saved": info["debug_saved"]})
            continue

        # robust filter
        m = mad_based_filter(rolls) & mad_based_filter(yaws)
        rolls2, yaws2, qs2 = rolls[m], yaws[m], qs[m]

        mean_roll, med_roll = weighted_stats(rolls2, qs2)
        mean_yaw,  med_yaw  = weighted_stats(yaws2, qs2)

        print(f"  size={info['W']}x{info['H']} fps={info['fps']:.2f} debug_saved={info['debug_saved']}")
        print(f"  samples kept: {int(rolls2.size)}")
        print(f"  roll_deg  mean={mean_roll:.4f}, median={med_roll:.4f}")
        print(f"  yaw_deg   mean={mean_yaw:.4f}, median={med_yaw:.4f}")

        # save csv
        csv_path = os.path.join(out_dir, f"{name}_roll_yaw.csv")
        data = np.stack([rolls2, yaws2, qs2], axis=1)
        np.savetxt(csv_path, data, delimiter=",", header="roll_deg,yaw_deg,q", comments="")
        print(f"  saved: {csv_path}")

        per_video_summary.append({
            "video": vp,
            "num": int(rolls2.size),
            "roll_mean": mean_roll,
            "roll_median": med_roll,
            "yaw_mean": mean_yaw,
            "yaw_median": med_yaw,
            "debug_saved": info["debug_saved"],
            "W": info["W"], "H": info["H"], "fps": info["fps"]
        })

        all_roll.append(rolls2)
        all_yaw.append(yaws2)
        all_q.append(qs2)

    if len(all_roll) == 0:
        print("\n[Global] No valid samples across all videos.")
        return

    all_roll = np.concatenate(all_roll, axis=0) if len(all_roll) else np.array([])
    all_yaw  = np.concatenate(all_yaw, axis=0)  if len(all_yaw)  else np.array([])
    all_q    = np.concatenate(all_q, axis=0)    if len(all_q)    else np.array([])

    if all_roll.size == 0:
        print("\n[Global] No valid samples across all videos.")
        return

    # global robust filter
    m = mad_based_filter(all_roll) & mad_based_filter(all_yaw)
    all_roll, all_yaw, all_q = all_roll[m], all_yaw[m], all_q[m]

    g_mean_roll, g_med_roll = weighted_stats(all_roll, all_q)
    g_mean_yaw,  g_med_yaw  = weighted_stats(all_yaw, all_q)

    summary = {
        "videos": per_video_summary,
        "global": {
            "num": int(all_roll.size),
            "roll_mean": g_mean_roll,
            "roll_median": g_med_roll,
            "yaw_mean": g_mean_yaw,
            "yaw_median": g_med_yaw,
        },
        "settings": {
            "sample_every": args.sample_every,
            "max_frames": args.max_frames,
            "undistort_alpha": args.alpha,
            "save_debug": args.save_debug,
            "debug_every": args.debug_every,
            "debug_max": args.debug_max,
            "filters": {
                "roi_y0_ratio": 0.40,
                "ground_y_ratio": 0.65,
                "angle_deg_keep": "25-80 or 100-155",
                "ransac_iters": 250,
                "ransac_thresh_px": 6.0,
                "ransac_min_inliers": 80,
                "sample_pts_per_seg": 15
            }
        }
    }

    print("\n========== [Global Summary] ==========")
    print(f"samples total: {summary['global']['num']}")
    print(f"ROLL deg: mean={g_mean_roll:.4f}, median={g_med_roll:.4f}")
    print(f"YAW  deg: mean={g_mean_yaw:.4f}, median={g_med_yaw:.4f}")

    json_path = os.path.join(args.out_dir, "roll_yaw_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"saved: {json_path}")


if __name__ == "__main__":
    main()
