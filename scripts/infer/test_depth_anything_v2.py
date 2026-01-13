import os
import sys
import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from models.depth import DepthAnythingV2TinyWrapper


def vis_depth(depth: np.ndarray) -> np.ndarray:
    d = depth.copy()
    d = d - d.min()
    d = d / (d.max() + 1e-6)
    d = (d * 255).astype(np.uint8)
    d_color = cv2.applyColorMap(d, cv2.COLORMAP_JET)
    return d_color


def main():
    # 1. 权重路径（不变）
    ckpt_path = os.path.join(
        ROOT,
        "models",
        "ckpts",
        "depth_anything_v2",
        "depth_anything_v2_vits.pth",
    )

    # 2. 统一测试输入路径：assets/images/demo_depth.jpg
    img_path = os.path.join(
        ROOT,
        "assets",
        "images",
        "test.jpg",    # ← 这里要和你实际图片文件名一致
    )

    # 3. 统一测试输出路径：outputs/depth_anything_v2/xxx.png
    save_dir = os.path.join(
        ROOT,
        "outputs",
        "depth_anything_v2",
    )
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "demo_depth_vis.png")

    # 4. 加载模型
    model = DepthAnythingV2TinyWrapper(
        ckpt_path=ckpt_path,
        device="cuda",
        input_size=518,
    )

    # 5. 读图
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {img_path}")

    # 6. 推理
    depth = model(img)

    # 7. 可视化 & 保存
    depth_vis = vis_depth(depth)
    concat = np.concatenate([img, depth_vis], axis=1)
    cv2.imwrite(save_path, concat)
    print(f"[OK] Saved depth vis to: {save_path}")


if __name__ == "__main__":
    main()
