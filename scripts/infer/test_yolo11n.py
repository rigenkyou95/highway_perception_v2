import os
import sys
import cv2
import time

# -----------------------------
# 统一根目录定位（根据 scripts/infer 路径自动找到项目根目录）
# -----------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.append(ROOT)

# -----------------------------
# 导入 YOLOv11（第三方包）
# -----------------------------
sys.path.append(os.path.join(ROOT, "third_party", "yolov11"))  # 更保险
from ultralytics import YOLO


def main():

    # ============================================================
    # 1. 加载 YOLOv11n 模型
    # ============================================================
    model_path = os.path.join(ROOT, "third_party", "yolov11", "yolo11n.pt")
    print(f"[INFO] Loading YOLOv11n from: {model_path}")
    model = YOLO(model_path)

    # ============================================================
    # 2. 读取测试图片（统一规范：assets/images/）
    # ============================================================
    img_path = os.path.join(ROOT, "assets", "images", "test.jpg")
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"[ERROR] test image not found: {img_path}")

    img = cv2.imread(img_path)

    # ============================================================
    # 3. 推理时间统计
    # ============================================================
    t0 = time.time()
    results = model(img)[0]
    t1 = time.time()

    print(f"[INFO] Inference time: {(t1 - t0) * 1000:.2f} ms")

    # ============================================================
    # 4. 获取检测框
    # ============================================================
    boxes = results.boxes.xyxy.cpu().numpy()
    scores = results.boxes.conf.cpu().numpy()
    class_ids = results.boxes.cls.cpu().numpy()
    print(f"[INFO] Detected objects: {len(boxes)}")

    # ============================================================
    # 5. 可视化检测框
    # ============================================================
    for (x1, y1, x2, y2), score, cls_id in zip(boxes, scores, class_ids):
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)),
                      (0, 255, 0), 2)
        cv2.putText(img,
                    f"{int(cls_id)} {score:.2f}",
                    (int(x1), int(y1) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 1)

    # ============================================================
    # 6. 输出路径（统一规范：outputs/yolov11n/）
    # ============================================================
    out_dir = os.path.join(ROOT, "outputs", "yolov11n")
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, "test_yolo11_vis.jpg")
    cv2.imwrite(out_path, img)

    print(f"[INFO] Saved visualization to: {out_path}")


if __name__ == "__main__":
    main()
