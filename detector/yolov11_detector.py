import os
import sys
sys.path.append(os.path.join(os.getcwd(), "third_party", "yolov11n"))

from ultralytics import YOLO
import numpy as np

class YOLOv11Detector:
    def __init__(self, weight_path="yolo11n.pt", device="cuda"):
        self.model = YOLO(weight_path)
        self.device = device

    def infer(self, img):
        """
        img: np.ndarray, BGR (cv2 读取的格式)
        return: bboxes (N,4), scores (N,), class_ids (N,)
        """
        results = self.model(img, device=self.device)[0]
        bboxes = results.boxes.xyxy.cpu().numpy()
        scores = results.boxes.conf.cpu().numpy()
        class_ids = results.boxes.cls.cpu().numpy()
        return bboxes, scores, class_ids
