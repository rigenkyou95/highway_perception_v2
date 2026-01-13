import os
import sys
from typing import Tuple

import cv2
import torch
import numpy as np
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.append(ROOT)


class DepthAnythingV2TinyWrapper(nn.Module):
    """
    Depth Anything V2 Small (vits) 封装
    输入：BGR 图像 (H, W, 3), np.uint8
    输出：深度图 (H, W), np.float32
    """

    def __init__(
        self,
        ckpt_path: str,
        device: str = "cuda",
        input_size: int = 518,
    ):
        super().__init__()

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.input_size = input_size

        from third_party.depth_anything_v2.depth_anything_v2.dpt import DepthAnythingV2

        # 🚩 关键：按 Small 配置构建模型（和 vits 权重匹配）
        self.net = DepthAnythingV2(
            encoder="vits",
            features=64,
            out_channels=[48, 96, 192, 384],
            use_bn=False,
            use_clstoken=False,
        )

        state = torch.load(ckpt_path, map_location=self.device)
        if "model" in state:      # 有的 checkpoint 外面包了一层
            state = state["model"]

        # 现在形状应该完全匹配，可以 strict=True
        self.net.load_state_dict(state, strict=True)
        self.net.to(self.device)
        self.net.eval()

    @torch.no_grad()
    def forward(self, img_bgr: np.ndarray) -> np.ndarray:
        """
        img_bgr: H x W x 3, np.uint8 (OpenCV)
        return: H x W, np.float32
        """
        # 使用官方 infer_image，里面会负责 resize / normalize / to(device)
        depth = self.net.infer_image(img_bgr, input_size=self.input_size)
        depth = depth.astype(np.float32)
        return depth
