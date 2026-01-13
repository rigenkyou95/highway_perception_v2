"""
TwinLiteNet++ 模型文件（纯模型定义）
-----------------------------------
结构：
    TwinLiteNetPP = TwinLiteNet(backbone) + refine heads(blocks_pp)

特点：
    - 输出两个语义分割任务：Drivable Area + Lane Line
    - 每个输出为 logits (N, 2, H, W)
    - refine head 通过 CBAM + 纵向卷积增强车道线连续性
    - 对可行驶区域用轻量卷积平滑处理

用法：
    from models.seg.twinlitenet_pp.TwinLitePP import TwinLiteNetPP
    model = TwinLiteNetPP(use_refine=True)
"""

import torch
import torch.nn as nn

# 同目录相对导入
from .TwinLite import TwinLiteNet
from .blocks_pp import LaneRefineHead, DrivableRefineHead


class TwinLiteNetPP(nn.Module):
    """
    TwinLiteNet++:
    ---------------------------------
    基于 TwinLiteNet（多任务双输出），加入 refine heads 提升精度：
        - Drivable refine：轻量平滑卷积
        - Lane refine：纵向卷积 + CBAM，提高车道线连续性
    """

    def __init__(self, p: int = 2, q: int = 3, use_refine: bool = True):
        """
        Args:
            p, q: TwinLiteNet backbone 的 dilation 参数
            use_refine: 是否启用 refine head
        """
        super().__init__()
        self.use_refine = use_refine

        # ===== 主干 Backbone: TwinLiteNet =====
        self.backbone = TwinLiteNet(p=p, q=q)

        # ===== refine heads =====
        if self.use_refine:
            # 可行驶区域 refine
            self.da_refine = DrivableRefineHead(in_ch=2, mid_ch=8)
            # 车道线 refine（更复杂）
            self.ll_refine = LaneRefineHead(in_ch=2, mid_ch=16)

    def forward(self, x):
        """
        Forward:
            输入: x = [N, 3, H, W]
            返回:
                da_logits = [N, 2, H, W]
                ll_logits = [N, 2, H, W]
        """
        # Backbone 输出两个 logits:
        #   da: drivable area
        #   ll: lane line
        da_logits, ll_logits = self.backbone(x)

        # =======================
        # refine heads
        # =======================
        if self.use_refine:
            da_logits = self.da_refine(da_logits)
            ll_logits = self.ll_refine(ll_logits)

        return da_logits, ll_logits


# 方便外部 import
__all__ = ["TwinLiteNetPP"]
