import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------
# 基础注意力：CBAM（Channel + Spatial）
# ----------------------------------------------------------
class ChannelAttention(nn.Module):
    def __init__(self, in_ch, ratio=8):
        super().__init__()
        hidden = max(1, in_ch // ratio)
        self.mlp = nn.Sequential(
            nn.Linear(in_ch, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, in_ch)
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        avg = self.avg_pool(x).view(b, c)
        max_, _ = torch.max(x.view(b, c, -1), dim=2)
        out = self.mlp(avg) + self.mlp(max_)
        out = self.sigmoid(out).view(b, c, 1, 1)
        return x * out


class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        max_, _ = torch.max(x, dim=1, keepdim=True)
        s = torch.cat([avg, max_], dim=1)
        s = self.conv(s)
        s = self.sigmoid(s)
        return x * s


class CBAM(nn.Module):
    def __init__(self, in_ch, ratio=8):
        super().__init__()
        self.ca = ChannelAttention(in_ch, ratio=ratio)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        return x


# ----------------------------------------------------------
# 车道线 refine head：
# - 输入：原始 lane logits（N, 2, H, W）
# - 目标：加强垂直方向连续性 + 注意力增强
# ----------------------------------------------------------
class LaneRefineHead(nn.Module):
    def __init__(self, in_ch=2, mid_ch=16):
        super().__init__()
        self.conv_in = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
        )

        # 垂直方向的卷积（增强纵向连续性）
        self.conv_vert = nn.Sequential(
            nn.Conv2d(mid_ch, mid_ch, kernel_size=(5, 1), padding=(2, 0), bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
        )

        self.cbam = CBAM(mid_ch)

        self.conv_out = nn.Conv2d(mid_ch, in_ch, kernel_size=1)

    def forward(self, x):
        """
        x: lane logits, shape [N, 2, H, W]
        """
        feat = self.conv_in(x)
        feat = self.conv_vert(feat)
        feat = self.cbam(feat)
        out = self.conv_out(feat)
        # 保持 logits 形式，后面再 softmax
        return out


# ----------------------------------------------------------
# Drivable 区域 refine head（轻量一点）
# ----------------------------------------------------------
class DrivableRefineHead(nn.Module):
    def __init__(self, in_ch=2, mid_ch=8):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, in_ch, kernel_size=1)
        )

    def forward(self, x):
        return self.conv(x)
