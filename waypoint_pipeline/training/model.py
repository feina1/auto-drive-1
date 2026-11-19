"""模型: 视觉 backbone (DINOv3 冻结特征 / TinyCNN) + waypoint head.

输入:
- 图像特征 [B, feat_dim] (DINOv3) 或图像张量 [B,3,H,W] (TinyCNN)
- proprio [B,2] = [speed/72, steering/0.6]  ← 时间 waypoints 必须知道当前车速

输出: [B, WP_COUNT, 2] ego 系 waypoints (x 前, y 左, 米)

为什么必须输入车速:
    waypoints 是"未来 0.5/1.0/.../3.0 s"的时间采样。同一张图像在 40 km/h 与
    72 km/h 下的 GT 点距差 1.8 倍, 单帧图像无法区分; 车速是必要观测量。
"""
from __future__ import annotations

import torch
import torch.nn as nn


class WaypointHead(nn.Module):
    """特征 + proprio → waypoints. 用于 DINOv3 预提取特征. """

    def __init__(self, feat_dim=384, wp_count=6, hidden_dim=256, prop_dim=2):
        super().__init__()
        in_dim = feat_dim + prop_dim
        self.mlp = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, wp_count * 2),
        )
        self.wp_count = wp_count

    def forward(self, feat, prop):
        x = torch.cat([feat, prop], dim=-1)
        return self.mlp(x).view(-1, self.wp_count, 2)


class TinyCNN(nn.Module):
    """轻量端到端 CNN (仅用于无 GPU / 无 DINOv3 权重时的冒烟测试).

    输入图像 128x72, 4 层 stride-2 卷积 + GAP, 约 0.2M 参数, CPU 可训练。
    它不是交付模型, 只用于验证数据/训练/预测/闭环整条链路可运行。
    """

    def __init__(self, wp_count=6, hidden_dim=256, prop_dim=2, image_size=(128, 72)):
        super().__init__()
        self.image_size = tuple(image_size)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 24, 5, stride=2, padding=2), nn.BatchNorm2d(24), nn.ReLU(inplace=True),
            nn.Conv2d(24, 48, 3, stride=2, padding=1), nn.BatchNorm2d(48), nn.ReLU(inplace=True),
            nn.Conv2d(48, 96, 3, stride=2, padding=1), nn.BatchNorm2d(96), nn.ReLU(inplace=True),
            nn.Conv2d(96, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(
            nn.Linear(128 + prop_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, wp_count * 2),
        )
        self.wp_count = wp_count

    def forward(self, image, prop):
        feat = self.encoder(image).flatten(1)
        return self.head(torch.cat([feat, prop], dim=-1)).view(-1, self.wp_count, 2)


def build_model(backbone: str, wp_count=6, feat_dim=384, hidden_dim=256,
                image_size=(128, 72)):
    if backbone == "dinov3":
        return WaypointHead(feat_dim=feat_dim, wp_count=wp_count, hidden_dim=hidden_dim)
    if backbone == "tinycnn":
        return TinyCNN(wp_count=wp_count, hidden_dim=hidden_dim, image_size=image_size)
    raise ValueError(f"未知 backbone: {backbone} (dinov3 / tinycnn)")


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def load_checkpoint(path, device="cpu"):
    """返回 (model, ckpt). model 已 eval()."""
    ckpt = torch.load(path, map_location=device)
    cfg = ckpt["config"]
    model = build_model(
        cfg["backbone"], wp_count=cfg["wp_count"], feat_dim=cfg.get("feat_dim", 384),
        hidden_dim=cfg.get("hidden_dim", 256), image_size=cfg.get("image_size", (128, 72)),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt