#!/usr/bin/env python3
"""模型: DINOv3 (frozen) + Waypoint Head.

输入: 1 帧 RGB → DINOv3 → [384] → MLP → [20, 2] waypoints
"""
from __future__ import annotations
from pathlib import Path
import torch
import torch.nn as nn


class WaypointHead(nn.Module):
    """从视觉特征预测 waypoints."""
    def __init__(self, feat_dim=384, wp_count=20, hidden_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, wp_count * 2),
        )
        self.wp_count = wp_count

    def forward(self, feat):
        out = self.mlp(feat)
        return out.view(-1, self.wp_count, 2)


class FullModel(nn.Module):
    """DINOv3 encoder + Waypoint Head (encoder frozen)."""
    def __init__(self, encoder, processor, head):
        super().__init__()
        self.encoder = encoder
        self.processor = processor
        self.head = head
        # 冻结 encoder
        for p in self.encoder.parameters():
            p.requires_grad = False

    def extract_features(self, images_pil):
        """从 PIL 图片列表提取 DINOv3 特征 → [B, 384]."""
        inputs = self.processor(images=images_pil, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(next(self.encoder.parameters()).device)
        with torch.no_grad():
            out = self.encoder(pixel_values)
        cls = out.last_hidden_state[:, 0]        # [B, 384]
        patches = out.last_hidden_state[:, 1:].mean(dim=1)  # [B, 384]
        feat = (cls + patches) / 2.0
        return feat

    def forward(self, features):
        """从预提取的特征预测 waypoints."""
        return self.head(features)


def build_model(device="cuda"):
    """构建完整模型."""
    from transformers import AutoModel, AutoImageProcessor
    MODEL_PATH = str(Path(__file__).resolve().parent / "dinov3-weights" / "vits16")
    
    encoder = AutoModel.from_pretrained(MODEL_PATH).to(device)
    encoder.eval()
    processor = AutoImageProcessor.from_pretrained(MODEL_PATH)
    head = WaypointHead().to(device)
    
    model = FullModel(encoder, processor, head)
    return model


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


if __name__ == "__main__":
    model = build_model()
    total, trainable = count_params(model)
    print(f"总参数: {total/1e6:.1f}M")
    print(f"可训练: {trainable/1e6:.3f}M")
    # 测试推理
    from PIL import Image
    import numpy as np
    img = Image.fromarray(np.random.randint(0, 255, (360, 640, 3), dtype=np.uint8))
    feat = model.extract_features([img])
    wp = model(feat)
    print(f"特征: {feat.shape}")
    print(f"Waypoints: {wp.shape}")
