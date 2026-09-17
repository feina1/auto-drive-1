"""Model: DINOv3 (Frozen) + Waypoint Head.

架构:
  Front RGB [3, 224, 224]
      ↓
  DINOv3 ViT-S/16 (frozen)
      ↓
  CLS token [384] + mean pool patches [384]
      ↓
  Waypoint Head (MLP)
      ↓
  Predicted Waypoints [6, 2]  (ego frame, 单位: 米)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class WaypointHead(nn.Module):
    """从视觉特征预测 waypoints.

    输入: 视觉特征 (CLS + pooled patches) → [batch, feat_dim]
    输出: [batch, wp_count, 2]
    """

    def __init__(self, feat_dim: int = 384, wp_count: int = 6, hidden_dim: int = 256):
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

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """feat: [B, D] → [B, wp_count, 2]"""
        out = self.mlp(feat)
        return out.view(-1, self.wp_count, 2)


class DrivingModel(nn.Module):
    """DINOv3 + Waypoint Head.

    DINOv3 冻结, 只训练 Waypoint Head.
    """

    def __init__(
        self,
        dinov3_model: nn.Module,
        wp_count: int = 6,
        hidden_dim: int = 256,
        freeze_encoder: bool = True,
        use_cls_only: bool = False,
    ):
        super().__init__()
        self.encoder = dinov3_model
        feat_dim = self.encoder.config.hidden_size  # 384 for ViT-S
        self.use_cls_only = use_cls_only

        # 冻结 encoder
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            self.encoder.eval()

        self.head = WaypointHead(feat_dim, wp_count, hidden_dim)

    def extract_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """从 DINOv3 提取特征.

        返回: [B, feat_dim]
        """
        with torch.no_grad():
            outputs = self.encoder(pixel_values)

        cls_token = outputs.last_hidden_state[:, 0]       # [B, D]

        if self.use_cls_only:
            return cls_token

        # CLS + mean pool of patch tokens (更丰富)
        patch_tokens = outputs.last_hidden_state[:, 1:]    # [B, N, D]
        pooled = patch_tokens.mean(dim=1)                  # [B, D]
        combined = (cls_token + pooled) / 2.0              # [B, D]
        return combined

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """pixel_values: [B, 3, 224, 224] → [B, wp_count, 2]"""
        feat = self.extract_features(pixel_values)
        return self.head(feat)

    def trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build_model(
    model_path: str,
    wp_count: int = 6,
    hidden_dim: int = 256,
    freeze_encoder: bool = True,
    use_cls_only: bool = False,
    device: str = "cuda",
) -> DrivingModel:
    """构建完整模型."""
    from transformers import AutoModel

    print(f"加载 DINOv3 encoder: {model_path}")
    encoder = AutoModel.from_pretrained(model_path).to(device)

    model = DrivingModel(
        encoder, wp_count, hidden_dim,
        freeze_encoder=freeze_encoder,
        use_cls_only=use_cls_only,
    ).to(device)

    print(f"  总参数: {model.total_params()/1e6:.1f}M")
    print(f"  可训练参数: {model.trainable_params()/1e6:.3f}M")
    return model


if __name__ == "__main__":
    from pathlib import Path
    MODEL_PATH = str(Path(__file__).resolve().parent / "dinov3-weights" / "vits16")
    model = build_model(MODEL_PATH)

    # 测试前向传播
    x = torch.randn(4, 3, 224, 224).cuda()
    with torch.no_grad():
        wp = model(x)
    print(f"\n测试: input={x.shape} → output={wp.shape}")
    print(f"  预期: [4, 6, 2]")
    assert wp.shape == (4, 6, 2), f"Shape mismatch: {wp.shape}"
    print("✓ 模型验证通过")
