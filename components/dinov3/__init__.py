"""DINOv3 公共视觉特征提取组件.

提供冻结的 DINOv3 ViT-S/16 encoder 作为项目的共享视觉特征提取器。

快速开始:
    from components.dinov3 import load_encoder, get_model_path

    encoder, processor = load_encoder(device="cuda")
"""
from .loader import get_model_path, check_weights, load_encoder, extract_features, WEIGHTS_DIR

__all__ = [
    "get_model_path",
    "check_weights",
    "load_encoder",
    "extract_features",
    "WEIGHTS_DIR",
]
