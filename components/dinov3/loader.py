"""DINOv3 公共加载模块.

提供统一的 DINOv3 模型加载接口, 供项目各模块共享使用。

用法:
    from components.dinov3 import load_encoder, get_model_path

    # 获取模型路径
    path = get_model_path()

    # 直接加载 encoder + processor
    encoder, processor = load_encoder(device="cuda")
"""
from __future__ import annotations
from pathlib import Path

# 组件根目录: components/dinov3/
_COMPONENT_DIR = Path(__file__).resolve().parent
# 权重目录: components/dinov3/weights/vits16/
WEIGHTS_DIR = _COMPONENT_DIR / "weights" / "vits16"


def get_model_path() -> str:
    """返回 DINOv3 权重目录的绝对路径 (str), 可直接传给 from_pretrained()."""
    return str(WEIGHTS_DIR)


def check_weights() -> bool:
    """检查权重文件是否已下载."""
    return (WEIGHTS_DIR / "model.safetensors").exists()


def load_encoder(device: str = "cuda"):
    """加载 DINOv3 encoder + processor.

    Args:
        device: 目标设备, "cuda" 或 "cpu"

    Returns:
        (encoder, processor) 元组
        - encoder: 已加载权重并设为 eval() 的 AutoModel
        - processor: AutoImageProcessor
    """
    from transformers import AutoModel, AutoImageProcessor

    model_path = get_model_path()
    encoder = AutoModel.from_pretrained(model_path).to(device)
    encoder.eval()
    processor = AutoImageProcessor.from_pretrained(model_path)
    return encoder, processor


def extract_features(encoder, processor, images_pil):
    """从 PIL 图片列表提取 DINOv3 特征.

    Args:
        encoder: DINOv3 encoder (AutoModel)
        processor: DINOv3 processor (AutoImageProcessor)
        images_pil: PIL Image 列表

    Returns:
        特征 tensor [B, 384], 融合 CLS + mean(patches)
    """
    import torch

    inputs = processor(images=images_pil, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(next(encoder.parameters()).device)
    with torch.no_grad():
        out = encoder(pixel_values)
    cls = out.last_hidden_state[:, 0]            # [B, 384]
    patches = out.last_hidden_state[:, 1:].mean(dim=1)  # [B, 384]
    feat = (cls + patches) / 2.0
    return feat
