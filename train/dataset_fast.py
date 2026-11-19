"""FastDataset: 预提取 DINOv3 特征, 训练时直接加载缓存特征.

支持:
  - 多数据源: 同时加载原始 PID 数据 + 扰动注入数据
  - 特征增强: Gaussian noise / dropout, 模拟分布偏移
  - 预提取: 训练时零推理开销

这样训练时不需要跑 DINOv3 推理, 速度提升 10-50x.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset

from dataset import (
    load_reference_path, compute_arc_lengths,
    compute_gt_waypoints, WP_COUNT, IMAGE_SIZE,
)


# ---- 特征增强参数 ----
FEAT_AUG_NOISE_STD = 0.02    # Gaussian noise 标准差
FEAT_AUG_DROPOUT_P = 0.05    # 特征 dropout 概率


class _SingleSessionDataset(Dataset):
    """单个 session 的预提取特征数据集."""

    def __init__(
        self,
        session_dir: str,
        ref_x: np.ndarray,
        ref_y: np.ndarray,
        arc_lens: np.ndarray,
        model,
        processor,
        device: str = "cuda",
        skip_frames: int = 1,
    ):
        self.session_dir = Path(session_dir)
        self.device = device

        frames_dir = self.session_dir / "frames"
        frame_files = sorted(frames_dir.glob("*.json"))
        frame_files = frame_files[::skip_frames]

        self.features = []
        self.gt_waypoints = []
        self.frame_ids = []

        print(f"  [{self.session_dir.parent.name}/{self.session_dir.name}] "
              f"预提取 {len(frame_files)} 帧...")
        t0 = time.time()

        model.eval()
        batch_imgs = []
        batch_indices = []

        for i, ff in enumerate(frame_files):
            with open(ff) as f:
                fd = json.load(f)

            if fd.get("offroad", False) or fd.get("collision", False):
                continue

            img_path = self.session_dir / fd["image"]
            image = Image.open(img_path).convert("RGB")
            batch_imgs.append(image)
            batch_indices.append((i, fd))

            if len(batch_imgs) >= 64 or i == len(frame_files) - 1:
                inputs = processor(images=batch_imgs, return_tensors="pt")
                pixel_values = inputs["pixel_values"].to(device)

                with torch.no_grad():
                    outputs = model(pixel_values)

                cls = outputs.last_hidden_state[:, 0]
                patches = outputs.last_hidden_state[:, 1:].mean(dim=1)
                combined = ((cls + patches) / 2.0).cpu()

                for j, (idx, frame_data) in enumerate(batch_indices):
                    pos = frame_data["position"]
                    heading = frame_data["heading"]
                    wp = compute_gt_waypoints(
                        ref_x, ref_y, arc_lens, pos["x"], pos["y"], heading
                    )
                    self.features.append(combined[j])
                    self.gt_waypoints.append(torch.from_numpy(wp))
                    self.frame_ids.append(frame_data["frame_id"])

                batch_imgs = []
                batch_indices = []

        elapsed = time.time() - t0
        print(f"    ✓ {len(self.features)} 帧, {elapsed:.1f}s")

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return {
            "features": self.features[idx],
            "gt_waypoints": self.gt_waypoints[idx],
            "frame_id": self.frame_ids[idx],
        }


class FastDrivingDataset(Dataset):
    """多数据源预提取特征数据集.

    Args:
        session_dirs: 多个录制目录的列表
        ref_file: 参考路径 JSON
        model: DINOv3 模型 (用于特征提取)
        processor: DINOv3 预处理器
        device: 设备
        skip_frames: 跳帧间隔
        augment: 是否启用特征增强
    """

    def __init__(
        self,
        session_dirs: list[str],
        ref_file: str,
        model,
        processor,
        device: str = "cuda",
        skip_frames: int = 1,
        augment: bool = True,
    ):
        self.augment = augment

        ref_x, ref_y = load_reference_path(ref_file)
        arc_lens = compute_arc_lengths(ref_x, ref_y)

        # 对每个 session 分别提取特征
        sub_datasets = []
        for sd in session_dirs:
            if not Path(sd).exists():
                print(f"  [WARNING] 目录不存在, 跳过: {sd}")
                continue
            sub = _SingleSessionDataset(
                sd, ref_x, ref_y, arc_lens, model, processor, device, skip_frames
            )
            if len(sub) > 0:
                sub_datasets.append(sub)

        # 合并
        if len(sub_datasets) == 1:
            self._inner = sub_datasets[0]
        else:
            self._inner = ConcatDataset(sub_datasets)

        total = len(self._inner)
        print(f"\n  === 合并 {len(sub_datasets)} 个数据源, 共 {total} 帧 ===")
        print(f"  特征增强: {'开启' if augment else '关闭'}")

    def __len__(self):
        return len(self._inner)

    def __getitem__(self, idx):
        sample = self._inner[idx]
        feat = sample["features"]

        # 特征增强 (训练时)
        if self.augment:
            # Gaussian noise
            noise = torch.randn_like(feat) * FEAT_AUG_NOISE_STD
            feat = feat + noise
            # Random feature dropout
            mask = torch.bernoulli(torch.ones_like(feat) * (1 - FEAT_AUG_DROPOUT_P))
            feat = feat * mask

        sample["features"] = feat
        return sample


class FastModel(torch.nn.Module):
    """简化模型: 只包含 Waypoint Head (特征已预提取)."""

    def __init__(self, feat_dim=384, wp_count=6, hidden_dim=256):
        super().__init__()
        self.head = torch.nn.Sequential(
            torch.nn.Linear(feat_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(hidden_dim, wp_count * 2),
        )
        self.wp_count = wp_count

    def forward(self, features):
        out = self.head(features)
        return out.view(-1, self.wp_count, 2)
