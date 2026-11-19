"""Dataset: 录制数据 → DINOv3 特征 + GT Waypoints.

数据流:
  records_pid/<session>/images/*.jpg  →  DINOv3 preprocess → [3, 224, 224]
  records_pid/<session>/frames/*.json →  position, heading
  reference_path.json                 →  GT waypoints (ego frame)

GT Waypoint 生成逻辑 (V0 中心线 Oracle):
  1. 在参考路径上找到车辆最近点 (投影)
  2. 沿路径向前采样 WP_DIST 个 waypoints, 每隔 WP_STEP 米
  3. 转换到 ego 坐标系 (x=前方, y=左方)
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split

# ---- GT Waypoint 参数 ----
WP_STEP = 5.0       # 每个 waypoint 间隔 5m
WP_COUNT = 6        # 6 个 waypoints: 5m, 10m, 15m, 20m, 25m, 30m
WP_DISTANCES = [WP_STEP * (i + 1) for i in range(WP_COUNT)]

# ---- DINOv3 输入参数 ----
IMAGE_SIZE = 224    # DINOv3 默认输入尺寸


def load_reference_path(ref_file: str) -> tuple[np.ndarray, np.ndarray]:
    """加载参考路径, 返回 (xs, ys) numpy arrays."""
    with open(ref_file) as f:
        data = json.load(f)
    xs = np.array(data["reference_path"]["x"], dtype=np.float64)
    ys = np.array(data["reference_path"]["y"], dtype=np.float64)
    return xs, ys


def find_closest_idx(ref_x: np.ndarray, ref_y: np.ndarray, x: float, y: float) -> int:
    """在参考路径上找最近点的 index."""
    diff_sq = (ref_x - x) ** 2 + (ref_y - y) ** 2
    return int(np.argmin(diff_sq))


def compute_arc_lengths(ref_x: np.ndarray, ref_y: np.ndarray) -> np.ndarray:
    """计算参考路径上每个点的累积弧长."""
    dx = np.diff(ref_x)
    dy = np.diff(ref_y)
    seg_lengths = np.sqrt(dx**2 + dy**2)
    return np.concatenate([[0.0], np.cumsum(seg_lengths)])


def compute_gt_waypoints(
    ref_x: np.ndarray,
    ref_y: np.ndarray,
    arc_lens: np.ndarray,
    vehicle_x: float,
    vehicle_y: float,
    vehicle_heading: float,
) -> np.ndarray:
    """计算 ego frame 下的 GT waypoints.

    返回 shape [WP_COUNT, 2], 单位: 米.
    ego frame: x=前方, y=左方.
    """
    closest_idx = find_closest_idx(ref_x, ref_y, vehicle_x, vehicle_y)
    closest_arc = arc_lens[closest_idx]

    waypoints = []
    for dist in WP_DISTANCES:
        target_arc = closest_arc + dist
        # 在弧长上找最近的点
        idx = int(np.searchsorted(arc_lens, target_arc))
        idx = min(idx, len(ref_x) - 1)
        wx, wy = float(ref_x[idx]), float(ref_y[idx])

        # 世界坐标 → ego 坐标
        dx = wx - vehicle_x
        dy = wy - vehicle_y
        cos_h = np.cos(vehicle_heading)
        sin_h = np.sin(vehicle_heading)
        ego_x = dx * cos_h + dy * sin_h     # 前方
        ego_y = -dx * sin_h + dy * cos_h    # 左方
        waypoints.append([ego_x, ego_y])

    return np.array(waypoints, dtype=np.float32)


class DrivingDataset(Dataset):
    """自动驾驶录制数据集.

    Args:
        session_dir: 录制会话目录 (含 images/ 和 frames/)
        ref_file: 参考路径 JSON 文件路径
        model_path: DINOv3 模型路径 (用于提取特征)
        transform: 可选的图像 transform (如果不提供则使用 DINOv3 processor)
    """

    def __init__(
        self,
        session_dir: str,
        ref_file: str,
        processor=None,
        skip_frames: int = 1,
    ):
        self.session_dir = Path(session_dir)
        self.images_dir = self.session_dir / "images"
        self.frames_dir = self.session_dir / "frames"
        self.processor = processor

        # 加载参考路径
        self.ref_x, self.ref_y = load_reference_path(ref_file)
        self.arc_lens = compute_arc_lengths(self.ref_x, self.ref_y)

        # 加载所有帧 JSON
        frame_files = sorted(self.frames_dir.glob("*.json"))
        self.frame_files = frame_files[::skip_frames]

        # 预计算所有 GT waypoints (加速训练)
        self._gt_cache: dict[int, np.ndarray] = {}
        self._valid_indices: list[int] = []
        for i, ff in enumerate(self.frame_files):
            with open(ff) as f:
                fd = json.load(f)
            pos = fd["position"]
            heading = fd["heading"]
            # 跳过 offroad / collision 帧
            if fd.get("offroad", False) or fd.get("collision", False):
                continue
            wp = compute_gt_waypoints(
                self.ref_x, self.ref_y, self.arc_lens,
                pos["x"], pos["y"], heading,
            )
            self._gt_cache[i] = wp
            self._valid_indices.append(i)

        print(f"  Dataset: {len(self._valid_indices)}/{len(self.frame_files)} valid frames "
              f"from {self.session_dir.name}")

    def __len__(self) -> int:
        return len(self._valid_indices)

    def __getitem__(self, idx: int):
        real_idx = self._valid_indices[idx]
        frame_file = self.frame_files[real_idx]

        with open(frame_file) as f:
            fd = json.load(f)

        # 加载图像
        img_path = self.session_dir / fd["image"]
        image = Image.open(img_path).convert("RGB")

        # DINOv3 预处理
        if self.processor is not None:
            inputs = self.processor(images=image, return_tensors="pt")
            pixel_values = inputs["pixel_values"].squeeze(0)  # [3, 224, 224]
        else:
            # fallback: 手动 resize + normalize
            image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
            arr = np.array(image, dtype=np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.485, 0.456, 0.406])
            arr = (arr - mean) / std
            pixel_values = torch.from_numpy(arr).permute(2, 0, 1)

        # GT waypoints
        waypoints = self._gt_cache[real_idx]

        # 车辆状态 (可选, 用于后续扩展)
        ego_state = np.array([
            fd["speed_kmh"] / 70.0,  # 归一化速度
        ], dtype=np.float32)

        return {
            "pixel_values": pixel_values,       # [3, 224, 224]
            "gt_waypoints": torch.from_numpy(waypoints),  # [6, 2]
            "ego_state": torch.from_numpy(ego_state),      # [1]
            "frame_id": fd["frame_id"],
        }


def build_dataloaders(
    session_dir: str,
    ref_file: str,
    processor=None,
    batch_size: int = 32,
    val_ratio: float = 0.15,
    skip_frames: int = 1,
    num_workers: int = 4,
) -> tuple[DataLoader, DataLoader]:
    """构建 train/val DataLoader."""
    dataset = DrivingDataset(session_dir, ref_file, processor, skip_frames)

    n_val = int(len(dataset) * val_ratio)
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    print(f"  Train: {n_train} samples | Val: {n_val} samples | Batch: {batch_size}")
    return train_loader, val_loader


if __name__ == "__main__":
    # 快速测试
    from transformers import AutoImageProcessor

    REF_FILE = str(Path(__file__).resolve().parent.parent / "simple_track_reference_clean.json")
    SESSION_DIR = str(Path(__file__).resolve().parent.parent / "records_pid" / "2026-09-13_18-22")
    MODEL_PATH = str(Path(__file__).resolve().parent / "dinov3-weights" / "vits16")

    print("加载 DINOv3 processor...")
    processor = AutoImageProcessor.from_pretrained(MODEL_PATH)

    print("构建 Dataset...")
    ds = DrivingDataset(SESSION_DIR, REF_FILE, processor, skip_frames=10)

    print("\n采样 3 帧验证:")
    for i in [0, len(ds) // 2, len(ds) - 1]:
        sample = ds[i]
        wp = sample["gt_waypoints"]
        print(f"  Frame {sample['frame_id']}: "
              f"img={sample['pixel_values'].shape}, "
              f"waypoints[0]=({wp[0,0]:.1f}, {wp[0,1]:.1f}), "
              f"waypoints[-1]=({wp[-1,0]:.1f}, {wp[-1,1]:.1f})")
