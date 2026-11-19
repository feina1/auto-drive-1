"""数据读取: 原始图像 Dataset / DINOv3 特征 Dataset.

图像来源为 dataset_waypoints*.npz 的 image_path 字段 (相对 npz 所在目录),
默认只用有图的帧。划分按圈进行 (leave-one-lap-out), 避免相邻帧泄漏。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from common import Frames, make_prop


def _load_image(path: Path, image_size, hflip=False):
    img = Image.open(path).convert("RGB")
    if img.size != tuple(image_size):
        img = img.resize(tuple(image_size), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    if hflip:
        arr = arr[:, ::-1, :]
    return torch.from_numpy(np.ascontiguousarray(arr.transpose(2, 0, 1)))


class ImageDataset(Dataset):
    """原始图像 + proprio + GT waypoints (TinyCNN 训练/预测用)."""

    def __init__(self, frames: Frames, indices, image_size=(128, 72), augment=False):
        self.frames = frames
        self.indices = np.asarray(indices, dtype=np.int64)
        self.image_size = tuple(image_size)
        self.augment = augment

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = int(self.indices[i])
        hflip = bool(self.augment and np.random.rand() < 0.5)
        img = _load_image(self.frames.image_abs(idx), self.image_size, hflip)
        wp = torch.from_numpy(self.frames.wp[idx].copy())
        if hflip:  # 图像镜像 → waypoint 的 y 取反
            wp[:, 1] = -wp[:, 1]
        prop = torch.from_numpy(
            make_prop([self.frames.speed[idx]], [self.frames.steering[idx]])[0]
        )
        return dict(image=img, prop=prop, wp=wp, index=idx)


class FeatureDataset(Dataset):
    """预提取 DINOv3 特征 + proprio + GT waypoints.

    特征增强: Gaussian noise + dropout, 模拟视觉特征抖动。
    """

    def __init__(self, features, prop, wp, indices, augment=False,
                 noise_std=0.02, dropout_p=0.05):
        self.features = features
        self.prop = prop
        self.wp = wp
        self.indices = np.asarray(indices, dtype=np.int64)
        self.augment = augment
        self.noise_std = float(noise_std)
        self.dropout_p = float(dropout_p)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = int(self.indices[i])
        feat = self.features[idx].clone()
        if self.augment:
            feat = feat + torch.randn_like(feat) * self.noise_std
            mask = torch.bernoulli(torch.ones_like(feat) * (1.0 - self.dropout_p))
            feat = feat * mask
        return dict(feature=feat, prop=self.prop[idx].clone(),
                    wp=self.wp[idx].clone(), index=idx)


def load_feature_cache(cache_path):
    """读取 extract_features.py 生成的缓存."""
    d = np.load(cache_path, allow_pickle=False)
    src = d["source_npz"] if "source_npz" in d else b""
    meta = {
        "source_npz": src.decode("utf-8", "ignore") if isinstance(src, bytes) else str(src),
        "feat_dim": int(d["features"].shape[1]),
        "count": int(d["features"].shape[0]),
    }
    return dict(
        features=torch.from_numpy(d["features"].astype(np.float32)),
        prop=torch.from_numpy(d["prop"].astype(np.float32)),
        wp=torch.from_numpy(d["wp"].astype(np.float32)),
        src_index=d["src_index"].astype(np.int64),
        speed=d["speed_kmh"].astype(np.float32),
        lap=d["lap"].astype(np.int32),
        t=d["t"].astype(np.float32),
        image_path=d["image_path"],
        meta=meta,
    )


def split_feature_cache(cache, frames: Frames, val_laps=None):
    """把特征缓存的 src_index 映射回原 npz 帧号, 并按圈划分."""
    src = cache["src_index"]
    lap = frames.lap[src]
    laps = sorted(int(x) for x in np.unique(lap) if x > 0)
    if not laps:  # 旧单圈数据
        legacy = frames.lap_split(None)
        tr = np.where(np.isin(src, legacy[0]))[0]
        va = np.where(np.isin(src, legacy[1]))[0]
        return tr, va, []
    if val_laps is None:
        val_laps = [laps[-1]]
    val_laps = [int(x) for x in np.asarray(val_laps).reshape(-1)]
    is_val = np.isin(lap, val_laps)
    tr, va = np.where(~is_val)[0], np.where(is_val)[0]
    if len(tr) == 0 or len(va) == 0:
        raise RuntimeError(f"特征缓存按圈划分为空: val_laps={val_laps}, laps={laps}")
    return tr, va, val_laps


def resolve_default_npz():
    """默认优先用 5 圈变速数据集, 不存在时退回单圈."""
    import config

    multi = config.MULTI_LAP_DATASET.with_suffix(".npz")
    single = config.OUT_DIR / "dataset_waypoints.npz"
    return multi if multi.exists() else single