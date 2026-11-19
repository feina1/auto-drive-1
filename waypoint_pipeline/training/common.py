"""训练栈公共工具: 路径 / 设备 / 数据划分 / 指标 / 随机种子.

所有模块都以 waypoint_pipeline 为工作根目录:
    waypoint_pipeline/
    ├── config.py                 # 数据规格与控制器参数
    ├── training/                 # 本目录
    └── output/                   # 轨迹、图片、npz、checkpoint
"""
from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

import numpy as np

TRAINING_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = TRAINING_DIR.parent
PROJECT_ROOT = PIPELINE_DIR.parent
for _p in (str(PIPELINE_DIR), str(PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import config  # noqa: E402

# ---- 车辆状态归一化 (与采集数据一致, 固定常数, 不依赖训练集统计) ----
SPEED_SCALE_KMH = 72.0     # 采集速度上限
STEER_SCALE = 0.6          # 专家转向幅值上限
PROP_DIM = 2               # [speed/72, steering/0.6]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_device(name: str = "auto"):
    import torch

    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def make_prop(speed_kmh, steering):
    """车辆状态 → 归一化 proprio 向量 [N,2]."""
    speed_kmh = np.asarray(speed_kmh, dtype=np.float32)
    steering = np.asarray(steering, dtype=np.float32)
    return np.stack([speed_kmh / SPEED_SCALE_KMH, steering / STEER_SCALE], axis=1)


class Frames:
    """dataset_waypoints*.npz 的轻量视图 (含图片路径与圈号)."""

    def __init__(self, npz_path):
        self.npz_path = Path(npz_path).resolve()
        if not self.npz_path.exists():
            raise FileNotFoundError(f"数据集不存在: {self.npz_path}")
        d = np.load(self.npz_path, allow_pickle=False)
        self.wp = d["wp_ego"].astype(np.float32)                 # [N,6,2]
        self.speed = d["speed_kmh"].astype(np.float32)
        self.steering = d["steering"].astype(np.float32)
        self.t = d["t"].astype(np.float32)
        self.lap = d["lap"].astype(np.int32) if "lap" in d else np.zeros(len(self.t), np.int32)
        self.lap_t = d["lap_t"].astype(np.float32) if "lap_t" in d else self.t
        self.yaw_rate = d["yaw_rate"].astype(np.float32) if "yaw_rate" in d else np.zeros(len(self.t), np.float32)
        self.image = d["image_path"] if "image_path" in d else np.array([""] * len(self.t))
        self.wp_count = int(self.wp.shape[1])
        self.image_mask = np.array([str(x) != "" for x in self.image], dtype=bool)
        # 旧单圈数据没有圈号, 用采集时的 9:1 时间序划分兜底
        self.legacy_train_mask = d["train_mask"].astype(bool) if "train_mask" in d else None
        self.legacy_val_mask = d["val_mask"].astype(bool) if "val_mask" in d else None

    def __len__(self):
        return len(self.wp)

    def image_abs(self, idx: int) -> Path:
        return self.npz_path.parent / str(self.image[idx])

    def available_laps(self):
        return sorted(int(x) for x in np.unique(self.lap) if x > 0)

    def lap_split(self, val_laps=None):
        """按圈划分训练/验证; 没有圈号时退回 npz 内的时间序 mask.

        Returns:
            (train_idx, val_idx, val_laps) — 只包含有图片的帧。
        """
        idx = np.where(self.image_mask)[0]
        laps = self.available_laps()
        if not laps:  # 旧单圈数据: 用采集时的时间序 mask
            if self.legacy_train_mask is None:
                raise RuntimeError("数据集既无圈号也无 train_mask, 无法划分")
            tr = np.where(self.legacy_train_mask & self.image_mask)[0]
            va = np.where(self.legacy_val_mask & self.image_mask)[0]
            return tr, va, []

        if val_laps is None:
            val_laps = [laps[-1]]
        val_laps = [int(x) for x in np.asarray(val_laps).reshape(-1)]
        is_val = np.isin(self.lap[idx], val_laps)
        tr, va = idx[~is_val], idx[is_val]
        if len(tr) == 0 or len(va) == 0:
            raise RuntimeError(
                f"按圈划分得到空集合: val_laps={val_laps}, 可用圈号={laps}, "
                f"有图帧={len(idx)}"
            )
        return tr, va, val_laps

    def summarize(self):
        laps = self.available_laps()
        per_lap = []
        for lap_id in laps:
            m = self.lap == lap_id
            per_lap.append(dict(
                lap=lap_id,
                frames=int(m.sum()),
                images=int((m & self.image_mask).sum()),
                speed_mean_kmh=round(float(self.speed[m].mean()), 2),
            ))
        return dict(
            npz=str(self.npz_path),
            frames=int(len(self)),
            images=int(self.image_mask.sum()),
            wp_count=self.wp_count,
            laps=laps,
            per_lap=per_lap,
        )


def per_point_rmse(pred: np.ndarray, gt: np.ndarray):
    """[N,K,2] → 每点 RMSE (米), ADE, FDE."""
    err2 = ((pred - gt) ** 2).sum(axis=-1)          # [N,K]
    rmse = np.sqrt(err2.mean(axis=0))               # [K]
    ade = float(np.sqrt(err2.mean()))
    fde = float(np.sqrt(err2[:, -1].mean()))
    return rmse, ade, fde


def write_json(path: Path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, ensure_ascii=False), encoding="utf-8")
    return path