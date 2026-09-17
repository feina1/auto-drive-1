"""Train: DINOv3 (Frozen) + Waypoint Head 训练脚本.

用法:
    cd train/
    python train.py

训练策略:
  - 多数据源: 原始 PID 数据 + 扰动注入数据
  - 特征增强: Gaussian noise + dropout
  - 预提取 DINOv3 特征 (一次), 训练时只跑 Waypoint Head
  - L2 loss on waypoints, 近处 waypoint 权重更高
  - AdamW + cosine annealing
  - 每 epoch 验证, 保存最优 checkpoint
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import random_split, DataLoader
from transformers import AutoImageProcessor, AutoModel

# 项目内模块
from dataset import WP_COUNT, load_reference_path, compute_arc_lengths, compute_gt_waypoints
from dataset_fast import FastDrivingDataset, FastModel

# ---- 路径配置 ----
TRAIN_DIR = Path(__file__).resolve().parent
PROJECT_DIR = TRAIN_DIR.parent
REF_FILE = str(PROJECT_DIR / "simple_track_reference_clean.json")
MODEL_PATH = str(TRAIN_DIR / "dinov3-weights" / "vits16")
CKPT_DIR = TRAIN_DIR / "checkpoints"

# ---- 数据源: 同时使用原始数据 + 扰动数据 ----
SESSION_DIRS = [
    # 原始 PID 数据
    str(PROJECT_DIR / "records_pid" / "2026-09-13_18-22"),
    # 扰动注入数据 (录制完成后自动包含)
]
# 自动扫描 records_perturbed 下的所有 session
PERTURBED_ROOT = PROJECT_DIR / "records_perturbed"
if PERTURBED_ROOT.exists():
    for d in sorted(PERTURBED_ROOT.iterdir()):
        if d.is_dir() and (d / "frames").exists():
            SESSION_DIRS.append(str(d))

# ---- 训练超参数 ----
BATCH_SIZE = 64
EPOCHS = 300
LR = 3e-4
WEIGHT_DECAY = 1e-4
ETA_MIN = 3e-6
VAL_RATIO = 0.15
SKIP_FRAMES = 2      # 跳帧: 每 2 帧取 1 帧 (减少冗余)

# ---- Waypoint loss 权重: 近处更重要 ----
WP_WEIGHTS = torch.tensor([3.0, 2.5, 2.0, 1.5, 1.0, 0.8])


def weighted_l2_loss(pred, target):
    """加权 L2 loss. pred/target: [B, WP_COUNT, 2]."""
    diff_sq = (pred - target).pow(2).sum(dim=-1)  # [B, WP_COUNT]
    weights = WP_WEIGHTS.to(pred.device).unsqueeze(0)  # [1, WP_COUNT]
    weighted = (diff_sq * weights).sum(dim=-1)  # [B]
    return weighted.mean()


def train_one_epoch(model, loader, optimizer):
    model.train()
    losses, l2s = [], []
    for batch in loader:
        feat = batch["features"].cuda()
        gt = batch["gt_waypoints"].cuda()

        pred = model(feat)
        loss = weighted_l2_loss(pred, gt)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            l2 = (pred - gt).pow(2).sum(-1).sqrt().mean().item()

        losses.append(loss.item())
        l2s.append(l2)

    return np.mean(losses), np.mean(l2s)


@torch.no_grad()
def validate(model, loader):
    model.eval()
    losses, l2s = [], []
    for batch in loader:
        feat = batch["features"].cuda()
        gt = batch["gt_waypoints"].cuda()

        pred = model(feat)
        loss = weighted_l2_loss(pred, gt)
        l2 = (pred - gt).pow(2).sum(-1).sqrt().mean().item()

        losses.append(loss.item())
        l2s.append(l2)

    return np.mean(losses), np.mean(l2s)


def main():
    print("=" * 60)
    print("  DINOv3 + Waypoint Head 训练")
    print("=" * 60)

    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n设备: {device}")
    print(f"数据源: {len(SESSION_DIRS)} 个")
    for sd in SESSION_DIRS:
        print(f"  - {sd}")

    # 1. 加载 DINOv3 encoder (冻结)
    print(f"\n加载 DINOv3: {MODEL_PATH}")
    processor = AutoImageProcessor.from_pretrained(MODEL_PATH)
    encoder = AutoModel.from_pretrained(MODEL_PATH).to(device)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    # 2. 构建数据集 (多数据源 + 特征增强)
    print(f"\n构建数据集 (skip={SKIP_FRAMES})...")
    full_dataset = FastDrivingDataset(
        session_dirs=SESSION_DIRS,
        ref_file=REF_FILE,
        model=encoder,
        processor=processor,
        device=device,
        skip_frames=SKIP_FRAMES,
        augment=True,
    )

    n_total = len(full_dataset)
    n_val = int(n_total * VAL_RATIO)
    n_train = n_total - n_val

    train_set, val_set = random_split(full_dataset, [n_train, n_val])
    train_loader = DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, pin_memory=True,
    )
    print(f"  Train: {n_train} | Val: {n_val} | Batch: {BATCH_SIZE}")

    # 3. 构建模型
    model = FastModel(feat_dim=384, wp_count=WP_COUNT, hidden_dim=256).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n模型参数: {n_params/1e3:.1f}K (可训练)")

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=ETA_MIN)

    # 4. 训练循环
    print(f"\n开始训练: {EPOCHS} epochs, LR={LR}, WD={WEIGHT_DECAY}")
    print("-" * 60)

    best_val_l2 = float("inf")
    history = []
    t_start = time.time()

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_l2 = train_one_epoch(model, train_loader, optimizer)
        val_loss, val_l2 = validate(model, val_loader)
        scheduler.step()

        lr_now = optimizer.param_groups[0]["lr"]
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_l2": train_l2,
            "val_loss": val_loss,
            "val_l2": val_l2,
            "lr": lr_now,
        })

        # 保存最优
        if val_l2 < best_val_l2:
            best_val_l2 = val_l2
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_l2": val_l2,
                "config": {
                    "feat_dim": 384,
                    "wp_count": WP_COUNT,
                    "hidden_dim": 256,
                    "session_dirs": SESSION_DIRS,
                    "skip_frames": SKIP_FRAMES,
                },
            }, CKPT_DIR / "best_model.pt")

        if epoch % 10 == 0 or epoch == 1:
            elapsed = time.time() - t_start
            print(f"Epoch {epoch:3d}/{EPOCHS} | "
                  f"train_l2={train_l2:.4f} val_l2={val_l2:.4f} | "
                  f"lr={lr_now:.6f} | best={best_val_l2:.4f} | "
                  f"{elapsed:.0f}s")

    elapsed_total = time.time() - t_start
    print("-" * 60)
    print(f"训练完成! 总耗时: {elapsed_total:.0f}s")
    print(f"最佳 val_l2: {best_val_l2:.4f}m")
    print(f"Checkpoint: {CKPT_DIR / 'best_model.pt'}")

    # 保存训练历史
    with open(CKPT_DIR / "train_history.json", "w") as f:
        json.dump(history, f, indent=2)

    # 释放 encoder
    del encoder


if __name__ == "__main__":
    main()
