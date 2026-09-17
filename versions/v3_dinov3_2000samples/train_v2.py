#!/usr/bin/env python3
"""训练: DINOv3 (frozen) + Waypoint Head.

用法:
    python train_v2.py                  # 完整训练
    python train_v2.py --quick          # 快速验证 (50 样本, 50 epochs)
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model_v2 import WaypointHead, build_model, count_params

TRAIN_DIR = Path(__file__).resolve().parent
DATA_DIR = TRAIN_DIR / "data"
CKPT_DIR = TRAIN_DIR / "checkpoints"
MANIFEST_FILE = DATA_DIR / "manifest.json"

WP_COUNT = 20
# 加权: 近处高, 远处低
LOSS_WEIGHTS = torch.tensor([2.5,2.5,2.5,2.0,2.0,1.5,1.5,1.2,1.0,1.0,
                              0.8,0.8,0.6,0.6,0.5,0.5,0.4,0.4,0.3,0.3])


class DrivingDataset(Dataset):
    """加载收集的数据, 提取 DINOv3 特征后缓存."""
    def __init__(self, manifest, features, gt_waypoints):
        self.features = features    # [N, 384]
        self.gt = gt_waypoints      # [N, 20, 2]
        assert len(features) == len(gt_waypoints)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.gt[idx]


def load_and_extract(model, manifest, device="cuda"):
    """加载所有图片, 提取 DINOv3 特征, 返回缓存 tensor."""
    print(f"提取 {len(manifest)} 个样本的 DINOv3 特征...")
    features = []
    gt_list = []
    skipped = 0

    for i, item in enumerate(manifest):
        try:
            img = Image.open(item["image"]).convert("RGB")
            feat = model.extract_features([img])  # [1, 384]
            features.append(feat[0].cpu())

            with open(item["gt"]) as f:
                gt_data = json.load(f)
            wp = torch.tensor(gt_data["waypoints"], dtype=torch.float32)
            gt_list.append(wp)
        except Exception as e:
            skipped += 1
            if skipped <= 3:
                print(f"  跳过 #{i}: {e}")

        if (i + 1) % 50 == 0:
            print(f"  已处理 {i+1}/{len(manifest)}")

    features = torch.stack(features)  # [N, 384]
    gt = torch.stack(gt_list)          # [N, 20, 2]
    print(f"  完成: {len(features)} 有效, {skipped} 跳过")
    return features, gt


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    # 加载 manifest
    with open(MANIFEST_FILE) as f:
        manifest = json.load(f)
    print(f"数据集: {len(manifest)} 个样本")

    # 构建模型
    model = build_model(device)
    head = model.head
    total, trainable = count_params(model)
    print(f"模型: 总参数 {total/1e6:.1f}M, 可训练 {trainable/1e6:.3f}M")

    # 提取特征 (缓存)
    features, gt = load_and_extract(model, manifest, device)
    features = features.to(device)
    gt = gt.to(device)

    # 特征增强函数
    def augment(feat):
        if args.augment:
            noise = torch.randn_like(feat) * 0.02
            mask = torch.rand_like(feat) > 0.05
            return (feat + noise) * mask
        return feat

    # 划分 train/val
    n = len(features)
    n_val = max(1, int(n * 0.2))
    n_train = n - n_val
    indices = torch.randperm(n)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]

    train_feat = features[train_idx]
    train_gt = gt[train_idx]
    val_feat = features[val_idx]
    val_gt = gt[val_idx]
    print(f"训练: {n_train}, 验证: {n_val}")

    # 优化器
    optimizer = AdamW(head.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=3e-6)
    weights = LOSS_WEIGHTS.to(device)

    # 训练循环
    history = []
    best_val_l2 = float('inf')

    print(f"\n开始训练: {args.epochs} epochs, batch_size={args.batch_size}")
    t_start = time.time()

    for epoch in range(args.epochs):
        head.train()
        perm = torch.randperm(n_train)
        train_losses = []

        for start in range(0, n_train, args.batch_size):
            batch_idx = perm[start:start+args.batch_size]
            batch_feat = augment(train_feat[batch_idx])
            batch_gt = train_gt[batch_idx]

            pred = head(batch_feat)
            # 加权 L2
            per_point = ((pred - batch_gt) ** 2).sum(dim=-1)  # [B, 20]
            loss = (per_point * weights.unsqueeze(0)).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        scheduler.step()

        # 验证
        head.eval()
        with torch.no_grad():
            val_pred = head(val_feat)
            val_per_point = ((val_pred - val_gt) ** 2).sum(dim=-1)
            val_loss = (val_per_point * weights.unsqueeze(0)).mean()
            val_l2 = val_per_point.mean(dim=-1).sqrt().mean()

        train_l2 = np.sqrt(np.mean(train_losses))
        elapsed = time.time() - t_start

        if (epoch + 1) % max(1, args.epochs // 20) == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:4d}/{args.epochs} | "
                  f"train_l2={train_l2:.3f}m val_l2={val_l2:.3f}m | "
                  f"{elapsed:.0f}s")

        history.append({
            "epoch": epoch + 1,
            "train_l2": float(train_l2),
            "val_l2": float(val_l2.item()),
            "time_s": int(elapsed),
        })

        # 保存最佳 (每 50 epoch 才存一次, 省磁盘)
        if val_l2.item() < best_val_l2 and (epoch + 1) % 10 == 0:
            best_val_l2 = val_l2.item()
            ckpt = {
                "epoch": epoch + 1,
                "val_l2": best_val_l2,
                "model_state_dict": head.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": {"feat_dim": 384, "wp_count": WP_COUNT, "hidden_dim": 256},
            }
            torch.save(ckpt, CKPT_DIR / "best_model.pt")

    # 最终保存
    total_time = time.time() - t_start
    print(f"\n训练完成: {total_time:.0f}s")
    print(f"最佳 val_l2: {best_val_l2:.4f}m")

    # 保存历史
    with open(CKPT_DIR / "train_history.json", "w") as f:
        json.dump(history, f, indent=1)

    # 最终一定存一个 checkpoint
    ckpt = {
        "epoch": args.epochs,
        "val_l2": best_val_l2,
        "model_state_dict": head.state_dict(),
        "config": {"feat_dim": 384, "wp_count": WP_COUNT, "hidden_dim": 256},
    }
    torch.save(ckpt, CKPT_DIR / "best_model.pt")
    print(f"Checkpoint 保存在: {CKPT_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--augment", action="store_true", default=True)
    parser.add_argument("--quick", action="store_true", help="快速验证: 50 epochs")
    args = parser.parse_args()

    if args.quick:
        args.epochs = 200
        print("=== 快速验证模式: 200 epochs ===")

    train(args)
