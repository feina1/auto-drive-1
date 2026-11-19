"""方案C 等分布训练: 每圈按时间间隔抽帧分到 train/val, 保证两边都有全部速度档.

划分策略:
  - 每圈内有图帧按时间排序;
  - 每隔 val_stride 帧取 1 帧归入 val, 其余归 train;
  - 这样 train 和 val 都包含 5 个速度档 (40/48/56/64/72 km/h) 的样本;
  - 相邻帧间隔 = val_stride × 0.5s, 默认 stride=5 → 2.5s (~40m), 避免画面泄漏.

用法:
    python training/train_interleaved.py --device cuda
    python training/train_interleaved.py --device cuda --val-stride 4 --epochs 200
    python training/train_interleaved.py --device cuda --eval-mode closed_loop

与 train.py 的区别仅在划分方式; 模型/损失/优化器/增强全部复用同一套代码.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from common import TRAINING_DIR, get_device, write_json, Frames
from dataset import FeatureDataset, load_feature_cache
from model import build_model, count_params


def interleaved_split(lap_arr, val_stride=5):
    """每圈内按 stride 抽帧分到 val, 其余 train.

    Args:
        lap_arr: [M] 每个特征缓存样本的圈号 (1..5)
        val_stride: 每隔 N 帧取 1 帧做 val

    Returns:
        train_idx, val_idx (numpy arrays, 特征缓存内的索引)
    """
    M = len(lap_arr)
    is_val = np.zeros(M, dtype=bool)
    for lap_id in np.unique(lap_arr):
        lap_indices = np.where(lap_arr == lap_id)[0]
        # 每 stride 个取第 0 个做 val
        val_positions = np.arange(0, len(lap_indices), val_stride)
        is_val[lap_indices[val_positions]] = True
    train_idx = np.where(~is_val)[0]
    val_idx = np.where(is_val)[0]
    return train_idx, val_idx


@torch.no_grad()
def evaluate(model, loader, device, loss_weights, wp_count):
    model.eval()
    total_ade, total_fde, total_loss, n = 0.0, 0.0, 0.0, 0
    per_point_sq = np.zeros(wp_count)
    for batch in loader:
        feat = batch["feature"].to(device)
        prop = batch["prop"].to(device)
        gt = batch["wp"].to(device)
        pred = model(feat, prop)
        diff = pred - gt
        dist = torch.norm(diff, dim=-1)  # [B, K]
        ade = dist.mean(dim=1)  # [B]
        fde = dist[:, -1]
        total_ade += ade.sum().item()
        total_fde += fde.sum().item()
        per_point_sq += (diff ** 2).sum(dim=(0, 2)).cpu().numpy()
        loss_fn = torch.nn.SmoothL1Loss(beta=1.0, reduction="none")
        w = torch.tensor(loss_weights[:wp_count], device=device).unsqueeze(0).unsqueeze(2)
        total_loss += (loss_fn(pred, gt) * w).sum().item()
        n += feat.shape[0]
    rmse = np.sqrt(per_point_sq / (n * 2))
    return dict(ade_m=total_ade / n, fde_m=total_fde / n,
                loss=total_loss / n, rmse_per_point=rmse.tolist())


def train_one_epoch(model, loader, optimizer, device, loss_weights, wp_count,
                    smooth_weight, grad_clip=5.0):
    model.train()
    total_loss, n = 0.0, 0
    for batch in loader:
        feat = batch["feature"].to(device)
        prop = batch["prop"].to(device)
        gt = batch["wp"].to(device)
        pred = model(feat, prop)

        # 加权 SmoothL1
        loss_fn = torch.nn.SmoothL1Loss(beta=1.0, reduction="none")
        w = torch.tensor(loss_weights[:wp_count], device=device).unsqueeze(0).unsqueeze(2)
        loss = (loss_fn(pred, gt) * w).mean()

        # 二阶差分平滑
        if smooth_weight > 0 and wp_count >= 3:
            d2 = pred[:, 2:] - 2 * pred[:, 1:-1] + pred[:, :-2]
            loss = loss + smooth_weight * (d2 ** 2).mean()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item() * feat.shape[0]
        n += feat.shape[0]
    return total_loss / n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=str, default=None,
                        help="特征缓存 npz (默认自动找)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--val-stride", type=int, default=5,
                        help="每圈内隔N帧取1帧做val (默认5, 即20%%val)")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--loss-weights", type=str, default="3,2.5,2,1.5,1,0.8")
    parser.add_argument("--smooth-weight", type=float, default=0.02)
    parser.add_argument("--patience", type=int, default=0, help="早停轮数, 0=关")
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = get_device(args.device)
    out_dir = Path(args.out_dir) if args.out_dir else TRAINING_DIR.parent / "output" / "training_interleaved"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 加载特征缓存
    feat_path = args.features
    if feat_path is None:
        # 自动找
        candidate = Path(__file__).resolve().parent.parent / "output" / "dataset_waypoints_5laps_features_dinov3.npz"
        if candidate.exists():
            feat_path = str(candidate)
        else:
            raise FileNotFoundError("请指定 --features 或先运行 extract_features.py")
    print(f"[train-C] 特征缓存: {feat_path}")
    cache = load_feature_cache(feat_path)
    features, prop, wp = cache["features"], cache["prop"], cache["wp"]
    lap_arr = cache["lap"]
    speed_arr = cache["speed"]
    M = features.shape[0]
    print(f"[train-C] 样本总数: {M}, 速度档: {sorted(set(speed_arr.astype(int)))}")

    # 方案C划分
    train_idx, val_idx = interleaved_split(lap_arr, val_stride=args.val_stride)
    print(f"[train-C] 划分 stride={args.val_stride}: train={len(train_idx)}, val={len(val_idx)}")
    for lap_id in sorted(set(lap_arr)):
        tr_n = (lap_arr[train_idx] == lap_id).sum()
        va_n = (lap_arr[val_idx] == lap_id).sum()
        spd = speed_arr[lap_arr == lap_id][0]
        print(f"  圈{lap_id} ({spd:.0f} km/h): train={tr_n} val={va_n}")

    # Dataset & DataLoader
    loss_weights = [float(x) for x in args.loss_weights.split(",")]
    wp_count = wp.shape[1]
    train_ds = FeatureDataset(features, prop, wp, train_idx, augment=True)
    val_ds = FeatureDataset(features, prop, wp, val_idx, augment=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Model
    model = build_model("dinov3", wp_count=wp_count, feat_dim=features.shape[1],
                        hidden_dim=args.hidden_dim).to(device)
    total, trainable = count_params(model)
    print(f"[train-C] 模型: {trainable:,} 可训练参数")

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.02)

    best_ade = float("inf")
    best_epoch = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, device,
                                     loss_weights, wp_count, args.smooth_weight)
        scheduler.step()
        val_metrics = evaluate(model, val_loader, device, loss_weights, wp_count)
        elapsed = time.time() - t0

        history.append(dict(epoch=epoch, train_loss=train_loss,
                           val_ade=val_metrics["ade_m"], val_fde=val_metrics["fde_m"],
                           val_loss=val_metrics["loss"], lr=scheduler.get_last_lr()[0]))

        if epoch % 10 == 0 or val_metrics["ade_m"] < best_ade:
            print(f"[train-C] {epoch:3d}/{args.epochs} train={train_loss:.4f} "
                  f"val_ade={val_metrics['ade_m']:.3f}m fde={val_metrics['fde_m']:.3f}m "
                  f"({elapsed:.1f}s)", flush=True)

        if val_metrics["ade_m"] < best_ade:
            best_ade = val_metrics["ade_m"]
            best_epoch = epoch
            torch.save(dict(
                model_state=model.state_dict(),
                config=dict(backbone="dinov3", wp_count=wp_count,
                           feat_dim=features.shape[1], hidden_dim=args.hidden_dim,
                           image_size=(128, 72), prop_dim=2),
                epoch=epoch, val=val_metrics,
                val_ade=best_ade,
                split_mode="interleaved", val_stride=args.val_stride,
                train_samples=len(train_idx), val_samples=len(val_idx),
                loss_weights=loss_weights, smooth_weight=args.smooth_weight,
            ), out_dir / "dinov3_interleaved_best.pt")

        if args.patience > 0 and epoch - best_epoch >= args.patience:
            print(f"[train-C] 早停 @ epoch={epoch}, patience={args.patience}")
            break

    print(f"[train-C] 完成: best epoch={best_epoch} val_ade={best_ade:.3f}m")
    write_json(out_dir / "dinov3_interleaved_split.json", dict(
        split_mode="interleaved", val_stride=args.val_stride,
        train_samples=len(train_idx), val_samples=len(val_idx),
        best_epoch=best_epoch, best_val_ade_m=best_ade,
    ))
    write_json(out_dir / "dinov3_interleaved_history.json", history)
    print(f"[train-C] → {out_dir}")


if __name__ == "__main__":
    main()
