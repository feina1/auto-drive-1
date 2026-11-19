"""训练 waypoint 模型 (DINOv3 冻结特征 + head, 或 TinyCNN 冒烟).

数据划分: 默认 leave-one-lap-out (最后一圈做验证), 避免相邻帧泄漏。
损失: 加权 SmoothL1 (近处权重大) + 二阶平滑项。

本机 (无 GPU) 用法:
    python training/train.py --backbone tinycnn --epochs 5 --limit-train 160

训练机 (GPU) 用法:
    python training/extract_features.py --device cuda
    python training/train.py --backbone dinov3 --device cuda --epochs 300
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from common import (Frames, get_device, per_point_rmse, set_seed, write_json,
                    PIPELINE_DIR)
from dataset import (FeatureDataset, ImageDataset, load_feature_cache,
                     resolve_default_npz, split_feature_cache)
from model import build_model, count_params

DEFAULT_WEIGHTS = [3.0, 2.5, 2.0, 1.5, 1.0, 0.8]


def parse_floats(text):
    return [float(x) for x in str(text).replace(" ", "").split(",") if x]


def forward_batch(model, batch, device, backbone, zero_prop=False):
    image = batch["image"].to(device) if "image" in batch else None
    feat = batch["feature"].to(device) if "feature" in batch else None
    prop = batch["prop"].to(device)
    if zero_prop:  # 消融: 去掉车速/转向输入, 其余完全一致
        prop = prop * 0.0
    wp = batch["wp"].to(device)
    if backbone == "dinov3":
        pred = model(feat, prop)
    else:
        pred = model(image, prop)
    return pred, wp


def compute_loss(pred, gt, weights, smooth_weight):
    per_point = F.smooth_l1_loss(pred, gt, beta=1.0, reduction="none").sum(dim=-1)  # [B,K]
    loss = (per_point * weights.unsqueeze(0)).mean()
    if smooth_weight > 0 and pred.shape[1] >= 3:
        d2 = pred[:, 2:] - 2.0 * pred[:, 1:-1] + pred[:, :-2]
        loss = loss + smooth_weight * d2.pow(2).sum(dim=-1).mean()
    return loss


@torch.no_grad()
def evaluate(model, loader, weights, smooth_weight, device, backbone, zero_prop=False):
    model.eval()
    losses, preds, gts = [], [], []
    for batch in loader:
        pred, gt = forward_batch(model, batch, device, backbone, zero_prop)
        losses.append(compute_loss(pred, gt, weights, smooth_weight).item())
        preds.append(pred.cpu().numpy())
        gts.append(gt.cpu().numpy())
    pred = np.concatenate(preds, axis=0)
    gt = np.concatenate(gts, axis=0)
    rmse, ade, fde = per_point_rmse(pred, gt)
    return dict(loss=float(np.mean(losses)), ade_m=ade, fde_m=fde,
                rmse_per_point_m=[round(float(x), 4) for x in rmse])


def make_loaders(args, frames, device):
    """返回 (train_loader, val_loader, model, meta)."""
    val_laps = parse_floats(args.val_laps) if args.val_laps else None
    image_size = tuple(int(x) for x in args.image_size.split("x"))

    if args.backbone == "dinov3":
        cache_path = Path(args.features) if args.features else \
            frames.npz_path.with_name(frames.npz_path.stem + "_features_dinov3.npz")
        if not cache_path.exists():
            raise SystemExit(
                f"缺少 DINOv3 特征缓存: {cache_path}\n"
                f"先运行: python training/extract_features.py --npz {frames.npz_path} --device cuda"
            )
        cache = load_feature_cache(cache_path)
        tr, va, val_laps = split_feature_cache(cache, frames, val_laps)
        if args.train_stride > 1:
            tr = tr[::args.train_stride]
        if args.limit_train > 0:
            tr = tr[:args.limit_train]
        train_ds = FeatureDataset(cache["features"], cache["prop"], cache["wp"], tr, augment=True)
        val_ds = FeatureDataset(cache["features"], cache["prop"], cache["wp"], va, augment=False)
        model = build_model("dinov3", wp_count=frames.wp_count,
                            feat_dim=cache["meta"]["feat_dim"], hidden_dim=args.hidden_dim)
        info = dict(feature_cache=str(cache_path), feat_dim=cache["meta"]["feat_dim"],
                    val_laps=[int(x) for x in val_laps])
    else:
        tr, va, val_laps = frames.lap_split(val_laps)
        if args.train_stride > 1:
            tr = tr[::args.train_stride]
        if args.limit_train > 0:
            tr = tr[:args.limit_train]
        train_ds = ImageDataset(frames, tr, image_size=image_size, augment=True)
        val_ds = ImageDataset(frames, va, image_size=image_size, augment=False)
        model = build_model("tinycnn", wp_count=frames.wp_count,
                            hidden_dim=args.hidden_dim, image_size=image_size)
        info = dict(image_size=list(image_size), val_laps=[int(x) for x in val_laps])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers)
    info.update(train_samples=len(train_ds), val_samples=len(val_ds))
    return train_loader, val_loader, model.to(device), info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", type=str, default=None,
                        help="数据集 npz (默认优先 dataset_waypoints_5laps.npz)")
    parser.add_argument("--backbone", choices=["dinov3", "tinycnn"], default="dinov3")
    parser.add_argument("--features", type=str, default=None, help="DINOv3 特征缓存路径")
    parser.add_argument("--val-laps", type=str, default=None,
                        help="验证圈号, 逗号分隔; 默认最后一圈, 如 5")
    parser.add_argument("--out-dir", type=str, default=str(PIPELINE_DIR / "output" / "training"))
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--image-size", type=str, default="128x72")
    parser.add_argument("--loss-weights", type=str, default=None,
                        help="6 个点的权重, 如 3,2.5,2,1.5,1,0.8")
    parser.add_argument("--smooth-weight", type=float, default=0.02)
    parser.add_argument("--limit-train", type=int, default=0, help="只用前 N 个训练样本 (冒烟)")
    parser.add_argument("--train-stride", type=int, default=1, help="训练帧抽稀步长 (数据密度消融)")
    parser.add_argument("--zero-prop", action="store_true",
                        help="消融: 把 proprio 置 0, 验证车速输入的必要性")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--patience", type=int, default=0, help="早停轮数, 0=关闭")
    args = parser.parse_args()

    set_seed(args.seed)
    npz_path = Path(args.npz) if args.npz else resolve_default_npz()
    frames = Frames(npz_path)
    device = get_device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[train] 数据: {frames.npz_path}  {frames.summarize()}")
    print(f"[train] backbone={args.backbone} device={device} out={out_dir}")

    train_loader, val_loader, model, split_info = make_loaders(args, frames, device)
    total, trainable = count_params(model)
    print(f"[train] 参数: 总 {total/1e6:.2f}M, 可训练 {trainable/1e6:.2f}M")
    print(f"[train] 划分: {split_info}")

    weights = torch.tensor(parse_floats(args.loss_weights) if args.loss_weights
                           else DEFAULT_WEIGHTS[:frames.wp_count],
                           dtype=torch.float32, device=device)
    if len(weights) != frames.wp_count:
        raise SystemExit(f"loss 权重数量 {len(weights)} != wp_count {frames.wp_count}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs,
                                                           eta_min=args.lr * 0.02)

    history = []
    best_ade = float("inf")
    best_epoch = 0
    stale = 0
    t_start = time.time()
    ckpt_path = out_dir / f"{args.backbone}_best.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for batch in train_loader:
            pred, gt = forward_batch(model, batch, device, args.backbone, args.zero_prop)
            loss = compute_loss(pred, gt, weights, args.smooth_weight)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            train_losses.append(loss.item())
        scheduler.step()

        val = evaluate(model, val_loader, weights, args.smooth_weight, device,
                       args.backbone, args.zero_prop)
        train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
        history.append(dict(epoch=epoch, train_loss=train_loss, **val,
                            lr=optimizer.param_groups[0]["lr"],
                            time_s=round(time.time() - t_start, 1)))

        improved = val["ade_m"] < best_ade
        if improved:
            best_ade, best_epoch, stale = val["ade_m"], epoch, 0
            torch.save({
                "epoch": epoch,
                "val": val,
                "model_state": model.state_dict(),
                "config": dict(backbone=args.backbone, wp_count=frames.wp_count,
                               feat_dim=split_info.get("feat_dim", 384),
                               hidden_dim=args.hidden_dim,
                               image_size=list(split_info.get("image_size", [128, 72])),
                               prop_dim=2),
                "frames": str(frames.npz_path),
                "feature_cache": split_info.get("feature_cache"),
                "val_laps": split_info.get("val_laps", []),
                "loss_weights": weights.detach().cpu().tolist(),
                "smooth_weight": args.smooth_weight,
                "zero_prop": bool(args.zero_prop),
            }, ckpt_path)
        else:
            stale += 1

        if epoch == 1 or epoch % max(1, args.epochs // 20) == 0 or improved:
            rmse = " ".join(f"{x:.2f}" for x in val["rmse_per_point_m"])
            print(f"[train] {epoch:4d}/{args.epochs} train={train_loss:.4f} "
                  f"val_ade={val['ade_m']:.3f}m fde={val['fde_m']:.3f}m "
                  f"rmse=[{rmse}] best={best_ade:.3f} ({time.time()-t_start:.0f}s)", flush=True)

        if args.patience > 0 and stale >= args.patience:
            print(f"[train] 早停: {args.patience} 轮无提升")
            break

    write_json(out_dir / f"{args.backbone}_history.json", history)
    write_json(out_dir / f"{args.backbone}_split.json", dict(
        npz=str(frames.npz_path), **{k: v for k, v in split_info.items()},
        best_epoch=best_epoch, best_val_ade_m=best_ade,
        zero_prop=bool(args.zero_prop), train_stride=args.train_stride,
    ))
    print(f"[train] 完成: best epoch={best_epoch} val_ade={best_ade:.3f}m")
    print(f"[train] checkpoint → {ckpt_path}")
    print(f"[train] 下一步: python training/predict.py --ckpt {ckpt_path} --device {args.device}")


if __name__ == "__main__":
    main()