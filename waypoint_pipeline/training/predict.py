"""离线预测与评估: 在验证圈上逐帧预测 waypoints, 输出指标与可视化.

指标:
- 每点 RMSE (t+0.5 ... t+3.0, 米)
- ADE (所有点平均位移误差) / FDE (末点误差)
- 按圈、按速度分桶统计

用法:
    python training/predict.py --ckpt output/training/dinov3_best.pt --device cuda
    python training/predict.py --ckpt output/training/tinycnn_best.pt --device cpu --limit 200
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from common import (Frames, TRAINING_DIR, get_device, per_point_rmse, write_json,
                    PIPELINE_DIR)
from dataset import (ImageDataset, load_feature_cache, resolve_default_npz,
                     split_feature_cache)
from model import load_checkpoint


def _batch_images(paths, image_size):
    from PIL import Image

    out = []
    for p in paths:
        img = Image.open(p).convert("RGB")
        if img.size != tuple(image_size):
            img = img.resize(tuple(image_size), Image.BILINEAR)
        out.append(np.asarray(img, dtype=np.float32) / 255.0)
    arr = np.stack(out).transpose(0, 3, 1, 2)
    return torch.from_numpy(np.ascontiguousarray(arr))


@torch.no_grad()
def predict_dinov3(model, ckpt, frames, cache, val_laps, device, limit=0):
    tr, va, used_laps = split_feature_cache(cache, frames, val_laps)
    if limit > 0:
        va = va[:limit]
    src = cache["src_index"][va]
    feat = cache["features"][va].to(device)
    prop = cache["prop"][va].to(device)
    if ckpt.get("zero_prop", False):
        prop = prop * 0.0
    preds = []
    bs = 512
    for i in range(0, len(va), bs):
        preds.append(model(feat[i:i + bs], prop[i:i + bs]).cpu().numpy())
    return np.concatenate(preds), src, used_laps, "feature"


@torch.no_grad()
def predict_tinycnn(model, ckpt, frames, val_laps, device, limit=0):
    tr, va, used_laps = frames.lap_split(val_laps)
    if limit > 0:
        va = va[:limit]
    ds = ImageDataset(frames, va, image_size=tuple(ckpt["config"]["image_size"]))
    preds = []
    bs = 64
    for i in range(0, len(ds), bs):
        sample = [ds[j] for j in range(i, min(i + bs, len(ds)))]
        image = torch.stack([s["image"] for s in sample]).to(device)
        prop = torch.stack([s["prop"] for s in sample]).to(device)
        if ckpt.get("zero_prop", False):
            prop = prop * 0.0
        preds.append(model(image, prop).cpu().numpy())
    return np.concatenate(preds), va, used_laps, "image"


@torch.no_grad()
def predict_dinov3_online(encoder, processor, model, frames, va, device, zero_prop=False):
    """没有特征缓存时, 现场对验证帧提特征 (需要 DINOv3 权重)."""
    from PIL import Image

    preds = []
    bs = 16
    for i in range(0, len(va), bs):
        idx = va[i:i + bs]
        imgs = [Image.open(frames.image_abs(int(j))).convert("RGB") for j in idx]
        inputs = processor(images=imgs, return_tensors="pt")
        out = encoder(inputs["pixel_values"].to(device))
        cls = out.last_hidden_state[:, 0]
        patches = out.last_hidden_state[:, 1:].mean(dim=1)
        feat = (cls + patches) / 2.0
        prop = torch.from_numpy(np.stack([
            frames.speed[idx] / 72.0, frames.steering[idx] / 0.6], axis=1)).float().to(device)
        if zero_prop:
            prop = prop * 0.0
        preds.append(model(feat, prop).cpu().numpy())
        if (i // bs) % 20 == 0:
            print(f"[predict] 在线提特征 {i + len(idx)}/{len(va)}", flush=True)
    return np.concatenate(preds)


def group_metrics(src, pred, gt, frames):
    """按圈 / 速度分桶统计."""
    lap = frames.lap[src]
    speed = frames.speed[src]
    out = {"by_lap": {}, "by_speed": {}}
    for lap_id in sorted(set(int(x) for x in np.unique(lap))):
        m = lap == lap_id
        if m.sum() == 0:
            continue
        _, ade, fde = per_point_rmse(pred[m], gt[m])
        out["by_lap"][str(lap_id)] = dict(
            frames=int(m.sum()), speed_mean_kmh=round(float(speed[m].mean()), 2),
            ade_m=round(ade, 4), fde_m=round(fde, 4),
        )
    edges = [0, 45, 52, 60, 68, 100]
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (speed >= lo) & (speed < hi)
        if m.sum() == 0:
            continue
        _, ade, fde = per_point_rmse(pred[m], gt[m])
        out["by_speed"][f"{lo}-{hi}"] = dict(
            frames=int(m.sum()), ade_m=round(ade, 4), fde_m=round(fde, 4))
    return out


def plot_samples(src, pred, gt, frames, out_path, n=6):
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    picks = np.linspace(0, len(src) - 1, min(n, len(src))).astype(int)
    cols = 3
    rows = int(np.ceil(len(picks) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.6 * rows), squeeze=False)
    for ax, k in zip(axes.ravel(), picks):
        i = int(src[k])
        ax.plot(gt[k, :, 0], gt[k, :, 1], "o-", color="black", lw=1.4, ms=4, label="GT")
        ax.plot(pred[k, :, 0], pred[k, :, 1], "o--", color="tab:red", lw=1.4, ms=4, label="pred")
        ax.scatter([0], [0], c="tab:green", marker="s", s=30, zorder=5, label="ego")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.3)
        ax.set_xlabel("x forward (m)")
        ax.set_ylabel("y left (m)")
        ax.set_title(f"lap {int(frames.lap[i])}  {frames.speed[i]:.0f} km/h", fontsize=9)
        if k == picks[0]:
            ax.legend(fontsize=7)
    fig.suptitle("Waypoint prediction vs GT (ego frame)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_rmse(rmse, out_path, wp_dt=0.5):
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    ts = np.arange(1, len(rmse) + 1) * wp_dt
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.bar(ts, rmse, width=0.32, color="tab:blue")
    ax.set_xlabel("prediction horizon (s)")
    ax.set_ylabel("RMSE (m)")
    ax.set_title("Per-point RMSE on validation lap(s)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--npz", type=str, default=None)
    parser.add_argument("--features", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--val-laps", type=str, default=None,
                        help="覆盖 checkpoint 里的验证圈号, 如 5")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    args = parser.parse_args()

    device = get_device(args.device)
    ckpt_path = Path(args.ckpt)
    model, ckpt = load_checkpoint(ckpt_path, device=device)
    backbone = ckpt["config"]["backbone"]
    frames = Frames(Path(args.npz) if args.npz else Path(ckpt.get("frames") or resolve_default_npz()))
    val_laps = ([int(x) for x in args.val_laps.split(",")] if args.val_laps
                else ckpt.get("val_laps") or None)
    out_dir = Path(args.out_dir) if args.out_dir else \
        PIPELINE_DIR / "output" / "training" / f"predict_{backbone}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[predict] ckpt={ckpt_path} backbone={backbone} device={device}")
    print(f"[predict] data={frames.npz_path} val_laps={val_laps}")

    if backbone == "dinov3":
        cache_path = Path(args.features) if args.features else \
            Path(ckpt.get("feature_cache") or frames.npz_path.with_name(
                frames.npz_path.stem + "_features_dinov3.npz"))
        if cache_path.exists():
            cache = load_feature_cache(cache_path)
            pred, src, used_laps, kind = predict_dinov3(model, ckpt, frames, cache, val_laps,
                                                        device, args.limit)
        else:
            # 在线提特征 (需要 DINOv3 权重)
            from common import PROJECT_ROOT
            from components.dinov3 import load_encoder
            import sys
            if str(PROJECT_ROOT) not in sys.path:
                sys.path.insert(0, str(PROJECT_ROOT))
            encoder, processor = load_encoder(device=str(device))
            tr, va, used_laps = frames.lap_split(val_laps)
            if args.limit > 0:
                va = va[:args.limit]
            pred = predict_dinov3_online(encoder, processor, model, frames, va, device,
                                         ckpt.get("zero_prop", False))
            src, kind = va, "image"
    else:
        pred, src, used_laps, kind = predict_tinycnn(model, ckpt, frames, val_laps,
                                                     device, args.limit)

    gt = frames.wp[src]
    rmse, ade, fde = per_point_rmse(pred, gt)
    metrics = dict(
        ckpt=str(ckpt_path), backbone=backbone, val_laps=used_laps,
        samples=int(len(src)), input=kind,
        ade_m=round(ade, 4), fde_m=round(fde, 4),
        rmse_per_point_m=[round(float(x), 4) for x in rmse],
        groups=group_metrics(src, pred, gt, frames),
    )
    write_json(out_dir / "pred_metrics.json", metrics)

    np.savez_compressed(out_dir / "pred_arrays.npz", pred=pred, gt=gt,
                        src_index=src, speed_kmh=frames.speed[src],
                        lap=frames.lap[src], t=frames.t[src])
    with open(out_dir / "pred_per_frame.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["src_index", "t", "lap", "speed_kmh", "ade_m", "fde_m"])
        for k in range(len(src)):
            e = np.linalg.norm(pred[k] - gt[k], axis=1)
            w.writerow([int(src[k]), float(frames.t[src[k]]), int(frames.lap[src[k]]),
                        float(frames.speed[src[k]]), float(np.sqrt((e ** 2).mean())), float(e[-1])])

    plot_samples(src, pred, gt, frames, out_dir / "pred_vs_gt.png")
    plot_rmse(rmse, out_dir / "per_point_rmse.png")

    print(f"[predict] ADE={ade:.3f}m FDE={fde:.3f}m 每点RMSE="
          f"{[round(float(x), 3) for x in rmse]}")
    for lap_id, g in metrics["groups"]["by_lap"].items():
        print(f"[predict]   第{lap_id}圈: {g['frames']} 帧 {g['speed_mean_kmh']:.0f} km/h "
              f"ADE={g['ade_m']:.3f} FDE={g['fde_m']:.3f}")
    print(f"[predict] → {out_dir}")


if __name__ == "__main__":
    main()