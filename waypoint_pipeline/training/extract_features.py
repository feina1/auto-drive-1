"""预提取 DINOv3 视觉特征 (冻结 encoder), 供 train.py / predict.py 直接使用.

为什么先提特征:
- DINOv3 ViT-S/16 是冻结的, 同一帧反复前向没有意义;
- 5 圈 5 Hz 约 1.8 万帧, 提一次特征 (GPU 2-5 分钟) 后, 训练 head 每 epoch 只需读 384 维向量;
- 训练机无 GPU 时也可在任意有 torch+transformers 的机器上提特征后传回。

用法 (在 waypoint_pipeline 目录):
    python training/extract_features.py \
        --npz output/dataset_waypoints_5laps.npz \
        --out output/dataset_waypoints_5laps_features_dinov3.npz \
        --device cuda
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from common import Frames, PROJECT_ROOT, get_device, make_prop, set_seed, write_json
from dataset import resolve_default_npz


def main():
    from PIL import Image

    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", type=str, default=None, help="dataset_waypoints*.npz")
    parser.add_argument("--out", type=str, default=None, help="特征缓存输出路径 (.npz)")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--limit", type=int, default=0, help="只处理前 N 帧 (调试)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    set_seed(args.seed)
    npz_path = Path(args.npz) if args.npz else resolve_default_npz()
    out_path = Path(args.out) if args.out else npz_path.with_name(npz_path.stem + "_features_dinov3.npz")
    frames = Frames(npz_path)
    idx = np.where(frames.image_mask)[0]
    if args.limit > 0:
        idx = idx[:args.limit]
    if len(idx) == 0:
        raise SystemExit(f"{npz_path} 里没有带图帧, 先用 collect_trajectory.py --images 采集")

    print(f"[feat] 数据: {npz_path}  带图帧: {len(idx)} / {len(frames)}")
    print(f"[feat] 输出: {out_path}")

    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from components.dinov3 import check_weights, load_encoder

    if not check_weights():
        raise SystemExit(
            "未找到 DINOv3 权重。先执行:\n"
            "    bash components/dinov3/download_weights.sh\n"
            "或把 weights/vits16 放到 components/dinov3/weights/vits16/"
        )

    device = get_device(args.device)
    print(f"[feat] device={device}")
    encoder, processor = load_encoder(device=str(device))

    features, src_index = [], []
    t0 = time.time()
    bs = max(1, args.batch_size)
    for start in range(0, len(idx), bs):
        chunk = idx[start:start + bs]
        imgs = [Image.open(frames.image_abs(int(i))).convert("RGB") for i in chunk]
        inputs = processor(images=imgs, return_tensors="pt")
        pix = inputs["pixel_values"].to(device)
        import torch

        with torch.no_grad():
            out = encoder(pix)
        cls = out.last_hidden_state[:, 0]
        patches = out.last_hidden_state[:, 1:].mean(dim=1)
        feat = ((cls + patches) / 2.0).detach().cpu().numpy().astype(np.float32)
        features.append(feat)
        src_index.extend(int(i) for i in chunk)
        done = min(start + bs, len(idx))
        if done % (bs * 10) < bs or done == len(idx):
            speed = done / max(time.time() - t0, 1e-6)
            print(f"[feat] {done}/{len(idx)}  ({speed:.1f} 帧/s)", flush=True)

    features = np.concatenate(features, axis=0)
    src_index = np.asarray(src_index, dtype=np.int64)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        features=features,
        prop=make_prop(frames.speed[src_index], frames.steering[src_index]),
        wp=frames.wp[src_index],
        src_index=src_index,
        speed_kmh=frames.speed[src_index],
        lap=frames.lap[src_index],
        t=frames.t[src_index],
        image_path=frames.image[src_index],
        source_npz=str(npz_path).encode("utf-8"),
    )
    meta = dict(
        source_npz=str(npz_path), out=str(out_path), count=int(len(src_index)),
        feat_dim=int(features.shape[1]), f_ps=round(len(src_index) / max(time.time() - t0, 1e-6), 1),
        device=str(device),
    )
    write_json(out_path.with_suffix(".json"), meta)
    print(f"[feat] 完成: {features.shape} → {out_path} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()