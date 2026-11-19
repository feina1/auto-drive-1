"""无数据自检: 验证 torch / 模型 / 损失 / 控制器 / 坐标变换 链路可用.

不依赖 MetaDrive、不依赖数据集、不依赖 GPU:
    python training/selftest.py
"""
from __future__ import annotations

import numpy as np
import torch

from common import make_prop
from model import build_model, count_params
from controllers import GeometricController


def main():
    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device("cpu")

    # 1) TinyCNN 前向 + 反向
    model = build_model("tinycnn", wp_count=6).to(device)
    total, trainable = count_params(model)
    images = torch.rand(8, 3, 72, 128)
    prop = torch.from_numpy(make_prop(np.full(8, 55.0), np.zeros(8)))
    gt = torch.zeros(8, 6, 2)
    gt[:, :, 0] = torch.arange(1, 7).float() * 15.0
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    losses = []
    for _ in range(12):
        pred = model(images, prop)
        loss = torch.nn.functional.smooth_l1_loss(pred, gt, beta=1.0)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
    assert all(np.isfinite(losses)), losses
    assert losses[-1] < losses[0], f"loss 未下降: {losses[0]} → {losses[-1]}"
    print(f"[selftest] TinyCNN ok: {total/1e3:.1f}K params, loss {losses[0]:.3f} → {losses[-1]:.3f}")

    # 2) DINOv3 head 前向 + 反向 (只测 head, 不加载权重)
    head = build_model("dinov3", wp_count=6, feat_dim=384).to(device)
    feat = torch.randn(4, 384)
    pred = head(feat, prop[:4])
    loss = torch.nn.functional.smooth_l1_loss(pred, gt[:4], beta=1.0)
    loss.backward()
    assert torch.isfinite(loss)
    print(f"[selftest] DINOv3 head ok: out={tuple(pred.shape)} loss={float(loss):.3f}")

    # 3) 控制器吃预测 waypoints
    ctrl = GeometricController()
    wp = np.array([[9.7, 0.0], [19.4, 0.1], [29.0, 0.5], [38.4, 1.2], [47.7, 2.3], [56.7, 3.9]])
    action = ctrl.compute(wp, 55.0, 0.1)
    assert len(action) == 2 and all(np.isfinite(action)), action
    print(f"[selftest] controller ok: action={np.round(action, 4).tolist()}")

    # 4) 坐标变换可逆
    from closed_loop import ego_to_world, world_to_ego

    pose0 = (10.0, 20.0, 0.4)
    back = world_to_ego(ego_to_world(wp, pose0), pose0)
    err = float(np.abs(back - wp).max())
    assert err < 1e-9, err
    print(f"[selftest] coordinate transform ok: max_err={err:.2e}")

    print("SELFTEST PASS")


if __name__ == "__main__":
    main()