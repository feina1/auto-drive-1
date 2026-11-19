"""闭环测试: 相机图像 → 模型 → 6 点 waypoints → 几何控制器, 在 MetaDrive 里实车跑圈.

与 validate_controller.py 的区别:
- validate_controller 喂的是 GT waypoints, 验证控制器本身;
- 本脚本喂模型预测, 验证"感知→waypoints→控制"整条链路。

用法 (本机无 GPU, 用 TinyCNN 冒烟):
    python training/closed_loop.py --ckpt output/training/tinycnn_best.pt \
        --speed 40 --laps 1 --max-seconds 30 --device cpu

训练机 (DINOv3):
    python training/closed_loop.py --ckpt output/training/dinov3_best.pt \
        --speed 56 --laps 1 --device cuda
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch

from common import TRAINING_DIR, get_device, write_json
import config
import traj_utils as tu
from controllers import GeometricController
from model import load_checkpoint


def ego_to_world(wp_ego, pose):
    """ego 系 [K,2] → 世界系, pose=(x,y,heading)."""
    x, y, h = pose
    c, s = np.cos(h), np.sin(h)
    wx = x + c * wp_ego[:, 0] - s * wp_ego[:, 1]
    wy = y + s * wp_ego[:, 0] + c * wp_ego[:, 1]
    return np.stack([wx, wy], axis=1)


def world_to_ego(pts, pose):
    x, y, h = pose
    c, s = np.cos(h), np.sin(h)
    dx, dy = pts[:, 0] - x, pts[:, 1] - y
    return np.stack([c * dx + s * dy, -s * dx + c * dy], axis=1)


def transform_cached_wp(wp_ego, pose_pred, pose_now):
    """把预测时刻的 ego waypoints 变换到当前 ego 系 (推理跳帧时用)."""
    return world_to_ego(ego_to_world(wp_ego, pose_pred), pose_now)


class Predictor:
    """统一封装 tinycnn / dinov3 两种 checkpoint."""

    def __init__(self, ckpt_path, device):
        self.device = device
        self.model, self.ckpt = load_checkpoint(ckpt_path, device=device)
        self.backbone = self.ckpt["config"]["backbone"]
        self.zero_prop = bool(self.ckpt.get("zero_prop", False))
        self.image_size = tuple(self.ckpt["config"].get("image_size", (128, 72)))
        self.encoder = self.processor = None
        if self.backbone == "dinov3":
            from components.dinov3 import load_encoder

            self.encoder, self.processor = load_encoder(device=str(device))
        print(f"[closed] 模型: {self.backbone} epoch={self.ckpt.get('epoch')} "
              f"val_ade={self.ckpt.get('val', {}).get('ade_m', float('nan')):.3f}m")

    @torch.no_grad()
    def predict(self, pil_img, speed_kmh, steering):
        prop = torch.tensor([[speed_kmh / 72.0, steering / 0.6]],
                            dtype=torch.float32, device=self.device)
        if self.zero_prop:
            prop = prop * 0.0
        if self.backbone == "dinov3":
            from components.dinov3 import extract_features

            feat = extract_features(self.encoder, self.processor, [pil_img]).to(self.device)
            wp = self.model(feat, prop)
        else:
            from PIL import Image

            img = pil_img.convert("RGB")
            if img.size != self.image_size:
                img = img.resize(self.image_size, Image.BILINEAR)
            arr = np.asarray(img, dtype=np.float32) / 255.0
            image = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
            wp = self.model(image, prop)
        return wp[0].detach().cpu().numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--speed", type=float, default=56.0, help="目标巡航速度 km/h")
    parser.add_argument("--laps", type=int, default=1)
    parser.add_argument("--max-seconds", type=float, default=0.0, help=">0 时限制时长 (冒烟)")
    parser.add_argument("--infer-every", type=int, default=1,
                        help="每 N 个控制 tick 推理一次, 中间用位姿变换复用 (CPU 可设 2-3)")
    parser.add_argument("--out-dir", type=str, default=str(config.OUT_DIR / "training" / "closed_loop"))
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    from simple_track import LapCounter

    device = get_device(args.device)
    predictor = Predictor(args.ckpt, device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ref_x, ref_y = tu.load_reference_path()
    env = tu.build_headless_env(with_camera=True, spawn_speed_kmh=args.speed)
    controller = GeometricController()
    tracker = tu.LateralErrorTracker(ref_x, ref_y)
    log = []
    lap_times = []
    prev_laps = 0
    lap_start_t = 0.0
    sim_t = 0.0
    tail_until = None
    infer_count = 0
    infer_time = 0.0
    wp_base = None          # 预测时刻 ego 系 waypoints (不随 tick 变换)
    wp_now = None           # 变换到当前 ego 系的 waypoints
    pose_pred = None
    aborted = None

    try:
        env.reset(seed=args.seed)
        origin = np.array(env.agent.position[:2], dtype=float)
        lap_counter = LapCounter(origin)
        print(f"[closed] 开始: 目标 {args.speed:.0f} km/h, {args.laps} 圈, "
              f"infer_every={args.infer_every}, device={device}")

        while True:
            v = env.agent
            pose_now = (float(v.position[0]), float(v.position[1]), float(v.heading_theta))

            if wp_base is None or len(log) % max(1, args.infer_every) == 0:
                t0 = time.time()
                bgr = tu.grab_rgb(env, v)
                from PIL import Image

                pil = Image.fromarray(bgr[:, :, ::-1])  # BGR → RGB
                wp_base = predictor.predict(pil, float(v.speed_km_h), float(v.steering))
                pose_pred = pose_now
                infer_time += time.time() - t0
                infer_count += 1
            wp_now = (wp_base if pose_pred == pose_now
                      else transform_cached_wp(wp_base, pose_pred, pose_now))

            action = controller.compute(wp_now, float(v.speed_km_h), config.SIM_DT)
            env.step(action)
            sim_t += config.SIM_DT

            v = env.agent
            pos = (float(v.position[0]), float(v.position[1]))
            offroad, collision = tu.vehicle_status(env, v)
            lat = tracker.signed_error(*pos)
            log.append(dict(
                t=round(sim_t, 3), x=pos[0], y=pos[1], heading=float(v.heading_theta),
                speed_kmh=float(v.speed_km_h), steering=float(v.steering),
                throttle_brake=float(v.throttle_brake),
                lat_err_m=round(float(lat), 4), offroad=bool(offroad),
                collision=bool(collision), wp0_x=float(wp_now[0, 0]),
                wp0_y=float(wp_now[0, 1]), lap=int(prev_laps + 1),
            ))

            laps = lap_counter.update(pos)
            if laps > prev_laps:
                lap_times.append(round(sim_t - lap_start_t, 2))
                lap_start_t = sim_t
                prev_laps = laps
                env.agent.reset_navigation()
                print(f"[closed] 完成第 {laps} 圈 @ {sim_t:.1f}s ({lap_times[-1]:.1f}s), "
                      f"均速 {np.mean([r['speed_kmh'] for r in log[-int(lap_times[-1]*10):]]):.1f} km/h")
                if laps >= args.laps:
                    tail_until = sim_t + 1.0
                else:
                    tail_until = None

            if len(log) % 500 == 0:
                print(f"[closed] t={sim_t:7.1f}s speed={v.speed_km_h:5.1f} "
                      f"lat={lat:+.2f}m wp0=({wp_now[0,0]:.1f},{wp_now[0,1]:.1f}) "
                      f"lap={laps + 1}", flush=True)

            if collision:
                aborted = "collision"
                print(f"[closed] 碰撞 @ t={sim_t:.1f}s, 停止")
                break
            if tail_until is not None and sim_t >= tail_until:
                break
            if args.max_seconds > 0 and sim_t >= args.max_seconds:
                aborted = aborted or "max_seconds"
                break
            if sim_t > 1300.0 * args.laps + 30.0:
                aborted = "timeout"
                break
    finally:
        env.close()

    lat_abs = np.array([abs(r["lat_err_m"]) for r in log])
    speed = np.array([r["speed_kmh"] for r in log])
    metrics = dict(
        ckpt=str(args.ckpt), backbone=predictor.backbone, target_speed_kmh=args.speed,
        laps_done=int(prev_laps), lap_times_s=lap_times,
        completed=bool(prev_laps >= args.laps), aborted=aborted,
        frames=len(log), duration_s=round(sim_t, 2),
        lat_abs_mean_m=round(float(lat_abs.mean()), 4) if len(log) else None,
        lat_abs_p95_m=round(float(np.percentile(lat_abs, 95)), 4) if len(log) else None,
        lat_abs_max_m=round(float(lat_abs.max()), 4) if len(log) else None,
        offroad_ratio=round(float(np.mean([r["offroad"] for r in log])), 5) if log else None,
        collision_frames=int(sum(r["collision"] for r in log)),
        speed_mean_kmh=round(float(speed.mean()), 2) if len(log) else None,
        speed_min_kmh=round(float(speed.min()), 2) if len(log) else None,
        infer_count=infer_count,
        infer_ms=round(1000.0 * infer_time / max(infer_count, 1), 2),
    )
    write_json(out_dir / "closed_loop_metrics.json", metrics)
    with open(out_dir / "closed_loop_log.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(log[0].keys()) if log else ["t"])
        w.writeheader()
        w.writerows(log)

    if log:
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        axes[0].plot([r["x"] for r in log], [r["y"] for r in log], lw=1.2, label="model drive")
        axes[0].set_aspect("equal", adjustable="datalim")
        axes[0].set_title("trajectory")
        axes[0].set_xlabel("x (m)")
        axes[0].set_ylabel("y (m)")
        axes[0].grid(alpha=0.3)
        axes[1].plot([r["t"] for r in log], [r["lat_err_m"] for r in log], lw=0.8)
        axes[1].set_title("signed lateral error")
        axes[1].set_xlabel("t (s)")
        axes[1].set_ylabel("m")
        axes[1].grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "closed_loop.png", dpi=140)
        plt.close(fig)

    print("[closed] 结果: " + " ".join(
        f"{k}={metrics[k]}" for k in ("completed", "frames", "lat_abs_mean_m",
                                      "lat_abs_max_m", "offroad_ratio",
                                      "collision_frames", "infer_ms")))
    print(f"[closed] → {out_dir}")


if __name__ == "__main__":
    main()