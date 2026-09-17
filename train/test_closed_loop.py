"""Closed-loop test: 在 MetaDrive 中用训练好的模型开车.

用法:
    cd train/
    python test_closed_loop.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

# MetaDrive
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from metadrive.component.sensors.rgb_camera import RGBCamera
from simple_track import LANE_NUM, LANE_WIDTH, TARGET_SPEED_KMH, LapCounter, SimpleTrackEnv
from runtime_config import get_render_config

# ---- 配置 ----
TRAIN_DIR = Path(__file__).resolve().parent
PROJECT_DIR = TRAIN_DIR.parent
REF_FILE = str(PROJECT_DIR / "simple_track_reference_clean.json")
MODEL_PATH = str(TRAIN_DIR / "dinov3-weights" / "vits16")
CKPT_FILE = str(TRAIN_DIR / "checkpoints" / "best_model.pt")

CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FOV = 640, 360, 60
TARGET_SPEED = TARGET_SPEED_KMH
MAX_STEER = 0.6


def load_model(device="cuda"):
    """加载训练好的模型 (DINOv3 encoder + 训练好的 head)."""
    from dataset_fast import FastModel

    ckpt = torch.load(CKPT_FILE, map_location=device)
    cfg = ckpt["config"]
    print(f"加载 checkpoint (epoch {ckpt['epoch']}, val_l2={ckpt['val_l2']:.3f}m)")

    # 加载 encoder
    encoder = AutoModel.from_pretrained(MODEL_PATH).to(device)
    encoder.eval()

    # 加载 head
    head = FastModel(cfg["feat_dim"], cfg["wp_count"], cfg["hidden_dim"]).to(device)
    head.load_state_dict(ckpt["model_state_dict"])
    head.eval()

    print(f"✓ 模型加载成功 (val_l2={ckpt['val_l2']:.3f}m)")
    return encoder, head


def load_reference_path():
    with open(REF_FILE) as f:
        data = json.load(f)
    return np.array(data["reference_path"]["x"]), np.array(data["reference_path"]["y"])


@torch.no_grad()
def predict_waypoints(encoder, head, processor, pil_img, device):
    """从图像预测 waypoints."""
    inputs = processor(images=pil_img, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)

    outputs = encoder(pixel_values)
    cls = outputs.last_hidden_state[:, 0]
    patches = outputs.last_hidden_state[:, 1:].mean(dim=1)
    feat = ((cls + patches) / 2.0)

    pred_wp = head(feat)
    return pred_wp[0].cpu().numpy()


def waypoints_to_control(pred_wp, speed_km_h, speed_integral, lat_err=0.0):
    """从预测 waypoints 计算转向.

    策略: 加权平均多个 waypoint 的方向, 近处权重高.
    安全: 横向误差大时降速, 防止失控.
    """
    # 过滤无效 waypoint
    valid = pred_wp[:, 0] > 1.0
    if valid.sum() == 0:
        return 0.0, 0.5, speed_integral

    valid_wp = pred_wp[valid]
    n = len(valid_wp)

    # 加权平均目标角度 (近处权重高)
    weights = np.linspace(n, 1, n)  # [n, n-1, ..., 1]
    weights = weights / weights.sum()

    angles = np.arctan2(valid_wp[:, 1], valid_wp[:, 0])
    avg_angle = np.average(angles, weights=weights)

    # 转向增益: 1.5 (更保守)
    steering = np.clip(avg_angle * 1.5, -MAX_STEER, MAX_STEER)

    # 安全降速: 误差越大速度越低
    if lat_err > 3.0:
        target_speed = 20.0  # 严重偏移, 极低速
    elif lat_err > 2.0:
        target_speed = 35.0
    elif lat_err > 1.0:
        target_speed = 50.0
    else:
        target_speed = TARGET_SPEED

    # 速度 PID
    speed_error = target_speed - speed_km_h
    speed_integral += speed_error * 0.1
    speed_integral = np.clip(speed_integral, -3.0, 3.0)
    throttle = np.clip(0.15 * speed_error + 0.02 * speed_integral, 0.0, 1.0)

    return float(steering), float(throttle), speed_integral


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 60)
    print("  Closed-Loop Test: DINOv3 + Waypoint Head")
    print("=" * 60)

    encoder, head = load_model(device)
    processor = AutoImageProcessor.from_pretrained(MODEL_PATH)
    ref_x, ref_y = load_reference_path()

    config = dict(
        use_render=True, manual_control=False, traffic_density=0.0,
        num_scenarios=1, start_seed=0, map_region_size=4096,
        window_size=(480, 270), show_interface=False,
        show_logo=False, show_fps=False, camera_fov=CAMERA_FOV,
        sensors=dict(rgb_camera=(RGBCamera, CAMERA_WIDTH, CAMERA_HEIGHT)),
        image_on_cuda=True, multi_thread_render=True, render_pipeline=False,
        **{k: v for k, v in get_render_config().items() if k not in ('image_on_cuda', 'multi_thread_render', 'render_pipeline')},
        random_lane_width=False, random_lane_num=False,
        out_of_route_done=False, on_continuous_line_done=False,
        map_config=dict(lane_num=LANE_NUM, lane_width=LANE_WIDTH),
        traffic_vehicle_config=dict(
            show_navi_mark=False, show_dest_mark=False,
            show_lidar=False, show_lane_line_detector=False, show_side_detector=False,
        ),
        vehicle_config=dict(
            show_navi_mark=False, show_line_to_navi_mark=False, show_lidar=True,
            spawn_velocity=[TARGET_SPEED / 3.6, 0.0], spawn_velocity_car_frame=True,
        ),
    )

    env = SimpleTrackEnv(config)
    try:
        env.reset(seed=0)
        origin_xy = np.array(env.agent.position[:2], dtype=float)
        lap_counter = LapCounter(origin_xy)
        env.engine.force_fps.disable()

        lat_errors = []
        prev_laps = 0
        frame = 0
        speed_integral = 0.0

        print(f"\n开始闭环测试! 目标: 1 圈, 速度 {TARGET_SPEED} km/h")
        print("按 Ctrl+C 停止\n")

        while True:
            # 1. 捕获图像
            from record import _capture_front_rgb
            rgb_bgr = _capture_front_rgb(env)
            pil_img = Image.fromarray(rgb_bgr[:, :, ::-1])

            # 2. 预测 waypoints
            pred_wp = predict_waypoints(encoder, head, processor, pil_img, device)

            # 3. 计算控制 (传入当前横向误差用于安全降速)
            vx, vy = float(env.agent.position[0]), float(env.agent.position[1])
            speed = float(env.agent.speed_km_h)
            # 先算横向误差 (用于安全降速)
            closest_idx = int(np.argmin((ref_x - vx)**2 + (ref_y - vy)**2))
            lat_err = np.sqrt((ref_x[closest_idx] - vx)**2 + (ref_y[closest_idx] - vy)**2)
            steering, throttle, speed_integral = waypoints_to_control(pred_wp, speed, speed_integral, lat_err)
            env.step([steering, throttle])
            frame += 1

            # 4. 记录横向误差
            lat_errors.append(lat_err)

            # 5. 安全终止: 横向误差持续过大
            if lat_err > 10.0:
                print(f"\n✗ 横向误差 {lat_err:.1f}m > 10m, 安全终止.")
                break

            # 5. 圈数
            if lap_counter.update(env.agent.position) > prev_laps:
                prev_laps = lap_counter.laps
                print(f"\n✓ 完成第 {prev_laps} 圈!")
                if prev_laps >= 1:
                    break

            # 6. HUD (每 200 帧渲染一次, 避免双重渲染拖慢速度)
            if frame % 200 == 0:
                env.render(text={
                    "Model": "DINOv3+WP", "Speed": f"{speed:.1f}",
                    "Steer": f"{steering:.3f}", "Throttle": f"{throttle:.3f}",
                    "LatErr": f"{lat_err:.2f}m", "Laps": str(lap_counter.laps),
                    "Frame": str(frame),
                })

            if frame % 200 == 0:
                avg = np.mean(lat_errors[-200:])
                print(f"Frame {frame:5d} | Spd: {speed:5.1f} | "
                      f"Steer: {steering:6.3f} | Err: {lat_err:5.2f}m | Avg: {avg:.2f}m")

    except KeyboardInterrupt:
        print("\n用户中断.")
    finally:
        env.close()

    if lat_errors:
        arr = np.array(lat_errors)
        print(f"\n{'='*60}")
        print(f"闭环测试统计:")
        print(f"  总帧数: {len(arr)}")
        print(f"  完成圈数: {prev_laps}")
        print(f"  Mean L2: {arr.mean():.3f}m")
        print(f"  Max L2:  {arr.max():.3f}m")
        print(f"  Median:  {np.median(arr):.3f}m")
        print(f"  90%ile:  {np.percentile(arr, 90):.3f}m")
        print(f"  >2m:     {(arr>2).sum()} ({100*(arr>2).sum()/len(arr):.1f}%)")
        print(f"  >3.5m:   {(arr>3.5).sum()} ({100*(arr>3.5).sum()/len(arr):.1f}%)")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
