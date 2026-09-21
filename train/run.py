#!/usr/bin/env python3
"""闭环测试: 模型预测局部路径 → PID 跟踪.

用法:
    python run.py              # 闭环测试 1 圈
    python run.py --max_frames 500  # 只跑 500 帧
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from components.dinov3 import get_model_path
from metadrive.component.sensors.rgb_camera import RGBCamera
from simple_track import LANE_NUM, LANE_WIDTH, TARGET_SPEED_KMH, LapCounter, SimpleTrackEnv
from pid_demo_autonomous import PathFollower
from record import _capture_front_rgb
from runtime_config import get_render_config
from model_v2 import WaypointHead

TRAIN_DIR = Path(__file__).resolve().parent
REF_FILE = str(PROJECT_ROOT / "simple_track_reference_lane_center.json")
MODEL_PATH = get_model_path()
CKPT_FILE = str(TRAIN_DIR / "checkpoints" / "best_model.pt")

CAMERA_W, CAMERA_H, CAMERA_FOV = 640, 360, 60
TARGET_SPEED = TARGET_SPEED_KMH
MAX_STEER = 0.6
WP_COUNT = 20


def load_model(device="cuda"):
    ckpt = torch.load(CKPT_FILE, map_location=device)
    print(f"加载 checkpoint (epoch {ckpt['epoch']}, val_l2={ckpt['val_l2']:.3f}m)")
    encoder = AutoModel.from_pretrained(MODEL_PATH).to(device)
    encoder.eval()
    processor = AutoImageProcessor.from_pretrained(MODEL_PATH)
    head = WaypointHead(**ckpt["config"]).to(device)
    head.load_state_dict(ckpt["model_state_dict"])
    head.eval()
    return encoder, processor, head


@torch.no_grad()
def predict(encoder, processor, head, pil_img, device):
    inputs = processor(images=pil_img, return_tensors="pt")
    pv = inputs["pixel_values"].to(device)
    out = encoder(pv)
    cls = out.last_hidden_state[:, 0]
    patches = out.last_hidden_state[:, 1:].mean(dim=1)
    feat = (cls + patches) / 2.0
    wp = head(feat)
    return wp[0].cpu().numpy()


def waypoints_to_pid(pred_wp, vehicle_pos, vehicle_heading, ref_x, ref_y):
    """将模型预测的 ego waypoints 转为 PID 可跟踪的世界坐标路径."""
    cos_h, sin_h = np.cos(vehicle_heading), np.sin(vehicle_heading)
    world_pts = []
    for wx, wy in pred_wp:
        # ego → world: x=前方, y=左方
        px = vehicle_pos[0] + wx * cos_h - wy * sin_h
        py = vehicle_pos[1] + wx * sin_h + wy * cos_h
        world_pts.append([px, py])
    return np.array(world_pts)


def follow_local_path(local_path_world, vehicle_pos, vehicle_heading, vehicle_speed,
                      speed_integral, target_speed=TARGET_SPEED):
    """用 Pure Pursuit 思想跟踪局部路径, 返回 [steering, throttle]."""
    if len(local_path_world) < 2:
        return 0.0, 0.3, speed_integral

    # 找最近的 waypoint
    dists = np.sqrt((local_path_world[:, 0] - vehicle_pos[0])**2 +
                    (local_path_world[:, 1] - vehicle_pos[1])**2)
    closest_idx = int(np.argmin(dists))

    # 前瞻点: 取前方 3~5m 处的 waypoint
    lookahead_dist = max(3.0, vehicle_speed / 3.6 * 0.5)  # 0.5 秒前瞻
    cum_dist = 0
    target_idx = closest_idx
    for i in range(closest_idx, len(local_path_world) - 1):
        seg = np.linalg.norm(local_path_world[i+1] - local_path_world[i])
        cum_dist += seg
        target_idx = i + 1
        if cum_dist >= lookahead_dist:
            break

    # 目标方向
    target = local_path_world[target_idx]
    dx = target[0] - vehicle_pos[0]
    dy = target[1] - vehicle_pos[1]
    target_heading = np.arctan2(dy, dx)

    # 航向误差 → 转向
    heading_err = target_heading - vehicle_heading
    # 归一化到 [-pi, pi]
    heading_err = (heading_err + np.pi) % (2 * np.pi) - np.pi
    steering = np.clip(heading_err * 2.0, -MAX_STEER, MAX_STEER)

    # 纵向 PID
    speed_err = target_speed - vehicle_speed
    speed_integral += speed_err * 0.1
    speed_integral = np.clip(speed_integral, -3.0, 3.0)
    throttle = np.clip(0.15 * speed_err + 0.02 * speed_integral, 0.0, 1.0)

    return float(steering), float(throttle), speed_integral


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_frames", type=int, default=20000)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder, processor, head = load_model(device)

    ref_x = np.array(json.load(open(REF_FILE))["reference_path"]["x"])
    ref_y = np.array(json.load(open(REF_FILE))["reference_path"]["y"])

    config = dict(
        use_render=False, manual_control=False, traffic_density=0.0,
        num_scenarios=1, start_seed=0, map_region_size=4096,
        window_size=(480, 270), show_interface=False,
        show_logo=False, show_fps=False, camera_fov=CAMERA_FOV,
        image_observation=True, norm_pixel=False,
        sensors=dict(rgb_camera=(RGBCamera, CAMERA_W, CAMERA_H)),
        image_on_cuda=True, multi_thread_render=True, render_pipeline=False,
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
            image_source="rgb_camera",
        ),
    )

    env = SimpleTrackEnv(config)
    try:
        env.reset(seed=0)
        origin_xy = np.array(env.agent.position[:2], dtype=float)
        lap_counter = LapCounter(origin_xy)
        env.engine.force_fps.disable()

        lat_errors = []
        frame = 0
        speed_integral = 0.0
        latest_local_path = None

        print(f"\n开始闭环测试! 速度 {TARGET_SPEED} km/h")
        print("按 Ctrl+C 停止\n")

        while frame < args.max_frames:
            # 1. 拍照
            img_bgr = _capture_front_rgb(env)
            pil_img = Image.fromarray(img_bgr[:, :, ::-1])

            # 2. 模型推理 (每帧)
            pred_wp = predict(encoder, processor, head, pil_img, device)

            # 3. ego → world
            vx, vy = float(env.agent.position[0]), float(env.agent.position[1])
            vh = env.agent.heading_theta
            vs = float(env.agent.speed_km_h)
            local_path_world = waypoints_to_pid(pred_wp, [vx, vy], vh, ref_x, ref_y)

            # 4. PID 跟踪
            steering, throttle, speed_integral = follow_local_path(
                local_path_world, [vx, vy], vh, vs, speed_integral)
            env.step([steering, throttle])
            frame += 1

            # 5. 计算横向误差 (与全局参考路径)
            closest_idx = int(np.argmin((ref_x - vx)**2 + (ref_y - vy)**2))
            lat_err = np.sqrt((ref_x[closest_idx] - vx)**2 + (ref_y[closest_idx] - vy)**2)
            lat_errors.append(lat_err)

            # 6. 安全终止
            if lat_err > 15.0:
                print(f"\n横向误差 {lat_err:.1f}m > 15m, 安全终止.")
                break

            # 7. 圈数检测
            if lap_counter.update(env.agent.position) > 0:
                print(f"\n完成 1 圈!")
                break

            # 8. HUD
            if frame % 200 == 0:
                avg = np.mean(lat_errors[-200:])
                print(f"Frame {frame:5d} | Spd: {vs:5.1f} | "
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
        print(f"  Mean L2: {arr.mean():.3f}m")
        print(f"  Max L2:  {arr.max():.3f}m")
        print(f"  Median:  {np.median(arr):.3f}m")
        print(f"  90%ile:  {np.percentile(arr, 90):.3f}m")
        print(f"  >2m:     {(arr>2).sum()} ({100*(arr>2).sum()/len(arr):.1f}%)")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
