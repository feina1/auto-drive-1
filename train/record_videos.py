#!/usr/bin/env python3
"""录制闭环测试视频 (V1 PID + V3 模型).

用法:
    python record_videos.py
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import cv2
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from metadrive.component.sensors.rgb_camera import RGBCamera
from simple_track import LANE_NUM, LANE_WIDTH, TARGET_SPEED_KMH, LapCounter, SimpleTrackEnv
from pid_demo_autonomous import PathFollower
from record import _capture_front_rgb
from runtime_config import get_render_config
from model_v2 import WaypointHead

TRAIN_DIR = PROJECT_ROOT / "train"
REF_FILE = str(PROJECT_ROOT / "simple_track_reference_clean.json")
MODEL_PATH = str(TRAIN_DIR / "dinov3-weights" / "vits16")
CKPT_FILE = str(TRAIN_DIR / "checkpoints" / "best_model.pt")
VIDEO_DIR = PROJECT_ROOT / "videos"
VIDEO_DIR.mkdir(exist_ok=True)

CAMERA_W, CAMERA_H, CAMERA_FOV = 640, 360, 60
TARGET_SPEED = TARGET_SPEED_KMH
MAX_STEER = 0.6
WP_COUNT = 20
FPS = 10


def make_env_config():
    return dict(
        use_render=False, manual_control=False, traffic_density=0.0,
        num_scenarios=1, start_seed=0, map_region_size=4096,
        window_size=(480, 270), show_interface=False,
        show_logo=False, show_fps=False, camera_fov=CAMERA_FOV,
        image_observation=True, norm_pixel=False,
        image_on_cuda=True, multi_thread_render=True, render_pipeline=False,
        random_lane_width=False, random_lane_num=False,
        out_of_route_done=False, on_continuous_line_done=False,
        map_config=dict(lane_num=LANE_NUM, lane_width=LANE_WIDTH),
        sensors=dict(rgb_camera=(RGBCamera, CAMERA_W, CAMERA_H)),
        traffic_vehicle_config=dict(
            show_navi_mark=False, show_dest_mark=False,
            show_lidar=False, show_lane_line_detector=False, show_side_detector=False,
        ),
        vehicle_config=dict(
            show_navi_mark=False, show_line_to_navi_mark=False, show_lidar=True,
            spawn_velocity=[TARGET_SPEED / 3.6, 0.0], spawn_velocity_car_frame=True,
        ),
    )


def add_text_overlay(frame, text_lines):
    """在帧上添加文字信息."""
    h, w = frame.shape[:2]
    # 半透明背景
    overlay = frame.copy()
    cv2.rectangle(overlay, (5, 5), (280, 20 + len(text_lines) * 22), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
    for i, line in enumerate(text_lines):
        cv2.putText(frame, line, (10, 22 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


# ============================================================
# V1: PID 纯控制器
# ============================================================
def record_v1(max_frames=3000):
    print("\n" + "="*60)
    print("录制 V1: PID Baseline")
    print("="*60)

    config = make_env_config()
    env = SimpleTrackEnv(config)
    env.reset(seed=0)

    ref_path = json.load(open(REF_FILE))["reference_path"]
    ref_x = np.array(ref_path["x"])
    ref_y = np.array(ref_path["y"])
    follower = PathFollower(ref_x, ref_y)

    video_path = str(VIDEO_DIR / "v1_pid_baseline.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(video_path, fourcc, FPS, (CAMERA_W, CAMERA_H))

    origin_xy = np.array(env.agent.position[:2], dtype=float)
    lap_counter = LapCounter(origin_xy)
    env.engine.force_fps.disable()

    frame = 0
    lat_errors = []
    try:
        while frame < max_frames:
            pos = env.agent.position
            heading = env.agent.heading_theta
            speed = env.agent.speed_km_h

            steering, throttle = follower.compute_control(
                float(pos[0]), float(pos[1]), heading, float(speed), dt=0.1)
            env.step([steering, throttle])

            img_bgr = _capture_front_rgb(env)
            # 添加文字
            closest_idx = int(np.argmin((ref_x - pos[0])**2 + (ref_y - pos[1])**2))
            lat_err = np.sqrt((ref_x[closest_idx] - pos[0])**2 + (ref_y[closest_idx] - pos[1])**2)
            lat_errors.append(lat_err)

            text = [
                f"V1: PID Baseline",
                f"Frame: {frame}  Speed: {speed:.0f} km/h",
                f"Lat Error: {lat_err:.2f}m",
            ]
            img_with_text = add_text_overlay(img_bgr.copy(), text)
            # RGB -> BGR for cv2
            writer.write(img_with_text[:, :, ::-1] if img_with_text.shape[2] == 3 else img_with_text)

            frame += 1
            if lat_err > 15.0:
                break
            if lap_counter.update(env.agent.position) > 0:
                print("完成 1 圈!")
                break
            if frame % 500 == 0:
                print(f"  V1 frame {frame}, lat_err={lat_err:.2f}m")
    finally:
        writer.release()
        env.close()

    arr = np.array(lat_errors)
    print(f"V1 视频已保存: {video_path}")
    print(f"V1 统计: {len(arr)} 帧, Mean L2={arr.mean():.3f}m")
    return len(arr), arr.mean()


# ============================================================
# V3: DINOv3 模型
# ============================================================
def load_model(device="cuda"):
    ckpt = torch.load(CKPT_FILE, map_location=device)
    print(f"加载 checkpoint (epoch {ckpt['epoch']}, val_l2={ckpt['val_l2']:.3f}m)")
    from transformers import AutoImageProcessor, AutoModel
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


def waypoints_to_pid(pred_wp, vehicle_pos, vehicle_heading):
    cos_h, sin_h = np.cos(vehicle_heading), np.sin(vehicle_heading)
    world_pts = []
    for wx, wy in pred_wp:
        px = vehicle_pos[0] + wx * cos_h - wy * sin_h
        py = vehicle_pos[1] + wx * sin_h + wy * cos_h
        world_pts.append([px, py])
    return np.array(world_pts)


def follow_local_path(local_path_world, vehicle_pos, vehicle_heading, vehicle_speed,
                      speed_integral, target_speed=TARGET_SPEED):
    if len(local_path_world) < 2:
        return 0.0, 0.3, speed_integral
    dists = np.sqrt((local_path_world[:, 0] - vehicle_pos[0])**2 +
                    (local_path_world[:, 1] - vehicle_pos[1])**2)
    closest_idx = int(np.argmin(dists))
    lookahead_dist = max(3.0, vehicle_speed / 3.6 * 0.5)
    cum_dist = 0
    target_idx = closest_idx
    for i in range(closest_idx, len(local_path_world) - 1):
        seg = np.linalg.norm(local_path_world[i+1] - local_path_world[i])
        cum_dist += seg
        target_idx = i + 1
        if cum_dist >= lookahead_dist:
            break
    target = local_path_world[target_idx]
    dx = target[0] - vehicle_pos[0]
    dy = target[1] - vehicle_pos[1]
    target_heading = np.arctan2(dy, dx)
    heading_err = target_heading - vehicle_heading
    heading_err = (heading_err + np.pi) % (2 * np.pi) - np.pi
    steering = np.clip(heading_err * 2.0, -MAX_STEER, MAX_STEER)
    speed_err = target_speed - vehicle_speed
    speed_integral += speed_err * 0.1
    speed_integral = np.clip(speed_integral, -3.0, 3.0)
    throttle = np.clip(0.15 * speed_err + 0.02 * speed_integral, 0.0, 1.0)
    return float(steering), float(throttle), speed_integral


def record_v3(max_frames=2000):
    print("\n" + "="*60)
    print("录制 V3: DINOv3 Model (2000 samples)")
    print("="*60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder, processor, head = load_model(device)

    ref_data = json.load(open(REF_FILE))["reference_path"]
    ref_x = np.array(ref_data["x"])
    ref_y = np.array(ref_data["y"])

    config = make_env_config()
    env = SimpleTrackEnv(config)
    env.reset(seed=0)

    video_path = str(VIDEO_DIR / "v3_dinov3_2000samples.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(video_path, fourcc, FPS, (CAMERA_W, CAMERA_H))

    origin_xy = np.array(env.agent.position[:2], dtype=float)
    lap_counter = LapCounter(origin_xy)
    env.engine.force_fps.disable()

    frame = 0
    speed_integral = 0.0
    lat_errors = []
    try:
        while frame < max_frames:
            img_bgr = _capture_front_rgb(env)
            pil_img = Image.fromarray(img_bgr[:, :, ::-1])

            pred_wp = predict(encoder, processor, head, pil_img, device)
            vx, vy = float(env.agent.position[0]), float(env.agent.position[1])
            vh = env.agent.heading_theta
            vs = float(env.agent.speed_km_h)
            local_path_world = waypoints_to_pid(pred_wp, [vx, vy], vh)

            steering, throttle, speed_integral = follow_local_path(
                local_path_world, [vx, vy], vh, vs, speed_integral)
            env.step([steering, throttle])

            closest_idx = int(np.argmin((ref_x - vx)**2 + (ref_y - vy)**2))
            lat_err = np.sqrt((ref_x[closest_idx] - vx)**2 + (ref_y[closest_idx] - vy)**2)
            lat_errors.append(lat_err)

            text = [
                f"V3: DINOv3 Model",
                f"Frame: {frame}  Speed: {vs:.0f} km/h",
                f"Lat Error: {lat_err:.2f}m  Mean: {np.mean(lat_errors):.2f}m",
            ]
            img_with_text = add_text_overlay(img_bgr.copy(), text)
            writer.write(img_with_text[:, :, ::-1] if img_with_text.shape[2] == 3 else img_with_text)

            frame += 1
            if lat_err > 15.0:
                print(f"  横向误差 {lat_err:.1f}m > 15m, 终止")
                break
            if lap_counter.update(env.agent.position) > 0:
                print("  完成 1 圈!")
                break
            if frame % 200 == 0:
                print(f"  V3 frame {frame}, lat_err={lat_err:.2f}m")
    finally:
        writer.release()
        env.close()

    arr = np.array(lat_errors)
    print(f"V3 视频已保存: {video_path}")
    print(f"V3 统计: {len(arr)} 帧, Mean L2={arr.mean():.3f}m")
    return len(arr), arr.mean()


if __name__ == "__main__":
    print("开始录制视频...")
    v1_frames, v1_l2 = record_v1(max_frames=3000)
    v3_frames, v3_l2 = record_v3(max_frames=2000)
    print("\n" + "="*60)
    print("视频录制完成!")
    print(f"  V1: {v1_frames} 帧, Mean L2={v1_l2:.3f}m")
    print(f"  V3: {v3_frames} 帧, Mean L2={v3_l2:.3f}m")
    print(f"  视频目录: {VIDEO_DIR}")
    print("="*60)
