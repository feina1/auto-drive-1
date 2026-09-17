#!/usr/bin/env python3
"""数据收集: 随机偏移 → PID 恢复 → 记录 (图片, GT waypoints).

用法:
    python collect_data.py --num 50       # 50 个样本 (快速验证)
    python collect_data.py --num 2000     # 2000 个样本 (正式)
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from metadrive.component.sensors.rgb_camera import RGBCamera
from simple_track import LANE_NUM, LANE_WIDTH, TARGET_SPEED_KMH, SimpleTrackEnv
from pid_demo_autonomous import PathFollower
from runtime_config import get_render_config

REF_FILE = str(PROJECT_ROOT / "simple_track_reference_clean.json")
DATA_DIR = Path(__file__).resolve().parent / "data"
IMG_DIR = DATA_DIR / "images"
GT_DIR = DATA_DIR / "gt"

CAMERA_W, CAMERA_H, CAMERA_FOV = 640, 360, 60
TARGET_SPEED = TARGET_SPEED_KMH
WP_COUNT = 20
WP_INTERVAL = 1.0


def load_ref_path():
    with open(REF_FILE) as f:
        data = json.load(f)
    x = np.array(data["reference_path"]["x"])
    y = np.array(data["reference_path"]["y"])
    return x, y


def compute_headings(x, y):
    h = np.zeros(len(x))
    for i in range(len(x) - 1):
        h[i] = np.arctan2(y[i+1] - y[i], x[i+1] - x[i])
    h[-1] = h[-2]
    return h


def world_to_ego(pts, ego_xy, ego_heading):
    cos_h, sin_h = np.cos(ego_heading), np.sin(ego_heading)
    ego = np.zeros((len(pts), 2))
    for i, (px, py) in enumerate(pts):
        dx, dy = px - ego_xy[0], py - ego_xy[1]
        ego[i, 0] = dx * cos_h + dy * sin_h
        ego[i, 1] = -dx * sin_h + dy * cos_h
    return ego


def resample_traj(traj, interval=1.0, count=20):
    """沿轨迹等间距采样 count 个点, 从 interval 米开始."""
    if len(traj) < 3:
        return None
    traj = np.array(traj)
    diffs = np.diff(traj, axis=0)
    seg_len = np.sqrt((diffs**2).sum(axis=1))
    cum = np.concatenate([[0], np.cumsum(seg_len)])
    total = cum[-1]
    if total < (count * interval * 0.3):
        return None
    targets = np.arange(1, count + 1) * interval
    sampled = np.zeros((count, 2))
    for i, td in enumerate(targets):
        if td > total:
            if i == 0:
                return None
            d = (traj[-1] - traj[-2]) / (np.linalg.norm(traj[-1] - traj[-2]) + 1e-10)
            sampled[i] = sampled[i-1] + interval * d
        else:
            idx = np.searchsorted(cum, td, side='right') - 1
            idx = min(idx, len(traj) - 2)
            t = (td - cum[idx]) / (seg_len[idx] + 1e-10)
            sampled[i] = traj[idx] + np.clip(t, 0, 1) * (traj[idx+1] - traj[idx])
    return sampled


def make_env():
    config = dict(
        use_render=False, manual_control=False, traffic_density=0.0,
        num_scenarios=1, start_seed=0, map_region_size=4096,
        window_size=(320, 180), show_interface=False,
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
            image_source="rgb_camera",
        ),
    )
    return SimpleTrackEnv(config)


def collect_one(env, ref_x, ref_y, ref_headings, sample_id):
    """收集一个样本: 随机偏移 → 拍照 → PID 恢复 → 返回数据."""
    # 1. 随机选参考路径上的位置 (跳过起终点)
    idx = np.random.randint(200, len(ref_x) - 200)
    base_x, base_y = ref_x[idx], ref_y[idx]
    base_h = ref_headings[idx]

    # 2. 随机偏移
    lat_off = np.random.uniform(-5.0, 5.0)
    hdg_off = np.random.uniform(-30, 30) * np.pi / 180

    # 法向量 (左侧 90°)
    nx, ny = -np.sin(base_h), np.cos(base_h)
    spawn_x = base_x + lat_off * nx
    spawn_y = base_y + lat_off * ny
    spawn_h = base_h + hdg_off

    # 3. 初始化环境
    env.reset(seed=0)
    vehicle = env.agent
    vehicle.set_position([spawn_x, spawn_y])
    vehicle.set_heading_theta(spawn_h)
    vehicle.set_velocity([TARGET_SPEED / 3.6 * np.cos(spawn_h),
                          TARGET_SPEED / 3.6 * np.sin(spawn_h)])

    # 走几步让物理稳定
    for _ in range(3):
        obs, _, _, _, _ = env.step([0.0, 0.3])

    # 4. 拍照 — 从 observation 获取图像
    if env.config["image_on_cuda"]:
        # CUDA tensor: 需要 .get() 转为 numpy, 取最后一帧 (stack_size)
        img_raw = obs["image"].get()  # numpy array
        if img_raw.ndim == 3:
            img_bgr = img_raw[..., :3]  # RGB → take first 3 channels
        else:
            img_bgr = img_raw[..., -1, :3]  # stacked: take last frame
    else:
        img_raw = obs["image"]
        if img_raw.ndim == 3:
            img_bgr = img_raw[..., :3]
        else:
            img_bgr = img_raw[..., -1, :3]
    img_rgb = img_bgr[:, :, ::-1].copy()  # BGR → RGB
    photo_x, photo_y = float(vehicle.position[0]), float(vehicle.position[1])
    photo_h = vehicle.heading_theta
    actual_x, actual_y, actual_h = photo_x, photo_y, photo_h

    # 5. PID 恢复
    follower = PathFollower(ref_x, ref_y)
    trajectory = [(actual_x, actual_y)]
    recovered = False
    dt = 0.1

    for step in range(80):
        control = follower.compute_control(actual_x, actual_y, actual_h,
                                           float(vehicle.speed_km_h), dt)
        env.step(control)
        actual_x, actual_y = float(vehicle.position[0]), float(vehicle.position[1])
        actual_h = vehicle.heading_theta
        trajectory.append((actual_x, actual_y))

        # 检查是否恢复 (放宽到 0.5m)
        closest_idx = int(np.argmin((ref_x - actual_x)**2 + (ref_y - actual_y)**2))
        lat_err = np.sqrt((ref_x[closest_idx] - actual_x)**2 + (ref_y[closest_idx] - actual_y)**2)
        if lat_err < 0.5 and step > 10:
            recovered = True
            break

        # 安全检查
        if lat_err > 15.0:
            break

    # 6. 生成 GT — 只取恢复段 (前 30 步), 避免 PID 收敛后变成正常行驶
    recovery_traj = trajectory[:31]  # 包含起点, 共 30 步
    gt_world = resample_traj(recovery_traj, interval=WP_INTERVAL, count=WP_COUNT)
    if gt_world is None:
        return None

    # 用拍照时的位置/朝向做 ego 坐标系
    gt_ego = world_to_ego(gt_world, [photo_x, photo_y], photo_h)

    # 初始偏移量 (用于统计)
    closest_to_spawn = int(np.argmin((ref_x - spawn_x)**2 + (ref_y - spawn_y)**2))
    init_lat_err = np.sqrt((ref_x[closest_to_spawn] - spawn_x)**2 +
                           (ref_y[closest_to_spawn] - spawn_y)**2)

    return {
        "image": img_rgb,
        "gt_ego": gt_ego,
        "meta": {
            "sample_id": sample_id,
            "ref_idx": int(idx),
            "lateral_offset": float(lat_off),
            "heading_offset_deg": float(hdg_off * 180 / np.pi),
            "recovered": recovered,
            "traj_len_m": float(np.sum(np.sqrt(np.sum(np.diff(np.array(trajectory), axis=0)**2, axis=1)))),
        }
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num", type=int, default=2000)
    parser.add_argument("--start", type=int, default=0)
    args = parser.parse_args()

    IMG_DIR.mkdir(parents=True, exist_ok=True)
    GT_DIR.mkdir(parents=True, exist_ok=True)

    ref_x, ref_y = load_ref_path()
    ref_headings = compute_headings(ref_x, ref_y)

    print(f"=" * 60)
    print(f"  数据收集: 目标 {args.num} 个样本 (从 #{args.start} 开始)")
    print(f"  输出: {DATA_DIR}")
    print(f"=" * 60)

    env = make_env()
    manifest = []
    success = 0
    fail = 0

    # 加载已有 manifest
    manifest_file = DATA_DIR / "manifest.json"
    if manifest_file.exists() and args.start > 0:
        with open(manifest_file) as f:
            manifest = json.load(f)
        success = len(manifest)
        print(f"  已有 {success} 个样本, 继续收集...")

    try:
        for i in range(args.start, args.num):
            t0 = time.time()
            result = collect_one(env, ref_x, ref_y, ref_headings, i)
            elapsed = time.time() - t0

            if result is None:
                fail += 1
                print(f"  [{i}] FAIL (轨迹太短) ({elapsed:.1f}s)")
                continue

            # 保存图片
            img_path = IMG_DIR / f"sample_{i:05d}.jpg"
            Image.fromarray(result["image"]).save(img_path, quality=90)

            # 保存 GT
            gt_path = GT_DIR / f"sample_{i:05d}.json"
            gt_data = {
                "waypoints": result["gt_ego"].tolist(),
                "meta": result["meta"],
            }
            with open(gt_path, "w") as f:
                json.dump(gt_data, f, indent=1)

            manifest.append({
                "sample_id": i,
                "image": str(img_path),
                "gt": str(gt_path),
                **result["meta"],
            })
            success += 1

            # 每 10 个保存一次 manifest
            if i % 10 == 0:
                with open(manifest_file, "w") as f:
                    json.dump(manifest, f, indent=1)

            lat_off = result["meta"]["lateral_offset"]
            recovered = "OK" if result["meta"]["recovered"] else "FAIL"
            print(f"  [{i}] lat={lat_off:+.1f}m rec={recovered} ({elapsed:.1f}s) "
                  f"success={success} fail={fail}")

    except KeyboardInterrupt:
        print("\n用户中断.")
    finally:
        # 保存 manifest
        with open(manifest_file, "w") as f:
            json.dump(manifest, f, indent=1)
        env.close()

    print(f"\n完成: success={success}, fail={fail}")
    print(f"数据保存在: {DATA_DIR}")


if __name__ == "__main__":
    main()
