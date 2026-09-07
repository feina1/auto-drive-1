#!/usr/bin/env python
"""PID 自动驾驶 + 前视 RGB 图像录制

融合两个既有脚本的实现方式:
- pid_demo_autonomous.py: 控制方式 = PathFollower (Pure Pursuit + PID)
- record.py:             录制方式 = RGBCamera 前视相机 + SessionRecorder
                          (images/*.jpg + frames/*.json + metadata.json)

说明:
- 车辆由 PID 算法全程自动驾驶, 无需 expert / 手动接管
- 按 RECORD_HZ(10Hz) 节奏录制前视 RGB 图像与车辆状态
  (每 physics step 0.1s 恰好录一帧, PID 日志与帧一一对应)

用法:
    python record_pid_images.py                    # Ctrl+C 停止
    RECORD_MAX_FRAMES=300 python record_pid_images.py  # 录满自动停止
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np

from metadrive.component.sensors.rgb_camera import RGBCamera
from metadrive.constants import HELP_MESSAGE

from runtime_config import get_render_config
from simple_track import (
    LANE_NUM,
    LANE_WIDTH,
    TARGET_SPEED_KMH,
    LapCounter,
    SimpleTrackEnv,
)

# ---- 控制方式: 复用 pid_demo_autonomous.py 的 PID/PurePursuit 控制器 ----
from pid_demo_autonomous import PathFollower

# ---- 录制方式: 复用 record.py 的录制组件 ----
from record import (
    RECORD_HZ,
    CAMERA_WIDTH,
    CAMERA_HEIGHT,
    CAMERA_FOV,
    SessionRecorder,
    _capture_front_rgb,
    _git_commit_short,
)

RECORDS_ROOT = Path(__file__).resolve().parent / "records_pid"
REF_PATH_FILE = Path(__file__).resolve().parent / "simple_track_reference_clean.json"
# 0 表示不限帧数, 一直录到 Ctrl+C 或跑满 RECORD_LAPS 圈
MAX_RECORD_FRAMES = int(os.environ.get("RECORD_MAX_FRAMES", "0") or 0)
RECORD_LAPS = int(os.environ.get("RECORD_LAPS", "3") or 3)  # 默认跑 3 圈


def load_reference_path():
    """加载与 simple_track 地图匹配的内侧车道中心线参考路径."""
    with open(REF_PATH_FILE, "r") as handle:
        data = json.load(handle)
    px = data["reference_path"]["x"]
    py = data["reference_path"]["y"]
    print(
        f"✓ 参考路径: {len(px)} 个点 | "
        f"全长 {data['metadata']['total_path_length_m']:.0f} m | {REF_PATH_FILE.name}"
    )
    return px, py


def _build_env_config():
    """环境配置: 与 record.py 相同的渲染/相机配置 (simple_track 环境)."""
    return dict(
        use_render=True,
        manual_control=False,
        traffic_density=0.0,
        num_scenarios=1,
        start_seed=0,
        map_region_size=4096,
        window_size=(960, 540),
        show_interface=False,
        show_logo=False,
        show_fps=False,
        camera_fov=CAMERA_FOV,
        sensors=dict(rgb_camera=(RGBCamera, CAMERA_WIDTH, CAMERA_HEIGHT)),
        **get_render_config(),
        random_lane_width=False,
        random_lane_num=False,
        out_of_route_done=False,
        on_continuous_line_done=False,
        map_config=dict(lane_num=LANE_NUM, lane_width=LANE_WIDTH),
        traffic_vehicle_config=dict(
            show_navi_mark=False,
            show_dest_mark=False,
            show_lidar=False,
            show_lane_line_detector=False,
            show_side_detector=False,
        ),
        vehicle_config=dict(
            show_navi_mark=False,
            show_line_to_navi_mark=False,
            show_lidar=True,
            spawn_velocity=[TARGET_SPEED_KMH / 3.6, 0.0],
            spawn_velocity_car_frame=True,
        ),
    )


def _new_session_dir() -> Path:
    """新建会话目录 records_pid/<时间戳>, 避免覆盖已有录制."""
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    session_dir = RECORDS_ROOT / stamp
    if session_dir.exists():
        suffix = 1
        while True:
            candidate = RECORDS_ROOT / f"{stamp}_{suffix:02d}"
            if not candidate.exists():
                session_dir = candidate
                break
            suffix += 1
    session_dir.mkdir(parents=True, exist_ok=False)
    return session_dir


def main():
    ref_path_x, ref_path_y = load_reference_path()

    session_dir = _new_session_dir()
    config = _build_env_config()
    step_dt = float(
        config.get("decision_repeat", 5) * config.get("physics_world_step_size", 0.02)
    )
    record_interval = 1.0 / float(RECORD_HZ)
    next_record_t = 0.0
    sim_time = 0.0

    env = SimpleTrackEnv(config)
    follower = None
    try:
        env.reset(seed=0)
        follower = PathFollower(ref_path_x, ref_path_y)

        origin_xy = np.array(env.agent.position[:2], dtype=float)
        lap_counter = LapCounter(origin_xy)
        env.engine.force_fps.disable()

        # ---- 元数据 (会话级) ----
        steer_pid = follower.steer_pid
        speed_pid = follower.speed_pid
        metadata = {
            "record_time": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "record_hz": RECORD_HZ,
            "map": "SimpleTrackMap",
            "camera": {
                "width": CAMERA_WIDTH,
                "height": CAMERA_HEIGHT,
                "fov": CAMERA_FOV,
            },
            "vehicle_type": "default",
            "controller": {
                "type": "PID+PurePursuit (PathFollower)",
                "steer_pid": {"kp": steer_pid.kp, "ki": steer_pid.ki, "kd": steer_pid.kd},
                "speed_pid": {
                    "kp": speed_pid.kp, "ki": speed_pid.ki, "kd": speed_pid.kd,
                },
                "look_dist_base": follower.look_dist_base,
                "look_dist_scale": follower.look_dist_scale,
                "max_steer": follower.max_steer,
                "target_speed_kmh": TARGET_SPEED_KMH,
            },
            "git_commit": _git_commit_short(),
        }
        recorder = SessionRecorder(session_dir, metadata)

        print(HELP_MESSAGE)
        print(f"Recording PID drive to: {session_dir}")
        print(f"Record rate: {RECORD_HZ} Hz | sim step dt ~{step_dt:.3f}s")
        print(f"Controller : PID+PurePursuit | target {TARGET_SPEED_KMH:.0f} km/h")
        print(f"Target laps: {RECORD_LAPS}")
        if MAX_RECORD_FRAMES:
            print(f"Max frames : {MAX_RECORD_FRAMES} (RECORD_MAX_FRAMES)")
        print("Press Ctrl+C to stop.\n")

        prev_laps = 0
        frame = 0
        while True:
            vehicle_x = float(env.agent.position[0])
            vehicle_y = float(env.agent.position[1])
            vehicle_heading = float(env.agent.heading_theta)
            vehicle_speed_km_h = float(env.agent.speed_km_h)

            action = follower.compute_control(
                vehicle_x, vehicle_y, vehicle_heading, vehicle_speed_km_h, step_dt
            )
            env.step(action)
            sim_time += step_dt
            frame += 1

            if lap_counter.update(env.agent.position) > prev_laps:
                env.agent.reset_navigation()
                prev_laps = lap_counter.laps
                print(f"\n✓ 完成第 {prev_laps} 圈!", flush=True)
                if prev_laps >= RECORD_LAPS:
                    print(f"\n✓ 已跑满 {RECORD_LAPS} 圈, 自动停止。")
                    break

            # ---- 按录制节奏落盘一帧 (图片 + json) ----
            if sim_time + 1e-9 >= next_record_t:
                rgb = _capture_front_rgb(env)
                recorder.record_frame(rgb, env.agent, env)
                next_record_t += record_interval
                if recorder.total_frames % 50 == 0:
                    print(f"Recorded {recorder.total_frames} frames...", flush=True)
                if MAX_RECORD_FRAMES and recorder.total_frames >= MAX_RECORD_FRAMES:
                    print(f"\n✓ 已录满 {recorder.total_frames} 帧, 自动停止。")
                    break

            lat_err = (
                follower.log_data[-1]["lateral_error"]
                if follower.log_data
                else 0.0
            )
            env.render(
                text={
                    "Recording": str(recorder.total_frames),
                    "Speed (km/h)": f"{vehicle_speed_km_h:.1f}",
                    "Steering": f"{action[0]:.3f}",
                    "Throttle": f"{action[1]:.3f}",
                    "Lat Err (m)": f"{lat_err:.2f}",
                    "Laps": str(lap_counter.laps),
                    "Control": "PID+PurePursuit",
                }
            )

            if frame % 200 == 0:
                print(
                    f"Frame {frame:5d} | Speed: {vehicle_speed_km_h:5.1f} km/h | "
                    f"Steer: {action[0]:6.3f} | Throttle: {action[1]:6.3f} | "
                    f"Lat Err: {lat_err:5.2f} m | Recorded: {recorder.total_frames}"
                )
    except KeyboardInterrupt:
        print("\n\nRecording stopped by user.")
    finally:
        if follower is not None:
            log_path = session_dir / "pid_control_log.csv"
            follower.save_log(str(log_path))
        env.close()

    print(f"Saved {recorder.total_frames} frames.")
    print(f"Output: {session_dir}")


if __name__ == "__main__":
    main()
