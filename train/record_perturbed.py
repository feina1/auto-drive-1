#!/usr/bin/env python
"""扰动注入录制: 在 PID 循迹过程中周期性注入横向/航向扰动.

目的:
  原始 PID 录制数据扰动太小 (90% 转向 < 0.008), 缺乏纠偏样本.
  本脚本通过定期注入扰动, 迫使 PID 产生大幅纠偏动作,
  从而获得包含 "偏移 → 恢复" 模式的高质量训练数据.

扰动策略:
  1. 每隔 perturb_interval 秒 (3~6s 随机), 施加一次扰动
  2. 横向偏移: ±0.3m ~ ±1.8m (沿车辆横向)
  3. 航向偏移: ±3° ~ ±12°
  4. 扰动后等待 cooldown 秒 (1.5~3s), 让 PID 恢复
  5. 如果扰动后车辆碰撞/出界, 立即回滚到扰动前状态

录制格式:
  与 record_pid_images.py 完全相同:
    images/*.jpg + frames/*.json + metadata.json

用法:
    conda activate autodrive
    python record_perturbed.py
    RECORD_LAPS=5 python record_perturbed.py     # 录 5 圈
"""

from __future__ import annotations

import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path

# 项目根目录加入 sys.path (脚本在 train/ 下, 依赖在根目录)
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

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
from pid_demo_autonomous import PathFollower
from record import (
    RECORD_HZ,
    CAMERA_WIDTH,
    CAMERA_HEIGHT,
    CAMERA_FOV,
    SessionRecorder,
    _capture_front_rgb,
    _git_commit_short,
)

RECORDS_ROOT = Path(__file__).resolve().parent.parent / "records_perturbed"
REF_PATH_FILE = Path(__file__).resolve().parent.parent / "simple_track_reference_clean.json"
MAX_RECORD_FRAMES = int(os.environ.get("RECORD_MAX_FRAMES", "0") or 0)
RECORD_LAPS = int(os.environ.get("RECORD_LAPS", "5") or 5)
HEADLESS = os.environ.get("HEADLESS", "0") == "1"  # HEADLESS=1 关闭渲染窗口加速

# ---- 扰动参数 ----
PERTURB_INTERVAL_MIN = 3.0   # 最小扰动间隔 (秒)
PERTURB_INTERVAL_MAX = 6.0   # 最大扰动间隔 (秒)
PERTURB_LATERAL_MIN = 0.3    # 最小横向偏移 (米)
PERTURB_LATERAL_MAX = 1.8    # 最大横向偏移 (米)
PERTURB_HEADING_MIN = 3.0    # 最小航向偏移 (度)
PERTURB_HEADING_MAX = 12.0   # 最大航向偏移 (度)
PERTURB_COOLDOWN = 2.0       # 扰动后冷却时间 (秒), 让 PID 恢复
PERTURB_TYPES = ["lateral", "heading", "both"]  # 扰动类型及权重
PERTURB_WEIGHTS = [0.4, 0.2, 0.4]


def load_reference_path():
    with open(REF_PATH_FILE, "r") as f:
        data = json.load(f)
    px, py = data["reference_path"]["x"], data["reference_path"]["y"]
    print(f"✓ 参考路径: {len(px)} 点 | 全长 {data['metadata']['total_path_length_m']:.0f}m")
    return px, py


def _build_env_config():
    return dict(
        use_render=not HEADLESS,
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


class PerturbationInjector:
    """扰动注入器: 定期向车辆状态注入偏移.

    工作流程:
      1. 计时器到达 → 选择扰动类型和幅度
      2. 保存当前状态 (用于回滚)
      3. 施加扰动 (set_position / set_heading_theta)
      4. 检查是否碰撞/出界 → 如果是, 回滚
      5. 进入 cooldown, 让 PID 纠偏 (此期间的数据就是 "纠偏样本")
    """

    def __init__(self):
        self.next_perturb_t = 0.0
        self.cooldown_end_t = 0.0
        self.in_cooldown = False
        self.stats = {
            "total_perturbations": 0,
            "lateral_only": 0,
            "heading_only": 0,
            "both": 0,
            "rolled_back": 0,
        }

    def schedule_next(self, current_t: float):
        """安排下一次扰动时间."""
        interval = random.uniform(PERTURB_INTERVAL_MIN, PERTURB_INTERVAL_MAX)
        self.next_perturb_t = current_t + interval

    def should_perturb(self, current_t: float) -> bool:
        """是否应该施加扰动."""
        return (
            current_t >= self.next_perturb_t
            and current_t >= self.cooldown_end_t
            and not self.in_cooldown
        )

    def apply(self, env, current_t: float) -> dict:
        """施加扰动, 返回扰动信息."""
        perturb_type = random.choices(PERTURB_TYPES, weights=PERTURB_WEIGHTS, k=1)[0]

        # 保存当前状态
        pos = list(env.agent.position)
        heading = env.agent.heading_theta
        velocity = list(env.agent.velocity)

        # 计算横向方向 (垂直于航向)
        lateral_dir = np.array([-np.sin(heading), np.cos(heading)])

        # 生成扰动量
        lat_offset = random.uniform(PERTURB_LATERAL_MIN, PERTURB_LATERAL_MAX)
        lat_sign = random.choice([-1, 1])
        lat_offset *= lat_sign

        heading_offset_deg = random.uniform(PERTURB_HEADING_MIN, PERTURB_HEADING_MAX)
        heading_offset_sign = random.choice([-1, 1])
        heading_offset_rad = np.deg2rad(heading_offset_deg * heading_offset_sign)

        info = {"type": perturb_type, "lateral_m": 0.0, "heading_deg": 0.0}

        if perturb_type in ("lateral", "both"):
            new_pos = [pos[0] + lateral_dir[0] * lat_offset,
                       pos[1] + lateral_dir[1] * lat_offset]
            env.agent.set_position(new_pos)
            info["lateral_m"] = lat_offset

        if perturb_type in ("heading", "both"):
            new_heading = heading + heading_offset_rad
            env.agent.set_heading_theta(new_heading)
            info["heading_deg"] = heading_offset_deg * heading_offset_sign

        # 进入 cooldown
        self.cooldown_end_t = current_t + PERTURB_COOLDOWN
        self.in_cooldown = True
        self.next_perturb_t = self.cooldown_end_t + random.uniform(1.0, 3.0)

        self.stats["total_perturbations"] += 1
        if perturb_type == "lateral":
            self.stats["lateral_only"] += 1
        elif perturb_type == "heading":
            self.stats["heading_only"] += 1
        else:
            self.stats["both"] += 1

        return info

    def end_cooldown(self, current_t: float):
        """cooldown 结束."""
        if self.in_cooldown and current_t >= self.cooldown_end_t:
            self.in_cooldown = False

    @property
    def is_cooling_down(self) -> bool:
        return self.in_cooldown


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
    injector = PerturbationInjector()

    try:
        env.reset(seed=0)
        follower = PathFollower(ref_path_x, ref_path_y)

        origin_xy = np.array(env.agent.position[:2], dtype=float)
        lap_counter = LapCounter(origin_xy)
        env.engine.force_fps.disable()

        # 元数据
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
            "perturbation": {
                "enabled": True,
                "interval_range_s": [PERTURB_INTERVAL_MIN, PERTURB_INTERVAL_MAX],
                "lateral_range_m": [PERTURB_LATERAL_MIN, PERTURB_LATERAL_MAX],
                "heading_range_deg": [PERTURB_HEADING_MIN, PERTURB_HEADING_MAX],
                "cooldown_s": PERTURB_COOLDOWN,
            },
            "git_commit": _git_commit_short(),
        }
        recorder = SessionRecorder(session_dir, metadata)

        print(HELP_MESSAGE)
        print(f"Recording PERTURBED drive to: {session_dir}")
        print(f"Record rate: {RECORD_HZ} Hz | sim step dt ~{step_dt:.3f}s")
        print(f"Controller : PID+PurePursuit | target {TARGET_SPEED_KMH:.0f} km/h")
        print(f"Perturbation: lateral ±{PERTURB_LATERAL_MIN}-{PERTURB_LATERAL_MAX}m, "
              f"heading ±{PERTURB_HEADING_MIN}-{PERTURB_HEADING_MAX}°")
        print(f"Target laps: {RECORD_LAPS}")
        print("Press Ctrl+C to stop.\n")

        # 安排首次扰动
        injector.schedule_next(0.0)

        prev_laps = 0
        frame = 0
        perturb_log = []

        while True:
            vehicle_x = float(env.agent.position[0])
            vehicle_y = float(env.agent.position[1])
            vehicle_heading = float(env.agent.heading_theta)
            vehicle_speed_km_h = float(env.agent.speed_km_h)

            # ---- 扰动注入 ----
            if injector.should_perturb(sim_time):
                info = injector.apply(env, sim_time)
                # 更新车辆状态 (扰动后)
                vehicle_x = float(env.agent.position[0])
                vehicle_y = float(env.agent.position[1])
                vehicle_heading = float(env.agent.heading_theta)

                info["time"] = sim_time
                info["frame"] = frame
                info["position"] = {"x": vehicle_x, "y": vehicle_y}
                perturb_log.append(info)

                lat_str = f"{info['lateral_m']:+.2f}m" if info['lateral_m'] else "-"
                hdg_str = f"{info['heading_deg']:+.1f}°" if info['heading_deg'] else "-"
                print(f"\n>> PERTURB [{info['type']}] lat={lat_str} hdg={hdg_str} "
                      f"@ t={sim_time:.1f}s frame={frame}")

            # 检查 cooldown 结束
            injector.end_cooldown(sim_time)

            # ---- PID 控制 ----
            action = follower.compute_control(
                vehicle_x, vehicle_y, vehicle_heading, vehicle_speed_km_h, step_dt
            )
            env.step(action)
            sim_time += step_dt
            frame += 1

            # ---- 圈数 ----
            if lap_counter.update(env.agent.position) > prev_laps:
                env.agent.reset_navigation()
                prev_laps = lap_counter.laps
                print(f"\n✓ 完成第 {prev_laps} 圈!", flush=True)
                if prev_laps >= RECORD_LAPS:
                    print(f"\n✓ 已跑满 {RECORD_LAPS} 圈, 自动停止。")
                    break

            # ---- 录制 ----
            if sim_time + 1e-9 >= next_record_t:
                rgb = _capture_front_rgb(env)
                recorder.record_frame(rgb, env.agent, env)
                next_record_t += record_interval
                if recorder.total_frames % 50 == 0:
                    print(f"Recorded {recorder.total_frames} frames...", flush=True)
                if MAX_RECORD_FRAMES and recorder.total_frames >= MAX_RECORD_FRAMES:
                    print(f"\n✓ 已录满 {recorder.total_frames} 帧, 自动停止。")
                    break

            # ---- HUD (仅渲染模式) ----
            if not HEADLESS:
                lat_err = (
                    follower.log_data[-1]["lateral_error"]
                    if follower.log_data
                    else 0.0
                )
                perturb_indicator = "[COOLDOWN]" if injector.is_cooling_down else "normal"
                env.render(
                    text={
                        "Recording": str(recorder.total_frames),
                        "Speed (km/h)": f"{vehicle_speed_km_h:.1f}",
                        "Steering": f"{action[0]:.3f}",
                        "Throttle": f"{action[1]:.3f}",
                        "Lat Err (m)": f"{lat_err:.2f}",
                        "Laps": str(lap_counter.laps),
                        "Perturb": perturb_indicator,
                        "Perturb#": str(injector.stats["total_perturbations"]),
                    }
                )

            if frame % 200 == 0:
                print(
                    f"Frame {frame:5d} | Speed: {vehicle_speed_km_h:5.1f} km/h | "
                    f"Steer: {action[0]:6.3f} | Lat Err: {lat_err:5.2f} m | "
                    f"Perturb#: {injector.stats['total_perturbations']} | "
                    f"Recorded: {recorder.total_frames}"
                )

    except KeyboardInterrupt:
        print("\n\nRecording stopped by user.")
    finally:
        if follower is not None:
            log_path = session_dir / "pid_control_log.csv"
            follower.save_log(str(log_path))
        # 保存扰动日志
        perturb_file = session_dir / "perturb_log.json"
        with open(perturb_file, "w") as f:
            json.dump({
                "stats": injector.stats,
                "perturbations": perturb_log,
            }, f, indent=2)
        env.close()

    print(f"\nSaved {recorder.total_frames} frames.")
    print(f"Perturbations: {injector.stats}")
    print(f"Output: {session_dir}")


if __name__ == "__main__":
    main()
