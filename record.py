#!/usr/bin/env python
"""Record ego front RGB + vehicle state while driving on SimpleTrackEnv."""

from __future__ import annotations

import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from metadrive.component.sensors.rgb_camera import RGBCamera
from metadrive.constants import HELP_MESSAGE

from runtime_config import (
    BACKEND,
    apply_runtime_expert_patch,
    get_backend_label,
    get_expert_fn,
    get_render_config,
)
from simple_track import (
    LANE_NUM,
    LANE_WIDTH,
    TARGET_SPEED_KMH,
    LapCounter,
    SimpleTrackEnv,
    spawn_nearby_traffic,
    spawn_track_traffic,
)

RECORD_HZ = 10
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 360
CAMERA_FOV = 60
JPG_QUALITY = 92
RECORDS_ROOT = Path(__file__).resolve().parent / "records"

TARGET_SPEED = TARGET_SPEED_KMH
apply_runtime_expert_patch(cruise_target_kmh=TARGET_SPEED)
EXPERT = get_expert_fn()


def _git_commit_short() -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def _split_control(vehicle):
    steering = float(vehicle.steering)
    tb = float(vehicle.throttle_brake)
    if tb >= 0.0:
        return steering, tb, 0.0
    return steering, 0.0, abs(tb)


def _vehicle_status(env, vehicle):
    offroad = bool(env._is_out_of_road(vehicle))
    collision = bool(
        vehicle.crash_vehicle
        or vehicle.crash_object
        or vehicle.crash_building
        or vehicle.crash_human
    )
    return offroad, collision


class SessionRecorder:
    """Write one frame to disk immediately (images + JSON)."""

    def __init__(self, session_dir: Path, metadata: dict):
        self.session_dir = session_dir
        self.images_dir = session_dir / "images"
        self.frames_dir = session_dir / "frames"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.frame_id = 0
        self.metadata_path = session_dir / "metadata.json"
        self._write_metadata(metadata)

    def _write_metadata(self, metadata: dict):
        with open(self.metadata_path, "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, ensure_ascii=False)
            handle.flush()
            os.fsync(handle.fileno())

    def _frame_name(self, frame_id: int) -> str:
        return f"{frame_id:06d}"

    def record_frame(self, rgb_bgr: np.ndarray, vehicle, env) -> int:
        self.frame_id += 1
        frame_id = self.frame_id
        stem = self._frame_name(frame_id)
        image_rel = f"images/{stem}.jpg"
        json_rel = f"frames/{stem}.json"
        image_path = self.session_dir / image_rel
        json_path = self.session_dir / json_rel

        ok = cv2.imwrite(
            str(image_path),
            rgb_bgr,
            [int(cv2.IMWRITE_JPEG_QUALITY), JPG_QUALITY],
        )
        if not ok:
            raise RuntimeError(f"Failed to write image: {image_path}")

        pos = vehicle.position
        steering, throttle, brake = _split_control(vehicle)
        offroad, collision = _vehicle_status(env, vehicle)
        frame_data = {
            "frame_id": frame_id,
            "image": image_rel.replace("\\", "/"),
            "position": {"x": float(pos[0]), "y": float(pos[1])},
            "heading": float(vehicle.heading_theta),
            "speed_kmh": float(vehicle.speed_km_h),
            "control": {
                "steering": steering,
                "throttle": throttle,
                "brake": brake,
            },
            "offroad": offroad,
            "collision": collision,
        }
        with open(json_path, "w", encoding="utf-8") as handle:
            json.dump(frame_data, handle, indent=2, ensure_ascii=False)
            handle.flush()
            os.fsync(handle.fileno())

        return frame_id

    @property
    def total_frames(self) -> int:
        return self.frame_id


def _build_env_config():
    return dict(
        use_render=True,
        manual_control=True,
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
            spawn_velocity=[TARGET_SPEED / 3.6, 0.0],
            spawn_velocity_car_frame=True,
        ),
    )


def _drive_action(env):
    if env.agent.expert_takeover:
        return EXPERT(env.agent)
    return [0, 0]


def _capture_front_rgb(env) -> np.ndarray:
    rgb_camera = env.engine.get_sensor("rgb_camera")
    env.engine.graphicsEngine.renderFrame()
    return rgb_camera.get_image(env.agent, mode="bgr")


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


def main():
    session_dir = _new_session_dir()
    record_time = datetime.now().strftime("%Y-%m-%d %H:%M")
    metadata = {
        "record_time": record_time,
        "record_hz": RECORD_HZ,
        "map": "SimpleTrackMap",
        "camera": {
            "width": CAMERA_WIDTH,
            "height": CAMERA_HEIGHT,
            "fov": CAMERA_FOV,
        },
        "vehicle_type": "default",
        "git_commit": _git_commit_short(),
        "backend": get_backend_label(),
        "runtime_config_backend": BACKEND,
    }
    recorder = SessionRecorder(session_dir, metadata)

    config = _build_env_config()
    step_dt = float(config.get("decision_repeat", 5) * config.get("physics_world_step_size", 0.02))
    record_interval = 1.0 / float(RECORD_HZ)
    next_record_t = 0.0
    sim_time = 0.0

    env = SimpleTrackEnv(config)
    try:
        env.reset(seed=0)
        spawn_nearby_traffic(env, count=14, gap_m=35.0, target_speed_kmh=TARGET_SPEED)
        spawn_track_traffic(env, num_vehicles=24, target_speed_kmh=TARGET_SPEED)
        lap_counter = LapCounter(np.array(env.agent.position[:2], dtype=float))
        env.agent.expert_takeover = True
        env.engine.force_fps.disable()

        print(HELP_MESSAGE)
        print(f"Recording to: {session_dir}")
        print(f"Target rate: {RECORD_HZ} Hz (sim step dt ~{step_dt:.3f}s)")
        print("Press Ctrl+C to stop.\n")

        prev_laps = 0
        while True:
            action = _drive_action(env)
            env.step(action)
            sim_time += step_dt

            if lap_counter.update(env.agent.position) > prev_laps:
                env.agent.reset_navigation()
                prev_laps = lap_counter.laps

            if sim_time + 1e-9 >= next_record_t:
                rgb = _capture_front_rgb(env)
                recorder.record_frame(rgb, env.agent, env)
                next_record_t += record_interval
                if recorder.total_frames % 50 == 0:
                    print(f"Recorded {recorder.total_frames} frames...", flush=True)

            env.render(
                text={
                    "Recording": str(recorder.total_frames),
                    "Speed (km/h)": f"{env.agent.speed_km_h:.1f}",
                    "Laps": str(lap_counter.laps),
                    "Auto-Drive (T)": "on" if env.agent.expert_takeover else "off",
                }
            )
    except KeyboardInterrupt:
        print("\nRecording stopped.")
        print(f"Saved {recorder.total_frames} frames.")
        print(f"Output: {session_dir}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
