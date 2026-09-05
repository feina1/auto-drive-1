#!/usr/bin/env python
"""Drive on the custom closed 4-lane bidirectional circuit."""
from datetime import datetime

import numpy as np
from metadrive.component.sensors.rgb_camera import RGBCamera
from metadrive.constants import HELP_MESSAGE

from record import (
    CAMERA_FOV,
    CAMERA_HEIGHT,
    CAMERA_WIDTH,
    RECORD_HZ,
    SessionRecorder,
    _capture_front_rgb,
    _git_commit_short,
    _new_session_dir,
)
from runtime_config import (
    BACKEND,
    apply_runtime_expert_patch,
    get_backend_label,
    get_expert_fn,
    get_render_config,
)
from loop_map import (
    LANE_NUM,
    LANE_WIDTH,
    TARGET_SPEED_KMH,
    LapCounter,
    SimpleTrackEnv,
)

TARGET_SPEED = TARGET_SPEED_KMH
apply_runtime_expert_patch(cruise_target_kmh=TARGET_SPEED)
EXPERT = get_expert_fn()


def _drive_action(env):
    """Use the patched runtime expert; cruise bias is applied inside the patch."""
    if env.agent.expert_takeover:
        return EXPERT(env.agent)
    return [0, 0]


if __name__ == "__main__":
    config = dict(
        use_render=True,
        manual_control=True,
        traffic_density=0.0,
        num_scenarios=1,
        start_seed=0,
        map_region_size=4096,
        camera_fov=CAMERA_FOV,
        sensors=dict(rgb_camera=(RGBCamera, CAMERA_WIDTH, CAMERA_HEIGHT)),
        **get_render_config(),
        random_lane_width=False,
        random_lane_num=False,
        out_of_route_done=False,
        on_continuous_line_done=False,
        map_config=dict(lane_num=LANE_NUM, lane_width=LANE_WIDTH),
        vehicle_config=dict(
            show_navi_mark=False,
            show_line_to_navi_mark=False,
            show_lidar=True,
            spawn_velocity=[TARGET_SPEED / 3.6, 0.0],
            spawn_velocity_car_frame=True,
        ),
    )

    env = SimpleTrackEnv(config)
    step_dt = float(config.get("decision_repeat", 5) * config.get("physics_world_step_size", 0.02))
    record_interval = 1.0 / float(RECORD_HZ)
    next_record_t = 0.0
    sim_time = 0.0
    origin_xy = None
    lap_counter = None
    recorder = None
    try:
        env.reset(seed=0)
        session_dir = _new_session_dir()
        recorder = SessionRecorder(
            session_dir,
            {
                "record_time": datetime.now().strftime("%Y-%m-%d %H:%M"),
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
            },
        )

        origin_xy = np.array(env.agent.position[:2], dtype=float)
        lap_counter = LapCounter(origin_xy)
        print(HELP_MESSAGE)
        gap = getattr(env.current_map, "close_gap_m", None)
        render_cfg = get_render_config()
        print(
            f"4-lane bidirectional loop ({LANE_NUM}x2), target speed {TARGET_SPEED:.0f} km/h, "
            f"backend={get_backend_label()} (runtime_config.BACKEND={BACKEND!r}), "
            f"render={render_cfg}, "
            + (f"close gap ~{gap:.1f} m" if gap else "")
        )
        print(f"Recording to: {session_dir} @ {RECORD_HZ} Hz")
        env.agent.expert_takeover = True
        prev_laps = 0
        while True:
            action = _drive_action(env)
            env.step(action)
            sim_time += step_dt
            if sim_time + 1e-9 >= next_record_t:
                recorder.record_frame(_capture_front_rgb(env), env.agent, env)
                next_record_t += record_interval
            rel = np.array(env.agent.position[:2], dtype=float) - origin_xy
            laps = lap_counter.update(env.agent.position)
            if laps > prev_laps:
                env.agent.reset_navigation()
                prev_laps = laps
            env.render(
                text={
                    "Recording": str(recorder.total_frames),
                    "Speed (km/h)": f"{env.agent.speed_km_h:.1f}",
                    "Laps": str(laps),
                    "X (m)": f"{rel[0]:.2f}",
                    "Y (m)": f"{rel[1]:.2f}",
                    "Pos (x,y)": f"({rel[0]:.2f}, {rel[1]:.2f})",
                    "Auto-Drive (T)": "on" if env.current_track_agent.expert_takeover else "off",
                    "Control": "W,A,S,D",
                }
            )
    finally:
        env.close()
