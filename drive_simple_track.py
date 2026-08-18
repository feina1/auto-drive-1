#!/usr/bin/env python
"""Drive on the custom closed 4-lane bidirectional circuit."""
import numpy as np
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

TARGET_SPEED = TARGET_SPEED_KMH
apply_runtime_expert_patch(cruise_target_kmh=TARGET_SPEED)
EXPERT = get_expert_fn()


def _drive_action(env):
    """Use the patched runtime expert; cruise bias is applied inside the patch."""
    if env.agent.expert_takeover:
        return EXPERT(env.agent)
    return [0, 0]


def _traffic_count(env):
    return len(env.engine.traffic_manager._traffic_vehicles)


if __name__ == "__main__":
    config = dict(
        use_render=True,
        manual_control=True,
        traffic_density=0.0,
        num_scenarios=1,
        start_seed=0,
        map_region_size=4096,
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

    env = SimpleTrackEnv(config)
    origin_xy = None
    lap_counter = None
    try:
        env.reset(seed=0)
        near = spawn_nearby_traffic(env, count=14, gap_m=35.0, target_speed_kmh=TARGET_SPEED)
        loop = spawn_track_traffic(env, num_vehicles=24, target_speed_kmh=TARGET_SPEED)
        origin_xy = np.array(env.agent.position[:2], dtype=float)
        lap_counter = LapCounter(origin_xy)
        print(HELP_MESSAGE)
        gap = getattr(env.current_map, "close_gap_m", None)
        render_cfg = get_render_config()
        print(
            f"4-lane bidirectional loop ({LANE_NUM}x2), target speed {TARGET_SPEED:.0f} km/h, "
            f"backend={get_backend_label()} (runtime_config.BACKEND={BACKEND!r}), "
            f"render={render_cfg}, "
            f"traffic={_traffic_count(env)} (near={near}, loop={loop})"
            + (f", close gap ~{gap:.1f} m" if gap else "")
        )
        env.agent.expert_takeover = True
        prev_laps = 0
        while True:
            action = _drive_action(env)
            env.step(action)
            rel = np.array(env.agent.position[:2], dtype=float) - origin_xy
            laps = lap_counter.update(env.agent.position)
            if laps > prev_laps:
                env.agent.reset_navigation()
                prev_laps = laps
            env.render(
                text={
                    "Speed (km/h)": f"{env.agent.speed_km_h:.1f}",
                    "Laps": str(laps),
                    "X (m)": f"{rel[0]:.2f}",
                    "Y (m)": f"{rel[1]:.2f}",
                    "Pos (x,y)": f"({rel[0]:.2f}, {rel[1]:.2f})",
                    "Traffic": str(_traffic_count(env)),
                    "Auto-Drive (T)": "on" if env.current_track_agent.expert_takeover else "off",
                    "Control": "W,A,S,D",
                }
            )
    finally:
        env.close()
