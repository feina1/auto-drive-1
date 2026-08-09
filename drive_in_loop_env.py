#!/usr/bin/env python
"""Drive on a random closed circuit (single lane)."""
from metadrive.constants import HELP_MESSAGE
from loop_map import LoopMetaDriveEnv

if __name__ == "__main__":
    config = dict(
        use_render=True,
        manual_control=True,
        traffic_density=0.0,
        num_scenarios=5,
        start_seed=0,
        random_lane_width=False,
        random_lane_num=False,
        out_of_route_done=False,
        on_continuous_line_done=False,
        map_config=dict(lane_num=1, lane_width=4),
        vehicle_config=dict(
            show_navi_mark=False,
            show_line_to_navi_mark=False,
            show_lidar=True,
        ),
    )

    env = LoopMetaDriveEnv(config)
    try:
        o, _ = env.reset(seed=0)
        print(HELP_MESSAGE)
        print("Circuit:", getattr(env.current_map, "circuit_label", "unknown"))
        print("Change seed in reset(seed=...) for another random track.")
        env.agent.expert_takeover = True
        while True:
            env.step([0, 0])
            speed_kmh = env.agent.speed_km_h
            track_info = getattr(env.current_map, "circuit_label", "random loop")
            env.render(
                text={
                    "Speed (km/h)": f"{speed_kmh:.1f}",
                    "Track": track_info[:48],
                    "Seed": env.current_seed,
                    "Auto-Drive (T)": "on" if env.current_track_agent.expert_takeover else "off",
                    "Control": "W,A,S,D",
                }
            )
    finally:
        env.close()
