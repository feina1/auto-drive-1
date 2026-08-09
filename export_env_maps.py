#!/usr/bin/env python
"""Export top-down PNG previews for MetaDrive env maps under metadrive/envs."""

import os
import traceback

import cv2

from metadrive.envs import (
    MetaDriveEnv,
    MultiAgentBottleneckEnv,
    MultiAgentIntersectionEnv,
    MultiAgentMetaDrive,
    MultiAgentParkingLotEnv,
    MultiAgentRoundaboutEnv,
    MultiAgentTinyInter,
    MultiAgentTollgateEnv,
    SafeMetaDriveEnv,
    VaryingDynamicsEnv,
)
from metadrive.envs.legacy_envs.mixed_traffic_env import MixedTrafficEnv
from metadrive.envs.marl_envs.marl_bidirection import MultiAgentBidirectionEnv
from metadrive.envs.marl_envs.marl_racing_env import MultiAgentRacingEnv
from metadrive.envs.multigoal_intersection import MultiGoalIntersectionEnv
from metadrive.envs.top_down_env import TopDownMetaDrive
from metadrive.utils.draw_top_down_map import draw_top_down_map

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "png")
RESOLUTION = (2048, 2048)
SEED = 0

BASE = dict(use_render=False, num_scenarios=1)

ENV_SPECS = [
    ("multi_agent_metadrive", MultiAgentMetaDrive, {"num_agents": 1}),
    ("marl_racing", MultiAgentRacingEnv, {"num_agents": 1}),
    ("marl_tollgate", MultiAgentTollgateEnv, {"num_agents": 1}),
    ("marl_parking_lot", MultiAgentParkingLotEnv, {"num_agents": 1}),
    ("marl_intersection", MultiAgentIntersectionEnv, {"num_agents": 1}),
    ("marl_inout_roundabout", MultiAgentRoundaboutEnv, {"num_agents": 1}),
    ("marl_bottleneck", MultiAgentBottleneckEnv, {"num_agents": 1}),
    ("marl_bidirection", MultiAgentBidirectionEnv, {"num_agents": 1}),
    ("marl_tinyinter", MultiAgentTinyInter, {"num_agents": 1}),
    ("metadrive_default", MetaDriveEnv, {"map": 3}),
    ("metadrive_map7", MetaDriveEnv, {"map": 7}),
    ("metadrive_map_CrCS", MetaDriveEnv, {"map": "CrCS"}),
    ("safe_metadrive", SafeMetaDriveEnv, {"map": 3}),
    ("varying_dynamics", VaryingDynamicsEnv, {"map": 3}),
    ("top_down_metadrive", TopDownMetaDrive, {"map": 3}),
    ("multigoal_intersection", MultiGoalIntersectionEnv, {}),
    ("mixed_traffic", MixedTrafficEnv, {"map": 3}),
]


def export_one(name, env_cls, extra_config, seed=SEED):
    path = os.path.join(OUTPUT_DIR, f"{name}.png")
    config = {**BASE, **extra_config}
    env = env_cls(config)
    try:
        env.reset(seed=seed)
        img = draw_top_down_map(env.current_map, resolution=RESOLUTION, semantic_map=True)
        cv2.imwrite(path, img)
        print(f"OK   {name} -> {path}")
        return True
    finally:
        env.close()


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ok, fail = 0, 0
    failed = []

    for name, env_cls, extra in ENV_SPECS:
        try:
            export_one(name, env_cls, extra)
            ok += 1
        except Exception as exc:
            fail += 1
            failed.append((name, str(exc)))
            print(f"FAIL {name}: {exc}")
            traceback.print_exc(limit=1)

    # Custom loop track from auto-drive-1 (not under metadrive/envs, but part of this project).
    try:
        from loop_map import LoopMetaDriveEnv

        export_one("loop_sketch_ref", LoopMetaDriveEnv, {"map_config": dict(lane_num=1, lane_width=4)}, seed=0)
        ok += 1
    except Exception as exc:
        fail += 1
        failed.append(("loop_sketch_ref", str(exc)))
        print(f"FAIL loop_sketch_ref: {exc}")

    print(f"\nDone: {ok} saved to {OUTPUT_DIR}, {fail} failed.")
    if failed:
        print("Failed:")
        for name, err in failed:
            print(f"  - {name}: {err}")


if __name__ == "__main__":
    main()
