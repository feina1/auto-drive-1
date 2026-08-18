"""Simple closed track built from straight and curve PG blocks."""

import logging
import math
import os

import cv2
import numpy as np

from metadrive.component.map.pg_map import PGMap
from metadrive.component.pg_space import Parameter
from metadrive.component.pgblock.curve import Curve
from metadrive.component.pgblock.first_block import FirstPGBlock
from metadrive.component.pgblock.straight import Straight
from metadrive.constants import DEFAULT_AGENT, TerminationState
from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.manager.pg_map_manager import PGMapManager
from metadrive.utils.draw_top_down_map import draw_top_down_map

from loop_map import MERGE_NODE, _force_close_with_blocks, _pre_close_gap
from runtime_config import get_render_config

logger = logging.getLogger(__name__)

LEFT = 0
RIGHT = 1
CURVE_TAIL = 12.0
LANE_NUM = 2  # lanes per direction; with adverse lanes => 4 lanes total
LANE_WIDTH = 3.5

# (type, ...) — straight: length; curve: radius, angle_deg, direction
TRACK_SEGMENTS = [
    # Outbound
    ("straight", 1000.0),
    ("curve", 500.0, 180.0, RIGHT),
    ("straight", 300.0),
    ("curve", 250.0, 90.0, LEFT),
    ("straight", 500.0),
    ("curve", 100.0, 90.0, LEFT),
    ("straight", 200.0),
    ("curve", 100.0, 180.0, RIGHT),
    ("straight", 300.0),
    ("curve", 100.0, 45.0, RIGHT),
    ("straight", 400.0),
    ("curve", 100.0, 45.0, LEFT),
    ("straight", 25.736),
    # Return (mirror)
    ("straight", 25.736),
    ("curve", 100.0, 45.0, LEFT),
    ("straight", 400.0),
    ("curve", 100.0, 45.0, RIGHT),
    ("straight", 300.0),
    ("curve", 100.0, 180.0, RIGHT),
    ("straight", 200.0),
    ("curve", 100.0, 90.0, LEFT),
    ("straight", 500.0),
    ("curve", 250.0, 90.0, LEFT),
    ("straight", 300.0),
    ("curve", 500.0, 180.0, RIGHT),
    ("straight", 1000.0),
]

OUTPUT_PNG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "res.png")
DEFAULT_TRAFFIC_COUNT = 28
TARGET_SPEED_KMH = 70.0
FIRST_BLOCK_LENGTH = 18.0


def estimate_track_length(include_first_block=True):
    """Approximate closed-loop driving distance in meters."""
    total = FIRST_BLOCK_LENGTH if include_first_block else 0.0
    for segment in TRACK_SEGMENTS:
        if segment[0] == "straight":
            total += float(segment[1])
        else:
            _, radius, angle, _ = segment
            total += float(radius) * math.radians(float(angle)) + CURVE_TAIL
    return total


class LapCounter:
    """Count completed laps by returning near the start after enough distance."""

    def __init__(self, origin_xy, min_lap_m=None, finish_radius=40.0):
        self.origin = np.array(origin_xy[:2], dtype=float)
        self.min_lap_m = float(min_lap_m or estimate_track_length() * 0.82)
        self.finish_radius = float(finish_radius)
        self.laps = 0
        self.travel_m = 0.0
        self.last_pos = self.origin.copy()
        self.inside_finish = True

    def update(self, position_xy):
        pos = np.array(position_xy[:2], dtype=float)
        self.travel_m += float(np.linalg.norm(pos - self.last_pos))
        self.last_pos = pos

        dist = float(np.linalg.norm(pos - self.origin))
        if dist >= self.finish_radius:
            self.inside_finish = False
        elif not self.inside_finish and self.travel_m >= self.min_lap_m:
            self.laps += 1
            self.travel_m = 0.0
            self.inside_finish = True
        return self.laps


def _iter_drivable_lanes(road_network, min_length=55.0):
    for _from, to_dict in road_network.graph.items():
        for _to, lane_list in to_dict.items():
            for lane in lane_list:
                if lane.length > min_length:
                    yield lane


def spawn_nearby_traffic(env, count=14, gap_m=35.0, target_speed_kmh=TARGET_SPEED_KMH):
    """Spawn traffic on the main straight right after the merge point."""
    from metadrive.component.pgblock.first_block import FirstPGBlock
    from metadrive.component.vehicle.vehicle_type import random_vehicle_type
    from metadrive.policy.idm_policy import IDMPolicy

    IDMPolicy.NORMAL_SPEED = float(target_speed_kmh)
    tm = env.engine.traffic_manager
    traffic_cfg = env.config["traffic_vehicle_config"].copy()
    base_cfg = env.config["vehicle_config"].copy()
    base_cfg.update(traffic_cfg)

    merge_node = FirstPGBlock.NODE_3
    candidate_lanes = []
    if merge_node in env.current_map.road_network.graph:
        for _to, lane_list in env.current_map.road_network.graph[merge_node].items():
            for lane in lane_list:
                if lane.length > 80.0:
                    candidate_lanes.append(lane)
    # Opposite-direction main straight into merge.
    for _from, to_dict in env.current_map.road_network.graph.items():
        if merge_node not in to_dict:
            continue
        for lane in to_dict[merge_node]:
            if lane.length > 80.0:
                candidate_lanes.append(lane)

    if not candidate_lanes:
        return 0

    longitudes = [gap_m * i for i in range(1, 8)]
    spawned = 0
    for lane in candidate_lanes:
        for longitude in longitudes:
            if spawned >= count:
                break
            if longitude >= lane.length - 15.0:
                continue
            vehicle_cfg = {
                **base_cfg,
                "spawn_lane_index": lane.index,
                "spawn_longitude": float(longitude),
                "spawn_lateral": 0.0,
            }
            try:
                vehicle_type = random_vehicle_type(env.np_random, [0.2, 0.3, 0.3, 0.2, 0.0])
                vehicle = tm.spawn_object(vehicle_type, vehicle_config=vehicle_cfg)
                tm.add_policy(vehicle.id, IDMPolicy, vehicle, env.engine.global_seed + 1000 + spawned)
                tm._traffic_vehicles.append(vehicle)
                spawned += 1
            except (AssertionError, TypeError, ValueError):
                continue
    logger.info("Spawned %s nearby traffic vehicles", spawned)
    return spawned


def spawn_track_traffic(env, num_vehicles=DEFAULT_TRAFFIC_COUNT, target_speed_kmh=TARGET_SPEED_KMH):
    """Spawn IDM traffic around the full closed loop."""
    from metadrive.component.vehicle.vehicle_type import random_vehicle_type
    from metadrive.policy.idm_policy import IDMPolicy

    IDMPolicy.NORMAL_SPEED = float(target_speed_kmh)
    rng = env.np_random
    lanes = list(_iter_drivable_lanes(env.current_map.road_network))
    rng.shuffle(lanes)

    traffic_cfg = env.config["traffic_vehicle_config"].copy()
    base_cfg = env.config["vehicle_config"].copy()
    base_cfg.update(traffic_cfg)

    spawned = 0
    tm = env.engine.traffic_manager
    for lane in lanes:
        if spawned >= num_vehicles:
            break
        margin = min(25.0, lane.length * 0.15)
        if lane.length <= 2 * margin + 5.0:
            continue
        longitude = float(rng.uniform(margin, lane.length - margin))
        vehicle_cfg = {
            **base_cfg,
            "spawn_lane_index": lane.index,
            "spawn_longitude": longitude,
            "spawn_lateral": 0.0,
        }
        try:
            vehicle_type = random_vehicle_type(rng, [0.2, 0.3, 0.3, 0.2, 0.0])
            vehicle = tm.spawn_object(vehicle_type, vehicle_config=vehicle_cfg)
            tm.add_policy(vehicle.id, IDMPolicy, vehicle, env.engine.global_seed + spawned)
            tm._traffic_vehicles.append(vehicle)
            spawned += 1
        except (AssertionError, TypeError, ValueError):
            continue
    logger.info("Spawned %s loop traffic vehicles (target %s)", spawned, num_vehicles)
    return spawned


class SimpleTrackMap(PGMap):
    """Closed course defined by TRACK_SEGMENTS plus a merge link back to the start."""

    CLOSE_POS_TOL = 50.0
    CLOSE_HEADING_TOL = 0.4
    FINAL_GAP_TOL = 35.0

    def _generate(self):
        parent_node_path = self.engine.worldNP
        physics_world = self.engine.physics_world
        lane_num = self.config["lane_num"]
        lane_width = self.config["lane_width"]

        first_block = FirstPGBlock(
            self.road_network,
            lane_width=lane_width,
            lane_num=lane_num,
            render_root_np=parent_node_path,
            physics_world=physics_world,
            remove_negative_lanes=False,
        )
        self.blocks.append(first_block)
        last_block = first_block
        block_index = 1

        for segment in TRACK_SEGMENTS:
            if segment[0] == "straight":
                _, length = segment
                block = Straight(
                    block_index,
                    last_block.get_socket(0),
                    self.road_network,
                    0,
                    remove_negative_lanes=False,
                    ignore_intersection_checking=True,
                )
                block.construct_from_config({Parameter.length: float(length)}, parent_node_path, physics_world)
            else:
                _, radius, angle, direction = segment
                block = Curve(
                    block_index,
                    last_block.get_socket(0),
                    self.road_network,
                    0,
                    remove_negative_lanes=False,
                    ignore_intersection_checking=True,
                )
                block.construct_from_config(
                    {
                        Parameter.length: CURVE_TAIL,
                        Parameter.radius: float(radius),
                        Parameter.angle: float(angle),
                        Parameter.dir: int(direction),
                    },
                    parent_node_path,
                    physics_world,
                )
            self.blocks.append(block)
            last_block = block
            block_index += 1

        pre_gap, pre_heading = _pre_close_gap(last_block, self.road_network)
        ok, last_block, close_blocks, gap = _force_close_with_blocks(
            last_block, self.road_network, parent_node_path, physics_world, block_index, 0
        )
        if not ok or gap > self.FINAL_GAP_TOL:
            raise RuntimeError(
                f"Failed to close track loop (pre_gap={pre_gap:.1f}m, final_gap={gap:.1f}m, "
                f"heading_err={pre_heading:.3f})"
            )
        self.blocks.extend(close_blocks)
        self.close_gap_m = float(gap)
        self.road_network.after_init()
        logger.info("Track loop closed with gap=%.1fm (pre=%.1fm)", gap, pre_gap)


class SimpleTrackMapManager(PGMapManager):
    def reset(self):
        config = self.engine.global_config.copy()
        current_seed = self.engine.global_seed
        if self.maps[current_seed] is None:
            track_map = self.spawn_object(SimpleTrackMap, map_config=config["map_config"], random_seed=None)
            self.current_map = track_map
            if config["store_map"]:
                self.maps[current_seed] = track_map
        else:
            self.current_map = self.maps[current_seed]
        self.load_map(self.current_map)


class SimpleTrackEnv(MetaDriveEnv):
    # Default terrain is 2048 m centered at origin; this track spans ~2060 m and
    # is offset in Y, so raise region size and re-center terrain on the map.
    MAP_REGION_SIZE = 4096

    def setup_engine(self):
        super().setup_engine()
        self.engine.update_manager("map_manager", SimpleTrackMapManager())

    def _post_process_config(self, config):
        config = super()._post_process_config(config)
        config["map_region_size"] = self.MAP_REGION_SIZE
        config.update(get_render_config())
        config["horizon"] = None
        config["agent_configs"][DEFAULT_AGENT]["spawn_lane_index"] = (
            FirstPGBlock.NODE_2,
            MERGE_NODE,
            0,
        )
        return config

    def reset(self, seed=None):
        obs, info = super().reset(seed=seed)
        self._loop_arrival_flags = {}
        if self.engine is not None and getattr(self.engine, "terrain", None) is not None:
            center = self.current_map.get_center_point()
            self.engine.terrain.reset(center)
            for _ in range(5):
                self.engine.graphicsEngine.renderFrame()
        return obs, info

    def _maybe_restart_loop_navigation(self, vehicle_id: str):
        """Re-plan the route after crossing the finish line so laps can continue."""
        if vehicle_id not in self.agents:
            return
        vehicle = self.agents[vehicle_id]
        if not self._is_arrive_destination(vehicle):
            self._loop_arrival_flags[vehicle_id] = False
            return
        if self._loop_arrival_flags.get(vehicle_id, False):
            return
        vehicle.reset_navigation()
        self._loop_arrival_flags[vehicle_id] = True

    def step(self, actions):
        ret = super().step(actions)
        for vehicle_id in self.agents:
            self._maybe_restart_loop_navigation(vehicle_id)
        return ret

    def done_function(self, vehicle_id: str):
        done, done_info = super().done_function(vehicle_id)
        if done_info[TerminationState.SUCCESS]:
            done = False
            done_info[TerminationState.SUCCESS] = False
        return done, done_info


def _track_summary():
    parts = []
    for segment in TRACK_SEGMENTS:
        if segment[0] == "straight":
            parts.append(f"S{segment[1]:g}m")
        else:
            _, radius, angle, direction = segment
            turn = "R" if direction == RIGHT else "L"
            parts.append(f"{turn}{angle:g}@R{radius:g}")
    return " -> ".join(parts)


def export_track_png(output_path=OUTPUT_PNG, resolution=(2048, 2048), seed=0):
    env = SimpleTrackEnv(
        dict(
            use_render=False,
            num_scenarios=1,
            traffic_density=0,
            map_config=dict(lane_num=LANE_NUM, lane_width=LANE_WIDTH),
        )
    )
    try:
        env.reset(seed=seed)
        img = draw_top_down_map(env.current_map, resolution=resolution, semantic_map=True)
        cv2.imwrite(output_path, img)
        print(f"Saved: {output_path} ({resolution[0]}x{resolution[1]})")
        print(f"Loop close gap: {getattr(env.current_map, 'close_gap_m', '?')} m")
        print(_track_summary())
        return output_path
    finally:
        env.close()


if __name__ == "__main__":
    export_track_png()
