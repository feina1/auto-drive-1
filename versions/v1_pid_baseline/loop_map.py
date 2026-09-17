"""Simple closed track built from straight and curve PG blocks."""

import logging
import math
import os

import cv2
import numpy as np

from metadrive.component.lane.straight_lane import StraightLane
from metadrive.component.map.pg_map import PGMap
from metadrive.component.pg_space import Parameter
from metadrive.component.pgblock.curve import Curve
from metadrive.component.pgblock.first_block import FirstPGBlock
from metadrive.component.pgblock.straight import Straight
from metadrive.constants import DEFAULT_AGENT, PGLineType, TerminationState
from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.manager.pg_map_manager import PGMapManager
from metadrive.utils.draw_top_down_map import draw_top_down_map
from metadrive.utils.math import wrap_to_pi
from metadrive.utils.pg.utils import get_lanes_bounding_box

from runtime_config import get_render_config

logger = logging.getLogger(__name__)

MERGE_NODE = FirstPGBlock.NODE_3  # ">>>"
CLOSE_POS_TOL = 18.0
CLOSE_HEADING_TOL = 0.38
CROSS_SAMPLE_STEP = 2.0
NEIGHBOR_ENDPOINT_TOL = 7.5


def _heading_diff(a, b):
    return abs(wrap_to_pi(a - b))


def _collect_lanes(road_network):
    lanes = []
    for to_dict in road_network.graph.values():
        for lane_list in to_dict.values():
            lanes.extend(lane_list)
    return lanes


def _lane_hits_lane(lane_a, lane_b, sample_step=CROSS_SAMPLE_STEP):
    if lane_a is lane_b:
        return False
    x_max_1, x_min_1, y_max_1, y_min_1 = get_lanes_bounding_box([lane_a])
    x_max_2, x_min_2, y_max_2, y_min_2 = get_lanes_bounding_box([lane_b])
    if x_min_1 > x_max_2 or x_min_2 > x_max_1 or y_min_1 > y_max_2 or y_min_2 > y_max_1:
        return False
    margin = max(1.5, sample_step)
    lat_pad = 0.8
    for i in np.arange(margin, max(margin + 1, lane_a.length - margin), sample_step):
        pt = lane_a.position(float(i), 0)
        lon, lat = lane_b.local_coordinates(pt)
        if 0 <= lon <= lane_b.length and abs(lat) <= lane_b.width_at(lon) / 2.0 + lat_pad:
            return True
    return False


def _lanes_are_neighbors(lane_a, lane_b, tol=NEIGHBOR_ENDPOINT_TOL):
    pts_a = [np.array(lane_a.start, dtype=float), np.array(lane_a.end, dtype=float)]
    pts_b = [np.array(lane_b.start, dtype=float), np.array(lane_b.end, dtype=float)]
    for pa in pts_a:
        for pb in pts_b:
            if float(np.linalg.norm(pa - pb)) < tol:
                return True
    for pa in pts_a:
        lon, lat = lane_b.local_coordinates(pa)
        if 0 <= lon <= lane_b.length and abs(lat) <= lane_b.width_at(lon) / 2.0 + 1.2:
            return True
    for pb in pts_b:
        lon, lat = lane_a.local_coordinates(pb)
        if 0 <= lon <= lane_a.length and abs(lat) <= lane_a.width_at(lon) / 2.0 + 1.2:
            return True
    return False


def _network_has_crossings(road_network, new_lane_start=0, ignore_lanes=None):
    ignore = {id(lane) for lane in (ignore_lanes or [])}
    lanes = _collect_lanes(road_network)
    if new_lane_start <= 0:
        for i, lane_a in enumerate(lanes):
            for lane_b in lanes[i + 1:]:
                if id(lane_a) in ignore or id(lane_b) in ignore:
                    continue
                if _lanes_are_neighbors(lane_a, lane_b):
                    continue
                if _lane_hits_lane(lane_a, lane_b) or _lane_hits_lane(lane_b, lane_a):
                    return True
        return False

    old_lanes = lanes[:new_lane_start]
    new_lanes = lanes[new_lane_start:]
    for lane_a in new_lanes:
        if id(lane_a) in ignore:
            continue
        for lane_b in old_lanes:
            if id(lane_b) in ignore:
                continue
            if _lanes_are_neighbors(lane_a, lane_b):
                continue
            if _lane_hits_lane(lane_a, lane_b) or _lane_hits_lane(lane_b, lane_a):
                return True
        for lane_b in new_lanes:
            if lane_a is lane_b or id(lane_b) in ignore:
                continue
            if _lanes_are_neighbors(lane_a, lane_b):
                continue
            if _lane_hits_lane(lane_a, lane_b) or _lane_hits_lane(lane_b, lane_a):
                return True
    return False


def _lane_crosses_network(lane, road_network):
    for other in _collect_lanes(road_network):
        if other is lane:
            continue
        if _lanes_are_neighbors(lane, other):
            continue
        if _lane_hits_lane(lane, other) or _lane_hits_lane(other, lane):
            return True
    return False


def _lane_end_pose(lane):
    return np.array(lane.end, dtype=float), lane.heading_theta_at(lane.length)


def _entry_pose(road_network):
    merge_lane = road_network.graph[FirstPGBlock.NODE_2][MERGE_NODE][0]
    return np.array(merge_lane.end, dtype=float), merge_lane.heading_theta_at(merge_lane.length)


def _pre_close_gap(last_block, road_network):
    last_road = last_block.get_socket(0).positive_road
    last_lane = road_network.graph[last_road.start_node][last_road.end_node][0]
    end_pos, end_heading = _lane_end_pose(last_lane)
    entry_pos, entry_heading = _entry_pose(road_network)
    gap = float(np.linalg.norm(end_pos - entry_pos))
    heading_gap = _heading_diff(end_heading, entry_heading)
    return gap, heading_gap


def _attach_merge_link(last_block, road_network, lane_width):
    last_road = last_block.get_socket(0).positive_road
    last_lane = road_network.graph[last_road.start_node][last_road.end_node][0]
    end_pos, end_heading = _lane_end_pose(last_lane)
    entry_pos, entry_heading = _entry_pose(road_network)
    gap = float(np.linalg.norm(end_pos - entry_pos))
    heading_gap = _heading_diff(end_heading, entry_heading)
    if gap > CLOSE_POS_TOL or heading_gap > CLOSE_HEADING_TOL:
        return False, gap, heading_gap

    closing_lane = StraightLane(
        end_pos,
        entry_pos,
        width=lane_width,
        line_types=(PGLineType.BROKEN, PGLineType.SIDE),
    )
    if closing_lane.length < 0.5:
        outgoing = list(road_network.graph[MERGE_NODE].keys())
        if not outgoing:
            return False, gap, heading_gap
        closing_lane = road_network.graph[MERGE_NODE][outgoing[0]][0]
    elif _lane_crosses_network(closing_lane, road_network):
        return False, gap, heading_gap

    road_network.add_lane(last_road.end_node, MERGE_NODE, closing_lane)
    return True, gap, heading_gap


def _force_close_with_blocks(last_block, road_network, render_np, physics_world, block_index, block_seed):
    """Add up to 4 closing blocks to align with the merge point."""
    current = last_block
    idx = block_index
    extra_blocks = []

    for _ in range(4):
        last_road = current.get_socket(0).positive_road
        lane_width = road_network.graph[last_road.start_node][last_road.end_node][0].width
        ok, gap, heading_gap = _attach_merge_link(current, road_network, lane_width)
        if ok:
            return True, current, extra_blocks, gap

        last_lane = road_network.graph[last_road.start_node][last_road.end_node][0]
        end_pos, end_heading = _lane_end_pose(last_lane)
        entry_pos, _entry_heading = _entry_pose(road_network)
        delta = entry_pos - end_pos
        dist = float(np.linalg.norm(delta))
        turn = wrap_to_pi(math.atan2(delta[1], delta[0]) - end_heading)

        if abs(turn) > 0.15:
            angle = min(max(abs(turn) * 180 / math.pi, 25), 135)
            cfg = {
                Parameter.length: float(max(25, min(45, dist * 0.25))),
                Parameter.radius: float(max(28, min(55, dist * 0.35))),
                Parameter.angle: float(angle),
                Parameter.dir: 0 if turn > 0 else 1,
            }
            block = Curve(
                idx,
                current.get_socket(0),
                road_network,
                block_seed,
                remove_negative_lanes=True,
                ignore_intersection_checking=True,
            )
            block.construct_from_config(cfg, render_np, physics_world)
            if _network_has_crossings(road_network):
                return False, current, extra_blocks, dist
            extra_blocks.append(block)
            current = block
            idx += 1
            continue

        if dist > 5:
            cfg = {Parameter.length: float(min(85, max(20, dist * 0.85)))}
            block = Straight(
                idx,
                current.get_socket(0),
                road_network,
                block_seed,
                remove_negative_lanes=True,
                ignore_intersection_checking=True,
            )
            block.construct_from_config(cfg, render_np, physics_world)
            if _network_has_crossings(road_network):
                return False, current, extra_blocks, dist
            extra_blocks.append(block)
            current = block
            idx += 1
            continue

        return False, current, extra_blocks, dist

    return False, current, extra_blocks, 9999


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
