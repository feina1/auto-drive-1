"""Random closed circuit tracks for single-agent MetaDrive demos."""

import logging
import math

import numpy as np

from metadrive.component.lane.circular_lane import CircularLane
from metadrive.component.lane.straight_lane import StraightLane
from metadrive.component.map.pg_map import PGMap
from metadrive.component.pg_space import Parameter
from metadrive.component.pgblock.curve import Curve
from metadrive.component.pgblock.first_block import FirstPGBlock
from metadrive.component.pgblock.pg_block import PGBlock
from metadrive.component.pgblock.straight import Straight
from metadrive.component.pg_space import ParameterSpace
from metadrive.component.road_network import Road
from metadrive.constants import DEFAULT_AGENT, PGLineType, TerminationState
from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.manager.pg_map_manager import PGMapManager
from metadrive.utils.pg.utils import get_lanes_bounding_box
from metadrive.utils.math import Vector, wrap_to_pi

logger = logging.getLogger(__name__)

MERGE_NODE = FirstPGBlock.NODE_3  # ">>>"
CLOSE_POS_TOL = 18.0
CLOSE_HEADING_TOL = 0.38
CROSS_SAMPLE_STEP = 2.0
NEIGHBOR_ENDPOINT_TOL = 7.5


def _heading_diff(a, b):
    return abs(wrap_to_pi(a - b))


def _scaled_side_lengths(rng, n_corners, lo=48.0, hi=165.0):
    """Random side lengths with bounded perimeter for easier loop closure."""
    perimeter = float(rng.uniform(680.0, 980.0))
    weights = rng.uniform(0.55, 1.45, size=n_corners)
    weights /= weights.sum()
    sides = [perimeter * float(w) for w in weights]
    sides = [float(np.clip(s, lo, hi)) for s in sides]
    scale = perimeter / sum(sides)
    return [float(np.clip(s * scale, lo, hi)) for s in sides]


def _split_length(total, parts, rng, minimum=12.0):
    """Split a straight side into several random segments with exact total length."""
    total = float(total)
    parts = max(1, int(parts))
    if parts == 1:
        return [total]

    remaining = total - minimum * parts
    if remaining <= 0:
        return [total / parts] * parts

    weights = rng.uniform(0.5, 1.5, size=parts)
    weights /= weights.sum()
    lengths = [minimum + remaining * w for w in weights]
    lengths[-1] = total - sum(lengths[:-1])
    return [float(max(minimum, l)) for l in lengths]


def _curve_item(rng, angle, turn_dir, label, radius=None, tail=None):
    angle = float(max(15.0, min(180.0, angle)))
    cfg = {
        Parameter.length: float(tail if tail is not None else rng.uniform(4, 18)),
        Parameter.radius: float(radius if radius is not None else rng.uniform(24, 62)),
        Parameter.angle: angle,
        Parameter.dir: turn_dir,
    }
    return Curve, cfg, label


def _turn_label(turn_dir):
    return "L" if int(turn_dir) == 0 else "R"


def _random_corner_angles(rng, n_corners):
    """Corner deflections (degrees) summing to 360 for a same-direction closed loop."""
    angles = rng.uniform(40.0, 110.0, size=n_corners)
    angles = angles / angles.sum() * 360.0
    for _ in range(32):
        angles = np.clip(angles, 28.0, 148.0)
        delta = 360.0 - float(angles.sum())
        if abs(delta) < 0.05:
            break
        free = (angles > 28.05) & (angles < 147.95)
        if not np.any(free):
            break
        share = free.astype(float)
        share /= share.sum()
        angles[free] += delta * share[free]
    angles[-1] += 360.0 - float(angles.sum())
    return [float(a) for a in angles]


def _chicane_items(rng, side_length, force=False):
    """S-bend on a straight: one left + one right, zero net heading change."""
    side_length = float(side_length)
    if side_length < 52.0:
        return []
    if not force and rng.random() > 0.48:
        return []
    bend = float(rng.uniform(14.0, min(34.0, side_length * 0.12)))
    radius = float(rng.uniform(34.0, 68.0))
    lead = max(12.0, side_length * float(rng.uniform(0.20, 0.36)))
    trail = max(12.0, side_length - lead - radius * bend * math.pi / 90.0 * 2.5)
    if trail < 12.0:
        lead = side_length * 0.3
        trail = side_length * 0.3
    first_dir = int(rng.choice([0, 1]))
    second_dir = 1 - first_dir
    return [
        _straight_item(lead),
        _curve_item(rng, bend, first_dir, _turn_label(first_dir), radius=radius, tail=rng.uniform(4, 12)),
        _curve_item(rng, bend, second_dir, _turn_label(second_dir), radius=radius, tail=rng.uniform(4, 12)),
        _straight_item(trail),
    ]


def _append_side(plan, rng, side_length, chicane_quota=None):
    side_length = float(side_length)
    if chicane_quota is not None and chicane_quota[0] > 0 and side_length >= 52.0:
        chicane = _chicane_items(rng, side_length, force=True)
        if chicane:
            chicane_quota[0] -= 1
            plan.extend(chicane)
            return
    if chicane_quota is None:
        chicane = _chicane_items(rng, side_length, force=False)
        if chicane:
            plan.extend(chicane)
            return
    for length in _split_length(side_length, int(rng.randint(2, 6)), rng, minimum=10.0):
        plan.append(_straight_item(length))


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
        check_new = lanes
        check_old = lanes
        for i, lane_a in enumerate(check_new):
            for lane_b in check_old[i + 1:]:
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


def _straight_item(length):
    return Straight, {Parameter.length: float(length)}, "S"


def _corner_items(rng, angle, turn_dir, wild=False):
    angle = float(angle)
    label = _turn_label(turn_dir)
    split_prob = 0.82 if wild else 0.58
    if angle >= 48.0 and rng.random() < split_prob:
        a1 = float(rng.uniform(max(18.0, angle * 0.22), min(88.0, angle * 0.78)))
        a2 = angle - a1
        r1 = float(rng.uniform(22, 64 if wild else 52))
        r2 = float(rng.uniform(22, 64 if wild else 52))
        return [
            _curve_item(rng, a1, turn_dir, label, radius=r1, tail=rng.uniform(3, 18)),
            _curve_item(rng, a2, turn_dir, label, radius=r2, tail=rng.uniform(3, 18)),
        ]
    radius = float(rng.uniform(24, 68 if wild else 54))
    return [_curve_item(rng, angle, turn_dir, label, radius=radius)]


def _circuit_plan(rng):
    """Wild irregular loop: same-direction corners + S-chicane straights (L and R)."""
    n_corners = int(rng.randint(8, 14))
    target = int(rng.randint(38, 58))
    turn_dir = int(rng.choice([0, 1]))
    corner_angles = _random_corner_angles(rng, n_corners)
    sides = _scaled_side_lengths(rng, n_corners, lo=48.0, hi=155.0)
    chicane_quota = [int(rng.randint(2, 4))]

    plan = []
    for side_idx in range(n_corners):
        _append_side(plan, rng, sides[side_idx], chicane_quota=chicane_quota)
        plan.extend(_corner_items(rng, corner_angles[side_idx], turn_dir, wild=True))

    while len(plan) < target:
        straight_indexes = [i for i, item in enumerate(plan) if item[2] == "S"]
        if not straight_indexes:
            break
        idx = int(rng.choice(straight_indexes))
        _, cfg, _ = plan[idx]
        old_len = float(cfg[Parameter.length])
        if old_len < 28:
            break
        left = old_len / 2.0
        plan[idx:idx + 1] = [_straight_item(left), _straight_item(old_len - left)]

    return plan, f"{n_corners}-gon"


def _minimal_irregular_plan(rng):
    n_corners = int(rng.choice([7, 8, 9, 10]))
    turn_dir = int(rng.choice([0, 1]))
    corner_angles = _random_corner_angles(rng, n_corners)
    sides = [float(rng.uniform(52, 95)) for _ in range(n_corners)]
    chicane_quota = [1]
    plan = []
    for side_idx in range(n_corners):
        _append_side(plan, rng, sides[side_idx], chicane_quota=chicane_quota)
        plan.extend(_corner_items(rng, corner_angles[side_idx], turn_dir, wild=False))
    return plan, f"{n_corners}-gon-basic"


def _fixed_straight(length):
    return Straight, {Parameter.length: float(length)}, "S"


def _fixed_curve(angle, turn_dir, radius=42.0, tail=8.0):
    label = _turn_label(turn_dir)
    return Curve, {
        Parameter.length: float(tail),
        Parameter.radius: float(radius),
        Parameter.angle: float(angle),
        Parameter.dir: int(turn_dir),
    }, label


def _symmetric_side_items(rng, total_length, n_parts=None, minimum=10.0):
    """Split one straight edge into random segments (same edge can vary)."""
    total_length = float(total_length)
    if n_parts is None:
        n_parts = int(rng.randint(2, 5))
    return [_straight_item(length) for length in _split_length(total_length, n_parts, rng, minimum=minimum)]


def _symmetric_chicane_items(rng, side_length, flipped=False):
    """S-bend on a straight edge: one left + one right, zero net heading change."""
    side_length = float(side_length)
    if side_length < 50.0:
        return _symmetric_side_items(rng, side_length)

    bend = float(rng.uniform(12.0, min(24.0, side_length * 0.08)))
    radius = float(rng.uniform(40.0, 58.0))
    lead_ratio = float(rng.uniform(0.24, 0.32))
    lead = max(12.0, side_length * lead_ratio)
    bend_cost = radius * bend * math.pi / 180.0 * 2.0
    trail = side_length - lead - bend_cost
    if trail < 12.0:
        return _symmetric_side_items(rng, side_length)

    first_dir = 1 if flipped else 0
    second_dir = 1 - first_dir
    return [
        _straight_item(lead),
        _curve_item(rng, bend, first_dir, _turn_label(first_dir), radius=radius, tail=6.0),
        _curve_item(rng, bend, second_dir, _turn_label(second_dir), radius=radius, tail=6.0),
        _straight_item(trail),
    ]


def _symmetric_corner_angles(rng, base_angle=90.0):
    """Four equal corners for reliable loop closure; micro jitter stays on same edge only."""
    base = float(base_angle * rng.uniform(0.97, 1.03))
    return [base, base, base, base]


def _symmetric_circuit_plan(rng, start_length=18.0):
    """
    Left-right symmetric closed loop.

    - Four edges (bottom / right / top / left) with matched opposite lengths.
    - Randomness lives in segment splits and mirrored S-chicane shape, not opposite totals.
    - Vertical limbs carry mirrored S-chicanes so the track includes both L and R.
    """
    scale = float(rng.uniform(0.92, 1.08))
    bottom_total = float(rng.uniform(88.0, 132.0)) * scale
    bottom = max(24.0, bottom_total - float(start_length))
    top = bottom_total
    right = float(rng.uniform(72.0, 112.0)) * scale
    left = right

    turn_dir = int(rng.choice([0, 1]))
    corner_angles = _symmetric_corner_angles(rng, base_angle=float(rng.uniform(88.0, 92.0)))

    plan = []
    plan.extend(_symmetric_side_items(rng, bottom, n_parts=int(rng.randint(2, 5))))
    plan.extend(_corner_items(rng, corner_angles[0], turn_dir, wild=rng.random() < 0.35))

    plan.extend(_symmetric_chicane_items(rng, right, flipped=False))
    plan.extend(_corner_items(rng, corner_angles[1], turn_dir, wild=rng.random() < 0.35))

    # Same total length as bottom, different random split pattern on the opposite edge.
    plan.extend(_symmetric_side_items(rng, top, n_parts=int(rng.randint(2, 6))))
    plan.extend(_corner_items(rng, corner_angles[2], turn_dir, wild=rng.random() < 0.35))

    plan.extend(_symmetric_chicane_items(rng, left, flipped=True))
    plan.extend(_corner_items(rng, corner_angles[3], turn_dir, wild=rng.random() < 0.35))

    return plan, "symmetric"


def _sketch_reference_plan(rng):
    """
    Hand-drawn reference loop: rounded rectangle with a right-side inward bay.

    Topology (clockwise):
      top -> top-right -> upper right -> bay (L / hairpin R / L) ->
      lower right -> bottom -> bottom-left -> left side -> top-left.
    """
    scale = float(rng.uniform(0.94, 1.06))
    s = scale
    top = 125.0 * s
    right_upper = 32.0 * s
    right_lower = 48.0 * s
    bay_depth = 46.0 * s
    left_side = 220.0 * s
    r_outer = 45.0 * s
    r_bay = 38.0 * s
    r_hairpin = 32.0 * s

    plan = [
        _fixed_straight(top),
        _fixed_curve(90, 1, r_outer, 8),
        _fixed_straight(right_upper),
        _fixed_curve(90, 0, r_bay, 8),
        _fixed_straight(bay_depth),
        _fixed_curve(180, 1, r_hairpin, 6),
        _fixed_straight(bay_depth),
        _fixed_curve(90, 0, r_bay, 8),
        _fixed_straight(right_lower),
        _fixed_curve(90, 1, r_outer, 8),
        _fixed_straight(top),
        _fixed_curve(90, 1, r_outer, 8),
        _fixed_straight(left_side),
        _fixed_curve(90, 1, r_outer, 8),
    ]
    return plan, "sketch-ref"


def _reliable_irregular_plan(rng):
    n_corners = int(rng.choice([8, 9, 10, 11]))
    turn_dir = int(rng.choice([0, 1]))
    corner_angles = _random_corner_angles(rng, n_corners)
    sides = _scaled_side_lengths(rng, n_corners, lo=52.0, hi=105.0)
    chicane_quota = [2]
    plan = []
    for side_idx in range(n_corners):
        _append_side(plan, rng, sides[side_idx], chicane_quota=chicane_quota)
        plan.extend(_corner_items(rng, corner_angles[side_idx], turn_dir, wild=True))
    return plan, f"{n_corners}-gon-lite"


def _lane_end_pose(lane):
    return np.array(lane.end, dtype=float), lane.heading_theta_at(lane.length)


def _pre_close_gap(last_block, road_network):
    last_road = last_block.get_socket(0).positive_road
    last_lane = road_network.graph[last_road.start_node][last_road.end_node][0]
    end_pos, end_heading = _lane_end_pose(last_lane)
    entry_pos, entry_heading = _entry_pose(road_network)
    gap = float(np.linalg.norm(end_pos - entry_pos))
    heading_gap = _heading_diff(end_heading, entry_heading)
    return gap, heading_gap


def _validate_spawn_lane(road_network):
    try:
        lane = road_network.graph[FirstPGBlock.NODE_2][MERGE_NODE][0]
    except (KeyError, IndexError):
        return False
    return lane.length > 0.5 and MERGE_NODE in road_network.graph


def _entry_pose(road_network):
    merge_lane = road_network.graph[FirstPGBlock.NODE_2][MERGE_NODE][0]
    return np.array(merge_lane.end, dtype=float), merge_lane.heading_theta_at(merge_lane.length)


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
                idx, current.get_socket(0), road_network, block_seed,
                remove_negative_lanes=True, ignore_intersection_checking=True,
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
                idx, current.get_socket(0), road_network, block_seed,
                remove_negative_lanes=True, ignore_intersection_checking=True,
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


def _build_fallback_circle(first_block, road_network, lane_num, lane_width, render_np, physics_world):
    entry_node = first_block.get_socket(0).positive_road.end_node
    entry_lane = first_block.get_socket(0).positive_road.get_lanes(road_network)[0]
    entry_point = np.array(entry_lane.end, dtype=float)
    center = Vector((entry_point[0], entry_point[1] + 70))
    line_types = (PGLineType.BROKEN, PGLineType.SIDE)
    road_network.add_lane(
        entry_node, "LM",
        CircularLane(center, 70, -math.pi / 2, math.pi, False, lane_width, line_types),
    )
    road_network.add_lane(
        "LM", entry_node,
        CircularLane(center, 70, math.pi / 2, math.pi, False, lane_width, line_types),
    )
    loop_block = _FallbackLoopBlock(
        1, first_block.get_socket(0), road_network, 0,
        remove_negative_lanes=True, ignore_intersection_checking=True,
    )
    loop_block.construct_block(render_np, physics_world)
    return [loop_block], "fallback circle"


class _FallbackLoopBlock(PGBlock):
    ID = "Q"
    SOCKET_NUM = 1
    PARAMETER_SPACE = ParameterSpace({})

    def _try_plug_into_previous_block(self) -> bool:
        socket_road = self.pre_block_socket.positive_road
        self.add_sockets(self.create_socket_from_positive_road(Road(socket_road.end_node, "LM")))
        return True


class LoopMap(PGMap):
    """Random closed circuit with straights and mixed left/right turns."""

    START_LENGTH = 18
    MAX_ATTEMPTS = 40
    PRE_CLOSE_GAP_TOL = 36.0
    FINAL_GAP_TOL = 30.0

    def _generate(self):
        parent_node_path = self.engine.worldNP
        physics_world = self.engine.physics_world
        lane_num = self.config["lane_num"]
        lane_width = self.config["lane_width"]
        seed = int(self.config.get("seed", 0))
        self.circuit_seed = seed

        for attempt in range(28):
            trial_rng = np.random.RandomState(seed + attempt * 3571)
            blocks, label = self._try_build_circuit(
                trial_rng,
                lane_num,
                lane_width,
                parent_node_path,
                physics_world,
                plan_fn=_symmetric_circuit_plan,
                symmetric=True,
            )
            if blocks is not None:
                self.blocks = blocks
                self.circuit_label = label
                self.circuit_segments = len(blocks) - 1
                self.road_network.after_init()
                logger.info("Symmetric circuit ready: %s (seed=%s, try=%s)", label, seed, attempt)
                return

        for attempt in range(12):
            trial_rng = np.random.RandomState(seed + 12000 + attempt * 3571)
            blocks, label = self._try_build_circuit(
                trial_rng,
                lane_num,
                lane_width,
                parent_node_path,
                physics_world,
                plan_fn=_sketch_reference_plan,
            )
            if blocks is not None:
                self.blocks = blocks
                self.circuit_label = label
                self.circuit_segments = len(blocks) - 1
                self.road_network.after_init()
                logger.info("Sketch circuit ready: %s (seed=%s, try=%s)", label, seed, attempt)
                return

        for attempt in range(self.MAX_ATTEMPTS):
            trial_rng = np.random.RandomState(seed + attempt * 7919)
            blocks, label = self._try_build_circuit(
                trial_rng, lane_num, lane_width, parent_node_path, physics_world
            )
            if blocks is not None:
                self.blocks = blocks
                self.circuit_label = label
                self.circuit_segments = len(blocks) - 1
                self.road_network.after_init()
                logger.info("Random circuit ready: %s (seed=%s, try=%s)", label, seed, attempt)
                return

        logger.warning("Random circuit failed, using simplified irregular loop (seed=%s)", seed)
        for lite_try in range(8):
            trial_rng = np.random.RandomState(seed + 99991 + lite_try * 4817)
            blocks, label = self._try_build_circuit(
                trial_rng, lane_num, lane_width, parent_node_path, physics_world, reliable=True
            )
            if blocks is not None:
                self.blocks = blocks
                self.circuit_label = label
                self.circuit_segments = len(blocks) - 1
                self.road_network.after_init()
                return

        logger.warning("Simplified loop failed, trying basic irregular loop (seed=%s)", seed)
        for extra in range(24):
            trial_rng = np.random.RandomState(seed + 19997 + extra * 3571)
            blocks, label = self._try_build_circuit(
                trial_rng, lane_num, lane_width, parent_node_path, physics_world, minimal=True
            )
            if blocks is not None:
                self.blocks = blocks
                self.circuit_label = label
                self.circuit_segments = len(blocks) - 1
                self.road_network.after_init()
                return

        logger.warning("Basic loop failed, using fallback circle (seed=%s)", seed)
        self.road_network = self.road_network_type()
        first_block = FirstPGBlock(
            self.road_network,
            lane_width=lane_width,
            lane_num=lane_num,
            render_root_np=parent_node_path,
            physics_world=physics_world,
            length=self.START_LENGTH,
            remove_negative_lanes=True,
        )
        blocks, label = _build_fallback_circle(
            first_block, self.road_network, lane_num, lane_width, parent_node_path, physics_world
        )
        self.blocks = [first_block] + blocks
        self.circuit_label = label
        self.circuit_segments = 2
        self.road_network.after_init()

    def _try_build_circuit(
        self, rng, lane_num, lane_width, render_np, physics_world,
        reliable=False, minimal=False, plan_fn=None, symmetric=False,
    ):
        road_network = self.road_network_type()
        first_block = FirstPGBlock(
            road_network,
            lane_width=lane_width,
            lane_num=lane_num,
            render_root_np=render_np,
            physics_world=physics_world,
            length=self.START_LENGTH,
            remove_negative_lanes=True,
        )
        blocks = [first_block]
        last_block = first_block
        if plan_fn is _symmetric_circuit_plan:
            plan, style = plan_fn(rng, start_length=self.START_LENGTH)
        elif plan_fn is not None:
            plan, style = plan_fn(rng)
        elif minimal:
            plan, style = _minimal_irregular_plan(rng)
        elif reliable:
            plan, style = _reliable_irregular_plan(rng)
        else:
            plan, style = _circuit_plan(rng)
        labels = []
        block_index = 1
        block_seed = int(rng.randint(0, 10000))

        for block_cls, cfg, label in plan:
            block = block_cls(
                block_index,
                last_block.get_socket(0),
                road_network,
                block_seed,
                remove_negative_lanes=True,
                ignore_intersection_checking=True,
            )
            block.construct_from_config(cfg, render_np, physics_world)
            blocks.append(block)
            last_block = block
            labels.append(label)
            block_index += 1

        if symmetric:
            pre_gap_tol = 48.0
            final_gap_tol = 40.0
            max_close_blocks = 1
        elif plan_fn is not None:
            pre_gap_tol = 42.0
            final_gap_tol = 36.0
            max_close_blocks = 2
        elif minimal:
            pre_gap_tol = 40.0
            final_gap_tol = 34.0
            max_close_blocks = 2
        elif reliable:
            pre_gap_tol = 38.0
            final_gap_tol = 32.0
            max_close_blocks = 2
        else:
            pre_gap_tol = self.PRE_CLOSE_GAP_TOL
            final_gap_tol = self.FINAL_GAP_TOL
            max_close_blocks = 2

        pre_gap, pre_heading = _pre_close_gap(last_block, road_network)
        if pre_gap > pre_gap_tol or pre_heading > CLOSE_HEADING_TOL:
            self._cleanup_blocks(blocks, physics_world)
            return None, None

        ok, last_block, close_blocks, gap = _force_close_with_blocks(
            last_block, road_network, render_np, physics_world, block_index, block_seed
        )
        if not ok or len(close_blocks) > max_close_blocks or gap > final_gap_tol:
            self._cleanup_blocks(blocks + close_blocks, physics_world)
            return None, None

        blocks.extend(close_blocks)

        if _network_has_crossings(road_network):
            self._cleanup_blocks(blocks, physics_world)
            return None, None

        if not _validate_spawn_lane(road_network):
            self._cleanup_blocks(blocks, physics_world)
            return None, None

        seq = "".join(labels)
        if "L" not in seq or "R" not in seq:
            self._cleanup_blocks(blocks, physics_world)
            return None, None

        self.road_network = road_network
        n_left = seq.count("L")
        n_right = seq.count("R")
        label = f"{style} {len(plan)}seg [{seq}] Lx{n_left} Rx{n_right} gap={gap:.1f}m nocross"
        return blocks, label

    @staticmethod
    def _cleanup_blocks(blocks, physics_world):
        for block in reversed(blocks):
            if hasattr(block, "destruct_block"):
                try:
                    block.destruct_block(physics_world)
                except (ValueError, AttributeError):
                    pass
            try:
                block.destroy()
            except Exception:
                pass


class LoopMapManager(PGMapManager):
    """Build one random circuit per scenario seed."""

    def reset(self):
        config = self.engine.global_config.copy()
        current_seed = self.engine.global_seed

        if self.maps[current_seed] is None:
            map_config = config["map_config"].copy()
            map_config["seed"] = current_seed
            loop_map = self.spawn_object(LoopMap, map_config=map_config, random_seed=None)
            self.current_map = loop_map
            if config["store_map"]:
                self.maps[current_seed] = loop_map
        else:
            loop_map = self.maps[current_seed]

        self.load_map(loop_map)


class LoopMetaDriveEnv(MetaDriveEnv):
    """MetaDriveEnv with a random closed circuit; laps do not end the episode."""

    def setup_engine(self):
        super().setup_engine()
        self.engine.update_manager("map_manager", LoopMapManager())

    def _post_process_config(self, config):
        config = super()._post_process_config(config)
        config["agent_configs"][DEFAULT_AGENT]["spawn_lane_index"] = (
            FirstPGBlock.NODE_2,
            MERGE_NODE,
            0,
        )
        return config

    def done_function(self, vehicle_id: str):
        done, done_info = super().done_function(vehicle_id)
        if done_info[TerminationState.SUCCESS]:
            done = False
            done_info[TerminationState.SUCCESS] = False
        return done, done_info
