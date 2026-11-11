"""公共工具: 参考路径 / 无头环境 / 轨迹读写 / 横向误差。

所有环境均为 use_render=False, 不弹任何窗口。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

import config

# 保证可以 import 项目根目录的既有模块 (simple_track / pid_demo_autonomous 等)
if str(config.PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(config.PROJECT_ROOT))


def load_reference_path():
    """加载整圈参考路径 (内侧车道中心线, ~1m 一个点)."""
    with open(config.REF_PATH_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    x = np.asarray(data["reference_path"]["x"], dtype=float)
    y = np.asarray(data["reference_path"]["y"], dtype=float)
    return x, y


def build_headless_env(with_camera: bool = False):
    """构建无头 SimpleTrackEnv (与 pid_demo_autonomous 相同车辆/地图配置)."""
    from simple_track import LANE_NUM, LANE_WIDTH, SimpleTrackEnv
    from runtime_config import get_render_config

    cfg = dict(
        use_render=False,
        manual_control=False,
        traffic_density=0.0,
        num_scenarios=1,
        start_seed=0,
        map_region_size=4096,
        show_interface=False,
        show_logo=False,
        show_fps=False,
        **get_render_config(),
        random_lane_width=False,
        random_lane_num=False,
        out_of_route_done=False,
        on_continuous_line_done=False,
        out_of_road_done=False,
        crash_vehicle_done=False,
        crash_object_done=False,
        crash_human_done=False,
        map_config=dict(lane_num=LANE_NUM, lane_width=LANE_WIDTH),
        vehicle_config=dict(
            show_navi_mark=False,
            show_line_to_navi_mark=False,
            show_lidar=False,
            spawn_velocity=[config.TARGET_SPEED_KMH / 3.6, 0.0],
            spawn_velocity_car_frame=True,
        ),
    )
    if with_camera:
        from metadrive.component.sensors.rgb_camera import RGBCamera

        # 关键: use_render=False 且 image_observation=False 时 base_env 会过滤掉所有相机,
        # 必须显式打开 image_observation; image_source 默认 main_camera 不存在会 KeyError
        cfg.update(
            image_observation=True,
            camera_fov=config.IMAGE_FOV_DEG,
            sensors=dict(rgb_camera=(RGBCamera, config.IMAGE_WIDTH, config.IMAGE_HEIGHT)),
        )
        cfg["vehicle_config"]["image_source"] = "rgb_camera"
    return SimpleTrackEnv(cfg)


def capture_rgb(env, vehicle, path: Path):
    """在当前位姿渲染并保存一帧 RGB (离屏, 不弹窗).

    无 main_camera 时 reset_sensors 不会把相机挂到车上, 必须手动 track 后再 renderFrame,
    否则读到的是旧位姿纹理。
    """
    import cv2
    from metadrive.constants import DEFAULT_SENSOR_HPR, DEFAULT_SENSOR_OFFSET

    cam = env.engine.get_sensor("rgb_camera")
    cam.track(vehicle.origin, DEFAULT_SENSOR_OFFSET, DEFAULT_SENSOR_HPR)
    env.engine.graphicsEngine.renderFrame()
    img = cam.get_image(vehicle, mode="bgr")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), img):
        raise IOError(f"图片写入失败: {path}")
    return path


def split_control(vehicle):
    """与 record.py 一致: 拆分 throttle_brake 为 throttle / brake."""
    steering = float(vehicle.steering)
    tb = float(vehicle.throttle_brake)
    if tb >= 0.0:
        return steering, tb, 0.0
    return steering, 0.0, abs(tb)


def vehicle_status(env, vehicle):
    offroad = bool(env._is_out_of_road(vehicle))
    collision = bool(
        vehicle.crash_vehicle
        or vehicle.crash_object
        or vehicle.crash_building
        or vehicle.crash_human
    )
    return offroad, collision


def save_trajectory(path: Path, frames: list, metadata: dict):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"metadata": metadata, "frames": frames}, f, ensure_ascii=False)
    return path


def load_trajectory(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["frames"], data["metadata"]


def frames_to_arrays(frames: list):
    """轨迹帧列表 → numpy 数组 (t, x, y, heading, speed_kmh, steering, throttle, brake)."""
    t = np.array([f["t"] for f in frames], dtype=float)
    x = np.array([f["position"]["x"] for f in frames], dtype=float)
    y = np.array([f["position"]["y"] for f in frames], dtype=float)
    heading = np.array([f["heading"] for f in frames], dtype=float)
    speed = np.array([f["speed_kmh"] for f in frames], dtype=float)
    steering = np.array([f["control"]["steering"] for f in frames], dtype=float)
    throttle = np.array([f["control"]["throttle"] for f in frames], dtype=float)
    brake = np.array([f["control"]["brake"] for f in frames], dtype=float)
    return dict(t=t, x=x, y=y, heading=heading, speed_kmh=speed,
                steering=steering, throttle=throttle, brake=brake)


class LateralErrorTracker:
    """对单圈基准路径的有符号横向误差 (正=路径左侧).

    查询沿路径里程单调推进 (局部窗口 + 兜底只允许前进),
    避免重叠/多圈路径上最近点跳圈导致误差虚高。
    """

    def __init__(self, ref_x, ref_y, spacing: float = 2.0):
        ref_x = np.asarray(ref_x, dtype=float)
        ref_y = np.asarray(ref_y, dtype=float)
        # 按弧长均匀重采样, 保证窗口索引与里程成正比
        seg = np.hypot(np.diff(ref_x), np.diff(ref_y))
        arc = np.concatenate([[0.0], np.cumsum(seg)])
        n = max(int(arc[-1] / spacing), 2)
        s_grid = np.linspace(0.0, arc[-1], n)
        self.px = np.interp(s_grid, arc, ref_x)
        self.py = np.interp(s_grid, arc, ref_y)
        self.spacing = float(arc[-1] / (n - 1))
        dx = np.diff(self.px)
        dy = np.diff(self.py)
        seg_len = np.hypot(dx, dy) + 1e-9
        self.nx = -dy / seg_len
        self.ny = dx / seg_len
        self.last_idx = 0

    def reset(self):
        """重置里程索引 (车辆被传送/新片段开始时调用)."""
        self.last_idx = 0

    def signed_error(self, x: float, y: float) -> float:
        d2 = (self.px - x) ** 2 + (self.py - y) ** 2
        n = len(self.px)
        # 局部窗口: 当前位置附近 ±60m, 且只允许前进 (闭环上防止跳圈)
        win = max(int(60.0 / self.spacing), 8)
        lo = max(0, self.last_idx - 4)
        hi = min(n, self.last_idx + win)
        idx = lo + int(np.argmin(d2[lo:hi]))
        if d2[idx] > 80.0 ** 2:
            # 兜底: 全路径搜索, 但只接受更靠前的点
            g = int(np.argmin(d2))
            if g >= self.last_idx:
                idx = g
        self.last_idx = max(self.last_idx, idx)
        j = min(idx, n - 2)
        return float((x - self.px[j]) * self.nx[j] + (y - self.py[j]) * self.ny[j])


def extract_lap_reference(traj_path, spacing: float = 2.0):
    """从采集轨迹提取单圈基准路径 (完圈时刻截断, 闭环).

    quick/无完圈信息时返回整段开环路径。
    返回 (ref_x, ref_y, closed: bool)。
    """
    frames, meta = load_trajectory(traj_path)
    arr = frames_to_arrays(frames)
    lap_t = meta.get("lap_time_s")
    if lap_t is not None and not meta.get("quick", False):
        keep = arr["t"] <= float(lap_t) + 1e-6
        x, y = arr["x"][keep], arr["y"][keep]
        closed = True
    else:
        x, y = arr["x"], arr["y"]
        closed = False
    if closed:
        # 首尾闭合 (完圈点与起点距离很小, 直接连接)
        x = np.append(x, x[0])
        y = np.append(y, y[0])
    return x, y, closed


def wrap_to_pi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi