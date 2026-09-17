#!/usr/bin/env python3
"""闭环测试: 验证多种赛道变化 × 多种驾驶风格的组合能否正常跑通.

不做可视化, 只输出统计结果.
"""
from __future__ import annotations
import sys, copy, math
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from metadrive.component.sensors.rgb_camera import RGBCamera
from simple_track import (
    LANE_NUM, LANE_WIDTH, TRACK_SEGMENTS, RIGHT, LEFT, CURVE_TAIL,
    SimpleTrackEnv, LapCounter,
)
from pid_demo_autonomous import PathFollower, PIDController
from runtime_config import get_render_config

# ============================================================
# 驾驶风格定义
# ============================================================
DRIVING_STYLES = {
    "normal": dict(
        kp=0.8, ki=0.08, kd=0.15, integral_max=1.0,
        look_dist_base=8.0, look_dist_scale=0.3,
        speed_kmh=70.0,
    ),
    "aggressive": dict(
        kp=1.2, ki=0.05, kd=0.25, integral_max=0.8,
        look_dist_base=6.0, look_dist_scale=0.2,
        speed_kmh=80.0,
    ),
    "conservative": dict(
        kp=0.5, ki=0.12, kd=0.08, integral_max=1.5,
        look_dist_base=12.0, look_dist_scale=0.4,
        speed_kmh=55.0,
    ),
    "distracted": dict(
        kp=0.4, ki=0.15, kd=0.05, integral_max=2.0,
        look_dist_base=10.0, look_dist_scale=0.35,
        speed_kmh=65.0,
    ),
}

# ============================================================
# 赛道变化生成
# ============================================================
def generate_track_variation(seed: int):
    """基于原始 TRACK_SEGMENTS 施加随机扰动, 生成新的赛道配置.
    
    关键: 赛道有镜像结构 (前半 outbound, 后半 return),
    配对段施加相同扰动, 保持对称性, 确保能闭合.
    只扰动中间段 (跳过首尾的长直道和闭合段).
    """
    rng = np.random.RandomState(seed)
    segments = list(TRACK_SEGMENTS)
    n = len(segments)
    # 首尾各有 1 段长直道 (FirstBlock + 最后闭合), 不扰动
    # 中间段: index 1 ~ n-2, 按镜像配对
    mid_start = 1
    mid_end = n - 1  # exclusive
    mid_len = mid_end - mid_start
    half = mid_len // 2
    
    for i in range(half):
        idx_out = mid_start + i       # outbound 段
        idx_ret = mid_end - 1 - i     # 对应的 return 段 (镜像)
        scale = rng.uniform(0.85, 1.15)  # ±15% 扰动
        
        for idx in [idx_out, idx_ret]:
            seg = segments[idx]
            if seg[0] == "straight":
                _, length = seg
                new_len = max(50.0, length * scale)
                segments[idx] = ("straight", new_len)
            else:
                _, radius, angle, direction = seg
                new_radius = max(50.0, radius * rng.uniform(0.85, 1.15))
                # 角度不变, 保持几何对称
                segments[idx] = ("curve", new_radius, angle, direction)
    
    return segments


def make_env_with_segments(segments, env_seed=0):
    """用指定的赛道配置创建环境."""
    import simple_track
    original_segments = simple_track.TRACK_SEGMENTS
    simple_track.TRACK_SEGMENTS = segments
    
    config = dict(
        use_render=False, manual_control=False, traffic_density=0.0,
        num_scenarios=1, start_seed=env_seed, map_region_size=4096,
        window_size=(320, 180), show_interface=False,
        show_logo=False, show_fps=False, camera_fov=60,
        image_observation=True, norm_pixel=False,
        image_on_cuda=True, multi_thread_render=True, render_pipeline=False,
        random_lane_width=False, random_lane_num=False,
        out_of_route_done=False, on_continuous_line_done=False,
        map_config=dict(lane_num=LANE_NUM, lane_width=LANE_WIDTH),
        sensors=dict(rgb_camera=(RGBCamera, 640, 360)),
        traffic_vehicle_config=dict(
            show_navi_mark=False, show_dest_mark=False,
            show_lidar=False, show_lane_line_detector=False, show_side_detector=False,
        ),
        vehicle_config=dict(
            show_navi_mark=False, show_line_to_navi_mark=False, show_lidar=True,
            spawn_velocity=[70.0 / 3.6, 0.0], spawn_velocity_car_frame=True,
            image_source="rgb_camera",
        ),
    )
    try:
        env = SimpleTrackEnv(config)
        env.reset(seed=env_seed)
        simple_track.TRACK_SEGMENTS = original_segments
        return env
    except Exception as e:
        simple_track.TRACK_SEGMENTS = original_segments
        print(f"    [DEBUG] env creation error: {e}")
        return None


class StylePathFollower:
    """可配置参数的 PathFollower."""
    
    def __init__(self, ref_x, ref_y, style_params):
        step = max(1, len(ref_x) // 3000)
        self.path_x = np.array(ref_x[::step])
        self.path_y = np.array(ref_y[::step])
        
        self.wheelbase = 2.5
        self.look_dist_base = style_params["look_dist_base"]
        self.look_dist_scale = style_params["look_dist_scale"]
        self.max_steer = 0.6
        
        self.steer_pid = PIDController(
            kp=style_params["kp"], ki=style_params["ki"],
            kd=style_params["kd"], integral_max=style_params["integral_max"],
        )
        self.speed_pid = PIDController(kp=0.15, ki=0.02, kd=0.01, integral_max=3.0)
        self.target_speed_ms = style_params["speed_kmh"] / 3.6
    
    def find_closest_idx(self, x, y):
        diff_sq = (self.path_x - x) ** 2 + (self.path_y - y) ** 2
        return int(np.argmin(diff_sq))
    
    def find_lookahead_point(self, x, y, speed_ms):
        lookahead_dist = self.look_dist_base + self.look_dist_scale * speed_ms
        closest_idx = self.find_closest_idx(x, y)
        target_idx = closest_idx
        n = len(self.path_x)
        for i in range(closest_idx + 1, min(closest_idx + 1000, n)):
            dist = np.sqrt((self.path_x[i] - x) ** 2 + (self.path_y[i] - y) ** 2)
            if dist >= lookahead_dist:
                target_idx = i
                break
            target_idx = i
        return float(self.path_x[target_idx]), float(self.path_y[target_idx]), target_idx
    
    def compute_signed_lateral_error(self, x, y, closest_idx):
        n = len(self.path_x)
        next_idx = min(closest_idx + 1, n - 1)
        path_dx = self.path_x[next_idx] - self.path_x[closest_idx]
        path_dy = self.path_y[next_idx] - self.path_y[closest_idx]
        path_len = np.sqrt(path_dx**2 + path_dy**2)
        if path_len < 1e-6:
            return 0.0, 0.0
        nx = -path_dy / path_len
        ny = path_dx / path_len
        dx = x - self.path_x[closest_idx]
        dy = y - self.path_y[closest_idx]
        signed_lat = dx * nx + dy * ny
        return signed_lat, abs(signed_lat)
    
    def compute_control(self, vehicle_x, vehicle_y, vehicle_heading, vehicle_speed_km_h, dt):
        vehicle_speed_ms = vehicle_speed_km_h / 3.6
        speed_error = self.target_speed_ms - vehicle_speed_ms
        throttle = np.clip(self.speed_pid.update(speed_error, dt), -1.0, 1.0)
        
        closest_idx = self.find_closest_idx(vehicle_x, vehicle_y)
        signed_lat_err, lat_err_abs = self.compute_signed_lateral_error(
            vehicle_x, vehicle_y, closest_idx
        )
        ref_x, ref_y, target_idx = self.find_lookahead_point(
            vehicle_x, vehicle_y, vehicle_speed_ms
        )
        path_dx = ref_x - self.path_x[closest_idx]
        path_dy = ref_y - self.path_y[closest_idx]
        target_heading = np.arctan2(path_dy, path_dx)
        
        heading_err = target_heading - vehicle_heading
        while heading_err > np.pi:
            heading_err -= 2 * np.pi
        while heading_err < -np.pi:
            heading_err += 2 * np.pi
        
        lateral_angle_correction = -np.arctan2(
            signed_lat_err, self.look_dist_base + self.look_dist_scale * vehicle_speed_ms
        )
        combined_error = heading_err + lateral_angle_correction
        steering = np.clip(
            self.steer_pid.update(combined_error, dt), -self.max_steer, self.max_steer
        )
        return [steering, throttle]


def extract_ref_path(env):
    """从环境中提取参考路径 (车道中心线)."""
    ref_x, ref_y = [], []
    for _from, to_dict in env.current_map.road_network.graph.items():
        for _to, lane_list in to_dict.items():
            for lane in lane_list:
                if lane.length > 50.0:
                    for s in np.linspace(0, 1, max(2, int(lane.length))):
                        pos = lane.position(s, 0)
                        ref_x.append(float(pos[0]))
                        ref_y.append(float(pos[1]))
    return np.array(ref_x), np.array(ref_y)


def run_one_test(segments, style_name, style_params, max_frames=3000, env_seed=0):
    """跑一个组合, 返回统计结果."""
    env = make_env_with_segments(segments, env_seed=env_seed)
    if env is None:
        return {"status": "FAIL_ENV", "reason": "赛道创建失败"}
    
    try:
        ref_x, ref_y = extract_ref_path(env)
        if len(ref_x) < 100:
            return {"status": "FAIL_REF", "reason": "参考路径太短"}
        
        follower = StylePathFollower(ref_x, ref_y, style_params)
        vehicle = env.agent
        
        origin_xy = np.array(vehicle.position[:2], dtype=float)
        lap_counter = LapCounter(origin_xy)
        
        lat_errors = []
        frames = 0
        
        for frame in range(max_frames):
            vx, vy = float(vehicle.position[0]), float(vehicle.position[1])
            vh = float(vehicle.heading_theta)
            vs = float(vehicle.speed_km_h)
            
            control = follower.compute_control(vx, vy, vh, vs, 0.1)
            obs, rew, done, trunc, info = env.step(control)
            
            laps = lap_counter.update(vehicle.position)
            
            # 记录横向误差
            closest_idx = follower.find_closest_idx(vx, vy)
            _, lat_err = follower.compute_signed_lateral_error(vx, vy, closest_idx)
            lat_errors.append(lat_err)
            
            # 安全检查
            if lat_err > 15.0:
                return {
                    "status": "FAIL_CRASH",
                    "frames": frame + 1,
                    "laps": laps,
                    "mean_lat": np.mean(lat_errors),
                    "max_lat": np.max(lat_errors),
                    "reason": f"横向误差 {lat_err:.1f}m > 15m",
                }
            
            frames = frame + 1
            
            # 跑完 2 圈就算通过
            if laps >= 2:
                return {
                    "status": "OK",
                    "frames": frames,
                    "laps": laps,
                    "mean_lat": np.mean(lat_errors),
                    "max_lat": np.max(lat_errors),
                }
        
        return {
            "status": "OK_TIMEOUT",
            "frames": frames,
            "laps": laps,
            "mean_lat": np.mean(lat_errors),
            "max_lat": np.max(lat_errors),
            "reason": f"跑完 {frames} 帧, {laps} 圈",
        }
    
    finally:
        env.close()


def main():
    # 生成赛道变化
    track_seeds = [42, 123, 456, 789, 1024]  # 5 种赛道
    style_names = list(DRIVING_STYLES.keys())  # 4 种风格
    
    print(f"测试矩阵: {len(track_seeds)} 赛道 × {len(style_names)} 风格 = {len(track_seeds) * len(style_names)} 组合")
    print(f"赛道 seeds: {track_seeds}")
    print(f"驾驶风格: {style_names}")
    print("=" * 80)
    
    results = []
    ok_count = 0
    fail_count = 0
    
    for t_idx, t_seed in enumerate(track_seeds):
        segments = generate_track_variation(t_seed)
        # 计算赛道长度
        total_len = 0
        for seg in segments:
            if seg[0] == "straight":
                total_len += seg[1]
            else:
                total_len += seg[1] * math.radians(seg[2]) + CURVE_TAIL
        
        for s_name in style_names:
            style_params = DRIVING_STYLES[s_name]
            label = f"track#{t_seed}(L={total_len:.0f}m) × {s_name}(v={style_params['speed_kmh']:.0f})"
            
            result = run_one_test(segments, s_name, style_params, env_seed=t_seed)
            result["label"] = label
            result["track_seed"] = t_seed
            result["style"] = s_name
            results.append(result)
            
            status_icon = "✓" if result["status"].startswith("OK") else "✗"
            if result["status"].startswith("OK"):
                ok_count += 1
            else:
                fail_count += 1
            
            detail = ""
            if "laps" in result:
                detail = f"{result['frames']}帧 {result['laps']}圈 mean_lat={result['mean_lat']:.2f}m"
            if "reason" in result:
                detail += f" [{result['reason']}]"
            
            print(f"  {status_icon} {label:55s} → {result['status']:12s} {detail}")
    
    print("=" * 80)
    print(f"总计: {ok_count} 通过, {fail_count} 失败 / {len(results)} 组合")
    
    if fail_count > 0:
        print("\n失败详情:")
        for r in results:
            if not r["status"].startswith("OK"):
                print(f"  {r['label']} → {r['status']}: {r.get('reason', '')}")


if __name__ == "__main__":
    main()
