#!/usr/bin/env python3
"""自动驾驶 PID 控制器 - 无头模式 + matplotlib BEV 可视化"""
import numpy as np
import json
import csv
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from runtime_config import get_render_config
from simple_track import (
    LANE_NUM,
    LANE_WIDTH,
    TARGET_SPEED_KMH,
    LapCounter,
    SimpleTrackEnv,
)


# ============================================================
# PID 控制器
# ============================================================
class PIDController:
    """通用 PID 控制器"""

    def __init__(self, kp, ki, kd, integral_max=5.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral_max = integral_max
        self.integral = 0.0
        self.last_error = None

    def update(self, error, dt):
        p_term = self.kp * error
        self.integral += error * dt
        self.integral = np.clip(self.integral, -self.integral_max, self.integral_max)
        i_term = self.ki * self.integral
        if self.last_error is None:
            d_term = 0.0
        else:
            d_term = self.kd * (error - self.last_error) / dt
        self.last_error = error
        return p_term + i_term + d_term

    def reset(self):
        self.integral = 0.0
        self.last_error = None


# ============================================================
# 路径跟踪控制器 (Pure Pursuit + PID)
# ============================================================
class PathFollower:
    """Pure Pursuit + PID 路径跟踪 - 参考路径就是内侧车道中心线

    核心思路:
    1. Pure Pursuit: 在参考路径上找前瞻点，计算目标航向
    2. PID: 输入 = 航向误差 + 横向偏移修正，输出转向角
       关键：横向偏移必须加入 PID 输入，否则车对准了路径方向但偏在一边时
       航向误差≈0，PID 不再修正，车就一直偏着走
    """

    def __init__(self, ref_x, ref_y):
        # 降采样参考路径（原始太密，1m 一个点）
        step = max(1, len(ref_x) // 3000)
        self.path_x = np.array(ref_x[::step])
        self.path_y = np.array(ref_y[::step])

        # Pure Pursuit 参数
        self.wheelbase = 2.5         # 轴距 (米), MetaDrive 默认车辆 ~2.5m
        self.look_dist_base = 8.0    # 基础前瞻距离
        self.look_dist_scale = 0.3   # 速度相关的前瞻距离系数
        self.max_steer = 0.6

        # 横向 PID: 输入 = 综合误差(航向 + 横向), 输出 = 转向角
        self.steer_pid = PIDController(kp=0.8, ki=0.08, kd=0.15, integral_max=1.0)
        # 纵向 PID
        self.speed_pid = PIDController(kp=0.15, ki=0.02, kd=0.01, integral_max=3.0)

        # 日志
        self.log_data = []

    def find_closest_idx(self, x, y):
        diff_sq = (self.path_x - x) ** 2 + (self.path_y - y) ** 2
        return int(np.argmin(diff_sq))

    def find_lookahead_point(self, x, y, speed_ms):
        """在参考路径上找前瞻点: 从最近点往前 lookahead_dist 距离"""
        lookahead_dist = self.look_dist_base + self.look_dist_scale * speed_ms
        closest_idx = self.find_closest_idx(x, y)

        # 从 closest_idx 往前搜索，找到距离 = lookahead_dist 的点
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
        """计算有符号横向误差: 正=在路径左侧, 负=在路径右侧"""
        n = len(self.path_x)
        next_idx = min(closest_idx + 1, n - 1)
        path_dx = self.path_x[next_idx] - self.path_x[closest_idx]
        path_dy = self.path_y[next_idx] - self.path_y[closest_idx]
        path_len = np.sqrt(path_dx**2 + path_dy**2)
        if path_len < 1e-6:
            return 0.0, 0.0
        # 路径单位法向量 (指向左侧)
        nx = -path_dy / path_len
        ny = path_dx / path_len
        # 车辆到路径点的向量 投影到法向量
        dx = x - self.path_x[closest_idx]
        dy = y - self.path_y[closest_idx]
        signed_lat = dx * nx + dy * ny
        return signed_lat, abs(signed_lat)

    def compute_control(self, vehicle_x, vehicle_y, vehicle_heading, vehicle_speed_km_h, dt):
        """返回 [steering, throttle]"""
        vehicle_speed_ms = vehicle_speed_km_h / 3.6
        target_speed_ms = TARGET_SPEED_KMH / 3.6

        # ---- 纵向 PID ----
        speed_error = target_speed_ms - vehicle_speed_ms
        throttle = np.clip(self.speed_pid.update(speed_error, dt), -1.0, 1.0)

        # ---- 横向: Pure Pursuit + PID ----
        closest_idx = self.find_closest_idx(vehicle_x, vehicle_y)
        signed_lat_err, lat_err_abs = self.compute_signed_lateral_error(
            vehicle_x, vehicle_y, closest_idx
        )

        # Pure Pursuit: 找前瞻点
        ref_x, ref_y, target_idx = self.find_lookahead_point(
            vehicle_x, vehicle_y, vehicle_speed_ms
        )

        # 目标航向: 从 closest 到 lookahead 的方向
        path_dx = ref_x - self.path_x[closest_idx]
        path_dy = ref_y - self.path_y[closest_idx]
        target_heading = np.arctan2(path_dy, path_dx)

        # 航向误差
        heading_err = target_heading - vehicle_heading
        while heading_err > np.pi:
            heading_err -= 2 * np.pi
        while heading_err < -np.pi:
            heading_err += 2 * np.pi

        # 综合误差 = 航向误差 + 横向偏移修正
        # 横向偏移修正: 车在路径左侧(signed_lat>0) → 需要右转 → 减小综合误差
        # 用比例系数把横向误差转换成等效的角度修正
        lateral_angle_correction = -np.arctan2(signed_lat_err, self.look_dist_base + self.look_dist_scale * vehicle_speed_ms)
        combined_error = heading_err + lateral_angle_correction

        # PID 控制转向
        steering = np.clip(self.steer_pid.update(combined_error, dt), -self.max_steer, self.max_steer)

        self.log_data.append({
            'speed_km_h': vehicle_speed_km_h,
            'throttle': throttle,
            'steering': steering,
            'lateral_error': lat_err_abs,
            'signed_lateral_error': signed_lat_err,
            'heading_error_deg': np.degrees(heading_err),
        })

        return [steering, throttle]

    def save_log(self, filename='pid_control_log.csv'):
        if not self.log_data:
            return
        with open(filename, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.log_data[0].keys())
            writer.writeheader()
            writer.writerows(self.log_data)
        print(f"✓ 日志已保存: {filename} ({len(self.log_data)} 条)")


# ============================================================
# BEV 鸟瞰图可视化 (matplotlib 小窗口)
# ============================================================
class BEVVisualizer:
    """matplotlib BEV 动态小窗口"""

    def __init__(self, ref_x, ref_y, map_bounds=None):
        self.ref_x = ref_x
        self.ref_y = ref_y
        self.traj_x = []
        self.traj_y = []

        # 降采样参考路径用于显示
        step = max(1, len(ref_x) // 2000)
        self.disp_ref_x = ref_x[::step]
        self.disp_ref_y = ref_y[::step]

        self.fig, self.ax = plt.subplots(1, 1, figsize=(7, 7))
        self.fig.canvas.manager.set_window_title('PID BEV Debug View')

        # 计算显示范围
        margin = 100
        if map_bounds is not None:
            self.x_min, self.x_max, self.y_min, self.y_max = map_bounds
        else:
            self.x_min = min(self.disp_ref_x) - margin
            self.x_max = max(self.disp_ref_x) + margin
            self.y_min = min(self.disp_ref_y) - margin
            self.y_max = max(self.disp_ref_y) + margin

        # 车辆标记
        self.vehicle_dot, = self.ax.plot([], [], 'r>', markersize=10, zorder=5)
        self.traj_line, = self.ax.plot([], [], 'r-', linewidth=0.8, alpha=0.6, zorder=3)
        self.info_text = self.ax.text(
            0.02, 0.98, '', transform=self.ax.transAxes,
            fontsize=8, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
        )

        self._setup_axes()

    def _setup_axes(self):
        self.ax.set_xlim(self.x_min, self.x_max)
        self.ax.set_ylim(self.y_min, self.y_max)
        self.ax.set_aspect('equal')
        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Y (m)')
        self.ax.set_title('BEV - PID Path Following (red=car, blue=ref path)')
        # 画参考路径
        self.ax.plot(self.disp_ref_x, self.disp_ref_y, 'b-', linewidth=0.5, alpha=0.5, zorder=1, label='Reference Path')
        self.ax.legend(loc='upper right', fontsize=7)
        self.ax.grid(True, linewidth=0.3, alpha=0.3)

    def update(self, x, y, heading, speed, steering, lat_err, laps, frame):
        self.traj_x.append(x)
        self.traj_y.append(y)

        self.vehicle_dot.set_data([x], [y])
        self.vehicle_dot.set_marker('>')
        # 旋转标记表示朝向
        import matplotlib.transforms as mtransforms
        t = mtransforms.Affine2D().rotate_deg_around(x, y, np.degrees(heading))
        self.vehicle_dot.set_transform(t + self.ax.transData)

        self.traj_line.set_data(self.traj_x, self.traj_y)

        self.info_text.set_text(
            f"Frame: {frame}\n"
            f"Speed: {speed:.1f} km/h\n"
            f"Steer: {steering:.3f}\n"
            f"Lat Err: {lat_err:.2f} m\n"
            f"Laps: {laps}\n"
            f"Pos: ({x:.1f}, {y:.1f})"
        )

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def save_snapshot(self, filename='bev_snapshot.png'):
        self.fig.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"✓ BEV 截图已保存: {filename}")

    def close(self):
        plt.close(self.fig)


# ============================================================
# 导出地图 + 路径 JPG
# ============================================================
def export_map_jpg(env, ref_x, ref_y, filename='map_and_path.jpg'):
    """无头模式导出地图+路径图"""
    try:
        from metadrive.utils.draw_top_down_map import draw_top_down_map
        img = draw_top_down_map(env.current_map, resolution=(2048, 2048), semantic_map=True)
        import cv2
        # 在地图上叠加参考路径
        # 需要获取地图的坐标映射关系
        # 简单方案：用 top_down_renderer 的 scaling
        from metadrive.engine.top_down_renderer import draw_top_down_map_native
        from metadrive.utils.utils import import_pygame
        pygame = import_pygame()
        surface = draw_top_down_map_native(
            env.current_map, semantic_map=True, return_surface=True,
            film_size=(4096, 4096),
        )
        scaling = surface.scaling
        # 获取 bounding box 中心
        b_box = env.current_map.road_network.get_bounding_box()
        cx = (b_box[0] + b_box[1]) / 2
        cy = (b_box[2] + b_box[3]) / 2
        w, h = surface.get_size()

        # 世界坐标 → 像素坐标
        def world2pix(wx, wy):
            px = int((wx - cx) * scaling + w / 2)
            py = int(h / 2 - (wy - cy) * scaling)
            return px, py

        # 在 surface 上画参考路径
        step = max(1, len(ref_x) // 5000)
        pts = []
        for i in range(0, len(ref_x), step):
            px, py = world2pix(ref_x[i], ref_y[i])
            if 0 <= px < w and 0 <= py < h:
                pts.append((px, py))
        for i in range(len(pts) - 1):
            pygame.draw.line(surface, (0, 100, 255), pts[i], pts[i + 1], 3)

        # 保存
        import pygame as pg
        pg.image.save(surface, filename)
        pg.quit()
        print(f"✓ 地图+路径已导出: {filename}")
    except Exception as e:
        print(f"✗ 导出地图失败: {e}")


# ============================================================
# 主函数
# ============================================================
def load_reference_path():
    with open('simple_track_reference_clean.json', 'r') as f:
        data = json.load(f)
    px = data['reference_path']['x']
    py = data['reference_path']['y']
    print(f"✓ 加载参考路径: {len(px)} 个点, 总长 {data['metadata']['total_path_length_m']:.0f} m")
    return px, py


if __name__ == "__main__":
    print("=" * 60)
    print("PID 自动驾驶 - MetaDrive 3D 可视化")
    print("=" * 60)

    ref_path_x, ref_path_y = load_reference_path()

    # ---- MetaDrive 3D 可视化模式 ----
    config = dict(
        use_render=True,            # 打开 3D 窗口
        manual_control=False,
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
            spawn_velocity=[TARGET_SPEED_KMH / 3.6, 0.0],
            spawn_velocity_car_frame=True,
        ),
    )

    env = SimpleTrackEnv(config)
    follower = None

    try:
        env.reset(seed=0)

        # 创建控制器
        follower = PathFollower(ref_path_x, ref_path_y)

        origin_xy = np.array(env.agent.position[:2], dtype=float)
        lap_counter = LapCounter(origin_xy)

        print(f"\n✓ 环境初始化完成 (3D 可视化模式)")
        print(f"  目标速度: {TARGET_SPEED_KMH:.0f} km/h")
        print(f"  起始位置: ({origin_xy[0]:.2f}, {origin_xy[1]:.2f})")

        prev_laps = 0
        frame = 0
        dt = 0.1

        print(f"\n开始驾驶... (按 ESC 退出)")
        print("-" * 60)

        while True:
            vehicle_x, vehicle_y = float(env.agent.position[0]), float(env.agent.position[1])
            vehicle_heading = float(env.agent.heading_theta)
            vehicle_speed_km_h = float(env.agent.speed_km_h)

            action = follower.compute_control(
                vehicle_x, vehicle_y, vehicle_heading, vehicle_speed_km_h, dt
            )

            env.step(action)

            laps = lap_counter.update(env.agent.position)
            if laps > prev_laps:
                env.agent.reset_navigation()
                prev_laps = laps
                print(f"\n✓ 完成第 {laps} 圈!")

            lat_err = follower.log_data[-1]['lateral_error'] if follower.log_data else 0.0

            env.render(
                text={
                    "Speed (km/h)": f"{vehicle_speed_km_h:.1f}",
                    "Steering": f"{action[0]:.3f}",
                    "Throttle": f"{action[1]:.3f}",
                    "Lat Err (m)": f"{lat_err:.2f}",
                    "Laps": str(laps),
                    "Control": "PID+PurePursuit",
                }
            )

            if frame % 200 == 0:
                print(f"Frame {frame:5d} | Speed: {vehicle_speed_km_h:5.1f} km/h | "
                      f"Steer: {action[0]:6.3f} | Throttle: {action[1]:6.3f} | "
                      f"Lat Err: {lat_err:5.2f} m | Laps: {laps}")

            frame += 1

    except KeyboardInterrupt:
        print("\n\n用户中断")
    finally:
        if follower:
            follower.save_log()
        env.close()
        print(f"\n✓ 演示结束 (共 {frame} 帧)")
        print("=" * 60)
