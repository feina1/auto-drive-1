#!/usr/bin/env python3
"""自动驾驶 PID 控制器 - 参考路径 + 车道中心修正"""
import numpy as np
import json
import csv

from runtime_config import get_render_config
from simple_track import (
    LANE_NUM,
    LANE_WIDTH,
    TARGET_SPEED_KMH,
    LapCounter,
    SimpleTrackEnv,
)


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
        """计算 PID 输出"""
        p_term = self.kp * error
        
        self.integral += error * dt
        self.integral = max(-self.integral_max, min(self.integral_max, self.integral))
        i_term = self.ki * self.integral
        
        if self.last_error is None:
            d_term = 0.0
        else:
            d_term = self.kd * (error - self.last_error) / dt
        self.last_error = error
        
        return p_term + i_term + d_term
    
    def reset(self):
        """重置控制器状态"""
        self.integral = 0.0
        self.last_error = None


class LaneCenterCorrector:
    """车道中心修正器 - 计算从参考路径到车道中心的偏移"""
    
    def __init__(self, env):
        self.env = env
    
    def get_correction_offset(self, ref_x, ref_y, vehicle_x, vehicle_y):
        """
        计算车道中心修正偏移
        
        返回: (corrected_target_x, corrected_target_y, lateral_error)
        """
        agent = self.env.agent
        lane = agent.lane
        
        if lane is None:
            # 没有车道信息，直接返回参考点
            return ref_x, ref_y, 0.0
        
        # 获取车辆在车道坐标系中的位置
        longitudinal, lateral = lane.local_coordinates(agent.position)
        
        # lateral 是偏离车道中心的距离
        # 我们要让车保持在车道中心，所以需要修正目标点
        
        # 获取车道中心线在当前位置的点
        center_pos = lane.position(longitudinal, 0)
        
        # 计算参考点到车道中心的偏移
        ref_to_center_x = center_pos[0] - ref_x
        ref_to_center_y = center_pos[1] - ref_y
        
        # 修正后的目标点（参考点 + 偏移到车道中心）
        corrected_x = ref_x + ref_to_center_x
        corrected_y = ref_y + ref_to_center_y
        
        return corrected_x, corrected_y, lateral


class PathFollower:
    """路径跟踪控制器：参考路径 + 车道中心修正"""
    
    def __init__(self, reference_path_x, reference_path_y, env):
        self.path_x = np.array(reference_path_x)
        self.path_y = np.array(reference_path_y)
        self.env = env
        self.corrector = LaneCenterCorrector(env)
        
        # 横向控制参数
        self.look_dist_base = 8.0
        self.look_dist_scale = 0.15
        self.max_steer = 0.6
        
        # 横向 PID
        self.steer_pid = PIDController(kp=0.8, ki=0.05, kd=0.2, integral_max=0.5)
        
        # 纵向 PID
        self.speed_pid = PIDController(kp=0.15, ki=0.02, kd=0.01, integral_max=3.0)
        
        # 日志
        self.log_data = []
    
    def find_lookahead_point(self, x, y, speed_ms):
        """Pure Pursuit: 找到前瞻点"""
        lookahead_dist = self.look_dist_base + self.look_dist_scale * speed_ms
        
        diff_sq = (self.path_x - x)**2 + (self.path_y - y)**2
        closest_idx = int(np.argmin(diff_sq))
        
        target_idx = closest_idx
        for i in range(closest_idx, min(closest_idx + 500, len(self.path_x))):
            tx, ty = float(self.path_x[i]), float(self.path_y[i])
            dist_sq = (tx - x)**2 + (ty - y)**2
            if dist_sq > lookahead_dist**2:
                target_idx = i
                break
            target_idx = i
        
        return float(self.path_x[target_idx]), float(self.path_y[target_idx]), target_idx
    
    def compute_control(self, vehicle_x, vehicle_y, vehicle_heading, vehicle_speed_km_h, dt):
        """计算控制指令 [steering, throttle]"""
        vehicle_speed_ms = vehicle_speed_km_h / 3.6
        target_speed_ms = TARGET_SPEED_KMH / 3.6
        
        # 1. 纵向控制
        speed_error = target_speed_ms - vehicle_speed_ms
        throttle = self.speed_pid.update(speed_error, dt)
        throttle = max(-1.0, min(1.0, throttle))
        
        # 2. 找到参考路径上的前瞻点
        ref_x, ref_y, target_idx = self.find_lookahead_point(
            vehicle_x, vehicle_y, vehicle_speed_ms
        )
        
        # 3. 应用车道中心修正
        corrected_x, corrected_y, lateral_error = self.corrector.get_correction_offset(
            ref_x, ref_y, vehicle_x, vehicle_y
        )
        
        # 4. 计算目标航向（使用修正后的点）
        if target_idx > 0:
            dx = float(self.path_x[target_idx] - self.path_x[target_idx - 1])
            dy = float(self.path_y[target_idx] - self.path_y[target_idx - 1])
            target_heading = np.arctan2(dy, dx)
        else:
            target_heading = vehicle_heading
        
        # 5. 计算航向误差
        angle_error = target_heading - vehicle_heading
        while angle_error > np.pi:
            angle_error -= 2 * np.pi
        while angle_error < -np.pi:
            angle_error += 2 * np.pi
        
        # 6. PID 控制转向
        steering_cmd = self.steer_pid.update(angle_error, dt)
        steering = max(-self.max_steer, min(self.max_steer, steering_cmd))
        
        # 记录日志
        self.log_data.append({
            'speed_km_h': vehicle_speed_km_h,
            'throttle': throttle,
            'steering': steering,
            'lateral_error': lateral_error,
            'angle_error_deg': np.degrees(angle_error),
        })
        
        return [steering, throttle]
    
    def save_log(self, filename='pid_control_log.csv'):
        """保存日志到 CSV"""
        if not self.log_data:
            return
        
        with open(filename, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.log_data[0].keys())
            writer.writeheader()
            writer.writerows(self.log_data)
        print(f"✓ 日志已保存: {filename}")


def load_reference_path():
    """加载参考路径"""
    with open('simple_track_reference_clean.json', 'r') as f:
        data = json.load(f)
    
    path_x = data['reference_path']['x']
    path_y = data['reference_path']['y']
    print(f"✓ 加载参考路径: {len(path_x)} 个点")
    return path_x, path_y


def _traffic_count(env):
    """获取交通车辆数量"""
    return len(env.engine.traffic_manager._traffic_vehicles)


if __name__ == "__main__":
    print("=" * 60)
    print("PID 自动驾驶 - 参考路径 + 车道中心修正")
    print("=" * 60)
    
    # 加载参考路径
    ref_path_x, ref_path_y = load_reference_path()
    
    # 配置环境
    TARGET_SPEED = TARGET_SPEED_KMH
    config = dict(
        use_render=True,
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
            spawn_velocity=[TARGET_SPEED / 3.6, 0.0],
            spawn_velocity_car_frame=True,
        ),
    )
    
    env = SimpleTrackEnv(config)
    
    try:
        env.reset(seed=0)
        
        # 创建路径跟踪控制器
        path_follower = PathFollower(ref_path_x, ref_path_y, env)
        
        origin_xy = np.array(env.agent.position[:2], dtype=float)
        lap_counter = LapCounter(origin_xy)
        
        print(f"\n✓ 环境初始化完成")
        print(f"  目标速度: {TARGET_SPEED:.0f} km/h")
        print(f"  起始位置: ({origin_xy[0]:.2f}, {origin_xy[1]:.2f})")
        print(f"  车道宽度: {LANE_WIDTH} m")
        print(f"  交通车辆: {_traffic_count(env)}")
        
        prev_laps = 0
        frame = 0
        dt = 0.1
        
        print("\n开始驾驶... (按 ESC 退出)")
        print("-" * 60)
        
        while True:
            vehicle_x, vehicle_y = env.agent.position[:2]
            vehicle_heading = env.agent.heading_theta
            vehicle_speed_km_h = env.agent.speed_km_h
            
            action = path_follower.compute_control(
                vehicle_x, vehicle_y, vehicle_heading, vehicle_speed_km_h, dt
            )
            
            env.step(action)
            
            laps = lap_counter.update(env.agent.position)
            if laps > prev_laps:
                env.agent.reset_navigation()
                prev_laps = laps
                print(f"\n✓ 完成第 {laps} 圈!")
            
            lateral_error = path_follower.log_data[-1]['lateral_error'] if path_follower.log_data else 0.0
            
            env.render(
                text={
                    "Speed (km/h)": f"{vehicle_speed_km_h:.1f}",
                    "Steering": f"{action[0]:.3f}",
                    "Throttle": f"{action[1]:.3f}",
                    "Lateral Err (m)": f"{lateral_error:.2f}",
                    "Laps": str(laps),
                    "Control": "Path + Lane Center",
                }
            )
            
            frame += 1
            
            if frame % 100 == 0:
                print(f"Frame {frame:4d} | Speed: {vehicle_speed_km_h:5.1f} km/h | "
                      f"Steer: {action[0]:6.3f} | Throttle: {action[1]:6.3f} | "
                      f"Lat Err: {lateral_error:5.2f} m")
    
    except KeyboardInterrupt:
        print("\n\n用户中断")
    
    finally:
        path_follower.save_log()
        env.close()
        print("\n✓ 演示结束")
        print("=" * 60)
