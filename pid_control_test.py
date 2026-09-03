#!/usr/bin/env python
"""
纵向 PID+ 横向 Pure Pursuit 控制示例
一步一步教你调试方法

使用文件：
    - simple_track_reference_clean.json: 参考路径数据
    - drive_simple_track.py (环境): 仿真环境
"""

import numpy as np
import json
import csv
from pathlib import Path


# ============================================================================
# 配置参数（先调这些，其他后面再说）
# ============================================================================
class Config:
    # 车辆基础参数
    VEHICLE_LENGTH = 3.5      # 车长 (米)
    MAX_STEER = 0.8           # 最大转向角 (弧度)≈45 度
    STEER_RATE = 2.0          # 转向速率限制 (弧度/秒)
    
    # PID 参数 - 从中间值开始调
    K_P_SPEED = 0.15          # 速度 P 增益 (太小响应慢，太大震荡)
    K_D_SPEED = 0.0           # 速度 D 增益 (一般不用)
    
    K_P_LATERAL = 0.8         # 横向 PID 的 P (影响转向灵敏度)
    K_D_LATERAL = 0.0         # 横向 PID 的 D(阻尼项，抑制震荡)
    
    LOOKDISTANCE = 5.0        # Pure Pursuit 前瞻距离 (米)
    LKP_DISTANCE = 10.0       # 前瞻距离随速度变化系数
    
    # 运行参数
    TARGET_SPEED = 70.0       # 目标速度 (km/h)
    
    # 调试选项
    ENABLE_DEBUG = True       # 是否记录日志到 CSV
    DEBUG_INTERVAL = 10       # 每隔几帧记录一次


# ============================================================================
# PID 控制器类（纵向和横向共用同一个逻辑）
# ============================================================================
class PIDController:
    """
    标准 PID 控制器
    
    作用：输入偏差 (error)，输出控制量 (output)
    
    公式：output = Kp * error + Ki * integral + Kd * derivative
    """
    def __init__(self, kp, ki=0.0, kd=0.0, max_output=1.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.max_output = max_output
        
        self.integral = 0.0
        self.last_error = None
        self.dt_last = None
    
    def reset(self):
        """重置积分项（用于启动或异常时）"""
        self.integral = 0.0
        self.last_error = None
        self.dt_last = None
    
    def update(self, error, dt=None):
        """
        计算 PID 输出
        
        参数:
            error: 当前偏差 (期望值 - 实际值)
            dt: 时间间隔 (秒), 如果为 None 则用上次的时间
            
        返回:
            control: 控制量 [-max_output, +max_output]
        """
        if dt is None:
            dt = self.dt_last if self.dt_last else 0.05  # 默认 50ms 周期
        
        # P 项：比例项 (当前误差)
        p = self.kp * error
        
        # I 项：积分项 (累积历史误差)
        self.integral += error * dt
        i = self.ki * self.integral
        
        # D 项：微分项 (误差变化率)
        d = 0.0
        if self.last_error is not None and self.dt_last is not None:
            error_change = error - self.last_error
            d = self.kd * (error_change / self.dt_last)
        
        # 合并三项
        output = p + i + d
        
        # 限制输出范围
        output = np.clip(output, -self.max_output, self.max_output)
        
        # 保存状态用于下一次计算
        self.last_error = error
        self.dt_last = dt
        
        return output


# ============================================================================
# Pure Pursuit 规划器
# ============================================================================
class PurePursuitPlanner:
    """
    Pure Pursuit 路径跟踪算法
    
    原理：在参考路径上找一个前瞻点，计算车辆朝向这个点的曲率半径
    转向角 = atan2(2 * L * sin(alpha) / R)
    
    其中：
        L: 前瞻距离
        alpha: 车辆与前瞻点的夹角
        R: 曲率半径
    """
    def __init__(self, reference_path):
        """
        初始化并加载参考路径
        
        参数:
            reference_path: 字典 {'x': [...], 'y': [...], 's_distances': [...]}
        """
        self.path_x = np.array(reference_path['x'])
        self.path_y = np.array(reference_path['y'])
        self.s_distances = np.array(reference_path['s_distances'])
        self.num_points = len(self.path_x)
    
    def find_nearest_s(self, x, y):
        """
        找到最近的参考路径点的位置 s
        
        参数:
            x, y: 车辆当前位置
            
        返回:
            s: 最近点的累计距离
            index: 最近点的索引
        """
        # 简单方法：计算所有点到车辆的距离
        dx = self.path_x - x
        dy = self.path_y - y
        distances = np.sqrt(dx**2 + dy**2)
        index = np.argmin(distances)
        s = float(self.s_distances[index])
        return s, int(index)
    
    def find_lookahead_point(self, vehicle_x, vehicle_y, vehicle_heading, speed):
        """
        找到前瞻点及其转向参数
        
        参数:
            vehicle_x, vehicle_y: 车辆位置
            vehicle_heading: 车辆航向 (弧度)
            speed: 当前速度 (m/s)
            
        返回:
            lookahead_point: (x, y, heading, curvature)
        """
        # 动态调整前瞻距离（速度越快，前瞻越远）
        look_distance = self.LOOKDISTANCE + self.LKP_DISTANCE * speed
        
        # 找到最近的参考点
        current_s, current_idx = self.find_nearest_s(vehicle_x, vehicle_y)
        
        # 沿路径搜索前瞻点
        for idx in range(current_idx, min(current_idx + 500, self.num_points)):
            tx = self.path_x[idx]
            ty = self.path_y[idx]
            ts = self.s_distances[idx]
            
            # 检查是否在前瞻距离之外
            dist_sq = (tx - vehicle_x)**2 + (ty - vehicle_y)**2
            if dist_sq > look_distance**2:
                # 找到了！
                target_index = idx
                break
        else:
            target_index = min(current_idx + 500, self.num_points - 1)
        
        # 获取前瞻点坐标
        lx = self.path_x[target_index]
        ly = self.path_y[target_index]
        
        # 计算前瞻点的航向
        if target_index > 0:
            prev_tx = self.path_x[target_index - 1]
            prev_ty = self.path_y[target_index - 1]
            target_heading = np.arctan2(ly - prev_ty, lx - prev_tx)
        else:
            target_heading = vehicle_heading
        
        # 计算相对于车辆的局部坐标系下的前瞻点
        dx = lx - vehicle_x
        dy = ly - vehicle_y
        
        # 旋转到车辆坐标系
        local_dx = dx * np.cos(-vehicle_heading) - dy * np.sin(-vehicle_heading)
        local_dy = dx * np.sin(-vehicle_heading) + dy * np.cos(-vehicle_heading)
        
        # 计算曲率半径
        if abs(local_dy) < 1e-6:
            curvature = 0.0
        else:
            # 圆弧拟合：R = L^2 / (4 * dy)
            local_dist = np.sqrt(local_dx**2 + local_dy**2)
            curvature = local_dist / (2.0 * local_dy)
        
        return {
            'x': float(lx),
            'y': float(ly),
            'heading': float(target_heading),
            'curvature': float(curvature),
            'distance_m': float(np.sqrt(local_dx**2 + local_dy**2))
        }


# ============================================================================
# 主控制器类
# ============================================================================
class AutonomousDriver:
    """
    自动驾驶控制器 - 整合纵向 PID + 横向 Pure Pursuit+PID
    
    使用方法：
        1. driver = AutonomousDriver(ref_data)
        2. driver.reset()
        3. while True:
               steering = driver.control_steering(vehicle_state)
               throttle = driver.control_throttle(vehicle_state)
    """
    def __init__(self, reference_data, config=Config):
        self.config = config
        
        # 加载参考路径
        self.planner = PurePursuitPlanner(reference_data)
        
        # PID 控制器
        self.speed_pid = PIDController(
            kp=config.K_P_SPEED, 
            kd=config.K_D_SPEED,
            max_output=1.0
        )
        self.lateral_pid = PIDController(
            kp=config.K_P_LATERAL,
            kd=config.K_D_LATERAL,
            max_output=1.0
        )
        
        # 车辆状态
        self.vehicle_x = 0.0
        self.vehicle_y = 0.0
        self.vehicle_heading = 0.0
        self.vehicle_speed = 0.0
        
        # 上一帧时间
        self.last_time = None
        
        # 日志记录
        self.debug_enabled = config.ENABLE_DEBUG
        self.log_file = None
        self.log_counter = 0
        
        if self.debug_enabled:
            self._init_log_file()
    
    def _init_log_file(self):
        """初始化 CSV 日志文件"""
        base_path = Path.cwd() / "pid_debug_log.csv"
        self.log_file = open(base_path, 'w', newline='')
        self.writer = csv.DictWriter(
            self.log_file,
            fieldnames=[
                'frame', 'time_s', 'speed_kmh', 'target_speed',
                'steering_cmd', 'actual_steer',
                'lateral_error', 'lookahead_dist', 'curvature',
                'throttle', 'brake', 'speed_error'
            ]
        )
        self.writer.writeheader()
        print(f"\n✓ 日志已开启：{base_path}")
    
    def close(self):
        """关闭资源"""
        if self.log_file:
            self.log_file.close()
    
    def reset(self, initial_pos, initial_heading=0.0):
        """重置控制器"""
        self.vehicle_x = initial_pos[0]
        self.vehicle_y = initial_pos[1]
        self.vehicle_heading = initial_heading
        self.vehicle_speed = 0.0
        self.last_time = None
        self.speed_pid.reset()
        self.lateral_pid.reset()
        self.log_counter = 0
    
    def step(self, action, frame, dt=None):
        """
        单步执行控制逻辑
        
        参数:
            action: [steering_wheel, throttle, brake] 外部输入的动作
            frame: 当前帧序号
            dt: 时间间隔
            
        返回:
            log_data: 当前帧日志数据（如果需要）
        """
        if dt is None:
            dt = 0.05  # 默认 50ms
        
        current_time = frame * dt
        
        # 如果是第一帧且 last_time 为 None
        if self.last_time is None:
            self.last_time = current_time
            return None
        
        # 计算时间间隔
        dt = current_time - self.last_time
        self.last_time = current_time
        
        # 1. 计算速度控制
        throttle_brake = self._compute_throttle_brake(frame, dt)
        
        # 2. 计算转向控制
        steering = self._compute_steering(action, frame, dt)
        
        # 3. 记录日志
        log_data = None
        if self.debug_enabled and self.log_counter == 0:
            log_data = self._generate_log(throttle_brake, steering, frame, dt)
            self.log_counter = self.config.DEBUG_INTERVAL
        else:
            self.log_counter -= 1
        
        return log_data
    
    def _compute_throttle_brake(self, frame, dt):
        """计算油门/刹车指令"""
        target_speed_ms = self.config.TARGET_SPEED / 3.6  # km/h -> m/s
        
        # 获取当前速度（假设 vehicle_speed 已被更新）
        speed_error = target_speed_ms - self.vehicle_speed
        
        # PID 控制
        throttle_command = self.speed_pid.update(speed_error, dt)
        
        # 分离为油门和刹车
        if throttle_command >= 0:
            throttle = throttle_command
            brake = 0.0
        else:
            throttle = 0.0
            brake = -throttle_command  # 负值表示刹车
        
        return {'throttle': throttle, 'brake': brake}
    
    def _compute_steering(self, action, frame, dt):
        """计算转向指令"""
        # 计算前瞻点
        lookahead = self.planner.find_lookahead_point(
            self.vehicle_x, 
            self.vehicle_y, 
            self.vehicle_heading,
            self.vehicle_speed
        )
        
        # 计算角度偏差 (track error)
        angle_error = lookahead['heading'] - self.vehicle_heading
        # 归一化到 [-pi, pi]
        while angle_error > np.pi:
            angle_error -= 2 * np.pi
        while angle_error < -np.pi:
            angle_error += 2 * np.pi
        
        # PID 控制
        steering_command = self.lateral_pid.update(angle_error, dt)
        
        # 平滑处理（转向限幅和限速率）
        actual_steer = action[0]  # 直接使用外部输入的转向
        if isinstance(actual_steer, (int, float)):
            actual_steer = float(actual_steer)
        
        # 转向限幅
        actual_steer = np.clip(actual_steer, -self.config.MAX_STEER, self.config.MAX_STEER)
        
        # 转向限速率
        steer_diff = actual_steer - action[0] if len(action) > 0 else 0
        max_step = self.config.STEER_RATE * dt
        steer_diff = np.clip(steer_diff, -max_step, max_step)
        
        # 生成动作序列（需要满足驱动环境的要求）
        steering_out = float(actual_steer)
        
        return steering_out
    
    def _generate_log(self, throttle_brake, steering, frame, dt):
        """生成日志数据"""
        # 计算前瞻点
        lookahead = self.planner.find_lookahead_point(
            self.vehicle_x, 
            self.vehicle_y, 
            self.vehicle_heading,
            self.vehicle_speed
        )
        
        # 计算角度偏差
        angle_error = lookahead['heading'] - self.vehicle_heading
        while angle_error > np.pi:
            angle_error -= 2 * np.pi
        while angle_error < -np.pi:
            angle_error += 2 * np.pi
        
        # 速度误差
        target_speed_ms = self.config.TARGET_SPEED / 3.6
        speed_error = target_speed_ms - self.vehicle_speed
        
        log_data = {
            'frame': frame,
            'time_s': frame * dt,
            'speed_kmh': self.vehicle_speed * 3.6,
            'target_speed': self.config.TARGET_SPEED,
            'steering_cmd': steering,
            'actual_steer': steering,
            'lateral_error': angle_error,
            'lookahead_dist': lookahead['distance_m'],
            'curvature': lookahead['curvature'],
            'throttle': throttle_brake['throttle'],
            'brake': throttle_brake['brake'],
            'speed_error': speed_error * 3.6  # 转换为 km/h
        }
        
        # 写入 CSV
        self.writer.writerow(log_data)
        
        return log_data
    
    def set_vehicle_state(self, x, y, heading, speed_kmh):
        """设置车辆状态（模拟反馈）"""
        self.vehicle_x = x
        self.vehicle_y = y
        self.vehicle_heading = heading
        self.vehicle_speed = speed_kmh / 3.6  # km/h -> m/s


# ============================================================================
# 测试函数
# ============================================================================
def test_controller():
    """测试控制器功能"""
    print("="*70)
    print("TESTING AUTONOMOUS DRIVER")
    print("="*70)
    
    # 加载参考路径
    ref_json = Path.cwd() / "simple_track_reference_clean.json"
    with open(ref_json, 'r') as f:
        ref_data = json.load(f)
    
    print(f"✓ 参考路径已加载：{len(ref_data['reference_path']['x'])} points")
    
    # 创建控制器
    driver = AutonomousDriver(ref_data, Config)
    
    try:
        # 初始化 - 从参考路径起点开始
        start_x = ref_data['reference_path']['x'][0]
        start_y = ref_data['reference_path']['y'][0]
        driver.reset((start_x, start_y), initial_heading=0.0)
        
        print(f"\n仿真测试开始...")
        print(f"{'Frame':>6} | {'Speed(km/h)':>10} | {'Steer(rad)':>10} | "
              f"{'Lateral(m)':>12} | {'Throttle':>8}")
        print("-"*70)
        
        # 模拟运行 100 帧
        for frame in range(100):
            # 模拟车辆状态（这里简化为直线行驶）
            # 实际使用时需要从 env.agent 获取真实状态
            mock_speed = 15.0 + 5.0 * np.sin(frame * 0.1)  # 模拟速度波动
            mock_heading = 0.0  # 模拟直线
            
            driver.set_vehicle_state(start_x, start_y, mock_heading, mock_speed)
            
            # 执行控制
            action = [0.0, 0.5, 0.0]  # 初始转向、油门、刹车
            log = driver.step(action, frame)
            
            # 打印进度
            if frame % 20 == 0 or frame == 99:
                print(f"{frame:>6} | {mock_speed:>10.1f} | {action[0]:>10.4f} | "
                      f"{log.get('lateral_error', 0)*30.:>12.2f} | {action[1]:>8.2f}")
        
        print("\n✓ 测试完成！CSV 日志已保存到：pid_debug_log.csv")
        
    finally:
        driver.close()


if __name__ == "__main__":
    test_controller()
