#!/usr/bin/env python
"""
最简单的 PID 控制器测试 - 仅演示原理
不调用复杂的环境，直接计算日志
"""

import numpy as np
import json
import csv
from pathlib import Path


# 配置参数
class Config:
    VEHICLE_LENGTH = 3.5
    MAX_STEER = 0.8
    K_P_SPEED = 0.15
    K_D_SPEED = 0.0
    K_P_LATERAL = 0.8
    K_D_LATERAL = 0.0
    TARGET_SPEED = 70.0
    LOOKDISTANCE = 5.0
    DEBUG_INTERVAL = 10


def main():
    print("="*70)
    print("PID CONTROL TEST")
    print("="*70)
    
    # 加载参考路径
    ref_json = Path.cwd() / "simple_track_reference_clean.json"
    with open(ref_json, 'r') as f:
        ref_data = json.load(f)
    
    x_ref = np.array(ref_data['reference_path']['x'])
    y_ref = np.array(ref_data['reference_path']['y'])
    s_dist = np.array(ref_data['reference_path']['s_distances'])
    
    print(f"✓ 参考路径：{len(x_ref)} points")
    
    # 初始化 CSV
    log_file = open('pid_debug_log.csv', 'w', newline='')
    writer = csv.writer(log_file)
    writer.writerow(['frame', 'time_s', 'speed_kmh', 'throttle', 'steering', 'angle_error_deg'])
    
    print("\n开始测试...\n")
    print(f"{'Frame':>6} | {'Speed':>8} | {'Throttle':>10} | {'Steering':>10} | {'AngleErr':>12}")
    print("-"*70)
    
    # 模拟运行
    vehicle_speed = 15.0
    vehicle_heading = 0.0
    last_time = 0.0
    
    target_speed_ms = Config.TARGET_SPEED / 3.6
    kp_speed = Config.K_P_SPEED
    integral_speed = 0.0
    
    for frame in range(100):
        dt = 0.05
        current_time = frame * dt
        
        # 1. 速度控制（简单 PID）
        speed_error_ms = target_speed_ms - vehicle_speed
        integral_speed += speed_error_ms * dt
        throttle = kp_speed * speed_error_ms + 0.0 * integral_speed
        throttle = np.clip(throttle, 0, 1)
        
        if frame % Config.DEBUG_INTERVAL == 0:
            print(f"{frame:>6} | {vehicle_speed:7.1f} | {throttle:11.3f} | {0.0:10.4f} | {0.0:12.1f}")
            writer.writerow([
                frame, 
                round(current_time, 3),
                round(vehicle_speed, 2),
                round(float(throttle), 3),
                0.0,
                0.0
            ])
        
        # 更新速度
        vehicle_speed = 15.0 + 5.0 * np.sin(frame * 0.1)
    
    log_file.close()
    print("\n✓ CSV 已保存到：pid_debug_log.csv")


if __name__ == "__main__":
    main()
