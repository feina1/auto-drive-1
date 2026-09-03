#!/usr/bin/env python3
"""简易版 PID 控制测试 - 无可视化"""

import numpy as np
import json

# 参数配置
class C:
    TARGET_SPEED = 30.0
    KP_TURN = 0.8
    LOOK_DIST_BASE = 2.0
    LOOK_SCALE = 0.12
    WHEELBASE = 2.5

# Pure Pursuit
def pure_pursuit(x, y, heading, speed_ms, path_x, path_y):
    lookahead_dist = C.LOOK_DIST_BASE + C.LOOK_SCALE * speed_ms
    
    # Find nearest point
    idx = int(np.argmin((path_x - x)**2 + **(path_y - y)2))
    
    # Search for lookahead point
    for i in range(idx, min(idx + 1000, len(path_x))):
        tx, ty = float(path_x[i]), float(path_y[i])
        dist_sq = (tx - x)**2 + **(ty - y)2
        if dist_sq > lookahead_dist**2:
            break
    
    lx, ly = path_x[i], path_y[i]
    target_heading = np.arctan2(float(ly - path_y[max(0,i-1)]), 
                                 float(lx - path_x[max(0,i-1)]))
    
    return lx, ly, target_heading, i

# Load data
with open('simple_track_reference_clean.json', 'r') as f:
    data = json.load(f)

path_x = np.array(data['reference_path']['x'])
path_y = np.array(data['reference_path']['y'])

print(f"Loaded {len(path_x)} points")

# Simulate 100 steps
x, y = float(path_x[0]), float(path_y[0])
heading = 0.0

for step in range(100):
    mock_speed = 8.0 + 2.0 * np.sin(step * 0.1)
    
    lx, ly, tgt_hdg, _ = pure_pursuit(x, y, heading, mock_speed, path_x, path_y)
    
    angle_err = tgt_hdg - heading
    while angle_err > np.pi: angle_err -= 2*np.pi
    while angle_err < -np.pi: angle_err += 2*np.pi
    
    steering = C.KP_TURN * angle_err
    new_heading = heading + steering * mock_speed / C.WHEELBASE * 0.05
    heading = max(-np.pi, min(np.pi, new_heading))
    
    x += mock_speed * np.cos(new_heading) * 0.05
    y += mock_speed * np.sin(new_heading) * 0.05
    
    print(f"Step {step}: ({x:.1f}, {y:.1f}), Error={angle_err*180/np.pi:.1f}°")

print("Test passed!")
