#!/usr/bin/env python
"""最简单的自动驾驶控制"""

import numpy as np
import json
import csv


class C:
    TARGET_SPEED = 30.0
    KP_SPEED = 0.1
    INTEGRAL_MAX = 5.0
    LOOK_DIST_BASE = 3.0
    LOOK_SCALE = 0.15
    KP_TURN = 0.6
    MAX_STEER = 0.6
    WHEELBASE = 2.5
    SAVE_CSV = True
    LOG_INTERVAL = 10


def pid_update(error, state, dt):
    p = C.KP_TURN * error
    integral = max(-C.INTEGRAL_MAX, min(C.INTEGRAL_MAX, state['integral'] + error * dt))
    derivative = 0.0 if state['last_error'] is None else (error - state['last_error']) / dt
    output = p + 0.0 * integral + 0.0 * derivative
    output = max(-C.MAX_STEER, min(C.MAX_STEER, output))
    state['integral'], state['last_error'] = integral, error
    return output, state


def find_lookahead_point(x, y, heading, speed_ms, path_x, path_y):
    lookahead_dist = C.LOOK_DIST_BASE + C.LOOK_SCALE * speed_ms
    
    dx_arr = path_x - x
    dy_arr = path_y - y
    idx = int(np.argmin(dx_arr**2 + dy_arr**2))
    
    for i in range(idx, min(idx + 1000, len(path_x))):
        tx = float(path_x[i])
        ty = float(path_y[i])
        d2 = (tx - x)**2 + **(ty - y)2
        if d2 > lookahead_dist**2:
            break
    else:
        i = min(idx + 1000, len(path_x) - 1)
    
    lx, ly = path_x[i], path_y[i]
    tgt_hdg = np.arctan2(float(ly - path_y[i-1]), float(lx - path_x[i-1])) if i > 0 else heading
    
    dx_l, dy_l = lx - x, ly - y
    local_dx = dx_l * np.cos(-heading) - dy_l * np.sin(-heading)
    local_dy = dx_l * np.sin(-heading) + dy_l * np.cos(-heading)
    curvature = 0.0 if abs(local_dy) < 0.01 else np.sqrt(local_dx**2 + local_dy**2) / (2.0 * local_dy)
    
    return lx, ly, tgt_hdg, curvature, i


def main():
    print("="*60)
    print("SIMPLE AUTONOMOUS DRIVING")
    print("="*60)
    
    with open('simple_track_reference_clean.json') as f:
        ref_data = json.load(f)
    
    path_x = np.array(ref_data['reference_path']['x'])
    path_y = np.array(ref_data['reference_path']['y'])
    
    print(f"✓ 路径点数：{len(path_x)}")
    
    steer_state = {'integral': 0.0, 'last_error': None}
    x, y, heading = float(path_x[0]), float(path_y[0]), 0.0
    
    log_file = open('pid_driving_log.csv', 'w', newline='') if C.SAVE_CSV else None
    writer = csv.writer(log_file) if log_file else None
    if writer:
        writer.writerow(['frame', 'time_s', 'speed_kmh', 'throttle', 'steering', 'angle_err_deg'])
    
    print(f"\n{'Frame':>6} | {'Speed':>8} | {'Throttle':>9} | {'Steer':>10} | {'AngleErr':>10}")
    print("-"*70)
    
    target_ms = C.TARGET_SPEED / 3.6
    last_time = 0.0
    
    for frame in range(500):
        dt = 0.05 if frame > 0 else 0.0
        current_time = frame * dt
        
        if frame == 0:
            last_time = current_time
            continue
        
        dt = current_time - last_time
        last_time = current_time
        
        true_speed = max(0.0, C.TARGET_SPEED * 0.9 + 5.0 * np.sin(frame * 0.1))
        speed_ms = true_speed / 3.6
        
        throttle = max(0.0, min(1.0, C.KP_SPEED * (target_ms - speed_ms)))
        
        lx, ly, tgt_hdg, curv, point_idx = find_lookahead_point(x, y, heading, speed_ms, path_x, path_y)
        
        angle_err = tgt_hdg - heading
        while angle_err > np.pi: angle_err -= 2*np.pi
        while angle_err < -np.pi: angle_err += 2*np.pi
        
        steering, steer_state = pid_update(angle_err, steer_state, dt)
        
        if frame % C.LOG_INTERVAL == 0:
            print(f"{frame:>6} | {true_speed:7.1f} | {throttle:8.2f} | {steering:9.4f} | {np.degrees(angle_err):9.1f}")
            if writer:
                writer.writerow([frame, round(current_time, 3), round(true_speed, 2), round(throttle, 3), round(steering, 4), np.degrees(angle_err)])
        
        new_hdg = max(-np.pi, min(np.pi, heading + steering * speed_ms / C.WHEELBASE * dt))
        heading = new_hdg
        x += speed_ms * np.cos(heading) * dt
        y += speed_ms * np.sin(heading) * dt
    
    if log_file:
        log_file.close()
    
    print(f"\n✓ CSV: pid_driving_log.csv")
    print("="*60)


if __name__ == "__main__":
    main()
